"""
Interventional SHAP — Feature attribution under do-calculus.

Replaces SHAP's observational value function E[f(X)|X_S = x_S] with
the interventional E[f(X)|do(X_S = x_S)] via a linear SCM. This
separates genuine causal drivers from correlated proxies.

Algorithm:
    1. Estimate a linear SCM from data (LiNGAM or PC + OLS)
    2. For each Shapley permutation and coalition S:
       a. Draw baseline rows from training data (row resampling)
       b. Overwrite X_S with instance values (intervention)
       c. Propagate to descendants of S via SCM structural equations
          (preserving original residuals from the baseline row)
       d. Non-descendants retain their baseline values (joint preserved)
       e. Evaluate f(x_modified) and average
    3. Compare with standard marginal Shapley values
    4. Classify features by causal role, flag confounders and colliders

Based on Heskes et al. (2020), Janzing et al. (2020).

Dependencies: numpy, scipy. Optional: causal-learn (DAG estimation), shap.
"""

import math
import numpy as np
from collections import deque

__all__ = ["LinearSCM", "run_interventional_shap"]


# ===========================================================================
# Linear Structural Causal Model
# ===========================================================================
class LinearSCM:
    """
    Linear SCM: X_j = Σ_{i ∈ pa(j)} B[j,i] · X_i + ε_j

    Convention: B[j,i] ≠ 0 means edge i → j (i is parent of j).
    """

    def __init__(self, B, feature_names):
        self.B = np.array(B, dtype=float)
        self.names = list(feature_names)
        self.p = len(feature_names)
        assert self.B.shape == (self.p, self.p)

        # Topological sort (raises if cyclic)
        self.topo_order = self._topological_sort()

        # Parent/child lookup
        self._parents = {}
        self._children = {}
        for j in range(self.p):
            self._parents[j] = [i for i in range(self.p)
                                if abs(self.B[j, i]) > 1e-12]
            self._children[j] = [k for k in range(self.p)
                                 if abs(self.B[k, j]) > 1e-12]

        # Precompute descendants and ancestors via BFS
        self._descendants = {j: self._bfs(j, self._children)
                             for j in range(self.p)}
        self._ancestors = {j: self._bfs(j, self._parents)
                           for j in range(self.p)}

        self.is_identified = True
        self._n_undirected_original = 0

    # --- Estimation ---

    @classmethod
    def from_data(cls, df, names, method="lingam", alpha=0.05, prune=0.05):
        """Estimate SCM from data via LiNGAM or PC + OLS."""
        data = df[names].dropna().values.astype(float)
        if method == "lingam":
            return cls._est_lingam(data, names, prune)
        elif method == "pc":
            return cls._est_pc(data, names, alpha, prune)
        raise ValueError(f"Unknown method: {method}")

    @classmethod
    def _est_lingam(cls, data, names, prune):
        from causallearn.search.FCMBased import lingam as cl_lingam
        model = cl_lingam.ICALiNGAM()
        model.fit(data)
        B = model.adjacency_matrix_.copy()
        B[np.abs(B) < prune] = 0.0
        return cls(B, names)

    @classmethod
    def _est_pc(cls, data, names, alpha, prune):
        from causallearn.search.ConstraintBased.PC import pc as run_pc
        from causallearn.utils.cit import fisherz
        p = len(names)
        cg = run_pc(data, alpha, fisherz, node_names=names)
        G = cg.G.graph  # p×p: -1=tail, 1=arrow, 0=none

        undirected = []
        for i in range(p):
            for j in range(i + 1, p):
                if G[i, j] == -1 and G[j, i] == -1:
                    undirected.append((i, j))
                    # Orient by variance heuristic (rough)
                    if np.var(data[:, i]) < np.var(data[:, j]):
                        G[i, j], G[j, i] = -1, 1
                    else:
                        G[j, i], G[i, j] = -1, 1

        # OLS coefficients for each node on its DAG parents
        B = np.zeros((p, p))
        for j in range(p):
            pa = [i for i in range(p)
                  if G[i, j] == -1 and G[j, i] == 1]
            if pa:
                X_pa = data[:, pa]
                beta = np.linalg.lstsq(X_pa, data[:, j], rcond=None)[0]
                for k, pi in enumerate(pa):
                    B[j, pi] = beta[k]

        B[np.abs(B) < prune] = 0.0
        scm = cls(B, names)
        scm.is_identified = len(undirected) == 0
        scm._n_undirected_original = len(undirected)
        return scm

    # --- Graph structure ---

    def _topological_sort(self):
        """Kahn's algorithm. Raises ValueError if cyclic."""
        in_deg = np.zeros(self.p, dtype=int)
        children = {j: [] for j in range(self.p)}
        for j in range(self.p):
            for i in range(self.p):
                if abs(self.B[j, i]) > 1e-12:
                    children[i].append(j)
                    in_deg[j] += 1
        q = deque(j for j in range(self.p) if in_deg[j] == 0)
        order = []
        while q:
            node = q.popleft()
            order.append(node)
            for c in children[node]:
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    q.append(c)
        if len(order) != self.p:
            raise ValueError("SCM has a cycle — cannot proceed.")
        return order

    def _bfs(self, start, adj):
        """BFS from start using adjacency dict (excludes start)."""
        visited = set()
        q = deque(adj.get(start, []))
        while q:
            n = q.popleft()
            if n not in visited:
                visited.add(n)
                q.extend(adj.get(n, []))
        return visited

    def descendants_of(self, S):
        """All descendants of node set S (excluding S itself)."""
        out = set()
        for j in S:
            out |= self._descendants[j]
        return out - S

    def ancestors_of(self, node):
        return set(self._ancestors[node])

    def node_roles(self, target_idx):
        """
        Classify each node's causal role relative to target.

        Returns dict {idx: role} where role is one of:
            direct_parent, ancestor, descendant, non_cause, target
        """
        tgt_parents = set(self._parents[target_idx])
        tgt_ancestors = self._ancestors[target_idx]
        tgt_descendants = self._descendants[target_idx]
        roles = {}
        for j in range(self.p):
            if j == target_idx:
                roles[j] = "target"
            elif j in tgt_parents:
                roles[j] = "direct_parent"
            elif j in tgt_ancestors:
                roles[j] = "ancestor"
            elif j in tgt_descendants:
                roles[j] = "descendant"
            else:
                roles[j] = "non_cause"
        return roles

    def causal_paths(self, source, target_idx, max_depth=6):
        """Find all directed paths from source to target (BFS, capped)."""
        paths = []
        q = deque([(source, [source])])
        while q:
            node, path = q.popleft()
            if len(path) > max_depth:
                continue
            for child in self._children.get(node, []):
                if child == target_idx:
                    paths.append(path + [child])
                elif child not in path:
                    q.append((child, path + [child]))
        return paths

    def find_colliders(self):
        """Nodes with 2+ parents where at least one pair is non-adjacent."""
        colliders = []
        for j in range(self.p):
            pa = self._parents[j]
            if len(pa) < 2:
                continue
            found = False
            for a in range(len(pa)):
                if found:
                    break
                for b in range(a + 1, len(pa)):
                    if (abs(self.B[pa[a], pa[b]]) < 1e-12
                            and abs(self.B[pa[b], pa[a]]) < 1e-12):
                        colliders.append({
                            "node": j, "name": self.names[j],
                            "parents": (self.names[pa[a]],
                                        self.names[pa[b]]),
                        })
                        found = True
                        break
        return colliders

    def total_causal_effects(self):
        """
        TCE[i,j] = total effect of do(X_j += 1) on X_i.
        For linear SCM: (I - B)^{-1} - I.
        """
        try:
            return np.linalg.inv(np.eye(self.p) - self.B) - np.eye(self.p)
        except np.linalg.LinAlgError:
            return np.zeros((self.p, self.p))

    # --- Intervention (vectorized) ---

    def intervene_batch(self, X_base, x_instance, S, propagate_set=None):
        """
        Apply do(X_S = x_S) to all rows of X_base.

        1. Copy X_base
        2. Overwrite columns in S with x_instance values
        3. For each descendant of S (in topological order):
           - Compute residual from original X_base row
           - Recompute using updated parent values + original residual

        This preserves the joint distribution of non-descendants
        and the idiosyncratic noise (residuals) of descendants.

        Parameters
        ----------
        X_base : (n, p) baseline samples
        x_instance : (p,) values to set for intervened nodes
        S : set of int — node indices to intervene on
        propagate_set : set of int — restrict propagation to these indices
        """
        X_mod = X_base.copy()
        for j in S:
            X_mod[:, j] = x_instance[j]

        desc = self.descendants_of(S)
        if propagate_set is not None:
            desc = desc & propagate_set

        for node in self.topo_order:
            if node not in desc:
                continue
            pa = self._parents[node]
            if not pa:
                continue
            B_pa = self.B[node, pa]
            residuals = X_base[:, node] - X_base[:, pa] @ B_pa
            X_mod[:, node] = X_mod[:, pa] @ B_pa + residuals

        return X_mod


# ===========================================================================
# Shapley value computation
# ===========================================================================

def _v_do(predict_fn, scm, bg, x_full, S_scm, feat_idx, feat_set):
    """E[f(X) | do(X_S = x_S)] via row resampling + SCM propagation."""
    if not S_scm:
        return np.mean(predict_fn(bg[:, feat_idx]))
    X_mod = scm.intervene_batch(bg, x_full, S_scm, feat_set)
    return np.mean(predict_fn(X_mod[:, feat_idx]))


def _v_marginal(predict_fn, bg, x_full, S_feat, feat_idx):
    """E[f(x_S, X_{-S})] with X_{-S} drawn from background rows."""
    if not S_feat:
        return np.mean(predict_fn(bg[:, feat_idx]))
    X = bg[:, feat_idx].copy()
    for k in S_feat:
        X[:, k] = x_full[feat_idx[k]]
    return np.mean(predict_fn(X))


def _compute_ishap(predict_fn, scm, bg_data, x_full, feat_idx,
                   n_bg=30, batch_size=50, max_perm=200,
                   min_perm=50, tol=0.05):
    """
    Compute interventional + marginal Shapley values for one instance.

    Uses permutation sampling with adaptive convergence:
    runs batches of permutations, stops when max(SE)/max(|φ|) < tol.

    Returns dict with phi_int, phi_mar, se_int, se_mar, n_perm, converged.
    """
    p = len(feat_idx)
    feat_set = set(feat_idx)

    bg_idx = np.random.choice(len(bg_data), min(n_bg, len(bg_data)),
                              replace=n_bg > len(bg_data))
    bg = bg_data[bg_idx]

    phi_int = np.zeros(p)
    phi_mar = np.zeros(p)
    ssq_int = np.zeros(p)
    ssq_mar = np.zeros(p)
    n_perm = 0
    converged = False

    while n_perm < max_perm:
        for _ in range(batch_size):
            perm = np.random.permutation(p)
            S_scm = set()
            S_feat = set()

            prev_int = _v_do(predict_fn, scm, bg, x_full, set(),
                             feat_idx, feat_set)
            prev_mar = _v_marginal(predict_fn, bg, x_full, set(),
                                   feat_idx)

            for k in range(p):
                j = perm[k]
                S_scm.add(feat_idx[j])
                S_feat.add(j)

                curr_int = _v_do(predict_fn, scm, bg, x_full,
                                 S_scm, feat_idx, feat_set)
                curr_mar = _v_marginal(predict_fn, bg, x_full,
                                       S_feat, feat_idx)

                d_int = curr_int - prev_int
                d_mar = curr_mar - prev_mar
                phi_int[j] += d_int
                phi_mar[j] += d_mar
                ssq_int[j] += d_int ** 2
                ssq_mar[j] += d_mar ** 2

                prev_int = curr_int
                prev_mar = curr_mar

            n_perm += 1

        if n_perm >= min_perm:
            mean_int = phi_int / n_perm
            var_int = ssq_int / n_perm - mean_int ** 2
            se = np.sqrt(np.maximum(var_int, 0) / n_perm)
            max_attr = max(np.max(np.abs(mean_int)), 1e-10)
            if np.max(se) / max_attr < tol:
                converged = True
                break

    n = n_perm
    mean_int = phi_int / n
    mean_mar = phi_mar / n
    se_int = np.sqrt(np.maximum(ssq_int / n - mean_int ** 2, 0) / n)
    se_mar = np.sqrt(np.maximum(ssq_mar / n - mean_mar ** 2, 0) / n)

    return {
        "phi_int": mean_int,
        "phi_mar": mean_mar,
        "se_int": se_int,
        "se_mar": se_mar,
        "n_perm": n,
        "converged": converged,
    }


# ===========================================================================
# DSW Integration
# ===========================================================================

def run_interventional_shap(df, analysis_id, config,
                            model=None, model_features=None):
    """Dispatch for interventional SHAP in DSW."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    if model is None:
        result["summary"] = ("Error: No ML model provided. "
                             "Train a model first, then run this analysis.")
        return result

    features = config.get("features") or model_features or []
    target = config.get("target", "")
    scm_method = config.get("scm_method", "lingam")
    alpha_pc = float(config.get("alpha", 0.05))
    n_bg = int(config.get("n_bg", 30))
    n_explain = int(config.get("n_explain", 20))
    max_perm = int(config.get("max_perm", 200))

    if not features:
        result["summary"] = "Error: No feature columns specified."
        return result
    if not target:
        result["summary"] = "Error: No target column specified."
        return result

    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        result["summary"] = f"Error: Columns not found: {missing}"
        return result

    # Prepare numeric data (encode categoricals)
    all_cols = list(features) + [target]
    data = df[all_cols].copy()
    cat_cols = data.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        data[col] = data[col].astype("category").cat.codes.astype(float)
    data = data.dropna()

    if len(data) < 30:
        result["summary"] = (f"Error: Need ≥30 complete rows, "
                             f"got {len(data)}.")
        return result

    p = len(features)
    feat_idx = list(range(p))
    target_idx = p

    # --- Step 1: Estimate SCM ---
    try:
        scm = LinearSCM.from_data(data, all_cols,
                                  method=scm_method, alpha=alpha_pc)
    except Exception as e:
        result["summary"] = f"Error estimating SCM ({scm_method}): {e}"
        return result

    # --- Step 2: Classify features ---
    roles = scm.node_roles(target_idx)
    colliders = scm.find_colliders()
    tce = scm.total_causal_effects()
    feat_set = set(feat_idx)

    # --- Step 3: Compute Shapley values ---
    data_arr = data.values.astype(float)
    predict_fn = lambda X: model.predict(X)

    n_explain = min(n_explain, len(data_arr))
    explain_idx = np.random.choice(len(data_arr), n_explain, replace=False)

    phi_int_all = np.zeros((n_explain, p))
    phi_mar_all = np.zeros((n_explain, p))
    se_int_all = np.zeros((n_explain, p))
    total_perm = 0
    n_converged = 0

    for i, idx in enumerate(explain_idx):
        x_full = data_arr[idx]
        res = _compute_ishap(
            predict_fn, scm, data_arr, x_full, feat_idx,
            n_bg=n_bg, batch_size=50, max_perm=max_perm,
            min_perm=50, tol=0.05,
        )
        phi_int_all[i] = res["phi_int"]
        phi_mar_all[i] = res["phi_mar"]
        se_int_all[i] = res["se_int"]
        total_perm += res["n_perm"]
        if res["converged"]:
            n_converged += 1

    # Aggregate: mean |φ| across explained instances
    mean_abs_int = np.mean(np.abs(phi_int_all), axis=0)
    mean_abs_mar = np.mean(np.abs(phi_mar_all), axis=0)
    mean_se = np.mean(se_int_all, axis=0)

    # Rank comparison
    rank_int = np.argsort(-mean_abs_int)
    rank_mar = np.argsort(-mean_abs_mar)

    # --- Step 4: Build summary ---
    role_labels = {
        "direct_parent": "Direct Cause",
        "ancestor": "Indirect Cause",
        "descendant": "Descendant",
        "non_cause": "Non-Cause (correlated)",
        "target": "Target",
    }
    role_colors = {
        "direct_parent": "good",
        "ancestor": "highlight",
        "descendant": "warning",
        "non_cause": "text",
    }

    lines = []
    lines.append(f"<<COLOR:accent>>{'═' * 70}<</COLOR>>")
    lines.append("<<COLOR:title>>INTERVENTIONAL SHAP (SCM-BASED)<</COLOR>>")
    lines.append(f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n")

    lines.append(f"<<COLOR:highlight>>SCM method:<</COLOR>> "
                 f"{'ICA-LiNGAM' if scm_method == 'lingam' else 'PC + OLS'}")
    lines.append(f"<<COLOR:highlight>>Variables:<</COLOR>> "
                 f"{p} features + target ({target})")
    lines.append(f"<<COLOR:highlight>>Instances explained:<</COLOR>> "
                 f"{n_explain}")
    lines.append(f"<<COLOR:highlight>>Convergence:<</COLOR>> "
                 f"{n_converged}/{n_explain} converged "
                 f"(avg {total_perm/n_explain:.0f} perms)")
    if not scm.is_identified:
        lines.append(f"<<COLOR:warning>>WARNING: {scm._n_undirected_original}"
                     f" edges were undirected in PC output — "
                     f"oriented by variance heuristic<</COLOR>>")

    # Feature-by-feature comparison
    lines.append(f"\n<<COLOR:accent>>── Feature Attribution "
                 f"(mean |SHAP|) ──<</COLOR>>")
    lines.append(f"{'Feature':<20} {'Standard':>10} {'Interv.':>10} "
                 f"{'Δ':>8} {'Role':<20}")
    lines.append(f"{'─' * 70}")

    for idx_f in rank_int:
        fname = features[idx_f]
        std_val = mean_abs_mar[idx_f]
        int_val = mean_abs_int[idx_f]
        delta = int_val - std_val
        role = roles.get(idx_f, "non_cause")
        rlabel = role_labels.get(role, role)
        lines.append(f"{fname:<20} {std_val:>10.4f} {int_val:>10.4f} "
                     f"{delta:>+8.4f} {rlabel:<20}")

    # Alerts
    alerts = []
    for idx_f in range(p):
        role = roles.get(idx_f, "non_cause")
        fname = features[idx_f]
        mar_rank = int(np.where(rank_mar == idx_f)[0][0])
        # Confounding alert: high standard SHAP but non-cause
        if role == "non_cause" and mar_rank < p // 2:
            alerts.append(
                f"<<COLOR:warning>>CONFOUNDING: {fname} ranks #{mar_rank+1} "
                f"in standard SHAP but has no causal path to {target}. "
                f"Importance may be spurious.<</COLOR>>")
        # Suppression: low standard SHAP but direct cause
        if role == "direct_parent" and mar_rank >= p // 2:
            alerts.append(
                f"<<COLOR:highlight>>SUPPRESSION: {fname} is a direct cause "
                f"of {target} but ranks only #{mar_rank+1} in standard SHAP. "
                f"Its effect may be masked by confounders.<</COLOR>>")

    # Collider alerts
    for coll in colliders:
        if roles.get(coll["node"], "") == "descendant":
            alerts.append(
                f"<<COLOR:warning>>COLLIDER: {coll['name']} is a descendant "
                f"with parents {coll['parents'][0]} and "
                f"{coll['parents'][1]}. Conditioning on it "
                f"can induce spurious associations.<</COLOR>>")

    if alerts:
        lines.append(f"\n<<COLOR:accent>>── Diagnostics ──<</COLOR>>")
        lines.extend(alerts)

    # Causal paths for top features
    lines.append(f"\n<<COLOR:accent>>── Causal Paths to {target} ──<</COLOR>>")
    for idx_f in rank_int[:5]:
        role = roles.get(idx_f, "non_cause")
        if role in ("direct_parent", "ancestor"):
            paths = scm.causal_paths(idx_f, target_idx)
            for path in paths[:3]:
                path_str = " → ".join(all_cols[n] for n in path)
                lines.append(f"  {path_str}")
                # Total causal effect
                tce_val = tce[target_idx, idx_f]
                lines.append(f"    Total causal effect: {tce_val:+.4f}")

    # Assumptions
    lines.append(f"\n<<COLOR:accent>>── Assumptions ──<</COLOR>>")
    lines.append("  Method: Interventional SHAP via learned SCM")
    lines.append("  1. SCM is linear, acyclic, no hidden confounders")
    lines.append("  2. SCM learned from observational data — "
                 "interventions may differ from plant reality")
    lines.append("  3. ML model is fully nonlinear; SCM only "
                 "governs how features co-move under intervention")
    lines.append(f"  4. Attribution stability: "
                 f"{'high' if n_converged >= n_explain * 0.8 else 'medium' if n_converged >= n_explain * 0.5 else 'low'}"
                 f" ({n_converged}/{n_explain} converged)")

    result["summary"] = "\n".join(lines)

    # --- Step 5: Plots ---
    # 1. Comparison bar chart (standard vs interventional)
    sorted_idx = rank_int[::-1]  # ascending for horizontal bars
    result["plots"].append({
        "title": "Standard vs Interventional SHAP (mean |φ|)",
        "data": [
            {
                "type": "bar", "orientation": "h",
                "y": [features[i] for i in sorted_idx],
                "x": [float(mean_abs_mar[i]) for i in sorted_idx],
                "name": "Standard (marginal)",
                "marker": {"color": "rgba(150,150,150,0.5)",
                           "line": {"color": "#999", "width": 1}},
            },
            {
                "type": "bar", "orientation": "h",
                "y": [features[i] for i in sorted_idx],
                "x": [float(mean_abs_int[i]) for i in sorted_idx],
                "name": "Interventional (SCM)",
                "marker": {"color": "rgba(74,159,110,0.6)",
                           "line": {"color": "#4a9f6e", "width": 1}},
            },
        ],
        "layout": {
            "template": "plotly_dark",
            "height": max(300, p * 30),
            "barmode": "group",
            "xaxis": {"title": "mean |SHAP value|"},
            "legend": {"x": 0.6, "y": 0.05},
        },
    })

    # 2. Role-colored interventional importance
    role_color_map = {
        "direct_parent": "#4a9f6e",
        "ancestor": "#6ab7d4",
        "descendant": "#d4a24a",
        "non_cause": "#888888",
    }
    bar_colors = [role_color_map.get(roles.get(i, "non_cause"), "#888")
                  for i in sorted_idx]

    result["plots"].append({
        "title": "Interventional SHAP by Causal Role",
        "data": [{
            "type": "bar", "orientation": "h",
            "y": [features[i] for i in sorted_idx],
            "x": [float(mean_abs_int[i]) for i in sorted_idx],
            "marker": {"color": bar_colors,
                       "line": {"color": "rgba(255,255,255,0.3)",
                                "width": 1}},
            "text": [role_labels.get(roles.get(i, ""), "")
                     for i in sorted_idx],
            "textposition": "outside",
        }],
        "layout": {
            "template": "plotly_dark",
            "height": max(300, p * 30),
            "xaxis": {"title": "mean |SHAP value|"},
            "annotations": [
                {"x": 0.95, "y": 1.05, "xref": "paper", "yref": "paper",
                 "text": ("<span style='color:#4a9f6e'>■</span> Direct Cause  "
                          "<span style='color:#6ab7d4'>■</span> Indirect  "
                          "<span style='color:#d4a24a'>■</span> Descendant  "
                          "<span style='color:#888'>■</span> Non-cause"),
                 "showarrow": False, "font": {"size": 10}},
            ],
        },
    })

    # 3. Discrepancy plot (standard - interventional)
    discrepancy = mean_abs_mar - mean_abs_int
    disc_sorted = np.argsort(np.abs(discrepancy))[::-1]
    top_disc = disc_sorted[:min(10, p)]
    disc_colors = ["#d94a4a" if discrepancy[i] > 0 else "#4a9f6e"
                   for i in top_disc[::-1]]

    result["plots"].append({
        "title": "SHAP Discrepancy (Standard − Interventional)",
        "data": [{
            "type": "bar", "orientation": "h",
            "y": [features[i] for i in top_disc[::-1]],
            "x": [float(discrepancy[i]) for i in top_disc[::-1]],
            "marker": {"color": disc_colors},
        }],
        "layout": {
            "template": "plotly_dark",
            "height": max(250, len(top_disc) * 28),
            "xaxis": {"title": "Δ (positive = inflated by correlation)"},
            "annotations": [
                {"x": 0.95, "y": 1.05, "xref": "paper", "yref": "paper",
                 "text": ("<span style='color:#d94a4a'>→</span> "
                          "Inflated by correlation  "
                          "<span style='color:#4a9f6e'>←</span> "
                          "Suppressed by confounding"),
                 "showarrow": False, "font": {"size": 10}},
            ],
        },
    })

    # --- Statistics ---
    result["statistics"] = {
        "test": "interventional_shap",
        "scm_method": scm_method,
        "n_features": p,
        "n_explain": n_explain,
        "n_converged": n_converged,
        "avg_perm": round(total_perm / max(n_explain, 1)),
        "features": [
            {
                "name": features[i],
                "shap_standard": round(float(mean_abs_mar[i]), 6),
                "shap_interventional": round(float(mean_abs_int[i]), 6),
                "discrepancy": round(float(discrepancy[i]), 6),
                "role": roles.get(i, "non_cause"),
                "se": round(float(mean_se[i]), 6),
                "tce_on_target": round(float(tce[target_idx, i]), 6),
            }
            for i in range(p)
        ],
        "alerts": [a.replace("<<COLOR:warning>>", "").replace("<</COLOR>>", "")
                   for a in alerts],
        "is_identified": scm.is_identified,
    }

    # Guide observation
    top_int = features[rank_int[0]]
    top_mar = features[rank_mar[0]]
    result["guide_observation"] = (
        f"Interventional SHAP: top causal driver is {top_int} "
        f"(|φ|={mean_abs_int[rank_int[0]]:.4f}). "
        + (f"Standard SHAP ranks {top_mar} first instead — "
           f"the difference indicates correlation vs causation."
           if top_int != top_mar else
           "Standard SHAP agrees on the top driver.")
        + f" {len(alerts)} diagnostic alerts."
    )

    return result
