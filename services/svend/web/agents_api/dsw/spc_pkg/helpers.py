"""SPC helper functions — Nelson rules, point rules, OOC markers."""


def _spc_nelson_rules(data, cl, ucl, lcl):
    """Check all 8 Nelson rules and return OOC indices + rule violations."""
    n = len(data)
    sigma = (ucl - cl) / 3 if ucl != cl else 1
    one_sigma_up = cl + sigma
    one_sigma_dn = cl - sigma
    two_sigma_up = cl + 2 * sigma
    two_sigma_dn = cl - 2 * sigma
    ooc_indices = set()
    violations = []

    # Rule 1: Point beyond 3σ (beyond control limits)
    for i in range(n):
        if data[i] > ucl or data[i] < lcl:
            ooc_indices.add(i)

    # Rule 2: 9 consecutive points same side of CL
    for i in range(8, n):
        window = data[i - 8 : i + 1]
        if all(v > cl for v in window) or all(v < cl for v in window):
            ooc_indices.update(range(i - 8, i + 1))
            violations.append(f"Rule 2: 9 same side at {i - 8 + 1}-{i + 1}")
            break

    # Rule 3: 6 consecutive points trending (all increasing or all decreasing)
    for i in range(5, n):
        window = data[i - 5 : i + 1]
        diffs = [window[j + 1] - window[j] for j in range(5)]
        if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
            ooc_indices.update(range(i - 5, i + 1))
            direction = "increasing" if diffs[0] > 0 else "decreasing"
            violations.append(f"Rule 3: 6 {direction} at {i - 5 + 1}-{i + 1}")
            break

    # Rule 4: 14 consecutive points alternating up and down
    if n >= 14:
        for i in range(13, n):
            window = data[i - 13 : i + 1]
            diffs = [window[j + 1] - window[j] for j in range(13)]
            if all(diffs[j] * diffs[j + 1] < 0 for j in range(12)):
                ooc_indices.update(range(i - 13, i + 1))
                violations.append(f"Rule 4: 14 alternating at {i - 13 + 1}-{i + 1}")
                break

    # Rule 5: 2 of 3 beyond 2σ (same side)
    for i in range(2, n):
        w = data[i - 2 : i + 1]
        if sum(1 for v in w if v > two_sigma_up) >= 2:
            ooc_indices.update(range(i - 2, i + 1))
        if sum(1 for v in w if v < two_sigma_dn) >= 2:
            ooc_indices.update(range(i - 2, i + 1))

    # Rule 6: 4 of 5 beyond 1σ (same side)
    for i in range(4, n):
        w = data[i - 4 : i + 1]
        if sum(1 for v in w if v > one_sigma_up) >= 4:
            ooc_indices.update(range(i - 4, i + 1))
        if sum(1 for v in w if v < one_sigma_dn) >= 4:
            ooc_indices.update(range(i - 4, i + 1))

    # Rule 7: 15 consecutive within 1σ (stratification — too little variation)
    if n >= 15:
        for i in range(14, n):
            window = data[i - 14 : i + 1]
            if all(one_sigma_dn <= v <= one_sigma_up for v in window):
                ooc_indices.update(range(i - 14, i + 1))
                violations.append(f"Rule 7: 15 within 1σ at {i - 14 + 1}-{i + 1}")
                break

    # Rule 8: 8 consecutive beyond 1σ on both sides (mixture pattern)
    if n >= 8:
        for i in range(7, n):
            window = data[i - 7 : i + 1]
            if all(v > one_sigma_up or v < one_sigma_dn for v in window):
                ooc_indices.update(range(i - 7, i + 1))
                violations.append(
                    f"Rule 8: 8 beyond 1σ (mixture) at {i - 7 + 1}-{i + 1}"
                )
                break

    return list(sorted(ooc_indices)), violations


def _spc_build_point_rules(data, cl, ucl, lcl, ooc_indices):
    """Build per-point Nelson rule annotations for OOC points.

    Returns dict {index: ["Rule 1: Beyond 3σ", ...]} for each OOC index.
    """
    n = len(data)
    sigma = (ucl - cl) / 3 if ucl != cl else 1
    one_sigma_up = cl + sigma
    one_sigma_dn = cl - sigma
    two_sigma_up = cl + 2 * sigma
    two_sigma_dn = cl - 2 * sigma
    ooc_set = set(ooc_indices)
    rules = {i: [] for i in ooc_indices}

    # Rule 1: Beyond 3σ
    for i in ooc_set:
        if data[i] > ucl or data[i] < lcl:
            rules[i].append("Rule 1: Beyond 3\u03c3")

    # Rule 2: 9 consecutive same side
    for i in range(8, n):
        window = data[i - 8 : i + 1]
        if all(v > cl for v in window) or all(v < cl for v in window):
            for j in range(i - 8, i + 1):
                if j in ooc_set:
                    rules[j].append("Rule 2: 9 same side")
            break

    # Rule 3: 6 consecutive trending
    for i in range(5, n):
        window = data[i - 5 : i + 1]
        diffs = [window[j + 1] - window[j] for j in range(5)]
        if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
            direction = "increasing" if diffs[0] > 0 else "decreasing"
            for j in range(i - 5, i + 1):
                if j in ooc_set:
                    rules[j].append(f"Rule 3: 6 {direction}")
            break

    # Rule 4: 14 alternating
    if n >= 14:
        for i in range(13, n):
            window = data[i - 13 : i + 1]
            diffs = [window[j + 1] - window[j] for j in range(13)]
            if all(diffs[j] * diffs[j + 1] < 0 for j in range(12)):
                for j in range(i - 13, i + 1):
                    if j in ooc_set:
                        rules[j].append("Rule 4: 14 alternating")
                break

    # Rule 5: 2 of 3 beyond 2σ
    for i in range(2, n):
        w = data[i - 2 : i + 1]
        if (
            sum(1 for v in w if v > two_sigma_up) >= 2
            or sum(1 for v in w if v < two_sigma_dn) >= 2
        ):
            for j in range(i - 2, i + 1):
                if j in ooc_set and "Rule 5: 2/3 beyond 2\u03c3" not in rules[j]:
                    rules[j].append("Rule 5: 2/3 beyond 2\u03c3")

    # Rule 6: 4 of 5 beyond 1σ
    for i in range(4, n):
        w = data[i - 4 : i + 1]
        if (
            sum(1 for v in w if v > one_sigma_up) >= 4
            or sum(1 for v in w if v < one_sigma_dn) >= 4
        ):
            for j in range(i - 4, i + 1):
                if j in ooc_set and "Rule 6: 4/5 beyond 1\u03c3" not in rules[j]:
                    rules[j].append("Rule 6: 4/5 beyond 1\u03c3")

    # Rule 7: 15 within 1σ (stratification)
    if n >= 15:
        for i in range(14, n):
            window = data[i - 14 : i + 1]
            if all(one_sigma_dn <= v <= one_sigma_up for v in window):
                for j in range(i - 14, i + 1):
                    if j in ooc_set:
                        rules[j].append("Rule 7: 15 within 1\u03c3")
                break

    # Rule 8: 8 beyond 1σ both sides (mixture)
    if n >= 8:
        for i in range(7, n):
            window = data[i - 7 : i + 1]
            if all(v > one_sigma_up or v < one_sigma_dn for v in window):
                for j in range(i - 7, i + 1):
                    if j in ooc_set:
                        rules[j].append("Rule 8: Mixture pattern")
                break

    return rules


def _spc_add_ooc_markers(plot_data, data, ooc_indices, point_rules=None):
    """Add red markers for OOC points and customdata to main trace for click-to-inspect."""
    n = len(data) if hasattr(data, "__len__") else 0
    ooc_set = set(ooc_indices) if ooc_indices else set()

    # Tag the first (main data) trace with customdata so every point is clickable
    if plot_data and n > 0:
        main_trace = plot_data[0]
        if "customdata" not in main_trace:
            main_trace["customdata"] = [
                [
                    i,
                    (
                        "; ".join(point_rules.get(i, []))
                        if point_rules and i in ooc_set
                        else ""
                    ),
                ]
                for i in range(n)
            ]
            main_trace["hovertemplate"] = (
                "Obs #%{customdata[0]}<br>Value: %{y:.4f}<extra></extra>"
            )

    if not ooc_indices:
        return
    ooc_x = ooc_indices
    ooc_y = [float(data[i]) for i in ooc_indices]
    trace = {
        "type": "scatter",
        "x": ooc_x,
        "y": ooc_y,
        "mode": "markers",
        "name": "Out of Control",
        "marker": {
            "color": "#d94a4a",
            "size": 9,
            "symbol": "diamond",
            "line": {"color": "#fff", "width": 1},
        },
        "showlegend": True,
    }
    # Add customdata for click-to-inspect
    if point_rules is not None:
        trace["customdata"] = [
            [i, "; ".join(point_rules.get(i, []))] for i in ooc_indices
        ]
        trace["hovertemplate"] = (
            "Obs #%{customdata[0]}<br>Value: %{y:.4f}<br>%{customdata[1]}<extra>OOC</extra>"
        )
    else:
        trace["customdata"] = [[i, ""] for i in ooc_indices]
        trace["hovertemplate"] = (
            "Obs #%{customdata[0]}<br>Value: %{y:.4f}<extra>OOC</extra>"
        )
    plot_data.append(trace)
