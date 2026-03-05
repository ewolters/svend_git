"""DSW Education Content — Advanced modules (SPC, ML, Reliability, Viz, Bayesian, etc.).

Split from education.py to stay under architecture file size limits.
"""

from agents_api.dsw.education import _extend

# ═══════════════════════════════════════════════════════════════════════════
# SPC Module
# ═══════════════════════════════════════════════════════════════════════════

_extend(
    {
        ("spc", "imr"): {
            "title": "Understanding I-MR Charts",
            "content": (
                "<dl>"
                "<dt>What is an I-MR chart?</dt>"
                "<dd>Individual-Moving Range chart — for monitoring a process when you have one "
                "measurement per time point (no subgroups). The I chart tracks individual values; "
                "the MR chart tracks consecutive differences.</dd>"
                "<dt>When to use</dt>"
                "<dd>Batch processes, expensive or destructive testing, slow processes where "
                "subgrouping isn't practical. The most common control chart type.</dd>"
                "<dt>Control limits</dt>"
                "<dd>Set at ±3σ from the mean. Points beyond limits signal special cause variation. "
                "σ is estimated from the average moving range (MR̄/1.128).</dd>"
                "<dt>Run rules</dt>"
                "<dd>Beyond 3σ limits, additional patterns signal special causes: 8 points on "
                "one side of center, 6 points trending, 2 of 3 beyond 2σ, etc.</dd>"
                "</dl>"
            ),
        },
        ("spc", "xbar_r"): {
            "title": "Understanding X̄-R Charts",
            "content": (
                "<dl>"
                "<dt>What is an X̄-R chart?</dt>"
                "<dd>Monitors the mean (X̄) and range (R) of subgroups. Best for subgroup "
                "sizes of 2–9. The X̄ chart detects mean shifts; the R chart detects "
                "spread changes.</dd>"
                "<dt>Why subgroups?</dt>"
                "<dd>Rational subgroups (short-term samples) separate within-group variation "
                "(common cause) from between-group variation. This makes the chart sensitive "
                "to process shifts.</dd>"
                "<dt>Reading the charts</dt>"
                "<dd>Always check the R chart first. If the R chart is out of control, the "
                "X̄ chart limits are unreliable. Investigate range signals before mean signals.</dd>"
                "<dt>Subgroup size</dt>"
                "<dd>Larger subgroups make the X̄ chart more sensitive but require more sampling. "
                "n = 5 is the classic default, balancing sensitivity and cost.</dd>"
                "</dl>"
            ),
        },
        ("spc", "xbar_s"): {
            "title": "Understanding X̄-S Charts",
            "content": (
                "<dl>"
                "<dt>What is an X̄-S chart?</dt>"
                "<dd>Like X̄-R but uses standard deviation instead of range. Preferred for "
                "subgroup sizes ≥ 10 where the range is an inefficient estimator of σ.</dd>"
                "<dt>X̄-R vs X̄-S</dt>"
                "<dd>For n &lt; 10, range and s are similarly efficient. For n ≥ 10, the range "
                "ignores most of the data (only min and max). Standard deviation uses all values.</dd>"
                "<dt>Interpreting the S chart</dt>"
                "<dd>Points above the UCL indicate increased variability (special cause). "
                "Points below the LCL indicate decreased variability — possibly good (process "
                "improvement) or suspicious (data truncation).</dd>"
                "<dt>Rational subgrouping</dt>"
                "<dd>Same principles as X̄-R: subgroups should represent short-term, within-group "
                "variation. Between-subgroup variation is what the X̄ chart monitors.</dd>"
                "</dl>"
            ),
        },
        ("spc", "p_chart"): {
            "title": "Understanding P Charts",
            "content": (
                "<dl>"
                "<dt>What is a P chart?</dt>"
                "<dd>Monitors the proportion of defective items per subgroup. Used when each item "
                "is classified as pass or fail and subgroup sizes may vary.</dd>"
                "<dt>Variable subgroup sizes</dt>"
                "<dd>P chart limits adjust with subgroup size — wider for smaller subgroups, "
                "narrower for larger ones. This is correct behavior, not a problem.</dd>"
                "<dt>Assumptions</dt>"
                "<dd>Each item is independent, and the probability of a defect is constant "
                "within each subgroup. Overdispersion (more variation than binomial predicts) "
                "is common — use Laney P' chart in that case.</dd>"
                "<dt>Minimum subgroup size</dt>"
                "<dd>The normal approximation works when np̄ ≥ 5 and n(1−p̄) ≥ 5. For very low "
                "defect rates, you need very large subgroups or should use a c/u chart instead.</dd>"
                "</dl>"
            ),
        },
        ("spc", "np_chart"): {
            "title": "Understanding NP Charts",
            "content": (
                "<dl>"
                "<dt>What is an NP chart?</dt>"
                "<dd>Monitors the count of defective items per subgroup (not the proportion). "
                "Simpler than P charts but requires equal subgroup sizes.</dd>"
                "<dt>NP vs P chart</dt>"
                "<dd>Use NP when subgroup sizes are constant (easier to interpret — counts "
                "are more intuitive than proportions). Use P when subgroup sizes vary.</dd>"
                "<dt>Interpreting signals</dt>"
                "<dd>Points above UCL: more defectives than expected (special cause). "
                "Points below LCL: fewer defectives (possibly improved process — investigate "
                "to sustain the improvement).</dd>"
                "<dt>Control limit calculation</dt>"
                "<dd>Based on the binomial distribution: CL = np̄, UCL = np̄ + 3√(np̄(1−p̄)), "
                "LCL = np̄ − 3√(np̄(1−p̄)).</dd>"
                "</dl>"
            ),
        },
        ("spc", "c_chart"): {
            "title": "Understanding C Charts",
            "content": (
                "<dl>"
                "<dt>What is a C chart?</dt>"
                "<dd>Monitors the count of defects per inspection unit. One unit can have "
                "multiple defects (unlike P charts which track defective items).</dd>"
                "<dt>Defects vs defectives</dt>"
                "<dd>A defective item fails overall. A defect is a single flaw — one item can "
                "have multiple defects. C charts track defects per unit when the inspection "
                "area/opportunity is constant.</dd>"
                "<dt>Assumptions</dt>"
                "<dd>Defects occur independently at a constant rate (Poisson process). "
                "Inspection area must be constant. If area varies, use U charts instead.</dd>"
                "<dt>Control limits</dt>"
                "<dd>Based on the Poisson distribution: CL = c̄, UCL = c̄ + 3√c̄, "
                "LCL = c̄ − 3√c̄.</dd>"
                "</dl>"
            ),
        },
        ("spc", "u_chart"): {
            "title": "Understanding U Charts",
            "content": (
                "<dl>"
                "<dt>What is a U chart?</dt>"
                "<dd>Monitors defects per unit when the inspection area or opportunity varies "
                "between subgroups. The rate-based counterpart to the C chart.</dd>"
                "<dt>U vs C chart</dt>"
                "<dd>Use U when inspection units differ in size (e.g., different length rolls, "
                "different area panels). Use C when units are identical.</dd>"
                "<dt>Overdispersion</dt>"
                "<dd>If the variation exceeds Poisson predictions, the Laney U' chart adjusts "
                "limits to avoid excessive false alarms.</dd>"
                "<dt>Rate interpretation</dt>"
                "<dd>The U chart plots defects per unit, making rates comparable across different "
                "sized inspection units. This is essential for fair comparison.</dd>"
                "</dl>"
            ),
        },
        ("spc", "cusum"): {
            "title": "Understanding CUSUM Charts",
            "content": (
                "<dl>"
                "<dt>What is a CUSUM chart?</dt>"
                "<dd>Cumulative Sum chart — accumulates deviations from a target value. More "
                "sensitive to small, sustained shifts than Shewhart charts.</dd>"
                "<dt>How it works</dt>"
                "<dd>Cumulates positive and negative deviations separately. When either cumulative "
                "sum exceeds a decision interval (H), a shift is signaled.</dd>"
                "<dt>When to use</dt>"
                "<dd>When detecting small shifts (0.5–2σ) is critical. CUSUM detects a 1σ shift "
                "about 5× faster than an I-MR chart.</dd>"
                "<dt>Parameters</dt>"
                "<dd><strong>k</strong>: Reference value (typically half the shift to detect). "
                "<strong>H</strong>: Decision interval (larger = fewer false alarms but slower "
                "detection). Common default: k = 0.5, H = 5.</dd>"
                "</dl>"
            ),
        },
        ("spc", "ewma"): {
            "title": "Understanding EWMA Charts",
            "content": (
                "<dl>"
                "<dt>What is an EWMA chart?</dt>"
                "<dd>Exponentially Weighted Moving Average — gives more weight to recent "
                "observations and less to older ones. Like CUSUM, it's sensitive to small shifts.</dd>"
                "<dt>The smoothing parameter (λ)</dt>"
                "<dd>Controls memory: small λ (0.05–0.1) for detecting small shifts, "
                "large λ (0.2–0.4) for larger shifts. λ = 1 reduces to a Shewhart chart.</dd>"
                "<dt>EWMA vs CUSUM vs Shewhart</dt>"
                "<dd>All detect large shifts similarly. For small shifts: CUSUM and EWMA are "
                "both much faster. EWMA is slightly easier to implement and interpret.</dd>"
                "<dt>Robustness</dt>"
                "<dd>EWMA is robust to non-normality due to the averaging effect. Good for "
                "individual measurements where normality is questionable.</dd>"
                "</dl>"
            ),
        },
        ("spc", "laney_p"): {
            "title": "Understanding Laney P' Charts",
            "content": (
                "<dl>"
                "<dt>What is a Laney P' chart?</dt>"
                "<dd>A modified P chart that adjusts for overdispersion (excess variation beyond "
                "what the binomial model predicts). Prevents excessive false alarms.</dd>"
                "<dt>When to use</dt>"
                "<dd>When large subgroups produce too many out-of-control signals on a standard "
                "P chart. Overdispersion is common with large samples because the binomial "
                "assumption of constant p is rarely exact.</dd>"
                "<dt>The sigma-Z factor</dt>"
                "<dd>Laney calculates a sigma-Z factor from the standardized residuals. "
                "If σ_Z ≈ 1, standard P chart is fine. If σ_Z &gt; 1, limits need widening.</dd>"
                "<dt>Real-world relevance</dt>"
                "<dd>Very common in high-volume manufacturing, healthcare (large patient counts), "
                "and any application where subgroup sizes are in the hundreds or thousands.</dd>"
                "</dl>"
            ),
        },
        ("spc", "laney_u"): {
            "title": "Understanding Laney U' Charts",
            "content": (
                "<dl>"
                "<dt>What is a Laney U' chart?</dt>"
                "<dd>A modified U chart that adjusts for overdispersion in defect rate data. "
                "The rate-based equivalent of the Laney P' chart.</dd>"
                "<dt>When to use</dt>"
                "<dd>When U chart limits are too tight due to large inspection areas, causing "
                "false alarms. The Poisson assumption (mean = variance) is violated.</dd>"
                "<dt>How it works</dt>"
                "<dd>Estimates the overdispersion factor from sigma-Z of standardized residuals, "
                "then inflates the control limits accordingly.</dd>"
                "<dt>Interpretation</dt>"
                "<dd>Same as U chart — signals indicate rate changes. But with Laney adjustment, "
                "signals are genuine process changes, not artifacts of Poisson model inadequacy.</dd>"
                "</dl>"
            ),
        },
        ("spc", "moving_average"): {
            "title": "Understanding Moving Average Charts",
            "content": (
                "<dl>"
                "<dt>What is a moving average chart?</dt>"
                "<dd>Plots the average of the last w observations at each time point. Smooths "
                "out noise to reveal trends and shifts that are hard to see in raw data.</dd>"
                "<dt>Window size (w)</dt>"
                "<dd>Larger windows = more smoothing but slower response. Smaller windows = "
                "less smoothing but faster response. w = 3–5 is common.</dd>"
                "<dt>Moving average vs EWMA</dt>"
                "<dd>Moving average gives equal weight to the last w observations and zero to "
                "all others. EWMA gives exponentially decreasing weights — smoother and more "
                "responsive to recent changes.</dd>"
                "<dt>Control limits</dt>"
                "<dd>Limits are tighter than I-MR because averaging reduces variation. They "
                "adjust for the number of observations in each average.</dd>"
                "</dl>"
            ),
        },
        ("spc", "zone_chart"): {
            "title": "Understanding Zone Charts",
            "content": (
                "<dl>"
                "<dt>What is a zone chart?</dt>"
                "<dd>Divides the control chart into zones (A, B, C — each 1σ wide) and assigns "
                "scores based on where points fall. Cumulates scores to detect patterns "
                "that individual run rules might miss.</dd>"
                "<dt>How scoring works</dt>"
                "<dd>Zone C (near center): 0 points. Zone B (1–2σ): 2 points. Zone A (2–3σ): "
                "4 points. Beyond 3σ: 8 points. Scores reset when the cumulative sum is "
                "signaled.</dd>"
                "<dt>When to use</dt>"
                "<dd>As a more sensitive alternative to standard Shewhart run rules. Zone charts "
                "detect small shifts faster because they accumulate evidence from successive "
                "near-misses.</dd>"
                "<dt>Interpretation</dt>"
                "<dd>A signal (cumulative score exceeds threshold) indicates the process has "
                "shifted. Investigate the sequence of scores to identify when the shift began.</dd>"
                "</dl>"
            ),
        },
        ("spc", "mewma"): {
            "title": "Understanding MEWMA Charts",
            "content": (
                "<dl>"
                "<dt>What is MEWMA?</dt>"
                "<dd>Multivariate EWMA — monitors multiple correlated variables simultaneously. "
                "It detects small shifts in the multivariate mean vector.</dd>"
                "<dt>Why multivariate?</dt>"
                "<dd>Running separate charts per variable misses shifts in the correlation "
                "structure. MEWMA catches shifts that no individual chart would detect.</dd>"
                "<dt>The smoothing parameter</dt>"
                "<dd>Like univariate EWMA, small r (0.05–0.1) detects small shifts; larger r "
                "detects larger shifts. The optimal r depends on the shift size you need to detect.</dd>"
                "<dt>Interpreting signals</dt>"
                "<dd>A signal means the multivariate mean has shifted, but doesn't tell you "
                "which variable changed. Follow up with individual charts or decomposition "
                "to identify the source.</dd>"
                "</dl>"
            ),
        },
        ("spc", "generalized_variance"): {
            "title": "Understanding Generalized Variance Charts",
            "content": (
                "<dl>"
                "<dt>What is a generalized variance chart?</dt>"
                "<dd>Monitors the determinant of the covariance matrix (|S|) over time. "
                "It detects changes in the overall multivariate spread — the volume of "
                "the data cloud in multivariate space.</dd>"
                "<dt>When to use</dt>"
                "<dd>When monitoring the consistency (not just the mean) of a multivariate "
                "process. Complements MEWMA which monitors the mean vector.</dd>"
                "<dt>Interpretation</dt>"
                "<dd>Increasing |S| means the process is becoming more variable (less consistent). "
                "Decreasing |S| means variables are becoming more tightly correlated or less variable.</dd>"
                "<dt>Practical application</dt>"
                "<dd>Monitoring multivariate process stability before computing multivariate "
                "capability indices. If the covariance structure changes, capability is undefined.</dd>"
                "</dl>"
            ),
        },
        ("spc", "capability"): {
            "title": "Understanding Process Capability Analysis",
            "content": (
                "<dl>"
                "<dt>What is process capability?</dt>"
                "<dd>A comparison of the voice of the process (natural variation) against the "
                "voice of the customer (specification limits). It answers: 'Can this process "
                "consistently meet requirements?'</dd>"
                "<dt>Cp vs Cpk</dt>"
                "<dd><strong>Cp</strong>: Potential capability — ignores centering, only considers "
                "spread. <strong>Cpk</strong>: Actual capability — accounts for how well the "
                "process is centered within specs. Cpk ≤ Cp always.</dd>"
                "<dt>Target values</dt>"
                "<dd>Cpk ≥ 1.33 for existing processes (4σ from nearest spec). Cpk ≥ 1.67 for "
                "new processes or safety-critical applications.</dd>"
                "<dt>Pp/Ppk vs Cp/Cpk</dt>"
                "<dd>Cp/Cpk use within-subgroup σ (process capability). Pp/Ppk use overall σ "
                "(process performance). The gap between them indicates how much variation comes "
                "from between-subgroup shifts.</dd>"
                "</dl>"
            ),
        },
        ("spc", "nonnormal_capability"): {
            "title": "Understanding Non-Normal Capability (SPC)",
            "content": (
                "<dl>"
                "<dt>Why non-normal capability in SPC?</dt>"
                "<dd>When control chart data is non-normal (skewed, bounded, or heavy-tailed), "
                "standard Cpk calculations based on ±3σ are wrong. The actual tail probabilities "
                "differ from what the normal distribution predicts.</dd>"
                "<dt>Methods</dt>"
                "<dd>Percentile-based (Cnpk): Uses actual data percentiles instead of ±3σ. "
                "Transformation-based: Box-Cox or Johnson transform to normality. "
                "Distribution fitting: Fit the correct distribution and compute from its CDF.</dd>"
                "<dt>Which to choose</dt>"
                "<dd>If the non-normal distribution is known (e.g., Weibull for time data), "
                "use distribution fitting. If unknown, percentile-based is safest. "
                "Transformation is a good middle ground.</dd>"
                "<dt>Impact</dt>"
                "<dd>For a skewed process, normal-based Cpk may be off by 0.3–0.5 or more. "
                "This directly affects defect rate predictions and acceptance decisions.</dd>"
                "</dl>"
            ),
        },
        ("spc", "between_within"): {
            "title": "Understanding Between/Within Capability",
            "content": (
                "<dl>"
                "<dt>What is between/within analysis?</dt>"
                "<dd>Decomposes total variation into between-subgroup and within-subgroup "
                "components. Shows whether variation comes from short-term repeatability or "
                "longer-term shifts.</dd>"
                "<dt>Why does this matter?</dt>"
                "<dd>If between-subgroup variation dominates, the process shifts over time — "
                "fix the assignable causes. If within-subgroup variation dominates, the "
                "inherent process noise is too high — fundamental process change needed.</dd>"
                "<dt>Cpk vs Ppk gap</dt>"
                "<dd>A large gap between Cpk (within) and Ppk (overall) quantifies the impact "
                "of between-subgroup variation. Closing this gap is a specific improvement target.</dd>"
                "<dt>Practical use</dt>"
                "<dd>Before launching improvement projects, this analysis tells you whether "
                "to focus on stabilizing the process (reducing shifts) or reducing inherent "
                "variability (process redesign).</dd>"
                "</dl>"
            ),
        },
        ("spc", "conformal_control"): {
            "title": "Understanding Conformal Control Charts",
            "content": (
                "<dl>"
                "<dt>What is a conformal control chart?</dt>"
                "<dd>A distribution-free control chart based on conformal prediction. It uses "
                "nonconformity scores rather than distributional assumptions to set control limits.</dd>"
                "<dt>When to use</dt>"
                "<dd>When data is non-normal and traditional charts are unreliable. Conformal "
                "methods provide valid coverage guarantees regardless of the data distribution.</dd>"
                "<dt>How it works</dt>"
                "<dd>Each new observation is compared against a calibration set to compute a "
                "p-value (conformal p-value). Points with very small p-values are out of control.</dd>"
                "<dt>Advantages</dt>"
                "<dd>Distribution-free, finite-sample valid, and handles multivariate data "
                "naturally. The main limitation is requiring a representative calibration set.</dd>"
                "</dl>"
            ),
        },
        ("spc", "conformal_monitor"): {
            "title": "Understanding Conformal Process Monitoring",
            "content": (
                "<dl>"
                "<dt>What is conformal monitoring?</dt>"
                "<dd>Continuous process monitoring using conformal prediction. Unlike one-time "
                "control charts, it adaptively updates as new data arrives and monitors "
                "for distributional changes in real time.</dd>"
                "<dt>Martingale approach</dt>"
                "<dd>Conformal p-values are combined via a betting martingale. When the "
                "martingale value exceeds a threshold, a change is detected — with "
                "mathematically guaranteed false alarm control.</dd>"
                "<dt>Advantages over traditional SPC</dt>"
                "<dd>No normality assumption, handles multivariate data, provides valid "
                "Type I error control in finite samples, and can detect distributional "
                "changes (not just mean shifts).</dd>"
                "<dt>When to use</dt>"
                "<dd>Complex processes with non-normal, multivariate, or non-stationary data "
                "where traditional SPC assumptions are violated.</dd>"
                "</dl>"
            ),
        },
        ("spc", "entropy_spc"): {
            "title": "Understanding Entropy-Based SPC",
            "content": (
                "<dl>"
                "<dt>What is entropy SPC?</dt>"
                "<dd>Monitors process entropy (information content) over time. Entropy measures "
                "the uncertainty or complexity of the process distribution — changes in entropy "
                "signal distributional shifts.</dd>"
                "<dt>Why entropy?</dt>"
                "<dd>Traditional SPC detects mean and variance shifts. Entropy detects any "
                "distributional change — including shape changes, modality changes, and "
                "tail behavior changes that mean/variance charts miss.</dd>"
                "<dt>How to interpret</dt>"
                "<dd>Increasing entropy: Process becoming more variable or unpredictable. "
                "Decreasing entropy: Process becoming more concentrated or predictable. "
                "Sudden changes: Distributional shift occurred.</dd>"
                "<dt>Applications</dt>"
                "<dd>Complex manufacturing processes where the failure mode affects the "
                "distribution shape rather than just the mean or variance.</dd>"
                "</dl>"
            ),
        },
        ("spc", "degradation_capability"): {
            "title": "Understanding Degradation Capability",
            "content": (
                "<dl>"
                "<dt>What is degradation capability?</dt>"
                "<dd>Assesses whether a degrading process will remain capable over its planned "
                "life. It models the degradation trend and projects when capability will drop "
                "below acceptable levels.</dd>"
                "<dt>How it works</dt>"
                "<dd>Fits a degradation model (linear, exponential, or power) to time-ordered "
                "capability measurements. Projects future capability with prediction intervals.</dd>"
                "<dt>Key output</dt>"
                "<dd>The time at which predicted Cpk drops below the threshold (e.g., 1.33). "
                "This drives maintenance scheduling and replacement planning.</dd>"
                "<dt>Applications</dt>"
                "<dd>Tool wear monitoring, equipment aging, filter degradation, battery capacity "
                "fade — any process where performance deteriorates predictably over time.</dd>"
                "</dl>"
            ),
        },
    }
)


# ═══════════════════════════════════════════════════════════════════════════
# ML Module
# ═══════════════════════════════════════════════════════════════════════════

_extend(
    {
        ("ml", "classification"): {
            "title": "Understanding Classification Models",
            "content": (
                "<dl>"
                "<dt>What is classification?</dt>"
                "<dd>Predicting a categorical outcome (pass/fail, defect type, severity level) "
                "from input features. Models learn the boundary between categories from "
                "labeled training data.</dd>"
                "<dt>Key metrics</dt>"
                "<dd><strong>Accuracy</strong>: Overall correct predictions. <strong>Precision</strong>: "
                "Of predicted positives, how many are correct. <strong>Recall</strong>: Of actual "
                "positives, how many were caught. <strong>F1</strong>: Harmonic mean of precision/recall.</dd>"
                "<dt>Confusion matrix</dt>"
                "<dd>Shows true positives, false positives, true negatives, false negatives. "
                "Reveals whether errors are missed defects (costly) or false alarms (wasteful).</dd>"
                "<dt>Which metric to optimize?</dt>"
                "<dd>In quality: usually recall (catching all defects) matters more than precision. "
                "The cost of missing a defect vs false alarm determines the right trade-off.</dd>"
                "</dl>"
            ),
        },
        ("ml", "regression_ml"): {
            "title": "Understanding ML Regression Models",
            "content": (
                "<dl>"
                "<dt>What is ML regression?</dt>"
                "<dd>Predicting a continuous outcome from input features using machine learning "
                "models (random forest, gradient boosting, neural networks). More flexible "
                "than linear regression but less interpretable.</dd>"
                "<dt>Key metrics</dt>"
                "<dd><strong>RMSE</strong>: Root Mean Square Error — in the units of the response. "
                "<strong>MAE</strong>: Mean Absolute Error — less sensitive to outliers. "
                "<strong>R²</strong>: Variance explained — same interpretation as linear regression.</dd>"
                "<dt>Train/test split</dt>"
                "<dd>Model performance on training data is optimistic. Always evaluate on held-out "
                "test data to assess generalization. Cross-validation gives more reliable estimates.</dd>"
                "<dt>Feature importance</dt>"
                "<dd>Shows which predictors matter most. Permutation importance is model-agnostic "
                "and reliable. Use SHAP values for detailed per-prediction explanations.</dd>"
                "</dl>"
            ),
        },
        ("ml", "model_compare"): {
            "title": "Understanding Model Comparison",
            "content": (
                "<dl>"
                "<dt>What does model comparison do?</dt>"
                "<dd>Trains multiple model types on the same data and compares their performance. "
                "Identifies which algorithm works best for your specific problem.</dd>"
                "<dt>Why compare?</dt>"
                "<dd>No single algorithm dominates. Random forest may beat gradient boosting on "
                "one dataset and lose on another. Systematic comparison prevents premature "
                "commitment to a suboptimal model.</dd>"
                "<dt>Fair comparison</dt>"
                "<dd>All models use the same train/test split and cross-validation folds. "
                "Performance is compared on the same held-out data to prevent information leakage.</dd>"
                "<dt>Beyond accuracy</dt>"
                "<dd>Consider training time, prediction speed, interpretability, and robustness "
                "to new data. The 'best' model balances all of these, not just test accuracy.</dd>"
                "</dl>"
            ),
        },
        ("ml", "xgboost"): {
            "title": "Understanding XGBoost",
            "content": (
                "<dl>"
                "<dt>What is XGBoost?</dt>"
                "<dd>Extreme Gradient Boosting — an ensemble method that sequentially builds "
                "decision trees, each correcting the errors of the previous ones. Often the "
                "best-performing algorithm for structured/tabular data.</dd>"
                "<dt>Key hyperparameters</dt>"
                "<dd><strong>n_estimators</strong>: Number of trees. <strong>max_depth</strong>: "
                "Tree complexity (3–6 typical). <strong>learning_rate</strong>: Step size "
                "(smaller = more trees needed but better generalization).</dd>"
                "<dt>Regularization</dt>"
                "<dd>Built-in L1 and L2 regularization prevents overfitting. XGBoost handles "
                "this better than random forests, especially with many features.</dd>"
                "<dt>Feature importance</dt>"
                "<dd>Gain-based importance shows which features contribute most to predictions. "
                "SHAP values provide more reliable and detailed feature attributions.</dd>"
                "</dl>"
            ),
        },
        ("ml", "lightgbm"): {
            "title": "Understanding LightGBM",
            "content": (
                "<dl>"
                "<dt>What is LightGBM?</dt>"
                "<dd>Light Gradient Boosting Machine — similar to XGBoost but faster on large "
                "datasets. Uses histogram-based splitting and leaf-wise tree growth.</dd>"
                "<dt>LightGBM vs XGBoost</dt>"
                "<dd>LightGBM is typically faster and more memory-efficient. XGBoost is more "
                "established with better documentation. Performance is usually similar — "
                "try both and compare.</dd>"
                "<dt>Handling categoricals</dt>"
                "<dd>LightGBM natively handles categorical features without one-hot encoding, "
                "which can improve both speed and accuracy.</dd>"
                "<dt>Overfitting risks</dt>"
                "<dd>Leaf-wise growth can overfit on small datasets. Use min_data_in_leaf and "
                "num_leaves to control complexity. Cross-validation is essential.</dd>"
                "</dl>"
            ),
        },
        ("ml", "shap_explain"): {
            "title": "Understanding SHAP Explanations",
            "content": (
                "<dl>"
                "<dt>What is SHAP?</dt>"
                "<dd>SHapley Additive exPlanations — a game-theoretic approach to explaining "
                "individual predictions. Each feature gets a SHAP value showing its contribution "
                "to moving the prediction from the average.</dd>"
                "<dt>Why SHAP over feature importance?</dt>"
                "<dd>SHAP values are consistent, locally accurate, and additive. They explain "
                "<em>each prediction</em> individually, not just global patterns. This reveals "
                "when and why a model makes mistakes.</dd>"
                "<dt>Reading the plots</dt>"
                "<dd><strong>Summary plot</strong>: Feature importance + direction of effect. "
                "<strong>Dependence plot</strong>: How SHAP values vary with feature values. "
                "<strong>Force plot</strong>: Contribution breakdown for a single prediction.</dd>"
                "<dt>Actionable insights</dt>"
                "<dd>SHAP reveals which factors drive predictions for specific cases, enabling "
                "targeted process adjustments rather than blanket changes.</dd>"
                "</dl>"
            ),
        },
        ("ml", "hyperparameter_tune"): {
            "title": "Understanding Hyperparameter Tuning",
            "content": (
                "<dl>"
                "<dt>What is hyperparameter tuning?</dt>"
                "<dd>Systematically searching for the best model configuration (tree depth, "
                "learning rate, regularization, etc.) to maximize performance. Like DOE for "
                "machine learning.</dd>"
                "<dt>Methods</dt>"
                "<dd><strong>Grid search</strong>: Try all combinations (exhaustive but slow). "
                "<strong>Random search</strong>: Random combinations (surprisingly effective). "
                "<strong>Bayesian optimization</strong>: Learns from past trials (most efficient).</dd>"
                "<dt>Cross-validation</dt>"
                "<dd>Each configuration is evaluated using k-fold cross-validation to prevent "
                "overfitting to the validation set. Never tune on the test set.</dd>"
                "<dt>Diminishing returns</dt>"
                "<dd>Default parameters often get you 90% of the way. Tuning typically improves "
                "performance by 1–5%. Focus on the most impactful hyperparameters first.</dd>"
                "</dl>"
            ),
        },
        ("ml", "clustering"): {
            "title": "Understanding Clustering Analysis",
            "content": (
                "<dl>"
                "<dt>What is clustering?</dt>"
                "<dd>Unsupervised learning that groups similar observations together without "
                "pre-defined labels. Discovers natural structure in data.</dd>"
                "<dt>K-Means vs hierarchical vs DBSCAN</dt>"
                "<dd><strong>K-Means</strong>: Fast, assumes spherical clusters, needs k specified. "
                "<strong>Hierarchical</strong>: Builds a tree of clusters, no k needed. "
                "<strong>DBSCAN</strong>: Finds arbitrary shapes, handles noise, no k needed.</dd>"
                "<dt>Choosing k</dt>"
                "<dd>Elbow method (diminishing returns in within-cluster variance), silhouette "
                "score (cluster separation quality), or domain knowledge.</dd>"
                "<dt>Applications in quality</dt>"
                "<dd>Identifying process modes, segmenting products, grouping failure patterns, "
                "and discovering hidden subpopulations in process data.</dd>"
                "</dl>"
            ),
        },
        ("ml", "pca"): {
            "title": "Understanding Principal Component Analysis",
            "content": (
                "<dl>"
                "<dt>What is PCA?</dt>"
                "<dd>Transforms correlated variables into uncorrelated principal components, "
                "ordered by variance explained. The first few PCs capture most of the information.</dd>"
                "<dt>How many PCs to keep?</dt>"
                "<dd>Keep enough to explain 80–95% of total variance. The scree plot shows "
                "diminishing returns — the 'elbow' indicates the right cutoff.</dd>"
                "<dt>Loading interpretation</dt>"
                "<dd>Loadings show how each original variable contributes to each PC. Large "
                "loadings (positive or negative) indicate strong contributors.</dd>"
                "<dt>Applications</dt>"
                "<dd>Dimensionality reduction for visualization, multicollinearity removal "
                "before regression, multivariate process monitoring (T² and SPE charts), "
                "and feature engineering.</dd>"
                "</dl>"
            ),
        },
        ("ml", "feature"): {
            "title": "Understanding Feature Selection/Engineering",
            "content": (
                "<dl>"
                "<dt>What is feature selection?</dt>"
                "<dd>Identifying which input variables are most relevant for prediction. Removing "
                "irrelevant features reduces overfitting, speeds training, and improves "
                "interpretability.</dd>"
                "<dt>Methods</dt>"
                "<dd><strong>Filter</strong>: Correlation, mutual information (fast, model-free). "
                "<strong>Wrapper</strong>: Forward/backward selection (model-specific). "
                "<strong>Embedded</strong>: LASSO, tree importance (built into model training).</dd>"
                "<dt>Feature engineering</dt>"
                "<dd>Creating new features from existing ones — interactions, polynomials, "
                "domain-specific transformations. Good features often matter more than model choice.</dd>"
                "<dt>Multicollinearity check</dt>"
                "<dd>Highly correlated features add noise without information. VIF &gt; 5 suggests "
                "removal. PCA can combine correlated features into independent components.</dd>"
                "</dl>"
            ),
        },
        ("ml", "bayesian_regression"): {
            "title": "Understanding Bayesian ML Regression",
            "content": (
                "<dl>"
                "<dt>What is Bayesian ML regression?</dt>"
                "<dd>Regression with full posterior distributions over predictions. Unlike point "
                "estimates, it quantifies prediction uncertainty — critical for decision-making.</dd>"
                "<dt>Prediction uncertainty</dt>"
                "<dd>Each prediction comes with a credible interval showing the range of "
                "plausible values. Wider intervals mean less certainty.</dd>"
                "<dt>Prior specification</dt>"
                "<dd>Priors encode existing knowledge about the relationship. Weakly informative "
                "priors regularize without dominating the data.</dd>"
                "<dt>When to use</dt>"
                "<dd>When prediction uncertainty matters (safety-critical, cost-sensitive), "
                "with small datasets, or when you have genuine prior knowledge to incorporate.</dd>"
                "</dl>"
            ),
        },
        ("ml", "gam"): {
            "title": "Understanding Generalized Additive Models (GAM)",
            "content": (
                "<dl>"
                "<dt>What is a GAM?</dt>"
                "<dd>A flexible regression model that uses smooth functions for each predictor: "
                "y = f₁(x₁) + f₂(x₂) + ... The smooth functions (splines) capture nonlinear "
                "effects while maintaining additive interpretability.</dd>"
                "<dt>GAM vs linear vs tree models</dt>"
                "<dd>More flexible than linear (captures curves). More interpretable than trees "
                "(each effect is a smooth function you can plot). Good middle ground.</dd>"
                "<dt>Partial dependence plots</dt>"
                "<dd>Each smooth function can be plotted individually — showing exactly how each "
                "predictor relates to the response. These are the GAM's greatest strength.</dd>"
                "<dt>Smoothing parameter</dt>"
                "<dd>Controls the wiggliness of each smooth function. Too smooth = underfitting. "
                "Too wiggly = overfitting. Cross-validation selects the optimal smoothness.</dd>"
                "</dl>"
            ),
        },
        ("ml", "isolation_forest"): {
            "title": "Understanding Isolation Forest",
            "content": (
                "<dl>"
                "<dt>What is Isolation Forest?</dt>"
                "<dd>An anomaly detection algorithm that isolates outliers by randomly partitioning "
                "data. Anomalies are easier to isolate (require fewer splits) than normal points.</dd>"
                "<dt>How it works</dt>"
                "<dd>Builds random trees by selecting random features and random split points. "
                "The average path length to isolate a point becomes its anomaly score — shorter "
                "paths = more anomalous.</dd>"
                "<dt>Contamination parameter</dt>"
                "<dd>The expected proportion of outliers. Setting it too high flags normal points; "
                "too low misses real anomalies. Start with domain knowledge of typical defect rates.</dd>"
                "<dt>Multivariate advantage</dt>"
                "<dd>Unlike z-scores, Isolation Forest detects multivariate anomalies — points that "
                "are normal on each variable individually but unusual in combination.</dd>"
                "</dl>"
            ),
        },
        ("ml", "gaussian_process"): {
            "title": "Understanding Gaussian Process Regression",
            "content": (
                "<dl>"
                "<dt>What is a Gaussian Process?</dt>"
                "<dd>A non-parametric Bayesian regression method that provides predictions with "
                "uncertainty estimates. It defines a distribution over functions, not just "
                "point predictions.</dd>"
                "<dt>Uncertainty quantification</dt>"
                "<dd>Predictions near training data have narrow uncertainty bands. Predictions "
                "far from training data have wide bands. This 'knows what it doesn't know'.</dd>"
                "<dt>Kernel functions</dt>"
                "<dd>The kernel defines the smoothness and structure of the learned function. "
                "RBF (squared exponential) is the default. Matérn kernels are common alternatives.</dd>"
                "<dt>Limitations</dt>"
                "<dd>Scales as O(n³) — impractical for &gt; 5000 data points without approximations. "
                "Best for smaller datasets where uncertainty quantification justifies the cost.</dd>"
                "</dl>"
            ),
        },
        ("ml", "pls"): {
            "title": "Understanding Partial Least Squares (PLS)",
            "content": (
                "<dl>"
                "<dt>What is PLS?</dt>"
                "<dd>A regression method designed for situations with many correlated predictors "
                "and few samples. It finds latent components that maximize covariance between "
                "predictors and response.</dd>"
                "<dt>PLS vs PCR vs OLS</dt>"
                "<dd>OLS fails with multicollinearity. PCR reduces predictors via PCA but ignores "
                "the response. PLS finds components that are relevant to both prediction and "
                "explaining the response — usually better.</dd>"
                "<dt>When to use</dt>"
                "<dd>Spectroscopy, chemometrics, sensor data — any domain where p &gt; n (more "
                "variables than observations) or variables are highly correlated.</dd>"
                "<dt>Number of components</dt>"
                "<dd>Cross-validation selects the optimal number. Too few = underfitting. "
                "Too many = overfitting (approaches OLS problems).</dd>"
                "</dl>"
            ),
        },
        ("ml", "sem"): {
            "title": "Understanding Structural Equation Modeling",
            "content": (
                "<dl>"
                "<dt>What is SEM?</dt>"
                "<dd>A framework for modeling complex relationships including latent variables "
                "(unobserved constructs), mediating effects, and simultaneous equations. "
                "Combines factor analysis with path analysis.</dd>"
                "<dt>Measurement vs structural model</dt>"
                "<dd>The measurement model defines how observed variables relate to latent "
                "constructs. The structural model defines relationships between constructs.</dd>"
                "<dt>Fit indices</dt>"
                "<dd><strong>CFI</strong> &gt; 0.95, <strong>RMSEA</strong> &lt; 0.06, "
                "<strong>SRMR</strong> &lt; 0.08 indicate good fit. No single index is sufficient — "
                "report multiple indices.</dd>"
                "<dt>When to use</dt>"
                "<dd>When you have theoretical models with latent variables (quality culture, "
                "employee engagement) or complex mediating/moderating relationships.</dd>"
                "</dl>"
            ),
        },
        ("ml", "regularized_regression"): {
            "title": "Understanding Regularized Regression",
            "content": (
                "<dl>"
                "<dt>What is regularization?</dt>"
                "<dd>Adding a penalty to the loss function to shrink coefficients toward zero. "
                "Prevents overfitting and handles multicollinearity.</dd>"
                "<dt>LASSO (L1) vs Ridge (L2) vs Elastic Net</dt>"
                "<dd><strong>LASSO</strong>: Shrinks some coefficients to exactly zero (feature selection). "
                "<strong>Ridge</strong>: Shrinks all coefficients but keeps all features. "
                "<strong>Elastic Net</strong>: Combines both — good default when unsure.</dd>"
                "<dt>Choosing the penalty strength (λ)</dt>"
                "<dd>Cross-validation selects the optimal λ. Larger λ = more shrinkage = simpler "
                "model. The CV curve shows the bias-variance trade-off.</dd>"
                "<dt>When to use</dt>"
                "<dd>When you have many predictors relative to observations, multicollinearity, "
                "or want automatic feature selection (LASSO).</dd>"
                "</dl>"
            ),
        },
        ("ml", "discriminant_analysis"): {
            "title": "Understanding Discriminant Analysis",
            "content": (
                "<dl>"
                "<dt>What is discriminant analysis?</dt>"
                "<dd>Finds linear combinations of features that best separate known groups. "
                "Both a classification method and a dimensionality reduction technique.</dd>"
                "<dt>LDA vs QDA</dt>"
                "<dd><strong>LDA</strong>: Assumes equal covariance matrices across groups (linear "
                "boundary). <strong>QDA</strong>: Allows different covariances (quadratic boundary). "
                "LDA is more stable with small samples.</dd>"
                "<dt>Discriminant functions</dt>"
                "<dd>The coefficients show which variables best discriminate between groups. "
                "The first function captures the most discrimination, the second captures "
                "the next most, etc.</dd>"
                "<dt>Applications</dt>"
                "<dd>Classifying products, defect typing, process state identification — any "
                "multivariate classification where interpretability matters.</dd>"
                "</dl>"
            ),
        },
        ("ml", "factor_analysis"): {
            "title": "Understanding Factor Analysis",
            "content": (
                "<dl>"
                "<dt>What is factor analysis?</dt>"
                "<dd>Identifies latent factors that explain the correlations among observed "
                "variables. Unlike PCA, it models measurement error explicitly.</dd>"
                "<dt>Factor analysis vs PCA</dt>"
                "<dd>PCA maximizes variance explained (data reduction). Factor analysis models "
                "the correlation structure (latent variable discovery). Use PCA for prediction, "
                "factor analysis for understanding constructs.</dd>"
                "<dt>Rotation</dt>"
                "<dd><strong>Varimax</strong>: Orthogonal rotation — factors are uncorrelated. "
                "<strong>Promax</strong>: Oblique rotation — factors can correlate. Oblique is "
                "usually more realistic for real-world data.</dd>"
                "<dt>Factor loadings</dt>"
                "<dd>Show how strongly each variable relates to each factor. Loadings &gt; 0.4 "
                "indicate a meaningful relationship. Cross-loadings suggest the factor "
                "structure isn't clean.</dd>"
                "</dl>"
            ),
        },
        ("ml", "correspondence_analysis"): {
            "title": "Understanding Correspondence Analysis",
            "content": (
                "<dl>"
                "<dt>What is correspondence analysis?</dt>"
                "<dd>A dimensionality reduction technique for categorical data (contingency tables). "
                "The categorical equivalent of PCA — visualizes the association between row and "
                "column categories.</dd>"
                "<dt>The biplot</dt>"
                "<dd>Row and column categories are plotted in the same space. Categories close "
                "together in the plot are associated. Distance from the origin indicates "
                "distinctiveness.</dd>"
                "<dt>Inertia</dt>"
                "<dd>The total inertia (chi-square/n) measures the overall association. The "
                "proportion explained by each dimension shows how well the 2D plot captures "
                "the full association structure.</dd>"
                "<dt>Applications</dt>"
                "<dd>Market research, defect-by-cause analysis, survey data exploration — any "
                "situation with two categorical variables and a contingency table.</dd>"
                "</dl>"
            ),
        },
        ("ml", "item_analysis"): {
            "title": "Understanding Item Analysis",
            "content": (
                "<dl>"
                "<dt>What is item analysis?</dt>"
                "<dd>Evaluates the quality of individual items in a test or survey. Identifies "
                "which items are good discriminators and which should be revised or removed.</dd>"
                "<dt>Key metrics</dt>"
                "<dd><strong>Cronbach's α</strong>: Internal consistency (reliability). ≥ 0.70 "
                "for research, ≥ 0.80 for applied use. <strong>Item-total correlation</strong>: "
                "How well each item correlates with the total score. &lt; 0.3 is weak.</dd>"
                "<dt>Difficulty and discrimination</dt>"
                "<dd>Difficulty: Proportion answering correctly (easy = high, hard = low). "
                "Discrimination: How well the item separates high and low performers.</dd>"
                "<dt>α if item deleted</dt>"
                "<dd>If removing an item increases Cronbach's α, the item is hurting reliability "
                "and should be revised. This is the most actionable metric.</dd>"
                "</dl>"
            ),
        },
    }
)


# ═══════════════════════════════════════════════════════════════════════════
# Reliability Module
# ═══════════════════════════════════════════════════════════════════════════

_extend(
    {
        ("reliability", "weibull"): {
            "title": "Understanding Weibull Reliability Analysis",
            "content": (
                "<dl>"
                "<dt>What is Weibull reliability analysis?</dt>"
                "<dd>The workhorse of reliability engineering. Fits time-to-failure data to a "
                "Weibull distribution to characterize failure patterns, estimate B-lives, and "
                "predict future failures.</dd>"
                "<dt>Shape parameter (β)</dt>"
                "<dd>β &lt; 1: Decreasing failure rate (infant mortality — burn-in helps). "
                "β = 1: Constant failure rate (random failures — no wear-out). "
                "β &gt; 1: Increasing failure rate (wear-out — scheduled replacement helps).</dd>"
                "<dt>Scale parameter (η)</dt>"
                "<dd>The characteristic life — 63.2% of units fail by time η. It sets the "
                "time scale of the failure process.</dd>"
                "<dt>Censored data handling</dt>"
                "<dd>Units still running (right-censored) provide partial information — they "
                "survived at least that long. Ignoring them biases results toward shorter lives.</dd>"
                "</dl>"
            ),
        },
        ("reliability", "lognormal"): {
            "title": "Understanding Lognormal Reliability",
            "content": (
                "<dl>"
                "<dt>When to use lognormal?</dt>"
                "<dd>When failure times are right-skewed and log(time) is approximately normal. "
                "Common for fatigue, crack growth, and degradation processes where failure "
                "results from multiplicative damage accumulation.</dd>"
                "<dt>Parameters</dt>"
                "<dd><strong>μ</strong>: Mean of log(time) — sets the median life. "
                "<strong>σ</strong>: Std dev of log(time) — controls the spread of failure times.</dd>"
                "<dt>Lognormal vs Weibull</dt>"
                "<dd>Lognormal has a non-monotonic hazard (increases then decreases). Weibull "
                "has a monotonic hazard. Use probability plots and goodness-of-fit tests to "
                "choose between them.</dd>"
                "<dt>Interpretation</dt>"
                "<dd>The median life (e^μ) is the time by which 50% of units fail. The spread "
                "σ determines how much variation exists in failure times.</dd>"
                "</dl>"
            ),
        },
        ("reliability", "exponential"): {
            "title": "Understanding Exponential Reliability",
            "content": (
                "<dl>"
                "<dt>When to use exponential?</dt>"
                "<dd>When the failure rate is constant over time — failures are purely random "
                "with no wear-out or break-in pattern. Equivalent to Weibull with β = 1.</dd>"
                "<dt>The memoryless property</dt>"
                "<dd>A used item has the same remaining life distribution as a new one. This is "
                "only realistic for electronic components and some truly random failure modes.</dd>"
                "<dt>MTBF</dt>"
                "<dd>Mean Time Between Failures = 1/λ (failure rate). The single parameter of "
                "the exponential distribution. MTBF is the expected life, not the median.</dd>"
                "<dt>Caution</dt>"
                "<dd>The constant failure rate assumption is rarely true for mechanical systems. "
                "Always verify with a probability plot or compare against Weibull before "
                "defaulting to exponential.</dd>"
                "</dl>"
            ),
        },
        ("reliability", "kaplan_meier"): {
            "title": "Understanding Kaplan-Meier Reliability",
            "content": (
                "<dl>"
                "<dt>What is Kaplan-Meier in reliability?</dt>"
                "<dd>A nonparametric reliability estimate — no distributional assumption needed. "
                "Estimates the survival function directly from failure and censoring times.</dd>"
                "<dt>Advantages</dt>"
                "<dd>No need to choose Weibull vs lognormal vs exponential. The KM estimator "
                "lets the data speak. Especially valuable when the failure distribution is "
                "unknown or complex.</dd>"
                "<dt>Comparison of populations</dt>"
                "<dd>Log-rank test compares survival curves between groups (e.g., two suppliers, "
                "two designs). Does not assume a specific distribution.</dd>"
                "<dt>Limitations</dt>"
                "<dd>Cannot extrapolate beyond the data range. For prediction at time horizons "
                "beyond the study, parametric models (Weibull, lognormal) are needed.</dd>"
                "</dl>"
            ),
        },
        ("reliability", "reliability_test_plan"): {
            "title": "Understanding Reliability Test Planning",
            "content": (
                "<dl>"
                "<dt>What is a reliability test plan?</dt>"
                "<dd>Determines sample size and test duration needed to demonstrate a reliability "
                "target with specified confidence. Balances testing cost against evidence quality.</dd>"
                "<dt>Test-to-failure vs time-terminated</dt>"
                "<dd>Test-to-failure: Run until all units fail (most information per unit). "
                "Time-terminated: Stop at a fixed time (known schedule, but censored data).</dd>"
                "<dt>Demonstration testing</dt>"
                "<dd>Tests n units for time T with no failures to demonstrate reliability R "
                "with confidence C. The formula: n = log(1−C) / log(R) (for exponential).</dd>"
                "<dt>Acceleration</dt>"
                "<dd>Run at elevated stress to cause failures faster. An acceleration factor "
                "relates test time to field time. Requires validated acceleration models.</dd>"
                "</dl>"
            ),
        },
        ("reliability", "distribution_id"): {
            "title": "Understanding Reliability Distribution Identification",
            "content": (
                "<dl>"
                "<dt>What is distribution identification?</dt>"
                "<dd>Systematically compares how well different lifetime distributions (Weibull, "
                "lognormal, exponential, normal, loglogistic) fit your failure data. Selects "
                "the best-fitting model for prediction.</dd>"
                "<dt>How to compare</dt>"
                "<dd>Probability plots (linearity indicates good fit), Anderson-Darling statistics "
                "(lower = better), and AIC/BIC for model comparison.</dd>"
                "<dt>Why does the distribution matter?</dt>"
                "<dd>Different distributions imply different failure patterns. Weibull with β &gt; 1 "
                "means scheduled replacement helps. Lognormal means median life is the key metric. "
                "Wrong distribution → wrong decisions.</dd>"
                "<dt>Multiple failure modes</dt>"
                "<dd>If data doesn't fit any single distribution well, consider mixture models — "
                "multiple failure mechanisms may be operating simultaneously.</dd>"
                "</dl>"
            ),
        },
        ("reliability", "accelerated_life"): {
            "title": "Understanding Accelerated Life Testing",
            "content": (
                "<dl>"
                "<dt>What is ALT?</dt>"
                "<dd>Accelerated Life Testing applies elevated stress (temperature, voltage, "
                "load) to induce failures faster, then extrapolates to normal operating "
                "conditions using physics-based models.</dd>"
                "<dt>Acceleration models</dt>"
                "<dd><strong>Arrhenius</strong>: Temperature-accelerated (chemical degradation). "
                "<strong>Inverse Power Law</strong>: Voltage/load-accelerated. "
                "<strong>Eyring</strong>: Temperature + humidity combined.</dd>"
                "<dt>Key assumption</dt>"
                "<dd>The failure mechanism must be the same at elevated and normal stress. If "
                "higher stress triggers a different failure mode, extrapolation is invalid.</dd>"
                "<dt>Planning</dt>"
                "<dd>At least 3 stress levels with adequate failures at each. The lowest stress "
                "should be close to normal use conditions for reliable extrapolation.</dd>"
                "</dl>"
            ),
        },
        ("reliability", "repairable_systems"): {
            "title": "Understanding Repairable Systems Analysis",
            "content": (
                "<dl>"
                "<dt>What are repairable systems?</dt>"
                "<dd>Systems that can be restored to operation after failure (unlike components "
                "that are replaced). Analysis focuses on failure patterns over the system's "
                "life — are failures getting more or less frequent?</dd>"
                "<dt>NHPP models</dt>"
                "<dd>Non-Homogeneous Poisson Process models capture changing failure rates. "
                "The power law (Crow-AMSAA) model is most common: m(t) = λ·t^β.</dd>"
                "<dt>Interpreting β</dt>"
                "<dd>β &lt; 1: Reliability growth (failures decreasing). β = 1: Constant rate. "
                "β &gt; 1: Reliability deterioration (failures increasing — system wearing out).</dd>"
                "<dt>MCF (Mean Cumulative Function)</dt>"
                "<dd>The nonparametric alternative — plots cumulative failures vs time. "
                "Concave = improving. Straight = constant. Convex = deteriorating.</dd>"
                "</dl>"
            ),
        },
        ("reliability", "warranty"): {
            "title": "Understanding Warranty Analysis",
            "content": (
                "<dl>"
                "<dt>What is warranty analysis?</dt>"
                "<dd>Predicts future warranty claims from current claim data and sales history. "
                "Accounts for reporting delays, IBNR (Incurred But Not Reported) claims, "
                "and fleet age distributions.</dd>"
                "<dt>The warranty data challenge</dt>"
                "<dd>Not all failures are reported during the warranty period — some occur "
                "after warranty expires, some aren't claimed. Analysis must adjust for these "
                "truncation and censoring effects.</dd>"
                "<dt>Forecasting</dt>"
                "<dd>Fit a lifetime distribution to warranty claims, estimate the total failure "
                "fraction, and project future claims by birth month (production cohort).</dd>"
                "<dt>Cost estimation</dt>"
                "<dd>Claims × average repair cost × remaining fleet exposure = financial reserve. "
                "This drives warranty accrual accounting and pricing decisions.</dd>"
                "</dl>"
            ),
        },
        ("reliability", "competing_risks"): {
            "title": "Understanding Competing Risks Analysis",
            "content": (
                "<dl>"
                "<dt>What are competing risks?</dt>"
                "<dd>When a unit can fail from multiple independent failure modes, and the "
                "first one to occur is what we observe. Each mode competes to be the cause "
                "of failure.</dd>"
                "<dt>Why separate analysis?</dt>"
                "<dd>Standard reliability analysis ignores the failure mode. Competing risks "
                "decomposes the overall reliability into mode-specific reliabilities, enabling "
                "targeted improvements.</dd>"
                "<dt>Cumulative incidence function</dt>"
                "<dd>The probability of failing from a specific mode by time t, accounting for "
                "the competing modes. Unlike mode-specific KM, it correctly accounts for the "
                "competition.</dd>"
                "<dt>Improvement prioritization</dt>"
                "<dd>Eliminating the dominant failure mode reveals the next mode's contribution. "
                "This guides R&D investment to the highest-impact improvement.</dd>"
                "</dl>"
            ),
        },
    }
)


# ═══════════════════════════════════════════════════════════════════════════
# Visualization Module
# ═══════════════════════════════════════════════════════════════════════════

_extend(
    {
        ("viz", "histogram"): {
            "title": "Understanding Histograms",
            "content": (
                "<dl>"
                "<dt>What is a histogram?</dt>"
                "<dd>A bar chart showing the distribution of continuous data by dividing values "
                "into bins and counting how many fall in each. The most fundamental tool for "
                "understanding data shape.</dd>"
                "<dt>Bin width matters</dt>"
                "<dd>Too few bins hide structure; too many create noise. The Freedman-Diaconis "
                "rule (bin width = 2·IQR·n^(−1/3)) usually works well.</dd>"
                "<dt>What to look for</dt>"
                "<dd>Shape (symmetric, skewed, bimodal), center (mean/median), spread, outliers, "
                "and gaps (which may indicate mixed populations).</dd>"
                "<dt>Histogram vs density plot</dt>"
                "<dd>Histograms are discrete (binned); density plots are smooth. Density plots "
                "avoid the bin width problem but are harder to interpret for small samples.</dd>"
                "</dl>"
            ),
        },
        ("viz", "boxplot"): {
            "title": "Understanding Boxplots",
            "content": (
                "<dl>"
                "<dt>What is a boxplot?</dt>"
                "<dd>A five-number summary visualization: minimum, Q1, median, Q3, maximum. "
                "The box spans the IQR; whiskers extend to the most extreme non-outlier values.</dd>"
                "<dt>Reading outliers</dt>"
                "<dd>Points beyond 1.5×IQR from the box are marked as outliers. They may be "
                "data errors, special causes, or genuine extreme values — always investigate.</dd>"
                "<dt>Comparing groups</dt>"
                "<dd>Side-by-side boxplots compare distributions across groups. Non-overlapping "
                "boxes suggest a significant difference. Overlapping medians with different "
                "spreads indicate a variance difference.</dd>"
                "<dt>Limitations</dt>"
                "<dd>Boxplots hide multimodality and detailed shape information. For small samples "
                "or complex distributions, add individual data points or use violin plots.</dd>"
                "</dl>"
            ),
        },
        ("viz", "scatter"): {
            "title": "Understanding Scatter Plots",
            "content": (
                "<dl>"
                "<dt>What is a scatter plot?</dt>"
                "<dd>Plots two continuous variables against each other to reveal relationships, "
                "clusters, outliers, and patterns.</dd>"
                "<dt>What to look for</dt>"
                "<dd><strong>Direction</strong>: Positive (both increase) or negative (one decreases). "
                "<strong>Strength</strong>: How tightly points cluster around a pattern. "
                "<strong>Form</strong>: Linear, curved, or no pattern. "
                "<strong>Outliers</strong>: Points far from the main cloud.</dd>"
                "<dt>Adding regression lines</dt>"
                "<dd>A trend line quantifies the relationship. But always check the scatter first — "
                "a regression line through a nonlinear pattern or clustered data is misleading.</dd>"
                "<dt>Overplotting</dt>"
                "<dd>With many points, individual values obscure each other. Use transparency, "
                "jitter, or 2D density (hexbin) plots for large datasets.</dd>"
                "</dl>"
            ),
        },
        ("viz", "heatmap"): {
            "title": "Understanding Heatmaps",
            "content": (
                "<dl>"
                "<dt>What is a heatmap?</dt>"
                "<dd>A matrix of values encoded as colors. Reveals patterns in two-dimensional "
                "data — correlations, clusters, and anomalies become visible as color patterns.</dd>"
                "<dt>Color scale</dt>"
                "<dd>Sequential (low-to-high) for magnitude. Diverging (two colors meeting at "
                "a midpoint) for deviations from a reference. Choose colorblind-safe palettes.</dd>"
                "<dt>Correlation heatmaps</dt>"
                "<dd>Show pairwise correlations between all variables. Hot spots indicate strong "
                "relationships. Blocks of correlated variables suggest underlying factors.</dd>"
                "<dt>Ordering matters</dt>"
                "<dd>Clustering rows and columns (dendrogram) groups similar items together, "
                "making patterns much easier to spot than random ordering.</dd>"
                "</dl>"
            ),
        },
        ("viz", "pareto"): {
            "title": "Understanding Pareto Charts",
            "content": (
                "<dl>"
                "<dt>What is a Pareto chart?</dt>"
                "<dd>A bar chart ordered by frequency plus a cumulative percentage line. Based "
                "on the Pareto principle: ~80% of effects come from ~20% of causes.</dd>"
                "<dt>How to read it</dt>"
                "<dd>Bars show individual category counts. The line shows cumulative percentage. "
                "Focus improvement efforts on the vital few (leftmost bars) that account for "
                "most of the total.</dd>"
                "<dt>Before/after comparison</dt>"
                "<dd>Create a Pareto chart before and after improvement. The rank order may "
                "change — previous top causes may drop, revealing new priorities.</dd>"
                "<dt>Stratified Pareto</dt>"
                "<dd>Break down by shift, machine, operator, or time period to find the root "
                "cause behind the top Pareto category.</dd>"
                "</dl>"
            ),
        },
        ("viz", "matrix"): {
            "title": "Understanding Matrix Plots",
            "content": (
                "<dl>"
                "<dt>What is a matrix plot?</dt>"
                "<dd>A grid of scatter plots for all pairs of variables. The fastest way to "
                "explore pairwise relationships in multivariate data.</dd>"
                "<dt>What to look for</dt>"
                "<dd>Strong linear patterns (potential predictors), clusters (subpopulations), "
                "outliers (unusual observations), and nonlinear relationships.</dd>"
                "<dt>Diagonal</dt>"
                "<dd>The diagonal shows each variable's own distribution (histogram or density). "
                "This combines univariate and bivariate exploration in one display.</dd>"
                "<dt>Practical use</dt>"
                "<dd>First step in regression analysis — identifies which predictors have "
                "relationships with the response and with each other (multicollinearity).</dd>"
                "</dl>"
            ),
        },
        ("viz", "timeseries"): {
            "title": "Understanding Time Series Plots",
            "content": (
                "<dl>"
                "<dt>What is a time series plot?</dt>"
                "<dd>Data plotted in time order — the most natural way to see process behavior "
                "over time. Reveals trends, cycles, shifts, and outliers.</dd>"
                "<dt>What to look for</dt>"
                "<dd>Trends (sustained increase/decrease), seasonality (repeating patterns), "
                "level shifts (sudden jumps), and increasing/decreasing spread.</dd>"
                "<dt>Vs control charts</dt>"
                "<dd>Time series plots show raw data. Control charts add statistical limits "
                "that distinguish common from special cause variation. Use time series for "
                "exploration, control charts for monitoring.</dd>"
                "<dt>Multiple series</dt>"
                "<dd>Overlay related variables to spot correlations, leads/lags, or common "
                "patterns. Normalize scales if comparing variables with different units.</dd>"
                "</dl>"
            ),
        },
        ("viz", "probability"): {
            "title": "Understanding Probability Plots",
            "content": (
                "<dl>"
                "<dt>What is a probability plot?</dt>"
                "<dd>Plots ordered data against theoretical quantiles from a reference "
                "distribution. If the data follows that distribution, the points form a "
                "straight line.</dd>"
                "<dt>Reading the plot</dt>"
                "<dd>Straight line = data matches the distribution. S-curve = heavy tails. "
                "Concave/convex departures = skewness. Gaps or clusters indicate data quality "
                "issues or mixed populations.</dd>"
                "<dt>Normal probability plot (Q-Q plot)</dt>"
                "<dd>The most common variant. Points above the line at the right end indicate "
                "right skew (heavy upper tail). Points below at both ends indicate light tails.</dd>"
                "<dt>Beyond normality</dt>"
                "<dd>Probability plots can use any reference distribution — Weibull, lognormal, "
                "exponential. The distribution giving the straightest line is the best fit.</dd>"
                "</dl>"
            ),
        },
        ("viz", "individual_value_plot"): {
            "title": "Understanding Individual Value Plots",
            "content": (
                "<dl>"
                "<dt>What is an individual value plot?</dt>"
                "<dd>Shows every data point for each group, with the mean and/or median marked. "
                "Unlike boxplots, no information is hidden — you see all the data.</dd>"
                "<dt>When to use</dt>"
                "<dd>Small to moderate samples where every observation matters. Boxplots can "
                "hide important patterns (bimodality, gaps) in small datasets.</dd>"
                "<dt>Jittering</dt>"
                "<dd>Points at the same value are offset horizontally to prevent overplotting. "
                "This reveals the density of the data at each value.</dd>"
                "<dt>Group comparison</dt>"
                "<dd>Side-by-side individual value plots show both the central tendency and "
                "the full spread of each group, making it easy to spot asymmetry and outliers.</dd>"
                "</dl>"
            ),
        },
        ("viz", "interval_plot"): {
            "title": "Understanding Interval Plots",
            "content": (
                "<dl>"
                "<dt>What is an interval plot?</dt>"
                "<dd>Shows the mean and confidence interval for each group. A clean visualization "
                "for comparing group means with their precision.</dd>"
                "<dt>Reading the intervals</dt>"
                "<dd>Non-overlapping intervals suggest a significant difference. However, "
                "overlapping intervals do NOT guarantee no difference — formal testing "
                "(ANOVA, t-test) is needed.</dd>"
                "<dt>CI width</dt>"
                "<dd>Wider intervals indicate less precision (smaller samples, more variability). "
                "Narrower intervals indicate more precise mean estimates.</dd>"
                "<dt>Best paired with</dt>"
                "<dd>Individual value plots (to see the raw data) and ANOVA/t-test results "
                "(for formal significance). Interval plots alone can be misleading.</dd>"
                "</dl>"
            ),
        },
        ("viz", "dotplot"): {
            "title": "Understanding Dot Plots",
            "content": (
                "<dl>"
                "<dt>What is a dot plot?</dt>"
                "<dd>Each observation is a dot, stacked vertically at its value. Shows the "
                "distribution shape without binning (unlike histograms). Best for small to "
                "moderate samples.</dd>"
                "<dt>Dot plot vs histogram</dt>"
                "<dd>Dot plots show every value exactly — no information loss from binning. "
                "Histograms are better for large datasets where individual points would overlap.</dd>"
                "<dt>Grouped dot plots</dt>"
                "<dd>Side-by-side grouped dot plots compare distributions across categories. "
                "Each dot is visible, making outliers and patterns immediately apparent.</dd>"
                "<dt>When to use</dt>"
                "<dd>Exploratory analysis with n &lt; 100, comparing a few groups, or when "
                "exact values matter (e.g., measurement data where rounding patterns "
                "reveal gage resolution issues).</dd>"
                "</dl>"
            ),
        },
        ("viz", "bubble"): {
            "title": "Understanding Bubble Charts",
            "content": (
                "<dl>"
                "<dt>What is a bubble chart?</dt>"
                "<dd>A scatter plot where a third variable is encoded as bubble size. Shows "
                "three dimensions of data in a 2D space.</dd>"
                "<dt>When to use</dt>"
                "<dd>When you want to visualize three continuous variables simultaneously — "
                "e.g., defect rate (x) vs cost (y) vs production volume (size).</dd>"
                "<dt>Perception issues</dt>"
                "<dd>Humans perceive area poorly. Encode the most important variable on the "
                "axes, not as size. Size should represent a secondary variable.</dd>"
                "<dt>Alternative</dt>"
                "<dd>For more than 3 variables, consider parallel coordinates or matrix plots "
                "instead of trying to encode too much in a single chart.</dd>"
                "</dl>"
            ),
        },
        ("viz", "parallel_coordinates"): {
            "title": "Understanding Parallel Coordinates",
            "content": (
                "<dl>"
                "<dt>What are parallel coordinates?</dt>"
                "<dd>Each variable gets a vertical axis; each observation is a line connecting "
                "its values across axes. Reveals multivariate patterns, clusters, and outliers "
                "in high-dimensional data.</dd>"
                "<dt>What to look for</dt>"
                "<dd>Lines that cross: negative correlation between adjacent axes. Lines that "
                "don't cross: positive correlation. Clusters of parallel lines: groups with "
                "similar profiles.</dd>"
                "<dt>Axis ordering</dt>"
                "<dd>The order of axes affects readability. Place related variables adjacent. "
                "Reordering can reveal hidden patterns.</dd>"
                "<dt>Use with clustering</dt>"
                "<dd>Color-code lines by cluster membership to see how clusters differ across "
                "all variables simultaneously.</dd>"
                "</dl>"
            ),
        },
        ("viz", "contour"): {
            "title": "Understanding Contour Plots",
            "content": (
                "<dl>"
                "<dt>What is a contour plot?</dt>"
                "<dd>Shows a 3D surface as 2D lines of equal value (like a topographic map). "
                "Essential for visualizing response surfaces in DOE.</dd>"
                "<dt>Reading contours</dt>"
                "<dd>Closely spaced lines = steep surface (rapid change). Widely spaced = flat "
                "region. Closed loops = peaks or valleys (optimal regions).</dd>"
                "<dt>In DOE context</dt>"
                "<dd>Contour plots show how two factors jointly affect a response. The optimal "
                "region (target contour) guides factor settings for process optimization.</dd>"
                "<dt>Filled vs line contours</dt>"
                "<dd>Filled contours use color gradients for easier reading. Line contours "
                "are better for overlay with data points.</dd>"
                "</dl>"
            ),
        },
        ("viz", "surface_3d"): {
            "title": "Understanding 3D Surface Plots",
            "content": (
                "<dl>"
                "<dt>What is a 3D surface plot?</dt>"
                "<dd>A three-dimensional visualization of how a response varies across two "
                "factors. Useful for seeing the overall shape of a response surface.</dd>"
                "<dt>When to use</dt>"
                "<dd>DOE response surface exploration, regression model visualization, "
                "probability density visualization. Best for presentations — contour plots "
                "are often more precise for analysis.</dd>"
                "<dt>Interaction</dt>"
                "<dd>Rotate the surface interactively to find the best viewing angle. "
                "Saddle shapes indicate interactions between factors.</dd>"
                "<dt>Limitations</dt>"
                "<dd>Can hide detail in the back. Perspective distorts distances. Always "
                "supplement with a contour plot for precise interpretation.</dd>"
                "</dl>"
            ),
        },
        ("viz", "contour_overlay"): {
            "title": "Understanding Contour Overlay Plots",
            "content": (
                "<dl>"
                "<dt>What is a contour overlay?</dt>"
                "<dd>Overlays contour plots for multiple responses on the same axes to find "
                "factor settings that satisfy all requirements simultaneously.</dd>"
                "<dt>The feasible region</dt>"
                "<dd>The area where all response contours meet their specifications. If no "
                "feasible region exists, the specs are mutually incompatible.</dd>"
                "<dt>Use in optimization</dt>"
                "<dd>After fitting response surfaces in DOE, the overlay plot shows the "
                "operating window — the range of factor settings that produce acceptable "
                "results on all responses.</dd>"
                "<dt>Multi-response optimization</dt>"
                "<dd>The desirability function approach quantifies how well each point "
                "satisfies all requirements. The overlay plot visualizes this trade-off.</dd>"
                "</dl>"
            ),
        },
        ("viz", "mosaic"): {
            "title": "Understanding Mosaic Plots",
            "content": (
                "<dl>"
                "<dt>What is a mosaic plot?</dt>"
                "<dd>A visual representation of a contingency table where tile areas are "
                "proportional to cell frequencies. Shows both marginal and conditional "
                "distributions simultaneously.</dd>"
                "<dt>Reading the plot</dt>"
                "<dd>Column widths show marginal proportions of one variable. Row heights "
                "within each column show conditional proportions of the other. Equal heights "
                "across columns = independence.</dd>"
                "<dt>Residual shading</dt>"
                "<dd>Tiles colored by Pearson residuals highlight cells that deviate from "
                "independence. Blue = more than expected. Red = fewer than expected.</dd>"
                "<dt>When to use</dt>"
                "<dd>Visualizing categorical associations before or alongside chi-square tests. "
                "Reveals which specific cells drive the overall association.</dd>"
                "</dl>"
            ),
        },
        ("viz", "bayes_spc_capability"): {
            "title": "Understanding Bayesian SPC Capability Visualization",
            "content": (
                "<dl>"
                "<dt>What does this show?</dt>"
                "<dd>Bayesian posterior distribution of capability indices (Cpk, Ppk) rather "
                "than single point estimates. Shows the full uncertainty in capability assessment.</dd>"
                "<dt>Why Bayesian?</dt>"
                "<dd>Traditional Cpk is a point estimate — it hides uncertainty. The posterior "
                "distribution shows the probability that true Cpk exceeds any threshold.</dd>"
                "<dt>Decision-making</dt>"
                "<dd>P(Cpk &gt; 1.33) = 0.95 is a much stronger statement than 'estimated Cpk = 1.5' "
                "because it accounts for sample size and estimation uncertainty.</dd>"
                "<dt>Prior influence</dt>"
                "<dd>With small samples, the prior matters. The analysis shows how the posterior "
                "shifts from prior to posterior as data accumulates.</dd>"
                "</dl>"
            ),
        },
        ("viz", "bayes_spc_changepoint"): {
            "title": "Understanding Bayesian SPC Changepoint Visualization",
            "content": (
                "<dl>"
                "<dt>What does this show?</dt>"
                "<dd>Posterior probability of a changepoint at each time position. Shows where "
                "the process most likely shifted, with uncertainty quantification.</dd>"
                "<dt>Posterior probability trace</dt>"
                "<dd>The height at each time point shows the probability that a change occurred "
                "there. Peaks indicate likely changepoints. Multiple peaks suggest multiple shifts.</dd>"
                "<dt>Vs classical changepoint</dt>"
                "<dd>Classical methods give point estimates. Bayesian methods give full "
                "probability distributions over changepoint locations and magnitudes.</dd>"
                "<dt>Segment analysis</dt>"
                "<dd>The data is partitioned at the most probable changepoint(s), and separate "
                "statistics are computed for each segment.</dd>"
                "</dl>"
            ),
        },
        ("viz", "bayes_spc_control"): {
            "title": "Understanding Bayesian SPC Control Visualization",
            "content": (
                "<dl>"
                "<dt>What does this show?</dt>"
                "<dd>Bayesian control limits derived from posterior predictive distributions "
                "rather than classical ±3σ limits. Valid for non-normal data and small samples.</dd>"
                "<dt>Predictive limits</dt>"
                "<dd>Based on where future observations are expected to fall given the data "
                "seen so far. They contract as more data arrives (uncertainty decreases).</dd>"
                "<dt>Advantages</dt>"
                "<dd>No normality assumption needed. Limits are honest about uncertainty with "
                "small samples — they start wide and narrow as evidence accumulates.</dd>"
                "<dt>Posterior predictive check</dt>"
                "<dd>Compares the observed data distribution against the posterior predictive "
                "distribution. Discrepancies indicate model misfit.</dd>"
                "</dl>"
            ),
        },
        ("viz", "bayes_spc_acceptance"): {
            "title": "Understanding Bayesian SPC Acceptance Visualization",
            "content": (
                "<dl>"
                "<dt>What does this show?</dt>"
                "<dd>Bayesian acceptance probability — the posterior probability that a lot "
                "meets quality requirements, given the sample data.</dd>"
                "<dt>Vs classical acceptance sampling</dt>"
                "<dd>Classical: accept if defects ≤ c (hard boundary). Bayesian: compute "
                "P(lot meets spec | data) and decide based on the probability.</dd>"
                "<dt>Prior information</dt>"
                "<dd>Bayesian acceptance can incorporate prior knowledge about the supplier's "
                "historical quality, updating as new lot data arrives.</dd>"
                "<dt>Risk quantification</dt>"
                "<dd>Directly computes the probability of accepting a bad lot (consumer's risk) "
                "and rejecting a good lot (producer's risk) for any decision threshold.</dd>"
                "</dl>"
            ),
        },
    }
)


# ═══════════════════════════════════════════════════════════════════════════
# Simulation Module
# ═══════════════════════════════════════════════════════════════════════════

_extend(
    {
        ("simulation", "tolerance_stackup"): {
            "title": "Understanding Tolerance Stackup Analysis",
            "content": (
                "<dl>"
                "<dt>What is tolerance stackup?</dt>"
                "<dd>Predicts the variation of an assembly dimension from the tolerances of "
                "its component parts. Answers: 'If each part is within spec, will the "
                "assembly be within spec?'</dd>"
                "<dt>Worst case vs RSS vs Monte Carlo</dt>"
                "<dd><strong>Worst case</strong>: Assumes all parts at their limits simultaneously "
                "(very conservative). <strong>RSS</strong>: Root Sum of Squares — assumes "
                "independent, centered, normal distributions. <strong>Monte Carlo</strong>: "
                "Simulates actual distributions (most realistic).</dd>"
                "<dt>Why Monte Carlo?</dt>"
                "<dd>Real parts aren't perfectly centered or normally distributed. Monte Carlo "
                "uses actual or fitted distributions, handling skewness, truncation, and "
                "correlation between dimensions.</dd>"
                "<dt>What-if exploration</dt>"
                "<dd>Adjust component tolerances to see which ones most affect assembly variation. "
                "This guides tolerance allocation — tighten where it matters, loosen where it doesn't.</dd>"
                "</dl>"
            ),
        },
        ("simulation", "variance_propagation"): {
            "title": "Understanding Variance Propagation",
            "content": (
                "<dl>"
                "<dt>What is variance propagation?</dt>"
                "<dd>Estimates how input variable uncertainties combine to affect an output "
                "through a mathematical model. Based on the delta method (Taylor series "
                "approximation) or Monte Carlo simulation.</dd>"
                "<dt>When to use</dt>"
                "<dd>When you have a transfer function Y = f(X₁, X₂, ...) and know the "
                "distributions of the inputs. Determines the output distribution without "
                "running physical experiments.</dd>"
                "<dt>Sensitivity analysis</dt>"
                "<dd>Partial derivatives (or Monte Carlo sensitivities) show which inputs "
                "contribute most to output variance. Focus improvement on the dominant "
                "contributors.</dd>"
                "<dt>Delta method vs Monte Carlo</dt>"
                "<dd>Delta method is fast but assumes linearity and normality. Monte Carlo "
                "handles any function and distribution but needs more computation.</dd>"
                "</dl>"
            ),
        },
    }
)


# ═══════════════════════════════════════════════════════════════════════════
# D-Type Module (centralized copies of existing education)
# ═══════════════════════════════════════════════════════════════════════════

_extend(
    {
        ("d_type", "d_chart"): {
            "title": "Understanding the D-Chart",
            "content": (
                "<dl>"
                "<dt>What is a D-Chart?</dt>"
                "<dd>A Divergence Chart monitors how much each factor level's distribution "
                "differs from the overall process distribution over time. Unlike traditional "
                "control charts that track means or ranges, it tracks <em>distributional shape</em> "
                "changes — catching shifts in spread, skew, or tails that mean/range charts miss.</dd>"
                "<dt>What is JSD (Jensen-Shannon Divergence)?</dt>"
                "<dd>A symmetric, bounded measure of how different two probability distributions "
                "are. JSD = 0 means identical; JSD = 1 (in bits) means completely different.</dd>"
                "<dt>What is the Noise Floor?</dt>"
                "<dd>The expected JSD from random sampling alone. Points above the noise floor "
                "indicate real, non-random divergence.</dd>"
                "<dt>What is the Information Score?</dt>"
                "<dd>A cumulative, recency-weighted sum of excess JSD above noise floor. Higher "
                "scores mean the factor consistently produces a different distribution.</dd>"
                "</dl>"
            ),
        },
        ("d_type", "d_cpk"): {
            "title": "Understanding D-Cpk",
            "content": (
                "<dl>"
                "<dt>What is D-Cpk?</dt>"
                "<dd>Attributes capability differences to specific factor levels. Standard Cpk "
                "tells you <em>if</em> the process is capable. D-Cpk tells you <em>which factors "
                "are dragging capability down</em>.</dd>"
                "<dt>Defect Efficiency</dt>"
                "<dd>The fraction of a factor's divergence that occurs in spec tails (where defects "
                "happen). High efficiency means the factor directly creates defects.</dd>"
                "<dt>Counterfactual Cpk</dt>"
                "<dd>The Cpk the process would have if that factor level were removed. The gap "
                "prioritizes improvement actions.</dd>"
                "<dt>Interpretation</dt>"
                "<dd>Factor above noise with high defect efficiency → priority target. Factor above "
                "noise with low defect efficiency → changes distribution but not defect rate.</dd>"
                "</dl>"
            ),
        },
        ("d_type", "d_nonnorm"): {
            "title": "Understanding D-NonNorm",
            "content": (
                "<dl>"
                "<dt>Why non-normal capability?</dt>"
                "<dd>Traditional Cpk assumes normality. Many real processes are skewed or bounded. "
                "KDE-based Ppk gives an honest capability estimate without distributional assumptions.</dd>"
                "<dt>Normality Penalty</dt>"
                "<dd>The difference between normal-assumption Ppk and KDE-based Ppk. Positive "
                "penalty means normal overstates capability.</dd>"
                "<dt>PPM comparison</dt>"
                "<dd>Compare KDE vs Normal PPM to see the real-world defect rate difference.</dd>"
                "<dt>When to use</dt>"
                "<dd>Any time you suspect non-normality. Always check the normality penalty — "
                "if it's large, normal-based capability is misleading.</dd>"
                "</dl>"
            ),
        },
        ("d_type", "d_equiv"): {
            "title": "Understanding D-Equiv",
            "content": (
                "<dl>"
                "<dt>What does D-Equiv test?</dt>"
                "<dd>Whether batches produce the same distribution of output — not just the same "
                "mean. Catches differences in spread, shape, and tails via JSD comparison.</dd>"
                "<dt>How is equivalence decided?</dt>"
                "<dd>Each batch's density is compared to a reference via JSD. Below the noise "
                "floor threshold = equivalent.</dd>"
                "<dt>Pairwise JSD heatmap</dt>"
                "<dd>Shows which batches are similar (cool colors) vs different (hot colors). "
                "Clusters of similar batches may indicate shared process conditions.</dd>"
                "<dt>Reference batch</dt>"
                "<dd>The 'known good' batch. All others are compared against it.</dd>"
                "</dl>"
            ),
        },
        ("d_type", "d_sig"): {
            "title": "Understanding D-Sig (Process Signatures)",
            "content": (
                "<dl>"
                "<dt>What is a Process Signature?</dt>"
                "<dd>A time-ordered measurement profile (e.g., temperature over a cycle). D-Sig "
                "compares profiles across groups to find where and how they diverge.</dd>"
                "<dt>Peak Divergence</dt>"
                "<dd>The time point where a group's distribution differs most from the reference — "
                "where to focus investigation.</dd>"
                "<dt>Interpretation</dt>"
                "<dd>Flat, low JSD trace: signatures match. Spike: localized divergence. "
                "Sustained elevation: fundamentally different process mode.</dd>"
                "<dt>Applications</dt>"
                "<dd>Batch process monitoring, forming press analysis, thermal profile comparison.</dd>"
                "</dl>"
            ),
        },
        ("d_type", "d_multi"): {
            "title": "Understanding D-Multi (Multivariate Capability)",
            "content": (
                "<dl>"
                "<dt>Why multivariate?</dt>"
                "<dd>Multiple correlated quality characteristics must be assessed jointly. A part "
                "can pass every individual spec but fail jointly.</dd>"
                "<dt>PCA and MCpk</dt>"
                "<dd>PCA rotates correlated variables into uncorrelated components. KDE-based "
                "capability on each PC gives MCpk — the minimum is the bottleneck.</dd>"
                "<dt>Hotelling's T²</dt>"
                "<dd>Multivariate distance from center. Points above UCL are multivariate "
                "outliers even if normal on individual variables.</dd>"
                "<dt>Interpreting MCpk</dt>"
                "<dd>≥ 1.33: Jointly capable. 1.0–1.33: Marginal — find the weak PC. "
                "&lt; 1.0: Not jointly capable.</dd>"
                "</dl>"
            ),
        },
    }
)


# ═══════════════════════════════════════════════════════════════════════════
# Bayesian Module (centralized copies — originals remain in bayesian.py)
# ═══════════════════════════════════════════════════════════════════════════

_extend(
    {
        ("bayesian", "bayes_regression"): {
            "title": "Understanding Bayesian Regression",
            "content": (
                "<dl>"
                "<dt>What is Bayesian regression?</dt>"
                "<dd>Regression that returns posterior distributions for each coefficient instead "
                "of point estimates. You get a full picture of uncertainty.</dd>"
                "<dt>Credible intervals</dt>"
                "<dd>A 95% credible interval means there is a 95% probability the true coefficient "
                "lies within that range — a direct probability statement.</dd>"
                "<dt>How do I know if a predictor matters?</dt>"
                "<dd>If the credible interval excludes zero, the predictor has a credible effect.</dd>"
                "<dt>Why Bayesian?</dt>"
                "<dd>Naturally handles small samples, avoids overfitting through regularizing "
                "priors, and gives probabilistic statements about parameters.</dd>"
                "</dl>"
            ),
        },
        ("bayesian", "bayes_ttest"): {
            "title": "Understanding the Bayesian t-Test",
            "content": (
                "<dl>"
                "<dt>What is a Bayesian t-test?</dt>"
                "<dd>Compares two groups using a Bayes Factor (BF₁₀) that quantifies evidence "
                "for a difference versus no difference.</dd>"
                "<dt>Bayes Factor interpretation</dt>"
                "<dd>BF₁₀ &gt; 3: moderate evidence for difference. BF₁₀ &gt; 10: strong. "
                "BF₁₀ &lt; ⅓: moderate evidence for no difference. Between ⅓–3: inconclusive.</dd>"
                "<dt>Cohen's d posterior</dt>"
                "<dd>The posterior distribution shows how uncertain we are about the standardized "
                "effect size.</dd>"
                "<dt>Why over classical?</dt>"
                "<dd>Classical t-tests cannot quantify evidence for the null. The Bayes Factor "
                "lets you say 'the data support no difference' with specific strength.</dd>"
                "</dl>"
            ),
        },
        ("bayesian", "bayes_ab"): {
            "title": "Understanding Bayesian A/B Testing",
            "content": (
                "<dl>"
                "<dt>What is Bayesian A/B testing?</dt>"
                "<dd>Compares two variants (A and B) using posterior distributions to compute "
                "the probability that one is better than the other.</dd>"
                "<dt>P(B &gt; A)</dt>"
                "<dd>The probability that variant B outperforms A. Unlike p-values, this is "
                "a direct probability statement that answers the decision-maker's question.</dd>"
                "<dt>Expected loss</dt>"
                "<dd>The expected cost of choosing the wrong variant. Even if P(B &gt; A) = 0.6, "
                "the expected loss may be negligible — making the decision low-risk.</dd>"
                "<dt>Stopping rules</dt>"
                "<dd>Bayesian testing allows peeking at results without inflating error rates. "
                "Stop when the expected loss is below your threshold.</dd>"
                "</dl>"
            ),
        },
        ("bayesian", "bayes_correlation"): {
            "title": "Understanding Bayesian Correlation",
            "content": (
                "<dl>"
                "<dt>What is Bayesian correlation?</dt>"
                "<dd>Estimates the correlation coefficient with a posterior distribution, giving "
                "both the best estimate and a credible interval.</dd>"
                "<dt>BF₁₀ for correlation</dt>"
                "<dd>Quantifies evidence for a non-zero correlation vs no relationship.</dd>"
                "<dt>Small sample advantage</dt>"
                "<dd>Especially useful with small samples where p-values are unreliable and "
                "you want honest uncertainty bounds.</dd>"
                "<dt>Evidence for independence</dt>"
                "<dd>BF₁₀ &lt; ⅓ provides positive evidence that there is no relationship — "
                "something classical tests can never tell you.</dd>"
                "</dl>"
            ),
        },
        ("bayesian", "bayes_anova"): {
            "title": "Understanding Bayesian ANOVA",
            "content": (
                "<dl>"
                "<dt>What is Bayesian ANOVA?</dt>"
                "<dd>Tests whether group means differ using Bayes Factors instead of p-values. "
                "Provides evidence for or against the null hypothesis of equal means.</dd>"
                "<dt>Inclusion Bayes Factor</dt>"
                "<dd>For each factor, the inclusion BF tells you how much the data support "
                "including that factor in the model vs excluding it.</dd>"
                "<dt>Posterior group means</dt>"
                "<dd>Each group mean gets a posterior distribution — showing the precision of "
                "the estimate for each group.</dd>"
                "<dt>Model comparison</dt>"
                "<dd>Compares all possible models (null, main effects, interaction) and "
                "reports their posterior probabilities.</dd>"
                "</dl>"
            ),
        },
        ("bayesian", "bayes_changepoint"): {
            "title": "Understanding Bayesian Changepoint Detection",
            "content": (
                "<dl>"
                "<dt>What is Bayesian changepoint detection?</dt>"
                "<dd>Identifies where a time series changed its statistical properties, with "
                "full posterior probability for each candidate changepoint location.</dd>"
                "<dt>Posterior probability</dt>"
                "<dd>Each time point gets a probability of being a changepoint. This quantifies "
                "uncertainty about <em>where</em> the change occurred, not just <em>if</em>.</dd>"
                "<dt>Multiple changepoints</dt>"
                "<dd>Can detect multiple changepoints simultaneously, with the posterior "
                "distribution of the number of changes.</dd>"
                "<dt>Applications</dt>"
                "<dd>Process monitoring, identifying regime changes, dating process shifts "
                "for root cause analysis.</dd>"
                "</dl>"
            ),
        },
        ("bayesian", "bayes_proportion"): {
            "title": "Understanding Bayesian Proportion Test",
            "content": (
                "<dl>"
                "<dt>What is a Bayesian proportion test?</dt>"
                "<dd>Estimates a proportion with a full posterior distribution (Beta distribution). "
                "The credible interval gives an honest range for the true proportion.</dd>"
                "<dt>Beta-Binomial model</dt>"
                "<dd>The conjugate prior (Beta) combined with binomial data gives an exact "
                "posterior. No normal approximation needed — works for any sample size.</dd>"
                "<dt>Comparison of proportions</dt>"
                "<dd>P(p₁ &gt; p₂) computed directly from the posteriors gives the probability "
                "that one proportion exceeds the other.</dd>"
                "<dt>Small sample advantage</dt>"
                "<dd>Classical proportion CIs are unreliable with small n or extreme proportions. "
                "The Bayesian interval is always valid.</dd>"
                "</dl>"
            ),
        },
        ("bayesian", "bayes_capability_prediction"): {
            "title": "Understanding Bayesian Capability Prediction",
            "content": (
                "<dl>"
                "<dt>What is Bayesian capability prediction?</dt>"
                "<dd>Estimates Cpk with full posterior uncertainty, then predicts future "
                "capability based on the current state of knowledge.</dd>"
                "<dt>P(Cpk &gt; target)</dt>"
                "<dd>The probability that true capability meets a specified target. More useful "
                "than a point estimate for decision-making.</dd>"
                "<dt>Predictive distribution</dt>"
                "<dd>Predicts where future individual measurements will fall, accounting for "
                "both process variation and parameter uncertainty.</dd>"
                "<dt>Sequential updating</dt>"
                "<dd>As new data arrives, the posterior updates — showing how capability "
                "knowledge improves with more samples.</dd>"
                "</dl>"
            ),
        },
        ("bayesian", "bayes_equivalence"): {
            "title": "Understanding Bayesian Equivalence Testing",
            "content": (
                "<dl>"
                "<dt>What is Bayesian equivalence testing?</dt>"
                "<dd>Tests whether two groups are equivalent within a practical margin using "
                "the posterior distribution of the difference, providing P(|δ| &lt; Δ).</dd>"
                "<dt>Vs TOST</dt>"
                "<dd>TOST gives a binary accept/reject. Bayesian equivalence gives the "
                "probability of equivalence — much more informative for decision-making.</dd>"
                "<dt>ROPE (Region of Practical Equivalence)</dt>"
                "<dd>The range of effect sizes considered practically equivalent to zero. "
                "The posterior probability within the ROPE determines equivalence.</dd>"
                "<dt>Applications</dt>"
                "<dd>Method comparison, supplier qualification, process transfer — any time "
                "you need to demonstrate that two things are equivalent.</dd>"
                "</dl>"
            ),
        },
        ("bayesian", "bayes_chi2"): {
            "title": "Understanding Bayesian Chi-Square Test",
            "content": (
                "<dl>"
                "<dt>What is a Bayesian chi-square test?</dt>"
                "<dd>Tests independence in contingency tables using Bayes Factors. Can provide "
                "evidence for independence (no association), unlike classical chi-square.</dd>"
                "<dt>Dirichlet-Multinomial model</dt>"
                "<dd>The conjugate model for categorical data. The posterior over cell "
                "probabilities is a Dirichlet distribution.</dd>"
                "<dt>BF interpretation</dt>"
                "<dd>BF₁₀ &gt; 3: evidence for association. BF₁₀ &lt; ⅓: evidence for independence.</dd>"
                "<dt>Small expected counts</dt>"
                "<dd>Unlike classical chi-square, the Bayesian version works well with small "
                "expected counts — no minimum sample size requirements.</dd>"
                "</dl>"
            ),
        },
        ("bayesian", "bayes_poisson"): {
            "title": "Understanding Bayesian Poisson Analysis",
            "content": (
                "<dl>"
                "<dt>What is Bayesian Poisson analysis?</dt>"
                "<dd>Estimates count rates with posterior uncertainty using the Gamma-Poisson "
                "conjugate model. The posterior rate has a Gamma distribution.</dd>"
                "<dt>Rate comparison</dt>"
                "<dd>P(λ₁ &gt; λ₂) computed from posterior gives the probability that one "
                "rate exceeds another.</dd>"
                "<dt>Prediction</dt>"
                "<dd>The posterior predictive distribution (negative binomial) predicts future "
                "counts accounting for rate uncertainty.</dd>"
                "<dt>Overdispersion</dt>"
                "<dd>If counts are overdispersed, a Gamma-Poisson mixture (negative binomial) "
                "model is used automatically.</dd>"
                "</dl>"
            ),
        },
        ("bayesian", "bayes_logistic"): {
            "title": "Understanding Bayesian Logistic Regression",
            "content": (
                "<dl>"
                "<dt>What is Bayesian logistic regression?</dt>"
                "<dd>Models binary outcomes with posterior distributions over coefficients and "
                "predictions. Quantifies uncertainty in odds ratios and predicted probabilities.</dd>"
                "<dt>Posterior odds ratios</dt>"
                "<dd>Each predictor's odds ratio has a credible interval. If the interval "
                "excludes 1, the predictor has a credible effect on the odds.</dd>"
                "<dt>Prediction uncertainty</dt>"
                "<dd>Each predicted probability has a credible interval — wider for predictions "
                "far from the training data.</dd>"
                "<dt>Regularization via priors</dt>"
                "<dd>Weakly informative priors naturally regularize, preventing the separation "
                "problem that plagues classical logistic regression with small samples.</dd>"
                "</dl>"
            ),
        },
        ("bayesian", "bayes_survival"): {
            "title": "Understanding Bayesian Survival Analysis",
            "content": (
                "<dl>"
                "<dt>What is Bayesian survival analysis?</dt>"
                "<dd>Estimates survival functions and hazard rates with full posterior "
                "uncertainty. Handles censored data and produces credible intervals for "
                "reliability predictions.</dd>"
                "<dt>Posterior survival curve</dt>"
                "<dd>The entire survival curve has a credible band — showing where the "
                "estimate is precise and where it's uncertain.</dd>"
                "<dt>B-life posteriors</dt>"
                "<dd>B10, B50 etc. have posterior distributions. P(B10 &gt; t) directly "
                "answers warranty and design life questions.</dd>"
                "<dt>Model comparison</dt>"
                "<dd>Compares Weibull, lognormal, and exponential fits using Bayes Factors "
                "to select the best lifetime distribution.</dd>"
                "</dl>"
            ),
        },
        ("bayesian", "bayes_meta"): {
            "title": "Understanding Bayesian Meta-Analysis",
            "content": (
                "<dl>"
                "<dt>What is Bayesian meta-analysis?</dt>"
                "<dd>Combines results across studies using hierarchical models. The posterior "
                "of the overall effect and between-study variance are estimated simultaneously.</dd>"
                "<dt>Heterogeneity</dt>"
                "<dd>τ² (between-study variance) gets a posterior distribution — you can compute "
                "P(τ² &gt; 0) to assess whether heterogeneity is real.</dd>"
                "<dt>Predictive distribution</dt>"
                "<dd>Predicts the effect in a new study, accounting for both the overall effect "
                "and the between-study variability.</dd>"
                "<dt>Advantages</dt>"
                "<dd>Handles small numbers of studies better than frequentist random-effects. "
                "Naturally incorporates prior information from domain expertise.</dd>"
                "</dl>"
            ),
        },
        ("bayesian", "bayes_demo"): {
            "title": "Understanding Bayesian Demonstration Testing",
            "content": (
                "<dl>"
                "<dt>What is Bayesian demonstration testing?</dt>"
                "<dd>Demonstrates that a reliability target is met using Bayesian analysis. "
                "Computes P(R &gt; target | data) — the probability the true reliability "
                "exceeds the requirement.</dd>"
                "<dt>Prior information</dt>"
                "<dd>Can incorporate prior test data or engineering judgment to reduce "
                "required sample sizes. The prior must be justified and documented.</dd>"
                "<dt>Zero-failure testing</dt>"
                "<dd>Even with zero failures, the posterior gives meaningful bounds on "
                "reliability — unlike classical tests which require at least one failure.</dd>"
                "<dt>Planning</dt>"
                "<dd>Determine the smallest test that achieves P(R &gt; target) ≥ 0.95 "
                "given the prior and expected failure rate.</dd>"
                "</dl>"
            ),
        },
        ("bayesian", "bayes_spares"): {
            "title": "Understanding Bayesian Spare Parts Planning",
            "content": (
                "<dl>"
                "<dt>What is Bayesian spare parts planning?</dt>"
                "<dd>Predicts spare parts demand using posterior predictive distributions. "
                "Accounts for uncertainty in failure rates and fleet size.</dd>"
                "<dt>Demand distribution</dt>"
                "<dd>The posterior predictive gives the full distribution of future demand — "
                "not just the expected value. This enables service level targeting.</dd>"
                "<dt>Service level</dt>"
                "<dd>Stock to meet P(demand ≤ stock) ≥ target service level. The Bayesian "
                "approach honestly accounts for rate uncertainty.</dd>"
                "<dt>Sequential updating</dt>"
                "<dd>As actual demands are observed, the posterior updates and inventory "
                "recommendations adjust automatically.</dd>"
                "</dl>"
            ),
        },
        ("bayesian", "bayes_system"): {
            "title": "Understanding Bayesian System Reliability",
            "content": (
                "<dl>"
                "<dt>What is Bayesian system reliability?</dt>"
                "<dd>Estimates the reliability of a system composed of multiple components, "
                "with full posterior uncertainty propagated from component-level data.</dd>"
                "<dt>System structure</dt>"
                "<dd>Series (all must work), parallel (at least one must work), or complex "
                "configurations. Component posteriors are combined according to the structure.</dd>"
                "<dt>Uncertainty propagation</dt>"
                "<dd>Component-level uncertainty flows through the system model to give "
                "system-level credible intervals.</dd>"
                "<dt>Importance measures</dt>"
                "<dd>Bayesian importance measures identify which component's uncertainty "
                "contributes most to system-level uncertainty — guiding test investments.</dd>"
                "</dl>"
            ),
        },
        ("bayesian", "bayes_warranty"): {
            "title": "Understanding Bayesian Warranty Analysis",
            "content": (
                "<dl>"
                "<dt>What is Bayesian warranty analysis?</dt>"
                "<dd>Predicts future warranty claims using posterior predictive distributions, "
                "accounting for uncertainty in failure rates and reporting patterns.</dd>"
                "<dt>IBNR estimation</dt>"
                "<dd>Incurred But Not Reported claims are estimated from the posterior, giving "
                "a distribution of the total claim count.</dd>"
                "<dt>Financial reserves</dt>"
                "<dd>The posterior predictive distribution of total cost drives warranty "
                "reserve requirements. Higher percentiles give more conservative reserves.</dd>"
                "<dt>Cohort analysis</dt>"
                "<dd>Different production cohorts may have different failure rates. The "
                "hierarchical model borrows strength across cohorts while allowing differences.</dd>"
                "</dl>"
            ),
        },
        ("bayesian", "bayes_repairable"): {
            "title": "Understanding Bayesian Repairable Systems",
            "content": (
                "<dl>"
                "<dt>What is Bayesian repairable systems analysis?</dt>"
                "<dd>Models the failure process of repairable systems using Bayesian NHPP "
                "(Non-Homogeneous Poisson Process) with posterior uncertainty.</dd>"
                "<dt>Growth or deterioration</dt>"
                "<dd>The posterior of the trend parameter β tells you P(β &gt; 1) = probability "
                "the system is deteriorating. P(β &lt; 1) = probability of reliability growth.</dd>"
                "<dt>Prediction</dt>"
                "<dd>Predicts the number of failures in the next time period with credible "
                "intervals, accounting for parameter uncertainty.</dd>"
                "<dt>Maintenance optimization</dt>"
                "<dd>The posterior hazard function guides optimal maintenance scheduling — "
                "repair when the hazard rate exceeds a cost-based threshold.</dd>"
                "</dl>"
            ),
        },
        ("bayesian", "bayes_rul"): {
            "title": "Understanding Bayesian Remaining Useful Life",
            "content": (
                "<dl>"
                "<dt>What is Bayesian RUL?</dt>"
                "<dd>Estimates the Remaining Useful Life of a degrading system using posterior "
                "predictive distributions of the degradation path.</dd>"
                "<dt>Degradation model</dt>"
                "<dd>A parametric degradation model is fitted with Bayesian inference. The "
                "posterior captures uncertainty in degradation rate and form.</dd>"
                "<dt>RUL distribution</dt>"
                "<dd>The time until degradation reaches a failure threshold has a posterior "
                "distribution — giving P(RUL &gt; t) for any time horizon.</dd>"
                "<dt>Condition monitoring</dt>"
                "<dd>As new degradation measurements arrive, the RUL estimate updates — "
                "becoming more precise as the system approaches failure.</dd>"
                "</dl>"
            ),
        },
        ("bayesian", "bayes_alt"): {
            "title": "Understanding Bayesian ALT",
            "content": (
                "<dl>"
                "<dt>What is Bayesian ALT?</dt>"
                "<dd>Bayesian Accelerated Life Testing combines prior knowledge with test data "
                "to estimate lifetime at normal conditions. Posterior uncertainty propagates "
                "through the acceleration model.</dd>"
                "<dt>Prior elicitation</dt>"
                "<dd>Engineering knowledge about activation energies, voltage coefficients, etc. "
                "is encoded as priors. This can substantially reduce required test samples.</dd>"
                "<dt>Model uncertainty</dt>"
                "<dd>The posterior includes uncertainty in both the lifetime distribution and "
                "the acceleration model parameters.</dd>"
                "<dt>Normal-use prediction</dt>"
                "<dd>Posterior predictive at normal stress gives the lifetime distribution "
                "with full uncertainty from extrapolation.</dd>"
                "</dl>"
            ),
        },
        ("bayesian", "bayes_comprisk"): {
            "title": "Understanding Bayesian Competing Risks",
            "content": (
                "<dl>"
                "<dt>What is Bayesian competing risks?</dt>"
                "<dd>Analyzes multiple failure modes with posterior distributions for each "
                "mode's parameters, accounting for the competition between modes.</dd>"
                "<dt>Mode-specific posteriors</dt>"
                "<dd>Each failure mode gets its own posterior lifetime distribution, enabling "
                "mode-specific reliability predictions.</dd>"
                "<dt>Elimination analysis</dt>"
                "<dd>P(system survives to time t | mode k eliminated) shows the benefit of "
                "eliminating each failure mode — directly quantifying improvement potential.</dd>"
                "<dt>Dependent modes</dt>"
                "<dd>Bayesian methods can model dependence between failure modes through "
                "copulas or shared frailty parameters.</dd>"
                "</dl>"
            ),
        },
    }
)


# ═══════════════════════════════════════════════════════════════════════════
# External Modules (causal, drift, anytime, bayes_msa, quality_econ, pbs, ishap)
# ═══════════════════════════════════════════════════════════════════════════

_extend(
    {
        ("causal", "causal_pc"): {
            "title": "Understanding PC Causal Discovery",
            "content": (
                "<dl>"
                "<dt>What is the PC algorithm?</dt>"
                "<dd>A constraint-based causal discovery method that builds a causal graph from "
                "observational data using conditional independence tests. Named after Peter "
                "Spirtes and Clark Glymour.</dd>"
                "<dt>How does it work?</dt>"
                "<dd>Starts with a fully connected graph and removes edges when conditional "
                "independence is detected. Then orients edges based on v-structures.</dd>"
                "<dt>What can it tell you?</dt>"
                "<dd>Which variables have direct causal relationships (edges), which are "
                "independent (no edge), and some edge directions. Not all directions can be "
                "determined from observational data alone.</dd>"
                "<dt>Assumptions</dt>"
                "<dd>Causal sufficiency (no hidden confounders), faithfulness (all independencies "
                "reflect the graph), and adequate sample size. Violations can produce false edges.</dd>"
                "</dl>"
            ),
        },
        ("causal", "causal_lingam"): {
            "title": "Understanding LiNGAM Causal Discovery",
            "content": (
                "<dl>"
                "<dt>What is LiNGAM?</dt>"
                "<dd>Linear Non-Gaussian Acyclic Model — discovers causal direction from "
                "observational data by exploiting non-Gaussianity of error terms.</dd>"
                "<dt>How is it different from PC?</dt>"
                "<dd>PC can leave some edge directions undetermined. LiNGAM can identify all "
                "causal directions — but requires non-Gaussian data and linear relationships.</dd>"
                "<dt>Causal ordering</dt>"
                "<dd>LiNGAM discovers the full causal ordering of variables — which variables "
                "cause which, and with what strength.</dd>"
                "<dt>Limitations</dt>"
                "<dd>Assumes linearity, no hidden confounders, and non-Gaussian distributions. "
                "For Gaussian data, the causal direction is not identifiable — use PC instead.</dd>"
                "</dl>"
            ),
        },
        ("drift", "drift_report"): {
            "title": "Understanding Drift Detection",
            "content": (
                "<dl>"
                "<dt>What is data drift?</dt>"
                "<dd>A change in the statistical properties of input data over time. Drift can "
                "degrade model performance and invalidate process assumptions.</dd>"
                "<dt>Types of drift</dt>"
                "<dd><strong>Covariate drift</strong>: Input distribution changes. "
                "<strong>Concept drift</strong>: The relationship between inputs and outputs changes. "
                "<strong>Prior drift</strong>: The target distribution changes.</dd>"
                "<dt>Detection methods</dt>"
                "<dd>Statistical tests (KS, PSI, chi-square) compare reference and current "
                "distributions. Alerts trigger when drift exceeds a threshold.</dd>"
                "<dt>Response to drift</dt>"
                "<dd>Investigate the cause (process change? data collection change?). Retrain "
                "models if necessary. Update control limits and baselines.</dd>"
                "</dl>"
            ),
        },
        ("anytime", "anytime_ab"): {
            "title": "Understanding Anytime-Valid A/B Testing",
            "content": (
                "<dl>"
                "<dt>What is anytime-valid testing?</dt>"
                "<dd>A/B testing with confidence sequences that remain valid no matter when you "
                "look at the data. No need to pre-specify a sample size or deal with peeking "
                "problems.</dd>"
                "<dt>Confidence sequences vs intervals</dt>"
                "<dd>A confidence sequence is a sequence of intervals that contains the true "
                "parameter at all time points simultaneously. Classical CIs are only valid "
                "at the planned stopping time.</dd>"
                "<dt>When to use</dt>"
                "<dd>Continuous monitoring of A/B tests where you want to stop as soon as "
                "the evidence is clear, without pre-committing to a fixed sample size.</dd>"
                "<dt>E-values</dt>"
                "<dd>The evidence measure used by anytime-valid tests. E-values can be "
                "multiplied across time (like betting) — large E-values indicate strong evidence.</dd>"
                "</dl>"
            ),
        },
        ("anytime", "anytime_onesample"): {
            "title": "Understanding Anytime-Valid One-Sample Test",
            "content": (
                "<dl>"
                "<dt>What is an anytime-valid one-sample test?</dt>"
                "<dd>Tests whether a process mean equals a target value with validity at any "
                "stopping time. No need to commit to a sample size in advance.</dd>"
                "<dt>Sequential monitoring</dt>"
                "<dd>Watch the confidence sequence as data arrives. When the sequence excludes "
                "the null value, you have evidence of a difference — at any time you choose to look.</dd>"
                "<dt>Type I error control</dt>"
                "<dd>The error rate is controlled across all possible stopping times, not just one. "
                "This is stronger than classical tests and prevents p-hacking by design.</dd>"
                "<dt>Applications</dt>"
                "<dd>Process monitoring where you want to detect shifts as quickly as possible "
                "while maintaining rigorous error control.</dd>"
                "</dl>"
            ),
        },
        ("bayes_msa", "bayes_msa"): {
            "title": "Understanding Bayesian Gage R&R",
            "content": (
                "<dl>"
                "<dt>What is Bayesian Gage R&R?</dt>"
                "<dd>A Bayesian approach to measurement system analysis that provides posterior "
                "distributions for all variance components instead of point estimates.</dd>"
                "<dt>Advantages over ANOVA-based MSA</dt>"
                "<dd>No negative variance components, valid with small samples, full uncertainty "
                "quantification, and P(%GRR &lt; threshold) for direct decision-making.</dd>"
                "<dt>Posterior %GRR</dt>"
                "<dd>The posterior distribution of %GRR (measurement system as % of total) gives "
                "P(%GRR &lt; 10%) = probability the system is acceptable.</dd>"
                "<dt>Prior information</dt>"
                "<dd>Previous MSA studies can be incorporated as informative priors, reducing "
                "required study size for subsequent assessments.</dd>"
                "</dl>"
            ),
        },
        ("quality_econ", "taguchi_loss"): {
            "title": "Understanding Taguchi Loss Function",
            "content": (
                "<dl>"
                "<dt>What is the Taguchi loss function?</dt>"
                "<dd>A quadratic model that quantifies the economic cost of deviation from a "
                "target value. Loss increases with the square of the deviation — not just "
                "beyond spec limits.</dd>"
                "<dt>Why it matters</dt>"
                "<dd>Traditional pass/fail ignores the cost of being close to spec limits. "
                "Taguchi loss shows that a process centered on target is economically "
                "superior to one barely meeting specs.</dd>"
                "<dt>Loss = k(y - T)²</dt>"
                "<dd>k is the loss coefficient (cost per unit deviation²), y is the measured "
                "value, T is the target. The expected loss depends on both the mean offset "
                "and the variance.</dd>"
                "<dt>Applications</dt>"
                "<dd>Cost of quality estimation, process optimization targeting, and justifying "
                "investment in variability reduction.</dd>"
                "</dl>"
            ),
        },
        ("quality_econ", "process_decision"): {
            "title": "Understanding Process Decision Analysis",
            "content": (
                "<dl>"
                "<dt>What is process decision analysis?</dt>"
                "<dd>Evaluates the economic trade-offs of process improvement decisions — whether "
                "the cost of changing the process is justified by the quality improvement.</dd>"
                "<dt>Cost-benefit framework</dt>"
                "<dd>Compares the cost of improvement (equipment, training, downtime) against "
                "the expected reduction in defect-related costs (scrap, rework, warranty).</dd>"
                "<dt>Break-even analysis</dt>"
                "<dd>The defect rate improvement needed to justify the investment. If the "
                "expected improvement exceeds break-even, proceed.</dd>"
                "<dt>Decision under uncertainty</dt>"
                "<dd>Monte Carlo simulation of costs and benefits gives a distribution of ROI, "
                "not just a point estimate.</dd>"
                "</dl>"
            ),
        },
        ("quality_econ", "lot_sentencing"): {
            "title": "Understanding Lot Sentencing Economics",
            "content": (
                "<dl>"
                "<dt>What is lot sentencing economics?</dt>"
                "<dd>Evaluates the total cost of an acceptance sampling program, including "
                "inspection costs, accepted defect costs, and rejected lot costs.</dd>"
                "<dt>Total cost model</dt>"
                "<dd>C_total = C_inspection × n + C_accept_defective × AOQ × N + "
                "C_reject × P(reject). Balances sampling cost against quality risk.</dd>"
                "<dt>Economic lot size</dt>"
                "<dd>The lot size that minimizes total cost per unit, considering sampling "
                "plan efficiency and defect cost.</dd>"
                "<dt>Optimization</dt>"
                "<dd>Finds the sampling plan (n, c) that minimizes total cost while meeting "
                "AQL and LTPD requirements.</dd>"
                "</dl>"
            ),
        },
        ("quality_econ", "cost_of_quality"): {
            "title": "Understanding Cost of Quality",
            "content": (
                "<dl>"
                "<dt>What is Cost of Quality (COQ)?</dt>"
                "<dd>The total cost of quality activities, categorized into prevention costs, "
                "appraisal costs, internal failure costs, and external failure costs.</dd>"
                "<dt>The four categories</dt>"
                "<dd><strong>Prevention</strong>: Training, process design, DOE. "
                "<strong>Appraisal</strong>: Inspection, testing, audits. "
                "<strong>Internal failure</strong>: Scrap, rework, downtime. "
                "<strong>External failure</strong>: Warranty, returns, lost customers.</dd>"
                "<dt>The optimum</dt>"
                "<dd>Total COQ is minimized when prevention + appraisal spending reduces "
                "failure costs more than it adds. The chart shows the trade-off.</dd>"
                "<dt>Hidden costs</dt>"
                "<dd>External failure costs are often underestimated. Lost customer lifetime "
                "value and reputation damage can dwarf direct warranty costs.</dd>"
                "</dl>"
            ),
        },
        ("pbs", "pbs_full"): {
            "title": "Understanding Process Behaviour Summary",
            "content": (
                "<dl>"
                "<dt>What is the PBS?</dt>"
                "<dd>A comprehensive process behaviour analysis that combines control chart "
                "assessment, capability evaluation, and distributional analysis with "
                "Bayesian uncertainty quantification.</dd>"
                "<dt>What does it produce?</dt>"
                "<dd>Stability assessment, capability indices with posteriors, distributional "
                "fit, and actionable recommendations in one integrated report.</dd>"
                "<dt>Bayesian advantage</dt>"
                "<dd>All estimates come with credible intervals. Small-sample capability "
                "assessments are honest about uncertainty.</dd>"
                "<dt>When to use</dt>"
                "<dd>Initial process characterization, periodic process reviews, or any time "
                "you need a complete process behaviour assessment.</dd>"
                "</dl>"
            ),
        },
        ("pbs", "pbs_belief"): {
            "title": "Understanding PBS Belief Assessment",
            "content": (
                "<dl>"
                "<dt>What is the belief assessment?</dt>"
                "<dd>Quantifies the probability that the process meets specified requirements "
                "using posterior predictive analysis.</dd>"
                "<dt>P(in spec)</dt>"
                "<dd>The posterior probability that a randomly selected unit will be within "
                "specification limits — directly answering the quality question.</dd>"
                "<dt>Evidence strength</dt>"
                "<dd>How much data supports the belief. With few samples, the belief is uncertain "
                "(wide credible interval). More data tightens the bound.</dd>"
                "<dt>Decision threshold</dt>"
                "<dd>Compare P(in spec) against your decision threshold (e.g., 0.9973 for 3σ). "
                "The posterior probability makes the decision transparent.</dd>"
                "</dl>"
            ),
        },
        ("pbs", "pbs_edetector"): {
            "title": "Understanding PBS E-Detector",
            "content": (
                "<dl>"
                "<dt>What is the E-detector?</dt>"
                "<dd>An anytime-valid sequential change detection method that monitors for "
                "process shifts using e-values — providing rigorous false alarm control.</dd>"
                "<dt>How it works</dt>"
                "<dd>E-values accumulate evidence of a change. When the e-value exceeds a "
                "threshold, a change is declared — valid at any stopping time.</dd>"
                "<dt>Vs CUSUM/EWMA</dt>"
                "<dd>E-detectors provide mathematically guaranteed false alarm rates without "
                "distributional assumptions. CUSUM/EWMA rely on parametric assumptions.</dd>"
                "<dt>Detection speed</dt>"
                "<dd>Typically detects shifts within a few observations of the change, with "
                "speed depending on the shift magnitude.</dd>"
                "</dl>"
            ),
        },
        ("pbs", "pbs_evidence"): {
            "title": "Understanding PBS Evidence Assessment",
            "content": (
                "<dl>"
                "<dt>What is the evidence assessment?</dt>"
                "<dd>Evaluates the strength of evidence that the process has shifted from its "
                "baseline behaviour, using Bayes Factors.</dd>"
                "<dt>Evidence grades</dt>"
                "<dd>Anecdotal, moderate, strong, very strong, decisive — based on the BF "
                "magnitude. The grade tells you how much to trust the shift signal.</dd>"
                "<dt>Sequential evidence</dt>"
                "<dd>Evidence accumulates as data arrives. You can monitor the BF trajectory "
                "to see when evidence became compelling.</dd>"
                "<dt>Two-sided evidence</dt>"
                "<dd>The BF can also provide evidence for stability (no shift) — important for "
                "confirming that a process hasn't changed after a suspected event.</dd>"
                "</dl>"
            ),
        },
        ("pbs", "pbs_predictive"): {
            "title": "Understanding PBS Predictive Analysis",
            "content": (
                "<dl>"
                "<dt>What is predictive analysis?</dt>"
                "<dd>Uses the posterior predictive distribution to forecast where future "
                "observations will fall, accounting for all sources of uncertainty.</dd>"
                "<dt>Prediction intervals</dt>"
                "<dd>Unlike control limits (which characterize past behaviour), prediction "
                "intervals give honest ranges for future values. They're always wider because "
                "they include parameter uncertainty.</dd>"
                "<dt>Calibration</dt>"
                "<dd>A well-calibrated model has prediction intervals that actually contain "
                "the specified proportion of future observations.</dd>"
                "<dt>Applications</dt>"
                "<dd>Forecasting process output, setting realistic expectations for customers, "
                "and planning for worst-case scenarios.</dd>"
                "</dl>"
            ),
        },
        ("pbs", "pbs_adaptive"): {
            "title": "Understanding PBS Adaptive Monitoring",
            "content": (
                "<dl>"
                "<dt>What is adaptive monitoring?</dt>"
                "<dd>Continuously updates the process model as new data arrives, adjusting "
                "control limits and capability estimates in real time.</dd>"
                "<dt>Learning rate</dt>"
                "<dd>Controls how quickly the model adapts. Too fast: overreacts to noise. "
                "Too slow: misses real shifts. The Bayesian framework optimizes this.</dd>"
                "<dt>Vs static charts</dt>"
                "<dd>Static charts use fixed limits from historical data. Adaptive monitoring "
                "evolves with the process — ideal for processes that improve or change over time.</dd>"
                "<dt>When to use</dt>"
                "<dd>New processes without historical baselines, processes undergoing improvement, "
                "or seasonal processes where the baseline shifts.</dd>"
                "</dl>"
            ),
        },
        ("pbs", "pbs_cpk"): {
            "title": "Understanding PBS Bayesian Cpk",
            "content": (
                "<dl>"
                "<dt>What is Bayesian Cpk?</dt>"
                "<dd>Capability index estimated with full posterior uncertainty rather than "
                "a point estimate. Gives P(Cpk &gt; 1.33) — the probability the process "
                "is truly capable.</dd>"
                "<dt>Small sample honesty</dt>"
                "<dd>With n = 30, a classical Cpk = 1.5 could easily have true value 1.1 or 2.0. "
                "The Bayesian posterior shows this uncertainty explicitly.</dd>"
                "<dt>Specification comparison</dt>"
                "<dd>Compare capability against multiple thresholds simultaneously — probability "
                "of meeting each one is computed from the posterior.</dd>"
                "<dt>Decision framework</dt>"
                "<dd>Accept/reject decisions based on P(Cpk &gt; threshold) rather than point "
                "estimates — reduces the risk of wrong decisions from sampling variability.</dd>"
                "</dl>"
            ),
        },
        ("pbs", "pbs_cpk_traj"): {
            "title": "Understanding PBS Cpk Trajectory",
            "content": (
                "<dl>"
                "<dt>What is a Cpk trajectory?</dt>"
                "<dd>Tracks how the Bayesian Cpk estimate evolves over time as more data "
                "accumulates. Shows whether capability is improving, stable, or degrading.</dd>"
                "<dt>Convergence</dt>"
                "<dd>The credible interval narrows as more data arrives. When it stabilizes, "
                "you have enough data for a reliable capability assessment.</dd>"
                "<dt>Trend detection</dt>"
                "<dd>A systematic downward trend signals degrading capability — triggering "
                "proactive intervention before the process becomes incapable.</dd>"
                "<dt>Monitoring frequency</dt>"
                "<dd>Update the trajectory at regular intervals (shift, day, batch) to maintain "
                "continuous visibility into capability health.</dd>"
                "</dl>"
            ),
        },
        ("pbs", "pbs_health"): {
            "title": "Understanding PBS Process Health",
            "content": (
                "<dl>"
                "<dt>What is process health?</dt>"
                "<dd>A composite assessment combining stability, capability, distributional fit, "
                "and trend analysis into an overall process health score.</dd>"
                "<dt>Health components</dt>"
                "<dd><strong>Stability</strong>: No special causes detected. "
                "<strong>Capability</strong>: Cpk meets target. <strong>Centering</strong>: "
                "Process centered on target. <strong>Trend</strong>: No degradation.</dd>"
                "<dt>Traffic light system</dt>"
                "<dd>Green: All components healthy. Yellow: One or more marginal. Red: One or "
                "more failing. Immediate visibility into process status.</dd>"
                "<dt>Actionable recommendations</dt>"
                "<dd>Each health component maps to specific improvement actions. The analysis "
                "prioritizes interventions by impact.</dd>"
                "</dl>"
            ),
        },
        ("ishap", "ishap"): {
            "title": "Understanding Interventional SHAP",
            "content": (
                "<dl>"
                "<dt>What is Interventional SHAP?</dt>"
                "<dd>An extension of SHAP that computes feature attributions based on causal "
                "interventions rather than observational conditioning. Gives causal "
                "explanations for model predictions.</dd>"
                "<dt>Interventional vs observational</dt>"
                "<dd>Standard SHAP conditions on features (observational). Interventional SHAP "
                "simulates what happens when you actively change a feature — reflecting the "
                "real-world effect of process adjustments.</dd>"
                "<dt>Why it matters</dt>"
                "<dd>Observational SHAP can give misleading feature importance when features are "
                "correlated. Interventional SHAP correctly attributes effects through the causal "
                "structure.</dd>"
                "<dt>Requirements</dt>"
                "<dd>A trained model and knowledge of the causal structure (which variables "
                "are causes vs effects). Without causal knowledge, falls back to standard SHAP.</dd>"
                "</dl>"
            ),
        },
    }
)
