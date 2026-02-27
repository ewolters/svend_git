"""Learning content: Machine Learning."""

from ._datasets import SHARED_DATASET  # noqa: F401


ML_SUPERVISED_CLASSIFICATION = {
    "id": "ml-supervised-classification",
    "title": "Classification: Predicting Categories",
    "intro": "Classification is about predicting which group something belongs to. Will this customer churn? Is this email spam? In this section, you'll build intuition for how classifiers draw boundaries between groups — and why accuracy alone can be deeply misleading.",
    "exercise": {
        "title": "Try It: Build a Churn Classifier",
        "steps": [
            "Examine the churn dataset — notice the 27% class imbalance",
            "Toggle features on/off to see how the confusion matrix changes",
            "Start with just tenure_months — observe the decision boundary",
            "Add contract_type — watch accuracy jump but notice precision/recall tradeoff",
            "Try adding senior_citizen (a red herring) — see almost no improvement",
            "Find the feature combination that maximizes F1 score"
        ],
        "dsw_type": "stats:logistic",
        "dsw_config": {
            "response": "churned",
            "predictors": ["tenure_months", "monthly_charges", "contract_type"],
        },
    },
    "content": """
## What is Classification?

Classification predicts **categorical outcomes**: yes/no, spam/not-spam, defective/good. Unlike regression (which predicts numbers), classification draws **decision boundaries** that separate groups.

### The Simplest Classifier: Logistic Regression

Despite its name, logistic regression is a classifier. It models the **probability** of belonging to a class:

$$P(Y=1|X) = \\frac{1}{1 + e^{-(\\beta_0 + \\beta_1 X_1 + \\beta_2 X_2 + \\ldots)}}$$

The sigmoid function squashes any linear combination into a probability between 0 and 1. We then apply a **threshold** (usually 0.5) to make a binary prediction.

### Why Accuracy Lies

Consider a dataset where 95% of customers don't churn. A model that **always predicts "no churn"** has 95% accuracy — but it's completely useless for finding churners.

This is the **accuracy paradox**. With imbalanced classes, you need better metrics:

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| **Precision** | TP / (TP + FP) | Of those predicted positive, how many actually are? |
| **Recall** | TP / (TP + FN) | Of actual positives, how many did we catch? |
| **F1 Score** | 2 × (P × R) / (P + R) | Harmonic mean — balances precision and recall |
| **AUC-ROC** | Area under ROC curve | Overall discriminative ability across all thresholds |

### Don't Report a Single Number — Report Uncertainty

A model that reports "AUC = 0.84" without a confidence interval is like a clinical trial that reports a mean without error bars. **Always report CIs on your metrics:**

$$\\text{AUC} = 0.84 \\;\\; [0.79, 0.89]_{95\\%}$$

Bootstrap your evaluation metric (resample the test set with replacement 1000 times, compute the metric each time) to get a CI. If the CI is wide, you don't know your model's true performance — and the difference between two models might not be real.

### The Precision-Recall Tradeoff

You can't maximize both precision and recall simultaneously. Lowering the classification threshold catches more true positives (higher recall) but also more false positives (lower precision).

**When to prioritize recall:** Medical diagnosis, fraud detection — missing a positive case is costly.

**When to prioritize precision:** Email spam filters — flagging legitimate email is annoying.

### Feature Importance

Not all features help. Adding irrelevant features can actually **hurt** performance by introducing noise. The key question is: does this feature carry **information** about the outcome?

Signs a feature is useful:
- Different distributions between classes (churners vs non-churners)
- Low p-value in univariate test
- Positive mutual information with the target

Signs a feature is noise:
- Nearly identical distributions between classes
- High p-value (> 0.1)
- Adding it doesn't improve cross-validated score
""",
    "interactive": {
        "type": "classifier_playground",
        "config": {
            "dataset": "churn",
            "features": [
                {"name": "tenure_months", "label": "Tenure (months)", "default_on": True},
                {"name": "monthly_charges", "label": "Monthly Charges", "default_on": True},
                {"name": "contract_type", "label": "Contract Type", "default_on": False},
                {"name": "tech_support", "label": "Tech Support", "default_on": False},
                {"name": "num_tickets", "label": "Support Tickets", "default_on": False},
                {"name": "senior_citizen", "label": "Senior Citizen", "default_on": False},
            ],
            "target": "churned",
            "threshold": 0.5,
        }
    },
    "key_takeaways": [
        "Classification predicts categories by drawing decision boundaries between groups",
        "Accuracy is misleading with imbalanced classes — use precision, recall, and F1",
        "The precision-recall tradeoff means you must decide which errors are more costly",
        "Always report confidence intervals on metrics — AUC=0.84 means nothing without [0.79, 0.89]",
        "Not all features help — irrelevant features add noise and hurt performance",
    ],
    "practice_questions": [
        {
            "question": "A fraud detection model has 99.5% accuracy but only 10% recall. Is this model useful? Why or why not?",
            "answer": "No, it's nearly useless for its purpose. 10% recall means it catches only 1 in 10 fraud cases. The 99.5% accuracy is misleading because fraud is rare — the model is mostly just predicting 'not fraud' for everything. A model with 85% accuracy but 80% recall would be far more valuable.",
            "hint": "Think about what the model's job is — catching fraud — and which metric measures that."
        },
        {
            "question": "You're building a model to predict whether manufactured parts are defective. Should you optimize for precision or recall?",
            "answer": "Recall — missing a defective part (false negative) means shipping a bad product to customers, which is more costly than flagging a good part for re-inspection (false positive). You want to catch as many defective parts as possible.",
            "hint": "Consider the cost of each type of error: missing a defective part vs. needlessly re-inspecting a good part."
        }
    ]
}


ML_SUPERVISED_REGRESSION = {
    "id": "ml-supervised-regression",
    "title": "Regression: Predicting Values",
    "intro": "Regression predicts continuous numbers — price, temperature, demand. Simple in concept, but tricky in practice. The biggest trap? Building a model that memorizes your data instead of learning the underlying pattern. Here you'll watch overfitting happen in real time.",
    "exercise": {
        "title": "Try It: See Overfitting Happen",
        "steps": [
            "Start with a linear model (degree 1) — see the straight-line fit",
            "Increase polynomial degree to 3 — notice the curve captures the real pattern",
            "Push degree to 10 — watch the model contort itself to fit every point",
            "Compare training error vs test error at each complexity level",
            "Add regularization (increase lambda) — watch the overfitting smooth out",
            "Find the sweet spot: enough complexity to capture signal, not noise"
        ],
        "dsw_type": "stats:regression",
        "dsw_config": {
            "response": "total_charges",
            "predictors": ["tenure_months", "monthly_charges"],
        },
    },
    "content": """
## Regression: From Lines to Curves

Linear regression fits a line: $y = \\beta_0 + \\beta_1 x$. But real relationships are rarely perfectly linear.

**Polynomial regression** adds flexibility:
- Degree 1: $y = \\beta_0 + \\beta_1 x$ (line)
- Degree 2: $y = \\beta_0 + \\beta_1 x + \\beta_2 x^2$ (parabola)
- Degree 10: $y = \\beta_0 + \\beta_1 x + \\ldots + \\beta_{10} x^{10}$ (wiggly mess)

### The Bias-Variance Tradeoff

Every model makes two types of errors:

**Bias** — error from oversimplifying. A straight line through curved data has high bias. It consistently misses the pattern.

**Variance** — error from over-sensitivity to training data. A degree-20 polynomial has high variance. It fits the training data perfectly but falls apart on new data.

$$\\text{Total Error} = \\text{Bias}^2 + \\text{Variance} + \\text{Irreducible Noise}$$

The sweet spot is a model complex enough to capture the real pattern, simple enough to generalize.

### Regularization: Taming Complexity

Regularization adds a **penalty** for model complexity:

**Ridge (L2):** $\\text{Loss} = \\sum(y_i - \\hat{y}_i)^2 + \\lambda \\sum \\beta_j^2$

Shrinks all coefficients toward zero. Good when all features contribute somewhat.

**Lasso (L1):** $\\text{Loss} = \\sum(y_i - \\hat{y}_i)^2 + \\lambda \\sum |\\beta_j|$

Drives some coefficients exactly to zero — automatic feature selection.

**Elastic Net:** Combines both L1 and L2 penalties.

$\\lambda$ controls the penalty strength:
- $\\lambda = 0$: No regularization (back to ordinary regression)
- $\\lambda \\to \\infty$: All coefficients shrink to zero (predicts the mean)

### Key Regression Diagnostics

1. **R² (coefficient of determination):** proportion of variance explained. R² = 0.85 means 85% of variation in y is explained by x.
2. **RMSE (root mean squared error):** average prediction error in the same units as y. **This is your effect size** — it tells you how far off predictions actually are in meaningful units.
3. **Residual plots:** should show random scatter. Patterns indicate model misspecification.
4. **Training vs test error:** if training error is much lower than test error, you're overfitting.

### Prediction Intervals: The Honest Prediction

A point prediction ("this customer will spend $142") is incomplete. A **prediction interval** communicates what you actually know:

$$\\hat{y} \\pm t_{\\alpha/2} \\cdot s \\cdot \\sqrt{1 + \\frac{1}{n} + \\frac{(x - \\bar{x})^2}{\\sum(x_i - \\bar{x})^2}}$$

"This customer will spend between $98 and $186 (95% PI)." The width tells you how useful the prediction really is. A prediction interval of [$-50, $400] means you know almost nothing — the model might have good R² but still be useless for individual decisions.
""",
    "interactive": {
        "type": "regression_playground",
        "config": {
            "show_polynomial_degree": True,
            "max_degree": 12,
            "show_regularization": True,
            "show_train_test_split": True,
            "show_residuals": True,
        }
    },
    "key_takeaways": [
        "More complex models fit training data better but may generalize worse",
        "The bias-variance tradeoff is the central tension in all predictive modeling",
        "Regularization (Ridge, Lasso) constrains complexity and prevents overfitting",
        "RMSE is your effect size — it tells you prediction error in meaningful units",
        "Always report prediction intervals, not just point predictions — width reveals true uncertainty",
    ],
    "practice_questions": [
        {
            "question": "Your model has R²=0.99 on training data but R²=0.45 on test data. What's happening and what should you do?",
            "answer": "Classic overfitting — the model memorized training data instead of learning the pattern. Solutions: reduce model complexity (fewer features, lower polynomial degree), add regularization (Ridge/Lasso), get more training data, or use cross-validation to select hyperparameters.",
            "hint": "The huge gap between training and test performance is the key diagnostic."
        },
        {
            "question": "When would you choose Lasso over Ridge regression?",
            "answer": "When you suspect many features are irrelevant. Lasso drives coefficients to exactly zero (automatic feature selection), while Ridge only shrinks them toward zero. If you have 50 features but suspect only 5 matter, Lasso will identify them. Ridge is better when most features contribute at least somewhat.",
            "hint": "Think about what L1 vs L2 penalty does to small coefficients."
        }
    ]
}


ML_UNSUPERVISED = {
    "id": "ml-unsupervised",
    "title": "Clustering & Dimensionality Reduction",
    "intro": "Unsupervised learning finds structure in data without labels. No one tells the algorithm what to look for — it discovers groups and patterns on its own. The catch? There's no 'correct answer' to check against, so evaluating results requires domain judgment.",
    "exercise": {
        "title": "Try It: Discover Customer Segments",
        "steps": [
            "Start with k=2 clusters on tenure vs monthly_charges",
            "Increase to k=3 — does the silhouette score improve?",
            "Try k=4, k=5 — notice diminishing returns",
            "Toggle PCA on — see the 2D projection of all features",
            "Examine cluster profiles — what business meaning do they have?",
            "Decide: how many segments actually make sense for this business?"
        ],
    },
    "content": """
## Clustering: Finding Natural Groups

Clustering groups data points so that items within a cluster are similar and items between clusters are different. The most common algorithm is **K-Means**.

### K-Means Algorithm

1. Choose k (number of clusters)
2. Randomly place k centroids
3. Assign each point to the nearest centroid
4. Move each centroid to the mean of its assigned points
5. Repeat steps 3-4 until convergence

### The Elbow Method and Silhouette Score

**How do you choose k?** There's no single right answer, but two tools help:

**Elbow Method:** Plot within-cluster sum of squares (WCSS) vs k. Look for the "elbow" where adding clusters stops helping much.

**Silhouette Score:** Measures how similar each point is to its own cluster vs other clusters.

$$s(i) = \\frac{b(i) - a(i)}{\\max(a(i), b(i))}$$

- $a(i)$ = average distance to points in same cluster
- $b(i)$ = average distance to nearest other cluster
- Score ranges from -1 (wrong cluster) to +1 (well-clustered)
- Score > 0.5 = reasonable structure, > 0.7 = strong structure

### PCA: Dimensionality Reduction

**Principal Component Analysis** finds the directions of maximum variance in your data.

If you have 20 features, PCA might reveal that 90% of the variation can be captured with just 3 components. This lets you:
- **Visualize** high-dimensional data in 2D
- **Remove noise** from low-variance dimensions
- **Speed up** subsequent analysis

Each principal component is a linear combination of original features. The first component captures the most variance, the second captures the most remaining variance (orthogonal to the first), and so on.

### Quantify Cluster Quality with CIs

A silhouette score of 0.58 tells you clusters are "reasonable" — but how stable is that number? **Bootstrap the silhouette score** (resample data with replacement, re-cluster, recompute) to get a CI:

$$\\text{Silhouette} = 0.58 \\;\\; [0.49, 0.66]_{95\\%}$$

Now compare k=3 vs k=4: if k=3 gives 0.58 [0.49, 0.66] and k=4 gives 0.54 [0.44, 0.63], the overlap tells you they're essentially equivalent — choose the simpler solution.

Beyond cluster quality, measure **practical significance** of cluster differences. If your "high-value" cluster's mean monthly spend is $85 [78, 92] and the "low-value" cluster is $82 [75, 89], the CIs overlap — the clusters may not represent meaningfully different customer segments despite what k-means tells you.

### Interpreting Cluster Results

Clustering finds groups — but **you** must decide if they're meaningful:

- Do clusters correspond to real business segments?
- Are clusters stable if you re-run with different initialization?
- Can you describe each cluster in simple language?
- **Do cluster means differ by a practically important amount** (check CIs on the differences)?

A 3-cluster solution might be: "Price-sensitive short-term customers", "Loyal long-term subscribers", and "New high-value customers." If you can't name the clusters — or if the clusters don't differ meaningfully on the metrics that matter — they may not be real.
""",
    "interactive": {
        "type": "clustering_explorer",
        "config": {
            "dataset": "churn",
            "default_features": ["tenure_months", "monthly_charges"],
            "all_features": ["tenure_months", "monthly_charges", "total_charges", "num_tickets"],
            "min_k": 2,
            "max_k": 8,
            "show_pca": True,
            "show_silhouette": True,
            "show_elbow": True,
        }
    },
    "key_takeaways": [
        "Clustering finds groups without labels — but you must judge if the groups are meaningful",
        "K-Means requires choosing k upfront — use the elbow method and silhouette score to guide you",
        "Bootstrap the silhouette score to get CIs — compare k values by their intervals, not point estimates",
        "Check practical significance: do cluster means differ by an amount that matters for decisions?",
        "Always validate clusters with domain knowledge — statistical groupings aren't always useful groupings",
    ],
    "practice_questions": [
        {
            "question": "You run K-Means with k=2 through k=10. The silhouette score peaks at k=3 (0.62) and k=5 (0.58). Which should you choose?",
            "answer": "Start with k=3. Higher silhouette score means better-defined clusters, and fewer clusters are easier to interpret and act on. Only choose k=5 if domain knowledge suggests 5 distinct segments exist and the k=3 solution merges genuinely different groups. Simpler models that explain the data well should be preferred.",
            "hint": "Consider both the silhouette scores and practical interpretability."
        },
        {
            "question": "PCA on your 15-feature dataset shows the first 3 components explain 85% of variance. Should you drop the other 12 dimensions?",
            "answer": "For visualization and exploration, yes — 3 components capturing 85% is excellent. For prediction, be more careful: the remaining 15% might contain signal relevant to your outcome. Try both and compare model performance. PCA is great for understanding structure but can discard useful information.",
            "hint": "It depends on the goal — visualization vs prediction have different requirements."
        }
    ]
}


ML_MODEL_VALIDATION = {
    "id": "ml-model-validation",
    "title": "Cross-Validation & Model Selection",
    "intro": "A model that looks great on your data might be terrible on new data. Cross-validation is the discipline of testing your model on data it hasn't seen. It's the single most important technique for building models you can trust.",
    "exercise": {
        "title": "Try It: Watch the Bias-Variance Tradeoff",
        "steps": [
            "Select 5-fold cross-validation and a linear model",
            "Observe: training accuracy is consistent across folds (low variance)",
            "Switch to a decision tree with max_depth=20 (very complex)",
            "Notice: training accuracy is near-perfect but test accuracy varies wildly",
            "Reduce depth to 5 — find the sweet spot",
            "Compare the bias-variance decomposition at each complexity level"
        ],
    },
    "content": """
## Why You Need Cross-Validation

**The fundamental problem:** You want to know how your model performs on data it hasn't seen. But you only have the data you have.

**The naive approach:** Train on all data, test on all data. Result: overly optimistic estimates. The model has already "seen the answers."

**Train/test split:** Better — hold out 20% for testing. But your estimate depends on which 20% you happened to hold out.

**K-fold cross-validation:** The gold standard.

### How K-Fold Works

1. Split data into k equal parts (typically k=5 or k=10)
2. For each fold:
   - Train on k-1 folds
   - Test on the remaining fold
3. Average the k test scores

Every data point gets used for both training AND testing, but never at the same time. The result is a **reliable estimate** of out-of-sample performance.

### Report the CI, Not Just the Mean

Never report "accuracy = 0.82" from cross-validation. Report the mean **and** the confidence interval:

$$\\text{Accuracy} = 0.82 \\pm 0.05 \\;\\; [0.77, 0.87]_{95\\%}$$

The CI comes from the variation across folds. If fold scores are [0.91, 0.68, 0.88, 0.72, 0.90], the wide CI tells you the model is **unstable** — much more informative than the mean alone. Two models with means of 0.82 and 0.80 are probably indistinguishable if their CIs overlap heavily. **Don't chase the last 1% of accuracy when uncertainty is 5%.**

### Stratified K-Fold

For classification with imbalanced classes (like 27% churn), use **stratified** k-fold. Each fold maintains the same class proportions as the full dataset. Without stratification, some folds might have 35% churn and others 20%, making fold-to-fold comparisons meaningless.

### Nested Cross-Validation

If you're tuning hyperparameters (like regularization strength λ), you need **two loops**:

- **Outer loop:** Evaluates model performance (test score)
- **Inner loop:** Selects best hyperparameters (validation score)

Without nesting, your hyperparameter selection leaks information from the test set, inflating your performance estimate.

### Model Selection Guidelines

| Symptom | Diagnosis | Remedy |
|---------|-----------|--------|
| High training error, high test error | Underfitting (high bias) | More features, more complexity, less regularization |
| Low training error, high test error | Overfitting (high variance) | More data, fewer features, more regularization |
| Low training error, low test error | Good fit | Deploy it |
| Highly variable test scores across folds | Unstable model | Simpler model, more data, ensemble methods |

### The Learning Curve

Plot training and test error vs training set size:
- If both converge at high error → model is too simple (underfitting)
- If large gap persists → model is too complex (overfitting) or needs more data
- If both converge at low error → model is well-calibrated
""",
    "interactive": {
        "type": "validation_visualizer",
        "config": {
            "k_folds": 5,
            "show_fold_visualization": True,
            "show_learning_curve": True,
            "show_bias_variance": True,
            "complexity_levels": ["linear", "depth_3", "depth_5", "depth_10", "depth_20"],
        }
    },
    "key_takeaways": [
        "Never evaluate a model on the data it was trained on — the result is meaningless",
        "K-fold cross-validation gives a reliable estimate of out-of-sample performance",
        "Report CV scores as mean ± CI, not just the mean — the interval reveals model stability",
        "Two models with overlapping CIs are probably equivalent — don't chase noise",
        "Nested cross-validation prevents information leakage when tuning hyperparameters",
    ],
    "practice_questions": [
        {
            "question": "Your 5-fold CV scores are [0.91, 0.68, 0.88, 0.72, 0.90]. Should you trust the average (0.82)?",
            "answer": "The average is concerning because of the high variance — scores range from 0.68 to 0.91. This signals an unstable model. Two folds (0.68 and 0.72) are much worse, possibly because those folds contained data from a different subpopulation. Investigate what's different about the low-scoring folds before trusting any single number. The model may need simplification or the data may have heterogeneous subgroups.",
            "hint": "Look at the spread, not just the mean. What does high fold-to-fold variance indicate?"
        },
        {
            "question": "You tune hyperparameters using 5-fold CV and get accuracy of 0.87. You then report this as your model's expected performance. What's wrong?",
            "answer": "Information leakage. You selected the hyperparameters that gave the best 5-fold CV score, so that score is optimistically biased — it's the best of many tries. You need nested CV: an inner loop to select hyperparameters and an outer loop to estimate true performance. The true out-of-sample performance is typically 2-5% lower than the number you'd report.",
            "hint": "Think about what happens when you pick the best of many options based on the same data."
        }
    ]
}


ML_FEATURE_ENGINEERING = {
    "id": "ml-feature-engineering",
    "title": "Feature Engineering",
    "intro": "Raw data is rarely in the right form for modeling. Feature engineering transforms raw inputs into representations that help models learn. A good feature can be worth more than a complex algorithm — and a bad one can make the best algorithm useless.",
    "exercise": {
        "title": "Try It: Transform Features",
        "steps": [
            "Start with raw tenure_months as the only feature — note the model score",
            "Apply log transform to monthly_charges — does it help for regression?",
            "Bin tenure into categories (new/mid/veteran) — compare to raw tenure for classification",
            "Create an interaction feature: tenure × monthly_charges — see if it captures something new",
            "One-hot encode contract_type — compare to ordinal encoding",
            "Compare model performance before and after engineering"
        ],
    },
    "content": """
## Why Feature Engineering Matters

The same underlying information, represented differently, can dramatically change model performance. Consider tenure in months:

- **Raw:** 1, 2, 3, ..., 72 — linear relationship assumed
- **Log-transformed:** log(1), log(2), ... — compresses right tail, useful for right-skewed data
- **Binned:** "new" (0-6mo), "growing" (7-24mo), "loyal" (25+mo) — captures non-linear effects
- **Squared:** 1, 4, 9, ... — captures U-shaped relationships

### Common Transformations

**Numeric Features:**

| Transform | When to use | Effect |
|-----------|-------------|--------|
| Log | Right-skewed data (income, prices) | Compresses large values, normalizes |
| Square root | Count data, moderate skew | Gentler than log |
| Binning | Non-linear relationships | Captures thresholds and step changes |
| Standardization | Features on different scales | Mean=0, SD=1 — critical for distance-based methods |
| Min-max scaling | Neural networks, bounded features | Maps to [0, 1] range |

**Categorical Features:**

| Encoding | When to use | How it works |
|----------|-------------|--------------|
| One-hot | Nominal categories (no order) | Creates binary column per category |
| Ordinal | Ordered categories (low/med/high) | Maps to integers (0, 1, 2) |
| Target encoding | High-cardinality categories | Replaces category with mean of target (with regularization) |
| Binary encoding | Many categories (50+) | Encodes category number in binary |

### Interaction Features

Sometimes the relationship between X and Y depends on a third variable Z. An **interaction term** captures this:

$$\\text{feature}_{new} = X_1 \\times X_2$$

Example: The effect of monthly charges on churn might depend on contract type. Month-to-month customers with high charges are very likely to churn, but two-year contract customers with high charges are not — they're locked in.

### Feature Selection: Less Can Be More

Adding features isn't always better. Irrelevant features:
- Add noise that obscures real signals
- Increase overfitting risk
- Slow down training and prediction

**Methods to remove useless features:**
1. **Correlation filter:** Remove features with near-zero correlation to target
2. **Variance filter:** Remove features with near-zero variance (nearly constant)
3. **Lasso:** Drives unimportant coefficients to exactly zero
4. **Recursive Feature Elimination:** Iteratively removes least important feature
""",
    "interactive": {
        "type": "feature_transformer",
        "config": {
            "available_transforms": ["log", "sqrt", "bin", "standardize", "square", "interaction"],
            "features": ["tenure_months", "monthly_charges", "total_charges", "num_tickets"],
            "categorical_features": ["contract_type", "internet_service", "tech_support"],
            "encodings": ["one_hot", "ordinal", "target"],
            "show_distribution_before_after": True,
            "show_model_comparison": True,
        }
    },
    "key_takeaways": [
        "Feature engineering is often more impactful than choosing a fancier algorithm",
        "Log transforms help with right-skewed data; binning captures non-linear thresholds",
        "Interaction features capture when the effect of one variable depends on another",
        "Measure improvement with CIs: if AUC goes from 0.81 [0.76,0.86] to 0.83 [0.78,0.88], the overlap says it's probably noise",
        "More features is not always better — irrelevant features add noise and cause overfitting",
    ],
    "practice_questions": [
        {
            "question": "You have a 'zip_code' column with 500 unique values. How should you encode it for a model?",
            "answer": "One-hot encoding would create 500 sparse columns — too many. Better options: (1) Target encoding — replace each zip with the mean of the target variable (with smoothing to avoid overfitting), (2) Group into regions (state, metro area) and one-hot encode those, or (3) Extract features from zip codes (median income, population density) that might actually predict the outcome.",
            "hint": "One-hot encoding with 500 categories creates a very wide, sparse matrix."
        },
        {
            "question": "Your model uses raw income (range $20K-$500K) and age (range 18-80). Should you standardize?",
            "answer": "It depends on the algorithm. For distance-based methods (K-means, KNN, SVM) or regularized regression (Ridge, Lasso): yes, absolutely — income's large scale would dominate. For tree-based methods (Random Forest, XGBoost): no need — they split on individual features and are scale-invariant. For neural networks: standardization helps gradient descent converge faster.",
            "hint": "Consider whether the algorithm is sensitive to feature scale."
        }
    ]
}


ML_ENSEMBLE_METHODS = {
    "id": "ml-ensemble-methods",
    "title": "Ensemble Methods",
    "intro": "A single decision tree is fragile and prone to overfitting. But what if you built 500 trees and let them vote? Ensemble methods combine many weak models into a strong one. They're behind most winning solutions in applied ML — and understanding why they work builds deep intuition about the bias-variance tradeoff.",
    "exercise": {
        "title": "Try It: From Weak to Strong",
        "steps": [
            "Start with a single decision tree — note its accuracy and variance across folds",
            "Enable bagging (Random Forest) with 10 trees — see variance drop sharply",
            "Increase to 100 trees — diminishing returns but even more stable",
            "Switch to boosting — see bias decrease as each tree corrects the previous",
            "Compare: Random Forest vs Boosted Trees vs single tree on test accuracy",
            "Examine: which approach wins, and why?"
        ],
    },
    "content": """
## Why Ensembles Work

A single model makes systematic errors. By combining many models, the individual errors tend to **cancel out** — if they're independent enough.

### The Wisdom of Crowds

Imagine 100 people each guess the weight of an ox. Individual guesses are noisy, but the **average** is remarkably accurate. This is the core idea behind ensembles.

For it to work, models must be **diverse** — if all 100 make the same mistakes, averaging doesn't help.

### Bagging: Reducing Variance

**B**ootstrap **Agg**regat**ing** (Bagging):

1. Create B bootstrap samples (random samples with replacement)
2. Train one model on each sample
3. Average predictions (regression) or take majority vote (classification)

**Random Forest** is bagging with an extra twist: at each split, only a random subset of features is considered. This forces trees to be different, increasing diversity.

$$\\text{Variance}_{\\text{ensemble}} \\approx \\frac{\\text{Variance}_{\\text{single}}}{B}$$

More trees = lower variance = more stable predictions. But bias stays the same — bagging reduces variance, not bias.

### Boosting: Reducing Bias

Boosting builds models **sequentially**, each one focusing on the mistakes of the previous:

1. Train model 1 on the full dataset
2. Give more weight to misclassified examples
3. Train model 2 on the reweighted data
4. Repeat, accumulating models

**AdaBoost** reweights examples. **Gradient Boosting** fits each new tree to the **residuals** (prediction errors) of the ensemble so far.

Boosting reduces **bias** — each new tree corrects systematic errors. But it can overfit if you add too many trees or make them too complex.

### Bagging vs Boosting

| Property | Bagging (Random Forest) | Boosting (XGBoost) |
|----------|------------------------|---------------------|
| Reduces | Variance | Bias |
| Trees | Independent, parallel | Sequential, corrective |
| Overfitting risk | Low | Higher (needs early stopping) |
| Best for | High-variance models | High-bias models |
| Tuning | Easy (more trees = better) | Careful (learning rate, depth, rounds) |

### Is the Improvement Real? CIs on Model Comparisons

You switch from Random Forest to XGBoost and accuracy goes from 0.82 to 0.84. Is that a real improvement? **Only if the confidence intervals don't overlap.**

Compare models using paired cross-validation: run both models on the same folds, then compute the CI on the **difference** in accuracy:

$$\\Delta\\text{accuracy} = 0.02 \\;\\; [-0.01, 0.05]_{95\\%}$$

If the CI includes zero, the models are statistically indistinguishable. Don't add complexity for 2% when uncertainty is 6%. The simpler, more interpretable model is worth more than a negligible accuracy gain.

### Practical Ensemble Tips

- **Start with Random Forest** — hard to overfit, minimal tuning
- **Graduate to Gradient Boosting** only when the CI on the improvement excludes zero
- **Use early stopping** with boosting to prevent overfitting
- **Stacking** (training a meta-model on predictions of base models) can squeeze out more performance but adds complexity
- **Report uncertainty:** Every model comparison should include the CI on the difference, not just two point estimates
""",
    "interactive": {
        "type": "ensemble_builder",
        "config": {
            "methods": ["single_tree", "bagging", "random_forest", "boosting"],
            "n_trees_slider": True,
            "show_individual_predictions": True,
            "show_ensemble_prediction": True,
            "show_variance_across_folds": True,
            "show_bias_variance_decomposition": True,
        }
    },
    "key_takeaways": [
        "Ensembles combine many weak models into one strong model by canceling out individual errors",
        "Bagging (Random Forest) reduces variance — makes unstable models more stable",
        "Boosting reduces bias — makes simple models more powerful by correcting errors sequentially",
        "Always compare models with CIs on the difference — a 2% gain with a [-1%, 5%] CI is noise, not signal",
        "Random Forest is hard to overfit and requires little tuning — prefer it unless boosting demonstrably improves the CI",
    ],
    "practice_questions": [
        {
            "question": "Your single decision tree has high training accuracy (95%) but poor test accuracy (72%). Would you try bagging or boosting?",
            "answer": "Bagging (Random Forest). The large gap between training and test accuracy indicates high variance (overfitting). Bagging reduces variance by averaging many trees trained on different bootstrap samples. Boosting would likely make overfitting worse since it reduces bias (which isn't the problem here).",
            "hint": "What kind of error does the model have — high bias or high variance?"
        },
        {
            "question": "You've trained a Random Forest with 500 trees. Adding more trees to 2000 improves training score slightly but test score stays the same. What's happening?",
            "answer": "Diminishing returns — Random Forest is already close to optimal for this data. More trees reduce variance further, but the variance was already low at 500 trees. The remaining error is either bias (the model can't capture the true relationship) or irreducible noise. To improve further, you'd need better features, more data, or a different algorithm (try gradient boosting to reduce bias).",
            "hint": "After a certain point, adding trees only reduces variance marginally."
        }
    ]
}


ML_INTERPRETABILITY = {
    "id": "ml-interpretability",
    "title": "Model Interpretability",
    "intro": "A model that predicts well but can't be explained is dangerous. In regulated industries, high-stakes decisions, and anywhere trust matters, you need to open the black box. Interpretability isn't just nice to have — it's how you catch bugs, build trust, and make better decisions.",
    "exercise": {
        "title": "Try It: Explain a Prediction",
        "steps": [
            "Look at the global feature importance — which features drive predictions overall?",
            "Pick a specific customer and see the SHAP-style contribution of each feature",
            "Find a customer where the model says 'high churn risk' — which features push the prediction up?",
            "Compare two similar customers with different predictions — what's the key difference?",
            "Toggle features off one at a time — which single removal changes predictions most?"
        ],
    },
    "content": """
## Why Interpretability Matters

**Model debugging:** A model that uses 'customer ID' as its top feature is probably leaking data, not learning patterns.

**Trust building:** Stakeholders won't act on predictions they don't understand. "The model says so" isn't persuasive.

**Regulatory compliance:** GDPR's "right to explanation" and industry regulations (healthcare, finance) may require explainable predictions.

**Scientific insight:** Sometimes understanding *why* is more valuable than *what*. Why do customers churn? That's actionable.

### Global vs Local Interpretability

**Global:** What features matter most across ALL predictions?
- Feature importance rankings
- Partial dependence plots
- Overall model behavior

**Local:** Why did the model make THIS specific prediction?
- Individual feature contributions
- Counterfactual explanations ("if tenure were 24 months instead of 3, prediction would flip")

### Feature Importance Methods

**Permutation Importance:** Shuffle one feature's values randomly. If accuracy drops a lot, that feature is important.
- Pro: Works for any model, simple to compute
- Con: Correlated features can hide each other's importance

**SHAP (SHapley Additive exPlanations):** Based on game theory. For each prediction, decompose it into contributions from each feature.

$$\\hat{y}(x) = \\phi_0 + \\sum_{j=1}^{M} \\phi_j$$

Where $\\phi_0$ is the base prediction (average) and $\\phi_j$ is the contribution of feature $j$.

**SHAP properties:**
- Contributions sum to the prediction (efficiency)
- Features with no effect get zero contribution (null player)
- Feature interactions are distributed fairly (symmetry)

### Partial Dependence Plots (PDP)

Show the **marginal effect** of one feature on predictions, averaging over all other features.

Example: A PDP for tenure might show:
- 0-6 months: high churn probability
- 6-24 months: sharply declining
- 24+ months: flat and low

This reveals the relationship the model learned — and you can validate it against domain knowledge.

### SHAP Values Are Effect Sizes

SHAP values are inherently **effect size measures** — each $\\phi_j$ tells you the magnitude and direction of a feature's contribution to a specific prediction, in the same units as the model output. This makes them the ML equivalent of regression coefficients:

- $\\phi_{\\text{tenure}} = -0.15$ means tenure pushes this customer's churn probability **down by 15 percentage points** from baseline
- $\\phi_{\\text{contract}} = +0.22$ means month-to-month contract pushes it **up by 22 points**

**Bootstrap SHAP for uncertainty:** Permutation importance and SHAP values are point estimates. Bootstrap the SHAP calculation (resample data, recompute) to get CIs on feature importance. If tenure's importance is 0.15 [0.08, 0.22] and tech_support is 0.12 [0.03, 0.21], their CIs overlap heavily — they may be equally important.

### When Interpretability Isn't Free

More interpretable models are often less accurate:

| Model | Interpretability | Typical Accuracy |
|-------|-----------------|------------------|
| Linear/Logistic Regression | High (coefficients are effects) | Lower |
| Decision Tree | High (follow the splits) | Lower |
| Random Forest | Medium (feature importance) | Higher |
| Gradient Boosting | Low (many sequential trees) | Highest |
| Neural Network | Very Low (weights are opaque) | Highest |

Often the accuracy difference is small (1-3%), and an interpretable model you trust is worth more than a black box with slightly better numbers.
""",
    "interactive": {
        "type": "shap_explorer",
        "config": {
            "dataset": "churn",
            "show_global_importance": True,
            "show_individual_explanation": True,
            "show_feature_toggle": True,
            "show_partial_dependence": True,
            "features": ["tenure_months", "monthly_charges", "contract_type", "tech_support", "num_tickets", "internet_service"],
            "target": "churned",
        }
    },
    "key_takeaways": [
        "Interpretability catches model bugs, builds trust, and satisfies regulatory requirements",
        "SHAP values are effect sizes — they tell you the magnitude and direction of each feature's contribution",
        "Bootstrap SHAP to get CIs on feature importance — don't rank features by point estimates alone",
        "Permutation importance measures how much accuracy drops when a feature is shuffled",
        "Sometimes a simpler, interpretable model is worth more than a slightly more accurate black box",
    ],
    "practice_questions": [
        {
            "question": "A SHAP analysis shows 'customer_id' has the highest importance in your churn model. What should you do?",
            "answer": "This is a red flag — customer_id has no real predictive power. The model is memorizing individual customers rather than learning patterns. This signals data leakage or severe overfitting. Remove customer_id from features immediately and retrain. This is exactly why interpretability matters: it caught a serious bug that accuracy metrics alone would miss.",
            "hint": "Should a random identifier have predictive power? What does it mean if it does?"
        },
        {
            "question": "Your manager wants to know why the model predicted a specific customer as 'high churn risk.' What interpretability method would you use?",
            "answer": "SHAP values for that individual customer. SHAP provides local explanations — it shows exactly how much each feature pushed the prediction up or down from the baseline. For example: 'Baseline churn probability is 27%. This customer's month-to-month contract adds +18%, their 3-month tenure adds +12%, but their low ticket count subtracts -5%, resulting in a 52% churn prediction.' This is concrete and actionable.",
            "hint": "You need local (individual prediction) interpretability, not global."
        }
    ]
}
