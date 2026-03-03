"""
Analyst Trainer

Main orchestrator for ML training.
Takes data + intent, produces model + educational report.
"""

import time
import pickle
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime
from typing import Optional

from .models import ModelType, TaskType, create_model, get_model_info
from .selector import ModelSelector, DataProfile, Priority
from .reporter import AnalystReporter, AnalystReport
from .eda import EDAReport


class WarningSeverity(Enum):
    """Severity levels for data quality warnings."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DataQualityWarning:
    """A data quality warning from EDA."""
    severity: WarningSeverity
    category: str  # "missing", "outliers", "correlation", "imbalance"
    message: str
    details: dict = field(default_factory=dict)

    def __str__(self):
        icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}[self.severity.value]
        return f"{icon} [{self.category.upper()}] {self.message}"


@dataclass
class DataQualityAssessment:
    """Assessment of data quality for ML training."""
    warnings: list[DataQualityWarning] = field(default_factory=list)
    overall_quality: str = "good"  # good, acceptable, poor
    training_recommended: bool = True

    @property
    def has_critical(self) -> bool:
        return any(w.severity == WarningSeverity.CRITICAL for w in self.warnings)

    @property
    def warning_count(self) -> int:
        return sum(1 for w in self.warnings if w.severity == WarningSeverity.WARNING)

    def summary(self) -> str:
        """Get a summary of data quality."""
        lines = [f"Data Quality: {self.overall_quality.upper()}"]
        if not self.warnings:
            lines.append("No data quality issues detected.")
        else:
            lines.append(f"Found {len(self.warnings)} issue(s):")
            for w in self.warnings:
                lines.append(f"  {w}")
        if not self.training_recommended:
            lines.append("\n⚠️ Training NOT recommended until issues are addressed.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "overall_quality": self.overall_quality,
            "training_recommended": self.training_recommended,
            "warnings": [
                {
                    "severity": w.severity.value,
                    "category": w.category,
                    "message": w.message,
                    "details": w.details,
                }
                for w in self.warnings
            ],
        }


def assess_data_quality(eda_report: EDAReport, target: str = None) -> DataQualityAssessment:
    """
    Assess data quality from an EDA report.

    Thresholds:
    - Missing > 10%: WARNING
    - Missing > 30%: CRITICAL
    - Outliers > 5%: WARNING
    - Outliers > 15%: CRITICAL
    - High correlation > 0.95: WARNING (multicollinearity)
    - Target imbalance > 10:1: WARNING
    """
    warnings = []

    # Check overall missing values
    if eda_report.total_missing_pct > 0.30:
        warnings.append(DataQualityWarning(
            severity=WarningSeverity.CRITICAL,
            category="missing",
            message=f"Critical missing data: {eda_report.total_missing_pct:.1%} of all values are missing",
            details={"missing_pct": eda_report.total_missing_pct},
        ))
    elif eda_report.total_missing_pct > 0.10:
        warnings.append(DataQualityWarning(
            severity=WarningSeverity.WARNING,
            category="missing",
            message=f"High missing data: {eda_report.total_missing_pct:.1%} of values are missing",
            details={"missing_pct": eda_report.total_missing_pct},
        ))

    # Check per-column missing
    # Note: Most sklearn models cannot handle ANY missing values, so we're
    # stricter here - any column with significant missing data is a problem.
    has_any_missing = any(col.missing_pct > 0 for col in eda_report.columns)

    for col in eda_report.columns:
        if col.missing_pct > 0.30:
            # >30% missing in any column = cannot train with most models
            warnings.append(DataQualityWarning(
                severity=WarningSeverity.CRITICAL,
                category="missing",
                message=f"Column '{col.name}' has {col.missing_pct:.1%} missing - cannot train with most sklearn models",
                details={"column": col.name, "missing_pct": col.missing_pct},
            ))
        elif col.missing_pct > 0.10:
            warnings.append(DataQualityWarning(
                severity=WarningSeverity.WARNING,
                category="missing",
                message=f"Column '{col.name}' has {col.missing_pct:.1%} missing - imputation recommended",
                details={"column": col.name, "missing_pct": col.missing_pct},
            ))
        elif col.missing_pct > 0:
            warnings.append(DataQualityWarning(
                severity=WarningSeverity.INFO,
                category="missing",
                message=f"Column '{col.name}' has {col.missing_pct:.1%} missing values",
                details={"column": col.name, "missing_pct": col.missing_pct},
            ))

    # Check outliers
    total_outliers = sum(col.outlier_count for col in eda_report.columns if col.has_outliers)
    total_numeric_values = sum(col.count for col in eda_report.columns if col.is_numeric)

    if total_numeric_values > 0:
        outlier_pct = total_outliers / total_numeric_values
        if outlier_pct > 0.15:
            warnings.append(DataQualityWarning(
                severity=WarningSeverity.CRITICAL,
                category="outliers",
                message=f"Excessive outliers: {outlier_pct:.1%} of numeric values are outliers",
                details={"outlier_pct": outlier_pct, "outlier_count": total_outliers},
            ))
        elif outlier_pct > 0.05:
            warnings.append(DataQualityWarning(
                severity=WarningSeverity.WARNING,
                category="outliers",
                message=f"High outlier rate: {outlier_pct:.1%} of numeric values are outliers",
                details={"outlier_pct": outlier_pct, "outlier_count": total_outliers},
            ))

    # Check for columns with extreme outliers
    for col in eda_report.columns:
        if col.has_outliers and col.count > 0:
            col_outlier_pct = col.outlier_count / col.count
            if col_outlier_pct > 0.10:
                warnings.append(DataQualityWarning(
                    severity=WarningSeverity.WARNING,
                    category="outliers",
                    message=f"Column '{col.name}' has {col_outlier_pct:.1%} outliers ({col.outlier_count} values)",
                    details={"column": col.name, "outlier_count": col.outlier_count},
                ))

    # Check for multicollinearity
    if eda_report.high_correlations:
        very_high = [(c1, c2, r) for c1, c2, r in eda_report.high_correlations if abs(r) > 0.95]
        if very_high:
            warnings.append(DataQualityWarning(
                severity=WarningSeverity.WARNING,
                category="correlation",
                message=f"Multicollinearity detected: {len(very_high)} feature pair(s) with r > 0.95",
                details={"pairs": very_high[:5]},  # First 5
            ))

    # Check target variable for imbalance (if categorical)
    if target:
        target_col = next((c for c in eda_report.columns if c.name == target), None)
        if target_col and target_col.is_categorical and target_col.top_values:
            values = target_col.top_values
            if len(values) >= 2:
                max_count = values[0][1]
                min_count = values[-1][1]
                if min_count > 0:
                    imbalance_ratio = max_count / min_count
                    if imbalance_ratio > 10:
                        warnings.append(DataQualityWarning(
                            severity=WarningSeverity.WARNING,
                            category="imbalance",
                            message=f"Class imbalance in target '{target}': {imbalance_ratio:.1f}:1 ratio",
                            details={"imbalance_ratio": imbalance_ratio, "class_distribution": values},
                        ))

    # Check for duplicate rows
    if eda_report.duplicate_pct > 0.05:
        warnings.append(DataQualityWarning(
            severity=WarningSeverity.WARNING,
            category="duplicates",
            message=f"High duplicate rate: {eda_report.duplicate_pct:.1%} of rows are duplicates",
            details={"duplicate_pct": eda_report.duplicate_pct},
        ))

    # Determine overall quality
    critical_count = sum(1 for w in warnings if w.severity == WarningSeverity.CRITICAL)
    warning_count = sum(1 for w in warnings if w.severity == WarningSeverity.WARNING)

    if critical_count > 0:
        overall_quality = "poor"
        training_recommended = False
    elif warning_count >= 3:
        overall_quality = "acceptable"
        training_recommended = True
    elif warning_count > 0:
        overall_quality = "acceptable"
        training_recommended = True
    else:
        overall_quality = "good"
        training_recommended = True

    return DataQualityAssessment(
        warnings=warnings,
        overall_quality=overall_quality,
        training_recommended=training_recommended,
    )


@dataclass
class TrainingRequest:
    """Request to train a model."""
    target: str  # Target column name
    intent: str = ""  # User's intent description

    # Optional: specify model
    model_type: ModelType = None  # None = auto-select

    # Optional: hyperparameters
    hyperparameters: dict = field(default_factory=dict)

    # Training options
    test_size: float = 0.2  # Fraction for test set
    random_state: int = 42
    priority: str = "balanced"  # accuracy, interpretability, speed, balanced

    # Preprocessing
    scale_features: bool = True
    encode_categorical: bool = True

    # EDA integration
    eda_report: Optional[EDAReport] = None  # Pre-computed EDA (avoids re-running)


@dataclass
class TrainingResult:
    """Result of model training."""
    model: object  # Trained sklearn model
    model_type: ModelType
    task_type: TaskType
    report: AnalystReport

    # For reproduction
    feature_names: list[str] = field(default_factory=list)
    target_name: str = ""
    scaler: object = None  # Fitted scaler if used
    encoder: object = None  # Fitted encoder if used

    # Code generation
    _code: str = ""

    # Data quality from EDA
    data_quality: Optional[DataQualityAssessment] = None

    @property
    def code(self) -> str:
        """Get reproducible Python code."""
        return self._code

    def predict(self, X):
        """Make predictions on new data."""
        import numpy as np

        # Apply same preprocessing
        if self.encoder is not None:
            X = self._encode(X)
        if self.scaler is not None:
            X = self.scaler.transform(X)

        return self.model.predict(X)

    def predict_proba(self, X):
        """Get probability predictions (classification only)."""
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("This model doesn't support probability predictions")

        if self.encoder is not None:
            X = self._encode(X)
        if self.scaler is not None:
            X = self.scaler.transform(X)

        return self.model.predict_proba(X)

    def _encode(self, X):
        """Apply categorical encoding."""
        import pandas as pd
        import numpy as np

        if isinstance(X, pd.DataFrame):
            X = X.copy()
            for col in X.select_dtypes(include=['object', 'category']).columns:
                if col in self.encoder.get('columns', []):
                    mapping = self.encoder['mappings'].get(col, {})
                    X[col] = X[col].map(mapping).fillna(-1)
            return X.values
        return X

    def save(self, directory: str) -> dict:
        """
        Save model and all artifacts.

        Saves:
        - model.pkl: Trained model
        - report.md: Educational report
        - report.json: Metrics and data
        - code.py: Reproducible training code
        - metadata.json: Feature info, preprocessing

        Returns dict of saved file paths.
        """
        import json

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        saved = {}

        # Save model
        model_path = directory / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        saved['model'] = str(model_path)

        # Save scaler if used
        if self.scaler is not None:
            scaler_path = directory / "scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            saved['scaler'] = str(scaler_path)

        # Save report
        report_md_path = directory / "report.md"
        report_md_path.write_text(self.report.to_markdown())
        saved['report_md'] = str(report_md_path)

        report_json_path = directory / "report.json"
        report_json_path.write_text(json.dumps(self.report.to_dict(), indent=2))
        saved['report_json'] = str(report_json_path)

        # Save code
        code_path = directory / "train.py"
        code_path.write_text(self.code)
        saved['code'] = str(code_path)

        # Save metadata
        metadata = {
            "model_type": self.model_type.value,
            "task_type": self.task_type.value,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "timestamp": datetime.now().isoformat(),
        }
        meta_path = directory / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        saved['metadata'] = str(meta_path)

        return saved

    @classmethod
    def load(cls, directory: str) -> "TrainingResult":
        """Load a saved training result."""
        import json

        directory = Path(directory)

        # Load model
        with open(directory / "model.pkl", 'rb') as f:
            model = pickle.load(f)

        # Load metadata
        with open(directory / "metadata.json", 'r') as f:
            metadata = json.load(f)

        # Load scaler if exists
        scaler = None
        scaler_path = directory / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

        # Load code
        code_path = directory / "train.py"
        code = code_path.read_text() if code_path.exists() else ""

        # Reconstruct (simplified - report would need to be regenerated)
        return cls(
            model=model,
            model_type=ModelType(metadata["model_type"]),
            task_type=TaskType(metadata["task_type"]),
            report=AnalystReport(
                model_type=ModelType(metadata["model_type"]),
                task_type=TaskType(metadata["task_type"]),
            ),
            feature_names=metadata.get("feature_names", []),
            target_name=metadata.get("target_name", ""),
            scaler=scaler,
            _code=code,
        )


class Analyst:
    """
    Main ML training service.

    Usage:
        analyst = Analyst()

        # Train with auto-selection
        result = analyst.train(
            data=df,
            target='outcome',
            intent='predict customer churn based on usage patterns',
        )

        # Get model
        model = result.model

        # Get educational report
        print(result.report.to_markdown())

        # Get reproducible code
        print(result.code)

        # Save everything
        result.save('output/')
    """

    def __init__(self):
        self.selector = ModelSelector()
        self.reporter = AnalystReporter()

    def train(
        self,
        data,
        target: str,
        intent: str = "",
        model_type: ModelType = None,
        priority: str = "balanced",
        test_size: float = 0.2,
        hyperparameters: dict = None,
        scale_features: bool = True,
        encode_categorical: bool = True,
        random_state: int = 42,
        eda_report: EDAReport = None,
        run_eda: bool = True,
        force_training: bool = False,
    ) -> TrainingResult:
        """
        Train a model on the data.

        Args:
            data: pandas DataFrame
            target: Name of target column
            intent: Description of what you're trying to predict (for documentation)
            model_type: Specific model to use (None = auto-select)
            priority: "accuracy", "interpretability", "speed", or "balanced"
            test_size: Fraction of data for testing
            hyperparameters: Override model hyperparameters
            scale_features: Whether to scale numeric features
            encode_categorical: Whether to encode categorical features
            random_state: Random seed for reproducibility
            eda_report: Pre-computed EDA report (avoids re-running)
            run_eda: Whether to run EDA if not provided (default True)
            force_training: Train even if data quality is poor (default False)

        Returns:
            TrainingResult with model, report, code, and data quality assessment

        Raises:
            ValueError: If data quality is poor and force_training=False
        """
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        hyperparameters = hyperparameters or {}

        # === DATA QUALITY CHECK ===
        data_quality = None
        if eda_report is not None:
            data_quality = assess_data_quality(eda_report, target)
        elif run_eda:
            from .eda import quick_eda
            eda_report = quick_eda(data, name="training_data", generate_charts=False)
            data_quality = assess_data_quality(eda_report, target)

        # Warn or block on poor data quality
        if data_quality and not data_quality.training_recommended:
            if not force_training:
                raise ValueError(
                    f"Data quality is POOR - training not recommended.\n"
                    f"{data_quality.summary()}\n\n"
                    f"Use force_training=True to override, or address data quality issues first."
                )

        # Profile the data
        profile = DataProfile.from_dataframe(data, target)
        task_type = TaskType.CLASSIFICATION if profile.n_classes > 0 else TaskType.REGRESSION

        # Auto-select model if not specified
        if model_type is None:
            priority_enum = Priority(priority)
            recommendations = self.selector.recommend(profile, priority_enum)
            model_type = recommendations[0].model_type if recommendations else ModelType.RANDOM_FOREST

        # Prepare data
        X = data.drop(columns=[target])
        y = data[target]
        feature_names = list(X.columns)

        # Encode categorical features
        encoder = None
        if encode_categorical:
            X, encoder = self._encode_categorical(X)

        # Convert to numpy
        X = X.values if hasattr(X, 'values') else np.array(X)
        y = y.values if hasattr(y, 'values') else np.array(y)

        # Encode target for classification if needed
        target_encoder = None
        if task_type == TaskType.CLASSIFICATION and not np.issubdtype(y.dtype, np.number):
            from sklearn.preprocessing import LabelEncoder
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale features
        scaler = None
        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Create model
        model = create_model(model_type, task_type, **hyperparameters)

        # Train
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Generate report
        report = self.reporter.create_report(
            model=model,
            model_type=model_type,
            task_type=task_type,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            target_name=target,
            training_time=training_time,
            data_quality=data_quality,
        )

        # Generate reproducible code
        code = self._generate_code(
            model_type=model_type,
            task_type=task_type,
            feature_names=feature_names,
            target_name=target,
            hyperparameters=hyperparameters,
            scale_features=scale_features,
            test_size=test_size,
            random_state=random_state,
            intent=intent,
        )

        return TrainingResult(
            model=model,
            model_type=model_type,
            task_type=task_type,
            report=report,
            feature_names=feature_names,
            target_name=target,
            scaler=scaler,
            encoder=encoder,
            _code=code,
            data_quality=data_quality,
        )

    def _encode_categorical(self, X):
        """Encode categorical columns."""
        import pandas as pd

        if not isinstance(X, pd.DataFrame):
            return X, None

        X = X.copy()
        encoder = {'columns': [], 'mappings': {}}

        for col in X.select_dtypes(include=['object', 'category']).columns:
            # Create mapping
            unique_values = X[col].unique()
            mapping = {v: i for i, v in enumerate(unique_values) if pd.notna(v)}
            mapping[None] = -1

            X[col] = X[col].map(mapping).fillna(-1).astype(int)
            encoder['columns'].append(col)
            encoder['mappings'][col] = mapping

        return X, encoder if encoder['columns'] else None

    def _generate_code(
        self,
        model_type: ModelType,
        task_type: TaskType,
        feature_names: list[str],
        target_name: str,
        hyperparameters: dict,
        scale_features: bool,
        test_size: float,
        random_state: int,
        intent: str,
    ) -> str:
        """Generate reproducible Python code."""

        info = get_model_info(model_type)

        # Model import and class
        model_imports = {
            ModelType.LINEAR_REGRESSION: ("from sklearn.linear_model import LinearRegression", "LinearRegression"),
            ModelType.LOGISTIC_REGRESSION: ("from sklearn.linear_model import LogisticRegression", "LogisticRegression"),
            ModelType.RANDOM_FOREST: (
                "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor",
                "RandomForestClassifier" if task_type == TaskType.CLASSIFICATION else "RandomForestRegressor"
            ),
            ModelType.SVM: (
                "from sklearn.svm import SVC, SVR",
                "SVC" if task_type == TaskType.CLASSIFICATION else "SVR"
            ),
            ModelType.KNN: (
                "from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor",
                "KNeighborsClassifier" if task_type == TaskType.CLASSIFICATION else "KNeighborsRegressor"
            ),
            ModelType.NAIVE_BAYES: ("from sklearn.naive_bayes import GaussianNB", "GaussianNB"),
            ModelType.GRADIENT_BOOSTING: (
                "from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor",
                "GradientBoostingClassifier" if task_type == TaskType.CLASSIFICATION else "GradientBoostingRegressor"
            ),
        }

        import_stmt, model_class = model_imports[model_type]

        # Build hyperparameters string
        hp_parts = [f"random_state={random_state}"]
        for k, v in hyperparameters.items():
            if isinstance(v, str):
                hp_parts.append(f"{k}='{v}'")
            else:
                hp_parts.append(f"{k}={v}")
        hp_str = ", ".join(hp_parts)

        # Generate code
        code = f'''"""
{info.name} Training Script

{f"Intent: {intent}" if intent else ""}

Model: {model_type.value}
Task: {task_type.value}
Features: {feature_names}
Target: {target_name}

Generated by Svend Analyst
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
{"from sklearn.preprocessing import StandardScaler" if scale_features else ""}
{import_stmt}
from sklearn.metrics import {"accuracy_score, classification_report" if task_type == TaskType.CLASSIFICATION else "mean_squared_error, r2_score"}


# === Load your data ===
# df = pd.read_csv('your_data.csv')

# For demonstration, create sample data:
# df = pd.DataFrame(...)


def train_model(df, target_column='{target_name}'):
    """Train the model and return it with metrics."""

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle categorical columns
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = pd.Categorical(X[col]).codes

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size={test_size}, random_state={random_state}
    )
'''

        if scale_features:
            code += '''
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
'''

        code += f'''
    # Create and train model
    model = {model_class}({hp_str})
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
'''

        if task_type == TaskType.CLASSIFICATION:
            code += '''
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\\nClassification Report:")
    print(classification_report(y_test, y_pred))
'''
        else:
            code += '''
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {np.sqrt(mse):.4f}")
    print(f"R² Score: {r2:.4f}")
'''

        code += f'''
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({{
            'feature': {feature_names},
            'importance': model.feature_importances_
        }}).sort_values('importance', ascending=False)
        print("\\nTop Features:")
        print(importance.head(10))
    elif hasattr(model, 'coef_'):
        importance = pd.DataFrame({{
            'feature': {feature_names},
            'importance': np.abs(model.coef_).flatten() if model.coef_.ndim > 1 else np.abs(model.coef_)
        }}).sort_values('importance', ascending=False)
        print("\\nTop Features (by coefficient magnitude):")
        print(importance.head(10))

    return model{"" if not scale_features else ", scaler"}


if __name__ == "__main__":
    # Load your data here
    # df = pd.read_csv('data.csv')

    # Example with sample data
    print("Replace this with your actual data loading")
    print("Example: df = pd.read_csv('your_data.csv')")
    print("Then: model = train_model(df)")
'''

        return code

    def compare_models(
        self,
        data,
        target: str,
        models: list[ModelType] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> list[tuple[ModelType, AnalystReport]]:
        """
        Compare multiple models on the same data.

        Returns list of (ModelType, AnalystReport) sorted by performance.
        """
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        # Profile data
        profile = DataProfile.from_dataframe(data, target)
        task_type = TaskType.CLASSIFICATION if profile.n_classes > 0 else TaskType.REGRESSION

        # Default to all compatible models
        if models is None:
            from .models import get_models_for_task
            models = get_models_for_task(task_type)

        # Prepare data
        X = data.drop(columns=[target])
        y = data[target]
        feature_names = list(X.columns)

        X, encoder = self._encode_categorical(X)
        X = X.values if hasattr(X, 'values') else np.array(X)
        y = y.values if hasattr(y, 'values') else np.array(y)

        if task_type == TaskType.CLASSIFICATION and not np.issubdtype(y.dtype, np.number):
            from sklearn.preprocessing import LabelEncoder
            y = LabelEncoder().fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train each model
        results = []
        for model_type in models:
            try:
                model = create_model(model_type, task_type)
                model.fit(X_train_scaled, y_train)

                report = self.reporter.create_report(
                    model=model,
                    model_type=model_type,
                    task_type=task_type,
                    X_train=X_train_scaled,
                    y_train=y_train,
                    X_test=X_test_scaled,
                    y_test=y_test,
                    feature_names=feature_names,
                    target_name=target,
                )
                results.append((model_type, report))
            except Exception as e:
                print(f"Failed to train {model_type.value}: {e}")

        # Sort by primary metric
        def get_score(item):
            model_type, report = item
            if report.metrics:
                # Use accuracy for classification, R² for regression
                for m in report.metrics:
                    if m.name in ["Accuracy", "R² Score"]:
                        return m.value
            return 0.0

        results.sort(key=get_score, reverse=True)

        return results


def quick_train(df, target: str, intent: str = "", **kwargs) -> TrainingResult:
    """
    Quick helper to train a model.

    Args:
        df: pandas DataFrame
        target: target column name
        intent: what you're trying to predict
        **kwargs: passed to Analyst.train()

    Returns:
        TrainingResult
    """
    analyst = Analyst()
    return analyst.train(df, target=target, intent=intent, **kwargs)
