"""
Factor Normalization

Standardize categorical values:
- Case normalization (Sales, SALES, sales → Sales)
- Whitespace trimming
- Fuzzy matching for typos (Slaes → Sales)
- Custom mappings

Reports all changes with before/after.
"""

import re
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class NormalizationChange:
    """A single normalization change."""
    row_index: int
    column: str
    original: str
    normalized: str
    method: str


@dataclass
class NormalizationResult:
    """Result of factor normalization."""
    changes: list[NormalizationChange] = field(default_factory=list)
    mappings: dict[str, dict[str, str]] = field(default_factory=dict)  # column -> {original: normalized}

    @property
    def total_changes(self) -> int:
        return len(self.changes)

    def by_column(self) -> dict[str, int]:
        counts = {}
        for change in self.changes:
            counts[change.column] = counts.get(change.column, 0) + 1
        return counts

    def summary(self) -> str:
        lines = [f"Normalization: {self.total_changes} changes"]
        by_col = self.by_column()
        for col, count in by_col.items():
            lines.append(f"  {col}: {count} values normalized")
            if col in self.mappings:
                for orig, norm in list(self.mappings[col].items())[:5]:
                    lines.append(f"    '{orig}' → '{norm}'")
                if len(self.mappings[col]) > 5:
                    lines.append(f"    ... and {len(self.mappings[col]) - 5} more")
        return "\n".join(lines)


class FactorNormalizer:
    """
    Normalize categorical factors for consistency.

    Usage:
        normalizer = FactorNormalizer()
        df_clean, result = normalizer.normalize(df, columns=['department', 'status'])
        print(result.summary())
    """

    def __init__(
        self,
        fuzzy_threshold: float = 0.85,  # Minimum similarity for fuzzy match
        min_frequency: int = 2,  # Minimum occurrences to be considered "canonical"
    ):
        self.fuzzy_threshold = fuzzy_threshold
        self.min_frequency = min_frequency

    def normalize(
        self,
        df,
        columns: list[str] = None,
        custom_mappings: dict[str, dict[str, str]] = None,
        case_style: str = "title",  # title, upper, lower, preserve
    ) -> tuple:
        """
        Normalize categorical columns.

        Args:
            df: pandas DataFrame
            columns: Columns to normalize (default: object columns)
            custom_mappings: Dict of column -> {original: normalized}
            case_style: How to handle case

        Returns:
            (cleaned_df, NormalizationResult)
        """
        import pandas as pd

        df = df.copy()
        custom_mappings = custom_mappings or {}

        result = NormalizationResult()

        # Default to object/string columns
        if columns is None:
            columns = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

        for col in columns:
            if col not in df.columns:
                continue

            # Get custom mapping if provided
            col_mapping = custom_mappings.get(col, {})

            # Normalize this column
            changes, final_mapping = self._normalize_column(
                df, col, col_mapping, case_style
            )

            result.changes.extend(changes)
            if final_mapping:
                result.mappings[col] = final_mapping

        return df, result

    def _normalize_column(
        self,
        df,
        column: str,
        custom_mapping: dict,
        case_style: str,
    ) -> tuple[list[NormalizationChange], dict]:
        """Normalize a single column."""
        import pandas as pd

        changes = []
        final_mapping = {}

        # Get unique values and their frequencies
        value_counts = df[column].value_counts()
        unique_values = df[column].dropna().unique()

        # Build canonical forms
        canonical = self._build_canonical_forms(unique_values, value_counts, case_style)

        # Apply custom mappings (override canonical)
        canonical.update(custom_mapping)

        # Try fuzzy matching for remaining values
        canonical = self._add_fuzzy_matches(unique_values, canonical)

        # Apply normalization
        for idx, value in df[column].items():
            if pd.isna(value):
                continue

            # Clean whitespace first
            cleaned = str(value).strip()
            cleaned = re.sub(r'\s+', ' ', cleaned)

            # Apply case normalization
            if case_style == "title":
                normalized = cleaned.title()
            elif case_style == "upper":
                normalized = cleaned.upper()
            elif case_style == "lower":
                normalized = cleaned.lower()
            else:
                normalized = cleaned

            # Check for mapping
            lookup_key = cleaned.lower()
            if lookup_key in canonical:
                normalized = canonical[lookup_key]

            # Record change if different
            if str(value) != normalized:
                changes.append(NormalizationChange(
                    row_index=idx,
                    column=column,
                    original=str(value),
                    normalized=normalized,
                    method="case" if str(value).lower() == normalized.lower() else "mapping",
                ))
                final_mapping[str(value)] = normalized

        # Apply all changes at once (handles categorical dtype properly)
        if final_mapping:
            # Convert categorical to object if needed
            if pd.api.types.is_categorical_dtype(df[column]):
                df[column] = df[column].astype(str)

            # Apply mapping
            df[column] = df[column].apply(
                lambda x: final_mapping.get(str(x), x) if pd.notna(x) else x
            )

        return changes, final_mapping

    def _build_canonical_forms(
        self,
        values,
        value_counts,
        case_style: str,
    ) -> dict:
        """Build mapping of lowercase -> canonical form."""
        canonical = {}

        # Group values by lowercase form
        groups = {}
        for val in values:
            if val is None:
                continue
            key = str(val).strip().lower()
            if key not in groups:
                groups[key] = []
            groups[key].append(str(val))

        # For each group, choose canonical form
        for key, variants in groups.items():
            if len(variants) == 1:
                # Only one form, use it (with case normalization)
                canonical[key] = self._apply_case(variants[0], case_style)
            else:
                # Multiple forms - pick most frequent
                best = max(variants, key=lambda v: value_counts.get(v, 0))
                canonical[key] = self._apply_case(best, case_style)

        return canonical

    def _apply_case(self, value: str, case_style: str) -> str:
        """Apply case style to value."""
        value = value.strip()
        if case_style == "title":
            return value.title()
        elif case_style == "upper":
            return value.upper()
        elif case_style == "lower":
            return value.lower()
        return value

    def _add_fuzzy_matches(self, values, canonical: dict) -> dict:
        """Add fuzzy matches for typos."""
        try:
            from difflib import SequenceMatcher
        except ImportError:
            return canonical

        # Get canonical values (the targets)
        targets = list(set(canonical.values()))

        for val in values:
            if val is None:
                continue

            key = str(val).strip().lower()
            if key in canonical:
                continue  # Already mapped

            # Try to fuzzy match to existing canonical values
            best_match = None
            best_score = 0

            for target in targets:
                score = SequenceMatcher(None, key, target.lower()).ratio()
                if score > best_score and score >= self.fuzzy_threshold:
                    best_score = score
                    best_match = target

            if best_match:
                canonical[key] = best_match

        return canonical

    def suggest_mappings(self, df, column: str) -> dict[str, list[str]]:
        """
        Suggest possible normalizations without applying them.

        Returns dict of canonical -> [variants]
        """
        import pandas as pd

        if column not in df.columns:
            return {}

        value_counts = df[column].value_counts()
        unique_values = df[column].dropna().unique()

        # Group by lowercase
        groups = {}
        for val in unique_values:
            key = str(val).strip().lower()
            if key not in groups:
                groups[key] = []
            groups[key].append(str(val))

        # Return groups with multiple variants
        suggestions = {}
        for key, variants in groups.items():
            if len(variants) > 1:
                # Pick most frequent as canonical
                best = max(variants, key=lambda v: value_counts.get(v, 0))
                suggestions[best] = [v for v in variants if v != best]

        return suggestions
