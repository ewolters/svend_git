"""
Design of Experiments (DOE)

Deterministic experiment design generation:
- Full factorial designs
- Fractional factorial designs
- Response surface designs (CCD, Box-Behnken)
- Latin squares
- Randomization

All outputs are structured and reproducible.
"""

import itertools
import random
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class Factor:
    """A factor (independent variable) in an experiment."""
    name: str
    levels: list[Any]  # The actual level values
    level_names: list[str] = None  # Optional human-readable names
    units: str = ""
    is_categorical: bool = False

    def __post_init__(self):
        if self.level_names is None:
            self.level_names = [str(lvl) for lvl in self.levels]

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    def coded_levels(self) -> list[int]:
        """Return coded levels (-1, 0, +1 or 0, 1, 2...)."""
        n = len(self.levels)
        if n == 2:
            return [-1, 1]
        elif n == 3:
            return [-1, 0, 1]
        else:
            return list(range(n))


@dataclass
class ExperimentRun:
    """A single experimental run (row in design matrix)."""
    run_id: int
    standard_order: int
    run_order: int  # Randomized order
    factor_levels: dict[str, Any]  # factor_name -> level value
    coded_levels: dict[str, int]  # factor_name -> coded level
    block: int = 1
    replicate: int = 1
    is_center_point: bool = False


@dataclass
class ExperimentDesign:
    """Complete experimental design."""
    name: str
    design_type: str
    factors: list[Factor]
    runs: list[ExperimentRun]

    # Design properties
    resolution: int = None  # For fractional factorials (III, IV, V)
    num_blocks: int = 1
    num_replicates: int = 1
    num_center_points: int = 0

    # Metadata
    notes: list[str] = field(default_factory=list)

    @property
    def num_runs(self) -> int:
        return len(self.runs)

    @property
    def factor_names(self) -> list[str]:
        return [f.name for f in self.factors]

    def to_matrix(self, coded: bool = False) -> list[list]:
        """Convert to design matrix (list of lists)."""
        matrix = []
        for run in sorted(self.runs, key=lambda r: r.run_order):
            if coded:
                row = [run.coded_levels[f.name] for f in self.factors]
            else:
                row = [run.factor_levels[f.name] for f in self.factors]
            matrix.append(row)
        return matrix

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            "name": self.name,
            "design_type": self.design_type,
            "factors": [
                {
                    "name": f.name,
                    "levels": f.levels,
                    "level_names": f.level_names,
                    "units": f.units,
                }
                for f in self.factors
            ],
            "runs": [
                {
                    "run_id": r.run_id,
                    "standard_order": r.standard_order,
                    "run_order": r.run_order,
                    "levels": r.factor_levels,
                    "coded": r.coded_levels,
                    "block": r.block,
                    "replicate": r.replicate,
                    "center_point": r.is_center_point,
                }
                for r in self.runs
            ],
            "properties": {
                "num_runs": self.num_runs,
                "resolution": self.resolution,
                "num_blocks": self.num_blocks,
                "num_replicates": self.num_replicates,
                "num_center_points": self.num_center_points,
            },
            "notes": self.notes,
        }

    def to_markdown(self) -> str:
        """Export as markdown table."""
        lines = [
            f"# {self.name}",
            "",
            f"**Design Type:** {self.design_type}",
            f"**Runs:** {self.num_runs}",
            "",
        ]

        if self.resolution:
            lines.append(f"**Resolution:** {self._roman(self.resolution)}")

        # Factors table
        lines.extend([
            "",
            "## Factors",
            "",
            "| Factor | Levels | Units |",
            "|--------|--------|-------|",
        ])
        for f in self.factors:
            levels_str = ", ".join(str(l) for l in f.levels)
            lines.append(f"| {f.name} | {levels_str} | {f.units} |")

        # Design matrix
        lines.extend([
            "",
            "## Design Matrix",
            "",
        ])

        # Header
        header = "| Run | " + " | ".join(f.name for f in self.factors) + " |"
        separator = "|-----|" + "|".join("---" for _ in self.factors) + "|"
        lines.append(header)
        lines.append(separator)

        # Rows (in run order)
        for run in sorted(self.runs, key=lambda r: r.run_order):
            values = [str(run.factor_levels[f.name]) for f in self.factors]
            marker = " *" if run.is_center_point else ""
            lines.append(f"| {run.run_order}{marker} | " + " | ".join(values) + " |")

        if self.num_center_points > 0:
            lines.append("")
            lines.append("*\\* Center point*")

        # Notes
        if self.notes:
            lines.extend(["", "## Notes", ""])
            for note in self.notes:
                lines.append(f"- {note}")

        return "\n".join(lines)

    def _roman(self, n: int) -> str:
        """Convert integer to Roman numeral."""
        numerals = [(5, 'V'), (4, 'IV'), (3, 'III'), (2, 'II'), (1, 'I')]
        result = ""
        for value, numeral in numerals:
            while n >= value:
                result += numeral
                n -= value
        return result


class DOEGenerator:
    """
    Generate experimental designs.

    All methods are deterministic given the same inputs and random seed.
    """

    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)

    def full_factorial(self, factors: list[Factor], replicates: int = 1,
                       center_points: int = 0, randomize: bool = True) -> ExperimentDesign:
        """
        Generate a full factorial design.

        Args:
            factors: List of factors with their levels
            replicates: Number of replicates for each run
            center_points: Number of center points (only for numeric factors)
            randomize: Whether to randomize run order
        """
        # Generate all combinations
        level_lists = [f.levels for f in factors]
        combinations = list(itertools.product(*level_lists))

        runs = []
        run_id = 1
        standard_order = 1

        for rep in range(1, replicates + 1):
            for combo in combinations:
                factor_levels = {f.name: combo[i] for i, f in enumerate(factors)}
                coded = {f.name: f.coded_levels()[f.levels.index(combo[i])]
                        for i, f in enumerate(factors)}

                runs.append(ExperimentRun(
                    run_id=run_id,
                    standard_order=standard_order,
                    run_order=run_id,  # Will be randomized later
                    factor_levels=factor_levels,
                    coded_levels=coded,
                    replicate=rep,
                ))
                run_id += 1
                standard_order += 1

        # Add center points (for 2-level numeric factors)
        if center_points > 0:
            numeric_factors = [f for f in factors if not f.is_categorical and f.num_levels == 2]
            if numeric_factors:
                center_levels = {}
                center_coded = {}
                for f in factors:
                    if f in numeric_factors:
                        # Midpoint between levels
                        center_levels[f.name] = (f.levels[0] + f.levels[1]) / 2
                        center_coded[f.name] = 0
                    else:
                        # Use first level for categorical
                        center_levels[f.name] = f.levels[0]
                        center_coded[f.name] = 0

                for _ in range(center_points):
                    runs.append(ExperimentRun(
                        run_id=run_id,
                        standard_order=run_id,
                        run_order=run_id,
                        factor_levels=center_levels.copy(),
                        coded_levels=center_coded.copy(),
                        is_center_point=True,
                    ))
                    run_id += 1

        # Randomize
        if randomize:
            run_orders = list(range(1, len(runs) + 1))
            self.rng.shuffle(run_orders)
            for i, run in enumerate(runs):
                run.run_order = run_orders[i]

        # Calculate design info
        num_factors = len(factors)
        total_runs = len(combinations) * replicates + center_points

        design = ExperimentDesign(
            name=f"{num_factors}-Factor Full Factorial",
            design_type=f"2^{num_factors} Full Factorial" if all(f.num_levels == 2 for f in factors)
                        else f"Full Factorial ({' x '.join(str(f.num_levels) for f in factors)})",
            factors=factors,
            runs=runs,
            resolution=num_factors,  # Full resolution
            num_replicates=replicates,
            num_center_points=center_points,
        )

        # Add notes
        design.notes.append(f"Total runs: {total_runs}")
        design.notes.append(f"Full factorial - all main effects and interactions estimable")

        # Add efficiency explanation for 2-level designs
        if all(f.num_levels == 2 for f in factors):
            design.notes.append(
                f"Note: A 2^{num_factors} design tests factor HIGH/LOW combinations efficiently. "
                f"If your process has more settings per factor, consider adding center points "
                f"or using a response surface design to capture curvature."
            )

        return design

    def fractional_factorial(self, factors: list[Factor], resolution: int = 4,
                             randomize: bool = True) -> ExperimentDesign:
        """
        Generate a fractional factorial design.

        Args:
            factors: List of 2-level factors
            resolution: Minimum resolution (3, 4, or 5)
            randomize: Whether to randomize run order
        """
        n = len(factors)

        # Determine fraction based on resolution
        # Resolution III: main effects confounded with 2FI
        # Resolution IV: main effects clear of 2FI, 2FI confounded with each other
        # Resolution V: main effects and 2FI clear

        if n <= 4:
            # Use full factorial for small designs
            return self.full_factorial(factors, randomize=randomize)

        # Standard 2^(n-p) designs
        # These are common fractional factorial designs
        designs = {
            5: {3: 4, 4: 3, 5: 1},   # 2^5: 16, 8, or 32 runs
            6: {3: 3, 4: 2, 5: 1},   # 2^6: 8, 16, or 32 runs
            7: {3: 4, 4: 3, 5: 2},   # 2^7: 8, 16, or 32 runs
            8: {3: 4, 4: 4, 5: 3},   # 2^8: 16, 16, or 32 runs
        }

        p = designs.get(n, {}).get(resolution, max(0, n - 5))
        num_runs = 2 ** (n - p)

        # Generate base design (first n-p factors as full factorial)
        base_factors = factors[:n-p]
        base_design = self.full_factorial(base_factors, randomize=False)

        # Add remaining factors as generators (confounded with interactions)
        runs = []
        for i, base_run in enumerate(base_design.runs):
            factor_levels = base_run.factor_levels.copy()
            coded_levels = base_run.coded_levels.copy()

            # Generate remaining factors from interactions
            base_coded = [coded_levels[f.name] for f in base_factors]

            for j, extra_factor in enumerate(factors[n-p:]):
                # Use product of some base factors as generator
                # This is simplified - real designs use specific generators
                generator_indices = [(j + k) % (n - p) for k in range(2)]
                generated_level = 1
                for idx in generator_indices:
                    generated_level *= base_coded[idx]

                coded_levels[extra_factor.name] = generated_level
                level_idx = 0 if generated_level == -1 else 1
                factor_levels[extra_factor.name] = extra_factor.levels[level_idx]

            runs.append(ExperimentRun(
                run_id=i + 1,
                standard_order=i + 1,
                run_order=i + 1,
                factor_levels=factor_levels,
                coded_levels=coded_levels,
            ))

        # Randomize
        if randomize:
            run_orders = list(range(1, len(runs) + 1))
            self.rng.shuffle(run_orders)
            for i, run in enumerate(runs):
                run.run_order = run_orders[i]

        design = ExperimentDesign(
            name=f"{n}-Factor Fractional Factorial",
            design_type=f"2^({n}-{p}) Fractional Factorial",
            factors=factors,
            runs=runs,
            resolution=resolution,
        )

        resolution_names = {3: "III", 4: "IV", 5: "V"}
        design.notes.append(f"Resolution {resolution_names.get(resolution, resolution)}")
        design.notes.append(f"Runs: {num_runs} (vs {2**n} for full factorial)")

        if resolution == 3:
            design.notes.append("Warning: Main effects confounded with 2-factor interactions")
        elif resolution == 4:
            design.notes.append("Main effects clear; 2FI confounded with each other")
        else:
            design.notes.append("Main effects and 2-factor interactions estimable")

        return design

    def central_composite(self, factors: list[Factor], alpha: str = "rotatable",
                          center_points: int = 5, randomize: bool = True) -> ExperimentDesign:
        """
        Generate a Central Composite Design (CCD) for response surface methodology.

        Args:
            factors: List of continuous 2-level factors
            alpha: Axial distance - "rotatable", "orthogonal", or numeric value
            center_points: Number of center points
            randomize: Whether to randomize
        """
        n = len(factors)

        # Calculate alpha
        if alpha == "rotatable":
            alpha_val = (2 ** n) ** 0.25
        elif alpha == "orthogonal":
            alpha_val = ((2 ** n) ** 0.5 * (2 ** n + 2 * n + center_points) ** 0.5 - 2 ** n) / (2 * n) ** 0.5
        else:
            alpha_val = float(alpha)

        runs = []
        run_id = 1

        # Factorial points (2^n)
        level_combos = list(itertools.product([-1, 1], repeat=n))
        for combo in level_combos:
            factor_levels = {}
            coded_levels = {}
            for i, f in enumerate(factors):
                coded_levels[f.name] = combo[i]
                # Convert coded to actual
                low, high = f.levels[0], f.levels[1]
                mid = (low + high) / 2
                half_range = (high - low) / 2
                factor_levels[f.name] = mid + combo[i] * half_range

            runs.append(ExperimentRun(
                run_id=run_id,
                standard_order=run_id,
                run_order=run_id,
                factor_levels=factor_levels,
                coded_levels=coded_levels,
            ))
            run_id += 1

        # Axial (star) points (2n)
        for i, f in enumerate(factors):
            for direction in [-alpha_val, alpha_val]:
                factor_levels = {}
                coded_levels = {}
                for j, f2 in enumerate(factors):
                    if i == j:
                        coded_levels[f2.name] = direction
                    else:
                        coded_levels[f2.name] = 0

                    # Convert to actual
                    low, high = f2.levels[0], f2.levels[1]
                    mid = (low + high) / 2
                    half_range = (high - low) / 2
                    factor_levels[f2.name] = mid + coded_levels[f2.name] * half_range

                runs.append(ExperimentRun(
                    run_id=run_id,
                    standard_order=run_id,
                    run_order=run_id,
                    factor_levels=factor_levels,
                    coded_levels=coded_levels,
                ))
                run_id += 1

        # Center points
        center_levels = {}
        center_coded = {}
        for f in factors:
            low, high = f.levels[0], f.levels[1]
            center_levels[f.name] = (low + high) / 2
            center_coded[f.name] = 0

        for _ in range(center_points):
            runs.append(ExperimentRun(
                run_id=run_id,
                standard_order=run_id,
                run_order=run_id,
                factor_levels=center_levels.copy(),
                coded_levels=center_coded.copy(),
                is_center_point=True,
            ))
            run_id += 1

        # Randomize
        if randomize:
            run_orders = list(range(1, len(runs) + 1))
            self.rng.shuffle(run_orders)
            for i, run in enumerate(runs):
                run.run_order = run_orders[i]

        design = ExperimentDesign(
            name=f"{n}-Factor Central Composite Design",
            design_type="CCD (Response Surface)",
            factors=factors,
            runs=runs,
            num_center_points=center_points,
        )

        design.notes.append(f"Alpha (axial distance): {alpha_val:.3f}")
        design.notes.append(f"Factorial points: {2**n}")
        design.notes.append(f"Axial points: {2*n}")
        design.notes.append(f"Center points: {center_points}")
        design.notes.append("Allows estimation of quadratic effects")

        return design

    def latin_square(self, treatments: list[str], rows: list[str] = None,
                     cols: list[str] = None, randomize: bool = True) -> ExperimentDesign:
        """
        Generate a Latin Square design.

        Args:
            treatments: List of treatment names
            rows: Row block names (default: Row 1, Row 2, ...)
            cols: Column block names (default: Col 1, Col 2, ...)
            randomize: Whether to randomize assignment
        """
        n = len(treatments)

        if rows is None:
            rows = [f"Row {i+1}" for i in range(n)]
        if cols is None:
            cols = [f"Col {i+1}" for i in range(n)]

        if len(rows) != n or len(cols) != n:
            raise ValueError("Latin square requires equal number of treatments, rows, and columns")

        # Generate standard Latin square
        # Each treatment appears once in each row and column
        square = [[treatments[(i + j) % n] for j in range(n)] for i in range(n)]

        # Randomize rows and columns if requested
        if randomize:
            # Shuffle row order
            row_order = list(range(n))
            self.rng.shuffle(row_order)
            square = [square[i] for i in row_order]

            # Shuffle column order
            col_order = list(range(n))
            self.rng.shuffle(col_order)
            square = [[row[j] for j in col_order] for row in square]

        # Create runs
        runs = []
        run_id = 1

        for i in range(n):
            for j in range(n):
                runs.append(ExperimentRun(
                    run_id=run_id,
                    standard_order=run_id,
                    run_order=run_id,
                    factor_levels={
                        "Row": rows[i],
                        "Column": cols[j],
                        "Treatment": square[i][j],
                    },
                    coded_levels={
                        "Row": i,
                        "Column": j,
                        "Treatment": treatments.index(square[i][j]),
                    },
                ))
                run_id += 1

        # Create factors for the design
        factors = [
            Factor("Row", rows, is_categorical=True),
            Factor("Column", cols, is_categorical=True),
            Factor("Treatment", treatments, is_categorical=True),
        ]

        design = ExperimentDesign(
            name=f"{n}x{n} Latin Square",
            design_type="Latin Square",
            factors=factors,
            runs=runs,
        )

        design.notes.append(f"Blocks two sources of variation (row and column)")
        design.notes.append(f"Each treatment appears exactly once per row and column")

        return design

    def randomized_block(self, treatments: list[str], blocks: int,
                         replicates_per_block: int = 1,
                         randomize: bool = True) -> ExperimentDesign:
        """
        Generate a Randomized Complete Block Design (RCBD).

        Args:
            treatments: List of treatment names
            blocks: Number of blocks
            replicates_per_block: Replicates of each treatment per block
            randomize: Whether to randomize within blocks
        """
        runs = []
        run_id = 1

        for block in range(1, blocks + 1):
            block_runs = []
            for treatment in treatments:
                for rep in range(replicates_per_block):
                    block_runs.append({
                        "treatment": treatment,
                        "replicate": rep + 1,
                    })

            # Randomize within block
            if randomize:
                self.rng.shuffle(block_runs)

            for run_info in block_runs:
                runs.append(ExperimentRun(
                    run_id=run_id,
                    standard_order=run_id,
                    run_order=run_id,
                    factor_levels={
                        "Block": f"Block {block}",
                        "Treatment": run_info["treatment"],
                    },
                    coded_levels={
                        "Block": block,
                        "Treatment": treatments.index(run_info["treatment"]),
                    },
                    block=block,
                    replicate=run_info["replicate"],
                ))
                run_id += 1

        factors = [
            Factor("Block", [f"Block {i+1}" for i in range(blocks)], is_categorical=True),
            Factor("Treatment", treatments, is_categorical=True),
        ]

        design = ExperimentDesign(
            name=f"Randomized Complete Block Design",
            design_type="RCBD",
            factors=factors,
            runs=runs,
            num_blocks=blocks,
            num_replicates=replicates_per_block,
        )

        design.notes.append(f"Treatments: {len(treatments)}")
        design.notes.append(f"Blocks: {blocks}")
        design.notes.append(f"Total runs: {len(runs)}")

        return design

    def d_optimal(
        self,
        factors: list[Factor],
        num_runs: int,
        model: str = "linear",  # linear, quadratic, interaction
        seed: int = None,
    ) -> ExperimentDesign:
        """
        Generate D-optimal design using coordinate exchange algorithm.

        D-optimality maximizes |X'X|, the determinant of the information matrix,
        which minimizes the volume of the confidence ellipsoid for parameters.

        Args:
            factors: List of Factor objects
            num_runs: Number of experiment runs (must be >= number of parameters)
            model: Model type - 'linear', 'interaction', or 'quadratic'
            seed: Random seed for reproducibility
        """
        import numpy as np
        if seed is not None:
            np.random.seed(seed)

        k = len(factors)

        # Calculate number of parameters based on model
        if model == "linear":
            n_params = 1 + k  # intercept + main effects
        elif model == "interaction":
            n_params = 1 + k + k * (k - 1) // 2  # + two-factor interactions
        else:  # quadratic
            n_params = 1 + k + k * (k - 1) // 2 + k  # + quadratic terms

        if num_runs < n_params:
            num_runs = n_params
            # Note: will add warning to notes

        # Generate candidate set - 3^k factorial for each factor
        candidates = []
        levels_per_factor = []
        for f in factors:
            if f.is_categorical:
                levels_per_factor.append(list(range(len(f.levels))))
            else:
                # Use -1, 0, +1 coded levels
                levels_per_factor.append([-1, 0, 1])

        # Build candidate set using itertools
        from itertools import product
        for combo in product(*levels_per_factor):
            candidates.append(list(combo))

        candidates = np.array(candidates)
        n_candidates = len(candidates)

        def build_model_matrix(X_coded, model_type):
            """Build expanded model matrix from coded factor levels."""
            n = X_coded.shape[0]
            k = X_coded.shape[1]
            cols = [np.ones(n)]  # intercept

            # Main effects
            for j in range(k):
                cols.append(X_coded[:, j])

            # Interactions
            if model_type in ["interaction", "quadratic"]:
                for i in range(k):
                    for j in range(i + 1, k):
                        cols.append(X_coded[:, i] * X_coded[:, j])

            # Quadratic
            if model_type == "quadratic":
                for j in range(k):
                    cols.append(X_coded[:, j] ** 2)

            return np.column_stack(cols)

        def d_efficiency(X):
            """Calculate D-efficiency (determinant of X'X)."""
            try:
                XtX = X.T @ X
                return np.linalg.det(XtX)
            except Exception:
                return 0.0

        # Initialize with random points from candidate set
        indices = np.random.choice(n_candidates, num_runs, replace=True)
        design_matrix = candidates[indices].copy()

        # Coordinate exchange algorithm
        max_iterations = 100
        improved = True
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            for i in range(num_runs):
                for j in range(k):
                    current_X = build_model_matrix(design_matrix, model)
                    current_det = d_efficiency(current_X)
                    best_det = current_det
                    best_level = design_matrix[i, j]

                    # Try all candidate levels for this factor
                    for level in levels_per_factor[j]:
                        if level != design_matrix[i, j]:
                            design_matrix[i, j] = level
                            new_X = build_model_matrix(design_matrix, model)
                            new_det = d_efficiency(new_X)

                            if new_det > best_det:
                                best_det = new_det
                                best_level = level
                                improved = True

                    design_matrix[i, j] = best_level

        # Build runs from final design matrix
        runs = []
        for idx, row in enumerate(design_matrix):
            coded = {}
            actual = {}
            for j, f in enumerate(factors):
                coded[f.name] = int(row[j])
                if f.is_categorical:
                    actual[f.name] = f.levels[int(row[j])]
                else:
                    # Map coded to actual: -1->low, 0->mid, 1->high
                    if len(f.levels) == 2:
                        low, high = f.levels[0], f.levels[1]
                        mid = (low + high) / 2
                    else:
                        low, mid, high = f.levels[0], f.levels[len(f.levels)//2], f.levels[-1]

                    if row[j] == -1:
                        actual[f.name] = low
                    elif row[j] == 0:
                        actual[f.name] = mid
                    else:
                        actual[f.name] = high

            run = ExperimentRun(
                run_id=idx + 1,
                standard_order=idx + 1,
                run_order=idx + 1,
                factor_levels=actual,
                coded_levels=coded,
            )
            runs.append(run)

        # Randomize run order
        run_orders = list(range(1, len(runs) + 1))
        np.random.shuffle(run_orders)
        for run, order in zip(runs, run_orders):
            run.run_order = order

        runs.sort(key=lambda r: r.run_order)

        design = ExperimentDesign(
            name=f"D-Optimal Design ({model} model)",
            design_type="d_optimal",
            factors=factors,
            runs=runs,
        )

        # Calculate final D-efficiency
        final_X = build_model_matrix(design_matrix, model)
        final_det = d_efficiency(final_X)

        design.notes.append(f"Model type: {model}")
        design.notes.append(f"Parameters: {n_params}")
        design.notes.append(f"Runs: {num_runs}")
        design.notes.append(f"D-efficiency (|X'X|): {final_det:.4f}")
        design.notes.append(f"Optimization iterations: {iteration}")

        if num_runs == n_params:
            design.notes.append("Saturated design - no degrees of freedom for error")

        return design

    def i_optimal(
        self,
        factors: list[Factor],
        num_runs: int,
        model: str = "linear",
        seed: int = None,
    ) -> ExperimentDesign:
        """
        Generate I-optimal design using coordinate exchange algorithm.

        I-optimality minimizes the average prediction variance over the design
        region, making it ideal for response surface optimization.

        Args:
            factors: List of Factor objects
            num_runs: Number of experiment runs
            model: Model type - 'linear', 'interaction', or 'quadratic'
            seed: Random seed for reproducibility
        """
        import numpy as np
        if seed is not None:
            np.random.seed(seed)

        k = len(factors)

        # Calculate number of parameters
        if model == "linear":
            n_params = 1 + k
        elif model == "interaction":
            n_params = 1 + k + k * (k - 1) // 2
        else:  # quadratic
            n_params = 1 + k + k * (k - 1) // 2 + k

        if num_runs < n_params:
            num_runs = n_params

        # Generate candidate set
        levels_per_factor = []
        for f in factors:
            if f.is_categorical:
                levels_per_factor.append(list(range(len(f.levels))))
            else:
                levels_per_factor.append([-1, 0, 1])

        from itertools import product
        candidates = np.array([list(c) for c in product(*levels_per_factor)])
        n_candidates = len(candidates)

        def build_model_matrix(X_coded, model_type):
            n = X_coded.shape[0]
            k = X_coded.shape[1]
            cols = [np.ones(n)]

            for j in range(k):
                cols.append(X_coded[:, j])

            if model_type in ["interaction", "quadratic"]:
                for i in range(k):
                    for j in range(i + 1, k):
                        cols.append(X_coded[:, i] * X_coded[:, j])

            if model_type == "quadratic":
                for j in range(k):
                    cols.append(X_coded[:, j] ** 2)

            return np.column_stack(cols)

        def i_criterion(X, candidates, model_type):
            """Calculate I-criterion (average prediction variance)."""
            try:
                XtX = X.T @ X
                XtX_inv = np.linalg.inv(XtX)

                # Build model matrix for all candidate points
                X_cand = build_model_matrix(candidates, model_type)

                # Average prediction variance over candidate set
                avg_var = 0
                for i in range(len(candidates)):
                    x_i = X_cand[i:i+1, :]
                    var_i = x_i @ XtX_inv @ x_i.T
                    avg_var += var_i[0, 0]

                return avg_var / len(candidates)
            except Exception:
                return float('inf')

        # Initialize with random points
        indices = np.random.choice(n_candidates, num_runs, replace=True)
        design_matrix = candidates[indices].copy()

        # Coordinate exchange to minimize I-criterion
        max_iterations = 100
        improved = True
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            for i in range(num_runs):
                for j in range(k):
                    current_X = build_model_matrix(design_matrix, model)
                    current_i = i_criterion(current_X, candidates, model)
                    best_i = current_i
                    best_level = design_matrix[i, j]

                    for level in levels_per_factor[j]:
                        if level != design_matrix[i, j]:
                            design_matrix[i, j] = level
                            new_X = build_model_matrix(design_matrix, model)
                            new_i = i_criterion(new_X, candidates, model)

                            if new_i < best_i:
                                best_i = new_i
                                best_level = level
                                improved = True

                    design_matrix[i, j] = best_level

        # Build runs
        runs = []
        for idx, row in enumerate(design_matrix):
            coded = {}
            actual = {}
            for j, f in enumerate(factors):
                coded[f.name] = int(row[j])
                if f.is_categorical:
                    actual[f.name] = f.levels[int(row[j])]
                else:
                    if len(f.levels) == 2:
                        low, high = f.levels[0], f.levels[1]
                        mid = (low + high) / 2
                    else:
                        low, mid, high = f.levels[0], f.levels[len(f.levels)//2], f.levels[-1]

                    if row[j] == -1:
                        actual[f.name] = low
                    elif row[j] == 0:
                        actual[f.name] = mid
                    else:
                        actual[f.name] = high

            run = ExperimentRun(
                run_id=idx + 1,
                standard_order=idx + 1,
                run_order=idx + 1,
                factor_levels=actual,
                coded_levels=coded,
            )
            runs.append(run)

        # Randomize
        run_orders = list(range(1, len(runs) + 1))
        np.random.shuffle(run_orders)
        for run, order in zip(runs, run_orders):
            run.run_order = order
        runs.sort(key=lambda r: r.run_order)

        design = ExperimentDesign(
            name=f"I-Optimal Design ({model} model)",
            design_type="i_optimal",
            factors=factors,
            runs=runs,
        )

        final_X = build_model_matrix(design_matrix, model)
        final_i = i_criterion(final_X, candidates, model)

        design.notes.append(f"Model type: {model}")
        design.notes.append(f"Parameters: {n_params}")
        design.notes.append(f"Runs: {num_runs}")
        design.notes.append(f"Average prediction variance: {final_i:.6f}")
        design.notes.append(f"Optimization iterations: {iteration}")

        return design

    def taguchi(
        self,
        factors: list[Factor],
        array_type: str = "auto",
    ) -> ExperimentDesign:
        """
        Generate Taguchi Orthogonal Array design.

        Standard orthogonal arrays for robust parameter design.
        Arrays: L4, L8, L9, L12, L16, L18, L27

        Args:
            factors: List of Factor objects (2 or 3 level factors)
            array_type: 'auto' to select automatically, or specify (e.g., 'L8', 'L9')
        """
        import numpy as np

        # Standard Taguchi orthogonal arrays
        # L4: 3 factors, 2 levels
        L4 = np.array([
            [1, 1, 1],
            [1, 2, 2],
            [2, 1, 2],
            [2, 2, 1],
        ])

        # L8: 7 factors, 2 levels
        L8 = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 2, 2, 2, 2],
            [1, 2, 2, 1, 1, 2, 2],
            [1, 2, 2, 2, 2, 1, 1],
            [2, 1, 2, 1, 2, 1, 2],
            [2, 1, 2, 2, 1, 2, 1],
            [2, 2, 1, 1, 2, 2, 1],
            [2, 2, 1, 2, 1, 1, 2],
        ])

        # L9: 4 factors, 3 levels
        L9 = np.array([
            [1, 1, 1, 1],
            [1, 2, 2, 2],
            [1, 3, 3, 3],
            [2, 1, 2, 3],
            [2, 2, 3, 1],
            [2, 3, 1, 2],
            [3, 1, 3, 2],
            [3, 2, 1, 3],
            [3, 3, 2, 1],
        ])

        # L12: 11 factors, 2 levels (Plackett-Burman type)
        L12 = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            [1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
            [1, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2],
            [1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1],
            [1, 2, 2, 2, 1, 2, 2, 1, 2, 1, 1],
            [2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1],
            [2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 2],
            [2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1],
            [2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 2],
            [2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 2],
            [2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 1],
        ])

        # L16: 15 factors, 2 levels
        L16 = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1],
            [1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
            [1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1],
            [1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1],
            [1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2],
            [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1],
            [2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1],
            [2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2],
            [2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1],
            [2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2],
            [2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2],
            [2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 2, 2, 1],
        ])

        # L18: 8 factors (1 x 2-level, 7 x 3-level) - mixed level
        L18 = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 2, 2, 2, 2, 2, 2],
            [1, 1, 3, 3, 3, 3, 3, 3],
            [1, 2, 1, 1, 2, 2, 3, 3],
            [1, 2, 2, 2, 3, 3, 1, 1],
            [1, 2, 3, 3, 1, 1, 2, 2],
            [1, 3, 1, 2, 1, 3, 2, 3],
            [1, 3, 2, 3, 2, 1, 3, 1],
            [1, 3, 3, 1, 3, 2, 1, 2],
            [2, 1, 1, 3, 3, 2, 2, 1],
            [2, 1, 2, 1, 1, 3, 3, 2],
            [2, 1, 3, 2, 2, 1, 1, 3],
            [2, 2, 1, 2, 3, 1, 3, 2],
            [2, 2, 2, 3, 1, 2, 1, 3],
            [2, 2, 3, 1, 2, 3, 2, 1],
            [2, 3, 1, 3, 2, 3, 1, 2],
            [2, 3, 2, 1, 3, 1, 2, 3],
            [2, 3, 3, 2, 1, 2, 3, 1],
        ])

        # L27: 13 factors, 3 levels
        L27 = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            [1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 1, 1, 1],
            [1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2],
            [1, 3, 3, 3, 1, 1, 1, 3, 3, 3, 2, 2, 2],
            [1, 3, 3, 3, 2, 2, 2, 1, 1, 1, 3, 3, 3],
            [1, 3, 3, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1],
            [2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            [2, 1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1],
            [2, 1, 2, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2],
            [2, 2, 3, 1, 1, 2, 3, 2, 3, 1, 3, 1, 2],
            [2, 2, 3, 1, 2, 3, 1, 3, 1, 2, 1, 2, 3],
            [2, 2, 3, 1, 3, 1, 2, 1, 2, 3, 2, 3, 1],
            [2, 3, 1, 2, 1, 2, 3, 3, 1, 2, 2, 3, 1],
            [2, 3, 1, 2, 2, 3, 1, 1, 2, 3, 3, 1, 2],
            [2, 3, 1, 2, 3, 1, 2, 2, 3, 1, 1, 2, 3],
            [3, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2],
            [3, 1, 3, 2, 2, 1, 3, 2, 1, 3, 2, 1, 3],
            [3, 1, 3, 2, 3, 2, 1, 3, 2, 1, 3, 2, 1],
            [3, 2, 1, 3, 1, 3, 2, 2, 1, 3, 3, 2, 1],
            [3, 2, 1, 3, 2, 1, 3, 3, 2, 1, 1, 3, 2],
            [3, 2, 1, 3, 3, 2, 1, 1, 3, 2, 2, 1, 3],
            [3, 3, 2, 1, 1, 3, 2, 3, 2, 1, 2, 1, 3],
            [3, 3, 2, 1, 2, 1, 3, 1, 3, 2, 3, 2, 1],
            [3, 3, 2, 1, 3, 2, 1, 2, 1, 3, 1, 3, 2],
        ])

        arrays = {
            "L4": (L4, 2, 3),     # (array, levels, max_factors)
            "L8": (L8, 2, 7),
            "L9": (L9, 3, 4),
            "L12": (L12, 2, 11),
            "L16": (L16, 2, 15),
            "L18": (L18, 3, 8),  # Note: first column is 2-level
            "L27": (L27, 3, 13),
        }

        k = len(factors)

        # Determine number of levels needed
        num_levels = max(len(f.levels) for f in factors)
        if num_levels < 2:
            num_levels = 2

        # Select appropriate array
        if array_type == "auto":
            # Find smallest array that fits
            if num_levels == 2:
                if k <= 3:
                    array_type = "L4"
                elif k <= 7:
                    array_type = "L8"
                elif k <= 11:
                    array_type = "L12"
                else:
                    array_type = "L16"
            else:  # 3 levels
                if k <= 4:
                    array_type = "L9"
                elif k <= 8:
                    array_type = "L18"
                else:
                    array_type = "L27"

        if array_type not in arrays:
            array_type = "L8"  # Default fallback

        array, base_levels, max_factors = arrays[array_type]

        # Trim array to number of factors
        if k > max_factors:
            k = max_factors
            factors = factors[:max_factors]

        oa = array[:, :k]

        # Build runs
        runs = []
        for idx, row in enumerate(oa):
            coded = {}
            actual = {}

            for j, f in enumerate(factors):
                level_idx = int(row[j]) - 1  # OA uses 1-based indexing
                # Ensure index is within bounds
                level_idx = min(level_idx, len(f.levels) - 1)

                coded[f.name] = int(row[j])
                actual[f.name] = f.levels[level_idx]

            run = ExperimentRun(
                run_id=idx + 1,
                standard_order=idx + 1,
                run_order=idx + 1,
                factor_levels=actual,
                coded_levels=coded,
            )
            runs.append(run)

        # Randomize run order
        run_orders = list(range(1, len(runs) + 1))
        np.random.shuffle(run_orders)
        for run, order in zip(runs, run_orders):
            run.run_order = order
        runs.sort(key=lambda r: r.run_order)

        design = ExperimentDesign(
            name=f"Taguchi {array_type} Orthogonal Array",
            design_type="taguchi",
            factors=factors,
            runs=runs,
        )

        design.notes.append(f"Array: {array_type}")
        design.notes.append(f"Factors: {k}")
        design.notes.append(f"Runs: {len(oa)}")
        design.notes.append(f"Base levels: {base_levels}")
        design.notes.append("Orthogonal array ensures balanced factor combinations")

        return design

    def plackett_burman(
        self,
        factors: list[Factor],
    ) -> ExperimentDesign:
        """
        Generate Plackett-Burman screening design.

        Resolution III design for efficient factor screening.
        N runs where N is a multiple of 4 (8, 12, 16, 20, 24, ...).

        Args:
            factors: List of Factor objects (2-level factors)
        """
        import numpy as np

        k = len(factors)

        # Find smallest N that can accommodate k factors (N-1 >= k)
        if k <= 7:
            N = 8
        elif k <= 11:
            N = 12
        elif k <= 15:
            N = 16
        elif k <= 19:
            N = 20
        else:
            N = 24

        # Generate first row based on N
        # These are standard Plackett-Burman first rows
        first_rows = {
            8: [1, 1, 1, -1, 1, -1, -1],
            12: [1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1],
            16: [1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1],
            20: [1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1],
            24: [1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1],
        }

        first_row = first_rows.get(N, first_rows[12])

        # Build design matrix by cyclic permutation
        design_matrix = []
        for i in range(N - 1):
            row = first_row[-(i):] + first_row[:-(i)] if i > 0 else first_row
            design_matrix.append(row[:k])  # Take only k columns

        # Add row of all -1s
        design_matrix.append([-1] * k)

        design_matrix = np.array(design_matrix)

        # Build runs
        runs = []
        for idx, row in enumerate(design_matrix):
            coded = {}
            actual = {}

            for j, f in enumerate(factors):
                coded[f.name] = int(row[j])
                # Map -1 to low, +1 to high
                if len(f.levels) >= 2:
                    actual[f.name] = f.levels[0] if row[j] < 0 else f.levels[1]
                else:
                    actual[f.name] = f.levels[0]

            run = ExperimentRun(
                run_id=idx + 1,
                standard_order=idx + 1,
                run_order=idx + 1,
                factor_levels=actual,
                coded_levels=coded,
            )
            runs.append(run)

        # Randomize
        run_orders = list(range(1, len(runs) + 1))
        np.random.shuffle(run_orders)
        for run, order in zip(runs, run_orders):
            run.run_order = order
        runs.sort(key=lambda r: r.run_order)

        design = ExperimentDesign(
            name=f"Plackett-Burman Design (N={N})",
            design_type="plackett_burman",
            factors=factors,
            runs=runs,
            resolution=3,
        )

        design.notes.append(f"Resolution III screening design")
        design.notes.append(f"Factors: {k}")
        design.notes.append(f"Runs: {N}")
        design.notes.append("Main effects are aliased with 2-factor interactions")

        return design

    def box_behnken(
        self,
        factors: list[Factor],
        center_points: int = 3,
    ) -> ExperimentDesign:
        """
        Generate Box-Behnken response surface design.

        RSM design without corner points (no extreme combinations).
        Requires 3+ factors. More economical than CCD for 3-5 factors.

        Args:
            factors: List of Factor objects (requires 3+)
            center_points: Number of center point runs
        """
        import numpy as np
        from itertools import combinations

        k = len(factors)
        if k < 3:
            raise ValueError("Box-Behnken requires at least 3 factors")

        # Box-Behnken uses pairs of factors at ±1 with others at 0
        runs_data = []

        # Generate design points for each pair of factors
        for i, j in combinations(range(k), 2):
            # 4 points for each pair: (±1, ±1) with others at 0
            for sign1 in [-1, 1]:
                for sign2 in [-1, 1]:
                    row = [0] * k
                    row[i] = sign1
                    row[j] = sign2
                    runs_data.append(row)

        # Add center points
        for _ in range(center_points):
            runs_data.append([0] * k)

        design_matrix = np.array(runs_data)

        # Build runs
        runs = []
        for idx, row in enumerate(design_matrix):
            coded = {}
            actual = {}
            is_center = all(v == 0 for v in row)

            for j, f in enumerate(factors):
                coded[f.name] = int(row[j])
                # Map -1, 0, +1 to actual levels
                if len(f.levels) >= 3:
                    low, mid, high = f.levels[0], f.levels[1], f.levels[2]
                elif len(f.levels) == 2:
                    low, high = f.levels[0], f.levels[1]
                    mid = (low + high) / 2
                else:
                    low = mid = high = f.levels[0]

                if row[j] == -1:
                    actual[f.name] = low
                elif row[j] == 0:
                    actual[f.name] = mid
                else:
                    actual[f.name] = high

            run = ExperimentRun(
                run_id=idx + 1,
                standard_order=idx + 1,
                run_order=idx + 1,
                factor_levels=actual,
                coded_levels=coded,
                is_center_point=is_center,
            )
            runs.append(run)

        # Randomize
        run_orders = list(range(1, len(runs) + 1))
        np.random.shuffle(run_orders)
        for run, order in zip(runs, run_orders):
            run.run_order = order
        runs.sort(key=lambda r: r.run_order)

        n_design_points = len(runs_data) - center_points

        design = ExperimentDesign(
            name=f"Box-Behnken Design",
            design_type="box_behnken",
            factors=factors,
            runs=runs,
            num_center_points=center_points,
        )

        design.notes.append(f"Response surface design without corner points")
        design.notes.append(f"Factors: {k}")
        design.notes.append(f"Design points: {n_design_points}")
        design.notes.append(f"Center points: {center_points}")
        design.notes.append(f"Total runs: {len(runs)}")

        return design

    def definitive_screening(
        self,
        factors: list[Factor],
    ) -> ExperimentDesign:
        """
        Generate Definitive Screening Design (DSD).

        Modern screening design where:
        - Main effects are orthogonal to 2-factor interactions
        - Can estimate quadratic effects
        - 2k + 1 runs for k factors

        Args:
            factors: List of Factor objects (3+ factors)
        """
        import numpy as np

        k = len(factors)
        if k < 3:
            raise ValueError("Definitive Screening requires at least 3 factors")

        # DSD construction: Conference matrix + foldover + center
        # For k factors: 2k runs from foldover pairs + 1 center

        runs_data = []

        # Generate conference matrix rows
        # Each row has exactly one 0 (one factor at center)
        # Other factors at ±1 in balanced pairs

        for i in range(k):
            # Row with factor i at 0, others at specified levels
            row1 = [0] * k
            row2 = [0] * k

            for j in range(k):
                if j == i:
                    row1[j] = 0
                    row2[j] = 0
                elif j < i:
                    row1[j] = 1 if (i + j) % 2 == 0 else -1
                    row2[j] = -row1[j]  # Foldover
                else:
                    row1[j] = 1 if (i + j) % 2 == 1 else -1
                    row2[j] = -row1[j]

            runs_data.append(row1)
            runs_data.append(row2)

        # Add center point
        runs_data.append([0] * k)

        design_matrix = np.array(runs_data)

        # Build runs
        runs = []
        for idx, row in enumerate(design_matrix):
            coded = {}
            actual = {}
            is_center = all(v == 0 for v in row)

            for j, f in enumerate(factors):
                coded[f.name] = int(row[j])
                if len(f.levels) >= 3:
                    low, mid, high = f.levels[0], f.levels[1], f.levels[2]
                elif len(f.levels) == 2:
                    low, high = f.levels[0], f.levels[1]
                    mid = (low + high) / 2
                else:
                    low = mid = high = f.levels[0]

                if row[j] == -1:
                    actual[f.name] = low
                elif row[j] == 0:
                    actual[f.name] = mid
                else:
                    actual[f.name] = high

            run = ExperimentRun(
                run_id=idx + 1,
                standard_order=idx + 1,
                run_order=idx + 1,
                factor_levels=actual,
                coded_levels=coded,
                is_center_point=is_center,
            )
            runs.append(run)

        # Randomize
        run_orders = list(range(1, len(runs) + 1))
        np.random.shuffle(run_orders)
        for run, order in zip(runs, run_orders):
            run.run_order = order
        runs.sort(key=lambda r: r.run_order)

        design = ExperimentDesign(
            name=f"Definitive Screening Design",
            design_type="definitive_screening",
            factors=factors,
            runs=runs,
            num_center_points=1,
        )

        design.notes.append(f"Modern screening design with orthogonal main effects")
        design.notes.append(f"Factors: {k}")
        design.notes.append(f"Runs: {2 * k + 1}")
        design.notes.append("Main effects orthogonal to 2-factor interactions")
        design.notes.append("Can detect and estimate quadratic effects")

        return design
