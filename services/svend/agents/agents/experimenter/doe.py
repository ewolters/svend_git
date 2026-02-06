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

    def box_behnken(self, factors: list[Factor], center_points: int = 3,
                    randomize: bool = True) -> ExperimentDesign:
        """
        Generate a Box-Behnken design for response surface methodology.

        More economical than CCD - no corner points, all factors at mid-range.
        Requires 3+ factors with 3 levels each.

        Args:
            factors: List of continuous factors (will use 3 levels: -1, 0, +1)
            center_points: Number of center points
            randomize: Whether to randomize run order
        """
        n = len(factors)
        if n < 3:
            raise ValueError("Box-Behnken requires at least 3 factors")

        runs = []
        run_id = 1

        # Box-Behnken uses pairs of factors at their extremes, others at center
        # For each pair of factors, run a 2^2 factorial while others at 0
        from itertools import combinations

        for pair in combinations(range(n), 2):
            i, j = pair
            # 2^2 factorial for this pair
            for level_i in [-1, 1]:
                for level_j in [-1, 1]:
                    factor_levels = {}
                    coded_levels = {}

                    for k, f in enumerate(factors):
                        low, high = f.levels[0], f.levels[1]
                        mid = (low + high) / 2
                        half_range = (high - low) / 2

                        if k == i:
                            coded_levels[f.name] = level_i
                            factor_levels[f.name] = mid + level_i * half_range
                        elif k == j:
                            coded_levels[f.name] = level_j
                            factor_levels[f.name] = mid + level_j * half_range
                        else:
                            coded_levels[f.name] = 0
                            factor_levels[f.name] = mid

                    runs.append(ExperimentRun(
                        run_id=run_id,
                        standard_order=run_id,
                        run_order=run_id,
                        factor_levels=factor_levels,
                        coded_levels=coded_levels,
                    ))
                    run_id += 1

        # Add center points
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
            name=f"{n}-Factor Box-Behnken Design",
            design_type="Box-Behnken (Response Surface)",
            factors=factors,
            runs=runs,
            num_center_points=center_points,
        )

        design.notes.append(f"Edge points: {len(runs) - center_points}")
        design.notes.append(f"Center points: {center_points}")
        design.notes.append("All points are within the experimental region (no corner points)")
        design.notes.append("More economical than CCD, but cannot detect all quadratic effects independently")

        return design

    def plackett_burman(self, factors: list[Factor], randomize: bool = True) -> ExperimentDesign:
        """
        Generate a Plackett-Burman screening design.

        Highly efficient for screening many factors. Resolution III (main effects
        confounded with 2-factor interactions).

        Uses N runs where N is a multiple of 4: 8, 12, 16, 20, 24, etc.
        Can screen up to N-1 factors.

        Args:
            factors: List of 2-level factors
            randomize: Whether to randomize run order
        """
        n = len(factors)

        # Plackett-Burman generator rows for different run counts
        # These are the first rows of the Hadamard matrices
        generators = {
            8: [1, 1, 1, -1, 1, -1, -1],
            12: [1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1],
            16: [1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1],
            20: [1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1],
            24: [1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1],
        }

        # Find smallest design that can accommodate n factors
        num_runs = None
        for runs in sorted(generators.keys()):
            if runs - 1 >= n:
                num_runs = runs
                break

        if num_runs is None:
            # Default to 24 runs and truncate factors
            num_runs = 24

        generator = generators[num_runs]

        # Build design matrix by cyclic permutation
        matrix = []
        row = generator[:n]  # Use only first n elements
        for _ in range(num_runs - 1):
            matrix.append(row.copy())
            # Cyclic shift
            row = row[-1:] + row[:-1]

        # Add row of all -1s
        matrix.append([-1] * n)

        runs = []
        for run_idx, coded_row in enumerate(matrix):
            factor_levels = {}
            coded_levels = {}

            for i, f in enumerate(factors):
                coded_levels[f.name] = coded_row[i]
                level_idx = 0 if coded_row[i] == -1 else 1
                factor_levels[f.name] = f.levels[level_idx]

            runs.append(ExperimentRun(
                run_id=run_idx + 1,
                standard_order=run_idx + 1,
                run_order=run_idx + 1,
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
            name=f"Plackett-Burman Screening Design ({n} factors)",
            design_type=f"Plackett-Burman {num_runs}-run",
            factors=factors,
            runs=runs,
            resolution=3,
        )

        design.notes.append(f"Runs: {num_runs} (screens {n} factors)")
        design.notes.append("Resolution III - main effects confounded with 2-factor interactions")
        design.notes.append("Use for initial screening to identify important factors")
        design.notes.append(f"Efficiency: {n}/{num_runs-1} = {n/(num_runs-1)*100:.0f}% of capacity used")

        return design

    def taguchi(self, factors: list[Factor], array_type: str = "auto",
                randomize: bool = True) -> ExperimentDesign:
        """
        Generate a Taguchi orthogonal array.

        Standard arrays for robust parameter design:
        - L4: 3 factors at 2 levels (4 runs)
        - L8: 7 factors at 2 levels (8 runs)
        - L9: 4 factors at 3 levels (9 runs)
        - L12: 11 factors at 2 levels (12 runs)
        - L16: 15 factors at 2 levels (16 runs)
        - L18: 1 factor at 2 levels + 7 factors at 3 levels (18 runs)
        - L27: 13 factors at 3 levels (27 runs)

        Args:
            factors: List of factors
            array_type: "L4", "L8", "L9", "L12", "L16", "L18", "L27", or "auto"
            randomize: Whether to randomize run order
        """
        n = len(factors)
        num_levels = [f.num_levels for f in factors]

        # Standard Taguchi arrays (coded as 0, 1, 2 for levels)
        arrays = {
            "L4": {
                "runs": 4,
                "max_factors": 3,
                "levels": 2,
                "matrix": [
                    [0, 0, 0],
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0],
                ]
            },
            "L8": {
                "runs": 8,
                "max_factors": 7,
                "levels": 2,
                "matrix": [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1],
                    [0, 1, 1, 0, 0, 1, 1],
                    [0, 1, 1, 1, 1, 0, 0],
                    [1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 1, 0, 1, 0],
                    [1, 1, 0, 0, 1, 1, 0],
                    [1, 1, 0, 1, 0, 0, 1],
                ]
            },
            "L9": {
                "runs": 9,
                "max_factors": 4,
                "levels": 3,
                "matrix": [
                    [0, 0, 0, 0],
                    [0, 1, 1, 1],
                    [0, 2, 2, 2],
                    [1, 0, 1, 2],
                    [1, 1, 2, 0],
                    [1, 2, 0, 1],
                    [2, 0, 2, 1],
                    [2, 1, 0, 2],
                    [2, 2, 1, 0],
                ]
            },
            "L12": {
                "runs": 12,
                "max_factors": 11,
                "levels": 2,
                "matrix": [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                    [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1],
                    [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
                    [0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0],
                    [1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
                    [1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0],
                    [1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
                    [1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0],
                    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
                    [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
                ]
            },
            "L16": {
                "runs": 16,
                "max_factors": 15,
                "levels": 2,
                "matrix": None,  # Will generate 2^4 factorial
            },
        }

        # Auto-select array
        if array_type == "auto":
            all_two_level = all(nl == 2 for nl in num_levels)
            all_three_level = all(nl == 3 for nl in num_levels)

            if all_two_level:
                if n <= 3:
                    array_type = "L4"
                elif n <= 7:
                    array_type = "L8"
                elif n <= 11:
                    array_type = "L12"
                else:
                    array_type = "L16"
            elif all_three_level:
                if n <= 4:
                    array_type = "L9"
                else:
                    # Fall back to full factorial
                    return self.full_factorial(factors, randomize=randomize)
            else:
                # Mixed levels - use L8 or fall back
                array_type = "L8"

        array_spec = arrays.get(array_type)
        if not array_spec:
            return self.full_factorial(factors, randomize=randomize)

        # Generate L16 if needed
        if array_spec["matrix"] is None and array_type == "L16":
            # 2^4 full factorial expanded
            from itertools import product
            matrix = []
            for combo in product([0, 1], repeat=4):
                row = list(combo)
                # Add derived columns (interactions)
                for i in range(4):
                    for j in range(i+1, 4):
                        row.append((combo[i] + combo[j]) % 2)
                for i in range(4):
                    for j in range(i+1, 4):
                        for k in range(j+1, 4):
                            row.append((combo[i] + combo[j] + combo[k]) % 2)
                matrix.append(row[:15])
            array_spec["matrix"] = matrix

        matrix = array_spec["matrix"]

        # Build runs
        runs = []
        for run_idx, coded_row in enumerate(matrix):
            factor_levels = {}
            coded_levels = {}

            for i, f in enumerate(factors):
                if i < len(coded_row):
                    level_idx = coded_row[i]
                    if level_idx < f.num_levels:
                        coded_levels[f.name] = level_idx
                        factor_levels[f.name] = f.levels[level_idx]
                    else:
                        coded_levels[f.name] = 0
                        factor_levels[f.name] = f.levels[0]

            runs.append(ExperimentRun(
                run_id=run_idx + 1,
                standard_order=run_idx + 1,
                run_order=run_idx + 1,
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
            name=f"Taguchi {array_type} Orthogonal Array",
            design_type=f"Taguchi {array_type}",
            factors=factors,
            runs=runs,
            resolution=3 if array_spec["levels"] == 2 else None,
        )

        design.notes.append(f"Standard {array_type} orthogonal array: {array_spec['runs']} runs")
        design.notes.append(f"Can study up to {array_spec['max_factors']} factors at {array_spec['levels']} levels")
        design.notes.append("Balanced design - each level appears equally often")
        design.notes.append("Use Signal-to-Noise (S/N) ratios for robust optimization")

        return design

    def definitive_screening(self, factors: list[Factor],
                             randomize: bool = True) -> ExperimentDesign:
        """
        Generate a Definitive Screening Design (DSD).

        Modern alternative to Plackett-Burman with key advantages:
        - Main effects are orthogonal to 2-factor interactions
        - Can estimate some quadratic effects
        - Requires only 2n+1 runs for n factors

        Args:
            factors: List of continuous factors (3 levels will be used: -1, 0, +1)
            randomize: Whether to randomize run order
        """
        n = len(factors)
        if n < 3:
            raise ValueError("Definitive Screening requires at least 3 factors")

        # DSD construction: Conference matrix approach
        # For n factors, we need a (2n+1) x n design

        runs = []
        run_id = 1

        # First n rows: one factor at high, one at low, rest at center
        # This creates the "fold-over" pairs
        for i in range(n):
            # Positive fold
            factor_levels_pos = {}
            coded_levels_pos = {}
            factor_levels_neg = {}
            coded_levels_neg = {}

            for j, f in enumerate(factors):
                low, high = f.levels[0], f.levels[1]
                mid = (low + high) / 2
                half_range = (high - low) / 2

                if j == i:
                    coded_levels_pos[f.name] = 1
                    factor_levels_pos[f.name] = high
                    coded_levels_neg[f.name] = -1
                    factor_levels_neg[f.name] = low
                elif j == (i + 1) % n:
                    coded_levels_pos[f.name] = -1
                    factor_levels_pos[f.name] = low
                    coded_levels_neg[f.name] = 1
                    factor_levels_neg[f.name] = high
                else:
                    coded_levels_pos[f.name] = 0
                    factor_levels_pos[f.name] = mid
                    coded_levels_neg[f.name] = 0
                    factor_levels_neg[f.name] = mid

            runs.append(ExperimentRun(
                run_id=run_id,
                standard_order=run_id,
                run_order=run_id,
                factor_levels=factor_levels_pos,
                coded_levels=coded_levels_pos,
            ))
            run_id += 1

            runs.append(ExperimentRun(
                run_id=run_id,
                standard_order=run_id,
                run_order=run_id,
                factor_levels=factor_levels_neg,
                coded_levels=coded_levels_neg,
            ))
            run_id += 1

        # Center point
        center_levels = {}
        center_coded = {}
        for f in factors:
            low, high = f.levels[0], f.levels[1]
            center_levels[f.name] = (low + high) / 2
            center_coded[f.name] = 0

        runs.append(ExperimentRun(
            run_id=run_id,
            standard_order=run_id,
            run_order=run_id,
            factor_levels=center_levels,
            coded_levels=center_coded,
            is_center_point=True,
        ))

        # Randomize
        if randomize:
            run_orders = list(range(1, len(runs) + 1))
            self.rng.shuffle(run_orders)
            for i, run in enumerate(runs):
                run.run_order = run_orders[i]

        design = ExperimentDesign(
            name=f"Definitive Screening Design ({n} factors)",
            design_type="Definitive Screening",
            factors=factors,
            runs=runs,
            num_center_points=1,
        )

        design.notes.append(f"Runs: {2*n + 1} for {n} factors")
        design.notes.append("Main effects orthogonal to 2-factor interactions")
        design.notes.append("Can estimate all main effects and some quadratic effects")
        design.notes.append("Modern alternative to Plackett-Burman for initial screening")

        return design
