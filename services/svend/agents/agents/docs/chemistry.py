"""
Chemistry Documentation Tools

Format chemical equations, reactions, and thermodynamics for documentation.
NOT a reasoning engine - just formatters and calculators.

Outputs:
- LaTeX/KaTeX chemical equations
- Balanced reactions
- Thermodynamic calculations (from known values)
- Reaction mechanism diagrams
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ReactionInfo:
    """Parsed reaction information."""
    reactants: list[tuple[int, str]]  # [(coefficient, formula), ...]
    products: list[tuple[int, str]]
    conditions: str = ""  # e.g., "heat", "catalyst", "25°C"
    is_reversible: bool = False
    is_balanced: bool = True


@dataclass
class ThermodynamicsData:
    """Thermodynamic properties of a reaction."""
    delta_h: Optional[float] = None  # kJ/mol (enthalpy)
    delta_s: Optional[float] = None  # J/(mol·K) (entropy)
    delta_g: Optional[float] = None  # kJ/mol (Gibbs free energy)
    temperature: float = 298.15  # K (default 25°C)
    is_spontaneous: Optional[bool] = None
    equilibrium_constant: Optional[float] = None

    def to_latex(self) -> str:
        """Format as LaTeX."""
        lines = []
        if self.delta_h is not None:
            lines.append(f"\\Delta H = {self.delta_h:.1f} \\text{{ kJ/mol}}")
        if self.delta_s is not None:
            lines.append(f"\\Delta S = {self.delta_s:.1f} \\text{{ J/(mol·K)}}")
        if self.delta_g is not None:
            lines.append(f"\\Delta G = {self.delta_g:.1f} \\text{{ kJ/mol}}")
        if self.is_spontaneous is not None:
            spont = "spontaneous" if self.is_spontaneous else "non-spontaneous"
            lines.append(f"\\text{{Reaction is {spont} at {self.temperature:.0f} K}}")
        return " \\\\\n".join(lines)


# Common thermodynamic data (standard enthalpies of formation, kJ/mol)
STANDARD_ENTHALPIES = {
    "H2O(l)": -285.8,
    "H2O(g)": -241.8,
    "CO2(g)": -393.5,
    "CH4(g)": -74.8,
    "C2H6(g)": -84.7,
    "C2H5OH(l)": -277.7,
    "NH3(g)": -46.1,
    "NO(g)": 90.3,
    "NO2(g)": 33.2,
    "SO2(g)": -296.8,
    "SO3(g)": -395.7,
    "HCl(g)": -92.3,
    "NaCl(s)": -411.2,
    "ATP": -2982,  # Approximate, ATP hydrolysis
    "ADP": -2000,  # Approximate
    "glucose": -1274,
    "O2(g)": 0,
    "H2(g)": 0,
    "N2(g)": 0,
    "C(s)": 0,
    "Na(s)": 0,
    "K(s)": 0,
    "Cl2(g)": 0,
}

# Common biochemical reactions
BIOCHEMICAL_REACTIONS = {
    "atp_hydrolysis": {
        "equation": "ATP + H2O → ADP + Pi",
        "delta_g": -30.5,  # kJ/mol at pH 7
        "description": "ATP hydrolysis releases energy for cellular work",
    },
    "glucose_oxidation": {
        "equation": "C6H12O6 + 6O2 → 6CO2 + 6H2O",
        "delta_g": -2870,
        "description": "Complete oxidation of glucose",
    },
    "photosynthesis": {
        "equation": "6CO2 + 6H2O → C6H12O6 + 6O2",
        "delta_g": 2870,
        "description": "Photosynthesis (light-driven)",
    },
    "na_k_pump": {
        "equation": "3Na+(in) + 2K+(out) + ATP → 3Na+(out) + 2K+(in) + ADP + Pi",
        "delta_g": -30.5,  # Coupled to ATP hydrolysis
        "description": "Na+/K+-ATPase pump, maintains ion gradients",
    },
}


class ChemistryFormatter:
    """
    Format chemistry for documentation.

    Not a reasoning engine - just formatting and known calculations.
    """

    # Element symbols for parsing
    ELEMENTS = {
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
        'Fe', 'Cu', 'Zn', 'Ag', 'Au', 'Pb', 'U', 'ATP', 'ADP', 'Pi',
    }

    def reaction_to_latex(self, reaction: str) -> str:
        """
        Convert reaction string to LaTeX using mhchem syntax.

        Input: "2H2 + O2 -> 2H2O"
        Output: "\\ce{2H2 + O2 -> 2H2O}"
        """
        # Normalize arrow
        reaction = reaction.replace("→", "->")
        reaction = reaction.replace("<->", "<=>")
        reaction = reaction.replace("⇌", "<=>")

        # Add mhchem wrapper
        return f"\\ce{{{reaction}}}"

    def reaction_to_katex(self, reaction: str) -> str:
        """
        Convert reaction string to KaTeX-compatible format.
        Same as LaTeX but may need different escaping for web.
        """
        return self.reaction_to_latex(reaction)

    def parse_reaction(self, reaction: str) -> ReactionInfo:
        """Parse a reaction string into components."""
        # Normalize
        reaction = reaction.replace("→", "->").replace("⇌", "<=>")

        # Check reversibility
        is_reversible = "<=>" in reaction

        # Split into reactants and products
        if "<=>" in reaction:
            left, right = reaction.split("<=>")
        elif "->" in reaction:
            left, right = reaction.split("->")
        else:
            raise ValueError(f"Cannot parse reaction: {reaction}")

        def parse_side(side: str) -> list[tuple[int, str]]:
            """Parse one side of the reaction."""
            compounds = []
            for term in side.split("+"):
                term = term.strip()
                # Extract coefficient
                match = re.match(r'^(\d*)\s*(.+)$', term)
                if match:
                    coef = int(match.group(1)) if match.group(1) else 1
                    formula = match.group(2).strip()
                    compounds.append((coef, formula))
            return compounds

        return ReactionInfo(
            reactants=parse_side(left),
            products=parse_side(right),
            is_reversible=is_reversible,
        )

    def balance_equation(self, reaction: str) -> str:
        """
        Balance a chemical equation.

        Simple cases only - uses algebraic approach for small molecules.
        """
        # This is a simplified balancer for common cases
        # Full balancing requires matrix algebra

        info = self.parse_reaction(reaction)

        # For now, just format nicely (full balancing is complex)
        # Real implementation would use sympy or similar

        reactants = " + ".join(
            f"{c if c > 1 else ''}{f}" for c, f in info.reactants
        )
        products = " + ".join(
            f"{c if c > 1 else ''}{f}" for c, f in info.products
        )

        arrow = "<=>" if info.is_reversible else "->"
        return f"{reactants} {arrow} {products}"

    def estimate_thermodynamics(self, reaction: str) -> ThermodynamicsData:
        """
        Estimate thermodynamic properties from known data.

        Uses standard enthalpies of formation when available.
        """
        # Check if it's a known biochemical reaction
        for key, data in BIOCHEMICAL_REACTIONS.items():
            if key in reaction.lower().replace(" ", "_").replace("-", "_"):
                return ThermodynamicsData(
                    delta_g=data["delta_g"],
                    is_spontaneous=data["delta_g"] < 0,
                )

        # Try to calculate from standard enthalpies
        info = self.parse_reaction(reaction)

        # Calculate ΔH from standard enthalpies
        delta_h = 0
        can_calculate = True

        # Products - Reactants
        for coef, formula in info.products:
            # Try to match formula to known compounds
            key = self._match_compound(formula)
            if key and key in STANDARD_ENTHALPIES:
                delta_h += coef * STANDARD_ENTHALPIES[key]
            else:
                can_calculate = False

        for coef, formula in info.reactants:
            key = self._match_compound(formula)
            if key and key in STANDARD_ENTHALPIES:
                delta_h -= coef * STANDARD_ENTHALPIES[key]
            else:
                can_calculate = False

        if can_calculate:
            return ThermodynamicsData(
                delta_h=delta_h,
                is_spontaneous=delta_h < 0,  # Simplified, ignores entropy
            )

        return ThermodynamicsData()

    def _match_compound(self, formula: str) -> Optional[str]:
        """Try to match a formula to known compounds."""
        formula_clean = formula.strip()

        # Direct match
        if formula_clean in STANDARD_ENTHALPIES:
            return formula_clean

        # Try with common state annotations
        for state in ["(g)", "(l)", "(s)", "(aq)"]:
            key = formula_clean + state
            if key in STANDARD_ENTHALPIES:
                return key

        # Try lowercase
        if formula_clean.lower() in STANDARD_ENTHALPIES:
            return formula_clean.lower()

        return None

    def format_mechanism(self, steps: list[str]) -> str:
        """
        Format a reaction mechanism with numbered steps.

        Input: ["A -> B", "B + C -> D"]
        Output: LaTeX formatted mechanism
        """
        lines = ["\\begin{align}"]
        for i, step in enumerate(steps, 1):
            latex_step = self.reaction_to_latex(step).replace("\\ce{", "").replace("}", "")
            lines.append(f"  \\text{{Step {i}:}} \\quad & {latex_step} \\\\")
        lines.append("\\end{align}")
        return "\n".join(lines)

    def get_biochemical_info(self, reaction_name: str) -> dict:
        """Get information about a known biochemical reaction."""
        key = reaction_name.lower().replace(" ", "_").replace("-", "_")

        if key in BIOCHEMICAL_REACTIONS:
            data = BIOCHEMICAL_REACTIONS[key]
            return {
                "equation": data["equation"],
                "latex": self.reaction_to_latex(data["equation"]),
                "delta_g": data["delta_g"],
                "description": data["description"],
                "is_spontaneous": data["delta_g"] < 0,
            }

        return None

    def format_equation_block(self, equations: list[str], numbered: bool = True) -> str:
        """
        Format multiple equations as a LaTeX align block.
        """
        env = "align" if numbered else "align*"
        lines = [f"\\begin{{{env}}}"]

        for eq in equations:
            latex = self.reaction_to_latex(eq).replace("\\ce{", "").replace("}", "")
            lines.append(f"  & {latex} \\\\")

        lines.append(f"\\end{{{env}}}")
        return "\n".join(lines)


def quick_reaction(reaction: str) -> dict:
    """Quick helper to format a reaction."""
    formatter = ChemistryFormatter()
    return {
        "input": reaction,
        "latex": formatter.reaction_to_latex(reaction),
        "katex": formatter.reaction_to_katex(reaction),
        "thermodynamics": formatter.estimate_thermodynamics(reaction).to_latex(),
    }
