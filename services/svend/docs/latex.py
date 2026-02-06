"""
LaTeX/KaTeX Formatting

Format mathematical content, equations, and documents in LaTeX.
Outputs can be used in:
- LaTeX documents (.tex)
- KaTeX for web rendering
- Markdown with math blocks

This is a FORMATTER, not a computer algebra system.
"""

from dataclasses import dataclass
from typing import Union
import re


@dataclass
class Equation:
    """A mathematical equation."""
    latex: str
    description: str = ""
    label: str = ""

    def inline(self) -> str:
        """Return inline math: $...$"""
        return f"${self.latex}$"

    def display(self) -> str:
        """Return display math: $$...$$"""
        return f"$${self.latex}$$"

    def latex_env(self, numbered: bool = True) -> str:
        """Return LaTeX equation environment."""
        env = "equation" if numbered else "equation*"
        if self.label and numbered:
            return f"\\begin{{{env}}}\n{self.latex}\n\\label{{{self.label}}}\n\\end{{{env}}}"
        return f"\\begin{{{env}}}\n{self.latex}\n\\end{{{env}}}"


class LaTeXFormatter:
    """
    Format content as LaTeX.

    Handles:
    - Mathematical equations
    - Tables
    - Document structure
    - Common notation
    """

    # Common symbol mappings
    SYMBOLS = {
        'alpha': r'\alpha',
        'beta': r'\beta',
        'gamma': r'\gamma',
        'delta': r'\delta',
        'Delta': r'\Delta',
        'epsilon': r'\epsilon',
        'theta': r'\theta',
        'lambda': r'\lambda',
        'mu': r'\mu',
        'sigma': r'\sigma',
        'Sigma': r'\Sigma',
        'omega': r'\omega',
        'Omega': r'\Omega',
        'pi': r'\pi',
        'phi': r'\phi',
        'psi': r'\psi',
        'infinity': r'\infty',
        'partial': r'\partial',
        'nabla': r'\nabla',
        'sqrt': r'\sqrt',
        'sum': r'\sum',
        'prod': r'\prod',
        'int': r'\int',
        'approx': r'\approx',
        'neq': r'\neq',
        'leq': r'\leq',
        'geq': r'\geq',
        'pm': r'\pm',
        'times': r'\times',
        'cdot': r'\cdot',
        'rightarrow': r'\rightarrow',
        'leftarrow': r'\leftarrow',
        'Rightarrow': r'\Rightarrow',
    }

    def fraction(self, numerator: str, denominator: str) -> str:
        """Create a fraction."""
        return f"\\frac{{{numerator}}}{{{denominator}}}"

    def sqrt(self, content: str, n: int = None) -> str:
        """Create square root or nth root."""
        if n and n != 2:
            return f"\\sqrt[{n}]{{{content}}}"
        return f"\\sqrt{{{content}}}"

    def subscript(self, base: str, sub: str) -> str:
        """Add subscript."""
        return f"{base}_{{{sub}}}"

    def superscript(self, base: str, sup: str) -> str:
        """Add superscript."""
        return f"{base}^{{{sup}}}"

    def integral(self, lower: str = None, upper: str = None,
                 integrand: str = "", var: str = "x") -> str:
        """Create integral."""
        if lower is not None and upper is not None:
            return f"\\int_{{{lower}}}^{{{upper}}} {integrand} \\, d{var}"
        elif lower is not None:
            return f"\\int_{{{lower}}} {integrand} \\, d{var}"
        return f"\\int {integrand} \\, d{var}"

    def sum(self, lower: str = None, upper: str = None, term: str = "") -> str:
        """Create summation."""
        if lower and upper:
            return f"\\sum_{{{lower}}}^{{{upper}}} {term}"
        elif lower:
            return f"\\sum_{{{lower}}} {term}"
        return f"\\sum {term}"

    def matrix(self, data: list[list], style: str = "pmatrix") -> str:
        """
        Create a matrix.

        Styles: pmatrix (parentheses), bmatrix (brackets), vmatrix (vertical bars)
        """
        rows = []
        for row in data:
            rows.append(" & ".join(str(x) for x in row))
        content = " \\\\\n".join(rows)
        return f"\\begin{{{style}}}\n{content}\n\\end{{{style}}}"

    def cases(self, conditions: list[tuple[str, str]]) -> str:
        """
        Create piecewise function.

        Args:
            conditions: List of (expression, condition) tuples
        """
        lines = []
        for expr, cond in conditions:
            lines.append(f"{expr} & \\text{{if }} {cond}")
        content = " \\\\\n".join(lines)
        return f"\\begin{{cases}}\n{content}\n\\end{{cases}}"

    def align(self, equations: list[str], numbered: bool = True) -> str:
        """Create aligned equations."""
        env = "align" if numbered else "align*"
        content = " \\\\\n".join(f"& {eq}" for eq in equations)
        return f"\\begin{{{env}}}\n{content}\n\\end{{{env}}}"

    def table(self, data: list[list], headers: list[str] = None,
              caption: str = None, label: str = None) -> str:
        """Create a LaTeX table."""
        n_cols = len(data[0]) if data else 0
        col_spec = "|" + "c|" * n_cols

        lines = [
            "\\begin{table}[h]",
            "\\centering",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\hline",
        ]

        if headers:
            lines.append(" & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\")
            lines.append("\\hline")

        for row in data:
            lines.append(" & ".join(str(x) for x in row) + " \\\\")

        lines.append("\\hline")
        lines.append("\\end{tabular}")

        if caption:
            lines.append(f"\\caption{{{caption}}}")
        if label:
            lines.append(f"\\label{{{label}}}")

        lines.append("\\end{table}")
        return "\n".join(lines)

    def figure(self, image_path: str, caption: str = None,
               label: str = None, width: str = "0.8\\textwidth") -> str:
        """Create a LaTeX figure."""
        lines = [
            "\\begin{figure}[h]",
            "\\centering",
            f"\\includegraphics[width={width}]{{{image_path}}}",
        ]

        if caption:
            lines.append(f"\\caption{{{caption}}}")
        if label:
            lines.append(f"\\label{{{label}}}")

        lines.append("\\end{figure}")
        return "\n".join(lines)

    def section(self, title: str, content: str, level: int = 1) -> str:
        """Create a section."""
        cmd = ["section", "subsection", "subsubsection"][min(level - 1, 2)]
        return f"\\{cmd}{{{title}}}\n\n{content}"

    def document(self, title: str, author: str, content: str,
                 packages: list[str] = None, document_class: str = "article") -> str:
        """
        Create a complete LaTeX document.
        """
        default_packages = [
            "amsmath",
            "amssymb",
            "graphicx",
            "hyperref",
            # "mhchem",  # For chemistry - requires texlive-science
        ]
        all_packages = list(set(default_packages + (packages or [])))

        lines = [
            f"\\documentclass{{{document_class}}}",
            "",
        ]

        for pkg in all_packages:
            lines.append(f"\\usepackage{{{pkg}}}")

        lines.extend([
            "",
            f"\\title{{{title}}}",
            f"\\author{{{author}}}",
            "\\date{\\today}",
            "",
            "\\begin{document}",
            "",
            "\\maketitle",
            "",
            content,
            "",
            "\\end{document}",
        ])

        return "\n".join(lines)

    def text_to_latex(self, text: str) -> str:
        """Convert plain text with simple notation to LaTeX."""
        # Replace common patterns
        result = text

        # Fractions: a/b -> \frac{a}{b}
        result = re.sub(r'(\w+)/(\w+)', r'\\frac{\1}{\2}', result)

        # Subscripts: x_i -> x_{i}
        result = re.sub(r'(\w)_(\w+)', r'\1_{\2}', result)

        # Superscripts: x^2 -> x^{2}
        result = re.sub(r'(\w)\^(\w+)', r'\1^{\2}', result)

        # Greek letters
        for name, latex in self.SYMBOLS.items():
            result = re.sub(rf'\b{name}\b', latex, result)

        return result


# Common physics/chemistry equations
COMMON_EQUATIONS = {
    "ideal_gas": Equation(
        latex="PV = nRT",
        description="Ideal gas law",
        label="eq:ideal_gas"
    ),
    "einstein_mass_energy": Equation(
        latex="E = mc^2",
        description="Mass-energy equivalence",
        label="eq:mass_energy"
    ),
    "gibbs_free_energy": Equation(
        latex="\\Delta G = \\Delta H - T\\Delta S",
        description="Gibbs free energy",
        label="eq:gibbs"
    ),
    "nernst": Equation(
        latex="E = E^0 - \\frac{RT}{nF}\\ln Q",
        description="Nernst equation",
        label="eq:nernst"
    ),
    "arrhenius": Equation(
        latex="k = A e^{-E_a/RT}",
        description="Arrhenius equation",
        label="eq:arrhenius"
    ),
    "schrodinger": Equation(
        latex="i\\hbar\\frac{\\partial}{\\partial t}\\Psi = \\hat{H}\\Psi",
        description="Time-dependent Schrödinger equation",
        label="eq:schrodinger"
    ),
    "michaelis_menten": Equation(
        latex="v = \\frac{V_{max}[S]}{K_m + [S]}",
        description="Michaelis-Menten enzyme kinetics",
        label="eq:michaelis_menten"
    ),
    "henderson_hasselbalch": Equation(
        latex="pH = pK_a + \\log\\frac{[A^-]}{[HA]}",
        description="Henderson-Hasselbalch equation",
        label="eq:henderson"
    ),
}


def get_equation(name: str) -> Equation:
    """Get a common equation by name."""
    return COMMON_EQUATIONS.get(name)


def quick_latex(content: str) -> str:
    """Quick helper to convert text to LaTeX."""
    formatter = LaTeXFormatter()
    return formatter.text_to_latex(content)
