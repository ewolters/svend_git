"""
Synara DSL: Formal Hypothesis Language

Users formulate falsifiable hypotheses using structured operators:
- Quantifiers: ALWAYS, NEVER, SOMETIMES, ALL, NONE, SOME
- Logical: AND, OR, XOR, NOT, IMPLIES (if...then)
- Comparison: >, <, >=, <=, =, !=
- Variables: [variable_name] references data columns

Examples:
    if [num_holidays] > 3 then [monthly_sales] < 100000
    ALWAYS [temperature] > 20 AND [temperature] < 30
    NEVER [defect_rate] > 0.05 WHEN [shift] = "night"

The DSL produces an AST that the logic engine evaluates against data.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Union, Any
from enum import Enum


class Quantifier(Enum):
    """Quantifiers define scope of hypothesis."""
    ALWAYS = "always"       # ∀x: P(x)
    NEVER = "never"         # ∀x: ¬P(x)
    SOMETIMES = "sometimes" # ∃x: P(x)
    ALL = "all"             # ∀x ∈ S: P(x)
    NONE = "none"           # ∀x ∈ S: ¬P(x)
    SOME = "some"           # ∃x ∈ S: P(x)


class LogicalOp(Enum):
    """Logical operators."""
    AND = "and"
    OR = "or"
    XOR = "xor"
    NOT = "not"
    IMPLIES = "implies"  # if...then


class ComparisonOp(Enum):
    """Comparison operators."""
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    EQ = "="
    NEQ = "!="


# =============================================================================
# AST Nodes
# =============================================================================

@dataclass
class Variable:
    """Reference to a data column: [column_name]"""
    name: str

    def __repr__(self):
        return f"[{self.name}]"


@dataclass
class Literal:
    """A literal value: number or string."""
    value: Union[int, float, str, bool]

    def __repr__(self):
        if isinstance(self.value, str):
            return f'"{self.value}"'
        return str(self.value)


@dataclass
class Comparison:
    """A comparison expression: [x] > 10"""
    left: Union[Variable, Literal, "Expression"]
    op: ComparisonOp
    right: Union[Variable, Literal, "Expression"]

    def __repr__(self):
        return f"({self.left} {self.op.value} {self.right})"


@dataclass
class LogicalExpr:
    """A logical expression combining other expressions."""
    op: LogicalOp
    operands: list[Union["Comparison", "LogicalExpr", "Quantified"]]

    def __repr__(self):
        if self.op == LogicalOp.NOT:
            return f"NOT {self.operands[0]}"
        return f"({f' {self.op.value.upper()} '.join(str(o) for o in self.operands)})"


@dataclass
class Implication:
    """An implication: if P then Q"""
    antecedent: Union[Comparison, LogicalExpr]
    consequent: Union[Comparison, LogicalExpr]

    def __repr__(self):
        return f"(IF {self.antecedent} THEN {self.consequent})"


@dataclass
class DomainCondition:
    """A domain restriction: WHEN [shift] = "night" """
    condition: Union[Comparison, LogicalExpr]

    def __repr__(self):
        return f"WHEN {self.condition}"


@dataclass
class Quantified:
    """A quantified hypothesis."""
    quantifier: Quantifier
    body: Union[Comparison, LogicalExpr, Implication]
    domain: Optional[DomainCondition] = None
    over: Optional[Variable] = None  # For ALL/NONE/SOME: the variable quantified over

    def __repr__(self):
        parts = [self.quantifier.name]
        if self.over:
            parts.append(str(self.over))
        parts.append(str(self.body))
        if self.domain:
            parts.append(str(self.domain))
        return " ".join(parts)


# The full hypothesis can be any of these
Expression = Union[Comparison, LogicalExpr, Implication, Quantified]


@dataclass
class Hypothesis:
    """A complete parsed hypothesis."""
    raw: str
    ast: Expression
    variables: list[str]
    quantifiers: list[Quantifier]
    is_falsifiable: bool = True
    parse_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "variables": self.variables,
            "quantifiers": [q.value for q in self.quantifiers],
            "is_falsifiable": self.is_falsifiable,
            "parse_errors": self.parse_errors,
            "structure": self._ast_to_dict(self.ast),
        }

    def _ast_to_dict(self, node) -> dict:
        if isinstance(node, Variable):
            return {"type": "variable", "name": node.name}
        elif isinstance(node, Literal):
            return {"type": "literal", "value": node.value}
        elif isinstance(node, Comparison):
            return {
                "type": "comparison",
                "op": node.op.value,
                "left": self._ast_to_dict(node.left),
                "right": self._ast_to_dict(node.right),
            }
        elif isinstance(node, LogicalExpr):
            return {
                "type": "logical",
                "op": node.op.value,
                "operands": [self._ast_to_dict(o) for o in node.operands],
            }
        elif isinstance(node, Implication):
            return {
                "type": "implication",
                "antecedent": self._ast_to_dict(node.antecedent),
                "consequent": self._ast_to_dict(node.consequent),
            }
        elif isinstance(node, DomainCondition):
            return {
                "type": "domain",
                "condition": self._ast_to_dict(node.condition),
            }
        elif isinstance(node, Quantified):
            result = {
                "type": "quantified",
                "quantifier": node.quantifier.value,
                "body": self._ast_to_dict(node.body),
            }
            if node.domain:
                result["domain"] = self._ast_to_dict(node.domain)
            if node.over:
                result["over"] = self._ast_to_dict(node.over)
            return result
        return {"type": "unknown"}


# =============================================================================
# Parser
# =============================================================================

class DSLParser:
    """
    Parse hypothesis strings into AST.

    Grammar (informal):
        hypothesis := quantified | implication | logical_expr | comparison
        quantified := QUANTIFIER [variable] body [domain]
        implication := IF condition THEN condition
        logical_expr := comparison ((AND|OR|XOR) comparison)*
        comparison := term COMP_OP term
        term := variable | literal | '(' logical_expr ')'
        variable := '[' IDENTIFIER ']'
        literal := NUMBER | STRING
        domain := WHEN condition
    """

    QUANTIFIERS = {
        "always": Quantifier.ALWAYS,
        "never": Quantifier.NEVER,
        "sometimes": Quantifier.SOMETIMES,
        "all": Quantifier.ALL,
        "none": Quantifier.NONE,
        "some": Quantifier.SOME,
    }

    COMPARISON_OPS = {
        ">=": ComparisonOp.GTE,
        "<=": ComparisonOp.LTE,
        ">": ComparisonOp.GT,
        "<": ComparisonOp.LT,
        "=": ComparisonOp.EQ,
        "!=": ComparisonOp.NEQ,
        "==": ComparisonOp.EQ,
    }

    def __init__(self):
        self.tokens: list[str] = []
        self.pos: int = 0
        self.variables: list[str] = []
        self.quantifiers: list[Quantifier] = []
        self.errors: list[str] = []

    def parse(self, text: str) -> Hypothesis:
        """Parse a hypothesis string."""
        self.tokens = self._tokenize(text)
        self.pos = 0
        self.variables = []
        self.quantifiers = []
        self.errors = []

        if not self.tokens:
            return Hypothesis(
                raw=text,
                ast=Literal(True),
                variables=[],
                quantifiers=[],
                is_falsifiable=False,
                parse_errors=["Empty hypothesis"],
            )

        try:
            ast = self._parse_hypothesis()
        except Exception as e:
            self.errors.append(str(e))
            ast = Literal(True)

        # Check falsifiability
        is_falsifiable = self._check_falsifiable(ast)

        return Hypothesis(
            raw=text,
            ast=ast,
            variables=list(set(self.variables)),
            quantifiers=self.quantifiers,
            is_falsifiable=is_falsifiable,
            parse_errors=self.errors,
        )

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize the input string."""
        # Pattern for tokens: variables, strings, numbers, operators, keywords
        pattern = r'''
            \[[^\]]+\]           |  # Variable: [name]
            "[^"]*"              |  # String: "value"
            '[^']*'              |  # String: 'value'
            \d+\.?\d*            |  # Number
            >=|<=|!=|==          |  # Multi-char operators
            [><=]                |  # Single-char operators
            \(|\)                |  # Parentheses
            \w+                     # Keywords/identifiers
        '''
        tokens = re.findall(pattern, text, re.VERBOSE | re.IGNORECASE)
        return [t.strip() for t in tokens if t.strip()]

    def _current(self) -> Optional[str]:
        """Get current token."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _advance(self) -> Optional[str]:
        """Advance and return current token."""
        token = self._current()
        self.pos += 1
        return token

    def _expect(self, expected: str) -> str:
        """Expect a specific token."""
        token = self._current()
        if token is None:
            raise ValueError(f"Expected '{expected}', got end of input")
        if token.lower() != expected.lower():
            raise ValueError(f"Expected '{expected}', got '{token}'")
        return self._advance()

    def _parse_hypothesis(self) -> Expression:
        """Parse a complete hypothesis."""
        token = self._current()
        if not token:
            raise ValueError("Empty expression")

        lower = token.lower()

        # Check for quantifier
        if lower in self.QUANTIFIERS:
            return self._parse_quantified()

        # Check for IF...THEN
        if lower == "if":
            return self._parse_implication()

        # Otherwise parse as logical expression
        return self._parse_logical_expr()

    def _parse_quantified(self) -> Quantified:
        """Parse quantified expression: ALWAYS [x] > 10 WHEN [y] = 5"""
        token = self._advance()
        quantifier = self.QUANTIFIERS[token.lower()]
        self.quantifiers.append(quantifier)

        # For ALL/NONE/SOME, check for variable being quantified over
        over = None
        if quantifier in (Quantifier.ALL, Quantifier.NONE, Quantifier.SOME):
            if self._current() and self._current().startswith("["):
                over = self._parse_variable()

        # Parse body
        body = self._parse_logical_expr_or_implication()

        # Check for domain condition
        domain = None
        if self._current() and self._current().lower() == "when":
            domain = self._parse_domain()

        return Quantified(
            quantifier=quantifier,
            body=body,
            domain=domain,
            over=over,
        )

    def _parse_implication(self) -> Implication:
        """Parse IF...THEN expression."""
        self._expect("if")
        antecedent = self._parse_logical_expr()
        self._expect("then")
        consequent = self._parse_logical_expr()
        return Implication(antecedent=antecedent, consequent=consequent)

    def _parse_logical_expr_or_implication(self) -> Union[LogicalExpr, Implication, Comparison]:
        """Parse either logical expression or implication."""
        if self._current() and self._current().lower() == "if":
            return self._parse_implication()
        return self._parse_logical_expr()

    def _parse_logical_expr(self) -> Union[LogicalExpr, Comparison]:
        """Parse logical expression: comparison ((AND|OR|XOR) comparison)*"""
        left = self._parse_comparison_or_group()

        # Check for logical operators
        while self._current() and self._current().lower() in ("and", "or", "xor"):
            op_token = self._advance().lower()
            op = LogicalOp(op_token)
            right = self._parse_comparison_or_group()

            if isinstance(left, LogicalExpr) and left.op == op:
                # Flatten: (a AND b) AND c -> (a AND b AND c)
                left.operands.append(right)
            else:
                left = LogicalExpr(op=op, operands=[left, right])

        return left

    def _parse_comparison_or_group(self) -> Union[Comparison, LogicalExpr]:
        """Parse comparison or parenthesized group."""
        if self._current() == "(":
            self._advance()  # consume '('
            expr = self._parse_logical_expr()
            if self._current() == ")":
                self._advance()  # consume ')'
            else:
                self.errors.append("Missing closing parenthesis")
            return expr

        if self._current() and self._current().lower() == "not":
            self._advance()
            operand = self._parse_comparison_or_group()
            return LogicalExpr(op=LogicalOp.NOT, operands=[operand])

        return self._parse_comparison()

    def _parse_comparison(self) -> Comparison:
        """Parse comparison: term COMP_OP term"""
        left = self._parse_term()

        op_token = self._current()
        if op_token not in self.COMPARISON_OPS:
            # Might be a bare variable/literal - create equality with True
            return Comparison(left=left, op=ComparisonOp.EQ, right=Literal(True))

        self._advance()
        op = self.COMPARISON_OPS[op_token]
        right = self._parse_term()

        return Comparison(left=left, op=op, right=right)

    def _parse_term(self) -> Union[Variable, Literal]:
        """Parse term: variable or literal."""
        token = self._current()
        if not token:
            raise ValueError("Expected term, got end of input")

        if token.startswith("["):
            return self._parse_variable()
        else:
            return self._parse_literal()

    def _parse_variable(self) -> Variable:
        """Parse variable: [column_name]"""
        token = self._advance()
        # Extract name from [name]
        name = token[1:-1]  # Remove brackets
        self.variables.append(name)
        return Variable(name=name)

    def _parse_literal(self) -> Literal:
        """Parse literal: number or string."""
        token = self._advance()

        # String literal
        if token.startswith('"') or token.startswith("'"):
            return Literal(value=token[1:-1])

        # Boolean
        if token.lower() == "true":
            return Literal(value=True)
        if token.lower() == "false":
            return Literal(value=False)

        # Number
        try:
            if "." in token:
                return Literal(value=float(token))
            return Literal(value=int(token))
        except ValueError:
            # Treat as string
            return Literal(value=token)

    def _parse_domain(self) -> DomainCondition:
        """Parse domain condition: WHEN condition"""
        self._expect("when")
        condition = self._parse_logical_expr()
        return DomainCondition(condition=condition)

    def _check_falsifiable(self, ast: Expression) -> bool:
        """Check if the hypothesis is falsifiable."""
        # A hypothesis is falsifiable if:
        # 1. It makes a specific claim about data
        # 2. It uses quantifiers properly
        # 3. It references at least one variable

        if not self.variables:
            self.errors.append("Hypothesis must reference at least one variable")
            return False

        # Check for tautologies
        if self._is_tautology(ast):
            self.errors.append("Hypothesis is a tautology")
            return False

        return True

    def _is_tautology(self, ast: Expression) -> bool:
        """Check if expression is a tautology."""
        # Simple tautology detection
        if isinstance(ast, Comparison):
            if isinstance(ast.left, Variable) and isinstance(ast.right, Variable):
                if ast.left.name == ast.right.name and ast.op == ComparisonOp.EQ:
                    return True  # [x] = [x]
        return False


# =============================================================================
# Pretty Printer
# =============================================================================

def format_hypothesis(hypothesis: Hypothesis, style: str = "natural") -> str:
    """
    Format hypothesis for display.

    style: "natural" - human readable
           "formal" - mathematical notation
           "code" - programming style
    """
    if style == "formal":
        return _format_formal(hypothesis.ast)
    elif style == "code":
        return _format_code(hypothesis.ast)
    else:
        return _format_natural(hypothesis.ast)


def _format_natural(node: Expression) -> str:
    """Format as natural language."""
    if isinstance(node, Variable):
        return node.name
    elif isinstance(node, Literal):
        return str(node.value)
    elif isinstance(node, Comparison):
        op_map = {
            ComparisonOp.GT: "is greater than",
            ComparisonOp.LT: "is less than",
            ComparisonOp.GTE: "is at least",
            ComparisonOp.LTE: "is at most",
            ComparisonOp.EQ: "equals",
            ComparisonOp.NEQ: "does not equal",
        }
        return f"{_format_natural(node.left)} {op_map[node.op]} {_format_natural(node.right)}"
    elif isinstance(node, LogicalExpr):
        if node.op == LogicalOp.NOT:
            return f"not ({_format_natural(node.operands[0])})"
        sep = f" {node.op.value} "
        return sep.join(_format_natural(o) for o in node.operands)
    elif isinstance(node, Implication):
        return f"if {_format_natural(node.antecedent)}, then {_format_natural(node.consequent)}"
    elif isinstance(node, Quantified):
        quant_map = {
            Quantifier.ALWAYS: "always",
            Quantifier.NEVER: "never",
            Quantifier.SOMETIMES: "sometimes",
            Quantifier.ALL: "for all",
            Quantifier.NONE: "for none",
            Quantifier.SOME: "for some",
        }
        result = f"{quant_map[node.quantifier]}: {_format_natural(node.body)}"
        if node.domain:
            result += f" (when {_format_natural(node.domain.condition)})"
        return result
    return str(node)


def _format_formal(node: Expression) -> str:
    """Format as mathematical notation."""
    if isinstance(node, Variable):
        return node.name
    elif isinstance(node, Literal):
        return str(node.value)
    elif isinstance(node, Comparison):
        return f"{_format_formal(node.left)} {node.op.value} {_format_formal(node.right)}"
    elif isinstance(node, LogicalExpr):
        if node.op == LogicalOp.NOT:
            return f"¬({_format_formal(node.operands[0])})"
        op_map = {LogicalOp.AND: "∧", LogicalOp.OR: "∨", LogicalOp.XOR: "⊕"}
        sep = f" {op_map.get(node.op, node.op.value)} "
        return f"({sep.join(_format_formal(o) for o in node.operands)})"
    elif isinstance(node, Implication):
        return f"({_format_formal(node.antecedent)} → {_format_formal(node.consequent)})"
    elif isinstance(node, Quantified):
        quant_map = {
            Quantifier.ALWAYS: "∀",
            Quantifier.NEVER: "∀¬",
            Quantifier.SOMETIMES: "∃",
            Quantifier.ALL: "∀",
            Quantifier.NONE: "∀¬",
            Quantifier.SOME: "∃",
        }
        result = f"{quant_map[node.quantifier]} {_format_formal(node.body)}"
        if node.domain:
            result = f"[{_format_formal(node.domain.condition)}] {result}"
        return result
    return str(node)


def _format_code(node: Expression) -> str:
    """Format as code/pseudocode."""
    if isinstance(node, Variable):
        return f"data['{node.name}']"
    elif isinstance(node, Literal):
        if isinstance(node.value, str):
            return f"'{node.value}'"
        return str(node.value)
    elif isinstance(node, Comparison):
        op_map = {
            ComparisonOp.GT: ">",
            ComparisonOp.LT: "<",
            ComparisonOp.GTE: ">=",
            ComparisonOp.LTE: "<=",
            ComparisonOp.EQ: "==",
            ComparisonOp.NEQ: "!=",
        }
        return f"({_format_code(node.left)} {op_map[node.op]} {_format_code(node.right)})"
    elif isinstance(node, LogicalExpr):
        if node.op == LogicalOp.NOT:
            return f"not {_format_code(node.operands[0])}"
        op_map = {LogicalOp.AND: "and", LogicalOp.OR: "or", LogicalOp.XOR: "^"}
        sep = f" {op_map.get(node.op, node.op.value)} "
        return f"({sep.join(_format_code(o) for o in node.operands)})"
    elif isinstance(node, Implication):
        return f"(not {_format_code(node.antecedent)} or {_format_code(node.consequent)})"
    elif isinstance(node, Quantified):
        quant_map = {
            Quantifier.ALWAYS: "all",
            Quantifier.NEVER: "not any",
            Quantifier.SOMETIMES: "any",
            Quantifier.ALL: "all",
            Quantifier.NONE: "not any",
            Quantifier.SOME: "any",
        }
        body = _format_code(node.body)
        if node.domain:
            domain = _format_code(node.domain.condition)
            return f"{quant_map[node.quantifier]}({body} for row in data if {domain})"
        return f"{quant_map[node.quantifier]}({body} for row in data)"
    return str(node)
