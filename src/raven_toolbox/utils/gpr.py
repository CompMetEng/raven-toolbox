"""GPR (gene-protein-reaction rule) linting.

Parsing of grRule strings in the grammar of RAVEN's ``parseGrRule``, assignment of rules to
reactions, the score-aggregation modes for GPRs, and a linter for GPRs that are not in
disjunctive normal form (DNF). The linter works on cobra's GPR AST. Syntax normalisation of
GPRs is done by cobra on assignment and is not repeated here.

A GPR is in DNF when it is an OR of AND-clauses of single genes, for example ``(G1 and G2) or G3``.
A rule in which an AND contains an OR, for example ``(G1 or G2) and (G3 or G4)``, is valid for cobra
but ambiguous for the isoenzyme and complex reasoning used across RAVEN and GECKO.
:func:`find_non_dnf_grrules` returns such rules as structured data.
"""
from __future__ import annotations

import ast
import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import cobra
from cobra.core.gene import GPR

#: Score-aggregation functions for combining gene scores across a GPR: isozymes
#: (genes joined by OR) and complex subunits (joined by AND). Shared by the
#: gene→reaction scoring (:mod:`raven_toolbox.init.score`) and low-score-gene
#: pruning (:mod:`raven_toolbox.init.genes`) so both validate the same way.
AGGREGATORS: dict[str, Callable] = {
    "min": min,
    "max": max,
    "median": statistics.median,
    "average": statistics.fmean,
}


def resolve_aggregators(
    isozyme_scoring: str, complex_scoring: str
) -> tuple[Callable, Callable]:
    """Validate the scoring-mode names and return ``(isozyme_fn, complex_fn)``.

    Raises ``ValueError`` naming the offending argument if either mode is not one
    of :data:`AGGREGATORS`.
    """
    for name, value in (
        ("isozyme_scoring", isozyme_scoring),
        ("complex_scoring", complex_scoring),
    ):
        if value not in AGGREGATORS:
            raise ValueError(f"{name} must be one of {sorted(AGGREGATORS)}; got {value!r}.")
    return AGGREGATORS[isozyme_scoring], AGGREGATORS[complex_scoring]


def _contains_or(node: ast.AST | None) -> bool:
    """True if the subtree of ``node`` contains an OR operator."""
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.Or):
            return True
        return any(_contains_or(value) for value in node.values)
    return False


def _is_dnf_node(node: ast.AST | None) -> bool:
    """True if the AST rooted at ``node`` is in disjunctive normal form.

    DNF here means no AND operator has an OR beneath it, that is, the rule is a single gene,
    an AND-complex of genes, or an OR of those.
    """
    if node is None or isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return not any(_contains_or(value) for value in node.values)
        # OR: every disjunct must itself be DNF
        return all(_is_dnf_node(value) for value in node.values)
    # Unknown node type: don't flag it as a problem.
    return True


def is_dnf(gpr: GPR | str | None) -> bool:
    """Return whether a GPR is in disjunctive normal form (OR of AND-complexes).

    Parameters
    ----------
    gpr
        A cobra :class:`~cobra.core.gene.GPR`, a grRule string, or ``None``.
        A string is read with :func:`parse_gr_rule`. An empty or ``None`` rule is in DNF.

    Raises
    ------
    ValueError
        If a rule string is not valid in the grammar of :func:`parse_gr_rule` (unbalanced
        parentheses, a dangling operator, two genes without an operator between them, ``()``).

    Examples
    --------
    >>> is_dnf("(G1 and G2) or G3")
    True
    >>> is_dnf("(G1 or G2) and G3")
    False
    """
    parsed = parse_gr_rule(gpr) if isinstance(gpr, str) else gpr
    if parsed is None:
        return True
    return _is_dnf_node(parsed.body)


# The characters for which MATLAB's isspace is true, the whitespace of RAVEN's parseGrRule. Python's
# str.isspace also counts U+0085, U+00A0, U+2007 and U+202F; here they are part of a word.
_RULE_SPACE = r"\t\n\x0b\x0c\r\x1c-\x20\u1680\u2000-\u2006\u2008-\u200a\u2028\u2029\u205f\u3000"
_RULE_TOKEN = re.compile(rf"\(|\)|[^{_RULE_SPACE}()]+")
_RULE_BLANK = re.compile(rf"[{_RULE_SPACE}]*")
_AND_WORDS = frozenset({"and", "&", "&&"})
_OR_WORDS = frozenset({"or", "|", "||"})


def rule_tokens(rule: str) -> list[tuple[str, str]]:
    """Split a gene rule into ``(kind, text)`` pairs.

    The kind is ``(``, ``)``, ``and``, ``or`` or ``gene``.

    A word is everything up to whitespace or a bracket, whitespace being the characters for which
    MATLAB's ``isspace`` is true (not U+00A0, U+0085, U+2007 or U+202F). The operators are
    ``and``/``or`` in any case and the symbols ``&``, ``&&``, ``|`` and ``||``, each as a whole
    word, as RAVEN's ``parseGrRule`` reads them; every other word is a gene id, so ``G1&G2`` is
    one gene.
    """
    tokens = []
    for match in _RULE_TOKEN.finditer(rule):
        word = match.group()
        low = word.lower()
        if word in ("(", ")"):
            tokens.append((word, word))
        elif low in _AND_WORDS:
            tokens.append(("and", word))
        elif low in _OR_WORDS:
            tokens.append(("or", word))
        else:
            tokens.append(("gene", word))
    return tokens


def parse_gr_rule(rule: str | None) -> GPR | None:
    """Parse a grRule string in the grammar of RAVEN's ``parseGrRule``.

    The grammar, with AND binding tighter than OR::

        expr   := term ( OR term )*
        term   := factor ( AND factor )*
        factor := gene | '(' expr ')'

    AND is ``and`` in any case, ``&`` or ``&&``; OR is ``or`` in any case, ``|`` or ``||``.
    Operators and brackets are separated from gene ids by whitespace or a bracket; any other
    word is a gene id. ``curate_model_from_tables``/``batch_curate``, :func:`is_dnf` and
    :func:`~raven_toolbox.manipulation.gpr_to_dnf` all read rules with this parser, so they
    accept and refuse the same rules as RAVEN.

    Parameters
    ----------
    rule
        A grRule such as ``"(G1 and G2) or G3"``.

    Returns
    -------
    GPR or None
        The parsed rule; ``None`` for ``None``, an empty or an all-blank rule.

    Raises
    ------
    ValueError
        If the rule is malformed: an operator without an operand, unbalanced or empty
        parentheses, or two genes without an operator between them. The messages are those
        of RAVEN's ``RAVEN:badGrRule`` error.
    """
    if rule is None or _RULE_BLANK.fullmatch(rule):
        return None
    tokens = rule_tokens(rule)

    def factor(pos: int) -> tuple[ast.expr, int]:
        if pos >= len(tokens):
            raise ValueError(f"grRule ends after an operator: {rule}")
        kind, text = tokens[pos]
        if kind == "(":
            node, pos = expr(pos + 1)
            if pos >= len(tokens) or tokens[pos][0] != ")":
                raise ValueError(f"Unbalanced parentheses in grRule: {rule}")
            return node, pos + 1
        if kind == "gene":
            return ast.Name(id=text, ctx=ast.Load()), pos + 1
        raise ValueError(f'Expected a gene but found "{text}" in grRule: {rule}')

    def chain(pos: int, sub, kind: str, op: ast.boolop) -> tuple[ast.expr, int]:
        node, pos = sub(pos)
        values = [node]
        while pos < len(tokens) and tokens[pos][0] == kind:
            node, pos = sub(pos + 1)
            values.append(node)
        if len(values) == 1:
            return values[0], pos
        return ast.BoolOp(op=op, values=values), pos

    def term(pos: int) -> tuple[ast.expr, int]:
        return chain(pos, factor, "and", ast.And())

    def expr(pos: int) -> tuple[ast.expr, int]:
        return chain(pos, term, "or", ast.Or())

    body, end = expr(0)
    if end < len(tokens):
        raise ValueError(f'Unexpected "{tokens[end][1]}" in grRule: {rule}')
    return GPR(ast.Expression(body=body))


def rule_to_gpr(rule: str | None) -> GPR:
    """The :class:`~cobra.core.gene.GPR` of a rule string as :func:`parse_gr_rule` reads it.

    An empty or ``None`` rule gives an empty GPR. A rule that :func:`parse_gr_rule` rejects
    is read with cobra's ``GPR.from_string``.
    """
    try:
        gpr = parse_gr_rule(rule)
    except ValueError:
        return GPR.from_string(rule)
    return gpr if gpr is not None else GPR()


def set_gene_rule(rxn: cobra.Reaction, rule: GPR | str | None) -> None:
    """Give ``rxn`` a gene rule in RAVEN's grammar.

    A :class:`~cobra.core.gene.GPR` is copied as it is; a string is read with
    :func:`rule_to_gpr`. Assigning a string to ``rxn.gene_reaction_rule`` reads it with
    cobra's grammar, which splits a gene id such as ``G1&G2`` into ``G1`` and ``G2``.

    Callers include the functions of ``raven_toolbox.manipulation`` (add, change,
    compartments, expand, irreversible, merge, transfer), ``raven_toolbox.localization.assign``,
    ``raven_toolbox.io.yaml``, ``raven_toolbox.io.mat``, ``raven_toolbox.init.genes``,
    ``raven_toolbox.init.taskfill``, ``raven_toolbox.reconstruction.homology.homology`` and
    ``raven_toolbox.reconstruction.kegg.assemble``. The duplicate
    merge and ``group_linear_reactions`` in ``raven_toolbox.manipulation.simplify`` assign
    ``gene_reaction_rule`` directly, and ``raven_toolbox.curation.batch`` assigns ``gpr``
    directly.
    """
    rxn.gpr = rule.copy() if isinstance(rule, GPR) else rule_to_gpr(rule)


@dataclass(frozen=True)
class GPRIssue:
    """A reaction whose GPR is flagged by the linter.

    Attributes
    ----------
    reaction_id
        ID of the reaction.
    gpr
        The (already cobra-normalised) grRule string.
    reason
        Human-readable explanation of why the reaction was flagged.
    """

    reaction_id: str
    gpr: str
    reason: str


_NON_DNF_REASON = (
    "GPR is not in disjunctive normal form (an AND clause contains an OR). "
    "Isoenzyme and complex reasoning assume an OR of AND-complexes, so it needs the expanded "
    'form: the rule "(G1 or G2) and (G3 or G4)" is equivalent to '
    '"(G1 and G3) or (G1 and G4) or (G2 and G3) or (G2 and G4)". '
    "expand_model splits a rule when its expanded form has more than one clause."
)


def find_non_dnf_grrules(model: cobra.Model) -> list[GPRIssue]:
    """Find reactions whose GPR is not in disjunctive normal form ("OR of AND-complexes").

    The check runs on cobra's GPR AST. Reactions with no GPR are skipped.

    Parameters
    ----------
    model : cobra.Model
        Model whose reactions are checked. It is not modified.

    Returns
    -------
    list of GPRIssue
        One entry per flagged reaction, in model reaction order. Empty if all
        GPRs are in DNF.
    """
    issues: list[GPRIssue] = []
    for rxn in model.reactions:
        if not rxn.gene_reaction_rule:
            continue
        if not is_dnf(rxn.gpr):
            issues.append(GPRIssue(rxn.id, rxn.gene_reaction_rule, _NON_DNF_REASON))
    return issues
