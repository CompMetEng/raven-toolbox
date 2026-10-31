"""Small parsing helpers shared across raven_toolbox."""
from __future__ import annotations

import re

# A metabolite written as ``name[comp]``. The name is greedy so that, for a
# pathological name that itself contains brackets, the *last* ``[...]`` is taken
# as the compartment (matching RAVEN getIndexes' ``max(strfind('['))`` rule).
_NAME_COMP_RE = re.compile(r"^(?P<name>.+)\[(?P<comp>[^\[\]]+)\]$")


def parse_name_comp(token: str) -> tuple[str, str | None]:
    """Split a ``name[comp]`` token into ``(name, compartment)``.

    This covers the part of the ``metcomps`` mode of RAVEN ``getIndexes`` and of ``addRxns``
    eqnType 3 that cobra lacks: reading a metabolite written as its name plus a compartment
    in square brackets, for example ``"ATP[c]"``.

    Returns ``(name, None)`` when there is no trailing ``[...]``.

    Examples
    --------
    >>> parse_name_comp("ATP[c]")
    ('ATP', 'c')
    >>> parse_name_comp("ATP")
    ('ATP', None)
    >>> parse_name_comp("weird[name][m]")
    ('weird[name]', 'm')
    """
    match = _NAME_COMP_RE.match(token.strip())
    if match:
        return match.group("name").strip(), match.group("comp").strip()
    return token.strip(), None


def subsystem_to_str(value) -> str:
    """Convert a reaction ``subsystem`` to a ``str``.

    cobra defines ``Reaction.subsystem`` as a string. RAVEN, MATLAB and YAML input can supply
    a list or tuple of subsystem names, since a reaction can belong to several. A list or
    tuple becomes one string of its non-blank parts joined by ``;``. ``None`` gives ``""``;
    any other value is converted with ``str``. The modules that render or compare subsystems
    call this function, so they handle subsystems identically.

    Examples
    --------
    >>> subsystem_to_str("glycolysis")
    'glycolysis'
    >>> subsystem_to_str(["glycolysis", "TCA cycle"])
    'glycolysis;TCA cycle'
    >>> subsystem_to_str(None)
    ''
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ";".join(str(s) for s in value if str(s).strip())
    return str(value)
