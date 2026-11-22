"""Exceptions shared across the NetLift packages.

``ContractError`` is raised whenever a value crosses one of the frozen
interfaces in ``docs/design/interfaces.md`` without meeting that interface's
invariants. It subclasses ``ValueError`` so callers that already catch bad
input keep working.
"""


class ContractError(ValueError):
    """Raised when a value breaks an invariant of a frozen interface."""
