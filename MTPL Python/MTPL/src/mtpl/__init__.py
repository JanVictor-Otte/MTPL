"""
mtpl — Python port of the MTPL (Music Theory Programming Library) core.

Mirrors the C++ header-only library at ``src/mtpl/``.  Templates are replaced
by dynamically-typed Python classes; arity checks and constant-propagation
safeguards are preserved as runtime assertions.
"""
from mtpl.core import *          # noqa: F401,F403
from mtpl.timed import *         # noqa: F401,F403
