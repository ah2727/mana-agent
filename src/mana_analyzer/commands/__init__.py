"""Command submodule exports for mana_analyzer CLI.

Import side-effects to register optional Typer commands.
"""

# Optional command module: do not crash if a build omits this file.
try:
    from .cli import *  # noqa: F401,F403
    from .ui_helpers import *
except ModuleNotFoundError:
    pass

