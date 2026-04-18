"""Command submodule exports for mana_analyzer CLI.

Import side-effects to register optional Typer commands.
"""

# Optional command module: do not crash if a build omits this file.
try:
    from .cli import *  # noqa: F401,F403
    from .cli_internal import *  # noqa: F401,F403
    from .search_cli import *  # noqa: F401,F403
    from .flow_cli import *  # noqa: F401,F403
    from .analyze_cli import *  # noqa: F401,F403
    from .ask_cli import *  # noqa: F401,F403
    from .deps_cli import *  # noqa: F401,F403
    from .graph_cli import *  # noqa: F401,F403
    from .describe_cli import *  # noqa: F401,F403
    from .report_cli import *  # noqa: F401,F403
    from .chat_cli import *  # noqa: F401,F403
    from .ui_helpers import *
    from .output import *  # noqa: F401,F403
except ModuleNotFoundError:
    pass
