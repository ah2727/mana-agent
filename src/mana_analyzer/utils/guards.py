from pathlib import Path
import os, typer, functools

def guard_root(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # 1) pull out the `path` argument
        raw_path = kwargs.get("path") or (args[0] if args else None)
        if raw_path is None:
            raise typer.BadParameter("Internal error: no 'path' to guard")

        abs_path = Path(raw_path).resolve()

        # 2) determine allowed_root: env var if set, otherwise just the path itself
        root_env = os.environ.get("ANALYZE_ROOT", "").strip()
        if root_env:
            abs_root = Path(root_env).resolve()
        else:
            # default to the very project you asked to analyze
            abs_root = abs_path

        # 3) now perform the check
        if abs_path != abs_root and abs_root not in abs_path.parents:
            typer.echo(
                f"✋  Path {abs_path} is outside allowed root {abs_root}.", err=True
            )
            raise typer.Exit(code=1)

        # 4) all good—run the real function
        return fn(*args, **kwargs)

    return wrapper
