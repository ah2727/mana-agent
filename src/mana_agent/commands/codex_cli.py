"""Operational commands for the optional Codex coding backend."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import typer

from mana_agent.config.settings import Settings
from mana_agent.integrations.codex.config import CodexSettings
from mana_agent.integrations.codex.health import check_codex_health

codex_app = typer.Typer(help="Inspect and authenticate the optional Codex coding backend.", no_args_is_help=True)


def _settings() -> CodexSettings:
    return CodexSettings.from_mana_settings(Settings())


@codex_app.command("status")
def codex_status(
    root: str | None = typer.Option(None, "--root-dir", "--repo", help="Repository to validate."),
) -> None:
    """Report enablement, executable version, and repository access."""

    repository = Path(root or ".").expanduser().resolve()
    report = check_codex_health(_settings(), repository)
    typer.echo(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))


@codex_app.command("doctor")
def codex_doctor(
    root: str | None = typer.Option(None, "--root-dir", "--repo", help="Repository to validate."),
) -> None:
    """Run the read-only health check and exit non-zero when unavailable."""

    repository = Path(root or ".").expanduser().resolve()
    report = check_codex_health(_settings(), repository)
    typer.echo(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    if not report.healthy:
        raise typer.Exit(code=1)


@codex_app.command("login")
def codex_login() -> None:
    """Delegate authentication to the official Codex CLI."""

    settings = _settings()
    completed = subprocess.run([settings.codex_bin, "login"], check=False)
    if completed.returncode != 0:
        raise typer.Exit(code=completed.returncode)


@codex_app.command("logout")
def codex_logout() -> None:
    """Delegate sign-out to the official Codex CLI."""

    settings = _settings()
    completed = subprocess.run([settings.codex_bin, "logout"], check=False)
    if completed.returncode != 0:
        raise typer.Exit(code=completed.returncode)


__all__ = ["codex_app"]
