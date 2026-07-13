from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
from typing import Any

import typer

telegram_app = typer.Typer(help="Configure and run the Telegram connector.")
webhook_app = typer.Typer(help="Register or remove the configured Telegram webhook.")
telegram_app.add_typer(webhook_app, name="webhook")


def _config() -> Any:
    from mana_agent.connectors.telegram.config import load_telegram_config
    return load_telegram_config()


async def _identity(config: Any, token: str | None = None) -> Any:
    from mana_agent.connectors.telegram.client import TelegramBotClient
    client = TelegramBotClient(token or config.bot_token, timeout_seconds=config.request_timeout_seconds)
    try:
        return await client.get_me()
    finally:
        await client.close()


@telegram_app.command("setup")
def setup(
    token: str = typer.Option(..., "--token", prompt=True, hide_input=True, help="Bot token used only for validation; it is not stored."),
    transport: str = typer.Option("auto", "--transport"),
    repository: Path = typer.Option(..., "--repository", exists=True, file_okay=False, resolve_path=True),
    allowed_user: list[int] = typer.Option([], "--allowed-user"),
    allowed_chat: list[int] = typer.Option([], "--allowed-chat"),
    bot_token_env: str = typer.Option("TELEGRAM_BOT_TOKEN", "--bot-token-env"),
) -> None:
    from mana_agent.connectors.telegram.config import TelegramConfig, save_telegram_config
    try:
        config = TelegramConfig(
            enabled=True, transport=transport, default_repository=str(repository),
            allowed_repository_roots=[str(repository)], allowed_users=allowed_user,
            allowed_chats=allowed_chat, bot_token_env=bot_token_env,
        )
        identity = asyncio.run(_identity(config, token=token))
    except (OSError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    save_telegram_config(config)
    typer.echo(f"Configured Telegram bot @{identity.username or identity.id}. Set {bot_token_env} in the connector process environment.")


@telegram_app.command("test")
def test_connector() -> None:
    config = _config()
    try:
        config.validate_runtime()
        identity = asyncio.run(_identity(config))
        typer.echo(f"Telegram credentials valid for @{identity.username or identity.id}; effective transport: {config.effective_transport}.")
        if config.effective_transport == "webhook":
            from mana_agent.connectors.telegram.client import TelegramBotClient
            async def info() -> dict[str, Any]:
                client = TelegramBotClient(config.bot_token, timeout_seconds=config.request_timeout_seconds)
                try:
                    return await client.get_webhook_info()
                finally:
                    await client.close()
            webhook = asyncio.run(info())
            typer.echo(f"Webhook pending updates: {int(webhook.get('pending_update_count') or 0)}.")
    except (OSError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@telegram_app.command("status")
def status() -> None:
    config = _config()
    from mana_agent.connectors.telegram.store import TelegramUpdateStore
    store = TelegramUpdateStore(config.database_path)
    typer.echo(f"Enabled: {config.enabled}")
    typer.echo(f"Configured transport: {config.transport}")
    typer.echo(f"Effective transport: {config.effective_transport}")
    typer.echo(f"Token configured: {bool(config.bot_token)}")
    typer.echo(f"Queue: {store.stats()}")
    typer.echo(f"Last completed update: {store.latest_completed_update() or 'none'}")
    typer.echo(f"Last error: {store.last_error() or 'none'}")


@telegram_app.command("info")
def info() -> None:
    config = _config()
    try:
        identity = asyncio.run(_identity(config))
    except (OSError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Bot ID: {identity.id}\nUsername: @{identity.username}" if identity.username else f"Bot ID: {identity.id}")


@telegram_app.command("start")
def start() -> None:
    config = _config()
    config.validate_runtime()
    pid_path = config.database_path.parent / "connector.pid"
    pid_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()), encoding="ascii")
    try:
        if config.effective_transport == "webhook":
            import uvicorn
            from mana_agent.api.app import create_app
            typer.echo(f"Starting Telegram webhook listener on {config.webhook.listen_host}:{config.webhook.listen_port}.")
            uvicorn.run(create_app(telegram_config=config), host=config.webhook.listen_host, port=config.webhook.listen_port)
        else:
            from mana_agent.connectors.telegram.connector import TelegramConnector
            async def polling() -> None:
                connector = TelegramConnector(config)
                await connector.start()
                typer.echo("Telegram polling started.")
                try:
                    assert connector._transport_task is not None
                    await connector._transport_task
                finally:
                    await connector.stop()
            asyncio.run(polling())
    finally:
        pid_path.unlink(missing_ok=True)


@telegram_app.command("stop")
def stop() -> None:
    config = _config()
    pid_path = config.database_path.parent / "connector.pid"
    try:
        pid = int(pid_path.read_text(encoding="ascii").strip())
        os.kill(pid, signal.SIGTERM)
    except FileNotFoundError as exc:
        raise typer.BadParameter("No running Telegram connector was recorded.") from exc
    except (ValueError, ProcessLookupError, PermissionError) as exc:
        pid_path.unlink(missing_ok=True)
        raise typer.BadParameter("The recorded Telegram connector process is not running.") from exc
    typer.echo("Telegram connector stop requested.")


@webhook_app.command("set")
def webhook_set() -> None:
    from mana_agent.connectors.telegram.client import TelegramBotClient
    config = _config()
    config.validate_runtime()
    async def run() -> bool:
        from urllib.parse import urljoin
        client = TelegramBotClient(config.bot_token, timeout_seconds=config.request_timeout_seconds)
        try:
            url = urljoin(config.webhook.public_url.rstrip("/") + "/", config.webhook.path.lstrip("/"))
            return await client.set_webhook(url, config.webhook_secret, drop_pending_updates=config.webhook.drop_pending_updates)
        finally:
            await client.close()
    asyncio.run(run())
    typer.echo("Telegram webhook registered.")


@webhook_app.command("delete")
def webhook_delete() -> None:
    from mana_agent.connectors.telegram.client import TelegramBotClient
    config = _config()
    async def run() -> bool:
        client = TelegramBotClient(config.bot_token, timeout_seconds=config.request_timeout_seconds)
        try:
            return await client.delete_webhook(drop_pending_updates=False)
        finally:
            await client.close()
    asyncio.run(run())
    typer.echo("Telegram webhook removed.")
