# Coding Agent Language Tooling Playbook

This document mirrors the language-aware command policy embedded in the coding-agent prompt.

## Purpose

- Keep tool execution deterministic and language-appropriate.
- Prevent noisy scans (`node_modules`, virtualenvs, build artifacts).
- Standardize install/test command selection across ecosystems.

## Detection Order

Before running install/test commands, detect the ecosystem from manifests/lockfiles:

- Python: `pyproject.toml`, `requirements*.txt`, `Pipfile`, `poetry.lock`, `uv.lock`, `tox.ini`
- Node/JS/TS: `package.json`, `package-lock.json`, `npm-shrinkwrap.json`, `pnpm-lock.yaml`, `yarn.lock`
- Rust: `Cargo.toml`, `Cargo.lock`
- Go: `go.mod`, `go.sum`
- Ruby: `Gemfile`, `Gemfile.lock`
- PHP: `composer.json`, `composer.lock`
- Dart/Flutter: `pubspec.yaml`, `.dart_tool/`
- JVM: `pom.xml`, `build.gradle`, `build.gradle.kts`, `gradlew`
- .NET: `*.sln`, `*.csproj`, `global.json`

## Search/Scan Exclusions

Ignore generated/vendor/cache paths unless explicitly requested:

- `node_modules/`
- `.venv/`
- `venv/`
- `__pycache__/`
- `.pytest_cache/`
- `.mypy_cache/`
- `.next/`
- `dist/`
- `build/`
- `coverage/`
- `target/`
- `vendor/`
- `out/`
- `.dart_tool/`
- `Pods/`
- `.mana_index/`

## Command Matrix

### Python

- Environment keywords: `.venv`, `venv`, `virtualenv`.
- Interpreter preference:
  - `.venv/bin/python` if present
  - otherwise `venv/bin/python` if present
- Install preference:
  1. `uv sync` (if `uv.lock` exists)
  2. `poetry install` (if `poetry.lock` exists)
  3. `python -m pip install -r requirements.txt`
  4. `pipenv install --dev` (Pipfile projects)
- Test preference:
  1. `pytest -q`
  2. `python -m pytest -q` (fallback)
  3. `tox -q` only when project config indicates tox

### Node / JavaScript / TypeScript

- Always exclude `node_modules/` from search.
- Install preference:
  1. `pnpm install --frozen-lockfile` (if `pnpm-lock.yaml`)
  2. `yarn install --frozen-lockfile` (if `yarn.lock`)
  3. `npm ci` (if `package-lock.json`/`npm-shrinkwrap.json`)
  4. `npm install` (fallback)
- Test preference:
  1. `pnpm test`
  2. `yarn test`
  3. `npm test`
- If no test script exists, report the blocker instead of inventing commands.

### Rust

- `cargo check` for quick validation.
- `cargo test` for test execution.

### Go

- `go test ./...`

### Ruby

- `bundle install`
- `bundle exec rspec` (or project-defined test task)

### PHP

- `composer install`
- `vendor/bin/phpunit` or `composer test` when configured

### Dart / Flutter

- Dart: `dart pub get`, `dart test`
- Flutter: `flutter pub get`, `flutter test`

### JVM / .NET

- Maven: `mvn test`
- Gradle: `./gradlew test`
- .NET: `dotnet test`

## Guardrails

- Pick one ecosystem based on detected manifests; do not run unrelated package managers.
- Prefer lockfile-aware commands over generic installs.
- After a failed command, inspect stderr and do at most one justified fallback.
- If required toolchain is missing, return a concrete blocker with the missing command/tool.
