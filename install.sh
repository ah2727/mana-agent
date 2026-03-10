#!/usr/bin/env bash
# GapCode installer
#
# Usage:
#   curl -fsSL https://gapgpt.app/install.sh | bash
#   curl -fsSL https://gapgpt.app/install.sh | bash -s -- --version 0.104.0
#
# Environment variables:
#   GAPCODE_INSTALL_DIR  Override install directory (default: ~/.gapcode/bin)
#   GAPCODE_VERSION      Install a specific version instead of latest

set -euo pipefail

RELEASES_BASE_URL="https://gapgpt.app/releases"
LATEST_VERSION_URL="https://gapgpt.app/api/v1/cli/latest-version"
LOG_EVENT_URL="https://gapgpt.app/api/v1/logs/event"
BINARY_NAME="gapcode"

INSTALL_ID=""
INSTALL_STARTED_AT=0
INSTALL_STATUS="started"
INSTALL_TARGET=""
INSTALL_VERSION=""
INSTALL_DIR=""
CURRENT_STAGE="bootstrap"
INSTALL_FAILURE_REASON=""
INSTALL_FAILURE_CMD=""
INSTALL_FAILURE_STAGE=""
INSTALL_TELEMETRY_ACTIVE=0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

info() { printf '\033[0;34m%s\033[0m\n' "$*"; }
success() { printf '\033[0;32m%s\033[0m\n' "$*"; }
warn() { printf '\033[0;33m%s\033[0m\n' "$*" >&2; }
error() {
  INSTALL_STATUS="failed"
  if [ -z "$INSTALL_FAILURE_REASON" ]; then
    INSTALL_FAILURE_REASON="$*"
  fi
  if [ -z "$INSTALL_FAILURE_STAGE" ]; then
    INSTALL_FAILURE_STAGE="$CURRENT_STAGE"
  fi
  printf '\033[0;31merror: %s\033[0m\n' "$*" >&2
  exit 1
}

need_cmd() {
  if ! command -v "$1" > /dev/null 2>&1; then
    error "need '$1' (command not found)"
  fi
}

json_escape() {
  printf '%s' "$1" | sed \
    -e 's/\\/\\\\/g' \
    -e 's/"/\\"/g' \
    -e ':a' -e 'N' -e '$!ba' \
    -e 's/\n/\\n/g' \
    -e 's/\r/\\r/g' \
    -e 's/\t/\\t/g'
}

generate_install_id() {
  if command -v uuidgen > /dev/null 2>&1; then
    uuidgen | tr '[:upper:]' '[:lower:]'
  else
    printf '%s-%s-%s' "$(date +%s)" "$$" "$RANDOM"
  fi
}

send_log_event() {
  local event_name="$1"
  local extra_json="${2:-}"

  if ! command -v curl > /dev/null 2>&1; then
    return 0
  fi

  local os_name arch_name now_ts duration_seconds payload
  os_name="$(uname -s 2>/dev/null || echo unknown)"
  arch_name="$(uname -m 2>/dev/null || echo unknown)"
  now_ts="$(date +%s 2>/dev/null || echo 0)"

  duration_seconds=0
  if [ "${INSTALL_STARTED_AT:-0}" -gt 0 ] 2>/dev/null; then
    duration_seconds=$((now_ts - INSTALL_STARTED_AT))
  fi

  payload="$(printf '{"name":"%s","data":{"source":"install.sh","install_id":"%s","status":"%s","os":"%s","arch":"%s","target":"%s","version":"%s","install_dir":"%s","duration_seconds":%s%s},"links":{}}' \
    "$(json_escape "$event_name")" \
    "$(json_escape "$INSTALL_ID")" \
    "$(json_escape "$INSTALL_STATUS")" \
    "$(json_escape "$os_name")" \
    "$(json_escape "$arch_name")" \
    "$(json_escape "$INSTALL_TARGET")" \
    "$(json_escape "$INSTALL_VERSION")" \
    "$(json_escape "$INSTALL_DIR")" \
    "$duration_seconds" \
    "${extra_json:+,$extra_json}")"

  curl -fsS -m 4 \
    -H 'Content-Type: application/json' \
    -d "$payload" \
    "$LOG_EVENT_URL" > /dev/null 2>&1 || true
}

on_install_error() {
  local _exit_code="$1"
  local line_no="$2"
  local command_str="$3"

  INSTALL_STATUS="failed"
  if [ -z "$INSTALL_FAILURE_REASON" ]; then
    INSTALL_FAILURE_REASON="command failed at line ${line_no}"
  fi
  if [ -z "$INSTALL_FAILURE_CMD" ]; then
    INSTALL_FAILURE_CMD="$command_str"
  fi
  if [ -z "$INSTALL_FAILURE_STAGE" ]; then
    INSTALL_FAILURE_STAGE="$CURRENT_STAGE"
  fi

  return 0
}

on_install_exit() {
  local exit_code="$1"

  if [ "$INSTALL_TELEMETRY_ACTIVE" -ne 1 ]; then
    return 0
  fi

  if [ "$exit_code" -eq 0 ] && [ "$INSTALL_STATUS" != "failed" ]; then
    INSTALL_STATUS="success"
    send_log_event "gapcode_install_succeeded"
    return 0
  fi

  INSTALL_STATUS="failed"
  if [ -z "$INSTALL_FAILURE_REASON" ]; then
    INSTALL_FAILURE_REASON="installation terminated unexpectedly"
  fi
  if [ -z "$INSTALL_FAILURE_STAGE" ]; then
    INSTALL_FAILURE_STAGE="$CURRENT_STAGE"
  fi

  send_log_event "gapcode_install_failed" "$(printf '"reason":"%s","failed_command":"%s","stage":"%s","exit_code":%s' \
    "$(json_escape "$INSTALL_FAILURE_REASON")" \
    "$(json_escape "$INSTALL_FAILURE_CMD")" \
    "$(json_escape "$INSTALL_FAILURE_STAGE")" \
    "$exit_code")"
}

# ---------------------------------------------------------------------------
# Detect platform
# ---------------------------------------------------------------------------

detect_target() {
  local os arch target
  CURRENT_STAGE="detect_target"

  os="$(uname -s)"
  arch="$(uname -m)"

  case "$os" in
    Linux)
      case "$arch" in
        x86_64)          target="x86_64-unknown-linux-gnu" ;;
        aarch64|arm64)   target="aarch64-unknown-linux-gnu" ;;
        *)               error "unsupported Linux architecture: $arch" ;;
      esac
      ;;
    Darwin)
      case "$arch" in
        x86_64)          target="x86_64-apple-darwin" ;;
        arm64|aarch64)   target="aarch64-apple-darwin" ;;
        *)               error "unsupported macOS architecture: $arch" ;;
      esac
      ;;
    *)
      error "unsupported operating system: $os (use install.ps1 on Windows)"
      ;;
  esac

  echo "$target"
}

# ---------------------------------------------------------------------------
# Fetch latest version from API
# ---------------------------------------------------------------------------

fetch_latest_version() {
  local response tag version
  CURRENT_STAGE="fetch_latest_version"
  need_cmd curl

  response="$(curl -fsSL "$LATEST_VERSION_URL")" \
    || error "failed to fetch latest version from $LATEST_VERSION_URL"

  # API returns {"tag_name":"rust-v0.104.0", "version":"0.104.0", ...}
  # Try the explicit "version" field first, fall back to parsing tag_name.
  if command -v jq > /dev/null 2>&1; then
    version="$(echo "$response" | jq -r '.version // empty')"
    if [ -z "$version" ]; then
      tag="$(echo "$response" | jq -r '.tag_name // empty')"
      version="${tag#rust-v}"
    fi
  else
    # Lightweight parse without jq
    version="$(echo "$response" | grep -oP '"version"\s*:\s*"\K[^"]+' 2>/dev/null || true)"
    if [ -z "$version" ]; then
      tag="$(echo "$response" | grep -oP '"tag_name"\s*:\s*"\K[^"]+' 2>/dev/null || true)"
      version="${tag#rust-v}"
    fi
  fi

  [ -n "$version" ] || error "could not determine latest version from API response"
  echo "$version"
}

# ---------------------------------------------------------------------------
# Download & install
# ---------------------------------------------------------------------------

download_and_install() {
  local version="$1"
  local target="$2"
  local install_dir="$3"
  CURRENT_STAGE="download_and_install"

  local archive_name="${BINARY_NAME}-${target}.tar.gz"
  local download_url="${RELEASES_BASE_URL}/v${version}/${archive_name}"

  local tmp_dir
  tmp_dir="$(mktemp -d)"

  info "Downloading GapCode v${version} for ${target}..."
  need_cmd curl
  need_cmd tar

  # Progress shown on stderr; only http_code captured in variable
  local http_code
  CURRENT_STAGE="download_archive"
  http_code="$(curl -fL -w '%{http_code}' -o "${tmp_dir}/${archive_name}" "$download_url")" \
    || error "download failed (HTTP ${http_code:-???}): ${download_url}"

  info "Extracting..."
  CURRENT_STAGE="extract_archive"
  tar -xzf "${tmp_dir}/${archive_name}" -C "$tmp_dir"

  # Archive may have binary as gapcode/gapcode-$target or, in a $target/ subdir, as codex/codex-$target
  local extracted_bin=""
  if [ -f "${tmp_dir}/${BINARY_NAME}" ]; then
    extracted_bin="${tmp_dir}/${BINARY_NAME}"
  elif [ -f "${tmp_dir}/${BINARY_NAME}-${target}" ]; then
    extracted_bin="${tmp_dir}/${BINARY_NAME}-${target}"
  elif [ -f "${tmp_dir}/${target}/codex-${target}" ]; then
    extracted_bin="${tmp_dir}/${target}/codex-${target}"
  elif [ -f "${tmp_dir}/${target}/codex" ]; then
    extracted_bin="${tmp_dir}/${target}/codex"
  else
    extracted_bin="$(find "$tmp_dir" -maxdepth 2 \( -name "${BINARY_NAME}*" -o -name "codex-${target}" -o \( -name "codex*" ! -name "*proxy*" \) \) -type f ! -name '*.tar.gz' ! -name '*.zst' | head -1)"
  fi

  [ -n "$extracted_bin" ] || error "could not find ${BINARY_NAME} binary in archive"

  CURRENT_STAGE="install_binary"
  mkdir -p "$install_dir"
  chmod +x "$extracted_bin"
  mv "$extracted_bin" "${install_dir}/${BINARY_NAME}"
  rm -rf "${tmp_dir:-}"

  success "Installed GapCode v${version} to ${install_dir}/${BINARY_NAME}"
}

# ---------------------------------------------------------------------------
# PATH setup
# ---------------------------------------------------------------------------

ensure_path() {
  local install_dir="$1"
  CURRENT_STAGE="ensure_path"

  case ":$PATH:" in
    *":${install_dir}:"*) return ;;
  esac

  local shell_name
  shell_name="$(basename "${SHELL:-/bin/sh}")"

  local profile_file=""
  case "$shell_name" in
    bash)
      if [ -f "$HOME/.bashrc" ]; then
        profile_file="$HOME/.bashrc"
      elif [ -f "$HOME/.bash_profile" ]; then
        profile_file="$HOME/.bash_profile"
      fi
      ;;
    zsh)
      profile_file="${ZDOTDIR:-$HOME}/.zshrc"
      ;;
    fish)
      ;;
    *)
      profile_file="$HOME/.profile"
      ;;
  esac

  local path_line="export PATH=\"${install_dir}:\$PATH\""

  if [ "$shell_name" = "fish" ]; then
    local fish_config="${XDG_CONFIG_HOME:-$HOME/.config}/fish/conf.d/gapcode.fish"
    mkdir -p "$(dirname "$fish_config")"
    echo "set -gx PATH \"${install_dir}\" \$PATH" > "$fish_config"
    info "Added ${install_dir} to PATH in ${fish_config}"
  elif [ -n "$profile_file" ]; then
    if ! grep -qF "$install_dir" "$profile_file" 2>/dev/null; then
      echo "" >> "$profile_file"
      echo "# GapCode" >> "$profile_file"
      echo "$path_line" >> "$profile_file"
      info "Added ${install_dir} to PATH in ${profile_file}"
    fi
  else
    warn "Could not detect shell profile. Add this to your shell config manually:"
    warn "  ${path_line}"
  fi

  export PATH="${install_dir}:$PATH"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
  local version=""
  local install_dir="${GAPCODE_INSTALL_DIR:-$HOME/.gapcode/bin}"
  CURRENT_STAGE="parse_arguments"

  while [ $# -gt 0 ]; do
    case "$1" in
      --version|-v)
        version="$2"
        shift 2
        ;;
      --dir|-d)
        install_dir="$2"
        shift 2
        ;;
      --help|-h)
        echo "Usage: install.sh [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --version, -v <VERSION>  Install a specific version (default: latest)"
        echo "  --dir, -d <DIR>          Install directory (default: ~/.gapcode/bin)"
        echo "  --help, -h               Show this help"
        exit 0
        ;;
      *)
        error "unknown option: $1"
        ;;
    esac
  done

  version="${version:-${GAPCODE_VERSION:-}}"
  INSTALL_DIR="$install_dir"
  INSTALL_VERSION="$version"
  INSTALL_TELEMETRY_ACTIVE=1
  send_log_event "gapcode_install_started"

  local target
  CURRENT_STAGE="resolve_target"
  target="$(detect_target)"
  INSTALL_TARGET="$target"

  if [ -z "$version" ]; then
    CURRENT_STAGE="resolve_version"
    info "Fetching latest version..."
    version="$(fetch_latest_version)"
    INSTALL_VERSION="$version"
  fi

  download_and_install "$version" "$target" "$install_dir"
  ensure_path "$install_dir"

  echo ""
  success "GapCode v${version} is ready! Run 'gapcode' to get started."
  info "Close and open a new terminal for 'gapcode' to be available in your PATH."
}

INSTALL_ID="$(generate_install_id)"
INSTALL_STARTED_AT="$(date +%s 2>/dev/null || echo 0)"
trap 'on_install_error $? $LINENO "$BASH_COMMAND"' ERR
trap 'on_install_exit $?' EXIT

main "$@"
