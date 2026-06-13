#!/usr/bin/env sh
set -eu

if [ "$#" -lt 1 ]; then
    echo "usage: scripts/run_tool.sh <tool> [args...]" >&2
    exit 2
fi

TOOL="$1"
shift
UV_BIN="${UV:-uv}"
UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
PREFER_VENV="${PYMAB_PREFER_VENV:-0}"
export UV_CACHE_DIR

if [ "$PREFER_VENV" = "1" ] && [ -x ".venv/bin/$TOOL" ]; then
    ".venv/bin/$TOOL" "$@"
    exit $?
fi

set +e
"$UV_BIN" run "$TOOL" "$@"
STATUS="$?"
set -e
if [ "$STATUS" -eq 0 ]; then
    exit 0
fi

echo "uv run $TOOL failed with status $STATUS; trying local .venv fallback" >&2

if [ -x ".venv/bin/$TOOL" ]; then
    ".venv/bin/$TOOL" "$@"
    exit $?
fi

case "$TOOL" in
    pytest)
        if [ -x ".venv/bin/python" ]; then
            ".venv/bin/python" -m unittest discover -v
            exit $?
        fi
        ;;
    python)
        if [ -x ".venv/bin/python" ]; then
            ".venv/bin/python" "$@"
            exit $?
        fi
        ;;
    pip-audit)
        if [ -x ".venv/bin/python" ]; then
            ".venv/bin/python" -m pip_audit "$@"
            exit $?
        fi
        ;;
    bandit)
        if [ -x ".venv/bin/python" ]; then
            ".venv/bin/python" -m bandit "$@"
            exit $?
        fi
        ;;
esac

echo "Could not find fallback for '$TOOL'. Run 'make sync' first." >&2
exit "$STATUS"
