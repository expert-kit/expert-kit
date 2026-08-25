#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

role="$1"
shift

GENERATED_DIR=$SCRIPT_DIR/generated
STATIC_DIR=$SCRIPT_DIR/compose

case "$role" in
  attention)
    files=(
      -f $GENERATED_DIR/compose.attention.dev.yaml
      -f $GENERATED_DIR/compose.attention.yaml
    )
    ;;
  expert)
    files=(
      -f $GENERATED_DIR/compose.expert.dev.yaml
      -f $GENERATED_DIR/compose.expert.yaml
    )
    ;;
  *)
    echo "unknown role: $role" >&2
    exit 1
    ;;
esac

exec docker compose "${files[@]}" "$@"
