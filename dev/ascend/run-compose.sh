#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GENERATED_DIR="$SCRIPT_DIR/generated"

usage() {
  echo "usage: $0 attention <compose-args...>" >&2
  echo "       $0 control <compose-args...>" >&2
  echo "       $0 expert <pool-id> <compose-args...>" >&2
  echo "       $0 image <compose-args...>" >&2
}

if (( $# < 1 )); then
  usage
  exit 2
fi

target="$1"
shift

case "$target" in
  attention)
    files=(
      -f "$GENERATED_DIR/compose.attention.dev.yaml"
      -f "$GENERATED_DIR/compose.attention.yaml"
    )
    ;;
  control)
    files=(
      -f "$GENERATED_DIR/compose.control.dev.yaml"
      -f "$GENERATED_DIR/compose.control.yaml"
    )
    ;;
  expert)
    if (( $# < 1 )); then
      echo "missing expert pool ID" >&2
      usage
      exit 2
    fi

    pool="$1"
    shift

    if [[ ! "$pool" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
      echo "invalid expert pool ID: $pool" >&2
      exit 2
    fi

    pool_dir="$GENERATED_DIR/$pool"
    if [[ ! -d "$pool_dir" ]]; then
      echo "unknown expert pool: $pool" >&2
      exit 2
    fi

    files=(
      -f "$pool_dir/compose.expert.dev.yaml"
      -f "$pool_dir/compose.expert.yaml"
    )
    ;;
  image)
    files=(
      -f "$GENERATED_DIR/compose.build.yaml"
    )
    ;;
  *)
    echo "unknown target: $target" >&2
    usage
    exit 2
    ;;
esac

exec docker compose "${files[@]}" "$@"
