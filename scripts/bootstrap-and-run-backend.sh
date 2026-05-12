#!/usr/bin/env sh
set -eu

APP_DIR="${APP_DIR:-/app}"
BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-8090}"

RUNTIME_ASSETS_REPO_URL="${RUNTIME_ASSETS_REPO_URL:-https://github.com/tomasfratrik/thesis-runtime-assets.git}"
RUNTIME_ASSETS_CACHE_DIR="${RUNTIME_ASSETS_CACHE_DIR:-/tmp/runtime-assets}"

MODEL_REPO_URL="${MODEL_REPO_URL:-https://huggingface.co/tomasfratrik/thesis-model}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$APP_DIR/artifacts/model-repo}"

runtime_assets_missing() {
    if [ ! -f "$APP_DIR/app_data/app.sqlite3" ]; then
        return 0
    fi
    if [ ! -f "$APP_DIR/artifacts/image_embeddings.npy" ]; then
        return 0
    fi
    if [ ! -f "$APP_DIR/artifacts/image_meta.json" ]; then
        return 0
    fi
    if [ ! -d "$APP_DIR/previews" ]; then
        return 0
    fi
    if [ -z "$(ls -A "$APP_DIR/previews" 2>/dev/null)" ]; then
        return 0
    fi
    return 1
}

clone_repo_if_missing() {
    repo_url="$1"
    target_dir="$2"
    if [ -d "$target_dir/.git" ]; then
        return 0
    fi

    rm -rf "$target_dir"
    git clone --depth 1 "$repo_url" "$target_dir"
    git -C "$target_dir" lfs pull
}

copy_if_present() {
    source_dir="$1"
    target_dir="$2"
    if [ ! -d "$source_dir" ]; then
        return 0
    fi

    mkdir -p "$target_dir"
    cp -a "$source_dir/." "$target_dir/"
}

copy_file_if_present() {
    source_file="$1"
    target_file="$2"
    if [ ! -f "$source_file" ]; then
        return 0
    fi

    mkdir -p "$(dirname "$target_file")"
    cp "$source_file" "$target_file"
}

bootstrap_runtime_assets() {
    if ! runtime_assets_missing; then
        return 0
    fi

    echo "[bootstrap] Downloading runtime assets from $RUNTIME_ASSETS_REPO_URL"
    clone_repo_if_missing "$RUNTIME_ASSETS_REPO_URL" "$RUNTIME_ASSETS_CACHE_DIR"

    mkdir -p "$APP_DIR/app_data" "$APP_DIR/artifacts" "$APP_DIR/previews"
    copy_if_present "$RUNTIME_ASSETS_CACHE_DIR/app_data" "$APP_DIR/app_data"
    copy_if_present "$RUNTIME_ASSETS_CACHE_DIR/artifacts" "$APP_DIR/artifacts"
    copy_if_present "$RUNTIME_ASSETS_CACHE_DIR/previews" "$APP_DIR/previews"
    copy_file_if_present "$RUNTIME_ASSETS_CACHE_DIR/app.sqlite3" "$APP_DIR/app_data/app.sqlite3"
    copy_file_if_present "$RUNTIME_ASSETS_CACHE_DIR/image_embeddings.npy" "$APP_DIR/artifacts/image_embeddings.npy"
    copy_file_if_present "$RUNTIME_ASSETS_CACHE_DIR/image_meta.json" "$APP_DIR/artifacts/image_meta.json"
}

resolve_checkpoint_path() {
    find "$MODEL_CACHE_DIR" -type f \( -name "*.pt" -o -name "*.pth" \) | sort | head -n 1
}

bootstrap_model() {
    if [ -n "${SNEAKER_MODEL_CHECKPOINT:-}" ] && [ -f "${SNEAKER_MODEL_CHECKPOINT}" ]; then
        echo "[bootstrap] Using existing checkpoint: ${SNEAKER_MODEL_CHECKPOINT}"
        return 0
    fi

    echo "[bootstrap] Downloading model from $MODEL_REPO_URL"
    clone_repo_if_missing "$MODEL_REPO_URL" "$MODEL_CACHE_DIR"

    checkpoint_path="$(resolve_checkpoint_path)"
    if [ -z "$checkpoint_path" ]; then
        echo "[bootstrap] No checkpoint file found in $MODEL_CACHE_DIR" >&2
        exit 1
    fi

    export SNEAKER_MODEL_CHECKPOINT="$checkpoint_path"
    echo "[bootstrap] Resolved checkpoint: $SNEAKER_MODEL_CHECKPOINT"
}

bootstrap_runtime_assets
bootstrap_model

exec uvicorn backend.app.api:app --host "$BACKEND_HOST" --port "$BACKEND_PORT"
