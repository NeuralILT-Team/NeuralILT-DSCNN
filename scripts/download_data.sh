#!/bin/bash
# Download LithoBench dataset from the official GitHub repository.
#
# Run this on the login node (has internet access):
#   bash scripts/download_data.sh
#   bash scripts/download_data.sh MetalSet       # just MetalSet
#   bash scripts/download_data.sh all             # all subsets
#
# The LithoBench dataset is hosted at:
#   https://github.com/shelljane/lithobench
#
# Dataset structure after download:
#   data/raw/MetalSet/target/   (16,472 layout tiles)
#   data/raw/MetalSet/litho/    (16,472 mask tiles)
#   data/raw/StdMetal/target/   (271 tiles)
#   data/raw/StdMetal/litho/    (271 tiles)
#   data/raw/StdContact/target/ (328 tiles)
#   data/raw/StdContact/litho/  (328 tiles)

set -euo pipefail

# go to project root
if [ -f "scripts/download_data.sh" ]; then
    cd "$(pwd)"
elif [ -f "download_data.sh" ]; then
    cd ..
fi

DATASET="${1:-MetalSet}"
DATA_DIR="data/raw"
REPO_URL="https://github.com/shelljane/lithobench"

mkdir -p "$DATA_DIR"

echo "============================================"
echo "LithoBench Dataset Download"
echo "============================================"

download_dataset() {
    local name="$1"
    local target_dir="${DATA_DIR}/${name}"

    if [ -d "${target_dir}/target" ] && [ -d "${target_dir}/litho" ]; then
        n_target=$(ls "${target_dir}/target/" 2>/dev/null | wc -l)
        echo "[OK] ${name} already exists (${n_target} tiles)"
        return 0
    fi

    echo ""
    echo "Downloading ${name}..."

    # Method 1: Try cloning just the data we need using sparse checkout
    # (avoids downloading the entire repo with code)
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT

    echo "  Cloning LithoBench repo (sparse checkout)..."
    git clone --depth 1 --filter=blob:none --sparse \
        "${REPO_URL}.git" "$TEMP_DIR/lithobench" 2>/dev/null || {

        # Method 2: If sparse checkout fails, try full clone
        echo "  Sparse checkout failed, trying full clone..."
        rm -rf "$TEMP_DIR/lithobench"
        git clone --depth 1 "${REPO_URL}.git" "$TEMP_DIR/lithobench" 2>/dev/null || {

            # Method 3: Download as zip
            echo "  Git clone failed, trying zip download..."
            curl -L -o "$TEMP_DIR/lithobench.zip" \
                "${REPO_URL}/archive/refs/heads/main.zip" 2>/dev/null || \
            wget -q -O "$TEMP_DIR/lithobench.zip" \
                "${REPO_URL}/archive/refs/heads/main.zip" 2>/dev/null || {
                echo "  ERROR: Could not download. Check internet connection."
                echo "  Manual download: ${REPO_URL}"
                return 1
            }
            cd "$TEMP_DIR"
            unzip -q lithobench.zip
            mv lithobench-main lithobench 2>/dev/null || true
            cd -
        }
    }

    # Find the dataset in the cloned/downloaded repo
    # LithoBench organizes data as: lithobench/data/<DatasetName>/
    # or sometimes: lithobench/<DatasetName>/
    FOUND=""
    for search_path in \
        "$TEMP_DIR/lithobench/data/${name}" \
        "$TEMP_DIR/lithobench/${name}" \
        "$TEMP_DIR/lithobench/dataset/${name}" \
        "$TEMP_DIR/lithobench/datasets/${name}"; do
        if [ -d "$search_path" ]; then
            FOUND="$search_path"
            break
        fi
    done

    if [ -z "$FOUND" ]; then
        echo "  WARNING: Could not find ${name} in downloaded repo."
        echo "  Searched in: $TEMP_DIR/lithobench/"
        echo ""
        echo "  The LithoBench dataset may need to be downloaded separately."
        echo "  Check: ${REPO_URL} for download instructions."
        echo ""
        echo "  You can also manually place the data at:"
        echo "    ${target_dir}/target/  (layout tiles)"
        echo "    ${target_dir}/litho/   (mask tiles)"

        # List what we found to help debug
        echo ""
        echo "  Contents of downloaded repo:"
        ls -la "$TEMP_DIR/lithobench/" 2>/dev/null || echo "  (empty)"
        find "$TEMP_DIR/lithobench/" -maxdepth 3 -type d 2>/dev/null | head -20
        return 1
    fi

    echo "  Found ${name} at: ${FOUND}"
    mkdir -p "$target_dir"
    cp -r "$FOUND"/* "$target_dir/"

    # verify
    if [ -d "${target_dir}/target" ]; then
        n=$(ls "${target_dir}/target/" 2>/dev/null | wc -l)
        echo "  [OK] ${name}: ${n} tiles downloaded"
    else
        echo "  WARNING: ${name} downloaded but target/ directory not found"
        echo "  Contents: $(ls ${target_dir}/ 2>/dev/null)"
    fi

    rm -rf "$TEMP_DIR"
    trap - EXIT
}

case "$DATASET" in
    MetalSet)
        download_dataset "MetalSet"
        ;;
    StdMetal)
        download_dataset "StdMetal"
        ;;
    StdContact)
        download_dataset "StdContact"
        ;;
    all)
        download_dataset "MetalSet"
        download_dataset "StdMetal"
        download_dataset "StdContact"
        ;;
    *)
        echo "Usage: bash scripts/download_data.sh [MetalSet|StdMetal|StdContact|all]"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "Download complete. Dataset location:"
echo "  ${DATA_DIR}/"
ls -la "${DATA_DIR}/" 2>/dev/null
echo ""
echo "Next step: preprocess the data"
echo "  python -m src.data.preprocess --all"
echo "============================================"
