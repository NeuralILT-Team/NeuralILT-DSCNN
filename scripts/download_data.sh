#!/bin/bash
# Download LithoBench dataset for NeuralILT-DSCNN experiments.
#
# The LithoBench dataset has two parts:
#   1. MetalSet (main data): hosted on Google Drive as lithodata.tar.gz
#      Contains target/ (layouts) and litho/ (masks) — 16,472 tile pairs
#   2. StdMetal/StdContact (generalization): in the LithoBench git repo
#      Contains .glp layout files (271 + 328 files)
#
# Run on the login node (has internet):
#   bash scripts/download_data.sh          # download MetalSet (main)
#   bash scripts/download_data.sh all      # download everything
#
# Source: https://github.com/shelljane/lithobench

set -euo pipefail

# go to project root
if [ -f "scripts/download_data.sh" ]; then
    cd "$(pwd)"
elif [ -f "download_data.sh" ]; then
    cd ..
fi

DATA_DIR="data/raw"
mkdir -p "$DATA_DIR"

# Google Drive file ID for lithodata.tar.gz
GDRIVE_FILE_ID="1MzYiRRxi8Eu2L6WHCfZ1DtRnjVNOl4vu"

echo "============================================"
echo "LithoBench Dataset Download"
echo "============================================"

# ─────────────────────────────────────────────────────────────────────
# Download from Google Drive (handles the confirmation page)
# ─────────────────────────────────────────────────────────────────────
download_gdrive() {
    local file_id="$1"
    local output="$2"

    echo "  Downloading from Google Drive..."
    echo "  File ID: ${file_id}"
    echo "  Output:  ${output}"
    echo ""

    # Method 1: gdown (Python tool, handles large files well)
    if command -v gdown &>/dev/null; then
        echo "  Using gdown..."
        gdown "https://drive.google.com/uc?id=${file_id}" -O "$output"
        return $?
    fi

    # Method 2: pip install gdown and use it
    if command -v pip &>/dev/null; then
        echo "  Installing gdown..."
        pip install --quiet gdown
        gdown "https://drive.google.com/uc?id=${file_id}" -O "$output"
        return $?
    fi

    # Method 3: curl with cookie handling (for large files)
    echo "  Using curl (may need confirmation for large files)..."
    local confirm_code
    confirm_code=$(curl -sc /tmp/gdrive_cookie \
        "https://drive.google.com/uc?export=download&id=${file_id}" \
        | grep -o 'confirm=[^&]*' | cut -d= -f2)

    if [ -n "$confirm_code" ]; then
        curl -Lb /tmp/gdrive_cookie \
            "https://drive.google.com/uc?export=download&confirm=${confirm_code}&id=${file_id}" \
            -o "$output"
    else
        curl -L \
            "https://drive.google.com/uc?export=download&id=${file_id}" \
            -o "$output"
    fi
    rm -f /tmp/gdrive_cookie
}

# ─────────────────────────────────────────────────────────────────────
# Download MetalSet from Google Drive
# ─────────────────────────────────────────────────────────────────────
download_metalset() {
    # Skip if data already exists
    if [ -d "${DATA_DIR}/MetalSet/target" ] && [ -d "${DATA_DIR}/MetalSet/litho" ]; then
        n=$(ls "${DATA_DIR}/MetalSet/target/" 2>/dev/null | wc -l)
        echo "[OK] MetalSet already exists at ${DATA_DIR}/MetalSet/ (${n} tiles) — skipping download"
        return 0
    fi

    local tarball="${DATA_DIR}/lithodata.tar.gz"

    # Skip download if tarball already exists (just extract)
    if [ -f "$tarball" ]; then
        echo "[OK] Tarball already exists at ${tarball} — skipping download, extracting..."
    else
        echo ""
        echo ">>> Downloading LithoBench data from Google Drive..."
        echo "    Source: https://drive.google.com/file/d/${GDRIVE_FILE_ID}"
        echo "    Size: ~15GB (contains all subsets: MetalSet, ViaSet, etc.)"
        echo "    We only use MetalSet (16,472 tiles) for this project."
        echo ""

        download_gdrive "$GDRIVE_FILE_ID" "$tarball"
    fi

    if [ ! -f "$tarball" ] || [ ! -s "$tarball" ]; then
        echo ""
        echo "ERROR: Download failed or file is empty."
        echo ""
        echo "Please download manually:"
        echo "  1. Open: https://drive.google.com/file/d/${GDRIVE_FILE_ID}/view"
        echo "  2. Click 'Download'"
        echo "  3. Upload to HPC: scp lithodata.tar.gz <user>@hpc:~/NeuralILT-DSCNN/${DATA_DIR}/"
        echo "  4. Run: bash scripts/download_data.sh extract"
        return 1
    fi

    # Verify the file is actually a tarball (not an HTML error page)
    local file_type
    file_type=$(file "$tarball" 2>/dev/null || echo "unknown")
    echo "File type: $file_type"

    if echo "$file_type" | grep -qi "html\|text\|ASCII"; then
        echo ""
        echo "ERROR: Downloaded file is HTML, not a tarball."
        echo "This usually means Google Drive served a confirmation page."
        echo ""
        echo "Fix: install gdown and retry:"
        echo "  pip install gdown"
        echo "  rm ${tarball}"
        echo "  bash scripts/download_data.sh MetalSet"
        echo ""
        echo "Or download manually:"
        echo "  1. Open: https://drive.google.com/file/d/${GDRIVE_FILE_ID}/view"
        echo "  2. Click 'Download'"
        echo "  3. Upload: scp lithodata.tar.gz <user>@hpc:~/NeuralILT-DSCNN/${DATA_DIR}/"
        echo "  4. Run: bash scripts/download_data.sh extract"
        rm -f "$tarball"
        return 1
    fi

    local file_size
    file_size=$(stat -f%z "$tarball" 2>/dev/null || stat -c%s "$tarball" 2>/dev/null || echo "0")
    echo "File size: $(echo "$file_size / 1048576" | bc 2>/dev/null || echo "$file_size") MB"

    echo ""
    echo "Extracting ONLY MetalSet from lithodata.tar.gz..."
    echo "  (skipping ViaSet and other subsets to save disk space)"
    echo ""

    # First, find the MetalSet path inside the tarball
    echo "  Scanning tarball for MetalSet path..."
    local metalset_path
    metalset_path=$(tar tzf "$tarball" | grep -m1 "MetalSet" | cut -d'/' -f1-2)

    if [ -z "$metalset_path" ]; then
        # try broader search
        metalset_path=$(tar tzf "$tarball" | grep -i "metal" | head -1 | cut -d'/' -f1-2)
    fi

    if [ -n "$metalset_path" ]; then
        echo "  Found MetalSet at: ${metalset_path}"
        echo "  Extracting (this takes a few minutes)..."

        # Extract only MetalSet — much faster and smaller than full extraction
        if command -v pv &>/dev/null; then
            pv "$tarball" | tar xzf - -C "$DATA_DIR/" --wildcards "*MetalSet*"
        else
            tar xzvf "$tarball" -C "$DATA_DIR/" --wildcards "*MetalSet*" 2>&1 | \
                awk 'NR % 500 == 0 {print "  " NR " files extracted..."}'
        fi
    else
        echo "  WARNING: Could not find MetalSet path in tarball."
        echo "  Listing tarball contents:"
        tar tzf "$tarball" | head -30
        echo ""
        echo "  Extracting everything (this will take longer)..."
        tar xzf "$tarball" -C "$DATA_DIR/"
    fi

    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Extraction failed (possibly out of disk space)."
        echo "Check: df -h ."
        echo "Try: rm -rf data/raw/ViaSet && bash scripts/download_data.sh extract"
        return 1
    fi

    # Find MetalSet wherever it landed
    if [ ! -d "${DATA_DIR}/MetalSet" ]; then
        FOUND=$(find "${DATA_DIR}" -maxdepth 4 -type d -name "MetalSet" | head -1)
        if [ -n "$FOUND" ] && [ "$FOUND" != "${DATA_DIR}/MetalSet" ]; then
            echo "  Moving MetalSet from $FOUND to ${DATA_DIR}/MetalSet"
            mv "$FOUND" "${DATA_DIR}/MetalSet"
        fi
    fi

    # Verify
    if [ -d "${DATA_DIR}/MetalSet/target" ]; then
        n=$(ls "${DATA_DIR}/MetalSet/target/" 2>/dev/null | wc -l)
        echo ""
        echo "[OK] MetalSet: ${n} layout tiles extracted"
    elif [ -d "${DATA_DIR}/MetalSet" ]; then
        echo ""
        echo "[OK] MetalSet directory found. Contents:"
        ls "${DATA_DIR}/MetalSet/"
    else
        echo ""
        echo "WARNING: MetalSet not found after extraction."
        echo "Contents of ${DATA_DIR}/:"
        ls -la "${DATA_DIR}/"
        find "${DATA_DIR}/" -maxdepth 3 -type d | head -20
    fi

    # Clean up: delete tarball and any unwanted subsets
    echo ""
    echo "Cleaning up to save disk space..."
    rm -rf "${DATA_DIR}/ViaSet" "${DATA_DIR}/StdContact" "${DATA_DIR}/ICCAD2013" 2>/dev/null
    rm -f "$tarball"
    echo "Cleaned up. Remaining:"
    du -sh "${DATA_DIR}/"* 2>/dev/null
}

# ─────────────────────────────────────────────────────────────────────
# Download StdMetal/StdContact from git repo (benchmark/ directory)
# ─────────────────────────────────────────────────────────────────────
download_benchmarks() {
    echo ""
    echo ">>> Downloading StdMetal and StdContact from LithoBench repo..."
    echo "    NOTE: The LithoBench git repo is large (~15GB with models/kernels)."
    echo "    We use sparse checkout to only download benchmark/ (~1MB)."
    echo ""

    local TEMP_DIR
    TEMP_DIR=$(mktemp -d)

    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/shelljane/lithobench.git "$TEMP_DIR/lithobench" 2>/dev/null

    cd "$TEMP_DIR/lithobench"
    git sparse-checkout set benchmark/StdMetal benchmark/StdContact 2>/dev/null || {
        # fallback: just use what we got
        echo "  Sparse checkout not supported, using full clone..."
        cd "$TEMP_DIR"
        rm -rf lithobench
        git clone --depth 1 https://github.com/shelljane/lithobench.git "$TEMP_DIR/lithobench"
    }
    cd - >/dev/null

    # Copy StdMetal
    if [ -d "$TEMP_DIR/lithobench/benchmark/StdMetal" ]; then
        mkdir -p "${DATA_DIR}/StdMetal"
        cp -r "$TEMP_DIR/lithobench/benchmark/StdMetal" "${DATA_DIR}/"
        n=$(ls "${DATA_DIR}/StdMetal/" 2>/dev/null | wc -l)
        echo "[OK] StdMetal: ${n} files"
    else
        echo "[SKIP] StdMetal not found in repo"
    fi

    # Copy StdContact
    if [ -d "$TEMP_DIR/lithobench/benchmark/StdContact" ]; then
        mkdir -p "${DATA_DIR}/StdContact"
        cp -r "$TEMP_DIR/lithobench/benchmark/StdContact" "${DATA_DIR}/"
        n=$(ls "${DATA_DIR}/StdContact/" 2>/dev/null | wc -l)
        echo "[OK] StdContact: ${n} files"
    else
        echo "[SKIP] StdContact not found in repo"
    fi

    rm -rf "$TEMP_DIR"
}

# ─────────────────────────────────────────────────────────────────────
# Extract a manually uploaded tarball
# ─────────────────────────────────────────────────────────────────────
extract_tarball() {
    local tarball="${DATA_DIR}/lithodata.tar.gz"
    if [ ! -f "$tarball" ]; then
        echo "No tarball found at ${tarball}"
        echo "Upload it first: scp lithodata.tar.gz <user>@hpc:~/NeuralILT-DSCNN/${DATA_DIR}/"
        return 1
    fi
    echo "Extracting ONLY MetalSet from ${tarball}..."
    echo "  (skipping ViaSet and other subsets to save disk space)"
    tar xzvf "$tarball" -C "$DATA_DIR/" --wildcards '*MetalSet*' 2>&1 | \
        awk 'NR % 500 == 0 {print "  " NR " files..."}'

    # Find and move MetalSet if needed
    if [ ! -d "${DATA_DIR}/MetalSet" ]; then
        FOUND=$(find "${DATA_DIR}" -maxdepth 4 -type d -name "MetalSet" | head -1)
        if [ -n "$FOUND" ]; then
            mv "$FOUND" "${DATA_DIR}/MetalSet"
        fi
    fi

    echo ""
    echo "Done. Contents:"
    du -sh "${DATA_DIR}/"* 2>/dev/null
}

# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
MODE="${1:-MetalSet}"

case "$MODE" in
    MetalSet)
        download_metalset
        ;;
    benchmarks|StdMetal|StdContact)
        download_benchmarks
        ;;
    extract)
        extract_tarball
        ;;
    all)
        download_metalset
        download_benchmarks
        ;;
    *)
        echo "Usage: bash scripts/download_data.sh [MetalSet|benchmarks|extract|all]"
        echo ""
        echo "  MetalSet    — download main dataset from Google Drive (~2GB)"
        echo "  benchmarks  — download StdMetal/StdContact from GitHub"
        echo "  extract     — extract a manually uploaded lithodata.tar.gz"
        echo "  all         — download everything"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "Dataset location: ${DATA_DIR}/"
ls -la "${DATA_DIR}/" 2>/dev/null
echo ""
echo "Next: python -m src.data.preprocess --all"
echo "============================================"
