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
    echo "Extracting MetalSet/target and MetalSet/litho from tarball..."
    echo "  (skipping ViaSet, gds, glp, resist, printed, levelsetILT)"
    echo ""

    # Find the MetalSet path prefix inside the tarball
    echo "  Scanning tarball structure..."
    local metalset_prefix
    metalset_prefix=$(tar tzf "$tarball" | grep -m1 "MetalSet/target/" | sed 's|MetalSet/target/.*|MetalSet|')

    if [ -n "$metalset_prefix" ]; then
        echo "  Found: ${metalset_prefix}/target/ and ${metalset_prefix}/litho/"
        echo "  Extracting only target + litho (this takes a few minutes)..."

        # Extract ONLY target/ and litho/ from MetalSet
        tar xzvf "$tarball" -C "$DATA_DIR/" \
            "${metalset_prefix}/target" \
            "${metalset_prefix}/litho" \
            2>&1 | awk 'NR % 500 == 0 {print "  " NR " files extracted..."}'
    else
        echo "  Could not find exact path. Trying wildcard extraction..."
        tar xzvf "$tarball" -C "$DATA_DIR/" --wildcards \
            '*MetalSet/target*' '*MetalSet/litho*' \
            2>&1 | awk 'NR % 500 == 0 {print "  " NR " files extracted..."}'
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

    # Clean up: delete tarball and unwanted subsets (keep StdMetal/StdContact for Exp 4)
    echo ""
    echo "Cleaning up to save disk space..."
    rm -rf "${DATA_DIR}/ViaSet" "${DATA_DIR}/ICCAD2013" 2>/dev/null
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
    echo ""

    # Check if already downloaded AND converted to PNG
    if [ -d "${DATA_DIR}/StdMetal/target" ] && [ -d "${DATA_DIR}/StdMetal/litho" ]; then
        png_count=$(ls "${DATA_DIR}/StdMetal/target/"*.png 2>/dev/null | wc -l)
        if [ "$png_count" -gt 0 ]; then
            echo "[OK] StdMetal already has PNG files (${png_count} tiles) — skipping"
            return 0
        fi
        echo "[INFO] StdMetal exists but no PNG files — will convert .glp to .png"
    fi

    local TEMP_DIR
    TEMP_DIR=$(mktemp -d)

    echo "  Cloning LithoBench repo (sparse checkout)..."

    # Try sparse checkout first (downloads only benchmark/ directory)
    if git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/shelljane/lithobench.git "$TEMP_DIR/lithobench" 2>&1; then

        cd "$TEMP_DIR/lithobench"
        git sparse-checkout set benchmark/ 2>&1 || true
        cd - >/dev/null
    else
        echo "  Sparse checkout failed, trying full shallow clone..."
        rm -rf "$TEMP_DIR/lithobench"
        git clone --depth 1 https://github.com/shelljane/lithobench.git "$TEMP_DIR/lithobench" 2>&1 || {
            echo ""
            echo "ERROR: git clone failed. Check internet access."
            echo "You're on: $(hostname)"
            echo "Try running this on the login node (has internet)."
            rm -rf "$TEMP_DIR"
            return 1
        }
    fi

    echo "  Repo contents:"
    ls "$TEMP_DIR/lithobench/benchmark/" 2>/dev/null || ls "$TEMP_DIR/lithobench/" 2>/dev/null

    # Copy StdMetal
    local stdmetal_src
    stdmetal_src=$(find "$TEMP_DIR/lithobench" -type d -name "StdMetal" | head -1)
    if [ -n "$stdmetal_src" ]; then
        echo "  Found StdMetal at: $stdmetal_src"
        mkdir -p "${DATA_DIR}/StdMetal"
        cp -r "$stdmetal_src"/* "${DATA_DIR}/StdMetal/" 2>/dev/null || \
            cp -r "$stdmetal_src" "${DATA_DIR}/"

        # StdMetal needs target/ and litho/ subdirs for our pipeline
        # The repo might have a different structure — check and adapt
        if [ ! -d "${DATA_DIR}/StdMetal/target" ]; then
            echo "  StdMetal doesn't have target/ subdir. Contents:"
            ls "${DATA_DIR}/StdMetal/" | head -10
            echo "  Creating target/ and litho/ from available files..."
            mkdir -p "${DATA_DIR}/StdMetal/target" "${DATA_DIR}/StdMetal/litho"
            # Move image files to target/ (they'll be used as both input and target)
            find "${DATA_DIR}/StdMetal" -maxdepth 1 -type f \( -name "*.png" -o -name "*.bmp" -o -name "*.jpg" -o -name "*.gds" -o -name "*.glp" \) \
                -exec mv {} "${DATA_DIR}/StdMetal/target/" \;
        fi

        n=$(find "${DATA_DIR}/StdMetal" -type f | wc -l)
        echo "[OK] StdMetal: ${n} files"
    else
        echo "[SKIP] StdMetal not found in repo"
        echo "  Searched in: $TEMP_DIR/lithobench/"
        find "$TEMP_DIR/lithobench" -maxdepth 3 -type d | head -20
    fi

    # Copy StdContact
    local stdcontact_src
    stdcontact_src=$(find "$TEMP_DIR/lithobench" -type d -name "StdContact" | head -1)
    if [ -n "$stdcontact_src" ]; then
        echo "  Found StdContact at: $stdcontact_src"
        mkdir -p "${DATA_DIR}/StdContact"
        cp -r "$stdcontact_src"/* "${DATA_DIR}/StdContact/" 2>/dev/null || \
            cp -r "$stdcontact_src" "${DATA_DIR}/"

        if [ ! -d "${DATA_DIR}/StdContact/target" ]; then
            mkdir -p "${DATA_DIR}/StdContact/target" "${DATA_DIR}/StdContact/litho"
            find "${DATA_DIR}/StdContact" -maxdepth 1 -type f \
                -exec mv {} "${DATA_DIR}/StdContact/target/" \;
        fi

        n=$(find "${DATA_DIR}/StdContact" -type f | wc -l)
        echo "[OK] StdContact: ${n} files"
    else
        echo "[SKIP] StdContact not found in repo"
    fi

    rm -rf "$TEMP_DIR"

    # Convert .glp files to PNG images
    echo ""
    echo ">>> Converting .glp files to PNG..."
    PYTHON_CMD="python3"
    if [ -f "venv/bin/python" ]; then
        PYTHON_CMD="venv/bin/python"
    fi

    for ds in StdMetal StdContact; do
        if [ -d "${DATA_DIR}/${ds}/target" ]; then
            glp_count=$(ls "${DATA_DIR}/${ds}/target/"*.glp 2>/dev/null | wc -l)
            if [ "$glp_count" -gt 0 ]; then
                echo "  Converting ${ds} (${glp_count} .glp files)..."
                $PYTHON_CMD scripts/convert_glp.py "${DATA_DIR}/${ds}" || {
                    echo "  WARNING: GLP conversion failed for ${ds}"
                    echo "  Try manually: $PYTHON_CMD scripts/convert_glp.py ${DATA_DIR}/${ds}"
                }
            fi
        fi
    done
}

# ─────────────────────────────────────────────────────────────────────
# Download tarball and extract StdMetal + StdContact (rendered PNGs)
# The git repo only has .glp files; rendered images are in the tarball.
# ─────────────────────────────────────────────────────────────────────
download_stdmetal_from_tarball() {
    # Skip if already have PNG images
    if [ -d "${DATA_DIR}/StdMetal/target" ]; then
        sample=$(ls "${DATA_DIR}/StdMetal/target/" | head -1)
        if echo "$sample" | grep -qi "\.png$\|\.bmp$"; then
            n=$(ls "${DATA_DIR}/StdMetal/target/" | wc -l)
            echo "[OK] StdMetal already has image files (${n} tiles) — skipping"
            return 0
        else
            echo "[WARN] StdMetal has .glp files, not images. Need to extract from tarball."
            rm -rf "${DATA_DIR}/StdMetal" "${DATA_DIR}/StdContact"
        fi
    fi

    local tarball="${DATA_DIR}/lithodata.tar.gz"

    # Download tarball if not present
    if [ ! -f "$tarball" ]; then
        echo ""
        echo ">>> Downloading tarball to extract StdMetal/StdContact..."
        echo "    (15GB download — only need StdMetal target+litho from it)"
        echo ""
        download_gdrive "$GDRIVE_FILE_ID" "$tarball"
    fi

    if [ ! -f "$tarball" ] || [ ! -s "$tarball" ]; then
        echo "ERROR: Tarball download failed."
        return 1
    fi

    echo ""
    echo "Extracting StdMetal and StdContact (target + litho only)..."

    # Extract StdMetal target + litho
    tar xzf "$tarball" -C "$DATA_DIR/" --wildcards \
        '*/StdMetal/target/*' '*/StdMetal/litho/*' \
        '*/StdContact/target/*' '*/StdContact/litho/*' \
        2>&1 | awk 'NR % 100 == 0 {print "  " NR " files..."}'

    # Move to correct location if nested
    for ds in StdMetal StdContact; do
        if [ ! -d "${DATA_DIR}/${ds}/target" ]; then
            FOUND=$(find "${DATA_DIR}" -maxdepth 4 -type d -name "${ds}" | head -1)
            if [ -n "$FOUND" ] && [ "$FOUND" != "${DATA_DIR}/${ds}" ]; then
                mkdir -p "${DATA_DIR}/${ds}"
                mv "$FOUND"/* "${DATA_DIR}/${ds}/" 2>/dev/null
                rmdir "$FOUND" 2>/dev/null
            fi
        fi
    done

    # Verify
    for ds in StdMetal StdContact; do
        if [ -d "${DATA_DIR}/${ds}/target" ]; then
            n=$(ls "${DATA_DIR}/${ds}/target/" 2>/dev/null | wc -l)
            echo "[OK] ${ds}: ${n} target tiles"
        else
            echo "[SKIP] ${ds}: not found in tarball"
        fi
    done

    # Clean up tarball to save space
    echo "Removing tarball to save disk space..."
    rm -f "$tarball"
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
    echo "Extracting MetalSet/target and MetalSet/litho from ${tarball}..."
    echo "  (skipping gds, glp, resist, printed, levelsetILT, ViaSet)"

    # Find the path prefix
    local metalset_prefix
    metalset_prefix=$(tar tzf "$tarball" | grep -m1 "MetalSet/target/" | sed 's|MetalSet/target/.*|MetalSet|')

    if [ -n "$metalset_prefix" ]; then
        tar xzvf "$tarball" -C "$DATA_DIR/" \
            "${metalset_prefix}/target" \
            "${metalset_prefix}/litho" \
            2>&1 | awk 'NR % 500 == 0 {print "  " NR " files..."}'
    else
        tar xzvf "$tarball" -C "$DATA_DIR/" --wildcards \
            '*MetalSet/target*' '*MetalSet/litho*' \
            2>&1 | awk 'NR % 500 == 0 {print "  " NR " files..."}'
    fi

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
        download_stdmetal_from_tarball
        ;;
    extract)
        extract_tarball
        ;;
    all)
        download_metalset
        download_stdmetal_from_tarball
        ;;
    *)
        echo "Usage: bash scripts/download_data.sh [MetalSet|benchmarks|extract|all]"
        echo ""
        echo "  MetalSet    — download main dataset from Google Drive"
        echo "  benchmarks  — download StdMetal/StdContact (re-downloads tarball if needed)"
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
