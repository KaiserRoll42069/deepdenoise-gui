#!/usr/bin/env bash
# ==============================================================================
# DeepDenoise Linux Platform Setup & Standalone Installer (Release v1)
#
# Original Deep-Rep Framework Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International)
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
# For academic / non-commercial research use only.
# ==============================================================================

set -e

RELEASE_TAG="v1"
REPO="KaiserRoll42069/deepdenoise-gui"
SOURCE_ZIP_URL="https://github.com/${REPO}/releases/download/${RELEASE_TAG}/deepdenoise-source.zip"

echo "=========================================================="
echo "  DeepDenoise GUI - Linux Platform Setup & Installer     "
echo "  Version: ${RELEASE_TAG} (Standalone Self-Configuring)  "
echo "=========================================================="

# 1. Determine Installation Directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR=""

if [ -f "${SCRIPT_DIR}/deeprep_gui.py" ]; then
    INSTALL_DIR="${SCRIPT_DIR}"
elif [ -f "${SCRIPT_DIR}/../../deeprep_gui.py" ]; then
    INSTALL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
    # Standalone script mode: install into $HOME/deepdenoise
    INSTALL_DIR="${HOME}/deepdenoise"
    mkdir -p "${INSTALL_DIR}"
    echo "[1/6] Standalone mode: Downloading source release from GitHub (${RELEASE_TAG})..."
    
    TMP_ZIP="/tmp/deepdenoise-source-${RELEASE_TAG}.zip"
    if command -v curl >/dev/null 2>&1; then
        curl -fL -o "${TMP_ZIP}" "${SOURCE_ZIP_URL}"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${TMP_ZIP}" "${SOURCE_ZIP_URL}"
    else
        echo "ERROR: Neither curl nor wget was found. Please install curl or wget."
        exit 1
    fi

    echo "  Extracting source code into ${INSTALL_DIR}..."
    if command -v unzip >/dev/null 2>&1; then
        unzip -q -o "${TMP_ZIP}" -d "${INSTALL_DIR}"
    else
        python3 -c "import zipfile; zipfile.ZipFile('${TMP_ZIP}').extractall('${INSTALL_DIR}')"
    fi
    rm -f "${TMP_ZIP}"
fi

cd "${INSTALL_DIR}"
echo "[1/6] Target directory configured: ${INSTALL_DIR}"

# 2. Python Environment Detection
echo -e "\n[2/6] Detecting Python 3 Installation..."
PYTHON_BIN=""
for candidate in python3 /usr/bin/python3 /usr/local/bin/python3 python; do
    if command -v $candidate >/dev/null 2>&1; then
        VER=$($candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        MAJOR=$(echo $VER | cut -d. -f1)
        MINOR=$(echo $VER | cut -d. -f2)
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 8 ]; then
            PYTHON_BIN=$(command -v $candidate)
            echo "  Found Python: ${PYTHON_BIN} (v${VER})"
            break
        fi
    fi
done

if [ -z "${PYTHON_BIN}" ]; then
    echo "ERROR: Python 3.8+ is required. Please install python3 (e.g., sudo apt install python3 python3-pip)."
    exit 1
fi

# Ensure pip is available
if ! ${PYTHON_BIN} -m pip --version >/dev/null 2>&1; then
    echo "  Installing pip..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y python3-pip python3-venv
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y python3-pip
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -S --noconfirm python-pip
    fi
fi

# 3. Verify PyTorch & GPU Acceleration
echo -e "\n[3/6] Verifying PyTorch Environment..."
TORCH_INFO=$(${PYTHON_BIN} -c "
try:
    import torch
    print(f'OK|{torch.__version__}|{torch.cuda.is_available()}|{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')
except Exception as e:
    print(f'MISSING|{e}')
" 2>/dev/null || echo "MISSING")

STATUS=$(echo "${TORCH_INFO}" | cut -d'|' -f1)
if [ "${STATUS}" = "OK" ]; then
    T_VER=$(echo "${TORCH_INFO}" | cut -d'|' -f2)
    CUDA_AVAIL=$(echo "${TORCH_INFO}" | cut -d'|' -f3)
    DEV_NAME=$(echo "${TORCH_INFO}" | cut -d'|' -f4)
    echo "  PyTorch is installed: v${T_VER}"
    if [ "${CUDA_AVAIL}" = "True" ]; then
        echo "  CUDA GPU Acceleration Active: ${DEV_NAME}"
    else
        echo "  Running in CPU mode."
    fi
else
    echo "  PyTorch not found. Installing PyTorch with CUDA 12.1..."
    ${PYTHON_BIN} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 || \
    ${PYTHON_BIN} -m pip install torch torchvision
fi

# 4. Verify Core Dependencies
echo -e "\n[4/6] Verifying Required Packages..."
REQ_PKGS=("rawpy" "tifffile" "numpy" "opencv-python" "scipy" "pillow" "exifread" "requests")
for pkg in "${REQ_PKGS[@]}"; do
    if ${PYTHON_BIN} -c "import ${pkg}" >/dev/null 2>&1; then
        echo "  [OK] ${pkg}"
    else
        echo "  Installing ${pkg}..."
        ${PYTHON_BIN} -m pip install "${pkg}"
    fi
done

# Ensure tkinter is available on Linux
if ! ${PYTHON_BIN} -c "import tkinter" >/dev/null 2>&1; then
    echo "  Installing python3-tk GUI toolkit..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y python3-tk
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y python3-tkinter
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -S --noconfirm tk
    fi
fi

# 5. Pretrained Weights & Dynamic Paths
echo -e "\n[5/6] Checking Pretrained Neural Weights..."
mkdir -p "${INSTALL_DIR}/pretrained_networks"
if [ -f "${INSTALL_DIR}/download_weights.py" ]; then
    ${PYTHON_BIN} "${INSTALL_DIR}/download_weights.py" "${INSTALL_DIR}/pretrained_networks"
fi

# Configure dynamic admin/local.py
cat << 'EOF' > "${INSTALL_DIR}/admin/local.py"
import os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = ''
        self.tensorboard_dir = os.path.join(base_path, 'tensorboard')
        self.pretrained_nets_dir = os.path.join(base_path, 'pretrained_networks')
        self.save_data_path = ''
        self.zurichraw2rgb_dir = ''
        self.burstsr_dir = ''
        self.synburstval_dir = ''
        self.openimages_dir = ''
        self.kpn_testset_path = ''
        self.bpn_color_testset_dir = ''
EOF

# Create run_deeprep.sh launcher
cat << EOF > "${INSTALL_DIR}/run_deeprep.sh"
#!/usr/bin/env bash
cd "${INSTALL_DIR}"
exec "${PYTHON_BIN}" "${INSTALL_DIR}/deeprep_gui.py" "\$@"
EOF
chmod +x "${INSTALL_DIR}/run_deeprep.sh"

# 6. Darktable Integration
echo -e "\n[6/6] Checking Darktable Installation..."
DT_LUA_CONTRIB="${HOME}/.config/darktable/lua/contrib"
if [ -d "${HOME}/.config/darktable" ]; then
    mkdir -p "${DT_LUA_CONTRIB}"
    cp -f "${INSTALL_DIR}/deeprep.lua" "${DT_LUA_CONTRIB}/deeprep.lua"
    echo "  Installed Darktable plugin to: ${DT_LUA_CONTRIB}/deeprep.lua"

    LUARC="${HOME}/.config/darktable/luarc"
    if [ ! -f "${LUARC}" ] || ! grep -q "contrib/deeprep" "${LUARC}"; then
        echo 'require "contrib/deeprep"' >> "${LUARC}"
        echo "  Enabled plugin in ${LUARC}"
    fi

    DARKTABLERC="${HOME}/.config/darktable/darktablerc"
    if [ -f "${DARKTABLERC}" ]; then
        sed -i '/^lua\/deeprep\/python_bin=/d' "${DARKTABLERC}"
        sed -i '/^lua\/deeprep\/script_path=/d' "${DARKTABLERC}"
        echo "lua/deeprep/python_bin=${PYTHON_BIN}" >> "${DARKTABLERC}"
        echo "lua/deeprep/script_path=${INSTALL_DIR}/deeprep_gui.py" >> "${DARKTABLERC}"
        echo "  Configured Darktable paths in darktablerc"
    fi
else
    echo "  Darktable config directory not detected. Plugin saved at ${INSTALL_DIR}/deeprep.lua"
fi

# Optional: Create Desktop Entry
DESKTOP_FILE="${HOME}/.local/share/applications/deepdenoise.desktop"
if [ -d "${HOME}/.local/share/applications" ]; then
    cat << EOF > "${DESKTOP_FILE}"
[Desktop Entry]
Type=Application
Name=DeepDenoise GUI
Comment=Deep Learning RAW Burst Restoration & Super-Resolution
Exec=${INSTALL_DIR}/run_deeprep.sh
Icon=camera-photo
Terminal=false
Categories=Graphics;Photography;
EOF
    chmod +x "${DESKTOP_FILE}"
    echo "  Created desktop application launcher: ${DESKTOP_FILE}"
fi

echo -e "\n=========================================================="
echo "  DeepDenoise Installation Complete!                     "
echo "  Launch the application using:                         "
echo "    ${INSTALL_DIR}/run_deeprep.sh                       "
echo "=========================================================="
