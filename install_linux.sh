#!/usr/bin/env bash
# ==============================================================================
# Deep-Rep Linux Environment Installer & Verifier
# Configures Deep-Rep for standalone execution on Linux (Ubuntu, Debian, Fedora, Arch, etc.)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "  Deep-Rep Toolkit - Linux Platform Setup & Installer     "
echo "=========================================================="

# 1. Disk Space Verification
FREE_KB=$(df -k "$SCRIPT_DIR" | awk 'NR==2 {print $4}')
FREE_GB=$((FREE_KB / 1024 / 1024))
echo "[1/6] Checking Disk Space: ${FREE_GB} GB free"

if [ "$FREE_GB" -lt 4 ]; then
    echo "ERROR: Less than 4 GB of free disk space remaining. Aborting."
    exit 1
fi

# 2. Locate Python 3
echo ""
echo "[2/6] Detecting System Python Installation..."
PYTHON_BIN=""

if command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
elif command -v python &> /dev/null; then
    PYTHON_BIN="python"
fi

if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: Python 3 could not be found. Please install Python 3 (python3, python3-pip, python3-tk)."
    exit 1
fi

PY_VER=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Found Python: $($PYTHON_BIN --version) ($PYTHON_BIN)"

# Check tkinter
if ! $PYTHON_BIN -c "import tkinter" &> /dev/null; then
    echo "  Notice: Tkinter not detected for $PYTHON_BIN. If GUI fails, install via: sudo apt install python3-tk"
fi

# 3. Check PyTorch & CUDA
echo ""
echo "[3/6] Verifying PyTorch Environment..."
TORCH_INFO=$($PYTHON_BIN -c "import torch; print(f'{torch.__version__}|{torch.cuda.is_available()}|{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 2> /dev/null || true)

if [ -n "$TORCH_INFO" ]; then
    IFS='|' read -r TVER TCUD TDEV <<< "$TORCH_INFO"
    echo "  PyTorch is installed: Version $TVER"
    if [ "$TCUD" = "True" ]; then
        echo "  CUDA is available on GPU: $TDEV"
    else
        echo "  CUDA is NOT available (CPU mode only)."
    fi
else
    echo "  PyTorch is not currently installed."
    echo "  Installing PyTorch..."
    if command -v nvidia-smi &> /dev/null; then
        echo "  Detected NVIDIA GPU. Installing PyTorch with CUDA support..."
        $PYTHON_BIN -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    else
        echo "  Installing CPU PyTorch..."
        $PYTHON_BIN -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
fi

# 4. Verify Core Dependencies
echo ""
echo "[4/6] Verifying Required Python Packages..."
PKGS=("rawpy" "tifffile" "numpy" "opencv-python" "scipy" "pillow" "exifread" "requests")

for pkg in "${PKGS[@]}"; do
    if $PYTHON_BIN -c "import $pkg" &> /dev/null; then
        echo "  [OK] $pkg is installed."
    else
        echo "  Installing $pkg..."
        $PYTHON_BIN -m pip install "$pkg"
    fi
done

# 5. Download Pretrained Weights & Configure Portable Paths
echo ""
echo "[5/6] Checking Pretrained Networks & Dynamic Paths..."
if [ -f "$SCRIPT_DIR/download_weights.py" ]; then
    $PYTHON_BIN "$SCRIPT_DIR/download_weights.py" "$SCRIPT_DIR/pretrained_networks"
fi

if [ -d "$SCRIPT_DIR/admin" ]; then
    cat << 'EOF' > "$SCRIPT_DIR/admin/local.py"
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
    echo "  Configured portable settings in $SCRIPT_DIR/admin/local.py"
fi

# 6. Darktable Integration
echo ""
echo "[6/6] Checking Darktable Installation..."
DT_LUA_CONTRIB="$HOME/.config/darktable/lua/contrib"
if [ -d "$HOME/.config/darktable" ]; then
    mkdir -p "$DT_LUA_CONTRIB"
    cp "$SCRIPT_DIR/deeprep.lua" "$DT_LUA_CONTRIB/deeprep.lua"
    echo "  Installed Darktable plugin to: $DT_LUA_CONTRIB/deeprep.lua"

    LUARC="$HOME/.config/darktable/luarc"
    if [ -f "$LUARC" ]; then
        if ! grep -q "contrib/deeprep" "$LUARC"; then
            echo 'require "contrib/deeprep"' >> "$LUARC"
            echo "  Enabled deeprep plugin in $LUARC"
        fi
    fi
fi

# Create Launcher Script
cat << EOF > "$SCRIPT_DIR/run_deeprep.sh"
#!/usr/bin/env bash
cd "$SCRIPT_DIR"
exec $PYTHON_BIN "$SCRIPT_DIR/deeprep_gui.py" "\$@"
EOF
chmod +x "$SCRIPT_DIR/run_deeprep.sh"

echo ""
echo "=========================================================="
echo "  Deep-Rep Linux Setup Complete!                         "
echo "  To launch the GUI, run: ./run_deeprep.sh                "
echo "=========================================================="
