<#
.SYNOPSIS
  Deep-Rep Windows Environment Installer & Verifier
.DESCRIPTION
  Configures Deep-Rep for standalone execution on Windows.
  Detects system Python, checks/installs dependencies, downloads pretrained model weights,
  configures portable settings, and installs Darktable integration.
#>

param(
    [switch]$InstallTorch = $false,
    [switch]$CpuOnly = $false,
    [switch]$NonInteractive = $false
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Deep-Rep Toolkit - Windows Platform Setup & Installer   " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Disk Space Verification
$drive = Get-PSDrive C
$freeGB = [math]::Round($drive.Free / 1GB, 2)
Write-Host "[1/6] Checking Disk Space: $freeGB GB free on C: drive" -ForegroundColor Yellow

if ($freeGB -lt 4.0) {
    Write-Host "ERROR: Less than 4 GB of free space remaining on C:. Aborting to preserve disk integrity." -ForegroundColor Red
    exit 1
}

# 2. Locate System Python
Write-Host "`n[2/6] Detecting System Python Installation..." -ForegroundColor Yellow
$pythonExe = $null

# Check common python locations
$candidatePaths = @(
    "python.exe",
    "C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
)

foreach ($p in $candidatePaths) {
    try {
        $ver = & $p --version 2>$null
        if ($LASTEXITCODE -eq 0 -and $ver) {
            $pythonExe = $p
            Write-Host "  Found Python: $p ($ver)" -ForegroundColor Green
            break
        }
    } catch {}
}

if (-not $pythonExe) {
    Write-Host "ERROR: No valid Python 3.8+ installation found on PATH or standard directories." -ForegroundColor Red
    Write-Host "Please install Python 3.10, 3.11, or 3.12 from python.org." -ForegroundColor Red
    exit 1
}

# 3. Check PyTorch & CUDA
Write-Host "`n[3/6] Verifying PyTorch Environment..." -ForegroundColor Yellow
$torchInstalled = $false
$cudaAvailable = $false

$torchCheck = & $pythonExe -c "import torch; print(f'{torch.__version__}|{torch.cuda.is_available()}|{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 2>$null
if ($LASTEXITCODE -eq 0 -and $torchCheck) {
    $parts = $torchCheck.Split("|")
    $torchInstalled = $true
    Write-Host "  PyTorch is installed: Version $($parts[0])" -ForegroundColor Green
    if ($parts[1] -eq "True") {
        $cudaAvailable = $true
        Write-Host "  CUDA is available on GPU: $($parts[2])" -ForegroundColor Green
    } else {
        Write-Host "  CUDA is NOT available (CPU mode only)." -ForegroundColor DarkYellow
    }
} else {
    Write-Host "  PyTorch is not currently installed in this Python environment." -ForegroundColor Yellow
    
    $shouldInstall = $InstallTorch
    if (-not $shouldInstall -and -not $NonInteractive) {
        $response = Read-Host "Would you like to install PyTorch now? Estimated space: ~3.5 GB (leaves > 20 GB free). [Y/n]"
        if ($response -eq "" -or $response -match "^[Yy]") {
            $shouldInstall = $true
        }
    }

    if ($shouldInstall) {
        if ($CpuOnly) {
            Write-Host "  Installing PyTorch (CPU version)..." -ForegroundColor Cyan
            & $pythonExe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        } else {
            Write-Host "  Installing PyTorch with CUDA 12.1 (NVIDIA GPU Acceleration)..." -ForegroundColor Cyan
            & $pythonExe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        }
        if ($LASTEXITCODE -ne 0) {
            Write-Host "WARNING: PyTorch install exited with code $LASTEXITCODE. You can install it manually later." -ForegroundColor Yellow
        }
    }
}

# 4. Verify Core Dependencies
Write-Host "`n[4/6] Verifying Required Python Packages..." -ForegroundColor Yellow
$reqs = @("rawpy", "tifffile", "numpy", "opencv-python", "scipy", "pillow", "exifread", "requests")
foreach ($pkg in $reqs) {
    $hasPkg = & $pythonExe -c "import $pkg; print('OK')" 2>$null
    if ($hasPkg -match "OK") {
        Write-Host "  [OK] $pkg is installed." -ForegroundColor Green
    } else {
        Write-Host "  Installing $pkg..." -ForegroundColor Cyan
        & $pythonExe -m pip install $pkg
    }
}

# 5. Download Pretrained Weights & Configure Portable Paths
Write-Host "`n[5/6] Checking Pretrained Networks & Dynamic Paths..." -ForegroundColor Yellow
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (Test-Path (Join-Path $scriptDir "deeprep_gui.py")) {
    $rootDir = $scriptDir
} elseif (Test-Path (Join-Path (Split-Path -Parent (Split-Path -Parent $scriptDir)) "deeprep_gui.py")) {
    $rootDir = Split-Path -Parent (Split-Path -Parent $scriptDir)
} else {
    $rootDir = $scriptDir
}

$weightsDir = Join-Path $rootDir "pretrained_networks"
$downloader = Join-Path $rootDir "download_weights.py"

if (Test-Path $downloader) {
    & $pythonExe $downloader $weightsDir
}

# Write portable admin/local.py
$adminDir = Join-Path $rootDir "admin"
if (Test-Path $adminDir) {
    $localPy = Join-Path $adminDir "local.py"
    $localContent = @"
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
"@
    Set-Content -Path $localPy -Value $localContent -Encoding UTF8
    Write-Host "  Configured portable settings in: $localPy" -ForegroundColor Green
}

# 6. Darktable Integration
Write-Host "`n[6/6] Checking Darktable Installation..." -ForegroundColor Yellow
$dtLuaContrib = "$env:LOCALAPPDATA\darktable\lua\contrib"
$deeprepLua = Join-Path $rootDir "deeprep.lua"
$deeprepGui = Join-Path $rootDir "deeprep_gui.py"

if (Test-Path $deeprepLua) {
    if (Test-Path $dtLuaContrib) {
        $destLua = Join-Path $dtLuaContrib "deeprep.lua"
        Copy-Item -Path $deeprepLua -Destination $destLua -Force
        Write-Host "  Installed Darktable plugin to: $destLua" -ForegroundColor Green
        
        $luarc = "$env:LOCALAPPDATA\darktable\luarc"
        if (Test-Path $luarc) {
            $luarcContent = Get-Content $luarc -Raw
            if ($luarcContent -notmatch "contrib/deeprep") {
                Add-Content -Path $luarc -Value "`nrequire `"contrib/deeprep`""
                Write-Host "  Enabled deeprep plugin in luarc ($luarc)" -ForegroundColor Green
            }
        }

        $darktablerc = "$env:LOCALAPPDATA\darktable\darktablerc"
        if (Test-Path $darktablerc) {
            $dtConfig = Get-Content $darktablerc
            $hasPy = $false
            $hasScript = $false
            $newConfig = $dtConfig | ForEach-Object {
                if ($_ -match "^lua/deeprep/python_bin=") {
                    $hasPy = $true
                    "lua/deeprep/python_bin=$pythonExe"
                } elseif ($_ -match "^lua/deeprep/script_path=") {
                    $hasScript = $true
                    "lua/deeprep/script_path=$deeprepGui"
                } else {
                    $_
                }
            }
            if (-not $hasPy) {
                $newConfig += "lua/deeprep/python_bin=$pythonExe"
            }
            if (-not $hasScript) {
                $newConfig += "lua/deeprep/script_path=$deeprepGui"
            }
            Set-Content -Path $darktablerc -Value $newConfig -Encoding UTF8
            Write-Host "  Configured native Darktable preferences in darktablerc" -ForegroundColor Green
        }
    } else {
        Write-Host "  Darktable Lua folder not detected at $dtLuaContrib. Plugin saved at $deeprepLua." -ForegroundColor DarkYellow
    }
}

# Create Launcher Batch File
$launcherBat = Join-Path $rootDir "run_deeprep.bat"
$batContent = @"
@echo off
setlocal
cd /d "%~dp0"
"$pythonExe" "%~dp0deeprep_gui.py" %*
"@
Set-Content -Path $launcherBat -Value $batContent -Encoding ASCII
Write-Host "  Created desktop launcher: $launcherBat" -ForegroundColor Green

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "  Deep-Rep Installation & Verification Complete!          " -ForegroundColor Green
Write-Host "  To launch the GUI, double-click run_deeprep.bat        " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
