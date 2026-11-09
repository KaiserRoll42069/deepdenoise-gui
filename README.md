# DeepDenoise GUI — Neural RAW Burst Restoration & Super-Resolution Toolkit

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-green.svg)]()

A standalone cross-platform desktop application, command-line tool, and Darktable integration plugin for deep-learning multi-frame RAW burst image restoration, super-resolution, and denoising.

---

## License & Attribution

> **Important Notice**: This repository is derived from and builds upon the [Deep-Rep](https://github.com/goutamgmb/deep-rep) neural restoration framework.
> 
> **Copyright (c) 2021 Huawei Technologies Co., Ltd.**  
> Licensed under **CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International)**.  
> You may obtain a copy of the License at [https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).  
> **This software is released strictly for academic and non-commercial research use only.** For commercial licensing, please contact Huawei Technologies Co., Ltd.
> 
> **Original Academic Publication**:  
> *Deep Reparametrization of Multi-Frame Super-Resolution and Denoising* (ICCV 2021 Oral)  
> Goutam Bhat, Martin Danelljan, Fisher Yu, Luc Van Gool, and Radu Timofte.  
> [[Arxiv Paper](https://arxiv.org/pdf/2108.08286.pdf)] • [[Original GitHub Repository](https://github.com/goutamgmb/deep-rep)]

---

## Key Features & Enhancements

- **Universal RAW Format Support (`.CR2` & `.DNG`)**:
  - Dynamically detects arbitrary Bayer Color Filter Array (CFA) layouts (`RGGB`, `GBRG`, `BGGR`, `GRBG`) and aligns channels into a standardized 4-channel tensor.
  - Dynamically synthesizes DNG baseline color matrices (`ColorMatrix2` D65 and `AsShotNeutral`) from camera sensor metadata for non-DNG files (such as Canon `.CR2`), guaranteeing true-to-life color rendering in RAW photo editors.
- **Cross-Platform & Zero-Compiler Portable**:
  - Includes a vectorized pure PyTorch optical flow correlation fallback (`pytorch_correlation_fallback`), eliminating the requirement for CuPy, MSVC, or compiled C++/CUDA toolchains. Runs out-of-the-box on standard Windows and Linux Python environments.
- **Standalone Desktop GUI (`deeprep_gui.py` / `DeepRepLauncher.exe`)**:
  - Built-in Tkinter graphical interface with empirical VRAM and Host RAM resource estimators.
  - Supports configurable tiling (128px to 512px) and selective hybrid precision (FP16 decoder + FP32 optimizer) to comfortably process 20+ megapixel bursts on 8 GB GPUs (such as RTX 2060).
- **Darktable Integration Plugin (`deeprep.lua`)**:
  - Native Lighttable panel plugin with flexible keyframe designation (via dropdown selection, highest star rating, or red color label).
  - One-click workflow to launch the interactive GUI or perform headless background batch restoration with automatic filmroll re-import and `deep-rep|restored` tagging.
- **Standard Adobe Linear RAW DNG Export**:
  - Generates 16-bit uncompressed chunky RGB Linear RAW DNGs (`PhotometricInterpretation = 34892`) with preserved EXIF camera metadata, exposure settings, lens tags, and restoration provenance.

---

## 🚀 Quick Download & Installation (No Git Clone Required)

You do **not** need to clone this repository to use DeepDenoise! Download the standalone installer for your OS directly from the **[Releases Tab (Release v1)](https://github.com/KaiserRoll42069/deepdenoise-gui/releases/tag/v1)**:

### 🪟 Windows 10 / 11
1. Download **[`DeepDenoise-Setup.exe`](https://github.com/KaiserRoll42069/deepdenoise-gui/releases/download/v1/DeepDenoise-Setup.exe)**.
2. Double-click to run. The all-in-one setup wizard contains the complete application files embedded inside, checks available disk space ($\ge 4.0\text{ GB}$ safety threshold), verifies Python 3.10–3.12 & PyTorch GPU acceleration, streams neural weights, configures the Darktable Lua plugin, and creates Desktop & Start Menu shortcuts.

### 🐧 Linux (Ubuntu, Debian, Fedora, Arch)
1. Download **[`install_linux.sh`](https://github.com/KaiserRoll42069/deepdenoise-gui/releases/download/v1/install_linux.sh)**.
2. In your terminal, run:
   ```bash
   chmod +x install_linux.sh
   ./install_linux.sh
   ```
   The installer automatically fetches the release-pinned source archive (`deepdenoise-source.zip`), configures dependencies, downloads pretrained weights, registers Darktable integration, and generates desktop application shortcuts.

---

### Alternative: Clone Repository Manually
If you have cloned the Git repository directly:
- **Windows**: Run `installers/windows/install_windows.bat`
- **Linux**: Run `installers/linux/install_linux.sh`

---

### Pretrained Neural Weights

Pretrained weights are **not** hosted directly in this Git repository to avoid repository bloat. They are automatically streamed on setup from the original authors' Google Drive repository using `download_weights.py`:

| Model Name | Checkpoint File | Size | Task |
| :--- | :--- | :--- | :--- |
| **BURST-SR Default** | `deeprep_sr_synthetic_default.pth` | 139 MB | Super-Resolution (1x, 2x, 4x) |
| **Burst Denoise** | `deeprep_denoise_color_pwcnet.pth` | 46 MB | Denoising & Color Restoration |
| **PWCNet Alignment** | `pwcnet-network-default.pth` | 36 MB | Multi-Frame Sub-Pixel Alignment |

To manually download or refresh the weights at any time:
```bash
python download_weights.py pretrained_networks
```

---

## Usage

### 1. Standalone Desktop GUI
- **Windows**: Double-click **`DeepRepLauncher.exe`** (or `run_deeprep.bat`).
- **Linux / CLI**: Run `python deeprep_gui.py` (or `./run_deeprep.sh`).

**Workflow**:
1. Select your **Keyframe** (the primary reference frame) and add one or more **Extra Frames** from your burst.
2. Select your restoration **Module**:
   - `burstsr`: Multi-frame super-resolution and reconstruction.
   - `burst_denoise`: Multi-frame noise reduction and color reconstruction.
3. Configure **Target Scale** (`1.0x` native sensor grid, `2.0x` balanced 4× pixels, or `4.0x` ultra 16× pixels).
4. Review the estimated peak VRAM and Host RAM meter.
5. Click **Process Burst & Save DNG**.

---

### 2. Darktable Lua Plugin Integration

1. Start Darktable.
2. In the **Lighttable** view, open the right-hand panel and locate **Deep-Rep Burst Restoration**.
3. Select your RAW burst images in the filmroll.
4. Choose your preferred **Keyframe Designation**:
   - *Select from List*: Pick a file from the dropdown.
   - *Highest Star Rating*: Mark your intended keyframe with 5 stars.
   - *Red Color Label*: Mark your keyframe with Darktable's red color label.
   - *First Selected*: Default order.
5. Click:
   - **Open in Deep-Rep GUI**: Pre-populates the standalone GUI with your selected images.
   - **Process in Background & Import**: Executes headless processing in the background and automatically imports the restored 16-bit Linear DNG into Darktable with the tag `deep-rep|restored`.

---

### 3. Headless CLI / Batch Processing

For automated scripts, processing pipelines, or remote servers without a display:

```bash
python deeprep_gui.py --headless \
  --keyframe /path/to/frame_01.CR2 \
  --extra-frames /path/to/frame_02.CR2 /path/to/frame_03.CR2 \
  --output-dir /path/to/output \
  --module burstsr \
  --sr-scale 2.0 \
  --tile-size 192 \
  --fp16
```

**CLI Arguments**:
- `--keyframe`, `-k`: Absolute or relative path to reference RAW image (`.cr2`, `.dng`).
- `--extra-frames`, `-e`: Space-separated paths to additional burst frames.
- `--output-dir`, `-o`: Directory to save the restored Adobe Linear RAW DNG.
- `--module`, `-m`: Restoration network (`burstsr`, `burst_denoise`, `synburst`).
- `--sr-scale`: Sensor resolution multiplier (`1.0`, `2.0`, `4.0`).
- `--tile-size`: Tiling window dimension (`128`, `192`, `256`, `384`, `512`).
- `--fp16` / `--no-fp16`: Toggle FP16 decoder acceleration.
- `--cpu`: Force CPU inference (disables CUDA).
- `--headless`: Run headlessly without opening the Tkinter window.

---

## Hardware Recommendations

- **GPU**: NVIDIA GPU with $\ge 6\text{ GB}$ VRAM recommended (tested on RTX 2060 8 GB). CPU execution is supported as fallback.
- **System RAM**: $\ge 16\text{ GB}$ recommended for 2.0x / 4.0x super-resolution on 20+ MP sensors.
- **Disk Space**: $\ge 4.0\text{ GB}$ free space required during installation.

---

## License Notice

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)**.

To view a copy of this license, visit [https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Academic and non-commercial research use only.
