# AGENTS.md — Deep-Rep Toolkit Engineering Guide for AI Agents

> **Note for AI Agents**: Read this document thoroughly before proposing, inspecting, modifying, or executing code in this repository. It documents non-obvious low-level system constraints, tensor shapes, precision boundaries, file synchronization rules, and DNG metadata architectures established across development sessions.

---

## 1. Repository Identity & Project Rules

- **Project Nature**: This is a private, personal local fork of [`goutamgmb/deep-rep`](https://github.com/goutamgmb/deep-rep), a PyTorch-based neural framework for multi-frame RAW burst image restoration, super-resolution, and denoising.
- **Git Policy**: **NEVER publish, push, or commit changes to the upstream remote repository.** All work is strictly local and offline.
- **Primary Deliverable**: `deeprep_gui.py` — A standalone Tkinter desktop GUI running on WSL2 that allows users to select RAW DNG bursts, run neural inference, monitor VRAM/RAM, and export restored images.
- **Dual-Location Synchronization Rule**:
  - The repository lives in WSL2 at: `/home/aronwsl/deep-rep`.
  - The Windows host scratch directory is: `C:\Users\Admin\.gemini\antigravity-ide\scratch` (mounted in WSL2 as `/mnt/c/Users/Admin/.gemini/antigravity-ide/scratch`).
  - Core deliverable files (`deeprep_gui.py`, `AGENTS.md`) exist in **both** locations. Whenever updating one location, **ALWAYS synchronize changes to the other location**.

---

## 2. Low-Level System Profile & Environment

- **OS / Subsystem**: Windows 11 Host running **WSL2 (Ubuntu)**.
- **Hardware Profile**:
  - **GPU**: NVIDIA GeForce RTX 2060 (8 GB VRAM, Turing Architecture, Compute Capability 7.5).
  - **RAM**: 16 GB Host System RAM.
- **Conda Environment**: `/home/aronwsl/miniconda3/envs/env-deeprep`
  - Activation: `source /home/aronwsl/miniconda3/bin/activate env-deeprep`
  - Python Executable: `/home/aronwsl/miniconda3/envs/env-deeprep/bin/python`
- **CLI & Script Execution Rules for Agents**:
  - **PowerShell Command Invocation**: When executing WSL commands via `run_command` in PowerShell, **NEVER pass multiline inline Python scripts with double quotes, `$`, or complex quotation marks (`python -c "..."`)**. PowerShell parses quotes and variables before invoking WSL, causing unexpected token and syntax errors.
  - **Best Practice**: Always write scratch scripts to a `.py` file on disk (in `scratch/deep-rep` or `/home/aronwsl/deep-rep`) and invoke them using `wsl --% -d Ubuntu /home/aronwsl/miniconda3/envs/env-deeprep/bin/python <script_path>`. The `--%` stop-parsing token ensures PowerShell passes all subsequent arguments raw without interpretation.
  - **Python Search Path**: Scratch scripts outside the repository root must set `PYTHONPATH=/home/aronwsl/deep-rep` so that imports (`admin`, `models`, `evaluation`) resolve properly.
  - **Process RAM Monitoring**: `psutil` is **NOT** installed in `env-deeprep`. Do not attempt to import `psutil`. Instead, parse Linux `/proc/self/status` for native, zero-dependency metrics: `VmRSS:` (current resident set size) and `VmHWM:` (peak resident memory / High Water Mark).
- **Installed Key Packages**:
  - `torch 2.4.1+cu121`, `torchvision 0.19.1+cu121`, `cupy-cuda12x 12.3.0`
  - `rawpy 0.21.0` (LibRaw bindings), `tifffile 2023.7.10`, `exifread 3.5.1`, `pillow 10.4.0`
  - *Note*: Standalone CLI `exiftool` and `piexif` are **NOT** installed; rely on `tifffile`, `exifread`, and `rawpy`.
- **Sample Datasets & Bursts for Verification**:
  - Sample bursts exist directly in `/home/aronwsl/deep-rep`:
    - **Canon PowerShot SX280 HS Real-World Burst**: `CRW_0101.DNG` through `CRW_0107.DNG` (7 frames, 12 MP sensor $3044 \times 4072$, packed Bayer $1522 \times 2036$, black level 127, white level 4095). *Note: Filenames are zero-padded 4-digit numbers (`CRW_0101.DNG` not `CRW_101.DNG`)*.
    - Canon PowerShot SX280 HS pairs: `input_1.dng`, `input_2.dng`
    - Leica M9 (with embedded XMP packet): `sample.dng`, `sample222.dng`, `sample3.dng`, `sample4.dng`
- **Critical PyTorch 2.4 WSL2 Caveat**:
  - Whenever loading checkpoints (`torch.load`), **ALWAYS** pass `map_location='cpu'` and `mmap=False`.
  - Failing to set `mmap=False` causes virtual memory mapping page faults across WSL2 virtual disk mount boundaries, resulting in uncatchable `SIGBUS` crashes.
- **Custom CUDA / C++ Extensions**:
  - Frame alignment relies on PWCNet with custom CuPy CUDA kernels (`external/pwcnet/correlation/correlation.py`).
  - The correlation kernel compiles via CuPy on the active CUDA context.

---

## 3. Image Pipeline & Tensor Specifications

### Input Handling & Generalized Bayer Alignment
- **Supported Formats**: `.dng`, `.DNG`, `.cr2`, `.CR2` (case-insensitive selection).
- **Library**: `rawpy` (LibRaw bindings).
- **Universal Bayer CFA Alignment**:
  - Cameras use different Bayer CFA pattern arrangements:
    - Canon PowerShot SX280 HS (`CRW_*.DNG`): `RGGB` (`raw_pattern = [[0, 1], [3, 2]]`).
    - Canon EOS 5D Mark II (`_MG_*.CR2`): `GBRG` (`raw_pattern = [[3, 2], [0, 1]]`).
    - Other sensors may use `BGGR` or `GRBG`.
  - **Dynamic Offset Detection**:
    - The pipeline reads `raw.raw_pattern` and `raw.color_desc` to find the exact 2D coordinate `(r_offset, c_offset)` where `'R'` is located.
    - Slicing `raw_vis[r_offset : r_offset + H_even, c_offset : c_offset + W_even]` mathematically guarantees that the top-left $2\times 2$ block is strictly **`RGGB`**:
      - `(0, 0)` is Red (Channel 0)
      - `(0, 1)` is Green 1 (Channel 1)
      - `(1, 0)` is Green 2 (Channel 2)
      - `(1, 1)` is Blue (Channel 3)
    - Dynamic channel indices `idx_r`, `idx_g1`, `idx_g2`, `idx_b` are used to apply sensor-specific `black_level_per_channel` individually to each channel.
- **Dynamic Normalization**:
  - Camera black levels vary per sensor (e.g. `127` on Canon SX280, `[1030, 1028, 1029, 1030]` on 5D Mark II, `0` on Leica M9).
  - White levels vary by sensor bit-depth (e.g. `4095` for 12-bit, `15600`/`16383` for 14-bit).
  - Normalization: `packed = (raw_data - black_level) / max(float(white_level - black_level), 1.0)`, clipped to `[0.0, 1.0]`.
- **4-Channel Bayer Packing**:
  - A 2D Bayer array of size $H \times W$ is packed into a 4-channel tensor of shape `[B, T, 4, H/2, W/2]`:
    - Channel 0: $R$ (`raw_vis[0::2, 0::2] - bl[idx_r]`)
    - Channel 1: $G_1$ (`raw_vis[0::2, 1::2] - bl[idx_g1]`)
    - Channel 2: $G_2$ (`raw_vis[1::2, 0::2] - bl[idx_g2]`)
    - Channel 3: $B$ (`raw_vis[1::2, 1::2] - bl[idx_b]`)

### Model Architectures & Channels
1. **`burst_denoise` (`deeprep_denoise_color_pwcnet.pth`)**:
   - `lr_encoder` expects **6 input channels**: 3 image channels $(R, \frac{G_1+G_2}{2}, B)$ concatenated with a 3-channel noise estimate tensor `[B, 3, H, W]`.
   - Spatial resolution: $1\times$ relative to packed Bayer ($\frac{H}{2} \times \frac{W}{2}$).
2. **`burstsr` / `synburst` (`deeprep_sr_synthetic_default.pth`)**:
   - `lr_encoder` expects **4 input channels** (packed Bayer $(R, G_1, G_2, B)$) without noise estimate.
   - Spatial resolution: The model has $2\times$ latent upsampling + $4\times$ sub-pixel convolution decoder, yielding **$8\times$ spatial expansion relative to packed Bayer** ($4\times$ sensor dimensions, $16\times$ pixel count).
   - For a 12 MP sensor ($3044 \times 4072$), native output is **~198 Megapixels** ($12176 \times 16288$).
   - The GUI provides a **Target Scale selector**:
     - `2.0x` (Recommended): Downsamples the $4\times$ sensor feature map to $2\times$ sensor grid (48 MP from 12 MP) via anti-aliased area supersampling on the GPU.
     - `1.0x`: Native sensor grid (12 MP from 12 MP), razor-sharp supersampled RGB.
     - `4.0x`: Full 16× pixel expansion (requires high Host RAM).

---

## 4. Precision & Memory Management

### Precision Boundaries (FP32 vs FP16)
- **Do NOT run naive FP16 across the entire network**:
  - **PWCNet Alignment**: Optical flow correlation volume calculations underflow/overflow in `float16`, producing `NaN` offsets and crashing.
  - **Recurrent Optimizer**: Recurrent gradient unrolling accumulates rounding errors in FP16, dropping fidelity to ~15 dB PSNR.
- **Selective / Hybrid Precision (Implemented in `predict_tiled`)**:
  - **Alignment Net**: Runs in full **FP32** (`torch.amp.autocast('cuda', enabled=False)`).
  - **Recurrent Optimizer**: Runs in **FP32**.
  - **High-Resolution Decoder (`hr_decoder`)**: Runs in **FP16** (`torch.amp.autocast('cuda', enabled=True)`).
  - This hybrid execution achieves **93.19 dB PSNR** fidelity relative to pure FP32 while cutting VRAM significantly.

### Recurrent Optimizer Memory Decoupling (Evaluation Patches)
- **The Autograd Graph Accumulation Problem**:
  - Upstream `SteepestDescentOptimizer` and `BackwardWarp` (`backward_warp.py`) were designed for end-to-end training.
  - `BackwardWarp` defaulted to `create_graph=True, retain_graph=True`, retaining the entire forward warping computation graph and all activations in memory.
  - Additionally, `y_current` was not detached between recurrent iterations, chaining the computation graph across all 3 optimization iterations and causing VRAM to peak at **7.26 GB** on 7-frame bursts.
- **The Solution (`OptimizedBackwardWarp` & `optimized_optimizer_forward`)**:
  - During evaluation (`not torch.is_grad_enabled()`), `create_graph=False` and `retain_graph=False` are enforced, bypassing second-order graph construction.
  - `y_current` is explicitly detached at the beginning and end of each optimization iteration: `y_current = (y_current - step).detach()`.
  - All intermediate iteration tensors (`y, y_warped, y_warped_ds, data_residual, grad_y_warped, grad_y, grad_y_current, h_warp, h, step`) are explicitly deleted.
  - **Result**: Optimizer execution time drops from **920 ms to 112 ms per tile (8.2× speedup)**, peak VRAM drops by over **1 GB**, and overall burst inference time on 7 frames drops from **124.26s down to 83.20s (33% faster)**.
- **Critical PyTorch Autograd Scoping Nuances**:
  - **Enclosing `with torch.enable_grad():`**: In `optimized_optimizer_forward`, both `y_current.requires_grad_(True)` AND the creation of `y = y_current.unsqueeze(1).expand(...)` MUST be executed inside `with torch.enable_grad():`. If `y` is created while autograd is disabled globally, PyTorch fails during `torch.autograd.grad` with `RuntimeError: One of the differentiated Tensors appears to not have been used in the graph`.
  - **`allow_unused=True`**: `torch.autograd.grad(y_warped, y, grad_y_warped, ...)` MUST pass `allow_unused=True`. In `self.warp_y`, the reference keyframe (`y[:, :1, ...]`) is concatenated directly without optical flow warping (`lispr_warp.warp`), so `y[:, :1]` has no autograd gradient edges in the warped tensor.
  - **Optimizer Internal Tensor Dimensions**: The encoder feature map $x$ has shape `[B, T, 256, H_enc, W_enc]` (256 latent channels). At tile size 192 with 7 frames, each 5D tensor (`x`, `weights`, `y_warped`, `data_residual`, `grad_y_warped`, `grad_y`) consumes ~264 MB. Detaching and deleting intermediate tensors every iteration is mandatory to keep memory flat.

### Host RAM & In-Place Pipeline Architecture
1. **Preallocated Burst Loading (`load_dng_burst`)**:
   - Rather than creating separate float32 numpy arrays for each DNG and calling `torch.stack`, preallocates a single contiguous tensor `burst_stack = torch.empty((1, T, 4, H//2, W//2))` and packs slices in-place.
2. **GPU Windowing & In-Place Accumulator Operations**:
   - The 2D Hann cosine blending window is applied on the GPU (`tile_out.mul_(window_2d_gpu)`) before copying to CPU.
   - Accumulation and normalization use in-place CPU operations: `output_accum.add_()`, `weight_accum.add_()`, `weight_accum.clamp_(min=1e-5)`, and `output_accum.div_(weight_accum)`.
   - Eliminates intermediate tensor duplications, saving **756 MB Host RAM at 2.0x scale** and **3.03 GB at 4.0x scale**.
3. **Direct uint16 Conversion & DNG Export**:
   - Converts the output float tensor to `torch.uint16` in-place on CPU (`output.mul_(65535.0).clamp_(0.0, 65535.0).round_().to(torch.uint16)`).
   - Deletes `output` and `burst_stack` and runs `gc.collect()` before writing the DNG.
   - `save_linear_dng()` accepts the pre-scaled uint16 array directly without creating redundant float copies, saving **2.26 GB Host RAM at 2.0x scale** and **9.06 GB at 4.0x scale**.
4. **Smart Cache Cleanup**:
   - Unconditional per-tile `torch.cuda.empty_cache()` calls are eliminated. Instead, `empty_cache()` is called only when reserved VRAM exceeds 80% of device capacity or periodically every 16 tiles, avoiding CUDA pipeline stalls.

### Empirical Resource Estimation Models
Resource estimation in `update_estimation_info()` is calibrated against empirical hardware benchmarks on RTX 2060 (8 GB VRAM) and 16 GB Host RAM:

- **`burstsr` / `synburst` (Peak Reserved VRAM)**:
  - Base VRAM (2 frames, FP16 decoder):
    - `128×128`: 0.85 GB | `192×192`: 1.75 GB *(Recommended)* | `256×256`: 3.15 GB | `384×384`: 6.85 GB | `512×512`: 12.20 GB
  - Scaling per additional frame ($T - 2$):
    - `128×128`: +0.18 GB/frame | `192×192`: +0.43 GB/frame | `256×256`: +0.75 GB/frame | `384×384`: +1.50 GB/frame | `512×512`: +2.60 GB/frame
  - Pure FP32 decoder overhead:
    - `128×128`: +0.25 GB | `192×192`: +0.85 GB | `256×256`: +1.55 GB | `384×384`: +3.50 GB | `512×512`: +6.00 GB

- **`burst_denoise` (Peak Reserved VRAM)**:
  - Base VRAM (2 frames):
    - `128×128`: 0.45 GB | `192×192`: 0.55 GB | `256×256`: 0.70 GB | `384×384`: 1.05 GB | `512×512`: 1.55 GB
  - Scaling per additional frame ($T - 2$):
    - `128×128`: +0.03 GB/frame | `192×192`: +0.08 GB/frame | `256×256`: +0.14 GB/frame | `384×384`: +0.35 GB/frame | `512×512`: +0.58 GB/frame

- **Peak Host RAM Formula (Both Modules)**:
  $$\text{RAM}_{\text{peak}} \approx 0.85\text{ GB (Base + Net)} + \frac{4 \times T \times H \times W}{1024^3} + \max\left(\frac{16 \times H_{\text{out}} \times W_{\text{out}}}{1024^3}, \frac{6 \times H_{\text{out}} \times W_{\text{out}}}{1024^3}\right) + \text{Headroom (0.45 GB)}$$

---

## 5. Output Standards & DNG Export

- **File Format**: **Adobe Linear RAW DNG (`*.dng` / `*.DNG`)**.
- **Case Matching**: Output file extension must match the case of the keyframe (`.dng` $\rightarrow$ `_restored.dng`, `.DNG` $\rightarrow$ `_restored.DNG`).
- **Tag Structure**:
  - `PhotometricInterpretation = 34892` (`LinearRaw`).
  - `SamplesPerPixel = 3` (Interleaved RGB, 16-bit uint16 scaled from `[0.0, 1.0]` to `[0, 65535]`).
  - `BitsPerSample = (16, 16, 16)`.
  - `PlanarConfiguration = 1` (Chunky RGB).
  - `DNGVersion = (1, 4, 0, 0)`.
  - `DNGBackwardVersion = (1, 1, 0, 0)`.

### Tag Filtering & Preservation Rules (`save_linear_dng`)
- **Tags That MUST Be Stripped / Excluded**:
  - **CFA / Bayer Pattern**: `CFAPattern` (33422, 41730), `CFARepeatPatternDim` (33421), `CFAPlaneColor` (50710), `CFALayout` (50711). *(Prevents RAW photo editors from attempting Bayer demosaicing on full 3-channel RGB data).*
  - **Sensor Black/White Levels**: `BlackLevel` (50714), `WhiteLevel` (50717), `BlackLevelRepeatDim` (50713), `BlackLevelDeltaH/V` (50715/50716), `LinearizationTable` (50712). *(Data is already zero-subtracted and normalized to full 16-bit range; keeping original black levels crushes dark tones).*
  - **Sensor Geometry / Crop Bounds**: `DefaultCropOrigin` (50719), `DefaultCropSize` (50720), `ActiveArea` (50829), `DefaultUserCrop` (50742). *(Output may be super-resolved or demosaiced; sensor crop coordinates miscrop output images).*
  - **Noise Profile**: `BaselineNoise` (50731). *(Neural burst processing alters/denoises the noise floor; original baseline noise metadata no longer applies and causes photo editors to apply inappropriate noise reduction).*
  - **TiffWriter Internal Layout & SubIFD Offsets**: `SubIFDs` (330), `NewSubfileType` (254), `SubfileType` (255), `ImageWidth` (256), `ImageLength` (257), `BitsPerSample` (258), `Compression` (259), `PhotometricInterpretation` (262), `StripOffsets` (273), `RowsPerStrip` (278), `StripByteCounts` (279), `TileWidth` (322), `TileLength` (323), `TileOffsets` (324), `TileByteCounts` (325), `ExifTag` (34665), `GPSTag` (34853), `InteroperabilityIFD` (40965).

- **Tags That MUST Be Preserved**:
  - **Camera Identification**: `Make` (271), `Model` (272), `UniqueCameraModel` (50708), `CameraSerialNumber` (50735), `BodySerialNumber` (42033), `CameraOwnerName` (42032), `Artist` (315), `Copyright` (33432).
  - **Timestamps**: `DateTime` (306), `DateTimeOriginal` (36867), `DateTimeDigitized` (36868), `OffsetTime` (36880), `OffsetTimeOriginal` (36881), `SubsecTime` (37520), `SubsecTimeOriginal` (37521).
  - **Exposure & Shooting Settings**: `ExposureTime` (33434), `FNumber` (33437), `ISOSpeedRatings` / `PhotographicSensitivity` (34855), `ShutterSpeedValue` (37377), `ApertureValue` (37378), `BrightnessValue` (37379), `ExposureBiasValue` (37380), `MaxApertureValue` (37381), `ExposureProgram` (34850), `MeteringMode` (37383), `Flash` (37385), `FocalLength` (37386), `FocalLengthIn35mmFilm` (41989), `ExposureMode` (41986), `WhiteBalance` (41987), `SensitivityType` (34864).
  - **Lens Information**: `LensInfo` (50736), `LensModel` (42036), `LensMake` (42035), `LensSerialNumber` (42037), `LensSpecification` (42034).
  - **Color Calibration & Matrices**: `ColorMatrix1` (50721), `ColorMatrix2` (50722), `CameraCalibration1` (50723), `CameraCalibration2` (50724), `AnalogBalance` (50727), `AsShotNeutral` (50728), `AsShotWhiteXY` (50729), `BaselineExposure` (50730), `BaselineSharpness` (50732), `LinearResponseLimit` (50734), `CalibrationIlluminant1` (50778), `CalibrationIlluminant2` (50779), `BestQualityScale` (50780), `RawDataUniqueID` (50781), `ForwardMatrix1` (50964), `ForwardMatrix2` (50965), `MakerNoteSafety` (50741), `ProfileName` (50936).
  - **XMP & Orientation**: `XMP` (700), `Orientation` (274), `XResolution` (282), `YResolution` (283), `ResolutionUnit` (296).

### Exif Sub-IFD Unpacking Nuance
- In many camera DNGs (such as Canon), shooting parameters are stored in the `ExifTag` (34665) sub-IFD dictionary rather than directly in IFD0.
- `tifffile` rejects tag 34665 in `extratags` (filtered by `TIFF.TAG_FILTERED`).
- **Solution**: Unpack `src_page.tags[34665].value` dictionary in Python, map tag names to standard TIFF tag IDs via `EXIF_TAG_MAP`, and write them directly as top-level extratags.
- Signed rationals (e.g. `ExposureBiasValue`, `ShutterSpeedValue`, `BrightnessValue`) must use `dtype='2i'` (SRATIONAL), while unsigned rationals use `'2I'`.

### Non-DNG Color Calibration Synthesis (CR2 and Proprietary RAWs)
- When restoring non-DNG files (such as Canon `.CR2`), the source TIFF IFDs do not contain standard DNG tags `ColorMatrix1` (50721) or `ColorMatrix2` (50722).
- Leaving these tags absent causes RAW developers (Darktable, Lightroom, RawTherapee) to render the Linear DNG with incorrect colorimetry or missing camera profile warnings.
- **Solution (`save_linear_dng`)**:
  - Automatically queries `rawpy` for the camera's baseline color transformation: `raw.rgb_xyz_matrix[:3, :3]`.
  - In LibRaw / dcraw, `rgb_xyz_matrix` contains the Adobe D65 color matrix coefficients mapping XYZ to camera RGB space (e.g. `[0.4716, 0.0603, -0.0830, -0.7798, 1.5474, 0.2480, -0.1496, 0.1937, 0.6651]` on Canon 5D Mark II).
  - Packs these 9 values into standard `ColorMatrix2` (tag 50722, SRATIONAL `2i`), pairs with `CalibrationIlluminant2 = 21` (D65 Daylight), and synthesizes `AsShotNeutral` (tag 50728, RATIONAL `2I`, 3 values) from `raw.camera_whitebalance`.
  - Maps `Model` to `UniqueCameraModel` (tag 50708).

### Provenance Tracking (`UserComment` & `ImageDescription`)
- Restored images include a structured EXIF `UserComment` (tag 37510) with standard 8-byte ASCII header (`b"ASCII\x00\x00\x00"`) and matching `ImageDescription` (tag 270):
  ```
  Deep-Rep Multi-Frame Burst Restoration
  Keyframe: <basename>
  Extra Frames (<N> frames: <frame1>, <frame2>, ...)
  Model: <model_filename> (Module: <module>)
  Tiling: Tile Size <tile_sz>x<tile_sz> px (Overlap: <overlap> px)
  Precision: FP16 (decoder) / FP32 (alignment & recurrent optimizer)
  Target Scale: <scale>
  ```
- Pass `metadata=None` to `TiffWriter.write()` along with `description=user_comment_text` to prevent `tifffile` from overwriting the description with automatic JSON shape metadata.

---

## 6. Automated Headless Testing & Regression Verification

When testing or continuing development on `deeprep_gui.py` across sessions or in background agent tasks, Tkinter UI windows cannot open interactively without an active X-server/Wayland display. Use the following headless execution pattern:

### CLI Headless Batch Mode
```bash
python deeprep_gui.py --headless \
  --keyframe /path/to/keyframe.CR2 \
  --extra-frames /path/to/frame2.CR2 /path/to/frame3.CR2 \
  --output-dir /path/to/output \
  --module burstsr \
  --sr-scale 2.0 \
  --tile-size 192 \
  --fp16
```

### Headless Python Execution Pattern
```python
import tkinter as tk
from deeprep_gui import DeepRepGUI

root = tk.Tk()
root.withdraw()  # Suppress graphical window
app = DeepRepGUI(root)

# Mock messagebox.showinfo to prevent blocking on UI popups
tk.messagebox.showinfo = lambda title, msg: print(f"[INFOBOX] {title}: {msg}")

# Configure paths and settings
app.keyframe_path.set('/home/aronwsl/deep-rep/CRW_0101.DNG')
app.extra_frames_paths = [f'/home/aronwsl/deep-rep/CRW_010{i}.DNG' for i in range(2, 8)]
app.output_dir.set('/home/aronwsl/deep-rep/output_test')
app.selected_module.set('burstsr')
app.update_names_dropdown()
app.sr_scale_var.set("2.0x (2× Sensor Grid - 4× Pixels, Balanced)")
app.tile_size_var.set("192 (192x192 px - Low VRAM, Recommended)")
app.use_fp16.set(True)
app.use_gpu.set(True)
app.update_estimation_info()

# Execute processing synchronously in the main thread for deterministic testing
app.process_burst()
root.destroy()
```

### Empirical Regression Verification Targets (`CRW_0101.DNG`–`CRW_0107.DNG`)
When verifying code modifications on the 7-frame Canon SX280 burst at tile size 192 (2.0x scale):
- **Total Pipeline Runtime**: $\le 95.0\text{s}$ (including DNG writing).
- **Peak VRAM Reserved**: $3.80\text{--}3.95\text{ GB}$ (must NOT exceed 4.0 GB).
- **Peak Host RAM**: $\le 3.0\text{ GB}$ (must NOT spike above 3.5 GB during DNG writing).
- **Lossless Quality Threshold**:
  - Restored image shape: `(6088, 8144, 3)`, dtype `uint16`.
  - PSNR relative to baseline: $\ge 93.19\text{ dB}$.
  - Max absolute 16-bit pixel difference: $\le 64$ out of 65535 ($<0.1\%$).
- **File Integrity**:
  - Must produce standard Adobe Linear RAW DNG (`PhotometricInterpretation = 34892`).
  - CFA tags (33421, 33422, 50710, 50711) and Black/White level tags (50714, 50717) must be completely absent.

---

## 7. Cross-Platform Architecture, Installers & Darktable Integration

### Pure PyTorch Correlation Volume Fallback (`pytorch_correlation_fallback`)
- **The CuPy Bottleneck**: Upstream Deep-Rep strictly required CuPy with custom compiled CUDA kernels (`external/pwcnet/correlation/correlation.py`). This caused major portability hurdles on Windows (requiring MSVC and CUDA Toolkit) and rendered CPU-only execution impossible (`NotImplementedError`).
- **The Pure PyTorch Solution**:
  - Implemented `pytorch_correlation_fallback(first, second, max_disp=4)`.
  - Uses 2D spatial constant padding and vectorized shifted slice inner-products across all 81 offsets in $[-4, 4] \times [-4, 4]$.
  - **Numerical Equivalence**: Max absolute tensor difference relative to the custom CuPy CUDA kernel is $\le 2.38 \times 10^{-7}$ (exact float32 precision limits).
  - Automatically activates when `cupy` is missing or when executing on CPU, DirectML, or non-NVIDIA hardware.

### Standalone Cross-Platform Installers
- **Windows**: `install_windows.ps1` (with `install_windows.bat` double-clickable launcher).
  - Verifies Python installation (>= 3.8, tested through 3.12).
  - Verifies free disk space on C: (enforces $\ge 4.0\text{ GB}$ safety threshold).
  - Checks/installs PyTorch (CUDA 12.1 or CPU) and core dependencies (`rawpy`, `tifffile`, `numpy`, `opencv-python`, `scipy`, `pillow`, `exifread`, `requests`).
  - Runs `download_weights.py` to stream model weights directly from Google Drive using standard Python `urllib` (no `gdown` dependency).
  - Writes portable dynamic `admin/local.py` and creates `run_deeprep.bat`.
- **Linux**: `install_linux.sh`.
  - Shell installer supporting Ubuntu/Debian, Fedora, Arch.
  - Verifies dependencies, downloads pretrained weights, configures portable settings, and generates `run_deeprep.sh`.

### Darktable Lua Plugin (`deeprep.lua`)
- **Location**: Installed to `$env:LOCALAPPDATA/darktable/lua/contrib/deeprep.lua` (Windows) and `~/.config/darktable/lua/contrib/deeprep.lua` (Linux), enabled via `luarc`.
- **UI Container**: Integrates into Darktable's Lighttable right panel (`deeprep_lib`).
- **Keyframe Designation**:
  - *Select from List*: Dynamically updates a dropdown with all selected filenames.
  - *Highest Star Rating*: The image with the most stars is the Keyframe; remaining images are Extra Frames.
  - *Red Color Label*: The image with a red color label is designated as Keyframe.
  - *First Selected*: Fallback to the first selected image.
- **Workflow Modes**:
  1. **Open in Deep-Rep GUI**: Forwards selected RAW files to `deeprep_gui.py` with all paths and settings pre-filled.
  2. **Process in Background & Import**: Executes headless inference (`--headless`) and automatically calls `dt.database.import()` to import the resulting Adobe Linear RAW DNG directly into the filmroll with `deep-rep|restored` tags.

### Standalone Windows Executables (`DeepRepInstaller.exe` & `DeepRepLauncher.exe`)
- **Zero-Dependency Native Compiles**:
  - Built using Windows .NET Framework 4.8 C# compiler (`C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe`).
  - Requires 0 external tools, 0 third-party packages, and adds negligible footprint (<25 KB total).
- **`DeepRepInstaller.exe`**:
  - Standalone GUI installer application for Windows.
  - Performs live system checks: available disk space on C: (verifies $\ge 4.0\text{ GB}$ headroom), detects host Python and PyTorch/CUDA installations.
  - Provides a single-click interface to run environment verification, download model weights, configure Darktable Lua, and generate desktop shortcuts.
- **`DeepRepLauncher.exe`**:
  - Clean GUI executable launcher (Windows subsystem, no black console window).
  - Automatically discovers host Python 3.10–3.12 and forwards command line arguments directly to `deeprep_gui.py`.

### Native Windows & CR2 Verification Benchmark
- **Canon EOS 5D Mark II CR2 Verification** (`_MG_9319_10-21-26.CR2` + `_MG_9320_10-21-26.CR2`):
  - **Environment**: Native Windows Host, Python 3.12.10, PyTorch `2.5.1+cu121`, NVIDIA RTX 2060 SUPER (8 GB VRAM).
  - **Tile Configuration**: Tile Size 192, Overlap 32, Scale 1.0x (216 tiles total).
  - **Results**: Executed in ~115s total, peak VRAM 3.82 GB, successfully wrote 120.9 MB Adobe Linear RAW DNG (`(3750, 5634, 3)`, uint16, `PhotometricInterpretation = 34892`).
  - **Darktable Compatibility**: Successfully validated with `darktable-cli.exe`; renders with accurate color calibration via synthesized `ColorMatrix2` and `AsShotNeutral`.

