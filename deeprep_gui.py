# ==============================================================================
# Deep-Rep RAW Burst Restoration GUI Toolkit
#
# Original Framework Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International)
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
# For academic / non-commercial research use only.
#
# Based on "Deep Reparametrization of Multi-Frame Super-Resolution and Denoising" (ICCV 2021)
# Authors: Goutam Bhat, Martin Danelljan, Fisher Yu, Luc Van Gool, Radu Timofte.
# Original Repository: https://github.com/goutamgmb/deep-rep
# ==============================================================================

import os
import sys
import types
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from fractions import Fraction
import rawpy
import numpy as np
import cv2
import tifffile
import torch
import torch.nn.functional as F
import traceback

# Force Python to add the deep-rep directory to its path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ============================================================================
# Pure PyTorch Correlation Volume Fallback (cross-platform & CuPy-independent)
# ============================================================================
def pytorch_correlation_fallback(first, second, max_disp=4):
    """
    Pure PyTorch fallback for optical flow correlation volume.
    Mathematically identical to CuPy kernel in external/pwcnet/correlation/correlation.py.
    Runs on CUDA, CPU, DirectML, MPS without requiring C++/CUDA compilation or CuPy.
    """
    B, C, H, W = first.shape
    padded_second = F.pad(second, (max_disp, max_disp, max_disp, max_disp), mode='constant', value=0)
    out_channels = []
    for dy in range(-max_disp, max_disp + 1):
        for dx in range(-max_disp, max_disp + 1):
            y_start = dy + max_disp
            x_start = dx + max_disp
            shifted = padded_second[:, :, y_start : y_start + H, x_start : x_start + W]
            corr = (first * shifted).mean(dim=1, keepdim=True)
            out_channels.append(corr)
    return torch.cat(out_channels, dim=1)

# Check if CuPy is available; if not or if running on non-CUDA, inject fallback
has_cupy = False
try:
    import cupy
    has_cupy = True
except Exception:
    has_cupy = False

if not has_cupy:
    corr_mod = types.ModuleType("external.pwcnet.correlation.correlation")
    corr_mod.FunctionCorrelation = lambda tenFirst, tenSecond: pytorch_correlation_fallback(tenFirst, tenSecond)
    class ModuleCorrelation(torch.nn.Module):
        def forward(self, tenFirst, tenSecond):
            return pytorch_correlation_fallback(tenFirst, tenSecond)
    corr_mod.ModuleCorrelation = ModuleCorrelation
    sys.modules['external.pwcnet.correlation.correlation'] = corr_mod
    print("Notice: CuPy not detected. Pure PyTorch optical flow correlation fallback active.")

# ============================================================================
# Deep-Rep Toolkit Imports & Settings
# ============================================================================
try:
    from evaluation.common_utils import network_param
    from admin.loading import load_network
    import admin.settings as ws_settings
    print("deep-rep modules loaded successfully.")
except ImportError as e:
    print(f"Critical Error: Could not load deep-rep modules. Details: {e}")

# ============================================================================
# Evaluation Memory Optimizations (SteepestDescentOptimizer & BackwardWarp)
# ============================================================================
try:
    import models.deeprep.backward_warp as bw
    import models.deeprep.optimizer_modules as opt_mod

    class OptimizedBackwardWarp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y_warped, y, grad_y_warped, offsets):
            is_training = torch.is_grad_enabled()
            grad_y = torch.autograd.grad(
                y_warped, y, grad_y_warped,
                create_graph=is_training,
                retain_graph=is_training,
                allow_unused=True
            )
            if is_training:
                ctx.save_for_backward(y_warped, y, grad_y_warped, offsets, grad_y[0])
            return grad_y

        @staticmethod
        def backward(ctx, grad_output):
            return bw.BackwardWarp.backward(ctx, grad_output)

    def optimized_optimizer_forward(self, y_init, x, offsets, num_iter=None, noise_estimate=None):
        num_iter = self.num_iter if num_iter is None else num_iter
        burst_sz = x.shape[1]

        with torch.no_grad():
            if num_iter > 0:
                weights = self.weight_predictor({'x': x, 'offsets': offsets}, noise_estimate=noise_estimate)
            else:
                weights = None

        y_current = y_init.clone().detach()

        for i in range(num_iter):
            y_current = y_current.detach()

            with torch.enable_grad():
                y_current.requires_grad_(True)
                y = y_current.unsqueeze(1).expand(-1, burst_sz, -1, -1, -1)
                y_warped = self.warp_y(y, offsets)

                with torch.no_grad():
                    y_warped_ds = self.degradation_operator(y_warped)
                    data_residual = self.compute_residual(y_warped_ds, x, weights)
                    grad_y_warped = self.degradation_operator.apply_transposed(data_residual * weights)
                    del y_warped_ds, data_residual

                grad_y = torch.autograd.grad(
                    y_warped, y, grad_y_warped,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=True
                )[0]

            del y, y_warped, grad_y_warped

            with torch.no_grad():
                grad_y_current = grad_y.sum(dim=1)
                del grad_y

                h_warp = self.warp_y(grad_y_current.unsqueeze(1).expand(-1, burst_sz, -1, -1, -1), offsets)
                h = weights * self.degradation_operator(h_warp)
                del h_warp

                ip_gg = self._sqr_norm(grad_y_current)
                ip_hh = self._sqr_norm(h)
                del h

                alpha = ip_gg / (ip_hh).clamp(1e-8)
                step = grad_y_current * alpha.view(*[-1 if d == 0 else 1 for d in range(grad_y_current.dim())])
                del grad_y_current, alpha

                y_current = (y_current - step).detach()

        return y_current, []

    bw.backward_warp = OptimizedBackwardWarp.apply
    opt_mod.SteepestDescentOptimizer.forward = optimized_optimizer_forward
    print("deep-rep evaluation memory optimizations active.")
except Exception as e:
    print(f"Note: Could not patch optimizer modules: {e}")


def float_to_rational(val, max_denominator=1000000):
    """Converts a float to a (numerator, denominator) tuple for TIFF rational tags."""
    f = Fraction(val).limit_denominator(max_denominator)
    return (f.numerator, f.denominator)


def predict_tiled(net, burst_tensor, noise_estimate=None, target_sensor_scale=2.0, tile_size=192, overlap=32, use_fp16=True, device='cuda', progress_cb=None):
    """
    Performs memory-safe overlapping tiled inference over high-resolution burst tensors.
    burst_tensor: [B, T, C, H, W]
    noise_estimate: [B, 3, H, W] or None
    target_sensor_scale: 1.0 (1x sensor), 2.0 (2x sensor), or 4.0 (4x sensor)
    use_fp16: Enable selective half-precision execution on decoder
    """
    import gc
    B, T, C, H, W = burst_tensor.shape
    stride = tile_size - overlap
    is_cuda = (device == 'cuda' or (isinstance(device, torch.device) and device.type == 'cuda'))

    # Fast probe without heavy forward allocation
    if hasattr(net, 'hr_decoder') and hasattr(net.hr_decoder, 'upsample_layer'):
        raw_tile_scale_h = 8
        raw_tile_scale_w = 8
        out_c = 3
    else:
        probe_sz = min(tile_size, 32)
        probe_tile = burst_tensor[:, :, :, :probe_sz, :probe_sz].to(device)
        probe_noise = noise_estimate[:, :, :probe_sz, :probe_sz].to(device) if noise_estimate is not None else None
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(use_fp16 and is_cuda)):
                if probe_noise is not None:
                    probe_out, _ = net(probe_tile, noise_estimate=probe_noise)
                else:
                    probe_out, _ = net(probe_tile)
        raw_tile_scale_h = probe_out.shape[-2] // probe_tile.shape[-2]
        raw_tile_scale_w = probe_out.shape[-1] // probe_tile.shape[-1]
        out_c = probe_out.shape[1]
        del probe_tile, probe_noise, probe_out
        if is_cuda:
            torch.cuda.empty_cache()

    # For super-resolution (where raw_tile_scale is 8x from packed Bayer = 4x sensor resolution):
    if raw_tile_scale_h > 1:
        model_sensor_scale = raw_tile_scale_h / 2.0  # 8 / 2 = 4x
        downsample_factor = max(model_sensor_scale / target_sensor_scale, 1.0)
        effective_scale_h = int(raw_tile_scale_h / downsample_factor)
        effective_scale_w = int(raw_tile_scale_w / downsample_factor)
    else:
        downsample_factor = 1.0
        effective_scale_h = 1
        effective_scale_w = 1

    out_H = int(H * effective_scale_h)
    out_W = int(W * effective_scale_w)
    out_tile_sz_h = int(tile_size * effective_scale_h)
    out_tile_sz_w = int(tile_size * effective_scale_w)

    output_accum = torch.zeros((B, out_c, out_H, out_W), dtype=torch.float32)
    weight_accum = torch.zeros((B, 1, out_H, out_W), dtype=torch.float32)

    # 2D Hann window for smooth cosine blending across tile overlaps
    win_h = torch.hann_window(out_tile_sz_h, periodic=False)
    win_w = torch.hann_window(out_tile_sz_w, periodic=False)
    window_2d_cpu = (win_h.unsqueeze(1) * win_w.unsqueeze(0)).unsqueeze(0).unsqueeze(0) + 1e-4
    window_2d_gpu = window_2d_cpu.to(device) if is_cuda else window_2d_cpu

    # Generate grid coordinates
    h_starts = list(range(0, H - tile_size + 1, stride))
    if len(h_starts) == 0 or h_starts[-1] != H - tile_size:
        h_starts.append(max(0, H - tile_size))

    w_starts = list(range(0, W - tile_size + 1, stride))
    if len(w_starts) == 0 or w_starts[-1] != W - tile_size:
        w_starts.append(max(0, W - tile_size))

    total_tiles = len(h_starts) * len(w_starts)
    tile_idx = 0

    for hs in h_starts:
        for ws in w_starts:
            he = hs + tile_size
            we = ws + tile_size

            tile_in = burst_tensor[:, :, :, hs:he, ws:we].to(device, non_blocking=True)
            tile_noise = noise_estimate[:, :, hs:he, ws:we].to(device, non_blocking=True) if noise_estimate is not None else None

            with torch.no_grad():
                # Selective execution: alignment in FP32, decoder in FP16 if requested
                if use_fp16 and is_cuda:
                    with torch.amp.autocast('cuda', enabled=False):
                        if getattr(net, 'alignment_net', None) is not None:
                            offsets, offsets_aux = net.alignment_net(tile_in.float())
                        else:
                            offsets, offsets_aux = None, {}

                        if getattr(net, 'use_noise_estimate', False) and tile_noise is not None:
                            T_in = tile_in.shape[1]
                            tile_noise_repeated = tile_noise.float().unsqueeze(1).repeat(1, T_in, 1, 1, 1)
                            enc_in = torch.cat((tile_in.float(), tile_noise_repeated), dim=2)
                        else:
                            tile_noise_repeated = None
                            enc_in = tile_in.float()

                        lr_enc = net.lr_encoder(enc_in)
                        del enc_in
                        y_init = net.hr_initializer(lr_enc['enc'])
                        y_optim, _ = net.optimizer(
                            y_init=y_init,
                            x=lr_enc['enc'],
                            offsets=offsets.float() if offsets is not None else None,
                            num_iter=None,
                            noise_estimate=tile_noise_repeated
                        )
                        del y_init, offsets, tile_noise_repeated
                        lr_enc['fused_enc'] = y_optim

                    with torch.amp.autocast('cuda', enabled=True):
                        pred_dict = net.hr_decoder(lr_enc)
                        tile_out = pred_dict['pred'].float()
                    del lr_enc, pred_dict
                else:
                    if tile_noise is not None:
                        tile_out, _ = net(tile_in, noise_estimate=tile_noise)
                    else:
                        tile_out, _ = net(tile_in)

                # Downsample on GPU immediately before transferring to host memory
                if downsample_factor > 1.0:
                    tile_out = F.interpolate(
                        tile_out,
                        size=(out_tile_sz_h, out_tile_sz_w),
                        mode='area'
                    )

                # Window tile on GPU before transfer to Host CPU
                tile_out.mul_(window_2d_gpu)

            tile_out_cpu = tile_out.cpu()
            del tile_in, tile_noise, tile_out

            out_hs = int(hs * effective_scale_h)
            out_he = out_hs + out_tile_sz_h
            out_ws = int(ws * effective_scale_w)
            out_we = out_ws + out_tile_sz_w

            output_accum[:, :, out_hs:out_he, out_ws:out_we].add_(tile_out_cpu)
            weight_accum[:, :, out_hs:out_he, out_ws:out_we].add_(window_2d_cpu)
            del tile_out_cpu

            # Periodic cache cleanup only if reserved memory nears limit
            if is_cuda and (tile_idx % 16 == 0):
                if torch.cuda.memory_reserved() > 0.80 * torch.cuda.get_device_properties(device).total_memory:
                    torch.cuda.empty_cache()

            tile_idx += 1
            if progress_cb:
                progress_cb(tile_idx, total_tiles)

    # In-place normalized accumulation
    weight_accum.clamp_(min=1e-5)
    output_accum.div_(weight_accum)
    del weight_accum
    gc.collect()

    return output_accum


def save_linear_dng(out_path, img_rgb_data, src_path, proc_info=None):
    """
    Saves a restored 3-channel linear image as an Adobe Linear RAW DNG (PhotometricInterpretation=34892)
    preserving camera matrices, white balance, EXIF shooting parameters (time, camera model, shutter speed,
    ISO, aperture, lens info, orientation, etc.), and adding provenance UserComment metadata.
    Supports both native DNGs and Canon CR2 (and other non-DNG RAWs) by synthesizing missing ColorMatrix
    and AsShotNeutral tags directly from rawpy.
    """
    H, W, C = img_rgb_data.shape
    assert C == 3, "Image must have 3 channels (RGB)"

    if img_rgb_data.dtype == np.uint16:
        img_uint16 = img_rgb_data
    else:
        img_uint16 = np.clip(img_rgb_data * 65535.0, 0, 65535).astype(np.uint16)

    EXCLUDE_TAGS = {
        # Layout & structural TIFF tags
        254, 255, 256, 257, 258, 259, 262, 266, 273, 277, 278, 279, 284, 317,
        322, 323, 324, 325, 330, 338, 339, 400, 32997, 32998, 34665, 34853, 40965,

        # CFA / Bayer pattern tags - MUST NEVER be present in full 3-channel Linear DNG
        33421, 33422, 41730, 50710, 50711,

        # Sensor Black/White levels - data is normalized [0.0, 1.0] and scaled to uint16
        50712, 50713, 50714, 50715, 50716, 50717,

        # Sensor geometry / crop rectangles (invalid for upscaled or restored images)
        50719, 50720, 50742, 50829,

        # Noise baseline - noise characteristics changed after multi-frame neural restoration
        50731,

        # DNG versions - explicitly specified as standard 1.4.0.0 / 1.1.0.0
        50706, 50707,

        # Tags managed directly by TiffWriter parameters
        270, 305, 37510,
    }

    EXIF_TAG_MAP = {
        'ExposureTime': (33434, '2I'),
        'FNumber': (33437, '2I'),
        'ExposureProgram': (34850, 'H'),
        'SpectralSensitivity': (34852, 's'),
        'ISOSpeedRatings': (34855, 'H'),
        'PhotographicSensitivity': (34855, 'H'),
        'SensitivityType': (34864, 'H'),
        'StandardOutputSensitivity': (34865, 'I'),
        'RecommendedExposureIndex': (34866, 'I'),
        'ISOSpeed': (34867, 'I'),
        'ExifVersion': (36864, 's'),
        'DateTimeOriginal': (36867, 's'),
        'DateTimeDigitized': (36868, 's'),
        'OffsetTime': (36880, 's'),
        'OffsetTimeOriginal': (36881, 's'),
        'OffsetTimeDigitized': (36882, 's'),
        'ShutterSpeedValue': (37377, '2i'),
        'ApertureValue': (37378, '2I'),
        'BrightnessValue': (37379, '2i'),
        'ExposureBiasValue': (37380, '2i'),
        'MaxApertureValue': (37381, '2I'),
        'SubjectDistance': (37382, '2I'),
        'MeteringMode': (37383, 'H'),
        'LightSource': (37384, 'H'),
        'Flash': (37385, 'H'),
        'FocalLength': (37386, '2I'),
        'SubjectArea': (37396, 'H'),
        'SubsecTime': (37520, 's'),
        'SubsecTimeOriginal': (37521, 's'),
        'SubsecTimeDigitized': (37522, 's'),
        'FlashpixVersion': (40960, 's'),
        'ColorSpace': (40961, 'H'),
        'FlashEnergy': (41483, '2I'),
        'SpatialFrequencyResponse': (41484, 'B'),
        'FocalPlaneXResolution': (41486, '2I'),
        'FocalPlaneYResolution': (41487, '2I'),
        'FocalPlaneResolutionUnit': (41488, 'H'),
        'SubjectLocation': (41492, 'H'),
        'ExposureIndex': (41493, '2I'),
        'SensingMethod': (41495, 'H'),
        'FileSource': (41728, 'B'),
        'SceneType': (41729, 'B'),
        'CustomRendered': (41985, 'H'),
        'ExposureMode': (41986, 'H'),
        'WhiteBalance': (41987, 'H'),
        'DigitalZoomRatio': (41988, '2I'),
        'FocalLengthIn35mmFilm': (41989, 'H'),
        'SceneCaptureType': (41990, 'H'),
        'GainControl': (41991, 'H'),
        'Contrast': (41992, 'H'),
        'Saturation': (41993, 'H'),
        'Sharpness': (41994, 'H'),
        'DeviceSettingDescription': (41995, 'B'),
        'SubjectDistanceRange': (41996, 'H'),
        'ImageUniqueID': (42016, 's'),
        'CameraOwnerName': (42032, 's'),
        'BodySerialNumber': (42033, 's'),
        'LensSpecification': (42034, '2I'),
        'LensMake': (42035, 's'),
        'LensModel': (42036, 's'),
        'LensSerialNumber': (42037, 's'),
    }

    extratags = []
    copied_tag_codes = set()

    def pack_exif_value(tag_code, default_dtype, val):
        if isinstance(val, (tuple, list)):
            if len(val) == 2 and isinstance(val[0], int) and isinstance(val[1], int):
                dtype = '2i' if (val[0] < 0 or 'i' in default_dtype) else '2I'
                return (tag_code, dtype, 1, tuple(val), False)
            elif len(val) > 2 and all(isinstance(x, int) for x in val) and len(val) % 2 == 0:
                dtype = '2i' if any(x < 0 for x in val) or 'i' in default_dtype else '2I'
                return (tag_code, dtype, len(val) // 2, tuple(val), False)
            elif all(isinstance(x, int) for x in val):
                dtype = default_dtype
                return (tag_code, dtype, len(val), tuple(val), False)
        elif isinstance(val, int):
            dtype = default_dtype if default_dtype in ('H', 'I', 'h', 'i', 'B', 'b') else 'H'
            return (tag_code, dtype, 1, val, False)
        elif isinstance(val, str):
            return (tag_code, 's', len(val) + 1, val, False)
        elif isinstance(val, bytes):
            return (tag_code, 'B', len(val), list(val), False)
        return None

    model_name = None
    try:
        with tifffile.TiffFile(src_path) as tif:
            src_page = tif.pages[0]

            # 1. Copy top-level TIFF/DNG tags
            for code, tag in src_page.tags.items():
                if code not in EXCLUDE_TAGS and code not in copied_tag_codes:
                    extratags.append((tag.code, tag.dtype, tag.count, tag.value, False))
                    copied_tag_codes.add(code)
                if code == 272:  # Model
                    model_name = tag.value

            # 2. Extract shooting EXIF metadata from ExifTag sub-IFD (tag 34665)
            if 34665 in src_page.tags:
                exif_tag_data = src_page.tags[34665].value
                if isinstance(exif_tag_data, dict):
                    for name, val in exif_tag_data.items():
                        if name in EXIF_TAG_MAP:
                            t_code, t_dtype = EXIF_TAG_MAP[name]
                            if t_code not in copied_tag_codes and t_code not in EXCLUDE_TAGS:
                                packed = pack_exif_value(t_code, t_dtype, val)
                                if packed is not None:
                                    extratags.append(packed)
                                    copied_tag_codes.add(t_code)

    except Exception as e:
        print(f"Warning reading tags from source ({src_path}): {e}")

    # 3. For non-DNG files (e.g. CR2) lacking DNG tags, synthesize ColorMatrix & AsShotNeutral from rawpy
    if 50721 not in copied_tag_codes and 50722 not in copied_tag_codes:
        try:
            with rawpy.imread(src_path) as raw:
                mat = raw.rgb_xyz_matrix[:3, :3]
                wb = raw.camera_whitebalance

                # ColorMatrix2 (tag 50722, SRATIONAL '2i', count 9)
                cm_rationals = []
                for row in mat:
                    for val in row:
                        num, den = float_to_rational(float(val), max_denominator=10000)
                        cm_rationals.extend([num, den])
                extratags.append((50722, '2i', 9, tuple(cm_rationals), False))
                copied_tag_codes.add(50722)

                # CalibrationIlluminant2 = 21 (D65 Daylight)
                extratags.append((50779, 'H', 1, 21, False))
                copied_tag_codes.add(50779)

                # AsShotNeutral (tag 50728, RATIONAL '2I', count 3)
                if wb and len(wb) >= 3 and wb[1] > 0:
                    asn_rationals = []
                    for c in range(3):
                        val = 1.0 / (wb[c] / wb[1])
                        num, den = float_to_rational(val, max_denominator=10000)
                        asn_rationals.extend([num, den])
                    extratags.append((50728, '2I', 3, tuple(asn_rationals), False))
                    copied_tag_codes.add(50728)

                # UniqueCameraModel (tag 50708)
                if model_name and 50708 not in copied_tag_codes:
                    extratags.append((50708, 's', len(model_name) + 1, model_name, False))
                    copied_tag_codes.add(50708)
        except Exception as e:
            print(f"Warning synthesizing color matrix for {src_path}: {e}")

    # 4. Standard DNG specification tags for 3-channel Linear RAW
    extratags.append((50706, 'B', 4, [1, 4, 0, 0], False))  # DNGVersion 1.4.0.0
    extratags.append((50707, 'B', 4, [1, 1, 0, 0], False))  # DNGBackwardVersion 1.1.0.0

    # 5. Construct user comment documenting processing provenance
    if proc_info:
        extra_count = len(proc_info.get('extra_frames', []))
        extra_desc = f"{extra_count} frame{'s' if extra_count != 1 else ''}: " + ", ".join(proc_info.get('extra_frames', [])) if extra_count > 0 else "None"
        user_comment_text = (
            f"Deep-Rep Multi-Frame Burst Restoration\n"
            f"Keyframe: {proc_info.get('keyframe', os.path.basename(src_path))}\n"
            f"Extra Frames ({extra_desc})\n"
            f"Model: {proc_info.get('model', 'Unknown')} (Module: {proc_info.get('module', 'Unknown')})\n"
            f"Tiling: Tile Size {proc_info.get('tile_size', 'Unknown')}x{proc_info.get('tile_size', 'Unknown')} px "
            f"(Overlap: {proc_info.get('overlap', 'Unknown')} px)\n"
            f"Precision: {proc_info.get('precision', 'Unknown')}\n"
            f"Target Scale: {proc_info.get('sr_scale', '1.0x')}"
        )
    else:
        user_comment_text = f"Processed with Deep-Rep neural burst restoration from {os.path.basename(src_path)}"

    user_comment_bytes = b"ASCII\x00\x00\x00" + user_comment_text.encode("utf-8")
    extratags.append((37510, 'B', len(user_comment_bytes), list(user_comment_bytes), False))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with tifffile.TiffWriter(out_path) as writer:
        writer.write(
            img_uint16,
            photometric='linear_raw',
            description=user_comment_text,
            software="Deep-Rep Toolkit (goutamgmb/deep-rep)",
            metadata=None,
            extratags=extratags
        )
    return out_path


class DeepRepGUI:
    def __init__(self, root=None):
        self.root = root
        if self.root:
            self.root.title("Deep-Rep Denoising and Restoration Toolkit")
            self.root.geometry("780x740")

        # Variables
        self.keyframe_path = tk.StringVar() if root else None
        self.extra_frames_paths = []
        self.output_dir = tk.StringVar(value=os.getcwd()) if root else os.getcwd()
        self.use_gpu = tk.BooleanVar(value=torch.cuda.is_available()) if root else torch.cuda.is_available()
        self.use_fp16 = tk.BooleanVar(value=True) if root else True
        self.tile_size_var = tk.StringVar(value="192 (192x192 px - Low VRAM, Recommended)") if root else "192"
        self.sr_scale_var = tk.StringVar(value="2.0x (2× Sensor Grid - 4× Pixels, Balanced)") if root else "2.0x"
        self.cached_keyframe_info = {}

        # Fallback variables if running headless without full Tkinter root
        if not root:
            self._keyframe_val = ""
            self._extra_frames_val = []
            self._output_dir_val = os.getcwd()
            self._use_gpu_val = torch.cuda.is_available()
            self._use_fp16_val = True
            self._tile_size_val = "192"
            self._sr_scale_val = "2.0x"

        # Map modules and names directly to the .pth files in pretrained_networks folder
        self.model_mapping = {
            "burst_denoise": {
                "deeprep_color": "deeprep_denoise_color_pwcnet.pth"
            },
            "burstsr": {
                "deeprep_default": "deeprep_sr_synthetic_default.pth"
            },
            "synburst": {
                "deeprep_default": "deeprep_sr_synthetic_default.pth"
            }
        }

        default_mod = list(self.model_mapping.keys())[0]
        self.selected_module = tk.StringVar(value=default_mod) if root else default_mod
        avail_names = list(self.model_mapping[default_mod].keys())
        self.selected_name = tk.StringVar(value=avail_names[0] if avail_names else "") if root else (avail_names[0] if avail_names else "")

        if self.root:
            self.build_ui()

    # Property helpers to seamlessly support both Tkinter StringVar and headless string values
    def get_keyframe(self):
        return self.keyframe_path.get() if self.keyframe_path else getattr(self, '_keyframe_val', '')

    def set_keyframe(self, path):
        if self.keyframe_path:
            self.keyframe_path.set(path)
        else:
            self._keyframe_val = path

    def get_output_dir(self):
        return self.output_dir.get() if isinstance(self.output_dir, tk.StringVar) else getattr(self, '_output_dir_val', os.getcwd())

    def set_output_dir(self, path):
        if isinstance(self.output_dir, tk.StringVar):
            self.output_dir.set(path)
        else:
            self._output_dir_val = path

    def get_module(self):
        return self.selected_module.get() if isinstance(self.selected_module, tk.StringVar) else str(self.selected_module)

    def set_module(self, mod):
        if isinstance(self.selected_module, tk.StringVar):
            self.selected_module.set(mod)
        else:
            self.selected_module = mod

    def get_name(self):
        return self.selected_name.get() if isinstance(self.selected_name, tk.StringVar) else str(self.selected_name)

    def set_name(self, name):
        if isinstance(self.selected_name, tk.StringVar):
            self.selected_name.set(name)
        else:
            self.selected_name = name

    def get_use_gpu(self):
        return self.use_gpu.get() if isinstance(self.use_gpu, tk.BooleanVar) else bool(getattr(self, '_use_gpu_val', True))

    def set_use_gpu(self, val):
        if isinstance(self.use_gpu, tk.BooleanVar):
            self.use_gpu.set(val)
        else:
            self._use_gpu_val = bool(val)

    def get_use_fp16(self):
        return self.use_fp16.get() if isinstance(self.use_fp16, tk.BooleanVar) else bool(getattr(self, '_use_fp16_val', True))

    def set_use_fp16(self, val):
        if isinstance(self.use_fp16, tk.BooleanVar):
            self.use_fp16.set(val)
        else:
            self._use_fp16_val = bool(val)

    def build_ui(self):
        main_frame = ttk.Frame(self.root, padding="15 15 15 15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Inputs ---
        ttk.Label(main_frame, text="Input Selection", font=("Arial", 11, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(0, 8))

        ttk.Label(main_frame, text="Keyframe (RAW: DNG / CR2):").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.keyframe_path, width=54).grid(row=1, column=1, padx=5, pady=3)
        ttk.Button(main_frame, text="Browse", command=self.browse_keyframe).grid(row=1, column=2, pady=3)

        ttk.Label(main_frame, text="Extra Frames (DNGs / CR2s):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.lbl_extra_frames = ttk.Label(main_frame, text="0 files selected")
        self.lbl_extra_frames.grid(row=2, column=1, sticky=tk.W, padx=5)
        ttk.Button(main_frame, text="Select Files", command=self.browse_extra_frames).grid(row=2, column=2, pady=3)

        # --- Settings ---
        ttk.Label(main_frame, text="Toolkit Options", font=("Arial", 11, "bold")).grid(row=3, column=0, sticky=tk.W, pady=(15, 8))

        # Cascading Module -> Name Dropdowns
        ttk.Label(main_frame, text="Target Module:").grid(row=4, column=0, sticky=tk.W, pady=3)
        self.module_dropdown = ttk.Combobox(main_frame, textvariable=self.selected_module, values=list(self.model_mapping.keys()), state="readonly", width=51)
        self.module_dropdown.grid(row=4, column=1, padx=5, pady=3)
        self.module_dropdown.bind("<<ComboboxSelected>>", self.on_settings_changed)

        ttk.Label(main_frame, text="Model Name:").grid(row=5, column=0, sticky=tk.W, pady=3)
        self.name_dropdown = ttk.Combobox(main_frame, textvariable=self.selected_name, state="readonly", width=51)
        self.name_dropdown.grid(row=5, column=1, padx=5, pady=3)
        self.update_names_dropdown()

        ttk.Label(main_frame, text="SR Scale (Output):").grid(row=6, column=0, sticky=tk.W, pady=3)
        self.sr_scale_combo = ttk.Combobox(
            main_frame,
            textvariable=self.sr_scale_var,
            values=[
                "2.0x (2× Sensor Grid - 4× Pixels, Balanced)",
                "1.0x (1× Sensor Grid - Native Sharp RGB)",
                "4.0x (4× Sensor Grid - 16× Pixels, High RAM)"
            ],
            state="readonly",
            width=51
        )
        self.sr_scale_combo.grid(row=6, column=1, padx=5, pady=3)
        self.sr_scale_combo.bind("<<ComboboxSelected>>", self.on_settings_changed)

        ttk.Label(main_frame, text="Tile Size (Pixels):").grid(row=7, column=0, sticky=tk.W, pady=3)
        self.tile_combo = ttk.Combobox(
            main_frame,
            textvariable=self.tile_size_var,
            values=[
                "128 (128x128 px - Lowest VRAM)",
                "192 (192x192 px - Low VRAM, Recommended)",
                "256 (256x256 px - Medium VRAM)",
                "384 (384x384 px - High VRAM)",
                "512 (512x512 px - Ultra-High VRAM)"
            ],
            state="readonly",
            width=51
        )
        self.tile_combo.grid(row=7, column=1, padx=5, pady=3)
        self.tile_combo.bind("<<ComboboxSelected>>", self.on_settings_changed)

        options_frame = ttk.Frame(main_frame)
        options_frame.grid(row=8, column=1, sticky=tk.W, pady=5)
        ttk.Checkbutton(options_frame, text="Use GPU (CUDA)", variable=self.use_gpu, command=self.update_estimation_info).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Checkbutton(options_frame, text="Half-Precision (FP16 Decoder)", variable=self.use_fp16, command=self.update_estimation_info).pack(side=tk.LEFT)

        # --- Output ---
        ttk.Label(main_frame, text="Output Directory:").grid(row=9, column=0, sticky=tk.W, pady=8)
        ttk.Entry(main_frame, textvariable=self.output_dir, width=54).grid(row=9, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output).grid(row=9, column=2)

        # --- Information & Resource Estimation Box ---
        info_frame = ttk.LabelFrame(main_frame, text="Output & Resource Estimation", padding="10 10 10 10")
        info_frame.grid(row=10, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(12, 10))

        self.lbl_output_info = ttk.Label(info_frame, text="Output Format: Adobe Linear RAW DNG (Full 3-channel RGB, 16-bit)", font=("Arial", 9, "bold"))
        self.lbl_output_info.pack(anchor=tk.W)

        self.lbl_resolution_info = ttk.Label(info_frame, text="Expected Resolution: Please select a keyframe RAW image", font=("Arial", 9))
        self.lbl_resolution_info.pack(anchor=tk.W, pady=(2, 2))

        self.lbl_memory_est = ttk.Label(info_frame, text="Estimated Peak VRAM: ~1.3 GB | Host RAM: ~2.2 GB", foreground="#006600", font=("Arial", 9, "bold"))
        self.lbl_memory_est.pack(anchor=tk.W, pady=(2, 0))

        # --- Action & Progress ---
        self.run_btn = ttk.Button(main_frame, text="Run Restoration / Denoising", command=self.start_processing)
        self.run_btn.grid(row=11, column=1, pady=15)

        self.prog_bar = ttk.Progressbar(main_frame, orient="horizontal", mode="determinate", length=420)
        self.prog_bar.grid(row=12, column=1, pady=(0, 5))

        self.status_var = tk.StringVar(value="Status: Ready")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var, font=("Arial", 9, "italic"), foreground="blue")
        self.status_label.grid(row=13, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))

    def parse_tile_size(self):
        val = self.tile_size_var.get() if isinstance(self.tile_size_var, tk.StringVar) else str(self.tile_size_var)
        if "128" in val:
            return 128
        elif "192" in val:
            return 192
        elif "256" in val:
            return 256
        elif "384" in val:
            return 384
        return 512

    def parse_sr_scale(self):
        val = self.sr_scale_var.get() if isinstance(self.sr_scale_var, tk.StringVar) else str(self.sr_scale_var)
        if "1.0x" in val:
            return 1.0
        elif "4.0x" in val:
            return 4.0
        return 2.0

    def update_names_dropdown(self, event=None):
        module = self.get_module()
        available_names = list(self.model_mapping.get(module, {}).keys())

        if hasattr(self, 'name_dropdown'):
            self.name_dropdown['values'] = available_names
        if available_names:
            self.set_name(available_names[0])
        else:
            self.set_name("")

    def on_settings_changed(self, event=None):
        module = self.get_module()
        if hasattr(self, 'sr_scale_combo'):
            if module == 'burst_denoise':
                self.sr_scale_combo.config(state="disabled")
            else:
                self.sr_scale_combo.config(state="readonly")
        self.update_names_dropdown()
        self.update_estimation_info()

    def browse_keyframe(self):
        # Support both DNG and CR2 formats
        filepath = filedialog.askopenfilename(filetypes=[
            ("All RAW files (*.dng, *.cr2)", "*.dng *.DNG *.cr2 *.CR2"),
            ("DNG files (*.dng)", "*.dng *.DNG"),
            ("Canon CR2 files (*.cr2)", "*.cr2 *.CR2"),
            ("All files", "*.*")
        ])
        if filepath:
            self.set_keyframe(filepath)
            self.read_keyframe_metadata(filepath)
            self.update_estimation_info()

    def browse_extra_frames(self):
        filepaths = filedialog.askopenfilenames(filetypes=[
            ("All RAW files (*.dng, *.cr2)", "*.dng *.DNG *.cr2 *.CR2"),
            ("DNG files (*.dng)", "*.dng *.DNG"),
            ("Canon CR2 files (*.cr2)", "*.cr2 *.CR2"),
            ("All files", "*.*")
        ])
        if filepaths:
            self.extra_frames_paths = list(filepaths)
            if hasattr(self, 'lbl_extra_frames'):
                self.lbl_extra_frames.config(text=f"{len(self.extra_frames_paths)} files selected")
            self.update_estimation_info()

    def browse_output(self):
        directory = filedialog.askdirectory()
        if directory:
            self.set_output_dir(directory)

    def read_keyframe_metadata(self, filepath):
        try:
            with rawpy.imread(filepath) as raw:
                H, W = raw.raw_image_visible.shape
                colors = raw.color_desc.decode("latin1")
                pat = raw.raw_pattern

                # Detect R position to support any Bayer mosaic (RGGB, GBRG, BGGR, GRBG)
                r_offset, c_offset = 0, 0
                for r in range(2):
                    for c in range(2):
                        if colors[pat[r, c]] == 'R':
                            r_offset, c_offset = r, c
                            break

                H_even = ((H - r_offset) // 2) * 2
                W_even = ((W - c_offset) // 2) * 2

                self.cached_keyframe_info = {
                    'height': H_even,
                    'width': W_even,
                    'orig_h': H,
                    'orig_w': W,
                    'r_offset': r_offset,
                    'c_offset': c_offset,
                    'make': getattr(raw, 'camera_make', 'Camera'),
                    'white_level': raw.white_level
                }
        except Exception as e:
            print(f"Error reading keyframe metadata: {e}")
            self.cached_keyframe_info = {}

    def update_estimation_info(self):
        if not self.root:
            return

        module = self.get_module()
        tile_sz = self.parse_tile_size()
        sr_scale = self.parse_sr_scale()
        total_frames = 1 + len(self.extra_frames_paths)

        kf = self.get_keyframe()
        if not self.cached_keyframe_info and kf and os.path.exists(kf):
            self.read_keyframe_metadata(kf)

        if self.cached_keyframe_info:
            raw_h = self.cached_keyframe_info['height']
            raw_w = self.cached_keyframe_info['width']
            packed_h = raw_h // 2
            packed_w = raw_w // 2

            if module == 'burst_denoise':
                out_h, out_w = packed_h, packed_w
                scale_desc = "1:1 Native Packed (Denoised Full RGB)"
                mp = (out_h * out_w) / 1e6
                self.lbl_resolution_info.config(
                    text=f"Expected Output: {out_h} × {out_w} px ({mp:.1f} MP) - {scale_desc}"
                )

                base_vram_map = {128: 0.45, 192: 0.55, 256: 0.70, 384: 1.05, 512: 1.55}
                per_frame_map = {128: 0.03, 192: 0.08, 256: 0.14, 384: 0.35, 512: 0.58}

                base_v = base_vram_map.get(tile_sz, 0.70)
                per_f = per_frame_map.get(tile_sz, 0.14)
                vram_est = base_v + max(0, total_frames - 2) * per_f

                burst_ram_gb = (total_frames * raw_h * raw_w * 4) / (1024**3)
                out_accum_gb = (16 * out_h * out_w) / (1024**3)
                dng_buf_gb = (6 * out_h * out_w) / (1024**3)
                ram_est = 0.85 + burst_ram_gb + max(out_accum_gb, dng_buf_gb) + 0.35

            else:  # burstsr or synburst
                out_h = int(raw_h * sr_scale)
                out_w = int(raw_w * sr_scale)
                scale_desc = f"{sr_scale:.1f}× Sensor Grid ({sr_scale**2:.0f}× Pixel Super-Resolution)"
                mp = (out_h * out_w) / 1e6
                self.lbl_resolution_info.config(
                    text=f"Expected Output: {out_h} × {out_w} px ({mp:.1f} MP) - {scale_desc}"
                )

                base_vram_map = {128: 0.85, 192: 1.75, 256: 3.15, 384: 6.85, 512: 12.20}
                per_frame_map = {128: 0.18, 192: 0.43, 256: 0.75, 384: 1.50, 512: 2.60}
                fp32_penalty_map = {128: 0.25, 192: 0.85, 256: 1.55, 384: 3.50, 512: 6.00}

                base_v = base_vram_map.get(tile_sz, 1.75)
                per_f = per_frame_map.get(tile_sz, 0.43)
                vram_est = base_v + max(0, total_frames - 2) * per_f
                if not self.get_use_fp16():
                    vram_est += fp32_penalty_map.get(tile_sz, 0.85)

                burst_ram_gb = (total_frames * raw_h * raw_w * 4) / (1024**3)
                out_accum_gb = (16 * out_h * out_w) / (1024**3)
                dng_buf_gb = (6 * out_h * out_w) / (1024**3)
                ram_est = 0.85 + burst_ram_gb + max(out_accum_gb, dng_buf_gb) + 0.45

            device_str = "GPU (CUDA)" if self.get_use_gpu() and torch.cuda.is_available() else "CPU"
            if device_str == "CPU":
                self.lbl_memory_est.config(
                    text=f"Device: CPU | Estimated Host RAM: ~{ram_est + vram_est:.1f} GB (Burst: {total_frames} frames)",
                    foreground="#003366"
                )
            else:
                if vram_est > 7.5:
                    status_color = "#cc0000"
                    vram_note = " [CRITICAL: >8 GB VRAM - Lower Tile Size!]"
                elif vram_est > 6.0:
                    status_color = "#cc6600"
                    vram_note = " [High VRAM]"
                else:
                    status_color = "#006600"
                    vram_note = ""

                self.lbl_memory_est.config(
                    text=f"Estimated Peak VRAM: ~{vram_est:.2f} GB (GPU){vram_note} | Host RAM: ~{ram_est:.1f} GB (Burst: {total_frames} frames)",
                    foreground=status_color
                )
        else:
            self.lbl_resolution_info.config(text="Expected Resolution: Please select a keyframe RAW image")
            self.lbl_memory_est.config(text="Estimated Peak VRAM: ~1.75 GB | Host RAM: ~2.4 GB")

    def show_error_dialog(self, short_msg, full_trace):
        if not self.root:
            print(f"[ERROR] {short_msg}\n{full_trace}")
            return
        err_win = tk.Toplevel(self.root)
        err_win.title("Processing Error")
        err_win.geometry("700x400")

        ttk.Label(err_win, text=short_msg, font=("Arial", 10, "bold"), foreground="red").pack(padx=10, pady=5, anchor=tk.W)

        text_area = tk.Text(err_win, wrap=tk.NONE)
        text_area.insert(tk.END, full_trace)
        text_area.config(state=tk.DISABLED, bg="#f4f4f4")

        y_scroll = ttk.Scrollbar(err_win, orient=tk.VERTICAL, command=text_area.yview)
        x_scroll = ttk.Scrollbar(err_win, orient=tk.HORIZONTAL, command=text_area.xview)
        text_area.config(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        text_area.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        ttk.Button(err_win, text="Close", command=err_win.destroy).pack(pady=10)

    def load_burst(self):
        """
        Loads and packs a burst of RAW images (.dng, .cr2, etc.) into a single contiguous
        4-channel tensor of shape [1, T, 4, H/2, W/2] with automatic Bayer alignment (RGGB).
        """
        import gc
        all_paths = [self.get_keyframe()] + self.extra_frames_paths
        T = len(all_paths)
        assert T >= 1, "At least one RAW file required."

        # Inspect keyframe Bayer mosaic and dimensions
        with rawpy.imread(all_paths[0]) as raw:
            H, W = raw.raw_image_visible.shape
            colors = raw.color_desc.decode("latin1")
            pat = raw.raw_pattern

            # Find 'R' coordinate to determine alignment offset
            r_offset, c_offset = 0, 0
            for r in range(2):
                for c in range(2):
                    if colors[pat[r, c]] == 'R':
                        r_offset, c_offset = r, c
                        break

            H_even = ((H - r_offset) // 2) * 2
            W_even = ((W - c_offset) // 2) * 2

            # Identify color channel indices matching RGGB packing
            idx_r = pat[r_offset, c_offset]
            idx_g1 = pat[r_offset, (c_offset + 1) % 2]
            idx_g2 = pat[(r_offset + 1) % 2, c_offset]
            idx_b = pat[(r_offset + 1) % 2, (c_offset + 1) % 2]

            bl_key = np.array(raw.black_level_per_channel, dtype=np.float32) if raw.black_level_per_channel else [0.0]*4
            wl_key = float(raw.white_level) if raw.white_level is not None else 16383.0

        burst_stack = torch.empty((1, T, 4, H_even // 2, W_even // 2), dtype=torch.float32)

        for i, path in enumerate(all_paths):
            with rawpy.imread(path) as raw:
                bl = np.array(raw.black_level_per_channel, dtype=np.float32) if raw.black_level_per_channel else bl_key
                wl = float(raw.white_level) if raw.white_level is not None else wl_key
                vis = raw.raw_image_visible[r_offset : r_offset + H_even, c_offset : c_offset + W_even]

                # Direct in-place packing into target tensor slice (RGGB normalized)
                burst_stack[0, i, 0] = torch.from_numpy(vis[0::2, 0::2].astype(np.float32)).sub_(bl[idx_r]).div_(max(wl - bl[idx_r], 1.0))
                burst_stack[0, i, 1] = torch.from_numpy(vis[0::2, 1::2].astype(np.float32)).sub_(bl[idx_g1]).div_(max(wl - bl[idx_g1], 1.0))
                burst_stack[0, i, 2] = torch.from_numpy(vis[1::2, 0::2].astype(np.float32)).sub_(bl[idx_g2]).div_(max(wl - bl[idx_g2], 1.0))
                burst_stack[0, i, 3] = torch.from_numpy(vis[1::2, 1::2].astype(np.float32)).sub_(bl[idx_b]).div_(max(wl - bl[idx_b], 1.0))

        burst_stack.clamp_(0.0, 1.0)
        gc.collect()
        return burst_stack

    # Backwards compatibility alias
    def load_dng_burst(self):
        return self.load_burst()

    def process_burst(self):
        import gc
        try:
            self.set_status("Status: Loading Model...")
            self.update_progress(0, 100)
            device = torch.device("cuda" if self.get_use_gpu() and torch.cuda.is_available() else "cpu")

            module = self.get_module()
            name = self.get_name()
            filename = self.model_mapping[module][name]

            # Dynamic pretrained networks resolution with relative fallback
            checkpoint_path = None
            try:
                settings = ws_settings.Settings()
                candidate = os.path.join(settings.env.pretrained_nets_dir, filename)
                if os.path.exists(candidate):
                    checkpoint_path = candidate
            except Exception:
                pass

            if checkpoint_path is None or not os.path.exists(checkpoint_path):
                # Fallback to local pretrained_networks directory
                local_dir = os.path.join(current_dir, "pretrained_networks")
                checkpoint_path = os.path.join(local_dir, filename)

            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Weight file not found at: {checkpoint_path}")

            net, _ = load_network(network_dir=checkpoint_path)
            net = net.to(device)
            net.eval()

            self.set_status("Status: Loading & Packing RAW burst...")
            burst_tensor = self.load_burst()

            # Determine input channel requirements based on the network's lr_encoder
            in_channels = getattr(net.lr_encoder.init_layer[0], 'in_channels', 4)
            use_noise_estimate = getattr(net, 'use_noise_estimate', False)

            if in_channels == 6 or in_channels == 3:
                # Convert 4-channel Bayer to 3-channel RGB: (R, (G1+G2)/2, B)
                burst_tensor_in = torch.stack(
                    (burst_tensor[:, :, 0], burst_tensor[:, :, 1:3].mean(dim=2), burst_tensor[:, :, 3]), dim=2
                )
                del burst_tensor
                gc.collect()
            else:
                burst_tensor_in = burst_tensor

            B, T, C, H, W = burst_tensor_in.shape

            if use_noise_estimate:
                noise_est = torch.zeros((B, 3, H, W), dtype=torch.float32)
            else:
                noise_est = None

            tile_sz = self.parse_tile_size()
            sr_scale = self.parse_sr_scale() if module != 'burst_denoise' else 1.0
            overlap = min(32, tile_sz // 4)
            use_fp16 = self.get_use_fp16()

            def on_tile_progress(cur, total):
                pct = int((cur / total) * 100)
                self.update_progress(pct, 100)
                self.set_status(f"Status: Processing Tile {cur}/{total} ({pct}%)...")

            self.set_status(f"Status: Starting Tiled Inference ({module})...")

            # Execute memory-safe overlapping tiled inference
            output = predict_tiled(
                net=net,
                burst_tensor=burst_tensor_in,
                noise_estimate=noise_est,
                target_sensor_scale=sr_scale,
                tile_size=tile_sz,
                overlap=overlap,
                use_fp16=use_fp16,
                device=device,
                progress_cb=on_tile_progress
            )

            self.set_status("Status: Converting and Saving Linear DNG...")

            # In-place conversion to uint16 directly to save Host RAM
            output.squeeze_(0)
            output = output.permute(1, 2, 0).contiguous()
            output.mul_(65535.0).clamp_(0.0, 65535.0).round_()
            output_uint16 = output.to(torch.uint16).numpy()

            del output, burst_tensor_in
            if 'burst_tensor' in locals():
                del burst_tensor
            gc.collect()

            out_dir = self.get_output_dir()
            os.makedirs(out_dir, exist_ok=True)
            keyframe_file = self.get_keyframe()
            base_name, in_ext = os.path.splitext(os.path.basename(keyframe_file))

            # Export is always Adobe Linear RAW DNG
            out_ext = ".DNG" if in_ext.isupper() else ".dng"
            out_file_dng = os.path.join(out_dir, f"{base_name}_restored{out_ext}")

            proc_info = {
                'keyframe': os.path.basename(keyframe_file),
                'extra_frames': [os.path.basename(p) for p in self.extra_frames_paths],
                'module': module,
                'model': filename,
                'tile_size': tile_sz,
                'overlap': overlap,
                'precision': "FP16 (decoder) / FP32 (alignment & recurrent optimizer)" if use_fp16 else "FP32",
                'sr_scale': f"{sr_scale:.1f}x" if module != 'burst_denoise' else "1.0x (denoised native)"
            }

            save_linear_dng(out_file_dng, output_uint16, keyframe_file, proc_info=proc_info)

            out_h, out_w, _ = output_uint16.shape
            del output_uint16
            gc.collect()

            self.set_status(f"Status: Success! Saved to {out_file_dng}")
            self.update_progress(100, 100)

            msg = (
                f"Restoration Complete!\n\n"
                f"Saved File: {out_file_dng}\n"
                f"Format: Adobe Linear RAW DNG (Photometric=34892)\n"
                f"Color Space: Native Linear Camera RGB (Full 3-Channel, 16-bit)\n"
                f"Resolution: {out_h} × {out_w} pixels ({out_h * out_w / 1e6:.1f} MP)\n\n"
                f"Ready for import and editing in Darktable, Lightroom, or RawTherapee."
            )
            if self.root:
                self.root.after(0, lambda: messagebox.showinfo("Processing Complete", msg))
            else:
                print(f"[SUCCESS] {msg}")

        except Exception as e:
            self.set_status("Status: Error occurred")
            err_trace = traceback.format_exc()
            if self.root:
                self.root.after(0, lambda e=e, trace=err_trace: self.show_error_dialog(str(e), trace))
            else:
                print(f"[ERROR] {e}\n{err_trace}")
            raise e
        finally:
            if hasattr(self, 'run_btn'):
                self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))

    def set_status(self, text):
        if hasattr(self, 'status_var') and self.root:
            self.root.after(0, lambda: self.status_var.set(text))
        else:
            print(text)

    def update_progress(self, current, maximum):
        if hasattr(self, 'prog_bar') and self.root:
            self.root.after(0, lambda: self.prog_bar.config(value=current, maximum=maximum))

    def start_processing(self):
        if not self.get_keyframe():
            if self.root:
                messagebox.showwarning("Missing Input", "Please select a Keyframe RAW image.")
            else:
                print("Missing Input: Please select a Keyframe RAW image.")
            return

        if hasattr(self, 'run_btn'):
            self.run_btn.config(state=tk.DISABLED)
        self.set_status("Status: Preparing to process...")
        self.update_progress(0, 100)

        threading.Thread(target=self.process_burst, daemon=True).start()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Deep-Rep RAW Burst Restoration and Denoising Toolkit")
    parser.add_argument("--keyframe", "-k", type=str, default=None, help="Path to Keyframe RAW image (.dng, .cr2, etc.)")
    parser.add_argument("--extra-frames", "-e", nargs="*", default=[], help="Paths to Extra RAW frames")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Output directory for restored Linear DNG")
    parser.add_argument("--module", "-m", choices=["burstsr", "burst_denoise", "synburst"], default=None, help="Deep-Rep module")
    parser.add_argument("--model-name", type=str, default=None, help="Specific model checkpoint name")
    parser.add_argument("--sr-scale", type=float, choices=[1.0, 2.0, 4.0], default=None, help="Super-resolution sensor scale")
    parser.add_argument("--tile-size", type=int, choices=[128, 192, 256, 384, 512], default=None, help="Tile size in pixels")
    parser.add_argument("--fp16", action="store_true", default=None, help="Enable FP16 decoder acceleration")
    parser.add_argument("--no-fp16", action="store_false", dest="fp16", help="Disable FP16 (pure FP32)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference instead of CUDA")
    parser.add_argument("--headless", "--batch", action="store_true", help="Execute processing headlessly without opening Tkinter GUI")
    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = parse_arguments()

    if args.headless:
        # Headless execution mode (for Darktable batch processing or CLI scripts)
        app = DeepRepGUI(root=None)
        if args.keyframe:
            app.set_keyframe(os.path.abspath(args.keyframe))
        if args.extra_frames:
            app.extra_frames_paths = [os.path.abspath(f) for f in args.extra_frames]
        if args.output_dir:
            app.set_output_dir(os.path.abspath(args.output_dir))
        elif args.keyframe:
            app.set_output_dir(os.path.dirname(os.path.abspath(args.keyframe)))
        if args.module:
            app.set_module(args.module)
            app.update_names_dropdown()
        if args.model_name:
            app.set_name(args.model_name)
        if args.sr_scale:
            app.sr_scale_var = f"{args.sr_scale:.1f}x"
        if args.tile_size:
            app.tile_size_var = str(args.tile_size)
        if args.fp16 is not None:
            app.set_use_fp16(args.fp16)
        if args.cpu:
            app.set_use_gpu(False)

        print("=== Deep-Rep Headless Mode ===")
        print(f"Keyframe: {app.get_keyframe()}")
        print(f"Extra Frames: {len(app.extra_frames_paths)} files")
        print(f"Module: {app.get_module()} | Model: {app.get_name()}")
        print(f"Tile Size: {app.parse_tile_size()} | SR Scale: {app.parse_sr_scale()}x | FP16: {app.get_use_fp16()}")

        try:
            app.process_burst()
            print("Processing finished successfully.")
            sys.exit(0)
        except Exception as e:
            print(f"Fatal error during processing: {e}")
            sys.exit(1)

    else:
        # Graphical GUI mode (Tkinter)
        root = tk.Tk()
        app = DeepRepGUI(root)

        if args.keyframe:
            app.set_keyframe(os.path.abspath(args.keyframe))
            app.read_keyframe_metadata(args.keyframe)
        if args.extra_frames:
            app.extra_frames_paths = [os.path.abspath(f) for f in args.extra_frames]
            if hasattr(app, 'lbl_extra_frames'):
                app.lbl_extra_frames.config(text=f"{len(app.extra_frames_paths)} files selected")
        if args.output_dir:
            app.set_output_dir(os.path.abspath(args.output_dir))
        if args.module:
            app.set_module(args.module)
            app.update_names_dropdown()
        if args.model_name:
            app.set_name(args.model_name)
        if args.sr_scale:
            for val in app.sr_scale_combo['values']:
                if f"{args.sr_scale:.1f}x" in val:
                    app.sr_scale_var.set(val)
                    break
        if args.tile_size:
            for val in app.tile_combo['values']:
                if str(args.tile_size) in val:
                    app.tile_size_var.set(val)
                    break
        if args.fp16 is not None:
            app.set_use_fp16(args.fp16)
        if args.cpu:
            app.set_use_gpu(False)

        app.update_estimation_info()
        root.mainloop()
