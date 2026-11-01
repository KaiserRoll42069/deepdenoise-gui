#!/usr/bin/env python3
"""
Downloads pretrained model weights for Deep-Rep toolkit directly from the original authors' Google Drive repository.
Works cross-platform (Windows, Linux, macOS) using standard library (urllib) without external dependencies.

Original Deep-Rep Framework Copyright (c) 2021 Huawei Technologies Co., Ltd.
Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International)
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
Academic / non-commercial research use only.
"""

import os
import sys
import urllib.request
import urllib.parse
import re

MODELS = {
    "pwcnet-network-default.pth": "1s11Ud1UMipk2AbZZAypLPRpnXOS9Y1KO",
    "deeprep_sr_synthetic_default.pth": "1GZ0S4BKaZbgiy9WmoYH4kAsXY7zDR_ks",
    "deeprep_denoise_color_pwcnet.pth": "1wk8NqoLSAZYrhRdzvMXAyRJd4txta7QZ"
}

def download_file_from_google_drive(file_id, destination):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    response = opener.open(req)
    initial_bytes = response.read(65536)
    
    # Check if Google Drive returned an HTML warning form for large files
    if b"download-form" in initial_bytes or b"Virus scan warning" in initial_bytes:
        text = initial_bytes.decode('utf-8', errors='ignore')
        
        # Extract form action URL
        action_match = re.search(r'<form id="download-form" action="([^"]+)"', text)
        action_url = action_match.group(1) if action_match else "https://drive.usercontent.google.com/download"
        
        # Extract all hidden input parameters (id, export, confirm, uuid, etc.)
        params = {}
        for m in re.finditer(r'<input\s+[^>]*name="([^"]+)"\s+[^>]*value="([^"]*)"', text):
            params[m.group(1)] = m.group(2)
        for m in re.finditer(r'<input\s+[^>]*value="([^"]*)"\s+[^>]*name="([^"]+)"', text):
            params[m.group(2)] = m.group(1)
            
        if "id" not in params:
            params["id"] = file_id
        if "export" not in params:
            params["export"] = "download"
        if "confirm" not in params:
            params["confirm"] = "t"
            
        confirm_url = f"{action_url}?{urllib.parse.urlencode(params)}"
        req2 = urllib.request.Request(confirm_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        response = opener.open(req2)
        initial_bytes = None
    
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    temp_dest = destination + ".tmp"
    
    total_downloaded = 0
    with open(temp_dest, "wb") as f:
        if initial_bytes is not None:
            f.write(initial_bytes)
            total_downloaded += len(initial_bytes)
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            total_downloaded += len(chunk)
            mb = total_downloaded / (1024 * 1024)
            sys.stdout.write(f"\r  Downloaded: {mb:.1f} MB...")
            sys.stdout.flush()
            
    print(f"\r  Downloaded: {total_downloaded / (1024 * 1024):.1f} MB. Done.")
    if os.path.exists(destination):
        os.remove(destination)
    os.rename(temp_dest, destination)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(script_dir, "pretrained_networks")
    if len(sys.argv) > 1:
        target_dir = os.path.abspath(sys.argv[1])
        
    os.makedirs(target_dir, exist_ok=True)
    print(f"Target directory for pretrained weights: {target_dir}")
    
    for filename, file_id in MODELS.items():
        dest = os.path.join(target_dir, filename)
        # Check if valid file already exists (> 1 MB)
        if os.path.exists(dest) and os.path.getsize(dest) > 1024 * 1024:
            print(f"[OK] {filename} already exists ({os.path.getsize(dest)/(1024**2):.1f} MB). Skipping.")
            continue
            
        print(f"[DOWNLOADING] {filename} from Google Drive...")
        try:
            download_file_from_google_drive(file_id, dest)
            print(f"[SUCCESS] Saved {filename} ({os.path.getsize(dest)/(1024**2):.1f} MB)")
        except Exception as e:
            print(f"[ERROR] Failed to download {filename}: {e}")

if __name__ == "__main__":
    main()
