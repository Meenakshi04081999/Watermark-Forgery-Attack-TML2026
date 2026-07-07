import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import requests
from PIL import Image

# CONFIG
ZIP_FILE = "Dataset.zip"  # Path to the downloaded dataset zip
DATASET_DIR = Path(".")
print("Current working dir:", os.getcwd())
print("Contents:", os.listdir("."))
print("WM_1 exists:", os.path.exists("watermarked_sources/WM_1"))
if os.path.exists("watermarked_sources/WM_1"):
    print("WM_1 contents:", os.listdir("watermarked_sources/WM_1"))# Unzipped folder
TEMP_OUT_DIR = Path("submission_temp")  # Temporary folder for forged images
FILE_PATH = "submission.zip"  # Final file to upload
ALPHA = 2.0

HIGHPASS_RADIUS = 3  # box-blur radius for high-pass; set to 0 to disable

# Leaderboard submission
BASE_URL  = "http://34.63.153.158"
API_KEY  = "c624f3b8d663751fcf05c23893ab116a"  # REPLACE WITH YOUR API KEY
TASK_ID   = "22-forging-task"
SUBMIT   = True  # Set to True to enable submission

CATEGORIES = [
    ("WM_1",  1,   25),
    ("WM_2",  26,  50),
    ("WM_3",  51,  75),
    ("WM_4",  76,  100),
    ("WM_5",  101, 125),
    ("WM_6",  126, 150),
    ("WM_7",  151, 175),
    ("WM_8",  176, 200),
]

# ── High-pass helper (pure NumPy, no scipy) ──────────────────────────────────
def box_blur(img, radius=3):
    """Separable box blur via cumulative sums. img: HxWx3 float array."""
    if radius <= 0:
        return img
    k = 2 * radius + 1
    pad = np.pad(img, ((radius + 1, radius), (radius + 1, radius), (0, 0)), mode="edge")
    cs = np.cumsum(pad, axis=0)
    out = (cs[k:, :, :] - cs[:-k, :, :]) / k
    cs = np.cumsum(out, axis=1)
    out = (cs[:, k:, :] - cs[:, :-k, :]) / k
    return out

# ── Step 1: Unzip dataset if needed ──────────────────────────────────────────
print("Dataset already extracted.")

 
TEMP_OUT_DIR.mkdir(exist_ok=True)
total_processed = 0
 
# ── Step 2: Average Residual Copy Attack ─────────────────────────────────────
print("\nBuilding forgery submission...")
 
for source_wm, target_start, target_stop in CATEGORIES:
    print(f"Processing {source_wm} → images {target_start}.png to {target_stop}.png ...")
 
    source_dir    = DATASET_DIR / "watermarked_sources" / source_wm
    source_images = sorted(source_dir.glob("*.png"))
    target_dir    = DATASET_DIR / "clean_targets"
 
    if not source_images:
        print(f"  [Warning] No source images found in {source_dir}")
        continue
 
    # Compute MEDIAN of watermarked source images (robust to unusual content)
    wm_arrays = []
    for p in source_images:
        arr = np.array(Image.open(p).convert("RGB")).astype(np.float32)
        wm_arrays.append(arr)
    median_wm = np.median(wm_arrays, axis=0)
 
    # Compute mean of clean target images for this batch
    clean_arrays = []
    for number in range(target_start, target_stop + 1):
        p = target_dir / f"{number}.png"
        arr = np.array(Image.open(p).convert("RGB")).astype(np.float32)
        clean_arrays.append(arr)
    mean_clean = np.mean(clean_arrays, axis=0)
 
    # Watermark residual = median(watermarked) - mean(clean)
    watermark_residual = median_wm - mean_clean
 
    # High-pass: remove low-frequency content bias, keep the watermark
    if HIGHPASS_RADIUS:
        watermark_residual = watermark_residual - box_blur(watermark_residual, HIGHPASS_RADIUS)
 
    # Apply watermark residual to each target image
    for number in range(target_start, target_stop + 1):
        target_path = target_dir / f"{number}.png"
        target_arr  = np.array(Image.open(target_path).convert("RGB")).astype(np.float32)
 
        # forged = clean_target + alpha * watermark_residual
        forged = target_arr + ALPHA * watermark_residual
        forged = np.clip(forged, 0, 255).astype(np.uint8)
 
        out_path = TEMP_OUT_DIR / target_path.name
        Image.fromarray(forged).save(out_path)
        total_processed += 1
 
    print(f"Done.")
 
print(f"\nSuccessfully forged {total_processed} images.")
if total_processed != 200:
    print(f"[WARNING] Expected 200, got {total_processed}. Submission may be rejected!")
 
# ── Step 3: Package into flat zip ────────────────────────────────────────────
print(f"Packaging into {FILE_PATH}...")
with zipfile.ZipFile(FILE_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for img_path in sorted(TEMP_OUT_DIR.glob("*.png")):
        zipf.write(img_path, arcname=img_path.name)
print(f"Saved submission to {FILE_PATH}")
 
# ── Step 4: Submit ────────────────────────────────────────────────────────────
if SUBMIT:
    if not os.path.isfile(FILE_PATH):
        print(f"File not found: {FILE_PATH}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(FILE_PATH, "rb") as f:
            files = {"file": (os.path.basename(FILE_PATH), f, "application/zip")}
            resp  = requests.post(
                f"{BASE_URL}/submit/{TASK_ID}",
                headers={"X-API-Key": API_KEY},
                files=files,
            )
        try:
            body = resp.json()
        except Exception:
            body = {"raw_text": resp.text}
 
        if resp.status_code == 413:
            print("Upload rejected: file too large (HTTP 413).", file=sys.stderr)
            sys.exit(1)
 
        resp.raise_for_status()
        print("Successfully submitted.")
        print("Server response:", body)
 
    except requests.exceptions.RequestException as e:
        detail = getattr(e, "response", None)
        print(f"Submission error: {e}")
        if detail is not None:
            try:    print("Server response:", detail.json())
            except: print("Server response:", detail.text)
        sys.exit(1)