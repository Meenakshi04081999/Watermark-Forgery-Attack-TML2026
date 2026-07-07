import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import requests
from PIL import Image

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_DIR   = Path("/home/atml_team032/watermarking")
TEMP_OUT_DIR  = Path("/home/atml_team032/watermarking/submission_temp")
FILE_PATH     = "/home/atml_team032/watermarking/submission.zip"

HIGHPASS_RADIUS = 3          # NumPy box-blur radius for high-pass; set to 0 to disable

# Per-category forgery strength. Start uniform, then tune the weak ones up/down.
ALPHA = {
    "WM_1": 2.0, "WM_2": 2.0, "WM_3": 2.0, "WM_4": 2.0,
    "WM_5": 2.0, "WM_6": 2.0, "WM_7": 2.0, "WM_8": 2.0,
}

# Leaderboard submission
BASE_URL = "http://34.63.153.158"
API_KEY  = "c624f3b8d663751fcf05c23893ab116a"         # REPLACE WITH YOUR API KEY
TASK_ID  = "22-forging-task"
SUBMIT   = True

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

target_dir = DATASET_DIR / "clean_targets"

# ── Pure-NumPy box blur (replaces scipy.ndimage.gaussian_filter) ──────────────
def box_blur(img, radius=3):
    """Separable box blur via cumulative sums. img: HxWx3 float array."""
    if radius <= 0:
        return img
    k = 2 * radius + 1
    pad = np.pad(img, ((radius + 1, radius), (radius + 1, radius), (0, 0)), mode="edge")
    cs = np.cumsum(pad, axis=0)
    blurred = (cs[k:, :, :] - cs[:-k, :, :]) / k
    cs = np.cumsum(blurred, axis=1)
    blurred = (cs[:, k:, :] - cs[:, :-k, :]) / k
    return blurred

# ── Step 1: Dataset assumed extracted ─────────────────────────────────────────
print("Dataset already extracted.")
TEMP_OUT_DIR.mkdir(exist_ok=True)
total_processed = 0

# ── Global clean mean over ALL 200 clean images (computed once) ───────────────
print("Computing global clean mean over all 200 targets...")
all_clean = []
for number in range(1, 201):
    arr = np.array(Image.open(target_dir / f"{number}.png").convert("RGB")).astype(np.float32)
    all_clean.append(arr)
mean_clean_global = np.mean(all_clean, axis=0)

# ── Step 2: Median + global-mean + high-pass, per-category ALPHA ──────────────
print("\nBuilding forgery submission...")

for source_wm, target_start, target_stop in CATEGORIES:
    print(f"Processing {source_wm} → images {target_start}.png to {target_stop}.png ...")

    source_dir    = DATASET_DIR / "watermarked_sources" / source_wm
    source_images = sorted(source_dir.glob("*.png"))
    if not source_images:
        print(f"  [Warning] No source images found in {source_dir}")
        continue

    # MEDIAN of watermarked sources (robust to unusual-content images)
    wm_arrays = [np.array(Image.open(p).convert("RGB")).astype(np.float32) for p in source_images]
    median_wm = np.median(wm_arrays, axis=0)

    # Residual vs GLOBAL clean mean, then high-pass to drop low-freq content bias
    watermark_residual = median_wm - mean_clean_global
    if HIGHPASS_RADIUS:
        watermark_residual = watermark_residual - box_blur(watermark_residual, HIGHPASS_RADIUS)

    a = ALPHA[source_wm]
    print(f"  alpha = {a}, residual std = {watermark_residual.std():.3f}")

    for number in range(target_start, target_stop + 1):
        target_path = target_dir / f"{number}.png"
        target_arr  = np.array(Image.open(target_path).convert("RGB")).astype(np.float32)

        forged = np.clip(target_arr + a * watermark_residual, 0, 255).astype(np.uint8)
        Image.fromarray(forged).save(TEMP_OUT_DIR / target_path.name)
        total_processed += 1

    print("Done.")

print(f"\nSuccessfully forged {total_processed} images.")
if total_processed != 200:
    print(f"[WARNING] Expected 200, got {total_processed}. Submission may be rejected!")

# ── Step 3: Package into flat zip ─────────────────────────────────────────────
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