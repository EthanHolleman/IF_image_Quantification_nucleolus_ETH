import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nd2
from csbdeep.utils import normalize

# ---- FIX: ensure these names exist (your code uses filters/morphology/measure/color/skio) ----
import skimage.filters as filters
import skimage.morphology as morphology
import skimage.measure as measure
import skimage.color as color
import skimage.io as skio
from skimage.color import rgb2gray
# --------------------------------------------------------------------------------------------

# Local utilities (works whether called as package or script)
try:
    from .utils import (
        parent_by_mode, normalize01, _ensure_dir, _save_gray, _save_label_tiff,
        overlay_labels, flatfield_correct
    )
except ImportError:
    from utils import (
        parent_by_mode, normalize01, _ensure_dir, _save_gray, _save_label_tiff,
        overlay_labels, flatfield_correct
    )


def get_CoLoc(
    nd2_file,
    nuclei_labels_path,
    npm1_labels_path,
    DAPI_IDX=0,
    NPM1_IDX=2,
    NC_IDX=3,
    nucleolar_component="FBL",
    output_dir="CoLoc_Results",
    group_name=None,
    bg_sigma=6.0,
    bg_subtract=True,
    min_object_area=40,
    save_intermediates=True,
    verbose=False,
):
    """
    Colocalization / rim enrichment style workflow.

    Notes:
    - This function expects nuclei/nucleoli labels saved as PNGs (or readable by skimage.io.imread).
    - `filters`, `morphology`, `measure`, `color`, `skio` are used throughout; they are now imported above.
    """

    # ------------------- I/O + sanity -------------------
    if group_name is None:
        group_name = "group"

    _ensure_dir(output_dir)

    if not os.path.exists(nd2_file):
        raise FileNotFoundError(f"ND2 not found: {nd2_file}")

    if not os.path.exists(nuclei_labels_path):
        raise FileNotFoundError(f"Nuclei labels not found: {nuclei_labels_path}")

    if not os.path.exists(npm1_labels_path):
        raise FileNotFoundError(f"Nucleoli labels not found: {npm1_labels_path}")

    base = os.path.splitext(os.path.basename(nd2_file))[0]
    out_base_dir = os.path.join(output_dir, f"{group_name}_{base}")
    _ensure_dir(out_base_dir)

    # ------------------- load labels -------------------
    # NOTE: labels were saved as PNG; skio.imread returns (H,W) or (H,W,3/4)
    nuc_lab = skio.imread(nuclei_labels_path)
    npm1_lab = skio.imread(npm1_labels_path)

    # If they’re RGB PNGs, convert to single channel first
    if nuc_lab.ndim == 3:
        nuc_lab = rgb2gray(nuc_lab)
    if npm1_lab.ndim == 3:
        npm1_lab = rgb2gray(npm1_lab)

    # Convert labels to integers; if they were stored as grayscale masks, this keeps IDs stable if present.
    print(type(nuc_lab))
    print(type(npm1_lab))
    nuc_lab = nuc_lab.astype(np.int32)
    npm1_lab = npm1_lab.astype(np.int32)

    # If labels are binary masks, label them
    if nuc_lab.max() <= 1:
        nuc_lab = measure.label(nuc_lab > 0)
    if npm1_lab.max() <= 1:
        npm1_lab = measure.label(npm1_lab > 0)

    # Optional cleanup
    if min_object_area is not None and min_object_area > 0:
        nuc_lab = morphology.remove_small_objects(nuc_lab, min_size=min_object_area)
        npm1_lab = morphology.remove_small_objects(npm1_lab, min_size=min_object_area)

    # ------------------- load ND2 & extract channels -------------------
    with nd2.ND2File(nd2_file) as f:
        # ND2 can be (t, c, y, x) or (c, y, x) depending on acquisition
        arr = f.asarray()

    # Normalize dimensions to (C, Y, X)
    if arr.ndim == 4:
        # assume (T, C, Y, X) -> take first timepoint
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected ND2 array shape: {arr.shape} (expected 3D C,Y,X or 4D T,C,Y,X)")

    C, H, W = arr.shape

    # guard channel indices
    for name, idx in [("DAPI_IDX", DAPI_IDX), ("NPM1_IDX", NPM1_IDX), ("NC_IDX", NC_IDX)]:
        if idx < 0 or idx >= C:
            raise IndexError(f"{name}={idx} out of range for ND2 with C={C} channels")
    
    # Debug: Check what we're actually getting
    if verbose:
        print(f"Array shape: {arr.shape}, dtype: {arr.dtype}")
        print(f"Array type: {type(arr)}")
        print(f"DAPI_IDX={DAPI_IDX}, NPM1_IDX={NPM1_IDX}, NC_IDX={NC_IDX}")
    
    # Ensure we're working with numpy arrays
    # The nd2 library might return a special array type, so explicitly convert
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    
    # Extract channels - ensure they're numpy arrays
    dapi_raw = arr[DAPI_IDX]
    npm1_raw = arr[NPM1_IDX]
    nc_raw = arr[NC_IDX]
    
    if verbose:
        print(f"Extracted channel types: dapi={type(dapi_raw)}, npm1={type(npm1_raw)}, nc={type(nc_raw)}")
    
    # Convert to float32
    dapi = np.asarray(dapi_raw, dtype=np.float32)
    npm1 = np.asarray(npm1_raw, dtype=np.float32)
    nc = np.asarray(nc_raw, dtype=np.float32)
    
    if verbose:
        print(f"Successfully converted channels to float32")
        print(f"Channel shapes: dapi={dapi.shape}, npm1={npm1.shape}, nc={nc.shape}")

    # ------------------- basic normalization -------------------
    dapi_n = normalize01(dapi)
    npm1_n = normalize01(npm1)
    nc_n   = normalize01(nc)

    # ------------------- optional background subtraction -------------------
    if bg_subtract:
        # smooth background estimate, subtract, clip
        nc_bg = filters.gaussian(nc_n, sigma=bg_sigma, preserve_range=True)
        nc_bs = np.clip(nc_n - nc_bg, 0, None)
        nc_n = normalize01(nc_bs)

    # ------------------- per-nucleus measurements -------------------
    # Example: mean NC signal in nucleoli vs nucleoplasm per nucleus
    results = []

    nuc_props = measure.regionprops(nuc_lab)

    for nuc in nuc_props:
        nuc_id = nuc.label
        nuc_mask = (nuc_lab == nuc_id)

        # nucleoli within this nucleus (intersection)
        # npm1_lab has nucleoli labels across image; intersect masks
        nucleoli_mask = (npm1_lab > 0) & nuc_mask
        nucleoplasm_mask = nuc_mask & (~nucleoli_mask)

        if nucleoli_mask.sum() == 0 or nucleoplasm_mask.sum() == 0:
            continue

        mean_nc_nucleoli = float(nc_n[nucleoli_mask].mean())
        mean_nc_nucleoplasm = float(nc_n[nucleoplasm_mask].mean())

        rim_enrichment = mean_nc_nucleoli / (mean_nc_nucleoplasm + 1e-9)

        results.append({
            "group": group_name,
            "image": base,
            "nucleus_id": int(nuc_id),
            "mean_nc_nucleoli": mean_nc_nucleoli,
            "mean_nc_nucleoplasm": mean_nc_nucleoplasm,
            "rim_enrichment": rim_enrichment,
        })

    df = pd.DataFrame(results)
    df_path = os.path.join(out_base_dir, "coloc_metrics.csv")
    df.to_csv(df_path, index=False)

    # ------------------- save intermediates / overlays -------------------
    if save_intermediates:
        _save_gray(os.path.join(out_base_dir, "DAPI_norm.png"), dapi_n)
        _save_gray(os.path.join(out_base_dir, "NPM1_norm.png"), npm1_n)
        _save_gray(os.path.join(out_base_dir, f"{nucleolar_component}_norm.png"), nc_n)

        # label overlays for quick sanity check
        nuc_overlay = overlay_labels(dapi_n, nuc_lab)
        ncl_overlay = overlay_labels(npm1_n, npm1_lab)

        print("IMSAVE")
        print(nuc_overlay)

        skio.imsave(
            os.path.join(out_base_dir, "nuclei_overlay.png"),
            (np.clip(nuc_overlay, 0, 1) * 255).astype(np.uint8),
            check_contrast=False,
        )
        skio.imsave(
            os.path.join(out_base_dir, "nucleoli_overlay.png"),
            (np.clip(ncl_overlay, 0, 1) * 255).astype(np.uint8),
            check_contrast=False,
        )

        # save labels as tiffs too if desired
        _save_label_tiff(nuc_lab, os.path.join(out_base_dir, "nuclei_labels.tif"))
        _save_label_tiff(npm1_lab, os.path.join(out_base_dir, "nucleoli_labels.tif"))

    if verbose:
        print(f"[ok] wrote: {df_path} (n={len(df)})")

    # return dataframe for interactive use
    return df