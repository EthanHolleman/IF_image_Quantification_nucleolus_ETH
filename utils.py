import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage import measure, color, io as skio
from skimage.util import img_as_uint

import nd2
from csbdeep.utils import normalize
from stardist.models import StarDist2D


# helpers 
def normalize01(a):
    a = a.astype(np.float32)
    mn, mx = float(np.nanmin(a)), float(np.nanmax(a))
    if mx <= mn:
        return np.zeros_like(a, dtype=np.float32)
    return (a - mn) / (mx - mn)

def parent_by_mode(child_labels, parent_labels):
    """Map child label -> parent label, based on mode of overlapping parent labels."""
    mapping = {}
    childs = np.unique(child_labels)
    childs = childs[childs != 0]
    for cl in childs:
        m = (child_labels == cl)
        pl = parent_labels[m]
        pl = pl[pl != 0]
        mapping[cl] = int(np.bincount(pl).argmax()) if pl.size else 0
    return mapping

def overlay_labels(gray, labels, edge_color=(1, 0, 0)):
    g = normalize01(gray)
    rgb = np.dstack([g, g, g])
    edges = measure.find_contours(labels, 0.5)  # quick & dirty edges
    # fallback to skimage's boundary finder if you prefer
    from skimage.segmentation import find_boundaries
    bmask = find_boundaries(labels, mode='outer')
    r, g2, b = edge_color
    rgb[..., 0][bmask] = r
    rgb[..., 1][bmask] = g2
    rgb[..., 2][bmask] = b
    return np.clip(rgb, 0, 1)

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _save_gray(img, path):
    skio.imsave(path, (normalize01(img) * 255).astype(np.uint8), check_contrast=False)

def _save_label_tiff(lbl, path):
    skio.imsave(path, img_as_uint(lbl), check_contrast=False)



def main2(
    nd2_file,
    nuclei_model,        # StarDist2D model for DAPI
    nucleoli_model,      # StarDist2D model for NPM1
    DAPI_IDX=0,
    NPM1_IDX=2,
    DAPI_NAME="DAPI",
    NPM1_NAME="NPM1",
    USE_MAX_PROJ=True,
    PIXEL_SIZE_UM=None,
    output_dir="outputs_stardist",
    group_name="",
    min_nucleus_area_px=400,     # filter tiny nuclei
    min_nucleolus_area_px=25,    # filter tiny nucleoli
    save_png=True, save_tiff=True,
    save_numpy=False, save_figure=True, figure_dpi=600
):
    # ---------- paths ----------
    file_stem = os.path.splitext(os.path.basename(nd2_file))[0]
    prefix = f"{group_name}_" if group_name else ""
    out_root = _ensure_dir(os.path.join(output_dir, prefix + file_stem))

    # ---------- load ND2 ----------
    rdr = nd2.ND2File(nd2_file)
    arr_raw = rdr.asarray()

    # reshape to (C, Y, X) by max projection if needed
    if arr_raw.ndim == 5:        # (T,Z,C,Y,X)
        T, Z, C, Y, X = arr_raw.shape
        arr = arr_raw.max(axis=(0, 1)) if USE_MAX_PROJ else arr_raw[0, 0]
        chan_count = C
        chan_names = []
        try:
            md = rdr.metadata
            if hasattr(md, "channels") and md.channels:
                chan_names = [getattr(ch, "name", f"ch{i}") or f"ch{i}"
                              for i, ch in enumerate(md.channels)]
        except Exception:
            pass
    elif arr_raw.ndim == 4:      # (Z,C,Y,X) or (T,C,Y,X)
        arr = arr_raw.max(axis=0) if USE_MAX_PROJ else arr_raw[0]
        chan_count = arr.shape[0]
        chan_names = [f"ch{i}" for i in range(chan_count)]
    elif arr_raw.ndim == 3:      # (C,Y,X)
        arr = arr_raw
        chan_count = arr.shape[0]
        chan_names = [f"ch{i}" for i in range(chan_count)]
    elif arr_raw.ndim == 2:      # (Y,X) single channel
        arr = arr_raw[None, ...]
        chan_count = 1
        chan_names = ["ch0"]
    else:
        rdr.close()
        raise ValueError(f"Unexpected ND2 shape: {arr_raw.shape}")
    rdr.close()

    def _resolve_channel(name_hint, fallback_idx, names, nC):
        if isinstance(fallback_idx, (list, tuple, np.ndarray)):
            fallback_idx = int(fallback_idx[0])
        if names:
            lname = [str(s).lower() for s in names]
            target = (str(name_hint).lower() if name_hint is not None else "")
            if target:
                for i, s in enumerate(lname):
                    if s == target:
                        return int(i)
                for i, s in enumerate(lname):
                    if target in s:
                        return int(i)
        if isinstance(fallback_idx, (int, np.integer)) and 0 <= fallback_idx < nC:
            return int(fallback_idx)
        return 0

    dapi_ch = int(np.clip(_resolve_channel(DAPI_NAME, DAPI_IDX, chan_names, chan_count), 0, chan_count-1))
    npm1_ch = int(np.clip(_resolve_channel(NPM1_NAME, NPM1_IDX, chan_names, chan_count), 0, chan_count-1))

    dapi = arr[dapi_ch].astype(np.float32)
    npm1 = arr[npm1_ch].astype(np.float32)

    # ---------- 1) nuclei via StarDist on DAPI ----------
    dapi_norm = normalize(dapi, 1, 99.8)   # percentile normalization (typical for StarDist)
    nuclei_labels, _ = nuclei_model.predict_instances(dapi_norm)

    # remove tiny nuclei
    props_nuc = measure.regionprops(nuclei_labels)
    keep = [p.label for p in props_nuc if p.area >= min_nucleus_area_px]
    mask_keep = np.isin(nuclei_labels, keep)
    nuclei = measure.label(mask_keep).astype(np.int32)

    # ---------- 2) nucleoli via StarDist on NPM1 ----------
    npm1_norm = normalize(npm1, 1, 99.8)
    nucleoli_labels, _ = nucleoli_model.predict_instances(npm1_norm)

    # confine nucleoli to inside nuclei, and remove tiny nucleoli
    nucleoli_labels[ nuclei == 0 ] = 0
    props_nol = measure.regionprops(nucleoli_labels)
    keep_nol = [p.label for p in props_nol if p.area >= min_nucleolus_area_px]
    mask_nol_keep = np.isin(nucleoli_labels, keep_nol)
    nucleoli = measure.label(mask_nol_keep).astype(np.int32)
    nucleoli[ nuclei == 0 ] = 0

    parent_map = parent_by_mode(nucleoli, nuclei)

    # per-nucleolus table
    props_nucleoli = measure.regionprops_table(
        nucleoli, intensity_image=npm1,
        properties=["label", "area", "mean_intensity", "eccentricity", "solidity", "perimeter"]
    )
    df_nucleo = pd.DataFrame(props_nucleoli)
    if len(df_nucleo):
        df_nucleo["parent_nucleus"] = df_nucleo["label"].map(parent_map).fillna(0).astype(int)
        df_nucleo = df_nucleo[df_nucleo["parent_nucleus"] > 0].copy()

    # nucleoli count per nucleus
    if len(df_nucleo):
        counts = (df_nucleo.groupby("parent_nucleus")
                  .size().rename("nucleoli_count").reset_index()
                  .rename(columns={"parent_nucleus": "nucleus_label"}))
    else:
        counts = pd.DataFrame(columns=["nucleus_label", "nucleoli_count"])

    # per-cell / per-nucleus table
    props_nuclei = measure.regionprops_table(
        nuclei, intensity_image=dapi,
        properties=["label", "area", "mean_intensity", "eccentricity", "solidity"]
    )
    df_cells = (pd.DataFrame(props_nuclei)
                .rename(columns={"label": "nucleus_label"})
                .merge(counts, on="nucleus_label", how="left")
                .fillna({"nucleoli_count": 0}))
    df_cells["nucleoli_count"] = df_cells["nucleoli_count"].astype(int)

    # pixel → µm^2 if pixel size given
    if PIXEL_SIZE_UM:
        px_area = float(PIXEL_SIZE_UM) ** 2
        if len(df_cells):
            df_cells["area_um2"] = df_cells["area"] * px_area
        if len(df_nucleo):
            df_nucleo["area_um2"] = df_nucleo["area"] * px_area

    total_area_nucleoli = float(df_nucleo["area"].sum()) if len(df_nucleo) else 0.0
    total_area_nuclei   = float(df_cells["area"].sum())  if len(df_cells)  else 1e-9
    ratio = total_area_nucleoli / total_area_nuclei

    if save_png:
        _save_gray(dapi, os.path.join(out_root, "raw_DAPI.png"))
        _save_gray(npm1, os.path.join(out_root, "raw_NPM1.png"))

    if save_tiff:
        _save_label_tiff(nuclei,   os.path.join(out_root, "nuclei_labels.tif"))
        _save_label_tiff(nucleoli, os.path.join(out_root, "nucleoli_labels.tif"))

    if save_numpy:
        np.save(os.path.join(out_root, "nuclei_labels.npy"), nuclei)
        np.save(os.path.join(out_root, "nucleoli_labels.npy"), nucleoli)

    if save_png:
        skio.imsave(
            os.path.join(out_root, "nuclei_labels.png"),
            (color.label2rgb(nuclei, bg_label=0, bg_color=(0,0,0)) * 255).astype(np.uint8),
            check_contrast=False,
        )
        skio.imsave(
            os.path.join(out_root, "nucleoli_labels.png"),
            (color.label2rgb(nucleoli, bg_label=0, bg_color=(0,0,0)) * 255).astype(np.uint8),
            check_contrast=False,
        )
        skio.imsave(
            os.path.join(out_root, "overlay_nuclei.png"),
            (overlay_labels(dapi, nuclei) * 255).astype(np.uint8),
            check_contrast=False,
        )
        skio.imsave(
            os.path.join(out_root, "overlay_nucleoli.png"),
            (overlay_labels(npm1, nucleoli) * 255).astype(np.uint8),
            check_contrast=False,
        )

    if save_figure:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes[0,0].imshow(normalize01(dapi), cmap="Blues"); axes[0,0].set_title("DAPI (raw nuclei)")
        axes[0,1].imshow(normalize01(npm1), cmap="Reds");  axes[0,1].set_title("NPM1 (raw nucleoli)")
        axes[1,0].imshow(overlay_labels(dapi, nuclei));    axes[1,0].set_title(f"Nuclei overlay ({nuclei.max()} nuclei)")
        axes[1,1].imshow(overlay_labels(npm1, nucleoli));  axes[1,1].set_title(f"Nucleoli overlay ({nucleoli.max()} nucleoli)")
        for ax in axes.ravel():
            ax.axis("off")
        plt.tight_layout()
        fig.savefig(os.path.join(out_root, "summary_2x2.png"), dpi=figure_dpi, bbox_inches="tight")
        plt.close(fig)

    # tables
    df_cells.to_csv(os.path.join(out_root, "per_cell.csv"), index=False)
    df_nucleo.to_csv(os.path.join(out_root, "per_nucleolus.csv"), index=False)

    # summary
    meta = {
        "file": nd2_file,
        "group": group_name,
        "channels": {
            "DAPI": DAPI_NAME,
            "NPM1": NPM1_NAME,
            "DAPI_idx": dapi_ch,
            "NPM1_idx": npm1_ch,
        },
        "n_nuclei": int(nuclei.max()),
        "n_nucleoli": int(nucleoli.max()),
        "total_area_nucleoli_px": total_area_nucleoli,
        "total_area_nuclei_px": total_area_nuclei,
        "nucleoli_over_nucleus_area_ratio": ratio,
        "pixel_size_um": PIXEL_SIZE_UM,
        "params": {
            "min_nucleus_area_px": min_nucleus_area_px,
            "min_nucleolus_area_px": min_nucleolus_area_px,
        },
    }
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "out_dir": out_root,
        "nuclei_count": int(nuclei.max()),
        "nucleoli_count": int(nucleoli.max()),
        "ratio": ratio,
        "cells_csv": os.path.join(out_root, "per_cell.csv"),
        "nucleoli_csv": os.path.join(out_root, "per_nucleolus.csv"),
        "summary_fig": os.path.join(out_root, "summary_2x2.png"),
    }