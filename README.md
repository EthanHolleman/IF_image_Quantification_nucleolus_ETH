# Nucleolar Analysis Pipeline

A comprehensive Python pipeline for analyzing nucleolar protein localization, colocalization, and rim enrichment in fluorescence microscopy images.

## Overview

This pipeline performs three main analyses on multi-channel ND2 microscopy files:

1. **Nuclei and Nucleoli Segmentation** - Uses StarDist deep learning models to segment nuclei (DAPI) and nucleoli (NPM1)
2. **Colocalization Analysis** - Quantifies colocalization between nucleolar markers (NPM1) and target proteins (FBL, UBF, etc.) using Manders coefficients
3. **Rim Enrichment Analysis** - Measures spatial distribution of proteins between nucleolar rim and core regions

## Features

- Batch processing of ND2 files organized by experimental groups
- Deep learning-based segmentation using StarDist
- Flatfield correction and background subtraction
- Multi-metric quantification (area, intensity, morphology)
- Statistical analysis with Mann-Whitney U tests
- Automated visualization generation
- Comprehensive CSV outputs for downstream analysis

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Architecture](#pipeline-architecture)
- [Step-by-Step Guide](#step-by-step-guide)
  - [Step 1: Nuclei and Nucleoli Segmentation](#step-1-nuclei-and-nucleoli-segmentation)
  - [Step 2: Colocalization Analysis](#step-2-colocalization-analysis)
  - [Step 3: Rim Enrichment Analysis](#step-3-rim-enrichment-analysis)
- [Input Data Format](#input-data-format)
- [Output Files](#output-files)
- [Key Metrics Explained](#key-metrics-explained)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Installation

### Requirements

- Python 3.8+

### Dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-image
pip install stardist csbdeep
pip install nd2  # for reading Nikon ND2 files
pip install tqdm  # for progress bars
```

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/nucleolar-analysis.git
cd nucleolar-analysis
```

2. Create the required directory structure:
```bash
mkdir -p scripts
```

3. Place the following files in the `scripts/` directory:
   - `utils.py` - Helper functions
   - `Segment_nuclei_and_nucleoli.py` - Segmentation pipeline
   - `Colocalization.py` - Colocalization analysis
   - `Rim_enrichment.py` - Rim enrichment analysis

---

## Quick Start

For a complete analysis, run the three Jupyter notebooks in order:

1. `Run_DAPI_NPM1_segmentation.ipynb` - Segment nuclei and nucleoli
2. `Run_UBF_FBL_coloc.ipynb` - Analyze colocalization
3. `Rim_enrichment_analysis.ipynb` - Measure rim enrichment and generate statistics

Each notebook is self-contained with example parameters.

---

## Pipeline Architecture

```
ND2 Files (Multi-channel microscopy)
         ↓
┌────────────────────────────────────────┐
│  STEP 1: Segmentation                  │
│  - StarDist on DAPI → Nuclei labels    │
│  - StarDist on NPM1 → Nucleoli labels  │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  STEP 2: Colocalization                │
│  - Segment target protein foci         │
│  - Calculate Manders M1/M2             │
│  - Jaccard index                       │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  STEP 3: Rim Enrichment                │
│  - Define rim vs core regions          │
│  - Area-based enrichment               │
│  - Intensity-based enrichment          │
│  - Statistical comparisons             │
└────────────────────────────────────────┘
         ↓
    CSV + Visualizations
```

---

## Step-by-Step Guide

### Step 1: Nuclei and Nucleoli Segmentation

**Purpose:** Segment nuclei using DAPI and nucleoli using NPM1 with StarDist deep learning models.

**Script:** `scripts/Segment_nuclei_and_nucleoli.py`  
**Notebook:** `Run_DAPI_NPM1_segmentation.ipynb`

#### What It Does:

1. Loads ND2 files and performs max projection (if Z-stacks/time series)
2. Applies StarDist segmentation to DAPI channel → nuclei labels
3. Applies StarDist segmentation to NPM1 channel → nucleoli labels
4. Filters out small objects (configurable thresholds)
5. Maps each nucleolus to its parent nucleus
6. Extracts morphological and intensity features
7. Saves label masks, overlays, and quantification tables

#### Key Parameters:

```python
main(
    nd2_file="path/to/file.nd2",
    nuclei_model=StarDist2D.from_pretrained('2D_versatile_fluo'),
    nucleoli_model=StarDist2D.from_pretrained('2D_versatile_fluo'),
    output_dir="./STARDIST_results",
    group_name="Control",  # Experimental group identifier
    DAPI_NAME="DAPI",      # Channel name or index
    NPM1_NAME="NPM1",
    min_nucleus_area_px=400,    # Filter small nuclei
    min_nucleolus_area_px=25    # Filter small nucleoli
)
```

#### Outputs:

- `nuclei_labels.tif` / `.png` - Nucleus segmentation masks
- `nucleoli_labels.tif` / `.png` - Nucleolus segmentation masks
- `per_cell.csv` - Per-nucleus measurements (area, intensity, nucleoli count)
- `per_nucleolus.csv` - Per-nucleolus measurements
- `summary_2x2.png` - Visualization grid
- `summary.json` - Metadata and summary statistics

#### Example Usage:

```python
from scripts.Segment_nuclei_and_nucleoli import main
from stardist.models import StarDist2D
import glob

nuclei_model = StarDist2D.from_pretrained('2D_versatile_fluo')
nucleoli_model = StarDist2D.from_pretrained('2D_versatile_fluo')

for nd2_file in glob.glob("data/Control/*.nd2"):
    results = main(
        nd2_file=nd2_file,
        nuclei_model=nuclei_model,
        nucleoli_model=nucleoli_model,
        output_dir="./outputs",
        group_name="Control"
    )
    print(f"Processed: {results['nuclei_count']} nuclei, {results['nucleoli_count']} nucleoli")
```

---

### Step 2: Colocalization Analysis

**Purpose:** Quantify colocalization between NPM1 (nucleolar marker) and target proteins (FBL, UBF, etc.).

**Script:** `scripts/Colocalization.py`  
**Notebook:** `Run_UBF_FBL_coloc.ipynb`

#### What It Does:

1. Loads pre-computed nuclei and nucleoli labels from Step 1
2. Loads the ND2 file to extract target protein channel (e.g., FBL)
3. Applies flatfield correction and background subtraction
4. Segments target protein foci using Otsu thresholding
5. Calculates global and per-nucleus colocalization metrics:
   - **Manders M1**: Fraction of NPM1 signal overlapping with target protein
   - **Manders M2**: Fraction of target protein signal overlapping with NPM1
   - **Jaccard Index**: Overlap between binary masks
6. Maps foci to parent nuclei
7. Saves masks, overlays, and quantification tables

#### Key Parameters:

```python
get_CoLoc(
    nd2_file="path/to/file.nd2",
    nucleolar_component="FBL",  # Target protein name
    nuclei_labels_path="outputs/Control_file/nuclei_labels.png",
    npm1_labels_path="outputs/Control_file/nucleoli_labels.png",
    output_dir="./coloc_results",
    group_name="Control",
    NC_IDX=3  # Channel index for target protein (0-indexed)
)
```

#### Outputs:

- `{target}_mask.tif` - Binary mask of target protein foci
- `per_nucleus_manders_NPM1_{target}.csv` - Per-nucleus colocalization coefficients
- `per_cell.csv` - Updated cell-level measurements
- `per_{target}_focus.csv` - Per-focus measurements
- `summary_2x3.png` - Visualization with segmentation overlays
- `summary.json` - Global Manders/Jaccard coefficients

#### Colocalization Metrics:

- **Manders M1 (NPM1 in Target)**: How much of the NPM1 signal is found in regions where the target protein is present. Values range 0-1; higher = more NPM1 colocalizes with target.
- **Manders M2 (Target in NPM1)**: How much of the target protein signal is found in NPM1+ regions. Values range 0-1; higher = more target protein is nucleolar.
- **Jaccard Index**: Spatial overlap of binary masks. J = Intersection / Union. Values range 0-1; 1 = perfect overlap.

#### Example Usage:

```python
from scripts.Colocalization import get_CoLoc

get_CoLoc(
    nd2_file="data/Control/file001.nd2",
    nucleolar_component="FBL",
    nuclei_labels_path="STARDIST_results/Control_file001/nuclei_labels.png",
    npm1_labels_path="STARDIST_results/Control_file001/nucleoli_labels.png",
    output_dir="./FBL_coloc_results",
    group_name="Control",
    NC_IDX=3
)
```

---

### Step 3: Rim Enrichment Analysis

**Purpose:** Measure spatial distribution of proteins between nucleolar rim and core regions.

**Script:** `scripts/Rim_enrichment.py`  
**Notebook:** `Rim_enrichment_analysis.ipynb`

#### What It Does:

1. Loads pre-segmented nucleoli, nuclei, and target protein masks
2. Loads raw intensity image for the target protein
3. Defines rim and core regions using morphological erosion
4. Calculates two types of enrichment:
   - **Area Enrichment**: Spatial crowding of foci in rim vs core
   - **Intensity Enrichment**: Protein concentration in rim vs core
5. Aggregates measurements per cell
6. Runs Mann-Whitney U tests between experimental groups
7. Generates violin plots with significance annotations

#### Key Parameters:

```python
analyze_rim_foci_enrichment(
    nd2_file="path/to/file.nd2",
    nuclei_labels_path="outputs/nuclei_labels.tif",
    nucleoli_labels_path="outputs/nucleoli_labels.tif",
    foci_mask_path="outputs/FBL_mask.tif",
    target_intensity_path="outputs/raw_FBL.png",  # Raw intensity image
    target_name="FBL",
    rim_width_px=4,  # Width of rim region in pixels
    output_dir="./rim_results",
    group_name="Control",
    save_visuals=True
)
```

#### Outputs:

- `rim_enrichment_{target}.csv` - Per-cell rim enrichment metrics
- `vis_{target}_rim.png` - Visualization of core (red) and rim (green) regions
- `All_Groups_{target}_Rim_Enrichment.csv` - Combined data from all groups
- `Stats_MannWhitney_{target}.csv` - Statistical test results
- `Violin_Plots_{target}_Rim_Enrichment_FULL.png` - Publication-ready plots

#### Key Metrics:

**Area-Based (Spatial Crowding):**
- `prop_foci_area_in_rim`: Fraction of foci area located in rim (0-1)
- `score_area_{target}`: Rim density / Core density (>1.0 = rim enrichment)

**Intensity-Based (Concentration):**
- `prop_intensity_in_rim`: Fraction of total protein signal in rim (0-1)
- `score_intensity_{target}`: Mean rim intensity / Mean core intensity (>1.0 = rim enrichment)

**Interpretation:**
- Scores **> 1.0**: Protein is enriched in the rim
- Scores **= 1.0**: No enrichment (uniform distribution)
- Scores **< 1.0**: Protein is depleted from the rim (core-enriched)

#### Statistical Analysis:

The pipeline automatically performs:
- Mann-Whitney U tests (non-parametric) between all group pairs
- Significance levels: `****` p<0.0001, `***` p<0.001, `**` p<0.01, `*` p<0.05, `ns` not significant
- Multiple comparison handling (consider Bonferroni correction if needed)

#### Example Usage:

```python
from scripts.Rim_enrichment import analyze_rim_foci_enrichment

df, nuclei = analyze_rim_foci_enrichment(
    nd2_file="data/Control/file001.nd2",
    nuclei_labels_path="coloc_results/Control_file001/nuclei_labels.tif",
    nucleoli_labels_path="coloc_results/Control_file001/nucleoli_labels.tif",
    foci_mask_path="coloc_results/Control_file001/FBL_mask.tif",
    target_intensity_path="coloc_results/Control_file001/raw_FBL.png",
    target_name="FBL",
    rim_width_px=4,
    output_dir="./rim_analysis",
    group_name="Control"
)

print(df[['prop_intensity_in_rim', 'score_intensity_FBL']].describe())
```

---

## Input Data Format

### Directory Structure

Organize your ND2 files by experimental groups:

```
data/
├── Control/
│   ├── file001.nd2
│   ├── file002.nd2
│   └── file003.nd2
├── Treated/
│   ├── file004.nd2
│   ├── file005.nd2
│   └── file006.nd2
└── Knockout/
    ├── file007.nd2
    └── file008.nd2
```

### ND2 Channel Requirements

Your ND2 files should contain at least these channels:
1. **DAPI** (nuclear stain)
2. **NPM1** (nucleolar marker)
3. **Target protein** (e.g., FBL, UBF) - for colocalization and rim analysis

The pipeline will attempt to identify channels by name. If automatic detection fails, you can specify channel indices (0-indexed).

---

## Output Files

### Directory Structure

```
outputs/
├── STARDIST_results/
│   └── Control_file001/
│       ├── nuclei_labels.tif
│       ├── nucleoli_labels.tif
│       ├── per_cell.csv
│       ├── per_nucleolus.csv
│       └── summary_2x2.png
├── FBL_coloc_results/
│   └── Control_file001/
│       ├── FBL_mask.tif
│       ├── per_nucleus_manders_NPM1_FBL.csv
│       ├── summary_2x3.png
│       └── summary.json
└── Rim_Enrichment_Results/
    ├── Control_file001/
    │   ├── rim_enrichment_FBL.csv
    │   └── vis_FBL_rim.png
    ├── All_Groups_FBL_Rim_Enrichment.csv
    ├── Stats_MannWhitney_FBL.csv
    └── Violin_Plots_FBL_Rim_Enrichment_FULL.png
```

### CSV File Descriptions

#### `per_cell.csv` (Step 1)
Columns: `nucleus_label`, `area`, `mean_intensity`, `eccentricity`, `solidity`, `nucleoli_count`, `area_um2` (if pixel size provided)

#### `per_nucleolus.csv` (Step 1)
Columns: `label`, `area`, `mean_intensity`, `eccentricity`, `solidity`, `perimeter`, `parent_nucleus`, `area_um2`

#### `per_nucleus_manders_NPM1_{target}.csv` (Step 2)
Columns: `nucleus_label`, `manders_M1_NPM1_in_{target}`, `manders_M2_{target}_in_NPM1`

#### `rim_enrichment_{target}.csv` (Step 3)
Key columns:
- `parent_nucleus`: Cell identifier
- `nucleoli_count`: Number of nucleoli per cell
- `prop_foci_area_in_rim`: Area fraction in rim
- `score_area_{target}`: Area enrichment score
- `prop_intensity_in_rim`: Intensity fraction in rim
- `score_intensity_{target}`: Intensity enrichment score
- `mean_int_core`, `mean_int_rim`: Mean intensities in each region

#### `Stats_MannWhitney_{target}.csv` (Step 3)
Columns: `Metric`, `Comparison`, `Group1`, `Group2`, `p-value`, `Significance`, `U-stat`

---

## Key Metrics Explained

### Segmentation Quality
- **Nucleus area**: Typical range 2000-10000 px² (depends on cell type and magnification)
- **Nucleolus area**: Typical range 100-2000 px²
- **Nucleoli per nucleus**: Typically 1-5 for most cell types

### Colocalization
- **Manders M1 = 0.7**: 70% of NPM1 signal overlaps with target protein
- **Manders M2 = 0.3**: 30% of target protein signal overlaps with NPM1
- **Jaccard = 0.5**: 50% overlap between binary masks

### Rim Enrichment
- **Area score = 1.5**: Foci are 1.5× more densely packed in rim than core
- **Intensity score = 2.0**: Protein concentration is 2× higher in rim than core
- **Proportion in rim = 0.4**: 40% of the total signal is in the rim region

---

## Customization

### Using Custom StarDist Models

If you've trained your own StarDist models:

```python
from stardist.models import StarDist2D

# Load custom models
nuclei_model = StarDist2D(None, name='my_nuclei_model', basedir='models/')
nucleoli_model = StarDist2D(None, name='my_nucleoli_model', basedir='models/')
```

### Adjusting Segmentation Thresholds

```python
# More stringent filtering
main(
    nd2_file=file,
    nuclei_model=nuclei_model,
    nucleoli_model=nucleoli_model,
    min_nucleus_area_px=1000,  # Larger minimum
    min_nucleolus_area_px=50
)
```

### Changing Rim Width

```python
# Wider rim region (8 pixels instead of 4)
analyze_rim_foci_enrichment(
    nd2_file=file,
    rim_width_px=8,
    # ... other parameters
)
```

### Custom Channel Indices

```python
# Explicitly specify channel indices
get_CoLoc(
    nd2_file=file,
    nucleolar_component="FBL",
    NC_IDX=4,  # FBL is in channel 4
    DAPI_IDX=0,
    NPM1_IDX=2
)
```

---

## Troubleshooting

### Common Issues

**Problem:** "Channel not found" error  
**Solution:** Manually specify channel indices using `DAPI_IDX`, `NPM1_IDX`, `NC_IDX` parameters

**Problem:** No nucleoli detected  
**Solution:** 
- Check NPM1 channel quality
- Adjust `min_nucleolus_area_px` threshold
- Verify nucleoli are within nuclei boundaries

**Problem:** Colocalization scores are all NaN  
**Solution:** 
- Ensure nuclei were detected in Step 1
- Verify target protein channel has signal
- Check that label paths point to correct files

**Problem:** Out of memory errors  
**Solution:** 
- Process files one at a time instead of loading all into memory
- Reduce figure DPI (`figure_dpi=300`)
- Disable numpy saving (`save_numpy=False`)

**Problem:** Rim enrichment scores seem incorrect  
**Solution:**
- Verify `target_intensity_path` points to the raw intensity image (not a mask)
- Check that rim width is appropriate for your nucleoli size
- Inspect `vis_{target}_rim.png` to verify rim/core regions

### Debug Tips

Enable verbose output:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check intermediate outputs:
```python
# After segmentation, verify labels
from skimage import io
labels = io.imread("outputs/nuclei_labels.tif")
print(f"Nuclei count: {labels.max()}")
print(f"Label shape: {labels.shape}")
```

---

## Dependencies and Versions

Tested with:
- Python 3.9
- NumPy 1.24.3
- pandas 2.0.3
- scikit-image 0.21.0
- StarDist 0.8.3
- CSBDeep 0.7.4
- nd2 0.5.3

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{nucleolar_analysis_pipeline,
  title={Nucleolar Analysis Pipeline},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/nucleolar-analysis}
}
```

---

## License

MIT License - see LICENSE file for details

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with a clear description of changes

---

## Contact

For questions or issues, please open a GitHub issue or contact [mannan.bhola@northwestern.edu]

---

## Acknowledgments

This pipeline uses:
- **StarDist** for deep learning-based segmentation ([Weigert et al., 2020](https://doi.org/10.1007/978-3-030-00934-2_30))
- **scikit-image** for image processing
- **nd2** library for reading Nikon microscopy files