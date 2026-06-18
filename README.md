# WMH Spatial Heterogeneity Project

## 1. Project Overview

This project analyzes the spatial heterogeneity of white matter hyperintensities (WMH) using baseline ADNI MRI data.

WMH lesions appear as bright regions on T2-weighted FLAIR MRI and are associated with aging, vascular risk factors, cognitive impairment, and neurodegenerative disease. The goal is to register individual WMH masks into MNI space, cluster population-level WMH voxels into dominant spatial regions, and evaluate associations between regional WMH burden and phenotypic or genetic variables.

The pipeline includes:

1. T2 FLAIR to T1 rigid registration.
2. T1 to MNI template registration.
3. WMH mask transformation into MNI space.
4. WMH mask smoothing and downsampling.
5. Population-level WMH voxel selection.
6. WMH voxel clustering into dominant spatial regions.
7. Regional WMH burden calculation.
8. Correlation analysis with phenotype variables.
9. Generation of report-ready figures and summary tables.

The final analysis used 30 ADNI baseline subjects and clustered the WMH pattern into 5 dominant regions.

## 2. Submitted Files

The submitted ZIP file contains the directly runnable code, final PDF report, supporting figures, and lightweight result summaries.

The submission structure is:

```text
.
├── README.md
├── requirements.txt
│
├── code/
│   └── wmh_project.py
│
├── report/
│   └── WMH_Project_Report.pdf
│
└── results_summary/
    ├── figures/
    │   ├── figure1_registration_qc.png
    │   ├── figure2_clustered_regions_mni.png
    │   └── figure3_correlation_grid.png
    │
    ├── cluster_metadata.csv
    ├── correlations.csv
    ├── merged_regional_phenotype.csv
    ├── regional_wmh_proportions.csv
    └── run_parameters.json
```

## 3. Data Not Included

Raw ADNI data and registered medical image volumes are not included in this submission.

The following files and folders are intentionally excluded:

```text
images.zip
MNI152_T1_1mm.nii.gz
all30m.xlsx
images_extracted/
registration/
*.nii
*.nii.gz
.venv/
__pycache__/
```

These files are excluded to reduce the submission size and to comply with ADNI data-use requirements. The submitted package contains code, the final report, supporting figures, and lightweight result summaries only.

## 4. Required Input Files for Reproduction

To rerun the full pipeline locally, prepare the following directory structure:

```text
E:\wmh_project_solution
│
├── images.zip
├── MNI152_T1_1mm.nii.gz
├── all30m.xlsx
│
└── wmh_project_direct_run
    ├── requirements.txt
    └── code
        └── wmh_project.py
```

Required input files:

* `images.zip`: ADNI subject folders containing T1 MRI, T2 FLAIR MRI, and WMH masks.
* `MNI152_T1_1mm.nii.gz`: MNI152 T1-weighted template.
* `all30m.xlsx`: Phenotypic and genetic variables for the 30 subjects.

These data files are required to reproduce the full analysis but are not included in the submitted ZIP file.

## 5. Software Requirements

The pipeline was implemented in Python.

Main Python dependencies include:

```text
numpy
pandas
openpyxl
scipy
scikit-learn
matplotlib
SimpleITK
```

All required packages can be installed using:

```powershell
pip install -r requirements.txt
```

## 6. Environment Setup

Open PowerShell and go to the project code directory:

```powershell
cd "E:\wmh_project_solution\wmh_project_direct_run"
```

Create a Python virtual environment:

```powershell
py -3 -m venv .venv
```

Activate the environment:

```powershell
. .\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
. .\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 7. Full Pipeline Run

Run the complete 30-subject analysis with:

```powershell
python code\wmh_project.py all `
  --images-zip "E:\wmh_project_solution\images.zip" `
  --mni-template "E:\wmh_project_solution\MNI152_T1_1mm.nii.gz" `
  --phenotype-xlsx "E:\wmh_project_solution\all30m.xlsx" `
  --output-dir "E:\wmh_project_solution\results" `
  --regions 5 `
  --downsample 4 `
  --prevalence-threshold 0.01 `
  --cluster-method kmeans
```

For Windows CMD, use the following single-line version:

```bat
python code\wmh_project.py all --images-zip "E:\wmh_project_solution\images.zip" --mni-template "E:\wmh_project_solution\MNI152_T1_1mm.nii.gz" --phenotype-xlsx "E:\wmh_project_solution\all30m.xlsx" --output-dir "E:\wmh_project_solution\results" --regions 5 --downsample 4 --prevalence-threshold 0.01 --cluster-method kmeans
```

## 8. Fast Test Run

Before running the complete pipeline, a small test run can be performed:

```powershell
python code\wmh_project.py all `
  --images-zip "E:\wmh_project_solution\images.zip" `
  --mni-template "E:\wmh_project_solution\MNI152_T1_1mm.nii.gz" `
  --phenotype-xlsx "E:\wmh_project_solution\all30m.xlsx" `
  --output-dir "E:\wmh_project_solution\results_test3" `
  --limit 3 `
  --regions 3 `
  --downsample 4 `
  --fast
```

Note that at least 2 successfully registered subjects are required for the population-level clustering step, so `--limit 1` will not work.

## 9. Pipeline Methods

### 9.1 Data Discovery

The script extracts `images.zip` and scans each subject folder for T1, T2 FLAIR, and WMH mask files. File matching is done by scoring candidate filenames against known naming patterns — for example, files containing `flair` score higher as T2 candidates, while files named `t1_brain_mask` are explicitly excluded from WMH mask selection to avoid a common mis-match. Subjects missing any of the three required files are skipped and reported in the log.

### 9.2 Image Registration

T2 FLAIR and T1 images come from different acquisition sequences, so direct voxel-intensity comparison is not meaningful. Mattes mutual information is used as the registration metric here because it measures statistical dependence between intensity distributions rather than requiring the two modalities to look alike.

The FLAIR image is first rigidly registered to the subject's own T1, handling only translation and rotation. The T1 is then registered to the MNI152 template in two stages: an affine pass to correct gross differences in position, orientation, and scale, followed by a B-spline nonrigid refinement to account for individual brain shape variation. The WMH mask is carried through both transforms using nearest-neighbor interpolation, which avoids introducing fractional lesion values at boundaries.

### 9.3 WMH Voxel Processing

Once all masks are in MNI space, they are Gaussian-smoothed and downsampled before population-level analysis. The smoothing step helps compensate for small residual registration errors across subjects. Downsampling by a factor of 4 keeps memory and compute time manageable without discarding meaningful spatial structure. Voxels present in fewer than 1% of subjects are excluded before clustering.

### 9.4 WMH Region Clustering

A voxel-by-subject matrix is constructed from the retained voxels. Each row represents one voxel and each column represents one subject's smoothed WMH value at that location. MiniBatch K-means is applied to group voxels into 5 clusters based on how similarly they behave across subjects — voxels that tend to be lesioned together end up in the same region. The choice of K=5 reflects a balance between regional interpretability and the sample size of 30 subjects.

### 9.5 Regional WMH Burden

For each subject and each clustered region, regional WMH burden is calculated as:

```text
regional WMH proportion = number of lesioned voxels in the region / total number of voxels in the region
```

This reduces each subject's high-dimensional WMH mask into 5 regional lesion features that are straightforward to use in downstream statistical analysis.

### 9.6 Phenotype Association Analysis

Regional WMH proportions are correlated with the following phenotypic and genetic variables:

```text
AGE
PTGENDER
APOE4
AV45
ADAS11
```

Pearson correlation coefficients and p-values are calculated for each phenotype-region pair. These are unadjusted bivariate correlations; no multiple comparison correction is applied at this stage.

## 10. Output Files

The default output directory is:

```text
E:\wmh_project_solution\results
```

Important output files include:

```text
results\manifest.csv

results\figures\figure1_registration_qc.png

results\clusters\figure2_clustered_regions_mni.png
results\clusters\regional_wmh_proportions.csv
results\clusters\cluster_metadata.csv

results\correlations\figure3_correlation_grid.png
results\correlations\correlations.csv
results\correlations\merged_regional_phenotype.csv

results\report\WMH_project_report.md

results\logs\registration_summary.csv
results\logs\run_config.json
results\logs\output_summary.json
```

The lightweight result summaries included in this submission are stored in:

```text
results_summary/
```

## 11. Core Supporting Figures

The submission includes three required supporting figures in:

```text
results_summary\figures\
```

The figures are:

1. `figure1_registration_qc.png`
   Representative registration quality-control results.

2. `figure2_clustered_regions_mni.png`
   Clustered dominant WMH regions overlaid in MNI space.

3. `figure3_correlation_grid.png`
   Correlation plots between regional WMH burden and phenotype variables.

These figures are also shown and discussed in the final PDF report.

## 12. Results Summary

In the completed run:

```text
Number of subjects: 30
Successful registrations: 30 / 30
Number of WMH regions: 5
Downsample factor: 4
WMH prevalence threshold: 0.01
Clustering method: MiniBatch K-means
Phenotype variables: AGE, PTGENDER, APOE4, AV45, ADAS11
```

All 30 subjects registered successfully. Among the phenotype variables, age showed the strongest unadjusted correlation with regional WMH burden, which is consistent with the known relationship between aging and WMH accumulation. APOE4 and AV45 associations were weaker and variable across regions. Full correlation statistics are in `results_summary/correlations.csv`.

## 13. Manual Manifest Correction

The script automatically generates:

```text
results\manifest.csv
```

If a subject's T1, T2 FLAIR, or WMH mask path is incorrectly detected, manually edit `manifest.csv` and rerun the pipeline with:

```powershell
python code\wmh_project.py all `
  --manifest "E:\wmh_project_solution\results\manifest.csv" `
  --mni-template "E:\wmh_project_solution\MNI152_T1_1mm.nii.gz" `
  --phenotype-xlsx "E:\wmh_project_solution\all30m.xlsx" `
  --output-dir "E:\wmh_project_solution\results" `
  --regions 5 `
  --downsample 4 `
  --prevalence-threshold 0.01 `
  --cluster-method kmeans
```

When using `--manifest`, the image ZIP file does not need to be scanned again.

## 14. Windows Notes

Place the project under an English-only path such as:

```text
E:\wmh_project_solution
```

Paths with Chinese characters or spaces can cause silent failures in SimpleITK's NIfTI reader — the file appears to load but the image data comes back empty. OneDrive-synced folders have the same issue because the file may not be locally available at read time.

The registration step is the slowest part of the pipeline. On a typical laptop it takes roughly 3–8 minutes per subject depending on image resolution and whether `--fast` is used. The script does not print per-iteration progress during registration, so the terminal will look idle for stretches of time. This is normal. To check whether things are actually running, look at the file modification timestamps under `results\registration\` — new `.tfm` and `.nii.gz` files should appear as each subject finishes.

## 15. Final Report

The final report is included as:

```text
report\WMH_Project_Report.pdf
```

The PDF covers the method description, registration results, clustered WMH regions, correlation analysis, discussion, limitations, and conclusion.
