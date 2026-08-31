# data/

This folder is where the pipeline reads and writes its data files. None of the CSVs
themselves are committed to this repo — the raw file alone is ~150MB, well past GitHub's
per-file limit — so this README (and the notebook outputs, which are committed) stands in
for the actual data.

## Getting the raw file

1. Download https://www.kaggle.com/datasets/new-york-city/nyc-inspections?resource=download 
2. Save it here as exactly:
   `data/DOHMH_New_York_City_Restaurant_Inspection_Results.csv`
3. Run the notebooks in order starting from `code/01_data_pull_clean.ipynb` — each one reads
   the previous step's output and writes its own, regenerating every other file listed below.

## Files that appear here once you run the pipeline

| File | Written by | Description |
|---|---|---|
| `DOHMH_New_York_City_Restaurant_Inspection_Results.csv` | *(you, manually — see above)* | Raw inspection data, one row per cited violation |
| `cleaned_inspections.csv` | `code/01_data_pull_clean.ipynb` | Deduplicated, standardized version of the raw file |
| `model_ready_inspections.csv` | `code/02_analysis.ipynb` | Row-level data with cuisine/keyword/income predictors attached, for the `03a`/`03b`/`03c` plots |

None of these are tracked in git — they're all listed in `.gitignore` and get regenerated
by re-running the notebooks.
