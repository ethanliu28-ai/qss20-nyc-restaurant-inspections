# qss20-nyc-restaurant-inspections

NYC Restaurant Inspection Disparities

This project asks whether NYC restaurant inspection outcomes vary systematically rather than purely reflecting restaurant-level food safety practices. One shared OLS model of inspection SCORE tests three predictor groups at once: (1) cuisine group, (2) keywords in the restaurant's own name (e.g. "bistro," "organic," "express"), and (3) the median household income of the zip code the restaurant sits in. A separate, independent analysis then checks whether scores themselves behave strangely right at NYC's legal A/B and B/C grade cutoffs. Together these test whether NYC's inspection system operates as a neutral arbiter of food safety or reflects economic and geographic inequality across the five boroughs.

Setup

1. `pip install -r requirements.txt`
2. Get the raw data file — see data/README.md.
3. Run the notebooks in `code/` in numeric order (below), starting from `01_data_pull_clean.ipynb`.

Notebooks

Run in numeric order — each step reads the previous step's output rather than redoing its work.

code/utils.py — shared helpers imported by every notebook below: path/column constants, cuisine grouping, keyword indicators, zip-code income lookup (real Census ACS pull or approximate fallback — see Data below), shared-model fitting, and forest-plot drawing.

code/01_data_pull_clean.ipynb — takes in data/DOHMH_New_York_City_Restaurant_Inspection_Results.csv (raw); drops exact duplicates, standardizes column names, parses inspection dates, cleans zip codes; outputs data/cleaned_inspections.csv.

code/02_analysis.ipynb — takes in data/cleaned_inspections.csv; groups cuisines into 7 categories, builds restaurant-name keyword indicators, merges in zip-code income quartile, and fits one shared OLS model of inspection SCORE on all three predictor groups; outputs output/shared_model_coefficients.csv (coefficients/CI for every predictor) and data/model_ready_inspections.csv (row-level data for the notebooks below's supporting plots).

code/03a_cuisine_forest_plot.ipynb — takes in 02's two output files; outputs forest_plot_cuisine.png (cuisine group coefficients).

code/03b_keyword_forest_plot.ipynb — same inputs as 03a; outputs forest_plot_keywords.png (restaurant-name keyword coefficients).

code/03c_income_forest_plot.ipynb — same inputs as 03a; outputs forest_plot_income.png (zip-code income-quartile coefficients).

code/04_grade_cutoff_bunching.ipynb — takes in data/cleaned_inspections.csv; collapses to one row per inspection and checks whether scores unusually cluster just below the A/B (13) and B/C (27) grade cutoffs; outputs score_distribution_grade_cutoffs.png.

Note on unit of analysis: 01 only keeps 8 columns (camis, dba, boro, zipcode, cuisine_description, inspection_date, score, grade), so multiple violation-citation rows belonging to the same inspection become identical on those columns and collapse into one row during deduplication. That means data/cleaned_inspections.csv — and everything downstream of it — is already at (close to) one row per inspection, not one row per violation, which is the correct unit for the OLS model in 02 (otherwise inspections with more violations would be overweighted).

Data

Source: [NYC DOHMH Restaurant Inspection Results](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j) (NYC Open Data), read from data/DOHMH_New_York_City_Restaurant_Inspection_Results.csv. Not tracked in git — the raw file is ~150MB, past GitHub's per-file limit. See data/README.md for exactly how to download it and what else lives in data/ once the pipeline has run.

Zip-code median household income: utils.INCOME_DATA_SOURCE currently = "fallback". Intended source is the Census Bureau's American Community Survey 5-year estimates by ZCTA, pulled via the public Census API (requires a free key — see utils._census_api_key) and cached to data/median_income_by_zip.csv. Until a key is available, income comes from utils.NYC_ZIP_INCOME_TIER_FALLBACK, a hand-built approximation from general knowledge of NYC neighborhood income patterns — NOT official Census data. This is a known limitation; flip INCOME_DATA_SOURCE to "real" and rerun 02 (then 03c) once a key is available.

Output

See output/ for generated figures and output/shared_model_coefficients.csv for the full regression table. Each forest plot shows unstandardized OLS coefficients (95% CI), i.e. the actual difference in inspection SCORE vs. the reference group, so effect sizes are directly comparable in score units across cuisines, keywords, and income quartiles.
