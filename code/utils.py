"""Shared data-loading, model-fitting, and plotting helpers for the notebook pipeline:

01_data_pull_clean.ipynb   -- reads the raw DOHMH CSV, cleans it, writes data/cleaned_inspections.csv
02_analysis.ipynb          -- reads the cleaned data, builds cuisine/keyword/income predictors,
                               fits ONE shared OLS model, and saves its coefficients
                               (output/shared_model_coefficients.csv) plus a model-ready row-level
                               table (data/model_ready_inspections.csv) for the supporting plots
03a/03b/03c_*_forest_plot.ipynb -- pure visualization: load 02's saved results and draw one
                               forest plot each (cuisine group / keywords / income quartile)
04_grade_cutoff_bunching.ipynb  -- independent analysis, reads the cleaned data directly

Coefficients throughout are unstandardized dummy variables, so each one is a direct
difference in inspection SCORE vs. its reference (omitted) category.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "DOHMH_New_York_City_Restaurant_Inspection_Results.csv"
CLEANED_DATA_PATH = REPO_ROOT / "data" / "cleaned_inspections.csv"  # written by 01, read by 02/04
MODEL_READY_PATH = REPO_ROOT / "data" / "model_ready_inspections.csv"  # written by 02, read by 03a/b/c
MODEL_COEFFICIENTS_PATH = REPO_ROOT / "output" / "shared_model_coefficients.csv"  # written by 02, read by 03a/b/c
OUTPUT_DIR = REPO_ROOT / "output"
INCOME_CACHE_PATH = REPO_ROOT / "data" / "median_income_by_zip.csv"
CENSUS_API_KEY_PATH = REPO_ROOT / ".census_api_key"
ACS_YEAR = 2022
RARE_CATEGORY_THRESHOLD = 200  # inspection types with fewer rows get collapsed into "Other"

# Keyword indicators to search for in the restaurant name (DBA), case-insensitive.
NAME_KEYWORDS = [
    "bistro", "cafe", "grill", "deli", "diner", "kitchen", "express",
    "fast", "gourmet", "organic", "fresh", "happy", "good", "family",
    "halal", "kosher", "take out",
]

# Cuisine Description (84 raw categories) grouped into 7 broad cuisine families.
CUISINE_GROUP_MAP = {
    # Asian
    "Afghan": "Asian", "Asian": "Asian", "Bangladeshi": "Asian", "Chinese": "Asian",
    "Chinese/Cuban": "Asian", "Chinese/Japanese": "Asian", "Filipino": "Asian",
    "Indian": "Asian", "Indonesian": "Asian", "Japanese": "Asian", "Korean": "Asian",
    "Pakistani": "Asian", "Thai": "Asian", "Vietnamese/Cambodian/Malaysia": "Asian",
    # European
    "Armenian": "European", "Continental": "European", "Czech": "European",
    "Eastern European": "European", "English": "European", "French": "European",
    "German": "European", "Greek": "European", "Irish": "European", "Italian": "European",
    "Mediterranean": "European", "Pizza": "European", "Pizza/Italian": "European",
    "Polish": "European", "Portuguese": "European", "Russian": "European",
    "Scandinavian": "European", "Spanish": "European", "Tapas": "European",
    # Latin American & Caribbean
    "Brazilian": "Latin American & Caribbean", "Caribbean": "Latin American & Caribbean",
    "Chilean": "Latin American & Caribbean",
    "Latin (Cuban, Dominican, Puerto Rican, South & Central American)": "Latin American & Caribbean",
    "Mexican": "Latin American & Caribbean", "Peruvian": "Latin American & Caribbean",
    "Tex-Mex": "Latin American & Caribbean",
    # Middle Eastern & African
    "African": "Middle Eastern & African", "Egyptian": "Middle Eastern & African",
    "Ethiopian": "Middle Eastern & African", "Iranian": "Middle Eastern & African",
    "Middle Eastern": "Middle Eastern & African", "Moroccan": "Middle Eastern & African",
    "Turkish": "Middle Eastern & African",
    # American & Southern / regional US
    "American": "American & Southern", "Australian": "American & Southern",
    "Barbecue": "American & Southern", "Cajun": "American & Southern",
    "Californian": "American & Southern", "Creole": "American & Southern",
    "Creole/Cajun": "American & Southern", "Hawaiian": "American & Southern",
    "Polynesian": "American & Southern", "Southwestern": "American & Southern",
    "Soul Food": "American & Southern", "Steak": "American & Southern",
    # Grab-and-Go / quick service formats
    "Bagels/Pretzels": "Grab-and-Go", "Bakery": "Grab-and-Go",
    "Bottled beverages, including water, sodas, juices, etc.": "Grab-and-Go",
    "CafÃ©/Coffee/Tea": "Grab-and-Go", "Chicken": "Grab-and-Go",
    "Delicatessen": "Grab-and-Go", "Donuts": "Grab-and-Go",
    "Fruits/Vegetables": "Grab-and-Go", "Hamburgers": "Grab-and-Go",
    "Hotdogs": "Grab-and-Go", "Hotdogs/Pretzels": "Grab-and-Go",
    "Ice Cream, Gelato, Yogurt, Ices": "Grab-and-Go",
    "Juice, Smoothies, Fruit Salads": "Grab-and-Go", "Nuts/Confectionary": "Grab-and-Go",
    "Pancakes/Waffles": "Grab-and-Go", "Salads": "Grab-and-Go", "Sandwiches": "Grab-and-Go",
    "Sandwiches/Salads/Mixed Buffet": "Grab-and-Go", "Soups": "Grab-and-Go",
    "Soups & Sandwiches": "Grab-and-Go",
    # Other / specialty / unclassified
    "Jewish/Kosher": "Other/Specialty", "Not Listed/Not Applicable": "Other/Specialty",
    "Other": "Other/Specialty", "Seafood": "Other/Specialty", "Vegetarian": "Other/Specialty",
}

INCOME_GROUP_LABELS = [
    "Q1 (lowest income)",
    "Q2 (lower-middle income)",
    "Q3 (upper-middle income)",
    "Q4 (highest income)",
]

# Which income source the notebooks use. "real" pulls actual Census ACS 5-year median
# household income by ZCTA (requires a free API key, see get_income_by_zip below).
# "fallback" uses NYC_ZIP_INCOME_TIER_FALLBACK, a hand-built approximation from general
# knowledge of NYC neighborhood income patterns -- NOT sourced from Census/ACS data, and
# not statistically derived quartiles. It exists only so the pipeline can be demoed while
# waiting on a Census API key. Flip this to "real" (and get_income_by_zip will cache the
# real pull to data/median_income_by_zip.csv) once a key is available -- no other code
# changes are needed.
INCOME_DATA_SOURCE = "fallback"

# APPROXIMATE income tier by zip code, based on general/well-known NYC neighborhood income
# reputations (e.g., Park Slope and the Upper East Side are widely known as high-income;
# the South Bronx and Brownsville are widely known as low-income). This is NOT official
# Census/ACS data, is not derived from any statistical income measure, and individual zip
# assignments may be wrong -- treat forest plot results built on it as illustrative only,
# pending real ACS data (see INCOME_DATA_SOURCE above).
NYC_ZIP_INCOME_TIER_FALLBACK = {
    # Manhattan
    "10001": "Q3 (upper-middle income)", "10002": "Q2 (lower-middle income)",
    "10003": "Q4 (highest income)", "10004": "Q4 (highest income)",
    "10005": "Q4 (highest income)", "10006": "Q4 (highest income)",
    "10007": "Q4 (highest income)", "10009": "Q3 (upper-middle income)",
    "10010": "Q4 (highest income)", "10011": "Q4 (highest income)",
    "10012": "Q4 (highest income)", "10013": "Q4 (highest income)",
    "10014": "Q4 (highest income)", "10016": "Q4 (highest income)",
    "10017": "Q4 (highest income)", "10018": "Q3 (upper-middle income)",
    "10019": "Q3 (upper-middle income)", "10021": "Q4 (highest income)",
    "10022": "Q4 (highest income)", "10023": "Q4 (highest income)",
    "10024": "Q4 (highest income)", "10025": "Q3 (upper-middle income)",
    "10026": "Q2 (lower-middle income)", "10027": "Q2 (lower-middle income)",
    "10028": "Q4 (highest income)", "10029": "Q1 (lowest income)",
    "10030": "Q2 (lower-middle income)", "10031": "Q2 (lower-middle income)",
    "10032": "Q2 (lower-middle income)", "10033": "Q2 (lower-middle income)",
    "10034": "Q2 (lower-middle income)", "10035": "Q1 (lowest income)",
    "10036": "Q3 (upper-middle income)", "10037": "Q1 (lowest income)",
    "10038": "Q4 (highest income)", "10039": "Q1 (lowest income)",
    "10040": "Q2 (lower-middle income)", "10044": "Q3 (upper-middle income)",
    "10111": "Q4 (highest income)", "10112": "Q4 (highest income)",
    "10128": "Q4 (highest income)", "10152": "Q4 (highest income)",
    "10153": "Q4 (highest income)", "10154": "Q4 (highest income)",
    "10165": "Q4 (highest income)", "10166": "Q4 (highest income)",
    "10167": "Q4 (highest income)", "10168": "Q4 (highest income)",
    "10169": "Q4 (highest income)", "10170": "Q4 (highest income)",
    "10171": "Q4 (highest income)", "10172": "Q4 (highest income)",
    "10173": "Q4 (highest income)", "10174": "Q4 (highest income)",
    "10175": "Q4 (highest income)", "10199": "Q4 (highest income)",
    "10280": "Q4 (highest income)", "10281": "Q4 (highest income)",
    "10282": "Q4 (highest income)",
    # Bronx
    "10451": "Q1 (lowest income)", "10452": "Q1 (lowest income)",
    "10453": "Q1 (lowest income)", "10454": "Q1 (lowest income)",
    "10455": "Q1 (lowest income)", "10456": "Q1 (lowest income)",
    "10457": "Q1 (lowest income)", "10458": "Q1 (lowest income)",
    "10459": "Q1 (lowest income)", "10460": "Q1 (lowest income)",
    "10461": "Q2 (lower-middle income)", "10462": "Q2 (lower-middle income)",
    "10463": "Q3 (upper-middle income)", "10464": "Q3 (upper-middle income)",
    "10465": "Q2 (lower-middle income)", "10466": "Q2 (lower-middle income)",
    "10467": "Q1 (lowest income)", "10468": "Q1 (lowest income)",
    "10469": "Q2 (lower-middle income)", "10470": "Q2 (lower-middle income)",
    "10471": "Q4 (highest income)", "10472": "Q1 (lowest income)",
    "10473": "Q1 (lowest income)", "10474": "Q1 (lowest income)",
    "10475": "Q2 (lower-middle income)",
    # Brooklyn
    "11201": "Q4 (highest income)", "11203": "Q2 (lower-middle income)",
    "11204": "Q2 (lower-middle income)", "11205": "Q3 (upper-middle income)",
    "11206": "Q2 (lower-middle income)", "11207": "Q1 (lowest income)",
    "11208": "Q1 (lowest income)", "11209": "Q3 (upper-middle income)",
    "11210": "Q2 (lower-middle income)", "11211": "Q4 (highest income)",
    "11212": "Q1 (lowest income)", "11213": "Q2 (lower-middle income)",
    "11214": "Q2 (lower-middle income)", "11215": "Q4 (highest income)",
    "11216": "Q3 (upper-middle income)", "11217": "Q4 (highest income)",
    "11218": "Q3 (upper-middle income)", "11219": "Q2 (lower-middle income)",
    "11220": "Q2 (lower-middle income)", "11221": "Q2 (lower-middle income)",
    "11222": "Q3 (upper-middle income)", "11223": "Q2 (lower-middle income)",
    "11224": "Q1 (lowest income)", "11225": "Q2 (lower-middle income)",
    "11226": "Q2 (lower-middle income)", "11228": "Q3 (upper-middle income)",
    "11229": "Q2 (lower-middle income)", "11230": "Q2 (lower-middle income)",
    "11231": "Q4 (highest income)", "11232": "Q2 (lower-middle income)",
    "11233": "Q1 (lowest income)", "11234": "Q2 (lower-middle income)",
    "11235": "Q2 (lower-middle income)", "11236": "Q2 (lower-middle income)",
    "11237": "Q2 (lower-middle income)", "11238": "Q4 (highest income)",
    "11239": "Q1 (lowest income)", "11249": "Q4 (highest income)",
    # Queens
    "11101": "Q3 (upper-middle income)", "11102": "Q3 (upper-middle income)",
    "11103": "Q3 (upper-middle income)", "11104": "Q3 (upper-middle income)",
    "11105": "Q3 (upper-middle income)", "11106": "Q3 (upper-middle income)",
    "11109": "Q4 (highest income)", "11354": "Q2 (lower-middle income)",
    "11355": "Q2 (lower-middle income)", "11356": "Q2 (lower-middle income)",
    "11357": "Q3 (upper-middle income)", "11358": "Q3 (upper-middle income)",
    "11360": "Q3 (upper-middle income)", "11361": "Q3 (upper-middle income)",
    "11362": "Q4 (highest income)", "11363": "Q4 (highest income)",
    "11364": "Q3 (upper-middle income)", "11365": "Q3 (upper-middle income)",
    "11366": "Q3 (upper-middle income)", "11367": "Q3 (upper-middle income)",
    "11368": "Q1 (lowest income)", "11369": "Q2 (lower-middle income)",
    "11370": "Q2 (lower-middle income)", "11372": "Q2 (lower-middle income)",
    "11373": "Q2 (lower-middle income)", "11374": "Q3 (upper-middle income)",
    "11375": "Q4 (highest income)", "11377": "Q2 (lower-middle income)",
    "11378": "Q2 (lower-middle income)", "11379": "Q3 (upper-middle income)",
    "11385": "Q2 (lower-middle income)", "11411": "Q3 (upper-middle income)",
    "11412": "Q2 (lower-middle income)", "11413": "Q2 (lower-middle income)",
    "11414": "Q3 (upper-middle income)", "11415": "Q3 (upper-middle income)",
    "11416": "Q2 (lower-middle income)", "11417": "Q2 (lower-middle income)",
    "11418": "Q2 (lower-middle income)", "11419": "Q2 (lower-middle income)",
    "11420": "Q2 (lower-middle income)", "11421": "Q2 (lower-middle income)",
    "11422": "Q3 (upper-middle income)", "11423": "Q2 (lower-middle income)",
    "11426": "Q3 (upper-middle income)", "11427": "Q3 (upper-middle income)",
    "11428": "Q3 (upper-middle income)", "11429": "Q3 (upper-middle income)",
    "11432": "Q2 (lower-middle income)", "11433": "Q1 (lowest income)",
    "11434": "Q2 (lower-middle income)", "11435": "Q2 (lower-middle income)",
    "11436": "Q1 (lowest income)", "11691": "Q1 (lowest income)",
    "11692": "Q1 (lowest income)", "11693": "Q2 (lower-middle income)",
    "11694": "Q3 (upper-middle income)", "11697": "Q4 (highest income)",
    # Staten Island
    "10301": "Q2 (lower-middle income)", "10302": "Q2 (lower-middle income)",
    "10303": "Q2 (lower-middle income)", "10304": "Q2 (lower-middle income)",
    "10305": "Q3 (upper-middle income)", "10306": "Q3 (upper-middle income)",
    "10307": "Q4 (highest income)", "10308": "Q3 (upper-middle income)",
    "10309": "Q4 (highest income)", "10310": "Q2 (lower-middle income)",
    "10312": "Q4 (highest income)", "10314": "Q3 (upper-middle income)",
}


def load_cleaned(usecols=None):
    """Read data/cleaned_inspections.csv (written by 01_data_pull_clean.ipynb)."""
    df = pd.read_csv(CLEANED_DATA_PATH, usecols=usecols, dtype={"zipcode": str})
    print(f"Loaded {len(df):,} rows from {CLEANED_DATA_PATH}")
    return df


def add_cuisine_group(df, cuisine_col="cuisine_description"):
    df["cuisine_group"] = df[cuisine_col].map(CUISINE_GROUP_MAP)
    unmapped = df.loc[df["cuisine_group"].isna(), cuisine_col].unique()
    assert len(unmapped) == 0, f"Unmapped cuisine categories found: {unmapped}"
    print("Cuisine group counts:")
    print(df["cuisine_group"].value_counts())
    return df


def add_keyword_indicators(df, name_col="dba"):
    name_lower = df[name_col].str.lower()
    keyword_cols = []
    for kw in NAME_KEYWORDS:
        col = f"name_{kw.replace(' ', '_')}"
        df[col] = name_lower.str.contains(kw, regex=False).astype(int)
        keyword_cols.append(col)
    return df, keyword_cols


def clean_zipcode(df, zip_col="zipcode"):
    """Coerce a raw (often float-like) zip code column to a clean 5-digit string, dropping bad rows."""
    df = df.copy()
    df[zip_col] = pd.to_numeric(df[zip_col], errors="coerce")
    df = df.dropna(subset=[zip_col])
    df[zip_col] = df[zip_col].astype(int).astype(str).str.zfill(5)
    return df


def _census_api_key():
    key = os.environ.get("CENSUS_API_KEY")
    if key:
        return key
    if CENSUS_API_KEY_PATH.exists():
        return CENSUS_API_KEY_PATH.read_text().strip()
    raise RuntimeError(
        "No Census API key found. Get a free key at "
        "https://api.census.gov/data/key_signup.html, then either set the "
        f"CENSUS_API_KEY environment variable or save the key to {CENSUS_API_KEY_PATH}."
    )


def get_income_by_zip():
    """Median household income by NY zip code (ACS 5-year, ZCTA-level), grouped into quartiles.

    Pulled once from the public Census ACS API and cached to data/median_income_by_zip.csv
    so later notebook runs don't repeat the API call.
    """
    if INCOME_CACHE_PATH.exists():
        income = pd.read_csv(INCOME_CACHE_PATH, dtype={"zipcode": str})
    else:
        url = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"
        params = {
            "get": "NAME,B19013_001E",
            "for": "zip code tabulation area:*",
            "in": "state:36",
            "key": _census_api_key(),
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
        income = pd.DataFrame(rows[1:], columns=rows[0])
        income = income.rename(columns={
            "B19013_001E": "median_income",
            "zip code tabulation area": "zipcode",
        })[["zipcode", "median_income"]]
        income["median_income"] = pd.to_numeric(income["median_income"], errors="coerce")
        income = income[income["median_income"] > 0]  # Census codes missing data as negative sentinels
        INCOME_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        income.to_csv(INCOME_CACHE_PATH, index=False)
        print(f"Pulled median household income for {len(income)} NY ZCTAs from Census ACS "
              f"{ACS_YEAR} 5-year estimates; cached to {INCOME_CACHE_PATH}")

    income = income.copy()
    income["income_group"] = pd.qcut(income["median_income"], q=4, labels=INCOME_GROUP_LABELS)
    return income


def get_income_by_zip_fallback():
    """APPROXIMATE income tier by zip, from NYC_ZIP_INCOME_TIER_FALLBACK. NOT official
    Census data -- see the warning on that dict. Used only while INCOME_DATA_SOURCE ==
    "fallback"."""
    print(
        "WARNING: using an APPROXIMATE income-tier table built from general knowledge of "
        "NYC neighborhood income patterns, NOT official Census ACS data (see "
        "NYC_ZIP_INCOME_TIER_FALLBACK in utils.py). Treat income results as illustrative "
        "only until INCOME_DATA_SOURCE is switched to \"real\"."
    )
    income = pd.DataFrame({
        "zipcode": list(NYC_ZIP_INCOME_TIER_FALLBACK.keys()),
        "income_group": list(NYC_ZIP_INCOME_TIER_FALLBACK.values()),
    })
    income["income_group"] = pd.Categorical(
        income["income_group"], categories=INCOME_GROUP_LABELS, ordered=True
    )
    return income


def get_income_data():
    """Single entry point the notebooks call for income-by-zip data. Switches between the
    real Census ACS pull and the approximate fallback table based on INCOME_DATA_SOURCE,
    so notebooks don't need to change when the data source changes."""
    if INCOME_DATA_SOURCE == "real":
        return get_income_by_zip()
    elif INCOME_DATA_SOURCE == "fallback":
        return get_income_by_zip_fallback()
    raise ValueError(f"Unknown INCOME_DATA_SOURCE: {INCOME_DATA_SOURCE!r}")


def fit_shared_model(y, X, cluster_groups=None):
    """Fit the shared OLS model. If cluster_groups is given (e.g. restaurant chain name),
    standard errors are clustered on it, so repeated inspections from the same restaurant
    or chain don't count as independent observations and understate the true uncertainty."""
    X = sm.add_constant(X.astype(float))
    if cluster_groups is not None:
        model = sm.OLS(y.astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": cluster_groups})
    else:
        model = sm.OLS(y.astype(float), X).fit()
    print(model.summary())
    return model


def save_model_coefficients(model, save_path=MODEL_COEFFICIENTS_PATH):
    """Save a tidy (term, coef, ci_low, ci_high, pvalue) table from a fitted model, so
    downstream notebooks can build forest plots without refitting the model."""
    ci = model.conf_int()
    coef_table = pd.DataFrame({
        "term": model.params.index,
        "coef": model.params.values,
        "ci_low": ci[0].values,
        "ci_high": ci[1].values,
        "pvalue": model.pvalues.values,
    })
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    coef_table.to_csv(save_path, index=False)
    print(f"Saved model coefficients to {save_path}")
    return coef_table


def load_model_coefficients():
    """Load the coefficient table saved by 02_shared_model.ipynb."""
    return pd.read_csv(MODEL_COEFFICIENTS_PATH)


def forest_plot(coef_table, prefix, label_fn, title, save_path):
    """Draw and save a forest plot of the rows in coef_table whose term matches `prefix`.

    coef_table: a DataFrame with columns [term, coef, ci_low, ci_high], e.g. from
    save_model_coefficients / load_model_coefficients. Coefficients are unstandardized, so
    each one is a direct difference in inspection SCORE relative to the reference
    (omitted) category.
    """
    rows = coef_table[coef_table["term"].str.startswith(prefix)]

    plot_df = pd.DataFrame({
        "label": [label_fn(t) for t in rows["term"]],
        "coef": rows["coef"].values,
        "ci_low": rows["ci_low"].values,
        "ci_high": rows["ci_high"].values,
    })
    plot_df["abs_coef"] = plot_df["coef"].abs()
    plot_df = plot_df.sort_values("abs_coef", ascending=True).reset_index(drop=True)

    fig_height = max(4, 0.45 * len(plot_df))
    fig, ax = plt.subplots(figsize=(9, fig_height))

    y_pos = np.arange(len(plot_df))
    colors = ["#1f77b4" if c >= 0 else "#d62728" for c in plot_df["coef"]]

    ax.errorbar(
        plot_df["coef"], y_pos,
        xerr=[plot_df["coef"] - plot_df["ci_low"], plot_df["ci_high"] - plot_df["coef"]],
        fmt="o", markersize=6, ecolor="gray", elinewidth=1.4, capsize=4,
        color="black", zorder=3,
    )
    ax.scatter(plot_df["coef"], y_pos, c=colors, s=45, zorder=4)

    ax.axvline(0, color="black", linestyle="--", linewidth=1, zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["label"], fontsize=10)
    ax.set_xlabel("Difference in inspection SCORE vs. reference group (95% CI)")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    print(f"Saved forest plot to {save_path}")
    plt.show()
    return plot_df
