"""Generate the Brazil EDA notebooks: layperson glossary + narrated insights.

Why generate notebooks from a Python script instead of hand-editing .ipynb
files? Two reasons: (1) the four notebooks share a large glossary and a
common "safe bootstrap" cell (see BOOT_PATH below) that would otherwise drift
out of sync across files, and (2) .ipynb is JSON — hand-editing it in a text
editor is error-prone. Edit this file, then run it, rather than editing the
notebooks directly:

    python scripts/build_eda_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

# Shared variable dictionary, shown at the top of every notebook so a reader
# who opens notebook 3 first (skipping 1 and 2) is not lost.
VAR_DICT = """
## Variable dictionary (read before the charts)

| Column / label in charts | Plain-English meaning | How to read values |
|---|---|---|
| `target_dropout_rate` / **Dropout rate (%)** | Official INEP "taxa de abandono": share of students who **stopped attending** during the school year, after the Census reference date | 0 = nobody left; 5 = 5% left. Higher = worse |
| `year` | School Census / attainment-rate year | Years present in the mart (currently 2018-2025) |
| `school_id` (`CO_ENTIDADE`) | Unique school code (INEP) | Join key between Census and attainment-rate data |
| `state_code` | Brazilian state abbreviation ("UF" = Unidade Federativa) | e.g. SP, BA, AM |
| `municipality_id` | IBGE municipality code | Geographic context |
| `admin_dependency_type` | Administrative network | 1=Federal, 2=State, 3=Municipal, 4=Private |
| `location_type` | School location type | 1=Urban, 2=Rural |
| `is_rural` | 1 if rural school | Shortcut of `location_type == 2` |
| `is_public` | 1 if public network (federal/state/municipal) | 0 = private |
| `has_water` | Has potable / public water | 1=yes, 0=no |
| `has_electricity` | Connected to public electricity | 1=yes, 0=no |
| `has_sewage` | Public sewage connection | 1=yes, 0=no |
| `has_internet` / `has_internet_for_students` | Internet at school / specifically available to students | 1=yes, 0=no |
| `has_library` | Library / reading room | 1=yes, 0=no |
| `has_computer_lab` | Computer lab | 1=yes, 0=no |
| `has_sports_court` | Sports court | 1=yes, 0=no |
| `enrollment_basic_ed` | Enrollment count (basic education total, when available) | Larger = bigger school |
| `enrollment_level` | Enrollment used for this level (Fundamental or Medio) | Filter requires >= 20 students |
| `teacher_count_basic_ed` | Number of teachers (basic education) | Staffing intensity |
| `student_teacher_ratio` | Students / teachers | Higher often means more crowded classes |
| `avg_class_size` | Enrollment / number of classes for this level | Higher = larger classes |
| `overage_enrollment_share` | Share of students above the expected age for this level | A known proxy for grade-age distortion, itself a dropout correlate |
| `dropout_rate_lag1` / `_lag2` | This school's own dropout rate 1 / 2 years earlier | Missing (NaN) for a school's first year in the panel |
| `dropout_rate_3yr_avg` / `_trend` | Trailing 3-year average / (lag1 - lag2) | Trend > 0 means the rate has been rising |
| `has_history` | 1 if the school has at least one prior year on record | 0 = first appearance in the 2018-2025 window |
| `approval_rate_lag1` / `failure_rate_lag1` | This school's own approval / grade-repetition rate, prior year | A rising failure rate is a known leading indicator of dropout |
| `municipal_dropout_rate_lag1` / `state_dropout_rate_lag1` | Average dropout rate across the municipality / state, prior year | A spatial-lag signal for local shocks a single school's history cannot capture |
| `is_covid_year` | 1 for school years 2020-2021 | INEP suspended normal grade-progression rules in these years |
| `risk_band` | low / moderate / high | Relative triage label inside the level |
| `high_risk` | 1 if school is in the elevated-risk group | Binary flag used for the triage/ranking metrics |
| **MAE / RMSE / R2** | Model error metrics (later notebooks / model folder) | Lower MAE/RMSE better; R2 closer to 1 better |

### Acronyms
| Acronym | Meaning |
|---|---|
| **INEP** | Brazil's federal education-statistics agency |
| **School Census** ("Censo Escolar") | Annual school census (structure + enrollment) |
| **School Attainment Rates** ("Taxas de Rendimento") | Official approval / failure / dropout rates |
| **EDA** | Exploratory Data Analysis |
| **UF** | Unidade Federativa (Brazilian state) |
| **Fundamental** | Ensino Fundamental: grades 1-9, roughly ages 6-14 |
| **Medio** | Ensino Medio: grades 10-12, Brazil's upper-secondary stage |
| **EJA** | Educacao de Jovens e Adultos (adult / youth education track) |
"""


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def write_nb(path: Path, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python (.venv)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    print("wrote", path)


# Split bootstrap: Path.cwd()/exists/resolve can hang on Desktop/OneDrive kernels
# (see src/utils.py for the same note applied to the library code).
BOOT_PATH = '''
# Cell A - project path only (no Path.cwd / exists / resolve)
import sys
ROOT = r"C:\\Users\\User\\Desktop\\Projeto Living Stone Foundation"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
'''

BOOT_IMPORTS = '''
# Cell B - imports (first run can take a minute for pandas/seaborn)
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

FEATURE_LABELS = {
    "is_rural": "Rural school (1=yes)",
    "is_public": "Public network (1=yes)",
    "has_internet": "Has internet (1=yes)",
    "has_internet_for_students": "Has internet for students (1=yes)",
    "has_computer_lab": "Has computer lab (1=yes)",
    "has_sports_court": "Has sports court (1=yes)",
    "has_library": "Has library (1=yes)",
    "has_water": "Has water (1=yes)",
    "has_electricity": "Has electricity (1=yes)",
    "has_sewage": "Has sewage (1=yes)",
    "enrollment_level": "Enrollment (this level)",
    "enrollment_basic_ed": "Basic-ed enrollment (total)",
    "teacher_count_basic_ed": "Number of teachers",
    "student_teacher_ratio": "Students per teacher",
    "avg_class_size": "Average class size",
    "overage_enrollment_share": "Share of over-age students",
    "admin_dependency_type": "Admin network code",
    "location_type": "Urban/rural code",
    "target_dropout_rate": "Dropout rate (%)",
    "dropout_rate_lag1": "Dropout rate, prior year",
    "dropout_rate_lag2": "Dropout rate, 2 years prior",
    "dropout_rate_3yr_avg": "Dropout rate, 3-year trailing avg",
    "dropout_rate_trend": "Dropout rate trend",
    "approval_rate_lag1": "Approval rate, prior year",
    "failure_rate_lag1": "Failure rate, prior year",
    "municipal_dropout_rate_lag1": "Municipality's dropout rate, prior year",
    "state_dropout_rate_lag1": "State's dropout rate, prior year",
}
print("Imports OK")
'''


def bootstrap_cells(*extra_import_lines: str) -> list:
    """OneDrive-safe setup: path cell, then imports cell, then brief note."""
    extra = "\n".join(extra_import_lines)
    imports = BOOT_IMPORTS
    if extra:
        imports = BOOT_IMPORTS.replace('print("Imports OK")', extra + '\nprint("Imports OK")')
    return [
        code(BOOT_PATH),
        md(
            """
### What this cell did
Added the project folder to Python's import path using a **fixed string** (no `Path.cwd()` / `exists` / `resolve`). Those path checks can freeze kernels on Desktop/OneDrive.
"""
        ),
        code(imports),
        md(
            """
### What this cell did
Loaded charting/table libraries. The **first** run can take ~30-90s while pandas/seaborn warm up - that is normal, not a freeze on `import sys`.
"""
        ),
    ]


def nb_validation():
    cells = [
        md(
            f"""
# Brazil data validation - INEP School Census + Attainment Rates

**Living Stone Foundation - Applied Data Lab**

**Who this notebook is for:** readers who did **not** build the project.
**Goal:** prove that the population (Brazilian basic-education **schools**) matches the labels we train on (official INEP **dropout rates**).

{VAR_DICT}

### How to read this notebook
1. Run cells top to bottom (`Shift+Enter`).
2. After every code cell, read the **Insight** markdown - that is the takeaway, not the raw JSON.
"""
        ),
        *bootstrap_cells(
            "import json",
            "from src.eda import compare_levels, validation_json_path",
        ),
        md("## 1. Official validation artifact"),
        code(
            """
path = validation_json_path()
payload = json.loads(path.read_text(encoding='utf-8'))
print('File:', path)
print('Source:', payload.get('source'))
print('Dropout-rate definition:', payload.get('definition_dropout_rate'))
print('Grain:', payload.get('grain'))
print('Years:', payload.get('years'))
print('Rows / schools:', payload.get('rows'), '/', payload.get('schools'))
print('Coverage:', json.dumps(payload.get('coverage'), indent=2))
print('Student-level labels available?', payload.get('student_level_labeled_dropout_available'))
"""
        ),
        md(
            """
### How to read this output
- **Source** should cite INEP School Attainment Rates (official), not a homemade proxy.
- **Definition** means: students who *left attending* during the year - not "failed the year" and not "never enrolled".
- **Grain = school x year** means each measurement is about a **school**, not a named student.
- **Coverage**
  - `fundamental_non_null_pct` near ~90%+: most schools have a Fundamental dropout-rate number.
  - `medio_non_null_pct` much lower (~20%+): only schools that offer Ensino Medio report that rate.
  - Means: Medio dropout rate is typically **higher** than Fundamental (e.g. ~3% vs ~1%).
- **Student-level labels available? -> False**: we cannot honestly build a student scorer with these open files.

### Insight (decision for the project)
Population and labels are aligned for a **Brazil school-level** early-warning system. They are **not** aligned with the old Kaggle higher-education student dataset - that is why it was removed (see `docs/scope_revision.md`).
"""
        ),
        md("## 2. Side-by-side mart summary (Fundamental vs Medio)"),
        code(
            """
summary = compare_levels()
# Friendlier column names for readers
pretty = summary.rename(columns={
    'level': 'Education level',
    'rows': 'School-year rows',
    'schools': 'Unique schools',
    'mean_dropout_rate': 'Mean dropout rate (%)',
    'median_dropout_rate': 'Median dropout rate (%)',
    'p90_dropout_rate': '90th percentile dropout rate (%)',
    'public_share': 'Share public schools',
    'rural_share': 'Share rural schools',
    'share_with_history': 'Share with prior-year history',
})
display(pretty)
"""
        ),
        md(
            """
### How to read this table
Each row is one education level after joining Census features to official dropout rates and keeping schools with enough enrollment (>= 20).

| Column | Takeaway question |
|---|---|
| School-year rows | Do we have enough data to model? |
| Mean vs median dropout rate | Is the outcome rare/skewed? (median near 0 with mean > median means many zeros + a right tail) |
| 90th percentile | What does a "high" school look like in this level? |
| Public / rural shares | What kind of schools dominate the sample? |
| Share with prior-year history | How much of the data can use the lag/history features (see notebooks 02/03)? |

### Insight
If **Medio** shows higher mean dropout rate and a heavier right tail than **Fundamental**, training **two models** (and two Streamlit apps) is justified: the risk regimes are different. A single pooled model would mix two different populations.
"""
        ),
    ]
    write_nb(NB_DIR / "01_eda_brazil_data_validation.ipynb", cells)


def nb_level(level: str, filename: str, title: str):
    level_title = "Ensino Fundamental" if level == "fundamental" else "Ensino Medio"
    cells = [
        md(
            f"""
# {title}

**Living Stone Foundation - Applied Data Lab**

**Audience:** non-authors of the project.
**Level in focus:** **{level_title}** (`{level}`).
**Question this notebook answers:** What does official school dropout look like, where is it higher, and which school characteristics - including the school's *own dropout history* - move together with it?

{VAR_DICT}
"""
        ),
        *bootstrap_cells(
            "from src.eda import load_level_mart, missingness_table, summary_by_state",
        ),
        code(
            f"""
df = load_level_mart('{level}')
print('Rows (school x year):', df.shape[0])
print('Columns:', df.shape[1])
print('Unique schools:', df['school_id'].nunique())
print('Years:', sorted(df['year'].dropna().astype(int).unique().tolist()))
display(df.head())
"""
        ),
        md(
            f"""
### How to read this preview
- Each **row** = one school in one year for **{level_title}**.
- `target_dropout_rate` is the official dropout % we will try to predict later.
- Feature columns (`has_*`, `is_*`, enrollment, teachers) come from the **School Census**.
- `dropout_rate_lag1` and related columns come from the school's **own prior-year** attainment history (see section 4).

### Insight
You are looking at a **school risk table**, not a student list. Interventions suggested by this project are about **which schools to prioritize**, not which child to call.
"""
        ),
        md("## 1. Distribution of the dropout rate"),
        code(
            """
display(df['target_dropout_rate'].describe().rename('Dropout rate (%)'))
fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(df['target_dropout_rate'], bins=40, ax=ax, color='#1f4e79')
ax.set_title('How common is each dropout rate?')
ax.set_xlabel('Official dropout rate (%)')
ax.set_ylabel('Number of school-year rows')
plt.tight_layout()
plt.show()
print('Share of rows with 0% dropout:', float((df['target_dropout_rate'] == 0).mean()))
print('Share of rows with dropout >= 5%:', float((df['target_dropout_rate'] >= 5).mean()))
"""
        ),
        md(
            f"""
### How to read this chart
- **X-axis:** dropout rate in percent (0-100 scale, usually concentrated near 0).
- **Y-axis:** how many school-year observations fall in that bin.
- A tall bar at 0% means many schools reported **no dropout** that year.

### Insight for {level_title}
Dropout is a **rare / skewed** school outcome. That is normal, but it has a sharp practical consequence covered in `docs/validity_and_english_revision.md`: because so many schools sit at exactly 0%, the MAE of a model that always predicts the *average* dropout rate is a surprisingly strong baseline - and the first version of this project's models did not beat it. Judge any model here against that baseline, not in isolation.
"""
        ),
        md("## 2. Data quality (missing values)"),
        code(
            """
miss = missingness_table(df).head(15).rename(columns={
    'pct_missing': 'Share missing',
    'n_missing': 'Count missing',
    'dtype': 'Data type',
})
display(miss)
"""
        ),
        md(
            """
### How to read this table
- **Share missing = 0.000** means the column is fully filled in this mart.
- History columns (`dropout_rate_lag1`, etc.) are expected to have some missingness - that is a school's first year in the panel, not a data-quality defect. `has_history` distinguishes the two cases explicitly (0 = no track record yet).

### Insight
For modeling we already kept every row with a non-null official dropout rate. If a *feature* is heavily missing, the pipeline's imputer fills it (median for numeric columns) rather than dropping the row - dropping would have silently discarded exactly the schools with the least track record (see `src/data_loader.py`).
"""
        ),
        md("## 3. Where is dropout higher? (by state / UF)"),
        code(
            """
g = summary_by_state(df, min_n=30).rename(columns={
    'state_code': 'State (UF)',
    'n': 'School-year rows',
    'mean_dropout_rate': 'Mean dropout rate (%)',
    'median_dropout_rate': 'Median dropout rate (%)',
})
display(g.head(15))
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(data=g.head(12), x='State (UF)', y='Mean dropout rate (%)', ax=ax, color='#9c2a2a')
ax.set_title('States with highest mean school dropout rate (only UF with >= 30 rows)')
ax.set_xlabel('State (UF)')
ax.set_ylabel('Mean dropout rate (%)')
plt.tight_layout()
plt.show()
"""
        ),
        md(
            f"""
### How to read this chart
- Each bar is a **state (UF)**.
- Height = average school dropout rate among schools of **{level_title}** in that state (with enough sample size).
- We require **n >= 30** so tiny samples do not create fake "worst state" headlines.

### Insight
Use this as a **map of attention**, not as a final ranking for newspapers:
- Different states have different networks (more municipal vs state schools).
- Coverage and school size differ.
- Still, persistently high bars suggest where network managers may want deeper local diagnostics. The trained model also uses a *lagged* version of this exact signal (`state_dropout_rate_lag1`) as a feature - see section 4.
"""
        ),
        md("## 4. Does a school's own dropout history predict this year's rate?"),
        code(
            """
hist_cols = [c for c in ['dropout_rate_lag1', 'dropout_rate_lag2', 'dropout_rate_3yr_avg',
                          'approval_rate_lag1', 'failure_rate_lag1',
                          'municipal_dropout_rate_lag1', 'state_dropout_rate_lag1']
             if c in df.columns]
corr = df[hist_cols + ['target_dropout_rate']].corr(numeric_only=True)['target_dropout_rate'].drop('target_dropout_rate')
corr_named = corr.rename(index=FEATURE_LABELS).sort_values()
display(corr_named.rename('Correlation with same-year dropout rate').to_frame())
ax = corr_named.plot(kind='barh', figsize=(9, 4), color='#c65911')
ax.set_title('History features vs. same-year dropout rate')
ax.set_xlabel('Correlation (-1 to +1)')
plt.tight_layout()
plt.show()
print('Share of rows with at least one prior year on record:', float(df['has_history'].mean()))
"""
        ),
        md(
            f"""
### How to read this chart
Same idea as the state chart above, but for the school's **own trailing history** instead of a same-year snapshot.

### Insight for {level_title}
This is the single biggest change in v3.0.0 of this project (see `docs/validity_and_english_revision.md`): earlier versions never computed these columns at all, so the model never got to use a school's own track record. `dropout_rate_lag1` is typically the single strongest feature in the entire dataset - stronger than any Census infrastructure variable - which matches the intuition that dropout is a slow-moving, largely persistent school-level condition, not something that appears out of nowhere each year.
"""
        ),
        md("## 5. Which school traits move with dropout? (simple correlations)"),
        code(
            """
feat = ['is_rural', 'is_public', 'has_internet', 'has_computer_lab', 'has_sports_court',
        'has_library', 'enrollment_level', 'teacher_count_basic_ed']
feat = [c for c in feat if c in df.columns]
tmp = df[feat + ['target_dropout_rate']].copy()
if 'teacher_count_basic_ed' in tmp.columns and 'enrollment_level' in tmp.columns:
    tmp['student_teacher_ratio'] = tmp['enrollment_level'] / tmp['teacher_count_basic_ed'].replace({0: pd.NA})
    feat = feat + ['student_teacher_ratio']
corr = tmp[feat + ['target_dropout_rate']].corr(numeric_only=True)['target_dropout_rate'].drop('target_dropout_rate')
corr_named = corr.rename(index=FEATURE_LABELS).sort_values()
display(corr_named.rename('Correlation with dropout rate').to_frame())
ax = corr_named.plot(kind='barh', figsize=(9, 5), color='#2a9d8f')
ax.set_title('Simple one-variable associations with dropout (same-year Census features)')
ax.set_xlabel('Correlation (-1 to +1). Positive = higher feature <-> higher dropout')
plt.tight_layout()
plt.show()
"""
        ),
        md(
            f"""
### How to read this chart
- Each bar is **one school characteristic**.
- Axis goes from -1 to +1:
  - **Positive** correlation: when the feature is higher/true, dropout tends to be **higher**.
  - **Negative** correlation: when the feature is higher/true, dropout tends to be **lower**.
- Example readings:
  - `Rural school (1=yes)` positive means rural schools show higher dropout **on average** in this table.
  - `Has internet (1=yes)` negative means schools with internet show lower dropout **on average**.

### Critical caveats (please keep)
1. Correlation is **not causation** (internet does not automatically "cause" retention).
2. This is **univariate** (one variable at a time). The ML model can reorder importance when variables are combined, and typically the history features from section 4 dominate all of these once combined - see `models/{level}/figures/global_importance_top.csv`.
3. Compare with the trained model's actual feature importances in section 6.

### Insight for {level_title}
Write down the 3 strongest bars (positive and negative) here, then compare them against section 4's history features and section 6's model importances. Same-year Census features are real, legitimate predictors (see `docs/data_card.md` for why they are not leakage) - they are just weaker than the school's own history.
"""
        ),
        md("## 6. Compare with the trained model's actual drivers"),
        code(
            f"""
imp_path = Path(ROOT) / 'models' / '{level}' / 'figures' / 'global_importance_top.csv'
try:
    imp = pd.read_csv(imp_path)
    imp['feature_label'] = imp['feature'].map(FEATURE_LABELS).fillna(imp['feature'])
    display(imp[['feature_label', 'importance']].head(12))
    ax = imp.head(10).set_index('feature_label')['importance'].plot(kind='barh', color='#1f4e79', figsize=(9, 4))
    ax.invert_yaxis()
    ax.set_title('Model feature importance (trained {level} model)')
    ax.set_xlabel('Importance (higher = model relies on this more)')
    plt.tight_layout(); plt.show()
except FileNotFoundError:
    print('Train first: python -m src.train --level {level}')
"""
        ),
        md(
            f"""
### How to read model importance
Unlike correlation, this answers: **"When the trained model predicts dropout, which inputs move the prediction most?"**

### Insight
If correlation and model importance **agree**, the narrative is robust.
If they **disagree**, prefer the model list for triage explanations, but still show both transparently - disagreement often means confounding (variables overlapping, e.g. rural schools also tending to have lower enrollment).

### Closing takeaway ({level_title})
1. Outcome = official school dropout %, skewed toward zero.
2. A school's **own dropout history** is the strongest available signal - new in v3.0.0.
3. Geography (state, and its own lagged average) highlights uneven and locally-correlated risk.
4. Structural/staffing traits associate with risk - useful for prioritization, not blame.
5. This notebook alone does **not** authorize student-level interventions, and no model here should be trusted until it is checked against a trivial baseline (see `docs/validity_and_english_revision.md`).
"""
        ),
    ]
    write_nb(NB_DIR / filename, cells)


def nb_compare():
    cells = [
        md(
            f"""
# Compare Ensino Fundamental vs Ensino Medio

**Living Stone Foundation - Applied Data Lab**

**Audience:** team leads / reviewers who need a clear reason for **two models and two Streamlit apps**.

**Core question:** Do dropout levels and school-trait associations differ enough between Fundamental and Medio that a single pooled Brazil model would be misleading?

{VAR_DICT}
"""
        ),
        *bootstrap_cells(
            "from src.eda import compare_levels, load_level_mart",
        ),
        code(
            """
summary = compare_levels().rename(columns={
    'level': 'Education level',
    'rows': 'School-year rows',
    'schools': 'Unique schools',
    'mean_dropout_rate': 'Mean dropout rate (%)',
    'median_dropout_rate': 'Median dropout rate (%)',
    'p90_dropout_rate': '90th percentile (%)',
    'public_share': 'Share public',
    'rural_share': 'Share rural',
    'share_with_history': 'Share with prior-year history',
})
display(summary)
fund = load_level_mart('fundamental')
med = load_level_mart('medio')
print('Loaded Fundamental rows:', len(fund), '| Medio rows:', len(med))
"""
        ),
        md(
            """
### How to read this summary
Compare the two rows:
- **Mean / 90th percentile dropout rate** - is Medio systematically "harder"?
- **Row counts** - Medio has fewer schools (only those offering upper secondary).
- **Public / rural shares** - different school mixes can change which features matter.

### Insight
A higher Medio mean with fewer schools already suggests **different operating regimes**. That is necessary (but not sufficient) evidence for separate models - the next charts check the shape of risk and the correlates.
"""
        ),
        md("## 1. Outcome shape: boxplots side by side"),
        code(
            """
fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
sns.boxplot(data=fund, y='target_dropout_rate', ax=ax[0], color='#4c78a8')
ax[0].set_title('Fundamental - dropout rate (%)')
ax[0].set_ylabel('Official dropout rate (%)')
sns.boxplot(data=med, y='target_dropout_rate', ax=ax[1], color='#e76f51')
ax[1].set_title('Medio - dropout rate (%)')
ax[1].set_ylabel('')
plt.suptitle('Are the risk distributions similar?', y=1.02)
plt.tight_layout()
plt.show()
print('Mean Fundamental: {:.2f}%'.format(fund['target_dropout_rate'].mean()))
print('Mean Medio:       {:.2f}%'.format(med['target_dropout_rate'].mean()))
print('P90 Fundamental:  {:.2f}%'.format(fund['target_dropout_rate'].quantile(0.9)))
print('P90 Medio:        {:.2f}%'.format(med['target_dropout_rate'].quantile(0.9)))
"""
        ),
        md(
            """
### How to read a boxplot (30-second guide)
- The **box** = middle 50% of schools.
- The line inside = **median** (half the schools are below it).
- Dots / whiskers show the spread and extreme schools.

If Medio's box and whiskers sit **higher** or stretch farther than Fundamental's, the levels do not share the same typical risk.

### Insight
When the distributions differ, a pooled model tends to under-serve one level (often Medio). Separate models keep errors and explanations honest per stage.
"""
        ),
        md("## 2. Do the associated school traits differ?"),
        code(
            """
def top_corr(df):
    cols = ['is_rural', 'is_public', 'has_internet', 'has_computer_lab', 'has_sports_court',
            'has_library', 'enrollment_level', 'dropout_rate_lag1']
    tmp = df.copy()
    tmp['student_teacher_ratio'] = tmp['enrollment_level'] / tmp['teacher_count_basic_ed'].replace({0: pd.NA})
    cols = [c for c in cols + ['student_teacher_ratio'] if c in tmp.columns]
    s = tmp[cols + ['target_dropout_rate']].corr(numeric_only=True)['target_dropout_rate'].drop('target_dropout_rate')
    return s.rename(index=FEATURE_LABELS)

cmp = pd.DataFrame({
    'Fundamental': top_corr(fund),
    'Medio': top_corr(med),
})
display(cmp)
ax = cmp.plot(kind='barh', figsize=(10, 6))
ax.set_title('Same school traits, two education levels - correlation with dropout')
ax.set_xlabel('Correlation with dropout rate (-1 to +1)')
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """
### How to read this comparison chart
- Each horizontal pair of bars is **one school trait** (see dictionary above), now including `dropout_rate_lag1` for direct comparison against the Census features.
- Ask:
  1. Do the **signs** agree (both positive or both negative)?
  2. Do the **magnitudes** differ a lot?
  3. Is the **ranking** of strongest traits different?

### Insight (argument for two Streamlit apps)
If Fundamental is more tied to **infrastructure / rurality** while Medio is more tied to **school size / staffing (students per teacher)** - with both levels agreeing that dropout history is the single strongest signal - then:
- counselors need **different explanation templates** per level, but the same history-first triage logic;
- one shared "top 5 drivers" list would mislead users about the Census-feature ranking specifically;
- therefore **two models + two apps** is the scientifically cleaner product design.

Still remember: correlations are associations, not proof of cause.
"""
        ),
        md("## 3. Trained-model drivers (if models already exist)"),
        code(
            """
rows = []
for level, label in [('fundamental', 'Fundamental'), ('medio', 'Medio')]:
    path = Path(ROOT) / 'models' / level / 'figures' / 'global_importance_top.csv'
    try:
        imp = pd.read_csv(path).head(8)
        imp['level'] = label
        imp['feature_label'] = imp['feature'].map(FEATURE_LABELS).fillna(imp['feature'])
        rows.append(imp)
    except FileNotFoundError:
        pass
if rows:
    both = pd.concat(rows, ignore_index=True)
    display(both[['level', 'feature_label', 'importance']])
else:
    print('Train models first: python -m src.train --level both')
"""
        ),
        md(
            """
### How to read this table
For each level, features are ordered by how much the **selected model** relies on them when predicting dropout.

### Final insight for reviewers
1. Official labels + Brazil basic education means population/data alignment is restored (see `docs/scope_revision.md`).
2. Fundamental and Medio differ in **level** and often in **drivers**.
3. Dual products are a narrowing of scope (one country), not a return to multi-country sprawl.
4. Every reported metric here should be read next to its trivial-baseline comparison in `models/{level}/metrics.json` - see `docs/validity_and_english_revision.md`.
5. Apps must show the limitation: **school triage**, not student scoring.
"""
        ),
    ]
    write_nb(NB_DIR / "04_compare_fundamental_vs_medio.ipynb", cells)


if __name__ == "__main__":
    nb_validation()
    nb_level("fundamental", "02_eda_fundamental.ipynb", "EDA - Ensino Fundamental")
    nb_level("medio", "03_eda_medio.ipynb", "EDA - Ensino Medio")
    nb_compare()
