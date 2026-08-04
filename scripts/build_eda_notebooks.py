"""Generate Brazil EDA notebooks with layperson glossaries + chart insights."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

# Shared variable dictionary (shown in every analytical notebook)
VAR_DICT = """
## Variable dictionary (read before the charts)

| Column / label in charts | Plain-English meaning | How to read values |
|---|---|---|
| `target_dropout_rate` / **Abandonment rate (%)** | Official INEP *taxa de abandono*: share of students who **stopped attending** during the school year after the census reference date | 0 = nobody left; 5 = 5% left. Higher = worse |
| `year` | School census / rendimento year | Years present in the mart (currently 2018–2025) |
| `school_id` (`CO_ENTIDADE`) | Unique school code (INEP) | Join key between Censo and Rendimento |
| `uf` | Brazilian state abbreviation | e.g. SP, BA, AM |
| `municipio_id` | IBGE municipality code | Geographic context |
| `tp_dependencia` | Administrative network | 1=Federal, 2=State, 3=Municipal, 4=Private |
| `tp_localizacao` | School location type | 1=Urban, 2=Rural |
| `is_rural` | 1 if rural school | Shortcut of `tp_localizacao == 2` |
| `is_public` | 1 if public network (federal/state/municipal) | 0 = private |
| `in_agua` | Has potable / public water | 1=yes, 0=no |
| `in_energia` | Connected to public electricity | 1=yes, 0=no |
| `in_esgoto` | Public sewage connection | 1=yes, 0=no |
| `in_internet` | Internet available at school | 1=yes, 0=no |
| `in_biblioteca` | Library / reading room | 1=yes, 0=no |
| `in_lab_info` | Computer lab | 1=yes, 0=no |
| `in_quadra` | Sports court | 1=yes, 0=no |
| `qt_mat_bas` | Enrollment count (basic education total, when available) | Larger = bigger school |
| `enrollment_level` | Enrollment used for this level (Fundamental or Médio) | Filter requires ≥ 20 students |
| `qt_doc_bas` | Number of teachers (basic education) | Staffing intensity |
| `student_teacher_ratio` | Students ÷ teachers | Higher often means more crowded classes |
| `risk_band` | low / moderate / high | Relative triage label inside the level |
| `high_risk` | 1 if school is in the elevated-risk group | Binary flag for triage demos |
| **MAE / RMSE / R²** | Model error metrics (later notebooks / model folder) | Lower MAE/RMSE better; R² closer to 1 better |

### Acronyms
| Acronym | Meaning |
|---|---|
| **INEP** | Brazilian federal education statistics institute |
| **Censo Escolar** | Annual school census (structure + enrollment) |
| **Taxas de Rendimento** | Official approval / failure / abandonment rates |
| **EDA** | Exploratory Data Analysis |
| **UF** | Federative unit (state) |
| **Fundamental** | Ensino Fundamental (approx. primary + lower secondary) |
| **Médio** | Ensino Médio (upper secondary) |
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


# Split bootstrap: Path.cwd()/exists/resolve can hang on Desktop/OneDrive kernels.
BOOT_PATH = '''
# Cell A — project path only (no Path.cwd / exists / resolve)
import sys
ROOT = r"C:\\Users\\User\\Desktop\\Projeto Living Stone Foundation"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
'''

BOOT_IMPORTS = '''
# Cell B — imports (first run can take a minute for pandas/seaborn)
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
    "in_internet": "Has internet (1=yes)",
    "in_lab_info": "Has computer lab (1=yes)",
    "in_quadra": "Has sports court (1=yes)",
    "in_biblioteca": "Has library (1=yes)",
    "in_agua": "Has water (1=yes)",
    "in_energia": "Has electricity (1=yes)",
    "in_esgoto": "Has sewage (1=yes)",
    "enrollment_level": "Enrollment (this level)",
    "qt_mat_bas": "Basic-ed enrollment (total)",
    "qt_doc_bas": "Number of teachers",
    "student_teacher_ratio": "Students per teacher",
    "tp_dependencia": "Admin network code",
    "tp_localizacao": "Urban/rural code",
    "target_dropout_rate": "Abandonment rate (%)",
}
print("Imports OK")
'''


def bootstrap_cells(*extra_import_lines: str) -> list:
    """OneDrive-safe setup: path cell, then imports cell, then brief note."""
    extra = "\n".join(extra_import_lines)
    imports = BOOT_IMPORTS
    if extra:
        imports = BOOT_IMPORTS.replace("print(\"Imports OK\")", extra + "\nprint(\"Imports OK\")")
    return [
        code(BOOT_PATH),
        md(
            """
### What this cell did
Added the project folder to Python’s import path using a **fixed string** (no `Path.cwd()` / `exists` / `resolve`). Those path checks can freeze kernels on Desktop/OneDrive.
"""
        ),
        code(imports),
        md(
            """
### What this cell did
Loaded charting/table libraries. The **first** run can take ~30–90s while pandas/seaborn warm up — that is normal, not a freeze on `import sys`.
"""
        ),
    ]


def nb_validation():
    cells = [
        md(
            f"""
# Brazil data validation — INEP Censo + Taxas de Rendimento

**Living Stone Foundation — Applied Data Lab**

**Who this notebook is for:** readers who did **not** build the project.  
**Goal:** prove that the population (Brazilian basic-education **schools**) matches the labels we train on (official INEP **abandonment rates**).

{VAR_DICT}

### How to read this notebook
1. Run cells top to bottom (`Shift+Enter`).
2. After every code cell, read the **Insight** markdown — that is the takeaway, not the raw JSON.
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
print('Abandonment definition:', payload.get('definition_abandono'))
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
- **Source** should cite INEP Taxas de Rendimento (official), not a homemade proxy.
- **Definition** means: students who *left attending* during the year — not “failed the year” and not “never enrolled”.
- **Grain = school × year** means each measurement is about a **school**, not a named student.
- **Coverage**
  - `fundamental_non_null_pct` near ~90%+: most schools have a Fundamental abandonment number.
  - `medio_non_null_pct` much lower (~20%+): only schools that offer Ensino Médio report that rate.
  - Means: Médio abandonment is typically **higher** than Fundamental (e.g. ~3% vs ~1%).
- **Student-level labels available? → False**: we cannot honestly build a student scorer with these open files.

### Insight (decision for the project)
Population and labels are aligned for a **Brazil school-level** early-warning system. They are **not** aligned with the old Kaggle higher-education student dataset — that is why it was removed.
"""
        ),
        md("## 2. Side-by-side mart summary (Fundamental vs Médio)"),
        code(
            """
summary = compare_levels()
# Friendlier column names for readers
pretty = summary.rename(columns={
    'level': 'Education level',
    'rows': 'School-year rows',
    'schools': 'Unique schools',
    'mean_abandono': 'Mean abandonment (%)',
    'median_abandono': 'Median abandonment (%)',
    'p90_abandono': '90th percentile abandonment (%)',
    'public_share': 'Share public schools',
    'rural_share': 'Share rural schools',
})
display(pretty)
"""
        ),
        md(
            """
### How to read this table
Each row is one education level after joining Censo features to official abandonment rates and keeping schools with enough enrollment (≥ 20).

| Column | Takeaway question |
|---|---|
| School-year rows | Do we have enough data to model? |
| Mean vs median abandonment | Is the outcome rare/skewed? (median near 0 with mean > median ⇒ many zeros + a right tail) |
| 90th percentile | What does a “high” school look like in this level? |
| Public / rural shares | What kind of schools dominate the sample? |

### Insight
If **Médio** shows higher mean abandonment and a heavier right tail than **Fundamental**, training **two models** (and two Streamlit apps) is justified: the risk regimes are different. A single pooled model would mix apples and oranges.
"""
        ),
    ]
    write_nb(NB_DIR / "01_eda_brazil_data_validation.ipynb", cells)


def nb_level(level: str, filename: str, title: str):
    level_title = "Ensino Fundamental" if level == "fundamental" else "Ensino Médio"
    cells = [
        md(
            f"""
# {title}

**Living Stone Foundation — Applied Data Lab**

**Audience:** non-authors of the project.  
**Level in focus:** **{level_title}** (`{level}`).  
**Question this notebook answers:** What does official school abandonment look like, where is it higher, and which school characteristics move together with it?

{VAR_DICT}
"""
        ),
        *bootstrap_cells(
            "from src.eda import load_level_mart, missingness_table, summary_by_uf",
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
- `target_dropout_rate` is the official abandonment % we will try to predict later.
- Feature columns (`in_*`, `is_*`, enrollment, teachers) come from the **Censo Escolar**.

### Insight
You are looking at a **school risk table**, not a student list. Interventions suggested by this project are about **which schools to prioritize**, not which child to call.
"""
        ),
        md("## 1. Distribution of the abandonment rate"),
        code(
            """
display(df['target_dropout_rate'].describe().rename('Abandonment rate (%)'))
fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(df['target_dropout_rate'], bins=40, ax=ax, color='#1f4e79')
ax.set_title('How common is each abandonment rate?')
ax.set_xlabel('Official abandonment rate (%)')
ax.set_ylabel('Number of school-year rows')
plt.tight_layout()
plt.show()
print('Share of rows with 0% abandonment:', float((df['target_dropout_rate'] == 0).mean()))
print('Share of rows with abandonment ≥ 5%:', float((df['target_dropout_rate'] >= 5).mean()))
"""
        ),
        md(
            f"""
### How to read this chart
- **X-axis:** abandonment rate in percent (0–100 scale, usually concentrated near 0).
- **Y-axis:** how many school-year observations fall in that bin.
- A tall bar at 0% means many schools reported **no abandonment** that year.

### Insight for {level_title}
Abandonment is a **rare / skewed** school outcome. That is normal. Practical consequences:
1. Averages can look “small” while a minority of schools is still in serious trouble (look at the right tail / 90th percentile).
2. Model quality should be judged with **MAE/RMSE**, not classification accuracy.
3. Triage should focus on the **high tail**, not on shaming schools with 0–1%.
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
- Columns near the top are the emptiest (if any).

### Insight
For modeling we already kept rows with a non-null official abandonment rate. If a *feature* is heavily missing, either drop it or impute carefully — never silently treat blanks as zeros without saying so.
"""
        ),
        md("## 3. Where is abandonment higher? (by state / UF)"),
        code(
            """
g = summary_by_uf(df, min_n=30).rename(columns={
    'uf': 'State (UF)',
    'n': 'School-year rows',
    'mean_abandono': 'Mean abandonment (%)',
    'median_abandono': 'Median abandonment (%)',
})
display(g.head(15))
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(data=g.head(12), x='State (UF)', y='Mean abandonment (%)', ax=ax, color='#9c2a2a')
ax.set_title('States with highest mean school abandonment (only UF with ≥ 30 rows)')
ax.set_xlabel('State (UF)')
ax.set_ylabel('Mean abandonment rate (%)')
plt.tight_layout()
plt.show()
"""
        ),
        md(
            f"""
### How to read this chart
- Each bar is a **state (UF)**.
- Height = average school abandonment rate among schools of **{level_title}** in that state (with enough sample size).
- We require **n ≥ 30** so tiny samples do not create fake “worst state” headlines.

### Insight
Use this as a **map of attention**, not as a final ranking for newspapers:
- Different states have different networks (more municipal vs state schools).
- Coverage and school size differ.
- Still, persistently high bars suggest where network managers may want deeper local diagnostics.
"""
        ),
        md("## 4. Which school traits move with abandonment? (simple correlations)"),
        code(
            """
feat = ['is_rural','is_public','in_internet','in_lab_info','in_quadra','in_biblioteca','enrollment_level','qt_doc_bas']
feat = [c for c in feat if c in df.columns]
tmp = df[feat + ['target_dropout_rate']].copy()
if 'qt_doc_bas' in tmp.columns and 'enrollment_level' in tmp.columns:
    tmp['student_teacher_ratio'] = tmp['enrollment_level'] / tmp['qt_doc_bas'].replace({0: pd.NA})
    feat = feat + ['student_teacher_ratio']
corr = tmp[feat + ['target_dropout_rate']].corr(numeric_only=True)['target_dropout_rate'].drop('target_dropout_rate')
corr_named = corr.rename(index=FEATURE_LABELS).sort_values()
display(corr_named.rename('Correlation with abandonment rate').to_frame())
ax = corr_named.plot(kind='barh', figsize=(9, 5), color='#2a9d8f')
ax.set_title('Simple one-variable associations with abandonment')
ax.set_xlabel('Correlation (−1 to +1). Positive = higher feature ↔ higher abandonment')
plt.tight_layout()
plt.show()
"""
        ),
        md(
            f"""
### How to read this chart
- Each bar is **one school characteristic**.
- Axis goes from −1 to +1:
  - **Positive** correlation: when the feature is higher/true, abandonment tends to be **higher**.
  - **Negative** correlation: when the feature is higher/true, abandonment tends to be **lower**.
- Example readings:
  - `Rural school (1=yes)` positive ⇒ rural schools show higher abandonment **on average** in this table.
  - `Has internet (1=yes)` negative ⇒ schools with internet show lower abandonment **on average**.

### Critical caveats (please keep)
1. Correlation is **not causation** (internet does not automatically “cause” retention).
2. This is **univariate** (one variable at a time). The ML model can reorder importance when variables are combined.
3. Compare later with `models/{level}/figures/global_importance_top.csv`.

### Insight for {level_title}
Write down the 3 strongest bars (positive and negative). Those become the plain-language story for counselors/managers in the Streamlit app for this level — and they may **differ** from the other education level (see notebook 04).
"""
        ),
        md("## 5. Optional: compare with the trained model drivers"),
        code(
            f"""
imp_path = Path(ROOT) / 'models' / '{level}' / 'figures' / 'global_importance_top.csv'
try:
    imp = pd.read_csv(imp_path)
    imp['feature_label'] = imp['feature'].map(FEATURE_LABELS).fillna(imp['feature'])
    display(imp[['feature_label','importance']].head(12))
    ax = imp.head(10).set_index('feature_label')['importance'].plot(kind='barh', color='#1f4e79', figsize=(9,4))
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
Unlike correlation, this answers: **“When the trained model predicts abandonment, which inputs move the prediction most?”**

### Insight
If correlation and model importance **agree** (e.g. rurality / public network / staffing), the narrative is robust.  
If they **disagree**, prefer the model list for triage explanations, but still show both transparently — disagreement often means confounding (variables overlapping).

### Closing takeaway ({level_title})
1. Outcome = official school abandonment %, skewed toward zero.  
2. Geography highlights uneven risk across states.  
3. Structural/staffing traits associate with risk — useful for prioritization, not blame.  
4. This notebook alone does **not** authorize student-level interventions.
"""
        ),
    ]
    write_nb(NB_DIR / filename, cells)


def nb_compare():
    cells = [
        md(
            f"""
# Compare Ensino Fundamental vs Ensino Médio

**Living Stone Foundation — Applied Data Lab**

**Audience:** team leads / reviewers who need a clear reason for **two models and two Streamlit apps**.

**Core question:** Do abandonment levels and school-trait associations differ enough between Fundamental and Médio that a single pooled Brazil model would be misleading?

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
    'mean_abandono': 'Mean abandonment (%)',
    'median_abandono': 'Median abandonment (%)',
    'p90_abandono': '90th percentile (%)',
    'public_share': 'Share public',
    'rural_share': 'Share rural',
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
- **Mean / 90th percentile abandonment** — is Médio systematically “harder”?
- **Row counts** — Médio has fewer schools (only those offering upper secondary).
- **Public / rural shares** — different school mixes can change which features matter.

### Insight
A higher Médio mean with fewer schools already suggests **different operating regimes**. That is necessary (but not sufficient) evidence for separate models — the next charts check the shape of risk and the correlates.
"""
        ),
        md("## 1. Outcome shape: boxplots side by side"),
        code(
            """
fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
sns.boxplot(data=fund, y='target_dropout_rate', ax=ax[0], color='#4c78a8')
ax[0].set_title('Fundamental — abandonment rate (%)')
ax[0].set_ylabel('Official abandonment rate (%)')
sns.boxplot(data=med, y='target_dropout_rate', ax=ax[1], color='#e76f51')
ax[1].set_title('Médio — abandonment rate (%)')
ax[1].set_ylabel('')
plt.suptitle('Are the risk distributions similar?', y=1.02)
plt.tight_layout()
plt.show()
print('Mean Fundamental: {:.2f}%'.format(fund['target_dropout_rate'].mean()))
print('Mean Médio:       {:.2f}%'.format(med['target_dropout_rate'].mean()))
print('P90 Fundamental:  {:.2f}%'.format(fund['target_dropout_rate'].quantile(0.9)))
print('P90 Médio:        {:.2f}%'.format(med['target_dropout_rate'].quantile(0.9)))
"""
        ),
        md(
            """
### How to read a boxplot (30-second guide)
- The **box** = middle 50% of schools.
- The line inside = **median** (half the schools are below it).
- Dots / whiskers show the spread and extreme schools.

If Médio’s box and whiskers sit **higher** or stretch farther than Fundamental’s, the levels do not share the same typical risk.

### Insight
When the distributions differ, a pooled model tends to under-serve one level (often Médio). Separate models keep errors and explanations honest per stage.
"""
        ),
        md("## 2. Do the associated school traits differ?"),
        code(
            """
def top_corr(df):
    cols = ['is_rural','is_public','in_internet','in_lab_info','in_quadra','in_biblioteca','enrollment_level']
    tmp = df.copy()
    tmp['student_teacher_ratio'] = tmp['enrollment_level'] / tmp['qt_doc_bas'].replace({0: pd.NA})
    cols = [c for c in cols + ['student_teacher_ratio'] if c in tmp.columns]
    s = tmp[cols + ['target_dropout_rate']].corr(numeric_only=True)['target_dropout_rate'].drop('target_dropout_rate')
    return s.rename(index=FEATURE_LABELS)

cmp = pd.DataFrame({
    'Fundamental': top_corr(fund),
    'Médio': top_corr(med),
})
display(cmp)
ax = cmp.plot(kind='barh', figsize=(10, 6))
ax.set_title('Same school traits, two education levels — correlation with abandonment')
ax.set_xlabel('Correlation with abandonment rate (−1 to +1)')
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """
### How to read this comparison chart
- Each horizontal pair of bars is **one school trait** (see dictionary above).
- Blue-ish / left series = **Fundamental**; orange / other series = **Médio** (legend in the plot).
- Ask:
  1. Do the **signs** agree (both positive or both negative)?
  2. Do the **magnitudes** differ a lot?
  3. Is the **ranking** of strongest traits different?

### Insight (argument for two Streamlit apps)
If Fundamental is more tied to **infrastructure / rurality** while Médio is more tied to **school size / staffing (students per teacher)**, then:
- counselors need **different explanation templates** per level;
- one shared “top 5 drivers” list would mislead users;
- therefore **two models + two apps** is the scientifically cleaner product design.

Still remember: correlations are associations, not proof of cause.
"""
        ),
        md("## 3. Trained-model drivers (if models already exist)"),
        code(
            """
rows = []
for level, label in [('fundamental','Fundamental'), ('medio','Médio')]:
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
    display(both[['level','feature_label','importance']])
else:
    print('Train models first: python -m src.train --level both')
"""
        ),
        md(
            """
### How to read this table
For each level, features are ordered by how much the **selected model** relies on them when predicting abandonment.

### Final insight for reviewers
1. Official labels + Brazil basic education ⇒ population/data alignment restored.  
2. Fundamental and Médio differ in **level** and often in **drivers**.  
3. Dual products are a narrowing of scope (one country), not a return to multi-country sprawl.  
4. Apps must show the limitation: **school triage**, not student scoring.
"""
        ),
    ]
    write_nb(NB_DIR / "04_compare_fundamental_vs_medio.ipynb", cells)


if __name__ == "__main__":
    nb_validation()
    nb_level("fundamental", "02_eda_fundamental.ipynb", "EDA — Ensino Fundamental")
    nb_level("medio", "03_eda_medio.ipynb", "EDA — Ensino Médio")
    nb_compare()
