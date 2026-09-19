# Milestone 1 — Define risk criteria

**Status:** Tasks 1–5 delivered and reviewed; the Foundation's decision is recorded in §6 and applied in code (§7).  
**Data:** INEP school × year marts, 2018–2025. Operational comparisons use **2025** (latest year).  
**Sections 1–5** are the note as it went to the Foundation and are left unedited, so the decision in §6 can be read against what was actually presented.

---

## 1. Current risk definition

Source: `src/etl/build_school_risk_marts.py` (`assign_risk_bands`).

The mart labels **observed** official dropout (`taxa de abandono`), not the model score.

| Output | Rule |
|---|---|
| `risk_band` | Rank-percentile within the education level (Fundamental and Médio separately, all years pooled): **low** ≤ 0.33, **moderate** 0.33–0.66, **high** > 0.66 |
| Absolute override | Dropout **≥ 5%** is always **high** |
| Floor protection | The minimum observed rate (almost always **0%**) is forced to **low**, so a large block of zeros cannot be labelled high because of tied ranks |
| `high_risk` | Binary 1 if (rank > 0.66) **or** (rate ≥ 5%), after the floor rule |

The 66th-percentile *value* is stored in the mart manifest for transparency only. It is **not** the decision cut. In the current marts that value is **0.2%** (Fundamental) and **2.5%** (Médio).

This label is what ranking metrics use (`precision@k` against `high_risk`). It is **not** what the Streamlit app uses to build a shortlist. The app ranks by **`pred_dropout_rate`** and lets the user export **top-N** (default 50) filtered by state/year.

---

## 2. How many schools are flagged today

School-year rows, all years 2018–2025:

| Level | Rows | Unique schools | `high_risk` = 1 | Share |
|---|---:|---:|---:|---:|
| Fundamental | 832,326 | 122,211 | 289,313 | **34.8%** |
| Médio | 225,199 | 32,612 | 76,346 | **33.9%** |

**2025 only** (one row per school):

| Level | Schools | Flagged `high_risk` | Share |
|---|---:|---:|---:|
| Fundamental | 101,530 | **26,610** | 26.2% |
| Médio | 28,917 | **7,189** | 24.9% |

That is already thousands of schools in a single year — not a list a team can follow up.

### Why the flag is so large (Fundamental)

About **65%** of Fundamental rows have **exactly 0%** dropout. Once zeros are pinned to *low*, almost every school with a non-zero rate sits above the 66th percentile. The *moderate* band almost disappears (4,997 rows vs 289,313 *high*). So “high risk” currently means **“not zero”** more than **“unusually severe.”**

Médio is less zero-inflated (~46% zeros) and the 66th percentile is 2.5%, so the three bands are more meaningful — but the high group is still ~one third of rows.

### 2025 examples (Fundamental)

| UF | Schools | Flagged | Share |
|---|---:|---:|---:|
| SP | 14,572 | 4,993 | 34% |
| BA | 9,319 | 3,215 | 35% |
| PE | 5,226 | 891 | 17% |
| AM | 3,220 | 1,298 | 40% |

A partner looking at **Bahia Fundamental** would still see **3,215** “high-risk” schools.

---

## 3. Candidate thresholds (2025, no model retraining)

Three families of rules. Counts are **number of schools that would enter the list**.

### Fundamental 2025 (101,530 schools)

| Rule | Schools | Share | Fits “actionable shortlist”? |
|---|---:|---:|---|
| Current `high_risk` label | 26,610 | 26.2% | No |
| Observed dropout ≥ 5% | 2,155 | 2.1% | Still large nationally |
| Observed dropout ≥ 10% | 562 | 0.6% | Smaller, but misses many non-zero schools in low-dropout states |
| Predicted dropout ≥ 5% | 670 | 0.7% | Same issue: 5% is rare in Fundamental (mean 0.5% in 2025) |
| Predicted top 10% national | 10,153 | 10.0% | Still thousands |
| Predicted top 5% national | 5,076 | 5.0% | Still thousands |
| Predicted top 50 **national** | 50 | ~0% | Too small; ignores most states |
| **Predicted top 50 per state** | **1,350** | 1.3% | **Yes — ~50 per UF** |
| **Predicted top 20 per state** | **540** | 0.5% | **Yes — tighter capacity** |

### Médio 2025 (28,917 schools)

| Rule | Schools | Share | Fits “actionable shortlist”? |
|---|---:|---:|---|
| Current `high_risk` label | 7,189 | 24.9% | No |
| Observed dropout ≥ 5% | 3,575 | 12.4% | Large (mean dropout is higher than Fundamental) |
| Observed dropout ≥ 10% | 958 | 3.3% | Possible national list, still big for one NGO |
| Predicted dropout ≥ 5% | 2,713 | 9.4% | Large |
| Predicted top 10% national | 2,892 | 10.0% | Large |
| Predicted top 50 **national** | 50 | ~0% | Too small |
| **Predicted top 50 per state** | **1,350** | 4.7% | **Yes** |
| **Predicted top 20 per state** | **540** | 1.9% | **Yes** |

A **5% absolute cut** does **not** mean the same thing in both levels: it is strict in Fundamental and relatively common in Médio. A single national percentage is a poor product rule unless the Foundation explicitly wants different cuts per level.

---

## 4. How the rankings compare (2025)

Overlap between lists (Jaccard = intersection / union).

**Fundamental**

| Pair | Intersection | Share of shorter list that sits inside the current/other list |
|---|---:|---|
| Current `high_risk` vs predicted top 10% national | 7,835 | 77% of the predicted top 10% are already labelled high |
| Current `high_risk` vs predicted top 50 per state | 1,022 | **76% of the 1,350 shortlist schools are currently `high_risk`** |
| Predicted top 10% vs observed top 10% | 5,818 | ~57% overlap |
| Predicted top 50 per state vs observed top 50 per state | 489 | ~36% overlap |

**Médio**

| Pair | Intersection | Note |
|---|---:|---|
| Current `high_risk` vs predicted top 10% national | 2,318 | 80% of predicted top 10% are currently high |
| Current `high_risk` vs predicted top 50 per state | 861 | 64% of the 1,350 shortlist are currently `high_risk` |
| Predicted top 10% vs observed top 10% | 1,583 | ~54% overlap |
| Predicted ≥ 5% vs observed ≥ 5% | 1,677 | 62% of predicted ≥ 5% also observed ≥ 5% |

Reading: a **short predicted list is mostly inside today’s high-risk set**, so tightening the list does not invent a new population — it **prioritizes within** the schools we already call elevated. Agreement between predicted and observed *top 50* is only moderate (~25–45%), which is expected: the model is for **early warning**, not a copy of the official rate already published.

### One-state snapshot (predicted top 50 vs current flag)

| Level | UF | Schools | Current high-risk | Of predicted top 50, how many are currently high-risk |
|---|---|---:|---:|---:|
| Fundamental | BA | 9,319 | 3,215 | 50 / 50 |
| Fundamental | PE | 5,226 | 891 | 36 / 50 |
| Fundamental | SP | 14,572 | 4,993 | 49 / 50 |
| Fundamental | AM | 3,220 | 1,298 | 41 / 50 |
| Médio | BA | 1,580 | 523 | 40 / 50 |
| Médio | PE | 1,093 | 81 | 26 / 50 |
| Médio | SP | 6,364 | 2,005 | 50 / 50 |

In Bahia Fundamental, moving from **3,215 flags** to **50 schools** is the difference between a census and a worklist. Those 50 have mean *observed* dropout of **9.9%** (mean predicted **6.8%**), versus a state mean of **0.9%**.

---

## 5. Findings for stakeholder review

### Recommended operational rule

**Capacity-constrained prioritization, inside a network:**

1. Filter to one **state or municipal network** (and one education level).
2. Rank by **`pred_dropout_rate`** (the prototype already does this).
3. Take **top N**, where N is follow-up capacity (default **20–50**).

This is the only family of rules that reliably produces a shortlist someone can use. The current `high_risk` label should stay as an **evaluation metric** until the Foundation explicitly wants it changed.

### Alternatives (if the Foundation prefers a threshold, not a count)

| Option | When it is reasonable | Risk |
|---|---|---|
| Absolute cut (e.g. predicted ≥ 5%) | Médio, or a policy that “5% dropout is unacceptable” | In Fundamental it flags few schools in some states and still hundreds nationally; 5% is not comparable across levels |
| Within-state top 10% | A state secretariat that can visit ~10% of schools | Still 10,152 Fundamental schools nationwide in 2025 |
| Top 50 national | A tiny pilot | Ignores most of the country |

### Decision the Foundation still needs

1. **What is “high risk” for action?** Top-N per network (recommended), a percentage cut, or a within-network percentile?
2. **What N (or what percentile) matches real capacity?** Analyst / NGO / public-sector team / school coordinator will imply different N.
3. **First validation geography:** one UF or one municipal network. Bahia Fundamental is a concrete candidate: 3,215 current flags vs a 50-school predicted list whose observed dropout is clearly elevated.
4. **Keep `high_risk` as an evaluation label?** Recommended yes, until after validation.

### Out of scope until that decision

- Notion task 6 — *Apply validated threshold* → **decided, see §6; implemented, see §7**
- Notion task 7 — *Recalculate risk lists* → **decided, see §6; implemented, see §7**
- Model retraining, hosting, and branch merge → still open

---

## 6. Decision received (tasks 6–7)

The Foundation reviewed the findings above and decided:

| Decision | Detail |
|---|---|
| **Rule** | **Top-N prioritization**, not a fixed dropout threshold — fixed thresholds behave differently in Fundamental and Médio |
| **N** | **50 by default**, adjustable by the end user (e.g. 20 / 50 / 100) |
| **Level** | Prioritize within a **municipal / educational network**, not at state level — interventions and resources are managed locally |
| **Who sets N** | The **team responsible for field implementation** (NGO or education department), based on how many schools they can realistically follow up on |
| **`high_risk`** | Kept in the backend as a **historical / evaluation label**; the end user sees the **prioritized Top-N list** instead of the classification |
| **Pilot** | Bahia is a useful mathematical example, but the real pilot follows **wherever the committed stakeholder is** |

### The one technical finding that changed the design

"Top 50 within a municipal network" is not a constraint at that granularity. Measured on 2025 (public schools, network = municipality × administrative dependency):

| Level | Granularity | Networks | Median size | Networks larger than 50 |
|---|---|---:|---:|---:|
| Fundamental | Municipal network | 9,392 | **3 schools** | 208 |
| Fundamental | State network | 75 | 355 schools | 53 |
| Médio | Municipal network | 6,176 | **1 school** | 27 |
| Médio | State network | 66 | 38 schools | 28 |

At municipal-network granularity, N = 50 returns the **entire network** for ~98% of municipalities. Nationwide that puts **70,491** Fundamental schools on "prioritized" lists — barely better than the 26,610 the old flag produced, and worse in the sense that it looks prioritized when it is not.

**Resolution:** the scope is a hierarchy the user picks, not a fixed level. Municipal network is a first-class option (the Foundation's intent — prioritize locally, inside a network someone actually manages — is preserved), but the default is the **state network**: one administrative network within one state, e.g. "BA / Municipal". That is still a single managed network, not a statewide pool of all schools regardless of who runs them, and it is the smallest unit where N = 50 discriminates. Where the chosen scope is smaller than N, the app says so explicitly instead of presenting the whole network as a prioritized subset.

---

## 7. What was implemented

- `src/prioritize.py` — the rule, isolated from model plumbing: filter to a scope, rank by `pred_dropout_rate`, take top N. Deterministic tie-break (about 1.9% of predictions are exact duplicates, so without one the same query returned different schools between runs).
- `app/ui_common.py` — the first tab is now **Prioritized school list**: year, scope, state, municipalities, administrative network, and N (20/50/100 or custom). `high_risk` is not shown. Fixed alongside: the app previously scored only the first 20,000 rows of a selection before ranking, so a large selection's "top 50" was not the top 50.
- `src/evaluate.py` — `evaluate_ranking_within_group`, reported in `models/<level>/metrics.json` as `test_ranking_within_network`. The existing national `test_ranking_metrics` is unchanged.
- `src/etl/build_school_risk_marts.py` — **unchanged.** `assign_risk_bands` and the `high_risk` label are exactly as before, which is what the Foundation asked for.

### What the prioritized list looks like (BA, municipal network, 2025, N = 50)

| | |
|---|---:|
| Schools in the pool | 7,193 |
| Currently flagged `high_risk` | 2,778 |
| Prioritized | **50** |
| Mean **observed** dropout in the list | **9.13%** |
| Mean observed dropout in the pool | 0.96% |

### Two caveats worth carrying into a pilot

1. **Within-network lift is genuinely lower than national.** On held-out schools, top-50 inside a state network gives precision 0.50 and lift 1.45 (Fundamental), against 2.18 for a single national ranking. That is expected — schools inside one network are more alike — and it is the number to plan against. The in-sample figures in `docs/milestone1_network_prioritization_stats.json` read higher (lift 2.17) because most schools there were seen during training.
2. **Some networks have nothing to prioritize.** Salvador's municipal Fundamental network averages 0.05% observed dropout, and its top 50 has *lower* observed dropout than the network as a whole — the ranking is sorting noise. A Top-N list always returns N rows, so the app now warns when the selected schools are predicted below the model's own margin of error.

---

## Reproducibility

```text
python scripts/analyze_high_risk_criteria.py        # tasks 1-5 (unchanged)
python scripts/analyze_network_prioritization.py    # tasks 6-7
```

The first writes `docs/milestone1_high_risk_stats.json`, the second `docs/milestone1_network_prioritization_stats.json`. Scoring uses the trained 2025-year slice of each mart; no retraining.
