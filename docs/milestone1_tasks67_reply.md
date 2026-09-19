# Reply to the project manager — Foundation decision on tasks 6–7

Draft for Slack. Answers the question asked: *"let me know if you think there are any
technical considerations we should discuss before you implement these changes."*

---

Hi Carolina,

Thanks — this unblocks tasks 6 and 7. My reading of the decisions:

- Top-N prioritization instead of a fixed dropout threshold
- N = 50 by default, adjustable by the end user (20 / 50 / 100)
- Prioritization within a municipal / educational network, not statewide
- The field implementation team sets N, based on their real follow-up capacity
- `high_risk` stays in the backend as a historical/evaluation label; the user sees the
  prioritized list
- Bahia is an example case; the pilot follows the committed stakeholder

All of that is implemented. **One technical consideration** is worth a minute of your time,
because it changed the design.

**Municipal networks are much smaller than "N = 50" assumes.** In the 2025 data, the median
Brazilian municipal network has **3 schools** in Ensino Fundamental and **1** in Ensino Médio.
Only 208 of 9,392 municipal networks have more than 50 schools. So "top 50 within a municipal
network" returns the *entire network* for about 98% of municipalities — nationwide that is
70,491 Fundamental schools on "prioritized" lists, which is barely better than the 26,610 the
old high-risk flag produced, and arguably worse because it looks prioritized when it is not.

**What I did instead of just picking one level.** The user chooses the scope:

- **State network** (e.g. "BA / Municipal" — all municipal schools in Bahia) — the default.
  Median 355 schools, so N = 50 is a real constraint.
- **Municipal network** (one municipality's own schools) — exactly what the Foundation asked
  for, available whenever the network is big enough, e.g. a capital city.
- **Selected municipalities pooled into one list** — for a team working across a consortium.

This keeps the Foundation's intent: prioritization happens inside a network someone actually
manages, never a statewide pool mixing state-run and municipally-run schools. It just does not
force a granularity where the ranking cannot do any work. Where the chosen scope has fewer
schools than N, the dashboard says so directly rather than passing the whole network off as a
prioritized subset.

**Two smaller things to flag before a pilot:**

1. **Within-network lift is genuinely lower than the headline model numbers**, and I would
   rather say so now than at a pilot review. On held-out schools, top 50 inside a state network
   catches high-risk schools at about 1.45x the base rate, versus 2.18x for a single national
   ranking. That is expected — schools inside one network resemble each other more — and it is
   still a useful signal: in Bahia's municipal network, the 50 prioritized schools average
   **9.1% observed dropout against 0.96% for the network as a whole**.

2. **Some networks have nothing to prioritize.** Salvador's municipal Fundamental network
   averages 0.05% dropout; its "top 50" actually has *lower* observed dropout than the network
   average, because the model is ordering noise. A Top-N list always returns N schools, so
   nothing in the output reveals this on its own — the dashboard now warns when the selected
   schools fall below the model's own margin of error. Worth knowing before someone shows a
   list to a secretariat with a near-zero dropout rate.

**One question that does block sizing the pilot:** which committed stakeholder / network are we
aiming at? N should be set against their actual capacity, and whether they manage a municipal
network or a state-wide one determines which scope the dashboard should open on for them. Bahia
works fine as the worked example in the meantime.

Happy to walk through the dashboard whenever useful.

Best,
Cauê
