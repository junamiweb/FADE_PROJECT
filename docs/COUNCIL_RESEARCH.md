# ועדת מחקר — FADE Council Research

עודכן: **2026-07-05**

**משימה:** לחקור רעיונות חדשים, ללמוד ממה שנכשל (ב-FADE וב-industry), ולהציע כיוונים — **בלי לבנות production** עד pre-reg + council.

Machine-readable: `fade/output/council_research.json`

---

## למה זה שונה מ"עוד batch"

| batch רגיל | ועדת מחקר |
|------------|-----------|
| מריץ pipeline | שואל "למה זה נכשל בעבר?" |
| מחפש hit גבוה | מחפש thesis שלא ב-graveyard |
| מוסיף features | בודק FP01–FP10 לפני build |
| תוצאה = artifact | תוצאה = hypothesis + pre-mortem |

---

## תפקידים (מחקר)

| תפקיד | אחריות |
|--------|---------|
| **Ideator** | מציע hypothesis; ממלא pre-mortem |
| **Historian** | graveyard + failure_patterns — "כבר ניסינו?" |
| **Prior Art** | QuantConnect, Bailey PBO, López de Prado, lifecycle |
| **Pre-mortem Skeptic** | איך זה נכשל לפני שרצים |
| **Chair** | מדרג backlog; בוחר 1–2 ל-pre-register |

**אין Builder** בשלב מחקר — רק אחרי RESEARCH_APPROVE.

---

## דפוסי כישלון (FP01–FP10)

ראה `council_research.json` → `failure_patterns`.

דוגמאות מהפרויקט:

- **FP01 atom_dilution** — candles, tiktok, ML על אותם atoms
- **FP02 holdout→lockbox** — BTC HIGH_VR+48
- **FP03 fee_drag** — 53% hit, PnL שלילי @5bps
- **FP06 persistence** — vol ATR 95%, straddle מטעה
- **FP07 ml_fantasy** — 52 מודלים, אף אחד לא מנצח rules

**כל רעיון חדש חייב לציין אילו FP הוא נמנע מפני.**

---

## מה FADE כבר עושה נכון (prior art פנימי)

- pre-registration לפני studies
- holdout quarantine + lockbox burned
- forward ledger = אמת
- sparse abstain
- SCOPE_GUARD + שלילים כנים ב-batch log

זה מיישר עם [QuantConnect Research Guide](https://www.quantconnect.com/docs/v2/writing-algorithms/key-concepts/research-guide) ו-[four-gate governance](https://www.equationstocapital.com/research/papers/paper-08-algorithmic-strategy-search-governance).

---

## מה industry למדה (סיכום)

| מקור | לקח |
|------|-----|
| **QuantConnect** | הגבל trials; ~16h לניסוי; OOS + paper |
| **Bailey PBO/DSR** | multiple testing מנפח Sharpe |
| **López de Prado** | purged CV, meta-labeling, triple-barrier |
| **Linitics lifecycle** | validation=destroy; decay; decommission |
| **>"90% strategies fail live"** | backtest ≠ truth |

---

## Backlog מחקר (R01–R10)

| ID | כיוון | חדשנות | הערה |
|----|--------|---------|------|
| **R02** | PBO/DSR על path_lean3 | בינונית | **מומלץ ראשון** — effort נמוך |
| **R05** | funding BTC-ETH spread | בינונית | orthogonal ל-batch 36 |
| **R01** | meta-labeling sparse PRIMARY | גבוהה | sandbox בלבד |
| **R04** | cross-sectional alt rank | גבוהה | לא pair trade (C נכשל) |
| **R08** | ROSE forward track | בינונית | alt screen winner |
| **R10** | abstention product | נמוך | = Track A forward |

פרטים + pre-mortem: `council_research.json`

---

## סessions מחקר

| session | תדירות | פקודה |
|---------|---------|--------|
| **RESEARCH_SCAN** | לפי בקשה | `council_research scan` |
| **RESEARCH_PLENARY** | שבועי | `council_research plenary` |
| **PROPOSE** | רעיון חדש | `council_research propose ...` |

---

## פקודות

```bash
python -m fade.pipeline.council_research scan
python -m fade.pipeline.council_research plenary
python -m fade.pipeline.council_research graveyard
python -m fade.pipeline.council_research propose "כותרת" --hypothesis "..." --avoid FP01,FP07
python -m fade.pipeline.council_research rank
```

---

## פרומpt — Ideator + Historian

```
Full Repository Path: ./
Readonly: true
Read: fade/output/council_research.json, docs/SCOPE_GUARD.md (batch log).
Task: Propose ONE new hypothesis not duplicated in graveyard.
For each: cite failure_patterns avoided, prior_art id, pre-mortem (3 bullets).
Output: append to research_backlog in council_research.json with status=proposed.
Do NOT implement code.
```

---

## פרומpt — RESEARCH_PLENARY

```
Readonly: true
Rank research_backlog by: forward_alignment, novelty, 1/effort, graveyard distance.
Pick top 2 for pre-registration ONLY (study entries, no pipeline).
Run pre-mortem Skeptic on top 2.
Update next_research_session in council_research.json.
```

---

## מעבר ממחקר ל-build

```
RESEARCH_APPROVE → pre_registration.json study entry → Builder → STUDY_REVIEW
```

**Gatekeeper:** אין `atoms.py` / core עד VALIDATED forward.

---

## קשר למועצה v2

- Forward Watcher — Track A/E (אמת)
- **Research Committee** — רעיונות חדשים (עתיד)
- Study Review — artifacts אחרי build
- Red Team — לפני ADVANCE

Charter כללי: `docs/FADE_COUNCIL.md`
