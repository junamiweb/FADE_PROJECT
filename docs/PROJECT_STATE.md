# FADE — מצב פרויקט (סיכום להמשך)

עודכן: **2026-07-23** (North star מחדד + Phase 0/1)

---

## ★ כוכב צפון (מחייב — 2026-07-23)

| | |
|--|--|
| **יעד סופי** | חיזוי מראש של השוק → מסחר → **רווח כלכלי נטו** (אחרי עמלות/slippage) |
| **גישה עכשיו** | מחקר קשוח — כי עדיין אין שיטה סחירה מוכחת |
| **מחקר ≠ מטרה** | מחקר הוא האמצעי. hit-rate / corr יפים בלי אופק PnL נטו = לא התקדמות |
| **מסחר בלי אמת** | רווח מדומה בלי lockbox / forward / pre-reg = סטייה |

**שאלת מבחן לכל עבודה:** האם זה מקרב אותנו לחיזוי **סחיר** עם רווח נטו — לא רק לדיוק יפה על holdout?

**מצב נוכחי מול היעד:** predictability דק, לרוב לא סחיר @5bps; ETH candidate + sparse PRIMARY באימות forward — עדיין לא validated.

---

## ★ Phase חדש — שלב 0 (חובה) + שלב 1

### שלב 0: האם ETH LOW_VR+min_hold=12 "מיחזר" את ה-lockbox?

**תשובה מפורשת:**

| שאלה | תשובה |
|------|--------|
| נבחר ע"י grid search **על** lockbox? | **לא** — batch 33 הריץ config ש-pre-registered מ-batch 32 |
| נבחר ע"י grid search **על holdout 70/30**? | **כן** — batch 32: 3 VR × 9 min_hold; ETH winner = LOW_VR+12 (+94.2%) |
| אותה מחלקת סיכון כמו BTC? | **כן** — post-hoc grid על holdout (BTC HIGH_VR+48 → lockbox −10.9%) |
| lockbox v1 מאמת את ETH? | **לא מספיק** — one-shot +23.2% @5bps הוא **מידע בלבד**; lockbox **BURNED** |

**סטטוס ETH LOW_VR+min_hold=12:** `candidate_not_validated` — **לא production**.

**Lockbox v1 (18%, SHA256 ב-manifest):** `BURNED` — אסור לבדוק עליו configs נוספים.  
**Lockbox v2:** שמור ל-final test עתידי — seal + hash **לפני** כל eval.

**אימות forward-only:** `eth_candidate_track.py` + ledger `fade/output/eth_candidate_outcomes.jsonl`  
**מדד v2 (מ-2026-07-05):** hold-cycle PnL נטו @ 5+5 bps, min_hold=12 — **מיושר ל-lockbox batch 33**.  
קריטריון: **≥100 hold-cycles**, profitable rate **≥55%** — רק אז `validated`.  
**next-bar hit** נשאר מידע משלים בלבד; signal אחד pre-fix (לפני v2) **לא** נספר ל-n=100.

Manifest: `fade/output/pre_registration.json` | `python -m fade.pipeline.pre_registration show`

### שלב 1: PRIMARY sparse (abstain-by-default) — ✅ פעיל

`forecast_tiers.py` — PRIMARY default = conviction tier **≥ HIGH** בלבד (elite/strong/high) + quality.  
`--legacy-primary` לשחזור ההתנהגות הישנה.  
`outcome_tracker run-all` — מצבר sparse PRIMARY + ETH candidate.

### שלבים 2–4 (ממתינים)

| # | משימה | סטטוס |
|---|--------|--------|
| 2 | ETH candidate forward validation | 🔄 0/100 hold-cycles (v2 PnL) |
| 3 | horizon sweep 4h/8h + PnL (pre-register לפני!) | ✅ **שלילי** — לא מנצח 1h |
| 4 | v0.3 data (funding ETH + lead-lag probe) | 🔄 exploratory |
| 5 | sparse PRIMARY holdout replay | ✅ ~58.2% BTC sparse |
| 6 | funding+streak combo | ✅ **שלילי** — no modulation |

### קריטריוני הצלחה (Phase חדש)

| מדד | סף | הערה |
|-----|-----|------|
| hit-rate | ≥58%, n≥500 | **lockbox v2 חדש בלבד** — לא v1 |
| PnL @5bps | חיובי one-shot | pre-register לפני seal |
| outcome_tracker | ≥55% profitable על 100 hold-cycles | **forward live, v2 PnL** |
| outcome_tracker (legacy) | next-bar hit | **משלים בלבד — לא קובע** |

---

## ★ Phase 2 — חמישה tracks אמיצים (pre-registered 2026-07-05)

**עקרון:** לא עוד אטומים על path_lean3. שאלות חדשות + forward כאמת. Core production ללא שינוי עד validation.

| Track | שאלה | מה בונים | סף הצלחה (exploratory) |
|-------|------|---------|------------------------|
| **A** | מוצר דליל | forward + paper replay על PRIMARY sparse | ~58% hit, כיסוי ~9%, PnL @5bps |
| **B** | מנוע טווח/תנודה | חיזוי range/vol במקום כיוון | lift 3pp+ על holdout |
| **C** | spread בין נכסים | ETH/BTC, SOL/BTC, ROSE/BTC mean-reversion | hit 55%+, PnL @5bps |
| **D** | מכונת מצבים | FSM: reversal / momentum / abstain לפי VR | מנצח sparse PRIMARY ב-holdout |
| **E** | מדד דעיכה חי | סחור רק כש-edge 90d חי | PnL forward טוב יותר מ-always-on |

**סדר ביצוע:**

1. **A** — הכי קרוב (Phase 1 כבר רץ); paper replay + המשך forward
2. **B + C** — holdout exploratory במקביל (נתונים כבר ירדו)
3. **D** — אחרי B/C (בונה על VR + conviction)
4. **E** — רץ ברקע על ledger PRIMARY

Manifest: `phase2_program_v1` + `phase2_a`…`phase2_e` ב-`pre_registration.json`

**מועצת בקרה:** `docs/FADE_COUNCIL.md` (v2 standing) | `council_board.json` | digest שעתי `council_digest.json`  
ועדות: Forward Watcher (שעתי) | **Research** (שבועי) | Study Review | Plenary | Red Team  
מחקר: `docs/COUNCIL_RESEARCH.md` — RESEARCH_PLENARY 2026-07-05: R02/R05/R01 pre-registered (no build)

---

## ★ מצב נוכחי — סיכום executive (batch 30: PnL + lockbox + generalization)

**שלושה פערים שנסגרו לפני חיפוש atoms נוסף:**

### 1. PnL reality v2 (`pnl_reality_check_v2.py`) — עלויות אמיתיות

| אסטרטגיה (holdout btc_1h) | 1 bps | **5 bps** | 10 bps |
|---|---|---|---|
| buy-and-hold | +82.7% | +82.7% | +82.7% |
| raw path_lean3 | +58.5% | **−99.0%** | −100% |
| conviction streak≥2 | +45.1% | **−94.1%** | −99.9% |
| conviction r≥2 + 3TF | +90.4% | **−39.7%** | −85.7% |
| PRIMARY policy | +65.0% | **−93.8%** | −99.9% |
| best min_hold (24 bars) | +393%* | **−12.7%** | −41.1% |

\* min_hold מנצח ב-1 bps בגלל turnover נמוך — **אף וריאנט לא חיובי ב-5 bps**.

**מסקנה כנה:** שיפור הדיוק (53%→54.6%) **לא** הופך את האות לסחיר. conviction gates מורידים turnover אך לא מספיק. **לא מכונת מסחר** — גם עם stack מלא.

### 2. Final lockbox (`final_lockbox.py`) — OOS אמיתי ללא multiple-comparisons

18% חדשים ביותר (2024-11-29 → 2026-07-05), SHA256 sealed ב-`fade/output/lockbox_manifest.json`. כרייה על 82% הראשונים בלבד; הערכה חד-פעמית.

| נכס | path_lean3 lockbox | vs 54.6% (batch 17) | conviction | PRIMARY |
|-----|-------------------|---------------------|------------|---------|
| **BTC** | **52.77%** (p=0.0033, n=9,762) | **−1.83pp** | 53.16% | 53.16% |
| **ETH** | **52.75%** (p=0.0033, n=8,470) | **−1.85pp** | 51.62% | 51.62% |

**תיוג:** SUSPICIOUSLY_LOW מול 54.6% — חשד לניפוח מ-~20+ השוואות atom-set על אותו holdout. Edge **עדיין מובהק סטטיסטית** (~2.8% lift) אך **קטן יותר** מהמספר ששימש כ-headline.

### 3. Generalization audit (`generalization_audit.py`)

**Bonferroni/Holm (7 atom sets, holdout אחיד):**

| atom_set | hit OOS | p_raw | p_holm | Holm sig? |
|----------|---------|-------|--------|-----------|
| path_lean3 | **54.04%** | 0.0033 | 0.0231 | Y |
| path_big | 53.94% | 0.0033 | 0.0231 | Y |
| path_min | 53.75% | 0.0033 | 0.0231 | Y |
| core5 | 53.11% | 0.0033 | 0.0231 | Y |
| *(כל 7)* | 53.1–54.0% | 0.0033 | 0.0231 | **כולם Y** |

path_lean3 **הכי גבוה** אך **לא ייחודי** אחרי correction — כל הסטים עוברים Holm (אותו null structure).

**דעיכה 2025–2026 vs pre-2025 (conviction_stability):**

| חוק | BTC 2025–26 | BTC pre-2025 | Δ | ETH 2025–26 | ETH pre-2025 | Δ |
|-----|-------------|--------------|---|-------------|--------------|---|
| streak≥3 | 54.46% | 58.62% | **−4.2pp** | 53.62% | 57.98% | **−4.4pp** |
| r≥2 + 3TF | 57.58% | 59.16% | **−1.6pp** | 53.47% | 58.17% | **−4.7pp** |

**DECAY DETECTED** — edge חלש בשנה האחרונה. ייתכן שינוי מבני (עמלות, נזילות, מייקרים) או regression to mean.

### מה הלאה (batch 31+)

**החלטה:** לא atoms חדשים, לא v0.3 — קודם אבחון דעיכה (batch 31).

---

## ★ Batch 31 — אבחון דעיכה + benchmark מניות

### H1: שינוי משטר (vol / funding) — BTC

| VR regime | recent hit | earlier hit | Δ |
|-----------|------------|-------------|---|
| LOW_VR | 53.73% | 58.73% | **−5.0pp** |
| NORMAL | 54.53% | 59.23% | −4.7pp |
| HIGH_VR | 55.01% | 57.90% | **−2.9pp** (הכי פחות דעיכה) |

דעיכה **אחידה** ב-vol (spread 2.1pp). ב-funding: spread 7.2pp — EXTREME_NEG funding **חזק יותר** ב-2025–26 (+2.3pp).

### H2: מגמת זמן רבעונית — BTC

- Pre-2024: slope **−0.00149/q**, corr **−0.476** → **MONOTONIC DECLINE** (algo-competition plausible)
- Mean rev_index: pre-2024 = **0.070**, 2025–26 = **0.039**
- ETH: slope חלש יותר (−0.00063/q, corr −0.277) — לא מובהק

### H3: מיקרו-סטרוקטורה

**אין דאטה** (spread, liquidity, exchange count). volume_zscore/range_pct = proxies חלשים בלבד.

### H5: benchmark מניות (`stock_reversal_benchmark.py`)

| נכס | rev_index | p-value | span |
|-----|-----------|---------|------|
| SPY | **−0.008** | 0.44 | ~730d yfinance |
| AAPL | **+0.003** | 0.80 | ~730d |
| BTC today (2024+) | **+0.045** | 0.0007 | |
| BTC 2018–19 | **+0.091** | 0.0007 | |

מניות ≈ efficient (rev≈0). BTC **עדיין** מראה reversal (+0.045) — לא התכנס לחלוטין למניות. BTC 2018–19 היה חזק פי 2.

**ענף שנבחר: C → ברירת מחדל A** (BTC: H2 monotonic + H1 funding-concentrated; אך BTC reversal עדיין חי → לא B מלא).

---

## ★ Batch 32 — Branch A: regime-gated min_hold PnL @ 5bps

`pnl_regime_minhold.py` — path_lean3 + VR gate + min_hold sweep.

| נכס | regime | min_hold | return @5bps | trades |
|-----|--------|----------|--------------|--------|
| BTC | **HIGH_VR** | 48 | **+76.8%** | 308 |
| ETH | **LOW_VR** | 12 | **+94.2%** | 658 |
| BTC ungated | — | 24 | −12.7% | 574 |

⚠️ **אזהרה anti-self-deception:** תוצאות אלו נבחרו post-hoc מ-grid (3 regimes × 9 min_hold) על **אותו holdout 30%** — סיכון overfit גבוה. BTC ו-ETH מעדיפים regimes **שונים** — אין אסטרטגיה אחידה.

### Batch 33 — lockbox one-shot (`regime_minhold_lockbox.py`)

קונפigurations pre-registered מ-batch 32, נבדקו **פעם אחת** על lockbox 18% (2024-11-29 → 2026-07-05):

| נכס | config | holdout @5bps | **lockbox @5bps** | dir_hit | TAG |
|-----|--------|---------------|-------------------|---------|-----|
| BTC | HIGH_VR + hold 48 | +76.8% | **−10.9%** | 52.0% | FAIL_SOFT (overfit) |
| ETH | LOW_VR + hold 12 | +94.2% | **+23.2%** | 52.5% | POSITIVE_LOCKBOX |

**מסקנה:** BTC holdout winner **לא** הכליל — overfit מאושר. ETH שורד @5bps על lockbox אמיתי (+23.2%) אך **רחוק** מ-+94.2% holdout (inflation ~4×). **אין** אסטרטגיית מסחר מאוחדת BTC+ETH. ETH-only regime gate **עשוי** להיות שווה מחקר המשך; לא production.

---

## ★ Batch 35 — Horizon sweep + v0.3 probe

**Pre-registered** (`horizon_sweep_4h_8h`) לפני הרצה. Holdout 70/30 בלבד — לא lockbox.

### PnL @ 5bps (path_lean3, min_hold=24 fixed)

| asset | interval | rev_index | hit | pnl raw | **pnl mh24** | vs 1h |
|-------|----------|-----------|-----|---------|--------------|-------|
| btc | 1h | 0.046 | 54.0% | −99% | **−12.7%** | — |
| btc | 4h | 0.042 | 53.8% | −81% | **−40.5%** | −27.8pp |
| btc | 8h | 0.038 | 51.9% | 0% | **−42.8%** | −30.1pp |
| eth | 1h | 0.042 | 53.6% | −99.7% | **−79.6%** | — |
| eth | 4h | 0.015 | 52.7% | −90% | **−17.1%** | +62.5pp |
| eth | 8h | 0.037 | 50.0% | −96% | **−29.9%** | +49.7pp |

**מסקנה:** **אף horizon לא חיובי @5bps.** 4h/8h לא פותרים fee drag. ETH 4h פחות גרוע מ-1h (−17% vs −80%) אך עדיין שלילי. **1h נשאר baseline.**

### v0.3 lead-lag (`lead_lag_probe.py`) — pre-registered

- BTC↔ETH return corr ≈ **0.80** (כל lag — synchronous)
- BTC streak≥2 contrarian → ETH lag-1: **52.65%** (n=36K) — שולי, לא מסחרי
- `funding_eth.csv` הורד (7,236 rows) — לשימוש עתידי

---

## ★ Batch 36 — Sparse PRIMARY replay + funding combo + forward score

### Sparse PRIMARY holdout (`sparse_primary_replay.py`) — pre-registered

| asset | sparse hit | n | coverage | elite tier |
|-------|------------|---|----------|------------|
| **BTC** | **58.17%** | 2,068 | 8.9% | **62.64%** (n=265) |
| ETH | 55.31% | 1,929 | 8.3% | 59.66% (n=238) |

**הערה:** holdout exploratory — **forward tracker** הוא האמת ל-Phase 1. זה מראה שהמטרה 58%+ **אפשרית** ב-coverage ~9%.

### Funding + streak combo — **REJECT** (holdout)

| bucket | BTC hit | ETH hit |
|--------|---------|---------|
| EXTREME_NEG | 50.7% | 51.3% |
| NEUTRAL | **53.3%** | **51.5%** |

אין מודולציה — EXTREME_NEG **לא** מחזק reversal (סותר את decay H1 descriptive). ETH funding 8h test: p=0.12 WEAK.

### Forward tracker (חי, אחרי refresh)

| track | scored | metric | progress |
|-------|--------|--------|----------|
| sparse PRIMARY | 2 | next-bar hit | 50% (התחלה) |
| ETH candidate | 1 pre-fix | next-bar only | **לא נספר** |
| ETH candidate v2 | 0 | hold-cycle PnL @10bps RT | **0/100** |

**תיקון batch 37:** `score_eth_candidate_pnl()` — PnL מחזור min_hold=12 דרך `pnl_sim._equity` (כמו lockbox).  
`report-candidate` מציג שני מדדים: next-bar (legacy) + hold-cycle (validation).

```bash
python -m fade.pipeline.outcome_tracker run-all   # יומי / GitHub Action (cron :05)
```

**Automation:** `.github/workflows/outcome-tracker.yml` — refresh + `run-all` + commit ledgers/CSVs.

---

## ★ GitHub + Actions — הקמת repo (2026-07-05)

| שלב | תיאור | סטטוס |
|-----|--------|--------|
| **1** | `.gitignore` — venv/env/.env/IDE + חריגי `fade/output/*.jsonl` | ✅ **הושלם** |
| **2** | יצירת repo Private ב-github.com/new (ללא README/gitignore) | ✅ `github.com/junamiweb/FADE_PROJECT` |
| **3** | `git init` + push ל-`main` | ✅ **הושלם** — `1d1c878` + `0d436fd` על `main` |
| **4** | Settings → Actions → Read and write permissions | ✅ (workflow push הצליח — כנראה מוגדר) |
| **5** | הרצה ידנית ראשונה (`workflow_dispatch`) | ✅ run `28736400758` — **success** |
| **6** | אימות commit אוטומטי + ledgers/state v2 | ✅ ראה למטה |

**שלב 1 (פרטים):** נוספו ל-`.gitignore`: `.venv/`, `venv/`, `env/`, `.env`, `.idea/`, `.vscode/`, `*.log`, `btc_1s.csv`, `btc_10m.csv`.  
`fade/output/` → `fade/output/*` (תיקון: `!` ל-jsonl/state/pre_registration עובדים; `training_suite.json` ושאר output נשארים מחוץ ל-git).

**לפני commit (שלב 3):** לבדוק `git status` — קבצים גדולים שלא נדרשים ל-workflow.  
**מוחרגים ב-gitignore:** `btc_1s.csv` (~40MB), `btc_10m.csv` (~33MB) — מחקר חד-פעמי, לא refresh.  
`btc_5m`/`eth_5m` (~60MB כל אחד) **ייכללו** — נדרשים ל-sparse PRIMARY multi-res.

**Remote:** `https://github.com/junamiweb/FADE_PROJECT.git`  
**Commits על GitHub:** `1d1c878` (initial), `0d436fd` (gitignore tokens + docs).  
**Push:** ✅ הצליח (2026-07-05). אזהרת GitHub: `btc_5m`/`eth_5m` >50MB (מותר, לא מומלץ — לא חוסם push).

**הרצה ידנית ראשונה (2026-07-05 ~09:35 UTC):**  
- Workflow: [run 28736400758](https://github.com/junamiweb/FADE_PROJECT/actions/runs/28736400758) — **success**  
- Auto-commit: `auto: refresh + outcome tracker 2026-07-05 09:35 UTC`  
- `eth_candidate_state.json`: `position=1`, `bars_in_position=3`, `last_bar_ts=2026-07-05 09:00` — **לא התאפס**  
- `eth_candidate_outcomes.jsonl`: pre-fix 05:00 (next-bar) + bar 09:00 `min_hold_no_trade`; **0 v2 hold-cycles** עדיין (לא היה `traded=true` מאז v2)  
- **cron:** מופעל (`5 * * * *`) — רק אחרי אימות זה; לעקוב אחרי 2–3 runs

**אבטחה:** PAT ב-`docs/git-token-junamiweb.txt` — **לבטל מיד** ב-GitHub (חשוף בצ'אט); הקובץ ב-`.gitignore`.

---

## ★ Batch 38 — ML suite + TikTok chart patterns (sandbox)

**Holdout 70/30 exploratory — לא production, לא core ML.**

### ML challenger suite (`ml_challenger_suite.py`)

| model | BTC hit | vs FADE 54.64% | ETH hit | vs FADE 53.27% |
|-------|---------|----------------|---------|----------------|
| hist_gradient_boosting | 53.31% | −1.33pp | 52.77% | −0.50pp |
| random_forest | 53.24% | −1.40pp | 52.58% | −0.69pp |
| extra_trees | 53.14% | −1.50pp | 52.94% | −0.33pp |
| gradient_boosting | 52.95% | −1.69pp | 52.66% | −0.61pp |
| logistic_regression | 52.64% | −2.00pp | 52.68% | −0.59pp |
| knn_k15 | 51.87% | −2.77pp | 51.56% | −1.71pp |

**מסקנה:** **אף מודל לא מנצח rules** (כמו batch 28–29).

### TikTok chart patterns (`tiktok_chart_holdout.py`)

**לא scraping TikTok** — שמות דפוסים פופולריים → גיאומטריה causal על OHLC.

| asset | path_lean3 | path_tiktok | Δ |
|-------|------------|-------------|---|
| BTC | 54.04% | 53.64% | **−0.40pp** REJECT |
| ETH | 53.55% | 53.29% | **−0.26pp** REJECT |

522/474 rules = **dilution** (כמו path_candles).

```bash
python -m fade.pipeline.ml_challenger_suite btc_1h.csv eth_1h.csv --save
python -m fade.pipeline.tiktok_chart_holdout btc_1h.csv eth_1h.csv --save
```

---

## ★ Batch 39 — ML suite extended (LightGBM, CatBoost, stacking, …)

**Holdout 70/30 sandbox — 52 models × 4 feature sets × 2 assets. לא production.**

### Feature sets

| set | atoms / notes |
|-----|----------------|
| path_lean3 | close_pos, range_pct, streak_signed |
| plus7 | lean3 + 4 extras |
| full9 | 9 path atoms |
| ml_rich | full9 + engineered lags/rolling |

### Best per asset (vs FADE rules ref)

| asset | FADE ref | best model | hit | Δ vs FADE | beats rules? |
|-------|----------|------------|-----|-----------|--------------|
| BTC | 54.64% | catboost (full9) | 53.65% | −0.99pp | **no** |
| ETH | 53.27% | mlp_64_32 (plus7) | 53.19% | −0.08pp | **no** |

גם stacking (RF+ET+HGB), LightGBM, CatBoost, SVM, AdaBoost, MLP, GaussianNB, baselines — **אף אחד לא עבר rules**.

**מסקנה:** אישור batch 38 — **rules > ML** גם עם מודלים ופיצ'רים נוספים. ETH MLP קרוב (−0.08pp) אך לא מנצח; לא מספיק לשינוי production.

```bash
python -m fade.pipeline.ml_challenger_suite btc_1h.csv eth_1h.csv \
  --features path_lean3 plus7 full9 ml_rich --save
# → fade/output/ml_challenger_suite.json (study ml_challenger_suite_v2_batch39)
```

---

**מנוע:** `path_lean3` (close_pos, range_pct, streak_signed) — ברירת מחדל production.  
**תחזית:** `forecast_tiers` → שכבות CONVICTION / FREQUENT / BALANCED / QUALITY + **PRIMARY** מומלץ.  
**יועץ:** `outcome_tracker` — sparse PRIMARY + **ETH candidate track** (JSONL forward).  
**Challengers (sandbox, לא core):** GB + LSTM — **נמוכים מ-rules** על holdout.

### תחזית PRIMARY (2026-07-05, דאטה עד ~05:30 UTC)

| נכס | PRIMARY | מקור | VR | הערה |
|-----|---------|------|-----|------|
| **BTC** | DOWN 54.6% | conviction STREAK≥2 | LOW_VR | סתירה: balanced UP |
| **ETH** | UP 52.6% | frequent (15+30+60) | — | frequent מנצח conviction DOWN |

### דיוק מאומת (holdout 30%, OOS) — ⚠️ lockbox נמוך יותר

| שכבה / מודל | BTC holdout 30% | BTC lockbox 18% | ETH lockbox |
|-------------|-----------------|-----------------|-------------|
| path_lean3 rules | **54.04%** (batch 30 refresh) | **52.77%** ★ true OOS | 52.75% |
| conviction-primary replay | 53.1% | 53.16% | 51.62% |
| ml_challenger GB | 53.21% | 52.49% |
| ml_challenger LSTM | 53.54% | 52.95% |
| path_candles (נדחה) | 53.31% | — |

### מסקנות מרכזיות (עד batch 30)

1. **Rules > ML** — GradientBoosting ו-LSTM על אותם atoms **לא** מנצחים path_lean3.
2. **דילution חוזר** — streak_big, path_both, path_candles **נדחו**; path_lean3 נשאר.
3. **Conviction סטציונרי היסטורית** — 20/20 שנה×חוק (BTC+ETH) מעל 50%, **אך 2025–26 חלש יותר**.
4. **VR filter** — HIGH_VR חלש יותר (54.1% reversal); filter: streak≥3 ב-HIGH_VR.
5. **PRIMARY logic** — frequent מנצח conviction בסתירה; abstain אם confidence חלש.
6. **★ PnL** — edge כיווני אמיתי **לא סחיר** ב-5 bps (batch 10 אושר batch 30).
7. **★ Lockbox** — true OOS ~52.8% (לא 54.6%); multiple-comparisons inflation.
8. **★ Holm** — path_lean3 הכי טוב אך לא ייחודי; כל 7 הסטים עוברים correction.
9. **★ Decay** — 2025–26 חלש 4–5pp; BTC H2 monotonic pre-2024 (corr −0.48).
10. **★ Stocks** — SPY/AAPL rev≈0; BTC עדיין +0.045 (לא matured fully).
11. **★ Regime PnL** — holdout: BTC +76.8% / ETH +94.2%; **lockbox: BTC −10.9% / ETH +23.2%** (batch 33).

---

## מקור הרעיון (`chatGPT.txt` — מלא)

| שלב | שורות | מה קרה |
|-----|-------|--------|
| **חזון** | 1–7 | כלי שצופה בבורסה/קריפטו, גרפים, למידה per-item, % סיכוי, משתפר |
| **10 שכבות** | 8–341 | Data Lake, CV, LSTM, regimes — אבל בסוף: BTC, 4h, כיול, backtest |
| **ADE** | 345–900 | Atom Discovery Engine, anti-self-deception, פיננסים = pressure test |
| **FADE** | 910–919 | **אתה** הצעת FADE → **ChatGPT**: Financial Atom Discovery Engine |
| **v0.1 spec** | 1118–2365 | 5 atoms, 1H, OHLCV, rules, 4h — **בלי regimes** |
| **prototype** | 2368–2537 | קוד בסיסי (bootstrap) — לא walk-forward |
| **DevOps** | 2750+ | imports, `python -m fade.pipeline.main`, הפניה ל-Cursor |

**מפת דרכים ChatGPT:** v0.1 → v0.2 (עוד נכס) → v0.3 (regimes) → ADE אוניברסלי

---

## ★★ אטומי-מסלול במנוע: האבחנה של תקרת ה-53% — 2026-07-03

**המהלך:** הוזרק אטום מסלול `streak_signed` (אורך רצף עם סימן, סיבתי, inclusive) ל-`atoms.py` + סטים חדשים (`core5_path`, `path_min`) + ביקוט מודע-מבנה (סף קבוע ‎±2 ל-streak: LOW=רצף-ירידה≥2, HIGH=רצף-עלייה≥2, MID=קצר) דרך `atom_fixed_thresholds` ב-Config.

**התוצאה (strict holdout, btc_1h, 70/30, frozen dev-rules):**

| קבוצת חוקים | חוקים | כיסוי | דיוק OOS |
|---|---|---|---|
| core5 (בסיס) | 98 | 19,719 | 53.26% |
| core5_path (כולל streak) | 222 | 22,668 | 53.21% |
| מכילים streak | 124 | 20,517 | 53.42% |
| **streak=LOW/HIGH (רצף≥2)** | 105 | 10,949 | **54.60%** (~9σ) |

**האבחנה שאושרה מקצה-לקצה:** האטום נושא edge אמיתי וחזק יותר (54.6% על 11K OOS) — **אבל רק כשמבודדים אותו**. כשהוא נבלע ב-222 חוקי קוניונקציה, ה-edge **נמהל חזרה ל-53.2%**. כלומר: **צוואר-הבקבוק אינו אוצר-המילים אלא מנגנון האגרגציה** — דקדוק ה-AND + סינון-confidence ממצע חוק ספציפי-חזק עם המון חוקים גנריים-חלשים.

**המהלך האדריכלי הבא (נגזר):** לא "עוד אטומים" אלא **לתת לחוק ספציפי-חזק לבטא את עצמו בלי דילול** — למשל חיזוי משוקלל-specificity/precision, או מסלול-חוקים ייעודי (path-rule track) שלא ממוצע עם קוניונקציות גנריות.

**★ הפתרון (`specificity_test.py`) — פרסימוניה מנצחת שקלול:** נבדקו 8 שיטות אגרגציה על holdout קפדני. **שקלול-specificity נכשל לגמרי** (size/size2/rarity/edge/argmax/mechanism_gate כולם ≈53.2% על core5_path — זהים לבסיס). אבל סט **`path_min` (4 אטומים: return_1h, volatility, volume_zscore, streak_signed) נתן 54.05%** על 14,264 חיזויים OOS (p=0.002) — עם שקלול רגיל, בלי טריק.

| סט | אטומים | דיוק OOS |
|---|---|---|
| core5 | 5 | 53.26% |
| core5_path | 6 | 53.21% |
| **path_min** | **4** | **54.05%** |

**האבחנה המאוחדת:** הדילול לא נפתר בצד האגרגציה — הוא **נמנע בצד אוצר-המילים**. הסרת אטומים מתואמים (return_6h, trend_slope) שמייצרים המוני קוניונקציות גנריות → אות הרצף לא נמהל מלכתחילה. שתי האבחנות (קולינאריות + דילול) מתאחדות: **בסיס אורתוגונלי רזה + אטום מסלול = שבירת תקרת 53% על btc_1h**.

**סוכנים ממוקדים (batch 17):**
- [חיפוש סט רזה](61c5d6f0-cbb1-4e5f-a327-2182063adb9c) (`lean_search.py`): אחרי תיקון holdout — מנצח btc_1h = **`path_lean3`** (close_pos, range_pct, streak_signed) **54.64%** (14 חוקים, n=12,157). 3 אטומים > 4. path_min=54.05% במקום 2.
- [הכללת path_min](bc59f990-26f0-430d-b7b5-21fe90ac51d0) (`path_min_generalization.py`): path_min מנצח core5 ב-**2/5** (btc_1h, btc_15m בלבד). ETH/30m/5m — core5 שווה או טוב יותר. **יתרון btc_1h לא מכליל** לרוב הנכסים/רזולוציות (edge חיובי כן קיים בכולם, אבל לא מעל core5).
- תיקון ליבה קטן: `holdout.py` — baseline מומנטום מ-`return_1h` כשאין `return_6h` (אפשר הערכת סטים רזים בלי workaround).

**אטום מסלול שני `streak_big` (מותנה-מגניטודה) — נדחה ביושר (batch 18):** נוסף אטום רצף שסופר רק תנועות גדולות (|תשואה| > k·תנודתיות-נגררת). **המנגנון לא אישש את ההשערה:** רצף-גדול מתהפך בערך כמו רצף רגיל (up≥2: 46.3% מול 44.5% — אפילו מעט פחות), רק up≥4 חזק (27.5%) אך n=51 (אשליה). **holdout:** path_big=53.98% < path_lean3=54.64%; **path_both (שני הרצפים)=53.60%** — הוספתו **מדללת** בחזרה. תוצאה שלילית שמחזקת את אבחנת הדילול: עוד אטום = עוד חוקים = מיהול. streak_big נשאר ב-pool למחקר אך לא נכנס לסט מנצח.

**למה לא מכליל? (`generalization_why.py`, batch 19):** path_lean3 מנצח core5 ב-**4/5** נכסים (לא 2/5 כמו path_min), אבל **גודל היתרון תלוי בעוצמת המנגנון המקומי** על ה-holdout:
- corr(rev_index, delta)=**+0.59** — ככל שהיפוך-רצפים חזק יותר בנכס, היתרון גדול יותר.
- corr(close_pos_edge, delta)=**+0.58** — אותו דפוס לאטום close_pos.
- btc_1h: rev=0.046, cp_edge=0.041 → **delta +1.38%** (הכי חזק).
- eth_1h: rev=0.041, cp=0.036 → delta +0.25% (קיים אך זעיר).
- btc_5m: rev=0.017, cp=0.008 → delta +0.07% (כמעט אפס).
- btc_30m: rev=0.038 אך delta **−0.05%** (יוצא דופן / רעש).

**מסקנה:** היתרון **לא ייחודי ל-btc_1h** — המנגנון קיים בכל הנכסים, אבל **מתדלדל** ככל שהרזולוציה מיקרו יותר / המנגנון חלש יותר. path_lean3 מכליל טוב יותר מ-path_min; הבעיה היא **עוצמה**, לא היעדר מנגנון.

**הגברת עוצמה: conviction gating (`conviction_gate.py`, batch 20)** — מחליפים כיסוי בדיוק, שני צירים עצמאיים, holdout מובהק:

| רצף≥L | כיסוי | דיוק | | הסכמת TF≥K | כיסוי | דיוק |
|---|---|---|---|---|---|---|
| 2 | 47.5% | 54.6% | | 2 | 48% | 54.3% |
| 3 | 21.6% | 55.3% | | 3 | 11.3% | **58.3%** |
| **4** | 9.7% | **57.1%** | | **4 (פה-אחד)** | 1.1% | **62.6%** |
| 6 | 1.8% | 56.5% | | | | |
| 7+ | <1% | ~50% רעש | | | | |

**תובנות:** (1) דיוק עולה מונוטונית עם עומק הרצף עד שיא **57.1% ברצף≥4** (n=2,252, p=0.0005), קורס לרעש ב-7+ (תמיכה דלילה — עקבי עם sequence_sweep). (2) שער הסכמת-רזולוציות **מאשש** את ה-62.6% הקודם עם חזית מלאה: 3 מסכימים = 58.3% על כיסוי שמיש (11%, n=2,625). שני הצירים עצמאיים וניתנים להגברה.

**שילוב + כיול + ETH (`conviction_combo.py`, batch 21):**

| שילוב (BTC, next 1h) | כיסוי | דיוק |
|---|---|---|
| רצף≥2 + 3 TF מסכימים | 8.9% | **58.2%** (n=2,067) |
| רצף≥3 + 3 TF מסכימים | 3.9% | **59.6%** (n=909) |
| רצף≥2 + 4 TF פה-אחד | 1.1% | **62.6%** (n=265) |

**כיול:** כל רמת conviction = אחוז מכויל כן (empirical hit על holdout). טבלה מלאה ב-`conviction_combo.py`. דוגמאות: streak≥4 → **57%**, combo L≥3 K≥3 → **60%**, combo L≥2 K≥4 → **63%**.

**ETH (multi-res מלא מ-batch 24):** מנגנון דומה ל-BTC — streak≥3 → 55.1%, multi-TF conviction פעיל עם eth_5m/15m/30m/1h.

**שילוב בתחזית (`forecast_tiers.py`, batches 22–29):**
- שכבות: CONVICTION → FREQUENT → BALANCED → QUALITY
- **PRIMARY:** frequent מנצח conviction בסתירה; abstain על conflict חלש
- VR regime מוצג; HIGH_VR מוריד streak≥2 ל-streak≥3

**יציבות בזמן (`conviction_stability.py`, batch 22):** מנגנון **סטציונרי** — streak≥3 ו-combo r≥2+3TF מעל 50% ב**כל שנה** 2017–2026 (20/20). 2025 חלש יותר (53.2%) אך עדיין חיובי. המנגנון לא ארטיפקט של תקופה אחת.

**path_lean3 כברירת מחדל + אימון (batch 23):** `DEFAULT_ATOM_SET = "path_lean3"`; `lean_config()` ב-main/forecast/forecast_tiers/inference.

**זיכרון path_lean3 (walk-forward, per-asset):**

| נכס | חוקים | hit OOS | lift |
|-----|-------|---------|------|
| btc_1h | 15 | 53.78% | +3.78% |
| eth_1h | 17 | 53.69% | +3.69% |
| btc_15m | 13 | 54.99% | +4.99% |
| btc_30m | 14 | 54.00% | +4.00% |
| eth_15m | 11 | 54.53% | +4.53% |
| eth_30m | 11 | 53.48% | +3.48% |
| btc_5m | 4 | 53.47% | +3.47% |
| eth_5m | 3 | 53.63% | +3.63% |

**דאטה:** BTC+ETH — 5m/15m/30m/1h מ-Binance (2017→). `download_history.py refresh` — incremental ל-8 קבצים.

**training_suite (path_lean3):** cum-hit mean **54.27%**, 10/10 jobs >50%, verdict: consistent positive edge.

---

## Production stack — batches 24–29 (כרונולוגיה)

### batch 24 — אימון רחב + ETH multi-res
- הורדת eth_15m/eth_30m; conviction multi-TF ל-ETH
- `forecast_tiers` + `conviction.py` — prefix אוטומטי (btc_/eth_)

### batch 25 — PRIMARY + יציבות ETH
- **`primary`** ב-forecast_tiers: שורה מומלצת אחת
- `conviction_stability btc_1h eth_1h` → **20/20** גם ל-ETH
- עדיפות ראשונית: quality > conviction > frequent > balanced

### batch 26 — conflict + replay + refresh
- **כלל סתירה (מעודכן):** FREQUENT vs conviction → **FREQUENT מנצח**
- `primary_replay.py` — holdout conviction path: BTC 53.1%, ETH 52.2%
- `python download_history.py refresh`

### batch 27 — outcome tracker
- `fade/output/primary_outcomes.jsonl` — log + score + report
- `python -m fade.pipeline.outcome_tracker run`

### batch 28 — 3 סוכנים (challenger + VR + abstain)
| רכיב | תוצאה |
|------|--------|
| `ml_challenger.py` | GB: BTC 53.21%, ETH 52.49% — **< rules** |
| `vr_gate_test.py` | reversal streak≥3: NORMAL 56.2% > HIGH 54.1% |
| PRIMARY abstain | conflict + confidence <53% / <52.5% |
| ASCII CLI | תוויות tier באנגלית בקונסול |

### batch 29 — 3 סוכנים (candles + VR filter + LSTM)
| רכיב | תוצאה | החלטה |
|------|--------|--------|
| `candle_patterns.py` + `candle_holdout.py` | path_candles 53.31% < lean3 54.04% | **REJECT** |
| VR→conviction filter | HIGH_VR דורש streak≥3 | **INTEGRATE** |
| `ml_challenger --lstm` | BTC 53.54% > GB, **< FADE 54.64%** | sandbox |

**נדחו (דילution / שלילי):** streak_big, path_both, path_candles, news, funding (חלש), regime-weighting.

**קולינאריות אטומים (`atom_redundancy.py`, [סוכן מומחה](80621324-7a83-4683-bcee-caae6015d996)):** 9 האטומים מתקבצים ל-~3 אשכולות סמנטיים (מומנטום/מיקום-נר, תנודתיות/טווח, נפח) — לא 2 מימדים כפי שחזינו, אבל עם זוגות |r|>0.6 כבדים (return_1h↔return_accel 0.72, return_6h↔trend_slope 0.70, volatility↔range_pct 0.67). PCA: 7 רכיבים ל-95% שונות. בסיס אורתוגונלי מינימלי מומלץ: `close_pos`, `return_6h`, `volume_zscore`, `volatility`. **משמעות משולבת:** גם אוצר-המילים (עודף אטומים מתואמים) וגם האגרגציה (דילול חוק חזק) תורמים לתקרת 53% — `streak_signed` הוא מימד חדש שלא נכלל בניתוח זה.

---

## ★ מנגנון חדש: היפוך רצפים (mean-reversion תוך-יומי) — 2026-07-03

**הפער שנסגר:** המנוע הקלאסי של FADE הוא *חסר-זיכרון* — הוא מנקד צירוף אטומים של **נר בודד** ברגע T ולעולם לא שואל מה **רצף** המצבים האחרונים אומר. שני מודולים חדשים סוגרים את הפער:

- `fade/pipeline/sequence_patterns.py` — מנבא n-gram: האם רצף כמו "UUU" חוזה את הנר הבא? (holdout + permutation + Bonferroni)
- `fade/pipeline/trend_structure.py` — מפת מומנטום/היפוך: אחרי N נרות באותו כיוון, מה סיכוי ההמשכה? (על פני רזולוציות)

**הממצא (BTC, holdout 30% לא-נראה, כולו מובהק p≤0.0005):**

רצפים **מתהפכים**, וההיפוך **מתחזק מונוטונית** עם אורך הרצף:

| רצף (15m) | P(המשך) | | רצף (1h) | ניבוי | דיוק |
|---|---|---|---|---|---|
| 2 | 47.7% | | UUU | ירידה | 55.5% |
| 3 | 45.7% | | UUUU | ירידה | 56.9% |
| 4 | 44.5% | | DDD | עלייה | 55.0% |
| 5 | 43.3% | | DDDD | עלייה | 57.2% |
| 6+ | **40.6%** | | 0-- (mag3) | עלייה | **60.6%** |

**מבנה על פני טיימפריימים:**
1. **נר בודד (+1) = הליכה אקראית** (~50%, לא מובהק ב-15m/30m/1h) — לכן המנוע החד-נרי תקוע ~53%. ה-edge חבוי ברצף.
2. **ההיפוך מתחזק עם הטיימפריים**: הכי חזק 15m–1h. ב-5m הנר הראשון מראה **מומנטום** (מיקרו-מבנה), היפוך נכנס מ-2 נרות.
3. **ברמה היומית ההיפוך נעלם** (לא מובהק) — חוסר-היעילות הוא **תוך-יומי בלבד**; ביומי BTC ~ random walk.

**המשמעות:** זהו מנגנון החיזוי הכי נקי, מונוטוני ואינטרפרטבילי בפרויקט. הוא מסביר *למה* המנוע החד-נרי מוגבל, ומצביע על כיוון חזק — שילוב מצב-רצף כאטום.

**עומק זיכרון (`sequence_sweep.py`, k=2..8):** ה-edge האגרגטיבי על ה-holdout נרווה סביב **k≈5–6** בכל הרזולוציות, והכי חזק ב-**1h (+2.65% ב-k=6)**. מעבר לזה (k≥7) מספר הדפוסים ששורדים Bonferroni **צונח ל-0** — למרות ש-best_hit הבודד מטפס עד 0.68. זהו **קו הגבול של ה-overfitting**: רצפים ארוכים = תמיכה דלילה + עונש השוואות מרובות → הדיוק הגבוה הוא אשליה, וה-FADE מסמן זאת ביושר (0 שורדים). מסקנה: **זיכרון אפקטיבי של השוק ~5–6 נרות**; מעבר לזה רק self-deception.

**מבנה רב-סקאלה (`scale_structure.py`) — מיקרו/מאקרו + קורולציות:**

1. **מעבר פאזה ממיקרו למאקרו (סולם היפוך, holdout):** ‎1s = **מומנטום** (המשך 60%, rev_index −0.10) — התמדת order-flow; דקות→שעה = **היפוך** שמתחזק לשיא ב-**1h (rev_index +0.046)**; **יומי = יעיל** (לא מובהק, p=0.42). מעבר פאזה נקי momentum→reversion→random-walk. *(1s: טווח קצר ורגיש לטיק — רמז, לא ודאות.)*
2. **מיקרו×מאקרו:** ההיפוך התוך-שעתי חזק (~54–55%) בכל ארבעת שילובי (רצף תוך-יומי × מגמה יומית). המגמה היומית מווסתת רק מעט (חזק יותר כשהרצף מיושר עם המאקרו) → ההיפוך מנגנון עצמאי, לא רק תיקון-בתוך-טרנד.
3. **קורולציה בין דפוסי זמן:** קורולציית תשואות בין רזולוציות = 1.0, אבל קורולציית **אות-ההיפוך** ≈ **0** (−0.07..−0.006). כל טיימפריים תופס מידע חצי-עצמאי → מסביר *למה* הסכמת-רזולוציות שיפרה דיוק (אותות כמעט אורתוגונליים).

**סוויטה משולבת (`pattern_suite.py`) — ארבעה כיוונים ביחד:**

| בדיקה | ממצא מרכזי |
|--------|------------|
| **A) מעבר פאזה** | מומנטום עד ~30s → היפוך מ-**60s** (crossover 30s→60s על דאטת 1s, ~7 ימים) |
| **B) ETH×BTC** | ETH rev_index +0.042 ≈ BTC +0.046; אותות corr **0.61**; הסכמה משותפת → **55.1%** (n=7118) vs סתירה 47.6% |
| **C) תנודתיות** | היפוך חזק בשתי המשטרים (~54.5%); פער high−low רק **+0.002** — כמעט ללא מודולציה |
| **D) אנסמבל** (חיזוי שעה **קדימה**): 1h בודד 52.8% → רוב 2+ **54.3%** → פה-אחד 4 TF **62.6%** (n=265, דליל — רמז חזק, לא ודאות) |

**מסקנה משולבת:** מבנה momentum→reversion קיים מ-60s עד 1h; ETH חולק אותו דפוס; תנודתיות כמעט לא משנה; שילוב אורתוגונלי של טיימפריימים מוסיף edge כשכולם מסכימים (בדומה ל-multi-res קודם, עכשיו על מנגנון הרצפים).

---

## מה בנוי ✅

```
fade/
  core/     atoms, candle_patterns, events, rules, targets, evaluator,
            calibration, predictor, significant_changes, regimes, conviction
  memory/   positive_rules_{asset}.json, negative_patterns_{asset}.json,
            calibration_{asset}.json, regime_stats_{asset}.json
  pipeline/ main, backtest, feasibility, forecast, forecast_tiers, inference,
            replay, plot, report, holdout, minute_vol, multi_res, horizon_sweep,
            grid_search, magnitude_sweep, learning_sim, learning_sim_multi,
            training_suite, correlation,
            sequence_patterns, trend_structure, sequence_sweep, scale_structure,
            pattern_suite, atom_redundancy, specificity_test, lean_search,
            path_min_generalization, generalization_why,
            conviction_gate, conviction_combo, conviction_stability,
            primary_replay, outcome_tracker,
            ml_challenger, ml_challenger_lstm,
            vr_gate_test, vr_conviction_test, candle_holdout
  output/   primary_outcomes.jsonl, training_suite.json, PNGs (gitignored)
  utils/    cache, logging
download_history.py   hourly, interval, asset, ethall, refresh
```

**הרצה — production (מומלץ):**
```bash
# רענון דאטה (incremental)
python download_history.py refresh

# תחזית PRIMARY + שכבות
python -m fade.pipeline.forecast_tiers btc_1h.csv
python -m fade.pipeline.forecast_tiers eth_1h.csv

# outcome tracker (יועץ)
python -m fade.pipeline.outcome_tracker run

# אימון + עדכון זיכרון (path_lean3)
python -c "from fade.config import lean_config; from fade.pipeline.main import run; run('btc_1h.csv', config=lean_config())"
```

**הרצה — validation / research:**
```bash
python -m fade.pipeline.holdout btc_1h.csv
python -m fade.pipeline.conviction_stability btc_1h.csv eth_1h.csv
python -m fade.pipeline.primary_replay
python -m fade.pipeline.vr_gate_test btc_1h.csv
python -m fade.pipeline.vr_conviction_test
python -m fade.pipeline.candle_holdout
python -m fade.pipeline.training_suite
python -m fade.pipeline.ml_challenger              # sandbox GB
python -m fade.pipeline.ml_challenger --lstm       # sandbox LSTM
```

**הרצה — legacy / מלא:**
```bash
python -m fade.pipeline.report btc_1h.csv
python -m fade.pipeline.report btc_1h.csv --fast   # בלי holdout (מהיר)

# למידה מלאה + עדכון זיכרון
python -m fade.pipeline.main btc.csv

# תחזית מהירה (קורא מ-positive memory)
python -m fade.pipeline.forecast btc.csv
python -m fade.pipeline.forecast btc.csv --json

# replay היסטורי — lift לאורך חלונות מתרחבים
python -m fade.pipeline.replay btc.csv

# גרפים (matplotlib)
python -m fade.pipeline.plot btc.csv

# מבחן: האם regime-weighting משפר את ההסתברות? (split כרונולוגי)
python -m fade.pipeline.regime_eval btc.csv

# מבחן holdout קפדני — האם ה-edge שורד על נתונים שלא נראו?
python -m fade.pipeline.holdout btc.csv

# מבחן היתכנות
python -m fade.pipeline.feasibility btc.csv
python -m fade.pipeline.feasibility eth.csv

# סימולציית למידה חיה (מסתיר עתיד, חושף לאט)
python -m fade.pipeline.learning_sim btc_1h.csv
python -m fade.pipeline.learning_sim btc_1h.csv --train-mode fixed --train-window 20000

# סוויטת אימונים (מטבעות × טווחים × רזולוציות)
python -m fade.pipeline.training_suite

# קורלציה בין נכסים
python -m fade.pipeline.correlation btc_1h.csv eth_1h.csv

# למידה מתקדמת — הסכמת רזולוציות על עתיד מוסתר
python -m fade.pipeline.learning_sim_multi

# חדשות/סנטימנט — הורדה מצטברת + מבחן
python download_news.py --query "bitcoin" --start 2017-01-01 --out news_btc.csv
python -m fade.pipeline.news_test --verify        # בדיקת יישור תאריכים בלבד
python -m fade.pipeline.news_test                 # + מבחן holdout מלא
```

---

## שיפורי דיוק — v0.3 (סריקת נוסחה מיטבית)

### 0. סורק־רשת עם הגנת שלושה־חלקים (`grid_search.py`)
כדי למצוא את הנוסחה הטובה ביותר בלי לרמות את עצמנו (בעיית ההשוואות המרובות), כל סדרה מפוצלת כרונולוגית ל-3:
פיתוח 55% (כרייה) → אימות 25% (דירוג ובחירת מנצח) → **מבחן 20% (נגיעה אחת בלבד, על המנצח)**.

סרקנו 27 שילובים: רזולוציה {15/30 דקות, שעה} × אופק {1,2,3} × אטומים {core5, plus7, full9}.

**המנצח:** שעה, אופק 1, **core5** (5 האטומים המקוריים).

| שלב | skill | פגיעה |
|-----|-------|-------|
| אימות (בחירה) | +0.0564 | 0.556 |
| **מבחן (נגיעה אחת)** | **+0.0253** | 0.526 (p=0.0033) |
| מס' ההשוואות (val→test) | **−0.0311** | — |

**שתי מסקנות חשובות:**
1. **פחות זה יותר:** הוספת אטומים (plus7/full9) לא שיפרה — אותו skill או פחות, למרות פי 3-7 יותר חוקים. הפשטות מנצחת.
2. **מס' החיפוש חשוף:** האימות הראה +5.6% אבל המבחן הכן +2.5%. הפער (3.1 נקודות) הוא בדיוק ניפוח החיפוש ש-FADE נועד לחשוף. ה-+2.5% הוא האמת, והוא מובהק (p=0.0033) ותואם את כל שאר מבחני ה-holdout.

הרצה: `python -m fade.pipeline.grid_search`

### 1. שילוב רזולוציות (`multi_res.py`)
שלוש רזולוציות שכל אחת עוברת holdout (15/30 דקות, שעה) חוזות את אותו יעד — כיוון שעה קדימה — ומצביעות יחד.

| מודל | כיסוי | פגיעה | lift | p |
|------|-------|-------|------|---|
| 15 דקות לבד | 19,387 | 0.531 | +0.031 | 0.0033 |
| 30 דקות לבד | 19,095 | 0.530 | +0.030 | 0.0033 |
| שעה לבד | 19,719 | 0.533 | +0.033 | 0.0033 |
| **שלושתן מסכימות (65%)** | 12,363 | **0.547** | **+0.047** | 0.0033 |

**מסקנה:** דרישת הסכמה חוצת־רזולוציות מעלה דיוק מ-~0.533 ל-**0.547** (בתמורה לכיסוי — פועל 65% מהזמן). שיפור אמיתי שבנוי על מה שהוכח.

הרצה: `python -m fade.pipeline.multi_res --target-min 60`

### 2. סריקת אופק (`horizon_sweep.py`)
בדיקת holdout לכל אופק. `skill` = פגיעה פחות ה-null (מנטרל דריפט).

| רזולוציה | אופק הכי טוב | skill |
|----------|-------------|-------|
| שעה | **1 צעד** | +0.0321 |
| 30 דקות | **1 צעד** | +0.0306 |
| 15 דקות | **1 צעד** | +0.0295 |

**מסקנה חד־משמעית:** האות הכי חזק באופק **צעד אחד** ודועך ככל שהאופק ארוך יותר. ברירת המחדל הקודמת (4 צעדים) הייתה חלשה בכ-50%. **`Config.forward_horizon` שונה מ-4 ל-1.**

הרצה: `python -m fade.pipeline.horizon_sweep btc_1h.csv`

### 3. יעד לפי גודל תנועה (`magnitude_sweep.py`, `core/targets.py`)
במקום "כל כיוון" — חיזוי תנועה **משמעותית** מעל סף X%. כרייה ומבחן holdout מחדש לכל סף.

| סף | שעה: hit | skill | כיסוי | 15 דק': hit | skill | כיסוי |
|----|---------|-------|-------|------------|-------|-------|
| 0% (כיוון בלבד) | 0.533 | +3.2% | 85% | 0.531 | +3.0% | 71% |
| **0.1%** | 0.537 | **+3.7%** | **70%** | 0.530 | +2.9% | 48% |
| 0.5% | 0.510 | +0.7% | 19% | 0.560 | +6.0% | 5% |
| **1.0%** | **0.573** | **+7.3%** | 5% | **0.616** | **+11.4%** | 1% |

**מסקנה:** ככל שהסף גבוה יותר, הדיוק עולה משמעותית — אבל הכיסוי יורד. זה בדיוק מה שרצינו: הפרדה בין כישרון לדריפט. הנוסחה המעשית:
- **תחזית תכופה:** סף 0% + שילוב רזולוציות (כיסוי גבוה, ~54.7% בהסכמה)
- **תחזית איכותית:** סף 0.1% על שעה (70% כיסוי, skill +3.7%)
- **איתות חזק:** סף 1% על 15 דקות (61.6% hit, skill +11.4%, אבל רק 1% מהזמן)

`Config.move_threshold` — ברירת מחדל 0 (כיוון). להגדיר 0.001 לתחזית איכותית.

הרצה: `python -m fade.pipeline.magnitude_sweep btc_1h.csv`

---

## השוואת רזולוציות — מבחן holdout קפדני (v0.3)

כל השורות עברו את אותו מבחן: חוקים נבחרים על 70%, מוקפאים, נבדקים על 30% שבהסגר, עם מבחן תמורות.

| נתונים | שורות | חוקים | hit | null | p-value | פסק דין |
|--------|-------|-------|-----|------|---------|---------|
| יומי, 11 שנה (Yahoo) | 4,307 | 19 | 0.521 | 0.527 | 0.73 | נכשל |
| שעתי, שנתיים (ישן) | 17,337 | 16 | 0.521 | 0.509 | 0.057 | גבולי |
| **שעה, 8 שנים (Binance)** | 77,671 | 93 | 0.525 | 0.504 | **0.0033** | עבר |
| **30 דקות** | 155,323 | 112 | 0.527 | 0.502 | **0.0033** | עבר |
| **15 דקות** | 310,632 | 85 | 0.527 | 0.502 | **0.0033** | עבר |
| **10 דקות** | 465,949 | 72 | 0.524 | 0.500 | **0.0033** | עבר |
| **5 דקות** | 931,881 | 63 | 0.521 | 0.500 | **0.0033** | עבר |
| דקה, סביב תנודתיות | 76,980 | 5 | 0.483 | 0.501 | 0.98 | נכשל |
| שנייה, 7 ימים | 605,000 | 235 | 0.601 | 0.591 | 0.0033 | דריפט (חלון קצר) |

**המסקנה המרכזית — טווח מתיקות של 5 דקות עד שעה:**
- **5 דקות → שעה** — האות אמיתי ומכליל בכל הרזולוציות (peak ~15-30 דקות). ה-null תמיד ~0.50, כלומר כישרון ולא דריפט. יותר נתונים הכריעו את הגבוליות השעתית (0.057 → 0.0033).
- **יומי** — אין אות. מה שנראה כרווח הוא רק מגמת עלייה (ה-null גבוה מהמודל).
- **דקה (סביב תנודתיות)** — אין אות, אפילו שלילי. הרעש שולט.
- **שנייה** — עבר טכנית אך ה-null ב-0.59 (דריפט של 7 ימים), לא מסקנה אמינה.

המבנה האטומי הוא תופעה תלוית־סקאלה: חי בטווח 5 דקות–שעה, נעלם למעלה (יום) ולמטה (דקה ומטה).

**קבצי נתונים:** `btc_{5m,15m,30m,1h}.csv`, `eth_{5m,15m,30m,1h}.csv` (Binance), `btc_daily.csv` (Yahoo), `btc_1m_vol.csv`, `btc_1s.csv`.  
**מוריד:** `download_history.py` — `hourly`, `interval`, `asset`, `ethall`, **`refresh`** (incremental, retry).

---

## סימולציית למידה חיה + סוויטת אימונים (v0.3)

במקום לחכות לזמן אמת: מסתירים את העתיד מהכלי, חושפים אותו לאט, ומודדים דיוק על כל נר שהכלי לא ראה בזמן כרייה. אלפי תחזיות OOS במקום מבחן אחד.

### סימולציית למידה (`learning_sim.py`)
```bash
python -m fade.pipeline.learning_sim btc_1h.csv --checkpoints 15
```

| מדד | BTC 1h מלא |
|-----|-----------|
| דיוק מצטבר OOS | **53.9%** |
| שיפור עצמי (חלונות מאוחרים − מוקדמים) | **−1.8%** |
| מסקנה | edge קבוע, **לא** משתפר עם יותר היסטוריה |

### בידוד הסייג: חלון קבוע (`--train-mode fixed`)
אם הירידה נובעת מכמות היסטוריה או מתקופה קשה יותר?

| מצב אימון | דיוק מצטבר | שיפור עצמי |
|-----------|-----------|-----------|
| מתרחב (כל העבר) | 53.9% | −1.8% |
| **חלון קבוע 20K נרות** | 53.3% | −1.7% |

**מסקנה:** גם עם חלון קבוע (אותה כמות היסטוריה בכל checkpoint) הדיוק יורד בתקופות המאוחרות. זה **אפקט תקופה** (2024–2025 קשה יותר מ-2018–2020), לא באג של חלון מתרחב.

### סוויטת אימונים (`training_suite.py`) — 8 הרצות
BTC + ETH, מוקדם/מאוחר/מלא, 30 דקות, 2 שנים.

| מדד | ערך |
|-----|-----|
| דיוק מצטבר ממוצע | **53.4%** |
| טווח | 51.3% – 54.7% |
| שיפור עצמי (7/8 ירידה) | ממוצע −1.3% |
| כל ההרצות מעל 50% | **8/8** |

### קורלציה BTC↔ETH (`correlation.py`)

| מדד | ערך |
|-----|-----|
| קורלציה שעתית | **+0.80** |
| אותו כיוון | 82% |
| BTC מוביל ETH (נר הבא) | **0.468** (אין lead-lag) |

**מסקנה:** ETH מיותר כמקור מידע נפרד. אין אות חוצה־נכסים.

### למידה מתקדמת — הסכמת רזולוציות (`learning_sim_multi.py`)
שלוש רזולוציות (15/30/60 דקות) כורות חוקים בנפרד על העבר שנחשף, חוזות את אותו עתיד מוסתר, ונדרשת הסכמה.

| מודל | דיוק מצטבר OOS | lift |
|------|---------------|------|
| שעה לבד | 53.9% | — |
| **שלוש מסכימות (64%)** | **55.8%** | **+1.9%** |

**מסקנה:** שיפור ההסכמה **שורד** גם בלמידה מתקדמת (לא רק holdout חד־פעמי). זה השיפור הכי אמין שיש לנו כרגע.

---

## חדשות/סנטימנט (GDELT) — נבדק ונדחה (v0.3)

הרעיון: להוסיף מקור מידע אורתוגונלי (לא-מחיר). הורדנו ארכיון GDELT יומי (טון סנטימנט + נפח סיקור) 2017–2025 עם חותמות פרסום אמיתיות, וטיפלנו בו כאטום רגיל שעובר את אותו מבחן holdout קפדני.

מוריד: `download_news.py` (מצטבר, מכבד rate-limit: מרווח 10ש', backoff 30–300ש'). מבחן: `python -m fade.pipeline.news_test --verify` (יישור) ובלי `--verify` (holdout מלא).

### בדיקת יישור תאריכים (קריטית — אין דליפה)
| lag | טון(D) מול תנועת | קורלציה |
|-----|------------------|---------|
| −1 | אתמול | **+0.099** (הכי חזק) |
| 0 | היום | +0.044 |
| **+1** | **מחר (היעד)** | **−0.006 (אפס)** |

**החדשות מגיבות למחיר, לא מקדימות אותו.** ה-lag העתידי האפסי מאשר שאין דליפה — אילו היה שם אות, זה היה חשוד.

### מבחן holdout (btc_daily + news, 2,555 ימים חופפים)
| מודל | חוקים | פגיעה | lift | p |
|------|-------|-------|------|---|
| מחיר (יומי) | 5 | 0.493 | −0.007 | 0.65 |
| חדשות | 2 | 0.515 | +0.015 | **1.00** |
| משולב | 26 | 0.509 | +0.009 | 0.35 |

**פסק דין: נדחה.** ה-p=1.00 של החדשות = כל התוצאה היא הטיית מחלקה (חוקים שאומרים "עלייה" תמיד), לא כישרון. חדשות = אינדיקטור מפגר, בלי edge מחוץ למדגם.

**המשמעות:** בדיוק כמו regime-weighting — רעיון סביר שנדחה במבחן כן. זה FADE עובד כמו שצריך. (הערה: 2019 חסר בגלל rate-limit; המסקנה יציבה על כל השנים הקיימות.)

### זווית שנייה: נפח סיקור → תנודתיות (`news_attention.py`)
אחרי שהטון נכשל, בדקנו את ההשערה ההגיונית באמת: לא "טון→כיוון" אלא "**קפיצת תשומת-לב→תנועה גדולה**" (מגניטודה, לא כיוון).

| מדד | ערך |
|-----|-----|
| פער גולמי \|תנועה\| (spike vs normal) | +0.40% (p=0.0195) |
| אחרי בקרת volatility-clustering | +0.22% (**55% שורד**) |
| **controlled p-value (permutation מדורג)** | **0.1484 (לא מובהק)** |
| קורלציה חלקית (מנוטרל תנודתיות מחיר) | +0.016 (מ-+0.036) |
| כיוון | 0.563 (לא אות כיוון) |

**פסק דין סופי: CLOSED.** הפער הגולמי אמיתי, אבל אחרי בקרה לתנודתיות המחיר — השארית **לא** מובהקת סטטיסטית (p=0.15). מה שנראה כ"אות חדשותי" הוא בעיקר **אשכולות תנודתיות** שהמחיר כבר מכיל. לא נכנס לנוסחה.

### זווית שלישית: דפוסים מותנים (`news_patterns.py`)
בדיקה של דפוס התנהגותי — לא "טון→כיוון" אלא תגובה מותנית (פאניקה→ריבאונד? אופוריה→נסיגה?). למידה על 70%, מבחן על 30% שלא נראו, 5 מצבים מוגדרים מראש + Bonferroni.

| מצב | כיוון | hit (holdout) | p |
|-----|-------|---------------|---|
| טון חיובי קיצוני | ירידה | 0.480 | 0.88 |
| שוק טון שלילי | עלייה | 0.543 | 0.41 |
| קפיצת תשומת-לב | עלייה | 0.563 | 0.20 |

**פסק דין: NO SIGNAL.** אף דפוס לא עבר, אף אחד לא שרד Bonferroni.

**תובנת הלימוד החשובה — אי-יציבות (non-stationarity):** מצב "טון חיובי קיצוני" (10% עליון בפיתוח) ירה על 592/758 ימי holdout — כי התפלגות הטון של GDELT זזה בזמן (החדשות נעשו חיוביות יותר). הסף שנלמד בעבר כבר לא רלוונטי. זהו **בדיוק** הכשל שהפיל את regime-weighting: דפוס שנלמד בעבר לא מכליל כי העולם זז מתחתיו.

הרצה: `python -m fade.pipeline.news_patterns`

---

## נתוני נגזרים — funding rate (v0.3)

מקור **אורתוגונלי** אמיתי (לא-מחיר), ובניגוד לחדשות יש לו בסיס מכני להקדים את המחיר: funding חיובי קיצוני = לונגים ממונפים מדי → לחץ תיקון מטה; funding שלילי = שורטים צפופים → short squeeze מעלה. השערה קונטראריאנית.

הורדנו היסטוריה מלאה מ-Binance (`download_funding.py`): 7,464 נקודות, כל 8 שעות, 2019-09→2026. (open interest נזנח — Binance נותנת רק 30 יום.)

מבחן (`funding_test.py`): funding בזמן T מול תנועת 8 השעות הבאות, עם holdout 70/30 קפדני + permutation.

| מדד | ערך |
|-----|-----|
| corr(funding, תנועה הבאה) | **−0.024** (שלילי = קונטראריאני, כיוון נכון!) |
| funding<0 → עלייה | 52.0% |
| funding>0 → עלייה | 50.5% |
| holdout hit (חוק קונטראריאני) | 0.526 |
| shuffle null | 0.521 |
| **p-value** | **0.44 (לא מובהק)** |

**פסק דין: WEAK.** בניגוד לחדשות (תגובתי, אפס), ל-funding יש **כיוון נכון ומנגנון אמיתי** — אבל חלש מדי לעבור את הרף באופק 8 שעות. שים לב שה-null הוא 0.521 (לא 0.50): לימי funding קיצוני יש ממילא הטיית עלייה, והחוק בקושי עובר אותה. לא נכנס לנוסחה, אבל המקור האורתוגונלי הכי מבטיח שנבדק עד כה.

הרצה: `python -m fade.pipeline.funding_test`

---

## מבחן המציאות — סימולציית רווח/הפסד (`pnl_sim.py`)

מדדנו דיוק, לא כסף. הבדיקה: חוקים מוקפאים על 70%, מסחר על ה-holdout (30%) עם עמלות ריאליות, מול קנה-והחזק. BTC שעה, 19,719 נרות (2023-11 → 2026-07).

| אסטרטגיה | ברוטו (0 עמלות) | עם 5 bps | עסקאות |
|----------|-----------------|----------|--------|
| קנה-והחזק | +23% | +23% | — |
| long-short | **+96%** (Sharpe 0.85) | **−99.9%** | 7,714 |
| conf-gated | +92% (Sharpe 0.94) | −99.6% | 10,305 |
| daily rebalance | −81% | −87% | 390 |

**המסקנה החשובה ביותר בפרויקט:**
1. **ה-edge אמיתי וגדול ברוטו** — long-short מכה את קנה-והחזק פי 4 (96% מול 23%) בלי עמלות. FADE באמת מזהה כיוון.
2. **אבל הוא מתחת לעלות העסקה.** האות מתהפך כמעט כל נר (7,714 עסקאות ⇒ hold-until-flip לא מוריד טורנאובר בכלל). רווח לעסקה ~3 bps מול עלות ~10 bps להיפוך. עמלות מוחקות הכל.
3. **דגימה יומית מפסידה גם ברוטו** (−81%) — זורקת את התזמון (שם ה-edge) ושורטת בשוק עולה.

**המשמעות:** FADE אינו מכונת כסף בתדירות שעה עם עמלות ריטייל. הוא **מודד חיזוי אמיתי** — ערכו כמנוע מחקר, כהטיה לפוזיציות קיימות, או לשחקן עם עמלות ~0. כדי להיות סחיר צריך edge-לעסקה > עלות: לסחור רק תנועות גדולות בביטחון גבוה (כיסוי נמוך מאוד).

הרצה: `python -m fade.pipeline.pnl_sim btc_1h.csv` · ברוטו: `--fee-bps 0 --slippage-bps 0`

הרצה: `python -m fade.pipeline.news_attention --z 1.5`

---

## מבחן holdout קפדני (BTC) — v0.3

`python -m fade.pipeline.holdout btc.csv` — dev 70% / holdout 30% בהסגר מוחלט. חוקים נכרים ונבחרים **רק** על הפיתוח, מוקפאים, ומופעלים על ה-holdout שלא נגע בבחירה.

| מדד | BTC |
|------|-----|
| Stable rules (frozen) | 16 |
| Holdout lift גולמי | +2.11% |
| **Shuffle null** | **0.5085** (דריפט מעלה!) |
| כישרון אמיתי מעל דריפט | ~1.3% |
| **p-value (300 shuffles)** | **0.057** |
| פסק דין | **WEAK — על הגבול, לא ניקה 0.05** |

**מסקנה כנה:** ה-edge **שרד כיוונית** על נתונים שלא נראו (בניגוד ל-regime-weighting שקרס), אבל הוא **marginal** — רכיבה חלקית על מגמה, וכישרון אמיתי גבולי. הרף (p≤0.05) נקבע מראש ולא הוזז.

---

## תוצאות היתכנות (עדכון v0.2)

| | BTC | ETH |
|---|---|---|
| Lift vs random | +1.06% | +1.66% |
| p-value (40 shuffles) | 0.049 | 0.049 |
| Segment consistency | 75% | 75% |
| POST_SHOCK lift (OOS) | +1.9% | +2.2% |
| Stable rules | 8 | 26 |
| Feasibility | PASS | PASS |

**מסקנה:** edge חלש אך אמיתי. היתכנות מאושרת.

---

## v0.2 — סטטוס

| # | משימה | סטטוס |
|---|--------|--------|
| P2 | Forecast CLI | ✅ **בוצע** |
| P0 | Regime-conditioned evaluation (OOS per regime) | ✅ **בוצע** |
| P1 | זיכרון נפרד לכל נכס | ✅ **בוצע** |
| P3 | Historical replay | ✅ **בוצע** |
| P4 | גרפים (matplotlib) | ✅ **בוצע** |

### v0.4 — path-aware production (batches 17–29)

| # | משימה | סטטוס |
|---|--------|--------|
| P5 | path_lean3 default + retrain BTC/ETH | ✅ |
| P6 | Conviction tiers + PRIMARY forecast | ✅ |
| P7 | ETH multi-res (5m/15m/30m/1h) | ✅ |
| P8 | outcome_tracker (live PRIMARY log) | ✅ |
| P9 | VR regime + conviction filter | ✅ |
| P10 | ml_challenger sandbox (GB/LSTM) | ✅ rules win |
| P11 | candle_patterns research | ⚠️ REJECT (dilution) |
| P12 | PnL reality v2 (costs on current engine) | ✅ **edge not tradeable @5bps** |
| P13 | Final lockbox (true unbiased OOS) | ✅ **52.8% BTC/ETH** |
| P14 | Generalization audit (Holm + decay) | ✅ decay 2025–26 |
| P15 | Decay diagnosis (H1/H2/H3) | ✅ branch C→A |
| P16 | Stock reversal benchmark | ✅ BTC still reverses |
| P17 | Regime min_hold PnL @5bps | ⚠️ holdout only |
| P18 | Regime min_hold lockbox one-shot | ✅ ETH +23%, BTC fail |
| P19 | Phase 0: lockbox BURN + pre-registration | ✅ |
| P20 | Phase 0: ETH candidate forward track | 🔄 0/100 hold-cycles v2 |
| P21 | Phase 1: sparse PRIMARY (tier≥HIGH) | ✅ default |
| P22 | Horizon sweep 4h/8h | ✅ negative — keep 1h |
| P23 | v0.3 lead-lag + funding_eth | 🔄 funding WEAK |
| P24 | Sparse PRIMARY holdout replay | ✅ BTC 58.2% @9% cov |
| P25 | Funding+streak combo | ❌ no modulation |

### v0.3 (מושהה — אחרי batch 30)

| # | משימה | סטטוס |
|---|--------|--------|
| — | Regime-weighted forecast (confidence לפי regime) | ⚠️ **נבדק ונדחה** |

**Regime-weighting:** המנגנון בנוי (`regime_confidence_scale`) אבל **כבוי כברירת מחדל** (`Config.regime_weighting_enabled=False`).

**למה נדחה — תוצאה מדעית חשובה:**
מבחן split כרונולוגי (`python -m fade.pipeline.regime_eval btc.csv`) הראה שאמינות ה-regime **לא יציבה בזמן**:

| | BTC | ETH |
|---|---|---|
| Brier delta (weighted−unweighted) | +0.012 | +0.001 |
| ECE delta | +0.050 | +0.001 |
| Verdict | NO GAIN | NO GAIN |

במחצית הראשונה של BTC דווקא NORMAL היה הכי אמין ו-POST_SHOCK קרס — **הפוך** מהאגרגציה על כל הדאטה. כלומר "POST_SHOCK מנצח" היה ארטיפקט, לא אות יציב. השקלול **מחמיר** את איכות ההסתברות → לא מופעל.

**המשמעות:** זה בדיוק מה ש-FADE נועד לתפוס — דחיית שיפור שנראה טוב באגרגציה אך לא מכליל. הכיוון (lift) לא הושפע ממילא.

---

## עקרונות

- **Core:** rule-based בלבד — walk-forward, holdout, permutation
- **Sandbox:** ML challengers (`ml_challenger*`) — השוואה בלבד, לא production
- Negative memory לפני mining; כיול per-asset
- Anti-self-deception: תוצאות שליליות מתועדות (candles, streak_big, news, ML<rules)

---

## משפט אחד

> **יעד:** רווח ממסחר על חיזוי מראש. **עכשיו:** מחקר בלבד — predictability דק (~53% lockbox), לרוב לא סחיר @5bps; ETH regime+min_hold רמז lockbox (+23%) לא מאומת forward; Rules > ML. אין עדיין מכונת מסחר.
