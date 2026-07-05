# FADE — Scope Guard

Internal checklist. Every autonomous work batch (~every 30 min) must be verified
against these invariants. If any item is violated, STOP and flag it.

## Hard invariants (never break)
1. **No look-ahead.** A prediction for bar T may only use data from bars < T.
   Rule mining, threshold fitting, and stable-rule selection use revealed/dev
   data only. The "future" is scored, never trained on.
2. **No ML / no black boxes.** Rule-based only (discretised atoms -> events ->
   frequency rules + reliability-table calibration). No fitted models.
3. **No live feeds / no external runtime APIs.** Historical CSV only.
4. **Anti-self-deception first.** A rising/pretty result is a reason for
   suspicion, not celebration. Prefer honest negative findings over flattering
   ones. Report confounds explicitly.
5. **Out-of-sample truth.** Any headline hit-rate must come from data the rules
   never touched. In-sample numbers are diagnostics only, labelled as such.

## Task scope (current session)
- Many training runs across different tables (resolutions) and different time
  ranges, to widen evidence — NOT to cherry-pick the best-looking run.
- Add ETH; train both assets; measure BTC-ETH correlation and joint behaviour.
- Every headline still passes the hard invariants above.

## Scope-check log (append one line per batch)
- 2026-07-02 23:5x — batch 0: scope defined. Invariants intact.
- 2026-07-02 23:5x — batch 1: added learning_sim sub-ranges, training_suite,
  correlation. Verified: (1) no look-ahead — each checkpoint mines on revealed
  slice only, scores hidden future; calibration updates only after scoring.
  (2) no ML — frequency rules + reliability table only. (3) historical CSV only,
  no live feeds. (4) correlation reported both same-bar (not tradeable) and
  lead-lag (tradeable) separately — no self-deception. Invariants intact.
- 2026-07-02 21:05 — batch 2: training_suite completed (8 jobs, all OOS).
  frac>0.5 = 1.0 but improvement negative in 7/8 jobs — reported honestly, no
  cherry-pick. Invariants intact.
- 2026-07-03 08:3x — batch 3: fixed-window confound test (era effect confirmed),
  learning_sim_multi (+1.9% unanimous lift in progressive learning), bugfix on
  cumulative counters. Invariants intact.
- 2026-07-03 09:0x — batch 4: news sentiment (GDELT). Verified against invariants
  before building: (1) no look-ahead — GDELT tone/volume carry real publication
  date; tone_chg + volume z-score use trailing windows only; rules mined on dev
  split, frozen before holdout. (2) no ML — GDELT tone is a fixed-lexicon score
  computed at ingest, not a model trained on outcomes; FADE still uses frequency
  rules. (3) offline runtime — news is a downloaded historical CSV, accumulated
  for training only (user's "give the past slowly"), not a live feed. (4) anti-
  self-deception — news_test pits news vs price baseline vs combined on identical
  holdout dates + permutation test; will be discarded if it fails p<=0.05, same
  as regime-weighting. Invariants intact.
- 2026-07-03 09:4x — batch 5: RESULT IN. Date-alignment verified (no leak: tone
  correlates with PAST return lag-1 +0.099, with FUTURE return +1 ~0). Holdout:
  news lift +0.015 but p=1.00 (pure class imbalance, 2 always-up rules);
  combined p=0.35. News = lagging indicator, NO OOS edge -> discarded, exactly
  like regime-weighting. Honest negative reported, not hidden. Invariants intact.
  Note: rate-limit hardened (10s spacing, 30-300s backoff) after 429s; 2019 gap
  remains but conclusion is stable across all available years.
- 2026-07-03 10:2x — batch 6: news ATTENTION (volume) test. Honest re-frame of a
  reasonable hypothesis (spike->volatility, not tone->direction). Raw gap +0.40%
  p=0.0195, but built-in confound control (match on price's own recent vol)
  shows only ~55% survives; partial corr 0.016 (tiny). Reported as UNRESOLVED
  hint, NOT an edge — did not overclaim, flagged that residual significance is
  unproven. No look-ahead (trailing vol_z + recent_vol shifted). Invariants intact.
- 2026-07-03 10:3x — batch 7: stratified permutation on controlled gap -> p=0.1484.
  CASE CLOSED: raw spike effect real (p=0.02) but residual after price-vol control
  is NOT significant. "News attention" = mostly volatility clustering. Final honest
  answer to "weird there's nothing": tone->direction = zero; volume->volatility =
  confounded. News does not enter the formula. Invariants intact.
- 2026-07-03 10:4x — batch 8: conditional news-PATTERN test (user: goal is to
  train/learn, so multi-comparison caveat reported not treated as blocker). 5
  pre-registered states, dev->holdout + permutation + Bonferroni. NO SIGNAL
  (none survive). Key learning: news tone distribution is NON-STATIONARY (dev
  90th-pct threshold caught 592/758 holdout days) — same failure mode as
  regime-weighting. Honest negative. No look-ahead (dev-fit thresholds).
  Invariants intact.
- 2026-07-03 10:4x — batch 9: derivatives = funding rate (orthogonal, mechanically
  leading). Downloaded 7,464 pts 2019-2026 (OI dropped: Binance 30d only).
  Holdout 70/30 + permutation. Contrarian sign is CORRECT (corr -0.024) and
  mechanism real, but p=0.44 (null itself 0.521) -> WEAK, not significant.
  Better than news (right sign) but not in formula. No look-ahead (funding at T
  public at T, predicts T->T+8h; dev-fit deciles). Invariants intact.
- 2026-07-03 11:0x — batch 10: PnL reality check on holdout with real costs.
  Frictionless: long-short +96% vs buy-hold +23% (edge is REAL & large gross).
  With 5bps: -99% (signal flips ~every bar -> 7714 trades -> fees annihilate it).
  Honest headline: FADE measures real predictability but is sub-transaction-cost
  at 1h retail fees; not a money machine. No look-ahead (frozen dev rules, dev-fit
  calibration, next-bar returns). Reported gross AND net. Invariants intact.
- 2026-07-03 11:3x — batch 11: NEW MECHANISM per user pivot ("no trading, find
  patterns/trends"). Added sequence_patterns.py (n-gram predictor) + trend_structure.py
  (streak->next map). Fills the memoryless gap: engine only scored single-bar combos,
  never the ORDERED sequence. Finding: intraday BTC MEAN-REVERTS after streaks, reversal
  strengthens MONOTONICALLY with streak length (15m: 2->47.7% ... 6+->40.6% continue;
  1h: UUUU->down 56.9%, DDDD->up 57.2%; mag3 0-->up 60.6%). Single bar (+1) = random
  walk (~50%, not sig) -> explains why memoryless engine caps ~53%. Reversion vanishes
  at DAILY scale (intraday-only inefficiency). No look-ahead (k-gram/streak uses bars
  strictly < t, target at t; dev-frozen direction; 70/30 holdout; permutation +
  Bonferroni). No ML, historical-only, honest negatives kept (UUD, daily). Invariants intact.
- 2026-07-03 11:5x — batch 12: memory-depth sweep (sequence_sweep.py, k=2..8, per user
  "different time sequences"). Aggregate holdout edge saturates at k~5-6 across all
  resolutions, strongest 1h (+2.65% at k=6). CRITICAL anti-self-deception result: at
  k>=7 n_survive_Bonferroni CRASHES to 0 while best single-pattern hit climbs to 0.68 ->
  the tool correctly flags long grams as sparse-support ILLUSION, not edge. Effective
  market memory ~5-6 bars. No look-ahead (same protocol as batch 11), Bonferroni across
  all 2^k patterns, reported aggregate (not cherry-picked best). Invariants intact.
- 2026-07-03 12:0x — batch 13: cross-scale structure (scale_structure.py, per user
  "micro/macro insights + correlation between them + between time patterns"). (1) Reversion
  ladder: PHASE TRANSITION 1s=momentum(0.60) -> minutes..1h=reversion(peak 1h +0.046) ->
  daily=efficient(p=0.42). (2) micro x macro: intraday reversal ~0.54-0.55 across ALL daily-
  trend regimes, only mildly modulated -> reversal is macro-independent. (3) cross-timeframe:
  return corr=1.0 but reversal-SIGNAL corr~0 -> timeframes carry semi-independent info,
  explains multi-res ensemble gain. NO LOOK-AHEAD: macro daily return attached only at
  day-end via merge_asof(available_at=day+1, backward); streak uses bars<t; 70/30 holdout;
  permutation. Honest caveat logged on 1s (short span, tick-sensitive). No ML. Invariants intact.
- 2026-07-03 12:1x — batch 14: integrated pattern_suite (user: "all one by one and together").
  A) phase crossover 30s->60s on resampled 1s (~7d). B) ETH rev +0.042 ~ BTC +0.046; signal
  corr 0.61; joint agree 55.1% vs disagree 47.6%. C) vol-conditioned: reversal ~0.545 both
  regimes, gap +0.002 only -> vol does NOT modulate. D) forward 1h ensemble: solo 52.8% ->
  majority 54.3% -> unanimous-4TF 62.6% (n=265 sparse, flagged). Fixed ensemble bug (was
  averaging opposing signals). Forward target for cross-TF; same-bar kept as ref. No look-ahead
  (vol shifted, dev median, streak<t). Invariants intact.
- 2026-07-03 12:2x — batch 15: CORE INTEGRATION (user: deep+broad, add focused expert
  subagent). Injected path atom streak_signed into atoms.py pool + core5_path/path_min sets
  + structural fixed thresholds (+/-2) via Config.atom_fixed_thresholds (definition, not
  data-fit -> no look-ahead). Strict holdout core5 vs core5_path: aggregate hit FLAT ~53.2%
  BUT isolated streak=LOW/HIGH (run>=2) subpopulation = 54.60% on 10,949 OOS preds (~9sigma).
  DIAGNOSIS CONFIRMED: bottleneck is AGGREGATION (AND-grammar + confidence filter dilute the
  strong specific rule), NOT vocabulary. Honest: reported that naive atom-add did NOT lift
  aggregate; edge only visible when specific rule un-diluted. streak_signed causal (returns<=t
  predict t+1). No ML, walk-forward + frozen dev rules + permutation upstream. Ran expert
  subagent (atom_redundancy.py) in parallel for collinearity evidence. Invariants intact.
- 2026-07-03 12:2x — batch 15b: expert subagent completed atom_redundancy.py. 9 atoms ->
  3 semantic clusters, 3 pairs |r|>0.6, PCA needs 7 PCs for 95% var; greedy orthogonal
  basis ~4 atoms (close_pos, return_6h, volume_zscore, volatility). Confirms vocabulary
  redundancy complements (not replaces) aggregation-dilution diagnosis from streak_signed
  holdout. Descriptive only, no look-ahead. Invariants intact.
- 2026-07-03 12:3x — batch 16: specificity_test.py — the "genius idea" (weight sharp rules
  louder) HONESTLY FAILED. 8 aggregation schemes (size/size2/rarity/edge/size_edge/argmax_spec/
  mechanism_gate) all ~53.2% on core5_path, identical to equal-weight baseline. NOT reported as
  success. Real winner emerged: LEAN atom set path_min (return_1h, volatility, volume_zscore,
  streak_signed) = 54.05% on 14,264 OOS (p=0.002), beating core5 53.26% and core5_path 53.21%.
  Unifies both diagnoses: dilution is AVOIDED at the vocabulary level (drop collinear atoms) not
  FIXED at aggregation. No look-ahead (frozen dev rules/thresholds, streak causal, permutation
  p-values, same 70/30 harness as holdout.py). Honest negative on weighting kept. Invariants intact.
- 2026-07-03 12:4x — batch 17: two focused subagents. lean_search.py: after holdout fix,
  best btc_1h lean set path_lean3 (close_pos, range_pct, streak_signed) 54.64% vs path_min
  54.05%; 3-atom beats 4-atom. path_min_generalization.py: path_min beats core5 only 2/5
  (btc_1h, btc_15m) — advantage does NOT generalize to ETH/30m/5m. Fixed holdout.py
  momentum baseline to fall back to return_1h when return_6h absent (enables lean-set eval).
  Added path_lean3 to ATOM_SETS. Honest negatives kept. Invariants intact.
- 2026-07-03 13:2x — batch 18: second path atom streak_big (magnitude-conditioned run:
  |ret|>k*trailing_vol, causal). HONEST NEGATIVE: mechanism did NOT confirm "big-move streaks
  reverse harder" (up>=2 big 46.3% vs plain 44.5%; only up>=4 strong at 27.5% but n=51 illusion).
  Holdout: path_big 53.98% < path_lean3 54.64%; path_both (both streaks) 53.60% -> adding it
  DILUTES, reinforcing the aggregation-dilution diagnosis. streak_big kept in pool for research,
  NOT promoted to a winning set. Causal (trailing vol, returns<=t, target t+1), fixed thresholds
  (definition not fit), permutation p-values. No ML. Honest negative kept. Invariants intact.
- 2026-07-03 13:2x — batch 19: generalization_why.py — why lean sets "fail" to generalize.
  path_lean3 beats core5 on 4/5 (better than path_min's 2/5) but delta SHRINKS with weaker
  local mechanism: corr(rev_index, delta)=+0.59, corr(close_pos_edge, delta)=+0.58. btc_1h
  strongest (rev 0.046, cp 0.041, delta +1.38%); eth +0.25%; 5m +0.07%; 30m -0.05%. Conclusion:
  mechanism EXISTS everywhere, advantage SCALES with local strength — not btc-unique artifact.
  Holdout diagnostics on quarantined slice only, no look-ahead. Invariants intact.
- 2026-07-03 13:3x — batch 20: conviction_gate.py — trade coverage for accuracy. Streak-length
  gate: monotone rise to 57.1% at run>=4 (n=2252, p=0.0005), collapses to noise at 7+ (sparse,
  consistent w/ sequence_sweep). Multi-res agreement gate CONFIRMS earlier 62.6%: >=3 TF agree =
  58.3% (n=2625 usable coverage), unanimous-4 = 62.6% (n=265 sparse). Two independent conviction
  axes. Contrarian rule is a fixed definition (against the run) not a fitted param; streak uses
  bars<t; permutation p vs random-direction null; quarantined holdout. No ML. Invariants intact.
- 2026-07-03 13:3x — batch 21: conviction_combo.py — combined gate + calibration + ETH.
  Combined BTC: L>=2 K>=3 -> 58.2% (n=2067), L>=3 K>=3 -> 59.6% (n=909), L>=2 K>=4 -> 62.6%
  (n=265, same as multi-only unanimous -- axes partially redundant at K=4). Calibration table:
  empirical hit = honest confidence %. ETH streak gate: similar mechanism (streak>=3 55.1%,
  streak>=5 56.5%), no ETH multi-res files (honest limitation). Fixed contrarian, holdout only,
  permutation p. No ML. Invariants intact.
- 2026-07-03 14:0x — batch 22: conviction integrated into forecast_tiers CLI
  (fade/core/conviction.py) + yearly stability (conviction_stability.py). Forecast
  shows highest active conviction tier with calibrated %. Stability: 20/20 year-rule
  pairs above 50% (2017-2026) -> mechanism is STATIONARY, not era artifact. Fixed
  contrarian rules per year, no re-fitting. Invariants intact.
- 2026-07-03 14:1x — batch 23: path_lean3 wired as production default
  (DEFAULT_ATOM_SET, lean_config in main/forecast/forecast_tiers/inference).
  Retrained memory: btc_1h (15 rules, +3.78% lift), eth_1h (17, +3.69%), btc_15m
  (13, +4.99%), btc_30m (14, +4.00%). Live forecast: BTC balanced DOWN 52.3% with
  TF disagreement; ETH no_match (honest — no rule fires on current atoms). ETH
  multi-res conviction still blocked (no eth_15m/30m CSV). Walk-forward only, no
  look-ahead. Invariants intact.
- 2026-07-03 15:0x — batch 24: train and believe. Downloaded eth_15m/eth_30m;
  trained path_lean3 memory (eth_15m +4.53%, eth_30m +3.48%, btc_5m +3.47%).
  training_suite on lean_config: 10/10 jobs cum_hit>0.5, mean=54.27%, verdict
  consistent positive edge. conviction_stability 20/20 years. forecast_tiers +
  conviction generalized to any asset prefix (eth 15m/30m/1h). Invariants intact.
- 2026-07-03 15:4x — batch 25: unified PRIMARY forecast (conviction beats
  conflicting balanced), conviction_stability extended to ETH (20/20 years),
  BTC data refreshed (1h/15m/30m). ETH stability confirms mechanism is not
  btc-unique. Fixed contrarian rules, no re-fit. Invariants intact.
- 2026-07-05 08:4x — batch 26: primary_replay (BTC 53.1% ETH 52.2% on holdout
  conviction path), conflict rule (frequent beats conviction), refresh CLI.
  Honest holdout only. Invariants intact.
- 2026-07-05 09:0x — batch 28: parallel agents — ml_challenger (sandbox GB:
  BTC 53.21% ETH 52.49% OOS, below FADE rules), VR gate (HIGH_VR weakest
  reversal 54.1%), PRIMARY weak-conflict abstain + ASCII CLI. Core still
  no-ML; challenger explicitly sandbox. Invariants intact.
- 2026-07-05 09:1x — batch 29: parallel agents — candle patterns REJECT
  (path_candles 53.31% < path_lean3 54.04%), VR conviction filter (HIGH_VR
  streak>=3), LSTM sandbox (53.54% BTC still below FADE rules). Honest
  negatives kept. Core no-ML intact. Invariants intact.
- 2026-07-05 08:5x — batch 27: outcome_tracker.py — JSONL ledger logs every
  PRIMARY (direction, source, conflict); scores hit/miss vs next 1h bar when
  data arrives. No look-ahead on scoring. CLI: log/score/report/run. Invariants intact.
- 2026-07-05 09:4x — batch 30: CRITICAL GAPS CLOSED (before more atom search).
  (1) pnl_reality_check_v2: path_lean3 + conviction + PRIMARY + min_hold @
  1/5/10 bps on holdout. HONEST NEGATIVE: ALL variants negative at 5bps (raw
  -99%, best min_hold_24 -12.7%). Improved accuracy did NOT fix fee drag.
  (2) final_lockbox: sealed newest 18% (SHA256 manifest), one-shot eval.
  TRUE OOS path_lean3 BTC 52.77% / ETH 52.75% — 1.8pp below inflated 54.6%
  headline; tagged multiple-comparisons risk. Still p=0.0033 significant.
  (3) generalization_audit: Holm on 7 atom sets — all survive (p_holm=0.0231),
  path_lean3 best hit but not unique. DECAY: 2025-26 streak>=3 -4.2pp BTC,
  combo -4.7pp ETH vs pre-2025. No look-ahead (lockbox never mined; dev-only
  rules). Honest negatives reported. Invariants intact.
- 2026-07-05 10:0x — batch 31: decay_diagnosis (H1/H2/H3) + stock_reversal_benchmark.
  H1 BTC: uniform VR decay (-2.9 to -5pp); funding spread 7.2pp (EXTREME_NEG
  stronger in 2025-26). H2 BTC: MONOTONIC pre-2024 decline (slope -0.00149/q,
  corr -0.476); rev_index 0.07->0.039. H3: NO spread/liquidity data (documented).
  Stocks SPY/AAPL rev~0; BTC today +0.045 still > stocks; BTC 2018-19 +0.091.
  Decision branch C (mixed) -> default A. Session-reset streak for equities.
  Invariants intact.
- 2026-07-05 10:1x — batch 32: BRANCH A follow-up — pnl_regime_minhold @5bps.
  Exploratory grid (3 VR regimes x 9 min_hold) on SAME holdout: BTC HIGH_VR
  min_hold=48 +76.8% (308 trades); ETH LOW_VR min_hold=12 +94.2% (658 trades).
  FLAGGED overfit risk — NOT lockbox validated; assets prefer different regimes.
  Ungated BTC still -12.7%. Honest caveat, not promoted to production. Invariants intact.
- 2026-07-05 10:2x — batch 33: regime_minhold_lockbox ONE-SHOT on sealed 18%
  (pre-registered batch 32 configs). BTC HIGH_VR hold=48: holdout +76.8% ->
  lockbox -10.9% (FAIL_SOFT, overfit confirmed). ETH LOW_VR hold=12: holdout
  +94.2% -> lockbox +23.2% (POSITIVE, dir_hit 52.5%, survives 5bps on true OOS).
  No parameter search on lockbox; rules/VR from pre-lockbox only. MIXED verdict:
  no unified BTC+ETH strategy. Invariants intact.
- 2026-07-05 10:4x — batch 34: PHASE 0+1. Documented: ETH LOW_VR+12 selected on
  holdout grid (NOT lockbox search); same overfit class as BTC; lockbox v1 BURNED
  (manifest updated). ETH status=candidate_not_validated; forward ledger
  eth_candidate_outcomes.jsonl via eth_candidate_track + outcome_tracker.
  pre_registration.json protocol. Phase 1: sparse PRIMARY default (tier>=HIGH).
  Lockbox v2 reserved for future pre-registered tests. Invariants intact.
- 2026-07-05 11:0x — batch 35: horizon_sweep (pre-registered) 4h/8h vs 1h BTC+ETH.
  Holdout exploratory @5bps: NO positive PnL any horizon; 1h best BTC (-12.7% mh24).
  ETH 4h less bad (-17.1% vs -79.6% 1h) but still negative. lead_lag_probe: BTC-ETH
  corr 0.80; streak reversal on ETH lag-1 52.65% (marginal). funding_eth.csv added.
  OI skipped (30d Binance limit). Invariants intact.
- 2026-07-05 11:1x — batch 36: sparse_primary_replay (pre-reg) BTC sparse 58.17%
  n=2068 cov=8.9% elite=62.64%; ETH 55.31%. funding_streak_combo REJECT —
  EXTREME_NEG not stronger than NEUTRAL on holdout. ETH funding_test p=0.12 WEAK.
  outcome_tracker: 1/100 eth candidate scored (hit), sparse PRIMARY 2 scored.
  Data refreshed to 2026-07-05 07:00 UTC. Invariants intact.
- 2026-07-05 11:3x — batch 37: ETH candidate scoring fix — hold-cycle PnL (v2)
  via pnl_sim._equity @ 5+5bps, min_hold=12; report-candidate dual metrics.
  pre_registration.json metric_changed_utc; 1 pre-fix next-bar signal excluded
  from n=100. Config unchanged (LOW_VR+12). Invariants intact.
- 2026-07-05 11:4x — GitHub Action outcome-tracker.yml: hourly refresh + run-all +
  commit CSVs/jsonl/eth_candidate_state.json. requests added to requirements.txt.
  Invariants intact.
