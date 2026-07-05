# FADE overnight run log — 2026-07-02

## מה בוצע אוטונומית

### נוסחה מנצחת (מאומתת)
- רזולוציה: שעה (Binance, 8 שנים)
- אופק: 1 bar
- אטומים: core5 (חמשת המקוריים)
- שילוב: הסכמת 15+30+60 דקות → ~54.7% hit
- גודל תנועה: סף 0.1% → skill +3.7%, כיסוי 70%

### כלים חדשים
- `forecast_tiers.py` — תחזית בשלוש רמות
- `report.py` — דוח מאוחד
- `inference.py` — חיזוי מהיר ללא walk-forward
- `magnitude_sweep.py` — סריקת סף תנועה
- `grid_search.py` — סריקת נוסחה עם הגנת overfit

### ריצת לילה (2026-07-02)
- תוקן באג ב-`backtest.py` (ייבוא `evaluate_by_regime`)
- זיכרון רוענן: `btc_1h`, `btc_30m`, `btc_15m`
- holdout רגרסיה: PASS (p=0.0033, lift +3.26%)
- תחזית נוכחית (שעה): DOWN 52.5% — רמת BALANCED פעילה; הסכמה מלאה לא פעילה (15/30 דקות ללא התאמה)

### פקודות שימושיות
```bash
# דוח מלא
python -m fade.pipeline.report btc_1h.csv

# תחזית בשלוש רמות
python -m fade.pipeline.forecast_tiers btc_1h.csv

# מבחן holdout קפדני
python -m fade.pipeline.holdout btc_1h.csv

# סריקת גודל תנועה
python -m fade.pipeline.magnitude_sweep btc_1h.csv
```

### מה לא לעשות (למדנו בכאב)
- regime-weighting — נדחה (לא יציב בזמן)
- יותר מ-5 אטומים — לא משפר
- דקה/שנייה — אין אות (רעש)
- יומי — אין אות (דריפט)
