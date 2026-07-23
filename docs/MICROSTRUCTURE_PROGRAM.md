# תוכנית מיקrostructure — שניות כבסיס יחיד

עודכן: **2026-07-23**  
סטטוס: **ACTIVE** (POC 1h הושלם; הרחבת ימים exploratory)

**כוכב צפון (כמו כל FADE):** חיזוי מראש → מסחר → רווח נטו.  
מיקרו הוא שכבת מחקר — אם אין אופק PnL נטו אחרי עלויות, אין קידום.

מקורות machine-readable:
- `fade/output/microstructure_board.json` — מנהל, צוות, סטטוס
- `fade/output/microstructure_test_plan.json` — רישום בדיקות לפני ביצוע

---

## עקרון מנחה

**שנייה אחת = שכבת האמת.**  
דקות, שעות וימים מסתכמים ממנה (rollup). לא מורידים ולא קוראים בנפרד קבצי 1h/5m אם אפשר לגזור מהשניות.

**מטרה:** להשוות **לחץ קנייה מול מכירה** מול **תנועת מחיר** במרווחי זמן שונים — לא לנחש נר הבא.

**מגבלה:** אין מחשב קוואנטום → קריאה בחתיכות, שמירת derivatives בלבד, לא raw ענק.

---

## מה יש / מה חסר

| נכס | יש | חסר לקנייה/מכירה |
|-----|-----|------------------|
| `btc_1s.csv` | ~605k שורות, ~7 ימים, OHLCV+volume | אין פיצול buy/sell |
| aggTrades (Binance) | לא הורד | price, qty, is_buyer_maker — **מקור ק/מ** |

**שלב 0 חובה:** לפני בדיקות ק/מ — להוריד aggTrades לאותו חלון 7 ימים, או להוסיף taker-buy מ-klines (גרסה חלשה יותר).

---

## ארכitektura קריאה יעילה

1. **Parquet לפי יום** — `data/micro/btc/YYYY-MM-DD.parquet` (שניות או aggTrades מקודד)
2. **קריאה ב-chunks** — יום אחד בזמן (~86,400 שורות), לא CSV שלם ב-RAM
3. **פירamide rollup** — חישוב חד-פעמי: 1s → 5s → 15s → 60s → 300s → 3600s → 86400s; שמירה לקובץ features
4. **DuckDB** (אופציונלי) — שאילתות SQL על parquet בלי טעינה מלאה
5. **אירועים בלבד** — אחרי שלב סינון: שמירת חלונות ±N שניות סביב קיצוניות, לא כל הימים

---

## שלבי בדיקה (לפני ביצוע — לא לדלג)

### P0 — אישור (עכשיו)
מנהל + צוות + תוכנית ב-JSON. **אין קוד עד אישור.**

### P1 — ביקורת נתונים (2–4 שעות)
- גודל aggTrades ל-7 ימים
- דגימת 1 שעה: כמה trades, כמה MB
- החלטה: aggTrades מלא vs taker-buy מ-klines

### P2 — צינור chunk (4–8 שעות)
- `micro_chunk_reader`: יום → features 1s
- benchmark RAM/זמן על `btc_1s.csv`
- פלט: `micro_pipeline_benchmark.json`

### P3 — קידוד ק/מ לשנייה (8–16 שעות)
- aggTrades → buy_vol, sell_vol, delta, trade_count per second
- join ל-OHLC 1s
- **לא** למידה — רק טבלת features

### P4 — rollup רב-קנה (4 שעות)
- מאותו קובץ 1s: כל המרווחים
- עמודות: delta, return, abs_return, imbalance_ratio

### P5 — סימון קיצוניות (4–8 שעות)
- קיצון delta (p99/p01)
- קיצון תנועה (|return| p99)
- **תגובה:** return בחלון t+1s…t+60s, t+300s, t+3600s
- שאלה: האם קיצון ק/מ **קודם** לקיצון מחיר?

### P6 — השוואה מרווחים (8 שעות)
- cross-correlation: delta@scale X vs return@scale Y (רק scales ≤ 3600s בהתחלה)
- lead-lag מוגבל: ±5 buckets — לא חיפוש אינסופי
- holdout זמני: 70/30 כרונולוגי על **7 ימים בלבד** = exploratory בלבד

### P7 — דוח מועצה
- JSON + המלצה: האם יש מבנה ששווה חלון הורדה ארוך יותר
- **לא** קידום ל-production

---

## קריטריוני הצלחה (מוקדמים)

| בדיקה | מטרה | כישלון |
|--------|------|--------|
| T1 chunk reader | 7 ימים < 8GB RAM, < 10 דק | crash / OOM |
| T2 buy/sell encode | >95% שניות עם join | >10% חורים |
| T3 extreme labels | ≥100 אירועי p99 | <20 |
| T4 lead-lag | corr מוגדר @ scale אחד | כל null |
| T5 stability | אותו סימן ב-half1 vs half2 | סתירה > 0.15 |

---

## מה אסור (FP)

- FP01: 50 features על אותם שניות
- FP03: corr בלי עמלות/slippage
- FP09: lead-lag על כל הפרמטרים (grid)
- look-ahead: delta ב-t משתמש ב-trades אחרי close של t

---

## שומר קצב API (חובה לפני aggTrades)

**תפקיד:** `api_guardian` — veto על הורדות; לא חורגים מ-rate limit.

**קיים בקוד:** backoff ב-`download_history.py`, `download_derivatives.py`, `download_marketcap.py`, `download_news.py`.

**חסר:** aggTrades downloader, מונה גלובלי, INCIDENT על 418.

**כלל:** T0 — audit שעה אחת aggTrades **לפני** bulk 7 ימים. 418 = עצירה מיידית.

---

## יחס ל-FADE core

- **לא** נוגע ב-path_lean3
- **לא** atoms חדשים
- sandbox `research_microstructure_v1`
- forward / 1h נשארים נפרדים עד VALIDATED
