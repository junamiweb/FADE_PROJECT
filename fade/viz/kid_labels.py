"""Kid-friendly Hebrew labels for charts and tier rankings."""

from __future__ import annotations

from fade.core.conviction import TIER_DEFS

ASSET_HE: dict[str, str] = {
    "btc": "ביטקוין",
    "eth": "איתריום",
    "spy": "מדד S&P 500",
    "aapl": "אפל",
}

DIRECTION_HE: dict[str, str] = {
    "UP": "למעלה ↑",
    "DOWN": "למטה ↓",
    "FLAT": "ללא כיוון",
}

CHANGE_TYPE_HE: dict[str, str] = {
    "PRICE_UP": "עלייה במחיר",
    "PRICE_DOWN": "ירידה במחיר",
    "VOLUME_SPIKE": "עלייה בנפח",
    "COMBINED_UP": "עלייה חזקה",
    "COMBINED_DOWN": "ירידה חזקה",
}

# (tier_id, kid_label, stars, short_explanation)
TIER_RANK_HE: list[tuple[str, str, int, str]] = [
    ("elite", "כוכב זהב", 5, "הכי חזק — כולם מסכימים"),
    ("strong", "כוכב כסף", 4, "חזק מאוד"),
    ("high", "כוכב נחושת", 3, "אות טוב"),
    ("multi3", "שלושה חברים", 3, "שלושה זמנים מסכימים"),
    ("streak4", "רצף ארוך", 2, "המחיר זז הרבה פעמים ברצף"),
    ("streak3", "רצף בינוני", 2, "רצף של כמה פעמים"),
    ("streak2", "רמז קטן", 1, "רמז עדין"),
]

TIER_HIT_PCT: dict[str, float] = {tid: hit * 100 for tid, _, _, _, hit, _ in TIER_DEFS}

KID_FONT = "Noto Sans Hebrew"


def he(text: str) -> str:
    """Render Hebrew for matplotlib (RTL-aware)."""
    try:
        from bidi.algorithm import get_display
        return get_display(text)
    except ImportError:
        return text


def apply_kid_font() -> None:
    """Configure matplotlib for readable Hebrew kid charts with Latin/digit fallback."""
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Noto Sans Hebrew", "DejaVu Sans", "sans-serif"]
    plt.rcParams["font.size"] = 13
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["legend.fontsize"] = 11


def asset_name_he(asset: str) -> str:
    stem = asset.lower().split("_")[0]
    return ASSET_HE.get(stem, asset.upper())


def stars(n: int) -> str:
    return "★" * n + "☆" * max(0, 5 - n)


def confidence_stars(pct: float | None) -> str:
    if pct is None:
        return ""
    if pct >= 60:
        return "★★★"
    if pct >= 55:
        return "★★☆"
    if pct >= 52:
        return "★☆☆"
    return "☆☆☆"


def confidence_label(pct: float | None) -> str:
    if pct is None:
        return "לא יודעים"
    if pct >= 60:
        return "סיכוי גבוה מאוד"
    if pct >= 55:
        return "סיכוי טוב"
    if pct >= 52:
        return "סיכוי בינוני"
    return "לא בטוח"


def rank_label(position: int) -> str:
    return f"מקום {position}"


def lift_label(lift: float) -> str:
    if lift >= 0.05:
        return "מעולה!"
    if lift >= 0.02:
        return "טוב"
    if lift >= 0:
        return "בסדר"
    return "חלש"


def tier_bar_label(tier_id: str, kid_label: str, star_count: int) -> str:
    return f"{stars(star_count)}  {kid_label}"
