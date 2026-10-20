import re
from typing import Dict, Any, Optional, Tuple
from src.config import CLASSES

# Class Synonyms and Keyword Mappings
ACTIVITY_SYNONYMS = {
    "Fighting": [
        "fighting", "fight", "fights", "brawl", "brawling", "assault", "assaults",
        "physical altercation", "punching", "beating", "scuffle", "violence", "violent"
    ],
    "Robbery": [
        "robbery", "robber", "robbers", "robbing", "mugging", "mugger", "theft", "armed robbery",
        "holdup", "stickup", "hold-up", "stealing from person"
    ],
    "Shoplifting": [
        "shoplifting", "shoplift", "shoplifter", "shoplifters", "store theft", "concealment",
        "concealing items", "pilfering", "retail theft"
    ],
    "Burglary": [
        "burglary", "burglar", "burglars", "break-in", "break in", "breaking and entering",
        "forced entry", "intruder", "trespassing", "broken into"
    ],
    "RoadAccidents": [
        "road accident", "road accidents", "car accident", "car crash", "traffic accident",
        "collision", "vehicular accident", "vehicle crash", "pedestrian hit", "crash", "accidents", "accident"
    ],
    "Vandalism": [
        "vandalism", "vandal", "vandals", "vandalizing", "defacement", "property damage",
        "damaging property", "graffiti", "smashing property", "destruction"
    ],
    "Normal": [
        "normal", "regular", "peaceful", "walking", "no incident", "ordinary"
    ]
}

# Prompt Injection Patterns to neutralize
INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+|previous\s+|prior\s+)*instructions",
    r"you\s+are\s+now",
    r"system\s*prompt",
    r"drop\s+table",
    r"delete\s+from",
    r"select\s+\*\s+from",
    r"<\s*script[^>]*>.*?<\s*/\s*script\s*>",
    r"<\s*script[^>]*>",
    r"javascript:",
    r"exec\s*\(",
    r"eval\s*\("
]

def sanitize_query(query: str) -> Tuple[str, bool]:
    """
    Strips dangerous characters and neutralizes prompt injection attempts.
    Returns (cleaned_query, was_injection_detected).
    """
    if not query:
        return "", False

    was_injection = False
    cleaned = query.strip()

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            was_injection = True
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    # Remove script tags or control characters
    cleaned = re.sub(r"[\x00-\x1F\x7F]", "", cleaned)
    return cleaned.strip(), was_injection

def extract_activity(text: str) -> Optional[str]:
    """
    Extracts canonical activity class from text using synonym mapping.
    Matches multi-word phrases first to prioritize specific terms like 'road accident'.
    """
    lower = text.lower()

    # Prioritize multi-word matches first
    for canonical, synonyms in ACTIVITY_SYNONYMS.items():
        multi_word = [s for s in synonyms if " " in s]
        for syn in multi_word:
            pattern = r"\b" + re.escape(syn) + r"\b"
            if re.search(pattern, lower):
                return canonical

    # Match single words
    for canonical, synonyms in ACTIVITY_SYNONYMS.items():
        single_word = [s for s in synonyms if " " not in s]
        for syn in single_word:
            pattern = r"\b" + re.escape(syn) + r"\b"
            if re.search(pattern, lower):
                return canonical

    return None

def extract_confidence(text: str) -> Optional[float]:
    """
    Extracts minimum confidence threshold from text.
    Handles 'confidence above 70%', 'confidence > 0.8', 'over 60% confidence', etc.
    """
    lower = text.lower()

    # Pattern: 'confidence above/greater than/over/at least/> 70%' or '70% confidence'
    pct_match = re.search(r"(?:confidence\s*(?:above|over|greater than|at least|>=|>)?\s*|above\s*|over\s*)(\d{1,3})\s*%", lower)
    if pct_match:
        val = float(pct_match.group(1)) / 100.0
        return min(1.0, max(0.0, val))

    # Pattern: '70% confidence'
    pct_match2 = re.search(r"(\d{1,3})\s*%\s*confidence", lower)
    if pct_match2:
        val = float(pct_match2.group(1)) / 100.0
        return min(1.0, max(0.0, val))

    # Pattern: 'confidence > 0.7' or 'confidence above 0.7'
    dec_match = re.search(r"confidence\s*(?:above|over|greater than|at least|>=|>)?\s*(0\.\d{1,3})", lower)
    if dec_match:
        return float(dec_match.group(1))

    # Qualitative terms
    if "high confidence" in lower or "highly confident" in lower:
        return 0.75
    if "medium confidence" in lower:
        return 0.50

    return None

def parse_hour_str(hour_val: str, am_pm: Optional[str]) -> int:
    """
    Converts 12-hour or 24-hour hour string to integer hour (0-23).
    """
    h = int(hour_val)
    if am_pm:
        am_pm = am_pm.lower()
        if am_pm == "pm" and h < 12:
            h += 12
        elif am_pm == "am" and h == 12:
            h = 0
    return h

def extract_time_filters(text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts wall-clock or video-relative time constraints.
    Examples:
    - 'after 8 PM', 'after 20:00'
    - 'before 6 PM', 'before 18:00'
    - 'between 6 PM and 10 PM', 'from 14:00 to 18:00'
    - 'after 10 seconds', 'first 5 seconds'
    """
    lower = text.lower()

    # 1. Video-relative seconds
    # 'after 10 seconds' / 'after 10s'
    rel_after = re.search(r"after\s+(\d+(?:\.\d+)?)\s*(?:seconds|sec|s\b)", lower)
    if rel_after:
        return {
            "type": "relative_after",
            "start_sec": float(rel_after.group(1)),
            "is_wall_clock": False,
            "raw": rel_after.group(0)
        }

    # 'first 5 seconds'
    rel_first = re.search(r"(?:first|in the first)\s+(\d+(?:\.\d+)?)\s*(?:seconds|sec|s\b)", lower)
    if rel_first:
        return {
            "type": "relative_between",
            "start_sec": 0.0,
            "end_sec": float(rel_first.group(1)),
            "is_wall_clock": False,
            "raw": rel_first.group(0)
        }

    # 2. Wall-clock: 'between 6 PM and 10 PM' / 'between 18:00 and 22:00'
    between_match = re.search(
        r"between\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:and|to|-)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
        lower
    )
    if between_match:
        h1 = parse_hour_str(between_match.group(1), between_match.group(3) or between_match.group(6))
        m1 = int(between_match.group(2) or 0)
        h2 = parse_hour_str(between_match.group(4), between_match.group(6))
        m2 = int(between_match.group(5) or 0)

        return {
            "type": "between",
            "start_hour": h1,
            "start_minute": m1,
            "end_hour": h2,
            "end_minute": m2,
            "is_wall_clock": True,
            "raw": between_match.group(0)
        }

    # 3. Wall-clock: 'after 8 PM' / 'after 20:00' / 'past 8:30 PM'
    after_match = re.search(r"(?:after|past|since|later than)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", lower)
    if after_match:
        h = parse_hour_str(after_match.group(1), after_match.group(3))
        m = int(after_match.group(2) or 0)
        return {
            "type": "after",
            "hour": h,
            "minute": m,
            "is_wall_clock": True,
            "raw": after_match.group(0)
        }

    # 4. Wall-clock: 'before 6 PM' / 'before 18:00' / 'prior to 9 AM'
    before_match = re.search(r"(?:before|prior to|earlier than|until)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", lower)
    if before_match:
        h = parse_hour_str(before_match.group(1), before_match.group(3))
        m = int(before_match.group(2) or 0)
        return {
            "type": "before",
            "hour": h,
            "minute": m,
            "is_wall_clock": True,
            "raw": before_match.group(0)
        }

    return None

def extract_video_id(text: str) -> Optional[str]:
    """
    Extracts UUID formatted video ID from query text.
    """
    uuid_match = re.search(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", text)
    if uuid_match:
        return uuid_match.group(0).lower()
    return None

def parse_natural_query(query: str, explicit_filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Parses a natural-language query into structured retrieval parameters.
    Merges with explicit filters if provided.
    """
    explicit = explicit_filters or {}
    cleaned_query, was_injection = sanitize_query(query)

    # 1. Activity Category
    activity = explicit.get("activity") or extract_activity(cleaned_query)

    # 2. Confidence Threshold
    min_confidence = explicit.get("min_confidence")
    if min_confidence is None:
        min_confidence = extract_confidence(cleaned_query)

    # 3. Video ID
    video_id = explicit.get("video_id") or extract_video_id(cleaned_query)

    # 4. Time Filter
    time_filter = extract_time_filters(cleaned_query)

    # 5. Result Limit
    limit = explicit.get("limit") or 10
    limit_match = re.search(r"(?:top|first|limit)\s+(\d+)", cleaned_query.lower())
    if limit_match and not explicit.get("limit"):
        limit = min(50, max(1, int(limit_match.group(1))))

    return {
        "raw_query": query,
        "clean_query": cleaned_query,
        "activity": activity,
        "min_confidence": min_confidence,
        "video_id": video_id,
        "time_filter": time_filter,
        "limit": limit,
        "was_injection_detected": was_injection
    }
