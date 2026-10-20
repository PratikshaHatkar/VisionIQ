from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

from src.api.storage import get_storage
from src.api.nlp.parser import parse_natural_query
from src.api.nlp.semantic import get_semantic_ranker
from src.api.nlp.synthesizer import generate_grounded_answer

def matches_wall_clock(
    event_start_sec: float,
    video_created_at: Optional[datetime],
    time_filter: Dict[str, Any]
) -> bool:
    """
    Evaluates whether an event satisfies a wall-clock time constraint (e.g. after 8 PM).
    Calculates actual wall-clock occurrence by offsetting video start timestamp.
    """
    if not video_created_at or not time_filter:
        return True

    # Calculate actual event time
    event_dt = video_created_at + timedelta(seconds=event_start_sec)
    ev_hour = event_dt.hour
    ev_minute = event_dt.minute

    filter_type = time_filter.get("type")

    if filter_type == "after":
        target_h = time_filter["hour"]
        target_m = time_filter.get("minute", 0)
        return (ev_hour > target_h) or (ev_hour == target_h and ev_minute >= target_m)

    elif filter_type == "before":
        target_h = time_filter["hour"]
        target_m = time_filter.get("minute", 0)
        return (ev_hour < target_h) or (ev_hour == target_h and ev_minute <= target_m)

    elif filter_type == "between":
        start_h = time_filter["start_hour"]
        end_h = time_filter["end_hour"]

        # Normal interval (e.g., between 6 PM (18) and 10 PM (22))
        if start_h <= end_h:
            return start_h <= ev_hour <= end_h
        else:
            # Range spanning midnight (e.g., between 10 PM (22) and 2 AM (2))
            return ev_hour >= start_h or ev_hour <= end_h

    return True

def matches_relative_time(
    event_start_sec: float,
    time_filter: Dict[str, Any]
) -> bool:
    """
    Evaluates video-relative time constraints (e.g. after 30 seconds into the video).
    """
    if not time_filter:
        return True

    filter_type = time_filter.get("type")

    if filter_type == "relative_after":
        return event_start_sec >= time_filter.get("start_sec", 0.0)

    elif filter_type == "relative_between":
        return (
            event_start_sec >= time_filter.get("start_sec", 0.0) and
            event_start_sec <= time_filter.get("end_sec", 999999.0)
        )

    return True

def execute_chat_query(
    query: str,
    video_id: Optional[str] = None,
    activity: Optional[str] = None,
    min_confidence: Optional[float] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Executes hybrid retrieval: natural-language parsing, database filtering,
    time reconciliation, semantic relevance ranking, and grounded answer synthesis.
    """
    repo = get_storage()
    ranker = get_semantic_ranker()

    # 1. Parse natural query and merge with explicit filters
    parsed = parse_natural_query(query, {
        "activity": activity,
        "video_id": video_id,
        "min_confidence": min_confidence,
        "limit": limit
    })

    time_notes = []
    if parsed.get("was_injection_detected"):
        time_notes.append("System security notice: Potentially unsafe control keywords were sanitized from the query.")

    # 2. Database Retrieval
    # Query database for candidate events using parsed or explicit filters
    _, candidate_events = repo.list_events(
        activity=parsed.get("activity"),
        video_id=parsed.get("video_id"),
        min_confidence=parsed.get("min_confidence"),
        limit=100,  # Broader pool for ranking and time evaluation
        offset=0
    )

    # 3. Lookup parent video metadata for filename and creation timestamp
    video_lookup = {}
    time_filtered_events = []
    time_filter = parsed.get("time_filter")

    for ev in candidate_events:
        vid_id = ev.get("video_id")
        video_doc = repo.get_video(vid_id) if vid_id else None

        filename = video_doc.get("filename", "") if video_doc else ""
        video_lookup[vid_id] = filename
        ev_enriched = dict(ev)
        ev_enriched["filename"] = filename

        # Evaluate time constraints
        if time_filter:
            if time_filter.get("is_wall_clock"):
                vid_created = video_doc.get("created_at") if video_doc else None
                if vid_created:
                    if matches_wall_clock(ev.get("start_timestamp", 0.0), vid_created, time_filter):
                        time_filtered_events.append(ev_enriched)
                else:
                    # Video has no created_at timestamp; include with note
                    time_filtered_events.append(ev_enriched)
            else:
                # Video-relative seconds
                if matches_relative_time(ev.get("start_timestamp", 0.0), time_filter):
                    time_filtered_events.append(ev_enriched)
        else:
            time_filtered_events.append(ev_enriched)

    if time_filter:
        if time_filter.get("is_wall_clock"):
            time_notes.append(
                f"Wall-clock filter ({time_filter['raw']}) evaluated against camera recording capture timestamps (UTC). "
                f"Elapsed video offsets are preserved."
            )
        else:
            time_notes.append(f"Video-relative time constraint applied: {time_filter['raw']}.")

    # 4. Semantic Ranking
    ranked_pairs = ranker.rank_events(
        query=parsed.get("clean_query", query),
        events=time_filtered_events,
        video_lookup=video_lookup,
        top_k=parsed.get("limit", 10)
    )

    final_events = []
    has_exact = bool(parsed.get("activity") or parsed.get("min_confidence") or parsed.get("video_id") or parsed.get("time_filter"))

    for item in ranked_pairs:
        if len(item) == 3:
            ev, score, sim = item
        else:
            ev, score = item
            sim = 1.0

        # When no exact filter is present, only include events that show real semantic relevance
        if not has_exact and sim < 0.05:
            continue

        ev_copy = dict(ev)
        ev_copy["relevance_score"] = score
        final_events.append(ev_copy)

    # 5. Determine Retrieval Method
    retrieval_method = "hybrid" if (has_exact and len(final_events) > 0) else ("exact_filter" if has_exact else "semantic")

    # 6. Evidence-Grounded Answer Synthesis
    grounded_res = generate_grounded_answer(
        query=query,
        parsed_filters=parsed,
        events=final_events,
        time_notes=time_notes
    )

    applied_filters = {
        "activity": parsed.get("activity"),
        "min_confidence": parsed.get("min_confidence"),
        "video_id": parsed.get("video_id"),
        "time_filter": time_filter.get("raw") if time_filter else None
    }

    return {
        "query": query,
        "answer": grounded_res["answer"],
        "applied_filters": applied_filters,
        "events": final_events,
        "total": len(final_events),
        "retrieval_method": retrieval_method,
        "limitations": grounded_res["limitations"]
    }
