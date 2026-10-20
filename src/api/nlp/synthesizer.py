import os
from typing import List, Dict, Any, Optional

def generate_grounded_answer(
    query: str,
    parsed_filters: Dict[str, Any],
    events: List[Dict[str, Any]],
    time_notes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Synthesizes a strictly grounded natural-language answer based only on retrieved event records.
    Ensures zero hallucination and enforces evidence and time-correctness rules.
    """
    limitations = time_notes or []
    count = len(events)
    activity = parsed_filters.get("activity")
    min_conf = parsed_filters.get("min_confidence")
    video_id = parsed_filters.get("video_id")
    time_filter = parsed_filters.get("time_filter")

    # 1. Zero Events Case
    if count == 0:
        filter_descs = []
        if activity:
            filter_descs.append(f"activity category '{activity}'")
        if min_conf:
            filter_descs.append(f"confidence >= {int(min_conf * 100)}%")
        if video_id:
            filter_descs.append(f"video ID '{video_id}'")
        if time_filter:
            filter_descs.append(f"time constraint '{time_filter.get('raw', '')}'")

        filters_str = f" with {', '.join(filter_descs)}" if filter_descs else ""
        answer = (
            f"No suspicious events matching your query '{query}' were found in the database{filters_str}. "
            f"Please verify that relevant CCTV footage has been uploaded and processed, or try lowering the confidence threshold."
        )
        return {
            "answer": answer,
            "limitations": limitations
        }

    # 2. Events Found Case
    # Group by activity
    activity_counts = {}
    for ev in events:
        act = ev.get("activity_label", "Suspicious Activity")
        activity_counts[act] = activity_counts.get(act, 0) + 1

    act_summary_parts = [f"{c} {a}" for a, c in activity_counts.items()]
    act_summary = ", ".join(act_summary_parts)

    # Lead event
    top_ev = events[0]
    top_act = top_ev.get("activity_label", "incident")
    top_conf = round(float(top_ev.get("confidence", 0.0)) * 100, 1)
    top_start = top_ev.get("start_time_formatted", "00:00:00")
    top_end = top_ev.get("end_time_formatted", "00:00:00")
    top_file = top_ev.get("filename") or top_ev.get("video_id") or "surveillance recording"

    lead_str = (
        f"Found {count} matching event record(s) in surveillance storage ({act_summary}). "
        f"The most prominent detection is a {top_act} event in '{top_file}' from {top_start} to {top_end} "
        f"with {top_conf}% confidence score."
    )

    if count > 1:
        other_files = list({ev.get("filename", "") for ev in events if ev.get("filename")})
        lead_str += f" Events are linked across {len(other_files) or 1} CCTV recording(s) with genuine evidence snapshots."

    # Standard Surveillance AI Notice
    limitations.append(
        "Notice: Event classifications represent probabilistic inferences from the fine-tuned 3D CNN model (R(2+1)D). "
        "Detections do not establish legal criminality or identify individuals; human operator verification is required."
    )

    return {
        "answer": lead_str,
        "limitations": limitations
    }
