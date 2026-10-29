from datetime import datetime, timezone
from typing import Optional, Dict, Any

def generate_incident_summary(
    activity_label: str,
    camera_name: str,
    location: Optional[str] = None,
    confidence: Optional[float] = None,
    timestamp: Optional[datetime] = None,
    duration_seconds: Optional[float] = None,
    stream_type: Optional[str] = None
) -> str:
    """
    Generates a concise, factual, and strictly grounded short incident summary
    for a live surveillance detection event.

    Example:
    'Possible fighting activity detected at 10:42:15 on Camera 1 (North Gate) with 88% confidence. Review the evidence to verify the incident.'
    """
    try:
        # 1. Format timestamp
        if timestamp:
            if isinstance(timestamp, str):
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime("%H:%M:%S")
                except Exception:
                    time_str = timestamp
            elif isinstance(timestamp, datetime):
                time_str = timestamp.strftime("%H:%M:%S")
            else:
                time_str = str(timestamp)
        else:
            time_str = datetime.now(timezone.utc).strftime("%H:%M:%S")

        # 2. Activity descriptor
        act = (activity_label or "suspicious").lower()

        # 3. Location descriptor
        loc_str = f" ({location})" if location and location.strip() else ""

        # 4. Confidence descriptor
        if confidence is not None:
            conf_val = round(float(confidence) * 100, 1)
            conf_str = f" with {conf_val}% confidence"
        else:
            conf_str = ""

        # 5. Ongoing duration descriptor
        dur_str = ""
        if duration_seconds is not None and duration_seconds > 2.0:
            dur_str = f" Ongoing event lasting {duration_seconds:.1f}s."

        # 6. Compose grounded summary
        summary = (
            f"Possible {act} activity detected at {time_str} on {camera_name}{loc_str}{conf_str}."
            f"{dur_str} Review the evidence to verify the incident."
        )
        return summary
    except Exception as e:
        # Fail-safe minimal fallback if formatting fails
        print(f"[Summarizer Warning] Error formatting summary: {e}")
        act_clean = (activity_label or "suspicious").lower()
        cam_clean = camera_name or "camera"
        return f"Possible {act_clean} activity detected on {cam_clean}. Review the evidence to verify the incident."
