import os
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import CLASSES

def format_event_description(event: Dict[str, Any], filename: str = "") -> str:
    """
    Creates a rich, searchable textual representation of a detected event.
    """
    activity = event.get("activity_label", "Unknown")
    confidence_pct = round(float(event.get("confidence", 0.0)) * 100, 1)
    start_time = event.get("start_time_formatted", "00:00:00")
    end_time = event.get("end_time_formatted", "00:00:00")
    start_sec = event.get("start_timestamp", 0.0)
    end_sec = event.get("end_timestamp", 0.0)
    video_id = event.get("video_id", "")

    fname_str = f" in surveillance file {filename}" if filename else ""
    return (
        f"Suspicious activity {activity} incident detected{fname_str} with {confidence_pct}% model confidence. "
        f"Occurrence time from {start_time} to {end_time} (seconds {start_sec} to {end_sec}). "
        f"Source recording video ID {video_id}. Action category: {activity}."
    )

class SemanticRanker:
    """
    Hybrid semantic retrieval and relevance ranking engine using TF-IDF vectorization
    and cosine similarity over stored event metadata.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )

    def rank_events(
        self,
        query: str,
        events: List[Dict[str, Any]],
        video_lookup: Dict[str, str] = None,
        top_k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Ranks candidate event records against the natural language query.
        Returns a list of tuples: (event_record, relevance_score).
        """
        if not events or not query.strip():
            return []

        v_lookup = video_lookup or {}

        # 1. Build text corpus
        corpus = [
            format_event_description(ev, v_lookup.get(ev.get("video_id"), ""))
            for ev in events
        ]

        try:
            # Fit and transform corpus + query
            all_texts = corpus + [query]
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)

            # Query vector is the last row, event vectors are rows 0 to len(corpus)-1
            query_vec = tfidf_matrix[-1]
            event_vecs = tfidf_matrix[:-1]

            # Compute cosine similarity
            sim_scores = cosine_similarity(query_vec, event_vecs).flatten()
        except Exception as e:
            # Fallback to confidence-based ranking if vectorizer encountered an issue
            sim_scores = np.zeros(len(events))

        # 2. Compute composite relevance score:
        # Score = 0.5 * Semantic_Similarity + 0.5 * Detection_Confidence
        ranked = []
        for idx, ev in enumerate(events):
            sim = float(sim_scores[idx]) if idx < len(sim_scores) else 0.0
            conf = float(ev.get("confidence", 0.0))

            # Composite ranking
            relevance = round(0.5 * sim + 0.5 * conf, 4)
            ranked.append((ev, relevance, sim))

        # Sort descending by relevance score
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

# Global ranker instance
ranker = SemanticRanker()

def get_semantic_ranker() -> SemanticRanker:
    return ranker
