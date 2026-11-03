"""
Visual Memory AI — Evidence-Backed Query & Retrieval Engine
Combines structured SQL filters, temporal logic, and vector retrieval.
Returns evidence-backed answers rather than hallucinating or guessing.
"""

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from packages.shared_types.models import QueryRequest, QueryResponse, MemoryEvent
from services.memory_worker.persistence import MemoryPersistenceService
from services.embedding.embedder import EmbeddingService


class QueryEngineService:
    """
    Decoupled query engine implementing explicit temporal rules,
    vector search, and structured SQL aggregation.
    """

    ALIASES = {
        "phone": "cell phone",
        "mobile": "cell phone",
        "smartphone": "cell phone",
        "iphone": "cell phone",
        "android": "cell phone",
        "laptop": "laptop",
        "macbook": "laptop",
        "notebook": "laptop",
        "computer": "laptop",
        "bottle": "bottle",
        "water bottle": "bottle",
        "cup": "cup",
        "mug": "cup",
        "coffee mug": "cup",
        "bag": "backpack",
        "backpack": "backpack",
        "keys": "keyboard",  # COCO keyboard proxy if keys not trained
        "remote": "remote",
        "mouse": "mouse",
        "book": "book",
        "glasses": "scissors",
    }

    def __init__(
        self,
        memory_service: MemoryPersistenceService,
        embedding_service: EmbeddingService,
    ):
        self.memory = memory_service
        self.embedder = embedding_service
        print("[QueryEngineService] Initialized.")

    def query(self, req: QueryRequest) -> QueryResponse:
        """
        Process user question with deterministic temporal rules and vector fallback.
        """
        q_raw = req.question.strip()
        q_lower = q_raw.lower()

        # 1. Meta / Inventory queries
        if any(kw in q_lower for kw in ["what objects", "what have you seen", "list objects", "inventory"]):
            stats = self.memory.get_stats()
            objs = stats["unique_objects"]
            if not objs:
                return QueryResponse(
                    question=q_raw,
                    answer="No visual memories recorded yet. Start camera ingestion to begin observing objects.",
                    is_uncertain=False,
                )
            obj_list_str = ", ".join(f"**{o}**" for o in sorted(objs))
            return QueryResponse(
                question=q_raw,
                answer=f"I have recorded visual memories for {len(objs)} distinct object categories: {obj_list_str}.",
                all_known_objects=sorted(objs),
            )

        # 2. Extract target object class
        target_class = self._extract_class(q_lower)

        # 3. Disappearance / Return / Movement queries
        if any(kw in q_lower for kw in ["disappear", "return", "leave", "gap", "come back"]):
            return self._handle_disappearance_query(q_raw, target_class)

        # 4. Count aggregation queries
        if any(kw in q_lower for kw in ["how many times", "count", "how often", "how many"]):
            return self._handle_count_query(q_raw, target_class)

        # 5. When / Last seen queries
        if any(kw in q_lower for kw in ["when", "time", "how long ago", "at what time"]):
            return self._handle_when_query(q_raw, target_class, req.zone_id)

        # 6. Where / Location queries
        if any(kw in q_lower for kw in ["where", "location", "find", "which room", "which desk", "which zone"]):
            return self._handle_where_query(q_raw, target_class, req.zone_id)

        # 7. Semantic / Descriptive search fallback (e.g., "red cup near window", "black backpack")
        return self._handle_semantic_query(q_raw, target_class)

    def _extract_class(self, q: str) -> Optional[str]:
        # Check aliases first
        for alias, canonical in self.ALIASES.items():
            if re.search(r'\b' + re.escape(alias) + r'\b', q):
                return canonical

        # Check known objects in database
        stats = self.memory.get_stats()
        for obj in stats["unique_objects"]:
            if re.search(r'\b' + re.escape(obj.lower()) + r'\b', q):
                return obj

        return None

    def _handle_where_query(self, question: str, target_class: Optional[str], zone_id: Optional[str]) -> QueryResponse:
        if not target_class:
            return self._handle_semantic_query(question, None)

        evt = self.memory.get_latest_event_for_object(target_class, zone_id=zone_id)
        if not evt:
            return QueryResponse(
                question=question,
                answer=f"I have no recorded sighting of a **{target_class}** in memory.",
                object_class=target_class,
                is_uncertain=True,
                disclaimer="Object has not been observed by active camera streams.",
            )

        time_ago_str = self._calculate_time_ago(evt.timestamp_utc)
        clean_zone = evt.zone_id.replace("_", " ").title()

        answer = (
            f"The **{target_class}** was last observed in **{clean_zone}** "
            f"on camera **{evt.camera_id}** ({time_ago_str}, confidence: {evt.confidence:.0%})."
        )

        return QueryResponse(
            question=question,
            answer=answer,
            object_class=target_class,
            camera_id=evt.camera_id,
            zone_id=evt.zone_id,
            timestamp_utc=evt.timestamp_utc,
            time_ago=time_ago_str,
            confidence=evt.confidence,
            event_id=evt.event_id,
            snapshot_uri=evt.snapshot_uri,
            crop_uri=evt.crop_uri,
            disclaimer="Reporting last known sighting timestamp—not assuming unobserved current location.",
        )

    def _handle_when_query(self, question: str, target_class: Optional[str], zone_id: Optional[str]) -> QueryResponse:
        if not target_class:
            return self._handle_semantic_query(question, None)

        evt = self.memory.get_latest_event_for_object(target_class, zone_id=zone_id)
        if not evt:
            return QueryResponse(
                question=question,
                answer=f"No sightings of **{target_class}** were found in memory.",
                object_class=target_class,
                is_uncertain=True,
            )

        time_ago_str = self._calculate_time_ago(evt.timestamp_utc)
        formatted_dt = self._format_timestamp(evt.timestamp_utc)
        clean_zone = evt.zone_id.replace("_", " ").title()

        answer = (
            f"The **{target_class}** was last seen at **{formatted_dt}** ({time_ago_str}) "
            f"in **{clean_zone}** (camera: {evt.camera_id})."
        )

        return QueryResponse(
            question=question,
            answer=answer,
            object_class=target_class,
            camera_id=evt.camera_id,
            zone_id=evt.zone_id,
            timestamp_utc=evt.timestamp_utc,
            time_ago=time_ago_str,
            confidence=evt.confidence,
            event_id=evt.event_id,
            snapshot_uri=evt.snapshot_uri,
            crop_uri=evt.crop_uri,
        )

    def _handle_count_query(self, question: str, target_class: Optional[str]) -> QueryResponse:
        if not target_class:
            return QueryResponse(
                question=question,
                answer="Could not identify which object category to count. Try asking e.g. 'How many times have you seen a bottle?'",
                is_uncertain=True,
            )

        events = self.memory.get_sightings_for_object(target_class, limit=1000)
        count = len(events)
        if count == 0:
            return QueryResponse(
                question=question,
                answer=f"I have not recorded any sightings of **{target_class}**.",
                object_class=target_class,
                sightings_count=0,
            )

        first_seen = self._format_timestamp(events[-1].timestamp_utc)
        last_seen = self._format_timestamp(events[0].timestamp_utc)
        latest = events[0]

        answer = (
            f"Recorded **{count} distinct sighting events** of **{target_class}**.\n"
            f"• First observation: {first_seen}\n"
            f"• Most recent observation: {last_seen} ({self._calculate_time_ago(latest.timestamp_utc)})"
        )

        return QueryResponse(
            question=question,
            answer=answer,
            object_class=target_class,
            sightings_count=count,
            camera_id=latest.camera_id,
            zone_id=latest.zone_id,
            timestamp_utc=latest.timestamp_utc,
            time_ago=self._calculate_time_ago(latest.timestamp_utc),
            confidence=latest.confidence,
            event_id=latest.event_id,
            snapshot_uri=latest.snapshot_uri,
            crop_uri=latest.crop_uri,
        )

    def _handle_disappearance_query(self, question: str, target_class: Optional[str]) -> QueryResponse:
        if not target_class:
            return QueryResponse(
                question=question,
                answer="Please specify the object to analyze (e.g. 'When did the laptop disappear and return?').",
                is_uncertain=True,
            )

        timeline = self.memory.get_timeline(object_class=target_class, limit=50)
        gaps = [e for e in timeline if e.gap_duration_sec and e.gap_duration_sec > 15]

        if not gaps:
            evt = self.memory.get_latest_event_for_object(target_class)
            if evt:
                return QueryResponse(
                    question=question,
                    answer=f"The **{target_class}** has been consistently observed in **{evt.zone_id}** without significant gaps detected.",
                    object_class=target_class,
                    event_id=evt.event_id,
                    snapshot_uri=evt.snapshot_uri,
                    crop_uri=evt.crop_uri,
                )
            return QueryResponse(
                question=question,
                answer=f"No observation records found for **{target_class}**.",
                is_uncertain=True,
            )

        # Describe most prominent gap
        top_gap = gaps[0]
        answer = (
            f"Detected visibility gap for **{target_class}**: "
            f"disappeared and reappeared at **{top_gap.time_display}** in **{top_gap.zone_id}** "
            f"(unobserved gap of ~{top_gap.gap_duration_sec} seconds). "
            f"Note: Gaps indicate when the object left camera visibility or was occluded."
        )

        return QueryResponse(
            question=question,
            answer=answer,
            object_class=target_class,
            camera_id=top_gap.camera_id,
            zone_id=top_gap.zone_id,
            timestamp_utc=top_gap.timestamp_utc,
            event_id=top_gap.id,
            snapshot_uri=top_gap.snapshot_uri,
            crop_uri=top_gap.crop_uri,
            confidence=top_gap.confidence,
            disclaimer="Visibility gaps reflect camera viewpoint coverage and occlusion.",
        )

    def _handle_semantic_query(self, question: str, target_class: Optional[str]) -> QueryResponse:
        """CLIP text-image vector similarity search."""
        query_vec = self.embedder.embed_text(question)
        matches = self.memory.vector_search(query_vec, top_k=5)

        if not matches:
            return QueryResponse(
                question=question,
                answer="No matching visual memories found for your description. Try specifying the object name.",
                is_uncertain=True,
            )

        # Filter for target class if identified
        best_evt, best_score = matches[0]
        if target_class:
            for evt, score in matches:
                if evt.class_name.lower() == target_class.lower():
                    best_evt, best_score = evt, score
                    break

        clean_zone = best_evt.zone_id.replace("_", " ").title()
        time_ago = self._calculate_time_ago(best_evt.timestamp_utc)

        answer = (
            f"Found visual memory matching '{question}': **{best_evt.class_name}** "
            f"in **{clean_zone}** ({time_ago}, semantic similarity: {best_score:.0%})."
        )

        return QueryResponse(
            question=question,
            answer=answer,
            object_class=best_evt.class_name,
            camera_id=best_evt.camera_id,
            zone_id=best_evt.zone_id,
            timestamp_utc=best_evt.timestamp_utc,
            time_ago=time_ago,
            confidence=best_evt.confidence,
            event_id=best_evt.event_id,
            snapshot_uri=best_evt.snapshot_uri,
            crop_uri=best_evt.crop_uri,
        )

    def _calculate_time_ago(self, timestamp_utc: str) -> str:
        try:
            dt = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            delta = now - dt
            sec = int(delta.total_seconds())
            if sec < 60:
                return f"{sec}s ago"
            elif sec < 3600:
                return f"{sec // 60}m ago"
            elif sec < 86400:
                return f"{sec // 3600}h ago"
            else:
                return f"{sec // 86400}d ago"
        except Exception:
            return timestamp_utc

    def _format_timestamp(self, timestamp_utc: str) -> str:
        try:
            dt = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00"))
            return dt.strftime("%b %d, %H:%M:%S UTC")
        except Exception:
            return timestamp_utc
