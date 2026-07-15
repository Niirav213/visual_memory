"""
Visual Memory AI — Query & Retrieval Engine
Processes natural language questions and retrieves relevant memories.
"""

import re
from datetime import datetime
from typing import List, Optional, Tuple

from src.memory.embedder import VisualEmbedder
from src.memory.vector_store import VectorStore
from src.memory.memory_db import MemoryDatabase, MemoryEntry


class QueryEngine:
    """
    Natural language query engine for visual memories.
    Converts user questions into memory lookups and generates responses.
    """

    # Common object name aliases
    ALIASES = {
        "phone": "cell phone",
        "mobile": "cell phone",
        "mobile phone": "cell phone",
        "smartphone": "cell phone",
        "bag": "backpack",
        "computer": "laptop",
        "notebook": "laptop",
        "mug": "cup",
        "glass": "cup",
        "watch": "clock",
        "monitor": "tv",
        "screen": "tv",
        "television": "tv",
        "keys": "keyboard",
    }

    def __init__(
        self,
        embedder: VisualEmbedder,
        vector_store: VectorStore,
        memory_db: MemoryDatabase,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.memory_db = memory_db
        print("[QueryEngine] Initialized")

    def query(self, user_question: str) -> Tuple[str, Optional[MemoryEntry]]:
        """
        Process a natural language question and return a response.

        Supports queries like:
        - "Where is my laptop?"
        - "When did I last see my phone?"
        - "How many times did you see a bottle?"
        - "What objects have you seen?"
        """
        question = user_question.lower().strip()

        # Handle meta-queries
        if any(kw in question for kw in ["what objects", "what have you seen", "list objects", "what did you see"]):
            return self._handle_list_objects()

        if any(kw in question for kw in ["how many memories", "total memories", "memory count"]):
            return self._handle_memory_count()

        # Extract object name from the question
        object_name = self._extract_object(question)
        if not object_name:
            # Fallback: use CLIP text embedding for semantic search
            return self._handle_semantic_search(user_question)

        # Determine query type
        if any(kw in question for kw in ["where", "location", "position", "place"]):
            return self._handle_where(user_question, object_name)
        elif any(kw in question for kw in ["when", "last seen", "last time", "recently"]):
            return self._handle_when(user_question, object_name)
        elif any(kw in question for kw in ["how many", "how often", "count", "times"]):
            return self._handle_count(object_name)
        else:
            return self._handle_where(user_question, object_name)

    def _extract_object(self, question: str) -> Optional[str]:
        """Extract the object name from a question using known objects."""
        known_objects = self.memory_db.get_unique_objects()
        all_names = set(known_objects)

        # Check for aliases first
        for alias, canonical in self.ALIASES.items():
            if alias in question:
                if canonical in all_names:
                    return canonical

        # Check for direct matches (longest match first)
        sorted_names = sorted(all_names, key=len, reverse=True)
        for name in sorted_names:
            if name in question:
                return name

        # Try individual words against known objects
        words = question.split()
        for word in words:
            clean = re.sub(r'[^\w\s]', '', word)
            if clean in all_names:
                return clean
            if clean in self.ALIASES and self.ALIASES[clean] in all_names:
                return self.ALIASES[clean]

        return None

    def _retrieve_best_memory(self, question: str, object_name: str) -> Optional[MemoryEntry]:
        """Retrieve the best memory entry using hybrid keyword-semantic matching."""
        # Find all memories matching the object name first to ensure we have them
        memories = self.memory_db.get_memories_by_object(object_name, limit=100)
        if not memories:
            return None

        # If only one memory exists, return it
        if len(memories) == 1:
            return memories[0]

        # If there are multiple memories, use CLIP + FAISS to find the best match for the query
        query_emb = self.embedder.embed_text(question)
        results = self.vector_store.search(query_emb, top_k=50)

        # Filter search results for the target object name
        for memory_id, score in results:
            memory = self.memory_db.get_memory(memory_id)
            if memory and memory.object_name == object_name:
                return memory

        # Fallback to the latest memory if no match in top FAISS results
        return memories[0]

    def _handle_where(self, question: str, object_name: str) -> Tuple[str, Optional[MemoryEntry]]:
        memory = self._retrieve_best_memory(question, object_name)
        if not memory:
            return f"🔍 I don't have any memory of seeing a **{object_name}**.", None

        time_str = self._format_time(memory.timestamp)
        region = memory.region.replace("-", " ")
        return (
            f"📍 The **{object_name}** was last seen in the **{region}** "
            f"of the frame at **{time_str}** "
            f"(confidence: {memory.confidence:.0%}).",
            memory
        )

    def _handle_when(self, question: str, object_name: str) -> Tuple[str, Optional[MemoryEntry]]:
        memory = self._retrieve_best_memory(question, object_name)
        if not memory:
            return f"🔍 I don't have any memory of seeing a **{object_name}**.", None

        time_str = self._format_time(memory.timestamp)
        ago = memory.time_ago
        return (
            f"🕐 The **{object_name}** was last seen at **{time_str}** "
            f"(**{ago}**), in the **{memory.region.replace('-', ' ')}** region.",
            memory
        )

    def _handle_count(self, object_name: str) -> Tuple[str, Optional[MemoryEntry]]:
        memories = self.memory_db.get_memories_by_object(object_name, limit=1000)
        count = len(memories)
        if count == 0:
            return f"🔍 I haven't seen any **{object_name}** yet.", None

        first = memories[-1]
        last = memories[0]
        return (
            f"📊 I've recorded **{count}** sightings of **{object_name}**.\n\n"
            f"  • First seen: {self._format_time(first.timestamp)}\n"
            f"  • Last seen: {self._format_time(last.timestamp)}",
            last
        )

    def _handle_list_objects(self) -> Tuple[str, Optional[MemoryEntry]]:
        objects = self.memory_db.get_unique_objects()
        if not objects:
            return "🔍 I haven't seen any objects yet. Start the camera to begin building memories!", None

        total = self.memory_db.get_total_count()
        obj_list = "\n".join(f"  • **{obj}**" for obj in objects)
        return (
            f"📋 I've seen **{len(objects)}** unique objects "
            f"across **{total}** total memories:\n\n{obj_list}",
            None
        )

    def _handle_memory_count(self) -> Tuple[str, Optional[MemoryEntry]]:
        total = self.memory_db.get_total_count()
        objects = self.memory_db.get_unique_objects()
        return (
            f"🧠 Memory status: **{total}** total memories "
            f"covering **{len(objects)}** unique objects.",
            None
        )

    def _handle_semantic_search(self, question: str) -> Tuple[str, Optional[MemoryEntry]]:
        """Use CLIP text embedding to find semantically similar memories."""
        query_emb = self.embedder.embed_text(question)
        results = self.vector_store.search(query_emb, top_k=3)

        if not results:
            return "🔍 I couldn't find any relevant memories. Try asking about a specific object!", None

        responses = ["🔎 Here are the closest matches from my memory:\n"]
        best_memory = None
        for i, (memory_id, score) in enumerate(results):
            memory = self.memory_db.get_memory(memory_id)
            if memory:
                if i == 0:
                    best_memory = memory
                time_str = self._format_time(memory.timestamp)
                region = memory.region.replace("-", " ")
                responses.append(
                    f"  • **{memory.object_name}** — {region} at {time_str} "
                    f"(match: {score:.0%})"
                )

        return "\n".join(responses), best_memory

    def _format_time(self, iso_timestamp: str) -> str:
        """Format ISO timestamp to human-readable string."""
        try:
            dt = datetime.fromisoformat(iso_timestamp)
            return dt.strftime("%I:%M:%S %p")
        except Exception:
            return iso_timestamp
