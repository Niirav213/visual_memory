"""
Unit tests for Query Engine Service.
"""

import pytest
from services.memory_worker.persistence import MemoryPersistenceService
from services.embedding.embedder import EmbeddingService
from services.query_engine.engine import QueryEngineService
from packages.shared_types.models import QueryRequest


@pytest.fixture
def query_engine(tmp_path):
    db_file = str(tmp_path / "test_mem.db")
    faiss_file = str(tmp_path / "test_mem.faiss")
    id_map = str(tmp_path / "test_id.json")
    media_dir = str(tmp_path / "media")

    mem_service = MemoryPersistenceService(
        db_path=db_file,
        faiss_path=faiss_file,
        id_map_path=id_map,
        media_root=media_dir,
    )
    # Seed a sample sighting
    mem_service.record_sighting(
        camera_id="camera_01",
        zone_id="study_desk",
        track_id="trk_1",
        class_name="laptop",
        confidence=0.95,
        bbox_xyxy=(100, 100, 400, 300),
        frame=None,
        force=True,
    )

    embedder = EmbeddingService(lazy_load=True)
    return QueryEngineService(mem_service, embedder)


def test_where_query_found(query_engine):
    res = query_engine.query(QueryRequest(question="Where is my laptop?"))
    assert res.object_class == "laptop"
    assert "study_desk" in res.zone_id
    assert res.camera_id == "camera_01"
    assert "study desk" in res.answer.lower()
    assert not res.is_uncertain


def test_where_query_not_found(query_engine):
    res = query_engine.query(QueryRequest(question="Where are my keys?"))
    # Keys not in memory -> reports not found without hallucinating
    assert res.is_uncertain


def test_count_query(query_engine):
    res = query_engine.query(QueryRequest(question="How many times did you see a laptop?"))
    assert res.object_class == "laptop"
    assert res.sightings_count == 1


def test_list_objects_query(query_engine):
    res = query_engine.query(QueryRequest(question="What objects have you seen?"))
    assert "laptop" in res.answer.lower()
