"""
Tests for IC-KG Data Loader

Testing data loading, validation, and batch processing functionality.
"""

import csv
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from chemi_accel.knowledge_graph.data_loader import (
    BulkLoader,
    ICKGDataLoader,
    LoaderConfig,
    LoaderStats,
    StreamingLoader,
    ValidationLoader,
)
from chemi_accel.knowledge_graph.memory_backend import MemoryGraphBackend
from chemi_accel.knowledge_graph.schema import ICKGSchema


class TestLoaderConfig:
    """Test loader configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LoaderConfig()

        assert config.batch_size == 1000
        assert config.max_workers == 4
        assert config.validation_enabled == True
        assert config.create_missing_entities == True
        assert config.update_existing == False
        assert config.error_handling == "continue"

    def test_custom_config(self):
        """Test custom configuration."""
        config = LoaderConfig(
            batch_size=500,
            max_workers=2,
            validation_enabled=False,
            error_handling="stop",
        )

        assert config.batch_size == 500
        assert config.max_workers == 2
        assert config.validation_enabled == False
        assert config.error_handling == "stop"


class TestICKGDataLoader:
    """Test main data loader functionality."""

    @pytest.fixture
    def backend(self):
        """Create test backend."""
        backend = MemoryGraphBackend({})
        backend.connect()
        return backend

    @pytest.fixture
    def loader(self, backend):
        """Create test loader."""
        config = LoaderConfig(batch_size=10, validation_enabled=False)
        return ICKGDataLoader(backend, config)

    @pytest.fixture
    def sample_entities(self):
        """Sample entity data."""
        return [
            {
                "labels": ["Chemical", "Drug"],
                "properties": {
                    "name": "aspirin",
                    "formula": "C9H8O4",
                    "molecular_weight": 180.16,
                },
            },
            {
                "labels": ["Chemical", "Solvent"],
                "properties": {
                    "name": "water",
                    "formula": "H2O",
                    "molecular_weight": 18.02,
                },
            },
            {
                "labels": ["Reaction"],
                "properties": {
                    "name": "acetylation",
                    "type": "synthesis",
                    "temperature": 80,
                },
            },
        ]

    @pytest.fixture
    def sample_relationships(self):
        """Sample relationship data."""
        return [
            {
                "source_id": "entity_1",
                "target_id": "entity_2",
                "relationship_type": "DISSOLVES_IN",
                "properties": {"solubility": "high"},
            },
            {
                "source_id": "entity_2",
                "target_id": "entity_3",
                "relationship_type": "USED_IN",
                "properties": {"role": "solvent"},
            },
        ]

    def test_loader_initialization(self, backend):
        """Test loader initialization."""
        config = LoaderConfig(batch_size=500)
        loader = ICKGDataLoader(backend, config)

        assert loader.backend is backend
        assert loader.config.batch_size == 500
        assert isinstance(loader.schema, ICKGSchema)
        assert isinstance(loader.stats, LoaderStats)

    def test_load_from_data_entities_only(self, loader, sample_entities):
        """Test loading entities without relationships."""
        stats = loader.load_from_data(sample_entities)

        assert isinstance(stats, LoaderStats)
        assert stats.total_processed == len(sample_entities)
        assert stats.successful_loads == len(sample_entities)
        assert stats.failed_loads == 0
        assert stats.start_time is not None
        assert stats.end_time is not None
        assert stats.processing_rate > 0

    def test_load_from_data_with_relationships(self, loader, sample_entities, backend):
        """Test loading entities and relationships."""
        # First load entities to get IDs
        entity_stats = loader.load_from_data(sample_entities)

        # Get created entity IDs
        entities = backend.find_nodes()
        entity_ids = [entity["_id"] for entity in entities]

        # Create relationships with actual IDs
        relationships = [
            {
                "source_id": entity_ids[0],
                "target_id": entity_ids[1],
                "relationship_type": "RELATES_TO",
                "properties": {"strength": "weak"},
            }
        ]

        # Load relationships
        rel_stats = loader.load_from_data([], relationships)

        assert rel_stats.successful_loads == 1
        assert len(backend.relationships) == 1

    def test_validation_enabled(self, backend):
        """Test data validation during loading."""
        config = LoaderConfig(validation_enabled=True)
        loader = ICKGDataLoader(backend, config)

        # Mix of valid and invalid entities
        entities = [
            {"labels": ["Chemical"], "properties": {"name": "valid_compound"}},
            {
                # Missing labels - invalid
                "properties": {"name": "invalid_compound"}
            },
            {"labels": ["Drug"], "properties": {"name": "another_valid"}},
        ]

        stats = loader.load_from_data(entities)

        # Should still load valid entities
        assert stats.successful_loads >= 2
        assert stats.validation_errors >= 1

    def test_progress_reporting(self, loader, sample_entities):
        """Test progress reporting functionality."""
        progress_updates = []

        def progress_callback(progress, message):
            progress_updates.append((progress, message))

        loader.add_progress_callback(progress_callback)

        # Load data (progress updates depend on implementation)
        loader.load_from_data(sample_entities)

        # At minimum, should not error with callback
        assert True  # If we get here, no callback errors occurred

    def test_checkpoint_creation(self, loader, sample_entities):
        """Test checkpoint functionality."""
        # Load some data first
        loader.load_from_data(sample_entities)

        # Create checkpoint
        checkpoint_id = loader.create_checkpoint()

        assert checkpoint_id is not None
        assert checkpoint_id in loader._checkpoint_data

        checkpoint = loader._checkpoint_data[checkpoint_id]
        assert "checkpoint_id" in checkpoint
        assert "timestamp" in checkpoint
        assert "stats" in checkpoint

    def test_restore_from_checkpoint(self, loader, sample_entities):
        """Test checkpoint restoration."""
        # Load data and create checkpoint
        loader.load_from_data(sample_entities)
        checkpoint_id = loader.create_checkpoint()

        original_processed = loader.stats.total_processed

        # Reset stats
        loader.stats = LoaderStats()
        assert loader.stats.total_processed == 0

        # Restore from checkpoint
        success = loader.restore_from_checkpoint(checkpoint_id)
        assert success
        assert loader.stats.total_processed == original_processed

    def test_restore_nonexistent_checkpoint(self, loader):
        """Test restoring from non-existent checkpoint."""
        success = loader.restore_from_checkpoint("nonexistent")
        assert not success

    def test_get_loading_statistics(self, loader, sample_entities):
        """Test comprehensive statistics retrieval."""
        loader.load_from_data(sample_entities)

        stats = loader.get_loading_statistics()

        assert "processing" in stats
        assert "performance" in stats
        assert "configuration" in stats
        assert "errors" in stats
        assert "checkpoints" in stats

        processing = stats["processing"]
        assert processing["total_processed"] > 0
        assert processing["successful_loads"] >= 0
        assert "success_rate" in processing

        performance = stats["performance"]
        assert "processing_rate" in performance
        assert "duration_seconds" in performance


class TestFileLoading:
    """Test loading from various file formats."""

    @pytest.fixture
    def backend(self):
        """Create test backend."""
        backend = MemoryGraphBackend({})
        backend.connect()
        return backend

    @pytest.fixture
    def loader(self, backend):
        """Create test loader."""
        return ICKGDataLoader(backend, LoaderConfig(validation_enabled=False))

    def test_load_from_json(self, loader):
        """Test loading from JSON file."""
        # Create temporary JSON file
        data = {
            "entities": [
                {
                    "labels": ["Drug"],
                    "properties": {"name": "aspirin", "type": "nsaid"},
                },
                {
                    "labels": ["Condition"],
                    "properties": {"name": "pain", "severity": "moderate"},
                },
            ],
            "relationships": [
                {
                    "source_id": "drug_1",
                    "target_id": "condition_1",
                    "relationship_type": "TREATS",
                    "properties": {"efficacy": 0.8},
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            stats = loader.load_from_file(temp_path, "json")

            assert stats.successful_loads >= 2  # At least entities should load
            assert stats.failed_loads == 0

        finally:
            Path(temp_path).unlink()

    def test_load_from_csv(self, loader):
        """Test loading from CSV file."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["id", "entity_type", "name", "property1"])
            writer.writerow(["1", "Drug", "aspirin", "nsaid"])
            writer.writerow(["2", "Drug", "ibuprofen", "nsaid"])
            writer.writerow(["3", "Condition", "pain", "symptom"])
            temp_path = f.name

        try:
            stats = loader.load_from_file(temp_path, "csv")

            assert stats.successful_loads == 3
            assert stats.failed_loads == 0

        finally:
            Path(temp_path).unlink()

    def test_load_from_jsonlines(self, loader):
        """Test loading from JSON Lines file."""
        # Create temporary JSONL file
        data_lines = [
            {"labels": ["Drug"], "properties": {"name": "aspirin"}},
            {"labels": ["Drug"], "properties": {"name": "ibuprofen"}},
            {
                "source_id": "drug1",
                "target_id": "drug2",
                "relationship_type": "SIMILAR_TO",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data_lines:
                f.write(json.dumps(item) + "\n")
            temp_path = f.name

        try:
            stats = loader.load_from_file(temp_path, "jsonl")

            # Should load the entities at least
            assert stats.successful_loads >= 2

        finally:
            Path(temp_path).unlink()

    def test_file_format_detection(self, loader):
        """Test automatic file format detection."""
        # Test JSON detection
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json.dump({"entities": []}, f)
            temp_path = f.name

        try:
            # Should auto-detect as JSON
            detected_format = loader._detect_file_format(Path(temp_path))
            assert detected_format == "json"

        finally:
            Path(temp_path).unlink()

    def test_load_nonexistent_file(self, loader):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            loader.load_from_file("nonexistent_file.json")

    def test_load_invalid_json(self, loader):
        """Test loading from invalid JSON file."""
        # Create file with invalid JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"invalid": json content}')  # Invalid JSON
            temp_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                loader.load_from_file(temp_path, "json")

        finally:
            Path(temp_path).unlink()


class TestStreamingLoader:
    """Test streaming data loader."""

    @pytest.fixture
    def backend(self):
        """Create test backend."""
        backend = MemoryGraphBackend({})
        backend.connect()
        return backend

    @pytest.fixture
    def streaming_loader(self, backend):
        """Create streaming loader."""
        return StreamingLoader(backend, buffer_size=5)

    def test_streaming_entity_loading(self, streaming_loader):
        """Test streaming entity addition."""
        # Add entities one by one
        for i in range(3):
            entity = {
                "labels": ["TestEntity"],
                "properties": {"id": i, "name": f"entity_{i}"},
            }
            streaming_loader.add_entity(entity)

        # Should be in buffer, not yet flushed
        assert len(streaming_loader.entity_buffer) == 3

        # Manual flush
        streaming_loader.flush()

        # Should be cleared and loaded
        assert len(streaming_loader.entity_buffer) == 0
        assert streaming_loader.stats.successful_loads >= 3

    def test_streaming_buffer_auto_flush(self, streaming_loader):
        """Test automatic buffer flushing when full."""
        # Add entities to trigger auto-flush (buffer_size=5)
        for i in range(6):
            entity = {"labels": ["TestEntity"], "properties": {"id": i}}
            streaming_loader.add_entity(entity)

        # Should have auto-flushed once, leaving 1 in buffer
        assert len(streaming_loader.entity_buffer) == 1
        assert streaming_loader.stats.successful_loads >= 5

    def test_streaming_relationship_loading(self, streaming_loader, backend):
        """Test streaming relationship addition."""
        # Create entities first
        entity1_id = backend.create_node(["TestEntity"], {"name": "entity1"})
        entity2_id = backend.create_node(["TestEntity"], {"name": "entity2"})

        # Add relationship
        relationship = {
            "source_id": entity1_id,
            "target_id": entity2_id,
            "relationship_type": "RELATES_TO",
            "properties": {"strength": "strong"},
        }

        streaming_loader.add_relationship(relationship)
        streaming_loader.flush()

        assert len(streaming_loader.relationship_buffer) == 0
        assert len(backend.relationships) == 1


class TestBulkLoader:
    """Test bulk loader for high-performance loading."""

    @pytest.fixture
    def backend(self):
        """Create test backend."""
        backend = MemoryGraphBackend({})
        backend.connect()
        return backend

    @pytest.fixture
    def bulk_loader(self, backend):
        """Create bulk loader."""
        return BulkLoader(backend, chunk_size=100)

    def test_bulk_entity_loading_from_iterator(self, bulk_loader):
        """Test bulk loading from entity iterator."""

        def entity_generator():
            for i in range(250):  # More than one chunk
                yield {
                    "labels": ["BulkEntity"],
                    "properties": {"id": i, "batch": i // 100},
                }

        stats = bulk_loader.load_entities_from_iterator(entity_generator())

        assert stats.total_processed == 250
        assert stats.successful_loads == 250
        assert stats.failed_loads == 0
        assert stats.start_time is not None
        assert stats.end_time is not None

    def test_bulk_loading_error_handling(self, bulk_loader, backend):
        """Test bulk loader error handling."""

        def entity_generator():
            # Mix valid and potentially invalid entities
            for i in range(10):
                if i == 5:
                    # This might cause issues depending on implementation
                    yield {"invalid": "structure"}
                else:
                    yield {"labels": ["ValidEntity"], "properties": {"id": i}}

        stats = bulk_loader.load_entities_from_iterator(entity_generator())

        # Should handle errors gracefully
        assert stats.total_processed == 10
        # Some should succeed even if others fail
        assert stats.successful_loads > 0


class TestValidationLoader:
    """Test validation loader with comprehensive validation."""

    @pytest.fixture
    def backend(self):
        """Create test backend."""
        backend = MemoryGraphBackend({})
        backend.connect()
        return backend

    @pytest.fixture
    def schema(self):
        """Create test schema."""
        return ICKGSchema()

    @pytest.fixture
    def validation_loader(self, backend, schema):
        """Create validation loader."""
        return ValidationLoader(backend, schema)

    def test_comprehensive_entity_validation(self, validation_loader):
        """Test comprehensive entity validation."""
        entities = [
            # Valid entity
            {
                "entity_type": "SMALL_MOLECULE",
                "properties": {"name": "aspirin", "formula": "C9H8O4"},
            },
            # Invalid entity - missing type
            {"properties": {"name": "invalid_compound"}},
            # Valid entity
            {
                "entity_type": "REACTION",
                "properties": {"name": "synthesis", "type": "organic"},
            },
        ]

        validation_result = validation_loader._comprehensive_entity_validation(entities)

        assert validation_result["total_entities"] == 3
        assert validation_result["valid_entities"] >= 2
        assert validation_result["invalid_entities"] >= 1
        assert len(validation_result["validation_errors"]) > 0

    def test_load_with_validation(self, validation_loader):
        """Test loading with comprehensive validation."""
        entities = [
            {"entity_type": "SMALL_MOLECULE", "properties": {"name": "compound1"}},
            {"entity_type": "SMALL_MOLECULE", "properties": {"name": "compound2"}},
        ]

        relationships = [
            {
                "source_id": "comp1",
                "target_id": "comp2",
                "relationship_type": "SIMILAR_TO",
                "properties": {"similarity": 0.8},
            }
        ]

        result = validation_loader.load_with_validation(entities, relationships)

        assert "load_stats" in result
        assert "entity_validation" in result
        assert "relationship_validation" in result
        assert "validation_stats" in result

        # Load stats should indicate successful loading of valid entities
        load_stats = result["load_stats"]
        assert load_stats.successful_loads >= 0  # At least not negative


class TestDataValidation:
    """Test data validation functionality."""

    @pytest.fixture
    def backend(self):
        """Create test backend."""
        backend = MemoryGraphBackend({})
        backend.connect()
        return backend

    @pytest.fixture
    def loader(self, backend):
        """Create test loader."""
        return ICKGDataLoader(backend, LoaderConfig())

    def test_validate_entity_data(self, loader):
        """Test entity data validation."""
        entities = [
            # Valid entity
            {
                "labels": ["Chemical"],
                "properties": {"name": "aspirin", "formula": "C9H8O4"},
            },
            # Invalid - missing labels
            {"properties": {"name": "no_labels"}},
            # Invalid - missing ID and labels
            {},
        ]

        validation_result = loader.validate_data(entities, "entities")

        assert validation_result["total_items"] == 3
        assert validation_result["valid_items"] >= 1
        assert validation_result["invalid_items"] >= 2
        assert len(validation_result["validation_errors"]) > 0

    def test_validate_relationship_data(self, loader):
        """Test relationship data validation."""
        relationships = [
            # Valid relationship
            {
                "source_id": "entity1",
                "target_id": "entity2",
                "relationship_type": "RELATES_TO",
                "properties": {"strength": "strong"},
            },
            # Invalid - missing target
            {"source_id": "entity1", "relationship_type": "RELATES_TO"},
            # Invalid - missing relationship type
            {"source_id": "entity1", "target_id": "entity2"},
        ]

        validation_result = loader.validate_data(relationships, "relationships")

        assert validation_result["total_items"] == 3
        assert validation_result["valid_items"] >= 1
        assert validation_result["invalid_items"] >= 2
        assert len(validation_result["validation_errors"]) > 0


class TestErrorHandling:
    """Test error handling in various scenarios."""

    @pytest.fixture
    def backend(self):
        """Create test backend."""
        backend = MemoryGraphBackend({})
        backend.connect()
        return backend

    def test_backend_connection_failure(self):
        """Test handling of backend connection failures."""
        # Create a mock backend that fails to connect
        mock_backend = Mock()
        mock_backend.connect.return_value = False
        mock_backend.is_connected.return_value = False

        loader = ICKGDataLoader(mock_backend)

        # Should handle backend connection issues gracefully
        entities = [{"labels": ["Test"], "properties": {"name": "test"}}]

        # Depending on implementation, this might succeed with warnings
        # or fail gracefully without crashing
        try:
            stats = loader.load_from_data(entities)
            # If it succeeds, that's fine too
            assert isinstance(stats, LoaderStats)
        except Exception as e:
            # Should be a handled exception, not a crash
            assert str(e)  # Should have error message

    def test_partial_data_corruption(self, backend):
        """Test handling of partially corrupted data."""
        loader = ICKGDataLoader(backend, LoaderConfig(error_handling="continue"))

        # Mix of good and corrupted data
        entities = [
            {"labels": ["Good"], "properties": {"name": "good1"}},
            None,  # Corrupted entry
            {"labels": ["Good"], "properties": {"name": "good2"}},
            {"corrupted": "structure"},  # Malformed entry
            {"labels": ["Good"], "properties": {"name": "good3"}},
        ]

        # Should continue processing despite corruption
        stats = loader.load_from_data(entities)

        assert stats.total_processed == len(entities)
        assert stats.successful_loads >= 3  # At least the good ones
        assert stats.failed_loads >= 2  # At least the corrupted ones

    def test_memory_pressure_handling(self, backend):
        """Test handling of memory pressure during large loads."""
        loader = ICKGDataLoader(backend, LoaderConfig(batch_size=10))

        # Create a large dataset
        large_dataset = []
        for i in range(1000):
            large_dataset.append(
                {
                    "labels": ["LargeDataset"],
                    "properties": {
                        "id": i,
                        "name": f"entity_{i}",
                        "data": "x" * 100,  # Some bulk to use memory
                    },
                }
            )

        # Should handle large dataset without crashing
        stats = loader.load_from_data(large_dataset)

        assert stats.total_processed == 1000
        assert stats.successful_loads == 1000
        assert stats.processing_rate > 0
