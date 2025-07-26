"""
IC-KG Data Loader

Efficient data loading, validation, and batch processing system
for the Integrated Chemistry Knowledge Graph.
"""

import csv
import json
import logging
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from .graph_database import BulkLoadConfig, GraphDatabaseBackend
from .ontology import ChemicalEntity, ChemicalRelationship
from .schema import EntityType, ICKGSchema, RelationshipType

logger = logging.getLogger(__name__)


@dataclass
class LoaderConfig:
    """Configuration for data loading operations."""

    batch_size: int = 1000
    max_workers: int = 4
    validation_enabled: bool = True
    create_missing_entities: bool = True
    update_existing: bool = False
    progress_reporting: bool = True
    error_handling: str = "continue"  # "continue", "stop", "skip_batch"
    checkpoint_interval: int = 10000
    compression: bool = False
    streaming: bool = False


@dataclass
class LoaderStats:
    """Statistics for data loading operations."""

    total_processed: int = 0
    successful_loads: int = 0
    failed_loads: int = 0
    validation_errors: int = 0
    duplicate_entities: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    processing_rate: float = 0.0
    errors: List[str] = field(default_factory=list)


class ICKGDataLoader:
    """Main data loader for IC-KG with multi-format support."""

    def __init__(self, backend: GraphDatabaseBackend, config: LoaderConfig = None):
        self.backend = backend
        self.config = config or LoaderConfig()
        self.schema = ICKGSchema()
        self.stats = LoaderStats()
        self._checkpoint_data = {}
        self._progress_callbacks = []

    def add_progress_callback(self, callback: Callable[[float, str], None]):
        """Add progress reporting callback."""
        self._progress_callbacks.append(callback)

    def _report_progress(self, progress: float, message: str):
        """Report progress to all registered callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(progress, message)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def load_from_file(
        self, file_path: Union[str, Path], file_format: str = None
    ) -> LoaderStats:
        """Load data from file with format auto-detection."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Auto-detect format if not specified
        if file_format is None:
            file_format = self._detect_file_format(file_path)

        logger.info(f"Loading data from {file_path} (format: {file_format})")

        self.stats = LoaderStats()
        self.stats.start_time = datetime.now()

        try:
            if file_format == "json":
                return self._load_from_json(file_path)
            elif file_format == "csv":
                return self._load_from_csv(file_path)
            elif file_format == "jsonl":
                return self._load_from_jsonlines(file_path)
            elif file_format == "rdf":
                return self._load_from_rdf(file_path)
            elif file_format == "pickle":
                return self._load_from_pickle(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

        finally:
            self.stats.end_time = datetime.now()
            if self.stats.start_time:
                duration = (self.stats.end_time - self.stats.start_time).total_seconds()
                self.stats.processing_rate = (
                    self.stats.total_processed / duration if duration > 0 else 0
                )

    def load_from_data(
        self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]] = None
    ) -> LoaderStats:
        """Load data from in-memory structures."""
        logger.info(
            f"Loading {len(entities)} entities and {len(relationships or [])} relationships"
        )

        self.stats = LoaderStats()
        self.stats.start_time = datetime.now()

        try:
            # Load entities first
            entity_stats = self._batch_load_entities(entities)

            # Load relationships if provided
            if relationships:
                rel_stats = self._batch_load_relationships(relationships)
                self._merge_stats(entity_stats, rel_stats)
            else:
                self.stats = entity_stats

            return self.stats

        finally:
            self.stats.end_time = datetime.now()
            if self.stats.start_time:
                duration = (self.stats.end_time - self.stats.start_time).total_seconds()
                self.stats.processing_rate = (
                    self.stats.total_processed / duration if duration > 0 else 0
                )

    def stream_load(
        self, data_iterator: Iterator[Dict[str, Any]], data_type: str = "mixed"
    ) -> Iterator[LoaderStats]:
        """Stream data loading with progress updates."""
        self.stats = LoaderStats()
        self.stats.start_time = datetime.now()

        batch_entities = []
        batch_relationships = []

        for i, item in enumerate(data_iterator):
            try:
                if data_type == "entities" or (
                    data_type == "mixed" and self._is_entity(item)
                ):
                    batch_entities.append(item)
                elif data_type == "relationships" or (
                    data_type == "mixed" and self._is_relationship(item)
                ):
                    batch_relationships.append(item)

                # Process batch when full
                if (
                    len(batch_entities) + len(batch_relationships)
                    >= self.config.batch_size
                ):
                    batch_stats = self._process_mixed_batch(
                        batch_entities, batch_relationships
                    )
                    self._merge_stats(self.stats, batch_stats)

                    batch_entities.clear()
                    batch_relationships.clear()

                    # Yield progress
                    yield self.stats

                # Checkpoint
                if i % self.config.checkpoint_interval == 0 and i > 0:
                    self._create_checkpoint()

            except Exception as e:
                self.stats.failed_loads += 1
                self.stats.errors.append(f"Item {i}: {str(e)}")

                if self.config.error_handling == "stop":
                    break

        # Process remaining items
        if batch_entities or batch_relationships:
            batch_stats = self._process_mixed_batch(batch_entities, batch_relationships)
            self._merge_stats(self.stats, batch_stats)

        self.stats.end_time = datetime.now()
        yield self.stats

    def validate_data(
        self, data: List[Dict[str, Any]], data_type: str = "entities"
    ) -> Dict[str, Any]:
        """Validate data against IC-KG schema."""
        validation_results = {
            "total_items": len(data),
            "valid_items": 0,
            "invalid_items": 0,
            "validation_errors": [],
            "schema_violations": [],
            "warnings": [],
        }

        for i, item in enumerate(data):
            try:
                if data_type == "entities":
                    is_valid, errors = self._validate_entity(item)
                elif data_type == "relationships":
                    is_valid, errors = self._validate_relationship(item)
                else:
                    is_valid, errors = False, ["Unknown data type"]

                if is_valid:
                    validation_results["valid_items"] += 1
                else:
                    validation_results["invalid_items"] += 1
                    validation_results["validation_errors"].extend(
                        [f"Item {i}: {error}" for error in errors]
                    )

            except Exception as e:
                validation_results["invalid_items"] += 1
                validation_results["validation_errors"].append(
                    f"Item {i}: Validation failed - {str(e)}"
                )

        return validation_results

    def create_checkpoint(self, checkpoint_id: str = None) -> str:
        """Create a data loading checkpoint."""
        if checkpoint_id is None:
            checkpoint_id = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        checkpoint_data = {
            "checkpoint_id": checkpoint_id,
            "timestamp": datetime.now().isoformat(),
            "stats": {
                "total_processed": self.stats.total_processed,
                "successful_loads": self.stats.successful_loads,
                "failed_loads": self.stats.failed_loads,
            },
            "config": self.config.__dict__,
            "backend_stats": self.backend.get_statistics(),
        }

        self._checkpoint_data[checkpoint_id] = checkpoint_data
        logger.info(f"Created checkpoint: {checkpoint_id}")

        return checkpoint_id

    def restore_from_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore from a checkpoint."""
        if checkpoint_id not in self._checkpoint_data:
            logger.error(f"Checkpoint not found: {checkpoint_id}")
            return False

        checkpoint = self._checkpoint_data[checkpoint_id]

        # Restore stats
        stats_data = checkpoint["stats"]
        self.stats.total_processed = stats_data["total_processed"]
        self.stats.successful_loads = stats_data["successful_loads"]
        self.stats.failed_loads = stats_data["failed_loads"]

        logger.info(f"Restored from checkpoint: {checkpoint_id}")
        return True

    def get_loading_statistics(self) -> Dict[str, Any]:
        """Get comprehensive loading statistics."""
        duration = 0
        if self.stats.start_time:
            end_time = self.stats.end_time or datetime.now()
            duration = (end_time - self.stats.start_time).total_seconds()

        return {
            "processing": {
                "total_processed": self.stats.total_processed,
                "successful_loads": self.stats.successful_loads,
                "failed_loads": self.stats.failed_loads,
                "validation_errors": self.stats.validation_errors,
                "duplicate_entities": self.stats.duplicate_entities,
                "success_rate": self.stats.successful_loads
                / max(self.stats.total_processed, 1),
            },
            "performance": {
                "processing_rate": self.stats.processing_rate,
                "duration_seconds": duration,
                "batch_size": self.config.batch_size,
                "max_workers": self.config.max_workers,
            },
            "configuration": {
                "validation_enabled": self.config.validation_enabled,
                "create_missing_entities": self.config.create_missing_entities,
                "update_existing": self.config.update_existing,
                "streaming": self.config.streaming,
            },
            "errors": self.stats.errors[-10:],  # Last 10 errors
            "checkpoints": list(self._checkpoint_data.keys()),
        }

    # Private methods

    def _detect_file_format(self, file_path: Path) -> str:
        """Auto-detect file format from extension and content."""
        extension = file_path.suffix.lower()

        format_map = {
            ".json": "json",
            ".jsonl": "jsonl",
            ".csv": "csv",
            ".tsv": "csv",
            ".rdf": "rdf",
            ".ttl": "rdf",
            ".pkl": "pickle",
            ".pickle": "pickle",
        }

        return format_map.get(extension, "json")  # Default to JSON

    def _load_from_json(self, file_path: Path) -> LoaderStats:
        """Load data from JSON file."""
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, dict):
            entities = data.get("entities", [])
            relationships = data.get("relationships", [])
        elif isinstance(data, list):
            # Assume all items are entities
            entities = data
            relationships = []
        else:
            raise ValueError("Invalid JSON structure")

        return self.load_from_data(entities, relationships)

    def _load_from_jsonlines(self, file_path: Path) -> LoaderStats:
        """Load data from JSON Lines file."""
        entities = []
        relationships = []

        with open(file_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())

                    if self._is_entity(item):
                        entities.append(item)
                    elif self._is_relationship(item):
                        relationships.append(item)

                    # Process in batches for large files
                    if (
                        len(entities) + len(relationships)
                        >= self.config.batch_size * 10
                    ):
                        batch_stats = self.load_from_data(entities, relationships)
                        self._merge_stats(self.stats, batch_stats)
                        entities.clear()
                        relationships.clear()

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    self.stats.failed_loads += 1

        # Process remaining items
        if entities or relationships:
            batch_stats = self.load_from_data(entities, relationships)
            self._merge_stats(self.stats, batch_stats)

        return self.stats

    def _load_from_csv(self, file_path: Path) -> LoaderStats:
        """Load data from CSV file."""
        entities = []

        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row_num, row in enumerate(reader, 1):
                try:
                    # Convert CSV row to entity format
                    entity = self._csv_row_to_entity(row)
                    entities.append(entity)

                    # Process in batches
                    if len(entities) >= self.config.batch_size:
                        batch_stats = self._batch_load_entities(entities)
                        self._merge_stats(self.stats, batch_stats)
                        entities.clear()

                except Exception as e:
                    logger.warning(f"Error processing CSV row {row_num}: {e}")
                    self.stats.failed_loads += 1

        # Process remaining entities
        if entities:
            batch_stats = self._batch_load_entities(entities)
            self._merge_stats(self.stats, batch_stats)

        return self.stats

    def _load_from_rdf(self, file_path: Path) -> LoaderStats:
        """Load data from RDF file."""
        try:
            import rdflib

            graph = rdflib.Graph()
            graph.parse(file_path)

            entities = []
            relationships = []

            # Convert RDF triples to entities and relationships
            subjects = set(graph.subjects())

            for subject in subjects:
                entity = self._rdf_subject_to_entity(graph, subject)
                if entity:
                    entities.append(entity)

            # Extract relationships from object properties
            for s, p, o in graph:
                if isinstance(o, rdflib.URIRef):  # Object property
                    relationship = {
                        "source_id": str(s),
                        "target_id": str(o),
                        "relationship_type": str(p).split("/")[-1].split("#")[-1],
                        "properties": {},
                    }
                    relationships.append(relationship)

            return self.load_from_data(entities, relationships)

        except ImportError:
            raise ImportError(
                "RDFLib required for RDF loading. Install with: pip install rdflib"
            )

    def _load_from_pickle(self, file_path: Path) -> LoaderStats:
        """Load data from pickle file."""
        import pickle

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            entities = data.get("entities", [])
            relationships = data.get("relationships", [])
        else:
            entities = data
            relationships = []

        return self.load_from_data(entities, relationships)

    def _batch_load_entities(self, entities: List[Dict[str, Any]]) -> LoaderStats:
        """Load entities in batches with validation."""
        stats = LoaderStats()
        stats.start_time = datetime.now()

        # Validate entities if enabled
        if self.config.validation_enabled:
            validation_results = self.validate_data(entities, "entities")
            stats.validation_errors = validation_results["invalid_items"]

        # Process in parallel if configured
        if self.config.max_workers > 1:
            stats = self._parallel_load_entities(entities)
        else:
            stats = self._sequential_load_entities(entities)

        stats.end_time = datetime.now()
        return stats

    def _batch_load_relationships(
        self, relationships: List[Dict[str, Any]]
    ) -> LoaderStats:
        """Load relationships in batches with validation."""
        stats = LoaderStats()
        stats.start_time = datetime.now()

        # Validate relationships if enabled
        if self.config.validation_enabled:
            validation_results = self.validate_data(relationships, "relationships")
            stats.validation_errors = validation_results["invalid_items"]

        # Process in parallel if configured
        if self.config.max_workers > 1:
            stats = self._parallel_load_relationships(relationships)
        else:
            stats = self._sequential_load_relationships(relationships)

        stats.end_time = datetime.now()
        return stats

    def _sequential_load_entities(self, entities: List[Dict[str, Any]]) -> LoaderStats:
        """Load entities sequentially."""
        stats = LoaderStats()

        # Use backend bulk loading if available
        bulk_config = BulkLoadConfig(
            batch_size=self.config.batch_size,
            validation=self.config.validation_enabled,
            create_missing_nodes=self.config.create_missing_entities,
        )

        result = self.backend.bulk_load_nodes(entities, bulk_config)

        stats.total_processed = result["total_nodes"]
        stats.successful_loads = result["created_nodes"]
        stats.failed_loads = result["failed_nodes"]
        stats.errors.extend(result.get("errors", []))

        return stats

    def _sequential_load_relationships(
        self, relationships: List[Dict[str, Any]]
    ) -> LoaderStats:
        """Load relationships sequentially."""
        stats = LoaderStats()

        # Use backend bulk loading if available
        bulk_config = BulkLoadConfig(
            batch_size=self.config.batch_size, validation=self.config.validation_enabled
        )

        result = self.backend.bulk_load_relationships(relationships, bulk_config)

        stats.total_processed = result["total_relationships"]
        stats.successful_loads = result["created_relationships"]
        stats.failed_loads = result["failed_relationships"]
        stats.errors.extend(result.get("errors", []))

        return stats

    def _parallel_load_entities(self, entities: List[Dict[str, Any]]) -> LoaderStats:
        """Load entities in parallel."""
        # Split entities into chunks for parallel processing
        chunk_size = max(1, len(entities) // self.config.max_workers)
        chunks = [
            entities[i : i + chunk_size] for i in range(0, len(entities), chunk_size)
        ]

        combined_stats = LoaderStats()

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [
                executor.submit(self._sequential_load_entities, chunk)
                for chunk in chunks
            ]

            for future in futures:
                try:
                    chunk_stats = future.result()
                    self._merge_stats(combined_stats, chunk_stats)
                except Exception as e:
                    combined_stats.failed_loads += len(chunks[0])  # Approximate
                    combined_stats.errors.append(f"Parallel processing error: {str(e)}")

        return combined_stats

    def _parallel_load_relationships(
        self, relationships: List[Dict[str, Any]]
    ) -> LoaderStats:
        """Load relationships in parallel."""
        # Split relationships into chunks for parallel processing
        chunk_size = max(1, len(relationships) // self.config.max_workers)
        chunks = [
            relationships[i : i + chunk_size]
            for i in range(0, len(relationships), chunk_size)
        ]

        combined_stats = LoaderStats()

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [
                executor.submit(self._sequential_load_relationships, chunk)
                for chunk in chunks
            ]

            for future in futures:
                try:
                    chunk_stats = future.result()
                    self._merge_stats(combined_stats, chunk_stats)
                except Exception as e:
                    combined_stats.failed_loads += len(chunks[0])  # Approximate
                    combined_stats.errors.append(f"Parallel processing error: {str(e)}")

        return combined_stats

    def _process_mixed_batch(
        self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]
    ) -> LoaderStats:
        """Process a mixed batch of entities and relationships."""
        combined_stats = LoaderStats()

        if entities:
            entity_stats = self._batch_load_entities(entities)
            self._merge_stats(combined_stats, entity_stats)

        if relationships:
            rel_stats = self._batch_load_relationships(relationships)
            self._merge_stats(combined_stats, rel_stats)

        return combined_stats

    def _validate_entity(self, entity: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate a single entity against schema."""
        errors = []

        # Check required fields
        if "_id" not in entity and "id" not in entity:
            errors.append("Missing entity ID")

        if "labels" not in entity and "entity_type" not in entity:
            errors.append("Missing entity type/labels")

        # Validate entity type if present
        entity_type = entity.get("entity_type") or (
            entity.get("labels", [{}])[0] if entity.get("labels") else None
        )
        if entity_type:
            try:
                EntityType(entity_type)
            except ValueError:
                errors.append(f"Invalid entity type: {entity_type}")

        # Additional schema validation could be added here

        return len(errors) == 0, errors

    def _validate_relationship(
        self, relationship: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """Validate a single relationship against schema."""
        errors = []

        # Check required fields
        required_fields = ["source_id", "target_id", "relationship_type"]
        for field in required_fields:
            if field not in relationship:
                errors.append(f"Missing required field: {field}")

        # Validate relationship type
        rel_type = relationship.get("relationship_type")
        if rel_type:
            try:
                RelationshipType(rel_type)
            except ValueError:
                errors.append(f"Invalid relationship type: {rel_type}")

        return len(errors) == 0, errors

    def _is_entity(self, item: Dict[str, Any]) -> bool:
        """Check if item is an entity."""
        return (
            "labels" in item
            or "entity_type" in item
            or ("source_id" not in item and "target_id" not in item)
        )

    def _is_relationship(self, item: Dict[str, Any]) -> bool:
        """Check if item is a relationship."""
        return "source_id" in item and "target_id" in item

    def _merge_stats(self, target: LoaderStats, source: LoaderStats):
        """Merge source stats into target stats."""
        target.total_processed += source.total_processed
        target.successful_loads += source.successful_loads
        target.failed_loads += source.failed_loads
        target.validation_errors += source.validation_errors
        target.duplicate_entities += source.duplicate_entities
        target.errors.extend(source.errors)

    def _csv_row_to_entity(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Convert CSV row to entity format."""
        # Simple conversion - could be made more sophisticated
        entity = {"labels": [row.get("entity_type", "Entity")], "properties": {}}

        # Move special fields
        if "id" in row:
            entity["_id"] = row.pop("id")
        if "entity_type" in row:
            row.pop("entity_type")

        # Add remaining fields as properties
        entity["properties"].update(row)

        return entity

    def _rdf_subject_to_entity(self, graph, subject) -> Optional[Dict[str, Any]]:
        """Convert RDF subject to entity format."""
        import rdflib

        entity = {"_id": str(subject), "labels": [], "properties": {}}

        # Get types
        for obj in graph.objects(subject, rdflib.RDF.type):
            type_name = str(obj).split("/")[-1].split("#")[-1]
            entity["labels"].append(type_name)

        # Get properties (data properties)
        for pred, obj in graph.predicate_objects(subject):
            if isinstance(obj, rdflib.Literal):
                prop_name = str(pred).split("/")[-1].split("#")[-1]
                entity["properties"][prop_name] = str(obj)

        return entity if entity["labels"] or entity["properties"] else None

    def _create_checkpoint(self):
        """Create an automatic checkpoint."""
        checkpoint_id = self.create_checkpoint()
        logger.debug(f"Auto-checkpoint created: {checkpoint_id}")


class BulkLoader:
    """Specialized bulk loader for high-performance data loading."""

    def __init__(self, backend: GraphDatabaseBackend, chunk_size: int = 10000):
        self.backend = backend
        self.chunk_size = chunk_size
        self.stats = LoaderStats()

    def load_entities_from_iterator(
        self, entity_iterator: Iterator[Dict[str, Any]]
    ) -> LoaderStats:
        """Load entities from an iterator with chunked processing."""
        self.stats = LoaderStats()
        self.stats.start_time = datetime.now()

        chunk = []

        for entity in entity_iterator:
            chunk.append(entity)

            if len(chunk) >= self.chunk_size:
                self._process_entity_chunk(chunk)
                chunk.clear()

        # Process remaining entities
        if chunk:
            self._process_entity_chunk(chunk)

        self.stats.end_time = datetime.now()
        return self.stats

    def _process_entity_chunk(self, chunk: List[Dict[str, Any]]):
        """Process a chunk of entities."""
        try:
            result = self.backend.bulk_load_nodes(chunk)

            self.stats.total_processed += result["total_nodes"]
            self.stats.successful_loads += result["created_nodes"]
            self.stats.failed_loads += result["failed_nodes"]
            self.stats.errors.extend(result.get("errors", []))

        except Exception as e:
            self.stats.failed_loads += len(chunk)
            self.stats.errors.append(f"Chunk processing failed: {str(e)}")


class StreamingLoader:
    """Streaming loader for real-time data ingestion."""

    def __init__(self, backend: GraphDatabaseBackend, buffer_size: int = 1000):
        self.backend = backend
        self.buffer_size = buffer_size
        self.entity_buffer = []
        self.relationship_buffer = []
        self.stats = LoaderStats()
        self._lock = threading.Lock()

    def add_entity(self, entity: Dict[str, Any]):
        """Add entity to streaming buffer."""
        with self._lock:
            self.entity_buffer.append(entity)

            if len(self.entity_buffer) >= self.buffer_size:
                self._flush_entity_buffer()

    def add_relationship(self, relationship: Dict[str, Any]):
        """Add relationship to streaming buffer."""
        with self._lock:
            self.relationship_buffer.append(relationship)

            if len(self.relationship_buffer) >= self.buffer_size:
                self._flush_relationship_buffer()

    def flush(self):
        """Flush all buffers."""
        with self._lock:
            self._flush_entity_buffer()
            self._flush_relationship_buffer()

    def _flush_entity_buffer(self):
        """Flush entity buffer to backend."""
        if not self.entity_buffer:
            return

        try:
            result = self.backend.bulk_load_nodes(self.entity_buffer)

            self.stats.total_processed += result["total_nodes"]
            self.stats.successful_loads += result["created_nodes"]
            self.stats.failed_loads += result["failed_nodes"]

            self.entity_buffer.clear()

        except Exception as e:
            self.stats.failed_loads += len(self.entity_buffer)
            self.stats.errors.append(f"Entity buffer flush failed: {str(e)}")
            self.entity_buffer.clear()

    def _flush_relationship_buffer(self):
        """Flush relationship buffer to backend."""
        if not self.relationship_buffer:
            return

        try:
            result = self.backend.bulk_load_relationships(self.relationship_buffer)

            self.stats.total_processed += result["total_relationships"]
            self.stats.successful_loads += result["created_relationships"]
            self.stats.failed_loads += result["failed_relationships"]

            self.relationship_buffer.clear()

        except Exception as e:
            self.stats.failed_loads += len(self.relationship_buffer)
            self.stats.errors.append(f"Relationship buffer flush failed: {str(e)}")
            self.relationship_buffer.clear()


class ValidationLoader:
    """Loader with comprehensive validation and error reporting."""

    def __init__(self, backend: GraphDatabaseBackend, schema: ICKGSchema):
        self.backend = backend
        self.schema = schema
        self.validation_stats = {
            "total_validated": 0,
            "schema_violations": 0,
            "data_quality_issues": 0,
            "validation_errors": [],
        }

    def load_with_validation(
        self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Load data with comprehensive validation."""
        # Validate entities
        entity_validation = self._comprehensive_entity_validation(entities)

        # Validate relationships
        relationship_validation = {}
        if relationships:
            relationship_validation = self._comprehensive_relationship_validation(
                relationships
            )

        # Load valid data
        valid_entities = entity_validation.get("valid_entities", [])
        valid_relationships = relationship_validation.get("valid_relationships", [])

        loader = ICKGDataLoader(self.backend)
        load_stats = loader.load_from_data(valid_entities, valid_relationships)

        return {
            "load_stats": load_stats,
            "entity_validation": entity_validation,
            "relationship_validation": relationship_validation,
            "validation_stats": self.validation_stats,
        }

    def _comprehensive_entity_validation(
        self, entities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform comprehensive entity validation."""
        results = {
            "total_entities": len(entities),
            "valid_entities": [],
            "invalid_entities": [],
            "validation_errors": [],
            "quality_warnings": [],
        }

        for i, entity in enumerate(entities):
            validation_result = self.schema.validate_entity(entity)

            if validation_result.is_valid:
                results["valid_entities"].append(entity)
            else:
                results["invalid_entities"].append(entity)
                results["validation_errors"].extend(
                    [f"Entity {i}: {error}" for error in validation_result.errors]
                )

        self.validation_stats["total_validated"] += len(entities)
        self.validation_stats["schema_violations"] += len(results["invalid_entities"])

        return results

    def _comprehensive_relationship_validation(
        self, relationships: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform comprehensive relationship validation."""
        results = {
            "total_relationships": len(relationships),
            "valid_relationships": [],
            "invalid_relationships": [],
            "validation_errors": [],
            "quality_warnings": [],
        }

        for i, relationship in enumerate(relationships):
            validation_result = self.schema.validate_relationship(relationship)

            if validation_result.is_valid:
                results["valid_relationships"].append(relationship)
            else:
                results["invalid_relationships"].append(relationship)
                results["validation_errors"].extend(
                    [f"Relationship {i}: {error}" for error in validation_result.errors]
                )

        return results
