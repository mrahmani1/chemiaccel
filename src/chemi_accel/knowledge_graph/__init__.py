"""
ChemiAccel Knowledge Graph Module

Comprehensive graph database integration and management system for the
Integrated Chemistry Knowledge Graph (IC-KG) with multi-backend support,
advanced semantic querying, reasoning, and efficient data loading.

Main components:
- Graph Database: Multi-backend graph storage (Neo4j, Jena, GPU, Memory)
- Semantic Query Engine: SPARQL, Cypher, Natural Language support with reasoning
- Query Engine: Advanced query optimization and caching
- Reasoning Engine: Rule-based inference and fact derivation
- NLP Processor: Natural language understanding and entity extraction
- Data Loading: Efficient bulk loading and streaming
- Ontology: Chemical entity type definitions and hierarchies
- Schema: Formal structure for entities and relationships
"""

# Data Loading
from .data_loader import (
    BulkLoader,
    ICKGDataLoader,
    LoaderConfig,
    LoaderStats,
    StreamingLoader,
    ValidationLoader,
)
from .gpu_graph_backend import GPUGraphBackend

# Graph Database Integration
from .graph_database import (
    BulkLoadConfig,
    GraphDatabaseBackend,
    GraphDatabaseManager,
    GraphDatabaseType,
    GraphQuery,
    GraphQueryResult,
    IndexConfig,
    get_graph_database,
)
from .jena_backend import JenaBackend
from .memory_backend import MemoryGraphBackend
from .neo4j_backend import Neo4jBackend

# NLP Processor
from .nlp_processor import (
    AdvancedNLPProcessor,
    ChemicalEntityExtractor,
    EntityMention,
    ExtractedEntity,
    IntentClassifier,
    ProcessedQuery,
    QueryContext,
    QueryIntent,
    QueryRewriter,
)

# Core Ontology and Schema
from .ontology import (
    ChemicalEntity,
    ChemicalOntology,
    ChemicalProperty,
    ChemicalRelationship,
    EntityType,
    PropertyType,
    RelationshipType,
)

# Query Engine
from .query_engine import (
    ChemicalQuery,
    GraphQueryEngine,
    QueryBuilder,
    QueryCache,
    QueryOptimizer,
    QueryPlan,
    QueryStats,
    SemanticQuery,
)

# Reasoning Rules
from .reasoning_rules import (
    AdvancedReasoningRule,
    ChemicalReasoningRuleLibrary,
    ConfidenceCalculator,
    ConfidenceModel,
    FactValidator,
    InferenceEngine,
    RuleType,
)
from .relationships import (
    ChemicalRelationshipManager,
    PreDefinedRelationships,
    RelationshipConstraints,
)
from .schema import (
    EntitySchema,
    ICKGSchema,
    KnowledgeGraphSchema,
    PropertySchema,
    RelationshipSchema,
    SchemaValidator,
)

# Semantic Parsers
from .semantic_parsers import (
    AdvancedSPARQLParser,
    ChemicalDSLParser,
    QueryOptimizationEngine,
    SemanticCypherExtensions,
)

# Semantic Query Engine
from .semantic_query_engine import (
    ChemicalEntityRecognizer,
    CypherSemanticProcessor,
    InferredFact,
    NaturalLanguageProcessor,
    QueryLanguage,
    ReasoningEngine,
    ReasoningRule,
    ReasoningStrategy,
    SemanticQueryContext,
    SemanticQueryEngine,
    SPARQLProcessor,
)

# Validation and Anomaly Detection
from .validation import (
    AnomalyType,
    DataQualityMetrics,
    DetectedAnomaly,
    ExperimentSuggestionEngine,
    KnowledgeGraphValidator,
    SeverityLevel,
    ValidationReport,
)

__all__ = [
    # Graph Database backends
    "get_graph_database",
    "GraphDatabaseBackend",
    "GraphDatabaseType",
    "GraphQuery",
    "GraphQueryResult",
    "IndexConfig",
    "BulkLoadConfig",
    "GraphDatabaseManager",
    "Neo4jBackend",
    "JenaBackend",
    "GPUGraphBackend",
    "MemoryGraphBackend",
    # Query Engine
    "GraphQueryEngine",
    "QueryBuilder",
    "ChemicalQuery",
    "SemanticQuery",
    "QueryCache",
    "QueryOptimizer",
    "QueryPlan",
    "QueryStats",
    # Data Loading
    "ICKGDataLoader",
    "LoaderConfig",
    "LoaderStats",
    "BulkLoader",
    "StreamingLoader",
    "ValidationLoader",
    # Core ontology classes
    "ChemicalOntology",
    "EntityType",
    "PropertyType",
    "RelationshipType",
    "ChemicalEntity",
    "ChemicalProperty",
    "ChemicalRelationship",
    # Schema management
    "KnowledgeGraphSchema",
    "SchemaValidator",
    "EntitySchema",
    "RelationshipSchema",
    "PropertySchema",
    "ICKGSchema",
    # Relationship management
    "ChemicalRelationshipManager",
    "RelationshipConstraints",
    "PreDefinedRelationships",
    # Semantic Query Engine
    "SemanticQueryEngine",
    "SemanticQueryContext",
    "QueryLanguage",
    "ReasoningStrategy",
    "InferredFact",
    "ReasoningRule",
    "SPARQLProcessor",
    "CypherSemanticProcessor",
    "NaturalLanguageProcessor",
    "ChemicalEntityRecognizer",
    "ReasoningEngine",
    # Semantic Parsers
    "ChemicalDSLParser",
    "AdvancedSPARQLParser",
    "SemanticCypherExtensions",
    "QueryOptimizationEngine",
    # Reasoning Rules
    "ChemicalReasoningRuleLibrary",
    "InferenceEngine",
    "AdvancedReasoningRule",
    "RuleType",
    "ConfidenceModel",
    "ConfidenceCalculator",
    "FactValidator",
    # NLP Processor
    "AdvancedNLPProcessor",
    "ChemicalEntityExtractor",
    "IntentClassifier",
    "QueryRewriter",
    "QueryIntent",
    "ExtractedEntity",
    "EntityMention",
    "QueryContext",
    "ProcessedQuery",
    # Validation and Anomaly Detection
    "KnowledgeGraphValidator",
    "DetectedAnomaly",
    "AnomalyType",
    "SeverityLevel",
    "ValidationReport",
    "DataQualityMetrics",
    "ExperimentSuggestionEngine",
]

# Version information
__version__ = "2.0.0"
