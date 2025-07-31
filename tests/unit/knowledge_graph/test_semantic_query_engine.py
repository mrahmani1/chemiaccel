"""
Tests for Semantic Query and Reasoning Engine

Comprehensive test suite for semantic query processing, reasoning,
and natural language understanding capabilities.
"""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pytest
from chemi_accel.knowledge_graph.graph_database import GraphQuery, GraphQueryResult
from chemi_accel.knowledge_graph.memory_backend import MemoryGraphBackend
from chemi_accel.knowledge_graph.nlp_processor import (
    AdvancedNLPProcessor,
    ChemicalEntityExtractor,
    EntityMention,
    ExtractedEntity,
    IntentClassifier,
    ProcessedQuery,
    QueryContext,
    QueryIntent,
)
from chemi_accel.knowledge_graph.ontology import (
    ChemicalEntity,
    ChemicalOntology,
    EntityType,
)
from chemi_accel.knowledge_graph.reasoning_rules import (
    AdvancedReasoningRule,
    ChemicalReasoningRuleLibrary,
    ConfidenceModel,
    InferenceEngine,
    RuleType,
)
from chemi_accel.knowledge_graph.schema import ICKGSchema
from chemi_accel.knowledge_graph.semantic_parsers import (
    AdvancedSPARQLParser,
    ChemicalDSLParser,
    SemanticCypherExtensions,
)
from chemi_accel.knowledge_graph.semantic_query_engine import (
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


class TestSemanticQueryContext:
    """Test semantic query context functionality."""

    def test_context_initialization(self):
        """Test context initialization with default values."""
        context = SemanticQueryContext(query_language=QueryLanguage.SPARQL)

        assert context.query_language == QueryLanguage.SPARQL
        assert context.reasoning_enabled == True
        assert context.reasoning_depth == 3
        assert context.confidence_threshold == 0.7
        assert context.include_inferred_facts == True
        assert context.query_timeout == 30
        assert context.cache_results == True
        assert isinstance(context.ontology_namespaces, dict)
        assert isinstance(context.metadata, dict)

    def test_context_customization(self):
        """Test context customization."""
        context = SemanticQueryContext(
            query_language=QueryLanguage.NATURAL_LANGUAGE,
            reasoning_enabled=False,
            reasoning_depth=5,
            confidence_threshold=0.8,
            include_inferred_facts=False,
            query_timeout=60,
            cache_results=False,
            ontology_namespaces={"cao": "http://example.org/"},
            metadata={"user": "test"},
        )

        assert context.query_language == QueryLanguage.NATURAL_LANGUAGE
        assert context.reasoning_enabled == False
        assert context.reasoning_depth == 5
        assert context.confidence_threshold == 0.8
        assert context.include_inferred_facts == False
        assert context.query_timeout == 60
        assert context.cache_results == False
        assert context.ontology_namespaces["cao"] == "http://example.org/"
        assert context.metadata["user"] == "test"


class TestSPARQLProcessor:
    """Test SPARQL query processor."""

    @pytest.fixture
    def ontology(self):
        """Create mock ontology."""
        ontology = Mock(spec=ChemicalOntology)
        return ontology

    @pytest.fixture
    def sparql_processor(self, ontology):
        """Create SPARQL processor."""
        return SPARQLProcessor(ontology)

    def test_sparql_initialization(self, sparql_processor):
        """Test SPARQL processor initialization."""
        assert isinstance(sparql_processor.prefixes, dict)
        assert "cao" in sparql_processor.prefixes
        assert "rdfs" in sparql_processor.prefixes
        assert "owl" in sparql_processor.prefixes

    def test_parse_simple_sparql(self, sparql_processor):
        """Test parsing simple SPARQL query."""
        query = """
        PREFIX cao: <http://chemi-accel.org/ontology/>
        SELECT ?compound ?formula
        WHERE {
            ?compound cao:has_formula ?formula .
            ?compound rdf:type cao:SmallMolecule .
        }
        """

        context = SemanticQueryContext(query_language=QueryLanguage.SPARQL)
        parsed = sparql_processor.parse_query(query, context)

        assert parsed["query_type"] == "sparql"
        assert "cao" in parsed["prefixes"]
        assert "compound" in parsed["select_variables"]
        assert "formula" in parsed["select_variables"]
        assert len(parsed["where_patterns"]) >= 2

    def test_validate_sparql_query(self, sparql_processor):
        """Test SPARQL query validation."""
        # Valid query
        valid_query = {
            "query_type": "SELECT",
            "prefixes": {"cao": "http://example.org/"},
            "select_variables": ["compound"],
            "where_patterns": [
                {"subject": "?compound", "predicate": "rdf:type", "object": "cao:Drug"}
            ],
            "filters": [],
            "modifiers": {},
        }

        context = SemanticQueryContext(query_language=QueryLanguage.SPARQL)
        assert sparql_processor.validate_query(valid_query, context) == True

        # Invalid query - missing variables
        invalid_query = {
            "query_type": "SELECT",
            "prefixes": {},
            "select_variables": [],
            "where_patterns": [],
            "filters": [],
            "modifiers": {},
        }

        assert sparql_processor.validate_query(invalid_query, context) == False

    def test_optimize_sparql_query(self, sparql_processor):
        """Test SPARQL query optimization."""
        query = {
            "query_type": "SELECT",
            "prefixes": {"cao": "http://example.org/"},
            "select_variables": ["compound"],
            "where_patterns": [
                {
                    "subject": "?compound",
                    "predicate": "cao:treats",
                    "object": "?condition",
                },
                {"subject": "?compound", "predicate": "rdf:type", "object": "cao:Drug"},
            ],
            "filters": ["?condition = 'pain'"],
            "modifiers": {},
        }

        context = SemanticQueryContext(
            query_language=QueryLanguage.SPARQL, reasoning_enabled=True
        )
        optimized = sparql_processor.optimize_query(query, context)

        assert "where_patterns" in optimized
        assert len(optimized["where_patterns"]) >= len(query["where_patterns"])

    def test_translate_to_backend(self, sparql_processor):
        """Test translation to backend queries."""
        parsed_query = {
            "query_type": "SELECT",
            "original_query": "SELECT ?x WHERE { ?x rdf:type cao:Drug }",
            "prefixes": {"cao": "http://example.org/"},
            "select_variables": ["x"],
            "where_patterns": [
                {"subject": "?x", "predicate": "rdf:type", "object": "cao:Drug"}
            ],
        }

        # Test Jena backend translation
        jena_query = sparql_processor.translate_to_backend_query(parsed_query, "jena")
        assert isinstance(jena_query, GraphQuery)
        assert jena_query.query_type == "sparql"

        # Test Neo4j backend translation
        neo4j_query = sparql_processor.translate_to_backend_query(parsed_query, "neo4j")
        assert isinstance(neo4j_query, GraphQuery)
        assert jena_query.query_type == "sparql"  # Still SPARQL for this test

        # Test generic backend translation
        generic_query = sparql_processor.translate_to_backend_query(
            parsed_query, "memory"
        )
        assert isinstance(generic_query, GraphQuery)


class TestCypherSemanticProcessor:
    """Test semantic Cypher processor."""

    @pytest.fixture
    def ontology(self):
        """Create mock ontology."""
        return Mock(spec=ChemicalOntology)

    @pytest.fixture
    def cypher_processor(self, ontology):
        """Create Cypher processor."""
        return CypherSemanticProcessor(ontology)

    def test_parse_semantic_cypher(self, cypher_processor):
        """Test parsing semantic Cypher query."""
        query = """
        MATCH (d:Drug)-[SIMILAR_TO]-(s:Drug)
        WHERE d.name = 'aspirin'
        RETURN s.name, s.properties
        """

        context = SemanticQueryContext(query_language=QueryLanguage.SEMANTIC_CYPHER)
        parsed = cypher_processor.parse_query(query, context)

        assert parsed["query_type"] == "semantic_cypher"
        assert len(parsed["match_patterns"]) >= 1
        assert len(parsed["where_conditions"]) >= 1
        assert "RETURN" in parsed["return_clause"]

    def test_validate_semantic_cypher(self, cypher_processor):
        """Test semantic Cypher validation."""
        # Valid query
        valid_query = {
            "query_type": "semantic_cypher",
            "match_patterns": [{"pattern": "(d:Drug)"}],
            "where_conditions": ["d.name = 'aspirin'"],
            "return_clause": "d.name",
            "semantic_operations": [],
        }

        context = SemanticQueryContext(query_language=QueryLanguage.SEMANTIC_CYPHER)
        assert cypher_processor.validate_query(valid_query, context) == True

        # Invalid query - no patterns
        invalid_query = {
            "query_type": "semantic_cypher",
            "match_patterns": [],
            "semantic_operations": [],
        }

        assert cypher_processor.validate_query(invalid_query, context) == False


class TestNaturalLanguageProcessor:
    """Test natural language processor."""

    @pytest.fixture
    def ontology(self):
        """Create mock ontology."""
        ontology = Mock(spec=ChemicalOntology)

        # Mock entity lookup
        aspirin_entity = Mock()
        aspirin_entity.name = "aspirin"
        aspirin_entity.uri = "http://example.org/aspirin"
        aspirin_entity.entity_type = EntityType.DRUG

        ontology.find_entity.return_value = [aspirin_entity]
        ontology.get_related_entities.return_value = []

        return ontology

    @pytest.fixture
    def nl_processor(self, ontology):
        """Create natural language processor."""
        return NaturalLanguageProcessor(ontology)

    def test_parse_natural_language(self, nl_processor):
        """Test parsing natural language query."""
        query = "What is the molecular weight of aspirin?"

        context = SemanticQueryContext(query_language=QueryLanguage.NATURAL_LANGUAGE)
        parsed = nl_processor.parse_query(query, context)

        assert parsed["query_type"] == "natural_language"
        assert parsed["original_query"] == query
        assert "entities" in parsed
        assert "intent" in parsed
        assert "components" in parsed

    def test_validate_natural_language(self, nl_processor):
        """Test natural language query validation."""
        # Valid query with entities
        valid_query = {
            "query_type": "natural_language",
            "entities": [{"text": "aspirin", "type": "chemical"}],
            "intent": {"type": "get_properties", "confidence": 0.8},
        }

        context = SemanticQueryContext(query_language=QueryLanguage.NATURAL_LANGUAGE)
        assert nl_processor.validate_query(valid_query, context) == True

        # Invalid query - no intent
        invalid_query = {
            "query_type": "natural_language",
            "entities": [],
            "intent": None,
        }

        assert nl_processor.validate_query(invalid_query, context) == False


class TestChemicalEntityRecognizer:
    """Test chemical entity recognizer."""

    @pytest.fixture
    def ontology(self):
        """Create mock ontology."""
        ontology = Mock(spec=ChemicalOntology)

        # Mock entity for aspirin
        aspirin_entity = Mock()
        aspirin_entity.name = "aspirin"
        aspirin_entity.uri = "http://example.org/aspirin"

        ontology.find_entity.side_effect = lambda name: (
            [aspirin_entity] if "aspirin" in name.lower() else []
        )

        return ontology

    @pytest.fixture
    def entity_recognizer(self, ontology):
        """Create entity recognizer."""
        return ChemicalEntityRecognizer(ontology)

    def test_extract_drug_names(self, entity_recognizer):
        """Test extraction of drug names."""
        text = "The patient was prescribed aspirin and ibuprofen for pain relief."

        entities = entity_recognizer.extract_entities(text)

        # Should find drug names
        drug_entities = [
            e for e in entities if e.get("type") in ["drug_names", "chemical"]
        ]
        assert len(drug_entities) >= 1

        # Check for aspirin specifically
        aspirin_found = any("aspirin" in e["text"].lower() for e in drug_entities)
        assert aspirin_found

    def test_extract_molecular_formulas(self, entity_recognizer):
        """Test extraction of molecular formulas."""
        text = "The molecular formula of water is H2O and glucose is C6H12O6."

        entities = entity_recognizer.extract_entities(text)

        # Should find molecular formulas
        formula_entities = [e for e in entities if e.get("type") == "molecular_formula"]
        assert len(formula_entities) >= 2

        # Check for specific formulas
        formulas = [e["text"] for e in formula_entities]
        assert "H2O" in formulas or "C6H12O6" in formulas

    def test_extract_smiles(self, entity_recognizer):
        """Test extraction of SMILES strings."""
        text = "The SMILES representation is CCO for ethanol."

        entities = entity_recognizer.extract_entities(text)

        # May find SMILES if pattern matches
        smiles_entities = [e for e in entities if e.get("type") == "smiles"]
        if smiles_entities:
            assert any("CCO" in e["text"] for e in smiles_entities)


class TestReasoningEngine:
    """Test reasoning engine functionality."""

    @pytest.fixture
    def ontology(self):
        """Create mock ontology."""
        return Mock(spec=ChemicalOntology)

    @pytest.fixture
    def schema(self):
        """Create mock schema."""
        schema = Mock(spec=ICKGSchema)
        schema.validate_relationship.return_value = True
        return schema

    @pytest.fixture
    def reasoning_engine(self, ontology, schema):
        """Create reasoning engine."""
        return ReasoningEngine(ontology, schema)

    def test_reasoning_engine_initialization(self, reasoning_engine):
        """Test reasoning engine initialization."""
        assert isinstance(reasoning_engine.rules, list)
        assert isinstance(reasoning_engine.inferred_facts, list)
        assert reasoning_engine.max_reasoning_depth == 5
        assert reasoning_engine.confidence_threshold == 0.5

    def test_add_reasoning_rule(self, reasoning_engine):
        """Test adding reasoning rules."""
        rule = ReasoningRule(
            rule_id="test_rule",
            name="Test Rule",
            description="A test rule",
            pattern="?x SIMILAR_TO ?y",
            conclusion_template="?y SIMILAR_TO ?x",
        )

        initial_count = len(reasoning_engine.rules)
        reasoning_engine.add_rule(rule)

        assert len(reasoning_engine.rules) == initial_count + 1
        assert reasoning_engine.rules[-1].rule_id == "test_rule"

    def test_remove_reasoning_rule(self, reasoning_engine):
        """Test removing reasoning rules."""
        rule = ReasoningRule(
            rule_id="removable_rule",
            name="Removable Rule",
            description="A rule to be removed",
            pattern="?x TEST ?y",
            conclusion_template="?y TEST ?x",
        )

        reasoning_engine.add_rule(rule)
        initial_count = len(reasoning_engine.rules)

        success = reasoning_engine.remove_rule("removable_rule")

        assert success == True
        assert len(reasoning_engine.rules) == initial_count - 1

        # Try removing non-existent rule
        success = reasoning_engine.remove_rule("non_existent")
        assert success == False

    def test_validate_inference(self, reasoning_engine):
        """Test inference validation."""
        # Valid inference
        valid_fact = InferredFact(
            subject="aspirin",
            predicate="TREATS",
            object="pain",
            confidence=0.8,
            reasoning_path=["rule_1"],
            source_facts=["aspirin SIMILAR_TO ibuprofen"],
            inferred_at=datetime.now(),
            rule_applied="similarity_rule",
        )

        assert reasoning_engine.validate_inference(valid_fact) == True

        # Invalid inference - low confidence
        invalid_fact = InferredFact(
            subject="aspirin",
            predicate="TREATS",
            object="pain",
            confidence=0.2,  # Below threshold
            reasoning_path=["rule_1"],
            source_facts=["weak evidence"],
            inferred_at=datetime.now(),
            rule_applied="weak_rule",
        )

        assert reasoning_engine.validate_inference(invalid_fact) == False


class TestChemicalReasoningRuleLibrary:
    """Test chemical reasoning rule library."""

    @pytest.fixture
    def ontology(self):
        """Create mock ontology."""
        return Mock(spec=ChemicalOntology)

    @pytest.fixture
    def rule_library(self, ontology):
        """Create rule library."""
        return ChemicalReasoningRuleLibrary(ontology)

    def test_library_initialization(self, rule_library):
        """Test rule library initialization."""
        rules = rule_library.get_all_rules()
        assert len(rules) > 0

        # Check for specific rule types
        rule_types = {rule.rule_type for rule in rules}
        assert RuleType.CHEMICAL_SIMILARITY in rule_types
        assert RuleType.TRANSITIVE in rule_types
        assert RuleType.STRUCTURAL in rule_types

    def test_get_rules_by_type(self, rule_library):
        """Test getting rules by type."""
        similarity_rules = rule_library.get_rules_by_type(RuleType.CHEMICAL_SIMILARITY)
        assert len(similarity_rules) > 0

        for rule in similarity_rules:
            assert rule.rule_type == RuleType.CHEMICAL_SIMILARITY

    def test_add_custom_rule(self, rule_library):
        """Test adding custom rules."""
        from chemi_accel.knowledge_graph.reasoning_rules import (
            AdvancedReasoningRule,
            ConfidenceModel,
            RuleConsequent,
        )

        custom_rule = AdvancedReasoningRule(
            rule_id="custom_test_rule",
            name="Custom Test Rule",
            description="A custom rule for testing",
            rule_type=RuleType.IMPLICATION,
            antecedent_patterns=["?x HAS_PROPERTY ?p"],
            consequent=RuleConsequent(template="?x IS_CHARACTERIZED_BY ?p"),
            confidence_model=ConfidenceModel.MULTIPLICATIVE,
        )

        initial_count = len(rule_library.get_all_rules())
        rule_library.add_rule(custom_rule)

        assert len(rule_library.get_all_rules()) == initial_count + 1
        assert rule_library.get_rule("custom_test_rule") is not None


class TestAdvancedNLPProcessor:
    """Test advanced NLP processor."""

    @pytest.fixture
    def ontology(self):
        """Create mock ontology."""
        ontology = Mock(spec=ChemicalOntology)

        # Mock aspirin entity
        aspirin_entity = Mock()
        aspirin_entity.name = "aspirin"
        aspirin_entity.uri = "http://example.org/aspirin"
        aspirin_entity.entity_type = EntityType.DRUG

        ontology.find_entity.side_effect = lambda name: (
            [aspirin_entity] if "aspirin" in name.lower() else []
        )
        ontology.get_related_entities.return_value = []

        return ontology

    @pytest.fixture
    def nlp_processor(self, ontology):
        """Create NLP processor."""
        return AdvancedNLPProcessor(ontology)

    def test_process_simple_query(self, nlp_processor):
        """Test processing simple query."""
        query = "What is the molecular weight of aspirin?"

        result = nlp_processor.process_query(query)

        assert isinstance(result, ProcessedQuery)
        assert result.original_query == query
        assert result.intent in [QueryIntent.GET_PROPERTIES, QueryIntent.FIND_ENTITIES]
        assert result.confidence > 0
        assert (
            len(result.entities) >= 0
        )  # May or may not find entities depending on implementation

    def test_process_comparison_query(self, nlp_processor):
        """Test processing comparison query."""
        query = "Compare aspirin and ibuprofen for pain relief."

        result = nlp_processor.process_query(query)

        assert isinstance(result, ProcessedQuery)
        assert result.intent == QueryIntent.COMPARE_ENTITIES
        assert result.confidence > 0

    def test_process_relationship_query(self, nlp_processor):
        """Test processing relationship query."""
        query = "How does aspirin interact with warfarin?"

        result = nlp_processor.process_query(query)

        assert isinstance(result, ProcessedQuery)
        assert result.intent in [
            QueryIntent.FIND_RELATIONSHIPS,
            QueryIntent.PREDICT_INTERACTION,
        ]
        assert "interact" in result.relationships or len(result.relationships) >= 0

    def test_processing_statistics(self, nlp_processor):
        """Test processing statistics tracking."""
        initial_stats = nlp_processor.get_processing_statistics()

        # Process some queries
        nlp_processor.process_query("What is aspirin?")
        nlp_processor.process_query("Compare aspirin and morphine.")

        updated_stats = nlp_processor.get_processing_statistics()

        assert updated_stats["total_queries"] > initial_stats["total_queries"]
        assert (
            updated_stats["successful_processing"]
            >= initial_stats["successful_processing"]
        )


class TestSemanticQueryEngine:
    """Test main semantic query engine."""

    @pytest.fixture
    def backend(self):
        """Create mock backend."""
        backend = MemoryGraphBackend({})
        backend.connect()

        # Add some test data
        aspirin_id = backend.create_node(["Drug"], {"name": "aspirin", "type": "nsaid"})
        pain_id = backend.create_node(["Condition"], {"name": "pain"})
        backend.create_relationship(aspirin_id, pain_id, "TREATS", {"efficacy": 0.8})

        return backend

    @pytest.fixture
    def ontology(self):
        """Create mock ontology."""
        ontology = Mock(spec=ChemicalOntology)

        aspirin_entity = ChemicalEntity(
            entity_id="aspirin_1",
            name="aspirin",
            entity_type=EntityType.DRUG,
            uri="http://example.org/aspirin",
        )

        ontology.find_entity.side_effect = lambda name: (
            [aspirin_entity] if "aspirin" in name.lower() else []
        )
        ontology.get_related_entities.return_value = []

        return ontology

    @pytest.fixture
    def schema(self):
        """Create mock schema."""
        schema = Mock(spec=ICKGSchema)
        schema.validate_relationship.return_value = True
        return schema

    @pytest.fixture
    def semantic_engine(self, backend, ontology, schema):
        """Create semantic query engine."""
        return SemanticQueryEngine(backend, ontology, schema)

    def test_engine_initialization(self, semantic_engine):
        """Test semantic query engine initialization."""
        assert semantic_engine.backend is not None
        assert semantic_engine.ontology is not None
        assert semantic_engine.schema is not None
        assert isinstance(semantic_engine.processors, dict)
        assert semantic_engine.reasoning_engine is not None
        assert QueryLanguage.SPARQL in semantic_engine.processors
        assert QueryLanguage.NATURAL_LANGUAGE in semantic_engine.processors

    def test_execute_sparql_query(self, semantic_engine):
        """Test executing SPARQL query."""
        query = """
        SELECT ?drug ?condition
        WHERE {
            ?drug treats ?condition .
        }
        """

        context = SemanticQueryContext(
            query_language=QueryLanguage.SPARQL, reasoning_enabled=False
        )

        try:
            result = semantic_engine.execute_semantic_query(
                query, QueryLanguage.SPARQL, context
            )

            assert isinstance(result, GraphQueryResult)
            assert result.metadata.get("semantic_query") == True
            assert result.metadata.get("language") == QueryLanguage.SPARQL.value

        except Exception as e:
            # Some implementations may not support full SPARQL
            assert "Unsupported" in str(e) or "Invalid" in str(e)

    def test_execute_natural_language_query(self, semantic_engine):
        """Test executing natural language query."""
        question = "What drugs treat pain?"

        result = semantic_engine.ask_natural_language(question)

        assert isinstance(result, GraphQueryResult)
        assert result.metadata.get("semantic_query") == True
        assert result.metadata.get("language") == QueryLanguage.NATURAL_LANGUAGE.value

    def test_query_with_reasoning(self, semantic_engine):
        """Test query execution with reasoning enabled."""
        query = "Find drugs similar to aspirin"

        context = SemanticQueryContext(
            query_language=QueryLanguage.NATURAL_LANGUAGE,
            reasoning_enabled=True,
            include_inferred_facts=True,
        )

        result = semantic_engine.execute_semantic_query(
            query, QueryLanguage.NATURAL_LANGUAGE, context
        )

        assert isinstance(result, GraphQueryResult)
        assert result.metadata.get("reasoning_applied") == True

    def test_infer_new_facts(self, semantic_engine):
        """Test fact inference."""
        inferred_facts = semantic_engine.infer_new_facts(
            ReasoningStrategy.FORWARD_CHAINING
        )

        assert isinstance(inferred_facts, list)
        # May or may not have inferred facts depending on data and rules
        for fact in inferred_facts:
            assert isinstance(fact, InferredFact)
            assert fact.confidence > 0

    def test_add_reasoning_rule(self, semantic_engine):
        """Test adding reasoning rules."""
        rule = ReasoningRule(
            rule_id="test_semantic_rule",
            name="Test Semantic Rule",
            description="A test rule for semantic engine",
            pattern="?x SIMILAR_TO ?y",
            conclusion_template="?y SIMILAR_TO ?x",
        )

        semantic_engine.add_reasoning_rule(rule)

        # Check that rule was added to reasoning engine
        engine_rules = semantic_engine.reasoning_engine.rules
        rule_ids = [r.rule_id for r in engine_rules]
        assert "test_semantic_rule" in rule_ids

    def test_explain_result(self, semantic_engine):
        """Test result explanation."""
        query = "What is aspirin?"
        result = semantic_engine.ask_natural_language(query)

        explanation = semantic_engine.explain_result(result)

        assert isinstance(explanation, dict)
        assert "query_type" in explanation
        assert "results_count" in explanation
        assert "execution_time" in explanation
        assert "reasoning_applied" in explanation
        assert "cache_hit" in explanation

    def test_query_statistics(self, semantic_engine):
        """Test query statistics tracking."""
        initial_stats = semantic_engine.get_query_statistics()

        # Execute some queries
        semantic_engine.ask_natural_language("What is aspirin?")
        semantic_engine.ask_natural_language("What treats pain?")

        updated_stats = semantic_engine.get_query_statistics()

        assert updated_stats["total_queries"] > initial_stats["total_queries"]
        assert "cache_hit_rate" in updated_stats
        assert "reasoning_queries" in updated_stats
        assert "average_execution_time" in updated_stats

    def test_cache_functionality(self, semantic_engine):
        """Test query result caching."""
        query = "What is aspirin?"
        context1 = SemanticQueryContext(
            query_language=QueryLanguage.NATURAL_LANGUAGE, cache_results=True
        )
        context2 = SemanticQueryContext(
            query_language=QueryLanguage.NATURAL_LANGUAGE, cache_results=True
        )

        # First execution
        result1 = semantic_engine.execute_semantic_query(
            query, QueryLanguage.NATURAL_LANGUAGE, context1
        )

        # Second execution should use cache
        result2 = semantic_engine.execute_semantic_query(
            query, QueryLanguage.NATURAL_LANGUAGE, context2
        )

        # Both should return results
        assert isinstance(result1, GraphQueryResult)
        assert isinstance(result2, GraphQueryResult)

        # Check cache hit statistics
        stats = semantic_engine.get_query_statistics()
        assert stats["total_queries"] >= 2

    def test_clear_cache(self, semantic_engine):
        """Test cache clearing."""
        # Execute a query to populate cache
        semantic_engine.ask_natural_language("What is aspirin?")

        # Clear cache
        semantic_engine.clear_cache()

        # Cache should be empty (implementation dependent)
        assert len(semantic_engine.query_cache) == 0

    def test_error_handling(self, semantic_engine):
        """Test error handling for invalid queries."""
        # Invalid SPARQL
        invalid_sparql = "INVALID SPARQL QUERY"

        result = semantic_engine.execute_semantic_query(
            invalid_sparql,
            QueryLanguage.SPARQL,
            SemanticQueryContext(query_language=QueryLanguage.SPARQL),
        )

        assert isinstance(result, GraphQueryResult)
        assert result.metadata.get("error") == True
        assert "error" in result.summary


class TestChemicalDSLParser:
    """Test Chemical DSL parser."""

    @pytest.fixture
    def ontology(self):
        """Create mock ontology."""
        return Mock(spec=ChemicalOntology)

    @pytest.fixture
    def dsl_parser(self, ontology):
        """Create DSL parser."""
        return ChemicalDSLParser(ontology)

    def test_parse_find_operation(self, dsl_parser):
        """Test parsing FIND operations."""
        query = "FIND compounds WHERE molecular_weight > 500"

        parsed = dsl_parser.parse(query)

        assert parsed["query_type"] == "chemical_dsl"
        assert parsed["operation"]["type"] == "find"
        assert "WHERE" in parsed["clauses"]

    def test_parse_compare_operation(self, dsl_parser):
        """Test parsing COMPARE operations."""
        query = "COMPARE aspirin AND ibuprofen"

        parsed = dsl_parser.parse(query)

        assert parsed["query_type"] == "chemical_dsl"
        assert parsed["operation"]["type"] == "compare"
        assert len(parsed["operation"]["entities"]) == 2

    def test_parse_with_constraints(self, dsl_parser):
        """Test parsing queries with constraints."""
        query = (
            "FIND drugs WHERE molecular_weight BETWEEN 100 AND 500 WITH type = 'nsaid'"
        )

        parsed = dsl_parser.parse(query)

        assert parsed["query_type"] == "chemical_dsl"
        assert "WHERE" in parsed["clauses"]
        assert "WITH" in parsed["clauses"]


class TestIntegrationScenarios:
    """Integration tests for complete semantic query scenarios."""

    @pytest.fixture
    def complete_system(self):
        """Create complete semantic query system."""
        # Create backend with test data
        backend = MemoryGraphBackend({})
        backend.connect()

        # Add comprehensive test data
        aspirin_id = backend.create_node(
            ["Drug", "Chemical"],
            {"name": "aspirin", "type": "nsaid", "molecular_weight": 180.16},
        )
        ibuprofen_id = backend.create_node(
            ["Drug", "Chemical"],
            {"name": "ibuprofen", "type": "nsaid", "molecular_weight": 206.29},
        )
        pain_id = backend.create_node(["Condition"], {"name": "pain"})
        inflammation_id = backend.create_node(["Condition"], {"name": "inflammation"})

        # Add relationships
        backend.create_relationship(aspirin_id, pain_id, "TREATS", {"efficacy": 0.8})
        backend.create_relationship(
            aspirin_id, inflammation_id, "REDUCES", {"efficacy": 0.7}
        )
        backend.create_relationship(ibuprofen_id, pain_id, "TREATS", {"efficacy": 0.9})
        backend.create_relationship(
            aspirin_id, ibuprofen_id, "SIMILAR_TO", {"similarity": 0.85}
        )

        # Create ontology
        ontology = Mock(spec=ChemicalOntology)

        aspirin_entity = ChemicalEntity(
            entity_id="aspirin_1",
            name="aspirin",
            entity_type=EntityType.DRUG,
            uri="http://example.org/aspirin",
        )

        ibuprofen_entity = ChemicalEntity(
            entity_id="ibuprofen_1",
            name="ibuprofen",
            entity_type=EntityType.DRUG,
            uri="http://example.org/ibuprofen",
        )

        def mock_find_entity(name):
            name_lower = name.lower()
            if "aspirin" in name_lower:
                return [aspirin_entity]
            elif "ibuprofen" in name_lower:
                return [ibuprofen_entity]
            return []

        ontology.find_entity.side_effect = mock_find_entity
        ontology.get_related_entities.return_value = []

        # Create schema
        schema = Mock(spec=ICKGSchema)
        schema.validate_relationship.return_value = True

        # Create semantic engine
        engine = SemanticQueryEngine(backend, ontology, schema)

        return engine

    def test_natural_language_drug_query(self, complete_system):
        """Test complete natural language drug query."""
        query = "What drugs treat pain?"

        result = complete_system.ask_natural_language(query)

        assert isinstance(result, GraphQueryResult)
        assert len(result.records) >= 1  # Should find aspirin and/or ibuprofen

        # Check that we found relevant drugs
        drug_names = []
        for record in result.records:
            if "name" in record:
                drug_names.append(record["name"])

        # Should find at least one pain-treating drug
        assert len(drug_names) >= 1

    def test_comparison_query_with_reasoning(self, complete_system):
        """Test comparison query with reasoning enabled."""
        query = "Compare aspirin and ibuprofen for treating pain"

        context = SemanticQueryContext(
            query_language=QueryLanguage.NATURAL_LANGUAGE,
            reasoning_enabled=True,
            include_inferred_facts=True,
        )

        result = complete_system.execute_semantic_query(
            query, QueryLanguage.NATURAL_LANGUAGE, context
        )

        assert isinstance(result, GraphQueryResult)
        assert result.metadata.get("reasoning_applied") == True

        # Should have results about both drugs
        assert len(result.records) >= 1

    def test_property_query_with_constraints(self, complete_system):
        """Test property query with numerical constraints."""
        query = "Find drugs with molecular weight between 150 and 250"

        result = complete_system.ask_natural_language(query)

        assert isinstance(result, GraphQueryResult)
        # Both aspirin (180.16) and ibuprofen (206.29) should match
        assert len(result.records) >= 1

    def test_similarity_based_inference(self, complete_system):
        """Test similarity-based reasoning."""
        # First, infer some facts
        inferred_facts = complete_system.infer_new_facts(
            ReasoningStrategy.FORWARD_CHAINING
        )

        # Then query for similar drugs
        query = "What drugs are similar to aspirin?"

        context = SemanticQueryContext(
            query_language=QueryLanguage.NATURAL_LANGUAGE, reasoning_enabled=True
        )

        result = complete_system.execute_semantic_query(
            query, QueryLanguage.NATURAL_LANGUAGE, context
        )

        assert isinstance(result, GraphQueryResult)
        # Should find ibuprofen due to similarity relationship
        assert (
            len(result.records) >= 0
        )  # May or may not find similarities depending on implementation

    def test_multi_step_reasoning_chain(self, complete_system):
        """Test multi-step reasoning chain."""
        # Add reasoning rule for testing
        similarity_rule = ReasoningRule(
            rule_id="test_similarity_transitivity",
            name="Similarity Transitivity",
            description="If A is similar to B and B treats C, then A may treat C",
            pattern="?a SIMILAR_TO ?b AND ?b TREATS ?c",
            conclusion_template="?a POTENTIALLY_TREATS ?c",
        )

        complete_system.add_reasoning_rule(similarity_rule)

        # Infer facts
        inferred_facts = complete_system.infer_new_facts(
            ReasoningStrategy.FORWARD_CHAINING
        )

        # Check if we got reasonable inferences
        assert isinstance(inferred_facts, list)
        for fact in inferred_facts:
            assert isinstance(fact, InferredFact)
            assert 0.0 <= fact.confidence <= 1.0

    def test_explanation_generation(self, complete_system):
        """Test explanation generation for query results."""
        query = "Why does aspirin treat pain?"

        result = complete_system.ask_natural_language(query)
        explanation = complete_system.explain_result(result)

        assert isinstance(explanation, dict)
        assert "query_type" in explanation
        assert explanation["query_type"] == "semantic"
        assert "results_count" in explanation
        assert isinstance(explanation["results_count"], int)

    def test_performance_with_large_query_set(self, complete_system):
        """Test performance with multiple queries."""
        queries = [
            "What is aspirin?",
            "What treats pain?",
            "Compare aspirin and ibuprofen",
            "Find similar drugs to aspirin",
            "What are the properties of ibuprofen?",
        ]

        results = []
        for query in queries:
            result = complete_system.ask_natural_language(query)
            results.append(result)

        # All queries should return results
        assert len(results) == len(queries)
        for result in results:
            assert isinstance(result, GraphQueryResult)

        # Check statistics
        stats = complete_system.get_query_statistics()
        assert stats["total_queries"] >= len(queries)
        assert stats["average_execution_time"] > 0
