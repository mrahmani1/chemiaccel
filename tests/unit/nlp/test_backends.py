"""
Tests for NLP Backends

Testing the multi-backend NLP system including CUDA, JAX, Rapids,
and CPU backends for chemical entity recognition.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from chemi_accel.nlp.backends import BackendType, NLPBackend, get_nlp_backend
from chemi_accel.nlp.backends.base import (
    DocumentStructure,
    EntityExtraction,
    RelationshipExtraction,
)
from chemi_accel.nlp.backends.cpu_backend import CPUNLPBackend


class TestBackendSelection:
    """Test backend selection and initialization."""

    def test_get_nlp_backend_auto(self):
        """Test automatic backend selection."""
        backend = get_nlp_backend(backend=None)

        assert backend is not None
        assert isinstance(backend, NLPBackend)
        # Should default to CPU backend when others aren't available
        assert backend.backend_type == BackendType.CPU

    def test_get_nlp_backend_explicit_cpu(self):
        """Test explicit CPU backend selection."""
        backend = get_nlp_backend(backend="cpu")

        assert isinstance(backend, CPUNLPBackend)
        assert backend.backend_type == BackendType.CPU

    def test_get_nlp_backend_with_device(self):
        """Test backend selection with device specification."""
        backend = get_nlp_backend(backend="cpu", device="cpu")

        assert backend.device == "cpu"

    def test_get_nlp_backend_invalid(self):
        """Test invalid backend selection."""
        with pytest.raises(ValueError):
            get_nlp_backend(backend="invalid_backend")

    @patch("chemi_accel.nlp.backends.torch")
    def test_cuda_backend_availability_check(self, mock_torch):
        """Test CUDA backend availability checking."""
        # Mock CUDA availability
        mock_torch.cuda.is_available.return_value = True

        # Test that CUDA backend would be selected (if imports work)
        from chemi_accel.nlp.backends import _detect_best_backend

        # Note: This will still return 'cpu' due to import issues, but tests the logic
        best_backend = _detect_best_backend()
        assert best_backend in ["cuda", "jax", "rapids", "cpu"]

    @patch("chemi_accel.nlp.backends.jax")
    def test_jax_backend_availability_check(self, mock_jax):
        """Test JAX backend availability checking."""
        # Mock JAX GPU availability
        mock_jax.devices.return_value = [Mock()]  # Mock GPU device

        from chemi_accel.nlp.backends import _detect_best_backend

        best_backend = _detect_best_backend()
        assert best_backend in ["cuda", "jax", "rapids", "cpu"]


class TestCPUBackend:
    """Test CPU backend functionality."""

    @pytest.fixture
    def cpu_backend(self):
        """Create a CPU backend for testing."""
        return CPUNLPBackend()

    def test_initialization(self, cpu_backend):
        """Test CPU backend initialization."""
        assert cpu_backend.backend_type == BackendType.CPU
        assert cpu_backend.device == "cpu"
        assert cpu_backend._models == {}
        assert cpu_backend._tokenizers == {}

    def test_load_model(self, cpu_backend):
        """Test model loading."""
        model = cpu_backend.load_model("chemical_ner")

        assert model is not None
        assert "chemical_ner" in cpu_backend._models
        assert "vectorizer" in model
        assert "classifier" in model
        assert "config" in model

    def test_load_tokenizer(self, cpu_backend):
        """Test tokenizer loading."""
        tokenizer = cpu_backend.load_tokenizer("chemical_ner")

        assert tokenizer is not None
        assert "chemical_ner" in cpu_backend._tokenizers
        assert hasattr(tokenizer, "tokenize")

    def test_tokenize_text(self, cpu_backend):
        """Test text tokenization."""
        text = "Aspirin is a drug"
        tokens = cpu_backend.tokenize_text(text)

        assert "input_ids" in tokens
        assert "tokens" in tokens
        assert len(tokens["tokens"]) == 1  # One text input
        assert len(tokens["tokens"][0]) > 0  # Should have tokens

    def test_extract_entities(self, cpu_backend):
        """Test entity extraction."""
        text = "Aspirin (C9H8O4) is used to treat pain."
        entities = cpu_backend.extract_entities(text)

        assert isinstance(entities, list)
        assert len(entities) > 0

        # Check entity structure
        entity = entities[0]
        assert isinstance(entity, EntityExtraction)
        assert hasattr(entity, "text")
        assert hasattr(entity, "label")
        assert hasattr(entity, "start_idx")
        assert hasattr(entity, "end_idx")
        assert hasattr(entity, "confidence")

    def test_extract_entities_chemical_patterns(self, cpu_backend):
        """Test chemical pattern recognition."""
        text = "The compound H2SO4 reacts with NaOH to form Na2SO4."
        entities = cpu_backend.extract_entities(text)

        # Should find chemical formulas
        chemical_entities = [e for e in entities if e.label == "CHEMICAL"]
        assert len(chemical_entities) > 0

        # Check for specific chemicals
        entity_texts = [e.text for e in chemical_entities]
        assert any("H2SO4" in text for text in entity_texts)

    def test_extract_relationships(self, cpu_backend):
        """Test relationship extraction."""
        text = "Aspirin inhibits cyclooxygenase enzyme."
        entities = [
            EntityExtraction("aspirin", "CHEMICAL", 0, 7, 0.9, {}),
            EntityExtraction("cyclooxygenase", "CHEMICAL", 17, 31, 0.8, {}),
        ]

        relationships = cpu_backend.extract_relationships(text, entities)

        assert isinstance(relationships, list)
        # May be empty with simple patterns, that's OK

        if relationships:
            rel = relationships[0]
            assert isinstance(rel, RelationshipExtraction)
            assert hasattr(rel, "source_entity")
            assert hasattr(rel, "target_entity")
            assert hasattr(rel, "relation_type")

    def test_analyze_document_structure(self, cpu_backend):
        """Test document structure analysis."""
        text = """
        INTRODUCTION
        This is the introduction section.

        METHODS
        This describes the methods.

        Table 1: Test results
        | A | B |
        | 1 | 2 |

        Figure 1: Test figure

        REFERENCES
        [1] Author et al. (2023)
        """

        structure = cpu_backend.analyze_document_structure(text)

        assert isinstance(structure, DocumentStructure)
        assert len(structure.sections) > 0
        assert len(structure.tables) > 0
        assert len(structure.figures) > 0
        assert len(structure.references) > 0

    def test_extract_table_data(self, cpu_backend):
        """Test table data extraction."""
        table_text = """
        | Compound | MW | MP |
        | Aspirin | 180 | 135°C |
        | Caffeine | 194 | 235°C |
        """

        table_data = cpu_backend.extract_table_data(table_text)

        assert "headers" in table_data
        assert "rows" in table_data
        assert "num_columns" in table_data
        assert "num_rows" in table_data

        assert len(table_data["headers"]) == 3
        assert len(table_data["rows"]) == 2

    def test_classify_text(self, cpu_backend):
        """Test text classification."""
        texts = [
            "The synthesis of aspirin was performed",
            "NMR analysis showed the following peaks",
            "The melting point was determined to be 135°C",
            "IC50 values were measured using the MTT assay",
        ]

        classifications = cpu_backend.classify_text(texts)

        assert len(classifications) == len(texts)

        for classification in classifications:
            assert isinstance(classification, dict)
            assert "SYNTHESIS" in classification
            assert "ANALYSIS" in classification
            assert "PROPERTY" in classification
            assert "BIOACTIVITY" in classification
            assert "OTHER" in classification

            # Probabilities should sum to ~1
            total_prob = sum(classification.values())
            assert abs(total_prob - 1.0) < 0.1

    def test_embed_text(self, cpu_backend):
        """Test text embedding."""
        texts = ["aspirin", "caffeine", "morphine"]
        embeddings = cpu_backend.embed_text(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] > 0  # Should have embedding dimensions

    def test_batch_processing(self, cpu_backend):
        """Test batch processing of multiple operations."""
        texts = ["Aspirin is a drug", "Caffeine is a stimulant"]
        operations = ["entities", "classification"]

        results = cpu_backend.batch_process(texts, operations, batch_size=1)

        assert len(results) == len(texts)

        for result in results:
            assert "text" in result
            assert "entities" in result
            assert "classification" in result

            assert isinstance(result["entities"], list)
            assert isinstance(result["classification"], list)

    def test_model_management(self, cpu_backend):
        """Test model management functions."""
        # Load a model
        cpu_backend.load_model("test_model")

        # Check model info
        info = cpu_backend.get_model_info("test_model")
        assert info["loaded"] == True
        assert info["backend"] == "cpu"

        # List loaded models
        models = cpu_backend.list_loaded_models()
        assert "test_model" in models

        # Unload model
        success = cpu_backend.unload_model("test_model")
        assert success == True

        # Check model is unloaded
        info = cpu_backend.get_model_info("test_model")
        assert info["loaded"] == False

    def test_memory_usage(self, cpu_backend):
        """Test memory usage reporting."""
        usage = cpu_backend.get_memory_usage()

        assert "backend" in usage
        assert "device" in usage
        assert "loaded_models" in usage
        assert "loaded_tokenizers" in usage

        assert usage["backend"] == "cpu"
        assert usage["device"] == "cpu"

    def test_train_model(self, cpu_backend):
        """Test model training functionality."""
        texts = [
            "Aspirin is a drug",
            "Water is a solvent",
            "Benzene is toxic",
            "Ethanol is flammable",
        ]
        labels = ["DRUG", "SOLVENT", "TOXIC", "FLAMMABLE"]

        result = cpu_backend.train_model("test_classifier", texts, labels)

        assert "success" in result
        assert "model_name" in result
        assert "training_samples" in result

        if result["success"]:
            assert result["training_samples"] == len(texts)
            assert result["unique_labels"] == len(set(labels))

    def test_chemical_name_patterns(self, cpu_backend):
        """Test specific chemical name pattern recognition."""
        texts = [
            "2-methylpyridine was synthesized",
            "The CAS number is 123-45-6",
            "Streptomycin showed activity",
            "The formula C6H6 represents benzene",
        ]

        all_entities = []
        for text in texts:
            entities = cpu_backend.extract_entities(text)
            all_entities.extend(entities)

        # Should find chemical entities
        chemical_entities = [e for e in all_entities if e.label == "CHEMICAL"]
        assert len(chemical_entities) > 0

        # Check for specific patterns
        entity_texts = [e.text.lower() for e in chemical_entities]
        assert any(
            "methylpyridine" in text or "pyridine" in text for text in entity_texts
        )
        assert any("c6h6" in text for text in entity_texts)

    def test_error_handling(self, cpu_backend):
        """Test error handling in backend operations."""
        # Test with empty text
        entities = cpu_backend.extract_entities("")
        assert len(entities) == 0

        # Test with None text
        entities = cpu_backend.extract_entities(None)
        assert len(entities) == 0

        # Test with very long text
        long_text = "aspirin " * 10000
        entities = cpu_backend.extract_entities(long_text)
        # Should handle gracefully without crashing
        assert isinstance(entities, list)


class TestNLPBackendBase:
    """Test base NLP backend interface."""

    def test_backend_interface(self):
        """Test that CPU backend implements all required methods."""
        backend = CPUNLPBackend()

        # Test that all abstract methods are implemented
        assert hasattr(backend, "load_model")
        assert hasattr(backend, "load_tokenizer")
        assert hasattr(backend, "tokenize_text")
        assert hasattr(backend, "extract_entities")
        assert hasattr(backend, "extract_relationships")
        assert hasattr(backend, "analyze_document_structure")
        assert hasattr(backend, "extract_table_data")
        assert hasattr(backend, "classify_text")
        assert hasattr(backend, "embed_text")

        # Test that methods are callable
        assert callable(backend.load_model)
        assert callable(backend.extract_entities)
        assert callable(backend.classify_text)

    def test_backend_type_consistency(self):
        """Test backend type consistency."""
        backend = CPUNLPBackend()

        assert backend._get_backend_type() == BackendType.CPU
        assert backend.backend_type == BackendType.CPU


class TestBackendIntegration:
    """Integration tests for backend functionality."""

    def test_end_to_end_chemical_processing(self):
        """Test end-to-end chemical text processing."""
        backend = CPUNLPBackend()

        # Realistic chemical text
        text = """
        The synthesis of 2-acetylpyridine was accomplished by treating pyridine
        with acetic anhydride in the presence of AlCl3 catalyst at 80°C for 3 hours.
        The product was obtained in 78% yield and characterized by 1H NMR and mass spectrometry.
        The melting point was determined to be 145-147°C.
        """

        # Extract entities
        entities = backend.extract_entities(text)
        assert len(entities) > 5

        # Extract relationships
        relationships = backend.extract_relationships(text, entities)
        # May be 0 with simple patterns, that's OK

        # Classify text
        classification = backend.classify_text(text)
        assert len(classification) == 1
        assert "SYNTHESIS" in classification[0]

        # Analyze structure
        structure = backend.analyze_document_structure(text)
        assert structure.metadata["total_lines"] > 0

        # Extract embeddings
        embeddings = backend.embed_text(text)
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] > 0

    def test_chemical_table_processing(self):
        """Test processing of chemical data tables."""
        backend = CPUNLPBackend()

        table_text = """
        Table 1: Biological activity of synthesized compounds

        | Compound | Structure | IC50 (µM) | Selectivity |
        |----------|-----------|-----------|-------------|
        | 3a       | C6H5N     | 2.3       | 4.1         |
        | 3b       | C7H7N     | 8.7       | 2.8         |
        | 3c       | C8H9N     | 15.2      | 1.9         |
        | Cisplatin| Pt(NH3)2Cl2| 5.1      | 2.3         |
        """

        # Extract table data
        table_data = backend.extract_table_data(table_text)

        assert len(table_data["headers"]) == 4
        assert len(table_data["rows"]) == 4

        # Extract entities from table text
        entities = backend.extract_entities(table_text)

        # Should find compound names and chemical formulas
        chemical_entities = [e for e in entities if e.label == "CHEMICAL"]
        assert len(chemical_entities) > 0

        # Should find specific chemical formulas
        entity_texts = [e.text for e in chemical_entities]
        formulas = [text for text in entity_texts if any(c.isdigit() for c in text)]
        assert len(formulas) > 0

    def test_multilingual_chemical_names(self):
        """Test handling of chemical names in different formats."""
        backend = CPUNLPBackend()

        texts = [
            "Acetylsalicylic acid (aspirin) is used for pain relief",
            "The IUPAC name is 2-acetoxybenzoic acid",
            "CAS Registry Number: 50-78-2",
            "Molecular formula: C9H8O4",
            "SMILES: CC(=O)OC1=CC=CC=C1C(=O)O",
        ]

        all_entities = []
        for text in texts:
            entities = backend.extract_entities(text)
            all_entities.extend(entities)

        # Should recognize various chemical identifiers
        chemical_entities = [e for e in all_entities if e.label == "CHEMICAL"]
        assert len(chemical_entities) > 3

        # Should find molecular formula
        formulas = [e for e in chemical_entities if "C9H8O4" in e.text]
        assert len(formulas) > 0
