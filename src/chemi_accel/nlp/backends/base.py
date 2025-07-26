"""
Base NLP Backend Interface

Abstract base class defining the interface for all NLP backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class BackendType(Enum):
    """Available NLP backend types."""

    CUDA = "cuda"
    JAX = "jax"
    RAPIDS = "rapids"
    CPU = "cpu"


@dataclass
class EntityExtraction:
    """Container for extracted entity information."""

    text: str
    label: str
    start_idx: int
    end_idx: int
    confidence: float
    properties: Dict[str, Any]


@dataclass
class RelationshipExtraction:
    """Container for extracted relationship information."""

    source_entity: EntityExtraction
    target_entity: EntityExtraction
    relation_type: str
    confidence: float
    context: str
    properties: Dict[str, Any]


@dataclass
class DocumentStructure:
    """Container for document structure analysis."""

    sections: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    figures: List[Dict[str, Any]]
    references: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class NLPBackend(ABC):
    """
    Abstract base class for NLP backends.

    Defines the interface that all NLP backends must implement for
    chemical entity recognition and information extraction.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device
        self.backend_type = self._get_backend_type()
        self._models = {}
        self._tokenizers = {}

    @abstractmethod
    def _get_backend_type(self) -> BackendType:
        """Return the backend type."""
        pass

    @abstractmethod
    def load_model(self, model_name: str, model_type: str = "transformer") -> Any:
        """
        Load a pre-trained model.

        Args:
            model_name: Name or path of the model
            model_type: Type of model ('transformer', 'crf', 'lstm', etc.)

        Returns:
            Loaded model object
        """
        pass

    @abstractmethod
    def load_tokenizer(self, tokenizer_name: str) -> Any:
        """
        Load a tokenizer.

        Args:
            tokenizer_name: Name or path of the tokenizer

        Returns:
            Loaded tokenizer object
        """
        pass

    @abstractmethod
    def tokenize_text(
        self, text: Union[str, List[str]], tokenizer_name: str = "default"
    ) -> Dict[str, Any]:
        """
        Tokenize input text.

        Args:
            text: Input text or list of texts
            tokenizer_name: Name of tokenizer to use

        Returns:
            Tokenized representation
        """
        pass

    @abstractmethod
    def extract_entities(
        self, text: Union[str, List[str]], model_name: str = "chemical_ner"
    ) -> List[EntityExtraction]:
        """
        Extract chemical entities from text.

        Args:
            text: Input text or list of texts
            model_name: Name of NER model to use

        Returns:
            List of extracted entities
        """
        pass

    @abstractmethod
    def extract_relationships(
        self,
        text: Union[str, List[str]],
        entities: Optional[List[EntityExtraction]] = None,
        model_name: str = "relation_extraction",
    ) -> List[RelationshipExtraction]:
        """
        Extract relationships between entities.

        Args:
            text: Input text or list of texts
            entities: Pre-extracted entities (optional)
            model_name: Name of relation extraction model

        Returns:
            List of extracted relationships
        """
        pass

    @abstractmethod
    def analyze_document_structure(self, text: str) -> DocumentStructure:
        """
        Analyze document structure (sections, tables, figures).

        Args:
            text: Full document text

        Returns:
            Document structure analysis
        """
        pass

    @abstractmethod
    def extract_table_data(self, table_text: str) -> Dict[str, Any]:
        """
        Extract structured data from table text.

        Args:
            table_text: Text representation of table

        Returns:
            Structured table data
        """
        pass

    @abstractmethod
    def classify_text(
        self,
        text: Union[str, List[str]],
        model_name: str = "text_classifier",
        labels: Optional[List[str]] = None,
    ) -> List[Dict[str, float]]:
        """
        Classify text into predefined categories.

        Args:
            text: Input text or list of texts
            model_name: Name of classification model
            labels: List of possible labels (optional)

        Returns:
            Classification probabilities for each label
        """
        pass

    @abstractmethod
    def embed_text(
        self, text: Union[str, List[str]], model_name: str = "chemical_embeddings"
    ) -> np.ndarray:
        """
        Generate text embeddings.

        Args:
            text: Input text or list of texts
            model_name: Name of embedding model

        Returns:
            Text embeddings as numpy array
        """
        pass

    def batch_process(
        self, texts: List[str], operations: List[str], batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of texts with multiple operations.

        Args:
            texts: List of input texts
            operations: List of operations to perform
            batch_size: Batch size for processing

        Returns:
            List of results for each text
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_results = []

            for text in batch_texts:
                text_result = {"text": text}

                if "entities" in operations:
                    text_result["entities"] = self.extract_entities(text)

                if "relationships" in operations:
                    entities = text_result.get("entities")
                    text_result["relationships"] = self.extract_relationships(
                        text, entities
                    )

                if "structure" in operations:
                    text_result["structure"] = self.analyze_document_structure(text)

                if "classification" in operations:
                    text_result["classification"] = self.classify_text(text)

                if "embeddings" in operations:
                    text_result["embeddings"] = self.embed_text(text)

                batch_results.append(text_result)

            results.extend(batch_results)

        return results

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a loaded model."""
        model = self._models.get(model_name)
        if model is None:
            return {"loaded": False}

        return {
            "loaded": True,
            "model_type": type(model).__name__,
            "backend": self.backend_type.value,
            "device": self.device,
        }

    def list_loaded_models(self) -> List[str]:
        """List all loaded model names."""
        return list(self._models.keys())

    def unload_model(self, model_name: str) -> bool:
        """Unload a model to free memory."""
        if model_name in self._models:
            del self._models[model_name]
            return True
        return False

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            "backend": self.backend_type.value,
            "device": self.device,
            "loaded_models": len(self._models),
            "loaded_tokenizers": len(self._tokenizers),
        }
