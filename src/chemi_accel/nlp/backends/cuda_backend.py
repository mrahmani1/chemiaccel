"""
CUDA-based NLP Backend

GPU-accelerated NLP processing using PyTorch and CUDA for chemical
entity recognition and information extraction.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    from transformers import (
        AutoModel,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoTokenizer,
        pipeline,
    )

    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None
    nn = None
    AutoTokenizer = None
    AutoModel = None
    AutoModelForTokenClassification = None
    AutoModelForSequenceClassification = None
    pipeline = None
    CUDA_AVAILABLE = False

from .base import (
    BackendType,
    DocumentStructure,
    EntityExtraction,
    NLPBackend,
    RelationshipExtraction,
)

logger = logging.getLogger(__name__)


class CudaNLPBackend(NLPBackend):
    """
    CUDA-accelerated NLP backend using PyTorch and Transformers.

    Provides GPU-accelerated chemical entity recognition, relationship extraction,
    and document processing capabilities.
    """

    def __init__(self, device: Optional[str] = None):
        if not torch:
            raise ImportError("PyTorch is required for CUDA backend")

        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA is not available")

        super().__init__(device)

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Pre-defined model configurations
        self.model_configs = {
            "chemical_ner": {
                "model_name": "allenai/scibert_scivocab_uncased",
                "task": "token-classification",
                "labels": [
                    "O",
                    "B-CHEMICAL",
                    "I-CHEMICAL",
                    "B-REACTION",
                    "I-REACTION",
                    "B-PROPERTY",
                    "I-PROPERTY",
                    "B-VALUE",
                    "I-VALUE",
                ],
            },
            "relation_extraction": {
                "model_name": "bert-base-uncased",
                "task": "sequence-classification",
                "labels": [
                    "NO_RELATION",
                    "BINDS_TO",
                    "INHIBITS",
                    "ACTIVATES",
                    "SIMILAR_TO",
                ],
            },
            "text_classifier": {
                "model_name": "allenai/scibert_scivocab_uncased",
                "task": "sequence-classification",
                "labels": ["SYNTHESIS", "ANALYSIS", "PROPERTY", "BIOACTIVITY", "OTHER"],
            },
            "chemical_embeddings": {
                "model_name": "allenai/scibert_scivocab_uncased",
                "task": "feature-extraction",
            },
        }

        logger.info(f"Initialized CUDA NLP backend on device: {self.device}")

    def _get_backend_type(self) -> BackendType:
        return BackendType.CUDA

    def load_model(self, model_name: str, model_type: str = "transformer") -> Any:
        """Load a pre-trained model onto GPU."""
        if model_name in self._models:
            return self._models[model_name]

        config = self.model_configs.get(model_name, {})
        huggingface_name = config.get("model_name", model_name)
        task = config.get("task", "feature-extraction")

        try:
            if task == "token-classification":
                model = AutoModelForTokenClassification.from_pretrained(
                    huggingface_name
                )
            elif task == "sequence-classification":
                model = AutoModelForSequenceClassification.from_pretrained(
                    huggingface_name
                )
            else:
                model = AutoModel.from_pretrained(huggingface_name)

            model = model.to(self.device)
            model.eval()

            self._models[model_name] = model
            logger.info(f"Loaded model {model_name} on {self.device}")

            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def load_tokenizer(self, tokenizer_name: str) -> Any:
        """Load a tokenizer."""
        if tokenizer_name in self._tokenizers:
            return self._tokenizers[tokenizer_name]

        config = self.model_configs.get(tokenizer_name, {})
        huggingface_name = config.get("model_name", tokenizer_name)

        try:
            tokenizer = AutoTokenizer.from_pretrained(huggingface_name)
            self._tokenizers[tokenizer_name] = tokenizer
            return tokenizer

        except Exception as e:
            logger.error(f"Failed to load tokenizer {tokenizer_name}: {e}")
            raise

    def tokenize_text(
        self, text: Union[str, List[str]], tokenizer_name: str = "chemical_ner"
    ) -> Dict[str, Any]:
        """Tokenize input text using specified tokenizer."""
        tokenizer = self.load_tokenizer(tokenizer_name)

        if isinstance(text, str):
            text = [text]

        # Tokenize with padding and truncation
        tokens = tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )

        # Move to device
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        return tokens

    def extract_entities(
        self, text: Union[str, List[str]], model_name: str = "chemical_ner"
    ) -> List[EntityExtraction]:
        """Extract chemical entities using CUDA-accelerated NER model."""
        if isinstance(text, str):
            text = [text]

        model = self.load_model(model_name)
        tokenizer = self.load_tokenizer(model_name)
        config = self.model_configs.get(model_name, {})
        labels = config.get("labels", ["O", "B-CHEMICAL", "I-CHEMICAL"])

        all_entities = []

        with torch.no_grad():
            for single_text in text:
                # Tokenize
                tokens = tokenizer(
                    single_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    return_offsets_mapping=True,
                )

                input_ids = tokens["input_ids"].to(self.device)
                attention_mask = tokens["attention_mask"].to(self.device)
                offset_mapping = tokens["offset_mapping"][0]

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_labels = torch.argmax(predictions, dim=-1)[0]

                # Convert predictions to entities
                entities = self._convert_predictions_to_entities(
                    single_text, predicted_labels, offset_mapping, labels
                )
                all_entities.extend(entities)

        return all_entities

    def _convert_predictions_to_entities(
        self,
        text: str,
        predictions: torch.Tensor,
        offset_mapping: torch.Tensor,
        labels: List[str],
    ) -> List[EntityExtraction]:
        """Convert model predictions to EntityExtraction objects."""
        entities = []
        current_entity = None

        for i, (pred_idx, (start, end)) in enumerate(zip(predictions, offset_mapping)):
            if start == 0 and end == 0:  # Special tokens
                continue

            label = labels[pred_idx.item()]

            if label.startswith("B-"):
                # Start of new entity
                if current_entity:
                    entities.append(current_entity)

                entity_type = label[2:]
                current_entity = EntityExtraction(
                    text=text[start:end],
                    label=entity_type,
                    start_idx=start,
                    end_idx=end,
                    confidence=0.9,  # Simplified confidence
                    properties={},
                )

            elif label.startswith("I-") and current_entity:
                # Continue current entity
                entity_type = label[2:]
                if entity_type == current_entity.label:
                    current_entity.text = text[current_entity.start_idx : end]
                    current_entity.end_idx = end

        if current_entity:
            entities.append(current_entity)

        return entities

    def extract_relationships(
        self,
        text: Union[str, List[str]],
        entities: Optional[List[EntityExtraction]] = None,
        model_name: str = "relation_extraction",
    ) -> List[RelationshipExtraction]:
        """Extract relationships between entities."""
        if isinstance(text, str):
            text = [text]

        if entities is None:
            entities = self.extract_entities(text)

        model = self.load_model(model_name)
        tokenizer = self.load_tokenizer(model_name)
        config = self.model_configs.get(model_name, {})
        relation_labels = config.get("labels", ["NO_RELATION", "BINDS_TO", "INHIBITS"])

        relationships = []

        # Extract relationships between entity pairs
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i + 1 :], i + 1):
                # Create input for relation classification
                relation_text = f"{entity1.text} [SEP] {entity2.text}"

                with torch.no_grad():
                    tokens = tokenizer(
                        relation_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=128,
                    )

                    input_ids = tokens["input_ids"].to(self.device)
                    attention_mask = tokens["attention_mask"].to(self.device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    predicted_label_idx = torch.argmax(predictions, dim=-1).item()
                    confidence = predictions[0][predicted_label_idx].item()

                    relation_type = relation_labels[predicted_label_idx]

                    if relation_type != "NO_RELATION" and confidence > 0.5:
                        relationships.append(
                            RelationshipExtraction(
                                source_entity=entity1,
                                target_entity=entity2,
                                relation_type=relation_type,
                                confidence=confidence,
                                context=relation_text,
                                properties={},
                            )
                        )

        return relationships

    def analyze_document_structure(self, text: str) -> DocumentStructure:
        """Analyze document structure using pattern matching and ML."""
        # Simplified implementation - would use more sophisticated models in practice
        sections = []
        tables = []
        figures = []
        references = []

        lines = text.split("\n")

        for i, line in enumerate(lines):
            line = line.strip()

            # Detect sections
            if line.isupper() and len(line) < 100:
                sections.append(
                    {"title": line, "start_line": i, "type": "section_header"}
                )

            # Detect tables (simplified)
            if "table" in line.lower() or "|" in line:
                tables.append({"content": line, "line": i, "type": "table"})

            # Detect figures
            if "figure" in line.lower() or "fig." in line.lower():
                figures.append({"content": line, "line": i, "type": "figure"})

            # Detect references
            if line.startswith("[") and "]" in line[:10]:
                references.append({"content": line, "line": i, "type": "reference"})

        return DocumentStructure(
            sections=sections,
            tables=tables,
            figures=figures,
            references=references,
            metadata={"total_lines": len(lines)},
        )

    def extract_table_data(self, table_text: str) -> Dict[str, Any]:
        """Extract structured data from table text."""
        # Simplified table parsing - would use more sophisticated methods
        lines = table_text.strip().split("\n")

        headers = []
        rows = []

        for line in lines:
            if "|" in line:
                columns = [col.strip() for col in line.split("|")]
                if not headers and columns:
                    headers = columns
                else:
                    rows.append(columns)

        return {
            "headers": headers,
            "rows": rows,
            "num_columns": len(headers),
            "num_rows": len(rows),
        }

    def classify_text(
        self,
        text: Union[str, List[str]],
        model_name: str = "text_classifier",
        labels: Optional[List[str]] = None,
    ) -> List[Dict[str, float]]:
        """Classify text into predefined categories."""
        if isinstance(text, str):
            text = [text]

        model = self.load_model(model_name)
        tokenizer = self.load_tokenizer(model_name)
        config = self.model_configs.get(model_name, {})
        class_labels = labels or config.get(
            "labels", ["CHEMICAL", "BIOLOGICAL", "OTHER"]
        )

        results = []

        with torch.no_grad():
            for single_text in text:
                tokens = tokenizer(
                    single_text, return_tensors="pt", truncation=True, max_length=512
                )

                input_ids = tokens["input_ids"].to(self.device)
                attention_mask = tokens["attention_mask"].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

                result = {
                    label: predictions[i].item() for i, label in enumerate(class_labels)
                }
                results.append(result)

        return results

    def embed_text(
        self, text: Union[str, List[str]], model_name: str = "chemical_embeddings"
    ) -> np.ndarray:
        """Generate text embeddings using transformer models."""
        if isinstance(text, str):
            text = [text]

        model = self.load_model(model_name)
        tokenizer = self.load_tokenizer(model_name)

        embeddings = []

        with torch.no_grad():
            for single_text in text:
                tokens = tokenizer(
                    single_text, return_tensors="pt", truncation=True, max_length=512
                )

                input_ids = tokens["input_ids"].to(self.device)
                attention_mask = tokens["attention_mask"].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                # Use mean pooling of last hidden states
                last_hidden = outputs.last_hidden_state
                attention_mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                )
                sum_embeddings = torch.sum(last_hidden * attention_mask_expanded, 1)
                sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
                embedding = sum_embeddings / sum_mask

                embeddings.append(embedding.cpu().numpy())

        return np.vstack(embeddings)

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get GPU memory usage statistics."""
        base_stats = super().get_memory_usage()

        if torch.cuda.is_available():
            gpu_stats = {
                "gpu_memory_allocated": torch.cuda.memory_allocated(self.device),
                "gpu_memory_reserved": torch.cuda.memory_reserved(self.device),
                "gpu_memory_free": torch.cuda.get_device_properties(
                    self.device
                ).total_memory
                - torch.cuda.memory_reserved(self.device),
            }
            base_stats.update(gpu_stats)

        return base_stats
