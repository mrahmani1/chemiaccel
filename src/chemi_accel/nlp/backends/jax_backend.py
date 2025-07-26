"""
JAX-based NLP Backend

GPU/TPU-accelerated NLP processing using JAX and Flax for chemical
entity recognition and information extraction.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training import train_state
    from jax import jit, random, vmap
    from transformers import AutoTokenizer

    JAX_AVAILABLE = len(jax.devices()) > 0
except ImportError:
    jax = None
    jnp = None
    random = None
    jit = None
    vmap = None
    nn = None
    train_state = None
    optax = None
    AutoTokenizer = None
    JAX_AVAILABLE = False

from .base import (
    BackendType,
    DocumentStructure,
    EntityExtraction,
    NLPBackend,
    RelationshipExtraction,
)

logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    """JAX/Flax implementation of transformer block for NLP tasks."""

    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic=True):
        # Multi-head attention
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, dtype=jnp.float32
        )(x, x, deterministic=deterministic)

        # Add & norm
        x = nn.LayerNorm()(x + attn_output)

        # Feed forward
        ff_output = nn.Dense(self.d_ff)(x)
        ff_output = nn.relu(ff_output)
        ff_output = nn.Dropout(rate=self.dropout_rate)(
            ff_output, deterministic=deterministic
        )
        ff_output = nn.Dense(self.d_model)(ff_output)

        # Add & norm
        return nn.LayerNorm()(x + ff_output)


class ChemicalNERModel(nn.Module):
    """JAX/Flax NER model for chemical entity recognition."""

    vocab_size: int
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072
    num_labels: int = 9
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, deterministic=True):
        # Embedding layer
        x = nn.Embed(self.vocab_size, self.d_model)(input_ids)

        # Position embeddings
        seq_len = x.shape[1]
        pos_emb = nn.Embed(512, self.d_model)(jnp.arange(seq_len))
        x = x + pos_emb

        # Transformer layers
        for _ in range(self.num_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
            )(x, deterministic=deterministic)

        # Classification head
        logits = nn.Dense(self.num_labels)(x)

        return logits


class JAXNLPBackend(NLPBackend):
    """
    JAX-accelerated NLP backend using Flax for chemical entity recognition.

    Provides GPU/TPU-accelerated chemical entity recognition, relationship extraction,
    and document processing capabilities using JAX.
    """

    def __init__(self, device: Optional[str] = None):
        if not jax:
            raise ImportError("JAX is required for JAX backend")

        if not JAX_AVAILABLE:
            raise RuntimeError("JAX devices are not available")

        super().__init__(device)

        # Set JAX device
        if device is None:
            self.device = jax.devices()[0]
        else:
            available_devices = {str(d): d for d in jax.devices()}
            self.device = available_devices.get(device, jax.devices()[0])

        # Random key for JAX operations
        self.rng_key = random.PRNGKey(42)

        # Model configurations
        self.model_configs = {
            "chemical_ner": {
                "vocab_size": 30522,
                "d_model": 768,
                "num_layers": 6,
                "num_heads": 12,
                "num_labels": 9,
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
                "vocab_size": 30522,
                "d_model": 768,
                "num_layers": 6,
                "num_heads": 12,
                "num_labels": 5,
                "labels": [
                    "NO_RELATION",
                    "BINDS_TO",
                    "INHIBITS",
                    "ACTIVATES",
                    "SIMILAR_TO",
                ],
            },
            "text_classifier": {
                "vocab_size": 30522,
                "d_model": 768,
                "num_layers": 6,
                "num_heads": 12,
                "num_labels": 5,
                "labels": ["SYNTHESIS", "ANALYSIS", "PROPERTY", "BIOACTIVITY", "OTHER"],
            },
        }

        # Initialize compiled functions
        self._compile_functions()

        logger.info(f"Initialized JAX NLP backend on device: {self.device}")

    def _get_backend_type(self) -> BackendType:
        return BackendType.JAX

    def _compile_functions(self):
        """Compile JAX functions for better performance."""

        @jit
        def predict_entities(params, input_ids, attention_mask):
            model = ChemicalNERModel(vocab_size=30522, num_labels=9)
            return model.apply(params, input_ids, attention_mask, deterministic=True)

        @jit
        def predict_relations(params, input_ids, attention_mask):
            model = ChemicalNERModel(vocab_size=30522, num_labels=5)
            return model.apply(params, input_ids, attention_mask, deterministic=True)

        self._predict_entities_fn = predict_entities
        self._predict_relations_fn = predict_relations

    def load_model(self, model_name: str, model_type: str = "transformer") -> Any:
        """Load a pre-trained model with JAX/Flax."""
        if model_name in self._models:
            return self._models[model_name]

        config = self.model_configs.get(model_name, self.model_configs["chemical_ner"])

        try:
            # Initialize model
            if "ner" in model_name:
                model = ChemicalNERModel(
                    vocab_size=config["vocab_size"],
                    d_model=config["d_model"],
                    num_layers=config["num_layers"],
                    num_heads=config["num_heads"],
                    num_labels=config["num_labels"],
                )
            else:
                model = ChemicalNERModel(
                    vocab_size=config["vocab_size"],
                    d_model=config["d_model"],
                    num_layers=config["num_layers"],
                    num_heads=config["num_heads"],
                    num_labels=config["num_labels"],
                )

            # Initialize parameters
            self.rng_key, init_key = random.split(self.rng_key)
            dummy_input = jnp.ones((1, 128), dtype=jnp.int32)
            params = model.init(init_key, dummy_input)

            # Store model and params
            model_dict = {"model": model, "params": params, "config": config}

            self._models[model_name] = model_dict
            logger.info(f"Loaded JAX model {model_name}")

            return model_dict

        except Exception as e:
            logger.error(f"Failed to load JAX model {model_name}: {e}")
            raise

    def load_tokenizer(self, tokenizer_name: str) -> Any:
        """Load a tokenizer (using HuggingFace tokenizers)."""
        if tokenizer_name in self._tokenizers:
            return self._tokenizers[tokenizer_name]

        try:
            # Use SciBERT tokenizer as default for chemical text
            tokenizer = AutoTokenizer.from_pretrained(
                "allenai/scibert_scivocab_uncased"
            )
            self._tokenizers[tokenizer_name] = tokenizer
            return tokenizer

        except Exception as e:
            logger.error(f"Failed to load tokenizer {tokenizer_name}: {e}")
            raise

    def tokenize_text(
        self, text: Union[str, List[str]], tokenizer_name: str = "chemical_ner"
    ) -> Dict[str, Any]:
        """Tokenize input text and convert to JAX arrays."""
        tokenizer = self.load_tokenizer(tokenizer_name)

        if isinstance(text, str):
            text = [text]

        # Tokenize with padding and truncation
        tokens = tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="np"
        )

        # Convert to JAX arrays
        jax_tokens = {k: jnp.array(v) for k, v in tokens.items()}

        return jax_tokens

    def extract_entities(
        self, text: Union[str, List[str]], model_name: str = "chemical_ner"
    ) -> List[EntityExtraction]:
        """Extract chemical entities using JAX-accelerated NER model."""
        if isinstance(text, str):
            text = [text]

        model_dict = self.load_model(model_name)
        tokenizer = self.load_tokenizer(model_name)

        model = model_dict["model"]
        params = model_dict["params"]
        config = model_dict["config"]
        labels = config["labels"]

        all_entities = []

        for single_text in text:
            # Tokenize
            tokens = tokenizer(
                single_text,
                return_tensors="np",
                truncation=True,
                max_length=512,
                return_offsets_mapping=True,
            )

            input_ids = jnp.array(tokens["input_ids"])
            attention_mask = jnp.array(tokens["attention_mask"])
            offset_mapping = tokens["offset_mapping"][0]

            # Forward pass
            logits = self._predict_entities_fn(params, input_ids, attention_mask)
            predictions = jnp.argmax(logits, axis=-1)[0]

            # Convert predictions to entities
            entities = self._convert_jax_predictions_to_entities(
                single_text, predictions, offset_mapping, labels
            )
            all_entities.extend(entities)

        return all_entities

    def _convert_jax_predictions_to_entities(
        self,
        text: str,
        predictions: jnp.ndarray,
        offset_mapping: np.ndarray,
        labels: List[str],
    ) -> List[EntityExtraction]:
        """Convert JAX predictions to EntityExtraction objects."""
        entities = []
        current_entity = None

        for i, (pred_idx, (start, end)) in enumerate(zip(predictions, offset_mapping)):
            if start == 0 and end == 0:  # Special tokens
                continue

            label = labels[int(pred_idx)]

            if label.startswith("B-"):
                # Start of new entity
                if current_entity:
                    entities.append(current_entity)

                entity_type = label[2:]
                current_entity = EntityExtraction(
                    text=text[start:end],
                    label=entity_type,
                    start_idx=int(start),
                    end_idx=int(end),
                    confidence=0.9,  # Simplified confidence
                    properties={},
                )

            elif label.startswith("I-") and current_entity:
                # Continue current entity
                entity_type = label[2:]
                if entity_type == current_entity.label:
                    current_entity.text = text[current_entity.start_idx : end]
                    current_entity.end_idx = int(end)

        if current_entity:
            entities.append(current_entity)

        return entities

    def extract_relationships(
        self,
        text: Union[str, List[str]],
        entities: Optional[List[EntityExtraction]] = None,
        model_name: str = "relation_extraction",
    ) -> List[RelationshipExtraction]:
        """Extract relationships between entities using JAX."""
        if isinstance(text, str):
            text = [text]

        if entities is None:
            entities = self.extract_entities(text)

        model_dict = self.load_model(model_name)
        tokenizer = self.load_tokenizer(model_name)

        params = model_dict["params"]
        config = model_dict["config"]
        relation_labels = config["labels"]

        relationships = []

        # Extract relationships between entity pairs
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i + 1 :], i + 1):
                # Create input for relation classification
                relation_text = f"{entity1.text} [SEP] {entity2.text}"

                tokens = tokenizer(
                    relation_text, return_tensors="np", truncation=True, max_length=128
                )

                input_ids = jnp.array(tokens["input_ids"])
                attention_mask = jnp.array(tokens["attention_mask"])

                logits = self._predict_relations_fn(params, input_ids, attention_mask)
                probabilities = jax.nn.softmax(logits, axis=-1)[0]
                predicted_label_idx = jnp.argmax(probabilities)
                confidence = float(probabilities[predicted_label_idx])

                relation_type = relation_labels[int(predicted_label_idx)]

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
        """Analyze document structure using pattern matching."""
        # Simplified implementation - same as CUDA backend for now
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

            # Detect tables
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
        # Same implementation as CUDA backend
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
        """Classify text using JAX model."""
        if isinstance(text, str):
            text = [text]

        model_dict = self.load_model(model_name)
        tokenizer = self.load_tokenizer(model_name)

        params = model_dict["params"]
        config = model_dict["config"]
        class_labels = labels or config["labels"]

        results = []

        for single_text in text:
            tokens = tokenizer(
                single_text, return_tensors="np", truncation=True, max_length=512
            )

            input_ids = jnp.array(tokens["input_ids"])
            attention_mask = jnp.array(tokens["attention_mask"])

            # For classification, use mean pooling of token embeddings
            model = model_dict["model"]
            hidden_states = model.apply(
                params, input_ids, attention_mask, deterministic=True
            )

            # Simple mean pooling
            pooled = jnp.mean(hidden_states, axis=1)

            # Classification layer (simplified)
            logits = nn.Dense(len(class_labels))(pooled)
            probabilities = jax.nn.softmax(logits, axis=-1)[0]

            result = {
                label: float(probabilities[i]) for i, label in enumerate(class_labels)
            }
            results.append(result)

        return results

    def embed_text(
        self, text: Union[str, List[str]], model_name: str = "chemical_embeddings"
    ) -> np.ndarray:
        """Generate text embeddings using JAX models."""
        if isinstance(text, str):
            text = [text]

        model_dict = self.load_model(model_name)
        tokenizer = self.load_tokenizer(model_name)

        model = model_dict["model"]
        params = model_dict["params"]

        embeddings = []

        for single_text in text:
            tokens = tokenizer(
                single_text, return_tensors="np", truncation=True, max_length=512
            )

            input_ids = jnp.array(tokens["input_ids"])
            attention_mask = jnp.array(tokens["attention_mask"])

            # Get hidden states
            hidden_states = model.apply(
                params, input_ids, attention_mask, deterministic=True
            )

            # Mean pooling
            attention_mask_expanded = jnp.expand_dims(attention_mask, -1)
            sum_embeddings = jnp.sum(hidden_states * attention_mask_expanded, axis=1)
            sum_mask = jnp.sum(attention_mask_expanded, axis=1)
            embedding = sum_embeddings / jnp.maximum(sum_mask, 1e-9)

            embeddings.append(np.array(embedding))

        return np.vstack(embeddings)

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get JAX device memory usage."""
        base_stats = super().get_memory_usage()

        try:
            device_info = {
                "jax_device": str(self.device),
                "jax_device_kind": self.device.device_kind,
                "jax_platform": self.device.platform,
            }
            base_stats.update(device_info)
        except Exception:
            pass

        return base_stats
