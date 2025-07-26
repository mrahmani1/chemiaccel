"""
Rapids-based NLP Backend

GPU-accelerated NLP processing using Rapids ecosystem (cuDF, cuPy, cuML)
for high-performance chemical entity recognition and text processing.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import cudf
    import cuml
    import cupy as cp
    from cuml.cluster import DBSCAN
    from cuml.ensemble import RandomForestClassifier
    from cuml.feature_extraction.text import TfidfVectorizer
    from cuml.linear_model import LogisticRegression

    RAPIDS_AVAILABLE = True
except ImportError:
    cudf = None
    cp = None
    cuml = None
    TfidfVectorizer = None
    DBSCAN = None
    LogisticRegression = None
    RandomForestClassifier = None
    RAPIDS_AVAILABLE = False

try:
    from transformers import AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer = None
    TRANSFORMERS_AVAILABLE = False

from .base import (
    BackendType,
    DocumentStructure,
    EntityExtraction,
    NLPBackend,
    RelationshipExtraction,
)

logger = logging.getLogger(__name__)


class RapidsNLPBackend(NLPBackend):
    """
    Rapids-accelerated NLP backend using cuDF and cuML.

    Provides GPU-accelerated text processing, feature extraction, and
    machine learning for chemical entity recognition.
    """

    def __init__(self, device: Optional[str] = None):
        if not RAPIDS_AVAILABLE:
            raise ImportError("Rapids (cuDF, cuML) is required for Rapids backend")

        super().__init__(device)

        # Default to GPU 0
        if device is None:
            self.device = "cuda:0"

        # Initialize Rapids memory pool
        try:
            import rmm

            rmm.reinitialize(
                pool_allocator=True, managed_memory=False, initial_pool_size="1GB"
            )
        except ImportError:
            logger.warning("RMM not available, using default memory management")

        # Model configurations for Rapids-based models
        self.model_configs = {
            "chemical_ner": {
                "vectorizer": "tfidf",
                "classifier": "logistic_regression",
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
                "ngram_range": (1, 3),
                "max_features": 10000,
            },
            "relation_extraction": {
                "vectorizer": "tfidf",
                "classifier": "random_forest",
                "labels": [
                    "NO_RELATION",
                    "BINDS_TO",
                    "INHIBITS",
                    "ACTIVATES",
                    "SIMILAR_TO",
                ],
                "ngram_range": (1, 2),
                "max_features": 5000,
            },
            "text_classifier": {
                "vectorizer": "tfidf",
                "classifier": "logistic_regression",
                "labels": ["SYNTHESIS", "ANALYSIS", "PROPERTY", "BIOACTIVITY", "OTHER"],
                "ngram_range": (1, 2),
                "max_features": 8000,
            },
        }

        # Chemical entity patterns for rule-based extraction
        self.chemical_patterns = [
            r"\b[A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?\d*)*\b",  # Chemical formulas
            r"\b\d+[A-Za-z]+\d*\b",  # Compound names with numbers
            r"\b[A-Z]{2,}[-\d]*\b",  # Abbreviations
            r"\b\w+yl\b|\b\w+ine\b|\b\w+ide\b|\b\w+ate\b",  # Chemical suffixes
        ]

        logger.info(f"Initialized Rapids NLP backend on device: {self.device}")

    def _get_backend_type(self) -> BackendType:
        return BackendType.RAPIDS

    def load_model(self, model_name: str, model_type: str = "ml") -> Any:
        """Load or initialize a Rapids-based ML model."""
        if model_name in self._models:
            return self._models[model_name]

        config = self.model_configs.get(model_name, self.model_configs["chemical_ner"])

        try:
            # Initialize vectorizer
            vectorizer = TfidfVectorizer(
                ngram_range=config["ngram_range"],
                max_features=config["max_features"],
                stop_words="english",
            )

            # Initialize classifier
            classifier_type = config["classifier"]
            if classifier_type == "logistic_regression":
                classifier = LogisticRegression(max_iter=1000)
            elif classifier_type == "random_forest":
                classifier = RandomForestClassifier(n_estimators=100, max_depth=10)
            else:
                classifier = LogisticRegression(max_iter=1000)

            model_dict = {
                "vectorizer": vectorizer,
                "classifier": classifier,
                "config": config,
                "trained": False,
            }

            self._models[model_name] = model_dict
            logger.info(f"Initialized Rapids model {model_name}")

            return model_dict

        except Exception as e:
            logger.error(f"Failed to load Rapids model {model_name}: {e}")
            raise

    def load_tokenizer(self, tokenizer_name: str) -> Any:
        """Load a tokenizer (fallback to HuggingFace if available)."""
        if tokenizer_name in self._tokenizers:
            return self._tokenizers[tokenizer_name]

        if TRANSFORMERS_AVAILABLE:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    "allenai/scibert_scivocab_uncased"
                )
                self._tokenizers[tokenizer_name] = tokenizer
                return tokenizer
            except Exception:
                pass

        # Fallback to simple tokenizer
        tokenizer = self._create_simple_tokenizer()
        self._tokenizers[tokenizer_name] = tokenizer
        return tokenizer

    def _create_simple_tokenizer(self):
        """Create a simple tokenizer for basic text processing."""
        import re

        class SimpleTokenizer:
            def __init__(self):
                self.pattern = re.compile(r"\b\w+\b")

            def tokenize(self, text):
                return self.pattern.findall(text.lower())

            def __call__(self, text, **kwargs):
                if isinstance(text, str):
                    text = [text]

                tokenized = [self.tokenize(t) for t in text]
                return {"input_ids": tokenized, "tokens": tokenized}

        return SimpleTokenizer()

    def tokenize_text(
        self, text: Union[str, List[str]], tokenizer_name: str = "chemical_ner"
    ) -> Dict[str, Any]:
        """Tokenize input text using Rapids-optimized processing."""
        tokenizer = self.load_tokenizer(tokenizer_name)

        if isinstance(text, str):
            text = [text]

        # Convert to cuDF Series for GPU processing
        text_series = cudf.Series(text)

        # Basic tokenization using cuDF string operations
        tokens = []
        for t in text:
            # Simple tokenization - split on whitespace and punctuation
            token_list = cudf.Series([t]).str.split().to_pandas().iloc[0]
            if token_list:
                tokens.append(token_list)
            else:
                tokens.append([])

        return {"input_ids": tokens, "tokens": tokens, "text_series": text_series}

    def extract_entities(
        self, text: Union[str, List[str]], model_name: str = "chemical_ner"
    ) -> List[EntityExtraction]:
        """Extract chemical entities using Rapids-accelerated ML models."""
        if isinstance(text, str):
            text = [text]

        # Convert to cuDF for GPU processing
        text_df = cudf.DataFrame({"text": text})

        # Rule-based entity extraction using cuDF string operations
        entities = []

        for idx, single_text in enumerate(text):
            text_entities = self._extract_entities_rule_based(single_text)
            entities.extend(text_entities)

        # If we have a trained ML model, use it for additional extraction
        model_dict = self.load_model(model_name)
        if model_dict.get("trained", False):
            ml_entities = self._extract_entities_ml(text, model_dict)
            entities.extend(ml_entities)

        return entities

    def _extract_entities_rule_based(self, text: str) -> List[EntityExtraction]:
        """Extract entities using rule-based patterns with cuDF."""
        import re

        entities = []

        for pattern in self.chemical_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entity = EntityExtraction(
                    text=match.group(),
                    label="CHEMICAL",
                    start_idx=match.start(),
                    end_idx=match.end(),
                    confidence=0.8,  # Rule-based confidence
                    properties={"method": "rule_based", "pattern": pattern},
                )
                entities.append(entity)

        return entities

    def _extract_entities_ml(
        self, texts: List[str], model_dict: Dict[str, Any]
    ) -> List[EntityExtraction]:
        """Extract entities using trained ML model."""
        vectorizer = model_dict["vectorizer"]
        classifier = model_dict["classifier"]
        labels = model_dict["config"]["labels"]

        # Vectorize texts
        text_vectors = vectorizer.transform(texts)

        # Predict
        predictions = classifier.predict(text_vectors)
        probabilities = classifier.predict_proba(text_vectors)

        entities = []

        for text_idx, (text, pred, probs) in enumerate(
            zip(texts, predictions, probabilities)
        ):
            # Convert predictions to entities (simplified)
            if pred > 0:  # Non-O label
                label = labels[pred] if pred < len(labels) else "UNKNOWN"
                confidence = float(cp.max(probs))

                entity = EntityExtraction(
                    text=text[:50],  # Simplified - take first 50 chars
                    label=label,
                    start_idx=0,
                    end_idx=min(50, len(text)),
                    confidence=confidence,
                    properties={"method": "ml_model"},
                )
                entities.append(entity)

        return entities

    def extract_relationships(
        self,
        text: Union[str, List[str]],
        entities: Optional[List[EntityExtraction]] = None,
        model_name: str = "relation_extraction",
    ) -> List[RelationshipExtraction]:
        """Extract relationships using Rapids ML models."""
        if isinstance(text, str):
            text = [text]

        if entities is None:
            entities = self.extract_entities(text)

        model_dict = self.load_model(model_name)
        relationships = []

        # Extract relationships between entity pairs
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i + 1 :], i + 1):
                # Create feature vector for relationship
                relation_text = f"{entity1.text} {entity2.text}"

                # Use TF-IDF vectorization
                vectorizer = model_dict["vectorizer"]

                # Simple relationship detection based on proximity and context
                confidence = self._calculate_relationship_confidence(
                    entity1, entity2, text[0]
                )

                if confidence > 0.5:
                    # Simplified relationship type assignment
                    relation_type = (
                        "BINDS_TO" if "bind" in text[0].lower() else "SIMILAR_TO"
                    )

                    relationships.append(
                        RelationshipExtraction(
                            source_entity=entity1,
                            target_entity=entity2,
                            relation_type=relation_type,
                            confidence=confidence,
                            context=relation_text,
                            properties={"method": "rapids_ml"},
                        )
                    )

        return relationships

    def _calculate_relationship_confidence(
        self, entity1: EntityExtraction, entity2: EntityExtraction, text: str
    ) -> float:
        """Calculate relationship confidence based on entity proximity and context."""
        # Simple proximity-based confidence
        distance = abs(entity1.start_idx - entity2.start_idx)
        max_distance = len(text)

        # Closer entities have higher confidence
        proximity_score = 1.0 - (distance / max_distance)

        # Context-based scoring
        context_words = ["binds", "inhibits", "activates", "similar", "interacts"]
        context_score = 0.0

        for word in context_words:
            if word in text.lower():
                context_score += 0.2

        return min(proximity_score + context_score, 1.0)

    def analyze_document_structure(self, text: str) -> DocumentStructure:
        """Analyze document structure using cuDF text processing."""
        # Convert to cuDF Series for GPU processing
        lines_series = cudf.Series(text.split("\n"))

        sections = []
        tables = []
        figures = []
        references = []

        # Use cuDF string operations for pattern matching
        for i, line in enumerate(lines_series.to_pandas()):
            line = line.strip()

            # Detect sections (uppercase lines)
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
            metadata={"total_lines": len(lines_series)},
        )

    def extract_table_data(self, table_text: str) -> Dict[str, Any]:
        """Extract structured data from table text using cuDF."""
        lines = table_text.strip().split("\n")

        # Convert to cuDF for processing
        lines_df = cudf.DataFrame({"lines": lines})

        # Filter lines that contain table separators
        table_lines = lines_df[lines_df["lines"].str.contains("\\|", na=False)]

        headers = []
        rows = []

        for line in table_lines["lines"].to_pandas():
            if "|" in line:
                columns = [col.strip() for col in line.split("|")]
                if not headers and columns:
                    headers = columns
                else:
                    rows.append(columns)

        # Create cuDF DataFrame from extracted table data
        if headers and rows:
            try:
                table_df = cudf.DataFrame(dict(zip(headers, zip(*rows))))
                return {
                    "headers": headers,
                    "rows": rows,
                    "num_columns": len(headers),
                    "num_rows": len(rows),
                    "cudf_table": table_df,
                }
            except Exception:
                pass

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
        """Classify text using Rapids ML models."""
        if isinstance(text, str):
            text = [text]

        model_dict = self.load_model(model_name)
        config = model_dict["config"]
        class_labels = labels or config["labels"]

        # Simple rule-based classification for demo
        results = []

        for single_text in text:
            text_lower = single_text.lower()
            scores = {}

            # Rule-based scoring
            if any(
                word in text_lower for word in ["synthesis", "prepare", "synthesize"]
            ):
                scores["SYNTHESIS"] = 0.8
            elif any(
                word in text_lower for word in ["analysis", "analyze", "characterize"]
            ):
                scores["ANALYSIS"] = 0.8
            elif any(
                word in text_lower
                for word in ["property", "properties", "characteristic"]
            ):
                scores["PROPERTY"] = 0.8
            elif any(
                word in text_lower
                for word in ["bioactivity", "activity", "inhibit", "bind"]
            ):
                scores["BIOACTIVITY"] = 0.8
            else:
                scores["OTHER"] = 0.6

            # Normalize scores
            total = sum(scores.values())
            if total > 0:
                scores = {k: v / total for k, v in scores.items()}

            # Fill missing labels
            for label in class_labels:
                if label not in scores:
                    scores[label] = 0.0

            results.append(scores)

        return results

    def embed_text(
        self, text: Union[str, List[str]], model_name: str = "chemical_embeddings"
    ) -> np.ndarray:
        """Generate text embeddings using Rapids TF-IDF."""
        if isinstance(text, str):
            text = [text]

        # Use TF-IDF vectorization as embeddings
        vectorizer = TfidfVectorizer(max_features=768)  # Fixed size embeddings

        try:
            embeddings = vectorizer.fit_transform(text)
            # Convert to dense numpy array
            if hasattr(embeddings, "toarray"):
                return embeddings.toarray()
            else:
                return cp.asnumpy(embeddings.todense())
        except Exception:
            # Fallback to simple bag-of-words
            vocab = set()
            for t in text:
                vocab.update(t.lower().split())

            vocab = list(vocab)[:768]  # Limit vocabulary size
            embeddings = np.zeros((len(text), len(vocab)))

            for i, t in enumerate(text):
                words = t.lower().split()
                for word in words:
                    if word in vocab:
                        embeddings[i, vocab.index(word)] += 1

            return embeddings

    def train_model(
        self, model_name: str, texts: List[str], labels: List[str]
    ) -> Dict[str, Any]:
        """Train a Rapids ML model on provided data."""
        model_dict = self.load_model(model_name)

        vectorizer = model_dict["vectorizer"]
        classifier = model_dict["classifier"]

        try:
            # Vectorize training texts
            X = vectorizer.fit_transform(texts)

            # Convert labels to cuDF Series
            y = cudf.Series(labels)

            # Train classifier
            classifier.fit(X, y)

            # Mark as trained
            model_dict["trained"] = True

            return {
                "model_name": model_name,
                "training_samples": len(texts),
                "unique_labels": len(set(labels)),
                "success": True,
            }

        except Exception as e:
            logger.error(f"Failed to train model {model_name}: {e}")
            return {"model_name": model_name, "success": False, "error": str(e)}

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get Rapids memory usage statistics."""
        base_stats = super().get_memory_usage()

        try:
            # Get GPU memory info using cuPy
            gpu_info = {
                "gpu_memory_used": cp.cuda.runtime.memGetInfo()[0],
                "gpu_memory_total": cp.cuda.runtime.memGetInfo()[1],
                "cudf_version": cudf.__version__,
                "cuml_version": cuml.__version__,
            }
            base_stats.update(gpu_info)
        except Exception:
            pass

        return base_stats
