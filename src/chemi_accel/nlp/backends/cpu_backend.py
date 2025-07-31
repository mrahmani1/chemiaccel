"""
CPU-based NLP Backend

CPU fallback backend for NLP processing when GPU acceleration is not available.
Uses scikit-learn and NLTK for chemical entity recognition and text processing.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
except ImportError:
    TfidfVectorizer = None
    LogisticRegression = None
    RandomForestClassifier = None
    train_test_split = None
    classification_report = None
    SKLEARN_AVAILABLE = False

try:
    import nltk
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    from nltk.tokenize import sent_tokenize, word_tokenize

    NLTK_AVAILABLE = True

    # Download required NLTK data
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger", quiet=True)

except ImportError:
    nltk = None
    word_tokenize = None
    sent_tokenize = None
    ne_chunk = None
    pos_tag = None
    NLTK_AVAILABLE = False

from .base import (
    BackendType,
    DocumentStructure,
    EntityExtraction,
    NLPBackend,
    RelationshipExtraction,
)

logger = logging.getLogger(__name__)


class CPUNLPBackend(NLPBackend):
    """
    CPU-based NLP backend using scikit-learn and NLTK.

    Provides chemical entity recognition, relationship extraction,
    and document processing using CPU-only libraries.
    """

    def __init__(self, device: Optional[str] = None):
        super().__init__(device)
        self.device = "cpu"

        # Chemical entity patterns for rule-based extraction
        self.chemical_patterns = {
            "molecular_formula": r"\b[A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?\d*)*\b",
            "iupac_name": r"\b\d*[a-z]+(?:-\d+[a-z]+)*(?:yl|ine|ide|ate|oic acid)\b",
            "cas_number": r"\b\d{2,7}-\d{2}-\d\b",
            "drug_name": r"\b[A-Z][a-z]+(?:mab|zumab|tinib|pril|olol|pine)\b",
            "chemical_class": r"\b(?:alcohol|ester|ether|aldehyde|ketone|amine|amide|acid)s?\b",
        }

        # Relationship patterns
        self.relationship_patterns = {
            "binds_to": [
                r"\bbind[s]?\s+to\b",
                r"\binteract[s]?\s+with\b",
                r"\bcomplex\s+with\b",
            ],
            "inhibits": [r"\binhibit[s]?\b", r"\bblock[s]?\b", r"\bsuppress[es]?\b"],
            "activates": [
                r"\bactivate[s]?\b",
                r"\bstimulate[s]?\b",
                r"\benhance[s]?\b",
            ],
            "similar_to": [
                r"\bsimilar\s+to\b",
                r"\banalog\s+of\b",
                r"\bresembl[es]?\b",
            ],
        }

        # Model configurations
        self.model_configs = {
            "chemical_ner": {
                "vectorizer_params": {"ngram_range": (1, 3), "max_features": 5000},
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
            },
            "relation_extraction": {
                "vectorizer_params": {"ngram_range": (1, 2), "max_features": 3000},
                "classifier": "random_forest",
                "labels": [
                    "NO_RELATION",
                    "BINDS_TO",
                    "INHIBITS",
                    "ACTIVATES",
                    "SIMILAR_TO",
                ],
            },
            "text_classifier": {
                "vectorizer_params": {"ngram_range": (1, 2), "max_features": 4000},
                "classifier": "logistic_regression",
                "labels": ["SYNTHESIS", "ANALYSIS", "PROPERTY", "BIOACTIVITY", "OTHER"],
            },
        }

        logger.info("Initialized CPU NLP backend")

    def _get_backend_type(self) -> BackendType:
        return BackendType.CPU

    def load_model(self, model_name: str, model_type: str = "sklearn") -> Any:
        """Load or initialize a scikit-learn model."""
        if model_name in self._models:
            return self._models[model_name]

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for CPU backend")

        config = self.model_configs.get(model_name, self.model_configs["chemical_ner"])

        try:
            # Initialize vectorizer
            vectorizer = TfidfVectorizer(**config["vectorizer_params"])

            # Initialize classifier
            classifier_type = config["classifier"]
            if classifier_type == "logistic_regression":
                classifier = LogisticRegression(max_iter=1000, random_state=42)
            elif classifier_type == "random_forest":
                classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                classifier = LogisticRegression(max_iter=1000, random_state=42)

            model_dict = {
                "vectorizer": vectorizer,
                "classifier": classifier,
                "config": config,
                "trained": False,
            }

            self._models[model_name] = model_dict
            logger.info(f"Initialized CPU model {model_name}")

            return model_dict

        except Exception as e:
            logger.error(f"Failed to load CPU model {model_name}: {e}")
            raise

    def load_tokenizer(self, tokenizer_name: str) -> Any:
        """Load a simple tokenizer."""
        if tokenizer_name in self._tokenizers:
            return self._tokenizers[tokenizer_name]

        # Create simple tokenizer
        tokenizer = self._create_simple_tokenizer()
        self._tokenizers[tokenizer_name] = tokenizer
        return tokenizer

    def _create_simple_tokenizer(self):
        """Create a simple word tokenizer."""

        class SimpleTokenizer:
            def __init__(self):
                self.word_pattern = re.compile(r"\b\w+\b")

            def tokenize(self, text):
                if NLTK_AVAILABLE:
                    return word_tokenize(text.lower())
                else:
                    return self.word_pattern.findall(text.lower())

            def __call__(self, text, **kwargs):
                if isinstance(text, str):
                    text = [text]

                tokenized = [self.tokenize(t) for t in text]
                return {"input_ids": tokenized, "tokens": tokenized}

        return SimpleTokenizer()

    def tokenize_text(
        self, text: Union[str, List[str]], tokenizer_name: str = "chemical_ner"
    ) -> Dict[str, Any]:
        """Tokenize input text using simple tokenization."""
        tokenizer = self.load_tokenizer(tokenizer_name)

        if isinstance(text, str):
            text = [text]

        # Tokenize texts
        tokenized = []
        for t in text:
            tokens = tokenizer.tokenize(t)
            tokenized.append(tokens)

        return {"input_ids": tokenized, "tokens": tokenized}

    def extract_entities(
        self, text: Union[str, List[str]], model_name: str = "chemical_ner"
    ) -> List[EntityExtraction]:
        """Extract chemical entities using rule-based patterns."""
        if isinstance(text, str):
            text = [text]

        all_entities = []

        for single_text in text:
            entities = self._extract_entities_rule_based(single_text)

            # Add NLTK-based named entity recognition if available
            if NLTK_AVAILABLE:
                nltk_entities = self._extract_entities_nltk(single_text)
                entities.extend(nltk_entities)

            all_entities.extend(entities)

        # Remove duplicates based on text span
        unique_entities = []
        seen_spans = set()

        for entity in all_entities:
            span = (entity.start_idx, entity.end_idx, entity.text)
            if span not in seen_spans:
                unique_entities.append(entity)
                seen_spans.add(span)

        return unique_entities

    def _extract_entities_rule_based(self, text: str) -> List[EntityExtraction]:
        """Extract entities using chemical patterns."""
        entities = []

        for pattern_name, pattern in self.chemical_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Skip very short matches
                if len(match.group()) < 2:
                    continue

                # Determine entity type based on pattern
                if pattern_name == "molecular_formula":
                    label = "CHEMICAL"
                elif pattern_name == "iupac_name":
                    label = "CHEMICAL"
                elif pattern_name == "cas_number":
                    label = "CHEMICAL"
                elif pattern_name == "drug_name":
                    label = "CHEMICAL"
                else:
                    label = "CHEMICAL"

                entity = EntityExtraction(
                    text=match.group(),
                    label=label,
                    start_idx=match.start(),
                    end_idx=match.end(),
                    confidence=0.7,  # Rule-based confidence
                    properties={"method": "rule_based", "pattern": pattern_name},
                )
                entities.append(entity)

        return entities

    def _extract_entities_nltk(self, text: str) -> List[EntityExtraction]:
        """Extract entities using NLTK named entity recognition."""
        entities = []

        try:
            # Tokenize and tag
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)

            # Find chemical-related entities based on POS tags
            for i, (token, pos) in enumerate(pos_tags):
                # Look for noun phrases that might be chemicals
                if pos in ["NN", "NNS", "NNP", "NNPS"] and len(token) > 2:
                    # Calculate position in original text
                    start_idx = text.lower().find(token.lower())
                    if start_idx != -1:
                        end_idx = start_idx + len(token)

                        entity = EntityExtraction(
                            text=token,
                            label="CHEMICAL",
                            start_idx=start_idx,
                            end_idx=end_idx,
                            confidence=0.5,
                            properties={"method": "nltk_pos", "pos_tag": pos},
                        )
                        entities.append(entity)

        except Exception as e:
            logger.debug(f"NLTK entity extraction failed: {e}")

        return entities

    def extract_relationships(
        self,
        text: Union[str, List[str]],
        entities: Optional[List[EntityExtraction]] = None,
        model_name: str = "relation_extraction",
    ) -> List[RelationshipExtraction]:
        """Extract relationships using pattern matching."""
        if isinstance(text, str):
            text = [text]

        if entities is None:
            entities = self.extract_entities(text)

        relationships = []

        for single_text in text:
            # Extract relationships between entity pairs
            for i, entity1 in enumerate(entities):
                for j, entity2 in enumerate(entities[i + 1 :], i + 1):
                    # Check if entities are in the same text
                    if not (
                        entity1.start_idx < len(single_text)
                        and entity2.start_idx < len(single_text)
                    ):
                        continue

                    # Find relationship patterns between entities
                    relation_type, confidence = self._find_relationship_pattern(
                        entity1, entity2, single_text
                    )

                    if relation_type and confidence > 0.3:
                        relationships.append(
                            RelationshipExtraction(
                                source_entity=entity1,
                                target_entity=entity2,
                                relation_type=relation_type,
                                confidence=confidence,
                                context=single_text[
                                    max(0, entity1.start_idx - 50) : entity2.end_idx
                                    + 50
                                ],
                                properties={"method": "pattern_matching"},
                            )
                        )

        return relationships

    def _find_relationship_pattern(
        self, entity1: EntityExtraction, entity2: EntityExtraction, text: str
    ) -> tuple:
        """Find relationship patterns between two entities."""
        # Get text between entities
        start_idx = min(entity1.end_idx, entity2.end_idx)
        end_idx = max(entity1.start_idx, entity2.start_idx)

        if start_idx >= end_idx:
            return None, 0.0

        between_text = text[start_idx:end_idx].lower()

        # Check for relationship patterns
        for relation_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                if re.search(pattern, between_text):
                    # Calculate confidence based on pattern strength and proximity
                    distance = abs(entity1.start_idx - entity2.start_idx)
                    proximity_score = max(
                        0.1, 1.0 - distance / 1000
                    )  # Closer = higher confidence

                    return relation_type.upper(), min(0.8, proximity_score + 0.3)

        # Check for general co-occurrence
        distance = abs(entity1.start_idx - entity2.start_idx)
        if distance < 100:  # Entities are close
            return "SIMILAR_TO", 0.3

        return None, 0.0

    def analyze_document_structure(self, text: str) -> DocumentStructure:
        """Analyze document structure using pattern matching."""
        lines = text.split("\n")

        sections = []
        tables = []
        figures = []
        references = []

        for i, line in enumerate(lines):
            line = line.strip()

            # Detect sections
            if line.isupper() and len(line) < 100 and len(line) > 3:
                sections.append(
                    {"title": line, "start_line": i, "type": "section_header"}
                )
            elif re.match(r"^\d+\.?\s+[A-Z]", line):  # Numbered sections
                sections.append(
                    {"title": line, "start_line": i, "type": "numbered_section"}
                )

            # Detect tables
            if "table" in line.lower() or "|" in line or "\t" in line:
                tables.append({"content": line, "line": i, "type": "table"})

            # Detect figures
            if re.search(r"\bfig(?:ure)?\s*\d+", line.lower()):
                figures.append({"content": line, "line": i, "type": "figure"})

            # Detect references
            if re.match(r"^\[\d+\]", line) or re.match(
                r"^\d+\.\s+[A-Z].*\(\d{4}\)", line
            ):
                references.append({"content": line, "line": i, "type": "reference"})

        return DocumentStructure(
            sections=sections,
            tables=tables,
            figures=figures,
            references=references,
            metadata={"total_lines": len(lines), "word_count": len(text.split())},
        )

    def extract_table_data(self, table_text: str) -> Dict[str, Any]:
        """Extract structured data from table text."""
        lines = table_text.strip().split("\n")

        headers = []
        rows = []

        for line in lines:
            # Try different separators
            separators = ["|", "\t", "  "]  # Pipe, tab, double space

            for sep in separators:
                if sep in line:
                    columns = [col.strip() for col in line.split(sep) if col.strip()]

                    if len(columns) > 1:
                        if not headers:
                            headers = columns
                        else:
                            rows.append(columns)
                        break

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
        """Classify text using rule-based classification."""
        if isinstance(text, str):
            text = [text]

        config = self.model_configs.get(
            model_name, self.model_configs["text_classifier"]
        )
        class_labels = labels or config["labels"]

        results = []

        # Classification keywords
        classification_keywords = {
            "SYNTHESIS": [
                "synthesis",
                "prepare",
                "synthesize",
                "reaction",
                "yield",
                "product",
            ],
            "ANALYSIS": [
                "analysis",
                "characterization",
                "spectroscopy",
                "nmr",
                "mass spec",
                "hplc",
            ],
            "PROPERTY": [
                "property",
                "properties",
                "melting point",
                "boiling point",
                "solubility",
            ],
            "BIOACTIVITY": [
                "activity",
                "bioactivity",
                "inhibition",
                "binding",
                "ic50",
                "ec50",
            ],
            "OTHER": [],
        }

        for single_text in text:
            text_lower = single_text.lower()
            scores = dict.fromkeys(class_labels, 0.0)

            # Score based on keyword presence
            for label, keywords in classification_keywords.items():
                if label in class_labels:
                    keyword_count = sum(
                        1 for keyword in keywords if keyword in text_lower
                    )
                    scores[label] = min(1.0, keyword_count * 0.3)

            # Normalize scores
            total_score = sum(scores.values())
            if total_score > 0:
                scores = {k: v / total_score for k, v in scores.items()}
            else:
                # Equal probability for all classes if no keywords found
                scores = {k: 1.0 / len(class_labels) for k in class_labels}

            results.append(scores)

        return results

    def embed_text(
        self, text: Union[str, List[str]], model_name: str = "chemical_embeddings"
    ) -> np.ndarray:
        """Generate text embeddings using TF-IDF."""
        if isinstance(text, str):
            text = [text]

        if not SKLEARN_AVAILABLE:
            # Fallback to simple bag-of-words
            return self._simple_embeddings(text)

        # Use TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=300, stop_words="english")

        try:
            embeddings = vectorizer.fit_transform(text)
            return embeddings.toarray()
        except Exception:
            return self._simple_embeddings(text)

    def _simple_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create simple bag-of-words embeddings."""
        # Create vocabulary
        vocab = set()
        for text in texts:
            words = re.findall(r"\b\w+\b", text.lower())
            vocab.update(words)

        vocab = list(vocab)[:300]  # Limit vocabulary size

        # Create embeddings
        embeddings = np.zeros((len(texts), len(vocab)))

        for i, text in enumerate(texts):
            words = re.findall(r"\b\w+\b", text.lower())
            for word in words:
                if word in vocab:
                    embeddings[i, vocab.index(word)] += 1

        return embeddings

    def train_model(
        self, model_name: str, texts: List[str], labels: List[str]
    ) -> Dict[str, Any]:
        """Train a CPU model on provided data."""
        if not SKLEARN_AVAILABLE:
            return {"success": False, "error": "scikit-learn not available"}

        model_dict = self.load_model(model_name)

        vectorizer = model_dict["vectorizer"]
        classifier = model_dict["classifier"]

        try:
            # Vectorize training texts
            X = vectorizer.fit_transform(texts)
            y = np.array(labels)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train classifier
            classifier.fit(X_train, y_train)

            # Evaluate
            train_score = classifier.score(X_train, y_train)
            test_score = classifier.score(X_test, y_test)

            # Mark as trained
            model_dict["trained"] = True

            return {
                "model_name": model_name,
                "training_samples": len(texts),
                "unique_labels": len(set(labels)),
                "train_accuracy": train_score,
                "test_accuracy": test_score,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Failed to train CPU model {model_name}: {e}")
            return {"model_name": model_name, "success": False, "error": str(e)}

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get CPU memory usage statistics."""
        base_stats = super().get_memory_usage()

        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()

            cpu_stats = {
                "memory_rss": memory_info.rss,
                "memory_vms": memory_info.vms,
                "cpu_percent": process.cpu_percent(),
            }
            base_stats.update(cpu_stats)
        except ImportError:
            pass

        return base_stats
