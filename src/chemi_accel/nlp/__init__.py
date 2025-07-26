"""
GPU-Accelerated Information Extraction (NLP & Entity Recognition)

This module provides GPU-accelerated NLP models for automated extraction of chemical
entities and relationships from unstructured text (patents, papers). It includes
modules for extracting structured data from tables, experimental reports, and databases,
mapping them to the IC-KG schema.

Features:
- GPU-accelerated transformer models with Rapids/PyTorch/JAX backends
- Chemical entity recognition (compounds, reactions, properties)
- Structured data extraction from tables and reports
- IC-KG schema mapping for extracted entities
- Multi-modal document processing (text, tables, figures)
"""

from .backends import NLPBackend, get_nlp_backend
from .entity_recognition import ChemicalEntityRecognizer
from .entity_recognition import EntityType as NEREntityType
from .schema_mapper import ICKGSchemaMapper
from .table_extractor import ExperimentalDataExtractor, TableExtractor
from .text_processor import DocumentProcessor, TextProcessor

__all__ = [
    "ChemicalEntityRecognizer",
    "NEREntityType",
    "TextProcessor",
    "DocumentProcessor",
    "TableExtractor",
    "ExperimentalDataExtractor",
    "ICKGSchemaMapper",
    "NLPBackend",
    "get_nlp_backend",
]

__version__ = "0.1.0"
