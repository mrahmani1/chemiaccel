"""
Table and Experimental Data Extraction

GPU-accelerated extraction of structured data from tables, experimental reports,
and scientific documents for chemical information processing.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

from .backends import NLPBackend, get_nlp_backend
from .entity_recognition import ChemicalEntityRecognizer, EntityExtraction

logger = logging.getLogger(__name__)


class TableType(Enum):
    """Types of tables that can be extracted."""

    EXPERIMENTAL_DATA = "experimental_data"
    COMPOUND_PROPERTIES = "compound_properties"
    REACTION_CONDITIONS = "reaction_conditions"
    ANALYTICAL_RESULTS = "analytical_results"
    BIOLOGICAL_ACTIVITY = "biological_activity"
    PHYSICAL_PROPERTIES = "physical_properties"
    SYNTHESIS_PROTOCOL = "synthesis_protocol"
    CHARACTERIZATION = "characterization"
    GENERIC = "generic"


@dataclass
class TableCell:
    """Represents a single cell in an extracted table."""

    value: str
    row: int
    column: int
    data_type: str = "text"  # text, number, chemical, unit
    confidence: float = 1.0
    entities: List[EntityExtraction] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedTable:
    """Container for extracted table data."""

    headers: List[str]
    cells: List[List[TableCell]]
    table_type: TableType
    title: Optional[str] = None
    caption: Optional[str] = None
    footnotes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0

    def to_dataframe(self) -> Optional[pd.DataFrame]:
        """Convert to pandas DataFrame if available."""
        if not PANDAS_AVAILABLE:
            return None

        data = []
        for row in self.cells:
            row_data = [cell.value for cell in row]
            data.append(row_data)

        try:
            return pd.DataFrame(data, columns=self.headers)
        except Exception:
            return None

    def get_cell_value(self, row: int, column: int) -> Optional[str]:
        """Get value at specific row and column."""
        try:
            return self.cells[row][column].value
        except IndexError:
            return None

    def get_column_values(self, column_name: str) -> List[str]:
        """Get all values in a specific column."""
        try:
            column_idx = self.headers.index(column_name)
            return [row[column_idx].value for row in self.cells]
        except (ValueError, IndexError):
            return []


@dataclass
class ExperimentalRecord:
    """Container for extracted experimental data."""

    compound: Optional[str] = None
    conditions: Dict[str, Any] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


class TableExtractor:
    """
    GPU-accelerated table extraction from scientific documents.

    Extracts structured data from tables in chemical and scientific literature,
    with specialized handling for experimental data, compound properties, and
    reaction conditions.
    """

    def __init__(self, backend: Optional[str] = None):
        self.backend = get_nlp_backend(backend)
        self.entity_recognizer = ChemicalEntityRecognizer()

        # Table detection patterns
        self.table_markers = [
            r"Table\s+\d+",
            r"TABLE\s+\d+",
            r"Tab\.\s*\d+",
            r"table\s+\d+",
            r"\|.*\|",  # Pipe-separated tables
            r"\t.*\t",  # Tab-separated tables
        ]

        # Chemical table type indicators
        self.table_type_indicators = {
            TableType.EXPERIMENTAL_DATA: [
                "experiment",
                "trial",
                "run",
                "condition",
                "parameter",
            ],
            TableType.COMPOUND_PROPERTIES: [
                "compound",
                "property",
                "molecular weight",
                "melting point",
                "boiling point",
            ],
            TableType.REACTION_CONDITIONS: [
                "reaction",
                "temperature",
                "time",
                "catalyst",
                "solvent",
                "yield",
            ],
            TableType.ANALYTICAL_RESULTS: [
                "analysis",
                "nmr",
                "hplc",
                "gc",
                "ms",
                "spectroscopy",
                "purity",
            ],
            TableType.BIOLOGICAL_ACTIVITY: [
                "activity",
                "ic50",
                "ec50",
                "ki",
                "inhibition",
                "binding",
                "assay",
            ],
            TableType.PHYSICAL_PROPERTIES: [
                "density",
                "viscosity",
                "refractive index",
                "conductivity",
                "thermal",
            ],
            TableType.CHARACTERIZATION: [
                "characterization",
                "elemental analysis",
                "melting point",
                "nmr data",
            ],
        }

        logger.info(
            f"Initialized TableExtractor with {self.backend.backend_type.value} backend"
        )

    def extract_tables(self, text: str) -> List[ExtractedTable]:
        """
        Extract all tables from text document.

        Args:
            text: Input text containing tables

        Returns:
            List of extracted tables
        """
        # Detect table regions
        table_regions = self._detect_table_regions(text)

        extracted_tables = []

        for region in table_regions:
            table = self._extract_single_table(region)
            if table:
                extracted_tables.append(table)

        return extracted_tables

    def _detect_table_regions(self, text: str) -> List[Dict[str, Any]]:
        """Detect regions in text that contain tables."""
        regions = []
        lines = text.split("\n")

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Check for table markers
            table_start = None
            for marker in self.table_markers:
                if re.search(marker, line, re.IGNORECASE):
                    table_start = i
                    break

            if table_start is not None:
                # Find table end
                table_end = self._find_table_end(lines, table_start)

                if table_end > table_start:
                    region_text = "\n".join(lines[table_start : table_end + 1])
                    regions.append(
                        {
                            "text": region_text,
                            "start_line": table_start,
                            "end_line": table_end,
                            "title": self._extract_table_title(lines, table_start),
                        }
                    )

                    i = table_end + 1
                else:
                    i += 1
            else:
                # Check for table-like structures without explicit markers
                if self._looks_like_table_line(line):
                    table_start = i
                    table_end = self._find_table_end(lines, table_start)

                    if table_end - table_start >= 2:  # Minimum table size
                        region_text = "\n".join(lines[table_start : table_end + 1])
                        regions.append(
                            {
                                "text": region_text,
                                "start_line": table_start,
                                "end_line": table_end,
                                "title": None,
                            }
                        )

                        i = table_end + 1
                    else:
                        i += 1
                else:
                    i += 1

        return regions

    def _looks_like_table_line(self, line: str) -> bool:
        """Check if a line looks like it contains table data."""
        # Check for common table separators
        separators = ["|", "\t"]
        separator_count = sum(line.count(sep) for sep in separators)

        if separator_count >= 2:
            return True

        # Check for multiple consecutive spaces (column alignment)
        if len(re.findall(r"\s{3,}", line)) >= 2:
            return True

        # Check for numeric data patterns
        if len(re.findall(r"\d+(?:\.\d+)?", line)) >= 3:
            return True

        return False

    def _find_table_end(self, lines: List[str], start: int) -> int:
        """Find the end of a table starting at given line."""
        i = start + 1
        consecutive_empty = 0

        while i < len(lines):
            line = lines[i].strip()

            if not line:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    return i - 2
            else:
                consecutive_empty = 0

                # Check if line continues to look like table data
                if not self._looks_like_table_line(line):
                    # Check if it's a table continuation (footnote, etc.)
                    if not self._is_table_continuation(line):
                        return i - 1

            i += 1

        return min(i - 1, len(lines) - 1)

    def _is_table_continuation(self, line: str) -> bool:
        """Check if line is a table continuation (footnote, etc.)."""
        continuation_patterns = [
            r"^\s*[a-z]\s*\)",  # Footnote markers like "a)"
            r"^\s*\*",  # Asterisk footnotes
            r"^\s*Note:",  # Note indicators
            r"^\s*\d+\s*\.",  # Numbered footnotes
        ]

        for pattern in continuation_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True

        return False

    def _extract_table_title(self, lines: List[str], table_start: int) -> Optional[str]:
        """Extract table title from surrounding lines."""
        # Look backwards for title
        for i in range(max(0, table_start - 3), table_start):
            line = lines[i].strip()
            if re.match(r"Table\s+\d+", line, re.IGNORECASE):
                # Extract everything after "Table X"
                match = re.search(r"Table\s+\d+[:.]\s*(.*)", line, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
                else:
                    return line

        return None

    def _extract_single_table(self, region: Dict[str, Any]) -> Optional[ExtractedTable]:
        """Extract a single table from a text region."""
        text = region["text"]
        lines = text.strip().split("\n")

        # Determine table format
        if "|" in text:
            return self._extract_pipe_table(lines, region)
        elif "\t" in text:
            return self._extract_tab_table(lines, region)
        else:
            return self._extract_space_table(lines, region)

    def _extract_pipe_table(
        self, lines: List[str], region: Dict[str, Any]
    ) -> Optional[ExtractedTable]:
        """Extract pipe-separated table."""
        headers = []
        rows = []

        for line in lines:
            line = line.strip()
            if "|" in line:
                # Split by pipe and clean
                columns = [col.strip() for col in line.split("|")]
                # Remove empty columns at start/end
                columns = [col for col in columns if col]

                if not headers and columns:
                    headers = columns
                elif columns and len(columns) == len(headers):
                    row_cells = []
                    for i, value in enumerate(columns):
                        cell = self._create_table_cell(value, len(rows), i)
                        row_cells.append(cell)
                    rows.append(row_cells)

        if headers and rows:
            table_type = self._classify_table_type(headers, rows)
            return ExtractedTable(
                headers=headers,
                cells=rows,
                table_type=table_type,
                title=region.get("title"),
                metadata={"format": "pipe", "region": region},
            )

        return None

    def _extract_tab_table(
        self, lines: List[str], region: Dict[str, Any]
    ) -> Optional[ExtractedTable]:
        """Extract tab-separated table."""
        headers = []
        rows = []

        for line in lines:
            if "\t" in line:
                columns = [col.strip() for col in line.split("\t")]
                columns = [col for col in columns if col]

                if not headers and columns:
                    headers = columns
                elif columns:
                    # Pad or truncate to match header length
                    while len(columns) < len(headers):
                        columns.append("")
                    columns = columns[: len(headers)]

                    row_cells = []
                    for i, value in enumerate(columns):
                        cell = self._create_table_cell(value, len(rows), i)
                        row_cells.append(cell)
                    rows.append(row_cells)

        if headers and rows:
            table_type = self._classify_table_type(headers, rows)
            return ExtractedTable(
                headers=headers,
                cells=rows,
                table_type=table_type,
                title=region.get("title"),
                metadata={"format": "tab", "region": region},
            )

        return None

    def _extract_space_table(
        self, lines: List[str], region: Dict[str, Any]
    ) -> Optional[ExtractedTable]:
        """Extract space-separated table using column alignment detection."""
        # Find column positions by analyzing alignment
        column_positions = self._detect_column_positions(lines)

        if len(column_positions) < 2:
            return None

        headers = []
        rows = []

        for line_idx, line in enumerate(lines):
            if not line.strip():
                continue

            columns = self._split_by_positions(line, column_positions)
            columns = [col.strip() for col in columns if col.strip()]

            if not headers and columns:
                headers = columns
            elif columns:
                # Ensure consistent column count
                while len(columns) < len(headers):
                    columns.append("")
                columns = columns[: len(headers)]

                row_cells = []
                for i, value in enumerate(columns):
                    cell = self._create_table_cell(value, len(rows), i)
                    row_cells.append(cell)
                rows.append(row_cells)

        if headers and rows:
            table_type = self._classify_table_type(headers, rows)
            return ExtractedTable(
                headers=headers,
                cells=rows,
                table_type=table_type,
                title=region.get("title"),
                metadata={"format": "space", "region": region},
            )

        return None

    def _detect_column_positions(self, lines: List[str]) -> List[int]:
        """Detect column positions in space-separated tables."""
        # Count spaces at each position across all lines
        max_len = max(len(line) for line in lines if line.strip())
        space_counts = [0] * max_len

        for line in lines:
            if not line.strip():
                continue
            for i, char in enumerate(line):
                if char == " ":
                    space_counts[i] += 1

        # Find positions with high space frequency (column separators)
        threshold = len([line for line in lines if line.strip()]) * 0.5
        separator_positions = [
            i for i, count in enumerate(space_counts) if count >= threshold
        ]

        # Group consecutive positions and take the middle
        column_positions = [0]  # Start at beginning

        if separator_positions:
            groups = []
            current_group = [separator_positions[0]]

            for pos in separator_positions[1:]:
                if pos - current_group[-1] <= 3:  # Group consecutive positions
                    current_group.append(pos)
                else:
                    groups.append(current_group)
                    current_group = [pos]
            groups.append(current_group)

            # Take middle position of each group
            for group in groups:
                column_positions.append(group[len(group) // 2])

        return sorted(set(column_positions))

    def _split_by_positions(self, line: str, positions: List[int]) -> List[str]:
        """Split line by column positions."""
        columns = []
        for i in range(len(positions)):
            start = positions[i]
            end = positions[i + 1] if i + 1 < len(positions) else len(line)
            columns.append(line[start:end])

        return columns

    def _create_table_cell(self, value: str, row: int, column: int) -> TableCell:
        """Create a TableCell with entity recognition."""
        # Clean the value
        value = value.strip()

        # Determine data type
        data_type = self._classify_cell_data_type(value)

        # Extract entities from the cell value
        entities = []
        if data_type in ["chemical", "text"] and len(value) > 2:
            try:
                entities = self.entity_recognizer.extract_entities(value)
            except Exception:
                pass  # Ignore entity extraction errors

        return TableCell(
            value=value,
            row=row,
            column=column,
            data_type=data_type,
            entities=entities,
            confidence=1.0,
        )

    def _classify_cell_data_type(self, value: str) -> str:
        """Classify the data type of a cell value."""
        if not value:
            return "empty"

        # Check for numbers
        if re.match(r"^-?\d+(?:\.\d+)?$", value):
            return "number"

        # Check for numbers with units
        if re.match(r"^-?\d+(?:\.\d+)?\s*[a-zA-Z°%]+$", value):
            return "number_with_unit"

        # Check for chemical formulas
        if re.match(r"^[A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?\d*)*$", value):
            return "chemical"

        # Check for ranges
        if re.match(r"^\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?", value):
            return "range"

        return "text"

    def _classify_table_type(
        self, headers: List[str], rows: List[List[TableCell]]
    ) -> TableType:
        """Classify the type of table based on headers and content."""
        header_text = " ".join(headers).lower()

        # Score each table type
        type_scores = {}

        for table_type, indicators in self.table_type_indicators.items():
            score = 0
            for indicator in indicators:
                if indicator in header_text:
                    score += 1

            # Also check cell content
            cell_content = ""
            for row in rows[:5]:  # Check first 5 rows
                for cell in row:
                    cell_content += " " + cell.value.lower()

            for indicator in indicators:
                if indicator in cell_content:
                    score += 0.5

            type_scores[table_type] = score

        # Return the type with highest score
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            if best_type[1] > 0:
                return best_type[0]

        return TableType.GENERIC

    def extract_experimental_data(
        self, table: ExtractedTable
    ) -> List[ExperimentalRecord]:
        """Extract experimental records from a table."""
        if table.table_type not in [
            TableType.EXPERIMENTAL_DATA,
            TableType.REACTION_CONDITIONS,
        ]:
            return []

        records = []

        # Find relevant columns
        compound_cols = self._find_columns_by_keywords(
            table.headers, ["compound", "chemical", "molecule", "name"]
        )
        condition_cols = self._find_columns_by_keywords(
            table.headers, ["temperature", "time", "pressure", "solvent"]
        )
        property_cols = self._find_columns_by_keywords(
            table.headers, ["yield", "purity", "melting", "boiling"]
        )
        result_cols = self._find_columns_by_keywords(
            table.headers, ["result", "outcome", "product", "activity"]
        )

        for row in table.cells:
            record = ExperimentalRecord()

            # Extract compound information
            for col_idx in compound_cols:
                if col_idx < len(row):
                    record.compound = row[col_idx].value
                    break

            # Extract conditions
            for col_idx in condition_cols:
                if col_idx < len(row) and col_idx < len(table.headers):
                    header = table.headers[col_idx]
                    value = row[col_idx].value
                    record.conditions[header] = self._parse_condition_value(value)

            # Extract properties
            for col_idx in property_cols:
                if col_idx < len(row) and col_idx < len(table.headers):
                    header = table.headers[col_idx]
                    value = row[col_idx].value
                    record.properties[header] = self._parse_property_value(value)

            # Extract results
            for col_idx in result_cols:
                if col_idx < len(row) and col_idx < len(table.headers):
                    header = table.headers[col_idx]
                    value = row[col_idx].value
                    record.results[header] = value

            if record.compound or record.conditions or record.properties:
                records.append(record)

        return records

    def _find_columns_by_keywords(
        self, headers: List[str], keywords: List[str]
    ) -> List[int]:
        """Find column indices that match given keywords."""
        matching_cols = []

        for i, header in enumerate(headers):
            header_lower = header.lower()
            for keyword in keywords:
                if keyword in header_lower:
                    matching_cols.append(i)
                    break

        return matching_cols

    def _parse_condition_value(self, value: str) -> Dict[str, Any]:
        """Parse a condition value and extract number and unit."""
        # Look for number + unit patterns
        match = re.search(r"(\d+(?:\.\d+)?)\s*([a-zA-Z°%]+)", value)
        if match:
            return {
                "value": float(match.group(1)),
                "unit": match.group(2),
                "raw": value,
            }

        return {"raw": value}

    def _parse_property_value(self, value: str) -> Dict[str, Any]:
        """Parse a property value and extract relevant information."""
        # Similar to condition parsing but with different context
        return self._parse_condition_value(value)


class ExperimentalDataExtractor:
    """
    Specialized extractor for experimental data from scientific papers.

    Focuses on extracting structured experimental protocols, conditions,
    and results from chemical literature.
    """

    def __init__(self, backend: Optional[str] = None):
        self.table_extractor = TableExtractor(backend)
        self.entity_recognizer = ChemicalEntityRecognizer()

        # Experimental section patterns
        self.experimental_patterns = [
            r"experimental\s+section",
            r"materials\s+and\s+methods",
            r"synthesis\s+of",
            r"general\s+procedure",
            r"typical\s+procedure",
        ]

    def extract_from_document(self, text: str) -> Dict[str, Any]:
        """
        Extract all experimental data from a document.

        Args:
            text: Full document text

        Returns:
            Dictionary containing extracted experimental data
        """
        # Find experimental sections
        experimental_sections = self._find_experimental_sections(text)

        # Extract tables
        tables = self.table_extractor.extract_tables(text)

        # Extract experimental records
        all_records = []
        for table in tables:
            records = self.table_extractor.extract_experimental_data(table)
            all_records.extend(records)

        # Extract synthesis procedures
        procedures = self._extract_synthesis_procedures(experimental_sections)

        return {
            "experimental_sections": experimental_sections,
            "tables": tables,
            "experimental_records": all_records,
            "synthesis_procedures": procedures,
            "summary": {
                "num_tables": len(tables),
                "num_records": len(all_records),
                "num_procedures": len(procedures),
            },
        }

    def _find_experimental_sections(self, text: str) -> List[Dict[str, Any]]:
        """Find sections containing experimental information."""
        sections = []
        lines = text.split("\n")

        for i, line in enumerate(lines):
            line_lower = line.lower().strip()

            for pattern in self.experimental_patterns:
                if re.search(pattern, line_lower):
                    # Extract section content
                    section_end = self._find_section_end(lines, i)
                    section_text = "\n".join(lines[i:section_end])

                    sections.append(
                        {
                            "title": line.strip(),
                            "content": section_text,
                            "start_line": i,
                            "end_line": section_end,
                        }
                    )
                    break

        return sections

    def _find_section_end(self, lines: List[str], start: int) -> int:
        """Find the end of a section."""
        i = start + 1

        while i < len(lines):
            line = lines[i].strip()

            # Check for new section headers
            if line.isupper() and len(line) < 100:
                return i

            # Check for numbered sections
            if re.match(r"^\d+\.", line):
                return i

            i += 1

        return len(lines)

    def _extract_synthesis_procedures(
        self, sections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract synthesis procedures from experimental sections."""
        procedures = []

        for section in sections:
            content = section["content"]

            # Find synthesis procedures
            procedure_patterns = [
                r"synthesis\s+of\s+(.+?)(?:\n|$)",
                r"preparation\s+of\s+(.+?)(?:\n|$)",
                r"(.+?)\s+was\s+prepared",
                r"to\s+a\s+solution\s+of\s+(.+?)",
            ]

            for pattern in procedure_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    compound = match.group(1).strip()

                    # Extract procedure steps
                    steps = self._extract_procedure_steps(content, match.start())

                    if steps:
                        procedures.append(
                            {
                                "compound": compound,
                                "steps": steps,
                                "section": section["title"],
                            }
                        )

        return procedures

    def _extract_procedure_steps(self, text: str, start_pos: int) -> List[str]:
        """Extract individual steps from a synthesis procedure."""
        # Find the procedure text (next few sentences)
        sentences = re.split(r"[.!?]+", text[start_pos : start_pos + 1000])

        steps = []
        for sentence in sentences[:5]:  # Take first 5 sentences
            sentence = sentence.strip()
            if len(sentence) > 20:  # Meaningful step
                steps.append(sentence)

        return steps
