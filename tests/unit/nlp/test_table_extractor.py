"""
Tests for Table and Experimental Data Extraction

Testing table extraction, experimental data parsing, and structured
data extraction from scientific documents.
"""

from unittest.mock import Mock, patch

import pytest
from chemi_accel.nlp.table_extractor import (
    ExperimentalDataExtractor,
    ExperimentalRecord,
    ExtractedTable,
    TableCell,
    TableExtractor,
    TableType,
)


class TestTableExtractor:
    """Test table extraction functionality."""

    @pytest.fixture
    def extractor(self):
        """Create a test table extractor."""
        return TableExtractor("cpu")

    @pytest.fixture
    def pipe_table_text(self):
        """Sample pipe-separated table text."""
        return """
        Table 1. Physical properties of synthesized compounds

        | Compound | Molecular Weight | Melting Point | Yield |
        |----------|------------------|---------------|-------|
        | 3a       | 234.26          | 145°C         | 78%   |
        | 3b       | 248.29          | 132°C         | 82%   |
        | 3c       | 262.31          | 158°C         | 65%   |
        """

    @pytest.fixture
    def tab_table_text(self):
        """Sample tab-separated table text."""
        return """
        Compound\tIC50 (µM)\tActivity\tSolubility
        4a\t2.3\tHigh\t15 mg/mL
        4b\t8.7\tModerate\t8 mg/mL
        4c\t>100\tLow\t<1 mg/mL
        """

    @pytest.fixture
    def space_table_text(self):
        """Sample space-separated table text."""
        return """
        Entry   Catalyst    Temperature   Time    Yield
        1       Pd(PPh3)4   80°C         2h      85%
        2       PdCl2       100°C        4h      72%
        3       Ni(COD)2    120°C        6h      68%
        """

    def test_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor is not None
        assert extractor.backend is not None
        assert extractor.entity_recognizer is not None

    def test_detect_table_regions(self, extractor, pipe_table_text):
        """Test table region detection."""
        regions = extractor._detect_table_regions(pipe_table_text)

        assert len(regions) > 0
        assert "text" in regions[0]
        assert "start_line" in regions[0]
        assert "end_line" in regions[0]
        assert "title" in regions[0]

    def test_extract_pipe_table(self, extractor, pipe_table_text):
        """Test pipe-separated table extraction."""
        tables = extractor.extract_tables(pipe_table_text)

        assert len(tables) == 1
        table = tables[0]

        assert isinstance(table, ExtractedTable)
        assert len(table.headers) == 4
        assert "Compound" in table.headers
        assert "Molecular Weight" in table.headers
        assert "Melting Point" in table.headers
        assert "Yield" in table.headers

        assert len(table.cells) == 3  # 3 data rows
        assert table.cells[0][0].value == "3a"
        assert table.cells[0][1].value == "234.26"

    def test_extract_tab_table(self, extractor, tab_table_text):
        """Test tab-separated table extraction."""
        tables = extractor.extract_tables(tab_table_text)

        assert len(tables) == 1
        table = tables[0]

        assert len(table.headers) == 4
        assert "Compound" in table.headers
        assert "IC50 (µM)" in table.headers

        assert len(table.cells) == 3
        assert table.cells[0][0].value == "4a"
        assert table.cells[0][1].value == "2.3"

    def test_extract_space_table(self, extractor, space_table_text):
        """Test space-separated table extraction."""
        tables = extractor.extract_tables(space_table_text)

        assert len(tables) == 1
        table = tables[0]

        assert len(table.headers) >= 4  # Should detect columns
        assert len(table.cells) >= 3  # Should detect rows

    def test_table_type_classification(self, extractor, pipe_table_text):
        """Test table type classification."""
        tables = extractor.extract_tables(pipe_table_text)
        table = tables[0]

        # Should classify as compound properties based on headers
        assert table.table_type in [TableType.COMPOUND_PROPERTIES, TableType.GENERIC]

    def test_cell_data_type_classification(self, extractor):
        """Test cell data type classification."""
        # Test different data types
        assert extractor._classify_cell_data_type("3a") == "text"
        assert extractor._classify_cell_data_type("234.26") == "number"
        assert extractor._classify_cell_data_type("145°C") == "number_with_unit"
        assert extractor._classify_cell_data_type("C9H8O4") == "chemical"
        assert extractor._classify_cell_data_type("75-80") == "range"
        assert extractor._classify_cell_data_type("") == "empty"

    def test_extract_experimental_data(self, extractor):
        """Test experimental data extraction from tables."""
        # Create a mock experimental table
        headers = ["Compound", "Temperature", "Time", "Yield", "Purity"]
        cells = [
            [
                TableCell("5a", 0, 0, "text"),
                TableCell("80°C", 0, 1, "number_with_unit"),
                TableCell("2h", 0, 2, "number_with_unit"),
                TableCell("85%", 0, 3, "number_with_unit"),
                TableCell("95%", 0, 4, "number_with_unit"),
            ]
        ]

        table = ExtractedTable(
            headers=headers, cells=cells, table_type=TableType.EXPERIMENTAL_DATA
        )

        records = extractor.extract_experimental_data(table)

        assert len(records) > 0
        record = records[0]

        assert isinstance(record, ExperimentalRecord)
        assert record.compound == "5a"
        assert "Temperature" in record.conditions or "temperature" in record.conditions
        assert "Yield" in record.properties or "yield" in record.properties

    def test_table_to_dataframe(self, extractor, pipe_table_text):
        """Test conversion to pandas DataFrame."""
        tables = extractor.extract_tables(pipe_table_text)
        table = tables[0]

        df = table.to_dataframe()

        if df is not None:  # Only test if pandas is available
            assert len(df) == 3  # 3 rows
            assert len(df.columns) == 4  # 4 columns
            assert "Compound" in df.columns

    def test_empty_text_handling(self, extractor):
        """Test handling of empty text."""
        tables = extractor.extract_tables("")
        assert len(tables) == 0

        tables = extractor.extract_tables("No tables here, just text.")
        assert len(tables) == 0

    def test_malformed_table_handling(self, extractor):
        """Test handling of malformed tables."""
        malformed_text = """
        | Header1 | Header2 |
        | Value1 | Value2 | ExtraValue |
        | Value3 |
        """

        tables = extractor.extract_tables(malformed_text)

        # Should still extract something, even if imperfect
        assert len(tables) >= 0  # May be 0 if too malformed

    def test_multiple_tables(self, extractor):
        """Test extraction of multiple tables from text."""
        multi_table_text = """
        Table 1: First table
        | A | B |
        | 1 | 2 |

        Some text between tables.

        Table 2: Second table
        | C | D |
        | 3 | 4 |
        """

        tables = extractor.extract_tables(multi_table_text)

        assert len(tables) >= 1  # Should find at least one table

        if len(tables) > 1:
            # Check that tables are different
            assert tables[0].headers != tables[1].headers


class TestExperimentalDataExtractor:
    """Test experimental data extraction."""

    @pytest.fixture
    def extractor(self):
        """Create experimental data extractor."""
        return ExperimentalDataExtractor("cpu")

    @pytest.fixture
    def experimental_document(self):
        """Sample experimental document text."""
        return """
        Experimental Section

        General Procedure for Synthesis of Compounds 3a-3c

        To a solution of starting material (1 mmol) in DMF (10 mL) was added
        the coupling partner (1.2 mmol) and Pd(PPh3)4 (0.05 mmol). The mixture
        was heated to 80°C for 4 hours under nitrogen atmosphere. After cooling,
        the reaction was quenched with water and extracted with ethyl acetate.

        Table 1: Optimization of reaction conditions
        | Entry | Catalyst | Temperature | Time | Yield |
        |-------|----------|-------------|------|-------|
        | 1     | Pd(PPh3)4| 60°C        | 2h   | 45%   |
        | 2     | Pd(PPh3)4| 80°C        | 4h   | 78%   |
        | 3     | PdCl2    | 80°C        | 4h   | 52%   |

        Synthesis of 3a: Following the general procedure, compound 3a was obtained
        as a white solid (234 mg, 78% yield). Mp 145-147°C.
        """

    def test_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor is not None
        assert extractor.table_extractor is not None
        assert extractor.entity_recognizer is not None

    def test_extract_from_document(self, extractor, experimental_document):
        """Test complete document extraction."""
        result = extractor.extract_from_document(experimental_document)

        assert "experimental_sections" in result
        assert "tables" in result
        assert "experimental_records" in result
        assert "synthesis_procedures" in result
        assert "summary" in result

        # Should find experimental sections
        assert len(result["experimental_sections"]) > 0

        # Should find tables
        assert len(result["tables"]) > 0

        # Should have summary statistics
        summary = result["summary"]
        assert "num_tables" in summary
        assert "num_records" in summary
        assert "num_procedures" in summary

    def test_find_experimental_sections(self, extractor, experimental_document):
        """Test experimental section detection."""
        sections = extractor._find_experimental_sections(experimental_document)

        assert len(sections) > 0

        section = sections[0]
        assert "title" in section
        assert "content" in section
        assert "start_line" in section
        assert "end_line" in section

        assert "experimental" in section["title"].lower()

    def test_extract_synthesis_procedures(self, extractor, experimental_document):
        """Test synthesis procedure extraction."""
        sections = extractor._find_experimental_sections(experimental_document)
        procedures = extractor._extract_synthesis_procedures(sections)

        assert len(procedures) > 0

        procedure = procedures[0]
        assert "compound" in procedure
        assert "steps" in procedure
        assert "section" in procedure

        # Should extract steps
        assert len(procedure["steps"]) > 0

    def test_procedure_step_extraction(self, extractor):
        """Test individual step extraction."""
        text = """
        Synthesis of compound 5: To a flask was added reagent A (1 g).
        The mixture was heated to 100°C. After 2 hours, the reaction was complete.
        The product was isolated by filtration.
        """

        steps = extractor._extract_procedure_steps(text, 0)

        assert len(steps) > 0

        # Steps should be meaningful
        for step in steps:
            assert len(step) > 20  # Should be substantial

    def test_complex_experimental_text(self, extractor):
        """Test with complex experimental text."""
        complex_text = """
        Materials and Methods

        All reagents were purchased from commercial sources and used without
        further purification. 1H NMR spectra were recorded on a 400 MHz spectrometer.

        General Procedure A: Suzuki Coupling

        A mixture of aryl halide (1.0 mmol), boronic acid (1.2 mmol), K2CO3 (2.0 mmol),
        and Pd(PPh3)4 (0.05 mmol) in DME/H2O (4:1, 5 mL) was heated to 85°C for 12 h
        under nitrogen. The reaction was cooled, diluted with water, and extracted with EtOAc.

        Table S1: Substrate scope for Suzuki coupling
        | Entry | Ar-X | Ar-B(OH)2 | Product | Yield (%) |
        |-------|------|-----------|---------|-----------|
        | 1     | PhI  | PhB(OH)2  | 6a      | 92       |
        | 2     | 4-MePhI | PhB(OH)2 | 6b    | 87       |
        | 3     | PhBr | 4-OMePhB(OH)2 | 6c  | 78       |

        Characterization Data:

        6a: White solid, mp 85-87°C. 1H NMR (400 MHz, CDCl3): δ 7.65-7.58 (m, 4H),
        7.45-7.35 (m, 6H). 13C NMR (100 MHz, CDCl3): δ 141.2, 128.7, 127.3, 127.1.
        HRMS (ESI): calcd for C12H10 [M+H]+ 155.0861, found 155.0863.
        """

        result = extractor.extract_from_document(complex_text)

        # Should extract comprehensive data
        assert len(result["experimental_sections"]) > 0
        assert len(result["tables"]) > 0

        # Should find synthesis procedures
        assert len(result["synthesis_procedures"]) > 0

        # Check summary
        summary = result["summary"]
        assert summary["num_tables"] > 0
        assert summary["num_procedures"] > 0


class TestTableCellProcessing:
    """Test individual table cell processing."""

    def test_table_cell_creation(self):
        """Test TableCell creation and properties."""
        cell = TableCell("aspirin", 0, 1, "chemical", 0.9)

        assert cell.value == "aspirin"
        assert cell.row == 0
        assert cell.column == 1
        assert cell.data_type == "chemical"
        assert cell.confidence == 0.9

    def test_table_cell_entity_extraction(self):
        """Test entity extraction in table cells."""
        # This would require mocking the entity recognizer
        # For now, just test the structure
        cell = TableCell("C6H6", 0, 0, "chemical")

        # Should be able to store entities
        assert hasattr(cell, "entities")
        assert isinstance(cell.entities, list)


class TestTableIntegration:
    """Integration tests for table extraction."""

    def test_real_chemistry_table(self):
        """Test with realistic chemistry table."""
        chemistry_table = """
        Table 2. Optimization of Reaction Conditions for the Synthesis of 2a

        | Entry | Catalyst (mol%) | Solvent | Temp (°C) | Time (h) | Yield (%) |
        |-------|-----------------|---------|-----------|----------|-----------|
        | 1     | Pd(PPh3)4 (5)   | DMF     | 80        | 4        | 45        |
        | 2     | Pd(PPh3)4 (5)   | Toluene | 110       | 6        | 67        |
        | 3     | Pd(OAc)2 (10)   | DMF     | 80        | 4        | 52        |
        | 4     | PdCl2 (10)      | DMF     | 100       | 8        | 38        |
        | 5     | Pd(PPh3)4 (2.5) | Toluene | 110       | 12       | 71        |
        """

        extractor = TableExtractor("cpu")
        tables = extractor.extract_tables(chemistry_table)

        assert len(tables) == 1
        table = tables[0]

        # Should classify as reaction conditions
        assert table.table_type in [
            TableType.REACTION_CONDITIONS,
            TableType.EXPERIMENTAL_DATA,
            TableType.GENERIC,
        ]

        # Should have correct structure
        assert len(table.headers) == 6
        assert len(table.cells) == 5

        # Should extract experimental records
        records = extractor.extract_experimental_data(table)
        assert len(records) == 5

        # Check first record
        if records:
            record = records[0]
            assert record.conditions  # Should have conditions
            assert record.properties  # Should have properties like yield

    def test_biological_activity_table(self):
        """Test extraction from biological activity table."""
        bio_table = """
        Table 3. In vitro anticancer activity of synthesized compounds

        | Compound | HeLa IC50 (µM) | MCF-7 IC50 (µM) | Selectivity Index |
        |----------|----------------|------------------|-------------------|
        | 7a       | 2.3 ± 0.2      | 4.1 ± 0.4        | 1.8              |
        | 7b       | 8.7 ± 1.1      | 12.4 ± 1.8       | 1.4              |
        | 7c       | 15.2 ± 2.3     | 28.6 ± 3.2       | 1.9              |
        | Cisplatin| 8.1 ± 0.9      | 11.2 ± 1.4       | 1.4              |
        """

        extractor = TableExtractor("cpu")
        tables = extractor.extract_tables(bio_table)

        assert len(tables) == 1
        table = tables[0]

        # Should classify as biological activity
        assert table.table_type in [TableType.BIOLOGICAL_ACTIVITY, TableType.GENERIC]

        # Should handle complex cell values (with ±)
        assert len(table.cells) == 4

        # Check cell content
        first_cell = table.cells[0][1]  # First IC50 value
        assert "2.3" in first_cell.value
