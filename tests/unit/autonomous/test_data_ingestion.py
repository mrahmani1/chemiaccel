"""
Tests for Real-time Experimental Data Ingestion and Validation

This module contains comprehensive tests for the data ingestion system including
APIs, parsers, validation, and streaming capabilities.
"""

import pytest
import asyncio
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from chemi_accel.autonomous.data_ingestion import (
    ExperimentalDataPoint,
    InstrumentConnection,
    RESTAPIIngestion,
    FileWatcherIngestion,
    DataIngestionManager,
    NMRDataParser,
    LCMSDataParser,
    AssayDataParser,
    create_default_parsers,
    create_sample_connections
)
from chemi_accel.autonomous.data_validation import (
    DataValidator,
    ValidationResult,
    SchemaValidationRule,
    RangeValidationRule,
    create_nmr_validator
)


class TestExperimentalDataPoint:
    """Test cases for ExperimentalDataPoint dataclass."""
    
    def test_data_point_creation(self):
        """Test basic data point creation."""
        timestamp = datetime.now(timezone.utc)
        data_point = ExperimentalDataPoint(
            data_id="test_123",
            experiment_id="exp_456",
            instrument_id="nmr_400",
            timestamp=timestamp,
            data_type="nmr",
            raw_data={"chemical_shifts": [7.5, 7.2, 3.8], "intensities": [100, 80, 120]},
            metadata={"solvent": "CDCl3", "temperature": 298}
        )
        
        assert data_point.data_id == "test_123"
        assert data_point.experiment_id == "exp_456"
        assert data_point.instrument_id == "nmr_400"
        assert data_point.timestamp == timestamp
        assert data_point.data_type == "nmr"
        assert len(data_point.raw_data["chemical_shifts"]) == 3
        assert data_point.metadata["solvent"] == "CDCl3"
        assert data_point.processing_status == "pending"
        assert len(data_point.quality_flags) == 0


class TestInstrumentConnection:
    """Test cases for InstrumentConnection configuration."""
    
    def test_connection_creation(self):
        """Test instrument connection creation."""
        connection = InstrumentConnection(
            instrument_id="lcms_qtof",
            instrument_type="lcms",
            connection_type="api",
            endpoint="http://lcms-server:8080/api/v1",
            credentials={"api_key": "test_key"},
            polling_interval=30.0,
            timeout=10.0,
            retry_attempts=3
        )
        
        assert connection.instrument_id == "lcms_qtof"
        assert connection.instrument_type == "lcms"
        assert connection.connection_type == "api"
        assert connection.endpoint == "http://lcms-server:8080/api/v1"
        assert connection.credentials["api_key"] == "test_key"
        assert connection.polling_interval == 30.0
        assert connection.timeout == 10.0
        assert connection.retry_attempts == 3
        assert connection.is_active is True


class TestRESTAPIIngestion:
    """Test cases for REST API data ingestion."""
    
    @pytest.fixture
    def sample_connection(self) -> InstrumentConnection:
        """Create sample connection for testing."""
        return InstrumentConnection(
            instrument_id="test_api",
            instrument_type="nmr",
            connection_type="api",
            endpoint="http://test-server:8080/api/v1",
            credentials={"api_key": "test_key"},
            polling_interval=30.0
        )
    
    @pytest.fixture
    def api_ingestion(self, sample_connection) -> RESTAPIIngestion:
        """Create API ingestion instance."""
        return RESTAPIIngestion(sample_connection)
    
    @pytest.mark.asyncio
    async def test_connection_success(self, api_ingestion):
        """Test successful API connection."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock successful health check
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await api_ingestion.connect()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_connection_failure(self, api_ingestion):
        """Test API connection failure."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock connection failure
            mock_response = AsyncMock()
            mock_response.status = 503
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await api_ingestion.connect()
            assert result is False
    
    @pytest.mark.asyncio
    async def test_data_fetch_success(self, api_ingestion):
        """Test successful data fetching."""
        # First connect
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response
            await api_ingestion.connect()
        
        # Mock data fetch
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "results": [
                    {
                        "id": "data_123",
                        "experiment_id": "exp_456",
                        "timestamp": "2024-01-01T12:00:00Z",
                        "data_type": "nmr",
                        "data": {"chemical_shifts": [7.5, 7.2], "intensities": [100, 80]},
                        "metadata": {"solvent": "CDCl3"}
                    }
                ]
            }
            mock_get.return_value.__aenter__.return_value = mock_response
            
            data_points = await api_ingestion.fetch_data()
            
            assert len(data_points) == 1
            assert data_points[0].data_id == "data_123"
            assert data_points[0].experiment_id == "exp_456"
            assert data_points[0].data_type == "nmr"
            assert len(data_points[0].raw_data["chemical_shifts"]) == 2
    
    @pytest.mark.asyncio
    async def test_validate_connection(self, api_ingestion):
        """Test connection validation."""
        # Without session
        result = await api_ingestion.validate_connection()
        assert result is False
        
        # With session
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response
            await api_ingestion.connect()
            
            result = await api_ingestion.validate_connection()
            assert result is True


class TestFileWatcherIngestion:
    """Test cases for file watcher data ingestion."""
    
    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_connection(self, temp_directory) -> InstrumentConnection:
        """Create sample file watcher connection."""
        return InstrumentConnection(
            instrument_id="test_file_watcher",
            instrument_type="nmr",
            connection_type="file_watcher",
            endpoint=str(temp_directory),
            polling_interval=5.0
        )
    
    @pytest.fixture
    def file_watcher(self, sample_connection) -> FileWatcherIngestion:
        """Create file watcher instance."""
        parser = NMRDataParser()
        return FileWatcherIngestion(sample_connection, parser)
    
    @pytest.mark.asyncio
    async def test_connection_success(self, file_watcher):
        """Test successful file watcher connection."""
        result = await file_watcher.connect()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_connection_failure(self):
        """Test file watcher connection failure."""
        connection = InstrumentConnection(
            instrument_id="test_fail",
            instrument_type="nmr",
            connection_type="file_watcher",
            endpoint="/nonexistent/directory",
            polling_interval=5.0
        )
        parser = NMRDataParser()
        watcher = FileWatcherIngestion(connection, parser)
        
        result = await watcher.connect()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_file_processing(self, file_watcher, temp_directory):
        """Test file processing."""
        await file_watcher.connect()
        
        # Create test file
        test_file = temp_directory / "test_spectrum.txt"
        test_file.write_text("7.5 100\n7.2 80\n3.8 120\n")
        
        # Mock parser to return test data
        with patch.object(file_watcher.data_parser, 'parse_file', return_value=[
            {
                "spectrum_type": "1H_NMR",
                "chemical_shifts": [7.5, 7.2, 3.8],
                "intensities": [100, 80, 120]
            }
        ]):
            data_points = await file_watcher.fetch_data()
            
            assert len(data_points) == 1
            assert data_points[0].instrument_id == "test_file_watcher"
            assert data_points[0].data_type == "nmr"
            assert "source_file" in data_points[0].metadata


class TestNMRDataParser:
    """Test cases for NMR data parser."""
    
    @pytest.fixture
    def nmr_parser(self) -> NMRDataParser:
        """Create NMR parser instance."""
        return NMRDataParser()
    
    @pytest.mark.asyncio
    async def test_generic_nmr_parsing(self, nmr_parser, tmp_path):
        """Test generic NMR file parsing."""
        # Create test file
        test_file = tmp_path / "test_nmr.txt"
        test_file.write_text("7.5 100\n7.2 80\n3.8 120\n")
        
        results = await nmr_parser.parse_file(test_file)
        
        assert len(results) == 1
        result = results[0]
        assert result["spectrum_type"] == "generic_nmr"
        assert len(result["chemical_shifts"]) == 3
        assert len(result["intensities"]) == 3
        assert result["chemical_shifts"][0] == 7.5
        assert result["intensities"][0] == 100
    
    @pytest.mark.asyncio
    async def test_jcamp_dx_parsing(self, nmr_parser, tmp_path):
        """Test JCAMP-DX format parsing."""
        # Create test JCAMP-DX file
        jcamp_content = """##TITLE=Test Spectrum
##SOLVENT=CDCl3
##OBSERVE FREQUENCY=400.13
##XYDATA=(X++(Y..Y))
7.5 100
7.2 80
3.8 120
##END
"""
        test_file = tmp_path / "test_spectrum.jdx"
        test_file.write_text(jcamp_content)
        
        results = await nmr_parser.parse_file(test_file)
        
        assert len(results) == 1
        result = results[0]
        assert result["spectrum_type"] == "1H_NMR"
        assert result["metadata"]["title"] == "Test Spectrum"
        assert result["metadata"]["solvent"] == "CDCl3"
        assert result["metadata"]["frequency"] == 400.13
    
    @pytest.mark.asyncio
    async def test_stream_parsing(self, nmr_parser):
        """Test NMR stream data parsing."""
        stream_data = json.dumps({
            "type": "1H_NMR",
            "shifts": [7.5, 7.2, 3.8],
            "intensities": [100, 80, 120],
            "integrations": [1.0, 2.0, 3.0],
            "parameters": {"solvent": "CDCl3", "temperature": 298}
        })
        
        results = await nmr_parser.parse_stream(stream_data)
        
        assert len(results) == 1
        result = results[0]
        assert result["spectrum_type"] == "1H_NMR"
        assert len(result["chemical_shifts"]) == 3
        assert result["metadata"]["solvent"] == "CDCl3"
    
    @pytest.mark.asyncio
    async def test_invalid_stream_data(self, nmr_parser):
        """Test handling of invalid stream data."""
        invalid_data = "invalid json"
        
        results = await nmr_parser.parse_stream(invalid_data)
        
        assert len(results) == 0


class TestLCMSDataParser:
    """Test cases for LC-MS data parser."""
    
    @pytest.fixture
    def lcms_parser(self) -> LCMSDataParser:
        """Create LC-MS parser instance."""
        return LCMSDataParser()
    
    @pytest.mark.asyncio
    async def test_csv_parsing(self, lcms_parser, tmp_path):
        """Test CSV LC-MS data parsing."""
        # Create test CSV file
        csv_content = """time,intensity,mass
1.0,1000,200.1
2.0,1500,200.1
3.0,800,200.1
4.0,600,200.1
"""
        test_file = tmp_path / "test_lcms.csv"
        test_file.write_text(csv_content)
        
        results = await lcms_parser.parse_file(test_file)
        
        assert len(results) == 1
        result = results[0]
        assert result["data_type"] == "lcms_chromatogram"
        assert len(result["chromatogram"]) == 4
        assert result["total_ion_current"] == 3900
        assert result["retention_time_range"] == [1.0, 4.0]
    
    @pytest.mark.asyncio
    async def test_mzml_parsing(self, lcms_parser, tmp_path):
        """Test mzML file parsing."""
        # Create mock mzML file
        test_file = tmp_path / "test_spectrum.mzml"
        test_file.write_text("<?xml version='1.0'?><mzML>mock mzML content</mzML>")
        
        results = await lcms_parser.parse_file(test_file)
        
        assert len(results) == 1
        result = results[0]
        assert result["data_type"] == "mzml"
        assert result["metadata"]["format"] == "mzML"
    
    @pytest.mark.asyncio
    async def test_stream_parsing(self, lcms_parser):
        """Test LC-MS stream data parsing."""
        stream_data = json.dumps({
            "chromatogram": [
                {"retention_time": 1.0, "intensity": 1000},
                {"retention_time": 2.0, "intensity": 1500}
            ],
            "spectra": [],
            "tic": 2500,
            "parameters": {"method": "C18_gradient"}
        })
        
        results = await lcms_parser.parse_stream(stream_data)
        
        assert len(results) == 1
        result = results[0]
        assert result["data_type"] == "lcms"
        assert len(result["chromatogram"]) == 2
        assert result["total_ion_current"] == 2500
        assert result["metadata"]["method"] == "C18_gradient"


class TestAssayDataParser:
    """Test cases for assay data parser."""
    
    @pytest.fixture
    def assay_parser(self) -> AssayDataParser:
        """Create assay parser instance."""
        return AssayDataParser()
    
    @pytest.mark.asyncio
    async def test_csv_parsing(self, assay_parser, tmp_path):
        """Test CSV assay data parsing."""
        csv_content = """compound_id,assay_type,result,unit,plate_id,well_id
COMP001,IC50,1.5,uM,PLATE01,A1
COMP002,IC50,3.2,uM,PLATE01,A2
COMP003,IC50,0.8,uM,PLATE01,A3
"""
        test_file = tmp_path / "assay_results.csv"
        test_file.write_text(csv_content)
        
        results = await assay_parser.parse_file(test_file)
        
        assert len(results) == 1
        result = results[0]
        assert result["data_type"] == "assay_results"
        assert len(result["results"]) == 3
        assert result["results"][0]["compound_id"] == "COMP001"
        assert result["results"][0]["result_value"] == 1.5
        assert result["results"][0]["unit"] == "uM"
    
    @pytest.mark.asyncio
    async def test_stream_parsing(self, assay_parser):
        """Test assay stream data parsing."""
        stream_data = json.dumps({
            "results": [
                {"compound_id": "COMP001", "result_value": 1.5, "unit": "uM"},
                {"compound_id": "COMP002", "result_value": 3.2, "unit": "uM"}
            ],
            "type": "IC50",
            "parameters": {"plate_id": "PLATE01", "date": "2024-01-01"}
        })
        
        results = await assay_parser.parse_stream(stream_data)
        
        assert len(results) == 1
        result = results[0]
        assert result["data_type"] == "assay"
        assert len(result["results"]) == 2
        assert result["assay_type"] == "IC50"
        assert result["metadata"]["plate_id"] == "PLATE01"


class TestDataIngestionManager:
    """Test cases for DataIngestionManager."""
    
    @pytest.fixture
    def sample_validator(self):
        """Create sample validator."""
        validator = Mock()
        validator.validate_data_point = AsyncMock(return_value=Mock(
            is_valid=True,
            confidence_score=0.9,
            errors=[],
            warnings=[]
        ))
        return validator
    
    @pytest.fixture
    def ingestion_manager(self, sample_validator) -> DataIngestionManager:
        """Create data ingestion manager."""
        return DataIngestionManager(validator=sample_validator)
    
    def test_register_connection(self, ingestion_manager):
        """Test connection registration."""
        connection = InstrumentConnection(
            instrument_id="test_instrument",
            instrument_type="nmr",
            connection_type="api",
            endpoint="http://test:8080",
            polling_interval=30.0
        )
        
        ingestion_manager.register_connection(connection)
        
        assert "test_instrument" in ingestion_manager.connections
        assert ingestion_manager.connections["test_instrument"] == connection
    
    def test_register_parser(self, ingestion_manager):
        """Test parser registration."""
        parser = NMRDataParser()
        
        ingestion_manager.register_parser("nmr", parser)
        
        assert "nmr" in ingestion_manager.parsers
        assert ingestion_manager.parsers["nmr"] == parser
    
    def test_callback_registration(self, ingestion_manager):
        """Test callback registration."""
        data_callback = Mock()
        error_callback = Mock()
        
        ingestion_manager.set_data_callback(data_callback)
        ingestion_manager.set_error_callback(error_callback)
        
        assert ingestion_manager.data_callback == data_callback
        assert ingestion_manager.error_callback == error_callback
    
    @pytest.mark.asyncio
    async def test_initialize_connections(self, ingestion_manager):
        """Test connection initialization."""
        # Register test connection
        connection = InstrumentConnection(
            instrument_id="test_api",
            instrument_type="nmr",
            connection_type="api",
            endpoint="http://test:8080",
            polling_interval=30.0
        )
        ingestion_manager.register_connection(connection)
        
        # Mock successful connection
        with patch('chemi_accel.autonomous.data_ingestion.RESTAPIIngestion.connect', return_value=True):
            await ingestion_manager.initialize_connections()
            
            assert "test_api" in ingestion_manager.apis
    
    @pytest.mark.asyncio
    async def test_connection_status(self, ingestion_manager):
        """Test connection status retrieval."""
        # Register test connection
        connection = InstrumentConnection(
            instrument_id="test_instrument",
            instrument_type="nmr",
            connection_type="api",
            endpoint="http://test:8080",
            polling_interval=30.0
        )
        ingestion_manager.register_connection(connection)
        
        status = await ingestion_manager.get_connection_status()
        
        assert "test_instrument" in status
        assert status["test_instrument"]["instrument_type"] == "nmr"
        assert status["test_instrument"]["connection_type"] == "api"
        assert status["test_instrument"]["is_configured"] is True
        assert status["test_instrument"]["is_connected"] is False
    
    @pytest.mark.asyncio
    async def test_manual_data_fetch(self, ingestion_manager):
        """Test manual data fetching."""
        # Setup mock API
        mock_api = AsyncMock()
        mock_api.fetch_data.return_value = [
            ExperimentalDataPoint(
                data_id="test_123",
                experiment_id="exp_456",
                instrument_id="test_instrument",
                timestamp=datetime.now(timezone.utc),
                data_type="nmr",
                raw_data={"test": "data"}
            )
        ]
        ingestion_manager.apis["test_instrument"] = mock_api
        
        data_points = await ingestion_manager.manual_data_fetch("test_instrument")
        
        assert len(data_points) == 1
        assert data_points[0].data_id == "test_123"
        mock_api.fetch_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_manual_fetch_no_connection(self, ingestion_manager):
        """Test manual fetch with no connection."""
        with pytest.raises(ValueError, match="No active connection"):
            await ingestion_manager.manual_data_fetch("nonexistent")


class TestFactoryFunctions:
    """Test cases for factory functions."""
    
    def test_create_default_parsers(self):
        """Test default parser creation."""
        parsers = create_default_parsers()
        
        assert "nmr" in parsers
        assert "lcms" in parsers
        assert "assay" in parsers
        assert isinstance(parsers["nmr"], NMRDataParser)
        assert isinstance(parsers["lcms"], LCMSDataParser)
        assert isinstance(parsers["assay"], AssayDataParser)
    
    def test_create_sample_connections(self):
        """Test sample connection creation."""
        connections = create_sample_connections()
        
        assert len(connections) == 3
        
        # Check NMR connection
        nmr_conn = next(c for c in connections if c.instrument_type == "nmr")
        assert nmr_conn.instrument_id == "nmr_400mhz"
        assert nmr_conn.connection_type == "file_watcher"
        assert nmr_conn.polling_interval == 30.0
        
        # Check LC-MS connection
        lcms_conn = next(c for c in connections if c.instrument_type == "lcms")
        assert lcms_conn.instrument_id == "lcms_qtof"
        assert lcms_conn.connection_type == "api"
        assert "api_key" in lcms_conn.credentials
        
        # Check assay connection
        assay_conn = next(c for c in connections if c.instrument_type == "assay")
        assert assay_conn.instrument_id == "assay_reader"
        assert assay_conn.connection_type == "api"


class TestIntegrationScenarios:
    """Integration test scenarios."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_ingestion(self, tmp_path):
        """Test complete end-to-end data ingestion."""
        # Create validator
        validator = create_nmr_validator()
        
        # Create manager
        manager = DataIngestionManager(validator=validator)
        
        # Register parser
        manager.register_parser("nmr", NMRDataParser())
        
        # Create test data file
        test_file = tmp_path / "test_nmr.txt"
        test_file.write_text("7.5 100\n7.2 80\n3.8 120\n")
        
        # Register file watcher connection
        connection = InstrumentConnection(
            instrument_id="test_nmr",
            instrument_type="nmr",
            connection_type="file_watcher",
            endpoint=str(tmp_path),
            polling_interval=1.0
        )
        manager.register_connection(connection)
        
        # Initialize connections
        await manager.initialize_connections()
        
        # Check connection status
        status = await manager.get_connection_status()
        assert status["test_nmr"]["is_connected"] is True
        
        # Manually fetch data
        data_points = await manager.manual_data_fetch("test_nmr")
        
        assert len(data_points) == 1
        assert data_points[0].instrument_id == "test_nmr"
        assert data_points[0].data_type == "nmr"
        assert "chemical_shifts" in data_points[0].raw_data
    
    @pytest.mark.asyncio
    async def test_multi_instrument_scenario(self):
        """Test scenario with multiple instruments."""
        validator = DataValidator()
        manager = DataIngestionManager(validator=validator)
        
        # Register multiple parsers
        parsers = create_default_parsers()
        for instrument_type, parser in parsers.items():
            manager.register_parser(instrument_type, parser)
        
        # Register multiple connections
        connections = create_sample_connections()
        for connection in connections:
            manager.register_connection(connection)
        
        # Check all connections registered
        assert len(manager.connections) == 3
        assert "nmr_400mhz" in manager.connections
        assert "lcms_qtof" in manager.connections
        assert "assay_reader" in manager.connections
        
        # Check all parsers registered
        assert len(manager.parsers) == 3
        assert "nmr" in manager.parsers
        assert "lcms" in manager.parsers
        assert "assay" in manager.parsers
    
    @pytest.mark.asyncio
    async def test_error_handling_scenario(self):
        """Test error handling in ingestion."""
        validator = Mock()
        manager = DataIngestionManager(validator=validator)
        
        # Setup error callback
        error_records = []
        def error_callback(source, error):
            error_records.append((source, str(error)))
        
        manager.set_error_callback(error_callback)
        
        # Register connection with invalid endpoint
        connection = InstrumentConnection(
            instrument_id="invalid_api",
            instrument_type="nmr",
            connection_type="api",
            endpoint="http://invalid:9999",
            polling_interval=1.0,
            timeout=1.0
        )
        manager.register_connection(connection)
        
        # Attempt initialization (should fail gracefully)
        await manager.initialize_connections()
        
        # Check that no APIs were created for failed connections
        assert "invalid_api" not in manager.apis