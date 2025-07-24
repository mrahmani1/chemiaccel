"""
Real-time Experimental Data Ingestion and Validation

This module provides robust APIs and data parsers for ingesting experimental results
from automated laboratory equipment including analytical instruments and assay readouts.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, Callable, AsyncGenerator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
import aiohttp
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from .data_validation import DataValidator, ValidationResult


logger = logging.getLogger(__name__)


@dataclass
class ExperimentalDataPoint:
    """Represents a single experimental data point."""
    data_id: str
    experiment_id: str
    instrument_id: str
    timestamp: datetime
    data_type: str
    raw_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_flags: List[str] = field(default_factory=list)
    processing_status: str = "pending"


@dataclass
class InstrumentConnection:
    """Configuration for connecting to laboratory instruments."""
    instrument_id: str
    instrument_type: str
    connection_type: str  # "api", "file_watcher", "tcp", "serial"
    endpoint: str
    credentials: Optional[Dict[str, str]] = None
    polling_interval: float = 30.0
    timeout: float = 10.0
    retry_attempts: int = 3
    is_active: bool = True


class DataIngestionAPI(ABC):
    """Abstract base class for data ingestion APIs."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the data source."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the data source."""
        pass
    
    @abstractmethod
    async def fetch_data(self) -> List[ExperimentalDataPoint]:
        """Fetch new experimental data."""
        pass
    
    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate that the connection is still active."""
        pass


class RESTAPIIngestion(DataIngestionAPI):
    """Data ingestion via REST API endpoints."""
    
    def __init__(self, connection: InstrumentConnection):
        self.connection = connection
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_fetch_time: Optional[datetime] = None
    
    async def connect(self) -> bool:
        """Establish HTTP session."""
        try:
            timeout = aiohttp.ClientTimeout(total=self.connection.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            # Test connection
            async with self.session.get(f"{self.connection.endpoint}/health") as response:
                if response.status == 200:
                    logger.info(f"Connected to {self.connection.instrument_id}")
                    return True
                else:
                    logger.error(f"Connection failed: HTTP {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Connection error for {self.connection.instrument_id}: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def fetch_data(self) -> List[ExperimentalDataPoint]:
        """Fetch data from REST endpoint."""
        if not self.session:
            raise RuntimeError("Not connected to data source")
        
        data_points = []
        
        try:
            # Build query parameters
            params = {}
            if self.last_fetch_time:
                params["since"] = self.last_fetch_time.isoformat()
            
            # Add authentication if configured
            headers = {}
            if self.connection.credentials:
                if "api_key" in self.connection.credentials:
                    headers["X-API-Key"] = self.connection.credentials["api_key"]
                elif "bearer_token" in self.connection.credentials:
                    headers["Authorization"] = f"Bearer {self.connection.credentials['bearer_token']}"
            
            async with self.session.get(
                f"{self.connection.endpoint}/data",
                params=params,
                headers=headers
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    for record in data.get("results", []):
                        data_point = ExperimentalDataPoint(
                            data_id=record.get("id", str(uuid.uuid4())),
                            experiment_id=record.get("experiment_id", "unknown"),
                            instrument_id=self.connection.instrument_id,
                            timestamp=datetime.fromisoformat(record.get("timestamp", datetime.now().isoformat())),
                            data_type=record.get("data_type", "unknown"),
                            raw_data=record.get("data", {}),
                            metadata=record.get("metadata", {}),
                            quality_flags=record.get("quality_flags", [])
                        )
                        data_points.append(data_point)
                    
                    self.last_fetch_time = datetime.now(timezone.utc)
                    logger.info(f"Fetched {len(data_points)} data points from {self.connection.instrument_id}")
                
                else:
                    logger.error(f"Data fetch failed: HTTP {response.status}")
        
        except Exception as e:
            logger.error(f"Error fetching data from {self.connection.instrument_id}: {e}")
        
        return data_points
    
    async def validate_connection(self) -> bool:
        """Validate connection health."""
        if not self.session:
            return False
        
        try:
            async with self.session.get(f"{self.connection.endpoint}/health") as response:
                return response.status == 200
        except:
            return False


class FileWatcherIngestion(DataIngestionAPI):
    """Data ingestion via file system monitoring."""
    
    def __init__(self, connection: InstrumentConnection, data_parser: 'DataParser'):
        self.connection = connection
        self.data_parser = data_parser
        self.watch_directory = Path(connection.endpoint)
        self.processed_files: set = set()
        self.is_connected = False
    
    async def connect(self) -> bool:
        """Initialize file watcher."""
        try:
            if not self.watch_directory.exists():
                logger.error(f"Watch directory does not exist: {self.watch_directory}")
                return False
            
            self.is_connected = True
            logger.info(f"File watcher initialized for {self.watch_directory}")
            return True
            
        except Exception as e:
            logger.error(f"File watcher initialization failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Stop file watcher."""
        self.is_connected = False
    
    async def fetch_data(self) -> List[ExperimentalDataPoint]:
        """Scan for new files and parse data."""
        if not self.is_connected:
            raise RuntimeError("File watcher not connected")
        
        data_points = []
        
        try:
            # Scan for new files
            pattern = "*.txt" if self.connection.instrument_type == "text" else "*.*"
            new_files = [
                f for f in self.watch_directory.glob(pattern)
                if f.is_file() and str(f) not in self.processed_files
            ]
            
            for file_path in new_files:
                try:
                    # Parse file data
                    parsed_data = await self.data_parser.parse_file(file_path)
                    
                    for data_record in parsed_data:
                        data_point = ExperimentalDataPoint(
                            data_id=str(uuid.uuid4()),
                            experiment_id=data_record.get("experiment_id", "file_based"),
                            instrument_id=self.connection.instrument_id,
                            timestamp=datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc),
                            data_type=self.connection.instrument_type,
                            raw_data=data_record,
                            metadata={
                                "source_file": str(file_path),
                                "file_size": file_path.stat().st_size,
                                "file_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                            }
                        )
                        data_points.append(data_point)
                    
                    self.processed_files.add(str(file_path))
                    logger.info(f"Processed file: {file_path}")
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
            
            logger.info(f"File watcher found {len(data_points)} new data points")
        
        except Exception as e:
            logger.error(f"File watcher scan error: {e}")
        
        return data_points
    
    async def validate_connection(self) -> bool:
        """Validate file system access."""
        return self.is_connected and self.watch_directory.exists()


class DataParser(ABC):
    """Abstract base class for instrument-specific data parsers."""
    
    @abstractmethod
    async def parse_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse data from a file."""
        pass
    
    @abstractmethod
    async def parse_stream(self, data_stream: str) -> List[Dict[str, Any]]:
        """Parse data from a stream."""
        pass


class NMRDataParser(DataParser):
    """Parser for NMR spectroscopy data."""
    
    async def parse_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse NMR data file."""
        data_records = []
        
        try:
            # Handle different NMR file formats
            if file_path.suffix.lower() == '.jdx':
                data_records = await self._parse_jcamp_dx(file_path)
            elif file_path.suffix.lower() == '.mnova':
                data_records = await self._parse_mnova(file_path)
            else:
                # Generic text-based format
                data_records = await self._parse_generic_nmr(file_path)
        
        except Exception as e:
            logger.error(f"NMR parsing error for {file_path}: {e}")
        
        return data_records
    
    async def parse_stream(self, data_stream: str) -> List[Dict[str, Any]]:
        """Parse NMR data from stream."""
        try:
            # Parse JSON stream for real-time NMR data
            data = json.loads(data_stream)
            return [self._normalize_nmr_data(data)]
        except Exception as e:
            logger.error(f"NMR stream parsing error: {e}")
            return []
    
    async def _parse_jcamp_dx(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse JCAMP-DX format NMR files."""
        data_record = {
            "spectrum_type": "1H_NMR",
            "chemical_shifts": [],
            "intensities": [],
            "integration_values": [],
            "metadata": {}
        }
        
        with open(file_path, 'r') as f:
            content = f.read()
            
            # Extract key metadata
            for line in content.split('\n'):
                if line.startswith('##TITLE='):
                    data_record["metadata"]["title"] = line.split('=', 1)[1].strip()
                elif line.startswith('##SOLVENT='):
                    data_record["metadata"]["solvent"] = line.split('=', 1)[1].strip()
                elif line.startswith('##OBSERVE FREQUENCY='):
                    data_record["metadata"]["frequency"] = float(line.split('=', 1)[1].strip())
            
            # Parse spectral data (simplified)
            if '##XYDATA=' in content:
                xy_section = content.split('##XYDATA=')[1].split('##')[0]
                for line in xy_section.strip().split('\n'):
                    if line.strip() and not line.startswith('('):
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            try:
                                shift = float(parts[0])
                                intensity = float(parts[1])
                                data_record["chemical_shifts"].append(shift)
                                data_record["intensities"].append(intensity)
                            except ValueError:
                                continue
        
        return [data_record]
    
    async def _parse_mnova(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse MNova format files."""
        # Placeholder for MNova-specific parsing
        return [{
            "spectrum_type": "mnova_format",
            "file_path": str(file_path),
            "parsed": False,
            "error": "MNova parser not yet implemented"
        }]
    
    async def _parse_generic_nmr(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse generic NMR text files."""
        try:
            df = pd.read_csv(file_path, sep=r'\s+', header=None)
            
            return [{
                "spectrum_type": "generic_nmr",
                "chemical_shifts": df.iloc[:, 0].tolist() if len(df.columns) > 0 else [],
                "intensities": df.iloc[:, 1].tolist() if len(df.columns) > 1 else [],
                "data_points": len(df),
                "metadata": {
                    "columns": len(df.columns),
                    "rows": len(df)
                }
            }]
        
        except Exception as e:
            logger.error(f"Generic NMR parsing error: {e}")
            return []
    
    def _normalize_nmr_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize NMR data to standard format."""
        return {
            "spectrum_type": raw_data.get("type", "unknown"),
            "chemical_shifts": raw_data.get("shifts", []),
            "intensities": raw_data.get("intensities", []),
            "integration_values": raw_data.get("integrations", []),
            "metadata": raw_data.get("parameters", {})
        }


class LCMSDataParser(DataParser):
    """Parser for LC-MS analytical data."""
    
    async def parse_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse LC-MS data file."""
        data_records = []
        
        try:
            if file_path.suffix.lower() == '.mzml':
                data_records = await self._parse_mzml(file_path)
            elif file_path.suffix.lower() == '.raw':
                data_records = await self._parse_raw_file(file_path)
            else:
                data_records = await self._parse_csv_lcms(file_path)
        
        except Exception as e:
            logger.error(f"LC-MS parsing error for {file_path}: {e}")
        
        return data_records
    
    async def parse_stream(self, data_stream: str) -> List[Dict[str, Any]]:
        """Parse LC-MS data from stream."""
        try:
            data = json.loads(data_stream)
            return [self._normalize_lcms_data(data)]
        except Exception as e:
            logger.error(f"LC-MS stream parsing error: {e}")
            return []
    
    async def _parse_mzml(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse mzML format files."""
        # Simplified mzML parsing - would use proper XML parser in production
        return [{
            "data_type": "mzml",
            "file_path": str(file_path),
            "chromatogram": [],
            "mass_spectra": [],
            "metadata": {"format": "mzML"}
        }]
    
    async def _parse_raw_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse instrument RAW files."""
        return [{
            "data_type": "raw_file",
            "file_path": str(file_path),
            "requires_conversion": True,
            "metadata": {"format": "RAW"}
        }]
    
    async def _parse_csv_lcms(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse CSV format LC-MS data."""
        try:
            df = pd.read_csv(file_path)
            
            chromatogram_data = []
            if "time" in df.columns and "intensity" in df.columns:
                for _, row in df.iterrows():
                    chromatogram_data.append({
                        "retention_time": row["time"],
                        "intensity": row["intensity"],
                        "mass": row.get("mass", None)
                    })
            
            return [{
                "data_type": "lcms_chromatogram",
                "chromatogram": chromatogram_data,
                "total_ion_current": df["intensity"].sum() if "intensity" in df.columns else 0,
                "retention_time_range": [df["time"].min(), df["time"].max()] if "time" in df.columns else [],
                "metadata": {
                    "columns": list(df.columns),
                    "data_points": len(df)
                }
            }]
        
        except Exception as e:
            logger.error(f"CSV LC-MS parsing error: {e}")
            return []
    
    def _normalize_lcms_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize LC-MS data to standard format."""
        return {
            "data_type": "lcms",
            "chromatogram": raw_data.get("chromatogram", []),
            "mass_spectra": raw_data.get("spectra", []),
            "total_ion_current": raw_data.get("tic", 0),
            "metadata": raw_data.get("parameters", {})
        }


class AssayDataParser(DataParser):
    """Parser for biochemical assay data."""
    
    async def parse_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse assay data file."""
        try:
            if file_path.suffix.lower() == '.xlsx':
                return await self._parse_excel_assay(file_path)
            else:
                return await self._parse_csv_assay(file_path)
        except Exception as e:
            logger.error(f"Assay parsing error for {file_path}: {e}")
            return []
    
    async def parse_stream(self, data_stream: str) -> List[Dict[str, Any]]:
        """Parse assay data from stream."""
        try:
            data = json.loads(data_stream)
            return [self._normalize_assay_data(data)]
        except Exception as e:
            logger.error(f"Assay stream parsing error: {e}")
            return []
    
    async def _parse_excel_assay(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse Excel-based assay data."""
        try:
            df = pd.read_excel(file_path)
            
            assay_results = []
            for _, row in df.iterrows():
                result = {
                    "compound_id": row.get("compound_id", "unknown"),
                    "assay_type": row.get("assay_type", "unknown"),
                    "result_value": row.get("result", None),
                    "unit": row.get("unit", ""),
                    "plate_id": row.get("plate_id", ""),
                    "well_id": row.get("well_id", ""),
                    "replicate": row.get("replicate", 1),
                    "metadata": {k: v for k, v in row.items() if k not in [
                        "compound_id", "assay_type", "result", "unit", "plate_id", "well_id", "replicate"
                    ]}
                }
                assay_results.append(result)
            
            return [{
                "data_type": "assay_results",
                "results": assay_results,
                "total_compounds": len(set(r["compound_id"] for r in assay_results)),
                "metadata": {"source_file": str(file_path)}
            }]
        
        except Exception as e:
            logger.error(f"Excel assay parsing error: {e}")
            return []
    
    async def _parse_csv_assay(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse CSV-based assay data."""
        try:
            df = pd.read_csv(file_path)
            
            # Similar logic to Excel parsing
            assay_results = []
            for _, row in df.iterrows():
                result = {
                    "compound_id": row.get("compound_id", "unknown"),
                    "assay_type": row.get("assay_type", "unknown"),
                    "result_value": row.get("result", None),
                    "unit": row.get("unit", ""),
                    "metadata": dict(row)
                }
                assay_results.append(result)
            
            return [{
                "data_type": "assay_results",
                "results": assay_results,
                "metadata": {"source_file": str(file_path)}
            }]
        
        except Exception as e:
            logger.error(f"CSV assay parsing error: {e}")
            return []
    
    def _normalize_assay_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize assay data to standard format."""
        return {
            "data_type": "assay",
            "results": raw_data.get("results", []),
            "assay_type": raw_data.get("type", "unknown"),
            "metadata": raw_data.get("parameters", {})
        }


class DataIngestionManager:
    """
    Main manager for coordinating data ingestion from multiple sources.
    
    Handles real-time data streaming, validation, and error recovery.
    """
    
    def __init__(self, validator: Optional[DataValidator] = None):
        self.connections: Dict[str, InstrumentConnection] = {}
        self.apis: Dict[str, DataIngestionAPI] = {}
        self.parsers: Dict[str, DataParser] = {}
        self.validator = validator or DataValidator()
        self.data_buffer: List[ExperimentalDataPoint] = []
        self.error_callback: Optional[Callable] = None
        self.data_callback: Optional[Callable] = None
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def register_connection(self, connection: InstrumentConnection) -> None:
        """Register a new instrument connection."""
        self.connections[connection.instrument_id] = connection
        logger.info(f"Registered connection: {connection.instrument_id}")
    
    def register_parser(self, instrument_type: str, parser: DataParser) -> None:
        """Register a data parser for a specific instrument type."""
        self.parsers[instrument_type] = parser
        logger.info(f"Registered parser for: {instrument_type}")
    
    def set_data_callback(self, callback: Callable[[List[ExperimentalDataPoint]], None]) -> None:
        """Set callback for handling validated data."""
        self.data_callback = callback
    
    def set_error_callback(self, callback: Callable[[str, Exception], None]) -> None:
        """Set callback for handling errors."""
        self.error_callback = callback
    
    async def initialize_connections(self) -> None:
        """Initialize all registered connections."""
        for instrument_id, connection in self.connections.items():
            try:
                # Create appropriate API instance
                if connection.connection_type == "api":
                    api = RESTAPIIngestion(connection)
                elif connection.connection_type == "file_watcher":
                    parser = self.parsers.get(connection.instrument_type)
                    if not parser:
                        logger.error(f"No parser registered for {connection.instrument_type}")
                        continue
                    api = FileWatcherIngestion(connection, parser)
                else:
                    logger.warning(f"Unsupported connection type: {connection.connection_type}")
                    continue
                
                # Attempt connection
                if await api.connect():
                    self.apis[instrument_id] = api
                    logger.info(f"Successfully connected to {instrument_id}")
                else:
                    logger.error(f"Failed to connect to {instrument_id}")
            
            except Exception as e:
                logger.error(f"Error initializing connection {instrument_id}: {e}")
                if self.error_callback:
                    self.error_callback(instrument_id, e)
    
    async def start_data_ingestion(self) -> None:
        """Start real-time data ingestion from all sources."""
        if self.is_running:
            logger.warning("Data ingestion already running")
            return
        
        self.is_running = True
        logger.info("Starting data ingestion...")
        
        # Create ingestion tasks for each connected API
        tasks = []
        for instrument_id, api in self.apis.items():
            connection = self.connections[instrument_id]
            task = asyncio.create_task(
                self._ingestion_loop(instrument_id, api, connection.polling_interval)
            )
            tasks.append(task)
        
        # Create validation and processing task
        validation_task = asyncio.create_task(self._validation_loop())
        tasks.append(validation_task)
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Data ingestion error: {e}")
        finally:
            self.is_running = False
    
    async def stop_data_ingestion(self) -> None:
        """Stop data ingestion and clean up connections."""
        self.is_running = False
        
        for api in self.apis.values():
            await api.disconnect()
        
        self.apis.clear()
        logger.info("Data ingestion stopped")
    
    async def _ingestion_loop(self, instrument_id: str, api: DataIngestionAPI, polling_interval: float) -> None:
        """Main ingestion loop for a single instrument."""
        while self.is_running:
            try:
                # Validate connection health
                if not await api.validate_connection():
                    logger.warning(f"Connection lost for {instrument_id}, attempting reconnect...")
                    if not await api.connect():
                        logger.error(f"Reconnection failed for {instrument_id}")
                        await asyncio.sleep(polling_interval * 2)  # Wait longer before retry
                        continue
                
                # Fetch new data
                data_points = await api.fetch_data()
                
                if data_points:
                    # Add to buffer for validation
                    self.data_buffer.extend(data_points)
                    logger.debug(f"Added {len(data_points)} data points from {instrument_id}")
            
            except Exception as e:
                logger.error(f"Error in ingestion loop for {instrument_id}: {e}")
                if self.error_callback:
                    self.error_callback(instrument_id, e)
            
            await asyncio.sleep(polling_interval)
    
    async def _validation_loop(self) -> None:
        """Main validation and processing loop."""
        while self.is_running:
            try:
                if self.data_buffer:
                    # Process batch of data points
                    batch = self.data_buffer[:100]  # Process in batches
                    self.data_buffer = self.data_buffer[100:]
                    
                    validated_data = []
                    
                    for data_point in batch:
                        # Validate data point
                        validation_result = await self.validator.validate_data_point(data_point)
                        
                        if validation_result.is_valid:
                            data_point.processing_status = "validated"
                            validated_data.append(data_point)
                        else:
                            data_point.processing_status = "failed_validation"
                            data_point.quality_flags.extend(validation_result.errors)
                            logger.warning(f"Validation failed for data point {data_point.data_id}: {validation_result.errors}")
                    
                    # Send validated data to callback
                    if validated_data and self.data_callback:
                        await asyncio.get_event_loop().run_in_executor(
                            self.executor, self.data_callback, validated_data
                        )
            
            except Exception as e:
                logger.error(f"Error in validation loop: {e}")
                if self.error_callback:
                    self.error_callback("validation", e)
            
            await asyncio.sleep(1.0)  # Validation runs every second
    
    async def get_connection_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all connections."""
        status = {}
        
        for instrument_id, connection in self.connections.items():
            api = self.apis.get(instrument_id)
            
            status[instrument_id] = {
                "instrument_type": connection.instrument_type,
                "connection_type": connection.connection_type,
                "is_configured": connection.is_active,
                "is_connected": api is not None,
                "is_healthy": await api.validate_connection() if api else False,
                "last_check": datetime.now().isoformat()
            }
        
        return status
    
    async def manual_data_fetch(self, instrument_id: str) -> List[ExperimentalDataPoint]:
        """Manually fetch data from a specific instrument."""
        api = self.apis.get(instrument_id)
        if not api:
            raise ValueError(f"No active connection for instrument: {instrument_id}")
        
        return await api.fetch_data()


def create_default_parsers() -> Dict[str, DataParser]:
    """Create default set of data parsers."""
    return {
        "nmr": NMRDataParser(),
        "lcms": LCMSDataParser(),
        "assay": AssayDataParser()
    }


def create_sample_connections() -> List[InstrumentConnection]:
    """Create sample instrument connections for testing."""
    return [
        InstrumentConnection(
            instrument_id="nmr_400mhz",
            instrument_type="nmr",
            connection_type="file_watcher",
            endpoint="/lab/data/nmr/output",
            polling_interval=30.0
        ),
        InstrumentConnection(
            instrument_id="lcms_qtof",
            instrument_type="lcms",
            connection_type="api",
            endpoint="http://lcms-server:8080/api/v1",
            credentials={"api_key": "lab_automation_key"},
            polling_interval=60.0
        ),
        InstrumentConnection(
            instrument_id="assay_reader",
            instrument_type="assay",
            connection_type="api",
            endpoint="http://plate-reader:9090/api",
            polling_interval=45.0
        )
    ]