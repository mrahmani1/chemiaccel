"""
Experimental Data Integration for Extreme Conditions

This module implements robust data ingestion pipelines for experimental data
obtained from extreme condition tests (e.g., shock experiments, high-temperature
creep, radiation exposure), to calibrate and validate non-equilibrium models.
"""

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import h5py

    HDF5_AVAILABLE = True
except ImportError:
    h5py = None
    HDF5_AVAILABLE = False

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

try:
    from scipy import interpolate, optimize, signal
    from sklearn.cluster import DBSCAN
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    SCIPY_AVAILABLE = True
except ImportError:
    signal = None
    interpolate = None
    optimize = None
    SCIPY_AVAILABLE = False

from ..core.exceptions import ChemiAccelError, DataIngestionError
from ..utils.logging import get_logger
from .simulation_engine import ExtremeConditionType, SimulationResults

logger = get_logger(__name__)


class ExperimentType(Enum):
    """Types of extreme condition experiments."""

    SHOCK_COMPRESSION = "shock_compression"
    HIGH_TEMPERATURE_CREEP = "high_temperature_creep"
    RADIATION_EXPOSURE = "radiation_exposure"
    THERMAL_CYCLING = "thermal_cycling"
    FATIGUE_TESTING = "fatigue_testing"
    IMPACT_TESTING = "impact_testing"
    CORROSION_TESTING = "corrosion_testing"
    TENSILE_TESTING = "tensile_testing"
    COMPRESSION_TESTING = "compression_testing"
    DYNAMIC_MECHANICAL_ANALYSIS = "dynamic_mechanical_analysis"


class DataFormat(Enum):
    """Supported experimental data formats."""

    CSV = "csv"
    EXCEL = "excel"
    HDF5 = "hdf5"
    JSON = "json"
    XML = "xml"
    CUSTOM_TEXT = "custom_text"
    BINARY = "binary"


@dataclass
class ExperimentalConditions:
    """Experimental conditions metadata."""

    # Environmental conditions
    temperature: float = 300.0  # K
    pressure: float = 1.01325e5  # Pa
    atmosphere: str = "air"
    humidity: float = 0.5  # Relative humidity

    # Loading conditions
    strain_rate: float = 1e-3  # s^-1
    stress_rate: float = 1e6  # Pa/s
    frequency: float = 0.0  # Hz (for cyclic loading)
    amplitude: float = 0.0  # Loading amplitude

    # Extreme condition parameters
    radiation_dose: float = 0.0  # Gray
    radiation_type: str = "neutron"
    shock_pressure: float = 0.0  # Pa
    thermal_gradient: float = 0.0  # K/m

    # Time parameters
    duration: float = 3600.0  # seconds
    preconditioning_time: float = 0.0  # seconds

    # Material parameters
    specimen_geometry: str = "cylindrical"
    specimen_dimensions: Dict[str, float] = field(default_factory=dict)
    initial_microstructure: str = "as_received"


@dataclass
class ExperimentalDataPoint:
    """Single experimental data point."""

    timestamp: float
    measured_values: Dict[str, float]
    conditions: ExperimentalConditions
    uncertainty: Dict[str, float] = field(default_factory=dict)
    quality_flag: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentalDataset:
    """Complete experimental dataset."""

    experiment_type: ExperimentType
    experiment_id: str
    data_points: List[ExperimentalDataPoint] = field(default_factory=list)

    # Dataset metadata
    material_composition: Dict[str, float] = field(default_factory=dict)
    processing_history: List[str] = field(default_factory=list)
    measurement_techniques: List[str] = field(default_factory=list)

    # Quality metrics
    data_completeness: float = 1.0
    measurement_accuracy: Dict[str, float] = field(default_factory=dict)
    reproducibility: Dict[str, float] = field(default_factory=dict)

    # Calibration information
    calibration_date: str = ""
    calibration_standards: List[str] = field(default_factory=list)


class BaseDataProcessor(ABC):
    """Base class for experimental data processors."""

    def __init__(self, backend: str = "cpu"):
        self.backend = backend
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def process_raw_data(
        self, raw_data: Any, conditions: ExperimentalConditions
    ) -> ExperimentalDataset:
        """Process raw experimental data into standardized format."""
        pass

    @abstractmethod
    def validate_data_quality(self, dataset: ExperimentalDataset) -> Dict[str, float]:
        """Validate quality of experimental data."""
        pass

    def _move_to_backend(self, array: np.ndarray) -> Union[np.ndarray, Any]:
        """Move array to appropriate backend."""
        if self.backend == "cuda" and CUPY_AVAILABLE:
            return cp.asarray(array)
        else:
            return array

    def _move_to_cpu(self, array: Union[np.ndarray, Any]) -> np.ndarray:
        """Move array back to CPU."""
        if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        else:
            return np.asarray(array)


class ShockExperimentProcessor(BaseDataProcessor):
    """Processor for shock compression experiment data."""

    def __init__(self, backend: str = "cpu"):
        super().__init__(backend)
        self.hugoniot_analyzer = HugoniotAnalyzer()

    def process_raw_data(
        self,
        raw_data: Union[str, pd.DataFrame, Dict],
        conditions: ExperimentalConditions,
    ) -> ExperimentalDataset:
        """Process shock experiment data."""

        self.logger.info("Processing shock experiment data")

        # Handle different input formats
        if isinstance(raw_data, str):
            # File path
            df = self._load_data_file(raw_data)
        elif isinstance(raw_data, pd.DataFrame):
            df = raw_data
        elif isinstance(raw_data, dict):
            df = pd.DataFrame(raw_data)
        else:
            raise DataIngestionError(f"Unsupported data format: {type(raw_data)}")

        # Create dataset
        dataset = ExperimentalDataset(
            experiment_type=ExperimentType.SHOCK_COMPRESSION,
            experiment_id=f"shock_{int(time.time())}",
        )

        # Process each row as a data point
        for idx, row in df.iterrows():
            data_point = self._create_shock_data_point(row, conditions)
            dataset.data_points.append(data_point)

        # Extract material information
        dataset.material_composition = self._extract_material_composition(df)
        dataset.measurement_techniques = [
            "VISAR",
            "pressure_gauge",
            "velocity_interferometry",
        ]

        # Analyze Hugoniot relationship
        hugoniot_params = self.hugoniot_analyzer.analyze_hugoniot(dataset)
        dataset.metadata = {"hugoniot_parameters": hugoniot_params}

        self.logger.info(f"Processed {len(dataset.data_points)} shock data points")

        return dataset

    def _create_shock_data_point(
        self, row: pd.Series, conditions: ExperimentalConditions
    ) -> ExperimentalDataPoint:
        """Create shock data point from row data."""

        # Standard shock measurement columns
        measured_values = {}
        uncertainty = {}

        # Time
        timestamp = row.get("time", row.get("t", 0.0))

        # Pressure measurements
        if "pressure" in row:
            measured_values["pressure"] = row["pressure"]
            uncertainty["pressure"] = row.get(
                "pressure_uncertainty", 0.05 * row["pressure"]
            )

        # Velocity measurements
        if "particle_velocity" in row:
            measured_values["particle_velocity"] = row["particle_velocity"]
            uncertainty["particle_velocity"] = row.get(
                "velocity_uncertainty", 0.02 * row["particle_velocity"]
            )

        if "shock_velocity" in row:
            measured_values["shock_velocity"] = row["shock_velocity"]
            uncertainty["shock_velocity"] = row.get(
                "shock_velocity_uncertainty", 0.02 * row["shock_velocity"]
            )

        # Density/compression
        if "density" in row:
            measured_values["density"] = row["density"]
            uncertainty["density"] = row.get(
                "density_uncertainty", 0.01 * row["density"]
            )

        # Temperature (if available)
        if "temperature" in row:
            measured_values["temperature"] = row["temperature"]
            uncertainty["temperature"] = row.get(
                "temperature_uncertainty", 0.05 * row["temperature"]
            )

        # Quality assessment
        quality_flag = True
        for key, value in measured_values.items():
            if np.isnan(value) or value < 0:
                quality_flag = False
                break

        return ExperimentalDataPoint(
            timestamp=timestamp,
            measured_values=measured_values,
            conditions=conditions,
            uncertainty=uncertainty,
            quality_flag=quality_flag,
            metadata={"measurement_method": "shock_compression"},
        )

    def validate_data_quality(self, dataset: ExperimentalDataset) -> Dict[str, float]:
        """Validate shock experiment data quality."""

        quality_metrics = {}

        # Data completeness
        total_points = len(dataset.data_points)
        valid_points = sum(1 for dp in dataset.data_points if dp.quality_flag)
        quality_metrics["completeness"] = (
            valid_points / total_points if total_points > 0 else 0.0
        )

        # Hugoniot consistency
        if total_points > 5:
            pressures = []
            particle_velocities = []

            for dp in dataset.data_points:
                if (
                    "pressure" in dp.measured_values
                    and "particle_velocity" in dp.measured_values
                ):
                    pressures.append(dp.measured_values["pressure"])
                    particle_velocities.append(dp.measured_values["particle_velocity"])

            if len(pressures) > 3:
                # Check Hugoniot relationship consistency
                hugoniot_r2 = self.hugoniot_analyzer.calculate_hugoniot_fit_quality(
                    pressures, particle_velocities
                )
                quality_metrics["hugoniot_consistency"] = hugoniot_r2

        # Measurement precision
        for measurement_type in ["pressure", "particle_velocity", "shock_velocity"]:
            uncertainties = []
            values = []

            for dp in dataset.data_points:
                if (
                    measurement_type in dp.measured_values
                    and measurement_type in dp.uncertainty
                ):
                    values.append(dp.measured_values[measurement_type])
                    uncertainties.append(dp.uncertainty[measurement_type])

            if values:
                relative_uncertainty = np.mean(
                    np.array(uncertainties) / np.array(values)
                )
                quality_metrics[f"{measurement_type}_precision"] = 1.0 - min(
                    1.0, relative_uncertainty
                )

        return quality_metrics

    def _load_data_file(self, file_path: str) -> pd.DataFrame:
        """Load data from file."""
        path = Path(file_path)

        if path.suffix.lower() == ".csv":
            return pd.read_csv(file_path)
        elif path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        elif path.suffix.lower() == ".json":
            with open(file_path) as f:
                data = json.load(f)
            return pd.DataFrame(data)
        else:
            raise DataIngestionError(f"Unsupported file format: {path.suffix}")

    def _extract_material_composition(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract material composition from data."""
        composition = {}

        # Look for composition columns
        comp_columns = [
            col
            for col in df.columns
            if "composition" in col.lower() or col in ["Fe", "Al", "Cu", "Ni", "Cr"]
        ]

        for col in comp_columns:
            if col in df.columns:
                # Use first non-null value
                value = df[col].dropna().iloc[0] if not df[col].dropna().empty else 0.0
                composition[col] = float(value)

        # Default composition if none found
        if not composition:
            composition = {"Fe": 1.0}

        return composition


class RadiationDataProcessor(BaseDataProcessor):
    """Processor for radiation exposure experiment data."""

    def __init__(self, backend: str = "cpu"):
        super().__init__(backend)
        self.damage_analyzer = RadiationDamageAnalyzer()

    def process_raw_data(
        self,
        raw_data: Union[str, pd.DataFrame, Dict],
        conditions: ExperimentalConditions,
    ) -> ExperimentalDataset:
        """Process radiation exposure experiment data."""

        self.logger.info("Processing radiation exposure data")

        # Handle different input formats
        if isinstance(raw_data, str):
            df = self._load_data_file(raw_data)
        elif isinstance(raw_data, pd.DataFrame):
            df = raw_data
        elif isinstance(raw_data, dict):
            df = pd.DataFrame(raw_data)
        else:
            raise DataIngestionError(f"Unsupported data format: {type(raw_data)}")

        # Create dataset
        dataset = ExperimentalDataset(
            experiment_type=ExperimentType.RADIATION_EXPOSURE,
            experiment_id=f"radiation_{int(time.time())}",
        )

        # Process each row as a data point
        for idx, row in df.iterrows():
            data_point = self._create_radiation_data_point(row, conditions)
            dataset.data_points.append(data_point)

        # Analyze radiation damage evolution
        damage_params = self.damage_analyzer.analyze_damage_evolution(dataset)
        dataset.metadata = {"damage_parameters": damage_params}

        self.logger.info(f"Processed {len(dataset.data_points)} radiation data points")

        return dataset

    def _create_radiation_data_point(
        self, row: pd.Series, conditions: ExperimentalConditions
    ) -> ExperimentalDataPoint:
        """Create radiation data point from row data."""

        measured_values = {}
        uncertainty = {}

        # Time/dose
        timestamp = row.get("time", row.get("exposure_time", 0.0))

        # Radiation dose
        if "dose" in row:
            measured_values["radiation_dose"] = row["dose"]
            uncertainty["radiation_dose"] = row.get(
                "dose_uncertainty", 0.05 * row["dose"]
            )

        # Mechanical properties
        for prop in ["yield_strength", "ultimate_strength", "elongation", "hardness"]:
            if prop in row:
                measured_values[prop] = row[prop]
                uncertainty[prop] = row.get(
                    f"{prop}_uncertainty", 0.05 * abs(row[prop])
                )

        # Microstructural measurements
        for measurement in ["defect_density", "dislocation_density", "void_fraction"]:
            if measurement in row:
                measured_values[measurement] = row[measurement]
                uncertainty[measurement] = row.get(
                    f"{measurement}_uncertainty", 0.1 * row[measurement]
                )

        # Dimensional changes
        if "swelling" in row:
            measured_values["swelling"] = row["swelling"]
            uncertainty["swelling"] = row.get(
                "swelling_uncertainty", 0.05 * abs(row["swelling"])
            )

        return ExperimentalDataPoint(
            timestamp=timestamp,
            measured_values=measured_values,
            conditions=conditions,
            uncertainty=uncertainty,
            quality_flag=True,
            metadata={"radiation_type": conditions.radiation_type},
        )

    def validate_data_quality(self, dataset: ExperimentalDataset) -> Dict[str, float]:
        """Validate radiation experiment data quality."""

        quality_metrics = {}

        # Data completeness
        total_points = len(dataset.data_points)
        valid_points = sum(1 for dp in dataset.data_points if dp.quality_flag)
        quality_metrics["completeness"] = (
            valid_points / total_points if total_points > 0 else 0.0
        )

        # Dose-response consistency
        if total_points > 5:
            doses = []
            properties = []

            for dp in dataset.data_points:
                if "radiation_dose" in dp.measured_values:
                    doses.append(dp.measured_values["radiation_dose"])

                    # Use yield strength as representative property
                    if "yield_strength" in dp.measured_values:
                        properties.append(dp.measured_values["yield_strength"])

            if len(doses) > 3 and len(properties) == len(doses):
                # Check monotonic relationship
                correlation = (
                    np.corrcoef(doses, properties)[0, 1] if len(doses) > 1 else 0.0
                )
                quality_metrics["dose_response_consistency"] = abs(correlation)

        return quality_metrics

    def _load_data_file(self, file_path: str) -> pd.DataFrame:
        """Load radiation data from file."""
        # Similar to shock processor but with radiation-specific handling
        path = Path(file_path)

        if path.suffix.lower() == ".csv":
            return pd.read_csv(file_path)
        elif path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        else:
            raise DataIngestionError(f"Unsupported file format: {path.suffix}")


class HighTemperatureDataProcessor(BaseDataProcessor):
    """Processor for high-temperature creep and related experiments."""

    def __init__(self, backend: str = "cpu"):
        super().__init__(backend)
        self.creep_analyzer = CreepAnalyzer()

    def process_raw_data(
        self,
        raw_data: Union[str, pd.DataFrame, Dict],
        conditions: ExperimentalConditions,
    ) -> ExperimentalDataset:
        """Process high-temperature experiment data."""

        self.logger.info("Processing high-temperature data")

        # Handle different input formats
        if isinstance(raw_data, str):
            df = self._load_data_file(raw_data)
        elif isinstance(raw_data, pd.DataFrame):
            df = raw_data
        elif isinstance(raw_data, dict):
            df = pd.DataFrame(raw_data)
        else:
            raise DataIngestionError(f"Unsupported data format: {type(raw_data)}")

        # Create dataset
        dataset = ExperimentalDataset(
            experiment_type=ExperimentType.HIGH_TEMPERATURE_CREEP,
            experiment_id=f"creep_{int(time.time())}",
        )

        # Process each row as a data point
        for idx, row in df.iterrows():
            data_point = self._create_creep_data_point(row, conditions)
            dataset.data_points.append(data_point)

        # Analyze creep behavior
        creep_params = self.creep_analyzer.analyze_creep_behavior(dataset)
        dataset.metadata = {"creep_parameters": creep_params}

        self.logger.info(f"Processed {len(dataset.data_points)} creep data points")

        return dataset

    def _create_creep_data_point(
        self, row: pd.Series, conditions: ExperimentalConditions
    ) -> ExperimentalDataPoint:
        """Create creep data point from row data."""

        measured_values = {}
        uncertainty = {}

        # Time
        timestamp = row.get("time", 0.0)

        # Strain measurements
        if "strain" in row:
            measured_values["strain"] = row["strain"]
            uncertainty["strain"] = row.get(
                "strain_uncertainty", 0.01 * abs(row["strain"])
            )

        if "creep_strain" in row:
            measured_values["creep_strain"] = row["creep_strain"]
            uncertainty["creep_strain"] = row.get(
                "creep_strain_uncertainty", 0.01 * abs(row["creep_strain"])
            )

        # Strain rate
        if "strain_rate" in row:
            measured_values["strain_rate"] = row["strain_rate"]
            uncertainty["strain_rate"] = row.get(
                "strain_rate_uncertainty", 0.1 * abs(row["strain_rate"])
            )

        # Temperature
        if "temperature" in row:
            measured_values["temperature"] = row["temperature"]
            uncertainty["temperature"] = row.get("temperature_uncertainty", 5.0)  # ±5K

        # Stress
        if "stress" in row:
            measured_values["stress"] = row["stress"]
            uncertainty["stress"] = row.get(
                "stress_uncertainty", 0.02 * abs(row["stress"])
            )

        return ExperimentalDataPoint(
            timestamp=timestamp,
            measured_values=measured_values,
            conditions=conditions,
            uncertainty=uncertainty,
            quality_flag=True,
            metadata={"test_type": "creep"},
        )

    def validate_data_quality(self, dataset: ExperimentalDataset) -> Dict[str, float]:
        """Validate high-temperature experiment data quality."""

        quality_metrics = {}

        # Data completeness
        total_points = len(dataset.data_points)
        valid_points = sum(1 for dp in dataset.data_points if dp.quality_flag)
        quality_metrics["completeness"] = (
            valid_points / total_points if total_points > 0 else 0.0
        )

        # Creep curve smoothness
        if total_points > 10:
            times = []
            strains = []

            for dp in dataset.data_points:
                times.append(dp.timestamp)
                if "strain" in dp.measured_values:
                    strains.append(dp.measured_values["strain"])
                elif "creep_strain" in dp.measured_values:
                    strains.append(dp.measured_values["creep_strain"])

            if len(strains) > 5:
                # Check for smooth, monotonic increase
                strain_gradient = np.gradient(strains)
                smoothness = 1.0 - np.std(strain_gradient) / (
                    np.mean(strain_gradient) + 1e-10
                )
                quality_metrics["curve_smoothness"] = max(0.0, smoothness)

        return quality_metrics

    def _load_data_file(self, file_path: str) -> pd.DataFrame:
        """Load high-temperature data from file."""
        path = Path(file_path)

        if path.suffix.lower() == ".csv":
            return pd.read_csv(file_path)
        elif path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        else:
            raise DataIngestionError(f"Unsupported file format: {path.suffix}")


class HugoniotAnalyzer:
    """Analyzer for Hugoniot relationships in shock data."""

    def analyze_hugoniot(self, dataset: ExperimentalDataset) -> Dict[str, float]:
        """Analyze Hugoniot relationship from shock data."""

        pressures = []
        particle_velocities = []
        shock_velocities = []

        for dp in dataset.data_points:
            if (
                "pressure" in dp.measured_values
                and "particle_velocity" in dp.measured_values
            ):
                pressures.append(dp.measured_values["pressure"])
                particle_velocities.append(dp.measured_values["particle_velocity"])

                if "shock_velocity" in dp.measured_values:
                    shock_velocities.append(dp.measured_values["shock_velocity"])

        params = {}

        # Fit linear Us-Up relation: Us = c0 + s * Up
        if len(particle_velocities) > 2 and len(shock_velocities) == len(
            particle_velocities
        ):
            if SCIPY_AVAILABLE:
                try:
                    fit_params = np.polyfit(particle_velocities, shock_velocities, 1)
                    params["hugoniot_slope"] = fit_params[0]  # s
                    params["bulk_sound_speed"] = fit_params[1]  # c0

                    # Calculate R-squared
                    predicted = np.polyval(fit_params, particle_velocities)
                    ss_res = np.sum((shock_velocities - predicted) ** 2)
                    ss_tot = np.sum((shock_velocities - np.mean(shock_velocities)) ** 2)
                    params["hugoniot_r_squared"] = (
                        1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                    )

                except:
                    params["hugoniot_slope"] = 1.0
                    params["bulk_sound_speed"] = 5000.0
                    params["hugoniot_r_squared"] = 0.0

        return params

    def calculate_hugoniot_fit_quality(
        self, pressures: List[float], particle_velocities: List[float]
    ) -> float:
        """Calculate quality of Hugoniot fit."""

        if len(pressures) < 3 or len(particle_velocities) != len(pressures):
            return 0.0

        # Calculate correlation coefficient
        correlation = np.corrcoef(pressures, particle_velocities)[0, 1]
        return abs(correlation)


class RadiationDamageAnalyzer:
    """Analyzer for radiation damage evolution."""

    def analyze_damage_evolution(
        self, dataset: ExperimentalDataset
    ) -> Dict[str, float]:
        """Analyze radiation damage evolution."""

        doses = []
        property_changes = []

        for dp in dataset.data_points:
            if "radiation_dose" in dp.measured_values:
                dose = dp.measured_values["radiation_dose"]
                doses.append(dose)

                # Calculate property degradation
                if "yield_strength" in dp.measured_values:
                    # Assume initial strength of 500 MPa (typical steel)
                    initial_strength = 500e6
                    current_strength = dp.measured_values["yield_strength"]
                    change = (current_strength - initial_strength) / initial_strength
                    property_changes.append(change)

        params = {}

        # Fit radiation damage model
        if len(doses) > 3 and len(property_changes) == len(doses):
            if SCIPY_AVAILABLE:
                try:
                    # Fit exponential damage model
                    def damage_model(dose, saturation, rate):
                        return saturation * (1 - np.exp(-rate * dose))

                    popt, pcov = optimize.curve_fit(
                        damage_model, doses, property_changes, p0=[0.5, 1e-3]
                    )

                    params["damage_saturation"] = popt[0]
                    params["damage_rate"] = popt[1]

                except:
                    params["damage_saturation"] = 0.3
                    params["damage_rate"] = 1e-3

        return params


class CreepAnalyzer:
    """Analyzer for creep behavior."""

    def analyze_creep_behavior(self, dataset: ExperimentalDataset) -> Dict[str, float]:
        """Analyze creep behavior from experimental data."""

        times = []
        strains = []

        for dp in dataset.data_points:
            times.append(dp.timestamp)

            if "creep_strain" in dp.measured_values:
                strains.append(dp.measured_values["creep_strain"])
            elif "strain" in dp.measured_values:
                strains.append(dp.measured_values["strain"])

        params = {}

        # Fit power law creep model: ε = A * t^n
        if len(times) > 5 and len(strains) == len(times):
            if SCIPY_AVAILABLE:
                try:
                    # Log-log fit for power law
                    log_times = np.log(np.array(times) + 1e-6)
                    log_strains = np.log(np.array(strains) + 1e-6)

                    fit_params = np.polyfit(log_times, log_strains, 1)
                    params["creep_exponent"] = fit_params[0]  # n
                    params["creep_coefficient"] = np.exp(fit_params[1])  # A

                    # Calculate minimum creep rate
                    strain_rates = np.gradient(strains, times)
                    params["minimum_creep_rate"] = np.min(
                        strain_rates[strain_rates > 0]
                    )

                except:
                    params["creep_exponent"] = 0.5
                    params["creep_coefficient"] = 1e-6
                    params["minimum_creep_rate"] = 1e-8

        return params


class ExtremeConditionsDataPipeline:
    """Main pipeline for extreme conditions experimental data integration."""

    def __init__(self, backend: str = "cpu"):
        self.backend = backend
        self.logger = get_logger(self.__class__.__name__)

        # Initialize processors
        self.processors = {
            ExperimentType.SHOCK_COMPRESSION: ShockExperimentProcessor(backend),
            ExperimentType.RADIATION_EXPOSURE: RadiationDataProcessor(backend),
            ExperimentType.HIGH_TEMPERATURE_CREEP: HighTemperatureDataProcessor(
                backend
            ),
        }

        # Data storage
        self.datasets = {}

    def ingest_experimental_data(
        self,
        data_source: Union[str, pd.DataFrame, Dict],
        experiment_type: ExperimentType,
        conditions: ExperimentalConditions,
        dataset_id: Optional[str] = None,
    ) -> str:
        """Ingest experimental data into the pipeline."""

        self.logger.info(f"Ingesting {experiment_type.value} experimental data")

        # Get appropriate processor
        if experiment_type not in self.processors:
            raise DataIngestionError(
                f"No processor available for {experiment_type.value}"
            )

        processor = self.processors[experiment_type]

        # Process the data
        dataset = processor.process_raw_data(data_source, conditions)

        # Validate data quality
        quality_metrics = processor.validate_data_quality(dataset)
        dataset.metadata["quality_metrics"] = quality_metrics

        # Store dataset
        if dataset_id is None:
            dataset_id = dataset.experiment_id

        self.datasets[dataset_id] = dataset

        self.logger.info(
            f"Successfully ingested dataset {dataset_id} with "
            f"{len(dataset.data_points)} data points"
        )

        return dataset_id

    def get_dataset(self, dataset_id: str) -> Optional[ExperimentalDataset]:
        """Retrieve dataset by ID."""
        return self.datasets.get(dataset_id)

    def list_datasets(self) -> List[str]:
        """List all available dataset IDs."""
        return list(self.datasets.keys())

    def validate_simulation_against_experiments(
        self,
        simulation_results: SimulationResults,
        experimental_dataset_id: str,
        validation_metrics: List[str] = None,
    ) -> Dict[str, float]:
        """Validate simulation results against experimental data."""

        if experimental_dataset_id not in self.datasets:
            raise ValueError(f"Dataset {experimental_dataset_id} not found")

        dataset = self.datasets[experimental_dataset_id]

        validation_results = {}

        if validation_metrics is None:
            validation_metrics = [
                "pressure",
                "strain",
                "temperature",
                "damage_fraction",
            ]

        for metric in validation_metrics:
            try:
                score = self._calculate_validation_score(
                    simulation_results, dataset, metric
                )
                validation_results[metric] = score

            except Exception as e:
                self.logger.warning(f"Validation failed for {metric}: {e}")
                validation_results[metric] = 0.0

        # Overall validation score
        validation_results["overall"] = np.mean(list(validation_results.values()))

        return validation_results

    def _calculate_validation_score(
        self,
        sim_results: SimulationResults,
        exp_dataset: ExperimentalDataset,
        metric: str,
    ) -> float:
        """Calculate validation score for a specific metric."""

        # Extract experimental values
        exp_values = []
        exp_times = []

        for dp in exp_dataset.data_points:
            if metric in dp.measured_values:
                exp_values.append(dp.measured_values[metric])
                exp_times.append(dp.timestamp)

        if not exp_values:
            return 0.0

        # Extract simulation values
        sim_values = []

        if metric == "pressure" and sim_results.pressures.size > 0:
            sim_values = sim_results.pressures
        elif metric == "strain" and sim_results.strain_tensor.size > 0:
            # Calculate equivalent strain
            for strain_tensor in sim_results.strain_tensor:
                eq_strain = self._calculate_equivalent_strain(strain_tensor)
                sim_values.append(eq_strain)
        elif metric == "temperature" and sim_results.temperatures.size > 0:
            sim_values = sim_results.temperatures
        elif metric == "damage_fraction" and sim_results.damage_fraction.size > 0:
            sim_values = sim_results.damage_fraction

        if not sim_values:
            return 0.0

        # Interpolate to common time base
        if SCIPY_AVAILABLE:
            try:
                # Use experimental time points
                sim_times = np.linspace(0, len(sim_values) - 1, len(sim_values))

                # Interpolate simulation to experimental times
                interp_func = interpolate.interp1d(
                    sim_times, sim_values, kind="linear", fill_value="extrapolate"
                )

                # Normalize experimental times to simulation time scale
                max_exp_time = max(exp_times) if exp_times else 1.0
                normalized_exp_times = [
                    t / max_exp_time * (len(sim_values) - 1) for t in exp_times
                ]

                sim_interp = interp_func(normalized_exp_times)

                # Calculate correlation coefficient
                correlation = np.corrcoef(exp_values, sim_interp)[0, 1]
                return abs(correlation) if not np.isnan(correlation) else 0.0

            except:
                return 0.0
        else:
            # Simple comparison without interpolation
            min_length = min(len(exp_values), len(sim_values))
            if min_length > 1:
                correlation = np.corrcoef(
                    exp_values[:min_length], sim_values[:min_length]
                )[0, 1]
                return abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                return 0.0

    def _calculate_equivalent_strain(self, strain_tensor: np.ndarray) -> float:
        """Calculate equivalent strain from strain tensor."""
        if strain_tensor.size == 0:
            return 0.0

        e11, e22, e33 = strain_tensor[0, 0], strain_tensor[1, 1], strain_tensor[2, 2]
        e12, e13, e23 = strain_tensor[0, 1], strain_tensor[0, 2], strain_tensor[1, 2]

        return np.sqrt(
            (2 / 3)
            * (e11**2 + e22**2 + e33**2 + 2 * (e12**2 + e13**2 + e23**2))
        )

    def export_dataset(
        self, dataset_id: str, output_path: str, format: DataFormat = DataFormat.CSV
    ) -> bool:
        """Export dataset to file."""

        if dataset_id not in self.datasets:
            return False

        dataset = self.datasets[dataset_id]

        try:
            if format == DataFormat.CSV:
                self._export_to_csv(dataset, output_path)
            elif format == DataFormat.JSON:
                self._export_to_json(dataset, output_path)
            elif format == DataFormat.HDF5 and HDF5_AVAILABLE:
                self._export_to_hdf5(dataset, output_path)
            else:
                raise DataIngestionError(f"Unsupported export format: {format}")

            return True

        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return False

    def _export_to_csv(self, dataset: ExperimentalDataset, output_path: str):
        """Export dataset to CSV format."""
        data_rows = []

        for dp in dataset.data_points:
            row = {"timestamp": dp.timestamp}
            row.update(dp.measured_values)
            row.update({f"{k}_uncertainty": v for k, v in dp.uncertainty.items()})
            row["quality_flag"] = dp.quality_flag
            data_rows.append(row)

        df = pd.DataFrame(data_rows)
        df.to_csv(output_path, index=False)

    def _export_to_json(self, dataset: ExperimentalDataset, output_path: str):
        """Export dataset to JSON format."""
        export_data = {
            "experiment_type": dataset.experiment_type.value,
            "experiment_id": dataset.experiment_id,
            "material_composition": dataset.material_composition,
            "processing_history": dataset.processing_history,
            "measurement_techniques": dataset.measurement_techniques,
            "metadata": dataset.metadata,
            "data_points": [],
        }

        for dp in dataset.data_points:
            dp_data = {
                "timestamp": dp.timestamp,
                "measured_values": dp.measured_values,
                "uncertainty": dp.uncertainty,
                "quality_flag": dp.quality_flag,
                "metadata": dp.metadata,
            }
            export_data["data_points"].append(dp_data)

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

    def _export_to_hdf5(self, dataset: ExperimentalDataset, output_path: str):
        """Export dataset to HDF5 format."""
        if not HDF5_AVAILABLE:
            raise DataIngestionError("HDF5 not available")

        with h5py.File(output_path, "w") as f:
            # Dataset metadata
            f.attrs["experiment_type"] = dataset.experiment_type.value
            f.attrs["experiment_id"] = dataset.experiment_id

            # Data points
            n_points = len(dataset.data_points)

            # Create datasets for each measurement type
            all_measurements = set()
            for dp in dataset.data_points:
                all_measurements.update(dp.measured_values.keys())

            timestamps = np.array([dp.timestamp for dp in dataset.data_points])
            f.create_dataset("timestamps", data=timestamps)

            for measurement in all_measurements:
                values = []
                uncertainties = []

                for dp in dataset.data_points:
                    values.append(dp.measured_values.get(measurement, np.nan))
                    uncertainties.append(dp.uncertainty.get(measurement, np.nan))

                f.create_dataset(f"measurements/{measurement}", data=np.array(values))
                f.create_dataset(
                    f"uncertainties/{measurement}", data=np.array(uncertainties)
                )
