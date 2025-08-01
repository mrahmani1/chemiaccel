"""
Real-time In-Operando Crystallization Control

This module implements real-time crystallization process control using in-operando
characterization data for continuous model calibration and adaptive process control.
Integrates with various analytical techniques for real-time monitoring and control.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from scipy.interpolate import interp1d
    from scipy.optimize import minimize_scalar
    from scipy.signal import find_peaks, savgol_filter

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .core import (
    CrystallizationCore,
    KineticsParameters,
    ProcessConditions,
    RealTimeData,
    SimulationConfig,
)
from .kinetics_modeling import CrystallizationKineticsModel

logger = logging.getLogger(__name__)


@dataclass
class ControllerSettings:
    """
    Settings for crystallization controller.

    Attributes:
        control_frequency: Control loop frequency (Hz)
        data_buffer_size: Size of data buffer for analysis
        prediction_horizon: Prediction horizon for MPC (seconds)
        safety_limits: Safety limits for process variables
        alarm_thresholds: Threshold values for alarms
        control_gains: PID/MPC controller gains
    """

    control_frequency: float = 1.0  # Hz
    data_buffer_size: int = 1000
    prediction_horizon: float = 300.0  # seconds
    safety_limits: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "temperature": (273.15, 373.15),
            "stirring_rate": (0.0, 1000.0),
            "ph": (1.0, 14.0),
            "supersaturation": (0.5, 10.0),
        }
    )
    alarm_thresholds: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "temperature": {"high": 360.0, "low": 280.0},
            "turbidity": {"high": 1000.0, "low": 0.1},
            "supersaturation": {"high": 8.0, "low": 0.8},
        }
    )
    control_gains: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "temperature": {"kp": 1.0, "ki": 0.1, "kd": 0.05},
            "stirring_rate": {"kp": 0.5, "ki": 0.05, "kd": 0.02},
        }
    )


class AnalyticalTechnique(ABC):
    """Abstract base class for analytical techniques."""

    @abstractmethod
    def process_data(self, raw_data: np.ndarray) -> Dict[str, Any]:
        """Process raw analytical data."""
        pass

    @abstractmethod
    def extract_features(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from processed data."""
        pass


class TurbidityAnalyzer(AnalyticalTechnique):
    """
    Turbidity analyzer for nucleation detection and crystal growth monitoring.
    """

    def __init__(self, baseline_window: int = 50):
        """
        Initialize turbidity analyzer.

        Args:
            baseline_window: Window size for baseline correction
        """
        self.baseline_window = baseline_window
        self.baseline_history = deque(maxlen=baseline_window)

    def process_data(self, raw_data: np.ndarray) -> Dict[str, Any]:
        """
        Process turbidity data.

        Args:
            raw_data: Raw turbidity measurements

        Returns:
            Processed turbidity data
        """
        if len(raw_data) == 0:
            return {"turbidity": 0.0, "baseline": 0.0, "corrected": 0.0}

        # Current turbidity
        current_turbidity = raw_data[-1] if len(raw_data) > 0 else 0.0

        # Update baseline
        if len(self.baseline_history) < 10:  # Not enough history
            baseline = current_turbidity
        else:
            baseline = np.mean(list(self.baseline_history)[:10])  # Use first 10 points

        self.baseline_history.append(current_turbidity)

        # Baseline-corrected turbidity
        corrected_turbidity = max(0.0, current_turbidity - baseline)

        # Smooth data if enough points
        if len(raw_data) >= 5:
            try:
                smoothed = savgol_filter(raw_data, min(5, len(raw_data)), 2)
                smoothed_current = smoothed[-1]
            except:
                smoothed_current = current_turbidity
        else:
            smoothed_current = current_turbidity

        return {
            "turbidity": current_turbidity,
            "baseline": baseline,
            "corrected": corrected_turbidity,
            "smoothed": smoothed_current,
        }

    def extract_features(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from turbidity data."""
        features = {
            "turbidity_level": processed_data.get("corrected", 0.0),
            "turbidity_rate": 0.0,  # Would need time series for rate
            "nucleation_detected": 1.0
            if processed_data.get("corrected", 0.0) > 5.0
            else 0.0,
        }

        return features


class RamanAnalyzer(AnalyticalTechnique):
    """
    Raman spectroscopy analyzer for solution concentration and polymorph identification.
    """

    def __init__(self, reference_spectra: Optional[Dict[str, np.ndarray]] = None):
        """
        Initialize Raman analyzer.

        Args:
            reference_spectra: Reference spectra for different polymorphs
        """
        self.reference_spectra = reference_spectra or {}
        self.baseline_correction = True

    def process_data(self, raw_data: np.ndarray) -> Dict[str, Any]:
        """
        Process Raman spectroscopy data.

        Args:
            raw_data: Raw Raman spectrum

        Returns:
            Processed spectral data
        """
        if len(raw_data) == 0:
            return {"spectrum": np.array([]), "peaks": [], "concentration": 0.0}

        spectrum = raw_data.copy()

        # Baseline correction
        if self.baseline_correction and len(spectrum) > 10:
            # Simple baseline correction using minimum values
            window_size = max(10, len(spectrum) // 20)
            baseline = np.convolve(
                spectrum, np.ones(window_size) / window_size, mode="same"
            )
            spectrum = spectrum - baseline + np.min(baseline)

        # Peak detection
        peaks = self._find_peaks(spectrum)

        # Estimate concentration from peak intensities
        concentration = self._estimate_concentration(spectrum, peaks)

        return {
            "spectrum": spectrum,
            "peaks": peaks,
            "concentration": concentration,
            "peak_intensities": [spectrum[p] for p in peaks] if peaks else [],
        }

    def extract_features(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from Raman data."""
        spectrum = processed_data.get("spectrum", np.array([]))
        peaks = processed_data.get("peaks", [])

        features = {
            "concentration": processed_data.get("concentration", 0.0),
            "n_peaks": len(peaks),
            "max_intensity": np.max(spectrum) if len(spectrum) > 0 else 0.0,
            "spectral_width": np.std(spectrum) if len(spectrum) > 0 else 0.0,
        }

        # Polymorph identification
        if self.reference_spectra and len(spectrum) > 0:
            polymorph_scores = self._identify_polymorph(spectrum)
            features.update(polymorph_scores)

        return features

    def _find_peaks(self, spectrum: np.ndarray) -> List[int]:
        """Find peaks in spectrum."""
        if len(spectrum) < 5:
            return []

        try:
            peaks, _ = find_peaks(spectrum, height=np.max(spectrum) * 0.1, distance=5)
            return peaks.tolist()
        except:
            return []

    def _estimate_concentration(self, spectrum: np.ndarray, peaks: List[int]) -> float:
        """Estimate concentration from spectral data."""
        if not peaks or len(spectrum) == 0:
            return 0.0

        # Use peak intensities to estimate concentration
        peak_intensities = [spectrum[p] for p in peaks]
        avg_intensity = np.mean(peak_intensities)

        # Simple linear relationship (would be calibrated in practice)
        concentration = avg_intensity / 1000.0  # Arbitrary scaling

        return max(0.0, concentration)

    def _identify_polymorph(self, spectrum: np.ndarray) -> Dict[str, float]:
        """Identify polymorph from spectral matching."""
        polymorph_scores = {}

        for polymorph_name, ref_spectrum in self.reference_spectra.items():
            if len(ref_spectrum) == len(spectrum):
                # Calculate correlation
                correlation = np.corrcoef(spectrum, ref_spectrum)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
                polymorph_scores[f"polymorph_{polymorph_name}_score"] = max(
                    0.0, correlation
                )

        return polymorph_scores


class PVMAnalyzer(AnalyticalTechnique):
    """
    Particle Vision and Measurement (PVM) analyzer for particle size and morphology.
    """

    def __init__(self, pixel_size: float = 1e-6):
        """
        Initialize PVM analyzer.

        Args:
            pixel_size: Size of each pixel in meters
        """
        self.pixel_size = pixel_size

    def process_data(self, raw_data: np.ndarray) -> Dict[str, Any]:
        """
        Process PVM image data.

        Args:
            raw_data: Raw image data (assumed to be particle contours/sizes)

        Returns:
            Processed particle data
        """
        if len(raw_data) == 0:
            return {"particle_count": 0, "mean_size": 0.0, "size_distribution": []}

        # Assume raw_data contains particle sizes in pixels
        particle_sizes_pixels = raw_data[raw_data > 0]  # Remove background
        particle_sizes_meters = particle_sizes_pixels * self.pixel_size

        if len(particle_sizes_meters) == 0:
            return {"particle_count": 0, "mean_size": 0.0, "size_distribution": []}

        # Calculate statistics
        particle_count = len(particle_sizes_meters)
        mean_size = np.mean(particle_sizes_meters)

        # Size distribution percentiles
        size_distribution = {
            "d10": np.percentile(particle_sizes_meters, 10),
            "d50": np.percentile(particle_sizes_meters, 50),
            "d90": np.percentile(particle_sizes_meters, 90),
        }

        return {
            "particle_count": particle_count,
            "mean_size": mean_size,
            "size_distribution": size_distribution,
            "particle_sizes": particle_sizes_meters.tolist(),
        }

    def extract_features(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from PVM data."""
        size_dist = processed_data.get("size_distribution", {})

        features = {
            "particle_count": processed_data.get("particle_count", 0),
            "mean_particle_size": processed_data.get("mean_size", 0.0),
            "d10": size_dist.get("d10", 0.0),
            "d50": size_dist.get("d50", 0.0),
            "d90": size_dist.get("d90", 0.0),
        }

        # Calculate span if possible
        if size_dist.get("d50", 0) > 0:
            features["span"] = (
                size_dist.get("d90", 0) - size_dist.get("d10", 0)
            ) / size_dist["d50"]
        else:
            features["span"] = 0.0

        return features


class InOperandoCharacterization:
    """
    In-operando characterization system integrating multiple analytical techniques.
    """

    def __init__(
        self,
        techniques: Optional[Dict[str, AnalyticalTechnique]] = None,
        data_fusion_enabled: bool = True,
    ):
        """
        Initialize in-operando characterization.

        Args:
            techniques: Dictionary of analytical techniques
            data_fusion_enabled: Whether to enable data fusion
        """
        self.techniques = techniques or {
            "turbidity": TurbidityAnalyzer(),
            "raman": RamanAnalyzer(),
            "pvm": PVMAnalyzer(),
        }
        self.data_fusion_enabled = data_fusion_enabled
        self.data_history = {
            technique: deque(maxlen=1000) for technique in self.techniques.keys()
        }

    def process_measurements(
        self, raw_measurements: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Process measurements from all techniques.

        Args:
            raw_measurements: Raw data from each technique

        Returns:
            Processed and fused measurement data
        """
        processed_data = {}
        all_features = {}

        # Process each technique
        for technique_name, technique in self.techniques.items():
            if technique_name in raw_measurements:
                raw_data = raw_measurements[technique_name]

                # Process data
                processed = technique.process_data(raw_data)
                processed_data[technique_name] = processed

                # Extract features
                features = technique.extract_features(processed)
                all_features.update(
                    {f"{technique_name}_{k}": v for k, v in features.items()}
                )

                # Store in history
                self.data_history[technique_name].append(processed)

        # Data fusion
        if self.data_fusion_enabled:
            fused_data = self._fuse_data(all_features)
            all_features.update(fused_data)

        return {
            "processed_data": processed_data,
            "features": all_features,
            "timestamp": time.time(),
        }

    def _fuse_data(self, features: Dict[str, float]) -> Dict[str, float]:
        """Fuse data from multiple techniques."""
        fused_data = {}

        # Concentration estimation from multiple sources
        concentration_sources = []
        if "raman_concentration" in features:
            concentration_sources.append(features["raman_concentration"])
        if "turbidity_turbidity_level" in features:
            # Convert turbidity to rough concentration estimate
            turbidity_conc = features["turbidity_turbidity_level"] / 100.0
            concentration_sources.append(turbidity_conc)

        if concentration_sources:
            fused_data["fused_concentration"] = np.mean(concentration_sources)

        # Particle size from PVM
        if "pvm_mean_particle_size" in features:
            fused_data["fused_particle_size"] = features["pvm_mean_particle_size"]

        # Nucleation detection
        nucleation_indicators = []
        if "turbidity_nucleation_detected" in features:
            nucleation_indicators.append(features["turbidity_nucleation_detected"])
        if "pvm_particle_count" in features and features["pvm_particle_count"] > 100:
            nucleation_indicators.append(1.0)

        if nucleation_indicators:
            fused_data["fused_nucleation_active"] = max(nucleation_indicators)
        else:
            fused_data["fused_nucleation_active"] = 0.0

        return fused_data


class AdaptiveProcessControl:
    """
    Adaptive process control using model predictive control (MPC) and feedback control.
    """

    def __init__(
        self,
        kinetics_model: CrystallizationKineticsModel,
        controller_settings: ControllerSettings,
    ):
        """
        Initialize adaptive process controller.

        Args:
            kinetics_model: Crystallization kinetics model
            controller_settings: Controller configuration
        """
        self.kinetics_model = kinetics_model
        self.settings = controller_settings
        self.setpoints = {}
        self.control_outputs = {}
        self.pid_states = {}  # PID controller states
        self.model_adaptation_enabled = True

        # Initialize PID states
        for variable in self.settings.control_gains.keys():
            self.pid_states[variable] = {
                "previous_error": 0.0,
                "integral": 0.0,
                "last_time": time.time(),
            }

    def update_setpoints(self, new_setpoints: Dict[str, float]) -> None:
        """Update control setpoints."""
        self.setpoints.update(new_setpoints)
        logger.info(f"Updated setpoints: {new_setpoints}")

    def calculate_control_actions(
        self, current_state: RealTimeData, features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate control actions based on current state and process model.

        Args:
            current_state: Current process state
            features: Extracted features from measurements

        Returns:
            Control actions for process variables
        """
        control_actions = {}
        current_time = time.time()

        # Temperature control
        if "temperature" in self.setpoints:
            temp_action = self._calculate_pid_action(
                "temperature",
                self.setpoints["temperature"],
                current_state.temperature,
                current_time,
            )
            control_actions["temperature"] = temp_action

        # Stirring rate control (based on particle size feedback)
        if "stirring_rate" in self.setpoints and "fused_particle_size" in features:
            target_size = self.setpoints.get("target_particle_size", 50e-6)
            current_size = features["fused_particle_size"]

            # Adjust stirring based on particle size deviation
            size_error = target_size - current_size
            stirring_adjustment = size_error * 1000  # Scaling factor

            new_stirring = (current_state.stirring_rate or 200.0) + stirring_adjustment
            new_stirring = np.clip(
                new_stirring,
                self.settings.safety_limits["stirring_rate"][0],
                self.settings.safety_limits["stirring_rate"][1],
            )

            control_actions["stirring_rate"] = new_stirring

        # Model predictive control for supersaturation
        if self.model_adaptation_enabled and "fused_concentration" in features:
            mpc_actions = self._calculate_mpc_actions(current_state, features)
            control_actions.update(mpc_actions)

        # Apply safety limits
        control_actions = self._apply_safety_limits(control_actions)

        self.control_outputs = control_actions
        return control_actions

    def _calculate_pid_action(
        self, variable: str, setpoint: float, current_value: float, current_time: float
    ) -> float:
        """Calculate PID control action."""
        if variable not in self.settings.control_gains:
            return current_value

        gains = self.settings.control_gains[variable]
        state = self.pid_states[variable]

        # Calculate error
        error = setpoint - current_value

        # Time difference
        dt = current_time - state["last_time"]
        if dt <= 0:
            dt = 1.0 / self.settings.control_frequency

        # Proportional term
        p_term = gains["kp"] * error

        # Integral term
        state["integral"] += error * dt
        i_term = gains["ki"] * state["integral"]

        # Derivative term
        d_error = (error - state["previous_error"]) / dt
        d_term = gains["kd"] * d_error

        # PID output
        output = current_value + p_term + i_term + d_term

        # Update state
        state["previous_error"] = error
        state["last_time"] = current_time

        return output

    def _calculate_mpc_actions(
        self, current_state: RealTimeData, features: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate model predictive control actions."""
        actions = {}

        # Simplified MPC for demonstration
        # In practice, this would involve optimization over prediction horizon

        current_conc = features.get("fused_concentration", 0.0)
        target_conc = self.setpoints.get("target_concentration", 0.1)

        if current_conc > 0:
            # Calculate supersaturation
            solubility = 0.05  # Assumed equilibrium solubility
            supersaturation = current_conc / solubility

            # If supersaturation is too high, recommend cooling
            if supersaturation > 3.0:
                temp_reduction = min(5.0, (supersaturation - 3.0) * 2.0)
                current_temp = current_state.temperature
                new_temp = current_temp - temp_reduction

                # Apply temperature limits
                temp_limits = self.settings.safety_limits["temperature"]
                new_temp = np.clip(new_temp, temp_limits[0], temp_limits[1])

                actions["temperature_mpc"] = new_temp

        return actions

    def _apply_safety_limits(self, actions: Dict[str, float]) -> Dict[str, float]:
        """Apply safety limits to control actions."""
        safe_actions = {}

        for variable, value in actions.items():
            # Remove MPC suffix for limit checking
            base_variable = variable.replace("_mpc", "")

            if base_variable in self.settings.safety_limits:
                limits = self.settings.safety_limits[base_variable]
                safe_value = np.clip(value, limits[0], limits[1])
                safe_actions[variable] = safe_value
            else:
                safe_actions[variable] = value

        return safe_actions

    def adapt_model(
        self, measurement_data: Dict[str, Any], control_history: List[Dict[str, float]]
    ) -> None:
        """Adapt the process model based on recent data."""
        if not self.model_adaptation_enabled:
            return

        # Simple parameter adaptation based on model-measurement mismatch
        # In practice, this would use more sophisticated techniques

        features = measurement_data.get("features", {})

        # Adapt kinetics parameters if we have particle size data
        if "fused_particle_size" in features and len(control_history) > 10:
            measured_size = features["fused_particle_size"]

            # Compare with model prediction (simplified)
            # In practice, would run model with recent conditions
            predicted_size = 30e-6  # Placeholder

            size_error = measured_size - predicted_size

            # Adjust growth rate constant
            if abs(size_error) > 5e-6:  # Significant error
                adjustment_factor = 1.0 + size_error / predicted_size * 0.1
                current_growth_rate = (
                    self.kinetics_model.kinetics_params.growth_rate_constant
                )
                new_growth_rate = current_growth_rate * adjustment_factor

                # Update model parameter
                self.kinetics_model.kinetics_params.growth_rate_constant = max(
                    1e-10, min(1e-6, new_growth_rate)
                )

                logger.info(f"Adapted growth rate constant to {new_growth_rate:.2e}")


class CrystallizationMonitor:
    """
    Crystallization process monitor with alarm management and data logging.
    """

    def __init__(self, controller_settings: ControllerSettings):
        """
        Initialize crystallization monitor.

        Args:
            controller_settings: Controller settings with alarm thresholds
        """
        self.settings = controller_settings
        self.active_alarms = set()
        self.alarm_history = []
        self.data_log = []

    def check_alarms(
        self, current_state: RealTimeData, features: Dict[str, float]
    ) -> List[str]:
        """
        Check for alarm conditions.

        Args:
            current_state: Current process state
            features: Extracted features

        Returns:
            List of active alarms
        """
        new_alarms = []

        # Temperature alarms
        if current_state.temperature:
            temp_thresholds = self.settings.alarm_thresholds.get("temperature", {})
            if current_state.temperature > temp_thresholds.get("high", 400.0):
                new_alarms.append("HIGH_TEMPERATURE")
            elif current_state.temperature < temp_thresholds.get("low", 250.0):
                new_alarms.append("LOW_TEMPERATURE")

        # Turbidity alarms
        turbidity_level = features.get("turbidity_turbidity_level", 0.0)
        turbidity_thresholds = self.settings.alarm_thresholds.get("turbidity", {})
        if turbidity_level > turbidity_thresholds.get("high", 1000.0):
            new_alarms.append("HIGH_TURBIDITY")

        # Supersaturation alarms
        concentration = features.get("fused_concentration", 0.0)
        if concentration > 0:
            supersaturation = concentration / 0.05  # Assumed solubility
            supersat_thresholds = self.settings.alarm_thresholds.get(
                "supersaturation", {}
            )
            if supersaturation > supersat_thresholds.get("high", 8.0):
                new_alarms.append("HIGH_SUPERSATURATION")
            elif supersaturation < supersat_thresholds.get("low", 0.8):
                new_alarms.append("LOW_SUPERSATURATION")

        # Update active alarms
        for alarm in new_alarms:
            if alarm not in self.active_alarms:
                self.active_alarms.add(alarm)
                self.alarm_history.append(
                    {"alarm": alarm, "timestamp": time.time(), "status": "ACTIVE"}
                )
                logger.warning(f"Alarm activated: {alarm}")

        # Check for cleared alarms
        cleared_alarms = []
        for alarm in list(self.active_alarms):
            if alarm not in new_alarms:
                cleared_alarms.append(alarm)
                self.active_alarms.remove(alarm)
                self.alarm_history.append(
                    {"alarm": alarm, "timestamp": time.time(), "status": "CLEARED"}
                )
                logger.info(f"Alarm cleared: {alarm}")

        return list(self.active_alarms)

    def log_data(
        self,
        current_state: RealTimeData,
        features: Dict[str, float],
        control_actions: Dict[str, float],
    ) -> None:
        """Log process data."""
        log_entry = {
            "timestamp": time.time(),
            "state": current_state.to_dict(),
            "features": features,
            "control_actions": control_actions,
            "active_alarms": list(self.active_alarms),
        }

        self.data_log.append(log_entry)

        # Limit log size
        if len(self.data_log) > 10000:
            self.data_log = self.data_log[-5000:]  # Keep recent half

    def get_process_summary(self) -> Dict[str, Any]:
        """Get current process summary."""
        if not self.data_log:
            return {"status": "No data available"}

        recent_data = self.data_log[-10:] if len(self.data_log) >= 10 else self.data_log

        # Calculate trends
        temperatures = [entry["state"].get("temperature", 0) for entry in recent_data]
        temp_trend = "stable"
        if len(temperatures) >= 3:
            if temperatures[-1] > temperatures[0] + 2.0:
                temp_trend = "increasing"
            elif temperatures[-1] < temperatures[0] - 2.0:
                temp_trend = "decreasing"

        summary = {
            "current_alarms": list(self.active_alarms),
            "alarm_count_24h": len(
                [a for a in self.alarm_history if time.time() - a["timestamp"] < 86400]
            ),
            "temperature_trend": temp_trend,
            "data_points_logged": len(self.data_log),
            "last_update": self.data_log[-1]["timestamp"] if self.data_log else None,
        }

        return summary


class RealTimeController(CrystallizationCore):
    """
    Main real-time crystallization controller integrating all components.
    """

    def __init__(
        self,
        kinetics_model: CrystallizationKineticsModel,
        controller_settings: Optional[ControllerSettings] = None,
        config: Optional[SimulationConfig] = None,
    ):
        """
        Initialize real-time controller.

        Args:
            kinetics_model: Crystallization kinetics model
            controller_settings: Controller settings
            config: Simulation configuration
        """
        super().__init__(config)
        self.kinetics_model = kinetics_model
        self.settings = controller_settings or ControllerSettings()

        self.characterization_system = None
        self.process_controller = None
        self.monitor = None

        # Control loop management
        self.control_active = False
        self.control_thread = None
        self.data_queue = Queue()

    def initialize(self) -> None:
        """Initialize the real-time controller."""
        # Initialize characterization system
        self.characterization_system = InOperandoCharacterization()

        # Initialize process controller
        self.process_controller = AdaptiveProcessControl(
            self.kinetics_model, self.settings
        )

        # Initialize monitor
        self.monitor = CrystallizationMonitor(self.settings)

        self._initialized = True
        logger.info("Real-time controller initialized")

    def validate_inputs(self, **kwargs) -> bool:
        """Validate input parameters."""
        return True

    def start_control_loop(self, setpoints: Dict[str, float]) -> None:
        """
        Start the real-time control loop.

        Args:
            setpoints: Initial setpoints for controlled variables
        """
        if not self.is_initialized():
            self.initialize()

        if self.control_active:
            logger.warning("Control loop already active")
            return

        # Set initial setpoints
        self.process_controller.update_setpoints(setpoints)

        # Start control thread
        self.control_active = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()

        logger.info("Real-time control loop started")

    def stop_control_loop(self) -> None:
        """Stop the real-time control loop."""
        self.control_active = False

        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=5.0)

        logger.info("Real-time control loop stopped")

    def add_measurement_data(self, measurement_data: Dict[str, np.ndarray]) -> None:
        """
        Add new measurement data to the processing queue.

        Args:
            measurement_data: Raw measurement data from analytical techniques
        """
        self.data_queue.put(measurement_data)

    def update_setpoints(self, new_setpoints: Dict[str, float]) -> None:
        """Update control setpoints during operation."""
        if self.process_controller:
            self.process_controller.update_setpoints(new_setpoints)

    def get_current_status(self) -> Dict[str, Any]:
        """Get current controller status."""
        status = {
            "control_active": self.control_active,
            "initialized": self.is_initialized(),
            "queue_size": self.data_queue.qsize(),
        }

        if self.monitor:
            status["process_summary"] = self.monitor.get_process_summary()

        if self.process_controller:
            status["last_control_outputs"] = self.process_controller.control_outputs

        return status

    def _control_loop(self) -> None:
        """Main control loop running in separate thread."""
        control_period = 1.0 / self.settings.control_frequency
        control_history = []

        while self.control_active:
            loop_start_time = time.time()

            try:
                # Get measurement data
                measurement_data = None
                try:
                    measurement_data = self.data_queue.get(timeout=control_period / 2)
                except Empty:
                    pass  # No new data available

                if measurement_data:
                    # Process measurements
                    processed_data = self.characterization_system.process_measurements(
                        measurement_data
                    )

                    # Extract current state
                    current_state = self._extract_current_state(processed_data)
                    features = processed_data.get("features", {})

                    # Check alarms
                    alarms = self.monitor.check_alarms(current_state, features)

                    # Calculate control actions
                    control_actions = self.process_controller.calculate_control_actions(
                        current_state, features
                    )

                    # Log data
                    self.monitor.log_data(current_state, features, control_actions)

                    # Store control history
                    control_history.append(control_actions)
                    if len(control_history) > 100:
                        control_history = control_history[-50:]  # Keep recent history

                    # Model adaptation
                    if len(control_history) >= 10:
                        self.process_controller.adapt_model(
                            processed_data, control_history
                        )

                    # Store results
                    self._simulation_data = {
                        "current_state": current_state,
                        "features": features,
                        "control_actions": control_actions,
                        "alarms": alarms,
                        "timestamp": time.time(),
                    }

            except Exception as e:
                logger.error(f"Error in control loop: {e}")

            # Sleep to maintain control frequency
            elapsed_time = time.time() - loop_start_time
            sleep_time = max(0, control_period - elapsed_time)
            time.sleep(sleep_time)

    def _extract_current_state(self, processed_data: Dict[str, Any]) -> RealTimeData:
        """Extract current state from processed data."""
        features = processed_data.get("features", {})

        # Create RealTimeData object
        current_state = RealTimeData(
            timestamp=time.time(),
            temperature=298.15,  # Default, would come from actual sensor
            turbidity=features.get("turbidity_turbidity_level", 0.0),
            supersaturation=features.get("fused_concentration", 0.0)
            / 0.05,  # Assumed solubility
            particle_count={"total": int(features.get("pvm_particle_count", 0))},
        )

        return current_state
