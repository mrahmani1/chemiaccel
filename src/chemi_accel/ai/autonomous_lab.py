"""
Autonomous Lab Protocol Generation and Execution.

This module provides capabilities for translating optimized experimental conditions
into machine-executable protocols and establishing communication with automated
laboratory equipment for direct integration and control.
"""

import asyncio
import json
import logging
import uuid
import warnings
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Optional dependencies with graceful fallbacks
try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    warnings.warn("Requests not available. HTTP communication will be disabled.")

try:
    import serial

    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    warnings.warn("PySerial not available. Serial communication will be disabled.")

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("Pandas not available. Some data handling features will be disabled.")

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    warnings.warn("PyYAML not available. YAML protocol support will be disabled.")

# Set up logging
logger = logging.getLogger(__name__)


class EquipmentType(Enum):
    """Types of laboratory equipment."""

    LIQUID_HANDLER = "liquid_handler"
    REACTOR = "reactor"
    SPECTROMETER = "spectrometer"
    CHROMATOGRAPH = "chromatograph"
    RHEOMETER = "rheometer"
    BALANCE = "balance"
    PUMP = "pump"
    HEATER = "heater"
    STIRRER = "stirrer"
    CENTRIFUGE = "centrifuge"
    PIPETTE = "pipette"
    DISPENSER = "dispenser"


class ProtocolFormat(Enum):
    """Supported protocol formats."""

    HAMILTON_VENUS = "hamilton_venus"
    TECAN_EVA = "tecan_eva"
    BECKMAN_BIOMEK = "beckman_biomek"
    OPENTRONS = "opentrons"
    JSON_PROTOCOL = "json_protocol"
    XML_PROTOCOL = "xml_protocol"
    YAML_PROTOCOL = "yaml_protocol"
    PYTHON_SCRIPT = "python_script"


class CommunicationProtocol(Enum):
    """Communication protocols for equipment."""

    HTTP_REST = "http_rest"
    SERIAL = "serial"
    TCP_IP = "tcp_ip"
    MODBUS = "modbus"
    OPC_UA = "opc_ua"
    SCPI = "scpi"
    MQTT = "mqtt"


class ActionType(Enum):
    """Types of experimental actions."""

    TRANSFER_LIQUID = "transfer_liquid"
    MIX = "mix"
    HEAT = "heat"
    COOL = "cool"
    WAIT = "wait"
    MEASURE = "measure"
    DISPENSE = "dispense"
    ASPIRATE = "aspirate"
    WASH = "wash"
    INCUBATE = "incubate"
    CENTRIFUGE = "centrifuge"
    FILTER = "filter"


class AutonomousLabError(Exception):
    """Base exception for autonomous lab errors."""

    pass


class EquipmentCommunicationError(AutonomousLabError):
    """Exception for equipment communication errors."""

    pass


class ProtocolGenerationError(AutonomousLabError):
    """Exception for protocol generation errors."""

    pass


@dataclass
class Reagent:
    """Represents a chemical reagent."""

    name: str
    cas_number: Optional[str] = None
    molecular_weight: Optional[float] = None
    density: Optional[float] = None
    concentration: Optional[float] = None
    purity: Optional[float] = None
    supplier: Optional[str] = None
    lot_number: Optional[str] = None
    expiry_date: Optional[datetime] = None
    storage_conditions: Optional[str] = None
    safety_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Vessel:
    """Represents a laboratory vessel."""

    id: str
    vessel_type: str
    volume_capacity: float
    current_volume: float = 0.0
    position: Optional[Tuple[int, int]] = None
    barcode: Optional[str] = None
    contents: List[Reagent] = field(default_factory=list)
    temperature: Optional[float] = None
    is_sealed: bool = False


@dataclass
class LabwareDefinition:
    """Defines laboratory equipment and labware."""

    name: str
    equipment_type: EquipmentType
    model: str
    manufacturer: str
    serial_number: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    communication_config: Dict[str, Any] = field(default_factory=dict)
    calibration_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentalAction:
    """Represents a single experimental action."""

    action_id: str
    action_type: ActionType
    equipment: str
    parameters: Dict[str, Any]
    source_vessels: List[str] = field(default_factory=list)
    destination_vessels: List[str] = field(default_factory=list)
    reagents: List[Reagent] = field(default_factory=list)
    duration: Optional[float] = None
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    error_handling: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentalProtocol:
    """Represents a complete experimental protocol."""

    protocol_id: str
    name: str
    description: str
    actions: List[ExperimentalAction]
    labware: List[LabwareDefinition]
    reagents: List[Reagent]
    vessels: List[Vessel]
    safety_requirements: List[str] = field(default_factory=list)
    estimated_duration: Optional[float] = None
    created_date: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Results from protocol execution."""

    protocol_id: str
    action_id: str
    success: bool
    start_time: datetime
    end_time: datetime
    actual_parameters: Dict[str, Any]
    measurements: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    quality_metrics: Dict[str, float] = field(default_factory=dict)


class EquipmentDriver(ABC):
    """Abstract base class for equipment drivers."""

    def __init__(self, labware_def: LabwareDefinition):
        self.labware_def = labware_def
        self.connected = False
        self.status = "idle"

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to equipment."""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from equipment."""
        pass

    @abstractmethod
    async def execute_action(self, action: ExperimentalAction) -> ExecutionResult:
        """Execute an experimental action."""
        pass

    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get equipment status."""
        pass

    @abstractmethod
    async def calibrate(self) -> bool:
        """Calibrate equipment."""
        pass


class LiquidHandlerDriver(EquipmentDriver):
    """Driver for robotic liquid handlers."""

    def __init__(self, labware_def: LabwareDefinition):
        super().__init__(labware_def)
        self.tip_racks = []
        self.plate_positions = {}

    async def connect(self) -> bool:
        """Connect to liquid handler."""
        try:
            # Simulate connection
            await asyncio.sleep(0.1)
            self.connected = True
            logger.info(f"Connected to liquid handler: {self.labware_def.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to liquid handler: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from liquid handler."""
        self.connected = False
        logger.info(f"Disconnected from liquid handler: {self.labware_def.name}")
        return True

    async def execute_action(self, action: ExperimentalAction) -> ExecutionResult:
        """Execute liquid handling action."""
        start_time = datetime.now()

        try:
            if action.action_type == ActionType.TRANSFER_LIQUID:
                result = await self._transfer_liquid(action)
            elif action.action_type == ActionType.MIX:
                result = await self._mix(action)
            elif action.action_type == ActionType.ASPIRATE:
                result = await self._aspirate(action)
            elif action.action_type == ActionType.DISPENSE:
                result = await self._dispense(action)
            elif action.action_type == ActionType.WASH:
                result = await self._wash_tips(action)
            else:
                raise AutonomousLabError(
                    f"Unsupported action type: {action.action_type}"
                )

            end_time = datetime.now()

            return ExecutionResult(
                protocol_id=action.action_id,
                action_id=action.action_id,
                success=True,
                start_time=start_time,
                end_time=end_time,
                actual_parameters=action.parameters,
                measurements=result,
            )

        except Exception as e:
            end_time = datetime.now()
            return ExecutionResult(
                protocol_id=action.action_id,
                action_id=action.action_id,
                success=False,
                start_time=start_time,
                end_time=end_time,
                actual_parameters=action.parameters,
                error_message=str(e),
            )

    async def _transfer_liquid(self, action: ExperimentalAction) -> Dict[str, Any]:
        """Perform liquid transfer."""
        volume = action.parameters.get("volume", 0.0)
        source = action.parameters.get("source")
        destination = action.parameters.get("destination")

        # Simulate transfer
        await asyncio.sleep(volume * 0.01)  # Time proportional to volume

        logger.info(f"Transferred {volume} µL from {source} to {destination}")

        return {
            "volume_transferred": volume,
            "source_position": source,
            "destination_position": destination,
            "accuracy": np.random.normal(0.98, 0.02),  # Simulate accuracy
        }

    async def _mix(self, action: ExperimentalAction) -> Dict[str, Any]:
        """Perform mixing."""
        cycles = action.parameters.get("mix_cycles", 5)
        volume = action.parameters.get("mix_volume", 100.0)

        await asyncio.sleep(cycles * 0.5)

        logger.info(f"Mixed {cycles} cycles with {volume} µL volume")

        return {
            "mix_cycles": cycles,
            "mix_volume": volume,
            "mixing_efficiency": np.random.normal(0.95, 0.03),
        }

    async def _aspirate(self, action: ExperimentalAction) -> Dict[str, Any]:
        """Aspirate liquid."""
        volume = action.parameters.get("volume", 0.0)
        source = action.parameters.get("source")

        await asyncio.sleep(0.1)

        logger.info(f"Aspirated {volume} µL from {source}")

        return {"volume_aspirated": volume, "source_position": source}

    async def _dispense(self, action: ExperimentalAction) -> Dict[str, Any]:
        """Dispense liquid."""
        volume = action.parameters.get("volume", 0.0)
        destination = action.parameters.get("destination")

        await asyncio.sleep(0.1)

        logger.info(f"Dispensed {volume} µL to {destination}")

        return {"volume_dispensed": volume, "destination_position": destination}

    async def _wash_tips(self, action: ExperimentalAction) -> Dict[str, Any]:
        """Wash pipette tips."""
        wash_cycles = action.parameters.get("wash_cycles", 3)

        await asyncio.sleep(wash_cycles * 0.2)

        logger.info(f"Washed tips with {wash_cycles} cycles")

        return {"wash_cycles": wash_cycles, "wash_efficiency": 0.99}

    async def get_status(self) -> Dict[str, Any]:
        """Get liquid handler status."""
        return {
            "connected": self.connected,
            "status": self.status,
            "tip_count": len(self.tip_racks),
            "active_plates": len(self.plate_positions),
            "temperature": 25.0,
            "humidity": 45.0,
        }

    async def calibrate(self) -> bool:
        """Calibrate liquid handler."""
        logger.info("Calibrating liquid handler...")
        await asyncio.sleep(2.0)
        logger.info("Calibration complete")
        return True


class ReactorDriver(EquipmentDriver):
    """Driver for chemical reactors."""

    def __init__(self, labware_def: LabwareDefinition):
        super().__init__(labware_def)
        self.current_temperature = 25.0
        self.current_pressure = 1.0
        self.stirring_speed = 0.0

    async def connect(self) -> bool:
        """Connect to reactor."""
        try:
            await asyncio.sleep(0.1)
            self.connected = True
            logger.info(f"Connected to reactor: {self.labware_def.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to reactor: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from reactor."""
        self.connected = False
        logger.info(f"Disconnected from reactor: {self.labware_def.name}")
        return True

    async def execute_action(self, action: ExperimentalAction) -> ExecutionResult:
        """Execute reactor action."""
        start_time = datetime.now()

        try:
            if action.action_type == ActionType.HEAT:
                result = await self._heat(action)
            elif action.action_type == ActionType.COOL:
                result = await self._cool(action)
            elif action.action_type == ActionType.MIX:
                result = await self._stir(action)
            elif action.action_type == ActionType.WAIT:
                result = await self._wait(action)
            elif action.action_type == ActionType.INCUBATE:
                result = await self._incubate(action)
            else:
                raise AutonomousLabError(
                    f"Unsupported action type: {action.action_type}"
                )

            end_time = datetime.now()

            return ExecutionResult(
                protocol_id=action.action_id,
                action_id=action.action_id,
                success=True,
                start_time=start_time,
                end_time=end_time,
                actual_parameters=action.parameters,
                measurements=result,
            )

        except Exception as e:
            end_time = datetime.now()
            return ExecutionResult(
                protocol_id=action.action_id,
                action_id=action.action_id,
                success=False,
                start_time=start_time,
                end_time=end_time,
                actual_parameters=action.parameters,
                error_message=str(e),
            )

    async def _heat(self, action: ExperimentalAction) -> Dict[str, Any]:
        """Heat reactor."""
        target_temp = action.parameters.get("temperature", 25.0)
        ramp_rate = action.parameters.get("ramp_rate", 5.0)  # °C/min

        temp_diff = abs(target_temp - self.current_temperature)
        heat_time = temp_diff / ramp_rate * 60  # Convert to seconds

        await asyncio.sleep(min(heat_time, 10.0))  # Cap at 10 seconds for simulation

        self.current_temperature = target_temp
        logger.info(f"Heated reactor to {target_temp}°C")

        return {
            "final_temperature": self.current_temperature,
            "heating_time": heat_time,
            "temperature_stability": np.random.normal(0.1, 0.05),
        }

    async def _cool(self, action: ExperimentalAction) -> Dict[str, Any]:
        """Cool reactor."""
        target_temp = action.parameters.get("temperature", 25.0)
        cooling_rate = action.parameters.get("cooling_rate", 3.0)  # °C/min

        temp_diff = abs(self.current_temperature - target_temp)
        cool_time = temp_diff / cooling_rate * 60

        await asyncio.sleep(min(cool_time, 10.0))

        self.current_temperature = target_temp
        logger.info(f"Cooled reactor to {target_temp}°C")

        return {
            "final_temperature": self.current_temperature,
            "cooling_time": cool_time,
            "temperature_stability": np.random.normal(0.1, 0.05),
        }

    async def _stir(self, action: ExperimentalAction) -> Dict[str, Any]:
        """Start stirring."""
        stirring_speed = action.parameters.get("stirring_speed", 300.0)
        duration = action.parameters.get("duration", 60.0)

        self.stirring_speed = stirring_speed
        await asyncio.sleep(min(duration, 5.0))  # Cap for simulation

        logger.info(f"Stirred at {stirring_speed} RPM for {duration} seconds")

        return {
            "stirring_speed": self.stirring_speed,
            "duration": duration,
            "mixing_efficiency": np.random.normal(0.92, 0.05),
        }

    async def _wait(self, action: ExperimentalAction) -> Dict[str, Any]:
        """Wait for specified time."""
        wait_time = action.parameters.get("duration", 60.0)

        await asyncio.sleep(min(wait_time, 5.0))  # Cap for simulation

        logger.info(f"Waited for {wait_time} seconds")

        return {"wait_time": wait_time, "temperature_drift": np.random.normal(0.0, 0.2)}

    async def _incubate(self, action: ExperimentalAction) -> Dict[str, Any]:
        """Incubate at conditions."""
        temperature = action.parameters.get("temperature", self.current_temperature)
        duration = action.parameters.get("duration", 3600.0)
        stirring = action.parameters.get("stirring_speed", 0.0)

        # Set conditions
        if temperature != self.current_temperature:
            await self._heat(
                ExperimentalAction(
                    action_id="temp_set",
                    action_type=ActionType.HEAT,
                    equipment=action.equipment,
                    parameters={"temperature": temperature},
                )
            )

        if stirring > 0:
            self.stirring_speed = stirring

        await asyncio.sleep(min(duration, 10.0))  # Cap for simulation

        logger.info(f"Incubated at {temperature}°C for {duration} seconds")

        return {
            "incubation_temperature": temperature,
            "incubation_time": duration,
            "stirring_speed": self.stirring_speed,
            "temperature_stability": np.random.normal(0.1, 0.03),
            "reaction_progress": np.random.uniform(0.7, 0.95),
        }

    async def get_status(self) -> Dict[str, Any]:
        """Get reactor status."""
        return {
            "connected": self.connected,
            "status": self.status,
            "temperature": self.current_temperature,
            "pressure": self.current_pressure,
            "stirring_speed": self.stirring_speed,
            "volume": 250.0,  # mL
            "atmosphere": "N2",
        }

    async def calibrate(self) -> bool:
        """Calibrate reactor."""
        logger.info("Calibrating reactor sensors...")
        await asyncio.sleep(3.0)
        logger.info("Reactor calibration complete")
        return True


class SpectrometerDriver(EquipmentDriver):
    """Driver for analytical spectrometers."""

    def __init__(self, labware_def: LabwareDefinition):
        super().__init__(labware_def)
        self.wavelength_range = (200, 800)  # nm
        self.resolution = 1.0  # nm

    async def connect(self) -> bool:
        """Connect to spectrometer."""
        try:
            await asyncio.sleep(0.2)
            self.connected = True
            logger.info(f"Connected to spectrometer: {self.labware_def.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to spectrometer: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from spectrometer."""
        self.connected = False
        logger.info(f"Disconnected from spectrometer: {self.labware_def.name}")
        return True

    async def execute_action(self, action: ExperimentalAction) -> ExecutionResult:
        """Execute measurement action."""
        start_time = datetime.now()

        try:
            if action.action_type == ActionType.MEASURE:
                result = await self._measure_spectrum(action)
            else:
                raise AutonomousLabError(
                    f"Unsupported action type: {action.action_type}"
                )

            end_time = datetime.now()

            return ExecutionResult(
                protocol_id=action.action_id,
                action_id=action.action_id,
                success=True,
                start_time=start_time,
                end_time=end_time,
                actual_parameters=action.parameters,
                measurements=result,
            )

        except Exception as e:
            end_time = datetime.now()
            return ExecutionResult(
                protocol_id=action.action_id,
                action_id=action.action_id,
                success=False,
                start_time=start_time,
                end_time=end_time,
                actual_parameters=action.parameters,
                error_message=str(e),
            )

    async def _measure_spectrum(self, action: ExperimentalAction) -> Dict[str, Any]:
        """Measure absorption spectrum."""
        wavelength_start = action.parameters.get("wavelength_start", 200)
        wavelength_end = action.parameters.get("wavelength_end", 800)
        integration_time = action.parameters.get("integration_time", 1.0)

        await asyncio.sleep(integration_time)

        # Generate synthetic spectrum
        wavelengths = np.arange(wavelength_start, wavelength_end, self.resolution)
        # Simple synthetic spectrum with peaks
        spectrum = (
            np.exp(-((wavelengths - 280) ** 2) / 1000) * 0.8
            + np.exp(-((wavelengths - 350) ** 2) / 2000) * 0.6
            + np.random.normal(0, 0.02, len(wavelengths))
        )

        logger.info(f"Measured spectrum from {wavelength_start} to {wavelength_end} nm")

        return {
            "wavelengths": wavelengths.tolist(),
            "absorbance": spectrum.tolist(),
            "integration_time": integration_time,
            "signal_to_noise": np.random.normal(150, 20),
            "baseline_drift": np.random.normal(0.001, 0.0005),
        }

    async def get_status(self) -> Dict[str, Any]:
        """Get spectrometer status."""
        return {
            "connected": self.connected,
            "status": self.status,
            "lamp_hours": 1250.5,
            "temperature": 22.5,
            "wavelength_range": self.wavelength_range,
            "resolution": self.resolution,
        }

    async def calibrate(self) -> bool:
        """Calibrate spectrometer."""
        logger.info("Calibrating spectrometer wavelength...")
        await asyncio.sleep(1.5)
        logger.info("Spectrometer calibration complete")
        return True


class ProtocolGenerator:
    """Generates machine-executable protocols from experimental conditions."""

    def __init__(self, format_type: ProtocolFormat = ProtocolFormat.JSON_PROTOCOL):
        self.format_type = format_type
        self.action_templates = self._load_action_templates()

    def generate_protocol(
        self,
        experimental_conditions: Dict[str, Any],
        labware: List[LabwareDefinition],
        reagents: List[Reagent],
    ) -> ExperimentalProtocol:
        """Generate experimental protocol from conditions."""
        protocol_id = str(uuid.uuid4())

        # Convert conditions to actions
        actions = self._conditions_to_actions(
            experimental_conditions, labware, reagents
        )

        # Create vessels based on requirements
        vessels = self._create_required_vessels(actions, reagents)

        # Estimate duration
        estimated_duration = sum(action.duration or 0 for action in actions)

        protocol = ExperimentalProtocol(
            protocol_id=protocol_id,
            name=f"Automated Protocol {protocol_id[:8]}",
            description="Auto-generated protocol from optimized experimental conditions",
            actions=actions,
            labware=labware,
            reagents=reagents,
            vessels=vessels,
            estimated_duration=estimated_duration,
        )

        return protocol

    def export_protocol(
        self, protocol: ExperimentalProtocol, output_path: Optional[Path] = None
    ) -> str:
        """Export protocol to specified format."""
        if self.format_type == ProtocolFormat.JSON_PROTOCOL:
            return self._export_json(protocol, output_path)
        elif self.format_type == ProtocolFormat.XML_PROTOCOL:
            return self._export_xml(protocol, output_path)
        elif self.format_type == ProtocolFormat.YAML_PROTOCOL:
            return self._export_yaml(protocol, output_path)
        elif self.format_type == ProtocolFormat.PYTHON_SCRIPT:
            return self._export_python(protocol, output_path)
        elif self.format_type == ProtocolFormat.HAMILTON_VENUS:
            return self._export_hamilton_venus(protocol, output_path)
        elif self.format_type == ProtocolFormat.OPENTRONS:
            return self._export_opentrons(protocol, output_path)
        else:
            raise ProtocolGenerationError(f"Unsupported format: {self.format_type}")

    def _load_action_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load action templates for protocol generation."""
        return {
            "liquid_transfer": {
                "equipment_type": EquipmentType.LIQUID_HANDLER,
                "base_duration": 10.0,
                "required_parameters": ["volume", "source", "destination"],
            },
            "heating": {
                "equipment_type": EquipmentType.REACTOR,
                "base_duration": 300.0,
                "required_parameters": ["temperature", "duration"],
            },
            "mixing": {
                "equipment_type": EquipmentType.STIRRER,
                "base_duration": 60.0,
                "required_parameters": ["speed", "duration"],
            },
            "measurement": {
                "equipment_type": EquipmentType.SPECTROMETER,
                "base_duration": 5.0,
                "required_parameters": ["measurement_type"],
            },
        }

    def _conditions_to_actions(
        self,
        conditions: Dict[str, Any],
        labware: List[LabwareDefinition],
        reagents: List[Reagent],
    ) -> List[ExperimentalAction]:
        """Convert experimental conditions to protocol actions."""
        actions = []
        action_counter = 0

        # Add reagent preparation actions
        for i, reagent in enumerate(reagents):
            if reagent.concentration and reagent.concentration > 0:
                action = ExperimentalAction(
                    action_id=f"action_{action_counter:03d}",
                    action_type=ActionType.TRANSFER_LIQUID,
                    equipment="liquid_handler_001",
                    parameters={
                        "volume": 1000.0,  # µL
                        "source": f"reagent_reservoir_{i}",
                        "destination": "reaction_vessel",
                        "reagent": reagent.name,
                    },
                    reagents=[reagent],
                    duration=10.0,
                )
                actions.append(action)
                action_counter += 1

        # Add mixing action
        if "stirring_speed" in conditions:
            action = ExperimentalAction(
                action_id=f"action_{action_counter:03d}",
                action_type=ActionType.MIX,
                equipment="stirrer_001",
                parameters={
                    "stirring_speed": conditions["stirring_speed"],
                    "duration": 60.0,
                },
                duration=60.0,
            )
            actions.append(action)
            action_counter += 1

        # Add heating action
        if "temperature" in conditions and conditions["temperature"] > 25:
            action = ExperimentalAction(
                action_id=f"action_{action_counter:03d}",
                action_type=ActionType.HEAT,
                equipment="reactor_001",
                parameters={"temperature": conditions["temperature"], "ramp_rate": 5.0},
                temperature=conditions["temperature"],
                duration=120.0,
            )
            actions.append(action)
            action_counter += 1

        # Add reaction time
        if "time" in conditions:
            reaction_time = conditions["time"] * 3600  # Convert hours to seconds
            action = ExperimentalAction(
                action_id=f"action_{action_counter:03d}",
                action_type=ActionType.INCUBATE,
                equipment="reactor_001",
                parameters={
                    "duration": reaction_time,
                    "temperature": conditions.get("temperature", 25.0),
                    "stirring_speed": conditions.get("stirring_speed", 300.0),
                },
                duration=reaction_time,
            )
            actions.append(action)
            action_counter += 1

        # Add measurement action
        action = ExperimentalAction(
            action_id=f"action_{action_counter:03d}",
            action_type=ActionType.MEASURE,
            equipment="spectrometer_001",
            parameters={
                "measurement_type": "uv_vis",
                "wavelength_start": 200,
                "wavelength_end": 800,
                "integration_time": 1.0,
            },
            duration=5.0,
        )
        actions.append(action)

        return actions

    def _create_required_vessels(
        self, actions: List[ExperimentalAction], reagents: List[Reagent]
    ) -> List[Vessel]:
        """Create vessels required for the protocol."""
        vessels = []

        # Main reaction vessel
        reaction_vessel = Vessel(
            id="reaction_vessel",
            vessel_type="reaction_vial",
            volume_capacity=2000.0,  # µL
            position=(1, 1),
        )
        vessels.append(reaction_vessel)

        # Reagent reservoirs
        for i, reagent in enumerate(reagents):
            reservoir = Vessel(
                id=f"reagent_reservoir_{i}",
                vessel_type="reagent_reservoir",
                volume_capacity=50000.0,  # µL
                current_volume=45000.0,
                contents=[reagent],
                position=(2, i + 1),
            )
            vessels.append(reservoir)

        # Waste container
        waste_vessel = Vessel(
            id="waste_container",
            vessel_type="waste",
            volume_capacity=100000.0,  # µL
            position=(3, 1),
        )
        vessels.append(waste_vessel)

        return vessels

    def _export_json(
        self, protocol: ExperimentalProtocol, output_path: Optional[Path] = None
    ) -> str:
        """Export protocol as JSON."""
        protocol_dict = {
            "protocol_id": protocol.protocol_id,
            "name": protocol.name,
            "description": protocol.description,
            "version": protocol.version,
            "created_date": protocol.created_date.isoformat(),
            "estimated_duration": protocol.estimated_duration,
            "actions": [
                {
                    "action_id": action.action_id,
                    "action_type": action.action_type.value,
                    "equipment": action.equipment,
                    "parameters": action.parameters,
                    "source_vessels": action.source_vessels,
                    "destination_vessels": action.destination_vessels,
                    "duration": action.duration,
                    "temperature": action.temperature,
                    "pressure": action.pressure,
                }
                for action in protocol.actions
            ],
            "labware": [
                {
                    "name": labware.name,
                    "equipment_type": labware.equipment_type.value,
                    "model": labware.model,
                    "manufacturer": labware.manufacturer,
                    "capabilities": labware.capabilities,
                    "parameters": labware.parameters,
                }
                for labware in protocol.labware
            ],
            "reagents": [
                {
                    "name": reagent.name,
                    "cas_number": reagent.cas_number,
                    "molecular_weight": reagent.molecular_weight,
                    "concentration": reagent.concentration,
                    "purity": reagent.purity,
                }
                for reagent in protocol.reagents
            ],
            "vessels": [
                {
                    "id": vessel.id,
                    "vessel_type": vessel.vessel_type,
                    "volume_capacity": vessel.volume_capacity,
                    "current_volume": vessel.current_volume,
                    "position": vessel.position,
                }
                for vessel in protocol.vessels
            ],
        }

        json_content = json.dumps(protocol_dict, indent=2)

        if output_path:
            output_path.write_text(json_content)

        return json_content

    def _export_xml(
        self, protocol: ExperimentalProtocol, output_path: Optional[Path] = None
    ) -> str:
        """Export protocol as XML."""
        root = ET.Element("experimental_protocol")
        root.set("id", protocol.protocol_id)
        root.set("version", protocol.version)

        # Protocol metadata
        metadata = ET.SubElement(root, "metadata")
        ET.SubElement(metadata, "name").text = protocol.name
        ET.SubElement(metadata, "description").text = protocol.description
        ET.SubElement(metadata, "created_date").text = protocol.created_date.isoformat()

        # Actions
        actions_elem = ET.SubElement(root, "actions")
        for action in protocol.actions:
            action_elem = ET.SubElement(actions_elem, "action")
            action_elem.set("id", action.action_id)
            action_elem.set("type", action.action_type.value)

            ET.SubElement(action_elem, "equipment").text = action.equipment

            params_elem = ET.SubElement(action_elem, "parameters")
            for key, value in action.parameters.items():
                param_elem = ET.SubElement(params_elem, "parameter")
                param_elem.set("name", key)
                param_elem.text = str(value)

        xml_content = ET.tostring(root, encoding="unicode")

        if output_path:
            output_path.write_text(xml_content)

        return xml_content

    def _export_yaml(
        self, protocol: ExperimentalProtocol, output_path: Optional[Path] = None
    ) -> str:
        """Export protocol as YAML."""
        if not YAML_AVAILABLE:
            raise ProtocolGenerationError("PyYAML not available for YAML export")

        protocol_dict = {
            "protocol_id": protocol.protocol_id,
            "name": protocol.name,
            "description": protocol.description,
            "version": protocol.version,
            "actions": [
                {
                    "action_id": action.action_id,
                    "action_type": action.action_type.value,
                    "equipment": action.equipment,
                    "parameters": action.parameters,
                    "duration": action.duration,
                }
                for action in protocol.actions
            ],
        }

        yaml_content = yaml.dump(protocol_dict, default_flow_style=False)

        if output_path:
            output_path.write_text(yaml_content)

        return yaml_content

    def _export_python(
        self, protocol: ExperimentalProtocol, output_path: Optional[Path] = None
    ) -> str:
        """Export protocol as Python script."""
        python_code = f'''#!/usr/bin/env python3
"""
Auto-generated experimental protocol: {protocol.name}
Protocol ID: {protocol.protocol_id}
Generated on: {protocol.created_date.isoformat()}
"""

import asyncio
from datetime import datetime

async def run_protocol():
    """Execute the experimental protocol."""
    print(f"Starting protocol: {protocol.name}")
    print(f"Protocol ID: {protocol.protocol_id}")
    print(f"Estimated duration: {protocol.estimated_duration} seconds")

    start_time = datetime.now()

'''

        for i, action in enumerate(protocol.actions):
            python_code += f"""
    # Action {i + 1}: {action.action_type.value}
    print(f"Executing action {action.action_id}: {action.action_type.value}")

    # Equipment: {action.equipment}
    # Parameters: {action.parameters}

    # Simulate action execution
    await asyncio.sleep({action.duration or 1.0})
    print(f"Completed action {action.action_id}")
"""

        python_code += """

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Protocol completed in {duration:.1f} seconds")

if __name__ == "__main__":
    asyncio.run(run_protocol())
"""

        if output_path:
            output_path.write_text(python_code)

        return python_code

    def _export_hamilton_venus(
        self, protocol: ExperimentalProtocol, output_path: Optional[Path] = None
    ) -> str:
        """Export protocol for Hamilton Venus liquid handler."""
        venus_code = f"""// Hamilton Venus Protocol: {protocol.name}
// Protocol ID: {protocol.protocol_id}
// Auto-generated on: {protocol.created_date.isoformat()}

#include "HSLLabwrAccess.hs_"
#include "HSLStarLine.hs_"

method {protocol.protocol_id.replace("-", "_")}()
{{
    // Initialize liquid handler
    Venus::Initialize();

    // Protocol variables
    variable volume;
    variable source_position;
    variable destination_position;

"""

        for action in protocol.actions:
            if action.action_type == ActionType.TRANSFER_LIQUID:
                volume = action.parameters.get("volume", 100)
                source = action.parameters.get("source", "A1")
                dest = action.parameters.get("destination", "B1")

                venus_code += f"""
    // {action.action_id}: Liquid transfer
    volume = {volume};
    source_position = "{source}";
    destination_position = "{dest}";

    Pipette(volume, source_position, destination_position);

"""
            elif action.action_type == ActionType.MIX:
                cycles = action.parameters.get("mix_cycles", 5)
                volume = action.parameters.get("mix_volume", 100)

                venus_code += f"""
    // {action.action_id}: Mixing
    Mix({volume}, "{action.parameters.get('position', 'A1')}", {cycles});

"""

        venus_code += """
    // Clean up
    Venus::Terminate();
}
"""

        if output_path:
            output_path.write_text(venus_code)

        return venus_code

    def _export_opentrons(
        self, protocol: ExperimentalProtocol, output_path: Optional[Path] = None
    ) -> str:
        """Export protocol for Opentrons robot."""
        opentrons_code = f"""from opentrons import protocol_api

# Protocol: {protocol.name}
# Protocol ID: {protocol.protocol_id}
# Auto-generated on: {protocol.created_date.isoformat()}

metadata = {{
    "protocolName": "{protocol.name}",
    "author": "ChemiAccel AutoLab",
    "description": "{protocol.description}",
    "apiLevel": "2.13"
}}

def run(protocol: protocol_api.ProtocolContext):
    # Load labware
    tips = protocol.load_labware("opentrons_96_tiprack_300ul", 1)
    plate = protocol.load_labware("corning_96_wellplate_360ul_flat", 2)
    reagent_plate = protocol.load_labware("nest_12_reservoir_15ml", 3)

    # Load instrument
    pipette = protocol.load_instrument("p300_single_gen2", "right", tip_racks=[tips])

    # Protocol actions
"""

        for action in protocol.actions:
            if action.action_type == ActionType.TRANSFER_LIQUID:
                volume = action.parameters.get("volume", 100)
                source = action.parameters.get("source", "A1")
                dest = action.parameters.get("destination", "A1")

                opentrons_code += f"""
    # {action.action_id}: Transfer liquid
    pipette.transfer({volume}, reagent_plate["{source}"], plate["{dest}"])
"""
            elif action.action_type == ActionType.MIX:
                cycles = action.parameters.get("mix_cycles", 5)
                volume = action.parameters.get("mix_volume", 100)
                position = action.parameters.get("position", "A1")

                opentrons_code += f"""
    # {action.action_id}: Mix
    pipette.mix({cycles}, {volume}, plate["{position}"])
"""

        if output_path:
            output_path.write_text(opentrons_code)

        return opentrons_code


class EquipmentManager:
    """Manages laboratory equipment and protocol execution."""

    def __init__(self):
        self.equipment_drivers = {}
        self.active_protocols = {}
        self.execution_results = []

    def register_equipment(self, equipment_id: str, driver: EquipmentDriver) -> None:
        """Register equipment driver."""
        self.equipment_drivers[equipment_id] = driver
        logger.info(f"Registered equipment: {equipment_id}")

    async def connect_all_equipment(self) -> Dict[str, bool]:
        """Connect to all registered equipment."""
        connection_results = {}

        for equipment_id, driver in self.equipment_drivers.items():
            try:
                success = await driver.connect()
                connection_results[equipment_id] = success
            except Exception as e:
                logger.error(f"Failed to connect to {equipment_id}: {e}")
                connection_results[equipment_id] = False

        return connection_results

    async def execute_protocol(
        self, protocol: ExperimentalProtocol
    ) -> List[ExecutionResult]:
        """Execute experimental protocol."""
        logger.info(f"Starting protocol execution: {protocol.protocol_id}")

        self.active_protocols[protocol.protocol_id] = protocol
        execution_results = []

        try:
            for action in protocol.actions:
                # Find appropriate equipment
                driver = self._find_equipment_for_action(action)

                if driver is None:
                    error_result = ExecutionResult(
                        protocol_id=protocol.protocol_id,
                        action_id=action.action_id,
                        success=False,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        actual_parameters=action.parameters,
                        error_message=f"No suitable equipment found for action {action.action_type}",
                    )
                    execution_results.append(error_result)
                    continue

                # Execute action
                result = await driver.execute_action(action)
                execution_results.append(result)

                # Check for errors
                if not result.success:
                    logger.error(
                        f"Action {action.action_id} failed: {result.error_message}"
                    )
                    # Could implement error handling/retry logic here

                logger.info(f"Completed action {action.action_id}")

            logger.info(f"Protocol {protocol.protocol_id} execution completed")

        except Exception as e:
            logger.error(f"Protocol execution failed: {e}")
            raise AutonomousLabError(f"Protocol execution failed: {e}")

        finally:
            if protocol.protocol_id in self.active_protocols:
                del self.active_protocols[protocol.protocol_id]

        self.execution_results.extend(execution_results)
        return execution_results

    def _find_equipment_for_action(
        self, action: ExperimentalAction
    ) -> Optional[EquipmentDriver]:
        """Find suitable equipment for action."""
        for equipment_id, driver in self.equipment_drivers.items():
            if equipment_id == action.equipment:
                return driver

            # Check equipment type compatibility
            if action.action_type in [
                ActionType.TRANSFER_LIQUID,
                ActionType.ASPIRATE,
                ActionType.DISPENSE,
            ]:
                if driver.labware_def.equipment_type == EquipmentType.LIQUID_HANDLER:
                    return driver
            elif action.action_type in [
                ActionType.HEAT,
                ActionType.COOL,
                ActionType.INCUBATE,
            ]:
                if driver.labware_def.equipment_type == EquipmentType.REACTOR:
                    return driver
            elif action.action_type == ActionType.MEASURE:
                if driver.labware_def.equipment_type == EquipmentType.SPECTROMETER:
                    return driver

        return None

    async def get_equipment_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all equipment."""
        status_dict = {}

        for equipment_id, driver in self.equipment_drivers.items():
            try:
                status = await driver.get_status()
                status_dict[equipment_id] = status
            except Exception as e:
                status_dict[equipment_id] = {"error": str(e)}

        return status_dict

    async def calibrate_equipment(
        self, equipment_id: Optional[str] = None
    ) -> Dict[str, bool]:
        """Calibrate equipment."""
        calibration_results = {}

        if equipment_id:
            if equipment_id in self.equipment_drivers:
                try:
                    success = await self.equipment_drivers[equipment_id].calibrate()
                    calibration_results[equipment_id] = success
                except Exception as e:
                    logger.error(f"Calibration failed for {equipment_id}: {e}")
                    calibration_results[equipment_id] = False
        else:
            # Calibrate all equipment
            for equipment_id, driver in self.equipment_drivers.items():
                try:
                    success = await driver.calibrate()
                    calibration_results[equipment_id] = success
                except Exception as e:
                    logger.error(f"Calibration failed for {equipment_id}: {e}")
                    calibration_results[equipment_id] = False

        return calibration_results


# Utility functions
def create_standard_labware() -> List[LabwareDefinition]:
    """Create standard laboratory equipment definitions."""
    return [
        LabwareDefinition(
            name="Hamilton STAR",
            equipment_type=EquipmentType.LIQUID_HANDLER,
            model="STAR",
            manufacturer="Hamilton",
            capabilities=["liquid_transfer", "mixing", "washing"],
            parameters={
                "max_volume": 1000.0,
                "min_volume": 0.5,
                "accuracy": 0.02,
                "precision": 0.01,
            },
        ),
        LabwareDefinition(
            name="IKA RCT Digital",
            equipment_type=EquipmentType.REACTOR,
            model="RCT Digital",
            manufacturer="IKA",
            capabilities=["heating", "stirring", "temperature_control"],
            parameters={
                "max_temperature": 300.0,
                "temperature_accuracy": 0.5,
                "stirring_range": [50, 1500],
            },
        ),
        LabwareDefinition(
            name="Ocean Optics USB4000",
            equipment_type=EquipmentType.SPECTROMETER,
            model="USB4000",
            manufacturer="Ocean Optics",
            capabilities=["uv_vis_spectroscopy", "kinetics", "absorbance"],
            parameters={
                "wavelength_range": [200, 1100],
                "resolution": 0.3,
                "integration_time_range": [0.003, 65],
            },
        ),
    ]


def create_standard_reagents() -> List[Reagent]:
    """Create standard reagent library."""
    return [
        Reagent(
            name="Benzaldehyde",
            cas_number="100-52-7",
            molecular_weight=106.12,
            density=1.044,
            purity=0.99,
            storage_conditions="Store in cool, dry place",
        ),
        Reagent(
            name="Sodium Hydroxide",
            cas_number="1310-73-2",
            molecular_weight=39.997,
            density=2.13,
            purity=0.98,
            storage_conditions="Store in tightly closed container",
        ),
        Reagent(
            name="Ethanol",
            cas_number="64-17-5",
            molecular_weight=46.07,
            density=0.789,
            purity=0.999,
            storage_conditions="Store in cool, ventilated area",
        ),
        Reagent(
            name="Water",
            cas_number="7732-18-5",
            molecular_weight=18.015,
            density=1.0,
            purity=1.0,
            storage_conditions="Room temperature",
        ),
    ]


async def setup_autonomous_lab() -> EquipmentManager:
    """Set up autonomous laboratory with standard equipment."""
    manager = EquipmentManager()

    # Create standard labware definitions
    labware_definitions = create_standard_labware()

    # Register equipment drivers
    for labware in labware_definitions:
        if labware.equipment_type == EquipmentType.LIQUID_HANDLER:
            driver = LiquidHandlerDriver(labware)
        elif labware.equipment_type == EquipmentType.REACTOR:
            driver = ReactorDriver(labware)
        elif labware.equipment_type == EquipmentType.SPECTROMETER:
            driver = SpectrometerDriver(labware)
        else:
            continue

        equipment_id = f"{labware.equipment_type.value}_001"
        manager.register_equipment(equipment_id, driver)

    # Connect to equipment
    await manager.connect_all_equipment()

    return manager


async def run_autonomous_experiment(
    experimental_conditions: Dict[str, Any], output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Run complete autonomous experiment."""
    # Set up lab
    manager = await setup_autonomous_lab()

    # Create reagents and labware
    reagents = create_standard_reagents()
    labware = create_standard_labware()

    # Generate protocol
    generator = ProtocolGenerator(ProtocolFormat.JSON_PROTOCOL)
    protocol = generator.generate_protocol(experimental_conditions, labware, reagents)

    # Export protocol if output directory specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        protocol_file = output_dir / f"protocol_{protocol.protocol_id}.json"
        generator.export_protocol(protocol, protocol_file)

    # Execute protocol
    results = await manager.execute_protocol(protocol)

    # Compile execution summary
    successful_actions = sum(1 for result in results if result.success)
    total_actions = len(results)
    total_time = sum(
        (result.end_time - result.start_time).total_seconds() for result in results
    )

    summary = {
        "protocol_id": protocol.protocol_id,
        "experimental_conditions": experimental_conditions,
        "total_actions": total_actions,
        "successful_actions": successful_actions,
        "success_rate": successful_actions / total_actions if total_actions > 0 else 0,
        "total_execution_time": total_time,
        "results": [
            {
                "action_id": result.action_id,
                "success": result.success,
                "duration": (result.end_time - result.start_time).total_seconds(),
                "measurements": result.measurements,
                "error": result.error_message,
            }
            for result in results
        ],
    }

    return summary
