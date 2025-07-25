"""
Closed-Loop Optimization with Robotic Platforms

This module implements comprehensive integration between ChemiAccel's AI decision-making
systems and external robotic control platforms, enabling real-time feedback loops and
automated experimental execution based on AI recommendations.
"""

import asyncio
import json
import logging
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

# Optional imports for different communication protocols
try:
    import requests

    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False

try:
    import websockets

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

try:
    import paho.mqtt.client as mqtt

    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False


logger = logging.getLogger(__name__)


class RobotStatus(Enum):
    """Robot operational status."""

    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


class ExperimentStatus(Enum):
    """Experiment execution status."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OptimizationObjective(Enum):
    """Optimization objectives for closed-loop systems."""

    MAXIMIZE_YIELD = "maximize_yield"
    MINIMIZE_TIME = "minimize_time"
    MAXIMIZE_PURITY = "maximize_purity"
    MINIMIZE_COST = "minimize_cost"
    MULTI_OBJECTIVE = "multi_objective"


@dataclass
class RobotCapability:
    """Represents a specific capability of a robotic platform."""

    capability_id: str
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: float = 0.0  # seconds
    resource_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentalAction:
    """Represents an action to be performed by a robotic platform."""

    action_id: str
    robot_id: str
    capability_id: str
    parameters: Dict[str, Any]
    priority: int = 1  # 1=low, 5=high
    dependencies: List[str] = field(default_factory=list)
    expected_duration: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ExperimentPlan:
    """Complete experimental plan for robotic execution."""

    plan_id: str
    experiment_id: str
    actions: List[ExperimentalAction]
    optimization_objective: OptimizationObjective
    target_parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RobotFeedback:
    """Feedback from robotic platform about experiment execution."""

    feedback_id: str
    robot_id: str
    action_id: str
    status: ExperimentStatus
    progress: float  # 0.0 to 1.0
    data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class OptimizationState:
    """Current state of closed-loop optimization."""

    iteration: int
    current_parameters: Dict[str, Any]
    objective_values: List[float]
    best_parameters: Dict[str, Any]
    best_objective: float
    convergence_criteria: Dict[str, Any] = field(default_factory=dict)
    is_converged: bool = False
    total_experiments: int = 0
    successful_experiments: int = 0


class CommunicationProtocol(ABC):
    """Abstract base class for robot communication protocols."""

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to robotic platform."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from robotic platform."""
        pass

    @abstractmethod
    async def send_command(
        self, robot_id: str, command: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send command to robot and get response."""
        pass

    @abstractmethod
    async def get_robot_status(self, robot_id: str) -> RobotStatus:
        """Get current robot status."""
        pass

    @abstractmethod
    def set_feedback_callback(self, callback: Callable[[RobotFeedback], None]) -> None:
        """Set callback for receiving robot feedback."""
        pass


class HTTPRobotProtocol(CommunicationProtocol):
    """HTTP-based communication protocol for robotic platforms."""

    def __init__(self, base_url: str, auth_token: Optional[str] = None):
        if not HTTP_AVAILABLE:
            raise ImportError("HTTP protocol requires 'requests' library")

        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.session = requests.Session()
        self.feedback_callback: Optional[Callable[[RobotFeedback], None]] = None
        self._polling_task: Optional[asyncio.Task] = None
        self._is_connected = False

    async def connect(self) -> bool:
        """Establish HTTP connection."""
        try:
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"

            response = self.session.get(
                f"{self.base_url}/health", headers=headers, timeout=10
            )
            response.raise_for_status()

            self._is_connected = True

            # Start feedback polling
            if self.feedback_callback:
                self._polling_task = asyncio.create_task(self._poll_feedback())

            logger.info(f"Connected to robot HTTP API at {self.base_url}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to robot HTTP API: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect HTTP session."""
        self._is_connected = False

        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass

        self.session.close()
        logger.info("Disconnected from robot HTTP API")

    async def send_command(
        self, robot_id: str, command: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send HTTP command to robot."""
        if not self._is_connected:
            raise RuntimeError("Not connected to robot API")

        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        try:
            response = self.session.post(
                f"{self.base_url}/robots/{robot_id}/commands",
                json=command,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Failed to send command to robot {robot_id}: {e}")
            raise

    async def get_robot_status(self, robot_id: str) -> RobotStatus:
        """Get robot status via HTTP."""
        if not self._is_connected:
            raise RuntimeError("Not connected to robot API")

        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        try:
            response = self.session.get(
                f"{self.base_url}/robots/{robot_id}/status", headers=headers, timeout=10
            )
            response.raise_for_status()
            status_data = response.json()
            return RobotStatus(status_data.get("status", "offline"))

        except Exception as e:
            logger.error(f"Failed to get robot {robot_id} status: {e}")
            return RobotStatus.OFFLINE

    def set_feedback_callback(self, callback: Callable[[RobotFeedback], None]) -> None:
        """Set feedback callback."""
        self.feedback_callback = callback

    async def _poll_feedback(self) -> None:
        """Poll for robot feedback via HTTP."""
        while self._is_connected:
            try:
                headers = {}
                if self.auth_token:
                    headers["Authorization"] = f"Bearer {self.auth_token}"

                response = self.session.get(
                    f"{self.base_url}/feedback", headers=headers, timeout=5
                )

                if response.status_code == 200:
                    feedback_data = response.json()
                    for item in feedback_data.get("feedback", []):
                        feedback = RobotFeedback(
                            feedback_id=item.get("feedback_id", str(uuid.uuid4())),
                            robot_id=item.get("robot_id", "unknown"),
                            action_id=item.get("action_id", "unknown"),
                            status=ExperimentStatus(item.get("status", "queued")),
                            progress=item.get("progress", 0.0),
                            data=item.get("data", {}),
                            error_message=item.get("error_message"),
                            timestamp=datetime.fromisoformat(
                                item.get(
                                    "timestamp", datetime.now(timezone.utc).isoformat()
                                )
                            ),
                        )

                        if self.feedback_callback:
                            self.feedback_callback(feedback)

                await asyncio.sleep(1.0)  # Poll every second

            except Exception as e:
                logger.warning(f"Error polling robot feedback: {e}")
                await asyncio.sleep(5.0)  # Wait longer on error


class MQTTRobotProtocol(CommunicationProtocol):
    """MQTT-based communication protocol for robotic platforms."""

    def __init__(
        self,
        broker_host: str,
        broker_port: int = 1883,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        if not MQTT_AVAILABLE:
            raise ImportError("MQTT protocol requires 'paho-mqtt' library")

        self.broker_host = broker_host
        self.broker_port = broker_port
        self.username = username
        self.password = password
        self.client = mqtt.Client()
        self.feedback_callback: Optional[Callable[[RobotFeedback], None]] = None
        self._is_connected = False

        # Set up MQTT callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    async def connect(self) -> bool:
        """Establish MQTT connection."""
        try:
            if self.username and self.password:
                self.client.username_pw_set(self.username, self.password)

            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()

            # Wait for connection
            for _ in range(50):  # 5 second timeout
                if self._is_connected:
                    break
                await asyncio.sleep(0.1)

            if self._is_connected:
                # Subscribe to feedback topics
                self.client.subscribe("robots/+/feedback")
                self.client.subscribe("robots/+/status")
                logger.info(
                    f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}"
                )
                return True
            else:
                logger.error("MQTT connection timeout")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect MQTT client."""
        self._is_connected = False
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("Disconnected from MQTT broker")

    async def send_command(
        self, robot_id: str, command: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send MQTT command to robot."""
        if not self._is_connected:
            raise RuntimeError("Not connected to MQTT broker")

        topic = f"robots/{robot_id}/commands"
        payload = json.dumps(command)

        result = self.client.publish(topic, payload, qos=1)
        result.wait_for_publish(timeout=10)

        if result.rc != 0:
            raise RuntimeError(f"Failed to publish command to topic {topic}")

        # For MQTT, we return a simple acknowledgment
        return {"status": "sent", "timestamp": datetime.now(timezone.utc).isoformat()}

    async def get_robot_status(self, robot_id: str) -> RobotStatus:
        """Get robot status - for MQTT this would be from cached status."""
        # In a real implementation, this would query a status cache
        # For now, return a default status
        return RobotStatus.IDLE

    def set_feedback_callback(self, callback: Callable[[RobotFeedback], None]) -> None:
        """Set feedback callback."""
        self.feedback_callback = callback

    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            self._is_connected = True
            logger.debug("MQTT client connected")
        else:
            logger.error(f"MQTT connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback."""
        self._is_connected = False
        logger.debug("MQTT client disconnected")

    def _on_message(self, client, userdata, msg):
        """MQTT message callback."""
        try:
            topic_parts = msg.topic.split("/")
            if len(topic_parts) >= 3 and topic_parts[0] == "robots":
                robot_id = topic_parts[1]
                message_type = topic_parts[2]

                payload = json.loads(msg.payload.decode())

                if message_type == "feedback" and self.feedback_callback:
                    feedback = RobotFeedback(
                        feedback_id=payload.get("feedback_id", str(uuid.uuid4())),
                        robot_id=robot_id,
                        action_id=payload.get("action_id", "unknown"),
                        status=ExperimentStatus(payload.get("status", "queued")),
                        progress=payload.get("progress", 0.0),
                        data=payload.get("data", {}),
                        error_message=payload.get("error_message"),
                        timestamp=datetime.fromisoformat(
                            payload.get(
                                "timestamp", datetime.now(timezone.utc).isoformat()
                            )
                        ),
                    )
                    self.feedback_callback(feedback)

        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")


class RobotManager:
    """
    Manages communication and coordination with multiple robotic platforms.
    """

    def __init__(self):
        self.robots: Dict[str, Dict[str, Any]] = {}
        self.protocols: Dict[str, CommunicationProtocol] = {}
        self.capabilities: Dict[str, List[RobotCapability]] = {}
        self.active_experiments: Dict[str, ExperimentPlan] = {}
        self.feedback_callbacks: List[Callable[[RobotFeedback], None]] = []
        self._status_cache: Dict[str, Tuple[RobotStatus, datetime]] = {}
        self._lock = threading.Lock()

    def register_robot(
        self,
        robot_id: str,
        protocol: CommunicationProtocol,
        capabilities: List[RobotCapability],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a robotic platform."""
        with self._lock:
            self.robots[robot_id] = {
                "metadata": metadata or {},
                "registered_at": datetime.now(timezone.utc),
                "last_seen": datetime.now(timezone.utc),
            }
            self.protocols[robot_id] = protocol
            self.capabilities[robot_id] = capabilities

            # Set up feedback handling
            protocol.set_feedback_callback(self._handle_robot_feedback)

            logger.info(
                f"Registered robot {robot_id} with {len(capabilities)} capabilities"
            )

    async def connect_all_robots(self) -> Dict[str, bool]:
        """Connect to all registered robots."""
        connection_results = {}

        for robot_id, protocol in self.protocols.items():
            try:
                success = await protocol.connect()
                connection_results[robot_id] = success
                if success:
                    self.robots[robot_id]["last_seen"] = datetime.now(timezone.utc)
            except Exception as e:
                logger.error(f"Failed to connect to robot {robot_id}: {e}")
                connection_results[robot_id] = False

        return connection_results

    async def disconnect_all_robots(self) -> None:
        """Disconnect from all robots."""
        for robot_id, protocol in self.protocols.items():
            try:
                await protocol.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting from robot {robot_id}: {e}")

    async def execute_experiment_plan(self, plan: ExperimentPlan) -> str:
        """Execute an experimental plan across robots."""
        with self._lock:
            self.active_experiments[plan.plan_id] = plan

        logger.info(
            f"Starting execution of experiment plan {plan.plan_id} with {len(plan.actions)} actions"
        )

        # Group actions by robot
        robot_actions: Dict[str, List[ExperimentalAction]] = {}
        for action in plan.actions:
            if action.robot_id not in robot_actions:
                robot_actions[action.robot_id] = []
            robot_actions[action.robot_id].append(action)

        # Execute actions on each robot
        execution_tasks = []
        for robot_id, actions in robot_actions.items():
            if robot_id in self.protocols:
                task = asyncio.create_task(
                    self._execute_robot_actions(robot_id, actions)
                )
                execution_tasks.append(task)

        # Wait for all robot executions to complete
        try:
            await asyncio.gather(*execution_tasks)
            logger.info(f"Completed execution of experiment plan {plan.plan_id}")
            return plan.plan_id
        except Exception as e:
            logger.error(f"Error executing experiment plan {plan.plan_id}: {e}")
            raise

    async def _execute_robot_actions(
        self, robot_id: str, actions: List[ExperimentalAction]
    ) -> None:
        """Execute actions on a specific robot."""
        protocol = self.protocols[robot_id]

        for action in sorted(actions, key=lambda a: a.priority, reverse=True):
            try:
                # Check robot status
                status = await protocol.get_robot_status(robot_id)
                if status != RobotStatus.IDLE:
                    logger.warning(
                        f"Robot {robot_id} not idle (status: {status}), waiting..."
                    )
                    # In a real implementation, would implement better queuing
                    await asyncio.sleep(5)

                # Send command
                command = {
                    "action_id": action.action_id,
                    "capability": action.capability_id,
                    "parameters": action.parameters,
                    "priority": action.priority,
                }

                response = await protocol.send_command(robot_id, command)
                logger.info(
                    f"Sent action {action.action_id} to robot {robot_id}: {response}"
                )

                # Wait for action completion (simplified)
                if action.expected_duration > 0:
                    await asyncio.sleep(action.expected_duration)

            except Exception as e:
                logger.error(
                    f"Failed to execute action {action.action_id} on robot {robot_id}: {e}"
                )
                raise

    def _handle_robot_feedback(self, feedback: RobotFeedback) -> None:
        """Handle feedback from robots."""
        logger.debug(
            f"Received feedback from robot {feedback.robot_id}: {feedback.status}"
        )

        # Update robot last seen time
        with self._lock:
            if feedback.robot_id in self.robots:
                self.robots[feedback.robot_id]["last_seen"] = datetime.now(timezone.utc)

        # Notify registered callbacks
        for callback in self.feedback_callbacks:
            try:
                callback(feedback)
            except Exception as e:
                logger.error(f"Error in feedback callback: {e}")

    def add_feedback_callback(self, callback: Callable[[RobotFeedback], None]) -> None:
        """Add a callback for robot feedback."""
        self.feedback_callbacks.append(callback)

    async def get_robot_status(self, robot_id: str) -> Optional[RobotStatus]:
        """Get current status of a robot."""
        if robot_id not in self.protocols:
            return None

        try:
            status = await self.protocols[robot_id].get_robot_status(robot_id)

            # Cache status
            with self._lock:
                self._status_cache[robot_id] = (status, datetime.now(timezone.utc))

            return status
        except Exception as e:
            logger.error(f"Failed to get status for robot {robot_id}: {e}")
            return None

    def get_robot_capabilities(self, robot_id: str) -> List[RobotCapability]:
        """Get capabilities of a specific robot."""
        return self.capabilities.get(robot_id, [])

    def find_robots_with_capability(self, capability_id: str) -> List[str]:
        """Find robots that have a specific capability."""
        matching_robots = []

        for robot_id, capabilities in self.capabilities.items():
            if any(cap.capability_id == capability_id for cap in capabilities):
                matching_robots.append(robot_id)

        return matching_robots

    def get_robot_summary(self) -> Dict[str, Any]:
        """Get summary of all registered robots."""
        with self._lock:
            return {
                "total_robots": len(self.robots),
                "robots": {
                    robot_id: {
                        "capabilities_count": len(self.capabilities.get(robot_id, [])),
                        "last_seen": info["last_seen"].isoformat(),
                        "metadata": info["metadata"],
                    }
                    for robot_id, info in self.robots.items()
                },
                "active_experiments": len(self.active_experiments),
            }


# Factory functions for common robot types


def create_http_hamilton_robot(
    base_url: str, auth_token: Optional[str] = None
) -> Tuple[str, CommunicationProtocol, List[RobotCapability]]:
    """Create Hamilton liquid handling robot with HTTP protocol."""
    robot_id = "hamilton_microlab_star"
    protocol = HTTPRobotProtocol(base_url, auth_token)

    capabilities = [
        RobotCapability(
            capability_id="liquid_handling",
            name="Liquid Handling",
            description="Precision liquid dispensing and aspiration",
            parameters={
                "min_volume_ul": 0.1,
                "max_volume_ul": 1000.0,
                "volume_precision": 0.1,
            },
            constraints={
                "max_parallel_channels": 8,
                "supported_labware": ["96_well_plate", "384_well_plate", "reservoir"],
            },
            estimated_duration=30.0,
            resource_requirements={"tips": 1, "channels": 1},
        ),
        RobotCapability(
            capability_id="plate_handling",
            name="Plate Handling",
            description="Automated plate movement and positioning",
            parameters={"plate_types": ["96_well", "384_well", "1536_well"]},
            constraints={"max_plate_height": 50.0, "max_plates_simultaneous": 4},  # mm
            estimated_duration=15.0,
            resource_requirements={"gripper": 1},
        ),
        RobotCapability(
            capability_id="mixing",
            name="Sample Mixing",
            description="Automated mixing and incubation",
            parameters={
                "mixing_speeds": [100, 300, 600, 1000],  # rpm
                "temperature_range": [4, 95],  # Celsius
            },
            estimated_duration=60.0,
            resource_requirements={"shaker": 1},
        ),
    ]

    return robot_id, protocol, capabilities


def create_mqtt_synthesis_robot(
    broker_host: str, broker_port: int = 1883
) -> Tuple[str, CommunicationProtocol, List[RobotCapability]]:
    """Create synthesis robot with MQTT protocol."""
    robot_id = "synthesis_platform_01"
    protocol = MQTTRobotProtocol(broker_host, broker_port)

    capabilities = [
        RobotCapability(
            capability_id="chemical_synthesis",
            name="Chemical Synthesis",
            description="Automated chemical reaction setup and execution",
            parameters={
                "reaction_types": [
                    "suzuki_coupling",
                    "click_chemistry",
                    "amide_formation",
                ],
                "temperature_range": [-78, 200],  # Celsius
                "pressure_range": [0.1, 10.0],  # bar
            },
            constraints={
                "max_reaction_volume": 50.0,  # mL
                "max_concurrent_reactions": 12,
            },
            estimated_duration=7200.0,  # 2 hours
            resource_requirements={"reaction_vessel": 1, "heating_block": 1},
        ),
        RobotCapability(
            capability_id="purification",
            name="Automated Purification",
            description="Column chromatography and purification",
            parameters={
                "column_types": ["silica", "C18", "size_exclusion"],
                "flow_rates": [0.5, 1.0, 2.0, 5.0],  # mL/min
            },
            estimated_duration=1800.0,  # 30 minutes
            resource_requirements={"column": 1, "pump": 1},
        ),
        RobotCapability(
            capability_id="analysis",
            name="In-line Analysis",
            description="Real-time reaction monitoring and analysis",
            parameters={
                "techniques": ["HPLC", "GC-MS", "NMR", "UV-Vis"],
                "sampling_interval": [1, 5, 10, 30],  # minutes
            },
            estimated_duration=600.0,  # 10 minutes
            resource_requirements={"analyzer": 1},
        ),
    ]

    return robot_id, protocol, capabilities
