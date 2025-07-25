"""
Tests for Robotic Control and Integration System

This module contains comprehensive tests for the robotic control system including
communication protocols, robot management, and experimental plan execution.
"""

import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from chemi_accel.autonomous.robotic_control import (
    CommunicationProtocol,
    ExperimentalAction,
    ExperimentPlan,
    ExperimentStatus,
    HTTPRobotProtocol,
    MQTTRobotProtocol,
    OptimizationObjective,
    RobotCapability,
    RobotFeedback,
    RobotManager,
    RobotStatus,
    create_http_hamilton_robot,
    create_mqtt_synthesis_robot,
)


class TestRobotCapability:
    """Test cases for RobotCapability dataclass."""

    def test_robot_capability_creation(self):
        """Test basic robot capability creation."""
        capability = RobotCapability(
            capability_id="liquid_handling",
            name="Liquid Handling",
            description="Precision liquid dispensing",
            parameters={"min_volume": 0.1, "max_volume": 1000.0},
            constraints={"max_channels": 8},
            estimated_duration=30.0,
            resource_requirements={"tips": 1},
        )

        assert capability.capability_id == "liquid_handling"
        assert capability.name == "Liquid Handling"
        assert capability.parameters["min_volume"] == 0.1
        assert capability.constraints["max_channels"] == 8
        assert capability.estimated_duration == 30.0
        assert capability.resource_requirements["tips"] == 1


class TestExperimentalAction:
    """Test cases for ExperimentalAction dataclass."""

    def test_experimental_action_creation(self):
        """Test basic experimental action creation."""
        action = ExperimentalAction(
            action_id="test_action_001",
            robot_id="hamilton_01",
            capability_id="liquid_handling",
            parameters={"volume": 100.0, "speed": "medium"},
            priority=3,
            dependencies=["prep_action"],
            expected_duration=120.0,
        )

        assert action.action_id == "test_action_001"
        assert action.robot_id == "hamilton_01"
        assert action.capability_id == "liquid_handling"
        assert action.parameters["volume"] == 100.0
        assert action.priority == 3
        assert "prep_action" in action.dependencies
        assert action.expected_duration == 120.0
        assert action.created_at is not None


class TestExperimentPlan:
    """Test cases for ExperimentPlan dataclass."""

    def test_experiment_plan_creation(self):
        """Test experiment plan creation with actions."""
        action1 = ExperimentalAction(
            action_id="action_1",
            robot_id="robot_1",
            capability_id="synthesis",
            parameters={"temperature": 80},
        )

        action2 = ExperimentalAction(
            action_id="action_2",
            robot_id="robot_2",
            capability_id="analysis",
            parameters={"technique": "NMR"},
            dependencies=["action_1"],
        )

        plan = ExperimentPlan(
            plan_id="test_plan_001",
            experiment_id="exp_001",
            actions=[action1, action2],
            optimization_objective=OptimizationObjective.MAXIMIZE_YIELD,
            target_parameters={"yield": 0.8},
            constraints={"max_time": 24},
        )

        assert plan.plan_id == "test_plan_001"
        assert plan.experiment_id == "exp_001"
        assert len(plan.actions) == 2
        assert plan.optimization_objective == OptimizationObjective.MAXIMIZE_YIELD
        assert plan.target_parameters["yield"] == 0.8
        assert plan.constraints["max_time"] == 24


class TestRobotFeedback:
    """Test cases for RobotFeedback dataclass."""

    def test_robot_feedback_creation(self):
        """Test robot feedback creation."""
        feedback = RobotFeedback(
            feedback_id="feedback_001",
            robot_id="hamilton_01",
            action_id="action_001",
            status=ExperimentStatus.RUNNING,
            progress=0.5,
            data={"temperature": 85.2, "volume_dispensed": 250.0},
            error_message=None,
        )

        assert feedback.feedback_id == "feedback_001"
        assert feedback.robot_id == "hamilton_01"
        assert feedback.action_id == "action_001"
        assert feedback.status == ExperimentStatus.RUNNING
        assert feedback.progress == 0.5
        assert feedback.data["temperature"] == 85.2
        assert feedback.error_message is None
        assert feedback.timestamp is not None


class TestHTTPRobotProtocol:
    """Test cases for HTTP Robot Protocol."""

    @pytest.fixture
    def http_protocol(self):
        """Create HTTP protocol instance."""
        return HTTPRobotProtocol("http://localhost:8080", "test_token")

    @patch("chemi_accel.autonomous.robotic_control.requests.Session")
    def test_http_protocol_creation(self, mock_session, http_protocol):
        """Test HTTP protocol initialization."""
        assert http_protocol.base_url == "http://localhost:8080"
        assert http_protocol.auth_token == "test_token"
        assert not http_protocol._is_connected

    @patch("chemi_accel.autonomous.robotic_control.requests.Session")
    @pytest.mark.asyncio
    async def test_http_connect_success(self, mock_session_class, http_protocol):
        """Test successful HTTP connection."""
        # Mock session and response
        mock_session = Mock()
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        # Mock the session attribute
        http_protocol.session = mock_session

        success = await http_protocol.connect()

        assert success is True
        assert http_protocol._is_connected is True
        mock_session.get.assert_called_once()

    @patch("chemi_accel.autonomous.robotic_control.requests.Session")
    @pytest.mark.asyncio
    async def test_http_connect_failure(self, mock_session_class, http_protocol):
        """Test failed HTTP connection."""
        # Mock session that raises exception
        mock_session = Mock()
        mock_session.get.side_effect = Exception("Connection failed")
        mock_session_class.return_value = mock_session
        http_protocol.session = mock_session

        success = await http_protocol.connect()

        assert success is False
        assert http_protocol._is_connected is False

    @patch("chemi_accel.autonomous.robotic_control.requests.Session")
    @pytest.mark.asyncio
    async def test_send_command(self, mock_session_class, http_protocol):
        """Test sending command via HTTP."""
        # Setup mocks
        mock_session = Mock()
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"status": "success", "command_id": "cmd_001"}
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session

        http_protocol.session = mock_session
        http_protocol._is_connected = True

        command = {"action": "dispense", "volume": 100}
        response = await http_protocol.send_command("robot_01", command)

        assert response["status"] == "success"
        assert response["command_id"] == "cmd_001"
        mock_session.post.assert_called_once()

    @patch("chemi_accel.autonomous.robotic_control.requests.Session")
    @pytest.mark.asyncio
    async def test_get_robot_status(self, mock_session_class, http_protocol):
        """Test getting robot status."""
        # Setup mocks
        mock_session = Mock()
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"status": "idle"}
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        http_protocol.session = mock_session
        http_protocol._is_connected = True

        status = await http_protocol.get_robot_status("robot_01")

        assert status == RobotStatus.IDLE
        mock_session.get.assert_called_once()

    def test_set_feedback_callback(self, http_protocol):
        """Test setting feedback callback."""
        callback = Mock()
        http_protocol.set_feedback_callback(callback)

        assert http_protocol.feedback_callback == callback


class TestMQTTRobotProtocol:
    """Test cases for MQTT Robot Protocol."""

    @pytest.fixture
    def mqtt_protocol(self):
        """Create MQTT protocol instance."""
        with patch("chemi_accel.autonomous.robotic_control.mqtt.Client"):
            return MQTTRobotProtocol("localhost", 1883, "user", "pass")

    def test_mqtt_protocol_creation(self, mqtt_protocol):
        """Test MQTT protocol initialization."""
        assert mqtt_protocol.broker_host == "localhost"
        assert mqtt_protocol.broker_port == 1883
        assert mqtt_protocol.username == "user"
        assert mqtt_protocol.password == "pass"
        assert not mqtt_protocol._is_connected

    @pytest.mark.asyncio
    async def test_mqtt_connect_success(self, mqtt_protocol):
        """Test successful MQTT connection."""
        # Mock successful connection
        mqtt_protocol._is_connected = True

        success = await mqtt_protocol.connect()

        # Since we mocked _is_connected to True, it should succeed
        assert success is True

    @pytest.mark.asyncio
    async def test_mqtt_send_command(self, mqtt_protocol):
        """Test sending command via MQTT."""
        # Mock client
        mock_client = Mock()
        mock_result = Mock()
        mock_result.rc = 0
        mock_result.wait_for_publish = Mock()
        mock_client.publish.return_value = mock_result

        mqtt_protocol.client = mock_client
        mqtt_protocol._is_connected = True

        command = {"action": "start_synthesis", "parameters": {"temp": 80}}
        response = await mqtt_protocol.send_command("synthesis_01", command)

        assert response["status"] == "sent"
        mock_client.publish.assert_called_once()

    def test_mqtt_on_message(self, mqtt_protocol):
        """Test MQTT message handling."""
        # Setup feedback callback
        callback = Mock()
        mqtt_protocol.set_feedback_callback(callback)

        # Create mock message
        mock_msg = Mock()
        mock_msg.topic = "robots/test_robot/feedback"
        mock_msg.payload.decode.return_value = json.dumps(
            {
                "feedback_id": "fb_001",
                "action_id": "action_001",
                "status": "completed",
                "progress": 1.0,
                "data": {"result": "success"},
            }
        )

        # Call message handler
        mqtt_protocol._on_message(None, None, mock_msg)

        # Verify callback was called
        callback.assert_called_once()
        feedback = callback.call_args[0][0]
        assert feedback.feedback_id == "fb_001"
        assert feedback.status == ExperimentStatus.COMPLETED


class TestRobotManager:
    """Test cases for RobotManager."""

    @pytest.fixture
    def robot_manager(self):
        """Create robot manager instance."""
        return RobotManager()

    @pytest.fixture
    def mock_protocol(self):
        """Create mock protocol."""
        protocol = Mock(spec=CommunicationProtocol)
        protocol.connect = AsyncMock(return_value=True)
        protocol.disconnect = AsyncMock()
        protocol.send_command = AsyncMock(return_value={"status": "success"})
        protocol.get_robot_status = AsyncMock(return_value=RobotStatus.IDLE)
        protocol.set_feedback_callback = Mock()
        return protocol

    @pytest.fixture
    def sample_capabilities(self):
        """Create sample robot capabilities."""
        return [
            RobotCapability(
                capability_id="liquid_handling",
                name="Liquid Handling",
                description="Precision pipetting",
                parameters={"min_volume": 0.1, "max_volume": 1000},
            ),
            RobotCapability(
                capability_id="heating",
                name="Temperature Control",
                description="Heating and cooling",
                parameters={"min_temp": -10, "max_temp": 200},
            ),
        ]

    def test_register_robot(self, robot_manager, mock_protocol, sample_capabilities):
        """Test robot registration."""
        robot_id = "hamilton_01"
        metadata = {"model": "Hamilton Microlab STAR", "location": "Lab A"}

        robot_manager.register_robot(
            robot_id, mock_protocol, sample_capabilities, metadata
        )

        assert robot_id in robot_manager.robots
        assert robot_id in robot_manager.protocols
        assert robot_id in robot_manager.capabilities
        assert robot_manager.robots[robot_id]["metadata"] == metadata
        assert len(robot_manager.capabilities[robot_id]) == 2
        mock_protocol.set_feedback_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_all_robots(
        self, robot_manager, mock_protocol, sample_capabilities
    ):
        """Test connecting to all registered robots."""
        robot_manager.register_robot("robot_01", mock_protocol, sample_capabilities)

        results = await robot_manager.connect_all_robots()

        assert "robot_01" in results
        assert results["robot_01"] is True
        mock_protocol.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_experiment_plan(
        self, robot_manager, mock_protocol, sample_capabilities
    ):
        """Test experiment plan execution."""
        robot_manager.register_robot("robot_01", mock_protocol, sample_capabilities)

        # Create test plan
        action = ExperimentalAction(
            action_id="test_action",
            robot_id="robot_01",
            capability_id="liquid_handling",
            parameters={"volume": 100},
        )

        plan = ExperimentPlan(
            plan_id="test_plan",
            experiment_id="test_exp",
            actions=[action],
            optimization_objective=OptimizationObjective.MAXIMIZE_YIELD,
        )

        # Execute plan
        plan_id = await robot_manager.execute_experiment_plan(plan)

        assert plan_id == "test_plan"
        assert plan.plan_id in robot_manager.active_experiments
        mock_protocol.send_command.assert_called()

    def test_find_robots_with_capability(
        self, robot_manager, mock_protocol, sample_capabilities
    ):
        """Test finding robots with specific capabilities."""
        robot_manager.register_robot("robot_01", mock_protocol, sample_capabilities)
        robot_manager.register_robot(
            "robot_02", mock_protocol, sample_capabilities[:1]
        )  # Only liquid handling

        # Find robots with liquid handling
        liquid_robots = robot_manager.find_robots_with_capability("liquid_handling")
        assert "robot_01" in liquid_robots
        assert "robot_02" in liquid_robots

        # Find robots with heating
        heating_robots = robot_manager.find_robots_with_capability("heating")
        assert "robot_01" in heating_robots
        assert "robot_02" not in heating_robots

    def test_robot_feedback_handling(
        self, robot_manager, mock_protocol, sample_capabilities
    ):
        """Test robot feedback handling."""
        robot_manager.register_robot("robot_01", mock_protocol, sample_capabilities)

        # Create feedback
        feedback = RobotFeedback(
            feedback_id="fb_001",
            robot_id="robot_01",
            action_id="action_001",
            status=ExperimentStatus.COMPLETED,
            progress=1.0,
            data={"result": "success"},
        )

        # Add callback to capture feedback
        received_feedback = []
        robot_manager.add_feedback_callback(lambda fb: received_feedback.append(fb))

        # Simulate feedback
        robot_manager._handle_robot_feedback(feedback)

        assert len(received_feedback) == 1
        assert received_feedback[0].feedback_id == "fb_001"
        assert received_feedback[0].status == ExperimentStatus.COMPLETED

    def test_get_robot_summary(self, robot_manager, mock_protocol, sample_capabilities):
        """Test robot summary generation."""
        robot_manager.register_robot("robot_01", mock_protocol, sample_capabilities)

        summary = robot_manager.get_robot_summary()

        assert summary["total_robots"] == 1
        assert "robot_01" in summary["robots"]
        assert summary["robots"]["robot_01"]["capabilities_count"] == 2
        assert "last_seen" in summary["robots"]["robot_01"]


class TestFactoryFunctions:
    """Test cases for factory functions."""

    def test_create_http_hamilton_robot(self):
        """Test Hamilton robot creation with HTTP protocol."""
        robot_id, protocol, capabilities = create_http_hamilton_robot(
            "http://hamilton.local:8080", "auth_token_123"
        )

        assert robot_id == "hamilton_microlab_star"
        assert isinstance(protocol, HTTPRobotProtocol)
        assert len(capabilities) == 3  # liquid_handling, plate_handling, mixing

        # Check capabilities
        capability_ids = [cap.capability_id for cap in capabilities]
        assert "liquid_handling" in capability_ids
        assert "plate_handling" in capability_ids
        assert "mixing" in capability_ids

    def test_create_mqtt_synthesis_robot(self):
        """Test synthesis robot creation with MQTT protocol."""
        with patch("chemi_accel.autonomous.robotic_control.mqtt.Client"):
            robot_id, protocol, capabilities = create_mqtt_synthesis_robot(
                "mqtt.lab.local", 1883
            )

        assert robot_id == "synthesis_platform_01"
        assert isinstance(protocol, MQTTRobotProtocol)
        assert len(capabilities) == 3  # chemical_synthesis, purification, analysis

        # Check capabilities
        capability_ids = [cap.capability_id for cap in capabilities]
        assert "chemical_synthesis" in capability_ids
        assert "purification" in capability_ids
        assert "analysis" in capability_ids


class TestIntegrationScenarios:
    """Integration test scenarios for robotic control system."""

    @pytest.mark.asyncio
    async def test_end_to_end_experiment_execution(self):
        """Test complete experiment execution workflow."""
        # Create robot manager
        manager = RobotManager()

        # Create mock protocols
        synthesis_protocol = Mock(spec=CommunicationProtocol)
        synthesis_protocol.connect = AsyncMock(return_value=True)
        synthesis_protocol.send_command = AsyncMock(
            return_value={"status": "started", "estimated_time": 7200}
        )
        synthesis_protocol.get_robot_status = AsyncMock(return_value=RobotStatus.IDLE)
        synthesis_protocol.set_feedback_callback = Mock()

        analysis_protocol = Mock(spec=CommunicationProtocol)
        analysis_protocol.connect = AsyncMock(return_value=True)
        analysis_protocol.send_command = AsyncMock(
            return_value={"status": "analyzing", "estimated_time": 600}
        )
        analysis_protocol.get_robot_status = AsyncMock(return_value=RobotStatus.IDLE)
        analysis_protocol.set_feedback_callback = Mock()

        # Register robots
        synthesis_capabilities = [
            RobotCapability(
                "chemical_synthesis",
                "Synthesis",
                "Chemical reactions",
                {"max_temp": 200},
            )
        ]
        analysis_capabilities = [
            RobotCapability(
                "analysis",
                "Analysis",
                "Analytical techniques",
                {"techniques": ["NMR", "HPLC"]},
            )
        ]

        manager.register_robot(
            "synthesis_01", synthesis_protocol, synthesis_capabilities
        )
        manager.register_robot("analysis_01", analysis_protocol, analysis_capabilities)

        # Connect robots
        connection_results = await manager.connect_all_robots()
        assert all(connection_results.values())

        # Create experiment plan
        synthesis_action = ExperimentalAction(
            action_id="synthesis_step",
            robot_id="synthesis_01",
            capability_id="chemical_synthesis",
            parameters={
                "reaction_type": "suzuki_coupling",
                "temperature": 80,
                "time": 4,
            },
            expected_duration=14400,
        )

        analysis_action = ExperimentalAction(
            action_id="analysis_step",
            robot_id="analysis_01",
            capability_id="analysis",
            parameters={"technique": "NMR", "sample_id": "synthesis_product"},
            dependencies=["synthesis_step"],
            expected_duration=600,
        )

        plan = ExperimentPlan(
            plan_id="integration_test_plan",
            experiment_id="integration_test_exp",
            actions=[synthesis_action, analysis_action],
            optimization_objective=OptimizationObjective.MAXIMIZE_YIELD,
            target_parameters={"yield": 0.75, "purity": 0.90},
        )

        # Execute plan
        executed_plan_id = await manager.execute_experiment_plan(plan)

        assert executed_plan_id == "integration_test_plan"
        assert plan.plan_id in manager.active_experiments

        # Verify commands were sent to both robots
        synthesis_protocol.send_command.assert_called()
        analysis_protocol.send_command.assert_called()

    @pytest.mark.asyncio
    async def test_robot_failure_handling(self):
        """Test handling of robot failures during experiment execution."""
        manager = RobotManager()

        # Create protocol that fails
        failing_protocol = Mock(spec=CommunicationProtocol)
        failing_protocol.connect = AsyncMock(return_value=False)  # Connection fails
        failing_protocol.send_command = AsyncMock(
            side_effect=Exception("Robot communication error")
        )
        failing_protocol.get_robot_status = AsyncMock(return_value=RobotStatus.ERROR)
        failing_protocol.set_feedback_callback = Mock()

        capabilities = [RobotCapability("test_capability", "Test", "Test capability")]
        manager.register_robot("failing_robot", failing_protocol, capabilities)

        # Try to connect
        connection_results = await manager.connect_all_robots()
        assert connection_results["failing_robot"] is False

        # Try to execute experiment (should handle gracefully)
        action = ExperimentalAction(
            action_id="test_action",
            robot_id="failing_robot",
            capability_id="test_capability",
            parameters={},
        )

        plan = ExperimentPlan(
            plan_id="failing_plan",
            experiment_id="failing_exp",
            actions=[action],
            optimization_objective=OptimizationObjective.MAXIMIZE_YIELD,
        )

        # This should not raise an exception but handle the error gracefully
        try:
            await manager.execute_experiment_plan(plan)
        except Exception as e:
            # Some exception handling is expected for communication failures
            assert "Robot communication error" in str(e) or "Robot" in str(e)

    @pytest.mark.asyncio
    async def test_concurrent_robot_operations(self):
        """Test concurrent operations on multiple robots."""
        manager = RobotManager()

        # Create multiple mock protocols
        protocols = []
        for i in range(3):
            protocol = Mock(spec=CommunicationProtocol)
            protocol.connect = AsyncMock(return_value=True)
            protocol.send_command = AsyncMock(
                return_value={"status": f"robot_{i}_working"}
            )
            protocol.get_robot_status = AsyncMock(return_value=RobotStatus.IDLE)
            protocol.set_feedback_callback = Mock()
            protocols.append(protocol)

        # Register robots
        capabilities = [
            RobotCapability("parallel_work", "Parallel", "Concurrent operations")
        ]
        for i, protocol in enumerate(protocols):
            manager.register_robot(f"robot_{i}", protocol, capabilities)

        # Connect all robots
        await manager.connect_all_robots()

        # Create plans for concurrent execution
        plans = []
        for i in range(3):
            action = ExperimentalAction(
                action_id=f"concurrent_action_{i}",
                robot_id=f"robot_{i}",
                capability_id="parallel_work",
                parameters={"task_id": i},
            )

            plan = ExperimentPlan(
                plan_id=f"concurrent_plan_{i}",
                experiment_id=f"concurrent_exp_{i}",
                actions=[action],
                optimization_objective=OptimizationObjective.MINIMIZE_TIME,
            )
            plans.append(plan)

        # Execute all plans concurrently
        execution_tasks = [manager.execute_experiment_plan(plan) for plan in plans]

        executed_plan_ids = await asyncio.gather(*execution_tasks)

        # Verify all plans executed
        assert len(executed_plan_ids) == 3
        for i, plan_id in enumerate(executed_plan_ids):
            assert plan_id == f"concurrent_plan_{i}"

        # Verify all protocols received commands
        for protocol in protocols:
            protocol.send_command.assert_called()

    @pytest.mark.asyncio
    async def test_real_time_feedback_processing(self):
        """Test real-time feedback processing from robots."""
        manager = RobotManager()

        # Create protocol with feedback simulation
        protocol = Mock(spec=CommunicationProtocol)
        protocol.connect = AsyncMock(return_value=True)
        protocol.send_command = AsyncMock(return_value={"status": "started"})
        protocol.get_robot_status = AsyncMock(return_value=RobotStatus.BUSY)
        protocol.set_feedback_callback = Mock()

        capabilities = [
            RobotCapability("feedback_test", "Feedback Test", "Test feedback")
        ]
        manager.register_robot("feedback_robot", protocol, capabilities)

        # Capture feedback
        received_feedbacks = []
        manager.add_feedback_callback(lambda fb: received_feedbacks.append(fb))

        # Simulate feedback sequence
        feedbacks = [
            RobotFeedback(
                "fb_1",
                "feedback_robot",
                "action_1",
                ExperimentStatus.RUNNING,
                0.25,
                {"step": "preparation"},
            ),
            RobotFeedback(
                "fb_2",
                "feedback_robot",
                "action_1",
                ExperimentStatus.RUNNING,
                0.50,
                {"step": "reaction"},
            ),
            RobotFeedback(
                "fb_3",
                "feedback_robot",
                "action_1",
                ExperimentStatus.RUNNING,
                0.75,
                {"step": "workup"},
            ),
            RobotFeedback(
                "fb_4",
                "feedback_robot",
                "action_1",
                ExperimentStatus.COMPLETED,
                1.0,
                {"step": "finished", "yield": 0.82},
            ),
        ]

        # Send feedback
        for feedback in feedbacks:
            manager._handle_robot_feedback(feedback)

        # Verify feedback was processed
        assert len(received_feedbacks) == 4
        assert received_feedbacks[0].progress == 0.25
        assert received_feedbacks[-1].status == ExperimentStatus.COMPLETED
        assert received_feedbacks[-1].data["yield"] == 0.82

    @pytest.mark.asyncio
    async def test_optimization_objective_routing(self):
        """Test routing experiments based on optimization objectives."""
        manager = RobotManager()

        # Create specialized robots
        synthesis_protocol = Mock(spec=CommunicationProtocol)
        synthesis_protocol.connect = AsyncMock(return_value=True)
        synthesis_protocol.send_command = AsyncMock(
            return_value={"status": "synthesis_started"}
        )
        synthesis_protocol.get_robot_status = AsyncMock(return_value=RobotStatus.IDLE)
        synthesis_protocol.set_feedback_callback = Mock()

        purification_protocol = Mock(spec=CommunicationProtocol)
        purification_protocol.connect = AsyncMock(return_value=True)
        purification_protocol.send_command = AsyncMock(
            return_value={"status": "purification_started"}
        )
        purification_protocol.get_robot_status = AsyncMock(
            return_value=RobotStatus.IDLE
        )
        purification_protocol.set_feedback_callback = Mock()

        # Register robots with different capabilities
        synthesis_capabilities = [
            RobotCapability("synthesis", "Synthesis", "Chemical synthesis")
        ]
        purification_capabilities = [
            RobotCapability("purification", "Purification", "Product purification")
        ]

        manager.register_robot(
            "synthesis_robot", synthesis_protocol, synthesis_capabilities
        )
        manager.register_robot(
            "purification_robot", purification_protocol, purification_capabilities
        )

        await manager.connect_all_robots()

        # Test different optimization objectives
        objectives_and_actions = [
            (OptimizationObjective.MAXIMIZE_YIELD, "synthesis"),
            (OptimizationObjective.MAXIMIZE_PURITY, "purification"),
            (OptimizationObjective.MINIMIZE_TIME, "synthesis"),  # Fast synthesis
        ]

        for objective, expected_capability in objectives_and_actions:
            # Find appropriate robot
            suitable_robots = manager.find_robots_with_capability(expected_capability)
            assert len(suitable_robots) > 0

            # Create plan with specific objective
            action = ExperimentalAction(
                action_id=f"obj_test_{objective.value}",
                robot_id=suitable_robots[0],
                capability_id=expected_capability,
                parameters={"objective": objective.value},
            )

            plan = ExperimentPlan(
                plan_id=f"obj_plan_{objective.value}",
                experiment_id=f"obj_exp_{objective.value}",
                actions=[action],
                optimization_objective=objective,
            )

            plan_id = await manager.execute_experiment_plan(plan)
            assert plan_id == f"obj_plan_{objective.value}"
