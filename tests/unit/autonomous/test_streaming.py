"""
Tests for Real-time Data Streaming System

This module contains comprehensive tests for the streaming system including
event processing, WebSocket communication, and stream management.
"""

import pytest
import asyncio
import json
import websockets
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from chemi_accel.autonomous.streaming import (
    StreamEvent,
    StreamSubscription,
    DataValidationProcessor,
    AggregationProcessor,
    AlertProcessor,
    WebSocketStreamer,
    StreamingManager,
    create_default_streaming_system
)
from chemi_accel.autonomous.data_ingestion import (
    ExperimentalDataPoint,
    DataIngestionManager
)
from chemi_accel.autonomous.data_validation import (
    DataValidator,
    ValidationResult
)


class TestStreamEvent:
    """Test cases for StreamEvent dataclass."""
    
    def test_stream_event_creation(self):
        """Test basic stream event creation."""
        timestamp = datetime.now(timezone.utc)
        event = StreamEvent(
            event_id="event_123",
            event_type="experimental_data",
            timestamp=timestamp,
            data={"instrument_id": "nmr_400", "value": 123.45},
            source="ingestion_nmr_400",
            priority=2
        )
        
        assert event.event_id == "event_123"
        assert event.event_type == "experimental_data"
        assert event.timestamp == timestamp
        assert event.data["instrument_id"] == "nmr_400"
        assert event.source == "ingestion_nmr_400"
        assert event.priority == 2
    
    def test_stream_event_default_priority(self):
        """Test stream event with default priority."""
        event = StreamEvent(
            event_id="event_456",
            event_type="alert",
            timestamp=datetime.now(timezone.utc),
            data={"message": "test alert"},
            source="alert_processor"
        )
        
        assert event.priority == 1  # Default priority


class TestStreamSubscription:
    """Test cases for StreamSubscription dataclass."""
    
    def test_subscription_creation(self):
        """Test stream subscription creation."""
        subscription = StreamSubscription(
            subscription_id="sub_123",
            client_id="client_456",
            event_types=["experimental_data", "validation_result"],
            filters={"instrument_id": "nmr_400", "priority": 2},
            is_active=True
        )
        
        assert subscription.subscription_id == "sub_123"
        assert subscription.client_id == "client_456"
        assert len(subscription.event_types) == 2
        assert subscription.filters["instrument_id"] == "nmr_400"
        assert subscription.is_active is True
        assert subscription.created_at is not None
    
    def test_subscription_defaults(self):
        """Test subscription with default values."""
        subscription = StreamSubscription(
            subscription_id="sub_789",
            client_id="client_123",
            event_types=["alert"]
        )
        
        assert len(subscription.filters) == 0
        assert subscription.is_active is True
        assert subscription.created_at is not None


class TestDataValidationProcessor:
    """Test cases for DataValidationProcessor."""
    
    @pytest.fixture
    def mock_validator(self):
        """Create mock validator."""
        validator = Mock(spec=DataValidator)
        validator.validate_data_point = AsyncMock(return_value=ValidationResult(
            is_valid=True,
            confidence_score=0.9,
            errors=[],
            warnings=["Minor warning"],
            quality_metrics={"quality_score": 0.85}
        ))
        return validator
    
    @pytest.fixture
    def validation_processor(self, mock_validator) -> DataValidationProcessor:
        """Create validation processor."""
        return DataValidationProcessor(mock_validator)
    
    @pytest.fixture
    def sample_experimental_event(self) -> StreamEvent:
        """Create sample experimental data event."""
        return StreamEvent(
            event_id="exp_event_123",
            event_type="experimental_data",
            timestamp=datetime.now(timezone.utc),
            data={
                "data_id": "data_123",
                "experiment_id": "exp_456",
                "instrument_id": "nmr_400",
                "data_type": "nmr",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "raw_data": {
                    "chemical_shifts": [7.5, 7.2, 3.8],
                    "intensities": [100, 80, 120]
                },
                "metadata": {"solvent": "CDCl3"}
            },
            source="ingestion_nmr_400",
            priority=2
        )
    
    @pytest.mark.asyncio
    async def test_process_experimental_data_event(self, validation_processor, sample_experimental_event):
        """Test processing of experimental data event."""
        result_event = await validation_processor.process_event(sample_experimental_event)
        
        assert result_event is not None
        assert result_event.event_type == "data_validation_result"
        assert result_event.source == "data_validation_processor"
        assert result_event.data["original_event_id"] == "exp_event_123"
        assert result_event.data["is_valid"] is True
        assert result_event.data["confidence_score"] == 0.9
        assert len(result_event.data["warnings"]) == 1
        assert result_event.priority == 1  # Valid data has low priority
    
    @pytest.mark.asyncio
    async def test_process_non_experimental_event(self, validation_processor):
        """Test processing of non-experimental data event."""
        alert_event = StreamEvent(
            event_id="alert_123",
            event_type="alert",
            timestamp=datetime.now(timezone.utc),
            data={"message": "Test alert"},
            source="alert_processor"
        )
        
        result_event = await validation_processor.process_event(alert_event)
        
        # Should return original event unchanged
        assert result_event == alert_event
    
    @pytest.mark.asyncio
    async def test_process_invalid_data_event(self, validation_processor, sample_experimental_event):
        """Test processing of invalid experimental data."""
        # Configure validator to return invalid result
        validation_processor.validator.validate_data_point.return_value = ValidationResult(
            is_valid=False,
            confidence_score=0.3,
            errors=["Invalid data format", "Missing required field"],
            warnings=[],
            quality_metrics={}
        )
        
        result_event = await validation_processor.process_event(sample_experimental_event)
        
        assert result_event.data["is_valid"] is False
        assert result_event.data["confidence_score"] == 0.3
        assert len(result_event.data["errors"]) == 2
        assert result_event.priority == 3  # Invalid data has high priority
    
    @pytest.mark.asyncio
    async def test_process_event_with_error(self, validation_processor, sample_experimental_event):
        """Test processing event that causes validation error."""
        # Configure validator to raise exception
        validation_processor.validator.validate_data_point.side_effect = Exception("Validation failed")
        
        result_event = await validation_processor.process_event(sample_experimental_event)
        
        assert result_event.event_type == "processing_error"
        assert result_event.source == "data_validation_processor"
        assert result_event.priority == 4  # Critical priority for errors
        assert "Validation failed" in result_event.data["error"]
    
    def test_validation_statistics(self, validation_processor):
        """Test validation statistics tracking."""
        initial_stats = validation_processor.validation_stats.copy()
        
        assert initial_stats["total_processed"] == 0
        assert initial_stats["valid_count"] == 0
        assert initial_stats["invalid_count"] == 0
        assert initial_stats["avg_confidence"] == 0.0
    
    def test_get_processor_info(self, validation_processor):
        """Test processor information retrieval."""
        info = validation_processor.get_processor_info()
        
        assert info["processor_type"] == "data_validation"
        assert "Validates experimental data quality" in info["description"]
        assert "statistics" in info


class TestAggregationProcessor:
    """Test cases for AggregationProcessor."""
    
    @pytest.fixture
    def aggregation_processor(self) -> AggregationProcessor:
        """Create aggregation processor with short window for testing."""
        return AggregationProcessor(window_size_seconds=0.1, aggregation_functions=["count", "mean"])
    
    @pytest.fixture
    def sample_events(self) -> List[StreamEvent]:
        """Create sample events for aggregation testing."""
        events = []
        base_time = datetime.now(timezone.utc)
        
        for i in range(5):
            event = StreamEvent(
                event_id=f"test_event_{i}",
                event_type="experimental_data",
                timestamp=base_time,
                data={"value": i * 10, "instrument_id": "test_instrument"},
                source="test_source",
                priority=1
            )
            events.append(event)
        
        return events
    
    @pytest.mark.asyncio
    async def test_event_buffering(self, aggregation_processor, sample_events):
        """Test event buffering before aggregation."""
        # Process events without triggering aggregation
        for event in sample_events[:3]:
            result = await aggregation_processor.process_event(event)
            assert result is None  # No aggregation yet
        
        # Check buffer size
        assert len(aggregation_processor.data_buffer) == 3
    
    @pytest.mark.asyncio
    async def test_aggregation_window_trigger(self, aggregation_processor, sample_events):
        """Test aggregation triggered by time window."""
        # Process events
        for event in sample_events:
            await aggregation_processor.process_event(event)
        
        # Wait for aggregation window to pass
        await asyncio.sleep(0.2)
        
        # Process one more event to trigger aggregation
        final_event = StreamEvent(
            event_id="trigger_event",
            event_type="experimental_data",
            timestamp=datetime.now(timezone.utc),
            data={"value": 100},
            source="test_source"
        )
        
        result = await aggregation_processor.process_event(final_event)
        
        assert result is not None
        assert result.event_type == "data_aggregation"
        assert result.source == "aggregation_processor"
        assert result.data["total_events"] == 5  # Previous events
        assert "summary" in result.data
    
    @pytest.mark.asyncio
    async def test_aggregation_by_source_and_type(self, aggregation_processor):
        """Test aggregation grouped by source and event type."""
        events = [
            StreamEvent("id1", "experimental_data", datetime.now(timezone.utc), {"value": 1}, "source1"),
            StreamEvent("id2", "experimental_data", datetime.now(timezone.utc), {"value": 2}, "source1"),
            StreamEvent("id3", "validation_result", datetime.now(timezone.utc), {"is_valid": True}, "source2"),
            StreamEvent("id4", "experimental_data", datetime.now(timezone.utc), {"value": 3}, "source2"),
        ]
        
        for event in events:
            await aggregation_processor.process_event(event)
        
        # Wait and trigger aggregation
        await asyncio.sleep(0.2)
        trigger_event = StreamEvent("trigger", "alert", datetime.now(timezone.utc), {}, "trigger_source")
        result = await aggregation_processor.process_event(trigger_event)
        
        assert result is not None
        summary = result.data["summary"]
        
        # Should have separate entries for each source-type combination
        assert "experimental_data_source1" in summary
        assert "experimental_data_source2" in summary
        assert "validation_result_source2" in summary
        
        # Check counts
        assert summary["experimental_data_source1"]["count"] == 2
        assert summary["experimental_data_source2"]["count"] == 1
        assert summary["validation_result_source2"]["count"] == 1
    
    def test_get_processor_info(self, aggregation_processor):
        """Test aggregation processor information."""
        info = aggregation_processor.get_processor_info()
        
        assert info["processor_type"] == "aggregation"
        assert info["window_size_seconds"] == 0.1
        assert "aggregation_functions" in info
        assert "buffer_size" in info


class TestAlertProcessor:
    """Test cases for AlertProcessor."""
    
    @pytest.fixture
    def alert_rules(self) -> List[Dict[str, Any]]:
        """Create sample alert rules."""
        return [
            {
                "rule_id": "validation_failure",
                "name": "Data Validation Failure",
                "event_types": ["data_validation_result"],
                "severity": "warning",
                "message": "Data validation failed",
                "conditions": [
                    {"field": "is_valid", "operator": "equals", "value": False}
                ]
            },
            {
                "rule_id": "high_temperature",
                "name": "High Temperature Alert",
                "event_types": ["experimental_data"],
                "severity": "critical",
                "message": "Temperature exceeds safe limits",
                "conditions": [
                    {"field": "raw_data.temperature", "operator": "greater_than", "value": 100.0}
                ]
            },
            {
                "rule_id": "instrument_error",
                "name": "Instrument Error",
                "sources": ["instrument_nmr_400"],
                "severity": "critical",
                "message": "Instrument malfunction detected",
                "conditions": [
                    {"field": "error_count", "operator": "greater_than", "value": 5}
                ]
            }
        ]
    
    @pytest.fixture
    def alert_processor(self, alert_rules) -> AlertProcessor:
        """Create alert processor."""
        return AlertProcessor(alert_rules)
    
    @pytest.mark.asyncio
    async def test_validation_failure_alert(self, alert_processor):
        """Test alert generation for validation failure."""
        validation_event = StreamEvent(
            event_id="validation_123",
            event_type="data_validation_result",
            timestamp=datetime.now(timezone.utc),
            data={
                "is_valid": False,
                "confidence_score": 0.3,
                "errors": ["Schema validation failed"],
                "data_id": "data_123"
            },
            source="validation_processor"
        )
        
        result = await alert_processor.process_event(validation_event)
        
        assert result is not None
        assert result.event_type == "alert"
        assert result.priority == 3  # Warning severity
        assert len(result.data["alerts"]) == 1
        
        alert = result.data["alerts"][0]
        assert alert["rule_id"] == "validation_failure"
        assert alert["severity"] == "warning"
        assert "Data validation failed" in alert["message"]
    
    @pytest.mark.asyncio
    async def test_temperature_alert(self, alert_processor):
        """Test alert generation for high temperature."""
        temp_event = StreamEvent(
            event_id="temp_123",
            event_type="experimental_data",
            timestamp=datetime.now(timezone.utc),
            data={
                "instrument_id": "oven_001",
                "raw_data": {"temperature": 150.0},
                "data_type": "temperature_reading"
            },
            source="instrument_oven_001"
        )
        
        result = await alert_processor.process_event(temp_event)
        
        assert result is not None
        assert result.event_type == "alert"
        assert result.priority == 4  # Critical severity
        
        alert = result.data["alerts"][0]
        assert alert["rule_id"] == "high_temperature"
        assert alert["severity"] == "critical"
        assert "Temperature exceeds safe limits" in alert["message"]
    
    @pytest.mark.asyncio
    async def test_no_alert_generation(self, alert_processor):
        """Test no alert generation for normal conditions."""
        normal_event = StreamEvent(
            event_id="normal_123",
            event_type="experimental_data",
            timestamp=datetime.now(timezone.utc),
            data={
                "instrument_id": "nmr_400",
                "raw_data": {"temperature": 25.0},
                "data_type": "nmr_spectrum"
            },
            source="instrument_nmr_400"
        )
        
        result = await alert_processor.process_event(normal_event)
        
        assert result is None  # No alerts should be generated
    
    @pytest.mark.asyncio
    async def test_multiple_alerts_single_event(self, alert_processor):
        """Test multiple alerts triggered by single event."""
        # Create event that matches multiple rules
        multi_alert_event = StreamEvent(
            event_id="multi_123",
            event_type="data_validation_result",
            timestamp=datetime.now(timezone.utc),
            data={
                "is_valid": False,  # Matches validation failure rule
                "temperature": 120.0,  # Would match temperature rule if it applied to this event type
                "confidence_score": 0.1
            },
            source="validation_processor"
        )
        
        result = await alert_processor.process_event(multi_alert_event)
        
        assert result is not None
        assert len(result.data["alerts"]) >= 1  # At least validation failure alert
    
    @pytest.mark.asyncio
    async def test_nested_field_evaluation(self, alert_processor):
        """Test evaluation of nested fields in conditions."""
        nested_event = StreamEvent(
            event_id="nested_123",
            event_type="experimental_data",
            timestamp=datetime.now(timezone.utc),
            data={
                "instrument_id": "reactor_001",
                "raw_data": {
                    "temperature": 150.0,  # Should trigger high_temperature alert
                    "pressure": 5.0
                },
                "metadata": {"operator": "test_user"}
            },
            source="instrument_reactor_001"
        )
        
        result = await alert_processor.process_event(nested_event)
        
        assert result is not None
        alert = result.data["alerts"][0]
        assert alert["rule_id"] == "high_temperature"
    
    def test_alert_history_tracking(self, alert_processor):
        """Test alert history tracking."""
        initial_count = len(alert_processor.alert_history)
        
        # Alert history should start empty
        assert initial_count == 0
    
    def test_get_processor_info(self, alert_processor):
        """Test alert processor information."""
        info = alert_processor.get_processor_info()
        
        assert info["processor_type"] == "alert"
        assert info["active_rules"] > 0
        assert "alert_history_size" in info


class TestWebSocketStreamer:
    """Test cases for WebSocketStreamer."""
    
    @pytest.fixture
    def websocket_streamer(self) -> WebSocketStreamer:
        """Create WebSocket streamer."""
        return WebSocketStreamer(host="localhost", port=8765)
    
    def test_streamer_initialization(self, websocket_streamer):
        """Test WebSocket streamer initialization."""
        assert websocket_streamer.host == "localhost"
        assert websocket_streamer.port == 8765
        assert len(websocket_streamer.clients) == 0
        assert len(websocket_streamer.subscriptions) == 0
        assert websocket_streamer.server is None
    
    @pytest.mark.asyncio
    async def test_subscription_handling(self, websocket_streamer):
        """Test subscription message handling."""
        # Mock WebSocket client
        mock_websocket = AsyncMock()
        client_id = "test_client_123"
        websocket_streamer.clients[client_id] = mock_websocket
        
        # Test subscription message
        subscription_message = json.dumps({
            "type": "subscribe",
            "subscription_id": "sub_123",
            "event_types": ["experimental_data", "validation_result"],
            "filters": {"instrument_id": "nmr_400"}
        })
        
        await websocket_streamer.handle_message(client_id, subscription_message)
        
        # Check subscription was created
        assert "sub_123" in websocket_streamer.subscriptions
        subscription = websocket_streamer.subscriptions["sub_123"]
        assert subscription.client_id == client_id
        assert "experimental_data" in subscription.event_types
        assert subscription.filters["instrument_id"] == "nmr_400"
        
        # Check confirmation was sent
        mock_websocket.send.assert_called_once()
        sent_message = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_message["type"] == "subscription_confirmed"
    
    @pytest.mark.asyncio
    async def test_unsubscription_handling(self, websocket_streamer):
        """Test unsubscription message handling."""
        # Setup existing subscription
        client_id = "test_client_456"
        subscription_id = "sub_456"
        mock_websocket = AsyncMock()
        websocket_streamer.clients[client_id] = mock_websocket
        websocket_streamer.subscriptions[subscription_id] = StreamSubscription(
            subscription_id=subscription_id,
            client_id=client_id,
            event_types=["alert"]
        )
        
        # Test unsubscription message
        unsubscription_message = json.dumps({
            "type": "unsubscribe",
            "subscription_id": subscription_id
        })
        
        await websocket_streamer.handle_message(client_id, unsubscription_message)
        
        # Check subscription was removed
        assert subscription_id not in websocket_streamer.subscriptions
        
        # Check confirmation was sent
        mock_websocket.send.assert_called_once()
        sent_message = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_message["type"] == "unsubscription_confirmed"
    
    @pytest.mark.asyncio
    async def test_ping_pong_handling(self, websocket_streamer):
        """Test ping-pong message handling."""
        client_id = "test_client_789"
        mock_websocket = AsyncMock()
        websocket_streamer.clients[client_id] = mock_websocket
        
        ping_message = json.dumps({"type": "ping"})
        
        await websocket_streamer.handle_message(client_id, ping_message)
        
        mock_websocket.send.assert_called_once()
        sent_message = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_message["type"] == "pong"
    
    @pytest.mark.asyncio
    async def test_event_broadcasting(self, websocket_streamer):
        """Test event broadcasting to subscribed clients."""
        # Setup client with subscription
        client_id = "broadcast_client"
        subscription_id = "broadcast_sub"
        mock_websocket = AsyncMock()
        
        websocket_streamer.clients[client_id] = mock_websocket
        websocket_streamer.subscriptions[subscription_id] = StreamSubscription(
            subscription_id=subscription_id,
            client_id=client_id,
            event_types=["experimental_data"],
            filters={"instrument_id": "nmr_400"}
        )
        
        # Create matching event
        test_event = StreamEvent(
            event_id="broadcast_event",
            event_type="experimental_data",
            timestamp=datetime.now(timezone.utc),
            data={"instrument_id": "nmr_400", "value": 123},
            source="test_source"
        )
        
        await websocket_streamer.broadcast_event(test_event)
        
        # Check event was sent to client
        mock_websocket.send.assert_called_once()
        sent_message = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_message["type"] == "event"
        assert sent_message["event"]["event_id"] == "broadcast_event"
        assert sent_message["event"]["data"]["instrument_id"] == "nmr_400"
    
    @pytest.mark.asyncio
    async def test_event_filtering(self, websocket_streamer):
        """Test event filtering based on subscription criteria."""
        # Setup client with filtered subscription
        client_id = "filtered_client"
        subscription_id = "filtered_sub"
        mock_websocket = AsyncMock()
        
        websocket_streamer.clients[client_id] = mock_websocket
        websocket_streamer.subscriptions[subscription_id] = StreamSubscription(
            subscription_id=subscription_id,
            client_id=client_id,
            event_types=["experimental_data"],
            filters={"instrument_id": "lcms_400"}  # Different instrument
        )
        
        # Create non-matching event
        test_event = StreamEvent(
            event_id="filtered_event",
            event_type="experimental_data",
            timestamp=datetime.now(timezone.utc),
            data={"instrument_id": "nmr_400", "value": 123},  # Wrong instrument
            source="test_source"
        )
        
        await websocket_streamer.broadcast_event(test_event)
        
        # Check event was NOT sent to client
        mock_websocket.send.assert_not_called()
    
    def test_subscription_matching(self, websocket_streamer):
        """Test subscription matching logic."""
        subscription = StreamSubscription(
            subscription_id="match_test",
            client_id="match_client",
            event_types=["experimental_data", "validation_result"],
            filters={"source": "instrument_nmr", "priority": 2}
        )
        
        # Test matching event
        matching_event = StreamEvent(
            event_id="match_1",
            event_type="experimental_data",
            timestamp=datetime.now(timezone.utc),
            data={"test": "data"},
            source="instrument_nmr",
            priority=3  # Higher priority should match
        )
        
        assert websocket_streamer._matches_subscription(matching_event, subscription) is True
        
        # Test non-matching event (wrong type)
        non_matching_event = StreamEvent(
            event_id="match_2",
            event_type="alert",  # Not in subscription
            timestamp=datetime.now(timezone.utc),
            data={"test": "data"},
            source="instrument_nmr",
            priority=3
        )
        
        assert websocket_streamer._matches_subscription(non_matching_event, subscription) is False
        
        # Test non-matching event (wrong source)
        non_matching_source = StreamEvent(
            event_id="match_3",
            event_type="experimental_data",
            timestamp=datetime.now(timezone.utc),
            data={"test": "data"},
            source="instrument_lcms",  # Wrong source
            priority=3
        )
        
        assert websocket_streamer._matches_subscription(non_matching_source, subscription) is False


class TestStreamingManager:
    """Test cases for StreamingManager."""
    
    @pytest.fixture
    def mock_data_ingestion_manager(self):
        """Create mock data ingestion manager."""
        manager = Mock(spec=DataIngestionManager)
        manager.set_data_callback = Mock()
        return manager
    
    @pytest.fixture
    def streaming_manager(self, mock_data_ingestion_manager) -> StreamingManager:
        """Create streaming manager."""
        return StreamingManager(mock_data_ingestion_manager)
    
    def test_initialization(self, streaming_manager, mock_data_ingestion_manager):
        """Test streaming manager initialization."""
        assert streaming_manager.data_ingestion_manager == mock_data_ingestion_manager
        assert len(streaming_manager.processors) == 0
        assert isinstance(streaming_manager.websocket_streamer, WebSocketStreamer)
        assert streaming_manager.is_running is False
        
        # Check callback was set
        mock_data_ingestion_manager.set_data_callback.assert_called_once()
    
    def test_add_processor(self, streaming_manager):
        """Test adding stream processors."""
        mock_validator = Mock()
        processor = DataValidationProcessor(mock_validator)
        
        streaming_manager.add_processor(processor)
        
        assert len(streaming_manager.processors) == 1
        assert streaming_manager.processors[0] == processor
    
    def test_enable_redis_streaming(self, streaming_manager):
        """Test enabling Redis streaming."""
        redis_url = "redis://test:6379"
        
        streaming_manager.enable_redis_streaming(redis_url)
        
        assert streaming_manager.redis_streamer is not None
        assert streaming_manager.redis_streamer.redis_url == redis_url
    
    @pytest.mark.asyncio
    async def test_handle_incoming_data(self, streaming_manager):
        """Test handling of incoming experimental data."""
        # Create sample data points
        data_points = [
            ExperimentalDataPoint(
                data_id="data_123",
                experiment_id="exp_456",
                instrument_id="nmr_400",
                timestamp=datetime.now(timezone.utc),
                data_type="nmr",
                raw_data={"chemical_shifts": [7.5, 7.2], "intensities": [100, 80]}
            ),
            ExperimentalDataPoint(
                data_id="data_456",
                experiment_id="exp_789",
                instrument_id="lcms_qtof",
                timestamp=datetime.now(timezone.utc),
                data_type="lcms",
                raw_data={"chromatogram": [{"time": 1.0, "intensity": 1000}]}
            )
        ]
        
        # Call the callback directly
        await streaming_manager._handle_incoming_data(data_points)
        
        # Check events were queued
        assert streaming_manager.event_queue.qsize() == 2
        
        # Verify event structure
        event = await streaming_manager.event_queue.get()
        assert event.event_type == "experimental_data"
        assert event.data["data_id"] == "data_123"
        assert event.source == "ingestion_nmr_400"
    
    def test_get_streaming_status(self, streaming_manager):
        """Test streaming status retrieval."""
        # Add some processors
        mock_validator = Mock()
        processor1 = DataValidationProcessor(mock_validator)
        processor2 = AggregationProcessor(window_size_seconds=60.0)
        
        streaming_manager.add_processor(processor1)
        streaming_manager.add_processor(processor2)
        
        status = streaming_manager.get_streaming_status()
        
        assert status["is_running"] is False
        assert status["processors_count"] == 2
        assert status["websocket_clients"] == 0
        assert status["websocket_subscriptions"] == 0
        assert status["redis_enabled"] is False
        assert status["queue_size"] == 0
        assert len(status["processors_info"]) == 2


class TestFactoryFunctions:
    """Test cases for factory functions."""
    
    def test_create_default_streaming_system(self):
        """Test creation of default streaming system."""
        mock_ingestion_manager = Mock(spec=DataIngestionManager)
        mock_ingestion_manager.set_data_callback = Mock()
        
        streaming_system = create_default_streaming_system(mock_ingestion_manager)
        
        assert isinstance(streaming_system, StreamingManager)
        assert len(streaming_system.processors) == 3  # Validation, Aggregation, Alert
        
        # Check processor types
        processor_types = [p.get_processor_info()["processor_type"] for p in streaming_system.processors]
        assert "data_validation" in processor_types
        assert "aggregation" in processor_types
        assert "alert" in processor_types


class TestIntegrationScenarios:
    """Integration test scenarios for streaming system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_event_processing(self):
        """Test complete end-to-end event processing."""
        # Setup components
        mock_ingestion_manager = Mock(spec=DataIngestionManager)
        mock_ingestion_manager.set_data_callback = Mock()
        
        streaming_manager = create_default_streaming_system(mock_ingestion_manager)
        
        # Create test data
        test_data_point = ExperimentalDataPoint(
            data_id="integration_test",
            experiment_id="exp_integration",
            instrument_id="nmr_400",
            timestamp=datetime.now(timezone.utc),
            data_type="nmr",
            raw_data={
                "chemical_shifts": [7.5, 7.2, 3.8] * 30,
                "intensities": [100, 80, 120] * 30
            }
        )
        
        # Process through system
        await streaming_manager._handle_incoming_data([test_data_point])
        
        # Verify event was queued
        assert streaming_manager.event_queue.qsize() == 1
        
        # Get and verify event
        event = await streaming_manager.event_queue.get()
        assert event.event_type == "experimental_data"
        assert event.data["instrument_id"] == "nmr_400"
        assert event.source == "ingestion_nmr_400"
    
    @pytest.mark.asyncio
    async def test_multi_processor_pipeline(self):
        """Test event processing through multiple processors."""
        # Create mock processors
        processor1 = Mock()
        processor1.process_event = AsyncMock(return_value=StreamEvent(
            event_id="processed_1",
            event_type="processed_data",
            timestamp=datetime.now(timezone.utc),
            data={"processed": True},
            source="processor_1"
        ))
        processor1.get_processor_info.return_value = {"processor_type": "test_processor_1"}
        
        processor2 = Mock()
        processor2.process_event = AsyncMock(return_value=StreamEvent(
            event_id="processed_2",
            event_type="final_result",
            timestamp=datetime.now(timezone.utc),
            data={"final": True},
            source="processor_2"
        ))
        processor2.get_processor_info.return_value = {"processor_type": "test_processor_2"}
        
        # Setup streaming manager
        mock_ingestion_manager = Mock(spec=DataIngestionManager)
        mock_ingestion_manager.set_data_callback = Mock()
        streaming_manager = StreamingManager(mock_ingestion_manager)
        
        streaming_manager.add_processor(processor1)
        streaming_manager.add_processor(processor2)
        
        # Create initial event
        initial_event = StreamEvent(
            event_id="initial",
            event_type="raw_data",
            timestamp=datetime.now(timezone.utc),
            data={"raw": True},
            source="test_source"
        )
        
        await streaming_manager.event_queue.put(initial_event)
        
        # Process single event from queue (simulate processing loop iteration)
        if not streaming_manager.event_queue.empty():
            event = await streaming_manager.event_queue.get()
            
            # Process through pipeline manually
            current_event = event
            for processor in streaming_manager.processors:
                processed_event = await processor.process_event(current_event)
                if processed_event:
                    current_event = processed_event
            
            # Verify processing occurred
            assert processor1.process_event.called
            assert processor2.process_event.called
    
    @pytest.mark.asyncio
    async def test_error_handling_in_processing(self):
        """Test error handling during event processing."""
        # Create processor that raises exception
        failing_processor = Mock()
        failing_processor.process_event = AsyncMock(side_effect=Exception("Processing failed"))
        failing_processor.get_processor_info.return_value = {"processor_type": "failing_processor"}
        
        # Create normal processor
        normal_processor = Mock()
        normal_processor.process_event = AsyncMock(return_value=StreamEvent(
            event_id="normal_result",
            event_type="normal_data",
            timestamp=datetime.now(timezone.utc),
            data={"normal": True},
            source="normal_processor"
        ))
        normal_processor.get_processor_info.return_value = {"processor_type": "normal_processor"}
        
        # Setup streaming manager
        mock_ingestion_manager = Mock(spec=DataIngestionManager)
        mock_ingestion_manager.set_data_callback = Mock()
        streaming_manager = StreamingManager(mock_ingestion_manager)
        
        streaming_manager.add_processor(failing_processor)
        streaming_manager.add_processor(normal_processor)
        
        # Create test event
        test_event = StreamEvent(
            event_id="test_error",
            event_type="test_data",
            timestamp=datetime.now(timezone.utc),
            data={"test": True},
            source="test_source"
        )
        
        await streaming_manager.event_queue.put(test_event)
        
        # Process event (should handle error gracefully)
        if not streaming_manager.event_queue.empty():
            event = await streaming_manager.event_queue.get()
            
            # Process through pipeline with error handling
            current_event = event
            for processor in streaming_manager.processors:
                try:
                    processed_event = await processor.process_event(current_event)
                    if processed_event:
                        current_event = processed_event
                except Exception:
                    # Error should be caught and logged, processing continues
                    pass
            
            # Normal processor should still be called
            assert normal_processor.process_event.called