"""
Real-time Data Streaming and Event Processing

This module implements real-time data streaming capabilities with event-driven
architecture for processing experimental data as it arrives from laboratory equipment.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import deque
try:
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

from .data_ingestion import ExperimentalDataPoint, DataIngestionManager
from .data_validation import DataValidator, ValidationResult


logger = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    """Represents a streaming event."""
    event_id: str
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    source: str
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical


@dataclass
class StreamSubscription:
    """Represents a client subscription to data streams."""
    subscription_id: str
    client_id: str
    event_types: List[str]
    filters: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class StreamProcessor(ABC):
    """Abstract base class for stream processors."""
    
    @abstractmethod
    async def process_event(self, event: StreamEvent) -> Optional[StreamEvent]:
        """Process a streaming event."""
        pass
    
    @abstractmethod
    def get_processor_info(self) -> Dict[str, Any]:
        """Get information about this processor."""
        pass


class DataValidationProcessor(StreamProcessor):
    """Processor that validates incoming experimental data."""
    
    def __init__(self, validator: DataValidator):
        self.validator = validator
        self.validation_stats = {
            "total_processed": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "avg_confidence": 0.0
        }
    
    async def process_event(self, event: StreamEvent) -> Optional[StreamEvent]:
        """Validate experimental data in streaming events."""
        if event.event_type != "experimental_data":
            return event
        
        try:
            # Extract data point from event
            data_point = ExperimentalDataPoint(
                data_id=event.data.get("data_id", event.event_id),
                experiment_id=event.data.get("experiment_id", "unknown"),
                instrument_id=event.data.get("instrument_id", "unknown"),
                timestamp=datetime.fromisoformat(event.data.get("timestamp", event.timestamp.isoformat())),
                data_type=event.data.get("data_type", "unknown"),
                raw_data=event.data.get("raw_data", {}),
                metadata=event.data.get("metadata", {}),
                quality_flags=event.data.get("quality_flags", [])
            )
            
            # Validate data point
            validation_result = await self.validator.validate_data_point(data_point)
            
            # Update statistics
            self.validation_stats["total_processed"] += 1
            if validation_result.is_valid:
                self.validation_stats["valid_count"] += 1
            else:
                self.validation_stats["invalid_count"] += 1
            
            # Update average confidence
            total = self.validation_stats["total_processed"]
            current_avg = self.validation_stats["avg_confidence"]
            self.validation_stats["avg_confidence"] = (
                (current_avg * (total - 1) + validation_result.confidence_score) / total
            )
            
            # Create validation event
            validation_event = StreamEvent(
                event_id=f"validation_{event.event_id}",
                event_type="data_validation_result",
                timestamp=datetime.now(timezone.utc),
                data={
                    "original_event_id": event.event_id,
                    "data_id": data_point.data_id,
                    "is_valid": validation_result.is_valid,
                    "confidence_score": validation_result.confidence_score,
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings,
                    "quality_metrics": validation_result.quality_metrics
                },
                source="data_validation_processor",
                priority=3 if not validation_result.is_valid else 1
            )
            
            return validation_event
            
        except Exception as e:
            logger.error(f"Error in data validation processor: {e}")
            
            error_event = StreamEvent(
                event_id=f"validation_error_{event.event_id}",
                event_type="processing_error",
                timestamp=datetime.now(timezone.utc),
                data={
                    "original_event_id": event.event_id,
                    "error": str(e),
                    "processor": "data_validation"
                },
                source="data_validation_processor",
                priority=4
            )
            
            return error_event
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Get processor information."""
        return {
            "processor_type": "data_validation",
            "description": "Validates experimental data quality",
            "statistics": self.validation_stats
        }


class AggregationProcessor(StreamProcessor):
    """Processor that aggregates data over time windows."""
    
    def __init__(self, window_size_seconds: float = 60.0, aggregation_functions: Optional[List[str]] = None):
        self.window_size = window_size_seconds
        self.aggregation_functions = aggregation_functions or ["count", "mean", "std"]
        self.data_buffer = deque()
        self.last_aggregation = datetime.now(timezone.utc)
    
    async def process_event(self, event: StreamEvent) -> Optional[StreamEvent]:
        """Aggregate data over time windows."""
        if event.event_type not in ["experimental_data", "data_validation_result"]:
            return event
        
        # Add to buffer
        self.data_buffer.append(event)
        
        # Check if aggregation window has passed
        now = datetime.now(timezone.utc)
        if (now - self.last_aggregation).total_seconds() >= self.window_size:
            return await self._create_aggregation_event(now)
        
        return None
    
    async def _create_aggregation_event(self, timestamp: datetime) -> StreamEvent:
        """Create aggregation event from buffered data."""
        buffer_data = list(self.data_buffer)
        self.data_buffer.clear()
        self.last_aggregation = timestamp
        
        # Aggregate by event type and source
        aggregations = {}
        
        for event in buffer_data:
            key = f"{event.event_type}_{event.source}"
            
            if key not in aggregations:
                aggregations[key] = {
                    "count": 0,
                    "events": [],
                    "sources": set(),
                    "event_types": set()
                }
            
            aggregations[key]["count"] += 1
            aggregations[key]["events"].append(event.data)
            aggregations[key]["sources"].add(event.source)
            aggregations[key]["event_types"].add(event.event_type)
        
        # Calculate statistics
        summary = {}
        for key, data in aggregations.items():
            summary[key] = {
                "count": data["count"],
                "sources": list(data["sources"]),
                "event_types": list(data["event_types"])
            }
            
            # Calculate numerical aggregations for validation results
            if "validation" in key:
                confidence_scores = [
                    event.get("confidence_score", 0) for event in data["events"]
                    if "confidence_score" in event
                ]
                
                if confidence_scores:
                    summary[key]["confidence_stats"] = {
                        "mean": sum(confidence_scores) / len(confidence_scores),
                        "min": min(confidence_scores),
                        "max": max(confidence_scores),
                        "count": len(confidence_scores)
                    }
        
        return StreamEvent(
            event_id=f"aggregation_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            event_type="data_aggregation",
            timestamp=timestamp,
            data={
                "window_size_seconds": self.window_size,
                "aggregation_period": {
                    "start": (timestamp - datetime.fromtimestamp(self.window_size, timezone.utc)).isoformat(),
                    "end": timestamp.isoformat()
                },
                "summary": summary,
                "total_events": len(buffer_data)
            },
            source="aggregation_processor",
            priority=1
        )
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Get processor information."""
        return {
            "processor_type": "aggregation",
            "description": "Aggregates data over time windows",
            "window_size_seconds": self.window_size,
            "aggregation_functions": self.aggregation_functions,
            "buffer_size": len(self.data_buffer)
        }


class AlertProcessor(StreamProcessor):
    """Processor that generates alerts based on data conditions."""
    
    def __init__(self, alert_rules: List[Dict[str, Any]]):
        self.alert_rules = alert_rules
        self.alert_history = deque(maxlen=1000)
    
    async def process_event(self, event: StreamEvent) -> Optional[StreamEvent]:
        """Check event against alert rules."""
        alerts = []
        
        for rule in self.alert_rules:
            if await self._evaluate_rule(event, rule):
                alert = {
                    "rule_id": rule.get("rule_id", "unknown"),
                    "rule_name": rule.get("name", "Unnamed Rule"),
                    "severity": rule.get("severity", "warning"),
                    "message": rule.get("message", f"Alert triggered for event {event.event_id}"),
                    "triggered_by": event.event_id,
                    "event_data": event.data
                }
                alerts.append(alert)
                self.alert_history.append(alert)
        
        if alerts:
            return StreamEvent(
                event_id=f"alert_{event.event_id}",
                event_type="alert",
                timestamp=datetime.now(timezone.utc),
                data={
                    "alerts": alerts,
                    "original_event": event.data
                },
                source="alert_processor",
                priority=4 if any(a["severity"] == "critical" for a in alerts) else 3
            )
        
        return None
    
    async def _evaluate_rule(self, event: StreamEvent, rule: Dict[str, Any]) -> bool:
        """Evaluate an alert rule against an event."""
        try:
            # Check event type filter
            if "event_types" in rule:
                if event.event_type not in rule["event_types"]:
                    return False
            
            # Check source filter
            if "sources" in rule:
                if event.source not in rule["sources"]:
                    return False
            
            # Check conditions
            conditions = rule.get("conditions", [])
            for condition in conditions:
                if not await self._evaluate_condition(event, condition):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating alert rule: {e}")
            return False
    
    async def _evaluate_condition(self, event: StreamEvent, condition: Dict[str, Any]) -> bool:
        """Evaluate a single condition."""
        field_path = condition.get("field", "")
        operator = condition.get("operator", "equals")
        expected_value = condition.get("value")
        
        # Extract field value from event data
        field_value = self._get_nested_field(event.data, field_path)
        
        if field_value is None:
            return False
        
        # Evaluate condition based on operator
        if operator == "equals":
            return field_value == expected_value
        elif operator == "not_equals":
            return field_value != expected_value
        elif operator == "greater_than":
            return float(field_value) > float(expected_value)
        elif operator == "less_than":
            return float(field_value) < float(expected_value)
        elif operator == "contains":
            return expected_value in str(field_value)
        elif operator == "regex":
            import re
            return bool(re.search(expected_value, str(field_value)))
        
        return False
    
    def _get_nested_field(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get nested field value using dot notation."""
        fields = field_path.split(".")
        current = data
        
        for field in fields:
            if isinstance(current, dict) and field in current:
                current = current[field]
            else:
                return None
        
        return current
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Get processor information."""
        return {
            "processor_type": "alert",
            "description": "Generates alerts based on data conditions",
            "active_rules": len(self.alert_rules),
            "alert_history_size": len(self.alert_history)
        }


class WebSocketStreamer:
    """WebSocket server for real-time data streaming to clients."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        if not WEBSOCKETS_AVAILABLE:
            logger.warning("WebSockets not available, WebSocketStreamer will have limited functionality")
        
        self.host = host
        self.port = port
        self.clients: Dict[str, Any] = {}  # Changed from WebSocketServerProtocol to Any
        self.subscriptions: Dict[str, StreamSubscription] = {}
        self.server = None
    
    async def start_server(self) -> None:
        """Start the WebSocket server."""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("Cannot start WebSocket server: websockets library not available")
            return
        
        self.server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port
        )
        logger.info(f"WebSocket server started on {self.host}:{self.port}")
    
    async def stop_server(self) -> None:
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("WebSocket server stopped")
    
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str) -> None:
        """Handle WebSocket client connection."""
        client_id = f"client_{datetime.now().timestamp()}"
        self.clients[client_id] = websocket
        
        logger.info(f"Client connected: {client_id}")
        
        try:
            async for message in websocket:
                await self.handle_message(client_id, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            # Cleanup
            if client_id in self.clients:
                del self.clients[client_id]
            
            # Remove subscriptions for this client
            to_remove = [sub_id for sub_id, sub in self.subscriptions.items() 
                        if sub.client_id == client_id]
            for sub_id in to_remove:
                del self.subscriptions[sub_id]
    
    async def handle_message(self, client_id: str, message: str) -> None:
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "subscribe":
                await self.handle_subscription(client_id, data)
            elif message_type == "unsubscribe":
                await self.handle_unsubscription(client_id, data)
            elif message_type == "ping":
                await self.send_to_client(client_id, {"type": "pong"})
            
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")
            await self.send_to_client(client_id, {
                "type": "error",
                "message": str(e)
            })
    
    async def handle_subscription(self, client_id: str, data: Dict[str, Any]) -> None:
        """Handle subscription request."""
        subscription_id = data.get("subscription_id", f"sub_{datetime.now().timestamp()}")
        event_types = data.get("event_types", [])
        filters = data.get("filters", {})
        
        subscription = StreamSubscription(
            subscription_id=subscription_id,
            client_id=client_id,
            event_types=event_types,
            filters=filters
        )
        
        self.subscriptions[subscription_id] = subscription
        
        await self.send_to_client(client_id, {
            "type": "subscription_confirmed",
            "subscription_id": subscription_id,
            "event_types": event_types
        })
        
        logger.info(f"Client {client_id} subscribed to {event_types}")
    
    async def handle_unsubscription(self, client_id: str, data: Dict[str, Any]) -> None:
        """Handle unsubscription request."""
        subscription_id = data.get("subscription_id")
        
        if subscription_id in self.subscriptions:
            del self.subscriptions[subscription_id]
            
            await self.send_to_client(client_id, {
                "type": "unsubscription_confirmed",
                "subscription_id": subscription_id
            })
    
    async def send_to_client(self, client_id: str, data: Dict[str, Any]) -> None:
        """Send data to a specific client."""
        if client_id in self.clients:
            try:
                await self.clients[client_id].send(json.dumps(data))
            except Exception as e:
                logger.error(f"Error sending to client {client_id}: {e}")
    
    async def broadcast_event(self, event: StreamEvent) -> None:
        """Broadcast event to subscribed clients."""
        for subscription in self.subscriptions.values():
            if self._matches_subscription(event, subscription):
                await self.send_to_client(subscription.client_id, {
                    "type": "event",
                    "event": {
                        "event_id": event.event_id,
                        "event_type": event.event_type,
                        "timestamp": event.timestamp.isoformat(),
                        "data": event.data,
                        "source": event.source,
                        "priority": event.priority
                    }
                })
    
    def _matches_subscription(self, event: StreamEvent, subscription: StreamSubscription) -> bool:
        """Check if event matches subscription criteria."""
        if not subscription.is_active:
            return False
        
        # Check event type filter
        if subscription.event_types and event.event_type not in subscription.event_types:
            return False
        
        # Check additional filters
        for filter_key, filter_value in subscription.filters.items():
            if filter_key == "source":
                if event.source != filter_value:
                    return False
            elif filter_key == "priority":
                if event.priority < filter_value:
                    return False
            elif filter_key == "instrument_id":
                if event.data.get("instrument_id") != filter_value:
                    return False
        
        return True


class RedisStreamer:
    """Redis-based streaming for high-throughput scenarios."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, RedisStreamer will have limited functionality")
        
        self.redis_url = redis_url
        self.redis_client: Optional[Any] = None  # Changed from aioredis.Redis to Any
        self.stream_name = "chemi_accel_stream"
    
    async def connect(self) -> None:
        """Connect to Redis."""
        if not REDIS_AVAILABLE:
            logger.error("Cannot connect to Redis: aioredis library not available")
            return
        
        self.redis_client = await aioredis.from_url(self.redis_url)
        logger.info("Connected to Redis for streaming")
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Disconnected from Redis")
    
    async def publish_event(self, event: StreamEvent) -> None:
        """Publish event to Redis stream."""
        if not self.redis_client:
            raise RuntimeError("Not connected to Redis")
        
        event_data = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "timestamp": event.timestamp.isoformat(),
            "data": json.dumps(event.data),
            "source": event.source,
            "priority": str(event.priority)
        }
        
        await self.redis_client.xadd(self.stream_name, event_data)
    
    async def consume_events(self, consumer_group: str, consumer_name: str) -> AsyncGenerator[StreamEvent, None]:
        """Consume events from Redis stream."""
        if not self.redis_client:
            raise RuntimeError("Not connected to Redis")
        
        try:
            # Create consumer group if it doesn't exist
            await self.redis_client.xgroup_create(
                self.stream_name, consumer_group, id="0", mkstream=True
            )
        except aioredis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
        
        while True:
            try:
                messages = await self.redis_client.xreadgroup(
                    consumer_group,
                    consumer_name,
                    {self.stream_name: ">"},
                    count=10,
                    block=1000
                )
                
                for stream, stream_messages in messages:
                    for message_id, fields in stream_messages:
                        try:
                            event = StreamEvent(
                                event_id=fields[b"event_id"].decode(),
                                event_type=fields[b"event_type"].decode(),
                                timestamp=datetime.fromisoformat(fields[b"timestamp"].decode()),
                                data=json.loads(fields[b"data"].decode()),
                                source=fields[b"source"].decode(),
                                priority=int(fields[b"priority"].decode())
                            )
                            
                            yield event
                            
                            # Acknowledge message
                            await self.redis_client.xack(self.stream_name, consumer_group, message_id)
                            
                        except Exception as e:
                            logger.error(f"Error processing stream message: {e}")
            
            except Exception as e:
                logger.error(f"Error consuming from Redis stream: {e}")
                await asyncio.sleep(1)


class StreamingManager:
    """
    Main manager for real-time data streaming system.
    
    Coordinates data ingestion, processing, and distribution to clients.
    """
    
    def __init__(self, data_ingestion_manager: DataIngestionManager):
        self.data_ingestion_manager = data_ingestion_manager
        self.processors: List[StreamProcessor] = []
        self.websocket_streamer = WebSocketStreamer()
        self.redis_streamer: Optional[RedisStreamer] = None
        self.event_queue = asyncio.Queue()
        self.is_running = False
        
        # Setup data ingestion callback
        self.data_ingestion_manager.set_data_callback(self._handle_incoming_data)
    
    def add_processor(self, processor: StreamProcessor) -> None:
        """Add a stream processor."""
        self.processors.append(processor)
        logger.info(f"Added processor: {processor.get_processor_info()['processor_type']}")
    
    def enable_redis_streaming(self, redis_url: str = "redis://localhost:6379") -> None:
        """Enable Redis-based streaming."""
        self.redis_streamer = RedisStreamer(redis_url)
    
    async def start_streaming(self) -> None:
        """Start the streaming system."""
        if self.is_running:
            logger.warning("Streaming system already running")
            return
        
        self.is_running = True
        
        # Start WebSocket server
        await self.websocket_streamer.start_server()
        
        # Connect to Redis if enabled
        if self.redis_streamer:
            await self.redis_streamer.connect()
        
        # Start processing loop
        asyncio.create_task(self._processing_loop())
        
        logger.info("Streaming system started")
    
    async def stop_streaming(self) -> None:
        """Stop the streaming system."""
        self.is_running = False
        
        await self.websocket_streamer.stop_server()
        
        if self.redis_streamer:
            await self.redis_streamer.disconnect()
        
        logger.info("Streaming system stopped")
    
    async def _handle_incoming_data(self, data_points: List[ExperimentalDataPoint]) -> None:
        """Handle incoming experimental data."""
        for data_point in data_points:
            event = StreamEvent(
                event_id=data_point.data_id,
                event_type="experimental_data",
                timestamp=data_point.timestamp,
                data={
                    "data_id": data_point.data_id,
                    "experiment_id": data_point.experiment_id,
                    "instrument_id": data_point.instrument_id,
                    "data_type": data_point.data_type,
                    "raw_data": data_point.raw_data,
                    "metadata": data_point.metadata,
                    "quality_flags": data_point.quality_flags,
                    "processing_status": data_point.processing_status
                },
                source=f"ingestion_{data_point.instrument_id}",
                priority=2
            )
            
            await self.event_queue.put(event)
    
    async def _processing_loop(self) -> None:
        """Main event processing loop."""
        while self.is_running:
            try:
                # Get event from queue
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Process event through all processors
                current_event = event
                
                for processor in self.processors:
                    try:
                        processed_event = await processor.process_event(current_event)
                        if processed_event:
                            # If processor generated a new event, add it to queue for further processing
                            if processed_event.event_id != current_event.event_id:
                                await self.event_queue.put(processed_event)
                            current_event = processed_event
                    except Exception as e:
                        logger.error(f"Error in processor {processor.__class__.__name__}: {e}")
                
                # Distribute processed event
                await self._distribute_event(current_event)
                
            except asyncio.TimeoutError:
                # No events to process, continue
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
    
    async def _distribute_event(self, event: StreamEvent) -> None:
        """Distribute event to all streaming channels."""
        # Send to WebSocket clients
        await self.websocket_streamer.broadcast_event(event)
        
        # Send to Redis stream if enabled
        if self.redis_streamer:
            await self.redis_streamer.publish_event(event)
    
    def get_streaming_status(self) -> Dict[str, Any]:
        """Get current streaming system status."""
        return {
            "is_running": self.is_running,
            "processors_count": len(self.processors),
            "websocket_clients": len(self.websocket_streamer.clients),
            "websocket_subscriptions": len(self.websocket_streamer.subscriptions),
            "redis_enabled": self.redis_streamer is not None,
            "queue_size": self.event_queue.qsize(),
            "processors_info": [p.get_processor_info() for p in self.processors]
        }


# Factory functions

def create_default_streaming_system(data_ingestion_manager: DataIngestionManager) -> StreamingManager:
    """Create a streaming system with default processors."""
    from .data_validation import DataValidator
    
    streaming_manager = StreamingManager(data_ingestion_manager)
    
    # Add default processors
    validator = DataValidator()
    streaming_manager.add_processor(DataValidationProcessor(validator))
    streaming_manager.add_processor(AggregationProcessor(window_size_seconds=60.0))
    
    # Add default alert rules
    default_alert_rules = [
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
            "rule_id": "low_confidence",
            "name": "Low Confidence Data",
            "event_types": ["data_validation_result"],
            "severity": "warning",
            "message": "Data has low confidence score",
            "conditions": [
                {"field": "confidence_score", "operator": "less_than", "value": 0.5}
            ]
        },
        {
            "rule_id": "processing_error",
            "name": "Processing Error",
            "event_types": ["processing_error"],
            "severity": "critical",
            "message": "Error occurred during data processing",
            "conditions": []
        }
    ]
    
    streaming_manager.add_processor(AlertProcessor(default_alert_rules))
    
    return streaming_manager