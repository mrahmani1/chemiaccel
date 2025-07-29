"""
Real-Time Dashboard for Self-Assembly Property Tracking.

This module provides specialized real-time dashboard capabilities for monitoring
self-assembly simulations as they run, with live updates of emergent properties,
phase transitions, and structural evolution.

Features:
- Real-time data streaming from running simulations
- Live property tracking with automatic alerts
- Dynamic phase transition detection and visualization
- Performance monitoring and optimization suggestions
- Multi-simulation comparison dashboards
- Automated report generation

Dependencies:
- dash: For web-based real-time dashboards
- plotly: For interactive real-time plots
- websockets: For real-time data communication
- threading: For background data processing
"""

import asyncio
import json
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Optional imports with graceful fallbacks
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Real-time visualization will be limited.")

try:
    import dash
    import dash_bootstrap_components as dbc
    from dash import Input, Output, State, callback, ctx, dcc, html
    from dash.exceptions import PreventUpdate

    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    warnings.warn("Dash not available. Real-time dashboards will be limited.")

try:
    import websockets

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    warnings.warn("WebSockets not available. Real-time communication will be limited.")

# Import from other modules
from .characterization import CharacterizationResult, PhaseType, StructureType
from .interactive_visualization import (
    DashboardConfig,
    DashboardError,
    PhaseTransitionEvent,
    TimeSeriesData,
    VisualizationConfig,
    VisualizationError,
)

# =============================================================================
# Real-Time Data Structures
# =============================================================================


class AlertLevel(Enum):
    """Alert levels for monitoring."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class MetricStatus(Enum):
    """Status of monitored metrics."""

    NORMAL = "normal"
    DEGRADING = "degrading"
    CRITICAL = "critical"
    IMPROVING = "improving"


@dataclass
class RealTimeMetric:
    """Real-time metric data structure."""

    name: str
    current_value: float
    target_value: Optional[float] = None
    tolerance: float = 0.1
    status: MetricStatus = MetricStatus.NORMAL
    trend: str = "stable"  # "increasing", "decreasing", "stable"
    last_updated: float = field(default_factory=time.time)
    history: deque = field(default_factory=lambda: deque(maxlen=1000))
    unit: str = ""
    category: str = "general"


@dataclass
class SimulationAlert:
    """Alert for simulation monitoring."""

    timestamp: float
    level: AlertLevel
    message: str
    metric_name: str
    current_value: float
    threshold_value: Optional[float] = None
    suggested_action: str = ""
    auto_resolved: bool = False


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics."""

    fps: float = 0.0  # Frames per second
    memory_usage: float = 0.0  # MB
    gpu_utilization: float = 0.0  # Percentage
    cpu_utilization: float = 0.0  # Percentage
    simulation_speed: float = 1.0  # Relative to real-time
    last_frame_time: float = 0.0  # seconds
    total_simulation_time: float = 0.0  # seconds


@dataclass
class RealTimeDashboardConfig(DashboardConfig):
    """Extended configuration for real-time dashboards."""

    # Real-time specific settings
    websocket_port: int = 8765
    max_data_points_realtime: int = 500
    alert_check_interval: float = 5.0  # seconds
    performance_update_interval: float = 2.0  # seconds

    # Alert settings
    enable_alerts: bool = True
    alert_email: Optional[str] = None
    alert_webhook: Optional[str] = None
    critical_alert_threshold: float = 0.9

    # Auto-optimization settings
    enable_auto_optimization: bool = False
    optimization_check_interval: float = 30.0  # seconds

    # Data persistence
    enable_data_logging: bool = True
    log_file_path: str = "simulation_log.jsonl"
    max_log_file_size: int = 100  # MB


# =============================================================================
# Real-Time Data Manager
# =============================================================================


class RealTimeDataManager:
    """
    Manages real-time data collection, processing, and distribution
    for self-assembly simulation monitoring.
    """

    def __init__(self, config: Optional[RealTimeDashboardConfig] = None):
        """
        Initialize the data manager.

        Args:
            config: Real-time dashboard configuration
        """
        self.config = config or RealTimeDashboardConfig()

        # Data storage
        self.metrics: Dict[str, RealTimeMetric] = {}
        self.alerts: List[SimulationAlert] = []
        self.performance_metrics = PerformanceMetrics()

        # Real-time data streams
        self.data_streams: Dict[str, deque] = {}
        self.subscribers: List[Callable] = []

        # Threading for background processing
        self.running = False
        self.data_thread: Optional[threading.Thread] = None
        self.alert_thread: Optional[threading.Thread] = None

        # WebSocket server for real-time communication
        self.websocket_server = None
        self.connected_clients = set()

        # Performance tracking
        self.last_update_time = time.time()
        self.frame_count = 0

        # Data logging
        self.log_file = None
        if self.config.enable_data_logging:
            self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup data logging."""
        try:
            self.log_file = open(self.config.log_file_path, "a")
        except Exception as e:
            warnings.warn(f"Failed to setup logging: {e}")

    def add_metric(
        self,
        name: str,
        initial_value: float = 0.0,
        target_value: Optional[float] = None,
        tolerance: float = 0.1,
        unit: str = "",
        category: str = "general",
    ) -> None:
        """
        Add a new metric for real-time tracking.

        Args:
            name: Metric name
            initial_value: Initial value
            target_value: Target value for monitoring
            tolerance: Tolerance for alerts
            unit: Unit of measurement
            category: Metric category
        """
        metric = RealTimeMetric(
            name=name,
            current_value=initial_value,
            target_value=target_value,
            tolerance=tolerance,
            unit=unit,
            category=category,
        )

        self.metrics[name] = metric
        self.data_streams[name] = deque(maxlen=self.config.max_data_points_realtime)

    def update_metric(
        self, name: str, value: float, timestamp: Optional[float] = None
    ) -> None:
        """
        Update a metric value.

        Args:
            name: Metric name
            value: New value
            timestamp: Timestamp (None for current time)
        """
        if name not in self.metrics:
            self.add_metric(name, value)

        metric = self.metrics[name]
        timestamp = timestamp or time.time()

        # Update metric
        previous_value = metric.current_value
        metric.current_value = value
        metric.last_updated = timestamp

        # Update history
        metric.history.append((timestamp, value))
        self.data_streams[name].append((timestamp, value))

        # Calculate trend
        if len(metric.history) >= 2:
            recent_values = [v for _, v in list(metric.history)[-10:]]
            if len(recent_values) >= 2:
                slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                if slope > 0.01:
                    metric.trend = "increasing"
                elif slope < -0.01:
                    metric.trend = "decreasing"
                else:
                    metric.trend = "stable"

        # Update status
        self._update_metric_status(metric)

        # Log data if enabled
        if self.log_file:
            self._log_data_point(name, value, timestamp)

        # Notify subscribers
        self._notify_subscribers(name, value, timestamp)

        # Update performance metrics
        self._update_performance_metrics()

    def _update_metric_status(self, metric: RealTimeMetric) -> None:
        """Update metric status based on target and tolerance."""
        if metric.target_value is None:
            return

        deviation = abs(metric.current_value - metric.target_value)
        relative_deviation = (
            deviation / abs(metric.target_value)
            if metric.target_value != 0
            else deviation
        )

        if relative_deviation <= metric.tolerance:
            metric.status = MetricStatus.NORMAL
        elif relative_deviation <= metric.tolerance * 2:
            metric.status = MetricStatus.DEGRADING
        else:
            metric.status = MetricStatus.CRITICAL

    def _update_performance_metrics(self) -> None:
        """Update performance tracking metrics."""
        current_time = time.time()

        # Calculate FPS
        self.frame_count += 1
        time_diff = current_time - self.last_update_time

        if time_diff >= 1.0:  # Update every second
            self.performance_metrics.fps = self.frame_count / time_diff
            self.frame_count = 0
            self.last_update_time = current_time

        # Update simulation time
        self.performance_metrics.total_simulation_time += time_diff
        self.performance_metrics.last_frame_time = time_diff

    def _log_data_point(self, name: str, value: float, timestamp: float) -> None:
        """Log data point to file."""
        try:
            log_entry = {
                "timestamp": timestamp,
                "metric": name,
                "value": value,
                "trend": self.metrics[name].trend,
                "status": self.metrics[name].status.value,
            }

            self.log_file.write(json.dumps(log_entry) + "\n")
            self.log_file.flush()

        except Exception as e:
            warnings.warn(f"Failed to log data: {e}")

    def _notify_subscribers(self, name: str, value: float, timestamp: float) -> None:
        """Notify all subscribers of data updates."""
        for subscriber in self.subscribers:
            try:
                subscriber(name, value, timestamp)
            except Exception as e:
                warnings.warn(f"Subscriber notification failed: {e}")

    def subscribe(self, callback: Callable[[str, float, float], None]) -> None:
        """
        Subscribe to data updates.

        Args:
            callback: Callback function (metric_name, value, timestamp)
        """
        self.subscribers.append(callback)

    def unsubscribe(self, callback: Callable) -> None:
        """Unsubscribe from data updates."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)

    def add_alert(
        self,
        level: AlertLevel,
        message: str,
        metric_name: str,
        current_value: float,
        threshold_value: Optional[float] = None,
        suggested_action: str = "",
    ) -> None:
        """Add a new alert."""
        alert = SimulationAlert(
            timestamp=time.time(),
            level=level,
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            suggested_action=suggested_action,
        )

        self.alerts.append(alert)

        # Keep only recent alerts
        max_alerts = 1000
        if len(self.alerts) > max_alerts:
            self.alerts = self.alerts[-max_alerts:]

    def get_recent_data(
        self, metric_name: str, time_window: float = 300.0  # 5 minutes
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get recent data for a metric.

        Args:
            metric_name: Name of metric
            time_window: Time window in seconds

        Returns:
            Tuple of (timestamps, values)
        """
        if metric_name not in self.data_streams:
            return np.array([]), np.array([])

        current_time = time.time()
        cutoff_time = current_time - time_window

        data = [(t, v) for t, v in self.data_streams[metric_name] if t >= cutoff_time]

        if not data:
            return np.array([]), np.array([])

        timestamps, values = zip(*data)
        return np.array(timestamps), np.array(values)

    def start_monitoring(self) -> None:
        """Start background monitoring threads."""
        if self.running:
            return

        self.running = True

        # Start alert monitoring thread
        if self.config.enable_alerts:
            self.alert_thread = threading.Thread(target=self._alert_monitor_loop)
            self.alert_thread.daemon = True
            self.alert_thread.start()

        # Start WebSocket server
        if WEBSOCKETS_AVAILABLE:
            self.data_thread = threading.Thread(target=self._start_websocket_server)
            self.data_thread.daemon = True
            self.data_thread.start()

    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self.running = False

        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def _alert_monitor_loop(self) -> None:
        """Background loop for alert monitoring."""
        while self.running:
            try:
                self._check_alerts()
                time.sleep(self.config.alert_check_interval)
            except Exception as e:
                warnings.warn(f"Alert monitoring error: {e}")

    def _check_alerts(self) -> None:
        """Check metrics for alert conditions."""
        for metric in self.metrics.values():
            if metric.status == MetricStatus.CRITICAL:
                self.add_alert(
                    level=AlertLevel.CRITICAL,
                    message=f"{metric.name} is critically out of range",
                    metric_name=metric.name,
                    current_value=metric.current_value,
                    threshold_value=metric.target_value,
                    suggested_action="Check simulation parameters and constraints",
                )

            elif metric.status == MetricStatus.DEGRADING:
                self.add_alert(
                    level=AlertLevel.WARNING,
                    message=f"{metric.name} is degrading",
                    metric_name=metric.name,
                    current_value=metric.current_value,
                    threshold_value=metric.target_value,
                    suggested_action="Monitor closely and consider intervention",
                )

    def _start_websocket_server(self) -> None:
        """Start WebSocket server for real-time communication."""
        if not WEBSOCKETS_AVAILABLE:
            return

        async def handle_client(websocket, path):
            self.connected_clients.add(websocket)
            try:
                await websocket.wait_closed()
            finally:
                self.connected_clients.remove(websocket)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        start_server = websockets.serve(
            handle_client, "localhost", self.config.websocket_port
        )

        loop.run_until_complete(start_server)
        loop.run_forever()

    async def broadcast_update(self, data: Dict[str, Any]) -> None:
        """Broadcast update to all connected WebSocket clients."""
        if not self.connected_clients:
            return

        message = json.dumps(data)
        disconnected = set()

        for client in self.connected_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)

        # Remove disconnected clients
        self.connected_clients -= disconnected


# =============================================================================
# Real-Time Dashboard
# =============================================================================


class RealTimeSelfAssemblyDashboard:
    """
    Real-time dashboard for monitoring self-assembly simulations
    with live updates and interactive controls.
    """

    def __init__(
        self, config: Optional[RealTimeDashboardConfig] = None, port: int = 8051
    ):
        """
        Initialize the real-time dashboard.

        Args:
            config: Dashboard configuration
            port: Port for web server
        """
        self.config = config or RealTimeDashboardConfig()
        self.port = port

        # Data manager
        self.data_manager = RealTimeDataManager(self.config)

        # Dashboard app
        self.app = None
        self.server_running = False

        # UI state
        self.selected_metrics = []
        self.time_window = 300.0  # 5 minutes
        self.auto_refresh = True

        if not DASH_AVAILABLE:
            warnings.warn("Dash not available. Real-time dashboard will be limited.")

    def add_simulation_metrics(
        self,
        micelle_count: int = 0,
        vesicle_count: int = 0,
        aggregation_number: float = 0.0,
        membrane_thickness: float = 0.0,
        cluster_stability: float = 0.0,
    ) -> None:
        """Add standard simulation metrics for tracking."""

        self.data_manager.add_metric(
            "micelle_count",
            micelle_count,
            target_value=10,
            tolerance=0.2,
            unit="count",
            category="structure",
        )

        self.data_manager.add_metric(
            "vesicle_count",
            vesicle_count,
            target_value=5,
            tolerance=0.3,
            unit="count",
            category="structure",
        )

        self.data_manager.add_metric(
            "aggregation_number",
            aggregation_number,
            target_value=50,
            tolerance=0.15,
            unit="molecules",
            category="assembly",
        )

        self.data_manager.add_metric(
            "membrane_thickness",
            membrane_thickness,
            target_value=4.0,
            tolerance=0.1,
            unit="nm",
            category="structure",
        )

        self.data_manager.add_metric(
            "cluster_stability",
            cluster_stability,
            target_value=0.8,
            tolerance=0.1,
            unit="",
            category="stability",
        )

    def update_from_characterization(
        self, result: CharacterizationResult, timestamp: Optional[float] = None
    ) -> None:
        """
        Update dashboard from characterization results.

        Args:
            result: Characterization result
            timestamp: Optional timestamp
        """
        metrics = result.structure_metrics
        timestamp = timestamp or time.time()

        # Update structural metrics
        if hasattr(metrics, "micelle_count"):
            self.data_manager.update_metric(
                "micelle_count", metrics.micelle_count, timestamp
            )

        if hasattr(metrics, "vesicle_count"):
            self.data_manager.update_metric(
                "vesicle_count", metrics.vesicle_count, timestamp
            )

        if hasattr(metrics, "aggregation_number"):
            self.data_manager.update_metric(
                "aggregation_number", metrics.aggregation_number, timestamp
            )

        if hasattr(metrics, "membrane_thickness"):
            self.data_manager.update_metric(
                "membrane_thickness", metrics.membrane_thickness, timestamp
            )

        if hasattr(metrics, "cluster_stability"):
            self.data_manager.update_metric(
                "cluster_stability", metrics.cluster_stability, timestamp
            )

        # Update phase information
        if hasattr(result, "dominant_phase"):
            phase_score = self._calculate_phase_score(result.dominant_phase)
            self.data_manager.update_metric("phase_stability", phase_score, timestamp)

    def _calculate_phase_score(self, phase_type: PhaseType) -> float:
        """Calculate a numeric score for phase stability."""
        phase_scores = {
            PhaseType.LIQUID: 0.2,
            PhaseType.MICELLE: 0.8,
            PhaseType.VESICLE: 0.9,
            PhaseType.BILAYER: 0.85,
            PhaseType.CRYSTAL: 0.95,
        }
        return phase_scores.get(phase_type, 0.5)

    def create_dashboard(self) -> None:
        """Create the real-time dashboard."""
        if not DASH_AVAILABLE:
            raise DashboardError("Dash not available for dashboard creation")

        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        # Create layout
        self.app.layout = self._create_realtime_layout()

        # Register callbacks
        self._register_realtime_callbacks()

    def _create_realtime_layout(self):
        """Create real-time dashboard layout."""
        return dbc.Container(
            [
                # Header with status indicators
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H1(
                                    "Real-Time Self-Assembly Monitor",
                                    className="text-center mb-3",
                                ),
                                html.Div(
                                    id="status-indicators", className="text-center mb-3"
                                ),
                            ]
                        )
                    ]
                ),
                # Control panel
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [
                                                html.H5(
                                                    "Real-Time Controls",
                                                    className="mb-0",
                                                )
                                            ]
                                        ),
                                        dbc.CardBody(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Label(
                                                                    "Time Window:"
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="time-window-selector",
                                                                    options=[
                                                                        {
                                                                            "label": "1 minute",
                                                                            "value": 60,
                                                                        },
                                                                        {
                                                                            "label": "5 minutes",
                                                                            "value": 300,
                                                                        },
                                                                        {
                                                                            "label": "15 minutes",
                                                                            "value": 900,
                                                                        },
                                                                        {
                                                                            "label": "1 hour",
                                                                            "value": 3600,
                                                                        },
                                                                    ],
                                                                    value=300,
                                                                    clearable=False,
                                                                ),
                                                            ],
                                                            width=4,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("Metrics:"),
                                                                dcc.Dropdown(
                                                                    id="realtime-metric-selector",
                                                                    options=[],  # Will be populated dynamically
                                                                    multi=True,
                                                                    value=[],
                                                                    clearable=False,
                                                                ),
                                                            ],
                                                            width=4,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.Br(),
                                                                dbc.ButtonGroup(
                                                                    [
                                                                        dbc.Button(
                                                                            "⏸️ Pause",
                                                                            id="pause-button",
                                                                            color="warning",
                                                                            size="sm",
                                                                        ),
                                                                        dbc.Button(
                                                                            "📊 Export",
                                                                            id="export-realtime-button",
                                                                            color="info",
                                                                            size="sm",
                                                                        ),
                                                                        dbc.Button(
                                                                            "🔄 Reset",
                                                                            id="reset-button",
                                                                            color="danger",
                                                                            size="sm",
                                                                        ),
                                                                    ]
                                                                ),
                                                            ],
                                                            width=4,
                                                        ),
                                                    ]
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=12,
                        )
                    ],
                    className="mb-4",
                ),
                # Main monitoring area
                dbc.Row(
                    [
                        # Real-time plots
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Live Property Tracking"),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="realtime-plot",
                                                    style={"height": "400px"},
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=8,
                        ),
                        # Current status and alerts
                        dbc.Col(
                            [
                                # Current values
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Current Values"),
                                        dbc.CardBody(id="current-values-display"),
                                    ],
                                    className="mb-3",
                                ),
                                # Recent alerts
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Alerts"),
                                        dbc.CardBody(id="alerts-display"),
                                    ]
                                ),
                            ],
                            width=4,
                        ),
                    ],
                    className="mb-4",
                ),
                # Performance monitoring
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Performance Metrics"),
                                        dbc.CardBody(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.H6(
                                                                    "Simulation Speed",
                                                                    className="text-center",
                                                                ),
                                                                html.H4(
                                                                    id="fps-display",
                                                                    className="text-center text-primary",
                                                                ),
                                                            ],
                                                            width=3,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.H6(
                                                                    "Memory Usage",
                                                                    className="text-center",
                                                                ),
                                                                html.H4(
                                                                    id="memory-display",
                                                                    className="text-center text-info",
                                                                ),
                                                            ],
                                                            width=3,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.H6(
                                                                    "GPU Utilization",
                                                                    className="text-center",
                                                                ),
                                                                html.H4(
                                                                    id="gpu-display",
                                                                    className="text-center text-success",
                                                                ),
                                                            ],
                                                            width=3,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.H6(
                                                                    "Simulation Time",
                                                                    className="text-center",
                                                                ),
                                                                html.H4(
                                                                    id="simtime-display",
                                                                    className="text-center text-secondary",
                                                                ),
                                                            ],
                                                            width=3,
                                                        ),
                                                    ]
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=12,
                        )
                    ],
                    className="mb-4",
                ),
                # Phase transition tracking
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Phase Transition Monitor"),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="phase-monitor-plot",
                                                    style={"height": "300px"},
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("System Health"),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="health-monitor-plot",
                                                    style={"height": "300px"},
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=6,
                        ),
                    ]
                ),
                # Auto-refresh interval
                dcc.Interval(
                    id="realtime-interval",
                    interval=self.config.update_interval,
                    n_intervals=0,
                    disabled=False,
                ),
                # Hidden div for state
                html.Div(id="dashboard-state", style={"display": "none"}),
            ],
            fluid=True,
        )

    def _register_realtime_callbacks(self):
        """Register real-time dashboard callbacks."""

        @self.app.callback(
            Output("realtime-metric-selector", "options"),
            Input("realtime-interval", "n_intervals"),
        )
        def update_metric_options(n_intervals):
            metrics = list(self.data_manager.metrics.keys())
            return [
                {"label": metric.replace("_", " ").title(), "value": metric}
                for metric in metrics
            ]

        @self.app.callback(
            Output("realtime-metric-selector", "value"),
            Input("realtime-metric-selector", "options"),
            State("realtime-metric-selector", "value"),
        )
        def set_default_metrics(options, current_value):
            if not current_value and options:
                # Default to first 3 metrics
                return [opt["value"] for opt in options[:3]]
            return current_value

        @self.app.callback(
            [
                Output("realtime-plot", "figure"),
                Output("current-values-display", "children"),
                Output("alerts-display", "children"),
                Output("status-indicators", "children"),
            ],
            [
                Input("realtime-interval", "n_intervals"),
                Input("time-window-selector", "value"),
                Input("realtime-metric-selector", "value"),
            ],
        )
        def update_realtime_display(n_intervals, time_window, selected_metrics):
            # Update time window
            self.time_window = time_window or 300

            # Create real-time plot
            fig = self._create_realtime_plot(selected_metrics)

            # Create current values display
            current_values = self._create_current_values_display()

            # Create alerts display
            alerts_display = self._create_alerts_display()

            # Create status indicators
            status_indicators = self._create_status_indicators()

            return fig, current_values, alerts_display, status_indicators

        @self.app.callback(
            [
                Output("fps-display", "children"),
                Output("memory-display", "children"),
                Output("gpu-display", "children"),
                Output("simtime-display", "children"),
            ],
            Input("realtime-interval", "n_intervals"),
        )
        def update_performance_display(n_intervals):
            perf = self.data_manager.performance_metrics

            fps_text = f"{perf.fps:.1f} FPS"
            memory_text = f"{perf.memory_usage:.0f} MB"
            gpu_text = f"{perf.gpu_utilization:.0f}%"
            simtime_text = f"{perf.total_simulation_time:.1f}s"

            return fps_text, memory_text, gpu_text, simtime_text

        @self.app.callback(
            Output("phase-monitor-plot", "figure"),
            Input("realtime-interval", "n_intervals"),
        )
        def update_phase_monitor(n_intervals):
            return self._create_phase_monitor_plot()

        @self.app.callback(
            Output("health-monitor-plot", "figure"),
            Input("realtime-interval", "n_intervals"),
        )
        def update_health_monitor(n_intervals):
            return self._create_health_monitor_plot()

        @self.app.callback(
            Output("realtime-interval", "disabled"),
            Input("pause-button", "n_clicks"),
            State("realtime-interval", "disabled"),
        )
        def toggle_pause(n_clicks, is_disabled):
            if n_clicks:
                return not is_disabled
            return is_disabled

    def _create_realtime_plot(self, selected_metrics: List[str]) -> go.Figure:
        """Create real-time plot for selected metrics."""
        if not PLOTLY_AVAILABLE or not selected_metrics:
            return go.Figure()

        fig = make_subplots(
            rows=len(selected_metrics),
            cols=1,
            shared_xaxes=True,
            subplot_titles=[
                metric.replace("_", " ").title() for metric in selected_metrics
            ],
            vertical_spacing=0.05,
        )

        for i, metric_name in enumerate(selected_metrics, 1):
            timestamps, values = self.data_manager.get_recent_data(
                metric_name, self.time_window
            )

            if len(timestamps) > 0:
                # Convert timestamps to relative time
                relative_times = (
                    (timestamps - timestamps[0]) if len(timestamps) > 1 else [0]
                )

                # Add main trace
                fig.add_trace(
                    go.Scatter(
                        x=relative_times,
                        y=values,
                        mode="lines+markers",
                        name=metric_name,
                        line=dict(width=2),
                        marker=dict(size=4),
                        hovertemplate="%{y:.3f}<extra></extra>",
                    ),
                    row=i,
                    col=1,
                )

                # Add target line if available
                metric = self.data_manager.metrics.get(metric_name)
                if metric and metric.target_value is not None:
                    fig.add_hline(
                        y=metric.target_value,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Target",
                        row=i,
                        col=1,
                    )

            # Update axis labels
            metric = self.data_manager.metrics.get(metric_name)
            ylabel = metric_name.replace("_", " ").title()
            if metric and metric.unit:
                ylabel += f" ({metric.unit})"

            fig.update_yaxes(title_text=ylabel, row=i, col=1)

        fig.update_xaxes(title_text="Time (s)", row=len(selected_metrics), col=1)
        fig.update_layout(
            height=150 * len(selected_metrics),
            showlegend=False,
            title="Real-Time Property Tracking",
            margin=dict(l=50, r=50, t=50, b=50),
        )

        return fig

    def _create_current_values_display(self) -> List:
        """Create current values display."""
        components = []

        for metric_name, metric in self.data_manager.metrics.items():
            # Status color
            status_colors = {
                MetricStatus.NORMAL: "success",
                MetricStatus.DEGRADING: "warning",
                MetricStatus.CRITICAL: "danger",
                MetricStatus.IMPROVING: "info",
            }

            color = status_colors.get(metric.status, "secondary")

            # Trend indicator
            trend_icons = {"increasing": "📈", "decreasing": "📉", "stable": "➡️"}

            trend_icon = trend_icons.get(metric.trend, "")

            # Value display
            value_text = f"{metric.current_value:.3f}"
            if metric.unit:
                value_text += f" {metric.unit}"

            components.append(
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H6(
                                    metric_name.replace("_", " ").title(),
                                    className="mb-1",
                                ),
                                html.H5(
                                    [
                                        value_text,
                                        html.Span(f" {trend_icon}", className="ms-2"),
                                    ],
                                    className=f"text-{color} mb-0",
                                ),
                                html.Small(
                                    f"Status: {metric.status.value}",
                                    className="text-muted",
                                ),
                            ]
                        )
                    ],
                    className="mb-2",
                )
            )

        return components

    def _create_alerts_display(self) -> List:
        """Create alerts display."""
        recent_alerts = self.data_manager.alerts[-10:]  # Last 10 alerts

        if not recent_alerts:
            return [html.P("No recent alerts", className="text-muted")]

        components = []

        for alert in reversed(recent_alerts):  # Most recent first
            # Alert color
            alert_colors = {
                AlertLevel.INFO: "info",
                AlertLevel.WARNING: "warning",
                AlertLevel.CRITICAL: "danger",
            }

            color = alert_colors.get(alert.level, "secondary")

            # Time ago
            time_ago = time.time() - alert.timestamp
            if time_ago < 60:
                time_str = f"{int(time_ago)}s ago"
            elif time_ago < 3600:
                time_str = f"{int(time_ago/60)}m ago"
            else:
                time_str = f"{int(time_ago/3600)}h ago"

            components.append(
                dbc.Alert(
                    [
                        html.Strong(f"{alert.level.value.upper()}: "),
                        alert.message,
                        html.Br(),
                        html.Small(
                            f"{time_str} - {alert.metric_name}", className="text-muted"
                        ),
                    ],
                    color=color,
                    className="py-2 mb-2",
                )
            )

        return components

    def _create_status_indicators(self) -> List:
        """Create status indicators."""
        # Overall system status
        critical_count = sum(
            1
            for m in self.data_manager.metrics.values()
            if m.status == MetricStatus.CRITICAL
        )
        warning_count = sum(
            1
            for m in self.data_manager.metrics.values()
            if m.status == MetricStatus.DEGRADING
        )

        if critical_count > 0:
            status_color = "danger"
            status_text = "CRITICAL"
        elif warning_count > 0:
            status_color = "warning"
            status_text = "WARNING"
        else:
            status_color = "success"
            status_text = "NORMAL"

        indicators = [
            dbc.Badge(
                f"System Status: {status_text}", color=status_color, className="me-2"
            ),
            dbc.Badge(
                f"🔴 {critical_count} Critical",
                color="danger" if critical_count > 0 else "secondary",
                className="me-2",
            ),
            dbc.Badge(
                f"🟡 {warning_count} Warning",
                color="warning" if warning_count > 0 else "secondary",
                className="me-2",
            ),
            dbc.Badge(
                f"FPS: {self.data_manager.performance_metrics.fps:.1f}",
                color="info",
                className="me-2",
            ),
        ]

        return indicators

    def _create_phase_monitor_plot(self) -> go.Figure:
        """Create phase transition monitoring plot."""
        if not PLOTLY_AVAILABLE:
            return go.Figure()

        # Get phase stability data
        timestamps, values = self.data_manager.get_recent_data(
            "phase_stability", self.time_window
        )

        if len(timestamps) == 0:
            return go.Figure().add_annotation(
                text="No phase data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        relative_times = (timestamps - timestamps[0]) if len(timestamps) > 1 else [0]

        fig = go.Figure()

        # Add phase stability trace
        fig.add_trace(
            go.Scatter(
                x=relative_times,
                y=values,
                mode="lines+markers",
                name="Phase Stability",
                line=dict(width=2, color="blue"),
                fill="tonexty",
            )
        )

        # Add phase transition markers
        recent_transitions = [
            alert
            for alert in self.data_manager.alerts
            if "phase" in alert.message.lower()
            and (time.time() - alert.timestamp) <= self.time_window
        ]

        for transition in recent_transitions:
            rel_time = (
                transition.timestamp - timestamps[0] if len(timestamps) > 0 else 0
            )
            fig.add_vline(
                x=rel_time,
                line_dash="dash",
                line_color="red",
                annotation_text="Transition",
            )

        fig.update_layout(
            title="Phase Stability Monitor",
            xaxis_title="Time (s)",
            yaxis_title="Stability Score",
            height=300,
        )

        return fig

    def _create_health_monitor_plot(self) -> go.Figure:
        """Create system health monitoring plot."""
        if not PLOTLY_AVAILABLE:
            return go.Figure()

        # Create health score based on metric statuses
        health_scores = []
        timestamps_list = []

        # Calculate health over time
        current_time = time.time()
        time_points = np.linspace(current_time - self.time_window, current_time, 20)

        for t in time_points:
            score = 0.0
            count = 0

            for metric in self.data_manager.metrics.values():
                if metric.history:
                    # Find closest timestamp
                    closest_idx = min(
                        range(len(metric.history)),
                        key=lambda i: abs(metric.history[i][0] - t),
                    )

                    status_scores = {
                        MetricStatus.NORMAL: 1.0,
                        MetricStatus.IMPROVING: 1.0,
                        MetricStatus.DEGRADING: 0.5,
                        MetricStatus.CRITICAL: 0.0,
                    }

                    score += status_scores.get(metric.status, 0.5)
                    count += 1

            if count > 0:
                health_scores.append(score / count)
                timestamps_list.append(t - time_points[0])

        fig = go.Figure()

        if health_scores:
            fig.add_trace(
                go.Scatter(
                    x=timestamps_list,
                    y=health_scores,
                    mode="lines+markers",
                    name="System Health",
                    line=dict(width=3, color="green"),
                    fill="tozeroy",
                    fillcolor="rgba(0,255,0,0.1)",
                )
            )

            # Add health zones
            fig.add_hline(
                y=0.8, line_dash="dash", line_color="green", annotation_text="Good"
            )
            fig.add_hline(
                y=0.5, line_dash="dash", line_color="orange", annotation_text="Caution"
            )
            fig.add_hline(
                y=0.2, line_dash="dash", line_color="red", annotation_text="Critical"
            )

        fig.update_layout(
            title="System Health Score",
            xaxis_title="Time (s)",
            yaxis_title="Health Score",
            yaxis=dict(range=[0, 1]),
            height=300,
        )

        return fig

    def start_monitoring(self) -> None:
        """Start real-time monitoring."""
        self.data_manager.start_monitoring()

    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        self.data_manager.stop_monitoring()
        self.server_running = False

    def run_dashboard(self, debug: bool = False) -> None:
        """
        Run the real-time dashboard server.

        Args:
            debug: Whether to run in debug mode
        """
        if not DASH_AVAILABLE:
            raise DashboardError("Dash not available")

        if not self.app:
            self.create_dashboard()

        # Start monitoring
        self.start_monitoring()

        try:
            self.server_running = True
            self.app.run_server(debug=debug, port=self.port, host="127.0.0.1")
        except Exception as e:
            self.server_running = False
            raise DashboardError(f"Failed to start real-time dashboard: {str(e)}")
        finally:
            self.stop_monitoring()


# =============================================================================
# Utility Functions and Demo
# =============================================================================


def create_demo_realtime_dashboard(port: int = 8051) -> RealTimeSelfAssemblyDashboard:
    """Create a demo real-time dashboard with synthetic data."""

    # Create dashboard
    dashboard = RealTimeSelfAssemblyDashboard(port=port)

    # Add simulation metrics
    dashboard.add_simulation_metrics()

    # Start background data generation
    def generate_synthetic_data():
        """Generate synthetic real-time data."""
        t = 0
        while True:
            # Simulate evolving properties
            micelle_count = 10 + 3 * np.sin(t / 10) + np.random.normal(0, 0.5)
            vesicle_count = 5 + 2 * np.cos(t / 15) + np.random.normal(0, 0.3)
            agg_number = 50 + 10 * np.sin(t / 8) + np.random.normal(0, 2)
            thickness = 4.0 + 0.5 * np.cos(t / 12) + np.random.normal(0, 0.1)
            stability = 0.8 + 0.15 * np.sin(t / 20) + np.random.normal(0, 0.05)

            # Update metrics
            dashboard.data_manager.update_metric("micelle_count", max(0, micelle_count))
            dashboard.data_manager.update_metric("vesicle_count", max(0, vesicle_count))
            dashboard.data_manager.update_metric(
                "aggregation_number", max(0, agg_number)
            )
            dashboard.data_manager.update_metric(
                "membrane_thickness", max(0, thickness)
            )
            dashboard.data_manager.update_metric(
                "cluster_stability", max(0, min(1, stability))
            )

            # Update performance metrics
            dashboard.data_manager.performance_metrics.fps = 30 + np.random.normal(0, 2)
            dashboard.data_manager.performance_metrics.memory_usage = (
                512 + np.random.normal(0, 50)
            )
            dashboard.data_manager.performance_metrics.gpu_utilization = (
                75 + np.random.normal(0, 10)
            )

            t += 1
            time.sleep(1)  # Update every second

    # Start data generation thread
    data_thread = threading.Thread(target=generate_synthetic_data)
    data_thread.daemon = True
    data_thread.start()

    return dashboard


if __name__ == "__main__":
    # Demo usage
    print("Creating demo real-time dashboard...")
    dashboard = create_demo_realtime_dashboard()

    print("Starting real-time monitoring...")
    dashboard.create_dashboard()

    print(f"Dashboard available at: http://localhost:{dashboard.port}")
    print("Press Ctrl+C to stop...")

    try:
        dashboard.run_dashboard(debug=True)
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
        dashboard.stop_monitoring()
