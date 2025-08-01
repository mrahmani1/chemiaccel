"""
Cloud Job Monitoring and Management Interfaces

Interactive dashboards for monitoring cloud computing jobs, resource utilization,
and cost optimization across multiple cloud platforms.
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class CloudJob:
    """Container for cloud job data."""

    job_id: str
    job_name: str
    status: str
    platform: str  # 'aws', 'gcp', 'azure'
    instance_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_minutes: Optional[float] = None
    cost_usd: Optional[float] = None
    cpu_utilization: Optional[float] = None
    memory_utilization: Optional[float] = None
    gpu_utilization: Optional[float] = None
    molecules_processed: Optional[int] = None
    error_message: Optional[str] = None
    region: Optional[str] = None


@dataclass
class ResourceMetrics:
    """Container for resource utilization metrics."""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    gpu_percent: Optional[float] = None
    network_io: Optional[float] = None
    disk_io: Optional[float] = None


@dataclass
class CostBreakdown:
    """Container for cost analysis data."""

    platform: str
    service_type: str
    cost_usd: float
    usage_hours: float
    cost_per_hour: float
    region: str


class CloudJobMonitor:
    """
    Interactive dashboards for cloud job monitoring and management.

    Features:
    - Real-time job status monitoring
    - Resource utilization tracking
    - Cost analysis and optimization
    - Performance metrics visualization
    - Multi-platform comparison
    - Predictive cost modeling
    """

    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize the cloud monitor.

        Args:
            theme: Plotly theme for styling
        """
        self.theme = theme
        self.colors = px.colors.qualitative.Set1
        self.platform_colors = {"aws": "#FF9900", "gcp": "#4285F4", "azure": "#0078D4"}

    def create_job_status_dashboard(
        self, jobs: List[CloudJob], title: str = "Cloud Job Status Dashboard"
    ) -> go.Figure:
        """
        Create job status overview dashboard.

        Args:
            jobs: List of CloudJob objects
            title: Dashboard title

        Returns:
            Plotly figure object
        """
        if not jobs:
            return go.Figure().add_annotation(
                text="No job data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=[
                "Job Status Distribution",
                "Jobs by Platform",
                "Instance Types",
                "Job Duration Distribution",
                "Cost by Platform",
                "Success Rate by Platform",
            ],
            specs=[
                [{"type": "pie"}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "bar"}, {"type": "bar"}],
            ],
        )

        # 1. Job Status Distribution
        status_counts = {}
        for job in jobs:
            status_counts[job.status] = status_counts.get(job.status, 0) + 1

        fig.add_trace(
            go.Pie(
                labels=list(status_counts.keys()),
                values=list(status_counts.values()),
                name="Status",
                marker=dict(colors=["green", "red", "orange", "blue"]),
            ),
            row=1,
            col=1,
        )

        # 2. Jobs by Platform
        platform_counts = {}
        for job in jobs:
            platform_counts[job.platform] = platform_counts.get(job.platform, 0) + 1

        fig.add_trace(
            go.Bar(
                x=list(platform_counts.keys()),
                y=list(platform_counts.values()),
                name="Platform",
                marker_color=[
                    self.platform_colors.get(p, "gray") for p in platform_counts.keys()
                ],
            ),
            row=1,
            col=2,
        )

        # 3. Instance Types
        instance_counts = {}
        for job in jobs:
            instance_counts[job.instance_type] = (
                instance_counts.get(job.instance_type, 0) + 1
            )

        # Limit to top 10 instance types
        top_instances = sorted(
            instance_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]

        fig.add_trace(
            go.Bar(
                x=[x[0] for x in top_instances],
                y=[x[1] for x in top_instances],
                name="Instance Types",
                marker_color=self.colors[2],
            ),
            row=1,
            col=3,
        )

        # 4. Job Duration Distribution
        durations = [
            job.duration_minutes for job in jobs if job.duration_minutes is not None
        ]
        if durations:
            fig.add_trace(
                go.Histogram(
                    x=durations, nbinsx=30, name="Duration", marker_color=self.colors[3]
                ),
                row=2,
                col=1,
            )

        # 5. Cost by Platform
        platform_costs = {}
        for job in jobs:
            if job.cost_usd is not None:
                platform_costs[job.platform] = (
                    platform_costs.get(job.platform, 0) + job.cost_usd
                )

        if platform_costs:
            fig.add_trace(
                go.Bar(
                    x=list(platform_costs.keys()),
                    y=list(platform_costs.values()),
                    name="Cost",
                    marker_color=[
                        self.platform_colors.get(p, "gray")
                        for p in platform_costs.keys()
                    ],
                ),
                row=2,
                col=2,
            )

        # 6. Success Rate by Platform
        platform_success = {}
        platform_total = {}

        for job in jobs:
            platform_total[job.platform] = platform_total.get(job.platform, 0) + 1
            if job.status == "completed":
                platform_success[job.platform] = (
                    platform_success.get(job.platform, 0) + 1
                )

        success_rates = {
            p: (platform_success.get(p, 0) / platform_total[p]) * 100
            for p in platform_total.keys()
        }

        fig.add_trace(
            go.Bar(
                x=list(success_rates.keys()),
                y=list(success_rates.values()),
                name="Success Rate (%)",
                marker_color=[
                    self.platform_colors.get(p, "gray") for p in success_rates.keys()
                ],
            ),
            row=2,
            col=3,
        )

        fig.update_layout(
            title=title, template=self.theme, width=1200, height=800, showlegend=False
        )

        return fig

    def create_resource_utilization_plot(
        self,
        jobs: List[CloudJob],
        resource_metrics: Optional[List[ResourceMetrics]] = None,
        title: str = "Resource Utilization Analysis",
    ) -> go.Figure:
        """
        Create resource utilization visualization.

        Args:
            jobs: List of CloudJob objects
            resource_metrics: Optional detailed resource metrics
            title: Plot title

        Returns:
            Plotly figure object
        """
        if resource_metrics:
            # Time series plot of resource metrics
            fig = make_subplots(
                rows=3,
                cols=1,
                subplot_titles=[
                    "CPU Utilization",
                    "Memory Utilization",
                    "GPU Utilization",
                ],
                shared_xaxes=True,
                vertical_spacing=0.05,
            )

            timestamps = [m.timestamp for m in resource_metrics]
            cpu_values = [m.cpu_percent for m in resource_metrics]
            memory_values = [m.memory_percent for m in resource_metrics]
            gpu_values = [
                m.gpu_percent for m in resource_metrics if m.gpu_percent is not None
            ]

            # CPU utilization
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=cpu_values,
                    mode="lines",
                    name="CPU %",
                    line=dict(color="blue", width=2),
                    fill="tonexty",
                ),
                row=1,
                col=1,
            )

            # Memory utilization
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=memory_values,
                    mode="lines",
                    name="Memory %",
                    line=dict(color="green", width=2),
                    fill="tonexty",
                ),
                row=2,
                col=1,
            )

            # GPU utilization (if available)
            if gpu_values:
                gpu_timestamps = [
                    m.timestamp for m in resource_metrics if m.gpu_percent is not None
                ]
                fig.add_trace(
                    go.Scatter(
                        x=gpu_timestamps,
                        y=gpu_values,
                        mode="lines",
                        name="GPU %",
                        line=dict(color="red", width=2),
                        fill="tonexty",
                    ),
                    row=3,
                    col=1,
                )
        else:
            # Use job-level utilization data
            fig = make_subplots(
                rows=1,
                cols=3,
                subplot_titles=[
                    "CPU Utilization",
                    "Memory Utilization",
                    "GPU Utilization",
                ],
            )

            # CPU utilization distribution
            cpu_utils = [
                job.cpu_utilization for job in jobs if job.cpu_utilization is not None
            ]
            if cpu_utils:
                fig.add_trace(
                    go.Histogram(
                        x=cpu_utils,
                        nbinsx=20,
                        name="CPU",
                        marker_color="blue",
                        opacity=0.7,
                    ),
                    row=1,
                    col=1,
                )

            # Memory utilization distribution
            mem_utils = [
                job.memory_utilization
                for job in jobs
                if job.memory_utilization is not None
            ]
            if mem_utils:
                fig.add_trace(
                    go.Histogram(
                        x=mem_utils,
                        nbinsx=20,
                        name="Memory",
                        marker_color="green",
                        opacity=0.7,
                    ),
                    row=1,
                    col=2,
                )

            # GPU utilization distribution
            gpu_utils = [
                job.gpu_utilization for job in jobs if job.gpu_utilization is not None
            ]
            if gpu_utils:
                fig.add_trace(
                    go.Histogram(
                        x=gpu_utils,
                        nbinsx=20,
                        name="GPU",
                        marker_color="red",
                        opacity=0.7,
                    ),
                    row=1,
                    col=3,
                )

        fig.update_layout(
            title=title, template=self.theme, width=1000, height=600, showlegend=False
        )

        return fig

    def create_cost_analysis_dashboard(
        self,
        jobs: List[CloudJob],
        cost_breakdown: Optional[List[CostBreakdown]] = None,
        title: str = "Cost Analysis Dashboard",
    ) -> go.Figure:
        """
        Create comprehensive cost analysis dashboard.

        Args:
            jobs: List of CloudJob objects
            cost_breakdown: Optional detailed cost breakdown
            title: Dashboard title

        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Cost Over Time",
                "Cost by Platform",
                "Cost per Molecule",
                "Cost Efficiency by Instance Type",
            ],
        )

        # 1. Cost Over Time
        jobs_with_cost = [
            job
            for job in jobs
            if job.cost_usd is not None and job.start_time is not None
        ]
        if jobs_with_cost:
            jobs_with_cost.sort(key=lambda x: x.start_time)

            dates = [job.start_time.date() for job in jobs_with_cost]
            daily_costs = {}

            for job in jobs_with_cost:
                date = job.start_time.date()
                daily_costs[date] = daily_costs.get(date, 0) + job.cost_usd

            fig.add_trace(
                go.Scatter(
                    x=list(daily_costs.keys()),
                    y=list(daily_costs.values()),
                    mode="lines+markers",
                    name="Daily Cost",
                    line=dict(color="blue", width=2),
                    marker=dict(size=6),
                ),
                row=1,
                col=1,
            )

        # 2. Cost by Platform
        platform_costs = {}
        for job in jobs:
            if job.cost_usd is not None:
                platform_costs[job.platform] = (
                    platform_costs.get(job.platform, 0) + job.cost_usd
                )

        if platform_costs:
            fig.add_trace(
                go.Bar(
                    x=list(platform_costs.keys()),
                    y=list(platform_costs.values()),
                    name="Platform Cost",
                    marker_color=[
                        self.platform_colors.get(p, "gray")
                        for p in platform_costs.keys()
                    ],
                ),
                row=1,
                col=2,
            )

        # 3. Cost per Molecule
        cost_per_molecule = []
        job_names = []

        for job in jobs:
            if (
                job.cost_usd is not None
                and job.molecules_processed is not None
                and job.molecules_processed > 0
            ):
                cost_per_molecule.append(job.cost_usd / job.molecules_processed)
                job_names.append(
                    job.job_name[:15] + "..."
                    if len(job.job_name) > 15
                    else job.job_name
                )

        if cost_per_molecule:
            fig.add_trace(
                go.Scatter(
                    x=job_names,
                    y=cost_per_molecule,
                    mode="markers",
                    name="Cost/Molecule",
                    marker=dict(
                        size=8,
                        color=cost_per_molecule,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="$/molecule"),
                    ),
                ),
                row=2,
                col=1,
            )

        # 4. Cost Efficiency by Instance Type
        instance_efficiency = {}
        instance_counts = {}

        for job in jobs:
            if (
                job.cost_usd is not None
                and job.molecules_processed is not None
                and job.molecules_processed > 0
            ):
                efficiency = (
                    job.molecules_processed / job.cost_usd
                )  # molecules per dollar
                instance_type = job.instance_type

                if instance_type not in instance_efficiency:
                    instance_efficiency[instance_type] = []
                instance_efficiency[instance_type].append(efficiency)

        # Calculate average efficiency per instance type
        avg_efficiency = {k: np.mean(v) for k, v in instance_efficiency.items()}

        if avg_efficiency:
            fig.add_trace(
                go.Bar(
                    x=list(avg_efficiency.keys()),
                    y=list(avg_efficiency.values()),
                    name="Molecules/$",
                    marker_color=self.colors[3],
                ),
                row=2,
                col=2,
            )

        fig.update_layout(
            title=title, template=self.theme, width=1000, height=700, showlegend=False
        )

        return fig

    def create_performance_metrics_plot(
        self, jobs: List[CloudJob], title: str = "Performance Metrics Analysis"
    ) -> go.Figure:
        """
        Create performance metrics visualization.

        Args:
            jobs: List of CloudJob objects
            title: Plot title

        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Throughput (Molecules/Hour)",
                "Duration vs Cost",
                "Platform Performance Comparison",
                "Instance Type Efficiency",
            ],
        )

        # 1. Throughput calculation
        throughputs = []
        job_names = []
        platforms = []

        for job in jobs:
            if (
                job.molecules_processed is not None
                and job.duration_minutes is not None
                and job.duration_minutes > 0
            ):
                throughput = (
                    job.molecules_processed / job.duration_minutes
                ) * 60  # molecules per hour
                throughputs.append(throughput)
                job_names.append(job.job_name)
                platforms.append(job.platform)

        if throughputs:
            colors = [self.platform_colors.get(p, "gray") for p in platforms]

            fig.add_trace(
                go.Scatter(
                    x=job_names,
                    y=throughputs,
                    mode="markers",
                    name="Throughput",
                    marker=dict(
                        size=10, color=colors, line=dict(width=1, color="white")
                    ),
                    text=platforms,
                    hovertemplate="Job: %{x}<br>Throughput: %{y:.1f} mol/hr<br>Platform: %{text}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # 2. Duration vs Cost scatter
        durations = []
        costs = []
        hover_text = []

        for job in jobs:
            if job.duration_minutes is not None and job.cost_usd is not None:
                durations.append(job.duration_minutes)
                costs.append(job.cost_usd)
                hover_text.append(
                    f"Job: {job.job_name}<br>Platform: {job.platform}<br>Instance: {job.instance_type}"
                )

        if durations and costs:
            fig.add_trace(
                go.Scatter(
                    x=durations,
                    y=costs,
                    mode="markers",
                    name="Duration vs Cost",
                    marker=dict(size=8, color="blue", opacity=0.7),
                    hovertemplate="%{hovertext}<br>Duration: %{x:.1f} min<br>Cost: $%{y:.2f}<extra></extra>",
                    hovertext=hover_text,
                ),
                row=1,
                col=2,
            )

        # 3. Platform Performance Comparison
        platform_performance = {}

        for job in jobs:
            if (
                job.molecules_processed is not None
                and job.duration_minutes is not None
                and job.duration_minutes > 0
            ):
                throughput = (job.molecules_processed / job.duration_minutes) * 60

                if job.platform not in platform_performance:
                    platform_performance[job.platform] = []
                platform_performance[job.platform].append(throughput)

        if platform_performance:
            platforms = list(platform_performance.keys())
            avg_performance = [np.mean(platform_performance[p]) for p in platforms]

            fig.add_trace(
                go.Bar(
                    x=platforms,
                    y=avg_performance,
                    name="Avg Performance",
                    marker_color=[
                        self.platform_colors.get(p, "gray") for p in platforms
                    ],
                ),
                row=2,
                col=1,
            )

        # 4. Instance Type Efficiency
        instance_performance = {}

        for job in jobs:
            if (
                job.molecules_processed is not None
                and job.cost_usd is not None
                and job.cost_usd > 0
            ):
                efficiency = job.molecules_processed / job.cost_usd

                if job.instance_type not in instance_performance:
                    instance_performance[job.instance_type] = []
                instance_performance[job.instance_type].append(efficiency)

        # Get top 8 instance types by efficiency
        if instance_performance:
            avg_instance_efficiency = {
                k: np.mean(v) for k, v in instance_performance.items()
            }
            top_instances = sorted(
                avg_instance_efficiency.items(), key=lambda x: x[1], reverse=True
            )[:8]

            fig.add_trace(
                go.Bar(
                    x=[x[0] for x in top_instances],
                    y=[x[1] for x in top_instances],
                    name="Efficiency",
                    marker_color=self.colors[0],
                ),
                row=2,
                col=2,
            )

        fig.update_layout(
            title=title, template=self.theme, width=1000, height=700, showlegend=False
        )

        return fig

    def create_predictive_cost_model(
        self, jobs: List[CloudJob], title: str = "Predictive Cost Modeling"
    ) -> go.Figure:
        """
        Create predictive cost model visualization.

        Args:
            jobs: List of CloudJob objects
            title: Plot title

        Returns:
            Plotly figure object
        """
        # Extract features for cost prediction
        features = []
        costs = []
        job_info = []

        for job in jobs:
            if (
                job.cost_usd is not None
                and job.duration_minutes is not None
                and job.molecules_processed is not None
            ):
                features.append(
                    [
                        job.duration_minutes,
                        job.molecules_processed,
                        1 if "gpu" in job.instance_type.lower() else 0,  # GPU flag
                    ]
                )
                costs.append(job.cost_usd)
                job_info.append(f"{job.platform} - {job.instance_type}")

        if len(features) < 5:
            return go.Figure().add_annotation(
                text="Insufficient data for cost modeling",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        features = np.array(features)
        costs = np.array(costs)

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Cost vs Duration",
                "Cost vs Molecules Processed",
                "Predicted vs Actual Cost",
                "Cost Distribution by Platform",
            ],
        )

        # 1. Cost vs Duration
        fig.add_trace(
            go.Scatter(
                x=features[:, 0],  # duration
                y=costs,
                mode="markers",
                name="Duration",
                marker=dict(size=8, color="blue", opacity=0.7),
                text=job_info,
                hovertemplate="Duration: %{x:.1f} min<br>Cost: $%{y:.2f}<br>%{text}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Add trend line
        if len(features) >= 3:
            z = np.polyfit(features[:, 0], costs, 1)
            x_trend = np.linspace(features[:, 0].min(), features[:, 0].max(), 100)
            y_trend = np.polyval(z, x_trend)

            fig.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=y_trend,
                    mode="lines",
                    name="Trend",
                    line=dict(color="red", width=2, dash="dash"),
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

        # 2. Cost vs Molecules Processed
        fig.add_trace(
            go.Scatter(
                x=features[:, 1],  # molecules
                y=costs,
                mode="markers",
                name="Molecules",
                marker=dict(size=8, color="green", opacity=0.7),
                text=job_info,
                hovertemplate="Molecules: %{x:,}<br>Cost: $%{y:.2f}<br>%{text}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # 3. Simple linear prediction model
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                features, costs, test_size=0.3, random_state=42
            )

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)

            fig.add_trace(
                go.Scatter(
                    x=y_test,
                    y=y_pred,
                    mode="markers",
                    name="Predictions",
                    marker=dict(size=8, color="purple", opacity=0.7),
                    hovertemplate="Actual: $%{x:.2f}<br>Predicted: $%{y:.2f}<extra></extra>",
                ),
                row=2,
                col=1,
            )

            # Add perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    name=f"Perfect (R² = {r2:.3f})",
                    line=dict(color="red", width=2, dash="dash"),
                    hoverinfo="skip",
                ),
                row=2,
                col=1,
            )

        except ImportError:
            fig.add_annotation(
                text="scikit-learn not available for modeling",
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=0.5,
                showarrow=False,
                row=2,
                col=1,
            )

        # 4. Cost Distribution by Platform
        platform_costs = {}
        for job in jobs:
            if job.cost_usd is not None:
                if job.platform not in platform_costs:
                    platform_costs[job.platform] = []
                platform_costs[job.platform].append(job.cost_usd)

        for platform, costs_list in platform_costs.items():
            fig.add_trace(
                go.Box(
                    y=costs_list,
                    name=platform,
                    marker_color=self.platform_colors.get(platform, "gray"),
                    showlegend=False,
                ),
                row=2,
                col=2,
            )

        fig.update_layout(
            title=title, template=self.theme, width=1000, height=700, showlegend=False
        )

        return fig

    def create_comprehensive_monitoring_dashboard(
        self,
        jobs: List[CloudJob],
        resource_metrics: Optional[List[ResourceMetrics]] = None,
        cost_breakdown: Optional[List[CostBreakdown]] = None,
    ) -> Dict[str, go.Figure]:
        """
        Create comprehensive cloud monitoring dashboard.

        Args:
            jobs: List of CloudJob objects
            resource_metrics: Optional resource utilization data
            cost_breakdown: Optional cost breakdown data

        Returns:
            Dictionary of figure objects
        """
        dashboard = {}

        # Job status overview
        dashboard["status"] = self.create_job_status_dashboard(jobs)

        # Resource utilization
        dashboard["resources"] = self.create_resource_utilization_plot(
            jobs, resource_metrics
        )

        # Cost analysis
        dashboard["costs"] = self.create_cost_analysis_dashboard(jobs, cost_breakdown)

        # Performance metrics
        dashboard["performance"] = self.create_performance_metrics_plot(jobs)

        # Predictive modeling
        dashboard["predictions"] = self.create_predictive_cost_model(jobs)

        return dashboard

    def export_monitoring_dashboard_html(
        self,
        dashboard: Dict[str, go.Figure],
        filename: str = "cloud_monitoring_dashboard.html",
    ):
        """
        Export monitoring dashboard as HTML file.

        Args:
            dashboard: Dictionary of figure objects
            filename: Output filename
        """
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
        <title>Cloud Job Monitoring Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .dashboard-header {
            background: rgba(255,255,255,0.95);
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .dashboard-container {
            max-width: 1400px;
            margin: 20px auto;
            padding: 0 20px;
        }
        .plot-container {
            background: rgba(255,255,255,0.95);
            margin: 20px 0;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        h1 {
            color: #2c3e50;
            font-size: 2.8em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        h2 {
            color: #34495e;
            font-size: 1.6em;
            margin-bottom: 15px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }
        .status-indicators {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .status-card {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .refresh-btn {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .refresh-btn:hover {
            background: #2980b9;
            transform: translateY(-2px);
        }
        </style>
        </head>
        <body>
        <button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button>
        <div class="dashboard-header">
        <h1>☁️ Cloud Job Monitoring Dashboard</h1>
        <p>Real-time monitoring and analysis of cloud computing jobs across multiple platforms</p>
        </div>
        <div class="dashboard-container">
        """

        plot_descriptions = {
            "status": "Overview of job statuses, platform distribution, and success rates.",
            "resources": "Resource utilization analysis including CPU, memory, and GPU usage.",
            "costs": "Comprehensive cost analysis and optimization insights.",
            "performance": "Performance metrics including throughput and efficiency analysis.",
            "predictions": "Predictive cost modeling and forecasting capabilities.",
        }

        for plot_name, fig in dashboard.items():
            plot_title = plot_name.replace("_", " ").title()
            description = plot_descriptions.get(
                plot_name, "Detailed analysis and monitoring."
            )

            html_content += '<div class="plot-container">'
            html_content += f"<h2>📊 {plot_title}</h2>"
            html_content += f'<p style="color: #7f8c8d; font-style: italic; margin-bottom: 20px;">{description}</p>'
            html_content += fig.to_html(include_plotlyjs=False, div_id=plot_name)
            html_content += "</div>"

        html_content += """
        </div>
        <script>
        // Auto-refresh every 5 minutes
        setTimeout(function(){
            location.reload();
        }, 300000);
        </script>
        </body>
        </html>
        """

        with open(filename, "w") as f:
            f.write(html_content)

        print(f"Cloud monitoring dashboard exported to {filename}")
