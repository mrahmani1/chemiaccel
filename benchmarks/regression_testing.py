"""
Automated Performance Regression Detection System.

This module provides advanced infrastructure for detecting performance regressions
across ChemiAccel versions, tracking performance trends, and automatically
alerting when significant performance degradations are detected.
"""

import hashlib
import json
import logging
import sqlite3
import statistics
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
    np = None

try:
    from scipy import stats
    from scipy.stats import kruskal, mannwhitneyu

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from ..src.chemi_accel.utils.logging import get_logger
except ImportError:
    # Fallback for standalone execution
    def get_logger():
        return logging.getLogger(__name__)


logger = get_logger()


class RegressionSeverity(Enum):
    """Severity levels for performance regressions."""

    CRITICAL = "critical"  # >50% degradation
    MAJOR = "major"  # 25-50% degradation
    MODERATE = "moderate"  # 10-25% degradation
    MINOR = "minor"  # 5-10% degradation
    NEGLIGIBLE = "negligible"  # <5% degradation


class RegressionStatus(Enum):
    """Status of regression detection."""

    DETECTED = "detected"
    INVESTIGATING = "investigating"
    CONFIRMED = "confirmed"
    FALSE_POSITIVE = "false_positive"
    RESOLVED = "resolved"


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression testing."""

    implementation: str
    backend: str
    descriptor: str
    dataset_size: int
    baseline_time: float
    baseline_throughput: float
    baseline_memory_mb: float
    version: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "implementation": self.implementation,
            "backend": self.backend,
            "descriptor": self.descriptor,
            "dataset_size": self.dataset_size,
            "baseline_time": self.baseline_time,
            "baseline_throughput": self.baseline_throughput,
            "baseline_memory_mb": self.baseline_memory_mb,
            "version": self.version,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceBaseline":
        """Create from dictionary."""
        return cls(
            implementation=data["implementation"],
            backend=data["backend"],
            descriptor=data["descriptor"],
            dataset_size=data["dataset_size"],
            baseline_time=data["baseline_time"],
            baseline_throughput=data["baseline_throughput"],
            baseline_memory_mb=data["baseline_memory_mb"],
            version=data["version"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class RegressionResult:
    """Result of a regression test."""

    test_name: str
    implementation: str
    backend: str
    descriptor: str
    dataset_size: int
    baseline: PerformanceBaseline
    current_time: float
    current_throughput: float
    current_memory_mb: float
    time_change_percent: float
    throughput_change_percent: float
    memory_change_percent: float
    is_regression: bool
    regression_severity: str  # "none", "minor", "major", "critical"
    confidence_level: float
    test_timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceDatabase:
    """SQLite database for storing performance metrics and regression history."""

    def __init__(self, db_path: str = "performance_metrics.db"):
        """Initialize performance database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Performance metrics table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    implementation TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    descriptor TEXT NOT NULL,
                    dataset_size INTEGER NOT NULL,
                    execution_time REAL NOT NULL,
                    throughput REAL NOT NULL,
                    memory_usage_mb REAL NOT NULL,
                    version TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Regression detection table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS regression_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    test_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    time_change_percent REAL NOT NULL,
                    throughput_change_percent REAL NOT NULL,
                    memory_change_percent REAL NOT NULL,
                    confidence_level REAL NOT NULL,
                    baseline_version TEXT NOT NULL,
                    current_version TEXT,
                    alert_sent BOOLEAN DEFAULT FALSE,
                    resolved_at TIMESTAMP,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Performance trends table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS performance_trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_type TEXT NOT NULL,
                    implementation TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    descriptor TEXT NOT NULL,
                    trend_direction TEXT NOT NULL,
                    trend_magnitude REAL NOT NULL,
                    trend_confidence REAL NOT NULL,
                    data_points INTEGER NOT NULL,
                    time_period_days INTEGER NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.commit()
            logger.info(f"Initialized performance database at {self.db_path}")

    def store_performance_metric(
        self,
        implementation: str,
        backend: str,
        descriptor: str,
        dataset_size: int,
        execution_time: float,
        throughput: float,
        memory_usage_mb: float,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store performance metric in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO performance_metrics
                (timestamp, implementation, backend, descriptor, dataset_size,
                 execution_time, throughput, memory_usage_mb, version, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    datetime.now().isoformat(),
                    implementation,
                    backend,
                    descriptor,
                    dataset_size,
                    execution_time,
                    throughput,
                    memory_usage_mb,
                    version,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            conn.commit()

    def store_regression_detection(
        self,
        test_name: str,
        severity: RegressionSeverity,
        status: RegressionStatus,
        time_change_percent: float,
        throughput_change_percent: float,
        memory_change_percent: float,
        confidence_level: float,
        baseline_version: str,
        current_version: str = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Store regression detection in database.

        Returns:
            ID of inserted regression record
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO regression_detections
                (timestamp, test_name, severity, status, time_change_percent,
                 throughput_change_percent, memory_change_percent, confidence_level,
                 baseline_version, current_version, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    datetime.now().isoformat(),
                    test_name,
                    severity.value,
                    status.value,
                    time_change_percent,
                    throughput_change_percent,
                    memory_change_percent,
                    confidence_level,
                    baseline_version,
                    current_version,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_performance_history(
        self,
        implementation: str,
        backend: str,
        descriptor: str,
        dataset_size: int,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get performance history for trend analysis."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT timestamp, execution_time, throughput, memory_usage_mb, version
                FROM performance_metrics
                WHERE implementation = ? AND backend = ? AND descriptor = ?
                  AND dataset_size = ? AND created_at >= datetime('now', '-{days} days')
                ORDER BY created_at ASC
            """,
                (implementation, backend, descriptor, dataset_size),
            )

            columns = [
                "timestamp",
                "execution_time",
                "throughput",
                "memory_usage_mb",
                "version",
            ]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_active_regressions(
        self, severity_filter: Optional[List[RegressionSeverity]] = None
    ) -> List[Dict[str, Any]]:
        """Get active regression detections."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            query = """
                SELECT id, test_name, severity, status, time_change_percent,
                       confidence_level, baseline_version, created_at
                FROM regression_detections
                WHERE status IN ('detected', 'investigating', 'confirmed')
                  AND resolved_at IS NULL
            """

            if severity_filter:
                severity_values = [s.value for s in severity_filter]
                placeholders = ",".join(["?"] * len(severity_values))
                query += f" AND severity IN ({placeholders})"
                cursor.execute(query, severity_values)
            else:
                cursor.execute(query)

            columns = [
                "id",
                "test_name",
                "severity",
                "status",
                "time_change_percent",
                "confidence_level",
                "baseline_version",
                "created_at",
            ]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]


class RegressionDetector:
    """Advanced regression detection with statistical analysis and alerting."""

    def __init__(
        self,
        database: PerformanceDatabase,
        confidence_threshold: float = 0.95,
        min_data_points: int = 5,
    ):
        """Initialize regression detector.

        Args:
            database: Performance database instance
            confidence_threshold: Statistical confidence threshold
            min_data_points: Minimum data points for trend analysis
        """
        self.database = database
        self.confidence_threshold = confidence_threshold
        self.min_data_points = min_data_points

        # Regression severity thresholds
        self.severity_thresholds = {
            RegressionSeverity.CRITICAL: 0.50,  # 50%+ degradation
            RegressionSeverity.MAJOR: 0.25,  # 25-50% degradation
            RegressionSeverity.MODERATE: 0.10,  # 10-25% degradation
            RegressionSeverity.MINOR: 0.05,  # 5-10% degradation
            RegressionSeverity.NEGLIGIBLE: 0.0,  # <5% degradation
        }

    def detect_regression(
        self,
        current_metrics: Dict[str, float],
        historical_data: List[Dict[str, Any]],
        test_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Detect performance regression using statistical analysis.

        Args:
            current_metrics: Current performance metrics
            historical_data: Historical performance data
            test_name: Name of the test

        Returns:
            Regression detection result or None if no regression
        """
        if len(historical_data) < self.min_data_points:
            logger.warning(
                f"Insufficient historical data for {test_name}: {len(historical_data)} points"
            )
            return None

        # Extract time series data
        historical_times = [float(h["execution_time"]) for h in historical_data]
        historical_throughputs = [float(h["throughput"]) for h in historical_data]
        historical_memory = [float(h["memory_usage_mb"]) for h in historical_data]

        # Calculate baseline statistics
        baseline_time = statistics.mean(historical_times)
        baseline_throughput = statistics.mean(historical_throughputs)
        baseline_memory = statistics.mean(historical_memory)

        # Calculate changes
        time_change = (
            current_metrics["execution_time"] - baseline_time
        ) / baseline_time
        throughput_change = (
            current_metrics["throughput"] - baseline_throughput
        ) / baseline_throughput
        memory_change = (
            current_metrics["memory_usage_mb"] - baseline_memory
        ) / baseline_memory

        # Statistical significance testing
        confidence_level = self._calculate_statistical_confidence(
            current_metrics["execution_time"], historical_times
        )

        # Determine regression severity
        severity = self._determine_severity(time_change)

        # Only report if statistically significant and above negligible threshold
        if (
            confidence_level >= self.confidence_threshold
            and severity != RegressionSeverity.NEGLIGIBLE
        ):
            return {
                "test_name": test_name,
                "severity": severity,
                "time_change_percent": time_change * 100,
                "throughput_change_percent": throughput_change * 100,
                "memory_change_percent": memory_change * 100,
                "confidence_level": confidence_level,
                "baseline_metrics": {
                    "execution_time": baseline_time,
                    "throughput": baseline_throughput,
                    "memory_usage_mb": baseline_memory,
                },
                "current_metrics": current_metrics,
                "historical_data_points": len(historical_data),
            }

        return None

    def _calculate_statistical_confidence(
        self, current_value: float, historical_values: List[float]
    ) -> float:
        """Calculate statistical confidence using appropriate tests."""
        if not SCIPY_AVAILABLE:
            # Fallback: simple statistical confidence based on standard deviations
            if len(historical_values) < 2:
                return 0.5

            mean_historical = statistics.mean(historical_values)
            std_historical = statistics.stdev(historical_values)

            if std_historical == 0:
                return 1.0 if current_value == mean_historical else 0.0

            # Z-score based confidence
            z_score = abs(current_value - mean_historical) / std_historical
            return min(0.99, z_score / 3.0)  # Rough approximation

        try:
            # Use Mann-Whitney U test for non-parametric comparison
            current_sample = [current_value] * 3  # Simulate multiple measurements
            statistic, p_value = mannwhitneyu(
                current_sample, historical_values, alternative="two-sided"
            )
            return 1.0 - p_value
        except Exception:
            # Fallback to simple confidence calculation
            return 0.75

    def _determine_severity(self, change_percent: float) -> RegressionSeverity:
        """Determine regression severity based on performance change."""
        abs_change = abs(change_percent)

        if abs_change >= self.severity_thresholds[RegressionSeverity.CRITICAL]:
            return RegressionSeverity.CRITICAL
        elif abs_change >= self.severity_thresholds[RegressionSeverity.MAJOR]:
            return RegressionSeverity.MAJOR
        elif abs_change >= self.severity_thresholds[RegressionSeverity.MODERATE]:
            return RegressionSeverity.MODERATE
        elif abs_change >= self.severity_thresholds[RegressionSeverity.MINOR]:
            return RegressionSeverity.MINOR
        else:
            return RegressionSeverity.NEGLIGIBLE

    def analyze_performance_trends(
        self,
        implementation: str,
        backend: str,
        descriptor: str,
        dataset_size: int,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Analyze performance trends over time.

        Args:
            implementation: Implementation name
            backend: Backend name
            descriptor: Descriptor name
            dataset_size: Dataset size
            days: Number of days to analyze

        Returns:
            Trend analysis results
        """
        historical_data = self.database.get_performance_history(
            implementation, backend, descriptor, dataset_size, days
        )

        if len(historical_data) < self.min_data_points:
            return {
                "sufficient_data": False,
                "data_points": len(historical_data),
                "message": "Insufficient data for trend analysis",
            }

        # Extract time series
        times = [
            datetime.fromisoformat(h["timestamp"]).timestamp() for h in historical_data
        ]
        execution_times = [float(h["execution_time"]) for h in historical_data]
        throughputs = [float(h["throughput"]) for h in historical_data]
        memory_usage = [float(h["memory_usage_mb"]) for h in historical_data]

        # Calculate trends
        time_trend = self._calculate_trend(times, execution_times)
        throughput_trend = self._calculate_trend(times, throughputs)
        memory_trend = self._calculate_trend(times, memory_usage)

        return {
            "sufficient_data": True,
            "data_points": len(historical_data),
            "time_period_days": days,
            "execution_time_trend": time_trend,
            "throughput_trend": throughput_trend,
            "memory_trend": memory_trend,
            "overall_health": self._assess_overall_health(
                time_trend, throughput_trend, memory_trend
            ),
        }

    def _calculate_trend(
        self, timestamps: List[float], values: List[float]
    ) -> Dict[str, Any]:
        """Calculate trend statistics for a time series."""
        if not SCIPY_AVAILABLE or len(timestamps) < 3:
            # Simple trend calculation
            if len(values) < 2:
                return {"direction": "stable", "magnitude": 0.0, "confidence": 0.0}

            first_half = values[: len(values) // 2]
            second_half = values[len(values) // 2 :]

            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)

            change = (second_avg - first_avg) / first_avg if first_avg != 0 else 0

            if abs(change) < 0.05:
                direction = "stable"
            elif change > 0:
                direction = "increasing"
            else:
                direction = "decreasing"

            return {
                "direction": direction,
                "magnitude": abs(change),
                "confidence": 0.7,  # Simple confidence estimate
            }

        try:
            # Linear regression for trend analysis
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                timestamps, values
            )

            # Determine trend direction and significance
            if abs(slope) < std_err * 2:  # Not statistically significant
                direction = "stable"
                magnitude = 0.0
            elif slope > 0:
                direction = "increasing"
                magnitude = abs(slope)
            else:
                direction = "decreasing"
                magnitude = abs(slope)

            confidence = abs(r_value)  # Correlation coefficient as confidence measure

            return {
                "direction": direction,
                "magnitude": magnitude,
                "confidence": confidence,
                "r_squared": r_value**2,
                "p_value": p_value,
            }
        except Exception as e:
            logger.warning(f"Trend calculation failed: {e}")
            return {"direction": "unknown", "magnitude": 0.0, "confidence": 0.0}

    def _assess_overall_health(
        self,
        time_trend: Dict[str, Any],
        throughput_trend: Dict[str, Any],
        memory_trend: Dict[str, Any],
    ) -> str:
        """Assess overall performance health based on trends."""
        # Performance is degrading if execution time is increasing
        # or throughput is decreasing significantly

        time_degrading = (
            time_trend["direction"] == "increasing"
            and time_trend["confidence"] > 0.7
            and time_trend["magnitude"] > 0.1
        )

        throughput_degrading = (
            throughput_trend["direction"] == "decreasing"
            and throughput_trend["confidence"] > 0.7
            and throughput_trend["magnitude"] > 0.1
        )

        memory_increasing = (
            memory_trend["direction"] == "increasing"
            and memory_trend["confidence"] > 0.7
            and memory_trend["magnitude"] > 0.2
        )

        if time_degrading or throughput_degrading:
            return "degrading"
        elif memory_increasing:
            return "concerning"
        elif (
            time_trend["direction"] == "decreasing" and time_trend["confidence"] > 0.7
        ) or (
            throughput_trend["direction"] == "increasing"
            and throughput_trend["confidence"] > 0.7
        ):
            return "improving"
        else:
            return "stable"


class RegressionTestSuite:
    """Comprehensive regression testing suite with automated detection."""

    def __init__(
        self,
        baseline_directory: str = "performance_baselines",
        results_directory: str = "regression_results",
        database_path: str = "performance_metrics.db",
    ):
        """Initialize regression test suite.

        Args:
            baseline_directory: Directory to store baseline data
            results_directory: Directory to store regression test results
            database_path: Path to performance metrics database
        """
        self.baseline_dir = Path(baseline_directory)
        self.results_dir = Path(results_directory)

        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.regression_results: List[RegressionResult] = []

        # Initialize database and detector
        self.database = PerformanceDatabase(database_path)
        self.detector = RegressionDetector(self.database)

        # Regression thresholds (legacy compatibility)
        self.thresholds = {
            "minor_regression": 0.10,  # 10% performance degradation
            "major_regression": 0.25,  # 25% performance degradation
            "critical_regression": 0.50,  # 50% performance degradation
        }

        # Load existing baselines
        self._load_baselines()

        logger.info(
            f"Initialized regression test suite with {len(self.baselines)} baselines"
        )

    def _load_baselines(self) -> None:
        """Load existing performance baselines."""
        baseline_file = self.baseline_dir / "baselines.json"

        if baseline_file.exists():
            try:
                with open(baseline_file) as f:
                    baseline_data = json.load(f)

                for key, data in baseline_data.items():
                    baseline = PerformanceBaseline.from_dict(data)
                    self.baselines[key] = baseline

                logger.info(f"Loaded {len(self.baselines)} performance baselines")

            except Exception as e:
                logger.error(f"Failed to load baselines: {e}")

    def _save_baselines(self) -> None:
        """Save performance baselines to disk."""
        baseline_file = self.baseline_dir / "baselines.json"

        baseline_data = {}
        for key, baseline in self.baselines.items():
            baseline_data[key] = baseline.to_dict()

        try:
            with open(baseline_file, "w") as f:
                json.dump(baseline_data, f, indent=2)

            logger.info(f"Saved {len(self.baselines)} performance baselines")

        except Exception as e:
            logger.error(f"Failed to save baselines: {e}")

    def _generate_baseline_key(
        self, implementation: str, backend: str, descriptor: str, dataset_size: int
    ) -> str:
        """Generate unique key for baseline."""
        key_string = f"{implementation}_{backend}_{descriptor}_{dataset_size}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]

    def set_baseline(
        self,
        implementation: str,
        backend: str,
        descriptor: str,
        dataset_size: int,
        execution_time: float,
        throughput: float,
        memory_usage_mb: float,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set performance baseline.

        Args:
            implementation: Implementation name
            backend: Backend name
            descriptor: Descriptor name
            dataset_size: Dataset size
            execution_time: Baseline execution time
            throughput: Baseline throughput
            memory_usage_mb: Baseline memory usage
            version: Version string
            metadata: Additional metadata
        """
        key = self._generate_baseline_key(
            implementation, backend, descriptor, dataset_size
        )

        baseline = PerformanceBaseline(
            implementation=implementation,
            backend=backend,
            descriptor=descriptor,
            dataset_size=dataset_size,
            baseline_time=execution_time,
            baseline_throughput=throughput,
            baseline_memory_mb=memory_usage_mb,
            version=version,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {},
        )

        self.baselines[key] = baseline
        self._save_baselines()

        logger.info(
            f"Set baseline for {implementation} {backend} {descriptor} "
            f"size {dataset_size}: {execution_time:.3f}s"
        )

    def run_automated_regression_test(
        self,
        implementation_func: callable,
        dataset: Any,
        implementation: str,
        backend: str,
        descriptor: str,
        dataset_size: int,
        test_name: str,
        iterations: int = 5,
        version: str = "current",
    ) -> Optional[RegressionResult]:
        """Run automated regression test with advanced detection.

        Args:
            implementation_func: Function to test
            dataset: Dataset to use
            implementation: Implementation name
            backend: Backend name
            descriptor: Descriptor name
            dataset_size: Dataset size
            test_name: Test name
            iterations: Number of test iterations
            version: Version identifier

        Returns:
            Regression test result or None if no baseline
        """
        logger.info(f"Running automated regression test: {test_name}")

        # Run performance test
        execution_times = []
        memory_usage_samples = []

        for i in range(iterations):
            # Measure memory before
            memory_before = self._get_memory_usage()

            # Time execution
            start_time = time.perf_counter()

            try:
                result = implementation_func(dataset)

                end_time = time.perf_counter()
                execution_time = end_time - start_time

                # Measure memory after
                memory_after = self._get_memory_usage()
                memory_used = memory_after - memory_before

                execution_times.append(execution_time)
                memory_usage_samples.append(memory_used)

            except Exception as e:
                logger.error(f"Regression test iteration {i} failed: {e}")
                return None

        # Calculate current metrics
        current_time = statistics.mean(execution_times)
        current_memory = statistics.mean(memory_usage_samples)
        current_throughput = dataset_size / current_time if current_time > 0 else 0.0

        current_metrics = {
            "execution_time": current_time,
            "throughput": current_throughput,
            "memory_usage_mb": current_memory,
        }

        # Store current performance data
        self.database.store_performance_metric(
            implementation,
            backend,
            descriptor,
            dataset_size,
            current_time,
            current_throughput,
            current_memory,
            version,
            metadata={
                "test_name": test_name,
                "iterations": iterations,
                "execution_times": execution_times,
                "memory_samples": memory_usage_samples,
            },
        )

        # Get historical data for regression detection
        historical_data = self.database.get_performance_history(
            implementation, backend, descriptor, dataset_size, days=30
        )

        # Detect regression
        regression_detection = self.detector.detect_regression(
            current_metrics, historical_data, test_name
        )

        if regression_detection:
            # Store regression detection
            regression_id = self.database.store_regression_detection(
                test_name=test_name,
                severity=regression_detection["severity"],
                status=RegressionStatus.DETECTED,
                time_change_percent=regression_detection["time_change_percent"],
                throughput_change_percent=regression_detection[
                    "throughput_change_percent"
                ],
                memory_change_percent=regression_detection["memory_change_percent"],
                confidence_level=regression_detection["confidence_level"],
                baseline_version="historical",
                current_version=version,
                metadata={
                    "historical_data_points": regression_detection[
                        "historical_data_points"
                    ],
                    "baseline_metrics": regression_detection["baseline_metrics"],
                },
            )

            # Send alert if critical or major regression
            if regression_detection["severity"] in [
                RegressionSeverity.CRITICAL,
                RegressionSeverity.MAJOR,
            ]:
                self._send_regression_alert(regression_detection, regression_id)

            logger.warning(
                f"Regression detected: {regression_detection['severity'].value} - "
                f"{regression_detection['time_change_percent']:.1f}% performance change"
            )
        else:
            logger.info(f"No regression detected for {test_name}")

        # Create legacy-compatible result
        baseline_key = self._generate_baseline_key(
            implementation, backend, descriptor, dataset_size
        )

        baseline = self.baselines.get(baseline_key)
        if baseline:
            # Calculate changes for legacy compatibility
            time_change = (
                (current_time - baseline.baseline_time) / baseline.baseline_time * 100
            )
            throughput_change = (
                (current_throughput - baseline.baseline_throughput)
                / baseline.baseline_throughput
                * 100
            )
            memory_change = (
                (current_memory - baseline.baseline_memory_mb)
                / baseline.baseline_memory_mb
                * 100
            )

            is_regression = abs(time_change) > self.thresholds["minor_regression"] * 100

            if time_change > self.thresholds["critical_regression"] * 100:
                severity = "critical"
            elif time_change > self.thresholds["major_regression"] * 100:
                severity = "major"
            elif time_change > self.thresholds["minor_regression"] * 100:
                severity = "minor"
            else:
                severity = "none"

            # Calculate confidence level
            std_time = (
                statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
            )
            confidence = (
                max(0.0, 1.0 - (std_time / current_time)) if current_time > 0 else 0.0
            )

            result = RegressionResult(
                test_name=test_name,
                implementation=implementation,
                backend=backend,
                descriptor=descriptor,
                dataset_size=dataset_size,
                baseline=baseline,
                current_time=current_time,
                current_throughput=current_throughput,
                current_memory_mb=current_memory,
                time_change_percent=time_change,
                throughput_change_percent=throughput_change,
                memory_change_percent=memory_change,
                is_regression=is_regression,
                regression_severity=severity,
                confidence_level=confidence,
                test_timestamp=datetime.now().isoformat(),
                metadata={
                    "iterations": iterations,
                    "execution_times": execution_times,
                    "memory_samples": memory_usage_samples,
                    "baseline_version": baseline.version,
                    "automated_detection": regression_detection is not None,
                },
            )

            self.regression_results.append(result)
            return result

        return None

    def _send_regression_alert(
        self, regression_detection: Dict[str, Any], regression_id: int
    ) -> None:
        """Send regression alert notification.

        This is a placeholder for integration with alerting systems like:
        - Email notifications
        - Slack/Teams webhooks
        - PagerDuty/Opsgenie alerts
        - Custom monitoring systems
        """
        alert_message = f"""
🚨 PERFORMANCE REGRESSION DETECTED 🚨

Test: {regression_detection['test_name']}
Severity: {regression_detection['severity'].value.upper()}
Performance Change: {regression_detection['time_change_percent']:+.1f}%
Throughput Change: {regression_detection['throughput_change_percent']:+.1f}%
Memory Change: {regression_detection['memory_change_percent']:+.1f}%
Confidence: {regression_detection['confidence_level']:.2%}

Baseline Metrics:
- Execution Time: {regression_detection['baseline_metrics']['execution_time']:.3f}s
- Throughput: {regression_detection['baseline_metrics']['throughput']:.0f} items/sec
- Memory: {regression_detection['baseline_metrics']['memory_usage_mb']:.1f} MB

Current Metrics:
- Execution Time: {regression_detection['current_metrics']['execution_time']:.3f}s
- Throughput: {regression_detection['current_metrics']['throughput']:.0f} items/sec
- Memory: {regression_detection['current_metrics']['memory_usage_mb']:.1f} MB

Regression ID: {regression_id}
Timestamp: {datetime.now().isoformat()}
        """.strip()

        logger.critical(f"REGRESSION ALERT: {alert_message}")

        # TODO: Implement actual alerting mechanisms
        # Examples:
        # - send_email_alert(alert_message)
        # - send_slack_webhook(alert_message)
        # - create_pagerduty_incident(regression_detection)

        # For now, write to alert file
        alert_file = self.results_dir / "regression_alerts.log"
        with open(alert_file, "a") as f:
            f.write(f"\n{datetime.now().isoformat()} - {alert_message}\n")
            f.write("=" * 80 + "\n")

    def run_regression_test(
        self,
        implementation_func: callable,
        dataset: Any,
        implementation: str,
        backend: str,
        descriptor: str,
        dataset_size: int,
        test_name: str,
        iterations: int = 5,
    ) -> Optional[RegressionResult]:
        """Run regression test against baseline.

        Args:
            implementation_func: Function to test
            dataset: Dataset to use
            implementation: Implementation name
            backend: Backend name
            descriptor: Descriptor name
            dataset_size: Dataset size
            test_name: Test name
            iterations: Number of test iterations

        Returns:
            Regression test result or None if no baseline
        """
        key = self._generate_baseline_key(
            implementation, backend, descriptor, dataset_size
        )

        if key not in self.baselines:
            logger.warning(
                f"No baseline found for {implementation} {backend} {descriptor} "
                f"size {dataset_size}"
            )
            return None

        baseline = self.baselines[key]

        logger.info(f"Running regression test: {test_name}")

        # Run performance test
        execution_times = []
        memory_usage_samples = []

        for i in range(iterations):
            # Measure memory before
            memory_before = self._get_memory_usage()

            # Time execution
            start_time = time.perf_counter()

            try:
                result = implementation_func(dataset)

                end_time = time.perf_counter()
                execution_time = end_time - start_time

                # Measure memory after
                memory_after = self._get_memory_usage()
                memory_used = memory_after - memory_before

                execution_times.append(execution_time)
                memory_usage_samples.append(memory_used)

            except Exception as e:
                logger.error(f"Regression test iteration {i} failed: {e}")
                return None

        # Calculate statistics
        current_time = statistics.mean(execution_times)
        current_memory = statistics.mean(memory_usage_samples)
        current_throughput = dataset_size / current_time if current_time > 0 else 0.0

        # Calculate changes
        time_change = (
            (current_time - baseline.baseline_time) / baseline.baseline_time * 100
        )
        throughput_change = (
            (current_throughput - baseline.baseline_throughput)
            / baseline.baseline_throughput
            * 100
        )
        memory_change = (
            (current_memory - baseline.baseline_memory_mb)
            / baseline.baseline_memory_mb
            * 100
        )

        # Determine regression severity
        is_regression = time_change > self.thresholds["minor_regression"] * 100

        if time_change > self.thresholds["critical_regression"] * 100:
            severity = "critical"
        elif time_change > self.thresholds["major_regression"] * 100:
            severity = "major"
        elif time_change > self.thresholds["minor_regression"] * 100:
            severity = "minor"
        else:
            severity = "none"

        # Calculate confidence level (simplified)
        std_time = (
            statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
        )
        confidence = (
            max(0.0, 1.0 - (std_time / current_time)) if current_time > 0 else 0.0
        )

        result = RegressionResult(
            test_name=test_name,
            implementation=implementation,
            backend=backend,
            descriptor=descriptor,
            dataset_size=dataset_size,
            baseline=baseline,
            current_time=current_time,
            current_throughput=current_throughput,
            current_memory_mb=current_memory,
            time_change_percent=time_change,
            throughput_change_percent=throughput_change,
            memory_change_percent=memory_change,
            is_regression=is_regression,
            regression_severity=severity,
            confidence_level=confidence,
            test_timestamp=datetime.now().isoformat(),
            metadata={
                "iterations": iterations,
                "execution_times": execution_times,
                "memory_samples": memory_usage_samples,
                "baseline_version": baseline.version,
            },
        )

        self.regression_results.append(result)

        if is_regression:
            logger.warning(
                f"Regression detected: {severity} - {time_change:.1f}% slower"
            )
        else:
            logger.info(f"No regression: {time_change:.1f}% change")

        return result

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
        except Exception:
            return 0.0

    def run_comprehensive_regression_suite(
        self,
        implementations: Dict[str, callable],
        datasets: Dict[int, Any],
        descriptors: List[str],
        test_name_prefix: str = "regression_test",
        use_automated_detection: bool = True,
    ) -> List[RegressionResult]:
        """Run comprehensive regression tests with optional automated detection.

        Args:
            implementations: Dictionary of implementation functions by backend
            datasets: Dictionary of datasets by size
            descriptors: List of descriptors to test
            test_name_prefix: Prefix for test names
            use_automated_detection: Whether to use automated regression detection

        Returns:
            List of regression test results
        """
        logger.info("Starting comprehensive regression test suite")

        all_results = []

        for backend, impl_func in implementations.items():
            for descriptor in descriptors:
                for dataset_size, dataset in datasets.items():
                    test_name = (
                        f"{test_name_prefix}_{backend}_{descriptor}_{dataset_size}"
                    )

                    try:
                        if use_automated_detection:
                            result = self.run_automated_regression_test(
                                impl_func,
                                dataset,
                                "chemi_accel",
                                backend,
                                descriptor,
                                dataset_size,
                                test_name,
                                version="current",
                            )
                        else:
                            result = self.run_regression_test(
                                impl_func,
                                dataset,
                                "chemi_accel",
                                backend,
                                descriptor,
                                dataset_size,
                                test_name,
                            )

                        if result:
                            all_results.append(result)

                    except Exception as e:
                        logger.error(f"Regression test failed for {test_name}: {e}")

        # Save results
        self.save_regression_results()

        logger.info(f"Completed regression suite: {len(all_results)} tests")

        return all_results

    def save_regression_results(self) -> None:
        """Save regression test results."""
        results_data = []

        for result in self.regression_results:
            results_data.append(
                {
                    "test_name": result.test_name,
                    "implementation": result.implementation,
                    "backend": result.backend,
                    "descriptor": result.descriptor,
                    "dataset_size": result.dataset_size,
                    "baseline": result.baseline.to_dict(),
                    "current_time": result.current_time,
                    "current_throughput": result.current_throughput,
                    "current_memory_mb": result.current_memory_mb,
                    "time_change_percent": result.time_change_percent,
                    "throughput_change_percent": result.throughput_change_percent,
                    "memory_change_percent": result.memory_change_percent,
                    "is_regression": result.is_regression,
                    "regression_severity": result.regression_severity,
                    "confidence_level": result.confidence_level,
                    "test_timestamp": result.test_timestamp,
                    "metadata": result.metadata,
                }
            )

        results_file = (
            self.results_dir
            / f"regression_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Saved regression results to {results_file}")

    def generate_regression_report(self) -> str:
        """Generate regression test report.

        Returns:
            Formatted regression report
        """
        if not self.regression_results:
            return "No regression test results available."

        report = ["Performance Regression Test Report", "=" * 40, ""]

        # Summary statistics
        total_tests = len(self.regression_results)
        regressions = [r for r in self.regression_results if r.is_regression]
        regression_count = len(regressions)

        severity_counts = {}
        for result in self.regression_results:
            severity = result.regression_severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        report.extend(
            [
                f"Total tests: {total_tests}",
                f"Regressions detected: {regression_count} ({regression_count/total_tests*100:.1f}%)",
                "Severity breakdown:",
                f"  - Critical: {severity_counts.get('critical', 0)}",
                f"  - Major: {severity_counts.get('major', 0)}",
                f"  - Minor: {severity_counts.get('minor', 0)}",
                f"  - None: {severity_counts.get('none', 0)}",
                "",
            ]
        )

        # Detailed regression results
        if regressions:
            report.append("Detected Regressions:")
            report.append("-" * 20)

            for result in sorted(
                regressions, key=lambda x: x.time_change_percent, reverse=True
            ):
                report.extend(
                    [
                        f"Test: {result.test_name}",
                        f"  Implementation: {result.implementation} ({result.backend})",
                        f"  Descriptor: {result.descriptor}",
                        f"  Dataset size: {result.dataset_size}",
                        f"  Severity: {result.regression_severity}",
                        f"  Time change: {result.time_change_percent:+.1f}%",
                        f"  Throughput change: {result.throughput_change_percent:+.1f}%",
                        f"  Memory change: {result.memory_change_percent:+.1f}%",
                        f"  Confidence: {result.confidence_level:.2f}",
                        f"  Baseline version: {result.baseline.version}",
                        "",
                    ]
                )

        # Performance improvements
        improvements = [
            r
            for r in self.regression_results
            if not r.is_regression and r.time_change_percent < -5.0
        ]

        if improvements:
            report.append("Performance Improvements:")
            report.append("-" * 25)

            for result in sorted(improvements, key=lambda x: x.time_change_percent):
                report.extend(
                    [
                        f"Test: {result.test_name}",
                        f"  Time improvement: {abs(result.time_change_percent):.1f}%",
                        f"  Throughput improvement: {result.throughput_change_percent:+.1f}%",
                        "",
                    ]
                )

        return "\n".join(report)

    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """Get comprehensive regression analysis including trends and predictions.

        Returns:
            Comprehensive analysis dictionary
        """
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "legacy_summary": self.get_regression_summary(),
            "active_regressions": self.database.get_active_regressions(),
            "trend_analysis": {},
            "recommendations": [],
            "health_score": 0.0,
        }

        # Perform trend analysis for each unique test configuration
        test_configs = set()
        for result in self.regression_results:
            config = (
                result.implementation,
                result.backend,
                result.descriptor,
                result.dataset_size,
            )
            test_configs.add(config)

        trend_results = []
        for implementation, backend, descriptor, dataset_size in test_configs:
            trend_analysis = self.detector.analyze_performance_trends(
                implementation, backend, descriptor, dataset_size
            )

            if trend_analysis["sufficient_data"]:
                trend_results.append(
                    {
                        "config": {
                            "implementation": implementation,
                            "backend": backend,
                            "descriptor": descriptor,
                            "dataset_size": dataset_size,
                        },
                        "trends": trend_analysis,
                    }
                )

        analysis["trend_analysis"] = trend_results

        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)

        # Calculate overall health score
        analysis["health_score"] = self._calculate_health_score(analysis)

        return analysis

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        # Check for critical regressions
        critical_regressions = [
            r for r in analysis["active_regressions"] if r["severity"] == "critical"
        ]

        if critical_regressions:
            recommendations.append(
                f"URGENT: {len(critical_regressions)} critical performance regressions detected. "
                "Immediate investigation and rollback may be required."
            )

        # Check for degrading trends
        degrading_trends = [
            t
            for t in analysis["trend_analysis"]
            if t["trends"]["overall_health"] == "degrading"
        ]

        if degrading_trends:
            recommendations.append(
                f"Performance degradation trend detected in {len(degrading_trends)} test configurations. "
                "Consider performance optimization or infrastructure scaling."
            )

        # Check for memory issues
        memory_concerns = [
            t
            for t in analysis["trend_analysis"]
            if t["trends"]["overall_health"] == "concerning"
        ]

        if memory_concerns:
            recommendations.append(
                "Memory usage trends indicate potential memory leaks or inefficient allocation. "
                "Review memory management in affected components."
            )

        # Check for improving performance
        improving_trends = [
            t
            for t in analysis["trend_analysis"]
            if t["trends"]["overall_health"] == "improving"
        ]

        if improving_trends:
            recommendations.append(
                f"Performance improvements detected in {len(improving_trends)} configurations. "
                "Consider updating baselines to reflect new performance expectations."
            )

        # General recommendations
        if len(analysis["active_regressions"]) == 0 and len(degrading_trends) == 0:
            recommendations.append(
                "System performance appears stable. Continue regular monitoring."
            )

        if not recommendations:
            recommendations.append("Insufficient data for recommendations.")

        return recommendations

    def _calculate_health_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall system health score (0.0 to 1.0)."""
        score = 1.0

        # Penalize active regressions
        active_regressions = analysis["active_regressions"]
        for regression in active_regressions:
            if regression["severity"] == "critical":
                score -= 0.3
            elif regression["severity"] == "major":
                score -= 0.15
            elif regression["severity"] == "moderate":
                score -= 0.08
            elif regression["severity"] == "minor":
                score -= 0.05

        # Penalize degrading trends
        degrading_count = sum(
            1
            for t in analysis["trend_analysis"]
            if t["trends"]["overall_health"] == "degrading"
        )
        score -= degrading_count * 0.1

        # Penalize concerning trends (memory issues)
        concerning_count = sum(
            1
            for t in analysis["trend_analysis"]
            if t["trends"]["overall_health"] == "concerning"
        )
        score -= concerning_count * 0.05

        # Bonus for improving trends
        improving_count = sum(
            1
            for t in analysis["trend_analysis"]
            if t["trends"]["overall_health"] == "improving"
        )
        score += min(0.1, improving_count * 0.02)

        return max(0.0, min(1.0, score))

    def generate_automated_report(self) -> str:
        """Generate comprehensive automated regression report.

        Returns:
            Formatted comprehensive report
        """
        analysis = self.get_comprehensive_analysis()

        report = ["Automated Performance Regression Analysis Report", "=" * 60, ""]

        # Executive Summary
        report.extend(
            [
                "EXECUTIVE SUMMARY",
                "-" * 20,
                f"Health Score: {analysis['health_score']:.1%} ({'EXCELLENT' if analysis['health_score'] > 0.9 else 'GOOD' if analysis['health_score'] > 0.7 else 'CONCERNING' if analysis['health_score'] > 0.5 else 'CRITICAL'})",
                f"Active Regressions: {len(analysis['active_regressions'])}",
                f"Trend Analysis Coverage: {len(analysis['trend_analysis'])} test configurations",
                f"Report Generated: {analysis['timestamp']}",
                "",
            ]
        )

        # Active Regressions
        if analysis["active_regressions"]:
            report.extend(["ACTIVE REGRESSIONS", "-" * 20])

            for regression in sorted(
                analysis["active_regressions"],
                key=lambda x: x["time_change_percent"],
                reverse=True,
            ):
                report.extend(
                    [
                        f"• {regression['test_name']} [{regression['severity'].upper()}]",
                        f"  Performance Change: {regression['time_change_percent']:+.1f}%",
                        f"  Confidence: {regression['confidence_level']:.1%}",
                        f"  Status: {regression['status']}",
                        f"  Detected: {regression['created_at']}",
                        "",
                    ]
                )

        # Trend Analysis
        if analysis["trend_analysis"]:
            report.extend(["PERFORMANCE TRENDS", "-" * 20])

            for trend_result in analysis["trend_analysis"]:
                config = trend_result["config"]
                trends = trend_result["trends"]

                report.extend(
                    [
                        f"Configuration: {config['implementation']} / {config['backend']} / {config['descriptor']}",
                        f"  Dataset Size: {config['dataset_size']:,} molecules",
                        f"  Overall Health: {trends['overall_health'].upper()}",
                        f"  Data Points: {trends['data_points']} over {trends['time_period_days']} days",
                        f"  Execution Time Trend: {trends['execution_time_trend']['direction']} (confidence: {trends['execution_time_trend']['confidence']:.1%})",
                        f"  Throughput Trend: {trends['throughput_trend']['direction']} (confidence: {trends['throughput_trend']['confidence']:.1%})",
                        f"  Memory Trend: {trends['memory_trend']['direction']} (confidence: {trends['memory_trend']['confidence']:.1%})",
                        "",
                    ]
                )

        # Recommendations
        report.extend(["RECOMMENDATIONS", "-" * 20])
        for i, rec in enumerate(analysis["recommendations"], 1):
            report.extend([f"{i}. {rec}", ""])

        # Legacy Summary (for compatibility)
        legacy = analysis["legacy_summary"]
        if legacy:
            report.extend(
                [
                    "LEGACY TEST RESULTS",
                    "-" * 20,
                    f"Total Tests: {legacy['total_tests']}",
                    f"Detected Regressions: {legacy['regression_count']}",
                    f"Regression Rate: {legacy['regression_rate']:.1%}",
                    f"Average Time Change: {legacy['average_time_change_percent']:+.1f}%",
                    f"Average Throughput Change: {legacy['average_throughput_change_percent']:+.1f}%",
                    "",
                ]
            )

        return "\n".join(report)

    def get_regression_summary(self) -> Dict[str, Any]:
        """Get legacy regression test summary statistics.

        Returns:
            Dictionary with summary statistics
        """
        if not self.regression_results:
            return {}

        total_tests = len(self.regression_results)
        regressions = [r for r in self.regression_results if r.is_regression]

        severity_counts = {}
        for result in self.regression_results:
            severity = result.regression_severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        avg_time_change = statistics.mean(
            [r.time_change_percent for r in self.regression_results]
        )
        avg_throughput_change = statistics.mean(
            [r.throughput_change_percent for r in self.regression_results]
        )

        return {
            "total_tests": total_tests,
            "regression_count": len(regressions),
            "regression_rate": (
                len(regressions) / total_tests if total_tests > 0 else 0.0
            ),
            "severity_counts": severity_counts,
            "average_time_change_percent": avg_time_change,
            "average_throughput_change_percent": avg_throughput_change,
            "worst_regression": (
                max(regressions, key=lambda x: x.time_change_percent)
                if regressions
                else None
            ),
            "best_improvement": (
                min(self.regression_results, key=lambda x: x.time_change_percent)
                if self.regression_results
                else None
            ),
        }


def run_automated_regression_tests(
    implementations: Dict[str, callable],
    datasets: Dict[int, Any],
    descriptors: List[str],
    baseline_directory: str = "performance_baselines",
    results_directory: str = "regression_results",
    database_path: str = "performance_metrics.db",
    generate_report: bool = True,
) -> Dict[str, Any]:
    """Run comprehensive automated regression tests with advanced detection.

    Args:
        implementations: Dictionary of implementation functions by backend
        datasets: Dictionary of datasets by size
        descriptors: List of descriptors to test
        baseline_directory: Directory containing baselines
        results_directory: Directory to save results
        database_path: Path to performance metrics database
        generate_report: Whether to generate automated report

    Returns:
        Comprehensive analysis results
    """
    suite = RegressionTestSuite(baseline_directory, results_directory, database_path)

    logger.info("Starting automated regression test suite")

    # Run all automated tests
    all_results = []
    for backend, impl_func in implementations.items():
        for descriptor in descriptors:
            for dataset_size, dataset in datasets.items():
                test_name = f"automated_{backend}_{descriptor}_{dataset_size}"

                try:
                    result = suite.run_automated_regression_test(
                        impl_func,
                        dataset,
                        "chemi_accel",
                        backend,
                        descriptor,
                        dataset_size,
                        test_name,
                        version="current",
                    )

                    if result:
                        all_results.append(result)

                except Exception as e:
                    logger.error(
                        f"Automated regression test failed for {test_name}: {e}"
                    )

    # Generate comprehensive analysis
    analysis = suite.get_comprehensive_analysis()

    # Generate and save automated report
    if generate_report:
        report = suite.generate_automated_report()
        report_file = (
            suite.results_dir
            / f"automated_regression_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

        with open(report_file, "w") as f:
            f.write(report)

        logger.info(f"Automated regression report saved to {report_file}")
        print(f"\nAutomated Regression Report saved to: {report_file}")

        # Print executive summary
        print("\n" + "=" * 60)
        print("AUTOMATED REGRESSION ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Health Score: {analysis['health_score']:.1%}")
        print(f"Active Regressions: {len(analysis['active_regressions'])}")
        print(f"Tests Analyzed: {len(analysis['trend_analysis'])} configurations")

        if analysis["recommendations"]:
            print("\nTop Recommendations:")
            for i, rec in enumerate(analysis["recommendations"][:3], 1):
                print(f"{i}. {rec[:100]}{'...' if len(rec) > 100 else ''}")

    # Save legacy results for compatibility
    suite.save_regression_results()

    logger.info(f"Completed automated regression suite: {len(all_results)} tests")

    return analysis


def run_regression_tests(
    implementations: Dict[str, callable],
    datasets: Dict[int, Any],
    descriptors: List[str],
    baseline_directory: str = "performance_baselines",
    results_directory: str = "regression_results",
) -> List[RegressionResult]:
    """Run legacy regression tests (for backward compatibility).

    Args:
        implementations: Dictionary of implementation functions by backend
        datasets: Dictionary of datasets by size
        descriptors: List of descriptors to test
        baseline_directory: Directory containing baselines
        results_directory: Directory to save results

    Returns:
        List of regression test results
    """
    suite = RegressionTestSuite(baseline_directory, results_directory)
    return suite.run_comprehensive_regression_suite(
        implementations, datasets, descriptors
    )


if __name__ == "__main__":
    # Demo automated regression detection system
    print("ChemiAccel Automated Performance Regression Detection System")
    print("=" * 65)

    # Initialize system
    suite = RegressionTestSuite()

    # Example: Simulate performance data and regression detection
    import random

    def mock_calculation_function(dataset):
        """Mock calculation function for demo."""
        # Simulate variable performance
        base_time = 0.1 + random.uniform(-0.02, 0.02)
        time.sleep(base_time)
        return {"descriptors": len(dataset) * [1.0]}

    # Generate mock dataset
    mock_dataset = [f"CCO{i}" for i in range(100)]

    print("\nRunning demonstration regression tests...")

    # Simulate several test runs to build history
    for i in range(5):
        version = f"v1.{i}"
        print(f"Running test iteration {i+1} (version {version})...")

        result = suite.run_automated_regression_test(
            mock_calculation_function,
            mock_dataset,
            "chemi_accel",
            "mock_backend",
            "mock_descriptor",
            len(mock_dataset),
            f"demo_test_{i}",
            version=version,
        )

        if result:
            print(f"  - Execution time: {result.current_time:.3f}s")
            print(f"  - Throughput: {result.current_throughput:.0f} items/sec")
            if result.is_regression:
                print(f"  - Regression detected: {result.regression_severity}")

    # Generate comprehensive analysis
    print("\nGenerating comprehensive analysis...")
    analysis = suite.get_comprehensive_analysis()

    print(f"\nSystem Health Score: {analysis['health_score']:.1%}")
    print(f"Active Regressions: {len(analysis['active_regressions'])}")

    if analysis["recommendations"]:
        print("\nRecommendations:")
        for i, rec in enumerate(analysis["recommendations"], 1):
            print(f"{i}. {rec}")

    # Generate and display report
    report = suite.generate_automated_report()
    print("\n" + "=" * 60)
    print("SAMPLE AUTOMATED REPORT")
    print("=" * 60)
    print(report[:1000] + "..." if len(report) > 1000 else report)
