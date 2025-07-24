"""
Performance Regression Testing for ChemiAccel.

This module provides automated performance regression testing to detect
performance degradations across different versions and configurations.
"""

import logging
import time
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib
from datetime import datetime

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
    np = None

try:
    from ..src.chemi_accel.utils.logging import get_logger
except ImportError:
    # Fallback for standalone execution
    import logging
    def get_logger():
        return logging.getLogger(__name__)

logger = get_logger()


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
            'implementation': self.implementation,
            'backend': self.backend,
            'descriptor': self.descriptor,
            'dataset_size': self.dataset_size,
            'baseline_time': self.baseline_time,
            'baseline_throughput': self.baseline_throughput,
            'baseline_memory_mb': self.baseline_memory_mb,
            'version': self.version,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceBaseline':
        """Create from dictionary."""
        return cls(
            implementation=data['implementation'],
            backend=data['backend'],
            descriptor=data['descriptor'],
            dataset_size=data['dataset_size'],
            baseline_time=data['baseline_time'],
            baseline_throughput=data['baseline_throughput'],
            baseline_memory_mb=data['baseline_memory_mb'],
            version=data['version'],
            timestamp=data['timestamp'],
            metadata=data.get('metadata', {})
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


class RegressionTestSuite:
    """Comprehensive regression testing suite."""
    
    def __init__(self, 
                 baseline_directory: str = "performance_baselines",
                 results_directory: str = "regression_results"):
        """Initialize regression test suite.
        
        Args:
            baseline_directory: Directory to store baseline data
            results_directory: Directory to store regression test results
        """
        self.baseline_dir = Path(baseline_directory)
        self.results_dir = Path(results_directory)
        
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.regression_results: List[RegressionResult] = []
        
        # Regression thresholds
        self.thresholds = {
            'minor_regression': 0.10,    # 10% performance degradation
            'major_regression': 0.25,    # 25% performance degradation
            'critical_regression': 0.50  # 50% performance degradation
        }
        
        # Load existing baselines
        self._load_baselines()
        
        logger.info(f"Initialized regression test suite with {len(self.baselines)} baselines")
    
    def _load_baselines(self) -> None:
        """Load existing performance baselines."""
        baseline_file = self.baseline_dir / "baselines.json"
        
        if baseline_file.exists():
            try:
                with open(baseline_file, 'r') as f:
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
            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f, indent=2)
            
            logger.info(f"Saved {len(self.baselines)} performance baselines")
            
        except Exception as e:
            logger.error(f"Failed to save baselines: {e}")
    
    def _generate_baseline_key(self, 
                              implementation: str,
                              backend: str,
                              descriptor: str,
                              dataset_size: int) -> str:
        """Generate unique key for baseline."""
        key_string = f"{implementation}_{backend}_{descriptor}_{dataset_size}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def set_baseline(self,
                    implementation: str,
                    backend: str,
                    descriptor: str,
                    dataset_size: int,
                    execution_time: float,
                    throughput: float,
                    memory_usage_mb: float,
                    version: str,
                    metadata: Optional[Dict[str, Any]] = None) -> None:
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
        key = self._generate_baseline_key(implementation, backend, descriptor, dataset_size)
        
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
            metadata=metadata or {}
        )
        
        self.baselines[key] = baseline
        self._save_baselines()
        
        logger.info(f"Set baseline for {implementation} {backend} {descriptor} "
                   f"size {dataset_size}: {execution_time:.3f}s")
    
    def run_regression_test(self,
                           implementation_func: callable,
                           dataset: Any,
                           implementation: str,
                           backend: str,
                           descriptor: str,
                           dataset_size: int,
                           test_name: str,
                           iterations: int = 5) -> Optional[RegressionResult]:
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
        key = self._generate_baseline_key(implementation, backend, descriptor, dataset_size)
        
        if key not in self.baselines:
            logger.warning(f"No baseline found for {implementation} {backend} {descriptor} "
                          f"size {dataset_size}")
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
        time_change = (current_time - baseline.baseline_time) / baseline.baseline_time * 100
        throughput_change = (current_throughput - baseline.baseline_throughput) / baseline.baseline_throughput * 100
        memory_change = (current_memory - baseline.baseline_memory_mb) / baseline.baseline_memory_mb * 100
        
        # Determine regression severity
        is_regression = time_change > self.thresholds['minor_regression'] * 100
        
        if time_change > self.thresholds['critical_regression'] * 100:
            severity = "critical"
        elif time_change > self.thresholds['major_regression'] * 100:
            severity = "major"
        elif time_change > self.thresholds['minor_regression'] * 100:
            severity = "minor"
        else:
            severity = "none"
        
        # Calculate confidence level (simplified)
        std_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
        confidence = max(0.0, 1.0 - (std_time / current_time)) if current_time > 0 else 0.0
        
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
                'iterations': iterations,
                'execution_times': execution_times,
                'memory_samples': memory_usage_samples,
                'baseline_version': baseline.version
            }
        )
        
        self.regression_results.append(result)
        
        if is_regression:
            logger.warning(f"Regression detected: {severity} - {time_change:.1f}% slower")
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
    
    def run_comprehensive_regression_suite(self,
                                         implementations: Dict[str, callable],
                                         datasets: Dict[int, Any],
                                         descriptors: List[str],
                                         test_name_prefix: str = "regression_test") -> List[RegressionResult]:
        """Run comprehensive regression tests.
        
        Args:
            implementations: Dictionary of implementation functions by backend
            datasets: Dictionary of datasets by size
            descriptors: List of descriptors to test
            test_name_prefix: Prefix for test names
            
        Returns:
            List of regression test results
        """
        logger.info("Starting comprehensive regression test suite")
        
        all_results = []
        
        for backend, impl_func in implementations.items():
            for descriptor in descriptors:
                for dataset_size, dataset in datasets.items():
                    test_name = f"{test_name_prefix}_{backend}_{descriptor}_{dataset_size}"
                    
                    try:
                        result = self.run_regression_test(
                            impl_func,
                            dataset,
                            "chemi_accel",  # Assuming ChemiAccel implementation
                            backend,
                            descriptor,
                            dataset_size,
                            test_name
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
            results_data.append({
                'test_name': result.test_name,
                'implementation': result.implementation,
                'backend': result.backend,
                'descriptor': result.descriptor,
                'dataset_size': result.dataset_size,
                'baseline': result.baseline.to_dict(),
                'current_time': result.current_time,
                'current_throughput': result.current_throughput,
                'current_memory_mb': result.current_memory_mb,
                'time_change_percent': result.time_change_percent,
                'throughput_change_percent': result.throughput_change_percent,
                'memory_change_percent': result.memory_change_percent,
                'is_regression': result.is_regression,
                'regression_severity': result.regression_severity,
                'confidence_level': result.confidence_level,
                'test_timestamp': result.test_timestamp,
                'metadata': result.metadata
            })
        
        results_file = self.results_dir / f"regression_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w') as f:
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
        
        report.extend([
            f"Total tests: {total_tests}",
            f"Regressions detected: {regression_count} ({regression_count/total_tests*100:.1f}%)",
            f"Severity breakdown:",
            f"  - Critical: {severity_counts.get('critical', 0)}",
            f"  - Major: {severity_counts.get('major', 0)}",
            f"  - Minor: {severity_counts.get('minor', 0)}",
            f"  - None: {severity_counts.get('none', 0)}",
            ""
        ])
        
        # Detailed regression results
        if regressions:
            report.append("Detected Regressions:")
            report.append("-" * 20)
            
            for result in sorted(regressions, key=lambda x: x.time_change_percent, reverse=True):
                report.extend([
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
                    ""
                ])
        
        # Performance improvements
        improvements = [r for r in self.regression_results 
                       if not r.is_regression and r.time_change_percent < -5.0]
        
        if improvements:
            report.append("Performance Improvements:")
            report.append("-" * 25)
            
            for result in sorted(improvements, key=lambda x: x.time_change_percent):
                report.extend([
                    f"Test: {result.test_name}",
                    f"  Time improvement: {abs(result.time_change_percent):.1f}%",
                    f"  Throughput improvement: {result.throughput_change_percent:+.1f}%",
                    ""
                ])
        
        return "\n".join(report)
    
    def get_regression_summary(self) -> Dict[str, Any]:
        """Get regression test summary statistics.
        
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
        
        avg_time_change = statistics.mean([r.time_change_percent for r in self.regression_results])
        avg_throughput_change = statistics.mean([r.throughput_change_percent for r in self.regression_results])
        
        return {
            'total_tests': total_tests,
            'regression_count': len(regressions),
            'regression_rate': len(regressions) / total_tests if total_tests > 0 else 0.0,
            'severity_counts': severity_counts,
            'average_time_change_percent': avg_time_change,
            'average_throughput_change_percent': avg_throughput_change,
            'worst_regression': max(regressions, key=lambda x: x.time_change_percent) if regressions else None,
            'best_improvement': min(self.regression_results, key=lambda x: x.time_change_percent) if self.regression_results else None
        }


def run_regression_tests(implementations: Dict[str, callable],
                        datasets: Dict[int, Any],
                        descriptors: List[str],
                        baseline_directory: str = "performance_baselines",
                        results_directory: str = "regression_results") -> List[RegressionResult]:
    """Run comprehensive regression tests.
    
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
    return suite.run_comprehensive_regression_suite(implementations, datasets, descriptors)