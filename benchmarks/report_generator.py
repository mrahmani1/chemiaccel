"""
Benchmark Report Generator for ChemiAccel.

This module generates comprehensive benchmark reports with statistical analysis,
visualizations, and performance comparisons.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import statistics
from datetime import datetime

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    pd = None
    np = None
    plt = None
    sns = None

from .benchmark_framework import BenchmarkResult, PerformanceComparison, BenchmarkSuite
from .scaling_tests import ScalingResult
from .regression_testing import RegressionResult
try:
    from ..src.chemi_accel.utils.logging import get_logger
except ImportError:
    # Fallback for standalone execution
    import logging
    def get_logger():
        return logging.getLogger(__name__)

logger = get_logger()


@dataclass
class ReportConfig:
    """Configuration for benchmark report generation."""
    title: str = "ChemiAccel Performance Benchmark Report"
    include_plots: bool = True
    include_statistical_analysis: bool = True
    include_scaling_analysis: bool = True
    include_regression_analysis: bool = True
    output_format: str = "html"  # "html", "markdown", "pdf"
    plot_style: str = "seaborn"
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300


class BenchmarkReportGenerator:
    """Generates comprehensive benchmark reports."""
    
    def __init__(self, config: ReportConfig, output_directory: str = "benchmark_reports"):
        """Initialize report generator.
        
        Args:
            config: Report configuration
            output_directory: Directory to save reports
        """
        self.config = config
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        if PLOTTING_AVAILABLE and self.config.plot_style:
            try:
                plt.style.use(self.config.plot_style)
            except Exception:
                logger.warning(f"Failed to set plot style: {self.config.plot_style}")
        
        logger.info(f"Initialized benchmark report generator")
    
    def generate_comprehensive_report(self,
                                    benchmark_results: List[BenchmarkResult],
                                    comparisons: List[PerformanceComparison],
                                    scaling_results: Optional[List[ScalingResult]] = None,
                                    regression_results: Optional[List[RegressionResult]] = None) -> str:
        """Generate comprehensive benchmark report.
        
        Args:
            benchmark_results: List of benchmark results
            comparisons: List of performance comparisons
            scaling_results: List of scaling test results (optional)
            regression_results: List of regression test results (optional)
            
        Returns:
            Path to generated report file
        """
        logger.info("Generating comprehensive benchmark report")
        
        # Generate report sections
        sections = []
        
        # Title and summary
        sections.append(self._generate_title_section())
        sections.append(self._generate_executive_summary(benchmark_results, comparisons))
        
        # Performance comparison analysis
        sections.append(self._generate_performance_analysis(benchmark_results, comparisons))
        
        # Statistical analysis
        if self.config.include_statistical_analysis:
            sections.append(self._generate_statistical_analysis(benchmark_results, comparisons))
        
        # Scaling analysis
        if self.config.include_scaling_analysis and scaling_results:
            sections.append(self._generate_scaling_analysis(scaling_results))
        
        # Regression analysis
        if self.config.include_regression_analysis and regression_results:
            sections.append(self._generate_regression_analysis(regression_results))
        
        # Recommendations
        sections.append(self._generate_recommendations(benchmark_results, comparisons))
        
        # Combine sections
        report_content = "\n\n".join(sections)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"benchmark_report_{timestamp}.{self.config.output_format}"
        report_path = self.output_dir / report_filename
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Generated benchmark report: {report_path}")
        
        return str(report_path)
    
    def _generate_title_section(self) -> str:
        """Generate title section."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if self.config.output_format == "html":
            return f"""
<h1>{self.config.title}</h1>
<p><strong>Generated:</strong> {timestamp}</p>
<hr>
"""
        else:
            return f"""
# {self.config.title}

**Generated:** {timestamp}

---
"""
    
    def _generate_executive_summary(self,
                                  benchmark_results: List[BenchmarkResult],
                                  comparisons: List[PerformanceComparison]) -> str:
        """Generate executive summary."""
        # Calculate summary statistics
        total_benchmarks = len(benchmark_results)
        successful_benchmarks = len([r for r in benchmark_results if r.status.value == "completed"])
        total_comparisons = len(comparisons)
        significant_improvements = len([c for c in comparisons if c.is_significant_improvement])
        
        # Calculate average speedup
        speedups = [c.speedup_factor for c in comparisons if c.speedup_factor > 0]
        avg_speedup = statistics.mean(speedups) if speedups else 0.0
        max_speedup = max(speedups) if speedups else 0.0
        
        # Find best performing configurations
        best_comparison = max(comparisons, key=lambda x: x.speedup_factor) if comparisons else None
        
        summary = f"""
## Executive Summary

### Key Results
- **Total Benchmarks:** {total_benchmarks} ({successful_benchmarks} successful)
- **Performance Comparisons:** {total_comparisons}
- **Significant Improvements:** {significant_improvements} ({significant_improvements/total_comparisons*100:.1f}%)
- **Average Speedup:** {avg_speedup:.2f}x
- **Maximum Speedup:** {max_speedup:.2f}x

### Best Performance
"""
        
        if best_comparison:
            summary += f"""
- **Configuration:** {best_comparison.chemi_accel_result.backend} backend
- **Descriptor:** {best_comparison.chemi_accel_result.descriptor}
- **Dataset Size:** {best_comparison.chemi_accel_result.dataset_size:,} molecules
- **Speedup:** {best_comparison.speedup_factor:.2f}x vs {best_comparison.reference_result.implementation}
- **Throughput:** {best_comparison.chemi_accel_result.throughput:.0f} molecules/second
"""
        
        return summary
    
    def _generate_performance_analysis(self,
                                     benchmark_results: List[BenchmarkResult],
                                     comparisons: List[PerformanceComparison]) -> str:
        """Generate performance analysis section."""
        analysis = "## Performance Analysis\n\n"
        
        # Group results by backend
        by_backend = {}
        for result in benchmark_results:
            if result.backend not in by_backend:
                by_backend[result.backend] = []
            by_backend[result.backend].append(result)
        
        # Backend performance comparison
        analysis += "### Performance by Backend\n\n"
        
        for backend, results in by_backend.items():
            successful_results = [r for r in results if r.status.value == "completed"]
            if not successful_results:
                continue
            
            avg_throughput = statistics.mean([r.throughput for r in successful_results])
            avg_memory = statistics.mean([r.memory_usage_mb for r in successful_results])
            
            analysis += f"""
**{backend.upper()} Backend:**
- Tests: {len(successful_results)}
- Average Throughput: {avg_throughput:.0f} molecules/second
- Average Memory Usage: {avg_memory:.1f} MB
"""
        
        # Descriptor performance comparison
        analysis += "\n### Performance by Descriptor\n\n"
        
        by_descriptor = {}
        for comparison in comparisons:
            descriptor = comparison.chemi_accel_result.descriptor
            if descriptor not in by_descriptor:
                by_descriptor[descriptor] = []
            by_descriptor[descriptor].append(comparison)
        
        for descriptor, desc_comparisons in by_descriptor.items():
            speedups = [c.speedup_factor for c in desc_comparisons]
            avg_speedup = statistics.mean(speedups)
            max_speedup = max(speedups)
            
            analysis += f"""
**{descriptor}:**
- Comparisons: {len(desc_comparisons)}
- Average Speedup: {avg_speedup:.2f}x
- Maximum Speedup: {max_speedup:.2f}x
"""
        
        # Generate performance plots
        if self.config.include_plots and PLOTTING_AVAILABLE:
            self._generate_performance_plots(benchmark_results, comparisons)
            
            if self.config.output_format == "html":
                analysis += '\n<img src="performance_comparison.png" alt="Performance Comparison">\n'
            else:
                analysis += '\n![Performance Comparison](performance_comparison.png)\n'
        
        return analysis
    
    def _generate_statistical_analysis(self,
                                     benchmark_results: List[BenchmarkResult],
                                     comparisons: List[PerformanceComparison]) -> str:
        """Generate statistical analysis section."""
        analysis = "## Statistical Analysis\n\n"
        
        # Speedup distribution analysis
        speedups = [c.speedup_factor for c in comparisons if c.speedup_factor > 0]
        
        if speedups:
            analysis += "### Speedup Distribution\n\n"
            analysis += f"""
- **Mean:** {statistics.mean(speedups):.2f}x
- **Median:** {statistics.median(speedups):.2f}x
- **Standard Deviation:** {statistics.stdev(speedups):.2f}x
- **Min:** {min(speedups):.2f}x
- **Max:** {max(speedups):.2f}x
"""
            
            # Percentiles
            if len(speedups) >= 4:
                sorted_speedups = sorted(speedups)
                n = len(sorted_speedups)
                
                p25 = sorted_speedups[n//4]
                p75 = sorted_speedups[3*n//4]
                p90 = sorted_speedups[9*n//10]
                p95 = sorted_speedups[19*n//20]
                
                analysis += f"""
- **25th Percentile:** {p25:.2f}x
- **75th Percentile:** {p75:.2f}x
- **90th Percentile:** {p90:.2f}x
- **95th Percentile:** {p95:.2f}x
"""
        
        # Statistical significance analysis
        significant_comparisons = [c for c in comparisons if c.is_significant_improvement]
        
        analysis += f"\n### Statistical Significance\n\n"
        analysis += f"""
- **Total Comparisons:** {len(comparisons)}
- **Statistically Significant:** {len(significant_comparisons)} ({len(significant_comparisons)/len(comparisons)*100:.1f}%)
- **Average p-value:** {statistics.mean([c.statistical_significance for c in comparisons]):.4f}
"""
        
        return analysis
    
    def _generate_scaling_analysis(self, scaling_results: List[ScalingResult]) -> str:
        """Generate scaling analysis section."""
        analysis = "## Scaling Analysis\n\n"
        
        # Group by scaling type
        by_scaling_type = {}
        for result in scaling_results:
            scaling_type = result.scaling_type.value
            if scaling_type not in by_scaling_type:
                by_scaling_type[scaling_type] = []
            by_scaling_type[scaling_type].append(result)
        
        for scaling_type, results in by_scaling_type.items():
            analysis += f"### {scaling_type.replace('_', ' ').title()} Scaling\n\n"
            
            # Calculate scaling statistics
            efficiencies = [r.scaling_efficiency for r in results]
            speedups = [r.actual_speedup for r in results]
            
            if efficiencies:
                analysis += f"""
- **Tests:** {len(results)}
- **Average Scaling Efficiency:** {statistics.mean(efficiencies):.3f}
- **Best Scaling Efficiency:** {max(efficiencies):.3f}
- **Average Speedup:** {statistics.mean(speedups):.2f}x
- **Best Speedup:** {max(speedups):.2f}x
"""
            
            # Best performing configuration
            best_result = max(results, key=lambda x: x.scaling_efficiency)
            analysis += f"""
**Best Configuration:**
- Implementation: {best_result.implementation}
- Descriptor: {best_result.descriptor}
- Hardware: {best_result.hardware_config.name}
- Scaling Efficiency: {best_result.scaling_efficiency:.3f}
"""
        
        return analysis
    
    def _generate_regression_analysis(self, regression_results: List[RegressionResult]) -> str:
        """Generate regression analysis section."""
        analysis = "## Regression Analysis\n\n"
        
        # Summary statistics
        total_tests = len(regression_results)
        regressions = [r for r in regression_results if r.is_regression]
        regression_count = len(regressions)
        
        analysis += f"### Regression Summary\n\n"
        analysis += f"""
- **Total Tests:** {total_tests}
- **Regressions Detected:** {regression_count} ({regression_count/total_tests*100:.1f}%)
"""
        
        # Severity breakdown
        severity_counts = {}
        for result in regression_results:
            severity = result.regression_severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        analysis += "\n**Severity Breakdown:**\n"
        for severity in ['critical', 'major', 'minor', 'none']:
            count = severity_counts.get(severity, 0)
            analysis += f"- {severity.title()}: {count}\n"
        
        # Worst regressions
        if regressions:
            analysis += "\n### Significant Regressions\n\n"
            
            worst_regressions = sorted(regressions, key=lambda x: x.time_change_percent, reverse=True)[:5]
            
            for i, result in enumerate(worst_regressions, 1):
                analysis += f"""
**{i}. {result.test_name}**
- Severity: {result.regression_severity}
- Performance Change: {result.time_change_percent:+.1f}%
- Configuration: {result.backend} backend, {result.descriptor}
- Dataset Size: {result.dataset_size:,} molecules
"""
        
        return analysis
    
    def _generate_recommendations(self,
                                benchmark_results: List[BenchmarkResult],
                                comparisons: List[PerformanceComparison]) -> str:
        """Generate recommendations section."""
        recommendations = "## Recommendations\n\n"
        
        # Find best configurations
        best_backend = self._find_best_backend(benchmark_results)
        best_descriptors = self._find_best_descriptors(comparisons)
        
        recommendations += "### Optimal Configurations\n\n"
        
        if best_backend:
            recommendations += f"""
**Recommended Backend:** {best_backend['name']}
- Average Throughput: {best_backend['avg_throughput']:.0f} molecules/second
- Average Memory Usage: {best_backend['avg_memory']:.1f} MB
- Success Rate: {best_backend['success_rate']:.1f}%
"""
        
        if best_descriptors:
            recommendations += "\n**Best Performing Descriptors:**\n"
            for i, desc in enumerate(best_descriptors[:3], 1):
                recommendations += f"{i}. {desc['name']} (avg speedup: {desc['avg_speedup']:.2f}x)\n"
        
        # Performance optimization recommendations
        recommendations += "\n### Performance Optimization\n\n"
        
        # Analyze memory usage patterns
        high_memory_results = [r for r in benchmark_results if r.memory_usage_mb > 1000]
        if high_memory_results:
            recommendations += "- Consider memory optimization for large datasets (>1GB usage detected)\n"
        
        # Analyze throughput patterns
        low_throughput_results = [r for r in benchmark_results if r.throughput < 100]
        if low_throughput_results:
            recommendations += "- Investigate performance bottlenecks for low-throughput configurations\n"
        
        # Scaling recommendations
        recommendations += "- Use batch processing for large datasets to optimize memory usage\n"
        recommendations += "- Consider GPU acceleration for embarrassingly parallel descriptors\n"
        recommendations += "- Implement streaming for datasets larger than available memory\n"
        
        return recommendations
    
    def _find_best_backend(self, benchmark_results: List[BenchmarkResult]) -> Optional[Dict[str, Any]]:
        """Find best performing backend."""
        by_backend = {}
        
        for result in benchmark_results:
            if result.status.value != "completed":
                continue
            
            if result.backend not in by_backend:
                by_backend[result.backend] = []
            by_backend[result.backend].append(result)
        
        if not by_backend:
            return None
        
        best_backend = None
        best_score = 0
        
        for backend, results in by_backend.items():
            avg_throughput = statistics.mean([r.throughput for r in results])
            avg_memory = statistics.mean([r.memory_usage_mb for r in results])
            success_rate = len(results) / len([r for r in benchmark_results if r.backend == backend]) * 100
            
            # Simple scoring: prioritize throughput and success rate
            score = avg_throughput * (success_rate / 100)
            
            if score > best_score:
                best_score = score
                best_backend = {
                    'name': backend,
                    'avg_throughput': avg_throughput,
                    'avg_memory': avg_memory,
                    'success_rate': success_rate
                }
        
        return best_backend
    
    def _find_best_descriptors(self, comparisons: List[PerformanceComparison]) -> List[Dict[str, Any]]:
        """Find best performing descriptors."""
        by_descriptor = {}
        
        for comparison in comparisons:
            descriptor = comparison.chemi_accel_result.descriptor
            if descriptor not in by_descriptor:
                by_descriptor[descriptor] = []
            by_descriptor[descriptor].append(comparison)
        
        descriptor_scores = []
        
        for descriptor, desc_comparisons in by_descriptor.items():
            speedups = [c.speedup_factor for c in desc_comparisons]
            avg_speedup = statistics.mean(speedups)
            
            descriptor_scores.append({
                'name': descriptor,
                'avg_speedup': avg_speedup,
                'comparisons': len(desc_comparisons)
            })
        
        return sorted(descriptor_scores, key=lambda x: x['avg_speedup'], reverse=True)
    
    def _generate_performance_plots(self,
                                  benchmark_results: List[BenchmarkResult],
                                  comparisons: List[PerformanceComparison]) -> None:
        """Generate performance visualization plots."""
        if not PLOTTING_AVAILABLE:
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.config.figure_size)
        fig.suptitle('ChemiAccel Performance Analysis')
        
        # Plot 1: Speedup comparison
        speedups = [c.speedup_factor for c in comparisons]
        implementations = [f"{c.chemi_accel_result.backend}\n{c.chemi_accel_result.descriptor}" 
                          for c in comparisons]
        
        ax1.bar(range(len(speedups)), speedups)
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Speedup Factor')
        ax1.set_title('Performance Speedup by Configuration')
        ax1.set_xticks(range(len(implementations)))
        ax1.set_xticklabels(implementations, rotation=45, ha='right')
        
        # Plot 2: Throughput comparison
        successful_results = [r for r in benchmark_results if r.status.value == "completed"]
        backends = list(set(r.backend for r in successful_results))
        
        throughput_data = []
        for backend in backends:
            backend_results = [r for r in successful_results if r.backend == backend]
            throughputs = [r.throughput for r in backend_results]
            throughput_data.append(throughputs)
        
        ax2.boxplot(throughput_data, labels=backends)
        ax2.set_ylabel('Throughput (molecules/s)')
        ax2.set_title('Throughput Distribution by Backend')
        
        # Plot 3: Memory usage vs dataset size
        dataset_sizes = [r.dataset_size for r in successful_results]
        memory_usage = [r.memory_usage_mb for r in successful_results]
        
        ax3.scatter(dataset_sizes, memory_usage, alpha=0.6)
        ax3.set_xlabel('Dataset Size')
        ax3.set_ylabel('Memory Usage (MB)')
        ax3.set_title('Memory Usage vs Dataset Size')
        ax3.set_xscale('log')
        
        # Plot 4: Performance improvement distribution
        improvements = [c.performance_improvement_percent for c in comparisons]
        
        ax4.hist(improvements, bins=20, alpha=0.7)
        ax4.set_xlabel('Performance Improvement (%)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Performance Improvement Distribution')
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "performance_comparison.png"
        plt.savefig(plot_file, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated performance plots: {plot_file}")


def generate_benchmark_report(benchmark_results: List[BenchmarkResult],
                            comparisons: List[PerformanceComparison],
                            scaling_results: Optional[List[ScalingResult]] = None,
                            regression_results: Optional[List[RegressionResult]] = None,
                            config: Optional[ReportConfig] = None,
                            output_directory: str = "benchmark_reports") -> str:
    """Generate comprehensive benchmark report.
    
    Args:
        benchmark_results: List of benchmark results
        comparisons: List of performance comparisons
        scaling_results: List of scaling test results (optional)
        regression_results: List of regression test results (optional)
        config: Report configuration (optional)
        output_directory: Directory to save report
        
    Returns:
        Path to generated report file
    """
    if config is None:
        config = ReportConfig()
    
    generator = BenchmarkReportGenerator(config, output_directory)
    return generator.generate_comprehensive_report(
        benchmark_results, comparisons, scaling_results, regression_results
    )