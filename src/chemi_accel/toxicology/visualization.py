"""
Toxicology visualization and reporting tools.

Provides comprehensive visualization capabilities for toxicology predictions,
risk assessments, and safer design optimization results.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .core import ToxicityData, ToxicityPrediction
from .risk_assessment import RiskScore
from .safer_design import SafeDesignCandidate


@dataclass
class RiskLandscape:
    """Risk landscape data structure for visualization."""

    compounds: List[str]
    risk_scores: List[float]
    confidence_levels: List[str]
    endpoint_contributions: List[Dict[str, float]]
    coordinates: Optional[np.ndarray] = None  # 2D coordinates for plotting

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for plotting."""
        data = {
            "compound": self.compounds,
            "risk_score": self.risk_scores,
            "confidence": self.confidence_levels,
        }

        if self.coordinates is not None:
            data["x"] = self.coordinates[:, 0]
            data["y"] = self.coordinates[:, 1]

        return pd.DataFrame(data)


class ToxicologyVisualizer:
    """
    Comprehensive visualization system for toxicology results.

    Provides methods for creating various plots and reports for toxicology
    predictions, risk assessments, and optimization results.
    """

    def __init__(self):
        self.color_schemes = {
            "risk": {
                "negligible": "#2ECC71",  # Green
                "low": "#F39C12",  # Orange
                "moderate": "#E67E22",  # Dark orange
                "high": "#E74C3C",  # Red
                "very_high": "#8E44AD",  # Purple
                "unacceptable": "#2C3E50",  # Dark gray
            },
            "confidence": {
                "very_low": "#BDC3C7",  # Light gray
                "low": "#95A5A6",  # Gray
                "moderate": "#3498DB",  # Blue
                "high": "#27AE60",  # Green
                "very_high": "#16A085",  # Teal
            },
        }

    def plot_toxicity_profile(
        self,
        toxicity_data: ToxicityData,
        include_uncertainty: bool = True,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Create comprehensive toxicity profile plot.

        Args:
            toxicity_data: Toxicity data to visualize
            include_uncertainty: Include uncertainty bars
            save_path: Path to save plot

        Returns:
            Plot description or path
        """

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Prepare data
            endpoints = []
            values = []
            uncertainties = []
            confidences = []

            for pred in toxicity_data.predicted_endpoints:
                endpoints.append(
                    pred.endpoint.endpoint_type.value.replace("_", " ").title()
                )
                values.append(pred.predicted_value)
                uncertainties.append(pred.uncertainty if include_uncertainty else 0)
                confidences.append(pred.confidence.value)

            if not endpoints:
                return "No toxicity predictions to visualize"

            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Toxicity values plot
            bars = ax1.barh(
                endpoints, values, color=[self._get_toxicity_color(v) for v in values]
            )

            if include_uncertainty:
                ax1.errorbar(
                    values,
                    range(len(values)),
                    xerr=uncertainties,
                    fmt="none",
                    color="black",
                    capsize=3,
                )

            ax1.set_xlabel("Predicted Value")
            ax1.set_title(f"Toxicity Profile - {toxicity_data.smiles}")
            ax1.grid(axis="x", alpha=0.3)

            # Confidence plot
            confidence_colors = [
                self.color_schemes["confidence"][c] for c in confidences
            ]
            ax2.barh(endpoints, [1] * len(endpoints), color=confidence_colors)
            ax2.set_xlabel("Confidence Level")
            ax2.set_xlim(0, 1)

            # Add confidence labels
            for i, conf in enumerate(confidences):
                ax2.text(
                    0.5,
                    i,
                    conf.replace("_", " ").title(),
                    ha="center",
                    va="center",
                    fontweight="bold",
                )

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                return f"Plot saved to {save_path}"
            else:
                plt.show()
                return "Toxicity profile plot displayed"

        except ImportError:
            # Fallback text-based visualization
            return self._create_text_toxicity_profile(
                toxicity_data, include_uncertainty
            )

    def plot_risk_assessment(
        self, risk_score: RiskScore, save_path: Optional[str] = None
    ) -> str:
        """Create risk assessment visualization."""

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

            # Overall risk gauge
            self._plot_risk_gauge(
                ax1, risk_score.overall_risk, risk_score.risk_category
            )

            # Endpoint contributions pie chart
            if risk_score.endpoint_contributions:
                wedges, texts, autotexts = ax2.pie(
                    risk_score.endpoint_contributions.values(),
                    labels=risk_score.endpoint_contributions.keys(),
                    autopct="%1.1f%%",
                    startangle=90,
                )
                ax2.set_title("Endpoint Contributions")

            # Exposure contributions
            if risk_score.exposure_contributions:
                ax3.bar(
                    risk_score.exposure_contributions.keys(),
                    risk_score.exposure_contributions.values(),
                )
                ax3.set_title("Exposure Contributions")
                ax3.tick_params(axis="x", rotation=45)

            # Monte Carlo uncertainty distribution
            if risk_score.monte_carlo_samples is not None:
                ax4.hist(
                    risk_score.monte_carlo_samples,
                    bins=50,
                    alpha=0.7,
                    color="skyblue",
                    edgecolor="black",
                )
                ax4.axvline(
                    risk_score.overall_risk,
                    color="red",
                    linestyle="--",
                    label=f"Point Estimate: {risk_score.overall_risk:.3f}",
                )
                ax4.axvline(
                    risk_score.uncertainty_range[0],
                    color="orange",
                    linestyle="--",
                    label="5th percentile",
                )
                ax4.axvline(
                    risk_score.uncertainty_range[1],
                    color="orange",
                    linestyle="--",
                    label="95th percentile",
                )
                ax4.set_xlabel("Risk Score")
                ax4.set_ylabel("Frequency")
                ax4.set_title("Uncertainty Distribution")
                ax4.legend()

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                return f"Risk assessment plot saved to {save_path}"
            else:
                plt.show()
                return "Risk assessment plot displayed"

        except ImportError:
            return self._create_text_risk_assessment(risk_score)

    def plot_safer_design_optimization(
        self,
        candidates: List[SafeDesignCandidate],
        parent_smiles: str,
        save_path: Optional[str] = None,
    ) -> str:
        """Visualize safer design optimization results."""

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

            # Optimization score vs similarity
            scores = [c.optimization_score for c in candidates]
            similarities = [c.similarity_to_parent for c in candidates]
            viabilities = [c.is_viable for c in candidates]

            scatter_colors = ["green" if v else "red" for v in viabilities]
            ax1.scatter(similarities, scores, c=scatter_colors, alpha=0.7)
            ax1.set_xlabel("Similarity to Parent")
            ax1.set_ylabel("Optimization Score")
            ax1.set_title("Optimization Score vs Similarity")
            ax1.grid(alpha=0.3)

            # Synthetic accessibility vs optimization score
            sa_scores = [c.synthetic_accessibility or 5.0 for c in candidates]
            ax2.scatter(sa_scores, scores, c=scatter_colors, alpha=0.7)
            ax2.set_xlabel("Synthetic Accessibility (1=easy, 10=hard)")
            ax2.set_ylabel("Optimization Score")
            ax2.set_title("Synthetic Accessibility vs Optimization")
            ax2.grid(alpha=0.3)

            # Top candidates comparison
            top_candidates = candidates[:10]
            candidate_names = [f"C{i+1}" for i in range(len(top_candidates))]
            overall_scores = [c.overall_desirability for c in top_candidates]

            bars = ax3.bar(
                candidate_names,
                overall_scores,
                color=["green" if c.is_viable else "red" for c in top_candidates],
            )
            ax3.set_xlabel("Candidate")
            ax3.set_ylabel("Overall Desirability")
            ax3.set_title("Top 10 Candidates")
            ax3.tick_params(axis="x", rotation=45)

            # Constraint violations analysis
            all_violations = []
            for candidate in candidates:
                all_violations.extend(candidate.constraint_violations)

            if all_violations:
                violation_counts = {}
                for violation in all_violations:
                    violation_counts[violation] = violation_counts.get(violation, 0) + 1

                top_violations = sorted(
                    violation_counts.items(), key=lambda x: x[1], reverse=True
                )[:8]

                if top_violations:
                    violation_names = [
                        v[0][:20] + "..." if len(v[0]) > 20 else v[0]
                        for v in top_violations
                    ]
                    violation_counts_list = [v[1] for v in top_violations]

                    ax4.barh(violation_names, violation_counts_list)
                    ax4.set_xlabel("Number of Violations")
                    ax4.set_title("Most Common Constraint Violations")

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                return f"Safer design optimization plot saved to {save_path}"
            else:
                plt.show()
                return "Safer design optimization plot displayed"

        except ImportError:
            return self._create_text_optimization_summary(candidates, parent_smiles)

    def create_risk_landscape(
        self,
        compounds: List[str],
        toxicity_data_list: List[ToxicityData],
        risk_scores: List[RiskScore],
    ) -> RiskLandscape:
        """Create risk landscape for multiple compounds."""

        # Extract data
        compound_names = compounds
        risk_values = [rs.overall_risk for rs in risk_scores]
        confidence_levels = [rs.confidence.value for rs in risk_scores]
        endpoint_contributions = [rs.endpoint_contributions for rs in risk_scores]

        # Create 2D coordinates using dimensionality reduction
        coordinates = self._create_2d_coordinates(toxicity_data_list)

        return RiskLandscape(
            compounds=compound_names,
            risk_scores=risk_values,
            confidence_levels=confidence_levels,
            endpoint_contributions=endpoint_contributions,
            coordinates=coordinates,
        )

    def plot_risk_landscape(
        self, risk_landscape: RiskLandscape, save_path: Optional[str] = None
    ) -> str:
        """Plot risk landscape visualization."""

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Risk score scatter plot
            if risk_landscape.coordinates is not None:
                scatter = ax1.scatter(
                    risk_landscape.coordinates[:, 0],
                    risk_landscape.coordinates[:, 1],
                    c=risk_landscape.risk_scores,
                    cmap="RdYlGn_r",
                    s=100,
                    alpha=0.7,
                )

                # Add compound labels
                for i, compound in enumerate(risk_landscape.compounds):
                    ax1.annotate(
                        compound[:10] + "..." if len(compound) > 10 else compound,
                        (
                            risk_landscape.coordinates[i, 0],
                            risk_landscape.coordinates[i, 1],
                        ),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.8,
                    )

                plt.colorbar(scatter, ax=ax1, label="Risk Score")
                ax1.set_xlabel("PC1")
                ax1.set_ylabel("PC2")
                ax1.set_title("Chemical Risk Landscape")

            # Risk distribution histogram
            ax2.hist(
                risk_landscape.risk_scores,
                bins=20,
                alpha=0.7,
                color="skyblue",
                edgecolor="black",
            )
            ax2.set_xlabel("Risk Score")
            ax2.set_ylabel("Number of Compounds")
            ax2.set_title("Risk Score Distribution")
            ax2.grid(alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                return f"Risk landscape plot saved to {save_path}"
            else:
                plt.show()
                return "Risk landscape plot displayed"

        except ImportError:
            return self._create_text_risk_landscape(risk_landscape)

    def generate_html_report(
        self,
        toxicity_data: ToxicityData,
        risk_score: Optional[RiskScore] = None,
        optimization_results: Optional[List[SafeDesignCandidate]] = None,
        output_path: str = "toxicology_report.html",
    ) -> str:
        """Generate comprehensive HTML report."""

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Comprehensive Toxicology Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007bff; }}
                .risk-high {{ border-left-color: #dc3545; }}
                .risk-moderate {{ border-left-color: #ffc107; }}
                .risk-low {{ border-left-color: #28a745; }}
                .prediction-table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                .prediction-table th, .prediction-table td {{
                    border: 1px solid #ddd; padding: 8px; text-align: left;
                }}
                .prediction-table th {{ background-color: #f2f2f2; }}
                .confidence-high {{ background-color: #d4edda; }}
                .confidence-moderate {{ background-color: #fff3cd; }}
                .confidence-low {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Comprehensive Toxicology Report</h1>
                <h2>Compound: {toxicity_data.smiles}</h2>
                <p>Overall Risk Score: {toxicity_data.overall_risk_score:.3f}</p>
            </div>

            <div class="section">
                <h3>Toxicity Predictions</h3>
                <table class="prediction-table">
                    <thead>
                        <tr>
                            <th>Endpoint</th>
                            <th>Species</th>
                            <th>Predicted Value</th>
                            <th>Confidence</th>
                            <th>Uncertainty</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        # Add toxicity predictions
        for pred in toxicity_data.predicted_endpoints:
            confidence_class = f"confidence-{pred.confidence.value.replace('_', '-')}"
            html_content += f"""
                        <tr class="{confidence_class}">
                            <td>{pred.endpoint.endpoint_type.value.replace('_', ' ').title()}</td>
                            <td>{pred.endpoint.species.value.title()}</td>
                            <td>{pred.predicted_value:.3f}</td>
                            <td>{pred.confidence.value.replace('_', ' ').title()}</td>
                            <td>{pred.uncertainty:.3f}</td>
                        </tr>
            """

        html_content += """
                    </tbody>
                </table>
            </div>
        """

        # Add risk assessment if provided
        if risk_score:
            risk_class = f"risk-{risk_score.risk_category.value.replace('_', '-')}"
            html_content += f"""
            <div class="section {risk_class}">
                <h3>Risk Assessment</h3>
                <p><strong>Risk Category:</strong> {risk_score.risk_category.value.replace('_', ' ').title()}</p>
                <p><strong>Overall Risk:</strong> {risk_score.overall_risk:.3f}</p>
                <p><strong>Confidence:</strong> {risk_score.confidence.value.replace('_', ' ').title()}</p>
                <p><strong>Uncertainty Range:</strong> {risk_score.uncertainty_range[0]:.3f} - {risk_score.uncertainty_range[1]:.3f}</p>

                <h4>Key Risk Drivers:</h4>
                <ul>
            """

            for driver in risk_score.key_drivers:
                html_content += f"<li>{driver}</li>"

            html_content += """
                </ul>
            </div>
            """

        # Add optimization results if provided
        if optimization_results:
            html_content += """
            <div class="section">
                <h3>Safer Design Optimization Results</h3>
                <table class="prediction-table">
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>SMILES</th>
                            <th>Overall Score</th>
                            <th>Viable</th>
                            <th>Similarity</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            for i, candidate in enumerate(optimization_results[:10]):
                viable_text = "Yes" if candidate.is_viable else "No"
                html_content += f"""
                        <tr>
                            <td>{i+1}</td>
                            <td>{candidate.smiles}</td>
                            <td>{candidate.overall_desirability:.3f}</td>
                            <td>{viable_text}</td>
                            <td>{candidate.similarity_to_parent:.3f}</td>
                        </tr>
                """

            html_content += """
                    </tbody>
                </table>
            </div>
            """

        html_content += """
        </body>
        </html>
        """

        # Write to file
        with open(output_path, "w") as f:
            f.write(html_content)

        return f"HTML report generated: {output_path}"

    def _get_toxicity_color(self, value: float) -> str:
        """Get color based on toxicity value."""
        if value < 10:
            return "#E74C3C"  # High toxicity - red
        elif value < 100:
            return "#F39C12"  # Moderate toxicity - orange
        elif value < 1000:
            return "#F1C40F"  # Low toxicity - yellow
        else:
            return "#2ECC71"  # Very low toxicity - green

    def _plot_risk_gauge(self, ax, risk_value: float, risk_category) -> None:
        """Plot risk gauge chart."""
        try:
            import matplotlib.patches as patches

            # Create semicircle gauge
            theta = np.linspace(0, np.pi, 100)
            x = np.cos(theta)
            y = np.sin(theta)

            # Background semicircle
            ax.fill_between(x, 0, y, alpha=0.3, color="lightgray")

            # Risk zones
            zones = [
                (0.0, 0.2, "#2ECC71"),  # Low
                (0.2, 0.5, "#F39C12"),  # Moderate
                (0.5, 0.8, "#E67E22"),  # High
                (0.8, 1.0, "#E74C3C"),  # Very high
            ]

            for start, end, color in zones:
                start_angle = start * np.pi
                end_angle = end * np.pi
                theta_zone = np.linspace(start_angle, end_angle, 50)
                x_zone = np.cos(theta_zone)
                y_zone = np.sin(theta_zone)
                ax.fill_between(x_zone, 0, y_zone, alpha=0.5, color=color)

            # Risk needle
            needle_angle = risk_value * np.pi
            needle_x = 0.8 * np.cos(needle_angle)
            needle_y = 0.8 * np.sin(needle_angle)
            ax.arrow(
                0,
                0,
                needle_x,
                needle_y,
                head_width=0.05,
                head_length=0.1,
                fc="black",
                ec="black",
                linewidth=3,
            )

            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(0, 1.1)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(
                f'Risk Score: {risk_value:.3f}\n({risk_category.value.replace("_", " ").title()})'
            )

        except Exception:
            # Fallback simple bar
            ax.bar([0], [risk_value], color=self._get_risk_color(risk_value))
            ax.set_ylim(0, 1)
            ax.set_title(f"Risk: {risk_value:.3f}")

    def _get_risk_color(self, risk_value: float) -> str:
        """Get color for risk value."""
        if risk_value < 0.2:
            return "#2ECC71"  # Green
        elif risk_value < 0.5:
            return "#F39C12"  # Orange
        elif risk_value < 0.8:
            return "#E67E22"  # Dark orange
        else:
            return "#E74C3C"  # Red

    def _create_2d_coordinates(
        self, toxicity_data_list: List[ToxicityData]
    ) -> np.ndarray:
        """Create 2D coordinates for compounds using PCA."""

        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            # Extract features from toxicity data
            features = []
            for toxicity_data in toxicity_data_list:
                feature_vector = []
                for pred in toxicity_data.predicted_endpoints:
                    feature_vector.append(pred.predicted_value)
                    feature_vector.append(pred.uncertainty)

                # Pad or truncate to fixed size
                target_size = 20
                if len(feature_vector) < target_size:
                    feature_vector.extend([0.0] * (target_size - len(feature_vector)))
                else:
                    feature_vector = feature_vector[:target_size]

                features.append(feature_vector)

            if len(features) < 2:
                # Not enough data for PCA
                return np.random.rand(len(features), 2)

            # Standardize and apply PCA
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            pca = PCA(n_components=2)
            coordinates = pca.fit_transform(features_scaled)

            return coordinates

        except ImportError:
            # Fallback random coordinates
            return np.random.rand(len(toxicity_data_list), 2)

    def _create_text_toxicity_profile(
        self, toxicity_data: ToxicityData, include_uncertainty: bool
    ) -> str:
        """Create text-based toxicity profile."""

        report = f"Toxicity Profile for {toxicity_data.smiles}\n"
        report += "=" * 50 + "\n\n"

        for pred in toxicity_data.predicted_endpoints:
            report += f"Endpoint: {pred.endpoint.endpoint_type.value.replace('_', ' ').title()}\n"
            report += f"  Species: {pred.endpoint.species.value.title()}\n"
            report += f"  Predicted Value: {pred.predicted_value:.3f}\n"
            report += (
                f"  Confidence: {pred.confidence.value.replace('_', ' ').title()}\n"
            )

            if include_uncertainty:
                report += f"  Uncertainty: {pred.uncertainty:.3f}\n"

            report += "\n"

        report += f"Overall Risk Score: {toxicity_data.overall_risk_score:.3f}\n"

        return report

    def _create_text_risk_assessment(self, risk_score: RiskScore) -> str:
        """Create text-based risk assessment."""

        report = "Risk Assessment Summary\n"
        report += "=" * 30 + "\n\n"
        report += f"Overall Risk: {risk_score.overall_risk:.3f}\n"
        report += f"Risk Category: {risk_score.risk_category.value.replace('_', ' ').title()}\n"
        report += (
            f"Confidence: {risk_score.confidence.value.replace('_', ' ').title()}\n"
        )
        report += f"Uncertainty Range: {risk_score.uncertainty_range[0]:.3f} - {risk_score.uncertainty_range[1]:.3f}\n\n"

        if risk_score.key_drivers:
            report += "Key Risk Drivers:\n"
            for driver in risk_score.key_drivers:
                report += f"- {driver}\n"

        return report

    def _create_text_optimization_summary(
        self, candidates: List[SafeDesignCandidate], parent_smiles: str
    ) -> str:
        """Create text-based optimization summary."""

        report = "Safer Design Optimization Results\n"
        report += f"Parent: {parent_smiles}\n"
        report += "=" * 50 + "\n\n"

        viable_count = sum(1 for c in candidates if c.is_viable)
        report += f"Total Candidates: {len(candidates)}\n"
        report += f"Viable Candidates: {viable_count}\n"
        report += f"Success Rate: {viable_count/len(candidates)*100:.1f}%\n\n"

        report += "Top 5 Candidates:\n"
        for i, candidate in enumerate(candidates[:5]):
            report += f"{i+1}. {candidate.smiles}\n"
            report += f"   Score: {candidate.overall_desirability:.3f}\n"
            report += f"   Viable: {'Yes' if candidate.is_viable else 'No'}\n"
            report += f"   Similarity: {candidate.similarity_to_parent:.3f}\n\n"

        return report

    def _create_text_risk_landscape(self, risk_landscape: RiskLandscape) -> str:
        """Create text-based risk landscape summary."""

        report = "Risk Landscape Summary\n"
        report += "=" * 30 + "\n\n"

        avg_risk = np.mean(risk_landscape.risk_scores)
        max_risk = max(risk_landscape.risk_scores)
        min_risk = min(risk_landscape.risk_scores)

        report += f"Number of Compounds: {len(risk_landscape.compounds)}\n"
        report += f"Average Risk: {avg_risk:.3f}\n"
        report += f"Risk Range: {min_risk:.3f} - {max_risk:.3f}\n\n"

        # Sort by risk and show top/bottom compounds
        risk_data = list(zip(risk_landscape.compounds, risk_landscape.risk_scores))
        risk_data.sort(key=lambda x: x[1], reverse=True)

        report += "Highest Risk Compounds:\n"
        for compound, risk in risk_data[:3]:
            report += f"- {compound}: {risk:.3f}\n"

        report += "\nLowest Risk Compounds:\n"
        for compound, risk in risk_data[-3:]:
            report += f"- {compound}: {risk:.3f}\n"

        return report
