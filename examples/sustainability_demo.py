"""
Comprehensive Sustainability Assessment Demo.

This example demonstrates the full sustainability assessment capabilities
including degradation modeling, recyclability assessment, and LCA integration
for materials science applications.
"""

import asyncio
import json
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chemi_accel.sustainability import (  # Degradation modeling; Recyclability assessment; LCA integration
    ComprehensiveRecyclabilityAssessor,
    DegradationMechanism,
    EnvironmentalStressor,
    ImpactCategory,
    LCAComparator,
    LCADataManager,
    MaterialCategory,
    MatrixLCACalculator,
    MultiScaleDegradationPredictor,
    RecyclingMethod,
    SimulationScale,
    create_accelerated_aging_conditions,
    create_biodegradable_composition,
    create_outdoor_weathering_conditions,
    create_polymer_manufacturing_process,
    create_recycling_process,
    create_standard_material,
    create_standard_plastic_composition,
)

# Check for optional dependencies
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SustainabilityDemo:
    """Comprehensive sustainability assessment demonstration."""

    def __init__(self, use_gpu: bool = True, verbose: bool = True):
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        self.verbose = verbose
        self.results = {}
        self.output_dir = None

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Using CPU-only implementations.")

        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Visualization features disabled.")

    def _log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            logger.info(message)

    async def demo_degradation_modeling(self) -> Dict[str, Any]:
        """Demonstrate comprehensive degradation modeling."""
        self._log("=== Degradation Modeling Demo ===")

        # Create multi-scale predictor
        predictor = MultiScaleDegradationPredictor(use_gpu=self.use_gpu)

        # Test materials
        materials = {
            "PET_Bottle": create_standard_material("polymer"),
            "Carbon_Fiber_Composite": create_standard_material("composite"),
        }

        # Environmental conditions
        conditions = {
            "Accelerated_Aging": create_accelerated_aging_conditions(
                temperature=353.15,  # 80°C
                humidity=0.85,
                uv_intensity=0.8,
                duration=1000.0,  # hours
            ),
            "Tropical_Climate": create_outdoor_weathering_conditions(
                "tropical", duration=8760.0  # 1 year
            ),
            "Arid_Climate": create_outdoor_weathering_conditions(
                "arid", duration=8760.0
            ),
        }

        degradation_results = {}

        for material_name, material in materials.items():
            self._log(f"Analyzing degradation for {material_name}")
            material_results = {}

            for condition_name, condition in conditions.items():
                self._log(f"  Under {condition_name} conditions")

                # Run multi-scale prediction
                prediction = await predictor.predict_degradation(material, condition)

                # Extract key metrics
                final_state = (
                    prediction.degradation_states[-1]
                    if prediction.degradation_states
                    else None
                )

                material_results[condition_name] = {
                    "half_life": prediction.half_life,
                    "failure_time": prediction.failure_time,
                    "final_mw_retention": final_state.molecular_weight_retention
                    if final_state
                    else None,
                    "dominant_mechanisms": [
                        mech.value for mech in prediction.dominant_mechanisms
                    ],
                    "uncertainty": prediction.uncertainty,
                    "degradation_trajectory": [
                        {
                            "time": state.time,
                            "mw_retention": state.molecular_weight_retention,
                            "mechanical_retention": state.mechanical_retention,
                        }
                        for state in prediction.degradation_states[
                            :10
                        ]  # First 10 points
                    ],
                }

            degradation_results[material_name] = material_results

        # Analyze results
        summary = self._analyze_degradation_results(degradation_results)

        self._log(
            f"Degradation modeling completed for {len(materials)} materials under {len(conditions)} conditions"
        )

        return {
            "materials_analyzed": len(materials),
            "conditions_tested": len(conditions),
            "degradation_results": degradation_results,
            "summary": summary,
        }

    def _analyze_degradation_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze degradation modeling results."""

        summary = {
            "most_durable_material": None,
            "most_aggressive_condition": None,
            "common_mechanisms": {},
            "performance_ranking": [],
        }

        # Find most durable material (highest retention under accelerated aging)
        best_retention = 0
        for material, conditions in results.items():
            if "Accelerated_Aging" in conditions:
                retention = conditions["Accelerated_Aging"]["final_mw_retention"]
                if retention and retention > best_retention:
                    best_retention = retention
                    summary["most_durable_material"] = material

        # Count degradation mechanisms
        all_mechanisms = []
        for material_results in results.values():
            for condition_results in material_results.values():
                all_mechanisms.extend(condition_results["dominant_mechanisms"])

        from collections import Counter

        mechanism_counts = Counter(all_mechanisms)
        summary["common_mechanisms"] = dict(mechanism_counts.most_common(5))

        return summary

    async def demo_recyclability_assessment(self) -> Dict[str, Any]:
        """Demonstrate recyclability assessment."""
        self._log("=== Recyclability Assessment Demo ===")

        # Create comprehensive assessor
        assessor = ComprehensiveRecyclabilityAssessor(use_gpu=self.use_gpu)

        # Test materials
        materials = {
            "PET_Bottle": create_standard_plastic_composition("PET"),
            "HDPE_Container": create_standard_plastic_composition("PE"),
            "PP_Packaging": create_standard_plastic_composition("PP"),
            "PLA_Film": create_biodegradable_composition("PLA"),
            "PHA_Coating": create_biodegradable_composition("PHA"),
        }

        recyclability_results = {}

        for material_name, composition in materials.items():
            self._log(f"Assessing recyclability for {material_name}")

            # Run comprehensive assessment
            assessment = await assessor.assess_comprehensive_recyclability(composition)

            # Extract key results
            recyclability_results[material_name] = {
                "material_category": assessment.material_category.value,
                "recommended_methods": [
                    method.value for method in assessment.recommended_methods
                ],
                "recyclability_scores": {
                    method.value: {
                        "overall_score": score.overall_score,
                        "feasibility": score.feasibility,
                        "confidence": score.confidence,
                        "technical_barriers": score.technical_barriers,
                        "advantages": score.advantages,
                    }
                    for method, score in assessment.recycling_scores.items()
                },
                "circularity_metrics": {
                    "circularity_index": assessment.circularity_metrics.circularity_index,
                    "recyclability_score": assessment.circularity_metrics.recyclability_score,
                    "reusability_score": assessment.circularity_metrics.reusability_score,
                    "biodegradability_score": assessment.circularity_metrics.biodegradability_score,
                    "economic_viability_score": assessment.circularity_metrics.economic_viability_score,
                },
                "contamination_sensitivity": assessment.contamination_sensitivity,
            }

        # Analyze results
        summary = self._analyze_recyclability_results(recyclability_results)

        self._log(f"Recyclability assessment completed for {len(materials)} materials")

        return {
            "materials_assessed": len(materials),
            "recyclability_results": recyclability_results,
            "summary": summary,
        }

    def _analyze_recyclability_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze recyclability assessment results."""

        summary = {
            "most_recyclable_material": None,
            "best_recycling_method": None,
            "biodegradable_materials": [],
            "circularity_ranking": [],
        }

        # Find most recyclable material (highest circularity index)
        best_circularity = 0
        for material, result in results.items():
            circularity = result["circularity_metrics"]["circularity_index"]
            if circularity > best_circularity:
                best_circularity = circularity
                summary["most_recyclable_material"] = material

        # Find biodegradable materials
        for material, result in results.items():
            if result["circularity_metrics"]["biodegradability_score"] > 0.5:
                summary["biodegradable_materials"].append(material)

        # Rank by circularity
        circularity_scores = [
            (material, result["circularity_metrics"]["circularity_index"])
            for material, result in results.items()
        ]
        summary["circularity_ranking"] = sorted(
            circularity_scores, key=lambda x: x[1], reverse=True
        )

        # Find most common recommended method
        all_methods = []
        for result in results.values():
            all_methods.extend(result["recommended_methods"])

        if all_methods:
            from collections import Counter

            method_counts = Counter(all_methods)
            summary["best_recycling_method"] = method_counts.most_common(1)[0][0]

        return summary

    async def demo_lca_integration(self) -> Dict[str, Any]:
        """Demonstrate LCA integration capabilities."""
        self._log("=== LCA Integration Demo ===")

        # Create LCA calculator
        calculator = MatrixLCACalculator(use_gpu=self.use_gpu)
        data_manager = LCADataManager()

        # Get characterization factors
        characterization_factors = (
            data_manager.create_default_characterization_factors()
        )

        # Create LCA scenarios
        scenarios = {
            "Virgin_PET_Production": (
                [create_polymer_manufacturing_process("PET")],
                characterization_factors,
            ),
            "Virgin_PE_Production": (
                [create_polymer_manufacturing_process("PE")],
                characterization_factors,
            ),
            "Mechanical_PET_Recycling": (
                [create_recycling_process("mechanical", "PET")],
                characterization_factors,
            ),
            "Chemical_PET_Recycling": (
                [create_recycling_process("chemical", "PET")],
                characterization_factors,
            ),
        }

        # Run LCA calculations
        lca_results = {}

        for scenario_name, (processes, factors) in scenarios.items():
            self._log(f"Calculating LCA for {scenario_name}")

            results = calculator.calculate_impacts(processes, factors)

            lca_results[scenario_name] = {
                "impact_scores": results.impact_scores,
                "calculation_time": results.calculation_time,
                "hotspots": [
                    {
                        "process": hotspot[0],
                        "impact": hotspot[1].value,
                        "contribution": hotspot[2],
                    }
                    for hotspot in results.hotspots[:5]  # Top 5 hotspots
                ],
                "uncertainty": results.uncertainty_analysis,
            }

        # Compare scenarios
        comparator = LCAComparator(calculator)
        comparison = comparator.compare_scenarios(scenarios)

        # Analyze LCA results
        summary = self._analyze_lca_results(lca_results, comparison)

        self._log(f"LCA integration completed for {len(scenarios)} scenarios")

        return {
            "scenarios_analyzed": len(scenarios),
            "lca_results": lca_results,
            "comparison": {
                "scenarios": comparison.scenarios,
                "dominant_scenarios": {
                    cat.value: scenario
                    for cat, scenario in comparison.dominant_scenarios.items()
                },
                "recommendations": comparison.recommendations,
                "trade_offs": {
                    f"{k[0].value}_vs_{k[1].value}": v
                    for k, v in comparison.trade_offs.items()
                },
            },
            "summary": summary,
        }

    def _analyze_lca_results(
        self, results: Dict[str, Any], comparison: Any
    ) -> Dict[str, Any]:
        """Analyze LCA results."""

        summary = {
            "lowest_impact_scenario": {},
            "highest_impact_scenario": {},
            "recycling_benefits": {},
            "calculation_performance": {},
        }

        # Find scenarios with lowest/highest impact for each category
        for category in ImpactCategory:
            cat_value = category.value

            scenario_impacts = [
                (scenario, result["impact_scores"].get(category, float("inf")))
                for scenario, result in results.items()
            ]

            # Sort by impact (lower is better)
            scenario_impacts.sort(key=lambda x: abs(x[1]))

            if scenario_impacts:
                summary["lowest_impact_scenario"][cat_value] = scenario_impacts[0][0]
                summary["highest_impact_scenario"][cat_value] = scenario_impacts[-1][0]

        # Calculate recycling benefits
        virgin_pet_climate = (
            results.get("Virgin_PET_Production", {})
            .get("impact_scores", {})
            .get(ImpactCategory.CLIMATE_CHANGE, 0)
        )
        mech_recycling_climate = (
            results.get("Mechanical_PET_Recycling", {})
            .get("impact_scores", {})
            .get(ImpactCategory.CLIMATE_CHANGE, 0)
        )

        if virgin_pet_climate and mech_recycling_climate:
            benefit = (
                (virgin_pet_climate - mech_recycling_climate) / virgin_pet_climate
            ) * 100
            summary["recycling_benefits"][
                "mechanical_recycling_climate_benefit_percent"
            ] = benefit

        # Performance metrics
        calc_times = [result["calculation_time"] for result in results.values()]
        summary["calculation_performance"] = {
            "average_time": np.mean(calc_times),
            "total_time": sum(calc_times),
            "fastest_calculation": min(calc_times),
            "slowest_calculation": max(calc_times),
        }

        return summary

    async def demo_integrated_material_assessment(self) -> Dict[str, Any]:
        """Demonstrate integrated assessment combining all three modules."""
        self._log("=== Integrated Material Assessment Demo ===")

        # Select a test material (PET bottle)
        material_name = "PET_Bottle_Comprehensive"

        # Degradation modeling
        self._log("Running degradation modeling...")
        degradation_predictor = MultiScaleDegradationPredictor(use_gpu=self.use_gpu)
        material_props = create_standard_material("polymer")
        aging_conditions = create_accelerated_aging_conditions(duration=2000.0)

        degradation_prediction = await degradation_predictor.predict_degradation(
            material_props, aging_conditions
        )

        # Recyclability assessment
        self._log("Running recyclability assessment...")
        recyclability_assessor = ComprehensiveRecyclabilityAssessor(
            use_gpu=self.use_gpu
        )
        material_composition = create_standard_plastic_composition("PET")

        recyclability_assessment = (
            await recyclability_assessor.assess_comprehensive_recyclability(
                material_composition
            )
        )

        # LCA assessment
        self._log("Running LCA assessment...")
        lca_calculator = MatrixLCACalculator(use_gpu=self.use_gpu)
        data_manager = LCADataManager()
        characterization_factors = (
            data_manager.create_default_characterization_factors()
        )

        # Compare virgin production vs recycling
        virgin_process = create_polymer_manufacturing_process("PET")
        recycling_process = create_recycling_process("mechanical", "PET")

        virgin_lca = lca_calculator.calculate_impacts(
            [virgin_process], characterization_factors
        )
        recycling_lca = lca_calculator.calculate_impacts(
            [recycling_process], characterization_factors
        )

        # Integrate results
        integrated_results = {
            "material_name": material_name,
            "degradation_analysis": {
                "half_life_hours": degradation_prediction.half_life,
                "dominant_mechanisms": [
                    mech.value for mech in degradation_prediction.dominant_mechanisms
                ],
                "final_property_retention": degradation_prediction.degradation_states[
                    -1
                ].molecular_weight_retention
                if degradation_prediction.degradation_states
                else None,
            },
            "recyclability_analysis": {
                "circularity_index": recyclability_assessment.circularity_metrics.circularity_index,
                "recommended_methods": [
                    method.value
                    for method in recyclability_assessment.recommended_methods
                ],
                "best_recycling_score": max(
                    score.overall_score
                    for score in recyclability_assessment.recycling_scores.values()
                )
                if recyclability_assessment.recycling_scores
                else 0,
            },
            "lca_analysis": {
                "virgin_production_impacts": virgin_lca.impact_scores,
                "recycling_impacts": recycling_lca.impact_scores,
                "recycling_benefits": {
                    cat.value: (
                        virgin_lca.impact_scores.get(cat, 0)
                        - recycling_lca.impact_scores.get(cat, 0)
                    )
                    for cat in ImpactCategory
                    if cat in virgin_lca.impact_scores
                    and cat in recycling_lca.impact_scores
                },
            },
        }

        # Generate sustainability recommendations
        recommendations = self._generate_sustainability_recommendations(
            integrated_results
        )
        integrated_results["recommendations"] = recommendations

        # Calculate overall sustainability score
        sustainability_score = self._calculate_sustainability_score(integrated_results)
        integrated_results["overall_sustainability_score"] = sustainability_score

        self._log(
            f"Integrated assessment completed. Sustainability score: {sustainability_score:.2f}/10"
        )

        return integrated_results

    def _generate_sustainability_recommendations(
        self, results: Dict[str, Any]
    ) -> List[str]:
        """Generate sustainability recommendations based on integrated results."""

        recommendations = []

        # Durability recommendations
        if results["degradation_analysis"]["half_life_hours"]:
            if results["degradation_analysis"]["half_life_hours"] < 1000:
                recommendations.append(
                    "Consider UV stabilizers to improve outdoor durability"
                )

        # Check dominant degradation mechanisms
        mechanisms = results["degradation_analysis"]["dominant_mechanisms"]
        if "uv_degradation" in mechanisms:
            recommendations.append(
                "Add UV absorbers and HALS (Hindered Amine Light Stabilizers)"
            )
        if "thermal_degradation" in mechanisms:
            recommendations.append(
                "Incorporate antioxidants to prevent thermal degradation"
            )
        if "hydrolytic_degradation" in mechanisms:
            recommendations.append("Use moisture barriers and desiccants in packaging")

        # Recyclability recommendations
        circularity = results["recyclability_analysis"]["circularity_index"]
        if circularity < 0.6:
            recommendations.append("Design for disassembly to improve recyclability")
            recommendations.append("Reduce use of additives that hinder recycling")

        # LCA recommendations
        recycling_benefits = results["lca_analysis"]["recycling_benefits"]
        climate_benefit = recycling_benefits.get("climate_change", 0)

        if climate_benefit > 0:
            recommendations.append(
                f"Mechanical recycling reduces climate impact by {climate_benefit:.1f} kg CO2-eq"
            )
        else:
            recommendations.append("Investigate alternative end-of-life options")

        # General recommendations
        if circularity > 0.7:
            recommendations.append(
                "Excellent circularity - promote closed-loop recycling"
            )

        return recommendations

    def _calculate_sustainability_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall sustainability score (0-10 scale)."""

        scores = []

        # Durability score (0-3 points)
        half_life = results["degradation_analysis"]["half_life_hours"]
        if half_life:
            if half_life > 5000:
                durability_score = 3.0
            elif half_life > 2000:
                durability_score = 2.0
            elif half_life > 1000:
                durability_score = 1.0
            else:
                durability_score = 0.5
        else:
            durability_score = 1.0  # Default
        scores.append(durability_score)

        # Circularity score (0-4 points)
        circularity = results["recyclability_analysis"]["circularity_index"]
        circularity_score = circularity * 4.0
        scores.append(circularity_score)

        # Environmental impact score (0-3 points)
        # Based on recycling benefits
        recycling_benefits = results["lca_analysis"]["recycling_benefits"]
        climate_benefit = recycling_benefits.get("climate_change", 0)

        if climate_benefit > 2.0:
            impact_score = 3.0
        elif climate_benefit > 1.0:
            impact_score = 2.0
        elif climate_benefit > 0:
            impact_score = 1.0
        else:
            impact_score = 0.5
        scores.append(impact_score)

        return sum(scores)

    def create_sustainability_visualization(self, results: Dict[str, Any]) -> str:
        """Create sustainability assessment visualization."""

        if not PLOTLY_AVAILABLE:
            self._log("Plotly not available. Skipping visualization.")
            return "Visualization not available"

        # Create comprehensive sustainability dashboard
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Degradation Timeline",
                "Recyclability Scores",
                "LCA Impact Comparison",
                "Sustainability Metrics",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "indicator"}],
            ],
        )

        # Plot degradation timeline (if data available)
        if (
            "degradation_modeling" in results
            and results["degradation_modeling"]["degradation_results"]
        ):
            # Get first material's first condition trajectory
            first_material = list(
                results["degradation_modeling"]["degradation_results"].keys()
            )[0]
            first_condition = list(
                results["degradation_modeling"]["degradation_results"][
                    first_material
                ].keys()
            )[0]
            trajectory = results["degradation_modeling"]["degradation_results"][
                first_material
            ][first_condition]["degradation_trajectory"]

            if trajectory:
                times = [point["time"] for point in trajectory]
                mw_retentions = [point["mw_retention"] for point in trajectory]

                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=mw_retentions,
                        mode="lines+markers",
                        name="MW Retention",
                    ),
                    row=1,
                    col=1,
                )

        # Plot recyclability scores (if data available)
        if (
            "recyclability_assessment" in results
            and results["recyclability_assessment"]["recyclability_results"]
        ):
            materials = []
            circularity_scores = []

            for material, data in results["recyclability_assessment"][
                "recyclability_results"
            ].items():
                materials.append(material)
                circularity_scores.append(
                    data["circularity_metrics"]["circularity_index"]
                )

            fig.add_trace(
                go.Bar(x=materials, y=circularity_scores, name="Circularity Index"),
                row=1,
                col=2,
            )

        # Plot LCA comparison (if data available)
        if "lca_integration" in results and results["lca_integration"]["lca_results"]:
            scenarios = []
            climate_impacts = []

            for scenario, data in results["lca_integration"]["lca_results"].items():
                scenarios.append(scenario)
                climate_impacts.append(
                    data["impact_scores"].get(ImpactCategory.CLIMATE_CHANGE, 0)
                )

            fig.add_trace(
                go.Bar(x=scenarios, y=climate_impacts, name="Climate Change Impact"),
                row=2,
                col=1,
            )

        # Add sustainability score indicator
        if "integrated_assessment" in results:
            score = results["integrated_assessment"]["overall_sustainability_score"]

            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=score,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Sustainability Score"},
                    gauge={
                        "axis": {"range": [None, 10]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 4], "color": "lightgray"},
                            {"range": [4, 7], "color": "yellow"},
                            {"range": [7, 10], "color": "green"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 8,
                        },
                    },
                ),
                row=2,
                col=2,
            )

        fig.update_layout(
            title="Comprehensive Sustainability Assessment Dashboard",
            height=800,
            showlegend=True,
        )

        # Save to file
        if self.output_dir:
            viz_path = self.output_dir / "sustainability_dashboard.html"
            fig.write_html(str(viz_path))
            return str(viz_path)

        return "Dashboard created (no output directory specified)"

    async def run_all_demos(self) -> Dict[str, Any]:
        """Run all sustainability demonstration scenarios."""
        self._log("Starting Comprehensive Sustainability Assessment Demonstrations")
        self._log("=" * 80)

        # Create output directory
        self.output_dir = Path(tempfile.mkdtemp(prefix="sustainability_demo_"))
        self._log(f"Output directory: {self.output_dir}")

        all_results = {}

        try:
            # Demo 1: Degradation Modeling
            all_results["degradation_modeling"] = await self.demo_degradation_modeling()

            # Demo 2: Recyclability Assessment
            all_results[
                "recyclability_assessment"
            ] = await self.demo_recyclability_assessment()

            # Demo 3: LCA Integration
            all_results["lca_integration"] = await self.demo_lca_integration()

            # Demo 4: Integrated Assessment
            all_results[
                "integrated_assessment"
            ] = await self.demo_integrated_material_assessment()

            self._log("=" * 80)
            self._log("All demonstrations completed successfully!")

            # Create visualization
            viz_path = self.create_sustainability_visualization(all_results)
            all_results["visualization_path"] = viz_path

            # Generate summary report
            summary_report = self.generate_summary_report(all_results)
            all_results["summary_report"] = summary_report

            return all_results

        except Exception as e:
            logger.error(f"Error during demonstrations: {e}")
            all_results["error"] = str(e)
            return all_results

    def generate_summary_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary report."""

        report = {
            "executive_summary": {},
            "key_findings": {},
            "performance_metrics": {},
            "recommendations": {},
        }

        # Executive summary
        report["executive_summary"] = {
            "total_demonstrations": 4,
            "materials_analyzed": {
                "degradation": results.get("degradation_modeling", {}).get(
                    "materials_analyzed", 0
                ),
                "recyclability": results.get("recyclability_assessment", {}).get(
                    "materials_assessed", 0
                ),
                "lca": results.get("lca_integration", {}).get("scenarios_analyzed", 0),
            },
            "overall_sustainability_score": results.get(
                "integrated_assessment", {}
            ).get("overall_sustainability_score", 0),
        }

        # Key findings
        if "degradation_modeling" in results:
            degradation_summary = results["degradation_modeling"].get("summary", {})
            report["key_findings"]["degradation"] = {
                "most_durable_material": degradation_summary.get(
                    "most_durable_material"
                ),
                "common_degradation_mechanisms": degradation_summary.get(
                    "common_mechanisms", {}
                ),
            }

        if "recyclability_assessment" in results:
            recyclability_summary = results["recyclability_assessment"].get(
                "summary", {}
            )
            report["key_findings"]["recyclability"] = {
                "most_recyclable_material": recyclability_summary.get(
                    "most_recyclable_material"
                ),
                "best_recycling_method": recyclability_summary.get(
                    "best_recycling_method"
                ),
                "biodegradable_materials": recyclability_summary.get(
                    "biodegradable_materials", []
                ),
            }

        if "lca_integration" in results:
            lca_summary = results["lca_integration"].get("summary", {})
            report["key_findings"]["lca"] = {
                "recycling_benefits": lca_summary.get("recycling_benefits", {}),
                "lowest_impact_scenarios": lca_summary.get(
                    "lowest_impact_scenario", {}
                ),
            }

        # Performance metrics
        if "lca_integration" in results:
            performance = results["lca_integration"]["summary"].get(
                "calculation_performance", {}
            )
            report["performance_metrics"] = {
                "average_calculation_time": performance.get("average_time", 0),
                "total_computation_time": performance.get("total_time", 0),
                "gpu_acceleration_used": self.use_gpu and TORCH_AVAILABLE,
            }

        # Recommendations
        if "integrated_assessment" in results:
            report["recommendations"] = results["integrated_assessment"].get(
                "recommendations", []
            )

        return report


async def main():
    """Main demonstration function."""
    print("Comprehensive Sustainability Assessment Demo")
    print("=" * 80)

    # Initialize demo
    demo = SustainabilityDemo(use_gpu=TORCH_AVAILABLE, verbose=True)

    # Run all demonstrations
    results = await demo.run_all_demos()

    # Save results
    results_file = Path("sustainability_demo_results.json")

    # Convert any non-serializable objects
    def convert_for_json(obj):
        if hasattr(obj, "value"):  # Handle Enum objects
            return obj.value
        elif isinstance(obj, (datetime, np.integer, np.floating)):
            return str(obj)
        elif isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj

    serializable_results = convert_for_json(results)

    with open(results_file, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)

    print(f"\nResults saved to {results_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("DEMONSTRATION SUMMARY")
    print("=" * 80)

    if "summary_report" in results:
        report = results["summary_report"]

        print("\nExecutive Summary:")
        exec_summary = report.get("executive_summary", {})
        print(f"  Total Demonstrations: {exec_summary.get('total_demonstrations', 0)}")
        print(
            f"  Overall Sustainability Score: {exec_summary.get('overall_sustainability_score', 0):.2f}/10"
        )

        materials_analyzed = exec_summary.get("materials_analyzed", {})
        print("  Materials Analyzed:")
        for category, count in materials_analyzed.items():
            print(f"    {category.title()}: {count}")

        print("\nKey Findings:")
        findings = report.get("key_findings", {})

        if "degradation" in findings:
            deg_findings = findings["degradation"]
            print(
                f"  Most Durable Material: {deg_findings.get('most_durable_material', 'N/A')}"
            )

        if "recyclability" in findings:
            rec_findings = findings["recyclability"]
            print(
                f"  Most Recyclable Material: {rec_findings.get('most_recyclable_material', 'N/A')}"
            )
            print(
                f"  Best Recycling Method: {rec_findings.get('best_recycling_method', 'N/A')}"
            )

        print("\nRecommendations:")
        recommendations = report.get("recommendations", [])
        for i, rec in enumerate(recommendations[:5], 1):  # Show first 5
            print(f"  {i}. {rec}")

    if "visualization_path" in results:
        print(f"\nVisualization: {results['visualization_path']}")

    print("\nTechnology Stack:")
    print(f"  PyTorch: {'Available' if TORCH_AVAILABLE else 'Not Available'}")
    print(f"  Plotly: {'Available' if PLOTLY_AVAILABLE else 'Not Available'}")
    print(f"  GPU Acceleration: {'Enabled' if demo.use_gpu else 'Disabled'}")

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
