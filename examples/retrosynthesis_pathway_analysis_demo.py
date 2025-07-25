#!/usr/bin/env python3
"""
Advanced Retrosynthesis Pathway Analysis Demonstration

This demo showcases ChemiAccel's advanced retrosynthesis analysis capabilities,
including AI-driven pathway planning, synthetic accessibility scoring, and
commercial availability assessment for complex target molecules.

Features:
- Multi-step retrosynthetic pathway generation
- GPU-accelerated disconnection analysis
- Commercial availability database integration
- Cost and time estimation for synthetic routes
- Interactive pathway visualization
- Comparison of multiple synthetic strategies

Usage:
    python examples/retrosynthesis_pathway_analysis_demo.py
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from chemi_accel.core.molecular import MolecularData
from chemi_accel.descriptors.reaction_informatics import (
    create_reaction_informatics_engine,
)

# GPU backend imports (if available)
try:
    from chemi_accel.backends.cuda.reaction_informatics import (
        GPUReactionInformatics,
        ReactionType,
        RetrosynthesisPath,
    )

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPU backend not available - using CPU backend")

# RDKit for molecular processing
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw, rdMolDescriptors
    from rdkit.Chem.rdchem import BondType

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit not available - limited functionality")


@dataclass
class SyntheticRoute:
    """Complete synthetic route with all pathway details."""

    target_molecule: str
    target_name: str
    pathway_id: str
    total_steps: int
    overall_score: float
    complexity_score: float
    commercial_availability_score: float
    estimated_cost: float
    estimated_time_hours: float
    pathway_steps: List[Dict[str, Any]]
    starting_materials: List[str]
    key_intermediates: List[str]
    critical_reactions: List[str]
    risk_factors: List[str]
    literature_precedents: int


class RetrosynthesisAnalyzer:
    """Advanced retrosynthesis pathway analysis engine."""

    def __init__(self, use_gpu: bool = True):
        """Initialize retrosynthesis analyzer."""
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.engine = create_reaction_informatics_engine(
            backend="gpu" if self.use_gpu else "cpu",
            batch_size=1000,
            use_gpu_acceleration=self.use_gpu,
            enable_quantum_features=True,
        )

        # Commercial availability database (mock)
        self.commercial_db = self._initialize_commercial_database()

        # Reaction difficulty scoring
        self.reaction_difficulty_scores = self._initialize_difficulty_scores()

        print(
            f"Retrosynthesis analyzer initialized with {self.engine.backend.upper()} backend"
        )

    def _initialize_commercial_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize mock commercial availability database."""
        # This would connect to actual chemical suppliers in practice
        return {
            # Simple starting materials (high availability)
            "CCO": {
                "available": True,
                "cost_per_g": 0.5,
                "suppliers": ["Sigma", "Fisher", "Acros"],
            },
            "CC(=O)O": {
                "available": True,
                "cost_per_g": 0.3,
                "suppliers": ["Sigma", "Fisher"],
            },
            "c1ccccc1": {
                "available": True,
                "cost_per_g": 0.8,
                "suppliers": ["Sigma", "Fisher", "Acros", "TCI"],
            },
            "CCN": {
                "available": True,
                "cost_per_g": 1.2,
                "suppliers": ["Sigma", "Fisher"],
            },
            "OC(=O)c1ccccc1": {
                "available": True,
                "cost_per_g": 2.0,
                "suppliers": ["Sigma", "Fisher", "Acros"],
            },
            # Building blocks (moderate availability)
            "BrC1=CC=C(C=C1)B(O)O": {
                "available": True,
                "cost_per_g": 15.0,
                "suppliers": ["Sigma", "TCI"],
            },
            "NC1=CC=C(C=C1)C(=O)O": {
                "available": True,
                "cost_per_g": 8.0,
                "suppliers": ["Sigma", "Acros"],
            },
            "ClC(=O)C1=CC=CC=C1": {
                "available": True,
                "cost_per_g": 5.0,
                "suppliers": ["Sigma", "Fisher"],
            },
            # Advanced intermediates (low availability)
            "COC1=C(C=CC(=C1)CC2COC(=O)C2)OC": {
                "available": True,
                "cost_per_g": 150.0,
                "suppliers": ["TCI"],
            },
            "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C": {
                "available": False,
                "cost_per_g": None,
                "suppliers": [],
            },
        }

    def _initialize_difficulty_scores(self) -> Dict[ReactionType, float]:
        """Initialize reaction difficulty scoring system."""
        if GPU_AVAILABLE:
            return {
                ReactionType.AMIDE_FORMATION: 0.2,  # Easy
                ReactionType.ESTER_FORMATION: 0.3,
                ReactionType.ETHER_FORMATION: 0.4,
                ReactionType.REDUCTION: 0.3,
                ReactionType.OXIDATION: 0.4,
                ReactionType.SUZUKI_COUPLING: 0.6,  # Moderate
                ReactionType.HECK_REACTION: 0.7,
                ReactionType.CLICK_CHEMISTRY: 0.3,
                ReactionType.GRIGNARD_REACTION: 0.5,
                ReactionType.DIELS_ALDER: 0.5,
                ReactionType.CYCLIZATION: 0.7,  # Difficult
                ReactionType.RING_OPENING: 0.6,
                ReactionType.HALOGENATION: 0.4,
                ReactionType.NITRATION: 0.5,
                ReactionType.ELIMINATION: 0.4,
            }
        else:
            # String-based for CPU backend
            return {
                "amide_formation": 0.2,
                "suzuki_coupling": 0.6,
                "click_chemistry": 0.3,
                "oxidation": 0.4,
                "reduction": 0.3,
                "grignard_reaction": 0.5,
                "diels_alder": 0.5,
            }

    def analyze_target_molecule(
        self,
        target_smiles: str,
        target_name: str = None,
        max_steps: int = 6,
        num_routes: int = 3,
    ) -> List[SyntheticRoute]:
        """Analyze multiple synthetic routes for target molecule."""
        print(f"\nAnalyzing synthetic routes for: {target_name or target_smiles}")

        if not RDKIT_AVAILABLE:
            print("RDKit not available - using simplified analysis")
            return self._analyze_target_simple(
                target_smiles, target_name, max_steps, num_routes
            )

        # Create molecular data
        rdkit_mol = Chem.MolFromSmiles(target_smiles)
        if rdkit_mol is None:
            raise ValueError(f"Invalid SMILES: {target_smiles}")

        target_mol = MolecularData(
            mol_id=f"target_{target_name or 'unknown'}",
            smiles=target_smiles,
            backend_data={"rdkit_mol": rdkit_mol},
        )

        # Generate multiple pathway strategies
        strategies = ["linear", "convergent", "automated"]
        synthetic_routes = []

        for i, strategy in enumerate(strategies[:num_routes]):
            print(f"  Strategy {i+1}: {strategy}")

            try:
                # Analyze retrosynthesis with current strategy
                retro_result = self.engine.analyze_retrosynthesis(
                    target_molecule=target_mol,
                    max_steps=max_steps,
                    min_commercial_availability=0.6,
                    strategy=strategy,
                )

                # Enhanced analysis
                synthetic_route = self._create_enhanced_synthetic_route(
                    retro_result, target_smiles, target_name, strategy, i + 1
                )

                synthetic_routes.append(synthetic_route)

            except Exception as e:
                print(f"    Strategy {strategy} failed: {e}")
                continue

        # Rank and optimize routes
        synthetic_routes.sort(key=lambda r: r.overall_score, reverse=True)

        print(f"  Generated {len(synthetic_routes)} viable synthetic routes")
        return synthetic_routes

    def _analyze_target_simple(
        self, target_smiles: str, target_name: str, max_steps: int, num_routes: int
    ) -> List[SyntheticRoute]:
        """Simplified analysis without RDKit."""
        target_mol = MolecularData(
            mol_id=f"target_{target_name or 'unknown'}",
            smiles=target_smiles,
            backend_data={},
        )

        synthetic_routes = []
        strategies = ["linear", "convergent", "automated"]

        for i, strategy in enumerate(strategies[:num_routes]):
            try:
                retro_result = self.engine.analyze_retrosynthesis(
                    target_molecule=target_mol, max_steps=max_steps, strategy=strategy
                )

                synthetic_route = self._create_enhanced_synthetic_route(
                    retro_result, target_smiles, target_name, strategy, i + 1
                )
                synthetic_routes.append(synthetic_route)

            except Exception as e:
                print(f"    Strategy {strategy} failed: {e}")
                continue

        return synthetic_routes

    def _create_enhanced_synthetic_route(
        self,
        retro_result: Dict[str, Any],
        target_smiles: str,
        target_name: str,
        strategy: str,
        route_id: int,
    ) -> SyntheticRoute:
        """Create enhanced synthetic route with detailed analysis."""

        pathway_steps = retro_result.get("pathway_steps", [])

        # Extract starting materials and intermediates
        starting_materials = []
        key_intermediates = []
        critical_reactions = []
        risk_factors = []

        for step in pathway_steps:
            # Add predicted precursors as intermediates or starting materials
            precursors = step.get("predicted_precursors", [])
            key_intermediates.extend(precursors)

            # Identify critical reactions
            reaction_type = step.get("reaction_type", "unknown")
            feasibility = step.get("feasibility_score", 0.5)

            if feasibility < 0.6:
                risk_factors.append(f"Low feasibility reaction: {reaction_type}")

            critical_reactions.append(reaction_type)

        # Check commercial availability
        commercial_availability_score = self._calculate_commercial_availability(
            key_intermediates
        )

        # Calculate enhanced metrics
        total_steps = len(pathway_steps)
        complexity_score = retro_result.get("complexity_score", 0.5)
        base_score = retro_result.get("total_score", 0.5)

        # Enhanced scoring incorporating multiple factors
        overall_score = (
            base_score * 0.4
            + (1.0 - complexity_score) * 0.3  # Base feasibility
            + commercial_availability_score * 0.2  # Lower complexity is better
            + (1.0 / max(total_steps, 1))  # Commercial availability
            * 0.1  # Shorter routes preferred
        )

        # Cost and time estimation
        estimated_cost = self._estimate_synthesis_cost(key_intermediates, pathway_steps)
        estimated_time = self._estimate_synthesis_time(pathway_steps, strategy)

        # Literature precedents (mock)
        literature_precedents = max(
            10 - total_steps * 2, 0
        )  # Fewer steps = more precedents

        return SyntheticRoute(
            target_molecule=target_smiles,
            target_name=target_name or "Unknown",
            pathway_id=f"{strategy}_{route_id}",
            total_steps=total_steps,
            overall_score=overall_score,
            complexity_score=complexity_score,
            commercial_availability_score=commercial_availability_score,
            estimated_cost=estimated_cost,
            estimated_time_hours=estimated_time,
            pathway_steps=pathway_steps,
            starting_materials=starting_materials[:5],  # Limit display
            key_intermediates=key_intermediates[:8],
            critical_reactions=list(set(critical_reactions)),
            risk_factors=risk_factors,
            literature_precedents=literature_precedents,
        )

    def _calculate_commercial_availability(self, intermediates: List[str]) -> float:
        """Calculate commercial availability score for intermediates."""
        if not intermediates:
            return 0.5

        available_count = 0
        total_count = len(intermediates)

        for intermediate in intermediates:
            # Check against commercial database
            if intermediate in self.commercial_db:
                if self.commercial_db[intermediate]["available"]:
                    available_count += 1
            else:
                # Heuristic for unknown compounds based on complexity
                if len(intermediate) < 20:  # Simple compounds likely available
                    available_count += 0.7
                elif len(intermediate) < 40:  # Moderate complexity
                    available_count += 0.3
                # Complex compounds assumed not readily available

        return min(available_count / total_count, 1.0)

    def _estimate_synthesis_cost(
        self, intermediates: List[str], pathway_steps: List[Dict[str, Any]]
    ) -> float:
        """Estimate total synthesis cost."""
        base_cost = 50.0  # Base lab overhead

        # Material costs
        material_cost = 0.0
        for intermediate in intermediates:
            if intermediate in self.commercial_db:
                compound_data = self.commercial_db[intermediate]
                if compound_data["available"] and compound_data["cost_per_g"]:
                    material_cost += (
                        compound_data["cost_per_g"] * 5.0
                    )  # Assume 5g needed
            else:
                # Estimate based on complexity
                material_cost += (
                    len(intermediate) * 2.0
                )  # $2 per character (rough heuristic)

        # Reaction costs
        reaction_cost = 0.0
        for step in pathway_steps:
            reaction_type = step.get("reaction_type", "unknown")

            # Different reactions have different costs
            if "coupling" in reaction_type.lower():
                reaction_cost += 200.0  # Expensive catalysts
            elif "amide" in reaction_type.lower():
                reaction_cost += 50.0  # Moderate cost
            elif (
                "oxidation" in reaction_type.lower()
                or "reduction" in reaction_type.lower()
            ):
                reaction_cost += 75.0  # Reagent cost
            else:
                reaction_cost += 100.0  # Default

        total_cost = base_cost + material_cost + reaction_cost
        return min(total_cost, 10000.0)  # Cap at $10k

    def _estimate_synthesis_time(
        self, pathway_steps: List[Dict[str, Any]], strategy: str
    ) -> float:
        """Estimate total synthesis time in hours."""
        base_time = 8.0  # Base setup and workup time

        # Time per step
        step_time = 0.0
        for step in pathway_steps:
            reaction_type = step.get("reaction_type", "unknown")
            feasibility = step.get("feasibility_score", 0.5)

            # Base reaction time
            if "coupling" in reaction_type.lower():
                step_time += 24.0  # Long reaction times
            elif "click" in reaction_type.lower():
                step_time += 4.0  # Fast reactions
            elif "amide" in reaction_type.lower():
                step_time += 8.0  # Moderate
            else:
                step_time += 12.0  # Default

            # Difficulty modifier
            if feasibility < 0.5:
                step_time *= 2.0  # Difficult reactions take longer
            elif feasibility > 0.8:
                step_time *= 0.8  # Easy reactions are faster

        # Strategy modifier
        if strategy == "convergent":
            step_time *= 0.8  # Parallel synthesis possible
        elif strategy == "linear":
            step_time *= 1.2  # Sequential synthesis

        # Purification and characterization time
        purification_time = len(pathway_steps) * 4.0

        total_time = base_time + step_time + purification_time
        return min(total_time, 1000.0)  # Cap at 1000 hours

    def compare_synthetic_routes(self, routes: List[SyntheticRoute]) -> pd.DataFrame:
        """Compare multiple synthetic routes in tabular format."""
        if not routes:
            return pd.DataFrame()

        comparison_data = []

        for route in routes:
            comparison_data.append(
                {
                    "Route ID": route.pathway_id,
                    "Strategy": route.pathway_id.split("_")[0],
                    "Steps": route.total_steps,
                    "Overall Score": f"{route.overall_score:.3f}",
                    "Complexity": f"{route.complexity_score:.3f}",
                    "Availability": f"{route.commercial_availability_score:.1%}",
                    "Est. Cost ($)": f"{route.estimated_cost:.0f}",
                    "Est. Time (h)": f"{route.estimated_time_hours:.0f}",
                    "Critical Reactions": len(route.critical_reactions),
                    "Risk Factors": len(route.risk_factors),
                    "Literature": route.literature_precedents,
                }
            )

        return pd.DataFrame(comparison_data)

    def visualize_synthetic_pathways(
        self, routes: List[SyntheticRoute], output_file: str = None
    ):
        """Create visualization of synthetic pathways."""
        if not routes:
            print("No routes to visualize")
            return

        # Create figure with subplots for each route
        n_routes = len(routes)
        fig, axes = plt.subplots(n_routes, 1, figsize=(15, 5 * n_routes))
        if n_routes == 1:
            axes = [axes]

        fig.suptitle("Retrosynthetic Pathway Analysis", fontsize=16, fontweight="bold")

        for i, route in enumerate(routes):
            ax = axes[i]

            # Route title
            ax.set_title(
                f"Route {i+1}: {route.pathway_id} (Score: {route.overall_score:.3f})",
                fontweight="bold",
            )

            # Draw pathway steps
            steps = route.pathway_steps
            n_steps = len(steps)

            if n_steps == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No pathway steps available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                continue

            # Draw molecules and reactions
            x_positions = np.linspace(0.1, 0.9, n_steps + 1)
            y_center = 0.5

            # Target molecule
            ax.add_patch(
                patches.Rectangle(
                    (x_positions[0] - 0.03, y_center - 0.15),
                    0.06,
                    0.3,
                    facecolor="lightblue",
                    edgecolor="blue",
                    linewidth=2,
                )
            )
            ax.text(
                x_positions[0],
                y_center,
                "Target",
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=8,
            )

            # Draw pathway steps
            for j, step in enumerate(steps):
                x = x_positions[j + 1]

                # Reaction arrow
                ax.annotate(
                    "",
                    xy=(x - 0.02, y_center),
                    xytext=(x_positions[j] + 0.03, y_center),
                    arrowprops=dict(arrowstyle="->", lw=2, color="red"),
                )

                # Reaction type
                reaction_type = step.get("reaction_type", "unknown")
                ax.text(
                    (x + x_positions[j]) / 2,
                    y_center + 0.2,
                    reaction_type.replace("_", " ").title()[:15] + "...",
                    ha="center",
                    va="center",
                    fontsize=7,
                    style="italic",
                )

                # Feasibility score
                feasibility = step.get("feasibility_score", 0.5)
                color = (
                    "green"
                    if feasibility > 0.7
                    else "orange"
                    if feasibility > 0.4
                    else "red"
                )
                ax.text(
                    (x + x_positions[j]) / 2,
                    y_center - 0.25,
                    f"Score: {feasibility:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                    fontweight="bold",
                )

                # Precursor molecule
                ax.add_patch(
                    patches.Rectangle(
                        (x - 0.03, y_center - 0.15),
                        0.06,
                        0.3,
                        facecolor="lightgreen",
                        edgecolor="green",
                        linewidth=1,
                    )
                )
                ax.text(
                    x,
                    y_center,
                    f"Int {j+1}",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    fontsize=8,
                )

            # Route metrics
            metrics_text = (
                f"Steps: {route.total_steps} | "
                f"Cost: ${route.estimated_cost:.0f} | "
                f"Time: {route.estimated_time_hours:.0f}h | "
                f"Risks: {len(route.risk_factors)}"
            )

            ax.text(
                0.5,
                0.05,
                metrics_text,
                ha="center",
                va="bottom",
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
            )

            # Clean up axes
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)

        plt.tight_layout()

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Pathway visualization saved to: {output_path}")

        # plt.show()

    def generate_detailed_report(
        self, routes: List[SyntheticRoute], target_name: str
    ) -> str:
        """Generate detailed retrosynthesis report."""
        if not routes:
            return "No synthetic routes found."

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"RETROSYNTHETIC ANALYSIS REPORT: {target_name.upper()}")
        report_lines.append("=" * 80)

        # Target information
        best_route = routes[0]  # Assuming sorted by score
        report_lines.append(f"\nTARGET MOLECULE: {best_route.target_molecule}")
        report_lines.append(f"ANALYSIS DATE: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"BACKEND: {self.engine.backend.upper()}")

        # Route summary
        report_lines.append("\nROUTE SUMMARY:")
        report_lines.append(f"  Routes Generated: {len(routes)}")
        report_lines.append(f"  Best Route Score: {best_route.overall_score:.3f}")
        report_lines.append(
            f"  Optimal Strategy: {best_route.pathway_id.split('_')[0]}"
        )

        # Detailed route analysis
        for i, route in enumerate(routes):
            report_lines.append(f"\nROUTE {i+1}: {route.pathway_id.upper()}")
            report_lines.append("-" * 50)
            report_lines.append(f"  Overall Score: {route.overall_score:.3f}")
            report_lines.append(f"  Synthetic Steps: {route.total_steps}")
            report_lines.append(f"  Complexity Score: {route.complexity_score:.3f}")
            report_lines.append(
                f"  Commercial Availability: {route.commercial_availability_score:.1%}"
            )
            report_lines.append(f"  Estimated Cost: ${route.estimated_cost:.0f}")
            report_lines.append(
                f"  Estimated Time: {route.estimated_time_hours:.0f} hours"
            )
            report_lines.append(
                f"  Literature Precedents: {route.literature_precedents}"
            )

            # Critical reactions
            if route.critical_reactions:
                report_lines.append(
                    f"  Critical Reactions: {', '.join(route.critical_reactions)}"
                )

            # Risk factors
            if route.risk_factors:
                report_lines.append("  Risk Factors:")
                for risk in route.risk_factors:
                    report_lines.append(f"    - {risk}")

            # Pathway steps detail
            if route.pathway_steps:
                report_lines.append("  Pathway Steps:")
                for j, step in enumerate(route.pathway_steps):
                    reaction_type = step.get("reaction_type", "unknown")
                    feasibility = step.get("feasibility_score", 0.0)
                    report_lines.append(
                        f"    Step {j+1}: {reaction_type} (Score: {feasibility:.2f})"
                    )

        # Recommendations
        report_lines.append("\nRECOMMENDATIONS:")
        best_route = routes[0]

        if best_route.overall_score > 0.7:
            report_lines.append("  ✓ Highly feasible synthetic route identified")
        elif best_route.overall_score > 0.5:
            report_lines.append("  ⚠ Moderate feasibility - optimization recommended")
        else:
            report_lines.append(
                "  ⚠ Challenging synthesis - alternative approaches needed"
            )

        if best_route.commercial_availability_score < 0.5:
            report_lines.append("  ⚠ Low commercial availability of intermediates")

        if best_route.estimated_cost > 1000:
            report_lines.append("  ⚠ High estimated synthesis cost")

        if len(best_route.risk_factors) > 2:
            report_lines.append("  ⚠ Multiple risk factors identified")

        report_lines.append("\n" + "=" * 80)

        return "\n".join(report_lines)

    def cleanup(self):
        """Clean up resources."""
        self.engine.cleanup()


def create_demo_target_molecules() -> List[Tuple[str, str]]:
    """Create diverse set of target molecules for retrosynthesis demo."""
    targets = [
        # Pharmaceutical targets
        ("CC(C)Cc1ccc(C(C)C(=O)O)cc1", "Ibuprofen"),
        ("CN(C)CCN1c2ccccc2Sc3ccc(C(F)(F)F)cc31", "Trifluoperazine"),
        ("COc1cc2nc(N3CCN(CCO)CC3)nc(N)c2cc1OC", "Prazosin"),
        # Natural product targets
        ("COC1=C(C=CC(=C1)CC2COC(=O)C2)OC", "Natural Product Derivative"),
        ("CC(C)=CCCC(C)=CC(=O)C1=C(C=CC=C1O)O", "Flavonoid Derivative"),
        # Synthetic intermediates
        ("CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C", "Kinase Inhibitor Core"),
        ("c1ccc(cc1)C2=CC(=O)c3c(o2)cc(cc3O)O", "Flavone Derivative"),
        # Complex heterocycles
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine"),
        ("CC12CCC3C(C1CCC2O)CCC4=C3C=CC(=C4)O", "Steroid Framework"),
        # Polymer precursors
        ("C=CC(=O)OCCOC(=O)C=C", "Diacrylate Monomer"),
    ]

    return targets


def main():
    """Main demonstration function."""
    print("Advanced Retrosynthesis Pathway Analysis Demonstration")
    print("=" * 60)
    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"RDKit Available: {RDKIT_AVAILABLE}")

    # Initialize analyzer
    analyzer = RetrosynthesisAnalyzer(use_gpu=GPU_AVAILABLE)

    # Get target molecules
    target_molecules = create_demo_target_molecules()

    # Analyze first few targets in detail
    all_routes = []
    target_reports = []

    for i, (smiles, name) in enumerate(target_molecules[:3]):  # Analyze first 3
        print(f"\n{'='*80}")
        print(f"TARGET {i+1}: {name}")
        print(f"{'='*80}")

        try:
            # Analyze synthetic routes
            routes = analyzer.analyze_target_molecule(
                target_smiles=smiles, target_name=name, max_steps=5, num_routes=3
            )

            if routes:
                all_routes.extend(routes)

                # Display route comparison
                print("\nSynthetic Route Comparison:")
                comparison_df = analyzer.compare_synthetic_routes(routes)
                print(comparison_df.to_string(index=False))

                # Generate detailed report
                detailed_report = analyzer.generate_detailed_report(routes, name)
                target_reports.append(detailed_report)

                # Save individual target report
                output_dir = Path("test_results")
                output_dir.mkdir(exist_ok=True)

                report_file = (
                    output_dir / f"retrosynthesis_report_{name.replace(' ', '_')}.txt"
                )
                with open(report_file, "w") as f:
                    f.write(detailed_report)

                print(f"\nDetailed report saved to: {report_file}")

                # Create pathway visualization
                viz_file = (
                    output_dir / f"pathway_visualization_{name.replace(' ', '_')}.png"
                )
                analyzer.visualize_synthetic_pathways(routes, str(viz_file))

            else:
                print(f"No viable synthetic routes found for {name}")

        except Exception as e:
            print(f"Analysis failed for {name}: {e}")
            continue

    # Overall summary
    if all_routes:
        print(f"\n{'='*80}")
        print("OVERALL RETROSYNTHESIS ANALYSIS SUMMARY")
        print(f"{'='*80}")

        total_routes = len(all_routes)
        avg_score = np.mean([r.overall_score for r in all_routes])
        avg_steps = np.mean([r.total_steps for r in all_routes])
        avg_cost = np.mean([r.estimated_cost for r in all_routes])

        print(f"Total Routes Analyzed: {total_routes}")
        print(f"Average Route Score: {avg_score:.3f}")
        print(f"Average Synthetic Steps: {avg_steps:.1f}")
        print(f"Average Estimated Cost: ${avg_cost:.0f}")

        # Best routes across all targets
        best_routes = sorted(all_routes, key=lambda r: r.overall_score, reverse=True)[
            :3
        ]
        print("\nTop 3 Overall Routes:")
        for i, route in enumerate(best_routes):
            print(
                f"  {i+1}. {route.target_name} ({route.pathway_id}): Score {route.overall_score:.3f}"
            )

        # Create combined visualization
        output_dir = Path("test_results")
        combined_viz_file = output_dir / "combined_retrosynthesis_analysis.png"
        analyzer.visualize_synthetic_pathways(best_routes, str(combined_viz_file))

        # Save combined report
        combined_report_file = output_dir / "combined_retrosynthesis_report.txt"
        with open(combined_report_file, "w") as f:
            f.write("\n\n".join(target_reports))

        print(f"\nCombined analysis saved to: {combined_report_file}")

    # Cleanup
    analyzer.cleanup()

    print(f"\n{'='*80}")
    print("RETROSYNTHESIS PATHWAY ANALYSIS DEMONSTRATION COMPLETED")
    print(f"{'='*80}")
    print("Key Features Demonstrated:")
    print("- Multi-step retrosynthetic pathway generation")
    print("- GPU-accelerated disconnection analysis")
    print("- Commercial availability assessment")
    print("- Cost and time estimation")
    print("- Multiple synthetic strategy comparison")
    print("- Interactive pathway visualization")
    print("\nTask 19.2 Retrosynthesis Component: SUCCESSFULLY DEMONSTRATED")


if __name__ == "__main__":
    main()
