"""
Reaction Intelligence Module

Advanced AI-driven analysis and optimization of chemical reactions including:
- Reaction condition and catalyst suggestion
- What-if scenario simulation
- Experimental feedback integration
- Knowledge graph integration for catalyst performance
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from ..core.exceptions import ChemiAccelError
from ..knowledge_graph.query_engine import QueryEngine
from .mechanism_prediction import MechanismPredictor, ReactionPathwayGenerator

logger = logging.getLogger(__name__)


@dataclass
class ReactionConditions:
    """Comprehensive reaction condition specification."""

    temperature: Optional[float] = None  # Kelvin
    pressure: Optional[float] = None  # Bar
    solvent: Optional[str] = None
    catalyst: Optional[str] = None
    catalyst_loading: Optional[float] = None  # mol%
    ph: Optional[float] = None
    reaction_time: Optional[float] = None  # Hours
    atmosphere: Optional[str] = None  # 'air', 'N2', 'Ar', etc.
    concentration: Optional[float] = None  # M
    additives: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "temperature": self.temperature,
            "pressure": self.pressure,
            "solvent": self.solvent,
            "catalyst": self.catalyst,
            "catalyst_loading": self.catalyst_loading,
            "ph": self.ph,
            "reaction_time": self.reaction_time,
            "atmosphere": self.atmosphere,
            "concentration": self.concentration,
            "additives": self.additives,
        }


@dataclass
class CatalystSuggestion:
    """Catalyst recommendation with performance metrics."""

    catalyst_name: str
    catalyst_smiles: Optional[str] = None
    predicted_yield: Optional[float] = None
    predicted_selectivity: Optional[float] = None
    activation_energy: Optional[float] = None  # kJ/mol
    confidence_score: float = 0.0
    literature_precedents: int = 0
    recommended_loading: Optional[float] = None  # mol%
    compatible_solvents: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ReactionScenario:
    """Complete reaction scenario specification."""

    reactants: List[str]  # SMILES
    products: List[str]  # SMILES
    conditions: ReactionConditions
    predicted_yield: Optional[float] = None
    predicted_selectivity: Optional[float] = None
    reaction_rate: Optional[float] = None
    side_products: List[str] = field(default_factory=list)
    energy_profile: Optional[Dict[str, float]] = None


@dataclass
class ExperimentalResult:
    """Experimental reaction outcome data."""

    scenario: ReactionScenario
    actual_yield: float
    actual_selectivity: Optional[float] = None
    actual_products: List[str] = field(default_factory=list)
    actual_rate: Optional[float] = None
    notes: str = ""
    experiment_date: Optional[str] = None
    researcher: Optional[str] = ""


class ReactionIntelligenceEngine:
    """Core engine for reaction intelligence and optimization."""

    def __init__(
        self,
        kg_engine: Optional[QueryEngine] = None,
        mechanism_predictor: Optional[MechanismPredictor] = None,
        pathway_generator: Optional[ReactionPathwayGenerator] = None,
    ):
        self.kg_engine = kg_engine
        self.mechanism_predictor = mechanism_predictor or MechanismPredictor()
        self.pathway_generator = pathway_generator or ReactionPathwayGenerator()
        self.experimental_data: List[ExperimentalResult] = []

    def analyze_reaction_requirements(
        self, reactants: List[str], products: List[str]
    ) -> Dict[str, Any]:
        """Analyze reaction requirements and constraints."""
        try:
            # Convert SMILES to molecules
            reactant_mols = [Chem.MolFromSmiles(smi) for smi in reactants]
            product_mols = [Chem.MolFromSmiles(smi) for smi in products]

            if any(mol is None for mol in reactant_mols + product_mols):
                raise ChemiAccelError("Invalid SMILES in reactants or products")

            # Analyze bond changes
            bond_changes = self._analyze_bond_changes(reactant_mols, product_mols)

            # Determine reaction type
            reaction_type = self._classify_reaction_type(bond_changes)

            # Estimate thermodynamic parameters
            thermo_params = self._estimate_thermodynamics(reactant_mols, product_mols)

            return {
                "bond_changes": bond_changes,
                "reaction_type": reaction_type,
                "thermodynamics": thermo_params,
                "complexity_score": len(bond_changes),
            }

        except Exception as e:
            logger.error(f"Error analyzing reaction requirements: {e}")
            raise ChemiAccelError(f"Failed to analyze reaction: {e}")

    def _analyze_bond_changes(
        self, reactants: List[Chem.Mol], products: List[Chem.Mol]
    ) -> List[Dict[str, Any]]:
        """Analyze bond formation and breaking patterns."""
        # Simplified bond change analysis
        changes = []

        for r_mol in reactants:
            for p_mol in products:
                # Compare bond patterns (simplified)
                r_bonds = set()
                p_bonds = set()

                for bond in r_mol.GetBonds():
                    r_bonds.add(
                        (
                            bond.GetBeginAtom().GetAtomicNum(),
                            bond.GetEndAtom().GetAtomicNum(),
                            bond.GetBondType(),
                        )
                    )

                for bond in p_mol.GetBonds():
                    p_bonds.add(
                        (
                            bond.GetBeginAtom().GetAtomicNum(),
                            bond.GetEndAtom().GetAtomicNum(),
                            bond.GetBondType(),
                        )
                    )

                formed = p_bonds - r_bonds
                broken = r_bonds - p_bonds

                if formed or broken:
                    changes.append(
                        {"bonds_formed": list(formed), "bonds_broken": list(broken)}
                    )

        return changes

    def _classify_reaction_type(self, bond_changes: List[Dict[str, Any]]) -> str:
        """Classify reaction based on bond changes."""
        if not bond_changes:
            return "unknown"

        total_formed = sum(len(change["bonds_formed"]) for change in bond_changes)
        total_broken = sum(len(change["bonds_broken"]) for change in bond_changes)

        if total_formed > total_broken:
            return "coupling"
        elif total_broken > total_formed:
            return "cleavage"
        else:
            return "rearrangement"

    def _estimate_thermodynamics(
        self, reactants: List[Chem.Mol], products: List[Chem.Mol]
    ) -> Dict[str, float]:
        """Estimate basic thermodynamic parameters."""
        # Simplified estimation based on molecular weight and complexity
        reactant_complexity = sum(mol.GetNumBonds() for mol in reactants)
        product_complexity = sum(mol.GetNumBonds() for mol in products)

        # Very rough estimates
        delta_h = (product_complexity - reactant_complexity) * 5.0  # kJ/mol per bond
        delta_s = np.random.normal(0, 50)  # J/mol·K (placeholder)

        return {
            "delta_h_estimated": delta_h,
            "delta_s_estimated": delta_s,
            "delta_g_298k": delta_h - 298.15 * delta_s / 1000,
        }


class CatalystSuggestionEngine:
    """Engine for suggesting optimal catalysts and conditions."""

    def __init__(self, intelligence_engine: ReactionIntelligenceEngine):
        self.intelligence_engine = intelligence_engine
        self.catalyst_database = self._initialize_catalyst_database()

    def suggest_catalysts(
        self, reactants: List[str], products: List[str], max_suggestions: int = 5
    ) -> List[CatalystSuggestion]:
        """Suggest optimal catalysts for the given reaction."""
        try:
            # Analyze reaction requirements
            analysis = self.intelligence_engine.analyze_reaction_requirements(
                reactants, products
            )

            reaction_type = analysis["reaction_type"]
            complexity = analysis["complexity_score"]

            # Filter catalysts by reaction type
            relevant_catalysts = self._filter_catalysts_by_type(reaction_type)

            # Score and rank catalysts
            scored_catalysts = []
            for catalyst_info in relevant_catalysts[: max_suggestions * 2]:
                score = self._score_catalyst(catalyst_info, analysis)

                suggestion = CatalystSuggestion(
                    catalyst_name=catalyst_info["name"],
                    catalyst_smiles=catalyst_info.get("smiles"),
                    confidence_score=score,
                    literature_precedents=catalyst_info.get("precedents", 0),
                    recommended_loading=catalyst_info.get("typical_loading"),
                    compatible_solvents=catalyst_info.get("solvents", []),
                    notes=catalyst_info.get("notes", ""),
                )

                scored_catalysts.append(suggestion)

            # Sort by confidence score and return top suggestions
            scored_catalysts.sort(key=lambda x: x.confidence_score, reverse=True)
            return scored_catalysts[:max_suggestions]

        except Exception as e:
            logger.error(f"Error suggesting catalysts: {e}")
            raise ChemiAccelError(f"Failed to suggest catalysts: {e}")

    def suggest_conditions(
        self, reactants: List[str], products: List[str], catalyst: Optional[str] = None
    ) -> ReactionConditions:
        """Suggest optimal reaction conditions."""
        try:
            analysis = self.intelligence_engine.analyze_reaction_requirements(
                reactants, products
            )

            reaction_type = analysis["reaction_type"]
            delta_g = analysis["thermodynamics"]["delta_g_298k"]

            # Base conditions on reaction type and thermodynamics
            conditions = ReactionConditions()

            if reaction_type == "coupling":
                conditions.temperature = 353.15  # 80°C
                conditions.solvent = "DMF"
                conditions.atmosphere = "N2"
            elif reaction_type == "cleavage":
                conditions.temperature = 373.15  # 100°C
                conditions.solvent = "toluene"
                conditions.atmosphere = "air"
            else:  # rearrangement
                conditions.temperature = 298.15  # RT
                conditions.solvent = "DCM"
                conditions.atmosphere = "air"

            # Adjust based on thermodynamics
            if delta_g > 0:  # Endergonic
                conditions.temperature += 50  # Higher temperature

            # Set other defaults
            conditions.pressure = 1.0  # 1 bar
            conditions.reaction_time = 12.0  # 12 hours
            conditions.concentration = 0.1  # 0.1 M

            if catalyst:
                conditions.catalyst = catalyst
                conditions.catalyst_loading = 5.0  # 5 mol%

            return conditions

        except Exception as e:
            logger.error(f"Error suggesting conditions: {e}")
            raise ChemiAccelError(f"Failed to suggest conditions: {e}")

    def _initialize_catalyst_database(self) -> List[Dict[str, Any]]:
        """Initialize catalyst database with common catalysts."""
        return [
            {
                "name": "Pd(PPh3)4",
                "smiles": "[Pd]",
                "types": ["coupling", "cross_coupling"],
                "precedents": 1000,
                "typical_loading": 5.0,
                "solvents": ["DMF", "THF", "toluene"],
                "notes": "Versatile palladium catalyst for cross-coupling",
            },
            {
                "name": "CuI",
                "smiles": "[Cu+].[I-]",
                "types": ["coupling", "click"],
                "precedents": 500,
                "typical_loading": 10.0,
                "solvents": ["DMF", "DMSO", "acetonitrile"],
                "notes": "Copper catalyst for click chemistry and coupling",
            },
            {
                "name": "Ru(bpy)3Cl2",
                "smiles": "[Ru+2]",
                "types": ["photocatalysis", "oxidation"],
                "precedents": 200,
                "typical_loading": 1.0,
                "solvents": ["acetonitrile", "DMF"],
                "notes": "Photocatalyst for visible light reactions",
            },
            {
                "name": "TFA",
                "smiles": "C(=O)(C(F)(F)F)O",
                "types": ["rearrangement", "cyclization"],
                "precedents": 800,
                "typical_loading": 100.0,
                "solvents": ["DCM", "neat"],
                "notes": "Strong acid catalyst",
            },
            {
                "name": "DBU",
                "smiles": "C1CCN2CCCN=C2C1",
                "types": ["elimination", "deprotection"],
                "precedents": 300,
                "typical_loading": 50.0,
                "solvents": ["THF", "DMF", "acetonitrile"],
                "notes": "Strong non-nucleophilic base",
            },
        ]

    def _filter_catalysts_by_type(self, reaction_type: str) -> List[Dict[str, Any]]:
        """Filter catalyst database by reaction type."""
        return [
            cat
            for cat in self.catalyst_database
            if reaction_type in cat.get("types", [])
        ]

    def _score_catalyst(
        self, catalyst_info: Dict[str, Any], analysis: Dict[str, Any]
    ) -> float:
        """Score catalyst suitability for the reaction."""
        base_score = 0.5

        # Boost score based on literature precedents
        precedents = catalyst_info.get("precedents", 0)
        precedent_score = min(precedents / 1000, 0.4)

        # Adjust based on reaction complexity
        complexity = analysis["complexity_score"]
        complexity_penalty = max(0, (complexity - 3) * 0.05)

        return base_score + precedent_score - complexity_penalty


class WhatIfSimulator:
    """Interactive what-if scenario simulation for reactions."""

    def __init__(self, intelligence_engine: ReactionIntelligenceEngine):
        self.intelligence_engine = intelligence_engine
        self.base_scenario: Optional[ReactionScenario] = None

    def set_base_scenario(self, scenario: ReactionScenario) -> None:
        """Set the base scenario for what-if analysis."""
        self.base_scenario = scenario

    def simulate_condition_change(
        self, parameter: str, new_value: Any
    ) -> ReactionScenario:
        """Simulate changing a single reaction parameter."""
        if not self.base_scenario:
            raise ChemiAccelError("No base scenario set")

        # Create modified scenario
        modified_scenario = ReactionScenario(
            reactants=self.base_scenario.reactants.copy(),
            products=self.base_scenario.products.copy(),
            conditions=ReactionConditions(**self.base_scenario.conditions.to_dict()),
        )

        # Apply parameter change
        if hasattr(modified_scenario.conditions, parameter):
            setattr(modified_scenario.conditions, parameter, new_value)
        else:
            raise ChemiAccelError(f"Unknown parameter: {parameter}")

        # Predict impact of change
        self._predict_scenario_outcome(modified_scenario)

        return modified_scenario

    def simulate_catalyst_change(self, new_catalyst: str) -> ReactionScenario:
        """Simulate changing the catalyst."""
        return self.simulate_condition_change("catalyst", new_catalyst)

    def simulate_temperature_sweep(
        self, temp_range: Tuple[float, float], num_points: int = 10
    ) -> List[ReactionScenario]:
        """Simulate temperature sweep."""
        temperatures = np.linspace(temp_range[0], temp_range[1], num_points)
        scenarios = []

        for temp in temperatures:
            scenario = self.simulate_condition_change("temperature", temp)
            scenarios.append(scenario)

        return scenarios

    def compare_scenarios(self, scenarios: List[ReactionScenario]) -> pd.DataFrame:
        """Compare multiple scenarios in tabular format."""
        data = []

        for i, scenario in enumerate(scenarios):
            row = {
                "scenario_id": i,
                "temperature": scenario.conditions.temperature,
                "catalyst": scenario.conditions.catalyst,
                "solvent": scenario.conditions.solvent,
                "predicted_yield": scenario.predicted_yield,
                "predicted_selectivity": scenario.predicted_selectivity,
                "reaction_rate": scenario.reaction_rate,
            }
            data.append(row)

        return pd.DataFrame(data)

    def _predict_scenario_outcome(self, scenario: ReactionScenario) -> None:
        """Predict outcomes for a given scenario."""
        # Simplified prediction model
        base_yield = 0.7
        base_rate = 1.0

        # Temperature effects
        if scenario.conditions.temperature:
            temp_factor = min(scenario.conditions.temperature / 298.15, 2.0)
            base_yield *= 0.8 + 0.2 * temp_factor
            base_rate *= temp_factor**2

        # Catalyst effects
        if scenario.conditions.catalyst:
            base_yield *= 1.2
            base_rate *= 10.0

        # Solvent effects
        if scenario.conditions.solvent == "DMF":
            base_yield *= 1.1
        elif scenario.conditions.solvent == "water":
            base_yield *= 0.8

        # Add some noise
        base_yield *= 0.9 + 0.2 * np.random.random()
        base_rate *= 0.8 + 0.4 * np.random.random()

        scenario.predicted_yield = min(base_yield, 1.0)
        scenario.predicted_selectivity = 0.85 + 0.1 * np.random.random()
        scenario.reaction_rate = base_rate


class ExperimentalFeedbackProcessor:
    """Process experimental feedback to refine predictions."""

    def __init__(self, intelligence_engine: ReactionIntelligenceEngine):
        self.intelligence_engine = intelligence_engine
        self.prediction_errors: List[Dict[str, float]] = []
        self.model_weights = {
            "yield_weight": 1.0,
            "selectivity_weight": 1.0,
            "rate_weight": 1.0,
            "temperature_sensitivity": 1.0,
            "catalyst_effect": 1.0,
        }

    def ingest_experimental_result(self, result: ExperimentalResult) -> None:
        """Ingest new experimental result for model refinement."""
        self.intelligence_engine.experimental_data.append(result)

        # Calculate prediction errors
        if result.scenario.predicted_yield is not None:
            yield_error = abs(result.actual_yield - result.scenario.predicted_yield)

            error_record = {
                "yield_error": yield_error,
                "temperature": result.scenario.conditions.temperature,
                "catalyst": result.scenario.conditions.catalyst,
                "solvent": result.scenario.conditions.solvent,
            }

            if result.scenario.predicted_selectivity and result.actual_selectivity:
                error_record["selectivity_error"] = abs(
                    result.actual_selectivity - result.scenario.predicted_selectivity
                )

            self.prediction_errors.append(error_record)

        logger.info(
            f"Ingested experimental result with {result.actual_yield:.2f} yield"
        )

    def analyze_prediction_accuracy(self) -> Dict[str, float]:
        """Analyze overall prediction accuracy."""
        if not self.prediction_errors:
            return {}

        yield_errors = [e["yield_error"] for e in self.prediction_errors]
        selectivity_errors = [
            e["selectivity_error"]
            for e in self.prediction_errors
            if "selectivity_error" in e
        ]

        analysis = {
            "mean_yield_error": np.mean(yield_errors),
            "std_yield_error": np.std(yield_errors),
            "num_experiments": len(self.prediction_errors),
        }

        if selectivity_errors:
            analysis.update(
                {
                    "mean_selectivity_error": np.mean(selectivity_errors),
                    "std_selectivity_error": np.std(selectivity_errors),
                }
            )

        return analysis

    def refine_model_weights(self) -> None:
        """Refine model weights based on experimental feedback."""
        if len(self.prediction_errors) < 5:
            logger.warning("Insufficient data for weight refinement")
            return

        # Analyze systematic errors
        catalyst_performance = {}
        temperature_effects = {}

        for error in self.prediction_errors:
            catalyst = error.get("catalyst", "none")
            temp = error.get("temperature", 298.15)

            if catalyst not in catalyst_performance:
                catalyst_performance[catalyst] = []
            catalyst_performance[catalyst].append(error["yield_error"])

            temp_bin = int(temp // 50) * 50  # 50K bins
            if temp_bin not in temperature_effects:
                temperature_effects[temp_bin] = []
            temperature_effects[temp_bin].append(error["yield_error"])

        # Adjust weights based on performance
        for catalyst, errors in catalyst_performance.items():
            mean_error = np.mean(errors)
            if mean_error > 0.2:  # High error
                self.model_weights["catalyst_effect"] *= 0.9
            elif mean_error < 0.1:  # Low error
                self.model_weights["catalyst_effect"] *= 1.1

        # Constrain weights
        for key in self.model_weights:
            self.model_weights[key] = np.clip(self.model_weights[key], 0.5, 2.0)

        logger.info(f"Updated model weights: {self.model_weights}")

    def suggest_next_experiments(
        self, base_reaction: Tuple[List[str], List[str]], num_suggestions: int = 3
    ) -> List[ReactionScenario]:
        """Suggest next experiments to improve model accuracy."""
        reactants, products = base_reaction

        # Identify underexplored parameter space
        tested_temps = set()
        tested_catalysts = set()

        for result in self.intelligence_engine.experimental_data:
            if result.scenario.conditions.temperature:
                tested_temps.add(result.scenario.conditions.temperature)
            if result.scenario.conditions.catalyst:
                tested_catalysts.add(result.scenario.conditions.catalyst)

        suggestions = []

        # Suggest temperature exploration
        if len(tested_temps) < 3:
            for temp in [298.15, 353.15, 398.15]:
                if temp not in tested_temps:
                    conditions = ReactionConditions(temperature=temp, solvent="DMF")
                    scenario = ReactionScenario(
                        reactants=reactants, products=products, conditions=conditions
                    )
                    suggestions.append(scenario)
                    if len(suggestions) >= num_suggestions:
                        break

        # Suggest catalyst exploration
        common_catalysts = ["Pd(PPh3)4", "CuI", "TFA"]
        for catalyst in common_catalysts:
            if catalyst not in tested_catalysts and len(suggestions) < num_suggestions:
                conditions = ReactionConditions(
                    temperature=353.15, catalyst=catalyst, solvent="DMF"
                )
                scenario = ReactionScenario(
                    reactants=reactants, products=products, conditions=conditions
                )
                suggestions.append(scenario)

        return suggestions[:num_suggestions]
