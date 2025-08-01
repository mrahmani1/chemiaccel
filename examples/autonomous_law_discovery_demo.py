"""
Autonomous Scientific Law & Causal Relationship Discovery Demo

This example demonstrates the comprehensive autonomous discovery capabilities
for uncovering scientific laws, causal relationships, and generating testable
hypotheses from chemical datasets.
"""

import warnings
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

from chemi_accel.ai.law_discovery import (
    AdaptiveLearningEngine,
    AutomatedExperimentSuggester,
    AutonomousLawDiscovery,
    CausalInferenceEngine,
    CausalRelationship,
    Hypothesis,
    HypothesisGenerator,
    KnowledgeGraphIntegrator,
    PatternRecognitionEngine,
    ScientificLaw,
    SymbolicRegressionEngine,
)


def generate_synthetic_chemical_data(n_samples: int = 500) -> pd.DataFrame:
    """
    Generate synthetic chemical reaction data with known relationships.

    This creates a realistic dataset with:
    - Temperature effects on reaction rates
    - Concentration dependencies
    - Catalyst effects
    - Pressure influences
    - Known causal relationships
    """
    np.random.seed(42)

    print("🧪 Generating synthetic chemical reaction dataset...")

    # Independent variables
    temperature = np.random.uniform(20, 100, n_samples)  # Celsius
    pressure = np.random.uniform(1, 10, n_samples)  # atm
    catalyst_conc = np.random.uniform(0.01, 0.5, n_samples)  # mol/L
    reactant_a_conc = np.random.uniform(0.1, 2.0, n_samples)  # mol/L
    reactant_b_conc = np.random.uniform(0.1, 1.5, n_samples)  # mol/L
    ph = np.random.uniform(2, 12, n_samples)

    # Known relationships (ground truth)
    # Arrhenius-like temperature dependence
    temp_factor = np.exp(-5000 / (8.314 * (temperature + 273.15)))

    # Reaction rate (complex relationship)
    reaction_rate = (
        0.1
        * temp_factor
        * catalyst_conc**0.5
        * reactant_a_conc
        * reactant_b_conc
        * np.sqrt(pressure)
        * (1 + 0.1 * np.random.randn(n_samples))
    )

    # Yield depends on reaction rate and conditions
    yield_percent = np.clip(
        20
        + 15 * np.log(reaction_rate + 0.01)
        + 2 * catalyst_conc
        + 0.5 * pressure
        - 0.3 * (ph - 7) ** 2
        + np.random.randn(n_samples),
        0,
        100,
    )

    # Selectivity (competing reactions)
    selectivity = np.clip(
        80
        - 0.2 * temperature
        + 10 * catalyst_conc
        - 0.1 * pressure
        + np.random.randn(n_samples),
        0,
        100,
    )

    # Side product formation
    side_product_rate = 0.02 * np.exp(
        (temperature - 60) / 20
    ) * reactant_a_conc**2 * (1 - catalyst_conc) + 0.01 * np.random.randn(n_samples)
    side_product_rate = np.clip(side_product_rate, 0, None)

    # Energy consumption
    energy_consumption = (
        100
        + 2 * temperature
        + 10 * pressure
        + 50 / (catalyst_conc + 0.01)
        + 10 * np.random.randn(n_samples)
    )

    # Cost (derived from multiple factors)
    cost_per_kg = (
        10
        + 0.1 * energy_consumption
        + 100 * catalyst_conc
        + 0.5 * pressure
        + np.random.randn(n_samples)
    )

    data = pd.DataFrame(
        {
            "temperature": temperature,
            "pressure": pressure,
            "catalyst_concentration": catalyst_conc,
            "reactant_a_concentration": reactant_a_conc,
            "reactant_b_concentration": reactant_b_conc,
            "ph": ph,
            "reaction_rate": reaction_rate,
            "yield_percent": yield_percent,
            "selectivity": selectivity,
            "side_product_rate": side_product_rate,
            "energy_consumption": energy_consumption,
            "cost_per_kg": cost_per_kg,
        }
    )

    print(
        f"✅ Generated dataset with {n_samples} samples and {len(data.columns)} variables"
    )
    print(f"📊 Variables: {list(data.columns)}")

    return data


def demonstrate_causal_inference(data: pd.DataFrame):
    """Demonstrate GPU-accelerated causal inference capabilities."""

    print("\n" + "=" * 80)
    print("🔍 CAUSAL INFERENCE ANALYSIS")
    print("=" * 80)

    # Initialize causal inference engine
    causal_engine = CausalInferenceEngine(backend="auto")
    print(f"🖥️  Using backend: {causal_engine.backend}")

    # Test specific causal relationships
    print("\n📈 Testing Granger causality: temperature → reaction_rate")
    granger_results = causal_engine.granger_causality_test(
        data, "temperature", "reaction_rate", max_lag=5
    )

    print("Granger Causality Test Results:")
    for lag, result in granger_results.items():
        f_stat = result["f_statistic"]
        print(f"  {lag}: F-statistic = {f_stat:.3f}")

    # Discover full causal network
    print("\n🌐 Discovering causal network structure...")
    variable_names = [
        "temperature",
        "pressure",
        "catalyst_concentration",
        "reaction_rate",
        "yield_percent",
        "selectivity",
    ]

    causal_relationships = causal_engine.discover_causal_network(
        data, variable_names, significance_threshold=0.1
    )

    print(f"🔗 Found {len(causal_relationships)} causal relationships:")
    for i, rel in enumerate(causal_relationships[:5], 1):  # Show top 5
        print(f"  {i}. {' + '.join(rel.cause_variables)} → {rel.effect_variable}")
        print(
            f"     Strength: {rel.causal_strength:.3f}, Confidence: {rel.confidence_score:.3f}"
        )
        print(f"     Mechanism: {rel.mechanism_hypothesis}")

    return causal_relationships


def demonstrate_symbolic_regression(data: pd.DataFrame):
    """Demonstrate symbolic AI for equation discovery."""

    print("\n" + "=" * 80)
    print("🧮 SYMBOLIC REGRESSION & EQUATION DISCOVERY")
    print("=" * 80)

    # Initialize symbolic regression engine
    symbolic_engine = SymbolicRegressionEngine(backend="auto")

    # Discover equations for reaction rate
    print("\n🔬 Discovering equations for reaction_rate...")
    feature_cols = ["temperature", "catalyst_concentration", "reactant_a_concentration"]
    X = data[feature_cols].values
    y = data["reaction_rate"].values

    discovered_laws = symbolic_engine.discover_equations(
        X, y, feature_cols, max_complexity=15
    )

    print(f"📜 Discovered {len(discovered_laws)} potential laws:")
    for i, law in enumerate(discovered_laws, 1):
        print(f"\n  Law {i}:")
        print(f"    Equation: {law.equation}")
        print(f"    R² = {law.r_squared:.4f}")
        print(f"    Variables: {law.variables}")
        print(f"    Coefficients: {law.coefficients}")
        print(f"    Domain: {law.domain_constraints}")

    # Discover equations for yield
    print("\n🎯 Discovering equations for yield_percent...")
    feature_cols_yield = ["reaction_rate", "catalyst_concentration", "pressure", "ph"]
    X_yield = data[feature_cols_yield].values
    y_yield = data["yield_percent"].values

    yield_laws = symbolic_engine.discover_equations(
        X_yield, y_yield, feature_cols_yield, max_complexity=12
    )

    print(f"📈 Discovered {len(yield_laws)} yield prediction laws:")
    for i, law in enumerate(yield_laws[:3], 1):  # Show top 3
        print(f"  {i}. {law.equation} (R² = {law.r_squared:.4f})")

    return discovered_laws + yield_laws


def demonstrate_pattern_recognition(data: pd.DataFrame):
    """Demonstrate advanced pattern recognition and hypothesis generation."""

    print("\n" + "=" * 80)
    print("🔍 PATTERN RECOGNITION & HYPOTHESIS GENERATION")
    print("=" * 80)

    # Initialize pattern recognition engine
    pattern_engine = PatternRecognitionEngine(backend="auto")

    # Discover patterns
    target_variables = ["reaction_rate", "yield_percent", "selectivity"]
    patterns = pattern_engine.discover_patterns(
        data, target_variables, min_confidence=0.6
    )

    print(f"🌟 Discovered {len(patterns)} significant patterns:")
    for i, pattern in enumerate(patterns, 1):
        print(f"\n  Pattern {i} ({pattern['type']}):")
        if pattern["type"] == "feature_importance":
            print(f"    Target: {pattern['target']}")
            print(f"    Important features: {pattern['important_features']}")
            print(f"    Confidence: {pattern['confidence']:.3f}")
        elif pattern["type"] == "correlation_cluster":
            print("    High correlations found:")
            for corr in pattern["correlations"][:3]:
                print(
                    f"      {corr['var1']} ↔ {corr['var2']}: r = {corr['correlation']:.3f}"
                )
        elif pattern["type"] == "nonlinear_relationship":
            print(f"    Target: {pattern['target']}")
            print(f"    Non-linear predictors: {pattern['nonlinear_predictors']}")
            print(f"    MI scores: {pattern['mi_scores']}")

    return patterns


def demonstrate_hypothesis_generation(patterns: List[Dict], causal_relationships: List):
    """Demonstrate hypothesis generation from patterns and relationships."""

    print("\n" + "=" * 80)
    print("💡 HYPOTHESIS GENERATION")
    print("=" * 80)

    # Initialize hypothesis generator
    hypothesis_generator = HypothesisGenerator()

    # Generate hypotheses
    hypotheses = hypothesis_generator.generate_hypotheses(
        patterns, causal_relationships, min_testability=0.6
    )

    print(f"🎯 Generated {len(hypotheses)} testable hypotheses:")

    for i, hyp in enumerate(hypotheses, 1):
        print(f"\n  Hypothesis {i}:")
        print(f"    Statement: {hyp.statement}")
        print(f"    Variables: {hyp.variables_involved}")
        print(f"    Confidence: {hyp.confidence_score:.3f}")
        print(f"    Testability: {hyp.testability_score:.3f}")
        print(f"    Method: {hyp.generation_method}")

        if hyp.suggested_experiments:
            print("    Suggested experiments:")
            for j, exp in enumerate(hyp.suggested_experiments[:2], 1):
                print(f"      {j}. {exp['type']}: {exp['description']}")

        if hyp.falsifiability_criteria:
            print(f"    Falsifiability: {hyp.falsifiability_criteria[0]}")

    return hypotheses


def demonstrate_experiment_suggestion(hypotheses: List[Hypothesis]):
    """Demonstrate automated experiment suggestion system."""

    print("\n" + "=" * 80)
    print("🧪 AUTOMATED EXPERIMENT SUGGESTION")
    print("=" * 80)

    # Initialize experiment suggester
    experiment_suggester = AutomatedExperimentSuggester()

    # Define available resources and constraints
    resources = {
        "cost_per_sample": 50,
        "cost_per_dose_group": 200,
        "cost_per_condition": 150,
        "cost_per_measurement": 25,
        "max_budget": 15000,
    }

    constraints = {
        "max_samples": 300,
        "max_subjects": 100,
        "max_duration_days": 90,
        "max_budget": 15000,
    }

    print(f"💰 Available budget: ${resources['max_budget']:,}")
    print(f"📊 Max samples: {constraints['max_samples']}")
    print(f"⏱️  Max duration: {constraints['max_duration_days']} days")

    all_experiments = []

    # Suggest experiments for top hypotheses
    for i, hypothesis in enumerate(hypotheses[:3], 1):  # Top 3 hypotheses
        print(f"\n🔬 Experiments for Hypothesis {i}:")
        print(f"   '{hypothesis.statement[:60]}...'")

        experiments = experiment_suggester.suggest_experiments(
            hypothesis, resources, constraints
        )

        for j, exp in enumerate(experiments, 1):
            print(f"\n  Experiment {i}.{j}: {exp['title']}")
            print(f"    Type: {exp['type']}")
            print(f"    Priority Score: {exp['priority_score']:.3f}")
            print(f"    Estimated Cost: ${exp['estimated_cost']:,}")
            print(f"    Duration: {exp['estimated_duration']} days")
            print(f"    Information Gain: {exp['information_gain']:.3f}")
            print(f"    Feasibility: {exp['feasibility']:.3f}")

            if "design" in exp:
                design = exp["design"]
                if "sample_size" in design:
                    print(f"    Sample Size: {design['sample_size']}")
                elif "sample_size_per_group" in design:
                    print(
                        f"    Sample Size per Group: {design['sample_size_per_group']}"
                    )

        all_experiments.extend(experiments)

    # Show prioritized experiment list
    print("\n📋 TOP RECOMMENDED EXPERIMENTS (sorted by priority):")
    sorted_experiments = sorted(
        all_experiments, key=lambda x: x["priority_score"], reverse=True
    )

    total_cost = 0
    for i, exp in enumerate(sorted_experiments[:5], 1):
        cost = exp["estimated_cost"]
        total_cost += cost
        print(f"  {i}. {exp['title']}")
        print(
            f"     Priority: {exp['priority_score']:.3f} | Cost: ${cost:,} | Duration: {exp['estimated_duration']}d"
        )
        if total_cost <= constraints["max_budget"]:
            print(f"     ✅ Fits in budget (running total: ${total_cost:,})")
        else:
            print(f"     ❌ Exceeds budget (would be: ${total_cost:,})")

    return all_experiments


def demonstrate_adaptive_learning(
    discovered_laws: List[ScientificLaw], data: pd.DataFrame
):
    """Demonstrate adaptive learning for law refinement."""

    print("\n" + "=" * 80)
    print("🔄 ADAPTIVE LEARNING & LAW REFINEMENT")
    print("=" * 80)

    if not discovered_laws:
        print("⚠️  No laws available for refinement demonstration")
        return []

    # Initialize adaptive learning engine
    adaptive_learner = AdaptiveLearningEngine()

    # Simulate new experimental data
    print("🆕 Simulating new experimental validation data...")
    np.random.seed(123)  # Different seed for new data

    new_data = pd.DataFrame(
        {
            "temperature": np.random.uniform(80, 120, 50),  # Extended range
            "catalyst_concentration": np.random.uniform(0.4, 0.8, 50),
            "reactant_a_concentration": np.random.uniform(1.8, 2.5, 50),
            "reaction_rate": np.random.uniform(0.5, 2.0, 50),  # New measurements
        }
    )

    # Simulated experimental validation results
    experimental_validation = [
        {
            "experiment_id": "validation_001",
            "type": "kinetic_study",
            "validation_outcome": "confirmed",
            "confidence": 0.92,
            "methodology": "UV-Vis spectroscopy tracking",
            "conditions": {"temperature_range": "80-120°C", "replications": 5},
        },
        {
            "experiment_id": "validation_002",
            "type": "factorial_design",
            "validation_outcome": "partially_confirmed",
            "confidence": 0.76,
            "methodology": "GC-MS product analysis",
            "conditions": {"factorial_levels": "2^3", "center_points": 3},
        },
        {
            "experiment_id": "validation_003",
            "type": "dose_response",
            "validation_outcome": "confirmed",
            "confidence": 0.88,
            "methodology": "HPLC quantitative analysis",
            "conditions": {"dose_levels": 8, "replicates": 4},
        },
    ]

    print(f"📊 New data points: {len(new_data)}")
    print(f"🧪 Validation experiments: {len(experimental_validation)}")

    refined_laws = []

    # Refine each discovered law
    for i, law in enumerate(discovered_laws[:3], 1):  # Refine top 3 laws
        print(f"\n🔬 Refining Law {i}:")
        print(f"   Original: {law.equation}")
        print(f"   Original R²: {law.r_squared:.4f}")
        print(f"   Original Confidence: {law.confidence_scores.get('overall', 0):.3f}")

        try:
            refined_law = adaptive_learner.refine_law(
                law, new_data, experimental_validation
            )

            print(f"   Refined R²: {refined_law.r_squared:.4f}")
            print(
                f"   New Confidence: {refined_law.confidence_scores.get('overall', 0):.3f}"
            )
            print(
                f"   Validation Score: {refined_law.confidence_scores.get('experimental_validation', 0):.3f}"
            )
            print(
                f"   Domain Updated: {len(refined_law.domain_constraints)} constraints"
            )
            print(
                f"   Refinement History: {len(refined_law.refinement_history)} entries"
            )

            # Show coefficient changes
            if refined_law.coefficients != law.coefficients:
                print("   Coefficient Updates:")
                for coef_name in law.coefficients:
                    if coef_name in refined_law.coefficients:
                        old_val = law.coefficients[coef_name]
                        new_val = refined_law.coefficients[coef_name]
                        change = new_val - old_val
                        print(
                            f"     {coef_name}: {old_val:.4f} → {new_val:.4f} (Δ{change:+.4f})"
                        )

            refined_laws.append(refined_law)

        except Exception as e:
            print(f"   ❌ Refinement failed: {e}")
            refined_laws.append(law)  # Keep original

    # Show refinement statistics
    print("\n📈 REFINEMENT STATISTICS:")
    print(f"   Total refinements performed: {len(adaptive_learner.refinement_history)}")

    if adaptive_learner.refinement_history:
        avg_confidence_change = np.mean(
            [r["confidence_change"] for r in adaptive_learner.refinement_history]
        )
        print(f"   Average confidence improvement: {avg_confidence_change:+.3f}")

    return refined_laws


def demonstrate_knowledge_graph_integration(discovered_laws: List[ScientificLaw]):
    """Demonstrate integration with intelligent chemical knowledge graph."""

    print("\n" + "=" * 80)
    print("🕸️  KNOWLEDGE GRAPH INTEGRATION")
    print("=" * 80)

    if not discovered_laws:
        print("⚠️  No laws available for knowledge graph integration")
        return []

    # Initialize knowledge graph integrator
    kg_integrator = KnowledgeGraphIntegrator(kg_backend="memory")

    print(f"🧠 Knowledge Graph Backend: {kg_integrator.kg_backend}")
    print(f"📚 Law Ontology Categories: {len(kg_integrator.law_ontology['law_types'])}")

    integration_results = []

    # Integrate top laws into knowledge graph
    for i, law in enumerate(discovered_laws[:3], 1):
        print(f"\n🔗 Integrating Law {i} into Knowledge Graph:")
        print(f"   Equation: {law.equation}")

        try:
            # Simulate existing laws for relationship detection
            existing_laws = discovered_laws[: i - 1] if i > 1 else []

            integration_result = kg_integrator.integrate_law(law, existing_laws)

            print("   ✅ Integration successful!")
            print(f"   Integration ID: {integration_result['integration_id']}")
            print(
                f"   Nodes added: {integration_result['graph_updates']['nodes_added']}"
            )
            print(
                f"   Relationships found: {integration_result['graph_updates']['relationships_added']}"
            )

            # Show law node details
            law_node = integration_result["law_node"]
            print(f"   Law Type: {law_node['properties']['law_type']}")
            print(
                f"   Validation Status: {law_node['properties']['validation_status']}"
            )
            print(
                f"   Applicability Domain: {law_node['properties']['applicability_domain']}"
            )

            # Show variable nodes
            var_nodes = integration_result["variable_nodes"]
            print(f"   Variable Nodes: {len(var_nodes)}")
            for var_node in var_nodes:
                var_name = var_node["properties"]["name"]
                var_category = var_node["properties"]["category"]
                var_unit = var_node["properties"]["measurement_unit"]
                print(f"     - {var_name} ({var_category}, {var_unit})")

            # Show relationships with other laws
            if integration_result["relationships"]:
                print("   Relationships with existing laws:")
                for rel in integration_result["relationships"]:
                    print(f"     - {rel['type']} (strength: {rel['strength']:.3f})")
                    print(f"       Common variables: {rel['common_variables']}")

            integration_results.append(integration_result)

        except Exception as e:
            print(f"   ❌ Integration failed: {e}")

    # Summary of knowledge graph state
    print("\n📊 KNOWLEDGE GRAPH SUMMARY:")
    total_nodes = sum(r["graph_updates"]["nodes_added"] for r in integration_results)
    total_relationships = sum(
        r["graph_updates"]["relationships_added"] for r in integration_results
    )

    print(f"   Total nodes in graph: {total_nodes}")
    print(f"   Total relationships: {total_relationships}")
    print(
        f"   Law types represented: {len(set(r['law_node']['properties']['law_type'] for r in integration_results))}"
    )

    return integration_results


def demonstrate_complete_autonomous_discovery(data: pd.DataFrame):
    """Demonstrate the complete autonomous discovery pipeline."""

    print("\n" + "=" * 80)
    print("🤖 COMPLETE AUTONOMOUS DISCOVERY PIPELINE")
    print("=" * 80)

    # Initialize the main orchestrator
    autonomous_discovery = AutonomousLawDiscovery(backend="auto")

    print(f"🖥️  Backend: {autonomous_discovery.causal_engine.backend}")
    print("🚀 Starting comprehensive autonomous discovery...")

    # Define target variables for prediction/explanation
    target_variables = ["reaction_rate", "yield_percent", "selectivity", "cost_per_kg"]

    # Define all variables for causal analysis
    all_variables = [
        "temperature",
        "pressure",
        "catalyst_concentration",
        "reactant_a_concentration",
        "reactant_b_concentration",
        "ph",
        "reaction_rate",
        "yield_percent",
        "selectivity",
        "side_product_rate",
        "energy_consumption",
        "cost_per_kg",
    ]

    # Configuration for discovery process
    discovery_config = {
        "max_equation_complexity": 12,
        "min_confidence": 0.65,
        "max_causal_lag": 3,
        "significance_threshold": 0.05,
    }

    print(f"🎯 Target variables: {target_variables}")
    print(f"📊 Total variables: {len(all_variables)}")
    print(f"⚙️  Discovery config: {discovery_config}")

    # Run complete discovery
    print("\n⏳ Running autonomous discovery (this may take a moment)...")

    try:
        results = autonomous_discovery.discover_laws_and_relationships(
            data=data,
            target_variables=target_variables,
            variable_names=all_variables,
            discovery_config=discovery_config,
        )

        print("✅ Discovery completed successfully!")

        # Display comprehensive results
        print("\n📈 DISCOVERY RESULTS SUMMARY:")
        summary = results["discovery_summary"]
        print(f"   Laws discovered: {summary['laws_discovered']}")
        print(f"   Causal relationships: {summary['causal_relationships']}")
        print(f"   Patterns identified: {summary['patterns_identified']}")
        print(f"   Hypotheses generated: {summary['hypotheses_generated']}")
        print(f"   Experiments suggested: {summary['experiments_suggested']}")

        # Show top discovered laws
        if results["discovered_laws"]:
            print("\n🏆 TOP DISCOVERED LAWS:")
            for i, law_dict in enumerate(results["discovered_laws"][:3], 1):
                print(f"   {i}. {law_dict['equation']}")
                print(f"      R² = {law_dict['r_squared']:.4f}")
                print(
                    f"      Confidence = {law_dict['confidence_scores'].get('overall', 0):.3f}"
                )

        # Show top causal relationships
        if results["causal_relationships"]:
            print("\n🔗 TOP CAUSAL RELATIONSHIPS:")
            for i, rel in enumerate(results["causal_relationships"][:3], 1):
                causes = " + ".join(rel["cause_variables"])
                effect = rel["effect_variable"]
                strength = rel["causal_strength"]
                print(f"   {i}. {causes} → {effect} (strength: {strength:.3f})")

        # Show top hypotheses
        if results["hypotheses"]:
            print("\n💡 TOP GENERATED HYPOTHESES:")
            for i, hyp in enumerate(results["hypotheses"][:3], 1):
                print(f"   {i}. {hyp['statement'][:80]}...")
                print(f"      Confidence: {hyp['confidence_score']:.3f}")
                print(f"      Testability: {hyp['testability_score']:.3f}")

        # Show experimental suggestions
        if results["experimental_suggestions"]:
            print("\n🧪 TOP EXPERIMENTAL SUGGESTIONS:")
            sorted_experiments = sorted(
                results["experimental_suggestions"],
                key=lambda x: x["priority_score"],
                reverse=True,
            )

            for i, exp in enumerate(sorted_experiments[:3], 1):
                print(f"   {i}. {exp['title']}")
                print(f"      Type: {exp['type']}")
                print(f"      Priority: {exp['priority_score']:.3f}")
                print(f"      Cost: ${exp['estimated_cost']:,}")

        # Show knowledge graph integrations
        if results["knowledge_graph_integrations"]:
            print("\n🕸️  KNOWLEDGE GRAPH INTEGRATIONS:")
            for i, kg_result in enumerate(results["knowledge_graph_integrations"], 1):
                print(f"   {i}. {kg_result['integration_id']}")
                print(f"      Nodes added: {kg_result['graph_updates']['nodes_added']}")
                print(
                    f"      Relationships: {kg_result['graph_updates']['relationships_added']}"
                )

        # Get additional insights
        print("\n📊 DISCOVERY INSIGHTS:")
        insights = autonomous_discovery.get_discovery_insights()
        print(f"   High-confidence laws: {insights['high_confidence_laws']}")
        print(f"   Validated laws: {insights['validated_laws']}")
        if insights["most_confident_law"]:
            print(f"   Most confident law: {insights['most_confident_law']}")

        return results

    except Exception as e:
        print(f"❌ Discovery failed: {e}")
        return None


def create_visualization_dashboard(data: pd.DataFrame, results: Dict[str, Any] = None):
    """Create visualization dashboard for discovery results."""

    print("\n" + "=" * 80)
    print("📊 CREATING VISUALIZATION DASHBOARD")
    print("=" * 80)

    # Set up the plotting style
    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Autonomous Scientific Law Discovery - Analysis Dashboard",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Correlation heatmap
    ax1 = axes[0, 0]
    key_vars = [
        "temperature",
        "pressure",
        "catalyst_concentration",
        "reaction_rate",
        "yield_percent",
    ]
    corr_matrix = data[key_vars].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="RdYlBu_r", center=0, ax=ax1, fmt=".2f")
    ax1.set_title("Variable Correlation Matrix")

    # 2. Temperature vs Reaction Rate
    ax2 = axes[0, 1]
    scatter = ax2.scatter(
        data["temperature"],
        data["reaction_rate"],
        c=data["catalyst_concentration"],
        cmap="viridis",
        alpha=0.6,
    )
    ax2.set_xlabel("Temperature (°C)")
    ax2.set_ylabel("Reaction Rate")
    ax2.set_title("Temperature vs Reaction Rate\n(colored by catalyst conc.)")
    plt.colorbar(scatter, ax=ax2, label="Catalyst Conc.")

    # 3. Yield distribution
    ax3 = axes[0, 2]
    ax3.hist(data["yield_percent"], bins=30, alpha=0.7, edgecolor="black")
    ax3.axvline(
        data["yield_percent"].mean(),
        color="red",
        linestyle="--",
        label=f'Mean: {data["yield_percent"].mean():.1f}%',
    )
    ax3.set_xlabel("Yield (%)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Yield Distribution")
    ax3.legend()

    # 4. Multi-dimensional relationship
    ax4 = axes[1, 0]
    ax4.scatter(
        data["reaction_rate"],
        data["yield_percent"],
        s=data["catalyst_concentration"] * 200,
        alpha=0.6,
        c=data["temperature"],
        cmap="coolwarm",
    )
    ax4.set_xlabel("Reaction Rate")
    ax4.set_ylabel("Yield (%)")
    ax4.set_title("Reaction Rate vs Yield\n(size: catalyst, color: temp)")

    # 5. Cost analysis
    ax5 = axes[1, 1]
    ax5.scatter(data["energy_consumption"], data["cost_per_kg"], alpha=0.6)
    ax5.set_xlabel("Energy Consumption")
    ax5.set_ylabel("Cost per kg")
    ax5.set_title("Energy vs Cost Relationship")

    # 6. Discovery results summary (if available)
    ax6 = axes[1, 2]
    if results:
        summary = results["discovery_summary"]
        categories = list(summary.keys())
        values = list(summary.values())

        bars = ax6.bar(range(len(categories)), values, color="skyblue", alpha=0.7)
        ax6.set_xticks(range(len(categories)))
        ax6.set_xticklabels(
            [cat.replace("_", " ").title() for cat in categories],
            rotation=45,
            ha="right",
        )
        ax6.set_ylabel("Count")
        ax6.set_title("Discovery Results Summary")

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax6.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.05,
                f"{value}",
                ha="center",
                va="bottom",
            )
    else:
        ax6.text(
            0.5,
            0.5,
            "No discovery results\navailable",
            ha="center",
            va="center",
            transform=ax6.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
        )
        ax6.set_title("Discovery Results")

    plt.tight_layout()
    plt.savefig("autonomous_law_discovery_dashboard.png", dpi=300, bbox_inches="tight")
    print("📈 Dashboard saved as 'autonomous_law_discovery_dashboard.png'")

    try:
        plt.show()
    except:
        print("📊 Dashboard created (display not available in current environment)")


def main():
    """Main demonstration function."""

    print("🚀 AUTONOMOUS SCIENTIFIC LAW & CAUSAL RELATIONSHIP DISCOVERY")
    print("=" * 80)
    print("This demo showcases the complete autonomous discovery pipeline for")
    print("uncovering scientific laws, causal relationships, and generating")
    print("testable hypotheses from chemical datasets.")
    print("=" * 80)

    # Generate synthetic dataset
    data = generate_synthetic_chemical_data(n_samples=600)

    # Individual component demonstrations
    causal_relationships = demonstrate_causal_inference(data)
    discovered_laws = demonstrate_symbolic_regression(data)
    patterns = demonstrate_pattern_recognition(data)
    hypotheses = demonstrate_hypothesis_generation(patterns, causal_relationships)
    experiments = demonstrate_experiment_suggestion(hypotheses)
    refined_laws = demonstrate_adaptive_learning(discovered_laws, data)
    kg_integrations = demonstrate_knowledge_graph_integration(
        refined_laws or discovered_laws
    )

    # Complete autonomous discovery
    complete_results = demonstrate_complete_autonomous_discovery(data)

    # Create visualization dashboard
    create_visualization_dashboard(data, complete_results)

    # Final summary
    print("\n" + "=" * 80)
    print("🎉 AUTONOMOUS DISCOVERY DEMONSTRATION COMPLETE!")
    print("=" * 80)

    if complete_results:
        summary = complete_results["discovery_summary"]
        print(f"✅ Successfully discovered {summary['laws_discovered']} scientific laws")
        print(f"✅ Identified {summary['causal_relationships']} causal relationships")
        print(f"✅ Generated {summary['hypotheses_generated']} testable hypotheses")
        print(f"✅ Suggested {summary['experiments_suggested']} validation experiments")

        print("\n🚀 This demonstrates ChemiAccel's ability to autonomously:")
        print("   • Discover causal relationships using GPU-accelerated algorithms")
        print("   • Generate symbolic equations through genetic programming")
        print("   • Identify non-obvious patterns in multi-modal datasets")
        print("   • Generate testable hypotheses with experimental protocols")
        print("   • Suggest optimal validation experiments with cost analysis")
        print("   • Adaptively refine laws based on new experimental data")
        print("   • Integrate discoveries into intelligent knowledge graphs")

        print(
            f"\n📊 Dataset analyzed: {data.shape[0]} samples × {data.shape[1]} variables"
        )
        print(f"🧮 Backend used: {complete_results['metadata']['backend_used']}")
        print(
            f"⏱️  Discovery completed at: {complete_results['metadata']['discovery_timestamp']}"
        )

    print("\n🔬 The autonomous discovery system successfully demonstrated:")
    print("   • GPU-accelerated causal inference with Granger causality testing")
    print("   • Symbolic AI for equation discovery using genetic programming")
    print("   • Advanced pattern recognition with feature importance analysis")
    print("   • Intelligent hypothesis generation with testability scoring")
    print("   • Automated experiment design with cost-benefit optimization")
    print("   • Adaptive learning for continuous law refinement")
    print("   • Knowledge graph integration for scientific law formalization")


if __name__ == "__main__":
    main()
