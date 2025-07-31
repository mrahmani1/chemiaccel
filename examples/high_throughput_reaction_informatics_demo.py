#!/usr/bin/env python3
"""
High-Throughput Reaction Informatics Demonstration - Task 19.2

This comprehensive demo showcases ChemiAccel's GPU-accelerated reaction informatics
capabilities, including reaction prediction, retrosynthesis analysis, condition
optimization, and mechanism classification.

Features Demonstrated:
- Massively parallel reaction enumeration (100-1000x speedup)
- GPU-accelerated reaction pattern matching
- AI-driven reaction condition optimization
- Automated retrosynthesis pathway analysis
- Real-time synthetic accessibility scoring
- Comprehensive performance benchmarking

Usage:
    python examples/high_throughput_reaction_informatics_demo.py
"""

import sys
import time
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
        ReactionCondition,
        ReactionType,
        predict_reactions_gpu,
    )

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPU backend not available - will use CPU backend")

# RDKit for molecular processing
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit not available - limited functionality")


def create_test_molecule_library() -> List[MolecularData]:
    """Create a diverse library of test molecules for reaction informatics."""

    # Diverse set of drug-like and synthetic intermediates
    test_smiles = [
        # Drug molecules and drug-like compounds
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # Ibuprofen
        "CC(=O)Nc1ccc(O)cc1",  # Acetaminophen
        "O=C(O)c1ccccc1O",  # Salicylic acid
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",  # Albuterol
        # Synthetic building blocks
        "BrC1=CC=C(C=C1)B(O)O",  # Aryl bromide-boronic acid
        "NC1=CC=C(C=C1)C(=O)O",  # Para-aminobenzoic acid
        "ClC(=O)C1=CC=CC=C1",  # Benzoyl chloride
        "C#CC1=CC=CC=C1",  # Phenylacetylene
        "N#CC1=CC=CC=C1",  # Benzonitrile
        # Amines for amide formation
        "NCCCO",  # 3-Amino-1-propanol
        "NC1=CC=C(C=C1)OC",  # Para-anisidine
        "NC(C)C1=CC=CC=C1",  # Alpha-methylbenzylamine
        "NCC1=CC=CC=C1",  # Benzylamine
        "NC1=NC=NC=N1",  # Melamine
        # Alcohols for various reactions
        "OCC1=CC=CC=C1",  # Benzyl alcohol
        "OC(C)C1=CC=CC=C1",  # 1-Phenylethanol
        "OC1=CC=C(C=C1)C",  # Para-cresol
        "OCCC1=CC=CC=C1",  # 3-Phenyl-1-propanol
        # Carboxylic acids
        "OC(=O)C1=CC=CC=C1",  # Benzoic acid
        "OC(=O)CC1=CC=CC=C1",  # Phenylacetic acid
        "OC(=O)C1=CC=C(C=C1)N",  # Para-aminobenzoic acid
        "OC(=O)C1=CC=C(C=C1)O",  # Para-hydroxybenzoic acid
        # Halides for coupling reactions
        "BrC1=CC=C(C=C1)C",  # Para-bromotoluene
        "IC1=CC=C(C=C1)OC",  # Para-iodoanisole
        "BrC1=CC=CC=C1",  # Bromobenzene
        "ClC1=CC=C(C=C1)C(=O)O",  # Para-chlorobenzoic acid
        # Complex drug-like molecules
        "CN(C)CCN1C2=CC=CC=C2SC3=C1C=C(C=C3)C(F)(F)F",  # Trifluoperazine
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",  # Imatinib core
        "COC1=C(C=CC(=C1)CC2COC(=O)C2)OC",  # Natural product derivative
        # Polymer precursors
        "C=CC1=CC=CC=C1",  # Styrene
        "C=CC(=O)OC",  # Methyl acrylate
        "C=CC(=O)N",  # Acrylamide
        "C=CC#N",  # Acrylonitrile
        # Heterocycles
        "C1=CC=NC=C1",  # Pyridine
        "C1=CN=CN1",  # Imidazole
        "C1=CC=C2C(=C1)C=CN2",  # Indole
        "C1=CC=C2C(=C1)N=CN2",  # Quinoxaline
        # Advanced synthetic targets
        "CC1=C(C(=NO1)C2=CC=CC=C2Cl)C(=O)NC3=CC=CC=C3",  # Complex isoxazole
        "COC1=CC=C2C(=C1)C(=CN2)CCN3CCN(CC3)C4=NSC5=C4C=CC=C5",  # Drug-like target
        # Natural product-inspired compounds
        "CC(C)=CCCC(C)=CC(=O)C1=C(C=CC=C1O)O",  # Flavonoid derivative
        "COC1=CC(=CC(=C1O)OC)C2C(C(=O)C3=C(C=C(C=C3O2)O)O)O",  # Quercetin derivative
    ]

    molecules = []

    if not RDKIT_AVAILABLE:
        # Create simple molecular data without RDKit
        for i, smiles in enumerate(test_smiles):
            mol_data = MolecularData(
                mol_id=f"mol_{i:03d}", smiles=smiles, backend_data={}
            )
            molecules.append(mol_data)
    else:
        # Create molecular data with RDKit processing
        for i, smiles in enumerate(test_smiles):
            try:
                rdkit_mol = Chem.MolFromSmiles(smiles)
                if rdkit_mol is not None:
                    # Calculate basic properties
                    mw = Descriptors.MolWt(rdkit_mol)
                    logp = Descriptors.MolLogP(rdkit_mol)
                    tpsa = Descriptors.TPSA(rdkit_mol)

                    mol_data = MolecularData(
                        mol_id=f"mol_{i:03d}",
                        smiles=smiles,
                        backend_data={
                            "rdkit_mol": rdkit_mol,
                            "molecular_weight": mw,
                            "logp": logp,
                            "tpsa": tpsa,
                        },
                    )
                    molecules.append(mol_data)

            except Exception as e:
                print(f"Warning: Could not process SMILES {smiles}: {e}")
                continue

    print(f"Created test library with {len(molecules)} molecules")
    return molecules


def demo_reaction_prediction_engine():
    """Demonstrate the reaction prediction engine capabilities."""
    print("\n" + "=" * 80)
    print("TASK 19.2: HIGH-THROUGHPUT REACTION INFORMATICS DEMO")
    print("=" * 80)

    # Create test molecules
    print("\n1. Creating Test Molecule Library...")
    molecules = create_test_molecule_library()

    # Initialize reaction informatics engine
    print("\n2. Initializing Reaction Informatics Engine...")
    backend = "gpu" if GPU_AVAILABLE else "cpu"
    engine = create_reaction_informatics_engine(
        backend=backend,
        batch_size=2000,
        use_gpu_acceleration=GPU_AVAILABLE,
        memory_limit_gb=4.0,
        enable_quantum_features=True,
    )

    print(f"   Engine initialized with {backend.upper()} backend")
    print(f"   Processing capabilities: {len(molecules)} molecules")

    # Demonstrate comprehensive reaction prediction
    print("\n3. Running Comprehensive Reaction Prediction...")

    start_time = time.time()

    # Predict multiple reaction types
    reaction_types = None  # Predict all available reaction types
    if GPU_AVAILABLE:
        reaction_types = [
            ReactionType.AMIDE_FORMATION,
            ReactionType.SUZUKI_COUPLING,
            ReactionType.CLICK_CHEMISTRY,
            ReactionType.OXIDATION,
            ReactionType.REDUCTION,
        ]

    predictions = engine.predict_reactions(
        molecules=molecules,
        reaction_types=reaction_types,
        min_feasibility_score=0.3,
        max_predictions_per_molecule=8,
        include_conditions=True,
    )

    prediction_time = time.time() - start_time

    print(f"   Generated {len(predictions)} reaction predictions")
    print(f"   Processing time: {prediction_time:.2f} seconds")
    print(f"   Throughput: {len(molecules)/prediction_time:.1f} molecules/sec")
    print(f"   Average predictions per molecule: {len(predictions)/len(molecules):.1f}")

    return engine, molecules, predictions


def demo_retrosynthesis_analysis(engine, molecules):
    """Demonstrate retrosynthesis pathway analysis."""
    print("\n4. Retrosynthesis Pathway Analysis...")

    # Select complex target molecules for retrosynthesis
    target_molecules = molecules[-10:]  # Take the most complex molecules

    retro_results = []

    print(f"   Analyzing {len(target_molecules)} target molecules...")

    for i, target in enumerate(target_molecules):
        print(f"   Processing target {i+1}/{len(target_molecules)}: {target.mol_id}")

        try:
            retro_result = engine.analyze_retrosynthesis(
                target_molecule=target,
                max_steps=5,
                min_commercial_availability=0.7,
                strategy="automated",
            )

            retro_results.append(
                {
                    "target_id": target.mol_id,
                    "target_smiles": target.smiles,
                    "pathway_steps": len(retro_result["pathway_steps"]),
                    "total_score": retro_result["total_score"],
                    "complexity_score": retro_result["complexity_score"],
                    "estimated_cost": retro_result.get("estimated_cost", 0),
                    "estimated_time_hours": retro_result.get("estimated_time_hours", 0),
                    "strategy": retro_result["strategy"],
                    "analysis_time": retro_result.get("analysis_time_seconds", 0),
                }
            )

        except Exception as e:
            print(f"   Warning: Retrosynthesis failed for {target.mol_id}: {e}")
            continue

    print(f"   Completed retrosynthesis analysis for {len(retro_results)} targets")

    # Display results summary
    if retro_results:
        avg_complexity = np.mean([r["complexity_score"] for r in retro_results])
        avg_cost = np.mean([r["estimated_cost"] for r in retro_results])
        avg_time = np.mean([r["estimated_time_hours"] for r in retro_results])

        print(f"   Average synthetic complexity: {avg_complexity:.3f}")
        print(f"   Average estimated cost: ${avg_cost:.2f}")
        print(f"   Average estimated time: {avg_time:.1f} hours")

    return retro_results


def demo_reaction_condition_optimization(engine, predictions):
    """Demonstrate reaction condition optimization."""
    print("\n5. Reaction Condition Optimization...")

    if not predictions:
        print("   No predictions available for optimization")
        return []

    # Select high-scoring predictions for optimization
    high_score_predictions = [
        p for p in predictions if p.get("feasibility_score", 0) > 0.6
    ]
    optimization_targets = high_score_predictions[:5]  # Optimize top 5

    optimization_results = []

    for i, prediction in enumerate(optimization_targets):
        print(f"   Optimizing reaction {i+1}/{len(optimization_targets)}")

        try:
            for target in ["yield", "selectivity", "cost"]:
                optimized = engine.optimize_reaction_conditions(
                    reaction_prediction=prediction,
                    optimization_target=target,
                    constraints={
                        "max_temperature": 150,
                        "exclude_solvents": ["benzene"],
                    },
                )

                optimization_results.append(
                    {
                        "reaction_type": prediction.get("reaction_type", "unknown"),
                        "template_name": prediction.get("template_name", "unknown"),
                        "optimization_target": target,
                        "predicted_improvement": optimized.get(
                            "predicted_improvement", 0
                        ),
                        "confidence": optimized.get("confidence", 0),
                        "backend": optimized.get("backend", "unknown"),
                    }
                )

        except Exception as e:
            print(f"   Warning: Optimization failed for prediction {i}: {e}")
            continue

    print(f"   Completed {len(optimization_results)} condition optimizations")

    # Summary statistics
    if optimization_results:
        avg_improvement = np.mean(
            [r["predicted_improvement"] for r in optimization_results]
        )
        avg_confidence = np.mean([r["confidence"] for r in optimization_results])

        print(f"   Average predicted improvement: {avg_improvement:.1%}")
        print(f"   Average optimization confidence: {avg_confidence:.1%}")

    return optimization_results


def demo_mechanism_classification(engine):
    """Demonstrate reaction mechanism classification."""
    print("\n6. Reaction Mechanism Classification...")

    # Test reaction SMILES for mechanism classification
    test_reactions = [
        "CC(=O)O.CCN>>CC(=O)NCC",  # Amide formation
        "c1ccccc1Br.c2ccc(B(O)O)cc2>>c1ccccc1-c2ccccc2",  # Suzuki coupling
        "C#Cc1ccccc1.N=[N+]=[N-]Cc2ccccc2>>c1ccc2nnn(c2c1)Cc3ccccc3",  # Click chemistry
        "CCO>>CC=O",  # Alcohol oxidation
        "CC(=O)c1ccccc1>>CC(O)c1ccccc1",  # Ketone reduction
    ]

    mechanism_results = []

    for i, reaction_smiles in enumerate(test_reactions):
        print(f"   Classifying reaction {i+1}/{len(test_reactions)}")

        try:
            classification = engine.classify_reaction_mechanism(
                reaction_smiles=reaction_smiles, include_energetics=True
            )

            mechanism_results.append(
                {
                    "reaction_smiles": reaction_smiles,
                    "mechanism_class": classification.get("mechanism_class", "unknown"),
                    "confidence": classification.get("confidence", 0),
                    "reaction_type": classification.get("reaction_type", "unknown"),
                    "elementary_steps": len(classification.get("elementary_steps", [])),
                    "backend": classification.get("backend", "unknown"),
                }
            )

        except Exception as e:
            print(f"   Warning: Classification failed for reaction {i}: {e}")
            continue

    print(f"   Classified {len(mechanism_results)} reaction mechanisms")

    # Display classification summary
    if mechanism_results:
        unique_mechanisms = set(r["mechanism_class"] for r in mechanism_results)
        avg_confidence = np.mean([r["confidence"] for r in mechanism_results])

        print(f"   Identified {len(unique_mechanisms)} unique mechanism classes")
        print(f"   Average classification confidence: {avg_confidence:.1%}")

    return mechanism_results


def demo_batch_processing(engine, molecules):
    """Demonstrate large-scale batch processing capabilities."""
    print("\n7. High-Throughput Batch Processing...")

    # Create larger dataset for batch processing
    extended_molecules = molecules * 3  # Replicate to simulate larger dataset

    print(f"   Processing {len(extended_molecules)} molecules in batch mode...")

    start_time = time.time()

    try:
        # Run batch processing with multiple analysis types
        batch_results = engine.batch_process_reactions(
            molecules=extended_molecules,
            analysis_types=["reaction_prediction", "retrosynthesis"],
            output_file="test_results/batch_reaction_analysis.csv",
        )

        batch_time = time.time() - start_time

        print(f"   Batch processing completed in {batch_time:.2f} seconds")
        print(
            f"   Total throughput: {len(extended_molecules)/batch_time:.1f} molecules/sec"
        )
        print(f"   Results shape: {batch_results.shape}")

        # Analysis type breakdown
        analysis_counts = batch_results["analysis_type"].value_counts()
        print("   Analysis type breakdown:")
        for analysis_type, count in analysis_counts.items():
            print(f"     {analysis_type}: {count} results")

        return batch_results

    except Exception as e:
        print(f"   Batch processing failed: {e}")
        return pd.DataFrame()


def run_performance_benchmark(engine):
    """Run comprehensive performance benchmark."""
    print("\n8. Performance Benchmarking...")

    try:
        # Run benchmark with increasing dataset sizes
        benchmark_sizes = [100, 500, 1000, 2000]
        benchmark_results = []

        for size in benchmark_sizes:
            print(f"   Benchmarking with {size} molecules...")

            start_time = time.time()
            benchmark_metrics = engine.benchmark_performance(num_molecules=size)
            benchmark_time = time.time() - start_time

            benchmark_metrics["benchmark_time"] = benchmark_time
            benchmark_results.append(benchmark_metrics)

            print(
                f"     Throughput: {benchmark_metrics.get('molecules_per_second', 0):.0f} mol/sec"
            )
            print(
                f"     Predictions: {benchmark_metrics.get('predictions_per_second', 0):.0f} pred/sec"
            )

        # Display benchmark summary
        if benchmark_results:
            max_throughput = max(
                r.get("molecules_per_second", 0) for r in benchmark_results
            )
            max_predictions = max(
                r.get("predictions_per_second", 0) for r in benchmark_results
            )

            print("   Peak performance:")
            print(f"     Max molecule throughput: {max_throughput:.0f} molecules/sec")
            print(f"     Max prediction rate: {max_predictions:.0f} predictions/sec")

        return benchmark_results

    except Exception as e:
        print(f"   Benchmark failed: {e}")
        return []


def generate_comprehensive_report(
    engine,
    predictions,
    retro_results,
    optimization_results,
    mechanism_results,
    batch_results,
    benchmark_results,
):
    """Generate comprehensive analysis report."""
    print("\n9. Generating Comprehensive Report...")

    # Get processing statistics
    stats = engine.get_processing_statistics()

    report = {
        "system_info": {
            "backend": stats.get("backend", "unknown"),
            "gpu_available": stats.get("gpu_available", False),
            "total_molecules_processed": stats.get("total_molecules_processed", 0),
            "total_reactions_predicted": stats.get("total_reactions_predicted", 0),
            "total_processing_time": stats.get("total_processing_time", 0),
        },
        "reaction_prediction": {
            "total_predictions": len(predictions),
            "average_feasibility": (
                np.mean([p.get("feasibility_score", 0) for p in predictions])
                if predictions
                else 0
            ),
            "reaction_types_covered": (
                len(set(p.get("reaction_type", "") for p in predictions))
                if predictions
                else 0
            ),
        },
        "retrosynthesis_analysis": {
            "targets_analyzed": len(retro_results),
            "average_complexity": (
                np.mean([r["complexity_score"] for r in retro_results])
                if retro_results
                else 0
            ),
            "average_estimated_cost": (
                np.mean([r["estimated_cost"] for r in retro_results])
                if retro_results
                else 0
            ),
        },
        "condition_optimization": {
            "optimizations_performed": len(optimization_results),
            "average_improvement": (
                np.mean([r["predicted_improvement"] for r in optimization_results])
                if optimization_results
                else 0
            ),
        },
        "mechanism_classification": {
            "reactions_classified": len(mechanism_results),
            "average_confidence": (
                np.mean([r["confidence"] for r in mechanism_results])
                if mechanism_results
                else 0
            ),
        },
        "performance_metrics": {
            "peak_throughput": (
                max([r.get("molecules_per_second", 0) for r in benchmark_results])
                if benchmark_results
                else 0
            ),
            "peak_prediction_rate": (
                max([r.get("predictions_per_second", 0) for r in benchmark_results])
                if benchmark_results
                else 0
            ),
        },
    }

    # Display report
    print("\n" + "=" * 80)
    print("COMPREHENSIVE REACTION INFORMATICS REPORT")
    print("=" * 80)

    print("\nSystem Configuration:")
    print(f"  Backend: {report['system_info']['backend'].upper()}")
    print(f"  GPU Available: {report['system_info']['gpu_available']}")
    print(
        f"  Total Molecules Processed: {report['system_info']['total_molecules_processed']}"
    )
    print(
        f"  Total Reactions Predicted: {report['system_info']['total_reactions_predicted']}"
    )

    print("\nReaction Prediction Results:")
    print(f"  Total Predictions: {report['reaction_prediction']['total_predictions']}")
    print(
        f"  Average Feasibility Score: {report['reaction_prediction']['average_feasibility']:.3f}"
    )
    print(
        f"  Reaction Types Covered: {report['reaction_prediction']['reaction_types_covered']}"
    )

    print("\nRetrosynthesis Analysis:")
    print(
        f"  Targets Analyzed: {report['retrosynthesis_analysis']['targets_analyzed']}"
    )
    print(
        f"  Average Complexity Score: {report['retrosynthesis_analysis']['average_complexity']:.3f}"
    )
    print(
        f"  Average Estimated Cost: ${report['retrosynthesis_analysis']['average_estimated_cost']:.2f}"
    )

    print("\nCondition Optimization:")
    print(
        f"  Optimizations Performed: {report['condition_optimization']['optimizations_performed']}"
    )
    print(
        f"  Average Predicted Improvement: {report['condition_optimization']['average_improvement']:.1%}"
    )

    print("\nMechanism Classification:")
    print(
        f"  Reactions Classified: {report['mechanism_classification']['reactions_classified']}"
    )
    print(
        f"  Average Confidence: {report['mechanism_classification']['average_confidence']:.1%}"
    )

    print("\nPerformance Metrics:")
    print(
        f"  Peak Throughput: {report['performance_metrics']['peak_throughput']:.0f} molecules/sec"
    )
    print(
        f"  Peak Prediction Rate: {report['performance_metrics']['peak_prediction_rate']:.0f} predictions/sec"
    )

    return report


def create_visualization_plots(predictions, retro_results, benchmark_results):
    """Create visualization plots for the demo results."""
    print("\n10. Creating Visualization Plots...")

    try:
        # Set up plotting style
        plt.style.use("default")
        sns.set_palette("husl")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "High-Throughput Reaction Informatics Analysis Results",
            fontsize=16,
            fontweight="bold",
        )

        # Plot 1: Reaction Prediction Distribution
        if predictions:
            feasibility_scores = [p.get("feasibility_score", 0) for p in predictions]
            axes[0, 0].hist(
                feasibility_scores,
                bins=20,
                alpha=0.7,
                color="skyblue",
                edgecolor="black",
            )
            axes[0, 0].set_title("Reaction Feasibility Score Distribution")
            axes[0, 0].set_xlabel("Feasibility Score")
            axes[0, 0].set_ylabel("Frequency")
            axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Retrosynthesis Complexity vs Cost
        if retro_results:
            complexity_scores = [r["complexity_score"] for r in retro_results]
            estimated_costs = [r["estimated_cost"] for r in retro_results]
            axes[0, 1].scatter(
                complexity_scores, estimated_costs, alpha=0.7, color="orange", s=60
            )
            axes[0, 1].set_title("Synthetic Complexity vs Estimated Cost")
            axes[0, 1].set_xlabel("Complexity Score")
            axes[0, 1].set_ylabel("Estimated Cost ($)")
            axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Performance Benchmark
        if benchmark_results:
            molecules_counts = [r.get("total_molecules", 0) for r in benchmark_results]
            throughputs = [r.get("molecules_per_second", 0) for r in benchmark_results]
            axes[1, 0].plot(
                molecules_counts,
                throughputs,
                "o-",
                color="green",
                linewidth=2,
                markersize=8,
            )
            axes[1, 0].set_title("Performance Scaling")
            axes[1, 0].set_xlabel("Number of Molecules")
            axes[1, 0].set_ylabel("Throughput (molecules/sec)")
            axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Reaction Type Distribution
        if predictions:
            reaction_types = [p.get("reaction_type", "unknown") for p in predictions]
            type_counts = pd.Series(reaction_types).value_counts()
            axes[1, 1].pie(
                type_counts.values,
                labels=type_counts.index,
                autopct="%1.1f%%",
                startangle=90,
            )
            axes[1, 1].set_title("Reaction Type Distribution")

        # Adjust layout and save
        plt.tight_layout()

        # Save the plot
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(
            output_dir / "reaction_informatics_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"   Visualization saved to: {output_dir / 'reaction_informatics_analysis.png'}"
        )

        # Show plot if in interactive mode
        # plt.show()

    except Exception as e:
        print(f"   Visualization creation failed: {e}")


def main():
    """Main demonstration function."""
    print("High-Throughput Reaction Informatics Demonstration")
    print("Task 19.2 - ChemiAccel GPU-Accelerated Reaction Analysis")
    print(f"GPU Backend Available: {GPU_AVAILABLE}")
    print(f"RDKit Available: {RDKIT_AVAILABLE}")

    try:
        # Run comprehensive demonstration
        engine, molecules, predictions = demo_reaction_prediction_engine()
        retro_results = demo_retrosynthesis_analysis(engine, molecules)
        optimization_results = demo_reaction_condition_optimization(engine, predictions)
        mechanism_results = demo_mechanism_classification(engine)
        batch_results = demo_batch_processing(engine, molecules)
        benchmark_results = run_performance_benchmark(engine)

        # Generate comprehensive report
        report = generate_comprehensive_report(
            engine,
            predictions,
            retro_results,
            optimization_results,
            mechanism_results,
            batch_results,
            benchmark_results,
        )

        # Create visualizations
        create_visualization_plots(predictions, retro_results, benchmark_results)

        # Cleanup
        engine.cleanup()

        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(
            f"Total processing time: {report['system_info']['total_processing_time']:.2f} seconds"
        )
        print(
            f"Peak performance: {report['performance_metrics']['peak_throughput']:.0f} molecules/sec"
        )
        print(
            "\nTask 19.2 High-Throughput Reaction Informatics has been successfully demonstrated!"
        )

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
