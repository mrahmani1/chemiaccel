#!/usr/bin/env python3
"""
ChemiAccel Workflow Integration Demo

This script demonstrates various integration workflows including:
- Basic descriptor calculation
- Scikit-learn pipeline integration
- Jupyter notebook features (when available)
- Dask distributed processing (when available)
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Import ChemiAccel
import chemi_accel as ca

# Optional imports
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not available - skipping ML demos")

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available - skipping plots")


def demo_basic_workflow():
    """Demonstrate basic ChemiAccel workflow."""
    print("\n" + "=" * 60)
    print("🧪 BASIC WORKFLOW DEMO")
    print("=" * 60)

    # Sample molecules
    molecules = [
        "CCO",  # ethanol
        "c1ccccc1",  # benzene
        "CC(=O)O",  # acetic acid
        "CCN(CC)CC",  # triethylamine
        "c1ccc2c(c1)ccccc2",  # naphthalene
        "CC(C)(C)O",  # tert-butanol
        "c1ccc(cc1)O",  # phenol
        "c1ccc(cc1)N",  # aniline
        "CC(C)CC(C)(C)C",  # branched alkane
        "c1ccncc1",  # pyridine
    ]

    print(f"📋 Processing {len(molecules)} molecules...")

    # Initialize calculator
    calc = ca.Calculator()
    print(f"🔧 Calculator initialized: {calc}")

    # Configure descriptors
    calc.configure_descriptors("basic")
    print(f"📊 Configured {len(calc.descriptor_sets)} descriptors")

    # Calculate descriptors
    results = calc.pandas(molecules)
    print(f"✅ Calculated descriptors for {len(results)} molecules")

    # Display results summary
    print("\n📈 Results Summary:")
    print(results.describe().round(3))

    # Show first few molecules
    print("\n🔍 First 5 molecules:")
    display_cols = ["smiles", "molecular_weight", "logp", "tpsa"]
    available_cols = [col for col in display_cols if col in results.columns]
    print(results[available_cols].head())

    return results


def demo_sklearn_integration():
    """Demonstrate scikit-learn integration."""
    if not SKLEARN_AVAILABLE:
        print("\n⚠️  Skipping scikit-learn demo - not available")
        return

    print("\n" + "=" * 60)
    print("🤖 SCIKIT-LEARN INTEGRATION DEMO")
    print("=" * 60)

    # Sample data with synthetic targets (e.g., boiling points)
    molecules = [
        "CCO",
        "c1ccccc1",
        "CC(=O)O",
        "CCN(CC)CC",
        "c1ccc2c(c1)ccccc2",
        "CC(C)(C)O",
        "c1ccc(cc1)O",
        "c1ccc(cc1)N",
        "CC(C)CC(C)(C)C",
        "c1ccncc1",
        "CCC",
        "CCCC",
        "CCCCC",
        "c1ccc(cc1)Cl",
        "CC(C)O",
    ]

    # Synthetic boiling points (for demo purposes)
    targets = [
        78.4,
        80.1,
        118.1,
        89.5,
        218.0,
        82.4,
        181.7,
        184.1,
        174.1,
        115.2,
        -42.1,
        -0.5,
        36.1,
        131.7,
        82.6,
    ]

    print(f"📋 Training data: {len(molecules)} molecules with target values")

    # Method 1: Using ChemiAccelTransformer
    print("\n🔧 Method 1: ChemiAccelTransformer")
    transformer = ca.ChemiAccelTransformer(descriptor_set="basic")
    X_descriptors = transformer.fit_transform(molecules)
    print(f"✅ Transformed to {X_descriptors.shape[1]} descriptors")

    # Method 2: Using Pipeline
    print("\n🔧 Method 2: Pipeline Integration")
    pipeline = ca.create_molecular_pipeline(
        RandomForestRegressor(n_estimators=50, random_state=42), descriptor_set="basic"
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        molecules, targets, test_size=0.3, random_state=42
    )

    # Fit pipeline
    pipeline.fit(X_train, y_train)
    print(f"✅ Pipeline fitted on {len(X_train)} training molecules")

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Model Performance:")
    print(f"   MSE: {mse:.2f}")
    print(f"   R²:  {r2:.3f}")

    # Show predictions vs actual
    print("\n🎯 Predictions vs Actual:")
    for i, (actual, pred) in enumerate(zip(y_test, y_pred)):
        print(
            f"   {X_test[i][:15]:15s} | Actual: {actual:6.1f} | Predicted: {pred:6.1f}"
        )

    # Method 3: Using ChemiAccelEstimator
    print("\n🔧 Method 3: ChemiAccelEstimator")
    estimator = ca.ChemiAccelEstimator(
        predictor=RandomForestRegressor(n_estimators=50, random_state=42),
        descriptor_set="basic",
    )

    estimator.fit(X_train, y_train)
    score = estimator.score(X_test, y_test)
    print(f"✅ Estimator R² score: {score:.3f}")


def demo_performance_benchmarking():
    """Demonstrate performance benchmarking."""
    print("\n" + "=" * 60)
    print("⚡ PERFORMANCE BENCHMARKING DEMO")
    print("=" * 60)

    calc = ca.Calculator()
    calc.configure_descriptors("basic")

    print("🏃 Running benchmark...")
    results = calc.benchmark_performance(num_molecules=1000, num_runs=3)

    print("\n📊 Benchmark Results:")
    print(f"   Backend: {results['backend_type']}")
    print(f"   Molecules: {results['num_molecules']}")
    print(f"   Descriptors: {results['num_descriptors']}")
    print(f"   Average time: {results['avg_time_seconds']:.3f} seconds")
    print(f"   Throughput: {results['molecules_per_second']:.1f} molecules/sec")
    print(
        f"   Descriptor throughput: {results['descriptors_per_second']:.1f} descriptors/sec"
    )


def demo_dask_integration():
    """Demonstrate Dask distributed processing."""
    try:
        import dask
        from dask.distributed import Client
    except ImportError:
        print("\n⚠️  Skipping Dask demo - not available")
        return

    print("\n" + "=" * 60)
    print("🌐 DASK DISTRIBUTED PROCESSING DEMO")
    print("=" * 60)

    # Create larger dataset for distributed processing
    base_molecules = [
        "CCO",
        "c1ccccc1",
        "CC(=O)O",
        "CCN(CC)CC",
        "c1ccc2c(c1)ccccc2",
        "CC(C)(C)O",
        "c1ccc(cc1)O",
        "c1ccc(cc1)N",
        "CC(C)CC(C)(C)C",
        "c1ccncc1",
    ]

    # Replicate to create larger dataset
    large_molecules = base_molecules * 100  # 1000 molecules
    print(f"📋 Processing {len(large_molecules)} molecules with Dask...")

    try:
        # Use distributed processing
        results = ca.distributed_calculate_descriptors(
            large_molecules,
            backend="cpu",  # Use CPU for demo
            descriptor_set="basic",
            chunk_size=200,
        )

        print(f"✅ Processed {len(results)} molecules")
        print(f"📊 Result shape: {results.shape}")
        print("🔍 Sample results:")
        print(results.head(3))

    except Exception as e:
        print(f"⚠️  Dask processing failed: {e}")


def demo_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n" + "=" * 60)
    print("🛡️  ERROR HANDLING DEMO")
    print("=" * 60)

    # Mix of valid and invalid molecules
    mixed_molecules = [
        "CCO",  # valid
        "invalid_smiles_123",  # invalid
        "c1ccccc1",  # valid
        "",  # invalid (empty)
        "CC(=O)O",  # valid
        "definitely_not_a_smiles",  # invalid
        "c1ccncc1",  # valid
    ]

    print(f"📋 Testing with {len(mixed_molecules)} molecules (some invalid)...")

    calc = ca.Calculator()
    calc.configure_descriptors("basic")

    # Test validation
    print("\n🔍 Validation Results:")
    valid_smiles, errors = calc.validate_molecules(mixed_molecules)
    print(f"   Valid molecules: {len(valid_smiles)}")
    print(f"   Invalid molecules: {len(errors)}")

    for error in errors:
        print(
            f"   ❌ Index {error['index']}: {error['smiles'][:20]} - {error['reason']}"
        )

    # Test calculation (should handle invalid molecules gracefully)
    print("\n⚡ Calculating descriptors (skipping invalid)...")
    results = calc.pandas(mixed_molecules)
    print(f"✅ Successfully processed {len(results)} valid molecules")


def demo_backend_comparison():
    """Demonstrate backend comparison."""
    print("\n" + "=" * 60)
    print("🔄 BACKEND COMPARISON DEMO")
    print("=" * 60)

    molecules = [
        "CCO",
        "c1ccccc1",
        "CC(=O)O",
        "CCN(CC)CC",
        "c1ccncc1",
    ] * 20  # 100 molecules

    # Get available backends
    available_backends = ca.get_available_backends()
    print(f"🔧 Available backends: {[b.value for b in available_backends]}")

    results = {}

    for backend in available_backends:
        try:
            print(f"\n⚡ Testing {backend.value} backend...")
            calc = ca.Calculator(backend=backend.value)
            calc.configure_descriptors("basic")

            # Run benchmark
            benchmark = calc.benchmark_performance(test_molecules=molecules, num_runs=2)

            results[backend.value] = benchmark
            print(
                f"   Throughput: {benchmark['molecules_per_second']:.1f} molecules/sec"
            )

        except Exception as e:
            print(f"   ❌ Failed: {e}")

    if len(results) > 1:
        print("\n📊 Backend Comparison:")
        for backend, result in results.items():
            print(
                f"   {backend:8s}: {result['molecules_per_second']:8.1f} molecules/sec"
            )


def main():
    """Run all demos."""
    print("🚀 ChemiAccel Workflow Integration Demo")
    print("=" * 60)

    try:
        # Basic workflow
        demo_basic_workflow()

        # Scikit-learn integration
        demo_sklearn_integration()

        # Performance benchmarking
        demo_performance_benchmarking()

        # Error handling
        demo_error_handling()

        # Backend comparison
        demo_backend_comparison()

        # Dask integration (if available)
        demo_dask_integration()

        print("\n" + "=" * 60)
        print("✅ All demos completed successfully!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
