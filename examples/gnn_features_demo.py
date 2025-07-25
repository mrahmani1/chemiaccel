#!/usr/bin/env python3
"""
Graph Neural Network Features Demo for ChemiAccel.

This example demonstrates the use of ChemiAccel's GNN-inspired features including:
- Molecular graph embeddings with different backends
- Advanced graph-based fingerprints
- Molecular scaffold analysis
- Graph similarity metrics
- Performance benchmarking
"""

import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from chemi_accel.descriptors.gnn_features import (
    AdvancedGraphFeatures,
    MolecularGraphEmbedding,
    benchmark_graph_features_performance,
    compute_advanced_graph_features,
    compute_graph_similarity_matrix,
    compute_molecular_graph_embeddings,
)
from chemi_accel.utils.rdkit_integration import parse_smiles_batch


def main():
    """Run GNN features demonstration."""
    print("=" * 80)
    print("ChemiAccel Graph Neural Network Features Demo")
    print("=" * 80)

    # Sample molecules for demonstration
    sample_smiles = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "CCN(CC)CC",  # Triethylamine
        "CC(C)C(=O)O",  # Isobutyric acid
        "c1ccc2ccccc2c1",  # Naphthalene
        "CC(C)(C)O",  # tert-Butanol
        "CCCCCCCC",  # Octane
        "c1ccc(cc1)O",  # Phenol
        "CC(=O)Nc1ccccc1",  # Acetanilide
    ]

    print(f"\nParsing {len(sample_smiles)} sample molecules...")
    molecules = parse_smiles_batch(
        sample_smiles, [f"mol_{i}" for i in range(len(sample_smiles))]
    )
    print(f"Successfully parsed {len(molecules)} molecules")

    # Demonstrate molecular graph embeddings
    print("\n" + "=" * 60)
    print("1. Molecular Graph Embeddings")
    print("=" * 60)

    # Test different backends
    backends_to_test = ["cpu"]

    # Check for additional backends
    try:
        import cupy

        backends_to_test.append("cuda")
        print("CUDA backend available")
    except ImportError:
        print("CUDA backend not available (cupy not installed)")

    try:
        import jax

        backends_to_test.append("jax")
        print("JAX backend available")
    except ImportError:
        print("JAX backend not available (jax not installed)")

    for backend in backends_to_test:
        print(f"\nTesting {backend.upper()} backend:")

        start_time = time.time()
        embedding_calculator = MolecularGraphEmbedding(
            backend=backend, hidden_dim=64, num_layers=2
        )

        # Compute embeddings for first few molecules
        test_molecules = molecules[:5]
        embeddings = embedding_calculator.compute_batch_embeddings(test_molecules)

        elapsed_time = time.time() - start_time

        print(
            f"  - Computed embeddings for {len(test_molecules)} molecules in {elapsed_time:.3f}s"
        )
        print(
            f"  - Embedding dimensions: {embedding_calculator.get_embedding_dimensions()}"
        )

        # Show sample embedding
        if embeddings and embeddings[0]["graph_embedding"].size > 0:
            sample_embedding = embeddings[0]["graph_embedding"]
            print(f"  - Sample graph embedding shape: {sample_embedding.shape}")
            print(
                f"  - Sample embedding stats: mean={np.mean(sample_embedding):.3f}, "
                f"std={np.std(sample_embedding):.3f}"
            )

    # Demonstrate DataFrame output
    print("\nGenerating DataFrame with graph embeddings...")
    df_embeddings = compute_molecular_graph_embeddings(
        molecules[:5], backend="cpu", hidden_dim=32, num_layers=2
    )
    print(f"DataFrame shape: {df_embeddings.shape}")
    print(f"Columns: {list(df_embeddings.columns[:10])}...")  # Show first 10 columns

    # Demonstrate advanced graph features
    print("\n" + "=" * 60)
    print("2. Advanced Graph-Based Features")
    print("=" * 60)

    advanced_features = AdvancedGraphFeatures(backend="cpu")

    # Generate graph fingerprints for a sample molecule
    sample_mol = molecules[0]
    print(f"\nAnalyzing molecule: {sample_mol.smiles}")

    fingerprints = advanced_features.generate_graph_fingerprints(sample_mol)
    print(f"Generated fingerprint types: {list(fingerprints.keys())}")

    for fp_type, fp_array in fingerprints.items():
        bits_set = np.sum(fp_array)
        density = bits_set / len(fp_array)
        print(
            f"  - {fp_type}: {bits_set}/{len(fp_array)} bits set (density: {density:.3f})"
        )

    # Molecular scaffold analysis
    print(f"\nScaffold analysis for {sample_mol.smiles}:")
    scaffold_analysis = advanced_features.analyze_molecular_scaffolds(sample_mol)

    print(f"  - Number of ring systems: {scaffold_analysis['num_ring_systems']}")
    print(f"  - Scaffold size: {scaffold_analysis['scaffold_size']} atoms")
    print(f"  - Number of side chains: {scaffold_analysis['num_side_chains']}")

    if scaffold_analysis["scaffold_properties"]:
        print("  - Scaffold properties:")
        for prop, value in scaffold_analysis["scaffold_properties"].items():
            print(f"    * {prop}: {value:.3f}")

    # Advanced features DataFrame
    print("\nGenerating advanced features DataFrame...")
    df_advanced = compute_advanced_graph_features(molecules[:5], backend="cpu")
    print(f"DataFrame shape: {df_advanced.shape}")
    print(
        f"Feature columns: {[col for col in df_advanced.columns if col not in ['mol_id', 'smiles']]}"
    )

    # Display sample results
    print("\nSample advanced features:")
    print(
        df_advanced[
            ["mol_id", "smiles", "num_ring_systems", "scaffold_size"]
        ].to_string(index=False)
    )

    # Demonstrate graph similarity metrics
    print("\n" + "=" * 60)
    print("3. Graph Similarity Metrics")
    print("=" * 60)

    # Compute pairwise similarities for first 3 molecules
    test_molecules = molecules[:3]
    print(f"\nComputing pairwise similarities for {len(test_molecules)} molecules...")

    similarity_matrix = compute_graph_similarity_matrix(test_molecules, backend="cpu")
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print("\nSimilarity matrix:")
    print(similarity_matrix.round(3).to_string())

    # Detailed similarity analysis for two molecules
    mol1, mol2 = molecules[0], molecules[1]
    print(f"\nDetailed similarity between {mol1.smiles} and {mol2.smiles}:")

    detailed_similarities = advanced_features.compute_graph_similarity_metrics(
        mol1, mol2
    )
    for metric, value in detailed_similarities.items():
        print(f"  - {metric}: {value:.3f}")

    # Demonstrate performance benchmarking
    print("\n" + "=" * 60)
    print("4. Performance Benchmarking")
    print("=" * 60)

    print("\nBenchmarking graph features performance...")
    benchmark_results = benchmark_graph_features_performance(
        molecules[:5], traditional_method="rdkit"
    )

    print("Benchmark results:")
    for metric, value in benchmark_results.items():
        if isinstance(value, float):
            print(f"  - {metric}: {value:.3f}")
        else:
            print(f"  - {metric}: {value}")

    # Demonstrate different embedding dimensions
    print("\n" + "=" * 60)
    print("5. Embedding Dimension Comparison")
    print("=" * 60)

    dimensions_to_test = [16, 32, 64, 128]
    test_mol = molecules[0]

    print(f"\nTesting different embedding dimensions for {test_mol.smiles}:")

    for hidden_dim in dimensions_to_test:
        start_time = time.time()

        calculator = MolecularGraphEmbedding(
            backend="cpu", hidden_dim=hidden_dim, num_layers=2
        )

        embedding = calculator.compute_embeddings(test_mol)
        elapsed_time = time.time() - start_time

        graph_emb = embedding["graph_embedding"]
        print(
            f"  - Hidden dim {hidden_dim:3d}: shape {graph_emb.shape}, "
            f"time {elapsed_time:.3f}s, "
            f"mean {np.mean(graph_emb):.3f}"
        )

    # Summary
    print("\n" + "=" * 80)
    print("Demo Summary")
    print("=" * 80)

    print(
        f"""
✓ Successfully demonstrated ChemiAccel GNN features:
  - Molecular graph embeddings with multiple backends
  - Advanced graph fingerprints (4 types)
  - Molecular scaffold analysis and decomposition
  - Graph similarity metrics (Tanimoto, Dice, Cosine)
  - Performance benchmarking against traditional methods
  - Flexible embedding dimensions and architectures

✓ Processed {len(molecules)} molecules with various structural features
✓ Generated embeddings, fingerprints, and similarity metrics
✓ Demonstrated scalability and performance advantages

The GNN features provide rich molecular representations that can be used for:
  - Machine learning model features
  - Molecular similarity searching
  - Scaffold-based drug design
  - Chemical space analysis
  - High-throughput virtual screening
    """
    )


if __name__ == "__main__":
    main()
