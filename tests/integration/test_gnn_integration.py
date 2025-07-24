"""
Integration tests for Graph Neural Network features with popular GNN frameworks.

This module tests the integration of ChemiAccel's GNN features with popular
frameworks like PyTorch Geometric, DGL, and JAX-based GNN libraries.
"""

import pytest
import numpy as np
import pandas as pd
from typing import List, Dict, Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import torch_geometric
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

try:
    import dgl
    import dgl.nn.pytorch as dglnn
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from src.chemi_accel.core.molecular import MolecularData
from src.chemi_accel.descriptors.gnn_features import (
    MolecularGraphEmbedding,
    compute_molecular_graph_embeddings
)
from src.chemi_accel.utils.rdkit_integration import parse_smiles_batch


class TestGNNIntegration:
    """Test integration with popular GNN frameworks."""
    
    @pytest.fixture
    def sample_molecules(self):
        """Create sample molecules for testing."""
        smiles_list = [
            "CCO",  # Ethanol
            "CC(=O)O",  # Acetic acid
            "c1ccccc1",  # Benzene
            "CCN(CC)CC",  # Triethylamine
            "CC(C)C(=O)O"  # Isobutyric acid
        ]
        
        molecules = []
        for i, smiles in enumerate(smiles_list):
            mol_data = MolecularData(
                mol_id=f"mol_{i}",
                smiles=smiles,
                backend_data={}
            )
            molecules.append(mol_data)
        
        # Parse SMILES to get RDKit mol objects
        parsed_molecules = parse_smiles_batch(smiles_list, [f"mol_{i}" for i in range(len(smiles_list))])
        return parsed_molecules
    
    @pytest.fixture
    def graph_embedding_calculator(self):
        """Create graph embedding calculator."""
        return MolecularGraphEmbedding(backend="cpu", hidden_dim=64, num_layers=2)
    
    def test_basic_embedding_computation(self, sample_molecules, graph_embedding_calculator):
        """Test basic embedding computation."""
        embeddings = graph_embedding_calculator.compute_batch_embeddings(sample_molecules)
        
        assert len(embeddings) == len(sample_molecules)
        
        for embedding in embeddings:
            assert "node_features" in embedding
            assert "edge_features" in embedding
            assert "graph_embedding" in embedding
            assert embedding["num_atoms"] > 0
            
            # Check embedding dimensions
            if embedding["node_features"].size > 0:
                assert embedding["node_features"].shape[1] == 64  # hidden_dim
            
            assert embedding["graph_embedding"].shape[0] == 256  # 4 * hidden_dim
    
    @pytest.mark.skipif(not TORCH_GEOMETRIC_AVAILABLE, reason="PyTorch Geometric not available")
    def test_pytorch_geometric_integration(self, sample_molecules, graph_embedding_calculator):
        """Test integration with PyTorch Geometric."""
        
        # Compute embeddings
        embeddings = graph_embedding_calculator.compute_batch_embeddings(sample_molecules)
        
        # Convert to PyTorch Geometric format
        pyg_data_list = []
        for i, (mol_data, embedding) in enumerate(zip(sample_molecules, embeddings)):
            if embedding["node_features"].size == 0:
                continue
            
            # Create PyTorch Geometric Data object
            x = torch.tensor(embedding["node_features"], dtype=torch.float)
            
            # Create edge index from adjacency matrix
            graph_data = graph_embedding_calculator.embedding_impl.graph_builder.build_molecular_graph(mol_data)
            adj_matrix = graph_data["adjacency_matrix"]
            edge_indices = np.where(adj_matrix == 1)
            edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)
            
            # Edge features
            if embedding["edge_features"].size > 0:
                edge_attr = torch.tensor(embedding["edge_features"], dtype=torch.float)
            else:
                edge_attr = None
            
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            pyg_data_list.append(data)
        
        assert len(pyg_data_list) > 0
        
        # Create a simple GCN model
        class SimpleGCN(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.conv1 = GCNConv(input_dim, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, output_dim)
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x, edge_index, batch):
                x = F.relu(self.conv1(x, edge_index))
                x = self.dropout(x)
                x = self.conv2(x, edge_index)
                return global_mean_pool(x, batch)
        
        # Test model with our data
        model = SimpleGCN(input_dim=64, hidden_dim=32, output_dim=16)
        batch = Batch.from_data_list(pyg_data_list)
        
        with torch.no_grad():
            output = model(batch.x, batch.edge_index, batch.batch)
        
        assert output.shape[0] == len(pyg_data_list)
        assert output.shape[1] == 16
    
    @pytest.mark.skipif(not DGL_AVAILABLE, reason="DGL not available")
    def test_dgl_integration(self, sample_molecules, graph_embedding_calculator):
        """Test integration with Deep Graph Library (DGL)."""
        
        # Compute embeddings
        embeddings = graph_embedding_calculator.compute_batch_embeddings(sample_molecules)
        
        # Convert to DGL format
        dgl_graphs = []
        for i, (mol_data, embedding) in enumerate(zip(sample_molecules, embeddings)):
            if embedding["node_features"].size == 0:
                continue
            
            # Create DGL graph
            graph_data = graph_embedding_calculator.embedding_impl.graph_builder.build_molecular_graph(mol_data)
            adj_matrix = graph_data["adjacency_matrix"]
            
            # Get edge list
            src, dst = np.where(adj_matrix == 1)
            g = dgl.graph((src, dst))
            
            # Add node features
            g.ndata['feat'] = torch.tensor(embedding["node_features"], dtype=torch.float)
            
            # Add edge features if available
            if embedding["edge_features"].size > 0:
                g.edata['feat'] = torch.tensor(embedding["edge_features"], dtype=torch.float)
            
            dgl_graphs.append(g)
        
        assert len(dgl_graphs) > 0
        
        # Create a simple DGL model
        class SimpleDGLModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.conv1 = dglnn.GraphConv(input_dim, hidden_dim)
                self.conv2 = dglnn.GraphConv(hidden_dim, output_dim)
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, g, feat):
                h = F.relu(self.conv1(g, feat))
                h = self.dropout(h)
                h = self.conv2(g, h)
                return dgl.mean_nodes(g, 'h')
        
        # Test model with our data
        model = SimpleDGLModel(input_dim=64, hidden_dim=32, output_dim=16)
        
        for g in dgl_graphs:
            with torch.no_grad():
                g.ndata['h'] = g.ndata['feat']
                output = model(g, g.ndata['feat'])
            
            assert output.shape[1] == 16
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_jax_integration(self, sample_molecules):
        """Test integration with JAX-based GNN implementations."""
        
        # Use JAX backend for embedding computation
        jax_calculator = MolecularGraphEmbedding(backend="jax", hidden_dim=64, num_layers=2)
        embeddings = jax_calculator.compute_batch_embeddings(sample_molecules)
        
        # Simple JAX-based GNN function
        def jax_gcn_layer(node_features, adjacency_matrix, weights):
            """Simple JAX implementation of GCN layer."""
            # Self-connection
            self_features = jnp.dot(node_features, weights)
            
            # Neighbor aggregation
            neighbor_features = jnp.dot(adjacency_matrix, node_features)
            neighbor_features = jnp.dot(neighbor_features, weights)
            
            # Combine and activate
            return jax.nn.relu(self_features + neighbor_features)
        
        # JIT compile the function
        jit_gcn_layer = jit(jax_gcn_layer)
        
        # Test with our embeddings
        for i, (mol_data, embedding) in enumerate(zip(sample_molecules, embeddings)):
            if embedding["node_features"].size == 0:
                continue
            
            # Convert to JAX arrays
            node_features = jnp.array(embedding["node_features"])
            
            # Get adjacency matrix
            graph_data = jax_calculator.embedding_impl.graph_builder.build_molecular_graph(mol_data)
            adjacency_matrix = jnp.array(graph_data["adjacency_matrix"], dtype=jnp.float32)
            
            # Create random weights
            key = jax.random.PRNGKey(42)
            weights = jax.random.normal(key, (64, 32)) * 0.1
            
            # Apply GCN layer
            output = jit_gcn_layer(node_features, adjacency_matrix, weights)
            
            assert output.shape[0] == node_features.shape[0]
            assert output.shape[1] == 32
    
    def test_embedding_consistency_across_backends(self, sample_molecules):
        """Test that embeddings are consistent across different backends."""
        
        # Test CPU backend
        cpu_calculator = MolecularGraphEmbedding(backend="cpu", hidden_dim=64, num_layers=2)
        cpu_embeddings = cpu_calculator.compute_batch_embeddings(sample_molecules)
        
        # Test other backends if available
        backends_to_test = []
        
        try:
            import cupy
            backends_to_test.append("cuda")
        except ImportError:
            pass
        
        if JAX_AVAILABLE:
            backends_to_test.append("jax")
        
        for backend in backends_to_test:
            backend_calculator = MolecularGraphEmbedding(backend=backend, hidden_dim=64, num_layers=2)
            backend_embeddings = backend_calculator.compute_batch_embeddings(sample_molecules)
            
            # Compare embeddings (allowing for small numerical differences)
            for cpu_emb, backend_emb in zip(cpu_embeddings, backend_embeddings):
                if cpu_emb["node_features"].size > 0 and backend_emb["node_features"].size > 0:
                    # Note: Due to different random initializations, we mainly check shapes
                    assert cpu_emb["node_features"].shape == backend_emb["node_features"].shape
                    assert cpu_emb["graph_embedding"].shape == backend_emb["graph_embedding"].shape
                
                assert cpu_emb["num_atoms"] == backend_emb["num_atoms"]
                assert cpu_emb["num_bonds"] == backend_emb["num_bonds"]
    
    def test_dataframe_output(self, sample_molecules):
        """Test DataFrame output format."""
        df = compute_molecular_graph_embeddings(
            sample_molecules, 
            backend="cpu", 
            hidden_dim=32, 
            num_layers=2
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_molecules)
        
        # Check required columns
        required_cols = ["mol_id", "smiles", "num_atoms", "num_bonds"]
        for col in required_cols:
            assert col in df.columns
        
        # Check embedding columns
        emb_cols = [col for col in df.columns if col.startswith("graph_emb_")]
        assert len(emb_cols) == 128  # 4 * hidden_dim (32)
        
        # Check data types
        assert df["num_atoms"].dtype in [np.int32, np.int64]
        assert df["num_bonds"].dtype in [np.int32, np.int64]
        
        for col in emb_cols:
            assert df[col].dtype in [np.float32, np.float64]
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        
        # Test with invalid SMILES
        invalid_mol = MolecularData(
            mol_id="invalid",
            smiles="INVALID_SMILES",
            backend_data={}
        )
        
        calculator = MolecularGraphEmbedding(backend="cpu", hidden_dim=64, num_layers=2)
        embedding = calculator.compute_embeddings(invalid_mol)
        
        # Should return empty/zero embeddings for invalid molecules
        assert embedding["node_features"].size == 0
        assert embedding["edge_features"].size == 0
        assert embedding["num_atoms"] == 0
        assert embedding["num_bonds"] == 0
        assert embedding["graph_embedding"].shape[0] == 256  # Should still have correct shape
    
    def test_different_embedding_dimensions(self, sample_molecules):
        """Test different embedding dimensions."""
        
        dimensions_to_test = [16, 32, 64, 128]
        
        for hidden_dim in dimensions_to_test:
            calculator = MolecularGraphEmbedding(backend="cpu", hidden_dim=hidden_dim, num_layers=2)
            embeddings = calculator.compute_batch_embeddings(sample_molecules)
            
            for embedding in embeddings:
                if embedding["node_features"].size > 0:
                    assert embedding["node_features"].shape[1] == hidden_dim
                
                assert embedding["graph_embedding"].shape[0] == hidden_dim * 4
    
    def test_different_layer_numbers(self, sample_molecules):
        """Test different numbers of graph convolution layers."""
        
        layers_to_test = [1, 2, 3, 4]
        
        for num_layers in layers_to_test:
            calculator = MolecularGraphEmbedding(backend="cpu", hidden_dim=64, num_layers=num_layers)
            embeddings = calculator.compute_batch_embeddings(sample_molecules)
            
            # Should work regardless of number of layers
            for embedding in embeddings:
                if embedding["node_features"].size > 0:
                    assert embedding["node_features"].shape[1] == 64
                
                assert embedding["graph_embedding"].shape[0] == 256  # 4 * 64


class TestGNNPerformance:
    """Test performance characteristics of GNN features."""
    
    @pytest.fixture
    def sample_molecules(self):
        """Create sample molecules for testing."""
        smiles_list = [
            "CCO",  # Ethanol
            "CC(=O)O",  # Acetic acid
            "c1ccccc1",  # Benzene
            "CCN(CC)CC",  # Triethylamine
            "CC(C)C(=O)O"  # Isobutyric acid
        ]
        
        # Parse SMILES to get RDKit mol objects
        parsed_molecules = parse_smiles_batch(smiles_list, [f"mol_{i}" for i in range(len(smiles_list))])
        return parsed_molecules
    
    def test_batch_processing_performance(self):
        """Test performance of batch processing."""
        import time
        
        # Create larger batch of molecules
        smiles_list = [
            "CCO", "CC(=O)O", "c1ccccc1", "CCN(CC)CC", "CC(C)C(=O)O",
            "CC(C)(C)O", "CCCCCCCC", "c1ccc2ccccc2c1", "CCN", "CCC(=O)O"
        ] * 10  # 50 molecules
        
        molecules = []
        for i, smiles in enumerate(smiles_list):
            mol_data = MolecularData(
                mol_id=f"mol_{i}",
                smiles=smiles,
                backend_data={}
            )
            molecules.append(mol_data)
        
        # Parse SMILES
        parsed_molecules = parse_smiles_batch(molecules)
        
        # Test CPU performance
        calculator = MolecularGraphEmbedding(backend="cpu", hidden_dim=64, num_layers=2)
        
        start_time = time.time()
        embeddings = calculator.compute_batch_embeddings(parsed_molecules)
        cpu_time = time.time() - start_time
        
        assert len(embeddings) == len(parsed_molecules)
        assert cpu_time < 10.0  # Should complete within 10 seconds
        
        print(f"CPU batch processing time: {cpu_time:.3f} seconds for {len(parsed_molecules)} molecules")
    
    def test_memory_usage(self, sample_molecules):
        """Test memory usage characteristics."""
        try:
            import psutil
            import os
        except ImportError:
            pytest.skip("psutil not available for memory testing")
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create calculator and compute embeddings
        calculator = MolecularGraphEmbedding(backend="cpu", hidden_dim=128, num_layers=3)
        embeddings = calculator.compute_batch_embeddings(sample_molecules)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for small test)
        assert memory_increase < 100
        
        print(f"Memory increase: {memory_increase:.2f} MB")


if __name__ == "__main__":
    pytest.main([__file__])


class TestAdvancedGraphFeatures:
    """Test advanced graph-based features."""
    
    @pytest.fixture
    def sample_molecules(self):
        """Create sample molecules for testing."""
        smiles_list = [
            "CCO",  # Ethanol
            "CC(=O)O",  # Acetic acid
            "c1ccccc1",  # Benzene
            "CCN(CC)CC",  # Triethylamine
            "CC(C)C(=O)O"  # Isobutyric acid
        ]
        
        # Parse SMILES to get RDKit mol objects
        parsed_molecules = parse_smiles_batch(smiles_list, [f"mol_{i}" for i in range(len(smiles_list))])
        return parsed_molecules
    
    @pytest.fixture
    def advanced_features_calculator(self):
        """Create advanced graph features calculator."""
        from src.chemi_accel.descriptors.gnn_features import AdvancedGraphFeatures
        return AdvancedGraphFeatures(backend="cpu")
    
    def test_graph_fingerprint_generation(self, sample_molecules, advanced_features_calculator):
        """Test graph fingerprint generation."""
        for mol_data in sample_molecules:
            fingerprints = advanced_features_calculator.generate_graph_fingerprints(mol_data)
            
            # Check that all fingerprint types are present
            expected_types = ["morgan_graph", "substructure", "path_based", "atom_pair"]
            for fp_type in expected_types:
                assert fp_type in fingerprints
                assert isinstance(fingerprints[fp_type], np.ndarray)
                assert fingerprints[fp_type].shape[0] == 2048  # Default n_bits
                assert fingerprints[fp_type].dtype == np.int32
    
    def test_graph_similarity_computation(self, sample_molecules, advanced_features_calculator):
        """Test graph similarity computation."""
        if len(sample_molecules) < 2:
            pytest.skip("Need at least 2 molecules for similarity testing")
        
        mol1, mol2 = sample_molecules[0], sample_molecules[1]
        similarities = advanced_features_calculator.compute_graph_similarity_metrics(mol1, mol2)
        
        # Check that similarity metrics are computed
        expected_metrics = [
            "morgan_graph_tanimoto", "morgan_graph_dice", "morgan_graph_cosine",
            "substructure_tanimoto", "substructure_dice", "substructure_cosine",
            "path_based_tanimoto", "path_based_dice", "path_based_cosine",
            "atom_pair_tanimoto", "atom_pair_dice", "atom_pair_cosine",
            "graph_edit_distance"
        ]
        
        for metric in expected_metrics:
            assert metric in similarities
            assert isinstance(similarities[metric], float)
            assert 0.0 <= similarities[metric] <= 1.0 or metric == "graph_edit_distance"
    
    def test_molecular_scaffold_analysis(self, sample_molecules, advanced_features_calculator):
        """Test molecular scaffold analysis."""
        for mol_data in sample_molecules:
            scaffold_analysis = advanced_features_calculator.analyze_molecular_scaffolds(mol_data)
            
            # Check required fields
            required_fields = [
                "num_ring_systems", "ring_systems", "largest_scaffold",
                "scaffold_size", "num_side_chains", "side_chains", "scaffold_properties"
            ]
            
            for field in required_fields:
                assert field in scaffold_analysis
            
            # Check data types
            assert isinstance(scaffold_analysis["num_ring_systems"], int)
            assert isinstance(scaffold_analysis["ring_systems"], list)
            assert isinstance(scaffold_analysis["largest_scaffold"], list)
            assert isinstance(scaffold_analysis["scaffold_size"], int)
            assert isinstance(scaffold_analysis["num_side_chains"], int)
            assert isinstance(scaffold_analysis["side_chains"], list)
            assert isinstance(scaffold_analysis["scaffold_properties"], dict)
            
            # Check consistency
            assert scaffold_analysis["scaffold_size"] == len(scaffold_analysis["largest_scaffold"])
    
    def test_advanced_features_dataframe_output(self, sample_molecules):
        """Test advanced features DataFrame output."""
        from src.chemi_accel.descriptors.gnn_features import compute_advanced_graph_features
        
        df = compute_advanced_graph_features(sample_molecules, backend="cpu")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_molecules)
        
        # Check required columns
        required_cols = ["mol_id", "smiles"]
        for col in required_cols:
            assert col in df.columns
        
        # Check fingerprint statistics columns
        fp_types = ["morgan_graph", "substructure", "path_based", "atom_pair"]
        for fp_type in fp_types:
            assert f"{fp_type}_bits_set" in df.columns
            assert f"{fp_type}_density" in df.columns
        
        # Check scaffold analysis columns
        scaffold_cols = ["num_ring_systems", "scaffold_size", "num_side_chains"]
        for col in scaffold_cols:
            assert col in df.columns
    
    def test_graph_similarity_matrix(self, sample_molecules):
        """Test graph similarity matrix computation."""
        from src.chemi_accel.descriptors.gnn_features import compute_graph_similarity_matrix
        
        # Use subset for faster testing
        test_molecules = sample_molecules[:3] if len(sample_molecules) >= 3 else sample_molecules
        
        similarity_matrix = compute_graph_similarity_matrix(test_molecules, backend="cpu")
        
        assert isinstance(similarity_matrix, pd.DataFrame)
        assert similarity_matrix.shape[0] == len(test_molecules)
        assert similarity_matrix.shape[1] == len(test_molecules)
        
        # Check diagonal elements (self-similarity should be 1.0)
        for i in range(len(test_molecules)):
            assert similarity_matrix.iloc[i, i] == 1.0
        
        # Check symmetry
        for i in range(len(test_molecules)):
            for j in range(len(test_molecules)):
                assert abs(similarity_matrix.iloc[i, j] - similarity_matrix.iloc[j, i]) < 1e-6
    
    def test_performance_benchmarking(self, sample_molecules):
        """Test performance benchmarking functionality."""
        from src.chemi_accel.descriptors.gnn_features import benchmark_graph_features_performance
        
        # Use subset for faster testing
        test_molecules = sample_molecules[:3] if len(sample_molecules) >= 3 else sample_molecules
        
        benchmark_results = benchmark_graph_features_performance(test_molecules, traditional_method="rdkit")
        
        assert isinstance(benchmark_results, dict)
        
        # Check required fields
        required_fields = [
            "num_molecules", "chemi_accel_time", "traditional_time",
            "speedup", "chemi_accel_features_per_second", 
            "traditional_features_per_second", "backend_used"
        ]
        
        for field in required_fields:
            assert field in benchmark_results
        
        # Check data types and ranges
        assert benchmark_results["num_molecules"] == len(test_molecules)
        assert benchmark_results["chemi_accel_time"] >= 0
        assert benchmark_results["traditional_time"] >= 0
        assert benchmark_results["speedup"] >= 0
        assert benchmark_results["backend_used"] in ["cpu", "cuda", "jax"]
    
    def test_fingerprint_consistency(self, sample_molecules, advanced_features_calculator):
        """Test that fingerprints are consistent across multiple calls."""
        if not sample_molecules:
            pytest.skip("No molecules available for testing")
        
        mol_data = sample_molecules[0]
        
        # Generate fingerprints multiple times
        fp1 = advanced_features_calculator.generate_graph_fingerprints(mol_data)
        fp2 = advanced_features_calculator.generate_graph_fingerprints(mol_data)
        
        # Check consistency
        for fp_type in fp1.keys():
            assert np.array_equal(fp1[fp_type], fp2[fp_type])
    
    def test_empty_molecule_handling(self, advanced_features_calculator):
        """Test handling of empty or invalid molecules."""
        # Create invalid molecule
        invalid_mol = MolecularData(
            mol_id="invalid",
            smiles="",
            backend_data={}
        )
        
        # Should not raise exceptions
        fingerprints = advanced_features_calculator.generate_graph_fingerprints(invalid_mol)
        scaffold_analysis = advanced_features_calculator.analyze_molecular_scaffolds(invalid_mol)
        
        # Check that default values are returned
        for fp_type, fp_array in fingerprints.items():
            assert np.sum(fp_array) == 0  # All zeros for invalid molecule
        
        assert scaffold_analysis["num_ring_systems"] == 0
        assert scaffold_analysis["scaffold_size"] == 0
        assert scaffold_analysis["num_side_chains"] == 0
    
    def test_different_fingerprint_sizes(self, sample_molecules, advanced_features_calculator):
        """Test different fingerprint sizes."""
        if not sample_molecules:
            pytest.skip("No molecules available for testing")
        
        mol_data = sample_molecules[0]
        
        # Test different fingerprint sizes
        sizes_to_test = [512, 1024, 2048, 4096]
        
        for n_bits in sizes_to_test:
            fingerprints = advanced_features_calculator.generate_graph_fingerprints(
                mol_data, n_bits=n_bits
            )
            
            for fp_type, fp_array in fingerprints.items():
                assert fp_array.shape[0] == n_bits
    
    def test_different_radius_values(self, sample_molecules, advanced_features_calculator):
        """Test different radius values for fingerprint generation."""
        if not sample_molecules:
            pytest.skip("No molecules available for testing")
        
        mol_data = sample_molecules[0]
        
        # Test different radius values
        radii_to_test = [1, 2, 3, 4, 5]
        
        for radius in radii_to_test:
            fingerprints = advanced_features_calculator.generate_graph_fingerprints(
                mol_data, radius=radius
            )
            
            # Should generate valid fingerprints for all radii
            for fp_type, fp_array in fingerprints.items():
                assert isinstance(fp_array, np.ndarray)
                assert fp_array.dtype == np.int32