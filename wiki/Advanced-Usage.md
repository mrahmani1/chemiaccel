# Advanced Usage Guide

This guide covers advanced ChemiAccel techniques for experienced users, including custom workflows, performance optimization, distributed computing, and integration with machine learning pipelines.

## Advanced Calculator Configurations

### Custom Backend Orchestration

```python
from chemi_accel import Calculator
from chemi_accel.core.orchestrator import BackendOrchestrator
from chemi_accel.core.config import ChemiAccelConfig

class AdaptiveBackendOrchestrator(BackendOrchestrator):
    """Custom orchestrator that adapts backend based on workload characteristics."""
    
    def select_backend(self, molecules, descriptors, **kwargs):
        """Intelligently select backend based on workload analysis."""
        dataset_size = len(molecules)
        descriptor_complexity = self._analyze_descriptor_complexity(descriptors)
        available_memory = self._get_available_memory()
        
        # Decision logic based on workload characteristics
        if dataset_size > 100000 and "cuda" in self.available_backends:
            if descriptor_complexity > 0.8 and available_memory > 16:
                return "cuda"
            elif descriptor_complexity > 0.5:
                return "cuda" if available_memory > 8 else "cpu"
        
        if dataset_size > 10000 and "jax" in self.available_backends:
            return "jax"
        
        return "cpu"
    
    def _analyze_descriptor_complexity(self, descriptors):
        """Analyze computational complexity of descriptor set."""
        complex_descriptors = {
            "quantum", "3d", "electronic", "conformer", "spectroscopic"
        }
        
        if isinstance(descriptors, str):
            descriptors = [descriptors]
        
        complexity_score = 0
        for desc in descriptors:
            if any(complex_type in desc.lower() for complex_type in complex_descriptors):
                complexity_score += 1
        
        return complexity_score / len(descriptors)

# Use custom orchestrator
config = ChemiAccelConfig(orchestrator=AdaptiveBackendOrchestrator())
calc = Calculator(config=config)
```

### Multi-Level Caching Strategy

```python
from chemi_accel.core.cache import CacheManager, RedisCache, FileCache, MemoryCache
import hashlib

class HierarchicalCacheManager(CacheManager):
    """Multi-level cache system with different storage backends."""
    
    def __init__(self):
        self.memory_cache = MemoryCache(max_size_mb=512)
        self.file_cache = FileCache(cache_dir="./cache", max_size_gb=10)
        self.redis_cache = RedisCache(host="localhost", port=6379, db=0)
        
    def get(self, key):
        """Get from cache with hierarchical lookup."""
        # Level 1: Memory cache (fastest)
        result = self.memory_cache.get(key)
        if result is not None:
            return result
        
        # Level 2: Redis cache (fast, shared)
        result = self.redis_cache.get(key)
        if result is not None:
            self.memory_cache.set(key, result)  # Promote to memory
            return result
        
        # Level 3: File cache (persistent)
        result = self.file_cache.get(key)
        if result is not None:
            self.memory_cache.set(key, result)
            self.redis_cache.set(key, result)
            return result
        
        return None
    
    def set(self, key, value):
        """Set in all cache levels."""
        self.memory_cache.set(key, value)
        self.redis_cache.set(key, value)
        self.file_cache.set(key, value)

# Configure advanced caching
config = ChemiAccelConfig(cache_manager=HierarchicalCacheManager())
calc = Calculator(config=config)
```

## Advanced Descriptor Engineering

### Dynamic Descriptor Selection

```python
from chemi_accel import Calculator
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class DynamicDescriptorSelector:
    """Dynamically select optimal descriptors for a given property."""
    
    def __init__(self, calc=None):
        self.calc = calc or Calculator()
        self.selected_descriptors = None
        self.selector_model = None
    
    def select_descriptors_for_property(self, 
                                      training_molecules, 
                                      target_property,
                                      method="mutual_info",
                                      n_descriptors=100):
        """Select optimal descriptors for predicting a specific property."""
        
        # Calculate comprehensive descriptor set
        all_descriptors = self.calc.pandas(
            training_molecules,
            descriptors="all"
        )
        
        # Remove non-numeric and constant columns
        numeric_descriptors = all_descriptors.select_dtypes(include=[np.number])
        constant_mask = numeric_descriptors.var() > 1e-8
        filtered_descriptors = numeric_descriptors.loc[:, constant_mask]
        
        if method == "mutual_info":
            # Mutual information based selection
            mi_scores = mutual_info_regression(filtered_descriptors, target_property)
            top_indices = np.argsort(mi_scores)[-n_descriptors:]
            self.selected_descriptors = filtered_descriptors.columns[top_indices].tolist()
            
        elif method == "rfe":
            # Recursive feature elimination
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rfe = RFE(estimator=rf, n_features_to_select=n_descriptors)
            rfe.fit(filtered_descriptors, target_property)
            self.selected_descriptors = filtered_descriptors.columns[rfe.support_].tolist()
            
        elif method == "correlation":
            # Correlation-based selection with redundancy filtering
            correlations = np.abs(np.corrcoef(filtered_descriptors.T, target_property)[:-1, -1])
            selected_indices = []
            
            # Greedy selection to minimize redundancy
            for _ in range(n_descriptors):
                remaining_indices = [i for i in range(len(correlations)) 
                                   if i not in selected_indices]
                if not remaining_indices:
                    break
                
                # Find descriptor with highest correlation to target
                best_idx = max(remaining_indices, key=lambda i: correlations[i])
                selected_indices.append(best_idx)
                
                # Reduce correlation scores for similar descriptors
                if len(selected_indices) > 1:
                    desc_corr = np.corrcoef(filtered_descriptors.iloc[:, selected_indices].T)
                    for i in remaining_indices:
                        similarity = np.max([abs(desc_corr[len(selected_indices)-1, 
                                                          selected_indices.index(j)]) 
                                           for j in selected_indices[:-1]] or [0])
                        correlations[i] *= (1 - similarity * 0.5)
            
            self.selected_descriptors = filtered_descriptors.columns[selected_indices].tolist()
        
        return self.selected_descriptors
    
    def calculate_selected_descriptors(self, molecules):
        """Calculate only the previously selected descriptors."""
        if self.selected_descriptors is None:
            raise ValueError("No descriptors selected. Run select_descriptors_for_property first.")
        
        return self.calc.pandas(molecules, descriptors=self.selected_descriptors)

# Example usage
selector = DynamicDescriptorSelector()

# Select descriptors optimal for solubility prediction
training_smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCCCCCCC"]  # Example
training_solubility = [-0.77, 2.13, -0.17, 4.09]  # log S values

optimal_descriptors = selector.select_descriptors_for_property(
    training_smiles, 
    training_solubility,
    method="mutual_info",
    n_descriptors=50
)

print(f"Selected {len(optimal_descriptors)} descriptors for solubility prediction")

# Use selected descriptors for new molecules
new_molecules = ["CCCCO", "c1ccc(cc1)O"]
results = selector.calculate_selected_descriptors(new_molecules)
```

### Custom Molecular Transformations

```python
from chemi_accel.core.transformations import MolecularTransformer
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np

class AdvancedMolecularTransformer(MolecularTransformer):
    """Advanced molecular transformations for descriptor engineering."""
    
    def __init__(self):
        super().__init__()
        self.transformations = {
            "tautomer_enumeration": self._enumerate_tautomers,
            "conformer_clustering": self._cluster_conformers,
            "fragment_analysis": self._analyze_fragments,
            "pharmacophore_matching": self._match_pharmacophores
        }
    
    def _enumerate_tautomers(self, mol, max_tautomers=10):
        """Enumerate tautomers and calculate ensemble properties."""
        from rdkit.Chem import rdMolEnumerator
        
        enumerator = rdMolEnumerator.TautomerEnumerator()
        tautomers = [enumerator.Canonicalize(mol)]
        
        # Add additional tautomers
        for taut in enumerator.Enumerate(mol):
            if len(tautomers) >= max_tautomers:
                break
            tautomers.append(taut)
        
        # Calculate ensemble average properties
        ensemble_props = {}
        prop_functions = {
            "mw": rdMolDescriptors.CalcExactMolWt,
            "logp": lambda m: rdMolDescriptors.CalcCrippenDescriptors(m)[0],
            "tpsa": rdMolDescriptors.CalcTPSA
        }
        
        for prop_name, prop_func in prop_functions.items():
            values = [prop_func(taut) for taut in tautomers if taut is not None]
            if values:
                ensemble_props[f"ensemble_avg_{prop_name}"] = np.mean(values)
                ensemble_props[f"ensemble_std_{prop_name}"] = np.std(values)
        
        return ensemble_props
    
    def _cluster_conformers(self, mol, num_conformers=50, rms_threshold=0.5):
        """Generate conformers and cluster them."""
        from rdkit.Chem import rdDistGeom, rdMolAlign
        
        # Add hydrogens and generate conformers
        mol_h = Chem.AddHs(mol)
        conf_ids = rdDistGeom.EmbedMultipleConfs(
            mol_h, 
            numConfs=num_conformers,
            randomSeed=42
        )
        
        if not conf_ids:
            return {"num_conformer_clusters": 0}
        
        # Optimize conformers
        from rdkit.Chem import rdForceFieldHelpers
        for conf_id in conf_ids:
            rdForceFieldHelpers.OptimizeMolecule(mol_h, confId=conf_id)
        
        # Cluster conformers by RMSD
        rms_matrix = []
        for i in range(len(conf_ids)):
            rms_row = []
            for j in range(len(conf_ids)):
                if i == j:
                    rms_row.append(0.0)
                else:
                    rms = rdMolAlign.CalcRMS(mol_h, mol_h, i, j)
                    rms_row.append(rms)
            rms_matrix.append(rms_row)
        
        # Simple clustering
        clusters = []
        used = set()
        
        for i in range(len(conf_ids)):
            if i in used:
                continue
            
            cluster = [i]
            used.add(i)
            
            for j in range(i + 1, len(conf_ids)):
                if j not in used and rms_matrix[i][j] < rms_threshold:
                    cluster.append(j)
                    used.add(j)
            
            clusters.append(cluster)
        
        return {
            "num_conformer_clusters": len(clusters),
            "largest_cluster_size": max(len(cluster) for cluster in clusters),
            "conformer_diversity": len(clusters) / len(conf_ids) if conf_ids else 0
        }
    
    def _analyze_fragments(self, mol):
        """Analyze molecular fragments and scaffolds."""
        from rdkit.Chem import rdMolDescriptors
        from rdkit.Chem.Scaffolds import MurckoScaffold
        
        # Murcko scaffold analysis
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        framework = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        
        # Fragment analysis
        fragments = []
        if mol.GetNumAtoms() > 10:  # Only for larger molecules
            # Break bonds and analyze fragments
            breakable_bonds = []
            for bond in mol.GetBonds():
                if not bond.IsInRing() and bond.GetBondType() == Chem.BondType.SINGLE:
                    breakable_bonds.append(bond.GetIdx())
            
            if breakable_bonds:
                # Break one bond and analyze resulting fragments
                bond_idx = breakable_bonds[0]  # Take first breakable bond
                fragmented = Chem.FragmentOnBonds(mol, [bond_idx])
                fragments = Chem.GetMolFrags(fragmented, asMols=True)
        
        return {
            "scaffold_atoms": scaffold.GetNumAtoms() if scaffold else 0,
            "framework_atoms": framework.GetNumAtoms() if framework else 0,
            "num_fragments": len(fragments),
            "scaffold_complexity": (scaffold.GetNumAtoms() / mol.GetNumAtoms()) if scaffold and mol.GetNumAtoms() > 0 else 0
        }
    
    def transform_molecules(self, molecules, transformations=None):
        """Apply multiple transformations to molecules."""
        if transformations is None:
            transformations = list(self.transformations.keys())
        
        results = []
        for smiles in molecules:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append({})
                continue
            
            mol_results = {"smiles": smiles}
            for transform_name in transformations:
                if transform_name in self.transformations:
                    try:
                        transform_results = self.transformations[transform_name](mol)
                        mol_results.update(transform_results)
                    except Exception as e:
                        print(f"Transformation {transform_name} failed for {smiles}: {e}")
            
            results.append(mol_results)
        
        return results

# Example usage
transformer = AdvancedMolecularTransformer()
molecules = ["CCO", "c1ccccc1", "CC(=O)Nc1ccc(cc1)O"]

transformed_results = transformer.transform_molecules(
    molecules,
    transformations=["tautomer_enumeration", "conformer_clustering", "fragment_analysis"]
)

for result in transformed_results:
    print(f"Molecule {result['smiles']}: {len(result)-1} additional properties calculated")
```

## Advanced Parallel Processing

### Custom Distributed Computing

```python
from chemi_accel.distributed import DistributedCalculator
from chemi_accel.core.communication import MessagePassing
import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

class HybridDistributedCalculator:
    """Hybrid distributed calculator combining process and thread parallelism."""
    
    def __init__(self, 
                 num_processes=None, 
                 num_threads_per_process=4,
                 use_gpu=True):
        self.num_processes = num_processes or mp.cpu_count()
        self.num_threads_per_process = num_threads_per_process
        self.use_gpu = use_gpu
        
    def _worker_process(self, process_id, molecule_chunk, descriptors, shared_config):
        """Worker process that handles a chunk of molecules."""
        from chemi_accel import Calculator
        import os
        
        # Set GPU device for this process (if using GPU)
        if self.use_gpu and "cuda" in shared_config.get("available_backends", []):
            gpu_id = process_id % shared_config.get("num_gpus", 1)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Create calculator for this process
        calc = Calculator(
            backend="cuda" if self.use_gpu else "cpu",
            num_threads=self.num_threads_per_process
        )
        
        try:
            results = calc.pandas(molecule_chunk, descriptors=descriptors)
            return {
                "process_id": process_id,
                "success": True,
                "results": results,
                "num_molecules": len(molecule_chunk)
            }
        except Exception as e:
            return {
                "process_id": process_id,
                "success": False,
                "error": str(e),
                "num_molecules": len(molecule_chunk)
            }
    
    def calculate_distributed(self, molecules, descriptors, chunk_size=None):
        """Distribute calculation across multiple processes."""
        if chunk_size is None:
            chunk_size = max(100, len(molecules) // (self.num_processes * 4))
        
        # Split molecules into chunks
        chunks = [molecules[i:i+chunk_size] 
                 for i in range(0, len(molecules), chunk_size)]
        
        # Shared configuration
        shared_config = {
            "available_backends": Calculator().available_backends,
            "num_gpus": self._get_num_gpus()
        }
        
        print(f"Processing {len(molecules)} molecules in {len(chunks)} chunks "
              f"across {self.num_processes} processes")
        
        # Execute in parallel
        results = []
        failed_chunks = []
        
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(
                    self._worker_process, 
                    i, 
                    chunk, 
                    descriptors, 
                    shared_config
                ): (i, chunk) 
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_id, chunk_molecules = future_to_chunk[future]
                
                try:
                    result = future.result()
                    if result["success"]:
                        results.append(result["results"])
                        print(f"Completed chunk {chunk_id}: {result['num_molecules']} molecules")
                    else:
                        print(f"Failed chunk {chunk_id}: {result['error']}")
                        failed_chunks.append((chunk_id, chunk_molecules))
                        
                except Exception as e:
                    print(f"Chunk {chunk_id} raised exception: {e}")
                    failed_chunks.append((chunk_id, chunk_molecules))
        
        # Combine successful results
        if results:
            import pandas as pd
            combined_results = pd.concat(results, ignore_index=True)
        else:
            combined_results = None
        
        # Handle failed chunks with fallback
        if failed_chunks:
            print(f"Retrying {len(failed_chunks)} failed chunks with CPU backend...")
            fallback_calc = Calculator(backend="cpu")
            
            for chunk_id, chunk_molecules in failed_chunks:
                try:
                    fallback_result = fallback_calc.pandas(chunk_molecules, descriptors=descriptors)
                    if combined_results is not None:
                        combined_results = pd.concat([combined_results, fallback_result], ignore_index=True)
                    else:
                        combined_results = fallback_result
                    print(f"Recovered chunk {chunk_id} with CPU backend")
                except Exception as e:
                    print(f"Chunk {chunk_id} failed permanently: {e}")
        
        return combined_results
    
    def _get_num_gpus(self):
        """Get number of available GPUs."""
        try:
            import cupy as cp
            return cp.cuda.runtime.getDeviceCount() if cp.cuda.is_available() else 0
        except ImportError:
            return 0

# Example usage
hybrid_calc = HybridDistributedCalculator(
    num_processes=8,
    num_threads_per_process=4,
    use_gpu=True
)

# Large dataset processing
large_molecules = ["CCO", "c1ccccc1", "CC(=O)O"] * 10000  # 30K molecules
results = hybrid_calc.calculate_distributed(
    large_molecules,
    descriptors=["molecular_weight", "logp", "tpsa", "morgan_fingerprint_2048"],
    chunk_size=1000
)

print(f"Processed {len(results)} molecules with hybrid distributed computing")
```

### Adaptive Load Balancing

```python
import time
import threading
from queue import Queue, Empty
from collections import deque

class AdaptiveLoadBalancer:
    """Adaptive load balancer that adjusts work distribution based on performance."""
    
    def __init__(self, workers, initial_batch_size=1000):
        self.workers = workers  # List of worker calculators
        self.worker_performance = {i: deque(maxlen=10) for i in range(len(workers))}
        self.worker_queues = {i: Queue() for i in range(len(workers))}
        self.results_queue = Queue()
        self.batch_sizes = {i: initial_batch_size for i in range(len(workers))}
        self.active = True
        
    def _worker_thread(self, worker_id, calculator):
        """Worker thread that processes molecules."""
        while self.active:
            try:
                task = self.worker_queues[worker_id].get(timeout=1)
                if task is None:  # Shutdown signal
                    break
                
                molecules, descriptors, task_id = task
                start_time = time.time()
                
                try:
                    results = calculator.pandas(molecules, descriptors=descriptors)
                    processing_time = time.time() - start_time
                    throughput = len(molecules) / processing_time
                    
                    # Record performance
                    self.worker_performance[worker_id].append(throughput)
                    
                    # Adjust batch size based on performance
                    self._adjust_batch_size(worker_id, throughput, processing_time)
                    
                    self.results_queue.put({
                        "task_id": task_id,
                        "worker_id": worker_id,
                        "success": True,
                        "results": results,
                        "throughput": throughput
                    })
                    
                except Exception as e:
                    self.results_queue.put({
                        "task_id": task_id,
                        "worker_id": worker_id,
                        "success": False,
                        "error": str(e)
                    })
                
                self.worker_queues[worker_id].task_done()
                
            except Empty:
                continue
    
    def _adjust_batch_size(self, worker_id, throughput, processing_time):
        """Adjust batch size based on worker performance."""
        performance_history = list(self.worker_performance[worker_id])
        current_batch_size = self.batch_sizes[worker_id]
        
        if len(performance_history) >= 3:
            recent_avg = sum(performance_history[-3:]) / 3
            overall_avg = sum(performance_history) / len(performance_history)
            
            # Increase batch size if performance is improving
            if recent_avg > overall_avg * 1.1:
                self.batch_sizes[worker_id] = min(current_batch_size * 1.2, 10000)
            # Decrease batch size if performance is degrading
            elif recent_avg < overall_avg * 0.9:
                self.batch_sizes[worker_id] = max(current_batch_size * 0.8, 100)
        
        # Adjust based on processing time (avoid too long tasks)
        if processing_time > 60:  # More than 1 minute
            self.batch_sizes[worker_id] = max(current_batch_size * 0.7, 100)
        elif processing_time < 5:  # Less than 5 seconds
            self.batch_sizes[worker_id] = min(current_batch_size * 1.3, 10000)
    
    def process_molecules(self, molecules, descriptors):
        """Process molecules with adaptive load balancing."""
        # Start worker threads
        threads = []
        for worker_id, calculator in enumerate(self.workers):
            thread = threading.Thread(
                target=self._worker_thread,
                args=(worker_id, calculator)
            )
            thread.start()
            threads.append(thread)
        
        # Distribute work adaptively
        molecule_chunks = []
        start_idx = 0
        task_id = 0
        
        while start_idx < len(molecules):
            # Select worker with best performance and available capacity
            best_worker = self._select_best_worker()
            batch_size = int(self.batch_sizes[best_worker])
            
            end_idx = min(start_idx + batch_size, len(molecules))
            chunk = molecules[start_idx:end_idx]
            
            self.worker_queues[best_worker].put((chunk, descriptors, task_id))
            molecule_chunks.append((task_id, len(chunk)))
            
            start_idx = end_idx
            task_id += 1
        
        # Collect results
        results = []
        completed_tasks = 0
        total_tasks = len(molecule_chunks)
        
        while completed_tasks < total_tasks:
            try:
                result = self.results_queue.get(timeout=10)
                if result["success"]:
                    results.append(result["results"])
                    print(f"Task {result['task_id']} completed by worker {result['worker_id']} "
                          f"({result['throughput']:.1f} mol/s)")
                else:
                    print(f"Task {result['task_id']} failed: {result['error']}")
                
                completed_tasks += 1
                
            except Empty:
                print("Waiting for results...")
        
        # Shutdown workers
        self.active = False
        for worker_id in range(len(self.workers)):
            self.worker_queues[worker_id].put(None)
        
        for thread in threads:
            thread.join()
        
        # Combine results
        if results:
            import pandas as pd
            return pd.concat(results, ignore_index=True)
        else:
            return None
    
    def _select_best_worker(self):
        """Select the best performing worker with available capacity."""
        worker_scores = {}
        
        for worker_id in range(len(self.workers)):
            queue_size = self.worker_queues[worker_id].qsize()
            performance_history = list(self.worker_performance[worker_id])
            
            # Score based on performance and queue size
            if performance_history:
                avg_performance = sum(performance_history) / len(performance_history)
            else:
                avg_performance = 1.0  # Default for new workers
            
            # Prefer workers with high performance and low queue size
            score = avg_performance / (1 + queue_size)
            worker_scores[worker_id] = score
        
        return max(worker_scores, key=worker_scores.get)

# Example usage
from chemi_accel import Calculator

# Create multiple workers with different configurations
workers = [
    Calculator(backend="cuda", device_id=0),
    Calculator(backend="cuda", device_id=1),  # Multi-GPU
    Calculator(backend="cpu", num_threads=8),
    Calculator(backend="jax")
]

load_balancer = AdaptiveLoadBalancer(workers)

# Process large dataset with adaptive load balancing
molecules = ["CCO", "c1ccccc1", "CC(=O)O"] * 5000  # 15K molecules
results = load_balancer.process_molecules(
    molecules,
    descriptors=["molecular_weight", "logp", "tpsa"]
)

print(f"Processed {len(results)} molecules with adaptive load balancing")
```

## Advanced AI Integration

### Reinforcement Learning for Molecular Optimization

```python
from chemi_accel import Calculator
from chemi_accel.ai.reinforcement_learning import MolecularRLAgent
import numpy as np
import gym
from gym import spaces

class MolecularOptimizationEnv(gym.Env):
    """Reinforcement learning environment for molecular optimization."""
    
    def __init__(self, target_properties, calc=None):
        super().__init__()
        self.calc = calc or Calculator(backend="cuda")
        self.target_properties = target_properties  # Dict of property targets
        
        # Action space: molecular modifications
        self.action_space = spaces.Discrete(10)  # 10 different modification types
        
        # Observation space: molecular descriptors
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(100,),  # 100 descriptors
            dtype=np.float32
        )
        
        self.current_molecule = None
        self.current_descriptors = None
        self.step_count = 0
        self.max_steps = 50
        
    def reset(self):
        """Reset environment with a random starting molecule."""
        starting_molecules = [
            "CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCC", "CCCC",
            "c1ccc(cc1)O", "CC(C)O", "CCCCO", "c1ccc(cc1)N"
        ]
        
        self.current_molecule = np.random.choice(starting_molecules)
        self.current_descriptors = self._calculate_descriptors(self.current_molecule)
        self.step_count = 0
        
        return self.current_descriptors
    
    def step(self, action):
        """Apply molecular modification and return new state."""
        # Apply modification based on action
        new_molecule = self._apply_modification(self.current_molecule, action)
        
        if new_molecule is None:
            # Invalid modification, penalize
            reward = -1.0
            done = True
            return self.current_descriptors, reward, done, {}
        
        # Calculate new descriptors
        new_descriptors = self._calculate_descriptors(new_molecule)
        
        if new_descriptors is None:
            # Invalid molecule, penalize
            reward = -1.0
            done = True
            return self.current_descriptors, reward, done, {}
        
        # Calculate reward based on property targets
        reward = self._calculate_reward(new_descriptors)
        
        # Update state
        self.current_molecule = new_molecule
        self.current_descriptors = new_descriptors
        self.step_count += 1
        
        # Check if done
        done = (self.step_count >= self.max_steps or 
                self._is_target_achieved(new_descriptors))
        
        return new_descriptors, reward, done, {
            "molecule": new_molecule,
            "properties": {k: new_descriptors[i] for i, k in enumerate(self.target_properties.keys())}
        }
    
    def _calculate_descriptors(self, molecule):
        """Calculate molecular descriptors."""
        try:
            result = self.calc.pandas(
                [molecule],
                descriptors=list(self.target_properties.keys()) + ["morgan_fingerprint_1024"]
            )
            
            # Combine property values and fingerprint
            property_values = [result.iloc[0][prop] for prop in self.target_properties.keys()]
            fingerprint = result.iloc[0]["morgan_fingerprint_1024"]
            
            # Pad or truncate to fixed size
            combined = np.array(property_values + list(fingerprint)[:97], dtype=np.float32)
            if len(combined) < 100:
                combined = np.pad(combined, (0, 100 - len(combined)))
            
            return combined[:100]
            
        except Exception as e:
            print(f"Error calculating descriptors for {molecule}: {e}")
            return None
    
    def _apply_modification(self, molecule, action):
        """Apply molecular modification based on action."""
        from rdkit import Chem
        from rdkit.Chem import rdMolStandardize
        
        mol = Chem.MolFromSmiles(molecule)
        if mol is None:
            return None
        
        try:
            # Define modification strategies
            if action == 0:  # Add methyl group
                return self._add_functional_group(mol, "C")
            elif action == 1:  # Add hydroxyl group
                return self._add_functional_group(mol, "O")
            elif action == 2:  # Add amino group
                return self._add_functional_group(mol, "N")
            elif action == 3:  # Remove random atom (if possible)
                return self._remove_atom(mol)
            elif action == 4:  # Change bond order
                return self._change_bond_order(mol)
            elif action == 5:  # Add ring
                return self._add_ring(mol)
            elif action == 6:  # Substitute hydrogen
                return self._substitute_hydrogen(mol)
            elif action == 7:  # Standardize molecule
                normalizer = rdMolStandardize.Normalizer()
                normalized = normalizer.normalize(mol)
                return Chem.MolToSmiles(normalized)
            else:
                return molecule  # No change
            
        except Exception as e:
            print(f"Modification failed: {e}")
            return None
    
    def _calculate_reward(self, descriptors):
        """Calculate reward based on how close descriptors are to targets."""
        reward = 0.0
        
        for i, (prop_name, target_value) in enumerate(self.target_properties.items()):
            current_value = descriptors[i]
            
            # Gaussian reward centered on target
            if isinstance(target_value, tuple):  # Range target
                if target_value[0] <= current_value <= target_value[1]:
                    reward += 1.0
                else:
                    distance = min(abs(current_value - target_value[0]),
                                 abs(current_value - target_value[1]))
                    reward -= distance / 100.0
            else:  # Point target
                distance = abs(current_value - target_value)
                reward += np.exp(-distance / target_value) if target_value != 0 else np.exp(-distance)
        
        return reward / len(self.target_properties)
    
    def _is_target_achieved(self, descriptors, tolerance=0.1):
        """Check if target properties are achieved within tolerance."""
        for i, (prop_name, target_value) in enumerate(self.target_properties.items()):
            current_value = descriptors[i]
            
            if isinstance(target_value, tuple):
                if not (target_value[0] <= current_value <= target_value[1]):
                    return False
            else:
                if abs(current_value - target_value) > abs(target_value * tolerance):
                    return False
        
        return True

# Example usage with PPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Define optimization targets
targets = {
    "molecular_weight": (200, 400),  # Drug-like range
    "logp": (1, 3),                  # Good permeability
    "tpsa": (40, 100)               # Good absorption
}

# Create environment
env = make_vec_env(
    lambda: MolecularOptimizationEnv(targets),
    n_envs=4  # Parallel environments
)

# Train RL agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10
)

print("Training RL agent for molecular optimization...")
model.learn(total_timesteps=100000)

# Test trained agent
test_env = MolecularOptimizationEnv(targets)
obs = test_env.reset()

for step in range(50):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)
    
    if done:
        print(f"Optimization completed at step {step}")
        print(f"Final molecule: {info['molecule']}")
        print(f"Final properties: {info['properties']}")
        break
```

### Multi-Objective Bayesian Optimization

```python
from chemi_accel import Calculator
from chemi_accel.ai.bayesian_optimization import MultiObjectiveBayesianOptimizer
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class AdvancedMolecularBayesianOptimizer:
    """Advanced Bayesian optimizer for multi-objective molecular design."""
    
    def __init__(self, calc=None):
        self.calc = calc or Calculator(backend="cuda")
        self.gp_models = {}
        self.evaluated_molecules = []
        self.evaluated_objectives = []
        
    def optimize_molecules(self, 
                          objective_functions,
                          initial_molecules,
                          n_iterations=50,
                          acquisition_function="ei",
                          batch_size=5):
        """Multi-objective Bayesian optimization of molecules."""
        
        # Initialize with provided molecules
        current_molecules = list(initial_molecules)
        
        # Evaluate initial molecules
        print("Evaluating initial molecules...")
        initial_objectives = self._evaluate_objectives(current_molecules, objective_functions)
        
        self.evaluated_molecules.extend(current_molecules)
        self.evaluated_objectives.extend(initial_objectives)
        
        # Initialize GP models for each objective
        for i, obj_name in enumerate(objective_functions.keys()):
            self.gp_models[obj_name] = GaussianProcessRegressor(
                kernel=Matern(length_scale=1.0, nu=2.5),
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5
            )
        
        for iteration in range(n_iterations):
            print(f"Bayesian optimization iteration {iteration + 1}/{n_iterations}")
            
            # Calculate molecular descriptors for current molecules
            descriptors = self._calculate_descriptors(self.evaluated_molecules)
            
            # Update GP models
            self._update_gp_models(descriptors, objective_functions)
            
            # Generate candidate molecules
            candidates = self._generate_candidates(
                descriptors, 
                objective_functions,
                acquisition_function,
                batch_size
            )
            
            if not candidates:
                print("No new candidates generated, stopping optimization")
                break
            
            # Evaluate candidates
            candidate_objectives = self._evaluate_objectives(candidates, objective_functions)
            
            # Update data
            self.evaluated_molecules.extend(candidates)
            self.evaluated_objectives.extend(candidate_objectives)
            
            # Report progress
            self._report_progress(iteration, candidates, candidate_objectives, objective_functions)
        
        return self._get_pareto_front(objective_functions)
    
    def _evaluate_objectives(self, molecules, objective_functions):
        """Evaluate objective functions for molecules."""
        # Calculate descriptors
        descriptors = self.calc.pandas(
            molecules,
            descriptors=list(set().union(*[obj["descriptors"] for obj in objective_functions.values()]))
        )
        
        objectives = []
        for i, molecule in enumerate(molecules):
            mol_objectives = {}
            for obj_name, obj_config in objective_functions.items():
                # Extract relevant descriptors
                relevant_descriptors = {
                    desc: descriptors.iloc[i][desc] 
                    for desc in obj_config["descriptors"]
                }
                
                # Calculate objective value
                if "function" in obj_config:
                    obj_value = obj_config["function"](relevant_descriptors)
                else:
                    # Simple weighted sum of descriptors
                    weights = obj_config.get("weights", {desc: 1.0 for desc in obj_config["descriptors"]})
                    obj_value = sum(weights.get(desc, 1.0) * relevant_descriptors[desc] 
                                  for desc in obj_config["descriptors"])
                
                mol_objectives[obj_name] = obj_value
            
            objectives.append(mol_objectives)
        
        return objectives
    
    def _calculate_descriptors(self, molecules):
        """Calculate descriptors for molecules."""
        return self.calc.pandas(
            molecules,
            descriptors=[
                "molecular_weight", "logp", "tpsa", "num_hbd", "num_hba",
                "morgan_fingerprint_1024"
            ]
        )
    
    def _update_gp_models(self, descriptors, objective_functions):
        """Update Gaussian Process models."""
        # Prepare feature matrix (use numerical descriptors only)
        X = descriptors.select_dtypes(include=[np.number]).values
        
        # Update each GP model
        for obj_name in objective_functions.keys():
            y = np.array([obj[obj_name] for obj in self.evaluated_objectives])
            self.gp_models[obj_name].fit(X, y)
    
    def _generate_candidates(self, descriptors, objective_functions, acquisition_function, batch_size):
        """Generate candidate molecules using acquisition function."""
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        
        # Generate candidate SMILES using simple mutations
        candidates = []
        base_molecules = self.evaluated_molecules[-10:]  # Use recent molecules as base
        
        for _ in range(batch_size * 10):  # Generate more candidates than needed
            base_mol = np.random.choice(base_molecules)
            candidate = self._mutate_molecule(base_mol)
            
            if candidate and candidate not in self.evaluated_molecules and candidate not in candidates:
                candidates.append(candidate)
                
                if len(candidates) >= batch_size * 2:
                    break
        
        if not candidates:
            return []
        
        # Calculate descriptors for candidates
        try:
            candidate_descriptors = self._calculate_descriptors(candidates)
        except Exception as e:
            print(f"Error calculating descriptors for candidates: {e}")
            return []
        
        # Calculate acquisition scores
        X_candidates = candidate_descriptors.select_dtypes(include=[np.number]).values
        acquisition_scores = self._calculate_acquisition_scores(
            X_candidates, 
            objective_functions, 
            acquisition_function
        )
        
        # Select best candidates
        top_indices = np.argsort(acquisition_scores)[-batch_size:]
        return [candidates[i] for i in top_indices]
    
    def _mutate_molecule(self, smiles):
        """Apply simple mutations to a molecule."""
        from rdkit import Chem
        from rdkit.Chem import rdMolStandardize
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        mutation_type = np.random.choice(["add_atom", "remove_atom", "change_atom"])
        
        try:
            if mutation_type == "add_atom" and mol.GetNumAtoms() < 50:
                # Add a random atom
                new_mol = Chem.RWMol(mol)
                atom_types = [6, 7, 8, 16]  # C, N, O, S
                new_atom = Chem.Atom(np.random.choice(atom_types))
                new_atom_idx = new_mol.AddAtom(new_atom)
                
                # Connect to random existing atom
                if mol.GetNumAtoms() > 0:
                    existing_atom_idx = np.random.randint(0, mol.GetNumAtoms())
                    new_mol.AddBond(existing_atom_idx, new_atom_idx, Chem.BondType.SINGLE)
                
                return Chem.MolToSmiles(new_mol)
                
            elif mutation_type == "remove_atom" and mol.GetNumAtoms() > 3:
                # Remove a random non-aromatic atom
                new_mol = Chem.RWMol(mol)
                removable_atoms = [i for i, atom in enumerate(mol.GetAtoms()) 
                                 if not atom.GetIsAromatic()]
                
                if removable_atoms:
                    atom_to_remove = np.random.choice(removable_atoms)
                    new_mol.RemoveAtom(atom_to_remove)
                    return Chem.MolToSmiles(new_mol)
                    
            elif mutation_type == "change_atom":
                # Change atom type
                new_mol = Chem.RWMol(mol)
                changeable_atoms = [i for i, atom in enumerate(mol.GetAtoms()) 
                                  if atom.GetAtomicNum() in [6, 7, 8, 16]]
                
                if changeable_atoms:
                    atom_idx = np.random.choice(changeable_atoms)
                    current_atomic_num = mol.GetAtomWithIdx(atom_idx).GetAtomicNum()
                    new_atomic_nums = [6, 7, 8, 16]
                    new_atomic_nums = [num for num in new_atomic_nums if num != current_atomic_num]
                    
                    if new_atomic_nums:
                        new_atomic_num = np.random.choice(new_atomic_nums)
                        new_mol.GetAtomWithIdx(atom_idx).SetAtomicNum(new_atomic_num)
                        return Chem.MolToSmiles(new_mol)
            
            return smiles  # Return original if mutation failed
            
        except Exception as e:
            return None
    
    def _calculate_acquisition_scores(self, X_candidates, objective_functions, acquisition_function):
        """Calculate acquisition function scores."""
        scores = np.zeros(len(X_candidates))
        
        for i, x in enumerate(X_candidates):
            x_reshaped = x.reshape(1, -1)
            
            if acquisition_function == "ei":  # Expected Improvement
                score = 0
                for obj_name in objective_functions.keys():
                    gp = self.gp_models[obj_name]
                    mu, sigma = gp.predict(x_reshaped, return_std=True)
                    
                    # Current best value for this objective
                    current_best = max(obj[obj_name] for obj in self.evaluated_objectives)
                    
                    # Expected improvement
                    with np.errstate(divide='warn'):
                        z = (mu - current_best) / sigma if sigma > 0 else 0
                        ei = (mu - current_best) * norm.cdf(z) + sigma * norm.pdf(z) if sigma > 0 else 0
                    
                    score += ei
                
                scores[i] = score
                
            elif acquisition_function == "ucb":  # Upper Confidence Bound
                score = 0
                for obj_name in objective_functions.keys():
                    gp = self.gp_models[obj_name]
                    mu, sigma = gp.predict(x_reshaped, return_std=True)
                    
                    # UCB with beta=2
                    ucb = mu + 2 * sigma
                    score += ucb
                
                scores[i] = score
        
        return scores
    
    def _get_pareto_front(self, objective_functions):
        """Extract Pareto front from evaluated molecules."""
        objectives_matrix = np.array([
            [obj[obj_name] for obj_name in objective_functions.keys()]
            for obj in self.evaluated_objectives
        ])
        
        # Find Pareto front
        pareto_mask = np.ones(len(objectives_matrix), dtype=bool)
        for i, obj_i in enumerate(objectives_matrix):
            for j, obj_j in enumerate(objectives_matrix):
                if i != j and np.all(obj_j >= obj_i) and np.any(obj_j > obj_i):
                    pareto_mask[i] = False
                    break
        
        pareto_molecules = [self.evaluated_molecules[i] for i in range(len(pareto_mask)) if pareto_mask[i]]
        pareto_objectives = [self.evaluated_objectives[i] for i in range(len(pareto_mask)) if pareto_mask[i]]
        
        return {
            "molecules": pareto_molecules,
            "objectives": pareto_objectives,
            "pareto_mask": pareto_mask
        }
    
    def _report_progress(self, iteration, candidates, candidate_objectives, objective_functions):
        """Report optimization progress."""
        print(f"  Generated {len(candidates)} new candidates")
        
        # Find best values for each objective
        all_objectives = self.evaluated_objectives
        for obj_name in objective_functions.keys():
            values = [obj[obj_name] for obj in all_objectives]
            best_value = max(values)
            best_idx = values.index(best_value)
            print(f"  Best {obj_name}: {best_value:.3f} (molecule: {self.evaluated_molecules[best_idx]})")

# Example usage
optimizer = AdvancedMolecularBayesianOptimizer()

# Define multi-objective optimization problem
objectives = {
    "solubility": {
        "descriptors": ["logp", "tpsa", "molecular_weight"],
        "function": lambda desc: -(desc["logp"] - 2)**2 - (desc["tpsa"] - 60)**2/1000  # Maximize
    },
    "drug_likeness": {
        "descriptors": ["molecular_weight", "num_hbd", "num_hba"],
        "function": lambda desc: (
            1.0 if 200 <= desc["molecular_weight"] <= 500 else 0.0
        ) + (
            1.0 if desc["num_hbd"] <= 5 else 0.0
        ) + (
            1.0 if desc["num_hba"] <= 10 else 0.0
        )
    }
}

# Initial molecules
initial_mols = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCC"]

# Run optimization
result = optimizer.optimize_molecules(
    objectives,
    initial_mols,
    n_iterations=20,
    batch_size=3
)

print(f"\nOptimization completed!")
print(f"Found {len(result['molecules'])} Pareto-optimal molecules:")
for mol, obj in zip(result['molecules'], result['objectives']):
    print(f"  {mol}: {obj}")
```

---

*This advanced guide demonstrates sophisticated ChemiAccel usage patterns. For additional advanced topics, see the domain-specific guides and [API Reference](API-Reference).*
