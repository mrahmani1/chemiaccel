"""
Stress testing for ChemiAccel memory management and resource limits.

This module provides comprehensive stress testing including memory management,
resource limits, polymer chain length and complexity testing, and system
performance under extreme conditions.
"""

import pytest
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import time
import threading
import gc
import psutil
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from src.chemi_accel.core.calculator import Calculator
from src.chemi_accel.descriptors.basic import BasicDescriptorCalculator
from src.chemi_accel.descriptors.constitutional import ConstitutionalDescriptorCalculator
from src.chemi_accel.descriptors.topological import TopologicalDescriptorCalculator
from src.chemi_accel.descriptors.fingerprints import FingerprintGenerator
from src.chemi_accel.polymer.monomer import MonomerAnalyzer
from src.chemi_accel.polymer.oligomer import OligomerProcessor
from src.chemi_accel.core.molecular import MolecularData
from src.chemi_accel.core.exceptions import ChemiAccelError, GPUMemoryError


class StressTestFramework:
    """Framework for comprehensive stress testing."""
    
    def __init__(self):
        """Initialize stress test framework."""
        self.calculators = {
            'basic': BasicDescriptorCalculator(),
            'constitutional': ConstitutionalDescriptorCalculator(use_gpu=False),
            'topological': TopologicalDescriptorCalculator(use_gpu=False),
            'fingerprint': FingerprintGenerator(use_gpu=False)
        }
        
        self.polymer_analyzers = {
            'monomer': MonomerAnalyzer(),
            'oligomer': OligomerProcessor()
        }
    
    def generate_stress_test_molecules(self, complexity_level: str = "medium") -> List[str]:
        """Generate molecules for stress testing.
        
        Args:
            complexity_level: "low", "medium", "high", or "extreme"
            
        Returns:
            List of SMILES strings for testing
        """
        molecules = []
        
        if complexity_level == "low":
            # Simple molecules
            base_molecules = ["C", "CC", "CCC", "CCO", "c1ccccc1"]
            molecules.extend(base_molecules * 100)
            
        elif complexity_level == "medium":
            # Moderately complex molecules
            base_molecules = [
                "CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCS",
                "c1ccc2ccccc2c1", "CC(C)C(=O)O", "c1ccncc1",
                "CC(=O)Nc1ccccc1", "CCc1ccc(cc1)O"
            ]
            molecules.extend(base_molecules * 50)
            
        elif complexity_level == "high":
            # Complex molecules
            base_molecules = [
                "CC(C)C(=O)Nc1ccc(cc1)C(=O)O",  # Ibuprofen-like
                "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
                "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
                "c1ccc2c(c1)ccc3c2ccc4c3ccc5c4cccc5",  # Large PAH
                "CCN(CC)CC(=O)Nc1c(cccc1C)C",  # Lidocaine-like
            ]
            molecules.extend(base_molecules * 20)
            
        elif complexity_level == "extreme":
            # Very complex molecules
            molecules.extend([
                "C" * 100,  # Long alkyl chain
                "c1ccc2c(c1)ccc3c2ccc4c3ccc5c4ccc6c5ccc7c6ccc8c7cccc8",  # Very large PAH
                "C1" + "C" * 50 + "1",  # Large ring
                "CC(C)" * 20 + "C",  # Highly branched
                "c1ccccc1" * 10,  # Multiple benzene rings
            ])
            
        return molecules
    
    def generate_polymer_stress_molecules(self) -> Dict[str, List[str]]:
        """Generate polymer molecules for stress testing."""
        return {
            "short_chains": [
                "CCCC",  # Butane
                "CCCCCC",  # Hexane
                "CCCCCCCC",  # Octane
                "CCc1ccccc1CCc2ccccc2",  # Styrene dimer
                "CC(C)CC(C)C",  # Propylene dimer
            ] * 20,
            
            "medium_chains": [
                "C" * 20,  # C20 alkane
                "C" * 30,  # C30 alkane
                "CC(C)" * 10 + "C",  # Branched C30
                ("CCc1ccccc1" * 5),  # Styrene pentamer
                ("CC(C)" * 15 + "C"),  # Propylene 15-mer
            ] * 10,
            
            "long_chains": [
                "C" * 50,  # C50 alkane
                "C" * 100,  # C100 alkane
                ("CC(C)" * 25 + "C"),  # Propylene 25-mer
                ("CCO" * 20),  # Ethylene oxide 20-mer
                "C" * 200,  # Very long chain
            ] * 5,
            
            "complex_polymers": [
                "CC(C)(C)c1ccc(cc1)C(C)(C)C" * 3,  # Polystyrene-like
                "CC(=O)OC" * 10,  # Polyester-like
                "NCCCCCCN" + "C(=O)CCCCC(=O)" * 5,  # Nylon-like
                "c1ccc(cc1)c2ccccc2" * 5,  # Polyphenyl
            ] * 3
        }
    
    def test_memory_scaling(self, max_memory_mb: int = 500) -> Dict[str, Any]:
        """Test memory usage scaling with molecule complexity.
        
        Args:
            max_memory_mb: Maximum allowed memory usage in MB
            
        Returns:
            Memory scaling test results
        """
        results = {
            "test_type": "memory_scaling",
            "max_memory_mb": max_memory_mb,
            "complexity_levels": {},
            "memory_exceeded": False,
            "peak_memory_mb": 0
        }
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        complexity_levels = ["low", "medium", "high"]
        
        for level in complexity_levels:
            print(f"Testing memory scaling for {level} complexity...")
            
            level_results = {
                "complexity": level,
                "molecules_processed": 0,
                "memory_before_mb": 0,
                "memory_after_mb": 0,
                "memory_increase_mb": 0,
                "processing_time": 0,
                "errors": []
            }
            
            level_results["memory_before_mb"] = process.memory_info().rss / 1024 / 1024
            start_time = time.time()
            
            try:
                molecules = self.generate_stress_test_molecules(level)
                calc = self.calculators['basic']
                
                for i, smiles in enumerate(molecules):
                    # Check memory usage periodically
                    if i % 50 == 0:
                        current_memory = process.memory_info().rss / 1024 / 1024
                        results["peak_memory_mb"] = max(results["peak_memory_mb"], current_memory)
                        
                        if current_memory - initial_memory > max_memory_mb:
                            results["memory_exceeded"] = True
                            level_results["errors"].append(f"Memory limit exceeded at molecule {i}")
                            break
                    
                    try:
                        if RDKIT_AVAILABLE:
                            mol = Chem.MolFromSmiles(smiles)
                            if mol is not None:
                                mol_data = MolecularData(
                                    smiles=smiles,
                                    mol_id=f"stress_{level}_{i}",
                                    num_atoms=mol.GetNumAtoms(),
                                    num_bonds=mol.GetNumBonds(),
                                    molecular_weight=Descriptors.MolWt(mol),
                                    backend_data={"rdkit_mol": mol}
                                )
                                
                                mw = calc.calculate_molecular_weight(mol_data)
                                level_results["molecules_processed"] += 1
                    
                    except Exception as e:
                        level_results["errors"].append(str(e))
                        if len(level_results["errors"]) > 10:  # Stop if too many errors
                            break
            
            except Exception as e:
                level_results["errors"].append(f"Level processing error: {str(e)}")
            
            level_results["processing_time"] = time.time() - start_time
            level_results["memory_after_mb"] = process.memory_info().rss / 1024 / 1024
            level_results["memory_increase_mb"] = (level_results["memory_after_mb"] - 
                                                 level_results["memory_before_mb"])
            
            results["complexity_levels"][level] = level_results
            
            # Force garbage collection between levels
            gc.collect()
        
        return results
    
    def test_polymer_chain_length_scaling(self) -> Dict[str, Any]:
        """Test performance scaling with polymer chain length."""
        results = {
            "test_type": "polymer_chain_scaling",
            "chain_categories": {},
            "performance_degradation": False
        }
        
        polymer_molecules = self.generate_polymer_stress_molecules()
        
        for category, molecules in polymer_molecules.items():
            print(f"Testing polymer chain scaling for {category}...")
            
            category_results = {
                "category": category,
                "molecules_tested": len(molecules),
                "successful_analyses": 0,
                "failed_analyses": 0,
                "average_processing_time": 0,
                "memory_usage_mb": 0,
                "errors": []
            }
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024
            
            total_time = 0
            monomer_analyzer = self.polymer_analyzers['monomer']
            
            for i, smiles in enumerate(molecules):
                start_time = time.time()
                
                try:
                    if RDKIT_AVAILABLE:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            mol_data = MolecularData(
                                smiles=smiles,
                                mol_id=f"polymer_stress_{category}_{i}",
                                num_atoms=mol.GetNumAtoms(),
                                num_bonds=mol.GetNumBonds(),
                                molecular_weight=Descriptors.MolWt(mol),
                                backend_data={"rdkit_mol": mol}
                            )
                            
                            # Try monomer analysis
                            analysis = monomer_analyzer.analyze_monomer(mol_data)
                            category_results["successful_analyses"] += 1
                
                except Exception as e:
                    category_results["failed_analyses"] += 1
                    category_results["errors"].append(str(e))
                
                processing_time = time.time() - start_time
                total_time += processing_time
                
                # Check for performance degradation
                if processing_time > 5.0:  # More than 5 seconds per molecule
                    results["performance_degradation"] = True
                    category_results["errors"].append(f"Slow processing: {processing_time:.2f}s for molecule {i}")
            
            category_results["average_processing_time"] = total_time / len(molecules) if molecules else 0
            
            memory_after = process.memory_info().rss / 1024 / 1024
            category_results["memory_usage_mb"] = memory_after - memory_before
            
            results["chain_categories"][category] = category_results
            
            # Cleanup between categories
            gc.collect()
        
        return results
    
    def test_concurrent_processing_stress(self, num_threads: int = 10, 
                                        molecules_per_thread: int = 50) -> Dict[str, Any]:
        """Test concurrent processing under stress conditions.
        
        Args:
            num_threads: Number of concurrent threads
            molecules_per_thread: Number of molecules per thread
            
        Returns:
            Concurrent processing test results
        """
        results = {
            "test_type": "concurrent_stress",
            "num_threads": num_threads,
            "molecules_per_thread": molecules_per_thread,
            "total_molecules": num_threads * molecules_per_thread,
            "thread_results": [],
            "overall_success_rate": 0,
            "race_conditions": 0,
            "deadlocks_detected": False,
            "memory_issues": 0
        }
        
        test_molecules = self.generate_stress_test_molecules("medium")
        
        def worker_function(thread_id: int) -> Dict[str, Any]:
            """Worker function for concurrent testing."""
            thread_result = {
                "thread_id": thread_id,
                "successful": 0,
                "failed": 0,
                "processing_time": 0,
                "memory_usage_mb": 0,
                "errors": []
            }
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024
            start_time = time.time()
            
            try:
                calc = BasicDescriptorCalculator()
                
                for i in range(molecules_per_thread):
                    molecule_idx = (thread_id * molecules_per_thread + i) % len(test_molecules)
                    smiles = test_molecules[molecule_idx]
                    
                    try:
                        if RDKIT_AVAILABLE:
                            mol = Chem.MolFromSmiles(smiles)
                            if mol is not None:
                                mol_data = MolecularData(
                                    smiles=smiles,
                                    mol_id=f"concurrent_{thread_id}_{i}",
                                    num_atoms=mol.GetNumAtoms(),
                                    num_bonds=mol.GetNumBonds(),
                                    molecular_weight=Descriptors.MolWt(mol),
                                    backend_data={"rdkit_mol": mol}
                                )
                                
                                mw = calc.calculate_molecular_weight(mol_data)
                                thread_result["successful"] += 1
                    
                    except Exception as e:
                        thread_result["failed"] += 1
                        thread_result["errors"].append(str(e))
            
            except Exception as e:
                thread_result["errors"].append(f"Thread error: {str(e)}")
            
            thread_result["processing_time"] = time.time() - start_time
            memory_after = process.memory_info().rss / 1024 / 1024
            thread_result["memory_usage_mb"] = memory_after - memory_before
            
            return thread_result
        
        # Run concurrent threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_function, i) for i in range(num_threads)]
            
            # Wait for completion with timeout
            completed_threads = 0
            for future in futures:
                try:
                    thread_result = future.result(timeout=60)  # 60 second timeout per thread
                    results["thread_results"].append(thread_result)
                    completed_threads += 1
                except Exception as e:
                    results["thread_results"].append({
                        "thread_id": -1,
                        "successful": 0,
                        "failed": molecules_per_thread,
                        "errors": [f"Thread timeout or error: {str(e)}"]
                    })
        
        # Analyze results
        total_successful = sum(r["successful"] for r in results["thread_results"])
        total_failed = sum(r["failed"] for r in results["thread_results"])
        total_processed = total_successful + total_failed
        
        if total_processed > 0:
            results["overall_success_rate"] = total_successful / total_processed
        
        # Check for race conditions (simple heuristic)
        if completed_threads != num_threads:
            results["race_conditions"] = num_threads - completed_threads
        
        # Check for memory issues
        for thread_result in results["thread_results"]:
            if thread_result["memory_usage_mb"] > 100:  # More than 100MB per thread
                results["memory_issues"] += 1
        
        return results
    
    def test_resource_exhaustion_recovery(self) -> Dict[str, Any]:
        """Test recovery from resource exhaustion scenarios."""
        results = {
            "test_type": "resource_exhaustion",
            "memory_exhaustion_test": {},
            "cpu_exhaustion_test": {},
            "recovery_successful": False
        }
        
        # Test memory exhaustion recovery
        print("Testing memory exhaustion recovery...")
        memory_test = self._test_memory_exhaustion_recovery()
        results["memory_exhaustion_test"] = memory_test
        
        # Test CPU exhaustion recovery
        print("Testing CPU exhaustion recovery...")
        cpu_test = self._test_cpu_exhaustion_recovery()
        results["cpu_exhaustion_test"] = cpu_test
        
        # Overall recovery assessment
        results["recovery_successful"] = (
            memory_test.get("recovery_successful", False) and
            cpu_test.get("recovery_successful", False)
        )
        
        return results
    
    def _test_memory_exhaustion_recovery(self) -> Dict[str, Any]:
        """Test recovery from memory exhaustion."""
        result = {
            "initial_memory_mb": 0,
            "peak_memory_mb": 0,
            "final_memory_mb": 0,
            "exhaustion_triggered": False,
            "recovery_successful": False,
            "recovery_time": 0
        }
        
        process = psutil.Process(os.getpid())
        result["initial_memory_mb"] = process.memory_info().rss / 1024 / 1024
        
        try:
            # Try to exhaust memory with large molecules
            large_molecules = ["C" * 500, "c1ccccc1" * 100, "C1" + "C" * 200 + "1"]
            calc = self.calculators['basic']
            
            for smiles in large_molecules:
                try:
                    if RDKIT_AVAILABLE:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            mol_data = MolecularData(
                                smiles=smiles,
                                mol_id=f"memory_exhaust_{hash(smiles)}",
                                num_atoms=mol.GetNumAtoms(),
                                num_bonds=mol.GetNumBonds(),
                                molecular_weight=Descriptors.MolWt(mol),
                                backend_data={"rdkit_mol": mol}
                            )
                            
                            mw = calc.calculate_molecular_weight(mol_data)
                            
                            current_memory = process.memory_info().rss / 1024 / 1024
                            result["peak_memory_mb"] = max(result["peak_memory_mb"], current_memory)
                
                except MemoryError:
                    result["exhaustion_triggered"] = True
                    break
                except Exception:
                    continue
            
            # Test recovery
            recovery_start = time.time()
            gc.collect()  # Force garbage collection
            
            # Try simple calculation after potential exhaustion
            try:
                simple_mol = Chem.MolFromSmiles("CCO") if RDKIT_AVAILABLE else None
                if simple_mol is not None:
                    simple_mol_data = MolecularData(
                        smiles="CCO",
                        mol_id="recovery_test",
                        num_atoms=simple_mol.GetNumAtoms(),
                        num_bonds=simple_mol.GetNumBonds(),
                        molecular_weight=Descriptors.MolWt(simple_mol),
                        backend_data={"rdkit_mol": simple_mol}
                    )
                    
                    recovery_mw = calc.calculate_molecular_weight(simple_mol_data)
                    result["recovery_successful"] = True
            
            except Exception:
                result["recovery_successful"] = False
            
            result["recovery_time"] = time.time() - recovery_start
        
        except Exception as e:
            result["error"] = str(e)
        
        result["final_memory_mb"] = process.memory_info().rss / 1024 / 1024
        return result
    
    def _test_cpu_exhaustion_recovery(self) -> Dict[str, Any]:
        """Test recovery from CPU exhaustion."""
        result = {
            "cpu_intensive_completed": False,
            "recovery_successful": False,
            "processing_time": 0
        }
        
        start_time = time.time()
        
        try:
            # CPU-intensive calculation with complex molecules
            complex_molecules = [
                "c1ccc2c(c1)ccc3c2ccc4c3ccc5c4ccc6c5cccc6",  # Large PAH
                "CC(C)" * 50 + "C",  # Highly branched
                "C1" + "C" * 30 + "1",  # Large ring
            ]
            
            calc = self.calculators['topological']  # More CPU-intensive
            
            for smiles in complex_molecules:
                try:
                    if RDKIT_AVAILABLE:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            mol_data = MolecularData(
                                smiles=smiles,
                                mol_id=f"cpu_exhaust_{hash(smiles)}",
                                num_atoms=mol.GetNumAtoms(),
                                num_bonds=mol.GetNumBonds(),
                                molecular_weight=Descriptors.MolWt(mol),
                                backend_data={"rdkit_mol": mol}
                            )
                            
                            # CPU-intensive calculation
                            wiener = calc.calculate_wiener_index(mol_data)
                
                except Exception:
                    continue
            
            result["cpu_intensive_completed"] = True
            
            # Test recovery with simple molecule
            try:
                simple_mol = Chem.MolFromSmiles("CCO") if RDKIT_AVAILABLE else None
                if simple_mol is not None:
                    simple_mol_data = MolecularData(
                        smiles="CCO",
                        mol_id="cpu_recovery_test",
                        num_atoms=simple_mol.GetNumAtoms(),
                        num_bonds=simple_mol.GetNumBonds(),
                        molecular_weight=Descriptors.MolWt(simple_mol),
                        backend_data={"rdkit_mol": simple_mol}
                    )
                    
                    recovery_wiener = calc.calculate_wiener_index(simple_mol_data)
                    result["recovery_successful"] = True
            
            except Exception:
                result["recovery_successful"] = False
        
        except Exception as e:
            result["error"] = str(e)
        
        result["processing_time"] = time.time() - start_time
        return result


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
class TestStressTesting:
    """Comprehensive stress testing for ChemiAccel."""
    
    def setup_method(self):
        """Set up stress test framework."""
        self.framework = StressTestFramework()
    
    def test_memory_scaling_stress(self):
        """Test memory usage scaling under stress."""
        results = self.framework.test_memory_scaling(max_memory_mb=300)
        
        # Should not exceed memory limits
        assert not results["memory_exceeded"], "Memory limit exceeded during stress test"
        
        # Should handle different complexity levels
        assert len(results["complexity_levels"]) > 0, "No complexity levels tested"
        
        # Check each complexity level
        for level, level_results in results["complexity_levels"].items():
            # Should process some molecules
            assert level_results["molecules_processed"] > 0, f"No molecules processed for {level}"
            
            # Memory increase should be reasonable
            assert level_results["memory_increase_mb"] < 200, \
                f"Excessive memory increase for {level}: {level_results['memory_increase_mb']} MB"
            
            # Should not have too many errors
            error_rate = len(level_results["errors"]) / max(1, level_results["molecules_processed"])
            assert error_rate < 0.1, f"High error rate for {level}: {error_rate:.2%}"
    
    def test_polymer_chain_length_stress(self):
        """Test performance with increasing polymer chain lengths."""
        results = self.framework.test_polymer_chain_length_scaling()
        
        # Should test multiple chain categories
        assert len(results["chain_categories"]) > 0, "No chain categories tested"
        
        # Should not show severe performance degradation
        assert not results["performance_degradation"], "Severe performance degradation detected"
        
        # Check each category
        for category, category_results in results["chain_categories"].items():
            # Should have some successful analyses
            total_tests = category_results["successful_analyses"] + category_results["failed_analyses"]
            if total_tests > 0:
                success_rate = category_results["successful_analyses"] / total_tests
                
                # Allow lower success rates for longer chains
                if "long" in category or "complex" in category:
                    assert success_rate > 0.3, f"Low success rate for {category}: {success_rate:.2%}"
                else:
                    assert success_rate > 0.7, f"Low success rate for {category}: {success_rate:.2%}"
            
            # Processing time should be reasonable
            assert category_results["average_processing_time"] < 2.0, \
                f"Slow processing for {category}: {category_results['average_processing_time']:.2f}s"
    
    def test_concurrent_processing_stress(self):
        """Test concurrent processing under stress conditions."""
        results = self.framework.test_concurrent_processing_stress(
            num_threads=5, molecules_per_thread=20  # Reduced for CI
        )
        
        # Should complete most threads
        completed_threads = len([r for r in results["thread_results"] if r["thread_id"] >= 0])
        assert completed_threads >= results["num_threads"] * 0.8, \
            f"Too many threads failed: {completed_threads}/{results['num_threads']}"
        
        # Should have reasonable success rate
        assert results["overall_success_rate"] > 0.7, \
            f"Low concurrent success rate: {results['overall_success_rate']:.2%}"
        
        # Should not have many race conditions
        assert results["race_conditions"] <= 1, f"Race conditions detected: {results['race_conditions']}"
        
        # Should not have excessive memory issues
        assert results["memory_issues"] < results["num_threads"] * 0.3, \
            f"Too many memory issues: {results['memory_issues']}"
    
    def test_resource_exhaustion_recovery(self):
        """Test recovery from resource exhaustion."""
        results = self.framework.test_resource_exhaustion_recovery()
        
        # Memory exhaustion test
        memory_test = results["memory_exhaustion_test"]
        assert "initial_memory_mb" in memory_test, "Memory test not completed"
        
        # Should recover from memory issues
        if memory_test.get("exhaustion_triggered", False):
            assert memory_test.get("recovery_successful", False), "Failed to recover from memory exhaustion"
        
        # CPU exhaustion test
        cpu_test = results["cpu_exhaustion_test"]
        assert "processing_time" in cpu_test, "CPU test not completed"
        
        # Should complete CPU-intensive tasks in reasonable time
        assert cpu_test["processing_time"] < 30, \
            f"CPU test took too long: {cpu_test['processing_time']:.2f}s"
    
    def test_large_batch_processing(self):
        """Test processing of large batches of molecules."""
        # Generate large batch
        molecules = self.framework.generate_stress_test_molecules("medium")[:200]  # Reduced for CI
        
        calc = self.framework.calculators['basic']
        
        successful = 0
        failed = 0
        start_time = time.time()
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        for i, smiles in enumerate(molecules):
            try:
                if RDKIT_AVAILABLE:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        mol_data = MolecularData(
                            smiles=smiles,
                            mol_id=f"batch_{i}",
                            num_atoms=mol.GetNumAtoms(),
                            num_bonds=mol.GetNumBonds(),
                            molecular_weight=Descriptors.MolWt(mol),
                            backend_data={"rdkit_mol": mol}
                        )
                        
                        mw = calc.calculate_molecular_weight(mol_data)
                        successful += 1
            
            except Exception:
                failed += 1
            
            # Check memory periodically
            if i % 50 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                assert memory_increase < 300, f"Excessive memory usage: {memory_increase} MB"
        
        processing_time = time.time() - start_time
        total_processed = successful + failed
        
        # Performance checks
        if total_processed > 0:
            success_rate = successful / total_processed
            assert success_rate > 0.8, f"Low batch success rate: {success_rate:.2%}"
            
            molecules_per_second = successful / processing_time
            assert molecules_per_second > 5, f"Slow batch processing: {molecules_per_second:.1f} mol/s"
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during extended processing."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        calc = self.framework.calculators['basic']
        molecules = ["CCO", "c1ccccc1", "CC(=O)O"] * 100
        
        memory_samples = []
        
        for i, smiles in enumerate(molecules):
            try:
                if RDKIT_AVAILABLE:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        mol_data = MolecularData(
                            smiles=smiles,
                            mol_id=f"leak_test_{i}",
                            num_atoms=mol.GetNumAtoms(),
                            num_bonds=mol.GetNumBonds(),
                            molecular_weight=Descriptors.MolWt(mol),
                            backend_data={"rdkit_mol": mol}
                        )
                        
                        mw = calc.calculate_molecular_weight(mol_data)
                
                # Sample memory every 25 molecules
                if i % 25 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_samples.append(current_memory - initial_memory)
            
            except Exception:
                continue
        
        # Force garbage collection
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024
        
        # Check for memory leaks
        if len(memory_samples) > 2:
            # Memory should not continuously increase
            memory_trend = np.polyfit(range(len(memory_samples)), memory_samples, 1)[0]
            assert memory_trend < 0.5, f"Potential memory leak detected: trend={memory_trend:.2f} MB per batch"
        
        # Final memory should not be excessive
        final_increase = final_memory - initial_memory
        assert final_increase < 100, f"Excessive final memory usage: {final_increase} MB"
    
    @pytest.mark.slow
    def test_extended_stress_run(self):
        """Extended stress test (marked as slow)."""
        # Run for extended period with various molecule types
        start_time = time.time()
        duration_minutes = 2  # 2 minute test
        
        calc = self.framework.calculators['basic']
        molecules = self.framework.generate_stress_test_molecules("medium")
        
        processed_count = 0
        error_count = 0
        
        while time.time() - start_time < duration_minutes * 60:
            for smiles in molecules:
                if time.time() - start_time >= duration_minutes * 60:
                    break
                
                try:
                    if RDKIT_AVAILABLE:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            mol_data = MolecularData(
                                smiles=smiles,
                                mol_id=f"extended_{processed_count}",
                                num_atoms=mol.GetNumAtoms(),
                                num_bonds=mol.GetNumBonds(),
                                molecular_weight=Descriptors.MolWt(mol),
                                backend_data={"rdkit_mol": mol}
                            )
                            
                            mw = calc.calculate_molecular_weight(mol_data)
                            processed_count += 1
                
                except Exception:
                    error_count += 1
        
        total_processed = processed_count + error_count
        
        # Should process reasonable number of molecules
        assert processed_count > 100, f"Too few molecules processed: {processed_count}"
        
        # Should maintain reasonable success rate
        if total_processed > 0:
            success_rate = processed_count / total_processed
            assert success_rate > 0.8, f"Low extended success rate: {success_rate:.2%}"


if __name__ == "__main__":
    # Run stress testing if called directly
    framework = StressTestFramework()
    
    print("Running comprehensive stress tests...")
    
    # Memory scaling test
    print("\n1. Memory scaling stress test...")
    memory_results = framework.test_memory_scaling(max_memory_mb=200)
    print(f"   Peak memory: {memory_results['peak_memory_mb']:.1f} MB")
    print(f"   Memory exceeded: {memory_results['memory_exceeded']}")
    
    # Polymer chain length test
    print("\n2. Polymer chain length stress test...")
    polymer_results = framework.test_polymer_chain_length_scaling()
    print(f"   Categories tested: {len(polymer_results['chain_categories'])}")
    print(f"   Performance degradation: {polymer_results['performance_degradation']}")
    
    # Concurrent processing test
    print("\n3. Concurrent processing stress test...")
    concurrent_results = framework.test_concurrent_processing_stress(num_threads=3, molecules_per_thread=10)
    print(f"   Success rate: {concurrent_results['overall_success_rate']:.2%}")
    print(f"   Race conditions: {concurrent_results['race_conditions']}")
    
    # Resource exhaustion test
    print("\n4. Resource exhaustion recovery test...")
    resource_results = framework.test_resource_exhaustion_recovery()
    print(f"   Recovery successful: {resource_results['recovery_successful']}")
    
    print("\nStress testing completed!")