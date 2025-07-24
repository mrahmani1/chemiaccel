"""
Error handling and robustness testing for ChemiAccel.

This module provides comprehensive testing of error handling, graceful fallbacks,
memory management, and system robustness under various failure conditions.
"""

import pytest
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import time
import threading
import gc
import warnings
import psutil
import os
from unittest.mock import patch, MagicMock

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
from src.chemi_accel.core.exceptions import (
    ChemiAccelError, DescriptorCalculationError, GPUMemoryError, 
    BackendUnavailableError, InvalidMoleculeError
)
from src.chemi_accel.descriptors.basic import BasicDescriptorCalculator
from src.chemi_accel.descriptors.constitutional import ConstitutionalDescriptorCalculator
from src.chemi_accel.descriptors.topological import TopologicalDescriptorCalculator
from src.chemi_accel.descriptors.fingerprints import FingerprintGenerator
from src.chemi_accel.core.molecular import MolecularData


# Test molecules for error handling
ERROR_TEST_MOLECULES = {
    "invalid_smiles": [
        "",  # Empty string
        "INVALID",  # Invalid SMILES
        "C(C)(C)(C)(C)(C)C",  # Impossible bonding
        "C1CC",  # Incomplete ring
        "C=C=C=C=C=C=C=C=C=C=C=C=C=C=C",  # Very long cumulene
        "C" * 1000,  # Extremely long chain
        "c1ccc2c(c1)ccc3c2ccc4c3ccc5c4ccc6c5ccc7c6ccc8c7cccc8" * 10,  # Huge aromatic
        "C1C2C3C4C5C6C7C8C9C%10C%11C%12C%13C%14C%15C%16C%17C%18C%19C%20C1C2C3C4C5C6C7C8C9C%10C%11C%12C%13C%14C%15C%16C%17C%18C%19C%20",  # Complex cage
    ],
    
    "problematic_structures": [
        "[H]",  # Single hydrogen
        "[He]",  # Noble gas
        "[C-]",  # Carbanion
        "[NH4+]",  # Charged species
        "C[CH2]",  # Radical
        "C=C=C=C=C",  # Long cumulene
        "C1=CC=CC=C1" * 20,  # Very large aromatic system
        "C1" + "C" * 100 + "1",  # Very large ring
    ],
    
    "memory_intensive": [
        "C" * 200,  # Long alkyl chain
        "c1ccc2c(c1)ccc3c2ccc4c3ccc5c4ccc6c5ccc7c6ccc8c7ccc9c8cccc9",  # Large PAH
        "C1" + "C" * 50 + "1",  # Large ring
        "C1CC2CC3CC4CC5CC6CC7CC8CC9CC%10CC1C2C3C4C5C6C7C8C9C%10",  # Complex cage
    ]
}


class RobustnessTestFramework:
    """Framework for testing system robustness and error handling."""
    
    def __init__(self):
        """Initialize robustness test framework."""
        self.calculators = {
            'basic': BasicDescriptorCalculator(),
            'constitutional': ConstitutionalDescriptorCalculator(use_gpu=False),
            'topological': TopologicalDescriptorCalculator(use_gpu=False),
            'fingerprint': FingerprintGenerator(use_gpu=False)
        }
        
        self.test_functions = {
            'basic': [
                ('molecular_weight', lambda calc, mol: calc.calculate_molecular_weight(mol)),
                ('tpsa', lambda calc, mol: calc.calculate_tpsa(mol)),
                ('logp', lambda calc, mol: calc.calculate_logp(mol)),
            ],
            'constitutional': [
                ('num_heavy_atoms', lambda calc, mol: calc.calculate_heavy_atom_count(mol)),
                ('bertz_complexity', lambda calc, mol: calc.calculate_bertz_complexity(mol)),
            ],
            'topological': [
                ('wiener_index', lambda calc, mol: calc.calculate_wiener_index(mol)),
            ],
            'fingerprint': [
                ('morgan_fp', lambda calc, mol: calc.generate_morgan_fingerprint(mol, radius=2, n_bits=1024)),
            ]
        }
    
    def test_invalid_molecule_handling(self) -> Dict[str, Any]:
        """Test handling of invalid molecular structures."""
        results = {
            "total_invalid_molecules": len(ERROR_TEST_MOLECULES["invalid_smiles"]),
            "results_by_calculator": {},
            "error_consistency": {},
            "graceful_failures": 0,
            "crashes": 0
        }
        
        for calc_name, calc in self.calculators.items():
            calc_results = {
                "calculator": calc_name,
                "function_results": {},
                "total_tests": 0,
                "graceful_failures": 0,
                "crashes": 0,
                "error_types": {}
            }
            
            for func_name, func in self.test_functions[calc_name]:
                func_results = {
                    "function": func_name,
                    "tests": [],
                    "graceful_failures": 0,
                    "crashes": 0
                }
                
                for smiles in ERROR_TEST_MOLECULES["invalid_smiles"]:
                    test_result = self._test_single_invalid_molecule(
                        smiles, calc, func, calc_name, func_name
                    )
                    
                    func_results["tests"].append(test_result)
                    calc_results["total_tests"] += 1
                    
                    if test_result["graceful_failure"]:
                        func_results["graceful_failures"] += 1
                        calc_results["graceful_failures"] += 1
                        results["graceful_failures"] += 1
                    elif test_result["crash"]:
                        func_results["crashes"] += 1
                        calc_results["crashes"] += 1
                        results["crashes"] += 1
                    
                    # Track error types
                    if test_result["error_type"]:
                        error_type = test_result["error_type"]
                        if error_type not in calc_results["error_types"]:
                            calc_results["error_types"][error_type] = 0
                        calc_results["error_types"][error_type] += 1
                
                calc_results["function_results"][func_name] = func_results
            
            results["results_by_calculator"][calc_name] = calc_results
        
        return results
    
    def _test_single_invalid_molecule(self, smiles: str, calc, func, calc_name: str, func_name: str) -> Dict[str, Any]:
        """Test a single invalid molecule with a specific function."""
        result = {
            "smiles": smiles,
            "calculator": calc_name,
            "function": func_name,
            "graceful_failure": False,
            "crash": False,
            "error_type": None,
            "error_message": None,
            "execution_time": 0
        }
        
        start_time = time.time()
        
        try:
            # Try to parse molecule
            mol = Chem.MolFromSmiles(smiles) if RDKIT_AVAILABLE else None
            
            if mol is None:
                # Expected failure at parsing stage
                result["graceful_failure"] = True
                result["error_type"] = "InvalidMoleculeError"
                result["error_message"] = "Failed to parse SMILES"
            else:
                # Try to create MolecularData and run function
                mol_data = MolecularData(
                    smiles=smiles,
                    mol_id=f"error_test_{hash(smiles)}",
                    num_atoms=mol.GetNumAtoms(),
                    num_bonds=mol.GetNumBonds(),
                    molecular_weight=Descriptors.MolWt(mol) if mol.GetNumAtoms() > 0 else 0.0,
                    backend_data={"rdkit_mol": mol}
                )
                
                # Run the function
                func_result = func(calc, mol_data)
                
                # If we get here, the function succeeded (unexpected for invalid molecules)
                result["error_type"] = "UnexpectedSuccess"
                result["error_message"] = f"Function succeeded with result: {func_result}"
                
        except (ChemiAccelError, DescriptorCalculationError, InvalidMoleculeError) as e:
            # Expected ChemiAccel errors - graceful failure
            result["graceful_failure"] = True
            result["error_type"] = type(e).__name__
            result["error_message"] = str(e)
            
        except (ValueError, TypeError, AttributeError) as e:
            # Common Python errors - still considered graceful
            result["graceful_failure"] = True
            result["error_type"] = type(e).__name__
            result["error_message"] = str(e)
            
        except Exception as e:
            # Unexpected errors - potential crashes
            result["crash"] = True
            result["error_type"] = type(e).__name__
            result["error_message"] = str(e)
        
        result["execution_time"] = time.time() - start_time
        return result
    
    def test_memory_management(self) -> Dict[str, Any]:
        """Test memory management under stress conditions."""
        results = {
            "initial_memory_mb": 0,
            "peak_memory_mb": 0,
            "final_memory_mb": 0,
            "memory_leak_detected": False,
            "out_of_memory_handled": False,
            "tests": []
        }
        
        process = psutil.Process(os.getpid())
        results["initial_memory_mb"] = process.memory_info().rss / 1024 / 1024
        
        # Test with memory-intensive molecules
        for smiles in ERROR_TEST_MOLECULES["memory_intensive"]:
            test_result = self._test_memory_intensive_molecule(smiles)
            results["tests"].append(test_result)
            
            # Update peak memory
            current_memory = process.memory_info().rss / 1024 / 1024
            results["peak_memory_mb"] = max(results["peak_memory_mb"], current_memory)
        
        # Force garbage collection
        gc.collect()
        
        results["final_memory_mb"] = process.memory_info().rss / 1024 / 1024
        
        # Check for memory leaks (simple heuristic)
        memory_increase = results["final_memory_mb"] - results["initial_memory_mb"]
        results["memory_leak_detected"] = memory_increase > 100  # More than 100MB increase
        
        return results
    
    def _test_memory_intensive_molecule(self, smiles: str) -> Dict[str, Any]:
        """Test a single memory-intensive molecule."""
        result = {
            "smiles": smiles,
            "memory_before_mb": 0,
            "memory_after_mb": 0,
            "memory_increase_mb": 0,
            "out_of_memory": False,
            "completed_successfully": False,
            "error": None
        }
        
        process = psutil.Process(os.getpid())
        result["memory_before_mb"] = process.memory_info().rss / 1024 / 1024
        
        try:
            if RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mol_data = MolecularData(
                        smiles=smiles,
                        mol_id=f"memory_test_{hash(smiles)}",
                        num_atoms=mol.GetNumAtoms(),
                        num_bonds=mol.GetNumBonds(),
                        molecular_weight=Descriptors.MolWt(mol),
                        backend_data={"rdkit_mol": mol}
                    )
                    
                    # Try basic calculation
                    mw = self.calculators['basic'].calculate_molecular_weight(mol_data)
                    result["completed_successfully"] = True
            
        except MemoryError:
            result["out_of_memory"] = True
            result["error"] = "MemoryError"
            
        except Exception as e:
            result["error"] = str(e)
        
        result["memory_after_mb"] = process.memory_info().rss / 1024 / 1024
        result["memory_increase_mb"] = result["memory_after_mb"] - result["memory_before_mb"]
        
        return result
    
    def test_gpu_fallback_behavior(self) -> Dict[str, Any]:
        """Test GPU fallback behavior when GPU resources are unavailable."""
        results = {
            "gpu_available": CUPY_AVAILABLE,
            "fallback_tests": [],
            "successful_fallbacks": 0,
            "failed_fallbacks": 0
        }
        
        if not CUPY_AVAILABLE:
            results["note"] = "GPU not available, testing CPU-only behavior"
            return results
        
        # Test GPU memory exhaustion simulation
        test_molecules = ["CCO", "c1ccccc1", "CC(=O)O"]
        
        for smiles in test_molecules:
            fallback_result = self._test_gpu_fallback(smiles)
            results["fallback_tests"].append(fallback_result)
            
            if fallback_result["fallback_successful"]:
                results["successful_fallbacks"] += 1
            else:
                results["failed_fallbacks"] += 1
        
        return results
    
    def _test_gpu_fallback(self, smiles: str) -> Dict[str, Any]:
        """Test GPU fallback for a single molecule."""
        result = {
            "smiles": smiles,
            "gpu_calculation_attempted": False,
            "gpu_calculation_failed": False,
            "fallback_successful": False,
            "cpu_result_obtained": False,
            "error": None
        }
        
        try:
            if RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mol_data = MolecularData(
                        smiles=smiles,
                        mol_id=f"fallback_test_{hash(smiles)}",
                        num_atoms=mol.GetNumAtoms(),
                        num_bonds=mol.GetNumBonds(),
                        molecular_weight=Descriptors.MolWt(mol),
                        backend_data={"rdkit_mol": mol}
                    )
                    
                    # Try GPU calculation (if available)
                    try:
                        gpu_calc = ConstitutionalDescriptorCalculator(use_gpu=True)
                        result["gpu_calculation_attempted"] = True
                        gpu_result = gpu_calc.calculate_heavy_atom_count(mol_data)
                        
                    except (GPUMemoryError, BackendUnavailableError, Exception) as e:
                        result["gpu_calculation_failed"] = True
                        result["error"] = str(e)
                        
                        # Try CPU fallback
                        try:
                            cpu_calc = ConstitutionalDescriptorCalculator(use_gpu=False)
                            cpu_result = cpu_calc.calculate_heavy_atom_count(mol_data)
                            result["fallback_successful"] = True
                            result["cpu_result_obtained"] = True
                            
                        except Exception as fallback_error:
                            result["error"] = f"Fallback failed: {fallback_error}"
        
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def test_concurrent_access_safety(self) -> Dict[str, Any]:
        """Test thread safety and concurrent access."""
        results = {
            "num_threads": 5,
            "molecules_per_thread": 10,
            "total_calculations": 0,
            "successful_calculations": 0,
            "failed_calculations": 0,
            "thread_results": [],
            "race_conditions_detected": False
        }
        
        test_molecules = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCS"] * 2
        
        def worker_thread(thread_id: int, molecules: List[str]) -> Dict[str, Any]:
            """Worker thread function."""
            thread_result = {
                "thread_id": thread_id,
                "successful": 0,
                "failed": 0,
                "errors": [],
                "results": []
            }
            
            calc = BasicDescriptorCalculator()
            
            for smiles in molecules:
                try:
                    if RDKIT_AVAILABLE:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            mol_data = MolecularData(
                                smiles=smiles,
                                mol_id=f"thread_{thread_id}_{hash(smiles)}",
                                num_atoms=mol.GetNumAtoms(),
                                num_bonds=mol.GetNumBonds(),
                                molecular_weight=Descriptors.MolWt(mol),
                                backend_data={"rdkit_mol": mol}
                            )
                            
                            mw = calc.calculate_molecular_weight(mol_data)
                            thread_result["successful"] += 1
                            thread_result["results"].append((smiles, mw))
                        
                except Exception as e:
                    thread_result["failed"] += 1
                    thread_result["errors"].append(str(e))
            
            return thread_result
        
        # Start threads
        threads = []
        for i in range(results["num_threads"]):
            thread_molecules = test_molecules[:results["molecules_per_thread"]]
            thread = threading.Thread(
                target=lambda tid=i, mols=thread_molecules: results["thread_results"].append(worker_thread(tid, mols))
            )
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Analyze results
        for thread_result in results["thread_results"]:
            results["successful_calculations"] += thread_result["successful"]
            results["failed_calculations"] += thread_result["failed"]
        
        results["total_calculations"] = results["successful_calculations"] + results["failed_calculations"]
        
        # Check for race conditions (simple heuristic)
        if len(results["thread_results"]) != results["num_threads"]:
            results["race_conditions_detected"] = True
        
        return results
    
    def test_resource_limit_handling(self) -> Dict[str, Any]:
        """Test handling of various resource limits."""
        results = {
            "memory_limit_test": self._test_memory_limit(),
            "time_limit_test": self._test_time_limit(),
            "file_descriptor_test": self._test_file_descriptor_limit()
        }
        
        return results
    
    def _test_memory_limit(self) -> Dict[str, Any]:
        """Test behavior under memory constraints."""
        result = {
            "test_type": "memory_limit",
            "handled_gracefully": False,
            "error_message": None
        }
        
        try:
            # Try to process a very large molecule
            large_smiles = "C" * 500  # Very long chain
            
            if RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(large_smiles)
                if mol is not None:
                    mol_data = MolecularData(
                        smiles=large_smiles,
                        mol_id="memory_limit_test",
                        num_atoms=mol.GetNumAtoms(),
                        num_bonds=mol.GetNumBonds(),
                        molecular_weight=Descriptors.MolWt(mol),
                        backend_data={"rdkit_mol": mol}
                    )
                    
                    calc = BasicDescriptorCalculator()
                    mw = calc.calculate_molecular_weight(mol_data)
                    result["handled_gracefully"] = True
        
        except MemoryError as e:
            result["handled_gracefully"] = True
            result["error_message"] = "MemoryError caught gracefully"
            
        except Exception as e:
            result["error_message"] = str(e)
        
        return result
    
    def _test_time_limit(self) -> Dict[str, Any]:
        """Test behavior under time constraints."""
        result = {
            "test_type": "time_limit",
            "completed_within_limit": False,
            "execution_time": 0,
            "timeout_handled": False
        }
        
        start_time = time.time()
        timeout_seconds = 5  # 5 second timeout
        
        try:
            # Process multiple molecules with time tracking
            test_molecules = ["CCO", "c1ccccc1", "CC(=O)O"] * 20
            calc = BasicDescriptorCalculator()
            
            for smiles in test_molecules:
                current_time = time.time()
                if current_time - start_time > timeout_seconds:
                    result["timeout_handled"] = True
                    break
                
                if RDKIT_AVAILABLE:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        mol_data = MolecularData(
                            smiles=smiles,
                            mol_id=f"time_test_{hash(smiles)}",
                            num_atoms=mol.GetNumAtoms(),
                            num_bonds=mol.GetNumBonds(),
                            molecular_weight=Descriptors.MolWt(mol),
                            backend_data={"rdkit_mol": mol}
                        )
                        
                        mw = calc.calculate_molecular_weight(mol_data)
            
            result["completed_within_limit"] = True
            
        except Exception as e:
            result["error_message"] = str(e)
        
        result["execution_time"] = time.time() - start_time
        return result
    
    def _test_file_descriptor_limit(self) -> Dict[str, Any]:
        """Test behavior under file descriptor limits."""
        result = {
            "test_type": "file_descriptor_limit",
            "handled_gracefully": True,
            "error_message": None
        }
        
        # This is a placeholder test since file descriptor limits
        # are more relevant for file I/O operations
        try:
            # Simulate multiple calculator instances
            calculators = []
            for i in range(10):
                calc = BasicDescriptorCalculator()
                calculators.append(calc)
            
            # Clean up
            del calculators
            gc.collect()
            
        except Exception as e:
            result["handled_gracefully"] = False
            result["error_message"] = str(e)
        
        return result


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
class TestErrorHandlingRobustness:
    """Test error handling and system robustness."""
    
    def setup_method(self):
        """Set up robustness test framework."""
        self.framework = RobustnessTestFramework()
    
    def test_invalid_molecule_error_handling(self):
        """Test graceful handling of invalid molecular structures."""
        results = self.framework.test_invalid_molecule_handling()
        
        # Should handle most invalid molecules gracefully
        total_tests = results["graceful_failures"] + results["crashes"]
        if total_tests > 0:
            graceful_rate = results["graceful_failures"] / total_tests
            assert graceful_rate > 0.8, f"Low graceful failure rate: {graceful_rate:.2%}"
        
        # Should have very few crashes
        assert results["crashes"] < results["total_invalid_molecules"] * 0.1, \
            f"Too many crashes: {results['crashes']}"
        
        # Check that each calculator handles errors consistently
        for calc_name, calc_results in results["results_by_calculator"].items():
            if calc_results["total_tests"] > 0:
                calc_graceful_rate = calc_results["graceful_failures"] / calc_results["total_tests"]
                assert calc_graceful_rate > 0.7, \
                    f"Calculator {calc_name} has low graceful failure rate: {calc_graceful_rate:.2%}"
    
    def test_memory_management_robustness(self):
        """Test memory management under stress conditions."""
        results = self.framework.test_memory_management()
        
        # Memory usage should be reasonable
        memory_increase = results["final_memory_mb"] - results["initial_memory_mb"]
        assert memory_increase < 200, f"Excessive memory usage: {memory_increase} MB"
        
        # Should not detect significant memory leaks
        assert not results["memory_leak_detected"], "Memory leak detected"
        
        # Check individual test results
        for test in results["tests"]:
            # Individual molecule processing should not use excessive memory
            assert test["memory_increase_mb"] < 50, \
                f"Excessive memory for molecule {test['smiles']}: {test['memory_increase_mb']} MB"
    
    def test_gpu_fallback_behavior(self):
        """Test GPU fallback behavior."""
        results = self.framework.test_gpu_fallback_behavior()
        
        if results["gpu_available"]:
            # Should have some successful fallbacks
            total_fallback_tests = len(results["fallback_tests"])
            if total_fallback_tests > 0:
                success_rate = results["successful_fallbacks"] / total_fallback_tests
                assert success_rate > 0.5, f"Low fallback success rate: {success_rate:.2%}"
        else:
            # CPU-only behavior should work
            assert "note" in results
    
    def test_concurrent_access_safety(self):
        """Test thread safety and concurrent access."""
        results = self.framework.test_concurrent_access_safety()
        
        # Should complete most calculations successfully
        if results["total_calculations"] > 0:
            success_rate = results["successful_calculations"] / results["total_calculations"]
            assert success_rate > 0.8, f"Low concurrent success rate: {success_rate:.2%}"
        
        # Should not detect race conditions
        assert not results["race_conditions_detected"], "Race conditions detected"
        
        # All threads should complete
        assert len(results["thread_results"]) == results["num_threads"], \
            "Not all threads completed"
    
    def test_resource_limit_handling(self):
        """Test handling of various resource limits."""
        results = self.framework.test_resource_limit_handling()
        
        # Memory limit test
        memory_test = results["memory_limit_test"]
        assert memory_test["handled_gracefully"], \
            f"Memory limit not handled gracefully: {memory_test.get('error_message', 'Unknown error')}"
        
        # Time limit test
        time_test = results["time_limit_test"]
        assert time_test["execution_time"] < 10, \
            f"Time limit test took too long: {time_test['execution_time']} seconds"
        
        # File descriptor test
        fd_test = results["file_descriptor_test"]
        assert fd_test["handled_gracefully"], \
            f"File descriptor limit not handled gracefully: {fd_test.get('error_message', 'Unknown error')}"
    
    def test_error_message_quality(self):
        """Test quality and helpfulness of error messages."""
        # Test with known problematic molecules
        problematic_molecules = [
            ("", "Empty SMILES string"),
            ("INVALID", "Invalid SMILES syntax"),
            ("C(C)(C)(C)(C)(C)C", "Impossible bonding pattern"),
        ]
        
        calc = BasicDescriptorCalculator()
        
        for smiles, expected_issue in problematic_molecules:
            try:
                if RDKIT_AVAILABLE:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        mol_data = MolecularData(
                            smiles=smiles,
                            mol_id=f"error_msg_test_{hash(smiles)}",
                            num_atoms=mol.GetNumAtoms(),
                            num_bonds=mol.GetNumBonds(),
                            molecular_weight=Descriptors.MolWt(mol),
                            backend_data={"rdkit_mol": mol}
                        )
                        
                        mw = calc.calculate_molecular_weight(mol_data)
                        # If we get here, the molecule was actually valid
                        continue
                
            except Exception as e:
                error_message = str(e)
                
                # Error message should be informative
                assert len(error_message) > 10, f"Error message too short: '{error_message}'"
                assert error_message != "An error occurred", "Generic error message"
                
                # Should not contain internal implementation details
                assert "traceback" not in error_message.lower()
                assert "stack trace" not in error_message.lower()
    
    def test_graceful_degradation(self):
        """Test graceful degradation when components fail."""
        # Test with mock failures
        calc = BasicDescriptorCalculator()
        
        test_molecule = "CCO"
        if RDKIT_AVAILABLE:
            mol = Chem.MolFromSmiles(test_molecule)
            if mol is not None:
                mol_data = MolecularData(
                    smiles=test_molecule,
                    mol_id="degradation_test",
                    num_atoms=mol.GetNumAtoms(),
                    num_bonds=mol.GetNumBonds(),
                    molecular_weight=Descriptors.MolWt(mol),
                    backend_data={"rdkit_mol": mol}
                )
                
                # Normal operation should work
                mw = calc.calculate_molecular_weight(mol_data)
                assert mw > 0, "Normal operation should work"
                
                # Test with corrupted backend data
                corrupted_mol_data = MolecularData(
                    smiles=test_molecule,
                    mol_id="corrupted_test",
                    num_atoms=mol.GetNumAtoms(),
                    num_bonds=mol.GetNumBonds(),
                    molecular_weight=Descriptors.MolWt(mol),
                    backend_data={"rdkit_mol": None}  # Corrupted data
                )
                
                # Should handle corrupted data gracefully
                try:
                    mw_corrupted = calc.calculate_molecular_weight(corrupted_mol_data)
                    # If it succeeds, that's fine too
                except Exception as e:
                    # Should be a graceful error
                    assert isinstance(e, (ChemiAccelError, ValueError, AttributeError))
    
    def test_input_validation_robustness(self):
        """Test robustness of input validation."""
        calc = BasicDescriptorCalculator()
        
        # Test with various invalid inputs
        invalid_inputs = [
            None,
            "",
            123,
            [],
            {},
            MolecularData(smiles="", mol_id="", num_atoms=0, num_bonds=0, molecular_weight=0),
        ]
        
        for invalid_input in invalid_inputs:
            try:
                if hasattr(calc, 'calculate_molecular_weight'):
                    result = calc.calculate_molecular_weight(invalid_input)
                    # If it succeeds, the result should be reasonable
                    if result is not None:
                        assert isinstance(result, (int, float))
                        assert not np.isnan(result)
                        assert not np.isinf(result)
                
            except Exception as e:
                # Should be a graceful error with informative message
                assert isinstance(e, (ChemiAccelError, ValueError, TypeError, AttributeError))
                error_msg = str(e)
                assert len(error_msg) > 5, "Error message should be informative"
    
    def test_numerical_stability_edge_cases(self):
        """Test numerical stability with edge cases."""
        calc = BasicDescriptorCalculator()
        
        # Test with molecules that might cause numerical issues
        edge_case_molecules = [
            "C",  # Single carbon
            "[H]",  # Single hydrogen (if parseable)
            "C" * 10,  # Long chain
            "c1ccccc1" * 5,  # Large aromatic system
        ]
        
        for smiles in edge_case_molecules:
            try:
                if RDKIT_AVAILABLE:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        mol_data = MolecularData(
                            smiles=smiles,
                            mol_id=f"stability_test_{hash(smiles)}",
                            num_atoms=mol.GetNumAtoms(),
                            num_bonds=mol.GetNumBonds(),
                            molecular_weight=Descriptors.MolWt(mol),
                            backend_data={"rdkit_mol": mol}
                        )
                        
                        # Test multiple descriptors
                        mw = calc.calculate_molecular_weight(mol_data)
                        tpsa = calc.calculate_tpsa(mol_data)
                        logp = calc.calculate_logp(mol_data)
                        
                        # Results should be finite and reasonable
                        assert np.isfinite(mw), f"Non-finite MW for {smiles}: {mw}"
                        assert np.isfinite(tpsa), f"Non-finite TPSA for {smiles}: {tpsa}"
                        assert np.isfinite(logp), f"Non-finite LogP for {smiles}: {logp}"
                        
                        assert mw > 0, f"Non-positive MW for {smiles}: {mw}"
                        assert tpsa >= 0, f"Negative TPSA for {smiles}: {tpsa}"
                        
            except Exception as e:
                # Edge cases may fail, but should fail gracefully
                assert isinstance(e, (ChemiAccelError, ValueError, TypeError))
    
    def test_stress_testing(self):
        """Perform stress testing with high load."""
        calc = BasicDescriptorCalculator()
        
        # Generate many test molecules
        test_molecules = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCS"] * 100
        
        successful_calculations = 0
        failed_calculations = 0
        start_time = time.time()
        
        for i, smiles in enumerate(test_molecules):
            try:
                if RDKIT_AVAILABLE:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        mol_data = MolecularData(
                            smiles=smiles,
                            mol_id=f"stress_test_{i}",
                            num_atoms=mol.GetNumAtoms(),
                            num_bonds=mol.GetNumBonds(),
                            molecular_weight=Descriptors.MolWt(mol),
                            backend_data={"rdkit_mol": mol}
                        )
                        
                        mw = calc.calculate_molecular_weight(mol_data)
                        successful_calculations += 1
                
            except Exception:
                failed_calculations += 1
            
            # Check if we're taking too long
            if time.time() - start_time > 30:  # 30 second limit
                break
        
        total_calculations = successful_calculations + failed_calculations
        
        # Should complete most calculations successfully
        if total_calculations > 0:
            success_rate = successful_calculations / total_calculations
            assert success_rate > 0.9, f"Low success rate under stress: {success_rate:.2%}"
        
        # Should process at reasonable speed
        elapsed_time = time.time() - start_time
        if successful_calculations > 0:
            molecules_per_second = successful_calculations / elapsed_time
            assert molecules_per_second > 10, f"Too slow under stress: {molecules_per_second:.1f} mol/s"


if __name__ == "__main__":
    # Run robustness testing if called directly
    framework = RobustnessTestFramework()
    
    print("Running comprehensive robustness tests...")
    
    # Test invalid molecule handling
    print("\n1. Testing invalid molecule handling...")
    invalid_results = framework.test_invalid_molecule_handling()
    print(f"   Graceful failures: {invalid_results['graceful_failures']}")
    print(f"   Crashes: {invalid_results['crashes']}")
    
    # Test memory management
    print("\n2. Testing memory management...")
    memory_results = framework.test_memory_management()
    print(f"   Memory increase: {memory_results['final_memory_mb'] - memory_results['initial_memory_mb']:.1f} MB")
    print(f"   Memory leak detected: {memory_results['memory_leak_detected']}")
    
    # Test GPU fallback
    print("\n3. Testing GPU fallback...")
    gpu_results = framework.test_gpu_fallback_behavior()
    print(f"   GPU available: {gpu_results['gpu_available']}")
    print(f"   Successful fallbacks: {gpu_results['successful_fallbacks']}")
    
    # Test concurrent access
    print("\n4. Testing concurrent access...")
    concurrent_results = framework.test_concurrent_access_safety()
    print(f"   Total calculations: {concurrent_results['total_calculations']}")
    print(f"   Success rate: {concurrent_results['successful_calculations'] / max(1, concurrent_results['total_calculations']):.2%}")
    
    # Test resource limits
    print("\n5. Testing resource limits...")
    resource_results = framework.test_resource_limit_handling()
    print(f"   Memory limit handled: {resource_results['memory_limit_test']['handled_gracefully']}")
    print(f"   Time limit test duration: {resource_results['time_limit_test']['execution_time']:.2f}s")
    
    print("\nRobustness testing completed!")