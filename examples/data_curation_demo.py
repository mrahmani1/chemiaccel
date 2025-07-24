"""
Comprehensive Demo for GPU-Accelerated Data Curation Pipeline - Task 20.1

This demo showcases the complete GPU-accelerated molecular data curation pipeline
including SMILES validation, salt stripping, neutralization, tautomer standardization,
InChIKey generation, deduplication, and PAINS filtering.

Features Demonstrated:
- High-performance parallel processing with multiple backends
- GPU acceleration for computational bottlenecks
- Comprehensive molecular standardization workflow
- Large-scale dataset processing capabilities
- Performance monitoring and optimization
- Integration with existing ChemiAccel infrastructure
"""

import time
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chemi_accel.data_curation.gpu_molecular_cleaning import (
    GPUMolecularCleaningPipeline, CleaningConfig
)
from chemi_accel.data_curation.parallel_validation import ValidationConfig
from chemi_accel.data_curation.tautomer_standardization import TautomerStandardizationConfig
from chemi_accel.data_curation.inchikey_deduplication import InChIKeyConfig, DeduplicationStrategy
from chemi_accel.data_curation.pains_substructure_filter import FilterConfig, FilterType, FilterRule, FilterAction
from chemi_accel.utils.logging import get_logger

logger = get_logger()


class DataCurationDemo:
    """Comprehensive demo of the data curation pipeline."""
    
    def __init__(self):
        """Initialize the demo."""
        self.setup_logging()
        logger.info("Starting ChemiAccel Data Curation Pipeline Demo")
        
        # Sample molecular dataset for demonstration
        self.sample_molecules = [
            # Clean molecules
            "CCO",  # Ethanol
            "c1ccccc1",  # Benzene
            "CC(=O)O",  # Acetic acid
            "CCN",  # Ethylamine
            "c1ccc(cc1)O",  # Phenol
            
            # Molecules with salts
            "CCO.[Na+].[Cl-]",  # Ethanol sodium chloride
            "CCN.[HCl]",  # Ethylamine hydrochloride
            "c1ccc(cc1)O.[K+].[Br-]",  # Phenol potassium bromide
            
            # Molecules needing neutralization
            "CC[NH3+]",  # Protonated ethylamine
            "c1ccc(cc1)[O-]",  # Phenoxide anion
            "CC(=O)[O-]",  # Acetate anion
            
            # Tautomers
            "CC(=O)CC",  # Acetone (keto form)
            "CC(O)=CC",  # Acetone (enol form)
            "c1[nH]c2ccccc2c1",  # Indole
            
            # Stereochemistry variations
            "C[C@H](O)C",  # (R)-2-propanol
            "C[C@@H](O)C",  # (S)-2-propanol
            "C/C=C/C",  # trans-2-butene
            "C/C=C\\C",  # cis-2-butene
            
            # Duplicates (same connectivity)
            "CCO",  # Duplicate ethanol
            "C(C)O",  # Different representation of ethanol
            "OCC",  # Another ethanol representation
            
            # PAINS molecules (simplified examples)
            "c1ccc2c(c1)c(=O)c3ccccc3c2=O",  # Anthraquinone (potential PAINS)
            "CC(=O)c1ccc(cc1)N(=O)=O",  # Nitroaromatic compound
            
            # Problem molecules
            "invalid_smiles",  # Invalid SMILES
            "",  # Empty string
            "C" * 100,  # Very long chain
            
            # Drug-like molecules
            "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # Ibuprofen
            "CC1=CC=C(C=C1)C(=O)O",  # p-Toluic acid
            "CN1CCC[C@H]1c2cccnc2",  # Nicotine
        ]
        
        # Extend with more complex examples
        self.complex_molecules = [
            # Natural products
            "CC1=C(C(CCC1)(C)C)CC/C(=C/CO)/C",  # Vitamin A precursor
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            
            # Pharmaceutical compounds
            "CC1=C(C=C(C=C1)C(=O)NC2=CC=C(C=C2)O)CC(C)C",  # Drug-like molecule
            "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O",  # β-agonist structure
            
            # Complex stereochemistry
            "C[C@H]1[C@@H]([C@H]([C@@H]([C@H](O1)O[C@@H]2[C@H](O[C@H]([C@@H]([C@H]2O)O)O)CO)O)O)O",  # Sugar
            
            # Large molecules
            "CC(C)CC1=CC=C(C=C1)[C@@H](C)C(=O)O" * 2,  # Dimer example
        ]
    
    def setup_logging(self):
        """Setup logging for the demo."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def create_sample_dataset(self, size: int = 1000, include_complex: bool = True) -> List[str]:
        """Create a sample dataset for testing.
        
        Args:
            size: Number of molecules in dataset
            include_complex: Whether to include complex molecules
            
        Returns:
            List of SMILES strings
        """
        logger.info(f"Creating sample dataset of {size} molecules")
        
        base_molecules = self.sample_molecules.copy()
        if include_complex:
            base_molecules.extend(self.complex_molecules)
        
        # Replicate and add variations to reach desired size
        dataset = []
        for i in range(size):
            mol = base_molecules[i % len(base_molecules)]
            
            # Add some variations
            if i % 10 == 0 and mol.startswith("c1ccccc1"):  # Benzene variations
                substituents = ["C", "O", "N", "Cl", "F", "Br"]
                sub = substituents[i % len(substituents)]
                mol = mol.replace("c1ccccc1", f"c1ccc(cc1){sub}")
            
            dataset.append(mol)
        
        # Shuffle for realistic distribution
        np.random.shuffle(dataset)
        
        logger.info(f"Created dataset with {len(dataset)} molecules")
        return dataset
    
    def demo_basic_pipeline(self):
        """Demonstrate basic pipeline functionality."""
        logger.info("=== Basic Pipeline Demo ===")
        
        # Create a small dataset for quick demonstration
        molecules = self.sample_molecules[:20]
        
        # Create basic configuration
        config = CleaningConfig(
            batch_size=10,
            enable_smiles_validation=True,
            enable_salt_removal=True,
            enable_neutralization=True,
            enable_tautomer_standardization=True,
            enable_inchikey_deduplication=True,
            enable_pains_filtering=True,
            parallel_backend="threading",
            use_gpu_acceleration=True
        )
        
        # Initialize pipeline
        pipeline = GPUMolecularCleaningPipeline(config)
        
        # Process molecules
        logger.info(f"Processing {len(molecules)} molecules...")
        start_time = time.time()
        
        cleaned_df, stats = pipeline.clean_molecular_dataset(molecules)
        
        processing_time = time.time() - start_time
        
        # Display results
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        logger.info(f"Input molecules: {stats.total_processed}")
        logger.info(f"Valid molecules: {stats.valid_molecules}")
        logger.info(f"Cleaned molecules: {len(cleaned_df)}")
        logger.info(f"Success rate: {stats.success_rate:.1%}")
        
        # Show sample results
        if len(cleaned_df) > 0:
            logger.info("\nSample cleaned molecules:")
            for i, row in cleaned_df.head().iterrows():
                logger.info(f"  {row.get('original_smiles', 'N/A')} -> {row.get('canonical_smiles', 'N/A')}")
        
        return cleaned_df, stats
    
    def demo_component_configurations(self):
        """Demonstrate individual component configurations."""
        logger.info("=== Component Configuration Demo ===")
        
        molecules = self.sample_molecules
        
        # Validation configuration
        validation_config = ValidationConfig(
            parallel_backend="threading",
            max_workers=4,
            cache_enabled=True,
            strict_mode=False
        )
        
        # Tautomer standardization configuration
        tautomer_config = TautomerStandardizationConfig(
            max_tautomers=100,
            include_stereochemistry=True,
            parallel_backend="threading",
            cache_enabled=True
        )
        
        # InChIKey generation configuration
        inchikey_config = InChIKeyConfig(
            deduplication_strategy=DeduplicationStrategy.INCHIKEY_ONLY,
            include_stereochemistry=True,
            parallel_backend="threading",
            cache_enabled=True
        )
        
        # PAINS filtering configuration
        filter_config = FilterConfig(
            enabled_filters=[
                FilterType.PAINS_A,
                FilterType.PAINS_B,
                FilterType.LIPINSKI_RULE_OF_5,
                FilterType.VEBER
            ],
            calculate_drug_properties=True,
            parallel_backend="threading",
            cache_enabled=True
        )
        
        # Create comprehensive configuration
        config = CleaningConfig(
            batch_size=20,
            validation_config=validation_config,
            tautomer_config=tautomer_config,
            inchikey_config=inchikey_config,
            filter_config=filter_config,
            include_intermediate_results=True,
            save_statistics=True
        )
        
        # Process with detailed configuration
        pipeline = GPUMolecularCleaningPipeline(config)
        
        logger.info(f"Processing {len(molecules)} molecules with detailed configuration...")
        cleaned_df, stats = pipeline.clean_molecular_dataset(molecules)
        
        # Display detailed statistics
        logger.info("\nDetailed Processing Statistics:")
        logger.info(f"  Total processed: {stats.total_processed}")
        logger.info(f"  Valid SMILES: {stats.valid_smiles}")
        logger.info(f"  Salt removed: {stats.salts_removed}")
        logger.info(f"  Neutralized: {stats.neutralized}")
        logger.info(f"  Tautomers standardized: {stats.tautomers_standardized}")
        logger.info(f"  Duplicates removed: {stats.duplicates_removed}")
        logger.info(f"  PAINS filtered: {stats.pains_filtered}")
        
        return cleaned_df, stats
    
    def demo_performance_scaling(self):
        """Demonstrate performance scaling with different dataset sizes."""
        logger.info("=== Performance Scaling Demo ===")
        
        # Test different dataset sizes
        sizes = [100, 500, 1000, 2000]
        results = {}
        
        # Standard configuration for performance testing
        config = CleaningConfig(
            batch_size=500,
            parallel_backend="threading",
            max_workers=4,
            use_gpu_acceleration=True,
            cache_enabled=True
        )
        
        pipeline = GPUMolecularCleaningPipeline(config)
        
        for size in sizes:
            logger.info(f"Testing performance with {size} molecules...")
            
            # Create dataset
            molecules = self.create_sample_dataset(size, include_complex=True)
            
            # Measure processing time
            start_time = time.time()
            cleaned_df, stats = pipeline.clean_molecular_dataset(molecules)
            processing_time = time.time() - start_time
            
            # Calculate metrics
            throughput = size / processing_time if processing_time > 0 else 0
            
            results[size] = {
                'processing_time': processing_time,
                'throughput': throughput,
                'success_rate': stats.success_rate,
                'clean_molecules': len(cleaned_df)
            }
            
            logger.info(f"  Size: {size}, Time: {processing_time:.2f}s, "
                       f"Throughput: {throughput:.1f} mol/s, Success: {stats.success_rate:.1%}")
        
        # Display scaling analysis
        logger.info("\nScaling Analysis:")
        base_size = sizes[0]
        base_time = results[base_size]['processing_time']
        
        for size in sizes[1:]:
            expected_time = base_time * (size / base_size)
            actual_time = results[size]['processing_time']
            scaling_efficiency = expected_time / actual_time if actual_time > 0 else 0
            
            logger.info(f"  {size} molecules: {scaling_efficiency:.2f}x scaling efficiency")
        
        return results
    
    def demo_advanced_filtering(self):
        """Demonstrate advanced filtering capabilities."""
        logger.info("=== Advanced Filtering Demo ===")
        
        # Create molecules with known PAINS patterns
        pains_molecules = [
            "c1ccc2c(c1)c(=O)c3ccccc3c2=O",  # Anthraquinone
            "CC(=O)c1ccc(cc1)N(=O)=O",  # Nitroaromatic
            "c1ccc(cc1)N=Nc2ccccc2",  # Azobenzene
            "c1ccc(cc1)S(=O)(=O)c2ccccc2",  # Sulfone
        ]
        
        # Create custom filter rules
        custom_rules = [
            FilterRule(
                name="nitro_groups",
                filter_type=FilterType.CUSTOM_SMARTS,
                action=FilterAction.FLAG,
                pattern="[N+](=O)[O-]",
                description="Flag molecules with nitro groups"
            ),
            FilterRule(
                name="sulfonyl_groups",
                filter_type=FilterType.CUSTOM_SMARTS,
                action=FilterAction.EXCLUDE,
                pattern="S(=O)(=O)",
                description="Exclude molecules with sulfonyl groups"
            )
        ]
        
        # Advanced filter configuration
        filter_config = FilterConfig(
            enabled_filters=[
                FilterType.PAINS_A,
                FilterType.PAINS_B,
                FilterType.LIPINSKI_RULE_OF_5,
                FilterType.VEBER,
                FilterType.EGAN
            ],
            custom_filter_rules=custom_rules,
            calculate_drug_properties=True,
            parallel_backend="threading"
        )
        
        config = CleaningConfig(
            filter_config=filter_config,
            enable_pains_filtering=True
        )
        
        # Test molecules including PAINS
        test_molecules = self.sample_molecules + pains_molecules
        
        pipeline = GPUMolecularCleaningPipeline(config)
        
        logger.info(f"Testing advanced filtering on {len(test_molecules)} molecules...")
        cleaned_df, stats = pipeline.clean_molecular_dataset(test_molecules)
        
        logger.info(f"Advanced filtering results:")
        logger.info(f"  Input molecules: {len(test_molecules)}")
        logger.info(f"  Passed filters: {len(cleaned_df)}")
        logger.info(f"  PAINS filtered: {stats.pains_filtered}")
        logger.info(f"  Custom filters applied: {len(custom_rules)}")
        
        # Show which molecules were filtered
        if hasattr(stats, 'filter_details'):
            for filter_name, count in stats.filter_details.items():
                logger.info(f"  {filter_name}: {count} molecules")
        
        return cleaned_df, stats
    
    def demo_large_scale_processing(self, size: int = 10000):
        """Demonstrate large-scale processing capabilities."""
        logger.info(f"=== Large Scale Processing Demo ({size:,} molecules) ===")
        
        # Create large dataset
        logger.info("Generating large dataset...")
        molecules = self.create_sample_dataset(size, include_complex=True)
        
        # Optimized configuration for large-scale processing
        config = CleaningConfig(
            batch_size=1000,
            parallel_backend="threading",
            max_workers=8,
            use_gpu_acceleration=True,
            cache_enabled=True,
            enable_streaming=True,
            memory_limit_gb=8.0
        )
        
        pipeline = GPUMolecularCleaningPipeline(config)
        
        # Process with progress monitoring
        logger.info(f"Processing {len(molecules):,} molecules...")
        start_time = time.time()
        
        cleaned_df, stats = pipeline.clean_molecular_dataset(
            molecules,
            smiles_column='smiles',
            output_file=f"large_scale_results_{size}.csv"
        )
        
        processing_time = time.time() - start_time
        
        # Calculate performance metrics
        throughput = len(molecules) / processing_time if processing_time > 0 else 0
        memory_efficiency = len(cleaned_df) / (config.memory_limit_gb * 1024)  # molecules per MB
        
        logger.info(f"Large-scale processing results:")
        logger.info(f"  Total time: {processing_time:.2f} seconds ({processing_time/60:.1f} minutes)")
        logger.info(f"  Throughput: {throughput:.1f} molecules/second")
        logger.info(f"  Success rate: {stats.success_rate:.1%}")
        logger.info(f"  Memory efficiency: {memory_efficiency:.1f} mol/MB")
        logger.info(f"  Output file: large_scale_results_{size}.csv")
        
        # Extrapolate to 100M dataset
        estimated_100m_time = (100_000_000 / throughput) / 3600 if throughput > 0 else float('inf')
        logger.info(f"  Estimated time for 100M molecules: {estimated_100m_time:.1f} hours")
        
        return cleaned_df, stats, throughput
    
    def demo_error_handling(self):
        """Demonstrate error handling and recovery."""
        logger.info("=== Error Handling Demo ===")
        
        # Create dataset with various error conditions
        error_molecules = [
            "CCO",  # Valid
            "invalid_smiles",  # Invalid SMILES
            "",  # Empty string
            "C" * 200,  # Very long molecule
            "CC(C)(C)(C)(C)(C)(C)(C)(C)(C)C",  # Highly branched
            "c1ccccc1",  # Valid
            "[Pt]",  # Metal
            "C1=CC=CC=C1",  # Alternative benzene representation
            None,  # None value
            123,  # Non-string
        ]
        
        # Configure for error handling
        config = CleaningConfig(
            batch_size=5,
            parallel_backend="threading",
            use_gpu_acceleration=False,  # Disable GPU for error testing
            cache_enabled=False  # Disable cache to see all errors
        )
        
        pipeline = GPUMolecularCleaningPipeline(config)
        
        logger.info(f"Processing {len(error_molecules)} molecules with error conditions...")
        
        try:
            cleaned_df, stats = pipeline.clean_molecular_dataset(error_molecules)
            
            logger.info("Error handling results:")
            logger.info(f"  Input molecules: {len(error_molecules)}")
            logger.info(f"  Successfully processed: {len(cleaned_df)}")
            logger.info(f"  Errors encountered: {stats.total_processed - stats.valid_molecules}")
            logger.info(f"  Error rate: {(1 - stats.success_rate):.1%}")
            
            if hasattr(stats, 'error_details'):
                logger.info("  Error breakdown:")
                for error_type, count in stats.error_details.items():
                    logger.info(f"    {error_type}: {count}")
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            logger.info("This demonstrates the robustness of error handling")
        
        return True
    
    def run_full_demo(self):
        """Run the complete demonstration."""
        logger.info("Starting ChemiAccel Data Curation Pipeline Comprehensive Demo")
        logger.info("=" * 70)
        
        try:
            # Basic functionality
            self.demo_basic_pipeline()
            
            # Component configurations
            self.demo_component_configurations()
            
            # Performance scaling
            self.demo_performance_scaling()
            
            # Advanced filtering
            self.demo_advanced_filtering()
            
            # Error handling
            self.demo_error_handling()
            
            # Large scale processing (smaller size for demo)
            self.demo_large_scale_processing(size=5000)
            
            logger.info("=" * 70)
            logger.info("Demo completed successfully!")
            logger.info("All pipeline components demonstrated:")
            logger.info("  ✓ SMILES validation and cleaning")
            logger.info("  ✓ Salt removal and neutralization")
            logger.info("  ✓ Tautomer standardization")
            logger.info("  ✓ InChIKey generation and deduplication")
            logger.info("  ✓ PAINS and custom filtering")
            logger.info("  ✓ GPU acceleration and parallel processing")
            logger.info("  ✓ Large-scale dataset handling")
            logger.info("  ✓ Error handling and recovery")
            
        except Exception as e:
            logger.error(f"Demo failed with error: {e}")
            raise


def main():
    """Main function to run the demo."""
    demo = DataCurationDemo()
    demo.run_full_demo()


if __name__ == "__main__":
    main()