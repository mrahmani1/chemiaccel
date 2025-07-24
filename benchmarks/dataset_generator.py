"""
Benchmark Dataset Generator for ChemiAccel.

This module generates standardized benchmark datasets with diverse molecular
structures for comprehensive performance testing.
"""

import logging
import random
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
    np = None

try:
    from ..src.chemi_accel.utils.logging import get_logger
except ImportError:
    # Fallback for standalone execution
    import logging
    def get_logger():
        return logging.getLogger(__name__)

logger = get_logger()


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    name: str
    description: str
    sizes: List[int] = field(default_factory=lambda: [100, 1000, 10000, 100000])
    molecular_weight_range: Tuple[float, float] = (50.0, 1000.0)
    complexity_distribution: Dict[str, float] = field(default_factory=lambda: {
        'simple': 0.3,      # Small molecules (< 20 atoms)
        'medium': 0.5,      # Medium molecules (20-50 atoms)
        'complex': 0.2      # Large molecules (> 50 atoms)
    })
    include_polymers: bool = False
    polymer_fraction: float = 0.1
    include_drug_like: bool = True
    drug_like_fraction: float = 0.4
    include_natural_products: bool = True
    natural_products_fraction: float = 0.2
    random_seed: int = 42
    output_directory: str = "benchmark_datasets"


@dataclass
class MolecularDataset:
    """Container for molecular dataset."""
    name: str
    size: int
    smiles: List[str]
    properties: Dict[str, List[Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_pandas(self) -> Optional[pd.DataFrame]:
        """Convert to pandas DataFrame."""
        if not PANDAS_AVAILABLE:
            return None
        
        data = {'smiles': self.smiles}
        data.update(self.properties)
        return pd.DataFrame(data)
    
    def save(self, filepath: Path) -> None:
        """Save dataset to file."""
        data = {
            'name': self.name,
            'size': self.size,
            'smiles': self.smiles,
            'properties': self.properties,
            'metadata': self.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved dataset {self.name} to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'MolecularDataset':
        """Load dataset from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        dataset = cls(
            name=data['name'],
            size=data['size'],
            smiles=data['smiles'],
            properties=data.get('properties', {}),
            metadata=data.get('metadata', {})
        )
        
        logger.info(f"Loaded dataset {dataset.name} from {filepath}")
        return dataset


class BenchmarkDatasetGenerator:
    """Generates standardized benchmark datasets for performance testing."""
    
    def __init__(self, config: DatasetConfig):
        """Initialize dataset generator.
        
        Args:
            config: Dataset generation configuration
        """
        self.config = config
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed for reproducibility
        random.seed(config.random_seed)
        if np is not None:
            np.random.seed(config.random_seed)
        
        # Load molecular templates
        self._load_molecular_templates()
        
        logger.info(f"Initialized dataset generator: {config.name}")
    
    def _load_molecular_templates(self) -> None:
        """Load molecular structure templates for dataset generation."""
        # Simple molecular templates (SMILES strings)
        self.simple_templates = [
            "CCO",  # Ethanol
            "CC(C)O",  # Isopropanol
            "CC(=O)O",  # Acetic acid
            "c1ccccc1",  # Benzene
            "CCc1ccccc1",  # Ethylbenzene
            "CC(C)(C)O",  # tert-Butanol
            "CCCCO",  # Butanol
            "CC(C)CO",  # Isobutanol
            "CCC(=O)O",  # Propanoic acid
            "c1ccc(O)cc1",  # Phenol
        ]
        
        # Medium complexity templates
        self.medium_templates = [
            "CC(C)(C)c1ccc(O)cc1",  # 4-tert-Butylphenol
            "COc1ccc(C(=O)O)cc1",  # 4-Methoxybenzoic acid
            "CC1=CC(=O)C=C(C)C1=O",  # 2,5-Dimethyl-1,4-benzoquinone
            "c1ccc2c(c1)ccc3c2cccc3",  # Anthracene
            "CC(C)(C)OC(=O)NC1CCCCC1",  # tert-Butyl cyclohexylcarbamate
            "COc1ccc(CCN)cc1OC",  # 3,4-Dimethoxyphenethylamine
            "CC1=C(C(=O)N(C1=O)C)C",  # 1,3-Dimethyl-2,4-dioxo-1,2,3,4-tetrahydropyrimidine
            "c1ccc(Cc2ccccc2)cc1",  # Diphenylmethane
            "CC(C)C(=O)Nc1ccccc1",  # N-Phenyl-2-methylpropanamide
            "COc1cc(C=O)ccc1O",  # Vanillin
        ]
        
        # Complex molecular templates
        self.complex_templates = [
            "CC1=C2C(=CC=C1)N(C3=C(N2)C=CC(=C3)C(=O)NCCCCN)C",  # Complex heterocycle
            "COc1cc(C2C(CO2)c3ccc(OC)c(OC)c3)cc(OC)c1OC",  # Complex natural product-like
            "CC(C)(C)OC(=O)N1CCC(CC1)N2CCN(CC2)c3ccc(Cl)cc3",  # Complex pharmaceutical-like
            "COc1ccc2c(c1)c(CC(=O)NCCCCN)c(C)n2C(=O)c4ccccc4",  # Complex indole derivative
            "CC1=C(C(=O)N(C1=O)C2CCCCC2)C(=O)Nc3ccc(Cl)cc3",  # Complex amide
            "COc1cc(C2=COc3c(C2=O)cc(OC)cc3OC)cc(OC)c1OC",  # Complex flavonoid-like
            "CC(C)(C)c1ccc(C(=O)N2CCN(CC2)c3ncccn3)cc1",  # Complex pyrimidine derivative
            "COc1ccc(C2CC(=O)c3c(O2)cc(OC)cc3OC)cc1OC",  # Complex chromone
            "CC1=C(C(=O)N(C1=O)C2CCC(CC2)N3CCOCC3)C",  # Complex morpholine derivative
            "COc1cc(C2=C(c3ccc(OC)cc3)C(=O)c4ccccc4C2=O)cc(OC)c1OC",  # Complex quinone
        ]
        
        # Drug-like molecules (based on known drugs)
        self.drug_like_templates = [
            "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # Ibuprofen-like
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin-like
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine-like
            "COc1cc2c(cc1OC)C(=O)C(CC2)c3ccc(OC)c(OC)c3",  # Natural product-like
            "CC(C)(C)NCC(c1ccc(O)cc1)O",  # Beta-blocker-like
            "COc1ccc(CCN(C)C)cc1",  # Phenethylamine-like
            "CC1=C(C(=O)N(C1=O)C)C(=O)NCCCCN",  # Anticonvulsant-like
            "COc1ccc(C(=O)NCCN2CCOCC2)cc1",  # Antiemetic-like
            "CC(C)NCC(c1ccc(O)c(CO)c1)O",  # Bronchodilator-like
            "COc1cc(C2CC(=O)Nc3ccccc32)cc(OC)c1OC",  # Alkaloid-like
        ]
        
        # Polymer building blocks
        self.polymer_templates = [
            "C=C",  # Ethylene
            "C=CC",  # Propylene
            "C=Cc1ccccc1",  # Styrene
            "C=CC(=O)OC",  # Methyl acrylate
            "C=CC(=O)O",  # Acrylic acid
            "C=CC(C)(C)C(=O)OC",  # Methyl methacrylate
            "C=CCl",  # Vinyl chloride
            "C=CC(=O)N",  # Acrylamide
            "C=Cc1ccc(O)cc1",  # 4-Vinylphenol
            "C=CC(F)(F)F",  # Trifluoroethylene
        ]
        
        logger.debug(f"Loaded {len(self.simple_templates)} simple, "
                    f"{len(self.medium_templates)} medium, "
                    f"{len(self.complex_templates)} complex templates")
    
    def _generate_variations(self, template: str, num_variations: int) -> List[str]:
        """Generate structural variations of a template molecule.
        
        Args:
            template: Base SMILES template
            num_variations: Number of variations to generate
            
        Returns:
            List of varied SMILES strings
        """
        variations = [template]  # Include original
        
        # Simple structural modifications
        modifications = [
            lambda s: s.replace('C', 'CC', 1) if 'C' in s else s,  # Add methyl
            lambda s: s.replace('c1ccccc1', 'c1ccc(C)cc1') if 'c1ccccc1' in s else s,  # Add methyl to benzene
            lambda s: s.replace('O', 'N') if s.count('O') == 1 else s,  # Replace O with N
            lambda s: s.replace('C(=O)', 'C(=S)') if 'C(=O)' in s else s,  # Replace C=O with C=S
            lambda s: s + 'C' if len(s) < 50 else s,  # Add carbon
            lambda s: s.replace('CC', 'CCC', 1) if 'CC' in s else s,  # Extend chain
        ]
        
        for _ in range(min(num_variations - 1, len(modifications))):
            if variations:
                base = random.choice(variations[:3])  # Use recent variations as base
                modification = random.choice(modifications)
                try:
                    modified = modification(base)
                    if modified != base and modified not in variations:
                        variations.append(modified)
                except Exception:
                    continue
        
        return variations[:num_variations]
    
    def generate_dataset(self, size: int) -> MolecularDataset:
        """Generate a molecular dataset of specified size.
        
        Args:
            size: Number of molecules to generate
            
        Returns:
            Generated molecular dataset
        """
        logger.info(f"Generating dataset of size {size}")
        
        smiles_list = []
        properties = {
            'complexity': [],
            'category': [],
            'molecular_weight_estimate': []
        }
        
        # Calculate target counts for each category
        simple_count = int(size * self.config.complexity_distribution['simple'])
        medium_count = int(size * self.config.complexity_distribution['medium'])
        complex_count = size - simple_count - medium_count
        
        drug_like_count = int(size * self.config.drug_like_fraction) if self.config.include_drug_like else 0
        polymer_count = int(size * self.config.polymer_fraction) if self.config.include_polymers else 0
        
        # Generate simple molecules
        for _ in range(simple_count):
            template = random.choice(self.simple_templates)
            variations = self._generate_variations(template, 3)
            smiles = random.choice(variations)
            
            smiles_list.append(smiles)
            properties['complexity'].append('simple')
            properties['category'].append('small_molecule')
            properties['molecular_weight_estimate'].append(self._estimate_molecular_weight(smiles))
        
        # Generate medium complexity molecules
        for _ in range(medium_count):
            template = random.choice(self.medium_templates)
            variations = self._generate_variations(template, 2)
            smiles = random.choice(variations)
            
            smiles_list.append(smiles)
            properties['complexity'].append('medium')
            properties['category'].append('medium_molecule')
            properties['molecular_weight_estimate'].append(self._estimate_molecular_weight(smiles))
        
        # Generate complex molecules
        for _ in range(complex_count):
            template = random.choice(self.complex_templates)
            variations = self._generate_variations(template, 2)
            smiles = random.choice(variations)
            
            smiles_list.append(smiles)
            properties['complexity'].append('complex')
            properties['category'].append('large_molecule')
            properties['molecular_weight_estimate'].append(self._estimate_molecular_weight(smiles))
        
        # Replace some molecules with drug-like structures
        if drug_like_count > 0:
            indices_to_replace = random.sample(range(len(smiles_list)), min(drug_like_count, len(smiles_list)))
            for idx in indices_to_replace:
                template = random.choice(self.drug_like_templates)
                variations = self._generate_variations(template, 2)
                smiles = random.choice(variations)
                
                smiles_list[idx] = smiles
                properties['category'][idx] = 'drug_like'
                properties['molecular_weight_estimate'][idx] = self._estimate_molecular_weight(smiles)
        
        # Replace some molecules with polymer building blocks
        if polymer_count > 0:
            indices_to_replace = random.sample(range(len(smiles_list)), min(polymer_count, len(smiles_list)))
            for idx in indices_to_replace:
                template = random.choice(self.polymer_templates)
                variations = self._generate_variations(template, 2)
                smiles = random.choice(variations)
                
                smiles_list[idx] = smiles
                properties['category'][idx] = 'polymer_building_block'
                properties['molecular_weight_estimate'][idx] = self._estimate_molecular_weight(smiles)
        
        # Shuffle to randomize order
        combined = list(zip(smiles_list, properties['complexity'], properties['category'], properties['molecular_weight_estimate']))
        random.shuffle(combined)
        
        smiles_list, properties['complexity'], properties['category'], properties['molecular_weight_estimate'] = zip(*combined)
        smiles_list = list(smiles_list)
        properties['complexity'] = list(properties['complexity'])
        properties['category'] = list(properties['category'])
        properties['molecular_weight_estimate'] = list(properties['molecular_weight_estimate'])
        
        # Create dataset
        dataset = MolecularDataset(
            name=f"{self.config.name}_size_{size}",
            size=size,
            smiles=smiles_list,
            properties=properties,
            metadata={
                'generation_config': {
                    'complexity_distribution': self.config.complexity_distribution,
                    'include_drug_like': self.config.include_drug_like,
                    'include_polymers': self.config.include_polymers,
                    'random_seed': self.config.random_seed
                },
                'statistics': {
                    'simple_count': simple_count,
                    'medium_count': medium_count,
                    'complex_count': complex_count,
                    'drug_like_count': drug_like_count,
                    'polymer_count': polymer_count
                }
            }
        )
        
        logger.info(f"Generated dataset with {len(smiles_list)} molecules")
        return dataset
    
    def _estimate_molecular_weight(self, smiles: str) -> float:
        """Estimate molecular weight from SMILES string.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Estimated molecular weight
        """
        # Simple estimation based on atom counts
        atomic_weights = {
            'C': 12.01, 'H': 1.008, 'O': 15.999, 'N': 14.007,
            'S': 32.06, 'P': 30.974, 'F': 18.998, 'Cl': 35.45,
            'Br': 79.904, 'I': 126.90
        }
        
        weight = 0.0
        
        # Count explicit atoms
        for atom, atomic_weight in atomic_weights.items():
            if atom in smiles:
                # Simple count (not accurate for complex SMILES)
                count = smiles.count(atom)
                weight += count * atomic_weight
        
        # Add estimated hydrogens (very rough approximation)
        carbon_count = smiles.count('C') + smiles.count('c')
        estimated_hydrogens = max(0, carbon_count * 2 - smiles.count('=') * 2 - smiles.count('#') * 4)
        weight += estimated_hydrogens * atomic_weights['H']
        
        return max(weight, 50.0)  # Minimum reasonable molecular weight
    
    def generate_all_datasets(self) -> Dict[int, MolecularDataset]:
        """Generate all configured dataset sizes.
        
        Returns:
            Dictionary mapping size to dataset
        """
        datasets = {}
        
        for size in self.config.sizes:
            logger.info(f"Generating dataset of size {size}")
            dataset = self.generate_dataset(size)
            datasets[size] = dataset
            
            # Save dataset
            filename = f"{self.config.name}_size_{size}.json"
            filepath = self.output_dir / filename
            dataset.save(filepath)
        
        logger.info(f"Generated {len(datasets)} datasets")
        return datasets
    
    def load_existing_datasets(self) -> Dict[int, MolecularDataset]:
        """Load existing datasets from disk.
        
        Returns:
            Dictionary mapping size to dataset
        """
        datasets = {}
        
        for size in self.config.sizes:
            filename = f"{self.config.name}_size_{size}.json"
            filepath = self.output_dir / filename
            
            if filepath.exists():
                try:
                    dataset = MolecularDataset.load(filepath)
                    datasets[size] = dataset
                    logger.info(f"Loaded existing dataset of size {size}")
                except Exception as e:
                    logger.warning(f"Failed to load dataset {filepath}: {e}")
        
        return datasets


def generate_benchmark_datasets(config: DatasetConfig) -> Dict[int, MolecularDataset]:
    """Generate benchmark datasets with specified configuration.
    
    Args:
        config: Dataset generation configuration
        
    Returns:
        Dictionary mapping size to generated datasets
    """
    generator = BenchmarkDatasetGenerator(config)
    return generator.generate_all_datasets()