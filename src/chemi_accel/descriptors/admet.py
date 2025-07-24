"""
ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) descriptor calculations.

This module provides specialized descriptors for drug discovery and ADMET prediction,
including absorption, distribution, metabolism, excretion, and toxicity descriptors.
These descriptors are crucial for pharmaceutical research and drug development.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
import math

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
    from rdkit.Chem import AllChem, rdchem, rdFreeSASA
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from ..core.molecular import MolecularData
from ..core.exceptions import ChemiAccelError, DescriptorCalculationError
from ..utils.logging import get_logger

logger = get_logger()


class ADMETDescriptorCalculator:
    """Calculator for ADMET descriptors for drug discovery applications."""
    
    def __init__(self):
        """Initialize ADMET descriptor calculator."""
        if not RDKIT_AVAILABLE:
            raise ChemiAccelError("RDKit is required for ADMET descriptor calculations")
        
        # Initialize PAINS filter catalog
        self._init_pains_filter()
        
        # Define ADMET descriptor functions with metadata
        self.descriptor_registry = {
            # Absorption descriptors
            "caco2_permeability": {
                "function": self._calculate_caco2_permeability,
                "description": "Predicted Caco-2 cell permeability",
                "category": "absorption",
                "units": "log(cm/s)"
            },
            "hia_prediction": {
                "function": self._calculate_hia_prediction,
                "description": "Human Intestinal Absorption prediction",
                "category": "absorption",
                "units": "probability"
            },
            "pgp_substrate": {
                "function": self._calculate_pgp_substrate,
                "description": "P-glycoprotein substrate prediction",
                "category": "absorption",
                "units": "probability"
            },
            
            # Distribution descriptors
            "bbb_permeability": {
                "function": self._calculate_bbb_permeability,
                "description": "Blood-brain barrier permeability",
                "category": "distribution",
                "units": "log(BB)"
            },
            "plasma_protein_binding": {
                "function": self._calculate_plasma_protein_binding,
                "description": "Plasma protein binding prediction",
                "category": "distribution",
                "units": "percentage"
            },
            "volume_distribution": {
                "function": self._calculate_volume_distribution,
                "description": "Volume of distribution prediction",
                "category": "distribution",
                "units": "L/kg"
            },
            
            # Metabolism descriptors
            "cyp1a2_inhibition": {
                "function": self._calculate_cyp1a2_inhibition,
                "description": "CYP1A2 enzyme inhibition prediction",
                "category": "metabolism",
                "units": "probability"
            },
            "cyp2c9_inhibition": {
                "function": self._calculate_cyp2c9_inhibition,
                "description": "CYP2C9 enzyme inhibition prediction",
                "category": "metabolism",
                "units": "probability"
            },
            "cyp2c19_inhibition": {
                "function": self._calculate_cyp2c19_inhibition,
                "description": "CYP2C19 enzyme inhibition prediction",
                "category": "metabolism",
                "units": "probability"
            },
            "cyp2d6_inhibition": {
                "function": self._calculate_cyp2d6_inhibition,
                "description": "CYP2D6 enzyme inhibition prediction",
                "category": "metabolism",
                "units": "probability"
            },
            "cyp3a4_inhibition": {
                "function": self._calculate_cyp3a4_inhibition,
                "description": "CYP3A4 enzyme inhibition prediction",
                "category": "metabolism",
                "units": "probability"
            },
            
            # Excretion descriptors
            "renal_clearance": {
                "function": self._calculate_renal_clearance,
                "description": "Renal clearance prediction",
                "category": "excretion",
                "units": "mL/min/kg"
            },
            "half_life": {
                "function": self._calculate_half_life,
                "description": "Plasma half-life prediction",
                "category": "excretion",
                "units": "hours"
            },
            
            # Toxicity descriptors
            "herg_inhibition": {
                "function": self._calculate_herg_inhibition,
                "description": "hERG potassium channel inhibition",
                "category": "toxicity",
                "units": "probability"
            },
            "ames_mutagenicity": {
                "function": self._calculate_ames_mutagenicity,
                "description": "AMES mutagenicity prediction",
                "category": "toxicity",
                "units": "probability"
            },
            "hepatotoxicity": {
                "function": self._calculate_hepatotoxicity,
                "description": "Hepatotoxicity prediction",
                "category": "toxicity",
                "units": "probability"
            },
            "skin_sensitization": {
                "function": self._calculate_skin_sensitization,
                "description": "Skin sensitization prediction",
                "category": "toxicity",
                "units": "probability"
            },
            "carcinogenicity": {
                "function": self._calculate_carcinogenicity,
                "description": "Carcinogenicity prediction",
                "category": "toxicity",
                "units": "probability"
            }
        }
    
    def _init_pains_filter(self):
        """Initialize PAINS (Pan-Assay Interference Compounds) filter."""
        try:
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            self.pains_catalog = FilterCatalog(params)
        except Exception as e:
            logger.warning(f"Could not initialize PAINS filter: {e}")
            self.pains_catalog = None
    
    def calculate_descriptors(self, molecules: Union[List[str], pd.Series], 
                            descriptor_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate ADMET descriptors for a list of molecules.
        
        Args:
            molecules: List of SMILES strings or pandas Series
            descriptor_names: List of descriptor names to calculate (default: all)
            
        Returns:
            DataFrame with calculated ADMET descriptors
        """
        if isinstance(molecules, pd.Series):
            molecules = molecules.tolist()
        
        if descriptor_names is None:
            descriptor_names = list(self.descriptor_registry.keys())
        
        results = []
        
        for i, smiles in enumerate(molecules):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.warning(f"Invalid SMILES: {smiles}")
                    result = {name: np.nan for name in descriptor_names}
                    result['smiles'] = smiles
                    results.append(result)
                    continue
                
                # Calculate requested descriptors
                result = {'smiles': smiles}
                for desc_name in descriptor_names:
                    if desc_name in self.descriptor_registry:
                        try:
                            desc_func = self.descriptor_registry[desc_name]["function"]
                            result[desc_name] = desc_func(mol)
                        except Exception as e:
                            logger.warning(f"Error calculating {desc_name} for {smiles}: {e}")
                            result[desc_name] = np.nan
                    else:
                        logger.warning(f"Unknown descriptor: {desc_name}")
                        result[desc_name] = np.nan
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing molecule {i}: {e}")
                result = {name: np.nan for name in descriptor_names}
                result['smiles'] = smiles
                results.append(result)
        
        return pd.DataFrame(results)
    
    # Absorption descriptors
    def _calculate_caco2_permeability(self, mol: Chem.Mol) -> float:
        """
        Calculate Caco-2 cell permeability prediction.
        
        Based on molecular descriptors that correlate with Caco-2 permeability.
        Uses a simplified model based on LogP, TPSA, and molecular weight.
        """
        try:
            logp = Crippen.MolLogP(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            mw = Descriptors.MolWt(mol)
            
            # Simplified Caco-2 permeability model
            # Based on literature correlations
            log_perm = -5.15 + 0.35 * logp - 0.01 * tpsa - 0.0015 * mw
            
            return log_perm
            
        except Exception as e:
            logger.warning(f"Error calculating Caco-2 permeability: {e}")
            return np.nan
    
    def _calculate_hia_prediction(self, mol: Chem.Mol) -> float:
        """
        Calculate Human Intestinal Absorption prediction.
        
        Based on molecular properties that influence absorption.
        """
        try:
            logp = Crippen.MolLogP(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            mw = Descriptors.MolWt(mol)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            
            # Simplified HIA model based on Lipinski-like rules
            score = 0.0
            
            # Favorable LogP range
            if 0 <= logp <= 3:
                score += 0.3
            elif -1 <= logp < 0 or 3 < logp <= 5:
                score += 0.1
            
            # TPSA contribution
            if tpsa <= 60:
                score += 0.3
            elif 60 < tpsa <= 90:
                score += 0.2
            elif 90 < tpsa <= 140:
                score += 0.1
            
            # Molecular weight contribution
            if mw <= 500:
                score += 0.2
            elif 500 < mw <= 600:
                score += 0.1
            
            # Hydrogen bonding
            if hbd <= 5 and hba <= 10:
                score += 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating HIA prediction: {e}")
            return np.nan
    
    def _calculate_pgp_substrate(self, mol: Chem.Mol) -> float:
        """
        Calculate P-glycoprotein substrate prediction.
        
        P-gp substrates are typically large, lipophilic molecules.
        """
        try:
            logp = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            
            # P-gp substrate likelihood based on molecular properties
            score = 0.0
            
            # High LogP favors P-gp substrate
            if logp > 3:
                score += 0.4
            elif logp > 1:
                score += 0.2
            
            # Higher MW favors P-gp substrate
            if mw > 400:
                score += 0.3
            elif mw > 300:
                score += 0.1
            
            # TPSA contribution
            if tpsa > 60:
                score += 0.3
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating P-gp substrate prediction: {e}")
            return np.nan
    
    # Distribution descriptors
    def _calculate_bbb_permeability(self, mol: Chem.Mol) -> float:
        """
        Calculate Blood-Brain Barrier permeability prediction.
        
        Based on LogP, TPSA, and molecular size.
        """
        try:
            logp = Crippen.MolLogP(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            mw = Descriptors.MolWt(mol)
            
            # BBB permeability model (log BB)
            # Based on literature correlations
            log_bb = -0.0148 * tpsa + 0.152 * logp + 0.139
            
            # Apply molecular weight penalty
            if mw > 450:
                log_bb -= 0.5
            
            return log_bb
            
        except Exception as e:
            logger.warning(f"Error calculating BBB permeability: {e}")
            return np.nan
    
    def _calculate_plasma_protein_binding(self, mol: Chem.Mol) -> float:
        """
        Calculate plasma protein binding prediction.
        
        Based on lipophilicity and molecular properties.
        """
        try:
            logp = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            
            # Simplified plasma protein binding model
            # Higher LogP and MW generally increase binding
            ppb = 50 + 15 * logp + 0.02 * mw
            
            # Constrain to reasonable range
            ppb = max(0, min(100, ppb))
            
            return ppb
            
        except Exception as e:
            logger.warning(f"Error calculating plasma protein binding: {e}")
            return np.nan
    
    def _calculate_volume_distribution(self, mol: Chem.Mol) -> float:
        """
        Calculate volume of distribution prediction.
        
        Based on lipophilicity and molecular properties.
        """
        try:
            logp = Crippen.MolLogP(mol)
            pka_basic = self._estimate_pka_basic(mol)
            
            # Simplified Vd model
            # Based on Oie-Tozer method approximation
            log_vd = 0.15 * logp + 0.1 * pka_basic - 0.8
            
            vd = 10 ** log_vd
            
            return vd
            
        except Exception as e:
            logger.warning(f"Error calculating volume of distribution: {e}")
            return np.nan
    
    # Metabolism descriptors
    def _calculate_cyp1a2_inhibition(self, mol: Chem.Mol) -> float:
        """Calculate CYP1A2 inhibition prediction."""
        return self._calculate_cyp_inhibition(mol, "1A2")
    
    def _calculate_cyp2c9_inhibition(self, mol: Chem.Mol) -> float:
        """Calculate CYP2C9 inhibition prediction."""
        return self._calculate_cyp_inhibition(mol, "2C9")
    
    def _calculate_cyp2c19_inhibition(self, mol: Chem.Mol) -> float:
        """Calculate CYP2C19 inhibition prediction."""
        return self._calculate_cyp_inhibition(mol, "2C19")
    
    def _calculate_cyp2d6_inhibition(self, mol: Chem.Mol) -> float:
        """Calculate CYP2D6 inhibition prediction."""
        return self._calculate_cyp_inhibition(mol, "2D6")
    
    def _calculate_cyp3a4_inhibition(self, mol: Chem.Mol) -> float:
        """Calculate CYP3A4 inhibition prediction."""
        return self._calculate_cyp_inhibition(mol, "3A4")
    
    def _calculate_cyp_inhibition(self, mol: Chem.Mol, cyp_type: str) -> float:
        """
        Generic CYP inhibition prediction based on molecular properties.
        
        This is a simplified model - in practice, more sophisticated
        machine learning models would be used.
        """
        try:
            logp = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            
            # Different CYP enzymes have different preferences
            cyp_params = {
                "1A2": {"logp_opt": 2.5, "mw_opt": 300, "tpsa_opt": 50},
                "2C9": {"logp_opt": 3.0, "mw_opt": 350, "tpsa_opt": 60},
                "2C19": {"logp_opt": 2.8, "mw_opt": 320, "tpsa_opt": 55},
                "2D6": {"logp_opt": 2.2, "mw_opt": 280, "tpsa_opt": 45},
                "3A4": {"logp_opt": 3.5, "mw_opt": 400, "tpsa_opt": 70}
            }
            
            params = cyp_params.get(cyp_type, cyp_params["3A4"])
            
            # Calculate similarity to optimal inhibitor properties
            logp_score = 1 - abs(logp - params["logp_opt"]) / 5.0
            mw_score = 1 - abs(mw - params["mw_opt"]) / 500.0
            tpsa_score = 1 - abs(tpsa - params["tpsa_opt"]) / 100.0
            
            # Combine scores
            inhibition_prob = (logp_score + mw_score + tpsa_score) / 3.0
            inhibition_prob = max(0, min(1, inhibition_prob))
            
            return inhibition_prob
            
        except Exception as e:
            logger.warning(f"Error calculating CYP {cyp_type} inhibition: {e}")
            return np.nan
    
    # Excretion descriptors
    def _calculate_renal_clearance(self, mol: Chem.Mol) -> float:
        """
        Calculate renal clearance prediction.
        
        Based on molecular properties affecting renal elimination.
        """
        try:
            logp = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            
            # Simplified renal clearance model
            # Lower LogP and smaller MW favor renal clearance
            cl_renal = 50 - 10 * logp + 100 / (mw + 100) + tpsa / 10
            
            # Constrain to reasonable range
            cl_renal = max(0, min(200, cl_renal))
            
            return cl_renal
            
        except Exception as e:
            logger.warning(f"Error calculating renal clearance: {e}")
            return np.nan
    
    def _calculate_half_life(self, mol: Chem.Mol) -> float:
        """
        Calculate plasma half-life prediction.
        
        Based on molecular stability and clearance properties.
        """
        try:
            logp = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            
            # Simplified half-life model
            # Higher LogP and MW generally increase half-life
            half_life = 2 + 0.5 * logp + 0.01 * mw
            
            # Apply constraints
            half_life = max(0.1, min(100, half_life))
            
            return half_life
            
        except Exception as e:
            logger.warning(f"Error calculating half-life: {e}")
            return np.nan
    
    # Toxicity descriptors
    def _calculate_herg_inhibition(self, mol: Chem.Mol) -> float:
        """
        Calculate hERG potassium channel inhibition prediction.
        
        hERG inhibition is a major cardiac safety concern.
        """
        try:
            logp = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            
            # hERG inhibitors tend to be lipophilic and basic
            pka_basic = self._estimate_pka_basic(mol)
            
            # Simplified hERG model
            score = 0.0
            
            # LogP contribution
            if 2 <= logp <= 5:
                score += 0.3
            elif logp > 5:
                score += 0.5
            
            # Molecular weight contribution
            if 300 <= mw <= 600:
                score += 0.2
            
            # Basic pKa contribution
            if pka_basic > 8:
                score += 0.3
            
            # TPSA contribution (lower TPSA increases risk)
            if tpsa < 60:
                score += 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating hERG inhibition: {e}")
            return np.nan
    
    def _calculate_ames_mutagenicity(self, mol: Chem.Mol) -> float:
        """
        Calculate AMES mutagenicity prediction.
        
        Based on structural alerts and molecular properties.
        """
        try:
            # Check for common mutagenic substructures
            mutagenic_smarts = [
                "[N+](=O)[O-]",  # Nitro groups
                "N=N",           # Azo groups
                "[#6]=[#6]-[#6]=[#6]",  # Conjugated systems
                "c1ccc2c(c1)ccc3c2ccc4c3cccc4",  # Polycyclic aromatics
                "[CH2][N+]",     # Alkylating agents
            ]
            
            alert_count = 0
            for smarts in mutagenic_smarts:
                if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
                    alert_count += 1
            
            # Base mutagenicity score on structural alerts
            mutagenicity = min(alert_count * 0.3, 1.0)
            
            return mutagenicity
            
        except Exception as e:
            logger.warning(f"Error calculating AMES mutagenicity: {e}")
            return np.nan
    
    def _calculate_hepatotoxicity(self, mol: Chem.Mol) -> float:
        """
        Calculate hepatotoxicity prediction.
        
        Based on molecular properties associated with liver toxicity.
        """
        try:
            logp = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            
            # Hepatotoxic compounds often have high lipophilicity
            score = 0.0
            
            if logp > 3:
                score += 0.4
            elif logp > 5:
                score += 0.6
            
            if mw > 500:
                score += 0.3
            
            # Check for reactive metabolite formation potential
            reactive_smarts = [
                "c1ccc(cc1)N",   # Anilines
                "c1ccc(cc1)O",   # Phenols
                "[CH3][CH2][CH2][CH2][CH2][CH2]",  # Long alkyl chains
            ]
            
            for smarts in reactive_smarts:
                if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
                    score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating hepatotoxicity: {e}")
            return np.nan
    
    def _calculate_skin_sensitization(self, mol: Chem.Mol) -> float:
        """
        Calculate skin sensitization prediction.
        
        Based on electrophilic and protein-reactive groups.
        """
        try:
            # Skin sensitizers often contain electrophilic groups
            sensitizing_smarts = [
                "C=C-C=O",       # α,β-unsaturated carbonyls
                "[CH2]Cl",       # Alkyl halides
                "N=C=O",         # Isocyanates
                "C(=O)Cl",       # Acid chlorides
                "[OH]c1ccccc1",  # Phenols
            ]
            
            alert_count = 0
            for smarts in sensitizing_smarts:
                if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
                    alert_count += 1
            
            sensitization = min(alert_count * 0.4, 1.0)
            
            return sensitization
            
        except Exception as e:
            logger.warning(f"Error calculating skin sensitization: {e}")
            return np.nan
    
    def _calculate_carcinogenicity(self, mol: Chem.Mol) -> float:
        """
        Calculate carcinogenicity prediction.
        
        Based on structural alerts for carcinogenic potential.
        """
        try:
            # Common carcinogenic structural alerts
            carcinogenic_smarts = [
                "c1ccc2c(c1)ccc3c2ccc4c3cccc4",  # Polycyclic aromatics
                "[N+](=O)[O-]",  # Nitro compounds
                "N=N",           # Azo compounds
                "C=C-C=C",       # Conjugated dienes
                "[NH2]c1ccccc1", # Aromatic amines
            ]
            
            alert_count = 0
            for smarts in carcinogenic_smarts:
                if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
                    alert_count += 1
            
            carcinogenicity = min(alert_count * 0.3, 1.0)
            
            return carcinogenicity
            
        except Exception as e:
            logger.warning(f"Error calculating carcinogenicity: {e}")
            return np.nan
    
    # Helper methods
    def _estimate_pka_basic(self, mol: Chem.Mol) -> float:
        """
        Estimate basic pKa of a molecule.
        
        This is a simplified estimation - more sophisticated methods
        would use dedicated pKa prediction tools.
        """
        try:
            # Look for basic nitrogen atoms
            basic_nitrogens = mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]"))
            
            if basic_nitrogens:
                # Rough estimate for aliphatic amines
                return 9.5
            
            # Look for aromatic nitrogens
            aromatic_nitrogens = mol.GetSubstructMatches(Chem.MolFromSmarts("c[nX2]"))
            
            if aromatic_nitrogens:
                # Rough estimate for aromatic amines
                return 5.0
            
            return 0.0  # No basic groups found
            
        except Exception:
            return 0.0
    
    def get_descriptor_info(self, descriptor_name: str) -> Dict[str, Any]:
        """Get information about a specific descriptor."""
        return self.descriptor_registry.get(descriptor_name, {})
    
    def list_descriptors(self, category: Optional[str] = None) -> List[str]:
        """List available descriptors, optionally filtered by category."""
        if category is None:
            return list(self.descriptor_registry.keys())
        
        return [name for name, info in self.descriptor_registry.items() 
                if info.get("category") == category]
    
    def get_categories(self) -> List[str]:
        """Get list of available descriptor categories."""
        categories = set()
        for info in self.descriptor_registry.values():
            if "category" in info:
                categories.add(info["category"])
        return sorted(list(categories))