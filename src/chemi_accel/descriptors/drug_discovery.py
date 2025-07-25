"""
Drug Discovery Specialized Features.

This module provides specialized descriptors and scoring functions for drug discovery,
including synthetic accessibility, drug-likeness scoring, lead-likeness assessments,
PAINS filtering, and natural product-likeness scoring.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import (
        AllChem,
        Crippen,
        Descriptors,
        Lipinski,
        rdchem,
        rdMolDescriptors,
    )
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    from rdkit.Chem.QED import qed

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from ..core.exceptions import ChemiAccelError
from ..utils.logging import get_logger

logger = get_logger()


class DrugDiscoveryDescriptorCalculator:
    """Calculator for drug discovery specialized descriptors and scoring functions."""

    def __init__(self):
        """Initialize drug discovery descriptor calculator."""
        if not RDKIT_AVAILABLE:
            raise ChemiAccelError(
                "RDKit is required for drug discovery descriptor calculations"
            )

        # Initialize filter catalogs
        self._init_filter_catalogs()

        # Define drug discovery descriptor functions with metadata
        self.descriptor_registry = {
            # Synthetic accessibility
            "synthetic_accessibility": {
                "function": self._calculate_synthetic_accessibility,
                "description": "Synthetic Accessibility Score (1=easy, 10=difficult)",
                "category": "synthetic",
                "units": "score",
            },
            # Drug-likeness scoring
            "qed_score": {
                "function": self._calculate_qed_score,
                "description": "Quantitative Estimate of Drug-likeness",
                "category": "drug_likeness",
                "units": "score",
            },
            "cns_mpo_score": {
                "function": self._calculate_cns_mpo_score,
                "description": "CNS Multi-Parameter Optimization score",
                "category": "drug_likeness",
                "units": "score",
            },
            "bbb_mpo_score": {
                "function": self._calculate_bbb_mpo_score,
                "description": "Blood-Brain Barrier MPO score",
                "category": "drug_likeness",
                "units": "score",
            },
            # Lead-likeness and fragment-likeness
            "lead_likeness": {
                "function": self._calculate_lead_likeness,
                "description": "Lead-likeness assessment",
                "category": "lead_likeness",
                "units": "score",
            },
            "fragment_likeness": {
                "function": self._calculate_fragment_likeness,
                "description": "Fragment-likeness assessment",
                "category": "fragment_likeness",
                "units": "score",
            },
            "rule_of_three": {
                "function": self._calculate_rule_of_three,
                "description": "Rule of Three compliance for fragments",
                "category": "fragment_likeness",
                "units": "violations",
            },
            # PAINS and structural alerts
            "pains_alerts": {
                "function": self._calculate_pains_alerts,
                "description": "Pan-Assay Interference Compounds alerts",
                "category": "structural_alerts",
                "units": "count",
            },
            "brenk_alerts": {
                "function": self._calculate_brenk_alerts,
                "description": "Brenk structural alerts",
                "category": "structural_alerts",
                "units": "count",
            },
            "nih_alerts": {
                "function": self._calculate_nih_alerts,
                "description": "NIH structural alerts",
                "category": "structural_alerts",
                "units": "count",
            },
            # Natural product-likeness
            "natural_product_likeness": {
                "function": self._calculate_natural_product_likeness,
                "description": "Natural Product-likeness score",
                "category": "natural_product",
                "units": "score",
            },
            # Additional drug discovery metrics
            "lipinski_violations": {
                "function": self._calculate_lipinski_violations,
                "description": "Number of Lipinski Rule of Five violations",
                "category": "drug_likeness",
                "units": "violations",
            },
            "veber_violations": {
                "function": self._calculate_veber_violations,
                "description": "Number of Veber rule violations",
                "category": "drug_likeness",
                "units": "violations",
            },
            "egan_violations": {
                "function": self._calculate_egan_violations,
                "description": "Number of Egan rule violations",
                "category": "drug_likeness",
                "units": "violations",
            },
            "muegge_violations": {
                "function": self._calculate_muegge_violations,
                "description": "Number of Muegge rule violations",
                "category": "drug_likeness",
                "units": "violations",
            },
        }

    def _init_filter_catalogs(self):
        """Initialize various filter catalogs for structural alerts."""
        try:
            # PAINS filter
            pains_params = FilterCatalogParams()
            pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            self.pains_catalog = FilterCatalog(pains_params)

            # Brenk filter
            brenk_params = FilterCatalogParams()
            brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
            self.brenk_catalog = FilterCatalog(brenk_params)

            # NIH filter
            nih_params = FilterCatalogParams()
            nih_params.AddCatalog(FilterCatalogParams.FilterCatalogs.NIH)
            self.nih_catalog = FilterCatalog(nih_params)

        except Exception as e:
            logger.warning(f"Could not initialize filter catalogs: {e}")
            self.pains_catalog = None
            self.brenk_catalog = None
            self.nih_catalog = None

    def calculate_descriptors(
        self,
        molecules: Union[List[str], pd.Series],
        descriptor_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Calculate drug discovery descriptors for a list of molecules.

        Args:
            molecules: List of SMILES strings or pandas Series
            descriptor_names: List of descriptor names to calculate (default: all)

        Returns:
            DataFrame with calculated drug discovery descriptors
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
                    result = dict.fromkeys(descriptor_names, np.nan)
                    result["smiles"] = smiles
                    results.append(result)
                    continue

                # Calculate requested descriptors
                result = {"smiles": smiles}
                for desc_name in descriptor_names:
                    if desc_name in self.descriptor_registry:
                        try:
                            desc_func = self.descriptor_registry[desc_name]["function"]
                            result[desc_name] = desc_func(mol)
                        except Exception as e:
                            logger.warning(
                                f"Error calculating {desc_name} for {smiles}: {e}"
                            )
                            result[desc_name] = np.nan
                    else:
                        logger.warning(f"Unknown descriptor: {desc_name}")
                        result[desc_name] = np.nan

                results.append(result)

            except Exception as e:
                logger.error(f"Error processing molecule {i}: {e}")
                result = dict.fromkeys(descriptor_names, np.nan)
                result["smiles"] = smiles
                results.append(result)

        return pd.DataFrame(results)

    # Synthetic accessibility
    def _calculate_synthetic_accessibility(self, mol: Chem.Mol) -> float:
        """
        Calculate Synthetic Accessibility Score.

        This is a simplified implementation. The full SA score requires
        fragment contribution data that would need to be loaded separately.
        """
        try:
            # Simplified SA score based on molecular complexity
            mw = Descriptors.MolWt(mol)
            num_rings = rdMolDescriptors.CalcNumRings(mol)
            num_heteroatoms = rdMolDescriptors.CalcNumHeteroatoms(mol)
            num_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
            num_stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)

            # Base complexity score
            complexity = 0.0

            # Molecular weight contribution
            complexity += min(mw / 100.0, 3.0)

            # Ring complexity
            complexity += num_rings * 0.5

            # Heteroatom complexity
            complexity += num_heteroatoms * 0.3

            # Rotatable bond flexibility penalty
            complexity += num_rotatable * 0.1

            # Stereochemistry complexity
            complexity += num_stereo * 0.4

            # Convert to SA score (1-10 scale, higher = more difficult)
            sa_score = min(1 + complexity, 10.0)

            return sa_score

        except Exception as e:
            logger.warning(f"Error calculating synthetic accessibility: {e}")
            return np.nan

    # Drug-likeness scoring
    def _calculate_qed_score(self, mol: Chem.Mol) -> float:
        """Calculate Quantitative Estimate of Drug-likeness (QED) score."""
        try:
            return qed(mol)
        except Exception as e:
            logger.warning(f"Error calculating QED score: {e}")
            return np.nan

    def _calculate_cns_mpo_score(self, mol: Chem.Mol) -> float:
        """
        Calculate CNS Multi-Parameter Optimization (MPO) score.

        Based on Wager et al. ACS Chem Neurosci. 2010, 1, 435-449.
        """
        try:
            logp = Crippen.MolLogP(mol)
            logd = self._estimate_logd(mol)  # Simplified LogD estimation
            mw = Descriptors.MolWt(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            pka = self._estimate_pka_basic(mol)

            # CNS MPO scoring functions
            scores = []

            # LogP score (optimal 1-3)
            if logp <= 1:
                logp_score = logp
            elif logp <= 3:
                logp_score = 1.0
            elif logp <= 5:
                logp_score = 1.0 - (logp - 3) / 2
            else:
                logp_score = 0.0
            scores.append(logp_score)

            # LogD score (optimal 1-2)
            if logd <= 1:
                logd_score = logd
            elif logd <= 2:
                logd_score = 1.0
            elif logd <= 4:
                logd_score = 1.0 - (logd - 2) / 2
            else:
                logd_score = 0.0
            scores.append(logd_score)

            # Molecular weight score (optimal <360)
            if mw <= 360:
                mw_score = 1.0
            elif mw <= 500:
                mw_score = 1.0 - (mw - 360) / 140
            else:
                mw_score = 0.0
            scores.append(mw_score)

            # TPSA score (optimal <90)
            if tpsa <= 90:
                tpsa_score = 1.0
            elif tpsa <= 120:
                tpsa_score = 1.0 - (tpsa - 90) / 30
            else:
                tpsa_score = 0.0
            scores.append(tpsa_score)

            # HBD score (optimal ≤1)
            if hbd <= 1:
                hbd_score = 1.0
            elif hbd <= 3:
                hbd_score = 1.0 - (hbd - 1) / 2
            else:
                hbd_score = 0.0
            scores.append(hbd_score)

            # pKa score (optimal 8-10)
            if 8 <= pka <= 10:
                pka_score = 1.0
            elif 6 <= pka < 8:
                pka_score = (pka - 6) / 2
            elif 10 < pka <= 12:
                pka_score = 1.0 - (pka - 10) / 2
            else:
                pka_score = 0.0
            scores.append(pka_score)

            # Calculate overall CNS MPO score
            cns_mpo = sum(scores)

            return cns_mpo

        except Exception as e:
            logger.warning(f"Error calculating CNS MPO score: {e}")
            return np.nan

    def _calculate_bbb_mpo_score(self, mol: Chem.Mol) -> float:
        """
        Calculate Blood-Brain Barrier MPO score.

        Simplified BBB penetration scoring based on key molecular properties.
        """
        try:
            logp = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            hbd = rdMolDescriptors.CalcNumHBD(mol)

            scores = []

            # LogP score (optimal 1-3)
            if 1 <= logp <= 3:
                logp_score = 1.0
            elif 0 <= logp < 1:
                logp_score = logp
            elif 3 < logp <= 5:
                logp_score = 1.0 - (logp - 3) / 2
            else:
                logp_score = 0.0
            scores.append(logp_score)

            # Molecular weight score (optimal <450)
            if mw <= 450:
                mw_score = 1.0
            elif mw <= 600:
                mw_score = 1.0 - (mw - 450) / 150
            else:
                mw_score = 0.0
            scores.append(mw_score)

            # TPSA score (optimal <60)
            if tpsa <= 60:
                tpsa_score = 1.0
            elif tpsa <= 90:
                tpsa_score = 1.0 - (tpsa - 60) / 30
            else:
                tpsa_score = 0.0
            scores.append(tpsa_score)

            # HBD score (optimal ≤2)
            if hbd <= 2:
                hbd_score = 1.0
            elif hbd <= 5:
                hbd_score = 1.0 - (hbd - 2) / 3
            else:
                hbd_score = 0.0
            scores.append(hbd_score)

            bbb_mpo = sum(scores)

            return bbb_mpo

        except Exception as e:
            logger.warning(f"Error calculating BBB MPO score: {e}")
            return np.nan

    # Lead-likeness and fragment-likeness
    def _calculate_lead_likeness(self, mol: Chem.Mol) -> float:
        """
        Calculate lead-likeness score based on Teague et al. criteria.

        Lead compounds typically have MW 200-350, LogP 1-3, etc.
        """
        try:
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)

            score = 0.0

            # Molecular weight (200-350)
            if 200 <= mw <= 350:
                score += 1.0
            elif 150 <= mw < 200 or 350 < mw <= 400:
                score += 0.5

            # LogP (1-3)
            if 1 <= logp <= 3:
                score += 1.0
            elif 0 <= logp < 1 or 3 < logp <= 4:
                score += 0.5

            # HBD (≤3)
            if hbd <= 3:
                score += 1.0

            # HBA (≤6)
            if hba <= 6:
                score += 1.0

            # Rotatable bonds (≤6)
            if rotatable <= 6:
                score += 1.0

            # Normalize to 0-1 scale
            lead_likeness = score / 5.0

            return lead_likeness

        except Exception as e:
            logger.warning(f"Error calculating lead-likeness: {e}")
            return np.nan

    def _calculate_fragment_likeness(self, mol: Chem.Mol) -> float:
        """
        Calculate fragment-likeness score.

        Fragments are typically small, simple molecules suitable for
        fragment-based drug discovery.
        """
        try:
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
            heavy_atoms = mol.GetNumHeavyAtoms()

            score = 0.0

            # Heavy atom count (≤22)
            if heavy_atoms <= 22:
                score += 1.0
            elif heavy_atoms <= 30:
                score += 0.5

            # Molecular weight (≤300)
            if mw <= 300:
                score += 1.0
            elif mw <= 350:
                score += 0.5

            # LogP (-1 to 3)
            if -1 <= logp <= 3:
                score += 1.0
            elif -2 <= logp < -1 or 3 < logp <= 4:
                score += 0.5

            # HBD (≤3)
            if hbd <= 3:
                score += 1.0

            # HBA (≤6)
            if hba <= 6:
                score += 1.0

            # Rotatable bonds (≤3)
            if rotatable <= 3:
                score += 1.0

            # Normalize to 0-1 scale
            fragment_likeness = score / 6.0

            return fragment_likeness

        except Exception as e:
            logger.warning(f"Error calculating fragment-likeness: {e}")
            return np.nan

    def _calculate_rule_of_three(self, mol: Chem.Mol) -> int:
        """
        Calculate Rule of Three violations for fragments.

        Rule of Three: MW ≤ 300, LogP ≤ 3, HBD ≤ 3, HBA ≤ 3, Rotatable ≤ 3
        """
        try:
            violations = 0

            mw = Descriptors.MolWt(mol)
            if mw > 300:
                violations += 1

            logp = Crippen.MolLogP(mol)
            if logp > 3:
                violations += 1

            hbd = rdMolDescriptors.CalcNumHBD(mol)
            if hbd > 3:
                violations += 1

            hba = rdMolDescriptors.CalcNumHBA(mol)
            if hba > 3:
                violations += 1

            rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
            if rotatable > 3:
                violations += 1

            return violations

        except Exception as e:
            logger.warning(f"Error calculating Rule of Three violations: {e}")
            return np.nan

    # PAINS and structural alerts
    def _calculate_pains_alerts(self, mol: Chem.Mol) -> int:
        """Calculate number of PAINS (Pan-Assay Interference Compounds) alerts."""
        try:
            if self.pains_catalog is None:
                return 0

            matches = self.pains_catalog.GetMatches(mol)
            return len(matches)

        except Exception as e:
            logger.warning(f"Error calculating PAINS alerts: {e}")
            return np.nan

    def _calculate_brenk_alerts(self, mol: Chem.Mol) -> int:
        """Calculate number of Brenk structural alerts."""
        try:
            if self.brenk_catalog is None:
                return 0

            matches = self.brenk_catalog.GetMatches(mol)
            return len(matches)

        except Exception as e:
            logger.warning(f"Error calculating Brenk alerts: {e}")
            return np.nan

    def _calculate_nih_alerts(self, mol: Chem.Mol) -> int:
        """Calculate number of NIH structural alerts."""
        try:
            if self.nih_catalog is None:
                return 0

            matches = self.nih_catalog.GetMatches(mol)
            return len(matches)

        except Exception as e:
            logger.warning(f"Error calculating NIH alerts: {e}")
            return np.nan

    # Natural product-likeness
    def _calculate_natural_product_likeness(self, mol: Chem.Mol) -> float:
        """
        Calculate Natural Product-likeness score.

        This is a simplified implementation. The full NP-likeness score
        requires training data from natural product databases.
        """
        try:
            # Simplified NP-likeness based on structural features
            # common in natural products

            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            num_rings = rdMolDescriptors.CalcNumRings(mol)
            num_stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
            try:
                num_sp3 = rdMolDescriptors.CalcFractionCsp3(mol)
            except AttributeError:
                # Fallback for older RDKit versions
                num_sp3 = 0.5  # Default moderate sp3 fraction

            score = 0.0

            # Natural products often have moderate MW
            if 200 <= mw <= 800:
                score += 0.2

            # Moderate lipophilicity
            if -1 <= logp <= 5:
                score += 0.2

            # Ring systems common in NPs
            if 1 <= num_rings <= 6:
                score += 0.2

            # Stereochemistry is common in NPs
            if num_stereo > 0:
                score += 0.2

            # High sp3 fraction is typical of NPs
            if num_sp3 > 0.4:
                score += 0.2

            return score

        except Exception as e:
            logger.warning(f"Error calculating natural product-likeness: {e}")
            return np.nan

    # Drug-likeness rule violations
    def _calculate_lipinski_violations(self, mol: Chem.Mol) -> int:
        """Calculate number of Lipinski Rule of Five violations."""
        try:
            violations = 0

            mw = Descriptors.MolWt(mol)
            if mw > 500:
                violations += 1

            logp = Crippen.MolLogP(mol)
            if logp > 5:
                violations += 1

            hbd = rdMolDescriptors.CalcNumHBD(mol)
            if hbd > 5:
                violations += 1

            hba = rdMolDescriptors.CalcNumHBA(mol)
            if hba > 10:
                violations += 1

            return violations

        except Exception as e:
            logger.warning(f"Error calculating Lipinski violations: {e}")
            return np.nan

    def _calculate_veber_violations(self, mol: Chem.Mol) -> int:
        """Calculate number of Veber rule violations."""
        try:
            violations = 0

            tpsa = rdMolDescriptors.CalcTPSA(mol)
            if tpsa > 140:
                violations += 1

            rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
            if rotatable > 10:
                violations += 1

            return violations

        except Exception as e:
            logger.warning(f"Error calculating Veber violations: {e}")
            return np.nan

    def _calculate_egan_violations(self, mol: Chem.Mol) -> int:
        """Calculate number of Egan rule violations."""
        try:
            violations = 0

            logp = Crippen.MolLogP(mol)
            if logp > 5.88:
                violations += 1

            tpsa = rdMolDescriptors.CalcTPSA(mol)
            if tpsa > 131.6:
                violations += 1

            return violations

        except Exception as e:
            logger.warning(f"Error calculating Egan violations: {e}")
            return np.nan

    def _calculate_muegge_violations(self, mol: Chem.Mol) -> int:
        """Calculate number of Muegge rule violations."""
        try:
            violations = 0

            mw = Descriptors.MolWt(mol)
            if not (200 <= mw <= 600):
                violations += 1

            logp = Crippen.MolLogP(mol)
            if not (-2 <= logp <= 5):
                violations += 1

            tpsa = rdMolDescriptors.CalcTPSA(mol)
            if tpsa > 150:
                violations += 1

            num_rings = rdMolDescriptors.CalcNumRings(mol)
            if num_rings > 7:
                violations += 1

            num_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
            if num_carbons < 5:
                violations += 1

            num_heteroatoms = rdMolDescriptors.CalcNumHeteroatoms(mol)
            if num_heteroatoms < 2:
                violations += 1

            rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
            if rotatable > 15:
                violations += 1

            hbd = rdMolDescriptors.CalcNumHBD(mol)
            if hbd > 5:
                violations += 1

            hba = rdMolDescriptors.CalcNumHBA(mol)
            if hba > 10:
                violations += 1

            return violations

        except Exception as e:
            logger.warning(f"Error calculating Muegge violations: {e}")
            return np.nan

    # Helper methods
    def _estimate_logd(self, mol: Chem.Mol, ph: float = 7.4) -> float:
        """
        Estimate LogD at physiological pH.

        This is a simplified estimation. More accurate LogD prediction
        would require pKa prediction and Henderson-Hasselbalch equation.
        """
        try:
            logp = Crippen.MolLogP(mol)

            # Simple approximation: LogD ≈ LogP for neutral molecules
            # For ionizable molecules, this would need pKa correction
            return logp

        except Exception:
            return np.nan

    def _estimate_pka_basic(self, mol: Chem.Mol) -> float:
        """
        Estimate basic pKa of a molecule.

        This is a simplified estimation.
        """
        try:
            # Look for basic nitrogen atoms
            basic_nitrogens = mol.GetSubstructMatches(
                Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")
            )

            if basic_nitrogens:
                return 9.5  # Rough estimate for aliphatic amines

            # Look for aromatic nitrogens
            aromatic_nitrogens = mol.GetSubstructMatches(Chem.MolFromSmarts("c[nX2]"))

            if aromatic_nitrogens:
                return 5.0  # Rough estimate for aromatic amines

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

        return [
            name
            for name, info in self.descriptor_registry.items()
            if info.get("category") == category
        ]

    def get_categories(self) -> List[str]:
        """Get list of available descriptor categories."""
        categories = set()
        for info in self.descriptor_registry.values():
            if "category" in info:
                categories.add(info["category"])
        return sorted(list(categories))
