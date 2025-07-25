"""
GFN2-xTB surrogate model integration for ChemiAccel.

Provides GPU-accelerated interface to fast GFN2-xTB surrogate models
for semi-empirical quantum chemical property prediction.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem

from ..core.molecular import MolecularData
from .base import BaseQMLModel, QMLModelError, QMLPrediction, QuantumProperty


class GFN2xTBSurrogate(nn.Module):
    """Neural network surrogate for GFN2-xTB calculations."""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GFN2xTBModel(BaseQMLModel):
    """
    GFN2-xTB surrogate model for fast semi-empirical quantum calculations.

    Uses neural network surrogates trained on GFN2-xTB data for rapid
    prediction of quantum properties with reasonable accuracy.
    """

    SUPPORTED_ELEMENTS = set(range(1, 87))  # H to Rn

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        precision: str = "float32",
        cache_predictions: bool = True,
        use_xtb_binary: bool = False,
        xtb_path: Optional[str] = None,
    ):
        """
        Initialize GFN2-xTB model.

        Args:
            model_path: Path to surrogate model files
            device: Computing device ('cpu', 'cuda', 'auto')
            precision: Numerical precision ('float32', 'float64')
            cache_predictions: Whether to cache predictions
            use_xtb_binary: Use actual xTB binary instead of surrogate
            xtb_path: Path to xTB executable
        """
        super().__init__(model_path, device, precision, cache_predictions)

        self.use_xtb_binary = use_xtb_binary
        self.xtb_path = xtb_path or "xtb"
        self._surrogate_model = None
        self._feature_scaler = None
        self._target_scaler = None

        # Property configurations
        self._property_configs = {
            QuantumProperty.TOTAL_ENERGY: {"units": "hartree", "index": 0},
            QuantumProperty.HOMO_ENERGY: {"units": "eV", "index": 1},
            QuantumProperty.LUMO_ENERGY: {"units": "eV", "index": 2},
            QuantumProperty.BANDGAP: {"units": "eV", "index": 3},
            QuantumProperty.DIPOLE_MOMENT: {"units": "debye", "index": 4},
            QuantumProperty.POLARIZABILITY: {"units": "au", "index": 5},
            QuantumProperty.PARTIAL_CHARGES: {
                "units": "e",
                "index": None,
            },  # Variable length
            QuantumProperty.FORMATION_ENERGY: {"units": "kcal/mol", "index": 6},
            QuantumProperty.IONIZATION_POTENTIAL: {"units": "eV", "index": 7},
            QuantumProperty.ELECTRON_AFFINITY: {"units": "eV", "index": 8},
        }

    def load_model(self) -> None:
        """Load GFN2-xTB surrogate model."""
        if self.use_xtb_binary:
            # Check if xTB binary is available
            try:
                result = subprocess.run(
                    [self.xtb_path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    raise QMLModelError("xTB binary not found or not working")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                raise QMLModelError(f"xTB binary not found at {self.xtb_path}")
        else:
            # Load neural network surrogate
            if self.model_path is None:
                # Use default pre-trained model
                self.model_path = (
                    Path(__file__).parent / "models" / "gfn2_xtb_surrogate.pt"
                )

            if not self.model_path.exists():
                raise QMLModelError(f"Surrogate model not found: {self.model_path}")

            try:
                # Load model architecture and weights
                checkpoint = torch.load(self.model_path, map_location=self.device)

                # Initialize surrogate model
                self._surrogate_model = GFN2xTBSurrogate(
                    input_dim=checkpoint["input_dim"],
                    hidden_dims=checkpoint["hidden_dims"],
                    output_dim=checkpoint["output_dim"],
                )

                self._surrogate_model.load_state_dict(checkpoint["model_state_dict"])
                self._surrogate_model.to(self.device)
                self._surrogate_model.eval()

                # Load scalers
                if "feature_scaler" in checkpoint:
                    self._feature_scaler = checkpoint["feature_scaler"]
                if "target_scaler" in checkpoint:
                    self._target_scaler = checkpoint["target_scaler"]

                # Set precision
                if self.precision == "float64":
                    self._surrogate_model = self._surrogate_model.double()
                else:
                    self._surrogate_model = self._surrogate_model.float()

            except Exception as e:
                raise QMLModelError(f"Failed to load surrogate model: {e}")

        self._is_loaded = True

    def validate_molecule(self, molecule: Union[str, Chem.Mol]) -> bool:
        """Validate if molecule is suitable for GFN2-xTB."""
        if isinstance(molecule, str):
            mol = Chem.MolFromSmiles(molecule)
        else:
            mol = molecule

        if mol is None:
            return False

        # Check supported elements
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() not in self.SUPPORTED_ELEMENTS:
                return False

        # Check reasonable molecule size (xTB can handle large molecules)
        if mol.GetNumAtoms() > 1000:
            return False

        return True

    def _compute_molecular_features(self, molecule: Chem.Mol) -> np.ndarray:
        """Compute molecular features for surrogate model input."""
        features = []

        # Basic molecular properties
        features.append(molecule.GetNumAtoms())
        features.append(molecule.GetNumBonds())
        features.append(Chem.rdMolDescriptors.CalcExactMolWt(molecule))
        features.append(Chem.rdMolDescriptors.CalcNumRotatableBonds(molecule))
        features.append(Chem.rdMolDescriptors.CalcNumAromaticRings(molecule))
        features.append(Chem.rdMolDescriptors.CalcNumAliphaticRings(molecule))

        # Electronic properties estimates
        features.append(Chem.rdMolDescriptors.CalcNumHBA(molecule))
        features.append(Chem.rdMolDescriptors.CalcNumHBD(molecule))
        features.append(Chem.rdMolDescriptors.CalcTPSA(molecule))

        # Atomic composition
        atomic_counts = [0] * 10  # Top 10 most common elements
        element_map = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 15: 5, 16: 6, 17: 7, 35: 8, 53: 9}

        for atom in molecule.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            if atomic_num in element_map:
                atomic_counts[element_map[atomic_num]] += 1

        features.extend(atomic_counts)

        # Fingerprint-based features (reduced dimensionality)
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=1024)
        fp_array = np.zeros((1024,))
        DataStructs.ConvertToNumpyArray(morgan_fp, fp_array)

        # Use PCA or feature selection to reduce fingerprint dimensions
        # For now, use first 100 bits as approximation
        features.extend(fp_array[:100].tolist())

        return np.array(features, dtype=np.float32)

    def _run_xtb_calculation(
        self, molecule: Chem.Mol, properties: List[QuantumProperty]
    ) -> Dict[QuantumProperty, Union[float, np.ndarray]]:
        """Run actual xTB calculation."""
        results = {}

        # Generate 3D coordinates
        if molecule.GetNumConformers() == 0:
            AllChem.EmbedMolecule(molecule, randomSeed=42)
            AllChem.UFFOptimizeMolecule(molecule)

        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            # Write XYZ format
            conf = molecule.GetConformer()
            f.write(f"{molecule.GetNumAtoms()}\n")
            f.write("Molecule generated by ChemiAccel\n")

            for i, atom in enumerate(molecule.GetAtoms()):
                pos = conf.GetAtomPosition(i)
                symbol = atom.GetSymbol()
                f.write(f"{symbol} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")

            xyz_file = f.name

        try:
            # Run GFN2-xTB calculation
            cmd = [self.xtb_path, xyz_file, "--gfn", "2", "--json", "--norestart"]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=tempfile.gettempdir(),
            )

            if result.returncode != 0:
                raise QMLModelError(f"xTB calculation failed: {result.stderr}")

            # Parse JSON output
            json_file = Path(xyz_file).with_suffix(".json")
            if json_file.exists():
                with open(json_file) as f:
                    xtb_data = json.load(f)

                # Extract properties
                if QuantumProperty.TOTAL_ENERGY in properties:
                    results[QuantumProperty.TOTAL_ENERGY] = xtb_data.get(
                        "total energy", 0.0
                    )

                if QuantumProperty.HOMO_ENERGY in properties:
                    results[QuantumProperty.HOMO_ENERGY] = xtb_data.get("HOMO", 0.0)

                if QuantumProperty.LUMO_ENERGY in properties:
                    results[QuantumProperty.LUMO_ENERGY] = xtb_data.get("LUMO", 0.0)

                if QuantumProperty.BANDGAP in properties:
                    homo = xtb_data.get("HOMO", 0.0)
                    lumo = xtb_data.get("LUMO", 0.0)
                    results[QuantumProperty.BANDGAP] = lumo - homo

                if QuantumProperty.DIPOLE_MOMENT in properties:
                    dipole = xtb_data.get("dipole moment", [0.0, 0.0, 0.0])
                    dipole_magnitude = np.linalg.norm(dipole)
                    results[QuantumProperty.DIPOLE_MOMENT] = dipole_magnitude

                if QuantumProperty.PARTIAL_CHARGES in properties:
                    charges = xtb_data.get("partial charges", [])
                    results[QuantumProperty.PARTIAL_CHARGES] = np.array(charges)

        finally:
            # Clean up temporary files
            Path(xyz_file).unlink(missing_ok=True)
            Path(xyz_file).with_suffix(".json").unlink(missing_ok=True)

        return results

    def predict(
        self,
        molecules: Union[MolecularData, List[MolecularData], List[str]],
        properties: Optional[List[QuantumProperty]] = None,
        **kwargs,
    ) -> Union[QMLPrediction, List[QMLPrediction]]:
        """
        Predict quantum properties using GFN2-xTB model.

        Args:
            molecules: Input molecules
            properties: Specific properties to compute
            **kwargs: Additional parameters

        Returns:
            QML prediction results
        """
        if not self._is_loaded:
            self.load_model()

        # Handle single molecule input
        single_input = not isinstance(molecules, list)
        if single_input:
            molecules = [molecules]

        # Set default properties
        if properties is None:
            properties = self.get_supported_properties()

        results = []

        for mol in molecules:
            try:
                # Convert to RDKit molecule
                if isinstance(mol, MolecularData):
                    rdkit_mol = mol.rdkit_mol
                elif isinstance(mol, str):
                    rdkit_mol = Chem.MolFromSmiles(mol)
                else:
                    rdkit_mol = mol

                # Validate molecule
                if not self.validate_molecule(rdkit_mol):
                    raise QMLModelError("Molecule not suitable for GFN2-xTB")

                if self.use_xtb_binary:
                    # Use actual xTB calculation
                    computed_properties = self._run_xtb_calculation(
                        rdkit_mol, properties
                    )
                else:
                    # Use surrogate model
                    features = self._compute_molecular_features(rdkit_mol)

                    # Apply feature scaling
                    if self._feature_scaler:
                        features = self._feature_scaler.transform(
                            features.reshape(1, -1)
                        )[0]

                    # Convert to tensor
                    feature_tensor = torch.FloatTensor(features).to(self.device)
                    if self.precision == "float64":
                        feature_tensor = feature_tensor.double()

                    # Predict with surrogate model
                    with torch.no_grad():
                        predictions = self._surrogate_model(feature_tensor.unsqueeze(0))
                        predictions = predictions.cpu().numpy()[0]

                    # Apply target scaling
                    if self._target_scaler:
                        predictions = self._target_scaler.inverse_transform(
                            predictions.reshape(1, -1)
                        )[0]

                    # Map predictions to properties
                    computed_properties = {}
                    for prop in properties:
                        if prop in self._property_configs:
                            config = self._property_configs[prop]
                            if config["index"] is not None:
                                computed_properties[prop] = predictions[config["index"]]

                # Create prediction object
                prediction = QMLPrediction(
                    properties=computed_properties,
                    model_name="GFN2-xTB"
                    + (" (Binary)" if self.use_xtb_binary else " (Surrogate)"),
                    confidence=0.85 if self.use_xtb_binary else 0.75,
                )

                results.append(prediction)

            except Exception as e:
                # Create error prediction
                error_prediction = QMLPrediction(
                    properties={},
                    confidence=0.0,
                    model_name="GFN2-xTB",
                )
                results.append(error_prediction)

                if not kwargs.get("ignore_errors", False):
                    raise QMLModelError(f"GFN2-xTB prediction failed: {e}")

        return results[0] if single_input else results

    def get_supported_properties(self) -> List[QuantumProperty]:
        """Get quantum properties supported by GFN2-xTB."""
        return list(self._property_configs.keys())

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed GFN2-xTB model information."""
        base_info = super().get_model_info()
        xtb_info = {
            "method": "GFN2-xTB",
            "calculation_type": "Binary" if self.use_xtb_binary else "Surrogate",
            "xtb_path": self.xtb_path if self.use_xtb_binary else None,
            "supported_elements": "H-Rn (Z=1-86)",
            "max_atoms": 1000,
        }
        return {**base_info, **xtb_info}
