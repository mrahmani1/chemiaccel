"""
Performance benchmarks for advanced RDKit integration features.

This module provides comprehensive performance tests comparing ChemiAccel's
advanced features against direct RDKit CPU implementations.
"""

import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from chemi_accel.descriptors.advanced import (
    DrugLikenessFilters,
    MaximumCommonSubstructure,
    PharmacophoreAnalyzer,
    SubstructureSearcher,
)
from chemi_accel.utils.rdkit_integration import parse_smiles_batch

# Extended test dataset with diverse molecular structures
BENCHMARK_SMILES = [
    # Small molecules
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "CCN(CC)CC",
    "CC(C)C(=O)O",
    # Drug-like molecules
    "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # Ibuprofen
    "CC(=O)Nc1ccc(O)cc1",  # Acetaminophen
    "CN1CCC[C@H]1c2cccnc2",  # Nicotine
    "CC1=C(C(=O)N(N1C)c2ccccc2)C(=O)O",  # Phenylbutazone
    "Cc1oncc1C(=O)Nc2ccc(cc2)N(=O)=O",  # Furazolidone
    # Natural products
    "CC(C)CCCC(C)CCCC(C)CCCC(C)C",  # Phytol
    "CC1=CC(=O)C=C(C1=O)C",  # 2,5-dimethyl-1,4-benzoquinone
    "c1ccc2c(c1)cccn2",  # Quinoline
    "CC(C)(C)c1ccc(O)cc1",  # 4-tert-butylphenol
    # Complex structures
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "CC(C)NCC(c1ccc(O)c(c1)O)O",  # Isoproterenol
    "COc1cc2c(cc1OC)C(=O)C(=O)c3cc(OC)c(OC)cc3-2",  # Dantron derivative
    # Aromatic compounds
    "c1ccc2c(c1)ccc3c2ccc4c3cccc4",  # Anthracene
    "c1ccc2cc3ccccc3cc2c1",  # Phenanthrene
    "c1ccc2c(c1)sc3c2cccc3",  # Dibenzothiophene
    # Heterocycles
    "c1cnc2c(c1)cccn2",  # 1,8-naphthyridine
    "c1coc2c1cccc2",  # Benzofuran
    "c1csc2c1cccc2",  # Benzothiophene
    "c1c[nH]c2c1cccc2",  # Indole
]


class AdvancedFeatureBenchmark:
    """Comprehensive benchmark suite for advanced features."""

    def __init__(self, molecules_list: List[str]):
        """Initialize benchmark with molecule list."""
        self.smiles_list = molecules_list
        self.molecules = parse_smiles_batch(molecules_list, error_handling="skip")
        self.results = {}

        print(f"Initialized benchmark with {len(self.molecules)} molecules")

    def benchmark_substructure_search(self, n_runs: int = 3) -> Dict[str, float]:
        """Benchmark substructure searching performance."""
        print("Benchmarking substructure search...")

        patterns = {
            "alcohol": "[OH]",
            "aromatic_ring": "c1ccccc1",
            "carbonyl": "[C]=O",
            "amine": "[N]",
            "ester": "[C](=O)[O][C]",
            "amide": "[C](=O)[N]",
            "nitro": "[N+](=O)[O-]",
            "sulfone": "[S](=O)(=O)",
            "phosphate": "[P](=O)([O])([O])[O]",
            "halogen": "[F,Cl,Br,I]",
        }

        # ChemiAccel implementation
        searcher = SubstructureSearcher(n_jobs=1)
        chemi_times = []

        for _ in range(n_runs):
            start_time = time.time()
            chemi_results = searcher.search_patterns(
                self.molecules, patterns, use_multiprocessing=False
            )
            chemi_times.append(time.time() - start_time)

        # Direct RDKit implementation
        rdkit_times = []

        for _ in range(n_runs):
            start_time = time.time()
            rdkit_results = self._rdkit_substructure_search(patterns)
            rdkit_times.append(time.time() - start_time)

        chemi_avg = np.mean(chemi_times)
        rdkit_avg = np.mean(rdkit_times)
        speedup = rdkit_avg / chemi_avg if chemi_avg > 0 else 0

        results = {
            "chemi_accel_time": chemi_avg,
            "rdkit_time": rdkit_avg,
            "speedup": speedup,
            "chemi_std": np.std(chemi_times),
            "rdkit_std": np.std(rdkit_times),
        }

        print(
            f"Substructure search - ChemiAccel: {chemi_avg:.4f}s, RDKit: {rdkit_avg:.4f}s, Speedup: {speedup:.2f}x"
        )

        return results

    def _rdkit_substructure_search(self, patterns: Dict[str, str]) -> pd.DataFrame:
        """Direct RDKit substructure search implementation."""
        try:
            from rdkit import Chem
        except ImportError:
            return pd.DataFrame()

        results = []
        for mol_data in self.molecules:
            rdkit_mol = mol_data.backend_data.get("rdkit_mol")
            if rdkit_mol is None:
                continue

            row = {"mol_id": mol_data.mol_id, "smiles": mol_data.smiles}

            for pattern_name, smarts in patterns.items():
                try:
                    pattern_mol = Chem.MolFromSmarts(smarts)
                    if pattern_mol:
                        matches = rdkit_mol.GetSubstructMatches(pattern_mol)
                        row[f"{pattern_name}_count"] = len(matches)
                    else:
                        row[f"{pattern_name}_count"] = 0
                except Exception:
                    row[f"{pattern_name}_count"] = 0

            results.append(row)

        return pd.DataFrame(results)

    def benchmark_mcs_calculation(self, n_runs: int = 3) -> Dict[str, float]:
        """Benchmark MCS calculation performance."""
        print("Benchmarking MCS calculation...")

        # Use smaller subsets for MCS (computationally expensive)
        test_pairs = [
            self.molecules[i : i + 2] for i in range(0, min(6, len(self.molecules)), 2)
        ]

        mcs_calc = MaximumCommonSubstructure(timeout=10)
        chemi_times = []

        for _ in range(n_runs):
            start_time = time.time()
            for pair in test_pairs:
                if len(pair) == 2:
                    try:
                        mcs_calc.calculate_mcs(pair)
                    except Exception:
                        pass
            chemi_times.append(time.time() - start_time)

        # Direct RDKit MCS
        rdkit_times = []

        for _ in range(n_runs):
            start_time = time.time()
            for pair in test_pairs:
                if len(pair) == 2:
                    try:
                        self._rdkit_mcs_calculation(pair)
                    except Exception:
                        pass
            rdkit_times.append(time.time() - start_time)

        chemi_avg = np.mean(chemi_times)
        rdkit_avg = np.mean(rdkit_times)
        speedup = rdkit_avg / chemi_avg if chemi_avg > 0 else 0

        results = {
            "chemi_accel_time": chemi_avg,
            "rdkit_time": rdkit_avg,
            "speedup": speedup,
            "num_pairs": len(test_pairs),
        }

        print(
            f"MCS calculation - ChemiAccel: {chemi_avg:.4f}s, RDKit: {rdkit_avg:.4f}s, Speedup: {speedup:.2f}x"
        )

        return results

    def _rdkit_mcs_calculation(self, molecules: List) -> Dict:
        """Direct RDKit MCS calculation."""
        try:
            from rdkit.Chem import rdFMCS

            rdkit_mols = []
            for mol_data in molecules:
                rdkit_mol = mol_data.backend_data.get("rdkit_mol")
                if rdkit_mol:
                    rdkit_mols.append(rdkit_mol)

            if len(rdkit_mols) >= 2:
                mcs_result = rdFMCS.FindMCS(rdkit_mols, timeout=10)
                return {
                    "smarts": mcs_result.smartsString,
                    "num_atoms": mcs_result.numAtoms,
                }
        except Exception:
            pass

        return {"smarts": "", "num_atoms": 0}

    def benchmark_pharmacophore_analysis(self, n_runs: int = 3) -> Dict[str, float]:
        """Benchmark pharmacophore analysis performance."""
        print("Benchmarking pharmacophore analysis...")

        analyzer = PharmacophoreAnalyzer()
        chemi_times = []

        for _ in range(n_runs):
            start_time = time.time()
            for mol_data in self.molecules:
                try:
                    analyzer.calculate_pharmacophore_fingerprint(mol_data)
                except Exception:
                    pass
            chemi_times.append(time.time() - start_time)

        # Direct RDKit pharmacophore analysis
        rdkit_times = []

        for _ in range(n_runs):
            start_time = time.time()
            for mol_data in self.molecules:
                try:
                    self._rdkit_pharmacophore_analysis(mol_data)
                except Exception:
                    pass
            rdkit_times.append(time.time() - start_time)

        chemi_avg = np.mean(chemi_times)
        rdkit_avg = np.mean(rdkit_times)
        speedup = rdkit_avg / chemi_avg if chemi_avg > 0 else 0

        results = {
            "chemi_accel_time": chemi_avg,
            "rdkit_time": rdkit_avg,
            "speedup": speedup,
        }

        print(
            f"Pharmacophore analysis - ChemiAccel: {chemi_avg:.4f}s, RDKit: {rdkit_avg:.4f}s, Speedup: {speedup:.2f}x"
        )

        return results

    def _rdkit_pharmacophore_analysis(self, mol_data) -> Dict:
        """Direct RDKit pharmacophore feature extraction."""
        try:
            from rdkit import Chem

            rdkit_mol = mol_data.backend_data.get("rdkit_mol")
            if rdkit_mol is None:
                return {}

            # Simple feature counting using SMARTS patterns
            features = {
                "donor": "[N;!H0;v3,v4&+1],[O,S;H1;+0],n&H1&+0",
                "acceptor": "[O,S;H1;v2;!$(*-*=[O,N,P,S])],[O,S;H0;v2],[O,S;-],[N;v3;!$(N-*=[O,N,P,S])],n&H0&+0",
                "aromatic": "[a]",
                "hydrophobic": "[CH3X4,CH2X3,CH1X2,F1X1,Cl1X1,Br1X1,I1X1]",
            }

            feature_counts = {}
            for feature_name, smarts in features.items():
                try:
                    pattern_mol = Chem.MolFromSmarts(smarts)
                    if pattern_mol:
                        matches = rdkit_mol.GetSubstructMatches(pattern_mol)
                        feature_counts[feature_name] = len(matches)
                    else:
                        feature_counts[feature_name] = 0
                except Exception:
                    feature_counts[feature_name] = 0

            return feature_counts

        except Exception:
            return {}

    def benchmark_drug_likeness_filters(self, n_runs: int = 3) -> Dict[str, float]:
        """Benchmark drug-likeness filtering performance."""
        print("Benchmarking drug-likeness filters...")

        filters = DrugLikenessFilters()
        chemi_times = []

        for _ in range(n_runs):
            start_time = time.time()
            filters.filter_drug_like_molecules(
                self.molecules,
                require_lipinski=True,
                require_veber=True,
                exclude_pains=False,  # Skip PAINS for speed
            )
            chemi_times.append(time.time() - start_time)

        # Direct RDKit implementation
        rdkit_times = []

        for _ in range(n_runs):
            start_time = time.time()
            self._rdkit_drug_likeness_batch()
            rdkit_times.append(time.time() - start_time)

        chemi_avg = np.mean(chemi_times)
        rdkit_avg = np.mean(rdkit_times)
        speedup = rdkit_avg / chemi_avg if chemi_avg > 0 else 0

        results = {
            "chemi_accel_time": chemi_avg,
            "rdkit_time": rdkit_avg,
            "speedup": speedup,
        }

        print(
            f"Drug-likeness filters - ChemiAccel: {chemi_avg:.4f}s, RDKit: {rdkit_avg:.4f}s, Speedup: {speedup:.2f}x"
        )

        return results

    def _rdkit_drug_likeness_batch(self) -> List[Dict]:
        """Direct RDKit drug-likeness calculation."""
        try:
            from rdkit.Chem import Crippen, Descriptors

            results = []
            for mol_data in self.molecules:
                rdkit_mol = mol_data.backend_data.get("rdkit_mol")
                if rdkit_mol is None:
                    continue

                try:
                    # Lipinski descriptors
                    mw = Descriptors.MolWt(rdkit_mol)
                    logp = Crippen.MolLogP(rdkit_mol)
                    hbd = Descriptors.NumHDonors(rdkit_mol)
                    hba = Descriptors.NumHAcceptors(rdkit_mol)

                    # Veber descriptors
                    tpsa = Descriptors.TPSA(rdkit_mol)
                    rotbonds = Descriptors.NumRotatableBonds(rdkit_mol)

                    # Calculate violations
                    lipinski_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])

                    veber_violations = sum([tpsa > 140, rotbonds > 10])

                    result = {
                        "mol_id": mol_data.mol_id,
                        "molecular_weight": mw,
                        "logp": logp,
                        "hbd": hbd,
                        "hba": hba,
                        "tpsa": tpsa,
                        "rotatable_bonds": rotbonds,
                        "lipinski_violations": lipinski_violations,
                        "veber_violations": veber_violations,
                        "lipinski_compliant": lipinski_violations == 0,
                        "veber_compliant": veber_violations == 0,
                    }

                    results.append(result)

                except Exception:
                    continue

            return results

        except Exception:
            return []

    def run_full_benchmark(self) -> Dict[str, Dict[str, float]]:
        """Run complete benchmark suite."""
        print(f"Running full benchmark suite with {len(self.molecules)} molecules...")
        print("=" * 60)

        results = {}

        # Substructure search benchmark
        results["substructure_search"] = self.benchmark_substructure_search()

        # MCS calculation benchmark
        results["mcs_calculation"] = self.benchmark_mcs_calculation()

        # Pharmacophore analysis benchmark
        results["pharmacophore_analysis"] = self.benchmark_pharmacophore_analysis()

        # Drug-likeness filters benchmark
        results["drug_likeness_filters"] = self.benchmark_drug_likeness_filters()

        self.results = results
        return results

    def generate_report(self, save_path: str = None) -> str:
        """Generate benchmark report."""
        if not self.results:
            return "No benchmark results available. Run benchmark first."

        report = []
        report.append("ChemiAccel Advanced Features Performance Benchmark")
        report.append("=" * 55)
        report.append(f"Dataset: {len(self.molecules)} molecules")
        report.append("")

        for feature_name, result in self.results.items():
            report.append(f"{feature_name.replace('_', ' ').title()}:")
            report.append(f"  ChemiAccel: {result['chemi_accel_time']:.4f}s")
            report.append(f"  RDKit:      {result['rdkit_time']:.4f}s")
            report.append(f"  Speedup:    {result['speedup']:.2f}x")

            if result["speedup"] > 1:
                report.append(f"  ✓ ChemiAccel is {result['speedup']:.2f}x faster")
            elif result["speedup"] < 1:
                report.append(f"  ⚠ RDKit is {1/result['speedup']:.2f}x faster")
            else:
                report.append("  = Similar performance")

            report.append("")

        # Summary
        speedups = [r["speedup"] for r in self.results.values()]
        avg_speedup = np.mean(speedups)

        report.append("Summary:")
        report.append(f"  Average speedup: {avg_speedup:.2f}x")
        report.append(f"  Best speedup:    {max(speedups):.2f}x")
        report.append(f"  Worst speedup:   {min(speedups):.2f}x")

        report_text = "\n".join(report)

        if save_path:
            Path(save_path).write_text(report_text)
            print(f"Report saved to: {save_path}")

        return report_text

    def plot_results(self, save_path: str = None):
        """Create visualization of benchmark results."""
        if not self.results:
            print("No results to plot. Run benchmark first.")
            return

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Prepare data for plotting
            features = []
            chemi_times = []
            rdkit_times = []
            speedups = []

            for feature_name, result in self.results.items():
                features.append(feature_name.replace("_", " ").title())
                chemi_times.append(result["chemi_accel_time"])
                rdkit_times.append(result["rdkit_time"])
                speedups.append(result["speedup"])

            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Plot 1: Execution times
            x = np.arange(len(features))
            width = 0.35

            ax1.bar(x - width / 2, chemi_times, width, label="ChemiAccel", alpha=0.8)
            ax1.bar(x + width / 2, rdkit_times, width, label="RDKit", alpha=0.8)

            ax1.set_xlabel("Features")
            ax1.set_ylabel("Execution Time (seconds)")
            ax1.set_title("Execution Time Comparison")
            ax1.set_xticks(x)
            ax1.set_xticklabels(features, rotation=45, ha="right")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Speedup factors
            colors = ["green" if s >= 1 else "red" for s in speedups]
            bars = ax2.bar(features, speedups, color=colors, alpha=0.7)

            ax2.axhline(y=1, color="black", linestyle="--", alpha=0.5)
            ax2.set_xlabel("Features")
            ax2.set_ylabel("Speedup Factor (ChemiAccel vs RDKit)")
            ax2.set_title("Performance Speedup")
            ax2.set_xticklabels(features, rotation=45, ha="right")
            ax2.grid(True, alpha=0.3)

            # Add speedup values on bars
            for bar, speedup in zip(bars, speedups):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.05,
                    f"{speedup:.2f}x",
                    ha="center",
                    va="bottom",
                )

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Plot saved to: {save_path}")

            plt.show()

        except ImportError:
            print("Matplotlib/Seaborn not available for plotting")


def run_benchmark():
    """Run the complete benchmark suite."""
    print("Starting ChemiAccel Advanced Features Benchmark")
    print("=" * 50)

    # Initialize benchmark
    benchmark = AdvancedFeatureBenchmark(BENCHMARK_SMILES)

    # Run benchmarks
    results = benchmark.run_full_benchmark()

    # Generate and display report
    report = benchmark.generate_report()
    print("\n" + report)

    # Save results
    benchmark.generate_report("benchmark_report.txt")
    benchmark.plot_results("benchmark_results.png")

    return results


if __name__ == "__main__":
    run_benchmark()
