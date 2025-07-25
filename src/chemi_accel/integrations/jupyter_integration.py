"""
Jupyter notebook integration for ChemiAccel.

This module provides rich display capabilities for molecular structures
and calculation results in Jupyter notebooks, including interactive
visualizations and progress bars.
"""

import warnings
from typing import List, Optional, Union

# Jupyter/IPython imports with error handling
try:
    from IPython import get_ipython
    from IPython.core.magic import Magics, line_magic, magics_class
    from IPython.display import HTML, SVG, Image, display

    JUPYTER_AVAILABLE = True
except ImportError:
    display = HTML = SVG = Image = None
    Magics = magics_class = line_magic = None
    get_ipython = lambda: None
    JUPYTER_AVAILABLE = False

# Optional dependencies for molecular visualization
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D

    RDKIT_AVAILABLE = True
except ImportError:
    Chem = Draw = rdMolDraw2D = None
    RDKIT_AVAILABLE = False

try:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = patches = None
    MATPLOTLIB_AVAILABLE = False

import pandas as pd

from ..core.calculator import Calculator
from ..utils.logging import get_logger

logger = get_logger()


class MolecularDisplayMixin:
    """Mixin class for rich molecular display in Jupyter notebooks.

    This mixin adds visualization capabilities to Calculator instances
    when used in Jupyter notebook environments.
    """

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        if not JUPYTER_AVAILABLE:
            return str(self)

        # Get calculator info
        info = self.get_backend_info()

        html = f"""
        <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #f9f9f9;">
            <h3 style="margin-top: 0; color: #333;">ChemiAccel Calculator</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 5px; font-weight: bold;">Backend:</td>
                    <td style="padding: 5px;">{info['backend_type']}</td>
                </tr>
                <tr>
                    <td style="padding: 5px; font-weight: bold;">Device:</td>
                    <td style="padding: 5px;">{info['device'] or 'auto'}</td>
                </tr>
                <tr>
                    <td style="padding: 5px; font-weight: bold;">Batch Size:</td>
                    <td style="padding: 5px;">{info['batch_size']}</td>
                </tr>
                <tr>
                    <td style="padding: 5px; font-weight: bold;">Configured Descriptors:</td>
                    <td style="padding: 5px;">{info['num_descriptors']}</td>
                </tr>
                <tr>
                    <td style="padding: 5px; font-weight: bold;">Available Descriptors:</td>
                    <td style="padding: 5px;">{info['available_descriptors']}</td>
                </tr>
            </table>
        </div>
        """
        return html

    def display_molecules(
        self,
        molecules: Union[List[str], pd.Series],
        max_display: int = 10,
        mols_per_row: int = 4,
        mol_size: tuple = (200, 200),
    ) -> None:
        """Display molecular structures in Jupyter notebook.

        Args:
            molecules: SMILES strings to display
            max_display: Maximum number of molecules to display
            mols_per_row: Number of molecules per row
            mol_size: Size of each molecule image (width, height)
        """
        if not JUPYTER_AVAILABLE:
            print("Jupyter display not available")
            return

        if not RDKIT_AVAILABLE:
            print("RDKit not available for molecular visualization")
            return

        # Convert input to list
        if isinstance(molecules, pd.Series):
            smiles_list = molecules.tolist()
        else:
            smiles_list = list(molecules)

        # Limit display
        if len(smiles_list) > max_display:
            smiles_list = smiles_list[:max_display]
            print(f"Displaying first {max_display} molecules out of {len(molecules)}")

        # Parse molecules
        mols = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mols.append((mol, smiles))
            else:
                mols.append((None, smiles))

        # Create grid display
        img = Draw.MolsToGridImage(
            [mol for mol, _ in mols if mol is not None],
            molsPerRow=mols_per_row,
            subImgSize=mol_size,
            legends=[
                smiles[:20] + ("..." if len(smiles) > 20 else "")
                for mol, smiles in mols
                if mol is not None
            ],
        )

        display(img)

    def display_descriptor_summary(self, results_df: pd.DataFrame) -> None:
        """Display summary of calculated descriptors.

        Args:
            results_df: DataFrame with calculated descriptors
        """
        if not JUPYTER_AVAILABLE:
            print(results_df.describe())
            return

        # Get descriptor columns (exclude mol_id and smiles)
        desc_cols = [
            col for col in results_df.columns if col not in ["mol_id", "smiles"]
        ]

        if not desc_cols:
            print("No descriptor columns found")
            return

        # Create summary statistics
        summary = results_df[desc_cols].describe()

        # Create HTML table
        html = f"""
        <div style="margin: 10px 0;">
            <h4>Descriptor Summary ({len(results_df)} molecules, {len(desc_cols)} descriptors)</h4>
            {summary.to_html(classes='table table-striped', table_id='descriptor-summary')}
        </div>
        <style>
        #descriptor-summary {{
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 12px;
        }}
        #descriptor-summary th, #descriptor-summary td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: right;
        }}
        #descriptor-summary th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        </style>
        """

        display(HTML(html))

    def plot_descriptor_distribution(
        self, results_df: pd.DataFrame, descriptor: str, bins: int = 30
    ) -> None:
        """Plot distribution of a specific descriptor.

        Args:
            results_df: DataFrame with calculated descriptors
            descriptor: Name of descriptor to plot
            bins: Number of histogram bins
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available for plotting")
            return

        if descriptor not in results_df.columns:
            print(f"Descriptor '{descriptor}' not found in results")
            return

        plt.figure(figsize=(8, 5))
        plt.hist(
            results_df[descriptor].dropna(), bins=bins, alpha=0.7, edgecolor="black"
        )
        plt.xlabel(descriptor)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {descriptor}")
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_descriptor_correlation(
        self, results_df: pd.DataFrame, descriptors: Optional[List[str]] = None
    ) -> None:
        """Plot correlation matrix of descriptors.

        Args:
            results_df: DataFrame with calculated descriptors
            descriptors: List of descriptors to include (None for all)
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available for plotting")
            return

        # Get descriptor columns
        desc_cols = [
            col for col in results_df.columns if col not in ["mol_id", "smiles"]
        ]

        if descriptors is not None:
            desc_cols = [col for col in desc_cols if col in descriptors]

        if len(desc_cols) < 2:
            print("Need at least 2 descriptors for correlation plot")
            return

        # Calculate correlation matrix
        corr_matrix = results_df[desc_cols].corr()

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)

        # Add colorbar
        plt.colorbar(im, ax=ax)

        # Set ticks and labels
        ax.set_xticks(range(len(desc_cols)))
        ax.set_yticks(range(len(desc_cols)))
        ax.set_xticklabels(desc_cols, rotation=45, ha="right")
        ax.set_yticklabels(desc_cols)

        # Add correlation values
        for i in range(len(desc_cols)):
            for j in range(len(desc_cols)):
                text = ax.text(
                    j,
                    i,
                    f"{corr_matrix.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

        plt.title("Descriptor Correlation Matrix")
        plt.tight_layout()
        plt.show()


if JUPYTER_AVAILABLE and magics_class is not None and line_magic is not None:

    @magics_class
    class ChemiAccelMagics(Magics):
        """IPython magic commands for ChemiAccel."""

        @line_magic
        def chemi_accel_info(self, line):
            """Display ChemiAccel system information."""
            from ..core.backend import get_available_backends
            from ..utils.config import get_config

            config = get_config()
            available_backends = get_available_backends()

            info_html = f"""
            <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px;">
                <h3>ChemiAccel System Information</h3>
                <p><strong>Available Backends:</strong> {', '.join([b.value for b in available_backends])}</p>
                <p><strong>Default Backend:</strong> {config.backend.preferred_backend.value}</p>
                <p><strong>Default Batch Size:</strong> {config.performance.batch_size}</p>
                <p><strong>Memory Pool Enabled:</strong> {config.performance.enable_memory_pool}</p>
            </div>
            """

            display(HTML(info_html))

        @line_magic
        def chemi_accel_benchmark(self, line):
            """Run a quick benchmark of ChemiAccel."""
            calc = Calculator()
            calc.configure_descriptors("basic")

            # Run benchmark
            results = calc.benchmark_performance(num_molecules=100, num_runs=1)

            benchmark_html = f"""
            <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px; background-color: #f0f8ff;">
                <h3>ChemiAccel Benchmark Results</h3>
                <p><strong>Backend:</strong> {results['backend_type']}</p>
                <p><strong>Molecules Processed:</strong> {results['num_molecules']}</p>
                <p><strong>Processing Time:</strong> {results['avg_time_seconds']:.3f} seconds</p>
                <p><strong>Throughput:</strong> {results['molecules_per_second']:.1f} molecules/second</p>
                <p><strong>Descriptor Throughput:</strong> {results['descriptors_per_second']:.1f} descriptors/second</p>
            </div>
            """

            display(HTML(benchmark_html))

else:
    # Create dummy class when Jupyter not available
    class ChemiAccelMagics:
        pass


def enable_rich_display():
    """Enable rich display features for ChemiAccel in Jupyter notebooks.

    This function registers display formatters and magic commands for
    enhanced Jupyter notebook integration.

    Example:
        >>> import chemi_accel as ca
        >>> ca.enable_rich_display()
        >>>
        >>> calc = ca.Calculator()
        >>> calc  # Will display rich HTML representation
    """
    if not JUPYTER_AVAILABLE:
        warnings.warn("Jupyter/IPython not available, rich display not enabled")
        return

    # Get IPython instance
    ip = get_ipython()
    if ip is None:
        warnings.warn("Not running in IPython/Jupyter environment")
        return

    # Add MolecularDisplayMixin to Calculator class
    if not hasattr(Calculator, "_repr_html_"):
        Calculator._repr_html_ = MolecularDisplayMixin._repr_html_
        Calculator.display_molecules = MolecularDisplayMixin.display_molecules
        Calculator.display_descriptor_summary = (
            MolecularDisplayMixin.display_descriptor_summary
        )
        Calculator.plot_descriptor_distribution = (
            MolecularDisplayMixin.plot_descriptor_distribution
        )
        Calculator.plot_descriptor_correlation = (
            MolecularDisplayMixin.plot_descriptor_correlation
        )

    # Register magic commands if available
    if hasattr(ChemiAccelMagics, "chemi_accel_info"):
        ip.register_magic_function(
            ChemiAccelMagics.chemi_accel_info, "line", "chemi_accel_info"
        )
        ip.register_magic_function(
            ChemiAccelMagics.chemi_accel_benchmark, "line", "chemi_accel_benchmark"
        )

    logger.info("ChemiAccel rich display features enabled")


def create_progress_callback(total_molecules: int):
    """Create a progress callback for long-running calculations.

    Args:
        total_molecules: Total number of molecules to process

    Returns:
        Callable: Progress callback function
    """
    if not JUPYTER_AVAILABLE:
        return lambda x: None

    try:
        from tqdm.notebook import tqdm

        pbar = tqdm(total=total_molecules, desc="Calculating descriptors")

        def update_progress(completed: int):
            pbar.n = completed
            pbar.refresh()

        return update_progress
    except ImportError:
        return lambda x: None


# Convenience function for notebook users
def quick_demo(smiles_list: Optional[List[str]] = None):
    """Run a quick demonstration of ChemiAccel features in Jupyter.

    Args:
        smiles_list: Optional list of SMILES to use for demo

    Example:
        >>> import chemi_accel as ca
        >>> ca.quick_demo()
    """
    if not JUPYTER_AVAILABLE:
        print("Jupyter notebook required for demo")
        return

    # Enable rich display
    enable_rich_display()

    # Default demo molecules
    if smiles_list is None:
        smiles_list = [
            "CCO",  # ethanol
            "c1ccccc1",  # benzene
            "CC(=O)O",  # acetic acid
            "CCN(CC)CC",  # triethylamine
            "c1ccc2c(c1)ccccc2",  # naphthalene
        ]

    print("🧪 ChemiAccel Quick Demo")
    print("=" * 50)

    # Initialize calculator
    calc = Calculator()
    calc.configure_descriptors("basic")

    # Display calculator info
    display(calc)

    # Display molecules
    print("\n📋 Input Molecules:")
    calc.display_molecules(smiles_list)

    # Calculate descriptors
    print("\n⚡ Calculating Descriptors...")
    results = calc.pandas(smiles_list)

    # Display results summary
    calc.display_descriptor_summary(results)

    # Plot example distribution
    if "molecular_weight" in results.columns:
        print("\n📊 Example Distribution Plot:")
        calc.plot_descriptor_distribution(results, "molecular_weight")

    print(
        "\n✅ Demo complete! Try calc.plot_descriptor_correlation(results) for more visualizations."
    )

    return calc, results
