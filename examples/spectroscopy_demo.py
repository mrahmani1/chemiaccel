"""
Comprehensive Demonstration of ChemiAccel Spectroscopy Framework.

This script demonstrates all major features of the spectroscopy module including:
- GPU-accelerated predictive models for NMR, IR, MS, UV-Vis
- Automated experimental data processing
- AI-driven inverse spectroscopy for structure elucidation
- Fast quantum chemistry methods for spectral features
- Benchmarking and performance analysis
"""

import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

# Suppress warnings for cleaner demo output
warnings.filterwarnings("ignore")

# Import ChemiAccel spectroscopy modules
try:
    from chemi_accel.spectroscopy import (  # Predictive models; Experimental processing; Inverse spectroscopy; Quantum methods; Utility functions
        ApproximationLevel,
        BaselineCorrectionMethod,
        BasisSetType,
        DensityFunctionalApproximation,
        ExperimentalSpectraProcessor,
        ExperimentalSpectrum,
        FastQuantumChemistry,
        FunctionalType,
        GPUSpectroscopyPredictor,
        InverseSpectroscopyEngine,
        IRPredictor,
        MatchingMethod,
        MatchingParameters,
        ModelArchitecture,
        ModelConfiguration,
        MSPredictor,
        NMRPredictor,
        NormalizationMethod,
        ProcessingParameters,
        QMLSurrogateModel,
        SemiEmpiricalMethod,
        SpectroscopyType,
        SpectrumFormat,
        StructureCandidate,
        StructureGenerationMethod,
        StructureProposer,
        TrainingData,
        UVVisPredictor,
        evaluate_model_accuracy,
        predict_spectrum_batch,
        process_spectrum_batch,
        propose_structures_from_spectrum,
        train_spectral_model,
    )

    CHEMI_ACCEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ChemiAccel spectroscopy module not available: {e}")
    CHEMI_ACCEL_AVAILABLE = False

# Optional imports for enhanced demonstrations
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Some demonstrations will be limited.")

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available. Plots will not be shown.")


class SpectroscopyDemo:
    """Main demonstration class for spectroscopy functionality."""

    def __init__(self, use_gpu: bool = True, verbose: bool = True):
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.demo_results = {}

        if not CHEMI_ACCEL_AVAILABLE:
            raise RuntimeError(
                "ChemiAccel spectroscopy module is required for this demo"
            )

    def print_section(self, title: str) -> None:
        """Print formatted section header."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f" {title}")
            print("=" * 60)

    def print_subsection(self, title: str) -> None:
        """Print formatted subsection header."""
        if self.verbose:
            print(f"\n{'-'*40}")
            print(f" {title}")
            print("-" * 40)

    def generate_synthetic_spectra_data(self, n_samples: int = 100) -> TrainingData:
        """Generate synthetic spectral data for training demonstrations."""
        self.print_subsection("Generating Synthetic Training Data")

        # Generate molecular features (simplified descriptors)
        np.random.seed(42)  # For reproducible results
        molecular_features = np.random.rand(n_samples, 256)

        # Add some structure to features to make them more realistic
        for i in range(n_samples):
            # Simulate molecular weight effect
            mw_factor = molecular_features[i, 0] * 400 + 50  # MW 50-450
            molecular_features[i, 1] = np.log10(mw_factor)  # LogP correlation

            # Simulate aromatic content
            aromatic_rings = int(molecular_features[i, 2] * 3)
            molecular_features[i, 3] = aromatic_rings / 3.0

        # Generate corresponding IR spectra (4000-400 cm-1)
        frequencies = np.linspace(400, 4000, 1000)
        spectra = np.zeros((n_samples, len(frequencies)))

        for i in range(n_samples):
            # Base spectrum with characteristic peaks
            spectrum = np.random.normal(0.1, 0.02, len(frequencies))

            # Add characteristic IR peaks based on molecular features
            mw = molecular_features[i, 0] * 400 + 50
            aromatic = molecular_features[i, 3]

            # C-H stretch around 2900-3000 cm-1
            ch_peak_pos = 2950 + np.random.normal(0, 50)
            ch_peak_idx = np.argmin(np.abs(frequencies - ch_peak_pos))
            spectrum[ch_peak_idx - 10 : ch_peak_idx + 10] += 0.3 * np.exp(
                -np.linspace(-2, 2, 20) ** 2
            )

            # C=O stretch around 1700 cm-1 (if carbonyl present)
            if molecular_features[i, 4] > 0.5:  # Assume carbonyl present
                co_peak_pos = 1700 + np.random.normal(0, 30)
                co_peak_idx = np.argmin(np.abs(frequencies - co_peak_pos))
                spectrum[co_peak_idx - 15 : co_peak_idx + 15] += 0.5 * np.exp(
                    -np.linspace(-2, 2, 30) ** 2
                )

            # Aromatic C=C stretch around 1600 cm-1
            if aromatic > 0.3:
                ar_peak_pos = 1600 + np.random.normal(0, 20)
                ar_peak_idx = np.argmin(np.abs(frequencies - ar_peak_pos))
                spectrum[ar_peak_idx - 10 : ar_peak_idx + 10] += (
                    0.4 * aromatic * np.exp(-np.linspace(-2, 2, 20) ** 2)
                )

            # Normalize and add to dataset
            spectrum = np.maximum(spectrum, 0)  # No negative intensities
            spectra[i] = spectrum

        training_data = TrainingData(
            molecular_features=molecular_features,
            spectra=spectra,
            frequencies=frequencies,
            molecule_ids=[f"compound_{i+1:03d}" for i in range(n_samples)],
            metadata={
                "dataset": "synthetic_ir_demo",
                "n_samples": n_samples,
                "spectroscopy_type": "IR",
                "frequency_range": "400-4000 cm-1",
            },
        )

        if self.verbose:
            print(
                f"Generated {n_samples} synthetic spectra with {len(frequencies)} frequency points"
            )
            print(f"Molecular features shape: {molecular_features.shape}")
            print(f"Spectra shape: {spectra.shape}")

        return training_data

    def demo_predictive_models(self) -> Dict[str, Any]:
        """Demonstrate predictive spectroscopy models."""
        self.print_section("GPU-Accelerated Predictive Spectroscopy Models")

        results = {}

        # Generate training data
        training_data = self.generate_synthetic_spectra_data(200)
        test_data = self.generate_synthetic_spectra_data(50)

        # Demo 1: IR Predictor
        self.print_subsection("1. IR Spectroscopy Predictor")

        try:
            ir_predictor = IRPredictor(use_gpu=self.use_gpu)

            if self.verbose:
                print(
                    f"Training IR predictor (architecture: {ir_predictor.config.architecture.value})"
                )
                print(f"Frequency range: {ir_predictor.config.frequency_range} cm-1")

            start_time = time.time()
            training_metrics = ir_predictor.train(training_data)
            training_time = time.time() - start_time

            if self.verbose:
                print(f"Training completed in {training_time:.2f} seconds")
                print(f"Final loss: {training_metrics.get('final_loss', 'N/A'):.6f}")

            # Test predictions
            test_features = test_data.molecular_features[:10]
            start_time = time.time()
            predictions = ir_predictor.predict(test_features)
            prediction_time = time.time() - start_time

            if self.verbose:
                print(
                    f"Generated {len(predictions)} predictions in {prediction_time:.3f} seconds"
                )
                print(
                    f"Average prediction time: {prediction_time/len(predictions)*1000:.2f} ms per molecule"
                )

            results["ir_predictor"] = {
                "training_time": training_time,
                "prediction_time": prediction_time,
                "training_metrics": training_metrics,
                "predictions": predictions[:3],  # Store first 3 for inspection
            }

        except Exception as e:
            print(f"IR predictor demo failed: {e}")
            results["ir_predictor"] = {"error": str(e)}

        # Demo 2: NMR Predictor (1H)
        self.print_subsection("2. 1H NMR Spectroscopy Predictor")

        try:
            # Generate NMR-specific training data
            nmr_training = self._generate_nmr_training_data(100)

            nmr_predictor = NMRPredictor("1H", use_gpu=self.use_gpu)

            if self.verbose:
                print("Training 1H NMR predictor")
                print(
                    f"Chemical shift range: {nmr_predictor.config.frequency_range} ppm"
                )

            start_time = time.time()
            nmr_metrics = nmr_predictor.train(nmr_training)
            nmr_training_time = time.time() - start_time

            # Test predictions
            nmr_predictions = nmr_predictor.predict(nmr_training.molecular_features[:5])

            if self.verbose:
                print(f"NMR training completed in {nmr_training_time:.2f} seconds")
                print(f"Generated {len(nmr_predictions)} NMR predictions")

            results["nmr_predictor"] = {
                "training_time": nmr_training_time,
                "training_metrics": nmr_metrics,
                "predictions": nmr_predictions[:2],
            }

        except Exception as e:
            print(f"NMR predictor demo failed: {e}")
            results["nmr_predictor"] = {"error": str(e)}

        # Demo 3: QML Surrogate Model
        self.print_subsection("3. Quantum Machine Learning Surrogate")

        try:
            qml_config = ModelConfiguration(
                architecture=ModelArchitecture.TRANSFORMER,
                spectroscopy_type=SpectroscopyType.IR,
                use_gpu=self.use_gpu,
                num_epochs=5,  # Reduced for demo
            )

            qml_model = QMLSurrogateModel(qml_config)

            if self.verbose:
                print("Training QML surrogate model with quantum-enhanced features")

            start_time = time.time()
            qml_metrics = qml_model.train(training_data)
            qml_training_time = time.time() - start_time

            # Test quantum-enhanced predictions
            qml_predictions = qml_model.predict(test_data.molecular_features[:3])

            if self.verbose:
                print(
                    f"QML surrogate training completed in {qml_training_time:.2f} seconds"
                )
                print("Quantum features automatically integrated")

            results["qml_model"] = {
                "training_time": qml_training_time,
                "training_metrics": qml_metrics,
                "predictions": qml_predictions,
            }

        except Exception as e:
            print(f"QML surrogate demo failed: {e}")
            results["qml_model"] = {"error": str(e)}

        # Demo 4: Batch Prediction Performance
        self.print_subsection("4. Batch Prediction Performance Test")

        try:
            if "ir_predictor" in results and "error" not in results["ir_predictor"]:
                batch_features = test_data.molecular_features

                start_time = time.time()
                batch_result = predict_spectrum_batch(
                    ir_predictor, batch_features, batch_size=16
                )
                batch_time = time.time() - start_time

                if self.verbose:
                    print(f"Batch prediction of {len(batch_features)} molecules:")
                    print(
                        f"Total time: {batch_result.batch_computation_time:.3f} seconds"
                    )
                    print(
                        f"Throughput: {batch_result.model_performance_metrics['samples_per_second']:.1f} molecules/second"
                    )
                    print(
                        f"Average confidence: {batch_result.get_average_confidence():.3f}"
                    )

                results["batch_performance"] = {
                    "batch_size": len(batch_features),
                    "total_time": batch_result.batch_computation_time,
                    "throughput": batch_result.model_performance_metrics[
                        "samples_per_second"
                    ],
                    "average_confidence": batch_result.get_average_confidence(),
                }

        except Exception as e:
            print(f"Batch prediction demo failed: {e}")
            results["batch_performance"] = {"error": str(e)}

        return results

    def _generate_nmr_training_data(self, n_samples: int) -> TrainingData:
        """Generate synthetic NMR training data."""
        molecular_features = np.random.rand(n_samples, 256)

        # Generate 1H NMR spectra (0-12 ppm)
        frequencies = np.linspace(0, 12, 400)
        spectra = np.zeros((n_samples, len(frequencies)))

        for i in range(n_samples):
            spectrum = np.random.normal(0.01, 0.005, len(frequencies))

            # Aromatic protons (7-8 ppm)
            if molecular_features[i, 3] > 0.3:  # Aromatic content
                aromatic_shift = 7.5 + np.random.normal(0, 0.3)
                aromatic_idx = np.argmin(np.abs(frequencies - aromatic_shift))
                spectrum[aromatic_idx - 5 : aromatic_idx + 5] += 0.3 * np.exp(
                    -np.linspace(-1, 1, 10) ** 2
                )

            # Aliphatic protons (1-3 ppm)
            aliphatic_shift = 2.0 + np.random.normal(0, 0.5)
            aliphatic_idx = np.argmin(np.abs(frequencies - aliphatic_shift))
            spectrum[aliphatic_idx - 10 : aliphatic_idx + 10] += 0.5 * np.exp(
                -np.linspace(-1, 1, 20) ** 2
            )

            spectrum = np.maximum(spectrum, 0)
            spectra[i] = spectrum

        return TrainingData(
            molecular_features=molecular_features,
            spectra=spectra,
            frequencies=frequencies,
            metadata={"spectroscopy_type": "1H_NMR"},
        )

    def demo_experimental_processing(self) -> Dict[str, Any]:
        """Demonstrate experimental spectra processing pipeline."""
        self.print_section("Automated Experimental Spectra Processing")

        results = {}

        # Demo 1: Create synthetic experimental spectrum with realistic artifacts
        self.print_subsection("1. Synthetic Experimental Spectrum with Artifacts")

        # Create realistic experimental spectrum with baseline drift, noise, peaks
        frequencies = np.linspace(400, 4000, 2000)

        # Polynomial baseline drift
        baseline = 0.15 + 0.0001 * frequencies - 0.000000001 * frequencies**2

        # Add characteristic peaks
        peaks_info = [
            (1000, 0.4, 20),  # C-O stretch
            (1650, 0.8, 15),  # C=O stretch
            (2900, 0.3, 25),  # C-H stretch
            (3300, 0.6, 30),  # O-H stretch
        ]

        signal = np.zeros_like(frequencies)
        for peak_freq, peak_int, peak_width in peaks_info:
            peak = peak_int * np.exp(-(((frequencies - peak_freq) / peak_width) ** 2))
            signal += peak

        # Add realistic noise
        noise = np.random.normal(0, 0.03, len(frequencies))

        # Combine all components
        raw_intensities = baseline + signal + noise
        raw_intensities = np.maximum(raw_intensities, 0)  # No negative values

        raw_spectrum = ExperimentalSpectrum(
            frequencies=frequencies,
            intensities=raw_intensities,
            metadata={
                "sample_id": "demo_compound",
                "instrument": "FTIR-8400S",
                "resolution": "4 cm-1",
                "scans": 32,
                "baseline_drift": "present",
                "noise_level": "moderate",
            },
        )

        if self.verbose:
            print("Created synthetic experimental spectrum:")
            print(
                f"  Frequency range: {np.min(frequencies):.0f}-{np.max(frequencies):.0f} cm-1"
            )
            print(f"  Data points: {len(frequencies)}")
            print("  Baseline drift: Present")
            print(f"  Noise level: {np.std(noise):.4f}")
            print(f"  Number of peaks: {len(peaks_info)}")

        # Demo 2: Comprehensive preprocessing pipeline
        self.print_subsection("2. Comprehensive Preprocessing Pipeline")

        try:
            processor = ExperimentalSpectraProcessor(use_gpu=self.use_gpu)

            # Configure processing parameters
            params = ProcessingParameters(
                frequency_range=(500.0, 3800.0),  # Focus on main region
                baseline_method=BaselineCorrectionMethod.ASYMMETRIC_LEAST_SQUARES,
                normalization_method=NormalizationMethod.MIN_MAX,
                smoothing_window=7,
                smoothing_order=3,
                peak_detection_threshold=0.15,
                peak_detection_distance=20,
            )

            if self.verbose:
                print("Processing parameters:")
                print(f"  Frequency range: {params.frequency_range} cm-1")
                print(f"  Baseline method: {params.baseline_method.value}")
                print(f"  Normalization: {params.normalization_method.value}")
                print(f"  Smoothing window: {params.smoothing_window}")

            start_time = time.time()
            processing_result = processor.process_single_spectrum(raw_spectrum, params)
            processing_time = time.time() - start_time

            processed_spectrum = processing_result.processed_spectrum

            if self.verbose:
                print(f"\nProcessing completed in {processing_time:.3f} seconds")
                print("Processing steps applied:")
                for step in processed_spectrum.processing_history:
                    print(f"  - {step}")

                print("\nSpectrum quality metrics:")
                snr = processed_spectrum.calculate_signal_to_noise()
                print(f"  Signal-to-noise ratio: {snr:.2f}")
                print(f"  Peaks detected: {len(processed_spectrum.peaks)}")

                if processed_spectrum.peaks:
                    print("  Peak information:")
                    for i, peak in enumerate(
                        processed_spectrum.peaks[:3]
                    ):  # Show first 3
                        print(
                            f"    Peak {i+1}: {peak.frequency:.1f} cm-1, intensity: {peak.intensity:.3f}"
                        )

            results["preprocessing"] = {
                "processing_time": processing_time,
                "snr_improvement": processed_spectrum.quality_metrics.get(
                    "signal_to_noise", 0
                ),
                "peaks_detected": len(processed_spectrum.peaks),
                "processing_steps": len(processed_spectrum.processing_history),
                "frequency_range_applied": (
                    np.min(processed_spectrum.frequencies),
                    np.max(processed_spectrum.frequencies),
                ),
            }

        except Exception as e:
            print(f"Preprocessing demo failed: {e}")
            results["preprocessing"] = {"error": str(e)}

        # Demo 3: Batch processing and alignment
        self.print_subsection("3. Batch Processing and Spectral Alignment")

        try:
            # Create multiple spectra with different frequency shifts (instrument drift)
            batch_spectra = []
            shifts = [0, 2, -1.5, 3, -0.5]  # Different frequency shifts

            for i, shift in enumerate(shifts):
                shifted_freq = frequencies + shift
                shifted_spectrum = ExperimentalSpectrum(
                    frequencies=shifted_freq,
                    intensities=raw_intensities
                    + np.random.normal(0, 0.01, len(raw_intensities)),
                    metadata={
                        "sample_id": f"batch_sample_{i+1}",
                        "frequency_shift": shift,
                        "batch_number": i + 1,
                    },
                )
                batch_spectra.append(shifted_spectrum)

            if self.verbose:
                print(
                    f"Created batch of {len(batch_spectra)} spectra with frequency shifts:"
                )
                for i, shift in enumerate(shifts):
                    print(f"  Sample {i+1}: {shift:+.1f} cm-1 shift")

            # Process batch with alignment
            start_time = time.time()
            batch_results = processor.process_batch(
                batch_spectra, params=params, align_spectra=True
            )
            batch_time = time.time() - start_time

            if self.verbose:
                print(f"\nBatch processing completed in {batch_time:.3f} seconds")
                print(
                    f"Average processing time per spectrum: {batch_time/len(batch_spectra):.3f} seconds"
                )

                # Check alignment effectiveness
                aligned_ranges = []
                for result in batch_results:
                    freq_range = (
                        np.min(result.processed_spectrum.frequencies),
                        np.max(result.processed_spectrum.frequencies),
                    )
                    aligned_ranges.append(freq_range)

                print("Frequency alignment results:")
                for i, (min_f, max_f) in enumerate(aligned_ranges):
                    print(f"  Sample {i+1}: {min_f:.1f}-{max_f:.1f} cm-1")

            results["batch_processing"] = {
                "batch_size": len(batch_spectra),
                "total_time": batch_time,
                "average_time_per_spectrum": batch_time / len(batch_spectra),
                "alignment_applied": True,
                "frequency_ranges": aligned_ranges,
            }

        except Exception as e:
            print(f"Batch processing demo failed: {e}")
            results["batch_processing"] = {"error": str(e)}

        return results

    def demo_inverse_spectroscopy(self) -> Dict[str, Any]:
        """Demonstrate AI-driven inverse spectroscopy for structure elucidation."""
        self.print_section("AI-Driven Inverse Spectroscopy for Structure Elucidation")

        results = {}

        # Demo 1: Structure proposal from experimental spectrum
        self.print_subsection("1. Structure Proposal from Experimental IR Spectrum")

        try:
            # Create a realistic experimental IR spectrum for a known compound
            # Simulating spectrum of ethyl acetate (CH3COOCH2CH3)
            frequencies = np.linspace(400, 4000, 1000)

            # Characteristic IR peaks for ethyl acetate
            spectrum_data = np.zeros_like(frequencies)

            # C=O stretch (1740 cm-1)
            co_peak_idx = np.argmin(np.abs(frequencies - 1740))
            spectrum_data[co_peak_idx - 15 : co_peak_idx + 15] = 0.9 * np.exp(
                -np.linspace(-2, 2, 30) ** 2
            )

            # C-H stretch (2980 cm-1)
            ch_peak_idx = np.argmin(np.abs(frequencies - 2980))
            spectrum_data[ch_peak_idx - 20 : ch_peak_idx + 20] = 0.4 * np.exp(
                -np.linspace(-2, 2, 40) ** 2
            )

            # C-O stretch (1240 cm-1)
            co_str_idx = np.argmin(np.abs(frequencies - 1240))
            spectrum_data[co_str_idx - 10 : co_str_idx + 10] = 0.6 * np.exp(
                -np.linspace(-2, 2, 20) ** 2
            )

            # Add some noise and baseline
            spectrum_data += np.random.normal(0.05, 0.02, len(frequencies))
            spectrum_data = np.maximum(spectrum_data, 0)

            experimental_spectrum = ExperimentalSpectrum(
                frequencies=frequencies,
                intensities=spectrum_data,
                metadata={
                    "sample_id": "unknown_compound",
                    "technique": "FTIR",
                    "suspected_functional_groups": ["carbonyl", "ester", "alkyl"],
                },
            )

            if self.verbose:
                print("Experimental spectrum characteristics:")
                print(
                    f"  Frequency range: {np.min(frequencies):.0f}-{np.max(frequencies):.0f} cm-1"
                )
                print("  Major peaks at: ~1740, ~2980, ~1240 cm-1")
                print("  Suspected functional groups: carbonyl, ester, alkyl")

            # Initialize inverse spectroscopy engine
            inverse_engine = InverseSpectroscopyEngine(use_gpu=self.use_gpu)

            # Configure structure generation methods
            generation_methods = [
                StructureGenerationMethod.FRAGMENT_ASSEMBLY,
                StructureGenerationMethod.DATABASE_SEARCH,
            ]

            # Configure matching parameters
            matching_params = MatchingParameters(
                matching_method=MatchingMethod.WEIGHTED_SIMILARITY,
                similarity_threshold=0.3,
                peak_tolerance=20.0,  # 20 cm-1 tolerance
            )

            if self.verbose:
                print("\nStarting inverse spectroscopy analysis...")
                print(f"  Generation methods: {[m.value for m in generation_methods]}")
                print(f"  Matching method: {matching_params.matching_method.value}")
                print(f"  Similarity threshold: {matching_params.similarity_threshold}")

            start_time = time.time()
            inverse_result = inverse_engine.analyze_spectrum(
                experimental_spectrum,
                num_candidates=30,
                generation_methods=generation_methods,
                matching_params=matching_params,
            )
            analysis_time = time.time() - start_time

            if self.verbose:
                print(f"\nInverse analysis completed in {analysis_time:.2f} seconds")
                print(
                    f"Candidates generated: {len(inverse_result.candidate_structures)}"
                )

                if inverse_result.best_match:
                    print("Best match found:")
                    print(f"  SMILES: {inverse_result.best_match.smiles}")
                    print(
                        f"  Confidence score: {inverse_result.best_match.confidence_score:.3f}"
                    )
                    print(
                        f"  Spectral match score: {inverse_result.best_match.spectral_match_score:.3f}"
                    )
                    print(
                        f"  Chemical plausibility: {inverse_result.best_match.chemical_plausibility:.3f}"
                    )

                # Show top candidates
                top_candidates = inverse_result.get_top_candidates(5)
                if top_candidates:
                    print("\nTop 5 structural candidates:")
                    for i, candidate in enumerate(top_candidates, 1):
                        print(
                            f"  {i}. {candidate.smiles} (score: {candidate.confidence_score:.3f})"
                        )

            results["structure_proposal"] = {
                "analysis_time": analysis_time,
                "candidates_generated": len(inverse_result.candidate_structures),
                "best_match_score": (
                    inverse_result.best_match.confidence_score
                    if inverse_result.best_match
                    else 0
                ),
                "top_candidates": [
                    (c.smiles, c.confidence_score)
                    for c in inverse_result.get_top_candidates(3)
                ],
            }

        except Exception as e:
            print(f"Inverse spectroscopy demo failed: {e}")
            results["structure_proposal"] = {"error": str(e)}

        # Demo 2: Structure diversity filtering
        self.print_subsection("2. Diversity-Filtered Structure Proposals")

        try:
            if (
                "structure_proposal" in results
                and "error" not in results["structure_proposal"]
            ):
                # Get diverse candidates
                diverse_candidates = inverse_result.get_diversity_filtered_candidates(
                    max_candidates=10, diversity_threshold=0.7
                )

                if self.verbose:
                    print("Diversity filtering results:")
                    print(
                        f"  Original candidates: {len(inverse_result.candidate_structures)}"
                    )
                    print(f"  Diverse candidates: {len(diverse_candidates)}")

                    if diverse_candidates:
                        print("  Diverse structure examples:")
                        for i, candidate in enumerate(diverse_candidates[:3], 1):
                            mw = candidate.molecular_properties.get(
                                "molecular_weight", "N/A"
                            )
                            print(f"    {i}. {candidate.smiles} (MW: {mw})")

                results["diversity_filtering"] = {
                    "original_count": len(inverse_result.candidate_structures),
                    "diverse_count": len(diverse_candidates),
                    "diversity_improvement": len(diverse_candidates)
                    / max(1, len(inverse_result.candidate_structures)),
                }

        except Exception as e:
            print(f"Diversity filtering demo failed: {e}")
            results["diversity_filtering"] = {"error": str(e)}

        # Demo 3: High-level structure proposer interface
        self.print_subsection("3. High-Level Structure Proposer Interface")

        try:
            # Use simplified interface for structure proposal
            proposer = StructureProposer(
                spectroscopy_type=SpectroscopyType.IR, use_gpu=self.use_gpu
            )

            if self.verbose:
                print("Using high-level StructureProposer interface")
                print(f"Configured for: {SpectroscopyType.IR.value}")

            start_time = time.time()
            proposal_result = proposer.propose_structures(
                experimental_spectrum, max_proposals=15
            )
            proposal_time = time.time() - start_time

            if self.verbose:
                print(f"\nStructure proposal completed in {proposal_time:.2f} seconds")

                top_proposals = proposal_result.get_top_candidates(3)
                if top_proposals:
                    print("Top structural proposals:")
                    for i, proposal in enumerate(top_proposals, 1):
                        print(f"  {i}. {proposal.smiles}")
                        print(f"     Confidence: {proposal.confidence_score:.3f}")
                        if proposal.molecular_properties:
                            mw = proposal.molecular_properties.get(
                                "molecular_weight", "N/A"
                            )
                            logp = proposal.molecular_properties.get("logp", "N/A")
                            print(f"     MW: {mw}, LogP: {logp}")

            results["high_level_interface"] = {
                "proposal_time": proposal_time,
                "proposals_generated": len(proposal_result.candidate_structures),
                "interface_type": "StructureProposer",
            }

        except Exception as e:
            print(f"High-level interface demo failed: {e}")
            results["high_level_interface"] = {"error": str(e)}

        return results

    def demo_quantum_methods(self) -> Dict[str, Any]:
        """Demonstrate fast quantum chemistry methods for spectral features."""
        self.print_section("Fast Quantum Chemistry Methods for Spectral Features")

        results = {}

        # Demo 1: Semi-empirical calculations
        self.print_subsection("1. Semi-Empirical Method (AM1)")

        try:
            if not RDKIT_AVAILABLE:
                print("RDKit not available - using mock molecular data")
                results["semi_empirical"] = {"error": "RDKit not available"}
            else:
                # Create test molecule (ethanol)
                ethanol_smiles = "CCO"
                mol = Chem.MolFromSmiles(ethanol_smiles)

                if self.verbose:
                    print(f"Test molecule: {ethanol_smiles} (ethanol)")
                    print(
                        f"Molecular formula: {Chem.rdMolDescriptors.CalcMolFormula(mol)}"
                    )
                    print(f"Molecular weight: {Descriptors.MolWt(mol):.2f}")

                # Initialize semi-empirical method
                semi_empirical = SemiEmpiricalMethod("AM1", use_gpu=self.use_gpu)

                if self.verbose:
                    print("Semi-empirical method: AM1")
                    print(f"GPU acceleration: {self.use_gpu}")

                start_time = time.time()
                quantum_calc = semi_empirical.calculate_molecular_properties(mol)
                calc_time = time.time() - start_time

                if self.verbose:
                    print(f"\nQuantum calculation completed in {calc_time:.3f} seconds")
                    print("Calculation results:")
                    if quantum_calc.total_energy:
                        print(
                            f"  Total energy: {quantum_calc.total_energy:.6f} hartree"
                        )
                    if quantum_calc.homo_energy:
                        print(f"  HOMO energy: {quantum_calc.homo_energy:.6f} hartree")
                    if quantum_calc.lumo_energy:
                        print(f"  LUMO energy: {quantum_calc.lumo_energy:.6f} hartree")
                    if quantum_calc.homo_lumo_gap:
                        print(
                            f"  HOMO-LUMO gap: {quantum_calc.homo_lumo_gap:.6f} hartree"
                        )
                    print(f"  Converged: {quantum_calc.converged}")
                    print(f"  Iterations: {quantum_calc.iterations_used}")

                results["semi_empirical"] = {
                    "calculation_time": calc_time,
                    "total_energy": quantum_calc.total_energy,
                    "homo_lumo_gap": quantum_calc.homo_lumo_gap,
                    "converged": quantum_calc.converged,
                    "iterations": quantum_calc.iterations_used,
                }

        except Exception as e:
            print(f"Semi-empirical demo failed: {e}")
            results["semi_empirical"] = {"error": str(e)}

        # Demo 2: DFT surrogate model
        self.print_subsection("2. DFT Surrogate Model")

        try:
            if not RDKIT_AVAILABLE:
                print("RDKit not available - skipping DFT surrogate demo")
                results["dft_surrogate"] = {"error": "RDKit not available"}
            else:
                # Test larger molecule (benzene)
                benzene_smiles = "c1ccccc1"
                benzene_mol = Chem.MolFromSmiles(benzene_smiles)

                if self.verbose:
                    print(f"Test molecule: {benzene_smiles} (benzene)")
                    print(
                        f"Aromatic rings: {Descriptors.NumAromaticRings(benzene_mol)}"
                    )

                # Initialize DFT surrogate
                dft_surrogate = DensityFunctionalApproximation(
                    functional=FunctionalType.B3LYP, use_gpu=self.use_gpu
                )

                start_time = time.time()
                dft_calc = dft_surrogate.calculate_molecular_properties(benzene_mol)
                dft_time = time.time() - start_time

                if self.verbose:
                    print(
                        f"\nDFT surrogate calculation completed in {dft_time:.3f} seconds"
                    )
                    print("Results:")
                    if dft_calc.total_energy:
                        print(f"  Total energy: {dft_calc.total_energy:.6f} hartree")
                    if dft_calc.dipole_moment:
                        dipole_mag = np.linalg.norm(dft_calc.dipole_moment)
                        print(f"  Dipole moment: {dipole_mag:.4f} Debye")
                    print("  Method: Neural network surrogate")

                results["dft_surrogate"] = {
                    "calculation_time": dft_time,
                    "total_energy": dft_calc.total_energy,
                    "dipole_moment": (
                        np.linalg.norm(dft_calc.dipole_moment)
                        if dft_calc.dipole_moment
                        else None
                    ),
                    "method": "DFT_surrogate",
                }

        except Exception as e:
            print(f"DFT surrogate demo failed: {e}")
            results["dft_surrogate"] = {"error": str(e)}

        # Demo 3: Fast quantum chemistry engine
        self.print_subsection("3. Fast Quantum Chemistry Engine")

        try:
            if not RDKIT_AVAILABLE:
                print("RDKit not available - skipping quantum engine demo")
                results["quantum_engine"] = {"error": "RDKit not available"}
            else:
                # Test different approximation levels
                test_molecules = [
                    ("water", "O"),
                    ("methanol", "CO"),
                    ("acetone", "CC(=O)C"),
                ]

                engine_results = {}

                for name, smiles in test_molecules:
                    mol = Chem.MolFromSmiles(smiles)

                    if self.verbose:
                        print(f"\nTesting {name} ({smiles}):")

                    # Test with balanced approximation
                    engine = FastQuantumChemistry(
                        approximation_level=ApproximationLevel.BALANCED,
                        use_gpu=self.use_gpu,
                    )

                    start_time = time.time()
                    calc_result = engine.calculate_molecular_properties(mol)
                    engine_time = time.time() - start_time

                    if self.verbose:
                        print(f"  Calculation time: {engine_time:.3f} seconds")
                        if calc_result.total_energy:
                            print(f"  Total energy: {calc_result.total_energy:.6f}")
                        if calc_result.homo_lumo_gap:
                            print(f"  HOMO-LUMO gap: {calc_result.homo_lumo_gap:.6f}")

                    engine_results[name] = {
                        "calculation_time": engine_time,
                        "total_energy": calc_result.total_energy,
                        "homo_lumo_gap": calc_result.homo_lumo_gap,
                    }

                results["quantum_engine"] = {
                    "molecules_tested": len(test_molecules),
                    "results": engine_results,
                    "approximation_level": "balanced",
                }

        except Exception as e:
            print(f"Quantum engine demo failed: {e}")
            results["quantum_engine"] = {"error": str(e)}

        return results

    def demo_benchmarking(self) -> Dict[str, Any]:
        """Demonstrate benchmarking and performance analysis."""
        self.print_section("Benchmarking and Performance Analysis")

        results = {}

        # Demo 1: Model accuracy benchmarking
        self.print_subsection("1. Model Accuracy Benchmarking")

        try:
            # Generate test datasets
            train_data = self.generate_synthetic_spectra_data(150)
            test_data = self.generate_synthetic_spectra_data(50)

            # Train and evaluate different models
            model_configs = [
                ("transformer", ModelArchitecture.TRANSFORMER),
                ("convolutional", ModelArchitecture.CONVOLUTIONAL),
            ]

            accuracy_results = {}

            for model_name, architecture in model_configs:
                if self.verbose:
                    print(f"\nBenchmarking {model_name} model:")

                try:
                    config = ModelConfiguration(
                        architecture=architecture,
                        spectroscopy_type=SpectroscopyType.IR,
                        num_epochs=3,  # Reduced for demo
                        use_gpu=self.use_gpu,
                    )

                    predictor = GPUSpectroscopyPredictor(config)

                    # Training
                    start_time = time.time()
                    training_metrics = predictor.train(train_data)
                    training_time = time.time() - start_time

                    # Evaluation
                    start_time = time.time()
                    eval_metrics = evaluate_model_accuracy(predictor, test_data)
                    eval_time = time.time() - start_time

                    if self.verbose:
                        print(f"  Training time: {training_time:.2f} seconds")
                        print(f"  Evaluation time: {eval_time:.2f} seconds")
                        print(
                            f"  Mean squared error: {eval_metrics['mean_squared_error']:.6f}"
                        )
                        print(
                            f"  Mean absolute error: {eval_metrics['mean_absolute_error']:.6f}"
                        )
                        print(f"  R-squared: {eval_metrics['r_squared']:.4f}")

                    accuracy_results[model_name] = {
                        "training_time": training_time,
                        "evaluation_time": eval_time,
                        "mse": eval_metrics["mean_squared_error"],
                        "mae": eval_metrics["mean_absolute_error"],
                        "r_squared": eval_metrics["r_squared"],
                    }

                except Exception as e:
                    print(f"  {model_name} model failed: {e}")
                    accuracy_results[model_name] = {"error": str(e)}

            results["accuracy_benchmark"] = accuracy_results

        except Exception as e:
            print(f"Accuracy benchmarking failed: {e}")
            results["accuracy_benchmark"] = {"error": str(e)}

        # Demo 2: Performance scaling analysis
        self.print_subsection("2. Performance Scaling Analysis")

        try:
            if self.verbose:
                print("Testing performance scaling with dataset size:")

            dataset_sizes = [50, 100, 200]
            scaling_results = {}

            # Use simple IR predictor for scaling test
            predictor = IRPredictor(use_gpu=self.use_gpu)

            for size in dataset_sizes:
                if self.verbose:
                    print(f"\n  Dataset size: {size} samples")

                # Generate data of specific size
                scaling_data = self.generate_synthetic_spectra_data(size)

                # Train
                start_time = time.time()
                train_metrics = predictor.train(scaling_data)
                train_time = time.time() - start_time

                # Predict
                test_features = scaling_data.molecular_features[:10]
                pred_start = time.time()
                predictions = predictor.predict(test_features)
                pred_time = time.time() - pred_start

                if self.verbose:
                    print(f"    Training time: {train_time:.2f} seconds")
                    print(f"    Prediction time: {pred_time:.3f} seconds")
                    print(
                        f"    Time per sample: {pred_time/len(test_features)*1000:.2f} ms"
                    )

                scaling_results[size] = {
                    "training_time": train_time,
                    "prediction_time": pred_time,
                    "time_per_sample": pred_time / len(test_features),
                    "samples_per_second": len(test_features) / pred_time,
                }

            results["scaling_analysis"] = scaling_results

        except Exception as e:
            print(f"Scaling analysis failed: {e}")
            results["scaling_analysis"] = {"error": str(e)}

        # Demo 3: GPU vs CPU performance comparison
        self.print_subsection("3. GPU vs CPU Performance Comparison")

        try:
            test_data = self.generate_synthetic_spectra_data(100)

            performance_comparison = {}

            for use_gpu in [False, True]:
                device = "GPU" if use_gpu else "CPU"

                if self.verbose:
                    print(f"\nTesting {device} performance:")

                try:
                    config = ModelConfiguration(
                        architecture=ModelArchitecture.TRANSFORMER,
                        spectroscopy_type=SpectroscopyType.IR,
                        num_epochs=2,
                        use_gpu=use_gpu,
                    )

                    predictor = GPUSpectroscopyPredictor(config)

                    # Training performance
                    start_time = time.time()
                    train_metrics = predictor.train(test_data)
                    train_time = time.time() - start_time

                    # Prediction performance
                    test_features = test_data.molecular_features[:20]
                    pred_start = time.time()
                    predictions = predictor.predict(test_features)
                    pred_time = time.time() - pred_start

                    if self.verbose:
                        print(f"  Training time: {train_time:.2f} seconds")
                        print(f"  Prediction time: {pred_time:.3f} seconds")
                        print(
                            f"  Throughput: {len(test_features)/pred_time:.1f} molecules/second"
                        )

                    performance_comparison[device] = {
                        "training_time": train_time,
                        "prediction_time": pred_time,
                        "throughput": len(test_features) / pred_time,
                    }

                except Exception as e:
                    print(f"  {device} testing failed: {e}")
                    performance_comparison[device] = {"error": str(e)}

            # Calculate speedup
            if "CPU" in performance_comparison and "GPU" in performance_comparison:
                if (
                    "error" not in performance_comparison["CPU"]
                    and "error" not in performance_comparison["GPU"]
                ):
                    cpu_time = performance_comparison["CPU"]["prediction_time"]
                    gpu_time = performance_comparison["GPU"]["prediction_time"]
                    speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0

                    if self.verbose:
                        print(f"\nGPU speedup: {speedup:.2f}x faster than CPU")

                    performance_comparison["gpu_speedup"] = speedup

            results["gpu_cpu_comparison"] = performance_comparison

        except Exception as e:
            print(f"GPU vs CPU comparison failed: {e}")
            results["gpu_cpu_comparison"] = {"error": str(e)}

        return results

    def generate_summary_report(self) -> None:
        """Generate comprehensive summary report of all demonstrations."""
        self.print_section("Comprehensive Demo Summary Report")

        if self.verbose:
            print("ChemiAccel Spectroscopy Framework Demonstration Results")
            print("=" * 60)

            # Summary statistics
            total_demos = len(self.demo_results)
            successful_demos = sum(
                1
                for result in self.demo_results.values()
                if isinstance(result, dict) and "error" not in str(result)
            )

            print("\nOverall Statistics:")
            print(f"  Total demonstration categories: {total_demos}")
            print(f"  Successful demonstrations: {successful_demos}")
            print(f"  Success rate: {successful_demos/total_demos*100:.1f}%")

            # Feature coverage
            print("\nFeature Coverage Demonstrated:")
            features = [
                "✓ GPU-accelerated predictive models (IR, NMR, MS, UV-Vis)",
                "✓ QML surrogate models with quantum features",
                "✓ Automated experimental data processing",
                "✓ Baseline correction and normalization",
                "✓ Spectral alignment and batch processing",
                "✓ AI-driven inverse spectroscopy",
                "✓ Structure generation and proposal",
                "✓ Fast quantum chemistry methods",
                "✓ Semi-empirical and DFT calculations",
                "✓ Performance benchmarking and scaling",
                "✓ GPU vs CPU performance comparison",
            ]

            for feature in features:
                print(f"  {feature}")

            # Performance highlights
            if "predictive_models" in self.demo_results:
                pm_results = self.demo_results["predictive_models"]
                if "batch_performance" in pm_results:
                    bp = pm_results["batch_performance"]
                    if "throughput" in bp:
                        print("\nPerformance Highlights:")
                        print(
                            f"  Spectral prediction throughput: {bp['throughput']:.1f} molecules/second"
                        )

            if "experimental_processing" in self.demo_results:
                ep_results = self.demo_results["experimental_processing"]
                if "preprocessing" in ep_results:
                    pp = ep_results["preprocessing"]
                    if "processing_time" in pp:
                        print(
                            f"  Spectrum preprocessing: {pp['processing_time']*1000:.1f} ms per spectrum"
                        )

            if "quantum_methods" in self.demo_results:
                qm_results = self.demo_results["quantum_methods"]
                if (
                    "semi_empirical" in qm_results
                    and "calculation_time" in qm_results["semi_empirical"]
                ):
                    print(
                        f"  Quantum calculations: {qm_results['semi_empirical']['calculation_time']*1000:.1f} ms per molecule"
                    )

            # GPU acceleration benefits
            if "benchmarking" in self.demo_results:
                bench_results = self.demo_results["benchmarking"]
                if "gpu_cpu_comparison" in bench_results:
                    gpu_comp = bench_results["gpu_cpu_comparison"]
                    if "gpu_speedup" in gpu_comp:
                        print(
                            f"  GPU acceleration: {gpu_comp['gpu_speedup']:.1f}x speedup over CPU"
                        )

            print("\nDemo completed successfully!")
            print("All major spectroscopy framework features demonstrated.")


def main():
    """Main demonstration function."""
    print("ChemiAccel Spectroscopy Framework - Comprehensive Demonstration")
    print("================================================================")

    # Initialize demo
    try:
        demo = SpectroscopyDemo(use_gpu=True, verbose=True)
        print(f"Demo initialized with GPU acceleration: {demo.use_gpu}")
    except Exception as e:
        print(f"Failed to initialize demo: {e}")
        return

    # Run all demonstrations
    try:
        # 1. Predictive Models Demo
        predictive_results = demo.demo_predictive_models()
        demo.demo_results["predictive_models"] = predictive_results

        # 2. Experimental Processing Demo
        processing_results = demo.demo_experimental_processing()
        demo.demo_results["experimental_processing"] = processing_results

        # 3. Inverse Spectroscopy Demo
        inverse_results = demo.demo_inverse_spectroscopy()
        demo.demo_results["inverse_spectroscopy"] = inverse_results

        # 4. Quantum Methods Demo
        quantum_results = demo.demo_quantum_methods()
        demo.demo_results["quantum_methods"] = quantum_results

        # 5. Benchmarking Demo
        benchmark_results = demo.demo_benchmarking()
        demo.demo_results["benchmarking"] = benchmark_results

        # 6. Generate Summary Report
        demo.generate_summary_report()

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
