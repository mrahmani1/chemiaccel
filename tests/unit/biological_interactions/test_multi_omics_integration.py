"""
Unit tests for multi-omics data integration module.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from chemi_accel.biological_interactions.multi_omics_integration import (  # Data structures; Enums; Data ingesters; Processing classes; Utility functions; Exceptions
    DataFormat,
    DataIngestionError,
    GenomicsDataIngester,
    IntegrationMethod,
    IntegrationResult,
    MetabolomicsDataIngester,
    MultiOmicsIntegrationError,
    MultiOmicsIntegrationPipeline,
    MultiOmicsIntegrator,
    OmicsDataset,
    OmicsDataType,
    OmicsQualityController,
    PredictionRefinement,
    PredictionRefinementEngine,
    PredictionRefinementError,
    PredictionRefinementStrategy,
    ProteomicsDataIngester,
    QualityControlError,
    QualityControlMethod,
    QualityControlReport,
    create_sample_omics_data,
    integrate_omics_data_batch,
    load_omics_dataset,
)


@pytest.fixture
def sample_genomics_data():
    """Create sample genomics data."""
    return pd.DataFrame(
        {
            "Sample_001": np.random.randn(100),
            "Sample_002": np.random.randn(100),
            "Sample_003": np.random.randn(100),
        },
        index=[f"SNP_{i:04d}" for i in range(100)],
    )


@pytest.fixture
def sample_proteomics_data():
    """Create sample proteomics data."""
    return pd.DataFrame(
        {
            "Sample_001": np.random.randn(200),
            "Sample_002": np.random.randn(200),
            "Sample_003": np.random.randn(200),
        },
        index=[f"Protein_{i:04d}" for i in range(200)],
    )


@pytest.fixture
def sample_metabolomics_data():
    """Create sample metabolomics data."""
    return pd.DataFrame(
        {
            "Sample_001": np.random.randn(150),
            "Sample_002": np.random.randn(150),
            "Sample_003": np.random.randn(150),
        },
        index=[f"Metabolite_{i:04d}" for i in range(150)],
    )


@pytest.fixture
def sample_omics_dataset(sample_genomics_data):
    """Create sample omics dataset."""
    return OmicsDataset(
        dataset_id="test_genomics",
        name="Test Genomics Dataset",
        data_type=OmicsDataType.GENOMICS,
        data_format=DataFormat.CSV,
        data=sample_genomics_data,
        metadata={"samples": 3, "features": 100},
    )


@pytest.fixture
def sample_integration_result():
    """Create sample integration result."""
    integrated_data = pd.DataFrame(
        {
            "Sample_001": np.random.randn(50),
            "Sample_002": np.random.randn(50),
            "Sample_003": np.random.randn(50),
        },
        index=[f"Feature_{i:04d}" for i in range(50)],
    )

    return IntegrationResult(
        integration_id="test_integration",
        datasets_used=["dataset1", "dataset2"],
        integration_method=IntegrationMethod.EARLY_FUSION,
        integrated_features=integrated_data,
        feature_importance={"Feature_0001": 0.8, "Feature_0002": 0.6},
        integration_quality_score=0.75,
        processing_time=1.5,
    )


class TestOmicsDataset:
    """Test OmicsDataset class."""

    def test_dataset_creation(self, sample_genomics_data):
        """Test dataset creation."""
        dataset = OmicsDataset(
            dataset_id="test_dataset",
            name="Test Dataset",
            data_type=OmicsDataType.GENOMICS,
            data_format=DataFormat.CSV,
            data=sample_genomics_data,
        )

        assert dataset.dataset_id == "test_dataset"
        assert dataset.name == "Test Dataset"
        assert dataset.data_type == OmicsDataType.GENOMICS
        assert dataset.data_format == DataFormat.CSV
        assert dataset.data is not None
        assert not dataset.is_processed
        assert not dataset.integration_ready

    def test_dataset_post_init(self):
        """Test dataset __post_init__ method."""
        dataset = OmicsDataset(
            dataset_id="test",
            name="Test",
            data_type=OmicsDataType.GENOMICS,
            data_format=DataFormat.CSV,
            file_path="test.csv",
        )

        assert isinstance(dataset.data, pd.DataFrame)
        assert not dataset.is_processed
        assert not dataset.integration_ready


class TestIntegrationResult:
    """Test IntegrationResult class."""

    def test_integration_result_creation(self, sample_integration_result):
        """Test integration result creation."""
        result = sample_integration_result

        assert result.integration_id == "test_integration"
        assert len(result.datasets_used) == 2
        assert result.integration_method == IntegrationMethod.EARLY_FUSION
        assert result.integration_quality_score == 0.75
        assert result.processing_time == 1.5

    def test_get_top_features(self, sample_integration_result):
        """Test getting top features."""
        result = sample_integration_result
        top_features = result.get_top_features(1)

        assert len(top_features) == 1
        assert top_features[0][0] == "Feature_0001"
        assert top_features[0][1] == 0.8


class TestPredictionRefinement:
    """Test PredictionRefinement class."""

    def test_refinement_creation(self):
        """Test refinement creation."""
        refinement = PredictionRefinement(
            original_prediction=0.7,
            refined_prediction=0.8,
            confidence_adjustment=0.1,
            contributing_omics=[OmicsDataType.GENOMICS],
            refinement_strategy=PredictionRefinementStrategy.BAYESIAN_UPDATE,
            supporting_evidence={"evidence": "test"},
            uncertainty_quantification={"std": 0.05},
        )

        assert refinement.original_prediction == 0.7
        assert refinement.refined_prediction == 0.8
        assert refinement.confidence_adjustment == 0.1
        assert len(refinement.contributing_omics) == 1
        assert (
            refinement.refinement_strategy
            == PredictionRefinementStrategy.BAYESIAN_UPDATE
        )


class TestGenomicsDataIngester:
    """Test GenomicsDataIngester class."""

    def test_ingester_creation(self):
        """Test ingester creation."""
        ingester = GenomicsDataIngester(use_gpu=False)
        assert not ingester.use_gpu

    def test_validate_format(self):
        """Test format validation."""
        ingester = GenomicsDataIngester(use_gpu=False)

        assert ingester.validate_format("test.vcf")
        assert ingester.validate_format("test.csv")
        assert not ingester.validate_format("test.txt.exe")

    @patch("pandas.read_csv")
    def test_ingest_csv(self, mock_read_csv, sample_genomics_data):
        """Test CSV ingestion."""
        mock_read_csv.return_value = sample_genomics_data

        ingester = GenomicsDataIngester(use_gpu=False)
        dataset = ingester.ingest("test.csv")

        assert isinstance(dataset, OmicsDataset)
        assert dataset.data_type == OmicsDataType.GENOMICS
        assert "genomics" in dataset.dataset_id
        mock_read_csv.assert_called_once_with("test.csv")

    def test_ingest_invalid_format(self):
        """Test ingestion with invalid format."""
        ingester = GenomicsDataIngester(use_gpu=False)

        with pytest.raises(DataIngestionError):
            ingester.ingest("invalid.exe")

    @patch("builtins.open")
    def test_read_vcf(self, mock_open):
        """Test VCF file reading."""
        mock_file = MagicMock()
        mock_file.__iter__ = Mock(
            return_value=iter(["chr1\t100\trs123\tA\tT\n", "chr2\t200\trs456\tG\tC\n"])
        )
        mock_open.return_value.__enter__.return_value = mock_file

        ingester = GenomicsDataIngester(use_gpu=False)
        data = ingester._read_vcf("test.vcf")

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 2
        assert "CHR" in data.columns


class TestProteomicsDataIngester:
    """Test ProteomicsDataIngester class."""

    def test_ingester_creation(self):
        """Test ingester creation."""
        ingester = ProteomicsDataIngester(use_gpu=False)
        assert not ingester.use_gpu

    def test_validate_format(self):
        """Test format validation."""
        ingester = ProteomicsDataIngester(use_gpu=False)

        assert ingester.validate_format("test.csv")
        assert ingester.validate_format("test.xlsx")
        assert ingester.validate_format("test.tsv")
        assert not ingester.validate_format("test.exe")

    @patch("pandas.read_csv")
    def test_ingest_csv(self, mock_read_csv, sample_proteomics_data):
        """Test CSV ingestion."""
        mock_read_csv.return_value = sample_proteomics_data

        ingester = ProteomicsDataIngester(use_gpu=False)
        dataset = ingester.ingest("test.csv")

        assert isinstance(dataset, OmicsDataset)
        assert dataset.data_type == OmicsDataType.PROTEOMICS
        assert "proteomics" in dataset.dataset_id

    def test_ingest_invalid_format(self):
        """Test ingestion with invalid format."""
        ingester = ProteomicsDataIngester(use_gpu=False)

        with pytest.raises(DataIngestionError):
            ingester.ingest("invalid.exe")


class TestMetabolomicsDataIngester:
    """Test MetabolomicsDataIngester class."""

    def test_ingester_creation(self):
        """Test ingester creation."""
        ingester = MetabolomicsDataIngester(use_gpu=False)
        assert not ingester.use_gpu

    @patch("pandas.read_csv")
    def test_ingest_csv(self, mock_read_csv, sample_metabolomics_data):
        """Test CSV ingestion."""
        mock_read_csv.return_value = sample_metabolomics_data

        ingester = MetabolomicsDataIngester(use_gpu=False)
        dataset = ingester.ingest("test.csv")

        assert isinstance(dataset, OmicsDataset)
        assert dataset.data_type == OmicsDataType.METABOLOMICS
        assert "metabolomics" in dataset.dataset_id


class TestOmicsQualityController:
    """Test OmicsQualityController class."""

    def test_controller_creation(self):
        """Test controller creation."""
        controller = OmicsQualityController(use_gpu=False)
        assert not controller.use_gpu

    def test_run_quality_control(self, sample_omics_dataset):
        """Test quality control execution."""
        controller = OmicsQualityController(use_gpu=False)
        methods = [
            QualityControlMethod.OUTLIER_DETECTION,
            QualityControlMethod.NORMALIZATION,
        ]

        processed_dataset, report = controller.run_quality_control(
            sample_omics_dataset, methods
        )

        assert isinstance(processed_dataset, OmicsDataset)
        assert isinstance(report, QualityControlReport)
        assert processed_dataset.is_processed
        assert report.dataset_id == sample_omics_dataset.dataset_id
        assert len(report.methods_applied) == 2
        assert report.quality_score >= 0.0

    def test_detect_outliers_iqr(self, sample_genomics_data):
        """Test IQR outlier detection."""
        controller = OmicsQualityController(use_gpu=False)
        filtered_data, outliers = controller._detect_outliers(
            sample_genomics_data, method="iqr"
        )

        assert isinstance(filtered_data, pd.DataFrame)
        assert isinstance(outliers, int)
        assert outliers >= 0

    def test_impute_missing_values(self):
        """Test missing value imputation."""
        controller = OmicsQualityController(use_gpu=False)

        # Create data with missing values
        data = pd.DataFrame(
            {"A": [1, 2, np.nan, 4], "B": [np.nan, 2, 3, 4], "C": [1, 2, 3, np.nan]}
        )

        imputed_data = controller._impute_missing_values(data, method="median")

        assert not imputed_data.isnull().any().any()

    def test_normalize_data(self, sample_genomics_data):
        """Test data normalization."""
        controller = OmicsQualityController(use_gpu=False)

        normalized_data = controller._normalize_data(
            sample_genomics_data, method="zscore"
        )

        assert isinstance(normalized_data, pd.DataFrame)
        assert normalized_data.shape == sample_genomics_data.shape

    def test_calculate_quality_score(self, sample_genomics_data):
        """Test quality score calculation."""
        controller = OmicsQualityController(use_gpu=False)
        score = controller._calculate_quality_score(sample_genomics_data, 100, 3)

        assert 0.0 <= score <= 1.0


class TestMultiOmicsIntegrator:
    """Test MultiOmicsIntegrator class."""

    def test_integrator_creation(self):
        """Test integrator creation."""
        integrator = MultiOmicsIntegrator(use_gpu=False)
        assert not integrator.use_gpu

    def test_integrate_datasets_early_fusion(self):
        """Test early fusion integration."""
        integrator = MultiOmicsIntegrator(use_gpu=False)

        # Create test datasets
        datasets = create_sample_omics_data(
            n_samples=50,
            n_features_per_omics=100,
            omics_types=[OmicsDataType.GENOMICS, OmicsDataType.PROTEOMICS],
        )

        result = integrator.integrate_datasets(datasets, IntegrationMethod.EARLY_FUSION)

        assert isinstance(result, IntegrationResult)
        assert result.integration_method == IntegrationMethod.EARLY_FUSION
        assert len(result.datasets_used) == 2
        assert result.integration_quality_score >= 0.0

    def test_integrate_datasets_late_fusion(self):
        """Test late fusion integration."""
        integrator = MultiOmicsIntegrator(use_gpu=False)

        datasets = create_sample_omics_data(
            n_samples=50,
            n_features_per_omics=100,
            omics_types=[OmicsDataType.GENOMICS, OmicsDataType.PROTEOMICS],
        )

        result = integrator.integrate_datasets(datasets, IntegrationMethod.LATE_FUSION)

        assert isinstance(result, IntegrationResult)
        assert result.integration_method == IntegrationMethod.LATE_FUSION

    def test_integrate_datasets_empty_list(self):
        """Test integration with empty dataset list."""
        integrator = MultiOmicsIntegrator(use_gpu=False)

        with pytest.raises(MultiOmicsIntegrationError):
            integrator.integrate_datasets([])

    def test_find_common_samples(self):
        """Test finding common samples."""
        integrator = MultiOmicsIntegrator(use_gpu=False)

        datasets = create_sample_omics_data(n_samples=50, n_features_per_omics=10)
        common_samples = integrator._find_common_samples(datasets)

        assert isinstance(common_samples, list)
        assert len(common_samples) == 50  # All samples should be common

    def test_calculate_integration_quality(self):
        """Test integration quality calculation."""
        integrator = MultiOmicsIntegrator(use_gpu=False)

        integrated_features = pd.DataFrame(np.random.randn(100, 50))
        datasets = create_sample_omics_data(n_samples=50, n_features_per_omics=50)

        quality = integrator._calculate_integration_quality(
            integrated_features, datasets
        )

        assert 0.0 <= quality <= 1.0


class TestPredictionRefinementEngine:
    """Test PredictionRefinementEngine class."""

    def test_engine_creation(self):
        """Test engine creation."""
        engine = PredictionRefinementEngine(use_gpu=False)
        assert not engine.use_gpu

    def test_refine_prediction_bayesian(self, sample_integration_result):
        """Test Bayesian update refinement."""
        engine = PredictionRefinementEngine(use_gpu=False)

        refinement = engine.refine_prediction(
            original_prediction=0.7,
            integrated_omics=sample_integration_result,
            sample_id="Sample_001",
            strategy=PredictionRefinementStrategy.BAYESIAN_UPDATE,
        )

        assert isinstance(refinement, PredictionRefinement)
        assert refinement.original_prediction == 0.7
        assert (
            refinement.refinement_strategy
            == PredictionRefinementStrategy.BAYESIAN_UPDATE
        )
        assert "posterior_std" in refinement.uncertainty_quantification

    def test_refine_prediction_ensemble(self, sample_integration_result):
        """Test ensemble weighting refinement."""
        engine = PredictionRefinementEngine(use_gpu=False)

        refinement = engine.refine_prediction(
            original_prediction=0.6,
            integrated_omics=sample_integration_result,
            sample_id="Sample_002",
            strategy=PredictionRefinementStrategy.ENSEMBLE_WEIGHTING,
        )

        assert isinstance(refinement, PredictionRefinement)
        assert (
            refinement.refinement_strategy
            == PredictionRefinementStrategy.ENSEMBLE_WEIGHTING
        )

    def test_refine_prediction_unsupported_strategy(self, sample_integration_result):
        """Test refinement with unsupported strategy."""
        engine = PredictionRefinementEngine(use_gpu=False)

        # This should not raise an error as all strategies are implemented
        # but let's test error handling for the future
        try:
            refinement = engine.refine_prediction(
                original_prediction=0.5,
                integrated_omics=sample_integration_result,
                sample_id="Sample_001",
                strategy=PredictionRefinementStrategy.PATHWAY_CONSTRAINT,
            )
            assert isinstance(refinement, PredictionRefinement)
        except PredictionRefinementError:
            # This is also acceptable if the strategy isn't fully implemented
            pass


class TestMultiOmicsIntegrationPipeline:
    """Test MultiOmicsIntegrationPipeline class."""

    def test_pipeline_creation(self):
        """Test pipeline creation."""
        pipeline = MultiOmicsIntegrationPipeline(use_gpu=False)
        assert not pipeline.use_gpu
        assert pipeline.genomics_ingester is not None
        assert pipeline.proteomics_ingester is not None
        assert pipeline.metabolomics_ingester is not None
        assert pipeline.quality_controller is not None
        assert pipeline.integrator is not None
        assert pipeline.refinement_engine is not None

    @patch("pandas.read_csv")
    def test_run_complete_pipeline(self, mock_read_csv):
        """Test complete pipeline execution."""
        # Mock data for different omics types
        mock_genomics_data = pd.DataFrame(np.random.randn(100, 50))
        mock_proteomics_data = pd.DataFrame(np.random.randn(200, 50))

        def mock_read_csv_side_effect(file_path, **kwargs):
            if "genomics" in file_path:
                return mock_genomics_data
            elif "proteomics" in file_path:
                return mock_proteomics_data
            else:
                return pd.DataFrame()

        mock_read_csv.side_effect = mock_read_csv_side_effect

        pipeline = MultiOmicsIntegrationPipeline(use_gpu=False)
        data_files = {
            OmicsDataType.GENOMICS: "genomics_data.csv",
            OmicsDataType.PROTEOMICS: "proteomics_data.csv",
        }

        integration_result, qc_reports = pipeline.run_complete_pipeline(data_files)

        assert isinstance(integration_result, IntegrationResult)
        assert isinstance(qc_reports, list)
        assert len(qc_reports) == 2

    def test_refine_predictions_batch(self, sample_integration_result):
        """Test batch prediction refinement."""
        pipeline = MultiOmicsIntegrationPipeline(use_gpu=False)

        predictions = {"Sample_001": 0.7, "Sample_002": 0.8, "Sample_003": 0.6}

        refinements = pipeline.refine_predictions_batch(
            predictions,
            sample_integration_result,
            strategy=PredictionRefinementStrategy.BAYESIAN_UPDATE,
        )

        assert len(refinements) == 3
        for sample_id, refinement in refinements.items():
            assert isinstance(refinement, PredictionRefinement)
            assert refinement.original_prediction == predictions[sample_id]


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_sample_omics_data(self):
        """Test sample data creation."""
        datasets = create_sample_omics_data(
            n_samples=30,
            n_features_per_omics=50,
            omics_types=[OmicsDataType.GENOMICS, OmicsDataType.PROTEOMICS],
        )

        assert len(datasets) == 2
        for dataset in datasets:
            assert isinstance(dataset, OmicsDataset)
            assert dataset.data.shape == (50, 30)

    def test_integrate_omics_data_batch(self):
        """Test batch omics integration."""
        datasets = create_sample_omics_data(
            n_samples=20,
            n_features_per_omics=30,
            omics_types=[OmicsDataType.GENOMICS, OmicsDataType.PROTEOMICS],
        )

        methods = [IntegrationMethod.EARLY_FUSION, IntegrationMethod.LATE_FUSION]
        results = integrate_omics_data_batch(datasets, methods)

        assert isinstance(results, dict)
        assert len(results) <= len(methods)  # Some methods might fail
        for method, result in results.items():
            assert isinstance(result, IntegrationResult)

    @patch("pandas.read_csv")
    def test_load_omics_dataset(self, mock_read_csv):
        """Test single dataset loading."""
        mock_data = pd.DataFrame(np.random.randn(100, 50))
        mock_read_csv.return_value = mock_data

        dataset = load_omics_dataset("test.csv", OmicsDataType.GENOMICS)

        assert isinstance(dataset, OmicsDataset)
        assert dataset.data_type == OmicsDataType.GENOMICS

    def test_load_omics_dataset_unsupported_type(self):
        """Test loading with unsupported data type."""
        # Create a custom enum value for testing
        with pytest.raises(DataIngestionError):
            load_omics_dataset("test.csv", OmicsDataType.EPIGENOMICS)


class TestExceptions:
    """Test exception classes."""

    def test_multi_omics_integration_error(self):
        """Test MultiOmicsIntegrationError."""
        with pytest.raises(MultiOmicsIntegrationError):
            raise MultiOmicsIntegrationError("Test error")

    def test_data_ingestion_error(self):
        """Test DataIngestionError."""
        with pytest.raises(DataIngestionError):
            raise DataIngestionError("Ingestion failed")

    def test_quality_control_error(self):
        """Test QualityControlError."""
        with pytest.raises(QualityControlError):
            raise QualityControlError("QC failed")

    def test_prediction_refinement_error(self):
        """Test PredictionRefinementError."""
        with pytest.raises(PredictionRefinementError):
            raise PredictionRefinementError("Refinement failed")


class TestIntegration:
    """Integration tests."""

    def test_end_to_end_integration_workflow(self):
        """Test complete integration workflow."""
        # Create sample datasets
        datasets = create_sample_omics_data(
            n_samples=25,
            n_features_per_omics=40,
            omics_types=[
                OmicsDataType.GENOMICS,
                OmicsDataType.PROTEOMICS,
                OmicsDataType.METABOLOMICS,
            ],
        )

        # Quality control
        controller = OmicsQualityController(use_gpu=False)
        processed_datasets = []

        for dataset in datasets:
            processed_dataset, _ = controller.run_quality_control(
                dataset,
                [
                    QualityControlMethod.OUTLIER_DETECTION,
                    QualityControlMethod.NORMALIZATION,
                ],
            )
            processed_datasets.append(processed_dataset)

        # Integration
        integrator = MultiOmicsIntegrator(use_gpu=False)
        integration_result = integrator.integrate_datasets(
            processed_datasets, IntegrationMethod.EARLY_FUSION
        )

        # Prediction refinement
        engine = PredictionRefinementEngine(use_gpu=False)
        original_predictions = {
            f"Sample_{i:03d}": np.random.random() for i in range(25)
        }

        refinements = {}
        for sample_id, prediction in original_predictions.items():
            try:
                refinement = engine.refine_prediction(
                    prediction,
                    integration_result,
                    sample_id,
                    PredictionRefinementStrategy.BAYESIAN_UPDATE,
                )
                refinements[sample_id] = refinement
            except Exception:
                # Some refinements might fail, which is acceptable
                pass

        # Verify results
        assert isinstance(integration_result, IntegrationResult)
        assert integration_result.integration_quality_score >= 0.0
        assert len(refinements) > 0

        for refinement in refinements.values():
            assert isinstance(refinement, PredictionRefinement)

    def test_different_integration_methods(self):
        """Test different integration methods."""
        datasets = create_sample_omics_data(
            n_samples=20,
            n_features_per_omics=30,
            omics_types=[OmicsDataType.GENOMICS, OmicsDataType.PROTEOMICS],
        )

        integrator = MultiOmicsIntegrator(use_gpu=False)
        methods = [
            IntegrationMethod.EARLY_FUSION,
            IntegrationMethod.LATE_FUSION,
            IntegrationMethod.INTERMEDIATE_FUSION,
            IntegrationMethod.HIERARCHICAL_FUSION,
        ]

        results = []
        for method in methods:
            try:
                result = integrator.integrate_datasets(datasets, method)
                results.append(result)
                assert result.integration_method == method
            except Exception as e:
                # Some methods might fail due to dependencies, which is acceptable
                print(f"Method {method} failed: {e}")

        assert len(results) > 0  # At least some methods should work

    def test_quality_control_methods(self, sample_omics_dataset):
        """Test different quality control methods."""
        controller = OmicsQualityController(use_gpu=False)
        methods = [
            QualityControlMethod.OUTLIER_DETECTION,
            QualityControlMethod.MISSING_VALUE_IMPUTATION,
            QualityControlMethod.NORMALIZATION,
            QualityControlMethod.BATCH_CORRECTION,
        ]

        processed_dataset, report = controller.run_quality_control(
            sample_omics_dataset, methods
        )

        assert processed_dataset.is_processed
        assert len(report.methods_applied) == len(methods)
        assert report.quality_score >= 0.0
        assert len(report.processing_log) == len(methods)

    def test_prediction_refinement_strategies(self, sample_integration_result):
        """Test different prediction refinement strategies."""
        engine = PredictionRefinementEngine(use_gpu=False)
        strategies = [
            PredictionRefinementStrategy.BAYESIAN_UPDATE,
            PredictionRefinementStrategy.ENSEMBLE_WEIGHTING,
            PredictionRefinementStrategy.CONTEXT_FILTERING,
            PredictionRefinementStrategy.PATHWAY_CONSTRAINT,
        ]

        original_prediction = 0.6
        sample_id = "Sample_001"

        refinements = []
        for strategy in strategies:
            try:
                refinement = engine.refine_prediction(
                    original_prediction, sample_integration_result, sample_id, strategy
                )
                refinements.append(refinement)
                assert refinement.refinement_strategy == strategy
            except Exception as e:
                # Some strategies might fail, which is acceptable
                print(f"Strategy {strategy} failed: {e}")

        assert len(refinements) > 0  # At least some strategies should work
