"""
Tests for Recyclability and Re-usability Assessment Module.

This module provides comprehensive testing for the recyclability assessment
framework including mechanical, chemical, and biological recycling predictors.
"""

import asyncio
from unittest.mock import Mock, patch

import numpy as np
import pytest
from chemi_accel.sustainability.recyclability_assessment import (  # Core predictors; Analysis components; Data structures; Utility functions; Exceptions
    BiologicalRecyclingPredictor,
    ChemicalRecyclingPredictor,
    CircularityMetrics,
    ComprehensiveRecyclabilityAssessor,
    InvalidCompositionError,
    MaterialCategory,
    MaterialComposition,
    MechanicalRecyclingPredictor,
    MolecularStructureAnalyzer,
    RecyclabilityAssessment,
    RecyclabilityAssessmentError,
    RecyclabilityMetric,
    RecyclabilityScore,
    RecyclingConditions,
    RecyclingMethod,
    UnsupportedRecyclingMethodError,
    assess_material_recyclability,
    create_biodegradable_composition,
    create_standard_plastic_composition,
)

# Check for optional dependencies
try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from rdkit import Chem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class TestMaterialComposition:
    """Test material composition data structure."""

    def test_composition_creation(self):
        """Test creating material composition."""
        composition = MaterialComposition(
            polymer_content={"PET": 0.95},
            additive_content={"stabilizer": 0.03, "colorant": 0.02},
            filler_content={},
            molecular_weight_distribution=(25000, 50000),
            crystallinity=0.3,
        )

        assert composition.polymer_content["PET"] == 0.95
        assert composition.additive_content["stabilizer"] == 0.03
        assert composition.molecular_weight_distribution == (25000, 50000)
        assert composition.crystallinity == 0.3

    def test_standard_plastic_compositions(self):
        """Test standard plastic composition creation."""
        pet = create_standard_plastic_composition("PET")
        assert isinstance(pet, MaterialComposition)
        assert "PET" in pet.polymer_content
        assert pet.crystallinity > 0

        pe = create_standard_plastic_composition("PE")
        assert isinstance(pe, MaterialComposition)
        assert "HDPE" in pe.polymer_content

        pp = create_standard_plastic_composition("PP")
        assert isinstance(pp, MaterialComposition)
        assert "PP" in pp.polymer_content

    def test_biodegradable_compositions(self):
        """Test biodegradable composition creation."""
        pla = create_biodegradable_composition("PLA")
        assert isinstance(pla, MaterialComposition)
        assert "PLA" in pla.polymer_content

        pha = create_biodegradable_composition("PHA")
        assert isinstance(pha, MaterialComposition)
        assert "PHA" in pha.polymer_content

        with pytest.raises(ValueError):
            create_biodegradable_composition("unknown")


class TestRecyclingConditions:
    """Test recycling conditions data structure."""

    def test_conditions_creation(self):
        """Test creating recycling conditions."""
        conditions = RecyclingConditions(
            method=RecyclingMethod.MECHANICAL_RECYCLING,
            temperature=180.0,
            pressure=2.0,
            time=30.0,
            preprocessing_required=True,
            energy_requirements=0.8,
        )

        assert conditions.method == RecyclingMethod.MECHANICAL_RECYCLING
        assert conditions.temperature == 180.0
        assert conditions.preprocessing_required is True
        assert conditions.energy_requirements == 0.8


class TestRecyclabilityScore:
    """Test recyclability score data structure."""

    def test_score_creation(self):
        """Test creating recyclability score."""
        metric_scores = {
            RecyclabilityMetric.PROCESSING_EASE: 0.8,
            RecyclabilityMetric.MATERIAL_RECOVERY_RATE: 0.75,
            RecyclabilityMetric.QUALITY_RETENTION: 0.7,
        }

        score = RecyclabilityScore(
            method=RecyclingMethod.MECHANICAL_RECYCLING,
            overall_score=0.75,
            metric_scores=metric_scores,
            feasibility=True,
            confidence=0.8,
            technical_barriers=["High contamination sensitivity"],
            advantages=["Established infrastructure", "Low energy requirements"],
        )

        assert score.method == RecyclingMethod.MECHANICAL_RECYCLING
        assert score.overall_score == 0.75
        assert score.feasibility is True
        assert len(score.technical_barriers) == 1
        assert len(score.advantages) == 2


class TestCircularityMetrics:
    """Test circularity metrics data structure."""

    def test_metrics_creation(self):
        """Test creating circularity metrics."""
        metrics = CircularityMetrics(
            circularity_index=0.7,
            recyclability_score=0.8,
            reusability_score=0.6,
            biodegradability_score=0.1,
            durability_score=0.9,
            design_for_recycling_score=0.7,
            material_compatibility_score=0.8,
            economic_viability_score=0.6,
            infrastructure_readiness_score=0.8,
            lifecycle_impact_score=0.3,
        )

        assert metrics.circularity_index == 0.7
        assert metrics.recyclability_score == 0.8
        assert metrics.biodegradability_score == 0.1
        assert metrics.infrastructure_readiness_score == 0.8


class TestMolecularStructureAnalyzer:
    """Test molecular structure analyzer."""

    def test_analyzer_creation(self):
        """Test analyzer initialization."""
        analyzer = MolecularStructureAnalyzer()

        assert isinstance(analyzer.structure_patterns, dict)
        assert "ester_bonds" in analyzer.structure_patterns
        assert "aromatic_rings" in analyzer.structure_patterns

    def test_structure_analysis_without_rdkit(self):
        """Test structure analysis without RDKit."""
        analyzer = MolecularStructureAnalyzer()

        # Simple SMILES string
        smiles = "CC(=O)OCC"  # Ethyl acetate

        with patch(
            "chemi_accel.sustainability.recyclability_assessment.RDKIT_AVAILABLE", False
        ):
            analysis = analyzer.analyze_structure(smiles)

        assert isinstance(analysis, dict)
        assert "molecular_weight" in analysis
        assert "functional_groups" in analysis
        assert analysis["molecular_weight"] > 0

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_structure_analysis_with_rdkit(self):
        """Test structure analysis with RDKit."""
        analyzer = MolecularStructureAnalyzer()

        # PET-like structure
        smiles = "CC(=O)OCC"

        analysis = analyzer.analyze_structure(smiles)

        assert isinstance(analysis, dict)
        assert "molecular_weight" in analysis
        assert "functional_groups" in analysis
        assert "hydrolyzable_bonds" in analysis
        assert analysis["molecular_weight"] > 0

    def test_invalid_smiles_handling(self):
        """Test handling of invalid SMILES."""
        analyzer = MolecularStructureAnalyzer()

        invalid_smiles = "invalid_smiles"
        analysis = analyzer.analyze_structure(invalid_smiles)

        assert isinstance(analysis, dict)
        # Should fallback to simplified analysis


class TestMechanicalRecyclingPredictor:
    """Test mechanical recycling predictor."""

    def test_predictor_creation(self):
        """Test predictor initialization."""
        predictor = MechanicalRecyclingPredictor()

        assert predictor.thermal_stability_threshold == 200.0
        assert isinstance(predictor.contamination_tolerance, dict)

    def test_recyclability_assessment(self):
        """Test recyclability assessment."""
        predictor = MechanicalRecyclingPredictor()
        composition = create_standard_plastic_composition("PET")

        assessment = predictor.assess_recyclability(composition)

        assert isinstance(assessment, RecyclabilityAssessment)
        assert RecyclingMethod.MECHANICAL_RECYCLING in assessment.recycling_scores
        assert isinstance(assessment.circularity_metrics, CircularityMetrics)
        assert len(assessment.processing_recommendations) > 0

    def test_method_feasibility_prediction(self):
        """Test method feasibility prediction."""
        predictor = MechanicalRecyclingPredictor()
        composition = create_standard_plastic_composition("PET")

        score = predictor.predict_method_feasibility(
            composition, RecyclingMethod.MECHANICAL_RECYCLING
        )

        assert isinstance(score, RecyclabilityScore)
        assert score.method == RecyclingMethod.MECHANICAL_RECYCLING
        assert 0.0 <= score.overall_score <= 1.0
        assert score.confidence > 0.0

    def test_unsupported_method(self):
        """Test handling of unsupported recycling method."""
        predictor = MechanicalRecyclingPredictor()
        composition = create_standard_plastic_composition("PET")

        with pytest.raises(ValueError):
            predictor.predict_method_feasibility(
                composition, RecyclingMethod.CHEMICAL_RECYCLING
            )

    def test_thermal_properties_scoring(self):
        """Test thermal properties scoring."""
        predictor = MechanicalRecyclingPredictor()

        # High degradation temperature
        high_temp_composition = MaterialComposition(
            polymer_content={"High_Temp_Polymer": 1.0},
            additive_content={},
            filler_content={},
            degradation_temp=350.0,
        )

        high_score = predictor._score_thermal_properties(high_temp_composition)

        # Low degradation temperature
        low_temp_composition = MaterialComposition(
            polymer_content={"Low_Temp_Polymer": 1.0},
            additive_content={},
            filler_content={},
            degradation_temp=150.0,
        )

        low_score = predictor._score_thermal_properties(low_temp_composition)

        assert high_score > low_score
        assert 0.0 <= high_score <= 1.0
        assert 0.0 <= low_score <= 1.0

    def test_material_purity_scoring(self):
        """Test material purity scoring."""
        predictor = MechanicalRecyclingPredictor()

        # High purity
        high_purity = MaterialComposition(
            polymer_content={"PET": 0.95},
            additive_content={"stabilizer": 0.05},
            filler_content={},
        )

        high_score = predictor._score_material_purity(high_purity)

        # Low purity
        low_purity = MaterialComposition(
            polymer_content={"PET": 0.60},
            additive_content={"various": 0.25},
            filler_content={"glass": 0.15},
        )

        low_score = predictor._score_material_purity(low_purity)

        assert high_score > low_score

    def test_contamination_tolerance_scoring(self):
        """Test contamination tolerance scoring."""
        predictor = MechanicalRecyclingPredictor()

        # Clean material
        clean_composition = MaterialComposition(
            polymer_content={"PET": 1.0},
            additive_content={},
            filler_content={},
            contaminants={},
        )

        clean_score = predictor._score_contamination_tolerance(clean_composition)

        # Contaminated material
        contaminated_composition = MaterialComposition(
            polymer_content={"PET": 0.8},
            additive_content={},
            filler_content={},
            contaminants={"paper": 0.1, "adhesive": 0.1},
        )

        contaminated_score = predictor._score_contamination_tolerance(
            contaminated_composition
        )

        assert clean_score > contaminated_score
        assert clean_score == 1.0

    def test_material_category_determination(self):
        """Test material category determination."""
        predictor = MechanicalRecyclingPredictor()

        # Thermoplastic
        thermoplastic = MaterialComposition(
            polymer_content={"PET": 1.0}, additive_content={}, filler_content={}
        )

        category = predictor._determine_material_category(thermoplastic)
        assert category == MaterialCategory.THERMOPLASTIC

        # Thermoset
        thermoset = MaterialComposition(
            polymer_content={"Epoxy": 1.0},
            additive_content={},
            filler_content={},
            crosslink_density=0.2,
        )

        category = predictor._determine_material_category(thermoset)
        assert category == MaterialCategory.THERMOSET


class TestChemicalRecyclingPredictor:
    """Test chemical recycling predictor."""

    def test_predictor_creation(self):
        """Test predictor initialization."""
        predictor = ChemicalRecyclingPredictor(use_gpu=False)

        assert predictor.use_gpu is False
        assert isinstance(predictor.structure_analyzer, MolecularStructureAnalyzer)
        assert isinstance(predictor.method_requirements, dict)

    def test_recyclability_assessment(self):
        """Test recyclability assessment."""
        predictor = ChemicalRecyclingPredictor(use_gpu=False)
        composition = create_standard_plastic_composition("PET")

        assessment = predictor.assess_recyclability(composition)

        assert isinstance(assessment, RecyclabilityAssessment)
        assert len(assessment.recycling_scores) > 1  # Multiple methods
        assert len(assessment.recommended_methods) >= 0
        assert isinstance(assessment.circularity_metrics, CircularityMetrics)

    def test_depolymerization_assessment(self):
        """Test depolymerization feasibility assessment."""
        predictor = ChemicalRecyclingPredictor(use_gpu=False)

        # PET with ester bonds - should be suitable for depolymerization
        pet_composition = MaterialComposition(
            polymer_content={"PET": 0.95},
            additive_content={"stabilizer": 0.05},
            filler_content={},
            chemical_structure="CC(=O)OCC",
        )

        score = predictor.predict_method_feasibility(
            pet_composition, RecyclingMethod.DEPOLYMERIZATION
        )

        assert isinstance(score, RecyclabilityScore)
        assert score.method == RecyclingMethod.DEPOLYMERIZATION
        assert 0.0 <= score.overall_score <= 1.0

    def test_pyrolysis_assessment(self):
        """Test pyrolysis feasibility assessment."""
        predictor = ChemicalRecyclingPredictor(use_gpu=False)
        composition = create_standard_plastic_composition("PE")

        score = predictor.predict_method_feasibility(
            composition, RecyclingMethod.PYROLYSIS
        )

        assert isinstance(score, RecyclabilityScore)
        assert score.method == RecyclingMethod.PYROLYSIS
        assert score.feasibility  # Pyrolysis is generally applicable

    def test_solvent_recycling_assessment(self):
        """Test solvent-based recycling assessment."""
        predictor = ChemicalRecyclingPredictor(use_gpu=False)

        # Non-crosslinked polymer should be suitable
        composition = MaterialComposition(
            polymer_content={"PS": 0.98},
            additive_content={"antioxidant": 0.02},
            filler_content={},
            crosslink_density=0.0,
        )

        score = predictor.predict_method_feasibility(
            composition, RecyclingMethod.SOLVENT_BASED_RECYCLING
        )

        assert isinstance(score, RecyclabilityScore)
        assert score.method == RecyclingMethod.SOLVENT_BASED_RECYCLING

    def test_hydrolysis_assessment(self):
        """Test hydrolysis feasibility assessment."""
        predictor = ChemicalRecyclingPredictor(use_gpu=False)

        # PET with hydrolyzable ester bonds
        composition = MaterialComposition(
            polymer_content={"PET": 1.0},
            additive_content={},
            filler_content={},
            chemical_structure="CC(=O)OCC",
        )

        score = predictor.predict_method_feasibility(
            composition, RecyclingMethod.HYDROLYSIS
        )

        assert isinstance(score, RecyclabilityScore)
        assert score.method == RecyclingMethod.HYDROLYSIS


class TestBiologicalRecyclingPredictor:
    """Test biological recycling predictor."""

    def test_predictor_creation(self):
        """Test predictor initialization."""
        predictor = BiologicalRecyclingPredictor()

        assert isinstance(predictor.biodegradation_patterns, dict)
        assert "ester" in predictor.biodegradation_patterns
        assert "aromatic" in predictor.biodegradation_patterns

    def test_recyclability_assessment(self):
        """Test recyclability assessment."""
        predictor = BiologicalRecyclingPredictor()
        composition = create_biodegradable_composition("PLA")

        assessment = predictor.assess_recyclability(composition)

        assert isinstance(assessment, RecyclabilityAssessment)
        assert assessment.material_category == MaterialCategory.BIODEGRADABLE
        assert len(assessment.recycling_scores) >= 1

    def test_general_biodegradation_assessment(self):
        """Test general biodegradation assessment."""
        predictor = BiologicalRecyclingPredictor()

        # Biodegradable polymer
        biodegradable_composition = MaterialComposition(
            polymer_content={"PLA": 0.98},
            additive_content={"plasticizer": 0.02},
            filler_content={},
        )

        score = predictor.predict_method_feasibility(
            biodegradable_composition, RecyclingMethod.BIOLOGICAL_RECYCLING
        )

        assert isinstance(score, RecyclabilityScore)
        assert score.method == RecyclingMethod.BIOLOGICAL_RECYCLING
        assert score.feasibility  # PLA should be biodegradable

    def test_enzymatic_recycling_assessment(self):
        """Test enzymatic recycling assessment."""
        predictor = BiologicalRecyclingPredictor()

        # PET - suitable for enzymatic recycling
        pet_composition = MaterialComposition(
            polymer_content={"PET": 0.95},
            additive_content={"stabilizer": 0.05},
            filler_content={},
        )

        score = predictor.predict_method_feasibility(
            pet_composition, RecyclingMethod.ENZYMATIC_RECYCLING
        )

        assert isinstance(score, RecyclabilityScore)
        assert score.method == RecyclingMethod.ENZYMATIC_RECYCLING
        # PET should be suitable for enzymatic recycling


class TestComprehensiveRecyclabilityAssessor:
    """Test comprehensive recyclability assessor."""

    def test_assessor_creation(self):
        """Test assessor initialization."""
        assessor = ComprehensiveRecyclabilityAssessor(use_gpu=False)

        assert len(assessor.predictors) == 3
        assert "mechanical" in assessor.predictors
        assert "chemical" in assessor.predictors
        assert "biological" in assessor.predictors
        assert sum(assessor.ensemble_weights.values()) == 1.0

    @pytest.mark.asyncio
    async def test_comprehensive_assessment(self):
        """Test comprehensive recyclability assessment."""
        assessor = ComprehensiveRecyclabilityAssessor(use_gpu=False)
        composition = create_standard_plastic_composition("PET")

        assessment = await assessor.assess_comprehensive_recyclability(composition)

        assert isinstance(assessment, RecyclabilityAssessment)
        assert (
            len(assessment.recycling_scores) > 1
        )  # Multiple methods from different predictors
        assert isinstance(assessment.circularity_metrics, CircularityMetrics)
        assert "comprehensive" in assessment.material_id

    @pytest.mark.asyncio
    async def test_assessment_with_error_handling(self):
        """Test assessment with error handling."""
        assessor = ComprehensiveRecyclabilityAssessor(use_gpu=False)

        # Mock one predictor to raise an error
        with patch.object(
            assessor.predictors["mechanical"], "assess_recyclability"
        ) as mock_assess:
            mock_assess.side_effect = Exception("Test error")

            composition = create_standard_plastic_composition("PE")
            assessment = await assessor.assess_comprehensive_recyclability(composition)

            # Should still complete with remaining predictors
            assert isinstance(assessment, RecyclabilityAssessment)


class TestUtilityFunctions:
    """Test utility functions."""

    @pytest.mark.asyncio
    async def test_assess_material_recyclability(self):
        """Test utility function for material assessment."""
        composition = create_standard_plastic_composition("PET")

        assessment = await assess_material_recyclability(composition, use_gpu=False)

        assert isinstance(assessment, RecyclabilityAssessment)
        assert len(assessment.recycling_scores) > 0

    @pytest.mark.asyncio
    async def test_assess_with_specific_methods(self):
        """Test assessment with specific methods."""
        composition = create_biodegradable_composition("PLA")

        assessment = await assess_material_recyclability(
            composition, methods=["mechanical", "biological"], use_gpu=False
        )

        assert isinstance(assessment, RecyclabilityAssessment)
        # Should only include results from specified methods


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_recycling_method_enum(self):
        """Test recycling method enumeration."""
        assert RecyclingMethod.MECHANICAL_RECYCLING.value == "mechanical_recycling"
        assert RecyclingMethod.CHEMICAL_RECYCLING.value == "chemical_recycling"
        assert RecyclingMethod.BIOLOGICAL_RECYCLING.value == "biological_recycling"

    def test_material_category_enum(self):
        """Test material category enumeration."""
        assert MaterialCategory.THERMOPLASTIC.value == "thermoplastic"
        assert MaterialCategory.THERMOSET.value == "thermoset"
        assert MaterialCategory.BIODEGRADABLE.value == "biodegradable"

    def test_recyclability_metric_enum(self):
        """Test recyclability metric enumeration."""
        assert RecyclabilityMetric.PROCESSING_EASE.value == "processing_ease"
        assert (
            RecyclabilityMetric.MATERIAL_RECOVERY_RATE.value == "material_recovery_rate"
        )
        assert RecyclabilityMetric.QUALITY_RETENTION.value == "quality_retention"

    def test_invalid_plastic_type(self):
        """Test handling of invalid plastic type."""
        # Should fall back to PET for unknown types
        composition = create_standard_plastic_composition("Unknown")
        assert isinstance(composition, MaterialComposition)

    def test_empty_composition(self):
        """Test handling of empty composition."""
        empty_composition = MaterialComposition(
            polymer_content={}, additive_content={}, filler_content={}
        )

        predictor = MechanicalRecyclingPredictor()
        assessment = predictor.assess_recyclability(empty_composition)

        assert isinstance(assessment, RecyclabilityAssessment)


class TestPerformance:
    """Test performance aspects."""

    def test_assessment_timing(self):
        """Test that assessments complete in reasonable time."""
        predictor = MechanicalRecyclingPredictor()
        composition = create_standard_plastic_composition("PET")

        import time

        start_time = time.time()
        assessment = predictor.assess_recyclability(composition)
        end_time = time.time()

        execution_time = end_time - start_time
        assert execution_time < 1.0  # Should complete within 1 second
        assert isinstance(assessment, RecyclabilityAssessment)

    @pytest.mark.asyncio
    async def test_async_assessment_timing(self):
        """Test async assessment timing."""
        import time

        start_time = time.time()

        composition = create_standard_plastic_composition("PET")
        assessment = await assess_material_recyclability(composition, use_gpu=False)

        end_time = time.time()
        execution_time = end_time - start_time

        assert execution_time < 5.0  # Should complete within 5 seconds
        assert isinstance(assessment, RecyclabilityAssessment)


if __name__ == "__main__":
    pytest.main([__file__])
