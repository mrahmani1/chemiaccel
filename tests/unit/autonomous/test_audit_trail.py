"""
Tests for Audit Trail System

This module contains comprehensive tests for the audit trail system including
compliance checking, digital signatures, and regulatory validation.
"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from chemi_accel.autonomous.audit_trail import (
    AuditConfiguration,
    AuditLevel,
    AuditMetadata,
    AuditTrailManager,
    ComplianceStandard,
    DigitalSignature,
    ExperimentalDesignAudit,
    MoleculeSelectionAudit,
    ResultValidationAudit,
    ValidationStatus,
)


class TestAuditConfiguration:
    """Test cases for AuditConfiguration dataclass."""

    def test_audit_configuration_default(self):
        """Test default audit configuration."""
        config = AuditConfiguration(
            audit_level=AuditLevel.DETAILED,
            compliance_standards=[ComplianceStandard.FDA_CFR21],
            retention_period_days=2555,  # 7 years
            require_digital_signatures=True,
            enable_automatic_validation=True,
            validation_rules={"min_confidence": 0.7},
        )

        assert config.audit_level == AuditLevel.DETAILED
        assert ComplianceStandard.FDA_CFR21 in config.compliance_standards
        assert config.retention_period_days == 2555
        assert config.require_digital_signatures is True
        assert config.enable_automatic_validation is True
        assert config.validation_rules["min_confidence"] == 0.7
        assert config.audit_storage_encrypted is True
        assert config.backup_frequency_hours == 24
        assert config.access_control_enabled is True

    def test_audit_configuration_custom(self):
        """Test custom audit configuration."""
        config = AuditConfiguration(
            audit_level=AuditLevel.REGULATORY,
            compliance_standards=[ComplianceStandard.GLP, ComplianceStandard.ISO17025],
            retention_period_days=3650,  # 10 years
            require_digital_signatures=True,
            enable_automatic_validation=False,
            validation_rules={"custom_rule": "value"},
            audit_storage_encrypted=True,
            backup_frequency_hours=12,
            access_control_enabled=True,
        )

        assert config.audit_level == AuditLevel.REGULATORY
        assert len(config.compliance_standards) == 2
        assert ComplianceStandard.GLP in config.compliance_standards
        assert ComplianceStandard.ISO17025 in config.compliance_standards
        assert config.retention_period_days == 3650
        assert config.enable_automatic_validation is False
        assert config.backup_frequency_hours == 12


class TestDigitalSignature:
    """Test cases for DigitalSignature dataclass."""

    def test_digital_signature_creation(self):
        """Test digital signature creation."""
        signature = DigitalSignature(
            signature_id="sig_001",
            signer_id="researcher_123",
            signer_role="principal_investigator",
            timestamp=datetime.now(timezone.utc),
            signature_hash="abc123def456",
            certificate_fingerprint="cert_fingerprint_789",
            validation_status=ValidationStatus.VALIDATED,
            signature_algorithm="SHA256withRSA",
        )

        assert signature.signature_id == "sig_001"
        assert signature.signer_id == "researcher_123"
        assert signature.signer_role == "principal_investigator"
        assert signature.signature_hash == "abc123def456"
        assert signature.certificate_fingerprint == "cert_fingerprint_789"
        assert signature.validation_status == ValidationStatus.VALIDATED
        assert signature.signature_algorithm == "SHA256withRSA"
        assert signature.timestamp is not None


class TestAuditMetadata:
    """Test cases for AuditMetadata dataclass."""

    def test_audit_metadata_creation(self):
        """Test audit metadata creation."""
        metadata = AuditMetadata(
            creation_timestamp=datetime.now(timezone.utc),
            creator_id="user_001",
            creator_role="researcher",
            system_version="1.0.0",
            compliance_flags={"fda_cfr21": True, "glp": True},
            data_integrity_hash="hash123",
            access_history=[
                {
                    "timestamp": "2024-01-01T10:00:00Z",
                    "user_id": "user_001",
                    "action": "create",
                }
            ],
            modification_history=[],
            validation_checks={"initial_validation": {"status": "passed"}},
        )

        assert metadata.creator_id == "user_001"
        assert metadata.creator_role == "researcher"
        assert metadata.system_version == "1.0.0"
        assert metadata.compliance_flags["fda_cfr21"] is True
        assert metadata.data_integrity_hash == "hash123"
        assert len(metadata.access_history) == 1
        assert len(metadata.modification_history) == 0
        assert "initial_validation" in metadata.validation_checks


class TestExperimentalDesignAudit:
    """Test cases for ExperimentalDesignAudit dataclass."""

    def test_experimental_design_audit_creation(self):
        """Test experimental design audit creation."""
        audit = ExperimentalDesignAudit(
            audit_id="audit_001",
            decision_id="decision_001",
            experiment_id="exp_001",
            design_methodology="factorial_design",
            ai_model_details={
                "model_name": "design_optimizer",
                "version": "2.0",
                "training_data": "dataset_v3",
            },
            input_parameters={
                "temperature_range": [50, 150],
                "catalyst_loading": [0.01, 0.1],
                "reaction_time": [1, 12],
            },
            design_rationale="Factorial design chosen to explore parameter interactions",
            alternative_designs_considered=[
                {
                    "design": "latin_hypercube",
                    "reason_rejected": "insufficient_coverage",
                },
                {"design": "random_sampling", "reason_rejected": "poor_efficiency"},
            ],
            risk_assessment={
                "safety_risk": "low",
                "cost_risk": "medium",
                "time_risk": "low",
            },
            expected_outcomes={
                "primary_endpoint": {"yield": 0.8},
                "secondary_endpoints": {"selectivity": 0.9, "purity": 0.95},
            },
            resource_justification={
                "computational": "high_dimensional_optimization_required",
                "experimental": "full_factorial_necessary_for_interactions",
            },
            ethical_considerations={
                "animal_testing": False,
                "environmental_impact": "minimal",
                "sustainability": "considered",
            },
            safety_assessments={
                "hazard_analysis": "completed",
                "risk_mitigation": "protocols_in_place",
            },
            regulatory_compliance={"fda_cfr21": True, "glp": True, "iso17025": False},
        )

        assert audit.audit_id == "audit_001"
        assert audit.decision_id == "decision_001"
        assert audit.experiment_id == "exp_001"
        assert audit.design_methodology == "factorial_design"
        assert audit.ai_model_details["model_name"] == "design_optimizer"
        assert len(audit.input_parameters) == 3
        assert "Factorial design" in audit.design_rationale
        assert len(audit.alternative_designs_considered) == 2
        assert audit.risk_assessment["safety_risk"] == "low"
        assert audit.expected_outcomes["primary_endpoint"]["yield"] == 0.8
        assert audit.regulatory_compliance["fda_cfr21"] is True
        assert len(audit.digital_signatures) == 0


class TestMoleculeSelectionAudit:
    """Test cases for MoleculeSelectionAudit dataclass."""

    def test_molecule_selection_audit_creation(self):
        """Test molecule selection audit creation."""
        audit = MoleculeSelectionAudit(
            audit_id="mol_audit_001",
            decision_id="mol_decision_001",
            selection_algorithm="active_learning_gp",
            training_data_provenance={
                "dataset_name": "chembl_v30",
                "data_version": "2024.01",
                "preprocessing": "standardized",
                "validation_split": "temporal",
            },
            bias_analysis={
                "bias_score": 0.15,
                "bias_sources": ["training_data_imbalance", "model_architecture"],
                "mitigation_strategies": ["data_augmentation", "fairness_constraints"],
            },
            diversity_justification={
                "diversity_metric": "tanimoto_distance",
                "diversity_score": 0.82,
                "justification": "ensures_broad_chemical_space_coverage",
            },
            novelty_assessment={
                "novelty_score": 0.75,
                "comparison_database": "scifinder",
                "patent_search": "completed",
            },
            intellectual_property_clearance={
                "molecule_1": True,
                "molecule_2": True,
                "molecule_3": False,
            },
            environmental_impact={
                "environmental_score": "green",
                "biodegradability": "high",
                "toxicity_prediction": "low_risk",
            },
            synthesis_feasibility={
                "feasibility_score": 0.85,
                "complexity_score": "medium",
                "reagent_availability": "high",
            },
            cost_benefit_analysis={
                "estimated_cost": 5000,
                "expected_value": 15000,
                "roi": 3.0,
            },
            literature_review={
                "papers_reviewed": 45,
                "relevant_studies": 12,
                "knowledge_gaps": "identified",
            },
            expert_consultation=[
                {
                    "expert_id": "expert_001",
                    "expertise": "medicinal_chemistry",
                    "consultation_date": "2024-01-15",
                    "recommendation": "approved",
                }
            ],
        )

        assert audit.audit_id == "mol_audit_001"
        assert audit.decision_id == "mol_decision_001"
        assert audit.selection_algorithm == "active_learning_gp"
        assert audit.training_data_provenance["dataset_name"] == "chembl_v30"
        assert audit.bias_analysis["bias_score"] == 0.15
        assert audit.diversity_justification["diversity_score"] == 0.82
        assert audit.novelty_assessment["novelty_score"] == 0.75
        assert audit.intellectual_property_clearance["molecule_1"] is True
        assert audit.environmental_impact["environmental_score"] == "green"
        assert audit.synthesis_feasibility["feasibility_score"] == 0.85
        assert audit.cost_benefit_analysis["roi"] == 3.0
        assert audit.literature_review["papers_reviewed"] == 45
        assert len(audit.expert_consultation) == 1


class TestResultValidationAudit:
    """Test cases for ResultValidationAudit dataclass."""

    def test_result_validation_audit_creation(self):
        """Test result validation audit creation."""
        audit = ResultValidationAudit(
            audit_id="val_audit_001",
            experiment_id="exp_001",
            validation_methodology="statistical_analysis_with_controls",
            statistical_analysis={
                "method": "anova",
                "p_value": 0.001,
                "confidence_interval": [0.75, 0.85],
                "effect_size": 0.8,
            },
            outlier_detection={
                "method": "z_score",
                "outliers_found": 2,
                "outlier_threshold": 3.0,
                "outliers_removed": True,
            },
            quality_control_checks={
                "control_samples": "passed",
                "blank_samples": "passed",
                "duplicate_analysis": "cv_within_limits",
            },
            measurement_uncertainty={
                "type_a_uncertainty": 0.02,
                "type_b_uncertainty": 0.015,
                "combined_uncertainty": 0.025,
                "coverage_factor": 2,
            },
            calibration_records=[
                {
                    "instrument": "hplc_001",
                    "calibration_date": "2024-01-01",
                    "calibration_standard": "caffeine",
                    "r_squared": 0.9998,
                }
            ],
            reference_standards=[
                {
                    "standard_name": "caffeine",
                    "purity": "99.9%",
                    "certificate_number": "cert_001",
                    "expiry_date": "2025-01-01",
                }
            ],
            analyst_qualifications={
                "analyst_id": "analyst_001",
                "certification": "chromatography_specialist",
                "training_date": "2023-06-01",
                "competency_assessment": "passed",
            },
            instrument_validation={
                "instrument_id": "hplc_001",
                "validation_protocol": "protocol_v2.1",
                "validation_date": "2024-01-01",
                "next_validation": "2025-01-01",
            },
            environmental_conditions={
                "temperature": "22±2°C",
                "humidity": "45±5%",
                "pressure": "1013±10 mbar",
            },
        )

        assert audit.audit_id == "val_audit_001"
        assert audit.experiment_id == "exp_001"
        assert audit.validation_methodology == "statistical_analysis_with_controls"
        assert audit.statistical_analysis["method"] == "anova"
        assert audit.statistical_analysis["p_value"] == 0.001
        assert audit.outlier_detection["method"] == "z_score"
        assert audit.outlier_detection["outliers_found"] == 2
        assert audit.quality_control_checks["control_samples"] == "passed"
        assert audit.measurement_uncertainty["combined_uncertainty"] == 0.025
        assert len(audit.calibration_records) == 1
        assert len(audit.reference_standards) == 1
        assert (
            audit.analyst_qualifications["certification"] == "chromatography_specialist"
        )
        assert audit.instrument_validation["instrument_id"] == "hplc_001"


class TestAuditTrailManager:
    """Test cases for AuditTrailManager class."""

    @pytest.fixture
    def audit_config(self):
        """Create test audit configuration."""
        return AuditConfiguration(
            audit_level=AuditLevel.DETAILED,
            compliance_standards=[ComplianceStandard.FDA_CFR21, ComplianceStandard.GLP],
            retention_period_days=2555,
            require_digital_signatures=True,
            enable_automatic_validation=False,  # Disable for testing
            validation_rules={"min_confidence": 0.7, "require_peer_review": True},
        )

    @pytest.fixture
    def audit_manager(self, audit_config):
        """Create audit trail manager instance."""
        return AuditTrailManager(audit_config)

    def test_audit_manager_initialization(self, audit_manager, audit_config):
        """Test audit manager initialization."""
        assert audit_manager.config == audit_config
        assert len(audit_manager.experimental_design_audits) == 0
        assert len(audit_manager.molecule_selection_audits) == 0
        assert len(audit_manager.result_validation_audits) == 0
        assert len(audit_manager.access_log) == 0
        assert len(audit_manager.validation_queue) == 0

        # Check compliance rules were initialized
        assert ComplianceStandard.FDA_CFR21 in audit_manager.compliance_rules
        assert ComplianceStandard.GLP in audit_manager.compliance_rules

    def test_compliance_rules_initialization(self, audit_manager):
        """Test compliance rules initialization."""
        rules = audit_manager.compliance_rules

        # Check FDA CFR 21 rules
        fda_rules = rules[ComplianceStandard.FDA_CFR21]
        assert fda_rules["digital_signatures_required"] is True
        assert fda_rules["audit_trail_completeness"] is True
        assert fda_rules["data_integrity_validation"] is True
        assert fda_rules["access_control_required"] is True

        # Check GLP rules
        glp_rules = rules[ComplianceStandard.GLP]
        assert glp_rules["study_protocol_required"] is True
        assert glp_rules["raw_data_retention"] is True
        assert glp_rules["quality_assurance_review"] is True

        # Check ISO 17025 rules
        iso_rules = rules[ComplianceStandard.ISO17025]
        assert iso_rules["measurement_traceability"] is True
        assert iso_rules["method_validation"] is True
        assert iso_rules["uncertainty_estimation"] is True

    @pytest.mark.asyncio
    async def test_create_experimental_design_audit(self, audit_manager):
        """Test creating experimental design audit."""
        decision_id = "decision_001"
        design_methodology = "response_surface_methodology"
        ai_model_details = {
            "model_name": "bayesian_optimizer",
            "version": "3.1",
            "confidence": 0.82,
        }
        input_parameters = {
            "temperature": {"min": 60, "max": 120},
            "pressure": {"min": 1, "max": 5},
            "catalyst": ["cat_a", "cat_b", "cat_c"],
        }
        design_rationale = "RSM chosen for its efficiency in optimization"
        alternative_designs = [
            {"design": "factorial", "rejection_reason": "too_many_experiments"},
            {"design": "random", "rejection_reason": "poor_convergence"},
        ]
        risk_assessment = {
            "technical_risk": "medium",
            "safety_risk": "low",
            "timeline_risk": "low",
        }
        user_id = "researcher_001"
        experiment_id = "exp_001"

        audit_id = await audit_manager.create_experimental_design_audit(
            decision_id=decision_id,
            design_methodology=design_methodology,
            ai_model_details=ai_model_details,
            input_parameters=input_parameters,
            design_rationale=design_rationale,
            alternative_designs=alternative_designs,
            risk_assessment=risk_assessment,
            user_id=user_id,
            experiment_id=experiment_id,
        )

        assert audit_id is not None
        assert audit_id in audit_manager.experimental_design_audits

        audit = audit_manager.experimental_design_audits[audit_id]
        assert audit.decision_id == decision_id
        assert audit.experiment_id == experiment_id
        assert audit.design_methodology == design_methodology
        assert audit.ai_model_details == ai_model_details
        assert audit.input_parameters == input_parameters
        assert audit.design_rationale == design_rationale
        assert len(audit.alternative_designs_considered) == 2
        assert audit.risk_assessment == risk_assessment

        # Check metadata
        assert audit.metadata.creator_id == user_id
        assert audit.metadata.data_integrity_hash is not None
        assert len(audit.metadata.access_history) == 1
        assert audit.metadata.access_history[0]["action"] == "create"

    @pytest.mark.asyncio
    async def test_create_molecule_selection_audit(self, audit_manager):
        """Test creating molecule selection audit."""
        decision_id = "mol_decision_001"
        selection_algorithm = "gaussian_process_active_learning"
        training_data_provenance = {
            "source": "chembl_30",
            "preprocessing": "standardized_smiles",
            "splits": {"train": 0.8, "val": 0.1, "test": 0.1},
        }
        selected_molecules = ["CCO", "CCC", "C1CCCCC1"]
        selection_criteria = {"diversity": 0.8, "novelty": 0.7, "synthesizability": 0.9}
        user_id = "chemist_002"

        audit_id = await audit_manager.create_molecule_selection_audit(
            decision_id=decision_id,
            selection_algorithm=selection_algorithm,
            training_data_provenance=training_data_provenance,
            selected_molecules=selected_molecules,
            selection_criteria=selection_criteria,
            user_id=user_id,
        )

        assert audit_id is not None
        assert audit_id in audit_manager.molecule_selection_audits

        audit = audit_manager.molecule_selection_audits[audit_id]
        assert audit.decision_id == decision_id
        assert audit.selection_algorithm == selection_algorithm
        assert audit.training_data_provenance == training_data_provenance

        # Check computed assessments
        assert "bias_score" in audit.bias_analysis
        assert "diversity_score" in audit.diversity_justification
        assert "novelty_score" in audit.novelty_assessment
        assert "environmental_score" in audit.environmental_impact
        assert "feasibility_score" in audit.synthesis_feasibility

        # Check metadata
        assert audit.metadata.creator_id == user_id
        assert audit.metadata.creator_role == "researcher"

    @pytest.mark.asyncio
    async def test_create_result_validation_audit(self, audit_manager):
        """Test creating result validation audit."""
        experiment_id = "exp_001"
        validation_methodology = "statistical_validation_with_qc"
        experimental_results = {"yield": 0.78, "purity": 0.94, "selectivity": 0.89}
        quality_metrics = {"r_squared": 0.95, "rmse": 0.03, "cv": 0.02}
        user_id = "analyst_001"

        audit_id = await audit_manager.create_result_validation_audit(
            experiment_id=experiment_id,
            validation_methodology=validation_methodology,
            experimental_results=experimental_results,
            quality_metrics=quality_metrics,
            user_id=user_id,
        )

        assert audit_id is not None
        assert audit_id in audit_manager.result_validation_audits

        audit = audit_manager.result_validation_audits[audit_id]
        assert audit.experiment_id == experiment_id
        assert audit.validation_methodology == validation_methodology

        # Check computed analyses
        assert "method" in audit.statistical_analysis
        assert "method" in audit.outlier_detection
        assert "control_samples" in audit.quality_control_checks
        assert "combined_uncertainty" in audit.measurement_uncertainty
        assert len(audit.calibration_records) > 0
        assert len(audit.reference_standards) > 0

        # Check metadata
        assert audit.metadata.creator_id == user_id
        assert audit.metadata.creator_role == "analyst"

    @pytest.mark.asyncio
    async def test_add_digital_signature(self, audit_manager):
        """Test adding digital signature to audit record."""
        # First create an audit record
        audit_id = await audit_manager.create_experimental_design_audit(
            decision_id="decision_001",
            design_methodology="factorial",
            ai_model_details={"model": "test"},
            input_parameters={"param": "value"},
            design_rationale="test rationale",
            alternative_designs=[],
            risk_assessment={},
            user_id="user_001",
        )

        # Add digital signature
        signature_id = await audit_manager.add_digital_signature(
            audit_id=audit_id,
            audit_type="experimental_design",
            signer_id="supervisor_001",
            signer_role="study_director",
            signature_data="digital_signature_data_here",
        )

        assert signature_id is not None

        audit = audit_manager.experimental_design_audits[audit_id]
        assert len(audit.digital_signatures) == 1

        signature = audit.digital_signatures[0]
        assert signature.signature_id == signature_id
        assert signature.signer_id == "supervisor_001"
        assert signature.signer_role == "study_director"
        assert signature.validation_status == ValidationStatus.PENDING
        assert signature.signature_hash is not None

    @pytest.mark.asyncio
    async def test_validate_audit_record_success(self, audit_manager):
        """Test successful audit record validation."""
        # Create audit record
        audit_id = await audit_manager.create_experimental_design_audit(
            decision_id="decision_001",
            design_methodology="factorial",
            ai_model_details={"model": "test"},
            input_parameters={"param": "value"},
            design_rationale="Comprehensive rationale for factorial design",
            alternative_designs=[
                {"design": "random", "reason": "insufficient_coverage"}
            ],
            risk_assessment={"safety": "low"},
            user_id="user_001",
        )

        # Add digital signature to meet requirements
        await audit_manager.add_digital_signature(
            audit_id=audit_id,
            audit_type="experimental_design",
            signer_id="supervisor_001",
            signer_role="study_director",
            signature_data="valid_signature",
        )

        # Validate the record
        validation_results = await audit_manager.validate_audit_record(
            audit_id=audit_id, audit_type="experimental_design"
        )

        assert validation_results is not None
        assert validation_results["audit_id"] == audit_id
        assert "validation_timestamp" in validation_results
        assert "compliance_checks" in validation_results
        assert "overall_status" in validation_results

        # Should have checks for each compliance standard
        assert "fda_cfr21" in validation_results["compliance_checks"]
        assert "glp" in validation_results["compliance_checks"]

        # Overall status should be reasonable
        assert validation_results["overall_status"] in [
            ValidationStatus.VALIDATED,
            ValidationStatus.REQUIRES_REVIEW,
            ValidationStatus.REJECTED,
        ]

    @pytest.mark.asyncio
    async def test_validate_nonexistent_audit_record(self, audit_manager):
        """Test validation of non-existent audit record."""
        validation_results = await audit_manager.validate_audit_record(
            audit_id="nonexistent_audit", audit_type="experimental_design"
        )

        assert validation_results["overall_status"] == ValidationStatus.REJECTED
        assert "Audit record not found" in validation_results["issues_found"]

    @pytest.mark.asyncio
    async def test_fda_cfr21_compliance_check(self, audit_manager):
        """Test FDA CFR 21 compliance checking."""
        # Create audit without digital signature
        audit_id = await audit_manager.create_experimental_design_audit(
            decision_id="decision_001",
            design_methodology="factorial",
            ai_model_details={"model": "test"},
            input_parameters={"param": "value"},
            design_rationale="test rationale",
            alternative_designs=[],
            risk_assessment={},
            user_id="user_001",
        )

        audit = audit_manager.experimental_design_audits[audit_id]

        # Check FDA CFR 21 compliance
        compliance_result = await audit_manager._check_fda_cfr21_compliance(
            audit, "experimental_design"
        )

        assert "fda_specific_issues" in compliance_result
        assert "fda_specific_recommendations" in compliance_result

    @pytest.mark.asyncio
    async def test_glp_compliance_check(self, audit_manager):
        """Test GLP compliance checking."""
        # Create audit without proper protocol
        audit_id = await audit_manager.create_experimental_design_audit(
            decision_id="decision_001",
            design_methodology="factorial",
            ai_model_details={"model": "test"},
            input_parameters={"param": "value"},
            design_rationale="",  # Empty rationale should trigger GLP issue
            alternative_designs=[],
            risk_assessment={},
            user_id="user_001",
        )

        audit = audit_manager.experimental_design_audits[audit_id]

        # Check GLP compliance
        compliance_result = await audit_manager._check_glp_compliance(
            audit, "experimental_design"
        )

        assert "glp_specific_issues" in compliance_result
        assert "glp_specific_recommendations" in compliance_result

        # Should detect missing study protocol/rationale
        if compliance_result["glp_specific_issues"]:
            assert any(
                "protocol" in issue.lower() or "rationale" in issue.lower()
                for issue in compliance_result["glp_specific_issues"]
            )

    @pytest.mark.asyncio
    async def test_iso17025_compliance_check(self, audit_manager):
        """Test ISO 17025 compliance checking."""
        # Create result validation audit
        audit_id = await audit_manager.create_result_validation_audit(
            experiment_id="exp_001",
            validation_methodology="statistical",
            experimental_results={"yield": 0.8},
            quality_metrics={"r_squared": 0.95},
            user_id="analyst_001",
        )

        audit = audit_manager.result_validation_audits[audit_id]

        # Check ISO 17025 compliance
        compliance_result = await audit_manager._check_iso17025_compliance(
            audit, "result_validation"
        )

        assert "iso_specific_issues" in compliance_result
        assert "iso_specific_recommendations" in compliance_result

    def test_calculate_integrity_hash(self, audit_manager):
        """Test data integrity hash calculation."""
        test_data = {"field1": "value1", "field2": 123, "field3": [1, 2, 3]}

        hash1 = audit_manager._calculate_integrity_hash(test_data)
        hash2 = audit_manager._calculate_integrity_hash(test_data)

        # Same data should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64-character hex string

        # Different data should produce different hash
        test_data_modified = test_data.copy()
        test_data_modified["field1"] = "modified_value"
        hash3 = audit_manager._calculate_integrity_hash(test_data_modified)

        assert hash1 != hash3

    def test_check_compliance_flags(self, audit_manager):
        """Test compliance flags checking."""
        audit_data = {"decision_id": "test_decision", "methodology": "test_method"}

        flags = audit_manager._check_compliance_flags(audit_data)

        # Should have flags for each compliance standard
        assert "fda_cfr21" in flags
        assert "glp" in flags

        # All flags should be boolean
        for flag_value in flags.values():
            assert isinstance(flag_value, bool)

    @pytest.mark.asyncio
    async def test_generate_compliance_report(self, audit_manager):
        """Test compliance report generation."""
        # Create some audit records
        await audit_manager.create_experimental_design_audit(
            decision_id="decision_001",
            design_methodology="factorial",
            ai_model_details={"model": "test"},
            input_parameters={"param": "value"},
            design_rationale="test rationale",
            alternative_designs=[],
            risk_assessment={},
            user_id="user_001",
        )

        await audit_manager.create_molecule_selection_audit(
            decision_id="mol_decision_001",
            selection_algorithm="gp",
            training_data_provenance={"source": "test"},
            selected_molecules=["CCO"],
            selection_criteria={"diversity": 0.8},
            user_id="chemist_001",
        )

        # Generate report
        report = await audit_manager.generate_compliance_report()

        assert "report_metadata" in report
        assert "audit_summary" in report
        assert "compliance_status" in report
        assert "validation_summary" in report
        assert "digital_signatures" in report
        assert "recommendations" in report

        # Check metadata
        metadata = report["report_metadata"]
        assert "generation_timestamp" in metadata
        assert "reporting_period" in metadata
        assert "compliance_standards" in metadata

        # Check audit summary
        summary = report["audit_summary"]
        assert summary["total_experimental_design_audits"] >= 1
        assert summary["total_molecule_selection_audits"] >= 1
        assert summary["total_result_validation_audits"] >= 0

        # Check compliance status has entries for each standard
        compliance_status = report["compliance_status"]
        assert "fda_cfr21" in compliance_status
        assert "glp" in compliance_status

    @pytest.mark.asyncio
    async def test_generate_compliance_report_with_date_range(self, audit_manager):
        """Test compliance report generation with date range."""
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        end_date = datetime.now(timezone.utc)

        report = await audit_manager.generate_compliance_report(
            start_date=start_date, end_date=end_date
        )

        metadata = report["report_metadata"]
        assert metadata["reporting_period"]["start"] == start_date.isoformat()
        assert metadata["reporting_period"]["end"] == end_date.isoformat()


class TestAuditTrailManagerIntegration:
    """Integration tests for audit trail manager."""

    @pytest.mark.asyncio
    async def test_complete_audit_workflow(self):
        """Test complete audit workflow from creation to validation."""
        config = AuditConfiguration(
            audit_level=AuditLevel.COMPREHENSIVE,
            compliance_standards=[ComplianceStandard.FDA_CFR21, ComplianceStandard.GLP],
            retention_period_days=2555,
            require_digital_signatures=True,
            enable_automatic_validation=False,
        )

        manager = AuditTrailManager(config)

        # Step 1: Create experimental design audit
        design_audit_id = await manager.create_experimental_design_audit(
            decision_id="decision_001",
            design_methodology="response_surface_methodology",
            ai_model_details={
                "model_name": "bayesian_optimizer",
                "version": "3.1",
                "confidence": 0.85,
            },
            input_parameters={
                "temperature": {"min": 60, "max": 120},
                "catalyst_loading": {"min": 0.01, "max": 0.1},
            },
            design_rationale="RSM selected for efficient parameter optimization with interaction effects",
            alternative_designs=[
                {
                    "design": "factorial",
                    "rejection_reason": "too_many_experiments_required",
                },
                {
                    "design": "random_sampling",
                    "rejection_reason": "poor_convergence_properties",
                },
            ],
            risk_assessment={
                "technical_risk": "low",
                "safety_risk": "minimal",
                "cost_risk": "acceptable",
            },
            user_id="researcher_001",
            experiment_id="exp_001",
        )

        # Step 2: Create molecule selection audit
        molecule_audit_id = await manager.create_molecule_selection_audit(
            decision_id="mol_decision_001",
            selection_algorithm="gaussian_process_with_acquisition",
            training_data_provenance={
                "dataset": "chembl_30_filtered",
                "preprocessing": "rdkit_standardization",
                "validation": "temporal_split",
            },
            selected_molecules=["CCO", "CCC", "C1CCCCC1"],
            selection_criteria={
                "diversity": 0.85,
                "novelty": 0.75,
                "synthesizability": 0.90,
            },
            user_id="chemist_002",
        )

        # Step 3: Create result validation audit
        validation_audit_id = await manager.create_result_validation_audit(
            experiment_id="exp_001",
            validation_methodology="statistical_analysis_with_qc_controls",
            experimental_results={"yield": 0.82, "purity": 0.96, "selectivity": 0.91},
            quality_metrics={"r_squared": 0.97, "rmse": 0.025, "cv_percent": 1.8},
            user_id="analyst_003",
        )

        # Step 4: Add digital signatures
        design_sig_id = await manager.add_digital_signature(
            audit_id=design_audit_id,
            audit_type="experimental_design",
            signer_id="supervisor_001",
            signer_role="study_director",
            signature_data="digital_signature_design_approval",
        )

        molecule_sig_id = await manager.add_digital_signature(
            audit_id=molecule_audit_id,
            audit_type="molecule_selection",
            signer_id="senior_chemist_001",
            signer_role="principal_investigator",
            signature_data="digital_signature_molecule_approval",
        )

        validation_sig_id = await manager.add_digital_signature(
            audit_id=validation_audit_id,
            audit_type="result_validation",
            signer_id="qa_manager_001",
            signer_role="quality_assurance",
            signature_data="digital_signature_validation_approval",
        )

        # Step 5: Validate all audit records
        design_validation = await manager.validate_audit_record(
            design_audit_id, "experimental_design"
        )

        molecule_validation = await manager.validate_audit_record(
            molecule_audit_id, "molecule_selection"
        )

        validation_validation = await manager.validate_audit_record(
            validation_audit_id, "result_validation"
        )

        # Step 6: Generate comprehensive compliance report
        compliance_report = await manager.generate_compliance_report()

        # Verify complete workflow
        assert len(manager.experimental_design_audits) == 1
        assert len(manager.molecule_selection_audits) == 1
        assert len(manager.result_validation_audits) == 1

        # Verify signatures were added
        design_audit = manager.experimental_design_audits[design_audit_id]
        molecule_audit = manager.molecule_selection_audits[molecule_audit_id]
        validation_audit = manager.result_validation_audits[validation_audit_id]

        assert len(design_audit.digital_signatures) == 1
        assert len(molecule_audit.digital_signatures) == 1
        assert len(validation_audit.digital_signatures) == 1

        # Verify validations completed
        assert design_validation["audit_id"] == design_audit_id
        assert molecule_validation["audit_id"] == molecule_audit_id
        assert validation_validation["audit_id"] == validation_audit_id

        # Verify compliance report
        assert (
            compliance_report["audit_summary"]["total_experimental_design_audits"] == 1
        )
        assert (
            compliance_report["audit_summary"]["total_molecule_selection_audits"] == 1
        )
        assert compliance_report["audit_summary"]["total_result_validation_audits"] == 1

        # All compliance standards should be present
        compliance_standards = compliance_report["report_metadata"][
            "compliance_standards"
        ]
        assert "fda_cfr21" in compliance_standards
        assert "glp" in compliance_standards

    @pytest.mark.asyncio
    async def test_parallel_audit_creation(self):
        """Test creating multiple audit records in parallel."""
        config = AuditConfiguration(
            audit_level=AuditLevel.DETAILED,
            compliance_standards=[ComplianceStandard.FDA_CFR21],
            retention_period_days=365,
            require_digital_signatures=False,
            enable_automatic_validation=False,
        )

        manager = AuditTrailManager(config)

        # Create multiple audits in parallel
        audit_tasks = []

        for i in range(5):
            task = manager.create_experimental_design_audit(
                decision_id=f"decision_{i:03d}",
                design_methodology="factorial",
                ai_model_details={"model": f"model_{i}", "version": "1.0"},
                input_parameters={"param": f"value_{i}"},
                design_rationale=f"Rationale for experiment {i}",
                alternative_designs=[],
                risk_assessment={"risk": "low"},
                user_id=f"user_{i:03d}",
                experiment_id=f"exp_{i:03d}",
            )
            audit_tasks.append(task)

        audit_ids = await asyncio.gather(*audit_tasks)

        # Verify all audits were created
        assert len(audit_ids) == 5
        assert len(manager.experimental_design_audits) == 5

        # Verify unique audit IDs
        assert len(set(audit_ids)) == 5

        # Verify all audits have correct data
        for i, audit_id in enumerate(audit_ids):
            audit = manager.experimental_design_audits[audit_id]
            assert audit.decision_id == f"decision_{i:03d}"
            assert audit.experiment_id == f"exp_{i:03d}"
            assert audit.metadata.creator_id == f"user_{i:03d}"
