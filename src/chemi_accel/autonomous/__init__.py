"""
Autonomous Laboratory Integration Module

This module provides comprehensive functionality for autonomous laboratory operations
including experiment planning, data ingestion, validation, streaming, and error handling.
"""

from datetime import datetime

# Adaptive Learning and Model Retraining
from .adaptive_learning import (
    AdaptiveLearningManager,
    DataBalancer,
    DataBalancingStrategy,
    ModelRetrainer,
    ModelSnapshot,
    PerformanceMonitor,
    RetrainingConfig,
    RetrainingStrategy,
    TrainingData,
    create_continual_learning_manager,
    create_default_adaptive_learning_manager,
    create_production_adaptive_manager,
)

# AI-Robot Integration
from .ai_robot_integration import (
    ActiveLearningDecisionEngine,
    AIDecision,
    AIDecisionEngine,
    AIDecisionType,
    AIRobotIntegrationManager,
    ExperimentalDesign,
    InverseDesignDecisionEngine,
    SynthesisComplexity,
    SynthesisTask,
    create_active_learning_robot_system,
    create_comprehensive_ai_robot_system,
    create_inverse_design_robot_system,
)
from .audit_trail import (
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

# Closed-Loop Optimization
from .closed_loop_optimization import (
    AcquisitionFunction,
    ActiveLearningAcquisition,
    ActiveLearningQuery,
    ClosedLoopOptimizer,
    ClosedLoopStrategy,
    ConvergenceCriteria,
    ExpectedImprovement,
    ExperimentResult,
    OptimizationConfig,
    UpperConfidenceBound,
    create_active_learning_optimizer,
    create_bayesian_optimizer,
    create_multi_objective_optimizer,
)

# Data Ingestion and Parsing
from .data_ingestion import (
    AssayDataParser,
    DataIngestionAPI,
    DataIngestionManager,
    DataParser,
    ExperimentalDataPoint,
    FileWatcherIngestion,
    InstrumentConnection,
    LCMSDataParser,
    NMRDataParser,
    RESTAPIIngestion,
    create_default_parsers,
    create_sample_connections,
)

# Data Validation and Quality Control
from .data_validation import (
    DataValidationRule,
    DataValidator,
    InstrumentSpecificValidationRule,
    QualityControlRule,
    RangeValidationRule,
    SchemaValidationRule,
    StatisticalValidationRule,
    TimestampValidationRule,
    ValidationResult,
    create_assay_validator,
    create_lcms_validator,
    create_nmr_validator,
)

# Auditable Traceability and Decision Logging
from .decision_logging import (
    AuditEvent,
    DecisionContext,
    DecisionLogger,
    DecisionRecord,
    DecisionType,
    ExperimentalDesignDecision,
    ExperimentLifecycleRecord,
    ExperimentPhase,
    InfluenceRelationship,
    InfluenceType,
    MoleculeSelectionDecision,
    SynthesisRouteDecision,
)

# Error Handling and Recovery
from .error_handling import (
    CircuitBreakerErrorHandler,
    ConnectionErrorHandler,
    DataValidationErrorHandler,
    ErrorHandler,
    ErrorRecord,
    ErrorRecoveryManager,
    ErrorSeverity,
    FileSystemErrorHandler,
    InstrumentErrorHandler,
    RecoveryAction,
    RecoveryStrategy,
    create_instrument_error_manager,
    create_validation_focused_manager,
)

# Experiment Planning and Instruction Generation
from .experiment_planner import (
    CharacterizationInstruction,
    ExperimentPlanner,
    InstructionSet,
    ProtocolStep,
    SynthesisInstruction,
)
from .influence_analyzer import (
    CorrelationType,
    FeedbackLoop,
    InfluenceAnalyzer,
    InfluenceMetric,
    InfluenceNetwork,
    InfluenceStrength,
    ResultCorrelation,
)
from .protocol_generator import (
    CustomProtocol,
    CustomProtocolAdapter,
    HamiltonAdapter,
    ProtocolGenerator,
    ProtocolTemplate,
    TecanAdapter,
    create_hamilton_protocol_template,
    create_tecan_protocol_template,
)

# Retraining Pipeline Orchestration
from .retraining_pipeline import (
    PipelineExecutor,
    PipelineJob,
    PipelineSchedule,
    PipelineScheduler,
    PipelineStatus,
    ResourceManager,
    ResourceRequirement,
    ResourceType,
    RetrainingPipelineManager,
    create_cloud_pipeline_manager,
    create_default_pipeline_manager,
    create_production_pipeline_manager,
)

# Robotic Control and Integration
from .robotic_control import (
    CommunicationProtocol,
    ExperimentalAction,
    ExperimentPlan,
    ExperimentStatus,
    HTTPRobotProtocol,
    MQTTRobotProtocol,
    OptimizationObjective,
    OptimizationState,
    RobotCapability,
    RobotFeedback,
    RobotManager,
    RobotStatus,
    create_http_hamilton_robot,
    create_mqtt_synthesis_robot,
)
from .schemas import (
    CHARACTERIZATION_SCHEMA,
    LIQUID_HANDLER_SCHEMA,
    REACTION_CONDITIONS_SCHEMA,
    SYNTHESIS_SCHEMA,
    SchemaValidator,
)

# Real-time Streaming and Event Processing
from .streaming import (
    AggregationProcessor,
    AlertProcessor,
    DataValidationProcessor,
    RedisStreamer,
    StreamEvent,
    StreamingManager,
    StreamProcessor,
    StreamSubscription,
    WebSocketStreamer,
    create_default_streaming_system,
)
from .traceability_manager import (
    TraceabilityConfig,
    TraceabilityManager,
    TraceabilityMetrics,
)

__all__ = [
    # Experiment Planning
    "ExperimentPlanner",
    "SynthesisInstruction",
    "CharacterizationInstruction",
    "InstructionSet",
    "ProtocolStep",
    # Protocol Generation
    "ProtocolGenerator",
    "ProtocolTemplate",
    "CustomProtocol",
    "TecanAdapter",
    "HamiltonAdapter",
    "CustomProtocolAdapter",
    "create_tecan_protocol_template",
    "create_hamilton_protocol_template",
    # Schema Validation
    "SchemaValidator",
    "LIQUID_HANDLER_SCHEMA",
    "SYNTHESIS_SCHEMA",
    "CHARACTERIZATION_SCHEMA",
    "REACTION_CONDITIONS_SCHEMA",
    # Data Ingestion
    "ExperimentalDataPoint",
    "InstrumentConnection",
    "DataIngestionAPI",
    "RESTAPIIngestion",
    "FileWatcherIngestion",
    "DataParser",
    "NMRDataParser",
    "LCMSDataParser",
    "AssayDataParser",
    "DataIngestionManager",
    "create_default_parsers",
    "create_sample_connections",
    # Data Validation
    "ValidationResult",
    "QualityControlRule",
    "DataValidationRule",
    "SchemaValidationRule",
    "RangeValidationRule",
    "StatisticalValidationRule",
    "TimestampValidationRule",
    "InstrumentSpecificValidationRule",
    "DataValidator",
    "create_nmr_validator",
    "create_lcms_validator",
    "create_assay_validator",
    # Streaming
    "StreamEvent",
    "StreamSubscription",
    "StreamProcessor",
    "DataValidationProcessor",
    "AggregationProcessor",
    "AlertProcessor",
    "WebSocketStreamer",
    "RedisStreamer",
    "StreamingManager",
    "create_default_streaming_system",
    # Error Handling
    "ErrorSeverity",
    "RecoveryAction",
    "ErrorRecord",
    "RecoveryStrategy",
    "ErrorHandler",
    "ConnectionErrorHandler",
    "DataValidationErrorHandler",
    "FileSystemErrorHandler",
    "InstrumentErrorHandler",
    "CircuitBreakerErrorHandler",
    "ErrorRecoveryManager",
    "create_instrument_error_manager",
    "create_validation_focused_manager",
    # Adaptive Learning
    "AdaptiveLearningManager",
    "ModelRetrainer",
    "PerformanceMonitor",
    "DataBalancer",
    "TrainingData",
    "ModelSnapshot",
    "RetrainingConfig",
    "RetrainingStrategy",
    "DataBalancingStrategy",
    "create_default_adaptive_learning_manager",
    "create_continual_learning_manager",
    "create_production_adaptive_manager",
    # Retraining Pipeline
    "RetrainingPipelineManager",
    "PipelineScheduler",
    "PipelineExecutor",
    "ResourceManager",
    "PipelineJob",
    "PipelineSchedule",
    "ResourceRequirement",
    "PipelineStatus",
    "ResourceType",
    "create_default_pipeline_manager",
    "create_production_pipeline_manager",
    "create_cloud_pipeline_manager",
    # Robotic Control
    "RobotStatus",
    "ExperimentStatus",
    "OptimizationObjective",
    "RobotCapability",
    "ExperimentalAction",
    "ExperimentPlan",
    "RobotFeedback",
    "OptimizationState",
    "CommunicationProtocol",
    "HTTPRobotProtocol",
    "MQTTRobotProtocol",
    "RobotManager",
    "create_http_hamilton_robot",
    "create_mqtt_synthesis_robot",
    # Closed-Loop Optimization
    "ClosedLoopStrategy",
    "ConvergenceCriteria",
    "OptimizationConfig",
    "ExperimentResult",
    "ActiveLearningQuery",
    "AcquisitionFunction",
    "ExpectedImprovement",
    "UpperConfidenceBound",
    "ActiveLearningAcquisition",
    "ClosedLoopOptimizer",
    "create_bayesian_optimizer",
    "create_active_learning_optimizer",
    "create_multi_objective_optimizer",
    # AI-Robot Integration
    "AIDecisionType",
    "SynthesisComplexity",
    "AIDecision",
    "SynthesisTask",
    "ExperimentalDesign",
    "AIDecisionEngine",
    "ActiveLearningDecisionEngine",
    "InverseDesignDecisionEngine",
    "AIRobotIntegrationManager",
    "create_comprehensive_ai_robot_system",
    "create_active_learning_robot_system",
    "create_inverse_design_robot_system",
    # Auditable Traceability and Decision Logging
    "DecisionType",
    "ExperimentPhase",
    "InfluenceType",
    "DecisionContext",
    "MoleculeSelectionDecision",
    "SynthesisRouteDecision",
    "ExperimentalDesignDecision",
    "DecisionRecord",
    "ExperimentLifecycleRecord",
    "InfluenceRelationship",
    "AuditEvent",
    "DecisionLogger",
    # Traceability Management
    "TraceabilityConfig",
    "TraceabilityMetrics",
    "TraceabilityManager",
    # Audit Trail System
    "AuditLevel",
    "ComplianceStandard",
    "ValidationStatus",
    "AuditConfiguration",
    "DigitalSignature",
    "AuditMetadata",
    "ExperimentalDesignAudit",
    "MoleculeSelectionAudit",
    "ResultValidationAudit",
    "AuditTrailManager",
    # Influence Analysis
    "InfluenceStrength",
    "CorrelationType",
    "InfluenceMetric",
    "ResultCorrelation",
    "InfluenceNetwork",
    "FeedbackLoop",
    "InfluenceAnalyzer",
]

# Module metadata
__version__ = "1.0.0"
__author__ = "ChemiAccel Development Team"
__description__ = "Autonomous Laboratory Integration for ChemiAccel Platform"

# Module-level constants
DEFAULT_POLLING_INTERVAL = 30.0  # seconds
DEFAULT_VALIDATION_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_ERROR_RETRY_COUNT = 3
DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 5


# Configuration helpers
def create_autonomous_lab_system(
    instrument_connections=None,
    validation_config=None,
    streaming_config=None,
    error_handling_config=None,
    adaptive_learning_config=None,
    robotic_config=None,
    traceability_config=None,
):
    """
    Create a complete autonomous laboratory system with all components.

    Args:
        instrument_connections: List of InstrumentConnection objects
        validation_config: Configuration for data validation
        streaming_config: Configuration for real-time streaming
        error_handling_config: Configuration for error handling
        adaptive_learning_config: Configuration for adaptive learning
        robotic_config: Configuration for robotic platforms and AI integration
        traceability_config: Configuration for auditable traceability and decision logging

    Returns:
        Dictionary containing all system components
    """
    # Use defaults if not provided
    if instrument_connections is None:
        instrument_connections = create_sample_connections()

    # Create data validation system
    if validation_config is None:
        data_validator = DataValidator()
    else:
        # Create specialized validator based on config
        validator_type = validation_config.get("type", "general")
        if validator_type == "nmr":
            data_validator = create_nmr_validator()
        elif validator_type == "lcms":
            data_validator = create_lcms_validator()
        elif validator_type == "assay":
            data_validator = create_assay_validator()
        else:
            data_validator = DataValidator()

    # Create data ingestion manager
    data_ingestion_manager = DataIngestionManager(validator=data_validator)

    # Register parsers
    parsers = create_default_parsers()
    for instrument_type, parser in parsers.items():
        data_ingestion_manager.register_parser(instrument_type, parser)

    # Register instrument connections
    for connection in instrument_connections:
        data_ingestion_manager.register_connection(connection)

    # Create streaming system
    streaming_manager = create_default_streaming_system(data_ingestion_manager)

    # Enable Redis streaming if configured
    if streaming_config and streaming_config.get("redis_enabled"):
        redis_url = streaming_config.get("redis_url", "redis://localhost:6379")
        streaming_manager.enable_redis_streaming(redis_url)

    # Create error recovery system
    if error_handling_config and error_handling_config.get("instrument_focused"):
        error_manager = create_instrument_error_manager(instrument_connections)
    elif error_handling_config and error_handling_config.get("validation_focused"):
        error_manager = create_validation_focused_manager()
    else:
        error_manager = ErrorRecoveryManager()

    # Create adaptive learning system
    if adaptive_learning_config:
        learning_type = adaptive_learning_config.get("type", "default")
        if learning_type == "continual":
            adaptive_learning_manager = create_continual_learning_manager()
        elif learning_type == "production":
            adaptive_learning_manager = create_production_adaptive_manager()
        else:
            adaptive_learning_manager = create_default_adaptive_learning_manager()
    else:
        adaptive_learning_manager = create_default_adaptive_learning_manager()

    # Create retraining pipeline
    if (
        adaptive_learning_config
        and adaptive_learning_config.get("pipeline_type") == "production"
    ):
        pipeline_manager = create_production_pipeline_manager(
            adaptive_learning_manager=adaptive_learning_manager,
            data_ingestion_manager=data_ingestion_manager,
        )
    elif (
        adaptive_learning_config
        and adaptive_learning_config.get("pipeline_type") == "cloud"
    ):
        cloud_config = adaptive_learning_config.get("cloud_config", {})
        pipeline_manager = create_cloud_pipeline_manager(
            adaptive_learning_manager=adaptive_learning_manager,
            cloud_config=cloud_config,
        )
    else:
        pipeline_manager = create_default_pipeline_manager(
            adaptive_learning_manager=adaptive_learning_manager
        )

    # Create experiment planner
    experiment_planner = ExperimentPlanner()

    # Create protocol generator
    protocol_generator = ProtocolGenerator()

    # Create robotic control and AI integration system
    robot_manager = None
    closed_loop_optimizer = None
    ai_robot_integration_manager = None

    if robotic_config:
        # Create robot manager
        robot_manager = RobotManager()

        # Register robotic platforms
        robots = robotic_config.get("robots", [])
        for robot_config in robots:
            robot_type = robot_config.get("type", "http")
            if robot_type == "http_hamilton":
                robot_id, protocol, capabilities = create_http_hamilton_robot(
                    robot_config.get("base_url", "http://localhost:8080"),
                    robot_config.get("auth_token"),
                )
                robot_manager.register_robot(
                    robot_id, protocol, capabilities, robot_config.get("metadata", {})
                )
            elif robot_type == "mqtt_synthesis":
                robot_id, protocol, capabilities = create_mqtt_synthesis_robot(
                    robot_config.get("broker_host", "localhost"),
                    robot_config.get("broker_port", 1883),
                )
                robot_manager.register_robot(
                    robot_id, protocol, capabilities, robot_config.get("metadata", {})
                )

        # Create closed-loop optimizer if requested
        optimization_config = robotic_config.get("optimization")
        if optimization_config:
            opt_type = optimization_config.get("type", "bayesian")
            parameter_space = optimization_config.get(
                "parameter_space",
                {
                    "temperature": (50, 200),
                    "reaction_time": (1, 12),
                    "concentration": (0.1, 2.0),
                },
            )
            objective = getattr(
                OptimizationObjective,
                optimization_config.get("objective", "MAXIMIZE_YIELD"),
            )

            if opt_type == "bayesian":
                closed_loop_optimizer = create_bayesian_optimizer(
                    robot_manager, adaptive_learning_manager, parameter_space, objective
                )
            elif opt_type == "active_learning":
                closed_loop_optimizer = create_active_learning_optimizer(
                    robot_manager, adaptive_learning_manager, parameter_space, objective
                )
            elif opt_type == "multi_objective":
                objective_weights = optimization_config.get(
                    "objective_weights", {"yield": 0.7, "purity": 0.3}
                )
                closed_loop_optimizer = create_multi_objective_optimizer(
                    robot_manager,
                    adaptive_learning_manager,
                    parameter_space,
                    objective_weights,
                )

        # Create AI-robot integration manager
        ai_integration_type = robotic_config.get("ai_integration_type", "comprehensive")
        if ai_integration_type == "comprehensive":
            ai_robot_integration_manager = create_comprehensive_ai_robot_system(
                robot_manager, adaptive_learning_manager, closed_loop_optimizer
            )
        elif ai_integration_type == "active_learning":
            ai_robot_integration_manager = create_active_learning_robot_system(
                robot_manager, adaptive_learning_manager
            )
        elif ai_integration_type == "inverse_design":
            ai_robot_integration_manager = create_inverse_design_robot_system(
                robot_manager, adaptive_learning_manager
            )

    # Create traceability and audit system
    traceability_manager = None
    audit_trail_manager = None
    influence_analyzer = None

    if traceability_config:
        # Create traceability manager
        if traceability_config.get("enable_traceability", True):
            from .traceability_manager import TraceabilityConfig, TraceabilityManager

            config = TraceabilityConfig(
                enable_real_time_logging=traceability_config.get(
                    "real_time_logging", True
                ),
                enable_influence_tracking=traceability_config.get(
                    "influence_tracking", True
                ),
                enable_performance_monitoring=traceability_config.get(
                    "performance_monitoring", True
                ),
                storage_backend=traceability_config.get("storage_backend", "sqlite"),
                storage_config=traceability_config.get("storage_config", {}),
                audit_retention_days=traceability_config.get("retention_days", 365),
                automatic_correlation_threshold=traceability_config.get(
                    "correlation_threshold", 0.7
                ),
                decision_context_depth=traceability_config.get("context_depth", 3),
            )

            traceability_manager = TraceabilityManager(
                config=config,
                robot_manager=robot_manager,
                optimizer=closed_loop_optimizer,
                adaptive_manager=adaptive_learning_manager,
            )

        # Create audit trail manager if compliance is required
        if traceability_config.get("enable_audit_trail", False):
            from .audit_trail import (
                AuditConfiguration,
                AuditLevel,
                AuditTrailManager,
                ComplianceStandard,
            )

            audit_level = getattr(
                AuditLevel, traceability_config.get("audit_level", "DETAILED")
            )
            compliance_standards = [
                getattr(ComplianceStandard, std)
                for std in traceability_config.get(
                    "compliance_standards", ["FDA_CFR21"]
                )
            ]

            audit_config = AuditConfiguration(
                audit_level=audit_level,
                compliance_standards=compliance_standards,
                retention_period_days=traceability_config.get(
                    "audit_retention_days", 2555
                ),
                require_digital_signatures=traceability_config.get(
                    "require_signatures", True
                ),
                enable_automatic_validation=traceability_config.get(
                    "auto_validation", True
                ),
                validation_rules=traceability_config.get("validation_rules", {}),
                audit_storage_encrypted=traceability_config.get(
                    "encrypted_storage", True
                ),
                backup_frequency_hours=traceability_config.get("backup_frequency", 24),
                access_control_enabled=traceability_config.get("access_control", True),
            )

            audit_trail_manager = AuditTrailManager(audit_config)

        # Create influence analyzer for advanced correlation analysis
        if (
            traceability_config.get("enable_influence_analysis", True)
            and traceability_manager
        ):
            from .influence_analyzer import InfluenceAnalyzer

            influence_analyzer = InfluenceAnalyzer(traceability_manager.decision_logger)

    return {
        "experiment_planner": experiment_planner,
        "protocol_generator": protocol_generator,
        "data_ingestion_manager": data_ingestion_manager,
        "data_validator": data_validator,
        "streaming_manager": streaming_manager,
        "error_manager": error_manager,
        "adaptive_learning_manager": adaptive_learning_manager,
        "pipeline_manager": pipeline_manager,
        "robot_manager": robot_manager,
        "closed_loop_optimizer": closed_loop_optimizer,
        "ai_robot_integration_manager": ai_robot_integration_manager,
        "traceability_manager": traceability_manager,
        "audit_trail_manager": audit_trail_manager,
        "influence_analyzer": influence_analyzer,
        "instrument_connections": instrument_connections,
    }


def get_system_status(system_components):
    """
    Get comprehensive status of all autonomous lab system components.

    Args:
        system_components: Dictionary from create_autonomous_lab_system()

    Returns:
        Dictionary with status information for all components
    """
    status = {}

    # Data ingestion status
    if "data_ingestion_manager" in system_components:
        ingestion_manager = system_components["data_ingestion_manager"]
        status["data_ingestion"] = {
            "registered_connections": len(ingestion_manager.connections),
            "active_apis": len(ingestion_manager.apis),
            "registered_parsers": len(ingestion_manager.parsers),
            "is_running": ingestion_manager.is_running,
        }

    # Validation status
    if "data_validator" in system_components:
        validator = system_components["data_validator"]
        validator_stats = validator.get_validation_statistics()
        status["validation"] = {
            "total_rules": validator_stats["total_rules"],
            "rule_types": validator_stats["rule_types"],
            "quality_thresholds": validator_stats["quality_thresholds"],
        }

    # Streaming status
    if "streaming_manager" in system_components:
        streaming_manager = system_components["streaming_manager"]
        streaming_status = streaming_manager.get_streaming_status()
        status["streaming"] = {
            "is_running": streaming_status["is_running"],
            "processors_count": streaming_status["processors_count"],
            "websocket_clients": streaming_status["websocket_clients"],
            "redis_enabled": streaming_status["redis_enabled"],
        }

    # Error handling status
    if "error_manager" in system_components:
        error_manager = system_components["error_manager"]
        error_stats = error_manager.get_error_statistics()
        status["error_handling"] = {
            "total_errors": error_stats["total_errors"],
            "recent_errors_1h": error_stats["recent_errors_1h"],
            "recovery_success_rate": error_stats["recovery_success_rate"],
            "handlers_count": error_stats["handlers_count"],
        }

    # Adaptive learning status
    if "adaptive_learning_manager" in system_components:
        adaptive_manager = system_components["adaptive_learning_manager"]
        adaptive_status = adaptive_manager.get_system_status()
        status["adaptive_learning"] = {
            "is_running": adaptive_status["is_running"],
            "retraining_in_progress": adaptive_status["retraining_in_progress"],
            "strategy": adaptive_status["config"]["strategy"],
            "balancing_strategy": adaptive_status["config"]["balancing_strategy"],
            "new_data_buffer_size": adaptive_status["new_data_buffer_size"],
            "total_new_samples": adaptive_status["total_new_samples"],
        }

    # Pipeline management status
    if "pipeline_manager" in system_components:
        pipeline_manager = system_components["pipeline_manager"]
        pipeline_status = pipeline_manager.get_pipeline_status()
        status["pipeline_management"] = {
            "is_running": pipeline_status["is_running"],
            "registered_models": pipeline_status["registered_models"],
            "resource_utilization": pipeline_status["resource_utilization"],
            "active_schedules": pipeline_status["scheduler_status"]["active_schedules"],
            "running_jobs": pipeline_status["executor_status"]["running_jobs_count"],
            "completed_jobs": pipeline_status["executor_status"][
                "completed_jobs_count"
            ],
        }

    # Traceability and audit status
    if "traceability_manager" in system_components:
        traceability_manager = system_components["traceability_manager"]
        if traceability_manager:
            traceability_summary = asyncio.run(
                traceability_manager.get_traceability_summary()
            )
            status["traceability"] = {
                "is_enabled": True,
                "total_decisions_logged": traceability_summary["system_status"][
                    "total_decisions_logged"
                ],
                "total_experiments_tracked": traceability_summary["system_status"][
                    "total_experiments_tracked"
                ],
                "total_influence_relationships": traceability_summary["system_status"][
                    "total_influence_relationships"
                ],
                "active_sessions": traceability_summary["system_status"][
                    "active_sessions"
                ],
                "experiment_success_rate": traceability_summary["performance_metrics"][
                    "experiment_success_rate"
                ],
                "molecules_with_history": traceability_summary["traceability_coverage"][
                    "molecules_with_history"
                ],
            }

    if "audit_trail_manager" in system_components:
        audit_manager = system_components["audit_trail_manager"]
        if audit_manager:
            status["audit_trail"] = {
                "is_enabled": True,
                "compliance_standards": [
                    std.value for std in audit_manager.config.compliance_standards
                ],
                "audit_level": audit_manager.config.audit_level.value,
                "digital_signatures_required": audit_manager.config.require_digital_signatures,
                "automatic_validation": audit_manager.config.enable_automatic_validation,
                "total_experimental_design_audits": len(
                    audit_manager.experimental_design_audits
                ),
                "total_molecule_selection_audits": len(
                    audit_manager.molecule_selection_audits
                ),
                "total_result_validation_audits": len(
                    audit_manager.result_validation_audits
                ),
                "pending_validations": len(audit_manager.validation_queue),
            }

    if "influence_analyzer" in system_components:
        influence_analyzer = system_components["influence_analyzer"]
        if influence_analyzer:
            influence_summary = asyncio.run(influence_analyzer.get_influence_summary())
            status["influence_analysis"] = {
                "is_enabled": True,
                "total_correlations_analyzed": influence_summary[
                    "total_correlations_analyzed"
                ],
                "active_influence_networks": influence_summary[
                    "active_influence_networks"
                ],
                "identified_feedback_loops": influence_summary[
                    "identified_feedback_loops"
                ],
                "influence_strength_distribution": influence_summary[
                    "influence_strength_distribution"
                ],
                "correlation_type_distribution": influence_summary[
                    "correlation_type_distribution"
                ],
            }

    # System overview
    status["system_overview"] = {
        "components_active": len(
            [k for k, v in status.items() if k != "system_overview"]
        ),
        "timestamp": datetime.now().isoformat(),
        "version": __version__,
        "has_adaptive_learning": "adaptive_learning_manager" in system_components,
        "has_pipeline_management": "pipeline_manager" in system_components,
        "has_traceability": "traceability_manager" in system_components
        and system_components["traceability_manager"] is not None,
        "has_audit_trail": "audit_trail_manager" in system_components
        and system_components["audit_trail_manager"] is not None,
        "has_influence_analysis": "influence_analyzer" in system_components
        and system_components["influence_analyzer"] is not None,
    }

    return status
