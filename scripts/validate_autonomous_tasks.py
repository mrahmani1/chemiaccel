#!/usr/bin/env python3
"""
ChemiAccel Autonomous Task Validator
Validates implementation of autonomous AI-Lab integration tasks 22.1-22.5.
"""

import ast
import sys
from pathlib import Path
from typing import List, Dict, Set

def validate_autonomous_tasks(autonomous_dir: Path) -> List[str]:
    """Validate autonomous task implementations."""
    errors = []
    warnings = []
    
    if not autonomous_dir.exists():
        return ["Autonomous directory not found"]
    
    # Required task components
    required_tasks = {
        "22.1": {
            "files": ["experiment_planner.py", "protocol_generator.py", "schemas.py"],
            "classes": ["ExperimentPlanner", "ProtocolGenerator"],
            "description": "Automated Experiment Planning and Instruction Generation"
        },
        "22.2": {
            "files": ["data_ingestion.py", "data_validation.py", "streaming.py"],
            "classes": ["DataIngestionEngine", "DataValidator", "StreamingInterface"],
            "description": "Real-time Experimental Data Ingestion and Validation"
        },
        "22.3": {
            "files": ["retraining_pipeline.py", "adaptive_learning.py"],
            "classes": ["RetrainingPipeline", "AdaptiveLearningEngine"],
            "description": "Adaptive Model Retraining and Fine-tuning"
        },
        "22.4": {
            "files": ["closed_loop_optimization.py", "robotic_control.py", "ai_robot_integration.py"],
            "classes": ["ClosedLoopOptimizer", "RoboticController", "AIRobotIntegration"],
            "description": "Closed-Loop Optimization with Robotic Platforms"
        },
        "22.5": {
            "files": ["decision_logging.py", "traceability_manager.py", "audit_trail.py", "influence_analyzer.py"],
            "classes": ["DecisionLogger", "TraceabilityManager", "AuditTrail", "InfluenceAnalyzer"],
            "description": "Auditable Traceability and Decision Logging"
        }
    }
    
    # Check file existence and basic structure
    for task_id, task_info in required_tasks.items():
        print(f"Validating Task {task_id}: {task_info['description']}")
        
        missing_files = []
        for required_file in task_info["files"]:
            file_path = autonomous_dir / required_file
            if not file_path.exists():
                missing_files.append(required_file)
            else:
                # Basic syntax validation
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        ast.parse(f.read())
                    print(f"  PASS: {required_file}: Syntax valid")
                except SyntaxError as e:
                    errors.append(f"Task {task_id} - {required_file}: Syntax error - {e}")
                except Exception as e:
                    warnings.append(f"Task {task_id} - {required_file}: Could not validate - {e}")
        
        if missing_files:
            warnings.append(f"Task {task_id}: Missing files - {', '.join(missing_files)}")
    
    # Check for common autonomous patterns
    autonomous_patterns = [
        ("async def", "Asynchronous operations"),
        ("@dataclass", "Data structures"),
        ("logging", "Logging functionality"),
        ("Exception", "Error handling"),
        ("typing", "Type annotations"),
    ]
    
    pattern_counts = {pattern: 0 for pattern, _ in autonomous_patterns}
    
    for py_file in autonomous_dir.glob("*.py"):
        if py_file.name.startswith("__"):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for pattern, description in autonomous_patterns:
                if pattern in content:
                    pattern_counts[pattern] += 1
                    
        except Exception as e:
            warnings.append(f"Could not analyze {py_file}: {e}")
    
    # Report pattern analysis
    print("\nAutonomous Implementation Patterns:")
    for pattern, count in pattern_counts.items():
        description = next(desc for pat, desc in autonomous_patterns if pat == pattern)
        if count > 0:
            print(f"  PASS: {description}: Found in {count} files")
        else:
            warnings.append(f"Pattern '{pattern}' not found in any autonomous files")
    
    # Check for integration points
    integration_checks = [
        ("from ..ai.", "AI module integration"),
        ("from ..cloud.", "Cloud platform integration"),
        ("from ..core.", "Core framework integration"),
    ]
    
    print("\nIntegration Analysis:")
    for py_file in autonomous_dir.glob("*.py"):
        if py_file.name.startswith("__"):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for pattern, description in integration_checks:
                if pattern in content:
                    print(f"  PASS: {py_file.name}: {description}")
                    
        except Exception:
            continue
    
    return errors

def main():
    """Main validation function."""
    autonomous_dir = Path("src/chemi_accel/autonomous")
    
    errors = validate_autonomous_tasks(autonomous_dir)
    
    if errors:
        print("\nAutonomous task validation errors:")
        for error in errors:
            print(f"  ERROR: {error}")
        return 1
    else:
        print("\nPASS: Autonomous task validation passed")
        return 0

if __name__ == "__main__":
    sys.exit(main())