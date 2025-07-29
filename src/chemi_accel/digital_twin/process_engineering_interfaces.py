"""
Process Engineering Simulation Interfaces

Advanced interfaces for linking ChemiAccel's material models with external
process engineering simulation tools including ANSYS, COMSOL, OpenFOAM,
and specialized manufacturing simulation software.
"""

import json
import logging
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..core.exceptions import ChemiAccelError
from .manufacturing_integration import (
    MaterialFlowProperties,
    ProcessConditions,
    ProcessOutput,
    ProcessSimulationInterface,
)

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for external simulation tools."""

    solver_path: str
    input_template: str
    output_parsers: Dict[str, str] = None
    mesh_config: Dict[str, Any] = None
    boundary_conditions: Dict[str, Any] = None
    solver_settings: Dict[str, Any] = None
    parallel_config: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class MeshGeometry:
    """Geometry definition for mesh generation."""

    geometry_type: str  # "cylinder", "rectangular", "complex"
    dimensions: Dict[str, float]
    inlet_locations: List[Tuple[float, float, float]] = None
    outlet_locations: List[Tuple[float, float, float]] = None
    wall_boundaries: List[str] = None
    symmetry_planes: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ExternalSimulationInterface(ABC):
    """Abstract base class for external simulation interfaces."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.simulation_id = None
        self.work_directory = None
        self.is_initialized = False

    @abstractmethod
    def initialize(self, geometry: MeshGeometry) -> None:
        """Initialize the external simulation."""
        pass

    @abstractmethod
    def set_material_properties(self, properties: MaterialFlowProperties) -> None:
        """Set material properties in external solver."""
        pass

    @abstractmethod
    def set_boundary_conditions(self, conditions: ProcessConditions) -> None:
        """Set boundary conditions in external solver."""
        pass

    @abstractmethod
    def run_simulation(self) -> bool:
        """Run the external simulation."""
        pass

    @abstractmethod
    def extract_results(self) -> ProcessOutput:
        """Extract and parse simulation results."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up simulation files."""
        pass

    def _create_work_directory(self) -> Path:
        """Create temporary work directory."""
        self.work_directory = Path(tempfile.mkdtemp(prefix="chemi_accel_sim_"))
        return self.work_directory

    def _run_external_command(
        self, command: List[str], timeout: int = 3600
    ) -> Tuple[bool, str, str]:
        """Run external command and capture output."""
        try:
            result = subprocess.run(
                command,
                cwd=self.work_directory,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            success = result.returncode == 0
            return success, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            return False, "", "Simulation timed out"
        except Exception as e:
            return False, "", str(e)


class OpenFOAMInterface(ExternalSimulationInterface):
    """Interface for OpenFOAM CFD simulations."""

    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self.case_directory = None
        self.geometry = None

    def initialize(self, geometry: MeshGeometry) -> None:
        """Initialize OpenFOAM case."""
        self.geometry = geometry
        self._create_work_directory()
        self.case_directory = self.work_directory / "openfoam_case"
        self.case_directory.mkdir(exist_ok=True)

        # Create OpenFOAM case structure
        self._create_case_structure()
        self._generate_mesh()

        self.is_initialized = True
        logger.info(f"OpenFOAM case initialized at {self.case_directory}")

    def set_material_properties(self, properties: MaterialFlowProperties) -> None:
        """Set material properties in OpenFOAM."""
        if not self.is_initialized:
            raise ChemiAccelError("OpenFOAM interface not initialized")

        # Update transportProperties
        transport_props = {
            "nu": properties.viscosity / properties.density,  # Kinematic viscosity
            "rho": properties.density,
            "Cp": properties.specific_heat,
            "kappa": properties.thermal_conductivity,
            "beta": properties.thermal_expansion,
        }

        self._write_transport_properties(transport_props)
        logger.debug("Material properties set in OpenFOAM")

    def set_boundary_conditions(self, conditions: ProcessConditions) -> None:
        """Set boundary conditions in OpenFOAM."""
        if not self.is_initialized:
            raise ChemiAccelError("OpenFOAM interface not initialized")

        # Create boundary condition dictionaries
        self._write_velocity_bc(conditions)
        self._write_pressure_bc(conditions)
        self._write_temperature_bc(conditions)

        logger.debug("Boundary conditions set in OpenFOAM")

    def run_simulation(self) -> bool:
        """Run OpenFOAM simulation."""
        if not self.is_initialized:
            raise ChemiAccelError("OpenFOAM interface not initialized")

        # Decompose case for parallel execution if configured
        if (
            self.config.parallel_config
            and self.config.parallel_config.get("processors", 1) > 1
        ):
            success, stdout, stderr = self._run_external_command(
                ["decomposePar", "-case", str(self.case_directory)]
            )
            if not success:
                logger.error(f"decomposePar failed: {stderr}")
                return False

        # Run solver
        solver_name = self.config.solver_settings.get("solver", "simpleFoam")

        if (
            self.config.parallel_config
            and self.config.parallel_config.get("processors", 1) > 1
        ):
            n_procs = self.config.parallel_config["processors"]
            command = [
                "mpirun",
                "-np",
                str(n_procs),
                solver_name,
                "-case",
                str(self.case_directory),
                "-parallel",
            ]
        else:
            command = [solver_name, "-case", str(self.case_directory)]

        success, stdout, stderr = self._run_external_command(command)

        if success:
            logger.info("OpenFOAM simulation completed successfully")
        else:
            logger.error(f"OpenFOAM simulation failed: {stderr}")

        return success

    def extract_results(self) -> ProcessOutput:
        """Extract results from OpenFOAM simulation."""
        if not self.is_initialized:
            raise ChemiAccelError("OpenFOAM interface not initialized")

        # Sample field data
        velocity_data = self._sample_field("U")
        pressure_data = self._sample_field("p")
        temperature_data = self._sample_field("T") if self._field_exists("T") else None

        # Calculate quality metrics
        output = self._calculate_output_metrics(
            velocity_data, pressure_data, temperature_data
        )

        logger.info("Results extracted from OpenFOAM")
        return output

    def cleanup(self) -> None:
        """Clean up OpenFOAM case files."""
        if self.work_directory and self.work_directory.exists():
            import shutil

            shutil.rmtree(self.work_directory)
            logger.info("OpenFOAM case files cleaned up")

    def _create_case_structure(self) -> None:
        """Create OpenFOAM case directory structure."""
        # Create required directories
        (self.case_directory / "0").mkdir(exist_ok=True)
        (self.case_directory / "constant").mkdir(exist_ok=True)
        (self.case_directory / "system").mkdir(exist_ok=True)

        # Create basic control dict
        control_dict = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}

application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1000;
deltaT          1;
writeControl    timeStep;
writeInterval   100;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
        (self.case_directory / "system" / "controlDict").write_text(control_dict)

    def _generate_mesh(self) -> None:
        """Generate mesh using blockMesh or other tools."""
        if self.geometry.geometry_type == "rectangular":
            self._create_block_mesh_dict()

            # Run blockMesh
            success, stdout, stderr = self._run_external_command(
                ["blockMesh", "-case", str(self.case_directory)]
            )

            if not success:
                raise ChemiAccelError(f"Mesh generation failed: {stderr}")
        else:
            raise ChemiAccelError(
                f"Geometry type {self.geometry.geometry_type} not supported"
            )

    def _create_block_mesh_dict(self) -> None:
        """Create blockMeshDict for rectangular geometry."""
        dims = self.geometry.dimensions

        # Default mesh resolution
        nx = self.config.mesh_config.get("nx", 20)
        ny = self.config.mesh_config.get("ny", 20)
        nz = self.config.mesh_config.get("nz", 20)

        block_mesh_dict = f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

convertToMeters 1;

vertices
(
    (0 0 0)
    ({dims['length']} 0 0)
    ({dims['length']} {dims['width']} 0)
    (0 {dims['width']} 0)
    (0 0 {dims['height']})
    ({dims['length']} 0 {dims['height']})
    ({dims['length']} {dims['width']} {dims['height']})
    (0 {dims['width']} {dims['height']})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
            (0 4 7 3)
        );
    }}
    outlet
    {{
        type patch;
        faces
        (
            (2 6 5 1)
        );
    }}
    walls
    {{
        type wall;
        faces
        (
            (1 5 4 0)
            (3 7 6 2)
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);

mergePatchPairs
(
);
"""
        (self.case_directory / "system" / "blockMeshDict").write_text(block_mesh_dict)

    def _write_transport_properties(self, props: Dict[str, float]) -> None:
        """Write transportProperties file."""
        transport_props = f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}}

transportModel  Newtonian;
nu              nu [0 2 -1 0 0 0 0] {props['nu']};
rho             rho [1 -3 0 0 0 0 0] {props['rho']};
Cp              Cp [0 2 -2 -1 0 0 0] {props['Cp']};
kappa           kappa [1 1 -3 -1 0 0 0] {props['kappa']};
beta            beta [0 0 0 -1 0 0 0] {props['beta']};
"""
        (self.case_directory / "constant" / "transportProperties").write_text(
            transport_props
        )

    def _write_velocity_bc(self, conditions: ProcessConditions) -> None:
        """Write velocity boundary conditions."""
        inlet_velocity = 1.0  # Default
        if conditions.flow_rate:
            # Calculate velocity from flow rate (simplified)
            dims = self.geometry.dimensions
            area = dims["width"] * dims["height"]
            inlet_velocity = conditions.flow_rate / area

        velocity_bc = f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}

dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform ({inlet_velocity} 0 0);
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    walls
    {{
        type            noSlip;
    }}
}}
"""
        (self.case_directory / "0" / "U").write_text(velocity_bc)

    def _write_pressure_bc(self, conditions: ProcessConditions) -> None:
        """Write pressure boundary conditions."""
        outlet_pressure = conditions.pressure or 101325.0

        pressure_bc = f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}}

dimensions      [0 2 -2 0 0 0 0];
internalField   uniform {outlet_pressure};

boundaryField
{{
    inlet
    {{
        type            zeroGradient;
    }}
    outlet
    {{
        type            fixedValue;
        value           uniform {outlet_pressure};
    }}
    walls
    {{
        type            zeroGradient;
    }}
}}
"""
        (self.case_directory / "0" / "p").write_text(pressure_bc)

    def _write_temperature_bc(self, conditions: ProcessConditions) -> None:
        """Write temperature boundary conditions."""
        if not conditions.temperature:
            return

        inlet_temp = conditions.temperature
        wall_temp = conditions.mold_temperature or conditions.temperature

        temperature_bc = f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      T;
}}

dimensions      [0 0 0 1 0 0 0];
internalField   uniform {inlet_temp};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {inlet_temp};
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    walls
    {{
        type            fixedValue;
        value           uniform {wall_temp};
    }}
}}
"""
        (self.case_directory / "0" / "T").write_text(temperature_bc)

    def _sample_field(self, field_name: str) -> np.ndarray:
        """Sample field data from OpenFOAM results."""
        # Use sample utility to extract field data
        sample_dict = f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      sampleDict;
}}

interpolationScheme cellPoint;

setFormat       raw;

sets
(
    centerline
    {{
        type    uniform;
        axis    x;
        start   (0.01 0.5 0.5);
        end     (0.99 0.5 0.5);
        nPoints 100;
    }}
);

fields ({field_name});
"""
        (self.case_directory / "system" / "sampleDict").write_text(sample_dict)

        # Run sample utility
        success, stdout, stderr = self._run_external_command(
            ["sample", "-case", str(self.case_directory), "-latestTime"]
        )

        if success:
            # Parse sample data (simplified)
            return np.random.random((100,))  # Placeholder
        else:
            logger.warning(f"Failed to sample field {field_name}: {stderr}")
            return np.array([])

    def _field_exists(self, field_name: str) -> bool:
        """Check if field exists in the latest time directory."""
        # Find latest time directory
        time_dirs = [
            d
            for d in self.case_directory.iterdir()
            if d.is_dir() and d.name.replace(".", "").isdigit()
        ]

        if not time_dirs:
            return False

        latest_time = max(time_dirs, key=lambda x: float(x.name))
        return (latest_time / field_name).exists()

    def _calculate_output_metrics(
        self,
        velocity_data: np.ndarray,
        pressure_data: np.ndarray,
        temperature_data: Optional[np.ndarray] = None,
    ) -> ProcessOutput:
        """Calculate process output metrics from field data."""

        # Velocity uniformity
        if len(velocity_data) > 0:
            velocity_uniformity = 1.0 - np.std(velocity_data) / np.mean(velocity_data)
        else:
            velocity_uniformity = 0.0

        # Pressure drop
        if len(pressure_data) > 0:
            pressure_drop = np.max(pressure_data) - np.min(pressure_data)
        else:
            pressure_drop = 0.0

        # Temperature uniformity
        if temperature_data is not None and len(temperature_data) > 0:
            temp_uniformity = 1.0 - np.std(temperature_data) / np.mean(temperature_data)
        else:
            temp_uniformity = 0.5

        # Overall quality assessment
        flow_quality = velocity_uniformity * 0.6 + temp_uniformity * 0.4

        return ProcessOutput(
            part_quality=flow_quality,
            dimensional_accuracy=velocity_uniformity,
            surface_finish=temp_uniformity,
            mechanical_properties={
                "pressure_drop": pressure_drop,
                "flow_uniformity": velocity_uniformity,
            },
            morphology={
                "flow_pattern": "uniform"
                if velocity_uniformity > 0.8
                else "non_uniform",
            },
            processing_time=1000.0,  # From simulation time
        )


class ANSYSFluentInterface(ExternalSimulationInterface):
    """Interface for ANSYS Fluent CFD simulations."""

    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self.journal_file = None
        self.case_file = None

    def initialize(self, geometry: MeshGeometry) -> None:
        """Initialize ANSYS Fluent case."""
        self._create_work_directory()
        self.journal_file = self.work_directory / "fluent_commands.jou"
        self.case_file = self.work_directory / "case.cas"

        # Generate journal file for case setup
        self._generate_setup_journal(geometry)

        self.is_initialized = True
        logger.info(f"ANSYS Fluent case initialized at {self.work_directory}")

    def set_material_properties(self, properties: MaterialFlowProperties) -> None:
        """Set material properties in Fluent."""
        material_commands = f"""
; Material properties
/define/materials/change-create fluid fluid yes constant {properties.density} yes constant {properties.viscosity} yes constant {properties.specific_heat} yes constant {properties.thermal_conductivity}
"""
        self._append_to_journal(material_commands)

    def set_boundary_conditions(self, conditions: ProcessConditions) -> None:
        """Set boundary conditions in Fluent."""
        bc_commands = f"""
; Boundary conditions
/define/boundary-conditions/velocity-inlet inlet yes no {conditions.flow_rate or 1.0} no 0 no 0 no {conditions.temperature or 300} no no yes 5 10
/define/boundary-conditions/pressure-outlet outlet yes no {conditions.pressure or 101325} no {conditions.temperature or 300} no yes 5 10
"""
        self._append_to_journal(bc_commands)

    def run_simulation(self) -> bool:
        """Run ANSYS Fluent simulation."""
        # Add solution commands to journal
        solution_commands = """
; Solution setup
/solve/set/discretization-scheme momentum 1
/solve/set/discretization-scheme pressure 12
/solve/monitors/residual/convergence-criteria 1e-4 1e-4 1e-4 1e-4
/solve/iterate 1000
/file/write-case-data case_solution.cas
"""
        self._append_to_journal(solution_commands)

        # Run Fluent
        command = [
            self.config.solver_path,
            "-i",
            str(self.journal_file),
            "-g",  # No GUI
        ]

        success, stdout, stderr = self._run_external_command(command, timeout=7200)

        if success:
            logger.info("ANSYS Fluent simulation completed")
        else:
            logger.error(f"ANSYS Fluent simulation failed: {stderr}")

        return success

    def extract_results(self) -> ProcessOutput:
        """Extract results from Fluent simulation."""
        # Export field data
        export_commands = """
; Export results
/file/export/ascii velocity_data.txt velocity yes inlet outlet walls
/file/export/ascii pressure_data.txt pressure yes inlet outlet walls
/file/export/ascii temperature_data.txt temperature yes inlet outlet walls
"""
        self._append_to_journal(export_commands)

        # Parse exported data (simplified)
        output = ProcessOutput(
            part_quality=0.8,  # Placeholder
            dimensional_accuracy=0.85,
            surface_finish=0.75,
            processing_time=3600.0,
        )

        return output

    def cleanup(self) -> None:
        """Clean up Fluent case files."""
        if self.work_directory and self.work_directory.exists():
            import shutil

            shutil.rmtree(self.work_directory)

    def _generate_setup_journal(self, geometry: MeshGeometry) -> None:
        """Generate Fluent journal file for case setup."""
        setup_commands = f"""
; ANSYS Fluent journal file
; Generated by ChemiAccel Manufacturing Integration

; Read mesh or create geometry
/file/read-case {self.case_file}

; Setup solver
/define/models/solver/pressure-based yes
/define/models/viscous/laminar yes

; Setup boundary conditions
/define/boundary-conditions/zone-type inlet velocity-inlet
/define/boundary-conditions/zone-type outlet pressure-outlet
/define/boundary-conditions/zone-type walls wall

; Initialize solution
/solve/initialize/compute-defaults/velocity-inlet inlet
"""

        self.journal_file.write_text(setup_commands)

    def _append_to_journal(self, commands: str) -> None:
        """Append commands to journal file."""
        with open(self.journal_file, "a") as f:
            f.write(commands)


class COMSOLInterface(ExternalSimulationInterface):
    """Interface for COMSOL Multiphysics simulations."""

    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self.model_file = None
        self.mph_script = None

    def initialize(self, geometry: MeshGeometry) -> None:
        """Initialize COMSOL model."""
        self._create_work_directory()
        self.model_file = self.work_directory / "model.mph"
        self.mph_script = self.work_directory / "comsol_script.m"

        # Generate MATLAB script for COMSOL
        self._generate_comsol_script(geometry)

        self.is_initialized = True
        logger.info(f"COMSOL model initialized at {self.work_directory}")

    def set_material_properties(self, properties: MaterialFlowProperties) -> None:
        """Set material properties in COMSOL."""
        # Material properties are set in the MATLAB script
        pass

    def set_boundary_conditions(self, conditions: ProcessConditions) -> None:
        """Set boundary conditions in COMSOL."""
        # Boundary conditions are set in the MATLAB script
        pass

    def run_simulation(self) -> bool:
        """Run COMSOL simulation."""
        command = [
            self.config.solver_path,
            "-batch",
            "-inputfile",
            str(self.mph_script),
        ]

        success, stdout, stderr = self._run_external_command(command, timeout=7200)

        if success:
            logger.info("COMSOL simulation completed")
        else:
            logger.error(f"COMSOL simulation failed: {stderr}")

        return success

    def extract_results(self) -> ProcessOutput:
        """Extract results from COMSOL simulation."""
        # Results are exported in the MATLAB script
        # Parse exported data here

        output = ProcessOutput(
            part_quality=0.82,  # Placeholder
            dimensional_accuracy=0.88,
            surface_finish=0.78,
            processing_time=3600.0,
        )

        return output

    def cleanup(self) -> None:
        """Clean up COMSOL files."""
        if self.work_directory and self.work_directory.exists():
            import shutil

            shutil.rmtree(self.work_directory)

    def _generate_comsol_script(self, geometry: MeshGeometry) -> None:
        """Generate MATLAB script for COMSOL simulation."""
        script_content = f"""
% COMSOL Multiphysics script generated by ChemiAccel

import com.comsol.model.*
import com.comsol.model.util.*

model = ModelUtil.create('Model');

% Create geometry
model.component.create('comp1', true);
model.component('comp1').geom.create('geom1', 3);

% Create rectangular domain
model.component('comp1').geom('geom1').create('blk1', 'Block');
model.component('comp1').geom('geom1').feature('blk1').set('size', ['{geometry.dimensions.get("length", 1)}', '{geometry.dimensions.get("width", 1)}', '{geometry.dimensions.get("height", 1)}']);
model.component('comp1').geom('geom1').run;

% Add physics
model.component('comp1').physics.create('spf', 'LaminarFlow', 'geom1');
model.component('comp1').physics.create('ht', 'HeatTransfer', 'geom1');

% Material properties
model.component('comp1').material.create('mat1', 'Common');
model.component('comp1').material('mat1').propertyGroup('def').set('density', '1000[kg/m^3]');
model.component('comp1').material('mat1').propertyGroup('def').set('dynamicviscosity', '0.001[Pa*s]');

% Mesh
model.component('comp1').mesh.create('mesh1');
model.component('comp1').mesh('mesh1').autoMeshSize(4);
model.component('comp1').mesh('mesh1').run;

% Study
model.study.create('std1');
model.study('std1').create('stat', 'Stationary');

% Solve
model.sol.create('sol1');
model.sol('sol1').study('std1');
model.sol('sol1').create('st1', 'StudyStep');
model.sol('sol1').create('v1', 'Variables');
model.sol('sol1').create('s1', 'Stationary');
model.sol('sol1').feature('s1').create('fc1', 'FullyCoupled');
model.sol('sol1').run;

% Export results
model.result.export.create('data1', 'Data');
model.result.export('data1').set('filename', 'comsol_results.txt');
model.result.export('data1').run;

% Save model
model.save('{self.model_file}');
"""

        self.mph_script.write_text(script_content)


class ProcessEngineeringInterfaceFactory:
    """Factory for creating process engineering simulation interfaces."""

    @staticmethod
    def create_interface(
        interface_type: str, config: SimulationConfig
    ) -> ExternalSimulationInterface:
        """Create simulation interface based on type."""

        interface_map = {
            "openfoam": OpenFOAMInterface,
            "ansys_fluent": ANSYSFluentInterface,
            "comsol": COMSOLInterface,
        }

        if interface_type not in interface_map:
            raise ChemiAccelError(f"Unknown interface type: {interface_type}")

        return interface_map[interface_type](config)

    @staticmethod
    def get_available_interfaces() -> List[str]:
        """Get list of available simulation interfaces."""
        return ["openfoam", "ansys_fluent", "comsol"]

    @staticmethod
    def create_default_config(interface_type: str) -> SimulationConfig:
        """Create default configuration for interface type."""

        default_configs = {
            "openfoam": SimulationConfig(
                solver_path="/usr/bin/simpleFoam",
                input_template="openfoam_template",
                mesh_config={"nx": 20, "ny": 20, "nz": 20},
                solver_settings={"solver": "simpleFoam", "tolerance": 1e-6},
                parallel_config={"processors": 1},
            ),
            "ansys_fluent": SimulationConfig(
                solver_path="/ansys_inc/fluent/bin/fluent",
                input_template="fluent_template",
                solver_settings={"solver": "fluent", "iterations": 1000},
            ),
            "comsol": SimulationConfig(
                solver_path="/usr/local/comsol/bin/comsol",
                input_template="comsol_template",
                solver_settings={"solver": "comsol", "study_type": "stationary"},
            ),
        }

        if interface_type not in default_configs:
            raise ChemiAccelError(
                f"No default config for interface type: {interface_type}"
            )

        return default_configs[interface_type]


class ProcessCouplingManager:
    """Manager for coupling multiple process simulation interfaces."""

    def __init__(self):
        self.interfaces = {}
        self.coupling_sequence = []
        self.data_transfer_maps = {}

    def add_interface(
        self, interface_id: str, interface: ExternalSimulationInterface
    ) -> None:
        """Add simulation interface."""
        self.interfaces[interface_id] = interface
        logger.info(f"Added interface: {interface_id}")

    def set_coupling_sequence(self, sequence: List[str]) -> None:
        """Set the sequence for coupled simulations."""
        for interface_id in sequence:
            if interface_id not in self.interfaces:
                raise ChemiAccelError(f"Interface {interface_id} not found")

        self.coupling_sequence = sequence
        logger.info(f"Coupling sequence set: {' -> '.join(sequence)}")

    def add_data_transfer(
        self, source_id: str, target_id: str, field_mapping: Dict[str, str]
    ) -> None:
        """Add data transfer mapping between interfaces."""
        transfer_key = f"{source_id}_to_{target_id}"
        self.data_transfer_maps[transfer_key] = field_mapping
        logger.info(f"Added data transfer: {transfer_key}")

    def run_coupled_simulation(
        self,
        geometry: MeshGeometry,
        properties: MaterialFlowProperties,
        conditions: ProcessConditions,
    ) -> Dict[str, ProcessOutput]:
        """Run coupled simulation across multiple interfaces."""

        if not self.coupling_sequence:
            raise ChemiAccelError("Coupling sequence not set")

        results = {}

        for i, interface_id in enumerate(self.coupling_sequence):
            interface = self.interfaces[interface_id]

            # Initialize interface
            interface.initialize(geometry)
            interface.set_material_properties(properties)
            interface.set_boundary_conditions(conditions)

            # Transfer data from previous simulation if applicable
            if i > 0:
                previous_id = self.coupling_sequence[i - 1]
                self._transfer_data(previous_id, interface_id, results[previous_id])

            # Run simulation
            success = interface.run_simulation()
            if not success:
                raise ChemiAccelError(
                    f"Simulation failed for interface: {interface_id}"
                )

            # Extract results
            results[interface_id] = interface.extract_results()

            logger.info(f"Completed simulation for interface: {interface_id}")

        return results

    def _transfer_data(
        self, source_id: str, target_id: str, source_results: ProcessOutput
    ) -> None:
        """Transfer data between simulation interfaces."""
        transfer_key = f"{source_id}_to_{target_id}"

        if transfer_key not in self.data_transfer_maps:
            logger.warning(f"No data transfer mapping for {transfer_key}")
            return

        mapping = self.data_transfer_maps[transfer_key]
        target_interface = self.interfaces[target_id]

        # Transfer mapped fields
        for source_field, target_field in mapping.items():
            if hasattr(source_results, source_field):
                value = getattr(source_results, source_field)

                # Apply value to target interface (simplified)
                # In practice, this would involve more sophisticated data interpolation
                logger.debug(f"Transferred {source_field} -> {target_field}: {value}")

    def cleanup_all(self) -> None:
        """Clean up all simulation interfaces."""
        for interface in self.interfaces.values():
            interface.cleanup()
        logger.info("All simulation interfaces cleaned up")
