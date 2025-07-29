"""
Interactive Self-Assembly Visualization Demo.

This comprehensive demonstration showcases the full capabilities of ChemiAccel's
interactive self-assembly visualization system, including:

1. 3D Simulation Visualization
2. Real-Time Property Tracking Dashboards
3. Phase Transition Analysis and Visualization
4. Comparative Multi-Simulation Analysis
5. Performance Monitoring and Optimization
6. Data Export and Report Generation

Usage:
    python examples/interactive_visualization_demo.py --demo [all|3d|dashboard|transitions|comparative|performance]

Requirements:
    - plotly>=5.0.0
    - dash>=2.0.0
    - dash-bootstrap-components>=1.0.0
    - numpy>=1.20.0
    - pandas>=1.3.0
"""

import argparse
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
from chemi_accel.self_assembly.characterization import (
    CharacterizationResult,
    PhaseType,
    StructureMetrics,
    StructureType,
)

# ChemiAccel imports
from chemi_accel.self_assembly.interactive_visualization import (
    ColorScheme,
    DashboardConfig,
    ExportFormat,
    InteractiveSelfAssemblyVisualizer,
    PhaseTransitionAnalyzer,
    PropertyTrackingDashboard,
    VisualizationConfig,
    VisualizationMode,
    create_comparative_visualization,
    create_demo_visualization,
    export_dashboard_data,
)
from chemi_accel.self_assembly.real_time_dashboard import (
    RealTimeDashboardConfig,
    RealTimeSelfAssemblyDashboard,
    create_demo_realtime_dashboard,
)

# Check for optional dependencies
try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Some visualizations will be limited.")

try:
    import dash

    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    print("Warning: Dash not available. Interactive dashboards will be limited.")


# =============================================================================
# Demo Data Generation
# =============================================================================


def create_synthetic_micelle_formation_trajectory(n_frames=100, n_particles=500):
    """
    Create a synthetic trajectory showing micelle formation process.

    Args:
        n_frames: Number of simulation frames
        n_particles: Number of particles

    Returns:
        List of trajectory frames and characterization results
    """
    print(
        f"Generating synthetic micelle formation trajectory ({n_frames} frames, {n_particles} particles)..."
    )

    trajectory = []
    characterization_results = []

    # Simulation parameters
    box_size = 20.0
    time_step = 0.01

    # Particle composition: 30% hydrophilic, 30% hydrophobic, 40% amphiphilic
    n_hydrophilic = int(0.3 * n_particles)
    n_hydrophobic = int(0.3 * n_particles)
    n_amphiphilic = n_particles - n_hydrophilic - n_hydrophobic

    particle_types = (
        ["hydrophilic"] * n_hydrophilic
        + ["hydrophobic"] * n_hydrophobic
        + ["amphiphilic"] * n_amphiphilic
    )

    # Initial random positions
    positions = np.random.uniform(0, box_size, (n_particles, 3))

    for frame_idx in range(n_frames):
        timestamp = frame_idx * time_step

        # Simulate micelle formation over time
        formation_progress = frame_idx / n_frames

        # Create clustering effect for amphiphilic molecules
        if formation_progress > 0.2:  # Start forming micelles after 20% of simulation
            # Create 3-5 micelle centers
            n_micelles = 3 + int(2 * formation_progress)
            micelle_centers = np.random.uniform(2, box_size - 2, (n_micelles, 3))

            # Move amphiphilic particles toward micelle centers
            amphiphilic_indices = [
                i for i, ptype in enumerate(particle_types) if ptype == "amphiphilic"
            ]

            for i in amphiphilic_indices:
                # Find nearest micelle center
                distances = [
                    np.linalg.norm(positions[i] - center) for center in micelle_centers
                ]
                nearest_center_idx = np.argmin(distances)
                nearest_center = micelle_centers[nearest_center_idx]

                # Move toward center with some randomness
                direction = nearest_center - positions[i]
                direction = direction / (np.linalg.norm(direction) + 1e-8)

                step_size = (
                    0.1 * formation_progress * (1 + 0.5 * np.random.uniform(-1, 1))
                )
                positions[i] += step_size * direction

                # Add thermal motion
                positions[i] += 0.05 * np.random.uniform(-1, 1, 3)

        # Keep particles in box
        positions = np.clip(positions, 0, box_size)

        # Add random motion to hydrophilic and hydrophobic particles
        for i, ptype in enumerate(particle_types):
            if ptype in ["hydrophilic", "hydrophobic"]:
                positions[i] += 0.1 * np.random.uniform(-1, 1, 3)

        # Create particles for this frame
        particles = []
        for i in range(n_particles):
            particle = type(
                "Particle",
                (),
                {
                    "position": type(
                        "Position",
                        (),
                        {
                            "x": positions[i, 0],
                            "y": positions[i, 1],
                            "z": positions[i, 2],
                        },
                    )(),
                    "particle_type": type(
                        "ParticleType", (), {"name": particle_types[i]}
                    )(),
                    "particle_id": i,
                    "velocity": type(
                        "Velocity",
                        (),
                        {
                            "x": np.random.uniform(-0.5, 0.5),
                            "y": np.random.uniform(-0.5, 0.5),
                            "z": np.random.uniform(-0.5, 0.5),
                        },
                    )(),
                    "energy": -1.0
                    + formation_progress * (-2.0)
                    + np.random.uniform(-0.5, 0.5),
                },
            )()
            particles.append(particle)

        # Create trajectory frame
        frame = type(
            "TrajectoryFrame",
            (),
            {
                "timestamp": timestamp,
                "particles": particles,
                "box": type(
                    "Box",
                    (),
                    {"length": box_size, "width": box_size, "height": box_size},
                )(),
                "total_energy": -100
                - formation_progress * 200
                + np.random.uniform(-50, 50),
                "temperature": 300.0
                + 10 * np.sin(frame_idx / 10)
                + np.random.uniform(-5, 5),
                "pressure": 1.0
                + 0.1 * formation_progress
                + np.random.uniform(-0.05, 0.05),
            },
        )()

        trajectory.append(frame)

        # Create structure metrics
        micelle_count = max(0, int(formation_progress * 5) + np.random.randint(-1, 2))
        aggregation_number = 30 + formation_progress * 20 + np.random.uniform(-5, 5)
        cluster_stability = formation_progress * 0.8 + np.random.uniform(-0.1, 0.1)

        structure_metrics = type(
            "StructureMetrics",
            (),
            {
                "micelle_count": micelle_count,
                "vesicle_count": 0,
                "aggregation_number": aggregation_number,
                "cluster_stability": max(0, min(1, cluster_stability)),
                "cluster_compactness": 0.5 + formation_progress * 0.3,
                "total_clusters": micelle_count,
                "largest_cluster_size": int(aggregation_number * 1.2),
                "membrane_thickness": 0.0,
                "bilayer_thickness": 0.0,
            },
        )()

        # Determine dominant phase
        if formation_progress < 0.3:
            dominant_phase = PhaseType.LIQUID
        elif micelle_count > 0:
            dominant_phase = PhaseType.MICELLE
        else:
            dominant_phase = PhaseType.LIQUID

        char_result = type(
            "CharacterizationResult",
            (),
            {
                "structure_metrics": structure_metrics,
                "dominant_phase": dominant_phase,
                "phase_fractions": {
                    PhaseType.LIQUID: 1.0 - formation_progress,
                    PhaseType.MICELLE: formation_progress,
                },
                "clustering_result": None,
            },
        )()

        characterization_results.append(char_result)

    print(f"Generated trajectory with {len(trajectory)} frames")
    return trajectory, characterization_results


def create_phase_transition_trajectory(n_frames=150, n_particles=400):
    """
    Create trajectory showing phase transitions between different self-assembled structures.

    Args:
        n_frames: Number of frames
        n_particles: Number of particles

    Returns:
        Trajectory and characterization results showing phase transitions
    """
    print(
        f"Generating phase transition trajectory ({n_frames} frames, {n_particles} particles)..."
    )

    trajectory = []
    characterization_results = []

    box_size = 18.0
    time_step = 0.02

    # Phase progression: Liquid -> Micelles -> Vesicles -> Bilayer
    phase_transitions = [
        (0.0, 0.25, PhaseType.LIQUID),  # 0-25%: Liquid phase
        (0.25, 0.6, PhaseType.MICELLE),  # 25-60%: Micelle formation
        (0.6, 0.85, PhaseType.VESICLE),  # 60-85%: Vesicle formation
        (0.85, 1.0, PhaseType.BILAYER),  # 85-100%: Bilayer formation
    ]

    # Particle types: More amphiphilic for complex structures
    n_amphiphilic = int(0.6 * n_particles)
    n_hydrophilic = int(0.2 * n_particles)
    n_hydrophobic = n_particles - n_amphiphilic - n_hydrophilic

    particle_types = (
        ["amphiphilic"] * n_amphiphilic
        + ["hydrophilic"] * n_hydrophilic
        + ["hydrophobic"] * n_hydrophobic
    )

    positions = np.random.uniform(0, box_size, (n_particles, 3))

    for frame_idx in range(n_frames):
        timestamp = frame_idx * time_step
        progress = frame_idx / n_frames

        # Determine current phase
        current_phase = PhaseType.LIQUID
        for start, end, phase in phase_transitions:
            if start <= progress < end:
                current_phase = phase
                break

        # Apply phase-specific dynamics
        if current_phase == PhaseType.LIQUID:
            # Random motion
            for i in range(n_particles):
                positions[i] += 0.15 * np.random.uniform(-1, 1, 3)

        elif current_phase == PhaseType.MICELLE:
            # Form 4-6 micelles
            n_micelles = 4 + int(2 * (progress - 0.25) / 0.35)
            micelle_centers = np.random.uniform(3, box_size - 3, (n_micelles, 3))

            for i, ptype in enumerate(particle_types):
                if ptype == "amphiphilic":
                    # Move toward nearest micelle center
                    distances = [
                        np.linalg.norm(positions[i] - center)
                        for center in micelle_centers
                    ]
                    nearest_idx = np.argmin(distances)
                    center = micelle_centers[nearest_idx]

                    direction = center - positions[i]
                    direction = direction / (np.linalg.norm(direction) + 1e-8)
                    positions[i] += 0.08 * direction + 0.03 * np.random.uniform(
                        -1, 1, 3
                    )

        elif current_phase == PhaseType.VESICLE:
            # Form 2-3 vesicles (hollow spheres)
            n_vesicles = 2 + int((progress - 0.6) / 0.25)
            vesicle_centers = np.random.uniform(4, box_size - 4, (n_vesicles, 3))
            vesicle_radius = 3.0

            for i, ptype in enumerate(particle_types):
                if ptype == "amphiphilic":
                    # Move toward vesicle surface
                    distances = [
                        np.linalg.norm(positions[i] - center)
                        for center in vesicle_centers
                    ]
                    nearest_idx = np.argmin(distances)
                    center = vesicle_centers[nearest_idx]

                    to_center = center - positions[i]
                    distance = np.linalg.norm(to_center)

                    if distance > 0:
                        # Move toward target radius
                        target_pos = center + (to_center / distance) * vesicle_radius
                        direction = target_pos - positions[i]
                        positions[i] += 0.06 * direction + 0.02 * np.random.uniform(
                            -1, 1, 3
                        )

        elif current_phase == PhaseType.BILAYER:
            # Form planar bilayer structure
            bilayer_z = box_size / 2

            for i, ptype in enumerate(particle_types):
                if ptype == "amphiphilic":
                    # Move toward bilayer plane
                    target_z = bilayer_z + 0.5 * np.random.uniform(-1, 1)
                    positions[i, 2] += 0.05 * (target_z - positions[i, 2])

                    # Random motion in x-y plane
                    positions[i, :2] += 0.03 * np.random.uniform(-1, 1, 2)

        # Keep particles in simulation box
        positions = np.clip(positions, 0, box_size)

        # Create frame
        particles = []
        for i in range(n_particles):
            particle = type(
                "Particle",
                (),
                {
                    "position": type(
                        "Position",
                        (),
                        {
                            "x": positions[i, 0],
                            "y": positions[i, 1],
                            "z": positions[i, 2],
                        },
                    )(),
                    "particle_type": type(
                        "ParticleType", (), {"name": particle_types[i]}
                    )(),
                    "particle_id": i,
                    "energy": -2.0 - progress * 3.0 + np.random.uniform(-0.5, 0.5),
                },
            )()
            particles.append(particle)

        frame = type(
            "TrajectoryFrame",
            (),
            {
                "timestamp": timestamp,
                "particles": particles,
                "box": type(
                    "Box",
                    (),
                    {"length": box_size, "width": box_size, "height": box_size},
                )(),
                "total_energy": -200 - progress * 500 + np.random.uniform(-100, 100),
                "temperature": 280 + 40 * progress + np.random.uniform(-5, 5),
                "pressure": 0.8 + 0.4 * progress + np.random.uniform(-0.1, 0.1),
            },
        )()

        trajectory.append(frame)

        # Create characterization results based on phase
        if current_phase == PhaseType.LIQUID:
            micelle_count = 0
            vesicle_count = 0
            membrane_thickness = 0.0
            bilayer_thickness = 0.0
            stability = 0.1 + 0.2 * progress

        elif current_phase == PhaseType.MICELLE:
            micelle_count = n_micelles
            vesicle_count = 0
            membrane_thickness = 0.0
            bilayer_thickness = 0.0
            stability = 0.4 + 0.3 * (progress - 0.25) / 0.35

        elif current_phase == PhaseType.VESICLE:
            micelle_count = 0
            vesicle_count = n_vesicles
            membrane_thickness = 2.5 + 0.5 * np.random.uniform(-1, 1)
            bilayer_thickness = 0.0
            stability = 0.7 + 0.2 * (progress - 0.6) / 0.25

        elif current_phase == PhaseType.BILAYER:
            micelle_count = 0
            vesicle_count = 0
            membrane_thickness = 0.0
            bilayer_thickness = 4.0 + 0.5 * np.random.uniform(-1, 1)
            stability = 0.9 + 0.1 * (progress - 0.85) / 0.15

        structure_metrics = type(
            "StructureMetrics",
            (),
            {
                "micelle_count": micelle_count,
                "vesicle_count": vesicle_count,
                "membrane_thickness": membrane_thickness,
                "bilayer_thickness": bilayer_thickness,
                "cluster_stability": min(1.0, stability),
                "cluster_compactness": 0.3 + 0.5 * progress,
                "aggregation_number": 20 + 30 * progress,
                "total_clusters": micelle_count
                + vesicle_count
                + (1 if bilayer_thickness > 0 else 0),
            },
        )()

        char_result = type(
            "CharacterizationResult",
            (),
            {
                "structure_metrics": structure_metrics,
                "dominant_phase": current_phase,
                "phase_fractions": {current_phase: 1.0},
                "clustering_result": None,
            },
        )()

        characterization_results.append(char_result)

    print(f"Generated phase transition trajectory with {len(trajectory)} frames")
    return trajectory, characterization_results


# =============================================================================
# Demo Functions
# =============================================================================


def demo_3d_visualization():
    """Demonstrate 3D visualization capabilities."""
    print("\n" + "=" * 60)
    print("3D VISUALIZATION DEMO")
    print("=" * 60)

    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Skipping 3D visualization demo.")
        return

    # Create visualizer with custom configuration
    config = VisualizationConfig(
        width=1400,
        height=900,
        particle_size_scale=1.5,
        color_scheme=ColorScheme.PARTICLE_TYPE,
        show_trajectory_traces=False,
        max_particles_3d=1000,
    )

    visualizer = InteractiveSelfAssemblyVisualizer(config=config, use_gpu=False)

    print("1. Creating synthetic micelle formation trajectory...")
    trajectory, char_results = create_synthetic_micelle_formation_trajectory(
        n_frames=50, n_particles=300
    )

    print("2. Loading trajectory into visualizer...")
    visualizer.load_trajectory(trajectory, char_results)

    print("3. Creating 3D visualizations...")

    # Static 3D visualization
    print("   - Static 3D particle visualization")
    fig_static = visualizer.create_3d_visualization(
        frame_index=25,  # Middle of simulation
        mode=VisualizationMode.PARTICLES_3D,
        color_by=ColorScheme.PARTICLE_TYPE,
    )

    # Show in browser
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)

    fig_static.write_html(str(output_dir / "3d_static_visualization.html"))
    print(
        f"   - Saved static visualization: {output_dir / '3d_static_visualization.html'}"
    )

    # Animated trajectory
    print("   - Animated trajectory visualization")
    fig_animated = visualizer.animate_trajectory(
        start_frame=0, end_frame=40, step=2, mode=VisualizationMode.PARTICLES_3D
    )

    fig_animated.write_html(str(output_dir / "3d_animated_trajectory.html"))
    print(
        f"   - Saved animated visualization: {output_dir / '3d_animated_trajectory.html'}"
    )

    # Different visualization modes
    print("   - Structure evolution visualization")
    visualizer.current_frame = 35
    fig_evolution = visualizer.create_3d_visualization(
        mode=VisualizationMode.STRUCTURE_EVOLUTION, color_by=ColorScheme.CLUSTER_ID
    )
    fig_evolution.write_html(str(output_dir / "3d_structure_evolution.html"))

    # Export static images
    print("4. Exporting static images...")
    try:
        visualizer.export_visualization(
            str(output_dir / "micelle_formation.png"),
            format=ExportFormat.PNG,
            width=1200,
            height=800,
        )
        print(f"   - Exported PNG: {output_dir / 'micelle_formation.png'}")
    except Exception as e:
        print(f"   - PNG export failed: {e}")

    print("\n✓ 3D Visualization Demo Complete!")
    print(f"📁 Output files saved to: {output_dir.absolute()}")


def demo_property_dashboard():
    """Demonstrate property tracking dashboard."""
    print("\n" + "=" * 60)
    print("PROPERTY TRACKING DASHBOARD DEMO")
    print("=" * 60)

    if not DASH_AVAILABLE:
        print("Dash not available. Skipping dashboard demo.")
        return

    # Create dashboard
    config = DashboardConfig(
        update_interval=500.0,  # Faster updates for demo
        chart_height=300,
        max_data_points=500,
    )

    dashboard = PropertyTrackingDashboard(config=config, port=8052)

    print("1. Generating demonstration data...")

    # Generate multiple simulation datasets
    time_points = np.linspace(0, 100, 200)

    # Micelle formation simulation
    micelle_progress = np.clip((time_points - 20) / 30, 0, 1)
    micelle_count = (
        10 * micelle_progress
        + 2 * np.sin(time_points / 5)
        + np.random.normal(0, 0.5, len(time_points))
    )
    aggregation_number = (
        30 + 20 * micelle_progress + np.random.normal(0, 2, len(time_points))
    )
    stability = (
        micelle_progress * 0.8
        + 0.1
        + 0.1 * np.sin(time_points / 8)
        + np.random.normal(0, 0.05, len(time_points))
    )

    dashboard.add_time_series(
        name="Micelle Formation",
        timestamps=time_points,
        properties={
            "micelle_count": np.maximum(0, micelle_count),
            "aggregation_number": np.maximum(0, aggregation_number),
            "cluster_stability": np.clip(stability, 0, 1),
        },
        labels={
            "micelle_count": "Number of Micelles",
            "aggregation_number": "Aggregation Number",
            "cluster_stability": "Cluster Stability",
        },
        units={
            "micelle_count": "count",
            "aggregation_number": "molecules",
            "cluster_stability": "fraction",
        },
    )

    # Vesicle formation simulation
    vesicle_progress = np.clip((time_points - 60) / 25, 0, 1)
    vesicle_count = 3 * vesicle_progress + np.random.normal(0, 0.3, len(time_points))
    membrane_thickness = (
        2.5
        + 1.5 * vesicle_progress
        + 0.3 * np.cos(time_points / 10)
        + np.random.normal(0, 0.1, len(time_points))
    )
    curvature = (
        0.1 * vesicle_progress
        + 0.02 * np.sin(time_points / 12)
        + np.random.normal(0, 0.01, len(time_points))
    )

    dashboard.add_time_series(
        name="Vesicle Formation",
        timestamps=time_points,
        properties={
            "vesicle_count": np.maximum(0, vesicle_count),
            "membrane_thickness": np.maximum(0, membrane_thickness),
            "mean_curvature": curvature,
        },
        labels={
            "vesicle_count": "Number of Vesicles",
            "membrane_thickness": "Membrane Thickness",
            "mean_curvature": "Mean Curvature",
        },
        units={
            "vesicle_count": "count",
            "membrane_thickness": "nm",
            "mean_curvature": "1/nm",
        },
    )

    print("2. Adding phase transition events...")

    # Add synthetic phase transitions
    transition_times = [25, 45, 65, 85]
    transition_pairs = [
        (PhaseType.LIQUID, PhaseType.MICELLE),
        (PhaseType.MICELLE, PhaseType.MICELLE),  # Micelle reorganization
        (PhaseType.MICELLE, PhaseType.VESICLE),
        (PhaseType.VESICLE, PhaseType.BILAYER),
    ]

    for t, (from_phase, to_phase) in zip(transition_times, transition_pairs):
        dashboard.add_phase_transition(
            timestamp=t,
            from_phase=from_phase,
            to_phase=to_phase,
            transition_probability=0.7 + 0.2 * np.random.rand(),
            order_parameter=0.5 + 0.3 * np.random.rand(),
            critical_temperature=290 + 30 * np.random.rand(),
            transition_duration=2 + 3 * np.random.rand(),
        )

    print("3. Adding parameter sweep data...")

    # Create parameter sweep for phase diagram
    temperatures = np.linspace(280, 340, 15)
    concentrations = np.linspace(0.05, 0.5, 15)
    T, C = np.meshgrid(temperatures, concentrations, indexing="ij")

    # Synthetic phase stability map
    stability_map = (
        np.sin((T - 300) / 20) * np.cos(C * 10)
        + 0.5 * np.sin((T - 310) / 15) * np.sin(C * 8)
        + 0.3 * np.random.rand(*T.shape)
    )

    phase_map = np.zeros_like(T)
    phase_map[stability_map < -0.5] = 0  # Liquid
    phase_map[(stability_map >= -0.5) & (stability_map < 0.2)] = 1  # Micelle
    phase_map[(stability_map >= 0.2) & (stability_map < 0.7)] = 2  # Vesicle
    phase_map[stability_map >= 0.7] = 3  # Bilayer

    dashboard.add_parameter_sweep(
        parameter_names=["Temperature", "Concentration"],
        parameter_values=np.array([temperatures, concentrations]),
        property_maps={"phase_stability": stability_map, "phase_identity": phase_map},
    )

    print("4. Creating and launching dashboard...")
    dashboard.create_dashboard()

    # Export dashboard data
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)

    export_dashboard_data(
        dashboard, str(output_dir / "dashboard_data.json"), format="json"
    )
    export_dashboard_data(
        dashboard, str(output_dir / "dashboard_data.csv"), format="csv"
    )

    print(f"   - Exported dashboard data to: {output_dir}")
    print(f"   - Dashboard will be available at: http://localhost:{dashboard.port}")
    print("\n🚀 Starting dashboard server...")
    print("   Press Ctrl+C to stop the dashboard")

    try:
        # Open browser after short delay
        def open_browser():
            time.sleep(2)
            webbrowser.open(f"http://localhost:{dashboard.port}")

        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()

        dashboard.run_dashboard(debug=False)

    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Dashboard error: {e}")


def demo_phase_transition_analysis():
    """Demonstrate phase transition analysis and visualization."""
    print("\n" + "=" * 60)
    print("PHASE TRANSITION ANALYSIS DEMO")
    print("=" * 60)

    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Skipping phase transition demo.")
        return

    print("1. Creating phase transition trajectory...")
    trajectory, char_results = create_phase_transition_trajectory(
        n_frames=120, n_particles=350
    )

    print("2. Analyzing phase transitions...")
    analyzer = PhaseTransitionAnalyzer(use_gpu=False)

    transitions = analyzer.analyze_phase_transitions(
        trajectory,
        char_results,
        temperature_range=(280, 320),
        concentration_range=(0.1, 0.5),
    )

    print(f"   - Detected {len(transitions)} phase transitions")

    for i, transition in enumerate(transitions):
        print(
            f"     {i+1}. Time {transition.timestamp:.2f}: "
            f"{transition.from_phase.value} → {transition.to_phase.value} "
            f"(p={transition.transition_probability:.3f})"
        )

    print("3. Creating phase transition visualizations...")

    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)

    # Create phase diagram
    temp_range = np.linspace(280, 320, 20)
    conc_range = np.linspace(0.1, 0.5, 20)
    T, C = np.meshgrid(temp_range, conc_range, indexing="ij")

    # Synthetic phase map based on temperature and concentration
    phase_map = np.zeros_like(T)

    # Define phase boundaries
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            temp = T[i, j]
            conc = C[i, j]

            if temp < 290:
                phase_map[i, j] = 0  # Liquid
            elif temp < 300 and conc > 0.2:
                phase_map[i, j] = 1  # Micelle
            elif temp < 310 and conc > 0.15:
                phase_map[i, j] = 2  # Vesicle
            elif conc > 0.25:
                phase_map[i, j] = 3  # Bilayer
            else:
                phase_map[i, j] = 0  # Liquid

    # Add noise to make it more realistic
    phase_map += 0.3 * np.random.rand(*phase_map.shape)
    phase_map = np.clip(phase_map, 0, 3).astype(int)

    fig_phase_diagram = analyzer.create_phase_diagram(
        temp_range, conc_range, phase_map, "Temperature (K)", "Concentration"
    )

    fig_phase_diagram.write_html(str(output_dir / "phase_diagram.html"))
    print(f"   - Phase diagram saved: {output_dir / 'phase_diagram.html'}")

    # Create transition network
    fig_network = analyzer.create_transition_network()
    fig_network.write_html(str(output_dir / "transition_network.html"))
    print(f"   - Transition network saved: {output_dir / 'transition_network.html'}")

    print("4. Creating trajectory visualization with phase transitions...")

    # Create visualizer for phase transition trajectory
    config = VisualizationConfig(
        width=1200,
        height=800,
        color_scheme=ColorScheme.PARTICLE_TYPE,
        max_particles_3d=400,
    )

    visualizer = InteractiveSelfAssemblyVisualizer(config=config)
    visualizer.load_trajectory(trajectory, char_results)

    # Create visualization at key transition points
    transition_frames = [
        int(t.timestamp / 0.02) for t in transitions
    ]  # Convert time to frame

    for i, frame_idx in enumerate(transition_frames):
        if frame_idx < len(trajectory):
            fig = visualizer.create_3d_visualization(
                frame_index=frame_idx, mode=VisualizationMode.PARTICLES_3D
            )

            transition = transitions[i]
            fig.update_layout(
                title=f"Phase Transition {i+1}: {transition.from_phase.value} → {transition.to_phase.value}"
            )

            filename = f"transition_{i+1}_{transition.from_phase.value}_to_{transition.to_phase.value}.html"
            fig.write_html(str(output_dir / filename))
            print(f"   - Transition visualization {i+1} saved: {output_dir / filename}")

    print("\n✓ Phase Transition Analysis Demo Complete!")
    print(f"📁 Output files saved to: {output_dir.absolute()}")


def demo_realtime_dashboard():
    """Demonstrate real-time monitoring dashboard."""
    print("\n" + "=" * 60)
    print("REAL-TIME DASHBOARD DEMO")
    print("=" * 60)

    if not DASH_AVAILABLE:
        print("Dash not available. Skipping real-time dashboard demo.")
        return

    print("1. Creating real-time dashboard...")
    dashboard = create_demo_realtime_dashboard(port=8053)

    print("2. Setting up real-time data simulation...")
    print("   - Micelle formation metrics")
    print("   - Vesicle formation metrics")
    print("   - Performance monitoring")
    print("   - Alert system")

    print("3. Launching real-time dashboard...")
    dashboard.create_dashboard()

    print(f"   - Real-time dashboard available at: http://localhost:{dashboard.port}")
    print("   - Data updates every second")
    print("   - Performance metrics included")
    print("   - Alert monitoring active")

    print("\n🚀 Starting real-time dashboard server...")
    print("   Press Ctrl+C to stop the dashboard")

    try:
        # Open browser
        def open_browser():
            time.sleep(2)
            webbrowser.open(f"http://localhost:{dashboard.port}")

        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()

        dashboard.run_dashboard(debug=False)

    except KeyboardInterrupt:
        print("\n🛑 Real-time dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Real-time dashboard error: {e}")


def demo_comparative_analysis():
    """Demonstrate comparative analysis of multiple simulations."""
    print("\n" + "=" * 60)
    print("COMPARATIVE ANALYSIS DEMO")
    print("=" * 60)

    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Skipping comparative analysis demo.")
        return

    print("1. Creating multiple simulation trajectories...")

    # Create three different simulation scenarios
    print("   - Scenario 1: Standard micelle formation")
    traj1, char1 = create_synthetic_micelle_formation_trajectory(
        n_frames=60, n_particles=200
    )

    print("   - Scenario 2: Fast micelle formation (higher concentration)")
    traj2, char2 = create_synthetic_micelle_formation_trajectory(
        n_frames=60, n_particles=300
    )
    # Modify for faster formation
    for frame in traj2:
        frame.temperature = getattr(frame, "temperature", 300) + 10

    print("   - Scenario 3: Phase transition sequence")
    traj3, char3 = create_phase_transition_trajectory(n_frames=60, n_particles=250)

    print("2. Creating visualizers for each scenario...")

    config = VisualizationConfig(width=400, height=400, max_particles_3d=250)

    viz1 = InteractiveSelfAssemblyVisualizer(config=config)
    viz1.load_trajectory(traj1, char1)

    viz2 = InteractiveSelfAssemblyVisualizer(config=config)
    viz2.load_trajectory(traj2, char2)

    viz3 = InteractiveSelfAssemblyVisualizer(config=config)
    viz3.load_trajectory(traj3, char3)

    print("3. Creating comparative visualization...")

    fig_comparative = create_comparative_visualization(
        visualizers=[viz1, viz2, viz3],
        labels=["Standard Formation", "Fast Formation", "Phase Transitions"],
        sync_frames=True,
    )

    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)

    fig_comparative.write_html(str(output_dir / "comparative_analysis.html"))
    print(
        f"   - Comparative visualization saved: {output_dir / 'comparative_analysis.html'}"
    )

    print("4. Creating comparative property analysis...")

    # Extract properties from all scenarios
    times1 = [f.timestamp for f in traj1]
    times2 = [f.timestamp for f in traj2]
    times3 = [f.timestamp for f in traj3]

    micelles1 = [c.structure_metrics.micelle_count for c in char1]
    micelles2 = [c.structure_metrics.micelle_count for c in char2]
    micelles3 = [c.structure_metrics.micelle_count for c in char3]

    # Create comparative plots
    fig_comparison = go.Figure()

    fig_comparison.add_trace(
        go.Scatter(
            x=times1,
            y=micelles1,
            mode="lines+markers",
            name="Standard Formation",
            line=dict(color="blue", width=2),
        )
    )

    fig_comparison.add_trace(
        go.Scatter(
            x=times2,
            y=micelles2,
            mode="lines+markers",
            name="Fast Formation",
            line=dict(color="red", width=2),
        )
    )

    fig_comparison.add_trace(
        go.Scatter(
            x=times3,
            y=micelles3,
            mode="lines+markers",
            name="Phase Transitions",
            line=dict(color="green", width=2),
        )
    )

    fig_comparison.update_layout(
        title="Comparative Micelle Formation Analysis",
        xaxis_title="Time (ns)",
        yaxis_title="Number of Micelles",
        width=1000,
        height=600,
    )

    fig_comparison.write_html(str(output_dir / "comparative_properties.html"))
    print(
        f"   - Property comparison saved: {output_dir / 'comparative_properties.html'}"
    )

    print("5. Generating comparative report...")

    # Create summary statistics
    final_micelles = [micelles1[-1], micelles2[-1], micelles3[-1]]
    formation_rates = [
        np.polyfit(times1, micelles1, 1)[0],
        np.polyfit(times2, micelles2, 1)[0],
        np.polyfit(times3, micelles3, 1)[0],
    ]

    report = f"""
Comparative Self-Assembly Analysis Report
========================================

Simulation Scenarios:
1. Standard Formation: {len(traj1)} frames, {len(traj1[0].particles)} particles
2. Fast Formation: {len(traj2)} frames, {len(traj2[0].particles)} particles
3. Phase Transitions: {len(traj3)} frames, {len(traj3[0].particles)} particles

Final Micelle Counts:
1. Standard: {final_micelles[0]:.1f}
2. Fast: {final_micelles[1]:.1f}
3. Transitions: {final_micelles[2]:.1f}

Formation Rates (micelles/ns):
1. Standard: {formation_rates[0]:.3f}
2. Fast: {formation_rates[1]:.3f}
3. Transitions: {formation_rates[2]:.3f}

Key Findings:
- Fast formation scenario shows {formation_rates[1]/formation_rates[0]:.1f}x faster micelle formation
- Phase transition scenario demonstrates structural evolution beyond simple micelles
- Standard formation provides stable, controlled micelle growth

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""

    with open(output_dir / "comparative_report.txt", "w") as f:
        f.write(report)

    print(f"   - Comparative report saved: {output_dir / 'comparative_report.txt'}")

    print("\n✓ Comparative Analysis Demo Complete!")
    print(f"📁 Output files saved to: {output_dir.absolute()}")


def demo_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print("\n" + "=" * 60)
    print("PERFORMANCE MONITORING DEMO")
    print("=" * 60)

    print("1. Creating large-scale simulation for performance testing...")

    # Create larger simulation for performance analysis
    start_time = time.time()
    trajectory, char_results = create_synthetic_micelle_formation_trajectory(
        n_frames=200, n_particles=1000
    )
    generation_time = time.time() - start_time

    print(
        f"   - Generated {len(trajectory)} frames with {len(trajectory[0].particles)} particles"
    )
    print(f"   - Generation time: {generation_time:.2f} seconds")

    print("2. Testing visualization performance...")

    config = VisualizationConfig(
        width=1200,
        height=800,
        max_particles_3d=500,  # Limit for performance
        use_gpu_acceleration=False,  # CPU-only for demo
    )

    visualizer = InteractiveSelfAssemblyVisualizer(config=config, use_gpu=False)

    # Load trajectory and measure performance
    start_time = time.time()
    visualizer.load_trajectory(trajectory, char_results)
    load_time = time.time() - start_time

    print(f"   - Trajectory loading time: {load_time:.2f} seconds")
    print(
        f"   - Memory usage: ~{len(trajectory) * len(trajectory[0].particles) * 50 / 1024 / 1024:.1f} MB estimated"
    )

    # Test visualization creation performance
    if PLOTLY_AVAILABLE:
        print("3. Testing 3D visualization rendering...")

        render_times = []
        frame_indices = [0, 50, 100, 150, 199]

        for frame_idx in frame_indices:
            start_time = time.time()
            fig = visualizer.create_3d_visualization(frame_index=frame_idx)
            render_time = time.time() - start_time
            render_times.append(render_time)
            print(f"   - Frame {frame_idx}: {render_time:.3f} seconds")

        avg_render_time = np.mean(render_times)
        print(f"   - Average render time: {avg_render_time:.3f} seconds")
        print(f"   - Estimated FPS: {1/avg_render_time:.1f}")

        # Test animation performance
        print("4. Testing animation creation...")
        start_time = time.time()
        fig_anim = visualizer.animate_trajectory(start_frame=0, end_frame=50, step=5)
        animation_time = time.time() - start_time
        print(f"   - Animation creation time: {animation_time:.2f} seconds")

        # Save performance test results
        output_dir = Path("demo_outputs")
        output_dir.mkdir(exist_ok=True)

        fig.write_html(str(output_dir / "performance_test_visualization.html"))
        fig_anim.write_html(str(output_dir / "performance_test_animation.html"))

    print("5. Testing dashboard performance...")

    if DASH_AVAILABLE:
        # Create dashboard with large dataset
        dashboard = PropertyTrackingDashboard(port=8054)

        # Generate large time series
        n_points = 5000
        timestamps = np.linspace(0, 500, n_points)

        start_time = time.time()
        dashboard.add_time_series(
            "large_dataset",
            timestamps,
            {
                "metric1": np.random.rand(n_points),
                "metric2": np.random.rand(n_points),
                "metric3": np.random.rand(n_points),
                "metric4": np.random.rand(n_points),
                "metric5": np.random.rand(n_points),
            },
        )
        data_load_time = time.time() - start_time

        print(
            f"   - Large dataset loading ({n_points} points): {data_load_time:.3f} seconds"
        )
        print(f"   - Estimated memory usage: ~{n_points * 5 * 8 / 1024 / 1024:.1f} MB")

    print("6. Generating performance report...")

    performance_report = f"""
Performance Monitoring Report
============================

Test Configuration:
- Trajectory: {len(trajectory)} frames, {len(trajectory[0].particles)} particles
- Platform: CPU-only (GPU acceleration disabled for demo)
- Visualization limits: {config.max_particles_3d} particles max

Performance Metrics:
"""

    if PLOTLY_AVAILABLE:
        performance_report += f"""
Trajectory Generation: {generation_time:.2f} seconds
Trajectory Loading: {load_time:.2f} seconds
Average 3D Render Time: {avg_render_time:.3f} seconds
Animation Creation: {animation_time:.2f} seconds
Estimated Visualization FPS: {1/avg_render_time:.1f}
"""

    if DASH_AVAILABLE:
        performance_report += f"""
Dashboard Data Loading: {data_load_time:.3f} seconds ({n_points} points)
"""

    performance_report += f"""

Optimization Recommendations:
- For >1000 particles: Enable GPU acceleration
- For real-time visualization: Limit to {config.max_particles_3d} particles
- For large datasets: Use downsampling (LTTB algorithm)
- For animations: Consider reducing frame rate or particle count

Performance Classification:
"""

    if avg_render_time < 0.1:
        performance_report += "- Visualization: EXCELLENT (>10 FPS)\n"
    elif avg_render_time < 0.5:
        performance_report += "- Visualization: GOOD (2-10 FPS)\n"
    else:
        performance_report += "- Visualization: ACCEPTABLE (<2 FPS)\n"

    performance_report += f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"

    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "performance_report.txt", "w") as f:
        f.write(performance_report)

    print(f"   - Performance report saved: {output_dir / 'performance_report.txt'}")

    print("\n✓ Performance Monitoring Demo Complete!")
    print(f"📁 Output files saved to: {output_dir.absolute()}")


def run_all_demos():
    """Run all demonstration modules."""
    print("🎯 CHEMICEL INTERACTIVE VISUALIZATION DEMO SUITE")
    print("=" * 60)
    print("This demo showcases the complete interactive visualization")
    print("capabilities of ChemiAccel's self-assembly module.")
    print("=" * 60)

    # Check dependencies
    missing_deps = []
    if not PLOTLY_AVAILABLE:
        missing_deps.append("plotly")
    if not DASH_AVAILABLE:
        missing_deps.append("dash")

    if missing_deps:
        print(f"⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("Some demo features will be limited.")
        print()

    # Run demos
    demo_functions = [
        ("3D Visualization", demo_3d_visualization),
        ("Property Dashboard", demo_property_dashboard),
        ("Phase Transitions", demo_phase_transition_analysis),
        ("Real-time Monitoring", demo_realtime_dashboard),
        ("Comparative Analysis", demo_comparative_analysis),
        ("Performance Monitoring", demo_performance_monitoring),
    ]

    for demo_name, demo_func in demo_functions:
        try:
            demo_func()
        except KeyboardInterrupt:
            print(f"\n🛑 {demo_name} demo interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ {demo_name} demo failed: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("🎉 DEMO SUITE COMPLETE!")
    print("=" * 60)
    print("Check the 'demo_outputs' directory for generated files.")
    print("Thank you for exploring ChemiAccel's visualization capabilities!")


# =============================================================================
# Main Demo Runner
# =============================================================================


def main():
    """Main demo runner with command line interface."""
    parser = argparse.ArgumentParser(
        description="ChemiAccel Interactive Self-Assembly Visualization Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python interactive_visualization_demo.py --demo all
  python interactive_visualization_demo.py --demo 3d
  python interactive_visualization_demo.py --demo dashboard
  python interactive_visualization_demo.py --demo transitions
  python interactive_visualization_demo.py --demo realtime
  python interactive_visualization_demo.py --demo comparative
  python interactive_visualization_demo.py --demo performance
        """,
    )

    parser.add_argument(
        "--demo",
        choices=[
            "all",
            "3d",
            "dashboard",
            "transitions",
            "realtime",
            "comparative",
            "performance",
        ],
        default="all",
        help="Which demo to run (default: all)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="demo_outputs",
        help="Output directory for generated files (default: demo_outputs)",
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open browser for dashboards",
    )

    args = parser.parse_args()

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"📁 Output directory: {output_dir.absolute()}")

    # Override browser opening if requested
    if args.no_browser:
        import webbrowser

        webbrowser.open = lambda url: print(f"Dashboard available at: {url}")

    # Run selected demo
    demo_map = {
        "3d": demo_3d_visualization,
        "dashboard": demo_property_dashboard,
        "transitions": demo_phase_transition_analysis,
        "realtime": demo_realtime_dashboard,
        "comparative": demo_comparative_analysis,
        "performance": demo_performance_monitoring,
        "all": run_all_demos,
    }

    selected_demo = demo_map[args.demo]

    try:
        selected_demo()
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n📁 Generated files are available in: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
