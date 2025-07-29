"""
Real-time Process Simulation and Property Evolution Demo

This example demonstrates the comprehensive real-time simulation capabilities
of ChemiAccel's digital twin system, including:
- Real-time process simulation with dynamic property prediction
- Interactive visualization of property and microstructure evolution
- Data streaming and processing
- Anomaly detection and performance monitoring
"""

import asyncio
import time
from pathlib import Path

import numpy as np

from src.chemi_accel.digital_twin.manufacturing_integration import (
    ManufacturingDigitalTwin,
    ProcessConditions,
    ProcessType,
)
from src.chemi_accel.digital_twin.multiscale_framework import MultiScaleFramework
from src.chemi_accel.digital_twin.realtime_data_streaming import (
    AnomalyDetectionProcessor,
    DataStreamManager,
    PropertyAggregationProcessor,
    RealTimeDataConnector,
    StreamingConfig,
)

# ChemiAccel imports
from src.chemi_accel.digital_twin.realtime_process_simulation import (
    MaterialProperty,
    MicrostructureState,
    PropertyType,
    RealTimeSimulationCore,
    SimulationConfig,
    SimulationMode,
)
from src.chemi_accel.digital_twin.realtime_visualization import (
    RealTimeVisualizer,
    VisualizationConfig,
    VisualizationMode,
)


def create_polymer_extrusion_conditions():
    """Create realistic polymer extrusion process conditions."""
    return ProcessConditions(
        temperature=220.0 + 273.15,  # 220°C in Kelvin
        pressure=15e6,  # 15 MPa
        material_composition={
            "polyethylene": 0.7,
            "polypropylene": 0.2,
            "additives": 0.1,
        },
    )


def create_initial_microstructure():
    """Create initial microstructure state for polymer."""
    return MicrostructureState(
        grain_size=0.5,  # μm
        grain_orientation=np.array([1.0, 0.0, 0.0]),
        phase_fractions={"crystalline": 0.3, "amorphous": 0.7},
        defect_density=0.001,
        crystallinity=0.3,
        molecular_orientation=np.array([0.8, 0.6, 0.0]),
        chain_length_distribution=np.array([1000, 5000, 10000, 20000, 50000]),
        crosslink_density=0.05,
    )


def create_process_schedule():
    """Create a schedule of process condition changes."""
    return [
        (
            30.0,
            ProcessConditions(  # Temperature ramp at 30s
                temperature=250.0 + 273.15,
                pressure=15e6,
                material_composition={
                    "polyethylene": 0.7,
                    "polypropylene": 0.2,
                    "additives": 0.1,
                },
            ),
        ),
        (
            60.0,
            ProcessConditions(  # Pressure increase at 60s
                temperature=250.0 + 273.15,
                pressure=20e6,
                material_composition={
                    "polyethylene": 0.7,
                    "polypropylene": 0.2,
                    "additives": 0.1,
                },
            ),
        ),
        (
            90.0,
            ProcessConditions(  # Cool down at 90s
                temperature=200.0 + 273.15,
                pressure=20e6,
                material_composition={
                    "polyethylene": 0.7,
                    "polypropylene": 0.2,
                    "additives": 0.1,
                },
            ),
        ),
    ]


async def run_basic_simulation_demo():
    """Run basic real-time simulation demonstration."""
    print("=" * 60)
    print("ChemiAccel Real-time Simulation Demo")
    print("=" * 60)

    # 1. Setup simulation configuration
    print("\n1. Setting up simulation configuration...")
    sim_config = SimulationConfig(
        mode=SimulationMode.ACCELERATED,  # Faster than real-time for demo
        time_step=0.1,
        max_simulation_time=120.0,  # 2 minutes
        update_frequency=10.0,  # 10 Hz updates
        enable_gpu_acceleration=False,  # Use CPU for compatibility
        adaptive_time_stepping=True,
        enable_visualization=True,
    )

    # 2. Create multiscale framework (mock for demo)
    print("2. Creating multiscale framework...")
    multiscale_framework = MultiScaleFramework()

    # 3. Initialize simulation core
    print("3. Initializing simulation core...")
    simulation_core = RealTimeSimulationCore(
        config=sim_config, multiscale_framework=multiscale_framework
    )

    # 4. Setup initial conditions
    print("4. Setting up initial conditions...")
    initial_conditions = create_polymer_extrusion_conditions()
    initial_microstructure = create_initial_microstructure()

    print(f"   Initial temperature: {initial_conditions.temperature:.1f} K")
    print(f"   Initial pressure: {initial_conditions.pressure/1e6:.1f} MPa")
    print(f"   Initial crystallinity: {initial_microstructure.crystallinity:.2f}")

    # 5. Initialize simulation
    print("5. Initializing simulation state...")
    initial_state = simulation_core.initialize_simulation(
        initial_conditions, initial_microstructure
    )

    print(f"   Simulation initialized at t={initial_state.time:.2f}s")
    print(
        f"   Initial properties: {len(initial_state.material_properties)} properties tracked"
    )

    # 6. Setup process schedule
    print("6. Setting up process schedule...")
    process_schedule = create_process_schedule()
    print(f"   {len(process_schedule)} process changes scheduled")

    # 7. Run simulation
    print("\n7. Running simulation...")
    print("   Starting real-time simulation (this may take a moment)...")

    start_time = time.time()
    simulation_history = await simulation_core.run_simulation(
        duration=sim_config.max_simulation_time, process_updates=process_schedule
    )
    end_time = time.time()

    print(f"   Simulation completed in {end_time - start_time:.2f} seconds")
    print(f"   Generated {len(simulation_history)} simulation states")

    # 8. Analyze results
    print("\n8. Analyzing simulation results...")

    # Extract property evolution
    times, tensile_strengths = simulation_core.get_property_evolution(
        "tensile_strength"
    )
    times, elastic_moduli = simulation_core.get_property_evolution("elastic_modulus")
    times, thermal_conductivities = simulation_core.get_property_evolution(
        "thermal_conductivity"
    )

    # Extract microstructure evolution
    times_micro, grain_sizes = simulation_core.get_microstructure_evolution(
        "grain_size"
    )
    times_micro, crystallinities = simulation_core.get_microstructure_evolution(
        "crystallinity"
    )

    print(f"   Property evolution tracked over {len(times)} time points")
    print(
        f"   Tensile strength range: {min(tensile_strengths):.1f} - {max(tensile_strengths):.1f} MPa"
    )
    print(
        f"   Elastic modulus range: {min(elastic_moduli):.1f} - {max(elastic_moduli):.1f} MPa"
    )
    print(
        f"   Crystallinity evolution: {min(crystallinities):.3f} - {max(crystallinities):.3f}"
    )

    # 9. Performance metrics
    print("\n9. Performance metrics...")
    metrics = simulation_core.get_performance_metrics()
    print(f"   Simulation time ratio: {metrics['time_ratio']:.2f}x real-time")
    print(f"   Average update rate: {metrics['updates_per_second']:.1f} Hz")
    print(f"   Memory usage: {metrics['memory_usage']/1000:.1f} KB")

    # 10. Export results
    print("\n10. Exporting results...")
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)

    trajectory_file = output_dir / "simulation_trajectory.json"
    simulation_core.export_trajectory(str(trajectory_file))
    print(f"    Trajectory exported to: {trajectory_file}")

    return simulation_core, simulation_history


async def run_visualization_demo():
    """Run interactive visualization demonstration."""
    print("\n" + "=" * 60)
    print("Interactive Visualization Demo")
    print("=" * 60)

    # Run basic simulation first
    simulation_core, simulation_history = await run_basic_simulation_demo()

    # Setup visualization
    print("\n1. Setting up visualization system...")

    try:
        viz_config = VisualizationConfig(
            mode=VisualizationMode.PLOTLY,
            update_interval=0.1,
            max_history_points=1000,
            enable_real_time_updates=False,  # Use post-simulation data
            figure_size=(1200, 800),
            enable_animations=True,
        )

        visualizer = RealTimeVisualizer(simulation_core, viz_config)
        print("   Visualization system initialized successfully")

        # Create property evolution plots
        print("\n2. Creating property evolution plots...")

        # Material properties plot
        prop_plot = visualizer.create_property_evolution_plot(
            properties=["tensile_strength", "elastic_modulus", "thermal_conductivity"]
        )
        print("   Created material properties evolution plot")

        # Microstructure evolution plot
        micro_plot = visualizer.create_microstructure_evolution_plot(
            parameters=["grain_size", "crystallinity", "defect_density"]
        )
        print("   Created microstructure evolution plot")

        # Process conditions plot
        process_plot = visualizer.create_process_conditions_plot(
            parameters=["temperature", "pressure"]
        )
        print("   Created process conditions plot")

        # Property correlation plot
        corr_plot = visualizer.create_property_correlation_plot(
            property_x="tensile_strength", property_y="elastic_modulus"
        )
        print("   Created property correlation plot")

        # 3. Populate with simulation data
        print("\n3. Populating plots with simulation data...")

        # Simulate real-time updates with historical data
        for state in simulation_history[::5]:  # Every 5th state for performance
            visualizer._on_simulation_update(state)

        print(f"   Processed {len(simulation_history)} simulation states")

        # 4. Save plots
        print("\n4. Saving visualization plots...")
        output_dir = Path("demo_output")

        visualizer.save_plot(
            "property_evolution", str(output_dir / "property_evolution.html")
        )
        visualizer.save_plot(
            "microstructure_evolution",
            str(output_dir / "microstructure_evolution.html"),
        )
        visualizer.save_plot(
            "process_conditions", str(output_dir / "process_conditions.html")
        )
        visualizer.save_plot(
            "property_correlation", str(output_dir / "property_correlation.html")
        )

        print("   All plots saved to demo_output/ directory")

        # 5. Export visualization data
        visualizer.export_data(str(output_dir / "visualization_data.json"))
        print("   Visualization data exported")

        # 6. Create comprehensive dashboard
        print("\n5. Creating comprehensive dashboard...")
        dashboard = visualizer.create_dashboard()

        # Save dashboard
        if hasattr(dashboard, "write_html"):
            dashboard.write_html(str(output_dir / "simulation_dashboard.html"))
            print("   Interactive dashboard saved as simulation_dashboard.html")

        print("\n✓ Visualization demo completed successfully!")
        print(
            f"   Open {output_dir}/simulation_dashboard.html in a web browser to view results"
        )

    except ImportError as e:
        print(f"   Warning: Visualization libraries not available ({e})")
        print("   Skipping visualization demo")
        print("   Install plotly and matplotlib for full visualization support")


async def run_streaming_demo():
    """Run data streaming and processing demonstration."""
    print("\n" + "=" * 60)
    print("Real-time Data Streaming Demo")
    print("=" * 60)

    # 1. Setup streaming configuration
    print("\n1. Setting up data streaming system...")

    streaming_config = StreamingConfig(
        mode=StreamingMode.PUSH,
        buffer_size=1000,
        batch_size=10,
        flush_interval=1.0,
        enable_persistence=False,  # Disable for demo
        max_concurrent_streams=4,
    )

    # 2. Create stream manager
    stream_manager = DataStreamManager(streaming_config)
    print("   Stream manager created")

    # 3. Add data processors
    print("\n2. Adding data processors...")

    # Property aggregation processor
    agg_processor = PropertyAggregationProcessor(
        window_size=10, aggregation_functions=["mean", "std", "min", "max"]
    )
    stream_manager.add_processor(agg_processor)
    print("   Added property aggregation processor")

    # Anomaly detection processor
    anomaly_processor = AnomalyDetectionProcessor(
        threshold_multiplier=2.5, min_samples=20
    )
    stream_manager.add_processor(anomaly_processor)
    print("   Added anomaly detection processor")

    # 4. Setup data capture
    print("\n3. Setting up data capture...")

    captured_packets = {
        "simulation_state": [],
        "aggregated_properties": [],
        "anomalies": [],
    }

    def capture_simulation_data(packet):
        captured_packets["simulation_state"].append(packet)

    def capture_aggregated_data(packet):
        captured_packets["aggregated_properties"].append(packet)
        print(f"   📊 Aggregated properties at t={packet.timestamp:.2f}")

    def capture_anomalies(packet):
        captured_packets["anomalies"].append(packet)
        anomalies = packet.payload.get("anomalies", {})
        print(f"   🚨 Anomaly detected: {list(anomalies.keys())}")

    # Subscribe to data streams
    stream_manager.subscribe("simulation_state", capture_simulation_data)
    stream_manager.subscribe("aggregated_properties", capture_aggregated_data)
    stream_manager.subscribe("anomalies", capture_anomalies)

    # 5. Create simulation for streaming
    print("\n4. Creating simulation for data streaming...")

    sim_config = SimulationConfig(
        mode=SimulationMode.ACCELERATED,
        time_step=0.05,
        max_simulation_time=60.0,  # 1 minute
        enable_gpu_acceleration=False,
    )

    multiscale_framework = MultiScaleFramework()
    simulation_core = RealTimeSimulationCore(sim_config, multiscale_framework)

    # 6. Connect simulation to streaming
    print("\n5. Connecting simulation to streaming system...")

    connector = RealTimeDataConnector(simulation_core, stream_manager)
    connector.setup_default_processing_pipeline()

    # 7. Start streaming
    print("\n6. Starting data streaming...")
    stream_manager.start_streaming()

    try:
        # Initialize and run simulation
        initial_conditions = create_polymer_extrusion_conditions()
        initial_microstructure = create_initial_microstructure()

        simulation_core.initialize_simulation(
            initial_conditions, initial_microstructure
        )

        print("   Simulation running with real-time data streaming...")

        # Add some process variations to trigger anomalies
        process_schedule = [
            (
                20.0,
                ProcessConditions(  # Normal variation
                    temperature=230.0 + 273.15,
                    pressure=16e6,
                    material_composition={
                        "polyethylene": 0.7,
                        "polypropylene": 0.2,
                        "additives": 0.1,
                    },
                ),
            ),
            (
                40.0,
                ProcessConditions(  # Extreme condition (should trigger anomaly)
                    temperature=350.0 + 273.15,  # Very high temperature
                    pressure=30e6,  # Very high pressure
                    material_composition={
                        "polyethylene": 0.7,
                        "polypropylene": 0.2,
                        "additives": 0.1,
                    },
                ),
            ),
        ]

        # Run simulation
        simulation_history = await simulation_core.run_simulation(
            duration=sim_config.max_simulation_time, process_updates=process_schedule
        )

        # Wait for processing to complete
        time.sleep(2.0)

        # 8. Analyze streaming results
        print("\n7. Analyzing streaming results...")

        print(
            f"   Simulation states captured: {len(captured_packets['simulation_state'])}"
        )
        print(
            f"   Aggregated data packets: {len(captured_packets['aggregated_properties'])}"
        )
        print(f"   Anomalies detected: {len(captured_packets['anomalies'])}")

        # Stream manager metrics
        metrics = stream_manager.get_metrics()
        print(f"   Total packets processed: {metrics['packets_processed']}")
        print(f"   Packets dropped: {metrics['packets_dropped']}")
        print(f"   Average processing time: {metrics['processing_time_avg']:.4f}s")

        # Stream info
        info = stream_manager.get_stream_info()
        print(f"   Active processors: {info['active_processors']}")
        print(f"   Subscriber count: {info['subscriber_count']}")

        print("\n✓ Streaming demo completed successfully!")

        return captured_packets

    finally:
        stream_manager.stop_streaming()
        print("   Streaming system stopped")


async def run_comprehensive_demo():
    """Run comprehensive demonstration of all features."""
    print("🚀 Starting ChemiAccel Real-time Simulation Comprehensive Demo")
    print("=" * 80)

    try:
        # 1. Basic simulation
        simulation_core, simulation_history = await run_basic_simulation_demo()

        # 2. Visualization demo
        await run_visualization_demo()

        # 3. Streaming demo
        captured_packets = await run_streaming_demo()

        # 4. Summary
        print("\n" + "=" * 80)
        print("📊 Demo Summary")
        print("=" * 80)

        print(f"✓ Real-time simulation: {len(simulation_history)} states generated")
        print("✓ Property evolution: Multiple material properties tracked")
        print("✓ Microstructure evolution: Grain size, crystallinity, defects")
        print("✓ Interactive visualization: Multiple plot types created")
        print(
            f"✓ Data streaming: {sum(len(packets) for packets in captured_packets.values())} packets processed"
        )
        print(
            f"✓ Anomaly detection: {len(captured_packets['anomalies'])} anomalies detected"
        )

        print("\n📁 Output files created in demo_output/:")
        output_dir = Path("demo_output")
        if output_dir.exists():
            for file in output_dir.glob("*"):
                print(f"   • {file.name}")

        print("\n🎉 All demonstrations completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Open simulation_dashboard.html in a web browser")
        print("   2. Examine the exported data files")
        print("   3. Modify process conditions and re-run simulations")
        print("   4. Integrate with your own manufacturing processes")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("   Check that all dependencies are installed")
        print("   Some features may require optional dependencies (plotly, matplotlib)")
        import traceback

        traceback.print_exc()


def create_simple_visualization_example():
    """Create a simple visualization example that works without full simulation."""
    print("\n" + "=" * 60)
    print("Simple Visualization Example (No External Dependencies)")
    print("=" * 60)

    # Generate synthetic data
    times = np.linspace(0, 100, 1000)

    # Simulate property evolution
    base_strength = 100.0
    strength_noise = np.random.normal(0, 2, len(times))
    temperature_effect = 10 * np.sin(times / 20)  # Temperature cycling effect
    tensile_strength = base_strength + temperature_effect + strength_noise

    base_modulus = 2000.0
    modulus_noise = np.random.normal(0, 50, len(times))
    modulus_trend = -0.5 * times  # Degradation over time
    elastic_modulus = base_modulus + modulus_trend + modulus_noise

    # Microstructure evolution
    initial_grain_size = 0.5
    grain_growth = 0.001 * times
    grain_size = initial_grain_size + grain_growth

    initial_crystallinity = 0.3
    crystallinity_change = 0.1 * (1 - np.exp(-times / 50))
    crystallinity = initial_crystallinity + crystallinity_change

    # Create simple plots using matplotlib if available
    try:
        import matplotlib.pyplot as plt

        output_dir = Path("demo_output")
        output_dir.mkdir(exist_ok=True)

        # Property evolution plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(
            times, tensile_strength, label="Tensile Strength", color="blue", linewidth=2
        )
        ax1_twin = ax1.twinx()
        ax1_twin.plot(
            times, elastic_modulus, label="Elastic Modulus", color="red", linewidth=2
        )

        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Tensile Strength (MPa)", color="blue")
        ax1_twin.set_ylabel("Elastic Modulus (MPa)", color="red")
        ax1.set_title("Material Property Evolution During Processing")
        ax1.grid(True, alpha=0.3)

        # Microstructure evolution plot
        ax2.plot(times, grain_size, label="Grain Size", color="green", linewidth=2)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(
            times, crystallinity, label="Crystallinity", color="orange", linewidth=2
        )

        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Grain Size (μm)", color="green")
        ax2_twin.set_ylabel("Crystallinity", color="orange")
        ax2.set_title("Microstructure Evolution During Processing")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / "simple_property_evolution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Property correlation plot
        plt.figure(figsize=(8, 6))
        plt.scatter(
            tensile_strength, elastic_modulus, c=times, cmap="viridis", alpha=0.6
        )
        plt.colorbar(label="Time (s)")
        plt.xlabel("Tensile Strength (MPa)")
        plt.ylabel("Elastic Modulus (MPa)")
        plt.title("Property Correlation Over Time")
        plt.grid(True, alpha=0.3)
        plt.savefig(
            output_dir / "property_correlation.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print("✓ Simple visualization plots created:")
        print(f"   • {output_dir}/simple_property_evolution.png")
        print(f"   • {output_dir}/property_correlation.png")

    except ImportError:
        print("   Matplotlib not available - creating data files instead")

        # Save data as CSV files
        output_dir = Path("demo_output")
        output_dir.mkdir(exist_ok=True)

        import csv

        # Property data
        with open(output_dir / "property_evolution.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Time",
                    "Tensile_Strength",
                    "Elastic_Modulus",
                    "Grain_Size",
                    "Crystallinity",
                ]
            )
            for i in range(len(times)):
                writer.writerow(
                    [
                        times[i],
                        tensile_strength[i],
                        elastic_modulus[i],
                        grain_size[i],
                        crystallinity[i],
                    ]
                )

        print("✓ Data files created:")
        print(f"   • {output_dir}/property_evolution.csv")

    return {
        "times": times,
        "tensile_strength": tensile_strength,
        "elastic_modulus": elastic_modulus,
        "grain_size": grain_size,
        "crystallinity": crystallinity,
    }


def print_demo_info():
    """Print information about the demo."""
    print(
        """
ChemiAccel Real-time Process Simulation Demo
==========================================

This demonstration showcases:

🔬 Real-time Process Simulation:
   • Dynamic material property prediction
   • Microstructure evolution tracking
   • Multi-physics coupling
   • Adaptive time stepping

📊 Interactive Visualization:
   • Real-time property evolution plots
   • Microstructure parameter tracking
   • Process condition monitoring
   • Property correlation analysis

📡 Data Streaming & Processing:
   • Real-time data ingestion
   • Property aggregation
   • Anomaly detection
   • Performance monitoring

🏭 Manufacturing Integration:
   • Polymer extrusion simulation
   • Process schedule management
   • Quality monitoring
   • Performance optimization

Features demonstrated:
✓ GPU-accelerated simulations (when available)
✓ Interactive web-based visualizations
✓ Real-time data streaming
✓ Anomaly detection algorithms
✓ Multi-objective optimization
✓ Digital twin integration

Dependencies:
• Core: numpy, torch
• Visualization: plotly, matplotlib (optional)
• Web server: flask (optional)
• Analysis: scipy, scikit-learn (optional)

Usage:
   python realtime_simulation_demo.py
"""
    )


if __name__ == "__main__":
    print_demo_info()

    # Check for async support
    try:
        # Run comprehensive demo
        asyncio.run(run_comprehensive_demo())
    except Exception as e:
        print(f"\nFull demo failed: {e}")
        print("Running simplified example instead...")

        # Fallback to simple example
        create_simple_visualization_example()

        print("\n💡 For full functionality, ensure all dependencies are installed:")
        print("   pip install plotly matplotlib flask scipy scikit-learn")
