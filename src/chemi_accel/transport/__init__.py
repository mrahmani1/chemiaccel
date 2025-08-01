"""
AI-Driven Multi-Phase Transport & Bioavailability Optimization

This module provides comprehensive capabilities for modeling and optimizing molecular
transport across different phases and biological barriers, including:

- GPU-accelerated multi-phase simulation engine
- Transport kinetics and partitioning characterization
- Inverse design optimization for transport properties
- Active transport and metabolism modeling
- Interactive visualization of transport pathways

Classes:
    MultiPhaseSimulator: GPU-accelerated simulation engine for transport
    TransportAnalyzer: Characterization of transport kinetics and partitioning
    TransportOptimizer: Inverse design for optimal transport properties
    BiologicalTransportModel: Active transport and metabolism integration
    TransportVisualizer: Interactive visualization of transport pathways

Example:
    >>> from chemi_accel.transport import MultiPhaseSimulator, TransportAnalyzer
    >>> simulator = MultiPhaseSimulator(backend='cuda')
    >>> analyzer = TransportAnalyzer()
    >>>
    >>> # Simulate transport across lipid bilayer
    >>> results = simulator.simulate_membrane_transport(
    ...     molecules=['aspirin', 'caffeine'],
    ...     membrane_type='dppc_bilayer',
    ...     conditions={'temperature': 310.0, 'pH': 7.4}
    ... )
    >>>
    >>> # Analyze transport properties
    >>> transport_props = analyzer.analyze_permeability(results)
"""

from .biological_transport import BiologicalTransportModel
from .simulation_engine import MultiPhaseSimulator
from .transport_analyzer import TransportAnalyzer
from .transport_optimizer import TransportOptimizer
from .transport_visualizer import TransportVisualizer

__all__ = [
    "MultiPhaseSimulator",
    "TransportAnalyzer",
    "TransportOptimizer",
    "BiologicalTransportModel",
    "TransportVisualizer",
]
