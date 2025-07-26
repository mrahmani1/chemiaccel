"""
Real-time Simulation Engine and Interactive Control

GPU-accelerated simulation engine for multi-scale models with interactive
user interface for real-time parameter modification and visualization.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .multiscale_framework import MultiScaleFramework

logger = logging.getLogger(__name__)


class SimulationState(Enum):
    """Simulation execution states."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class SimulationParameters:
    """Parameters for real-time simulation."""
    dt: float = 0.01
    max_time: float = 100.0
    update_frequency: float = 10.0  # Hz
    auto_save_interval: float = 10.0  # seconds
    visualization_enabled: bool = True
    interactive_mode: bool = True


@dataclass
class SimulationState:
    """Current state of the simulation."""
    current_time: float = 0.0
    step_count: int = 0
    frame_rate: float = 0.0
    last_update: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)


class InteractiveController:
    """Interactive controller for real-time parameter modification."""
    
    def __init__(self):
        self.parameter_callbacks = {}
        self.command_queue = []
        self.lock = threading.Lock()
    
    def register_parameter(self, name: str, callback: Callable) -> None:
        """Register a parameter with its update callback."""
        self.parameter_callbacks[name] = callback
    
    def update_parameter(self, name: str, value: Any) -> bool:
        """Update a parameter in real-time."""
        if name in self.parameter_callbacks:
            try:
                self.parameter_callbacks[name](value)
                return True
            except Exception as e:
                logger.error(f"Error updating parameter {name}: {e}")
        return False
    
    def queue_command(self, command: str, **kwargs) -> None:
        """Queue a command for execution."""
        with self.lock:
            self.command_queue.append({'command': command, 'kwargs': kwargs})
    
    def process_commands(self) -> List[Dict[str, Any]]:
        """Process queued commands."""
        with self.lock:
            commands = self.command_queue.copy()
            self.command_queue.clear()
        return commands


class UserInterface:
    """Simple user interface for simulation control."""
    
    def __init__(self, controller: InteractiveController):
        self.controller = controller
        self.widgets = {}
        self.callbacks = {}
    
    def create_slider(self, name: str, min_val: float, max_val: float, 
                     initial: float, callback: Callable = None) -> None:
        """Create a parameter slider (placeholder for actual UI)."""
        self.widgets[name] = {
            'type': 'slider',
            'min': min_val,
            'max': max_val,
            'value': initial,
            'callback': callback
        }
        
        if callback:
            self.controller.register_parameter(name, callback)
    
    def create_button(self, name: str, callback: Callable) -> None:
        """Create a button (placeholder for actual UI)."""
        self.widgets[name] = {
            'type': 'button',
            'callback': callback
        }
    
    def update_display(self, simulation_state: Dict[str, Any]) -> None:
        """Update the display with current simulation state."""
        # Placeholder for actual UI update
        pass


class RealtimeSimulationEngine:
    """
    GPU-accelerated real-time simulation engine for the digital twin.
    
    This engine coordinates multi-scale models and provides interactive
    control capabilities for real-time parameter modification.
    """
    
    def __init__(self, multiscale_framework: MultiScaleFramework):
        self.framework = multiscale_framework
        self.state = SimulationState.IDLE
        self.parameters = SimulationParameters()
        self.current_state = SimulationState()
        
        # Interactive components
        self.controller = InteractiveController()
        self.ui = UserInterface(self.controller)
        
        # Threading
        self.simulation_thread = None
        self.stop_event = threading.Event()
        
        # Performance monitoring
        self.frame_times = []
        self.performance_stats = {}
        
        # Callbacks
        self.update_callbacks = []
        self.frame_callbacks = []
        
        self._setup_interactive_controls()
    
    def _setup_interactive_controls(self) -> None:
        """Setup interactive parameter controls."""
        
        # Temperature control
        def update_temperature(value):
            for model_id in self.framework.models:
                try:
                    state = self.framework.get_model_state(model_id)
                    state['temperature'] = value
                    self.framework.set_model_state(model_id, state)
                except Exception as e:
                    logger.error(f"Error updating temperature for {model_id}: {e}")
        
        self.controller.register_parameter('temperature', update_temperature)
        
        # Pressure control
        def update_pressure(value):
            for model_id in self.framework.models:
                try:
                    state = self.framework.get_model_state(model_id)
                    state['pressure'] = value
                    self.framework.set_model_state(model_id, state)
                except Exception as e:
                    logger.error(f"Error updating pressure for {model_id}: {e}")
        
        self.controller.register_parameter('pressure', update_pressure)
        
        # Applied stress control
        def update_stress(value):
            self.current_state.parameters['applied_stress'] = value
        
        self.controller.register_parameter('applied_stress', update_stress)
    
    def add_update_callback(self, callback: Callable) -> None:
        """Add callback for simulation updates."""
        self.update_callbacks.append(callback)
    
    def add_frame_callback(self, callback: Callable) -> None:
        """Add callback for frame updates."""
        self.frame_callbacks.append(callback)
    
    def start_simulation(self) -> None:
        """Start the real-time simulation."""
        if self.state == SimulationState.RUNNING:
            logger.warning("Simulation is already running")
            return
        
        self.state = SimulationState.RUNNING
        self.stop_event.clear()
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self._simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        logger.info("Real-time simulation started")
    
    def pause_simulation(self) -> None:
        """Pause the simulation."""
        if self.state == SimulationState.RUNNING:
            self.state = SimulationState.PAUSED
            logger.info("Simulation paused")
    
    def resume_simulation(self) -> None:
        """Resume the simulation."""
        if self.state == SimulationState.PAUSED:
            self.state = SimulationState.RUNNING
            logger.info("Simulation resumed")
    
    def stop_simulation(self) -> None:
        """Stop the simulation."""
        self.state = SimulationState.STOPPED
        self.stop_event.set()
        
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=5.0)
        
        logger.info("Simulation stopped")
    
    def reset_simulation(self) -> None:
        """Reset simulation to initial state."""
        self.stop_simulation()
        
        self.current_state = SimulationState()
        self.frame_times = []
        self.performance_stats = {}
        
        # Reset models
        for model_id in self.framework.models:
            # Would reset model states here
            pass
        
        self.state = SimulationState.IDLE
        logger.info("Simulation reset")
    
    def _simulation_loop(self) -> None:
        """Main simulation loop running in separate thread."""
        try:
            last_frame_time = time.time()
            target_frame_interval = 1.0 / self.parameters.update_frequency
            
            while not self.stop_event.is_set() and self.state != SimulationState.STOPPED:
                
                # Check if paused
                if self.state == SimulationState.PAUSED:
                    time.sleep(0.1)
                    continue
                
                frame_start = time.time()
                
                # Process interactive commands
                self._process_interactive_commands()
                
                # Perform simulation step
                if self.state == SimulationState.RUNNING:
                    self._simulation_step()
                
                # Update callbacks
                self._call_update_callbacks()
                
                # Frame timing
                frame_end = time.time()
                frame_duration = frame_end - frame_start
                self.frame_times.append(frame_duration)
                
                # Keep only recent frame times
                if len(self.frame_times) > 100:
                    self.frame_times = self.frame_times[-100:]
                
                # Calculate frame rate
                self.current_state.frame_rate = 1.0 / frame_duration if frame_duration > 0 else 0
                
                # Sleep to maintain target frame rate
                sleep_time = target_frame_interval - frame_duration
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                last_frame_time = frame_end
                
        except Exception as e:
            logger.error(f"Error in simulation loop: {e}")
            self.state = SimulationState.ERROR
    
    def _process_interactive_commands(self) -> None:
        """Process interactive commands from the controller."""
        commands = self.controller.process_commands()
        
        for command_dict in commands:
            command = command_dict['command']
            kwargs = command_dict['kwargs']
            
            try:
                if command == 'pause':
                    self.pause_simulation()
                elif command == 'resume':
                    self.resume_simulation()
                elif command == 'stop':
                    self.stop_simulation()
                elif command == 'reset':
                    self.reset_simulation()
                elif command == 'update_parameter':
                    param_name = kwargs.get('name')
                    param_value = kwargs.get('value')
                    if param_name and param_value is not None:
                        self.controller.update_parameter(param_name, param_value)
                
            except Exception as e:
                logger.error(f"Error processing command {command}: {e}")
    
    def _simulation_step(self) -> None:
        """Perform one simulation step."""
        try:
            # Step the multi-scale framework
            step_kwargs = self.current_state.parameters.copy()
            results = self.framework.step(self.parameters.dt, **step_kwargs)
            
            # Update current state
            self.current_state.current_time += self.parameters.dt
            self.current_state.step_count += 1
            self.current_state.last_update = time.time()
            
            # Store results for callbacks
            self._store_step_results(results)
            
        except Exception as e:
            logger.error(f"Error in simulation step: {e}")
            self.state = SimulationState.ERROR
    
    def _store_step_results(self, results: Dict[str, Any]) -> None:
        """Store step results for visualization and analysis."""
        # Store in current state
        self.current_state.parameters.update(results)
        
        # Update performance statistics
        self._update_performance_stats()
    
    def _update_performance_stats(self) -> None:
        """Update performance statistics."""
        if self.frame_times:
            self.performance_stats = {
                'avg_frame_time': np.mean(self.frame_times),
                'min_frame_time': np.min(self.frame_times),
                'max_frame_time': np.max(self.frame_times),
                'frame_rate': self.current_state.frame_rate,
                'total_steps': self.current_state.step_count,
                'simulation_time': self.current_state.current_time
            }
    
    def _call_update_callbacks(self) -> None:
        """Call registered update callbacks."""
        for callback in self.update_callbacks:
            try:
                callback(self.current_state)
            except Exception as e:
                logger.error(f"Error in update callback: {e}")
        
        for callback in self.frame_callbacks:
            try:
                callback(self.performance_stats)
            except Exception as e:
                logger.error(f"Error in frame callback: {e}")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current simulation state."""
        return {
            'state': self.state.value,
            'time': self.current_state.current_time,
            'steps': self.current_state.step_count,
            'frame_rate': self.current_state.frame_rate,
            'parameters': self.current_state.parameters.copy(),
            'performance': self.performance_stats.copy()
        }
    
    def set_parameter(self, name: str, value: Any) -> bool:
        """Set a simulation parameter interactively."""
        return self.controller.update_parameter(name, value)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return self.performance_stats.copy()
    
    def save_state(self, filepath: str) -> None:
        """Save current simulation state."""
        import json
        
        state_data = {
            'simulation_state': self.get_current_state(),
            'framework_state': self.framework.get_simulation_summary(),
            'parameters': {
                'dt': self.parameters.dt,
                'max_time': self.parameters.max_time,
                'update_frequency': self.parameters.update_frequency
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
        
        logger.info(f"Simulation state saved to {filepath}")
    
    def load_state(self, filepath: str) -> None:
        """Load simulation state from file."""
        import json
        
        with open(filepath, 'r') as f:
            state_data = json.load(f)
        
        # Restore parameters
        if 'parameters' in state_data:
            params = state_data['parameters']
            self.parameters.dt = params.get('dt', self.parameters.dt)
            self.parameters.max_time = params.get('max_time', self.parameters.max_time)
            self.parameters.update_frequency = params.get('update_frequency', self.parameters.update_frequency)
        
        # Restore simulation state
        if 'simulation_state' in state_data:
            sim_state = state_data['simulation_state']
            self.current_state.current_time = sim_state.get('time', 0.0)
            self.current_state.step_count = sim_state.get('steps', 0)
            self.current_state.parameters = sim_state.get('parameters', {})
        
        logger.info(f"Simulation state loaded from {filepath}")
    
    def create_ui_controls(self) -> None:
        """Create UI controls for interactive simulation."""
        
        # Temperature slider
        self.ui.create_slider(
            'temperature', 200.0, 500.0, 300.0,
            lambda value: self.set_parameter('temperature', value)
        )
        
        # Pressure slider
        self.ui.create_slider(
            'pressure', 0.1, 10.0, 1.0,
            lambda value: self.set_parameter('pressure', value)
        )
        
        # Applied stress slider
        self.ui.create_slider(
            'applied_stress', 0.0, 1e6, 0.0,
            lambda value: self.set_parameter('applied_stress', value)
        )
        
        # Control buttons
        self.ui.create_button('start', self.start_simulation)
        self.ui.create_button('pause', self.pause_simulation)
        self.ui.create_button('stop', self.stop_simulation)
        self.ui.create_button('reset', self.reset_simulation)
        
        logger.info("UI controls created")
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data for visualization."""
        return {
            'time_series': self.framework.history[-100:] if self.framework.history else [],
            'current_state': self.get_current_state(),
            'material_properties': self.framework.get_material_properties(),
            'performance_metrics': self.get_performance_metrics()
        }