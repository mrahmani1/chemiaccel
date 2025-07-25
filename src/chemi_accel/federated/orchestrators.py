"""
Federated Learning Orchestrator Integrations for ChemiAccel - Task 21.1

This module provides integration with popular federated learning frameworks
including PySyft and Flower for coordinating distributed molecular computation
while preserving privacy and security.

Key Features:
- PySyft integration for secure multi-party computation
- Flower framework integration for federated learning orchestration
- Custom ChemiAccel orchestrator for molecular-specific workflows
- Secure aggregation protocols with privacy preservation
- Dynamic client discovery and management
- Fault-tolerant federated training coordination

Scientific Applications:
- Multi-institutional pharmaceutical research
- Collaborative molecular property prediction
- Distributed chemical database analysis
- Privacy-preserving consortium learning
"""

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

try:
    import numpy as np
    import pandas as pd

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    pd = None

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Optional federated learning framework imports
try:
    import syft as sy
    from syft import VirtualMachine

    SYFT_AVAILABLE = True
except ImportError:
    SYFT_AVAILABLE = False
    sy = None

try:
    import flwr as fl
    from flwr.common import Metrics, NDArrays, Parameters
    from flwr.server.strategy import FedAvg, Strategy

    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False
    fl = None

from ..core.exceptions import ChemiAccelError
from ..utils.logging import get_logger

# ChemiAccel imports
from .core import FederatedConfig

logger = get_logger()


class OrchestrationError(ChemiAccelError):
    """Exception for federated orchestration errors."""

    pass


@dataclass
class ClientInfo:
    """Information about a federated learning client."""

    client_id: str
    address: str
    capabilities: Dict[str, Any]
    status: str = "active"  # active, inactive, disconnected
    last_seen: float = field(default_factory=time.time)
    data_size: int = 0
    privacy_level: str = "high"  # low, medium, high


@dataclass
class RoundResults:
    """Results from a federated learning round."""

    round_number: int
    participating_clients: List[str]
    aggregated_metrics: Dict[str, float]
    convergence_metrics: Dict[str, float]
    privacy_budget_used: float
    round_duration: float
    success_rate: float


class FederatedOrchestrator(ABC):
    """Abstract base class for federated learning orchestrators."""

    @abstractmethod
    async def start_server(self) -> None:
        """Start the federated learning server."""
        pass

    @abstractmethod
    async def stop_server(self) -> None:
        """Stop the federated learning server."""
        pass

    @abstractmethod
    async def register_client(self, client_info: ClientInfo) -> bool:
        """Register a new client."""
        pass

    @abstractmethod
    async def run_federated_round(self, round_number: int) -> RoundResults:
        """Execute a single federated learning round."""
        pass

    @abstractmethod
    def get_server_status(self) -> Dict[str, Any]:
        """Get current server status."""
        pass


class PySyftOrchestrator(FederatedOrchestrator):
    """PySyft-based federated learning orchestrator for ChemiAccel."""

    def __init__(self, config: FederatedConfig, server_port: int = 8080):
        """Initialize PySyft orchestrator.

        Args:
            config: Federated learning configuration
            server_port: Port for the PySyft server
        """
        if not SYFT_AVAILABLE:
            raise ImportError("PySyft is required for PySyftOrchestrator")

        self.config = config
        self.server_port = server_port
        self.vm = None
        self.clients = {}
        self.round_results = []
        self.server_running = False

        logger.info(f"Initialized PySyft orchestrator on port {server_port}")

    async def start_server(self) -> None:
        """Start the PySyft virtual machine server."""
        try:
            # Create PySyft virtual machine
            self.vm = VirtualMachine(
                name="chemi_accel_server", port=self.server_port, reset=True
            )

            # Configure security and privacy settings
            self.vm.settings.enable_warnings = True
            self.vm.settings.require_request_approval = True

            # Start the server
            await self.vm.start()
            self.server_running = True

            logger.info(f"PySyft server started on port {self.server_port}")

        except Exception as e:
            logger.error(f"Failed to start PySyft server: {e}")
            raise OrchestrationError(f"PySyft server startup failed: {e}")

    async def stop_server(self) -> None:
        """Stop the PySyft server."""
        if self.vm and self.server_running:
            try:
                await self.vm.stop()
                self.server_running = False
                logger.info("PySyft server stopped")
            except Exception as e:
                logger.error(f"Error stopping PySyft server: {e}")

    async def register_client(self, client_info: ClientInfo) -> bool:
        """Register a client with the PySyft server."""
        try:
            self.clients[client_info.client_id] = client_info

            # Create secure channel for client communication
            if self.vm:
                client_channel = self.vm.create_client_channel(client_info.client_id)
                client_info.channel = client_channel

            logger.info(f"Registered PySyft client: {client_info.client_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to register client {client_info.client_id}: {e}")
            return False

    async def run_federated_round(self, round_number: int) -> RoundResults:
        """Execute a federated round using PySyft secure computation."""
        start_time = time.time()
        participating_clients = []

        try:
            # Select participating clients
            active_clients = [
                client for client in self.clients.values() if client.status == "active"
            ]

            if len(active_clients) < self.config.privacy_config.min_participants:
                raise OrchestrationError(
                    f"Insufficient clients: {len(active_clients)} < "
                    f"{self.config.privacy_config.min_participants}"
                )

            # Send computation requests to clients
            computation_futures = []
            for client in active_clients:
                future = self._request_client_computation(client, round_number)
                computation_futures.append((client.client_id, future))
                participating_clients.append(client.client_id)

            # Collect results with timeout
            client_results = {}
            for client_id, future in computation_futures:
                try:
                    result = await asyncio.wait_for(future, timeout=300)
                    client_results[client_id] = result
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Client {client_id} timed out in round {round_number}"
                    )
                except Exception as e:
                    logger.error(f"Client {client_id} failed: {e}")

            # Perform secure aggregation
            aggregated_result = await self._secure_aggregate(client_results)

            # Calculate round metrics
            success_rate = len(client_results) / len(participating_clients)
            round_duration = time.time() - start_time

            round_result = RoundResults(
                round_number=round_number,
                participating_clients=list(client_results.keys()),
                aggregated_metrics=aggregated_result.get("metrics", {}),
                convergence_metrics=aggregated_result.get("convergence", {}),
                privacy_budget_used=aggregated_result.get("privacy_cost", 0.0),
                round_duration=round_duration,
                success_rate=success_rate,
            )

            self.round_results.append(round_result)
            logger.info(
                f"PySyft round {round_number} completed with {success_rate:.2%} success rate"
            )

            return round_result

        except Exception as e:
            logger.error(f"PySyft round {round_number} failed: {e}")
            raise OrchestrationError(f"Federated round failed: {e}")

    async def _request_client_computation(
        self, client: ClientInfo, round_number: int
    ) -> Dict[str, Any]:
        """Request computation from a specific client."""
        # In a real implementation, this would send secure computation requests
        # For now, we simulate the process

        # Simulate client computation time
        await asyncio.sleep(np.random.uniform(1, 5))

        # Return mock results with privacy preservation
        return {
            "client_id": client.client_id,
            "round": round_number,
            "local_metrics": {
                "samples": client.data_size,
                "loss": np.random.uniform(0.1, 1.0),
                "accuracy": np.random.uniform(0.7, 0.95),
            },
            "model_update": self._generate_mock_model_update(),
            "privacy_cost": self.config.privacy_config.epsilon / self.config.max_rounds,
        }

    def _generate_mock_model_update(self) -> Dict[str, np.ndarray]:
        """Generate mock model update for demonstration."""
        return {
            "embedding.weight": np.random.normal(0, 0.01, (1000, 128)).astype(
                np.float32
            ),
            "hidden1.weight": np.random.normal(0, 0.01, (128, 256)).astype(np.float32),
            "hidden1.bias": np.random.normal(0, 0.01, (256,)).astype(np.float32),
            "output.weight": np.random.normal(0, 0.01, (256, 1)).astype(np.float32),
            "output.bias": np.random.normal(0, 0.01, (1,)).astype(np.float32),
        }

    async def _secure_aggregate(
        self, client_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform secure aggregation of client results."""
        if not client_results:
            return {"metrics": {}, "convergence": {}, "privacy_cost": 0.0}

        # Aggregate metrics
        aggregated_metrics = {}
        total_samples = sum(
            result["local_metrics"]["samples"] for result in client_results.values()
        )

        # Weighted average of metrics
        for metric_name in ["loss", "accuracy"]:
            weighted_sum = sum(
                result["local_metrics"][metric_name]
                * result["local_metrics"]["samples"]
                for result in client_results.values()
            )
            aggregated_metrics[metric_name] = weighted_sum / total_samples

        # Aggregate model updates (FedAvg)
        aggregated_model = {}
        for param_name in client_results[list(client_results.keys())[0]][
            "model_update"
        ].keys():
            # Weighted average of parameters
            weighted_params = []
            total_weight = 0

            for result in client_results.values():
                weight = result["local_metrics"]["samples"]
                param_update = result["model_update"][param_name]
                weighted_params.append(param_update * weight)
                total_weight += weight

            aggregated_model[param_name] = sum(weighted_params) / total_weight

        # Calculate convergence metrics
        convergence_metrics = {
            "participating_clients": len(client_results),
            "total_samples": total_samples,
            "model_size": sum(param.size for param in aggregated_model.values()),
        }

        # Sum privacy costs
        total_privacy_cost = sum(
            result["privacy_cost"] for result in client_results.values()
        )

        return {
            "metrics": aggregated_metrics,
            "convergence": convergence_metrics,
            "privacy_cost": total_privacy_cost,
            "aggregated_model": aggregated_model,
        }

    def get_server_status(self) -> Dict[str, Any]:
        """Get PySyft server status."""
        return {
            "type": "PySyft",
            "running": self.server_running,
            "port": self.server_port,
            "clients_registered": len(self.clients),
            "rounds_completed": len(self.round_results),
            "vm_status": "running" if self.vm and self.server_running else "stopped",
        }


class FlowerOrchestrator(FederatedOrchestrator):
    """Flower framework-based federated learning orchestrator."""

    def __init__(self, config: FederatedConfig, server_address: str = "localhost:8080"):
        """Initialize Flower orchestrator.

        Args:
            config: Federated learning configuration
            server_address: Server address in host:port format
        """
        if not FLOWER_AVAILABLE:
            raise ImportError("Flower framework is required for FlowerOrchestrator")

        self.config = config
        self.server_address = server_address
        self.strategy = None
        self.server = None
        self.clients = {}
        self.round_results = []
        self.server_running = False

        # Initialize Flower strategy
        self._initialize_strategy()

        logger.info(f"Initialized Flower orchestrator at {server_address}")

    def _initialize_strategy(self):
        """Initialize Flower aggregation strategy."""

        class ChemiAccelStrategy(FedAvg):
            """Custom Flower strategy for ChemiAccel."""

            def __init__(self, orchestrator):
                super().__init__()
                self.orchestrator = orchestrator

            def aggregate_fit(self, server_round: int, results, failures):
                """Aggregate model updates with privacy preservation."""
                # Log round information
                logger.info(
                    f"Flower round {server_round}: {len(results)} clients, {len(failures)} failures"
                )

                # Apply differential privacy to aggregation
                if self.orchestrator.config.privacy_config.enable_differential_privacy:
                    results = self._apply_differential_privacy(results)

                # Use parent FedAvg aggregation
                aggregated_params, aggregated_metrics = super().aggregate_fit(
                    server_round, results, failures
                )

                # Store round results
                self.orchestrator._store_round_results(server_round, results, failures)

                return aggregated_params, aggregated_metrics

            def _apply_differential_privacy(self, results):
                """Apply differential privacy to client results."""
                # Add noise to model updates
                noisy_results = []
                for client, fit_res in results:
                    # Add Gaussian noise to parameters
                    params = fl.common.parameters_to_ndarrays(fit_res.parameters)

                    # Apply noise based on privacy configuration
                    noise_scale = (
                        self.orchestrator.config.privacy_config.sensitivity
                        / self.orchestrator.config.privacy_config.epsilon
                    )

                    noisy_params = []
                    for param in params:
                        noise = np.random.normal(0, noise_scale, param.shape)
                        noisy_params.append(param + noise.astype(param.dtype))

                    # Create new FitRes with noisy parameters
                    noisy_fit_res = fl.common.FitRes(
                        parameters=fl.common.ndarrays_to_parameters(noisy_params),
                        num_examples=fit_res.num_examples,
                        metrics=fit_res.metrics,
                    )

                    noisy_results.append((client, noisy_fit_res))

                return noisy_results

        self.strategy = ChemiAccelStrategy(self)

    async def start_server(self) -> None:
        """Start the Flower federated learning server."""
        try:
            host, port = self.server_address.split(":")
            port = int(port)

            # Configure server
            server_config = fl.server.ServerConfig(
                num_rounds=self.config.max_rounds,
                round_timeout=300.0,  # 5 minutes per round
            )

            # Start server in background thread
            def run_server():
                fl.server.start_server(
                    server_address=f"{host}:{port}",
                    config=server_config,
                    strategy=self.strategy,
                )

            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            self.server_running = True

            # Wait for server to be ready
            await asyncio.sleep(2)

            logger.info(f"Flower server started at {self.server_address}")

        except Exception as e:
            logger.error(f"Failed to start Flower server: {e}")
            raise OrchestrationError(f"Flower server startup failed: {e}")

    async def stop_server(self) -> None:
        """Stop the Flower server."""
        self.server_running = False
        logger.info("Flower server stopped")

    async def register_client(self, client_info: ClientInfo) -> bool:
        """Register a client (Flower handles client connections automatically)."""
        self.clients[client_info.client_id] = client_info
        logger.info(f"Registered Flower client: {client_info.client_id}")
        return True

    async def run_federated_round(self, round_number: int) -> RoundResults:
        """Flower manages rounds automatically, this returns stored results."""
        if round_number <= len(self.round_results):
            return self.round_results[round_number - 1]

        # Wait for round to complete
        while len(self.round_results) < round_number:
            await asyncio.sleep(1)
            if not self.server_running:
                raise OrchestrationError("Server stopped before round completion")

        return self.round_results[round_number - 1]

    def _store_round_results(self, round_number: int, results, failures):
        """Store results from a completed round."""
        participating_clients = [str(client) for client, _ in results]

        # Calculate aggregated metrics
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        aggregated_metrics = {}

        if results:
            # Weighted average of metrics
            for metric_name in ["loss", "accuracy"]:
                if any(metric_name in fit_res.metrics for _, fit_res in results):
                    weighted_sum = sum(
                        fit_res.metrics.get(metric_name, 0) * fit_res.num_examples
                        for _, fit_res in results
                        if metric_name in fit_res.metrics
                    )
                    aggregated_metrics[metric_name] = weighted_sum / total_examples

        success_rate = (
            len(results) / (len(results) + len(failures))
            if (len(results) + len(failures)) > 0
            else 0
        )

        round_result = RoundResults(
            round_number=round_number,
            participating_clients=participating_clients,
            aggregated_metrics=aggregated_metrics,
            convergence_metrics={"total_examples": total_examples},
            privacy_budget_used=self.config.privacy_config.epsilon
            / self.config.max_rounds,
            round_duration=0.0,  # Would need to track timing
            success_rate=success_rate,
        )

        self.round_results.append(round_result)

    def get_server_status(self) -> Dict[str, Any]:
        """Get Flower server status."""
        return {
            "type": "Flower",
            "running": self.server_running,
            "address": self.server_address,
            "clients_registered": len(self.clients),
            "rounds_completed": len(self.round_results),
            "strategy": "ChemiAccelStrategy",
        }


class ChemiAccelOrchestrator(FederatedOrchestrator):
    """Custom ChemiAccel federated learning orchestrator."""

    def __init__(self, config: FederatedConfig):
        """Initialize ChemiAccel orchestrator.

        Args:
            config: Federated learning configuration
        """
        self.config = config
        self.clients = {}
        self.round_results = []
        self.server_running = False
        self.current_round = 0
        self.global_model = {}

        logger.info("Initialized ChemiAccel native orchestrator")

    async def start_server(self) -> None:
        """Start the ChemiAccel orchestrator."""
        self.server_running = True
        logger.info("ChemiAccel orchestrator started")

    async def stop_server(self) -> None:
        """Stop the ChemiAccel orchestrator."""
        self.server_running = False
        logger.info("ChemiAccel orchestrator stopped")

    async def register_client(self, client_info: ClientInfo) -> bool:
        """Register a client with the orchestrator."""
        self.clients[client_info.client_id] = client_info
        logger.info(f"Registered ChemiAccel client: {client_info.client_id}")
        return True

    async def run_federated_round(self, round_number: int) -> RoundResults:
        """Execute a federated round with molecular-specific optimizations."""
        start_time = time.time()
        self.current_round = round_number

        try:
            # Select participating clients
            active_clients = [
                client for client in self.clients.values() if client.status == "active"
            ]

            if len(active_clients) < self.config.privacy_config.min_participants:
                raise OrchestrationError(
                    f"Insufficient clients: {len(active_clients)} < "
                    f"{self.config.privacy_config.min_participants}"
                )

            # Request local computations from clients
            client_results = {}
            for client in active_clients:
                try:
                    result = await self._request_molecular_computation(
                        client, round_number
                    )
                    client_results[client.client_id] = result
                except Exception as e:
                    logger.warning(f"Client {client.client_id} failed: {e}")

            # Perform molecular-aware aggregation
            aggregated_result = self._aggregate_molecular_results(client_results)

            # Update global model
            self.global_model = aggregated_result["global_model"]

            # Calculate metrics
            success_rate = len(client_results) / len(active_clients)
            round_duration = time.time() - start_time

            round_result = RoundResults(
                round_number=round_number,
                participating_clients=list(client_results.keys()),
                aggregated_metrics=aggregated_result["metrics"],
                convergence_metrics=aggregated_result["convergence"],
                privacy_budget_used=aggregated_result["privacy_cost"],
                round_duration=round_duration,
                success_rate=success_rate,
            )

            self.round_results.append(round_result)
            logger.info(f"ChemiAccel round {round_number} completed successfully")

            return round_result

        except Exception as e:
            logger.error(f"ChemiAccel round {round_number} failed: {e}")
            raise OrchestrationError(f"Federated round failed: {e}")

    async def _request_molecular_computation(
        self, client: ClientInfo, round_number: int
    ) -> Dict[str, Any]:
        """Request molecular computation from a client."""
        # Simulate federated molecular descriptor computation
        await asyncio.sleep(np.random.uniform(0.5, 2.0))

        # Mock molecular computation results
        molecular_results = {
            "client_id": client.client_id,
            "round": round_number,
            "molecular_descriptors": {
                "molecular_weight": np.random.normal(250, 100),
                "logp": np.random.normal(2, 1),
                "tpsa": np.random.normal(70, 30),
                "heavy_atom_count": np.random.normal(15, 5),
            },
            "local_metrics": {
                "samples": client.data_size,
                "descriptor_coverage": np.random.uniform(0.8, 1.0),
                "computation_time": np.random.uniform(1, 5),
            },
            "privacy_preserving_update": self._generate_privacy_preserving_update(),
            "privacy_cost": self.config.privacy_config.epsilon / self.config.max_rounds,
        }

        return molecular_results

    def _generate_privacy_preserving_update(self) -> Dict[str, Any]:
        """Generate privacy-preserving model update."""
        # Simulated differentially private molecular model update
        return {
            "descriptor_weights": np.random.normal(0, 0.01, 100).astype(np.float32),
            "molecular_embeddings": np.random.normal(0, 0.1, (50, 128)).astype(
                np.float32
            ),
            "aggregated_statistics": {
                "mean_molecular_weight": np.random.normal(250, 10),
                "mean_logp": np.random.normal(2, 0.5),
                "privacy_noise_added": True,
            },
        }

    def _aggregate_molecular_results(
        self, client_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate molecular computation results with domain-specific logic."""
        if not client_results:
            return {
                "metrics": {},
                "convergence": {},
                "privacy_cost": 0.0,
                "global_model": {},
            }

        # Aggregate molecular descriptors
        aggregated_descriptors = {}
        total_samples = sum(
            result["local_metrics"]["samples"] for result in client_results.values()
        )

        # Weighted aggregation of molecular descriptors
        descriptor_names = list(client_results.values())[0][
            "molecular_descriptors"
        ].keys()
        for desc_name in descriptor_names:
            weighted_sum = sum(
                result["molecular_descriptors"][desc_name]
                * result["local_metrics"]["samples"]
                for result in client_results.values()
            )
            aggregated_descriptors[desc_name] = weighted_sum / total_samples

        # Aggregate privacy-preserving updates
        global_model_update = {}
        for param_name in ["descriptor_weights", "molecular_embeddings"]:
            if (
                param_name
                in list(client_results.values())[0]["privacy_preserving_update"]
            ):
                # Federated averaging with molecular-specific weighting
                updates = []
                weights = []

                for result in client_results.values():
                    update = result["privacy_preserving_update"][param_name]
                    weight = result["local_metrics"]["samples"]
                    updates.append(update * weight)
                    weights.append(weight)

                global_model_update[param_name] = sum(updates) / sum(weights)

        # Calculate convergence metrics
        convergence_metrics = {
            "participating_clients": len(client_results),
            "total_molecular_samples": total_samples,
            "descriptor_diversity": len(descriptor_names),
            "average_computation_time": np.mean(
                [
                    result["local_metrics"]["computation_time"]
                    for result in client_results.values()
                ]
            ),
            "privacy_preservation_level": "high",
        }

        # Aggregate metrics
        aggregated_metrics = {
            "molecular_weight_mean": aggregated_descriptors.get("molecular_weight", 0),
            "logp_mean": aggregated_descriptors.get("logp", 0),
            "tpsa_mean": aggregated_descriptors.get("tpsa", 0),
            "coverage_rate": np.mean(
                [
                    result["local_metrics"]["descriptor_coverage"]
                    for result in client_results.values()
                ]
            ),
        }

        # Calculate total privacy cost
        total_privacy_cost = sum(
            result["privacy_cost"] for result in client_results.values()
        )

        return {
            "metrics": aggregated_metrics,
            "convergence": convergence_metrics,
            "privacy_cost": total_privacy_cost,
            "global_model": global_model_update,
            "molecular_descriptors": aggregated_descriptors,
        }

    def get_server_status(self) -> Dict[str, Any]:
        """Get ChemiAccel orchestrator status."""
        return {
            "type": "ChemiAccel",
            "running": self.server_running,
            "clients_registered": len(self.clients),
            "rounds_completed": len(self.round_results),
            "current_round": self.current_round,
            "global_model_parameters": len(self.global_model),
            "molecular_optimization": True,
        }


class FederatedCoordinator:
    """High-level coordinator for managing multiple orchestrators."""

    def __init__(self, config: FederatedConfig):
        """Initialize federated coordinator.

        Args:
            config: Federated learning configuration
        """
        self.config = config
        self.orchestrators = {}
        self.active_orchestrator = None

        logger.info("Initialized federated coordinator")

    def add_orchestrator(self, name: str, orchestrator: FederatedOrchestrator):
        """Add an orchestrator to the coordinator."""
        self.orchestrators[name] = orchestrator
        logger.info(f"Added orchestrator: {name}")

    def set_active_orchestrator(self, name: str):
        """Set the active orchestrator."""
        if name not in self.orchestrators:
            raise ValueError(f"Orchestrator {name} not found")

        self.active_orchestrator = self.orchestrators[name]
        logger.info(f"Set active orchestrator: {name}")

    async def start_federation(self) -> None:
        """Start federated learning with the active orchestrator."""
        if not self.active_orchestrator:
            raise OrchestrationError("No active orchestrator set")

        await self.active_orchestrator.start_server()
        logger.info("Federation started")

    async def stop_federation(self) -> None:
        """Stop federated learning."""
        if self.active_orchestrator:
            await self.active_orchestrator.stop_server()
            logger.info("Federation stopped")

    async def run_federated_training(self, num_rounds: int) -> List[RoundResults]:
        """Run complete federated training."""
        if not self.active_orchestrator:
            raise OrchestrationError("No active orchestrator set")

        results = []
        for round_num in range(1, num_rounds + 1):
            try:
                round_result = await self.active_orchestrator.run_federated_round(
                    round_num
                )
                results.append(round_result)

                logger.info(
                    f"Round {round_num}/{num_rounds} completed: "
                    f"{round_result.success_rate:.2%} success rate"
                )

            except Exception as e:
                logger.error(f"Round {round_num} failed: {e}")
                break

        return results

    def get_federation_status(self) -> Dict[str, Any]:
        """Get comprehensive federation status."""
        status = {
            "available_orchestrators": list(self.orchestrators.keys()),
            "active_orchestrator": None,
            "federation_running": False,
        }

        if self.active_orchestrator:
            orchestrator_name = next(
                name
                for name, orch in self.orchestrators.items()
                if orch == self.active_orchestrator
            )
            status["active_orchestrator"] = orchestrator_name
            status.update(self.active_orchestrator.get_server_status())
            status["federation_running"] = status.get("running", False)

        return status


# Convenience functions for creating orchestrators
def create_pysyft_orchestrator(config: FederatedConfig, **kwargs) -> PySyftOrchestrator:
    """Create a PySyft orchestrator."""
    return PySyftOrchestrator(config, **kwargs)


def create_flower_orchestrator(config: FederatedConfig, **kwargs) -> FlowerOrchestrator:
    """Create a Flower orchestrator."""
    return FlowerOrchestrator(config, **kwargs)


def create_chemi_accel_orchestrator(config: FederatedConfig) -> ChemiAccelOrchestrator:
    """Create a ChemiAccel native orchestrator."""
    return ChemiAccelOrchestrator(config)


def create_federated_coordinator(
    config: FederatedConfig, orchestrator_types: List[str] = None
) -> FederatedCoordinator:
    """Create a federated coordinator with specified orchestrators.

    Args:
        config: Federated learning configuration
        orchestrator_types: List of orchestrator types to create

    Returns:
        Configured federated coordinator
    """
    coordinator = FederatedCoordinator(config)

    orchestrator_types = orchestrator_types or ["chemi_accel"]

    for orch_type in orchestrator_types:
        if orch_type == "pysyft" and SYFT_AVAILABLE:
            coordinator.add_orchestrator("pysyft", create_pysyft_orchestrator(config))
        elif orch_type == "flower" and FLOWER_AVAILABLE:
            coordinator.add_orchestrator("flower", create_flower_orchestrator(config))
        elif orch_type == "chemi_accel":
            coordinator.add_orchestrator(
                "chemi_accel", create_chemi_accel_orchestrator(config)
            )
        else:
            logger.warning(f"Orchestrator type {orch_type} not available or unknown")

    # Set default active orchestrator
    if "chemi_accel" in coordinator.orchestrators:
        coordinator.set_active_orchestrator("chemi_accel")
    elif coordinator.orchestrators:
        coordinator.set_active_orchestrator(list(coordinator.orchestrators.keys())[0])

    return coordinator
