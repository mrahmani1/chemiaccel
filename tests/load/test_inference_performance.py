"""
Load Testing and Performance Optimization for ChemiAccel Inference Server.

This module provides comprehensive load testing capabilities for the inference server,
including stress testing, latency measurement, throughput analysis, and performance
optimization recommendations.

Features:
- Multi-threaded load testing with configurable concurrency
- Realistic molecular dataset generation for testing
- Latency and throughput measurement
- Memory and GPU utilization monitoring
- Performance bottleneck identification
- Automated optimization recommendations
- Load testing reports and visualizations
"""

import asyncio
import json
import queue
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import asyncio

    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    import nvidia_ml_py3 as nvml
    import psutil

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

import numpy as np
from chemi_accel.utils.logging import get_logger

logger = get_logger()


@dataclass
class LoadTestConfig:
    """Configuration for load testing."""

    # Server configuration
    server_url: str = "http://localhost:8000"
    timeout: float = 300.0

    # Load test parameters
    concurrent_users: int = 10
    requests_per_user: int = 100
    ramp_up_time: float = 30.0  # seconds
    test_duration: float = 300.0  # seconds

    # Request configuration
    single_molecule_ratio: float = 0.3  # 30% single molecule requests
    batch_sizes: List[int] = field(default_factory=lambda: [1, 5, 10, 50, 100])
    use_streaming: bool = False

    # Dataset configuration
    molecule_pool_size: int = 10000
    include_complex_molecules: bool = True
    error_rate: float = 0.05  # 5% invalid molecules

    # Monitoring configuration
    monitor_system: bool = True
    monitor_gpu: bool = True
    sample_interval: float = 1.0  # seconds


@dataclass
class RequestResult:
    """Result of a single request."""

    request_id: str
    request_type: str  # 'single', 'batch', 'stream'
    molecule_count: int
    start_time: float
    end_time: float
    status_code: int
    success: bool
    error_message: Optional[str] = None
    response_size: int = 0

    @property
    def latency(self) -> float:
        """Request latency in seconds."""
        return self.end_time - self.start_time

    @property
    def throughput(self) -> float:
        """Molecules per second."""
        return self.molecule_count / self.latency if self.latency > 0 else 0.0


@dataclass
class SystemMetrics:
    """System performance metrics at a point in time."""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    gpu_utilization: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0


@dataclass
class LoadTestResults:
    """Results of a load test run."""

    config: LoadTestConfig
    start_time: float
    end_time: float
    request_results: List[RequestResult]
    system_metrics: List[SystemMetrics]

    @property
    def total_duration(self) -> float:
        """Total test duration in seconds."""
        return self.end_time - self.start_time

    @property
    def total_requests(self) -> int:
        """Total number of requests."""
        return len(self.request_results)

    @property
    def successful_requests(self) -> int:
        """Number of successful requests."""
        return len([r for r in self.request_results if r.success])

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        return (self.successful_requests / max(1, self.total_requests)) * 100

    @property
    def average_latency(self) -> float:
        """Average request latency in seconds."""
        latencies = [r.latency for r in self.request_results if r.success]
        return statistics.mean(latencies) if latencies else 0.0

    @property
    def p95_latency(self) -> float:
        """95th percentile latency."""
        latencies = [r.latency for r in self.request_results if r.success]
        return np.percentile(latencies, 95) if latencies else 0.0

    @property
    def requests_per_second(self) -> float:
        """Average requests per second."""
        return self.total_requests / max(1, self.total_duration)

    @property
    def total_molecules_processed(self) -> int:
        """Total molecules processed successfully."""
        return sum(r.molecule_count for r in self.request_results if r.success)

    @property
    def molecules_per_second(self) -> float:
        """Average molecules processed per second."""
        return self.total_molecules_processed / max(1, self.total_duration)


class MoleculeDataGenerator:
    """Generate realistic molecular datasets for load testing."""

    def __init__(self, pool_size: int = 10000, include_complex: bool = True):
        """Initialize molecule data generator.

        Args:
            pool_size: Size of molecule pool to generate
            include_complex: Include complex molecules
        """
        self.pool_size = pool_size
        self.include_complex = include_complex
        self.molecule_pool = self._generate_molecule_pool()

        logger.info(f"Generated molecule pool with {len(self.molecule_pool)} molecules")

    def _generate_molecule_pool(self) -> List[str]:
        """Generate a pool of molecules for testing."""
        # Basic molecules
        basic_molecules = [
            "CCO",  # Ethanol
            "c1ccccc1",  # Benzene
            "CC(=O)O",  # Acetic acid
            "CCN",  # Ethylamine
            "c1ccc(cc1)O",  # Phenol
            "CC(C)O",  # Isopropanol
            "c1ccc2c(c1)cccn2",  # Quinoline
            "CC1=CC=CC=C1",  # Toluene
            "CCc1ccccc1",  # Ethylbenzene
            "c1ccc(cc1)N",  # Aniline
            "CC(=O)C",  # Acetone
            "CCCC",  # Butane
            "c1ccc(cc1)C(=O)O",  # Benzoic acid
            "CCc1ccc(cc1)O",  # 4-Ethylphenol
            "CC(C)C",  # Isobutane
            "CCN(CC)CC",  # Triethylamine
            "c1ccc(cc1)C=O",  # Benzaldehyde
        ]

        # Complex molecules (if enabled)
        complex_molecules = [
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            "CC1=C(C(CCC1)(C)C)CC/C(=C/CO)/C",  # Vitamin A precursor
            "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # Ibuprofen
            "CN1CCC[C@H]1c2cccnc2",  # Nicotine
            "CC1=C(C=C(C=C1)C(=O)NC2=CC=C(C=C2)O)CC(C)C",  # Drug-like
            "C[C@H]1[C@@H]([C@H]([C@@H]([C@H](O1)O)O)O)O",  # Sugar
        ]

        # Error molecules (for testing error handling)
        error_molecules = [
            "invalid_smiles",
            "",
            "C" * 200,  # Very long
            "[Pt][Au][Ag]",  # Unusual metals
        ]

        pool = basic_molecules * (self.pool_size // len(basic_molecules) // 3)

        if self.include_complex:
            pool.extend(
                complex_molecules * (self.pool_size // len(complex_molecules) // 3)
            )

        # Add some errors
        error_count = int(self.pool_size * 0.05)
        pool.extend(random.choices(error_molecules, k=error_count))

        # Fill remaining with variations
        while len(pool) < self.pool_size:
            base_mol = random.choice(basic_molecules)
            # Add simple variations
            if "c1ccccc1" in base_mol:
                substituents = ["C", "O", "N", "Cl", "F"]
                sub = random.choice(substituents)
                base_mol = base_mol.replace("c1ccccc1", f"c1ccc(cc1){sub}")
            pool.append(base_mol)

        random.shuffle(pool)
        return pool[: self.pool_size]

    def get_random_molecules(self, count: int) -> List[str]:
        """Get random molecules from the pool.

        Args:
            count: Number of molecules to return

        Returns:
            List of SMILES strings
        """
        return random.choices(self.molecule_pool, k=count)

    def get_single_molecule(self) -> str:
        """Get a single random molecule."""
        return random.choice(self.molecule_pool)


class SystemMonitor:
    """Monitor system and GPU performance during load testing."""

    def __init__(self, config: LoadTestConfig):
        """Initialize system monitor.

        Args:
            config: Load test configuration
        """
        self.config = config
        self.monitoring = False
        self.metrics: List[SystemMetrics] = []
        self.monitor_thread: Optional[threading.Thread] = None

        # Initialize GPU monitoring if available
        self.gpu_available = False
        if config.monitor_gpu and MONITORING_AVAILABLE:
            try:
                nvml.nvmlInit()
                self.gpu_available = True
                logger.info("GPU monitoring initialized")
            except Exception as e:
                logger.warning(f"GPU monitoring not available: {e}")

    def start_monitoring(self):
        """Start system monitoring."""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        logger.info("System monitoring started")

    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

        logger.info("System monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics.append(metrics)
                time.sleep(self.config.sample_interval)
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                time.sleep(self.config.sample_interval)

    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        timestamp = time.time()

        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)

        # Network
        net_io = psutil.net_io_counters()
        network_bytes_sent = net_io.bytes_sent
        network_bytes_recv = net_io.bytes_recv

        # GPU metrics
        gpu_utilization = 0.0
        gpu_memory_used_gb = 0.0
        gpu_memory_total_gb = 0.0

        if self.gpu_available:
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = nvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = gpu_util.gpu

                gpu_mem = nvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_used_gb = gpu_mem.used / (1024**3)
                gpu_memory_total_gb = gpu_mem.total / (1024**3)
            except Exception:
                pass

        return SystemMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_gb=memory_available_gb,
            gpu_utilization=gpu_utilization,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_memory_total_gb=gpu_memory_total_gb,
            network_bytes_sent=network_bytes_sent,
            network_bytes_recv=network_bytes_recv,
        )

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        if not self.metrics:
            return {}

        cpu_usage = [m.cpu_percent for m in self.metrics]
        memory_usage = [m.memory_percent for m in self.metrics]
        gpu_usage = [m.gpu_utilization for m in self.metrics if m.gpu_utilization > 0]

        summary = {
            "cpu_avg": statistics.mean(cpu_usage),
            "cpu_max": max(cpu_usage),
            "memory_avg": statistics.mean(memory_usage),
            "memory_max": max(memory_usage),
            "samples_collected": len(self.metrics),
        }

        if gpu_usage:
            summary.update(
                {
                    "gpu_avg": statistics.mean(gpu_usage),
                    "gpu_max": max(gpu_usage),
                    "gpu_memory_max_gb": max(
                        m.gpu_memory_used_gb for m in self.metrics
                    ),
                }
            )

        return summary


class LoadTester:
    """Main load testing engine."""

    def __init__(self, config: LoadTestConfig):
        """Initialize load tester.

        Args:
            config: Load test configuration
        """
        self.config = config
        self.molecule_generator = MoleculeDataGenerator(
            config.molecule_pool_size, config.include_complex_molecules
        )
        self.system_monitor = SystemMonitor(config)
        self.results: List[RequestResult] = []
        self.result_queue = queue.Queue()

        logger.info(f"Load tester initialized for {config.server_url}")

    async def run_async_load_test(self) -> LoadTestResults:
        """Run asynchronous load test."""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp is required for async load testing")

        logger.info("Starting async load test...")
        start_time = time.time()

        # Start system monitoring
        if self.config.monitor_system:
            self.system_monitor.start_monitoring()

        try:
            # Create aiohttp session
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Create user tasks
                user_tasks = []
                for user_id in range(self.config.concurrent_users):
                    # Stagger user start times for ramp-up
                    delay = (
                        user_id / self.config.concurrent_users
                    ) * self.config.ramp_up_time
                    task = asyncio.create_task(
                        self._async_user_session(session, user_id, delay)
                    )
                    user_tasks.append(task)

                # Wait for all users to complete
                await asyncio.gather(*user_tasks, return_exceptions=True)

        finally:
            # Stop monitoring
            if self.config.monitor_system:
                self.system_monitor.stop_monitoring()

        end_time = time.time()

        # Collect results from queue
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())

        logger.info(
            f"Async load test completed: {len(results)} requests in {end_time - start_time:.2f}s"
        )

        return LoadTestResults(
            config=self.config,
            start_time=start_time,
            end_time=end_time,
            request_results=results,
            system_metrics=self.system_monitor.metrics,
        )

    async def _async_user_session(
        self, session: aiohttp.ClientSession, user_id: int, delay: float
    ):
        """Simulate a single user's session."""
        # Wait for ramp-up delay
        if delay > 0:
            await asyncio.sleep(delay)

        session_start = time.time()
        requests_made = 0

        while (
            requests_made < self.config.requests_per_user
            and time.time() - session_start < self.config.test_duration
        ):
            try:
                # Decide request type
                if random.random() < self.config.single_molecule_ratio:
                    # Single molecule request
                    result = await self._make_single_request(session, user_id)
                else:
                    # Batch request
                    batch_size = random.choice(self.config.batch_sizes)
                    if self.config.use_streaming and batch_size > 50:
                        result = await self._make_stream_request(
                            session, user_id, batch_size
                        )
                    else:
                        result = await self._make_batch_request(
                            session, user_id, batch_size
                        )

                self.result_queue.put(result)
                requests_made += 1

                # Small delay between requests
                await asyncio.sleep(random.uniform(0.1, 1.0))

            except Exception as e:
                logger.error(f"User {user_id} request failed: {e}")
                # Create error result
                error_result = RequestResult(
                    request_id=f"user_{user_id}_req_{requests_made}",
                    request_type="error",
                    molecule_count=0,
                    start_time=time.time(),
                    end_time=time.time(),
                    status_code=0,
                    success=False,
                    error_message=str(e),
                )
                self.result_queue.put(error_result)
                requests_made += 1

    async def _make_single_request(
        self, session: aiohttp.ClientSession, user_id: int
    ) -> RequestResult:
        """Make a single molecule prediction request."""
        request_id = f"user_{user_id}_single_{time.time()}"
        smiles = self.molecule_generator.get_single_molecule()

        payload = {"smiles": smiles, "include_properties": True}

        start_time = time.time()

        async with session.post(
            f"{self.config.server_url}/predict/single", json=payload
        ) as response:
            response_text = await response.text()
            end_time = time.time()

            success = response.status == 200
            response_size = len(response_text)

            return RequestResult(
                request_id=request_id,
                request_type="single",
                molecule_count=1,
                start_time=start_time,
                end_time=end_time,
                status_code=response.status,
                success=success,
                response_size=response_size,
                error_message=None if success else response_text,
            )

    async def _make_batch_request(
        self, session: aiohttp.ClientSession, user_id: int, batch_size: int
    ) -> RequestResult:
        """Make a batch prediction request."""
        request_id = f"user_{user_id}_batch_{batch_size}_{time.time()}"
        smiles_list = self.molecule_generator.get_random_molecules(batch_size)

        payload = {"smiles": smiles_list, "include_properties": True}

        start_time = time.time()

        async with session.post(
            f"{self.config.server_url}/predict/batch", json=payload
        ) as response:
            response_text = await response.text()
            end_time = time.time()

            success = response.status == 200
            response_size = len(response_text)

            return RequestResult(
                request_id=request_id,
                request_type="batch",
                molecule_count=batch_size,
                start_time=start_time,
                end_time=end_time,
                status_code=response.status,
                success=success,
                response_size=response_size,
                error_message=None if success else response_text,
            )

    async def _make_stream_request(
        self, session: aiohttp.ClientSession, user_id: int, batch_size: int
    ) -> RequestResult:
        """Make a streaming prediction request."""
        request_id = f"user_{user_id}_stream_{batch_size}_{time.time()}"
        smiles_list = self.molecule_generator.get_random_molecules(batch_size)

        payload = {"smiles": smiles_list, "include_properties": True}

        start_time = time.time()

        try:
            async with session.post(
                f"{self.config.server_url}/predict/stream", json=payload
            ) as response:
                response_size = 0
                async for chunk in response.content.iter_chunked(1024):
                    response_size += len(chunk)

                end_time = time.time()
                success = response.status == 200

                return RequestResult(
                    request_id=request_id,
                    request_type="stream",
                    molecule_count=batch_size,
                    start_time=start_time,
                    end_time=end_time,
                    status_code=response.status,
                    success=success,
                    response_size=response_size,
                )

        except Exception as e:
            end_time = time.time()
            return RequestResult(
                request_id=request_id,
                request_type="stream",
                molecule_count=batch_size,
                start_time=start_time,
                end_time=end_time,
                status_code=0,
                success=False,
                error_message=str(e),
            )

    def run_sync_load_test(self) -> LoadTestResults:
        """Run synchronous load test using requests library."""
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("requests library is required for sync load testing")

        logger.info("Starting sync load test...")
        start_time = time.time()

        # Start system monitoring
        if self.config.monitor_system:
            self.system_monitor.start_monitoring()

        try:
            # Create thread pool for concurrent users
            with ThreadPoolExecutor(
                max_workers=self.config.concurrent_users
            ) as executor:
                # Submit user tasks
                futures = []
                for user_id in range(self.config.concurrent_users):
                    delay = (
                        user_id / self.config.concurrent_users
                    ) * self.config.ramp_up_time
                    future = executor.submit(self._sync_user_session, user_id, delay)
                    futures.append(future)

                # Wait for completion
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"User session failed: {e}")

        finally:
            # Stop monitoring
            if self.config.monitor_system:
                self.system_monitor.stop_monitoring()

        end_time = time.time()

        # Collect results
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())

        logger.info(
            f"Sync load test completed: {len(results)} requests in {end_time - start_time:.2f}s"
        )

        return LoadTestResults(
            config=self.config,
            start_time=start_time,
            end_time=end_time,
            request_results=results,
            system_metrics=self.system_monitor.metrics,
        )

    def _sync_user_session(self, user_id: int, delay: float):
        """Synchronous user session."""
        if delay > 0:
            time.sleep(delay)

        session = requests.Session()
        session_start = time.time()
        requests_made = 0

        while (
            requests_made < self.config.requests_per_user
            and time.time() - session_start < self.config.test_duration
        ):
            try:
                # Decide request type and make request
                if random.random() < self.config.single_molecule_ratio:
                    result = self._sync_single_request(session, user_id)
                else:
                    batch_size = random.choice(self.config.batch_sizes)
                    result = self._sync_batch_request(session, user_id, batch_size)

                self.result_queue.put(result)
                requests_made += 1

                # Small delay
                time.sleep(random.uniform(0.1, 1.0))

            except Exception as e:
                logger.error(f"Sync user {user_id} request failed: {e}")
                error_result = RequestResult(
                    request_id=f"user_{user_id}_sync_error_{requests_made}",
                    request_type="error",
                    molecule_count=0,
                    start_time=time.time(),
                    end_time=time.time(),
                    status_code=0,
                    success=False,
                    error_message=str(e),
                )
                self.result_queue.put(error_result)
                requests_made += 1

    def _sync_single_request(
        self, session: requests.Session, user_id: int
    ) -> RequestResult:
        """Synchronous single request."""
        request_id = f"user_{user_id}_sync_single_{time.time()}"
        smiles = self.molecule_generator.get_single_molecule()

        payload = {"smiles": smiles, "include_properties": True}

        start_time = time.time()
        response = session.post(
            f"{self.config.server_url}/predict/single",
            json=payload,
            timeout=self.config.timeout,
        )
        end_time = time.time()

        return RequestResult(
            request_id=request_id,
            request_type="single",
            molecule_count=1,
            start_time=start_time,
            end_time=end_time,
            status_code=response.status_code,
            success=response.status_code == 200,
            response_size=len(response.content),
            error_message=None if response.status_code == 200 else response.text,
        )

    def _sync_batch_request(
        self, session: requests.Session, user_id: int, batch_size: int
    ) -> RequestResult:
        """Synchronous batch request."""
        request_id = f"user_{user_id}_sync_batch_{batch_size}_{time.time()}"
        smiles_list = self.molecule_generator.get_random_molecules(batch_size)

        payload = {"smiles": smiles_list, "include_properties": True}

        start_time = time.time()
        response = session.post(
            f"{self.config.server_url}/predict/batch",
            json=payload,
            timeout=self.config.timeout,
        )
        end_time = time.time()

        return RequestResult(
            request_id=request_id,
            request_type="batch",
            molecule_count=batch_size,
            start_time=start_time,
            end_time=end_time,
            status_code=response.status_code,
            success=response.status_code == 200,
            response_size=len(response.content),
            error_message=None if response.status_code == 200 else response.text,
        )


class LoadTestAnalyzer:
    """Analyze load test results and generate reports."""

    def __init__(self, results: LoadTestResults):
        """Initialize analyzer with test results.

        Args:
            results: Load test results to analyze
        """
        self.results = results

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of load test results."""
        # Basic statistics
        successful_results = [r for r in self.results.request_results if r.success]
        latencies = [r.latency for r in successful_results]
        throughputs = [r.throughput for r in successful_results if r.throughput > 0]

        # Request type breakdown
        request_types = {}
        for result in self.results.request_results:
            req_type = result.request_type
            if req_type not in request_types:
                request_types[req_type] = {"count": 0, "success": 0}
            request_types[req_type]["count"] += 1
            if result.success:
                request_types[req_type]["success"] += 1

        # Latency percentiles
        latency_percentiles = {}
        if latencies:
            for p in [50, 75, 90, 95, 99]:
                latency_percentiles[f"p{p}"] = np.percentile(latencies, p)

        # System metrics summary
        system_summary = {}
        if self.results.system_metrics:
            cpu_usage = [m.cpu_percent for m in self.results.system_metrics]
            memory_usage = [m.memory_percent for m in self.results.system_metrics]
            gpu_usage = [
                m.gpu_utilization
                for m in self.results.system_metrics
                if m.gpu_utilization > 0
            ]

            system_summary = {
                "cpu_avg": statistics.mean(cpu_usage),
                "cpu_max": max(cpu_usage),
                "memory_avg": statistics.mean(memory_usage),
                "memory_max": max(memory_usage),
            }

            if gpu_usage:
                system_summary.update(
                    {"gpu_avg": statistics.mean(gpu_usage), "gpu_max": max(gpu_usage)}
                )

        summary = {
            "test_config": {
                "concurrent_users": self.results.config.concurrent_users,
                "requests_per_user": self.results.config.requests_per_user,
                "test_duration": self.results.total_duration,
                "server_url": self.results.config.server_url,
            },
            "performance_metrics": {
                "total_requests": self.results.total_requests,
                "successful_requests": self.results.successful_requests,
                "success_rate": self.results.success_rate,
                "requests_per_second": self.results.requests_per_second,
                "molecules_per_second": self.results.molecules_per_second,
                "average_latency": self.results.average_latency,
                "p95_latency": self.results.p95_latency,
            },
            "latency_percentiles": latency_percentiles,
            "request_types": request_types,
            "system_metrics": system_summary,
            "throughput_stats": {
                "avg_molecules_per_second": (
                    statistics.mean(throughputs) if throughputs else 0
                ),
                "max_molecules_per_second": max(throughputs) if throughputs else 0,
            },
        }

        return summary

    def generate_optimization_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        # Check success rate
        if self.results.success_rate < 95:
            recommendations.append(
                f"Low success rate ({self.results.success_rate:.1f}%). "
                "Consider reducing concurrent users or increasing server resources."
            )

        # Check latency
        if self.results.p95_latency > 5.0:
            recommendations.append(
                f"High P95 latency ({self.results.p95_latency:.2f}s). "
                "Consider optimizing model inference or increasing batch sizes."
            )

        # Check system resources
        if self.results.system_metrics:
            cpu_max = max(m.cpu_percent for m in self.results.system_metrics)
            memory_max = max(m.memory_percent for m in self.results.system_metrics)

            if cpu_max > 90:
                recommendations.append(
                    f"High CPU usage ({cpu_max:.1f}%). "
                    "Consider scaling horizontally or optimizing CPU-bound operations."
                )

            if memory_max > 90:
                recommendations.append(
                    f"High memory usage ({memory_max:.1f}%). "
                    "Consider implementing memory optimization or increasing available RAM."
                )

            # GPU recommendations
            gpu_metrics = [
                m for m in self.results.system_metrics if m.gpu_utilization > 0
            ]
            if gpu_metrics:
                gpu_max = max(m.gpu_utilization for m in gpu_metrics)
                if gpu_max > 95:
                    recommendations.append(
                        f"Very high GPU utilization ({gpu_max:.1f}%). "
                        "Consider adding more GPUs or optimizing GPU memory usage."
                    )
                elif gpu_max < 50:
                    recommendations.append(
                        f"Low GPU utilization ({gpu_max:.1f}%). "
                        "Consider increasing batch sizes or optimizing GPU usage."
                    )

        # Check throughput
        if self.results.molecules_per_second < 100:
            recommendations.append(
                f"Low throughput ({self.results.molecules_per_second:.1f} mol/s). "
                "Consider optimizing the inference pipeline or using larger batch sizes."
            )

        return recommendations

    def save_results(self, output_dir: str = "load_test_results"):
        """Save detailed results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save summary report
        summary = self.generate_summary_report()
        summary_file = output_path / f"load_test_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Save detailed results
        results_data = []
        for result in self.results.request_results:
            results_data.append(
                {
                    "request_id": result.request_id,
                    "request_type": result.request_type,
                    "molecule_count": result.molecule_count,
                    "latency": result.latency,
                    "throughput": result.throughput,
                    "status_code": result.status_code,
                    "success": result.success,
                    "response_size": result.response_size,
                    "error_message": result.error_message,
                }
            )

        results_file = output_path / f"load_test_details_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

        # Save system metrics
        if self.results.system_metrics:
            metrics_data = []
            for metric in self.results.system_metrics:
                metrics_data.append(
                    {
                        "timestamp": metric.timestamp,
                        "cpu_percent": metric.cpu_percent,
                        "memory_percent": metric.memory_percent,
                        "memory_available_gb": metric.memory_available_gb,
                        "gpu_utilization": metric.gpu_utilization,
                        "gpu_memory_used_gb": metric.gpu_memory_used_gb,
                        "gpu_memory_total_gb": metric.gpu_memory_total_gb,
                    }
                )

            metrics_file = output_path / f"system_metrics_{timestamp}.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics_data, f, indent=2)

        # Generate recommendations
        recommendations = self.generate_optimization_recommendations()
        recommendations_file = output_path / f"recommendations_{timestamp}.txt"
        with open(recommendations_file, "w") as f:
            f.write("Load Test Performance Recommendations\n")
            f.write("=" * 50 + "\n\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n\n")

        logger.info(f"Load test results saved to {output_path}")

        return {
            "summary_file": str(summary_file),
            "details_file": str(results_file),
            "metrics_file": str(metrics_file) if self.results.system_metrics else None,
            "recommendations_file": str(recommendations_file),
        }


async def run_comprehensive_load_test(
    server_url: str = "http://localhost:8000",
    concurrent_users: int = 10,
    duration: float = 300.0,
) -> LoadTestResults:
    """Run a comprehensive load test with default settings.

    Args:
        server_url: URL of the inference server
        concurrent_users: Number of concurrent users
        duration: Test duration in seconds

    Returns:
        Load test results
    """
    config = LoadTestConfig(
        server_url=server_url,
        concurrent_users=concurrent_users,
        test_duration=duration,
        requests_per_user=1000,
        monitor_system=True,
        monitor_gpu=True,
    )

    tester = LoadTester(config)

    if AIOHTTP_AVAILABLE:
        results = await tester.run_async_load_test()
    else:
        results = tester.run_sync_load_test()

    # Analyze and save results
    analyzer = LoadTestAnalyzer(results)
    files_saved = analyzer.save_results()

    # Print summary
    summary = analyzer.generate_summary_report()
    print("\n" + "=" * 60)
    print("LOAD TEST SUMMARY")
    print("=" * 60)
    print(f"Total Requests: {summary['performance_metrics']['total_requests']}")
    print(f"Success Rate: {summary['performance_metrics']['success_rate']:.1f}%")
    print(f"Requests/sec: {summary['performance_metrics']['requests_per_second']:.1f}")
    print(
        f"Molecules/sec: {summary['performance_metrics']['molecules_per_second']:.1f}"
    )
    print(f"Avg Latency: {summary['performance_metrics']['average_latency']:.3f}s")
    print(f"P95 Latency: {summary['performance_metrics']['p95_latency']:.3f}s")

    if summary["system_metrics"]:
        print(
            f"CPU Usage: {summary['system_metrics']['cpu_avg']:.1f}% avg, {summary['system_metrics']['cpu_max']:.1f}% max"
        )
        print(
            f"Memory Usage: {summary['system_metrics']['memory_avg']:.1f}% avg, {summary['system_metrics']['memory_max']:.1f}% max"
        )
        if "gpu_avg" in summary["system_metrics"]:
            print(
                f"GPU Usage: {summary['system_metrics']['gpu_avg']:.1f}% avg, {summary['system_metrics']['gpu_max']:.1f}% max"
            )

    recommendations = analyzer.generate_optimization_recommendations()
    if recommendations:
        print("\nRECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

    print(f"\nDetailed results saved to: {files_saved['summary_file']}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ChemiAccel Inference Server Load Test"
    )
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--users", type=int, default=10, help="Concurrent users")
    parser.add_argument(
        "--duration", type=float, default=300, help="Test duration in seconds"
    )
    parser.add_argument(
        "--async", action="store_true", help="Use async testing", dest="use_async"
    )

    args = parser.parse_args()

    if args.use_async and AIOHTTP_AVAILABLE:
        asyncio.run(run_comprehensive_load_test(args.server, args.users, args.duration))
    else:
        # Run sync version
        config = LoadTestConfig(
            server_url=args.server,
            concurrent_users=args.users,
            test_duration=args.duration,
        )
        tester = LoadTester(config)
        results = tester.run_sync_load_test()

        analyzer = LoadTestAnalyzer(results)
        analyzer.save_results()

        summary = analyzer.generate_summary_report()
        print(json.dumps(summary, indent=2, default=str))
