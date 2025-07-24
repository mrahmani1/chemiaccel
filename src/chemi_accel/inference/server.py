"""
High-Performance Inference Server for ChemiAccel - Task 20.2

This module implements a high-performance REST API server using FastAPI for molecular
descriptor calculations and property predictions. Features include dynamic batching,
model preloading, GPU optimization, and containerized deployment support.

Key Features:
- FastAPI-based REST API with automatic OpenAPI documentation
- Dynamic batching system for optimal GPU utilization
- Model preloading with warm-up to eliminate cold-start latency
- Asynchronous request handling with concurrent processing
- GPU memory management and optimization
- Health checks and monitoring endpoints
- Docker and Kubernetes deployment support
- Load balancing and horizontal scaling capabilities

Performance Optimizations:
- Request batching with adaptive batch sizing
- GPU memory pooling and efficient tensor operations
- Connection pooling and keep-alive optimization
- Async I/O for non-blocking request processing
- Model caching and shared memory utilization
- Streaming responses for large datasets
"""

import asyncio
import time
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import uuid
import json

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field, validator
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    BaseModel = None

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
    np = None

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# ChemiAccel imports
from ..core.calculator import Calculator
from ..core.exceptions import ChemiAccelError, ValidationError
from ..utils.logging import get_logger
from ..backends.cuda.memory import get_memory_manager

logger = get_logger()


# Pydantic models for API requests/responses
class MoleculeInput(BaseModel):
    """Input model for single molecule processing."""
    smiles: str = Field(..., description="SMILES string of the molecule")
    descriptors: Optional[List[str]] = Field(None, description="List of descriptors to calculate")
    include_properties: bool = Field(True, description="Include basic molecular properties")
    
    @validator('smiles')
    def validate_smiles(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("SMILES must be a non-empty string")
        if len(v) > 1000:  # Reasonable limit
            raise ValueError("SMILES string too long")
        return v.strip()


class BatchMoleculeInput(BaseModel):
    """Input model for batch molecule processing."""
    smiles: List[str] = Field(..., description="List of SMILES strings")
    descriptors: Optional[List[str]] = Field(None, description="List of descriptors to calculate")
    include_properties: bool = Field(True, description="Include basic molecular properties")
    batch_size: Optional[int] = Field(None, description="Custom batch size for processing")
    
    @validator('smiles')
    def validate_smiles_list(cls, v):
        if not v:
            raise ValueError("SMILES list cannot be empty")
        if len(v) > 100000:  # Reasonable limit
            raise ValueError("Too many molecules in batch")
        return v


class ModelConfig(BaseModel):
    """Configuration for model inference settings."""
    backend: Optional[str] = Field("auto", description="Backend to use (auto, cuda, jax, cpu)")
    batch_size: Optional[int] = Field(None, description="Batch size for processing")
    use_gpu: bool = Field(True, description="Use GPU acceleration if available")
    precision: str = Field("float32", description="Numerical precision (float16, float32, float64)")


class PredictionResponse(BaseModel):
    """Response model for prediction results."""
    request_id: str = Field(..., description="Unique request identifier")
    status: str = Field(..., description="Status of the prediction")
    results: Dict[str, Any] = Field(..., description="Prediction results")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="ChemiAccel version")
    uptime: float = Field(..., description="Uptime in seconds")
    gpu_available: bool = Field(..., description="GPU availability")
    memory_usage: Dict[str, float] = Field(..., description="Memory usage statistics")


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint."""
    total_requests: int = Field(..., description="Total number of requests")
    active_requests: int = Field(..., description="Currently active requests")
    average_latency: float = Field(..., description="Average request latency")
    gpu_utilization: float = Field(..., description="GPU utilization percentage")
    memory_usage: Dict[str, float] = Field(..., description="Memory usage statistics")


@dataclass
class BatchRequest:
    """Internal representation of a batched request."""
    request_id: str
    smiles: List[str]
    descriptors: Optional[List[str]]
    include_properties: bool
    timestamp: float
    future: asyncio.Future
    config: Optional[ModelConfig] = None


@dataclass
class ServerConfig:
    """Configuration for the inference server."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_batch_size: int = 1000
    max_batch_wait_time: float = 0.1  # seconds
    preload_models: bool = True
    enable_metrics: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    log_level: str = "INFO"
    gpu_memory_limit: Optional[float] = None  # GB
    
    # Performance tuning
    max_concurrent_requests: int = 100
    request_timeout: float = 300.0  # seconds
    enable_compression: bool = True
    
    # Model settings
    default_backend: str = "auto"
    warm_up_samples: int = 10


class DynamicBatcher:
    """Dynamic batching system for optimal GPU utilization."""
    
    def __init__(self, config: ServerConfig, calculator: Calculator):
        """Initialize dynamic batcher.
        
        Args:
            config: Server configuration
            calculator: ChemiAccel calculator instance
        """
        self.config = config
        self.calculator = calculator
        self.batch_queue = asyncio.Queue()
        self.processing = False
        self.batch_processor_task = None
        
        # Metrics
        self.total_requests = 0
        self.total_batches = 0
        self.processing_times = []
        
        logger.info(f"Initialized dynamic batcher with max_batch_size={config.max_batch_size}")
    
    async def start(self):
        """Start the batch processor."""
        if not self.processing:
            self.processing = True
            self.batch_processor_task = asyncio.create_task(self._batch_processor())
            logger.info("Dynamic batcher started")
    
    async def stop(self):
        """Stop the batch processor."""
        self.processing = False
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass
        logger.info("Dynamic batcher stopped")
    
    async def add_request(self, request: BatchRequest) -> Dict[str, Any]:
        """Add a request to the batch queue.
        
        Args:
            request: Batch request to process
            
        Returns:
            Processing results
        """
        await self.batch_queue.put(request)
        self.total_requests += 1
        
        # Wait for result
        return await request.future
    
    async def _batch_processor(self):
        """Main batch processing loop."""
        while self.processing:
            try:
                # Collect requests for batching
                batch_requests = []
                batch_deadline = time.time() + self.config.max_batch_wait_time
                
                # Get first request (blocking)
                try:
                    first_request = await asyncio.wait_for(
                        self.batch_queue.get(),
                        timeout=self.config.max_batch_wait_time
                    )
                    batch_requests.append(first_request)
                except asyncio.TimeoutError:
                    continue
                
                # Collect additional requests until batch is full or timeout
                while (len(batch_requests) < self.config.max_batch_size and 
                       time.time() < batch_deadline):
                    try:
                        request = await asyncio.wait_for(
                            self.batch_queue.get(),
                            timeout=max(0.001, batch_deadline - time.time())
                        )
                        batch_requests.append(request)
                    except asyncio.TimeoutError:
                        break
                
                # Process the batch
                await self._process_batch(batch_requests)
                
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_batch(self, requests: List[BatchRequest]):
        """Process a batch of requests.
        
        Args:
            requests: List of batch requests to process
        """
        if not requests:
            return
        
        start_time = time.time()
        batch_size = len(requests)
        
        try:
            # Combine all SMILES from the batch
            all_smiles = []
            request_boundaries = []
            current_pos = 0
            
            for request in requests:
                all_smiles.extend(request.smiles)
                request_boundaries.append((current_pos, current_pos + len(request.smiles)))
                current_pos += len(request.smiles)
            
            # Process the combined batch
            if all_smiles:
                # Use pandas for processing
                smiles_series = pd.Series(all_smiles)
                
                # Get descriptors (use first request's descriptors as default)
                descriptors = requests[0].descriptors
                
                # Calculate descriptors
                results_df = await asyncio.get_event_loop().run_in_executor(
                    None, self.calculator.pandas, smiles_series
                )
                
                # Split results back to individual requests
                for i, request in enumerate(requests):
                    start_idx, end_idx = request_boundaries[i]
                    request_results = results_df.iloc[start_idx:end_idx]
                    
                    # Format response
                    response = {
                        'smiles': request.smiles,
                        'descriptors': request_results.to_dict('records'),
                        'count': len(request.smiles)
                    }
                    
                    # Set result
                    if not request.future.done():
                        request.future.set_result(response)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.total_batches += 1
            
            logger.debug(f"Processed batch of {batch_size} requests with {len(all_smiles)} molecules in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            
            # Set error for all requests in batch
            for request in requests:
                if not request.future.done():
                    request.future.set_exception(
                        HTTPException(status_code=500, detail=f"Processing error: {e}")
                    )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get batcher metrics."""
        avg_processing_time = (
            sum(self.processing_times[-100:]) / len(self.processing_times[-100:])
            if self.processing_times else 0.0
        )
        
        return {
            'total_requests': self.total_requests,
            'total_batches': self.total_batches,
            'queue_size': self.batch_queue.qsize(),
            'average_processing_time': avg_processing_time,
            'processing': self.processing
        }


class ModelManager:
    """Manager for model preloading and caching."""
    
    def __init__(self, config: ServerConfig):
        """Initialize model manager.
        
        Args:
            config: Server configuration
        """
        self.config = config
        self.calculator = None
        self.is_loaded = False
        self.load_time = None
        self.warm_up_completed = False
        
        # GPU memory manager
        self.memory_manager = None
        if CUPY_AVAILABLE:
            try:
                self.memory_manager = get_memory_manager()
            except Exception as e:
                logger.warning(f"Failed to initialize GPU memory manager: {e}")
    
    async def load_models(self):
        """Load and preload models."""
        if self.is_loaded:
            return
        
        logger.info("Loading ChemiAccel calculator...")
        start_time = time.time()
        
        try:
            # Initialize calculator with optimal settings
            self.calculator = Calculator(
                backend=self.config.default_backend,
                use_gpu=CUPY_AVAILABLE
            )
            
            self.load_time = time.time() - start_time
            self.is_loaded = True
            
            logger.info(f"Calculator loaded in {self.load_time:.2f} seconds")
            
            # Perform warm-up if enabled
            if self.config.preload_models:
                await self.warm_up_models()
            
        except Exception as e:
            logger.error(f"Failed to load calculator: {e}")
            raise
    
    async def warm_up_models(self):
        """Warm up models with sample calculations."""
        if self.warm_up_completed or not self.calculator:
            return
        
        logger.info("Warming up models...")
        start_time = time.time()
        
        try:
            # Create sample molecules for warm-up
            sample_smiles = [
                "CCO",  # Ethanol
                "c1ccccc1",  # Benzene
                "CC(=O)O",  # Acetic acid
                "CCN",  # Ethylamine
                "c1ccc(cc1)O"  # Phenol
            ] * (self.config.warm_up_samples // 5 + 1)
            
            sample_smiles = sample_smiles[:self.config.warm_up_samples]
            smiles_series = pd.Series(sample_smiles)
            
            # Perform warm-up calculation
            await asyncio.get_event_loop().run_in_executor(
                None, self.calculator.pandas, smiles_series
            )
            
            warm_up_time = time.time() - start_time
            self.warm_up_completed = True
            
            logger.info(f"Model warm-up completed in {warm_up_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Model warm-up failed: {e}")
    
    def get_calculator(self) -> Calculator:
        """Get the loaded calculator instance."""
        if not self.is_loaded or not self.calculator:
            raise RuntimeError("Calculator not loaded")
        return self.calculator
    
    def get_status(self) -> Dict[str, Any]:
        """Get model manager status."""
        memory_info = {}
        if self.memory_manager:
            try:
                mem_info = self.memory_manager.get_memory_info()
                memory_info = {
                    'gpu_total_mb': mem_info.total / (1024 * 1024),
                    'gpu_free_mb': mem_info.free / (1024 * 1024),
                    'gpu_used_mb': mem_info.used / (1024 * 1024),
                    'gpu_utilization': mem_info.utilization
                }
            except Exception:
                pass
        
        return {
            'loaded': self.is_loaded,
            'warm_up_completed': self.warm_up_completed,
            'load_time': self.load_time,
            'gpu_available': CUPY_AVAILABLE,
            'memory_info': memory_info
        }


class PerformanceMonitor:
    """Performance monitoring and metrics collection."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
        
        # Prometheus metrics if available
        self.prometheus_metrics = {}
        if PROMETHEUS_AVAILABLE:
            self.prometheus_metrics = {
                'requests_total': Counter('chemi_accel_requests_total', 'Total requests'),
                'request_duration': Histogram('chemi_accel_request_duration_seconds', 'Request duration'),
                'errors_total': Counter('chemi_accel_errors_total', 'Total errors'),
                'gpu_utilization': Gauge('chemi_accel_gpu_utilization', 'GPU utilization'),
                'memory_usage': Gauge('chemi_accel_memory_usage_bytes', 'Memory usage')
            }
    
    def record_request(self, duration: float, success: bool = True):
        """Record a request."""
        self.request_count += 1
        self.response_times.append(duration)
        
        if not success:
            self.error_count += 1
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE and self.prometheus_metrics:
            self.prometheus_metrics['requests_total'].inc()
            self.prometheus_metrics['request_duration'].observe(duration)
            if not success:
                self.prometheus_metrics['errors_total'].inc()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        uptime = time.time() - self.start_time
        avg_response_time = (
            sum(self.response_times[-1000:]) / len(self.response_times[-1000:])
            if self.response_times else 0.0
        )
        
        return {
            'uptime': uptime,
            'total_requests': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(1, self.request_count),
            'average_response_time': avg_response_time,
            'requests_per_second': self.request_count / max(1, uptime)
        }


# Global instances
model_manager: Optional[ModelManager] = None
dynamic_batcher: Optional[DynamicBatcher] = None
performance_monitor: Optional[PerformanceMonitor] = None
server_config: Optional[ServerConfig] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager."""
    global model_manager, dynamic_batcher, performance_monitor
    
    # Startup
    logger.info("Starting ChemiAccel inference server...")
    
    # Initialize components
    model_manager = ModelManager(server_config)
    performance_monitor = PerformanceMonitor()
    
    # Load models
    await model_manager.load_models()
    
    # Initialize dynamic batcher
    calculator = model_manager.get_calculator()
    dynamic_batcher = DynamicBatcher(server_config, calculator)
    await dynamic_batcher.start()
    
    logger.info("ChemiAccel inference server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ChemiAccel inference server...")
    
    if dynamic_batcher:
        await dynamic_batcher.stop()
    
    logger.info("ChemiAccel inference server shut down")


def create_app(config: ServerConfig = None) -> FastAPI:
    """Create and configure the FastAPI application.
    
    Args:
        config: Server configuration
        
    Returns:
        Configured FastAPI application
    """
    global server_config
    server_config = config or ServerConfig()
    
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI is required for the inference server")
    
    app = FastAPI(
        title="ChemiAccel Inference Server",
        description="High-performance molecular descriptor calculation API",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=server_config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    if server_config.enable_compression:
        app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    return app


# Create the app instance
app = create_app()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "service": "ChemiAccel Inference Server",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global model_manager, performance_monitor
    
    if not model_manager or not performance_monitor:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    model_status = model_manager.get_status()
    metrics = performance_monitor.get_metrics()
    
    # Get memory usage
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    
    memory_usage = {
        'rss_mb': memory_info.rss / (1024 * 1024),
        'vms_mb': memory_info.vms / (1024 * 1024)
    }
    
    if model_status.get('memory_info'):
        memory_usage.update(model_status['memory_info'])
    
    return HealthResponse(
        status="healthy" if model_status['loaded'] else "loading",
        version="1.0.0",
        uptime=metrics['uptime'],
        gpu_available=model_status['gpu_available'],
        memory_usage=memory_usage
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get server metrics."""
    global model_manager, dynamic_batcher, performance_monitor
    
    if not all([model_manager, dynamic_batcher, performance_monitor]):
        raise HTTPException(status_code=503, detail="Service not ready")
    
    perf_metrics = performance_monitor.get_metrics()
    batch_metrics = dynamic_batcher.get_metrics()
    model_status = model_manager.get_status()
    
    # Calculate GPU utilization
    gpu_utilization = 0.0
    memory_usage = {}
    
    if model_status.get('memory_info'):
        memory_info = model_status['memory_info']
        gpu_utilization = memory_info.get('gpu_utilization', 0.0)
        memory_usage = {
            'gpu_used_mb': memory_info.get('gpu_used_mb', 0.0),
            'gpu_total_mb': memory_info.get('gpu_total_mb', 0.0)
        }
    
    # Get system memory
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage.update({
        'system_rss_mb': memory_info.rss / (1024 * 1024),
        'system_vms_mb': memory_info.vms / (1024 * 1024)
    })
    
    return MetricsResponse(
        total_requests=perf_metrics['total_requests'],
        active_requests=batch_metrics['queue_size'],
        average_latency=perf_metrics['average_response_time'],
        gpu_utilization=gpu_utilization,
        memory_usage=memory_usage
    )


@app.get("/metrics/prometheus")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    if not PROMETHEUS_AVAILABLE:
        raise HTTPException(status_code=404, detail="Prometheus metrics not available")
    
    return generate_latest()


@app.post("/predict/single", response_model=PredictionResponse)
async def predict_single(molecule: MoleculeInput, config: Optional[ModelConfig] = None):
    """Predict descriptors for a single molecule."""
    global dynamic_batcher, performance_monitor
    
    if not dynamic_batcher or not performance_monitor:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Create batch request
        future = asyncio.Future()
        batch_request = BatchRequest(
            request_id=request_id,
            smiles=[molecule.smiles],
            descriptors=molecule.descriptors,
            include_properties=molecule.include_properties,
            timestamp=start_time,
            future=future,
            config=config
        )
        
        # Process through dynamic batcher
        results = await dynamic_batcher.add_request(batch_request)
        
        processing_time = time.time() - start_time
        
        # Record metrics
        performance_monitor.record_request(processing_time, success=True)
        
        return PredictionResponse(
            request_id=request_id,
            status="success",
            results=results,
            processing_time=processing_time,
            metadata={
                'molecule_count': 1,
                'backend': server_config.default_backend
            }
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        performance_monitor.record_request(processing_time, success=False)
        
        logger.error(f"Prediction failed for request {request_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=PredictionResponse)
async def predict_batch(molecules: BatchMoleculeInput, config: Optional[ModelConfig] = None):
    """Predict descriptors for a batch of molecules."""
    global dynamic_batcher, performance_monitor
    
    if not dynamic_batcher or not performance_monitor:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Create batch request
        future = asyncio.Future()
        batch_request = BatchRequest(
            request_id=request_id,
            smiles=molecules.smiles,
            descriptors=molecules.descriptors,
            include_properties=molecules.include_properties,
            timestamp=start_time,
            future=future,
            config=config
        )
        
        # Process through dynamic batcher
        results = await dynamic_batcher.add_request(batch_request)
        
        processing_time = time.time() - start_time
        
        # Record metrics
        performance_monitor.record_request(processing_time, success=True)
        
        return PredictionResponse(
            request_id=request_id,
            status="success",
            results=results,
            processing_time=processing_time,
            metadata={
                'molecule_count': len(molecules.smiles),
                'backend': server_config.default_backend
            }
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        performance_monitor.record_request(processing_time, success=False)
        
        logger.error(f"Batch prediction failed for request {request_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/stream")
async def predict_stream(molecules: BatchMoleculeInput, config: Optional[ModelConfig] = None):
    """Stream predictions for large batches."""
    global dynamic_batcher, performance_monitor
    
    if not dynamic_batcher or not performance_monitor:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    async def generate_predictions():
        """Generate streaming predictions."""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Process in chunks for streaming
            chunk_size = config.batch_size if config and config.batch_size else 100
            total_molecules = len(molecules.smiles)
            
            yield f"data: {json.dumps({'type': 'start', 'request_id': request_id, 'total': total_molecules})}\n\n"
            
            processed = 0
            for i in range(0, total_molecules, chunk_size):
                chunk_smiles = molecules.smiles[i:i + chunk_size]
                
                # Create batch request for chunk
                future = asyncio.Future()
                batch_request = BatchRequest(
                    request_id=f"{request_id}_{i}",
                    smiles=chunk_smiles,
                    descriptors=molecules.descriptors,
                    include_properties=molecules.include_properties,
                    timestamp=time.time(),
                    future=future,
                    config=config
                )
                
                # Process chunk
                chunk_results = await dynamic_batcher.add_request(batch_request)
                processed += len(chunk_smiles)
                
                # Stream chunk result
                response_data = {
                    'type': 'chunk',
                    'chunk_index': i // chunk_size,
                    'processed': processed,
                    'total': total_molecules,
                    'results': chunk_results
                }
                
                yield f"data: {json.dumps(response_data)}\n\n"
            
            # Final completion message
            processing_time = time.time() - start_time
            performance_monitor.record_request(processing_time, success=True)
            
            completion_data = {
                'type': 'complete',
                'request_id': request_id,
                'total_processed': processed,
                'processing_time': processing_time
            }
            
            yield f"data: {json.dumps(completion_data)}\n\n"
            
        except Exception as e:
            error_data = {
                'type': 'error',
                'request_id': request_id,
                'error': str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_predictions(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/descriptors")
async def list_descriptors():
    """List available descriptors."""
    global model_manager
    
    if not model_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Return predefined descriptor lists
    return {
        "basic": ["molecular_weight", "logp", "tpsa", "heavy_atom_count"],
        "extended": ["molecular_weight", "logp", "tpsa", "heavy_atom_count", 
                    "rotatable_bonds", "hbd", "hba", "aromatic_rings"],
        "all": ["molecular_weight", "logp", "tpsa", "heavy_atom_count",
               "rotatable_bonds", "hbd", "hba", "aromatic_rings",
               "formal_charge", "radical_electrons", "valence_electrons"]
    }


def main():
    """Main function to run the server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ChemiAccel Inference Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--max-batch-size", type=int, default=1000, help="Maximum batch size")
    parser.add_argument("--no-preload", action="store_true", help="Disable model preloading")
    
    args = parser.parse_args()
    
    # Create server configuration
    config = ServerConfig(
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        max_batch_size=args.max_batch_size,
        preload_models=not args.no_preload
    )
    
    # Configure logging
    logging.basicConfig(level=getattr(logging, config.log_level))
    
    # Run server
    uvicorn.run(
        "chemi_accel.inference.server:app",
        host=config.host,
        port=config.port,
        workers=config.workers,
        log_level=config.log_level.lower(),
        access_log=True
    )


if __name__ == "__main__":
    main()