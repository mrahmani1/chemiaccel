"""
CUDA Streaming System for Large Dataset Processing.

This module provides streaming capabilities to process datasets larger than
GPU memory by implementing efficient data streaming and batch processing.
"""

import logging
from typing import Iterator, List, Any, Optional, Callable, Union, Tuple, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import time

try:
    import cupy as cp
    import numpy as np
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None
    np = None

# Type definitions for when CUDA is not available
if not CUDA_AVAILABLE:
    from typing import Any
    class MockArray:
        pass
    # Create mock modules for type annotations
    class MockCp:
        ndarray = MockArray
    class MockNp:
        ndarray = MockArray
    if cp is None:
        cp = MockCp()
    if np is None:
        np = MockNp()

from .memory import get_memory_manager, CUDAMemoryManager
from ...core.exceptions import GPUMemoryError
from ...utils.logging import get_logger

logger = get_logger()


@dataclass
class StreamingConfig:
    """Configuration for streaming operations."""
    batch_size: Optional[int] = None  # Auto-optimize if None
    num_streams: int = 2  # Number of CUDA streams
    prefetch_batches: int = 2  # Number of batches to prefetch
    max_memory_usage: float = 0.8  # Maximum GPU memory usage
    overlap_computation: bool = True  # Overlap data transfer and computation


class CUDAStream:
    """Wrapper for CUDA stream operations."""
    
    def __init__(self, stream_id: int):
        """Initialize CUDA stream.
        
        Args:
            stream_id: Unique stream identifier
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")
        
        self.stream_id = stream_id
        self._stream = cp.cuda.Stream()
        self._event = cp.cuda.Event()
        self._busy = False
        
    def __enter__(self):
        """Enter stream context."""
        self._stream.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit stream context."""
        self._stream.__exit__(exc_type, exc_val, exc_tb)
    
    def synchronize(self):
        """Synchronize stream."""
        self._stream.synchronize()
    
    def record_event(self):
        """Record event in stream."""
        self._event.record(self._stream)
    
    def wait_event(self, event):
        """Wait for event in stream."""
        self._stream.wait_event(event)
    
    @property
    def is_busy(self) -> bool:
        """Check if stream is busy."""
        return self._busy
    
    def set_busy(self, busy: bool):
        """Set stream busy status."""
        self._busy = busy


class DataStreamer:
    """Handles streaming of large datasets to GPU."""
    
    def __init__(self, 
                 config: StreamingConfig,
                 memory_manager: Optional[CUDAMemoryManager] = None):
        """Initialize data streamer.
        
        Args:
            config: Streaming configuration
            memory_manager: CUDA memory manager instance
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")
        
        self.config = config
        self.memory_manager = memory_manager or get_memory_manager()
        
        # Initialize CUDA streams
        self._streams = [CUDAStream(i) for i in range(config.num_streams)]
        self._current_stream = 0
        
        # Threading for async operations
        self._executor = ThreadPoolExecutor(max_workers=config.num_streams)
        self._prefetch_queue = queue.Queue(maxsize=config.prefetch_batches)
        self._stop_prefetch = threading.Event()
        
        logger.info(f"Initialized data streamer with {config.num_streams} streams")
    
    def _get_next_stream(self) -> CUDAStream:
        """Get next available CUDA stream."""
        # Simple round-robin selection
        stream = self._streams[self._current_stream]
        self._current_stream = (self._current_stream + 1) % len(self._streams)
        return stream
    
    def _estimate_batch_memory(self, sample_data: Any) -> int:
        """Estimate memory required for a batch item.
        
        Args:
            sample_data: Sample data item
            
        Returns:
            Estimated memory in bytes
        """
        if isinstance(sample_data, np.ndarray):
            return sample_data.nbytes
        elif isinstance(sample_data, (list, tuple)):
            # Estimate based on first few items
            total_size = 0
            for item in sample_data[:min(10, len(sample_data))]:
                if isinstance(item, np.ndarray):
                    total_size += item.nbytes
                elif isinstance(item, str):
                    total_size += len(item.encode('utf-8'))
                else:
                    total_size += 1024  # Conservative estimate
            return total_size * len(sample_data) // min(10, len(sample_data))
        else:
            return 1024  # Conservative default
    
    def _optimize_batch_size(self, data_iterator: Iterator[Any]) -> int:
        """Optimize batch size based on available memory.
        
        Args:
            data_iterator: Iterator over data items
            
        Returns:
            Optimal batch size
        """
        if self.config.batch_size is not None:
            return self.config.batch_size
        
        # Sample first item to estimate memory requirements
        try:
            sample_item = next(data_iterator)
            item_memory = self._estimate_batch_memory(sample_item)
            
            # Get optimal batch size from memory manager
            optimal_size = self.memory_manager.optimize_batch_size(
                single_item_memory=item_memory,
                max_batch_size=10000
            )
            
            logger.info(f"Auto-optimized batch size: {optimal_size}")
            return optimal_size
            
        except StopIteration:
            logger.warning("Empty data iterator, using default batch size")
            return 1000
    
    def _create_batches(self, 
                       data_iterator: Iterator[Any], 
                       batch_size: int) -> Iterator[List[Any]]:
        """Create batches from data iterator.
        
        Args:
            data_iterator: Iterator over data items
            batch_size: Size of each batch
            
        Yields:
            Batches of data items
        """
        batch = []
        for item in data_iterator:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        # Yield remaining items
        if batch:
            yield batch
    
    def _transfer_to_gpu(self, batch: List[Any], stream: CUDAStream) -> cp.ndarray:
        """Transfer batch to GPU memory.
        
        Args:
            batch: Batch of data items
            stream: CUDA stream for transfer
            
        Returns:
            GPU array
        """
        with stream:
            # Convert batch to numpy array if needed
            if isinstance(batch[0], np.ndarray):
                cpu_array = np.stack(batch)
            else:
                # Handle other data types (strings, etc.)
                cpu_array = np.array(batch)
            
            # Transfer to GPU
            gpu_array = cp.asarray(cpu_array)
            stream.record_event()
            
            return gpu_array
    
    def _prefetch_worker(self, 
                        batch_iterator: Iterator[List[Any]],
                        process_fn: Callable[[cp.ndarray], Any]) -> None:
        """Worker function for prefetching and processing batches.
        
        Args:
            batch_iterator: Iterator over batches
            process_fn: Function to process GPU batches
        """
        try:
            for batch in batch_iterator:
                if self._stop_prefetch.is_set():
                    break
                
                # Get next available stream
                stream = self._get_next_stream()
                
                # Wait for stream to be available
                while stream.is_busy:
                    time.sleep(0.001)  # Small delay
                
                stream.set_busy(True)
                
                try:
                    # Transfer to GPU
                    gpu_batch = self._transfer_to_gpu(batch, stream)
                    
                    # Process batch
                    with stream:
                        result = process_fn(gpu_batch)
                        stream.synchronize()
                    
                    # Add result to queue
                    self._prefetch_queue.put((result, len(batch)), timeout=10)
                    
                finally:
                    stream.set_busy(False)
                    
        except Exception as e:
            logger.error(f"Error in prefetch worker: {e}")
            self._prefetch_queue.put((None, 0))  # Signal error
    
    def stream_process(self, 
                      data_iterator: Iterator[Any],
                      process_fn: Callable[[cp.ndarray], Any],
                      batch_size: Optional[int] = None) -> Iterator[Tuple[Any, int]]:
        """Stream process large dataset.
        
        Args:
            data_iterator: Iterator over data items
            process_fn: Function to process GPU batches
            batch_size: Batch size (auto-optimize if None)
            
        Yields:
            Tuples of (processed_result, batch_size)
        """
        # Optimize batch size if not provided
        if batch_size is None:
            batch_size = self._optimize_batch_size(data_iterator)
        
        # Create batch iterator
        batch_iterator = self._create_batches(data_iterator, batch_size)
        
        if self.config.overlap_computation:
            # Use async processing with prefetching
            self._stop_prefetch.clear()
            
            # Start prefetch worker
            future = self._executor.submit(
                self._prefetch_worker, batch_iterator, process_fn
            )
            
            try:
                while True:
                    try:
                        result, batch_len = self._prefetch_queue.get(timeout=30)
                        if result is None:  # Error signal
                            break
                        yield result, batch_len
                    except queue.Empty:
                        logger.warning("Prefetch queue timeout")
                        break
            finally:
                self._stop_prefetch.set()
                future.result(timeout=10)  # Wait for worker to finish
        else:
            # Synchronous processing
            for batch in batch_iterator:
                stream = self._get_next_stream()
                
                with stream:
                    gpu_batch = self._transfer_to_gpu(batch, stream)
                    result = process_fn(gpu_batch)
                    stream.synchronize()
                
                yield result, len(batch)
    
    def close(self):
        """Close streamer and cleanup resources."""
        self._stop_prefetch.set()
        self._executor.shutdown(wait=True)
        
        # Synchronize all streams
        for stream in self._streams:
            stream.synchronize()
        
        logger.info("Closed data streamer")


class StreamingProcessor:
    """High-level interface for streaming molecular data processing."""
    
    def __init__(self, 
                 config: Optional[StreamingConfig] = None,
                 memory_manager: Optional[CUDAMemoryManager] = None):
        """Initialize streaming processor.
        
        Args:
            config: Streaming configuration
            memory_manager: CUDA memory manager instance
        """
        self.config = config or StreamingConfig()
        self.memory_manager = memory_manager or get_memory_manager()
        self._streamer: Optional[DataStreamer] = None
    
    def process_large_dataset(self,
                            data_iterator: Iterator[Any],
                            process_fn: Callable[[cp.ndarray], Any],
                            batch_size: Optional[int] = None,
                            progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Any]:
        """Process large dataset with streaming.
        
        Args:
            data_iterator: Iterator over data items
            process_fn: Function to process GPU batches
            batch_size: Batch size (auto-optimize if None)
            progress_callback: Optional progress callback (processed, total)
            
        Returns:
            List of processed results
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")
        
        # Initialize streamer
        self._streamer = DataStreamer(self.config, self.memory_manager)
        
        results = []
        total_processed = 0
        
        try:
            # Process data in streaming fashion
            for result, batch_len in self._streamer.stream_process(
                data_iterator, process_fn, batch_size
            ):
                results.append(result)
                total_processed += batch_len
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(total_processed, -1)  # -1 for unknown total
                
                logger.debug(f"Processed batch of {batch_len} items "
                           f"(total: {total_processed})")
            
            logger.info(f"Completed streaming processing of {total_processed} items")
            return results
            
        finally:
            if self._streamer:
                self._streamer.close()
    
    def estimate_memory_requirements(self, 
                                   sample_data: Any,
                                   dataset_size: int) -> Dict[str, int]:
        """Estimate memory requirements for dataset.
        
        Args:
            sample_data: Sample data item
            dataset_size: Total dataset size
            
        Returns:
            Dictionary with memory estimates
        """
        if not self._streamer:
            self._streamer = DataStreamer(self.config, self.memory_manager)
        
        item_memory = self._streamer._estimate_batch_memory(sample_data)
        total_memory = item_memory * dataset_size
        
        mem_info = self.memory_manager.get_memory_info()
        optimal_batch_size = self.memory_manager.optimize_batch_size(item_memory)
        
        return {
            'item_memory_bytes': item_memory,
            'total_memory_bytes': total_memory,
            'gpu_memory_bytes': mem_info.total,
            'gpu_free_bytes': mem_info.free,
            'optimal_batch_size': optimal_batch_size,
            'requires_streaming': total_memory > mem_info.free,
            'estimated_batches': (dataset_size + optimal_batch_size - 1) // optimal_batch_size
        }