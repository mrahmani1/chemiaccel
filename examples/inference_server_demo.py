"""
Comprehensive Demo for ChemiAccel High-Performance Inference Server - Task 20.2

This demo showcases the complete inference server functionality including:
- FastAPI REST API with automatic documentation
- Dynamic batching for optimal GPU utilization
- Model preloading with warm-up for zero cold-start latency
- Async and streaming endpoints
- Performance monitoring and metrics
- Load testing and optimization examples

The demo includes client examples for all API endpoints and demonstrates
real-world usage patterns for production deployment.
"""

import asyncio
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import random

try:
    import aiohttp
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceServerDemo:
    """Comprehensive demo of the inference server capabilities."""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        """Initialize demo with server URL.
        
        Args:
            server_url: URL of the inference server
        """
        self.server_url = server_url
        self.session = None
        
        # Sample molecules for testing
        self.sample_molecules = [
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
            # More complex molecules
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # Ibuprofen
        ]
        
        logger.info(f"Initialized demo for server: {server_url}")
    
    def create_sample_dataset(self, size: int, include_errors: bool = False) -> List[str]:
        """Create a sample dataset for testing.
        
        Args:
            size: Number of molecules to generate
            include_errors: Include some invalid SMILES for error testing
            
        Returns:
            List of SMILES strings
        """
        dataset = []
        
        # Fill with random samples from our molecule pool
        for i in range(size):
            mol = random.choice(self.sample_molecules)
            
            # Add some variations
            if i % 5 == 0 and "c1ccccc1" in mol:
                # Add substituents to benzene rings
                substituents = ["C", "O", "N", "Cl", "F"]
                sub = random.choice(substituents)
                mol = mol.replace("c1ccccc1", f"c1ccc(cc1){sub}")
            
            dataset.append(mol)
        
        # Add some errors if requested
        if include_errors:
            error_count = int(size * 0.05)  # 5% errors
            error_smiles = ["invalid_smiles", "", "C" * 100, "[Pt][Au]"]
            for _ in range(error_count):
                dataset.append(random.choice(error_smiles))
        
        random.shuffle(dataset)
        return dataset
    
    def check_server_health(self) -> Dict[str, Any]:
        """Check server health and readiness."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"Server health: {health_data['status']}")
                logger.info(f"GPU available: {health_data['gpu_available']}")
                logger.info(f"Uptime: {health_data['uptime']:.1f} seconds")
                return health_data
            else:
                logger.error(f"Health check failed: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            return {}
    
    def demo_single_prediction(self):
        """Demonstrate single molecule prediction."""
        logger.info("=== Single Molecule Prediction Demo ===")
        
        # Test a simple molecule
        test_molecule = "CCO"  # Ethanol
        
        payload = {
            "smiles": test_molecule,
            "descriptors": ["molecular_weight", "logp", "tpsa"],
            "include_properties": True
        }
        
        logger.info(f"Predicting descriptors for: {test_molecule}")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.server_url}/predict/single",
                json=payload,
                timeout=30
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Prediction successful in {processing_time:.3f}s")
                logger.info(f"Request ID: {result['request_id']}")
                logger.info(f"Server processing time: {result['processing_time']:.3f}s")
                logger.info(f"Results: {json.dumps(result['results'], indent=2)}")
                return result
            else:
                logger.error(f"Prediction failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
    
    def demo_batch_prediction(self, batch_size: int = 10):
        """Demonstrate batch prediction.
        
        Args:
            batch_size: Number of molecules in the batch
        """
        logger.info(f"=== Batch Prediction Demo (size={batch_size}) ===")
        
        # Create test batch
        test_molecules = self.create_sample_dataset(batch_size)
        
        payload = {
            "smiles": test_molecules,
            "descriptors": ["molecular_weight", "logp", "tpsa"],
            "include_properties": True,
            "batch_size": batch_size
        }
        
        logger.info(f"Predicting descriptors for {len(test_molecules)} molecules")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.server_url}/predict/batch",
                json=payload,
                timeout=60
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Batch prediction successful in {processing_time:.3f}s")
                logger.info(f"Request ID: {result['request_id']}")
                logger.info(f"Server processing time: {result['processing_time']:.3f}s")
                logger.info(f"Molecules processed: {result['metadata']['molecule_count']}")
                
                # Calculate throughput
                throughput = batch_size / processing_time
                logger.info(f"Throughput: {throughput:.1f} molecules/second")
                
                # Show first few results
                results = result['results']
                if 'descriptors' in results:
                    logger.info(f"Sample results (first 3):")
                    for i, desc in enumerate(results['descriptors'][:3]):
                        logger.info(f"  {test_molecules[i]} -> {desc}")
                
                return result
            else:
                logger.error(f"Batch prediction failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Batch request failed: {e}")
            return None
    
    async def demo_streaming_prediction(self, batch_size: int = 100):
        """Demonstrate streaming prediction for large batches.
        
        Args:
            batch_size: Number of molecules to process
        """
        logger.info(f"=== Streaming Prediction Demo (size={batch_size}) ===")
        
        if not REQUESTS_AVAILABLE:
            logger.error("aiohttp is required for streaming demo")
            return None
        
        # Create large test batch
        test_molecules = self.create_sample_dataset(batch_size)
        
        payload = {
            "smiles": test_molecules,
            "descriptors": ["molecular_weight", "logp", "tpsa"],
            "include_properties": True
        }
        
        logger.info(f"Starting streaming prediction for {len(test_molecules)} molecules")
        start_time = time.time()
        
        try:
            timeout = aiohttp.ClientTimeout(total=300)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.server_url}/predict/stream",
                    json=payload
                ) as response:
                    
                    if response.status != 200:
                        logger.error(f"Streaming failed: {response.status}")
                        return None
                    
                    total_processed = 0
                    chunk_count = 0
                    
                    async for line in response.content:
                        line_text = line.decode('utf-8').strip()
                        
                        if line_text.startswith('data: '):
                            data_json = line_text[6:]  # Remove 'data: ' prefix
                            try:
                                data = json.loads(data_json)
                                
                                if data['type'] == 'start':
                                    logger.info(f"Stream started for {data['total']} molecules")
                                
                                elif data['type'] == 'chunk':
                                    chunk_count += 1
                                    processed = data['processed']
                                    total = data['total']
                                    progress = (processed / total) * 100
                                    
                                    logger.info(f"Chunk {chunk_count}: {processed}/{total} molecules ({progress:.1f}%)")
                                    total_processed = processed
                                
                                elif data['type'] == 'complete':
                                    processing_time = time.time() - start_time
                                    logger.info(f"Streaming complete!")
                                    logger.info(f"Total processed: {data['total_processed']}")
                                    logger.info(f"Total time: {processing_time:.3f}s")
                                    logger.info(f"Throughput: {data['total_processed'] / processing_time:.1f} mol/s")
                                    return data
                                
                                elif data['type'] == 'error':
                                    logger.error(f"Stream error: {data['error']}")
                                    return None
                                    
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse stream data: {data_json}")
            
        except Exception as e:
            logger.error(f"Streaming request failed: {e}")
            return None
    
    def demo_concurrent_requests(self, num_requests: int = 5, batch_size: int = 10):
        """Demonstrate concurrent request handling.
        
        Args:
            num_requests: Number of concurrent requests
            batch_size: Size of each batch
        """
        logger.info(f"=== Concurrent Requests Demo ({num_requests} requests) ===")
        
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def make_request(request_id: int) -> Dict[str, Any]:
            """Make a single batch request."""
            test_molecules = self.create_sample_dataset(batch_size)
            
            payload = {
                "smiles": test_molecules,
                "descriptors": ["molecular_weight", "logp"],
                "include_properties": True
            }
            
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{self.server_url}/predict/batch",
                    json=payload,
                    timeout=60
                )
                
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        'request_id': request_id,
                        'success': True,
                        'processing_time': processing_time,
                        'server_time': result['processing_time'],
                        'molecule_count': batch_size
                    }
                else:
                    return {
                        'request_id': request_id,
                        'success': False,
                        'error': f"HTTP {response.status_code}",
                        'processing_time': processing_time
                    }
                    
            except Exception as e:
                processing_time = time.time() - start_time
                return {
                    'request_id': request_id,
                    'success': False,
                    'error': str(e),
                    'processing_time': processing_time
                }
        
        # Execute concurrent requests
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                
                if result['success']:
                    logger.info(f"Request {result['request_id']}: {result['processing_time']:.3f}s")
                else:
                    logger.error(f"Request {result['request_id']} failed: {result['error']}")
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful = [r for r in results if r['success']]
        success_rate = len(successful) / len(results) * 100
        
        if successful:
            avg_latency = sum(r['processing_time'] for r in successful) / len(successful)
            total_molecules = sum(r.get('molecule_count', 0) for r in successful)
            throughput = total_molecules / total_time
            
            logger.info(f"Concurrent test results:")
            logger.info(f"  Success rate: {success_rate:.1f}%")
            logger.info(f"  Average latency: {avg_latency:.3f}s")
            logger.info(f"  Total throughput: {throughput:.1f} molecules/second")
            logger.info(f"  Total time: {total_time:.3f}s")
        
        return results
    
    def demo_error_handling(self):
        """Demonstrate error handling capabilities."""
        logger.info("=== Error Handling Demo ===")
        
        # Test invalid SMILES
        invalid_molecules = [
            "invalid_smiles",
            "",
            "C" * 200,  # Very long
            "[Pt][Au][Ag]",  # Unusual
            None  # This will cause a validation error
        ]
        
        for i, invalid_smiles in enumerate(invalid_molecules):
            logger.info(f"Testing invalid molecule {i+1}: {repr(invalid_smiles)}")
            
            try:
                if invalid_smiles is None:
                    # This should cause a validation error
                    payload = {"invalid_field": "test"}
                else:
                    payload = {
                        "smiles": invalid_smiles,
                        "include_properties": True
                    }
                
                response = requests.post(
                    f"{self.server_url}/predict/single",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.warning(f"  Unexpected success: {result}")
                else:
                    logger.info(f"  Expected error: {response.status_code} - {response.text[:100]}")
                    
            except Exception as e:
                logger.info(f"  Expected exception: {e}")
    
    def demo_performance_monitoring(self):
        """Demonstrate performance monitoring endpoints."""
        logger.info("=== Performance Monitoring Demo ===")
        
        # Get current metrics
        try:
            response = requests.get(f"{self.server_url}/metrics", timeout=10)
            if response.status_code == 200:
                metrics = response.json()
                logger.info("Current server metrics:")
                logger.info(f"  Total requests: {metrics['total_requests']}")
                logger.info(f"  Active requests: {metrics['active_requests']}")
                logger.info(f"  Average latency: {metrics['average_latency']:.3f}s")
                logger.info(f"  GPU utilization: {metrics['gpu_utilization']:.1f}%")
                
                memory_usage = metrics['memory_usage']
                logger.info(f"  Memory usage:")
                for key, value in memory_usage.items():
                    if 'mb' in key.lower():
                        logger.info(f"    {key}: {value:.1f} MB")
                    else:
                        logger.info(f"    {key}: {value:.1f}")
                
                return metrics
            else:
                logger.error(f"Failed to get metrics: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Metrics request failed: {e}")
            return None
    
    def demo_descriptors_list(self):
        """Demonstrate available descriptors endpoint."""
        logger.info("=== Available Descriptors Demo ===")
        
        try:
            response = requests.get(f"{self.server_url}/descriptors", timeout=10)
            if response.status_code == 200:
                descriptors = response.json()
                logger.info("Available descriptor sets:")
                for set_name, descriptor_list in descriptors.items():
                    logger.info(f"  {set_name}: {len(descriptor_list)} descriptors")
                    logger.info(f"    {', '.join(descriptor_list)}")
                return descriptors
            else:
                logger.error(f"Failed to get descriptors: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Descriptors request failed: {e}")
            return None
    
    def demo_performance_comparison(self):
        """Demonstrate performance comparison between request types."""
        logger.info("=== Performance Comparison Demo ===")
        
        # Test different batch sizes
        batch_sizes = [1, 10, 50, 100]
        results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            # Create test data
            test_molecules = self.create_sample_dataset(batch_size)
            
            payload = {
                "smiles": test_molecules,
                "descriptors": ["molecular_weight", "logp", "tpsa"],
                "include_properties": True
            }
            
            # Time the request
            start_time = time.time()
            
            try:
                if batch_size == 1:
                    # Use single endpoint for size 1
                    payload_single = {
                        "smiles": test_molecules[0],
                        "descriptors": ["molecular_weight", "logp", "tpsa"],
                        "include_properties": True
                    }
                    response = requests.post(
                        f"{self.server_url}/predict/single",
                        json=payload_single,
                        timeout=30
                    )
                else:
                    # Use batch endpoint
                    response = requests.post(
                        f"{self.server_url}/predict/batch",
                        json=payload,
                        timeout=60
                    )
                
                total_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    server_time = result['processing_time']
                    
                    throughput = batch_size / total_time
                    server_throughput = batch_size / server_time
                    
                    results[batch_size] = {
                        'total_time': total_time,
                        'server_time': server_time,
                        'throughput': throughput,
                        'server_throughput': server_throughput,
                        'overhead': total_time - server_time
                    }
                    
                    logger.info(f"  Total time: {total_time:.3f}s")
                    logger.info(f"  Server time: {server_time:.3f}s")
                    logger.info(f"  Throughput: {throughput:.1f} mol/s")
                    logger.info(f"  Server throughput: {server_throughput:.1f} mol/s")
                    
                else:
                    logger.error(f"  Failed: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"  Error: {e}")
        
        # Summary
        if results:
            logger.info("\nPerformance Summary:")
            logger.info("Batch Size | Total Time | Server Time | Throughput | Server Throughput")
            logger.info("-" * 70)
            for batch_size, data in results.items():
                logger.info(f"{batch_size:9d} | {data['total_time']:9.3f}s | {data['server_time']:10.3f}s | {data['throughput']:9.1f} | {data['server_throughput']:14.1f}")
        
        return results
    
    def run_comprehensive_demo(self):
        """Run all demo scenarios."""
        logger.info("Starting ChemiAccel Inference Server Comprehensive Demo")
        logger.info("=" * 70)
        
        # Check server health
        health = self.check_server_health()
        if not health or health.get('status') != 'healthy':
            logger.error("Server is not healthy. Aborting demo.")
            return
        
        try:
            # Basic functionality demos
            self.demo_single_prediction()
            print()
            
            self.demo_batch_prediction(batch_size=10)
            print()
            
            # Advanced features
            asyncio.run(self.demo_streaming_prediction(batch_size=50))
            print()
            
            self.demo_concurrent_requests(num_requests=3, batch_size=5)
            print()
            
            # Error handling
            self.demo_error_handling()
            print()
            
            # Monitoring and introspection
            self.demo_descriptors_list()
            print()
            
            self.demo_performance_monitoring()
            print()
            
            # Performance comparison
            self.demo_performance_comparison()
            print()
            
            logger.info("=" * 70)
            logger.info("Demo completed successfully!")
            logger.info("All inference server features demonstrated:")
            logger.info("  ✓ Single molecule prediction")
            logger.info("  ✓ Batch prediction with dynamic batching")
            logger.info("  ✓ Streaming prediction for large datasets")
            logger.info("  ✓ Concurrent request handling")
            logger.info("  ✓ Error handling and validation")
            logger.info("  ✓ Performance monitoring and metrics")
            logger.info("  ✓ Descriptor introspection")
            logger.info("  ✓ Performance optimization")
            
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        except Exception as e:
            logger.error(f"Demo failed with error: {e}")
            raise


def main():
    """Main function to run the demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ChemiAccel Inference Server Demo")
    parser.add_argument("--server", default="http://localhost:8000", 
                       help="Server URL (default: http://localhost:8000)")
    parser.add_argument("--single", action="store_true", 
                       help="Run only single prediction demo")
    parser.add_argument("--batch", type=int, metavar="SIZE",
                       help="Run only batch prediction demo with specified size")
    parser.add_argument("--stream", type=int, metavar="SIZE",
                       help="Run only streaming prediction demo with specified size")
    parser.add_argument("--concurrent", type=int, metavar="NUM",
                       help="Run only concurrent requests demo with specified number")
    parser.add_argument("--health", action="store_true",
                       help="Check server health only")
    
    args = parser.parse_args()
    
    demo = InferenceServerDemo(args.server)
    
    if args.health:
        demo.check_server_health()
    elif args.single:
        demo.demo_single_prediction()
    elif args.batch:
        demo.demo_batch_prediction(args.batch)
    elif args.stream:
        asyncio.run(demo.demo_streaming_prediction(args.stream))
    elif args.concurrent:
        demo.demo_concurrent_requests(args.concurrent)
    else:
        # Run comprehensive demo
        demo.run_comprehensive_demo()


if __name__ == "__main__":
    main()