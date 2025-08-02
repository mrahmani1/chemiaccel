# Distributed Computing

ChemiAccel provides comprehensive distributed computing capabilities for scaling molecular descriptor calculations across multiple nodes, regions, and cloud providers. This guide covers multi-node setups, load balancing, and fault-tolerant distributed architectures.

## Overview

Distributed computing features include:
- **Multi-Node Processing**: Scale across hundreds of compute nodes
- **Load Balancing**: Intelligent work distribution and resource management
- **Fault Tolerance**: Automatic failure detection and recovery
- **Cross-Platform Support**: Seamless integration across different environments
- **Dynamic Scaling**: Elastic resource allocation based on workload
- **Data Partitioning**: Efficient data distribution strategies

## Quick Start

### Basic Distributed Setup

```python
from chemi_accel.distributed import DistributedManager, ClusterConfig
from chemi_accel import Calculator

# Configure distributed cluster
cluster_config = ClusterConfig(
    master_node="192.168.1.10:8000",
    worker_nodes=["192.168.1.11:8000", "192.168.1.12:8000", "192.168.1.13:8000"],
    backend="cuda",
    communication_backend="nccl"
)

# Initialize distributed manager
distributed_manager = DistributedManager(cluster_config)

# Start distributed calculation
molecules = load_large_molecule_dataset()  # 5M+ molecules
results = distributed_manager.calculate_distributed(
    molecules=molecules,
    descriptors="comprehensive",
    batch_size=10000,
    redundancy_factor=1.2
)

print(f"Processed {len(results)} molecules across {len(cluster_config.worker_nodes)} nodes")
```

## Multi-Node Architecture

### Distributed Cluster Manager

```python
from chemi_accel.distributed import ClusterManager, NodeConfig
import threading
import queue
import time

class ChemiAccelDistributedCluster:
    """Distributed cluster management for ChemiAccel."""
    
    def __init__(self, cluster_config):
        self.cluster_config = cluster_config
        self.cluster_manager = ClusterManager(cluster_config)
        self.task_queue = queue.PriorityQueue()
        self.result_queue = queue.Queue()
        self.node_status = {}
        self.active_tasks = {}
        
    def initialize_cluster(self):
        """Initialize distributed cluster with health checks."""
        
        print("Initializing distributed cluster...")
        
        # Initialize master node
        master_node = self.cluster_manager.initialize_master(
            bind_address=self.cluster_config.master_node,
            port=self.cluster_config.master_port
        )
        
        # Initialize worker nodes
        worker_nodes = []
        for worker_address in self.cluster_config.worker_nodes:
            try:
                worker = self.cluster_manager.initialize_worker(
                    worker_address=worker_address,
                    master_address=self.cluster_config.master_node,
                    capabilities=self.get_node_capabilities(worker_address)
                )
                worker_nodes.append(worker)
                self.node_status[worker_address] = "healthy"
                print(f"Worker {worker_address} initialized successfully")
                
            except Exception as e:
                print(f"Failed to initialize worker {worker_address}: {e}")
                self.node_status[worker_address] = "failed"
        
        # Start health monitoring
        self.start_health_monitoring()
        
        return master_node, worker_nodes
    
    def get_node_capabilities(self, node_address):
        """Query node capabilities and resources."""
        
        try:
            node_info = self.cluster_manager.query_node_info(node_address)
            
            capabilities = {
                'cpu_cores': node_info.get('cpu_cores', 8),
                'memory_gb': node_info.get('memory_gb', 32),
                'gpu_count': node_info.get('gpu_count', 0),
                'gpu_memory_gb': node_info.get('gpu_memory_gb', 0),
                'network_bandwidth': node_info.get('network_bandwidth', '1Gb/s'),
                'storage_type': node_info.get('storage_type', 'HDD'),
                'cuda_compute_capability': node_info.get('cuda_cc', None)
            }
            
            return capabilities
            
        except Exception as e:
            print(f"Failed to query capabilities for {node_address}: {e}")
            return self.get_default_capabilities()
    
    def get_default_capabilities(self):
        """Default node capabilities as fallback."""
        return {
            'cpu_cores': 8,
            'memory_gb': 32,
            'gpu_count': 1,
            'gpu_memory_gb': 16,
            'network_bandwidth': '1Gb/s',
            'storage_type': 'SSD',
            'cuda_compute_capability': '7.0'
        }
    
    def partition_workload(self, molecules, descriptors, partition_strategy="balanced"):
        """Partition workload across available nodes."""
        
        healthy_nodes = [node for node, status in self.node_status.items() 
                        if status == "healthy"]
        
        if not healthy_nodes:
            raise RuntimeError("No healthy nodes available")
        
        partitions = []
        
        if partition_strategy == "balanced":
            # Equal distribution
            chunk_size = len(molecules) // len(healthy_nodes)
            
            for i, node in enumerate(healthy_nodes):
                start_idx = i * chunk_size
                if i == len(healthy_nodes) - 1:  # Last node gets remainder
                    end_idx = len(molecules)
                else:
                    end_idx = (i + 1) * chunk_size
                
                partition = {
                    'node_address': node,
                    'molecules': molecules[start_idx:end_idx],
                    'descriptors': descriptors,
                    'partition_id': i,
                    'priority': 1
                }
                partitions.append(partition)
        
        elif partition_strategy == "capability_weighted":
            # Weight by node capabilities
            node_weights = []
            for node in healthy_nodes:
                capabilities = self.get_node_capabilities(node)
                weight = (capabilities['cpu_cores'] + 
                         capabilities['gpu_count'] * 4 +  # GPUs worth 4x CPU cores
                         capabilities['memory_gb'] / 8)   # Memory factor
                node_weights.append(weight)
            
            total_weight = sum(node_weights)
            
            start_idx = 0
            for i, (node, weight) in enumerate(zip(healthy_nodes, node_weights)):
                chunk_size = int(len(molecules) * (weight / total_weight))
                if i == len(healthy_nodes) - 1:  # Last node gets remainder
                    end_idx = len(molecules)
                else:
                    end_idx = start_idx + chunk_size
                
                partition = {
                    'node_address': node,
                    'molecules': molecules[start_idx:end_idx],
                    'descriptors': descriptors,
                    'partition_id': i,
                    'priority': 1
                }
                partitions.append(partition)
                start_idx = end_idx
        
        return partitions
    
    def execute_distributed_calculation(self, molecules, descriptors, 
                                      partition_strategy="capability_weighted"):
        """Execute distributed calculation with fault tolerance."""
        
        # Partition workload
        partitions = self.partition_workload(molecules, descriptors, partition_strategy)
        
        # Submit tasks to queue
        for partition in partitions:
            task = {
                'task_id': f"task_{partition['partition_id']}",
                'partition': partition,
                'retry_count': 0,
                'max_retries': 3
            }
            self.task_queue.put((partition['priority'], task))
        
        # Start task execution threads
        execution_threads = []
        for i in range(min(len(partitions), 10)):  # Max 10 concurrent tasks
            thread = threading.Thread(target=self.task_execution_worker)
            thread.daemon = True
            thread.start()
            execution_threads.append(thread)
        
        # Monitor progress and collect results
        results = self.monitor_and_collect_results(len(partitions))
        
        return results
    
    def task_execution_worker(self):
        """Worker thread for executing distributed tasks."""
        
        while True:
            try:
                # Get task from queue
                priority, task = self.task_queue.get(timeout=60)
                
                task_id = task['task_id']
                partition = task['partition']
                
                print(f"Executing {task_id} on {partition['node_address']}")
                
                # Mark task as active
                self.active_tasks[task_id] = {
                    'start_time': time.time(),
                    'node_address': partition['node_address'],
                    'molecules_count': len(partition['molecules'])
                }
                
                # Execute task on worker node
                try:
                    result = self.cluster_manager.execute_remote_calculation(
                        node_address=partition['node_address'],
                        molecules=partition['molecules'],
                        descriptors=partition['descriptors'],
                        timeout=3600  # 1 hour timeout
                    )
                    
                    # Task completed successfully
                    execution_time = time.time() - self.active_tasks[task_id]['start_time']
                    
                    result_data = {
                        'task_id': task_id,
                        'partition_id': partition['partition_id'],
                        'node_address': partition['node_address'],
                        'results': result,
                        'execution_time': execution_time,
                        'status': 'completed'
                    }
                    
                    self.result_queue.put(result_data)
                    del self.active_tasks[task_id]
                    
                    print(f"Task {task_id} completed in {execution_time:.2f}s")
                
                except Exception as e:
                    # Task failed - handle retry
                    print(f"Task {task_id} failed: {e}")
                    
                    task['retry_count'] += 1
                    if task['retry_count'] <= task['max_retries']:
                        # Retry on different node if possible
                        healthy_nodes = [node for node, status in self.node_status.items() 
                                       if status == "healthy" and node != partition['node_address']]
                        
                        if healthy_nodes:
                            # Reassign to different node
                            partition['node_address'] = healthy_nodes[0]
                            print(f"Retrying {task_id} on {partition['node_address']} (attempt {task['retry_count']})")
                            
                            # Re-queue task with higher priority
                            self.task_queue.put((0, task))  # Priority 0 for retries
                        else:
                            # No healthy nodes available
                            result_data = {
                                'task_id': task_id,
                                'partition_id': partition['partition_id'],
                                'node_address': partition['node_address'],
                                'error': str(e),
                                'status': 'failed'
                            }
                            self.result_queue.put(result_data)
                    else:
                        # Max retries reached
                        result_data = {
                            'task_id': task_id,
                            'partition_id': partition['partition_id'],
                            'node_address': partition['node_address'],
                            'error': f"Max retries ({task['max_retries']}) reached: {e}",
                            'status': 'failed'
                        }
                        self.result_queue.put(result_data)
                    
                    if task_id in self.active_tasks:
                        del self.active_tasks[task_id]
                
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Task execution worker error: {e}")
    
    def monitor_and_collect_results(self, expected_results):
        """Monitor task execution and collect results."""
        
        collected_results = []
        failed_tasks = []
        
        while len(collected_results) + len(failed_tasks) < expected_results:
            try:
                result = self.result_queue.get(timeout=300)  # 5 minute timeout
                
                if result['status'] == 'completed':
                    collected_results.append(result)
                    print(f"Collected result from {result['node_address']} "
                          f"({len(collected_results)}/{expected_results})")
                else:
                    failed_tasks.append(result)
                    print(f"Task failed: {result['task_id']} - {result.get('error', 'Unknown error')}")
                
            except queue.Empty:
                print("Timeout waiting for results. Checking active tasks...")
                self.check_and_handle_stalled_tasks()
        
        # Combine results from all partitions
        final_results = self.combine_partition_results(collected_results)
        
        if failed_tasks:
            print(f"Warning: {len(failed_tasks)} tasks failed")
            for failed_task in failed_tasks:
                print(f"  Failed: {failed_task['task_id']} - {failed_task.get('error')}")
        
        return final_results
    
    def combine_partition_results(self, partition_results):
        """Combine results from all partitions into final dataset."""
        
        # Sort results by partition_id to maintain order
        partition_results.sort(key=lambda x: x['partition_id'])
        
        combined_results = []
        total_execution_time = 0
        
        for partition_result in partition_results:
            if 'results' in partition_result:
                combined_results.extend(partition_result['results'])
                total_execution_time += partition_result['execution_time']
        
        print(f"Combined {len(combined_results)} results from {len(partition_results)} partitions")
        print(f"Total execution time: {total_execution_time:.2f}s")
        print(f"Average partition time: {total_execution_time / len(partition_results):.2f}s")
        
        return combined_results
    
    def start_health_monitoring(self):
        """Start health monitoring for cluster nodes."""
        
        def health_monitor():
            while True:
                for node_address in self.cluster_config.worker_nodes:
                    try:
                        # Ping node for health check
                        health_status = self.cluster_manager.ping_node(node_address, timeout=10)
                        
                        if health_status['status'] == 'healthy':
                            self.node_status[node_address] = "healthy"
                        else:
                            self.node_status[node_address] = "unhealthy"
                            print(f"Node {node_address} is unhealthy: {health_status.get('error')}")
                    
                    except Exception as e:
                        self.node_status[node_address] = "failed"
                        print(f"Health check failed for {node_address}: {e}")
                
                time.sleep(30)  # Check every 30 seconds
        
        health_thread = threading.Thread(target=health_monitor)
        health_thread.daemon = True
        health_thread.start()
    
    def check_and_handle_stalled_tasks(self):
        """Check for and handle stalled tasks."""
        
        current_time = time.time()
        stalled_tasks = []
        
        for task_id, task_info in self.active_tasks.items():
            if current_time - task_info['start_time'] > 3600:  # 1 hour timeout
                stalled_tasks.append(task_id)
        
        for task_id in stalled_tasks:
            task_info = self.active_tasks[task_id]
            print(f"Handling stalled task: {task_id} on {task_info['node_address']}")
            
            # Mark node as failed
            self.node_status[task_info['node_address']] = "failed"
            
            # Remove from active tasks
            del self.active_tasks[task_id]
            
            # Could re-queue task here if needed

# Example: Large-scale distributed calculation
cluster_config = ClusterConfig(
    master_node="192.168.1.10:8000",
    master_port=8000,
    worker_nodes=[
        "192.168.1.11:8000",
        "192.168.1.12:8000", 
        "192.168.1.13:8000",
        "192.168.1.14:8000",
        "192.168.1.15:8000"
    ],
    communication_backend="nccl",
    fault_tolerance=True
)

# Initialize distributed cluster
distributed_cluster = ChemiAccelDistributedCluster(cluster_config)
master_node, worker_nodes = distributed_cluster.initialize_cluster()

print(f"Cluster initialized with {len(worker_nodes)} healthy workers")

# Load large molecule dataset
molecules = load_massive_dataset()  # 10M+ molecules
print(f"Loaded {len(molecules)} molecules for processing")

# Execute distributed calculation
results = distributed_cluster.execute_distributed_calculation(
    molecules=molecules,
    descriptors="comprehensive",
    partition_strategy="capability_weighted"
)

print(f"Distributed calculation completed: {len(results)} results")
```

## Dynamic Load Balancing

### Adaptive Work Distribution

```python
from chemi_accel.distributed import LoadBalancer, WorkloadPredictor

class DynamicLoadBalancer:
    """Dynamic load balancing for distributed ChemiAccel calculations."""
    
    def __init__(self, cluster_nodes):
        self.cluster_nodes = cluster_nodes
        self.load_balancer = LoadBalancer()
        self.workload_predictor = WorkloadPredictor()
        self.node_performance_history = {}
        
    def initialize_performance_tracking(self):
        """Initialize performance tracking for all nodes."""
        
        for node in self.cluster_nodes:
            self.node_performance_history[node] = {
                'throughput_history': [],
                'latency_history': [],
                'error_rate_history': [],
                'resource_utilization': [],
                'last_update': time.time()
            }
    
    def measure_node_performance(self, node_address, sample_molecules=None):
        """Measure current performance metrics for a node."""
        
        if sample_molecules is None:
            # Use small benchmark dataset
            sample_molecules = generate_benchmark_molecules(100)
        
        try:
            start_time = time.time()
            
            # Execute benchmark calculation
            calc = Calculator(backend="cuda")  # Assuming GPU nodes
            results = calc.pandas(
                sample_molecules, 
                descriptors="basic",
                batch_size=50
            )
            
            end_time = time.time()
            
            # Calculate performance metrics
            execution_time = end_time - start_time
            throughput = len(sample_molecules) / execution_time  # molecules/second
            latency = execution_time / len(sample_molecules)     # seconds/molecule
            
            # Get resource utilization
            resource_usage = self.get_node_resource_usage(node_address)
            
            performance_metrics = {
                'throughput': throughput,
                'latency': latency,
                'cpu_usage': resource_usage.get('cpu_percent', 0),
                'memory_usage': resource_usage.get('memory_percent', 0),
                'gpu_usage': resource_usage.get('gpu_percent', 0),
                'error_rate': 0.0,  # No errors in successful benchmark
                'timestamp': time.time()
            }
            
            return performance_metrics
            
        except Exception as e:
            print(f"Performance measurement failed for {node_address}: {e}")
            return {
                'throughput': 0,
                'latency': float('inf'),
                'error_rate': 1.0,
                'timestamp': time.time()
            }
    
    def update_performance_history(self, node_address, metrics):
        """Update performance history for a node."""
        
        if node_address not in self.node_performance_history:
            self.initialize_performance_tracking()
        
        history = self.node_performance_history[node_address]
        
        # Add new metrics
        history['throughput_history'].append(metrics['throughput'])
        history['latency_history'].append(metrics['latency'])
        history['error_rate_history'].append(metrics['error_rate'])
        history['resource_utilization'].append({
            'cpu': metrics.get('cpu_usage', 0),
            'memory': metrics.get('memory_usage', 0),
            'gpu': metrics.get('gpu_usage', 0)
        })
        history['last_update'] = metrics['timestamp']
        
        # Keep only recent history (last 100 measurements)
        max_history = 100
        for key in ['throughput_history', 'latency_history', 'error_rate_history', 'resource_utilization']:
            if len(history[key]) > max_history:
                history[key] = history[key][-max_history:]
    
    def calculate_node_weights(self):
        """Calculate current weights for load balancing."""
        
        node_weights = {}
        
        for node_address in self.cluster_nodes:
            # Get recent performance metrics
            current_metrics = self.measure_node_performance(node_address)
            self.update_performance_history(node_address, current_metrics)
            
            history = self.node_performance_history[node_address]
            
            if not history['throughput_history']:
                # No history available, use default weight
                node_weights[node_address] = 1.0
                continue
            
            # Calculate weighted performance score
            recent_throughput = np.mean(history['throughput_history'][-10:])  # Last 10 measurements
            recent_error_rate = np.mean(history['error_rate_history'][-10:])
            recent_cpu_usage = np.mean([r['cpu'] for r in history['resource_utilization'][-10:]])
            recent_memory_usage = np.mean([r['memory'] for r in history['resource_utilization'][-10:]])
            
            # Calculate composite score
            # Higher throughput is better, lower error rate is better, moderate resource usage is better
            throughput_score = recent_throughput / 100  # Normalize (assuming 100 mol/s is reference)
            error_penalty = 1 - recent_error_rate
            resource_efficiency = 1 - max(0, (recent_cpu_usage + recent_memory_usage) / 200 - 0.8)
            
            composite_score = throughput_score * error_penalty * resource_efficiency
            node_weights[node_address] = max(0.1, composite_score)  # Minimum weight of 0.1
        
        # Normalize weights to sum to 1
        total_weight = sum(node_weights.values())
        if total_weight > 0:
            node_weights = {node: weight / total_weight for node, weight in node_weights.items()}
        
        return node_weights
    
    def distribute_workload_dynamically(self, molecules, descriptors):
        """Distribute workload based on current node performance."""
        
        # Calculate current node weights
        node_weights = self.calculate_node_weights()
        
        print("Current node weights:")
        for node, weight in node_weights.items():
            print(f"  {node}: {weight:.3f}")
        
        # Estimate workload for different descriptor types
        workload_complexity = self.workload_predictor.estimate_complexity(
            molecules[:100],  # Sample for estimation
            descriptors
        )
        
        # Distribute molecules based on weights and complexity
        distributions = []
        start_idx = 0
        
        for node_address, weight in node_weights.items():
            # Calculate chunk size based on weight
            chunk_size = int(len(molecules) * weight)
            
            # Adjust for workload complexity
            if workload_complexity > 1.5:  # High complexity
                chunk_size = int(chunk_size * 0.8)  # Reduce load
            elif workload_complexity < 0.5:  # Low complexity
                chunk_size = int(chunk_size * 1.2)  # Increase load
            
            # Ensure we don't exceed available molecules
            end_idx = min(start_idx + chunk_size, len(molecules))
            
            if start_idx < len(molecules):
                distribution = {
                    'node_address': node_address,
                    'molecules': molecules[start_idx:end_idx],
                    'descriptors': descriptors,
                    'expected_performance': node_weights[node_address],
                    'estimated_time': self.estimate_execution_time(
                        len(molecules[start_idx:end_idx]), 
                        workload_complexity, 
                        node_weights[node_address]
                    )
                }
                distributions.append(distribution)
                start_idx = end_idx
        
        # Handle remaining molecules (assign to best performing node)
        if start_idx < len(molecules):
            best_node = max(node_weights.items(), key=lambda x: x[1])[0]
            
            # Find the distribution for the best node and add remaining molecules
            for dist in distributions:
                if dist['node_address'] == best_node:
                    dist['molecules'].extend(molecules[start_idx:])
                    break
        
        return distributions
    
    def adaptive_rebalancing(self, active_distributions):
        """Perform adaptive rebalancing during execution."""
        
        # Monitor progress of active distributions
        rebalancing_needed = False
        slow_nodes = []
        fast_nodes = []
        
        for distribution in active_distributions:
            node_address = distribution['node_address']
            expected_time = distribution['estimated_time']
            
            # Check actual progress
            actual_progress = self.get_task_progress(distribution['task_id'])
            elapsed_time = time.time() - distribution['start_time']
            
            if actual_progress > 0:
                estimated_completion = elapsed_time / actual_progress
                
                if estimated_completion > expected_time * 1.5:  # 50% slower than expected
                    slow_nodes.append({
                        'node_address': node_address,
                        'distribution': distribution,
                        'slowdown_factor': estimated_completion / expected_time
                    })
                    rebalancing_needed = True
                
                elif estimated_completion < expected_time * 0.7:  # 30% faster than expected
                    fast_nodes.append({
                        'node_address': node_address,
                        'distribution': distribution,
                        'speedup_factor': expected_time / estimated_completion
                    })
        
        if rebalancing_needed and fast_nodes:
            # Redistribute work from slow nodes to fast nodes
            self.redistribute_work(slow_nodes, fast_nodes)
    
    def redistribute_work(self, slow_nodes, fast_nodes):
        """Redistribute work from slow to fast nodes."""
        
        print("Performing adaptive rebalancing...")
        
        for slow_node in slow_nodes:
            remaining_work = self.get_remaining_work(slow_node['distribution'])
            
            if len(remaining_work) > 100:  # Only redistribute if significant work remains
                # Calculate how much work to redistribute
                redistribute_fraction = 0.3  # Redistribute 30% of remaining work
                work_to_redistribute = remaining_work[:int(len(remaining_work) * redistribute_fraction)]
                
                # Find best fast node for redistribution
                best_fast_node = max(fast_nodes, key=lambda x: x['speedup_factor'])
                
                # Create new task for fast node
                new_task = {
                    'node_address': best_fast_node['node_address'],
                    'molecules': work_to_redistribute,
                    'descriptors': slow_node['distribution']['descriptors'],
                    'task_type': 'rebalancing',
                    'original_node': slow_node['node_address']
                }
                
                # Submit redistributed task
                self.submit_rebalancing_task(new_task)
                
                # Update slow node's work
                self.update_node_workload(
                    slow_node['node_address'],
                    remaining_work[int(len(remaining_work) * redistribute_fraction):]
                )
                
                print(f"Redistributed {len(work_to_redistribute)} molecules from "
                      f"{slow_node['node_address']} to {best_fast_node['node_address']}")

# Example: Dynamic load balancing
cluster_nodes = [
    "192.168.1.11:8000",
    "192.168.1.12:8000",
    "192.168.1.13:8000",
    "192.168.1.14:8000"
]

load_balancer = DynamicLoadBalancer(cluster_nodes)
load_balancer.initialize_performance_tracking()

# Large dataset for processing
molecules = load_large_dataset()  # 2M molecules

# Dynamic workload distribution
distributions = load_balancer.distribute_workload_dynamically(
    molecules=molecules,
    descriptors="comprehensive"
)

print(f"Workload distributed across {len(distributions)} nodes:")
for dist in distributions:
    print(f"  {dist['node_address']}: {len(dist['molecules'])} molecules "
          f"(estimated time: {dist['estimated_time']:.1f}s)")

# Execute with adaptive rebalancing
results = execute_with_rebalancing(distributions, load_balancer)
```

## Cross-Platform Integration

### Hybrid Cloud-HPC Setup

```python
from chemi_accel.distributed import HybridClusterManager, CloudNodeManager

class HybridDistributedSetup:
    """Hybrid setup combining HPC clusters and cloud resources."""
    
    def __init__(self):
        self.hybrid_manager = HybridClusterManager()
        self.cloud_manager = CloudNodeManager()
        self.hpc_nodes = []
        self.cloud_nodes = []
        
    def setup_hybrid_cluster(self, hpc_config, cloud_config):
        """Set up hybrid cluster with HPC and cloud nodes."""
        
        # Initialize HPC nodes
        if hpc_config:
            self.hpc_nodes = self.hybrid_manager.initialize_hpc_nodes(
                scheduler=hpc_config['scheduler'],  # SLURM, PBS, etc.
                partition=hpc_config['partition'],
                nodes=hpc_config['node_list'],
                job_config=hpc_config['job_config']
            )
            print(f"Initialized {len(self.hpc_nodes)} HPC nodes")
        
        # Initialize cloud nodes
        if cloud_config:
            self.cloud_nodes = self.cloud_manager.provision_cloud_nodes(
                provider=cloud_config['provider'],  # AWS, GCP, Azure
                instance_types=cloud_config['instance_types'],
                regions=cloud_config['regions'],
                count=cloud_config['node_count']
            )
            print(f"Provisioned {len(self.cloud_nodes)} cloud nodes")
        
        # Set up inter-cluster communication
        self.setup_inter_cluster_networking()
        
        return self.hpc_nodes + self.cloud_nodes
    
    def setup_inter_cluster_networking(self):
        """Set up networking between HPC and cloud resources."""
        
        # Configure VPN or direct connect between HPC and cloud
        network_config = {
            'hpc_network': '10.0.0.0/16',
            'cloud_network': '172.16.0.0/16',
            'gateway_node': 'gateway.hpc.university.edu',
            'encryption': True,
            'compression': True
        }
        
        self.hybrid_manager.configure_network_bridge(network_config)
        
        # Test connectivity
        connectivity_results = self.hybrid_manager.test_inter_cluster_connectivity(
            self.hpc_nodes, 
            self.cloud_nodes
        )
        
        print(f"Network connectivity test: {connectivity_results['success_rate']:.1%} success rate")
    
    def intelligent_workload_placement(self, molecules, descriptors, cost_constraints=None):
        """Intelligently place workload across HPC and cloud resources."""
        
        # Analyze workload characteristics
        workload_analysis = self.analyze_workload_requirements(molecules, descriptors)
        
        # Get current resource costs and availability
        hpc_availability = self.get_hpc_availability()
        cloud_costs = self.get_cloud_costs()
        
        # Optimization objective: minimize cost while meeting performance requirements
        placement_strategy = self.optimize_placement(
            workload_analysis=workload_analysis,
            hpc_availability=hpc_availability,
            cloud_costs=cloud_costs,
            cost_constraints=cost_constraints
        )
        
        # Distribute workload according to strategy
        hpc_workload = []
        cloud_workload = []
        
        for placement in placement_strategy:
            if placement['resource_type'] == 'hpc':
                hpc_workload.append({
                    'nodes': placement['nodes'],
                    'molecules': placement['molecules'],
                    'descriptors': descriptors,
                    'estimated_cost': 0,  # HPC typically has fixed allocation
                    'estimated_time': placement['estimated_time']
                })
            else:  # cloud
                cloud_workload.append({
                    'nodes': placement['nodes'],
                    'molecules': placement['molecules'],
                    'descriptors': descriptors,
                    'estimated_cost': placement['estimated_cost'],
                    'estimated_time': placement['estimated_time']
                })
        
        return hpc_workload, cloud_workload
    
    def execute_hybrid_calculation(self, molecules, descriptors, cost_budget=None):
        """Execute calculation across hybrid infrastructure."""
        
        # Plan workload placement
        hpc_workload, cloud_workload = self.intelligent_workload_placement(
            molecules, descriptors, cost_budget
        )
        
        # Execute on HPC nodes
        hpc_futures = []
        for workload in hpc_workload:
            future = self.hybrid_manager.submit_hpc_job(
                nodes=workload['nodes'],
                molecules=workload['molecules'],
                descriptors=workload['descriptors']
            )
            hpc_futures.append(future)
        
        # Execute on cloud nodes
        cloud_futures = []
        for workload in cloud_workload:
            future = self.cloud_manager.submit_cloud_job(
                nodes=workload['nodes'],
                molecules=workload['molecules'],
                descriptors=workload['descriptors']
            )
            cloud_futures.append(future)
        
        # Monitor execution and collect results
        all_results = []
        total_cost = 0
        
        # Collect HPC results
        for future in hpc_futures:
            result = future.get(timeout=3600)  # 1 hour timeout
            all_results.extend(result['data'])
            print(f"HPC job completed: {len(result['data'])} molecules processed")
        
        # Collect cloud results
        for future in cloud_futures:
            result = future.get(timeout=3600)
            all_results.extend(result['data'])
            total_cost += result['cost']
            print(f"Cloud job completed: {len(result['data'])} molecules processed, "
                  f"cost: ${result['cost']:.2f}")
        
        # Clean up cloud resources
        self.cloud_manager.terminate_cloud_nodes(self.cloud_nodes)
        
        execution_summary = {
            'total_molecules': len(all_results),
            'hpc_jobs': len(hpc_futures),
            'cloud_jobs': len(cloud_futures),
            'total_cost': total_cost,
            'results': all_results
        }
        
        return execution_summary

# Example: Hybrid HPC-Cloud setup
hybrid_setup = HybridDistributedSetup()

# Configure HPC resources
hpc_config = {
    'scheduler': 'slurm',
    'partition': 'gpu',
    'node_list': ['node[001-020]'],
    'job_config': {
        'time_limit': '12:00:00',
        'nodes_per_job': 4,
        'gpus_per_node': 2
    }
}

# Configure cloud resources
cloud_config = {
    'provider': 'aws',
    'instance_types': ['g4dn.xlarge', 'g4dn.2xlarge'],
    'regions': ['us-west-2', 'us-east-1'],
    'node_count': 10
}

# Set up hybrid cluster
all_nodes = hybrid_setup.setup_hybrid_cluster(hpc_config, cloud_config)
print(f"Hybrid cluster ready with {len(all_nodes)} total nodes")

# Execute large-scale calculation with cost optimization
molecules = load_massive_dataset()  # 50M molecules
cost_budget = 500.0  # $500 budget

execution_summary = hybrid_setup.execute_hybrid_calculation(
    molecules=molecules,
    descriptors="comprehensive",
    cost_budget=cost_budget
)

print(f"Hybrid execution completed:")
print(f"  Total molecules processed: {execution_summary['total_molecules']}")
print(f"  HPC jobs: {execution_summary['hpc_jobs']}")
print(f"  Cloud jobs: {execution_summary['cloud_jobs']}")
print(f"  Total cost: ${execution_summary['total_cost']:.2f}")
```

## Best Practices

### 1. Network Optimization
```python
# Optimize network communication for distributed computing
def optimize_network_settings():
    network_config = {
        'tcp_window_size': '64MB',
        'tcp_congestion_control': 'bbr',
        'infiniband_settings': {
            'mtu': 4096,
            'flow_control': True
        },
        'compression': 'lz4',  # Fast compression for data transfer
        'encryption': 'aes-128'  # Balance security and performance
    }
    return network_config
```

### 2. Fault Recovery
```python
# Implement comprehensive fault recovery
def setup_fault_recovery(distributed_manager):
    recovery_config = {
        'checkpoint_interval': 300,  # 5 minutes
        'max_retries': 3,
        'retry_backoff': 'exponential',
        'health_check_interval': 30,
        'automatic_failover': True
    }
    
    distributed_manager.configure_fault_recovery(recovery_config)
```

### 3. Resource Monitoring
```python
# Monitor resource usage across distributed cluster
def setup_cluster_monitoring():
    monitoring_config = {
        'metrics': ['cpu_usage', 'memory_usage', 'gpu_usage', 'network_io', 'storage_io'],
        'collection_interval': 10,  # seconds
        'alerting_thresholds': {
            'cpu_usage': 90,
            'memory_usage': 85,
            'gpu_usage': 95
        },
        'dashboard_enabled': True
    }
    return monitoring_config
```

---

*For specific deployment scenarios, see the [Cloud Deployment](Cloud-Deployment) and [HPC Integration](HPC-Integration) guides.*
