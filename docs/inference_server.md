# ChemiAccel High-Performance Inference Server

## Overview

The ChemiAccel High-Performance Inference Server is a production-ready REST API service for molecular descriptor calculations and property predictions. Built with FastAPI, it provides scalable, GPU-accelerated molecular analysis with advanced features like dynamic batching, model preloading, and containerized deployment.

## Key Features

### 🚀 High Performance
- **Dynamic Batching**: Automatic request batching for optimal GPU utilization
- **Model Preloading**: Zero cold-start latency with warm-up procedures
- **GPU Acceleration**: CUDA and RAPIDS integration for maximum throughput
- **Async Processing**: Non-blocking request handling with concurrent processing

### 🔧 Production Ready
- **FastAPI Framework**: Automatic OpenAPI documentation and validation
- **Containerized Deployment**: Docker and Kubernetes support
- **Load Balancing**: Built-in support for horizontal scaling
- **Health Monitoring**: Comprehensive metrics and health check endpoints

### 📊 Monitoring & Observability
- **Prometheus Metrics**: Built-in metrics collection
- **Performance Monitoring**: Real-time system and GPU utilization tracking
- **Request Tracing**: Detailed request/response logging and timing

### 🛡️ Enterprise Features
- **Rate Limiting**: Configurable request rate limiting
- **Error Handling**: Robust error recovery and reporting
- **Security**: CORS, authentication-ready architecture
- **Scalability**: Horizontal and vertical scaling support

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -e .[inference]

# Start the server
python -m chemi_accel.inference.server --host 0.0.0.0 --port 8000

# Server will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

### Docker Deployment

```bash
# Build and run with Docker Compose
cd docker
docker-compose -f docker-compose.inference.yml up -d

# GPU-enabled deployment (requires NVIDIA Docker)
docker-compose -f docker-compose.inference.yml --profile monitoring up -d
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes cluster
kubectl apply -f k8s/inference-deployment.yaml

# Check deployment status
kubectl get pods -n chemi-accel
```

## API Endpoints

### Core Prediction Endpoints

#### Single Molecule Prediction
```http
POST /predict/single
Content-Type: application/json

{
    "smiles": "CCO",
    "descriptors": ["molecular_weight", "logp", "tpsa"],
    "include_properties": true
}
```

#### Batch Prediction
```http
POST /predict/batch
Content-Type: application/json

{
    "smiles": ["CCO", "c1ccccc1", "CC(=O)O"],
    "descriptors": ["molecular_weight", "logp", "tpsa"],
    "include_properties": true,
    "batch_size": 100
}
```

#### Streaming Prediction
```http
POST /predict/stream
Content-Type: application/json

{
    "smiles": ["CCO", "c1ccccc1", ...],  // Large dataset
    "descriptors": ["molecular_weight", "logp", "tpsa"],
    "include_properties": true
}
```

### Monitoring Endpoints

#### Health Check
```http
GET /health
```

Response:
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "uptime": 3600.0,
    "gpu_available": true,
    "memory_usage": {
        "rss_mb": 1024.5,
        "gpu_used_mb": 2048.0,
        "gpu_total_mb": 8192.0
    }
}
```

#### Performance Metrics
```http
GET /metrics
```

#### Prometheus Metrics
```http
GET /metrics/prometheus
```

### Utility Endpoints

#### Available Descriptors
```http
GET /descriptors
```

Response:
```json
{
    "basic": ["molecular_weight", "logp", "tpsa", "heavy_atom_count"],
    "extended": ["molecular_weight", "logp", "tpsa", "heavy_atom_count", "rotatable_bonds", "hbd", "hba"],
    "all": ["molecular_weight", "logp", "tpsa", "heavy_atom_count", "rotatable_bonds", "hbd", "hba", "aromatic_rings"]
}
```

## Configuration

### Server Configuration

The server can be configured through environment variables or command-line arguments:

```bash
# Environment variables
export CHEMI_ACCEL_ENV=production
export CHEMI_ACCEL_LOG_LEVEL=INFO
export CHEMI_ACCEL_MAX_BATCH_SIZE=1000
export CHEMI_ACCEL_PRELOAD_MODELS=true
export CHEMI_ACCEL_USE_GPU=true

# Command line
python -m chemi_accel.inference.server \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level INFO \
    --max-batch-size 1000
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `host` | `0.0.0.0` | Host to bind to |
| `port` | `8000` | Port to bind to |
| `workers` | `1` | Number of worker processes |
| `max-batch-size` | `1000` | Maximum batch size for dynamic batching |
| `max-batch-wait-time` | `0.1` | Maximum wait time for batching (seconds) |
| `preload-models` | `true` | Enable model preloading |
| `log-level` | `INFO` | Logging level |
| `enable-metrics` | `true` | Enable Prometheus metrics |
| `cors-origins` | `["*"]` | CORS allowed origins |

## Performance Optimization

### Dynamic Batching

The server automatically batches incoming requests for optimal GPU utilization:

- **Adaptive Batch Sizing**: Automatically adjusts batch size based on system load
- **Timeout-Based Batching**: Processes batches when timeout is reached
- **GPU Memory Management**: Optimizes memory usage for maximum throughput

### Model Preloading

Models are preloaded on server startup to eliminate cold-start latency:

- **Warm-up Procedures**: Runs sample calculations during initialization
- **Memory Pre-allocation**: Pre-allocates GPU memory for optimal performance
- **Caching**: Intelligent caching of intermediate results

### Load Balancing

The server supports multiple load balancing strategies:

- **Round Robin**: Distributes requests evenly across instances
- **Least Connections**: Routes to instance with fewest active connections
- **GPU Utilization**: Routes based on GPU utilization levels

## Monitoring and Observability

### Prometheus Metrics

The server exposes comprehensive metrics for monitoring:

```prometheus
# Request metrics
chemi_accel_requests_total{method="POST",endpoint="/predict/batch"}
chemi_accel_request_duration_seconds{method="POST",endpoint="/predict/batch"}
chemi_accel_errors_total{error_type="validation_error"}

# System metrics
chemi_accel_gpu_utilization
chemi_accel_memory_usage_bytes{type="gpu"}
chemi_accel_batch_size{type="current"}
```

### Health Checks

The server provides multiple health check endpoints:

- **Liveness Probe**: `/health` - Basic server health
- **Readiness Probe**: `/health` - Service readiness
- **Startup Probe**: `/health` - Startup completion

### Logging

Structured logging with configurable levels:

```json
{
    "timestamp": "2024-01-01T12:00:00Z",
    "level": "INFO",
    "message": "Processed batch request",
    "request_id": "req_123456",
    "batch_size": 100,
    "processing_time": 0.150,
    "gpu_utilization": 85.5
}
```

## Load Testing

### Running Load Tests

```bash
# Basic load test
python tests/load/test_inference_performance.py \
    --server http://localhost:8000 \
    --users 10 \
    --duration 300

# Advanced load test
python -m pytest tests/load/test_inference_performance.py \
    --server http://localhost:8000 \
    --users 50 \
    --duration 600 \
    -v --tb=short
```

### Performance Benchmarks

Expected performance characteristics:

| Metric | GPU (V100) | GPU (T4) | CPU Only |
|--------|------------|----------|----------|
| Single Requests | 100 req/s | 50 req/s | 20 req/s |
| Batch Throughput | 5000 mol/s | 2000 mol/s | 500 mol/s |
| P95 Latency | <100ms | <200ms | <500ms |
| Max Batch Size | 10000 | 5000 | 1000 |

## Deployment Scenarios

### Development

```bash
# Local development with hot reload
uvicorn chemi_accel.inference.server:app --reload --host 0.0.0.0 --port 8000
```

### Staging

```bash
# Docker deployment with monitoring
docker-compose -f docker-compose.inference.yml --profile monitoring up -d
```

### Production

```bash
# Kubernetes deployment with autoscaling
kubectl apply -f k8s/inference-deployment.yaml

# Verify deployment
kubectl get pods -n chemi-accel
kubectl get hpa -n chemi-accel
```

### High Availability

For high availability deployments:

1. **Multiple Replicas**: Run multiple server instances
2. **Load Balancing**: Use NGINX or Traefik for load distribution
3. **Health Monitoring**: Configure proper health checks
4. **Auto-scaling**: Set up HPA based on CPU/GPU utilization
5. **Resource Limits**: Configure appropriate resource requests/limits

## Troubleshooting

### Common Issues

#### Server Won't Start
```bash
# Check logs
docker logs chemi-accel-inference-gpu

# Common causes:
# - CUDA driver issues
# - Memory allocation problems
# - Port conflicts
```

#### Poor Performance
```bash
# Check metrics
curl http://localhost:8000/metrics

# Check GPU utilization
nvidia-smi

# Common causes:
# - Small batch sizes
# - CPU bottlenecks
# - Memory limitations
```

#### Health Check Failures
```bash
# Check health endpoint
curl http://localhost:8000/health

# Check server logs
kubectl logs -n chemi-accel deployment/chemi-accel-inference-gpu

# Common causes:
# - Model loading failures
# - Resource exhaustion
# - Network connectivity issues
```

### Performance Tuning

#### Batch Size Optimization
```python
# Test different batch sizes
for batch_size in [10, 50, 100, 500, 1000]:
    # Measure throughput
    result = benchmark_batch_size(batch_size)
    print(f"Batch size {batch_size}: {result.throughput} mol/s")
```

#### Memory Optimization
```bash
# Monitor memory usage
nvidia-smi -l 1

# Adjust batch size based on memory
export CHEMI_ACCEL_MAX_BATCH_SIZE=500  # Reduce if OOM errors
```

#### Scaling Configuration
```yaml
# Kubernetes HPA configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: chemi-accel-hpa
spec:
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
```

## Security Considerations

### Authentication
```python
# Add authentication middleware
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.middleware("http")
async def authenticate_request(request: Request, call_next):
    # Implement authentication logic
    return await call_next(request)
```

### Rate Limiting
```python
# Configure rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict/batch")
@limiter.limit("10/minute")
async def predict_batch(request: Request, ...):
    ...
```

### Network Security
```yaml
# Network policy for Kubernetes
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: chemi-accel-network-policy
spec:
  podSelector:
    matchLabels:
      app: chemi-accel-inference
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: allowed-namespace
    ports:
    - protocol: TCP
      port: 8000
```

## API Client Examples

### Python Client

```python
import requests
import asyncio
import aiohttp

class ChemiAccelClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def predict_single(self, smiles, descriptors=None):
        payload = {
            "smiles": smiles,
            "descriptors": descriptors,
            "include_properties": True
        }
        response = requests.post(f"{self.base_url}/predict/single", json=payload)
        return response.json()
    
    def predict_batch(self, smiles_list, descriptors=None):
        payload = {
            "smiles": smiles_list,
            "descriptors": descriptors,
            "include_properties": True
        }
        response = requests.post(f"{self.base_url}/predict/batch", json=payload)
        return response.json()
    
    async def predict_stream(self, smiles_list, descriptors=None):
        payload = {
            "smiles": smiles_list,
            "descriptors": descriptors,
            "include_properties": True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/predict/stream", json=payload) as response:
                async for line in response.content:
                    if line.startswith(b'data: '):
                        yield json.loads(line[6:].decode())

# Usage
client = ChemiAccelClient()

# Single prediction
result = client.predict_single("CCO")
print(result)

# Batch prediction
results = client.predict_batch(["CCO", "c1ccccc1", "CC(=O)O"])
print(results)

# Streaming prediction
async def stream_example():
    async for chunk in client.predict_stream(large_smiles_list):
        print(f"Processed: {chunk['processed']}/{chunk['total']}")

asyncio.run(stream_example())
```

### JavaScript Client

```javascript
class ChemiAccelClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async predictSingle(smiles, descriptors = null) {
        const response = await fetch(`${this.baseUrl}/predict/single`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                smiles,
                descriptors,
                include_properties: true
            })
        });
        return response.json();
    }
    
    async predictBatch(smilesList, descriptors = null) {
        const response = await fetch(`${this.baseUrl}/predict/batch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                smiles: smilesList,
                descriptors,
                include_properties: true
            })
        });
        return response.json();
    }
    
    async* predictStream(smilesList, descriptors = null) {
        const response = await fetch(`${this.baseUrl}/predict/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                smiles: smilesList,
                descriptors,
                include_properties: true
            })
        });
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    yield JSON.parse(line.slice(6));
                }
            }
        }
    }
}

// Usage
const client = new ChemiAccelClient();

// Single prediction
client.predictSingle('CCO').then(result => console.log(result));

// Batch prediction
client.predictBatch(['CCO', 'c1ccccc1']).then(results => console.log(results));

// Streaming prediction
(async () => {
    for await (const chunk of client.predictStream(largeSMILESList)) {
        console.log(`Processed: ${chunk.processed}/${chunk.total}`);
    }
})();
```

## Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/chemi-accel.git
cd chemi-accel

# Install development dependencies
pip install -e .[dev,inference]

# Run tests
pytest tests/

# Run load tests
python tests/load/test_inference_performance.py

# Start development server
python -m chemi_accel.inference.server --log-level DEBUG
```

### Testing

```bash
# Unit tests
pytest tests/unit/test_inference_server.py

# Integration tests
pytest tests/integration/test_api_endpoints.py

# Performance tests
pytest tests/load/test_inference_performance.py

# Coverage report
pytest --cov=chemi_accel.inference tests/
```

## Support

For support and questions:

- 📖 Documentation: [docs.chemi-accel.com](https://docs.chemi-accel.com)
- 🐛 Bug Reports: [GitHub Issues](https://github.com/your-org/chemi-accel/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-org/chemi-accel/discussions)
- 📧 Email: support@chemi-accel.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.