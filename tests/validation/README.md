# ChemiAccel Validation Test Suite

This directory contains comprehensive validation tests for ChemiAccel, ensuring accuracy, robustness, and performance across all components.

## Overview

The validation suite provides systematic testing of ChemiAccel's descriptor calculations against established reference implementations (RDKit, Mordred) with comprehensive coverage of:

- **Reference Implementation Validation**: Bit-level accuracy testing against RDKit/Mordred
- **Statistical Validation**: Large-scale statistical analysis of descriptor distributions
- **Edge Case Testing**: Robustness with challenging molecular structures
- **Polymer Validation**: Specialized testing for polymer processing capabilities
- **Error Handling**: Graceful failure and recovery testing
- **Stress Testing**: Memory management and performance under extreme conditions

## Test Modules

### Core Validation Tests

#### `test_reference_validation.py`
Comprehensive validation against reference implementations:
- Systematic comparison with RDKit/Mordred for all descriptors
- Configurable tolerance levels for different descriptor types
- Performance benchmarking and timing analysis
- Batch vs individual calculation consistency testing

#### `test_bit_level_accuracy.py`
Bit-level accuracy validation for critical descriptors:
- Exact numerical reproducibility testing
- Floating-point precision analysis
- Integer descriptor exactness verification
- Numerical stability across different inputs

#### `test_statistical_validation.py`
Statistical analysis of descriptor calculations:
- Distribution analysis across large molecular datasets
- Correlation testing with reference implementations
- Outlier detection and statistical property validation
- Performance statistics and scaling analysis

#### `test_edge_case_validation.py`
Robustness testing with challenging molecular structures:
- Invalid SMILES handling
- Charged species and radicals
- Unusual bonding patterns and strained rings
- Large molecules and complex stereochemistry
- Pathological cases and error recovery

#### `test_polymer_validation.py`
Polymer-specific validation against experimental databases:
- Monomer analysis and reactivity prediction
- Oligomer processing and conformational analysis
- Ring-opening polymerization validation
- Polymer-specific descriptor accuracy

### Robustness and Stress Testing

#### `test_error_handling_robustness.py`
Comprehensive error handling and system robustness:
- Graceful handling of invalid molecular structures
- GPU fallback behavior testing
- Concurrent access safety
- Resource limit handling
- Memory safety validation

#### `test_stress_testing.py`
Performance and memory management under stress:
- Memory scaling with molecule complexity
- Polymer chain length performance testing
- Concurrent processing stress testing
- Resource exhaustion recovery
- Memory leak detection

## Running the Tests

### Quick Validation
Run core validation tests:
```bash
python -m pytest tests/validation/test_reference_validation.py -v
python -m pytest tests/validation/test_bit_level_accuracy.py -v
```

### Comprehensive Validation
Run all validation tests:
```bash
python tests/validation/run_comprehensive_validation.py
```

### Include Stress Testing
Run with slow/stress tests:
```bash
python tests/validation/run_comprehensive_validation.py --include-slow
```

### Custom Output Directory
Specify custom output directory:
```bash
python tests/validation/run_comprehensive_validation.py --output-dir my_results
```

## Test Configuration

### Tolerance Configuration
Modify tolerance levels in `test_reference_validation.py`:
```python
tolerance_config = {
    "molecular_weight": 1e-6,  # Exact match required
    "tpsa": 1e-4,             # Small numerical tolerance
    "logp": 1e-3,             # Moderate tolerance
    # ... additional descriptors
}
```

### Test Dataset Customization
Modify test molecules in individual test files:
```python
VALIDATION_DATASET = {
    "simple_molecules": ["C", "CC", "CCO", ...],
    "aromatic_compounds": ["c1ccccc1", "Cc1ccccc1", ...],
    # ... additional categories
}
```

## Validation Results

### Result Structure
```json
{
  "summary": {
    "overall_status": "PASSED|PARTIAL|FAILED",
    "tests_passed": 6,
    "tests_failed": 0,
    "total_duration": 45.2,
    "pass_rate": 1.0
  },
  "detailed_results": {
    "reference_validation": { ... },
    "bit_level_accuracy": { ... },
    "statistical_validation": { ... },
    "edge_case_validation": { ... },
    "polymer_validation": { ... },
    "error_handling": { ... },
    "stress_testing": { ... }
  }
}
```

### Interpreting Results

#### Reference Validation
- **Exact Matches**: Number of descriptors with bit-perfect accuracy
- **Within Tolerance**: Descriptors within acceptable numerical tolerance
- **Performance**: Speed comparison with reference implementations

#### Statistical Validation
- **Distribution Properties**: Mean, std, skewness, kurtosis of descriptor values
- **Correlation**: Pearson/Spearman correlation with reference implementations
- **Outlier Detection**: Identification of unusual descriptor values

#### Edge Case Validation
- **Graceful Failures**: Percentage of invalid molecules handled gracefully
- **Error Types**: Classification of error types encountered
- **Recovery**: System recovery after encountering problematic molecules

#### Polymer Validation
- **Monomer Analysis**: Success rate for monomer reactivity analysis
- **Oligomer Processing**: Accuracy of polymer chain analysis
- **Property Prediction**: Validation against experimental polymer databases

## Performance Benchmarks

### Expected Performance Targets
- **Molecular Weight**: >1000 molecules/second
- **TPSA**: >500 molecules/second  
- **Morgan Fingerprints**: >200 molecules/second
- **Topological Descriptors**: >100 molecules/second
- **3D Descriptors**: >50 molecules/second

### Memory Usage Targets
- **Basic Descriptors**: <1MB per 1000 molecules
- **Fingerprints**: <10MB per 1000 molecules
- **3D Descriptors**: <50MB per 1000 molecules
- **Polymer Analysis**: <100MB per 100 oligomers

## Continuous Integration

### CI Configuration
The validation suite is designed for CI/CD integration:
```yaml
# Example GitHub Actions configuration
- name: Run ChemiAccel Validation
  run: |
    python tests/validation/run_comprehensive_validation.py
    if [ $? -ne 0 ]; then
      echo "Validation failed"
      exit 1
    fi
```

### Test Markers
Use pytest markers for selective testing:
```bash
# Run only fast tests
pytest tests/validation/ -m "not slow"

# Run only GPU tests (if available)
pytest tests/validation/ -m "gpu"

# Run only polymer tests
pytest tests/validation/ -m "polymer"
```

## Troubleshooting

### Common Issues

#### RDKit Not Available
```
ImportError: No module named 'rdkit'
```
Install RDKit: `conda install -c conda-forge rdkit-python`

#### GPU Tests Failing
```
BackendUnavailableError: CUDA backend not available
```
Ensure CUDA and cupy are properly installed, or run CPU-only tests.

#### Memory Issues
```
MemoryError: Unable to allocate array
```
Reduce test dataset size or run tests individually.

#### Slow Performance
```
Performance too slow: 5.2 molecules/sec
```
Check system resources and consider running on more powerful hardware.

### Debug Mode
Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Profiling
Profile memory usage:
```bash
python -m memory_profiler tests/validation/run_comprehensive_validation.py
```

## Contributing

### Adding New Validation Tests
1. Create test file following naming convention: `test_*_validation.py`
2. Implement validation framework class
3. Add comprehensive test cases with proper assertions
4. Include performance benchmarks
5. Update this README with test description

### Test Quality Guidelines
- **Comprehensive Coverage**: Test all major code paths
- **Realistic Data**: Use diverse, realistic molecular structures
- **Clear Assertions**: Provide informative error messages
- **Performance Aware**: Include timing and memory usage checks
- **Reproducible**: Ensure tests are deterministic and reproducible

### Validation Standards
- **Accuracy**: ≥99% agreement with reference implementations
- **Robustness**: ≥90% graceful failure rate for invalid inputs
- **Performance**: Meet or exceed performance targets
- **Memory**: No memory leaks or excessive usage
- **Concurrency**: Thread-safe operation under concurrent access

## References

- [RDKit Documentation](https://www.rdkit.org/docs/)
- [Mordred Documentation](https://mordred-descriptor.github.io/documentation/)
- [ChemiAccel Design Document](../../.kiro/specs/chemi-accel/design.md)
- [ChemiAccel Requirements](../../.kiro/specs/chemi-accel/requirements.md)
