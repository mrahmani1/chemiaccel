# ChemiAccel Toxicology Examples

This directory contains comprehensive examples demonstrating ChemiAccel's AI-driven predictive toxicology and environmental fate modeling capabilities.

## Overview

ChemiAccel's toxicology module provides state-of-the-art computational tools for:

- **GPU-accelerated multi-endpoint toxicology prediction** (53.1)
- **Advanced environmental degradation pathway & product prediction** (53.2)  
- **Metabolic pathway and metabolite toxicity prediction** (53.3)
- **Holistic risk assessment and uncertainty quantification** (53.4)
- **Inverse design for safer-by-design chemicals** (53.5)

## Files in this Directory

### `comprehensive_toxicology_example.py`
Complete workflow demonstration including:
- Multi-endpoint toxicity prediction across species
- Environmental fate modeling and degradation pathways
- Metabolic pathway prediction and metabolite toxicity
- Holistic risk assessment with uncertainty quantification
- Safer-by-design chemical optimization
- Comprehensive report generation

### Example Output
The examples generate detailed HTML reports and comparative analyses showing:
- Toxicity profiles with confidence intervals
- Risk scores and category classifications  
- Environmental fate predictions
- Metabolic network analysis
- Optimization candidates with safety improvements

## Quick Start

```python
from chemi_accel.toxicology import ToxicologyCalculator, ToxicityEndpointType, SpeciesType

# Initialize calculator with GPU acceleration
calc = ToxicologyCalculator(backend="cuda", use_gpu=True)

# Predict comprehensive toxicity profile
toxicity_data = calc.predict_toxicity(
    "c1ccccc1",  # Benzene
    endpoints=[
        ToxicityEndpointType.ACUTE_ORAL,
        ToxicityEndpointType.CARCINOGENICITY,
        ToxicityEndpointType.AQUATIC_ACUTE
    ],
    species=[SpeciesType.RAT, SpeciesType.FISH],
    include_uncertainty=True
)

print(f"Overall risk score: {toxicity_data.overall_risk_score:.3f}")
```

## Key Features Demonstrated

### 1. Multi-Endpoint Toxicity Prediction
- Simultaneous prediction across 17+ toxicity endpoints
- Multiple species extrapolation (rat, mouse, fish, daphnia, etc.)
- GPU-accelerated neural networks with uncertainty quantification
- Feature importance analysis and model explanations

### 2. Environmental Fate Modeling
- Multi-compartment fate prediction (air, water, soil, sediment)
- Degradation pathway analysis (hydrolysis, photolysis, biodegradation)
- Persistence and bioaccumulation assessment
- Multimedia fugacity modeling

### 3. Metabolic Pathway Prediction
- CYP450 enzyme-mediated Phase I metabolism
- Phase II conjugation reactions (glucuronidation, sulfation)
- Metabolite toxicity prediction
- Drug-drug interaction assessment
- Pharmacokinetic simulation

### 4. Holistic Risk Assessment
- Integration of toxicity, exposure, and environmental fate
- Monte Carlo uncertainty propagation
- Multiple exposure scenario analysis
- Risk categorization and confidence assessment

### 5. Safer Design Optimization
- Multi-objective optimization for reduced toxicity
- Molecular transformation and bioisosteric replacement
- Constraint satisfaction with viable candidate identification
- Synthetic accessibility and similarity analysis

## Running the Examples

### Prerequisites
```bash
# Install ChemiAccel with toxicology dependencies
pip install chemi-accel[toxicology]

# Optional: Install additional dependencies for full functionality
pip install torch torchvision  # For GPU-accelerated models
pip install rdkit-pypi  # For molecular feature extraction
pip install matplotlib seaborn  # For visualizations
```

### Basic Usage
```bash
# Run comprehensive example
python comprehensive_toxicology_example.py

# This will generate:
# - Console output with analysis progress
# - HTML reports for detailed results  
# - Risk comparison tables
# - Optimization candidates
```

### Advanced Configuration
```python
# Configure for different hardware
calc = ToxicologyCalculator(
    backend="cuda",  # "cuda", "jax", "cpu"
    use_gpu=True,
    uncertainty_quantification=True
)

# Customize prediction endpoints
endpoints = [
    ToxicityEndpointType.ACUTE_ORAL,
    ToxicityEndpointType.CARCINOGENICITY, 
    ToxicityEndpointType.REPRODUCTIVE_TOXICITY,
    ToxicityEndpointType.AQUATIC_CHRONIC
]

# Define exposure scenarios
from chemi_accel.toxicology import ExposureScenario, ExposureRoute

exposure_scenarios = [
    ExposureScenario(
        scenario_name="consumer_oral",
        exposure_route=ExposureRoute.ORAL,
        exposure_level=0.01,  # mg/kg/day
        exposure_duration=365,
        exposure_frequency=1.0,
        population="adults"
    )
]
```

## Expected Results

### Toxicity Predictions
```
Benzene Toxicity Profile:
├── Acute Oral (Rat): 930 mg/kg (High confidence)
├── Carcinogenicity: 0.95 probability (High confidence)  
├── Aquatic Acute: 15 mg/L (Moderate confidence)
└── Overall Risk Score: 0.847
```

### Risk Assessment
```
Risk Assessment Summary:
├── Risk Category: Very High
├── Overall Risk: 0.847 ± 0.12
├── Key Drivers: 
│   ├── High carcinogenicity potential
│   ├── Acute toxicity concerns
│   └── Environmental persistence
└── Recommendations: Consider alternatives
```

### Safer Design Results  
```
Optimization Results:
├── Viable Candidates: 7/20
├── Best Candidate: c1ccc(O)cc1 (phenol)
├── Risk Reduction: 45% improvement
├── Similarity: 0.89 to parent
└── Constraints Met: All hard constraints satisfied
```

## Visualization and Reporting

The examples generate comprehensive visualizations including:

- **Toxicity Profiles**: Bar charts with uncertainty intervals
- **Risk Landscapes**: 2D scatter plots of compound risk space  
- **Optimization Progress**: Pareto frontiers and constraint satisfaction
- **Environmental Fate**: Compartment distribution over time
- **Metabolic Networks**: Pathway diagrams with metabolite toxicity

## Integration with Other ChemiAccel Modules

```python
# Combine with molecular descriptors
from chemi_accel import Calculator

# Calculate descriptors for QSAR analysis
calc = Calculator()
descriptors = calc.pandas(["c1ccccc1", "CCO"])

# Use with toxicology predictions
tox_calc = ToxicologyCalculator()
toxicity = tox_calc.predict_toxicity(["c1ccccc1", "CCO"])

# Correlate descriptors with toxicity endpoints
correlation_analysis = correlate_descriptors_toxicity(descriptors, toxicity)
```

## Performance Notes

- **GPU Acceleration**: 10-100x speedup for large datasets
- **Batch Processing**: Optimized for hundreds to thousands of compounds
- **Memory Efficiency**: Streaming processing for large-scale analysis
- **Uncertainty Quantification**: Monte Carlo with 1000-10000 samples typical

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size or use CPU backend
   calc = ToxicologyCalculator(backend="cpu")
   ```

2. **Missing Dependencies**
   ```bash
   # Install optional dependencies
   pip install rdkit-pypi torch matplotlib
   ```

3. **Model Loading Issues**
   ```python
   # Models downloaded on first use - ensure internet connection
   # Or set CHEMI_ACCEL_MODEL_PATH environment variable  
   ```

### Performance Optimization

```python
# For large datasets
calc = ToxicologyCalculator(
    backend="cuda",
    batch_size=256,  # Adjust based on GPU memory
    num_workers=4    # Parallel data loading
)

# Enable model caching
calc.enable_model_caching(cache_dir="./models/")
```

## Citation

If you use ChemiAccel's toxicology capabilities in your research, please cite:

```bibtex
@software{chemi_accel_toxicology,
  title={ChemiAccel: AI-Driven Predictive Toxicology and Environmental Fate Modeling},
  author={ChemiAccel Development Team},
  year={2024},
  url={https://github.com/chemi-accel/chemi-accel}
}
```

## Support

For questions, issues, or feature requests:
- GitHub Issues: https://github.com/chemi-accel/chemi-accel/issues
- Documentation: https://chemi-accel.readthedocs.io/
- Community Forum: https://community.chemi-accel.org/

## License

ChemiAccel is released under the MIT License. See LICENSE file for details.
