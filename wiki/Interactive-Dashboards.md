# Interactive Dashboards

ChemiAccel provides sophisticated interactive visualization dashboards for analyzing molecular properties, optimization results, and computational performance. These web-based tools offer real-time exploration of complex datasets.

## Overview

The visualization system includes:
- **Multi-Objective Optimization Dashboards** - Pareto front analysis and trade-off visualization
- **Molecular Generation Visualization** - Generated molecule galleries and property analysis
- **Polymer Property Dashboards** - Composition-property relationships and thermal/mechanical analysis
- **Cloud Monitoring Interface** - Real-time job monitoring and resource utilization

## Multi-Objective Optimization Dashboards

### Basic Usage

```python
from chemi_accel.visualization import MultiObjectiveVisualizer
from chemi_accel.ai.multi_objective_optimization import OptimizationResult
import numpy as np

# Initialize visualizer
viz = MultiObjectiveVisualizer()

# Example optimization data
objectives = np.random.rand(100, 2)  # 100 solutions, 2 objectives
pareto_mask = np.random.choice([True, False], 100, p=[0.2, 0.8])  # 20% Pareto optimal

# Create 2D Pareto front plot
fig = viz.plot_pareto_front_2d(
    objectives=objectives,
    pareto_mask=pareto_mask,
    objective_names=['Solubility', 'Permeability'],
    title='Drug-like Property Optimization'
)

fig.show()
```

### Advanced Dashboard Creation

```python
# Create comprehensive optimization dashboard
result = OptimizationResult(
    objectives=objectives,
    parameters=np.random.rand(100, 5),  # 5 design parameters
    pareto_front=objectives[pareto_mask],
    hypervolume=0.75,
    convergence_history=[0.3, 0.5, 0.65, 0.72, 0.75],
    generation_data=[]
)

# Generate complete dashboard
dashboard = viz.create_interactive_dashboard(
    result=result,
    objective_names=['Solubility (log S)', 'Permeability (log P)'],
    parameter_names=['MW', 'LogP', 'TPSA', 'HBD', 'HBA']
)

# Save as HTML
viz.save_dashboard_html(dashboard, 'optimization_results.html')
```

### 3D Pareto Front Visualization

```python
# For 3-objective optimization
objectives_3d = np.random.rand(100, 3)
pareto_mask_3d = np.random.choice([True, False], 100, p=[0.15, 0.85])

fig_3d = viz.plot_pareto_front_3d(
    objectives=objectives_3d,
    pareto_mask=pareto_mask_3d,
    objective_names=['Efficacy', 'Safety', 'Druggability'],
    title='Multi-Objective Drug Design'
)

fig_3d.show()
```

### Convergence Analysis

```python
# Plot optimization convergence
convergence_data = {
    'Hypervolume': [0.1, 0.3, 0.5, 0.65, 0.72, 0.75, 0.76],
    'Best Objective 1': [0.2, 0.4, 0.6, 0.75, 0.82, 0.85, 0.86],
    'Best Objective 2': [0.3, 0.5, 0.7, 0.78, 0.83, 0.87, 0.88]
}

fig_conv = viz.plot_convergence_history(convergence_data)
fig_conv.show()
```

## Molecular Generation Visualization

### Generated Molecule Gallery

```python
from chemi_accel.visualization import MolecularGenerationVisualizer, GeneratedMolecule
from rdkit import Chem

# Initialize visualizer
mol_viz = MolecularGenerationVisualizer()

# Example generated molecules
molecules = []
smiles_list = [
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",  # Drug-like
    "C1=CC=C(C=C1)C2=CC=CC=C2",  # Biphenyl
]

for i, smiles in enumerate(smiles_list):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        molecules.append(GeneratedMolecule(
            smiles=smiles,
            mol=mol,
            properties={
                'molecular_weight': Chem.rdMolDescriptors.CalcExactMolWt(mol),
                'logp': Chem.rdMolDescriptors.CalcCrippenDescriptors(mol)[0],
                'tpsa': Chem.rdMolDescriptors.CalcTPSA(mol)
            },
            generation_score=np.random.rand(),
            validity_score=1.0,
            novelty_score=np.random.rand(),
            diversity_score=np.random.rand()
        ))

# Create molecule gallery
gallery = mol_viz.create_molecule_gallery(
    molecules=molecules,
    sort_by='generation_score',
    n_display=20
)

gallery.show()
```

### Property Distribution Analysis

```python
# Analyze property distributions
prop_dist = mol_viz.plot_property_distributions(
    molecules=molecules,
    properties=['molecular_weight', 'logp', 'tpsa'],
    title='Generated Molecule Properties'
)

prop_dist.show()
```

### Chemical Space Visualization

```python
# 2D chemical space plot
chem_space = mol_viz.plot_chemical_space_2d(
    molecules=molecules,
    x_property='molecular_weight',
    y_property='logp',
    color_by='generation_score',
    title='Chemical Space Coverage'
)

chem_space.show()
```

### Comprehensive Generation Dashboard

```python
# Create complete molecular generation analysis
gen_dashboard = mol_viz.create_comprehensive_dashboard(
    molecules=molecules,
    properties=['molecular_weight', 'logp', 'tpsa']
)

# Export as HTML
mol_viz.export_dashboard_html(gen_dashboard, 'molecular_generation_analysis.html')
```

## Polymer Property Dashboards

### Basic Polymer Analysis

```python
from chemi_accel.visualization import PolymerPropertyDashboard, PolymerData

# Initialize dashboard
polymer_dash = PolymerPropertyDashboard()

# Example polymer data
polymers = [
    PolymerData(
        name="Polyethylene",
        monomers=["ethylene"],
        composition={"ethylene": 1.0},
        properties={
            'glass_transition_temp': -120,
            'melting_temp': 135,
            'youngs_modulus': 1.1,
            'tensile_strength': 30,
            'density': 0.92
        },
        molecular_weight=50000,
        structure_representation="CC"
    ),
    PolymerData(
        name="Polystyrene", 
        monomers=["styrene"],
        composition={"styrene": 1.0},
        properties={
            'glass_transition_temp': 100,
            'melting_temp': 240,
            'youngs_modulus': 3.0,
            'tensile_strength': 45,
            'density': 1.05
        },
        molecular_weight=75000,
        structure_representation="CC(c1ccccc1)"
    )
]

# Create property overview
overview = polymer_dash.create_property_overview_dashboard(polymers)
overview.show()
```

### Thermal Properties Analysis

```python
# Thermal properties dashboard
thermal = polymer_dash.create_thermal_properties_dashboard(
    polymers=polymers,
    title='Polymer Thermal Analysis'
)

thermal.show()
```

### Mechanical Properties Analysis

```python
# Mechanical properties dashboard
mechanical = polymer_dash.create_mechanical_properties_dashboard(
    polymers=polymers,
    title='Mechanical Property Analysis'
)

mechanical.show()
```

### Copolymer Analysis

```python
from chemi_accel.visualization import CopolymerData

# Example copolymer data
copolymers = []
for ratio_a in np.linspace(0, 1, 21):
    ratio_b = 1 - ratio_a
    copolymers.append(CopolymerData(
        monomer_a="styrene",
        monomer_b="acrylonitrile", 
        ratio_a=ratio_a,
        ratio_b=ratio_b,
        properties={
            'glass_transition_temp': 100 * ratio_a + 104 * ratio_b,
            'tensile_strength': 45 * ratio_a + 60 * ratio_b
        }
    ))

# Composition-property relationship
comp_prop = polymer_dash.create_composition_property_plot(
    copolymers=copolymers,
    property_name='glass_transition_temp',
    title='Styrene-Acrylonitrile Copolymer Tg'
)

comp_prop.show()
```

### Complete Polymer Dashboard

```python
# Generate comprehensive polymer dashboard
complete_dashboard = polymer_dash.create_comprehensive_polymer_dashboard(
    polymers=polymers,
    copolymers=copolymers
)

# Export to HTML
polymer_dash.export_dashboard_html(complete_dashboard, 'polymer_analysis.html')
```

## Cloud Monitoring Dashboard

### Job Status Monitoring

```python
from chemi_accel.visualization import CloudJobMonitor, CloudJob
from datetime import datetime, timedelta

# Initialize monitor
cloud_monitor = CloudJobMonitor()

# Example cloud jobs
jobs = []
platforms = ['aws', 'gcp', 'azure']
statuses = ['completed', 'running', 'failed', 'queued']

for i in range(50):
    job = CloudJob(
        job_id=f"job_{i:03d}",
        job_name=f"molecular_analysis_{i}",
        status=np.random.choice(statuses, p=[0.7, 0.15, 0.1, 0.05]),
        platform=np.random.choice(platforms),
        instance_type=np.random.choice(['g4dn.xlarge', 'n1-standard-4', 'Standard_NC6']),
        start_time=datetime.now() - timedelta(hours=np.random.randint(1, 48)),
        duration_minutes=np.random.exponential(30),
        cost_usd=np.random.exponential(5),
        cpu_utilization=np.random.uniform(20, 95),
        memory_utilization=np.random.uniform(30, 90),
        gpu_utilization=np.random.uniform(40, 98) if np.random.random() > 0.3 else None,
        molecules_processed=np.random.randint(100, 10000),
        region=np.random.choice(['us-east-1', 'europe-west1', 'eastus'])
    )
    jobs.append(job)

# Create job status dashboard  
status_dash = cloud_monitor.create_job_status_dashboard(jobs)
status_dash.show()
```

### Resource Utilization Analysis

```python
# Resource utilization visualization
resource_util = cloud_monitor.create_resource_utilization_plot(
    jobs=jobs,
    title='Cloud Resource Utilization'
)

resource_util.show()
```

### Cost Analysis

```python
# Cost analysis dashboard
cost_analysis = cloud_monitor.create_cost_analysis_dashboard(
    jobs=jobs,
    title='Cloud Computing Cost Analysis'
)

cost_analysis.show()
```

### Performance Metrics

```python
# Performance metrics analysis
performance = cloud_monitor.create_performance_metrics_plot(
    jobs=jobs,
    title='Cloud Job Performance Analysis'
)

performance.show()
```

### Complete Monitoring Dashboard

```python
# Generate comprehensive monitoring dashboard
monitoring_dashboard = cloud_monitor.create_comprehensive_monitoring_dashboard(
    jobs=jobs
)

# Export with auto-refresh capability
cloud_monitor.export_monitoring_dashboard_html(
    monitoring_dashboard, 
    'cloud_monitoring.html'
)
```

## Dashboard Customization

### Theming

```python
# Use different themes
viz_dark = MultiObjectiveVisualizer(theme="plotly_dark")
viz_white = MultiObjectiveVisualizer(theme="plotly_white")
viz_minimal = MultiObjectiveVisualizer(theme="simple_white")
```

### Custom Colors

```python
# Customize colors
viz.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
viz.platform_colors = {
    'aws': '#FF9900',
    'gcp': '#4285F4', 
    'azure': '#0078D4',
    'local': '#28a745'
}
```

### Interactive Features

All dashboards support:
- **Hover information** - Detailed data on hover
- **Zoom and pan** - Interactive exploration
- **Selection tools** - Select data points for analysis
- **Export options** - Save as PNG, SVG, HTML, or PDF
- **Real-time updates** - Live data refresh capabilities

## Integration with Jupyter

### Notebook Integration

```python
# Display in Jupyter notebooks
import plotly.offline as pyo
pyo.init_notebook_mode(connected=True)

# Show interactive plots inline
fig.show()

# Or use IPython display
from IPython.display import display, HTML
display(HTML(fig.to_html(include_plotlyjs=True)))
```

### Widget Integration

```python
from ipywidgets import interact, Dropdown, IntSlider

@interact(
    backend=Dropdown(options=['cpu', 'cuda', 'jax'], value='cuda'),
    n_molecules=IntSlider(min=10, max=1000, value=100),
    property_x=Dropdown(options=['molecular_weight', 'logp', 'tpsa'], value='molecular_weight'),
    property_y=Dropdown(options=['molecular_weight', 'logp', 'tpsa'], value='logp')
)
def interactive_chemical_space(backend, n_molecules, property_x, property_y):
    # Generate data and create plot
    # ... (implementation)
    fig.show()
```

## Best Practices

### Performance Optimization

```python
# For large datasets, use sampling
if len(molecules) > 1000:
    sample_size = 1000
    molecules = np.random.choice(molecules, sample_size, replace=False)

# Use appropriate plot types for data size
if len(data) > 10000:
    # Use density plots instead of scatter
    fig = px.density_heatmap(df, x='prop1', y='prop2')
else:
    fig = px.scatter(df, x='prop1', y='prop2')
```

### Memory Management

```python
# Clear large figures from memory
del fig
import gc
gc.collect()

# Use lazy loading for dashboards
def create_dashboard_on_demand():
    # Only create plots when needed
    return dashboard
```

### Export Options

```python
# Export individual plots
fig.write_html("plot.html")
fig.write_image("plot.png", width=800, height=600)
fig.write_image("plot.pdf")

# Export dashboard with custom styling
html_string = fig.to_html(
    include_plotlyjs='cdn',  # Use CDN for smaller files
    config={'displayModeBar': True}  # Show toolbar
)
```

## Troubleshooting

### Common Issues

**Plots not displaying:**
```python
# Ensure plotly is properly installed
pip install plotly kaleido

# For Jupyter notebooks
import plotly.io as pio
pio.renderers.default = "notebook"
```

**Memory issues with large datasets:**
```python
# Use chunked processing
def process_in_chunks(data, chunk_size=1000):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

# Sample data for visualization
if len(data) > 5000:
    data = data.sample(n=5000)
```

**Slow rendering:**
```python
# Reduce data points for interactive plots
fig.update_traces(marker=dict(size=4))  # Smaller markers
fig.update_layout(hovermode='closest')  # Optimize hover
```

See the [FAQ](FAQ) for more troubleshooting tips.

---

*Next: Explore [Multi-Objective Optimization](Multi-Objective-Optimization) for advanced optimization workflows.*
