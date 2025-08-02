# Examples Gallery

This gallery showcases real-world applications of ChemiAccel across different domains, from basic descriptor calculations to advanced AI-driven molecular design.

## Quick Start Examples

### Hello World - Your First Calculation

```python
from chemi_accel import Calculator

# Create calculator
calc = Calculator()

# Calculate basic descriptors for common molecules
molecules = ["CCO", "c1ccccc1", "CC(=O)O", "CCN"]
results = calc.pandas(molecules)

print(f"Calculated {len(results.columns)} descriptors for {len(molecules)} molecules")
print(results.head())
```

### Load Data from File

```python
import pandas as pd
from chemi_accel import Calculator

# Load molecules from CSV file
df = pd.read_csv("molecules.csv")
smiles_list = df["smiles"].tolist()

# Calculate descriptors
calc = Calculator(backend="cuda")  # Use GPU acceleration
results = calc.pandas(smiles_list)

# Save results
results.to_csv("molecular_descriptors.csv", index=False)
print(f"Processed {len(results)} molecules")
```

## Drug Discovery Examples

### Lipinski Rule of Five Analysis

```python
from chemi_accel import Calculator
import pandas as pd
import matplotlib.pyplot as plt

# Load drug-like molecules
calc = Calculator()
molecules = [
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",  # Celecoxib
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",   # Caffeine
    "CC(=O)OC1=CC=CC=C1C(=O)O",       # Aspirin
    "CN(C)CCOC1=CC=C(C=C1)C2CCCC3=C2C=CC(=C3)O"  # Tamoxifen
]

# Calculate Lipinski descriptors
lipinski_descriptors = [
    "molecular_weight", "logp", "num_hbd", "num_hba", "tpsa"
]

results = calc.pandas(molecules, descriptors=lipinski_descriptors)

# Apply Lipinski filters
def apply_lipinski_rule_of_five(df):
    """Apply Lipinski's Rule of Five."""
    return df[
        (df["molecular_weight"] <= 500) &
        (df["logp"] <= 5) &
        (df["num_hbd"] <= 5) &
        (df["num_hba"] <= 10)
    ]

drug_like = apply_lipinski_rule_of_five(results)
print(f"Drug-like molecules: {len(drug_like)}/{len(results)}")

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Molecular weight vs LogP
axes[0,0].scatter(results["molecular_weight"], results["logp"], alpha=0.7)
axes[0,0].axvline(x=500, color='r', linestyle='--', label='MW limit')
axes[0,0].axhline(y=5, color='r', linestyle='--', label='LogP limit')
axes[0,0].set_xlabel("Molecular Weight")
axes[0,0].set_ylabel("LogP")
axes[0,0].legend()

# Hydrogen bond donors vs acceptors
axes[0,1].scatter(results["num_hbd"], results["num_hba"], alpha=0.7)
axes[0,1].axvline(x=5, color='r', linestyle='--', label='HBD limit')
axes[0,1].axhline(y=10, color='r', linestyle='--', label='HBA limit')
axes[0,1].set_xlabel("H-bond Donors")
axes[0,1].set_ylabel("H-bond Acceptors")
axes[0,1].legend()

plt.tight_layout()
plt.show()
```

### ADMET Property Prediction

```python
from chemi_accel import Calculator
from chemi_accel.ai.admet import ADMETPredictor

# Initialize ADMET predictor
calc = Calculator(backend="cuda")
admet_predictor = ADMETPredictor()

# Drug candidates
drug_candidates = [
    "COc1cc2c(cc1OC)CCN(C2)CC3=CC=C(C=C3)C(=O)O",  # Novel compound 1
    "Cc1ccc(cc1)S(=O)(=O)N2CCC(CC2)N3CCN(CC3)c4ccccn4",  # Novel compound 2
    "CC(C)(C)NCC(O)COc1cccc2c1cccc2"  # Novel compound 3
]

# Calculate molecular descriptors
descriptors = calc.pandas(
    drug_candidates,
    descriptors=[
        "molecular_weight", "logp", "tpsa", "num_hbd", "num_hba",
        "num_rotatable_bonds", "morgan_fingerprint_2048"
    ]
)

# Predict ADMET properties
admet_results = admet_predictor.predict_all(drug_candidates)

# Combine results
combined_results = pd.concat([descriptors, admet_results], axis=1)

print("ADMET Predictions:")
print(combined_results[["molecular_weight", "logp", "bioavailability", 
                       "bbb_permeability", "cyp_inhibition"]].round(3))
```

### High-Throughput Virtual Screening

```python
from chemi_accel import Calculator
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def process_chunk(smiles_chunk, chunk_id):
    """Process a chunk of molecules."""
    calc = Calculator(backend="cuda")
    
    try:
        results = calc.pandas(
            smiles_chunk,
            descriptors=[
                "molecular_weight", "logp", "tpsa", "num_hbd", "num_hba",
                "morgan_fingerprint_2048", "maccs_keys"
            ]
        )
        results["chunk_id"] = chunk_id
        return results
    except Exception as e:
        print(f"Error processing chunk {chunk_id}: {e}")
        return None

# Load large molecular database
large_database = pd.read_csv("pubchem_subset_1M.csv")
smiles_list = large_database["smiles"].tolist()

# Split into chunks for parallel processing
chunk_size = 10000
chunks = [smiles_list[i:i+chunk_size] for i in range(0, len(smiles_list), chunk_size)]

print(f"Processing {len(smiles_list)} molecules in {len(chunks)} chunks")

# Process chunks in parallel
with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_chunk, chunk, i) 
               for i, chunk in enumerate(chunks)]
    
    results = []
    for i, future in enumerate(futures):
        chunk_result = future.result()
        if chunk_result is not None:
            results.append(chunk_result)
            print(f"Completed chunk {i+1}/{len(chunks)}")

# Combine all results
final_results = pd.concat(results, ignore_index=True)

# Apply drug-likeness filters
drug_like_results = final_results[
    (final_results["molecular_weight"] <= 500) &
    (final_results["logp"] <= 5) &
    (final_results["num_hbd"] <= 5) &
    (final_results["num_hba"] <= 10) &
    (final_results["tpsa"] <= 140)
]

print(f"Found {len(drug_like_results)} drug-like compounds from {len(final_results)} total")

# Save results
drug_like_results.to_parquet("drug_like_compounds.parquet")
```

## Polymer Science Examples

### Copolymer Property Prediction

```python
from chemi_accel import Calculator
from chemi_accel.polymer import CopolymerAnalyzer, MonomerDatabase
import numpy as np
import matplotlib.pyplot as plt

# Initialize polymer-specific calculator
calc = Calculator(backend="cuda")
copolymer_analyzer = CopolymerAnalyzer()
monomer_db = MonomerDatabase()

# Define monomers for copolymer study
monomers = {
    "styrene": "C=Cc1ccccc1",
    "acrylonitrile": "C=CC#N", 
    "methyl_methacrylate": "C=C(C)C(=O)OC",
    "vinyl_acetate": "C=COC(=O)C"
}

# Calculate monomer descriptors
monomer_descriptors = {}
for name, smiles in monomers.items():
    descriptors = calc.pandas(
        [smiles],
        descriptors=[
            "molecular_weight", "logp", "tpsa", "dipole_moment",
            "polarizability", "homo_lumo_gap"
        ]
    )
    monomer_descriptors[name] = descriptors.iloc[0]

# Predict copolymer properties for different compositions
compositions = np.linspace(0, 1, 21)  # 0% to 100% of first monomer
copolymer_properties = []

for ratio_A in compositions:
    ratio_B = 1 - ratio_A
    
    # Predict glass transition temperature
    tg_prediction = copolymer_analyzer.predict_glass_transition_temperature(
        monomer_a="styrene",
        monomer_b="acrylonitrile", 
        ratio_a=ratio_A,
        ratio_b=ratio_B
    )
    
    # Predict mechanical properties
    mechanical_props = copolymer_analyzer.predict_mechanical_properties(
        monomer_a="styrene",
        monomer_b="acrylonitrile",
        ratio_a=ratio_A,
        ratio_b=ratio_B
    )
    
    copolymer_properties.append({
        "ratio_styrene": ratio_A,
        "ratio_acrylonitrile": ratio_B,
        "glass_transition_temp": tg_prediction,
        "tensile_strength": mechanical_props["tensile_strength"],
        "young_modulus": mechanical_props["young_modulus"]
    })

# Convert to DataFrame and visualize
copolymer_df = pd.DataFrame(copolymer_properties)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Glass transition temperature
axes[0].plot(copolymer_df["ratio_styrene"], copolymer_df["glass_transition_temp"], 'b-o')
axes[0].set_xlabel("Styrene Fraction")
axes[0].set_ylabel("Glass Transition Temp (°C)")
axes[0].set_title("Tg vs Composition")

# Tensile strength
axes[1].plot(copolymer_df["ratio_styrene"], copolymer_df["tensile_strength"], 'r-s')
axes[1].set_xlabel("Styrene Fraction")
axes[1].set_ylabel("Tensile Strength (MPa)")
axes[1].set_title("Tensile Strength vs Composition")

# Young's modulus
axes[2].plot(copolymer_df["ratio_styrene"], copolymer_df["young_modulus"], 'g-^')
axes[2].set_xlabel("Styrene Fraction")
axes[2].set_ylabel("Young's Modulus (GPa)")
axes[2].set_title("Young's Modulus vs Composition")

plt.tight_layout()
plt.show()

# Find optimal composition for specific properties
optimal_tg = copolymer_df.loc[copolymer_df["glass_transition_temp"].idxmin()]
print(f"Optimal composition for low Tg: {optimal_tg['ratio_styrene']:.2f} styrene")

optimal_strength = copolymer_df.loc[copolymer_df["tensile_strength"].idxmax()]
print(f"Optimal composition for high strength: {optimal_strength['ratio_styrene']:.2f} styrene")
```

### Polymer Design Optimization

```python
from chemi_accel import Calculator
from chemi_accel.ai.multi_objective_optimization import MultiObjectiveOptimizer
from chemi_accel.polymer import PolymerGenerator

# Initialize components
calc = Calculator(backend="cuda")
optimizer = MultiObjectiveOptimizer()
polymer_generator = PolymerGenerator()

# Define optimization objectives
def evaluate_polymer(monomer_smiles):
    """Evaluate polymer properties for optimization."""
    # Calculate monomer descriptors
    descriptors = calc.pandas(
        [monomer_smiles],
        descriptors=["molecular_weight", "logp", "tpsa", "dipole_moment"]
    ).iloc[0]
    
    # Predict polymer properties
    polymer_props = polymer_generator.predict_homopolymer_properties(monomer_smiles)
    
    # Define objectives (to be maximized or minimized)
    objectives = {
        "glass_transition_temp": polymer_props["tg"],  # Target: 100°C
        "tensile_strength": polymer_props["tensile_strength"],  # Maximize
        "transparency": polymer_props["transparency"],  # Maximize
        "cost": polymer_props["estimated_cost"]  # Minimize
    }
    
    return objectives

# Generate initial monomer population
initial_monomers = polymer_generator.generate_monomers(
    num_monomers=100,
    functional_groups=["vinyl", "acrylate", "styrene"],
    molecular_weight_range=(80, 300)
)

# Run multi-objective optimization
optimization_result = optimizer.optimize(
    evaluate_function=evaluate_polymer,
    initial_population=initial_monomers,
    objectives=["glass_transition_temp", "tensile_strength", "transparency", "-cost"],
    population_size=50,
    num_generations=20,
    crossover_prob=0.8,
    mutation_prob=0.1
)

# Analyze results
pareto_front = optimization_result.pareto_front
print(f"Found {len(pareto_front)} Pareto-optimal solutions")

# Select best compromise solution
compromise_solution = optimizer.select_compromise_solution(
    pareto_front,
    weights={"glass_transition_temp": 0.3, "tensile_strength": 0.3, 
             "transparency": 0.2, "cost": 0.2}
)

print(f"Best compromise monomer: {compromise_solution['monomer']}")
print(f"Properties: {compromise_solution['objectives']}")
```

## Materials Science Examples

### Crystal Structure Analysis

```python
from chemi_accel import Calculator
from chemi_accel.materials import MaterialsCalculator, CrystalStructureAnalyzer
import numpy as np

# Initialize materials calculator
materials_calc = MaterialsCalculator(use_gpu=True)
crystal_analyzer = CrystalStructureAnalyzer()

# Load crystal structures from CIF files
crystal_files = [
    "perovskite_BaTiO3.cif",
    "spinel_MgAl2O4.cif", 
    "fluorite_CaF2.cif",
    "rutile_TiO2.cif"
]

crystal_data = []
for cif_file in crystal_files:
    try:
        # Load and analyze crystal structure
        structure = materials_calc.load_cif(cif_file)
        
        # Calculate structural descriptors
        descriptors = materials_calc.calculate_structure_descriptors(structure)
        
        # Calculate electronic properties
        electronic_props = materials_calc.calculate_electronic_properties(
            structure, method="DFT-PBE"
        )
        
        # Calculate mechanical properties
        mechanical_props = materials_calc.calculate_mechanical_properties(structure)
        
        crystal_data.append({
            "name": cif_file.split(".")[0],
            "space_group": descriptors["space_group"],
            "density": descriptors["density"],
            "band_gap": electronic_props["band_gap"],
            "bulk_modulus": mechanical_props["bulk_modulus"],
            "shear_modulus": mechanical_props["shear_modulus"],
            "formation_energy": electronic_props["formation_energy"]
        })
        
        print(f"Processed {cif_file}")
        
    except Exception as e:
        print(f"Error processing {cif_file}: {e}")

# Convert to DataFrame for analysis
materials_df = pd.DataFrame(crystal_data)

# Analyze structure-property relationships
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Density vs Band Gap
axes[0,0].scatter(materials_df["density"], materials_df["band_gap"])
axes[0,0].set_xlabel("Density (g/cm³)")
axes[0,0].set_ylabel("Band Gap (eV)")
axes[0,0].set_title("Density vs Band Gap")

# Bulk Modulus vs Shear Modulus
axes[0,1].scatter(materials_df["bulk_modulus"], materials_df["shear_modulus"])
axes[0,1].set_xlabel("Bulk Modulus (GPa)")
axes[0,1].set_ylabel("Shear Modulus (GPa)")
axes[0,1].set_title("Bulk vs Shear Modulus")

# Formation Energy vs Band Gap
axes[1,0].scatter(materials_df["formation_energy"], materials_df["band_gap"])
axes[1,0].set_xlabel("Formation Energy (eV/atom)")
axes[1,0].set_ylabel("Band Gap (eV)")
axes[1,0].set_title("Formation Energy vs Band Gap")

# Property correlation heatmap
numeric_cols = ["density", "band_gap", "bulk_modulus", "shear_modulus", "formation_energy"]
correlation_matrix = materials_df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, ax=axes[1,1])
axes[1,1].set_title("Property Correlations")

plt.tight_layout()
plt.show()

print("\nMaterials Analysis Results:")
print(materials_df.round(3))
```

### High-Throughput Materials Screening

```python
from chemi_accel.materials import MaterialsCalculator, MaterialsDatabase
from chemi_accel.ai.active_learning import ActiveLearningAgent
import pandas as pd

# Initialize components
materials_calc = MaterialsCalculator(use_gpu=True)
materials_db = MaterialsDatabase()
active_learner = ActiveLearningAgent()

# Load materials database
candidate_materials = materials_db.load_database("materials_project_subset.json")
print(f"Loaded {len(candidate_materials)} candidate materials")

# Define target properties for screening
target_properties = {
    "band_gap": (1.0, 3.0),      # eV, suitable for photovoltaics
    "formation_energy": (-2.0, 0.0),  # eV/atom, thermodynamically stable
    "bulk_modulus": (50, 200),    # GPa, mechanical stability
}

# Initial screening with fast descriptors
print("Performing initial screening...")
initial_descriptors = []
for i, material in enumerate(candidate_materials[:1000]):  # Start with subset
    try:
        descriptors = materials_calc.calculate_fast_descriptors(material)
        descriptors["material_id"] = i
        initial_descriptors.append(descriptors)
        
        if i % 200 == 0:
            print(f"Processed {i+1}/1000 materials")
            
    except Exception as e:
        print(f"Error processing material {i}: {e}")

initial_df = pd.DataFrame(initial_descriptors)

# Apply initial filters
filtered_materials = initial_df[
    (initial_df["estimated_band_gap"] >= target_properties["band_gap"][0]) &
    (initial_df["estimated_band_gap"] <= target_properties["band_gap"][1]) &
    (initial_df["estimated_formation_energy"] >= target_properties["formation_energy"][0]) &
    (initial_df["estimated_formation_energy"] <= target_properties["formation_energy"][1])
]

print(f"Initial screening: {len(filtered_materials)} materials passed filters")

# Active learning for detailed property prediction
selected_for_detailed_calc = active_learner.select_next_batch(
    candidates=filtered_materials,
    batch_size=50,
    strategy="uncertainty_sampling"
)

# Perform detailed calculations on selected materials
detailed_results = []
for material_id in selected_for_detailed_calc:
    material = candidate_materials[material_id]
    
    try:
        # Full DFT calculation
        detailed_props = materials_calc.calculate_detailed_properties(
            material, method="DFT-PBE", include_phonons=True
        )
        
        detailed_results.append({
            "material_id": material_id,
            "formula": material["formula"],
            "band_gap": detailed_props["band_gap"],
            "formation_energy": detailed_props["formation_energy"],
            "bulk_modulus": detailed_props["bulk_modulus"],
            "thermal_conductivity": detailed_props["thermal_conductivity"]
        })
        
    except Exception as e:
        print(f"Detailed calculation failed for material {material_id}: {e}")

detailed_df = pd.DataFrame(detailed_results)

# Final filtering and ranking
final_candidates = detailed_df[
    (detailed_df["band_gap"] >= target_properties["band_gap"][0]) &
    (detailed_df["band_gap"] <= target_properties["band_gap"][1]) &
    (detailed_df["formation_energy"] >= target_properties["formation_energy"][0]) &
    (detailed_df["formation_energy"] <= target_properties["formation_energy"][1]) &
    (detailed_df["bulk_modulus"] >= target_properties["bulk_modulus"][0]) &
    (detailed_df["bulk_modulus"] <= target_properties["bulk_modulus"][1])
]

# Rank by multi-objective score
def calculate_score(row):
    """Calculate multi-objective score for material ranking."""
    band_gap_score = 1 - abs(row["band_gap"] - 2.0) / 2.0  # Target: 2.0 eV
    stability_score = 1 + row["formation_energy"] / 2.0    # More negative is better
    mechanical_score = row["bulk_modulus"] / 200.0         # Higher is better
    
    return (band_gap_score + stability_score + mechanical_score) / 3

final_candidates["score"] = final_candidates.apply(calculate_score, axis=1)
final_candidates = final_candidates.sort_values("score", ascending=False)

print(f"\nTop 10 Materials for Photovoltaic Applications:")
print(final_candidates.head(10)[["formula", "band_gap", "formation_energy", 
                                "bulk_modulus", "score"]].round(3))
```

## AI and Machine Learning Examples

### Molecular Generation with GANs

```python
from chemi_accel import Calculator
from chemi_accel.ai.generative_chemistry import MolecularGAN, MolecularVAE
from chemi_accel.ai.molecular_optimization import MolecularOptimizer
import pandas as pd

# Initialize AI components
calc = Calculator(backend="cuda")
mol_gan = MolecularGAN()
molecular_optimizer = MolecularOptimizer()

# Load training data
training_molecules = pd.read_csv("chembl_molecules.csv")["smiles"].tolist()

# Calculate descriptors for training set
print("Calculating descriptors for training set...")
training_descriptors = calc.pandas(
    training_molecules[:10000],  # Use subset for faster training
    descriptors=[
        "molecular_weight", "logp", "tpsa", "num_hbd", "num_hba",
        "morgan_fingerprint_2048"
    ]
)

# Train molecular GAN
print("Training molecular GAN...")
mol_gan.train(
    training_molecules=training_molecules[:10000],
    training_descriptors=training_descriptors,
    epochs=100,
    batch_size=64,
    learning_rate=0.0002
)

# Generate new molecules
print("Generating new molecules...")
generated_molecules = mol_gan.generate(
    num_molecules=1000,
    temperature=0.8,
    filter_valid=True
)

print(f"Generated {len(generated_molecules)} valid molecules")

# Calculate properties of generated molecules
generated_descriptors = calc.pandas(
    generated_molecules,
    descriptors=["molecular_weight", "logp", "tpsa", "num_hbd", "num_hba"]
)

# Analyze generation quality
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

properties = ["molecular_weight", "logp", "tpsa", "num_hbd", "num_hba"]

for i, prop in enumerate(properties):
    row, col = divmod(i, 3)
    
    # Plot distributions
    axes[row, col].hist(training_descriptors[prop], alpha=0.5, label="Training", bins=50)
    axes[row, col].hist(generated_descriptors[prop], alpha=0.5, label="Generated", bins=50)
    axes[row, col].set_xlabel(prop)
    axes[row, col].set_ylabel("Frequency")
    axes[row, col].legend()
    axes[row, col].set_title(f"{prop} Distribution")

# Novelty analysis
axes[1, 2].remove()  # Remove unused subplot

plt.tight_layout()
plt.show()

# Optimize generated molecules for specific properties
optimization_targets = {
    "molecular_weight": (300, 500),  # Drug-like range
    "logp": (0, 3),                  # Good permeability
    "tpsa": (40, 140),              # Good absorption
    "num_hbd": (0, 5),              # Lipinski rule
    "num_hba": (0, 10)              # Lipinski rule
}

optimized_molecules = molecular_optimizer.optimize_molecules(
    initial_molecules=generated_molecules,
    targets=optimization_targets,
    optimization_steps=50,
    learning_rate=0.01
)

print(f"Optimized {len(optimized_molecules)} molecules for drug-likeness")

# Evaluate optimization results
optimized_descriptors = calc.pandas(
    optimized_molecules,
    descriptors=["molecular_weight", "logp", "tpsa", "num_hbd", "num_hba"]
)

# Calculate drug-likeness scores
def calculate_drug_likeness(df):
    """Calculate drug-likeness score based on Lipinski rules."""
    score = 0
    score += (df["molecular_weight"] <= 500).astype(int)
    score += (df["logp"] <= 5).astype(int)
    score += (df["num_hbd"] <= 5).astype(int)
    score += (df["num_hba"] <= 10).astype(int)
    score += (df["tpsa"] <= 140).astype(int)
    return score / 5  # Normalize to 0-1

generated_scores = calculate_drug_likeness(generated_descriptors)
optimized_scores = calculate_drug_likeness(optimized_descriptors)

print(f"Average drug-likeness score:")
print(f"Generated: {generated_scores.mean():.3f}")
print(f"Optimized: {optimized_scores.mean():.3f}")
print(f"Improvement: {(optimized_scores.mean() - generated_scores.mean()):.3f}")
```

### Property Prediction with Neural Networks

```python
from chemi_accel import Calculator
from chemi_accel.ai.property_prediction import PropertyPredictor, NeuralNetworkModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Load experimental data
data = pd.read_csv("experimental_data.csv")
molecules = data["smiles"].tolist()
target_property = data["solubility"].values  # log S

# Calculate comprehensive descriptors
calc = Calculator(backend="cuda")
descriptors = calc.pandas(
    molecules,
    descriptors=[
        "molecular_weight", "logp", "tpsa", "num_hbd", "num_hba",
        "num_rotatable_bonds", "morgan_fingerprint_2048", "maccs_keys",
        "topological_descriptors", "electronic_descriptors"
    ]
)

print(f"Calculated {descriptors.shape[1]} descriptors for {len(molecules)} molecules")

# Prepare data for machine learning
X = descriptors.select_dtypes(include=[np.number])  # Only numeric columns
y = target_property

# Handle missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Feature selection
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(score_func=f_regression, k=500)  # Select top 500 features
X_selected = selector.fit_transform(X_imputed, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# Train neural network model
property_predictor = PropertyPredictor()
nn_model = NeuralNetworkModel(
    input_dim=X_selected.shape[1],
    hidden_layers=[1024, 512, 256, 128],
    dropout_rate=0.3,
    learning_rate=0.001
)

print("Training neural network...")
history = property_predictor.train(
    model=nn_model,
    X_train=X_train,
    y_train=y_train,
    X_val=X_test,
    y_val=y_test,
    epochs=200,
    batch_size=64,
    early_stopping=True,
    patience=20
)

# Make predictions
y_pred = property_predictor.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
print(f"RMSE: {np.sqrt(mse):.4f}")

# Plot results
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Training history
axes[0].plot(history['loss'], label='Training Loss')
axes[0].plot(history['val_loss'], label='Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].set_title('Training History')

# Predicted vs Actual
axes[1].scatter(y_test, y_pred, alpha=0.6)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Solubility')
axes[1].set_ylabel('Predicted Solubility')
axes[1].set_title(f'Predicted vs Actual (R² = {r2:.3f})')

# Residuals
residuals = y_test - y_pred
axes[2].scatter(y_pred, residuals, alpha=0.6)
axes[2].axhline(y=0, color='r', linestyle='--')
axes[2].set_xlabel('Predicted Solubility')
axes[2].set_ylabel('Residuals')
axes[2].set_title('Residual Plot')

plt.tight_layout()
plt.show()

# Feature importance analysis
feature_importance = property_predictor.get_feature_importance(
    X_test, y_test, method="permutation"
)

# Get selected feature names
selected_features = X.columns[selector.get_support()]
importance_df = pd.DataFrame({
    'feature': selected_features,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance_df.head(10))

# Apply model to new molecules
new_molecules = [
    "CCc1ccc(cc1)S(=O)(=O)N2CCC(CC2)C(=O)O",  # Novel compound 1
    "COc1ccc(cc1)C2=CC(=O)c3c(O2)cc(cc3)O",   # Novel compound 2
    "Nc1ccc(cc1)S(=O)(=O)Nc2cccc(c2)C(=O)O"   # Novel compound 3
]

# Calculate descriptors for new molecules
new_descriptors = calc.pandas(
    new_molecules,
    descriptors=list(X.columns)  # Use same descriptors as training
)

# Preprocess new data (same steps as training)
new_X = new_descriptors.select_dtypes(include=[np.number])
new_X_imputed = imputer.transform(new_X)
new_X_selected = selector.transform(new_X_imputed)

# Make predictions
new_predictions = property_predictor.predict(new_X_selected)

print(f"\nPredictions for new molecules:")
for i, (mol, pred) in enumerate(zip(new_molecules, new_predictions)):
    print(f"Molecule {i+1}: Predicted solubility = {pred:.3f} log S")
```

## Cloud and Distributed Computing Examples

### AWS SageMaker Integration

```python
from chemi_accel import Calculator
from chemi_accel.cloud.aws import SageMakerJob, S3DataManager
import boto3
import pandas as pd

# Setup AWS components
s3_manager = S3DataManager(bucket_name="my-chemiaccel-bucket")
sagemaker_job = SageMakerJob()

# Upload data to S3
local_data = pd.read_csv("large_molecule_dataset.csv")
s3_data_uri = s3_manager.upload_dataframe(
    dataframe=local_data,
    key="datasets/molecules_1M.csv"
)

print(f"Uploaded data to S3: {s3_data_uri}")

# Define SageMaker job configuration
job_config = {
    "job_name": "chemiaccel-descriptor-calculation",
    "instance_type": "ml.g4dn.2xlarge",  # GPU instance
    "instance_count": 4,                  # Multi-instance
    "input_data_uri": s3_data_uri,
    "output_data_uri": "s3://my-chemiaccel-bucket/results/",
    "max_runtime_seconds": 3600,          # 1 hour limit
    "hyperparameters": {
        "backend": "cuda",
        "batch_size": "5000",
        "descriptors": "molecular_weight,logp,tpsa,morgan_fingerprint_2048"
    }
}

# Submit job to SageMaker
job_name = sagemaker_job.submit_job(config=job_config)
print(f"Submitted SageMaker job: {job_name}")

# Monitor job progress
import time
while True:
    status = sagemaker_job.get_job_status(job_name)
    print(f"Job status: {status}")
    
    if status in ["Completed", "Failed", "Stopped"]:
        break
    
    time.sleep(60)  # Check every minute

# Download results if successful
if status == "Completed":
    results_uri = f"s3://my-chemiaccel-bucket/results/{job_name}/output/results.csv"
    results_df = s3_manager.download_dataframe(results_uri)
    
    print(f"Downloaded results: {results_df.shape}")
    print(results_df.head())
    
    # Save locally
    results_df.to_csv("sagemaker_results.csv", index=False)
else:
    print(f"Job failed with status: {status}")
    logs = sagemaker_job.get_job_logs(job_name)
    print(f"Error logs: {logs}")
```

### Multi-Node Distributed Processing

```python
from chemi_accel import Calculator
from chemi_accel.distributed import DistributedCalculator, ClusterManager
import pandas as pd

# Setup distributed cluster
cluster_manager = ClusterManager()

# Define cluster configuration
cluster_config = {
    "master_node": "192.168.1.100:8000",
    "worker_nodes": [
        "192.168.1.101:8000",
        "192.168.1.102:8000", 
        "192.168.1.103:8000",
        "192.168.1.104:8000"
    ],
    "backend": "cuda",  # Each node has GPU
    "communication_backend": "nccl"
}

# Initialize distributed calculator
dist_calc = DistributedCalculator(cluster_config)

# Load large dataset
large_dataset = pd.read_csv("molecules_10M.csv")
molecules = large_dataset["smiles"].tolist()

print(f"Processing {len(molecules)} molecules across {len(cluster_config['worker_nodes'])} nodes")

# Distribute calculation across cluster
results = dist_calc.pandas(
    molecules,
    descriptors=[
        "molecular_weight", "logp", "tpsa", "num_hbd", "num_hba",
        "morgan_fingerprint_2048", "topological_descriptors"
    ],
    batch_size=10000,  # Large batches for distributed processing
    strategy="load_balanced"  # Balance load across nodes
)

print(f"Completed distributed calculation: {results.shape}")

# Gather performance statistics
perf_stats = dist_calc.get_performance_stats()
print(f"Total processing time: {perf_stats['total_time']:.2f} seconds")
print(f"Average GPU utilization: {perf_stats['avg_gpu_utilization']:.1f}%")
print(f"Network communication overhead: {perf_stats['communication_overhead']:.1f}%")

# Save results
results.to_parquet("distributed_results.parquet")

# Cleanup cluster
dist_calc.shutdown()
```

## Performance Benchmarking Examples

### Backend Performance Comparison

```python
from chemi_accel import Calculator
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np

# Generate test datasets of different sizes
dataset_sizes = [100, 500, 1000, 5000, 10000, 50000]
test_molecules = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCC"] * 10000

benchmarks = []

for size in dataset_sizes:
    molecules = test_molecules[:size]
    
    # Test each available backend
    for backend in ["cpu", "cuda", "jax"]:
        try:
            calc = Calculator(backend=backend)
            
            # Warm-up run
            calc.pandas(molecules[:10], descriptors=["molecular_weight"])
            
            # Benchmark run
            start_time = time.time()
            results = calc.pandas(
                molecules,
                descriptors=["molecular_weight", "logp", "tpsa", "morgan_fingerprint_2048"]
            )
            end_time = time.time()
            
            processing_time = end_time - start_time
            throughput = len(molecules) / processing_time
            
            benchmarks.append({
                "backend": backend,
                "dataset_size": size,
                "processing_time": processing_time,
                "throughput": throughput,
                "success": True
            })
            
            print(f"{backend.upper()} - {size} molecules: {processing_time:.2f}s ({throughput:.1f} mol/s)")
            
        except Exception as e:
            benchmarks.append({
                "backend": backend,
                "dataset_size": size,
                "processing_time": np.nan,
                "throughput": np.nan,
                "success": False,
                "error": str(e)
            })
            print(f"{backend.upper()} - {size} molecules: FAILED ({e})")

# Convert to DataFrame and analyze
benchmark_df = pd.DataFrame(benchmarks)
successful_benchmarks = benchmark_df[benchmark_df["success"] == True]

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Processing time comparison
for backend in successful_benchmarks["backend"].unique():
    backend_data = successful_benchmarks[successful_benchmarks["backend"] == backend]
    axes[0].plot(backend_data["dataset_size"], backend_data["processing_time"], 
                 marker='o', label=backend.upper())

axes[0].set_xlabel("Dataset Size (molecules)")
axes[0].set_ylabel("Processing Time (seconds)")
axes[0].set_title("Processing Time vs Dataset Size")
axes[0].legend()
axes[0].set_xscale("log")
axes[0].set_yscale("log")

# Throughput comparison
for backend in successful_benchmarks["backend"].unique():
    backend_data = successful_benchmarks[successful_benchmarks["backend"] == backend]
    axes[1].plot(backend_data["dataset_size"], backend_data["throughput"], 
                 marker='s', label=backend.upper())

axes[1].set_xlabel("Dataset Size (molecules)")
axes[1].set_ylabel("Throughput (molecules/second)")
axes[1].set_title("Throughput vs Dataset Size")
axes[1].legend()
axes[1].set_xscale("log")

plt.tight_layout()
plt.show()

# Summary statistics
print("\nBenchmark Summary:")
for backend in successful_benchmarks["backend"].unique():
    backend_data = successful_benchmarks[successful_benchmarks["backend"] == backend]
    avg_throughput = backend_data["throughput"].mean()
    max_throughput = backend_data["throughput"].max()
    print(f"{backend.upper()}: Average {avg_throughput:.1f} mol/s, Peak {max_throughput:.1f} mol/s")
```

### Memory Usage Analysis

```python
import psutil
import gc
from chemi_accel import Calculator
import matplotlib.pyplot as plt
import numpy as np

def monitor_memory_usage():
    """Monitor system and GPU memory usage."""
    # System memory
    system_memory = psutil.virtual_memory()
    
    # GPU memory (if available)
    gpu_memory = None
    try:
        import cupy as cp
        if cp.cuda.is_available():
            mempool = cp.get_default_memory_pool()
            gpu_memory = {
                "used": mempool.used_bytes(),
                "total": cp.cuda.runtime.memGetInfo()[1]
            }
    except ImportError:
        pass
    
    return {
        "system_used": system_memory.used,
        "system_total": system_memory.total,
        "system_percent": system_memory.percent,
        "gpu_memory": gpu_memory
    }

# Test memory usage with different dataset sizes
dataset_sizes = [1000, 5000, 10000, 25000, 50000]
test_molecules = ["CCO", "c1ccccc1", "CC(=O)O"] * 20000

memory_usage = []

for size in dataset_sizes:
    molecules = test_molecules[:size]
    
    # Measure baseline memory
    gc.collect()
    baseline_memory = monitor_memory_usage()
    
    # Calculate descriptors
    calc = Calculator(backend="cuda")
    results = calc.pandas(
        molecules,
        descriptors=["molecular_weight", "logp", "tpsa", "morgan_fingerprint_2048"]
    )
    
    # Measure peak memory
    peak_memory = monitor_memory_usage()
    
    memory_usage.append({
        "dataset_size": size,
        "baseline_system_gb": baseline_memory["system_used"] / (1024**3),
        "peak_system_gb": peak_memory["system_used"] / (1024**3),
        "system_increase_gb": (peak_memory["system_used"] - baseline_memory["system_used"]) / (1024**3),
        "baseline_gpu_gb": baseline_memory["gpu_memory"]["used"] / (1024**3) if baseline_memory["gpu_memory"] else 0,
        "peak_gpu_gb": peak_memory["gpu_memory"]["used"] / (1024**3) if peak_memory["gpu_memory"] else 0,
    })
    
    print(f"Size {size}: System +{memory_usage[-1]['system_increase_gb']:.2f} GB, "
          f"GPU {memory_usage[-1]['peak_gpu_gb']:.2f} GB")
    
    # Cleanup
    del results, calc
    gc.collect()

# Plot memory usage
memory_df = pd.DataFrame(memory_usage)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# System memory usage
axes[0].plot(memory_df["dataset_size"], memory_df["system_increase_gb"], 'b-o', label="System RAM")
axes[0].set_xlabel("Dataset Size (molecules)")
axes[0].set_ylabel("Memory Increase (GB)")
axes[0].set_title("System Memory Usage")
axes[0].grid(True)

# GPU memory usage
if memory_df["peak_gpu_gb"].sum() > 0:
    axes[1].plot(memory_df["dataset_size"], memory_df["peak_gpu_gb"], 'r-s', label="GPU Memory")
    axes[1].set_xlabel("Dataset Size (molecules)")
    axes[1].set_ylabel("GPU Memory Usage (GB)")
    axes[1].set_title("GPU Memory Usage")
    axes[1].grid(True)

plt.tight_layout()
plt.show()

# Memory efficiency analysis
memory_df["memory_per_molecule_mb"] = (memory_df["system_increase_gb"] * 1024) / memory_df["dataset_size"]
print("\nMemory Efficiency:")
print(f"Average memory per molecule: {memory_df['memory_per_molecule_mb'].mean():.2f} MB")
print(f"Memory scaling factor: {memory_df['memory_per_molecule_mb'].std():.2f} MB std")
```

---

*These examples demonstrate ChemiAccel's capabilities across different domains and scales. For more specific use cases, see the domain-specific guides in the wiki.*
