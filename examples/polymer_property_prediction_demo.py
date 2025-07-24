"""
Polymer Property Prediction Demo

This example demonstrates the use of ChemiAccel's Predictive Polymer Property Engine
for predicting glass transition temperature, Young's modulus, thermal stability,
and density from monomer structures.
"""

import pandas as pd
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Import ChemiAccel components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.chemi_accel.core.calculator import Calculator
from src.chemi_accel.polymer.predictive import (
    PolymerPropertyEngine,
    PolymerModelTrainer,
    predict_polymer_properties,
    predict_glass_transition_temp,
    predict_youngs_modulus
)


def main():
    """Main demonstration function."""
    print("ChemiAccel Polymer Property Prediction Demo")
    print("=" * 50)
    
    # Initialize calculator
    calculator = Calculator()
    
    # Sample polymer monomers
    polymer_monomers = pd.DataFrame({
        'name': [
            'Ethylene',
            'Propylene', 
            'Styrene',
            'Methyl Methacrylate',
            'Vinyl Chloride',
            'Acrylonitrile',
            'Vinyl Acetate',
            'Butadiene',
            'Isoprene',
            'Acrylic Acid'
        ],
        'smiles': [
            'C=C',
            'C=CC',
            'C=Cc1ccccc1',
            'C=C(C)C(=O)OC',
            'C=CCl',
            'C=CC#N',
            'C=CC(=O)OC',
            'C=CC=C',
            'C=C(C)C=C',
            'C=CC(=O)O'
        ]
    })
    
    print(f"Analyzing {len(polymer_monomers)} polymer monomers:")
    for i, row in polymer_monomers.iterrows():
        print(f"  {row['name']}: {row['smiles']}")
    print()
    
    # Demo 1: Basic property prediction
    print("Demo 1: Basic Property Prediction")
    print("-" * 30)
    
    try:
        # Use convenience functions
        print("Predicting glass transition temperatures...")
        tg_results = predict_glass_transition_temp(polymer_monomers['smiles'], calculator)
        
        print("Predicting Young's modulus...")
        modulus_results = predict_youngs_modulus(polymer_monomers['smiles'], calculator)
        
        # Combine results
        results = polymer_monomers.copy()
        results['predicted_tg'] = tg_results['glass_transition_temp']
        results['predicted_modulus'] = modulus_results['youngs_modulus']
        
        print("\nPrediction Results:")
        print(results[['name', 'predicted_tg', 'predicted_modulus']].to_string(index=False))
        
    except Exception as e:
        print(f"Basic prediction failed (expected without trained models): {e}")
        print("Proceeding with model training demo...")
    
    print()
    
    # Demo 2: Model training and fine-tuning
    print("Demo 2: Model Training and Fine-tuning")
    print("-" * 40)
    
    # Create a temporary model directory
    model_dir = Path("temp_polymer_models")
    model_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize model trainer
        trainer = PolymerModelTrainer(calculator, str(model_dir))
        
        print("Generating synthetic training data...")
        training_data = trainer.generate_synthetic_training_data(n_samples=500)
        print(f"Generated {len(training_data)} synthetic samples")
        
        print("\nTraining polymer property models...")
        training_results = trainer.train_all_models(training_data)
        
        print("\nTraining Results:")
        for prop, metrics in training_results.items():
            if 'error' in metrics:
                print(f"  {prop}: ERROR - {metrics['error']}")
            else:
                print(f"  {prop}: MAE={metrics.get('mae', 'N/A'):.3f}, R²={metrics.get('r2', 'N/A'):.3f}")
        
        # Now use the trained models
        print("\nUsing trained models for prediction...")
        engine = PolymerPropertyEngine(calculator, str(model_dir))
        
        predictions = engine.predict_properties(
            polymer_monomers['smiles'], 
            include_uncertainty=True
        )
        
        # Display results
        results_df = polymer_monomers.copy()
        for col in predictions.columns:
            if col != 'smiles':
                results_df[col] = predictions[col]
        
        print("\nDetailed Prediction Results:")
        display_cols = ['name', 'glass_transition_temp', 'youngs_modulus', 'thermal_stability', 'density']
        available_cols = [col for col in display_cols if col in results_df.columns]
        print(results_df[available_cols].to_string(index=False))
        
    except Exception as e:
        print(f"Model training demo failed: {e}")
        print("This is expected if scikit-learn is not available")
    
    print()
    
    # Demo 3: Fine-tuning with experimental data
    print("Demo 3: Fine-tuning with Experimental Data")
    print("-" * 45)
    
    # Simulated experimental data
    experimental_data = pd.DataFrame({
        'smiles': ['C=C', 'C=CC', 'C=Cc1ccccc1', 'C=C(C)C(=O)OC'],
        'glass_transition_temp': [-120, -20, 100, 105],
        'youngs_modulus': [0.2, 1.5, 3.2, 2.8],
        'polymer_name': ['PE', 'PP', 'PS', 'PMMA']
    })
    
    try:
        # Fine-tune the glass transition model
        print("Fine-tuning glass transition temperature model...")
        metrics = trainer.fine_tune_model(
            experimental_data, 
            'glass_transition_temp',
            validation_split=0.25
        )
        
        print(f"Fine-tuning results:")
        print(f"  MAE: {metrics.mae:.2f}°C")
        print(f"  R²: {metrics.r2:.3f}")
        print(f"  Training samples: {metrics.training_samples}")
        print(f"  Validation samples: {metrics.validation_samples}")
        
        # Test the fine-tuned model
        print("\nTesting fine-tuned model...")
        fine_tuned_predictions = engine.predict_glass_transition_temp(experimental_data['smiles'])
        
        comparison = experimental_data.copy()
        comparison['predicted_tg'] = fine_tuned_predictions['glass_transition_temp']
        comparison['error'] = comparison['predicted_tg'] - comparison['glass_transition_temp']
        
        print("\nExperimental vs Predicted:")
        print(comparison[['polymer_name', 'glass_transition_temp', 'predicted_tg', 'error']].to_string(index=False))
        
    except Exception as e:
        print(f"Fine-tuning demo failed: {e}")
    
    print()
    
    # Demo 4: Batch processing
    print("Demo 4: Batch Processing")
    print("-" * 25)
    
    try:
        # Generate a larger dataset
        large_dataset = pd.Series([
            'C=C', 'C=CC', 'C=CCC', 'C=CCCC', 'C=CCCCC',
            'C=Cc1ccccc1', 'C=Cc1ccc(C)cc1', 'C=Cc1ccc(F)cc1',
            'C=CC(=O)O', 'C=CC(=O)OC', 'C=CC(=O)OCC'
        ] * 10)  # 110 molecules total
        
        print(f"Processing {len(large_dataset)} molecules in batches...")
        
        batch_results = engine.batch_predict(large_dataset, batch_size=20)
        
        print(f"Batch processing completed!")
        print(f"Results shape: {batch_results.shape}")
        
        # Show summary statistics
        if 'glass_transition_temp' in batch_results.columns:
            tg_stats = batch_results['glass_transition_temp'].describe()
            print(f"\nGlass Transition Temperature Statistics:")
            print(f"  Mean: {tg_stats['mean']:.1f}°C")
            print(f"  Std: {tg_stats['std']:.1f}°C")
            print(f"  Range: {tg_stats['min']:.1f} to {tg_stats['max']:.1f}°C")
        
    except Exception as e:
        print(f"Batch processing demo failed: {e}")
    
    print()
    
    # Demo 5: Model information and diagnostics
    print("Demo 5: Model Information and Diagnostics")
    print("-" * 45)
    
    try:
        model_info = engine.get_model_info()
        
        print("Available Models:")
        for prop, info in model_info.items():
            print(f"\n{prop}:")
            print(f"  Model Type: {info.get('model_type', 'Unknown')}")
            print(f"  Has Scaler: {info.get('has_scaler', False)}")
            print(f"  Has Metrics: {info.get('has_metrics', False)}")
            if 'mae' in info:
                print(f"  MAE: {info['mae']:.3f}")
            if 'r2' in info:
                print(f"  R²: {info['r2']:.3f}")
            if 'training_samples' in info:
                print(f"  Training Samples: {info['training_samples']}")
        
    except Exception as e:
        print(f"Model info demo failed: {e}")
    
    print()
    
    # Cleanup
    print("Cleaning up temporary files...")
    try:
        import shutil
        if model_dir.exists():
            shutil.rmtree(model_dir)
        print("Cleanup completed.")
    except Exception as e:
        print(f"Cleanup failed: {e}")
    
    print("\nDemo completed!")
    print("=" * 50)


def create_visualization_demo():
    """Create visualizations of polymer property predictions."""
    print("\nVisualization Demo")
    print("-" * 20)
    
    # Sample data for visualization
    polymers = ['PE', 'PP', 'PS', 'PMMA', 'PVC', 'PET']
    tg_values = [-120, -20, 100, 105, 81, 69]
    modulus_values = [0.2, 1.5, 3.2, 2.8, 2.9, 2.8]
    
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for visualization demo")
        return
    
    try:
        # Create a simple scatter plot
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.bar(polymers, tg_values)
        plt.title('Glass Transition Temperature')
        plt.ylabel('Temperature (°C)')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        plt.bar(polymers, modulus_values)
        plt.title("Young's Modulus")
        plt.ylabel('Modulus (GPa)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('polymer_properties_demo.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Visualization saved as 'polymer_properties_demo.png'")
        
    except ImportError:
        print("Matplotlib not available for visualization demo")
    except Exception as e:
        print(f"Visualization demo failed: {e}")


if __name__ == "__main__":
    main()
    create_visualization_demo()