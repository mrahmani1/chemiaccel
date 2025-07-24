"""
Integration tests for the Predictive Polymer Property Engine.

This module tests the end-to-end functionality of the polymer property
prediction system, including model training and prediction workflows.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from src.chemi_accel.core.calculator import Calculator
from src.chemi_accel.polymer.predictive import (
    PolymerPropertyEngine,
    PolymerModelTrainer,
    predict_polymer_properties,
    predict_glass_transition_temp
)


class TestPolymerPropertyIntegration:
    """Integration tests for polymer property prediction."""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create a temporary directory for model storage."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_polymers(self):
        """Sample polymer data for testing."""
        return pd.DataFrame({
            'name': ['Ethylene', 'Propylene', 'Styrene'],
            'smiles': ['C=C', 'C=CC', 'C=Cc1ccccc1']
        })
    
    def test_end_to_end_workflow(self, temp_model_dir, sample_polymers):
        """Test complete workflow from training to prediction."""
        # Mock calculator
        calculator = Mock(spec=Calculator)
        
        # Create trainer
        trainer = PolymerModelTrainer(calculator, temp_model_dir)
        
        # Generate synthetic training data
        training_data = trainer.generate_synthetic_training_data(n_samples=50)
        
        assert len(training_data) == 50
        assert 'smiles' in training_data.columns
        assert 'glass_transition_temp' in training_data.columns
        
        # Mock descriptor calculation for training
        with patch.object(trainer.descriptor_calc, 'calculate_polymer_descriptors') as mock_calc:
            # Generate mock descriptors
            mock_descriptors = pd.DataFrame({
                'molecular_weight': np.random.rand(50) * 100,
                'logp': np.random.rand(50) * 5,
                'tpsa': np.random.rand(50) * 100,
                'aromatic_content': np.random.rand(50),
                'backbone_flexibility': np.random.rand(50),
            }, index=training_data.index)
            mock_calc.return_value = mock_descriptors
            
            # Train models (this will use default models if sklearn is available)
            try:
                training_results = trainer.train_all_models(training_data)
                
                # Check that some models were trained
                assert len(training_results) > 0
                
                # Create property engine with trained models
                engine = PolymerPropertyEngine(calculator, temp_model_dir)
                
                # Mock descriptor calculation for prediction
                with patch.object(engine.descriptor_calc, 'calculate_polymer_descriptors') as mock_pred_calc:
                    pred_descriptors = pd.DataFrame({
                        'molecular_weight': [28.0, 42.0, 104.0],
                        'logp': [1.0, 1.5, 2.5],
                        'tpsa': [0.0, 0.0, 0.0],
                        'aromatic_content': [0.0, 0.0, 0.6],
                        'backbone_flexibility': [0.8, 0.7, 0.3],
                    }, index=sample_polymers.index)
                    mock_pred_calc.return_value = pred_descriptors
                    
                    # Make predictions
                    predictions = engine.predict_properties(sample_polymers['smiles'])
                    
                    # Verify predictions
                    assert len(predictions) == len(sample_polymers)
                    assert 'smiles' in predictions.columns
                    
                    # Check that at least some properties were predicted
                    property_columns = [col for col in predictions.columns if col != 'smiles']
                    assert len(property_columns) > 0
                    
                    print(f"Successfully predicted {len(property_columns)} properties for {len(predictions)} polymers")
                    
            except ImportError:
                # sklearn not available, skip training test
                pytest.skip("scikit-learn not available for model training")
    
    def test_convenience_functions(self, sample_polymers):
        """Test convenience functions work correctly."""
        calculator = Mock(spec=Calculator)
        
        # Mock the entire prediction pipeline
        with patch('src.chemi_accel.polymer.predictive.property_engine.PolymerPropertyEngine') as mock_engine_class:
            mock_engine = Mock()
            mock_engine.predict_properties.return_value = pd.DataFrame({
                'smiles': sample_polymers['smiles'],
                'glass_transition_temp': [100.0, 120.0, 150.0]
            })
            mock_engine.predict_glass_transition_temp.return_value = pd.DataFrame({
                'smiles': sample_polymers['smiles'],
                'glass_transition_temp': [100.0, 120.0, 150.0]
            })
            mock_engine_class.return_value = mock_engine
            
            # Test predict_polymer_properties
            results1 = predict_polymer_properties(sample_polymers['smiles'], calculator)
            assert len(results1) == len(sample_polymers)
            
            # Test predict_glass_transition_temp
            results2 = predict_glass_transition_temp(sample_polymers['smiles'], calculator)
            assert len(results2) == len(sample_polymers)
            assert 'glass_transition_temp' in results2.columns
    
    def test_descriptor_calculation_integration(self, sample_polymers):
        """Test descriptor calculation integration."""
        calculator = Mock(spec=Calculator)
        
        from src.chemi_accel.polymer.predictive.descriptors import PolymerDescriptorCalculator
        
        desc_calc = PolymerDescriptorCalculator(calculator)
        
        # Mock validation to avoid issues
        with patch('src.chemi_accel.utils.validation.validate_smiles_series') as mock_validate:
            mock_validate.return_value = sample_polymers['smiles']
            
            descriptors = desc_calc.calculate_polymer_descriptors(sample_polymers['smiles'])
            
            # Check descriptor structure
            assert isinstance(descriptors, pd.DataFrame)
            assert len(descriptors) == len(sample_polymers)
            
            # Check for expected descriptor categories
            expected_descriptors = [
                'molecular_weight', 'vinyl_groups', 'backbone_flexibility',
                'tg_predictor_1', 'aromatic_content', 'modulus_predictor_1'
            ]
            
            for desc in expected_descriptors:
                assert desc in descriptors.columns, f"Missing descriptor: {desc}"
            
            # Check that values are reasonable
            assert descriptors.isna().sum().sum() == 0, "Found NaN values"
            assert np.isfinite(descriptors.values).all(), "Found infinite values"
    
    def test_model_info_and_diagnostics(self, temp_model_dir):
        """Test model information and diagnostic features."""
        calculator = Mock(spec=Calculator)
        engine = PolymerPropertyEngine(calculator, temp_model_dir)
        
        # Test model info (should work even without trained models)
        model_info = engine.get_model_info()
        assert isinstance(model_info, dict)
        
        # If default models are initialized, check their info
        if len(engine.models) > 0:
            for prop, info in model_info.items():
                assert 'model_type' in info
                assert 'has_scaler' in info
                assert 'has_metrics' in info
    
    def test_error_handling(self, temp_model_dir):
        """Test error handling in various scenarios."""
        calculator = Mock(spec=Calculator)
        engine = PolymerPropertyEngine(calculator, temp_model_dir)
        
        # Test with empty input
        with pytest.raises(ValueError):
            engine.predict_properties(pd.Series([]))
        
        # Test with invalid SMILES (should be handled gracefully)
        invalid_smiles = pd.Series(['INVALID_SMILES', ''])
        
        with patch.object(engine.descriptor_calc, 'calculate_polymer_descriptors') as mock_calc:
            mock_calc.return_value = pd.DataFrame()
            
            # Should not raise exception, but return empty or default results
            try:
                results = engine.predict_properties(invalid_smiles)
                # Should handle gracefully
                assert isinstance(results, pd.DataFrame)
            except Exception as e:
                # If it does raise an exception, it should be informative
                assert "validation" in str(e).lower() or "invalid" in str(e).lower()
    
    def test_batch_processing(self, temp_model_dir):
        """Test batch processing functionality."""
        calculator = Mock(spec=Calculator)
        engine = PolymerPropertyEngine(calculator, temp_model_dir)
        
        # Create a larger dataset
        large_dataset = pd.Series(['C=C', 'C=CC', 'C=CCC'] * 10)  # 30 molecules
        
        with patch.object(engine, 'predict_properties') as mock_predict:
            # Mock batch predictions
            mock_predict.side_effect = [
                pd.DataFrame({'smiles': ['C=C'] * 10, 'prop': range(10)}),
                pd.DataFrame({'smiles': ['C=CC'] * 10, 'prop': range(10, 20)}),
                pd.DataFrame({'smiles': ['C=CCC'] * 10, 'prop': range(20, 30)})
            ]
            
            results = engine.batch_predict(large_dataset, batch_size=10)
            
            assert len(results) == 30
            assert mock_predict.call_count == 3  # 3 batches


if __name__ == '__main__':
    pytest.main([__file__, '-v'])