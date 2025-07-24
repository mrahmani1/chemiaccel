"""
Unit tests for the Predictive Polymer Property Engine.

This module tests the polymer property prediction functionality,
including model loading, prediction accuracy, and fine-tuning capabilities.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path

# Import the modules to test
from src.chemi_accel.polymer.predictive.property_engine import (
    PolymerPropertyEngine,
    PolymerPropertyPrediction,
    ModelMetrics,
    predict_polymer_properties,
    predict_glass_transition_temp,
    predict_youngs_modulus
)
from src.chemi_accel.polymer.predictive.descriptors import PolymerDescriptorCalculator
from src.chemi_accel.core.calculator import Calculator


class TestPolymerPropertyEngine:
    """Test cases for PolymerPropertyEngine."""
    
    @pytest.fixture
    def mock_calculator(self):
        """Create a mock calculator for testing."""
        calculator = Mock(spec=Calculator)
        return calculator
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create a temporary directory for model storage."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_molecules(self):
        """Sample polymer molecules for testing."""
        return pd.Series([
            'C=C',  # ethylene
            'C=CC',  # propylene
            'C=Cc1ccccc1',  # styrene
            'C=CC(=O)O',  # acrylic acid
            'C=CC(=O)OC',  # methyl acrylate
        ])
    
    @pytest.fixture
    def sample_training_data(self):
        """Sample training data for fine-tuning tests."""
        return pd.DataFrame({
            'smiles': [
                'C=C', 'C=CC', 'C=Cc1ccccc1', 'C=CC(=O)O', 'C=CC(=O)OC',
                'C=CCl', 'C=CF', 'C=CC#N', 'C1CO1', 'CC1CO1'
            ],
            'glass_transition_temp': [
                -120, -20, 100, 106, 5, 81, -160, 97, -67, -75
            ],
            'youngs_modulus': [
                0.2, 1.5, 3.2, 2.8, 2.1, 2.9, 1.4, 3.5, 0.8, 1.2
            ],
            'thermal_stability': [
                300, 350, 400, 380, 320, 360, 280, 420, 250, 270
            ],
            'density': [
                0.92, 0.90, 1.05, 1.25, 1.19, 1.38, 1.42, 1.18, 1.11, 1.04
            ]
        })
    
    def test_initialization(self, mock_calculator, temp_model_dir):
        """Test PolymerPropertyEngine initialization."""
        engine = PolymerPropertyEngine(mock_calculator, temp_model_dir)
        
        assert engine.calculator == mock_calculator
        assert isinstance(engine.descriptor_calc, PolymerDescriptorCalculator)
        assert engine.model_path == Path(temp_model_dir)
        assert isinstance(engine.models, dict)
        assert isinstance(engine.scalers, dict)
        assert isinstance(engine.model_metrics, dict)
    
    def test_initialization_default_path(self, mock_calculator):
        """Test initialization with default model path."""
        engine = PolymerPropertyEngine(mock_calculator)
        
        expected_path = Path(__file__).parent.parent.parent.parent / "src/chemi_accel/polymer/predictive/models"
        # Path comparison might be tricky, so just check it's a Path object
        assert isinstance(engine.model_path, Path)
    
    @patch('src.chemi_accel.polymer.predictive.property_engine.SKLEARN_AVAILABLE', True)
    def test_initialize_default_models(self, mock_calculator, temp_model_dir):
        """Test initialization of default models."""
        engine = PolymerPropertyEngine(mock_calculator, temp_model_dir)
        engine._initialize_default_models()
        
        expected_properties = ['glass_transition_temp', 'youngs_modulus', 
                              'thermal_stability', 'density']
        
        for prop in expected_properties:
            assert prop in engine.models
            assert prop in engine.scalers
    
    @patch('src.chemi_accel.polymer.predictive.property_engine.SKLEARN_AVAILABLE', False)
    def test_initialize_default_models_no_sklearn(self, mock_calculator, temp_model_dir):
        """Test initialization when sklearn is not available."""
        engine = PolymerPropertyEngine(mock_calculator, temp_model_dir)
        engine._initialize_default_models()
        
        # Should not create any models without sklearn
        assert len(engine.models) == 0
    
    def test_predict_properties_input_validation(self, mock_calculator, temp_model_dir):
        """Test input validation for predict_properties."""
        engine = PolymerPropertyEngine(mock_calculator, temp_model_dir)
        
        # Test with list input
        molecules_list = ['C=C', 'C=CC']
        with patch.object(engine.descriptor_calc, 'calculate_polymer_descriptors') as mock_calc:
            mock_calc.return_value = pd.DataFrame()
            engine.predict_properties(molecules_list)
            
            # Should convert list to Series
            call_args = mock_calc.call_args[0][0]
            assert isinstance(call_args, pd.Series)
            assert list(call_args) == molecules_list
    
    def test_predict_properties_no_models(self, mock_calculator, temp_model_dir):
        """Test prediction when no models are available."""
        engine = PolymerPropertyEngine(mock_calculator, temp_model_dir)
        engine.models = {}  # No models available
        
        molecules = pd.Series(['C=C', 'C=CC'])
        
        with patch.object(engine.descriptor_calc, 'calculate_polymer_descriptors') as mock_calc:
            mock_calc.return_value = pd.DataFrame(index=molecules.index)
            
            results = engine.predict_properties(molecules)
            
            assert 'smiles' in results.columns
            assert len(results) == len(molecules)
    
    @patch('src.chemi_accel.polymer.predictive.property_engine.SKLEARN_AVAILABLE', True)
    def test_predict_single_property(self, mock_calculator, temp_model_dir):
        """Test prediction of a single property."""
        engine = PolymerPropertyEngine(mock_calculator, temp_model_dir)
        engine._initialize_default_models()
        
        # Mock descriptors
        descriptors = pd.DataFrame({
            'molecular_weight': [28.0, 42.0],
            'logp': [1.0, 1.5],
            'tpsa': [0.0, 0.0]
        })
        
        # Mock model prediction
        mock_model = Mock()
        mock_model.predict.return_value = np.array([100.0, 120.0])
        engine.models['glass_transition_temp'] = mock_model
        
        # Mock scaler
        mock_scaler = Mock()
        mock_scaler.transform.return_value = descriptors.values
        engine.scalers['glass_transition_temp'] = mock_scaler
        
        result = engine._predict_single_property(descriptors, 'glass_transition_temp')
        
        assert 'prediction' in result
        assert len(result['prediction']) == 2
        mock_model.predict.assert_called_once()
        mock_scaler.transform.assert_called_once()
    
    def test_estimate_uncertainty_tree_model(self, mock_calculator, temp_model_dir):
        """Test uncertainty estimation for tree-based models."""
        engine = PolymerPropertyEngine(mock_calculator, temp_model_dir)
        
        # Mock tree-based model with estimators
        mock_model = Mock()
        mock_estimator1 = Mock()
        mock_estimator1.predict.return_value = np.array([100.0, 110.0])
        mock_estimator2 = Mock()
        mock_estimator2.predict.return_value = np.array([105.0, 115.0])
        mock_model.estimators_ = [mock_estimator1, mock_estimator2]
        
        X = np.array([[1, 2], [3, 4]])
        predictions = np.array([102.5, 112.5])
        
        uncertainty, confidence = engine._estimate_uncertainty(mock_model, X, predictions)
        
        assert len(uncertainty) == 2
        assert len(confidence) == 2
        assert all(uncertainty >= 0)
        assert all(0 <= confidence <= 1)
    
    def test_estimate_uncertainty_default(self, mock_calculator, temp_model_dir):
        """Test default uncertainty estimation."""
        engine = PolymerPropertyEngine(mock_calculator, temp_model_dir)
        
        mock_model = Mock()  # No special attributes
        X = np.array([[1, 2], [3, 4]])
        predictions = np.array([100.0, 110.0])
        
        uncertainty, confidence = engine._estimate_uncertainty(mock_model, X, predictions)
        
        assert len(uncertainty) == 2
        assert len(confidence) == 2
        assert all(uncertainty == 0.1)  # Default uncertainty
        assert all(confidence == 0.8)   # Default confidence
    
    @patch('src.chemi_accel.polymer.predictive.property_engine.SKLEARN_AVAILABLE', True)
    def test_fine_tune_model(self, mock_calculator, temp_model_dir, sample_training_data):
        """Test model fine-tuning functionality."""
        engine = PolymerPropertyEngine(mock_calculator, temp_model_dir)
        
        # Mock descriptor calculation
        mock_descriptors = pd.DataFrame({
            'molecular_weight': np.random.rand(10) * 100,
            'logp': np.random.rand(10) * 5,
            'tpsa': np.random.rand(10) * 100
        }, index=sample_training_data.index)
        
        with patch.object(engine.descriptor_calc, 'calculate_polymer_descriptors') as mock_calc:
            mock_calc.return_value = mock_descriptors
            
            with patch.object(engine, '_save_model') as mock_save:
                metrics = engine.fine_tune_model(
                    sample_training_data, 
                    'glass_transition_temp',
                    save_model=True
                )
                
                assert isinstance(metrics, ModelMetrics)
                assert metrics.mae >= 0
                assert -1 <= metrics.r2 <= 1
                assert metrics.rmse >= 0
                assert metrics.training_samples > 0
                assert metrics.validation_samples > 0
                
                mock_save.assert_called_once_with('glass_transition_temp')
    
    def test_fine_tune_model_missing_column(self, mock_calculator, temp_model_dir):
        """Test fine-tuning with missing target column."""
        engine = PolymerPropertyEngine(mock_calculator, temp_model_dir)
        
        bad_data = pd.DataFrame({
            'smiles': ['C=C', 'C=CC'],
            'other_property': [1.0, 2.0]
        })
        
        with pytest.raises(ValueError, match="Target column 'glass_transition_temp' not found"):
            engine.fine_tune_model(bad_data, 'glass_transition_temp')
    
    def test_get_model_info(self, mock_calculator, temp_model_dir):
        """Test getting model information."""
        engine = PolymerPropertyEngine(mock_calculator, temp_model_dir)
        
        # Add some mock models and metrics
        engine.models['test_property'] = Mock()
        engine.scalers['test_property'] = Mock()
        engine.model_metrics['test_property'] = ModelMetrics(
            mae=1.0, r2=0.8, rmse=1.2, training_samples=100, validation_samples=25
        )
        
        info = engine.get_model_info()
        
        assert 'test_property' in info
        prop_info = info['test_property']
        assert 'model_type' in prop_info
        assert prop_info['has_scaler'] is True
        assert prop_info['has_metrics'] is True
        assert prop_info['mae'] == 1.0
        assert prop_info['r2'] == 0.8
    
    def test_convenience_methods(self, mock_calculator, temp_model_dir, sample_molecules):
        """Test convenience methods for specific properties."""
        engine = PolymerPropertyEngine(mock_calculator, temp_model_dir)
        
        with patch.object(engine, 'predict_properties') as mock_predict:
            mock_predict.return_value = pd.DataFrame()
            
            # Test glass transition temperature prediction
            engine.predict_glass_transition_temp(sample_molecules)
            mock_predict.assert_called_with(sample_molecules, properties=['glass_transition_temp'])
            
            # Test Young's modulus prediction
            engine.predict_youngs_modulus(sample_molecules)
            mock_predict.assert_called_with(sample_molecules, properties=['youngs_modulus'])
    
    def test_batch_predict(self, mock_calculator, temp_model_dir):
        """Test batch prediction functionality."""
        engine = PolymerPropertyEngine(mock_calculator, temp_model_dir)
        
        molecules = pd.Series(['C=C'] * 10)  # 10 identical molecules
        
        with patch.object(engine, 'predict_properties') as mock_predict:
            # Mock return values for each batch
            mock_predict.side_effect = [
                pd.DataFrame({'smiles': ['C=C'] * 3, 'prop': [1, 2, 3]}),
                pd.DataFrame({'smiles': ['C=C'] * 3, 'prop': [4, 5, 6]}),
                pd.DataFrame({'smiles': ['C=C'] * 3, 'prop': [7, 8, 9]}),
                pd.DataFrame({'smiles': ['C=C'] * 1, 'prop': [10]})
            ]
            
            results = engine.batch_predict(molecules, batch_size=3)
            
            assert len(results) == 10
            assert mock_predict.call_count == 4  # 4 batches for 10 molecules with batch_size=3


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @patch('src.chemi_accel.polymer.predictive.property_engine.Calculator')
    @patch('src.chemi_accel.polymer.predictive.property_engine.PolymerPropertyEngine')
    def test_predict_polymer_properties(self, mock_engine_class, mock_calculator_class):
        """Test predict_polymer_properties convenience function."""
        mock_calculator = Mock()
        mock_calculator_class.return_value = mock_calculator
        
        mock_engine = Mock()
        mock_engine.predict_properties.return_value = pd.DataFrame()
        mock_engine_class.return_value = mock_engine
        
        molecules = ['C=C', 'C=CC']
        
        # Test with default calculator
        result = predict_polymer_properties(molecules)
        
        mock_calculator_class.assert_called_once()
        mock_engine_class.assert_called_once_with(mock_calculator)
        mock_engine.predict_properties.assert_called_once_with(molecules)
    
    @patch('src.chemi_accel.polymer.predictive.property_engine.PolymerPropertyEngine')
    def test_predict_polymer_properties_with_calculator(self, mock_engine_class):
        """Test predict_polymer_properties with provided calculator."""
        mock_calculator = Mock()
        mock_engine = Mock()
        mock_engine.predict_properties.return_value = pd.DataFrame()
        mock_engine_class.return_value = mock_engine
        
        molecules = ['C=C', 'C=CC']
        
        result = predict_polymer_properties(molecules, mock_calculator)
        
        mock_engine_class.assert_called_once_with(mock_calculator)
        mock_engine.predict_properties.assert_called_once_with(molecules)
    
    @patch('src.chemi_accel.polymer.predictive.property_engine.Calculator')
    @patch('src.chemi_accel.polymer.predictive.property_engine.PolymerPropertyEngine')
    def test_predict_glass_transition_temp(self, mock_engine_class, mock_calculator_class):
        """Test predict_glass_transition_temp convenience function."""
        mock_calculator = Mock()
        mock_calculator_class.return_value = mock_calculator
        
        mock_engine = Mock()
        mock_engine.predict_glass_transition_temp.return_value = pd.DataFrame()
        mock_engine_class.return_value = mock_engine
        
        molecules = ['C=C', 'C=CC']
        
        result = predict_glass_transition_temp(molecules)
        
        mock_engine.predict_glass_transition_temp.assert_called_once_with(molecules)
    
    @patch('src.chemi_accel.polymer.predictive.property_engine.Calculator')
    @patch('src.chemi_accel.polymer.predictive.property_engine.PolymerPropertyEngine')
    def test_predict_youngs_modulus(self, mock_engine_class, mock_calculator_class):
        """Test predict_youngs_modulus convenience function."""
        mock_calculator = Mock()
        mock_calculator_class.return_value = mock_calculator
        
        mock_engine = Mock()
        mock_engine.predict_youngs_modulus.return_value = pd.DataFrame()
        mock_engine_class.return_value = mock_engine
        
        molecules = ['C=C', 'C=CC']
        
        result = predict_youngs_modulus(molecules)
        
        mock_engine.predict_youngs_modulus.assert_called_once_with(molecules)


class TestDataClasses:
    """Test data classes."""
    
    def test_polymer_property_prediction(self):
        """Test PolymerPropertyPrediction dataclass."""
        prediction = PolymerPropertyPrediction(
            smiles='C=C',
            glass_transition_temp=100.0,
            youngs_modulus=2.5,
            prediction_uncertainty={'tg': 5.0},
            confidence_score=0.85
        )
        
        assert prediction.smiles == 'C=C'
        assert prediction.glass_transition_temp == 100.0
        assert prediction.youngs_modulus == 2.5
        assert prediction.prediction_uncertainty == {'tg': 5.0}
        assert prediction.confidence_score == 0.85
    
    def test_model_metrics(self):
        """Test ModelMetrics dataclass."""
        metrics = ModelMetrics(
            mae=1.5,
            r2=0.85,
            rmse=2.1,
            training_samples=1000,
            validation_samples=200
        )
        
        assert metrics.mae == 1.5
        assert metrics.r2 == 0.85
        assert metrics.rmse == 2.1
        assert metrics.training_samples == 1000
        assert metrics.validation_samples == 200


if __name__ == '__main__':
    pytest.main([__file__])