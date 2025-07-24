"""
Input validation utilities for ChemiAccel.

This module provides validation functions for molecular data inputs,
ensuring data quality and consistency across the library.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


def validate_smiles_series(smiles: Union[pd.Series, List[str], np.ndarray]) -> pd.Series:
    """
    Validate and clean a series of SMILES strings.
    
    Args:
        smiles: SMILES strings to validate
        
    Returns:
        Validated pandas Series of SMILES strings
        
    Raises:
        ValueError: If input is invalid or contains no valid SMILES
    """
    # Convert to pandas Series if needed
    if isinstance(smiles, list):
        smiles = pd.Series(smiles)
    elif isinstance(smiles, np.ndarray):
        smiles = pd.Series(smiles)
    elif not isinstance(smiles, pd.Series):
        raise ValueError(f"Expected pandas Series, list, or numpy array, got {type(smiles)}")
    
    # Check for empty input
    if len(smiles) == 0:
        raise ValueError("Input SMILES series is empty")
    
    # Check for null values
    null_count = smiles.isnull().sum()
    if null_count > 0:
        logger.warning(f"Found {null_count} null SMILES strings, removing them")
        smiles = smiles.dropna()
    
    # Check for empty strings
    empty_count = (smiles == '').sum()
    if empty_count > 0:
        logger.warning(f"Found {empty_count} empty SMILES strings, removing them")
        smiles = smiles[smiles != '']
    
    # Basic SMILES validation
    invalid_smiles = []
    for idx, smi in smiles.items():
        if not isinstance(smi, str):
            invalid_smiles.append(idx)
        elif len(smi.strip()) == 0:
            invalid_smiles.append(idx)
    
    if invalid_smiles:
        logger.warning(f"Found {len(invalid_smiles)} invalid SMILES strings, removing them")
        smiles = smiles.drop(invalid_smiles)
    
    # Check if any valid SMILES remain
    if len(smiles) == 0:
        raise ValueError("No valid SMILES strings found after validation")
    
    logger.info(f"Validated {len(smiles)} SMILES strings")
    return smiles


def validate_molecular_dataframe(df: pd.DataFrame, 
                                smiles_column: str = 'smiles',
                                required_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Validate a DataFrame containing molecular data.
    
    Args:
        df: DataFrame to validate
        smiles_column: Name of column containing SMILES strings
        required_columns: List of required column names
        
    Returns:
        Validated DataFrame
        
    Raises:
        ValueError: If DataFrame is invalid
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame, got {type(df)}")
    
    if len(df) == 0:
        raise ValueError("DataFrame is empty")
    
    # Check for SMILES column
    if smiles_column not in df.columns:
        raise ValueError(f"SMILES column '{smiles_column}' not found in DataFrame")
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Validate SMILES
    try:
        validated_smiles = validate_smiles_series(df[smiles_column])
        df = df.loc[validated_smiles.index]
    except ValueError as e:
        raise ValueError(f"SMILES validation failed: {e}")
    
    logger.info(f"Validated DataFrame with {len(df)} rows")
    return df


def validate_property_values(values: Union[pd.Series, np.ndarray, List[float]], 
                           property_name: str,
                           min_value: Optional[float] = None,
                           max_value: Optional[float] = None) -> pd.Series:
    """
    Validate property values for reasonableness.
    
    Args:
        values: Property values to validate
        property_name: Name of the property for error messages
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Validated pandas Series
        
    Raises:
        ValueError: If values are invalid
    """
    # Convert to pandas Series
    if isinstance(values, list):
        values = pd.Series(values)
    elif isinstance(values, np.ndarray):
        values = pd.Series(values)
    elif not isinstance(values, pd.Series):
        raise ValueError(f"Expected pandas Series, list, or numpy array, got {type(values)}")
    
    # Check for empty input
    if len(values) == 0:
        raise ValueError(f"No {property_name} values provided")
    
    # Check for null/infinite values
    null_count = values.isnull().sum()
    inf_count = np.isinf(values).sum()
    
    if null_count > 0:
        logger.warning(f"Found {null_count} null {property_name} values")
    
    if inf_count > 0:
        logger.warning(f"Found {inf_count} infinite {property_name} values")
        values = values[~np.isinf(values)]
    
    # Check value ranges
    if min_value is not None:
        below_min = (values < min_value).sum()
        if below_min > 0:
            logger.warning(f"Found {below_min} {property_name} values below minimum {min_value}")
    
    if max_value is not None:
        above_max = (values > max_value).sum()
        if above_max > 0:
            logger.warning(f"Found {above_max} {property_name} values above maximum {max_value}")
    
    return values


def validate_descriptor_matrix(descriptors: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a matrix of molecular descriptors.
    
    Args:
        descriptors: DataFrame containing molecular descriptors
        
    Returns:
        Validated descriptor DataFrame
        
    Raises:
        ValueError: If descriptor matrix is invalid
    """
    if not isinstance(descriptors, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame, got {type(descriptors)}")
    
    if len(descriptors) == 0:
        raise ValueError("Descriptor matrix is empty")
    
    if descriptors.shape[1] == 0:
        raise ValueError("Descriptor matrix has no columns")
    
    # Check for null values
    null_counts = descriptors.isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Found null values in descriptor columns: {null_counts[null_counts > 0].to_dict()}")
    
    # Check for infinite values
    inf_counts = np.isinf(descriptors.select_dtypes(include=[np.number])).sum()
    if inf_counts.sum() > 0:
        logger.warning(f"Found infinite values in descriptor columns: {inf_counts[inf_counts > 0].to_dict()}")
    
    # Check for constant columns (no variance)
    numeric_cols = descriptors.select_dtypes(include=[np.number]).columns
    constant_cols = []
    
    for col in numeric_cols:
        if descriptors[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        logger.warning(f"Found constant descriptor columns: {constant_cols}")
    
    logger.info(f"Validated descriptor matrix: {descriptors.shape[0]} samples, {descriptors.shape[1]} features")
    return descriptors


def validate_training_data(training_data: pd.DataFrame,
                          smiles_column: str = 'smiles',
                          target_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Validate training data for machine learning models.
    
    Args:
        training_data: Training dataset
        smiles_column: Name of SMILES column
        target_columns: Names of target property columns
        
    Returns:
        Validated training DataFrame
        
    Raises:
        ValueError: If training data is invalid
    """
    # Basic DataFrame validation
    training_data = validate_molecular_dataframe(
        training_data, 
        smiles_column=smiles_column,
        required_columns=target_columns
    )
    
    # Validate target columns if specified
    if target_columns:
        for col in target_columns:
            if col in training_data.columns:
                try:
                    validate_property_values(training_data[col], col)
                except ValueError as e:
                    logger.warning(f"Target column '{col}' validation warning: {e}")
    
    # Check for sufficient data
    min_samples = 10
    if len(training_data) < min_samples:
        raise ValueError(f"Insufficient training data: {len(training_data)} samples (minimum {min_samples})")
    
    logger.info(f"Validated training data: {len(training_data)} samples")
    return training_data


def validate_prediction_input(molecules: Union[pd.Series, List[str], np.ndarray],
                             properties: Optional[List[str]] = None) -> tuple:
    """
    Validate input for property prediction.
    
    Args:
        molecules: SMILES strings for prediction
        properties: List of properties to predict
        
    Returns:
        Tuple of (validated_molecules, validated_properties)
        
    Raises:
        ValueError: If input is invalid
    """
    # Validate molecules
    validated_molecules = validate_smiles_series(molecules)
    
    # Validate properties list
    validated_properties = properties
    if properties is not None:
        if not isinstance(properties, list):
            raise ValueError(f"Properties must be a list, got {type(properties)}")
        
        if len(properties) == 0:
            raise ValueError("Properties list is empty")
        
        # Check for valid property names (basic validation)
        valid_properties = [
            'glass_transition_temp', 'youngs_modulus', 'thermal_stability', 
            'density', 'molecular_weight', 'logp', 'tpsa'
        ]
        
        invalid_properties = [p for p in properties if not isinstance(p, str)]
        if invalid_properties:
            raise ValueError(f"Invalid property names (must be strings): {invalid_properties}")
    
    return validated_molecules, validated_properties


# Property-specific validation ranges
PROPERTY_RANGES = {
    'glass_transition_temp': (-200, 500),  # °C
    'youngs_modulus': (0.001, 1000),      # GPa
    'thermal_stability': (50, 800),        # °C
    'density': (0.1, 10.0),               # g/cm³
    'molecular_weight': (10, 10000),       # g/mol
    'logp': (-10, 20),                     # dimensionless
    'tpsa': (0, 1000),                     # Ų
}


def validate_property_prediction_results(results: pd.DataFrame,
                                       expected_properties: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Validate property prediction results.
    
    Args:
        results: DataFrame with prediction results
        expected_properties: List of expected property columns
        
    Returns:
        Validated results DataFrame
        
    Raises:
        ValueError: If results are invalid
    """
    if not isinstance(results, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame, got {type(results)}")
    
    if len(results) == 0:
        raise ValueError("Results DataFrame is empty")
    
    # Check for expected properties
    if expected_properties:
        missing_props = [prop for prop in expected_properties if prop not in results.columns]
        if missing_props:
            logger.warning(f"Missing expected properties in results: {missing_props}")
    
    # Validate property ranges
    for prop, (min_val, max_val) in PROPERTY_RANGES.items():
        if prop in results.columns:
            values = results[prop].dropna()
            if len(values) > 0:
                out_of_range = ((values < min_val) | (values > max_val)).sum()
                if out_of_range > 0:
                    logger.warning(f"Found {out_of_range} {prop} values outside expected range "
                                 f"({min_val}, {max_val})")
    
    logger.info(f"Validated prediction results: {len(results)} samples, {len(results.columns)} columns")
    return results