"""
Scikit-learn integration for ChemiAccel.

This module provides scikit-learn compatible transformers and estimators
that can be used in ML pipelines for molecular descriptor calculation.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd

# Scikit-learn imports with error handling
try:
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.utils import check_random_state
    from sklearn.utils.validation import check_array, check_is_fitted

    SKLEARN_AVAILABLE = True
except ImportError:
    # Create dummy base classes if sklearn not available
    class BaseEstimator:
        pass

    class TransformerMixin:
        pass

    SKLEARN_AVAILABLE = False

from ..core.calculator import Calculator
from ..core.exceptions import ChemiAccelError, ValidationError
from ..utils.logging import get_logger

logger = get_logger()


class ChemiAccelTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for molecular descriptor calculation.

    This transformer integrates ChemiAccel into scikit-learn pipelines,
    enabling seamless molecular descriptor calculation as part of ML workflows.
    It follows scikit-learn conventions for fit/transform methods.

    Parameters:
        backend (str): Backend to use ('auto', 'cuda', 'jax', 'cpu')
        device (int or str, optional): Device specification for GPU backends
        batch_size (int or str): Batch size for processing
        descriptor_set (str or list): Descriptors to calculate
        handle_invalid (str): How to handle invalid molecules ('drop', 'impute', 'error')
        impute_value (float): Value to use for imputation if handle_invalid='impute'

    Attributes:
        calculator_ (Calculator): Fitted ChemiAccel calculator instance
        feature_names_ (list): Names of calculated descriptors
        n_features_out_ (int): Number of output features

    Example:
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> import chemi_accel as ca
        >>>
        >>> # Create pipeline with ChemiAccel transformer
        >>> pipeline = Pipeline([
        >>>     ('descriptors', ca.ChemiAccelTransformer(descriptor_set='basic')),
        >>>     ('regressor', RandomForestRegressor())
        >>> ])
        >>>
        >>> # Fit pipeline with SMILES and target values
        >>> smiles = ['CCO', 'c1ccccc1', 'CC(=O)O']
        >>> targets = [1.2, 2.3, 1.8]
        >>> pipeline.fit(smiles, targets)
        >>>
        >>> # Make predictions on new molecules
        >>> new_smiles = ['CCC', 'c1cccnc1']
        >>> predictions = pipeline.predict(new_smiles)
    """

    def __init__(
        self,
        backend: str = "auto",
        device: Optional[Union[int, str]] = None,
        batch_size: Union[int, str] = "auto",
        descriptor_set: Union[str, List[str]] = "basic",
        handle_invalid: str = "drop",
        impute_value: float = 0.0,
    ):
        """Initialize ChemiAccel transformer.

        Args:
            backend: Backend type to use
            device: Device specification for GPU backends
            batch_size: Batch size for processing
            descriptor_set: Descriptors to calculate
            handle_invalid: How to handle invalid molecules ('drop', 'impute', 'error')
            impute_value: Value for imputation if handle_invalid='impute'
        """
        if not SKLEARN_AVAILABLE:
            raise ChemiAccelError(
                "scikit-learn is required for ChemiAccelTransformer. "
                "Install with: pip install scikit-learn"
            )

        self.backend = backend
        self.device = device
        self.batch_size = batch_size
        self.descriptor_set = descriptor_set
        self.handle_invalid = handle_invalid
        self.impute_value = impute_value

        # Validate parameters
        if handle_invalid not in ["drop", "impute", "error"]:
            raise ValidationError(
                input_name="handle_invalid",
                input_value=handle_invalid,
                expected_type="'drop', 'impute', or 'error'",
                reason="Invalid invalid molecule handling strategy",
            )

    def fit(self, X, y=None):
        """Fit the transformer by initializing the calculator.

        Args:
            X: Input SMILES strings (array-like)
            y: Target values (ignored, for sklearn compatibility)

        Returns:
            self: Fitted transformer instance
        """
        # Convert input to list of strings
        X = self._validate_input(X)

        # Initialize calculator
        self.calculator_ = Calculator(
            backend=self.backend, device=self.device, batch_size=self.batch_size
        )

        # Configure descriptors
        self.calculator_.configure_descriptors(self.descriptor_set)

        # Store feature information
        self.feature_names_ = self.calculator_.descriptor_sets.copy()
        self.n_features_out_ = len(self.feature_names_)

        logger.info(
            f"ChemiAccelTransformer fitted with {self.n_features_out_} descriptors"
        )
        return self

    def transform(self, X):
        """Transform SMILES strings to molecular descriptors.

        Args:
            X: Input SMILES strings (array-like)

        Returns:
            np.ndarray: Molecular descriptors array of shape (n_samples, n_features)
        """
        check_is_fitted(self, "calculator_")

        # Convert input to list of strings
        X = self._validate_input(X)

        if len(X) == 0:
            return np.empty((0, self.n_features_out_))

        try:
            # Calculate descriptors
            descriptors_df = self.calculator_.pandas(X)

            # Handle invalid molecules
            if len(descriptors_df) != len(X):
                if self.handle_invalid == "error":
                    raise ChemiAccelError(
                        f"Invalid molecules found: expected {len(X)}, got {len(descriptors_df)}"
                    )
                elif self.handle_invalid == "drop":
                    logger.warning(
                        f"Dropped {len(X) - len(descriptors_df)} invalid molecules"
                    )
                # For 'impute', we'll handle missing values below

            # Extract descriptor columns (exclude mol_id and smiles)
            descriptor_cols = [
                col for col in descriptors_df.columns if col not in ["mol_id", "smiles"]
            ]

            if self.handle_invalid == "impute" and len(descriptors_df) != len(X):
                # Create full array with imputed values
                result = np.full((len(X), len(descriptor_cols)), self.impute_value)

                # Fill in calculated values where available
                valid_indices = descriptors_df.index.tolist()
                if valid_indices:
                    result[valid_indices] = descriptors_df[descriptor_cols].values

                logger.warning(
                    f"Imputed {len(X) - len(descriptors_df)} invalid molecules "
                    f"with value {self.impute_value}"
                )
            else:
                result = descriptors_df[descriptor_cols].values

            return result.astype(np.float64)

        except Exception as e:
            raise ChemiAccelError(f"Transformation failed: {e}")

    def fit_transform(self, X, y=None):
        """Fit transformer and transform input in one step.

        Args:
            X: Input SMILES strings
            y: Target values (ignored)

        Returns:
            np.ndarray: Transformed molecular descriptors
        """
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names.

        Args:
            input_features: Input feature names (ignored)

        Returns:
            np.ndarray: Output feature names
        """
        check_is_fitted(self, "feature_names_")
        return np.array(self.feature_names_)

    def _validate_input(self, X):
        """Validate and convert input to list of SMILES strings.

        Args:
            X: Input data

        Returns:
            List[str]: List of SMILES strings
        """
        if isinstance(X, pd.Series):
            return X.tolist()
        elif isinstance(X, (list, tuple)):
            return list(X)
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                return X.tolist()
            elif X.ndim == 2 and X.shape[1] == 1:
                return X.flatten().tolist()
            else:
                raise ValidationError(
                    input_name="X",
                    input_value=f"array with shape {X.shape}",
                    expected_type="1D array or 2D array with 1 column",
                    reason="Invalid array shape for SMILES input",
                )
        else:
            raise ValidationError(
                input_name="X",
                input_value=type(X).__name__,
                expected_type="list, array, or pandas Series",
                reason="Unsupported input type for SMILES strings",
            )

    def _more_tags(self):
        """Additional tags for sklearn compatibility."""
        return {
            "requires_fit": True,
            "X_types": ["1darray"],
            "allow_nan": False,
            "requires_positive_X": False,
        }


class ChemiAccelEstimator(BaseEstimator):
    """Scikit-learn compatible estimator for molecular property prediction.

    This estimator combines molecular descriptor calculation with a
    scikit-learn predictor for end-to-end molecular property prediction.

    Parameters:
        predictor: Scikit-learn predictor instance
        backend (str): ChemiAccel backend to use
        descriptor_set: Descriptors to calculate
        handle_invalid (str): How to handle invalid molecules

    Example:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> import chemi_accel as ca
        >>>
        >>> # Create estimator with random forest predictor
        >>> estimator = ca.ChemiAccelEstimator(
        >>>     predictor=RandomForestRegressor(n_estimators=100),
        >>>     descriptor_set='basic'
        >>> )
        >>>
        >>> # Fit with SMILES and target values
        >>> smiles = ['CCO', 'c1ccccc1', 'CC(=O)O']
        >>> targets = [1.2, 2.3, 1.8]
        >>> estimator.fit(smiles, targets)
        >>>
        >>> # Predict on new molecules
        >>> predictions = estimator.predict(['CCC', 'c1cccnc1'])
    """

    def __init__(
        self,
        predictor,
        backend: str = "auto",
        descriptor_set: Union[str, List[str]] = "basic",
        handle_invalid: str = "drop",
    ):
        """Initialize ChemiAccel estimator.

        Args:
            predictor: Scikit-learn predictor instance
            backend: ChemiAccel backend to use
            descriptor_set: Descriptors to calculate
            handle_invalid: How to handle invalid molecules
        """
        if not SKLEARN_AVAILABLE:
            raise ChemiAccelError(
                "scikit-learn is required for ChemiAccelEstimator. "
                "Install with: pip install scikit-learn"
            )

        self.predictor = predictor
        self.backend = backend
        self.descriptor_set = descriptor_set
        self.handle_invalid = handle_invalid

    def fit(self, X, y):
        """Fit the estimator.

        Args:
            X: Input SMILES strings
            y: Target values

        Returns:
            self: Fitted estimator
        """
        # Initialize transformer
        self.transformer_ = ChemiAccelTransformer(
            backend=self.backend,
            descriptor_set=self.descriptor_set,
            handle_invalid=self.handle_invalid,
        )

        # Transform input to descriptors
        X_descriptors = self.transformer_.fit_transform(X)

        # Handle case where invalid molecules were dropped
        if self.handle_invalid == "drop" and len(X_descriptors) != len(y):
            # Need to align y with valid molecules
            valid_smiles, _ = Calculator().validate_molecules(X)
            if len(valid_smiles) != len(X_descriptors):
                raise ChemiAccelError(
                    "Mismatch between valid molecules and descriptors"
                )

            # Create mask for valid molecules
            valid_mask = []
            valid_idx = 0
            for smiles in X:
                if valid_idx < len(valid_smiles) and smiles == valid_smiles[valid_idx]:
                    valid_mask.append(True)
                    valid_idx += 1
                else:
                    valid_mask.append(False)

            y = np.array(y)[valid_mask]

        # Fit predictor
        self.predictor.fit(X_descriptors, y)

        return self

    def predict(self, X):
        """Make predictions.

        Args:
            X: Input SMILES strings

        Returns:
            np.ndarray: Predictions
        """
        check_is_fitted(self, "transformer_")

        # Transform to descriptors
        X_descriptors = self.transformer_.transform(X)

        # Make predictions
        return self.predictor.predict(X_descriptors)

    def score(self, X, y):
        """Return prediction score.

        Args:
            X: Input SMILES strings
            y: True target values

        Returns:
            float: Prediction score
        """
        return self.predictor.score(self.transformer_.transform(X), y)


# Convenience functions
def create_molecular_pipeline(predictor, descriptor_set: str = "basic", **kwargs):
    """Create a scikit-learn pipeline with ChemiAccel transformer.

    Args:
        predictor: Scikit-learn predictor instance
        descriptor_set: Descriptors to calculate
        **kwargs: Additional arguments for ChemiAccelTransformer

    Returns:
        sklearn.pipeline.Pipeline: Pipeline with ChemiAccel transformer

    Example:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> import chemi_accel as ca
        >>>
        >>> pipeline = ca.create_molecular_pipeline(
        >>>     RandomForestRegressor(),
        >>>     descriptor_set='basic'
        >>> )
    """
    if not SKLEARN_AVAILABLE:
        raise ChemiAccelError("scikit-learn is required for pipeline creation")

    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            (
                "descriptors",
                ChemiAccelTransformer(descriptor_set=descriptor_set, **kwargs),
            ),
            ("predictor", predictor),
        ]
    )
