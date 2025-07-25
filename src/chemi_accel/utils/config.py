"""
Configuration management for ChemiAccel.

Provides centralized configuration handling with support for environment
variables, configuration files, and runtime settings.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..core.exceptions import ConfigurationError


class BackendType(Enum):
    """Supported backend types."""

    AUTO = "auto"
    CUDA = "cuda"
    JAX = "jax"
    CPU = "cpu"


@dataclass
class PerformanceConfig:
    """Performance-related configuration settings."""

    batch_size: Union[int, str] = "auto"  # "auto" or specific batch size
    max_memory_usage: float = 0.8  # Maximum fraction of GPU memory to use
    enable_memory_pool: bool = True  # Enable memory pooling for efficiency
    num_workers: int = -1  # Number of worker threads (-1 for auto)
    prefetch_factor: int = 2  # Number of batches to prefetch


@dataclass
class BackendConfig:
    """Backend-specific configuration settings."""

    preferred_backend: BackendType = BackendType.AUTO
    cuda_device: Union[int, str] = "auto"  # CUDA device ID or "auto"
    jax_platform: str = "auto"  # JAX platform preference
    enable_jit: bool = True  # Enable JIT compilation for JAX
    enable_mixed_precision: bool = False  # Enable mixed precision computation


@dataclass
class LoggingConfig:
    """Logging configuration settings."""

    level: str = "INFO"
    enable_console: bool = True
    log_file: Optional[str] = None
    format_string: Optional[str] = None
    enable_performance_logging: bool = False


@dataclass
class ValidationConfig:
    """Input validation configuration."""

    strict_validation: bool = True  # Enable strict input validation
    skip_invalid_molecules: bool = True  # Skip invalid molecules instead of failing
    max_molecule_size: int = 1000  # Maximum number of atoms per molecule
    enable_sanitization: bool = True  # Enable molecule sanitization


@dataclass
class ChemiAccelConfig:
    """Main configuration class for ChemiAccel."""

    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    # Additional settings
    cache_descriptors: bool = False  # Enable descriptor caching
    cache_dir: Optional[str] = None  # Cache directory path
    random_seed: Optional[int] = None  # Random seed for reproducibility


class ConfigManager:
    """Manages configuration for ChemiAccel with multiple sources."""

    _instance: Optional["ConfigManager"] = None
    _config: Optional[ChemiAccelConfig] = None

    # Environment variable prefix
    ENV_PREFIX = "CHEMI_ACCEL_"

    # Default config file locations
    DEFAULT_CONFIG_PATHS = [
        Path.home() / ".chemi_accel" / "config.json",
        Path.cwd() / "chemi_accel_config.json",
        Path.cwd() / ".chemi_accel.json",
    ]

    def __new__(cls) -> "ConfigManager":
        """Singleton pattern for configuration manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize configuration manager."""
        if self._config is None:
            self._config = self._load_config()

    def _load_config(self) -> ChemiAccelConfig:
        """Load configuration from multiple sources."""
        # Start with default configuration
        config = ChemiAccelConfig()

        # Load from config file if available
        config_file = self._find_config_file()
        if config_file:
            file_config = self._load_from_file(config_file)
            config = self._merge_configs(config, file_config)

        # Override with environment variables
        env_config = self._load_from_env()
        config = self._merge_configs(config, env_config)

        return config

    def _find_config_file(self) -> Optional[Path]:
        """Find the first available configuration file."""
        for config_path in self.DEFAULT_CONFIG_PATHS:
            if config_path.exists():
                return config_path
        return None

    def _load_from_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(config_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise ConfigurationError(
                parameter="config_file",
                value=str(config_path),
                reason=f"Failed to load configuration file: {e}",
            )

    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}

        # Map environment variables to config structure
        env_mappings = {
            f"{self.ENV_PREFIX}BACKEND": ("backend", "preferred_backend"),
            f"{self.ENV_PREFIX}CUDA_DEVICE": ("backend", "cuda_device"),
            f"{self.ENV_PREFIX}BATCH_SIZE": ("performance", "batch_size"),
            f"{self.ENV_PREFIX}MAX_MEMORY": ("performance", "max_memory_usage"),
            f"{self.ENV_PREFIX}LOG_LEVEL": ("logging", "level"),
            f"{self.ENV_PREFIX}LOG_FILE": ("logging", "log_file"),
            f"{self.ENV_PREFIX}STRICT_VALIDATION": ("validation", "strict_validation"),
        }

        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if section not in env_config:
                    env_config[section] = {}
                env_config[section][key] = self._parse_env_value(value)

        return env_config

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try to parse as JSON first (for complex values)
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        # Parse common boolean values
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        elif value.lower() in ("false", "no", "0", "off"):
            return False

        # Try to parse as number
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # Return as string
        return value

    def _merge_configs(
        self, base_config: ChemiAccelConfig, override_dict: Dict[str, Any]
    ) -> ChemiAccelConfig:
        """Merge configuration dictionary into base configuration."""
        config_dict = asdict(base_config)

        for section, values in override_dict.items():
            if section in config_dict and isinstance(values, dict):
                config_dict[section].update(values)
            else:
                config_dict[section] = values

        return self._dict_to_config(config_dict)

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ChemiAccelConfig:
        """Convert dictionary to ChemiAccelConfig object."""
        try:
            # Handle nested dataclass conversion
            if "performance" in config_dict:
                config_dict["performance"] = PerformanceConfig(
                    **config_dict["performance"]
                )
            if "backend" in config_dict:
                # Convert backend string to enum if needed
                if "preferred_backend" in config_dict["backend"]:
                    backend_val = config_dict["backend"]["preferred_backend"]
                    if isinstance(backend_val, str):
                        config_dict["backend"]["preferred_backend"] = BackendType(
                            backend_val
                        )
                config_dict["backend"] = BackendConfig(**config_dict["backend"])
            if "logging" in config_dict:
                config_dict["logging"] = LoggingConfig(**config_dict["logging"])
            if "validation" in config_dict:
                config_dict["validation"] = ValidationConfig(
                    **config_dict["validation"]
                )

            return ChemiAccelConfig(**config_dict)
        except (TypeError, ValueError) as e:
            raise ConfigurationError(
                parameter="config_structure",
                value=config_dict,
                reason=f"Invalid configuration structure: {e}",
            )

    def get_config(self) -> ChemiAccelConfig:
        """Get the current configuration."""
        return self._config

    def update_config(self, **kwargs) -> None:
        """Update configuration with new values."""
        config_dict = asdict(self._config)

        for key, value in kwargs.items():
            if "." in key:
                # Handle nested keys like 'backend.preferred_backend'
                section, subkey = key.split(".", 1)
                if section in config_dict:
                    config_dict[section][subkey] = value
            else:
                config_dict[key] = value

        self._config = self._dict_to_config(config_dict)

    def save_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration to file."""
        if config_path is None:
            config_path = self.DEFAULT_CONFIG_PATHS[0]

        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = asdict(self._config)

        # Convert enums to strings for JSON serialization
        if "backend" in config_dict and "preferred_backend" in config_dict["backend"]:
            config_dict["backend"]["preferred_backend"] = config_dict["backend"][
                "preferred_backend"
            ].value

        try:
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
        except OSError as e:
            raise ConfigurationError(
                parameter="config_file",
                value=str(config_path),
                reason=f"Failed to save configuration: {e}",
            )


# Global configuration manager instance
_global_config_manager = ConfigManager()


# Convenience functions
def get_config() -> ChemiAccelConfig:
    """Get the global configuration."""
    return _global_config_manager.get_config()


def update_config(**kwargs) -> None:
    """Update global configuration."""
    _global_config_manager.update_config(**kwargs)


def save_config(config_path: Optional[Union[str, Path]] = None) -> None:
    """Save global configuration to file."""
    _global_config_manager.save_config(config_path)
