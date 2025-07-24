"""
Logging utilities for ChemiAccel.

Provides centralized logging configuration with support for different
log levels, formatters, and output destinations.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any
from enum import Enum


class LogLevel(Enum):
    """Enumeration of supported log levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class ChemiAccelLogger:
    """Centralized logger for ChemiAccel with configurable formatting."""
    
    _instance: Optional['ChemiAccelLogger'] = None
    _logger: Optional[logging.Logger] = None
    
    def __new__(cls) -> 'ChemiAccelLogger':
        """Singleton pattern to ensure single logger instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the logger if not already initialized."""
        if self._logger is None:
            self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Set up the logger with default configuration."""
        self._logger = logging.getLogger('chemi_accel')
        self._logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if not self._logger.handlers:
            self._add_console_handler()
    
    def _add_console_handler(self) -> None:
        """Add console handler with formatted output."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        self._logger.addHandler(console_handler)
    
    def configure(
        self,
        level: Union[LogLevel, str, int] = LogLevel.INFO,
        format_string: Optional[str] = None,
        log_file: Optional[Union[str, Path]] = None,
        enable_console: bool = True
    ) -> None:
        """Configure the logger with custom settings.
        
        Args:
            level: Logging level (LogLevel enum, string, or int)
            format_string: Custom format string for log messages
            log_file: Optional file path for log output
            enable_console: Whether to enable console output
        """
        # Clear existing handlers
        self._logger.handlers.clear()
        
        # Set log level
        if isinstance(level, LogLevel):
            self._logger.setLevel(level.value)
        elif isinstance(level, str):
            self._logger.setLevel(getattr(logging, level.upper()))
        else:
            self._logger.setLevel(level)
        
        # Set up formatter
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        formatter = logging.Formatter(
            fmt=format_string,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add console handler if enabled
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self._logger.level)
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)
        
        # Add file handler if specified
        if log_file is not None:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(self._logger.level)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance."""
        return self._logger
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._logger.critical(message, **kwargs)


# Global logger instance
_global_logger = ChemiAccelLogger()

# Convenience functions for direct logging
def get_logger() -> logging.Logger:
    """Get the global ChemiAccel logger."""
    return _global_logger.get_logger()

def configure_logging(
    level: Union[LogLevel, str, int] = LogLevel.INFO,
    format_string: Optional[str] = None,
    log_file: Optional[Union[str, Path]] = None,
    enable_console: bool = True
) -> None:
    """Configure global logging settings.
    
    Args:
        level: Logging level
        format_string: Custom format string
        log_file: Optional log file path
        enable_console: Enable console output
    """
    _global_logger.configure(level, format_string, log_file, enable_console)

def debug(message: str, **kwargs) -> None:
    """Log debug message using global logger."""
    _global_logger.debug(message, **kwargs)

def info(message: str, **kwargs) -> None:
    """Log info message using global logger."""
    _global_logger.info(message, **kwargs)

def warning(message: str, **kwargs) -> None:
    """Log warning message using global logger."""
    _global_logger.warning(message, **kwargs)

def error(message: str, **kwargs) -> None:
    """Log error message using global logger."""
    _global_logger.error(message, **kwargs)

def critical(message: str, **kwargs) -> None:
    """Log critical message using global logger."""
    _global_logger.critical(message, **kwargs)