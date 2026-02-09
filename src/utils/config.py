"""Configuration management utilities"""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration handler for the project"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize configuration
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Path to config value (e.g., 'data.test_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
            
        Example:
            >>> config = Config()
            >>> test_size = config.get('data.test_size')
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str):
        """Allow dictionary-like access"""
        return self.config[key]
    
    def __repr__(self):
        return f"Config(config_path='{self.config_path}')"
