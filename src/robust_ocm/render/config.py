"""
Configuration management for the render module
"""

import json
import os
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY


# Alignment mapping
ALIGN_MAP = {
    "LEFT": TA_LEFT,
    "CENTER": TA_CENTER,
    "RIGHT": TA_RIGHT,
    "JUSTIFY": TA_JUSTIFY,
}


class Config:
    """Configuration manager for rendering"""
    
    @staticmethod
    def load_config(config_path):
        """
        Load configuration from JSON file
        
        Args:
            config_path: Path to configuration JSON file
            
        Returns:
            dict: Configuration dictionary with converted values
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Convert colors
        color_fields = ['page-bg-color', 'font-color', 'para-bg-color', 'para-border-color']
        for field in color_fields:
            if field in config and isinstance(config[field], str):
                config[field] = colors.HexColor(config[field])
        
        # Convert alignment
        if 'alignment' in config and isinstance(config['alignment'], str):
            config['alignment'] = ALIGN_MAP.get(config['alignment'], TA_JUSTIFY)
        
        # Convert page size
        if 'page-size' in config and isinstance(config['page-size'], str):
            config['page-size'] = tuple(map(float, config['page-size'].split(',')))
        
        # Validate that all required fields are present and properly formatted
        Config.validate_config(config)
        
        return config
    
    @staticmethod
    def merge_configs(base_config, item_config):
        """
        Merge base configuration with item-specific configuration
        
        Args:
            base_config: Base configuration dictionary
            item_config: Item-specific configuration dictionary
            
        Returns:
            dict: Merged configuration
        """
        # Simply merge - no defaults to fall back to
        config = {**base_config, **item_config}
        
        # Process special fields in item config
        if 'page-size' in item_config and isinstance(item_config['page-size'], str):
            config['page-size'] = tuple(map(float, item_config['page-size'].split(',')))
        
        color_fields = ['page-bg-color', 'font-color', 'para-bg-color', 'para-border-color']
        for field in color_fields:
            if field in item_config and isinstance(item_config[field], str):
                config[field] = colors.HexColor(item_config[field])
        
        if 'alignment' in item_config and isinstance(item_config['alignment'], str):
            config['alignment'] = ALIGN_MAP.get(item_config['alignment'], TA_JUSTIFY)
        
        return config
    
    @staticmethod
    def get_default_config():
        """
        Get default configuration
        
        Returns:
            dict: Default configuration dictionary
        """
        return {
            'page-size': (595, 842),  # A4
            'dpi': 72,
            'margin-x': 20,
            'margin-y': 20,
            'font-path': None,  # Must be provided
            'font-size': 9,
            'line-height': 10,
            'font-color': colors.HexColor('#000000'),
            'alignment': TA_JUSTIFY,
            'horizontal-scale': 1.0,
            'first-line-indent': 0,
            'left-indent': 0,
            'right-indent': 0,
            'space-after': 0,
            'space-before': 0,
            'border-width': 0,
            'border-padding': 0,
            'page-bg-color': colors.HexColor('#FFFFFF'),
            'para-bg-color': colors.HexColor('#FFFFFF'),
            'auto-crop-width': False,
            'auto-crop-last-page': False,
            'newline-markup': '<br/>',
            'remove-line-breaks': False
        }
    
    @staticmethod
    def validate_config(config):
        """
        Validate configuration dictionary
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required fields
        required_fields = ['font-path', 'page-size', 'font-size', 'line-height', 'margin-x', 'margin-y', 
                          'font-color', 'alignment', 'page-bg-color', 'para-bg-color']
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Required field '{field}' is missing from configuration")
        
        font_path = config['font-path']
        # Handle relative paths
        if not os.path.isabs(font_path):
            # Try to resolve relative to common locations
            possible_bases = [
                os.getcwd(),
                os.path.join(os.getcwd(), 'config'),
                os.path.dirname(os.path.abspath(__file__)),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'config')
            ]
            for base in possible_bases:
                test_path = os.path.join(base, font_path)
                if os.path.exists(test_path):
                    font_path = test_path
                    config['font-path'] = font_path
                    break
        
        if not os.path.exists(font_path):
            raise ValueError(f"Font file not found: {font_path}")
        
        # Validate numeric values
        numeric_fields = ['font-size', 'line-height', 'margin-x', 'margin-y', 'dpi']
        for field in numeric_fields:
            if field in config and not isinstance(config[field], (int, float)):
                raise ValueError(f"{field} must be a number")
        
        # Validate page size
        if 'page-size' in config:
            page_size = config['page-size']
            if not isinstance(page_size, (list, tuple)) or len(page_size) != 2:
                raise ValueError("page-size must be a tuple/list of two numbers")
            if not all(isinstance(x, (int, float)) for x in page_size):
                raise ValueError("page-size values must be numbers")