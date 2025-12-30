from .perturbations import register_perturbation, Perturbation
import random
import os
from typing import Any

# Text-based perturbations

@register_perturbation('font_weight')
class FontWeightPerturbation(Perturbation):
    def apply(self, data: Any, weight: str = 'bold', **kwargs) -> Any:
        """
        Apply font weight perturbation by modifying the text with markdown.
        If data is str, return modified str with markdown formatting.
        If data is dict with 'text' or 'context', modify the text field.
        """
        if isinstance(data, str):
            # Apply markdown formatting for bold text
            if weight == 'bold':
                # Wrap the entire text in markdown bold
                return f"**{data}**"
            elif weight == 'italic':
                return f"*{data}*"
            elif weight == 'bold_italic':
                return f"***{data}***"
            else:
                return data
        elif isinstance(data, dict):
            # Handle different text field names
            text_field = None
            if 'context' in data:
                text_field = 'context'
            elif 'text' in data:
                text_field = 'text'
            
            if text_field:
                if weight == 'bold':
                    data[text_field] = f"**{data[text_field]}**"
                elif weight == 'italic':
                    data[text_field] = f"*{data[text_field]}*"
                elif weight == 'bold_italic':
                    data[text_field] = f"***{data[text_field]}***"
            return data
        else:
            raise TypeError("Font weight perturbation requires text string or dict with 'text'/'context' key")

@register_perturbation('reduced_font_size')
class ReducedFontSizePerturbation(Perturbation):
    def apply(self, data: Any, font_size: int = 8, **kwargs) -> Any:
        """
        Apply reduced font size to simulate smaller text rendering.
        
        This perturbation sets a specific font size directly.
        
        Args:
            data: Config dictionary containing rendering settings
            font_size: Target font size in points (default 8)
                      Examples: 6 (very small), 8 (small), 10 (medium), 12 (large)
        
        Returns:
            Modified config dictionary with specified font-size
        
        Note:
            - font_size 6: Very small text
            - font_size 8: Small text (default)
            - font_size 10: Medium-small text
            - font_size 12: Medium text (closer to default 11)
        """
        if not isinstance(data, dict):
            raise TypeError("Reduced font size perturbation requires a config dict")
        
        # Ensure minimum font size to avoid rendering issues
        if font_size < 6:
            font_size = 6
        
        # Update font size in config
        data['font-size'] = font_size
        
        return data

@register_perturbation('tighter_layout')
class TighterLayoutPerturbation(Perturbation):
    def apply(self, data: Any, line_height_factor: float = 0.8, **kwargs) -> Any:
        """
        Apply tighter layout by reducing line height.
        
        This perturbation creates a more compact text layout by reducing line height.
        
        Args:
            data: Config dictionary containing rendering settings
            line_height_factor: Line height scaling factor (default 0.8 for 80% of original)
                                Examples: 0.5 (very tight), 0.7 (tight), 0.8 (default), 0.9 (slightly tight)
        
        Returns:
            Modified config dictionary with tighter layout settings
        
        Note:
            - line_height_factor 0.5: Very tight line spacing (may cause text overlap)
            - line_height_factor 0.7: Tight line spacing
            - line_height_factor 0.8: Tight line spacing (default)
            - line_height_factor 0.9: Slightly tight line spacing
        """
        if not isinstance(data, dict):
            raise TypeError("Tighter layout perturbation requires a config dict")
        
        # Get current line height
        original_line_height = data.get('line-height', 12)
        
        # Calculate new line height
        new_line_height = int(original_line_height * line_height_factor)
        
        # Update line height in config
        data['line-height'] = new_line_height
        
        return data

@register_perturbation('homoglyph_substitution')
class HomoglyphSubstitutionPerturbation(Perturbation):
    def apply(self, data: Any, substitution_rate: float = 0.1, **kwargs) -> Any:
        """
        Apply homoglyph substitution to text.
        If data is str, return modified str.
        If data is dict with 'text', modify the text field.
        """
        if isinstance(data, str):
            # Extended homoglyph mapping for more comprehensive adversarial attacks
            # Organized as character -> list of possible substitutions
            CONFUSABLE_MAP = {
                # Digits ↔ Letters
                "0": ["O", "o"],
                "1": ["l", "I", "|"],
                "2": ["Z"],
                "5": ["S"],
                "6": ["G"],
                "8": ["B"],
                "9": ["g", "q"],

                # Shape-similar single characters
                "c": ["e"],
                "i": ["j"],
                "u": ["v"],
                "n": ["h"],

                # Multi-character OCR confusions (VERY realistic)
                "m": ["rn"],
                "w": ["vv"],
                "d": ["cl"],
                "n": ["ri"],
                "u": ["li"],

                # Punctuation & symbols
                ".": [","],
                ",": ["'"],
                "'": ["`"],
                "-": ["–", "—"],
                "_": ["-"],
                ":": [";"],
                "|": ["I", "l"],
            }
            
            result = []
            for char in data:
                if char in CONFUSABLE_MAP and random.random() < substitution_rate:
                    # Randomly select from the list of possible substitutions
                    substitution = random.choice(CONFUSABLE_MAP[char])
                    result.append(substitution)
                else:
                    result.append(char)
            return ''.join(result)
        elif isinstance(data, dict) and 'text' in data:
            data['text'] = self.apply(data['text'], substitution_rate=substitution_rate, **kwargs)
            return data
        else:
            raise TypeError("Homoglyph substitution requires text string or dict with 'text' key")

@register_perturbation('line_height_compression')
class LineHeightCompressionPerturbation(Perturbation):
    def apply(self, data: Any, compression_factor: float = 0.8, **kwargs) -> Any:
        """
        Apply line height compression by modifying config.
        """
        if not isinstance(data, dict):
            raise TypeError("Line height compression requires a config dict")
        
        if 'line-height' in data:
            data['line-height'] = data['line-height'] * compression_factor
        elif 'font-size' in data:
            # If no explicit line-height, use font-size as base
            data['line-height'] = data['font-size'] * compression_factor
        return data

@register_perturbation('tofu')
class TofuPerturbation(Perturbation):
    def apply(self, data: Any, **kwargs) -> Any:
        """
        Apply tofu perturbation by setting font to Verdana, which may cause tofu for unsupported characters.
        """
        if not isinstance(data, dict):
            raise TypeError("Tofu perturbation requires a config dict")
        
        # Set font-path to Verdana to potentially cause tofu rendering
        data['font-path'] = './config/Verdana.ttf'
        return data

@register_perturbation('dpi_downscale')
class DPIDownscalePerturbation(Perturbation):
    def apply(self, data: Any, dpi: int = 72, **kwargs) -> Any:
        """
        Apply DPI downscaling by modifying the config's DPI value.
        This simulates lower resolution scans/captures at the rendering level.
        
        Args:
            data: Config dictionary containing rendering settings
            dpi: Target DPI value (default 72 for web/screen resolution)
        
        Returns:
            Modified config dictionary with updated DPI
        """
        if not isinstance(data, dict):
            raise TypeError("DPI downscaling perturbation requires a config dict")
        
        # Update the DPI value in the config
        data['dpi'] = dpi
        
        return data