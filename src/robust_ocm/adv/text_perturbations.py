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

@register_perturbation('kerning_collisions')
class KerningCollisionsPerturbation(Perturbation):
    def apply(self, data: Any, collision_factor: float = 0.1, **kwargs) -> Any:
        """
        Apply kerning collision by modifying config.
        This is a placeholder - ReportLab doesn't directly support kerning adjustments.
        """
        if not isinstance(data, dict):
            raise TypeError("Kerning perturbation requires a config dict")
        
        # Placeholder: could adjust font size or spacing if possible
        # For now, slightly reduce font size to simulate tighter spacing
        if 'font-size' in data:
            data['font-size'] = max(6, data['font-size'] * (1 - collision_factor))
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