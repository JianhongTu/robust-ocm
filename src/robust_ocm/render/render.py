"""
Main render module that orchestrates PDF generation, bbox extraction, and image processing
"""

import os
import hashlib

try:
    from .pdf_generator import PDFGenerator
    from .bbox_extractor import BBoxExtractor
    from .image_processor import ImageProcessor
    from .config import Config
except ImportError:
    from robust_ocm.render.pdf_generator import PDFGenerator
    from robust_ocm.render.bbox_extractor import BBoxExtractor
    from robust_ocm.render.image_processor import ImageProcessor
    from robust_ocm.render.config import Config


class TextRenderer:
    """Main class for rendering text to images with bounding box extraction"""
    
    def __init__(self, config_path=None, config_dict=None):
        """
        Initialize text renderer
        
        Args:
            config_path: Path to configuration JSON file
            config_dict: Configuration dictionary (overrides config_path if provided)
        """
        # Load configuration
        if config_dict is None:
            if config_path is None:
                raise ValueError("Must provide either config_path or config_dict")
            self.config = Config.load_config(config_path)
        else:
            self.config = config_dict.copy()
            # Convert special fields if needed
            self.config = Config.merge_configs(Config.get_default_config(), self.config)
        
        # Validate configuration
        Config.validate_config(self.config)
        
        # Initialize components
        self.pdf_generator = PDFGenerator(self.config)
        self.image_processor = ImageProcessor(self.config)
        
        # Initialize bbox extractor only if needed
        self._bbox_extractor = None
    
    @property
    def bbox_extractor(self):
        """Lazy initialization of bbox extractor"""
        if self._bbox_extractor is None:
            self._bbox_extractor = BBoxExtractor()
        return self._bbox_extractor
    
    def render_text(self, text, output_dir, unique_id=None, extraction_level=None):
        """
        Render text to images with optional bounding box extraction
        
        Args:
            text: Input text content
            output_dir: Directory to save output images
            unique_id: Unique identifier (auto-generated if not provided)
            extraction_level: "word" or "line" (None to skip bbox extraction)
            
        Returns:
            dict: Result containing image_paths and optionally bboxes
        """
        # Generate unique ID if not provided
        if unique_id is None:
            unique_id = hashlib.md5(text.encode()).hexdigest()[:16]
        
        # Generate PDF
        pdf_bytes, unique_id = self.pdf_generator.text_to_pdf_bytes(text, unique_id)
        
        # Convert PDF to images
        image_paths = self.image_processor.pdf_to_images(pdf_bytes, unique_id, output_dir)
        
        result = {
            'unique_id': unique_id,
            'image_paths': image_paths
        }
        
        # Extract bounding boxes if requested
        if extraction_level:
            bboxes = self.bbox_extractor.extract_bboxes_from_pdf(
                pdf_bytes, self.config, extraction_level
            )
            result['bboxes'] = bboxes
        
        return result
    
    def render_batch_item(self, item, output_dir, extraction_level="line"):
        """
        Render a single batch item (for multiprocessing)
        
        Args:
            item: Dictionary containing at least 'context' and 'unique_id' keys
            output_dir: Output directory
            extraction_level: Bbox extraction level
            
        Returns:
            dict: Updated item with rendering results
        """
        # Get text and unique_id
        text = item.get('context', '')
        if not text:
            raise ValueError("Item must have 'context' field")
        
        unique_id = item.get('unique_id')
        if not unique_id:
            raise ValueError("Item must have 'unique_id' field")
        
        # Merge item-specific config
        item_config = item.get('config', {}) or {}
        merged_config = Config.merge_configs(self.config, item_config)
        
        # Temporarily update config for this item
        original_config = self.config
        self.config = merged_config
        self.pdf_generator = PDFGenerator(self.config)
        self.image_processor = ImageProcessor(self.config)
        
        try:
            # Render the item
            result = self.render_text(
                text=text,
                output_dir=output_dir,
                unique_id=unique_id,
                extraction_level=extraction_level
            )
            
            # Update item with results
            item['image_paths'] = result['image_paths']
            if 'bboxes' in result:
                item['bboxes'] = result['bboxes']
            
            return item
        
        finally:
            # Restore original config
            self.config = original_config
            self.pdf_generator = PDFGenerator(self.config)
            self.image_processor = ImageProcessor(self.config)