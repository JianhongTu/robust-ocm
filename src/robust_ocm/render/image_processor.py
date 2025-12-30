"""
Image processing module for converting PDF to PNG with optional cropping
"""

import os
import gc
import numpy as np
from PIL import Image

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Warning: PyMuPDF not installed. Cannot convert PDF to images.")
    print("Install with: pip install pymupdf")
    fitz = None


class ImageProcessor:
    """Handles PDF to image conversion and processing"""
    
    def __init__(self, config):
        """
        Initialize image processor with configuration
        
        Args:
            config: Configuration dictionary with image processing settings
        """
        self.config = config
    
    def pdf_to_images(self, pdf_bytes, unique_id, output_dir):
        """
        Convert PDF bytes to PNG images using PyMuPDF
        
        Args:
            pdf_bytes: PDF file content as bytes
            unique_id: Unique identifier for the file naming
            output_dir: Directory to save the images
            
        Returns:
            list: List of generated image paths
        """
        if fitz is None:
            raise ImportError("PyMuPDF is required for PDF to image conversion. Install with: pip install pymupdf")
        
        # Create output directory (flat structure)
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract configuration parameters
        dpi = self.config.get('dpi', 72)
        horizontal_scale = self.config.get('horizontal-scale', 1.0)
        auto_crop_last_page = self.config.get('auto-crop-last-page', False)
        auto_crop_width = self.config.get('auto-crop-width', False)
        margin_x = self.config.get('margin-x', 20)
        margin_y = self.config.get('margin-y', 20)
        
        # Open PDF with PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        num_pages = len(doc)
        image_paths = []
        
        # Calculate zoom factor for DPI scaling
        # PDF default is 72 DPI, so zoom = target_dpi / 72
        zoom = (dpi / 72.0) * horizontal_scale
        mat = fitz.Matrix(zoom, zoom)
        
        for page_num in range(num_pages):
            page = doc.load_page(page_num)
            
            # Render page to pixmap with DPI scaling
            pix = page.get_pixmap(matrix=mat)
            
            # Convert pixmap to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Adaptive cropping
            if auto_crop_width or (auto_crop_last_page and page_num + 1 == num_pages):
                img = self._apply_adaptive_cropping(
                    img, page_num + 1, num_pages, margin_x, margin_y,
                    auto_crop_width, auto_crop_last_page
                )
            
            # Generate representative filename
            out_path = os.path.join(output_dir, f"{unique_id}_page_{page_num + 1:03d}.png")
            img.save(out_path, 'PNG')
            image_paths.append(os.path.abspath(out_path))
            img.close()
        
        doc.close()
        del pdf_bytes
        gc.collect()
        
        return image_paths
    
    def _apply_adaptive_cropping(self, img, page_num, total_pages, margin_x, margin_y, 
                                auto_crop_width, auto_crop_last_page):
        """
        Apply adaptive cropping to the image based on content
        
        Args:
            img: PIL Image object
            page_num: Current page number
            total_pages: Total number of pages
            margin_x: Horizontal margin
            margin_y: Vertical margin
            auto_crop_width: Whether to crop width
            auto_crop_last_page: Whether to crop last page
            
        Returns:
            PIL.Image: Cropped image
        """
        gray = np.array(img.convert("L"))
        bg_gray = np.median(gray[:2, :2])
        tolerance = 5
        mask = np.abs(gray - bg_gray) > tolerance
        
        if auto_crop_width:
            cols = np.where(mask.any(axis=0))[0]
            if cols.size:
                rightmost_col = cols[-1] + 1
                right = min(img.width, rightmost_col + margin_x)
                img = img.crop((0, 0, right, img.height))
        
        if auto_crop_last_page and page_num == total_pages:
            rows = np.where(mask.any(axis=1))[0]
            if rows.size:
                last_row = rows[-1]
                lower = min(img.height, last_row + margin_y)
                img = img.crop((0, 0, img.width, lower))
        
        return img