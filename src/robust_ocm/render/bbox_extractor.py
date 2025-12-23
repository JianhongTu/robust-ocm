"""
Bounding box extraction module using PyMuPDF
"""

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Warning: PyMuPDF not installed. Cannot extract PDF bounding boxes.")
    print("Install with: pip install PyMuPDF")
    fitz = None


class BBoxExtractor:
    """Handles bounding box extraction from PDFs"""
    
    def __init__(self):
        """Initialize bbox extractor"""
        if fitz is None:
            raise ImportError("PyMuPDF is required for bbox extraction. Install with: pip install PyMuPDF")
    
    def extract_bboxes_from_pdf(self, pdf_bytes, config, extraction_level="line"):
        """
        Extract bounding boxes from PDF using PyMuPDF
        
        Args:
            pdf_bytes: PDF file content as bytes
            config: Configuration dictionary containing DPI and scaling settings
            extraction_level: "word" for word-level, "line" for line-level (default: "line")
            
        Returns:
            list: List of bounding boxes per page, each as [x0, y0, x1, y1, text]
        """
        # Open PDF from bytes
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        all_pages_bboxes = []
        
        # Scale factors to convert PDF points to pixel coordinates
        dpi = config.get('dpi', 72)
        horizontal_scale = config.get('horizontal-scale', 1.0)
        scale_x = dpi / 72.0 * horizontal_scale
        scale_y = dpi / 72.0
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            if extraction_level == "word":
                page_bboxes = self._extract_word_level_bboxes(page, scale_x, scale_y)
            elif extraction_level == "line":
                page_bboxes = self._extract_line_level_bboxes(page, scale_x, scale_y)
            else:
                raise ValueError(f"Unsupported extraction level: {extraction_level}")
            
            all_pages_bboxes.append(page_bboxes)
        
        doc.close()
        return all_pages_bboxes
    
    def _extract_word_level_bboxes(self, page, scale_x, scale_y):
        """Extract word-level bounding boxes"""
        # get_text("words") returns list of: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        words = page.get_text("words")
        
        page_bboxes = []
        for word_info in words:
            if len(word_info) >= 5:
                x0, y0, x1, y1, word_text = word_info[:5]
                
                # Validate coordinates are finite numbers
                if not all(isinstance(coord, (int, float)) and not (coord == float('inf') or coord == float('-inf') or coord != coord) for coord in [x0, y0, x1, y1]):
                    continue
                
                # Convert from PDF points to pixel coordinates
                pixel_x0 = x0 * scale_x
                pixel_y0 = y0 * scale_y
                pixel_x1 = x1 * scale_x
                pixel_y1 = y1 * scale_y
                
                # Validate pixel coordinates are finite
                if any(coord == float('inf') or coord == float('-inf') or coord != coord for coord in [pixel_x0, pixel_y0, pixel_x1, pixel_y1]):
                    continue
                
                # Round to integers and ensure valid coordinates
                pixel_x0 = max(0, int(round(pixel_x0)))
                pixel_y0 = max(0, int(round(pixel_y0)))
                pixel_x1 = max(0, int(round(pixel_x1)))
                pixel_y1 = max(0, int(round(pixel_y1)))
                
                # Ensure x1 > x0 and y1 > y0
                if pixel_x1 <= pixel_x0:
                    pixel_x1 = pixel_x0 + 1
                if pixel_y1 <= pixel_y0:
                    pixel_y1 = pixel_y0 + 1
                
                # Clean up word text (remove excessive whitespace)
                word_text = word_text.strip()
                if word_text:  # Only include non-empty words
                    page_bboxes.append([pixel_x0, pixel_y0, pixel_x1, pixel_y1, word_text])
        
        return page_bboxes
    
    def _extract_line_level_bboxes(self, page, scale_x, scale_y):
        """Extract line-level bounding boxes using heuristic grouping"""
        # Get structured text data
        text_dict = page.get_text("dict")
        
        page_bboxes = []
        for block in text_dict["blocks"]:
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                if not line["spans"]:
                    continue
                
                # Group all spans in this line to create line-level bbox
                line_text = ""
                line_x0 = float('inf')
                line_y0 = float('inf') 
                line_x1 = float('-inf')
                line_y1 = float('-inf')
                
                for span in line["spans"]:
                    span_text = span.get("text", "")
                    if not span_text.strip():
                        continue
                        
                    span_bbox = span.get("bbox", [0, 0, 0, 0])
                    if len(span_bbox) >= 4:
                        x0, y0, x1, y1 = span_bbox[:4]
                        
                        # Validate coordinates are finite numbers
                        if all(isinstance(coord, (int, float)) and not (coord == float('inf') or coord == float('-inf') or coord != coord) for coord in [x0, y0, x1, y1]):
                            # Update line bounds
                            line_x0 = min(line_x0, x0)
                            line_y0 = min(line_y0, y0)
                            line_x1 = max(line_x1, x1)
                            line_y1 = max(line_y1, y1)
                        
                        # Concatenate text
                        if line_text and not line_text.endswith(' '):
                            line_text += ' '
                        line_text += span_text
                
                # Skip if no valid spans found or coordinates are invalid
                if (line_x0 == float('inf') or line_y0 == float('inf') or 
                    line_x1 == float('-inf') or line_y1 == float('-inf') or
                    not line_text.strip()):
                    continue
                
                # Convert from PDF points to pixel coordinates
                pixel_x0 = line_x0 * scale_x
                pixel_y0 = line_y0 * scale_y
                pixel_x1 = line_x1 * scale_x
                pixel_y1 = line_y1 * scale_y
                
                # Validate pixel coordinates are finite
                if any(coord == float('inf') or coord == float('-inf') or coord != coord for coord in [pixel_x0, pixel_y0, pixel_x1, pixel_y1]):
                    continue
                
                # Round to integers and ensure valid coordinates
                pixel_x0 = max(0, int(round(pixel_x0)))
                pixel_y0 = max(0, int(round(pixel_y0)))
                pixel_x1 = max(0, int(round(pixel_x1)))
                pixel_y1 = max(0, int(round(pixel_y1)))
                
                # Ensure x1 > x0 and y1 > y0
                if pixel_x1 <= pixel_x0:
                    pixel_x1 = pixel_x0 + 1
                if pixel_y1 <= pixel_y0:
                    pixel_y1 = pixel_y0 + 1
                
                # Clean up line text
                line_text = line_text.strip()
                if line_text:  # Only include non-empty lines
                    page_bboxes.append([pixel_x0, pixel_y0, pixel_x1, pixel_y1, line_text])
        
        return page_bboxes