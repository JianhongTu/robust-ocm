"""
PDF generation module for converting text to PDF using ReportLab
"""

import io
import os
import re
import hashlib
from xml.sax.saxutils import escape

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib import colors


class PDFGenerator:
    """Handles PDF generation from text content"""
    
    def __init__(self, config):
        """
        Initialize PDF generator with configuration
        
        Args:
            config: Configuration dictionary with PDF generation settings
        """
        self.config = config
        self._register_font()
    
    def _register_font(self):
        """Register the font for PDF generation"""
        font_path = self.config.get('font-path')
        if font_path:
            font_name = os.path.basename(font_path).split('.')[0]
            try:
                pdfmetrics.registerFont(TTFont(font_name, font_path))
            except:
                pass  # Font already registered
    
    def text_to_pdf_bytes(self, text, unique_id=None):
        """
        Convert text to PDF bytes
        
        Args:
            text: Input text content
            unique_id: Unique identifier (optional, auto-generated if not provided)
            
        Returns:
            tuple: (pdf_bytes, unique_id)
        """
        # Generate unique ID if not provided
        if unique_id is None:
            unique_id = hashlib.md5(text.encode()).hexdigest()[:16]
        
        # Extract configuration parameters
        page_size = self.config.get('page-size', A4)
        margin_x = self.config.get('margin-x', 20)
        margin_y = self.config.get('margin-y', 20)
        font_path = self.config.get('font-path')
        assert font_path, "Must provide font-path"
        
        font_name = os.path.basename(font_path).split('.')[0]
        font_size = self.config.get('font-size', 9)
        line_height = self.config.get('line-height') or (font_size + 1)
        
        page_bg_color = self.config.get('page-bg-color', colors.HexColor('#FFFFFF'))
        font_color = self.config.get('font-color', colors.HexColor('#000000'))
        para_bg_color = self.config.get('para-bg-color', colors.HexColor('#FFFFFF'))
        para_border_color = self.config.get('para-border-color', colors.HexColor('#FFFFFF'))
        
        first_line_indent = self.config.get('first-line-indent', 0)
        left_indent = self.config.get('left-indent', 0)
        right_indent = self.config.get('right-indent', 0)
        alignment = self.config.get('alignment', TA_JUSTIFY)
        space_before = self.config.get('space-before', 0)
        space_after = self.config.get('space-after', 0)
        border_width = self.config.get('border-width', 0)
        border_padding = self.config.get('border-padding', 0)
        
        newline_markup = self.config.get('newline-markup', '<br/>')
        
        # Create PDF in memory
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=page_size,
            leftMargin=margin_x,
            rightMargin=margin_x,
            topMargin=margin_y,
            bottomMargin=margin_y,
        )
        
        # Create paragraph style
        styles = getSampleStyleSheet()
        RE_CJK = re.compile(r'[\u4E00-\u9FFF]')
        
        custom = ParagraphStyle(
            name="Custom",
            parent=styles["Normal"],
            fontName=font_name,
            fontSize=font_size,
            leading=line_height,
            textColor=font_color,
            backColor=para_bg_color,
            borderColor=para_border_color,
            borderWidth=border_width,
            borderPadding=border_padding,
            firstLineIndent=first_line_indent,
            wordWrap="CJK" if RE_CJK.search(text) else None,
            leftIndent=left_indent,
            rightIndent=right_indent,
            alignment=alignment,
            spaceBefore=space_before,
            spaceAfter=space_after,
        )
        
        # Process text
        def replace_spaces(s):
            return re.sub(r' {2,}', lambda m: '&nbsp;'*len(m.group()), s)
        
        text = text.replace('\xad', '').replace('\u200b', '')
        processed_text = replace_spaces(escape(text))
        parts = processed_text.split('\n')
        
        # Create paragraphs in batches
        story = []
        turns = 30
        for i in range(0, len(parts), turns):
            tmp_text = newline_markup.join(parts[i:i+turns])
            story.append(Paragraph(tmp_text, custom))
        
        # Build PDF
        doc.build(
            story,
            onFirstPage=lambda c, d: (c.saveState(), c.setFillColor(page_bg_color), 
                                    c.rect(0, 0, page_size[0], page_size[1], stroke=0, fill=1), 
                                    c.restoreState()),
            onLaterPages=lambda c, d: (c.saveState(), c.setFillColor(page_bg_color), 
                                     c.rect(0, 0, page_size[0], page_size[1], stroke=0, fill=1), 
                                     c.restoreState())
        )
        
        pdf_bytes = buf.getvalue()
        buf.close()
        
        return pdf_bytes, unique_id