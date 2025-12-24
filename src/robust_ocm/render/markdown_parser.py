"""
Markdown parser for ReportLab integration
"""

import re
from xml.sax.saxutils import escape
from reportlab.platypus import Paragraph, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY


class MarkdownParser:
    """Parse markdown text and convert to ReportLab compatible elements"""
    
    def __init__(self, base_style, font_name, font_size, font_color):
        """
        Initialize markdown parser
        
        Args:
            base_style: Base ParagraphStyle to extend
            font_name: Font name to use
            font_size: Base font size
            font_color: Font color
        """
        self.base_style = base_style
        self.font_name = font_name
        self.font_size = font_size
        self.font_color = font_color
        self.styles = getSampleStyleSheet()
        
        # Create markdown-specific styles
        self._create_markdown_styles()
    
    def _create_markdown_styles(self):
        """Create styles for different markdown elements"""
        # Header styles
        for i in range(1, 7):
            style_name = f"Heading{i}"
            if style_name not in self.styles:
                self.styles.add(ParagraphStyle(
                    name=style_name,
                    parent=self.styles["Heading1"],
                    fontName=self.font_name,
                    fontSize=self.font_size * (2 - i * 0.2),  # H1=2x, H6=1x
                    textColor=self.font_color,
                    spaceBefore=self.font_size * (1.5 - i * 0.2),
                    spaceAfter=self.font_size * (0.5 - i * 0.05),
                    bold=True
                ))
        
        # Code style
        if "Code" not in self.styles:
            self.styles.add(ParagraphStyle(
                name="Code",
                parent=self.styles["Normal"],
                fontName="Courier",
                fontSize=self.font_size * 0.9,
                textColor=self.font_color,
                backColor="#f5f5f5",
                borderPadding=4
            ))
        
        # Blockquote style
        if "Blockquote" not in self.styles:
            self.styles.add(ParagraphStyle(
                name="Blockquote",
                parent=self.styles["Normal"],
                fontName=self.font_name,
                fontSize=self.font_size,
                textColor="#666666",
                leftIndent=20,
                borderWidth=0,
                borderLeftWidth=3,
                borderColor="#cccccc",
                spaceBefore=6,
                spaceAfter=6
            ))
    
    def parse(self, text):
        """
        Parse markdown text and return list of ReportLab elements
        
        Args:
            text: Markdown text to parse
            
        Returns:
            list: List of ReportLab flowable elements
        """
        elements = []
        lines = text.split('\n')
        current_paragraph = []
        in_code_block = False
        code_block_content = []
        
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            
            # Handle code blocks
            if line.strip().startswith('```'):
                if not in_code_block:
                    # Start code block
                    in_code_block = True
                    code_block_content = []
                else:
                    # End code block
                    in_code_block = False
                    code_text = '\n'.join(code_block_content)
                    if code_text.strip():
                        elements.append(Paragraph(
                            f'<font name="Courier">{escape(code_text)}</font>',
                            self.styles["Code"]
                        ))
                i += 1
                continue
            
            if in_code_block:
                code_block_content.append(line)
                i += 1
                continue
            
            # Handle headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Flush current paragraph
                if current_paragraph:
                    elements.append(Paragraph(' '.join(current_paragraph), self.base_style))
                    current_paragraph = []
                
                level = len(header_match.group(1))
                content = self._parse_inline_markdown(header_match.group(2))
                elements.append(Paragraph(content, self.styles[f"Heading{level}"]))
                i += 1
                continue
            
            # Handle blockquotes
            if line.strip().startswith('>'):
                # Flush current paragraph
                if current_paragraph:
                    elements.append(Paragraph(' '.join(current_paragraph), self.base_style))
                    current_paragraph = []
                
                quote_content = line.strip()[1:].strip()
                quote_content = self._parse_inline_markdown(quote_content)
                elements.append(Paragraph(quote_content, self.styles["Blockquote"]))
                i += 1
                continue
            
            # Handle lists
            list_match = re.match(r'^(\s*)([*+-]|\d+\.)\s+(.+)$', line)
            if list_match:
                # Flush current paragraph
                if current_paragraph:
                    elements.append(Paragraph(' '.join(current_paragraph), self.base_style))
                    current_paragraph = []
                
                indent = len(list_match.group(1))
                marker = list_match.group(2)
                content = self._parse_inline_markdown(list_match.group(3))
                
                # Simple list item (could be enhanced for nested lists)
                list_text = f"{'&nbsp;' * indent * 4}{marker} {content}"
                elements.append(Paragraph(list_text, self.base_style))
                i += 1
                continue
            
            # Handle horizontal rules
            if re.match(r'^[-*_]{3,}$', line.strip()):
                # Flush current paragraph
                if current_paragraph:
                    elements.append(Paragraph(' '.join(current_paragraph), self.base_style))
                    current_paragraph = []
                
                # Add horizontal rule (using a line with border)
                elements.append(Paragraph('<hr/>', self.base_style))
                i += 1
                continue
            
            # Regular paragraph text
            if line.strip():
                parsed_line = self._parse_inline_markdown(line)
                current_paragraph.append(parsed_line)
            else:
                # Empty line - end current paragraph
                if current_paragraph:
                    elements.append(Paragraph(' '.join(current_paragraph), self.base_style))
                    current_paragraph = []
            
            i += 1
        
        # Flush final paragraph
        if current_paragraph:
            elements.append(Paragraph(' '.join(current_paragraph), self.base_style))
        
        return elements
    
    def _parse_inline_markdown(self, text):
        """
        Parse inline markdown elements (bold, italic, code, links)
        
        Args:
            text: Text to parse
            
        Returns:
            str: Text with ReportLab-compatible formatting
        """
        # Escape HTML first
        text = escape(text)
        
        # Code spans (inline code)
        text = re.sub(r'`([^`]+)`', r'<font name="Courier">\1</font>', text)
        
        # Bold text (**text** or __text__)
        text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'__([^_]+)__', r'<b>\1</b>', text)
        
        # Italic text (*text* or _text_)
        text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', text)
        text = re.sub(r'(?<!_)_([^_]+)_(?!_)', r'<i>\1</i>', text)
        
        # Links [text](url) - convert to basic text for now
        # Could be enhanced with actual links if needed
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        return text