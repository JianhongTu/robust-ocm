# -*- coding: utf-8 -*-
"""
Text normalization and preprocessing for OCR evaluation.

This module provides functions to normalize text, formulas, and tables
following the OmniDocBench preprocessing steps.
"""

import re
import unicodedata
from bs4 import BeautifulSoup
from pylatexenc.latex2text import LatexNodes2Text


def remove_markdown_fences(content: str) -> str:
    """Remove markdown code fences from content."""
    content = re.sub(r"^```markdown\n?", "", content, flags=re.MULTILINE)
    content = re.sub(r"^```html\n?", "", content, flags=re.MULTILINE)
    content = re.sub(r"^```latex\n?", "", content, flags=re.MULTILINE)
    content = re.sub(r"```\n?$", "", content, flags=re.MULTILINE)
    return content


def replace_repeated_chars(input_str: str) -> str:
    """Standardize consecutive characters."""
    input_str = re.sub(r"_{4,}", "____", input_str)
    input_str = re.sub(r" {4,}", "    ", input_str)
    return input_str


def fullwidth_to_halfwidth(s: str) -> str:
    """Convert full-width characters to half-width."""
    result = []
    for char in s:
        code = ord(char)
        if code == 0x3000:
            code = 0x0020
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        result.append(chr(code))
    return "".join(result)


def textblock_to_unicode(text: str) -> str:
    """Convert inline LaTeX formulas to Unicode text."""
    inline_reg = re.compile(r"\$(.*?)\$|\\\((.*?)\\\)", re.DOTALL)
    inline_matches = inline_reg.finditer(text)
    removal_positions = []

    for match in inline_matches:
        position = [match.start(), match.end()]
        content = match.group(1) if match.group(1) is not None else match.group(2)
        clean_content = re.sub(r"\\([\\_&%^])", "", content)

        try:
            if any(char in clean_content for char in r"\^_"):
                if clean_content.endswith("\\"):
                    clean_content += " "
                unicode_content = LatexNodes2Text().latex_to_text(clean_content)
                removal_positions.append((position[0], position[1], unicode_content))
        except Exception:
            continue

    for start, end, unicode_content in sorted(removal_positions, reverse=True):
        text = text[:start] + unicode_content.strip() + text[end:]

    return text


def normalize_formula(text: str) -> str:
    """Normalize math formulas for matching.

    Removes various LaTeX commands and delimiters, then converts to lowercase.
    """
    filter_list = [
        "\\mathbf",
        "\\mathrm",
        "\\mathnormal",
        "\\mathit",
        "\\mathbb",
        "\\mathcal",
        "\\mathscr",
        "\\mathfrak",
        "\\mathsf",
        "\\mathtt",
        "\\textbf",
        "\\text",
        "\\boldmath",
        "\\boldsymbol",
        "\\operatorname",
        "\\bm",
        "\\symbfit",
        "\\mathbfcal",
        "\\symbf",
        "\\scriptscriptstyle",
        "\\notag",
        "\\setlength",
        "\\coloneqq",
        "\\space",
        "\\thickspace",
        "\\thinspace",
        "\\medspace",
        "\\nobreakspace",
        "\\negmedspace",
        "\\quad",
        "\\qquad",
        "\\enspace",
        "\\substackw",
        " ",
        "$$",
        "\\left",
        "\\right",
        "\\displaystyle",
        "\\text",
    ]

    text = text.strip().strip("$").strip("\n")
    pattern = re.compile(r"\\\[(.+?)(?<!\\)\\\]")
    match = pattern.search(text)

    if match:
        text = match.group(1).strip()

    tag_pattern = re.compile(r"\\tag\{.*?\}")
    text = tag_pattern.sub("", text)
    hspace_pattern = re.compile(r"\\hspace\{.*?\}")
    text = hspace_pattern.sub("", text)
    begin_pattern = re.compile(r"\\begin\{.*?\}")
    text = begin_pattern.sub("", text)
    end_pattern = re.compile(r"\\end\{.*?\}")
    text = end_pattern.sub("", text)
    col_sep = re.compile(r"\\arraycolsep.*?\}")
    text = col_sep.sub("", text)
    text = text.strip(".")

    for filter_text in filter_list:
        text = text.replace(filter_text, "")

    text = text.lower()
    return text


def normalize_html_table(text: str) -> str:
    """Normalize HTML tables for evaluation."""
    def process_table_html(html_content: str) -> str:
        soup = BeautifulSoup(html_content, "html.parser")
        th_tags = soup.find_all("th")
        for th in th_tags:
            th.name = "td"
        thead_tags = soup.find_all("thead")
        for thead in thead_tags:
            thead.unwrap()
        math_tags = soup.find_all("math")
        for math_tag in math_tags:
            alttext = math_tag.get("alttext", "")
            alttext = f"${alttext}$"
            if alttext:
                math_tag.replace_with(alttext)
        span_tags = soup.find_all("span")
        for span in span_tags:
            span.unwrap()
        return str(soup)

    table_res = ""
    if "<table" in text.replace(" ", "").replace("'", '"'):
        text = process_table_html(text)
        table_res = text.replace("\n", "")
        table_res = unicodedata.normalize("NFKC", table_res).strip()
        pattern = r"<table\b[^>]*>(.*)</table>"
        tables = re.findall(pattern, table_res, re.DOTALL | re.IGNORECASE)
        table_res = "".join(tables)
        table_res = re.sub(r'( style=".*?")', "", table_res)
        table_res = re.sub(r'( height=".*?")', "", table_res)
        table_res = re.sub(r'( width=".*?")', "", table_res)
        table_res = re.sub(r'( align=".*?")', "", table_res)
        table_res = re.sub(r'( class=".*?")', "", table_res)
        table_res = re.sub(r"</?tbody>", "", table_res)
        table_res = re.sub(r"\s+", " ", table_res)
        table_res = '<html><body><table border="1" >' + table_res + "</table></body></html>"

    return table_res


def normalize_latex_table(text: str) -> str:
    """Normalize LaTeX tables for evaluation."""
    special_strings = [
        [r"\\vspace\{.*?\}", ""],
        [r"\\hspace\{.*?\}", ""],
        [r"\\rule\{.*?\}\{.*?\}", ""],
        [r"\\addlinespace\[.*?\]", ""],
        [r"\\addlinespace", ""],
        [r"\\renewcommand\{\\arraystretch\}\{.*?\}", ""],
        [r"\\arraystretch\{.*?\}", ""],
        [r"(row|column)?colors?\{[^}]*\}(\{[^}]*\}){0,2}", ""],
        [r"\\color\{.*?\}", ""],
        [r"\\textcolor\{.*?\}", ""],
        [r"\\rowcolor(\[.*?\])?\{.*?\}", ""],
        [r"\\columncolor(\[.*?\])?\{.*?\}", ""],
        [r"\\cellcolor(\[.*?\])?\{.*?\}", ""],
        [r"\\colorbox\{.*?\}", ""],
        [r"\\(tiny|scriptsize|footnotesize|small|normalsize|large|Large|LARGE|huge|Huge)", ""],
    ]

    for pattern, replacement in special_strings:
        text = re.sub(pattern, replacement, text)

    return text


def clean_string(text: str) -> str:
    """Clean and normalize text string."""
    text = text.strip()
    text = fullwidth_to_halfwidth(text)
    text = replace_repeated_chars(text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_text(text: str) -> str:
    """Normalize text for evaluation."""
    text = remove_markdown_fences(text)
    text = textblock_to_unicode(text)
    text = clean_string(text)
    return text