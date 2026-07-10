#!/usr/bin/env python3
"""Render a Markdown file to PDF (markdown -> HTML -> WeasyPrint).

Usage: md2pdf.py INPUT.md [OUTPUT.pdf]
"""
import sys
from pathlib import Path

import markdown
from weasyprint import HTML

CSS = """
@page { size: A4; margin: 2.2cm 2cm; @bottom-center { content: counter(page) " / " counter(pages); font-size: 8pt; color: #888; } }
body { font-family: "DejaVu Sans", sans-serif; font-size: 9.5pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 19pt; border-bottom: 2px solid #2c5f8a; padding-bottom: 4px; color: #2c5f8a; }
h2 { font-size: 14pt; color: #2c5f8a; border-bottom: 1px solid #ccd; padding-bottom: 2px; margin-top: 1.6em; }
h3 { font-size: 11.5pt; color: #333; margin-top: 1.3em; }
h4 { font-size: 10pt; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.3pt; background: #f4f4f6; padding: 0 2px; border-radius: 2px; }
pre { background: #f4f4f6; border: 1px solid #ddd; border-radius: 4px; padding: 7px 9px; font-size: 8pt; white-space: pre-wrap; word-wrap: break-word; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 0.7em 0; font-size: 8.5pt; }
th, td { border: 1px solid #bbb; padding: 3px 6px; text-align: left; vertical-align: top; }
th { background: #e8eef4; }
blockquote { border-left: 3px solid #2c5f8a; margin-left: 0; padding-left: 10px; color: #444; }
a { color: #2c5f8a; text-decoration: none; }
li { margin: 0.15em 0; }
"""


def main() -> None:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".pdf")
    html_body = markdown.markdown(
        src.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
    )
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_body}</body></html>"
    HTML(string=html).write_pdf(str(dst))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
