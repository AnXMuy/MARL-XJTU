"""Export the formal Markdown report to standalone HTML and printable PDF."""
import base64
import html
import io
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = (ROOT / 'docs/技术报告.md').read_text()
    def equation(match):
        latex = match.group(1).strip().replace('\\operatorname{clip}', '\\mathrm{clip}')
        buffer = io.BytesIO()
        plt.figure(figsize=(12, 1))
        plt.text(.01, .5, '$' + latex + '$', fontsize=13)
        plt.axis('off')
        plt.savefig(buffer, format='svg', bbox_inches='tight', transparent=True)
        plt.close()
        encoded = base64.b64encode(buffer.getvalue()).decode()
        return '\n\n<div class="equation"><img alt="' + html.escape(latex, quote=True) + '" src="data:image/svg+xml;base64,' + encoded + '"></div>\n\n'
    source = re.sub(r'\\\[(.*?)\\\]', equation, source, flags=re.S)
    source = re.sub(r'\$\$(.*?)\$\$', equation, source, flags=re.S)
    body = MarkdownIt('commonmark', {'html': True}).enable('table').render(source)
    def embed(match):
        path = (ROOT / 'docs' / match.group(1)).resolve()
        if not path.is_file():
            return match.group(0)
        mime = 'image/png' if path.suffix == '.png' else 'image/svg+xml'
        return 'src="data:' + mime + ';base64,' + base64.b64encode(path.read_bytes()).decode() + '"'
    body = re.sub(r'src="(\.\./assets/[^\"]+)"', embed, body)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>栖居 MARL 技术报告</title><style>
body{font:14px/1.75 "Noto Sans CJK SC","Microsoft YaHei",sans-serif;color:#172329;max-width:1100px;margin:40px auto;padding:0 24px}
h1{font-size:30px}h2{margin-top:36px;border-bottom:1px solid #bbb}h3{margin-top:25px}table{border-collapse:collapse;width:100%;font-size:11px;margin:18px 0}th,td{border:1px solid #c8cdd0;padding:5px 7px}th{background:#edf3f3}tr{break-inside:avoid}thead{display:table-header-group}img{max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f5;padding:12px;font-size:11px}code{overflow-wrap:anywhere}a{color:#11675f}.equation{text-align:center;margin:18px 0}h1,h2,h3,h4{break-after:avoid}@page{size:A4;margin:16mm 13mm}@media print{body{margin:0;padding:0;font-size:10px}table{font-size:8px}h1{font-size:23px}h2{font-size:17px}h3{font-size:13px}pre{font-size:8px}img{max-height:220mm;object-fit:contain}}
</style><body>''' + body + '</body></html>'
    output = ROOT / 'docs/技术报告.html'
    output.write_text(page)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get('CHROMIUM_PATH'), args=['--no-sandbox'])
        tab = browser.new_page()
        tab.goto(output.as_uri(), wait_until='load')
        tab.pdf(path=str(ROOT / 'docs/技术报告.pdf'), print_background=True, display_header_footer=True,
                header_template='<span></span>', footer_template='<div style="font-size:8px;text-align:center;width:100%"><span class="pageNumber"></span> / <span class="totalPages"></span></div>')
        browser.close()
    print('Exported docs/技术报告.html and docs/技术报告.pdf')


if __name__ == '__main__':
    main()
