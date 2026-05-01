from __future__ import annotations

import html
import re
import subprocess
import sys
import tempfile
from io import StringIO
from pathlib import Path

import markdown
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPORT_MD = ROOT / "final out" / "master_report.md"
REPORT_DOCX = ROOT / "final out" / "master_report.docx"


def path_to_uri(path_str: str) -> str:
    return Path(path_str).resolve().as_uri()


def normalize_parsed_table(df: pd.DataFrame) -> pd.DataFrame:
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed_cols and "model_id" in df.columns:
        other_cols = [col for col in df.columns if col != "model_id"]
        merged = (
            df[other_cols]
            .fillna("")
            .astype(str)
            .apply(lambda row: " ".join(part.strip() for part in row if part.strip()), axis=1)
        )
        terminal_name = "best_params_json"
        named_tail = [col for col in other_cols if not str(col).startswith("Unnamed:")]
        if named_tail:
            terminal_name = str(named_tail[-1])
        df = pd.DataFrame({"model_id": df["model_id"].astype(str), terminal_name: merged})
    return df


def try_code_block_table(block_text: str) -> str | None:
    try:
        df = pd.read_fwf(StringIO(block_text))
    except Exception:
        return None

    if df.empty or df.shape[1] < 2:
        return None

    df = normalize_parsed_table(df)
    return (
        '<div class="table-wrap">'
        + df.to_html(index=False, border=0, classes=["report-table"], justify="left")
        + "</div>"
    )


def preprocess_markdown(markdown_text: str) -> str:
    out_lines: list[str] = []
    in_code_block = False
    code_lines: list[str] = []
    code_fence = ""

    for line in markdown_text.splitlines():
        if line.startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_lines = []
                code_fence = line[3:].strip()
            else:
                block_text = "\n".join(code_lines)
                table_html = try_code_block_table(block_text)
                if table_html is not None:
                    out_lines.append(table_html)
                else:
                    escaped = html.escape(block_text)
                    out_lines.append(f"<pre><code>{escaped}</code></pre>")
                in_code_block = False
                code_lines = []
                code_fence = ""
            continue

        if in_code_block:
            code_lines.append(line)
        else:
            out_lines.append(line)

    if in_code_block and code_lines:
        escaped = html.escape("\n".join(code_lines))
        out_lines.append(f"<pre><code>{escaped}</code></pre>")

    return "\n".join(out_lines)


def markdown_to_html(markdown_text: str) -> str:
    markdown_text = preprocess_markdown(markdown_text)
    html_body = markdown.markdown(
        markdown_text,
        extensions=["fenced_code", "tables", "sane_lists"],
        output_format="html5",
    )
    html_body = re.sub(
        r'src="([A-Za-z]:/[^"]+)"',
        lambda match: f'src="{path_to_uri(match.group(1))}"',
        html_body,
    )
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Final Teammate Report</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      font-size: 11pt;
      line-height: 1.35;
      margin: 24px;
    }}
    h1, h2, h3, h4, h5, h6 {{
      font-family: Arial, sans-serif;
    }}
    pre, code {{
      font-family: Consolas, "Courier New", monospace;
      white-space: pre-wrap;
      word-wrap: break-word;
    }}
    pre {{
      border: 1px solid #cccccc;
      padding: 8px;
      background: #f7f7f7;
    }}
    img {{
      max-width: 100%;
      height: auto;
      margin: 8px 0 16px 0;
    }}
    table {{
      border-collapse: collapse;
      margin: 8px 0 16px 0;
      width: 100%;
      font-size: 8pt;
    }}
    th, td {{
      border: 1px solid #999999;
      padding: 6px 8px;
      text-align: left;
      vertical-align: top;
      word-break: break-word;
    }}
    .table-wrap {{
      margin: 8px 0 16px 0;
    }}
  </style>
</head>
<body>
{html_body}
</body>
</html>
"""


def word_save_html_as_docx(html_path: Path, docx_path: Path) -> None:
    powershell = f"""
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {{
    $doc = $word.Documents.Open("{html_path}")
    $doc.SaveAs([ref] "{docx_path}", [ref] 16)
    $doc.Close()
}} finally {{
    $word.Quit()
}}
"""
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", powershell],
        check=True,
        cwd=ROOT,
    )


def main() -> int:
    if not REPORT_MD.exists():
        print(f"Missing report: {REPORT_MD}", file=sys.stderr)
        return 1

    markdown_text = REPORT_MD.read_text(encoding="utf-8")
    html_text = markdown_to_html(markdown_text)

    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = Path(tmpdir) / "master_report_export.html"
        html_path.write_text(html_text, encoding="utf-8")
        word_save_html_as_docx(html_path, REPORT_DOCX)

    print(REPORT_DOCX)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
