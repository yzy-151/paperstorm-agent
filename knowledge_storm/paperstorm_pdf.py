import html
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path


class PdfRenderError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def markdown_to_print_html(markdown_text: str, title: str = "PaperStorm 调研报告"):
    try:
        import markdown
        from latex2mathml.converter import convert as latex_to_mathml
    except ImportError as error:
        raise PdfRenderError(
            "pdf_dependency_missing",
            "生成 PDF 需要 Markdown 和 latex2mathml 依赖。",
        ) from error

    text_with_math = _replace_latex_math(markdown_text, latex_to_mathml)
    body = markdown.markdown(
        text_with_math,
        extensions=["fenced_code", "tables", "sane_lists"],
        output_format="html5",
    )
    escaped_title = html.escape(str(title or "PaperStorm 调研报告"))
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src &#x27;none&#x27;; style-src &#x27;unsafe-inline&#x27;; img-src data:">
  <title>{title}</title>
  <style>
    @page {{
      size: A4;
      margin: 18mm 17mm 20mm;
      @bottom-center {{
        content: "PaperStorm · " counter(page) " / " counter(pages);
        color: #64748b;
        font-size: 9pt;
      }}
    }}
    * {{ box-sizing: border-box; }}
    html {{ color: #172033; background: #fff; }}
    body {{
      margin: 0 auto;
      max-width: 176mm;
      font-family: "Microsoft YaHei", "Noto Sans CJK SC", "Segoe UI", sans-serif;
      font-size: 11pt;
      line-height: 1.72;
      letter-spacing: 0;
    }}
    h1, h2, h3, h4 {{
      color: #0f172a;
      page-break-after: avoid;
      break-after: avoid-page;
      margin: 1.2em 0 .55em;
      line-height: 1.3;
    }}
    h1 {{ font-size: 25pt; border-bottom: 2px solid #0f766e; padding-bottom: 7mm; }}
    h2 {{ font-size: 17pt; border-left: 3px solid #0f766e; padding-left: 3mm; }}
    h3 {{ font-size: 13pt; }}
    p, li {{ orphans: 3; widows: 3; }}
    a {{ color: #0f766e; text-decoration: none; overflow-wrap: anywhere; }}
    blockquote {{
      margin: 1em 0;
      padding: .25em 1em;
      border-left: 3px solid #94a3b8;
      color: #475569;
      background: #f8fafc;
    }}
    table {{ width: 100%; border-collapse: collapse; margin: 1em 0; page-break-inside: avoid; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 7px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #f1f5f9; }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #0f172a;
      color: #e2e8f0;
      padding: 12px;
      page-break-inside: avoid;
      border-radius: 3px;
    }}
    code {{ font-family: "Cascadia Mono", Consolas, monospace; font-size: 9.5pt; }}
    math {{ max-width: 100%; overflow: hidden; }}
    .math-block {{ display: block; text-align: center; margin: 1em 0; page-break-inside: avoid; }}
    hr {{ border: 0; border-top: 1px solid #cbd5e1; margin: 1.5em 0; }}
    img {{ max-width: 100%; page-break-inside: avoid; }}
  </style>
</head>
<body>
{body}
</body>
</html>
""".format(title=escaped_title, body=body)


def discover_browser_executable():
    candidates = [
        os.getenv("PAPERSTORM_PDF_BROWSER", ""),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def validate_pdf(path):
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise PdfRenderError(
            "pdf_dependency_missing", "验证 PDF 需要 pypdf 依赖。"
        ) from error
    pdf_path = Path(path)
    if not pdf_path.is_file() or pdf_path.stat().st_size <= 8:
        raise PdfRenderError("pdf_render_error", "浏览器未生成有效的 PDF 文件。")
    try:
        reader = PdfReader(str(pdf_path))
        text_length = sum(
            len((page.extract_text() or "").strip()) for page in reader.pages
        )
    except Exception as error:
        raise PdfRenderError("pdf_render_error", "生成的 PDF 无法读取。") from error
    if not reader.pages:
        raise PdfRenderError("pdf_render_error", "生成的 PDF 没有页面。")
    if text_length <= 0:
        raise PdfRenderError("pdf_render_error", "生成的 PDF 无法提取正文文字。")
    return {
        "page_count": len(reader.pages),
        "text_length": text_length,
        "size_bytes": pdf_path.stat().st_size,
    }


class PaperStormPdfRenderer:
    def __init__(
        self,
        browser_path=None,
        command_runner=None,
        pdf_validator=None,
        timeout_seconds=120,
        output_wait_seconds=10,
    ):
        self.browser_path = Path(browser_path) if browser_path else None
        self.command_runner = command_runner or subprocess.run
        self.pdf_validator = pdf_validator or validate_pdf
        self.timeout_seconds = max(10, int(timeout_seconds))
        self.output_wait_seconds = max(0.1, float(output_wait_seconds))

    def render(self, markdown_path, output_pdf, title="PaperStorm 调研报告"):
        source_path = Path(markdown_path)
        output_path = Path(output_pdf)
        if not source_path.is_file():
            raise PdfRenderError("pdf_source_missing", "没有找到可用于生成 PDF 的文章。")
        markdown_text = source_path.read_text(encoding="utf-8", errors="replace")
        if not markdown_text.strip():
            raise PdfRenderError("pdf_source_empty", "文章内容为空，无法生成 PDF。")

        browser = self.browser_path or discover_browser_executable()
        if not browser:
            raise PdfRenderError(
                "pdf_renderer_unavailable",
                "未找到 Microsoft Edge 或 Google Chrome，无法生成 PDF。",
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        html_path = output_path.with_suffix(".print.html")
        html_path.write_text(
            markdown_to_print_html(markdown_text, title=title),
            encoding="utf-8",
        )
        with tempfile.TemporaryDirectory(
            prefix="paperstorm-pdf-", ignore_cleanup_errors=True
        ) as profile_dir:
            command = [
                str(browser),
                "--headless=new",
                "--disable-gpu",
                "--disable-extensions",
                "--no-pdf-header-footer",
                "--user-data-dir={0}".format(profile_dir),
                "--print-to-pdf={0}".format(output_path.resolve()),
                html_path.resolve().as_uri(),
            ]
            try:
                completed = self.command_runner(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    shell=False,
                )
            except subprocess.TimeoutExpired as error:
                raise PdfRenderError(
                    "pdf_render_timeout", "浏览器生成 PDF 超时。"
                ) from error
            except OSError as error:
                raise PdfRenderError(
                    "pdf_renderer_unavailable", "无法启动 PDF 浏览器渲染器。"
                ) from error

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise PdfRenderError(
                "pdf_render_error",
                "浏览器生成 PDF 失败：{0}".format(detail[:400] or completed.returncode),
            )
        _wait_for_output_file(output_path, self.output_wait_seconds)
        metrics = self.pdf_validator(output_path)
        return {
            "path": str(output_path),
            "html_path": str(html_path),
            **metrics,
        }


def _replace_latex_math(markdown_text, converter):
    def block(match):
        formula = match.group(1).strip()
        try:
            return '<div class="math-block">{0}</div>'.format(
                converter(formula, display="block")
            )
        except Exception:
            return match.group(0)

    def inline(match):
        formula = match.group(1).strip()
        try:
            return converter(formula)
        except Exception:
            return match.group(0)

    converted = re.sub(r"\$\$(.+?)\$\$", block, markdown_text, flags=re.DOTALL)
    return re.sub(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", inline, converted)


def _wait_for_output_file(path, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    previous_size = -1
    stable_reads = 0
    while time.monotonic() < deadline:
        if path.is_file():
            size = path.stat().st_size
            if size > 8 and size == previous_size:
                stable_reads += 1
                if stable_reads >= 2:
                    return
            else:
                stable_reads = 0
            previous_size = size
        time.sleep(0.05)
