import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from knowledge_storm.paperstorm_pdf import (
    PaperStormPdfRenderer,
    PdfRenderError,
    markdown_to_print_html,
)


class PaperStormPdfTest(unittest.TestCase):
    def test_markdown_to_print_html_supports_chinese_tables_code_and_math(self):
        markdown = """# 无源互调抑制

正文包含行内公式 $y=x^2$ 和引用 [1]。

$$
E = mc^2
$$

| 方法 | 效果 |
| --- | --- |
| 神经网络 | 良好 |

```python
print("PIM")
```
"""

        html = markdown_to_print_html(markdown, title="无源互调抑制")

        self.assertIn('<html lang="zh-CN">', html)
        self.assertIn("<h1>无源互调抑制</h1>", html)
        self.assertIn("<table>", html)
        self.assertIn("<code", html)
        self.assertIn("<math", html)
        self.assertIn("@page", html)
        self.assertIn("page-break", html)
        self.assertIn("default-src &#x27;none&#x27;", html)

    def test_renderer_uses_browser_without_shell_and_registers_verified_pdf(self):
        calls = []

        def fake_runner(command, **kwargs):
            calls.append((command, kwargs))
            output_arg = next(
                item for item in command if item.startswith("--print-to-pdf=")
            )
            Path(output_arg.split("=", 1)[1]).write_bytes(b"%PDF-fake")
            return mock.Mock(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            markdown_path = Path(temp_dir) / "article.md"
            output_path = Path(temp_dir) / "paperstorm_report.pdf"
            markdown_path.write_text("# 中文报告\n\n$x^2$", encoding="utf-8")
            renderer = PaperStormPdfRenderer(
                browser_path=Path(temp_dir) / "msedge.exe",
                command_runner=fake_runner,
                pdf_validator=lambda path: {
                    "page_count": 2,
                    "text_length": 128,
                    "size_bytes": path.stat().st_size,
                },
            )

            result = renderer.render(
                markdown_path=markdown_path,
                output_pdf=output_path,
                title="中文报告",
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(result["page_count"], 2)
            self.assertTrue(Path(result["html_path"]).exists())
            command, kwargs = calls[0]
            self.assertIsInstance(command, list)
            self.assertFalse(kwargs.get("shell", False))
            self.assertTrue(
                any(item.startswith("--print-to-pdf=") for item in command)
            )

    def test_renderer_reports_missing_browser_with_typed_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            markdown_path = Path(temp_dir) / "article.md"
            markdown_path.write_text("# Report", encoding="utf-8")
            renderer = PaperStormPdfRenderer(browser_path="")

            with mock.patch(
                "knowledge_storm.paperstorm_pdf.discover_browser_executable",
                return_value=None,
            ):
                with self.assertRaises(PdfRenderError) as caught:
                    renderer.render(
                        markdown_path,
                        Path(temp_dir) / "paperstorm_report.pdf",
                    )

        self.assertEqual(caught.exception.code, "pdf_renderer_unavailable")

    def test_renderer_waits_for_browser_to_finish_writing_pdf(self):
        def delayed_runner(command, **_kwargs):
            output_arg = next(
                item for item in command if item.startswith("--print-to-pdf=")
            )
            output_path = Path(output_arg.split("=", 1)[1])

            def write_later():
                time.sleep(0.08)
                output_path.write_bytes(b"%PDF-delayed")

            threading.Thread(target=write_later, daemon=True).start()
            return mock.Mock(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            markdown_path = Path(temp_dir) / "article.md"
            output_path = Path(temp_dir) / "paperstorm_report.pdf"
            markdown_path.write_text("# Delayed PDF", encoding="utf-8")
            renderer = PaperStormPdfRenderer(
                browser_path=Path(temp_dir) / "msedge.exe",
                command_runner=delayed_runner,
                pdf_validator=lambda path: {
                    "page_count": 1,
                    "text_length": 11,
                    "size_bytes": path.stat().st_size,
                },
                output_wait_seconds=1,
            )

            result = renderer.render(markdown_path, output_path)

        self.assertEqual(result["page_count"], 1)

    def test_renderer_rejects_missing_or_empty_article(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            renderer = PaperStormPdfRenderer(browser_path="edge.exe")
            with self.assertRaises(PdfRenderError) as missing:
                renderer.render(
                    Path(temp_dir) / "missing.md",
                    Path(temp_dir) / "report.pdf",
                )
            empty = Path(temp_dir) / "empty.md"
            empty.write_text("   ", encoding="utf-8")
            with self.assertRaises(PdfRenderError) as blank:
                renderer.render(empty, Path(temp_dir) / "report.pdf")

        self.assertEqual(missing.exception.code, "pdf_source_missing")
        self.assertEqual(blank.exception.code, "pdf_source_empty")


if __name__ == "__main__":
    unittest.main()
