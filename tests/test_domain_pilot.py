import json
import tempfile
import unittest
from pathlib import Path


class DomainPilotDatasetTests(unittest.TestCase):
    def test_loader_requires_exact_count_unique_questions_and_grounded_quotes(self):
        from knowledge_storm.evaluation.domain_pilot import load_domain_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "corpus.jsonl"
            cases = root / "cases.jsonl"
            corpus.write_text(
                json.dumps(
                    {
                        "chunk_id": "paper-a::p1::c1",
                        "document_id": "paper-a",
                        "title": "PIM Paper",
                        "content": "无源互调由无源器件的非线性接触产生。",
                        "metadata": {"page_number": 1},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            cases.write_text(
                json.dumps(
                    {
                        "case_id": "pim-001",
                        "question": "无源互调的主要成因是什么？",
                        "reference_answer": "无源器件中的非线性接触。",
                        "evidence_chunk_ids": ["paper-a::p1::c1"],
                        "evidence_quote": "无源互调由无源器件的非线性接触产生",
                        "category": "mechanism",
                        "difficulty": "basic",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            dataset = load_domain_dataset(corpus, cases, expected_case_count=1)

        self.assertEqual("paperstorm-pim-domain-pilot", dataset.name)
        self.assertEqual(1, len(dataset.documents))
        self.assertEqual(1, len(dataset.cases))
        self.assertEqual(
            "无源器件中的非线性接触。", dataset.cases[0].answers[0]
        )
        self.assertEqual("mechanism", dataset.cases[0].metadata["category"])

    def test_loader_rejects_missing_evidence_and_ungrounded_quote(self):
        from knowledge_storm.evaluation.domain_pilot import validate_domain_rows

        corpus = [
            {
                "chunk_id": "chunk-1",
                "document_id": "paper",
                "title": "Paper",
                "content": "真实证据内容",
            }
        ]
        bad_missing = [
            {
                "case_id": "q1",
                "question": "问题？",
                "reference_answer": "答案",
                "evidence_chunk_ids": ["missing"],
                "evidence_quote": "真实证据内容",
                "category": "method",
            }
        ]
        bad_quote = [dict(bad_missing[0], evidence_chunk_ids=["chunk-1"], evidence_quote="幻觉引文")]

        with self.assertRaisesRegex(ValueError, "missing evidence chunk"):
            validate_domain_rows(corpus, bad_missing, expected_case_count=1)
        with self.assertRaisesRegex(ValueError, "evidence quote is not grounded"):
            validate_domain_rows(corpus, bad_quote, expected_case_count=1)

    def test_loader_rejects_duplicate_questions_and_missing_category_coverage(self):
        from knowledge_storm.evaluation.domain_pilot import validate_domain_rows

        corpus = [
            {
                "chunk_id": "chunk-1",
                "document_id": "paper",
                "title": "Paper",
                "content": "证据内容",
            }
        ]
        duplicate = [
            {
                "case_id": "q1",
                "question": "同一个问题？",
                "reference_answer": "答案",
                "evidence_chunk_ids": ["chunk-1"],
                "evidence_quote": "证据内容",
                "category": "method",
            },
            {
                "case_id": "q2",
                "question": "同一个问题？",
                "reference_answer": "答案",
                "evidence_chunk_ids": ["chunk-1"],
                "evidence_quote": "证据内容",
                "category": "method",
            },
        ]

        with self.assertRaisesRegex(ValueError, "duplicate question"):
            validate_domain_rows(corpus, duplicate, expected_case_count=2)
        with self.assertRaisesRegex(ValueError, "missing required categories"):
            validate_domain_rows(
                corpus,
                duplicate[:1],
                expected_case_count=1,
                required_categories=("method", "limitation"),
            )


class DomainPilotPreparationTests(unittest.TestCase):
    def test_generation_prompt_preserves_authoritative_evidence_fields(self):
        from examples.storm_examples.prepare_pim_domain_pilot import (
            build_generation_prompt,
        )

        prompt = build_generation_prompt(
            [
                {
                    "slot_id": "slot-001",
                    "category": "mechanism",
                    "chunk_id": "paper::p2::c1",
                    "source_title": "PIM Paper",
                    "page_number": 2,
                    "excerpt": "无源互调由接触非线性产生。",
                }
            ]
        )

        self.assertIn("slot-001", prompt)
        self.assertIn("mechanism", prompt)
        self.assertIn("无源互调由接触非线性产生", prompt)
        self.assertIn("evidence_quote", prompt)

    def test_generation_chunk_selection_is_deterministic_and_source_balanced(self):
        from examples.storm_examples.prepare_pim_domain_pilot import (
            select_generation_chunks,
        )

        rows = []
        for document_index in range(5):
            for chunk_index in range(15):
                rows.append(
                    {
                        "chunk_id": "doc-{0}::chunk-{1}".format(
                            document_index, chunk_index
                        ),
                        "document_id": "doc-{0}".format(document_index),
                        "title": "paper {0}".format(document_index),
                        "content": (
                            "无源互调 PIM 非线性 接触 波束赋形 抑制方法 实验结果 "
                            + "有效证据" * 40
                        ),
                        "metadata": {"page_number": chunk_index + 1},
                    }
                )

        first = select_generation_chunks(rows, count=50, seed=55)
        second = select_generation_chunks(list(reversed(rows)), count=50, seed=55)

        self.assertEqual(
            [row["chunk_id"] for row in first],
            [row["chunk_id"] for row in second],
        )
        self.assertEqual(50, len(first))
        counts = {}
        for row in first:
            counts[row["document_id"]] = counts.get(row["document_id"], 0) + 1
        self.assertEqual({"doc-{0}".format(index): 10 for index in range(5)}, counts)

    def test_hermes_json_parser_accepts_fence_and_rejects_non_array(self):
        from examples.storm_examples.prepare_pim_domain_pilot import (
            parse_hermes_json_array,
        )

        payload = parse_hermes_json_array(
            '```json\n[{"slot_id":"slot-1","question":"问题？"}]\n```'
        )

        self.assertEqual("slot-1", payload[0]["slot_id"])
        with self.assertRaisesRegex(ValueError, "JSON array"):
            parse_hermes_json_array('{"slot_id":"slot-1"}')

    def test_case_binding_rejects_duplicate_generated_slots(self):
        from examples.storm_examples.prepare_pim_domain_pilot import (
            bind_generated_cases,
        )

        slots = [
            {
                "slot_id": "slot-001",
                "case_id": "pim-001",
                "category": "mechanism",
                "chunk_id": "chunk-1",
                "source_title": "paper",
                "excerpt": "无源互调由接触非线性产生。",
            }
        ]
        generated = [
            {
                "slot_id": "slot-001",
                "question": "问题一？",
                "reference_answer": "答案",
                "evidence_quote": "无源互调由接触非线性产生",
            },
            {
                "slot_id": "slot-001",
                "question": "问题二？",
                "reference_answer": "答案",
                "evidence_quote": "无源互调由接触非线性产生",
            },
        ]

        with self.assertRaisesRegex(ValueError, "duplicate generated slot"):
            bind_generated_cases(slots, generated)

    def test_case_binding_repairs_near_exact_quote_to_verbatim_source(self):
        from examples.storm_examples.prepare_pim_domain_pilot import (
            bind_generated_cases,
        )

        slots = [
            {
                "slot_id": "slot-001",
                "case_id": "pim-001",
                "category": "mechanism",
                "chunk_id": "chunk-1",
                "source_title": "paper",
                "excerpt": "它不但保留了小波分析的优点，而且还继承了神经网络处理复杂非线性问题能力的优势。",
            }
        ]
        generated = [
            {
                "slot_id": "slot-001",
                "question": "小波神经网络有什么优势？",
                "reference_answer": "兼具小波分析和神经网络的优势。",
                "evidence_quote": "它还继承了神经网络处理复杂非线性问题能力的优势",
            }
        ]

        cases = bind_generated_cases(slots, generated)

        self.assertIn(cases[0]["evidence_quote"], slots[0]["excerpt"])
        self.assertNotEqual(generated[0]["evidence_quote"], cases[0]["evidence_quote"])

    def test_case_binding_rejects_quote_outside_documented_length(self):
        from examples.storm_examples.prepare_pim_domain_pilot import bind_generated_cases

        slots = [{"slot_id": "slot-001", "case_id": "pim-001", "category": "method", "chunk_id": "c1", "source_title": "p", "excerpt": "短引用不能成为可靠证据。"}]
        generated = [{"slot_id": "slot-001", "question": "问题？", "reference_answer": "答案", "evidence_quote": "短引用"}]

        with self.assertRaisesRegex(ValueError, "quote length"):
            bind_generated_cases(slots, generated)


class DomainPilotRunnerTests(unittest.TestCase):
    def test_best_profile_prefers_recall_then_latency(self):
        from examples.storm_examples.run_pim_domain_pilot import select_best_profile

        reports = {
            "legacy": {"metrics": {"recall_at_5": 0.7, "query_p95_ms": 20}},
            "bge": {"metrics": {"recall_at_5": 0.8, "query_p95_ms": 30}},
            "gte": {"metrics": {"recall_at_5": 0.8, "query_p95_ms": 25}},
        }

        self.assertEqual("gte", select_best_profile(reports, top_k=5))

    def test_answer_prompt_contains_retrieved_evidence_without_reference_answer(self):
        from examples.storm_examples.answer_pim_domain_pilot import build_answer_prompt

        prompt = build_answer_prompt(
            [
                {
                    "case_id": "pim-001",
                    "question": "无源互调的成因是什么？",
                    "reference_answer": "不应泄漏的标准答案",
                    "contexts": [
                        {
                            "chunk_id": "chunk-1",
                            "title": "paper",
                            "content": "证据原文",
                        }
                    ],
                }
            ]
        )

        self.assertIn("chunk-1", prompt)
        self.assertIn("证据原文", prompt)
        self.assertNotIn("不应泄漏的标准答案", prompt)

    def test_answer_f1_uses_multilingual_tokens(self):
        from examples.storm_examples.answer_pim_domain_pilot import answer_f1

        self.assertEqual(1.0, answer_f1("无源互调 PIM", "无源互调 PIM"))
        self.assertGreater(answer_f1("无源互调由非线性产生", "非线性导致无源互调"), 0.0)

    def test_answer_preparation_reports_stale_case_and_chunk_ids(self):
        from examples.storm_examples.answer_pim_domain_pilot import prepare_answer_cases

        corpus = [{"chunk_id": "chunk-1", "content": "evidence", "title": "paper"}]
        cases = [{"case_id": "case-1", "question": "q", "reference_answer": "a", "evidence_chunk_ids": ["chunk-1"]}]

        with self.assertRaisesRegex(ValueError, "unknown prediction case_id"):
            prepare_answer_cases(corpus, cases, [{"case_id": "stale", "ranked_document_ids": []}])
        with self.assertRaisesRegex(ValueError, "unknown retrieved chunk_id"):
            prepare_answer_cases(corpus, cases, [{"case_id": "case-1", "ranked_document_ids": ["stale"]}])


if __name__ == "__main__":
    unittest.main()
