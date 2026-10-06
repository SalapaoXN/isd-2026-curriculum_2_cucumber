import unittest

from rag.retrieval.retrieve import lexical_topic_match


class CompoundTopicMatchingTests(unittest.TestCase):
    def test_split_query_matches_exact_compound_token(self):
        self.assertTrue(lexical_topic_match(
            "cyber security", {"name_en": "CYBERSECURITY FUNDAMENTALS"}
        ))

    def test_compound_query_matches_contiguous_split_tokens(self):
        self.assertTrue(lexical_topic_match(
            "cybersecurity", {"name_en": "CYBER SECURITY"}
        ))

    def test_partial_tokens_do_not_match(self):
        for topic, title in (
            ("security", "CYBERSECURITY"),
            ("data", "DATABASE SYSTEMS"),
            ("web", "WEBSITE DESIGN"),
            ("data base", "DATABASES"),
        ):
            with self.subTest(topic=topic, title=title):
                self.assertFalse(lexical_topic_match(topic, {"name_en": title}))

    def test_existing_phrase_matching_remains(self):
        self.assertTrue(lexical_topic_match(
            "distributed systems", {"name_en": "ADVANCED DISTRIBUTED SYSTEMS"}
        ))
        self.assertFalse(lexical_topic_match(
            "distributed systems", {"name_en": "DISTRIBUTED DATABASE SYSTEMS"}
        ))

    def test_thai_matching_is_not_compacted(self):
        self.assertTrue(lexical_topic_match("ฐานข้อมูล", {"name_th": "ระบบฐานข้อมูล"}))
        self.assertFalse(lexical_topic_match("ฐาน ข้อมูล", {"name_th": "ระบบฐานข้อมูล"}))


if __name__ == "__main__":
    unittest.main()
