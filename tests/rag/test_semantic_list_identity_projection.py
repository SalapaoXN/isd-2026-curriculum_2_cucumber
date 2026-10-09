"""Six collection projection controls; unrelated code extraction stays bounded."""

import re
from types import SimpleNamespace
import unittest

from rag.semantic.executor import (
    _claim_line, _codes, _collection_course_codes, _retained_from_claims,
)


A = ['06016405', '06016410', '06016412', '06016414', '06016415',
     '06016419', '06016420', '06016424', '06016425', '06066302']
B = ['06016404', '06016416', '06016417', '06016418', '06016421',
     '06016422', '06016423', '06016426', '06016427', '06066102']


class ListIdentityProjectionTests(unittest.TestCase):
    def project(self, codes):
        # Same shape as a grounded list claim: ordered canonical row mappings.
        value = tuple({'course_code': code, 'name_en': f'Title {index}',
                       'provenance': ({'source_page': 1},)}
                      for index, code in enumerate(codes))
        claim = SimpleNamespace(operation='list', status='complete', value=value)
        retained, _ = _retained_from_claims((claim,), 'IT', 'it-2565')
        return _claim_line('list', value), [row['course_code'] for row in retained]

    def test_case_a_all_ten(self):
        line, retained = self.project(A)
        self.assertEqual(re.findall(r'\d{8}', line), A)
        self.assertEqual(retained, A)

    def test_case_b_concrete_codes_not_placeholder(self):
        line, retained = self.project(B + ['90644xxx'])
        self.assertEqual(re.findall(r'\d{8}', line), B)
        self.assertEqual(retained, B)
        self.assertNotIn('90644xxx', retained)
        self.assertEqual(_collection_course_codes({'course_code': '90644xxx'}), [])

    def test_ninth_course_survives(self):
        line, retained = self.project(A[:9])
        self.assertEqual(re.findall(r'\d{8}', line), A[:9])
        self.assertEqual(retained, A[:9])

    def test_dedup_source_order_and_metadata(self):
        value = {'courses': [
            {'course_code': A[2], 'name_en': A[0], 'provenance': [{'reference': A[1]}]},
            {'course_code': A[2]}, {'course_code': A[0]}, A[1], A[0],
        ]}
        self.assertEqual(_collection_course_codes(value), [A[2], A[0], A[1]])
        self.assertEqual(
            re.findall(r'(?m)^- (\d{8})(?: —|$)', _claim_line('list', value)),
            [A[2], A[0], A[1]],
        )

    def test_retention_twenty_bound_and_explicit_display_subset(self):
        for count in (20, 21):
            codes = [f'{10000000 + index:08d}' for index in range(count)]
            line, retained = self.project(codes)
            self.assertEqual(retained, codes[:20])
            self.assertEqual(re.findall(r'\d{8}', line), codes[:20])
            if count > 20:
                self.assertIn(f'แสดง 20 จาก {count} รายวิชา', line)
            else:
                self.assertNotIn('แสดง', line)

    def test_non_list_identity_and_generic_limit_unchanged(self):
        value = {'course_code': '06016454', 'name_en': 'UX TOOLS'}
        self.assertEqual(_claim_line('identity', value), '06016454 UX TOOLS')
        self.assertEqual(_codes(A), A[:8])
