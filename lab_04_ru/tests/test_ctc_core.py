import unittest
from scripts.ctc_core import collapse_path, enumerate_paths, sum_transcript_probability


class CoreTests(unittest.TestCase):
    def test_collapse(self):
        self.assertEqual(collapse_path(['b', 'b']), 'b')
        self.assertEqual(collapse_path(['b', 'blank', 'b']), 'bb')
        self.assertEqual(collapse_path([]), '')
        self.assertEqual(collapse_path(['blank', 'b', 'b', 'blank', 'a', 'a']), 'ba')

    def test_paths(self):
        rows = [{'blank': .1, 'b': .8, 'a': .1}, {'blank': .2, 'b': .3, 'a': .5}, {'blank': .2, 'b': .1, 'a': .7}]
        records = enumerate_paths(rows, ['blank', 'b', 'a'])
        self.assertEqual(len(records), 27)
        self.assertEqual(sum(r['transcript'] == 'ba' for r in records), 5)
        self.assertAlmostEqual(sum_transcript_probability(records, 'ba'), .661, places=12)
        self.assertAlmostEqual(sum(r['probability'] for r in records), 1)
        self.assertEqual(enumerate_paths([], ['blank']), [{'path': [], 'transcript': '', 'probability': 1.0}])

    def test_zero_and_invalid_values(self):
        records = enumerate_paths([{'blank': 1.0, 'a': 0.0}], ['blank', 'a'])
        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]['probability'], 0.0)
        for bad in (float('nan'), float('inf'), -0.1, 1.1, '0.5', 10**1000, -(10**1000)):
            with self.assertRaises(ValueError):
                enumerate_paths([{'blank': bad, 'a': 0.5}], ['blank', 'a'])
        with self.assertRaises(ValueError):
            enumerate_paths([], ['blank', 'blank'])
        with self.assertRaises(ValueError):
            collapse_path(['unknown'])

    def test_invalid(self):
        with self.assertRaises(ValueError):
            enumerate_paths([{'blank': 1}], ['blank', 'a'])
        with self.assertRaises(ValueError):
            enumerate_paths([{'blank': .5, 'a': .5}] * 14, ['blank', 'a'])
