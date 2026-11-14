import unittest
from scripts.ctc_core import enumerate_paths, sum_transcript_probability
from scripts.ctc_forward import extended_target, forward_table, transcript_probability


class ForwardTests(unittest.TestCase):
    def test_forward_table(self):
        rows = [{'blank': .1, 'b': .8, 'a': .1}, {'blank': .2, 'b': .3, 'a': .5}, {'blank': .2, 'b': .1, 'a': .7}]
        table = forward_table(rows, 'ba')
        self.assertEqual(extended_target('ba'), ['blank', 'b', 'blank', 'a', 'blank'])
        for actual, expected in zip(table, [[.1,.8,0,0,0],[.02,.27,.16,.4,0],[.004,.029,.086,.581,.08]]):
            for a, e in zip(actual, expected):
                self.assertAlmostEqual(a, e, places=12)
        self.assertAlmostEqual(transcript_probability(table), .661, places=12)
        for target in ('', 'b', 'a', 'ba', 'bb', 'ab'):
            self.assertAlmostEqual(transcript_probability(forward_table(rows, target)), sum_transcript_probability(enumerate_paths(rows, ['blank','b','a']), target), places=12)

    def test_repeat_empty_and_errors(self):
        rows = [{'blank':.5,'b':.5}] * 3
        self.assertEqual(transcript_probability(forward_table(rows[:2], 'bb')), 0)
        self.assertAlmostEqual(transcript_probability(forward_table(rows, 'bb')), .125)
        for target in ('', 'b', 'bb', 'bbb'):
            self.assertAlmostEqual(transcript_probability(forward_table(rows, target)), sum_transcript_probability(enumerate_paths(rows, ['blank', 'b']), target), places=12)
        self.assertAlmostEqual(transcript_probability(forward_table(rows, '')), .125)
        self.assertEqual(transcript_probability([]), 0)
        with self.assertRaises(ValueError):
            forward_table(rows, 'a')
        with self.assertRaises(ValueError):
            forward_table([{'blank':.3,'b':.3}], 'b')
