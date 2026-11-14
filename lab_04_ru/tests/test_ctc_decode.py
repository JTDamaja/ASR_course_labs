import unittest
from collections import defaultdict
from scripts.ctc_core import enumerate_paths
from scripts.ctc_decode import greedy_decode, prefix_beam_decode, prefix_transitions


class DecodeTests(unittest.TestCase):
    def test_fixed(self):
        rows = [{'blank':.1,'a':.6,'b':.3},{'blank':.45,'a':.2,'b':.35},{'blank':.45,'a':.2,'b':.35}]
        alphabet = ['blank','a','b']
        self.assertEqual(greedy_decode(rows, alphabet), (['a','blank','blank'], 'a'))
        best, ledger = prefix_beam_decode(rows, alphabet, 5)
        self.assertEqual(best, 'ab')
        self.assertAlmostEqual(dict(ledger)['ab'], .3115)
        self.assertAlmostEqual(dict(ledger)['a'], .2215)
        expected = defaultdict(float)
        for r in enumerate_paths(rows, alphabet):
            expected[r['transcript']] += r['probability']
        unpruned = dict(prefix_beam_decode(rows, alphabet, 100)[1])
        for key, value in expected.items():
            self.assertAlmostEqual(unpruned[key], value, places=12)

    def test_accumulate_then_prune_each_frame(self):
        alphabet = ['blank', 'a']
        rows = [{'blank': .5, 'a': .5}, {'blank': .5, 'a': .5}]
        best, ledger = prefix_beam_decode(rows, alphabet, 2)
        self.assertEqual(best, 'a')
        self.assertAlmostEqual(dict(ledger)['a'], .75)
        pruned_rows = [{'blank': .6, 'a': .4}, {'blank': .1, 'a': .9}]
        best, ledger = prefix_beam_decode(pruned_rows, alphabet, 1)
        self.assertEqual(best, 'a')
        self.assertAlmostEqual(ledger[0][1], .54)

    def test_zero_and_ties(self):
        rows = [{'blank': 1.0, 'a': 0.0}, {'blank': 0.0, 'a': 1.0}]
        self.assertEqual(greedy_decode(rows, ['blank', 'a']), (['blank', 'a'], 'a'))
        self.assertEqual(prefix_beam_decode(rows, ['blank', 'a'], 10)[0], 'a')
        tied = [{'blank': 0.0, 'a': .5, 'b': .5}]
        self.assertEqual([x[0] for x in prefix_beam_decode(tied, ['blank','a','b'], 2)[1]], ['a', 'b'])

    def test_edges(self):
        self.assertEqual(greedy_decode([], ['blank','a']), ([], ''))
        self.assertEqual(prefix_beam_decode([], ['blank','a'], 1), ('', [('',1.0)]))
        self.assertEqual(greedy_decode([{'blank':.5,'a':.5}], ['blank','a'])[0], ['blank'])
        self.assertEqual(prefix_transitions('a', .2, .3, 'a', .5), [('a', .15), ('aa', .1)])
        with self.assertRaises(ValueError):
            prefix_beam_decode([], ['blank','a'], 0)
