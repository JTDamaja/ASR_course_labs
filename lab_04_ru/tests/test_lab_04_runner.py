import csv
import json
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    def test_outputs_and_cwd(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            for name, cwd in [('one', ROOT), ('two', base)]:
                subprocess.run([sys.executable, str(ROOT / 'scripts/run_lab_04.py'), '--output-dir', str(base / name)], cwd=cwd, check=True, capture_output=True, text=True)
            files = {'paths.csv', 'forward_table.csv', 'decoding.csv', 'summary.json', 'forward_table.svg'}
            self.assertEqual({p.name for p in (base / 'one').iterdir()}, files)
            for name in files:
                self.assertEqual((base / 'one' / name).read_bytes(), (base / 'two' / name).read_bytes())
            relative = subprocess.run([sys.executable, str(ROOT / 'scripts/run_lab_04.py'), '--output-dir', 'relative'], cwd=base, check=True, capture_output=True, text=True)
            self.assertEqual(relative.returncode, 0)
            self.assertTrue((base / 'relative/summary.json').is_file())
            with (base / 'one/paths.csv').open(newline='') as f:
                self.assertEqual(len(list(csv.DictReader(f))), 27)
            with (base / 'one/forward_table.csv').open(newline='') as f:
                table = list(csv.reader(f))
                self.assertEqual(table[0], ['frame', '0:blank', '1:b', '2:blank', '3:a', '4:blank'])
                self.assertEqual(len(table), 4)
            with (base / 'one/decoding.csv').open(newline='') as f:
                decoded = list(csv.DictReader(f))
                self.assertEqual(decoded[0]['method'], 'greedy')
                self.assertEqual(decoded[0]['retained_mass'], '')
                self.assertEqual(decoded[1]['method'], 'beam')
            summary = json.loads((base / 'one/summary.json').read_text())
            self.assertAlmostEqual(summary['transcript_probability'], .661)
            self.assertEqual(summary['greedy_transcript'], 'a')
            self.assertEqual(summary['beam_best'], 'ab')
            svg = ET.parse(base / 'one/forward_table.svg')
            svg_text = ''.join(svg.getroot().itertext())
            self.assertIn('0.581', svg_text)
            self.assertIn('Синтетическая прямая таблица CTC', svg_text)
