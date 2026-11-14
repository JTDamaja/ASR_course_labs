"""Создать детерминированные результаты для синтетической лабораторной работы CTC."""
import argparse
import csv
import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ctc_core import enumerate_paths, sum_transcript_probability
from scripts.ctc_forward import extended_target, forward_table
from scripts.ctc_decode import greedy_decode, prefix_beam_decode


def write_csv(path, header, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs')
    out = parser.parse_args().output_dir
    fixtures = {name: json.loads((ROOT / 'data' / (name + '.json')).read_text(encoding='utf-8')) for name in ('forward', 'decoding', 'repeated')}
    f, d = fixtures['forward'], fixtures['decoding']
    records = enumerate_paths(f['emissions'], f['alphabet'], f['blank'])
    table = forward_table(f['emissions'], f['target'], f['blank'])
    states = extended_target(f['target'], f['blank'])
    path, greedy = greedy_decode(d['emissions'], d['alphabet'], d['blank'])
    best, ledger = prefix_beam_decode(d['emissions'], d['alphabet'], d['beam_width'], d['blank'])
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'paths.csv', ['path', 'transcript', 'probability'], [(json.dumps(r['path'], separators=(',', ':')), r['transcript'], r['probability']) for r in records])
    write_csv(out / 'forward_table.csv', ['frame'] + [f'{i}:{s}' for i, s in enumerate(states)], [[i + 1] + row for i, row in enumerate(table)])
    write_csv(out / 'decoding.csv', ['method', 'width', 'path', 'transcript', 'retained_mass'], [['greedy', '', json.dumps(path, separators=(',', ':')), greedy, '']] + [['beam', d['beam_width'], '', prefix, score] for prefix, score in ledger])
    summary = {'fixtures': ['forward.json', 'decoding.json', 'repeated.json'], 'target': f['target'], 'repeated_target': fixtures['repeated']['target'], 'path_count': len(records), 'matching_path_count': sum(r['transcript'] == f['target'] for r in records), 'transcript_probability': sum_transcript_probability(records, f['target']), 'final_alpha_cells': table[-1][-2:], 'greedy_path': path, 'greedy_transcript': greedy, 'beam_width': d['beam_width'], 'beam_best': best, 'beam_ledger': ledger}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    width = 120 + 110 * len(states)
    height = 100 + 60 * len(table)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="10" y="25" font-size="16">Синтетическая прямая таблица CTC (терминальные ячейки выделены)</text>']
    for s, label in enumerate(states):
        svg.append(f'<text x="{120 + s * 110}" y="55" font-size="14">{s}: {escape(label)}</text>')
    for t, row in enumerate(table):
        y = 65 + t * 60
        svg.append(f'<text x="10" y="{y + 30}" font-size="14">кадр {t + 1}</text>')
        for s, value in enumerate(row):
            x = 115 + s * 110
            fill = '#ffe69b' if t == len(table) - 1 and s >= len(states) - 2 else '#eeeeee'
            svg.append(f'<rect x="{x}" y="{y}" width="100" height="45" fill="{fill}" stroke="#333"/><text x="{x + 15}" y="{y + 29}" font-size="16">{value:.3f}</text>')
    svg.append('</svg>')
    (out / 'forward_table.svg').write_text('\n'.join(svg) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
