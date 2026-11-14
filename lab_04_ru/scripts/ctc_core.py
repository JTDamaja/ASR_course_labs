"""Small, independently enumerated CTC paths."""
import itertools
import math

BLANK = 'blank'


def validate(emissions, alphabet, blank=BLANK):
    if not isinstance(blank, str) or not blank:
        raise ValueError('blank must be a nonempty string')
    if not isinstance(alphabet, list) or not alphabet or any(not isinstance(s, str) or (s != blank and len(s) != 1) for s in alphabet):
        raise ValueError('alphabet must contain blank and single-character labels')
    if len(set(alphabet)) != len(alphabet) or alphabet.count(blank) != 1:
        raise ValueError('alphabet must contain unique symbols and blank exactly once')
    if not isinstance(emissions, list):
        raise ValueError('emissions must be a list')
    for row in emissions:
        if not isinstance(row, dict) or set(row) != set(alphabet):
            raise ValueError('emission row keys must match alphabet')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1 or not math.isfinite(v) for v in row.values()):
            raise ValueError('probabilities must be finite numbers in [0, 1]')
        if not math.isclose(sum(row.values()), 1.0, abs_tol=1e-9, rel_tol=0):
            raise ValueError('emission row must sum to 1')


def merge_then_remove(path, blank):
    # напишите ваш код здесь
    raise NotImplementedError('STUDENT: complete collapse')


def collapse_path(path, blank=BLANK):
    if not isinstance(blank, str) or not blank:
        raise ValueError('blank must be a nonempty string')
    if any(not isinstance(s, str) or (s != blank and len(s) != 1) for s in path):
        raise ValueError('path symbols must be blank or single-character labels')
    return merge_then_remove(path, blank)


def enumerate_paths(emissions, alphabet, blank=BLANK):
    validate(emissions, alphabet, blank)
    if len(alphabet) ** len(emissions) > 10000:
        raise ValueError('enumeration exceeds 10000 paths')
    records = []
    for path in itertools.product(alphabet, repeat=len(emissions)):
        probability = math.prod(row[symbol] for row, symbol in zip(emissions, path))
        records.append({'path': list(path), 'transcript': collapse_path(path, blank), 'probability': probability})
    return records


def sum_matching(records, target):
    # напишите ваш код здесь
    raise NotImplementedError('STUDENT: complete aggregate')


def sum_transcript_probability(records, target):
    return sum_matching(records, target)
