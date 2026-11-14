"""CTC forward recursion in probability space."""
from scripts.ctc_core import BLANK, validate


def extended_target(target, blank=BLANK):
    if not isinstance(target, str) or not isinstance(blank, str) or not blank:
        raise ValueError('target and blank must be strings')
    states = [blank]
    for symbol in target:
        if symbol == blank:
            raise ValueError('target cannot contain blank')
        states.extend([symbol, blank])
    return states


def incoming_sum(previous, extended, state, blank):
    # напишите ваш код здесь
    raise NotImplementedError('STUDENT: complete incoming')


def terminal_sum(last_row):
    # напишите ваш код здесь
    raise NotImplementedError('STUDENT: complete terminal')


def forward_table(emissions, target, blank=BLANK):
    states = extended_target(target, blank)
    if not isinstance(emissions, list):
        raise ValueError('emissions must be a list')
    if not emissions:
        return []
    if not isinstance(emissions[0], dict):
        raise ValueError('emission row must be a dictionary')
    alphabet = list(emissions[0])
    validate(emissions, alphabet, blank)
    if any(symbol not in alphabet for symbol in target):
        raise ValueError('target symbol missing from emissions')
    first = [0.0] * len(states)
    first[0] = emissions[0][blank]
    if len(states) > 1:
        first[1] = emissions[0][states[1]]
    table = [first]
    for row in emissions[1:]:
        previous = table[-1]
        table.append([row[symbol] * incoming_sum(previous, states, s, blank) for s, symbol in enumerate(states)])
    return table


def transcript_probability(table):
    return terminal_sum(table[-1]) if table else 0.0
