#!/usr/bin/env python3
"""Independently bind native qualifying row identities to the original text order."""
import datetime as dt
from pathlib import Path
import numpy as np
import upstream_repro as u


def verify_order(text_path, raw_users, users, items, dates):
    if users.ndim != 1 or items.shape != users.shape or dates.shape != users.shape:
        raise ValueError('Qualifying identity shapes differ')
    if raw_users.ndim != 1 or np.any(raw_users[1:] <= raw_users[:-1]):
        raise ValueError('Raw user mapping must be strictly increasing')
    origin = dt.date(1999, 11, 11)
    date_cache = {}
    row = 0
    movie = None
    with Path(text_path).open() as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            if line.endswith(':'):
                movie = int(line[:-1]) - 1
                continue
            if movie is None or row >= len(users):
                raise ValueError('Unexpected qualifying text row')
            raw_user, date = line.split(',')
            if date not in date_cache:
                date_cache[date] = (dt.date.fromisoformat(date) - origin).days
            user_index = int(users[row])
            if not 0 <= user_index < len(raw_users):
                raise ValueError('Invalid native user index at row '+str(row))
            if (int(raw_users[user_index]) != int(raw_user) or
                    int(items[row]) != movie or int(dates[row]) != date_cache[date]):
                raise ValueError('Qualifying identity/order mismatch at row '+str(row))
            row += 1
    if row != len(users):
        raise ValueError('Qualifying text/native row counts differ')
    return row


def main():
    u.validate_fresh()
    paths = {'text': u.FRESH/'data/raw/qualifying.txt',
             'raw_users': u.FRESH/'data/fulltrain/uidx_to_uid.npy',
             **{key: u.FRESH/'data/qual'/(key+'.npy')
                for key in ['user_idxs', 'item_idxs', 'dates']}}
    before = {key: u.sha(path) for key, path in paths.items()}
    count = verify_order(paths['text'], *[np.load(paths[key], mmap_mode='r')
                        for key in ['raw_users', 'user_idxs', 'item_idxs', 'dates']])
    if count != 2817131 or before != {key: u.sha(path) for key, path in paths.items()}:
        raise ValueError('Incomplete or changed qualifying identity inputs')
    u.write(u.ROOT/'experiments/upstream-production-bank/qualifying-order-verification.json',
            dict(rows=count, all_user_item_date_identities_equal=True,
                 source_pin=u.PIN, verifier_sha256=u.sha(Path(__file__)),
                 input_sha256={str(paths[key].relative_to(u.ROOT)): digest
                               for key, digest in before.items()},
                 qualifying_outcomes_read=False,
                 scope='All native qualifying row identities equal the original qualifying.txt parse order; independently parsed with Python dates. No ratings or recovered split outcomes read.'))
    print('Verified all', count, 'qualifying user/item/date identities in original order.')


if __name__ == '__main__':
    main()
