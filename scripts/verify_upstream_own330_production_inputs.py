#!/usr/bin/env python3
"""Separate source-column reconstruction of both full own330 production streams."""
import hashlib
import numpy as np
import upstream_own330_production_inputs as data

u = data.u


def main():
    declaration = u.read(data.REPORT / 'declaration.json')
    transport = u.read(data.REPORT / 'transport.json')
    if transport['declaration_sha256'] != u.sha(data.REPORT / 'declaration.json'):
        raise ValueError('Original production declaration changed')
    data.raw.check(declaration['source_sha256'])
    records = []
    for query, count in data.COUNTS.items():
        reader = data.Inputs(query, declaration['rows'])
        digests = [hashlib.sha256(), hashlib.sha256()]
        try:
            for lo in range(0, count, 16384):
                hi = min(lo+16384, count)
                z, f = reader.block(lo, hi)
                oldz = reader.design.read(lo, hi)
                oldf = reader.features.read(lo, hi)
                expected_z = np.empty((hi-lo, 406), dtype=np.float64)
                expected_f = np.empty((hi-lo, 556), dtype=np.float32)
                expected_z[:, :182] = oldz[:, :182]
                expected_z[:, -1] = oldz[:, -1]
                expected_f[:, :243] = oldf[:, :243]
                expected_f[:, 302:392] = oldf[:, 243:333]
                for j, source in enumerate(reader.raw):
                    value = source.read(lo, hi).copy()
                    value[value < 0] = 0
                    value[value > 6] = 6
                    expected_z[:, 182+j] = value
                    position = 243+j if j < 59 else 392+j-59
                    expected_f[:, position] = value - np.float32(3.6)
                if (not np.array_equal(z.view('u8'), expected_z.view('u8')) or
                        not np.array_equal(f.view('u4'), expected_f.view('u4'))):
                    raise ValueError('Original source-column production operator differs')
                digests[0].update(expected_z.tobytes(order='C'))
                digests[1].update(expected_f.tobytes(order='C'))
        finally:
            reader.close()
        actual = [h.hexdigest() for h in digests]
        if actual != transport['c_order_sha256'][query]:
            raise ValueError('Full independently reconstructed production stream differs')
        records.append(dict(query=query, rows=count, c_order_sha256=actual, all_columns_bitwise_equal=True))
    data.raw.check(declaration['source_sha256'])
    u.write(data.REPORT / 'independent-verification.json', dict(
        state='complete_both_full_paired_own330_source_column_reconstructions',
        declaration_sha256=u.sha(data.REPORT / 'declaration.json'),
        transport_sha256=u.sha(data.REPORT / 'transport.json'),
        records=records, qualifying_labels_read=False, models_fitted=False))


if __name__ == '__main__':
    main()
