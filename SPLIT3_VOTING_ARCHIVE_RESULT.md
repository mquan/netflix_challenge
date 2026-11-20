# Exact voting inputs preserved; idle RAM released for full neural fitting

After both matched arms and all30 saved models passed replay, the idle
original feature owner still held a915,456,880-byte table. Its0.878GiB
footprint put the next full16GiB neural reservation over the24GiB total
limit by about0.457GiB.

All130 features and1,760,494 rows are now stored in ten lossless zstd parts,
337,122,577 bytes total. Each original full feature hash and the complete
feature-major payload hash match. Every archived byte was independently
decoded twice, including the complete aggregate original hash:

`77e47e5c5636b909fc45e103ca5c9bf2fa312c75d41aac9c31c987c2f227bb63`

Part contents, the complete catalog and relevant directories were synced
before release. The original owner then exited through its existing
release-file contract. No source/model/input value was changed, no feature
was omitted and no byte was discarded without an exact archive.

The freed RAM let the full557-input own330 neural worker start immediately
under its unchanged16GiB/24GiB guards. The original source/proof records
remain the historical fitting evidence. Restored inputs use the archive's
exact full-payload provenance in place of the former ephemeral RAM owner.

When the original disk/checkpoint reserve permits, restore every byte with:

```sh
.venv/bin/python scripts/upstream_split3_voting_archive.py --restore
```

The verified output is
`artifacts/upstream-split3-voting-archive/probe3-feature-major.f32`,
little-endian float32, C order, shape130×1,760,494. Restoration verifies all
compressed part hashes, the full byte count and original payload SHA256.
The original compressed archive stays available after restoration.

[Complete archive proof](experiments/upstream-split3-voting-archive/verification.json) ·
[Original owner release](experiments/upstream-split3-voting-archive/release-verification.json).
