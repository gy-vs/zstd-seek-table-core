# Zstandard frame core

Run `npm ci`, `npm test`, and `npm run build`.

## Framing kernel

`parseDescriptor` / `readBlockHeader` (in `src/frames.ts`, re-exported from the package
root) parse the frame header descriptor byte and block headers. `parseFrameHeader(data,
offset)` locates the optional frame header fields — window descriptor, dictionary id,
and frame content size — and returns the dictionary selector plus the total header
size. The content size is a `bigint` because the 8-byte field form exceeds 2^53.

## Seekable archives

`parseSeekTable(data)` locates the seek table from the tail of a seekable zstd file and
validates it before trusting a single byte of it:

- the footer magic `0x8F92EAB1` at the very end of the input — anything else fails
  with `NOT_SEEKABLE` and no sequential scan is ever attempted;
- the seek table descriptor's reserved bits (5-6) must be zero; bit 7 selects whether
  every entry carries a 4-byte checksum field;
- the entry count must agree with the skippable frame's declared payload size, and the
  table must sit inside the dedicated skippable frame `0x184D2A5E` — a similar
  skippable magic (e.g. `0x184D2A50`) is rejected, its payload is never treated as an
  index;
- the cumulative compressed sizes plus the table's own bytes must exactly span the
  input, so the table bytes are never counted into a data frame offset.

All accumulation is done in `bigint` (up to 2^32 entries of 2^32 bytes would overflow a
JS number), and the entry count is bounds-checked before any entry array is allocated,
so malicious counts and lengths cannot escape the buffer or lose precision. The result
is a `SeekTableIndex` whose entries expose half-open `bigint` intervals
`[compressedOffset, compressedEnd)` and `[decompressedOffset, decompressedEnd)` per
data frame, plus `tableOffset`/`tableSize` for the table itself. Failures raise
`SeekTableError` with a machine-readable `code`.

## Random-access window reads

`createSeekableReader({ data, decoder, repository? | resolver?, index? })` returns a
reader whose `read({ offset, length, signal? })` slices a window out of the
decompressed stream without decoding the whole file:

- only data frames intersecting the window are decoded; a zero-length window decodes
  nothing;
- the injected decoder receives the boundary-validated compressed bytes of exactly one
  frame and its frame index (`context.frameId`), and must return that frame's
  decompressed bytes as a `Uint8Array`;
- if the decoded length differs from the decompressed size declared by the seek table,
  the read fails with `FRAME_SIZE_MISMATCH` instead of returning truncated data or
  padding it from the next frame;
- the frame's dictionary selector comes from its own header: a missing field stays
  `none`, a zero-valued field stays the `application` convention, and an explicit id is
  resolved through the versioned dictionary resolver (`createDictionaryFrameResolver`),
  so the decoder receives a pinned revision snapshot and the lease is released when the
  decode finishes — including on failure and cancellation paths;
- errors raised by the reader itself are `SeekableReadError`s carrying the compressed
  and decompressed coordinates of the offending frame or window; dictionary resolution
  and decoder errors propagate unchanged with the frame index as `frameId`.

**Decode sharing and staleness.** Concurrent reads that hit the same frame share one
in-flight decode (one decoder call, one dictionary lease). Cancelling one window only
detaches that waiter; the underlying decode is aborted when its last waiter cancels.
The shared job is evicted as soon as it settles and is never reused by a later read, so
every new read resolves its dictionary lease at the repository's current revision — a
dictionary update can neither disturb an in-flight read (its lease is pinned by the
resolver) nor leak stale content into a read that starts after the update.
