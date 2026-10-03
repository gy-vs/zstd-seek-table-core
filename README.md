# Zstandard frame core

Run `npm ci`, `npm test`, and `npm run build`.

## Seekable Zstandard support

The core parses Zstandard block/frame header primitives, resolves versioned
explicit dictionaries, and adds random access over the
[seekable format](https://github.com/facebook/zstd/blob/dev/contrib/seekable_format/zstd_seekable_format.md).

### Indexing

```ts
import { parseSeekTable, createSeekableReader } from './dist/index.js';

// Parses the seek table that lives in the dedicated skippable frame at EOF.
// Plain Zstandard inputs fail with SEEK_TABLE_NOT_FOUND instead of falling
// back to a sequential scan; generic 0x184D2A5x skippable frames are never
// mistaken for an index.
const index = parseSeekTable(bytes);

for (const entry of index.entries) {
  // Half-open bigint intervals in both coordinate spaces.
  entry.compressedStart;   // bigint
  entry.compressedEnd;
  entry.decompressedStart;
  entry.decompressedEnd;
  entry.checksum;          // raw uint32 when the descriptor requests checksums
}
index.tableOffset;         // table bytes are never counted as data frames
index.compressedLength;
index.decompressedLength;
```

Validation covers the magic, descriptor reserved bits, the two reserved bytes,
entry count/body divisibility, the per-entry compressed/decompressed sizes and
optional checksum, accumulated offsets against the buffer, data frame magics
and the total compressed length. All offsets are `bigint`; counts and
accumulated lengths cannot overflow JS numbers or escape the buffer.

`locateFrameHeader(data, entry)` positions the optional frame header fields
(window descriptor, dictionary id with `none` vs. application id `0` kept
distinct, and the frame content size) and reports the first-block offset.

### Window reads

```ts
const reader = createSeekableReader(bytes, { repository });

const window = await reader.readWindow({
  start: 100_000,
  length: 4_096,
  decoder,     // injected InjectedFrameDecoder<Uint8Array>
  signal,      // optional AbortSignal
});

window.data;  // Uint8Array slice of the decompressed range
window.frames; // exactly the frames intersecting the window
```

Only frames intersecting the requested decompressed range reach the injected
decoder; each receives the boundary-validated whole-frame bytes plus the frame
index as `frameId`. Decoded output must match the seek table's declared size
exactly (`SEEK_FRAME_DECODED_SIZE_MISMATCH` otherwise); header content size
disagreements fail as `SEEK_FRAME_SIZE_MISMATCH` before decoding. Explicit
dictionary frames acquire a protected version snapshot through the versioned
repository; the lease is released after the frame settles, including failure
and cancellation paths.

Concurrent windows hitting the same in-flight frame share one decode and one
dictionary lease; nothing is cached after settlement, so a later request always
observes the current dictionary revision. Cancelling every waiter aborts the
shared decode; a single cancelled window never disturbs the others. All errors
are `SeekableError` values carrying `code` plus compressed/decompressed
coordinates.
