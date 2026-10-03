import { describe, expect, it, vi } from 'vitest';
import {
  APPLICATION_DICTIONARY,
  NO_DICTIONARY,
  InMemoryVersionedDictionaryRepository,
  SeekableError,
  SeekableWindowAbortError,
  createSeekableReader,
  explicitDictionary,
  locateFrameHeader,
  parseSeekTable,
  type DictionarySelector,
  type FrameDecoderContext,
  type ResolvedDictionary,
} from '../src/index.js';

// --- Tiny sample builders --------------------------------------------------
//
// Data frames are real Zstandard frames containing single *raw* blocks, so the
// frame headers are genuine but the injected decoder decides how the blocks are
// turned into bytes.

const FRAME_MAGIC = [0x28, 0xb5, 0x2f, 0xfd];

function u32le(value: number): number[] {
  return [value & 0xff, (value >>> 8) & 0xff, (value >>> 16) & 0xff, (value >>> 24) & 0xff];
}

function dictionaryField(selector: DictionarySelector): { descriptorMask: number; bytes: number[] } {
  if (selector.kind === 'none') return { descriptorMask: 0, bytes: [] };
  if (selector.kind === 'application') return { descriptorMask: 1, bytes: [0] };
  const id = selector.id;
  if (id <= 0xff) return { descriptorMask: 1, bytes: [id] };
  if (id <= 0xffff) return { descriptorMask: 2, bytes: u32le(id).slice(0, 2) };
  return { descriptorMask: 3, bytes: u32le(id) };
}

function buildDataFrame(
  payload: Uint8Array,
  options: {
    dictionary?: DictionarySelector;
    contentChecksum?: boolean;
    /** Override the encoded content size field; null omits an empty-frame FCS. */
    fcsOverride?: number | null;
  } = {},
): Uint8Array {
  const selector = options.dictionary ?? NO_DICTIONARY;
  const dict = dictionaryField(selector);
  const checksumBit = options.contentChecksum ? 0x04 : 0;
  // Single segment (0x20), FCS flag 0 => one-byte FCS; dictionary flag in low bits.
  const descriptor = 0x20 | dict.descriptorMask | checksumBit;
  const header: number[] = [...FRAME_MAGIC, descriptor, ...dict.bytes];
  if (options.fcsOverride !== null) {
    header.push(...(options.fcsOverride === undefined ? [payload.length === 256 ? 0 : payload.length] : u32le(options.fcsOverride).slice(0, 1)));
  }
  // Single raw block: last=1, type=0, block size in 3-byte little endian.
  const blockHeader = [
    1 | (payload.length << 3) & 0xff,
    ((payload.length >>> 8) << 3) & 0xff,
    ((payload.length >>> 16) << 3) & 0xff,
  ];
  return new Uint8Array([...header, ...blockHeader, ...payload]);
}

function buildSeekTableFile(
  frames: { compressed: Uint8Array; decompressedSize: number; checksum?: number | null }[],
  options: { descriptor?: number; reserved?: [number, number]; checksums?: boolean } = {},
): { file: Uint8Array; frameBytes: Uint8Array[]; tableOffset: number } {
  const checksums = options.checksums ?? false;
  const descriptor = options.descriptor ?? (checksums ? 0x01 : 0x00);
  const reserved = options.reserved ?? ([0, 0] as [number, number]);
  const frameBytes = frames.map((frame) => frame.compressed);
  const payload: number[] = [descriptor, ...reserved];
  for (let index = 0; index < frames.length; index += 1) {
    const frame = frames[index]!;
    payload.push(...u32le(frame.compressed.length));
    payload.push(...u32le(frame.decompressedSize));
    if (checksums) {
      payload.push(...u32le(frame.checksum ?? (0x90000000 + index)));
    }
  }
  const table: number[] = [0xbf, 0xea, 0x92, 0x8f, ...u32le(payload.length), ...payload];
  const tableOffset = frameBytes.reduce((sum, frame) => sum + frame.length, 0);
  return {
    file: new Uint8Array([...frameBytes.flatMap((frame) => [...frame]), ...table]),
    frameBytes,
    tableOffset,
  };
}

/** Table offset derived by scanning for the magic; mirrors real-world lookup. */
function tableOffset(file: Uint8Array): number {
  for (let offset = file.length - 8; offset >= 0; offset -= 1) {
    if (
      file[offset] === 0xbf && file[offset + 1] === 0xea &&
      file[offset + 2] === 0x92 && file[offset + 3] === 0x8f
    ) {
      const payloadLength = (
        file[offset + 4]! | (file[offset + 5]! << 8) |
        (file[offset + 6]! << 16) | (file[offset + 7]! << 24)
      ) >>> 0;
      if (offset + 8 + payloadLength === file.length) {
        return offset;
      }
    }
  }
  throw new Error('no table magic found in test helper');
}

function concat(...arrays: Uint8Array[]): Uint8Array {
  const total = arrays.reduce((sum, array) => sum + array.length, 0);
  const out = new Uint8Array(total);
  let cursor = 0;
  for (const array of arrays) {
    out.set(array, cursor);
    cursor += array.length;
  }
  return out;
}

// Decoder: parses the genuine frame header minimally and emits the raw block
// payload; optionally validates the resolved dictionary, or fails.
interface RawBlockDecoderOptions {
  failIndex?: number;
  failError?: Error;
  failWhen?: (entry: FrameDecoderContext) => boolean;
  onDictionary?: (dictionary: ResolvedDictionary, context: FrameDecoderContext) => void;
  /** Returns a different length than declared for a frame. */
  outputFor?: (frameIndex: number, frame: Uint8Array) => Uint8Array;
  /** Keeps the injected decode pending until this promise resolves. */
  gate?: Promise<void>;
  /** Observed per-frame abort events from the decode context signal. */
  onAbort?: (context: FrameDecoderContext) => void;
}

function rawBlockDecoder(options: RawBlockDecoderOptions = {}) {
  const decodeFrame = vi.fn(async (context: FrameDecoderContext, dictionary: ResolvedDictionary): Promise<Uint8Array> => {
    options.onDictionary?.(dictionary, context);
    if (options.onAbort) {
      context.signal.addEventListener('abort', () => options.onAbort!(context), { once: true });
    }
    if (options.gate) {
      await options.gate;
    }
    if (options.failIndex === Number(context.frameId) || options.failWhen?.(context)) {
      throw options.failError ?? new Error(`injected decode failure for frame ${String(context.frameId)}`);
    }
    if (options.outputFor) {
      return options.outputFor(Number(context.frameId), context.data);
    }
    const frame = context.data;
    // Frame magic (4) + descriptor (1); single segment, no window descriptor.
    const descriptorByte = frame[4]!;
    let cursor = 5;
    const dictFlag = descriptorByte & 3;
    if (dictFlag === 1) cursor += 1;
    else if (dictFlag === 2) cursor += 2;
    else if (dictFlag === 3) cursor += 4;
    // One-byte FCS.
    cursor += 1;
    const blockHeader = frame[cursor]! | (frame[cursor + 1]! << 8) | (frame[cursor + 2]! << 16);
    const blockSize = blockHeader >>> 3;
    // Must hand back exactly the whole frame's raw bytes; bounds are real.
    if (cursor + 3 + blockSize !== frame.length) {
      throw new Error(`frame ${String(context.frameId)} raw block runs past frame boundary`);
    }
    return frame.subarray(cursor + 3, cursor + 3 + blockSize);
  });
  return { decodeFrame, mock: decodeFrame.mock };
}

function sampleFile(checksums = false) {
  const a = new Uint8Array([10, 11, 12]);
  const b = new Uint8Array([20, 21, 22, 23, 24]);
  const c = new Uint8Array([30, 31]);
  const fa = buildDataFrame(a);
  const fb = buildDataFrame(b);
  const fc = buildDataFrame(c);
  const { file, tableOffset: sampleTableOffset } = buildSeekTableFile(
    [
      { compressed: fa, decompressedSize: a.length },
      { compressed: fb, decompressedSize: b.length },
      { compressed: fc, decompressedSize: c.length },
    ],
    { checksums },
  );
  return { file, payloads: [a, b, c], frames: [fa, fb, fc], tableOffset: sampleTableOffset };
}

const flush = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

function expectSeekError(fn: () => unknown): SeekableError {
  try {
    fn();
  } catch (error) {
    expect(error).toBeInstanceOf(SeekableError);
    return error as SeekableError;
  }
  throw new Error('expected parseSeekTable to throw');
}

// --- Seek table parsing ----------------------------------------------------

describe('parseSeekTable', () => {
  it('builds half-open bigint intervals in both coordinate spaces', () => {
    const { file, payloads } = sampleFile();
    const index = parseSeekTable(file);

    expect(index.entries).toHaveLength(3);
    expect(index.checksumPerFrame).toBe(false);
    expect(index.decompressedLength).toBe(BigInt(payloads[0]!.length + payloads[1]!.length + payloads[2]!.length));
    expect(index.compressedLength).toBe(index.tableOffset);

    let compressedCursor = 0n;
    let decompressedCursor = 0n;
    for (const entry of index.entries) {
      expect(entry.compressedStart).toBe(compressedCursor);
      expect(entry.decompressedStart).toBe(decompressedCursor);
      expect(entry.compressedEnd - entry.compressedStart).toBe(BigInt(entry.index === 0 ? 12 : entry.index === 1 ? 14 : 11));
      expect(entry.checksum).toBeNull();
      compressedCursor = entry.compressedEnd;
      decompressedCursor = entry.decompressedEnd;
    }
    expect(compressedCursor).toBe(index.tableOffset);
    // The table frame bytes are not part of the data-frame interval space.
    expect(compressedCursor + BigInt(8 + index.tableFramePayloadLength)).toBe(BigInt(file.length));
  });

  it('exposes per-frame checksums only when the descriptor asks for them', () => {
    expect(parseSeekTable(sampleFile(true).file).entries.map((entry) => entry.checksum))
      .toEqual([0x90000000, 0x90000001, 0x90000002]);
    expect(parseSeekTable(sampleFile(false).file).checksumPerFrame).toBe(false);
  });

  it('refuses a non-seekable input instead of scanning the whole file', () => {
    const plainFrame = buildDataFrame(new Uint8Array([1, 2, 3]));
    expect(() => parseSeekTable(plainFrame)).toThrow(SeekableError);
    try {
      parseSeekTable(plainFrame);
      throw new Error('expected throw');
    } catch (error) {
      expect(error).toMatchObject({ code: 'SEEK_TABLE_NOT_FOUND' });
    }
  });

  it('ignores generic skippable magics and magic-looking payloads', () => {
    const { file } = sampleFile();
    // A skippable frame at the tail must never be treated as an index.
    const payload = new Uint8Array(20);
    const fake = concat(
      file,
      new Uint8Array([0x50, 0x2a, 0x4d, 0x18, ...u32le(payload.length), ...payload]),
    );
    expect(expectSeekError(() => parseSeekTable(fake)).code).toBe('SEEK_TABLE_NOT_FOUND');

    // Same bytes as the seek table magic but embedded in frame data, not at the
    // position derived from EOF: the trailing real table still wins.
    const poisoned = new Uint8Array([0xbf, 0xea, 0x92, 0x8f, 1, 2, 3]);
    const embedded = buildDataFrame(concat(new Uint8Array(poisoned), new Uint8Array(10)));
    const { file: withEmbedded } = buildSeekTableFile([
      { compressed: embedded, decompressedSize: 7 + 10 + 5 },
    ]);
    expect(parseSeekTable(withEmbedded).entries).toHaveLength(1);
  });

  it('rejects descriptor reserved bits and non-zero reserved bytes', () => {
    const frame = buildDataFrame(new Uint8Array([1]));
    expect(expectSeekError(() => parseSeekTable(buildSeekTableFile(
      [{ compressed: frame, decompressedSize: 1 }],
      { descriptor: 0x02 },
    ).file)).code).toBe('SEEK_TABLE_DESCRIPTOR_RESERVED');
    expect(expectSeekError(() => parseSeekTable(buildSeekTableFile(
      [{ compressed: frame, decompressedSize: 1 }],
      { reserved: [1, 0] },
    ).file)).code).toBe('SEEK_TABLE_RESERVED_BYTES');
  });

  it('rejects entry bodies that do not divide by the entry size', () => {
    const frame = buildDataFrame(new Uint8Array([1]));
    const built = buildSeekTableFile([{ compressed: frame, decompressedSize: 1 }]);
    // One trailing byte plus a payload length claiming it: body (12 bytes) is
    // not a multiple of the 8-byte entry size.
    const grown = concat(built.file, new Uint8Array([0]));
    grown.set(u32le(grown.length - built.tableOffset - 8), built.tableOffset + 4);
    expect(expectSeekError(() => parseSeekTable(grown)).code).toBe('SEEK_TABLE_BAD_ENTRY_LENGTH');

    // Enabling checksums without enlarging the body breaks division by 12 for
    // a one-entry (8-byte) table.
    const single = buildSeekTableFile([
      { compressed: frame, decompressedSize: 1 },
    ]);
    single.file[single.tableOffset + 8] = 0x01;
    expect(expectSeekError(() => parseSeekTable(single.file)).code).toBe('SEEK_TABLE_BAD_ENTRY_LENGTH');
  });

  it('rejects reserved bits in compressed sizes', () => {
    const frame = buildDataFrame(new Uint8Array([1, 2, 3]));
    const { file, tableOffset: table } = buildSeekTableFile([{ compressed: frame, decompressedSize: 3 }]);
    file.set(u32le(frame.length | 0x40000000), table + 8 + 3);
    expect(expectSeekError(() => parseSeekTable(file)).code).toBe('SEEK_TABLE_RESERVED_SIZE_BITS');
  });

  it('rejects entries whose accumulated compressed length exceeds the table offset', () => {
    const frame = buildDataFrame(new Uint8Array([1, 2, 3]));
    const { file, tableOffset: table } = buildSeekTableFile([{ compressed: frame, decompressedSize: 3 }]);
    // Inflate the declared compressed size past the real frame.
    file.set(u32le(frame.length + 100), table + 8 + 3);
    expect(expectSeekError(() => parseSeekTable(file)).code).toBe('SEEK_TABLE_OFFSET_OUT_OF_BOUNDS');
  });

  it('rejects total length mismatch and bad data frame magics', () => {
    const frame = buildDataFrame(new Uint8Array([1, 2, 3, 4]));
    const built = buildSeekTableFile([{ compressed: frame, decompressedSize: 4 }]);
    // Corrupt one byte of the data frame magic.
    built.file[1] = 0x00;
    expect(expectSeekError(() => parseSeekTable(built.file)).code).toBe('SEEK_FRAME_BAD_MAGIC');

    const short = buildSeekTableFile([{ compressed: frame, decompressedSize: 4 }]);
    // Claim a smaller compressed size: sums will not meet the table offset.
    short.file.set(u32le(4), short.tableOffset + 8 + 3);
    expect(expectSeekError(() => parseSeekTable(short.file)).code).toBe('SEEK_TABLE_TOTAL_LENGTH_MISMATCH');
  });

  it('rejects a malicious extra entry whose claimed frame has no magic', () => {
    const frame = buildDataFrame(new Uint8Array([1, 2, 3, 4]));
    // Payload of one real entry plus one zero entry (8 zero bytes); the second
    // entry claims a zero-sized frame starting inside the table, which fails on
    // the offset/magic checks rather than reading out of bounds.
    const built = buildSeekTableFile([{ compressed: frame, decompressedSize: 4 }]);
    const grown = concat(built.file, new Uint8Array(8));
    grown.set(u32le(grown.length - built.tableOffset - 8), built.tableOffset + 4);
    expect(['SEEK_TABLE_OFFSET_OUT_OF_BOUNDS', 'SEEK_FRAME_BAD_MAGIC'])
      .toContain(expectSeekError(() => parseSeekTable(grown)).code);
  });
});

// --- Frame header optional field location ---------------------------------

describe('locateFrameHeader', () => {
  it('locates the one-byte content size after the single-segment header', () => {
    const payload = new Uint8Array([7, 8, 9]);
    const frame = buildDataFrame(payload, { dictionary: explicitDictionary(42) });
    const { file } = buildSeekTableFile([{ compressed: frame, decompressedSize: payload.length }]);
    const index = parseSeekTable(file);
    const header = locateFrameHeader(file, index.entries[0]!);
    expect(header.dictionary).toEqual(explicitDictionary(42));
    expect(header.dictionaryFieldLength).toBe(1);
    expect(header.contentSize).toBe(3n);
    expect(header.headerSize).toBe(frame.length - payload.length - 3);
  });

  it('keeps the absent field distinct from the zero application id', () => {
    const without = buildDataFrame(new Uint8Array([1]));
    const withZero = buildDataFrame(new Uint8Array([1]), { dictionary: APPLICATION_DICTIONARY });
    expect(locateFrameHeader(without, { index: 0, compressedStart: 0n, compressedEnd: BigInt(without.length) }).dictionary)
      .toBe(NO_DICTIONARY);
    expect(locateFrameHeader(withZero, { index: 0, compressedStart: 0n, compressedEnd: BigInt(withZero.length) }).dictionary)
      .toEqual(APPLICATION_DICTIONARY);
  });

  it('locates fields after the window descriptor in non-single-segment frames', () => {
    const payload = new Uint8Array([1, 2, 3, 4, 5]);
    const dictId = 0x0123;
    const descriptor = 0x40 | 0x02; // FCS flag 1 (2-byte FCS) + dict flag 2
    const frame = new Uint8Array([
      ...FRAME_MAGIC,
      descriptor,
      0x40,                 // window descriptor
      dictId & 0xff, (dictId >>> 8) & 0xff,
      payload.length & 0xff, (payload.length >>> 8) & 0xff,
      // single raw block
      ((payload.length << 3) | 1) & 0xff,
      ((payload.length >>> 8) << 3) & 0xff,
      ((payload.length >>> 16) << 3) & 0xff,
      ...payload,
    ]);
    const header = locateFrameHeader(frame, { index: 0, compressedStart: 0n, compressedEnd: BigInt(frame.length) });
    expect(header.singleSegment).toBe(false);
    expect(header.windowDescriptor).toBe(0x40);
    expect(header.dictionary).toEqual(explicitDictionary(dictId));
    expect(header.dictionaryFieldLength).toBe(2);
    expect(header.contentSize).toBe(5n);
    expect(header.headerSize).toBe(4 + 1 + 1 + 2 + 2);
  });

  it('rejects reserved descriptor bits and truncated headers', () => {
    const frame = buildDataFrame(new Uint8Array([1]));
    frame[4] = frame[4]! | 0x08;
    expect(expectSeekError(() => locateFrameHeader(frame, { index: 0, compressedStart: 0n, compressedEnd: BigInt(frame.length) })))
      .toMatchObject({ code: 'SEEK_FRAME_HEADER_RESERVED_BITS' });
    expect(expectSeekError(() => locateFrameHeader(frame.subarray(0, 4), { index: 0, compressedStart: 0n, compressedEnd: 4n })))
      .toMatchObject({ code: 'SEEK_FRAME_HEADER_TRUNCATED' });
  });
});

// --- Random window reads ---------------------------------------------------

describe('createSeekableReader.readWindow', () => {
  it('decodes only the frames intersecting the window', async () => {
    const { file, payloads } = sampleFile();
    const decoder = rawBlockDecoder();
    const reader = createSeekableReader(file);

    // Last byte of frame 0 through all of frame 1 -> frames 0 and 1 only.
    const window = await reader.readWindow({ start: 2, length: 6, decoder });
    expect(Array.from(window.data)).toEqual([
      ...payloads[0]!.slice(2),
      ...payloads[1]!,
    ]);
    expect(window.frames.map((entry) => entry.index)).toEqual([0, 1]);
    expect(decoder.mock.calls.map((call) => call[0]!.frameId)).toEqual([0, 1]);
    // Every decoder call receives exactly the bounded whole-frame bytes.
    expect(BigInt(decoder.mock.calls[0]![0]!.data.length)).toBe(reader.index.entries[0]!.compressedEnd - reader.index.entries[0]!.compressedStart);
    expect(BigInt(decoder.mock.calls[1]![0]!.data.length)).toBe(reader.index.entries[1]!.compressedEnd - reader.index.entries[1]!.compressedStart);
  });

  it('handles zero-length windows without decoding anything and touches the tail', async () => {
    const { file, payloads } = sampleFile();
    const decoder = rawBlockDecoder();
    const reader = createSeekableReader(file);

    const empty = await reader.readWindow({ start: 4, length: 0, decoder });
    expect(empty.data).toHaveLength(0);
    expect(empty.frames).toEqual([]);
    expect(decoder.decodeFrame).not.toHaveBeenCalled();

    const tail = await reader.readWindow({
      start: payloads[0]!.length + payloads[1]!.length,
      length: payloads[2]!.length,
      decoder,
    });
    expect(Array.from(tail.data)).toEqual(Array.from(payloads[2]!));
    expect(tail.frames.map((entry) => entry.index)).toEqual([2]);

    await expect(reader.readWindow({ start: 0, length: 0, decoder })).resolves.toMatchObject({ start: 0n, end: 0n });
  });

  it('rejects windows outside the decompressed bounds', async () => {
    const reader = createSeekableReader(sampleFile().file);
    await expect(reader.readWindow({ start: 11, length: 0, decoder: rawBlockDecoder() }))
      .rejects.toMatchObject({ code: 'SEEK_WINDOW_OUT_OF_BOUNDS' });
    await expect(reader.readWindow({ start: 10, length: 2, decoder: rawBlockDecoder() }))
      .rejects.toMatchObject({ code: 'SEEK_WINDOW_OUT_OF_BOUNDS' });
    await expect(reader.readWindow({ start: -1, length: 1, decoder: rawBlockDecoder() }))
      .rejects.toMatchObject({ code: 'SEEK_WINDOW_BAD_ARGUMENT' });
  });

  it('fails when the declared size differs from the decoded length and never mixes frames', async () => {
    const payload = new Uint8Array([1, 2, 3]);
    const frame = buildDataFrame(payload);
    // The decoder returns one byte too few; the table declares three, so the
    // result must be a size error, not a truncated window.
    const { file } = buildSeekTableFile([{ compressed: frame, decompressedSize: 3 }]);
    const shortDecoder = rawBlockDecoder({
      outputFor: (_frameIndex, frameBytes) => {
        const rawStart = frameBytes.length - payload.length;
        return frameBytes.subarray(rawStart, rawStart + 2);
      },
    });
    await expect(createSeekableReader(file).readWindow({ start: 0, length: 3, decoder: shortDecoder }))
      .rejects.toMatchObject({
        code: 'SEEK_FRAME_DECODED_SIZE_MISMATCH',
        frameIndex: 0,
        compressedStart: 0n,
      });
    expect(shortDecoder.mock.calls[0]![0]!.data).toBeInstanceOf(Uint8Array);

    // Header FCS disagreeing with the table fails before invoking the decoder.
    const lyingFrame = buildDataFrame(payload, { fcsOverride: 99 });
    const { file: lyingFile } = buildSeekTableFile([{ compressed: lyingFrame, decompressedSize: 3 }]);
    const lyingDecoder = rawBlockDecoder();
    await expect(createSeekableReader(lyingFile).readWindow({ start: 0, length: 3, decoder: lyingDecoder }))
      .rejects.toMatchObject({ code: 'SEEK_FRAME_SIZE_MISMATCH', frameIndex: 0 });
    expect(lyingDecoder.decodeFrame).not.toHaveBeenCalled();
  });

  it('resolves explicit dictionary frames through the repository but not no/application selectors', async () => {
    const repository = new InMemoryVersionedDictionaryRepository();
    repository.add({ id: 42, revision: 'rev-a', content: new Uint8Array([5, 6]) });
    const fNone = buildDataFrame(new Uint8Array([1]));
    const fApp = buildDataFrame(new Uint8Array([2]), { dictionary: APPLICATION_DICTIONARY });
    const fExplicit = buildDataFrame(new Uint8Array([3]), { dictionary: explicitDictionary(42) });
    const { file } = buildSeekTableFile([
      { compressed: fNone, decompressedSize: 1 },
      { compressed: fApp, decompressedSize: 1 },
      { compressed: fExplicit, decompressedSize: 1 },
    ]);

    const seen: ResolvedDictionary[] = [];
    const decoder = rawBlockDecoder({ onDictionary: (dictionary) => seen.push(dictionary) });
    const reader = createSeekableReader(file, { repository });
    await reader.readWindow({ start: 0, length: 3, decoder });

    expect(seen[0]).toEqual({ kind: 'none' });
    expect(seen[1]).toEqual({ kind: 'application', id: 0 });
    expect(seen[2]).toMatchObject({ kind: 'explicit', lease: { id: 42, revision: 'rev-a' } });
    expect(repository.activeLeaseCount).toBe(0);

    // A reader without a repository fails explicit-dictionary frames explicitly.
    const orphan = createSeekableReader(file);
    await expect(orphan.readWindow({ start: 2, length: 1, decoder: rawBlockDecoder() }))
      .rejects.toMatchObject({ code: 'SEEK_DICTIONARY_RESOLUTION_FAILED', frameIndex: 2 });
  });

  it('uses a dictionary snapshot protected against mid-read updates and releases the lease', async () => {
    const repository = new InMemoryVersionedDictionaryRepository();
    repository.add({ id: 7, revision: 'old', content: new Uint8Array([1]) });
    const frame = buildDataFrame(new Uint8Array([9, 9]), { dictionary: explicitDictionary(7) });
    const { file } = buildSeekTableFile([{ compressed: frame, decompressedSize: 2 }]);

    let seenRevision: string | undefined;
    let holdGate: () => void = () => undefined;
    const held = new Promise<void>((resolve) => { holdGate = resolve; });
    const decoder = rawBlockDecoder({
      gate: held,
      onDictionary: (dictionary) => {
        if (dictionary.kind === 'explicit') seenRevision = String(dictionary.lease.revision);
      },
    });
    const reader = createSeekableReader(file, { repository });

    const windowPromise = reader.readWindow({ start: 0, length: 2, decoder });
    await flush();
    // Update while the injected decode is still running.
    repository.publishUpdate({ id: 7, revision: 'new', content: new Uint8Array([2]) });
    holdGate();
    await windowPromise;
    expect(seenRevision).toBe('old');
    expect(repository.activeLeaseCount).toBe(0);

    // After the read settles, a new read observes the new revision.
    const updated = await reader.readWindow({
      start: 0, length: 2,
      decoder: rawBlockDecoder({
        onDictionary: (dictionary) => {
          if (dictionary.kind === 'explicit') seenRevision = String(dictionary.lease.revision);
        },
      }),
    });
    expect(updated.data).toHaveLength(2);
    expect(seenRevision).toBe('new');
    expect(repository.activeLeaseCount).toBe(0);
  });

  it('shares one frame decode (and one lease) across parallel windows and never caches it', async () => {
    const repository = new InMemoryVersionedDictionaryRepository();
    repository.add({ id: 3, revision: 'r1', content: new Uint8Array([8]) });
    const frame = buildDataFrame(new Uint8Array([4, 5, 6, 7]), { dictionary: explicitDictionary(3) });
    const { file } = buildSeekTableFile([{ compressed: frame, decompressedSize: 4 }]);

    const decoder = rawBlockDecoder();
    const reader = createSeekableReader(file, { repository });
    const [w1, w2] = await Promise.all([
      reader.readWindow({ start: 0, length: 2, decoder }),
      reader.readWindow({ start: 2, length: 2, decoder }),
    ]);
    expect(Array.from(w1.data)).toEqual([4, 5]);
    expect(Array.from(w2.data)).toEqual([6, 7]);
    expect(decoder.mock.calls).toHaveLength(1);
    expect(repository.activeLeaseCount).toBe(0);

    // Nothing is retained: a later window decodes again.
    await reader.readWindow({ start: 1, length: 1, decoder });
    expect(decoder.mock.calls).toHaveLength(2);
    expect(repository.activeLeaseCount).toBe(0);
  });

  it('keeps a shared decode alive when one window cancels and fails the remaining window on decode error', async () => {
    const repository = new InMemoryVersionedDictionaryRepository();
    repository.add({ id: 9, revision: 'r', content: new Uint8Array() });
    const frame = buildDataFrame(new Uint8Array([1, 2, 3, 4]), { dictionary: explicitDictionary(9) });
    const { file } = buildSeekTableFile([{ compressed: frame, decompressedSize: 4 }]);

    let cancelGate: () => void = () => undefined;
    const gate = new Promise<void>((resolve) => { cancelGate = resolve; });
    const decoder = rawBlockDecoder({ gate });
    const reader = createSeekableReader(file, { repository });
    const cancelController = new AbortController();

    const cancelled = reader.readWindow({ start: 0, length: 1, decoder, signal: cancelController.signal });
    const kept = reader.readWindow({ start: 1, length: 1, decoder });
    await flush();
    cancelController.abort('not needed');
    await expect(cancelled).rejects.toBeInstanceOf(SeekableWindowAbortError);

    // The surviving window still gets its result from the shared decode.
    cancelGate();
    const result = await kept;
    expect(Array.from(result.data)).toEqual([2]);
    expect(decoder.mock.calls).toHaveLength(1);
    expect(repository.activeLeaseCount).toBe(0);
  });

  it('does not let one frame failure poison other frames or later requests', async () => {
    const { file } = sampleFile();
    const decoder = rawBlockDecoder({ failIndex: 1, failError: new Error('boom') });
    const reader = createSeekableReader(file);

    const results = await Promise.allSettled([
      reader.readWindow({ start: 0, length: 3, decoder }),
      reader.readWindow({ start: 3, length: 5, decoder }),
      reader.readWindow({ start: 8, length: 2, decoder }),
    ]);
    expect(results[0]!.status).toBe('fulfilled');
    expect(results[1]!.status).toBe('rejected');
    expect((results[1] as PromiseRejectedResult).reason).toMatchObject({
      code: 'SEEK_FRAME_DECODE_FAILED',
      frameIndex: 1,
      cause: new Error('boom'),
    });
    expect(results[2]!.status).toBe('fulfilled');
    expect(Array.from((results[2] as PromiseFulfilledResult<{ data: Uint8Array }>).value.data)).toEqual([30, 31]);

    // A later request on frame 1 is a fresh decode and can succeed if the
    // decoder is swapped; the failed outcome was not cached.
    const retryDecoder = rawBlockDecoder();
    const retry = await reader.readWindow({ start: 4, length: 1, decoder: retryDecoder });
    expect(Array.from(retry.data)).toEqual([21]);
  });

  it('aborts the shared decode and releases the lease only once every window cancels', async () => {
    const repository = new InMemoryVersionedDictionaryRepository();
    repository.add({ id: 11, revision: 'r', content: new Uint8Array() });
    const frame = buildDataFrame(new Uint8Array([1, 2]), { dictionary: explicitDictionary(11) });
    const { file } = buildSeekTableFile([{ compressed: frame, decompressedSize: 2 }]);

    let gateResolve: () => void = () => undefined;
    const gate = new Promise<void>((resolve) => { gateResolve = resolve; });
    const abortSpy = vi.fn();
    const decoder = rawBlockDecoder({ gate, onAbort: () => abortSpy() });
    const reader = createSeekableReader(file, { repository });

    const controllerA = new AbortController();
    const controllerB = new AbortController();
    const a = reader.readWindow({ start: 0, length: 1, decoder, signal: controllerA.signal });
    const b = reader.readWindow({ start: 1, length: 1, decoder, signal: controllerB.signal });
    await flush();

    controllerA.abort();
    await expect(a).rejects.toBeInstanceOf(SeekableWindowAbortError);
    // Still one live participant: decode/load signal must not be aborted yet.
    expect(abortSpy).not.toHaveBeenCalled();
    expect(repository.activeLeaseCount).toBe(1);

    controllerB.abort();
    await expect(b).rejects.toBeInstanceOf(SeekableWindowAbortError);
    await flush();
    expect(abortSpy).toHaveBeenCalledTimes(1);
    expect(decoder.mock.calls).toHaveLength(1);

    // Let the decoder settle after the abort; the lease is released anyway.
    gateResolve();
    await flush();
    await flush();
    expect(repository.activeLeaseCount).toBe(0);
  });

  it('isolates cancellation across frames of one multi-frame window and never caches it', async () => {
    const { file } = sampleFile();
    // Frame 1's decode is held behind a gate while frames 0 and 2 return fast.
    let gateResolve: () => void = () => undefined;
    const gate = new Promise<void>((resolve) => { gateResolve = resolve; });
    const gating = rawBlockDecoder({ gate, onAbort: () => gateResolve() });
    const plain = rawBlockDecoder();

    const controller = new AbortController();
    const reader = createSeekableReader(file);
    // Separate decoder instances so only frame 1's flight is gated.
    const windowPromise = reader.readWindow({
      start: 2,
      length: 6,
      decoder: {
        decodeFrame: (context: FrameDecoderContext, dictionary: ResolvedDictionary) =>
          Number(context.frameId) === 1
            ? gating.decodeFrame(context, dictionary)
            : plain.decodeFrame(context, dictionary),
      },
      signal: controller.signal,
    });

    await flush();
    expect(gating.mock.calls).toHaveLength(1);
    controller.abort('cancel');
    await expect(windowPromise).rejects.toBeInstanceOf(SeekableWindowAbortError);
    gateResolve();
    await flush();

    // Frame 1's aborted outcome is not cached: a fresh window decodes every
    // touched frame anew and succeeds end to end.
    const retry = await reader.readWindow({ start: 2, length: 6, decoder: rawBlockDecoder() });
    expect(Array.from(retry.data)).toEqual([12, 20, 21, 22, 23, 24]);
    expect(retry.frames.map((entry) => entry.index)).toEqual([0, 1]);
  });

  it('aborts before decoding when the signal is already aborted', async () => {
    const { file } = sampleFile();
    const decoder = rawBlockDecoder();
    const controller = new AbortController();
    controller.abort('gone');
    const reader = createSeekableReader(file);
    await expect(reader.readWindow({ start: 0, length: 1, decoder, signal: controller.signal }))
      .rejects.toBeInstanceOf(SeekableWindowAbortError);
    expect(decoder.decodeFrame).not.toHaveBeenCalled();
  });
});
