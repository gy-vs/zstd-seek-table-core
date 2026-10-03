import { describe, expect, it, vi } from 'vitest';
import {
  APPLICATION_DICTIONARY,
  InMemoryVersionedDictionaryRepository,
  NO_DICTIONARY,
  SeekTableError,
  SeekableReadError,
  createDictionaryFrameResolver,
  createSeekableReader,
  explicitDictionary,
  parseFrameHeader,
  parseSeekTable,
  readBlockHeader,
  type FrameDecoderContext,
  type ResolvedDictionary,
} from '../src/index.js';

interface Deferred<T> {
  promise: Promise<T>;
  resolve(value: T | PromiseLike<T>): void;
  reject(reason?: unknown): void;
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolveValue, rejectValue) => {
    resolve = resolveValue;
    reject = rejectValue;
  });
  return { promise, resolve, reject };
}

const flush = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

const text = (value: string) => new Uint8Array([...value].map((char) => char.charCodeAt(0)));
const asString = (bytes: Uint8Array) => String.fromCharCode(...bytes);

function concat(parts: readonly Uint8Array[]): Uint8Array {
  const output = new Uint8Array(parts.reduce((total, part) => total + part.length, 0));
  let cursor = 0;
  for (const part of parts) {
    output.set(part, cursor);
    cursor += part.length;
  }
  return output;
}

interface RawFrameOptions {
  readonly dictionaryIdFlag?: 0 | 1 | 2 | 3;
  readonly dictionaryId?: number;
}

/** Builds a genuine single-segment zstd frame carrying one raw (uncompressed) block. */
function rawBlockFrame(payload: Uint8Array, options: RawFrameOptions = {}): Uint8Array {
  if (payload.length > 255) {
    throw new RangeError('test frames use the one-byte content size field');
  }
  const flag = options.dictionaryIdFlag ?? 0;
  const dictionaryId = options.dictionaryId ?? 0;
  const dictionaryBytes: number[] = [];
  for (let index = 0; index < [0, 1, 2, 4][flag]!; index += 1) {
    dictionaryBytes.push((dictionaryId >>> (8 * index)) & 0xff);
  }
  const descriptor = 0x20 | flag; // single segment, one-byte content size field
  const blockHeaderValue = 1 | (payload.length << 3); // last block, raw type
  return concat([
    new Uint8Array([0x28, 0xb5, 0x2f, 0xfd, descriptor, ...dictionaryBytes, payload.length]),
    new Uint8Array([blockHeaderValue & 0xff, (blockHeaderValue >> 8) & 0xff, (blockHeaderValue >> 16) & 0xff]),
    payload,
  ]);
}

interface SeekTableOptions {
  readonly checksumFlag?: boolean;
  readonly checksum?: number;
}

/** Builds a seekable file: data frames, then the seek table in its skippable frame. */
function seekableFile(
  frames: readonly Uint8Array[],
  decompressedSizes: readonly number[],
  options: SeekTableOptions = {},
): Uint8Array {
  const checksumFlag = options.checksumFlag ?? false;
  const entrySize = checksumFlag ? 12 : 8;
  const table = new Uint8Array(frames.length * entrySize + 9);
  const view = new DataView(table.buffer);
  frames.forEach((frame, frameIndex) => {
    const entryOffset = frameIndex * entrySize;
    view.setUint32(entryOffset, frame.length, true);
    view.setUint32(entryOffset + 4, decompressedSizes[frameIndex]!, true);
    if (checksumFlag) {
      view.setUint32(entryOffset + 8, options.checksum ?? 0, true);
    }
  });
  const footerOffset = frames.length * entrySize;
  view.setUint32(footerOffset, frames.length, true);
  table[footerOffset + 4] = checksumFlag ? 0x80 : 0;
  view.setUint32(footerOffset + 5, 0x8f92eab1, true);
  const skippableHeader = new Uint8Array(8);
  const skippableView = new DataView(skippableHeader.buffer);
  skippableView.setUint32(0, 0x184d2a5e, true);
  skippableView.setUint32(4, table.length, true);
  return concat([...frames, skippableHeader, table]);
}

/** A real (if minimal) decoder for raw-block frames, built on the framing kernel. */
function extractRawBlocks(data: Uint8Array): Uint8Array {
  const header = parseFrameHeader(data, 0);
  if (header === null) {
    throw new Error('test decoder received a frame without a valid header');
  }
  const output: number[] = [];
  let cursor = header.headerSize;
  for (;;) {
    const block = readBlockHeader(data.subarray(cursor));
    if (block === null) {
      throw new Error('test decoder hit a truncated block header');
    }
    cursor += 3;
    if (block.type !== 0) {
      throw new Error(`test decoder only supports raw blocks, got type ${block.type}`);
    }
    output.push(...data.subarray(cursor, cursor + block.size));
    cursor += block.size;
    if (block.last) {
      return new Uint8Array(output);
    }
  }
}

function rawDecoder(
  implementation?: (context: FrameDecoderContext, dictionary: ResolvedDictionary) => Uint8Array | Promise<Uint8Array>,
) {
  const decodeFrame = vi.fn(
    implementation ?? ((context: FrameDecoderContext) => extractRawBlocks(context.data)),
  );
  return { decodeFrame, mock: decodeFrame.mock };
}

const PAYLOADS = ['hello ', 'seekable ', 'world'].map(text);

function standardFile(options?: SeekTableOptions) {
  const frames = PAYLOADS.map((payload) => rawBlockFrame(payload));
  const file = seekableFile(frames, PAYLOADS.map((payload) => payload.length), options);
  return { frames, file };
}

function parseError(data: Uint8Array): SeekTableError {
  try {
    parseSeekTable(data);
  } catch (error) {
    expect(error).toBeInstanceOf(SeekTableError);
    return error as SeekTableError;
  }
  throw new Error('parseSeekTable should have thrown');
}

async function failureOf(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error('expected the promise to reject');
}

function patched(data: Uint8Array, offset: number, ...bytes: number[]): Uint8Array {
  const copy = data.slice();
  copy.set(bytes, offset);
  return copy;
}

function patchedUint32(data: Uint8Array, offset: number, value: number): Uint8Array {
  const copy = data.slice();
  new DataView(copy.buffer).setUint32(offset, value, true);
  return copy;
}

describe('parseSeekTable', () => {
  it('parses a table without checksums into half-open bigint intervals', () => {
    const { file, frames } = standardFile();
    const index = parseSeekTable(file);

    expect(index.frameCount).toBe(3);
    expect(index.checksumFlag).toBe(false);
    let compressed = 0n;
    let decompressed = 0n;
    index.entries.forEach((entry, frameIndex) => {
      expect(entry.frameIndex).toBe(frameIndex);
      expect(entry.compressedOffset).toBe(compressed);
      expect(entry.compressedEnd).toBe(compressed + BigInt(frames[frameIndex]!.length));
      expect(entry.decompressedOffset).toBe(decompressed);
      expect(entry.decompressedEnd).toBe(decompressed + BigInt(PAYLOADS[frameIndex]!.length));
      expect(entry.checksum).toBeNull();
      compressed = entry.compressedEnd;
      decompressed = entry.decompressedEnd;
    });
    expect(index.totalCompressedSize).toBe(compressed);
    expect(index.totalDecompressedSize).toBe(20n);
    // The seek table bytes are not part of any data frame interval.
    expect(index.tableOffset).toBe(compressed);
    expect(index.tableOffset + index.tableSize).toBe(BigInt(file.length));
  });

  it('parses a table with per-frame checksum fields', () => {
    const { file } = standardFile({ checksumFlag: true, checksum: 0x0badf00d });
    const index = parseSeekTable(file);

    expect(index.checksumFlag).toBe(true);
    expect(index.entries.map((entry) => entry.checksum)).toEqual([0x0badf00d, 0x0badf00d, 0x0badf00d]);
    expect(index.totalDecompressedSize).toBe(20n);
  });

  it('rejects non-seekable input instead of falling back to a sequential scan', () => {
    expect(parseError(text('definitely not a seekable zstd file........')).code).toBe('NOT_SEEKABLE');
    expect(parseError(text('tiny')).code).toBe('NOT_SEEKABLE');
    // A bare concatenation of data frames without a table fails explicitly.
    const { frames } = standardFile();
    expect(parseError(concat(frames)).code).toBe('NOT_SEEKABLE');
  });

  it('rejects reserved descriptor bits but tolerates unused ones', () => {
    const { file } = standardFile();
    expect(parseError(patched(file, file.length - 5, 0x40)).code).toBe('SEEK_TABLE_RESERVED_BITS');
    expect(parseError(patched(file, file.length - 5, 0x20)).code).toBe('SEEK_TABLE_RESERVED_BITS');
    expect(() => parseSeekTable(patched(file, file.length - 5, 0x1f))).not.toThrow();
  });

  it('rejects a table wrapped in a similar but wrong skippable magic', () => {
    const { file } = standardFile();
    const tableOffset = Number(parseSeekTable(file).tableOffset);
    const error = parseError(patchedUint32(file, tableOffset, 0x184d2a50));
    expect(error.code).toBe('SEEK_TABLE_SKIPPABLE_MAGIC');
    expect(error.message).toContain('0x184d2a50');
  });

  it('rejects declared table lengths that disagree with the entry count', () => {
    const { file } = standardFile();
    const index = parseSeekTable(file);
    const tableOffset = Number(index.tableOffset);

    // Skippable frame declares a payload size that does not match the entry count.
    expect(parseError(patchedUint32(file, tableOffset + 4, Number(index.tableSize) - 8 + 4)).code)
      .toBe('SEEK_TABLE_SIZE_MISMATCH');

    // Physically shrink the entries to two but keep the stale declared payload size.
    const entryStart = tableOffset + 8;
    const shrunk = concat([file.subarray(0, entryStart + 16), file.subarray(entryStart + 24)]);
    expect(parseError(patchedUint32(shrunk, shrunk.length - 9, 2)).code).toBe('SEEK_TABLE_SIZE_MISMATCH');

    // Same inconsistency through the 12-byte checksum entry layout.
    const { file: checksummed } = standardFile({ checksumFlag: true });
    const checksummedOffset = Number(parseSeekTable(checksummed).tableOffset);
    const checksummedEntries = checksummedOffset + 8;
    const shrunkChecksummed = concat([
      checksummed.subarray(0, checksummedEntries + 24),
      checksummed.subarray(checksummedEntries + 36),
    ]);
    expect(parseError(patchedUint32(shrunkChecksummed, shrunkChecksummed.length - 9, 2)).code)
      .toBe('SEEK_TABLE_SIZE_MISMATCH');

    // A zeroed entry count shifts the computed table start into entry data, so the
    // dedicated skippable magic is no longer where it must be.
    expect(parseError(patchedUint32(file, file.length - 9, 0)).code).toBe('SEEK_TABLE_SKIPPABLE_MAGIC');
  });

  it('rejects malicious frame counts without oversized allocation or overflow', () => {
    const { file } = standardFile();
    expect(parseError(patchedUint32(file, file.length - 9, 0xffffffff)).code)
      .toBe('SEEK_TABLE_OUT_OF_BOUNDS');
    // One extra frame shifts the computed table start into entry data.
    expect(parseError(patchedUint32(file, file.length - 9, 4)).code)
      .toBe('SEEK_TABLE_SKIPPABLE_MAGIC');
  });

  it('rejects tables whose frames do not exactly span the input', () => {
    const { file, frames } = standardFile();
    const tableOffset = Number(parseSeekTable(file).tableOffset);
    const grown = patchedUint32(file, tableOffset + 8, frames[0]!.length + 1);
    expect(parseError(grown).code).toBe('SEEK_TABLE_SPAN_MISMATCH');
  });
});

describe('parseFrameHeader', () => {
  it('locates optional fields in single-segment frames', () => {
    const frame = rawBlockFrame(text('abc'), { dictionaryIdFlag: 1, dictionaryId: 7 });
    const header = parseFrameHeader(frame, 0);

    expect(header).toMatchObject({
      windowDescriptor: null,
      dictionaryIdFieldLength: 1,
      contentSizeFieldLength: 1,
      headerSize: 7,
    });
    expect(header?.dictionary).toEqual(explicitDictionary(7));
    expect(header?.contentSize).toBe(3n);
  });

  it('accounts for the window descriptor before the dictionary id', () => {
    // magic, descriptor (2-byte dictionary id, not single segment), window descriptor, id
    const frame = new Uint8Array([0x28, 0xb5, 0x2f, 0xfd, 0x02, 0xab, 0x34, 0x12]);
    const header = parseFrameHeader(frame, 0);

    expect(header?.windowDescriptor).toBe(0xab);
    expect(header?.dictionary).toEqual(explicitDictionary(0x1234));
    expect(header?.contentSize).toBeNull();
    expect(header?.headerSize).toBe(8);
  });

  it('distinguishes a missing dictionary field from a zero-valued one', () => {
    expect(parseFrameHeader(rawBlockFrame(text('x')), 0)?.dictionary).toBe(NO_DICTIONARY);
    expect(parseFrameHeader(rawBlockFrame(text('x'), { dictionaryIdFlag: 1, dictionaryId: 0 }), 0)?.dictionary)
      .toBe(APPLICATION_DICTIONARY);
  });

  it('returns null for bad magic and truncated headers', () => {
    expect(parseFrameHeader(text('nope'))).toBeNull();
    // Declares a 4-byte dictionary id plus a 1-byte content size but is too short.
    expect(parseFrameHeader(new Uint8Array([0x28, 0xb5, 0x2f, 0xfd, 0x23, 0x01]))).toBeNull();
  });
});

describe('seekable reader', () => {
  it('fails fast on non-seekable input instead of scanning', () => {
    try {
      createSeekableReader({ data: text('plain bytes'), decoder: rawDecoder() });
      expect.unreachable();
    } catch (error) {
      expect((error as SeekTableError).code).toBe('NOT_SEEKABLE');
    }
  });

  it('reads a window spanning frames and decodes only those frames', async () => {
    const { file, frames } = standardFile();
    const decoder = rawDecoder();
    const reader = createSeekableReader({ data: file, decoder });

    expect(asString(await reader.read({ offset: 4, length: 10 }))).toBe('o seekable');
    expect(decoder.mock.calls.map((call) => call[0].frameId)).toEqual([0, 1]);
    // The decoder received exactly the indexed compressed bytes of each frame.
    expect(decoder.mock.calls[0]?.[0].data).toEqual(frames[0]);
    expect(decoder.mock.calls[1]?.[0].data).toEqual(frames[1]);
  });

  it('returns empty zero-length windows without touching any frame', async () => {
    const { file } = standardFile();
    const decoder = rawDecoder();
    const repository = new InMemoryVersionedDictionaryRepository();
    const loadSpy = vi.spyOn(repository, 'loadDictionary');
    const reader = createSeekableReader({ data: file, decoder, repository });

    expect(await reader.read({ offset: 5, length: 0 })).toEqual(new Uint8Array(0));
    expect(await reader.read({ offset: 20, length: 0 })).toEqual(new Uint8Array(0));
    expect(decoder.mock.calls).toHaveLength(0);
    expect(loadSpy).not.toHaveBeenCalled();
  });

  it('reads the tail end and rejects out-of-range windows with coordinates', async () => {
    const { file } = standardFile();
    const reader = createSeekableReader({ data: file, decoder: rawDecoder() });

    expect(asString(await reader.read({ offset: 15n, length: 5n }))).toBe('world');
    expect(asString(await reader.read({ offset: 19, length: 1 }))).toBe('d');

    const pastEnd = await failureOf(reader.read({ offset: 21, length: 0 }));
    expect(pastEnd).toBeInstanceOf(SeekableReadError);
    expect(pastEnd).toMatchObject({ code: 'WINDOW_OUT_OF_RANGE', windowOffset: 21n, windowLength: 0n });
    await expect(reader.read({ offset: 20, length: 1 })).rejects.toMatchObject({ code: 'WINDOW_OUT_OF_RANGE' });
  });

  it('validates window coordinates before touching any frame', async () => {
    const { file } = standardFile();
    const decoder = rawDecoder();
    const reader = createSeekableReader({ data: file, decoder });
    const controller = new AbortController();
    controller.abort('gone');

    await expect(reader.read({ offset: -1, length: 1 })).rejects.toMatchObject({ code: 'WINDOW_OUT_OF_RANGE' });
    await expect(reader.read({ offset: 1.5, length: 1 })).rejects.toBeInstanceOf(TypeError);
    await expect(reader.read({ offset: 0, length: 1, signal: controller.signal }))
      .rejects.toMatchObject({ code: 'SEEKABLE_READ_ABORTED', name: 'AbortError' });
    expect(decoder.mock.calls).toHaveLength(0);
  });

  it('reads windows from a table with per-frame checksums', async () => {
    const { file } = standardFile({ checksumFlag: true, checksum: 0x01020304 });
    const decoder = rawDecoder();
    const reader = createSeekableReader({ data: file, decoder });

    expect(reader.index.entries[2]?.checksum).toBe(0x01020304);
    expect(asString(await reader.read({ offset: 6, length: 9 }))).toBe('seekable ');
    expect(decoder.mock.calls.map((call) => call[0].frameId)).toEqual([1]);
  });

  it('refuses truncated or oversized decode results instead of mixing frames', async () => {
    const { file } = standardFile();

    const shortReader = createSeekableReader({
      data: file,
      decoder: rawDecoder((context) => extractRawBlocks(context.data).subarray(1)),
    });
    const failure = await failureOf(shortReader.read({ offset: 6, length: 9 }));
    expect(failure).toBeInstanceOf(SeekableReadError);
    expect(failure).toMatchObject({
      code: 'FRAME_SIZE_MISMATCH',
      frameIndex: 1,
      compressedOffset: shortReader.index.entries[1]!.compressedOffset,
      compressedEnd: shortReader.index.entries[1]!.compressedEnd,
      decompressedOffset: shortReader.index.entries[1]!.decompressedOffset,
      decompressedEnd: shortReader.index.entries[1]!.decompressedEnd,
    });

    const longReader = createSeekableReader({
      data: file,
      decoder: rawDecoder((context) => concat([extractRawBlocks(context.data), new Uint8Array([0])])),
    });
    await expect(longReader.read({ offset: 0, length: 6 }))
      .rejects.toMatchObject({ code: 'FRAME_SIZE_MISMATCH', frameIndex: 0 });
  });

  it('keeps no-dictionary, application dictionary, and explicit dictionary frames distinct', async () => {
    const payloads = ['aa', 'bb', 'cc'].map(text);
    const frames = [
      rawBlockFrame(payloads[0]!),
      rawBlockFrame(payloads[1]!, { dictionaryIdFlag: 1, dictionaryId: 0 }),
      rawBlockFrame(payloads[2]!, { dictionaryIdFlag: 1, dictionaryId: 7 }),
    ];
    const file = seekableFile(frames, [2, 2, 2]);
    const repository = new InMemoryVersionedDictionaryRepository();
    repository.add({ id: 7, revision: 'rev-1', content: text('dictionary') });
    const loadSpy = vi.spyOn(repository, 'loadDictionary');
    const seen: ResolvedDictionary[] = [];
    const decoder = rawDecoder((context, dictionary) => {
      seen.push(dictionary);
      return extractRawBlocks(context.data);
    });
    const reader = createSeekableReader({ data: file, decoder, repository });

    expect(asString(await reader.read({ offset: 0, length: 6 }))).toBe('aabbcc');
    expect(loadSpy).toHaveBeenCalledTimes(1);
    expect(loadSpy).toHaveBeenCalledWith(7, expect.objectContaining({ signal: expect.any(AbortSignal) }));
    expect(seen[0]).toEqual({ kind: 'none' });
    expect(seen[1]).toEqual({ kind: 'application', id: 0 });
    expect(seen[2]).toMatchObject({ kind: 'explicit', lease: { id: 7, revision: 'rev-1' } });
    expect(repository.activeLeaseCount).toBe(0);
  });

  it('shares one in-flight decode between parallel windows hitting the same frame', async () => {
    const { file } = standardFile();
    const gate = deferred<void>();
    const decoder = rawDecoder(async (context) => {
      await gate.promise;
      return extractRawBlocks(context.data);
    });
    const reader = createSeekableReader({ data: file, decoder });

    const first = reader.read({ offset: 4, length: 4 });
    const second = reader.read({ offset: 5, length: 3 });
    await flush();
    expect(decoder.mock.calls.map((call) => call[0].frameId)).toEqual([0]);

    gate.resolve();
    expect(asString(await first)).toBe('o se');
    expect(asString(await second)).toBe(' se');
    expect(decoder.mock.calls.map((call) => call[0].frameId)).toEqual([0, 1]);
  });

  it('keeps the shared decode for the surviving window when one window cancels', async () => {
    const { file } = standardFile();
    const gate = deferred<void>();
    const decoder = rawDecoder(async (context) => {
      await gate.promise;
      return extractRawBlocks(context.data);
    });
    const reader = createSeekableReader({ data: file, decoder });
    const controller = new AbortController();

    const cancelled = reader.read({ offset: 4, length: 4, signal: controller.signal });
    const kept = reader.read({ offset: 5, length: 3 });
    await flush();
    controller.abort('not needed');

    const failure = await failureOf(cancelled);
    expect(failure).toBeInstanceOf(SeekableReadError);
    expect(failure).toMatchObject({ code: 'SEEKABLE_READ_ABORTED', name: 'AbortError', cause: 'not needed' });

    gate.resolve();
    expect(asString(await kept)).toBe(' se');
    expect(decoder.mock.calls.map((call) => call[0].frameId)).toEqual([0, 1]);
  });

  it('aborts the underlying decode when the last waiter cancels and still releases the lease', async () => {
    const frames = [rawBlockFrame(text('data'), { dictionaryIdFlag: 1, dictionaryId: 7 })];
    const file = seekableFile(frames, [4]);
    const repository = new InMemoryVersionedDictionaryRepository();
    repository.add({ id: 7, revision: 'rev-1', content: text('dictionary') });
    const gate = deferred<void>();
    const seenSignals: AbortSignal[] = [];
    const decoder = rawDecoder(async (context) => {
      seenSignals.push(context.signal);
      await gate.promise;
      return extractRawBlocks(context.data);
    });
    const reader = createSeekableReader({ data: file, decoder, repository });
    const controller = new AbortController();

    const pending = reader.read({ offset: 0, length: 4, signal: controller.signal });
    await flush();
    expect(repository.activeLeaseCount).toBe(1);

    controller.abort('gone');
    const failure = await failureOf(pending);
    expect(failure).toMatchObject({ code: 'SEEKABLE_READ_ABORTED', frameIndex: 0, cause: 'gone' });
    expect(seenSignals[0]?.aborted).toBe(true);

    gate.resolve();
    await flush();
    expect(repository.activeLeaseCount).toBe(0);
  });

  it('pins the dictionary revision for an in-flight read and reloads after an update', async () => {
    const frames = [rawBlockFrame(text('data'), { dictionaryIdFlag: 1, dictionaryId: 7 })];
    const file = seekableFile(frames, [4]);
    const repository = new InMemoryVersionedDictionaryRepository();
    repository.add({ id: 7, revision: 'rev-old', content: text('old') });
    const resolver = createDictionaryFrameResolver(repository);
    const revisions: unknown[] = [];
    const gate = deferred<void>();
    const decoder = rawDecoder(async (context, dictionary) => {
      revisions.push(dictionary.kind === 'explicit' ? dictionary.lease.revision : null);
      await gate.promise;
      return extractRawBlocks(context.data);
    });
    const reader = createSeekableReader({ data: file, decoder, resolver });

    const first = reader.read({ offset: 0, length: 4 });
    await flush();
    repository.publishUpdate({ id: 7, revision: 'rev-new', content: text('new') });
    gate.resolve();
    expect(asString(await first)).toBe('data');
    expect(revisions).toEqual(['rev-old']);
    expect(repository.activeLeaseCount).toBe(0);

    // A settled decode is never reused: the next read resolves a fresh lease.
    expect(asString(await reader.read({ offset: 0, length: 4 }))).toBe('data');
    expect(revisions).toEqual(['rev-old', 'rev-new']);
    expect(repository.activeLeaseCount).toBe(0);
  });

  it('does not let a failed frame decode poison other windows or later reads', async () => {
    const frames = [
      rawBlockFrame(text('good'), { dictionaryIdFlag: 1, dictionaryId: 7 }),
      rawBlockFrame(text('bad!'), { dictionaryIdFlag: 1, dictionaryId: 7 }),
    ];
    const file = seekableFile(frames, [4, 4]);
    const repository = new InMemoryVersionedDictionaryRepository();
    repository.add({ id: 7, revision: 'rev-1', content: text('dict') });
    let failFrame1 = true;
    const decoder = rawDecoder((context) => {
      if (context.frameId === 1 && failFrame1) {
        failFrame1 = false;
        throw new Error('boom');
      }
      return extractRawBlocks(context.data);
    });
    const reader = createSeekableReader({ data: file, decoder, repository });

    await expect(reader.read({ offset: 4, length: 4 })).rejects.toThrow('boom');
    expect(repository.activeLeaseCount).toBe(0);

    expect(asString(await reader.read({ offset: 0, length: 4 }))).toBe('good');
    expect(asString(await reader.read({ offset: 4, length: 4 }))).toBe('bad!');
    expect(repository.activeLeaseCount).toBe(0);
  });

  it('propagates dictionary resolution failures with the frame identity', async () => {
    const frames = [rawBlockFrame(text('data'), { dictionaryIdFlag: 1, dictionaryId: 99 })];
    const file = seekableFile(frames, [4]);
    const reader = createSeekableReader({
      data: file,
      decoder: rawDecoder(),
      repository: new InMemoryVersionedDictionaryRepository(),
    });

    const failure = await failureOf(reader.read({ offset: 0, length: 4 }));
    expect(failure).toMatchObject({
      name: 'DictionaryResolutionError',
      code: 'DICTIONARY_NOT_FOUND',
      frameId: 0,
      dictionaryId: 99,
    });
  });
});
