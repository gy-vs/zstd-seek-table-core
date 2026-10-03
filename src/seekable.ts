import {
  DictionaryNotFoundError,
  createDictionaryFrameResolver,
  type DictionaryFrameResolver,
  type InjectedFrameDecoder,
  type VersionedDictionaryRepository,
} from './dictionary.js';
import {
  SKIPPABLE_FRAME_HEADER_SIZE,
  isSkippableFrameMagic,
  parseFrameHeader,
  readUint32LE,
} from './frames.js';

export const SEEK_TABLE_MAGIC = 0x8f92eab1;
export const SEEK_TABLE_SKIPPABLE_MAGIC = 0x184d2a5e;
export const SEEK_TABLE_FOOTER_SIZE = 9;
export const SEEK_TABLE_ENTRY_SIZE = 8;
export const SEEK_TABLE_CHECKSUM_SIZE = 4;

const SEEK_TABLE_CHECKSUM_FLAG = 0x80;
const SEEK_TABLE_RESERVED_BITS = 0x60;

export interface SeekTableEntry {
  readonly frameIndex: number;
  /** Half-open interval [compressedOffset, compressedEnd) of the frame in the compressed stream. */
  readonly compressedOffset: bigint;
  readonly compressedEnd: bigint;
  /** Half-open interval [decompressedOffset, decompressedEnd) of the frame in the decompressed stream. */
  readonly decompressedOffset: bigint;
  readonly decompressedEnd: bigint;
  /** Low 32 bits of the XXH64 checksum of the decompressed data, when the table carries checksums. */
  readonly checksum: number | null;
}

export interface SeekTableIndex {
  readonly entries: readonly SeekTableEntry[];
  readonly frameCount: number;
  readonly checksumFlag: boolean;
  /** Offset of the skippable frame carrying the seek table; equal to totalCompressedSize. */
  readonly tableOffset: bigint;
  /** Total size of that skippable frame (header + payload). Never part of any frame interval. */
  readonly tableSize: bigint;
  readonly totalCompressedSize: bigint;
  readonly totalDecompressedSize: bigint;
}

export type SeekTableErrorCode =
  | 'NOT_SEEKABLE'
  | 'SEEK_TABLE_RESERVED_BITS'
  | 'SEEK_TABLE_OUT_OF_BOUNDS'
  | 'SEEK_TABLE_SKIPPABLE_MAGIC'
  | 'SEEK_TABLE_SIZE_MISMATCH'
  | 'SEEK_TABLE_SPAN_MISMATCH';

export class SeekTableError extends Error {
  readonly code: SeekTableErrorCode;

  constructor(code: SeekTableErrorCode, message: string) {
    super(message);
    this.name = 'SeekTableError';
    this.code = code;
  }
}

/**
 * Locates the seek table from the tail of a seekable zstd file and validates it:
 * footer magic, descriptor reserved bits, entry count against the skippable frame's
 * declared payload size, the dedicated skippable magic 0x184D2A5E, and the invariant
 * that data frames plus the table exactly span the input. Non-seekable input fails
 * with NOT_SEEKABLE; no sequential scan is ever attempted.
 */
export function parseSeekTable(data: Uint8Array): SeekTableIndex {
  if (!(data instanceof Uint8Array)) {
    throw new TypeError('Seek table input must be a Uint8Array');
  }
  const fileSize = data.length;
  if (fileSize < SEEK_TABLE_FOOTER_SIZE + SKIPPABLE_FRAME_HEADER_SIZE) {
    throw new SeekTableError(
      'NOT_SEEKABLE',
      `Input of ${fileSize} bytes is too small to hold a seek table; refusing to scan for one`,
    );
  }

  if (readUint32LE(data, fileSize - 4) !== SEEK_TABLE_MAGIC) {
    throw new SeekTableError(
      'NOT_SEEKABLE',
      'Input does not end with the seek table magic 0x8F92EAB1; it is not a seekable zstd file and no sequential scan was attempted',
    );
  }

  const descriptor = data[fileSize - 5]!;
  if ((descriptor & SEEK_TABLE_RESERVED_BITS) !== 0) {
    throw new SeekTableError(
      'SEEK_TABLE_RESERVED_BITS',
      `Seek table descriptor 0x${descriptor.toString(16).padStart(2, '0')} has reserved bits 5-6 set`,
    );
  }
  const checksumFlag = (descriptor & SEEK_TABLE_CHECKSUM_FLAG) !== 0;

  const frameCount = readUint32LE(data, fileSize - SEEK_TABLE_FOOTER_SIZE);
  const entrySize = SEEK_TABLE_ENTRY_SIZE + (checksumFlag ? SEEK_TABLE_CHECKSUM_SIZE : 0);
  // frameCount is an uint32 and entrySize is at most 12, so payloadSize stays far
  // below 2^53 and this multiplication cannot lose precision.
  const payloadSize = SEEK_TABLE_FOOTER_SIZE + frameCount * entrySize;
  const tableOffset = fileSize - SKIPPABLE_FRAME_HEADER_SIZE - payloadSize;
  if (tableOffset < 0) {
    throw new SeekTableError(
      'SEEK_TABLE_OUT_OF_BOUNDS',
      `Seek table of ${payloadSize} bytes for ${frameCount} frames does not fit in an input of ${fileSize} bytes`,
    );
  }

  const skippableMagic = readUint32LE(data, tableOffset);
  if (skippableMagic !== SEEK_TABLE_SKIPPABLE_MAGIC) {
    const similar = isSkippableFrameMagic(skippableMagic)
      ? '; it is a skippable frame, but not the dedicated seek table frame 0x184D2A5E, so its payload is not trusted as an index'
      : '';
    throw new SeekTableError(
      'SEEK_TABLE_SKIPPABLE_MAGIC',
      `Expected the seek table skippable frame magic 0x184D2A5E at offset ${tableOffset}, found 0x${skippableMagic.toString(16).padStart(8, '0')}${similar}`,
    );
  }

  const declaredPayloadSize = readUint32LE(data, tableOffset + 4);
  if (declaredPayloadSize !== payloadSize) {
    throw new SeekTableError(
      'SEEK_TABLE_SIZE_MISMATCH',
      `Seek table skippable frame declares ${declaredPayloadSize} payload bytes, but ${frameCount} entries of ${entrySize} bytes plus the ${SEEK_TABLE_FOOTER_SIZE}-byte footer require ${payloadSize}`,
    );
  }

  const entries: SeekTableEntry[] = [];
  let cursor = tableOffset + SKIPPABLE_FRAME_HEADER_SIZE;
  // Accumulate in bigint: up to 2^32 frames of up to 2^32 bytes each would overflow
  // a JS number, and a corrupted table must never wrap around into a valid-looking sum.
  let compressedOffset = 0n;
  let decompressedOffset = 0n;
  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    const compressedSize = BigInt(readUint32LE(data, cursor));
    const decompressedSize = BigInt(readUint32LE(data, cursor + 4));
    cursor += SEEK_TABLE_ENTRY_SIZE;
    let checksum: number | null = null;
    if (checksumFlag) {
      checksum = readUint32LE(data, cursor);
      cursor += SEEK_TABLE_CHECKSUM_SIZE;
    }
    entries.push(Object.freeze({
      frameIndex,
      compressedOffset,
      compressedEnd: compressedOffset + compressedSize,
      decompressedOffset,
      decompressedEnd: decompressedOffset + decompressedSize,
      checksum,
    }));
    compressedOffset += compressedSize;
    decompressedOffset += decompressedSize;
  }

  const tableSize = BigInt(SKIPPABLE_FRAME_HEADER_SIZE + payloadSize);
  if (compressedOffset + tableSize !== BigInt(fileSize)) {
    throw new SeekTableError(
      'SEEK_TABLE_SPAN_MISMATCH',
      `Data frames cover ${compressedOffset} compressed bytes and the seek table occupies ${tableSize} bytes at offset ${tableOffset}, which does not add up to the input size ${fileSize}`,
    );
  }

  return Object.freeze({
    entries: Object.freeze(entries),
    frameCount,
    checksumFlag,
    tableOffset: BigInt(tableOffset),
    tableSize,
    totalCompressedSize: compressedOffset,
    totalDecompressedSize: decompressedOffset,
  });
}

export type SeekableReadErrorCode =
  | 'WINDOW_OUT_OF_RANGE'
  | 'INDEX_ENTRY_OUT_OF_BOUNDS'
  | 'FRAME_HEADER_INVALID'
  | 'FRAME_SIZE_MISMATCH'
  | 'DECODER_RESULT_INVALID'
  | 'SEEKABLE_READ_ABORTED';

export interface SeekableReadCoordinates {
  readonly frameIndex?: number;
  readonly compressedOffset?: bigint;
  readonly compressedEnd?: bigint;
  readonly decompressedOffset?: bigint;
  readonly decompressedEnd?: bigint;
  readonly windowOffset?: bigint;
  readonly windowLength?: bigint;
}

export class SeekableReadError extends Error {
  readonly code: SeekableReadErrorCode;
  readonly frameIndex?: number;
  readonly compressedOffset?: bigint;
  readonly compressedEnd?: bigint;
  readonly decompressedOffset?: bigint;
  readonly decompressedEnd?: bigint;
  readonly windowOffset?: bigint;
  readonly windowLength?: bigint;

  constructor(
    code: SeekableReadErrorCode,
    message: string,
    coordinates: SeekableReadCoordinates = {},
    options?: { cause?: unknown },
  ) {
    super(message, options);
    this.name = 'SeekableReadError';
    this.code = code;
    this.frameIndex = coordinates.frameIndex;
    this.compressedOffset = coordinates.compressedOffset;
    this.compressedEnd = coordinates.compressedEnd;
    this.decompressedOffset = coordinates.decompressedOffset;
    this.decompressedEnd = coordinates.decompressedEnd;
    this.windowOffset = coordinates.windowOffset;
    this.windowLength = coordinates.windowLength;
  }
}

export class SeekableReadAbortError extends SeekableReadError {
  constructor(coordinates: SeekableReadCoordinates, reason?: unknown) {
    const where = coordinates.frameIndex !== undefined
      ? ` of frame ${coordinates.frameIndex}`
      : coordinates.windowOffset !== undefined
        ? ` at decompressed offset ${coordinates.windowOffset}`
        : '';
    super('SEEKABLE_READ_ABORTED', `Seekable read${where} was aborted`, coordinates, { cause: reason });
    this.name = 'AbortError';
  }
}

export interface SeekableReadRequest {
  readonly offset: number | bigint;
  readonly length: number | bigint;
  readonly signal?: AbortSignal;
}

export interface SeekableReader {
  readonly index: SeekTableIndex;
  read(request: SeekableReadRequest): Promise<Uint8Array>;
}

export interface SeekableReaderOptions {
  readonly data: Uint8Array;
  readonly decoder: InjectedFrameDecoder<Uint8Array>;
  /** Provide either a ready-made resolver or a repository to build one from. */
  readonly resolver?: DictionaryFrameResolver;
  readonly repository?: VersionedDictionaryRepository;
  /** A pre-parsed index; parsed from `data` when omitted. */
  readonly index?: SeekTableIndex;
}

interface FrameJob {
  readonly controller: AbortController;
  readonly promise: Promise<Uint8Array>;
  waiters: number;
  settled: boolean;
}

function entryCoordinates(entry: SeekTableEntry): SeekableReadCoordinates {
  return {
    frameIndex: entry.frameIndex,
    compressedOffset: entry.compressedOffset,
    compressedEnd: entry.compressedEnd,
    decompressedOffset: entry.decompressedOffset,
    decompressedEnd: entry.decompressedEnd,
  };
}

function toBigIntCoordinate(value: unknown, name: string): bigint {
  if (typeof value === 'bigint' || (typeof value === 'number' && Number.isSafeInteger(value))) {
    const coordinate = BigInt(value);
    if (coordinate < 0n) {
      throw new SeekableReadError(
        'WINDOW_OUT_OF_RANGE',
        `Seekable read ${name} must not be negative, got ${coordinate}`,
      );
    }
    return coordinate;
  }
  throw new TypeError(`Seekable read ${name} must be a non-negative safe integer or bigint, got ${String(value)}`);
}

function validateIndex(index: SeekTableIndex): void {
  if (!index || !Array.isArray(index.entries)) {
    throw new TypeError('Seek table index must provide an entries array');
  }
  let compressed = 0n;
  let decompressed = 0n;
  index.entries.forEach((entry, position) => {
    if (
      entry.frameIndex !== position
      || typeof entry.compressedOffset !== 'bigint'
      || typeof entry.compressedEnd !== 'bigint'
      || typeof entry.decompressedOffset !== 'bigint'
      || typeof entry.decompressedEnd !== 'bigint'
      || entry.compressedOffset !== compressed
      || entry.compressedEnd < compressed
      || entry.decompressedOffset !== decompressed
      || entry.decompressedEnd < decompressed
    ) {
      throw new TypeError(
        `Seek table index entry ${position} does not continue the compressed and decompressed streams contiguously from 0`,
      );
    }
    compressed = entry.compressedEnd;
    decompressed = entry.decompressedEnd;
  });
  if (index.totalCompressedSize !== compressed || index.totalDecompressedSize !== decompressed) {
    throw new TypeError('Seek table index totals do not match its entries');
  }
}

/**
 * Creates a random-access reader over a seekable zstd file. A read only decodes the
 * data frames intersecting the requested decompressed window; the injected decoder
 * receives the boundary-validated compressed bytes of exactly one frame plus its
 * frame index, and its result must match the decompressed size declared by the seek
 * table or the read fails instead of returning truncated or cross-frame data.
 *
 * Decode sharing: concurrent reads that hit the same frame share one in-flight
 * decode. The shared job is evicted as soon as it settles and is never reused by a
 * later read, so every new read resolves its dictionary lease at the repository's
 * current revision — a dictionary update can neither disturb an in-flight read (its
 * lease is pinned by the resolver) nor leak stale content into a post-update read.
 */
export function createSeekableReader(options: SeekableReaderOptions): SeekableReader {
  if (options == null || typeof options !== 'object') {
    throw new TypeError('Seekable reader options are required');
  }
  const { data, decoder } = options;
  if (!(data instanceof Uint8Array)) {
    throw new TypeError('Seekable reader data must be a Uint8Array');
  }
  if (!decoder || typeof decoder.decodeFrame !== 'function') {
    throw new TypeError('Seekable reader decoder must implement decodeFrame');
  }
  const index = options.index ?? parseSeekTable(data);
  validateIndex(index);
  const resolver = options.resolver ?? createDictionaryFrameResolver(
    options.repository ?? {
      loadDictionary: (id: number) => Promise.reject(new DictionaryNotFoundError(id)),
    },
  );
  const jobs = new Map<number, FrameJob>();
  const neverAborted = new AbortController();

  async function decodeAndValidate(entry: SeekTableEntry, signal: AbortSignal): Promise<Uint8Array> {
    const coordinates = entryCoordinates(entry);
    if (
      entry.compressedOffset < 0n
      || entry.compressedEnd < entry.compressedOffset
      || entry.compressedEnd > BigInt(data.length)
    ) {
      throw new SeekableReadError(
        'INDEX_ENTRY_OUT_OF_BOUNDS',
        `Frame ${entry.frameIndex} compressed interval [${entry.compressedOffset}, ${entry.compressedEnd}) escapes the ${data.length}-byte input`,
        coordinates,
      );
    }
    const compressed = data.subarray(Number(entry.compressedOffset), Number(entry.compressedEnd));
    const header = parseFrameHeader(compressed, 0);
    if (header === null) {
      throw new SeekableReadError(
        'FRAME_HEADER_INVALID',
        `Frame ${entry.frameIndex} at compressed offset ${entry.compressedOffset} does not start with a valid zstd frame header`,
        coordinates,
      );
    }
    const decoded = await resolver.resolveFrame<Uint8Array>({
      frameId: entry.frameIndex,
      data: compressed,
      dictionary: header.dictionary,
      decoder,
      signal,
    });
    if (!(decoded instanceof Uint8Array)) {
      throw new SeekableReadError(
        'DECODER_RESULT_INVALID',
        `Decoder for frame ${entry.frameIndex} did not return a Uint8Array`,
        coordinates,
      );
    }
    const declared = entry.decompressedEnd - entry.decompressedOffset;
    if (BigInt(decoded.length) !== declared) {
      throw new SeekableReadError(
        'FRAME_SIZE_MISMATCH',
        `Frame ${entry.frameIndex} declares ${declared} decompressed bytes in the seek table, but the decoder returned ${decoded.length}; the result is discarded instead of being truncated or padded from the next frame`,
        coordinates,
      );
    }
    return decoded;
  }

  function frameBytes(entry: SeekTableEntry, signal: AbortSignal): Promise<Uint8Array> {
    let job = jobs.get(entry.frameIndex);
    if (job === undefined) {
      const controller = new AbortController();
      const created: FrameJob = {
        controller,
        promise: decodeAndValidate(entry, controller.signal),
        waiters: 0,
        settled: false,
      };
      const evict = () => {
        created.settled = true;
        if (jobs.get(entry.frameIndex) === created) {
          jobs.delete(entry.frameIndex);
        }
      };
      created.promise.then(evict, evict);
      jobs.set(entry.frameIndex, created);
      job = created;
    }
    const active = job;
    active.waiters += 1;
    return new Promise<Uint8Array>((resolve, reject) => {
      let released = false;
      const release = () => {
        if (released) {
          return;
        }
        released = true;
        signal.removeEventListener('abort', onAbort);
        active.waiters -= 1;
        if (active.waiters === 0 && !active.settled) {
          active.controller.abort(signal.reason);
        }
      };
      const onAbort = () => {
        release();
        reject(new SeekableReadAbortError(entryCoordinates(entry), signal.reason));
      };
      if (signal.aborted) {
        onAbort();
        return;
      }
      signal.addEventListener('abort', onAbort, { once: true });
      active.promise.then(
        (bytes) => {
          release();
          resolve(bytes);
        },
        (error) => {
          release();
          reject(error);
        },
      );
    });
  }

  async function read(request: SeekableReadRequest): Promise<Uint8Array> {
    const offset = toBigIntCoordinate(request?.offset, 'offset');
    const length = toBigIntCoordinate(request?.length, 'length');
    const signal = request.signal ?? neverAborted.signal;
    if (!(signal instanceof AbortSignal)) {
      throw new TypeError('Seekable read signal must be an AbortSignal');
    }
    const windowCoordinates = { windowOffset: offset, windowLength: length };
    if (signal.aborted) {
      throw new SeekableReadAbortError(windowCoordinates, signal.reason);
    }
    const total = index.totalDecompressedSize;
    if (offset > total || length > total - offset) {
      throw new SeekableReadError(
        'WINDOW_OUT_OF_RANGE',
        `Window [${offset}, ${offset + length}) escapes the decompressed stream of ${total} bytes`,
        windowCoordinates,
      );
    }
    if (length > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new SeekableReadError(
        'WINDOW_OUT_OF_RANGE',
        `Window of ${length} bytes cannot be materialized as a Uint8Array`,
        windowCoordinates,
      );
    }
    if (length === 0n) {
      return new Uint8Array(0);
    }

    const windowEnd = offset + length;
    const hits: SeekTableEntry[] = [];
    for (const entry of index.entries) {
      if (entry.decompressedEnd <= entry.decompressedOffset) {
        continue; // empty frames contribute no bytes to any window
      }
      if (entry.decompressedEnd <= offset) {
        continue;
      }
      if (entry.decompressedOffset >= windowEnd) {
        break;
      }
      hits.push(entry);
    }

    const output = new Uint8Array(Number(length));
    for (const entry of hits) {
      const bytes = await frameBytes(entry, signal);
      const sliceStart = offset > entry.decompressedOffset ? offset - entry.decompressedOffset : 0n;
      const sliceEnd = windowEnd < entry.decompressedEnd
        ? windowEnd - entry.decompressedOffset
        : entry.decompressedEnd - entry.decompressedOffset;
      const target = entry.decompressedOffset + sliceStart - offset;
      output.set(bytes.subarray(Number(sliceStart), Number(sliceEnd)), Number(target));
    }
    return output;
  }

  return Object.freeze({ index, read });
}
