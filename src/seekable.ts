import {
  createDictionaryFrameResolver,
  readDictionarySelector,
  type DictionarySelector,
  type InjectedFrameDecoder,
  type VersionedDictionaryRepository,
} from './dictionary.js';
import { parseDescriptor } from './frame-header.js';

// Zstandard frame magic (0xFD2FB528, little endian).
export const ZSTD_FRAME_MAGIC = 0xfd2fb528;
// Generic skippable frames use 0x184D2A50..0x184D2A5F: these are NOT seek tables.
export const ZSTD_SKIPPABLE_MAGIC_START = 0x184d2a50;
export const ZSTD_SKIPPABLE_MAGIC_MASK = 0xfffffff0;
// Dedicated seek table skippable frame magic (0x8F92EABF, little endian).
export const ZSTD_SEEK_TABLE_MAGIC = 0x8f92eabf;
export const ZSTD_SEEK_TABLE_MAX_ENTRIES = 0x10000000;

const FRAME_MAGIC_BYTES = [0x28, 0xb5, 0x2f, 0xfd] as const;
const SEEK_TABLE_MAGIC_BYTES = [0xbf, 0xea, 0x92, 0x8f] as const;
const SEEK_TABLE_HEADER_SIZE = 3; // descriptor byte + two reserved bytes
const SEEK_TABLE_ENTRY_SIZE = 8;
const SEEK_TABLE_ENTRY_CHECKSUM_SIZE = 12;
const COMPRESSED_SIZE_RESERVED_MASK = 0xc0000000; // top two bits are reserved
const FRAME_HEADER_DESCRIPTOR_RESERVED_MASK = 0x18; // bits 4 and 3 are reserved

function readU32Le(data: Uint8Array, offset: number): number {
  return (
    data[offset]! |
    (data[offset + 1]! << 8) |
    (data[offset + 2]! << 16) |
    (data[offset + 3]! << 24)
  ) >>> 0;
}

function readU64Le(data: Uint8Array, offset: number): bigint {
  let value = 0n;
  for (let index = 7; index >= 0; index -= 1) {
    value = (value << 8n) | BigInt(data[offset + index]!);
  }
  return value;
}

export type SeekableErrorCode =
  | 'SEEK_TABLE_NOT_FOUND'
  | 'SEEK_TABLE_TRUNCATED'
  | 'SEEK_TABLE_DESCRIPTOR_RESERVED'
  | 'SEEK_TABLE_RESERVED_BYTES'
  | 'SEEK_TABLE_BAD_ENTRY_LENGTH'
  | 'SEEK_TABLE_TOO_MANY_ENTRIES'
  | 'SEEK_TABLE_RESERVED_SIZE_BITS'
  | 'SEEK_TABLE_OFFSET_OUT_OF_BOUNDS'
  | 'SEEK_TABLE_TOTAL_LENGTH_MISMATCH'
  | 'SEEK_FRAME_OUT_OF_BOUNDS'
  | 'SEEK_FRAME_BAD_MAGIC'
  | 'SEEK_FRAME_HEADER_TRUNCATED'
  | 'SEEK_FRAME_HEADER_RESERVED_BITS'
  | 'SEEK_FRAME_SIZE_MISMATCH'
  | 'SEEK_DICTIONARY_RESOLUTION_FAILED'
  | 'SEEK_FRAME_DECODE_FAILED'
  | 'SEEK_FRAME_DECODED_SIZE_MISMATCH'
  | 'SEEK_WINDOW_BAD_ARGUMENT'
  | 'SEEK_WINDOW_OUT_OF_BOUNDS'
  | 'SEEK_WINDOW_ABORTED';

export interface SeekableCoordinates {
  readonly frameIndex?: number;
  readonly compressedStart?: bigint;
  readonly compressedEnd?: bigint;
  readonly decompressedStart?: bigint;
  readonly decompressedEnd?: bigint;
}

export interface SeekableErrorOptions {
  cause?: unknown;
  coordinates?: SeekableCoordinates;
}

export class SeekableError extends Error {
  readonly code: SeekableErrorCode;
  readonly frameIndex?: number;
  readonly compressedStart?: bigint;
  readonly compressedEnd?: bigint;
  readonly decompressedStart?: bigint;
  readonly decompressedEnd?: bigint;

  constructor(code: SeekableErrorCode, message: string, options?: SeekableErrorOptions) {
    super(message, options?.cause === undefined ? undefined : { cause: options.cause });
    this.name = 'SeekableError';
    this.code = code;
    const coordinates = options?.coordinates;
    if (coordinates) {
      this.frameIndex = coordinates.frameIndex;
      this.compressedStart = coordinates.compressedStart;
      this.compressedEnd = coordinates.compressedEnd;
      this.decompressedStart = coordinates.decompressedStart;
      this.decompressedEnd = coordinates.decompressedEnd;
    }
  }
}

export class SeekableWindowAbortError extends SeekableError {
  constructor(message: string, options?: SeekableErrorOptions) {
    super('SEEK_WINDOW_ABORTED', message, options);
    this.name = 'AbortError';
  }
}

/** One data frame described by the seek table; intervals are half-open bigints. */
export interface SeekTableEntry {
  readonly index: number;
  readonly compressedStart: bigint;
  readonly compressedEnd: bigint;
  readonly decompressedStart: bigint;
  readonly decompressedEnd: bigint;
  /** Raw per-frame checksum when the descriptor requests checksums, else null. */
  readonly checksum: number | null;
}

export interface SeekTableIndex {
  readonly entries: readonly SeekTableEntry[];
  /** Offset of the seek table skippable frame (also the total data-frame byte count). */
  readonly tableOffset: bigint;
  /** Declared payload length of the seek table skippable frame. */
  readonly tableFramePayloadLength: number;
  readonly checksumPerFrame: boolean;
  readonly compressedLength: bigint;
  readonly decompressedLength: bigint;
}

function magicAt(data: Uint8Array, offset: number, bytes: readonly number[]): boolean {
  return (
    data[offset] === bytes[0] &&
    data[offset + 1] === bytes[1] &&
    data[offset + 2] === bytes[2] &&
    data[offset + 3] === bytes[3]
  );
}

/**
 * Locates the seek table frame. The skippable frame is required to be the last
 * frame of the file, so its position is derived directly from the length field
 * four bytes before EOF: a real index therefore never has to be searched for
 * inside frame payload, and a seek table magic buried in data bytes cannot be
 * mistaken for one. Only when EOF does not end on a valid seek table frame
 * (truncated or misframed input) is a bounded backwards scan attempted.
 */
function findSeekTableFrame(data: Uint8Array): { offset: number; payloadLength: number } | null {
  if (data.length < 8 + SEEK_TABLE_HEADER_SIZE) {
    return null;
  }

  const payloadLength = readU32Le(data, data.length - 4);
  if (BigInt(payloadLength) + 8n <= BigInt(data.length)) {
    const offset = data.length - 8 - payloadLength;
    if (magicAt(data, offset, SEEK_TABLE_MAGIC_BYTES)) {
      return { offset, payloadLength };
    }
    // A generic skippable frame at the tail is definitive: this is not a
    // seekable file, and frame data must not be scanned for a doppelgänger.
    const tailMagic = readU32Le(data, offset);
    if ((tailMagic & ZSTD_SKIPPABLE_MAGIC_MASK) === ZSTD_SKIPPABLE_MAGIC_START) {
      return null;
    }
  }

  for (let offset = data.length - 8; offset >= 0; offset -= 1) {
    if (!magicAt(data, offset, SEEK_TABLE_MAGIC_BYTES)) {
      continue;
    }
    const candidateLength = readU32Le(data, offset + 4);
    if (BigInt(offset) + 8n + BigInt(candidateLength) === BigInt(data.length)) {
      return { offset, payloadLength: candidateLength };
    }
  }
  return null;
}

export function parseSeekTable(data: Uint8Array): SeekTableIndex {
  if (!(data instanceof Uint8Array)) {
    throw new TypeError('Seekable input must be a Uint8Array');
  }

  const candidate = findSeekTableFrame(data);
  if (candidate === null) {
    throw new SeekableError(
      'SEEK_TABLE_NOT_FOUND',
      'No seek table frame was found at the end of the buffer; input is not in Zstandard seekable format',
    );
  }

  const { offset: tableOffset, payloadLength } = candidate;
  const payloadStart = tableOffset + 8;

  if (payloadLength < SEEK_TABLE_HEADER_SIZE) {
    throw new SeekableError(
      'SEEK_TABLE_TRUNCATED',
      `Seek table payload is ${payloadLength} bytes, need at least ${SEEK_TABLE_HEADER_SIZE}`,
      { coordinates: { compressedStart: BigInt(tableOffset) } },
    );
  }

  const descriptor = data[payloadStart]!;
  if ((descriptor & 0xfe) !== 0) {
    throw new SeekableError(
      'SEEK_TABLE_DESCRIPTOR_RESERVED',
      `Seek table descriptor 0x${descriptor.toString(16).padStart(2, '0')} sets reserved bits`,
      { coordinates: { compressedStart: BigInt(tableOffset) } },
    );
  }
  if (data[payloadStart + 1] !== 0 || data[payloadStart + 2] !== 0) {
    throw new SeekableError(
      'SEEK_TABLE_RESERVED_BYTES',
      'Seek table reserved bytes after the descriptor must be zero',
      { coordinates: { compressedStart: BigInt(tableOffset) } },
    );
  }

  const checksumPerFrame = (descriptor & 1) === 1;
  const entrySize = checksumPerFrame
    ? SEEK_TABLE_ENTRY_CHECKSUM_SIZE
    : SEEK_TABLE_ENTRY_SIZE;
  const bodyLength = payloadLength - SEEK_TABLE_HEADER_SIZE;
  if (bodyLength % entrySize !== 0) {
    throw new SeekableError(
      'SEEK_TABLE_BAD_ENTRY_LENGTH',
      `Seek table body of ${bodyLength} bytes is not a multiple of ${entrySize}-byte entries`,
      { coordinates: { compressedStart: BigInt(tableOffset) } },
    );
  }

  const count = bodyLength / entrySize;
  if (count > ZSTD_SEEK_TABLE_MAX_ENTRIES) {
    throw new SeekableError(
      'SEEK_TABLE_TOO_MANY_ENTRIES',
      `Seek table declares ${count} frames, limit is ${ZSTD_SEEK_TABLE_MAX_ENTRIES}`,
      { coordinates: { compressedStart: BigInt(tableOffset) } },
    );
  }

  const entries: SeekTableEntry[] = [];
  let compressedCursor = 0n;
  let decompressedCursor = 0n;
  let recordOffset = payloadStart + SEEK_TABLE_HEADER_SIZE;

  for (let index = 0; index < count; index += 1) {
    const compressedSize = readU32Le(data, recordOffset);
    const decompressedSize = readU32Le(data, recordOffset + 4);
    const checksum = checksumPerFrame ? readU32Le(data, recordOffset + 8) : null;

    if ((compressedSize & COMPRESSED_SIZE_RESERVED_MASK) !== 0) {
      throw new SeekableError(
        'SEEK_TABLE_RESERVED_SIZE_BITS',
        `Seek table entry ${index} sets reserved bits in its compressed size`,
        {
          coordinates: {
            frameIndex: index,
            compressedStart: compressedCursor,
            decompressedStart: decompressedCursor,
          },
        },
      );
    }

    const compressedStart = compressedCursor;
    const compressedEnd = compressedCursor + BigInt(compressedSize);
    const decompressedStart = decompressedCursor;
    const decompressedEnd = decompressedCursor + BigInt(decompressedSize);

    // A malicious entry may never reach into or past the table frame.
    if (compressedEnd > BigInt(tableOffset)) {
      throw new SeekableError(
        'SEEK_TABLE_OFFSET_OUT_OF_BOUNDS',
        `Seek table entry ${index} extends past the seek table frame at byte ${tableOffset}`,
        {
          coordinates: {
            frameIndex: index,
            compressedStart,
            compressedEnd,
            decompressedStart,
            decompressedEnd,
          },
        },
      );
    }

    if (
      compressedSize === 0 ||
      compressedEnd - compressedStart < 4n ||
      compressedEnd + 4n > BigInt(data.length) ||
      !magicAt(data, Number(compressedStart), FRAME_MAGIC_BYTES)
    ) {
      throw new SeekableError(
        'SEEK_FRAME_BAD_MAGIC',
        `Data frame ${index} at compressed offset ${compressedStart} does not start with a Zstandard frame magic`,
        {
          coordinates: {
            frameIndex: index,
            compressedStart,
            compressedEnd,
            decompressedStart,
            decompressedEnd,
          },
        },
      );
    }

    entries.push(
      Object.freeze({
        index,
        compressedStart,
        compressedEnd,
        decompressedStart,
        decompressedEnd,
        checksum,
      }),
    );

    compressedCursor = compressedEnd;
    decompressedCursor = decompressedEnd;
    recordOffset += entrySize;
  }

  // The table's own bytes are never part of the data-frame coordinate space.
  if (compressedCursor !== BigInt(tableOffset)) {
    throw new SeekableError(
      'SEEK_TABLE_TOTAL_LENGTH_MISMATCH',
      `Seek table entries cover ${compressedCursor} compressed bytes but data frames end at ${tableOffset}`,
      {
        coordinates: {
          compressedStart: compressedCursor,
          compressedEnd: BigInt(tableOffset),
          decompressedEnd: decompressedCursor,
        },
      },
    );
  }

  return Object.freeze({
    entries: Object.freeze(entries),
    tableOffset: BigInt(tableOffset),
    tableFramePayloadLength: payloadLength,
    checksumPerFrame,
    compressedLength: compressedCursor,
    decompressedLength: decompressedCursor,
  });
}

// --- Frame header optional field location ---------------------------------

export interface ZstdFrameHeader {
  readonly descriptor: number;
  readonly singleSegment: boolean;
  readonly contentChecksum: boolean;
  readonly dictionaryIdFlag: number;
  readonly contentSizeFlag: number;
  readonly windowDescriptor: number | null;
  readonly windowDescriptorOffset: number | null;
  readonly dictionary: DictionarySelector;
  readonly dictionaryFieldLength: 0 | 1 | 2 | 4;
  readonly contentSize: bigint | null;
  readonly contentSizeOffset: number | null;
  /** Offset of the first block header, relative to the frame start. */
  readonly headerSize: number;
}

function contentSizeFieldSize(contentSizeFlag: number, singleSegment: boolean): 0 | 1 | 2 | 4 | 8 {
  switch (contentSizeFlag) {
    case 0:
      return singleSegment ? 1 : 0;
    case 1:
      return 2;
    case 2:
      return 4;
    default:
      return singleSegment ? 8 : 4;
  }
}

/**
 * Locates every optional frame header field (window descriptor, dictionary id,
 * frame content size) starting at `offset` in `data`. The returned selector
 * keeps "field absent" ({@link NO_DICTIONARY}) distinct from a zero field
 * ({@link APPLICATION_DICTIONARY}); callers must not merge the two.
 */
export function locateFrameHeader(
  data: Uint8Array,
  entry: Pick<SeekTableEntry, 'index' | 'compressedStart' | 'compressedEnd'>,
): ZstdFrameHeader {
  const start = Number(entry.compressedStart);
  const end = Number(entry.compressedEnd);

  if (start < 0 || end > data.length || start + 4 > data.length) {
    throw new SeekableError(
      'SEEK_FRAME_OUT_OF_BOUNDS',
      `Frame ${entry.index} bytes are outside the buffer`,
      {
        coordinates: {
          frameIndex: entry.index,
          compressedStart: entry.compressedStart,
          compressedEnd: entry.compressedEnd,
        },
      },
    );
  }
  if (!magicAt(data, start, FRAME_MAGIC_BYTES)) {
    throw new SeekableError(
      'SEEK_FRAME_BAD_MAGIC',
      `Frame ${entry.index} does not start with a Zstandard frame magic`,
      {
        coordinates: {
          frameIndex: entry.index,
          compressedStart: entry.compressedStart,
          compressedEnd: entry.compressedEnd,
        },
      },
    );
  }

  const descriptorOffset = start + 4;
  if (descriptorOffset >= end || descriptorOffset >= data.length) {
    throw new SeekableError(
      'SEEK_FRAME_HEADER_TRUNCATED',
      `Frame ${entry.index} has no frame header descriptor`,
      {
        coordinates: {
          frameIndex: entry.index,
          compressedStart: entry.compressedStart,
          compressedEnd: entry.compressedEnd,
        },
      },
    );
  }
  const descriptorByte = data[descriptorOffset]!;
  if ((descriptorByte & FRAME_HEADER_DESCRIPTOR_RESERVED_MASK) !== 0) {
    throw new SeekableError(
      'SEEK_FRAME_HEADER_RESERVED_BITS',
      `Frame ${entry.index} descriptor sets reserved bits`,
      {
        coordinates: {
          frameIndex: entry.index,
          compressedStart: entry.compressedStart,
          compressedEnd: entry.compressedEnd,
        },
      },
    );
  }
  const flags = parseDescriptor(descriptorByte);

  let cursor = descriptorOffset + 1;
  let windowDescriptor: number | null = null;
  let windowDescriptorOffset: number | null = null;
  if (!flags.singleSegment) {
    if (cursor >= end || cursor >= data.length) {
      throw new SeekableError(
        'SEEK_FRAME_HEADER_TRUNCATED',
        `Frame ${entry.index} is missing its window descriptor`,
        {
          coordinates: {
            frameIndex: entry.index,
            compressedStart: entry.compressedStart,
            compressedEnd: entry.compressedEnd,
          },
        },
      );
    }
    windowDescriptorOffset = cursor;
    windowDescriptor = data[cursor]!;
    cursor += 1;
  }

  const dictionaryRead = readDictionarySelector(data, descriptorOffset, cursor);
  if (dictionaryRead === null) {
    throw new SeekableError(
      'SEEK_FRAME_HEADER_TRUNCATED',
      `Frame ${entry.index} dictionary id field is truncated`,
      {
        coordinates: {
          frameIndex: entry.index,
          compressedStart: entry.compressedStart,
          compressedEnd: entry.compressedEnd,
        },
      },
    );
  }
  cursor = dictionaryRead.nextOffset;
  if (cursor > end) {
    throw new SeekableError(
      'SEEK_FRAME_HEADER_TRUNCATED',
      `Frame ${entry.index} header runs past its compressed frame bytes`,
      {
        coordinates: {
          frameIndex: entry.index,
          compressedStart: entry.compressedStart,
          compressedEnd: entry.compressedEnd,
        },
      },
    );
  }

  const fieldSize = contentSizeFieldSize(flags.contentSizeFlag, flags.singleSegment);
  let contentSize: bigint | null = null;
  let contentSizeOffset: number | null = null;
  if (fieldSize > 0) {
    if (cursor + fieldSize > end || cursor + fieldSize > data.length) {
      throw new SeekableError(
        'SEEK_FRAME_HEADER_TRUNCATED',
        `Frame ${entry.index} content size field is truncated`,
        {
          coordinates: {
            frameIndex: entry.index,
            compressedStart: entry.compressedStart,
            compressedEnd: entry.compressedEnd,
          },
        },
      );
    }
    contentSizeOffset = cursor;
    if (fieldSize === 8) {
      contentSize = readU64Le(data, cursor);
    } else {
      let value = 0n;
      for (let index = 0; index < fieldSize; index += 1) {
        value |= BigInt(data[cursor + index]!) << BigInt(8 * index);
      }
      // A one-byte field of zero encodes 256; a four-byte zero without the
      // single segment flag means "unknown".
      if (fieldSize === 1 && value === 0n) {
        value = 256n;
      }
      if (fieldSize === 4 && !flags.singleSegment && value === 0n) {
        contentSize = null;
      } else {
        contentSize = value;
      }
    }
    cursor += fieldSize;
  }

  return {
    descriptor: descriptorByte,
    singleSegment: flags.singleSegment,
    contentChecksum: flags.checksum,
    dictionaryIdFlag: flags.dictionaryIdFlag,
    contentSizeFlag: flags.contentSizeFlag,
    windowDescriptor,
    windowDescriptorOffset,
    dictionary: dictionaryRead.selector,
    dictionaryFieldLength: dictionaryRead.fieldLength,
    contentSize,
    contentSizeOffset,
    headerSize: cursor - start,
  };
}

// --- Random window reads ---------------------------------------------------

export type SeekableFrameDecoder = InjectedFrameDecoder<Uint8Array>;

export interface SeekableReaderOptions {
  readonly repository?: VersionedDictionaryRepository;
}

export interface ReadWindowRequest {
  readonly start: number | bigint;
  readonly length: number | bigint;
  readonly decoder: SeekableFrameDecoder;
  readonly signal?: AbortSignal;
}

export interface SeekableWindow {
  readonly data: Uint8Array;
  readonly start: bigint;
  readonly end: bigint;
  readonly frames: readonly SeekTableEntry[];
}

interface Participant {
  /** Settled exactly once: either by its own abort or by flight settlement. */
  settled: boolean;
  signal: AbortSignal;
  resolve(value: Uint8Array): void;
  reject(reason: unknown): void;
  onAbort(): void;
}

interface FrameFlight {
  readonly decoder: SeekableFrameDecoder;
  readonly controller: AbortController;
  /** Aborted participants stay registered until settlement so their listener
   * is removed on the settle path as well; they never receive a second result. */
  readonly participants: Set<Participant>;
  settled: boolean;
}

function coordinatesOf(entry: SeekTableEntry): SeekableCoordinates {
  return {
    frameIndex: entry.index,
    compressedStart: entry.compressedStart,
    compressedEnd: entry.compressedEnd,
    decompressedStart: entry.decompressedStart,
    decompressedEnd: entry.decompressedEnd,
  };
}

function offsetFrom(value: number | bigint, name: string): bigint {
  if (typeof value === 'bigint') {
    if (value < 0n) {
      throw new SeekableError('SEEK_WINDOW_BAD_ARGUMENT', `${name} must be non-negative`);
    }
    return value;
  }
  if (typeof value === 'number' && Number.isSafeInteger(value) && value >= 0) {
    return BigInt(value);
  }
  throw new SeekableError(
    'SEEK_WINDOW_BAD_ARGUMENT',
    `${name} must be a non-negative safe integer or bigint, got ${String(value)}`,
  );
}

function isAbortLike(reason: unknown): boolean {
  const candidate = reason as { name?: unknown; code?: unknown } | null;
  return candidate?.name === 'AbortError' || candidate?.code === 'ABORT_ERR';
}

export interface SeekableReader {
  readonly index: SeekTableIndex;
  readWindow(request: ReadWindowRequest): Promise<SeekableWindow>;
}

export function createSeekableReader(
  data: Uint8Array,
  options: SeekableReaderOptions = {},
): SeekableReader {
  const index = parseSeekTable(data);
  const resolver = createDictionaryFrameResolver(
    options.repository ?? {
      async loadDictionary() {
        throw new Error('No versioned dictionary repository was configured for this reader');
      },
    },
  );

  // In-flight frames are coalesced across concurrent readWindow calls, but
  // nothing is cached after a flight settles: a later request always starts a
  // new resolveFrame() call and therefore observes the current repository
  // revision. A stale dictionary snapshot can never reach a new request.
  const flights = new Map<number, FrameFlight[]>();

  function frameError(
    entry: SeekTableEntry,
    code: SeekableErrorCode,
    message: string,
    cause?: unknown,
  ): SeekableError {
    return new SeekableError(code, message, { cause, coordinates: coordinatesOf(entry) });
  }

  function settleFlight(
    entry: SeekTableEntry,
    flight: FrameFlight,
    outcome: { ok: true; value: unknown } | { ok: false; reason: unknown },
  ): void {
    flight.settled = true;
    const current = flights.get(entry.index);
    if (current) {
      const remaining = current.filter((candidate) => candidate !== flight);
      if (remaining.length === 0) {
        flights.delete(entry.index);
      } else {
        flights.set(entry.index, remaining);
      }
    }

    let error: SeekableError | null = null;
    let bytes: Uint8Array | null = null;

    if (outcome.ok) {
      if (!(outcome.value instanceof Uint8Array)) {
        error = frameError(
          entry,
          'SEEK_FRAME_DECODE_FAILED',
          `Decoder for frame ${entry.index} returned ${typeof outcome.value} instead of Uint8Array`,
        );
      } else {
        const declared = entry.decompressedEnd - entry.decompressedStart;
        if (BigInt(outcome.value.length) !== declared) {
          error = frameError(
            entry,
            'SEEK_FRAME_DECODED_SIZE_MISMATCH',
            `Frame ${entry.index} decoded to ${outcome.value.length} bytes but the seek table declares ${declared}`,
          );
        } else {
          bytes = outcome.value;
        }
      }
    } else if (outcome.reason instanceof SeekableError) {
      error = outcome.reason;
    } else if (isAbortLike(outcome.reason)) {
      // The shared controller is aborted only once every participant has left,
      // so an abort outcome is expected to find only already-settled waiters.
      error = new SeekableWindowAbortError(
        `Frame ${entry.index} decode was aborted`,
        { cause: outcome.reason, coordinates: coordinatesOf(entry) },
      );
    } else if (
      typeof outcome.reason === 'object' &&
      outcome.reason !== null &&
      'code' in outcome.reason &&
      ((outcome.reason as { code: unknown }).code === 'FRAME_DICTIONARY_OPERATION_ABORTED' ||
        (outcome.reason as { code?: unknown }).code === 'DICTIONARY_NOT_FOUND' ||
        (outcome.reason as { code?: unknown }).code === 'DUPLICATE_DICTIONARY_REVISION' ||
        (outcome.reason as { code?: unknown }).code === 'DICTIONARY_LOAD_FAILED' ||
        (outcome.reason as { code?: unknown }).code === 'INVALID_DICTIONARY_ID')
    ) {
      error = frameError(
        entry,
        'SEEK_DICTIONARY_RESOLUTION_FAILED',
        `Frame ${entry.index} failed while resolving its dictionary`,
        outcome.reason,
      );
    } else {
      error = frameError(
        entry,
        'SEEK_FRAME_DECODE_FAILED',
        `Decoder for frame ${entry.index} failed`,
        outcome.reason,
      );
    }

    // Every participant (including ones that already aborted) is cleaned up
    // here; already-settled ones get no second resolution and therefore results
    // can never cross-talk between requests.
    for (const participant of flight.participants) {
      participant.signal.removeEventListener('abort', participant.onAbort);
      if (!participant.settled) {
        participant.settled = true;
        if (error) {
          participant.reject(error);
        } else {
          participant.resolve(bytes!);
        }
      }
    }
    flight.participants.clear();
  }

  function startFlight(entry: SeekTableEntry, decoder: SeekableFrameDecoder): FrameFlight {
    const controller = new AbortController();
    const flight: FrameFlight = { decoder, controller, participants: new Set(), settled: false };

    // The frame bytes handed to the decoder are exactly the bytes the index
    // bounds to this frame, validated against the buffer, never spanning a
    // neighbour frame.
    const frameBytes = data.subarray(Number(entry.compressedStart), Number(entry.compressedEnd));
    const header = locateFrameHeader(data, entry);
    const declared = entry.decompressedEnd - entry.decompressedStart;
    if (header.contentSize !== null && header.contentSize !== declared) {
      queueMicrotask(() => {
        settleFlight(entry, flight, {
          ok: false,
          reason: frameError(
            entry,
            'SEEK_FRAME_SIZE_MISMATCH',
            `Frame ${entry.index} header content size ${header.contentSize} does not match seek table size ${declared}`,
          ),
        });
      });
      return flight;
    }

    const promise = resolver.resolveFrame({
      frameId: entry.index,
      data: frameBytes,
      dictionary: header.dictionary,
      decoder,
      signal: controller.signal,
    });
    promise.then(
      (value) => settleFlight(entry, flight, { ok: true, value }),
      (reason) => settleFlight(entry, flight, { ok: false, reason }),
    );
    return flight;
  }

  function countActive(flight: FrameFlight): number {
    let count = 0;
    for (const participant of flight.participants) {
      if (!participant.settled) count += 1;
    }
    return count;
  }

  function joinFrame(
    entry: SeekTableEntry,
    decoder: SeekableFrameDecoder,
    signal: AbortSignal,
  ): Promise<Uint8Array> {
    const frameFlights = flights.get(entry.index);
    let flight = frameFlights?.find((candidate) => candidate.decoder === decoder && !candidate.settled);
    if (!flight) {
      // locateFrameHeader / size checks surface as synchronously thrown,
      // deterministic errors before any participant is registered.
      flight = startFlight(entry, decoder);
      if (!frameFlights) {
        flights.set(entry.index, [flight]);
      } else {
        frameFlights.push(flight);
      }
    }

    return new Promise<Uint8Array>((resolve, reject) => {
      const participant: Participant = {
        settled: false,
        signal,
        resolve,
        reject,
        onAbort() {
          if (participant.settled || flight!.settled) {
            return;
          }
          participant.settled = true;
          reject(
            new SeekableWindowAbortError(
              `Window read was aborted while frame ${entry.index} was decoding`,
              { coordinates: coordinatesOf(entry) },
            ),
          );
          // The shared decode (and its dictionary lease) is cancelled only once
          // every window waiting on it has left; surviving windows keep it alive.
          if (countActive(flight!) === 0) {
            flight!.controller.abort(signal.reason);
          }
        },
      };
      flight.participants.add(participant);
      signal.addEventListener('abort', participant.onAbort, { once: true });
    });
  }

  async function readWindow(request: ReadWindowRequest): Promise<SeekableWindow> {
    if (!request || typeof request !== 'object') {
      throw new SeekableError('SEEK_WINDOW_BAD_ARGUMENT', 'readWindow requires a request object');
    }
    if (!request.decoder || typeof request.decoder.decodeFrame !== 'function') {
      throw new SeekableError('SEEK_WINDOW_BAD_ARGUMENT', 'decoder must implement decodeFrame');
    }
    const signal = request.signal ?? new AbortController().signal;
    if (!(signal instanceof AbortSignal)) {
      throw new SeekableError('SEEK_WINDOW_BAD_ARGUMENT', 'signal must be an AbortSignal');
    }

    const start = offsetFrom(request.start, 'start');
    const length = offsetFrom(request.length, 'length');
    if (length > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new SeekableError(
        'SEEK_WINDOW_BAD_ARGUMENT',
        `Window length ${length} exceeds the maximum materialisable length`,
      );
    }
    const end = start + length;
    if (start > index.decompressedLength || end > index.decompressedLength) {
      throw new SeekableError(
        'SEEK_WINDOW_OUT_OF_BOUNDS',
        `Window [${start}, ${end}) exceeds decompressed length ${index.decompressedLength}`,
        {
          coordinates: {
            compressedStart: index.compressedLength,
            decompressedStart: 0n,
            decompressedEnd: index.decompressedLength,
          },
        },
      );
    }

    if (signal.aborted) {
      throw new SeekableWindowAbortError('Window read was aborted before it started');
    }

    const intersecting = index.entries.filter(
      (entry) => entry.decompressedStart < end && entry.decompressedEnd > start,
    );

    // Zero-length windows resolve without touching any frame at all.
    if (length === 0n) {
      return {
        data: new Uint8Array(0),
        start,
        end,
        frames: Object.freeze([]),
      };
    }

    // Register every participant synchronously so parallel readWindow calls
    // that hit the same frame share one decode (and one dictionary lease).
    const tasks = intersecting.map((entry) => joinFrame(entry, request.decoder, signal));
    const decoded = await Promise.all(tasks);

    const output = new Uint8Array(Number(length));
    decoded.forEach((bytes, position) => {
      const entry = intersecting[position]!;
      const pieceStart = start > entry.decompressedStart ? start : entry.decompressedStart;
      const pieceEnd = end < entry.decompressedEnd ? end : entry.decompressedEnd;
      const sourceStart = Number(pieceStart - entry.decompressedStart);
      const sourceEnd = Number(pieceEnd - entry.decompressedStart);
      output.set(bytes.subarray(sourceStart, sourceEnd), Number(pieceStart - start));
    });

    return {
      data: output,
      start,
      end,
      frames: Object.freeze(intersecting.slice()),
    };
  }

  return Object.freeze({ index, readWindow });
}
