import {
  APPLICATION_DICTIONARY,
  NO_DICTIONARY,
  explicitDictionary,
  type DictionarySelector,
} from './dictionary.js';

export type FrameHeader={singleSegment:boolean;checksum:boolean,dictionaryIdFlag:number,contentSizeFlag:number};export function parseDescriptor(byte:number):FrameHeader{return{contentSizeFlag:byte>>6,dictionaryIdFlag:byte&3,checksum:Boolean(byte&4),singleSegment:Boolean(byte&32)}}export function readBlockHeader(data:Uint8Array){if(data.length<3)return null;const value=data[0]|data[1]<<8|data[2]<<16;return{last:Boolean(value&1),type:(value>>1)&3,size:value>>3}}

export const ZSTD_FRAME_MAGIC = 0xfd2fb528;
export const SKIPPABLE_FRAME_MAGIC_BASE = 0x184d2a50;
export const SKIPPABLE_FRAME_MAGIC_MASK = 0xfffffff0;
export const SKIPPABLE_FRAME_HEADER_SIZE = 8;

export function isSkippableFrameMagic(magic: number): boolean {
  return (magic & SKIPPABLE_FRAME_MAGIC_MASK) === SKIPPABLE_FRAME_MAGIC_BASE;
}

export function readUint32LE(data: Uint8Array, offset: number): number {
  return (
    (data[offset]! | (data[offset + 1]! << 8) | (data[offset + 2]! << 16) | (data[offset + 3]! << 24)) >>> 0
  );
}

const DICTIONARY_ID_FIELD_LENGTHS = [0, 1, 2, 4] as const;

export interface ZstdFrameHeader {
  /** Offset of the frame magic inside the input. */
  readonly offset: number;
  readonly descriptor: FrameHeader;
  /** Window descriptor byte, present unless the frame is a single-segment frame. */
  readonly windowDescriptor: number | null;
  readonly dictionary: DictionarySelector;
  readonly dictionaryIdFieldLength: 0 | 1 | 2 | 4;
  readonly contentSizeFieldLength: 0 | 1 | 2 | 4 | 8;
  /** Frame content size when the optional field is present; bigint because the 8-byte form exceeds 2^53. */
  readonly contentSize: bigint | null;
  /** Total frame header size in bytes, from the magic number to the first block byte. */
  readonly headerSize: number;
}

/**
 * Locates the optional fields of a zstd frame header (window descriptor, dictionary id,
 * frame content size) starting at `offset`. Returns null when the bytes at `offset` are
 * not a zstd frame header or the header is truncated; skippable frames are not data
 * frames and therefore also yield null.
 */
export function parseFrameHeader(data: Uint8Array, offset = 0): ZstdFrameHeader | null {
  if (!Number.isInteger(offset) || offset < 0 || data.length - offset < 6) {
    return null;
  }
  if (readUint32LE(data, offset) !== ZSTD_FRAME_MAGIC) {
    return null;
  }

  const descriptor = parseDescriptor(data[offset + 4]!);
  let cursor = offset + 5;

  let windowDescriptor: number | null = null;
  if (!descriptor.singleSegment) {
    windowDescriptor = data[cursor]!;
    cursor += 1;
  }

  const dictionaryIdFieldLength = DICTIONARY_ID_FIELD_LENGTHS[descriptor.dictionaryIdFlag]!;
  if (data.length - cursor < dictionaryIdFieldLength) {
    return null;
  }
  let rawDictionaryId = 0;
  for (let index = 0; index < dictionaryIdFieldLength; index += 1) {
    rawDictionaryId += data[cursor + index]! * 2 ** (8 * index);
  }
  // A missing field and a present field carrying zero stay distinct selectors: the
  // former means "no dictionary", the latter the application-dictionary convention.
  const dictionary: DictionarySelector =
    dictionaryIdFieldLength === 0
      ? NO_DICTIONARY
      : rawDictionaryId === 0
        ? APPLICATION_DICTIONARY
        : explicitDictionary(rawDictionaryId);
  cursor += dictionaryIdFieldLength;

  const contentSizeFieldLength: 0 | 1 | 2 | 4 | 8 =
    descriptor.contentSizeFlag === 0
      ? descriptor.singleSegment
        ? 1
        : 0
      : descriptor.contentSizeFlag === 1
        ? 2
        : descriptor.contentSizeFlag === 2
          ? 4
          : 8;
  if (data.length - cursor < contentSizeFieldLength) {
    return null;
  }
  let contentSize: bigint | null = null;
  if (contentSizeFieldLength > 0) {
    let value = 0n;
    for (let index = 0; index < contentSizeFieldLength; index += 1) {
      value |= BigInt(data[cursor + index]!) << BigInt(8 * index);
    }
    contentSize = descriptor.contentSizeFlag === 1 ? value + 256n : value;
  }
  cursor += contentSizeFieldLength;

  return Object.freeze({
    offset,
    descriptor,
    windowDescriptor,
    dictionary,
    dictionaryIdFieldLength,
    contentSizeFieldLength,
    contentSize,
    headerSize: cursor - offset,
  });
}
