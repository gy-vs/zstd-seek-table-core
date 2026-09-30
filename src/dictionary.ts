export type FrameId = number | string;
export type DictionaryRevision = number | string;

export type DictionarySelector =
  | { readonly kind: 'none' }
  | { readonly kind: 'application'; readonly id: 0 }
  | { readonly kind: 'explicit'; readonly id: number };

export const NO_DICTIONARY: DictionarySelector = Object.freeze({ kind: 'none' });
export const APPLICATION_DICTIONARY: DictionarySelector = Object.freeze({ kind: 'application', id: 0 });

const DICTIONARY_ID_FIELD_LENGTHS = [0, 1, 2, 4] as const;

export interface ParsedDictionarySelector {
  readonly selector: DictionarySelector;
  readonly fieldLength: 0 | 1 | 2 | 4;
  readonly nextOffset: number;
}

export function explicitDictionary(id: number): DictionarySelector {
  if (!Number.isInteger(id) || id < 1 || id > 0xffffffff) {
    throw new RangeError(`Explicit dictionary id must be an uint32 greater than zero, got ${String(id)}`);
  }
  return Object.freeze({ kind: 'explicit', id });
}

export function readDictionarySelector(
  data: Uint8Array,
  descriptorOffset = 0,
): ParsedDictionarySelector | null {
  if (!Number.isInteger(descriptorOffset) || descriptorOffset < 0 || descriptorOffset >= data.length) {
    return null;
  }

  const flag = data[descriptorOffset]! & 3;
  const fieldLength = DICTIONARY_ID_FIELD_LENGTHS[flag]!;
  const valueOffset = descriptorOffset + 1;

  if (fieldLength === 0) {
    return { selector: NO_DICTIONARY, fieldLength, nextOffset: valueOffset };
  }

  if (data.length < valueOffset + fieldLength) {
    return null;
  }

  let rawId = 0;
  for (let index = 0; index < fieldLength; index += 1) {
    rawId += data[valueOffset + index]! * 2 ** (8 * index);
  }

  const selector = rawId === 0 ? APPLICATION_DICTIONARY : explicitDictionary(rawId);
  return { selector, fieldLength, nextOffset: valueOffset + fieldLength };
}

export function dictionarySummary(content: Uint8Array): string {
  let hash = 0x811c9dc5;
  for (let index = 0; index < content.length; index += 1) {
    hash ^= content[index]!;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return `fnv1a-${hash.toString(16).padStart(8, '0')}-${content.length}`;
}

export interface DictionarySnapshot {
  readonly id: number;
  readonly revision: DictionaryRevision;
  readonly content: Uint8Array;
  readonly summary: string;
}

export type DictionaryLease = DictionarySnapshot;

interface ManagedDictionaryLease extends DictionarySnapshot {
  release?(): void | Promise<void>;
}

export interface LoadedDictionary {
  id: number;
  revision: DictionaryRevision;
  content: Uint8Array;
  summary?: string;
  release?(): void | Promise<void>;
}

export interface DictionaryLoadContext {
  readonly signal: AbortSignal;
}

export interface VersionedDictionaryRepository {
  loadDictionary(
    id: number,
    context: DictionaryLoadContext,
  ): Promise<LoadedDictionary | null | undefined>;
}

export interface VersionedDictionaryRecord {
  id: number;
  revision: DictionaryRevision;
  content: Uint8Array;
  summary?: string;
}

export class DictionaryNotFoundError extends Error {
  readonly code = 'DICTIONARY_NOT_FOUND' as const;

  constructor(public readonly id: number) {
    super(`Dictionary ${id} was not found`);
    this.name = 'DictionaryNotFoundError';
  }
}

export class DuplicateDictionaryRevisionError extends Error {
  readonly code = 'DUPLICATE_DICTIONARY_REVISION' as const;

  constructor(
    public readonly id: number,
    public readonly revisions: readonly DictionaryRevision[],
  ) {
    super(`Dictionary ${id} has more than one matching revision: ${revisions.join(', ')}`);
    this.name = 'DuplicateDictionaryRevisionError';
  }
}

export type DictionaryResolutionErrorCode =
  | 'DICTIONARY_NOT_FOUND'
  | 'DUPLICATE_DICTIONARY_REVISION'
  | 'DICTIONARY_LOAD_FAILED'
  | 'INVALID_DICTIONARY_ID'
  | 'FRAME_DICTIONARY_OPERATION_ABORTED';

export interface DictionaryResolutionErrorOptions {
  cause?: unknown;
  revisions?: readonly DictionaryRevision[];
}

export class DictionaryResolutionError extends Error {
  readonly revisions?: readonly DictionaryRevision[];

  constructor(
    public readonly code: DictionaryResolutionErrorCode,
    public readonly frameId: FrameId,
    public readonly dictionaryId: number | null,
    message?: string,
    options?: DictionaryResolutionErrorOptions,
  ) {
    super(message, options?.cause === undefined ? undefined : { cause: options.cause });
    this.name = 'DictionaryResolutionError';
    if (options?.revisions) {
      this.revisions = options.revisions;
    }
  }
}

export class FrameDictionaryAbortError extends DictionaryResolutionError {
  constructor(frameId: FrameId, dictionaryId: number | null, reason?: unknown) {
    super(
      'FRAME_DICTIONARY_OPERATION_ABORTED',
      frameId,
      dictionaryId,
      `Frame ${String(frameId)} was aborted while requesting dictionary ${dictionaryId === null ? '<none>' : dictionaryId}`,
      { cause: reason },
    );
    this.name = 'AbortError';
  }
}

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isAbortError(error: unknown): boolean {
  const candidate = error as { name?: unknown; code?: unknown } | null;
  return candidate?.name === 'AbortError' || candidate?.code === 'ABORT_ERR';
}

function wrapLoadFailure(error: unknown, frameId: FrameId, dictionaryId: number): DictionaryResolutionError {
  if (error instanceof DuplicateDictionaryRevisionError) {
    return new DictionaryResolutionError(
      'DUPLICATE_DICTIONARY_REVISION',
      frameId,
      dictionaryId,
      `Duplicate revisions for dictionary ${dictionaryId} requested by frame ${String(frameId)}`,
      { cause: error, revisions: error.revisions },
    );
  }

  if (
    error instanceof DictionaryNotFoundError ||
    (error as { code?: unknown } | null)?.code === 'DICTIONARY_NOT_FOUND'
  ) {
    return new DictionaryResolutionError(
      'DICTIONARY_NOT_FOUND',
      frameId,
      dictionaryId,
      `Dictionary ${dictionaryId} requested by frame ${String(frameId)} was not found`,
      { cause: error },
    );
  }

  if (isAbortError(error)) {
    return new FrameDictionaryAbortError(frameId, dictionaryId, error);
  }

  return new DictionaryResolutionError(
    'DICTIONARY_LOAD_FAILED',
    frameId,
    dictionaryId,
    `Failed to load dictionary ${dictionaryId} for frame ${String(frameId)}: ${describeError(error)}`,
    { cause: error },
  );
}

function normalizeLease(loaded: LoadedDictionary, requestedId: number): ManagedDictionaryLease {
  if (loaded.id !== requestedId) {
    throw new TypeError(`Repository returned dictionary ${loaded.id}, expected ${requestedId}`);
  }
  if (
    (typeof loaded.revision !== 'number' && typeof loaded.revision !== 'string') ||
    (typeof loaded.revision === 'number' && !Number.isFinite(loaded.revision)) ||
    loaded.revision === ''
  ) {
    throw new TypeError(`Repository returned an invalid revision for dictionary ${requestedId}`);
  }
  if (!(loaded.content instanceof Uint8Array)) {
    throw new TypeError(`Repository returned invalid content for dictionary ${requestedId}`);
  }
  if (loaded.summary !== undefined && typeof loaded.summary !== 'string') {
    throw new TypeError(`Repository returned an invalid summary for dictionary ${requestedId}`);
  }

  const content = loaded.content.slice();
  const summary = loaded.summary ?? dictionarySummary(content);
  const release = typeof loaded.release === 'function' ? loaded.release.bind(loaded) : undefined;

  return Object.freeze(
    release
      ? { id: requestedId, revision: loaded.revision, content, summary, release }
      : { id: requestedId, revision: loaded.revision, content, summary },
  );
}

function validateVersionedRecord(record: VersionedDictionaryRecord): void {
  if (!Number.isInteger(record.id) || record.id < 1 || record.id > 0xffffffff) {
    throw new RangeError(`Invalid dictionary id: ${String(record.id)}`);
  }
  if (
    (typeof record.revision !== 'number' && typeof record.revision !== 'string') ||
    (typeof record.revision === 'number' && !Number.isFinite(record.revision)) ||
    record.revision === ''
  ) {
    throw new RangeError(`Invalid revision for dictionary ${record.id}`);
  }
  if (!(record.content instanceof Uint8Array)) {
    throw new TypeError(`Invalid content for dictionary ${record.id}`);
  }
}

export class InMemoryVersionedDictionaryRepository implements VersionedDictionaryRepository {
  private readonly current = new Map<number, VersionedDictionaryRecord>();
  activeLeaseCount = 0;

  add(record: VersionedDictionaryRecord): void {
    validateVersionedRecord(record);
    const existing = this.current.get(record.id);
    if (existing) {
      throw new DuplicateDictionaryRevisionError(record.id, [existing.revision, record.revision]);
    }
    this.current.set(record.id, record);
  }

  publishUpdate(record: VersionedDictionaryRecord): void {
    validateVersionedRecord(record);
    this.current.set(record.id, record);
  }

  currentRevision(id: number): DictionaryRevision | undefined {
    return this.current.get(id)?.revision;
  }

  async loadDictionary(id: number, context: DictionaryLoadContext): Promise<LoadedDictionary> {
    if (context.signal.aborted) {
      throw context.signal.reason ?? new Error('Dictionary load was aborted');
    }
    const record = this.current.get(id);
    if (!record) {
      throw new DictionaryNotFoundError(id);
    }

    this.activeLeaseCount += 1;
    return {
      id: record.id,
      revision: record.revision,
      content: record.content,
      summary: record.summary,
      release: () => {
        this.activeLeaseCount -= 1;
      },
    };
  }
}

export type ResolvedDictionary =
  | { readonly kind: 'none' }
  | { readonly kind: 'application'; readonly id: 0 }
  | { readonly kind: 'explicit'; readonly lease: DictionaryLease };

export interface FrameDecoderContext {
  readonly frameId: FrameId;
  readonly data: Uint8Array;
  readonly signal: AbortSignal;
}

export interface InjectedFrameDecoder<T> {
  decodeFrame(context: FrameDecoderContext, dictionary: ResolvedDictionary): T | Promise<T>;
}

export interface DictionaryFrameRequest<T> {
  readonly frameId: FrameId;
  readonly data: Uint8Array;
  readonly dictionary: DictionarySelector;
  readonly decoder: InjectedFrameDecoder<T>;
  readonly signal?: AbortSignal;
}

export interface DictionaryFrameResolver {
  resolveFrame<T>(request: DictionaryFrameRequest<T>): Promise<T>;
}

interface Waiter<T> {
  frameId: FrameId;
  data: Uint8Array;
  decoder: InjectedFrameDecoder<T>;
  signal: AbortSignal;
  resolve: (value: T) => void;
  reject: (reason: unknown) => void;
  abortHandler: () => void;
  dispatching: boolean;
}

interface InFlightLoad {
  waiters: Set<Waiter<unknown>>;
  controller: AbortController;
}

type DecoderOutcome<T> =
  | { ok: true; value: T }
  | { ok: false; reason: unknown };

export function createDictionaryFrameResolver(repository: VersionedDictionaryRepository): DictionaryFrameResolver {
  const inFlight = new Map<number, InFlightLoad>();
  const neverAborted = new AbortController();

  function requestedId(selector: DictionarySelector): number | null {
    if (selector.kind === 'explicit') {
      return selector.id;
    }
    return selector.kind === 'application' ? 0 : null;
  }

  function decoderContext(waiter: Waiter<any>): FrameDecoderContext {
    return { frameId: waiter.frameId, data: waiter.data, signal: waiter.signal };
  }

  function detach(waiter: Waiter<any>): void {
    waiter.dispatching = true;
    waiter.signal.removeEventListener('abort', waiter.abortHandler);
  }

  async function dispatchWithoutLease<T>(
    waiter: Waiter<T>,
    dictionary: ResolvedDictionary,
  ): Promise<void> {
    detach(waiter);
    try {
      waiter.resolve(await waiter.decoder.decodeFrame(decoderContext(waiter), dictionary));
    } catch (error) {
      waiter.reject(error);
    }
  }

  async function releaseLease(lease: ManagedDictionaryLease): Promise<void> {
    if (lease.release) {
      await lease.release();
    }
  }

  function settle<T>(waiter: Waiter<T>, outcome: DecoderOutcome<T>): void {
    if (outcome.ok) {
      waiter.resolve(outcome.value);
    } else {
      waiter.reject(outcome.reason);
    }
  }

  async function runLoad(id: number, entry: InFlightLoad): Promise<void> {
    let lease: ManagedDictionaryLease;

    try {
      const loaded = await repository.loadDictionary(id, { signal: entry.controller.signal });
      if (loaded == null) {
        throw new DictionaryNotFoundError(id);
      }
      lease = normalizeLease(loaded, id);
    } catch (error) {
      if (inFlight.get(id) === entry) {
        inFlight.delete(id);
      }
      for (const waiter of entry.waiters) {
        detach(waiter);
        waiter.reject(wrapLoadFailure(error, waiter.frameId, id));
      }
      entry.waiters.clear();
      return;
    }

    if (inFlight.get(id) === entry) {
      inFlight.delete(id);
    }

    const waiters = [...entry.waiters];
    entry.waiters.clear();

    if (waiters.length === 0) {
      await releaseLease(lease).catch(() => undefined);
      return;
    }

    let remaining = waiters.length;
    await Promise.all(waiters.map(async (waiter) => {
      detach(waiter);
      let outcome: DecoderOutcome<unknown>;
      try {
        outcome = {
          ok: true,
          value: await waiter.decoder.decodeFrame(
            decoderContext(waiter),
            { kind: 'explicit', lease },
          ),
        };
      } catch (reason) {
        outcome = { ok: false, reason };
      }

      remaining -= 1;
      if (remaining > 0) {
        settle(waiter, outcome);
        return;
      }

      try {
        await releaseLease(lease);
        settle(waiter, outcome);
      } catch (releaseError) {
        if (outcome.ok) {
          waiter.reject(releaseError);
        } else {
          waiter.reject(new AggregateError(
            [outcome.reason, releaseError],
            'Decoder and dictionary release both failed',
          ));
        }
      }
    }));
  }

  function resolveFrame<T>(request: DictionaryFrameRequest<T>): Promise<T> {
    const { frameId, data, decoder, dictionary } = request;
    const signal = request.signal ?? neverAborted.signal;
    const dictionaryId = requestedId(dictionary);

    if (frameId == null) {
      return Promise.reject(new TypeError('frameId is required'));
    }
    if (!(data instanceof Uint8Array)) {
      return Promise.reject(new TypeError(`Frame ${String(frameId)} data must be a Uint8Array`));
    }
    if (!decoder || typeof decoder.decodeFrame !== 'function') {
      return Promise.reject(new TypeError(`Frame ${String(frameId)} decoder must implement decodeFrame`));
    }
    if (!dictionary || typeof dictionary.kind !== 'string') {
      return Promise.reject(new DictionaryResolutionError(
        'INVALID_DICTIONARY_ID',
        frameId,
        dictionaryId,
        `Frame ${String(frameId)} has an invalid dictionary selector`,
      ));
    }
    if (dictionary.kind === 'explicit') {
      try {
        explicitDictionary(dictionary.id);
      } catch (error) {
        return Promise.reject(new DictionaryResolutionError(
          'INVALID_DICTIONARY_ID',
          frameId,
          dictionary.id,
          `Frame ${String(frameId)} requested invalid dictionary ${String(dictionary.id)}`,
          { cause: error },
        ));
      }
    }
    if (!(signal instanceof AbortSignal)) {
      return Promise.reject(new TypeError(`Frame ${String(frameId)} signal must be an AbortSignal`));
    }
    if (signal.aborted) {
      return Promise.reject(new FrameDictionaryAbortError(frameId, dictionaryId, signal.reason));
    }

    return new Promise<T>((resolve, reject) => {
      const waiter: Waiter<T> = {
        frameId,
        data,
        decoder,
        signal,
        resolve,
        reject,
        dispatching: false,
        abortHandler: () => undefined,
      };

      if (dictionary.kind === 'none' || dictionary.kind === 'application') {
        waiter.abortHandler = () => {
          if (waiter.dispatching) {
            return;
          }
          detach(waiter);
          reject(new FrameDictionaryAbortError(frameId, dictionaryId, signal.reason));
        };
        signal.addEventListener('abort', waiter.abortHandler, { once: true });
        void dispatchWithoutLease(
          waiter,
          dictionary.kind === 'none' ? { kind: 'none' } : { kind: 'application', id: 0 },
        );
        return;
      }

      const id = dictionary.id;
      let entry = inFlight.get(id);
      if (!entry) {
        entry = { waiters: new Set<Waiter<unknown>>(), controller: new AbortController() };
        inFlight.set(id, entry);
        void runLoad(id, entry);
      }

      waiter.abortHandler = () => {
        if (waiter.dispatching || !entry) {
          return;
        }
        if (!entry.waiters.delete(waiter as unknown as Waiter<unknown>)) {
          return;
        }
        waiter.dispatching = true;
        signal.removeEventListener('abort', waiter.abortHandler);
        reject(new FrameDictionaryAbortError(frameId, id, signal.reason));

        if (entry.waiters.size === 0) {
          if (inFlight.get(id) === entry) {
            inFlight.delete(id);
          }
          entry.controller.abort(signal.reason);
        }
      };

      entry.waiters.add(waiter as unknown as Waiter<unknown>);
      signal.addEventListener('abort', waiter.abortHandler, { once: true });
    });
  }

  return { resolveFrame };
}
