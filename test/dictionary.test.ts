import { describe, expect, it, vi } from 'vitest';
import {
  APPLICATION_DICTIONARY,
  DictionaryResolutionError,
  DuplicateDictionaryRevisionError,
  FrameDictionaryAbortError,
  InMemoryVersionedDictionaryRepository,
  NO_DICTIONARY,
  createDictionaryFrameResolver,
  dictionarySummary,
  explicitDictionary,
  readDictionarySelector,
  type LoadedDictionary,
  type VersionedDictionaryRepository,
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

interface PendingLoad {
  id: number;
  signal: AbortSignal;
  result: Deferred<LoadedDictionary | null>;
}

class ControlledRepository implements VersionedDictionaryRepository {
  readonly pending: PendingLoad[] = [];
  readonly loadDictionary = vi.fn((id: number, { signal }: { signal: AbortSignal }) => {
    const result = deferred<LoadedDictionary | null>();
    this.pending.push({ id, signal, result });
    signal.addEventListener(
      'abort',
      () => result.reject(Object.assign(new Error('load aborted'), { name: 'AbortError' })),
      { once: true },
    );
    return result.promise;
  });

  get last(): PendingLoad {
    const pending = this.pending.at(-1);
    if (!pending) throw new Error('No dictionary load is pending');
    return pending;
  }

  resolveDictionary(overrides: Partial<LoadedDictionary> & { id: number }): void {
    this.last.result.resolve({
      revision: 'rev-1',
      content: new Uint8Array([1, 2, 3]),
      ...overrides,
    });
  }
}

function testDecoder(implementation?: (...args: any[]) => unknown) {
  const decodeFrame = vi.fn(
    implementation ?? (async (_context: unknown, dictionary: unknown) => dictionary),
  );
  return { decodeFrame, mock: decodeFrame.mock };
}

describe('readDictionarySelector', () => {
  it('distinguishes no dictionary from the id=0 application convention', () => {
    expect(readDictionarySelector(new Uint8Array([0]))?.selector).toBe(NO_DICTIONARY);
    expect(readDictionarySelector(new Uint8Array([1, 0]))?.selector).toEqual(APPLICATION_DICTIONARY);
  });

  it('reads one, two, and four byte explicit dictionary ids', () => {
    expect(readDictionarySelector(new Uint8Array([1, 0x2a]))?.selector).toEqual(explicitDictionary(42));
    expect(readDictionarySelector(new Uint8Array([2, 0x34, 0x12]))?.selector).toEqual(explicitDictionary(0x1234));
    expect(readDictionarySelector(new Uint8Array([3, 0x78, 0x56, 0x34, 0x12]))?.selector)
      .toEqual(explicitDictionary(0x12345678));
  });

  it('reports the next frame offset and truncated fields', () => {
    expect(readDictionarySelector(new Uint8Array([3, 1, 2]))).toBeNull();
    expect(readDictionarySelector(new Uint8Array([2, 0x34, 0x12]), 0)?.nextOffset).toBe(3);
  });

  it('reads from a supplied descriptor offset', () => {
    const frame = new Uint8Array([0x28, 0xb5, 0x2f, 0xfd, 1, 0x2a]);
    expect(readDictionarySelector(frame, 4)?.selector).toEqual(explicitDictionary(42));
  });
});

describe('dictionary frame resolver', () => {
  it('does not consult the repository for no dictionary or application id 0', async () => {
    const repository = { loadDictionary: vi.fn() };
    const resolver = createDictionaryFrameResolver(repository);
    const decodeFrame = vi.fn(async () => 'decoded');

    await expect(resolver.resolveFrame({
      frameId: 'frame-none',
      data: new Uint8Array([1]),
      dictionary: NO_DICTIONARY,
      decoder: { decodeFrame },
    })).resolves.toBe('decoded');

    await expect(resolver.resolveFrame({
      frameId: 'frame-app',
      data: new Uint8Array([2]),
      dictionary: APPLICATION_DICTIONARY,
      decoder: { decodeFrame },
    })).resolves.toBe('decoded');

    expect(repository.loadDictionary).not.toHaveBeenCalled();
    expect(decodeFrame).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({ frameId: 'frame-none' }),
      { kind: 'none' },
    );
    expect(decodeFrame).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({ frameId: 'frame-app' }),
      { kind: 'application', id: 0 },
    );
  });

  it('merges concurrent requests for the same explicit dictionary', async () => {
    const repository = new ControlledRepository();
    const resolver = createDictionaryFrameResolver(repository);
    const release = vi.fn(async () => undefined);
    const decoder = testDecoder();

    const first = resolver.resolveFrame({
      frameId: 'frame-1',
      data: new Uint8Array([1]),
      dictionary: explicitDictionary(42),
      decoder,
    });
    const second = resolver.resolveFrame({
      frameId: 'frame-2',
      data: new Uint8Array([2]),
      dictionary: explicitDictionary(42),
      decoder,
    });

    await flush();
    expect(repository.loadDictionary).toHaveBeenCalledTimes(1);

    repository.resolveDictionary({
      id: 42,
      revision: 'rev-1',
      content: new Uint8Array([7, 8, 9]),
      release,
    });

    const dictionaries = await Promise.all([first, second]);
    expect(dictionaries[0]).toEqual({
      kind: 'explicit',
      lease: expect.objectContaining({ id: 42, revision: 'rev-1' }),
    });
    expect(dictionaries[1]?.kind).toBe('explicit');
    expect(dictionaries[0]?.kind === 'explicit' && dictionaries[1]?.kind === 'explicit').toBe(true);
    if (dictionaries[0]?.kind === 'explicit' && dictionaries[1]?.kind === 'explicit') {
      expect(dictionaries[0].lease).toBe(dictionaries[1].lease);
      expect(dictionaries[0].lease.summary).toBe(dictionarySummary(new Uint8Array([7, 8, 9])));
    }
    expect(release).toHaveBeenCalledTimes(1);
  });

  it('keeps the merged load when one caller cancels before load completion', async () => {
    const repository = new ControlledRepository();
    const resolver = createDictionaryFrameResolver(repository);
    const firstController = new AbortController();
    const release = vi.fn();
    const decoder = testDecoder();

    const first = resolver.resolveFrame({
      frameId: 'frame-cancelled',
      data: new Uint8Array([1]),
      dictionary: explicitDictionary(99),
      decoder,
      signal: firstController.signal,
    });
    const second = resolver.resolveFrame({
      frameId: 'frame-kept',
      data: new Uint8Array([2]),
      dictionary: explicitDictionary(99),
      decoder,
    });

    await flush();
    firstController.abort('not needed');
    await expect(first).rejects.toMatchObject({
      name: 'AbortError',
      code: 'FRAME_DICTIONARY_OPERATION_ABORTED',
      frameId: 'frame-cancelled',
      dictionaryId: 99,
    });
    expect(repository.pending.at(-1)?.signal.aborted).toBe(false);

    repository.resolveDictionary({ id: 99, release });
    await expect(second).resolves.toMatchObject({ kind: 'explicit' });
    expect(decoder.mock.calls).toHaveLength(1);
    expect(decoder.mock.calls[0]?.[0]).toMatchObject({ frameId: 'frame-kept' });
    expect(release).toHaveBeenCalledTimes(1);
  });

  it('aborts the repository load only after every waiter cancels', async () => {
    const repository = new ControlledRepository();
    const resolver = createDictionaryFrameResolver(repository);
    const controller = new AbortController();
    const decoder = testDecoder();

    const frame = resolver.resolveFrame({
      frameId: 'frame-only',
      data: new Uint8Array([1]),
      dictionary: explicitDictionary(77),
      decoder,
      signal: controller.signal,
    });

    await flush();
    controller.abort();
    await expect(frame).rejects.toBeInstanceOf(FrameDictionaryAbortError);
    expect(repository.pending[0]?.signal.aborted).toBe(true);
    expect(decoder.mock.calls).toHaveLength(0);
  });

  it('reports a missing dictionary with frame id and requested dictionary id', async () => {
    const repository = new ControlledRepository();
    const resolver = createDictionaryFrameResolver(repository);

    const frame = resolver.resolveFrame({
      frameId: 404,
      data: new Uint8Array(),
      dictionary: explicitDictionary(123),
      decoder: testDecoder(),
    });

    await flush();
    repository.last.result.resolve(null);

    await expect(frame).rejects.toMatchObject({
      name: 'DictionaryResolutionError',
      code: 'DICTIONARY_NOT_FOUND',
      frameId: 404,
      dictionaryId: 123,
    });
    await expect(frame).rejects.toThrow('frame 404');
    await expect(frame).rejects.toThrow('Dictionary 123');
  });

  it('maps duplicate id revisions to a frame-scoped resolution error', async () => {
    const repository = new ControlledRepository();
    const resolver = createDictionaryFrameResolver(repository);
    const frame = resolver.resolveFrame({
      frameId: 'dup-frame',
      data: new Uint8Array(),
      dictionary: explicitDictionary(55),
      decoder: testDecoder(),
    });

    await flush();
    repository.last.result.reject(new DuplicateDictionaryRevisionError(55, ['rev-a', 'rev-b']));

    await expect(frame).rejects.toMatchObject({
      code: 'DUPLICATE_DICTIONARY_REVISION',
      frameId: 'dup-frame',
      dictionaryId: 55,
      revisions: ['rev-a', 'rev-b'],
    });
  });

  it('wraps repository load failures without losing frame and dictionary ids', async () => {
    const repository = new ControlledRepository();
    const resolver = createDictionaryFrameResolver(repository);
    const frame = resolver.resolveFrame({
      frameId: 'failed-frame',
      data: new Uint8Array(),
      dictionary: explicitDictionary(7),
      decoder: testDecoder(),
    });
    const failure = new Error('repository unavailable');

    await flush();
    repository.last.result.reject(failure);

    await expect(frame).rejects.toMatchObject({
      code: 'DICTIONARY_LOAD_FAILED',
      frameId: 'failed-frame',
      dictionaryId: 7,
      cause: failure,
    });
  });

  it('freezes the loaded revision across a repository update and loads the new revision later', async () => {
    const repository = new InMemoryVersionedDictionaryRepository();
    const resolver = createDictionaryFrameResolver(repository);
    repository.add({ id: 321, revision: 'rev-old', content: new Uint8Array([1, 1, 1]) });
    const bothInDecoder = deferred<void>();
    const decoderGate = deferred<void>();
    let entered = 0;
    const decoder = testDecoder(async (_context: unknown, dictionary: unknown) => {
      entered += 1;
      if (entered === 2) bothInDecoder.resolve();
      await decoderGate.promise;
      return dictionary;
    });

    const first = resolver.resolveFrame({
      frameId: 'old-frame-1',
      data: new Uint8Array(),
      dictionary: explicitDictionary(321),
      decoder,
    });
    const second = resolver.resolveFrame({
      frameId: 'old-frame-2',
      data: new Uint8Array(),
      dictionary: explicitDictionary(321),
      decoder,
    });

    await bothInDecoder.promise;
    expect(decoder.mock.calls[0]?.[1]).toMatchObject({
      kind: 'explicit',
      lease: {
        revision: 'rev-old',
        summary: dictionarySummary(new Uint8Array([1, 1, 1])),
        content: new Uint8Array([1, 1, 1]),
      },
    });

    // Updating the versioned store after frame start cannot change either in-flight frame.
    repository.publishUpdate({ id: 321, revision: 'rev-new', content: new Uint8Array([2, 2, 2]) });
    decoderGate.resolve();
    await Promise.all([first, second]);
    expect(repository.activeLeaseCount).toBe(0);

    // The next frame is not merged and selects the repository's new revision.
    const updated = await resolver.resolveFrame({
      frameId: 'new-frame',
      data: new Uint8Array(),
      dictionary: explicitDictionary(321),
      decoder: testDecoder(),
    });
    expect(updated).toMatchObject({ kind: 'explicit', lease: { revision: 'rev-new' } });
    expect(repository.activeLeaseCount).toBe(0);
  });

  it('does not release the shared handle until a late decoder finishes', async () => {
    const repository = new ControlledRepository();
    const resolver = createDictionaryFrameResolver(repository);
    const release = vi.fn(async () => undefined);
    const firstGate = deferred<void>();
    const secondGate = deferred<void>();
    const decoder = testDecoder(async (context: { frameId: string }) => {
      if (context.frameId === 'late-1') {
        await firstGate.promise;
        return 'first';
      }
      await secondGate.promise;
      return 'second';
    });

    const first = resolver.resolveFrame({
      frameId: 'late-1',
      data: new Uint8Array(),
      dictionary: explicitDictionary(88),
      decoder,
    });
    const second = resolver.resolveFrame({
      frameId: 'late-2',
      data: new Uint8Array(),
      dictionary: explicitDictionary(88),
      decoder,
    });

    await flush();
    repository.resolveDictionary({ id: 88, release });
    await flush();
    expect(release).not.toHaveBeenCalled();

    firstGate.resolve();
    await flush();
    expect(release).not.toHaveBeenCalled();

    secondGate.resolve();
    await expect(Promise.all([first, second])).resolves.toEqual(['first', 'second']);
    expect(release).toHaveBeenCalledTimes(1);
  });

  it('allows concurrent frames using different dictionary ids', async () => {
    const repository = new InMemoryVersionedDictionaryRepository();
    const resolver = createDictionaryFrameResolver(repository);
    repository.add({ id: 1, revision: 'a', content: new Uint8Array([1]) });
    repository.add({ id: 2, revision: 'b', content: new Uint8Array([2]) });

    const [first, second] = await Promise.all([
      resolver.resolveFrame({
        frameId: 'f-a',
        data: new Uint8Array(),
        dictionary: explicitDictionary(1),
        decoder: testDecoder(),
      }),
      resolver.resolveFrame({
        frameId: 'f-b',
        data: new Uint8Array(),
        dictionary: explicitDictionary(2),
        decoder: testDecoder(),
      }),
    ]);

    expect(first).toMatchObject({ lease: { id: 1, revision: 'a' } });
    expect(second).toMatchObject({ lease: { id: 2, revision: 'b' } });
    expect(repository.activeLeaseCount).toBe(0);
  });

  it('rejects invalid explicit ids and already-aborted requests with frame context', async () => {
    const repository = new ControlledRepository();
    const resolver = createDictionaryFrameResolver(repository);
    const controller = new AbortController();
    controller.abort('gone');

    await expect(resolver.resolveFrame({
      frameId: 'bad-id',
      data: new Uint8Array(),
      dictionary: { kind: 'explicit', id: 0 } as never,
      decoder: testDecoder(),
    })).rejects.toBeInstanceOf(DictionaryResolutionError);

    await expect(resolver.resolveFrame({
      frameId: 'pre-aborted',
      data: new Uint8Array(),
      dictionary: NO_DICTIONARY,
      decoder: testDecoder(),
      signal: controller.signal,
    })).rejects.toMatchObject({
      frameId: 'pre-aborted',
      dictionaryId: null,
      cause: 'gone',
    });
  });
});
