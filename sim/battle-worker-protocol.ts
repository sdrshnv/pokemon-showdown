/**
 * Pokezero battle worker protocol
 *
 * The wire format is one MessagePack object prefixed by its unsigned 32-bit
 * length in network byte order. Tensor payloads are standard MessagePack bin
 * values containing little-endian data; no msgpackr-specific extensions or
 * record tables are used, so non-JavaScript clients can decode the protocol.
 *
 * @license MIT
 */

import { Packr } from 'msgpackr';
import type { EncodedBattleState } from './battle-tensors';
import type { PRNGSeed } from './prng';

export const BATTLE_WORKER_PROTOCOL_VERSION = 'pokezero-battle-worker-v1';
export const BATTLE_WORKER_DEFAULT_MAX_FRAME_SIZE = 64 * 1024 * 1024;
export const BATTLE_WORKER_BYTE_ORDER = 'little' as const;

export type WorkerSideID = 'p1' | 'p2';

export interface WorkerHelloRequest {
	type: 'hello';
	protocolVersion?: string;
}

export interface WorkerStartRequest {
	type: 'start';
	battleId: string;
	formatId?: 'gen9randombattle';
	battleSeed: PRNGSeed;
	teamSeeds?: { p1: PRNGSeed, p2: PRNGSeed };
	teams?: { p1: PokemonSet[], p2: PokemonSet[] };
	trainingTargets?: boolean;
	includeInputLog?: boolean;
}

export interface WorkerActionRequest {
	type: 'action';
	battleId: string;
	side: WorkerSideID;
	requestId: number;
	actionIndex: number;
}

export type BattleWorkerRequest = WorkerHelloRequest | WorkerStartRequest | WorkerActionRequest;

export interface PlayerSafeBattleEvent {
	schemaVersion: string;
	schemaHash: string;
	sequence: number;
	command: string;
	category: 'state' | 'transient' | 'cosmetic';
	args: string[];
	annotations: { key: string, value: string }[];
	actor?: { side: WorkerSideID, publicId: string | null };
	target?: { side: WorkerSideID, publicId: string | null };
	side?: WorkerSideID;
	effect?: string;
	stateChanging: boolean;
}

export interface PrivilegedOpponentPokemonTarget {
	targetId: string;
	targetIdKind: 'public' | 'learner';
	publicEntityId: string | null;
	initialTeamSlot: number;
	initial: {
		species: string,
		ability: string,
		item: string,
		teraType: string,
		moves: string[],
	};
	current: {
		species: string,
		ability: string,
		item: string,
		teraType: string,
		exactHp: number,
		maxHp: number,
		status: string,
		fainted: boolean,
		active: boolean,
		teamPosition: number,
		moves: { id: string, pp: number, maxPp: number }[],
	};
	publicKnowledge: {
		initial: {
			species: boolean,
			ability: boolean,
			item: boolean,
			teraType: boolean,
			moves: boolean[],
		},
		current: {
			exactHp: boolean,
			ability: boolean,
			item: boolean,
			pp: boolean[],
		},
	};
}

export interface PrivilegedTargets {
	schemaVersion: 'ps-gen9-randombattle-privileged-v1';
	observerSide: WorkerSideID;
	opponent: PrivilegedOpponentPokemonTarget[];
}

export interface WorkerHelloMessage {
	type: 'hello';
	protocolVersion: typeof BATTLE_WORKER_PROTOCOL_VERSION;
	frame: {
		length: 'uint32be',
		encoding: 'messagepack',
		maxBytes: number,
		tensorByteOrder: typeof BATTLE_WORKER_BYTE_ORDER,
	};
	capabilities: {
		formats: readonly ['gen9randombattle'],
		actions: number,
		multipleBattles: true,
		jointDecisions: true,
		privilegedTargets: true,
	};
	tensorSchemaVersion: string;
	tensorSchemaHash: string;
	contractSchemaHash: string;
	eventSchemaVersion: string;
	eventSchemaHash: string;
	randomBattleDataHash: string;
	simulatorCommit: string | null;
}

export interface WorkerStartedMessage {
	type: 'start';
	battleId: string;
	status: 'started';
	formatId: 'gen9randombattle';
	battleSeed: PRNGSeed;
	teamSeeds: { p1: PRNGSeed, p2: PRNGSeed } | null;
	trainingTargets: boolean;
}

export interface WorkerObservationMessage {
	type: 'observation';
	battleId: string;
	side: WorkerSideID;
	requestId?: number;
	jointDecisionGroup?: string;
	requestState: string;
	needsAction: false;
	playerObservation: EncodedBattleState | null;
	eventSchemaVersion: string;
	eventSchemaHash: string;
	events: PlayerSafeBattleEvent[];
	privilegedTargets?: PrivilegedTargets;
	retry?: { actionIndex: number, reason: string };
}

export interface WorkerDecisionMessage {
	type: 'decision';
	battleId: string;
	side: WorkerSideID;
	requestId: number;
	jointDecisionGroup: string;
	requestState: string;
	needsAction: true;
	playerObservation: EncodedBattleState;
	eventSchemaVersion: string;
	eventSchemaHash: string;
	events: PlayerSafeBattleEvent[];
	privilegedTargets?: PrivilegedTargets;
	retry?: { actionIndex: number, reason: string };
}

export interface WorkerActionMessage {
	type: 'action';
	battleId: string;
	side: WorkerSideID;
	requestId: number;
	jointDecisionGroup?: string;
	actionIndex: number;
	status: 'queued' | 'submitted' | 'rejected';
	reason?: string;
}

export interface WorkerTerminalMessage {
	type: 'terminal';
	battleId: string;
	winner: WorkerSideID | null;
	tie: boolean;
	turns: number;
	battleSeed: PRNGSeed;
	teamSeeds: { p1: PRNGSeed, p2: PRNGSeed } | null;
	inputLogDigest: string;
	inputLog?: string[];
}

export interface WorkerErrorMessage {
	type: 'error';
	code: string;
	message: string;
	battleId?: string;
	side?: WorkerSideID;
	requestId?: number;
	fatal: boolean;
}

export type BattleWorkerMessage =
	WorkerHelloMessage | WorkerStartedMessage | WorkerObservationMessage | WorkerDecisionMessage |
	WorkerActionMessage | WorkerTerminalMessage | WorkerErrorMessage;

export class BattleWorkerProtocolError extends Error {
	readonly code: string;

	constructor(code: string, message: string) {
		super(message);
		this.name = 'BattleWorkerProtocolError';
		this.code = code;
	}
}

const packr = new Packr({
	useRecords: false,
	mapsAsObjects: true,
	moreTypes: false,
	structuredClone: false,
	encodeUndefinedAsNil: true,
	copyBuffers: true,
});

const nativeLittleEndian = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

export function encodeMessagePackFrame(
	message: unknown, maxFrameSize = BATTLE_WORKER_DEFAULT_MAX_FRAME_SIZE
): Buffer {
	validateTensorEnvelopes(message);
	const payload = packr.pack(toWireValue(message));
	if (!payload.length) throw new BattleWorkerProtocolError('EMPTY_FRAME', `MessagePack payload cannot be empty`);
	if (payload.length > maxFrameSize) {
		throw new BattleWorkerProtocolError(
			'FRAME_TOO_LARGE', `MessagePack payload is ${payload.length} bytes; maximum is ${maxFrameSize}`
		);
	}
	const frame = Buffer.allocUnsafe(payload.length + 4);
	frame.writeUInt32BE(payload.length, 0);
	payload.copy(frame, 4);
	return frame;
}

export function decodeMessagePackPayload(payload: Buffer): unknown {
	if (!payload.length) throw new BattleWorkerProtocolError('EMPTY_FRAME', `MessagePack payload cannot be empty`);
	try {
		return packr.unpack(payload);
	} catch (error: any) {
		throw new BattleWorkerProtocolError('INVALID_MESSAGEPACK', error?.message || `${error}`);
	}
}

export class MessagePackFrameDecoder {
	readonly maxFrameSize: number;
	private buffer = Buffer.alloc(0);

	constructor(maxFrameSize = BATTLE_WORKER_DEFAULT_MAX_FRAME_SIZE) {
		if (!Number.isInteger(maxFrameSize) || maxFrameSize < 1 || maxFrameSize > 0xFFFFFFFF) {
			throw new RangeError(`Invalid maximum frame size ${maxFrameSize}`);
		}
		this.maxFrameSize = maxFrameSize;
	}

	push(chunk: Buffer | Uint8Array): unknown[] {
		if (!chunk.length) return [];
		const input = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk.buffer, chunk.byteOffset, chunk.byteLength);
		this.buffer = this.buffer.length ? Buffer.concat([this.buffer, input]) : Buffer.from(input);

		const messages: unknown[] = [];
		let offset = 0;
		while (this.buffer.length - offset >= 4) {
			const length = this.buffer.readUInt32BE(offset);
			if (!length) throw new BattleWorkerProtocolError('EMPTY_FRAME', `Frame length cannot be zero`);
			if (length > this.maxFrameSize) {
				throw new BattleWorkerProtocolError(
					'FRAME_TOO_LARGE', `Incoming frame is ${length} bytes; maximum is ${this.maxFrameSize}`
				);
			}
			if (this.buffer.length - offset - 4 < length) break;
			const payload = this.buffer.subarray(offset + 4, offset + 4 + length);
			messages.push(decodeMessagePackPayload(payload));
			offset += length + 4;
		}
		if (offset) this.buffer = Buffer.from(this.buffer.subarray(offset));
		if (this.buffer.length > this.maxFrameSize + 4) {
			throw new BattleWorkerProtocolError('FRAME_TOO_LARGE', `Buffered frame exceeds maximum size`);
		}
		return messages;
	}

	finish() {
		if (this.buffer.length) {
			throw new BattleWorkerProtocolError(
				'TRUNCATED_FRAME', `Input ended with ${this.buffer.length} bytes of an incomplete frame`
			);
		}
	}
}

export interface MessagePackFrameWriterOptions {
	maxFrameSize?: number;
	maxQueuedBytes?: number;
	maxQueuedFrames?: number;
}

export class MessagePackFrameWriter {
	readonly maxFrameSize: number;
	readonly maxQueuedBytes: number;
	readonly maxQueuedFrames: number;
	private readonly output: NodeJS.WritableStream;
	private readonly queue: Buffer[] = [];
	private queuedBytes = 0;
	private flushing = false;
	private failure: Error | null = null;
	private idleResolvers: (() => void)[] = [];

	constructor(output: NodeJS.WritableStream, options: MessagePackFrameWriterOptions = {}) {
		this.output = output;
		this.maxFrameSize = options.maxFrameSize ?? BATTLE_WORKER_DEFAULT_MAX_FRAME_SIZE;
		this.maxQueuedBytes = options.maxQueuedBytes ?? this.maxFrameSize * 2;
		this.maxQueuedFrames = options.maxQueuedFrames ?? 1024;
	}

	write(message: unknown) {
		if (this.failure) throw this.failure;
		const frame = encodeMessagePackFrame(message, this.maxFrameSize);
		if (this.queue.length >= this.maxQueuedFrames || this.queuedBytes + frame.length > this.maxQueuedBytes) {
			throw new BattleWorkerProtocolError(
				'OUTPUT_BACKPRESSURE',
				`Output queue limit exceeded (${this.queue.length} frames, ${this.queuedBytes} bytes queued)`
			);
		}
		this.queue.push(frame);
		this.queuedBytes += frame.length;
		if (!this.flushing) void this.flush();
	}

	async idle() {
		this.assertHealthy();
		if (!this.flushing && !this.queue.length) return;
		await new Promise<void>(resolve => {
			this.idleResolvers.push(resolve);
		});
		this.assertHealthy();
	}

	private assertHealthy() {
		if (this.failure) throw new BattleWorkerProtocolError('OUTPUT_FAILED', this.failure.message);
	}

	private async flush() {
		this.flushing = true;
		try {
			while (this.queue.length) {
				const frame = this.queue[0];
				if (!this.output.write(frame)) await this.waitForDrain();
				this.queue.shift();
				this.queuedBytes -= frame.length;
			}
		} catch (error: any) {
			this.failure = error instanceof Error ? error : new Error(`${error}`);
			this.queue.length = 0;
			this.queuedBytes = 0;
		} finally {
			this.flushing = false;
			for (const resolve of this.idleResolvers.splice(0)) resolve();
		}
	}

	private waitForDrain() {
		return new Promise<void>((resolve, reject) => {
			const onDrain = () => {
				cleanup();
				resolve();
			};
			const onError = (error: Error) => {
				cleanup();
				reject(error);
			};
			const cleanup = () => {
				this.output.removeListener('drain', onDrain);
				this.output.removeListener('error', onError);
			};
			this.output.once('drain', onDrain);
			this.output.once('error', onError);
		});
	}
}

function validateTensorEnvelopes(value: unknown, path = '$', seen = new Set<object>()) {
	if (!value || typeof value !== 'object') return;
	if (seen.has(value)) {
		throw new BattleWorkerProtocolError('CYCLIC_MESSAGE', `Cyclic value at ${path}`);
	}
	seen.add(value);
	if (Array.isArray(value)) {
		for (let i = 0; i < value.length; i++) validateTensorEnvelopes(value[i], `${path}[${i}]`, seen);
		seen.delete(value);
		return;
	}
	if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer) {
		seen.delete(value as object);
		return;
	}

	const record = value as { [key: string]: unknown };
	if (typeof record.dtype === 'string' && Array.isArray(record.shape) && record.data !== undefined) {
		const bytesPerElement = { float32: 4, int32: 4, uint8: 1 }[record.dtype];
		if (!bytesPerElement) {
			throw new BattleWorkerProtocolError('INVALID_TENSOR', `Unsupported dtype at ${path}: ${record.dtype}`);
		}
		let elements = 1;
		for (const dimension of record.shape) {
			if (!Number.isInteger(dimension) || (dimension as number) < 0) {
				throw new BattleWorkerProtocolError('INVALID_TENSOR', `Invalid shape at ${path}`);
			}
			elements *= dimension as number;
		}
		const data = record.data;
		if (!ArrayBuffer.isView(data)) {
			throw new BattleWorkerProtocolError('INVALID_TENSOR', `Tensor data at ${path} is not a typed array`);
		}
		if (data.byteLength !== elements * bytesPerElement) {
			throw new BattleWorkerProtocolError(
				'INVALID_TENSOR',
				`Tensor data at ${path} has ${data.byteLength} bytes; expected ${elements * bytesPerElement}`
			);
		}
	}
	for (const [key, child] of Object.entries(record)) {
		validateTensorEnvelopes(child, `${path}.${key}`, seen);
	}
	seen.delete(value);
}

function toWireValue(value: any, seen = new Map<object, any>()): any {
	if (!value || typeof value !== 'object') return value;
	if (Buffer.isBuffer(value)) return Buffer.from(value);
	if (ArrayBuffer.isView(value)) return typedArrayToLittleEndianBuffer(value);
	if (value instanceof ArrayBuffer) return typedArrayToLittleEndianBuffer(new Uint8Array(value));
	if (seen.has(value)) throw new BattleWorkerProtocolError('CYCLIC_MESSAGE', `Cannot encode cyclic message`);
	if (typeof value.dtype === 'string' && Array.isArray(value.shape) && ArrayBuffer.isView(value.data)) {
		return {
			dtype: value.dtype,
			shape: [...value.shape],
			data: typedArrayToLittleEndianBuffer(value.data),
		};
	}

	if (Array.isArray(value)) {
		const result: any[] = [];
		seen.set(value, result);
		for (const child of value) result.push(toWireValue(child, seen));
		return result;
	}
	const result: { [key: string]: any } = {};
	seen.set(value, result);
	for (const [key, child] of Object.entries(value)) {
		if (child !== undefined) result[key] = toWireValue(child, seen);
	}
	return result;
}

function typedArrayToLittleEndianBuffer(view: ArrayBufferView): Buffer {
	const bytes = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
	const elementSize = (view as ArrayBufferView & { BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT || 1;
	if (nativeLittleEndian || elementSize === 1) return Buffer.from(bytes);

	const result = Buffer.allocUnsafe(view.byteLength);
	for (let offset = 0; offset < bytes.length; offset += elementSize) {
		for (let byte = 0; byte < elementSize; byte++) {
			result[offset + byte] = bytes[offset + elementSize - byte - 1];
		}
	}
	return result;
}
