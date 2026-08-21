/**
 * Long-lived, multiplexed Pokezero simulator worker.
 *
 * @license MIT
 */

import { createHash } from 'node:crypto';
import {
	GEN9_RANDOM_BATTLE_EVENT_SCHEMA_HASH,
	GEN9_RANDOM_BATTLE_EVENT_SCHEMA_VERSION,
	Gen9RandomBattleObservationTracker,
	type Gen9RandomBattleEvent,
} from './battle-observation';
import { BattleStream, getPlayerStreams } from './battle-stream';
import { GEN9_RANDOM_BATTLE_TENSOR_MANIFEST, type EncodedBattleState } from './battle-tensors';
import { PRNG, type PRNGSeed } from './prng';
import type { ChoiceRequest } from './side';
import type { Pokemon } from './pokemon';
import { toID } from './dex';
import {
	BATTLE_WORKER_DEFAULT_MAX_FRAME_SIZE,
	BATTLE_WORKER_PROTOCOL_VERSION,
	type BattleWorkerMessage,
	type BattleWorkerRequest,
	type PlayerSafeBattleEvent,
	type PrivilegedOpponentPokemonTarget,
	type PrivilegedTargets,
	type WorkerActionRequest,
	type WorkerDecisionMessage,
	type WorkerObservationMessage,
	type WorkerSideID,
	type WorkerStartRequest,
} from './battle-worker-protocol';

const SIDES: readonly WorkerSideID[] = ['p1', 'p2'];
const MAX_BATTLE_ID_LENGTH = 128;

export interface BattleWorkerOptions {
	maxBattles?: number;
	maxFrameSize?: number;
	simulatorCommit?: string | null;
}

interface TrackerUpdate {
	observation: EncodedBattleState | null;
	events: PlayerSafeBattleEvent[];
	eventSchemaVersion: string;
	eventSchemaHash: string;
	requestState: string;
	needsAction: boolean;
	terminal: boolean;
}

interface SideChunkUpdate extends TrackerUpdate {
	request: ChoiceRequest | null | undefined;
	errors: string[];
}

interface PendingSideRequest extends SideChunkUpdate {
	side: WorkerSideID;
}

interface ActiveDecision {
	side: WorkerSideID;
	requestId: number;
	group: string;
	observation: EncodedBattleState;
	ready: boolean;
	actionIndex?: number;
	choice?: string;
	submitted: boolean;
}

interface PendingSimulatorError {
	actionIndex: number;
	reason: string;
	events: PlayerSafeBattleEvent[];
	timer: NodeJS.Immediate;
}

interface PersistentPublicKnowledge {
	species: boolean;
	ability: boolean;
	item: boolean;
	teraType: boolean;
	moves: Set<string>;
}

export class BattleWorker {
	readonly maxBattles: number;
	readonly maxFrameSize: number;
	readonly simulatorCommit: string | null;
	private readonly send: (message: BattleWorkerMessage) => void;
	private readonly battles = new Map<string, BattleSession>();
	private closed = false;

	constructor(send: (message: BattleWorkerMessage) => void, options: BattleWorkerOptions = {}) {
		this.send = send;
		this.maxBattles = options.maxBattles ?? 64;
		this.maxFrameSize = options.maxFrameSize ?? BATTLE_WORKER_DEFAULT_MAX_FRAME_SIZE;
		this.simulatorCommit = options.simulatorCommit ?? null;
		if (!Number.isInteger(this.maxBattles) || this.maxBattles < 1) {
			throw new RangeError(`Invalid maximum battle count ${this.maxBattles}`);
		}
		this.emitHello();
	}

	get battleCount() {
		return this.battles.size;
	}

	receive(message: unknown) {
		if (this.closed) throw new Error(`Battle worker is closed`);
		try {
			const request = validateWorkerRequest(message);
			switch (request.type) {
			case 'hello':
				if (request.protocolVersion && request.protocolVersion !== BATTLE_WORKER_PROTOCOL_VERSION) {
					this.emitError(
						'PROTOCOL_MISMATCH',
						`Client requested ${request.protocolVersion}; worker uses ${BATTLE_WORKER_PROTOCOL_VERSION}`
					);
					return;
				}
				this.emitHello();
				return;
			case 'start':
				this.startBattle(request);
				return;
			case 'action':
				this.handleAction(request);
				return;
			}
		} catch (error: any) {
			this.emitError(error?.code || 'INVALID_MESSAGE', error?.message || `${error}`);
		}
	}

	close() {
		if (this.closed) return;
		this.closed = true;
		for (const session of [...this.battles.values()]) session.close();
		this.battles.clear();
	}

	private startBattle(request: WorkerStartRequest) {
		try {
			validateStartRequest(request);
		} catch (error: any) {
			this.emitError('START_FAILED', error?.message || `${error}`, request.battleId);
			return;
		}
		if (this.battles.has(request.battleId)) {
			this.emitError('DUPLICATE_BATTLE', `Battle ${request.battleId} already exists`, request.battleId);
			return;
		}
		if (this.battles.size >= this.maxBattles) {
			this.emitError(
				'BATTLE_BACKPRESSURE', `Worker already hosts its maximum of ${this.maxBattles} battles`, request.battleId
			);
			return;
		}

		let session: BattleSession;
		try {
			session = new BattleSession(
				request,
				message => this.send(message),
				() => this.battles.delete(request.battleId)
			);
			this.battles.set(request.battleId, session);
			session.start();
			this.send({
				type: 'start',
				battleId: request.battleId,
				status: 'started',
				formatId: 'gen9randombattle',
				battleSeed: request.battleSeed,
				teamSeeds: request.teamSeeds || null,
				trainingTargets: !!request.trainingTargets,
			});
		} catch (error: any) {
			this.battles.delete(request.battleId);
			this.emitError(
				'START_FAILED', error?.message || `${error}`, request.battleId
			);
		}
	}

	private handleAction(request: WorkerActionRequest) {
		const session = this.battles.get(request.battleId);
		if (!session) {
			this.emitError('UNKNOWN_BATTLE', `Battle ${request.battleId} does not exist`, request.battleId);
			return;
		}
		session.receiveAction(request);
	}

	private emitHello() {
		this.send({
			type: 'hello',
			protocolVersion: BATTLE_WORKER_PROTOCOL_VERSION,
			frame: {
				length: 'uint32be',
				encoding: 'messagepack',
				maxBytes: this.maxFrameSize,
				tensorByteOrder: 'little',
			},
			capabilities: {
				formats: ['gen9randombattle'],
				actions: GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.actions.length,
				multipleBattles: true,
				jointDecisions: true,
				privilegedTargets: true,
			},
			tensorSchemaVersion: GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.schemaVersion,
			tensorSchemaHash: GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.tensorSchemaHash,
			contractSchemaHash: GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.schemaHash,
			eventSchemaVersion: GEN9_RANDOM_BATTLE_EVENT_SCHEMA_VERSION,
			eventSchemaHash: GEN9_RANDOM_BATTLE_EVENT_SCHEMA_HASH,
			randomBattleDataHash: GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.randomBattleDataHash,
			simulatorCommit: this.simulatorCommit,
		});
	}

	private emitError(code: string, message: string, battleId?: string) {
		this.send({ type: 'error', code, message, battleId, fatal: false });
	}
}

class BattleSession {
	readonly battleId: string;
	readonly battleSeed: PRNGSeed;
	readonly teamSeeds: { p1: PRNGSeed, p2: PRNGSeed } | null;
	readonly trainingTargets: boolean;
	readonly includeInputLog: boolean;
	private readonly send: (message: BattleWorkerMessage) => void;
	private readonly onClose: () => void;
	private readonly request: WorkerStartRequest;
	private readonly stream = new BattleStream({ keepAlive: true });
	private readonly streams = getPlayerStreams(this.stream);
	private readonly trackers = {
		p1: new Gen9RandomBattleObservationTracker('p1', { strictEvents: true }),
		p2: new Gen9RandomBattleObservationTracker('p2', { strictEvents: true }),
	};
	private readonly requestIds: Record<WorkerSideID, number> = { p1: 0, p2: 0 };
	private readonly eventSequences: Record<WorkerSideID, number> = { p1: -1, p2: -1 };
	private readonly initialSlots = new Map<Pokemon, { targetId: string, slot: number }>();
	private readonly publicIds: Record<WorkerSideID, Map<Pokemon, string>> = {
		p1: new Map(), p2: new Map(),
	};
	private readonly publicKnowledge: Record<WorkerSideID, Map<Pokemon, PersistentPublicKnowledge>> = {
		p1: new Map(), p2: new Map(),
	};
	private readonly pendingCycle = new Map<WorkerSideID, PendingSideRequest>();
	private readonly decisions = new Map<WorkerSideID, ActiveDecision>();
	private readonly pendingErrors = new Map<WorkerSideID, PendingSimulatorError>();
	private readonly terminalSides = new Set<WorkerSideID>();
	private jointGroup = 0;
	private started = false;
	private closed = false;
	private terminalSent = false;

	constructor(
		request: WorkerStartRequest,
		send: (message: BattleWorkerMessage) => void,
		onClose: () => void,
	) {
		this.battleId = request.battleId;
		this.request = request;
		this.battleSeed = request.battleSeed;
		this.teamSeeds = request.teamSeeds || null;
		this.trainingTargets = !!request.trainingTargets;
		this.includeInputLog = !!request.includeInputLog;
		this.send = send;
		this.onClose = onClose;
	}

	start() {
		if (this.started) throw new Error(`Battle ${this.battleId} was already started`);
		this.started = true;
		void this.pumpSide('p1');
		void this.pumpSide('p2');
		void this.drainStream(this.streams.omniscient);
		void this.drainStream(this.streams.spectator);
		void this.drainStream(this.streams.p3);
		void this.drainStream(this.streams.p4);

		const start = { formatid: 'gen9randombattle', seed: this.battleSeed };
		const p1 = this.playerOptions('p1');
		const p2 = this.playerOptions('p2');
		void this.streams.omniscient.write(
			`>start ${JSON.stringify(start)}\n` +
			`>player p1 ${JSON.stringify(p1)}\n` +
			`>player p2 ${JSON.stringify(p2)}`
		);
		this.captureInitialSlots();
	}

	receiveAction(request: WorkerActionRequest) {
		if (this.closed || this.terminalSent) {
			this.rejectAction(request, 'Battle is already terminal');
			return;
		}
		const decision = this.decisions.get(request.side);
		if (!decision || decision.requestId !== request.requestId) {
			this.rejectAction(request, `Stale or unknown request ID`);
			return;
		}
		if (!decision.ready) {
			this.rejectAction(request, `Joint decision group has not been fully emitted`);
			return;
		}
		if (decision.choice !== undefined || decision.submitted) {
			this.rejectAction(request, `An action is already queued for this request`);
			return;
		}

		let choice: string;
		try {
			choice = this.trackers[request.side].decodeAction(request.actionIndex);
		} catch (error: any) {
			const reason = error?.message || `${error}`;
			this.rejectAction(request, reason, decision.group);
			this.emitLocalRetry(decision, request.actionIndex, reason);
			return;
		}

		decision.actionIndex = request.actionIndex;
		decision.choice = choice;
		this.send({
			type: 'action', battleId: this.battleId, side: request.side,
			requestId: request.requestId, jointDecisionGroup: decision.group,
			actionIndex: request.actionIndex, status: 'queued',
		});
		this.submitReadyActions();
	}

	close() {
		if (this.closed) return;
		this.closed = true;
		for (const pending of this.pendingErrors.values()) clearImmediate(pending.timer);
		this.pendingErrors.clear();
		this.pendingCycle.clear();
		this.decisions.clear();
		void this.streams.omniscient.writeEnd();
		this.onClose();
	}

	private playerOptions(side: WorkerSideID): PlayerOptions {
		const options: PlayerOptions = { name: `Pokezero ${side}` };
		if (this.request.teams) {
			options.team = this.request.teams[side];
		} else {
			options.seed = this.request.teamSeeds![side];
		}
		return options;
	}

	private captureInitialSlots() {
		const battle = this.stream.battle;
		if (!battle) throw new Error(`Battle stream did not create a battle`);
		for (const side of SIDES) {
			for (const [index, pokemon] of battle.getSide(side).pokemon.entries()) {
				this.initialSlots.set(pokemon, { targetId: `learner:${side}:team:${index + 1}`, slot: index + 1 });
			}
		}
	}

	private async pumpSide(side: WorkerSideID) {
		try {
			for await (const chunk of this.streams[side]) {
				if (this.closed) return;
				this.handleSideChunk(side, chunk);
			}
		} catch (error: any) {
			if (!this.closed) this.fail('PLAYER_STREAM_FAILED', error);
		}
	}

	private async drainStream(stream: AsyncIterable<string>) {
		try {
			for await (const chunk of stream) {
				void chunk;
				if (this.closed) return;
			}
		} catch (error: any) {
			if (!this.closed) this.fail('BATTLE_STREAM_FAILED', error);
		}
	}

	private handleSideChunk(side: WorkerSideID, chunk: string) {
		const update = this.receiveTrackerUpdate(side, chunk);
		if (update.terminal) {
			this.terminalSides.add(side);
			if (update.events.length || update.observation) this.emitPassiveObservation(side, update);
			this.maybeEmitTerminal();
			return;
		}

		if (update.errors.length) {
			this.handleSimulatorError(side, update);
			if (!update.request) return;
		}
		if (update.request !== undefined && update.request !== null) {
			this.handleRequest(side, update);
			return;
		}
		if (update.events.length || update.observation) this.emitPassiveObservation(side, update);
	}

	private receiveTrackerUpdate(side: WorkerSideID, chunk: string): SideChunkUpdate {
		const update = this.trackers[side].receiveUpdate(chunk);
		const tracked: TrackerUpdate = {
			observation: update.observation,
			events: update.events.map(event => normalizeTrackedEvent(event, ++this.eventSequences[side])),
			eventSchemaVersion: update.eventSchemaVersion,
			eventSchemaHash: update.eventSchemaHash,
			requestState: update.requestState,
			needsAction: update.needsAction,
			terminal: update.terminal,
		};
		if (!tracked.observation) tracked.observation = this.tryEncode(side);
		return {
			...tracked,
			request: parseRequest(chunk),
			errors: parseErrors(chunk),
		};
	}

	private handleRequest(side: WorkerSideID, update: SideChunkUpdate) {
		const request = update.request!;
		const pendingError = this.pendingErrors.get(side);
		if ((request as any).update || pendingError) {
			if (pendingError) {
				clearImmediate(pendingError.timer);
				this.pendingErrors.delete(side);
				update.events = [...pendingError.events, ...update.events];
				this.emitRetry(side, update, pendingError.actionIndex, pendingError.reason);
			} else {
				const current = this.decisions.get(side);
				this.emitRetry(side, update, current?.actionIndex ?? -1, 'Simulator updated the request');
			}
			return;
		}

		this.pendingCycle.set(side, { ...update, side });
		if (SIDES.every(currentSide => this.pendingCycle.has(currentSide))) this.flushRequestCycle();
	}

	private flushRequestCycle() {
		const group = `${this.battleId}:joint:${++this.jointGroup}`;
		const messages: (WorkerDecisionMessage | WorkerObservationMessage)[] = [];
		const newDecisions: ActiveDecision[] = [];
		for (const side of SIDES) {
			const update = this.pendingCycle.get(side)!;
			const requestId = ++this.requestIds[side];
			const needsAction = requestNeedsAction(update.request!);
			const base = {
				battleId: this.battleId,
				side,
				requestId,
				jointDecisionGroup: group,
				requestState: update.requestState || requestState(update.request!),
				playerObservation: update.observation,
				eventSchemaVersion: update.eventSchemaVersion,
				eventSchemaHash: update.eventSchemaHash,
				events: update.events,
				...this.trainingPayload(side, update.observation, update.events),
			};
			if (needsAction) {
				if (!update.observation) throw new Error(`Actionable request for ${side} has no observation`);
				assertActionableMask(update.observation, side);
				const decision: ActiveDecision = {
					side, requestId, group, observation: update.observation,
					ready: false, submitted: false,
				};
				this.decisions.set(side, decision);
				newDecisions.push(decision);
				messages.push({ ...base, type: 'decision', needsAction: true, playerObservation: update.observation });
			} else {
				this.decisions.delete(side);
				messages.push({ ...base, type: 'observation', needsAction: false });
			}
		}
		this.pendingCycle.clear();
		for (const message of messages) this.send(message);
		for (const decision of newDecisions) decision.ready = true;
	}

	private emitPassiveObservation(side: WorkerSideID, update: TrackerUpdate) {
		this.send({
			type: 'observation',
			battleId: this.battleId,
			side,
			requestState: update.terminal ? 'terminal' : 'update',
			needsAction: false,
			playerObservation: update.observation,
			eventSchemaVersion: update.eventSchemaVersion,
			eventSchemaHash: update.eventSchemaHash,
			events: update.events,
			...this.trainingPayload(side, update.observation, update.events),
		});
	}

	private handleSimulatorError(side: WorkerSideID, update: SideChunkUpdate) {
		const decision = this.decisions.get(side);
		if (!decision?.submitted || decision.actionIndex === undefined) {
			this.fail('UNEXPECTED_SIMULATOR_ERROR', new Error(update.errors.join('; ')), side);
			return;
		}
		const reason = update.errors.join('; ');
		decision.submitted = false;
		decision.choice = undefined;
		this.send({
			type: 'action', battleId: this.battleId, side,
			requestId: decision.requestId, jointDecisionGroup: decision.group,
			actionIndex: decision.actionIndex, status: 'rejected', reason,
		});
		const timer = setImmediate(() => {
			const pending = this.pendingErrors.get(side);
			if (!pending || pending.timer !== timer || this.closed) return;
			this.pendingErrors.delete(side);
			this.emitRetry(side, update, pending.actionIndex, pending.reason);
		});
		this.pendingErrors.set(side, {
			actionIndex: decision.actionIndex,
			reason,
			events: update.events,
			timer,
		});
	}

	private emitLocalRetry(decision: ActiveDecision, actionIndex: number, reason: string) {
		const side = decision.side;
		const event: PlayerSafeBattleEvent = {
			schemaVersion: GEN9_RANDOM_BATTLE_EVENT_SCHEMA_VERSION,
			schemaHash: GEN9_RANDOM_BATTLE_EVENT_SCHEMA_HASH,
			sequence: ++this.eventSequences[side],
			command: 'choice-rejected',
			category: 'transient',
			args: [reason],
			annotations: [
				{ key: 'private', value: 'true' },
				{ key: 'actionIndex', value: `${actionIndex}` },
			],
			stateChanging: false,
		};
		this.emitRetry(side, {
			observation: this.trackers[side].encode(),
			events: [event],
			eventSchemaVersion: GEN9_RANDOM_BATTLE_EVENT_SCHEMA_VERSION,
			eventSchemaHash: GEN9_RANDOM_BATTLE_EVENT_SCHEMA_HASH,
			requestState: 'retry',
			needsAction: true,
			terminal: false,
			request: undefined,
			errors: [],
		}, actionIndex, reason);
	}

	private emitRetry(side: WorkerSideID, update: SideChunkUpdate, actionIndex: number, reason: string) {
		const current = this.decisions.get(side);
		if (!current) {
			this.fail('RETRY_WITHOUT_DECISION', new Error(reason), side);
			return;
		}
		const observation = update.observation || this.tryEncode(side);
		if (!observation) {
			this.fail('RETRY_WITHOUT_OBSERVATION', new Error(reason), side);
			return;
		}
		assertActionableMask(observation, side);
		const decision: ActiveDecision = {
			side,
			requestId: ++this.requestIds[side],
			group: current.group,
			observation,
			ready: false,
			submitted: false,
		};
		this.decisions.set(side, decision);
		this.send({
			type: 'decision',
			battleId: this.battleId,
			side,
			requestId: decision.requestId,
			jointDecisionGroup: decision.group,
			requestState: update.requestState || 'retry',
			needsAction: true,
			playerObservation: observation,
			eventSchemaVersion: update.eventSchemaVersion,
			eventSchemaHash: update.eventSchemaHash,
			events: update.events,
			retry: { actionIndex, reason },
			...this.trainingPayload(side, observation, update.events),
		});
		decision.ready = true;
	}

	private submitReadyActions() {
		const active = [...this.decisions.values()];
		if (!active.length || active.some(decision => decision.choice === undefined)) return;
		for (const side of SIDES) {
			const decision = this.decisions.get(side);
			if (!decision || decision.submitted) continue;
			decision.submitted = true;
			try {
				void this.streams[side].write(decision.choice!);
				this.send({
					type: 'action', battleId: this.battleId, side,
					requestId: decision.requestId, jointDecisionGroup: decision.group,
					actionIndex: decision.actionIndex!, status: 'submitted',
				});
			} catch (error: any) {
				this.fail('ACTION_SUBMISSION_FAILED', error, side, decision.requestId);
				return;
			}
		}
	}

	private rejectAction(request: WorkerActionRequest, reason: string, jointDecisionGroup?: string) {
		this.send({
			type: 'action', battleId: request.battleId, side: request.side,
			requestId: request.requestId, jointDecisionGroup,
			actionIndex: request.actionIndex, status: 'rejected', reason,
		});
	}

	private tryEncode(side: WorkerSideID) {
		try {
			return this.trackers[side].encode();
		} catch {
			return null;
		}
	}

	private trainingPayload(
		side: WorkerSideID,
		observation: EncodedBattleState | null,
		events: PlayerSafeBattleEvent[],
	): { privilegedTargets?: PrivilegedTargets } {
		if (!this.trainingTargets) return {};
		return { privilegedTargets: this.buildPrivilegedTargets(side, observation, events) };
	}

	private buildPrivilegedTargets(
		side: WorkerSideID,
		observation: EncodedBattleState | null,
		events: PlayerSafeBattleEvent[],
	): PrivilegedTargets {
		const battle = this.stream.battle;
		if (!battle) throw new Error(`Battle ${this.battleId} is not initialized`);
		const opponentSide = side === 'p1' ? 'p2' : 'p1';
		this.updatePublicIdentityBindings(side, observation);
		this.updatePublicKnowledge(side, observation, events);
		const opponent = battle.getSide(opponentSide).pokemon.map(pokemon => {
			const stable = this.initialSlots.get(pokemon);
			if (!stable) throw new Error(`Missing stable target ID for ${opponentSide} Pokémon`);
			const publicEntityId = this.publicIds[side].get(pokemon) || null;
			return privilegedPokemonTarget(
				pokemon,
				publicEntityId || stable.targetId,
				publicEntityId,
				stable.slot,
				this.publicKnowledge[side].get(pokemon),
				observation,
			);
		});
		opponent.sort((a, b) => a.initialTeamSlot - b.initialTeamSlot);
		return {
			schemaVersion: 'ps-gen9-randombattle-privileged-v1',
			observerSide: side,
			opponent,
		};
	}

	private updatePublicIdentityBindings(side: WorkerSideID, observation: EncodedBattleState | null) {
		if (!observation) return;
		const battle = this.stream.battle!;
		const opponentSide = side === 'p1' ? 'p2' : 'p1';
		const opponent = battle.getSide(opponentSide);
		const publicIds = this.publicIds[side];
		for (let slot = 0; slot < observation.entityIds.foe.length; slot++) {
			const publicId = observation.entityIds.foe[slot];
			if (!publicId || !binaryField(observation, `foe.slot${slot + 1}.active`)) continue;
			const active = opponent.active[0];
			if (active) bindPublicId(publicIds, active, publicId);
		}

		for (let slot = 0; slot < observation.entityIds.foe.length; slot++) {
			const publicId = observation.entityIds.foe[slot];
			if (!publicId || [...publicIds.values()].includes(publicId)) continue;
			const species = categoricalField(observation, `foe.slot${slot + 1}.species`, 'species');
			if (!species) continue;
			const candidates = opponent.pokemon.filter(pokemon => {
				if (publicIds.has(pokemon)) return false;
				return [toID(pokemon.set.species), pokemon.baseSpecies.id, pokemon.species.id].includes(species as ID);
			});
			if (candidates.length === 1) bindPublicId(publicIds, candidates[0], publicId);
		}
	}

	private updatePublicKnowledge(
		side: WorkerSideID,
		observation: EncodedBattleState | null,
		events: PlayerSafeBattleEvent[],
	) {
		if (!observation) return;
		const publicIds = this.publicIds[side];
		for (const [pokemon, publicId] of publicIds) {
			const slot = observation.entityIds.foe.indexOf(publicId);
			if (slot < 0) continue;
			const knowledge = getPersistentKnowledge(this.publicKnowledge[side], pokemon);
			const prefix = `foe.slot${slot + 1}`;
			const species = categoricalField(observation, `${prefix}.species`, 'species');
			const ability = categoricalField(observation, `${prefix}.ability`, 'abilities');
			const item = categoricalField(observation, `${prefix}.item`, 'items');
			const teraType = categoricalField(observation, `${prefix}.teraType`, 'types');
			if (binaryField(observation, `${prefix}.speciesKnown`) && species === toID(pokemon.set.species)) {
				knowledge.species = true;
			}
			if (binaryField(observation, `${prefix}.abilityKnown`) && ability === toID(pokemon.set.ability)) {
				knowledge.ability = true;
			}
			if (binaryField(observation, `${prefix}.itemKnown`) && item === toID(pokemon.set.item)) {
				knowledge.item = true;
			}
			if (binaryField(observation, `${prefix}.teraTypeKnown`) && teraType === toID(pokemon.set.teraType)) {
				knowledge.teraType = true;
			}
			for (let moveSlot = 1; moveSlot <= 4; moveSlot++) {
				if (!binaryField(observation, `${prefix}.move${moveSlot}.revealed`)) continue;
				const move = categoricalField(observation, `${prefix}.move${moveSlot}.id`, 'moves');
				if (move) knowledge.moves.add(move);
			}
		}

		for (const event of events) {
			const publicId = event.actor?.publicId;
			if (!publicId) continue;
			const pokemon = [...publicIds].find(([, id]) => id === publicId)?.[0];
			if (!pokemon) continue;
			const knowledge = getPersistentKnowledge(this.publicKnowledge[side], pokemon);
			const revealed = toID(event.args[1]);
			if (event.command === 'move' && pokemon.set.moves.some(move => toID(move) === revealed)) {
				knowledge.moves.add(revealed);
			}
			if (['-item', '-enditem'].includes(event.command) && revealed === toID(pokemon.set.item)) {
				knowledge.item = true;
			}
			if (['-ability', '-endability'].includes(event.command) && revealed === toID(pokemon.set.ability)) {
				knowledge.ability = true;
			}
			if (event.command === '-terastallize' && revealed === toID(pokemon.set.teraType)) {
				knowledge.teraType = true;
			}
			for (const annotation of event.annotations || []) {
				const match = /^(item|ability):\s*(.+)$/i.exec(annotation.value);
				if (!match) continue;
				if (toID(match[1]) === 'item' && toID(match[2]) === toID(pokemon.set.item)) knowledge.item = true;
				if (toID(match[1]) === 'ability' && toID(match[2]) === toID(pokemon.set.ability)) {
					knowledge.ability = true;
				}
			}
		}
	}

	private maybeEmitTerminal() {
		if (this.terminalSent || !SIDES.every(side => this.terminalSides.has(side))) return;
		const battle = this.stream.battle;
		if (!battle) return;
		this.terminalSent = true;
		const inputLog = [...battle.inputLog];
		let winner: WorkerSideID | null = null;
		for (const side of SIDES) {
			if (battle.getSide(side).name === battle.winner) winner = side;
		}
		this.send({
			type: 'terminal',
			battleId: this.battleId,
			winner,
			tie: !winner,
			turns: battle.turn,
			battleSeed: this.battleSeed,
			teamSeeds: this.teamSeeds,
			inputLogDigest: createHash('sha256').update(inputLog.join('\n')).digest('hex'),
			...(this.includeInputLog ? { inputLog } : {}),
		});
		this.close();
	}

	private fail(code: string, error: any, side?: WorkerSideID, requestId?: number) {
		if (this.closed) return;
		this.send({
			type: 'error',
			code,
			message: error?.message || `${error}`,
			battleId: this.battleId,
			side,
			requestId,
			fatal: true,
		});
		this.close();
	}
}

function validateWorkerRequest(message: unknown): BattleWorkerRequest {
	if (!message || typeof message !== 'object' || typeof (message as any).type !== 'string') {
		throw new Error(`Worker request must be an object with a type`);
	}
	const type = (message as any).type;
	if (!['hello', 'start', 'action'].includes(type)) throw new Error(`Unsupported worker request type ${type}`);
	if (type === 'action') {
		const action = message as WorkerActionRequest;
		validateBattleId(action.battleId);
		if (!SIDES.includes(action.side)) throw new Error(`Invalid side ${action.side}`);
		if (!Number.isSafeInteger(action.requestId) || action.requestId < 1) {
			throw new Error(`Invalid request ID ${action.requestId}`);
		}
		if (!Number.isInteger(action.actionIndex) || action.actionIndex < 0) {
			throw new Error(`Invalid action index ${action.actionIndex}`);
		}
	}
	return message as BattleWorkerRequest;
}

function validateStartRequest(request: WorkerStartRequest) {
	validateBattleId(request.battleId);
	if (request.formatId && request.formatId !== 'gen9randombattle') {
		throw new Error(`Unsupported battle format ${request.formatId}`);
	}
	validateSeed(request.battleSeed, 'battleSeed');
	if (!!request.teamSeeds === !!request.teams) {
		throw new Error(`Start request must provide exactly one of teamSeeds or teams`);
	}
	if (request.teamSeeds) {
		validateSeed(request.teamSeeds.p1, 'teamSeeds.p1');
		validateSeed(request.teamSeeds.p2, 'teamSeeds.p2');
	}
	if (request.teams) {
		for (const side of SIDES) {
			if (!Array.isArray(request.teams[side]) || !request.teams[side].length || request.teams[side].length > 6) {
				throw new Error(`teams.${side} must contain between one and six Pokémon`);
			}
		}
	}
}

function validateBattleId(battleId: unknown): asserts battleId is string {
	if (typeof battleId !== 'string' || !battleId.length || battleId.length > MAX_BATTLE_ID_LENGTH ||
		!/^[A-Za-z0-9_.:-]+$/.test(battleId)) {
		throw new Error(`Invalid battle ID ${JSON.stringify(battleId)}`);
	}
}

function validateSeed(seed: unknown, path: string): asserts seed is PRNGSeed {
	if (typeof seed !== 'string') throw new Error(`${path} must be a Pokémon Showdown PRNG seed string`);
	try {
		new PRNG(seed as PRNGSeed);
	} catch (error: any) {
		throw new Error(`Invalid ${path}: ${error?.message || error}`);
	}
}

function parseRequest(chunk: string): ChoiceRequest | null | undefined {
	let result: ChoiceRequest | null | undefined;
	for (const line of chunk.split('\n')) {
		if (!line.startsWith('|request|')) continue;
		result = JSON.parse(line.slice('|request|'.length));
	}
	return result;
}

function parseErrors(chunk: string) {
	const errors: string[] = [];
	for (const line of chunk.split('\n')) {
		if (line.startsWith('|error|')) errors.push(line.slice('|error|'.length));
	}
	return errors;
}

function requestNeedsAction(request: ChoiceRequest) {
	return !request.wait;
}

function requestState(request: ChoiceRequest) {
	if (request.wait) return 'wait';
	if (request.teamPreview) return 'team-preview';
	if (request.forceSwitch) return request.forceSwitch.some(Boolean) ? 'forced-switch' : 'wait';
	return 'move';
}

function assertActionableMask(observation: EncodedBattleState, side: WorkerSideID) {
	if (!observation.actionMask.data.some(Boolean)) {
		throw new Error(`Actionable request for ${side} has an all-zero action mask`);
	}
}

function normalizeTrackedEvent(event: Gen9RandomBattleEvent, sequence: number): PlayerSafeBattleEvent {
	return {
		schemaVersion: event.schemaVersion,
		schemaHash: event.schemaHash,
		sequence,
		command: event.command,
		category: event.category,
		args: [...event.args],
		annotations: event.annotations.map(annotation => ({ ...annotation })),
		...(event.actor && (event.actor.side === 'p1' || event.actor.side === 'p2') ? {
			actor: { side: event.actor.side, publicId: event.actor.publicId },
		} : {}),
		...(event.target && (event.target.side === 'p1' || event.target.side === 'p2') ? {
			target: { side: event.target.side, publicId: event.target.publicId },
		} : {}),
		...(event.side === 'p1' || event.side === 'p2' ? { side: event.side } : {}),
		...(event.effect ? { effect: event.effect } : {}),
		stateChanging: event.stateChanging,
	};
}

function privilegedPokemonTarget(
	pokemon: Pokemon,
	targetId: string,
	publicEntityId: string | null,
	initialTeamSlot: number,
	knowledge: PersistentPublicKnowledge | undefined,
	observation: EncodedBattleState | null,
): PrivilegedOpponentPokemonTarget {
	const initialMoves = pokemon.set.moves.map(move => toID(move));
	const publicSlot = publicEntityId && observation ? observation.entityIds.foe.indexOf(publicEntityId) : -1;
	const prefix = publicSlot >= 0 ? `foe.slot${publicSlot + 1}` : null;
	const publicMoves = new Map<string, number>();
	if (prefix && observation) {
		for (let moveSlot = 1; moveSlot <= 4; moveSlot++) {
			const move = categoricalField(observation, `${prefix}.move${moveSlot}.id`, 'moves');
			if (move) publicMoves.set(move, moveSlot);
		}
	}
	return {
		targetId,
		targetIdKind: publicEntityId ? 'public' : 'learner',
		publicEntityId,
		initialTeamSlot,
		initial: {
			species: toID(pokemon.set.species || pokemon.baseSpecies.id),
			ability: toID(pokemon.set.ability),
			item: toID(pokemon.set.item),
			teraType: toID(pokemon.set.teraType),
			moves: initialMoves,
		},
		current: {
			species: pokemon.species.id,
			ability: pokemon.ability,
			item: pokemon.item,
			teraType: toID(pokemon.teraType),
			exactHp: pokemon.hp,
			maxHp: pokemon.maxhp,
			status: pokemon.status,
			fainted: pokemon.fainted,
			active: pokemon.isActive,
			teamPosition: pokemon.position + 1,
			moves: pokemon.moveSlots.map(move => ({ id: move.id, pp: move.pp, maxPp: move.maxpp })),
		},
		publicKnowledge: {
			initial: {
				species: !!knowledge?.species,
				ability: !!knowledge?.ability,
				item: !!knowledge?.item,
				teraType: !!knowledge?.teraType,
				moves: initialMoves.map(move => !!knowledge?.moves.has(move)),
			},
			current: {
				exactHp: !!prefix && !!observation && binaryField(observation, `${prefix}.fainted`),
				ability: !!prefix && !!observation && binaryField(observation, `${prefix}.abilityKnown`),
				item: !!prefix && !!observation && binaryField(observation, `${prefix}.itemKnown`),
				pp: pokemon.moveSlots.map(move => {
					const moveSlot = publicMoves.get(move.id);
					return !!moveSlot && !!prefix && !!observation &&
						binaryField(observation, `${prefix}.move${moveSlot}.ppKnown`);
				}),
			},
		},
	};
}

function bindPublicId(bindings: Map<Pokemon, string>, pokemon: Pokemon, publicId: string) {
	for (const [otherPokemon, otherId] of bindings) {
		if (otherPokemon !== pokemon && otherId === publicId) bindings.delete(otherPokemon);
	}
	bindings.set(pokemon, publicId);
}

function getPersistentKnowledge(
	knowledge: Map<Pokemon, PersistentPublicKnowledge>, pokemon: Pokemon
): PersistentPublicKnowledge {
	let result = knowledge.get(pokemon);
	if (!result) {
		result = { species: false, ability: false, item: false, teraType: false, moves: new Set() };
		knowledge.set(pokemon, result);
	}
	return result;
}

function binaryField(observation: EncodedBattleState, label: string) {
	const index = observation.binary.labels.indexOf(label);
	return index >= 0 && !!observation.binary.data[index];
}

function categoricalField(
	observation: EncodedBattleState,
	label: string,
	vocabulary: 'species' | 'moves' | 'items' | 'abilities' | 'types',
) {
	const index = observation.categorical.labels.indexOf(label);
	if (index < 0) return '';
	const token = observation.categorical.data[index];
	if (token <= GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.reservedTokens.unknown) return '';
	return GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.vocabularies[vocabulary][token] || '';
}
