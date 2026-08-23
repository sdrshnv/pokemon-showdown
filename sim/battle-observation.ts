import { Dex, toID } from './dex';
import {
	canonicalSpeciesId,
	gen9RandomBattleRequestNeedsAction,
	getGen9RandomBattleRequestState,
	GEN9_RANDOM_BATTLE_ACTION_LABELS,
	GEN9_RANDOM_BATTLE_TENSOR_MANIFEST,
	isGen9RevivalBlessingRequest,
	type EncodedBattleState,
	type Gen9RandomBattleEntityIds,
	type Gen9RandomBattleRequestState,
	type Gen9RandomBattleResult,
} from './battle-tensors';
import type { ChoiceRequest, MoveRequest, PokemonSwitchRequestData } from './side';

const BOOST_IDS: BoostID[] = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion'];
const STAT_IDS = ['atk', 'def', 'spa', 'spd', 'spe'] as const;
const SIDE_CONDITIONS = new Set<ID>([
	'stealthrock', 'spikes', 'toxicspikes', 'stickyweb', 'reflect', 'lightscreen', 'auroraveil',
	'tailwind', 'safeguard', 'mist', 'luckychant',
] as ID[]);
const PSEUDOWEATHER = new Set<ID>(['trickroom', 'gravity', 'magicroom', 'wonderroom'] as ID[]);
const TERRAIN = new Set<ID>(['electricterrain', 'grassyterrain', 'mistyterrain', 'psychicterrain'] as ID[]);
const NONE = 0;
const UNKNOWN = 1;

export const GEN9_RANDOM_BATTLE_EVENT_SCHEMA_VERSION =
	GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.events.schemaVersion;
export const GEN9_RANDOM_BATTLE_EVENT_SCHEMA_HASH =
	GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.events.schemaHash;

export type Gen9RandomBattleEventCategory = 'state' | 'transient' | 'cosmetic';

export interface Gen9RandomBattleProtocolAnnotation {
	readonly key: string;
	readonly value: string;
}

export interface Gen9RandomBattleEventEntityRef {
	readonly side: SideID;
	readonly publicId: string | null;
}

export interface Gen9RandomBattleEvent {
	readonly schemaVersion: string;
	readonly schemaHash: string;
	readonly sequence: number;
	readonly command: string;
	readonly category: Gen9RandomBattleEventCategory;
	readonly stateChanging: boolean;
	readonly args: readonly string[];
	readonly annotations: readonly Gen9RandomBattleProtocolAnnotation[];
	readonly actor?: Gen9RandomBattleEventEntityRef;
	readonly target?: Gen9RandomBattleEventEntityRef;
	readonly side?: SideID;
	readonly effect?: ID;
}

export interface Gen9RandomBattleObservationUpdate {
	readonly eventSchemaVersion: string;
	readonly eventSchemaHash: string;
	readonly observation: EncodedBattleState | null;
	readonly events: readonly Gen9RandomBattleEvent[];
	readonly requestState: Gen9RandomBattleRequestState;
	readonly needsAction: boolean;
	readonly terminal: boolean;
	readonly result: Gen9RandomBattleResult;
}

export interface Gen9RandomBattleObservationOptions {
	/** Reject every protocol command which is not classified by the checked-in event manifest. */
	readonly strictEvents?: boolean;
}

interface ParsedDetails {
	species: ID;
	level: number;
	terastallized: ID;
}

interface ParsedCondition {
	hp: number;
	status: ID;
	fainted: boolean;
}

interface TrackedPokemon {
	publicId: string;
	ident: string;
	baseSpecies: ID;
	species: ID;
	level: number;
	hp: number;
	status: ID;
	fainted: boolean;
	active: boolean;
	ability: ID;
	baseAbility: ID;
	abilityKnown: boolean;
	item: ID;
	itemKnown: boolean;
	teraType: ID;
	teraTypeKnown: boolean;
	terastallized: ID;
	types: ID[];
	moves: ID[];
	movePP: Map<ID, number>;
	moveMaxPP: Map<ID, number>;
	movePPKnown: Set<ID>;
	baseMoves: ID[];
	baseMovePP: Map<ID, number>;
	baseMoveMaxPP: Map<ID, number>;
	baseMovePPKnown: Set<ID>;
	boosts: SparseBoostsTable;
	stats: Partial<StatsExceptHPTable>;
	transformed: boolean;
}

interface TimedEffect {
	id: ID;
	upkeeps: number;
}

interface IllusionSnapshot {
	previous: TrackedPokemon;
	candidate: TrackedPokemon;
}

interface LabelWriter {
	continuous: Float32Array;
	categorical: Int32Array;
	binary: Uint8Array;
}

const continuousIndex = labelIndex(GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.fields.continuous);
const categoricalIndex = labelIndex(GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.fields.categorical);
const binaryIndex = labelIndex(GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.fields.binary);
const vocabularyMaps = Object.fromEntries(Object.entries(GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.vocabularies).map(
	([name, values]) => [name, new Map(values.map((value, index) => [value, index]))]
)) as { [K in keyof typeof GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.vocabularies]: Map<string, number> };
const stateEventCommands = new Set(GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.events.stateCommands);
const transientEventCommands = new Set(GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.events.transientCommands);
const cosmeticEventCommands = new Set(GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.events.cosmeticCommands);

/**
 * Builds player observations exclusively from the protocol sent to that player.
 * Pass chunks read from the corresponding stream returned by getPlayerStreams.
 */
export class Gen9RandomBattleObservationTracker {
	readonly side: SideID;
	readonly formatid = 'gen9randombattle' as ID;
	readonly strictEvents: boolean;

	private request: ChoiceRequest | null = null;
	private initialized = false;
	private own: TrackedPokemon[] = [];
	private foe: TrackedPokemon[] = [];
	private active = new Map<SideID, TrackedPokemon>();
	private teamSize = new Map<SideID, number>();
	private sideConditions = new Map<SideID, Map<ID, number>>();
	private illusionSnapshots = new Map<SideID, IllusionSnapshot>();
	private nextPublicId = new Map<SideID, number>([['p1', 1], ['p2', 1]]);
	private players = new Map<SideID, string>();
	private eventSequence = 0;
	private turn = 0;
	private ended = false;
	private result: Gen9RandomBattleResult = 'ongoing';
	private weather: TimedEffect | null = null;
	private terrain: TimedEffect | null = null;
	private pseudoWeather = new Set<ID>();

	constructor(side: SideID, options: Gen9RandomBattleObservationOptions = {}) {
		if (side !== 'p1' && side !== 'p2') {
			throw new Error(`Gen 9 Random Battle observation tracker only supports p1 or p2`);
		}
		this.side = side;
		this.strictEvents = !!options.strictEvents;
		this.sideConditions.set('p1', new Map());
		this.sideConditions.set('p2', new Map());
	}

	receive(chunk: string): EncodedBattleState | null {
		return this.receiveUpdate(chunk).observation;
	}

	receiveUpdate(chunk: string): Gen9RandomBattleObservationUpdate {
		let receivedRequest = false;
		const events: Gen9RandomBattleEvent[] = [];
		for (const line of chunk.split('\n')) {
			if (!line.startsWith('|')) continue;
			const parts = line.slice(1).split('|');
			try {
				this.assertClassifiedCommand(parts[0]);
				if (parts[0] === 'request') receivedRequest = true;
				this.receiveParts(parts);
				this.receiveGenericReveal(parts.slice(1));
				events.push(this.structureEvent(parts));
			} catch (err: any) {
				throw new Error(`Invalid battle protocol line ${JSON.stringify(line)}: ${err.message}`);
			}
		}
		const requestState = getGen9RandomBattleRequestState(this.request, this.ended);
		return {
			eventSchemaVersion: GEN9_RANDOM_BATTLE_EVENT_SCHEMA_VERSION,
			eventSchemaHash: GEN9_RANDOM_BATTLE_EVENT_SCHEMA_HASH,
			observation: this.initialized && (receivedRequest || this.ended) ? this.encode() : null,
			events,
			requestState,
			needsAction: gen9RandomBattleRequestNeedsAction(this.request, this.ended),
			terminal: this.ended,
			result: this.result,
		};
	}

	encode(): EncodedBattleState {
		if (!this.initialized) {
			throw new Error(`Cannot encode an observation before receiving a choice request`);
		}
		const writer: LabelWriter = {
			continuous: new Float32Array(GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.fields.continuous.length),
			categorical: new Int32Array(GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.fields.categorical.length),
			binary: new Uint8Array(GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.fields.binary.length),
		};
		this.writeGlobal(writer);
		this.writeSide(writer, 'you', this.own, this.teamSize.get(this.side) || this.own.length, true);
		const foeSide = this.side === 'p1' ? 'p2' : 'p1';
		this.writeSide(writer, 'foe', this.foe, this.teamSize.get(foeSide) || 0, false);
		return {
			schemaVersion: GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.schemaVersion,
			schemaHash: GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.schemaHash,
			visibility: 'player', formatid: this.formatid, side: this.side,
			requestState: getGen9RandomBattleRequestState(this.request, this.ended),
			needsAction: gen9RandomBattleRequestNeedsAction(this.request, this.ended),
			result: this.result,
			entityIds: this.buildEntityIds(),
			continuous: {
				data: writer.continuous, shape: [writer.continuous.length],
				labels: GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.fields.continuous, dtype: 'float32',
			},
			categorical: {
				data: writer.categorical, shape: [writer.categorical.length],
				labels: GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.fields.categorical, dtype: 'int32',
			},
			binary: {
				data: writer.binary, shape: [writer.binary.length],
				labels: GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.fields.binary, dtype: 'uint8',
			},
			actionMask: this.buildActionMask(),
		};
	}

	decodeAction(actionIndex: number): string {
		const mask = this.buildActionMask();
		if (!Number.isInteger(actionIndex) || actionIndex < 0 || actionIndex >= mask.data.length) {
			throw new Error(`Invalid Gen 9 Random Battle action index: ${actionIndex}`);
		}
		if (!mask.data[actionIndex]) {
			throw new Error(`Illegal Gen 9 Random Battle action: ${GEN9_RANDOM_BATTLE_ACTION_LABELS[actionIndex]}`);
		}
		const request = asMoveRequest(this.request);
		if (actionIndex < 8) {
			if (!request) throw new Error(`Move action requested without an active move request`);
			const move = this.actionMoves(request)[actionIndex % 4];
			if (!move) throw new Error(`Move action has no request move`);
			return `move ${move.id}${actionIndex >= 4 ? ' terastallize' : ''}`;
		}
		return `switch ${actionIndex - 7}`;
	}

	/**
	 * Correlates a player-visible protocol ident with the opaque ID used by event and slot metadata.
	 * During an unrevealed Illusion this intentionally resolves the public disguise; callers must
	 * rebind privileged truth after the corresponding `replace` event.
	 */
	getPublicEntityIdForIdent(ident: string): string | null {
		if (!isPokemonIdent(ident)) return null;
		return this.findPokemon(ident)?.publicId || null;
	}

	private receiveParts(parts: string[]) {
		const [command, ...args] = parts;
		switch (command) {
		case 'gen':
			if (args[0] !== '9') throw new Error(`unsupported generation ${args[0]}`);
			break;
		case 'gametype':
			if (args[0] !== 'singles') throw new Error(`unsupported game type ${args[0]}`);
			break;
		case 'tier':
			if (toID(args[0]) !== this.formatid) throw new Error(`unsupported format ${args[0]}`);
			break;
		case 'teamsize': {
			const side = parseSide(args[0]);
			const size = Number(args[1]);
			if (!Number.isInteger(size) || size < 1 || size > 6) throw new Error(`invalid team size ${args[1]}`);
			this.teamSize.set(side, size);
			break;
		}
		case 'player': {
			const side = parseSide(args[0]);
			this.players.set(side, args[1] || '');
			break;
		}
		case 'request':
			this.receiveRequest(JSON.parse(args.join('|')));
			break;
		case 'turn':
			this.turn = Number(args[0]) || 0;
			break;
		case 'win':
			this.ended = true;
			this.request = null;
			this.result = args[0] === this.players.get(this.side) ? 'win' : 'loss';
			break;
		case 'tie':
			this.ended = true;
			this.request = null;
			this.result = 'tie';
			break;
		case 'switch': case 'drag':
			this.receiveSwitch(args);
			break;
		case 'replace':
			this.receiveReplace(args);
			break;
		case 'detailschange':
			this.receiveDetailsChange(args);
			break;
		case '-formechange':
			this.receiveFormeChange(args);
			break;
		case 'move':
			this.receiveMove(args);
			break;
		case 'faint':
			this.updatePokemon(args[0], pokemon => {
				pokemon.hp = 0; pokemon.status = 'fnt' as ID; pokemon.fainted = true;
			});
			break;
		case '-damage': case '-heal':
			this.updateCondition(args[0], args[1]);
			break;
		case '-sethp':
			for (let i = 0; i + 1 < args.length && !args[i].startsWith('['); i += 2) {
				this.updateCondition(args[i], args[i + 1]);
			}
			break;
		case '-status':
			this.updatePokemon(args[0], pokemon => {
				pokemon.status = toID(args[1]);
			});
			break;
		case '-curestatus':
			this.updatePokemon(args[0], pokemon => {
				pokemon.status = '' as ID;
			});
			break;
		case '-cureteam':
			for (const pokemon of this.teamForIdent(args[0])) pokemon.status = '' as ID;
			break;
		case '-boost': case '-unboost': case '-setboost':
			this.receiveBoost(command, args);
			break;
		case '-clearboost': case '-clearpositiveboost': case '-clearnegativeboost':
			this.clearBoosts(command, args[0]);
			break;
		case '-clearallboost':
			for (const pokemon of this.active.values()) pokemon.boosts = {};
			break;
		case '-swapboost':
			this.swapBoosts(args[0], args[1], args[2]);
			break;
		case '-copyboost':
			this.copyBoosts(args[0], args[1]);
			break;
		case '-invertboost':
			this.updatePokemon(args[0], pokemon => {
				for (const boost of BOOST_IDS) pokemon.boosts[boost] = -(pokemon.boosts[boost] || 0);
			});
			break;
		case '-item':
			this.updatePokemon(args[0], pokemon => {
				pokemon.item = toID(args[1]); pokemon.itemKnown = true;
			});
			break;
		case '-enditem':
			this.updatePokemon(args[0], pokemon => {
				pokemon.item = '' as ID; pokemon.itemKnown = true;
			});
			break;
		case '-ability':
			this.updatePokemon(args[0], pokemon => {
				pokemon.ability = toID(args[1]); pokemon.abilityKnown = true;
				if (!args.some(arg => arg.startsWith('[from]'))) pokemon.baseAbility = pokemon.ability;
			});
			break;
		case '-endability':
			this.updatePokemon(args[0], pokemon => {
				pokemon.ability = '' as ID; pokemon.abilityKnown = true;
			});
			break;
		case '-terastallize':
			this.updatePokemon(args[0], pokemon => {
				pokemon.terastallized = toID(args[1]); pokemon.teraType = pokemon.terastallized;
				pokemon.teraTypeKnown = true; pokemon.types = [pokemon.terastallized];
			});
			break;
		case '-transform':
			this.receiveTransform(args[0], args[1]);
			break;
		case '-start':
			this.receiveVolatileStart(args);
			break;
		case '-weather':
			if (!args.includes('[upkeep]')) this.weather = toID(args[0]) === 'none' ? null : { id: toID(args[0]), upkeeps: 0 };
			break;
		case '-fieldstart':
			this.fieldStart(args[0]);
			break;
		case '-fieldend':
			this.fieldEnd(args[0]);
			break;
		case '-sidestart':
			this.sideStart(args[0], args[1]);
			break;
		case '-sideend':
			this.sideEnd(args[0], args[1]);
			break;
		case '-swapsideconditions': {
			const p1 = this.sideConditions.get('p1')!;
			this.sideConditions.set('p1', this.sideConditions.get('p2')!);
			this.sideConditions.set('p2', p1);
			break;
		}
		case 'upkeep':
			if (this.weather) this.weather.upkeeps++;
			if (this.terrain) this.terrain.upkeeps++;
			break;
		}
	}

	private receiveRequest(request: ChoiceRequest) {
		if (!request?.side || request.side.id !== this.side) throw new Error(`request belongs to another side`);
		this.request = request;
		this.initialized = true;
		this.teamSize.set(this.side, request.side.pokemon.length);
		const previous = new Map(this.own.map(pokemon => [pokemon.ident, pokemon]));
		this.own = request.side.pokemon.map(
			(data, index) => this.ownFromRequest(data, previous.get(data.ident), index)
		);
		for (const pokemon of this.own) {
			if (pokemon.active) this.active.set(this.side, pokemon);
		}
	}

	private ownFromRequest(data: PokemonSwitchRequestData, previous: TrackedPokemon | undefined, index: number) {
		const details = parseDetails(data.details);
		const condition = parseCondition(data.condition);
		const activeRequest = asMoveRequest(this.request)?.active[0];
		const transformed = !!previous?.transformed && data.active;
		const moves = data.moves.map(toID);
		const movePP = new Map(previous?.movePP || []);
		const moveMaxPP = new Map(previous?.moveMaxPP || []);
		const movePPKnown = new Set(previous?.movePPKnown || []);
		for (const move of moves) {
			if (!moveMaxPP.has(move)) moveMaxPP.set(move, maximumMovePP(move));
			if (!movePP.has(move)) movePP.set(move, 1);
			movePPKnown.add(move);
		}
		for (const move of activeRequest?.moves || []) {
			if (typeof move.pp === 'number' && move.maxpp) {
				movePP.set(move.id, move.pp / move.maxpp);
				moveMaxPP.set(move.id, move.maxpp);
				movePPKnown.add(move.id);
			}
		}
		return {
			publicId: previous?.publicId || `you:${index + 1}`,
			ident: data.ident, baseSpecies: transformed ? previous.baseSpecies : details.species,
			species: transformed ? previous.species : details.species,
			level: details.level, hp: condition.hp,
			status: condition.status, fainted: condition.fainted, active: data.active,
			ability: data.ability || data.baseAbility, baseAbility: data.baseAbility, abilityKnown: true,
			item: data.item, itemKnown: true, teraType: toID(data.teraType), teraTypeKnown: true,
			terastallized: toID(data.terastallized),
			types: transformed ? [...previous.types] : pokemonTypes(details.species, toID(data.terastallized)),
			moves, movePP, moveMaxPP, movePPKnown,
			baseMoves: transformed ? [...previous.baseMoves] : [...moves],
			baseMovePP: transformed ? new Map(previous.baseMovePP) : new Map(movePP),
			baseMoveMaxPP: transformed ? new Map(previous.baseMoveMaxPP) : new Map(moveMaxPP),
			baseMovePPKnown: transformed ? new Set(previous.baseMovePPKnown) : new Set(movePPKnown),
			boosts: data.active ? previous?.boosts || {} : {}, stats: { ...data.stats }, transformed,
		} as TrackedPokemon;
	}

	private receiveSwitch(args: string[]) {
		const side = sideFromIdent(args[0]);
		this.illusionSnapshots.delete(side);
		const details = parseDetails(args[1]);
		const condition = parseCondition(args[2]);
		const team = side === this.side ? this.own : this.foe;
		for (const pokemon of team) {
			if (pokemon.active) pokemon.boosts = {};
			pokemon.active = false;
		}
		let pokemon = team.find(candidate => candidate.ident === inactiveIdent(args[0])) ||
			team.find(candidate => candidate.baseSpecies === details.species);
		let previous: TrackedPokemon | null = null;
		if (!pokemon) {
			pokemon = emptyPokemon(inactiveIdent(args[0]), this.allocatePublicId(side));
			team.push(pokemon);
		} else if (side !== this.side) {
			previous = clonePokemon(pokemon);
		}
		if (pokemon.transformed) restoreBaseMoves(pokemon);
		Object.assign(pokemon, {
			ident: inactiveIdent(args[0]), baseSpecies: details.species, species: details.species, level: details.level,
			hp: condition.hp, status: condition.status, fainted: condition.fainted, active: true,
			terastallized: details.terastallized, types: pokemonTypes(details.species, details.terastallized),
			boosts: {}, transformed: false,
		});
		if (pokemon.baseAbility) {
			pokemon.ability = pokemon.baseAbility; pokemon.abilityKnown = true;
		} else if (side !== this.side) {
			pokemon.ability = '' as ID; pokemon.abilityKnown = false;
		}
		if (details.terastallized) {
			pokemon.teraType = details.terastallized; pokemon.teraTypeKnown = true;
		}
		if (previous) {
			const candidate = clonePokemon(pokemon);
			candidate.publicId = this.allocatePublicId(side);
			candidate.ability = '' as ID; candidate.baseAbility = '' as ID; candidate.abilityKnown = false;
			candidate.item = '' as ID; candidate.itemKnown = false;
			candidate.moves = []; candidate.movePP.clear();
			this.illusionSnapshots.set(side, { previous, candidate });
		}
		this.active.set(side, pokemon);
	}

	private receiveReplace(args: string[]) {
		const side = sideFromIdent(args[0]);
		let pokemon = this.active.get(side);
		if (!pokemon) throw new Error(`replace has no active Pokémon`);
		const illusion = this.illusionSnapshots.get(side);
		if (illusion) {
			Object.assign(pokemon, illusion.previous);
			pokemon = illusion.candidate;
			this.teamForSide(side).push(pokemon);
			this.active.set(side, pokemon);
			this.illusionSnapshots.delete(side);
		}
		const details = parseDetails(args[1]);
		const condition = parseCondition(args[2]);
		pokemon.ident = inactiveIdent(args[0]); pokemon.baseSpecies = details.species;
		pokemon.species = details.species; pokemon.level = details.level;
		pokemon.hp = condition.hp; pokemon.status = condition.status; pokemon.fainted = condition.fainted;
		pokemon.terastallized = details.terastallized; pokemon.types = pokemonTypes(details.species, details.terastallized);
	}

	private receiveDetailsChange(args: string[]) {
		this.updatePokemon(args[0], pokemon => {
			const details = parseDetails(args[1]);
			pokemon.species = details.species; pokemon.level = details.level;
			pokemon.terastallized = details.terastallized;
			pokemon.types = pokemonTypes(details.species, details.terastallized);
			if (args[2]) Object.assign(pokemon, parseCondition(args[2]));
		});
	}

	private receiveFormeChange(args: string[]) {
		this.updatePokemon(args[0], pokemon => {
			pokemon.species = canonicalSpeciesId(Dex, toID(args[1]));
			pokemon.types = pokemonTypes(pokemon.species, pokemon.terastallized);
			if (args[2]) Object.assign(pokemon, parseCondition(args[2]));
		});
	}

	private receiveMove(args: string[]) {
		const move = toID(args[1]);
		const calledMove = args.slice(2).some(arg => arg.startsWith('[from]'));
		const target = args[2] && isPokemonIdent(args[2]) ? this.findPokemon(args[2]) : null;
		const ppCost = calledMove ? 0 : 1 + (target?.abilityKnown && target.ability === 'pressure' ? 1 : 0);
		this.updatePokemon(args[0], pokemon => {
			if (!pokemon.moves.includes(move) && pokemon.moves.length < 4) {
				pokemon.moves.push(move);
				const maxPP = pokemon.transformed ? 5 : maximumMovePP(move);
				pokemon.moveMaxPP.set(move, maxPP);
				pokemon.movePP.set(move, 1);
				pokemon.movePPKnown.add(move);
			}
			if (ppCost && pokemon.movePPKnown.has(move)) {
				const maxPP = pokemon.moveMaxPP.get(move) || maximumMovePP(move);
				pokemon.movePP.set(move, clamp01((pokemon.movePP.get(move) ?? 1) - ppCost / maxPP));
			}
			if (!pokemon.transformed) {
				if (!pokemon.baseMoves.includes(move) && pokemon.baseMoves.length < 4) pokemon.baseMoves.push(move);
				pokemon.baseMovePP.set(move, pokemon.movePP.get(move) ?? 1);
				pokemon.baseMoveMaxPP.set(move, pokemon.moveMaxPP.get(move) || maximumMovePP(move));
				if (pokemon.movePPKnown.has(move)) pokemon.baseMovePPKnown.add(move);
			}
		});
	}

	private receiveTransform(sourceIdent: string, targetIdent: string) {
		const source = this.findPokemon(sourceIdent);
		const target = this.findPokemon(targetIdent);
		if (!source || !target) return;
		source.species = target.species;
		source.types = [...target.types];
		source.boosts = { ...target.boosts };
		source.transformed = true;
		if (target.abilityKnown) {
			source.ability = target.ability;
			source.abilityKnown = true;
		}
		source.moves = [...target.moves];
		source.movePP = new Map(target.moves.map(move => [move, 1]));
		source.moveMaxPP = new Map(target.moves.map(move => [move, 5]));
		source.movePPKnown = new Set(target.movePPKnown);
	}

	private receiveVolatileStart(args: string[]) {
		const effect = effectID(args[1]);
		if (effect !== 'typechange' && effect !== 'typeadd') return;
		this.updatePokemon(args[0], pokemon => {
			const types = (args[2] || '').split('/').map(toID).filter(Boolean);
			if (effect === 'typechange') pokemon.types = types;
			if (effect === 'typeadd') pokemon.types = [...pokemon.types, ...types].slice(0, 2);
		});
	}

	private receiveBoost(command: string, args: string[]) {
		const boost = toID(args[1]) as BoostID;
		if (!BOOST_IDS.includes(boost)) return;
		const amount = Number(args[2]);
		if (!Number.isFinite(amount)) throw new Error(`invalid boost amount ${args[2]}`);
		this.updatePokemon(args[0], pokemon => {
			const current = pokemon.boosts[boost] || 0;
			pokemon.boosts[boost] = clampBoost(command === '-setboost' ? amount :
				current + (command === '-unboost' ? -amount : amount));
		});
	}

	private clearBoosts(command: string, ident: string) {
		this.updatePokemon(ident, pokemon => {
			for (const boost of BOOST_IDS) {
				const value = pokemon.boosts[boost] || 0;
				if (command === '-clearboost' || command === '-clearpositiveboost' && value > 0 ||
					command === '-clearnegativeboost' && value < 0) pokemon.boosts[boost] = 0;
			}
		});
	}

	private swapBoosts(sourceIdent: string, targetIdent: string, statText?: string) {
		const source = this.findPokemon(sourceIdent);
		const target = this.findPokemon(targetIdent);
		if (!source || !target) return;
		const stats = statText ? statText.split(',').map(toID) as BoostID[] : BOOST_IDS;
		for (const stat of stats) {
			const value = source.boosts[stat] || 0;
			source.boosts[stat] = target.boosts[stat] || 0;
			target.boosts[stat] = value;
		}
	}

	private copyBoosts(sourceIdent: string, targetIdent: string) {
		const source = this.findPokemon(sourceIdent);
		const target = this.findPokemon(targetIdent);
		if (source && target) target.boosts = { ...source.boosts };
	}

	private updateCondition(ident: string, text: string) {
		this.updatePokemon(ident, pokemon => Object.assign(pokemon, parseCondition(text)));
	}

	private fieldStart(effectText: string) {
		const effect = effectID(effectText);
		if (TERRAIN.has(effect)) this.terrain = { id: effect, upkeeps: 0 };
		if (PSEUDOWEATHER.has(effect)) this.pseudoWeather.add(effect);
	}

	private fieldEnd(effectText: string) {
		const effect = effectID(effectText);
		if (TERRAIN.has(effect)) this.terrain = null;
		this.pseudoWeather.delete(effect);
	}

	private sideStart(sideText: string, effectText: string) {
		const side = parseSide(sideText.split(':')[0]);
		const effect = effectID(effectText);
		if (!SIDE_CONDITIONS.has(effect)) return;
		const conditions = this.sideConditions.get(side)!;
		const maxLayers = effect === 'spikes' ? 3 : effect === 'toxicspikes' ? 2 : 1;
		conditions.set(effect, Math.min(maxLayers, (conditions.get(effect) || 0) + 1));
	}

	private sideEnd(sideText: string, effectText: string) {
		const side = parseSide(sideText.split(':')[0]);
		this.sideConditions.get(side)!.delete(effectID(effectText));
	}

	private updatePokemon(ident: string, update: (pokemon: TrackedPokemon) => void) {
		const pokemon = this.findPokemon(ident);
		if (!pokemon) return;
		update(pokemon);
		const illusion = this.illusionSnapshots.get(sideFromIdent(ident));
		if (illusion && illusion.candidate !== pokemon) update(illusion.candidate);
	}

	private findPokemon(ident: string) {
		const side = sideFromIdent(ident);
		const active = this.active.get(side);
		if (ident.includes(`${side}a:`) && active) return active;
		return this.teamForSide(side).find(pokemon => pokemon.ident === inactiveIdent(ident));
	}

	private teamForIdent(ident: string) {
		return this.teamForSide(sideFromIdent(ident));
	}

	private teamForSide(side: SideID) {
		return side === this.side ? this.own : this.foe;
	}

	private writeGlobal(writer: LabelWriter) {
		const norm = GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.normalization;
		setContinuous(writer, 'battle.turn', clamp01(this.turn / norm.maxTurns));
		setContinuous(writer, 'battle.weatherDuration', publicDuration(this.weather) / norm.maxDuration);
		setContinuous(writer, 'battle.terrainDuration', publicDuration(this.terrain) / norm.maxDuration);
		const foeSide = this.side === 'p1' ? 'p2' : 'p1';
		const ownFainted = this.own.filter(pokemon => pokemon.fainted).length;
		const foeFainted = this.foe.filter(pokemon => pokemon.fainted).length;
		setContinuous(writer, 'you.pokemonLeft', ((this.teamSize.get(this.side) || 0) - ownFainted) / norm.maxTeamSize);
		setContinuous(writer, 'you.totalFainted', ownFainted / norm.maxTeamSize);
		setContinuous(writer, 'foe.pokemonLeft', ((this.teamSize.get(foeSide) || 0) - foeFainted) / norm.maxTeamSize);
		setContinuous(writer, 'foe.totalFainted', foeFainted / norm.maxTeamSize);
		setContinuous(writer, 'foe.revealedCount', this.foe.length / norm.maxTeamSize);
		setCategorical(writer, 'battle.weather', vocab('weather', this.weather?.id));
		setCategorical(writer, 'battle.terrain', vocab('terrain', this.terrain?.id));
		setCategorical(writer, 'battle.request', vocab(
			'requestStates', getGen9RandomBattleRequestState(this.request, this.ended)
		));
		setCategorical(writer, 'battle.result', vocab('results', this.result));
		setBinary(writer, 'battle.ended', this.ended);
		for (const effect of PSEUDOWEATHER) setBinary(writer, `battle.pseudoWeather.${effect}`, this.pseudoWeather.has(effect));
		setBinary(writer, 'you.teraUsed', this.own.some(pokemon => !!pokemon.terastallized));
		setBinary(writer, 'foe.teraUsed', this.foe.some(pokemon => !!pokemon.terastallized));
		const active = asMoveRequest(this.request)?.active[0];
		setBinary(writer, 'you.canTerastallize', !!active?.canTerastallize);
		setBinary(writer, 'you.trapped', !!active?.trapped);
		setBinary(writer, 'you.maybeTrapped', !!active?.maybeTrapped);
		setBinary(writer, 'you.maybeDisabled', !!active?.maybeDisabled);
		setBinary(writer, 'you.maybeLocked', !!active?.maybeLocked);
		setBinary(writer, 'you.noCancel', !!this.request?.noCancel);
		setBinary(writer, 'battle.needsAction', gen9RandomBattleRequestNeedsAction(this.request, this.ended));
		setBinary(writer, 'battle.isRetry', !!this.request && 'update' in this.request && !!this.request.update);
		setBinary(writer, 'battle.isRevivalRequest', isGen9RevivalBlessingRequest(this.request));
		for (const side of ['p1', 'p2'] as SideID[]) {
			const prefix = side === this.side ? 'you' : 'foe';
			const conditions = this.sideConditions.get(side)!;
			for (const effect of SIDE_CONDITIONS) {
				setContinuous(writer, `${prefix}.sideCondition.${effect}`, (conditions.get(effect) || 0) / 3);
			}
		}
	}

	private writeSide(writer: LabelWriter, prefix: 'you' | 'foe', team: TrackedPokemon[], teamSize: number, own: boolean) {
		for (let i = 0; i < 6; i++) {
			const pokemon = team[i];
			const slot = `${prefix}.slot${i + 1}`;
			if (!pokemon) {
				if (!own && i < teamSize) this.writeUnknownPokemon(writer, slot);
				continue;
			}
			this.writePokemon(writer, slot, pokemon, own);
		}
	}

	private writeUnknownPokemon(writer: LabelWriter, prefix: string) {
		setBinary(writer, `${prefix}.present`, true);
		for (const field of ['species', 'ability', 'item', 'teraType', 'terastallized', 'type1', 'type2', 'status']) {
			setCategorical(writer, `${prefix}.${field}`, UNKNOWN);
		}
		for (let i = 1; i <= 4; i++) {
			setCategorical(writer, `${prefix}.move${i}.id`, UNKNOWN);
			setBinary(writer, `${prefix}.move${i}.present`, true);
		}
	}

	private writePokemon(writer: LabelWriter, prefix: string, pokemon: TrackedPokemon, own: boolean) {
		setContinuous(writer, `${prefix}.hp`, pokemon.hp);
		setContinuous(writer, `${prefix}.level`, clamp01(pokemon.level / 100));
		for (const boost of BOOST_IDS) setContinuous(writer, `${prefix}.boost.${boost}`, (pokemon.boosts[boost] || 0) / 6);
		for (const stat of STAT_IDS) {
			setContinuous(
				writer, `${prefix}.stat.${stat}`,
				own ? clamp01((pokemon.stats[stat] || 0) / GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.normalization.maxStat) : 0
			);
		}
		setCategorical(writer, `${prefix}.species`, vocab('species', pokemon.species));
		setCategorical(writer, `${prefix}.ability`, pokemon.abilityKnown ? vocab('abilities', pokemon.ability) : UNKNOWN);
		setCategorical(writer, `${prefix}.item`, pokemon.itemKnown ? vocab('items', pokemon.item) : UNKNOWN);
		setCategorical(writer, `${prefix}.teraType`, pokemon.teraTypeKnown ? vocab('types', pokemon.teraType) : UNKNOWN);
		setCategorical(writer, `${prefix}.terastallized`, vocab('types', pokemon.terastallized));
		setCategorical(writer, `${prefix}.type1`, vocab('types', pokemon.types[0]));
		setCategorical(writer, `${prefix}.type2`, vocab('types', pokemon.types[1]));
		setCategorical(writer, `${prefix}.status`, vocab('statuses', pokemon.status));
		for (const [field, value] of Object.entries({
			present: true, active: pokemon.active, revealed: true, fainted: pokemon.fainted,
			hpKnown: true, levelKnown: true, speciesKnown: true, abilityKnown: pokemon.abilityKnown,
			itemKnown: pokemon.itemKnown, teraTypeKnown: pokemon.teraTypeKnown, typeKnown: true, statusKnown: true,
			statsKnown: own,
		})) setBinary(writer, `${prefix}.${field}`, value);
		const activeRequest = own && pokemon.active ? asMoveRequest(this.request)?.active[0] : null;
		for (let i = 0; i < 4; i++) {
			const move = pokemon.moves[i];
			setContinuous(writer, `${prefix}.move${i + 1}.pp`, move ? pokemon.movePP.get(move) || 0 : 0);
			setCategorical(writer, `${prefix}.move${i + 1}.id`, move ? vocab('moves', move) : own ? NONE : UNKNOWN);
			setBinary(writer, `${prefix}.move${i + 1}.present`, own ? !!move : true);
			setBinary(writer, `${prefix}.move${i + 1}.revealed`, !!move);
			setBinary(writer, `${prefix}.move${i + 1}.ppKnown`, !!move && pokemon.movePPKnown.has(move));
			const disabled = !!move && !!activeRequest?.moves.find(requestMove => requestMove.id === move)?.disabled;
			setBinary(writer, `${prefix}.move${i + 1}.disabled`, disabled);
		}
	}

	private buildActionMask() {
		const data = new Uint8Array(GEN9_RANDOM_BATTLE_ACTION_LABELS.length);
		const request = this.request;
		const moveRequest = asMoveRequest(request);
		if (moveRequest) {
			const moves = this.actionMoves(moveRequest);
			for (let i = 0; i < 4; i++) {
				if (!moves[i] || moves[i]!.disabled) continue;
				data[i] = 1;
				if (moveRequest.active[0].canTerastallize) data[i + 4] = 1;
			}
			if (!moveRequest.active[0].trapped) this.writeSwitchMask(data, request!);
		} else if (request && 'forceSwitch' in request && request.forceSwitch) {
			this.writeSwitchMask(data, request);
		}
		if (gen9RandomBattleRequestNeedsAction(request, this.ended) && !data.some(Boolean)) {
			throw new Error(`Actionable Gen 9 Random Battle request produced an all-zero action mask`);
		}
		return { data, shape: [data.length] as const, labels: GEN9_RANDOM_BATTLE_ACTION_LABELS, dtype: 'uint8' as const };
	}

	private actionMoves(request: MoveRequest) {
		const active = this.own.find(pokemon => pokemon.active);
		const result = Array(4).fill(null) as (MoveRequest['active'][number]['moves'][number] | null)[];
		for (let i = 0; i < Math.min(active?.moves.length || 0, 4); i++) {
			result[i] = request.active[0].moves.find(move => move.id === active!.moves[i]) || null;
		}
		if (!result.some(Boolean) && request.active[0].moves.length === 1) result[0] = request.active[0].moves[0];
		return result;
	}

	private writeSwitchMask(data: Uint8Array, request: ChoiceRequest) {
		const revivalBlessing = isGen9RevivalBlessingRequest(request);
		for (let i = 0; i < Math.min(request.side.pokemon.length, 6); i++) {
			const pokemon = request.side.pokemon[i];
			if (revivalBlessing ? pokemon.condition.includes('fnt') : !pokemon.active && !pokemon.condition.includes('fnt')) {
				data[i + 8] = 1;
			}
		}
	}

	private assertClassifiedCommand(command: string) {
		if (stateEventCommands.has(command) || transientEventCommands.has(command) || cosmeticEventCommands.has(command)) {
			return;
		}
		if (this.strictEvents) throw new Error(`unclassified battle protocol command ${JSON.stringify(command)}`);
	}

	private receiveGenericReveal(args: readonly string[]) {
		const annotations = parseProtocolAnnotations(args);
		const ownerText = annotations.find(annotation => annotation.key === 'of')?.value ||
			args.find(isPokemonIdent);
		if (!ownerText || !isPokemonIdent(ownerText)) return;
		const sources = [
			...annotations.filter(annotation => annotation.key === 'from').map(annotation => annotation.value),
			...annotations.filter(annotation => annotation.key === 'item' || annotation.key === 'ability')
				.map(annotation => `${annotation.key}: ${annotation.value}`),
			...args.filter(arg => /^(?:item|ability):/i.test(arg)),
		];
		for (const source of sources) {
			const match = /^(item|ability):\s*(.+)$/i.exec(source);
			if (!match) continue;
			const effect = toID(match[2]);
			this.updatePokemon(ownerText, pokemon => {
				if (toID(match[1]) === 'item') {
					pokemon.item = effect; pokemon.itemKnown = true;
				} else {
					pokemon.ability = effect; pokemon.abilityKnown = true;
				}
			});
		}
	}

	private structureEvent(parts: readonly string[]): Gen9RandomBattleEvent {
		const [command, ...args] = parts;
		const idents = args.filter(isPokemonIdent);
		const actor = idents[0] ? this.eventEntityRef(idents[0]) : undefined;
		const target = idents[1] ? this.eventEntityRef(idents[1]) : undefined;
		let side = actor?.side;
		if (!side && args[0] && /^(?:p1|p2)(?::|$)/.test(args[0])) side = parseSide(args[0].slice(0, 2));
		return {
			schemaVersion: GEN9_RANDOM_BATTLE_EVENT_SCHEMA_VERSION,
			schemaHash: GEN9_RANDOM_BATTLE_EVENT_SCHEMA_HASH,
			sequence: this.eventSequence++,
			command,
			category: eventCategory(command),
			stateChanging: stateEventCommands.has(command),
			args,
			annotations: parseProtocolAnnotations(args),
			...(actor ? { actor } : {}),
			...(target ? { target } : {}),
			...(side ? { side } : {}),
			...(eventEffect(command, args) ? { effect: eventEffect(command, args) } : {}),
		};
	}

	private eventEntityRef(ident: string): Gen9RandomBattleEventEntityRef {
		return { side: sideFromIdent(ident), publicId: this.findPokemon(ident)?.publicId || null };
	}

	private allocatePublicId(side: SideID) {
		const next = this.nextPublicId.get(side) || 1;
		this.nextPublicId.set(side, next + 1);
		return `${side === this.side ? 'you' : 'foe'}:${next}`;
	}

	private buildEntityIds(): Gen9RandomBattleEntityIds {
		const maxTeamSize = GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.normalization.maxTeamSize;
		const you = Array<string | null>(maxTeamSize).fill(null);
		const foe = Array<string | null>(maxTeamSize).fill(null);
		for (let i = 0; i < Math.min(this.own.length, maxTeamSize); i++) you[i] = this.own[i].publicId;
		for (let i = 0; i < Math.min(this.foe.length, maxTeamSize); i++) foe[i] = this.foe[i].publicId;
		return { you, foe };
	}
}

function emptyPokemon(ident: string, publicId: string): TrackedPokemon {
	return {
		publicId, ident, baseSpecies: '' as ID, species: '' as ID,
		level: 100, hp: 0, status: '' as ID, fainted: false, active: false,
		ability: '' as ID, baseAbility: '' as ID, abilityKnown: false, item: '' as ID, itemKnown: false,
		teraType: '' as ID, teraTypeKnown: false, terastallized: '' as ID, types: [], moves: [],
		movePP: new Map(), moveMaxPP: new Map(), movePPKnown: new Set(),
		baseMoves: [], baseMovePP: new Map(), baseMoveMaxPP: new Map(), baseMovePPKnown: new Set(),
		boosts: {}, stats: {}, transformed: false,
	};
}

function clonePokemon(pokemon: TrackedPokemon): TrackedPokemon {
	return {
		...pokemon,
		types: [...pokemon.types],
		moves: [...pokemon.moves],
		movePP: new Map(pokemon.movePP),
		moveMaxPP: new Map(pokemon.moveMaxPP),
		movePPKnown: new Set(pokemon.movePPKnown),
		baseMoves: [...pokemon.baseMoves],
		baseMovePP: new Map(pokemon.baseMovePP),
		baseMoveMaxPP: new Map(pokemon.baseMoveMaxPP),
		baseMovePPKnown: new Set(pokemon.baseMovePPKnown),
		boosts: { ...pokemon.boosts },
		stats: { ...pokemon.stats },
	};
}

function restoreBaseMoves(pokemon: TrackedPokemon) {
	pokemon.moves = [...pokemon.baseMoves];
	pokemon.movePP = new Map(pokemon.baseMovePP);
	pokemon.moveMaxPP = new Map(pokemon.baseMoveMaxPP);
	pokemon.movePPKnown = new Set(pokemon.baseMovePPKnown);
	pokemon.transformed = false;
}

function parseDetails(text: string): ParsedDetails {
	if (!text) throw new Error(`missing Pokémon details`);
	const parts = text.split(',').map(part => part.trim());
	let level = 100;
	let terastallized = '' as ID;
	for (const part of parts.slice(1)) {
		if (/^L\d+$/.test(part)) level = Number(part.slice(1));
		if (part.startsWith('tera:')) terastallized = toID(part.slice(5));
	}
	return { species: canonicalSpeciesId(Dex, Dex.species.get(parts[0]).id), level, terastallized };
}

function parseCondition(text: string): ParsedCondition {
	const parts = (text || '').trim().split(/\s+/);
	const fainted = parts.includes('fnt');
	if (fainted || parts[0] === '0' || parts[0] === '0/0') {
		return { hp: 0, status: fainted ? 'fnt' as ID : '' as ID, fainted };
	}
	const match = /^(\d+)\/(\d+)/.exec(parts[0]);
	const hp = match && Number(match[2]) ? Number(match[1]) / Number(match[2]) : 0;
	return { hp: clamp01(hp), status: toID(parts[1]), fainted: false };
}

function pokemonTypes(species: ID, terastallized: ID) {
	return terastallized ? [terastallized] : Dex.species.get(species).types.map(toID);
}

function inactiveIdent(ident: string) {
	return ident.replace(/^(p[12])[a-f]:/, '$1:');
}

function sideFromIdent(ident: string) {
	return parseSide(ident.slice(0, 2));
}

function parseSide(text: string): SideID {
	if (text !== 'p1' && text !== 'p2') throw new Error(`invalid side ${text}`);
	return text;
}

function effectID(text: string) {
	return toID(text.includes(':') ? text.slice(text.indexOf(':') + 1) : text);
}

function asMoveRequest(request: ChoiceRequest | null): MoveRequest | null {
	return request && 'active' in request && request.active ? request : null;
}

function maximumMovePP(moveId: ID) {
	const move = Dex.moves.get(moveId);
	if (!move.exists || !move.pp) return 1;
	return move.noPPBoosts ? move.pp : Math.floor(move.pp * 8 / 5);
}

function isPokemonIdent(text: string) {
	return /^p[12](?:[a-f])?:\s/.test(text);
}

function parseProtocolAnnotations(args: readonly string[]): Gen9RandomBattleProtocolAnnotation[] {
	const result: Gen9RandomBattleProtocolAnnotation[] = [];
	for (const arg of args) {
		const match = /^\[([^\]]+)\]\s*(.*)$/.exec(arg);
		if (match) result.push({ key: toID(match[1]), value: match[2] });
	}
	return result;
}

function eventCategory(command: string): Gen9RandomBattleEventCategory {
	if (stateEventCommands.has(command)) return 'state';
	if (transientEventCommands.has(command)) return 'transient';
	return 'cosmetic';
}

function eventEffect(command: string, args: readonly string[]): ID | undefined {
	let value: string | undefined;
	if (command === 'move' || command === 'cant') value = args[1];
	if (command === '-item' || command === '-ability' || command === '-status' || command.includes('boost')) {
		value = args[1];
	}
	if (command === '-weather') value = args[0];
	if (command.startsWith('-field')) value = args[0];
	if (command.startsWith('-side')) value = args[1];
	if (['-start', '-end', '-singlemove', '-singleturn', '-activate'].includes(command)) value = args[1];
	if (!value || isPokemonIdent(value) || value.startsWith('[')) return undefined;
	return effectID(value);
}

function publicDuration(effect: TimedEffect | null) {
	if (!effect) return 0;
	return Math.max(0, (effect.upkeeps < 5 ? 5 : 8) - effect.upkeeps);
}

function vocab(name: keyof typeof vocabularyMaps, value?: string | ID) {
	if (!value) return NONE;
	return vocabularyMaps[name].get(toID(value)) ?? UNKNOWN;
}

function labelIndex(labels: readonly string[]) {
	return new Map(labels.map((label, index) => [label, index]));
}

function setContinuous(writer: LabelWriter, label: string, value: number) {
	const index = continuousIndex.get(label);
	if (index === undefined) throw new Error(`Unknown continuous tensor field ${label}`);
	writer.continuous[index] = Number.isFinite(value) ? value : 0;
}

function setCategorical(writer: LabelWriter, label: string, value: number) {
	const index = categoricalIndex.get(label);
	if (index === undefined) throw new Error(`Unknown categorical tensor field ${label}`);
	writer.categorical[index] = value;
}

function setBinary(writer: LabelWriter, label: string, value: boolean | number) {
	const index = binaryIndex.get(label);
	if (index === undefined) throw new Error(`Unknown binary tensor field ${label}`);
	writer.binary[index] = value ? 1 : 0;
}

function clampBoost(value: number) {
	return Math.max(-6, Math.min(6, value));
}

function clamp01(value: number) {
	return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}
