/**
 * Simulator
 * Pokemon Showdown - http://pokemonshowdown.com/
 *
 * Here's where all the simulator APIs get exported for general use.
 * `require('pokemon-showdown')` imports from here.
 *
 * @license MIT
 */

// battle simulation

export { Battle } from './battle';
export {
	canonicalSpeciesId,
	decodeGen9RandomBattleAction,
	encodeBattleState,
	encodeOmniscientBattleState,
	encodePlayerBattleState,
	gen9RandomBattleRequestNeedsAction,
	getGen9RandomBattleRequestState,
	GEN9_RANDOM_BATTLE_ACTION_LABELS,
	GEN9_RANDOM_BATTLE_TENSOR_MANIFEST,
	GEN9_SINGLES_ACTION_LABELS,
	isGen9RevivalBlessingRequest,
	type EncodedBattleState,
	type EncodedTensor,
	type Gen9RandomBattleEntityIds,
	type Gen9RandomBattleRequestState,
	type Gen9RandomBattleResult,
	type Gen9RandomBattleTensorManifest,
} from './battle-tensors';
export { BattleStream, getPlayerStreams } from './battle-stream';
export {
	GEN9_RANDOM_BATTLE_EVENT_SCHEMA_HASH,
	GEN9_RANDOM_BATTLE_EVENT_SCHEMA_VERSION,
	Gen9RandomBattleObservationTracker,
	type Gen9RandomBattleEvent,
	type Gen9RandomBattleEventCategory,
	type Gen9RandomBattleEventEntityRef,
	type Gen9RandomBattleObservationOptions,
	type Gen9RandomBattleObservationUpdate,
	type Gen9RandomBattleProtocolAnnotation,
} from './battle-observation';
export { Pokemon } from './pokemon';
export { PRNG } from './prng';
export { Side } from './side';

// dex API

export { Dex, toID } from './dex';

// teams API

export { Teams } from './teams';
export { TeamValidator } from './team-validator';

// misc libraries

export * from '../lib';
