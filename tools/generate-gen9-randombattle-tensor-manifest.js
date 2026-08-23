'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const { Dex, toID } = require('../dist/sim/dex');
const { canonicalSpeciesId } = require('../dist/sim/battle-tensors');
const randomSets = require('../data/random-battles/gen9/sets.json');

const SCHEMA_VERSION = 'ps-gen9-randombattle-v4';
const EVENT_SCHEMA_VERSION = 'ps-gen9-randombattle-events-v1';
const MAX_TEAM_SIZE = 6;
const MAX_MOVE_SLOTS = 4;
const BOOST_IDS = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion'];
const STAT_IDS = ['atk', 'def', 'spa', 'spd', 'spe'];
const PSEUDOWEATHER_IDS = ['trickroom', 'gravity', 'magicroom', 'wonderroom'];
const SIDE_CONDITION_IDS = [
	'stealthrock', 'spikes', 'toxicspikes', 'stickyweb',
	'reflect', 'lightscreen', 'auroraveil', 'tailwind',
	'safeguard', 'mist', 'luckychant',
];

// Commands are deliberately classified in the checked-in contract. Consumers can run the
// observation tracker in strict mode to fail closed when the simulator starts emitting a
// protocol command which has not been audited for player visibility and training semantics.
const STATE_EVENT_COMMANDS = [
	'request', 'turn', 'win', 'tie', 'switch', 'drag', 'replace', 'detailschange', 'faint',
	'-damage', '-heal', '-sethp', '-status', '-curestatus', '-cureteam',
	'-boost', '-unboost', '-setboost', '-clearboost', '-clearpositiveboost',
	'-clearnegativeboost', '-clearallboost', '-swapboost', '-copyboost', '-invertboost',
	'-item', '-enditem', '-ability', '-endability', '-terastallize', '-transform', '-formechange',
	'-start', '-end', '-singlemove', '-singleturn', '-mustrecharge', '-weather',
	'-fieldstart', '-fieldend', '-sidestart', '-sideend', '-swapsideconditions', 'upkeep',
	'swap', 'clearpoke', 'poke', 'updatepoke',
];
const TRANSIENT_EVENT_COMMANDS = [
	'move', 'cant', 'error', 'choice-rejected', '-activate', '-anim', '-block', '-burst', '-candynamax', '-center',
	'-combine', '-crit', '-fail', '-fieldactivate', '-hitcount', '-immune', '-mega', '-miss',
	'-notarget', '-nothing', '-ohko', '-prepare', '-primal', '-resisted', '-supereffective',
	'-waiting', '-zbroken', '-zpower',
];
const COSMETIC_EVENT_COMMANDS = [
	'', 'gen', 'gametype', 'tier', 'teamsize', 'player', 'rated', 'rule', 'start', 'teampreview',
	'inactive', 'inactiveoff', 't:', 't', 'message', '-message', 'html', 'raw', 'j', 'join', 'l',
	'leave', 'n', 'name', 'uhtml', 'uhtmlchange', '-hint', 'debug', 'bigerror', 'custom', 'showteam',
	'event',
];

function vocabulary(values) {
	return ['__none__', '__unknown__', ...new Set(values.map(toID).filter(Boolean))].sort((a, b) => {
		if (a === '__none__') return -1;
		if (b === '__none__') return 1;
		if (a === '__unknown__') return -1;
		if (b === '__unknown__') return 1;
		return a.localeCompare(b);
	});
}

function pokemonLabels(prefix, continuous, categorical, binary) {
	continuous.push(`${prefix}.hp`, `${prefix}.level`);
	for (const boost of BOOST_IDS) continuous.push(`${prefix}.boost.${boost}`);
	for (const stat of STAT_IDS) continuous.push(`${prefix}.stat.${stat}`);

	categorical.push(
		`${prefix}.species`, `${prefix}.ability`, `${prefix}.item`, `${prefix}.teraType`,
		`${prefix}.terastallized`, `${prefix}.type1`, `${prefix}.type2`, `${prefix}.status`
	);

	for (const field of [
		'present', 'active', 'revealed', 'fainted', 'hpKnown', 'levelKnown', 'speciesKnown',
		'abilityKnown', 'itemKnown', 'teraTypeKnown', 'typeKnown', 'statusKnown', 'statsKnown',
	]) {
		binary.push(`${prefix}.${field}`);
	}

	for (let i = 1; i <= MAX_MOVE_SLOTS; i++) {
		continuous.push(`${prefix}.move${i}.pp`);
		categorical.push(`${prefix}.move${i}.id`);
		binary.push(
			`${prefix}.move${i}.present`, `${prefix}.move${i}.revealed`,
			`${prefix}.move${i}.ppKnown`, `${prefix}.move${i}.disabled`
		);
	}
}

function tensorFields() {
	const continuous = [
		'battle.turn', 'battle.weatherDuration', 'battle.terrainDuration',
		'you.pokemonLeft', 'you.totalFainted',
		'foe.pokemonLeft', 'foe.totalFainted', 'foe.revealedCount',
	];
	const categorical = ['battle.weather', 'battle.terrain', 'battle.request', 'battle.result'];
	const binary = [
		'battle.ended',
		...PSEUDOWEATHER_IDS.map(id => `battle.pseudoWeather.${id}`),
		'you.teraUsed', 'foe.teraUsed', 'you.canTerastallize', 'you.trapped',
		'you.maybeTrapped', 'you.maybeDisabled', 'you.maybeLocked', 'you.noCancel',
		'battle.needsAction', 'battle.isRetry', 'battle.isRevivalRequest',
	];

	for (const side of ['you', 'foe']) {
		for (const condition of SIDE_CONDITION_IDS) {
			continuous.push(`${side}.sideCondition.${condition}`);
		}
		for (let i = 1; i <= MAX_TEAM_SIZE; i++) {
			pokemonLabels(`${side}.slot${i}`, continuous, categorical, binary);
		}
	}
	return { continuous, categorical, binary };
}

const dex = Dex.forFormat('gen9randombattle');
const randomSpecies = Object.keys(randomSets).map(id => dex.species.get(id));
const randomBaseSpecies = new Set(randomSpecies.map(species => species.baseSpecies));
// Cosmetic formes (Florges-Yellow, Vivillon-Jungle, ...) are mechanically identical to their base
// species and are normalized away by the encoder (`canonicalSpeciesId` in battle-tensors.ts), so
// the vocabulary should express exactly the base-species tokens the encoder can emit, not the
// cosmetic variants the random-battle team generator happens to sample.
const species = dex.species.all()
	.filter(entry => entry.exists && randomBaseSpecies.has(entry.baseSpecies))
	.map(entry => entry.id)
	.filter(id => canonicalSpeciesId(dex, id) === id);
const moves = ['struggle', 'recharge'];
for (const speciesData of Object.values(randomSets)) {
	for (const set of speciesData.sets) {
		moves.push(...set.movepool);
	}
}
// Current ability is dynamic public state, not just an initial-set field. Form changes (notably
// Terapagos) and ability-copying effects can expose standard abilities absent from sets.json.
const abilities = dex.abilities.all()
	.filter(ability => ability.exists && (ability.isNonstandard === null || ability.isNonstandard === 'Past'))
	.map(ability => ability.id);
const items = dex.items.all()
	.filter(item => item.exists && (item.isNonstandard === null || item.isNonstandard === 'Past'))
	.map(item => item.id);

const eventCore = {
	schemaVersion: EVENT_SCHEMA_VERSION,
	stateCommands: STATE_EVENT_COMMANDS,
	transientCommands: TRANSIENT_EVENT_COMMANDS,
	cosmeticCommands: COSMETIC_EVENT_COMMANDS,
};
const events = {
	...eventCore,
	schemaHash: crypto.createHash('sha256').update(JSON.stringify(eventCore)).digest('hex'),
};
const randomBattleDataHash = crypto.createHash('sha256').update(JSON.stringify(randomSets)).digest('hex');

const core = {
	schemaVersion: SCHEMA_VERSION,
	supportedFormatIds: ['gen9randombattle'],
	reservedTokens: { none: 0, unknown: 1 },
	normalization: {
		maxTurns: 200,
		maxDuration: 8,
		maxSideConditionLayers: 3,
		maxStat: 1000,
		maxTeamSize: MAX_TEAM_SIZE,
		maxMoveSlots: MAX_MOVE_SLOTS,
	},
	vocabularies: {
		species: vocabulary(species),
		moves: vocabulary(moves),
		items: vocabulary(items),
		abilities: vocabulary(abilities),
		types: vocabulary(dex.types.all().map(type => type.name)),
		weather: vocabulary([
			'sunnyday', 'desolateland', 'raindance', 'primordialsea', 'sandstorm', 'snow', 'hail', 'deltastream',
		]),
		terrain: vocabulary(['electricterrain', 'grassyterrain', 'mistyterrain', 'psychicterrain']),
		statuses: vocabulary(['brn', 'frz', 'par', 'psn', 'slp', 'tox', 'fnt']),
		requestStates: vocabulary(['move', 'switch', 'revive', 'wait', 'retry', 'terminal']),
		results: vocabulary(['ongoing', 'win', 'loss', 'tie']),
	},
	fields: tensorFields(),
	events,
	eventSchemaVersion: events.schemaVersion,
	eventSchemaHash: events.schemaHash,
	randomBattleDataHash,
	actionCount: 14,
	actions: [
		'move:slot1', 'move:slot2', 'move:slot3', 'move:slot4',
		'tera:slot1', 'tera:slot2', 'tera:slot3', 'tera:slot4',
		'switch:slot1', 'switch:slot2', 'switch:slot3',
		'switch:slot4', 'switch:slot5', 'switch:slot6',
	],
};
const schemaHash = crypto.createHash('sha256').update(JSON.stringify(core)).digest('hex');
const manifest = { ...core, schemaHash, tensorSchemaHash: schemaHash };
const output = path.resolve(__dirname, '../data/random-battles/gen9/tensor-manifest.json');
fs.writeFileSync(output, `${JSON.stringify(manifest, null, 4)}\n`);
console.log(`Wrote ${path.relative(process.cwd(), output)} (${schemaHash})`);
