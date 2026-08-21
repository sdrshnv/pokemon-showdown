'use strict';

const crypto = require('crypto');

const assert = require('./../../assert');
const common = require('./../../common');
const Sim = require('./../../../dist/sim');
const randomSets = require('./../../../data/random-battles/gen9/sets.json');

const {
	decodeGen9RandomBattleAction,
	encodeBattleState,
	encodeOmniscientBattleState,
	Gen9RandomBattleObservationTracker,
	GEN9_RANDOM_BATTLE_EVENT_SCHEMA_HASH,
	GEN9_RANDOM_BATTLE_EVENT_SCHEMA_VERSION,
	GEN9_RANDOM_BATTLE_ACTION_LABELS,
	GEN9_RANDOM_BATTLE_TENSOR_MANIFEST,
} = Sim;
const { extractChannelMessages } = require('./../../../dist/sim/battle');

const TEAMS = [[
	{
		species: 'Blissey', ability: 'Natural Cure', item: 'Heavy-Duty Boots', teraType: 'Fairy',
		moves: ['softboiled', 'seismictoss', 'protect', 'toxic'],
	},
	{
		species: 'Chansey', ability: 'Natural Cure', item: 'Eviolite', teraType: 'Normal',
		moves: ['softboiled', 'seismictoss', 'stealthrock', 'thunderwave'],
	},
], [
	{
		species: 'Snorlax', ability: 'Thick Fat', item: 'Leftovers', teraType: 'Normal',
		moves: ['rest', 'sleeptalk', 'bodyslam', 'earthquake'],
	},
	{
		species: 'Mew', ability: 'Synchronize', item: 'Leftovers', teraType: 'Psychic',
		moves: ['psychic', 'protect', 'uturn', 'willowisp'],
	},
]];

function publicHpBucket(hp, maxhp) {
	let percentage = Math.ceil(100 * hp / maxhp);
	if (percentage === 100 && hp < maxhp) percentage = 99;
	return percentage;
}

function findSharedPublicHpPair(maxhp) {
	for (let hp = maxhp - 1; hp > 1; hp--) {
		if (publicHpBucket(hp, maxhp) === publicHpBucket(hp - 1, maxhp)) return [hp - 1, hp];
	}
	throw new Error(`No matching public HP pair found for max HP ${maxhp}`);
}

function indexOf(tensor, label) {
	const index = tensor.labels.indexOf(label);
	assert.notEqual(index, -1, `Missing tensor field ${label}`);
	return index;
}

describe('Gen 9 Random Battle tensors', () => {
	let battle;
	beforeEach(() => {
		battle = common.createBattle({ formatid: 'gen9randombattle' }, TEAMS);
	});
	afterEach(() => {
		battle.destroy();
		battle = null;
	});

	it('should expose an immutable manifest with a valid schema hash', () => {
		const { schemaHash, tensorSchemaHash, ...core } = GEN9_RANDOM_BATTLE_TENSOR_MANIFEST;
		const actualHash = crypto.createHash('sha256').update(JSON.stringify(core)).digest('hex');
		assert.equal(actualHash, schemaHash);
		assert.equal(tensorSchemaHash, schemaHash);
		const { schemaHash: eventSchemaHash, ...eventCore } = GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.events;
		assert.equal(crypto.createHash('sha256').update(JSON.stringify(eventCore)).digest('hex'), eventSchemaHash);
		assert.equal(eventSchemaHash, GEN9_RANDOM_BATTLE_EVENT_SCHEMA_HASH);
		assert.equal(GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.eventSchemaVersion,
			GEN9_RANDOM_BATTLE_EVENT_SCHEMA_VERSION);
		assert.equal(GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.eventSchemaHash, eventSchemaHash);
		assert.equal(GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.randomBattleDataHash,
			crypto.createHash('sha256').update(JSON.stringify(randomSets)).digest('hex'));
		assert.equal(GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.actionCount, 14);
		assert.equal(GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.actions.length, 14);
		assert.equal(GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.reservedTokens.none, 0);
		assert.equal(GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.reservedTokens.unknown, 1);
		assert(Object.isFrozen(GEN9_RANDOM_BATTLE_TENSOR_MANIFEST));
		assert(Object.isFrozen(GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.vocabularies.species));
	});

	it('should cover every species, move, and ability in the Random Battle set data', () => {
		const vocabularies = GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.vocabularies;
		const species = new Set(vocabularies.species);
		const moves = new Set(vocabularies.moves);
		const abilities = new Set(vocabularies.abilities);
		for (const [speciesId, speciesData] of Object.entries(randomSets)) {
			assert(species.has(Sim.toID(speciesId)), `Missing species token for ${speciesId}`);
			for (const set of speciesData.sets) {
				for (const move of set.movepool) {
					assert(moves.has(Sim.toID(move)), `Missing move token for ${move}`);
				}
				for (const ability of set.abilities) {
					assert(abilities.has(Sim.toID(ability)), `Missing ability token for ${ability}`);
				}
			}
		}
		for (const ability of ['Tera Shift', 'Tera Shell', 'Teraform Zero']) {
			assert(abilities.has(Sim.toID(ability)), `Missing form-change ability token for ${ability}`);
		}
	});

	it('should emit tensors matching the checked-in field contract', () => {
		const encoded = encodeBattleState(battle, 'p1');
		const manifest = GEN9_RANDOM_BATTLE_TENSOR_MANIFEST;

		assert.equal(encoded.schemaVersion, manifest.schemaVersion);
		assert.equal(encoded.schemaHash, manifest.schemaHash);
		assert.deepEqual(encoded.continuous.shape, [246]);
		assert.deepEqual(encoded.categorical.shape, [148]);
		assert.deepEqual(encoded.binary.shape, [364]);
		assert.deepEqual(encoded.actionMask.shape, [14]);
		assert.deepEqual(encoded.continuous.labels, manifest.fields.continuous);
		assert.deepEqual(encoded.categorical.labels, manifest.fields.categorical);
		assert.deepEqual(encoded.binary.labels, manifest.fields.binary);
		assert.deepEqual(encoded.actionMask.labels, manifest.actions);
		assert(encoded.continuous.data instanceof Float32Array);
		assert(encoded.categorical.data instanceof Int32Array);
		assert(encoded.binary.data instanceof Uint8Array);
		assert(encoded.actionMask.data instanceof Uint8Array);
		assert.equal(encoded.requestState, 'move');
		assert.equal(encoded.needsAction, true);
		assert.equal(encoded.result, 'ongoing');
		assert(encoded.entityIds.you[0]);
		assert(encoded.entityIds.you[1]);
		assert(encoded.entityIds.foe[0]);
	});

	it('should expose own private stats without leaking opponent stats', () => {
		const encoded = encodeBattleState(battle, 'p1');
		assert(encoded.continuous.data[indexOf(encoded.continuous, 'you.slot1.stat.def')] > 0);
		assert.equal(encoded.binary.data[indexOf(encoded.binary, 'you.slot1.statsKnown')], 1);
		assert.equal(encoded.continuous.data[indexOf(encoded.continuous, 'foe.slot1.stat.def')], 0);
		assert.equal(encoded.binary.data[indexOf(encoded.binary, 'foe.slot1.statsKnown')], 0);
	});

	it('should distinguish known, unknown, and absent categorical values', () => {
		const encoded = encodeBattleState(battle, 'p1');
		const ownItem = indexOf(encoded.categorical, 'you.slot1.item');
		const foeItem = indexOf(encoded.categorical, 'foe.slot1.item');
		const foeItemKnown = indexOf(encoded.binary, 'foe.slot1.itemKnown');
		const absentSpecies = indexOf(encoded.categorical, 'you.slot3.species');

		assert(encoded.categorical.data[ownItem] > 1);
		assert.equal(encoded.categorical.data[foeItem], 1);
		assert.equal(encoded.binary.data[foeItemKnown], 0);
		assert.equal(encoded.categorical.data[absentSpecies], 0);
	});

	it('should hide exact opponent HP while retaining exact own HP', () => {
		const playerLabels = encodeBattleState(battle, 'p1').continuous.labels;
		const ownHpIndex = playerLabels.indexOf('you.slot1.hp');
		const foeHpIndex = playerLabels.indexOf('foe.slot1.hp');
		const [ownLowHp, ownHighHp] = findSharedPublicHpPair(battle.p1.active[0].maxhp);
		const [foeLowHp, foeHighHp] = findSharedPublicHpPair(battle.p2.active[0].maxhp);

		battle.p1.active[0].sethp(ownLowHp);
		battle.p2.active[0].sethp(foeLowHp);
		const playerLow = encodeBattleState(battle, 'p1');
		const fullLow = encodeOmniscientBattleState(battle, 'p1');

		battle.p1.active[0].sethp(ownHighHp);
		battle.p2.active[0].sethp(foeHighHp);
		const playerHigh = encodeBattleState(battle, 'p1');
		const fullHigh = encodeOmniscientBattleState(battle, 'p1');

		assert.notEqual(playerLow.continuous.data[ownHpIndex], playerHigh.continuous.data[ownHpIndex]);
		assert.equal(playerLow.continuous.data[foeHpIndex], playerHigh.continuous.data[foeHpIndex]);
		assert.notEqual(fullLow.continuous.data[foeHpIndex], fullHigh.continuous.data[foeHpIndex]);
	});

	it('should encode and decode ordinary move, Tera, and switch actions', () => {
		const encoded = encodeBattleState(battle, 'p1');
		assert.deepEqual(encoded.actionMask.labels, GEN9_RANDOM_BATTLE_ACTION_LABELS);
		assert.deepEqual([...encoded.actionMask.data], [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0]);
		assert.equal(decodeGen9RandomBattleAction(battle, 'p1', 0), 'move softboiled');
		assert.equal(decodeGen9RandomBattleAction(battle, 'p1', 4), 'move softboiled terastallize');
		assert.equal(decodeGen9RandomBattleAction(battle, 'p1', 9), 'switch 2');
		assert.throws(() => decodeGen9RandomBattleAction(battle, 'p1', 8));
	});

	it('should map synthetic Struggle and Recharge requests to move slot one', () => {
		for (const moveSlot of battle.p1.active[0].moveSlots) moveSlot.pp = 0;
		battle.makeRequest('move');
		let encoded = encodeBattleState(battle, 'p1');
		assert.equal(encoded.actionMask.data[0], 1);
		assert.equal(decodeGen9RandomBattleAction(battle, 'p1', 0), 'move struggle');

		battle.p1.activeRequest.active[0].moves = [{ move: 'Recharge', id: 'recharge' }];
		encoded = encodeBattleState(battle, 'p1');
		assert.equal(encoded.actionMask.data[0], 1);
		assert.equal(decodeGen9RandomBattleAction(battle, 'p1', 0), 'move recharge');
	});

	it('should represent Revival Blessing as an actionable fainted-slot selection', () => {
		const revivalBattle = common.createBattle({ formatid: 'gen9randombattle' }, [[
			{ species: 'Corviknight', ability: 'Pressure', moves: ['memento'] },
			{ species: 'Pawmot', ability: 'Natural Cure', moves: ['revivalblessing'] },
			{ species: 'Wynaut', ability: 'Shadow Tag', moves: ['splash'] },
		], [
			{ species: 'Goodra', ability: 'Sap Sipper', moves: ['sleeptalk'] },
		]]);
		try {
			revivalBattle.makeChoices('move memento', 'auto');
			revivalBattle.makeChoices('switch 2', '');
			revivalBattle.makeChoices('move revivalblessing', 'auto');
			const encoded = encodeBattleState(revivalBattle, 'p1');
			assert.equal(encoded.requestState, 'revive');
			assert.equal(encoded.needsAction, true);
			assert.equal(encoded.binary.data[indexOf(encoded.binary, 'battle.isRevivalRequest')], 1);
			assert.deepEqual([...encoded.actionMask.data], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]);
			assert.equal(decodeGen9RandomBattleAction(revivalBattle, 'p1', 9), 'switch 2');

			const revivalTracker = new Gen9RandomBattleObservationTracker('p1');
			const tracked = revivalTracker.receive(`|request|${JSON.stringify(revivalBattle.p1.activeRequest)}`);
			assert.equal(tracked.requestState, 'revive');
			assert.deepEqual([...tracked.actionMask.data], [...encoded.actionMask.data]);
			assert.equal(revivalTracker.decodeAction(9), 'switch 2');
		} finally {
			revivalBattle.destroy();
		}
	});

	it('should encode terminal outcomes without requiring an action', () => {
		battle.win('p1');
		const winner = encodeBattleState(battle, 'p1');
		const loser = encodeBattleState(battle, 'p2');
		assert.equal(winner.requestState, 'terminal');
		assert.equal(winner.result, 'win');
		assert.equal(loser.result, 'loss');
		assert.equal(winner.needsAction, false);
		assert.deepEqual([...winner.actionMask.data], Array(14).fill(0));
	});

	it('should reject formats outside the manifest contract', () => {
		const unsupported = common.createBattle({ preview: false }, TEAMS);
		assert.throws(() => encodeBattleState(unsupported, 'p1'), /does not support format/);
		unsupported.destroy();
	});
});

describe('Gen 9 Random Battle protocol observations', () => {
	let battle;
	let tracker;
	let logPosition;

	function flush(request = true) {
		const log = battle.log.slice(logPosition).join('\n');
		logPosition = battle.log.length;
		if (log) tracker.receive(extractChannelMessages(log, [1])[1].join('\n'));
		return request ? tracker.receive(`|request|${JSON.stringify(battle.p1.activeRequest)}`) : null;
	}

	beforeEach(() => {
		battle = common.createBattle({ formatid: 'gen9randombattle' }, TEAMS);
		tracker = new Gen9RandomBattleObservationTracker('p1');
		logPosition = 0;
	});
	afterEach(() => {
		battle.destroy();
		battle = null;
	});

	it('should build the stable tensor contract and decode actions from player protocol', () => {
		const encoded = flush();
		assert(encoded);
		assert.deepEqual(encoded.continuous.shape, [246]);
		assert.deepEqual(encoded.categorical.shape, [148]);
		assert.deepEqual(encoded.binary.shape, [364]);
		assert.deepEqual([...encoded.actionMask.data], [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0]);
		assert.equal(tracker.decodeAction(0), 'move softboiled');
		assert.equal(tracker.decodeAction(4), 'move softboiled terastallize');
		assert.equal(tracker.decodeAction(9), 'switch 2');
		assert.throws(() => tracker.decodeAction(8), /Illegal/);
	});

	it('should retain opponent knowledge after switches', () => {
		flush();
		battle.makeChoices('move protect', 'move bodyslam');
		flush();
		tracker.receive('|-boost|p2a: Snorlax|atk|2');
		battle.makeChoices('move protect', 'switch 2');
		const encoded = flush();

		const snorlaxActive = indexOf(encoded.binary, 'foe.slot1.active');
		const snorlaxMove = indexOf(encoded.categorical, 'foe.slot1.move1.id');
		const snorlaxMoveRevealed = indexOf(encoded.binary, 'foe.slot1.move1.revealed');
		const snorlaxPPKnown = indexOf(encoded.binary, 'foe.slot1.move1.ppKnown');
		const snorlaxPP = indexOf(encoded.continuous, 'foe.slot1.move1.pp');
		const snorlaxAttack = indexOf(encoded.continuous, 'foe.slot1.boost.atk');
		const mewActive = indexOf(encoded.binary, 'foe.slot2.active');
		assert.equal(encoded.binary.data[snorlaxActive], 0);
		assert.equal(encoded.categorical.data[snorlaxMove],
			GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.vocabularies.moves.indexOf('bodyslam'));
		assert.equal(encoded.binary.data[snorlaxMoveRevealed], 1);
		assert.equal(encoded.binary.data[snorlaxPPKnown], 1);
		assert(encoded.continuous.data[snorlaxPP] > 0 && encoded.continuous.data[snorlaxPP] < 1);
		assert.equal(encoded.continuous.data[snorlaxAttack], 0);
		assert.equal(encoded.binary.data[mewActive], 1);
	});

	it('should track public reveals and lower-bound durations without private state', () => {
		flush();
		tracker.receive([
			'|-damage|p2a: Snorlax|90/100|[from] item: Leftovers',
			'|-damage|p1a: Blissey|90/100|[from] ability: Thick Fat|[of] p2a: Snorlax',
			'|-weather|RainDance|[from] move: Rain Dance',
		].join('\n'));
		let encoded = tracker.encode();
		assert(encoded.categorical.data[indexOf(encoded.categorical, 'foe.slot1.ability')] > 1);
		assert(encoded.categorical.data[indexOf(encoded.categorical, 'foe.slot1.item')] > 1);
		assert.equal(encoded.continuous.data[indexOf(encoded.continuous, 'battle.weatherDuration')], 5 / 8);

		tracker.receive('|upkeep\n|upkeep\n|upkeep\n|upkeep\n|upkeep');
		encoded = tracker.encode();
		assert.equal(encoded.continuous.data[indexOf(encoded.continuous, 'battle.weatherDuration')], 3 / 8);
	});

	it('should model Transform using the target state and reset it on switch', () => {
		const initial = flush();
		const publicId = initial.entityIds.foe[0];
		tracker.receive('|move|p2a: Snorlax|Transform|p1a: Blissey');
		tracker.receive('|-transform|p2a: Snorlax|p1a: Blissey|[from] move: Transform');
		tracker.receive('|move|p2a: Snorlax|Soft-Boiled|p2a: Snorlax');
		let encoded = tracker.encode();
		const blissey = GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.vocabularies.species.indexOf('blissey');
		assert.equal(encoded.categorical.data[indexOf(encoded.categorical, 'foe.slot1.species')], blissey);
		assert.equal(encoded.entityIds.foe[0], publicId);
		assert.equal(tracker.getPublicEntityIdForIdent('p2a: Snorlax'), publicId);
		assert.equal(encoded.binary.data[indexOf(encoded.binary, 'foe.slot1.move1.ppKnown')], 1);

		tracker.receive('|switch|p2a: Mew|Mew, L80|100/100');
		tracker.receive('|switch|p2a: Snorlax|Snorlax, L80|100/100');
		encoded = tracker.encode();
		const snorlax = GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.vocabularies.species.indexOf('snorlax');
		assert.equal(encoded.categorical.data[indexOf(encoded.categorical, 'foe.slot1.species')], snorlax);
		const transform = GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.vocabularies.moves.indexOf('transform');
		assert.equal(encoded.categorical.data[indexOf(encoded.categorical, 'foe.slot1.move1.id')], transform);
		assert.equal(encoded.entityIds.foe[0], publicId);
	});

	it('should correct an apparent known switch-in when Illusion is revealed', () => {
		flush();
		tracker.receive('|switch|p2a: Snorlax|Snorlax, L80|100/100');
		tracker.receive('|move|p2a: Snorlax|Knock Off|p1a: Blissey');
		tracker.receive('|replace|p2a: Zoroark|Zoroark, L80|100/100');
		const encoded = tracker.encode();

		assert.equal(encoded.binary.data[indexOf(encoded.binary, 'foe.slot1.active')], 0);
		assert.equal(encoded.binary.data[indexOf(encoded.binary, 'foe.slot2.active')], 1);
		const zoroark = GEN9_RANDOM_BATTLE_TENSOR_MANIFEST.vocabularies.species.indexOf('zoroark');
		assert.equal(encoded.categorical.data[indexOf(encoded.categorical, 'foe.slot2.species')], zoroark);
		assert.equal(encoded.binary.data[indexOf(encoded.binary, 'foe.slot2.move1.revealed')], 1);
		assert(encoded.entityIds.foe[0]);
		assert(encoded.entityIds.foe[1]);
		assert.notEqual(encoded.entityIds.foe[0], encoded.entityIds.foe[1]);
		assert.equal(tracker.getPublicEntityIdForIdent('p2a: Zoroark'), encoded.entityIds.foe[1]);
	});

	it('should emit versioned structured event deltas and fail closed in strict mode', () => {
		flush();
		const update = tracker.receiveUpdate([
			'|move|p2a: Snorlax|Body Slam|p1a: Blissey',
			'|-damage|p1a: Blissey|90/100|[from] move: Body Slam',
		].join('\n'));
		assert.equal(update.eventSchemaVersion, GEN9_RANDOM_BATTLE_EVENT_SCHEMA_VERSION);
		assert.equal(update.eventSchemaHash, GEN9_RANDOM_BATTLE_EVENT_SCHEMA_HASH);
		assert.equal(update.events.length, 2);
		assert.equal(update.events[0].category, 'transient');
		assert.equal(update.events[0].stateChanging, false);
		assert.equal(update.events[0].actor.publicId, update.observation?.entityIds.foe[0] ||
		tracker.getPublicEntityIdForIdent('p2a: Snorlax'));
		assert.equal(update.events[1].category, 'state');
		assert.equal(update.events[1].stateChanging, true);
		assert.deepEqual(update.events[1].annotations, [{ key: 'from', value: 'move: Body Slam' }]);

		const strict = new Gen9RandomBattleObservationTracker('p1', { strictEvents: true });
		strict.receive(extractChannelMessages(battle.log.join('\n'), [1])[1].join('\n'));
		strict.receive(`|request|${JSON.stringify(battle.p1.activeRequest)}`);
		assert.throws(() => strict.receive('|-future-state-change|p1a: Blissey|value'), /unclassified/);
	});

	it('should audit an auto-play battle without unclassified state changes or deadlocks', () => {
		const auditBattle = common.createBattle({ formatid: 'gen9randombattle' }, [[
			{ species: 'Blissey', ability: 'Natural Cure', moves: ['seismictoss'] },
			{ species: 'Mew', ability: 'Synchronize', moves: ['psychic'] },
		], [
			{ species: 'Snorlax', ability: 'Thick Fat', moves: ['bodyslam'] },
			{ species: 'Chansey', ability: 'Natural Cure', moves: ['seismictoss'] },
		]]);
		const trackers = {
			p1: new Gen9RandomBattleObservationTracker('p1', { strictEvents: true }),
			p2: new Gen9RandomBattleObservationTracker('p2', { strictEvents: true }),
		};
		let position = 0;
		const audit = () => {
			const log = auditBattle.log.slice(position).join('\n');
			position = auditBattle.log.length;
			const messages = extractChannelMessages(log, [1, 2]);
			if (messages[1].length) trackers.p1.receiveUpdate(messages[1].join('\n'));
			if (messages[2].length) trackers.p2.receiveUpdate(messages[2].join('\n'));
			for (const side of ['p1', 'p2']) {
				const request = auditBattle[side].activeRequest;
				if (request) trackers[side].receiveUpdate(`|request|${JSON.stringify(request)}`);
			}
		};
		try {
			audit();
			for (let i = 0; i < 80 && !auditBattle.ended; i++) {
				const choices = ['p1', 'p2'].map(side => auditBattle[side].activeRequest?.wait ? '' : 'auto');
				auditBattle.makeChoices(...choices);
				audit();
			}
			assert(auditBattle.turn > 1);
			for (const side of ['p1', 'p2']) {
				const encoded = trackers[side].encode();
				assert(!encoded.needsAction || encoded.actionMask.data.some(Boolean));
			}
		} finally {
			auditBattle.destroy();
		}
	});

	it('should distinguish retry and terminal observation updates', () => {
		flush();
		const retry = JSON.parse(JSON.stringify(battle.p1.activeRequest));
		retry.update = true;
		let update = tracker.receiveUpdate(`|request|${JSON.stringify(retry)}`);
		assert.equal(update.requestState, 'retry');
		assert.equal(update.needsAction, true);
		assert(update.observation.actionMask.data.some(Boolean));

		update = tracker.receiveUpdate('|player|p1|Alice\n|player|p2|Bob\n|win|Alice');
		assert.equal(update.terminal, true);
		assert.equal(update.result, 'win');
		assert.equal(update.observation.requestState, 'terminal');
		assert.equal(update.observation.needsAction, false);
		assert.deepEqual([...update.observation.actionMask.data], Array(14).fill(0));
	});

	it('should reject unsupported metadata and encoding before initialization', () => {
		assert.throws(() => tracker.encode(), /before receiving a choice request/);
		assert.throws(() => tracker.receive('|gen|8'), /unsupported generation/);
		assert.throws(() => tracker.receive('|gametype|doubles'), /unsupported game type/);
	});
});
