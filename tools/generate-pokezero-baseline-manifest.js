'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const { Dex, toID } = require('../dist/sim/dex');
const tensorManifest = require('../data/random-battles/gen9/tensor-manifest.json');

const SCHEMA_VERSION = 'pokezero-gen9-baselines-v1';
const dex = Dex.forFormat('gen9randombattle');
const typeTokens = new Map(tensorManifest.vocabularies.types.map((name, index) => [toID(name), index]));

const moves = tensorManifest.vocabularies.moves.map((id, token) => {
	const move = dex.moves.get(id);
	if (token < 2 || !move.exists) {
		return { token, id, basePower: 0, typeToken: 1, category: 'Status' };
	}
	return {
		token,
		id: move.id,
		basePower: Math.max(0, move.basePower || 0),
		typeToken: typeTokens.get(toID(move.type)) ?? 1,
		category: move.category,
	};
});

const typeEffectiveness = tensorManifest.vocabularies.types.map((attacking, attackToken) => (
	tensorManifest.vocabularies.types.map((defending, defenseToken) => {
		if (attackToken < 2 || defenseToken < 2) return 1;
		if (!dex.getImmunity(attacking, defending)) return 0;
		return 2 ** dex.getEffectiveness(attacking, defending);
	})
));

const core = {
	schemaVersion: SCHEMA_VERSION,
	tensorSchemaHash: tensorManifest.schemaHash,
	moves,
	typeEffectiveness,
};
const dataHash = crypto.createHash('sha256').update(JSON.stringify(core)).digest('hex');
const output = path.resolve(__dirname, '../data/random-battles/gen9/pokezero-baselines.json');
fs.writeFileSync(output, `${JSON.stringify({ ...core, dataHash }, null, 4)}\n`);
console.log(`Wrote ${path.relative(process.cwd(), output)} (${dataHash})`);
