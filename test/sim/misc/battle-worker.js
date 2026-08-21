'use strict';

const { EventEmitter } = require('node:events');
const { spawn } = require('node:child_process');
const { execFileSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { PassThrough } = require('node:stream');

const assert = require('./../../assert');
const { BattleWorker } = require('./../../../dist/sim/battle-worker');
const {
	resolveCleanSimulatorCommit,
	runBattleWorkerProcess,
} = require('./../../../dist/sim/battle-worker-process');
const {
	BATTLE_WORKER_DEFAULT_MAX_FRAME_SIZE,
	BattleWorkerProtocolError,
	MessagePackFrameDecoder,
	MessagePackFrameWriter,
	encodeMessagePackFrame,
} = require('./../../../dist/sim/battle-worker-protocol');

const BATTLE_SEED = '1,2,3,4';
const TEAM_SEEDS = { p1: '5,6,7,8', p2: '9,10,11,12' };
const TEST_COMMIT = '0123456789abcdef0123456789abcdef01234567';

function createWorker(options = {}) {
	const messages = [];
	const worker = new BattleWorker(message => messages.push(message), options);
	return { worker, messages };
}

async function waitFor(messages, predicate, description) {
	const deadline = Date.now() + 3000;
	while (Date.now() < deadline) {
		const result = predicate(messages);
		if (result) return result;
		await new Promise(resolve => {
			setTimeout(resolve, 5);
		});
	}
	assert.fail(`Timed out waiting for ${description}`);
}

function decisionsFor(messages, battleId) {
	return messages.filter(message => message.type === 'decision' && message.battleId === battleId);
}

function firstLegalAction(decision) {
	return [...decision.playerObservation.actionMask.data].findIndex(Boolean);
}

describe('Pokezero battle worker framing', () => {
	it('should use fragmented uint32be MessagePack frames and canonical little-endian tensor buffers', () => {
		const message = {
			type: 'tensor-test',
			tensor: {
				dtype: 'int32', shape: [2], labels: ['a', 'b'],
				data: new Int32Array([0x01020304, -2]),
			},
		};
		const frame = encodeMessagePackFrame(message);
		assert.equal(frame.readUInt32BE(0), frame.length - 4);

		const decoder = new MessagePackFrameDecoder();
		assert.deepEqual(decoder.push(frame.subarray(0, 3)), []);
		assert.deepEqual(decoder.push(frame.subarray(3, 9)), []);
		const decoded = decoder.push(frame.subarray(9));
		assert.equal(decoded.length, 1);
		assert(Buffer.isBuffer(decoded[0].tensor.data));
		assert.deepEqual([...decoded[0].tensor.data.subarray(0, 4)], [4, 3, 2, 1]);
		assert.deepEqual(decoded[0].tensor.shape, [2]);
		assert.deepEqual(Object.keys(decoded[0].tensor).sort(), ['data', 'dtype', 'shape']);
		decoder.finish();
	});

	it('should reject oversized, truncated, and inconsistent tensor frames', () => {
		const decoder = new MessagePackFrameDecoder(8);
		const oversized = Buffer.alloc(4);
		oversized.writeUInt32BE(9);
		assert.throws(() => decoder.push(oversized), BattleWorkerProtocolError);

		const truncated = new MessagePackFrameDecoder();
		truncated.push(Buffer.from([0, 0, 0, 2, 0x80]));
		assert.throws(() => truncated.finish(), /incomplete frame/);

		assert.throws(() => encodeMessagePackFrame({
			dtype: 'float32', shape: [2], data: new Float32Array(1),
		}), /expected 8/);
		assert.equal(BATTLE_WORKER_DEFAULT_MAX_FRAME_SIZE, 64 * 1024 * 1024);
	});

	it('should bound output queued behind a blocked writable', async () => {
		class BlockedOutput extends EventEmitter {
			write() { return false; }
		}
		const output = new BlockedOutput();
		const writer = new MessagePackFrameWriter(output, { maxQueuedFrames: 1, maxQueuedBytes: 1024 });
		writer.write({ type: 'first' });
		assert.throws(() => writer.write({ type: 'second' }), /Output queue limit exceeded/);
		output.emit('drain');
		await writer.idle();
	});

	it('should run as a long-lived framed in-process transport with an injected test identity', async () => {
		const input = new PassThrough();
		const output = new PassThrough();
		const messages = [];
		const decoder = new MessagePackFrameDecoder();
		output.on('data', chunk => messages.push(...decoder.push(chunk)));
		const { worker, writer } = runBattleWorkerProcess(input, output, { simulatorCommit: TEST_COMMIT });
		try {
			await waitFor(messages, current => current.length >= 1, 'worker process hello');
			assert.equal(messages[0].type, 'hello');
			assert.equal(messages[0].simulatorCommit, TEST_COMMIT);
			input.write(encodeMessagePackFrame({ type: 'hello' }));
			await waitFor(messages, current => current.length >= 2, 'worker process hello response');
			assert.equal(messages[1].protocolVersion, messages[0].protocolVersion);
			input.end();
			await writer.idle();
			decoder.finish();
		} finally {
			worker.close();
		}
	});

	it('should derive a clean Git HEAD and reject dirty or non-Git roots', () => {
		const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'ps-worker-identity-'));
		const nonGitRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'ps-worker-no-git-'));
		try {
			fs.writeFileSync(path.join(tempRoot, 'tracked.txt'), 'clean\n');
			execFileSync('git', ['init', '--quiet'], { cwd: tempRoot });
			execFileSync('git', ['add', 'tracked.txt'], { cwd: tempRoot });
			execFileSync('git', [
				'-c', 'user.name=Pokezero Test', '-c', 'user.email=pokezero@example.invalid',
				'commit', '--quiet', '-m', 'identity test',
			], { cwd: tempRoot });
			const expected = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: tempRoot, encoding: 'utf8' }).trim();
			assert.equal(resolveCleanSimulatorCommit(tempRoot, tempRoot), expected);

			fs.writeFileSync(path.join(tempRoot, 'untracked.txt'), 'dirty\n');
			assert.throws(() => resolveCleanSimulatorCommit(tempRoot, tempRoot), /dirty/);
			assert.throws(() => resolveCleanSimulatorCommit(nonGitRoot, nonGitRoot), /clean Git checkout/);
		} finally {
			fs.rmSync(tempRoot, { recursive: true, force: true });
			fs.rmSync(nonGitRoot, { recursive: true, force: true });
		}
	});

	it('should keep executable stdout empty when production identity validation fails', async () => {
		const repoRoot = path.resolve(__dirname, '../../..');
		const nonGitCwd = fs.mkdtempSync(path.join(os.tmpdir(), 'ps-worker-executable-'));
		const child = spawn(process.execPath, [path.join(repoRoot, 'dist/sim/battle-worker-process.js')], {
			cwd: nonGitCwd,
			stdio: ['ignore', 'pipe', 'pipe'],
		});
		const stdout = [];
		const stderr = [];
		child.stdout.on('data', chunk => stdout.push(chunk));
		child.stderr.on('data', chunk => stderr.push(chunk));
		try {
			const exit = await new Promise(resolve => {
				child.once('exit', (code, signal) => resolve({ code, signal }));
			});
			assert.deepEqual(exit, { code: 1, signal: null });
			assert.equal(Buffer.concat(stdout).length, 0);
			assert(Buffer.concat(stderr).toString().includes('refused startup'));
		} finally {
			if (child.exitCode === null && child.signalCode === null) child.kill();
			fs.rmSync(nonGitCwd, { recursive: true, force: true });
		}
	});
});

describe('Pokezero battle worker', () => {
	it('should not trust an environment-provided simulator identity', () => {
		const previous = process.env.POKEZERO_SIMULATOR_COMMIT;
		process.env.POKEZERO_SIMULATOR_COMMIT = TEST_COMMIT;
		try {
			const { worker, messages } = createWorker();
			assert.equal(messages[0].simulatorCommit, null);
			worker.close();
		} finally {
			if (previous === undefined) {
				delete process.env.POKEZERO_SIMULATOR_COMMIT;
			} else {
				process.env.POKEZERO_SIMULATOR_COMMIT = previous;
			}
		}
	});

	it('should multiplex deterministic battles and keep privileged labels separate and identity-aligned', async () => {
		const { worker, messages } = createWorker({ maxBattles: 4, simulatorCommit: TEST_COMMIT });
		try {
			for (const battleId of ['deterministic-a', 'deterministic-b']) {
				worker.receive({
					type: 'start', battleId, battleSeed: BATTLE_SEED, teamSeeds: TEAM_SEEDS,
					trainingTargets: true,
				});
			}
			await waitFor(
				messages,
				current => decisionsFor(current, 'deterministic-a').length === 2 &&
					decisionsFor(current, 'deterministic-b').length === 2,
				'initial decisions for both battles'
			);
			assert.equal(worker.battleCount, 2);
			const hello = messages[0];
			assert.equal(hello.type, 'hello');
			assert.equal(hello.frame.length, 'uint32be');
			assert.equal(hello.frame.tensorByteOrder, 'little');
			assert(/^[0-9a-f]{64}$/.test(hello.tensorSchemaHash));
			assert(/^[0-9a-f]{64}$/.test(hello.eventSchemaHash));
			assert(/^[0-9a-f]{64}$/.test(hello.randomBattleDataHash));

			const first = decisionsFor(messages, 'deterministic-a')[0];
			const second = decisionsFor(messages, 'deterministic-b')[0];
			assert.deepEqual(
				[...first.playerObservation.continuous.data],
				[...second.playerObservation.continuous.data]
			);
			assert.deepEqual(
				[...first.playerObservation.categorical.data],
				[...second.playerObservation.categorical.data]
			);
			assert(!('privilegedTargets' in first.playerObservation));
			assert.equal(first.privilegedTargets.opponent.length, 6);
			const publicIds = new Set(first.playerObservation.entityIds.foe.filter(Boolean));
			for (const target of first.privilegedTargets.opponent) {
				if (target.publicEntityId) {
					assert.equal(target.targetId, target.publicEntityId);
					assert(publicIds.has(target.targetId));
					assert.equal(target.targetIdKind, 'public');
				} else {
					assert(/^learner:p2:team:[1-6]$/.test(target.targetId));
					assert.equal(target.targetIdKind, 'learner');
					assert(!publicIds.has(target.targetId));
				}
				assert.equal(target.initial.moves.length, target.publicKnowledge.initial.moves.length);
				assert.equal(target.current.moves.length, target.publicKnowledge.current.pp.length);
			}
		} finally {
			worker.close();
		}
	});

	it('should emit both joint decisions before accepting either action and reject stale/illegal actions', async () => {
		const { worker, messages } = createWorker();
		try {
			worker.receive({
				type: 'start', battleId: 'joint', battleSeed: BATTLE_SEED, teamSeeds: TEAM_SEEDS,
			});
			const initial = await waitFor(
				messages,
				current => decisionsFor(current, 'joint').length === 2 && decisionsFor(current, 'joint'),
				'paired initial decisions'
			);
			assert.equal(initial[0].jointDecisionGroup, initial[1].jointDecisionGroup);
			const lastDecisionIndex = Math.max(...initial.map(decision => messages.indexOf(decision)));
			assert(messages.findIndex(message => message.type === 'action') === -1);

			worker.receive({
				type: 'action', battleId: 'joint', side: initial[0].side,
				requestId: initial[0].requestId, actionIndex: firstLegalAction(initial[0]),
			});
			assert(!messages.some(message => message.type === 'action' && message.status === 'submitted'));
			assert(messages.indexOf(initial[0]) <= lastDecisionIndex && messages.indexOf(initial[1]) <= lastDecisionIndex);

			worker.receive({
				type: 'action', battleId: 'joint', side: initial[1].side,
				requestId: initial[1].requestId + 100, actionIndex: firstLegalAction(initial[1]),
			});
			assert(messages.some(message => message.type === 'action' && message.status === 'rejected' &&
				message.reason.includes('Stale')));

			worker.receive({
				type: 'action', battleId: 'joint', side: initial[1].side,
				requestId: initial[1].requestId, actionIndex: firstLegalAction(initial[1]),
			});
			await waitFor(
				messages,
				current => current.filter(message => message.type === 'action' && message.status === 'submitted').length === 2,
				'joint action submission'
			);
			const submitted = messages.filter(message => message.type === 'action' && message.status === 'submitted');
			assert.deepEqual(submitted.map(message => message.side), ['p1', 'p2']);

			const next = await waitFor(
				messages,
				current => decisionsFor(current, 'joint').length >= 4 && decisionsFor(current, 'joint').slice(-2),
				'next paired decisions'
			);
			const maskedAction = [...next[0].playerObservation.actionMask.data].findIndex(value => !value);
			assert.notEqual(maskedAction, -1);
			worker.receive({
				type: 'action', battleId: 'joint', side: next[0].side,
				requestId: next[0].requestId, actionIndex: maskedAction,
			});
			await waitFor(
				messages,
				current => decisionsFor(current, 'joint').some(decision => decision.retry?.actionIndex === maskedAction),
				'illegal-action retry'
			);
			const retry = decisionsFor(messages, 'joint').find(decision => decision.retry?.actionIndex === maskedAction);
			assert(retry.requestId > next[0].requestId);
			assert(retry.events.some(event => event.command === 'choice-rejected' && event.category === 'transient'));
			const sideEvents = messages.filter(message => message.battleId === 'joint' && message.side === retry.side)
				.flatMap(message => message.events || []);
			for (let i = 1; i < sideEvents.length; i++) {
				assert(sideEvents[i].sequence > sideEvents[i - 1].sequence, `Event sequence must be strictly increasing`);
			}
			for (const event of sideEvents) {
				assert.equal(typeof event.schemaVersion, 'string');
				assert(/^[0-9a-f]{64}$/.test(event.schemaHash));
				assert(['state', 'transient', 'cosmetic'].includes(event.category));
			}
		} finally {
			worker.close();
		}
	});

	it('should finish a short battle with replay metadata and isolate invalid starts', async () => {
		const { worker, messages } = createWorker({ maxBattles: 2 });
		const teams = {
			p1: [{
				species: 'Magikarp', ability: 'Swift Swim', item: '', teraType: 'Water',
				moves: ['splash'], nature: 'Hardy', level: 1,
			}],
			p2: [{
				species: 'Mewtwo', ability: 'Pressure', item: 'Life Orb', teraType: 'Psychic',
				moves: ['psystrike'], nature: 'Modest', level: 100,
			}],
		};
		try {
			worker.receive({ type: 'start', battleId: 'bad', battleSeed: 'not-a-seed', teamSeeds: TEAM_SEEDS });
			assert(messages.some(message => message.type === 'error' && message.code === 'START_FAILED'));

			worker.receive({
				type: 'start', battleId: 'terminal', battleSeed: BATTLE_SEED, teams,
				includeInputLog: true,
			});
			const decisions = await waitFor(
				messages,
				current => decisionsFor(current, 'terminal').length === 2 && decisionsFor(current, 'terminal'),
				'short-battle decisions'
			);
			for (const decision of decisions) {
				worker.receive({
					type: 'action', battleId: 'terminal', side: decision.side,
					requestId: decision.requestId, actionIndex: firstLegalAction(decision),
				});
			}
			const terminal = await waitFor(
				messages,
				current => current.find(message => message.type === 'terminal' && message.battleId === 'terminal'),
				'terminal result'
			);
			assert.equal(terminal.winner, 'p2');
			assert.equal(terminal.tie, false);
			assert(/^[0-9a-f]{64}$/.test(terminal.inputLogDigest));
			assert(Array.isArray(terminal.inputLog));
			assert.equal(worker.battleCount, 0);
		} finally {
			worker.close();
		}
	});
});
