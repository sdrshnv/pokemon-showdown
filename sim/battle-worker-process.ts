/**
 * stdin/stdout process entrypoint for the Pokezero battle worker.
 *
 * @license MIT
 */

import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, realpathSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { BattleWorker } from './battle-worker';
import {
	BATTLE_WORKER_DEFAULT_MAX_FRAME_SIZE,
	MessagePackFrameDecoder,
	MessagePackFrameWriter,
	type BattleWorkerMessage,
} from './battle-worker-protocol';

export interface BattleWorkerProcessOptions {
	maxBattles?: number;
	maxFrameSize?: number;
	maxPendingMessages?: number;
	maxQueuedOutputBytes?: number;
	maxQueuedOutputFrames?: number;
	/** In-process test injection only. The executable entrypoint never supplies this. */
	simulatorCommit?: string;
}

export function resolveCleanSimulatorCommit(
	cwd = process.cwd(), showdownRoot = findPokemonShowdownRoot(__dirname)
) {
	const realCwd = realpath(cwd, 'worker cwd');
	const realShowdownRoot = realpath(showdownRoot, 'Pokémon Showdown root');
	if (realCwd !== realShowdownRoot) {
		throw new Error(`Worker must be started from the Pokémon Showdown root ${realShowdownRoot}`);
	}

	const gitRoot = runGit(['rev-parse', '--show-toplevel'], realCwd, 'resolve the Git worktree root');
	if (realpath(gitRoot, 'Git worktree root') !== realShowdownRoot) {
		throw new Error(`Worker cwd is not the Pokémon Showdown Git worktree root`);
	}
	const commit = runGit(['rev-parse', '--verify', 'HEAD'], realCwd, 'resolve Git HEAD');
	if (!/^(?:[0-9a-f]{40}|[0-9a-f]{64})$/.test(commit)) {
		throw new Error(`Git returned an invalid HEAD commit ${JSON.stringify(commit)}`);
	}
	const status = runGit(
		['status', '--porcelain=v1', '--untracked-files=all', '--ignore-submodules=none'],
		realCwd,
		'inspect the Git worktree'
	);
	if (status) throw new Error(`Refusing to start from a dirty Pokémon Showdown Git worktree`);
	return commit;
}

export function runBattleWorkerProcess(
	input: NodeJS.ReadableStream = process.stdin,
	output: NodeJS.WritableStream = process.stdout,
	options: BattleWorkerProcessOptions = {},
) {
	const maxFrameSize = options.maxFrameSize ?? BATTLE_WORKER_DEFAULT_MAX_FRAME_SIZE;
	const maxPendingMessages = options.maxPendingMessages ?? 1024;
	if (!Number.isInteger(maxPendingMessages) || maxPendingMessages < 1) {
		throw new RangeError(`Invalid maximum pending message count ${maxPendingMessages}`);
	}
	const simulatorCommit = options.simulatorCommit ?? resolveCleanSimulatorCommit();
	if (!/^(?:[0-9a-f]{40}|[0-9a-f]{64})$/.test(simulatorCommit)) {
		throw new Error(`Invalid injected simulator commit ${JSON.stringify(simulatorCommit)}`);
	}
	const decoder = new MessagePackFrameDecoder(maxFrameSize);
	const writer = new MessagePackFrameWriter(output, {
		maxFrameSize,
		maxQueuedBytes: options.maxQueuedOutputBytes,
		maxQueuedFrames: options.maxQueuedOutputFrames,
	});
	const queue: unknown[] = [];
	let draining = false;
	let ended = false;
	let failed = false;
	const worker = new BattleWorker(message => writer.write(message), {
		maxBattles: options.maxBattles,
		maxFrameSize,
		simulatorCommit,
	});

	const emitProcessError = (code: string, error: any) => {
		if (failed) return;
		failed = true;
		const message: BattleWorkerMessage = {
			type: 'error',
			code,
			message: error?.message || `${error}`,
			fatal: true,
		};
		try {
			writer.write(message);
		} catch (writeError: any) {
			process.stderr.write(`Pokezero worker failed: ${writeError?.message || writeError}\n`);
		}
		input.pause?.();
		worker.close();
		process.exitCode = 1;
	};

	const finishIfDone = async () => {
		if (!ended || draining || queue.length) return;
		worker.close();
		try {
			await writer.idle();
		} catch (error: any) {
			process.stderr.write(`Pokezero worker output failed: ${error?.message || error}\n`);
			process.exitCode = 1;
		}
	};

	const drain = async () => {
		if (draining || failed) return;
		draining = true;
		try {
			while (queue.length) {
				worker.receive(queue.shift());
				if (failed) break;
				if (queue.length < Math.ceil(maxPendingMessages / 2)) input.resume?.();
				await Promise.resolve();
			}
		} catch (error: any) {
			emitProcessError('WORKER_PROCESS_FAILED', error);
		} finally {
			draining = false;
			void finishIfDone();
		}
	};

	input.on('data', (chunk: Buffer | Uint8Array | string) => {
		if (failed) return;
		try {
			const bytes = typeof chunk === 'string' ? Buffer.from(chunk) : chunk;
			const messages = decoder.push(bytes);
			if (queue.length + messages.length > maxPendingMessages) {
				throw new Error(`Input queue exceeds ${maxPendingMessages} messages`);
			}
			queue.push(...messages);
			if (queue.length >= maxPendingMessages) input.pause?.();
			void drain();
		} catch (error: any) {
			emitProcessError(error?.code || 'FRAME_DECODE_FAILED', error);
		}
	});
	input.on('end', () => {
		ended = true;
		try {
			decoder.finish();
		} catch (error: any) {
			emitProcessError(error?.code || 'FRAME_DECODE_FAILED', error);
		}
		void finishIfDone();
	});
	input.on('error', error => emitProcessError('INPUT_FAILED', error));
	output.on('error', error => emitProcessError('OUTPUT_FAILED', error));

	return { worker, writer };
}

function findPokemonShowdownRoot(start: string) {
	let current = resolve(start);
	while (true) {
		const packagePath = join(current, 'package.json');
		if (existsSync(packagePath)) {
			try {
				if (JSON.parse(readFileSync(packagePath, 'utf8')).name === 'pokemon-showdown') return current;
			} catch {}
		}
		const parent = dirname(current);
		if (parent === current) throw new Error(`Could not locate the Pokémon Showdown package root`);
		current = parent;
	}
}

function realpath(path: string, description: string) {
	try {
		return realpathSync(path);
	} catch (error: any) {
		throw new Error(`Could not resolve ${description}: ${error?.message || error}`);
	}
}

function runGit(args: string[], cwd: string, action: string) {
	try {
		return execFileSync('git', args, {
			cwd,
			encoding: 'utf8',
			stdio: ['ignore', 'pipe', 'pipe'],
		}).trim();
	} catch {
		throw new Error(`Could not ${action}; worker requires a clean Git checkout`);
	}
}

if (require.main === module) {
	try {
		runBattleWorkerProcess();
	} catch (error: any) {
		process.stderr.write(`Pokezero battle worker refused startup: ${error?.message || error}\n`);
		process.exitCode = 1;
	}
}
