# Gen 9 Random Battle tensors

`battle-tensors.ts` encodes a `[Gen 9] Random Battle` information state from one
side's perspective. Contract v2 returns four one-dimensional tensors plus
decision/result and public-entity metadata:

- `continuous`: normalized numeric features such as HP, PP, boosts, and durations.
- `categorical`: stable integer tokens intended for embedding layers.
- `binary`: presence, visibility, knowledge, and boolean state flags.
- `actionMask`: legal policy actions in the order specified by the manifest.

`requestState`, `needsAction`, and `result` distinguish ordinary moves, forced
switches, Revival Blessing target selection, private retry requests, waits, and
terminal win/loss/tie states. An actionable request is guaranteed to have at
least one legal action; an all-zero mask is valid only when `needsAction` is
false. The policy action space remains 14 actions (four moves, four Tera moves,
and six team-slot selections). Revival Blessing uses those same team-slot
actions but makes only fainted targets legal.

The checked-in contract is `data/random-battles/gen9/tensor-manifest.json`. It
contains every tensor label in order, all categorical vocabularies, normalization
constants, supported formats, action labels, and SHA-256 hashes. It exposes
`tensorSchemaHash`, `eventSchemaHash`, and `randomBattleDataHash`; checkpoints,
workers, and replay data must fail closed on a mismatch. Token `0` means
not applicable or absent. Token `1` means that a value exists but is unknown to
the observing player. Real vocabulary entries start at token `2`.

Own slots include the private non-HP stats supplied in the player's request.
Opponent move PP is retained only when it can be derived from the player's
protocol history, with `move*.ppKnown` separating known values from placeholders.

The encoder reads the public HP/details representation for the opponent and does
not expose hidden item, ability, Tera type, or moves in player mode. The binary
`*Known` and `move*.revealed` fields distinguish unknown values from known empty
values. `encodeOmniscientBattleState` is only for simulator diagnostics and
determinized search; it must not be used as a player's policy/value observation.

Player mode's direct `Battle` encoder is a privacy-safe snapshot. Inactive
opponent details that cannot be reconstructed safely from the live simulator
object are marked unknown, even if they were revealed earlier.

For a complete player information state, construct a
`Gen9RandomBattleObservationTracker` for `p1` or `p2` and pass it every chunk
from the matching stream returned by `getPlayerStreams`. `receive` returns an
observation when the chunk contains a new choice request; `encode` returns the
latest observation at any later point. The tracker only consumes public battle
messages and that player's private request, retaining revealed opponent facts
across switches without reading the simulator's hidden state. Its
`decodeAction` method maps a legal policy index back to a simulator command.

Every encoded state also contains `entityIds.you` and `entityIds.foe`. These are
opaque, perspective-local IDs for aligning recurrent events and revealed belief
targets; they do not encode hidden team position. Unrevealed opponent reserves
remain `null`. Illusion aliases are corrected when the public `replace` event is
received. `getPublicEntityIdForIdent` lets simulator bridges correlate the
currently visible protocol ident, but callers must rebind privileged truth after
an Illusion reveal.

## Structured event deltas

`Gen9RandomBattleObservationTracker.receiveUpdate` consumes the same player stream
as `receive` and returns the optional observation together with every structured
event in the chunk. `receive` remains as a compatibility wrapper returning only
the observation. Events carry a sequence number, command/category, raw player-safe
arguments, parsed annotations, stable actor/target IDs where known, and whether
the command changes tracked state. Event chunks are emitted even when the side is
waiting and no policy decision is required.

Construct the tracker with `{ strictEvents: true }` for audits. Strict mode rejects
any protocol command not classified by the event manifest. Cosmetic and metadata
commands are explicitly allowlisted, while transient move outcomes (failure,
immunity, critical hits, activations, and similar events) remain available to a
recurrent model without expanding the fixed snapshot for every volatile effect.
Generic `[from] item:` and `[from] ability:` annotations update public reveal
knowledge using `[of]` when present.

## Updating the schema

Build the simulator, update `SCHEMA_VERSION` (and `EVENT_SCHEMA_VERSION` when the
event wire contract changes) in the generator, and regenerate the manifest:

```sh
node build
node tools/generate-gen9-randombattle-tensor-manifest.js
```

Any change to a vocabulary, field order, normalization constant, classified event
set, or action label changes the corresponding hash. Model checkpoints and replay
data must store the schema versions and all three compatibility hashes.

The generator includes all current Random Battle set species and their formes,
all set moves, all Gen 9/Past standard abilities and items. The broader ability
vocabulary is required because form changes and ability-copying effects can change
public current ability. New Random Battle set data intentionally fails the vocabulary
coverage test until the manifest is regenerated under a new schema version.
