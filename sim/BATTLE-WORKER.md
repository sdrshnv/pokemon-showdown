# Pokezero battle worker

Build the simulator before starting the worker:

```sh
npm run build
npm run battle-worker
```

`battle-worker` writes only framed protocol messages to stdout. Each message is
standard MessagePack prefixed by an unsigned 32-bit big-endian payload length.
Diagnostics and startup failures go to stderr.

## Simulator identity

The executable must be started with the Pokémon Showdown repository root as its
current working directory. Before emitting its `hello` frame, it resolves Git
`HEAD`, verifies that the current directory is the repository root, and checks
tracked, untracked, and submodule state. It refuses to start from a dirty tree,
a non-Git directory, or a repository without a valid commit.

The resulting commit hash is reported as `hello.simulatorCommit`. Environment
variables cannot override it. The programmatic APIs retain an explicit commit
option solely so unit tests and other in-process harnesses can supply a fixture
identity without spawning a production executable.
