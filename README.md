# cypress

`cypress` is a local semantic navigation and knowledge index for project workspaces.

It keeps the fast terminal jump flow (`cypress pick --query "<query>"`) and supports:
- semantic + lexical + frecency + graph ranking
- an interactive picker for fast selection
- local API + MCP server for agent integrations

## Quick Start

```bash
cd /path/to/repo

# Initialize project-local Cypress state and runtime provisioning
cypress install

# Track and index your workspace roots
cypress add ~/c-projects
cypress index

# Search (non-interactive)
cypress search adobe efs

# Interactive picker (prints a selected path)
cypress pick --query "adobe efs"

# To cd (a subprocess cannot change the parent shell's cwd)
cd "$(cypress pick --query 'adobe efs')"
```

## Publish via GitHub API

If you need API-based sync (instead of `git push`), use:

```bash
# preview
scripts/publish-gh-api.sh --owner stouffer-labs --repo cypress --dry-run

# sync tracked files
scripts/publish-gh-api.sh --owner stouffer-labs --repo cypress
```

## Rust CLI Status

- `./cypress` now prefers a native Rust binary if present:
  - `target/release/cypress`
  - `target/debug/cypress`
  - no non-native fallback in normal usage
- Native Rust commands implemented now:
  - `cypress config` (interactive config editor + set/show helpers)
  - `cypress autotune` (history-driven ranking auto-tuning)
  - `cypress init`
  - `cypress install`
  - `cypress add`
  - `cypress del`
  - `cypress roots`
  - `cypress index` (native for `embed_backend=local-hash|ollama|local(alias)`)
  - `cypress refresh` (native for `embed_backend=local-hash|ollama|local(alias)`)
  - `cypress search` (native for `embed_backend=local-hash|ollama|local(alias)`)
  - `cypress watch` (native for `embed_backend=local-hash|ollama|local(alias)`)
  - `cypress pick` (native for `embed_backend=local-hash|ollama|local(alias)`)
  - `cypress daemon` (native API daemon lifecycle: start/stop/restart/status/logs)
  - `cypress bench` (native command surface; FalkorDB benchmark path implemented)
  - `cypress graph view` (graph web server)
  - `cypress graph open` (start/open graph web UI)
  - `cypress graph neighbors` (terminal neighbor table)
  - `cypress graph lineage` (terminal lineage table)
  - `cypress doctor`
  - `cypress version`
- `cypress api` and `cypress mcp` are native Rust.
- Terminal graph renderers were intentionally removed (`cypress graph text|ui`); use web viewer + table commands.
- `embed_backend=local` is treated as a native compatibility alias for `local-hash`.

Build Rust binary:

```bash
cargo build --release -p cypress
```

Optional shell integration: `source scripts/cypress-shell.sh` enables interactive `cd` navigation via `cypress <query>`.
Note: daemon-backed rich picker parsing uses `jq`; if `jq` is unavailable it falls back to direct CLI picker mode.
Shell daemon controls now delegate to native `cypress daemon ...`.
Daemon picker keybinds: `f` switch to files, `p` switch to projects, `Enter`/`c` cd target dir, `o` open selected file (file rows).
Daemon picker now live re-queries while typing (debounced) and keeps file previews synced to the current query.

## Picker UX

Inside `cypress pick`:
- `Enter`: print the selected path to stdout
- Use command substitution to `cd`, for example: `cd "$(cypress pick --query 'adobe efs')"`

## CLI Commands

```bash
cypress init [--root <path>]          # initialize config/db and embedding backend
cypress install [--no-system-install] [--no-download] [--no-shell-hook] [--venv <path>] [--bench]
cypress config [edit|show|set <key> <value>|autotune]
cypress autotune [--dry-run] [--deep] [--max-events <n>] [--limit <n>]
cypress add <path>                    # start tracking path
cypress del <path>                    # stop tracking path
cypress roots                         # list tracked roots
cypress index                         # incremental index pass
cypress refresh [path ...]            # force refresh all or specific paths
cypress watch --interval 30 --debounce-ms 900   # event-driven (fswatch) with polling fallback
cypress search [--view projects|files] <query...>
cypress pick --query "<query>" --view projects|files
cypress daemon [start|stop|restart|status|logs [n]] [--host 127.0.0.1] [--port 8765]
cypress self-test [--query "<text>"] [--lifecycle] [--timeout <seconds>]
cypress doctor                        # runtime checks
cypress graph [doctor|status|start|stop|provision|view|open|neighbors|lineage] [--host 127.0.0.1] [--port 8780] [--path /path/to/project-or-child] [--limit 20] [--threshold 0.60] [--depth 2] [--no-system-install] [--no-download] [--from /path/to/falkordb]
cypress api --port 8765               # local JSON API
cypress mcp [serve|doctor]            # MCP server over stdio / readiness check
cypress bench [plan|doctor|export|run] # backend benchmark harness
```

`cypress config` is an interactive settings mode:
- opens a full-screen terminal editor by default in interactive terminals
- keybinds: `↑/↓` move, `Enter` edit, `←/→` cycle enum settings, `a` autotune, `s` save, `q` discard
- non-interactive fallback path remains available (prompt/fzf style)
- run `autotune` directly from the config menu

`cypress watch` notes:
- uses event-driven sync when `fswatch` is available, with automatic polling fallback
- `cypress install` attempts to install `fswatch` automatically (when Homebrew is available and system installs are allowed)
- `--debounce-ms` controls event-batch delay before indexing
- `--quiet` suppresses watch progress output

`cypress autotune` learns from `selection_events` and tunes ranking knobs automatically:
- objective: maximize weighted MRR/hit-rate over your real queries and picks
- tuned families: chunk weights, project weights, graph expansion weights/limits, lexical/vector candidate sizes
- `--deep` runs additional tuning rounds for higher quality at the cost of runtime
- `--dry-run` previews recommended values without writing config
- each run writes a report artifact:
  - `./.cypress/autotune/latest.json`
  - `./.cypress/autotune/latest.md`
  - timestamped snapshots under `./.cypress/autotune/report-*.{json,md}`

`cypress self-test --lifecycle` runs a one-shot native lifecycle probe:
- daemon start/health/stop on a temporary port
- managed-local FalkorDB start/ping/stop on a temporary port
- post-probe retrieval search verification

`cypress` uses a hybrid retrieval pipeline:
- `retrieval_backend = "falkordb"` (default)
- optional `retrieval_backend = "sqlite"` (compat mode)
- ranking is chunk-first: each result carries top chunk evidence with `semantic/lexical/graph/relation/quality` scoring
- relation feedback labels are applied during related-chunk ranking with configurable multipliers:
  - `rank_relation_quality_good_boost`
  - `rank_relation_quality_weak_penalty`
  - `rank_relation_quality_wrong_penalty`
- graph-aware re-ranking expands from top semantic chunks via `RELATED` project neighbors with hop penalty
- when `retrieval_backend = "falkordb"`, Rust `index/search` use FalkorDB at:
  - `CYPRESS_FALKOR_URL` (default: `falkor://127.0.0.1:6379`)
  - `CYPRESS_FALKOR_GRAPH` (default: `cypress`)
- Falkor lifecycle automation is app-managed (project-local runtime state):
  - `CYPRESS_FALKOR_RUNTIME_MODE=managed-local|docker|manual` (default: `managed-local`)
  - `CYPRESS_FALKOR_RUNTIME_CMD` (optional override command for `managed-local`)
  - `CYPRESS_FALKOR_MODULE_PATH` (optional explicit module path when running via Redis module mode)
  - `CYPRESS_FALKOR_DOCKER_CONTAINER` (optional container name for `docker` mode)
  - runtime metadata/logs are stored under `./.cypress/runtime/graph/`
- `cypress install` now provisions runtime strategy and persists:
  - `falkor_runtime_mode` / `falkor_runtime_cmd` / `falkor_docker_container` in config
  - attempts to provision `fswatch` (via Homebrew) so `cypress watch` can run in low-latency event mode
  - installs shell integration into `~/.bashrc` and `~/.zshrc` by default (opt out: `--no-shell-hook`)
  - default behavior: prefer managed-local runtime (standalone `falkordb` binary when available)
  - if standalone `falkordb` is unavailable, managed-local can run Falkor via `redis-server --loadmodule <falkordb-module>`
  - if required runtime assets are missing, Cypress can provision them locally under `./.cypress/bin/`
  - if config is currently docker and local FalkorDB becomes available, install can switch to managed-local automatically
  - docker mode remains optional/explicit only, not auto-selected
- native Rust `api`/`mcp` are included by default (no Python runtime required)
- benchmark backend status in native Rust:
  - `falkordb`: supported
    - query benchmark path is native and writes `./.cypress/bench/report.json` + `report.md`
    - `--dataset-limit` / `--target-chunks` are currently informational in native FalkorDB mode
  - `lancedb`: reported as unsupported placeholder in current native path
- `cypress graph provision` creates/updates a project-local managed runtime wrapper in `./.cypress/bin/falkordb`
  - use `cypress graph provision --no-system-install` to provision only from already-installed local binaries
  - use `cypress graph provision --no-download` to skip release download attempts
  - use `cypress graph provision --from /absolute/path/to/falkordb` to provision from a specific local binary
  - advanced overrides: `CYPRESS_FALKOR_RELEASE_API` and `CYPRESS_FALKOR_DOWNLOAD_URL`
- first-run bootstrap: if runtime keys are missing, `search/index/pick/watch` auto-persist a runtime mode before startup
- if runtime mode is `manual` and FalkorDB is unreachable, `search/index/pick/watch` now auto-attempts a managed-local upgrade (including local runtime provisioning) unless runtime mode is explicitly forced via env
- managed-local reconciliation: if FalkorDB is already running and reachable, Cypress adopts the listener PID into runtime state so `cypress graph stop/restart` remains consistent even if state was missing/stale
- no system service is required; `cypress graph start` launches the selected runtime mode directly
- fallback policy is explicit (not automatic): choose a runtime mode up front, and Cypress does not silently switch modes
- FalkorDB sync is incremental at project level:
  - changed projects (mtime delta) have project/file/chunk nodes refreshed
  - removed projects are deleted from graph
  - `RELATED` edges are refreshed from current SQLite edge state each run

## API

Run API:

```bash
cypress api --host 127.0.0.1 --port 8765
```

Endpoints:
- `GET /health`
- `GET /tracked`
- `GET /search?q=<query>&limit=<n>&view=projects`
- `GET /search?q=<query>&limit=<n>&view=files`
- `GET /chunks/search?q=<query>&limit=<n>`
- `GET /chunks/related?chunk_id=<id>&limit=<n>`
- `GET /chunks/get?chunk_id=<id>&max_chars=<n>`
- `GET /docs/read?path=<abs_or_tilde_path>&max_chars=<n>`
- `GET /chunks/feedback?chunk_id=<id>&decision=suppressed|active&quality=good|weak|wrong|unspecified&limit=<n>`
- `GET /graph/neighbors?path=<abs_or_tilde_path>&limit=<n>`
- `GET /graph/view/data?focus=<project_path>&limit=<n>`
- `GET /graph/view/chunks?path=<project_path>&limit=<n>`
- `GET /graph/view/related?chunk_id=<id>&limit=<n>`
- `POST /chunks/feedback/suppress` with `{ "source_chunk_id": <id>, "target_chunk_id": <id>, "relation": "<kind>", "note": "..." }`
- `POST /chunks/feedback/restore` with `{ "source_chunk_id": <id>, "target_chunk_id": <id>, "relation": "<kind>", "note": "..." }`
- `POST /chunks/feedback/quality` with `{ "source_chunk_id": <id>, "target_chunk_id": <id>, "relation": "<kind>", "quality_label": "good|weak|wrong|unspecified", "note": "..." }`
- `POST /refresh` with `{ "path": "..."} ` or `{ "paths": ["..."] }`
- `POST /select` with `{ "path": "...", "query": "..."} `
- `POST /tracked/add` with `{ "path": "..."} ` or `{ "paths": ["..."] }`
- `POST /tracked/del` with `{ "path": "..."} ` or `{ "paths": ["..."] }`

Chunk lineage payloads (`/chunks/search`, `/chunks/related`, `/chunks/get`, `/docs/read`, MCP `search_chunks`, `get_related_chunks`) include a stable `schema` field (`chunk-search-v1`, `chunk-related-v1`, `chunk-get-v1`, `doc-read-v1`) for contract-safe consumers.

Graph viewer:

```bash
cypress graph view --host 127.0.0.1 --port 8780
# open http://127.0.0.1:8780/
```

Open graph viewer in one command:

```bash
cypress graph open
```

`graph open` prints the active URL. If port `8780` is occupied/stale, Cypress can launch on a different free port.

If the browser says `No graph data yet`:

```bash
cypress doctor                 # verify db path + tracked roots
cypress graph neighbors --limit 5
cypress index                 # only needed if neighbors are empty
cypress graph open
```

Terminal graph summaries (table output):

```bash
# direct outgoing neighbors for a project
cypress graph neighbors --path ~/c-projects/202601-adobe-efs-cost-calculation --limit 12 --threshold 0.60

# lineage around a focus project (incoming + outgoing + relay edges when depth > 1)
cypress graph lineage --path ~/c-projects/202601-adobe-efs-cost-calculation --depth 2 --threshold 0.60 --limit 20
```

Graph viewer highlights:
- project graph canvas with selectable project nodes
- chunk inspector per project with chunk-level drill-down
- related chunk panel with explainability (`Why`) and recursive `Drill` traversal
- relation curation in viewer:
  - `Suppress` / `Restore`
  - quality labels per relation (`Good` / `Weak` / `Wrong` / `Clear`)
  - local feedback history (decision + quality + note + timestamps)

## MCP

Run MCP server:

```bash
cypress mcp
```

MCP tools:
- `search_projects(query, limit)`
- `search_files(query, limit)`
- `search_chunks(query, limit)`
- `get_related_chunks(chunk_id, limit)`
- `list_relation_feedback(chunk_id, decision?, quality?, limit)`
- `suppress_relation(source_chunk_id, target_chunk_id, relation, note?)`
- `restore_relation(source_chunk_id, target_chunk_id, relation, note?)`
- `set_relation_quality(source_chunk_id, target_chunk_id, relation, quality_label, note?)`
- `get_project_neighbors(path, limit)`
- `list_tracked_roots()`
- `add_tracked_root(path, refresh)`
- `remove_tracked_root(path, refresh)`
- `run_incremental_index()`
- `run_forced_refresh(paths?)`

MCP resource:
- `cypress://status`

Codex registration:

```bash
codex mcp add cypress -- "$(pwd)/cypress" mcp
```

## Paths

- When running via repo launcher (`./cypress`) or `scripts/cypress-shell.sh`:
  - Config (default): `<repo>/.cypress/config.toml`
  - Data (default): `<repo>/.cypress/`
  - DB (default): `<repo>/.cypress/cypress.db`
- When running `target/.../cypress` directly without env overrides:
  - defaults are current-working-directory local (`./.cypress/...`)

Override via env:
- `CYPRESS_CONFIG=/path/to/config.toml`
- `CYPRESS_DATA_DIR=/path/to/state_dir`
