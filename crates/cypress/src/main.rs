use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::f64::consts::PI;
use std::ffi::OsString;
use std::fs;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, BufWriter, IsTerminal, Read, Write};
use std::net::{TcpListener, TcpStream};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::Component;
use std::path::{Path, PathBuf};
use std::process::{self, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::Duration;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use rusqlite::{params, params_from_iter, Connection, OpenFlags, OptionalExtension};
use serde_json::Value;
use sha1::{Digest, Sha1};

mod falkor_store;

fn main() {
    let args: Vec<OsString> = env::args_os().skip(1).collect();
    if args.is_empty() {
        print_help();
        return;
    }

    let first = args[0].to_string_lossy().to_string();
    match first.as_str() {
        "-h" | "--help" | "help" => {
            print_help();
        }
        "-V" | "--version" | "version" => {
            println!("cypress {}", env!("CARGO_PKG_VERSION"));
        }
        "doctor" => run_doctor(),
        "config" => run_config_cmd(&args[1..]),
        "autotune" => run_autotune_cmd(&args[1..]),
        "init" => run_init(&args[1..]),
        "install" => run_install_cmd(&args[1..]),
        "add" => run_add(&args[1..]),
        "del" => run_del(&args[1..]),
        "roots" => run_roots(),
        "index" => run_index_cmd(&args[1..]),
        "refresh" => run_refresh_cmd(&args[1..]),
        "watch" => run_watch_cmd(&args[1..]),
        "search" => run_search_cmd(&args[1..]),
        "pick" => run_pick_cmd(&args[1..]),
        "daemon" => run_daemon_cmd(&args[1..]),
        "bench" => run_bench_cmd(&args[1..]),
        "api" => run_api_cmd(&args[1..]),
        "mcp" => run_mcp_cmd(&args[1..]),
        "self-test" => run_self_test_cmd(&args[1..]),
        "graph" => run_graph_cmd(&args[1..]),
        "legacy" => run_legacy_cmd(&args[1..]),
        _ => {
            eprintln!("error: unknown command '{}'", first);
            eprintln!();
            print_help();
            process::exit(2);
        }
    }
}

fn print_help() {
    println!("cypress (rust)");
    println!();
    println!("Native commands:");
    println!("  cypress doctor");
    println!("  cypress config [edit|show|set <key> <value>|autotune] [options]");
    println!("  cypress autotune [--dry-run] [--deep] [--max-events <n>] [--limit <n>]");
    println!("  cypress version");
    println!("  cypress init [--root <path>] [--embed-backend <local-hash|ollama|local(alias)>] [--embed-model <id>] [--retrieval-backend <falkordb|sqlite>] [--local-embed-dim <n>]");
    println!("  cypress install [--no-system-install] [--no-download] [--no-shell-hook] [--legacy] [--venv <path>] [--bench]");
    println!("  cypress add <path> [path ...] [--refresh|--no-refresh]");
    println!("  cypress del <path> [path ...] [--refresh|--no-refresh]");
    println!("  cypress roots");
    println!("  cypress index");
    println!("  cypress refresh [path ...]");
    println!("  cypress watch [--interval <seconds>] [--debounce-ms <ms>] [--once] [--quiet]");
    println!("  cypress search [--view projects|files] [--limit <n>] <query...>");
    println!(
        "  cypress pick [--query <text>] [--view projects|files] [--limit <n>] [--emit-path-file <path>]"
    );
    println!(
        "  cypress daemon [start|stop|restart|status|logs [n]] [--host <addr>] [--port <n>] [--timeout <seconds>]"
    );
    println!("  cypress bench [plan|doctor|export|run] [options]");
    println!("  cypress graph [doctor|status|start|stop|provision|view|open|neighbors|lineage]");
    println!("  cypress api [--host <addr>] [--port <n>]");
    println!("  cypress mcp [serve|doctor]");
    println!("  cypress self-test");
    println!();
    println!("Compatibility:");
    println!("  cypress legacy <args...>   # explicit Python bridge");
}

fn run_doctor() {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let state_dir = data_dir(&cwd);
    let cfg_path = config_path(&cwd);
    let db_path = db_path(&cwd);

    let cfg = load_config_values(&cfg_path);
    let root = cfg
        .get("root")
        .map(PathBuf::from)
        .unwrap_or_else(|| expand_tilde("~/c-projects"));
    let embed_backend = cfg
        .get("embed_backend")
        .cloned()
        .unwrap_or_else(|| "local-hash".to_string());
    let embed_model = cfg
        .get("embed_model")
        .cloned()
        .unwrap_or_else(|| "sentence-transformers/all-MiniLM-L6-v2".to_string());
    let retrieval_backend = cfg
        .get("retrieval_backend")
        .cloned()
        .unwrap_or_else(|| "falkordb".to_string());
    let local_embed_dim = cfg
        .get("local_embed_dim")
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(384);

    let root_exists = root.is_dir();
    let cfg_exists = cfg_path.exists();
    let db_exists = db_path.exists();
    let fzf_installed = command_exists("fzf");
    let fswatch_installed = command_exists("fswatch");
    let tracked_roots = tracked_roots_count(&db_path).unwrap_or(0);
    let db_ready = database_ready(&db_path);

    println!("config: {}", cfg_path.display());
    println!("db: {}", db_path.display());
    println!(
        "root exists: {} ({})",
        yes_no(root_exists),
        root.to_string_lossy()
    );
    println!("fzf installed: {}", yes_no(fzf_installed));
    println!(
        "fswatch installed: {}{}",
        yes_no(fswatch_installed),
        if fswatch_installed {
            ""
        } else {
            " (watch uses polling fallback)"
        }
    );
    println!("embed backend configured: {}", embed_backend);
    if embed_backend == "local" {
        println!("embed backend note: 'local' is treated as native local-hash alias");
    }
    println!("embed model configured: {}", embed_model);
    println!("retrieval backend configured: {}", retrieval_backend);
    println!(
        "falkor runtime configured: mode={} cmd={} container={}",
        cfg.get("falkor_runtime_mode")
            .cloned()
            .unwrap_or_else(|| "managed-local".to_string()),
        cfg.get("falkor_runtime_cmd")
            .cloned()
            .unwrap_or_else(|| "<auto>".to_string()),
        cfg.get("falkor_docker_container")
            .cloned()
            .unwrap_or_else(|| "cypress-falkordb".to_string())
    );
    println!("local embed dim: {}", local_embed_dim);
    println!("embed backend active: {}", embed_backend);
    println!("retrieval backend active: {}", retrieval_backend);
    if retrieval_backend == "falkordb" {
        match falkor_store::ping() {
            Ok(_) => println!("falkordb reachable: yes"),
            Err(err) => {
                println!("falkordb reachable: no");
                eprintln!("warning: {}", err);
            }
        }
    }
    println!("tracked roots: {}", tracked_roots);
    println!("database ready: {}", yes_no(db_ready && db_exists));

    if !cfg_exists {
        eprintln!(
            "warning: config is missing; initialize with: cypress init --root {}",
            state_dir.to_string_lossy()
        );
    }
}

#[derive(Clone, Debug)]
struct AutotuneOptions {
    dry_run: bool,
    deep: bool,
    max_events: usize,
    limit: usize,
}

#[derive(Clone, Debug)]
struct AutotuneExample {
    query: String,
    path: String,
    weight: f64,
}

#[derive(Clone, Debug)]
struct AutotuneOutcome {
    cfg: ConfigValues,
    examples_used: usize,
    baseline_mrr: f64,
    baseline_hit1: f64,
    baseline_hit3: f64,
    best_mrr: f64,
    best_hit1: f64,
    best_hit3: f64,
    candidates_tested: usize,
    used_history: bool,
}

fn parse_autotune_options(args: &[OsString]) -> AutotuneOptions {
    let mut dry_run = false;
    let mut deep = false;
    let mut max_events = 320usize;
    let mut limit = 40usize;
    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        match s.as_str() {
            "--dry-run" => {
                dry_run = true;
            }
            "--deep" => {
                deep = true;
            }
            "--max-events" => {
                i += 1;
                let raw = arg_value(args, i, "--max-events");
                max_events = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --max-events must be an integer");
                    process::exit(2);
                });
            }
            "--limit" => {
                i += 1;
                let raw = arg_value(args, i, "--limit");
                limit = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --limit must be an integer");
                    process::exit(2);
                });
            }
            other if other.starts_with("--max-events=") => {
                let raw = other.trim_start_matches("--max-events=");
                max_events = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --max-events must be an integer");
                    process::exit(2);
                });
            }
            other if other.starts_with("--limit=") => {
                let raw = other.trim_start_matches("--limit=");
                limit = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --limit must be an integer");
                    process::exit(2);
                });
            }
            other => {
                eprintln!("error: unknown autotune option '{}'", other);
                process::exit(2);
            }
        }
        i += 1;
    }
    AutotuneOptions {
        dry_run,
        deep,
        max_events: max_events.clamp(20, 5000),
        limit: limit.clamp(5, 120),
    }
}

fn config_rows() -> Vec<(&'static str, &'static str)> {
    vec![
        ("root", "Tracked root directory"),
        ("embed_backend", "Embedding backend"),
        ("embed_model", "Embedding model id"),
        ("retrieval_backend", "Retrieval backend"),
        ("falkor_runtime_mode", "Falkor runtime mode"),
        ("falkor_runtime_cmd", "Managed-local runtime command"),
        ("falkor_docker_container", "Docker container name"),
        ("local_embed_dim", "Embedding dimension"),
        ("max_chars_per_project", "Indexing cap per project"),
        ("lexical_candidates", "Lexical candidates"),
        ("vector_candidates", "Vector candidates"),
        ("rank_chunk_semantic_weight", "Chunk score semantic weight"),
        ("rank_chunk_lexical_weight", "Chunk score lexical weight"),
        ("rank_chunk_graph_weight", "Chunk score graph weight"),
        ("rank_quality_mix", "Chunk quality mixing factor"),
        (
            "rank_relation_quality_good_boost",
            "Related score boost for quality=good",
        ),
        (
            "rank_relation_quality_weak_penalty",
            "Related score penalty for quality=weak",
        ),
        (
            "rank_relation_quality_wrong_penalty",
            "Related score penalty for quality=wrong",
        ),
        ("rank_project_content_weight", "Project rank content weight"),
        (
            "rank_project_semantic_weight",
            "Project rank semantic weight",
        ),
        (
            "rank_project_path_weight",
            "Project rank path keyword weight",
        ),
        ("rank_project_graph_weight", "Project rank graph weight"),
        (
            "rank_project_frecency_weight",
            "Project rank frecency weight",
        ),
        ("graph_seed_limit", "Graph expansion seed chunk count"),
        ("graph_neighbor_limit", "Graph neighbor traversal limit"),
        (
            "graph_same_project_high",
            "Same-project graph weight (high semantic)",
        ),
        (
            "graph_same_project_low",
            "Same-project graph weight (low semantic)",
        ),
        ("graph_related_base", "Related-project graph base"),
        ("graph_related_scale", "Related-project graph scale"),
        ("graph_related_cap", "Related-project graph cap"),
    ]
}

fn config_enum_options(key: &str) -> Option<Vec<&'static str>> {
    match key {
        "embed_backend" => Some(vec!["local-hash", "local", "ollama"]),
        "retrieval_backend" => Some(vec!["falkordb", "sqlite"]),
        "falkor_runtime_mode" => Some(vec!["managed-local", "manual", "docker"]),
        _ => None,
    }
}

fn config_value_string(cfg: &ConfigValues, key: &str) -> Option<String> {
    match key {
        "root" => Some(cfg.root.to_string_lossy().to_string()),
        "embed_backend" => Some(cfg.embed_backend.clone()),
        "embed_model" => Some(cfg.embed_model.clone()),
        "retrieval_backend" => Some(cfg.retrieval_backend.clone()),
        "falkor_runtime_mode" => Some(cfg.falkor_runtime_mode.clone()),
        "falkor_runtime_cmd" => Some(cfg.falkor_runtime_cmd.clone()),
        "falkor_docker_container" => Some(cfg.falkor_docker_container.clone()),
        "local_embed_dim" => Some(cfg.local_embed_dim.to_string()),
        "max_chars_per_project" => Some(cfg.max_chars_per_project.to_string()),
        "lexical_candidates" => Some(cfg.lexical_candidates.to_string()),
        "vector_candidates" => Some(cfg.vector_candidates.to_string()),
        "rank_chunk_semantic_weight" => Some(format!("{:.6}", cfg.rank_chunk_semantic_weight)),
        "rank_chunk_lexical_weight" => Some(format!("{:.6}", cfg.rank_chunk_lexical_weight)),
        "rank_chunk_graph_weight" => Some(format!("{:.6}", cfg.rank_chunk_graph_weight)),
        "rank_quality_mix" => Some(format!("{:.6}", cfg.rank_quality_mix)),
        "rank_relation_quality_good_boost" => {
            Some(format!("{:.6}", cfg.rank_relation_quality_good_boost))
        }
        "rank_relation_quality_weak_penalty" => {
            Some(format!("{:.6}", cfg.rank_relation_quality_weak_penalty))
        }
        "rank_relation_quality_wrong_penalty" => {
            Some(format!("{:.6}", cfg.rank_relation_quality_wrong_penalty))
        }
        "rank_project_content_weight" => Some(format!("{:.6}", cfg.rank_project_content_weight)),
        "rank_project_semantic_weight" => Some(format!("{:.6}", cfg.rank_project_semantic_weight)),
        "rank_project_path_weight" => Some(format!("{:.6}", cfg.rank_project_path_weight)),
        "rank_project_graph_weight" => Some(format!("{:.6}", cfg.rank_project_graph_weight)),
        "rank_project_frecency_weight" => Some(format!("{:.6}", cfg.rank_project_frecency_weight)),
        "graph_seed_limit" => Some(cfg.graph_seed_limit.to_string()),
        "graph_neighbor_limit" => Some(cfg.graph_neighbor_limit.to_string()),
        "graph_same_project_high" => Some(format!("{:.6}", cfg.graph_same_project_high)),
        "graph_same_project_low" => Some(format!("{:.6}", cfg.graph_same_project_low)),
        "graph_related_base" => Some(format!("{:.6}", cfg.graph_related_base)),
        "graph_related_scale" => Some(format!("{:.6}", cfg.graph_related_scale)),
        "graph_related_cap" => Some(format!("{:.6}", cfg.graph_related_cap)),
        _ => None,
    }
}

fn config_set_value(cfg: &mut ConfigValues, key: &str, raw: &str) -> Result<(), String> {
    let value = raw.trim();
    match key {
        "root" => {
            if value.is_empty() {
                return Err("root must not be empty".to_string());
            }
            cfg.root = normalize_path(value);
        }
        "embed_backend" => {
            let v = value.to_lowercase();
            if !matches!(v.as_str(), "local-hash" | "local" | "ollama") {
                return Err("embed_backend must be one of: local-hash, local, ollama".to_string());
            }
            cfg.embed_backend = v;
        }
        "embed_model" => {
            if value.is_empty() {
                return Err("embed_model must not be empty".to_string());
            }
            cfg.embed_model = value.to_string();
        }
        "retrieval_backend" => {
            let v = value.to_lowercase();
            if !matches!(v.as_str(), "falkordb" | "sqlite") {
                return Err("retrieval_backend must be one of: falkordb, sqlite".to_string());
            }
            cfg.retrieval_backend = v;
        }
        "falkor_runtime_mode" => {
            let v = value.to_lowercase();
            if !matches!(v.as_str(), "managed-local" | "manual" | "docker") {
                return Err(
                    "falkor_runtime_mode must be one of: managed-local, manual, docker".to_string(),
                );
            }
            cfg.falkor_runtime_mode = v;
        }
        "falkor_runtime_cmd" => {
            cfg.falkor_runtime_cmd = value.to_string();
        }
        "falkor_docker_container" => {
            cfg.falkor_docker_container = value.to_string();
        }
        "local_embed_dim" => {
            cfg.local_embed_dim = value
                .parse::<i64>()
                .map_err(|_| "local_embed_dim must be an integer".to_string())?
                .max(64);
        }
        "max_chars_per_project" => {
            cfg.max_chars_per_project = value
                .parse::<i64>()
                .map_err(|_| "max_chars_per_project must be an integer".to_string())?
                .clamp(1000, 500_000);
        }
        "lexical_candidates" => {
            cfg.lexical_candidates = value
                .parse::<i64>()
                .map_err(|_| "lexical_candidates must be an integer".to_string())?
                .clamp(10, 5000);
        }
        "vector_candidates" => {
            cfg.vector_candidates = value
                .parse::<i64>()
                .map_err(|_| "vector_candidates must be an integer".to_string())?
                .clamp(10, 5000);
        }
        "rank_chunk_semantic_weight" => {
            cfg.rank_chunk_semantic_weight = value
                .parse::<f64>()
                .map_err(|_| "rank_chunk_semantic_weight must be a number".to_string())?
                .clamp(0.0, 1.0);
        }
        "rank_chunk_lexical_weight" => {
            cfg.rank_chunk_lexical_weight = value
                .parse::<f64>()
                .map_err(|_| "rank_chunk_lexical_weight must be a number".to_string())?
                .clamp(0.0, 1.0);
        }
        "rank_chunk_graph_weight" => {
            cfg.rank_chunk_graph_weight = value
                .parse::<f64>()
                .map_err(|_| "rank_chunk_graph_weight must be a number".to_string())?
                .clamp(0.0, 1.0);
        }
        "rank_quality_mix" => {
            cfg.rank_quality_mix = value
                .parse::<f64>()
                .map_err(|_| "rank_quality_mix must be a number".to_string())?
                .clamp(0.0, 1.0);
        }
        "rank_relation_quality_good_boost" => {
            cfg.rank_relation_quality_good_boost = value
                .parse::<f64>()
                .map_err(|_| "rank_relation_quality_good_boost must be a number".to_string())?
                .clamp(0.0, 1.0);
        }
        "rank_relation_quality_weak_penalty" => {
            cfg.rank_relation_quality_weak_penalty = value
                .parse::<f64>()
                .map_err(|_| "rank_relation_quality_weak_penalty must be a number".to_string())?
                .clamp(0.0, 1.0);
        }
        "rank_relation_quality_wrong_penalty" => {
            cfg.rank_relation_quality_wrong_penalty = value
                .parse::<f64>()
                .map_err(|_| "rank_relation_quality_wrong_penalty must be a number".to_string())?
                .clamp(0.0, 1.0);
        }
        "rank_project_content_weight" => {
            cfg.rank_project_content_weight = value
                .parse::<f64>()
                .map_err(|_| "rank_project_content_weight must be a number".to_string())?
                .clamp(0.0, 2.0);
        }
        "rank_project_semantic_weight" => {
            cfg.rank_project_semantic_weight = value
                .parse::<f64>()
                .map_err(|_| "rank_project_semantic_weight must be a number".to_string())?
                .clamp(0.0, 2.0);
        }
        "rank_project_path_weight" => {
            cfg.rank_project_path_weight = value
                .parse::<f64>()
                .map_err(|_| "rank_project_path_weight must be a number".to_string())?
                .clamp(0.0, 2.0);
        }
        "rank_project_graph_weight" => {
            cfg.rank_project_graph_weight = value
                .parse::<f64>()
                .map_err(|_| "rank_project_graph_weight must be a number".to_string())?
                .clamp(0.0, 2.0);
        }
        "rank_project_frecency_weight" => {
            cfg.rank_project_frecency_weight = value
                .parse::<f64>()
                .map_err(|_| "rank_project_frecency_weight must be a number".to_string())?
                .clamp(0.0, 2.0);
        }
        "graph_seed_limit" => {
            cfg.graph_seed_limit = value
                .parse::<i64>()
                .map_err(|_| "graph_seed_limit must be an integer".to_string())?
                .clamp(2, 64);
        }
        "graph_neighbor_limit" => {
            cfg.graph_neighbor_limit = value
                .parse::<i64>()
                .map_err(|_| "graph_neighbor_limit must be an integer".to_string())?
                .clamp(8, 500);
        }
        "graph_same_project_high" => {
            cfg.graph_same_project_high = value
                .parse::<f64>()
                .map_err(|_| "graph_same_project_high must be a number".to_string())?
                .clamp(0.0, 1.0);
        }
        "graph_same_project_low" => {
            cfg.graph_same_project_low = value
                .parse::<f64>()
                .map_err(|_| "graph_same_project_low must be a number".to_string())?
                .clamp(0.0, 1.0);
        }
        "graph_related_base" => {
            cfg.graph_related_base = value
                .parse::<f64>()
                .map_err(|_| "graph_related_base must be a number".to_string())?
                .clamp(0.0, 1.0);
        }
        "graph_related_scale" => {
            cfg.graph_related_scale = value
                .parse::<f64>()
                .map_err(|_| "graph_related_scale must be a number".to_string())?
                .clamp(0.0, 2.0);
        }
        "graph_related_cap" => {
            cfg.graph_related_cap = value
                .parse::<f64>()
                .map_err(|_| "graph_related_cap must be a number".to_string())?
                .clamp(0.0, 1.0);
        }
        _ => return Err(format!("unknown config key '{}'", key)),
    }
    Ok(())
}

fn print_config_values(cfg: &ConfigValues) {
    for (key, hint) in config_rows() {
        if let Some(v) = config_value_string(cfg, key) {
            println!("{:<30} = {:<24} # {}", key, v, hint);
        }
    }
}

fn prompt_line(prompt: &str) -> Result<String, String> {
    print!("{}", prompt);
    std::io::stdout()
        .flush()
        .map_err(|e| format!("failed flushing stdout: {}", e))?;
    let mut line = String::new();
    std::io::stdin()
        .read_line(&mut line)
        .map_err(|e| format!("failed reading input: {}", e))?;
    Ok(line.trim_end_matches(&['\r', '\n'][..]).to_string())
}

fn tty_ui_available() -> bool {
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        return false;
    }
    if !command_exists("stty") {
        return false;
    }
    run_stty_capture(["-g"]).is_ok()
}

#[derive(Clone, Debug)]
enum ConfigTuiMode {
    Navigate,
    Edit { key: String, buffer: String },
}

#[derive(Clone, Debug)]
struct ConfigTuiState {
    selected: usize,
    scroll: usize,
    dirty: bool,
    discard_armed: bool,
    status: String,
    mode: ConfigTuiMode,
}

enum ConfigKey {
    Up,
    Down,
    Left,
    Right,
    PageUp,
    PageDown,
    Home,
    End,
    Enter,
    Tab,
    Esc,
    Backspace,
    CtrlC,
    Char(char),
    Unknown,
}

struct ConfigTuiGuard {
    stty_state: Option<String>,
}

impl Drop for ConfigTuiGuard {
    fn drop(&mut self) {
        let mut stdout = std::io::stdout();
        let _ = stdout.write_all(b"\x1b[0m\x1b[?25h\x1b[?1049l");
        let _ = stdout.flush();
        if let Some(state) = &self.stty_state {
            let _ = run_stty(args_slice([state.as_str()]));
        }
    }
}

fn enter_config_tui_mode() -> Result<ConfigTuiGuard, String> {
    if !command_exists("stty") {
        return Err("stty is required for full-screen config mode".to_string());
    }
    let state_str = run_stty_capture(["-g"]).map_err(|_| "failed reading tty mode".to_string())?;
    run_stty(["raw", "-echo", "min", "0", "time", "1"])
        .map_err(|_| "failed switching tty to raw mode".to_string())?;

    let mut stdout = std::io::stdout();
    stdout
        .write_all(b"\x1b[?1049h\x1b[?25l")
        .map_err(|e| format!("failed entering alternate screen: {}", e))?;
    stdout
        .flush()
        .map_err(|e| format!("failed flushing terminal init: {}", e))?;

    Ok(ConfigTuiGuard {
        stty_state: Some(state_str),
    })
}

fn args_slice<const N: usize>(arr: [&str; N]) -> [&str; N] {
    arr
}

fn run_stty<const N: usize>(args: [&str; N]) -> Result<(), String> {
    let mut cmd = String::from("stty");
    for arg in args {
        cmd.push(' ');
        cmd.push_str(arg);
    }
    cmd.push_str(" < /dev/tty > /dev/tty 2>/dev/null");
    let status = Command::new("bash")
        .arg("-lc")
        .arg(cmd)
        .status()
        .map_err(|e| format!("failed running stty: {}", e))?;
    if status.success() {
        Ok(())
    } else {
        Err("stty returned non-zero status".to_string())
    }
}

fn run_stty_capture<const N: usize>(args: [&str; N]) -> Result<String, String> {
    let mut cmd = String::from("stty");
    for arg in args {
        cmd.push(' ');
        cmd.push_str(arg);
    }
    cmd.push_str(" < /dev/tty 2>/dev/null");
    let out = Command::new("bash")
        .arg("-lc")
        .arg(cmd)
        .output()
        .map_err(|e| format!("failed running stty capture: {}", e))?;
    if !out.status.success() {
        return Err("stty capture returned non-zero status".to_string());
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn read_config_key() -> Result<Option<ConfigKey>, String> {
    let mut stdin = std::io::stdin();
    let mut buf = [0u8; 32];
    let n = stdin
        .read(&mut buf)
        .map_err(|e| format!("failed reading key input: {}", e))?;
    if n == 0 {
        return Ok(None);
    }
    let mut seq = buf[..n].to_vec();
    if seq[0] == 0x1b && seq.len() == 1 {
        let mut extra = [0u8; 16];
        let n2 = stdin
            .read(&mut extra)
            .map_err(|e| format!("failed reading escape sequence: {}", e))?;
        if n2 > 0 {
            seq.extend_from_slice(&extra[..n2]);
        }
    }
    let key = match seq[0] {
        0x03 => ConfigKey::CtrlC,
        0x09 => ConfigKey::Tab,
        b'\r' | b'\n' => ConfigKey::Enter,
        0x7f | 0x08 => ConfigKey::Backspace,
        0x1b => {
            if seq.len() >= 3 && seq[1] == b'[' {
                match seq[2] {
                    b'A' => ConfigKey::Up,
                    b'B' => ConfigKey::Down,
                    b'C' => ConfigKey::Right,
                    b'D' => ConfigKey::Left,
                    b'H' => ConfigKey::Home,
                    b'F' => ConfigKey::End,
                    b'5' if seq.get(3) == Some(&b'~') => ConfigKey::PageUp,
                    b'6' if seq.get(3) == Some(&b'~') => ConfigKey::PageDown,
                    b'1' if seq.get(3) == Some(&b'~') => ConfigKey::Home,
                    b'4' if seq.get(3) == Some(&b'~') => ConfigKey::End,
                    _ => ConfigKey::Esc,
                }
            } else {
                ConfigKey::Esc
            }
        }
        b => {
            if (0x20..=0x7e).contains(&b) {
                ConfigKey::Char(b as char)
            } else {
                ConfigKey::Unknown
            }
        }
    };
    Ok(Some(key))
}

fn terminal_size_fallback() -> (usize, usize) {
    let cols = env::var("COLUMNS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(120);
    let rows = env::var("LINES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(36);
    (cols.max(40), rows.max(12))
}

fn terminal_size_stty() -> (usize, usize) {
    if let Ok(raw) = run_stty_capture(["size"]) {
        let mut parts = raw.split_whitespace();
        if let (Some(rows), Some(cols)) = (parts.next(), parts.next()) {
            if let (Ok(r), Ok(c)) = (rows.parse::<usize>(), cols.parse::<usize>()) {
                return (c.max(40), r.max(12));
            }
        }
    }
    if command_exists("tput") {
        let cols = Command::new("bash")
            .arg("-lc")
            .arg("tput cols < /dev/tty 2>/dev/null")
            .output()
            .ok()
            .and_then(|o| {
                if o.status.success() {
                    String::from_utf8_lossy(&o.stdout)
                        .trim()
                        .parse::<usize>()
                        .ok()
                } else {
                    None
                }
            });
        let rows = Command::new("bash")
            .arg("-lc")
            .arg("tput lines < /dev/tty 2>/dev/null")
            .output()
            .ok()
            .and_then(|o| {
                if o.status.success() {
                    String::from_utf8_lossy(&o.stdout)
                        .trim()
                        .parse::<usize>()
                        .ok()
                } else {
                    None
                }
            });
        if let (Some(c), Some(r)) = (cols, rows) {
            return (c.max(40), r.max(12));
        }
    }
    terminal_size_fallback()
}

fn clipped(s: &str, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }
    let count = s.chars().count();
    if count <= max_width {
        return s.to_string();
    }
    if max_width <= 1 {
        return "…".to_string();
    }
    let mut out = String::new();
    for ch in s.chars().take(max_width.saturating_sub(1)) {
        out.push(ch);
    }
    out.push('…');
    out
}

fn config_rows_count() -> usize {
    config_rows().len()
}

fn cycle_enum_setting(cfg: &mut ConfigValues, key: &str, direction: i32) -> Result<bool, String> {
    let Some(options) = config_enum_options(key) else {
        return Ok(false);
    };
    if options.is_empty() {
        return Ok(false);
    }
    let current = config_value_string(cfg, key).unwrap_or_default();
    let mut pos = options.iter().position(|v| *v == current).unwrap_or(0) as i32;
    pos += direction.signum();
    if pos < 0 {
        pos = options.len() as i32 - 1;
    }
    if pos as usize >= options.len() {
        pos = 0;
    }
    let next = options[pos as usize];
    config_set_value(cfg, key, next)?;
    Ok(true)
}

fn draw_config_tui(
    stdout: &mut std::io::Stdout,
    cfg: &ConfigValues,
    cfg_path: &Path,
    state: &mut ConfigTuiState,
) -> Result<(), String> {
    let rows = config_rows();
    if rows.is_empty() {
        return Err("no config rows available".to_string());
    }
    let (width, height) = terminal_size_stty();

    let header_rows = 4usize;
    let footer_rows = 3usize;
    let min_height = header_rows + footer_rows + 2;
    let mut visible_rows = height.saturating_sub(header_rows + footer_rows);
    if height < min_height {
        visible_rows = 1;
    }
    if state.selected >= rows.len() {
        state.selected = rows.len().saturating_sub(1);
    }
    if state.selected < state.scroll {
        state.scroll = state.selected;
    }
    if state.selected >= state.scroll + visible_rows {
        state.scroll = state
            .selected
            .saturating_sub(visible_rows.saturating_sub(1));
    }

    let title = "cypress config";
    let mode_text = match &state.mode {
        ConfigTuiMode::Navigate => "mode: navigate",
        ConfigTuiMode::Edit { .. } => "mode: edit",
    };
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!(
        "\x1b[36;1m{}\x1b[0m \x1b[90m{}\x1b[0m",
        clipped(title, 20),
        clipped(mode_text, width.saturating_sub(22))
    ));
    lines.push(format!(
        "\x1b[90m{}\x1b[0m",
        clipped(&format!("config: {}", cfg_path.display()), width)
    ));
    lines.push(format!(
        "\x1b[90m{}\x1b[0m",
        clipped(
            "↑/↓ move  ←/→ cycle enums  Enter edit  a autotune  s save  q discard",
            width
        )
    ));

    let value_col = 34usize;
    let key_w = 30usize;
    let value_w = 34usize;
    lines.push(format!(
        "\x1b[1m{}\x1b[0m",
        clipped(
            &format!(
                "{:<key_w$} {:<value_w$} {}",
                "key",
                "value",
                "description",
                key_w = key_w,
                value_w = value_w
            ),
            width
        )
    ));

    for i in 0..visible_rows {
        let row_index = state.scroll + i;
        if row_index >= rows.len() {
            break;
        }
        let (key, hint) = rows[row_index];
        let value = config_value_string(cfg, key).unwrap_or_default();
        let key_txt = clipped(key, key_w);
        let val_txt = clipped(&value, value_w);
        let hint_txt = clipped(hint, width.saturating_sub(value_col + value_w + 2).max(1));
        let row_text = clipped(
            &format!(
                "{:<key_w$} {:<value_w$} {}",
                key_txt,
                val_txt,
                hint_txt,
                key_w = key_w,
                value_w = value_w
            ),
            width,
        );
        if row_index == state.selected {
            lines.push(format!("\x1b[7m{}\x1b[0m", row_text));
        } else {
            lines.push(row_text);
        }
    }

    let mode_line = match &state.mode {
        ConfigTuiMode::Navigate => {
            "navigate: Enter edit | s save | q discard | a autotune".to_string()
        }
        ConfigTuiMode::Edit { key, buffer } => {
            format!("edit {} = {}  (Enter apply, Esc cancel)", key, buffer)
        }
    };
    while lines.len() + footer_rows < height {
        lines.push(String::new());
    }
    lines.push(format!(
        "\x1b[90m{}\x1b[0m",
        clipped(
            &format!(
                "item {}/{}{}",
                state.selected + 1,
                rows.len(),
                if state.dirty { "  [unsaved]" } else { "" }
            ),
            width
        )
    ));
    lines.push(format!("\x1b[33m{}\x1b[0m", clipped(&mode_line, width)));
    lines.push(format!("\x1b[36m{}\x1b[0m", clipped(&state.status, width)));

    let payload = format!("\x1b[H\x1b[2J{}", lines.join("\n"));
    stdout
        .write_all(payload.as_bytes())
        .map_err(|e| format!("failed writing TUI frame: {}", e))?;
    stdout
        .flush()
        .map_err(|e| format!("failed flushing TUI frame: {}", e))?;
    Ok(())
}

fn run_config_tui(cwd: &Path, cfg_path: &Path, working: &mut ConfigValues) -> Result<bool, String> {
    let _guard = enter_config_tui_mode()?;
    let mut stdout = std::io::stdout();

    let mut state = ConfigTuiState {
        selected: 0,
        scroll: 0,
        dirty: false,
        discard_armed: false,
        status: "full-screen config loaded".to_string(),
        mode: ConfigTuiMode::Navigate,
    };
    let total_rows = config_rows_count();
    if total_rows == 0 {
        return Err("no editable config rows".to_string());
    }
    loop {
        draw_config_tui(&mut stdout, working, cfg_path, &mut state)?;
        let Some(key_event) = read_config_key()? else {
            continue;
        };
        match &mut state.mode {
            ConfigTuiMode::Navigate => match key_event {
                ConfigKey::Up => {
                    if state.selected > 0 {
                        state.selected -= 1;
                    }
                    state.discard_armed = false;
                }
                ConfigKey::Down => {
                    if state.selected + 1 < total_rows {
                        state.selected += 1;
                    }
                    state.discard_armed = false;
                }
                ConfigKey::PageUp => {
                    state.selected = state.selected.saturating_sub(10);
                    state.discard_armed = false;
                }
                ConfigKey::PageDown => {
                    state.selected = (state.selected + 10).min(total_rows.saturating_sub(1));
                    state.discard_armed = false;
                }
                ConfigKey::Home => {
                    state.selected = 0;
                    state.discard_armed = false;
                }
                ConfigKey::End => {
                    state.selected = total_rows.saturating_sub(1);
                    state.discard_armed = false;
                }
                ConfigKey::Left | ConfigKey::Right => {
                    let rows = config_rows();
                    let key = rows[state.selected].0;
                    let direction = if matches!(key_event, ConfigKey::Left) {
                        -1
                    } else {
                        1
                    };
                    match cycle_enum_setting(working, key, direction) {
                        Ok(true) => {
                            state.dirty = true;
                            state.status = format!(
                                "updated {} -> {}",
                                key,
                                config_value_string(working, key).unwrap_or_default()
                            );
                        }
                        Ok(false) => {
                            state.status =
                                "selected key has no enum options; press Enter to edit".to_string();
                        }
                        Err(e) => {
                            state.status = e;
                        }
                    }
                    state.discard_armed = false;
                }
                ConfigKey::Enter => {
                    let rows = config_rows();
                    let key = rows[state.selected].0.to_string();
                    let value = config_value_string(working, &key).unwrap_or_default();
                    state.mode = ConfigTuiMode::Edit { key, buffer: value };
                    state.status =
                        "editing value; press Enter to apply or Esc to cancel".to_string();
                    state.discard_armed = false;
                }
                ConfigKey::Char('s') => {
                    write_config_file(cfg_path, working)?;
                    state.status = format!("config saved: {}", cfg_path.display());
                    return Ok(true);
                }
                ConfigKey::Char('a') => {
                    state.status = "running autotune...".to_string();
                    draw_config_tui(&mut stdout, working, cfg_path, &mut state)?;
                    let result = (|| -> Result<AutotuneOutcome, String> {
                        let dbp = db_path(cwd);
                        ensure_db_schema(&dbp)?;
                        let conn = open_db_rw(&dbp)?;
                        autotune_recommendation(&conn, working, 320, 40, false)
                    })();
                    match result {
                        Ok(outcome) => {
                            *working = outcome.cfg;
                            state.dirty = true;
                            state.status = format!(
                                "autotune applied: mrr {:.4}->{:.4}, hit3 {:.4}->{:.4}",
                                outcome.baseline_mrr,
                                outcome.best_mrr,
                                outcome.baseline_hit3,
                                outcome.best_hit3
                            );
                        }
                        Err(e) => {
                            state.status = format!("autotune failed: {}", e);
                        }
                    }
                    state.discard_armed = false;
                }
                ConfigKey::Esc | ConfigKey::Char('q') => {
                    if state.dirty && !state.discard_armed {
                        state.status =
                            "unsaved changes: press q again to discard, or s to save".to_string();
                        state.discard_armed = true;
                    } else {
                        return Ok(false);
                    }
                }
                ConfigKey::CtrlC => {
                    return Ok(false);
                }
                _ => {}
            },
            ConfigTuiMode::Edit { key, buffer } => match key_event {
                ConfigKey::Esc => {
                    state.mode = ConfigTuiMode::Navigate;
                    state.status = "edit cancelled".to_string();
                }
                ConfigKey::Enter => match config_set_value(working, key, buffer) {
                    Ok(_) => {
                        state.dirty = true;
                        state.status = format!(
                            "updated {} = {}",
                            key,
                            config_value_string(working, key).unwrap_or_default()
                        );
                        state.mode = ConfigTuiMode::Navigate;
                    }
                    Err(e) => {
                        state.status = e;
                    }
                },
                ConfigKey::Backspace => {
                    buffer.pop();
                }
                ConfigKey::Char(c) => {
                    buffer.push(c);
                }
                ConfigKey::CtrlC => return Ok(false),
                _ => {}
            },
        }
    }
}

fn select_config_menu_key(cfg: &ConfigValues) -> Result<Option<String>, String> {
    let mut rows: Vec<(String, String, String)> = vec![
        (
            "@save".to_string(),
            "Save and exit".to_string(),
            "Write config to disk".to_string(),
        ),
        (
            "@autotune".to_string(),
            "Autotune ranking".to_string(),
            "Use selection history to tune weights".to_string(),
        ),
        (
            "@discard".to_string(),
            "Discard and exit".to_string(),
            "Exit without saving".to_string(),
        ),
    ];
    for (key, hint) in config_rows() {
        rows.push((
            key.to_string(),
            config_value_string(cfg, key).unwrap_or_default(),
            hint.to_string(),
        ));
    }

    if tty_ui_available() && command_exists("fzf") {
        let mut payload = String::new();
        for (key, value, hint) in &rows {
            payload.push_str(&format!("{}\t{}\t{}\n", key, value, hint));
        }
        let mut cmd = Command::new("fzf");
        cmd.arg("--height=80%")
            .arg("--layout=reverse")
            .arg("--border")
            .arg("--delimiter=\t")
            .arg("--with-nth=1,2,3")
            .arg("--prompt")
            .arg("cypress config> ")
            .arg("--header")
            .arg("key | value | description (Enter: select)")
            .arg("--no-sort")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("failed launching config picker: {}", e))?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(payload.as_bytes())
                .map_err(|e| format!("failed writing config picker input: {}", e))?;
        }
        let out = child
            .wait_with_output()
            .map_err(|e| format!("failed waiting for config picker: {}", e))?;
        if !out.status.success() {
            return Ok(None);
        }
        let line = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if line.is_empty() {
            return Ok(None);
        }
        let key = line.split('\t').next().unwrap_or("").trim().to_string();
        if key.is_empty() {
            return Ok(None);
        }
        return Ok(Some(key));
    }

    for (idx, (_key, value, hint)) in rows.iter().enumerate() {
        println!("{:>2}. {:<26} {:<24} {}", idx + 1, rows[idx].0, value, hint);
    }
    let raw = prompt_line("select item number (empty to cancel): ")?;
    if raw.trim().is_empty() {
        return Ok(None);
    }
    let idx = raw
        .trim()
        .parse::<usize>()
        .map_err(|_| "invalid index".to_string())?;
    if idx == 0 || idx > rows.len() {
        return Err("index out of range".to_string());
    }
    Ok(Some(rows[idx - 1].0.clone()))
}

fn select_enum_option(key: &str, current: &str) -> Result<Option<String>, String> {
    let options = match config_enum_options(key) {
        Some(v) => v,
        None => return Ok(None),
    };
    if tty_ui_available() && command_exists("fzf") {
        let mut payload = String::new();
        for option in &options {
            payload.push_str(option);
            payload.push('\n');
        }
        let mut cmd = Command::new("fzf");
        cmd.arg("--height=40%")
            .arg("--layout=reverse")
            .arg("--border")
            .arg("--prompt")
            .arg(format!("{} (current: {})> ", key, current))
            .arg("--header")
            .arg("choose value")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("failed launching enum selector: {}", e))?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(payload.as_bytes())
                .map_err(|e| format!("failed writing enum selector input: {}", e))?;
        }
        let out = child
            .wait_with_output()
            .map_err(|e| format!("failed waiting for enum selector: {}", e))?;
        if !out.status.success() {
            return Ok(None);
        }
        let value = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if value.is_empty() {
            return Ok(None);
        }
        return Ok(Some(value));
    }
    println!("{} options: {}", key, options.join(", "));
    let raw = prompt_line(&format!("new value [{}]: ", current))?;
    if raw.trim().is_empty() {
        return Ok(None);
    }
    Ok(Some(raw.trim().to_string()))
}

fn edit_config_key_interactive(cfg: &mut ConfigValues, key: &str) -> Result<(), String> {
    let current = config_value_string(cfg, key).unwrap_or_default();
    if let Some(chosen) = select_enum_option(key, &current)? {
        config_set_value(cfg, key, &chosen)?;
        return Ok(());
    }
    println!("editing {} (current='{}')", key, current);
    println!("tip: enter ':empty' to clear string values, empty input keeps current value");
    let raw = prompt_line("new value: ")?;
    let next = if raw.trim().is_empty() {
        current
    } else if raw.trim() == ":empty" {
        String::new()
    } else {
        raw.trim().to_string()
    };
    config_set_value(cfg, key, &next)
}

fn run_config_edit_legacy(cwd: &Path) -> Result<(), String> {
    let cfg_path = config_path(cwd);
    let mut working = ConfigValues::from_map(load_config_values(&cfg_path));
    loop {
        let selected = select_config_menu_key(&working)?;
        let Some(key) = selected else {
            println!("config edit cancelled (no changes saved)");
            return Ok(());
        };
        match key.as_str() {
            "@save" => {
                write_config_file(&cfg_path, &working)?;
                println!("config saved: {}", cfg_path.display());
                return Ok(());
            }
            "@discard" => {
                println!("config changes discarded");
                return Ok(());
            }
            "@autotune" => {
                let dbp = db_path(cwd);
                ensure_db_schema(&dbp)?;
                let conn = open_db_rw(&dbp)?;
                let outcome = autotune_recommendation(&conn, &working, 320, 40, false)?;
                working = outcome.cfg;
                println!(
                    "autotune: examples={} baseline_mrr={:.4} best_mrr={:.4} candidates={}",
                    outcome.examples_used,
                    outcome.baseline_mrr,
                    outcome.best_mrr,
                    outcome.candidates_tested
                );
                println!(
                    "autotune: baseline_hit1={:.4} best_hit1={:.4} baseline_hit3={:.4} best_hit3={:.4}",
                    outcome.baseline_hit1,
                    outcome.best_hit1,
                    outcome.baseline_hit3,
                    outcome.best_hit3
                );
                if !outcome.used_history {
                    println!("autotune: used heuristic initialization (not enough history)");
                }
            }
            _ => {
                edit_config_key_interactive(&mut working, &key)?;
                if let Some(v) = config_value_string(&working, &key) {
                    println!("updated {} = {}", key, v);
                }
            }
        }
    }
}

fn run_config_edit(cwd: &Path) -> Result<(), String> {
    let cfg_path = config_path(cwd);
    let mut working = ConfigValues::from_map(load_config_values(&cfg_path));
    if tty_ui_available() {
        match run_config_tui(cwd, &cfg_path, &mut working) {
            Ok(saved) => {
                if saved {
                    println!("config saved: {}", cfg_path.display());
                } else {
                    println!("config changes discarded");
                }
                return Ok(());
            }
            Err(e) => {
                eprintln!(
                    "warning: full-screen config unavailable ({}); falling back to legacy editor",
                    e
                );
            }
        }
    }
    run_config_edit_legacy(cwd)
}

fn run_config_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: cypress config [edit|show|set <key> <value>|autotune] [options]");
        println!("examples:");
        println!("  cypress config                  # interactive editor");
        println!("  keybinds in editor: ↑/↓ move, Enter edit, ←/→ cycle enums, a autotune, s save, q discard");
        println!("  cypress config show             # print current values");
        println!("  cypress config set graph_seed_limit 12");
        println!("  cypress config autotune --dry-run --deep --max-events 500");
        return;
    }
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let action = args
        .first()
        .map(|v| v.to_string_lossy().to_string())
        .unwrap_or_else(|| "edit".to_string());
    match action.as_str() {
        "edit" => {
            run_config_edit(&cwd).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
        }
        "show" => {
            let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
            print_config_values(&cfg);
        }
        "set" => {
            if args.len() < 3 {
                eprintln!("error: usage: cypress config set <key> <value>");
                process::exit(2);
            }
            let key = args[1].to_string_lossy().to_string();
            let value = args[2..]
                .iter()
                .map(|v| v.to_string_lossy().to_string())
                .collect::<Vec<_>>()
                .join(" ");
            let cfg_path = config_path(&cwd);
            let mut cfg = ConfigValues::from_map(load_config_values(&cfg_path));
            config_set_value(&mut cfg, &key, &value).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(2);
            });
            write_config_file(&cfg_path, &cfg).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
            println!(
                "config updated: {}={}",
                key,
                config_value_string(&cfg, &key).unwrap_or_default()
            );
        }
        "autotune" => {
            run_autotune_cmd(&args[1..]);
        }
        other => {
            eprintln!("error: unknown config action '{}'", other);
            process::exit(2);
        }
    }
}

fn event_path_to_project(raw: &str, known_projects: &HashSet<String>) -> Option<String> {
    let mut cur = normalize_path(raw);
    loop {
        let candidate = cur.to_string_lossy().to_string();
        if known_projects.contains(&candidate) {
            return Some(candidate);
        }
        if !cur.pop() {
            break;
        }
    }
    None
}

fn load_autotune_examples(
    conn: &Connection,
    max_events: usize,
    known_projects: &HashSet<String>,
) -> Result<Vec<AutotuneExample>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT query, path, COUNT(*) AS c, MAX(selected_at) AS last_seen
FROM selection_events
WHERE length(trim(query)) > 0 AND length(trim(path)) > 0
GROUP BY query, path
ORDER BY last_seen DESC
LIMIT ?1
"#,
        )
        .map_err(|e| format!("failed preparing autotune event query: {}", e))?;
    let rows = stmt
        .query_map(params![max_events as i64], |row| {
            let query: String = row.get(0)?;
            let path: String = row.get(1)?;
            let count: i64 = row.get(2)?;
            Ok((query, path, count))
        })
        .map_err(|e| format!("failed querying autotune events: {}", e))?;
    let mut out = Vec::new();
    for row in rows {
        let (query, path, count) =
            row.map_err(|e| format!("failed reading autotune event row: {}", e))?;
        let Some(project_path) = event_path_to_project(&path, known_projects) else {
            continue;
        };
        out.push(AutotuneExample {
            query,
            path: project_path,
            weight: (count.max(1) as f64).sqrt(),
        });
    }
    Ok(out)
}

fn evaluate_candidate_mrr(
    conn: &Connection,
    cfg: &ConfigValues,
    examples: &[AutotuneExample],
    limit: usize,
) -> Result<(f64, f64, f64), String> {
    if examples.is_empty() {
        return Ok((0.0, 0.0, 0.0));
    }
    let mut grouped: HashMap<String, Vec<&AutotuneExample>> = HashMap::new();
    for ex in examples {
        grouped.entry(ex.query.clone()).or_default().push(ex);
    }
    let mut ranks_by_query: HashMap<String, HashMap<String, usize>> = HashMap::new();
    for query in grouped.keys() {
        let rows = rank_projects_native(conn, cfg, query, limit)?;
        let mut rank_map: HashMap<String, usize> = HashMap::new();
        for (idx, row) in rows.iter().enumerate() {
            rank_map.insert(row.path.clone(), idx + 1);
        }
        ranks_by_query.insert(query.clone(), rank_map);
    }
    let mut total_weight = 0.0f64;
    let mut mrr_sum = 0.0f64;
    let mut hit1_sum = 0.0f64;
    let mut hit3_sum = 0.0f64;
    for ex in examples {
        total_weight += ex.weight;
        let rank = ranks_by_query
            .get(&ex.query)
            .and_then(|m| m.get(&ex.path))
            .copied();
        if let Some(r) = rank {
            mrr_sum += ex.weight * (1.0 / r as f64);
            if r == 1 {
                hit1_sum += ex.weight;
            }
            if r <= 3 {
                hit3_sum += ex.weight;
            }
        }
    }
    if total_weight <= 0.0 {
        return Ok((0.0, 0.0, 0.0));
    }
    Ok((
        mrr_sum / total_weight,
        hit1_sum / total_weight,
        hit3_sum / total_weight,
    ))
}

fn unique_i64(values: Vec<i64>) -> Vec<i64> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for v in values {
        if seen.insert(v) {
            out.push(v);
        }
    }
    out
}

fn unique_f64(values: Vec<f64>) -> Vec<f64> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for v in values {
        let norm = (v * 10000.0).round() / 10000.0;
        let key = format!("{:.4}", norm);
        if seen.insert(key) {
            out.push(norm);
        }
    }
    out
}

fn normalize3(a: f64, b: f64, c: f64) -> (f64, f64, f64) {
    let av = a.max(0.0);
    let bv = b.max(0.0);
    let cv = c.max(0.0);
    let sum = av + bv + cv;
    if sum <= 0.0 {
        return (0.66, 0.24, 0.10);
    }
    (av / sum, bv / sum, cv / sum)
}

fn normalize5(vals: [f64; 5]) -> [f64; 5] {
    let mut v = vals;
    for item in &mut v {
        *item = item.max(0.0);
    }
    let sum: f64 = v.iter().sum();
    if sum <= 0.0 {
        return [0.58, 0.14, 0.10, 0.10, 0.08];
    }
    [v[0] / sum, v[1] / sum, v[2] / sum, v[3] / sum, v[4] / sum]
}

fn autotune_key(cfg: &ConfigValues) -> String {
    format!(
        "{:.4}|{:.4}|{:.4}|{:.4}|{:.4}|{:.4}|{:.4}|{:.4}|{:.4}|{:.4}|{}|{}|{}|{}",
        cfg.rank_chunk_semantic_weight,
        cfg.rank_chunk_lexical_weight,
        cfg.rank_chunk_graph_weight,
        cfg.rank_project_content_weight,
        cfg.rank_project_semantic_weight,
        cfg.rank_project_path_weight,
        cfg.rank_project_graph_weight,
        cfg.rank_project_frecency_weight,
        cfg.graph_same_project_high,
        cfg.graph_same_project_low,
        cfg.graph_seed_limit,
        cfg.graph_neighbor_limit,
        cfg.lexical_candidates,
        cfg.vector_candidates
    )
}

fn maybe_promote_candidate(
    conn: &Connection,
    examples: &[AutotuneExample],
    limit: usize,
    candidate: ConfigValues,
    tested: &mut usize,
    seen: &mut HashSet<String>,
    best_cfg: &mut ConfigValues,
    best_metrics: &mut (f64, f64, f64),
) -> Result<(), String> {
    let key = autotune_key(&candidate);
    if !seen.insert(key) {
        return Ok(());
    }
    let metrics = evaluate_candidate_mrr(conn, &candidate, examples, limit)?;
    *tested += 1;
    if metrics.0 > best_metrics.0 + 1e-9
        || ((metrics.0 - best_metrics.0).abs() <= 1e-9 && metrics.1 > best_metrics.1 + 1e-9)
    {
        *best_cfg = candidate;
        *best_metrics = metrics;
    }
    Ok(())
}

fn autotune_recommendation(
    conn: &Connection,
    base_cfg: &ConfigValues,
    max_events: usize,
    limit: usize,
    deep: bool,
) -> Result<AutotuneOutcome, String> {
    let known_projects: HashSet<String> = list_project_paths(conn)?.into_iter().collect();
    let mut examples = load_autotune_examples(conn, max_events, &known_projects)?;
    if examples.len() > 120 {
        examples.truncate(120);
    }
    let used_history = examples.len() >= 6;
    let baseline = evaluate_candidate_mrr(conn, base_cfg, &examples, limit)?;

    if !used_history {
        let mut heuristic = base_cfg.clone();
        let project_count = known_projects.len() as i64;
        heuristic.graph_seed_limit = if project_count < 80 { 8 } else { 12 };
        heuristic.graph_neighbor_limit = (project_count / 2).clamp(80, 180);
        heuristic.rank_chunk_graph_weight = if project_count > 200 { 0.12 } else { 0.10 };
        heuristic.rank_project_graph_weight = if project_count > 200 { 0.12 } else { 0.10 };
        let (cs, cl, cg) = normalize3(
            heuristic.rank_chunk_semantic_weight,
            heuristic.rank_chunk_lexical_weight,
            heuristic.rank_chunk_graph_weight,
        );
        heuristic.rank_chunk_semantic_weight = cs;
        heuristic.rank_chunk_lexical_weight = cl;
        heuristic.rank_chunk_graph_weight = cg;
        let p = normalize5([
            heuristic.rank_project_content_weight,
            heuristic.rank_project_semantic_weight,
            heuristic.rank_project_path_weight,
            heuristic.rank_project_graph_weight,
            heuristic.rank_project_frecency_weight,
        ]);
        heuristic.rank_project_content_weight = p[0];
        heuristic.rank_project_semantic_weight = p[1];
        heuristic.rank_project_path_weight = p[2];
        heuristic.rank_project_graph_weight = p[3];
        heuristic.rank_project_frecency_weight = p[4];
        let heuristic_metrics = evaluate_candidate_mrr(conn, &heuristic, &examples, limit)?;
        return Ok(AutotuneOutcome {
            cfg: heuristic,
            examples_used: examples.len(),
            baseline_mrr: baseline.0,
            baseline_hit1: baseline.1,
            baseline_hit3: baseline.2,
            best_mrr: heuristic_metrics.0,
            best_hit1: heuristic_metrics.1,
            best_hit3: heuristic_metrics.2,
            candidates_tested: 1,
            used_history: false,
        });
    }

    let mut best_cfg = base_cfg.clone();
    let mut best_metrics = baseline;
    let mut tested = 0usize;
    let mut seen = HashSet::new();
    seen.insert(autotune_key(&best_cfg));
    let rounds = if deep { 2 } else { 1 };

    for _round in 0..rounds {
        let chunk_graph_vals = unique_f64(vec![
            (best_cfg.rank_chunk_graph_weight * 0.75).clamp(0.02, 0.35),
            best_cfg.rank_chunk_graph_weight.clamp(0.02, 0.35),
            (best_cfg.rank_chunk_graph_weight * 1.25).clamp(0.02, 0.35),
        ]);
        let project_graph_vals = unique_f64(vec![
            (best_cfg.rank_project_graph_weight * 0.75).clamp(0.02, 0.45),
            best_cfg.rank_project_graph_weight.clamp(0.02, 0.45),
            (best_cfg.rank_project_graph_weight * 1.25).clamp(0.02, 0.45),
        ]);
        let seed_vals = unique_i64(vec![
            (best_cfg.graph_seed_limit - 2).clamp(2, 64),
            best_cfg.graph_seed_limit.clamp(2, 64),
            (best_cfg.graph_seed_limit + 2).clamp(2, 64),
        ]);
        let neighbor_vals = unique_i64(vec![
            (best_cfg.graph_neighbor_limit - 20).clamp(8, 500),
            best_cfg.graph_neighbor_limit.clamp(8, 500),
            (best_cfg.graph_neighbor_limit + 20).clamp(8, 500),
        ]);

        for chunk_graph in &chunk_graph_vals {
            for project_graph in &project_graph_vals {
                for seed in &seed_vals {
                    for neighbor in &neighbor_vals {
                        let mut candidate = best_cfg.clone();
                        candidate.rank_chunk_graph_weight = *chunk_graph;
                        candidate.rank_project_graph_weight = *project_graph;
                        candidate.graph_seed_limit = *seed;
                        candidate.graph_neighbor_limit = *neighbor;
                        let (cs, cl, cg) = normalize3(
                            candidate.rank_chunk_semantic_weight,
                            candidate.rank_chunk_lexical_weight,
                            candidate.rank_chunk_graph_weight,
                        );
                        candidate.rank_chunk_semantic_weight = cs;
                        candidate.rank_chunk_lexical_weight = cl;
                        candidate.rank_chunk_graph_weight = cg;
                        let p = normalize5([
                            candidate.rank_project_content_weight,
                            candidate.rank_project_semantic_weight,
                            candidate.rank_project_path_weight,
                            candidate.rank_project_graph_weight,
                            candidate.rank_project_frecency_weight,
                        ]);
                        candidate.rank_project_content_weight = p[0];
                        candidate.rank_project_semantic_weight = p[1];
                        candidate.rank_project_path_weight = p[2];
                        candidate.rank_project_graph_weight = p[3];
                        candidate.rank_project_frecency_weight = p[4];
                        maybe_promote_candidate(
                            conn,
                            &examples,
                            limit,
                            candidate,
                            &mut tested,
                            &mut seen,
                            &mut best_cfg,
                            &mut best_metrics,
                        )?;
                    }
                }
            }
        }

        let same_high_vals = unique_f64(vec![
            (best_cfg.graph_same_project_high - 0.06).clamp(0.20, 0.98),
            best_cfg.graph_same_project_high.clamp(0.20, 0.98),
            (best_cfg.graph_same_project_high + 0.06).clamp(0.20, 0.98),
        ]);
        let same_low_vals = unique_f64(vec![
            (best_cfg.graph_same_project_low - 0.06).clamp(0.10, 0.95),
            best_cfg.graph_same_project_low.clamp(0.10, 0.95),
            (best_cfg.graph_same_project_low + 0.06).clamp(0.10, 0.95),
        ]);
        let related_scale_vals = unique_f64(vec![
            (best_cfg.graph_related_scale - 0.10).clamp(0.10, 1.40),
            best_cfg.graph_related_scale.clamp(0.10, 1.40),
            (best_cfg.graph_related_scale + 0.10).clamp(0.10, 1.40),
        ]);
        let related_base_vals = unique_f64(vec![
            (best_cfg.graph_related_base - 0.05).clamp(0.02, 0.50),
            best_cfg.graph_related_base.clamp(0.02, 0.50),
            (best_cfg.graph_related_base + 0.05).clamp(0.02, 0.50),
        ]);
        let related_cap_vals = unique_f64(vec![
            (best_cfg.graph_related_cap - 0.08).clamp(0.20, 0.99),
            best_cfg.graph_related_cap.clamp(0.20, 0.99),
            (best_cfg.graph_related_cap + 0.08).clamp(0.20, 0.99),
        ]);
        for high in &same_high_vals {
            for low in &same_low_vals {
                if low > high {
                    continue;
                }
                for scale in &related_scale_vals {
                    for base in &related_base_vals {
                        for cap in &related_cap_vals {
                            let mut candidate = best_cfg.clone();
                            candidate.graph_same_project_high = *high;
                            candidate.graph_same_project_low = *low;
                            candidate.graph_related_scale = *scale;
                            candidate.graph_related_base = *base;
                            candidate.graph_related_cap = (*cap).max(*base);
                            maybe_promote_candidate(
                                conn,
                                &examples,
                                limit,
                                candidate,
                                &mut tested,
                                &mut seen,
                                &mut best_cfg,
                                &mut best_metrics,
                            )?;
                        }
                    }
                }
            }
        }

        let chunk_templates = vec![
            normalize3(
                best_cfg.rank_chunk_semantic_weight,
                best_cfg.rank_chunk_lexical_weight,
                best_cfg.rank_chunk_graph_weight,
            ),
            (0.72, 0.18, 0.10),
            (0.62, 0.23, 0.15),
            (0.54, 0.36, 0.10),
            (0.56, 0.20, 0.24),
        ];
        for (s, l, g) in chunk_templates {
            let mut candidate = best_cfg.clone();
            let (ns, nl, ng) = normalize3(s, l, g);
            candidate.rank_chunk_semantic_weight = ns;
            candidate.rank_chunk_lexical_weight = nl;
            candidate.rank_chunk_graph_weight = ng;
            maybe_promote_candidate(
                conn,
                &examples,
                limit,
                candidate,
                &mut tested,
                &mut seen,
                &mut best_cfg,
                &mut best_metrics,
            )?;
        }

        let project_templates = vec![
            normalize5([
                best_cfg.rank_project_content_weight,
                best_cfg.rank_project_semantic_weight,
                best_cfg.rank_project_path_weight,
                best_cfg.rank_project_graph_weight,
                best_cfg.rank_project_frecency_weight,
            ]),
            [0.62, 0.12, 0.08, 0.12, 0.06],
            [0.50, 0.10, 0.08, 0.24, 0.08],
            [0.52, 0.14, 0.18, 0.10, 0.06],
            [0.48, 0.12, 0.08, 0.10, 0.22],
        ];
        for template in project_templates {
            let mut candidate = best_cfg.clone();
            let p = normalize5(template);
            candidate.rank_project_content_weight = p[0];
            candidate.rank_project_semantic_weight = p[1];
            candidate.rank_project_path_weight = p[2];
            candidate.rank_project_graph_weight = p[3];
            candidate.rank_project_frecency_weight = p[4];
            maybe_promote_candidate(
                conn,
                &examples,
                limit,
                candidate,
                &mut tested,
                &mut seen,
                &mut best_cfg,
                &mut best_metrics,
            )?;
        }

        let lexical_vals = unique_i64(vec![
            (best_cfg.lexical_candidates - 40).clamp(20, 5000),
            best_cfg.lexical_candidates.clamp(20, 5000),
            (best_cfg.lexical_candidates + 40).clamp(20, 5000),
        ]);
        let vector_vals = unique_i64(vec![
            (best_cfg.vector_candidates - 40).clamp(20, 5000),
            best_cfg.vector_candidates.clamp(20, 5000),
            (best_cfg.vector_candidates + 40).clamp(20, 5000),
        ]);
        for lexical in &lexical_vals {
            for vector in &vector_vals {
                let mut candidate = best_cfg.clone();
                candidate.lexical_candidates = *lexical;
                candidate.vector_candidates = *vector;
                maybe_promote_candidate(
                    conn,
                    &examples,
                    limit,
                    candidate,
                    &mut tested,
                    &mut seen,
                    &mut best_cfg,
                    &mut best_metrics,
                )?;
            }
        }
    }

    Ok(AutotuneOutcome {
        cfg: best_cfg,
        examples_used: examples.len(),
        baseline_mrr: baseline.0,
        baseline_hit1: baseline.1,
        baseline_hit3: baseline.2,
        best_mrr: best_metrics.0,
        best_hit1: best_metrics.1,
        best_hit3: best_metrics.2,
        candidates_tested: tested.max(1),
        used_history: true,
    })
}

fn autotune_snapshot(cfg: &ConfigValues) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for (key, _) in config_rows() {
        if let Some(value) = config_value_string(cfg, key) {
            out.insert(key.to_string(), value);
        }
    }
    out
}

fn write_autotune_report(
    cwd: &Path,
    opts: &AutotuneOptions,
    baseline_cfg: &ConfigValues,
    outcome: &AutotuneOutcome,
    applied: bool,
    cfg_path: &Path,
) -> Result<(PathBuf, PathBuf), String> {
    let report_dir = data_dir(cwd).join("autotune");
    fs::create_dir_all(&report_dir)
        .map_err(|e| format!("failed creating autotune report dir: {}", e))?;

    let ts = now_ts();
    let ts_ms = (ts * 1000.0).round() as i64;
    let run_json = report_dir.join(format!("report-{}.json", ts_ms));
    let run_md = report_dir.join(format!("report-{}.md", ts_ms));
    let latest_json = report_dir.join("latest.json");
    let latest_md = report_dir.join("latest.md");

    let before = autotune_snapshot(baseline_cfg);
    let after = autotune_snapshot(&outcome.cfg);
    let mut changed: Vec<Value> = Vec::new();
    for (key, after_value) in &after {
        let before_value = before.get(key).cloned().unwrap_or_default();
        if before_value != *after_value {
            changed.push(serde_json::json!({
                "key": key,
                "before": before_value,
                "after": after_value
            }));
        }
    }

    let report = serde_json::json!({
        "schema": "autotune-report-v1",
        "generated_at": ts,
        "generated_at_ms": ts_ms,
        "applied": applied,
        "config_path": cfg_path,
        "options": {
            "dry_run": opts.dry_run,
            "deep": opts.deep,
            "max_events": opts.max_events,
            "limit": opts.limit
        },
        "summary": {
            "examples_used": outcome.examples_used,
            "used_history": outcome.used_history,
            "candidates_tested": outcome.candidates_tested,
            "baseline_mrr": outcome.baseline_mrr,
            "best_mrr": outcome.best_mrr,
            "delta_mrr": outcome.best_mrr - outcome.baseline_mrr,
            "baseline_hit1": outcome.baseline_hit1,
            "best_hit1": outcome.best_hit1,
            "delta_hit1": outcome.best_hit1 - outcome.baseline_hit1,
            "baseline_hit3": outcome.baseline_hit3,
            "best_hit3": outcome.best_hit3,
            "delta_hit3": outcome.best_hit3 - outcome.baseline_hit3
        },
        "baseline_config": before,
        "recommended_config": after,
        "changed": changed
    });
    let report_raw = serde_json::to_string_pretty(&report)
        .map_err(|e| format!("failed to serialize autotune report: {}", e))?;
    fs::write(&run_json, format!("{}\n", report_raw)).map_err(|e| {
        format!(
            "failed writing autotune report '{}': {}",
            run_json.display(),
            e
        )
    })?;
    fs::write(&latest_json, format!("{}\n", report_raw)).map_err(|e| {
        format!(
            "failed writing autotune latest report '{}': {}",
            latest_json.display(),
            e
        )
    })?;

    let mut md_lines: Vec<String> = Vec::new();
    md_lines.push("# Cypress Autotune Report".to_string());
    md_lines.push(String::new());
    md_lines.push(format!("- generated_at: {:.3}", ts));
    md_lines.push(format!("- applied: {}", yes_no(applied)));
    md_lines.push(format!("- config_path: `{}`", cfg_path.display()));
    md_lines.push(format!(
        "- options: dry_run={} deep={} max_events={} limit={}",
        yes_no(opts.dry_run),
        yes_no(opts.deep),
        opts.max_events,
        opts.limit
    ));
    md_lines.push(String::new());
    md_lines.push("## Metrics".to_string());
    md_lines.push(String::new());
    md_lines.push(format!(
        "- examples_used: {} (history={})",
        outcome.examples_used,
        yes_no(outcome.used_history)
    ));
    md_lines.push(format!(
        "- candidates_tested: {}",
        outcome.candidates_tested
    ));
    md_lines.push(format!(
        "- mrr: {:.4} -> {:.4} (delta {:+.4})",
        outcome.baseline_mrr,
        outcome.best_mrr,
        outcome.best_mrr - outcome.baseline_mrr
    ));
    md_lines.push(format!(
        "- hit@1: {:.4} -> {:.4} (delta {:+.4})",
        outcome.baseline_hit1,
        outcome.best_hit1,
        outcome.best_hit1 - outcome.baseline_hit1
    ));
    md_lines.push(format!(
        "- hit@3: {:.4} -> {:.4} (delta {:+.4})",
        outcome.baseline_hit3,
        outcome.best_hit3,
        outcome.best_hit3 - outcome.baseline_hit3
    ));
    md_lines.push(String::new());
    md_lines.push("## Changed Settings".to_string());
    md_lines.push(String::new());
    if changed.is_empty() {
        md_lines.push("- none".to_string());
    } else {
        for row in &changed {
            let key = row.get("key").and_then(|v| v.as_str()).unwrap_or_default();
            let before_v = row
                .get("before")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let after_v = row
                .get("after")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            md_lines.push(format!("- `{}`: `{}` -> `{}`", key, before_v, after_v));
        }
    }
    let md_raw = format!("{}\n", md_lines.join("\n"));
    fs::write(&run_md, &md_raw).map_err(|e| {
        format!(
            "failed writing autotune markdown '{}': {}",
            run_md.display(),
            e
        )
    })?;
    fs::write(&latest_md, &md_raw).map_err(|e| {
        format!(
            "failed writing autotune latest markdown '{}': {}",
            latest_md.display(),
            e
        )
    })?;
    Ok((latest_json, latest_md))
}

fn run_autotune_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: cypress autotune [--dry-run] [--deep] [--max-events <n>] [--limit <n>]");
        println!("notes:");
        println!("  - tunes ranking settings from historical selection events");
        println!("  - use --deep for a larger candidate sweep (slower)");
        println!("  - writes tuned config unless --dry-run is used");
        return;
    }
    let opts = parse_autotune_options(args);
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let dbp = db_path(&cwd);
    ensure_db_schema(&dbp).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    let conn = open_db_rw(&dbp).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    let cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    let outcome = autotune_recommendation(&conn, &cfg, opts.max_events, opts.limit, opts.deep)
        .unwrap_or_else(|e| {
            eprintln!("error: autotune failed: {}", e);
            process::exit(1);
        });

    println!(
        "autotune: examples={} baseline_mrr={:.4} best_mrr={:.4} candidates={}",
        outcome.examples_used, outcome.baseline_mrr, outcome.best_mrr, outcome.candidates_tested
    );
    println!(
        "autotune: baseline_hit1={:.4} best_hit1={:.4} baseline_hit3={:.4} best_hit3={:.4}",
        outcome.baseline_hit1, outcome.best_hit1, outcome.baseline_hit3, outcome.best_hit3
    );
    if !outcome.used_history {
        println!("autotune: not enough history; applied heuristic defaults");
    }
    println!(
        "autotune: rank_chunk_graph_weight={:.4} rank_project_graph_weight={:.4} graph_seed_limit={} graph_neighbor_limit={}",
        outcome.cfg.rank_chunk_graph_weight,
        outcome.cfg.rank_project_graph_weight,
        outcome.cfg.graph_seed_limit,
        outcome.cfg.graph_neighbor_limit
    );

    if opts.dry_run {
        println!("autotune: dry-run enabled; config not written");
    } else {
        write_config_file(&cfg_path, &outcome.cfg).unwrap_or_else(|e| {
            eprintln!("error: failed writing tuned config: {}", e);
            process::exit(1);
        });
        println!("autotune: config updated -> {}", cfg_path.display());
    }
    let (report_json, report_md) =
        write_autotune_report(&cwd, &opts, &cfg, &outcome, !opts.dry_run, &cfg_path)
            .unwrap_or_else(|e| {
                eprintln!("error: failed writing autotune report: {}", e);
                process::exit(1);
            });
    println!("autotune: report_json={}", report_json.display());
    println!("autotune: report_md={}", report_md.display());
}

fn run_graph_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: cypress graph [doctor|status|start|stop|provision|view|open|neighbors|lineage] [--wait <seconds>] [--host <addr>] [--port <n>] [--path <project-or-child-path>] [--limit <n>] [--threshold <0..1>] [--depth <1..3>] [--no-system-install] [--no-download] [--force] [--from <falkordb-binary-path>]");
        println!("quick examples:");
        println!("  cypress graph open");
        println!(
            "  cypress graph neighbors --path ~/c-projects/202601-storage-cost-analysis --limit 12"
        );
        println!(
            "  cypress graph lineage --path ~/c-projects/202601-storage-cost-analysis --depth 2 --threshold 0.60"
        );
        println!("env:");
        println!("  CYPRESS_FALKOR_URL        (default: falkor://127.0.0.1:6379)");
        println!("  CYPRESS_FALKOR_GRAPH      (default: cypress)");
        println!(
            "  CYPRESS_FALKOR_RUNTIME_MODE (managed-local|docker|manual; default: managed-local)"
        );
        println!("  CYPRESS_FALKOR_RUNTIME_CMD  (optional command for managed-local mode)");
        println!("  CYPRESS_FALKOR_DOCKER_CONTAINER (default: cypress-falkordb)");
        return;
    }

    let mut action = "status".to_string();
    let mut wait_seconds: u64 = 12;
    let mut allow_system_install = true;
    let mut allow_download = true;
    let mut force = false;
    let mut source_path: Option<PathBuf> = None;
    let mut view_host = "127.0.0.1".to_string();
    let mut view_port: u16 = 8780;
    let mut graph_focus: Option<String> = None;
    let mut graph_path: Option<String> = None;
    let mut graph_limit: usize = 80;
    let mut graph_threshold: f64 = 0.0;
    let mut graph_depth: usize = 1;

    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        match s.as_str() {
            "doctor" | "status" | "start" | "stop" | "provision" | "view" | "open"
            | "neighbors" | "lineage" | "text" | "ui" => {
                action = s;
            }
            "--no-system-install" => {
                allow_system_install = false;
            }
            "--no-download" => {
                allow_download = false;
            }
            "--force" => {
                force = true;
            }
            "--from" => {
                i += 1;
                let raw = arg_value(args, i, "--from");
                source_path = Some(normalize_path(&raw));
            }
            "--wait" => {
                i += 1;
                let value = arg_value(args, i, "--wait");
                wait_seconds = value.parse::<u64>().unwrap_or_else(|_| {
                    eprintln!("error: --wait must be an integer number of seconds");
                    process::exit(2);
                });
            }
            "--host" => {
                i += 1;
                view_host = arg_value(args, i, "--host");
            }
            "--focus" => {
                i += 1;
                let raw = arg_value(args, i, "--focus");
                let trimmed = raw.trim().to_string();
                if !trimmed.is_empty() {
                    graph_focus = Some(trimmed);
                }
            }
            "--path" => {
                i += 1;
                let raw = arg_value(args, i, "--path");
                let trimmed = raw.trim().to_string();
                if !trimmed.is_empty() {
                    graph_path = Some(trimmed);
                }
            }
            "--limit" => {
                i += 1;
                let value = arg_value(args, i, "--limit");
                graph_limit = value.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --limit must be an integer");
                    process::exit(2);
                });
            }
            "--threshold" => {
                i += 1;
                let value = arg_value(args, i, "--threshold");
                graph_threshold = value.parse::<f64>().unwrap_or_else(|_| {
                    eprintln!("error: --threshold must be a decimal between 0 and 1");
                    process::exit(2);
                });
            }
            "--depth" => {
                i += 1;
                let value = arg_value(args, i, "--depth");
                graph_depth = value.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --depth must be an integer (1..3)");
                    process::exit(2);
                });
            }
            "--port" => {
                i += 1;
                let value = arg_value(args, i, "--port");
                view_port = value.parse::<u16>().unwrap_or_else(|_| {
                    eprintln!("error: --port must be an integer between 1 and 65535");
                    process::exit(2);
                });
            }
            x if x.starts_with("--wait=") => {
                let value = x.trim_start_matches("--wait=");
                wait_seconds = value.parse::<u64>().unwrap_or_else(|_| {
                    eprintln!("error: --wait must be an integer number of seconds");
                    process::exit(2);
                });
            }
            x if x.starts_with("--host=") => {
                view_host = x.trim_start_matches("--host=").to_string();
            }
            x if x.starts_with("--focus=") => {
                let raw = x.trim_start_matches("--focus=").trim().to_string();
                if !raw.is_empty() {
                    graph_focus = Some(raw);
                }
            }
            x if x.starts_with("--path=") => {
                let raw = x.trim_start_matches("--path=").trim().to_string();
                if !raw.is_empty() {
                    graph_path = Some(raw);
                }
            }
            x if x.starts_with("--limit=") => {
                let value = x.trim_start_matches("--limit=");
                graph_limit = value.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --limit must be an integer");
                    process::exit(2);
                });
            }
            x if x.starts_with("--threshold=") => {
                let value = x.trim_start_matches("--threshold=");
                graph_threshold = value.parse::<f64>().unwrap_or_else(|_| {
                    eprintln!("error: --threshold must be a decimal between 0 and 1");
                    process::exit(2);
                });
            }
            x if x.starts_with("--depth=") => {
                let value = x.trim_start_matches("--depth=");
                graph_depth = value.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --depth must be an integer (1..3)");
                    process::exit(2);
                });
            }
            x if x.starts_with("--port=") => {
                let value = x.trim_start_matches("--port=");
                view_port = value.parse::<u16>().unwrap_or_else(|_| {
                    eprintln!("error: --port must be an integer between 1 and 65535");
                    process::exit(2);
                });
            }
            x if x == "--layout"
                || x.starts_with("--layout=")
                || x == "--renderer"
                || x.starts_with("--renderer=")
                || x == "--ui-backend"
                || x.starts_with("--ui-backend=") =>
            {
                eprintln!(
                    "error: terminal graph renderers were removed; use `cypress graph open` for visual graph and `cypress graph neighbors|lineage` for terminal output"
                );
                process::exit(2);
            }
            other => {
                if other.starts_with('-') {
                    eprintln!("error: unknown graph action/option '{}'", other);
                    process::exit(2);
                }
                if graph_path.is_none() {
                    graph_path = Some(other.to_string());
                } else {
                    eprintln!("error: unexpected extra argument '{}'", other);
                    process::exit(2);
                }
            }
        }
        i += 1;
    }

    match action.as_str() {
        "status" => {
            let _ = print_graph_status(false);
        }
        "doctor" => {
            let ok = print_graph_status(true);
            if !ok {
                process::exit(1);
            }
        }
        "start" => {
            start_falkordb(wait_seconds).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
            let _ = print_graph_status(false);
        }
        "stop" => {
            stop_falkordb(wait_seconds).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
            let _ = print_graph_status(false);
        }
        "provision" => {
            let runtime_cmd = provision_falkordb_managed_local(
                allow_system_install,
                allow_download,
                force,
                source_path.clone(),
            )
            .unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
            let cfg_path = persist_managed_local_runtime_config(&runtime_cmd).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
            println!("falkordb: provisioned managed-local runtime");
            println!("runtime_cmd: {}", runtime_cmd);
            println!("config: {}", cfg_path.display());
        }
        "view" => {
            serve_graph_viewer(&view_host, view_port).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
        }
        "open" => {
            run_graph_open_cmd(&view_host, view_port).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
        }
        "neighbors" => {
            run_graph_neighbors_cmd(
                graph_path.as_deref().or(graph_focus.as_deref()),
                graph_limit,
                graph_threshold,
            )
            .unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
        }
        "lineage" => {
            run_graph_lineage_cmd(
                graph_path.as_deref().or(graph_focus.as_deref()),
                graph_limit,
                graph_threshold,
                graph_depth,
            )
            .unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
        }
        "text" | "ui" => {
            eprintln!(
                "error: `cypress graph {}` is removed; use `cypress graph open` (browser UI) or `cypress graph neighbors|lineage` (terminal tables)",
                action
            );
            process::exit(2);
        }
        _ => unreachable!(),
    }
}

fn display_path_compact(path: &str) -> String {
    if let Ok(home) = env::var("HOME") {
        if path == home {
            return "~".to_string();
        }
        let prefix = format!("{}/", home);
        if let Some(rest) = path.strip_prefix(&prefix) {
            return format!("~/{}", rest);
        }
    }
    path.to_string()
}

fn open_url_in_default_browser(url: &str) -> Result<(), String> {
    if cfg!(target_os = "macos") && command_exists("open") {
        let status = Command::new("open")
            .arg(url)
            .status()
            .map_err(|e| format!("failed launching browser via `open`: {}", e))?;
        if status.success() {
            return Ok(());
        }
        return Err(format!("`open` exited with status {}", status));
    }
    if command_exists("xdg-open") {
        let status = Command::new("xdg-open")
            .arg(url)
            .status()
            .map_err(|e| format!("failed launching browser via `xdg-open`: {}", e))?;
        if status.success() {
            return Ok(());
        }
        return Err(format!("`xdg-open` exited with status {}", status));
    }
    Err("no supported browser opener found (tried `open` and `xdg-open`)".to_string())
}

fn graph_viewer_has_nodes(host: &str, port: u16) -> Option<bool> {
    let url = format!("http://{}:{}/graph/view/data?limit=1", host, port);
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(2))
        .build();
    let resp = agent.get(&url).call().ok()?;
    let body = resp.into_string().ok()?;
    let payload: Value = serde_json::from_str(&body).ok()?;
    let nodes = payload
        .get("nodes")
        .and_then(Value::as_array)
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    Some(nodes)
}

fn local_graph_has_nodes(cwd: &Path) -> bool {
    let dbp = db_path(cwd);
    let Ok(conn) = open_db_read_only(&dbp) else {
        return false;
    };
    conn.query_row("SELECT COUNT(*) FROM project_edges", [], |row| {
        row.get::<_, i64>(0)
    })
    .map(|count| count > 0)
    .unwrap_or(false)
}

fn ensure_graph_viewer_running(
    host: &str,
    port: u16,
    prefer_fresh_when_existing_empty: bool,
) -> Result<u16, String> {
    if api_health_host_port(host, port) {
        if prefer_fresh_when_existing_empty
            && matches!(graph_viewer_has_nodes(host, port), Some(false))
        {
            eprintln!(
                "graph open: existing viewer on port {} has no graph nodes; launching fresh viewer on a new port",
                port
            );
        } else {
            return Ok(port);
        }
    }
    let mut use_port = port;
    if find_free_port(host, use_port, 1).is_none() {
        let Some(free) = find_free_port(host, use_port.saturating_add(1), 200) else {
            return Err(format!(
                "port {} is busy and no free fallback port was found",
                use_port
            ));
        };
        eprintln!(
            "graph open: port {} busy; launching viewer on {}",
            use_port, free
        );
        use_port = free;
    }

    let exe =
        env::current_exe().map_err(|e| format!("failed resolving current executable: {}", e))?;
    let mut child = Command::new(exe)
        .arg("graph")
        .arg("view")
        .arg("--host")
        .arg(host)
        .arg("--port")
        .arg(use_port.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("failed launching graph viewer in background: {}", e))?;

    let deadline = Instant::now() + Duration::from_secs(8);
    while Instant::now() < deadline {
        if api_health_host_port(host, use_port) {
            return Ok(use_port);
        }
        if let Some(status) = child
            .try_wait()
            .map_err(|e| format!("failed checking graph viewer process: {}", e))?
        {
            return Err(format!("graph viewer exited early with status {}", status));
        }
        thread::sleep(Duration::from_millis(120));
    }
    Err(format!(
        "graph viewer did not become healthy at http://{}:{}/health within timeout",
        host, use_port
    ))
}

fn resolve_graph_target_project(
    conn: &Connection,
    raw_path: Option<&str>,
) -> Result<String, String> {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut candidate = raw_path.map(normalize_path).unwrap_or_else(|| cwd.clone());
    if candidate.is_file() {
        if let Some(parent) = candidate.parent() {
            candidate = parent.to_path_buf();
        }
    }
    let normalized = normalize_path(candidate.to_string_lossy().as_ref())
        .to_string_lossy()
        .to_string();
    if let Some(project_path) = resolve_project_for_target(conn, &normalized)? {
        return Ok(project_path);
    }
    let needle = path_basename(&normalized);
    let suggestions = suggest_project_paths(conn, &needle, 5)?;
    let mut msg = format!(
        "no indexed project found for '{}'; run `cypress index` or pass a tracked project path",
        display_path_compact(&normalized)
    );
    if !suggestions.is_empty() {
        msg.push_str("\nclosest indexed projects:");
        for s in suggestions {
            msg.push_str(&format!("\n- {}", display_path_compact(&s)));
        }
    }
    Err(msg)
}

fn run_graph_open_cmd(host: &str, port: u16) -> Result<(), String> {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let bound_port = ensure_graph_viewer_running(host, port, local_graph_has_nodes(&cwd))?;
    let url = format!("http://{}:{}/", host, bound_port);
    println!("graph viewer: {}", url);
    if let Err(err) = open_url_in_default_browser(&url) {
        eprintln!("warning: {}", err);
        eprintln!("hint: open this URL manually in your browser");
    }
    Ok(())
}

fn run_graph_neighbors_cmd(
    raw_path: Option<&str>,
    limit: usize,
    min_weight: f64,
) -> Result<(), String> {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let dbp = db_path(&cwd);
    ensure_db_schema(&dbp)?;
    let conn = open_db_read_only(&dbp)?;
    let project_path = resolve_graph_target_project(&conn, raw_path)?;

    let limit = limit.max(1).min(200);
    let min_weight = min_weight.clamp(0.0, 1.0);
    let rows = list_neighbors_by_path(&conn, &project_path, limit * 3)?;
    let mut filtered: Vec<(String, String, f64)> = rows
        .into_iter()
        .filter(|(_, _, w)| *w >= min_weight)
        .take(limit)
        .collect();
    filtered.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));

    println!("project: {}", display_path_compact(&project_path));
    println!(
        "neighbors: {} (threshold >= {:.2})",
        filtered.len(),
        min_weight
    );
    if filtered.is_empty() {
        println!("(no outgoing neighbors matched current threshold)");
        return Ok(());
    }
    println!(
        "{:<4} {:>7}  {:<18} {}",
        "rank", "weight", "relation", "target"
    );
    for (idx, (dst, kind, weight)) in filtered.iter().enumerate() {
        println!(
            "{:<4} {:>7.3}  {:<18} {}",
            idx + 1,
            weight,
            clipped(kind, 18),
            display_path_compact(dst)
        );
    }
    Ok(())
}

fn run_graph_lineage_cmd(
    raw_path: Option<&str>,
    limit: usize,
    min_weight: f64,
    depth: usize,
) -> Result<(), String> {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let dbp = db_path(&cwd);
    ensure_db_schema(&dbp)?;
    let conn = open_db_read_only(&dbp)?;
    let project_path = resolve_graph_target_project(&conn, raw_path)?;

    let limit = limit.max(1).min(200);
    let min_weight = min_weight.clamp(0.0, 1.0);
    let depth = depth.clamp(1, 3);
    let data = load_graph_text_data(&cwd, Some(&project_path), (limit * 6).max(40))?;
    let subgraph =
        build_focus_subgraph(&data, &project_path, (limit * 3).max(24), min_weight, depth);

    let mut incoming: Vec<(String, String, f64)> = Vec::new();
    let mut outgoing: Vec<(String, String, f64)> = Vec::new();
    let mut relays: Vec<(String, String, String, f64)> = Vec::new();
    for edge in &subgraph.edges {
        if edge.weight < min_weight {
            continue;
        }
        if edge.source == project_path {
            outgoing.push((edge.target.clone(), edge.kind.clone(), edge.weight));
        } else if edge.target == project_path {
            incoming.push((edge.source.clone(), edge.kind.clone(), edge.weight));
        } else {
            relays.push((
                edge.source.clone(),
                edge.target.clone(),
                edge.kind.clone(),
                edge.weight,
            ));
        }
    }
    incoming.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    outgoing.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    relays.sort_by(|a, b| b.3.total_cmp(&a.3).then_with(|| a.0.cmp(&b.0)));

    println!("focus: {}", display_path_compact(&project_path));
    println!(
        "lineage: depth={} threshold>={:.2} nodes={} edges={}",
        depth,
        min_weight,
        subgraph.nodes.len(),
        subgraph.edges.len()
    );

    println!();
    println!("direct edges:");
    println!(
        "{:<4} {:<4} {:>7}  {:<18} {}",
        "rank", "dir", "weight", "relation", "path"
    );
    let mut rank = 1usize;
    for (src, kind, weight) in incoming.iter().take(limit) {
        println!(
            "{:<4} {:<4} {:>7.3}  {:<18} {}",
            rank,
            "in",
            weight,
            clipped(kind, 18),
            display_path_compact(src)
        );
        rank += 1;
    }
    for (dst, kind, weight) in outgoing.iter().take(limit) {
        println!(
            "{:<4} {:<4} {:>7.3}  {:<18} {}",
            rank,
            "out",
            weight,
            clipped(kind, 18),
            display_path_compact(dst)
        );
        rank += 1;
    }
    if rank == 1 {
        println!("(no direct lineage edges matched current threshold)");
    }

    if depth > 1 {
        println!();
        println!("neighbor relay edges:");
        println!(
            "{:<4} {:>7}  {:<18} {} -> {}",
            "rank", "weight", "relation", "source", "target"
        );
        let mut relay_rank = 1usize;
        for (src, dst, kind, weight) in relays.iter().take(limit) {
            println!(
                "{:<4} {:>7.3}  {:<18} {} -> {}",
                relay_rank,
                weight,
                clipped(kind, 18),
                clipped(&display_path_compact(src), 42),
                clipped(&display_path_compact(dst), 42)
            );
            relay_rank += 1;
        }
        if relay_rank == 1 {
            println!("(no relay edges at current threshold/depth)");
        }
    }
    Ok(())
}

fn resolve_project_for_target(conn: &Connection, target: &str) -> Result<Option<String>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT path
FROM projects
WHERE path = ?1 OR ?1 LIKE path || '/%'
ORDER BY length(path) DESC
LIMIT 1
"#,
        )
        .map_err(|e| format!("failed preparing project resolution query: {}", e))?;
    let found: Option<String> = stmt
        .query_row(params![target], |row| row.get(0))
        .optional()
        .map_err(|e| format!("failed resolving target project: {}", e))?;
    Ok(found)
}

fn suggest_project_paths(
    conn: &Connection,
    needle: &str,
    limit: usize,
) -> Result<Vec<String>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT path
FROM projects
WHERE lower(path) LIKE lower(?1)
ORDER BY path ASC
LIMIT ?2
"#,
        )
        .map_err(|e| format!("failed preparing project suggestion query: {}", e))?;
    let pattern = if needle.trim().is_empty() {
        "%".to_string()
    } else {
        format!("%{}%", needle.trim())
    };
    let rows = stmt
        .query_map(params![pattern, limit as i64], |row| {
            row.get::<_, String>(0)
        })
        .map_err(|e| format!("failed querying project suggestions: {}", e))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(|e| format!("failed reading project suggestion row: {}", e))?);
    }
    Ok(out)
}

#[derive(Clone, Debug)]
struct GraphTextEdge {
    source: String,
    target: String,
    kind: String,
    weight: f64,
}

#[derive(Clone, Debug)]
struct GraphTextData {
    nodes: Vec<String>,
    edges: Vec<GraphTextEdge>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphTextLayout {
    Cluster,
    Lineage,
    Matrix,
}

impl GraphTextLayout {
    fn name(self) -> &'static str {
        match self {
            GraphTextLayout::Cluster => "cluster",
            GraphTextLayout::Lineage => "lineage",
            GraphTextLayout::Matrix => "matrix",
        }
    }

    fn toggle(self) -> Self {
        match self {
            GraphTextLayout::Cluster => GraphTextLayout::Lineage,
            GraphTextLayout::Lineage => GraphTextLayout::Matrix,
            GraphTextLayout::Matrix => GraphTextLayout::Cluster,
        }
    }
}

fn parse_graph_text_layout(raw: &str) -> Option<GraphTextLayout> {
    match raw.trim().to_lowercase().as_str() {
        "cluster" => Some(GraphTextLayout::Cluster),
        "lineage" => Some(GraphTextLayout::Lineage),
        "matrix" => Some(GraphTextLayout::Matrix),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphTextRenderer {
    Auto,
    Graphviz,
    Ascii,
}

impl GraphTextRenderer {
    fn name(self) -> &'static str {
        match self {
            GraphTextRenderer::Auto => "auto",
            GraphTextRenderer::Graphviz => "graphviz",
            GraphTextRenderer::Ascii => "ascii",
        }
    }

    fn toggle(self) -> Self {
        match self {
            GraphTextRenderer::Auto => GraphTextRenderer::Graphviz,
            GraphTextRenderer::Graphviz => GraphTextRenderer::Ascii,
            GraphTextRenderer::Ascii => GraphTextRenderer::Auto,
        }
    }
}

fn parse_graph_text_renderer(raw: &str) -> Option<GraphTextRenderer> {
    match raw.trim().to_lowercase().as_str() {
        "auto" => Some(GraphTextRenderer::Auto),
        "graphviz" => Some(GraphTextRenderer::Graphviz),
        "ascii" => Some(GraphTextRenderer::Ascii),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphUiBackend {
    Auto,
    Vt,
    Kitty,
    Iterm,
    Text,
}

impl GraphUiBackend {
    fn name(self) -> &'static str {
        match self {
            GraphUiBackend::Auto => "auto",
            GraphUiBackend::Vt => "vt",
            GraphUiBackend::Kitty => "kitty",
            GraphUiBackend::Iterm => "iterm",
            GraphUiBackend::Text => "text",
        }
    }

    fn toggle(self) -> Self {
        match self {
            GraphUiBackend::Auto => GraphUiBackend::Vt,
            GraphUiBackend::Vt => GraphUiBackend::Kitty,
            GraphUiBackend::Kitty => GraphUiBackend::Iterm,
            GraphUiBackend::Iterm => GraphUiBackend::Text,
            GraphUiBackend::Text => GraphUiBackend::Auto,
        }
    }
}

fn parse_graph_ui_backend(raw: &str) -> Option<GraphUiBackend> {
    match raw.trim().to_lowercase().as_str() {
        "auto" => Some(GraphUiBackend::Auto),
        "vt" => Some(GraphUiBackend::Vt),
        "kitty" => Some(GraphUiBackend::Kitty),
        "iterm" => Some(GraphUiBackend::Iterm),
        "text" => Some(GraphUiBackend::Text),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphUiBackendResolved {
    Vt,
    Kitty,
    Iterm,
    Text,
}

impl GraphUiBackendResolved {
    fn name(self) -> &'static str {
        match self {
            GraphUiBackendResolved::Vt => "vt-24bit",
            GraphUiBackendResolved::Kitty => "kitty",
            GraphUiBackendResolved::Iterm => "iterm-inline",
            GraphUiBackendResolved::Text => "text-fallback",
        }
    }
}

fn terminal_supports_kitty_graphics() -> bool {
    env::var("KITTY_WINDOW_ID").is_ok()
        || env::var("TERM")
            .map(|v| v.to_lowercase().contains("kitty"))
            .unwrap_or(false)
}

fn terminal_supports_iterm_inline() -> bool {
    env::var("TERM_PROGRAM")
        .map(|v| v == "iTerm.app")
        .unwrap_or(false)
        || env::var("WEZTERM_PANE").is_ok()
}

fn terminal_is_apple_terminal() -> bool {
    env::var("TERM_PROGRAM")
        .map(|v| v == "Apple_Terminal")
        .unwrap_or(false)
}

fn resolve_graph_ui_backend(choice: GraphUiBackend) -> (GraphUiBackendResolved, Option<String>) {
    if !command_exists("dot") {
        return (
            GraphUiBackendResolved::Text,
            Some("dot not found; using text fallback".to_string()),
        );
    }
    match choice {
        GraphUiBackend::Auto => {
            if terminal_supports_kitty_graphics() {
                (GraphUiBackendResolved::Kitty, None)
            } else if terminal_supports_iterm_inline() {
                (GraphUiBackendResolved::Iterm, None)
            } else if terminal_is_apple_terminal() {
                (
                    GraphUiBackendResolved::Text,
                    Some("Apple Terminal detected; defaulting to clean text mode".to_string()),
                )
            } else {
                (GraphUiBackendResolved::Vt, None)
            }
        }
        GraphUiBackend::Vt => (GraphUiBackendResolved::Vt, None),
        GraphUiBackend::Kitty => {
            if terminal_supports_kitty_graphics() {
                (GraphUiBackendResolved::Kitty, None)
            } else {
                (
                    GraphUiBackendResolved::Text,
                    Some("kitty backend unsupported in this terminal; using text".to_string()),
                )
            }
        }
        GraphUiBackend::Iterm => {
            if terminal_supports_iterm_inline() {
                (GraphUiBackendResolved::Iterm, None)
            } else {
                (
                    GraphUiBackendResolved::Text,
                    Some("iterm backend unsupported in this terminal; using text".to_string()),
                )
            }
        }
        GraphUiBackend::Text => (GraphUiBackendResolved::Text, None),
    }
}

#[derive(Clone, Debug)]
struct GraphUiState {
    backend: GraphUiBackend,
    focus_idx: usize,
    limit: usize,
    min_weight: f64,
    depth: usize,
    cluster_cross_edges: bool,
    status: String,
}

fn run_graph_ui_cmd(
    focus_hint: Option<&str>,
    limit: usize,
    initial_backend: Option<GraphUiBackend>,
    initial_min_weight: f64,
    initial_depth: usize,
) -> Result<(), String> {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut state = GraphUiState {
        backend: initial_backend.unwrap_or(GraphUiBackend::Auto),
        focus_idx: 0,
        limit: limit.max(12).min(360),
        min_weight: initial_min_weight.clamp(0.0, 1.0),
        depth: initial_depth.clamp(1, 3),
        cluster_cross_edges: false,
        status: "graph loaded".to_string(),
    };
    let mut data = load_graph_text_data(&cwd, focus_hint, state.limit)?;
    if data.nodes.is_empty() {
        println!(
            "graph ui: no graph nodes found; run `cypress add <path>` then `cypress index` (example: `cypress add ~/c-projects && cypress index`)"
        );
        return Ok(());
    }
    state.focus_idx = graph_text_focus_index(&data.nodes, focus_hint);

    if !tty_ui_available() {
        let (width, mut height) = terminal_size_fallback();
        if height > 40 {
            height = 40;
        }
        let focus_idx = state.focus_idx.min(data.nodes.len().saturating_sub(1));
        let focus = &data.nodes[focus_idx];
        let subgraph = build_focus_subgraph(
            &data,
            focus,
            state.limit.min(100).max(8),
            state.min_weight,
            state.depth,
        );
        let lines = render_focus_star_cluster(
            focus,
            width.max(80),
            height.saturating_sub(6).max(8),
            &subgraph.nodes,
            &subgraph.edges,
        );
        println!("{}", lines.join("\n"));
        return Ok(());
    }

    let _guard = enter_config_tui_mode()?;
    let mut stdout = std::io::stdout();
    loop {
        draw_graph_ui_tui(&mut stdout, &data, &state)?;
        let Some(key) = read_config_key()? else {
            thread::sleep(Duration::from_millis(120));
            continue;
        };
        match key {
            ConfigKey::Esc | ConfigKey::CtrlC | ConfigKey::Char('q') => return Ok(()),
            ConfigKey::Right | ConfigKey::Down | ConfigKey::Char('j') | ConfigKey::Char('l') => {
                if !data.nodes.is_empty() {
                    state.focus_idx = (state.focus_idx + 1) % data.nodes.len();
                    state.status =
                        format!("focus: {}", path_basename(&data.nodes[state.focus_idx]));
                }
            }
            ConfigKey::Left | ConfigKey::Up | ConfigKey::Char('k') | ConfigKey::Char('h') => {
                if !data.nodes.is_empty() {
                    if state.focus_idx == 0 {
                        state.focus_idx = data.nodes.len().saturating_sub(1);
                    } else {
                        state.focus_idx -= 1;
                    }
                    state.status =
                        format!("focus: {}", path_basename(&data.nodes[state.focus_idx]));
                }
            }
            ConfigKey::Char('b') => {
                state.backend = state.backend.toggle();
                state.status = format!("ui backend: {}", state.backend.name());
            }
            ConfigKey::Char('x') => {
                state.cluster_cross_edges = !state.cluster_cross_edges;
                state.status = format!(
                    "cross-edges: {}",
                    if state.cluster_cross_edges {
                        "on"
                    } else {
                        "off"
                    }
                );
            }
            ConfigKey::Char('w') => {
                state.min_weight = (state.min_weight + 0.05).min(0.95);
                state.status = format!("edge threshold: {:.2}", state.min_weight);
            }
            ConfigKey::Char('s') => {
                state.min_weight = (state.min_weight - 0.05).max(0.0);
                state.status = format!("edge threshold: {:.2}", state.min_weight);
            }
            ConfigKey::Char(']') => {
                state.depth = (state.depth + 1).min(3);
                state.status = format!("hop depth: {}", state.depth);
            }
            ConfigKey::Char('[') => {
                state.depth = state.depth.saturating_sub(1).max(1);
                state.status = format!("hop depth: {}", state.depth);
            }
            ConfigKey::Char('+') | ConfigKey::Char('=') => {
                state.limit = (state.limit + 10).min(360);
                let focus = data.nodes.get(state.focus_idx).cloned();
                match load_graph_text_data(&cwd, focus.as_deref(), state.limit) {
                    Ok(next) => {
                        data = next;
                        state.focus_idx = graph_text_focus_index(&data.nodes, focus.as_deref());
                        state.status = format!("node limit: {}", state.limit);
                    }
                    Err(e) => state.status = format!("reload failed: {}", e),
                }
            }
            ConfigKey::Char('-') => {
                state.limit = state.limit.saturating_sub(10).max(12);
                let focus = data.nodes.get(state.focus_idx).cloned();
                match load_graph_text_data(&cwd, focus.as_deref(), state.limit) {
                    Ok(next) => {
                        data = next;
                        state.focus_idx = graph_text_focus_index(&data.nodes, focus.as_deref());
                        state.status = format!("node limit: {}", state.limit);
                    }
                    Err(e) => state.status = format!("reload failed: {}", e),
                }
            }
            ConfigKey::Char('r') => {
                let focus = data.nodes.get(state.focus_idx).cloned();
                match load_graph_text_data(&cwd, focus.as_deref(), state.limit) {
                    Ok(next) => {
                        data = next;
                        state.focus_idx = graph_text_focus_index(&data.nodes, focus.as_deref());
                        state.status = "graph reloaded".to_string();
                    }
                    Err(e) => state.status = format!("reload failed: {}", e),
                }
            }
            _ => {}
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct GraphPoint {
    x: i32,
    y: i32,
}

#[derive(Clone, Debug)]
struct GraphTextState {
    layout: GraphTextLayout,
    renderer: GraphTextRenderer,
    focus_idx: usize,
    limit: usize,
    min_weight: f64,
    depth: usize,
    cluster_cross_edges: bool,
    status: String,
}

fn run_graph_text_cmd(
    focus_hint: Option<&str>,
    limit: usize,
    initial_layout: Option<GraphTextLayout>,
    initial_renderer: Option<GraphTextRenderer>,
    initial_min_weight: f64,
    initial_depth: usize,
) -> Result<(), String> {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut state = GraphTextState {
        layout: initial_layout.unwrap_or(GraphTextLayout::Lineage),
        renderer: initial_renderer.unwrap_or(GraphTextRenderer::Auto),
        focus_idx: 0,
        limit: limit.max(12).min(360),
        min_weight: initial_min_weight.clamp(0.0, 1.0),
        depth: initial_depth.clamp(1, 3),
        cluster_cross_edges: false,
        status: "graph loaded".to_string(),
    };
    let mut data = load_graph_text_data(&cwd, focus_hint, state.limit)?;
    if data.nodes.is_empty() {
        println!("graph text: no graph nodes found; run `cypress index` first");
        return Ok(());
    }
    state.focus_idx = graph_text_focus_index(&data.nodes, focus_hint);
    if !tty_ui_available() {
        let (width, mut height) = terminal_size_fallback();
        if height > 44 {
            height = 44;
        }
        let lines = render_graph_text_snapshot(&data, &state, width, height);
        println!("{}", lines.join("\n"));
        return Ok(());
    }

    let _guard = match enter_config_tui_mode() {
        Ok(v) => v,
        Err(err) => {
            let (width, mut height) = terminal_size_fallback();
            if height > 44 {
                height = 44;
            }
            let lines = render_graph_text_snapshot(&data, &state, width, height);
            println!("{}", lines.join("\n"));
            eprintln!(
                "graph text: interactive TUI unavailable ({}); rendered static snapshot",
                err
            );
            return Ok(());
        }
    };
    let mut stdout = std::io::stdout();
    loop {
        draw_graph_text_tui(&mut stdout, &data, &state)?;
        let Some(key) = read_config_key()? else {
            continue;
        };
        match key {
            ConfigKey::Esc | ConfigKey::CtrlC | ConfigKey::Char('q') => return Ok(()),
            ConfigKey::Char('t') | ConfigKey::Tab => {
                state.layout = state.layout.toggle();
                state.status = format!("layout: {}", state.layout.name());
            }
            ConfigKey::Char('v') => {
                state.renderer = state.renderer.toggle();
                state.status = format!("renderer: {}", state.renderer.name());
            }
            ConfigKey::Char('x') => {
                state.cluster_cross_edges = !state.cluster_cross_edges;
                state.status = format!(
                    "cluster cross-edges: {}",
                    if state.cluster_cross_edges {
                        "on"
                    } else {
                        "off"
                    }
                );
            }
            ConfigKey::Right | ConfigKey::Down | ConfigKey::Char('j') | ConfigKey::Char('l') => {
                if !data.nodes.is_empty() {
                    state.focus_idx = (state.focus_idx + 1) % data.nodes.len();
                    state.status =
                        format!("focus: {}", path_basename(&data.nodes[state.focus_idx]));
                }
            }
            ConfigKey::Left | ConfigKey::Up | ConfigKey::Char('k') | ConfigKey::Char('h') => {
                if !data.nodes.is_empty() {
                    if state.focus_idx == 0 {
                        state.focus_idx = data.nodes.len().saturating_sub(1);
                    } else {
                        state.focus_idx -= 1;
                    }
                    state.status =
                        format!("focus: {}", path_basename(&data.nodes[state.focus_idx]));
                }
            }
            ConfigKey::Char('+') | ConfigKey::Char('=') => {
                state.limit = (state.limit + 10).min(360);
                let focus = data.nodes.get(state.focus_idx).cloned();
                match load_graph_text_data(&cwd, focus.as_deref(), state.limit) {
                    Ok(next) => {
                        data = next;
                        state.focus_idx = graph_text_focus_index(&data.nodes, focus.as_deref());
                        state.status = format!("node limit: {}", state.limit);
                    }
                    Err(e) => state.status = format!("reload failed: {}", e),
                }
            }
            ConfigKey::Char('-') => {
                state.limit = state.limit.saturating_sub(10).max(12);
                let focus = data.nodes.get(state.focus_idx).cloned();
                match load_graph_text_data(&cwd, focus.as_deref(), state.limit) {
                    Ok(next) => {
                        data = next;
                        state.focus_idx = graph_text_focus_index(&data.nodes, focus.as_deref());
                        state.status = format!("node limit: {}", state.limit);
                    }
                    Err(e) => state.status = format!("reload failed: {}", e),
                }
            }
            ConfigKey::Char('r') => {
                let focus = data.nodes.get(state.focus_idx).cloned();
                match load_graph_text_data(&cwd, focus.as_deref(), state.limit) {
                    Ok(next) => {
                        data = next;
                        state.focus_idx = graph_text_focus_index(&data.nodes, focus.as_deref());
                        state.status = "graph reloaded".to_string();
                    }
                    Err(e) => state.status = format!("reload failed: {}", e),
                }
            }
            ConfigKey::Char('w') => {
                state.min_weight = (state.min_weight + 0.05).min(0.95);
                state.status = format!("edge threshold: {:.2}", state.min_weight);
            }
            ConfigKey::Char('s') => {
                state.min_weight = (state.min_weight - 0.05).max(0.0);
                state.status = format!("edge threshold: {:.2}", state.min_weight);
            }
            ConfigKey::Char(']') => {
                state.depth = (state.depth + 1).min(3);
                state.status = format!("hop depth: {}", state.depth);
            }
            ConfigKey::Char('[') => {
                state.depth = state.depth.saturating_sub(1).max(1);
                state.status = format!("hop depth: {}", state.depth);
            }
            _ => {}
        }
    }
}

fn graph_text_focus_index(nodes: &[String], focus_hint: Option<&str>) -> usize {
    if nodes.is_empty() {
        return 0;
    }
    if let Some(raw) = focus_hint {
        let target = normalize_path(raw).to_string_lossy().to_string();
        if let Some(idx) = nodes.iter().position(|n| *n == target) {
            return idx;
        }
    }
    0
}

fn load_graph_text_data(
    cwd: &Path,
    focus_hint: Option<&str>,
    limit: usize,
) -> Result<GraphTextData, String> {
    let dbp = db_path(cwd);
    ensure_db_schema(&dbp)?;
    let conn = open_db_rw(&dbp)?;
    let limit = limit.max(12).min(600);
    let mut payload = graph_view_data_json(&conn, focus_hint, limit)?;

    let mut nodes: Vec<String> = Vec::new();
    if let Some(items) = payload.get_mut("nodes").and_then(|v| v.as_array_mut()) {
        let mut seen: HashSet<String> = HashSet::new();
        for item in items.iter() {
            let path = item
                .get("path")
                .and_then(|v| v.as_str())
                .or_else(|| item.get("id").and_then(|v| v.as_str()))
                .unwrap_or_default()
                .trim()
                .to_string();
            if path.is_empty() {
                continue;
            }
            let normalized = normalize_path(&path).to_string_lossy().to_string();
            if seen.insert(normalized.clone()) {
                nodes.push(normalized);
            }
        }
    }

    let mut edges: Vec<GraphTextEdge> = Vec::new();
    if let Some(items) = payload.get("edges").and_then(|v| v.as_array()) {
        for item in items {
            let source_raw = item
                .get("source")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim();
            let target_raw = item
                .get("target")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim();
            if source_raw.is_empty() || target_raw.is_empty() {
                continue;
            }
            let source = normalize_path(source_raw).to_string_lossy().to_string();
            let target = normalize_path(target_raw).to_string_lossy().to_string();
            if source == target {
                continue;
            }
            let kind = item
                .get("kind")
                .and_then(|v| v.as_str())
                .unwrap_or("related")
                .trim()
                .to_string();
            let weight = item.get("weight").and_then(|v| v.as_f64()).unwrap_or(0.0);
            edges.push(GraphTextEdge {
                source,
                target,
                kind,
                weight,
            });
        }
    }

    if nodes.is_empty() && focus_hint.is_some() {
        return load_graph_text_data(cwd, None, limit);
    }
    Ok(GraphTextData { nodes, edges })
}

fn draw_graph_text_tui(
    stdout: &mut std::io::Stdout,
    data: &GraphTextData,
    state: &GraphTextState,
) -> Result<(), String> {
    let (width, height) = terminal_size_stty();
    let lines = render_graph_text_snapshot(data, state, width, height);
    let payload = format!("\x1b[H\x1b[2J{}", lines.join("\n"));
    stdout
        .write_all(payload.as_bytes())
        .map_err(|e| format!("failed writing graph text frame: {}", e))?;
    stdout
        .flush()
        .map_err(|e| format!("failed flushing graph text frame: {}", e))
}

fn draw_graph_ui_tui(
    stdout: &mut std::io::Stdout,
    data: &GraphTextData,
    state: &GraphUiState,
) -> Result<(), String> {
    if data.nodes.is_empty() {
        return Err("graph ui: no graph nodes".to_string());
    }
    let (width, height) = terminal_size_stty();
    let focus_idx = state.focus_idx.min(data.nodes.len().saturating_sub(1));
    let focus = &data.nodes[focus_idx];
    let subgraph = build_focus_subgraph(
        data,
        focus,
        state.limit.min(140).max(8),
        state.min_weight,
        state.depth,
    );
    let (backend, backend_note) = resolve_graph_ui_backend(state.backend);
    let header_rows = 5usize;
    let mut canvas_h = height.saturating_sub(header_rows).max(8);
    if backend == GraphUiBackendResolved::Text {
        canvas_h = canvas_h.min(18);
    }
    let canvas = render_graph_ui_canvas(
        focus,
        &subgraph,
        width.max(80),
        canvas_h,
        backend,
        state.cluster_cross_edges,
    );

    let mut payload = String::new();
    payload.push_str("\x1b[H\x1b[2J");
    payload.push_str(&clipped(
        &format!(
            "\x1b[36;1mcypress graph[ui]\x1b[0m nodes={} edges={} visible_nodes={} visible_edges={} backend={} w>={:.2} depth={} cross={}",
            data.nodes.len(),
            data.edges.len(),
            subgraph.nodes.len(),
            subgraph.edges.len(),
            backend.name(),
            state.min_weight,
            state.depth,
            if state.cluster_cross_edges { "on" } else { "off" }
        ),
        width,
    ));
    payload.push('\n');
    payload.push_str(&clipped(
        "keys: b backend  x cross-edges  ←/→ or j/k focus  w/s threshold  [/ ] depth  r reload  +/- scope  q exit",
        width,
    ));
    payload.push('\n');
    payload.push_str(&clipped(&format!("focus: {}", focus), width));
    payload.push('\n');
    let mut status = state.status.clone();
    if let Some(note) = backend_note {
        if !status.is_empty() {
            status.push_str(" | ");
        }
        status.push_str(&note);
    }
    payload.push_str(&clipped(&format!("status: {}", status), width));
    payload.push('\n');
    payload.push_str(&clipped(
        "hint: use iTerm2/Kitty/WezTerm for richer pixel rendering; Terminal.app is limited",
        width,
    ));
    payload.push('\n');
    payload.push_str(&canvas);
    if !payload.ends_with('\n') {
        payload.push('\n');
    }

    stdout
        .write_all(payload.as_bytes())
        .map_err(|e| format!("failed writing graph ui frame: {}", e))?;
    stdout
        .flush()
        .map_err(|e| format!("failed flushing graph ui frame: {}", e))
}

fn render_graph_ui_canvas(
    focus: &str,
    subgraph: &GraphTextSubgraph,
    width: usize,
    height: usize,
    backend: GraphUiBackendResolved,
    include_cross_edges: bool,
) -> String {
    match backend {
        GraphUiBackendResolved::Text => {
            render_focus_star_cluster(focus, width, height, &subgraph.nodes, &subgraph.edges)
                .join("\n")
        }
        GraphUiBackendResolved::Vt => {
            let dot_src = graph_ui_dot_source(focus, subgraph, width, height, include_cross_edges);
            match run_dot_render(&dot_src, "vt-24bit", "neato") {
                Ok(bytes) => String::from_utf8_lossy(&bytes).to_string(),
                Err(err) => {
                    let mut lines = render_focus_star_cluster(
                        focus,
                        width,
                        height,
                        &subgraph.nodes,
                        &subgraph.edges,
                    );
                    lines.push(format!(
                        "graph ui: vt render failed: {}",
                        clipped(&err, 120)
                    ));
                    lines.join("\n")
                }
            }
        }
        GraphUiBackendResolved::Kitty => {
            let dot_src = graph_ui_dot_source(focus, subgraph, width, height, include_cross_edges);
            let render = run_dot_render(&dot_src, "kittyz", "neato")
                .or_else(|_| run_dot_render(&dot_src, "kitty", "neato"));
            match render {
                Ok(bytes) => String::from_utf8_lossy(&bytes).to_string(),
                Err(err) => {
                    let mut lines = render_focus_star_cluster(
                        focus,
                        width,
                        height,
                        &subgraph.nodes,
                        &subgraph.edges,
                    );
                    lines.push(format!(
                        "graph ui: kitty render failed: {}",
                        clipped(&err, 120)
                    ));
                    lines.join("\n")
                }
            }
        }
        GraphUiBackendResolved::Iterm => {
            let dot_src = graph_ui_dot_source(focus, subgraph, width, height, include_cross_edges);
            match run_dot_render(&dot_src, "png", "neato") {
                Ok(png) => {
                    let encoded = BASE64_STANDARD.encode(png);
                    format!(
                        "\x1b]1337;File=inline=1;width=100%;preserveAspectRatio=1:{}\x07",
                        encoded
                    )
                }
                Err(err) => {
                    let mut lines = render_focus_star_cluster(
                        focus,
                        width,
                        height,
                        &subgraph.nodes,
                        &subgraph.edges,
                    );
                    lines.push(format!(
                        "graph ui: iterm inline render failed: {}",
                        clipped(&err, 120)
                    ));
                    lines.join("\n")
                }
            }
        }
    }
}

fn run_dot_render(dot_src: &str, format: &str, engine: &str) -> Result<Vec<u8>, String> {
    let mut child = Command::new("dot")
        .arg(format!("-K{}", engine))
        .arg(format!("-T{}", format))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed spawning dot: {}", e))?;
    if let Some(stdin) = child.stdin.as_mut() {
        stdin
            .write_all(dot_src.as_bytes())
            .map_err(|e| format!("failed writing dot input: {}", e))?;
    } else {
        return Err("dot stdin unavailable".to_string());
    }
    let out = child
        .wait_with_output()
        .map_err(|e| format!("failed reading dot output: {}", e))?;
    if out.status.success() {
        Ok(out.stdout)
    } else {
        Err(format!(
            "dot render failed ({}): {}",
            out.status,
            clipped(&String::from_utf8_lossy(&out.stderr), 200)
        ))
    }
}

fn graph_ui_dot_source(
    focus: &str,
    subgraph: &GraphTextSubgraph,
    width: usize,
    height: usize,
    include_cross_edges: bool,
) -> String {
    let (nodes, edges) =
        cluster_draw_data(focus, &subgraph.nodes, &subgraph.edges, include_cross_edges);
    let w_in = ((width as f64) / 10.0).clamp(8.0, 26.0);
    let h_in = ((height as f64) / 4.0).clamp(4.0, 16.0);
    let mut node_ids: HashMap<String, String> = HashMap::new();
    for (idx, node) in nodes.iter().enumerate() {
        node_ids.insert(node.clone(), format!("n{}", idx));
    }
    let mut dot = format!(
        "digraph G {{\n\
         graph [bgcolor=\"#0b0f14\",pad=0.20,overlap=false,splines=true,outputorder=edgesfirst,size=\"{:.2},{:.2}!\",ratio=fill];\n\
         node [shape=circle,style=filled,fontname=\"Menlo\",fontsize=11,fontcolor=\"#e5e7eb\",color=\"#93c5fd\",fillcolor=\"#1f2937\",penwidth=1.4,width=0.84,height=0.84,fixedsize=true];\n\
         edge [color=\"#6b7280\",penwidth=1.2,arrowsize=0.55];\n",
        w_in, h_in
    );
    for node in &nodes {
        let id = node_ids
            .get(node)
            .cloned()
            .unwrap_or_else(|| "n0".to_string());
        let label = short_graph_label(node, 18);
        if node == focus {
            dot.push_str(&format!(
                "{} [label=\"{}\",fillcolor=\"#0f172a\",color=\"#f59e0b\",penwidth=2.6,width=1.02,height=1.02];\n",
                id,
                dot_escape(&label)
            ));
        } else {
            dot.push_str(&format!("{} [label=\"{}\"];\n", id, dot_escape(&label)));
        }
    }
    for edge in &edges {
        let Some(src) = node_ids.get(&edge.source) else {
            continue;
        };
        let Some(dst) = node_ids.get(&edge.target) else {
            continue;
        };
        let focus_edge = edge.source == focus || edge.target == focus;
        if focus_edge {
            dot.push_str(&format!(
                "{} -> {} [color=\"#f59e0b\",penwidth=2.1];\n",
                src, dst
            ));
        } else {
            dot.push_str(&format!("{} -> {};\n", src, dst));
        }
    }
    dot.push_str("}\n");
    dot
}

fn render_graph_text_snapshot(
    data: &GraphTextData,
    state: &GraphTextState,
    width: usize,
    height: usize,
) -> Vec<String> {
    if data.nodes.is_empty() {
        return vec!["graph text: no graph nodes".to_string()];
    }
    let focus_idx = state.focus_idx.min(data.nodes.len().saturating_sub(1));
    let focus = &data.nodes[focus_idx];
    let header_rows = 4usize;
    let footer_rows = if state.layout == GraphTextLayout::Cluster {
        12usize
    } else {
        6usize
    };
    let mut canvas_h = height.saturating_sub(header_rows + footer_rows).max(8);
    if state.layout == GraphTextLayout::Lineage {
        canvas_h = canvas_h.min(18);
    } else if state.layout == GraphTextLayout::Matrix {
        canvas_h = canvas_h.min(20);
    } else if state.layout == GraphTextLayout::Cluster {
        canvas_h = canvas_h.min(18);
    }
    let canvas_w = width.max(48);

    let layout_cap = match state.layout {
        GraphTextLayout::Matrix => 30,
        GraphTextLayout::Cluster => 18,
        GraphTextLayout::Lineage => 16,
    };
    let visible_limit = state.limit.min(layout_cap).max(8);
    let subgraph = build_focus_subgraph(data, focus, visible_limit, state.min_weight, state.depth);
    let mut renderer_used = "n/a".to_string();
    let mut renderer_note: Option<String> = None;
    let mut cluster_legend_nodes: Option<Vec<String>> = None;
    let canvas_lines = match state.layout {
        GraphTextLayout::Lineage => {
            render_lineage_diagram(focus, &subgraph.edges, canvas_w, canvas_h)
        }
        GraphTextLayout::Matrix => {
            render_graph_matrix(focus, &subgraph.nodes, &subgraph.edges, canvas_w, canvas_h)
        }
        GraphTextLayout::Cluster => {
            let (legend_nodes, _) = cluster_draw_data(
                focus,
                &subgraph.nodes,
                &subgraph.edges,
                state.cluster_cross_edges,
            );
            cluster_legend_nodes = Some(legend_nodes);
            let (lines, used, note) = render_graph_cluster(
                focus,
                canvas_w,
                canvas_h,
                &subgraph.nodes,
                &subgraph.edges,
                state.renderer,
                state.cluster_cross_edges,
            );
            renderer_used = used;
            renderer_note = note;
            lines
        }
    };
    let neighbor_limit = if state.layout == GraphTextLayout::Cluster {
        3
    } else {
        4
    };
    let neighbor_lines = focus_edge_summary_lines(focus, &subgraph.edges, neighbor_limit, width);
    let mut out: Vec<String> = Vec::new();
    out.push(format!(
        "\x1b[36;1mcypress graph[text:{}]\x1b[0m nodes={} edges={} visible_nodes={} visible_edges={} limit={} w>={:.2} depth={} renderer={} cross={}",
        state.layout.name(),
        data.nodes.len(),
        data.edges.len(),
        subgraph.nodes.len(),
        subgraph.edges.len(),
        state.limit
        ,
        state.min_weight,
        state.depth,
        renderer_used,
        if state.cluster_cross_edges {
            "on"
        } else {
            "off"
        }
    ));
    out.push(clipped(
        "keys: t/Tab layout  v renderer  x cross-edges  \u{2190}/\u{2192} or j/k focus  w/s threshold  [/ ] depth  r reload  +/- limit  q exit",
        width,
    ));
    out.push(clipped(&format!("focus: {}", focus), width));
    let mut status_line = state.status.clone();
    if let Some(note) = renderer_note {
        if !status_line.is_empty() {
            status_line.push_str(" | ");
        }
        status_line.push_str(&note);
    }
    out.push(clipped(&format!("status: {}", status_line), width));
    out.extend(canvas_lines);
    for line in neighbor_lines {
        out.push(clipped(&line, width));
    }
    if state.layout == GraphTextLayout::Cluster {
        let legend_nodes = cluster_legend_nodes.as_deref().unwrap_or(&subgraph.nodes);
        let legend_lines = cluster_node_legend_lines(focus, legend_nodes, width, 8);
        for line in legend_lines {
            out.push(clipped(&line, width));
        }
    }
    while out.len() < height {
        out.push(String::new());
    }
    out.truncate(height);
    out
}

#[derive(Clone, Debug)]
struct GraphTextSubgraph {
    nodes: Vec<String>,
    edges: Vec<GraphTextEdge>,
}

fn build_focus_subgraph(
    data: &GraphTextData,
    focus: &str,
    max_nodes: usize,
    min_weight: f64,
    depth: usize,
) -> GraphTextSubgraph {
    let max_nodes = max_nodes.max(6);
    let depth = depth.clamp(1, 3);
    let min_weight = min_weight.clamp(0.0, 1.0);

    let mut eligible_edges: Vec<GraphTextEdge> = data
        .edges
        .iter()
        .filter(|e| e.weight >= min_weight)
        .cloned()
        .collect();
    if eligible_edges.is_empty() {
        let mut fallback: Vec<GraphTextEdge> = data
            .edges
            .iter()
            .filter(|e| e.source == focus || e.target == focus)
            .cloned()
            .collect();
        fallback.sort_by(|a, b| b.weight.total_cmp(&a.weight));
        eligible_edges = fallback.into_iter().take(20).collect();
    }
    eligible_edges.sort_by(|a, b| b.weight.total_cmp(&a.weight));

    let mut selected: HashSet<String> = HashSet::new();
    selected.insert(focus.to_string());
    let mut frontier: HashSet<String> = HashSet::new();
    frontier.insert(focus.to_string());
    for _ in 0..depth {
        if selected.len() >= max_nodes {
            break;
        }
        let mut candidates: Vec<(f64, String)> = Vec::new();
        for edge in &eligible_edges {
            if frontier.contains(&edge.source) && !selected.contains(&edge.target) {
                candidates.push((edge.weight, edge.target.clone()));
            }
            if frontier.contains(&edge.target) && !selected.contains(&edge.source) {
                candidates.push((edge.weight * 0.97, edge.source.clone()));
            }
        }
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut next_frontier: HashSet<String> = HashSet::new();
        for (_, node) in candidates {
            if selected.len() >= max_nodes {
                break;
            }
            if selected.insert(node.clone()) {
                next_frontier.insert(node);
            }
        }
        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }

    if selected.len() < max_nodes {
        for edge in &eligible_edges {
            if selected.contains(&edge.source) && selected.insert(edge.target.clone()) {
                if selected.len() >= max_nodes {
                    break;
                }
            }
            if selected.contains(&edge.target) && selected.insert(edge.source.clone()) {
                if selected.len() >= max_nodes {
                    break;
                }
            }
        }
    }

    let mut sub_edges: Vec<GraphTextEdge> = eligible_edges
        .iter()
        .filter(|e| selected.contains(&e.source) && selected.contains(&e.target))
        .cloned()
        .collect();
    sub_edges.sort_by(|a, b| b.weight.total_cmp(&a.weight));
    let edge_cap = (max_nodes * 3).max(16);
    if sub_edges.len() > edge_cap {
        sub_edges.truncate(edge_cap);
    }

    let mut score_by_node: HashMap<String, f64> = HashMap::new();
    score_by_node.insert(focus.to_string(), 9999.0);
    for edge in &sub_edges {
        let focus_boost = if edge.source == focus || edge.target == focus {
            1.25
        } else {
            1.0
        };
        *score_by_node.entry(edge.source.clone()).or_insert(0.0) += edge.weight * focus_boost;
        *score_by_node.entry(edge.target.clone()).or_insert(0.0) += edge.weight * focus_boost;
    }

    let mut nodes: Vec<String> = selected.into_iter().collect();
    nodes.sort_by(|a, b| {
        if a == focus {
            return std::cmp::Ordering::Less;
        }
        if b == focus {
            return std::cmp::Ordering::Greater;
        }
        let sa = *score_by_node.get(a).unwrap_or(&0.0);
        let sb = *score_by_node.get(b).unwrap_or(&0.0);
        sb.total_cmp(&sa).then_with(|| a.cmp(b))
    });
    GraphTextSubgraph {
        nodes,
        edges: sub_edges,
    }
}

fn focus_edge_summary_lines(
    focus: &str,
    edges: &[GraphTextEdge],
    max_lines: usize,
    width: usize,
) -> Vec<String> {
    let mut rels: Vec<(f64, String)> = Vec::new();
    let mut out_count = 0usize;
    let mut in_count = 0usize;
    for edge in edges {
        if edge.source == focus {
            out_count += 1;
            rels.push((
                edge.weight,
                format!(
                    "out  -> {}  ({:.2}, {})",
                    path_basename(&edge.target),
                    edge.weight,
                    clipped(&edge.kind, 14)
                ),
            ));
        } else if edge.target == focus {
            in_count += 1;
            rels.push((
                edge.weight,
                format!(
                    "in   <- {}  ({:.2}, {})",
                    path_basename(&edge.source),
                    edge.weight,
                    clipped(&edge.kind, 14)
                ),
            ));
        }
    }
    rels.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut lines = vec![clipped(
        &format!(
            "neighbors (focused subgraph): outgoing={} incoming={}",
            out_count, in_count
        ),
        width,
    )];
    for (_, text) in rels.into_iter().take(max_lines.max(1)) {
        lines.push(clipped(&format!("  {}", text), width));
    }
    if lines.len() == 1 {
        lines.push(clipped("  (no connected edges for current focus)", width));
    }
    lines
}

fn render_graph_cluster(
    focus: &str,
    width: usize,
    height: usize,
    visible_nodes: &[String],
    visible_edges: &[GraphTextEdge],
    renderer: GraphTextRenderer,
    include_cross_edges: bool,
) -> (Vec<String>, String, Option<String>) {
    if !include_cross_edges {
        return (
            render_focus_star_cluster(focus, width, height, visible_nodes, visible_edges),
            "focus-star".to_string(),
            Some("press x to show full cross-edge cluster".to_string()),
        );
    }
    let dot_available = command_exists("dot");
    match renderer {
        GraphTextRenderer::Graphviz => {
            if !dot_available {
                return (
                    render_graph_ascii(
                        focus,
                        width,
                        height,
                        visible_nodes,
                        visible_edges,
                        include_cross_edges,
                    ),
                    "ascii".to_string(),
                    Some("graphviz not found on PATH; using ascii".to_string()),
                );
            }
            match render_graph_graphviz_plain(
                focus,
                width,
                height,
                visible_nodes,
                visible_edges,
                include_cross_edges,
            ) {
                Ok(lines) => (lines, "graphviz".to_string(), None),
                Err(err) => (
                    render_graph_ascii(
                        focus,
                        width,
                        height,
                        visible_nodes,
                        visible_edges,
                        include_cross_edges,
                    ),
                    "ascii".to_string(),
                    Some(format!(
                        "graphviz failed; using ascii ({})",
                        clipped(&err, 80)
                    )),
                ),
            }
        }
        GraphTextRenderer::Auto => {
            if dot_available {
                match render_graph_graphviz_plain(
                    focus,
                    width,
                    height,
                    visible_nodes,
                    visible_edges,
                    include_cross_edges,
                ) {
                    Ok(lines) => (lines, "graphviz".to_string(), None),
                    Err(_) => (
                        render_graph_ascii(
                            focus,
                            width,
                            height,
                            visible_nodes,
                            visible_edges,
                            include_cross_edges,
                        ),
                        "ascii".to_string(),
                        Some("graphviz parse fallback to ascii".to_string()),
                    ),
                }
            } else {
                (
                    render_graph_ascii(
                        focus,
                        width,
                        height,
                        visible_nodes,
                        visible_edges,
                        include_cross_edges,
                    ),
                    "ascii".to_string(),
                    None,
                )
            }
        }
        GraphTextRenderer::Ascii => (
            render_graph_ascii(
                focus,
                width,
                height,
                visible_nodes,
                visible_edges,
                include_cross_edges,
            ),
            "ascii".to_string(),
            None,
        ),
    }
}

fn render_focus_star_cluster(
    focus: &str,
    width: usize,
    height: usize,
    visible_nodes: &[String],
    visible_edges: &[GraphTextEdge],
) -> Vec<String> {
    let mut canvas = vec![vec![' '; width]; height];
    if visible_nodes.is_empty() {
        return vec![String::new(); height];
    }
    let (draw_nodes, draw_edges) = cluster_draw_data(focus, visible_nodes, visible_edges, false);
    let labels = build_cluster_label_map(&draw_nodes, focus);

    let mut in_map: HashMap<String, f64> = HashMap::new();
    let mut out_map: HashMap<String, f64> = HashMap::new();
    for edge in &draw_edges {
        if edge.source == focus {
            let cur = out_map.get(&edge.target).copied().unwrap_or(0.0);
            if edge.weight > cur {
                out_map.insert(edge.target.clone(), edge.weight);
            }
        } else if edge.target == focus {
            let cur = in_map.get(&edge.source).copied().unwrap_or(0.0);
            if edge.weight > cur {
                in_map.insert(edge.source.clone(), edge.weight);
            }
        }
    }
    let mut incoming: Vec<(String, f64)> = in_map.into_iter().collect();
    let mut outgoing: Vec<(String, f64)> = out_map.into_iter().collect();
    incoming.sort_by(|a, b| b.1.total_cmp(&a.1));
    outgoing.sort_by(|a, b| b.1.total_cmp(&a.1));
    let max_side = ((height.saturating_sub(4)) / 2).clamp(3, 8);
    incoming.truncate(max_side);
    outgoing.truncate(max_side);

    let focus_x = if incoming.is_empty() && !outgoing.is_empty() {
        (width as i32 / 3).max(8)
    } else if outgoing.is_empty() && !incoming.is_empty() {
        (width as i32 * 2 / 3).max(8)
    } else {
        (width as i32 / 2).max(8)
    };
    let focus_y = if incoming.is_empty() || outgoing.is_empty() {
        4i32.min(height.saturating_sub(3) as i32).max(2)
    } else {
        (height as i32 / 2).max(3)
    };
    let focus_pt = GraphPoint {
        x: focus_x,
        y: focus_y,
    };
    let left_x = ((focus_pt.x - 44).max(2)).min((focus_pt.x - 8).max(2));
    let right_x = ((focus_pt.x + 44).min(width as i32 - 28)).max(focus_pt.x + 8);
    let left_mid = (focus_pt.x - 8).max(5);
    let right_mid = (focus_pt.x + 8).min(width.saturating_sub(6) as i32);

    let max_neighbors = incoming.len().max(outgoing.len()) as i32;
    let band = (max_neighbors + 1).max(4).min((height as i32 / 4).max(6));
    let min_slot_y = (focus_pt.y - band).max(1);
    let max_slot_y = (focus_pt.y + band).min(height.saturating_sub(2) as i32);
    let left_slots = spread_slots(incoming.len(), min_slot_y, max_slot_y);
    let right_slots = spread_slots(outgoing.len(), min_slot_y, max_slot_y);

    for (idx, (_node, _w)) in incoming.iter().enumerate() {
        let y = left_slots.get(idx).copied().unwrap_or(focus_pt.y);
        draw_manhattan_edge(
            &mut canvas,
            focus_pt,
            GraphPoint { x: left_x + 1, y },
            left_mid,
        );
    }
    for (idx, (_node, _w)) in outgoing.iter().enumerate() {
        let y = right_slots.get(idx).copied().unwrap_or(focus_pt.y);
        draw_manhattan_edge(
            &mut canvas,
            focus_pt,
            GraphPoint {
                x: right_x.saturating_sub(2),
                y,
            },
            right_mid,
        );
    }

    let focus_tag = labels
        .get(focus)
        .cloned()
        .unwrap_or_else(|| "[*]".to_string());
    let focus_title = format!("{} {}", focus_tag, short_graph_label(focus, 18));
    canvas_put_text(
        &mut canvas,
        (focus_pt.x - 1).max(0),
        focus_pt.y,
        &focus_title,
    );

    for (idx, (node, w)) in incoming.iter().enumerate() {
        let y = left_slots.get(idx).copied().unwrap_or(focus_pt.y);
        let tag = labels
            .get(node)
            .cloned()
            .unwrap_or_else(|| "[??]".to_string());
        let txt = format!("{} {} ({:.2})", tag, short_graph_label(node, 16), w);
        canvas_put_text(&mut canvas, left_x, y, &txt);
    }
    for (idx, (node, w)) in outgoing.iter().enumerate() {
        let y = right_slots.get(idx).copied().unwrap_or(focus_pt.y);
        let tag = labels
            .get(node)
            .cloned()
            .unwrap_or_else(|| "[??]".to_string());
        let txt = format!("{} {} ({:.2})", tag, short_graph_label(node, 16), w);
        canvas_put_text(&mut canvas, right_x, y, &txt);
    }

    canvas
        .into_iter()
        .map(|row| row.into_iter().collect::<String>())
        .collect()
}

fn cluster_draw_data(
    focus: &str,
    visible_nodes: &[String],
    visible_edges: &[GraphTextEdge],
    include_cross_edges: bool,
) -> (Vec<String>, Vec<GraphTextEdge>) {
    let mut focus_edges: Vec<GraphTextEdge> = Vec::new();
    let mut non_focus_edges: Vec<GraphTextEdge> = Vec::new();
    for edge in visible_edges {
        if edge.source == focus || edge.target == focus {
            focus_edges.push(edge.clone());
        } else {
            non_focus_edges.push(edge.clone());
        }
    }
    focus_edges.sort_by(|a, b| b.weight.total_cmp(&a.weight));
    non_focus_edges.sort_by(|a, b| b.weight.total_cmp(&a.weight));

    let mut draw_edges: Vec<GraphTextEdge> = focus_edges.into_iter().take(16).collect();
    if include_cross_edges {
        draw_edges.extend(non_focus_edges.into_iter().take(8));
    }

    let mut draw_nodes: Vec<String> = Vec::new();
    let mut seen_nodes: HashSet<String> = HashSet::new();
    let push_node = |out: &mut Vec<String>, seen: &mut HashSet<String>, node: &str| {
        if seen.insert(node.to_string()) {
            out.push(node.to_string());
        }
    };
    push_node(&mut draw_nodes, &mut seen_nodes, focus);
    for edge in &draw_edges {
        push_node(&mut draw_nodes, &mut seen_nodes, &edge.source);
        push_node(&mut draw_nodes, &mut seen_nodes, &edge.target);
    }
    if draw_nodes.len() < 10 {
        for node in visible_nodes {
            push_node(&mut draw_nodes, &mut seen_nodes, node);
            if draw_nodes.len() >= 10 {
                break;
            }
        }
    }
    (draw_nodes, draw_edges)
}

fn render_graph_ascii(
    focus: &str,
    width: usize,
    height: usize,
    visible_nodes: &[String],
    visible_edges: &[GraphTextEdge],
    include_cross_edges: bool,
) -> Vec<String> {
    let mut canvas = vec![vec![' '; width]; height];
    if visible_nodes.is_empty() {
        return vec![String::new(); height];
    }
    let (draw_nodes, draw_edges) =
        cluster_draw_data(focus, visible_nodes, visible_edges, include_cross_edges);
    let positions = layout_cluster_positions(&draw_nodes, focus, width, height);

    for edge in &draw_edges {
        let Some(a) = positions.get(&edge.source) else {
            continue;
        };
        let Some(b) = positions.get(&edge.target) else {
            continue;
        };
        draw_graph_line(&mut canvas, *a, *b);
    }

    let labels = build_cluster_label_map(&draw_nodes, focus);
    let center_x = (width as i32) / 2;
    for node in &draw_nodes {
        let Some(pt) = positions.get(node) else {
            continue;
        };
        let token = labels
            .get(node)
            .cloned()
            .unwrap_or_else(|| "[??]".to_string());
        let token_len = token.chars().count() as i32;
        let token_x = (pt.x - (token_len / 2)).clamp(0, width.saturating_sub(1) as i32);
        canvas_put_text(&mut canvas, token_x, pt.y, &token);
        let label = short_graph_label(node, 12);
        let label_len = label.chars().count() as i32;
        let label_x = if pt.x <= center_x {
            pt.x + token_len / 2 + 2
        } else {
            (pt.x - token_len / 2 - label_len - 2).max(0)
        };
        canvas_put_text(&mut canvas, label_x, pt.y, &label);
    }

    canvas
        .into_iter()
        .map(|row| row.into_iter().collect::<String>())
        .collect()
}

#[derive(Default)]
struct GraphvizPlainLayout {
    width: f64,
    height: f64,
    node_positions: HashMap<String, (f64, f64)>,
}

fn render_graph_graphviz_plain(
    focus: &str,
    width: usize,
    height: usize,
    visible_nodes: &[String],
    visible_edges: &[GraphTextEdge],
    include_cross_edges: bool,
) -> Result<Vec<String>, String> {
    let (draw_nodes, draw_edges) =
        cluster_draw_data(focus, visible_nodes, visible_edges, include_cross_edges);
    if draw_nodes.is_empty() {
        return Ok(vec![String::new(); height]);
    }

    let mut node_ids: HashMap<String, String> = HashMap::new();
    let mut id_to_node: HashMap<String, String> = HashMap::new();
    let mut dot = String::from(
        "digraph G {\n\
         graph [overlap=false,splines=false,outputorder=edgesfirst,pad=0.12,nodesep=0.20,ranksep=0.24];\n\
         node [shape=box,style=rounded,fontname=\"Menlo\",fontsize=10];\n\
         edge [fontname=\"Menlo\",fontsize=8,dir=none];\n",
    );
    for (idx, node) in draw_nodes.iter().enumerate() {
        let id = format!("n{}", idx);
        node_ids.insert(node.clone(), id.clone());
        id_to_node.insert(id.clone(), node.clone());
        let mut label = short_graph_label(node, 22);
        if node == focus {
            label = format!("* {}", label);
        }
        dot.push_str(&format!("{} [label=\"{}\"];\n", id, dot_escape(&label)));
    }
    for edge in &draw_edges {
        let Some(src) = node_ids.get(&edge.source) else {
            continue;
        };
        let Some(dst) = node_ids.get(&edge.target) else {
            continue;
        };
        dot.push_str(&format!("{} -> {};\n", src, dst));
    }
    dot.push_str("}\n");

    let mut child = Command::new("dot")
        .arg("-Kneato")
        .arg("-Tplain")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed spawning dot: {}", e))?;
    if let Some(stdin) = child.stdin.as_mut() {
        stdin
            .write_all(dot.as_bytes())
            .map_err(|e| format!("failed writing graphviz input: {}", e))?;
    } else {
        return Err("dot stdin unavailable".to_string());
    }
    let output = child
        .wait_with_output()
        .map_err(|e| format!("failed reading dot output: {}", e))?;
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "dot exited with status {}: {}",
            output.status,
            clipped(err.trim(), 120)
        ));
    }
    let plain = String::from_utf8_lossy(&output.stdout);
    let layout = parse_graphviz_plain_layout(&plain)?;

    let mut canvas = vec![vec![' '; width]; height];
    for edge in &draw_edges {
        let Some(src_id) = node_ids.get(&edge.source) else {
            continue;
        };
        let Some(dst_id) = node_ids.get(&edge.target) else {
            continue;
        };
        let Some((sx, sy)) = layout.node_positions.get(src_id) else {
            continue;
        };
        let Some((dx, dy)) = layout.node_positions.get(dst_id) else {
            continue;
        };
        let a = map_graphviz_point((*sx, *sy), width, height, layout.width, layout.height);
        let b = map_graphviz_point((*dx, *dy), width, height, layout.width, layout.height);
        draw_graph_line(&mut canvas, a, b);
    }

    let labels = build_cluster_label_map(&draw_nodes, focus);
    let center_x = (width as i32) / 2;
    for (id, (x, y)) in &layout.node_positions {
        let Some(node) = id_to_node.get(id) else {
            continue;
        };
        let pt = map_graphviz_point((*x, *y), width, height, layout.width, layout.height);
        let token = labels
            .get(node)
            .cloned()
            .unwrap_or_else(|| "[??]".to_string());
        let token_len = token.chars().count() as i32;
        let token_x = (pt.x - (token_len / 2)).clamp(0, width.saturating_sub(1) as i32);
        canvas_put_text(&mut canvas, token_x, pt.y, &token);
        let label = short_graph_label(node, 16);
        let label_len = label.chars().count() as i32;
        let label_x = if pt.x <= center_x {
            pt.x + token_len / 2 + 2
        } else {
            (pt.x - token_len / 2 - label_len - 2).max(0)
        };
        canvas_put_text(&mut canvas, label_x, pt.y, &label);
    }

    Ok(canvas
        .into_iter()
        .map(|row| row.into_iter().collect::<String>())
        .collect())
}

fn parse_graphviz_plain_layout(raw: &str) -> Result<GraphvizPlainLayout, String> {
    let mut out = GraphvizPlainLayout::default();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        match parts[0] {
            "graph" if parts.len() >= 4 => {
                out.width = parts[2].parse::<f64>().unwrap_or(0.0);
                out.height = parts[3].parse::<f64>().unwrap_or(0.0);
            }
            "node" if parts.len() >= 4 => {
                let id = parts[1].to_string();
                let x = parts[2].parse::<f64>().unwrap_or(0.0);
                let y = parts[3].parse::<f64>().unwrap_or(0.0);
                out.node_positions.insert(id, (x, y));
            }
            _ => {}
        }
    }
    if out.node_positions.is_empty() {
        return Err("graphviz plain output had no node positions".to_string());
    }
    if out.width <= 0.0 || out.height <= 0.0 {
        out.width = 1.0;
        out.height = 1.0;
    }
    Ok(out)
}

fn map_graphviz_point(
    pt: (f64, f64),
    width: usize,
    height: usize,
    graph_width: f64,
    graph_height: f64,
) -> GraphPoint {
    let graph_width = graph_width.max(1.0);
    let graph_height = graph_height.max(1.0);
    let margin_x = 2i32;
    let margin_y = 1i32;
    let usable_w = width.saturating_sub(4).max(4) as f64;
    let usable_h = height.saturating_sub(3).max(3) as f64;
    let x = margin_x + ((pt.0 / graph_width) * usable_w).round() as i32;
    let y = margin_y + (((graph_height - pt.1) / graph_height) * usable_h).round() as i32;
    GraphPoint {
        x: x.clamp(0, width.saturating_sub(1) as i32),
        y: y.clamp(0, height.saturating_sub(1) as i32),
    }
}

fn dot_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' | '\r' => out.push(' '),
            _ => out.push(ch),
        }
    }
    out
}

fn layout_cluster_positions(
    nodes: &[String],
    focus: &str,
    width: usize,
    height: usize,
) -> HashMap<String, GraphPoint> {
    let mut out: HashMap<String, GraphPoint> = HashMap::new();
    if nodes.is_empty() {
        return out;
    }
    let cx = (width as i32 / 2).max(4);
    let cy = (height as i32 / 2).max(3);
    out.insert(focus.to_string(), GraphPoint { x: cx, y: cy });

    let others: Vec<&String> = nodes.iter().filter(|n| n.as_str() != focus).collect();
    if others.is_empty() {
        return out;
    }
    let rx = ((width as i32 / 2) - 12).max(8) as f64;
    let ry = ((height as i32 / 2) - 4).max(4) as f64;
    for (idx, node) in others.iter().enumerate() {
        let theta = -PI / 2.0 + ((idx as f64) * (2.0 * PI / others.len() as f64));
        let x = (cx as f64 + rx * theta.cos()).round() as i32;
        let y = (cy as f64 + ry * theta.sin()).round() as i32;
        out.insert(
            (*node).clone(),
            GraphPoint {
                x: x.clamp(1, width.saturating_sub(2) as i32),
                y: y.clamp(1, height.saturating_sub(2) as i32),
            },
        );
    }
    out
}

fn draw_graph_line(canvas: &mut [Vec<char>], a: GraphPoint, b: GraphPoint) {
    let mut x = a.x;
    let mut y = a.y;
    let dx = (b.x - a.x).abs();
    let sx = if a.x < b.x { 1 } else { -1 };
    let dy = -(b.y - a.y).abs();
    let sy = if a.y < b.y { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        if x == b.x && y == b.y {
            break;
        }
        let mut nx = x;
        let mut ny = y;
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            nx += sx;
        }
        if e2 <= dx {
            err += dx;
            ny += sy;
        }
        if !(nx == b.x && ny == b.y) && !(nx == a.x && ny == a.y) {
            let step_ch = if nx == x {
                '|'
            } else if ny == y {
                '-'
            } else if (nx - x) == (ny - y) {
                '\\'
            } else {
                '/'
            };
            canvas_put_char(canvas, nx, ny, step_ch, false);
        }
        x = nx;
        y = ny;
    }
}

fn draw_manhattan_edge(canvas: &mut [Vec<char>], from: GraphPoint, to: GraphPoint, mid_x: i32) {
    draw_hline(canvas, from.y, from.x, mid_x);
    draw_vline(canvas, mid_x, from.y, to.y);
    draw_hline(canvas, to.y, mid_x, to.x);
}

fn draw_hline(canvas: &mut [Vec<char>], y: i32, x0: i32, x1: i32) {
    if canvas.is_empty() {
        return;
    }
    let (a, b) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
    for x in a..=b {
        canvas_put_char(canvas, x, y, '-', false);
    }
}

fn draw_vline(canvas: &mut [Vec<char>], x: i32, y0: i32, y1: i32) {
    if canvas.is_empty() {
        return;
    }
    let (a, b) = if y0 <= y1 { (y0, y1) } else { (y1, y0) };
    for y in a..=b {
        canvas_put_char(canvas, x, y, '|', false);
    }
}

fn spread_slots(count: usize, min_y: i32, max_y: i32) -> Vec<i32> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![(min_y + max_y) / 2];
    }
    let span = (max_y - min_y).max(1) as f64;
    let mut out = Vec::with_capacity(count);
    for idx in 0..count {
        let t = (idx + 1) as f64 / (count + 1) as f64;
        let y = (min_y as f64 + span * t).round() as i32;
        out.push(y.clamp(min_y, max_y));
    }
    out
}

fn canvas_put_char(canvas: &mut [Vec<char>], x: i32, y: i32, ch: char, overwrite: bool) {
    if canvas.is_empty() {
        return;
    }
    let h = canvas.len() as i32;
    let w = canvas[0].len() as i32;
    if x < 0 || y < 0 || x >= w || y >= h {
        return;
    }
    let cell = &mut canvas[y as usize][x as usize];
    if overwrite {
        *cell = ch;
        return;
    }
    if *cell == ' ' {
        *cell = ch;
        return;
    }
    if *cell == ch {
        return;
    }
    if *cell == 'o' || *cell == '*' {
        return;
    }
    *cell = '+';
}

fn canvas_put_text(canvas: &mut [Vec<char>], x: i32, y: i32, text: &str) {
    if canvas.is_empty() {
        return;
    }
    let h = canvas.len() as i32;
    let w = canvas[0].len() as i32;
    if y < 0 || y >= h {
        return;
    }
    let mut cursor = x;
    for ch in text.chars() {
        if cursor >= 0 && cursor < w {
            canvas[y as usize][cursor as usize] = ch;
        }
        cursor += 1;
        if cursor >= w {
            break;
        }
    }
}

fn short_graph_label(path: &str, max_width: usize) -> String {
    clipped(&path_basename(path), max_width.max(4))
}

fn build_cluster_label_map(nodes: &[String], focus: &str) -> HashMap<String, String> {
    let mut map: HashMap<String, String> = HashMap::new();
    map.insert(focus.to_string(), "[*]".to_string());
    let mut idx = 1usize;
    for node in nodes {
        if node == focus {
            continue;
        }
        map.insert(node.clone(), format!("[{:02}]", idx));
        idx += 1;
    }
    map
}

fn cluster_node_legend_lines(
    focus: &str,
    nodes: &[String],
    width: usize,
    max_lines: usize,
) -> Vec<String> {
    if max_lines == 0 || nodes.is_empty() {
        return Vec::new();
    }
    let labels = build_cluster_label_map(nodes, focus);
    let mut lines = Vec::new();
    lines.push(clipped("node legend:", width));
    lines.push(clipped(
        &format!(
            "  [*] {}",
            short_graph_label(focus, width.saturating_sub(6))
        ),
        width,
    ));
    let mut count = 0usize;
    for node in nodes {
        if node == focus {
            continue;
        }
        let Some(tag) = labels.get(node) else {
            continue;
        };
        lines.push(clipped(
            &format!(
                "  {} {}",
                tag,
                short_graph_label(node, width.saturating_sub(8))
            ),
            width,
        ));
        count += 1;
        if count >= max_lines.saturating_sub(2) {
            break;
        }
    }
    lines
}

fn render_lineage_diagram(
    focus: &str,
    edges: &[GraphTextEdge],
    width: usize,
    height: usize,
) -> Vec<String> {
    let mut incoming: Vec<(f64, String)> = Vec::new();
    let mut outgoing: Vec<(f64, String)> = Vec::new();
    for edge in edges {
        if edge.target == focus {
            incoming.push((edge.weight, short_graph_label(&edge.source, 30)));
        } else if edge.source == focus {
            outgoing.push((edge.weight, short_graph_label(&edge.target, 30)));
        }
    }
    incoming.sort_by(|a, b| b.0.total_cmp(&a.0));
    outgoing.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut lines: Vec<String> = Vec::new();
    lines.push(clipped(
        "lineage lanes: incoming -> focus -> outgoing (filtered by threshold/depth)",
        width,
    ));
    lines.push(clipped(
        &format!("focus node: {}", short_graph_label(focus, 42)),
        width,
    ));
    lines.push(clipped(
        &format!("incoming={} outgoing={}", incoming.len(), outgoing.len()),
        width,
    ));

    let max_rows = height.saturating_sub(5).max(1);
    if incoming.is_empty() {
        for (idx, (_, node)) in outgoing.iter().take(max_rows).enumerate() {
            lines.push(clipped(
                &format!("  {:>2}. [*] -> {}", idx + 1, node),
                width,
            ));
        }
    } else if outgoing.is_empty() {
        for (idx, (_, node)) in incoming.iter().take(max_rows).enumerate() {
            lines.push(clipped(
                &format!("  {:>2}. {} -> [*]", idx + 1, node),
                width,
            ));
        }
    } else {
        let left_w = ((width.saturating_sub(16)) / 2).clamp(10, 32);
        let right_w = width.saturating_sub(left_w + 10).clamp(12, 44);
        let rows = incoming.len().max(outgoing.len()).max(1).min(max_rows);
        for idx in 0..rows {
            let left = incoming
                .get(idx)
                .map(|(_, s)| clipped(s, left_w))
                .unwrap_or_else(String::new);
            let right = outgoing
                .get(idx)
                .map(|(_, s)| clipped(s, right_w))
                .unwrap_or_else(String::new);
            lines.push(clipped(
                &format!("{:<left_w$} -> [*] -> {}", left, right, left_w = left_w),
                width,
            ));
        }
    }

    if lines.len() < height {
        lines.push(clipped(
            "hint: w/s threshold, [/] depth, t layout, v renderer",
            width,
        ));
    }
    while lines.len() < height {
        lines.push(String::new());
    }
    lines.truncate(height);
    lines
}

fn render_graph_matrix(
    focus: &str,
    nodes: &[String],
    edges: &[GraphTextEdge],
    width: usize,
    height: usize,
) -> Vec<String> {
    if nodes.is_empty() {
        return vec![String::new(); height];
    }
    let mut ordered: Vec<String> = Vec::new();
    ordered.push(focus.to_string());
    for node in nodes {
        if node != focus {
            ordered.push(node.clone());
        }
        if ordered.len() >= 10 {
            break;
        }
    }
    let mut edge_map: HashMap<(String, String), f64> = HashMap::new();
    for edge in edges {
        let k = (edge.source.clone(), edge.target.clone());
        let cur = edge_map.get(&k).copied().unwrap_or(0.0);
        if edge.weight > cur {
            edge_map.insert(k, edge.weight);
        }
    }
    let mut lines: Vec<String> = Vec::new();
    lines.push(clipped(
        "adjacency matrix: value = directed edge weight (source row -> target col)",
        width,
    ));
    let mut header = String::from("     ");
    for idx in 0..ordered.len() {
        header.push_str(&format!("{:>4}", idx));
    }
    lines.push(clipped(&header, width));
    for (row_idx, src) in ordered.iter().enumerate() {
        let mut row = format!("{:>3} ", row_idx);
        for dst in &ordered {
            let value = edge_map
                .get(&(src.clone(), dst.clone()))
                .copied()
                .unwrap_or(0.0);
            if src == dst {
                row.push_str("   .");
            } else if value <= 0.0 {
                row.push_str("   -");
            } else {
                row.push_str(&format!("{:>4}", (value * 10.0).round() as i32));
            }
        }
        lines.push(clipped(&row, width));
    }
    lines.push(clipped("legend:", width));
    for (idx, node) in ordered.iter().enumerate() {
        lines.push(clipped(
            &format!(
                "  {:>2}: {}",
                idx,
                short_graph_label(node, width.saturating_sub(8))
            ),
            width,
        ));
    }
    while lines.len() < height {
        lines.push(String::new());
    }
    lines.truncate(height);
    lines
}

#[derive(Debug)]
struct ShellCommandOutput {
    exit_code: i32,
    stdout: String,
    stderr: String,
}

fn run_shell_capture(command: &str) -> Result<ShellCommandOutput, String> {
    let output = Command::new("bash")
        .arg("-lc")
        .arg(command)
        .output()
        .map_err(|e| format!("failed running shell command '{}': {}", command, e))?;
    Ok(ShellCommandOutput {
        exit_code: output.status.code().unwrap_or(1),
        stdout: String::from_utf8_lossy(&output.stdout).trim().to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
    })
}

fn falkor_url() -> String {
    env::var("CYPRESS_FALKOR_URL").unwrap_or_else(|_| "falkor://127.0.0.1:6379".to_string())
}

fn falkor_graph_name() -> String {
    env::var("CYPRESS_FALKOR_GRAPH").unwrap_or_else(|_| "cypress".to_string())
}

fn falkor_debug_enabled() -> bool {
    env::var("CYPRESS_DEBUG_FALKOR")
        .ok()
        .map(|v| {
            let t = v.trim().to_lowercase();
            t == "1" || t == "true" || t == "yes" || t == "on"
        })
        .unwrap_or(false)
}

fn runtime_config_value(key: &str) -> Option<String> {
    let cwd = env::current_dir().ok()?;
    let map = load_config_values(&config_path(&cwd));
    map.get(key)
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn falkor_runtime_mode() -> String {
    let raw = env::var("CYPRESS_FALKOR_RUNTIME_MODE")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .or_else(|| runtime_config_value("falkor_runtime_mode"))
        .unwrap_or_else(|| "managed-local".to_string());
    let mode = raw.trim().to_lowercase();
    if matches!(mode.as_str(), "managed-local" | "docker" | "manual") {
        mode
    } else {
        "managed-local".to_string()
    }
}

fn falkor_docker_container() -> String {
    env::var("CYPRESS_FALKOR_DOCKER_CONTAINER")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .or_else(|| runtime_config_value("falkor_docker_container"))
        .unwrap_or_else(|| "cypress-falkordb".to_string())
}

fn falkor_endpoint() -> (String, u16) {
    let url = falkor_url();
    let trimmed = url.trim();
    let no_scheme = trimmed
        .strip_prefix("falkor://")
        .or_else(|| trimmed.strip_prefix("redis://"))
        .unwrap_or(trimmed);
    let host_port = no_scheme
        .split('/')
        .next()
        .unwrap_or("127.0.0.1:6379")
        .trim();
    let mut parts = host_port.split(':');
    let host = parts
        .next()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or("127.0.0.1")
        .trim()
        .to_string();
    let port = parts
        .next()
        .and_then(|v| v.trim().parse::<u16>().ok())
        .unwrap_or(6379);
    (host, port)
}

fn parse_leading_command_path(raw_cmd: &str) -> Option<PathBuf> {
    let trimmed = raw_cmd.trim();
    if trimmed.is_empty() {
        return None;
    }
    let token = if let Some(rest) = trimmed.strip_prefix('\'') {
        let end = rest.find('\'')?;
        rest[..end].to_string()
    } else if let Some(rest) = trimmed.strip_prefix('"') {
        let end = rest.find('"')?;
        rest[..end].to_string()
    } else {
        trimmed.split_whitespace().next()?.to_string()
    };
    if token.trim().is_empty() {
        return None;
    }
    let path = expand_tilde(token);
    if path.is_absolute() {
        Some(path)
    } else {
        resolve_command_path(&path.to_string_lossy())
    }
}

fn managed_local_cmd_is_usable(raw_cmd: &str) -> bool {
    let Some(path) = parse_leading_command_path(raw_cmd) else {
        return false;
    };
    if !is_executable_file(&path) {
        return false;
    }
    !is_shared_library_artifact(&path)
}

fn falkor_managed_local_cmd() -> Option<String> {
    if let Ok(raw) = env::var("CYPRESS_FALKOR_RUNTIME_CMD") {
        let cmd = raw.trim();
        if !cmd.is_empty() && managed_local_cmd_is_usable(cmd) {
            return Some(cmd.to_string());
        }
    }
    if let Some(cmd) = runtime_config_value("falkor_runtime_cmd") {
        if managed_local_cmd_is_usable(&cmd) {
            return Some(cmd);
        }
    }
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let state_bin = data_dir(&cwd).join("bin");
    let (_, port) = falkor_endpoint();
    let wrapper = state_bin.join("falkordb");
    if is_executable_file(&wrapper) {
        return Some(format!(
            "{} --port {}",
            shell_escape(&wrapper.to_string_lossy()),
            port
        ));
    }
    if let Some(path) = find_falkordb_binary_on_path() {
        return Some(format!(
            "{} --port {}",
            shell_escape(&path.to_string_lossy()),
            port
        ));
    }
    None
}

fn auto_bootstrap_falkor_runtime_if_unset() {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let map = load_config_values(&cfg_path);
    if map.contains_key("falkor_runtime_mode") {
        return;
    }
    let mut cfg = ConfigValues::from_map(map);
    if let Ok(cmd) = provision_falkordb_managed_local(false, false, false, None) {
        cfg.falkor_runtime_mode = "managed-local".to_string();
        cfg.falkor_runtime_cmd = cmd;
        if cfg.falkor_docker_container.trim().is_empty() {
            cfg.falkor_docker_container = "cypress-falkordb".to_string();
        }
        if let Err(e) = write_config_file(&cfg_path, &cfg) {
            eprintln!(
                "warning: failed to persist auto-bootstrapped Falkor runtime config: {}",
                e
            );
            return;
        }
        eprintln!("falkordb: auto-bootstrap configured runtime mode=managed-local");
        return;
    }
    cfg.falkor_runtime_mode = "manual".to_string();
    cfg.falkor_runtime_cmd.clear();
    if cfg.falkor_docker_container.trim().is_empty() {
        cfg.falkor_docker_container = "cypress-falkordb".to_string();
    }
    if let Err(e) = write_config_file(&cfg_path, &cfg) {
        eprintln!(
            "warning: failed to persist auto-bootstrapped Falkor runtime config: {}",
            e
        );
        return;
    }
    eprintln!("falkordb: auto-bootstrap configured runtime mode=manual");
}

fn graph_runtime_dir_path(cwd: &Path) -> PathBuf {
    data_dir(cwd).join("runtime").join("graph")
}

fn graph_runtime_log_path(cwd: &Path) -> PathBuf {
    graph_runtime_dir_path(cwd).join("falkordb.log")
}

fn graph_runtime_state_path(cwd: &Path) -> PathBuf {
    graph_runtime_dir_path(cwd).join("falkordb-runtime.json")
}

fn load_graph_runtime_state(cwd: &Path) -> Option<Value> {
    let path = graph_runtime_state_path(cwd);
    let raw = fs::read_to_string(path).ok()?;
    serde_json::from_str::<Value>(&raw).ok()
}

fn save_graph_runtime_state(
    cwd: &Path,
    mode: &str,
    pid: Option<u32>,
    command: Option<&str>,
    container: Option<&str>,
) -> Result<(), String> {
    let runtime_dir = graph_runtime_dir_path(cwd);
    fs::create_dir_all(&runtime_dir).map_err(|e| {
        format!(
            "failed to create graph runtime dir '{}': {}",
            runtime_dir.display(),
            e
        )
    })?;
    let mut obj = serde_json::Map::new();
    obj.insert("mode".to_string(), Value::String(mode.to_string()));
    obj.insert(
        "started_unix_s".to_string(),
        Value::from(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        ),
    );
    obj.insert("url".to_string(), Value::String(falkor_url()));
    obj.insert("graph".to_string(), Value::String(falkor_graph_name()));
    if let Some(pid) = pid {
        obj.insert("pid".to_string(), Value::from(pid));
    }
    if let Some(command) = command {
        obj.insert("command".to_string(), Value::String(command.to_string()));
    }
    if let Some(container) = container {
        obj.insert(
            "container".to_string(),
            Value::String(container.to_string()),
        );
    }
    let text = serde_json::to_string_pretty(&Value::Object(obj))
        .map_err(|e| format!("failed to serialize runtime state: {}", e))?;
    let path = graph_runtime_state_path(cwd);
    fs::write(&path, text)
        .map_err(|e| format!("failed to write runtime state '{}': {}", path.display(), e))
}

fn clear_graph_runtime_state(cwd: &Path) {
    let path = graph_runtime_state_path(cwd);
    let _ = fs::remove_file(path);
}

fn shell_words(input: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut chars = input.chars().peekable();
    let mut in_single = false;
    let mut in_double = false;

    while let Some(ch) = chars.next() {
        if in_single {
            if ch == '\'' {
                in_single = false;
            } else {
                cur.push(ch);
            }
            continue;
        }
        if in_double {
            match ch {
                '"' => in_double = false,
                '\\' => {
                    if let Some(next) = chars.next() {
                        cur.push(next);
                    }
                }
                _ => cur.push(ch),
            }
            continue;
        }
        match ch {
            '\'' => in_single = true,
            '"' => in_double = true,
            '\\' => {
                if let Some(next) = chars.next() {
                    cur.push(next);
                }
            }
            c if c.is_whitespace() => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            _ => cur.push(ch),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn extract_loadmodule_path(raw_cmd: &str) -> Option<String> {
    let words = shell_words(raw_cmd);
    for idx in 0..words.len() {
        let tok = words[idx].trim();
        if tok == "--loadmodule" {
            if idx + 1 < words.len() {
                let v = words[idx + 1].trim();
                if !v.is_empty() {
                    return Some(v.to_string());
                }
            }
            continue;
        }
        if let Some(v) = tok.strip_prefix("--loadmodule=") {
            let vv = v.trim();
            if !vv.is_empty() {
                return Some(vv.to_string());
            }
        }
    }
    None
}

fn listener_pid_for_port(port: u16) -> Option<u32> {
    if !command_exists("lsof") {
        return None;
    }
    let out = run_shell_capture(&format!(
        "lsof -nP -tiTCP:{} -sTCP:LISTEN 2>/dev/null | head -n1",
        port
    ))
    .ok()?;
    if out.exit_code != 0 {
        return None;
    }
    out.stdout
        .lines()
        .find_map(|line| line.trim().parse::<u32>().ok())
}

fn pid_command_line(pid: u32) -> Option<String> {
    let out = run_shell_capture(&format!("ps -p {} -o command= 2>/dev/null", pid)).ok()?;
    if out.exit_code != 0 {
        return None;
    }
    let cmd = out.stdout.trim();
    if cmd.is_empty() {
        None
    } else {
        Some(cmd.to_string())
    }
}

fn managed_local_pid_matches_config(pid: u32, configured_cmd: &str) -> bool {
    let running = match pid_command_line(pid) {
        Some(v) => v,
        None => return false,
    };

    if let Some(expected_path) = parse_leading_command_path(configured_cmd) {
        if let Some(expected_prog) = expected_path.file_name().and_then(|n| n.to_str()) {
            if !expected_prog.is_empty()
                && !running.contains(expected_prog)
                && !(expected_prog == "falkordb" && running.contains("falkordb-real"))
            {
                return false;
            }
        }
    }

    // Some daemons (notably redis-server) rewrite argv and hide loadmodule args in `ps` output.
    // So module-path matching is best-effort only and does not gate adoption.
    let _ = extract_loadmodule_path(configured_cmd).and_then(|module_path| {
        if running.contains(&module_path) {
            Some(())
        } else {
            Path::new(&module_path)
                .file_name()
                .and_then(|n| n.to_str())
                .filter(|name| !name.is_empty() && running.contains(name))
                .map(|_| ())
        }
    });

    true
}

fn discover_managed_local_listener(configured_cmd: &str) -> Option<u32> {
    let (_, port) = falkor_endpoint();
    let pid = listener_pid_for_port(port)?;
    if managed_local_pid_matches_config(pid, configured_cmd) {
        Some(pid)
    } else {
        None
    }
}

fn reconcile_managed_local_runtime_state() -> Option<u32> {
    if falkor_runtime_mode() != "managed-local" {
        return None;
    }
    let cmd = falkor_managed_local_cmd()?;
    let pid = discover_managed_local_listener(&cmd)?;
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let state_pid = load_graph_runtime_state(&cwd).and_then(|state| {
        if state.get("mode").and_then(Value::as_str) == Some("managed-local") {
            state.get("pid").and_then(Value::as_u64).map(|v| v as u32)
        } else {
            None
        }
    });
    if state_pid == Some(pid) {
        return Some(pid);
    }
    if save_graph_runtime_state(&cwd, "managed-local", Some(pid), Some(&cmd), None).is_ok() {
        println!(
            "falkordb: adopted running managed-local runtime (pid={})",
            pid
        );
        Some(pid)
    } else {
        None
    }
}

fn pid_is_alive(pid: u32) -> bool {
    let out = run_shell_capture(&format!("kill -0 {} >/dev/null 2>&1", pid));
    matches!(out, Ok(out) if out.exit_code == 0)
}

fn wait_for_falkor_up(timeout: Duration) -> Result<(), String> {
    let deadline = Instant::now() + timeout;
    let mut last_err = String::new();
    while Instant::now() < deadline {
        match falkor_store::ping() {
            Ok(_) => return Ok(()),
            Err(e) => {
                last_err = e;
                thread::sleep(Duration::from_millis(250));
            }
        }
    }
    if last_err.is_empty() {
        last_err = "timeout waiting for FalkorDB".to_string();
    }
    Err(last_err)
}

fn wait_for_falkor_down(timeout: Duration) -> Result<(), String> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if falkor_store::ping().is_err() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(250));
    }
    Err("timeout waiting for FalkorDB to stop".to_string())
}

fn start_falkordb_managed_local(wait_seconds: u64) -> Result<(), String> {
    let cmd = falkor_managed_local_cmd().ok_or_else(|| {
        "managed-local runtime needs a FalkorDB command; install `falkordb` or set CYPRESS_FALKOR_RUNTIME_CMD".to_string()
    })?;
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let runtime_dir = graph_runtime_dir_path(&cwd);
    fs::create_dir_all(&runtime_dir).map_err(|e| {
        format!(
            "failed to create graph runtime dir '{}': {}",
            runtime_dir.display(),
            e
        )
    })?;
    let log_path = graph_runtime_log_path(&cwd);
    let log_out = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| format!("failed to open log '{}': {}", log_path.display(), e))?;
    let log_err = log_out
        .try_clone()
        .map_err(|e| format!("failed to clone log handle '{}': {}", log_path.display(), e))?;
    println!("falkordb: starting managed-local runtime");
    println!("command: {}", cmd);
    println!("log: {}", log_path.display());
    let mut proc = Command::new("bash");
    proc.arg("-lc").arg(format!("exec {}", cmd));
    proc.stdout(Stdio::from(log_out));
    proc.stderr(Stdio::from(log_err));
    let child = proc
        .spawn()
        .map_err(|e| format!("failed to spawn managed-local runtime '{}': {}", cmd, e))?;
    let pid = child.id();
    drop(child);
    save_graph_runtime_state(&cwd, "managed-local", Some(pid), Some(&cmd), None)?;
    match wait_for_falkor_up(Duration::from_secs(wait_seconds.max(1))) {
        Ok(_) => {
            println!("falkordb: running (pid={})", pid);
            Ok(())
        }
        Err(e) => {
            let _ = run_shell_capture(&format!("kill {} >/dev/null 2>&1 || true", pid));
            clear_graph_runtime_state(&cwd);
            Err(format!(
                "managed-local runtime did not become ready: {} (log: {})",
                e,
                log_path.display()
            ))
        }
    }
}

fn stop_falkordb_managed_local(wait_seconds: u64) -> Result<(), String> {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let state_path = graph_runtime_state_path(&cwd);
    let managed_cmd = falkor_managed_local_cmd();

    let mut pid = if let Some(state) = load_graph_runtime_state(&cwd) {
        if state.get("mode").and_then(Value::as_str) != Some("managed-local") {
            return Err("runtime state exists but mode is not managed-local".to_string());
        }
        state
            .get("pid")
            .and_then(Value::as_u64)
            .ok_or_else(|| "runtime state does not contain a pid".to_string())? as u32
    } else if let Some(cmd) = managed_cmd.as_ref() {
        if let Some(found_pid) = discover_managed_local_listener(cmd) {
            let _ =
                save_graph_runtime_state(&cwd, "managed-local", Some(found_pid), Some(cmd), None);
            println!(
                "falkordb: adopted running managed-local runtime (pid={}) for stop",
                found_pid
            );
            found_pid
        } else {
            return Err(format!(
                "no managed-local runtime state found at '{}'; nothing to stop",
                state_path.display()
            ));
        }
    } else {
        return Err(format!(
            "no managed-local runtime state found at '{}'; nothing to stop",
            state_path.display()
        ));
    };

    if !pid_is_alive(pid) {
        clear_graph_runtime_state(&cwd);
        if let Some(cmd) = managed_cmd.as_ref() {
            if let Some(found_pid) = discover_managed_local_listener(cmd) {
                let _ = save_graph_runtime_state(
                    &cwd,
                    "managed-local",
                    Some(found_pid),
                    Some(cmd),
                    None,
                );
                println!(
                    "falkordb: runtime state pid {} stale; stopping discovered pid {}",
                    pid, found_pid
                );
                pid = found_pid;
            } else {
                println!(
                    "falkordb: managed-local pid {} is not alive; cleared stale runtime state",
                    pid
                );
                return Ok(());
            }
        } else {
            println!(
                "falkordb: managed-local pid {} is not alive; cleared stale runtime state",
                pid
            );
            return Ok(());
        }
    }
    println!("falkordb: stopping managed-local pid {}", pid);
    let out = run_shell_capture(&format!("kill {} >/dev/null 2>&1 || true", pid))?;
    if out.exit_code != 0 {
        return Err(format!(
            "failed to send stop signal to managed-local pid {}",
            pid
        ));
    }
    let deadline = Instant::now() + Duration::from_secs(wait_seconds.max(1));
    while Instant::now() < deadline {
        if !pid_is_alive(pid) {
            break;
        }
        thread::sleep(Duration::from_millis(250));
    }
    if pid_is_alive(pid) {
        let _ = run_shell_capture(&format!("kill -9 {} >/dev/null 2>&1 || true", pid));
    }
    let _ = wait_for_falkor_down(Duration::from_secs(2));
    clear_graph_runtime_state(&cwd);
    println!("falkordb: stopped managed-local runtime");
    Ok(())
}

fn docker_falkor_service_status() -> Option<String> {
    if !command_exists("docker") {
        return None;
    }
    let container = falkor_docker_container();
    let running = run_shell_capture(&format!(
        "docker ps --filter name=^/{}$ --format '{{{{.Status}}}}' 2>/dev/null",
        container
    ))
    .ok()?;
    if running.exit_code == 0 && !running.stdout.trim().is_empty() {
        return Some(format!("{} ({})", running.stdout.trim(), container));
    }
    let existing = run_shell_capture(&format!(
        "docker ps -a --filter name=^/{}$ --format '{{{{.Status}}}}' 2>/dev/null",
        container
    ))
    .ok()?;
    if existing.exit_code == 0 && !existing.stdout.trim().is_empty() {
        return Some(format!("{} ({})", existing.stdout.trim(), container));
    }
    None
}

fn start_falkordb_docker(wait_seconds: u64) -> Result<(), String> {
    if !command_exists("docker") {
        return Err("runtime mode=docker requires docker to be installed".to_string());
    }
    let (_, port) = falkor_endpoint();
    let container = falkor_docker_container();
    let cmd = format!(
        "docker start {c} >/dev/null 2>&1 || docker run -d --name {c} -p {p}:{p} falkordb/falkordb:latest",
        c = container,
        p = port
    );
    println!("falkordb: starting docker runtime via `{}`", cmd);
    let out = run_shell_capture(&cmd)?;
    if out.exit_code != 0 {
        let detail = if !out.stderr.is_empty() {
            out.stderr
        } else if !out.stdout.is_empty() {
            out.stdout
        } else {
            "unknown error".to_string()
        };
        return Err(format!("failed to start docker runtime: {}", detail));
    }
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    save_graph_runtime_state(&cwd, "docker", None, None, Some(&container))?;
    wait_for_falkor_up(Duration::from_secs(wait_seconds.max(1)))
        .map_err(|e| format!("docker runtime did not become ready: {}", e))?;
    println!("falkordb: running (docker container={})", container);
    Ok(())
}

fn stop_falkordb_docker(wait_seconds: u64) -> Result<(), String> {
    if !command_exists("docker") {
        return Err("runtime mode=docker requires docker to be installed".to_string());
    }
    let container = falkor_docker_container();
    let cmd = format!("docker stop {} >/dev/null 2>&1 || true", container);
    println!("falkordb: stopping docker runtime via `{}`", cmd);
    let out = run_shell_capture(&cmd)?;
    if out.exit_code != 0 {
        let detail = if !out.stderr.is_empty() {
            out.stderr
        } else if !out.stdout.is_empty() {
            out.stdout
        } else {
            "unknown error".to_string()
        };
        return Err(format!("failed to stop docker runtime: {}", detail));
    }
    let _ = wait_for_falkor_down(Duration::from_secs(wait_seconds.max(1)));
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    clear_graph_runtime_state(&cwd);
    println!("falkordb: stopped docker runtime");
    Ok(())
}

fn start_falkordb(wait_seconds: u64) -> Result<(), String> {
    if falkor_store::ping().is_ok() {
        if falkor_runtime_mode() == "managed-local" {
            let _ = reconcile_managed_local_runtime_state();
        }
        println!("falkordb: already running");
        return Ok(());
    }
    match falkor_runtime_mode().as_str() {
        "managed-local" => start_falkordb_managed_local(wait_seconds),
        "docker" => start_falkordb_docker(wait_seconds),
        "manual" => Err(
            "runtime mode=manual does not auto-start; start FalkorDB yourself and retry"
                .to_string(),
        ),
        other => Err(format!(
            "invalid CYPRESS_FALKOR_RUNTIME_MODE '{}'; expected managed-local|docker|manual",
            other
        )),
    }
}

fn stop_falkordb(wait_seconds: u64) -> Result<(), String> {
    match falkor_runtime_mode().as_str() {
        "managed-local" => stop_falkordb_managed_local(wait_seconds),
        "docker" => stop_falkordb_docker(wait_seconds),
        "manual" => Err(
            "runtime mode=manual does not auto-stop; stop FalkorDB with your own process manager"
                .to_string(),
        ),
        other => Err(format!(
            "invalid CYPRESS_FALKOR_RUNTIME_MODE '{}'; expected managed-local|docker|manual",
            other
        )),
    }
}

fn print_graph_status(doctor_mode: bool) -> bool {
    let url = falkor_url();
    let graph = falkor_graph_name();
    let mode = falkor_runtime_mode();
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let state_path = graph_runtime_state_path(&cwd);
    let log_path = graph_runtime_log_path(&cwd);

    println!("falkordb url: {}", url);
    println!("falkordb graph: {}", graph);
    println!("runtime mode: {}", mode);
    println!("runtime state: {}", state_path.display());
    println!("graph viewer url: http://127.0.0.1:8780/");
    if mode == "managed-local" {
        let cmd = falkor_managed_local_cmd().unwrap_or_else(|| "<none>".to_string());
        println!("managed-local command: {}", cmd);
        println!("managed-local log: {}", log_path.display());
    } else if mode == "docker" {
        println!("docker container: {}", falkor_docker_container());
        if let Some(service_status) = docker_falkor_service_status() {
            println!("docker status: {}", service_status);
        }
    } else if mode == "manual" {
        println!("manual mode: Cypress will not auto-start/stop FalkorDB");
    }
    if let Some(state) = load_graph_runtime_state(&cwd) {
        if let Some(mode) = state.get("mode").and_then(Value::as_str) {
            println!("runtime state mode: {}", mode);
        }
        if let Some(pid) = state.get("pid").and_then(Value::as_u64) {
            println!(
                "runtime state pid: {} (alive={})",
                pid,
                yes_no(pid_is_alive(pid as u32))
            );
        }
    }

    match falkor_store::ping() {
        Ok(_) => {
            println!("reachable: yes");
            true
        }
        Err(err) => {
            println!("reachable: no");
            println!("reason: {}", err);
            if doctor_mode {
                match mode.as_str() {
                    "managed-local" => {
                        if falkor_managed_local_cmd().is_none() {
                            println!("hint: install a `falkordb` binary or set CYPRESS_FALKOR_RUNTIME_CMD, then run `cypress graph start`");
                        } else {
                            println!("hint: run `cypress graph start`");
                        }
                    }
                    "docker" => {
                        if !command_exists("docker") {
                            println!("hint: install docker or switch to managed-local mode");
                        } else {
                            println!("hint: run `cypress graph start`");
                        }
                    }
                    "manual" => {
                        println!("hint: start FalkorDB with your own command, or set CYPRESS_FALKOR_RUNTIME_MODE=managed-local");
                    }
                    _ => println!(
                        "hint: set CYPRESS_FALKOR_RUNTIME_MODE to managed-local|docker|manual"
                    ),
                }
            }
            false
        }
    }
}

fn ensure_retrieval_backend_ready(
    cfg: &ConfigValues,
    auto_start: bool,
    context: &str,
) -> Result<(), String> {
    if cfg.retrieval_backend != "falkordb" {
        return Ok(());
    }
    auto_bootstrap_falkor_runtime_if_unset();
    match falkor_store::ping() {
        Ok(_) => {
            if falkor_runtime_mode() == "managed-local" {
                let _ = reconcile_managed_local_runtime_state();
            }
            return Ok(());
        }
        Err(original_err) => {
            if !auto_start {
                return Err(format!(
                    "FalkorDB is required for {} but unreachable: {}",
                    context, original_err
                ));
            }
            eprintln!(
                "falkordb: unreachable for {}, attempting auto-start...",
                context
            );
            let first_start = start_falkordb(12);
            if first_start.is_err()
                && falkor_runtime_mode() == "manual"
                && env::var("CYPRESS_FALKOR_RUNTIME_MODE")
                    .ok()
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty())
                    .is_none()
                && env::var("SEMCD_FALKOR_RUNTIME_MODE")
                    .ok()
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty())
                    .is_none()
            {
                eprintln!("falkordb: runtime mode=manual; attempting automatic upgrade to managed-local...");
                let runtime_cmd = if let Some(cmd) = falkor_managed_local_cmd() {
                    Some(cmd)
                } else {
                    provision_falkordb_managed_local(true, true, false, None).ok()
                };
                if let Some(runtime_cmd) = runtime_cmd {
                    match persist_managed_local_runtime_config(&runtime_cmd) {
                        Ok(cfg_path) => {
                            eprintln!(
                                "falkordb: switched runtime mode to managed-local (config: {})",
                                cfg_path.display()
                            );
                            if let Err(e) = start_falkordb(12) {
                                return Err(format!(
                                    "FalkorDB is required for {} and auto-start failed after switching to managed-local: {} (run `cypress graph doctor` / `cypress graph start`; mode={})",
                                    context,
                                    e,
                                    falkor_runtime_mode()
                                ));
                            }
                        }
                        Err(e) => {
                            return Err(format!(
                                "FalkorDB is required for {} and auto-start failed; could not persist managed-local runtime config: {}",
                                context, e
                            ));
                        }
                    }
                } else {
                    return Err(format!(
                        "FalkorDB is required for {} and auto-start failed in manual mode; no managed-local runtime could be provisioned (run `cypress install` or `cypress graph provision`, then retry)",
                        context
                    ));
                }
            } else if let Err(e) = first_start {
                return Err(format!(
                    "FalkorDB is required for {} and auto-start failed: {} (run `cypress graph doctor` / `cypress graph start`; mode={})",
                    context,
                    e,
                    falkor_runtime_mode()
                ));
            }
        }
    }
    falkor_store::ping().map_err(|e| {
        format!(
            "FalkorDB is required for {} but still unreachable after start: {}",
            context, e
        )
    })
}

fn run_init(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: cypress init [--root <path>] [--embed-backend <local-hash|ollama|local(alias)>] [--embed-model <id>] [--retrieval-backend <falkordb|sqlite>] [--local-embed-dim <n>]");
        return;
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let db_path = db_path(&cwd);

    let mut cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    let mut explicit_model = false;

    let mut i = 0usize;
    while i < args.len() {
        let arg = args[i].to_string_lossy().to_string();
        match arg.as_str() {
            "--root" => {
                i += 1;
                let v = arg_value(args, i, "--root");
                cfg.root = normalize_path(&v);
            }
            "--embed-backend" => {
                i += 1;
                let mut v = arg_value(args, i, "--embed-backend").to_lowercase();
                if v == "auto" {
                    v = "local-hash".to_string();
                }
                if !matches!(v.as_str(), "local" | "local-hash" | "ollama") {
                    eprintln!("error: invalid --embed-backend '{}'", v);
                    process::exit(2);
                }
                cfg.embed_backend = v;
            }
            "--embed-model" => {
                i += 1;
                cfg.embed_model = arg_value(args, i, "--embed-model");
                explicit_model = true;
            }
            "--retrieval-backend" => {
                i += 1;
                let raw = arg_value(args, i, "--retrieval-backend").to_lowercase();
                let v = if raw == "lancedb" {
                    "falkordb".to_string()
                } else {
                    raw
                };
                if !matches!(v.as_str(), "falkordb" | "sqlite") {
                    eprintln!("error: invalid --retrieval-backend '{}'", v);
                    process::exit(2);
                }
                cfg.retrieval_backend = v;
            }
            "--local-embed-dim" => {
                i += 1;
                let v = arg_value(args, i, "--local-embed-dim");
                let parsed = v.parse::<i64>().unwrap_or_else(|_| {
                    eprintln!("error: --local-embed-dim must be an integer");
                    process::exit(2);
                });
                cfg.local_embed_dim = parsed.max(64);
            }
            other => {
                eprintln!("error: unknown option '{}'", other);
                process::exit(2);
            }
        }
        i += 1;
    }

    if !explicit_model {
        if cfg.embed_backend == "ollama" {
            cfg.embed_model = "qwen3-embedding".to_string();
        } else if (cfg.embed_backend == "local" || cfg.embed_backend == "local-hash")
            && (cfg.embed_model.is_empty() || cfg.embed_model == "qwen3-embedding")
        {
            cfg.embed_model = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        }
    }

    write_config_file(&cfg_path, &cfg).unwrap_or_else(|e| {
        eprintln!("error: failed to write config: {}", e);
        process::exit(1);
    });

    ensure_db_schema(&db_path).unwrap_or_else(|e| {
        eprintln!("error: failed to initialize database: {}", e);
        process::exit(1);
    });
    ensure_tracked_root(&db_path, &cfg.root, now_ts()).unwrap_or_else(|e| {
        eprintln!("error: failed to register tracked root: {}", e);
        process::exit(1);
    });

    println!("config: {}", cfg_path.display());
    println!("database: {}", db_path.display());
    println!("root: {}", cfg.root.display());
    println!("embed_backend: {}", cfg.embed_backend);
    println!("embed_model: {}", cfg.embed_model);
    println!("retrieval_backend: {}", cfg.retrieval_backend);
    println!("local_embed_dim: {}", cfg.local_embed_dim);
}

fn run_install_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: cypress install [--no-system-install] [--no-download] [--no-shell-hook] [--legacy] [--venv <path>] [--bench]");
        println!("notes:");
        println!("  - sets up project-local Cypress state under ./.cypress");
        println!("  - provisions FalkorDB runtime mode/command in config");
        println!("  - attempts to install fswatch for low-latency event-driven watch");
        println!("  - installs shell hook into ~/.bashrc and ~/.zshrc by default");
        println!("  - native Rust api/mcp runtime is included by default");
        println!("  - use --legacy only if you explicitly want the Python compatibility runtime");
        return;
    }

    let mut legacy_venv: Option<String> = None;
    let mut legacy_bench = false;
    let mut run_legacy_install = false;
    let mut allow_system_install = true;
    let mut allow_download = true;
    let mut install_shell_hook = true;

    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        match s.as_str() {
            "--venv" => {
                i += 1;
                legacy_venv = Some(arg_value(args, i, "--venv"));
                run_legacy_install = true;
            }
            "--bench" => {
                legacy_bench = true;
                run_legacy_install = true;
            }
            "--legacy" => {
                run_legacy_install = true;
            }
            "--no-legacy" => {
                run_legacy_install = false;
            }
            "--no-system-install" => {
                allow_system_install = false;
            }
            "--no-download" => {
                allow_download = false;
            }
            "--no-shell-hook" => {
                install_shell_hook = false;
            }
            other => {
                eprintln!("error: unknown option '{}'", other);
                process::exit(2);
            }
        }
        i += 1;
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let dbp = db_path(&cwd);
    let state_dir = data_dir(&cwd);

    fs::create_dir_all(&state_dir).unwrap_or_else(|e| {
        eprintln!(
            "error: failed to create state dir '{}': {}",
            state_dir.display(),
            e
        );
        process::exit(1);
    });
    fs::create_dir_all(state_dir.join("bin")).unwrap_or_else(|e| {
        eprintln!("error: failed to create runtime bin dir: {}", e);
        process::exit(1);
    });
    fs::create_dir_all(graph_runtime_dir_path(&cwd)).unwrap_or_else(|e| {
        eprintln!("error: failed to create graph runtime dir: {}", e);
        process::exit(1);
    });

    let mut cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    ensure_db_schema(&dbp).unwrap_or_else(|e| {
        eprintln!("error: failed to initialize database: {}", e);
        process::exit(1);
    });
    ensure_tracked_root(&dbp, &cfg.root, now_ts()).unwrap_or_else(|e| {
        eprintln!("error: failed to register tracked root: {}", e);
        process::exit(1);
    });

    let mut notes: Vec<String> = Vec::new();
    let mut fswatch_ready = command_exists("fswatch");
    if !fswatch_ready {
        if allow_system_install {
            match try_install_fswatch_with_homebrew() {
                Ok(true) => {
                    fswatch_ready = true;
                    notes.push("fswatch installed; watch will use event-driven mode".to_string());
                }
                Ok(false) => {
                    notes
                        .push("fswatch not available; watch will use polling fallback".to_string());
                }
                Err(err) => {
                    notes.push(format!(
                        "fswatch install attempt failed (watch will use polling fallback): {}",
                        err
                    ));
                }
            }
        } else {
            notes.push(
                "fswatch missing and system install disabled; watch will use polling fallback"
                    .to_string(),
            );
        }
    } else {
        notes.push("fswatch detected; watch will use event-driven mode".to_string());
    }
    let mut mode = cfg.falkor_runtime_mode.clone();
    let mut runtime_cmd = cfg.falkor_runtime_cmd.clone();
    let mut mode_forced_by_env = false;
    let mut container = if cfg.falkor_docker_container.trim().is_empty() {
        "cypress-falkordb".to_string()
    } else {
        cfg.falkor_docker_container.clone()
    };

    if let Ok(raw) = env::var("CYPRESS_FALKOR_RUNTIME_MODE") {
        let v = raw.trim().to_lowercase();
        if matches!(v.as_str(), "managed-local" | "docker" | "manual") {
            mode = v;
            mode_forced_by_env = true;
            notes.push("runtime mode sourced from CYPRESS_FALKOR_RUNTIME_MODE".to_string());
        }
    }
    if let Ok(raw) = env::var("CYPRESS_FALKOR_RUNTIME_CMD") {
        let v = raw.trim().to_string();
        if !v.is_empty() {
            runtime_cmd = v;
            notes.push("runtime command sourced from CYPRESS_FALKOR_RUNTIME_CMD".to_string());
        }
    }
    if let Ok(raw) = env::var("CYPRESS_FALKOR_DOCKER_CONTAINER") {
        let v = raw.trim().to_string();
        if !v.is_empty() {
            container = v;
            notes.push("docker container sourced from CYPRESS_FALKOR_DOCKER_CONTAINER".to_string());
        }
    }

    if mode == "managed-local" {
        if runtime_cmd.trim().is_empty() {
            match provision_falkordb_managed_local(
                allow_system_install,
                allow_download,
                false,
                None,
            ) {
                Ok(cmd) => {
                    runtime_cmd = cmd;
                    notes.push("managed-local runtime provisioned in .cypress/bin".to_string());
                }
                Err(err) => {
                    notes.push(format!("managed-local provisioning unavailable: {}", err));
                }
            }
        }
        if runtime_cmd.trim().is_empty() {
            mode = "manual".to_string();
            notes.push("no local FalkorDB binary found; selected runtime mode=manual".to_string());
        }
    } else if mode == "docker" {
        if !mode_forced_by_env && runtime_cmd.trim().is_empty() {
            if let Ok(cmd) =
                provision_falkordb_managed_local(allow_system_install, allow_download, false, None)
            {
                mode = "managed-local".to_string();
                runtime_cmd = cmd;
                notes.push(
                    "managed-local runtime provisioned; switched runtime mode from docker"
                        .to_string(),
                );
            } else {
                mode = "manual".to_string();
                notes.push(
                    "docker mode is no longer auto-selected; switched runtime mode to manual"
                        .to_string(),
                );
            }
        }
        if mode == "docker" && !command_exists("docker") {
            if let Some(cmd) = detect_falkordb_local_cmd(&cwd) {
                mode = "managed-local".to_string();
                runtime_cmd = cmd;
                notes
                    .push("docker unavailable; switched runtime mode to managed-local".to_string());
            } else {
                mode = "manual".to_string();
                notes.push("docker unavailable; switched runtime mode to manual".to_string());
            }
        }
    }

    if mode != "managed-local" {
        runtime_cmd.clear();
    }
    cfg.falkor_runtime_mode = mode.clone();
    cfg.falkor_runtime_cmd = runtime_cmd.clone();
    cfg.falkor_docker_container = container.clone();

    write_config_file(&cfg_path, &cfg).unwrap_or_else(|e| {
        eprintln!("error: failed to write config: {}", e);
        process::exit(1);
    });

    println!("install: complete");
    println!("config: {}", cfg_path.display());
    println!("database: {}", dbp.display());
    println!("state_dir: {}", state_dir.display());
    println!("runtime_mode: {}", mode);
    println!(
        "watch_mode: {}",
        if fswatch_ready {
            "event-driven (fswatch)"
        } else {
            "polling fallback"
        }
    );
    if mode == "managed-local" {
        println!("runtime_cmd: {}", runtime_cmd);
    }
    if mode == "docker" {
        println!("docker_container: {}", container);
    }
    if mode == "manual" {
        println!("runtime_note: managed-local FalkorDB is not configured yet; run `cypress graph provision`");
    }
    for note in notes {
        println!("note: {}", note);
    }
    if install_shell_hook {
        match install_default_shell_hooks(&cwd) {
            Ok(messages) => {
                for line in messages {
                    println!("shell_hook: {}", line);
                }
            }
            Err(err) => {
                eprintln!("warning: shell hook setup failed: {}", err);
            }
        }
    } else {
        println!("shell_hook: skipped (--no-shell-hook)");
    }

    let legacy_venv_path = if let Some(raw) = &legacy_venv {
        expand_tilde(raw)
    } else {
        state_dir.join("venv")
    };
    let legacy_cmd_present = legacy_venv_path.join("bin").join("cypress").exists();
    let should_run_legacy =
        run_legacy_install && (legacy_bench || legacy_venv.is_some() || !legacy_cmd_present);

    if run_legacy_install && !legacy_bench && legacy_venv.is_none() && legacy_cmd_present {
        println!(
            "install: legacy runtime already present at {}; skipping legacy reinstall",
            legacy_venv_path.display()
        );
    }

    if should_run_legacy {
        println!("install: delegating to legacy installer for api/mcp runtime...");
        let mut bridge_args: Vec<OsString> = vec![OsString::from("install")];
        if legacy_bench {
            bridge_args.push(OsString::from("--bench"));
        }
        if let Some(path) = legacy_venv {
            bridge_args.push(OsString::from("--venv"));
            bridge_args.push(OsString::from(path));
        }
        let code = run_legacy_bridge_status(&bridge_args);
        if code != 0 {
            if legacy_bench {
                eprintln!("error: legacy installer failed with status {}", code);
                process::exit(code);
            }
            eprintln!(
                "warning: legacy installer failed with status {} (api/mcp bridge commands may be unavailable)",
                code
            );
        }
    } else {
        println!("install: legacy Python runtime not requested (native-only install complete)");
    }
}

fn ensure_shell_hook_in_rc_file(rc_path: &Path, shell_script: &Path) -> Result<bool, String> {
    let marker_start = "# >>> cypress shell >>>";
    let marker_end = "# <<< cypress shell <<<";
    let script = shell_script.to_string_lossy().to_string();
    let block = format!(
        "{start}\nif [ -f {script} ]; then\n  source {script}\nfi\n{end}\n",
        start = marker_start,
        script = shell_escape(&script),
        end = marker_end
    );
    let existing = fs::read_to_string(rc_path).unwrap_or_default();
    if existing.contains(marker_start) || existing.contains(&script) {
        return Ok(false);
    }
    let mut updated = existing;
    if !updated.is_empty() && !updated.ends_with('\n') {
        updated.push('\n');
    }
    if !updated.is_empty() {
        updated.push('\n');
    }
    updated.push_str(&block);
    fs::write(rc_path, updated).map_err(|e| {
        format!(
            "failed writing shell rc file '{}': {}",
            rc_path.display(),
            e
        )
    })?;
    Ok(true)
}

fn install_default_shell_hooks(cwd: &Path) -> Result<Vec<String>, String> {
    let repo = find_repo_root().unwrap_or_else(|| cwd.to_path_buf());
    let shell_script = repo.join("scripts").join("cypress-shell.sh");
    if !shell_script.exists() {
        return Err(format!(
            "shell integration script not found at '{}'",
            shell_script.display()
        ));
    }
    let home = env::var("HOME")
        .map(PathBuf::from)
        .map_err(|_| "HOME is not set; cannot install shell hooks".to_string())?;
    let rc_files = [home.join(".bashrc"), home.join(".zshrc")];
    let mut messages: Vec<String> = Vec::new();
    for rc in rc_files {
        match ensure_shell_hook_in_rc_file(&rc, &shell_script) {
            Ok(true) => messages.push(format!("installed -> {}", rc.display())),
            Ok(false) => messages.push(format!("already present -> {}", rc.display())),
            Err(err) => messages.push(format!("failed -> {} ({})", rc.display(), err)),
        }
    }
    messages.push("open a new shell (or source your rc file) to activate `cypress`".to_string());
    Ok(messages)
}

fn detect_falkordb_local_cmd(cwd: &Path) -> Option<String> {
    let (_, port) = falkor_endpoint();
    let state_bin = data_dir(cwd).join("bin");
    let wrapper = state_bin.join("falkordb");
    if is_executable_file(&wrapper) {
        return Some(format!(
            "{} --port {}",
            shell_escape(&wrapper.to_string_lossy()),
            port
        ));
    }
    if let Some(path) = find_falkordb_binary_on_path() {
        return Some(format!(
            "{} --port {}",
            shell_escape(&path.to_string_lossy()),
            port
        ));
    }
    None
}

fn resolve_command_path(name: &str) -> Option<PathBuf> {
    let out = run_shell_capture(&format!("command -v {}", shell_escape(name))).ok()?;
    if out.exit_code != 0 {
        return None;
    }
    let line = out.stdout.lines().next()?.trim();
    if line.is_empty() {
        return None;
    }
    let path = PathBuf::from(line);
    if !path.is_absolute() || !is_executable_file(&path) {
        return None;
    }
    Some(path)
}

fn find_falkordb_binary_on_path() -> Option<PathBuf> {
    for name in ["falkordb", "falkordb-server"] {
        if let Some(path) = resolve_command_path(name) {
            return Some(path);
        }
    }
    None
}

fn find_redis_server_binary_on_path() -> Option<PathBuf> {
    for name in ["redis-server", "valkey-server", "redis-stack-server"] {
        if let Some(path) = resolve_command_path(name) {
            return Some(path);
        }
    }
    None
}

fn known_falkordb_module_locations() -> Vec<PathBuf> {
    vec![
        PathBuf::from("/opt/homebrew/opt/falkordb/lib/falkordb.so"),
        PathBuf::from("/opt/homebrew/opt/falkordb/lib/falkordb.dylib"),
        PathBuf::from("/usr/local/opt/falkordb/lib/falkordb.so"),
        PathBuf::from("/usr/local/opt/falkordb/lib/falkordb.dylib"),
    ]
}

fn find_falkordb_module_path(state_bin: &Path) -> Option<PathBuf> {
    if let Ok(raw) = env::var("CYPRESS_FALKOR_MODULE_PATH") {
        let p = expand_tilde(raw);
        if p.is_file() {
            return Some(p);
        }
    }

    if state_bin.is_dir() {
        if let Ok(entries) = fs::read_dir(state_bin) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                let name = path
                    .file_name()
                    .and_then(|v| v.to_str())
                    .unwrap_or_default()
                    .to_lowercase();
                if !name.contains("falkordb") {
                    continue;
                }
                if name.ends_with(".so") || name.ends_with(".dylib") || name.ends_with(".dll") {
                    return Some(path);
                }
            }
        }
    }

    for path in known_falkordb_module_locations() {
        if path.is_file() {
            return Some(path);
        }
    }
    None
}

fn falkor_release_api_url() -> String {
    env::var("CYPRESS_FALKOR_RELEASE_API").unwrap_or_else(|_| {
        "https://api.github.com/repos/FalkorDB/FalkorDB/releases/latest".to_string()
    })
}

fn falkor_direct_download_url() -> Option<String> {
    env::var("CYPRESS_FALKOR_DOWNLOAD_URL")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn current_platform_tokens() -> (Vec<&'static str>, Vec<&'static str>) {
    let os_tokens = match env::consts::OS {
        "macos" => vec!["macos", "darwin", "osx"],
        "linux" => vec!["linux"],
        other => vec![other],
    };
    let arch_tokens = match env::consts::ARCH {
        "aarch64" => vec!["aarch64", "arm64"],
        "x86_64" => vec!["x86_64", "amd64"],
        other => vec![other],
    };
    (os_tokens, arch_tokens)
}

fn score_falkor_asset(name: &str, os_tokens: &[&str], arch_tokens: &[&str]) -> i64 {
    let n = name.to_lowercase();
    if !n.contains("falkordb") {
        return -1;
    }
    for bad in [".so", ".dylib", ".dll", ".a", ".o", "module", "plugin"] {
        if n.ends_with(bad) || n.contains(&format!("{}.", bad)) || n.contains(bad) {
            return -1;
        }
    }
    for bad in [
        "sha256",
        "checksum",
        ".sig",
        "signature",
        "source",
        "src",
        ".deb",
        ".rpm",
    ] {
        if n.contains(bad) {
            return -1;
        }
    }
    let mut score = 10i64;
    if os_tokens.iter().any(|t| n.contains(t)) {
        score += 40;
    } else {
        score -= 20;
    }
    if arch_tokens.iter().any(|t| n.contains(t)) || n.contains("universal") {
        score += 30;
    } else {
        score -= 10;
    }
    if n.ends_with(".tar.gz") || n.ends_with(".tgz") || n.ends_with(".zip") {
        score += 20;
    } else {
        score -= 10;
    }
    if n.contains("server") || n.contains("standalone") {
        score += 12;
    }
    if n.contains("debug") {
        score -= 8;
    }
    score
}

fn score_falkor_module_asset(name: &str, os_tokens: &[&str], arch_tokens: &[&str]) -> i64 {
    let n = name.to_lowercase();
    if !n.contains("falkordb") {
        return -1;
    }
    if !(n.ends_with(".so") || n.ends_with(".dylib") || n.ends_with(".dll")) {
        return -1;
    }
    for bad in ["sha256", "checksum", ".sig", "signature", "source", "src"] {
        if n.contains(bad) {
            return -1;
        }
    }
    let mut score = 10i64;
    if os_tokens.iter().any(|t| n.contains(t)) {
        score += 45;
    } else {
        score -= 25;
    }
    if arch_tokens.iter().any(|t| n.contains(t)) || n.contains("universal") {
        score += 35;
    } else {
        score -= 20;
    }
    if n.contains("debug") {
        score -= 8;
    }
    score
}

fn is_shared_library_artifact(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|v| v.to_str())
        .unwrap_or_default()
        .to_lowercase();
    if name.ends_with(".so")
        || name.ends_with(".dylib")
        || name.ends_with(".dll")
        || name.ends_with(".a")
        || name.ends_with(".o")
    {
        return true;
    }
    if command_exists("file") {
        if let Ok(out) =
            run_shell_capture(&format!("file {}", shell_escape(&path.to_string_lossy())))
        {
            if out.exit_code == 0 {
                let desc = format!("{} {}", out.stdout, out.stderr).to_lowercase();
                if desc.contains("shared library")
                    || desc.contains("dynamically linked shared library")
                {
                    return true;
                }
            }
        }
    }
    false
}

fn extract_archive_to_dir(archive_path: &Path, extract_dir: &Path) -> Result<(), String> {
    if extract_dir.exists() {
        fs::remove_dir_all(extract_dir).map_err(|e| {
            format!(
                "failed clearing extract dir '{}': {}",
                extract_dir.display(),
                e
            )
        })?;
    }
    fs::create_dir_all(extract_dir).map_err(|e| {
        format!(
            "failed creating extract dir '{}': {}",
            extract_dir.display(),
            e
        )
    })?;
    let file_name = archive_path
        .file_name()
        .and_then(|v| v.to_str())
        .unwrap_or_default()
        .to_lowercase();
    let cmd = if file_name.ends_with(".zip") {
        if !command_exists("unzip") {
            return Err("cannot extract .zip archive because `unzip` is not installed".to_string());
        }
        format!(
            "unzip -oq {} -d {}",
            shell_escape(&archive_path.to_string_lossy()),
            shell_escape(&extract_dir.to_string_lossy())
        )
    } else if file_name.ends_with(".tar.gz") || file_name.ends_with(".tgz") {
        format!(
            "tar -xzf {} -C {}",
            shell_escape(&archive_path.to_string_lossy()),
            shell_escape(&extract_dir.to_string_lossy())
        )
    } else if file_name.ends_with(".tar.xz") {
        format!(
            "tar -xJf {} -C {}",
            shell_escape(&archive_path.to_string_lossy()),
            shell_escape(&extract_dir.to_string_lossy())
        )
    } else {
        return Err(format!(
            "unsupported archive format '{}'",
            archive_path.display()
        ));
    };
    let out = run_shell_capture(&cmd)?;
    if out.exit_code != 0 {
        let detail = if !out.stderr.is_empty() {
            out.stderr
        } else if !out.stdout.is_empty() {
            out.stdout
        } else {
            "unknown extraction error".to_string()
        };
        return Err(format!("failed extracting archive: {}", detail));
    }
    Ok(())
}

fn find_falkordb_binary_in_tree(root: &Path, max_depth: usize) -> Option<PathBuf> {
    if !root.exists() {
        return None;
    }
    let mut stack: Vec<(PathBuf, usize)> = vec![(root.to_path_buf(), 0)];
    let mut candidates: Vec<PathBuf> = Vec::new();
    while let Some((dir, depth)) = stack.pop() {
        if depth > max_depth {
            continue;
        }
        let entries = fs::read_dir(&dir).ok()?;
        for entry in entries {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.is_dir() {
                stack.push((path, depth + 1));
                continue;
            }
            let name = path
                .file_name()
                .and_then(|v| v.to_str())
                .unwrap_or_default()
                .to_lowercase();
            if name == "falkordb" || name == "falkordb-server" {
                candidates.push(path);
            }
        }
    }
    candidates.sort();
    candidates.into_iter().next()
}

fn ensure_unix_executable(path: &Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        let mut perms = fs::metadata(path)
            .map_err(|e| format!("failed reading metadata '{}': {}", path.display(), e))?
            .permissions();
        perms.set_mode(0o755);
        fs::set_permissions(path, perms).map_err(|e| {
            format!(
                "failed setting executable permission '{}': {}",
                path.display(),
                e
            )
        })?;
    }
    Ok(())
}

fn download_falkordb_binary_from_release(cwd: &Path, state_bin: &Path) -> Result<PathBuf, String> {
    let (asset_name, download_url) = if let Some(url) = falkor_direct_download_url() {
        (
            url.rsplit('/')
                .next()
                .unwrap_or("falkordb-download.bin")
                .to_string(),
            url,
        )
    } else {
        let api_url = falkor_release_api_url();
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(20))
            .build();
        let response = agent
            .get(&api_url)
            .set("User-Agent", "cypress-cli")
            .set("Accept", "application/vnd.github+json")
            .call()
            .map_err(|e| format!("failed fetching release metadata '{}': {}", api_url, e))?;
        let mut raw = String::new();
        response
            .into_reader()
            .read_to_string(&mut raw)
            .map_err(|e| format!("failed reading release metadata body: {}", e))?;
        let json: Value = serde_json::from_str(&raw)
            .map_err(|e| format!("failed parsing release metadata JSON: {}", e))?;
        let assets = json
            .get("assets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| "release metadata did not include assets".to_string())?;
        let (os_tokens, arch_tokens) = current_platform_tokens();
        let mut best: Option<(i64, String, String)> = None;
        for asset in assets {
            let Some(name) = asset.get("name").and_then(|v| v.as_str()) else {
                continue;
            };
            let Some(url) = asset.get("browser_download_url").and_then(|v| v.as_str()) else {
                continue;
            };
            let score = score_falkor_asset(name, &os_tokens, &arch_tokens);
            if score < 0 {
                continue;
            }
            match &best {
                Some((best_score, _, _)) if *best_score >= score => {}
                _ => best = Some((score, name.to_string(), url.to_string())),
            }
        }
        let Some((_, name, url)) = best else {
            return Err("failed to select a FalkorDB release asset for this platform".to_string());
        };
        (name, url)
    };

    let downloads_dir = data_dir(cwd).join("downloads");
    fs::create_dir_all(&downloads_dir).map_err(|e| {
        format!(
            "failed creating downloads dir '{}': {}",
            downloads_dir.display(),
            e
        )
    })?;
    let archive_path = downloads_dir.join(&asset_name);

    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(60))
        .build();
    let response = agent
        .get(&download_url)
        .set("User-Agent", "cypress-cli")
        .call()
        .map_err(|e| format!("failed downloading release asset '{}': {}", download_url, e))?;
    let mut out = fs::File::create(&archive_path).map_err(|e| {
        format!(
            "failed creating download file '{}': {}",
            archive_path.display(),
            e
        )
    })?;
    let mut reader = response.into_reader();
    std::io::copy(&mut reader, &mut out).map_err(|e| {
        format!(
            "failed writing downloaded asset '{}': {}",
            archive_path.display(),
            e
        )
    })?;

    let binary_path = if asset_name.to_lowercase().ends_with(".tar.gz")
        || asset_name.to_lowercase().ends_with(".tgz")
        || asset_name.to_lowercase().ends_with(".zip")
        || asset_name.to_lowercase().ends_with(".tar.xz")
    {
        let extract_dir = downloads_dir.join("extract");
        extract_archive_to_dir(&archive_path, &extract_dir)?;
        find_falkordb_binary_in_tree(&extract_dir, 8).ok_or_else(|| {
            format!(
                "downloaded release did not contain a `falkordb` binary (asset='{}')",
                asset_name
            )
        })?
    } else {
        archive_path.clone()
    };

    if is_shared_library_artifact(&binary_path) {
        return Err(format!(
            "selected release artifact is a shared library, not an executable server binary: {}",
            binary_path.display()
        ));
    }

    let installed = state_bin.join("falkordb-real");
    fs::copy(&binary_path, &installed).map_err(|e| {
        format!(
            "failed installing FalkorDB binary to '{}': {}",
            installed.display(),
            e
        )
    })?;
    ensure_unix_executable(&installed)?;
    Ok(installed)
}

fn download_falkordb_module_from_release(cwd: &Path, state_bin: &Path) -> Result<PathBuf, String> {
    let api_url = falkor_release_api_url();
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(20))
        .build();
    let response = agent
        .get(&api_url)
        .set("User-Agent", "cypress-cli")
        .set("Accept", "application/vnd.github+json")
        .call()
        .map_err(|e| format!("failed fetching release metadata '{}': {}", api_url, e))?;
    let mut raw = String::new();
    response
        .into_reader()
        .read_to_string(&mut raw)
        .map_err(|e| format!("failed reading release metadata body: {}", e))?;
    let json: Value = serde_json::from_str(&raw)
        .map_err(|e| format!("failed parsing release metadata JSON: {}", e))?;
    let assets = json
        .get("assets")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "release metadata did not include assets".to_string())?;
    let (os_tokens, arch_tokens) = current_platform_tokens();
    let mut best: Option<(i64, String, String)> = None;
    for asset in assets {
        let Some(name) = asset.get("name").and_then(|v| v.as_str()) else {
            continue;
        };
        let Some(url) = asset.get("browser_download_url").and_then(|v| v.as_str()) else {
            continue;
        };
        let score = score_falkor_module_asset(name, &os_tokens, &arch_tokens);
        if score < 0 {
            continue;
        }
        match &best {
            Some((best_score, _, _)) if *best_score >= score => {}
            _ => best = Some((score, name.to_string(), url.to_string())),
        }
    }
    let Some((_, asset_name, download_url)) = best else {
        return Err("failed to select a FalkorDB module asset for this platform".to_string());
    };

    let downloads_dir = data_dir(cwd).join("downloads");
    fs::create_dir_all(&downloads_dir).map_err(|e| {
        format!(
            "failed creating downloads dir '{}': {}",
            downloads_dir.display(),
            e
        )
    })?;
    let module_download_path = downloads_dir.join(&asset_name);
    let response = agent
        .get(&download_url)
        .set("User-Agent", "cypress-cli")
        .call()
        .map_err(|e| format!("failed downloading module asset '{}': {}", download_url, e))?;
    let mut out = fs::File::create(&module_download_path).map_err(|e| {
        format!(
            "failed creating downloaded module file '{}': {}",
            module_download_path.display(),
            e
        )
    })?;
    let mut reader = response.into_reader();
    std::io::copy(&mut reader, &mut out).map_err(|e| {
        format!(
            "failed writing downloaded module '{}': {}",
            module_download_path.display(),
            e
        )
    })?;

    let ext = if asset_name.to_lowercase().ends_with(".dylib") {
        ".dylib"
    } else if asset_name.to_lowercase().ends_with(".dll") {
        ".dll"
    } else {
        ".so"
    };
    let installed = state_bin.join(format!("falkordb-module{}", ext));
    fs::copy(&module_download_path, &installed).map_err(|e| {
        format!(
            "failed installing FalkorDB module to '{}': {}",
            installed.display(),
            e
        )
    })?;
    ensure_unix_executable(&installed)?;
    Ok(installed)
}

fn write_runtime_wrapper(wrapper_path: &Path, target_path: &Path) -> Result<(), String> {
    if let Some(parent) = wrapper_path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            format!(
                "failed creating runtime bin dir '{}': {}",
                parent.display(),
                e
            )
        })?;
    }
    let script = format!(
        "#!/usr/bin/env bash\nset -euo pipefail\nexec {} \"$@\"\n",
        shell_escape(&target_path.to_string_lossy())
    );
    fs::write(wrapper_path, script)
        .map_err(|e| format!("failed writing wrapper '{}': {}", wrapper_path.display(), e))?;
    #[cfg(unix)]
    {
        let mut perms = fs::metadata(wrapper_path)
            .map_err(|e| {
                format!(
                    "failed reading wrapper metadata '{}': {}",
                    wrapper_path.display(),
                    e
                )
            })?
            .permissions();
        perms.set_mode(0o755);
        fs::set_permissions(wrapper_path, perms).map_err(|e| {
            format!(
                "failed setting wrapper permissions '{}': {}",
                wrapper_path.display(),
                e
            )
        })?;
    }
    Ok(())
}

fn provision_falkordb_managed_local(
    allow_system_install: bool,
    allow_download: bool,
    force: bool,
    preferred_source: Option<PathBuf>,
) -> Result<String, String> {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let state_bin = data_dir(&cwd).join("bin");
    fs::create_dir_all(&state_bin).map_err(|e| {
        format!(
            "failed creating runtime bin dir '{}': {}",
            state_bin.display(),
            e
        )
    })?;
    let wrapper = state_bin.join("falkordb");
    let installed_real = state_bin.join("falkordb-real");
    let (_, port) = falkor_endpoint();

    if is_executable_file(&wrapper) && !force {
        if is_shared_library_artifact(&installed_real) {
            let _ = fs::remove_file(&wrapper);
            let _ = fs::remove_file(&installed_real);
        } else {
            return Ok(format!(
                "{} --port {}",
                shell_escape(&wrapper.to_string_lossy()),
                port
            ));
        }
    }

    if is_executable_file(&installed_real) && !force {
        if is_shared_library_artifact(&installed_real) {
            let _ = fs::remove_file(&installed_real);
        } else {
            write_runtime_wrapper(&wrapper, &installed_real)?;
            return Ok(format!(
                "{} --port {}",
                shell_escape(&wrapper.to_string_lossy()),
                port
            ));
        }
    }

    let mut source_binary = if let Some(path) = preferred_source {
        if !is_executable_file(&path) {
            return Err(format!(
                "provided --from path is not an executable file: {}",
                path.display()
            ));
        }
        Some(path)
    } else {
        find_falkordb_binary_on_path()
    };
    if let Some(path) = source_binary.as_ref() {
        if is_shared_library_artifact(path) {
            source_binary = None;
        }
    }
    let mut download_error = String::new();
    if source_binary.is_none() && allow_download {
        match download_falkordb_binary_from_release(&cwd, &state_bin) {
            Ok(path) => {
                if is_shared_library_artifact(&path) {
                    download_error = format!(
                        "downloaded artifact is a shared library, not a server executable: {}",
                        path.display()
                    );
                } else {
                    source_binary = Some(path);
                }
            }
            Err(err) => download_error = err,
        }
    }
    let mut brew_error = String::new();
    if source_binary.is_none() && allow_system_install {
        let installed = try_install_falkordb_with_homebrew()?;
        if installed {
            source_binary = find_falkordb_binary_on_path().or_else(|| {
                let real = state_bin.join("falkordb-real");
                if is_executable_file(&real) {
                    Some(real)
                } else {
                    None
                }
            });
            if let Some(path) = source_binary.as_ref() {
                if is_shared_library_artifact(path) {
                    source_binary = None;
                    brew_error = "homebrew provided a shared library artifact, not a server binary"
                        .to_string();
                }
            }
        } else {
            brew_error = "homebrew install path did not produce a usable binary".to_string();
        }
    }

    if let Some(source_binary) = source_binary {
        if normalize_lexical(&source_binary) != normalize_lexical(&wrapper) {
            write_runtime_wrapper(&wrapper, &source_binary)?;
        }
        return Ok(format!(
            "{} --port {}",
            shell_escape(&wrapper.to_string_lossy()),
            port
        ));
    }

    let mut redis_bin = find_redis_server_binary_on_path();
    let mut redis_error = String::new();
    if redis_bin.is_none() && allow_system_install {
        let installed = try_install_redis_with_homebrew()?;
        if installed {
            redis_bin = find_redis_server_binary_on_path();
        }
    }
    if redis_bin.is_none() {
        redis_error = "redis-server is not available".to_string();
    }

    let mut module_path = find_falkordb_module_path(&state_bin);
    let mut module_error = String::new();
    if module_path.is_none() && allow_download {
        match download_falkordb_module_from_release(&cwd, &state_bin) {
            Ok(path) => module_path = Some(path),
            Err(err) => module_error = err,
        }
    }
    if module_path.is_none() && allow_system_install {
        let _ = try_install_falkordb_with_homebrew();
        module_path = find_falkordb_module_path(&state_bin);
    }
    if let (Some(redis_bin), Some(module_path)) = (redis_bin.clone(), module_path.clone()) {
        let _ = ensure_unix_executable(&module_path);
        let runtime_cmd = format!(
            "{} --loadmodule {} --port {} --save '' --appendonly no",
            shell_escape(&redis_bin.to_string_lossy()),
            shell_escape(&module_path.to_string_lossy()),
            port
        );
        return Ok(runtime_cmd);
    }

    let mut details: Vec<String> = vec![
        "unable to provision managed-local FalkorDB runtime".to_string(),
        "no standalone `falkordb` server binary found".to_string(),
    ];
    if allow_download && !download_error.is_empty() {
        details.push(format!(
            "binary download attempt failed: {}",
            download_error
        ));
    }
    if allow_system_install && !brew_error.is_empty() {
        details.push(format!("system install attempt failed: {}", brew_error));
    }
    if !redis_error.is_empty() {
        details.push(redis_error);
    }
    if allow_download && !module_error.is_empty() {
        details.push(format!("module download attempt failed: {}", module_error));
    }
    details.push(
        "you can also pin an explicit runtime command via CYPRESS_FALKOR_RUNTIME_CMD".to_string(),
    );
    Err(details.join("; "))
}

fn persist_managed_local_runtime_config(runtime_cmd: &str) -> Result<PathBuf, String> {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let mut cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    cfg.falkor_runtime_mode = "managed-local".to_string();
    cfg.falkor_runtime_cmd = runtime_cmd.trim().to_string();
    if cfg.falkor_docker_container.trim().is_empty() {
        cfg.falkor_docker_container = "cypress-falkordb".to_string();
    }
    write_config_file(&cfg_path, &cfg)?;
    Ok(cfg_path)
}

fn try_install_falkordb_with_homebrew() -> Result<bool, String> {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let state_bin = data_dir(&cwd).join("bin");
    if find_falkordb_binary_on_path().is_some() || find_falkordb_module_path(&state_bin).is_some() {
        return Ok(true);
    }
    if !command_exists("brew") {
        return Ok(false);
    }
    println!("install: attempting Homebrew install for FalkorDB...");
    let out = run_shell_capture(
        "export HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_ENV_HINTS=1 GIT_TERMINAL_PROMPT=0; \
         brew install falkordb >/dev/null 2>&1 || brew install falkordb/falkordb/falkordb >/dev/null 2>&1",
    )?;
    if out.exit_code != 0 {
        return Ok(false);
    }
    Ok(command_exists("falkordb")
        || command_exists("falkordb-server")
        || find_falkordb_module_path(&state_bin).is_some())
}

fn try_install_redis_with_homebrew() -> Result<bool, String> {
    if find_redis_server_binary_on_path().is_some() {
        return Ok(true);
    }
    if !command_exists("brew") {
        return Ok(false);
    }
    println!("install: attempting Homebrew install for Redis...");
    let out = run_shell_capture("brew install redis >/dev/null 2>&1")?;
    if out.exit_code != 0 {
        return Ok(false);
    }
    Ok(find_redis_server_binary_on_path().is_some())
}

fn try_install_fswatch_with_homebrew() -> Result<bool, String> {
    if command_exists("fswatch") {
        return Ok(true);
    }
    if !command_exists("brew") {
        return Ok(false);
    }
    println!("install: attempting Homebrew install for fswatch...");
    let out = run_shell_capture("brew install fswatch >/dev/null 2>&1")?;
    if out.exit_code != 0 {
        return Ok(false);
    }
    Ok(command_exists("fswatch"))
}

fn run_add(args: &[OsString]) {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: cypress add <path> [path ...] [--refresh|--no-refresh]");
        return;
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let db_path = db_path(&cwd);
    let cfg = ConfigValues::from_map(load_config_values(&cfg_path));

    ensure_db_schema(&db_path).unwrap_or_else(|e| {
        eprintln!("error: failed to initialize database: {}", e);
        process::exit(1);
    });

    let mut refresh = true;
    let mut inputs: Vec<String> = Vec::new();
    for raw in args {
        let s = raw.to_string_lossy().to_string();
        match s.as_str() {
            "--refresh" => refresh = true,
            "--no-refresh" => refresh = false,
            x if x.starts_with('-') => {
                eprintln!("error: unknown option '{}'", x);
                process::exit(2);
            }
            _ => inputs.push(s),
        }
    }

    if inputs.is_empty() {
        eprintln!("error: no paths provided");
        process::exit(2);
    }

    let mut added: Vec<PathBuf> = Vec::new();
    for raw in inputs {
        let path = normalize_path(&raw);
        if !path.is_dir() {
            eprintln!("skip (not a directory): {}", path.display());
            continue;
        }
        ensure_tracked_root(&db_path, &path, now_ts()).unwrap_or_else(|e| {
            eprintln!("error: failed to add root '{}': {}", path.display(), e);
            process::exit(1);
        });
        added.push(path);
    }

    if added.is_empty() {
        eprintln!("error: no valid directories were added.");
        process::exit(1);
    }

    println!("tracked roots added:");
    for p in &added {
        println!("- {}", p.display());
    }

    if refresh {
        let force_paths: HashSet<PathBuf> = added.iter().cloned().collect();
        run_index_with_strategy(
            &cwd,
            &cfg,
            Some(added.clone()),
            true,
            Some(force_paths),
            false,
            "add refresh",
        )
        .unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            process::exit(1);
        });
    } else {
        let _ = cfg; // preserve load parity; currently unused when refresh disabled.
    }
}

fn run_del(args: &[OsString]) {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: cypress del <path> [path ...] [--refresh|--no-refresh]");
        return;
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let db_path = db_path(&cwd);
    let _cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    ensure_db_schema(&db_path).unwrap_or_else(|e| {
        eprintln!("error: failed to initialize database: {}", e);
        process::exit(1);
    });

    let mut refresh = true;
    let mut inputs: Vec<String> = Vec::new();
    for raw in args {
        let s = raw.to_string_lossy().to_string();
        match s.as_str() {
            "--refresh" => refresh = true,
            "--no-refresh" => refresh = false,
            x if x.starts_with('-') => {
                eprintln!("error: unknown option '{}'", x);
                process::exit(2);
            }
            _ => inputs.push(s),
        }
    }

    if inputs.is_empty() {
        eprintln!("error: no paths provided");
        process::exit(2);
    }

    let mut removed = 0i64;
    for raw in inputs {
        let path = normalize_path(&raw);
        removed += remove_tracked_root(&db_path, &path).unwrap_or_else(|e| {
            eprintln!("error: failed to remove root '{}': {}", path.display(), e);
            process::exit(1);
        });
    }
    println!("tracked roots removed: {}", removed);

    if refresh {
        run_index_with_strategy(&cwd, &_cfg, None, false, None, true, "delete refresh")
            .unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
    }
}

fn run_roots() {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let db_path = db_path(&cwd);
    let cfg = ConfigValues::from_map(load_config_values(&cfg_path));

    ensure_db_schema(&db_path).unwrap_or_else(|e| {
        eprintln!("error: failed to initialize database: {}", e);
        process::exit(1);
    });

    let mut rows = list_tracked_roots(&db_path).unwrap_or_else(|e| {
        eprintln!("error: failed to list tracked roots: {}", e);
        process::exit(1);
    });
    if rows.is_empty() {
        ensure_tracked_root(&db_path, &cfg.root, now_ts()).unwrap_or_else(|e| {
            eprintln!("error: failed to ensure default tracked root: {}", e);
            process::exit(1);
        });
        rows = list_tracked_roots(&db_path).unwrap_or_default();
    }

    println!("tracked roots: {}", rows.len());
    for p in rows {
        println!("- {}", p.display());
    }
}

fn run_index_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: cypress index");
        return;
    }
    if !args.is_empty() {
        let other = args[0].to_string_lossy();
        eprintln!("error: unknown argument '{}'", other);
        process::exit(2);
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    run_index_with_strategy(&cwd, &cfg, None, false, None, true, "index").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
}

fn run_refresh_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: cypress refresh [path ...]");
        return;
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let cfg = ConfigValues::from_map(load_config_values(&cfg_path));

    let mut scoped: Vec<PathBuf> = Vec::new();
    for raw in args {
        let s = raw.to_string_lossy().to_string();
        if s.starts_with('-') {
            eprintln!("error: unknown option '{}'", s);
            process::exit(2);
        }
        let p = normalize_path(&s);
        if !p.is_dir() {
            eprintln!("skip (not a directory): {}", p.display());
            continue;
        }
        scoped.push(p);
    }

    let (scope_roots, force_paths, remove_missing) = if scoped.is_empty() {
        (None, None, true)
    } else {
        let force_set: HashSet<PathBuf> = scoped.iter().cloned().collect();
        (Some(scoped.clone()), Some(force_set), false)
    };

    run_index_with_strategy(
        &cwd,
        &cfg,
        scope_roots,
        true,
        force_paths,
        remove_missing,
        "refresh",
    )
    .unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
}

fn run_search_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: cypress search [--view projects|files] [--limit <n>] <query...>");
        return;
    }

    let mut limit: usize = 20;
    let mut view = "projects".to_string();
    let mut query_parts: Vec<String> = Vec::new();

    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        if s == "--limit" {
            i += 1;
            let v = arg_value(args, i, "--limit");
            limit = v.parse::<usize>().unwrap_or_else(|_| {
                eprintln!("error: --limit must be an integer");
                process::exit(2);
            });
            i += 1;
            continue;
        }
        if let Some(v) = s.strip_prefix("--limit=") {
            limit = v.parse::<usize>().unwrap_or_else(|_| {
                eprintln!("error: --limit must be an integer");
                process::exit(2);
            });
            i += 1;
            continue;
        }
        if s == "--view" {
            i += 1;
            view = arg_value(args, i, "--view").to_lowercase();
            i += 1;
            continue;
        }
        if let Some(v) = s.strip_prefix("--view=") {
            view = v.to_lowercase();
            i += 1;
            continue;
        }
        if s.starts_with('-') {
            eprintln!("error: unknown option '{}'", s);
            process::exit(2);
        }
        query_parts.push(s);
        i += 1;
    }

    if view != "projects" && view != "files" {
        eprintln!("error: --view must be one of: projects, files");
        process::exit(2);
    }
    if query_parts.is_empty() {
        eprintln!("error: query is empty");
        process::exit(2);
    }
    if limit == 0 {
        limit = 1;
    }

    let query = query_parts.join(" ");
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    ensure_retrieval_backend_ready(&cfg, true, "search").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });

    ensure_native_embed_backend(&cfg, "search").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        eprintln!(
            "hint: use `cypress init --embed-backend local-hash` (or ollama), or run `cypress legacy search ...`"
        );
        process::exit(1);
    });

    let dbp = db_path(&cwd);
    let conn = open_db_rw(&dbp).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });

    if view == "files" {
        let rows = rank_files_native(&conn, &cfg, &query, limit).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            process::exit(1);
        });
        if rows.is_empty() {
            println!("No file results found.");
            return;
        }
        print_file_results(&rows);
        return;
    }

    let rows = rank_projects_native(&conn, &cfg, &query, limit).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    if rows.is_empty() {
        println!("No results found.");
        return;
    }
    print_project_results(&rows);
}

struct FswatchStream {
    child: std::process::Child,
    rx: Receiver<PathBuf>,
}

impl Drop for FswatchStream {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start_fswatch_stream(roots: &[PathBuf]) -> Result<FswatchStream, String> {
    if roots.is_empty() {
        return Err("no watch roots configured".to_string());
    }
    let mut cmd = Command::new("fswatch");
    cmd.arg("-0").arg("-r").arg("--latency").arg("0.2");
    for root in roots {
        cmd.arg(root);
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::null());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed launching fswatch: {}", e))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "failed acquiring fswatch stdout".to_string())?;
    let (tx, rx) = mpsc::channel::<PathBuf>();
    thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut buf: Vec<u8> = Vec::new();
        loop {
            buf.clear();
            match reader.read_until(0, &mut buf) {
                Ok(0) => break,
                Ok(_) => {
                    while matches!(buf.last(), Some(0 | b'\n' | b'\r')) {
                        buf.pop();
                    }
                    if buf.is_empty() {
                        continue;
                    }
                    let raw = String::from_utf8_lossy(&buf).trim().to_string();
                    if raw.is_empty() {
                        continue;
                    }
                    if tx.send(PathBuf::from(raw)).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
    Ok(FswatchStream { child, rx })
}

fn watch_stats_changed(stats: &IndexStats) -> bool {
    stats.updated_projects > 0
        || stats.removed_projects > 0
        || stats.vectorized_projects > 0
        || stats.chunk_vectors > 0
        || stats.vector_failures > 0
}

fn print_watch_tick(label: &str, stats: &IndexStats, quiet: bool) {
    if !watch_stats_changed(stats) && quiet {
        return;
    }
    let now = chrono_like_now();
    println!(
        "[{}] {} updated={} removed={} vectorized={} chunk_vectors={} skipped={}",
        now,
        label,
        stats.updated_projects,
        stats.removed_projects,
        stats.vectorized_projects,
        stats.chunk_vectors,
        stats.skipped_projects
    );
    if stats.vector_failures > 0 {
        println!(
            "[{}] {} vector_failures={}",
            now, label, stats.vector_failures
        );
    }
}

fn path_depth(path: &Path) -> usize {
    path.components().count()
}

fn longest_prefix_match<'a>(path: &Path, candidates: &'a [PathBuf]) -> Option<&'a PathBuf> {
    let mut best: Option<&PathBuf> = None;
    let mut best_depth = 0usize;
    for candidate in candidates {
        if path == candidate || path.starts_with(candidate) {
            let d = path_depth(candidate);
            if d >= best_depth {
                best_depth = d;
                best = Some(candidate);
            }
        }
    }
    best
}

fn normalize_watch_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return normalize_lexical(path);
    }
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    normalize_lexical(&cwd.join(path))
}

fn watch_path_relevant(path: &Path) -> bool {
    for comp in path.components() {
        if let Component::Normal(name) = comp {
            let seg = name.to_string_lossy();
            if seg.starts_with('.') || is_skip_dir(&seg) {
                return false;
            }
        }
    }
    let Some(name) = path.file_name().and_then(|v| v.to_str()) else {
        return false;
    };
    if name.starts_with('.') {
        return false;
    }
    let ext = Path::new(name)
        .extension()
        .and_then(|v| v.to_str())
        .unwrap_or("")
        .to_lowercase();
    if ext.is_empty() {
        return true;
    }
    is_indexable_suffix(&format!(".{}", ext))
}

fn derive_watch_targets(
    pending_paths: &HashSet<PathBuf>,
    tracked_roots: &[PathBuf],
) -> (Vec<PathBuf>, HashSet<PathBuf>) {
    if pending_paths.is_empty() || tracked_roots.is_empty() {
        return (Vec::new(), HashSet::new());
    }
    let known_projects = discover_projects(tracked_roots);
    let mut scope_set: HashSet<String> = HashSet::new();
    let mut force_set: HashSet<String> = HashSet::new();

    for raw in pending_paths {
        let path = normalize_watch_path(raw);
        if !watch_path_relevant(&path) {
            continue;
        }
        let Some(root) = longest_prefix_match(&path, tracked_roots) else {
            continue;
        };
        if let Some(project) = longest_prefix_match(&path, &known_projects) {
            let key = project.to_string_lossy().to_string();
            scope_set.insert(key.clone());
            force_set.insert(key);
        } else {
            scope_set.insert(root.to_string_lossy().to_string());
        }
    }

    let mut scope_roots: Vec<PathBuf> = scope_set.into_iter().map(|v| normalize_path(&v)).collect();
    scope_roots.sort();
    let force_paths: HashSet<PathBuf> = force_set.into_iter().map(|v| normalize_path(&v)).collect();
    (scope_roots, force_paths)
}

fn run_watch_event_loop(
    cwd: &Path,
    cfg: &ConfigValues,
    tracked_roots: &[PathBuf],
    interval_seconds: f64,
    debounce_ms: u64,
    quiet: bool,
) -> Result<(), String> {
    ensure_retrieval_backend_ready(cfg, true, "watch event-loop")?;
    let mut stream = start_fswatch_stream(tracked_roots)?;
    if !quiet {
        println!(
            "watch mode: fswatch event stream (debounce={}ms, full sweep every {:.1}s)",
            debounce_ms, interval_seconds
        );
    }
    let debounce = Duration::from_millis(debounce_ms.max(100));
    let sweep_every = Duration::from_secs_f64(interval_seconds.max(1.0));
    let mut pending_paths: HashSet<PathBuf> = HashSet::new();
    let mut last_event_at: Option<Instant> = None;
    let mut last_sweep_at = Instant::now();

    loop {
        match stream.rx.recv_timeout(Duration::from_millis(200)) {
            Ok(path) => {
                pending_paths.insert(path);
                last_event_at = Some(Instant::now());
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                return Err("fswatch stream disconnected".to_string());
            }
        }
        if let Some(status) = stream
            .child
            .try_wait()
            .map_err(|e| format!("failed checking fswatch status: {}", e))?
        {
            return Err(format!("fswatch exited unexpectedly: {}", status));
        }

        if !pending_paths.is_empty()
            && last_event_at
                .map(|t| t.elapsed() >= debounce)
                .unwrap_or(false)
        {
            let (scope_roots, force_paths) = derive_watch_targets(&pending_paths, tracked_roots);
            pending_paths.clear();
            last_event_at = None;
            if scope_roots.is_empty() {
                continue;
            }
            ensure_retrieval_backend_ready(cfg, true, "watch events")?;
            let stats = run_native_index(
                cwd,
                cfg,
                Some(scope_roots),
                false,
                force_paths,
                false,
                !quiet,
                "watch events",
            )?;
            print_watch_tick("event", &stats, quiet);
        }

        if last_sweep_at.elapsed() >= sweep_every {
            ensure_retrieval_backend_ready(cfg, true, "watch sweep")?;
            let stats = run_native_index(
                cwd,
                cfg,
                None,
                false,
                HashSet::new(),
                true,
                !quiet,
                "watch sweep",
            )?;
            print_watch_tick("sweep", &stats, quiet);
            last_sweep_at = Instant::now();
        }
    }
}

fn run_watch_polling_loop(cwd: &Path, cfg: &ConfigValues, interval_seconds: f64, quiet: bool) {
    if !quiet {
        println!("watch mode: polling");
    }
    loop {
        ensure_retrieval_backend_ready(cfg, true, "watch poll").unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            process::exit(1);
        });
        let stats = run_native_index(
            cwd,
            cfg,
            None,
            false,
            HashSet::new(),
            true,
            !quiet,
            "watch poll",
        )
        .unwrap_or_else(|e| {
            eprintln!("error: watch poll failed: {}", e);
            process::exit(1);
        });
        print_watch_tick("poll", &stats, quiet);
        thread::sleep(Duration::from_secs_f64(interval_seconds));
    }
}

fn run_watch_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "usage: cypress watch [--interval <seconds>] [--debounce-ms <ms>] [--once] [--quiet]"
        );
        return;
    }

    let mut interval_seconds = 30.0f64;
    let mut debounce_ms: u64 = 900;
    let mut once = false;
    let mut quiet = false;
    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        match s.as_str() {
            "--once" => {
                once = true;
            }
            "--quiet" => {
                quiet = true;
            }
            "--interval" => {
                i += 1;
                let raw = arg_value(args, i, "--interval");
                interval_seconds = raw.parse::<f64>().unwrap_or_else(|_| {
                    eprintln!("error: --interval must be a number");
                    process::exit(2);
                });
            }
            "--debounce-ms" => {
                i += 1;
                let raw = arg_value(args, i, "--debounce-ms");
                debounce_ms = raw.parse::<u64>().unwrap_or_else(|_| {
                    eprintln!("error: --debounce-ms must be an integer");
                    process::exit(2);
                });
            }
            other if other.starts_with("--interval=") => {
                let raw = other.trim_start_matches("--interval=");
                interval_seconds = raw.parse::<f64>().unwrap_or_else(|_| {
                    eprintln!("error: --interval must be a number");
                    process::exit(2);
                });
            }
            other if other.starts_with("--debounce-ms=") => {
                let raw = other.trim_start_matches("--debounce-ms=");
                debounce_ms = raw.parse::<u64>().unwrap_or_else(|_| {
                    eprintln!("error: --debounce-ms must be an integer");
                    process::exit(2);
                });
            }
            other => {
                eprintln!("error: unknown option '{}'", other);
                process::exit(2);
            }
        }
        i += 1;
    }

    interval_seconds = interval_seconds.max(1.0);
    debounce_ms = debounce_ms.clamp(100, 5000);

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    ensure_native_embed_backend(&cfg, "watch").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        eprintln!(
            "hint: use `cypress init --embed-backend local-hash` (or ollama), or run `cypress legacy watch ...`"
        );
        process::exit(1);
    });
    ensure_retrieval_backend_ready(&cfg, true, "watch").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });

    let dbp = db_path(&cwd);
    let conn = open_db_rw(&dbp).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    let tracked_roots = resolve_roots(&conn, &cfg, None).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });

    if !quiet {
        println!(
            "watching tracked roots (default root: {}, tracked={})",
            cfg.root.display(),
            tracked_roots.len()
        );
        println!(
            "interval: {:.1}s  debounce: {}ms",
            interval_seconds, debounce_ms
        );
        println!("press Ctrl-C to stop");
    }

    let bootstrap = run_native_index(
        &cwd,
        &cfg,
        None,
        false,
        HashSet::new(),
        true,
        !quiet,
        "watch bootstrap",
    )
    .unwrap_or_else(|e| {
        eprintln!("error: watch bootstrap failed: {}", e);
        process::exit(1);
    });
    print_watch_tick("bootstrap", &bootstrap, quiet);
    if once {
        return;
    }

    if command_exists("fswatch") {
        match run_watch_event_loop(
            &cwd,
            &cfg,
            &tracked_roots,
            interval_seconds,
            debounce_ms,
            quiet,
        ) {
            Ok(()) => return,
            Err(e) => {
                if !quiet {
                    eprintln!("watch: fswatch mode failed ({}); using polling fallback", e);
                }
            }
        }
    }
    if !quiet {
        println!("watch: fswatch unavailable; using polling fallback");
    }
    run_watch_polling_loop(&cwd, &cfg, interval_seconds, quiet);
}

fn chrono_like_now() -> String {
    let out = run_shell_capture("date '+%Y-%m-%d %H:%M:%S'");
    match out {
        Ok(v) if v.exit_code == 0 && !v.stdout.trim().is_empty() => v.stdout.trim().to_string(),
        _ => "now".to_string(),
    }
}

fn run_pick_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "usage: cypress pick [--query <text>] [--view projects|files] [--limit <n>] [--emit-path-file <path>]"
        );
        return;
    }

    let mut query = String::new();
    let mut view = "projects".to_string();
    let mut limit: usize = 30;
    let mut emit_path_file = String::new();
    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        match s.as_str() {
            "--query" => {
                i += 1;
                query = arg_value(args, i, "--query");
            }
            "--view" => {
                i += 1;
                view = arg_value(args, i, "--view").to_lowercase();
            }
            "--limit" => {
                i += 1;
                let raw = arg_value(args, i, "--limit");
                limit = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --limit must be an integer");
                    process::exit(2);
                });
            }
            "--emit-path-file" => {
                i += 1;
                emit_path_file = arg_value(args, i, "--emit-path-file");
            }
            other => {
                eprintln!("error: unknown option '{}'", other);
                process::exit(2);
            }
        }
        i += 1;
    }

    if limit == 0 {
        limit = 1;
    }
    if view != "projects" && view != "files" {
        eprintln!("error: --view must be one of: projects, files");
        process::exit(2);
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    ensure_native_embed_backend(&cfg, "pick").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        eprintln!(
            "hint: use `cypress init --embed-backend local-hash` (or ollama), or run `cypress legacy pick ...`"
        );
        process::exit(1);
    });

    ensure_retrieval_backend_ready(&cfg, true, "pick").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    let dbp = db_path(&cwd);
    let conn = open_db_rw(&dbp).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    let candidates = if view == "files" {
        let rows = rank_files_native(&conn, &cfg, &query, limit).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            process::exit(1);
        });
        if rows.is_empty() {
            eprintln!("error: no file matches found.");
            process::exit(1);
        }
        rows.into_iter()
            .map(|item| PickCandidate {
                path: item.path,
                detail: format!(
                    "{:.3}\tsem:{:.2} lex:{:.2} gr:{:.2} rel:{} q:{:.2} chunk:{}:{} project:{}",
                    item.score,
                    item.semantic,
                    item.lexical,
                    item.graph,
                    item.relation,
                    item.quality,
                    item.chunk_id,
                    item.chunk_index,
                    item.project_path
                ),
            })
            .collect::<Vec<_>>()
    } else {
        let rows = rank_projects_native(&conn, &cfg, &query, limit).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            process::exit(1);
        });
        if rows.is_empty() {
            eprintln!("error: no indexed projects found. run `cypress index` first.");
            process::exit(1);
        }
        rows.into_iter()
            .map(|item| PickCandidate {
                path: item.path,
                detail: format!(
                    "{:.3}\tsem:{:.2} lex:{:.2} fr:{:.2} gr:{:.2}",
                    item.score, item.semantic, item.lexical, item.frecency, item.graph
                ),
            })
            .collect::<Vec<_>>()
    };

    let selected = pick_candidate_path(&candidates, &query, &view).unwrap_or_else(|e| {
        eprintln!("error: picker failed: {}", e);
        process::exit(1);
    });
    let Some(selected_path) = selected else {
        process::exit(1);
    };

    if !emit_path_file.trim().is_empty() {
        let mut out_path = expand_tilde(&emit_path_file);
        if !out_path.is_absolute() {
            out_path = cwd.join(out_path);
        }
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent).unwrap_or_else(|e| {
                eprintln!(
                    "error: failed to create emit path parent '{}': {}",
                    parent.display(),
                    e
                );
                process::exit(1);
            });
        }
        fs::write(&out_path, format!("{}\n", selected_path)).unwrap_or_else(|e| {
            eprintln!(
                "error: failed writing emit path file '{}': {}",
                out_path.display(),
                e
            );
            process::exit(1);
        });
    }

    record_selection_event(&conn, &query, &selected_path, now_ts()).unwrap_or_else(|e| {
        eprintln!("warning: failed to record selection event: {}", e);
    });
    println!("{}", selected_path);
}

struct PickCandidate {
    path: String,
    detail: String,
}

fn pick_candidate_path(
    candidates: &[PickCandidate],
    query: &str,
    view: &str,
) -> Result<Option<String>, String> {
    if candidates.is_empty() {
        return Ok(None);
    }
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        return Ok(Some(candidates[0].path.clone()));
    }
    if !command_exists("fzf") {
        return Ok(Some(candidates[0].path.clone()));
    }

    let mut payload = String::new();
    for item in candidates {
        payload.push_str(&format!("{}\t{}\n", item.path, item.detail));
    }
    let prompt = if query.trim().is_empty() {
        "cypress> ".to_string()
    } else {
        format!("cypress> {} ", query.trim())
    };
    let mut cmd = Command::new("fzf");
    cmd.arg("--height=70%")
        .arg("--layout=reverse")
        .arg("--border")
        .arg("--delimiter=\t")
        .arg("--with-nth=1,2,3")
        .arg("--prompt")
        .arg(prompt)
        .arg("--header")
        .arg(format!("{} path | score | signals", view))
        .arg("--no-sort")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed to launch fzf: {}", e))?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(payload.as_bytes())
            .map_err(|e| format!("failed writing picker input: {}", e))?;
    }
    let out = child
        .wait_with_output()
        .map_err(|e| format!("failed waiting for fzf: {}", e))?;
    if !out.status.success() {
        return Ok(None);
    }
    let line = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if line.is_empty() {
        return Ok(None);
    }
    let path = line.split('\t').next().unwrap_or("").trim().to_string();
    if path.is_empty() {
        return Ok(None);
    }
    Ok(Some(path))
}

#[derive(Clone, Debug)]
struct BenchDatasetMeta {
    created_at: f64,
    source_db: String,
    model: String,
    dim: i64,
    chunks: i64,
}

fn run_bench_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_bench_help();
        return;
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let bench_root = data_dir(&cwd).join("bench");
    fs::create_dir_all(&bench_root).unwrap_or_else(|e| {
        eprintln!(
            "error: failed to create bench dir '{}': {}",
            bench_root.display(),
            e
        );
        process::exit(1);
    });

    let mut action = "plan".to_string();
    let mut action_set = false;
    let mut model_key_opt: Option<String> = None;
    let mut export_limit: Option<usize> = None;
    let mut dataset_path_opt: Option<PathBuf> = None;
    let mut queries_path_opt: Option<PathBuf> = None;
    let mut backends: Vec<String> = Vec::new();
    let mut k: usize = 20;
    let mut repeats: usize = 10;
    let mut warmup: usize = 1;
    let mut rebuild = false;
    let mut dataset_limit: Option<usize> = None;
    let mut dataset_target: Option<usize> = None;

    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        if !action_set
            && !s.starts_with('-')
            && matches!(s.as_str(), "plan" | "doctor" | "export" | "run")
        {
            action = s;
            action_set = true;
            i += 1;
            continue;
        }
        match s.as_str() {
            "--model-key" => {
                i += 1;
                model_key_opt = Some(arg_value(args, i, "--model-key").trim().to_string());
            }
            "--limit" => {
                i += 1;
                let raw = arg_value(args, i, "--limit");
                let parsed = raw.parse::<i64>().unwrap_or_else(|_| {
                    eprintln!("error: --limit must be an integer");
                    process::exit(2);
                });
                export_limit = if parsed <= 0 {
                    None
                } else {
                    Some(parsed as usize)
                };
            }
            "--dataset" => {
                i += 1;
                dataset_path_opt = Some(normalize_path(&arg_value(args, i, "--dataset")));
            }
            "--queries" => {
                i += 1;
                queries_path_opt = Some(normalize_path(&arg_value(args, i, "--queries")));
            }
            "--backend" => {
                i += 1;
                let backend = arg_value(args, i, "--backend").trim().to_lowercase();
                if !backend.is_empty() {
                    backends.push(backend);
                }
            }
            "--k" => {
                i += 1;
                let raw = arg_value(args, i, "--k");
                k = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --k must be an integer");
                    process::exit(2);
                });
                k = k.max(1);
            }
            "--repeats" => {
                i += 1;
                let raw = arg_value(args, i, "--repeats");
                repeats = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --repeats must be an integer");
                    process::exit(2);
                });
                repeats = repeats.max(1);
            }
            "--warmup" => {
                i += 1;
                let raw = arg_value(args, i, "--warmup");
                warmup = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --warmup must be an integer");
                    process::exit(2);
                });
            }
            "--rebuild" => {
                rebuild = true;
            }
            "--dataset-limit" => {
                i += 1;
                let raw = arg_value(args, i, "--dataset-limit");
                let parsed = raw.parse::<i64>().unwrap_or_else(|_| {
                    eprintln!("error: --dataset-limit must be an integer");
                    process::exit(2);
                });
                dataset_limit = if parsed <= 0 {
                    None
                } else {
                    Some(parsed as usize)
                };
            }
            "--target-chunks" => {
                i += 1;
                let raw = arg_value(args, i, "--target-chunks");
                let parsed = raw.parse::<i64>().unwrap_or_else(|_| {
                    eprintln!("error: --target-chunks must be an integer");
                    process::exit(2);
                });
                dataset_target = if parsed <= 0 {
                    None
                } else {
                    Some(parsed as usize)
                };
            }
            other if other.starts_with("--model-key=") => {
                model_key_opt = Some(other.trim_start_matches("--model-key=").trim().to_string());
            }
            other if other.starts_with("--limit=") => {
                let raw = other.trim_start_matches("--limit=").trim();
                let parsed = raw.parse::<i64>().unwrap_or_else(|_| {
                    eprintln!("error: --limit must be an integer");
                    process::exit(2);
                });
                export_limit = if parsed <= 0 {
                    None
                } else {
                    Some(parsed as usize)
                };
            }
            other if other.starts_with("--dataset=") => {
                let raw = other.trim_start_matches("--dataset=").trim();
                dataset_path_opt = Some(normalize_path(raw));
            }
            other if other.starts_with("--queries=") => {
                let raw = other.trim_start_matches("--queries=").trim();
                queries_path_opt = Some(normalize_path(raw));
            }
            other if other.starts_with("--backend=") => {
                let backend = other.trim_start_matches("--backend=").trim().to_lowercase();
                if !backend.is_empty() {
                    backends.push(backend);
                }
            }
            other if other.starts_with("--k=") => {
                let raw = other.trim_start_matches("--k=").trim();
                k = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --k must be an integer");
                    process::exit(2);
                });
                k = k.max(1);
            }
            other if other.starts_with("--repeats=") => {
                let raw = other.trim_start_matches("--repeats=").trim();
                repeats = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --repeats must be an integer");
                    process::exit(2);
                });
                repeats = repeats.max(1);
            }
            other if other.starts_with("--warmup=") => {
                let raw = other.trim_start_matches("--warmup=").trim();
                warmup = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --warmup must be an integer");
                    process::exit(2);
                });
            }
            other if other.starts_with("--dataset-limit=") => {
                let raw = other.trim_start_matches("--dataset-limit=").trim();
                let parsed = raw.parse::<i64>().unwrap_or_else(|_| {
                    eprintln!("error: --dataset-limit must be an integer");
                    process::exit(2);
                });
                dataset_limit = if parsed <= 0 {
                    None
                } else {
                    Some(parsed as usize)
                };
            }
            other if other.starts_with("--target-chunks=") => {
                let raw = other.trim_start_matches("--target-chunks=").trim();
                let parsed = raw.parse::<i64>().unwrap_or_else(|_| {
                    eprintln!("error: --target-chunks must be an integer");
                    process::exit(2);
                });
                dataset_target = if parsed <= 0 {
                    None
                } else {
                    Some(parsed as usize)
                };
            }
            other => {
                eprintln!("error: unknown bench action/option '{}'", other);
                process::exit(2);
            }
        }
        i += 1;
    }

    let dataset_path = dataset_path_opt.unwrap_or_else(|| bench_root.join("dataset.jsonl"));
    let queries_path = queries_path_opt.unwrap_or_else(|| bench_root.join("queries.txt"));
    let meta_path = bench_meta_path(&dataset_path);

    let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
    let model_key = model_key_opt.unwrap_or_else(|| default_bench_model_key(&cfg));

    match action.as_str() {
        "plan" => {
            println!("benchmark plan: docs/BENCHMARK_PLAN.md");
            println!("bench dir: {}", bench_root.display());
            println!("next:");
            println!("  cypress bench doctor");
            println!("  cypress bench export");
            println!("  cypress bench run --backend falkordb");
        }
        "doctor" => {
            let cfg_path = config_path(&cwd);
            let dbp = db_path(&cwd);
            println!("config: {}", cfg_path.display());
            println!("db: {}", dbp.display());
            println!("bench dir: {}", bench_root.display());
            println!("model key: {}", model_key);

            if dbp.exists() {
                match open_db_read_only(&dbp) {
                    Ok(conn) => {
                        let chunks: i64 = conn
                            .query_row("SELECT COUNT(*) FROM project_chunks", [], |row| row.get(0))
                            .unwrap_or(0);
                        let vectors: i64 = conn
                            .query_row(
                                "SELECT COUNT(*) FROM project_chunk_vectors WHERE model = ?1",
                                params![model_key.clone()],
                                |row| row.get(0),
                            )
                            .unwrap_or(0);
                        println!("chunks in sqlite: {}", chunks);
                        println!("chunk vectors for model: {}", vectors);
                    }
                    Err(err) => println!("sqlite: error ({})", err),
                }
            } else {
                println!("sqlite: error (database file missing)");
            }
            println!(
                "queries file: {} ({})",
                queries_path.display(),
                if queries_path.exists() {
                    "exists"
                } else {
                    "missing"
                }
            );
            println!(
                "falkordb reachable: {}",
                yes_no(falkor_store::ping().is_ok())
            );
            println!("lancedb native bench: no");
        }
        "export" => {
            write_default_bench_queries(&queries_path).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
            let dbp = db_path(&cwd);
            let conn = open_db_rw(&dbp).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
            let meta = export_chunk_dataset_native(
                &conn,
                &dbp,
                &model_key,
                &dataset_path,
                &meta_path,
                export_limit,
            )
            .unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
            println!("dataset: {}", dataset_path.display());
            println!("meta: {}", meta_path.display());
            println!("chunks: {}", meta.chunks);
            println!("dim: {}", meta.dim);
            println!("model: {}", meta.model);
            println!("queries: {}", queries_path.display());
        }
        "run" => {
            write_default_bench_queries(&queries_path).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
            if !dataset_path.exists() {
                println!("bench: dataset missing; exporting from sqlite index...");
                let dbp = db_path(&cwd);
                let conn = open_db_rw(&dbp).unwrap_or_else(|e| {
                    eprintln!("error: {}", e);
                    process::exit(1);
                });
                export_chunk_dataset_native(
                    &conn,
                    &dbp,
                    &model_key,
                    &dataset_path,
                    &meta_path,
                    None,
                )
                .unwrap_or_else(|e| {
                    eprintln!("error: {}", e);
                    process::exit(1);
                });
            }

            let queries = load_bench_queries(&queries_path).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
            if queries.is_empty() {
                eprintln!("error: no queries found in {}", queries_path.display());
                process::exit(1);
            }

            if backends.is_empty() {
                backends.push("falkordb".to_string());
            }
            if dataset_target.is_some() && !rebuild {
                println!("bench: target-chunks requested; forcing --rebuild for correctness");
                rebuild = true;
            }

            let report_path = bench_root.join("report.json");
            let report_md_path = bench_root.join("report.md");

            let dim = bench_dim_from_meta_or_db(&meta_path, &db_path(&cwd), &model_key, 384)
                .unwrap_or(384) as usize;
            let mut backend_results: serde_json::Map<String, Value> = serde_json::Map::new();
            for backend in &backends {
                let value = match backend.as_str() {
                    "falkordb" | "falkordblite" => run_bench_falkordb_native(
                        &cwd,
                        &cfg,
                        &queries,
                        &model_key,
                        dim,
                        k,
                        repeats,
                        warmup,
                        rebuild,
                        dataset_limit,
                        dataset_target,
                    ),
                    "lancedb" => Ok(serde_json::json!({
                        "status": "unsupported",
                        "error": "native LanceDB benchmark is not implemented in Rust yet"
                    })),
                    _ => Ok(serde_json::json!({"status": "unknown_backend"})),
                }
                .unwrap_or_else(|e| serde_json::json!({"status": "error", "error": e}));
                backend_results.insert(backend.clone(), value);
            }

            let report = serde_json::json!({
                "created_at": now_ts(),
                "model_key": model_key,
                "dataset_jsonl": dataset_path.to_string_lossy().to_string(),
                "dataset_meta": meta_path.to_string_lossy().to_string(),
                "queries_file": queries_path.to_string_lossy().to_string(),
                "query_count": queries.len(),
                "k": k,
                "repeats": repeats,
                "warmup": warmup,
                "dataset_limit": dataset_limit,
                "dataset_target": dataset_target,
                "rebuild": rebuild,
                "backends": Value::Object(backend_results.clone()),
            });
            let report_raw = serde_json::to_string_pretty(&report)
                .map_err(|e| format!("failed to serialize report JSON: {}", e))
                .and_then(|v| {
                    fs::write(&report_path, format!("{}\n", v)).map_err(|e| {
                        format!("failed writing report '{}': {}", report_path.display(), e)
                    })
                });
            if let Err(e) = report_raw {
                eprintln!("error: {}", e);
                process::exit(1);
            }

            let mut lines: Vec<String> = Vec::new();
            lines.push("# Cypress Bench Report".to_string());
            lines.push(String::new());
            lines.push(format!("- Dataset: `{}`", dataset_path.display()));
            lines.push(format!(
                "- Queries: `{}` ({} queries)",
                queries_path.display(),
                queries.len()
            ));
            lines.push(format!(
                "- k: {}, repeats: {}, warmup: {}",
                k, repeats, warmup
            ));
            if let Some(v) = dataset_limit {
                lines.push(format!("- dataset_limit: {}", v));
            }
            if let Some(v) = dataset_target {
                lines.push(format!("- target_chunks: {}", v));
            }
            lines.push(format!("- Backends: {}", backends.join(", ")));
            lines.push(String::new());
            for backend in &backends {
                let res = backend_results.get(backend).cloned().unwrap_or_else(
                    || serde_json::json!({"status": "error", "error": "missing backend result"}),
                );
                lines.push(format!("## {}", backend));
                lines.push(String::new());
                let status = res
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("error")
                    .to_string();
                lines.push(format!("- status: {}", status));
                if status != "ok" {
                    let err = res
                        .get("error")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if !err.is_empty() {
                        lines.push(format!("- error: `{}`", err));
                    }
                    lines.push(String::new());
                    continue;
                }
                let total = res
                    .get("total")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({}));
                lines.push(format!(
                    "- connect_s: {:.3}",
                    res.get("connect_s").and_then(Value::as_f64).unwrap_or(0.0)
                ));
                lines.push(format!(
                    "- ingest_s: {:.3}",
                    res.get("ingest_s").and_then(Value::as_f64).unwrap_or(0.0)
                ));
                lines.push(format!(
                    "- index_s: {:.3}",
                    res.get("index_s").and_then(Value::as_f64).unwrap_or(0.0)
                ));
                lines.push(format!(
                    "- total p50_ms: {:.2}",
                    total.get("p50_ms").and_then(Value::as_f64).unwrap_or(0.0)
                ));
                lines.push(format!(
                    "- total p95_ms: {:.2}",
                    total.get("p95_ms").and_then(Value::as_f64).unwrap_or(0.0)
                ));
                lines.push(String::new());
            }
            fs::write(&report_md_path, format!("{}\n", lines.join("\n"))).unwrap_or_else(|e| {
                eprintln!(
                    "error: failed writing report markdown '{}': {}",
                    report_md_path.display(),
                    e
                );
                process::exit(1);
            });
            println!("report: {}", report_path.display());
            println!("report_md: {}", report_md_path.display());
        }
        other => {
            eprintln!("error: unknown bench action '{}'", other);
            process::exit(2);
        }
    }
}

fn print_bench_help() {
    println!("usage: cypress bench [plan|doctor|export|run] [options]");
    println!("options:");
    println!("  --model-key <key>        Vector model key for dataset export");
    println!("  --limit <n>              Optional max rows during export (0=all)");
    println!("  --dataset <path>         Dataset JSONL path");
    println!("  --queries <path>         Queries file path");
    println!("  --backend <name>         Backend to benchmark (repeatable)");
    println!("  --k <n>                  Top-k per query (default: 20)");
    println!("  --repeats <n>            Measured repeats per query (default: 10)");
    println!("  --warmup <n>             Warmup repeats per query (default: 1)");
    println!("  --rebuild                Refresh FalkorDB graph from SQLite before benchmark");
    println!("  --dataset-limit <n>      Included for compatibility (currently informational)");
    println!("  --target-chunks <n>      Included for compatibility (currently informational)");
}

fn default_bench_model_key(cfg: &ConfigValues) -> String {
    let backend = cfg.embed_backend.trim().to_lowercase();
    if backend == "local-hash" {
        return format!("local-hash-v1:{}", cfg.local_embed_dim.max(64));
    }
    if backend == "ollama" {
        let model = if cfg.embed_model.trim().is_empty() {
            "qwen3-embedding".to_string()
        } else {
            cfg.embed_model.trim().to_string()
        };
        return format!("ollama:{}", model);
    }
    if backend == "local" || backend == "auto" {
        let model = if cfg.embed_model.trim().is_empty() {
            "sentence-transformers/all-MiniLM-L6-v2".to_string()
        } else {
            cfg.embed_model.trim().to_string()
        };
        return format!("st:{}", model);
    }
    format!("{}:{}", backend, cfg.embed_model.trim())
}

fn bench_meta_path(dataset_path: &Path) -> PathBuf {
    dataset_path.with_extension("meta.json")
}

fn write_default_bench_queries(path: &Path) -> Result<(), String> {
    if path.exists() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            format!(
                "failed to create queries directory '{}': {}",
                parent.display(),
                e
            )
        })?;
    }
    let body = [
        "# One query per line. Lines starting with # are ignored.",
        "storage replication",
        "semantic layer",
        "auth flow",
        "inference pricing",
        "vector search",
    ]
    .join("\n")
        + "\n";
    fs::write(path, body)
        .map_err(|e| format!("failed writing default queries '{}': {}", path.display(), e))
}

fn load_bench_queries(path: &Path) -> Result<Vec<String>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("failed reading queries file '{}': {}", path.display(), e))?;
    let mut out = Vec::new();
    for line in raw.lines() {
        let q = line.trim();
        if q.is_empty() || q.starts_with('#') {
            continue;
        }
        out.push(q.to_string());
    }
    Ok(out)
}

fn export_chunk_dataset_native(
    conn: &Connection,
    source_db: &Path,
    model_key: &str,
    out_jsonl: &Path,
    out_meta_json: &Path,
    limit: Option<usize>,
) -> Result<BenchDatasetMeta, String> {
    if let Some(parent) = out_jsonl.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed creating dataset dir '{}': {}", parent.display(), e))?;
    }
    if let Some(parent) = out_meta_json.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed creating metadata dir '{}': {}", parent.display(), e))?;
    }

    let sql = if limit.is_some() {
        r#"
SELECT
    pc.id AS chunk_id,
    p.path AS project_path,
    pc.doc_path,
    pc.doc_rel_path,
    pc.text,
    pc.chunk_index,
    pcv.dim,
    pcv.vector
FROM project_chunks pc
JOIN projects p ON p.id = pc.project_id
JOIN project_chunk_vectors pcv ON pcv.chunk_id = pc.id
WHERE pcv.model = ?1
ORDER BY pc.id
LIMIT ?2
"#
    } else {
        r#"
SELECT
    pc.id AS chunk_id,
    p.path AS project_path,
    pc.doc_path,
    pc.doc_rel_path,
    pc.text,
    pc.chunk_index,
    pcv.dim,
    pcv.vector
FROM project_chunks pc
JOIN projects p ON p.id = pc.project_id
JOIN project_chunk_vectors pcv ON pcv.chunk_id = pc.id
WHERE pcv.model = ?1
ORDER BY pc.id
"#
    };

    let mut stmt = conn
        .prepare(sql)
        .map_err(|e| format!("failed preparing benchmark export query: {}", e))?;
    let mut rows = if let Some(v) = limit {
        stmt.query(params![model_key, v as i64])
            .map_err(|e| format!("failed running benchmark export query: {}", e))?
    } else {
        stmt.query(params![model_key])
            .map_err(|e| format!("failed running benchmark export query: {}", e))?
    };

    let out_file = fs::File::create(out_jsonl).map_err(|e| {
        format!(
            "failed creating dataset file '{}': {}",
            out_jsonl.display(),
            e
        )
    })?;
    let mut writer = BufWriter::new(out_file);
    let mut dim = 0i64;
    let mut chunks = 0i64;
    while let Some(row) = rows
        .next()
        .map_err(|e| format!("failed reading benchmark export row: {}", e))?
    {
        let chunk_id: i64 = row
            .get(0)
            .map_err(|e| format!("failed reading chunk_id: {}", e))?;
        let project_path: String = row
            .get(1)
            .map_err(|e| format!("failed reading project_path: {}", e))?;
        let doc_path: String = row
            .get(2)
            .map_err(|e| format!("failed reading doc_path: {}", e))?;
        let doc_rel_path: String = row
            .get(3)
            .map_err(|e| format!("failed reading doc_rel_path: {}", e))?;
        let text: String = row
            .get(4)
            .map_err(|e| format!("failed reading text: {}", e))?;
        let chunk_index: i64 = row
            .get(5)
            .map_err(|e| format!("failed reading chunk_index: {}", e))?;
        let row_dim: i64 = row
            .get(6)
            .map_err(|e| format!("failed reading dim: {}", e))?;
        let vector_blob: Vec<u8> = row
            .get(7)
            .map_err(|e| format!("failed reading vector blob: {}", e))?;

        if dim <= 0 {
            dim = row_dim.max(1);
        }
        let rec = serde_json::json!({
            "chunk_id": chunk_id,
            "project_path": project_path,
            "doc_path": doc_path,
            "doc_rel_path": doc_rel_path,
            "chunk_index": chunk_index,
            "text": text,
            "vector_b64_f32": BASE64_STANDARD.encode(vector_blob),
        });
        writer
            .write_all(rec.to_string().as_bytes())
            .and_then(|_| writer.write_all(b"\n"))
            .map_err(|e| format!("failed writing dataset record: {}", e))?;
        chunks += 1;
    }
    writer
        .flush()
        .map_err(|e| format!("failed flushing dataset output: {}", e))?;

    if chunks == 0 {
        return Err(format!(
            "No chunk vectors found for model key '{}'. Run `cypress index` first or pass --model-key.",
            model_key
        ));
    }

    let meta = BenchDatasetMeta {
        created_at: now_ts(),
        source_db: source_db.to_string_lossy().to_string(),
        model: model_key.to_string(),
        dim: dim.max(1),
        chunks,
    };
    let meta_json = serde_json::json!({
        "created_at": meta.created_at,
        "source_db": meta.source_db,
        "model": meta.model,
        "dim": meta.dim,
        "chunks": meta.chunks,
    });
    let text = serde_json::to_string_pretty(&meta_json)
        .map_err(|e| format!("failed serializing dataset metadata: {}", e))?;
    fs::write(out_meta_json, format!("{}\n", text)).map_err(|e| {
        format!(
            "failed writing metadata '{}': {}",
            out_meta_json.display(),
            e
        )
    })?;
    Ok(meta)
}

fn bench_dim_from_meta_or_db(
    meta_path: &Path,
    db_path: &Path,
    model_key: &str,
    default_dim: i64,
) -> Option<i64> {
    if meta_path.exists() {
        if let Ok(raw) = fs::read_to_string(meta_path) {
            if let Ok(value) = serde_json::from_str::<Value>(&raw) {
                if let Some(dim) = value.get("dim").and_then(Value::as_i64) {
                    if dim > 0 {
                        return Some(dim);
                    }
                }
            }
        }
    }
    if db_path.exists() {
        if let Ok(conn) = open_db_read_only(db_path) {
            if let Ok(dim) = conn.query_row(
                "SELECT dim FROM project_chunk_vectors WHERE model = ?1 LIMIT 1",
                params![model_key],
                |row| row.get::<_, i64>(0),
            ) {
                if dim > 0 {
                    return Some(dim);
                }
            }
        }
    }
    Some(default_dim.max(1))
}

fn run_bench_falkordb_native(
    cwd: &Path,
    _cfg: &ConfigValues,
    queries: &[String],
    model_key: &str,
    dim: usize,
    k: usize,
    repeats: usize,
    warmup: usize,
    rebuild: bool,
    dataset_limit: Option<usize>,
    dataset_target: Option<usize>,
) -> Result<Value, String> {
    let mut ingest_s = 0.0f64;
    let index_s = 0.0f64;
    if rebuild {
        let t = Instant::now();
        let conn = open_db_rw(&db_path(cwd))?;
        let _ = falkor_store::sync_from_sqlite(&conn, model_key)?;
        ingest_s = t.elapsed().as_secs_f64();
    }

    let embedder = LocalHashEmbedder::new(dim.max(64));
    let mut query_vectors: Vec<(String, Vec<f32>)> = Vec::new();
    for q in queries {
        query_vectors.push((q.clone(), embedder.embed_one(q)?));
    }

    let stats = falkor_store::benchmark_hybrid_queries(&query_vectors, k, repeats, warmup)?;
    let mut result = serde_json::json!({
        "status": "ok",
        "connect_s": stats.connect_s,
        "ingest_s": ingest_s,
        "index_s": index_s,
        "vector": summarize_latencies_ms(&stats.vector_ms),
        "fts": summarize_latencies_ms(&stats.fts_ms),
        "total": summarize_latencies_ms(&stats.total_ms),
    });
    if dataset_limit.is_some() || dataset_target.is_some() {
        result["note"] = Value::String(
            "dataset-limit/target-chunks are informational only for native falkordb benchmark"
                .to_string(),
        );
    }
    Ok(result)
}

fn summarize_latencies_ms(values: &[f64]) -> Value {
    if values.is_empty() {
        return serde_json::json!({
            "count": 0.0,
            "min_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
            "mean_ms": 0.0,
        });
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let count = sorted.len() as f64;
    let min_ms = *sorted.first().unwrap_or(&0.0);
    let max_ms = *sorted.last().unwrap_or(&0.0);
    let mean_ms = sorted.iter().sum::<f64>() / count.max(1.0);
    serde_json::json!({
        "count": count,
        "min_ms": min_ms,
        "p50_ms": percentile_ms(&sorted, 50.0),
        "p95_ms": percentile_ms(&sorted, 95.0),
        "max_ms": max_ms,
        "mean_ms": mean_ms,
    })
}

fn percentile_ms(sorted_values: &[f64], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    if sorted_values.len() == 1 {
        return sorted_values[0];
    }
    let k = ((sorted_values.len() - 1) as f64) * (p / 100.0);
    let floor = k.floor() as usize;
    let ceil = k.ceil() as usize;
    if floor == ceil {
        return sorted_values[floor];
    }
    let d0 = sorted_values[floor] * ((ceil as f64) - k);
    let d1 = sorted_values[ceil] * (k - (floor as f64));
    d0 + d1
}

fn api_pid_path(cwd: &Path) -> PathBuf {
    data_dir(cwd).join("api.pid")
}

fn api_port_path(cwd: &Path) -> PathBuf {
    data_dir(cwd).join("api.port")
}

fn api_log_path(cwd: &Path) -> PathBuf {
    data_dir(cwd).join("api.log")
}

fn daemon_default_host() -> String {
    env::var("CYPRESS_API_HOST")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .or_else(|| env::var("SEMCD_API_HOST").ok())
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "127.0.0.1".to_string())
}

fn daemon_default_port() -> u16 {
    env::var("CYPRESS_API_PORT")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .or_else(|| env::var("SEMCD_API_PORT").ok())
        .and_then(|v| v.trim().parse::<u16>().ok())
        .unwrap_or(8765)
}

fn daemon_default_timeout() -> u64 {
    env::var("CYPRESS_API_START_TIMEOUT")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .or_else(|| env::var("SEMCD_API_START_TIMEOUT").ok())
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(8)
        .max(1)
}

fn read_saved_api_port(cwd: &Path) -> Option<u16> {
    let path = api_port_path(cwd);
    let raw = fs::read_to_string(path).ok()?;
    raw.trim().parse::<u16>().ok()
}

fn read_saved_api_pid(cwd: &Path) -> Option<u32> {
    let path = api_pid_path(cwd);
    let raw = fs::read_to_string(path).ok()?;
    raw.trim().parse::<u32>().ok()
}

fn persist_api_runtime_state(cwd: &Path, pid: u32, port: u16) -> Result<(), String> {
    let data = data_dir(cwd);
    fs::create_dir_all(&data)
        .map_err(|e| format!("failed creating data dir '{}': {}", data.display(), e))?;
    fs::write(api_pid_path(cwd), format!("{}\n", pid))
        .map_err(|e| format!("failed writing api pid file: {}", e))?;
    fs::write(api_port_path(cwd), format!("{}\n", port))
        .map_err(|e| format!("failed writing api port file: {}", e))?;
    Ok(())
}

fn clear_api_runtime_state(cwd: &Path) {
    let _ = fs::remove_file(api_pid_path(cwd));
    let _ = fs::remove_file(api_port_path(cwd));
}

fn api_health_host_port(host: &str, port: u16) -> bool {
    let Ok(mut stream) = TcpStream::connect((host, port)) else {
        return false;
    };
    let req = format!(
        "GET /health HTTP/1.1\r\nHost: {}:{}\r\nConnection: close\r\n\r\n",
        host, port
    );
    if stream.write_all(req.as_bytes()).is_err() {
        return false;
    }
    let mut buf = [0u8; 64];
    let Ok(n) = stream.read(&mut buf) else {
        return false;
    };
    if n == 0 {
        return false;
    }
    let head = String::from_utf8_lossy(&buf[..n]);
    head.starts_with("HTTP/1.1 200") || head.starts_with("HTTP/1.0 200")
}

fn port_available(host: &str, port: u16) -> bool {
    match TcpStream::connect((host, port)) {
        Ok(_) => false,
        Err(err) => {
            matches!(
                err.kind(),
                std::io::ErrorKind::ConnectionRefused
                    | std::io::ErrorKind::TimedOut
                    | std::io::ErrorKind::AddrNotAvailable
                    | std::io::ErrorKind::PermissionDenied
                    | std::io::ErrorKind::NotConnected
                    | std::io::ErrorKind::WouldBlock
                    | std::io::ErrorKind::Other
            )
        }
    }
}

fn find_free_port(host: &str, start: u16, span: usize) -> Option<u16> {
    for offset in 0..span.max(1) {
        let port = start.saturating_add(offset as u16);
        if port == 0 {
            continue;
        }
        if port_available(host, port) {
            return Some(port);
        }
    }
    None
}

fn tail_file_lines(path: &Path, lines: usize) -> Result<(), String> {
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("failed reading log '{}': {}", path.display(), e))?;
    let items: Vec<&str> = raw.lines().collect();
    let start = items.len().saturating_sub(lines.max(1));
    for line in &items[start..] {
        println!("{}", line);
    }
    Ok(())
}

fn spawn_api_daemon(cwd: &Path, host: &str, port: u16) -> Result<u32, String> {
    let log_path = api_log_path(cwd);
    if let Some(parent) = log_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed creating api log dir '{}': {}", parent.display(), e))?;
    }
    let log_out = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| format!("failed opening api log '{}': {}", log_path.display(), e))?;
    let log_err = log_out.try_clone().map_err(|e| {
        format!(
            "failed cloning api log handle '{}': {}",
            log_path.display(),
            e
        )
    })?;
    let exe = env::current_exe().map_err(|e| format!("failed resolving current exe: {}", e))?;
    let child = Command::new(exe)
        .arg("api")
        .arg("--host")
        .arg(host)
        .arg("--port")
        .arg(port.to_string())
        .stdout(Stdio::from(log_out))
        .stderr(Stdio::from(log_err))
        .spawn()
        .map_err(|e| format!("failed launching daemon api process: {}", e))?;
    Ok(child.id())
}

fn run_daemon_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "usage: cypress daemon [start|stop|restart|status|logs [n]] [--host <addr>] [--port <n>] [--timeout <seconds>]"
        );
        return;
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut action = "status".to_string();
    let mut host = daemon_default_host();
    let mut port = daemon_default_port();
    if let Some(saved) = read_saved_api_port(&cwd) {
        port = saved;
    }
    let mut timeout_s = daemon_default_timeout();
    let mut logs_n = 60usize;
    let mut action_set = false;

    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        match s.as_str() {
            "start" | "stop" | "restart" | "status" | "logs" => {
                action = s;
                action_set = true;
            }
            "--host" => {
                i += 1;
                host = arg_value(args, i, "--host");
            }
            "--port" => {
                i += 1;
                let raw = arg_value(args, i, "--port");
                port = raw.parse::<u16>().unwrap_or_else(|_| {
                    eprintln!("error: --port must be an integer between 1 and 65535");
                    process::exit(2);
                });
            }
            "--timeout" => {
                i += 1;
                let raw = arg_value(args, i, "--timeout");
                timeout_s = raw.parse::<u64>().unwrap_or_else(|_| {
                    eprintln!("error: --timeout must be an integer number of seconds");
                    process::exit(2);
                });
                timeout_s = timeout_s.max(1);
            }
            other if other.starts_with("--host=") => {
                host = other.trim_start_matches("--host=").to_string();
            }
            other if other.starts_with("--port=") => {
                let raw = other.trim_start_matches("--port=");
                port = raw.parse::<u16>().unwrap_or_else(|_| {
                    eprintln!("error: --port must be an integer between 1 and 65535");
                    process::exit(2);
                });
            }
            other if other.starts_with("--timeout=") => {
                let raw = other.trim_start_matches("--timeout=");
                timeout_s = raw.parse::<u64>().unwrap_or_else(|_| {
                    eprintln!("error: --timeout must be an integer number of seconds");
                    process::exit(2);
                });
                timeout_s = timeout_s.max(1);
            }
            other => {
                if action == "logs" && !other.starts_with('-') {
                    logs_n = other.parse::<usize>().unwrap_or_else(|_| {
                        eprintln!("error: logs count must be an integer");
                        process::exit(2);
                    });
                } else if !action_set {
                    action = other.to_string();
                    action_set = true;
                } else {
                    eprintln!("error: unknown option '{}'", other);
                    process::exit(2);
                }
            }
        }
        i += 1;
    }

    match action.as_str() {
        "start" => {
            if api_health_host_port(&host, port) {
                if let Some(pid) = read_saved_api_pid(&cwd) {
                    let _ = persist_api_runtime_state(&cwd, pid, port);
                } else {
                    let _ = fs::write(api_port_path(&cwd), format!("{}\n", port));
                }
                println!("daemon: running (http://{}:{})", host, port);
                return;
            }
            if !port_available(&host, port) {
                if let Some(free) = find_free_port(&host, daemon_default_port(), 100) {
                    eprintln!("cypress daemon: port {} busy; using {}", port, free);
                    port = free;
                } else {
                    eprintln!(
                        "error: no free {} port available near {}",
                        host,
                        daemon_default_port()
                    );
                    process::exit(1);
                }
            }
            let pid = spawn_api_daemon(&cwd, &host, port).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
            let deadline = Instant::now() + Duration::from_secs(timeout_s);
            while Instant::now() < deadline {
                if api_health_host_port(&host, port) {
                    persist_api_runtime_state(&cwd, pid, port).unwrap_or_else(|e| {
                        eprintln!("warning: {}", e);
                    });
                    println!("daemon: running (http://{}:{})", host, port);
                    return;
                }
                if !pid_is_alive(pid) {
                    break;
                }
                thread::sleep(Duration::from_millis(100));
            }
            let _ = run_shell_capture(&format!("kill {} >/dev/null 2>&1 || true", pid));
            clear_api_runtime_state(&cwd);
            eprintln!(
                "cypress daemon failed to start; see {}",
                api_log_path(&cwd).display()
            );
            process::exit(1);
        }
        "stop" => {
            if let Some(pid) = read_saved_api_pid(&cwd) {
                let _ = run_shell_capture(&format!("kill {} >/dev/null 2>&1 || true", pid));
                let deadline = Instant::now() + Duration::from_secs(timeout_s);
                while Instant::now() < deadline {
                    if !pid_is_alive(pid) {
                        break;
                    }
                    thread::sleep(Duration::from_millis(100));
                }
                if pid_is_alive(pid) {
                    let _ = run_shell_capture(&format!("kill -9 {} >/dev/null 2>&1 || true", pid));
                }
            }
            clear_api_runtime_state(&cwd);
            println!("daemon: stopped");
        }
        "restart" => {
            if let Some(pid) = read_saved_api_pid(&cwd) {
                let _ = run_shell_capture(&format!("kill {} >/dev/null 2>&1 || true", pid));
                thread::sleep(Duration::from_millis(150));
                if pid_is_alive(pid) {
                    let _ = run_shell_capture(&format!("kill -9 {} >/dev/null 2>&1 || true", pid));
                }
            }
            clear_api_runtime_state(&cwd);
            if !port_available(&host, port) {
                if let Some(free) = find_free_port(&host, daemon_default_port(), 100) {
                    eprintln!("cypress daemon: port {} busy; using {}", port, free);
                    port = free;
                } else {
                    eprintln!(
                        "error: no free {} port available near {}",
                        host,
                        daemon_default_port()
                    );
                    process::exit(1);
                }
            }
            let pid = spawn_api_daemon(&cwd, &host, port).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
            let deadline = Instant::now() + Duration::from_secs(timeout_s);
            while Instant::now() < deadline {
                if api_health_host_port(&host, port) {
                    persist_api_runtime_state(&cwd, pid, port).unwrap_or_else(|e| {
                        eprintln!("warning: {}", e);
                    });
                    println!("daemon: running (http://{}:{})", host, port);
                    return;
                }
                if !pid_is_alive(pid) {
                    break;
                }
                thread::sleep(Duration::from_millis(100));
            }
            let _ = run_shell_capture(&format!("kill {} >/dev/null 2>&1 || true", pid));
            clear_api_runtime_state(&cwd);
            eprintln!(
                "cypress daemon failed to start; see {}",
                api_log_path(&cwd).display()
            );
            process::exit(1);
        }
        "status" => {
            let mut status_port = port;
            if let Some(saved) = read_saved_api_port(&cwd) {
                status_port = saved;
            }
            if api_health_host_port(&host, status_port) {
                println!("daemon: running (http://{}:{})", host, status_port);
            } else {
                println!("daemon: stopped");
                process::exit(1);
            }
        }
        "logs" => {
            let log = api_log_path(&cwd);
            if !log.exists() {
                eprintln!("error: log file not found: {}", log.display());
                process::exit(1);
            }
            tail_file_lines(&log, logs_n).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
        }
        other => {
            eprintln!("error: unknown daemon action '{}'", other);
            process::exit(2);
        }
    }
}

fn graph_viewer_html() -> &'static str {
    r##"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Cypress Graph Viewer</title>
  <style>
    :root {
      --bg: #0b1220;
      --panel: #111a2d;
      --line: #223252;
      --text: #d9e4ff;
      --muted: #9fb4d9;
      --accent: #65d9a7;
      --accent2: #7eb5ff;
      --warn: #ffd37a;
    }
    html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text); font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, "Courier New", monospace; }
    .wrap { display: grid; grid-template-rows: auto 1fr; min-height: 100vh; }
    .top {
      display: flex; gap: 10px; align-items: center; padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(120deg, #0f172a, #0c1a35 48%, #0f2c2b);
    }
    input, button, select {
      background: #0b162b; color: var(--text); border: 1px solid var(--line); border-radius: 7px;
      padding: 8px 10px; font: inherit;
    }
    button { cursor: pointer; }
    button:hover { border-color: var(--accent2); }
    .grid { display: grid; grid-template-columns: 1.4fr 1fr; min-height: 0; }
    .panel { min-height: 0; border-right: 1px solid var(--line); }
    .panel:last-child { border-right: none; }
    #graphBox { position: relative; height: calc(100vh - 58px); }
    svg { width: 100%; height: 100%; display: block; background: radial-gradient(circle at 40% 30%, #14213d 0%, #0b1220 70%); }
    .label { fill: var(--text); font-size: 12px; pointer-events: none; }
    .muted { color: var(--muted); }
    .right { height: calc(100vh - 58px); overflow: auto; padding: 10px; }
    .card { border: 1px solid var(--line); border-radius: 10px; padding: 10px; margin-bottom: 10px; background: var(--panel); }
    .small { font-size: 12px; color: var(--muted); }
    .mono { white-space: pre-wrap; word-break: break-word; }
    .chunk-actions { margin-top: 8px; display: flex; gap: 8px; }
    .chip { color: var(--warn); font-size: 12px; }
    a { color: var(--accent2); text-decoration: none; }
  </style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <strong>Cypress Graph Viewer</strong>
    <input id="focus" placeholder="focus project path (optional)" style="flex:1" />
    <select id="limit">
      <option value="80">80 nodes</option>
      <option value="120" selected>120 nodes</option>
      <option value="200">200 nodes</option>
      <option value="350">350 nodes</option>
    </select>
    <button id="reload">Reload</button>
  </div>
  <div class="grid">
    <div class="panel" id="graphBox"><svg id="graph"></svg></div>
    <div class="panel right">
      <div class="card">
        <div><strong>Selected Project</strong></div>
        <div id="selectedProject" class="small">none</div>
      </div>
      <div class="card">
        <div><strong>Selected Chunk</strong></div>
        <div id="selectedChunk" class="small">none</div>
      </div>
      <div class="card">
        <div><strong>Project Chunks</strong></div>
        <div id="chunks" class="small">select a project node</div>
      </div>
      <div class="card">
        <div><strong>Related Chunks</strong></div>
        <div id="related" class="small">select a chunk</div>
      </div>
      <div class="card">
        <div><strong>Explainability</strong></div>
        <div id="explain" class="small">select a related chunk</div>
      </div>
      <div class="card">
        <div><strong>Suppressed Relations</strong></div>
        <div id="suppressed" class="small">select a chunk</div>
      </div>
    </div>
  </div>
</div>
<script>
const q = (id) => document.getElementById(id);
const state = {
  nodes: [],
  edges: [],
  selectedProject: "",
  selectedChunk: 0,
  relatedRows: [],
  relatedSource: null,
  selectedRelationIdx: -1,
  feedbackRows: []
};
async function jget(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}
async function jpost(path, payload) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}
function esc(v) {
  return String(v ?? "").replace(/[&<>"]/g, c => ({ "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;" }[c]));
}
function relationReason(rel) {
  switch (String(rel || "")) {
    case "same_file": return "Connected because both chunks are from the same file.";
    case "same_project": return "Connected because both chunks are in the same project.";
    case "project_edge": return "Connected through a project-level RELATED graph edge.";
    case "related_project": return "Connected through graph expansion from a related project.";
    case "seed": return "Top seed chunk directly retrieved for this context.";
    default: return "Connected by hybrid semantic/lexical ranking and graph context.";
  }
}
function relationQualityBadge(v) {
  const qv = String(v || "unspecified");
  if (qv === "good") return "good";
  if (qv === "weak") return "weak";
  if (qv === "wrong") return "wrong";
  return "unspecified";
}
function renderExplain() {
  if (!state.relatedRows.length || state.selectedRelationIdx < 0 || !state.relatedSource) {
    q("explain").innerHTML = `<div class="small">select a related chunk</div>`;
    return;
  }
  const src = state.relatedSource;
  const r = state.relatedRows[state.selectedRelationIdx];
  if (!r) {
    q("explain").innerHTML = `<div class="small">select a related chunk</div>`;
    return;
  }
  q("explain").innerHTML = `
    <div><span class="chip">${esc(r.relation)}</span> #${esc(r.chunk_id)}:${esc(r.chunk_index)} ${esc(r.doc_rel_path || "")}</div>
    <div class="small">source: #${esc(src.chunk_id)}:${esc(src.chunk_index)} ${esc(src.doc_rel_path || "")}</div>
    <div class="small">score=${Number(r.score || 0).toFixed(3)} relation_weight=${Number(r.relation_weight || 0).toFixed(3)}</div>
    <div class="small">semantic=${Number(r.semantic || 0).toFixed(3)} lexical=${Number(r.lexical || 0).toFixed(3)} quality=${Number(r.quality || 0).toFixed(3)}</div>
    <div class="small">relation_quality=${esc(relationQualityBadge(r.relation_quality))} multiplier=${Number(r.relation_quality_multiplier || 1).toFixed(3)}</div>
    <div class="small">${esc(relationReason(r.relation))}</div>
  `;
}
function setExplain(idx) {
  state.selectedRelationIdx = Number(idx || 0);
  renderExplain();
}
function drillChunk(chunkId) {
  if (!chunkId) return;
  loadRelated(Number(chunkId)).catch((e) => alert(e.message));
}
async function loadSuppressed(chunkId) {
  if (!chunkId) {
    state.feedbackRows = [];
    q("suppressed").innerHTML = `<div class="small">select a chunk</div>`;
    return;
  }
  const data = await jget(`/chunks/feedback?chunk_id=${encodeURIComponent(chunkId)}&decision=suppressed&limit=120`);
  const rows = data.results || [];
  state.feedbackRows = rows;
  if (!rows.length) {
    q("suppressed").innerHTML = `<div class="small">none</div>`;
    return;
  }
  q("suppressed").innerHTML = rows.map((r, i) => `
    <div style="padding:8px 0;border-bottom:1px solid #223252;">
      <div><span class="chip">${esc(r.relation)}</span> #${esc(r.target_chunk_id)}:${esc(r.target_chunk_index)} ${esc(r.target_doc_rel_path || "")}</div>
      <div class="small">quality=${esc(relationQualityBadge(r.quality_label))}</div>
      <div class="small">updated=${new Date((Number(r.updated_at || 0) * 1000)).toLocaleString()} source=${esc(r.source || "-")}</div>
      ${r.note ? `<div class="small mono">${esc(r.note)}</div>` : ""}
      <div class="chunk-actions">
        <button onclick="restoreSuppressed(${Number(i)})">Restore</button>
      </div>
    </div>
  `).join("");
}
async function suppressRelated(idx) {
  const i = Number(idx || 0);
  const row = state.relatedRows[i];
  if (!row || !state.selectedChunk) return;
  await jpost("/chunks/feedback/suppress", {
    source_chunk_id: Number(state.selectedChunk),
    target_chunk_id: Number(row.chunk_id || 0),
    relation: String(row.relation || ""),
    note: "suppressed via graph viewer",
  });
  await loadRelated(Number(state.selectedChunk));
}
async function restoreSuppressed(idx) {
  const i = Number(idx || 0);
  const row = state.feedbackRows[i];
  if (!row || !state.selectedChunk) return;
  await jpost("/chunks/feedback/restore", {
    source_chunk_id: Number(state.selectedChunk),
    target_chunk_id: Number(row.target_chunk_id || 0),
    relation: String(row.relation || ""),
    note: "restored via graph viewer",
  });
  await loadRelated(Number(state.selectedChunk));
}
async function setRelatedQuality(idx, qualityLabel) {
  const i = Number(idx || 0);
  const row = state.relatedRows[i];
  if (!row || !state.selectedChunk) return;
  await jpost("/chunks/feedback/quality", {
    source_chunk_id: Number(state.selectedChunk),
    target_chunk_id: Number(row.chunk_id || 0),
    relation: String(row.relation || ""),
    quality_label: String(qualityLabel || "unspecified"),
    note: "quality set via graph viewer",
  });
  await loadRelated(Number(state.selectedChunk));
}
async function loadGraph() {
  const focus = q("focus").value.trim();
  const limit = q("limit").value;
  const url = `/graph/view/data?limit=${encodeURIComponent(limit)}${focus ? `&focus=${encodeURIComponent(focus)}` : ""}`;
  const data = await jget(url);
  state.nodes = data.nodes || [];
  state.edges = data.edges || [];
  drawGraph();
}
function drawGraph() {
  const svg = q("graph");
  svg.innerHTML = "";
  const box = svg.getBoundingClientRect();
  const w = Math.max(300, box.width);
  const h = Math.max(300, box.height);
  const cx = w / 2;
  const cy = h / 2;
  const n = state.nodes.length;
  if (!n) {
    svg.innerHTML = `<text x="24" y="40" class="label">No graph data yet. Run cypress index first.</text>`;
    return;
  }
  const radius = Math.max(70, Math.min(w, h) * 0.40);
  const pos = new Map();
  state.nodes.forEach((node, i) => {
    const a = (Math.PI * 2 * i / n) - (Math.PI / 2);
    const x = cx + Math.cos(a) * radius;
    const y = cy + Math.sin(a) * radius;
    pos.set(node.id, { x, y, node });
  });
  state.edges.forEach((e) => {
    const s = pos.get(e.source);
    const t = pos.get(e.target);
    if (!s || !t) return;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", s.x); line.setAttribute("y1", s.y);
    line.setAttribute("x2", t.x); line.setAttribute("y2", t.y);
    const w = Math.max(0.6, Math.min(4.0, Number(e.weight || 0.5) * 2.5));
    line.setAttribute("stroke-width", w);
    line.setAttribute("stroke", "rgba(126,181,255,0.38)");
    svg.appendChild(line);
  });
  state.nodes.forEach((node) => {
    const p = pos.get(node.id);
    if (!p) return;
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    c.setAttribute("cx", p.x); c.setAttribute("cy", p.y);
    c.setAttribute("r", state.selectedProject === node.path ? 8 : 6);
    c.setAttribute("fill", state.selectedProject === node.path ? "#65d9a7" : "#7eb5ff");
    c.style.cursor = "pointer";
    c.onclick = () => selectProject(node.path);
    const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("x", p.x + 9); t.setAttribute("y", p.y + 4);
    t.setAttribute("class", "label");
    t.textContent = node.label || node.path;
    g.appendChild(c); g.appendChild(t);
    svg.appendChild(g);
  });
}
async function selectProject(path) {
  state.selectedProject = path;
  state.selectedChunk = 0;
  state.relatedRows = [];
  state.relatedSource = null;
  state.selectedRelationIdx = -1;
  state.feedbackRows = [];
  q("selectedProject").textContent = path;
  q("selectedChunk").textContent = "none";
  q("related").innerHTML = `<div class="small">select a chunk</div>`;
  q("suppressed").innerHTML = `<div class="small">select a chunk</div>`;
  renderExplain();
  drawGraph();
  const data = await jget(`/graph/view/chunks?path=${encodeURIComponent(path)}&limit=40`);
  const rows = data.chunks || [];
  if (!rows.length) {
    q("chunks").innerHTML = `<div class="small">no chunks</div>`;
    return;
  }
  q("chunks").innerHTML = rows.map((r) => `
    <div style="padding:8px 0;border-bottom:1px solid #223252;">
      <div><span class="chip">#${esc(r.chunk_id)}:${esc(r.chunk_index)}</span> ${esc(r.doc_rel_path)}</div>
      <div class="small">tokens=${Number(r.token_count || 0)}</div>
      <div class="small mono">${esc(r.excerpt)}</div>
      <div class="chunk-actions">
        <button onclick="loadRelated(${Number(r.chunk_id || 0)})">Related</button>
      </div>
    </div>
  `).join("");
}
async function loadRelated(chunkId) {
  if (!chunkId) return;
  state.selectedChunk = chunkId;
  const data = await jget(`/graph/view/related?chunk_id=${encodeURIComponent(chunkId)}&limit=20`);
  const src = data.source || {};
  const rows = data.results || [];
  state.relatedSource = src;
  state.relatedRows = rows;
  state.selectedRelationIdx = rows.length ? 0 : -1;
  q("selectedChunk").textContent = src.chunk_id
    ? `#${src.chunk_id}:${src.chunk_index} ${src.doc_rel_path || ""}`
    : `#${chunkId}`;
  if (!rows.length) {
    q("related").innerHTML = `<div class="small">no related chunks for ${esc(chunkId)}</div>`;
    renderExplain();
    await loadSuppressed(Number(chunkId));
    return;
  }
  q("related").innerHTML = `
    <div class="small">source: #${esc(src.chunk_id)}:${esc(src.chunk_index)} ${esc(src.doc_rel_path || "")}</div>
    ${rows.map((r, i) => `
      <div style="padding:8px 0;border-bottom:1px solid #223252;">
        <div><span class="chip">${esc(r.relation)}</span> #${esc(r.chunk_id)}:${esc(r.chunk_index)} ${esc(r.doc_rel_path)}</div>
        <div class="small">score=${Number(r.score || 0).toFixed(3)} relation_weight=${Number(r.relation_weight || 0).toFixed(3)} sem=${Number(r.semantic || 0).toFixed(2)} lex=${Number(r.lexical || 0).toFixed(2)} q=${Number(r.quality || 0).toFixed(2)}</div>
        <div class="small">relation_quality=${esc(relationQualityBadge(r.relation_quality))} multiplier=${Number(r.relation_quality_multiplier || 1).toFixed(2)}</div>
        <div class="small mono">${esc(r.excerpt)}</div>
        <div class="chunk-actions">
          <button onclick="setExplain(${Number(i)})">Why</button>
          <button onclick="drillChunk(${Number(r.chunk_id || 0)})">Drill</button>
          <button onclick="suppressRelated(${Number(i)}).catch(e => alert(e.message))">Suppress</button>
          <button onclick="setRelatedQuality(${Number(i)}, 'good').catch(e => alert(e.message))">Good</button>
          <button onclick="setRelatedQuality(${Number(i)}, 'weak').catch(e => alert(e.message))">Weak</button>
          <button onclick="setRelatedQuality(${Number(i)}, 'wrong').catch(e => alert(e.message))">Wrong</button>
          <button onclick="setRelatedQuality(${Number(i)}, 'unspecified').catch(e => alert(e.message))">Clear</button>
        </div>
      </div>
    `).join("")}
  `;
  renderExplain();
  await loadSuppressed(Number(chunkId));
}
q("reload").onclick = () => loadGraph().catch((e) => alert(e.message));
loadGraph().catch((e) => alert(e.message));
</script>
</body>
</html>"##
}

fn send_http_text(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &str,
) -> Result<(), String> {
    let payload = body.as_bytes();
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "OK",
    };
    let headers = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status,
        reason,
        content_type,
        payload.len()
    );
    stream
        .write_all(headers.as_bytes())
        .and_then(|_| stream.write_all(payload))
        .map_err(|e| format!("failed writing HTTP text response: {}", e))
}

fn serve_graph_viewer(host: &str, port: u16) -> Result<(), String> {
    let listener = TcpListener::bind((host, port))
        .map_err(|e| format!("failed to bind graph viewer listener: {}", e))?;
    println!(
        "cypress graph viewer listening on http://{}:{}/",
        host, port
    );
    println!(
        "endpoints: GET /, GET /health, GET /graph/view/data, GET /graph/view/chunks, GET /graph/view/related, GET /chunks/feedback, POST /chunks/feedback/suppress, POST /chunks/feedback/restore, POST /chunks/feedback/quality"
    );
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else {
            continue;
        };
        thread::spawn(move || {
            let req = match parse_http_request(&mut stream) {
                Ok(Some(v)) => v,
                Ok(None) => return,
                Err(err) => {
                    eprintln!("graph-viewer: {}", err);
                    return;
                }
            };
            if req.method == "GET" && (req.path == "/" || req.path == "/index.html") {
                if let Err(err) = send_http_text(
                    &mut stream,
                    200,
                    "text/html; charset=utf-8",
                    graph_viewer_html(),
                ) {
                    eprintln!("graph-viewer: {}", err);
                }
                return;
            }
            let (status, payload) = handle_api_request(req);
            if let Err(err) = send_http_json(&mut stream, status, &payload) {
                eprintln!("graph-viewer: {}", err);
            }
        });
    }
    Ok(())
}

fn run_api_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: cypress api [--host <addr>] [--port <n>]");
        return;
    }
    let mut host = "127.0.0.1".to_string();
    let mut port: u16 = 8765;

    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        match s.as_str() {
            "--host" => {
                i += 1;
                host = arg_value(args, i, "--host");
            }
            "--port" => {
                i += 1;
                let raw = arg_value(args, i, "--port");
                port = raw.parse::<u16>().unwrap_or_else(|_| {
                    eprintln!("error: --port must be an integer between 1 and 65535");
                    process::exit(2);
                });
            }
            other => {
                eprintln!("error: unknown option '{}'", other);
                process::exit(2);
            }
        }
        i += 1;
    }
    serve_api_native(&host, port).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
}

#[derive(Debug)]
struct ApiRequest {
    method: String,
    path: String,
    query: HashMap<String, String>,
    body: Vec<u8>,
}

fn serve_api_native(host: &str, port: u16) -> Result<(), String> {
    let listener = TcpListener::bind((host, port))
        .map_err(|e| format!("failed to bind api listener: {}", e))?;
    println!("cypress api listening on http://{}:{}", host, port);
    println!(
        "endpoints: GET /health, GET /search, GET /chunks/search, GET /chunks/related, GET /chunks/get, GET /docs/read, GET /chunks/feedback, GET /tracked, GET /graph/neighbors, GET /graph/view/data, GET /graph/view/chunks, GET /graph/view/related, POST /chunks/feedback/suppress, POST /chunks/feedback/restore, POST /chunks/feedback/quality, POST /refresh, POST /select, POST /tracked/add, POST /tracked/del"
    );
    for stream in listener.incoming() {
        let Ok(stream) = stream else {
            continue;
        };
        thread::spawn(move || {
            if let Err(err) = handle_api_connection(stream) {
                eprintln!("api: {}", err);
            }
        });
    }
    Ok(())
}

fn handle_api_connection(mut stream: TcpStream) -> Result<(), String> {
    let req = match parse_http_request(&mut stream)? {
        Some(v) => v,
        None => return Ok(()),
    };
    let (status, payload) = handle_api_request(req);
    send_http_json(&mut stream, status, &payload)
}

fn parse_http_request(stream: &mut TcpStream) -> Result<Option<ApiRequest>, String> {
    let cloned = stream
        .try_clone()
        .map_err(|e| format!("failed to clone stream: {}", e))?;
    let mut reader = BufReader::new(cloned);
    let mut request_line = String::new();
    if reader
        .read_line(&mut request_line)
        .map_err(|e| format!("failed reading request line: {}", e))?
        == 0
    {
        return Ok(None);
    }
    let request_line = request_line.trim_end_matches(['\r', '\n']).to_string();
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("").to_string();
    if method.is_empty() || target.is_empty() {
        return Ok(None);
    }

    let mut headers: HashMap<String, String> = HashMap::new();
    loop {
        let mut line = String::new();
        let n = reader
            .read_line(&mut line)
            .map_err(|e| format!("failed reading header line: {}", e))?;
        if n == 0 {
            break;
        }
        let line = line.trim_end_matches(['\r', '\n']).to_string();
        if line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }

    let content_len = headers
        .get("content-length")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let mut body = vec![0u8; content_len];
    if content_len > 0 {
        reader
            .read_exact(&mut body)
            .map_err(|e| format!("failed reading request body: {}", e))?;
    }

    let (path_raw, query_raw) = match target.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (target, String::new()),
    };
    Ok(Some(ApiRequest {
        method,
        path: path_raw,
        query: parse_query_params(&query_raw),
        body,
    }))
}

fn parse_query_params(raw: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for part in raw.split('&') {
        if part.is_empty() {
            continue;
        }
        let (k, v) = match part.split_once('=') {
            Some((k, v)) => (k, v),
            None => (part, ""),
        };
        out.insert(url_decode_component(k), url_decode_component(v));
    }
    out
}

fn hex_val(c: char) -> Option<u8> {
    match c {
        '0'..='9' => Some((c as u8) - b'0'),
        'a'..='f' => Some((c as u8) - b'a' + 10),
        'A'..='F' => Some((c as u8) - b'A' + 10),
        _ => None,
    }
}

fn url_decode_component(raw: &str) -> String {
    let mut out: Vec<u8> = Vec::with_capacity(raw.len());
    let chars: Vec<char> = raw.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        let ch = chars[i];
        if ch == '+' {
            out.push(b' ');
            i += 1;
            continue;
        }
        if ch == '%' && i + 2 < chars.len() {
            if let (Some(h1), Some(h2)) = (hex_val(chars[i + 1]), hex_val(chars[i + 2])) {
                out.push((h1 << 4) | h2);
                i += 3;
                continue;
            }
        }
        let mut buf = [0u8; 4];
        let encoded = ch.encode_utf8(&mut buf);
        out.extend_from_slice(encoded.as_bytes());
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

fn parse_limit(raw: Option<&String>, default: usize, max_limit: usize) -> usize {
    let parsed = raw.and_then(|v| v.parse::<usize>().ok()).unwrap_or(default);
    parsed.max(1).min(max_limit)
}

fn ensure_native_embed_backend(cfg: &ConfigValues, context: &str) -> Result<(), String> {
    if cfg.embed_backend == "local"
        || cfg.embed_backend == "local-hash"
        || cfg.embed_backend == "ollama"
    {
        return Ok(());
    }
    Err(format!(
        "{} requires native embed_backend local-hash/ollama (or local alias) (current='{}')",
        context, cfg.embed_backend
    ))
}

fn list_neighbors_by_path(
    conn: &Connection,
    path: &str,
    limit: usize,
) -> Result<Vec<(String, String, f64)>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT pe.dst, pe.kind, pe.weight
FROM projects p
JOIN project_edges pe ON pe.src_project_id = p.id
WHERE p.path = ?1
ORDER BY pe.weight DESC, pe.dst ASC
LIMIT ?2
"#,
        )
        .map_err(|e| format!("failed preparing neighbor query: {}", e))?;
    let rows = stmt
        .query_map(params![path, limit as i64], |row| {
            let dst: String = row.get(0)?;
            let kind: String = row.get(1)?;
            let weight: f64 = row.get(2)?;
            Ok((dst, kind, weight))
        })
        .map_err(|e| format!("failed querying neighbors: {}", e))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(|e| format!("failed reading neighbor row: {}", e))?);
    }
    Ok(out)
}

fn payload_paths(body: &Value) -> Vec<String> {
    if let Some(paths) = body.get("paths").and_then(|v| v.as_array()) {
        return paths
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();
    }
    if let Some(path) = body.get("path").and_then(|v| v.as_str()) {
        return vec![path.to_string()];
    }
    Vec::new()
}

fn payload_i64(body: &Value, key: &str) -> i64 {
    if let Some(v) = body.get(key) {
        if let Some(n) = v.as_i64() {
            return n;
        }
        if let Some(s) = v.as_str() {
            if let Ok(n) = s.trim().parse::<i64>() {
                return n;
            }
        }
    }
    0
}

fn payload_string(body: &Value, key: &str) -> String {
    body.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn unique_valid_dirs(paths: &[String]) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for raw in paths {
        let p = normalize_path(raw);
        if !p.exists() || !p.is_dir() {
            continue;
        }
        let key = p.to_string_lossy().to_string();
        if seen.insert(key) {
            out.push(p);
        }
    }
    out
}

fn stats_payload_json(stats: &IndexStats) -> Value {
    serde_json::json!({
        "total_projects": stats.total_projects,
        "updated_projects": stats.updated_projects,
        "skipped_projects": stats.skipped_projects,
        "removed_projects": stats.removed_projects,
        "vectorized_projects": stats.vectorized_projects,
        "vector_failures": stats.vector_failures,
        "tracked_roots": stats.tracked_roots,
        "graph_edges": stats.graph_edges,
        "chunk_rows": stats.chunk_rows,
        "chunk_vectors": stats.chunk_vectors,
        "retrieval_backend": stats.retrieval_backend,
        "retrieval_synced_chunks": stats.retrieval_synced_chunks,
        "retrieval_error": stats.retrieval_error,
    })
}

fn path_basename(path: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path)
        .to_string()
}

fn graph_view_data_json(
    conn: &Connection,
    focus: Option<&str>,
    limit: usize,
) -> Result<Value, String> {
    let use_limit = limit.max(10).min(600);
    let mut selected: Vec<String> = Vec::new();
    if let Some(raw_focus) = focus {
        let focus_path = normalize_path(raw_focus).to_string_lossy().to_string();
        selected.push(focus_path.clone());
        let outgoing = list_neighbors_by_path(conn, &focus_path, use_limit)?;
        for (dst, _, _) in outgoing {
            selected.push(dst);
        }
        let mut incoming_stmt = conn
            .prepare(
                r#"
SELECT src.path, pe.kind, pe.weight
FROM project_edges pe
JOIN projects src ON src.id = pe.src_project_id
WHERE pe.dst = ?1
ORDER BY pe.weight DESC
LIMIT ?2
"#,
            )
            .map_err(|e| format!("failed preparing incoming neighbor query: {}", e))?;
        let incoming = incoming_stmt
            .query_map(params![focus_path, use_limit as i64], |row| {
                let path: String = row.get(0)?;
                let kind: String = row.get(1)?;
                let weight: f64 = row.get(2)?;
                Ok((path, kind, weight))
            })
            .map_err(|e| format!("failed querying incoming neighbor rows: {}", e))?;
        for row in incoming {
            let (path, _, _) =
                row.map_err(|e| format!("failed reading incoming neighbor row: {}", e))?;
            selected.push(path);
        }
    } else {
        let mut stmt = conn
            .prepare(
                r#"
SELECT p.path, (
    COALESCE(out_deg.cnt, 0) + COALESCE(in_deg.cnt, 0)
) AS degree
FROM projects p
LEFT JOIN (
    SELECT src_project_id, COUNT(*) AS cnt
    FROM project_edges
    GROUP BY src_project_id
) out_deg ON out_deg.src_project_id = p.id
LEFT JOIN (
    SELECT dst, COUNT(*) AS cnt
    FROM project_edges
    GROUP BY dst
) in_deg ON in_deg.dst = p.path
ORDER BY degree DESC, p.path ASC
LIMIT ?1
"#,
            )
            .map_err(|e| format!("failed preparing project degree query: {}", e))?;
        let rows = stmt
            .query_map(params![use_limit as i64], |row| row.get::<_, String>(0))
            .map_err(|e| format!("failed querying project degree rows: {}", e))?;
        for row in rows {
            selected.push(row.map_err(|e| format!("failed reading project degree row: {}", e))?);
        }
    }

    selected.sort();
    selected.dedup();
    if selected.is_empty() {
        return Ok(serde_json::json!({"focus": focus, "nodes": [], "edges": []}));
    }
    if selected.len() > use_limit {
        selected.truncate(use_limit);
    }
    let selected_set: HashSet<String> = selected.iter().cloned().collect();

    let nodes: Vec<Value> = selected
        .iter()
        .map(|path| {
            serde_json::json!({
                "id": path,
                "label": path_basename(path),
                "path": path
            })
        })
        .collect();

    let mut edge_stmt = conn
        .prepare(
            r#"
SELECT src.path, pe.dst, pe.kind, pe.weight
FROM project_edges pe
JOIN projects src ON src.id = pe.src_project_id
ORDER BY pe.weight DESC, src.path ASC, pe.dst ASC
"#,
        )
        .map_err(|e| format!("failed preparing graph edge query: {}", e))?;
    let edge_rows = edge_stmt
        .query_map([], |row| {
            let src: String = row.get(0)?;
            let dst: String = row.get(1)?;
            let kind: String = row.get(2)?;
            let weight: f64 = row.get(3)?;
            Ok((src, dst, kind, weight))
        })
        .map_err(|e| format!("failed querying graph edge rows: {}", e))?;

    let mut edges: Vec<Value> = Vec::new();
    for row in edge_rows {
        let (src, dst, kind, weight) =
            row.map_err(|e| format!("failed reading graph edge row: {}", e))?;
        if selected_set.contains(&src) || selected_set.contains(&dst) {
            edges.push(serde_json::json!({
                "source": src,
                "target": dst,
                "kind": kind,
                "weight": weight
            }));
            if edges.len() >= use_limit.saturating_mul(8) {
                break;
            }
        }
    }

    Ok(serde_json::json!({
        "focus": focus.unwrap_or(""),
        "nodes": nodes,
        "edges": edges
    }))
}

fn project_chunks_preview_json(
    conn: &Connection,
    project_path: &str,
    limit: usize,
) -> Result<Value, String> {
    let target = normalize_path(project_path).to_string_lossy().to_string();
    let mut stmt = conn
        .prepare(
            r#"
SELECT pc.id, pc.doc_path, pc.doc_rel_path, pc.chunk_index, pc.token_count, pc.text
FROM project_chunks pc
JOIN projects p ON p.id = pc.project_id
WHERE p.path = ?1
ORDER BY pc.updated_at DESC, pc.id DESC
LIMIT ?2
"#,
        )
        .map_err(|e| format!("failed preparing project chunk preview query: {}", e))?;
    let rows = stmt
        .query_map(params![target, limit.max(1).min(400) as i64], |row| {
            let chunk_id: i64 = row.get(0)?;
            let doc_path: String = row.get(1)?;
            let doc_rel_path: String = row.get(2)?;
            let chunk_index: i64 = row.get(3)?;
            let token_count: i64 = row.get(4)?;
            let text: String = row.get(5)?;
            Ok((
                chunk_id,
                doc_path,
                doc_rel_path,
                chunk_index,
                token_count,
                text,
            ))
        })
        .map_err(|e| format!("failed querying project chunk preview rows: {}", e))?;
    let mut chunks: Vec<Value> = Vec::new();
    for row in rows {
        let (chunk_id, doc_path, doc_rel_path, chunk_index, token_count, text) =
            row.map_err(|e| format!("failed reading project chunk preview row: {}", e))?;
        chunks.push(serde_json::json!({
            "chunk_id": chunk_id,
            "path": doc_path,
            "doc_rel_path": doc_rel_path,
            "chunk_index": chunk_index,
            "token_count": token_count,
            "excerpt": clip_text(&text, 260),
        }));
    }
    Ok(serde_json::json!({"path": project_path, "count": chunks.len(), "chunks": chunks}))
}

fn send_http_json(stream: &mut TcpStream, status: u16, payload: &Value) -> Result<(), String> {
    let body =
        serde_json::to_vec(payload).map_err(|e| format!("failed serializing JSON: {}", e))?;
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "OK",
    };
    let headers = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status,
        reason,
        body.len()
    );
    stream
        .write_all(headers.as_bytes())
        .and_then(|_| stream.write_all(&body))
        .map_err(|e| format!("failed writing HTTP response: {}", e))
}

fn parse_json_body(body: &[u8]) -> Value {
    if body.is_empty() {
        return Value::Object(serde_json::Map::new());
    }
    serde_json::from_slice::<Value>(body)
        .ok()
        .filter(|v| v.is_object())
        .unwrap_or_else(|| Value::Object(serde_json::Map::new()))
}

fn handle_api_request(req: ApiRequest) -> (u16, Value) {
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => {
            return (200, serde_json::json!({"ok": true, "time": now_ts()}));
        }
        ("GET", "/tracked") => {
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
            let dbp = db_path(&cwd);
            if let Err(e) = ensure_db_schema(&dbp) {
                return (500, serde_json::json!({"error": e}));
            }
            if let Ok(rows) = list_tracked_roots(&dbp) {
                if rows.is_empty() {
                    if let Err(e) = ensure_tracked_root(&dbp, &cfg.root, now_ts()) {
                        return (500, serde_json::json!({"error": e}));
                    }
                }
            }
            let rows = match list_tracked_roots(&dbp) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let values: Vec<String> = rows
                .into_iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect();
            return (200, serde_json::json!({"tracked_roots": values}));
        }
        ("GET", "/search") => {
            let query = req.query.get("q").cloned().unwrap_or_default();
            let q = query.trim().to_string();
            if q.is_empty() {
                return (
                    400,
                    serde_json::json!({"error": "Missing query parameter 'q'."}),
                );
            }
            let limit = parse_limit(req.query.get("limit"), 20, 200);
            let view = req
                .query
                .get("view")
                .cloned()
                .unwrap_or_else(|| "projects".to_string())
                .trim()
                .to_lowercase();
            if view != "projects" && view != "files" {
                return (
                    400,
                    serde_json::json!({"error": "Invalid view. Use 'projects' or 'files'."}),
                );
            }
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
            if let Err(e) = ensure_native_embed_backend(&cfg, "api search") {
                return (503, serde_json::json!({"error": e}));
            }
            if let Err(e) = ensure_retrieval_backend_ready(&cfg, true, "api search") {
                return (503, serde_json::json!({"error": e}));
            }
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            if view == "files" {
                let rows = match rank_files_native(&conn, &cfg, &q, limit) {
                    Ok(v) => v,
                    Err(e) => return (503, serde_json::json!({"error": e})),
                };
                let results: Vec<Value> = rows
                    .into_iter()
                    .map(|item| {
                        let evidence: Vec<Value> = item
                            .evidence
                            .into_iter()
                            .map(|ev| {
                                serde_json::json!({
                                    "chunk_id": ev.chunk_id,
                                    "chunk_index": ev.chunk_index,
                                    "doc_path": ev.doc_path,
                                    "doc_rel_path": ev.doc_rel_path,
                                    "score": ev.score,
                                    "semantic": ev.semantic,
                                    "lexical": ev.lexical,
                                    "graph": ev.graph,
                                    "relation": ev.relation,
                                    "quality": ev.quality,
                                    "excerpt": ev.excerpt,
                                })
                            })
                            .collect();
                        serde_json::json!({
                            "path": item.path,
                            "project_path": item.project_path,
                            "doc_rel_path": item.doc_rel_path,
                            "chunk_id": item.chunk_id,
                            "chunk_index": item.chunk_index,
                            "score": item.score,
                            "semantic": item.semantic,
                            "lexical": item.lexical,
                            "graph": item.graph,
                            "relation": item.relation,
                            "quality": item.quality,
                            "excerpt": item.excerpt,
                            "evidence": evidence,
                        })
                    })
                    .collect();
                return (
                    200,
                    serde_json::json!({"query": q, "view": "files", "results": results}),
                );
            }
            let rows = match rank_projects_native(&conn, &cfg, &q, limit) {
                Ok(v) => v,
                Err(e) => return (503, serde_json::json!({"error": e})),
            };
            let results: Vec<Value> = rows
                .into_iter()
                .map(|item| {
                    let evidence: Vec<Value> = item
                        .evidence
                        .into_iter()
                        .map(|ev| {
                            serde_json::json!({
                                "chunk_id": ev.chunk_id,
                                "chunk_index": ev.chunk_index,
                                "doc_path": ev.doc_path,
                                "doc_rel_path": ev.doc_rel_path,
                                "score": ev.score,
                                "semantic": ev.semantic,
                                "lexical": ev.lexical,
                                "graph": ev.graph,
                                "relation": ev.relation,
                                "quality": ev.quality,
                                "excerpt": ev.excerpt,
                            })
                        })
                        .collect();
                    serde_json::json!({
                        "path": item.path,
                        "score": item.score,
                        "semantic": item.semantic,
                        "lexical": item.lexical,
                        "frecency": item.frecency,
                        "graph": item.graph,
                        "evidence": evidence,
                    })
                })
                .collect();
            return (
                200,
                serde_json::json!({"query": q, "view": "projects", "results": results}),
            );
        }
        ("GET", "/chunks/search") => {
            let query = req.query.get("q").cloned().unwrap_or_default();
            let q = query.trim().to_string();
            if q.is_empty() {
                return (
                    400,
                    serde_json::json!({"error": "Missing query parameter 'q'."}),
                );
            }
            let limit = parse_limit(req.query.get("limit"), 30, 300);
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
            if let Err(e) = ensure_native_embed_backend(&cfg, "api chunks/search") {
                return (503, serde_json::json!({"error": e}));
            }
            if let Err(e) = ensure_retrieval_backend_ready(&cfg, true, "api chunks/search") {
                return (503, serde_json::json!({"error": e}));
            }
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let rows = match rank_chunks_native(&conn, &cfg, &q, limit) {
                Ok(v) => v,
                Err(e) => return (503, serde_json::json!({"error": e})),
            };
            let results: Vec<Value> = rows.iter().map(ranked_chunk_result_json).collect();
            return (
                200,
                serde_json::json!({
                    "schema": chunk_search_schema(),
                    "query": q,
                    "count": results.len(),
                    "results": results
                }),
            );
        }
        ("GET", "/chunks/related") => {
            let chunk_id = req
                .query
                .get("chunk_id")
                .and_then(|v| v.parse::<i64>().ok())
                .unwrap_or(0);
            if chunk_id <= 0 {
                return (
                    400,
                    serde_json::json!({"error": "Missing or invalid query parameter 'chunk_id'."}),
                );
            }
            let limit = parse_limit(req.query.get("limit"), 20, 200);
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
            if let Err(e) = ensure_native_embed_backend(&cfg, "api chunks/related") {
                return (503, serde_json::json!({"error": e}));
            }
            if let Err(e) = ensure_retrieval_backend_ready(&cfg, true, "api chunks/related") {
                return (503, serde_json::json!({"error": e}));
            }
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let (source, rows) = match related_chunks_native(&conn, &cfg, chunk_id, limit) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let results: Vec<Value> = rows.iter().map(related_chunk_result_json).collect();
            return (
                200,
                serde_json::json!({
                    "schema": chunk_related_schema(),
                    "source": source_chunk_json(&source),
                    "count": results.len(),
                    "results": results
                }),
            );
        }
        ("GET", "/chunks/get") => {
            let chunk_id = req
                .query
                .get("chunk_id")
                .and_then(|v| v.parse::<i64>().ok())
                .unwrap_or(0);
            if chunk_id <= 0 {
                return (
                    400,
                    serde_json::json!({"error": "Missing or invalid query parameter 'chunk_id'."}),
                );
            }
            let max_chars = parse_limit(req.query.get("max_chars"), 8000, 500_000);
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let chunk = match indexed_chunk_by_id(&conn, chunk_id) {
                Ok(Some(v)) => v,
                Ok(None) => {
                    return (
                        404,
                        serde_json::json!({"error": format!("chunk {} not found", chunk_id)}),
                    );
                }
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let (text, truncated, text_chars) = truncate_text_chars(&chunk.text, max_chars);
            let returned_chars = text.chars().count();
            return (
                200,
                serde_json::json!({
                    "schema": chunk_get_schema(),
                    "chunk": {
                        "chunk_id": chunk.chunk_id,
                        "chunk_index": chunk.chunk_index,
                        "path": chunk.doc_path,
                        "project_path": chunk.project_path,
                        "doc_rel_path": chunk.doc_rel_path,
                        "doc_mtime": chunk.doc_mtime,
                        "token_count": chunk.token_count,
                        "text_chars": text_chars,
                        "returned_chars": returned_chars,
                        "truncated": truncated,
                        "text": text
                    }
                }),
            );
        }
        ("GET", "/docs/read") => {
            let path = req.query.get("path").cloned().unwrap_or_default();
            let raw_path = path.trim().to_string();
            if raw_path.is_empty() {
                return (
                    400,
                    serde_json::json!({"error": "Missing query parameter 'path'."}),
                );
            }
            let max_chars = parse_limit(req.query.get("max_chars"), 120_000, 2_000_000);
            let normalized_path = normalize_path(&raw_path).to_string_lossy().to_string();
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let mut chunks = match indexed_doc_chunks_by_path(&conn, &normalized_path) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            if chunks.is_empty() && raw_path != normalized_path {
                chunks = match indexed_doc_chunks_by_path(&conn, &raw_path) {
                    Ok(v) => v,
                    Err(e) => return (500, serde_json::json!({"error": e})),
                };
            }
            if chunks.is_empty() {
                return (
                    404,
                    serde_json::json!({"error": "document path is not indexed"}),
                );
            }
            let first = &chunks[0];
            let mut full_text = String::new();
            let mut token_total: i64 = 0;
            let mut chunk_refs: Vec<Value> = Vec::with_capacity(chunks.len());
            for chunk in &chunks {
                full_text.push_str(&chunk.text);
                token_total += chunk.token_count;
                chunk_refs.push(serde_json::json!({
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count
                }));
            }
            let (text, truncated, text_chars) = truncate_text_chars(&full_text, max_chars);
            let returned_chars = text.chars().count();
            return (
                200,
                serde_json::json!({
                    "schema": doc_read_schema(),
                    "path": first.doc_path.clone(),
                    "project_path": first.project_path.clone(),
                    "doc_rel_path": first.doc_rel_path.clone(),
                    "doc_mtime": first.doc_mtime,
                    "chunk_count": chunks.len(),
                    "token_count": token_total,
                    "text_chars": text_chars,
                    "returned_chars": returned_chars,
                    "truncated": truncated,
                    "chunks": chunk_refs,
                    "text": text
                }),
            );
        }
        ("GET", "/chunks/feedback") => {
            let chunk_id = req
                .query
                .get("chunk_id")
                .and_then(|v| v.parse::<i64>().ok())
                .unwrap_or(0);
            if chunk_id <= 0 {
                return (
                    400,
                    serde_json::json!({"error": "Missing or invalid query parameter 'chunk_id'."}),
                );
            }
            let decision = req
                .query
                .get("decision")
                .map(|v| v.trim().to_lowercase())
                .filter(|v| !v.is_empty());
            if let Some(ref d) = decision {
                if d != "suppressed" && d != "active" {
                    return (
                        400,
                        serde_json::json!({"error": "Invalid decision. Use 'suppressed' or 'active'."}),
                    );
                }
            }
            let quality = req
                .query
                .get("quality")
                .map(|v| v.trim().to_lowercase())
                .filter(|v| !v.is_empty());
            if let Some(ref qv) = quality {
                if normalize_relation_quality_label(qv).is_none() {
                    return (
                        400,
                        serde_json::json!({"error": "Invalid quality. Use 'good', 'weak', 'wrong', or 'unspecified'."}),
                    );
                }
            }
            let limit = parse_limit(req.query.get("limit"), 100, 2000);
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let rows = match list_chunk_relation_feedback(
                &conn,
                chunk_id,
                decision.as_deref(),
                quality.as_deref(),
                limit,
            ) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let results: Vec<Value> = rows.iter().map(relation_feedback_row_json).collect();
            return (
                200,
                serde_json::json!({
                    "source_chunk_id": chunk_id,
                    "decision": decision.unwrap_or_else(|| "all".to_string()),
                    "quality": quality.unwrap_or_else(|| "all".to_string()),
                    "count": results.len(),
                    "results": results
                }),
            );
        }
        ("POST", "/chunks/feedback/suppress") => {
            let body = parse_json_body(&req.body);
            let source_chunk_id = payload_i64(&body, "source_chunk_id");
            let target_chunk_id = payload_i64(&body, "target_chunk_id");
            let relation = payload_string(&body, "relation");
            let note = payload_string(&body, "note");
            if source_chunk_id <= 0 || target_chunk_id <= 0 || relation.is_empty() {
                return (
                    400,
                    serde_json::json!({"error": "source_chunk_id, target_chunk_id, and relation are required."}),
                );
            }
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            if let Err(e) = apply_chunk_relation_decision(
                &conn,
                source_chunk_id,
                target_chunk_id,
                &relation,
                "suppressed",
                &note,
                "api",
                now_ts(),
            ) {
                return (500, serde_json::json!({"error": e}));
            }
            return (
                200,
                serde_json::json!({
                    "ok": true,
                    "source_chunk_id": source_chunk_id,
                    "target_chunk_id": target_chunk_id,
                    "relation": relation,
                    "decision": "suppressed",
                    "note": note
                }),
            );
        }
        ("POST", "/chunks/feedback/restore") => {
            let body = parse_json_body(&req.body);
            let source_chunk_id = payload_i64(&body, "source_chunk_id");
            let target_chunk_id = payload_i64(&body, "target_chunk_id");
            let relation = payload_string(&body, "relation");
            let note = payload_string(&body, "note");
            if source_chunk_id <= 0 || target_chunk_id <= 0 || relation.is_empty() {
                return (
                    400,
                    serde_json::json!({"error": "source_chunk_id, target_chunk_id, and relation are required."}),
                );
            }
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            if let Err(e) = apply_chunk_relation_decision(
                &conn,
                source_chunk_id,
                target_chunk_id,
                &relation,
                "active",
                &note,
                "api",
                now_ts(),
            ) {
                return (500, serde_json::json!({"error": e}));
            }
            return (
                200,
                serde_json::json!({
                    "ok": true,
                    "source_chunk_id": source_chunk_id,
                    "target_chunk_id": target_chunk_id,
                    "relation": relation,
                    "decision": "active",
                    "note": note
                }),
            );
        }
        ("POST", "/chunks/feedback/quality") => {
            let body = parse_json_body(&req.body);
            let source_chunk_id = payload_i64(&body, "source_chunk_id");
            let target_chunk_id = payload_i64(&body, "target_chunk_id");
            let relation = payload_string(&body, "relation");
            let quality_label = {
                let q = payload_string(&body, "quality_label");
                if q.is_empty() {
                    payload_string(&body, "quality")
                } else {
                    q
                }
            };
            let note = payload_string(&body, "note");
            if source_chunk_id <= 0 || target_chunk_id <= 0 || relation.is_empty() {
                return (
                    400,
                    serde_json::json!({"error": "source_chunk_id, target_chunk_id, and relation are required."}),
                );
            }
            let normalized = match normalize_relation_quality_label(&quality_label) {
                Some(v) => v.to_string(),
                None => {
                    return (
                        400,
                        serde_json::json!({"error": "quality_label must be one of: good, weak, wrong, unspecified"}),
                    );
                }
            };
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let persisted = match set_chunk_relation_quality(
                &conn,
                source_chunk_id,
                target_chunk_id,
                &relation,
                &normalized,
                &note,
                "api",
                now_ts(),
            ) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            return (
                200,
                serde_json::json!({
                    "ok": true,
                    "source_chunk_id": source_chunk_id,
                    "target_chunk_id": target_chunk_id,
                    "relation": relation,
                    "quality_label": persisted,
                    "note": note
                }),
            );
        }
        ("GET", "/graph/neighbors") => {
            let path = req.query.get("path").cloned().unwrap_or_default();
            let path = path.trim().to_string();
            if path.is_empty() {
                return (
                    400,
                    serde_json::json!({"error": "Missing query parameter 'path'."}),
                );
            }
            let limit = parse_limit(req.query.get("limit"), 20, 500);
            let target = normalize_path(&path).to_string_lossy().to_string();
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let rows = match list_neighbors_by_path(&conn, &target, limit) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let neighbors: Vec<Value> = rows
                .into_iter()
                .map(|(dst, kind, weight)| {
                    serde_json::json!({
                        "dst": dst,
                        "kind": kind,
                        "weight": weight,
                    })
                })
                .collect();
            return (
                200,
                serde_json::json!({"path": target, "neighbors": neighbors}),
            );
        }
        ("GET", "/graph/view/data") => {
            let focus = req
                .query
                .get("focus")
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty());
            let limit = parse_limit(req.query.get("limit"), 120, 600);
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let payload = match graph_view_data_json(&conn, focus.as_deref(), limit) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            return (200, payload);
        }
        ("GET", "/graph/view/chunks") => {
            let path = req.query.get("path").cloned().unwrap_or_default();
            let target = path.trim().to_string();
            if target.is_empty() {
                return (
                    400,
                    serde_json::json!({"error": "Missing query parameter 'path'."}),
                );
            }
            let limit = parse_limit(req.query.get("limit"), 40, 400);
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let payload = match project_chunks_preview_json(&conn, &target, limit) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            return (200, payload);
        }
        ("GET", "/graph/view/related") => {
            let chunk_id = req
                .query
                .get("chunk_id")
                .and_then(|v| v.parse::<i64>().ok())
                .unwrap_or(0);
            if chunk_id <= 0 {
                return (
                    400,
                    serde_json::json!({"error": "Missing or invalid query parameter 'chunk_id'."}),
                );
            }
            let limit = parse_limit(req.query.get("limit"), 20, 200);
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
            if let Err(e) = ensure_native_embed_backend(&cfg, "api graph/view/related") {
                return (503, serde_json::json!({"error": e}));
            }
            if let Err(e) = ensure_retrieval_backend_ready(&cfg, true, "api graph/view/related") {
                return (503, serde_json::json!({"error": e}));
            }
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let (source, rows) = match related_chunks_native(&conn, &cfg, chunk_id, limit) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let results: Vec<Value> = rows.iter().map(related_chunk_result_json).collect();
            return (
                200,
                serde_json::json!({
                    "schema": chunk_related_schema(),
                    "source": source_chunk_json(&source),
                    "count": results.len(),
                    "results": results
                }),
            );
        }
        ("POST", "/refresh") => {
            let body = parse_json_body(&req.body);
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
            if let Err(e) = ensure_native_embed_backend(&cfg, "api refresh") {
                return (503, serde_json::json!({"error": e}));
            }
            if let Err(e) = ensure_retrieval_backend_ready(&cfg, true, "api refresh") {
                return (503, serde_json::json!({"error": e}));
            }
            let raw_paths = payload_paths(&body);
            let result = if raw_paths.is_empty() {
                run_native_index(
                    &cwd,
                    &cfg,
                    None,
                    true,
                    HashSet::new(),
                    true,
                    false,
                    "api refresh",
                )
            } else {
                let roots = unique_valid_dirs(&raw_paths);
                if roots.is_empty() {
                    return (
                        400,
                        serde_json::json!({"error": "No valid directory paths provided."}),
                    );
                }
                let force_paths: HashSet<PathBuf> = roots.iter().cloned().collect();
                run_native_index(
                    &cwd,
                    &cfg,
                    Some(roots),
                    true,
                    force_paths,
                    false,
                    false,
                    "api refresh",
                )
            };
            let stats = match result {
                Ok(v) => v,
                Err(e) => return (503, serde_json::json!({"error": e})),
            };
            return (
                200,
                serde_json::json!({"stats": stats_payload_json(&stats)}),
            );
        }
        ("POST", "/select") => {
            let body = parse_json_body(&req.body);
            let path = body
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            if path.is_empty() {
                return (
                    400,
                    serde_json::json!({"error": "Missing path in request body."}),
                );
            }
            let query = body
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            let target = normalize_path(&path).to_string_lossy().to_string();
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            if let Err(e) = record_selection_event(&conn, &query, &target, now_ts()) {
                return (500, serde_json::json!({"error": e}));
            }
            return (200, serde_json::json!({"ok": true}));
        }
        ("POST", "/tracked/add") => {
            let body = parse_json_body(&req.body);
            let paths = payload_paths(&body);
            let roots = unique_valid_dirs(&paths);
            if roots.is_empty() {
                return (
                    400,
                    serde_json::json!({"error": "No valid directory paths provided."}),
                );
            }
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let dbp = db_path(&cwd);
            if let Err(e) = ensure_db_schema(&dbp) {
                return (500, serde_json::json!({"error": e}));
            }
            for root in roots {
                if let Err(e) = ensure_tracked_root(&dbp, &root, now_ts()) {
                    return (500, serde_json::json!({"error": e}));
                }
            }
            let rows = match list_tracked_roots(&dbp) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let values: Vec<String> = rows
                .into_iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect();
            return (200, serde_json::json!({"tracked_roots": values}));
        }
        ("POST", "/tracked/del") => {
            let body = parse_json_body(&req.body);
            let paths = payload_paths(&body);
            if paths.is_empty() {
                return (
                    400,
                    serde_json::json!({"error": "Missing path(s) in request body."}),
                );
            }
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let dbp = db_path(&cwd);
            if let Err(e) = ensure_db_schema(&dbp) {
                return (500, serde_json::json!({"error": e}));
            }
            let mut removed: i64 = 0;
            for raw in paths {
                removed += remove_tracked_root(&dbp, &normalize_path(&raw)).unwrap_or(0);
            }
            let rows = match list_tracked_roots(&dbp) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let values: Vec<String> = rows
                .into_iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect();
            return (
                200,
                serde_json::json!({"removed": removed, "tracked_roots": values}),
            );
        }
        _ => {}
    }
    (404, serde_json::json!({"error": "Not found."}))
}

fn run_mcp_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: cypress mcp [serve|doctor]");
        return;
    }
    let action = args
        .first()
        .map(|v| v.to_string_lossy().to_string())
        .unwrap_or_else(|| "serve".to_string())
        .trim()
        .to_lowercase();
    match action.as_str() {
        "serve" => serve_mcp_native().unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            process::exit(1);
        }),
        "doctor" => run_mcp_doctor(),
        other => {
            eprintln!("error: unknown mcp action '{}'", other);
            process::exit(2);
        }
    }
}

fn resolve_cypress_command_path_native() -> Option<PathBuf> {
    if let Ok(path) = env::current_exe() {
        if is_executable_file(&path) {
            return Some(path);
        }
    }
    let cwd = env::current_dir().ok()?;
    let repo_candidate = cwd.join("cypress");
    if is_executable_file(&repo_candidate) {
        return Some(repo_candidate);
    }
    if let Some(path) = resolve_command_path("cypress") {
        return Some(path);
    }
    if let Ok(raw) = env::var("CYPRESS_APP_VENV").or_else(|_| env::var("SEMCD_APP_VENV")) {
        let v = raw.trim();
        if !v.is_empty() {
            let path = expand_tilde(v).join("bin").join("cypress");
            if is_executable_file(&path) {
                return Some(path);
            }
        }
    }
    let data_candidate = data_dir(&cwd).join("venv").join("bin").join("cypress");
    if is_executable_file(&data_candidate) {
        return Some(data_candidate);
    }
    if let Some(path) = resolve_command_path("semcd") {
        return Some(path);
    }
    None
}

fn run_mcp_doctor() {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    let dbfile = db_path(&cwd);
    let root_ok = cfg.root.exists() && cfg.root.is_dir();
    let cfg_ok = cfg_path.exists();
    let db_ok = dbfile.exists();
    let mcp_ok = true;

    let cmd_path = resolve_cypress_command_path_native();
    let cmd_ok = cmd_path.is_some();
    let cmd_display = cmd_path
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "-".to_string());

    let ready = cfg_ok && root_ok && db_ok && cmd_ok;
    println!(
        "mcp doctor: ready={} mcp_sdk={} command={} config={} db={} root={}",
        yes_no(ready),
        yes_no(mcp_ok),
        cmd_display,
        yes_no(cfg_ok),
        yes_no(db_ok),
        yes_no(root_ok)
    );
    if !ready {
        if !cmd_ok {
            println!("hint: ensure a stable cypress command path is available");
        }
        if !cfg_ok || !db_ok {
            println!("hint: run `cypress init` (and optionally `cypress index`) in this workspace");
        }
    }
}

fn mcp_success_result(data: Value) -> Value {
    let text = serde_json::to_string_pretty(&data).unwrap_or_else(|_| "{}".to_string());
    serde_json::json!({
        "content": [{"type": "text", "text": text}],
        "structuredContent": data,
        "isError": false
    })
}

fn mcp_error_response(id: Value, code: i64, message: &str) -> Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": code, "message": message}
    })
}

fn mcp_response(id: Value, result: Value) -> Value {
    serde_json::json!({"jsonrpc": "2.0", "id": id, "result": result})
}

fn mcp_tool_specs() -> Vec<Value> {
    vec![
        serde_json::json!({
            "name": "search_projects",
            "description": "Semantic search across tracked projects with ranked evidence docs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 12}
                },
                "required": ["query"]
            }
        }),
        serde_json::json!({
            "name": "search_files",
            "description": "Semantic search across indexed files (with project context).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 20}
                },
                "required": ["query"]
            }
        }),
        serde_json::json!({
            "name": "search_chunks",
            "description": "Semantic+keyword search across indexed chunks/segments.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 30}
                },
                "required": ["query"]
            }
        }),
        serde_json::json!({
            "name": "get_related_chunks",
            "description": "Retrieve chunks related to a source chunk using graph lineage and semantic similarity.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "integer"},
                    "limit": {"type": "integer", "default": 20}
                },
                "required": ["chunk_id"]
            }
        }),
        serde_json::json!({
            "name": "list_relation_feedback",
            "description": "List relation curation feedback for a source chunk.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "integer"},
                    "decision": {"type": "string", "description": "Optional: suppressed|active"},
                    "quality": {"type": "string", "description": "Optional: good|weak|wrong|unspecified"},
                    "limit": {"type": "integer", "default": 120}
                },
                "required": ["chunk_id"]
            }
        }),
        serde_json::json!({
            "name": "suppress_relation",
            "description": "Suppress a relation from source chunk to target chunk.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_chunk_id": {"type": "integer"},
                    "target_chunk_id": {"type": "integer"},
                    "relation": {"type": "string"},
                    "note": {"type": "string"}
                },
                "required": ["source_chunk_id", "target_chunk_id", "relation"]
            }
        }),
        serde_json::json!({
            "name": "restore_relation",
            "description": "Restore a previously suppressed relation from source chunk to target chunk.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_chunk_id": {"type": "integer"},
                    "target_chunk_id": {"type": "integer"},
                    "relation": {"type": "string"},
                    "note": {"type": "string"}
                },
                "required": ["source_chunk_id", "target_chunk_id", "relation"]
            }
        }),
        serde_json::json!({
            "name": "set_relation_quality",
            "description": "Set relation quality label for a source->target relation (good/weak/wrong/unspecified).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_chunk_id": {"type": "integer"},
                    "target_chunk_id": {"type": "integer"},
                    "relation": {"type": "string"},
                    "quality_label": {"type": "string"},
                    "note": {"type": "string"}
                },
                "required": ["source_chunk_id", "target_chunk_id", "relation", "quality_label"]
            }
        }),
        serde_json::json!({
            "name": "get_project_neighbors",
            "description": "Return relationship graph neighbors for a project path.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer", "default": 20}
                },
                "required": ["path"]
            }
        }),
        serde_json::json!({
            "name": "list_tracked_roots",
            "description": "List tracked root directories that feed indexing/search.",
            "inputSchema": {"type": "object", "properties": {}}
        }),
        serde_json::json!({
            "name": "add_tracked_root",
            "description": "Track a new root directory and optionally index it immediately.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "refresh": {"type": "boolean", "default": true}
                },
                "required": ["path"]
            }
        }),
        serde_json::json!({
            "name": "remove_tracked_root",
            "description": "Stop tracking a root directory and optionally refresh index.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "refresh": {"type": "boolean", "default": true}
                },
                "required": ["path"]
            }
        }),
        serde_json::json!({
            "name": "run_incremental_index",
            "description": "Run incremental index across all tracked roots.",
            "inputSchema": {"type": "object", "properties": {}}
        }),
        serde_json::json!({
            "name": "run_forced_refresh",
            "description": "Force-refresh all tracked roots or a supplied subset.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        }),
    ]
}

fn mcp_status_resource() -> Result<String, String> {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
    let dbp = db_path(&cwd);
    let conn = open_db_rw(&dbp)?;
    let roots = list_tracked_roots_conn(&conn)?;
    let projects: i64 = conn
        .query_row("SELECT COUNT(*) FROM projects", [], |row| row.get(0))
        .unwrap_or(0);
    Ok(format!(
        "root={}\nembed_backend={}\nembed_model={}\nretrieval_backend={}\ntracked_roots={}\nprojects={}\ndb={}\n",
        cfg.root.to_string_lossy(),
        cfg.embed_backend,
        cfg.embed_model,
        cfg.retrieval_backend,
        roots.len(),
        projects,
        dbp.display()
    ))
}

fn mcp_tool_call(name: &str, args: &Value) -> Result<Value, String> {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
    let dbp = db_path(&cwd);
    ensure_db_schema(&dbp)?;
    let conn = open_db_rw(&dbp)?;

    match name {
        "search_projects" => {
            let query = args
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            if query.is_empty() {
                return Err("query must be non-empty".to_string());
            }
            let limit = args
                .get("limit")
                .and_then(|v| v.as_i64())
                .unwrap_or(12)
                .clamp(1, 50) as usize;
            ensure_native_embed_backend(&cfg, "mcp search_projects")?;
            ensure_retrieval_backend_ready(&cfg, true, "mcp search_projects")?;
            let rows = rank_projects_native(&conn, &cfg, &query, limit)?;
            let results: Vec<Value> = rows
                .into_iter()
                .map(|item| {
                    let evidence: Vec<Value> = item
                        .evidence
                        .into_iter()
                        .map(|ev| {
                            serde_json::json!({
                                "chunk_id": ev.chunk_id,
                                "chunk_index": ev.chunk_index,
                                "doc_path": ev.doc_path,
                                "doc_rel_path": ev.doc_rel_path,
                                "score": ev.score,
                                "semantic": ev.semantic,
                                "lexical": ev.lexical,
                                "graph": ev.graph,
                                "relation": ev.relation,
                                "quality": ev.quality,
                                "excerpt": ev.excerpt,
                            })
                        })
                        .collect();
                    serde_json::json!({
                        "path": item.path,
                        "score": item.score,
                        "semantic": item.semantic,
                        "lexical": item.lexical,
                        "frecency": item.frecency,
                        "graph": item.graph,
                        "evidence": evidence,
                    })
                })
                .collect();
            Ok(serde_json::json!({"query": query, "count": results.len(), "results": results}))
        }
        "search_files" => {
            let query = args
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            if query.is_empty() {
                return Err("query must be non-empty".to_string());
            }
            let limit = args
                .get("limit")
                .and_then(|v| v.as_i64())
                .unwrap_or(20)
                .clamp(1, 100) as usize;
            ensure_native_embed_backend(&cfg, "mcp search_files")?;
            ensure_retrieval_backend_ready(&cfg, true, "mcp search_files")?;
            let rows = rank_files_native(&conn, &cfg, &query, limit)?;
            let results: Vec<Value> = rows
                .into_iter()
                .map(|item| {
                    let evidence: Vec<Value> = item
                        .evidence
                        .into_iter()
                        .map(|ev| {
                            serde_json::json!({
                                "chunk_id": ev.chunk_id,
                                "chunk_index": ev.chunk_index,
                                "doc_path": ev.doc_path,
                                "doc_rel_path": ev.doc_rel_path,
                                "score": ev.score,
                                "semantic": ev.semantic,
                                "lexical": ev.lexical,
                                "graph": ev.graph,
                                "relation": ev.relation,
                                "quality": ev.quality,
                                "excerpt": ev.excerpt,
                            })
                        })
                        .collect();
                    serde_json::json!({
                        "path": item.path,
                        "project_path": item.project_path,
                        "doc_rel_path": item.doc_rel_path,
                        "chunk_id": item.chunk_id,
                        "chunk_index": item.chunk_index,
                        "score": item.score,
                        "semantic": item.semantic,
                        "lexical": item.lexical,
                        "graph": item.graph,
                        "relation": item.relation,
                        "quality": item.quality,
                        "excerpt": item.excerpt,
                        "evidence": evidence,
                    })
                })
                .collect();
            Ok(serde_json::json!({"query": query, "count": results.len(), "results": results}))
        }
        "search_chunks" => {
            let query = args
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            if query.is_empty() {
                return Err("query must be non-empty".to_string());
            }
            let limit = args
                .get("limit")
                .and_then(|v| v.as_i64())
                .unwrap_or(30)
                .clamp(1, 200) as usize;
            ensure_native_embed_backend(&cfg, "mcp search_chunks")?;
            ensure_retrieval_backend_ready(&cfg, true, "mcp search_chunks")?;
            let rows = rank_chunks_native(&conn, &cfg, &query, limit)?;
            let results: Vec<Value> = rows.iter().map(ranked_chunk_result_json).collect();
            Ok(serde_json::json!({
                "schema": chunk_search_schema(),
                "query": query,
                "count": results.len(),
                "results": results
            }))
        }
        "get_related_chunks" => {
            let chunk_id = args.get("chunk_id").and_then(|v| v.as_i64()).unwrap_or(0);
            if chunk_id <= 0 {
                return Err("chunk_id must be a positive integer".to_string());
            }
            let limit = args
                .get("limit")
                .and_then(|v| v.as_i64())
                .unwrap_or(20)
                .clamp(1, 200) as usize;
            ensure_native_embed_backend(&cfg, "mcp get_related_chunks")?;
            ensure_retrieval_backend_ready(&cfg, true, "mcp get_related_chunks")?;
            let (source, rows) = related_chunks_native(&conn, &cfg, chunk_id, limit)?;
            let results: Vec<Value> = rows.iter().map(related_chunk_result_json).collect();
            Ok(serde_json::json!({
                "schema": chunk_related_schema(),
                "source": source_chunk_json(&source),
                "count": results.len(),
                "results": results
            }))
        }
        "list_relation_feedback" => {
            let chunk_id = args.get("chunk_id").and_then(|v| v.as_i64()).unwrap_or(0);
            if chunk_id <= 0 {
                return Err("chunk_id must be a positive integer".to_string());
            }
            let decision = args
                .get("decision")
                .and_then(|v| v.as_str())
                .map(|v| v.trim().to_lowercase())
                .filter(|v| !v.is_empty());
            if let Some(ref d) = decision {
                if d != "suppressed" && d != "active" {
                    return Err("decision must be 'suppressed' or 'active'".to_string());
                }
            }
            let quality = args
                .get("quality")
                .and_then(|v| v.as_str())
                .map(|v| v.trim().to_lowercase())
                .filter(|v| !v.is_empty());
            if let Some(ref qv) = quality {
                if normalize_relation_quality_label(qv).is_none() {
                    return Err(
                        "quality must be one of: good, weak, wrong, unspecified".to_string()
                    );
                }
            }
            let limit = args
                .get("limit")
                .and_then(|v| v.as_i64())
                .unwrap_or(120)
                .clamp(1, 2000) as usize;
            let rows = list_chunk_relation_feedback(
                &conn,
                chunk_id,
                decision.as_deref(),
                quality.as_deref(),
                limit,
            )?;
            let results: Vec<Value> = rows.iter().map(relation_feedback_row_json).collect();
            Ok(serde_json::json!({
                "source_chunk_id": chunk_id,
                "decision": decision.unwrap_or_else(|| "all".to_string()),
                "quality": quality.unwrap_or_else(|| "all".to_string()),
                "count": results.len(),
                "results": results
            }))
        }
        "suppress_relation" => {
            let source_chunk_id = args
                .get("source_chunk_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let target_chunk_id = args
                .get("target_chunk_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let relation = args
                .get("relation")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            if source_chunk_id <= 0 || target_chunk_id <= 0 || relation.is_empty() {
                return Err(
                    "source_chunk_id, target_chunk_id, and relation are required".to_string(),
                );
            }
            let note = args
                .get("note")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            apply_chunk_relation_decision(
                &conn,
                source_chunk_id,
                target_chunk_id,
                &relation,
                "suppressed",
                &note,
                "mcp",
                now_ts(),
            )?;
            Ok(serde_json::json!({
                "ok": true,
                "source_chunk_id": source_chunk_id,
                "target_chunk_id": target_chunk_id,
                "relation": relation,
                "decision": "suppressed",
                "note": note
            }))
        }
        "restore_relation" => {
            let source_chunk_id = args
                .get("source_chunk_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let target_chunk_id = args
                .get("target_chunk_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let relation = args
                .get("relation")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            if source_chunk_id <= 0 || target_chunk_id <= 0 || relation.is_empty() {
                return Err(
                    "source_chunk_id, target_chunk_id, and relation are required".to_string(),
                );
            }
            let note = args
                .get("note")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            apply_chunk_relation_decision(
                &conn,
                source_chunk_id,
                target_chunk_id,
                &relation,
                "active",
                &note,
                "mcp",
                now_ts(),
            )?;
            Ok(serde_json::json!({
                "ok": true,
                "source_chunk_id": source_chunk_id,
                "target_chunk_id": target_chunk_id,
                "relation": relation,
                "decision": "active",
                "note": note
            }))
        }
        "set_relation_quality" => {
            let source_chunk_id = args
                .get("source_chunk_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let target_chunk_id = args
                .get("target_chunk_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let relation = args
                .get("relation")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            let quality_label = args
                .get("quality_label")
                .and_then(|v| v.as_str())
                .or_else(|| args.get("quality").and_then(|v| v.as_str()))
                .unwrap_or_default()
                .trim()
                .to_string();
            if source_chunk_id <= 0 || target_chunk_id <= 0 || relation.is_empty() {
                return Err(
                    "source_chunk_id, target_chunk_id, and relation are required".to_string(),
                );
            }
            let normalized = normalize_relation_quality_label(&quality_label)
                .ok_or_else(|| {
                    "quality_label must be one of: good, weak, wrong, unspecified".to_string()
                })?
                .to_string();
            let note = args
                .get("note")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            let persisted = set_chunk_relation_quality(
                &conn,
                source_chunk_id,
                target_chunk_id,
                &relation,
                &normalized,
                &note,
                "mcp",
                now_ts(),
            )?;
            Ok(serde_json::json!({
                "ok": true,
                "source_chunk_id": source_chunk_id,
                "target_chunk_id": target_chunk_id,
                "relation": relation,
                "quality_label": persisted,
                "note": note
            }))
        }
        "get_project_neighbors" => {
            let path = args
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            if path.is_empty() {
                return Err("path must be non-empty".to_string());
            }
            let limit = args
                .get("limit")
                .and_then(|v| v.as_i64())
                .unwrap_or(20)
                .clamp(1, 200) as usize;
            let target = normalize_path(&path).to_string_lossy().to_string();
            let rows = list_neighbors_by_path(&conn, &target, limit)?;
            let neighbors: Vec<Value> = rows
                .into_iter()
                .map(|(dst, kind, weight)| serde_json::json!({"dst": dst, "kind": kind, "weight": weight}))
                .collect();
            Ok(
                serde_json::json!({"path": target, "count": neighbors.len(), "neighbors": neighbors}),
            )
        }
        "list_tracked_roots" => {
            let rows = list_tracked_roots_conn(&conn)?;
            let roots: Vec<String> = rows
                .into_iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect();
            Ok(serde_json::json!({"count": roots.len(), "roots": roots}))
        }
        "add_tracked_root" => {
            let path = args
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            if path.is_empty() {
                return Err("path must be non-empty".to_string());
            }
            let refresh = args
                .get("refresh")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let root = normalize_path(&path);
            if !root.exists() || !root.is_dir() {
                return Err(format!("path is not a directory: {}", root.display()));
            }
            ensure_tracked_root_conn(&conn, &root, now_ts())?;
            let mut out = serde_json::json!({"added": root.to_string_lossy(), "refreshed": false});
            if refresh {
                ensure_native_embed_backend(&cfg, "mcp add_tracked_root refresh")?;
                ensure_retrieval_backend_ready(&cfg, true, "mcp add_tracked_root refresh")?;
                let mut force_paths = HashSet::new();
                force_paths.insert(root.clone());
                let stats = run_native_index(
                    &cwd,
                    &cfg,
                    Some(vec![root.clone()]),
                    true,
                    force_paths,
                    false,
                    false,
                    "mcp add root refresh",
                )?;
                out["refreshed"] = Value::Bool(true);
                out["stats"] = stats_payload_json(&stats);
            }
            Ok(out)
        }
        "remove_tracked_root" => {
            let path = args
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            if path.is_empty() {
                return Err("path must be non-empty".to_string());
            }
            let refresh = args
                .get("refresh")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let root = normalize_path(&path).to_string_lossy().to_string();
            let removed = remove_tracked_root(&dbp, &normalize_path(&root))?;
            let mut out = serde_json::json!({"removed": removed, "path": root, "refreshed": false});
            if refresh {
                ensure_native_embed_backend(&cfg, "mcp remove_tracked_root refresh")?;
                ensure_retrieval_backend_ready(&cfg, true, "mcp remove_tracked_root refresh")?;
                let stats = run_native_index(
                    &cwd,
                    &cfg,
                    None,
                    false,
                    HashSet::new(),
                    true,
                    false,
                    "mcp remove root refresh",
                )?;
                out["refreshed"] = Value::Bool(true);
                out["stats"] = stats_payload_json(&stats);
            }
            Ok(out)
        }
        "run_incremental_index" => {
            ensure_native_embed_backend(&cfg, "mcp incremental index")?;
            ensure_retrieval_backend_ready(&cfg, true, "mcp incremental index")?;
            let stats = run_native_index(
                &cwd,
                &cfg,
                None,
                false,
                HashSet::new(),
                true,
                false,
                "mcp incremental index",
            )?;
            Ok(serde_json::json!({"mode": "incremental", "stats": stats_payload_json(&stats)}))
        }
        "run_forced_refresh" => {
            ensure_native_embed_backend(&cfg, "mcp forced refresh")?;
            ensure_retrieval_backend_ready(&cfg, true, "mcp forced refresh")?;
            let paths: Vec<String> = args
                .get("paths")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if paths.is_empty() {
                let stats = run_native_index(
                    &cwd,
                    &cfg,
                    None,
                    true,
                    HashSet::new(),
                    true,
                    false,
                    "mcp forced refresh",
                )?;
                return Ok(
                    serde_json::json!({"mode": "forced_all", "stats": stats_payload_json(&stats)}),
                );
            }
            let roots = unique_valid_dirs(&paths);
            if roots.is_empty() {
                return Err("no valid directory paths provided".to_string());
            }
            let force_paths: HashSet<PathBuf> = roots.iter().cloned().collect();
            let stats = run_native_index(
                &cwd,
                &cfg,
                Some(roots.clone()),
                true,
                force_paths,
                false,
                false,
                "mcp forced refresh",
            )?;
            let paths_out: Vec<String> = roots
                .into_iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect();
            Ok(
                serde_json::json!({"mode": "forced_scoped", "paths": paths_out, "stats": stats_payload_json(&stats)}),
            )
        }
        _ => Err(format!("unknown tool '{}'", name)),
    }
}

fn read_mcp_frame<R: BufRead + Read>(reader: &mut R) -> Result<Option<Value>, String> {
    let mut headers = HashMap::new();
    let mut saw_header = false;
    loop {
        let mut line = String::new();
        let n = reader
            .read_line(&mut line)
            .map_err(|e| format!("failed reading MCP header: {}", e))?;
        if n == 0 {
            if saw_header {
                return Err("unexpected EOF while reading MCP headers".to_string());
            }
            return Ok(None);
        }
        let line = line.trim_end_matches(['\r', '\n']).to_string();
        if line.is_empty() {
            break;
        }
        saw_header = true;
        if let Some((k, v)) = line.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }
    let len = headers
        .get("content-length")
        .and_then(|v| v.parse::<usize>().ok())
        .ok_or_else(|| "MCP frame missing Content-Length".to_string())?;
    let mut body = vec![0u8; len];
    reader
        .read_exact(&mut body)
        .map_err(|e| format!("failed reading MCP frame body: {}", e))?;
    let value =
        serde_json::from_slice::<Value>(&body).map_err(|e| format!("invalid MCP JSON: {}", e))?;
    Ok(Some(value))
}

fn write_mcp_frame<W: Write>(writer: &mut W, value: &Value) -> Result<(), String> {
    let body =
        serde_json::to_vec(value).map_err(|e| format!("failed serializing MCP JSON: {}", e))?;
    let header = format!("Content-Length: {}\r\n\r\n", body.len());
    writer
        .write_all(header.as_bytes())
        .and_then(|_| writer.write_all(&body))
        .and_then(|_| writer.flush())
        .map_err(|e| format!("failed writing MCP frame: {}", e))
}

fn serve_mcp_native() -> Result<(), String> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = stdout.lock();

    while let Some(msg) = read_mcp_frame(&mut reader)? {
        let id = msg.get("id").cloned();
        let method = msg
            .get("method")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        if method.starts_with("notifications/") {
            continue;
        }
        let Some(id) = id else {
            continue;
        };
        let params = msg.get("params").cloned().unwrap_or_else(|| Value::Null);
        let response = match method.as_str() {
            "initialize" => {
                let protocol = params
                    .get("protocolVersion")
                    .and_then(|v| v.as_str())
                    .unwrap_or("2025-03-26");
                mcp_response(
                    id,
                    serde_json::json!({
                        "protocolVersion": protocol,
                        "capabilities": {
                            "tools": {"listChanged": false},
                            "resources": {"listChanged": false}
                        },
                        "serverInfo": {
                            "name": "cypress",
                            "version": env!("CARGO_PKG_VERSION")
                        },
                        "instructions": "Semantic project memory server for local files. Use search to find relevant projects by meaning and evidence docs."
                    }),
                )
            }
            "ping" => mcp_response(id, serde_json::json!({})),
            "tools/list" => mcp_response(id, serde_json::json!({"tools": mcp_tool_specs()})),
            "tools/call" => {
                let tool_name = params
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                if tool_name.is_empty() {
                    mcp_error_response(id, -32602, "tools/call requires tool name")
                } else {
                    let args = params
                        .get("arguments")
                        .cloned()
                        .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
                    match mcp_tool_call(&tool_name, &args) {
                        Ok(data) => mcp_response(id, mcp_success_result(data)),
                        Err(err) => mcp_error_response(id, -32000, &err),
                    }
                }
            }
            "resources/list" => mcp_response(
                id,
                serde_json::json!({
                    "resources": [{
                        "uri": "cypress://status",
                        "name": "cypress status",
                        "description": "Quick status snapshot for agent context hydration.",
                        "mimeType": "text/plain"
                    }]
                }),
            ),
            "resources/read" => {
                let uri = params
                    .get("uri")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                if uri != "cypress://status" {
                    mcp_error_response(id, -32602, "unknown resource uri")
                } else {
                    match mcp_status_resource() {
                        Ok(text) => mcp_response(
                            id,
                            serde_json::json!({
                                "contents": [{
                                    "uri": "cypress://status",
                                    "mimeType": "text/plain",
                                    "text": text
                                }]
                            }),
                        ),
                        Err(err) => mcp_error_response(id, -32000, &err),
                    }
                }
            }
            "prompts/list" => mcp_response(id, serde_json::json!({"prompts": []})),
            _ => mcp_error_response(id, -32601, "method not found"),
        };
        write_mcp_frame(&mut writer, &response)?;
    }
    Ok(())
}

fn run_self_test_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: cypress self-test [--query <text>] [--lifecycle] [--timeout <seconds>]");
        println!("notes:");
        println!("  --lifecycle runs daemon and managed-local FalkorDB lifecycle probes on temporary ports");
        return;
    }
    let mut query = "storage".to_string();
    let mut lifecycle = false;
    let mut timeout_s = 8u64;
    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        match s.as_str() {
            "--query" => {
                i += 1;
                query = arg_value(args, i, "--query");
            }
            "--lifecycle" => {
                lifecycle = true;
            }
            "--timeout" => {
                i += 1;
                let raw = arg_value(args, i, "--timeout");
                timeout_s = raw.parse::<u64>().unwrap_or_else(|_| {
                    eprintln!("error: --timeout must be an integer number of seconds");
                    process::exit(2);
                });
                timeout_s = timeout_s.max(2);
            }
            x if x.starts_with("--timeout=") => {
                let raw = x.trim_start_matches("--timeout=");
                timeout_s = raw.parse::<u64>().unwrap_or_else(|_| {
                    eprintln!("error: --timeout must be an integer number of seconds");
                    process::exit(2);
                });
                timeout_s = timeout_s.max(2);
            }
            other => {
                eprintln!("error: unknown option '{}'", other);
                process::exit(2);
            }
        }
        i += 1;
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
    let dbp = db_path(&cwd);
    let mut ok = true;

    println!("self-test: config={}", config_path(&cwd).display());
    println!("self-test: db={}", dbp.display());
    if !config_path(&cwd).exists() {
        println!("self-test: config file missing");
        ok = false;
    }
    if !dbp.exists() {
        println!("self-test: database missing");
        ok = false;
    }

    if cfg.retrieval_backend == "falkordb" {
        match falkor_store::ping() {
            Ok(_) => println!("self-test: falkordb reachable=yes"),
            Err(err) => {
                println!("self-test: falkordb reachable=no ({})", err);
                ok = false;
            }
        }
    }

    if ensure_native_embed_backend(&cfg, "self-test search").is_ok() {
        if ensure_retrieval_backend_ready(&cfg, false, "self-test search").is_ok() {
            match open_db_rw(&dbp).and_then(|conn| rank_projects_native(&conn, &cfg, &query, 3)) {
                Ok(rows) => println!("self-test: search results={}", rows.len()),
                Err(err) => {
                    println!("self-test: search failed ({})", err);
                    ok = false;
                }
            }
        }
    } else {
        println!("self-test: search skipped (non-native embed backend)");
    }

    let mcp_cmd_ok = resolve_cypress_command_path_native().is_some();
    println!("self-test: mcp command path={}", yes_no(mcp_cmd_ok));
    if !mcp_cmd_ok {
        ok = false;
    }

    println!("self-test: api command available=yes");
    println!("self-test: mcp command available=yes");

    if lifecycle {
        println!("self-test: lifecycle enabled=yes");
        match run_self_test_lifecycle_probe(&cwd, &cfg, &query, timeout_s) {
            Ok(lines) => {
                for line in lines {
                    println!("self-test:lifecycle: {}", line);
                }
            }
            Err(err) => {
                println!("self-test: lifecycle failed ({})", err);
                ok = false;
            }
        }
    }

    if !ok {
        process::exit(1);
    }
}

fn run_self_test_lifecycle_probe(
    cwd: &Path,
    cfg: &ConfigValues,
    query: &str,
    timeout_s: u64,
) -> Result<Vec<String>, String> {
    let mut lines: Vec<String> = Vec::new();
    let (daemon_port, daemon_pid) = run_self_test_daemon_probe(cwd, timeout_s)?;
    lines.push(format!(
        "daemon probe ok (host={} port={} pid={})",
        daemon_default_host(),
        daemon_port,
        daemon_pid
    ));

    if cfg.retrieval_backend == "falkordb" {
        let graph_port = run_self_test_graph_probe(cwd, timeout_s)?;
        lines.push(format!(
            "graph probe ok (runtime=managed-local temp_port={})",
            graph_port
        ));
        ensure_retrieval_backend_ready(cfg, true, "self-test lifecycle retrieval")?;
        let dbp = db_path(cwd);
        let rows = open_db_rw(&dbp).and_then(|conn| rank_projects_native(&conn, cfg, query, 3))?;
        lines.push(format!("post-probe search results={}", rows.len()));
    } else {
        lines.push(format!(
            "graph probe skipped (retrieval backend={})",
            cfg.retrieval_backend
        ));
    }
    Ok(lines)
}

fn run_self_test_daemon_probe(cwd: &Path, timeout_s: u64) -> Result<(u16, u32), String> {
    let host = daemon_default_host();
    let start_port = daemon_default_port().saturating_add(20);
    let port = find_free_port(&host, start_port, 300)
        .ok_or_else(|| "no free port available for daemon probe".to_string())?;
    let pid = spawn_api_daemon(cwd, &host, port)?;
    let deadline = Instant::now() + Duration::from_secs(timeout_s.max(2));
    let mut up = false;
    while Instant::now() < deadline {
        if api_health_host_port(&host, port) {
            up = true;
            break;
        }
        if !pid_is_alive(pid) {
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }
    if !up {
        let _ = run_shell_capture(&format!("kill {} >/dev/null 2>&1 || true", pid));
        return Err(format!(
            "daemon probe failed to become healthy on {}:{} (pid={})",
            host, port, pid
        ));
    }
    let _ = run_shell_capture(&format!("kill {} >/dev/null 2>&1 || true", pid));
    let stop_deadline = Instant::now() + Duration::from_secs(timeout_s.max(2));
    let mut down = false;
    while Instant::now() < stop_deadline {
        if !api_health_host_port(&host, port) {
            down = true;
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }
    if !down {
        let _ = run_shell_capture(&format!("kill -9 {} >/dev/null 2>&1 || true", pid));
        let hard_deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < hard_deadline {
            if !api_health_host_port(&host, port) {
                down = true;
                break;
            }
            thread::sleep(Duration::from_millis(120));
        }
    }
    if !down {
        return Err(format!(
            "daemon probe endpoint {}:{} did not stop cleanly (pid={})",
            host, port, pid
        ));
    }
    Ok((port, pid))
}

fn build_managed_local_probe_cmd(cwd: &Path, base_cmd: &str, port: u16) -> Result<String, String> {
    let program = parse_leading_command_path(base_cmd).ok_or_else(|| {
        format!(
            "unable to parse runtime command program from '{}'",
            base_cmd
        )
    })?;
    let file_name = program
        .file_name()
        .and_then(|v| v.to_str())
        .unwrap_or("")
        .to_lowercase();
    let base_lower = base_cmd.to_lowercase();
    if file_name.contains("redis")
        || file_name.contains("valkey")
        || base_lower.contains("--loadmodule")
    {
        let module_path = extract_loadmodule_path(base_cmd)
            .map(PathBuf::from)
            .or_else(|| find_falkordb_module_path(&data_dir(cwd).join("bin")))
            .ok_or_else(|| {
                "unable to locate Falkor module path for managed-local probe".to_string()
            })?;
        if !module_path.exists() {
            return Err(format!(
                "managed-local probe module does not exist at '{}'",
                module_path.display()
            ));
        }
        return Ok(format!(
            "{} --loadmodule {} --port {} --save '' --appendonly no",
            shell_escape(&program.to_string_lossy()),
            shell_escape(&module_path.to_string_lossy()),
            port
        ));
    }
    Ok(format!(
        "{} --port {}",
        shell_escape(&program.to_string_lossy()),
        port
    ))
}

fn run_self_test_graph_probe(cwd: &Path, timeout_s: u64) -> Result<u16, String> {
    let base_cmd = falkor_managed_local_cmd().ok_or_else(|| {
        "managed-local runtime command is not configured; run `cypress install` or `cypress graph provision`".to_string()
    })?;
    let host = "127.0.0.1";
    let port = find_free_port(host, 6391, 300)
        .ok_or_else(|| "no free port available for graph probe".to_string())?;
    let probe_cmd = build_managed_local_probe_cmd(cwd, &base_cmd, port)?;

    let state_path = graph_runtime_state_path(cwd);
    let state_backup = fs::read(&state_path).ok();
    let prev_url = env::var_os("CYPRESS_FALKOR_URL");
    let prev_graph = env::var_os("CYPRESS_FALKOR_GRAPH");
    let prev_mode = env::var_os("CYPRESS_FALKOR_RUNTIME_MODE");
    let prev_cmd = env::var_os("CYPRESS_FALKOR_RUNTIME_CMD");

    let probe_graph = format!("cypress_selftest_{}", process::id());
    let probe_url = format!("falkor://{}:{}", host, port);

    let mut start_err: Option<String> = None;
    let mut stop_err: Option<String> = None;

    let _ = fs::remove_file(&state_path);
    env::set_var("CYPRESS_FALKOR_URL", &probe_url);
    env::set_var("CYPRESS_FALKOR_GRAPH", &probe_graph);
    env::set_var("CYPRESS_FALKOR_RUNTIME_MODE", "managed-local");
    env::set_var("CYPRESS_FALKOR_RUNTIME_CMD", &probe_cmd);

    if let Err(e) = start_falkordb(timeout_s.max(2)) {
        start_err = Some(format!("graph probe start failed: {}", e));
    } else if let Err(e) = falkor_store::ping() {
        start_err = Some(format!("graph probe ping failed after start: {}", e));
    }

    if start_err.is_none() {
        if let Err(e) = stop_falkordb(timeout_s.max(2)) {
            stop_err = Some(format!("graph probe stop failed: {}", e));
        }
    }

    match prev_url {
        Some(v) => env::set_var("CYPRESS_FALKOR_URL", v),
        None => env::remove_var("CYPRESS_FALKOR_URL"),
    }
    match prev_graph {
        Some(v) => env::set_var("CYPRESS_FALKOR_GRAPH", v),
        None => env::remove_var("CYPRESS_FALKOR_GRAPH"),
    }
    match prev_mode {
        Some(v) => env::set_var("CYPRESS_FALKOR_RUNTIME_MODE", v),
        None => env::remove_var("CYPRESS_FALKOR_RUNTIME_MODE"),
    }
    match prev_cmd {
        Some(v) => env::set_var("CYPRESS_FALKOR_RUNTIME_CMD", v),
        None => env::remove_var("CYPRESS_FALKOR_RUNTIME_CMD"),
    }

    if let Some(bytes) = state_backup {
        if let Some(parent) = state_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(&state_path, bytes);
    } else {
        let _ = fs::remove_file(&state_path);
    }

    if let Some(err) = start_err {
        return Err(err);
    }
    if let Some(err) = stop_err {
        return Err(err);
    }
    Ok(port)
}

fn run_index_with_strategy(
    cwd: &Path,
    cfg: &ConfigValues,
    scope_roots: Option<Vec<PathBuf>>,
    force_all: bool,
    force_paths: Option<HashSet<PathBuf>>,
    remove_missing: bool,
    reason: &str,
) -> Result<(), String> {
    ensure_retrieval_backend_ready(cfg, true, reason)?;
    ensure_native_embed_backend(cfg, reason)?;

    let stats = run_native_index(
        cwd,
        cfg,
        scope_roots,
        force_all,
        force_paths.unwrap_or_default(),
        remove_missing,
        true,
        reason,
    )?;
    print_index_stats(&stats, cfg);
    Ok(())
}

#[derive(Default)]
struct IndexStats {
    total_projects: i64,
    updated_projects: i64,
    skipped_projects: i64,
    removed_projects: i64,
    vectorized_projects: i64,
    vector_failures: i64,
    tracked_roots: i64,
    graph_edges: i64,
    chunk_rows: i64,
    chunk_vectors: i64,
    retrieval_backend: String,
    retrieval_synced_chunks: i64,
    retrieval_error: String,
}

#[derive(Clone)]
struct ProjectDoc {
    path: PathBuf,
    title: String,
    summary: String,
    mtime: f64,
}

#[derive(Clone)]
struct ProjectChunk {
    doc_path: String,
    doc_rel_path: String,
    doc_mtime: f64,
    chunk_index: i64,
    token_count: i64,
    text_hash: String,
    text: String,
}

struct ProjectCorpus {
    doc: ProjectDoc,
    chunks: Vec<ProjectChunk>,
}

struct ExistingProject {
    id: i64,
    path: String,
    title: String,
    summary: String,
    project_mtime: f64,
}

fn run_native_index(
    cwd: &Path,
    cfg: &ConfigValues,
    scope_roots: Option<Vec<PathBuf>>,
    force_all: bool,
    force_paths: HashSet<PathBuf>,
    remove_missing: bool,
    emit_progress: bool,
    reason: &str,
) -> Result<IndexStats, String> {
    let dbp = db_path(cwd);
    let conn = open_db_rw(&dbp)?;
    let roots = resolve_roots(&conn, cfg, scope_roots)?;
    if roots.is_empty() {
        return Err("No tracked roots configured. Add one with `cypress add <path>`.".to_string());
    }
    let projects = discover_projects(&roots);
    let embedder = build_embedder(cfg)?;
    let model_key = embedder.model_key();
    let mode = if force_all {
        "forced refresh"
    } else {
        "incremental index"
    };
    if emit_progress {
        println!(
            "{}: {} started (backend={}, model={})",
            reason, mode, cfg.embed_backend, cfg.embed_model
        );
    }

    let mut stats = IndexStats {
        tracked_roots: roots.len() as i64,
        retrieval_backend: cfg.retrieval_backend.clone(),
        ..Default::default()
    };
    stats.total_projects = projects.len() as i64;

    let mut keep_paths: Vec<String> = Vec::new();
    let mut docs_by_id: HashMap<i64, ProjectDoc> = HashMap::new();
    let now = now_ts();

    for (idx, project_dir) in projects.iter().enumerate() {
        let project_path = project_dir.to_string_lossy().to_string();
        keep_paths.push(project_path.clone());
        let forced = force_all || is_under_any(project_dir, &force_paths);
        let latest_mtime = project_latest_mtime(project_dir)?;
        let existing = get_project_by_path(&conn, &project_path)?;
        if let Some(row) = existing {
            if !forced && row.project_mtime >= latest_mtime {
                let project_vec_ready = has_project_vector(&conn, row.id, &model_key)?;
                let chunk_vec_ready = has_project_chunk_vectors(&conn, row.id, &model_key)?;
                let has_chunks = has_project_chunks(&conn, row.id)?;
                if project_vec_ready && (chunk_vec_ready || !has_chunks) {
                    stats.skipped_projects += 1;
                    docs_by_id.insert(
                        row.id,
                        ProjectDoc {
                            path: PathBuf::from(row.path),
                            title: row.title,
                            summary: row.summary,
                            mtime: row.project_mtime,
                        },
                    );
                    if emit_progress {
                        println!(
                            "[{}/{}] skip {}",
                            idx + 1,
                            projects.len(),
                            project_dir
                                .file_name()
                                .and_then(|s| s.to_str())
                                .unwrap_or("project")
                        );
                    }
                    continue;
                }
            }
        }

        let corpus = collect_project_corpus(
            project_dir,
            cfg.max_chars_per_project as usize,
            latest_mtime,
        )?;
        let project_id = upsert_project(
            &conn,
            &corpus.doc.path.to_string_lossy(),
            &corpus.doc.title,
            &corpus.doc.summary,
            corpus.doc.mtime,
            now,
        )?;
        stats.updated_projects += 1;
        docs_by_id.insert(project_id, corpus.doc.clone());

        match embedder.embed_one(&format!("{}\n{}", corpus.doc.title, corpus.doc.summary)) {
            Ok(pvec) => {
                set_project_vector(&conn, project_id, &model_key, &pvec)?;
                stats.vectorized_projects += 1;
            }
            Err(_) => {
                stats.vector_failures += 1;
            }
        }

        let (rows, vecs, failures) = reindex_project_chunks(
            &conn,
            project_id,
            &model_key,
            embedder.as_ref(),
            &corpus.chunks,
            now,
        )?;
        stats.chunk_rows += rows;
        stats.chunk_vectors += vecs;
        stats.vector_failures += failures;
        if emit_progress {
            println!(
                "[{}/{}] index {} chunks={} chunk_vecs={} vector_failures={}",
                idx + 1,
                projects.len(),
                project_dir
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("project"),
                rows,
                vecs,
                failures
            );
        }
    }

    if remove_missing {
        stats.removed_projects = remove_projects_not_in(&conn, &keep_paths)?;
    }
    stats.graph_edges = rebuild_relationship_edges(&conn, &docs_by_id)?;
    if cfg.retrieval_backend == "falkordb" {
        let retrieval = falkor_store::sync_from_sqlite(&conn, &model_key)
            .map_err(|e| format!("falkordb sync failed: {}", e))?;
        stats.retrieval_synced_chunks = retrieval.chunk_nodes;
    }

    conn.execute_batch("PRAGMA optimize;")
        .map_err(|e| format!("database optimize failed: {}", e))?;
    Ok(stats)
}

fn print_index_stats(stats: &IndexStats, cfg: &ConfigValues) {
    println!("default root: {}", cfg.root.display());
    println!("tracked roots indexed: {}", stats.tracked_roots);
    println!("projects found: {}", stats.total_projects);
    println!("projects updated: {}", stats.updated_projects);
    println!("projects skipped (unchanged): {}", stats.skipped_projects);
    println!("projects removed: {}", stats.removed_projects);
    println!("vectors refreshed: {}", stats.vectorized_projects);
    println!("chunks indexed: {}", stats.chunk_rows);
    println!("chunk vectors refreshed: {}", stats.chunk_vectors);
    println!("graph edges refreshed: {}", stats.graph_edges);
    println!("retrieval backend: {}", stats.retrieval_backend);
    println!("retrieval chunks synced: {}", stats.retrieval_synced_chunks);
    if !stats.retrieval_error.is_empty() {
        println!("retrieval sync warning: {}", stats.retrieval_error);
    }
    if stats.vector_failures > 0 {
        println!("vector failures: {}", stats.vector_failures);
    }
}

#[derive(Clone)]
struct EvidenceHit {
    chunk_id: i64,
    chunk_index: i64,
    doc_path: String,
    doc_rel_path: String,
    score: f64,
    semantic: f64,
    lexical: f64,
    graph: f64,
    relation: String,
    quality: f64,
    excerpt: String,
}

#[derive(Clone)]
struct RankedResult {
    path: String,
    score: f64,
    lexical: f64,
    semantic: f64,
    frecency: f64,
    graph: f64,
    evidence: Vec<EvidenceHit>,
}

#[derive(Clone)]
struct RankedFileResult {
    path: String,
    project_path: String,
    doc_rel_path: String,
    chunk_id: i64,
    chunk_index: i64,
    score: f64,
    semantic: f64,
    lexical: f64,
    graph: f64,
    relation: String,
    quality: f64,
    excerpt: String,
    evidence: Vec<EvidenceHit>,
}

#[derive(Clone)]
struct RankedChunkResult {
    chunk_id: i64,
    chunk_index: i64,
    path: String,
    project_path: String,
    doc_rel_path: String,
    score: f64,
    semantic: f64,
    lexical: f64,
    graph: f64,
    relation: String,
    quality: f64,
    excerpt: String,
}

#[derive(Clone)]
struct RelatedChunkResult {
    chunk_id: i64,
    chunk_index: i64,
    path: String,
    project_path: String,
    doc_rel_path: String,
    relation: String,
    relation_weight: f64,
    relation_quality: String,
    relation_quality_multiplier: f64,
    score: f64,
    semantic: f64,
    lexical: f64,
    quality: f64,
    excerpt: String,
}

fn chunk_search_schema() -> &'static str {
    "chunk-search-v1"
}

fn chunk_related_schema() -> &'static str {
    "chunk-related-v1"
}

fn chunk_get_schema() -> &'static str {
    "chunk-get-v1"
}

fn doc_read_schema() -> &'static str {
    "doc-read-v1"
}

fn ranked_chunk_result_json(item: &RankedChunkResult) -> Value {
    serde_json::json!({
        "chunk_id": item.chunk_id,
        "chunk_index": item.chunk_index,
        "path": item.path,
        "project_path": item.project_path,
        "doc_rel_path": item.doc_rel_path,
        "score": item.score,
        "semantic": item.semantic,
        "lexical": item.lexical,
        "graph": item.graph,
        "relation": item.relation,
        "quality": item.quality,
        "excerpt": item.excerpt,
    })
}

fn related_chunk_result_json(item: &RelatedChunkResult) -> Value {
    serde_json::json!({
        "chunk_id": item.chunk_id,
        "chunk_index": item.chunk_index,
        "path": item.path,
        "project_path": item.project_path,
        "doc_rel_path": item.doc_rel_path,
        "relation": item.relation,
        "relation_weight": item.relation_weight,
        "relation_quality": item.relation_quality,
        "relation_quality_multiplier": item.relation_quality_multiplier,
        "score": item.score,
        "semantic": item.semantic,
        "lexical": item.lexical,
        "quality": item.quality,
        "excerpt": item.excerpt,
    })
}

#[derive(Clone)]
struct ChunkSignal {
    chunk_id: i64,
    chunk_index: i64,
    project_path: String,
    doc_path: String,
    doc_rel_path: String,
    semantic: f64,
    lexical: f64,
    graph: f64,
    relation: String,
    quality: f64,
    excerpt: String,
}

fn print_project_results(results: &[RankedResult]) {
    for (idx, item) in results.iter().enumerate() {
        println!(
            "{:>2}. {}\n    score={:.3} semantic={:.3} lexical={:.3} frecency={:.3} graph={:.3}",
            idx + 1,
            item.path,
            item.score,
            item.semantic,
            item.lexical,
            item.frecency,
            item.graph
        );
        for ev in item.evidence.iter().take(4) {
            println!(
                "      - {}#{} (chunk {}) score={:.3} sem={:.3} lex={:.3} gr={:.3} rel={} q={:.2}\n        {}",
                ev.doc_rel_path,
                ev.chunk_index,
                ev.chunk_id,
                ev.score,
                ev.semantic,
                ev.lexical,
                ev.graph,
                ev.relation,
                ev.quality,
                ev.excerpt
            );
        }
    }
}

fn print_file_results(results: &[RankedFileResult]) {
    for (idx, item) in results.iter().enumerate() {
        println!(
            "{:>2}. {}\n    project={}\n    chunk_id={} chunk_index={}\n    score={:.3} semantic={:.3} lexical={:.3} graph={:.3} relation={} quality={:.2}\n    {}",
            idx + 1,
            item.path,
            item.project_path,
            item.chunk_id,
            item.chunk_index,
            item.score,
            item.semantic,
            item.lexical,
            item.graph,
            item.relation,
            item.quality,
            item.excerpt
        );
        for ev in item.evidence.iter().take(4) {
            println!(
                "      - {}#{} (chunk {}) score={:.3} sem={:.3} lex={:.3} gr={:.3} rel={} q={:.2}\n        {}",
                ev.doc_rel_path,
                ev.chunk_index,
                ev.chunk_id,
                ev.score,
                ev.semantic,
                ev.lexical,
                ev.graph,
                ev.relation,
                ev.quality,
                ev.excerpt
            );
        }
    }
}

fn rank_projects_native(
    conn: &Connection,
    cfg: &ConfigValues,
    query: &str,
    limit: usize,
) -> Result<Vec<RankedResult>, String> {
    let existing_paths = list_project_paths(conn)?;
    if existing_paths.is_empty() {
        return Ok(Vec::new());
    }
    let q = query.trim();
    if q.is_empty() {
        return rank_by_frecency_only(conn, limit);
    }
    let keyword_focused = is_keyword_focused_query(q);

    let embedder = build_embedder(cfg)?;
    let query_vector = embedder.embed_one(q)?;
    let model_key = embedder.model_key();

    let project_semantic = project_semantic_scores(
        conn,
        &model_key,
        &query_vector,
        cfg.vector_candidates.max(1) as usize,
    )?;
    let mut fused = if cfg.retrieval_backend == "falkordb" {
        match falkor_store::search_hybrid_chunks(
            q,
            &query_vector,
            std::cmp::max(80, (cfg.vector_candidates.max(1) * 3) as usize),
            cfg.lexical_candidates.max(1) as usize,
        ) {
            Ok(hybrid) => chunk_signals_for_ids(conn, &hybrid.semantic, &hybrid.lexical)?,
            Err(err) => {
                if falkor_debug_enabled() {
                    eprintln!(
                        "warning: falkordb hybrid query failed in project ranking ({}); using sqlite hybrid fallback",
                        err
                    );
                }
                let lexical_chunks =
                    search_lexical_chunks_sqlite(conn, q, cfg.lexical_candidates.max(1) as usize)?;
                let semantic_chunks = semantic_chunk_scores(
                    conn,
                    &model_key,
                    &query_vector,
                    std::cmp::max(80, (cfg.vector_candidates.max(1) * 3) as usize),
                )?;
                fuse_chunk_signals(&lexical_chunks, &semantic_chunks)
            }
        }
    } else {
        let lexical_chunks =
            search_lexical_chunks_sqlite(conn, q, cfg.lexical_candidates.max(1) as usize)?;
        let semantic_chunks = semantic_chunk_scores(
            conn,
            &model_key,
            &query_vector,
            std::cmp::max(80, (cfg.vector_candidates.max(1) * 3) as usize),
        )?;
        fuse_chunk_signals(&lexical_chunks, &semantic_chunks)
    };
    apply_graph_chunk_expansion(conn, &mut fused, cfg)?;
    let project_evidence = project_evidence(&fused, cfg);
    let project_content = project_content_scores(&project_evidence);
    let frecency = frecency_scores(conn)?;
    let graph = graph_scores(conn)?;
    let path_keywords = path_keyword_scores(&existing_paths, q);

    let existing_set: HashSet<String> = existing_paths.iter().cloned().collect();
    let mut all_paths: HashSet<String> = HashSet::new();
    for p in project_content.keys() {
        if existing_set.contains(p) {
            all_paths.insert(p.clone());
        }
    }
    for p in project_semantic.keys() {
        if existing_set.contains(p) {
            all_paths.insert(p.clone());
        }
    }
    for p in frecency.keys() {
        if existing_set.contains(p) {
            all_paths.insert(p.clone());
        }
    }
    for p in graph.keys() {
        if existing_set.contains(p) {
            all_paths.insert(p.clone());
        }
    }
    for p in path_keywords.keys() {
        if existing_set.contains(p) {
            all_paths.insert(p.clone());
        }
    }
    if all_paths.is_empty() {
        for p in existing_paths.iter().take(limit.max(1)) {
            all_paths.insert(p.clone());
        }
    }

    let mut out: Vec<RankedResult> = Vec::new();
    for path in all_paths {
        let evidence = project_evidence.get(&path).cloned().unwrap_or_default();
        let lexical = evidence.iter().map(|e| e.lexical).fold(0.0f64, f64::max);
        let sem_from_chunks = evidence.iter().map(|e| e.semantic).fold(0.0f64, f64::max);
        let semantic = sem_from_chunks.max(*project_semantic.get(&path).unwrap_or(&0.0));
        let content = *project_content.get(&path).unwrap_or(&0.0);
        let fr = *frecency.get(&path).unwrap_or(&0.0);
        let gr = *graph.get(&path).unwrap_or(&0.0);
        let path_kw = *path_keywords.get(&path).unwrap_or(&0.0);
        let mut score = if cfg.embed_backend == "local-hash" {
            if keyword_focused {
                (0.18 * content) + (0.10 * semantic) + (0.54 * path_kw) + (0.12 * fr) + (0.06 * gr)
            } else {
                (0.32 * content) + (0.22 * semantic) + (0.24 * path_kw) + (0.12 * fr) + (0.10 * gr)
            }
        } else if keyword_focused {
            (0.42 * content) + (0.18 * semantic) + (0.24 * path_kw) + (0.10 * fr) + (0.06 * gr)
        } else {
            (cfg.rank_project_content_weight * content)
                + (cfg.rank_project_semantic_weight * semantic)
                + (cfg.rank_project_path_weight * path_kw)
                + (cfg.rank_project_graph_weight * gr)
                + (cfg.rank_project_frecency_weight * fr)
        };
        if keyword_focused && path_kw < 0.20 && lexical < 0.05 {
            score *= 0.45;
        }
        if is_generic_container(&path) && path_kw < 0.4 {
            score *= 0.82;
        }
        out.push(RankedResult {
            path,
            score,
            lexical,
            semantic,
            frecency: fr,
            graph: gr,
            evidence: evidence.into_iter().take(4).collect(),
        });
    }
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    out.truncate(limit.max(1));
    Ok(out)
}

fn rank_files_native(
    conn: &Connection,
    cfg: &ConfigValues,
    query: &str,
    limit: usize,
) -> Result<Vec<RankedFileResult>, String> {
    let q = query.trim();
    if q.is_empty() {
        return Ok(Vec::new());
    }
    let keyword_focused = is_keyword_focused_query(q);
    let embedder = build_embedder(cfg)?;
    let query_vector = embedder.embed_one(q)?;
    let model_key = embedder.model_key();

    let project_semantic = project_semantic_scores(
        conn,
        &model_key,
        &query_vector,
        std::cmp::max(120, (cfg.vector_candidates.max(1) * 2) as usize),
    )?;
    let mut fused = if cfg.retrieval_backend == "falkordb" {
        match falkor_store::search_hybrid_chunks(
            q,
            &query_vector,
            std::cmp::max(160, (cfg.vector_candidates.max(1) * 4) as usize),
            std::cmp::max(120, (cfg.lexical_candidates.max(1) * 2) as usize),
        ) {
            Ok(hybrid) => chunk_signals_for_ids(conn, &hybrid.semantic, &hybrid.lexical)?,
            Err(err) => {
                if falkor_debug_enabled() {
                    eprintln!(
                        "warning: falkordb hybrid query failed in file ranking ({}); using sqlite hybrid fallback",
                        err
                    );
                }
                let lexical_chunks = search_lexical_chunks_sqlite(
                    conn,
                    q,
                    std::cmp::max(120, (cfg.lexical_candidates.max(1) * 2) as usize),
                )?;
                let semantic_chunks = semantic_chunk_scores(
                    conn,
                    &model_key,
                    &query_vector,
                    std::cmp::max(160, (cfg.vector_candidates.max(1) * 4) as usize),
                )?;
                fuse_chunk_signals(&lexical_chunks, &semantic_chunks)
            }
        }
    } else {
        let lexical_chunks = search_lexical_chunks_sqlite(
            conn,
            q,
            std::cmp::max(120, (cfg.lexical_candidates.max(1) * 2) as usize),
        )?;
        let semantic_chunks = semantic_chunk_scores(
            conn,
            &model_key,
            &query_vector,
            std::cmp::max(160, (cfg.vector_candidates.max(1) * 4) as usize),
        )?;
        fuse_chunk_signals(&lexical_chunks, &semantic_chunks)
    };
    if fused.is_empty() {
        return Ok(Vec::new());
    }
    if keyword_focused {
        let path_signals = keyword_path_chunk_scores(
            conn,
            q,
            std::cmp::max(220, (cfg.lexical_candidates.max(1) * 3) as usize),
        )?;
        if !path_signals.is_empty() {
            fused = fuse_chunk_signals(&path_signals, &fused);
        }
    }
    apply_graph_chunk_expansion(conn, &mut fused, cfg)?;

    let frecency = frecency_scores(conn)?;
    let project_paths = list_project_paths(conn)?;
    let project_path_keywords = path_keyword_scores(&project_paths, q);
    let mut by_file: HashMap<String, RankedFileResult> = HashMap::new();
    for row in fused.values() {
        let content = chunk_base_score(row, cfg);
        let project_sem = *project_semantic.get(&row.project_path).unwrap_or(&0.0);
        let fr = *frecency.get(&row.project_path).unwrap_or(&0.0);
        let doc_kw = doc_keyword_score(&row.doc_rel_path, q);
        let project_kw = *project_path_keywords.get(&row.project_path).unwrap_or(&0.0);
        let kw = doc_kw.max(0.72 * project_kw);
        let mut score = if cfg.embed_backend == "local-hash" {
            if keyword_focused {
                (0.16 * content)
                    + (0.08 * project_sem)
                    + (0.36 * kw)
                    + (0.26 * row.lexical)
                    + (0.08 * fr)
                    + (0.06 * row.graph)
            } else {
                (0.34 * content)
                    + (0.20 * project_sem)
                    + (0.20 * kw)
                    + (0.10 * row.lexical)
                    + (0.08 * fr)
                    + (0.08 * row.graph)
            }
        } else if keyword_focused {
            (0.42 * content)
                + (0.12 * project_sem)
                + (0.22 * kw)
                + (0.10 * row.lexical)
                + (0.08 * fr)
                + (0.06 * row.graph)
        } else {
            (0.56 * content) + (0.14 * project_sem) + (0.08 * fr) + (0.08 * kw) + (0.14 * row.graph)
        };
        if keyword_focused && kw < 0.20 && row.lexical < 0.05 {
            score *= 0.40;
        }
        score *= path_noise_penalty(&row.doc_rel_path);
        let candidate = RankedFileResult {
            path: row.doc_path.clone(),
            project_path: row.project_path.clone(),
            doc_rel_path: row.doc_rel_path.clone(),
            chunk_id: row.chunk_id,
            chunk_index: row.chunk_index,
            score,
            semantic: row.semantic,
            lexical: row.lexical,
            graph: row.graph,
            relation: row.relation.clone(),
            quality: row.quality,
            excerpt: row.excerpt.clone(),
            evidence: Vec::new(),
        };
        let prev = by_file.get(&row.doc_path);
        if prev.is_none() || candidate.score > prev.map(|p| p.score).unwrap_or(0.0) {
            by_file.insert(row.doc_path.clone(), candidate);
        }
    }
    for (doc_path, item) in &mut by_file {
        let mut support: Vec<EvidenceHit> = Vec::new();
        for row in fused.values() {
            let include = row.doc_path == *doc_path
                || (row.project_path == item.project_path
                    && row.doc_path != *doc_path
                    && row.graph >= 0.55)
                || (row.project_path != item.project_path && row.graph >= 0.72);
            if !include {
                continue;
            }
            support.push(evidence_hit_from_chunk(row, chunk_base_score(row, cfg)));
        }
        support.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| b.graph.total_cmp(&a.graph))
                .then_with(|| b.chunk_id.cmp(&a.chunk_id))
        });
        support.truncate(4);
        item.evidence = support;
    }

    let mut out: Vec<RankedFileResult> = by_file.into_values().collect();
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    out.truncate(limit.max(1));
    Ok(out)
}

fn rank_chunks_native(
    conn: &Connection,
    cfg: &ConfigValues,
    query: &str,
    limit: usize,
) -> Result<Vec<RankedChunkResult>, String> {
    let q = query.trim();
    if q.is_empty() {
        return Ok(Vec::new());
    }
    let embedder = build_embedder(cfg)?;
    let query_vector = embedder.embed_one(q)?;
    let model_key = embedder.model_key();

    let project_semantic = project_semantic_scores(
        conn,
        &model_key,
        &query_vector,
        std::cmp::max(120, (cfg.vector_candidates.max(1) * 2) as usize),
    )?;
    let mut fused = if cfg.retrieval_backend == "falkordb" {
        match falkor_store::search_hybrid_chunks(
            q,
            &query_vector,
            std::cmp::max(220, (cfg.vector_candidates.max(1) * 5) as usize),
            std::cmp::max(220, (cfg.lexical_candidates.max(1) * 3) as usize),
        ) {
            Ok(hybrid) => chunk_signals_for_ids(conn, &hybrid.semantic, &hybrid.lexical)?,
            Err(err) => {
                if falkor_debug_enabled() {
                    eprintln!(
                        "warning: falkordb hybrid query failed in chunk ranking ({}); using sqlite hybrid fallback",
                        err
                    );
                }
                let lexical_chunks = search_lexical_chunks_sqlite(
                    conn,
                    q,
                    std::cmp::max(220, (cfg.lexical_candidates.max(1) * 3) as usize),
                )?;
                let semantic_chunks = semantic_chunk_scores(
                    conn,
                    &model_key,
                    &query_vector,
                    std::cmp::max(220, (cfg.vector_candidates.max(1) * 5) as usize),
                )?;
                fuse_chunk_signals(&lexical_chunks, &semantic_chunks)
            }
        }
    } else {
        let lexical_chunks = search_lexical_chunks_sqlite(
            conn,
            q,
            std::cmp::max(220, (cfg.lexical_candidates.max(1) * 3) as usize),
        )?;
        let semantic_chunks = semantic_chunk_scores(
            conn,
            &model_key,
            &query_vector,
            std::cmp::max(220, (cfg.vector_candidates.max(1) * 5) as usize),
        )?;
        fuse_chunk_signals(&lexical_chunks, &semantic_chunks)
    };
    if fused.is_empty() {
        return Ok(Vec::new());
    }
    apply_graph_chunk_expansion(conn, &mut fused, cfg)?;

    let frecency = frecency_scores(conn)?;
    let mut out: Vec<RankedChunkResult> = Vec::new();
    for row in fused.values() {
        let content = chunk_base_score(row, cfg);
        let project_sem = *project_semantic.get(&row.project_path).unwrap_or(&0.0);
        let fr = *frecency.get(&row.project_path).unwrap_or(&0.0);
        let kw = doc_keyword_score(&row.doc_rel_path, q);
        let score = (0.54 * content)
            + (0.16 * project_sem)
            + (0.10 * fr)
            + (0.08 * kw)
            + (0.12 * row.graph);
        out.push(RankedChunkResult {
            chunk_id: row.chunk_id,
            chunk_index: row.chunk_index,
            path: row.doc_path.clone(),
            project_path: row.project_path.clone(),
            doc_rel_path: row.doc_rel_path.clone(),
            score,
            semantic: row.semantic,
            lexical: row.lexical,
            graph: row.graph,
            relation: row.relation.clone(),
            quality: row.quality,
            excerpt: row.excerpt.clone(),
        });
    }
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    out.truncate(limit.max(1));
    Ok(out)
}

#[derive(Clone)]
struct SourceChunk {
    chunk_id: i64,
    chunk_index: i64,
    project_path: String,
    doc_path: String,
    doc_rel_path: String,
    text: String,
}

#[derive(Clone)]
struct IndexedChunkRow {
    chunk_id: i64,
    chunk_index: i64,
    project_path: String,
    doc_path: String,
    doc_rel_path: String,
    doc_mtime: f64,
    token_count: i64,
    text: String,
}

#[derive(Clone)]
struct ChunkRelationFeedbackRow {
    src_chunk_id: i64,
    dst_chunk_id: i64,
    relation: String,
    decision: String,
    quality_label: String,
    note: String,
    source: String,
    created_at: f64,
    updated_at: f64,
    dst_chunk_index: i64,
    dst_doc_path: String,
    dst_doc_rel_path: String,
    dst_project_path: String,
}

fn apply_chunk_relation_decision(
    conn: &Connection,
    src_chunk_id: i64,
    dst_chunk_id: i64,
    relation: &str,
    decision: &str,
    note: &str,
    source: &str,
    ts: f64,
) -> Result<(), String> {
    let rel = relation.trim();
    if src_chunk_id <= 0 || dst_chunk_id <= 0 {
        return Err("source_chunk_id and target_chunk_id must be positive integers".to_string());
    }
    if rel.is_empty() {
        return Err("relation must be non-empty".to_string());
    }
    if !matches!(decision, "active" | "suppressed") {
        return Err("decision must be 'active' or 'suppressed'".to_string());
    }
    conn.execute(
        r#"
INSERT INTO chunk_relation_feedback(
    src_chunk_id, dst_chunk_id, relation, decision, note, source, created_at, updated_at
)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)
ON CONFLICT(src_chunk_id, dst_chunk_id, relation) DO UPDATE SET
    decision = excluded.decision,
    note = excluded.note,
    source = excluded.source,
    updated_at = excluded.updated_at
"#,
        params![
            src_chunk_id,
            dst_chunk_id,
            rel,
            decision,
            note.trim(),
            source.trim(),
            ts
        ],
    )
    .map_err(|e| format!("failed writing relation feedback: {}", e))?;
    Ok(())
}

fn normalize_relation_quality_label(raw: &str) -> Option<&'static str> {
    let v = raw.trim().to_lowercase();
    match v.as_str() {
        "" | "unspecified" | "clear" | "none" => Some("unspecified"),
        "good" => Some("good"),
        "weak" => Some("weak"),
        "wrong" => Some("wrong"),
        _ => None,
    }
}

fn relation_quality_multiplier(cfg: &ConfigValues, quality_label: &str) -> f64 {
    match quality_label {
        "good" => 1.0 + cfg.rank_relation_quality_good_boost.clamp(0.0, 1.0),
        "weak" => (1.0 - cfg.rank_relation_quality_weak_penalty.clamp(0.0, 1.0)).max(0.0),
        "wrong" => (1.0 - cfg.rank_relation_quality_wrong_penalty.clamp(0.0, 1.0)).max(0.0),
        _ => 1.0,
    }
}

fn set_chunk_relation_quality(
    conn: &Connection,
    src_chunk_id: i64,
    dst_chunk_id: i64,
    relation: &str,
    quality_label: &str,
    note: &str,
    source: &str,
    ts: f64,
) -> Result<String, String> {
    let rel = relation.trim();
    if src_chunk_id <= 0 || dst_chunk_id <= 0 {
        return Err("source_chunk_id and target_chunk_id must be positive integers".to_string());
    }
    if rel.is_empty() {
        return Err("relation must be non-empty".to_string());
    }
    let label = normalize_relation_quality_label(quality_label).ok_or_else(|| {
        "quality_label must be one of: good, weak, wrong, unspecified".to_string()
    })?;
    conn.execute(
        r#"
INSERT INTO chunk_relation_feedback(
    src_chunk_id, dst_chunk_id, relation, decision, quality_label, note, source, created_at, updated_at
)
VALUES (?1, ?2, ?3, 'active', ?4, ?5, ?6, ?7, ?7)
ON CONFLICT(src_chunk_id, dst_chunk_id, relation) DO UPDATE SET
    quality_label = excluded.quality_label,
    note = CASE
        WHEN excluded.note <> '' THEN excluded.note
        ELSE chunk_relation_feedback.note
    END,
    source = excluded.source,
    updated_at = excluded.updated_at
"#,
        params![
            src_chunk_id,
            dst_chunk_id,
            rel,
            label,
            note.trim(),
            source.trim(),
            ts
        ],
    )
    .map_err(|e| format!("failed writing relation quality: {}", e))?;
    Ok(label.to_string())
}

fn list_chunk_relation_feedback(
    conn: &Connection,
    src_chunk_id: i64,
    decision: Option<&str>,
    quality_label: Option<&str>,
    limit: usize,
) -> Result<Vec<ChunkRelationFeedbackRow>, String> {
    let decision_filter = decision.unwrap_or("").trim().to_string();
    let quality_filter = quality_label.unwrap_or("").trim().to_string();
    let mut stmt = conn
        .prepare(
            r#"
SELECT
    f.src_chunk_id,
    f.dst_chunk_id,
    f.relation,
    f.decision,
    f.quality_label,
    f.note,
    f.source,
    f.created_at,
    f.updated_at,
    COALESCE(pc.chunk_index, -1) AS dst_chunk_index,
    COALESCE(pc.doc_path, '') AS dst_doc_path,
    COALESCE(pc.doc_rel_path, '') AS dst_doc_rel_path,
    COALESCE(p.path, '') AS dst_project_path
FROM chunk_relation_feedback f
LEFT JOIN project_chunks pc ON pc.id = f.dst_chunk_id
LEFT JOIN projects p ON p.id = pc.project_id
WHERE f.src_chunk_id = ?1
  AND (?2 = '' OR f.decision = ?2)
  AND (?3 = '' OR f.quality_label = ?3)
ORDER BY f.updated_at DESC, f.id DESC
LIMIT ?4
"#,
        )
        .map_err(|e| format!("failed preparing relation feedback query: {}", e))?;
    let rows = stmt
        .query_map(
            params![
                src_chunk_id,
                decision_filter,
                quality_filter,
                limit.max(1).min(2000) as i64
            ],
            |row| {
                Ok(ChunkRelationFeedbackRow {
                    src_chunk_id: row.get(0)?,
                    dst_chunk_id: row.get(1)?,
                    relation: row.get(2)?,
                    decision: row.get(3)?,
                    quality_label: row.get(4)?,
                    note: row.get(5)?,
                    source: row.get(6)?,
                    created_at: row.get(7)?,
                    updated_at: row.get(8)?,
                    dst_chunk_index: row.get(9)?,
                    dst_doc_path: row.get(10)?,
                    dst_doc_rel_path: row.get(11)?,
                    dst_project_path: row.get(12)?,
                })
            },
        )
        .map_err(|e| format!("failed querying relation feedback rows: {}", e))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(|e| format!("failed reading relation feedback row: {}", e))?);
    }
    Ok(out)
}

fn suppressed_relation_set(
    conn: &Connection,
    src_chunk_id: i64,
) -> Result<HashSet<(i64, String)>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT dst_chunk_id, relation
FROM chunk_relation_feedback
WHERE src_chunk_id = ?1
  AND decision = 'suppressed'
"#,
        )
        .map_err(|e| format!("failed preparing suppressed relation query: {}", e))?;
    let rows = stmt
        .query_map(params![src_chunk_id], |row| {
            let dst: i64 = row.get(0)?;
            let relation: String = row.get(1)?;
            Ok((dst, relation))
        })
        .map_err(|e| format!("failed querying suppressed relation rows: {}", e))?;
    let mut out: HashSet<(i64, String)> = HashSet::new();
    for row in rows {
        let (dst, relation) =
            row.map_err(|e| format!("failed reading suppressed relation row: {}", e))?;
        out.insert((dst, relation));
    }
    Ok(out)
}

fn active_relation_quality_map(
    conn: &Connection,
    src_chunk_id: i64,
) -> Result<HashMap<(i64, String), String>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT dst_chunk_id, relation, quality_label
FROM chunk_relation_feedback
WHERE src_chunk_id = ?1
  AND decision = 'active'
"#,
        )
        .map_err(|e| format!("failed preparing relation quality query: {}", e))?;
    let rows = stmt
        .query_map(params![src_chunk_id], |row| {
            let dst: i64 = row.get(0)?;
            let relation: String = row.get(1)?;
            let quality_label: String = row.get(2)?;
            Ok((dst, relation, quality_label))
        })
        .map_err(|e| format!("failed querying relation quality rows: {}", e))?;
    let mut out: HashMap<(i64, String), String> = HashMap::new();
    for row in rows {
        let (dst, relation, quality_label) =
            row.map_err(|e| format!("failed reading relation quality row: {}", e))?;
        out.insert((dst, relation), quality_label);
    }
    Ok(out)
}

fn relation_feedback_row_json(row: &ChunkRelationFeedbackRow) -> Value {
    serde_json::json!({
        "source_chunk_id": row.src_chunk_id,
        "target_chunk_id": row.dst_chunk_id,
        "relation": row.relation,
        "decision": row.decision,
        "quality_label": row.quality_label,
        "note": row.note,
        "source": row.source,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "target_chunk_index": row.dst_chunk_index,
        "target_path": row.dst_doc_path,
        "target_doc_rel_path": row.dst_doc_rel_path,
        "target_project_path": row.dst_project_path,
    })
}

fn source_chunk_json(source: &SourceChunk) -> Value {
    serde_json::json!({
        "chunk_id": source.chunk_id,
        "chunk_index": source.chunk_index,
        "path": source.doc_path,
        "project_path": source.project_path,
        "doc_rel_path": source.doc_rel_path,
    })
}

#[cfg(test)]
mod chunk_contract_tests {
    use super::*;

    #[test]
    fn chunk_schema_constants_are_stable() {
        assert_eq!(chunk_search_schema(), "chunk-search-v1");
        assert_eq!(chunk_related_schema(), "chunk-related-v1");
        assert_eq!(chunk_get_schema(), "chunk-get-v1");
        assert_eq!(doc_read_schema(), "doc-read-v1");
    }

    #[test]
    fn ranked_chunk_json_contract_fields() {
        let item = RankedChunkResult {
            chunk_id: 11,
            chunk_index: 3,
            path: "/tmp/a.md".to_string(),
            project_path: "/tmp".to_string(),
            doc_rel_path: "a.md".to_string(),
            score: 0.9,
            semantic: 0.8,
            lexical: 0.7,
            graph: 0.6,
            relation: "direct".to_string(),
            quality: 1.0,
            excerpt: "hello".to_string(),
        };
        let json = ranked_chunk_result_json(&item);
        let obj = json.as_object().expect("expected object");
        assert_eq!(obj.len(), 12);
        for key in [
            "chunk_id",
            "chunk_index",
            "path",
            "project_path",
            "doc_rel_path",
            "score",
            "semantic",
            "lexical",
            "graph",
            "relation",
            "quality",
            "excerpt",
        ] {
            assert!(obj.contains_key(key), "missing key: {}", key);
        }
    }

    #[test]
    fn related_chunk_json_contract_fields() {
        let item = RelatedChunkResult {
            chunk_id: 19,
            chunk_index: 7,
            path: "/tmp/b.md".to_string(),
            project_path: "/tmp".to_string(),
            doc_rel_path: "b.md".to_string(),
            relation: "same_project".to_string(),
            relation_weight: 0.82,
            relation_quality: "good".to_string(),
            relation_quality_multiplier: 1.08,
            score: 0.88,
            semantic: 0.77,
            lexical: 0.66,
            quality: 0.95,
            excerpt: "world".to_string(),
        };
        let json = related_chunk_result_json(&item);
        let obj = json.as_object().expect("expected object");
        assert_eq!(obj.len(), 14);
        for key in [
            "chunk_id",
            "chunk_index",
            "path",
            "project_path",
            "doc_rel_path",
            "relation",
            "relation_weight",
            "relation_quality",
            "relation_quality_multiplier",
            "score",
            "semantic",
            "lexical",
            "quality",
            "excerpt",
        ] {
            assert!(obj.contains_key(key), "missing key: {}", key);
        }
    }

    #[test]
    fn relation_feedback_json_contract_fields() {
        let row = ChunkRelationFeedbackRow {
            src_chunk_id: 10,
            dst_chunk_id: 12,
            relation: "same_project".to_string(),
            decision: "active".to_string(),
            quality_label: "weak".to_string(),
            note: "test".to_string(),
            source: "unit".to_string(),
            created_at: 1.0,
            updated_at: 2.0,
            dst_chunk_index: 3,
            dst_doc_path: "/tmp/a.md".to_string(),
            dst_doc_rel_path: "a.md".to_string(),
            dst_project_path: "/tmp".to_string(),
        };
        let json = relation_feedback_row_json(&row);
        let obj = json.as_object().expect("expected object");
        assert!(obj.contains_key("quality_label"));
        assert_eq!(
            obj.get("quality_label")
                .and_then(|v| v.as_str())
                .unwrap_or_default(),
            "weak"
        );
    }

    #[test]
    fn relation_quality_label_normalization_contract() {
        assert_eq!(normalize_relation_quality_label("good"), Some("good"));
        assert_eq!(normalize_relation_quality_label("weak"), Some("weak"));
        assert_eq!(normalize_relation_quality_label("wrong"), Some("wrong"));
        assert_eq!(
            normalize_relation_quality_label("clear"),
            Some("unspecified")
        );
        assert_eq!(normalize_relation_quality_label(""), Some("unspecified"));
        assert_eq!(normalize_relation_quality_label("bogus"), None);
    }

    #[test]
    fn relation_feedback_quality_filters_and_preserves_decision() {
        let conn = Connection::open_in_memory().expect("open in-memory sqlite");
        init_schema(&conn).expect("init schema");
        conn.execute_batch(
            r#"
INSERT INTO projects(path, title, summary, project_mtime, last_indexed)
VALUES ('/tmp/p', 'p', 'p', 0, 0);
INSERT INTO project_chunks(project_id, doc_path, doc_rel_path, doc_mtime, chunk_index, token_count, text_hash, text, updated_at)
VALUES (1, '/tmp/p/a.md', 'a.md', 0, 0, 10, 'h1', 'alpha', 0);
INSERT INTO project_chunks(project_id, doc_path, doc_rel_path, doc_mtime, chunk_index, token_count, text_hash, text, updated_at)
VALUES (1, '/tmp/p/b.md', 'b.md', 0, 1, 10, 'h2', 'beta', 0);
"#,
        )
        .expect("seed rows");

        apply_chunk_relation_decision(
            &conn,
            1,
            2,
            "same_project",
            "suppressed",
            "suppress first",
            "test",
            10.0,
        )
        .expect("suppress relation");
        let set_label = set_chunk_relation_quality(
            &conn,
            1,
            2,
            "same_project",
            "good",
            "quality set",
            "test",
            11.0,
        )
        .expect("set quality");
        assert_eq!(set_label, "good");

        let rows = list_chunk_relation_feedback(&conn, 1, Some("suppressed"), Some("good"), 20)
            .expect("query feedback");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].decision, "suppressed");
        assert_eq!(rows[0].quality_label, "good");

        let cleared = set_chunk_relation_quality(
            &conn,
            1,
            2,
            "same_project",
            "unspecified",
            "",
            "test",
            12.0,
        )
        .expect("clear quality");
        assert_eq!(cleared, "unspecified");
        let rows_after =
            list_chunk_relation_feedback(&conn, 1, None, Some("good"), 20).expect("query cleared");
        assert_eq!(rows_after.len(), 0);
    }

    #[test]
    fn mcp_specs_include_relation_quality_tools() {
        let specs = mcp_tool_specs();
        let mut names: HashSet<String> = HashSet::new();
        for spec in specs {
            if let Some(name) = spec.get("name").and_then(|v| v.as_str()) {
                names.insert(name.to_string());
            }
        }
        assert!(names.contains("list_relation_feedback"));
        assert!(names.contains("set_relation_quality"));
    }

    #[test]
    fn source_chunk_json_contract_fields() {
        let source = SourceChunk {
            chunk_id: 5,
            chunk_index: 2,
            project_path: "/tmp".to_string(),
            doc_path: "/tmp/c.md".to_string(),
            doc_rel_path: "c.md".to_string(),
            text: "body".to_string(),
        };
        let json = source_chunk_json(&source);
        let obj = json.as_object().expect("expected object");
        assert_eq!(obj.len(), 5);
        for key in [
            "chunk_id",
            "chunk_index",
            "path",
            "project_path",
            "doc_rel_path",
        ] {
            assert!(obj.contains_key(key), "missing key: {}", key);
        }
    }

    #[test]
    fn truncate_text_chars_contract() {
        let (full, full_truncated, full_chars) = truncate_text_chars("abcdef", 12);
        assert_eq!(full, "abcdef");
        assert!(!full_truncated);
        assert_eq!(full_chars, 6);

        let (clipped, clipped_truncated, clipped_chars) = truncate_text_chars("abcdef", 3);
        assert_eq!(clipped, "abc");
        assert!(clipped_truncated);
        assert_eq!(clipped_chars, 6);
    }
}

fn source_chunk_by_id(conn: &Connection, chunk_id: i64) -> Result<Option<SourceChunk>, String> {
    conn.query_row(
        r#"
SELECT
    pc.id,
    pc.chunk_index,
    p.path AS project_path,
    pc.doc_path,
    pc.doc_rel_path,
    pc.text
FROM project_chunks pc
JOIN projects p ON p.id = pc.project_id
WHERE pc.id = ?1
"#,
        params![chunk_id],
        |row| {
            Ok(SourceChunk {
                chunk_id: row.get(0)?,
                chunk_index: row.get(1)?,
                project_path: row.get(2)?,
                doc_path: row.get(3)?,
                doc_rel_path: row.get(4)?,
                text: row.get(5)?,
            })
        },
    )
    .optional()
    .map_err(|e| format!("failed loading source chunk {}: {}", chunk_id, e))
}

fn indexed_chunk_by_id(
    conn: &Connection,
    chunk_id: i64,
) -> Result<Option<IndexedChunkRow>, String> {
    conn.query_row(
        r#"
SELECT
    pc.id,
    pc.chunk_index,
    p.path AS project_path,
    pc.doc_path,
    pc.doc_rel_path,
    pc.doc_mtime,
    pc.token_count,
    pc.text
FROM project_chunks pc
JOIN projects p ON p.id = pc.project_id
WHERE pc.id = ?1
"#,
        params![chunk_id],
        |row| {
            Ok(IndexedChunkRow {
                chunk_id: row.get(0)?,
                chunk_index: row.get(1)?,
                project_path: row.get(2)?,
                doc_path: row.get(3)?,
                doc_rel_path: row.get(4)?,
                doc_mtime: row.get(5)?,
                token_count: row.get(6)?,
                text: row.get(7)?,
            })
        },
    )
    .optional()
    .map_err(|e| format!("failed loading chunk {}: {}", chunk_id, e))
}

fn indexed_doc_chunks_by_path(
    conn: &Connection,
    doc_path: &str,
) -> Result<Vec<IndexedChunkRow>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT
    pc.id,
    pc.chunk_index,
    p.path AS project_path,
    pc.doc_path,
    pc.doc_rel_path,
    pc.doc_mtime,
    pc.token_count,
    pc.text
FROM project_chunks pc
JOIN projects p ON p.id = pc.project_id
WHERE pc.doc_path = ?1
ORDER BY pc.chunk_index ASC
"#,
        )
        .map_err(|e| format!("failed preparing indexed doc chunks query: {}", e))?;
    let rows = stmt
        .query_map(params![doc_path], |row| {
            Ok(IndexedChunkRow {
                chunk_id: row.get(0)?,
                chunk_index: row.get(1)?,
                project_path: row.get(2)?,
                doc_path: row.get(3)?,
                doc_rel_path: row.get(4)?,
                doc_mtime: row.get(5)?,
                token_count: row.get(6)?,
                text: row.get(7)?,
            })
        })
        .map_err(|e| format!("failed querying indexed doc chunks: {}", e))?;
    let mut out: Vec<IndexedChunkRow> = Vec::new();
    for row in rows {
        out.push(row.map_err(|e| format!("failed reading indexed doc chunk row: {}", e))?);
    }
    Ok(out)
}

fn truncate_text_chars(text: &str, max_chars: usize) -> (String, bool, usize) {
    let total_chars = text.chars().count();
    if total_chars <= max_chars {
        return (text.to_string(), false, total_chars);
    }
    let clipped: String = text.chars().take(max_chars).collect();
    (clipped, true, total_chars)
}

fn project_neighbor_weights(
    conn: &Connection,
    path: &str,
    limit: usize,
) -> Result<HashMap<String, f64>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT path, MAX(weight) AS w
FROM (
    SELECT pe.dst AS path, pe.weight AS weight
    FROM project_edges pe
    JOIN projects src ON src.id = pe.src_project_id
    WHERE src.path = ?1
    UNION ALL
    SELECT src.path AS path, pe.weight AS weight
    FROM project_edges pe
    JOIN projects src ON src.id = pe.src_project_id
    WHERE pe.dst = ?1
)
GROUP BY path
ORDER BY w DESC, path ASC
LIMIT ?2
"#,
        )
        .map_err(|e| format!("failed preparing project neighbor weights query: {}", e))?;
    let rows = stmt
        .query_map(params![path, limit as i64], |row| {
            let p: String = row.get(0)?;
            let w: f64 = row.get(1)?;
            Ok((p, w))
        })
        .map_err(|e| format!("failed querying project neighbor weights: {}", e))?;
    let mut raw: Vec<(String, f64)> = Vec::new();
    for row in rows {
        raw.push(row.map_err(|e| format!("failed reading project neighbor weight row: {}", e))?);
    }
    if raw.is_empty() {
        return Ok(HashMap::new());
    }
    let hi = raw.iter().map(|(_, w)| *w).fold(0.0f64, f64::max);
    if hi <= 0.0 {
        return Ok(raw.into_iter().map(|(p, _)| (p, 0.0)).collect());
    }
    Ok(raw.into_iter().map(|(p, w)| (p, w / hi)).collect())
}

fn related_chunks_native(
    conn: &Connection,
    cfg: &ConfigValues,
    chunk_id: i64,
    limit: usize,
) -> Result<(SourceChunk, Vec<RelatedChunkResult>), String> {
    let source = source_chunk_by_id(conn, chunk_id)?
        .ok_or_else(|| format!("chunk {} was not found", chunk_id))?;
    let query_text: String = source.text.chars().take(2400).collect();
    let candidate_limit = std::cmp::max(40, limit.max(1) * 8).min(400);
    let ranked = rank_chunks_native(conn, cfg, &query_text, candidate_limit)?;
    let neighbor_weights = project_neighbor_weights(conn, &source.project_path, 120)?;
    let suppressed = suppressed_relation_set(conn, source.chunk_id)?;
    let quality_feedback = active_relation_quality_map(conn, source.chunk_id)?;

    let mut out: Vec<RelatedChunkResult> = Vec::new();
    for row in ranked {
        if row.chunk_id == source.chunk_id {
            continue;
        }
        let (relation, relation_weight) = if row.path == source.doc_path {
            ("same_file".to_string(), 1.0)
        } else if row.project_path == source.project_path {
            ("same_project".to_string(), 0.82)
        } else if let Some(weight) = neighbor_weights.get(&row.project_path) {
            (
                "project_edge".to_string(),
                (0.55 + (0.45 * *weight)).clamp(0.0, 1.0),
            )
        } else {
            continue;
        };
        if suppressed.contains(&(row.chunk_id, relation.clone())) {
            continue;
        }
        let relation_quality = quality_feedback
            .get(&(row.chunk_id, relation.clone()))
            .cloned()
            .unwrap_or_else(|| "unspecified".to_string());
        let relation_quality_weight = relation_quality_multiplier(cfg, &relation_quality);
        let score = ((0.72 * row.score) + (0.28 * relation_weight)) * relation_quality_weight;
        out.push(RelatedChunkResult {
            chunk_id: row.chunk_id,
            chunk_index: row.chunk_index,
            path: row.path,
            project_path: row.project_path,
            doc_rel_path: row.doc_rel_path,
            relation,
            relation_weight,
            relation_quality,
            relation_quality_multiplier: relation_quality_weight,
            score,
            semantic: row.semantic,
            lexical: row.lexical,
            quality: row.quality,
            excerpt: row.excerpt,
        });
    }
    if out.is_empty() {
        return Ok((source, Vec::new()));
    }
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    out.truncate(limit.max(1));
    Ok((source, out))
}

fn list_project_paths(conn: &Connection) -> Result<Vec<String>, String> {
    let mut stmt = conn
        .prepare("SELECT path FROM projects ORDER BY path")
        .map_err(|e| format!("failed preparing project list query: {}", e))?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|e| format!("failed querying project paths: {}", e))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(|e| format!("failed reading project path row: {}", e))?);
    }
    Ok(out)
}

fn project_semantic_scores(
    conn: &Connection,
    model: &str,
    query_vector: &[f32],
    keep_top: usize,
) -> Result<HashMap<String, f64>, String> {
    let qnorm = vector_norm(query_vector);
    if qnorm == 0.0 {
        return Ok(HashMap::new());
    }
    let mut stmt = conn
        .prepare(
            r#"
SELECT p.path, pv.norm, pv.vector
FROM projects p
JOIN project_vectors pv ON pv.project_id = p.id
WHERE pv.model = ?1
"#,
        )
        .map_err(|e| format!("failed preparing project vector query: {}", e))?;
    let rows = stmt
        .query_map(params![model], |row| {
            let path: String = row.get(0)?;
            let norm: f64 = row.get(1)?;
            let blob: Vec<u8> = row.get(2)?;
            Ok((path, norm, blob))
        })
        .map_err(|e| format!("failed querying project vectors: {}", e))?;

    let mut scores: Vec<(f64, String)> = Vec::new();
    for row in rows {
        let (path, vnorm, blob) =
            row.map_err(|e| format!("failed reading project vector row: {}", e))?;
        if vnorm == 0.0 {
            continue;
        }
        let vec = blob_to_f32_vec(&blob);
        let sim = ((cosine_raw(query_vector, &vec, qnorm, vnorm) + 1.0) / 2.0).clamp(0.0, 1.0);
        scores.push((sim, path));
    }
    scores.sort_by(|a, b| b.0.total_cmp(&a.0));
    if scores.is_empty() {
        return Ok(HashMap::new());
    }
    scores.truncate(keep_top.max(1));
    let lo = scores.iter().map(|v| v.0).fold(f64::INFINITY, f64::min);
    let hi = scores.iter().map(|v| v.0).fold(f64::NEG_INFINITY, f64::max);
    if (hi - lo).abs() < f64::EPSILON {
        let mut out = HashMap::new();
        for (_, path) in scores {
            out.insert(path, 1.0);
        }
        return Ok(out);
    }
    let span = hi - lo;
    let mut out = HashMap::new();
    for (score, path) in scores {
        out.insert(path, ((score - lo) / span).clamp(0.0, 1.0));
    }
    Ok(out)
}

fn search_lexical_chunks_sqlite(
    conn: &Connection,
    query: &str,
    limit: usize,
) -> Result<HashMap<i64, ChunkSignal>, String> {
    let fts = fts_query(query);
    if fts.is_empty() {
        return Ok(HashMap::new());
    }
    let mut stmt = conn
        .prepare(
            r#"
SELECT
    pc.id AS chunk_id,
    p.path AS project_path,
    pc.doc_path,
    pc.doc_rel_path,
    pc.chunk_index,
    pc.text,
    bm25(chunk_fts) AS lexical_bm25
FROM chunk_fts
JOIN project_chunks pc ON pc.id = chunk_fts.rowid
JOIN projects p ON p.id = pc.project_id
WHERE chunk_fts MATCH ?1
ORDER BY lexical_bm25
LIMIT ?2
"#,
        )
        .map_err(|e| format!("failed preparing lexical chunk query: {}", e))?;
    let rows = stmt
        .query_map(params![fts, limit as i64], |row| {
            let chunk_id: i64 = row.get(0)?;
            let project_path: String = row.get(1)?;
            let doc_path: String = row.get(2)?;
            let doc_rel_path: String = row.get(3)?;
            let chunk_index: i64 = row.get(4)?;
            let text: String = row.get(5)?;
            let bm25: f64 = row.get(6)?;
            Ok((
                chunk_id,
                project_path,
                doc_path,
                doc_rel_path,
                chunk_index,
                text,
                bm25,
            ))
        })
        .map_err(|e| format!("failed querying lexical chunks: {}", e))?;

    let mut raw: Vec<(i64, String, String, String, i64, String, f64)> = Vec::new();
    for row in rows {
        raw.push(row.map_err(|e| format!("failed reading lexical chunk row: {}", e))?);
    }
    if raw.is_empty() {
        return Ok(HashMap::new());
    }
    let lo = raw.iter().map(|r| r.6).fold(f64::INFINITY, f64::min);
    let hi = raw.iter().map(|r| r.6).fold(f64::NEG_INFINITY, f64::max);
    let span = hi - lo;
    let mut out: HashMap<i64, ChunkSignal> = HashMap::new();
    for (chunk_id, project_path, doc_path, doc_rel_path, chunk_index, text, bm25) in raw {
        let lexical = if span.abs() < f64::EPSILON {
            1.0
        } else {
            1.0 - ((bm25 - lo) / span)
        };
        out.insert(
            chunk_id,
            ChunkSignal {
                chunk_id,
                chunk_index,
                project_path,
                doc_path,
                doc_rel_path: doc_rel_path.clone(),
                semantic: 0.0,
                lexical: lexical.clamp(0.0, 1.0),
                graph: 0.0,
                relation: "direct".to_string(),
                quality: content_quality(&doc_rel_path, &text),
                excerpt: clip_text(&text, 190),
            },
        );
    }
    Ok(out)
}

fn keyword_path_chunk_scores(
    conn: &Connection,
    query: &str,
    keep_top: usize,
) -> Result<HashMap<i64, ChunkSignal>, String> {
    let q_tokens: Vec<String> = all_word_tokens(query)
        .into_iter()
        .filter(|t| t.len() >= 2)
        .collect();
    if q_tokens.is_empty() {
        return Ok(HashMap::new());
    }
    let mut out: HashMap<i64, ChunkSignal> = HashMap::new();
    let per_token_limit = (keep_top.max(1) * 3).clamp(50, 2500);
    let mut stmt = conn
        .prepare(
            r#"
SELECT
    pc.id AS chunk_id,
    p.path AS project_path,
    pc.doc_path,
    pc.doc_rel_path,
    pc.chunk_index,
    pc.text
FROM project_chunks pc
JOIN projects p ON p.id = pc.project_id
WHERE lower(pc.doc_path) LIKE ?1
   OR lower(pc.doc_rel_path) LIKE ?2
   OR lower(p.path) LIKE ?3
LIMIT ?4
"#,
        )
        .map_err(|e| format!("failed preparing keyword path chunk query: {}", e))?;

    let q_n = q_tokens.len() as f64;
    for token in &q_tokens {
        let pattern = format!("%{}%", token.to_lowercase());
        let rows = stmt
            .query_map(
                params![pattern, pattern, pattern, per_token_limit as i64],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, i64>(4)?,
                        row.get::<_, String>(5)?,
                    ))
                },
            )
            .map_err(|e| format!("failed querying keyword path chunks: {}", e))?;
        for row in rows {
            let (chunk_id, project_path, doc_path, doc_rel_path, chunk_index, text) =
                row.map_err(|e| format!("failed reading keyword path chunk row: {}", e))?;
            let doc_rel_lower = doc_rel_path.to_lowercase();
            let project_lower = project_path.to_lowercase();
            let doc_hits = q_tokens
                .iter()
                .filter(|tok| doc_rel_lower.contains(tok.as_str()))
                .count() as f64;
            let proj_hits = q_tokens
                .iter()
                .filter(|tok| project_lower.contains(tok.as_str()))
                .count() as f64;
            let lexical = ((doc_hits / q_n).max(0.65 * (proj_hits / q_n))).clamp(0.0, 1.0);
            if lexical <= 0.0 {
                continue;
            }
            let entry = out.entry(chunk_id).or_insert_with(|| ChunkSignal {
                chunk_id,
                chunk_index,
                project_path: project_path.clone(),
                doc_path: doc_path.clone(),
                doc_rel_path: doc_rel_path.clone(),
                semantic: 0.0,
                lexical,
                graph: 0.0,
                relation: "path_keyword".to_string(),
                quality: content_quality(&doc_rel_path, &text),
                excerpt: clip_text(&text, 190),
            });
            if lexical > entry.lexical {
                entry.lexical = lexical;
                entry.relation = "path_keyword".to_string();
            }
        }
    }

    if out.len() <= keep_top.max(1) {
        return Ok(out);
    }
    let mut pairs: Vec<(i64, ChunkSignal)> = out.into_iter().collect();
    pairs.sort_by(|a, b| b.1.lexical.total_cmp(&a.1.lexical));
    pairs.truncate(keep_top.max(1));
    Ok(pairs.into_iter().collect())
}

fn semantic_chunk_scores(
    conn: &Connection,
    model: &str,
    query_vector: &[f32],
    keep_top: usize,
) -> Result<HashMap<i64, ChunkSignal>, String> {
    let qnorm = vector_norm(query_vector);
    if qnorm == 0.0 {
        return Ok(HashMap::new());
    }
    let mut stmt = conn
        .prepare(
            r#"
SELECT
    pc.id AS chunk_id,
    p.path AS project_path,
    pc.doc_path,
    pc.doc_rel_path,
    pc.chunk_index,
    pc.text,
    pcv.norm,
    pcv.vector
FROM project_chunks pc
JOIN projects p ON p.id = pc.project_id
JOIN project_chunk_vectors pcv ON pcv.chunk_id = pc.id
WHERE pcv.model = ?1
"#,
        )
        .map_err(|e| format!("failed preparing semantic chunk query: {}", e))?;
    let rows = stmt
        .query_map(params![model], |row| {
            let chunk_id: i64 = row.get(0)?;
            let project_path: String = row.get(1)?;
            let doc_path: String = row.get(2)?;
            let doc_rel_path: String = row.get(3)?;
            let chunk_index: i64 = row.get(4)?;
            let text: String = row.get(5)?;
            let norm: f64 = row.get(6)?;
            let blob: Vec<u8> = row.get(7)?;
            Ok((
                chunk_id,
                project_path,
                doc_path,
                doc_rel_path,
                chunk_index,
                text,
                norm,
                blob,
            ))
        })
        .map_err(|e| format!("failed querying semantic chunks: {}", e))?;

    let mut scored: Vec<(f64, i64, String, String, String, i64, String)> = Vec::new();
    for row in rows {
        let (chunk_id, project_path, doc_path, doc_rel_path, chunk_index, text, vnorm, blob) =
            row.map_err(|e| format!("failed reading semantic chunk row: {}", e))?;
        if vnorm == 0.0 {
            continue;
        }
        let vec = blob_to_f32_vec(&blob);
        let sim = ((cosine_raw(query_vector, &vec, qnorm, vnorm) + 1.0) / 2.0).clamp(0.0, 1.0);
        scored.push((
            sim,
            chunk_id,
            project_path,
            doc_path,
            doc_rel_path,
            chunk_index,
            text,
        ));
    }
    if scored.is_empty() {
        return Ok(HashMap::new());
    }
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    scored.truncate(keep_top.max(1));
    let lo = scored.iter().map(|r| r.0).fold(f64::INFINITY, f64::min);
    let hi = scored.iter().map(|r| r.0).fold(f64::NEG_INFINITY, f64::max);
    let span = hi - lo;
    let mut out = HashMap::new();
    for (score, chunk_id, project_path, doc_path, doc_rel_path, chunk_index, text) in scored {
        let semantic = if span.abs() < f64::EPSILON {
            1.0
        } else {
            (score - lo) / span
        };
        out.insert(
            chunk_id,
            ChunkSignal {
                chunk_id,
                chunk_index,
                project_path,
                doc_path,
                doc_rel_path: doc_rel_path.clone(),
                semantic: semantic.clamp(0.0, 1.0),
                lexical: 0.0,
                graph: 0.0,
                relation: "direct".to_string(),
                quality: content_quality(&doc_rel_path, &text),
                excerpt: clip_text(&text, 190),
            },
        );
    }
    Ok(out)
}

fn fuse_chunk_signals(
    lexical_chunks: &HashMap<i64, ChunkSignal>,
    semantic_chunks: &HashMap<i64, ChunkSignal>,
) -> HashMap<i64, ChunkSignal> {
    let mut out = semantic_chunks.clone();
    for (chunk_id, lex) in lexical_chunks {
        if let Some(existing) = out.get_mut(chunk_id) {
            existing.lexical = existing.lexical.max(lex.lexical);
            existing.quality = existing.quality.max(lex.quality);
            if existing.excerpt.is_empty() && !lex.excerpt.is_empty() {
                existing.excerpt = lex.excerpt.clone();
            }
        } else {
            out.insert(*chunk_id, lex.clone());
        }
    }
    out
}

fn chunk_base_score(row: &ChunkSignal, cfg: &ConfigValues) -> f64 {
    // Graph is treated as contextual support with hop-penalty baked into row.graph.
    let direct = (cfg.rank_chunk_semantic_weight * row.semantic)
        + (cfg.rank_chunk_lexical_weight * row.lexical)
        + (cfg.rank_chunk_graph_weight * row.graph);
    let quality_mix = cfg.rank_quality_mix.clamp(0.0, 1.0);
    direct * ((1.0 - quality_mix) + (quality_mix * row.quality))
}

fn apply_graph_chunk_expansion(
    conn: &Connection,
    fused: &mut HashMap<i64, ChunkSignal>,
    cfg: &ConfigValues,
) -> Result<(), String> {
    if fused.is_empty() {
        return Ok(());
    }
    let seed_limit = cfg.graph_seed_limit.max(1) as usize;
    let neighbor_limit = cfg.graph_neighbor_limit.max(1) as usize;

    let mut seeds: Vec<(i64, f64, String)> = fused
        .values()
        .map(|row| {
            let seed_score = (0.78 * row.semantic) + (0.22 * row.lexical);
            (row.chunk_id, seed_score, row.project_path.clone())
        })
        .collect();
    seeds.sort_by(|a, b| b.1.total_cmp(&a.1));
    seeds.truncate(seed_limit.max(1));
    if seeds.is_empty() {
        return Ok(());
    }

    let mut seed_projects: HashSet<String> = HashSet::new();
    for (chunk_id, _, project_path) in &seeds {
        seed_projects.insert(project_path.clone());
        if let Some(row) = fused.get_mut(chunk_id) {
            row.graph = row.graph.max(1.0);
            row.relation = "seed".to_string();
        }
    }

    let mut neighbor_weights: HashMap<String, f64> = HashMap::new();
    for project_path in &seed_projects {
        let neighbors = project_neighbor_weights(conn, project_path, neighbor_limit.max(1))?;
        for (dst, weight) in neighbors {
            if dst == *project_path {
                continue;
            }
            let hop_penalized = (cfg.graph_related_base + (cfg.graph_related_scale * weight))
                .clamp(0.0, cfg.graph_related_cap.clamp(0.0, 1.0));
            let entry = neighbor_weights.entry(dst).or_insert(0.0);
            if hop_penalized > *entry {
                *entry = hop_penalized;
            }
        }
    }

    for row in fused.values_mut() {
        if row.relation == "seed" {
            continue;
        }
        if seed_projects.contains(&row.project_path) {
            let same_project = if row.semantic >= 0.72 {
                cfg.graph_same_project_high
            } else {
                cfg.graph_same_project_low
            };
            if same_project > row.graph {
                row.graph = same_project;
                row.relation = "same_project".to_string();
            }
            continue;
        }
        if let Some(weight) = neighbor_weights.get(&row.project_path) {
            if *weight > row.graph {
                row.graph = *weight;
                row.relation = "related_project".to_string();
            }
        }
    }

    Ok(())
}

fn evidence_hit_from_chunk(row: &ChunkSignal, score: f64) -> EvidenceHit {
    EvidenceHit {
        chunk_id: row.chunk_id,
        chunk_index: row.chunk_index,
        doc_path: row.doc_path.clone(),
        doc_rel_path: row.doc_rel_path.clone(),
        score,
        semantic: row.semantic,
        lexical: row.lexical,
        graph: row.graph,
        relation: row.relation.clone(),
        quality: row.quality,
        excerpt: row.excerpt.clone(),
    }
}

fn chunk_signals_for_ids(
    conn: &Connection,
    semantic_scores: &HashMap<i64, f64>,
    lexical_scores: &HashMap<i64, f64>,
) -> Result<HashMap<i64, ChunkSignal>, String> {
    let mut all_ids: Vec<i64> = semantic_scores
        .keys()
        .chain(lexical_scores.keys())
        .copied()
        .collect();
    if all_ids.is_empty() {
        return Ok(HashMap::new());
    }
    all_ids.sort_unstable();
    all_ids.dedup();

    let mut out: HashMap<i64, ChunkSignal> = HashMap::new();
    for batch in all_ids.chunks(300) {
        let placeholders = std::iter::repeat("?")
            .take(batch.len())
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            r#"
SELECT
    pc.id AS chunk_id,
    p.path AS project_path,
    pc.doc_path,
    pc.doc_rel_path,
    pc.chunk_index,
    pc.text
FROM project_chunks pc
JOIN projects p ON p.id = pc.project_id
WHERE pc.id IN ({})
"#,
            placeholders
        );
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| format!("failed preparing falkordb chunk lookup: {}", e))?;
        let rows = stmt
            .query_map(params_from_iter(batch.iter().copied()), |row| {
                let chunk_id: i64 = row.get(0)?;
                let project_path: String = row.get(1)?;
                let doc_path: String = row.get(2)?;
                let doc_rel_path: String = row.get(3)?;
                let chunk_index: i64 = row.get(4)?;
                let text: String = row.get(5)?;
                Ok((
                    chunk_id,
                    project_path,
                    doc_path,
                    doc_rel_path,
                    chunk_index,
                    text,
                ))
            })
            .map_err(|e| format!("failed querying falkordb chunk lookup: {}", e))?;
        for row in rows {
            let (chunk_id, project_path, doc_path, doc_rel_path, chunk_index, text) =
                row.map_err(|e| format!("failed reading falkordb chunk lookup row: {}", e))?;
            out.insert(
                chunk_id,
                ChunkSignal {
                    chunk_id,
                    chunk_index,
                    project_path,
                    doc_path,
                    doc_rel_path: doc_rel_path.clone(),
                    semantic: *semantic_scores.get(&chunk_id).unwrap_or(&0.0),
                    lexical: *lexical_scores.get(&chunk_id).unwrap_or(&0.0),
                    graph: 0.0,
                    relation: "direct".to_string(),
                    quality: content_quality(&doc_rel_path, &text),
                    excerpt: clip_text(&text, 190),
                },
            );
        }
    }
    Ok(out)
}

fn project_evidence(
    fused_chunks: &HashMap<i64, ChunkSignal>,
    cfg: &ConfigValues,
) -> HashMap<String, Vec<EvidenceHit>> {
    let mut by_project: HashMap<String, Vec<&ChunkSignal>> = HashMap::new();
    for chunk in fused_chunks.values() {
        by_project
            .entry(chunk.project_path.clone())
            .or_default()
            .push(chunk);
    }

    let mut out: HashMap<String, Vec<EvidenceHit>> = HashMap::new();
    for (project_path, rows) in by_project {
        let mut hits: Vec<EvidenceHit> = Vec::new();
        for row in rows {
            hits.push(evidence_hit_from_chunk(row, chunk_base_score(row, cfg)));
        }
        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| b.graph.total_cmp(&a.graph))
                .then_with(|| b.chunk_id.cmp(&a.chunk_id))
        });
        hits.truncate(6);
        out.insert(project_path, hits);
    }
    out
}

fn project_content_scores(
    project_evidence: &HashMap<String, Vec<EvidenceHit>>,
) -> HashMap<String, f64> {
    let mut out = HashMap::new();
    for (project_path, evidence) in project_evidence {
        if evidence.is_empty() {
            continue;
        }
        let top = evidence.iter().map(|e| e.score).fold(0.0f64, f64::max);
        let n = evidence.len().min(3);
        let mean_top = evidence.iter().take(n).map(|e| e.score).sum::<f64>() / n as f64;
        out.insert(project_path.clone(), (0.65 * top) + (0.35 * mean_top));
    }
    out
}

fn fts_query(query: &str) -> String {
    let mut tokens = all_word_tokens(query);
    if tokens.is_empty() {
        return String::new();
    }
    if tokens.len() > 12 {
        tokens.truncate(12);
    }
    tokens
        .into_iter()
        .map(|t| format!("{}*", t))
        .collect::<Vec<_>>()
        .join(" ")
}

fn frecency_scores(conn: &Connection) -> Result<HashMap<String, f64>, String> {
    let now = now_ts();
    let since = now - (120.0 * 86400.0);
    let mut stmt = conn
        .prepare(
            r#"
SELECT path, selected_at
FROM selection_events
WHERE selected_at >= ?1
"#,
        )
        .map_err(|e| format!("failed preparing selection events query: {}", e))?;
    let rows = stmt
        .query_map(params![since], |row| {
            let path: String = row.get(0)?;
            let selected_at: f64 = row.get(1)?;
            Ok((path, selected_at))
        })
        .map_err(|e| format!("failed querying selection events: {}", e))?;

    let mut raw: HashMap<String, f64> = HashMap::new();
    for row in rows {
        let (path, selected_at) =
            row.map_err(|e| format!("failed reading selection event row: {}", e))?;
        let age_days = ((now - selected_at) / 86400.0).max(0.0);
        let weight = (-age_days / 14.0).exp();
        *raw.entry(path).or_insert(0.0) += weight;
    }
    if raw.is_empty() {
        return Ok(HashMap::new());
    }
    let hi = raw.values().fold(0.0f64, |acc, v| acc.max(*v));
    if hi <= 0.0 {
        return Ok(HashMap::new());
    }
    Ok(raw.into_iter().map(|(k, v)| (k, v / hi)).collect())
}

fn rank_by_frecency_only(conn: &Connection, limit: usize) -> Result<Vec<RankedResult>, String> {
    let frecency = frecency_scores(conn)?;
    let graph = graph_scores(conn)?;
    let mut stmt = conn
        .prepare("SELECT path FROM projects ORDER BY path")
        .map_err(|e| format!("failed preparing projects query: {}", e))?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|e| format!("failed querying projects: {}", e))?;
    let mut ranked: Vec<RankedResult> = Vec::new();
    for row in rows {
        let path = row.map_err(|e| format!("failed reading project row: {}", e))?;
        let fr = *frecency.get(&path).unwrap_or(&0.0);
        let gr = *graph.get(&path).unwrap_or(&0.0);
        ranked.push(RankedResult {
            path,
            score: (0.75 * fr) + (0.25 * gr),
            lexical: 0.0,
            semantic: 0.0,
            frecency: fr,
            graph: gr,
            evidence: Vec::new(),
        });
    }
    ranked.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| b.path.cmp(&a.path))
    });
    ranked.truncate(limit.max(1));
    Ok(ranked)
}

fn graph_scores(conn: &Connection) -> Result<HashMap<String, f64>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT dst AS path, SUM(weight) AS edge_score
FROM project_edges
GROUP BY dst
"#,
        )
        .map_err(|e| format!("failed preparing graph score query: {}", e))?;
    let rows = stmt
        .query_map([], |row| {
            let path: String = row.get(0)?;
            let edge_score: f64 = row.get(1)?;
            Ok((path, edge_score))
        })
        .map_err(|e| format!("failed querying graph scores: {}", e))?;
    let mut raw: Vec<(String, f64)> = Vec::new();
    for row in rows {
        raw.push(row.map_err(|e| format!("failed reading graph score row: {}", e))?);
    }
    if raw.is_empty() {
        return Ok(HashMap::new());
    }
    let hi = raw.iter().map(|r| r.1).fold(0.0f64, f64::max);
    if hi <= 0.0 {
        return Ok(HashMap::new());
    }
    Ok(raw.into_iter().map(|(p, s)| (p, s / hi)).collect())
}

fn clip_text(text: &str, max_chars: usize) -> String {
    let value = collapse_whitespace(text);
    if value.chars().count() <= max_chars {
        return value;
    }
    let mut clipped: String = value.chars().take(max_chars).collect();
    clipped = clipped.trim_end().to_string();
    clipped.push_str("...");
    clipped
}

fn content_quality(doc_rel_path: &str, text: &str) -> f64 {
    let n = text.chars().count().max(1) as f64;
    let letters = text.chars().filter(|ch| ch.is_alphabetic()).count() as f64;
    let punctuation_like = text
        .chars()
        .filter(|ch| matches!(ch, '{' | '}' | '[' | ']' | ',' | ':' | '"' | '\\' | '/'))
        .count() as f64;
    let escaped_newlines = text.matches("\\n").count() as f64;
    let tokens = all_word_tokens(text);
    let token_n = tokens.len();
    if token_n == 0 {
        return 0.1;
    }
    let long_tokens = tokens.iter().filter(|tok| tok.len() >= 24).count() as f64;
    let hex_tokens = tokens.iter().filter(|tok| is_hex_token(tok)).count() as f64;
    let unique_ratio = tokens.iter().collect::<HashSet<_>>().len() as f64 / token_n as f64;

    let mut score: f64 = 1.0;
    if token_n >= 80 {
        if (escaped_newlines / n) > 0.008 {
            score *= 0.68;
        }
        if (punctuation_like / n) > 0.24 && (letters / n) < 0.45 {
            score *= 0.70;
        }
        if (long_tokens / token_n as f64) > 0.20 {
            score *= 0.72;
        }
        if (hex_tokens / token_n as f64) > 0.08 {
            score *= 0.68;
        }
        if unique_ratio < 0.18 {
            score *= 0.78;
        }
    }

    let name = doc_rel_path.to_lowercase();
    if name.ends_with("package-lock.json")
        || name.ends_with("yarn.lock")
        || name.ends_with("pnpm-lock.yaml")
    {
        score *= 0.35;
    }
    if name.ends_with(".metadata.json") {
        score *= 0.70;
    }
    if name.contains("session_") || format!("/{}/", name).contains("/sessions/") {
        score *= 0.35;
    }
    if name.ends_with(".min.js") {
        score *= 0.55;
    }
    score.clamp(0.08, 1.0)
}

fn is_hex_token(token: &str) -> bool {
    if token.len() < 16 {
        return false;
    }
    token.chars().all(|ch| ch.is_ascii_hexdigit())
}

fn path_keyword_scores(paths: &[String], query: &str) -> HashMap<String, f64> {
    let q_tokens: Vec<String> = all_word_tokens(query)
        .into_iter()
        .filter(|t| t.len() >= 2)
        .collect();
    if q_tokens.is_empty() {
        return HashMap::new();
    }
    let q_set: HashSet<String> = q_tokens.into_iter().collect();
    let q_n = q_set.len() as f64;
    let mut out = HashMap::new();
    for path in paths {
        let name = Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();
        let n_tokens: HashSet<String> = all_word_tokens(&name).into_iter().collect();
        if n_tokens.is_empty() {
            continue;
        }
        let exact = q_set.intersection(&n_tokens).count() as f64 / q_n;
        let fuzzy_hits = q_set
            .iter()
            .filter(|tok| name.contains((*tok).as_str()))
            .count() as f64;
        let fuzzy = fuzzy_hits / q_n;
        let score = exact.max(0.8 * fuzzy).clamp(0.0, 1.0);
        if score > 0.0 {
            out.insert(path.clone(), score);
        }
    }
    out
}

fn doc_keyword_score(doc_rel_path: &str, query: &str) -> f64 {
    let q_tokens: Vec<String> = all_word_tokens(query)
        .into_iter()
        .filter(|t| t.len() >= 2)
        .collect();
    if q_tokens.is_empty() {
        return 0.0;
    }
    let rel = doc_rel_path.to_lowercase();
    let rel_tokens: HashSet<String> = all_word_tokens(&rel).into_iter().collect();
    let q_set: HashSet<String> = q_tokens.into_iter().collect();
    let q_n = q_set.len() as f64;
    let exact_hits = q_set.intersection(&rel_tokens).count() as f64;
    let fuzzy_hits = q_set
        .iter()
        .filter(|token| rel.contains((*token).as_str()))
        .count() as f64;
    let exact = exact_hits / q_n;
    let fuzzy = fuzzy_hits / q_n;
    exact.max(0.8 * fuzzy).clamp(0.0, 1.0)
}

fn is_keyword_focused_query(query: &str) -> bool {
    let q_tokens: Vec<String> = all_word_tokens(query)
        .into_iter()
        .filter(|t| t.len() >= 2)
        .collect();
    if q_tokens.is_empty() {
        return false;
    }
    q_tokens.len() <= 2 && q_tokens.iter().all(|t| t.len() <= 32)
}

fn path_noise_penalty(doc_rel_path: &str) -> f64 {
    let p = doc_rel_path.to_lowercase();
    let mut penalty = 1.0f64;
    if p.starts_with("tmp/") || p.contains("/tmp/") {
        penalty *= 0.55;
    }
    if p.starts_with("state/") || p.contains("/state/") {
        penalty *= 0.72;
    }
    if p.starts_with("archived/") || p.contains("/archived/") {
        penalty *= 0.82;
    }
    if p.ends_with(".json") {
        penalty *= 0.92;
    }
    penalty.clamp(0.25, 1.0)
}

fn is_generic_container(path: &str) -> bool {
    let name = Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    matches!(
        name.as_str(),
        "archive"
            | "archived"
            | "archives"
            | "misc"
            | "tmp"
            | "temp"
            | "scratch"
            | "old"
            | "backup"
            | "backups"
    )
}

fn cosine_raw(a: &[f32], b: &[f32], anorm: f64, bnorm: f64) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0.0f64;
    for i in 0..n {
        dot += (a[i] as f64) * (b[i] as f64);
    }
    let denom = anorm * bnorm;
    if denom == 0.0 {
        return 0.0;
    }
    dot / denom
}

fn blob_to_f32_vec(blob: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(blob.len() / 4);
    for chunk in blob.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

fn all_word_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            cur.push(ch.to_ascii_lowercase());
        } else if !cur.is_empty() {
            out.push(cur.clone());
            cur.clear();
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn run_legacy_cmd(args: &[OsString]) -> ! {
    let wants_help = args.len() == 1 && (args[0] == "-h" || args[0] == "--help");
    if args.is_empty() || wants_help {
        eprintln!("usage: cypress legacy <args...>");
        eprintln!("example: cypress legacy install --bench");
        process::exit(2);
    }
    process::exit(run_legacy_bridge_status(args));
}

fn run_legacy_bridge_status(args: &[OsString]) -> i32 {
    let repo = find_repo_root()
        .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let python = env::var("CYPRESS_PYTHON").unwrap_or_else(|_| "python3".to_string());

    let mut cmd = Command::new(&python);
    cmd.arg("-m").arg("semcd.cli");
    cmd.args(args);
    cmd.stdin(process::Stdio::inherit());
    cmd.stdout(process::Stdio::inherit());
    cmd.stderr(process::Stdio::inherit());

    let src_dir = repo.join("src");
    let mut py_path = src_dir.to_string_lossy().to_string();
    if let Ok(existing) = env::var("PYTHONPATH") {
        if !existing.trim().is_empty() {
            py_path.push(':');
            py_path.push_str(&existing);
        }
    }
    cmd.env("PYTHONPATH", py_path);
    if env::var("CYPRESS_CONFIG").is_err() && env::var("SEMCD_CONFIG").is_err() {
        cmd.env(
            "CYPRESS_CONFIG",
            config_path(&cwd).to_string_lossy().to_string(),
        );
    }

    let status = cmd.status().unwrap_or_else(|err| {
        eprintln!(
            "error: failed to run legacy bridge via '{}': {}",
            python, err
        );
        process::exit(1);
    });
    status.code().unwrap_or(1)
}

#[derive(Clone, Debug)]
struct ConfigValues {
    root: PathBuf,
    embed_backend: String,
    embed_model: String,
    retrieval_backend: String,
    falkor_runtime_mode: String,
    falkor_runtime_cmd: String,
    falkor_docker_container: String,
    local_embed_dim: i64,
    max_chars_per_project: i64,
    lexical_candidates: i64,
    vector_candidates: i64,
    rank_chunk_semantic_weight: f64,
    rank_chunk_lexical_weight: f64,
    rank_chunk_graph_weight: f64,
    rank_quality_mix: f64,
    rank_relation_quality_good_boost: f64,
    rank_relation_quality_weak_penalty: f64,
    rank_relation_quality_wrong_penalty: f64,
    rank_project_content_weight: f64,
    rank_project_semantic_weight: f64,
    rank_project_path_weight: f64,
    rank_project_graph_weight: f64,
    rank_project_frecency_weight: f64,
    graph_seed_limit: i64,
    graph_neighbor_limit: i64,
    graph_same_project_high: f64,
    graph_same_project_low: f64,
    graph_related_base: f64,
    graph_related_scale: f64,
    graph_related_cap: f64,
}

impl ConfigValues {
    fn from_map(map: std::collections::HashMap<String, String>) -> Self {
        let mut embed_backend = map
            .get("embed_backend")
            .cloned()
            .unwrap_or_else(|| "local-hash".to_string())
            .trim()
            .to_lowercase();
        if embed_backend == "auto" {
            embed_backend = "local-hash".to_string();
        }
        if !matches!(embed_backend.as_str(), "local" | "local-hash" | "ollama") {
            embed_backend = "local-hash".to_string();
        }

        let mut retrieval_backend = map
            .get("retrieval_backend")
            .cloned()
            .unwrap_or_else(|| "falkordb".to_string())
            .trim()
            .to_lowercase();
        if retrieval_backend == "lancedb" {
            retrieval_backend = "falkordb".to_string();
        }
        if !matches!(retrieval_backend.as_str(), "falkordb" | "sqlite") {
            retrieval_backend = "falkordb".to_string();
        }

        let mut embed_model = map
            .get("embed_model")
            .cloned()
            .unwrap_or_else(|| "sentence-transformers/all-MiniLM-L6-v2".to_string());
        if (embed_backend == "local" || embed_backend == "local-hash")
            && (embed_model.trim().is_empty() || embed_model == "qwen3-embedding")
        {
            embed_model = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        }
        if embed_backend == "ollama"
            && (embed_model.trim().is_empty()
                || embed_model == "sentence-transformers/all-MiniLM-L6-v2")
        {
            embed_model = "qwen3-embedding".to_string();
        }

        let local_embed_dim = map
            .get("local_embed_dim")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(384)
            .max(64);
        let mut falkor_runtime_mode = map
            .get("falkor_runtime_mode")
            .cloned()
            .unwrap_or_else(|| "managed-local".to_string())
            .trim()
            .to_lowercase();
        if !matches!(
            falkor_runtime_mode.as_str(),
            "managed-local" | "docker" | "manual"
        ) {
            falkor_runtime_mode = "managed-local".to_string();
        }
        let falkor_runtime_cmd = map
            .get("falkor_runtime_cmd")
            .cloned()
            .unwrap_or_default()
            .trim()
            .to_string();
        let falkor_docker_container = map
            .get("falkor_docker_container")
            .cloned()
            .unwrap_or_else(|| "cypress-falkordb".to_string())
            .trim()
            .to_string();
        let max_chars_per_project = map
            .get("max_chars_per_project")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(12000);
        let lexical_candidates = map
            .get("lexical_candidates")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(120);
        let vector_candidates = map
            .get("vector_candidates")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(120);
        let rank_chunk_semantic_weight = map
            .get("rank_chunk_semantic_weight")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.66)
            .clamp(0.0, 1.0);
        let rank_chunk_lexical_weight = map
            .get("rank_chunk_lexical_weight")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.24)
            .clamp(0.0, 1.0);
        let rank_chunk_graph_weight = map
            .get("rank_chunk_graph_weight")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.10)
            .clamp(0.0, 1.0);
        let rank_quality_mix = map
            .get("rank_quality_mix")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.70)
            .clamp(0.0, 1.0);
        let rank_relation_quality_good_boost = map
            .get("rank_relation_quality_good_boost")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.08)
            .clamp(0.0, 1.0);
        let rank_relation_quality_weak_penalty = map
            .get("rank_relation_quality_weak_penalty")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.20)
            .clamp(0.0, 1.0);
        let rank_relation_quality_wrong_penalty = map
            .get("rank_relation_quality_wrong_penalty")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.65)
            .clamp(0.0, 1.0);
        let rank_project_content_weight = map
            .get("rank_project_content_weight")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.58)
            .clamp(0.0, 2.0);
        let rank_project_semantic_weight = map
            .get("rank_project_semantic_weight")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.14)
            .clamp(0.0, 2.0);
        let rank_project_path_weight = map
            .get("rank_project_path_weight")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.10)
            .clamp(0.0, 2.0);
        let rank_project_graph_weight = map
            .get("rank_project_graph_weight")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.10)
            .clamp(0.0, 2.0);
        let rank_project_frecency_weight = map
            .get("rank_project_frecency_weight")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.08)
            .clamp(0.0, 2.0);
        let graph_seed_limit = map
            .get("graph_seed_limit")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(10)
            .clamp(2, 64);
        let graph_neighbor_limit = map
            .get("graph_neighbor_limit")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(100)
            .clamp(8, 500);
        let graph_same_project_high = map
            .get("graph_same_project_high")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.88)
            .clamp(0.0, 1.0);
        let graph_same_project_low = map
            .get("graph_same_project_low")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.76)
            .clamp(0.0, 1.0);
        let graph_related_base = map
            .get("graph_related_base")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.20)
            .clamp(0.0, 1.0);
        let graph_related_scale = map
            .get("graph_related_scale")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.70)
            .clamp(0.0, 2.0);
        let graph_related_cap = map
            .get("graph_related_cap")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.90)
            .clamp(0.0, 1.0);

        Self {
            root: map
                .get("root")
                .map(|v| normalize_path(v))
                .unwrap_or_else(|| normalize_path("~/c-projects")),
            embed_backend,
            embed_model,
            retrieval_backend,
            falkor_runtime_mode,
            falkor_runtime_cmd,
            falkor_docker_container,
            local_embed_dim,
            max_chars_per_project,
            lexical_candidates,
            vector_candidates,
            rank_chunk_semantic_weight,
            rank_chunk_lexical_weight,
            rank_chunk_graph_weight,
            rank_quality_mix,
            rank_relation_quality_good_boost,
            rank_relation_quality_weak_penalty,
            rank_relation_quality_wrong_penalty,
            rank_project_content_weight,
            rank_project_semantic_weight,
            rank_project_path_weight,
            rank_project_graph_weight,
            rank_project_frecency_weight,
            graph_seed_limit,
            graph_neighbor_limit,
            graph_same_project_high,
            graph_same_project_low,
            graph_related_base,
            graph_related_scale,
            graph_related_cap,
        }
    }
}

fn arg_value(args: &[OsString], index: usize, flag: &str) -> String {
    let Some(v) = args.get(index) else {
        eprintln!("error: {} expects a value", flag);
        process::exit(2);
    };
    v.to_string_lossy().to_string()
}

fn normalize_path(raw: &str) -> PathBuf {
    let mut path = expand_tilde(raw);
    if !path.is_absolute() {
        if let Ok(cwd) = env::current_dir() {
            path = cwd.join(path);
        }
    }
    if let Ok(canonical) = path.canonicalize() {
        return canonical;
    }
    normalize_lexical(&path)
}

fn normalize_lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                let _ = out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn write_config_file(path: &Path, cfg: &ConfigValues) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("failed creating config dir: {}", e))?;
    }
    let lines = [
        format!("root = \"{}\"", toml_escape(&cfg.root.to_string_lossy())),
        format!("embed_backend = \"{}\"", toml_escape(&cfg.embed_backend)),
        format!("embed_model = \"{}\"", toml_escape(&cfg.embed_model)),
        format!(
            "retrieval_backend = \"{}\"",
            toml_escape(&cfg.retrieval_backend)
        ),
        format!(
            "falkor_runtime_mode = \"{}\"",
            toml_escape(&cfg.falkor_runtime_mode)
        ),
        format!(
            "falkor_runtime_cmd = \"{}\"",
            toml_escape(&cfg.falkor_runtime_cmd)
        ),
        format!(
            "falkor_docker_container = \"{}\"",
            toml_escape(&cfg.falkor_docker_container)
        ),
        format!("local_embed_dim = {}", cfg.local_embed_dim),
        format!("max_chars_per_project = {}", cfg.max_chars_per_project),
        format!("lexical_candidates = {}", cfg.lexical_candidates),
        format!("vector_candidates = {}", cfg.vector_candidates),
        format!(
            "rank_chunk_semantic_weight = {:.6}",
            cfg.rank_chunk_semantic_weight
        ),
        format!(
            "rank_chunk_lexical_weight = {:.6}",
            cfg.rank_chunk_lexical_weight
        ),
        format!(
            "rank_chunk_graph_weight = {:.6}",
            cfg.rank_chunk_graph_weight
        ),
        format!("rank_quality_mix = {:.6}", cfg.rank_quality_mix),
        format!(
            "rank_relation_quality_good_boost = {:.6}",
            cfg.rank_relation_quality_good_boost
        ),
        format!(
            "rank_relation_quality_weak_penalty = {:.6}",
            cfg.rank_relation_quality_weak_penalty
        ),
        format!(
            "rank_relation_quality_wrong_penalty = {:.6}",
            cfg.rank_relation_quality_wrong_penalty
        ),
        format!(
            "rank_project_content_weight = {:.6}",
            cfg.rank_project_content_weight
        ),
        format!(
            "rank_project_semantic_weight = {:.6}",
            cfg.rank_project_semantic_weight
        ),
        format!(
            "rank_project_path_weight = {:.6}",
            cfg.rank_project_path_weight
        ),
        format!(
            "rank_project_graph_weight = {:.6}",
            cfg.rank_project_graph_weight
        ),
        format!(
            "rank_project_frecency_weight = {:.6}",
            cfg.rank_project_frecency_weight
        ),
        format!("graph_seed_limit = {}", cfg.graph_seed_limit),
        format!("graph_neighbor_limit = {}", cfg.graph_neighbor_limit),
        format!(
            "graph_same_project_high = {:.6}",
            cfg.graph_same_project_high
        ),
        format!("graph_same_project_low = {:.6}", cfg.graph_same_project_low),
        format!("graph_related_base = {:.6}", cfg.graph_related_base),
        format!("graph_related_scale = {:.6}", cfg.graph_related_scale),
        format!("graph_related_cap = {:.6}", cfg.graph_related_cap),
        String::new(),
    ];
    fs::write(path, lines.join("\n")).map_err(|e| format!("failed writing config: {}", e))
}

fn toml_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn ensure_db_schema(db_path: &Path) -> Result<(), String> {
    let _ = open_db_rw(db_path)?;
    Ok(())
}

fn ensure_tracked_root(db_path: &Path, path: &Path, added_at: f64) -> Result<(), String> {
    let conn = open_db_rw(db_path)?;
    ensure_tracked_root_conn(&conn, path, added_at)
}

fn remove_tracked_root(db_path: &Path, path: &Path) -> Result<i64, String> {
    let conn = open_db_rw(db_path)?;
    conn.execute(
        "DELETE FROM tracked_roots WHERE path = ?1",
        params![path.to_string_lossy().to_string()],
    )
    .map(|n| n as i64)
    .map_err(|e| format!("failed to remove tracked root: {}", e))
}

fn list_tracked_roots(db_path: &Path) -> Result<Vec<PathBuf>, String> {
    let conn = open_db_rw(db_path)?;
    list_tracked_roots_conn(&conn)
}

fn record_selection_event(
    conn: &Connection,
    query: &str,
    path: &str,
    selected_at: f64,
) -> Result<(), String> {
    conn.execute(
        r#"
INSERT INTO selection_events(query, path, selected_at)
VALUES (?1, ?2, ?3)
"#,
        params![query, path, selected_at],
    )
    .map_err(|e| format!("failed recording selection event: {}", e))?;
    Ok(())
}

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn open_db_rw(db_path: &Path) -> Result<Connection, String> {
    if let Some(parent) = db_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("failed creating db dir: {}", e))?;
    }
    let conn = Connection::open(db_path).map_err(|e| format!("failed opening database: {}", e))?;
    conn.execute_batch(
        r#"
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
"#,
    )
    .map_err(|e| format!("failed setting db pragmas: {}", e))?;
    init_schema(&conn)?;
    Ok(conn)
}

fn open_db_read_only(db_path: &Path) -> Result<Connection, String> {
    let conn = Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| format!("failed opening database readonly: {}", e))?;
    Ok(conn)
}

fn init_schema(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(
        r#"
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    project_mtime REAL NOT NULL,
    last_indexed REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS project_vectors (
    project_id INTEGER PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    norm REAL NOT NULL,
    vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS project_chunks (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    doc_path TEXT NOT NULL,
    doc_rel_path TEXT NOT NULL,
    doc_mtime REAL NOT NULL,
    chunk_index INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    text_hash TEXT NOT NULL,
    text TEXT NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(project_id, doc_path, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_project_chunks_project
    ON project_chunks(project_id);
CREATE INDEX IF NOT EXISTS idx_project_chunks_doc
    ON project_chunks(doc_path);

CREATE TABLE IF NOT EXISTS project_chunk_vectors (
    chunk_id INTEGER PRIMARY KEY REFERENCES project_chunks(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    norm REAL NOT NULL,
    vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS selection_events (
    id INTEGER PRIMARY KEY,
    query TEXT NOT NULL,
    path TEXT NOT NULL,
    selected_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS tracked_roots (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    added_at REAL NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1))
);

CREATE INDEX IF NOT EXISTS idx_tracked_roots_enabled_path
    ON tracked_roots(enabled, path);

CREATE TABLE IF NOT EXISTS project_edges (
    src_project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    dst TEXT NOT NULL,
    kind TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY(src_project_id, dst, kind)
);

CREATE INDEX IF NOT EXISTS idx_project_edges_dst
    ON project_edges(dst);
CREATE INDEX IF NOT EXISTS idx_project_edges_src_kind
    ON project_edges(src_project_id, kind);

CREATE TABLE IF NOT EXISTS chunk_relation_feedback (
    id INTEGER PRIMARY KEY,
    src_chunk_id INTEGER NOT NULL REFERENCES project_chunks(id) ON DELETE CASCADE,
    dst_chunk_id INTEGER NOT NULL REFERENCES project_chunks(id) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    decision TEXT NOT NULL CHECK(decision IN ('active', 'suppressed')),
    quality_label TEXT NOT NULL DEFAULT 'unspecified' CHECK(quality_label IN ('unspecified', 'good', 'weak', 'wrong')),
    note TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'user',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(src_chunk_id, dst_chunk_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_chunk_relation_feedback_src_decision_time
    ON chunk_relation_feedback(src_chunk_id, decision, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_chunk_relation_feedback_dst
    ON chunk_relation_feedback(dst_chunk_id);

CREATE INDEX IF NOT EXISTS idx_selection_events_path_time
    ON selection_events(path, selected_at DESC);
CREATE INDEX IF NOT EXISTS idx_selection_events_query_time
    ON selection_events(query, selected_at DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS project_fts USING fts5(
    path,
    title,
    summary,
    content='projects',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS projects_ai AFTER INSERT ON projects BEGIN
    INSERT INTO project_fts(rowid, path, title, summary)
    VALUES (new.id, new.path, new.title, new.summary);
END;

CREATE TRIGGER IF NOT EXISTS projects_ad AFTER DELETE ON projects BEGIN
    INSERT INTO project_fts(project_fts, rowid, path, title, summary)
    VALUES ('delete', old.id, old.path, old.title, old.summary);
END;

CREATE TRIGGER IF NOT EXISTS projects_au AFTER UPDATE ON projects BEGIN
    INSERT INTO project_fts(project_fts, rowid, path, title, summary)
    VALUES ('delete', old.id, old.path, old.title, old.summary);
    INSERT INTO project_fts(rowid, path, title, summary)
    VALUES (new.id, new.path, new.title, new.summary);
END;

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    doc_path,
    doc_rel_path,
    text,
    content='project_chunks',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS project_chunks_ai AFTER INSERT ON project_chunks BEGIN
    INSERT INTO chunk_fts(rowid, doc_path, doc_rel_path, text)
    VALUES (new.id, new.doc_path, new.doc_rel_path, new.text);
END;

CREATE TRIGGER IF NOT EXISTS project_chunks_ad AFTER DELETE ON project_chunks BEGIN
    INSERT INTO chunk_fts(chunk_fts, rowid, doc_path, doc_rel_path, text)
    VALUES ('delete', old.id, old.doc_path, old.doc_rel_path, old.text);
END;

CREATE TRIGGER IF NOT EXISTS project_chunks_au AFTER UPDATE ON project_chunks BEGIN
    INSERT INTO chunk_fts(chunk_fts, rowid, doc_path, doc_rel_path, text)
    VALUES ('delete', old.id, old.doc_path, old.doc_rel_path, old.text);
    INSERT INTO chunk_fts(rowid, doc_path, doc_rel_path, text)
    VALUES (new.id, new.doc_path, new.doc_rel_path, new.text);
END;
"#,
    )
    .map_err(|e| {
        format!(
            "failed initializing db schema (ensure sqlite build includes FTS5): {}",
            e
        )
    })?;
    ensure_vector_model_column(conn)?;
    ensure_relation_feedback_quality_column(conn)?;
    Ok(())
}

fn ensure_vector_model_column(conn: &Connection) -> Result<(), String> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(project_vectors)")
        .map_err(|e| format!("failed inspecting project_vectors schema: {}", e))?;
    let mut rows = stmt
        .query([])
        .map_err(|e| format!("failed reading project_vectors schema: {}", e))?;
    let mut has_model = false;
    while let Some(row) = rows
        .next()
        .map_err(|e| format!("failed iterating schema rows: {}", e))?
    {
        let name: String = row
            .get(1)
            .map_err(|e| format!("failed reading schema column name: {}", e))?;
        if name == "model" {
            has_model = true;
            break;
        }
    }
    if !has_model {
        conn.execute(
            "ALTER TABLE project_vectors ADD COLUMN model TEXT NOT NULL DEFAULT ''",
            [],
        )
        .map_err(|e| format!("failed migrating project_vectors.model: {}", e))?;
        conn.execute("UPDATE project_vectors SET model = '' WHERE model = ''", [])
            .map_err(|e| format!("failed finalizing project_vectors.model migration: {}", e))?;
    }
    Ok(())
}

fn ensure_relation_feedback_quality_column(conn: &Connection) -> Result<(), String> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(chunk_relation_feedback)")
        .map_err(|e| format!("failed inspecting chunk_relation_feedback schema: {}", e))?;
    let mut rows = stmt
        .query([])
        .map_err(|e| format!("failed reading chunk_relation_feedback schema: {}", e))?;
    let mut has_quality = false;
    while let Some(row) = rows
        .next()
        .map_err(|e| format!("failed iterating schema rows: {}", e))?
    {
        let name: String = row
            .get(1)
            .map_err(|e| format!("failed reading schema column name: {}", e))?;
        if name == "quality_label" {
            has_quality = true;
            break;
        }
    }
    if !has_quality {
        conn.execute(
            "ALTER TABLE chunk_relation_feedback ADD COLUMN quality_label TEXT NOT NULL DEFAULT 'unspecified'",
            [],
        )
        .map_err(|e| format!("failed migrating chunk_relation_feedback.quality_label: {}", e))?;
        conn.execute(
            "UPDATE chunk_relation_feedback SET quality_label = 'unspecified' WHERE quality_label = '' OR quality_label IS NULL",
            [],
        )
        .map_err(|e| {
            format!(
                "failed finalizing chunk_relation_feedback.quality_label migration: {}",
                e
            )
        })?;
    }
    conn.execute_batch(
        r#"
CREATE INDEX IF NOT EXISTS idx_chunk_relation_feedback_src_quality
    ON chunk_relation_feedback(src_chunk_id, quality_label, updated_at DESC);
"#,
    )
    .map_err(|e| {
        format!(
            "failed ensuring chunk_relation_feedback quality index: {}",
            e
        )
    })?;
    Ok(())
}

fn ensure_tracked_root_conn(conn: &Connection, path: &Path, added_at: f64) -> Result<(), String> {
    conn.execute(
        r#"
INSERT INTO tracked_roots(path, added_at, enabled)
VALUES (?1, ?2, 1)
ON CONFLICT(path) DO UPDATE SET enabled = 1
"#,
        params![path.to_string_lossy().to_string(), added_at],
    )
    .map_err(|e| format!("failed ensuring tracked root: {}", e))?;
    Ok(())
}

fn list_tracked_roots_conn(conn: &Connection) -> Result<Vec<PathBuf>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT path
FROM tracked_roots
WHERE enabled = 1
ORDER BY path
"#,
        )
        .map_err(|e| format!("failed preparing tracked roots query: {}", e))?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|e| format!("failed querying tracked roots: {}", e))?;

    let mut out = Vec::new();
    for row in rows {
        let p = row.map_err(|e| format!("failed reading tracked root row: {}", e))?;
        out.push(PathBuf::from(p));
    }
    Ok(out)
}

fn resolve_roots(
    conn: &Connection,
    cfg: &ConfigValues,
    scope_roots: Option<Vec<PathBuf>>,
) -> Result<Vec<PathBuf>, String> {
    if let Some(roots) = scope_roots {
        let mut out = Vec::new();
        for root in roots {
            let p = normalize_path(&root.to_string_lossy());
            if p.is_dir() {
                out.push(p);
            }
        }
        return Ok(out);
    }

    let mut rows = list_tracked_roots_conn(conn)?;
    if rows.is_empty() {
        ensure_tracked_root_conn(conn, &cfg.root, now_ts())?;
        rows = list_tracked_roots_conn(conn)?;
    }
    Ok(rows
        .into_iter()
        .map(|p| normalize_path(&p.to_string_lossy()))
        .collect())
}

fn discover_projects(roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<PathBuf> = Vec::new();

    for root in roots {
        if !root.is_dir() {
            continue;
        }
        let mut children: Vec<PathBuf> = Vec::new();
        if let Ok(rd) = fs::read_dir(root) {
            for entry in rd.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with('.') || is_skip_dir(&name) {
                    continue;
                }
                if let Ok(ft) = entry.file_type() {
                    if ft.is_dir() {
                        children.push(normalize_path(&entry.path().to_string_lossy()));
                    }
                }
            }
        }
        children.sort_by_key(|p| p.file_name().map(|s| s.to_string_lossy().to_lowercase()));
        let candidates = if !children.is_empty() && looks_like_workspace_root(root, &children) {
            children
        } else {
            vec![normalize_path(&root.to_string_lossy())]
        };
        for candidate in candidates {
            let key = candidate.to_string_lossy().to_string();
            if seen.insert(key) {
                out.push(candidate);
            }
        }
    }

    out.sort_by_key(|p| p.file_name().map(|s| s.to_string_lossy().to_lowercase()));
    out
}

fn looks_like_workspace_root(root: &Path, children: &[PathBuf]) -> bool {
    children.len() >= 2 && !has_indexable_files_in_root(root)
}

fn has_indexable_files_in_root(root: &Path) -> bool {
    let Ok(rd) = fs::read_dir(root) else {
        return false;
    };
    for entry in rd.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        let Ok(ft) = entry.file_type() else {
            continue;
        };
        if !ft.is_file() {
            continue;
        }
        let lname = name.to_lowercase();
        if lname == "readme" || lname == "readme.md" || lname == "notes.txt" {
            return true;
        }
        let ext = Path::new(&name)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();
        if is_indexable_suffix(&format!(".{}", ext)) {
            return true;
        }
    }
    false
}

fn project_latest_mtime(project_dir: &Path) -> Result<f64, String> {
    let mut newest = file_mtime(project_dir).unwrap_or(0.0);
    let mut stack = vec![project_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in rd.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            let Ok(ft) = entry.file_type() else {
                continue;
            };
            if ft.is_dir() {
                if name.starts_with('.') || is_skip_dir(&name) {
                    continue;
                }
                stack.push(path);
                continue;
            }
            if !ft.is_file() || name.starts_with('.') {
                continue;
            }
            if let Ok(meta) = entry.metadata() {
                let mt = metadata_mtime(&meta);
                if mt > newest {
                    newest = mt;
                }
            }
        }
    }
    Ok(newest)
}

fn collect_project_corpus(
    project_dir: &Path,
    max_chars: usize,
    newest_mtime: f64,
) -> Result<ProjectCorpus, String> {
    const MAX_FILE_BYTES: u64 = 2_000_000;
    const MAX_FILES_PER_PROJECT: usize = 600;
    const MAX_CHUNKS_PER_FILE: usize = 28;
    const MAX_CHUNKS_PER_PROJECT: usize = 1200;
    const CHUNK_SIZE_CHARS: usize = 1000;
    const CHUNK_OVERLAP_CHARS: usize = 180;
    const SUMMARY_SNIPPET_CHARS: usize = 900;
    const SUMMARY_SNIPPET_FILES: usize = 30;

    let mut file_names: Vec<String> = Vec::new();
    let mut candidate_files: Vec<(f64, PathBuf, String)> = Vec::new();

    let mut stack = vec![project_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in rd.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            let Ok(ft) = entry.file_type() else {
                continue;
            };
            if ft.is_dir() {
                if name.starts_with('.') || is_skip_dir(&name) {
                    continue;
                }
                stack.push(path);
                continue;
            }
            if !ft.is_file() || name.starts_with('.') {
                continue;
            }

            let rel = path
                .strip_prefix(project_dir)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();
            file_names.push(rel.clone());
            let ext = path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_lowercase();
            if !is_indexable_suffix(&format!(".{}", ext)) {
                continue;
            }
            let Ok(meta) = entry.metadata() else {
                continue;
            };
            if meta.len() == 0 || meta.len() > MAX_FILE_BYTES {
                continue;
            }
            candidate_files.push((metadata_mtime(&meta), path, rel));
        }
    }

    candidate_files.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut chunks: Vec<ProjectChunk> = Vec::new();
    let mut snippets: Vec<String> = Vec::new();
    let mut indexed_files = 0i64;

    for (doc_mtime, path, rel) in candidate_files.into_iter().take(MAX_FILES_PER_PROJECT) {
        let Some(text) = read_for_index(&path) else {
            continue;
        };
        indexed_files += 1;
        if snippets.len() < SUMMARY_SNIPPET_FILES {
            let snippet: String = text.chars().take(SUMMARY_SNIPPET_CHARS).collect();
            snippets.push(format!("{}\n{}", rel, snippet));
        }
        for (chunk_index, chunk_text) in chunk_text(
            &text,
            CHUNK_SIZE_CHARS,
            CHUNK_OVERLAP_CHARS,
            MAX_CHUNKS_PER_FILE,
        )
        .into_iter()
        .enumerate()
        {
            if chunks.len() >= MAX_CHUNKS_PER_PROJECT {
                break;
            }
            let token_count = word_tokens(&chunk_text).len() as i64;
            if token_count == 0 {
                continue;
            }
            let mut hasher = Sha1::new();
            hasher.update(chunk_text.as_bytes());
            let text_hash = format!("{:x}", hasher.finalize());
            chunks.push(ProjectChunk {
                doc_path: normalize_path(&path.to_string_lossy())
                    .to_string_lossy()
                    .to_string(),
                doc_rel_path: rel.clone(),
                doc_mtime,
                chunk_index: chunk_index as i64,
                token_count,
                text_hash,
                text: chunk_text,
            });
        }
        if chunks.len() >= MAX_CHUNKS_PER_PROJECT {
            break;
        }
    }

    let names_section = file_names
        .into_iter()
        .take(500)
        .collect::<Vec<_>>()
        .join(" ");
    let snippet_section = snippets.join("\n\n");
    let title = project_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("project")
        .replace('-', " ")
        .replace('_', " ");
    let mut summary = format!(
        "project {}\nindexed_files {}\nfiles {}\n\n{}",
        project_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("project"),
        indexed_files,
        names_section,
        snippet_section
    );
    if summary.chars().count() > max_chars {
        summary = summary.chars().take(max_chars).collect();
    }

    Ok(ProjectCorpus {
        doc: ProjectDoc {
            path: normalize_path(&project_dir.to_string_lossy()),
            title,
            summary,
            mtime: newest_mtime,
        },
        chunks,
    })
}

fn read_for_index(path: &Path) -> Option<String> {
    let raw = fs::read(path).ok()?;
    let text = String::from_utf8_lossy(&raw);
    let cleaned = collapse_whitespace(&text);
    if cleaned.is_empty() {
        return None;
    }
    Some(cleaned.chars().take(80_000).collect())
}

fn collapse_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn chunk_text(text: &str, size: usize, overlap: usize, max_chunks: usize) -> Vec<String> {
    let cleaned = collapse_whitespace(text);
    if cleaned.is_empty() {
        return Vec::new();
    }
    let chars: Vec<char> = cleaned.chars().collect();
    if chars.len() <= size {
        return vec![cleaned];
    }
    let mut out = Vec::new();
    let mut start = 0usize;
    let step = size.saturating_sub(overlap).max(1);
    let n = chars.len();
    while start < n && out.len() < max_chunks {
        let mut end = (start + size).min(n);
        let mut window: String = chars[start..end].iter().collect();
        if end < n {
            if let Some(split) = window.rfind(' ') {
                if split > ((size as f32) * 0.60) as usize {
                    window = window[..split].to_string();
                    end = start + window.chars().count();
                }
            }
        }
        let trimmed = window.trim();
        if !trimmed.is_empty() {
            out.push(trimmed.to_string());
        }
        if end >= n {
            break;
        }
        start += step;
    }
    out
}

fn word_tokens(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            cur.push(ch.to_ascii_lowercase());
        } else if cur.len() >= 2 {
            out.push(cur.clone());
            cur.clear();
        } else {
            cur.clear();
        }
    }
    if cur.len() >= 2 {
        out.push(cur);
    }
    out
}

fn is_under_any(path: &Path, candidates: &HashSet<PathBuf>) -> bool {
    if candidates.is_empty() {
        return false;
    }
    let resolved = normalize_path(&path.to_string_lossy());
    for candidate in candidates {
        if resolved == *candidate || resolved.starts_with(candidate) {
            return true;
        }
    }
    false
}

fn get_project_by_path(conn: &Connection, path: &str) -> Result<Option<ExistingProject>, String> {
    conn.query_row(
        "SELECT id, path, title, summary, project_mtime FROM projects WHERE path = ?1",
        params![path],
        |row| {
            Ok(ExistingProject {
                id: row.get(0)?,
                path: row.get(1)?,
                title: row.get(2)?,
                summary: row.get(3)?,
                project_mtime: row.get(4)?,
            })
        },
    )
    .optional()
    .map_err(|e| format!("failed fetching project row: {}", e))
}

fn upsert_project(
    conn: &Connection,
    path: &str,
    title: &str,
    summary: &str,
    project_mtime: f64,
    last_indexed: f64,
) -> Result<i64, String> {
    let existing: Option<i64> = conn
        .query_row(
            "SELECT id FROM projects WHERE path = ?1",
            params![path],
            |row| row.get(0),
        )
        .optional()
        .map_err(|e| format!("failed checking existing project: {}", e))?;
    if let Some(id) = existing {
        conn.execute(
            r#"
UPDATE projects
SET title = ?1, summary = ?2, project_mtime = ?3, last_indexed = ?4
WHERE id = ?5
"#,
            params![title, summary, project_mtime, last_indexed, id],
        )
        .map_err(|e| format!("failed updating project row: {}", e))?;
        Ok(id)
    } else {
        conn.execute(
            r#"
INSERT INTO projects(path, title, summary, project_mtime, last_indexed)
VALUES (?1, ?2, ?3, ?4, ?5)
"#,
            params![path, title, summary, project_mtime, last_indexed],
        )
        .map_err(|e| format!("failed inserting project row: {}", e))?;
        Ok(conn.last_insert_rowid())
    }
}

fn has_project_vector(conn: &Connection, project_id: i64, model: &str) -> Result<bool, String> {
    let row: Option<i64> = conn
        .query_row(
            "SELECT 1 FROM project_vectors WHERE project_id = ?1 AND model = ?2 LIMIT 1",
            params![project_id, model],
            |row| row.get(0),
        )
        .optional()
        .map_err(|e| format!("failed checking project vector: {}", e))?;
    Ok(row.is_some())
}

fn has_project_chunks(conn: &Connection, project_id: i64) -> Result<bool, String> {
    let row: Option<i64> = conn
        .query_row(
            "SELECT 1 FROM project_chunks WHERE project_id = ?1 LIMIT 1",
            params![project_id],
            |row| row.get(0),
        )
        .optional()
        .map_err(|e| format!("failed checking project chunks: {}", e))?;
    Ok(row.is_some())
}

fn has_project_chunk_vectors(
    conn: &Connection,
    project_id: i64,
    model: &str,
) -> Result<bool, String> {
    let row: Option<i64> = conn
        .query_row(
            r#"
SELECT 1
FROM project_chunks pc
JOIN project_chunk_vectors pcv ON pcv.chunk_id = pc.id
WHERE pc.project_id = ?1 AND pcv.model = ?2
LIMIT 1
"#,
            params![project_id, model],
            |row| row.get(0),
        )
        .optional()
        .map_err(|e| format!("failed checking chunk vectors: {}", e))?;
    Ok(row.is_some())
}

fn set_project_vector(
    conn: &Connection,
    project_id: i64,
    model: &str,
    vector: &[f32],
) -> Result<(), String> {
    let norm = vector_norm(vector);
    let blob = f32_blob(vector);
    conn.execute(
        r#"
INSERT INTO project_vectors(project_id, model, dim, norm, vector)
VALUES (?1, ?2, ?3, ?4, ?5)
ON CONFLICT(project_id) DO UPDATE SET
    model = excluded.model,
    dim = excluded.dim,
    norm = excluded.norm,
    vector = excluded.vector
"#,
        params![project_id, model, vector.len() as i64, norm, blob],
    )
    .map_err(|e| format!("failed upserting project vector: {}", e))?;
    Ok(())
}

fn clear_project_chunks(conn: &Connection, project_id: i64) -> Result<(), String> {
    conn.execute(
        "DELETE FROM project_chunks WHERE project_id = ?1",
        params![project_id],
    )
    .map_err(|e| format!("failed clearing project chunks: {}", e))?;
    Ok(())
}

fn upsert_project_chunk(
    conn: &Connection,
    project_id: i64,
    chunk: &ProjectChunk,
    updated_at: f64,
) -> Result<i64, String> {
    let existing: Option<i64> = conn
        .query_row(
            r#"
SELECT id
FROM project_chunks
WHERE project_id = ?1 AND doc_path = ?2 AND chunk_index = ?3
"#,
            params![project_id, chunk.doc_path, chunk.chunk_index],
            |row| row.get(0),
        )
        .optional()
        .map_err(|e| format!("failed checking project chunk row: {}", e))?;
    if let Some(chunk_id) = existing {
        conn.execute(
            r#"
UPDATE project_chunks
SET doc_rel_path = ?1, doc_mtime = ?2, token_count = ?3, text_hash = ?4, text = ?5, updated_at = ?6
WHERE id = ?7
"#,
            params![
                chunk.doc_rel_path,
                chunk.doc_mtime,
                chunk.token_count,
                chunk.text_hash,
                chunk.text,
                updated_at,
                chunk_id
            ],
        )
        .map_err(|e| format!("failed updating project chunk row: {}", e))?;
        Ok(chunk_id)
    } else {
        conn.execute(
            r#"
INSERT INTO project_chunks(
    project_id, doc_path, doc_rel_path, doc_mtime, chunk_index, token_count, text_hash, text, updated_at
) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
"#,
            params![
                project_id,
                chunk.doc_path,
                chunk.doc_rel_path,
                chunk.doc_mtime,
                chunk.chunk_index,
                chunk.token_count,
                chunk.text_hash,
                chunk.text,
                updated_at
            ],
        )
        .map_err(|e| format!("failed inserting project chunk row: {}", e))?;
        Ok(conn.last_insert_rowid())
    }
}

fn set_project_chunk_vector(
    conn: &Connection,
    chunk_id: i64,
    model: &str,
    vector: &[f32],
) -> Result<(), String> {
    let norm = vector_norm(vector);
    let blob = f32_blob(vector);
    conn.execute(
        r#"
INSERT INTO project_chunk_vectors(chunk_id, model, dim, norm, vector)
VALUES (?1, ?2, ?3, ?4, ?5)
ON CONFLICT(chunk_id) DO UPDATE SET
    model = excluded.model,
    dim = excluded.dim,
    norm = excluded.norm,
    vector = excluded.vector
"#,
        params![chunk_id, model, vector.len() as i64, norm, blob],
    )
    .map_err(|e| format!("failed upserting chunk vector: {}", e))?;
    Ok(())
}

fn reindex_project_chunks(
    conn: &Connection,
    project_id: i64,
    model_key: &str,
    embedder: &dyn Embedder,
    chunks: &[ProjectChunk],
    now: f64,
) -> Result<(i64, i64, i64), String> {
    const CHUNK_EMBED_BATCH: usize = 24;
    clear_project_chunks(conn, project_id)?;
    if chunks.is_empty() {
        return Ok((0, 0, 0));
    }
    let mut chunk_ids: Vec<i64> = Vec::with_capacity(chunks.len());
    let mut texts: Vec<String> = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let chunk_id = upsert_project_chunk(conn, project_id, chunk, now)?;
        chunk_ids.push(chunk_id);
        texts.push(chunk.text.clone());
    }

    let mut vectorized = 0i64;
    let mut failures = 0i64;
    for i in (0..chunk_ids.len()).step_by(CHUNK_EMBED_BATCH) {
        let end = (i + CHUNK_EMBED_BATCH).min(chunk_ids.len());
        let batch_ids = &chunk_ids[i..end];
        let batch_texts = &texts[i..end];
        match embedder.embed_many(batch_texts) {
            Ok(vectors) => {
                for (chunk_id, vector) in batch_ids.iter().zip(vectors.iter()) {
                    set_project_chunk_vector(conn, *chunk_id, model_key, vector)?;
                    vectorized += 1;
                }
                if vectors.len() < batch_ids.len() {
                    failures += (batch_ids.len() - vectors.len()) as i64;
                }
            }
            Err(_) => {
                failures += batch_ids.len() as i64;
            }
        }
    }
    Ok((chunks.len() as i64, vectorized, failures))
}

fn remove_projects_not_in(conn: &Connection, keep_paths: &[String]) -> Result<i64, String> {
    if keep_paths.is_empty() {
        let removed = conn
            .execute("DELETE FROM projects", [])
            .map_err(|e| format!("failed clearing projects table: {}", e))?;
        return Ok(removed as i64);
    }
    let keep_set: HashSet<String> = keep_paths.iter().cloned().collect();
    let mut stmt = conn
        .prepare("SELECT path FROM projects ORDER BY path")
        .map_err(|e| format!("failed preparing project list query: {}", e))?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|e| format!("failed listing existing projects: {}", e))?;
    let mut delete_paths: Vec<String> = Vec::new();
    for row in rows {
        let path = row.map_err(|e| format!("failed reading existing project row: {}", e))?;
        if !keep_set.contains(&path) {
            delete_paths.push(path);
        }
    }
    let mut removed = 0i64;
    for path in delete_paths {
        removed += conn
            .execute("DELETE FROM projects WHERE path = ?1", params![path])
            .map_err(|e| format!("failed deleting stale project row: {}", e))?
            as i64;
    }
    Ok(removed)
}

fn rebuild_relationship_edges(
    conn: &Connection,
    docs_by_id: &HashMap<i64, ProjectDoc>,
) -> Result<i64, String> {
    if docs_by_id.is_empty() {
        return Ok(0);
    }
    let mut node_meta: HashMap<i64, (String, HashSet<String>, HashSet<String>)> = HashMap::new();
    for (project_id, doc) in docs_by_id {
        let signal_tokens: HashSet<String> = word_tokens(&doc.summary)
            .into_iter()
            .filter(|t| !is_graph_stopword(t))
            .collect();
        let mut name_tokens: HashSet<String> = word_tokens(&doc.title).into_iter().collect();
        if let Some(name) = doc.path.file_name().and_then(|s| s.to_str()) {
            for t in word_tokens(name) {
                name_tokens.insert(t);
            }
        }
        node_meta.insert(
            *project_id,
            (
                doc.path.to_string_lossy().to_string(),
                signal_tokens,
                name_tokens,
            ),
        );
    }

    let mut total_edges = 0i64;
    for (src_id, (_src_path, src_tokens, src_name_tokens)) in &node_meta {
        let mut edges: Vec<(String, String, f64)> = Vec::new();
        for (dst_id, (dst_path, _dst_tokens, dst_name_tokens)) in &node_meta {
            if src_id == dst_id {
                continue;
            }
            let mention = src_tokens.intersection(dst_name_tokens).count();
            let mention_score = if mention >= 2 {
                0.8
            } else if mention == 1 && dst_name_tokens.len() <= 2 {
                0.4
            } else {
                0.0
            };

            let overlap = jaccard(src_name_tokens, dst_name_tokens);
            let overlap_score = if overlap >= 0.25 { overlap } else { 0.0 };
            let weight = mention_score + overlap_score;
            if weight < 0.45 {
                continue;
            }
            let rounded = ((weight.min(2.0) * 1000.0).round()) / 1000.0;
            edges.push((dst_path.clone(), "semantic_related".to_string(), rounded));
        }
        edges.sort_by(|a, b| b.2.total_cmp(&a.2));
        if edges.len() > 40 {
            edges.truncate(40);
        }
        set_project_edges(conn, *src_id, &edges)?;
        total_edges += edges.len() as i64;
    }
    Ok(total_edges)
}

fn set_project_edges(
    conn: &Connection,
    project_id: i64,
    edges: &[(String, String, f64)],
) -> Result<(), String> {
    conn.execute(
        "DELETE FROM project_edges WHERE src_project_id = ?1",
        params![project_id],
    )
    .map_err(|e| format!("failed clearing project edges: {}", e))?;
    for (dst, kind, weight) in edges {
        conn.execute(
            r#"
INSERT INTO project_edges(src_project_id, dst, kind, weight)
VALUES (?1, ?2, ?3, ?4)
"#,
            params![project_id, dst, kind, weight],
        )
        .map_err(|e| format!("failed inserting project edge: {}", e))?;
    }
    Ok(())
}

fn f32_blob(vector: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vector.len() * 4);
    for v in vector {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn vector_norm(vector: &[f32]) -> f64 {
    let mut sum = 0.0f64;
    for v in vector {
        let f = *v as f64;
        sum += f * f;
    }
    sum.sqrt()
}

fn metadata_mtime(meta: &fs::Metadata) -> f64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn file_mtime(path: &Path) -> Option<f64> {
    let meta = fs::metadata(path).ok()?;
    Some(metadata_mtime(&meta))
}

fn is_skip_dir(name: &str) -> bool {
    matches!(
        name,
        ".git"
            | ".hg"
            | ".svn"
            | "__pycache__"
            | ".cache"
            | ".mypy_cache"
            | ".pytest_cache"
            | "node_modules"
            | ".venv"
            | "venv"
            | ".idea"
            | "cdk.out"
            | ".next"
            | "dist"
            | "build"
            | "target"
    )
}

fn is_indexable_suffix(suffix: &str) -> bool {
    matches!(
        suffix,
        ".md"
            | ".markdown"
            | ".txt"
            | ".rst"
            | ".adoc"
            | ".html"
            | ".htm"
            | ".py"
            | ".js"
            | ".ts"
            | ".tsx"
            | ".jsx"
            | ".go"
            | ".rs"
            | ".java"
            | ".c"
            | ".cc"
            | ".cpp"
            | ".h"
            | ".hpp"
            | ".sh"
            | ".bash"
            | ".zsh"
            | ".yaml"
            | ".yml"
            | ".toml"
            | ".json"
            | ".sql"
    )
}

fn is_graph_stopword(tok: &str) -> bool {
    matches!(
        tok,
        "the"
            | "and"
            | "for"
            | "with"
            | "from"
            | "into"
            | "this"
            | "that"
            | "project"
            | "file"
            | "files"
            | "readme"
            | "docs"
            | "notes"
            | "src"
            | "test"
            | "tests"
    )
}

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count();
    if inter == 0 {
        return 0.0;
    }
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    inter as f64 / union as f64
}

trait Embedder {
    fn model_key(&self) -> String;
    fn embed_many(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String>;
    fn embed_one(&self, text: &str) -> Result<Vec<f32>, String> {
        let rows = self.embed_many(&[text.to_string()])?;
        rows.into_iter()
            .next()
            .ok_or_else(|| "No embedding returned.".to_string())
    }
}

fn build_embedder(cfg: &ConfigValues) -> Result<Box<dyn Embedder>, String> {
    match cfg.embed_backend.as_str() {
        "local-hash" | "local" => Ok(Box::new(LocalHashEmbedder::new(
            cfg.local_embed_dim as usize,
        ))),
        "ollama" => Ok(Box::new(OllamaEmbedder::new(&cfg.embed_model))),
        other => Err(format!(
            "native index does not support backend '{}' yet; use local-hash or ollama",
            other
        )),
    }
}

struct OllamaEmbedder {
    model: String,
    host: String,
    keep_alive: Option<String>,
    timeout_sec: u64,
}

impl OllamaEmbedder {
    fn new(model: &str) -> Self {
        let model_name = if model.trim().is_empty() {
            "qwen3-embedding".to_string()
        } else {
            model.trim().to_string()
        };
        let host = env::var("OLLAMA_HOST")
            .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string())
            .trim_end_matches('/')
            .to_string();
        let keep_alive = env::var("CYPRESS_OLLAMA_KEEP_ALIVE")
            .ok()
            .or_else(|| env::var("SEMCD_OLLAMA_KEEP_ALIVE").ok())
            .unwrap_or_else(|| "24h".to_string())
            .trim()
            .to_string();
        let keep_alive = if keep_alive.is_empty() {
            None
        } else {
            Some(keep_alive)
        };
        Self {
            model: model_name,
            host,
            keep_alive,
            timeout_sec: 60,
        }
    }

    fn request_embed(
        &self,
        payload: &Value,
        allow_retry_without_keep_alive: bool,
    ) -> Result<Value, String> {
        let url = format!("{}/api/embed", self.host);
        let body = payload.to_string();
        let agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(self.timeout_sec))
            .build();
        let response = agent
            .post(&url)
            .set("Content-Type", "application/json")
            .send_string(&body);
        match response {
            Ok(resp) => {
                let raw = resp
                    .into_string()
                    .map_err(|e| format!("failed reading Ollama response: {}", e))?;
                serde_json::from_str::<Value>(&raw)
                    .map_err(|e| format!("failed parsing Ollama response JSON: {}", e))
            }
            Err(ureq::Error::Status(code, resp)) => {
                if code == 400 && allow_retry_without_keep_alive && payload.get("keep_alive").is_some() {
                    let mut retry_payload = payload.clone();
                    if let Some(obj) = retry_payload.as_object_mut() {
                        obj.remove("keep_alive");
                    }
                    return self.request_embed(&retry_payload, false);
                }
                let detail = resp.into_string().unwrap_or_default();
                Err(format!(
                    "Ollama embedding request failed (HTTP {}): {}",
                    code, detail
                ))
            }
            Err(ureq::Error::Transport(e)) => Err(format!(
                "Ollama embedding request failed: {}. Ensure Ollama is running and model '{}' is available.",
                e, self.model
            )),
        }
    }

    fn parse_vectors(data: &Value) -> Result<Vec<Vec<f32>>, String> {
        if let Some(embeddings) = data.get("embeddings").and_then(|v| v.as_array()) {
            let mut out = Vec::new();
            for item in embeddings {
                let Some(arr) = item.as_array() else {
                    continue;
                };
                let mut row = Vec::with_capacity(arr.len());
                for num in arr {
                    if let Some(f) = num.as_f64() {
                        row.push(f as f32);
                    }
                }
                if !row.is_empty() {
                    out.push(row);
                }
            }
            return Ok(out);
        }
        if let Some(single) = data.get("embedding").and_then(|v| v.as_array()) {
            let mut row = Vec::with_capacity(single.len());
            for num in single {
                if let Some(f) = num.as_f64() {
                    row.push(f as f32);
                }
            }
            if !row.is_empty() {
                return Ok(vec![row]);
            }
        }
        Err("Unexpected Ollama embed response format.".to_string())
    }
}

impl Embedder for OllamaEmbedder {
    fn model_key(&self) -> String {
        format!("ollama:{}", self.model)
    }

    fn embed_many(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut payload = serde_json::json!({
            "model": self.model,
            "input": texts,
        });
        if let Some(keep_alive) = &self.keep_alive {
            if let Some(obj) = payload.as_object_mut() {
                obj.insert("keep_alive".to_string(), Value::String(keep_alive.clone()));
            }
        }
        let data = self.request_embed(&payload, true)?;
        Self::parse_vectors(&data)
    }
}

struct LocalHashEmbedder {
    dim: usize,
    synonym_map: HashMap<String, Vec<String>>,
}

impl LocalHashEmbedder {
    fn new(dim: usize) -> Self {
        let use_dim = dim.max(64);
        let groups: &[&[&str]] = &[
            &[
                "semantic", "meaning", "ontology", "taxonomy", "model", "schema", "layer",
            ],
            &["api", "service", "endpoint", "backend"],
            &["ui", "frontend", "interface", "ux"],
            &["storage", "database", "db", "persistence"],
            &["auth", "authentication", "login", "identity"],
            &["agent", "assistant", "automation"],
        ];
        let mut synonym_map: HashMap<String, Vec<String>> = HashMap::new();
        for group in groups {
            for token in *group {
                let mut list = Vec::new();
                for other in *group {
                    if other != token {
                        list.push((*other).to_string());
                    }
                }
                synonym_map.insert((*token).to_string(), list);
            }
        }
        Self {
            dim: use_dim,
            synonym_map,
        }
    }

    fn model_key_local(&self) -> String {
        format!("local-hash-v1:{}", self.dim)
    }

    fn embed_one_local(&self, text: &str) -> Vec<f32> {
        let mut vec = vec![0.0f32; self.dim];
        let normalized = text.to_lowercase();
        let tokens = word_tokens(&normalized);
        if tokens.is_empty() {
            return vec;
        }

        for token in &tokens {
            self.add_feature(&mut vec, &format!("t:{}", token), 1.0);
            if let Some(expanded) = self.synonym_map.get(token) {
                for syn in expanded {
                    self.add_feature(&mut vec, &format!("s:{}", syn), 0.35);
                }
            }
        }
        for pair in tokens.windows(2) {
            self.add_feature(&mut vec, &format!("b:{}_{}", pair[0], pair[1]), 0.8);
        }

        let compact: String = normalized.chars().filter(|c| !c.is_whitespace()).collect();
        let compact_chars: Vec<char> = compact.chars().collect();
        if compact_chars.len() >= 3 {
            for tri in compact_chars.windows(3) {
                let trigram: String = tri.iter().collect();
                self.add_feature(&mut vec, &format!("c:{}", trigram), 0.15);
            }
        }

        let norm = vector_norm(&vec);
        if norm > 0.0 {
            for v in &mut vec {
                *v = (*v as f64 / norm) as f32;
            }
        }
        vec
    }

    fn add_feature(&self, vec: &mut [f32], feature: &str, weight: f32) {
        let mut hasher = Sha1::new();
        hasher.update(feature.as_bytes());
        let digest = hasher.finalize();
        let mut first = [0u8; 8];
        first.copy_from_slice(&digest[..8]);
        let idx = (u64::from_le_bytes(first) as usize) % self.dim;
        let sign = if (digest[8] & 1) == 0 { 1.0 } else { -1.0 };
        vec[idx] += sign * weight;
    }
}

impl Embedder for LocalHashEmbedder {
    fn model_key(&self) -> String {
        self.model_key_local()
    }

    fn embed_many(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        Ok(texts.iter().map(|t| self.embed_one_local(t)).collect())
    }

    fn embed_one(&self, text: &str) -> Result<Vec<f32>, String> {
        Ok(self.embed_one_local(text))
    }
}

fn yes_no(v: bool) -> &'static str {
    if v {
        "yes"
    } else {
        "no"
    }
}

fn command_exists(name: &str) -> bool {
    Command::new("sh")
        .arg("-lc")
        .arg(format!("command -v {} >/dev/null 2>&1", shell_escape(name)))
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn is_executable_file(path: &Path) -> bool {
    let meta = match fs::metadata(path) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if !meta.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        return (meta.permissions().mode() & 0o111) != 0;
    }
    #[cfg(not(unix))]
    {
        true
    }
}

fn tracked_roots_count(db_path: &Path) -> Option<i64> {
    if !db_path.exists() {
        return Some(0);
    }
    let conn = open_db_read_only(db_path).ok()?;
    conn.query_row(
        "SELECT COUNT(*) FROM tracked_roots WHERE enabled = 1",
        [],
        |row| row.get::<_, i64>(0),
    )
    .ok()
}

fn database_ready(db_path: &Path) -> bool {
    if !db_path.exists() {
        return false;
    }
    match open_db_read_only(db_path) {
        Ok(conn) => conn
            .query_row("SELECT 1", [], |row| row.get::<_, i64>(0))
            .map(|_| true)
            .unwrap_or(false),
        Err(_) => false,
    }
}

fn config_path(cwd: &Path) -> PathBuf {
    if let Ok(path) = env::var("CYPRESS_CONFIG") {
        return expand_tilde(path);
    }
    if let Ok(path) = env::var("SEMCD_CONFIG") {
        return expand_tilde(path);
    }
    data_dir(cwd).join("config.toml")
}

fn db_path(cwd: &Path) -> PathBuf {
    if let Ok(path) = env::var("CYPRESS_DATA_DIR") {
        return expand_tilde(path).join("cypress.db");
    }
    if let Ok(path) = env::var("SEMCD_DATA_DIR") {
        return expand_tilde(path).join("cypress.db");
    }
    data_dir(cwd).join("cypress.db")
}

fn data_dir(cwd: &Path) -> PathBuf {
    if let Ok(path) = env::var("CYPRESS_DATA_DIR") {
        return expand_tilde(path);
    }
    if let Ok(path) = env::var("SEMCD_DATA_DIR") {
        return expand_tilde(path);
    }

    if let Some(existing) = find_existing_state_dir(cwd) {
        return existing;
    }
    if let Some(git_root) = find_parent_with_marker(cwd, ".git") {
        return git_root.join(".cypress");
    }
    cwd.join(".cypress")
}

fn find_existing_state_dir(start: &Path) -> Option<PathBuf> {
    let mut cur = start.to_path_buf();
    loop {
        let candidate = cur.join(".cypress");
        if candidate.is_dir() {
            return Some(candidate);
        }
        if !cur.pop() {
            return None;
        }
    }
}

fn find_parent_with_marker(start: &Path, marker: &str) -> Option<PathBuf> {
    let mut cur = start.to_path_buf();
    loop {
        if cur.join(marker).exists() {
            return Some(cur);
        }
        if !cur.pop() {
            return None;
        }
    }
}

fn find_repo_root() -> Option<PathBuf> {
    if let Ok(v) = env::var("CYPRESS_REPO_DIR") {
        let p = expand_tilde(v);
        if p.join("src/semcd/cli.py").exists() {
            return Some(p);
        }
    }
    if let Ok(v) = env::var("SEMCD_REPO_DIR") {
        let p = expand_tilde(v);
        if p.join("src/semcd/cli.py").exists() {
            return Some(p);
        }
    }
    let cwd = env::current_dir().ok()?;
    let mut cur = cwd;
    loop {
        if cur.join("src/semcd/cli.py").exists() && cur.join("pyproject.toml").exists() {
            return Some(cur);
        }
        if !cur.pop() {
            return None;
        }
    }
}

fn load_config_values(path: &Path) -> std::collections::HashMap<String, String> {
    let mut out = std::collections::HashMap::new();
    let data = match fs::read_to_string(path) {
        Ok(v) => v,
        Err(_) => return out,
    };
    for raw in data.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let key = k.trim().to_string();
        let mut val = v.trim().to_string();
        if let Some(comment_idx) = val.find('#') {
            val = val[..comment_idx].trim().to_string();
        }
        if val.starts_with('"') && val.ends_with('"') && val.len() >= 2 {
            val = val[1..val.len() - 1].to_string();
        }
        out.insert(key, val);
    }
    out
}

fn expand_tilde<S: AsRef<str>>(s: S) -> PathBuf {
    let raw = s.as_ref();
    if raw == "~" {
        if let Ok(home) = env::var("HOME") {
            return PathBuf::from(home);
        }
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        if let Ok(home) = env::var("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(raw)
}

fn shell_escape(s: &str) -> String {
    let escaped = s.replace('\'', "'\"'\"'");
    format!("'{}'", escaped)
}
