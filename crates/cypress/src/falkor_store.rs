use std::collections::HashMap;
use std::env;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;
use std::time::Instant;

use rusqlite::{params, Connection};

#[derive(Default, Clone, Debug)]
pub struct SyncStats {
    pub chunk_nodes: i64,
}

#[derive(Default, Clone, Debug)]
pub struct HybridChunkScores {
    pub semantic: HashMap<i64, f64>,
    pub lexical: HashMap<i64, f64>,
}

#[derive(Default, Clone, Debug)]
pub struct BenchLatencyStats {
    pub connect_s: f64,
    pub vector_ms: Vec<f64>,
    pub fts_ms: Vec<f64>,
    pub total_ms: Vec<f64>,
}

#[derive(Debug, Clone)]
enum Resp {
    Simple(String),
    Error(String),
    Integer(i64),
    Bulk(Option<Vec<u8>>),
    Array(Option<Vec<Resp>>),
}

struct RedisClient {
    reader: BufReader<TcpStream>,
    writer: TcpStream,
}

impl RedisClient {
    fn connect() -> Result<Self, String> {
        let raw_url = env::var("CYPRESS_FALKOR_URL")
            .unwrap_or_else(|_| "falkor://127.0.0.1:6379".to_string());
        let (host, port) = parse_falkor_url(&raw_url)?;
        let addr = format!("{}:{}", host, port);
        let mut addrs = addr
            .to_socket_addrs()
            .map_err(|e| format!("failed to resolve FalkorDB address '{}': {}", addr, e))?;
        let target = addrs
            .next()
            .ok_or_else(|| format!("failed to resolve FalkorDB address '{}'", addr))?;
        let stream = TcpStream::connect_timeout(&target, Duration::from_secs(4))
            .map_err(|e| format!("failed to connect to FalkorDB at '{}': {}", addr, e))?;
        stream
            .set_read_timeout(Some(Duration::from_secs(90)))
            .map_err(|e| format!("failed setting read timeout: {}", e))?;
        stream
            .set_write_timeout(Some(Duration::from_secs(90)))
            .map_err(|e| format!("failed setting write timeout: {}", e))?;
        let writer = stream
            .try_clone()
            .map_err(|e| format!("failed cloning FalkorDB socket: {}", e))?;
        Ok(Self {
            reader: BufReader::new(stream),
            writer,
        })
    }

    fn cmd(&mut self, args: &[String]) -> Result<Resp, String> {
        let mut payload = Vec::<u8>::new();
        payload.extend_from_slice(format!("*{}\r\n", args.len()).as_bytes());
        for arg in args {
            let bytes = arg.as_bytes();
            payload.extend_from_slice(format!("${}\r\n", bytes.len()).as_bytes());
            payload.extend_from_slice(bytes);
            payload.extend_from_slice(b"\r\n");
        }
        self.writer
            .write_all(&payload)
            .map_err(|e| format!("failed writing FalkorDB command: {}", e))?;
        self.writer
            .flush()
            .map_err(|e| format!("failed flushing FalkorDB command: {}", e))?;

        let resp = read_resp(&mut self.reader)?;
        if let Resp::Error(err) = &resp {
            return Err(format!("falkordb error: {}", err));
        }
        Ok(resp)
    }

    fn graph_query(&mut self, query: &str) -> Result<Resp, String> {
        let graph = env::var("CYPRESS_FALKOR_GRAPH").unwrap_or_else(|_| "cypress".to_string());
        let args = vec![
            "GRAPH.QUERY".to_string(),
            graph,
            query.to_string(),
            "--compact".to_string(),
        ];
        self.cmd(&args)
    }
}

pub fn sync_from_sqlite(conn: &Connection, model_key: &str) -> Result<SyncStats, String> {
    let mut client = RedisClient::connect()?;
    let sqlite_projects = load_sqlite_projects(conn)?;
    let graph_projects = load_graph_projects(&mut client)?;

    let mut removed_paths: Vec<String> = graph_projects
        .keys()
        .filter(|p| !sqlite_projects.contains_key(*p))
        .cloned()
        .collect();
    removed_paths.sort();

    let mut changed_paths: Vec<String> = Vec::new();
    for (path, meta) in &sqlite_projects {
        let changed = match graph_projects.get(path) {
            None => true,
            Some(existing_mtime) => (existing_mtime - meta.mtime).abs() > 0.000_01,
        };
        if changed {
            changed_paths.push(path.clone());
        }
    }
    changed_paths.sort();

    for path in &removed_paths {
        delete_project_subgraph(&mut client, path)?;
    }

    let mut chunk_nodes = 0i64;
    for path in &changed_paths {
        let Some(meta) = sqlite_projects.get(path) else {
            continue;
        };
        upsert_project_node(&mut client, path, meta)?;
        clear_project_content_subgraph(&mut client, path)?;
        sync_project_files_for_path(conn, &mut client, path)?;
        chunk_nodes += sync_project_chunks_for_path(conn, &mut client, path, model_key)?;
    }

    if !removed_paths.is_empty() || !changed_paths.is_empty() || graph_projects.is_empty() {
        refresh_all_project_edges(conn, &mut client)?;
        ensure_indexes(&mut client, model_key, conn)?;
    }

    Ok(SyncStats { chunk_nodes })
}

pub fn search_hybrid_chunks(
    query: &str,
    query_vector: &[f32],
    semantic_limit: usize,
    lexical_limit: usize,
) -> Result<HybridChunkScores, String> {
    let mut client = RedisClient::connect()?;
    let vec_query = format!(
        "CALL db.idx.vector.queryNodes('Chunk', 'embedding', {}, vecf32({})) \
YIELD node, score \
WITH node, score WHERE node.chunk_id IS NOT NULL \
RETURN node.chunk_id AS chunk_id, toFloat(score) AS score \
ORDER BY score DESC LIMIT {}",
        semantic_limit.max(1),
        vector_literal(query_vector),
        semantic_limit.max(1)
    );
    let lex_query = format!(
        "CALL db.idx.fulltext.queryNodes('Chunk', {}) \
YIELD node, score \
WITH node, score WHERE node.chunk_id IS NOT NULL \
RETURN node.chunk_id AS chunk_id, toFloat(score) AS score \
ORDER BY score DESC LIMIT {}",
        cypher_string_literal(query),
        lexical_limit.max(1)
    );

    let semantic_rows = query_id_score_rows(&mut client, &vec_query)?;
    let lexical_rows = query_id_score_rows(&mut client, &lex_query)?;

    Ok(HybridChunkScores {
        semantic: normalize_scores(&semantic_rows),
        lexical: normalize_scores(&lexical_rows),
    })
}

pub fn ping() -> Result<(), String> {
    let mut client = RedisClient::connect()?;
    let resp = client.cmd(&["PING".to_string()])?;
    match resp {
        Resp::Simple(s) if s.to_uppercase() == "PONG" => Ok(()),
        other => Err(format!("unexpected PING response: {:?}", other)),
    }
}

pub fn benchmark_hybrid_queries(
    queries: &[(String, Vec<f32>)],
    limit: usize,
    repeats: usize,
    warmup: usize,
) -> Result<BenchLatencyStats, String> {
    let t_connect = Instant::now();
    let mut client = RedisClient::connect()?;
    let connect_s = t_connect.elapsed().as_secs_f64();
    let use_limit = limit.max(1);
    let use_repeats = repeats.max(1);
    let use_warmup = warmup;

    let mut out = BenchLatencyStats {
        connect_s,
        ..Default::default()
    };

    for (query, query_vector) in queries {
        let vec_query = format!(
            "CALL db.idx.vector.queryNodes('Chunk', 'embedding', {}, vecf32({})) \
YIELD node, score \
WITH node, score WHERE node.chunk_id IS NOT NULL \
RETURN node.chunk_id AS chunk_id, toFloat(score) AS score \
ORDER BY score DESC LIMIT {}",
            use_limit,
            vector_literal(query_vector),
            use_limit
        );
        let lex_query = format!(
            "CALL db.idx.fulltext.queryNodes('Chunk', {}) \
YIELD node, score \
WITH node, score WHERE node.chunk_id IS NOT NULL \
RETURN node.chunk_id AS chunk_id, toFloat(score) AS score \
ORDER BY score DESC LIMIT {}",
            cypher_string_literal(query),
            use_limit
        );

        for _ in 0..use_warmup {
            let _ = query_id_score_rows(&mut client, &vec_query);
            let _ = query_id_score_rows(&mut client, &lex_query);
        }

        for _ in 0..use_repeats {
            let t_vec = Instant::now();
            let vec_rows = query_id_score_rows(&mut client, &vec_query)?;
            let vec_ms = t_vec.elapsed().as_secs_f64() * 1000.0;

            let t_fts = Instant::now();
            let lex_rows = query_id_score_rows(&mut client, &lex_query)?;
            let fts_ms = t_fts.elapsed().as_secs_f64() * 1000.0;

            let t_fuse = Instant::now();
            let vec_ids: Vec<i64> = vec_rows.into_iter().map(|(id, _)| id).collect();
            let lex_ids: Vec<i64> = lex_rows.into_iter().map(|(id, _)| id).collect();
            let _fused = rrf_fuse(&[vec_ids, lex_ids], use_limit);
            let fuse_ms = t_fuse.elapsed().as_secs_f64() * 1000.0;

            out.vector_ms.push(vec_ms);
            out.fts_ms.push(fts_ms);
            out.total_ms.push(vec_ms + fts_ms + fuse_ms);
        }
    }

    Ok(out)
}

#[derive(Clone, Debug)]
struct SqliteProjectMeta {
    title: String,
    summary: String,
    mtime: f64,
}

fn load_sqlite_projects(conn: &Connection) -> Result<HashMap<String, SqliteProjectMeta>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT path, title, summary, project_mtime
FROM projects
ORDER BY id
"#,
        )
        .map_err(|e| format!("failed preparing project sync query: {}", e))?;
    let rows = stmt
        .query_map([], |row| {
            let path: String = row.get(0)?;
            let title: String = row.get(1)?;
            let summary: String = row.get(2)?;
            let mtime: f64 = row.get(3)?;
            Ok((path, title, summary, mtime))
        })
        .map_err(|e| format!("failed querying projects for sync: {}", e))?;

    let mut out: HashMap<String, SqliteProjectMeta> = HashMap::new();
    for row in rows {
        let (path, title, summary, mtime) =
            row.map_err(|e| format!("failed reading project sync row: {}", e))?;
        out.insert(
            path,
            SqliteProjectMeta {
                title,
                summary,
                mtime,
            },
        );
    }
    Ok(out)
}

fn load_graph_projects(client: &mut RedisClient) -> Result<HashMap<String, f64>, String> {
    let rows = extract_graph_rows(
        client.graph_query("MATCH (p:Project) RETURN p.path AS path, p.mtime AS mtime")?,
    )?;
    let mut out: HashMap<String, f64> = HashMap::new();
    for row in rows {
        if row.len() < 2 {
            continue;
        }
        let Some(path) = decode_cell_string(&row[0]) else {
            continue;
        };
        let mtime = decode_cell_f64(&row[1]).unwrap_or(0.0);
        out.insert(path, mtime);
    }
    Ok(out)
}

fn upsert_project_node(
    client: &mut RedisClient,
    path: &str,
    meta: &SqliteProjectMeta,
) -> Result<(), String> {
    let q = format!(
        "MERGE (p:Project {{path:{}}}) \
SET p.title={}, p.summary={}, p.mtime={}",
        cypher_string_literal(path),
        cypher_string_literal(&meta.title),
        cypher_string_literal(&meta.summary),
        f64_literal(meta.mtime),
    );
    client
        .graph_query(&q)
        .map_err(|e| format!("failed upserting project '{}': {}", path, e))?;
    Ok(())
}

fn delete_project_subgraph(client: &mut RedisClient, path: &str) -> Result<(), String> {
    let q = format!(
        "MATCH (p:Project {{path:{}}}) DETACH DELETE p",
        cypher_string_literal(path)
    );
    client
        .graph_query(&q)
        .map_err(|e| format!("failed deleting project '{}' in FalkorDB: {}", path, e))?;
    Ok(())
}

fn clear_project_content_subgraph(client: &mut RedisClient, path: &str) -> Result<(), String> {
    let delete_chunks = format!(
        "MATCH (p:Project {{path:{}}})-[:CONTAINS_FILE]->(:File)-[:CONTAINS_CHUNK]->(c:Chunk) \
DETACH DELETE c",
        cypher_string_literal(path)
    );
    client
        .graph_query(&delete_chunks)
        .map_err(|e| format!("failed clearing chunk subgraph for '{}': {}", path, e))?;

    let delete_files = format!(
        "MATCH (p:Project {{path:{}}})-[:CONTAINS_FILE]->(f:File) DETACH DELETE f",
        cypher_string_literal(path)
    );
    client
        .graph_query(&delete_files)
        .map_err(|e| format!("failed clearing file subgraph for '{}': {}", path, e))?;
    Ok(())
}

fn sync_project_files_for_path(
    conn: &Connection,
    client: &mut RedisClient,
    project_path: &str,
) -> Result<(), String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT DISTINCT
    pc.doc_path,
    pc.doc_rel_path,
    pc.doc_mtime
FROM project_chunks pc
JOIN projects p ON p.id = pc.project_id
WHERE p.path = ?1
ORDER BY pc.doc_path
"#,
        )
        .map_err(|e| format!("failed preparing project file sync query: {}", e))?;
    let rows = stmt
        .query_map(params![project_path], |row| {
            let doc_path: String = row.get(0)?;
            let doc_rel_path: String = row.get(1)?;
            let doc_mtime: f64 = row.get(2)?;
            Ok((doc_path, doc_rel_path, doc_mtime))
        })
        .map_err(|e| {
            format!(
                "failed querying files for project '{}': {}",
                project_path, e
            )
        })?;

    for row in rows {
        let (doc_path, doc_rel_path, doc_mtime) =
            row.map_err(|e| format!("failed reading file sync row: {}", e))?;
        let q = format!(
            "MATCH (p:Project {{path:{}}}) \
MERGE (f:File {{path:{}}}) \
SET f.rel_path={}, f.project_path={}, f.mtime={} \
MERGE (p)-[:CONTAINS_FILE]->(f)",
            cypher_string_literal(project_path),
            cypher_string_literal(&doc_path),
            cypher_string_literal(&doc_rel_path),
            cypher_string_literal(project_path),
            f64_literal(doc_mtime),
        );
        client
            .graph_query(&q)
            .map_err(|e| format!("failed syncing file '{}': {}", doc_path, e))?;
    }
    Ok(())
}

fn sync_project_chunks_for_path(
    conn: &Connection,
    client: &mut RedisClient,
    project_path: &str,
    model_key: &str,
) -> Result<i64, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT
    pc.id,
    pc.doc_path,
    pc.doc_rel_path,
    pc.chunk_index,
    pc.token_count,
    pc.text_hash,
    pc.text,
    pcv.vector
FROM project_chunks pc
JOIN project_chunk_vectors pcv ON pcv.chunk_id = pc.id
JOIN projects p ON p.id = pc.project_id
WHERE pcv.model = ?1
  AND p.path = ?2
ORDER BY pc.id
"#,
        )
        .map_err(|e| format!("failed preparing chunk sync query: {}", e))?;
    let rows = stmt
        .query_map(params![model_key, project_path], |row| {
            let chunk_id: i64 = row.get(0)?;
            let doc_path: String = row.get(1)?;
            let doc_rel_path: String = row.get(2)?;
            let chunk_index: i64 = row.get(3)?;
            let token_count: i64 = row.get(4)?;
            let text_hash: String = row.get(5)?;
            let text: String = row.get(6)?;
            let vector_blob: Vec<u8> = row.get(7)?;
            Ok((
                chunk_id,
                doc_path,
                doc_rel_path,
                chunk_index,
                token_count,
                text_hash,
                text,
                vector_blob,
            ))
        })
        .map_err(|e| {
            format!(
                "failed querying chunks for project '{}': {}",
                project_path, e
            )
        })?;

    let mut n = 0i64;
    for row in rows {
        let (
            chunk_id,
            doc_path,
            doc_rel_path,
            chunk_index,
            token_count,
            text_hash,
            text,
            vector_blob,
        ) = row.map_err(|e| format!("failed reading chunk sync row: {}", e))?;
        let vector = blob_to_f32_vec(&vector_blob);
        let q = format!(
            "MATCH (f:File {{path:{}}}) \
MERGE (c:Chunk {{chunk_id:{}}}) \
SET c.doc_path={}, c.doc_rel_path={}, c.chunk_index={}, c.token_count={}, c.text_hash={}, c.text={}, c.embedding=vecf32({}) \
MERGE (f)-[:CONTAINS_CHUNK]->(c)",
            cypher_string_literal(&doc_path),
            chunk_id,
            cypher_string_literal(&doc_path),
            cypher_string_literal(&doc_rel_path),
            chunk_index,
            token_count,
            cypher_string_literal(&text_hash),
            cypher_string_literal(&text),
            vector_literal(&vector),
        );
        client
            .graph_query(&q)
            .map_err(|e| format!("failed syncing chunk {}: {}", chunk_id, e))?;
        n += 1;
    }
    Ok(n)
}

fn refresh_all_project_edges(conn: &Connection, client: &mut RedisClient) -> Result<i64, String> {
    exec_ignore_error(client, "MATCH ()-[r:RELATED]->() DELETE r");

    let mut stmt = conn
        .prepare(
            r#"
SELECT
    src.path AS src_path,
    pe.dst,
    pe.kind,
    pe.weight
FROM project_edges pe
JOIN projects src ON src.id = pe.src_project_id
ORDER BY src.path, pe.dst, pe.kind
"#,
        )
        .map_err(|e| format!("failed preparing edge sync query: {}", e))?;
    let rows = stmt
        .query_map([], |row| {
            let src_path: String = row.get(0)?;
            let dst_path: String = row.get(1)?;
            let kind: String = row.get(2)?;
            let weight: f64 = row.get(3)?;
            Ok((src_path, dst_path, kind, weight))
        })
        .map_err(|e| format!("failed querying edges for sync: {}", e))?;

    let mut n = 0i64;
    for row in rows {
        let (src_path, dst_path, kind, weight) =
            row.map_err(|e| format!("failed reading edge sync row: {}", e))?;
        let q = format!(
            "MATCH (a:Project {{path:{}}}), (b:Project {{path:{}}}) \
MERGE (a)-[r:RELATED {{kind:{}}}]->(b) \
SET r.weight={}",
            cypher_string_literal(&src_path),
            cypher_string_literal(&dst_path),
            cypher_string_literal(&kind),
            f64_literal(weight),
        );
        client
            .graph_query(&q)
            .map_err(|e| format!("failed syncing edge {} -> {}: {}", src_path, dst_path, e))?;
        n += 1;
    }
    Ok(n)
}

fn ensure_indexes(
    client: &mut RedisClient,
    model_key: &str,
    conn: &Connection,
) -> Result<(), String> {
    let dim = vector_dim_for_model(conn, model_key)?;
    exec_ignore_error(client, "CREATE INDEX FOR (p:Project) ON (p.path)");
    exec_ignore_error(client, "CREATE INDEX FOR (f:File) ON (f.path)");
    exec_ignore_error(client, "CREATE INDEX FOR (c:Chunk) ON (c.chunk_id)");
    exec_ignore_error(
        client,
        &format!(
            "CREATE VECTOR INDEX FOR (c:Chunk) ON (c.embedding) OPTIONS {{dimension: {}, similarityFunction: 'cosine'}}",
            dim.max(1)
        ),
    );
    exec_ignore_error(
        client,
        "CALL db.idx.fulltext.createNodeIndex('Chunk', 'text')",
    );
    Ok(())
}

fn vector_dim_for_model(conn: &Connection, model_key: &str) -> Result<usize, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT dim
FROM project_chunk_vectors
WHERE model = ?1
LIMIT 1
"#,
        )
        .map_err(|e| format!("failed preparing vector dim query: {}", e))?;
    match stmt.query_row(params![model_key], |row| row.get::<_, i64>(0)) {
        Ok(dim) => Ok(dim.max(1) as usize),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(384),
        Err(e) => Err(format!("failed querying vector dim: {}", e)),
    }
}

fn query_id_score_rows(client: &mut RedisClient, query: &str) -> Result<Vec<(i64, f64)>, String> {
    let resp = client.graph_query(query)?;
    let rows = extract_graph_rows(resp)?;
    let mut out = Vec::new();
    for row in rows {
        if row.len() < 2 {
            continue;
        }
        let Some(id) = decode_cell_i64(&row[0]) else {
            continue;
        };
        let Some(score) = decode_cell_f64(&row[1]) else {
            continue;
        };
        out.push((id, score));
    }
    Ok(out)
}

fn extract_graph_rows(resp: Resp) -> Result<Vec<Vec<Resp>>, String> {
    let Resp::Array(Some(top)) = resp else {
        return Err("invalid GRAPH.QUERY response".to_string());
    };
    if top.len() < 2 {
        return Err("invalid GRAPH.QUERY payload shape".to_string());
    }
    let data = top
        .into_iter()
        .nth(1)
        .ok_or_else(|| "missing GRAPH.QUERY result set".to_string())?;
    match data {
        Resp::Array(Some(rows)) => {
            let mut out = Vec::new();
            for row in rows {
                if let Resp::Array(Some(cols)) = row {
                    out.push(cols);
                }
            }
            Ok(out)
        }
        _ => Ok(Vec::new()),
    }
}

fn decode_cell_i64(cell: &Resp) -> Option<i64> {
    let (marker, val) = typed_cell(cell)?;
    match marker {
        3 => decode_i64(val),
        5 => decode_f64(val).map(|v| v as i64),
        2 => decode_string(val).and_then(|s| s.parse::<i64>().ok()),
        _ => decode_i64(val)
            .or_else(|| decode_f64(val).map(|v| v as i64))
            .or_else(|| decode_string(val).and_then(|s| s.parse::<i64>().ok())),
    }
}

fn decode_cell_string(cell: &Resp) -> Option<String> {
    let (marker, val) = typed_cell(cell)?;
    match marker {
        2 => decode_string(val),
        3 => decode_i64(val).map(|v| v.to_string()),
        5 => decode_f64(val).map(|v| format!("{:.6}", v)),
        _ => decode_string(val)
            .or_else(|| decode_i64(val).map(|v| v.to_string()))
            .or_else(|| decode_f64(val).map(|v| format!("{:.6}", v))),
    }
}

fn decode_cell_f64(cell: &Resp) -> Option<f64> {
    let (marker, val) = typed_cell(cell)?;
    match marker {
        5 => decode_f64(val),
        3 => decode_i64(val).map(|v| v as f64),
        2 => decode_string(val).and_then(|s| s.parse::<f64>().ok()),
        _ => decode_f64(val)
            .or_else(|| decode_i64(val).map(|v| v as f64))
            .or_else(|| decode_string(val).and_then(|s| s.parse::<f64>().ok())),
    }
}

fn typed_cell(cell: &Resp) -> Option<(i64, &Resp)> {
    let Resp::Array(Some(parts)) = cell else {
        return None;
    };
    if parts.len() != 2 {
        return None;
    }
    let marker = decode_i64(&parts[0])?;
    Some((marker, &parts[1]))
}

fn decode_i64(val: &Resp) -> Option<i64> {
    match val {
        Resp::Integer(v) => Some(*v),
        Resp::Simple(s) => s.parse::<i64>().ok(),
        Resp::Bulk(Some(b)) => std::str::from_utf8(b).ok()?.parse::<i64>().ok(),
        _ => None,
    }
}

fn decode_f64(val: &Resp) -> Option<f64> {
    match val {
        Resp::Integer(v) => Some(*v as f64),
        Resp::Simple(s) => s.parse::<f64>().ok(),
        Resp::Bulk(Some(b)) => std::str::from_utf8(b).ok()?.parse::<f64>().ok(),
        _ => None,
    }
}

fn decode_string(val: &Resp) -> Option<String> {
    match val {
        Resp::Simple(s) => Some(s.clone()),
        Resp::Bulk(Some(b)) => String::from_utf8(b.clone()).ok(),
        _ => None,
    }
}

fn normalize_scores(rows: &[(i64, f64)]) -> HashMap<i64, f64> {
    if rows.is_empty() {
        return HashMap::new();
    }
    let mut dedup: HashMap<i64, f64> = HashMap::new();
    for (id, score) in rows {
        let prev = dedup.get(id).copied().unwrap_or(f64::NEG_INFINITY);
        if *score > prev {
            dedup.insert(*id, *score);
        }
    }
    if dedup.is_empty() {
        return HashMap::new();
    }
    let lo = dedup.values().copied().fold(f64::INFINITY, f64::min);
    let hi = dedup.values().copied().fold(f64::NEG_INFINITY, f64::max);
    if (hi - lo).abs() < f64::EPSILON {
        return dedup.into_keys().map(|id| (id, 1.0)).collect();
    }
    let span = hi - lo;
    dedup
        .into_iter()
        .map(|(id, score)| (id, ((score - lo) / span).clamp(0.0, 1.0)))
        .collect()
}

fn rrf_fuse(lists: &[Vec<i64>], limit: usize) -> Vec<i64> {
    let k = 60.0f64;
    let mut scores: HashMap<i64, f64> = HashMap::new();
    for ids in lists {
        for (rank, id) in ids.iter().enumerate() {
            let entry = scores.entry(*id).or_insert(0.0);
            *entry += 1.0 / (k + (rank as f64) + 1.0);
        }
    }
    let mut ranked: Vec<(i64, f64)> = scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked
        .into_iter()
        .take(limit.max(1))
        .map(|(id, _)| id)
        .collect()
}

fn exec_ignore_error(client: &mut RedisClient, query: &str) {
    let _ = client.graph_query(query);
}

fn blob_to_f32_vec(blob: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(blob.len() / 4);
    for chunk in blob.chunks_exact(4) {
        out.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

fn vector_literal(values: &[f32]) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(values.len());
    for value in values {
        if value.is_finite() {
            parts.push(format!("{:.8}", value));
        } else {
            parts.push("0.0".to_string());
        }
    }
    format!("[{}]", parts.join(","))
}

fn f64_literal(value: f64) -> String {
    if value.is_finite() {
        format!("{:.6}", value)
    } else {
        "0.0".to_string()
    }
}

fn cypher_string_literal(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + 2);
    out.push('\'');
    for ch in input.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out.push('\'');
    out
}

fn parse_falkor_url(raw: &str) -> Result<(String, u16), String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("empty FalkorDB url".to_string());
    }
    let no_scheme = trimmed
        .strip_prefix("falkor://")
        .or_else(|| trimmed.strip_prefix("redis://"))
        .unwrap_or(trimmed);
    let authority = no_scheme.split('/').next().unwrap_or(no_scheme);
    let host_port = authority
        .split('@')
        .next_back()
        .ok_or_else(|| format!("invalid FalkorDB URL '{}'", raw))?;
    let (host, port) = if let Some(idx) = host_port.rfind(':') {
        let host = host_port[..idx].trim();
        let port_str = host_port[idx + 1..].trim();
        let port = port_str
            .parse::<u16>()
            .map_err(|e| format!("invalid FalkorDB port '{}': {}", port_str, e))?;
        (host.to_string(), port)
    } else {
        (host_port.trim().to_string(), 6379u16)
    };
    if host.is_empty() {
        return Err(format!("invalid FalkorDB URL '{}': missing host", raw));
    }
    Ok((host, port))
}

fn read_resp(reader: &mut BufReader<TcpStream>) -> Result<Resp, String> {
    let mut prefix = [0u8; 1];
    reader
        .read_exact(&mut prefix)
        .map_err(|e| format!("failed reading RESP type prefix: {}", e))?;
    match prefix[0] {
        b'+' => Ok(Resp::Simple(read_line(reader)?)),
        b'-' => Ok(Resp::Error(read_line(reader)?)),
        b':' => {
            let n = read_line(reader)?
                .parse::<i64>()
                .map_err(|e| format!("failed parsing RESP integer: {}", e))?;
            Ok(Resp::Integer(n))
        }
        b'$' => {
            let len = read_line(reader)?
                .parse::<isize>()
                .map_err(|e| format!("failed parsing RESP bulk length: {}", e))?;
            if len < 0 {
                return Ok(Resp::Bulk(None));
            }
            let mut buf = vec![0u8; len as usize];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("failed reading RESP bulk body: {}", e))?;
            let mut crlf = [0u8; 2];
            reader
                .read_exact(&mut crlf)
                .map_err(|e| format!("failed reading RESP bulk trailer: {}", e))?;
            Ok(Resp::Bulk(Some(buf)))
        }
        b'*' => {
            let count = read_line(reader)?
                .parse::<isize>()
                .map_err(|e| format!("failed parsing RESP array length: {}", e))?;
            if count < 0 {
                return Ok(Resp::Array(None));
            }
            let mut out = Vec::with_capacity(count as usize);
            for _ in 0..count {
                out.push(read_resp(reader)?);
            }
            Ok(Resp::Array(Some(out)))
        }
        other => Err(format!("unsupported RESP prefix byte: {}", other)),
    }
}

fn read_line(reader: &mut BufReader<TcpStream>) -> Result<String, String> {
    let mut line = Vec::<u8>::new();
    reader
        .read_until(b'\n', &mut line)
        .map_err(|e| format!("failed reading RESP line: {}", e))?;
    if line.len() < 2 || line[line.len() - 2] != b'\r' || line[line.len() - 1] != b'\n' {
        return Err("malformed RESP line ending".to_string());
    }
    line.truncate(line.len() - 2);
    String::from_utf8(line).map_err(|e| format!("invalid UTF-8 in RESP line: {}", e))
}
