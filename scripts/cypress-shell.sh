#!/usr/bin/env bash

# Primary shell integration entrypoint for Cypress.
# Source this from ~/.bashrc.

_semcd_script_dir() {
  local src
  src="${BASH_SOURCE[0]}"
  while [[ -h "$src" ]]; do
    src="$(readlink "$src")"
  done
  cd "$(dirname "$src")" >/dev/null 2>&1 && pwd
}

_semcd_default_app_venv() {
  local repo_dir
  repo_dir="$(_semcd_repo_dir)"
  printf '%s\n' "$repo_dir/.cypress/venv"
}

_semcd_repo_dir() {
  cd "$(_semcd_script_dir)/.." >/dev/null 2>&1 && pwd
}

SEMCD_REPO_DIR="${SEMCD_REPO_DIR:-${CYPRESS_REPO_DIR:-$(_semcd_repo_dir)}}"
SEMCD_ROOT="${SEMCD_ROOT:-${CYPRESS_ROOT:-$HOME/c-projects}}"
SEMCD_EMBED_BACKEND="${SEMCD_EMBED_BACKEND:-${CYPRESS_EMBED_BACKEND:-local-hash}}"
SEMCD_EMBED_MODEL="${SEMCD_EMBED_MODEL:-${CYPRESS_EMBED_MODEL:-}}"
SEMCD_AUTO_BOOTSTRAP="${SEMCD_AUTO_BOOTSTRAP:-${CYPRESS_AUTO_BOOTSTRAP:-1}}"
SEMCD_INSTALL_DEPS="${SEMCD_INSTALL_DEPS:-${CYPRESS_INSTALL_DEPS:-1}}"
SEMCD_APP_VENV="${SEMCD_APP_VENV:-${CYPRESS_APP_VENV:-$(_semcd_default_app_venv)}}"
SEMCD_USE_DAEMON="${SEMCD_USE_DAEMON:-${CYPRESS_USE_DAEMON:-1}}"
SEMCD_API_HOST="${SEMCD_API_HOST:-${CYPRESS_API_HOST:-127.0.0.1}}"
SEMCD_API_PORT_DEFAULT="8765"
SEMCD_API_PORT="${SEMCD_API_PORT:-${CYPRESS_API_PORT:-$SEMCD_API_PORT_DEFAULT}}"
SEMCD_API_URL="${SEMCD_API_URL:-${CYPRESS_API_URL:-}}"
SEMCD_API_START_TIMEOUT="${SEMCD_API_START_TIMEOUT:-${CYPRESS_API_START_TIMEOUT:-8}}"
SEMCD_API_SEARCH_TIMEOUT="${SEMCD_API_SEARCH_TIMEOUT:-${CYPRESS_API_SEARCH_TIMEOUT:-60}}"
_SEMCD_DAEMON_BROKEN="${_SEMCD_DAEMON_BROKEN:-0}"
_SEMCD_PICK_RESULT="${_SEMCD_PICK_RESULT:-}"
_SEMCD_PICK_ACTION="${_SEMCD_PICK_ACTION:-cd_dir}"
_SEMCD_PICK_FILE="${_SEMCD_PICK_FILE:-}"

_semcd_debug_daemon() {
  [[ "${CYPRESS_DEBUG_DAEMON:-${SEMCD_DEBUG_DAEMON:-0}}" == "1" ]]
}

_semcd_cmd() {
  local cmd_path
  local data_dir_override config_override
  data_dir_override="${CYPRESS_DATA_DIR:-${SEMCD_DATA_DIR:-$SEMCD_REPO_DIR/.cypress}}"
  config_override="${CYPRESS_CONFIG:-${SEMCD_CONFIG:-$data_dir_override/config.toml}}"
  if [[ -x "$SEMCD_REPO_DIR/cypress" ]]; then
    CYPRESS_DATA_DIR="$data_dir_override" \
      SEMCD_DATA_DIR="$data_dir_override" \
      CYPRESS_CONFIG="$config_override" \
      SEMCD_CONFIG="$config_override" \
      "$SEMCD_REPO_DIR/cypress" "$@"
    return
  fi
  if [[ -x "$SEMCD_APP_VENV/bin/cypress" ]]; then
    CYPRESS_DATA_DIR="$data_dir_override" \
      SEMCD_DATA_DIR="$data_dir_override" \
      CYPRESS_CONFIG="$config_override" \
      SEMCD_CONFIG="$config_override" \
      "$SEMCD_APP_VENV/bin/cypress" "$@"
    return
  fi
  cmd_path="$(type -P cypress || true)"
  if [[ -n "$cmd_path" ]]; then
    CYPRESS_DATA_DIR="$data_dir_override" \
      SEMCD_DATA_DIR="$data_dir_override" \
      CYPRESS_CONFIG="$config_override" \
      SEMCD_CONFIG="$config_override" \
      "$cmd_path" "$@"
    return
  fi
  cmd_path="$(type -P semcd || true)"
  if [[ -n "$cmd_path" ]]; then
    CYPRESS_DATA_DIR="$data_dir_override" \
      SEMCD_DATA_DIR="$data_dir_override" \
      CYPRESS_CONFIG="$config_override" \
      SEMCD_CONFIG="$config_override" \
      "$cmd_path" "$@"
    return
  fi
  echo "cypress command not found; build/install the Rust CLI first (e.g. cargo build --release -p cypress)" >&2
  return 127
}

semcd_install() {
  _semcd_cmd install "$@"
}

_semcd_data_dir() {
  if [[ -n "${CYPRESS_DATA_DIR:-}" ]]; then
    printf '%s\n' "$CYPRESS_DATA_DIR"
  elif [[ -n "${SEMCD_DATA_DIR:-}" ]]; then
    printf '%s\n' "$SEMCD_DATA_DIR"
  else
    # Default to repo-local state directory, not ~/.local.
    printf '%s\n' "$SEMCD_REPO_DIR/.cypress"
  fi
}

_semcd_api_pid_path() {
  printf '%s\n' "$(_semcd_data_dir)/api.pid"
}

_semcd_api_port_path() {
  printf '%s\n' "$(_semcd_data_dir)/api.port"
}

_semcd_api_log_path() {
  printf '%s\n' "$(_semcd_data_dir)/api.log"
}

_semcd_saved_port() {
  local path val
  path="$(_semcd_api_port_path)"
  [[ -f "$path" ]] || return 1
  val="$(cat "$path" 2>/dev/null || true)"
  [[ "$val" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$val"
}

_semcd_api_url() {
  if [[ -n "${SEMCD_API_URL:-}" ]]; then
    printf '%s\n' "$SEMCD_API_URL"
  else
    local port saved
    port="$SEMCD_API_PORT"
    if [[ "$SEMCD_API_PORT" == "$SEMCD_API_PORT_DEFAULT" ]]; then
      if saved="$(_semcd_saved_port)"; then
        port="$saved"
      fi
    fi
    printf 'http://%s:%s\n' "$SEMCD_API_HOST" "$port"
  fi
}

_semcd_api_health() {
  if ! command -v curl >/dev/null 2>&1; then
    return 1
  fi
  curl -fsS --max-time 1 "$(_semcd_api_url)/health" >/dev/null 2>&1
}

_semcd_api_health_host_port() {
  local host="$1"
  local port="$2"
  if ! command -v curl >/dev/null 2>&1; then
    return 1
  fi
  curl -fsS --max-time 1 "http://${host}:${port}/health" >/dev/null 2>&1
}

_semcd_port_available() {
  local host="$1"
  local port="$2"
  if command -v lsof >/dev/null 2>&1; then
    if lsof -nP -iTCP:"${host}:${port}" -sTCP:LISTEN >/dev/null 2>&1; then
      return 1
    fi
    return 0
  fi
  if command -v nc >/dev/null 2>&1; then
    if nc -z "$host" "$port" >/dev/null 2>&1; then
      return 1
    fi
    return 0
  fi
  # No probing tool available; assume available and let startup validate.
  return 0
}

_semcd_find_free_port() {
  local host="$1"
  local start="$2"
  local port
  for ((port = start; port < start + 100; port++)); do
    if _semcd_port_available "$host" "$port"; then
      printf '%s\n' "$port"
      return 0
    fi
  done
  return 1
}

_semcd_launch_api() {
  local host="$1"
  local port="$2"
  local log_path="$3"
  local semcd_bin

  if [[ -x "$SEMCD_APP_VENV/bin/cypress" ]]; then
    nohup "$SEMCD_APP_VENV/bin/cypress" api --host "$host" --port "$port" >>"$log_path" 2>&1 &
    return 0
  fi

  if [[ -x "$SEMCD_APP_VENV/bin/semcd" ]]; then
    nohup "$SEMCD_APP_VENV/bin/semcd" api --host "$host" --port "$port" >>"$log_path" 2>&1 &
    return 0
  fi

  semcd_bin="$(type -P cypress || true)"
  if [[ -n "$semcd_bin" ]]; then
    nohup "$semcd_bin" api --host "$host" --port "$port" >>"$log_path" 2>&1 &
    return 0
  fi

  semcd_bin="$(type -P semcd || true)"
  if [[ -n "$semcd_bin" ]]; then
    nohup "$semcd_bin" api --host "$host" --port "$port" >>"$log_path" 2>&1 &
    return 0
  fi
  echo "cypress api launch failed: no cypress command found" >&2
  return 1
}

_semcd_start_api() {
  local url host port timeout_s
  url="$(_semcd_api_url)"
  if _semcd_api_health; then
    _SEMCD_DAEMON_BROKEN=0
    return 0
  fi
  if [[ -n "${SEMCD_API_URL:-}" ]]; then
    echo "cypress daemon: cannot auto-start custom API URL ($url)" >&2
    return 1
  fi
  host="$SEMCD_API_HOST"
  port="$SEMCD_API_PORT"
  timeout_s="${SEMCD_API_START_TIMEOUT:-8}"
  _semcd_cmd daemon start --host "$host" --port "$port" --timeout "$timeout_s" >/dev/null 2>&1 || return 1
  _semcd_api_health
}

_semcd_stop_api() {
  _semcd_cmd daemon stop >/dev/null 2>&1 || true
}

semcd_daemon() {
  _semcd_cmd daemon "$@"
}

_semcd_urlencode() {
  local input="$1"
  local output=""
  local i ch hex
  local LC_ALL=C
  for ((i = 0; i < ${#input}; i++)); do
    ch="${input:i:1}"
    case "$ch" in
      [a-zA-Z0-9.~_-])
        output+="$ch"
        ;;
      ' ')
        output+="%20"
        ;;
      *)
        printf -v hex '%02X' "'$ch"
        output+="%${hex}"
        ;;
    esac
  done
  printf '%s\n' "$output"
}

_semcd_json_escape() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//$'\n'/\\n}"
  s="${s//$'\r'/\\r}"
  s="${s//$'\t'/\\t}"
  printf '%s' "$s"
}

_semcd_record_selection_api() {
  local query="$1"
  local path="$2"
  local payload
  payload="{\"query\":\"$(_semcd_json_escape "$query")\",\"path\":\"$(_semcd_json_escape "$path")\"}"
  curl -fsS --max-time 2 -X POST \
    -H "Content-Type: application/json" \
    -d "$payload" \
    "$(_semcd_api_url)/select" >/dev/null 2>&1 || return 1
}

_semcd_write_picker_feed_script() {
  local script_path="$1"
  cat >"$script_path" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

view="${1:-projects}"
query="${2:-}"
api_url="${CYPRESS_PICK_API_URL:-}"
timeout_s="${CYPRESS_PICK_TIMEOUT:-60}"
verbose_metrics="${CYPRESS_PICK_VERBOSE_METRICS:-0}"

if [[ -z "$api_url" ]]; then
  exit 1
fi
if ! command -v curl >/dev/null 2>&1 || ! command -v jq >/dev/null 2>&1; then
  exit 1
fi

case "$view" in
  projects) limit=40 ;;
  files) limit=120 ;;
  *) exit 1 ;;
esac

encoded="$(printf '%s' "$query" | jq -sRr @uri)"
response="$(curl -fsS --max-time "$timeout_s" "${api_url}/search?q=${encoded}&limit=${limit}&view=${view}" 2>/dev/null || true)"
[[ -n "$response" ]] || exit 0

if [[ "$view" == "projects" ]]; then
  printf '%s' "$response" | jq -r '
    def clean: tostring | gsub("\\s+";" ");
    def num: (tonumber? // 0);
    def f3z:
      (num as $n
       | (($n * 1000) | round) as $m
       | (if $m < 0 then (-$m) else $m end) as $a
       | (($a / 1000) | floor | tostring) as $ip
       | (($a % 1000) | tostring) as $fp
       | (($fp | length) as $l | ("0" * (3 - $l)) + $fp) as $ff
       | (if $m < 0 then "-" else "" end) + $ip + "." + $ff);
    def homepath:
      tostring as $s
      | ($ENV.HOME // "") as $h
      | if ($h != "" and ($s | startswith($h + "/"))) then
          "~/" + ($s[(($h | length) + 1):])
        else
          $s
        end;
    def trunc($n):
      tostring as $s
      | if ($s | length) <= $n then
          $s
        else
          ($s[0:($n - 3)] + "...")
        end;
    def rpad($n):
      tostring as $s
      | if ($s | length) >= $n then
          ($s[0:$n])
        else
          $s + (" " * ($n - ($s | length)))
        end;
    def lpad($n):
      tostring as $s
      | if ($s | length) >= $n then
          $s
        else
          (" " * ($n - ($s | length))) + $s
        end;
    def preview_esc: gsub("\t";"    ");
    (.results // [])[]
    | (.path // "" | tostring) as $path
    | select($path != "")
    | ($path | homepath) as $short_path
    | ((.score // 0) | f3z) as $score
    | ((.semantic // 0) | f3z) as $sem
    | ((.lexical // 0) | f3z) as $lex
    | ((.frecency // 0) | f3z) as $fr
    | ((.graph // 0) | f3z) as $gr
    | (
        if (($ENV.CYPRESS_PICK_VERBOSE_METRICS // "0") == "1") then
          ("s:" + $sem + " l:" + $lex + " f:" + $fr + " g:" + $gr)
        else
          ("sem:" + $sem + " fr:" + $fr)
        end
      ) as $metrics
    | (
        "directory: " + $short_path + "\n"
        + "score: " + $score + " sem=" + $sem + " lex=" + $lex + " fr=" + $fr + " gr=" + $gr
        + (
            if ((.evidence // []) | length) > 0 then
              "\n\ntop chunks:\n"
              + (
                  (.evidence // [])[:4]
                  | to_entries
                  | map(
                      "  " + ((.key + 1)|tostring) + ". "
                      + ((.value.doc_rel_path // "-")|tostring)
                      + "#" + ((((.value.chunk_index // 0) | tonumber? // 0) | tostring))
                      + " score=" + ((.value.score // 0) | f3z)
                      + " rel=" + ((.value.relation // "direct") | tostring)
                      + "\n     " + ((.value.excerpt // "") | clean | trunc(180))
                    )
                  | join("\n\n")
                )
            else
              "\n\nno chunk evidence available"
            end
          )
      ) as $preview
    | (
        ($short_path | trunc(46) | rpad(46))
        + " | " + ($score | lpad(6))
        + " | " + ("dir" | rpad(3))
        + " | " + $metrics
      ) as $row
    | [
        ("D:" + $path),
        $row,
        "",
        "",
        "",
        ($preview | preview_esc)
      ]
    | @tsv
  '
else
  printf '%s' "$response" | jq -r '
    def clean: tostring | gsub("\\s+";" ");
    def num: (tonumber? // 0);
    def f3z:
      (num as $n
       | (($n * 1000) | round) as $m
       | (if $m < 0 then (-$m) else $m end) as $a
       | (($a / 1000) | floor | tostring) as $ip
       | (($a % 1000) | tostring) as $fp
       | (($fp | length) as $l | ("0" * (3 - $l)) + $fp) as $ff
       | (if $m < 0 then "-" else "" end) + $ip + "." + $ff);
    def homepath:
      tostring as $s
      | ($ENV.HOME // "") as $h
      | if ($h != "" and ($s | startswith($h + "/"))) then
          "~/" + ($s[(($h | length) + 1):])
        else
          $s
        end;
    def trunc($n):
      tostring as $s
      | if ($s | length) <= $n then
          $s
        else
          ($s[0:($n - 3)] + "...")
        end;
    def rpad($n):
      tostring as $s
      | if ($s | length) >= $n then
          ($s[0:$n])
        else
          $s + (" " * ($n - ($s | length)))
        end;
    def lpad($n):
      tostring as $s
      | if ($s | length) >= $n then
          $s
        else
          (" " * ($n - ($s | length))) + $s
        end;
    def preview_esc: gsub("\t";"    ");
    (.results // [])[]
    | (.path // "" | tostring) as $path
    | (.project_path // "" | tostring) as $project
    | select($path != "" and $project != "")
    | ((.doc_rel_path // "" | tostring) as $rel
       | if $rel == "" then ($path | split("/") | last) else $rel end) as $display
    | ($project | split("/") | last) as $proj_name
    | ($path | homepath) as $short_path
    | ($project | homepath) as $short_project
    | ((.score // 0) | f3z) as $score
    | ((.semantic // 0) | f3z) as $sem
    | ((.lexical // 0) | f3z) as $lex
    | ((.graph // 0) | f3z) as $gr
    | ((.quality // 0) | f3z) as $q
    | (
        if (($ENV.CYPRESS_PICK_VERBOSE_METRICS // "0") == "1") then
          ("s:" + $sem + " l:" + $lex + " g:" + $gr + " q:" + $q)
        else
          ("sem:" + $sem + " q:" + $q)
        end
      ) as $metrics
    | (
        "file: " + $short_path + "\n"
        + "project: " + $short_project + "\n"
        + "score: " + $score + " sem=" + $sem + " lex=" + $lex + " gr=" + $gr + " q=" + $q
        + "\n\nchunk excerpt:\n"
        + ((.excerpt // "") | clean | trunc(220))
        + (
            if ((.evidence // []) | length) > 0 then
              "\n\nrelated chunks:\n"
              + (
                  (.evidence // [])[:4]
                  | to_entries
                  | map(
                      "  " + ((.key + 1)|tostring) + ". "
                      + ((.value.doc_rel_path // "-")|tostring)
                      + "#" + ((((.value.chunk_index // 0) | tonumber? // 0) | tostring))
                      + " score=" + ((.value.score // 0) | f3z)
                      + " rel=" + ((.value.relation // "direct") | tostring)
                      + "\n     " + ((.value.excerpt // "") | clean | trunc(180))
                    )
                  | join("\n\n")
                )
            else
              ""
            end
          )
      ) as $preview
    | (
        ($display | trunc(46) | rpad(46))
        + " | " + ($score | lpad(6))
        + " | " + (("proj:" + ($proj_name | trunc(16))) | rpad(21))
        + " | " + $metrics
      ) as $row
    | [
        ("F:" + $path),
        $row,
        "",
        "",
        "",
        ($preview | preview_esc)
      ]
    | @tsv
  '
fi
EOF
  chmod +x "$script_path"
}

_semcd_pick_via_api() {
  local mode="${1:-auto}"
  shift || true
  local query="$*"
  local selected_output selected_path selected_line selected_key
  local pressed_key timeout_s feed_cmd current_view prompt header change_bind initial_lines
  local fzf_query fzf_key rest selected_line_candidate noninteractive_view lines
  _SEMCD_PICK_RESULT=""
  _SEMCD_PICK_ACTION="cd_dir"
  _SEMCD_PICK_FILE=""
  case "$mode" in
    dirs|files|auto) ;;
    *) mode="auto" ;;
  esac

  if ! command -v curl >/dev/null 2>&1 || ! command -v jq >/dev/null 2>&1; then
    if _semcd_debug_daemon; then
      echo "cypress daemon picker needs curl+jq; falling back to direct CLI picker..." >&2
    fi
    return 1
  fi

  timeout_s="${SEMCD_API_SEARCH_TIMEOUT:-60}"
  feed_cmd="$(mktemp -t cypress-feed.XXXXXX)" || return 1
  _semcd_write_picker_feed_script "$feed_cmd"

  if command -v fzf >/dev/null 2>&1 && [[ -t 0 && -t 1 ]]; then
    current_view="projects"
    if [[ "$mode" == "files" ]]; then
      current_view="files"
    fi
    while true; do
      if [[ "$current_view" == "files" ]]; then
        prompt='cypress[file]> '
        if [[ "${CYPRESS_PICK_VERBOSE_METRICS:-0}" == "1" ]]; then
          header="$(printf '%-46s | %6s | %-21s | %s' "file" "score" "project" "s/l/g/q") | type to re-query | Enter/c=cd-dir o=open-file p=dirs"
        else
          header="$(printf '%-46s | %6s | %-21s | %s' "file" "score" "project" "sem/q") | type to re-query | Enter/c=cd-dir o=open-file p=dirs"
        fi
      else
        prompt='cypress[dir]> '
        if [[ "${CYPRESS_PICK_VERBOSE_METRICS:-0}" == "1" ]]; then
          header="$(printf '%-46s | %6s | %-3s | %s' "project path" "score" "typ" "s/l/f/g") | type to re-query | Enter/c=cd-dir f=files o=open-file"
        else
          header="$(printf '%-46s | %6s | %-3s | %s' "project path" "score" "typ" "sem/freq") | type to re-query | Enter/c=cd-dir f=files o=open-file"
        fi
      fi
      change_bind="change:reload(sleep 0.18; \"$feed_cmd\" \"$current_view\" {q} || true)"
      initial_lines="$(
        CYPRESS_PICK_API_URL="$(_semcd_api_url)" \
        CYPRESS_PICK_TIMEOUT="$timeout_s" \
        "$feed_cmd" "$current_view" "$query" 2>/dev/null || true
      )"
      if [[ -z "$initial_lines" && "$mode" == "auto" ]]; then
        if [[ "$current_view" == "projects" ]]; then
          current_view="files"
        else
          current_view="projects"
        fi
        continue
      fi
      selected_output="$(
        CYPRESS_PICK_API_URL="$(_semcd_api_url)" \
        CYPRESS_PICK_TIMEOUT="$timeout_s" \
        printf '%s\n' "$initial_lines" | fzf \
          --disabled \
          --print-query \
          --expect=o,c,ctrl-o,f,p,tab,btab,ctrl-f,ctrl-d \
          --bind="$change_bind" \
          --height=70% \
          --layout=reverse \
          --border \
          --delimiter=$'\t' \
          --with-nth=2 \
          --prompt="$prompt" \
          --header="$header" \
          --query="$query" \
          --preview='printf "%b\n" {6}' \
          --preview-window=down,50%,wrap,border-top \
          --no-sort
      )" || {
        rm -f "$feed_cmd"
        return 1
      }

      fzf_query="${selected_output%%$'\n'*}"
      rest="${selected_output#*$'\n'}"
      if [[ "$rest" == "$selected_output" ]]; then
        rm -f "$feed_cmd"
        return 1
      fi
      if [[ "$rest" == *$'\n'* ]]; then
        fzf_key="${rest%%$'\n'*}"
        selected_line_candidate="${rest#*$'\n'}"
        selected_line_candidate="${selected_line_candidate%%$'\n'*}"
      else
        fzf_key="$rest"
        selected_line_candidate=""
      fi
      query="$fzf_query"

      case "$fzf_key" in
        f|tab|ctrl-f)
          current_view="files"
          continue
          ;;
        p|btab|ctrl-d)
          current_view="projects"
          continue
          ;;
      esac

      if [[ -z "$selected_line_candidate" ]]; then
        rm -f "$feed_cmd"
        return 1
      fi
      pressed_key="$fzf_key"
      selected_line="$selected_line_candidate"
      break
    done
    rm -f "$feed_cmd"

    selected_key="${selected_line%%$'\t'*}"
    if [[ -z "$selected_key" ]]; then
      return 1
    fi
    if [[ "$selected_key" == D:* ]]; then
      selected_path="${selected_key#D:}"
      _SEMCD_PICK_ACTION="cd_dir"
      _SEMCD_PICK_FILE=""
    elif [[ "$selected_key" == F:* ]]; then
      _SEMCD_PICK_FILE="${selected_key#F:}"
      if [[ "$pressed_key" == "o" || "$pressed_key" == "ctrl-o" ]]; then
        selected_path="$_SEMCD_PICK_FILE"
        _SEMCD_PICK_ACTION="open_file"
      else
        selected_path="$(dirname "$_SEMCD_PICK_FILE")"
        _SEMCD_PICK_ACTION="cd_dir"
      fi
    else
      return 1
    fi
  else
    noninteractive_view="projects"
    if [[ "$mode" == "files" ]]; then
      noninteractive_view="files"
    fi
    lines="$(
      CYPRESS_PICK_API_URL="$(_semcd_api_url)" \
      CYPRESS_PICK_TIMEOUT="$timeout_s" \
      "$feed_cmd" "$noninteractive_view" "$query" 2>/dev/null || true
    )"
    if [[ -z "$lines" && "$mode" == "auto" ]]; then
      lines="$(
        CYPRESS_PICK_API_URL="$(_semcd_api_url)" \
        CYPRESS_PICK_TIMEOUT="$timeout_s" \
        "$feed_cmd" files "$query" 2>/dev/null || true
      )"
    fi
    rm -f "$feed_cmd"
    selected_line="$(printf '%s\n' "$lines" | sed -n '1p')"
    selected_key="${selected_line%%$'\t'*}"
    if [[ -z "$selected_key" ]]; then
      echo "cypress: no semantic results for '$query'" >&2
      return 1
    fi
    if [[ "$selected_key" == D:* ]]; then
      selected_path="${selected_key#D:}"
      _SEMCD_PICK_ACTION="cd_dir"
      _SEMCD_PICK_FILE=""
    elif [[ "$selected_key" == F:* ]]; then
      _SEMCD_PICK_FILE="${selected_key#F:}"
      selected_path="$(dirname "$_SEMCD_PICK_FILE")"
      _SEMCD_PICK_ACTION="cd_dir"
    else
      return 1
    fi
  fi

  [[ -n "$selected_path" ]] || return 1
  if [[ "$_SEMCD_PICK_ACTION" == "open_file" && -n "$_SEMCD_PICK_FILE" ]]; then
    _semcd_record_selection_api "$query" "$(dirname "$_SEMCD_PICK_FILE")" || true
  else
    _semcd_record_selection_api "$query" "$selected_path" || true
  fi
  _SEMCD_PICK_RESULT="$selected_path"
  return 0
}

_semcd_pick() {
  local mode="${1:-auto}"
  shift || true
  local query="$*"
  local selected tmp_path
  _SEMCD_PICK_RESULT=""
  _SEMCD_PICK_ACTION="cd_dir"
  _SEMCD_PICK_FILE=""
  case "$mode" in
    dirs|files|auto) ;;
    *) mode="auto" ;;
  esac
  if [[ "$SEMCD_USE_DAEMON" == "1" && "${_SEMCD_DAEMON_BROKEN:-0}" != "1" ]]; then
    if _semcd_start_api && _semcd_api_health; then
      _SEMCD_DAEMON_BROKEN=0
      _semcd_pick_via_api "$mode" "$query" && return 0
    else
      _SEMCD_DAEMON_BROKEN=1
      if _semcd_debug_daemon; then
        echo "cypress daemon unavailable; using direct CLI picker for this shell session." >&2
      fi
    fi
  fi

  if [[ "$mode" == "files" ]]; then
    echo "cypress: file picker mode requires the daemon/API path. Run: cypress daemon start" >&2
    return 1
  fi

  tmp_path="$(mktemp -t cypress-pick.XXXXXX)"
  if _semcd_cmd pick --query "$query" --emit-path-file "$tmp_path"; then
    selected="$(cat "$tmp_path" 2>/dev/null || true)"
    rm -f "$tmp_path"
    selected="${selected%$'\n'}"
    if [[ -z "$selected" ]]; then
      echo "cypress: no result selected for '$query'" >&2
      return 1
    fi
    _SEMCD_PICK_RESULT="$selected"
    return 0
  fi
  rm -f "$tmp_path"
  return 1
}

_semcd_open_file() {
  local file="$1"
  if [[ -z "$file" || ! -f "$file" ]]; then
    echo "cypress: selected file does not exist: $file" >&2
    return 1
  fi
  if [[ -n "${VISUAL:-}" ]]; then
    "${VISUAL}" "$file"
    return $?
  fi
  if [[ -n "${EDITOR:-}" ]]; then
    "${EDITOR}" "$file"
    return $?
  fi
  if command -v open >/dev/null 2>&1; then
    open "$file"
    return $?
  fi
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$file" >/dev/null 2>&1
    return $?
  fi
  echo "cypress: no editor configured (set \$EDITOR or \$VISUAL)" >&2
  return 1
}

semcd_bootstrap() {
  local do_index=1
  local do_install=1
  local root="$SEMCD_ROOT"
  local backend="$SEMCD_EMBED_BACKEND"
  local model="${SEMCD_EMBED_MODEL:-}"
  local retrieval=""
  local local_dim=""

  local i=1
  while [[ $i -le $# ]]; do
    local arg="${!i}"
    case "$arg" in
      --no-index)
        do_index=0
        ;;
      --no-install)
        do_install=0
        ;;
      --root)
        i=$((i + 1))
        if [[ $i -gt $# ]]; then
          echo "cypress bootstrap: --root expects a value" >&2
          return 2
        fi
        root="${!i}"
        ;;
      --embed-backend)
        i=$((i + 1))
        if [[ $i -gt $# ]]; then
          echo "cypress bootstrap: --embed-backend expects a value" >&2
          return 2
        fi
        backend="${!i}"
        ;;
      --embed-model)
        i=$((i + 1))
        if [[ $i -gt $# ]]; then
          echo "cypress bootstrap: --embed-model expects a value" >&2
          return 2
        fi
        model="${!i}"
        ;;
      --retrieval-backend)
        i=$((i + 1))
        if [[ $i -gt $# ]]; then
          echo "cypress bootstrap: --retrieval-backend expects a value" >&2
          return 2
        fi
        retrieval="${!i}"
        ;;
      --local-embed-dim)
        i=$((i + 1))
        if [[ $i -gt $# ]]; then
          echo "cypress bootstrap: --local-embed-dim expects a value" >&2
          return 2
        fi
        local_dim="${!i}"
        ;;
      *)
        echo "cypress bootstrap: unknown option '$arg'" >&2
        return 2
        ;;
    esac
    i=$((i + 1))
  done

  if [[ -z "$model" ]]; then
    if [[ "$backend" == "ollama" ]]; then
      model="qwen3-embedding"
    else
      model="sentence-transformers/all-MiniLM-L6-v2"
    fi
  fi

  if [[ "$do_install" == "1" ]]; then
    _semcd_cmd install || return 1
  fi

  local -a init_args
  init_args=(init --root "$root" --embed-backend "$backend" --embed-model "$model")
  if [[ -n "$retrieval" ]]; then
    init_args+=(--retrieval-backend "$retrieval")
  fi
  if [[ -n "$local_dim" ]]; then
    init_args+=(--local-embed-dim "$local_dim")
  fi
  _semcd_cmd "${init_args[@]}" || return 1

  if [[ "$do_index" == "1" ]]; then
    _semcd_cmd index || return 1
  fi
}

_semcd_config_path() {
  if [[ -n "${CYPRESS_CONFIG:-}" ]]; then
    printf '%s\n' "$CYPRESS_CONFIG"
  elif [[ -n "${SEMCD_CONFIG:-}" ]]; then
    printf '%s\n' "$SEMCD_CONFIG"
  else
    # Default to repo-local config, not ~/.config.
    printf '%s\n' "$SEMCD_REPO_DIR/.cypress/config.toml"
  fi
}

_semcd_db_path() {
  local data_dir_override="${CYPRESS_DATA_DIR:-${SEMCD_DATA_DIR:-}}"
  if [[ -n "$data_dir_override" ]]; then
    printf '%s\n' "$data_dir_override/cypress.db"
  else
    printf '%s\n' "$SEMCD_REPO_DIR/.cypress/cypress.db"
  fi
}

_semcd_setup_complete() {
  local cfg db
  cfg="$(_semcd_config_path)"
  db="$(_semcd_db_path)"
  [[ -f "$cfg" && -f "$db" ]]
}

_semcd_ensure_initialized() {
  if _semcd_setup_complete; then
    return 0
  fi
  _semcd_cmd init --root "$SEMCD_ROOT"
}

_semcd_ensure_bootstrap() {
  if [[ "$SEMCD_AUTO_BOOTSTRAP" != "1" ]]; then
    return 0
  fi
  if [[ "${_SEMCD_BOOTSTRAPPED:-0}" == "1" ]]; then
    return 0
  fi
  if _semcd_setup_complete; then
    if command -v fzf >/dev/null 2>&1 || [[ "$SEMCD_INSTALL_DEPS" != "1" ]]; then
      _SEMCD_BOOTSTRAPPED=1
      return 0
    fi
  fi
  semcd_bootstrap || return 1
  _SEMCD_BOOTSTRAPPED=1
}

cypress() {
  local sub="${1:-}"

  case "$sub" in
    init)
      shift || true
      _semcd_cmd init "$@"
      return
      ;;
    add|del|roots|refresh|index|watch|doctor|api|neighbors|search|pick|mcp)
      shift || true
      _semcd_ensure_initialized || return
      _semcd_cmd "$sub" "$@"
      return
      ;;
    graph|bench|self-test|version|legacy)
      shift || true
      _semcd_cmd "$sub" "$@"
      return
      ;;
    help|-h|--help)
      _semcd_cmd --help
      return
      ;;
    install)
      shift || true
      semcd_install "$@"
      return
      ;;
    daemon)
      shift || true
      semcd_daemon "$@"
      return
      ;;
    bootstrap)
      shift || true
      semcd_bootstrap "$@"
      return
      ;;
  esac

  local target
  local mode="auto"
  if [[ "${1:-}" == "--files" || "${1:-}" == "-f" ]]; then
    mode="files"
    shift || true
  elif [[ "${1:-}" == "--dirs" || "${1:-}" == "--directories" || "${1:-}" == "-d" ]]; then
    mode="dirs"
    shift || true
  fi

  local query="$*"
  if [[ -z "$query" ]]; then
    if [[ ! -t 0 ]]; then
      echo "usage: cypress <semantic query> (or run from an interactive shell)" >&2
      return 2
    fi
    read -r -e -p "cypress query: " query
    [[ -n "$query" ]] || return
  fi
  _semcd_ensure_bootstrap || return
  _semcd_pick "$mode" "$query" || {
    if [[ "$mode" == "files" && "$SEMCD_USE_DAEMON" != "1" ]]; then
      return 1
    fi
    echo "cypress: no result selected for '$query'" >&2
    return 1
  }
  target="${_SEMCD_PICK_RESULT:-}"
  local action="${_SEMCD_PICK_ACTION:-cd_dir}"
  local picked_file="${_SEMCD_PICK_FILE:-}"
  if [[ -z "$target" ]]; then
    echo "cypress: no result selected for '$query'" >&2
    return 1
  fi
  if [[ "$action" == "open_file" ]]; then
    _semcd_open_file "${picked_file:-$target}" || return
    return 0
  fi
  if [[ -n "$target" && -d "$target" ]]; then
    cd "$target" || return
  else
    echo "cypress: selected path is not a directory: $target" >&2
    return 1
  fi
}

semcd_index() {
  _semcd_cmd index
}

semcd_watch() {
  _semcd_cmd watch "$@"
}

semcd_refresh() {
  _semcd_cmd refresh "$@"
}

# Legacy alias for older shell usage.
s() {
  cypress "$@"
}

cypress_install() {
  semcd_install "$@"
}

cypress_bootstrap() {
  semcd_bootstrap "$@"
}

cypress_index() {
  semcd_index "$@"
}

cypress_watch() {
  semcd_watch "$@"
}

cypress_refresh() {
  semcd_refresh "$@"
}

cypress_daemon() {
  semcd_daemon "$@"
}

sd() {
  cypress --dirs "$@"
}

sf() {
  cypress --files "$@"
}

_semcd_alt_s_widget() {
  local q target
  _semcd_ensure_bootstrap || return
  read -r -e -p "cypress query: " q
  _semcd_pick "$q" || return
  target="${_SEMCD_PICK_RESULT:-}"
  if [[ -n "$target" && -d "$target" ]]; then
    cd "$target" || return
  fi
}

# Optional Alt-s keybind for bash.
if [[ -n "${BASH_VERSION:-}" && -t 0 ]]; then
  bind -x '"\es":"_semcd_alt_s_widget"' >/dev/null 2>&1 || true
fi
