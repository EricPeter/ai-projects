#!/usr/bin/env bash
# Run every AI portfolio project in sequence, cleaning up fully between each one.
#
# Usage:
#   ./run_all.sh                   # run all projects
#   ./run_all.sh 21 22 23          # run specific project numbers
#   ./run_all.sh --dry-run         # show what would run, don't execute
#   ./run_all.sh --timeout 300     # per-project health-check timeout in seconds (default 180)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs/$(date +%Y-%m-%d_%H-%M-%S)"
TIMEOUT=180
DRY_RUN=false
SELECTED=()

BOLD='\033[1m';  DIM='\033[2m';    RESET='\033[0m'
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'

while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)  DRY_RUN=true; shift ;;
    --timeout)  TIMEOUT="$2"; shift 2 ;;
    [0-9]*)     SELECTED+=("$1"); shift ;;
    *)          echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"

declare -a PROJECTS=(
  "01-production-rag|docker|8000|/health||"
  "02-model-router|docker|8001|/health||"
  "03-multi-agent-research|python|||uv run python -m src.main 'Test query'|Requires ANTHROPIC_API_KEY and interactive human-approval gate"
  "04-eval-harness|python|||bash scripts/run_eval.sh|"
  "05-observability|docker|8005|/health||"
  "06-security-guardrails|docker|8006|/health||"
  "07-local-dev|docker|8007|/health||Requires local Ollama model pull (>4 GB)"
  "08-streaming-copilot|node|||npm run build|"
  "09-fine-tuning|python|||bash scripts/run_pipeline.sh|"
  "10-multi-tenant-saas|docker|8010|/health||"
  "11-cicd-ai|python|||echo 'CI/CD project — run via GitHub Actions'|Smoke test requires a live deployment target"
  "12-vector-db|docker|8012|/health||"
  "13-agent-memory|docker|8013|/health||"
  "14-inference-server|docker|8014|/health||"
  "15-hitl-workflow|docker|8015|/health||"
  "16-agentic-automation|docker|8016|/health||"
  "17-domain-benchmark|python|||uv run python scripts/run_benchmark.py|"
  "18-sunflower-translator|docker|8080|/health||"
  "19-uganda-health-bot|docker|8080|/health||"
  "20-model-drift-monitor|docker|8020|/health||"
  "21-model-serving|docker|8021|/health||"
  "22-ab-model-rollout|docker|8022|/health|bash scripts/run_experiment.sh 200|"
  "23-auto-retraining|docker|8023|/health|uv run python scripts/simulate_drift.py --clean 30 --drift 30 --interval 0.05|"
)

declare -a SUMMARY_NAMES=()
declare -a SUMMARY_STATUS=()
declare -a SUMMARY_TIMES=()

# ── Formatting helpers ────────────────────────────────────────────────────────

HR=$(printf '─%.0s' {1..62})
DHR=$(printf '═%.0s' {1..62})

# Write to terminal + master log
log() { echo -e "$*" | tee -a "$LOG_DIR/run_all.log"; }

# Thin full-width rule
hr()  { log "$HR"; }

# Section label: bold uppercase title + thin underline
section() {
  local label; label=$(echo "$1" | tr '[:lower:]' '[:upper:]')
  log ""
  log "  ${BOLD}${label}${RESET}"
  log "  $(printf '─%.0s' {1..58})"
}

# 4-space indent for content piped through (log file gets raw, terminal gets indented)
indent() { sed 's/^/    /'; }

# Format seconds as "1m 07s" or "43s"
fmt_time() {
  local s="$1"
  [[ $s -ge 60 ]] && printf "%dm %02ds" $((s / 60)) $((s % 60)) || printf "%ds" "$s"
}

# Dot-progress health check — prints on one line, ends with ✓ or ✗
wait_healthy() {
  local port="$1" path="$2" log_file="$3"
  local deadline=$((SECONDS + TIMEOUT)) start=$SECONDS elapsed

  printf "  ⏳  :${port}${path} "
  printf "  waiting :${port}${path} " >> "$log_file"

  while [[ $SECONDS -lt $deadline ]]; do
    if curl -sf "http://localhost:${port}${path}" > /dev/null 2>&1; then
      elapsed=$((SECONDS - start))
      printf " ${GREEN}✓${RESET}  (${elapsed}s)\n"
      printf " ✓  (${elapsed}s)\n" >> "$log_file"
      return 0
    fi
    printf "·"
    printf "·" >> "$log_file"
    sleep 3
  done

  printf " ${RED}✗  timed out${RESET}\n"
  printf " ✗  timed out\n" >> "$log_file"
  return 1
}

# ── Project runners ───────────────────────────────────────────────────────────

run_docker() {
  local dir="$1" port="$2" health_path="$3" demo="$4" log_file="$5"

  section "Build"
  # Full build output → log only (too verbose for terminal); show service table after
  if ! docker compose up --build -d >> "$log_file" 2>&1; then
    log "  ${RED}build failed — see $(basename "$log_file")${RESET}"
    return 1
  fi
  docker compose ps --format "table {{.Service}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null \
    | tee -a "$log_file" | indent || true

  # Health
  if [[ -n "$port" ]]; then
    section "Health"
    if ! wait_healthy "$port" "$health_path" "$log_file"; then
      log ""
      log "  ${RED}health check failed — dumping container logs:${RESET}"
      section "Container logs"
      docker compose logs --no-color --tail=40 2>&1 | tee -a "$log_file" | indent || true
      docker compose down -v --remove-orphans >> "$log_file" 2>&1 || true
      return 1
    fi
    log ""
    curl -sf "http://localhost:${port}${health_path}" 2>/dev/null \
      | python3 -m json.tool 2>/dev/null \
      | head -n 40 \
      | tee -a "$log_file" | indent \
      || log "    (non-JSON response)"
  fi

  # Tests (acceptance + regression + A/B)
  if [[ -d "tests" ]]; then
    section "Tests"
    BASE_URL="http://localhost:${port}" \
      uv run pytest tests/ -v --tb=short --no-header -q \
      2>&1 | tee -a "$log_file" | indent \
      || log "  ${YELLOW}⚠  some tests failed${RESET}"
  fi

  # Logs
  section "Container logs  (last 20 lines)"
  docker compose logs --no-color --tail=20 2>&1 \
    | tee -a "$log_file" | indent || true

  # Demo
  if [[ -n "$demo" ]]; then
    section "Demo"
    log "  ${DIM}\$ $demo${RESET}"
    log ""
    eval "$demo" 2>&1 \
      | tee -a "$log_file" | indent \
      || log "  ${YELLOW}⚠  demo exited non-zero${RESET}"
  fi

  # Cleanup
  section "Cleanup"
  docker compose down -v --remove-orphans 2>&1 \
    | tee -a "$log_file" | indent || true
  docker image prune -f --filter "label=com.docker.compose.project=$(basename "$dir")" \
    >> "$log_file" 2>&1 || true

  return 0
}

run_python() {
  local dir="$1" cmd="$2" log_file="$3"

  if [[ -f "pyproject.toml" ]] && command -v uv &>/dev/null; then
    section "Dependencies"
    uv sync 2>&1 | tee -a "$log_file" | indent || true
  fi

  section "Run"
  log "  ${DIM}\$ $cmd${RESET}"
  log ""
  if ! eval "$cmd" 2>&1 | tee -a "$log_file" | indent; then
    cleanup_python
    return 1
  fi

  if [[ -d "tests" ]]; then
    section "Tests"
    uv run pytest tests/ -v --tb=short --no-header -q \
      2>&1 | tee -a "$log_file" | indent \
      || log "  ${YELLOW}⚠  some tests failed${RESET}"
  fi

  cleanup_python
  return 0
}

cleanup_python() {
  rm -rf .venv __pycache__ .pytest_cache
  find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
  find . -name "*.pyc" -delete 2>/dev/null || true
}

run_node() {
  local cmd="$1" log_file="$2"

  section "Dependencies"
  npm install 2>&1 | tail -5 | tee -a "$log_file" | indent || true

  section "Run"
  log "  ${DIM}\$ $cmd${RESET}"
  log ""
  if ! eval "$cmd" 2>&1 | tee -a "$log_file" | indent; then
    rm -rf node_modules .next
    return 1
  fi
  rm -rf node_modules .next
  return 0
}

# ── Main loop ─────────────────────────────────────────────────────────────────

total=${#PROJECTS[@]}
[[ ${#SELECTED[@]} -gt 0 ]] && total=${#SELECTED[@]}
proj_idx=0
total_start=$SECONDS

log ""
log "$DHR"
printf -v _hdr "  AI Portfolio Runner%*s%s" $((62 - 19 - 8)) "" "$(date '+%Y-%m-%d  %H:%M:%S')"
log "${BOLD}${_hdr}${RESET}"
log "  ${total} projects   ·   logs → $LOG_DIR"
log "$DHR"

for entry in "${PROJECTS[@]}"; do
  IFS='|' read -r slug type health_port health_path demo skip_reason <<< "$entry"

  if [[ ${#SELECTED[@]} -gt 0 ]]; then
    num="${slug%%-*}"
    match=false
    for sel in "${SELECTED[@]}"; do
      [[ "$num" == "$sel" || "$num" == "0$sel" ]] && match=true
    done
    $match || continue
  fi

  proj_idx=$((proj_idx + 1))
  SUMMARY_NAMES+=("$slug")

  log ""
  hr
  printf -v _phdr "  [%02d/%02d]  %-42s%s" "$proj_idx" "$total" "$slug" "$(date +%H:%M:%S)"
  log "${BOLD}${_phdr}${RESET}"
  hr

  # Skip
  if [[ -n "$skip_reason" ]]; then
    log "  ${YELLOW}⏭  ${skip_reason}${RESET}"
    hr
    SUMMARY_STATUS+=("SKIP")
    SUMMARY_TIMES+=("-")
    continue
  fi

  dir="$SCRIPT_DIR/$slug"
  if [[ ! -d "$dir" ]]; then
    log "  ${RED}✗  directory not found${RESET}"
    hr
    SUMMARY_STATUS+=("FAIL")
    SUMMARY_TIMES+=("-")
    continue
  fi

  log_file="$LOG_DIR/${slug}.log"
  start_time=$SECONDS

  if $DRY_RUN; then
    case "$type" in
      docker) log "  up --build  →  health :${health_port}${health_path}  →  ${demo:-<no demo>}  →  down -v" ;;
      python) log "  $demo" ;;
      node)   log "  npm install  &&  $demo" ;;
    esac
    hr
    SUMMARY_STATUS+=("DRY")
    SUMMARY_TIMES+=("-")
    continue
  fi

  pushd "$dir" > /dev/null
  status="FAIL"
  case "$type" in
    docker) run_docker "$dir" "$health_port" "$health_path" "$demo" "$log_file" && status="PASS" || true ;;
    python) run_python "$dir" "$demo" "$log_file"                                && status="PASS" || true ;;
    node)   run_node   "$demo" "$log_file"                                       && status="PASS" || true ;;
  esac
  popd > /dev/null

  elapsed=$((SECONDS - start_time))
  SUMMARY_STATUS+=("$status")
  SUMMARY_TIMES+=("$(fmt_time $elapsed)")

  log ""
  if [[ "$status" == "PASS" ]]; then
    log "  ${GREEN}✓  done  $(fmt_time $elapsed)${RESET}"
  else
    log "  ${RED}✗  failed after $(fmt_time $elapsed)${RESET}  →  $log_file"
  fi
  hr
done

# ── Summary ───────────────────────────────────────────────────────────────────

log ""
log "$DHR"
log "${BOLD}  RESULTS${RESET}"
log "$DHR"
log ""

pass=0; fail=0; skip=0
for i in "${!SUMMARY_NAMES[@]}"; do
  name="${SUMMARY_NAMES[$i]}"
  st="${SUMMARY_STATUS[$i]}"
  t="${SUMMARY_TIMES[$i]}"
  case "$st" in
    PASS) icon="${GREEN}✓${RESET}"; colour="$RESET";  pass=$((pass + 1)); label="$t" ;;
    FAIL) icon="${RED}✗${RESET}";  colour="$RED";     fail=$((fail + 1)); label="failed" ;;
    SKIP) icon="${YELLOW}⏭${RESET}"; colour="$DIM";  skip=$((skip + 1)); label="skipped" ;;
    DRY)  icon="·";                colour="$DIM";                         label="dry-run" ;;
    *)    icon="?";                colour="$RESET";                        label="$t" ;;
  esac
  printf "  ${icon}  ${colour}%-44s${RESET}  %s\n" "$name" "$label" \
    | tee -a "$LOG_DIR/run_all.log"
done

total_elapsed=$((SECONDS - total_start))
log ""
log "  $(printf '─%.0s' {1..58})"
printf "  ${GREEN}✓ PASS %-4s${RESET}  ${RED}✗ FAIL %-4s${RESET}  ${YELLOW}⏭ SKIP %-4s${RESET}  total: %s\n" \
  "$pass" "$fail" "$skip" "$(fmt_time $total_elapsed)" \
  | tee -a "$LOG_DIR/run_all.log"
log ""
log "$DHR"
log ""
