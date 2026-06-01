#!/usr/bin/env bash
# burst_twelve.sh — send one detailed request per twelve-factor principle and
# report the real token usage from each response's SSE `done` event.
#
# Usage:
#   ./scripts/burst_twelve.sh
#
# Environment overrides:
#   API_URL=http://localhost:8080         service base URL
#   API_KEY=demo-pro-key                  X-API-Key header
#   DELAY=2                               seconds between requests
set -euo pipefail

API_URL="${API_URL:-http://localhost:8080}"
API_KEY="${API_KEY:-demo-free-key}"
DELAY="${DELAY:-2}"
RETRY_WAIT="${RETRY_WAIT:-5}"             # fixed seconds to wait after a 429
MAX_RETRIES="${MAX_RETRIES:-20}"          # cap on per-factor 429 retries
RETRY_429_COUNT=0                         # script-wide tally of 429s seen

# The twelve factors in canonical order.
FACTORS=(
  "Codebase"
  "Dependencies"
  "Config"
  "Backing services"
  "Build, release, run"
  "Processes"
  "Port binding"
  "Concurrency"
  "Disposability"
  "Dev/prod parity"
  "Logs"
  "Admin processes"
)

# Per-factor prompt template. The five-part structure is what forces a long
# completion (~600 output tokens) so each request lands around 1,500 total
# tokens once you add the ~870 input tokens from system + RAG context.
build_prompt() {
  local factor="$1"
  cat <<EOF
Provide a comprehensive analysis of the "$factor" principle from the twelve-factor app methodology. Cover all five of these in order, and be specific:

1. The precise definition of this factor and the original problem it solves.
2. Three concrete code or configuration examples that implement it correctly (give real snippets, not just descriptions).
3. Two common antipatterns that violate this factor, with an explanation of what breaks in production when each one is shipped.
4. How this factor interacts with at least two other twelve-factor principles — name them and explain the interaction.
5. Modern adaptations for serverless and Kubernetes environments (mention specific platforms or APIs).

Be thorough and keep the numbered structure throughout.
EOF
}

# Running totals.
total_input=0
total_output=0
total_cost=0
cache_hits=0
errors=0
ok=0

# Returns (via stdout) one line per request:
#   STATUS=ok    model=... input=... output=... cost=... cache_hit=...
#   STATUS=429   retry_after=...
#   STATUS=err   http=...      (or no_done_event)
#
# Implementation: spawn `curl` as a subprocess and read its stdout line-by-line.
# As soon as we see the `done` event (or detect a 429 from the status line) we
# terminate the curl process explicitly — that's the only reliable way to avoid
# hanging when the server keeps the SSE socket open after the final event.
make_request() {
  local factor="$1"
  local prompt
  prompt="$(build_prompt "$factor")"

  python3 - "$API_URL" "$API_KEY" "$prompt" <<'PY'
import json, signal, subprocess, sys

base, key, message = sys.argv[1], sys.argv[2], sys.argv[3]
body = json.dumps({"message": message})

cmd = [
    "curl", "-sN", "-i", "--no-buffer",
    "-X", "POST", base.rstrip("/") + "/chat/stream",
    "-H", "Content-Type: application/json",
    "-H", f"X-API-Key: {key}",
    "-d", body,
]
proc = subprocess.Popen(
    cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
)

status = None
retry_after = None
in_body = False
result = None
try:
    for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")

        # Status line, e.g. "HTTP/1.1 200 OK"
        if status is None and line.startswith("HTTP/"):
            parts = line.split()
            if len(parts) >= 2:
                status = parts[1]
            continue

        # Headers — until the blank line that separates them from the body.
        if not in_body:
            if line == "":
                in_body = True
                if status == "429":
                    result = f"STATUS=429 retry_after={retry_after or 'unknown'}"
                    break
                if status and not status.startswith("2"):
                    result = f"STATUS=err http={status}"
                    break
                continue
            if line.lower().startswith("retry-after:"):
                retry_after = line.split(":", 1)[1].strip()
            continue

        # SSE body — look for the terminal `done` event and stop there.
        if not line.startswith("data:"):
            continue
        try:
            d = json.loads(line[5:].lstrip())
        except json.JSONDecodeError:
            continue
        if d.get("type") != "done":
            continue
        u = d.get("usage", {})
        inp = int(u.get("input_tokens", 0))
        out = int(u.get("output_tokens", 0))
        cost = float(d.get("cost_usd", 0) or 0)
        hit = 1 if d.get("cache_hit") else 0
        model = d.get("model", "?")
        result = (f"STATUS=ok model={model} input={inp} output={out} "
                  f"total={inp+out} cost={cost:.4f} cache_hit={hit}")
        break
finally:
    # Make sure curl exits — it would otherwise sit on the SSE socket forever
    # waiting for the server to close. terminate -> kill if it lingers.
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=1)
    except ProcessLookupError:
        pass

print(result or "STATUS=err no_response")
PY
}

echo "Burst run against $API_URL (key: $API_KEY, ${DELAY}s spacing)"
echo "=================================================================="
printf "%-3s %-22s %-32s %7s %7s %7s %8s %5s\n" \
  "#" "FACTOR" "MODEL" "IN" "OUT" "TOTAL" "COST$" "CACHE"
echo "------------------------------------------------------------------"

i=0
for factor in "${FACTORS[@]}"; do
  i=$((i+1))
  attempt=0
  done_this_factor=0

  # Inner loop: re-issue the same request on 429 with a fixed RETRY_WAIT pause
  # until we succeed, hit MAX_RETRIES, or get a non-429 error.
  while (( done_this_factor == 0 )); do
    result="$(make_request "$factor")"

    case "$result" in
      STATUS=ok*)
        # Pull fields from "k=v k=v ..." into shell vars.
        eval "$(echo "$result" | sed 's/^STATUS=ok //')"
        printf "%-3s %-22s %-32s %7d %7d %7d %8.4f %5d\n" \
          "$i" "$factor" "$model" "$input" "$output" "$total" "$cost" "$cache_hit"
        total_input=$(( total_input + input ))
        total_output=$(( total_output + output ))
        total_cost=$(python3 -c "print(f'{$total_cost + $cost:.6f}')")
        cache_hits=$(( cache_hits + cache_hit ))
        ok=$(( ok + 1 ))
        done_this_factor=1
        ;;
      STATUS=429*)
        retry_after=""
        eval "$(echo "$result" | sed 's/^STATUS=429 //')"
        RETRY_429_COUNT=$(( RETRY_429_COUNT + 1 ))
        attempt=$(( attempt + 1 ))
        if (( attempt > MAX_RETRIES )); then
          printf "%-3s %-22s → 429 (Retry-After: %ss) — giving up after %d attempts\n" \
            "$i" "$factor" "${retry_after:-?}" "$MAX_RETRIES"
          errors=$(( errors + 1 ))
          done_this_factor=1
        else
          # Fixed RETRY_WAIT regardless of what the server suggests — we want to
          # keep poking at the bucket. Server's Retry-After is shown for info.
          printf "%-3s %-22s → 429 (Retry-After: %ss) — sleeping %ds and retrying (attempt %d/%d)\n" \
            "$i" "$factor" "${retry_after:-?}" "$RETRY_WAIT" "$attempt" "$MAX_RETRIES"
          sleep "$RETRY_WAIT"
        fi
        ;;
      *)
        printf "%-3s %-22s %s\n" "$i" "$factor" "→ ${result#STATUS=}"
        errors=$(( errors + 1 ))
        done_this_factor=1
        ;;
    esac
  done

  # Pace requests between *different* factors so we don't immediately re-trip
  # the bucket after a successful call.
  if (( i < ${#FACTORS[@]} )); then
    sleep "$DELAY"
  fi
done

echo "=================================================================="
echo "Summary"
echo "  Factors:          ${#FACTORS[@]}"
echo "  Successful:       $ok"
echo "  Cache hits:       $cache_hits"
echo "  Errored:          $errors"
echo "  429 retries:      $RETRY_429_COUNT  (slept ${RETRY_WAIT}s each)"
echo "  Total input:      $total_input tokens"
echo "  Total output:     $total_output tokens"
echo "  Total tokens:     $(( total_input + total_output ))"
printf  "  Total cost:       \$%s\n" "$total_cost"
