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
#   STATUS=err   http=...      (or no_done_event / connection_err)
#
# Implemented via a streaming urllib call so we close the socket the instant we
# see a `done` event — keeps the script from hanging on SSE responses where the
# server doesn't close the connection after the final event.
make_request() {
  local factor="$1"
  local prompt
  prompt="$(build_prompt "$factor")"

  python3 - "$API_URL" "$API_KEY" "$prompt" <<'PY'
import json, sys, urllib.request, urllib.error

base, key, message = sys.argv[1], sys.argv[2], sys.argv[3]
body = json.dumps({"message": message}).encode("utf-8")
req = urllib.request.Request(
    base.rstrip("/") + "/chat/stream",
    data=body,
    method="POST",
    headers={"Content-Type": "application/json", "X-API-Key": key},
)
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line.startswith("data:"):
                continue
            payload = line[5:].lstrip()
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if d.get("type") != "done":
                continue
            # Found the terminal event — extract the fields we care about and
            # break out so the `with` block closes the socket immediately.
            u = d.get("usage", {})
            inp = int(u.get("input_tokens", 0))
            out = int(u.get("output_tokens", 0))
            cost = float(d.get("cost_usd", 0) or 0)
            hit = 1 if d.get("cache_hit") else 0
            model = d.get("model", "?")
            print(f"STATUS=ok model={model} input={inp} output={out} "
                  f"total={inp+out} cost={cost:.4f} cache_hit={hit}")
            break
        else:
            print("STATUS=err no_done_event")
except urllib.error.HTTPError as e:
    if e.code == 429:
        retry = e.headers.get("Retry-After", "unknown")
        print(f"STATUS=429 retry_after={retry}")
    else:
        print(f"STATUS=err http={e.code}")
except Exception as e:  # connection refused, timeout, etc.
    print(f"STATUS=err {type(e).__name__}")
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
      ;;
    STATUS=429*)
      # Pull "retry_after=..." into shell var.
      retry_after=""
      eval "$(echo "$result" | sed 's/^STATUS=429 //')"
      printf "%-3s %-22s → 429 rate limited (Retry-After: %ss)\n" \
        "$i" "$factor" "${retry_after:-?}"
      errors=$(( errors + 1 ))
      ;;
    *)
      printf "%-3s %-22s %s\n" "$i" "$factor" "→ ${result#STATUS=}"
      errors=$(( errors + 1 ))
      ;;
  esac

  # Pace requests so we don't trip the 20K/180s bucket on demo-pro.
  if (( i < ${#FACTORS[@]} )); then
    sleep "$DELAY"
  fi
done

echo "=================================================================="
echo "Summary"
echo "  Requests sent:    ${#FACTORS[@]}"
echo "  Successful:       $ok"
echo "  Cache hits:       $cache_hits"
echo "  Rate-limited/err: $errors"
echo "  Total input:      $total_input tokens"
echo "  Total output:     $total_output tokens"
echo "  Total tokens:     $(( total_input + total_output ))"
printf  "  Total cost:       \$%s\n" "$total_cost"
