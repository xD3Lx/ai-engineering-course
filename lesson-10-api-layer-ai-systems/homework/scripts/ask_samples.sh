#!/usr/bin/env bash
# ask_samples.sh — fire 20 short, natural twelve-factor questions at
# /chat/stream and report the real usage from each response's SSE `done` event.
#
# Unlike burst_twelve.sh (one heavy prompt per factor, built to stress the rate
# limiter), this sends questions a real user might ask. Handy for smoke-testing
# RAG answers, warming the semantic cache, and eyeballing latency / cost.
#
# Usage:
#   ./scripts/ask_samples.sh
#   SHOW_ANSWERS=1 ./scripts/ask_samples.sh
#
# Environment overrides:
#   API_URL=http://localhost:8080         service base URL
#   API_KEY=demo-pro-key                  X-API-Key header
#   DELAY=1                               seconds between requests
#   SHOW_ANSWERS=1                        also print each streamed answer
set -euo pipefail

API_URL="${API_URL:-http://localhost:8080}"
API_KEY="${API_KEY:-demo-pro-key}"
DELAY="${DELAY:-1}"
RETRY_WAIT="${RETRY_WAIT:-5}"             # fixed seconds to wait after a 429
MAX_RETRIES="${MAX_RETRIES:-10}"          # cap on per-question 429 retries
SHOW_ANSWERS="${SHOW_ANSWERS:-0}"         # 1 = print answer text too
RETRY_429_COUNT=0                         # script-wide tally of 429s seen

# 20 sample questions about the twelve-factor app methodology — short and
# answerable from data/twelve.md.
QUESTIONS=(
  "What is the twelve-factor app methodology and what problem does it solve?"
  "What does the Codebase factor say about the relationship between a codebase and an app?"
  "How should a twelve-factor app declare and isolate its dependencies?"
  "Why does the Config factor recommend storing configuration in environment variables?"
  "How does the twelve-factor model treat backing services like databases and queues?"
  "Explain the three stages of the build, release, run factor and why they're kept separate."
  "Why should processes in a twelve-factor app be stateless and share-nothing?"
  "What does 'export services via port binding' mean and how does it make an app self-contained?"
  "How does the Concurrency factor use the process model to scale an app out?"
  "What does the Disposability factor require around startup and shutdown?"
  "What is dev/prod parity and which gaps does it try to keep small?"
  "How should a twelve-factor app handle logs, and why shouldn't it manage log files itself?"
  "How should admin and management tasks be run under the twelve-factor methodology?"
  "Why is storing session state in process memory considered a violation of the factors?"
  "How do the twelve factors apply to serverless and Kubernetes deployments?"
  "What's the recommended way to run database migrations in a twelve-factor app?"
  "Why separate config that varies between deploys from config that's constant?"
  "What are the risks of bundling a web server into the app instead of using port binding?"
  "How does treating backing services as attached resources improve portability?"
  "How do the twelve factors together support continuous deployment and horizontal scaling?"
)

# Send one question and stream its SSE response. Emits one line on stdout:
#   STATUS=ok model=... input=... output=... total=... cost=... cache_hit=... latency=... ttft=...
#   STATUS=429 retry_after=...
#   STATUS=err http=...            (or no_response)
# When SHOW_ANSWERS=1, a second line "ANSWER=<text>" follows an ok result.
#
# Implementation: spawn curl as a subprocess and read its stdout line-by-line.
# IMPORTANT: we capture the `done` event but keep reading until the server
# closes the socket (EOF). The handler runs `log_usage()` *after* yielding the
# done event, so disconnecting early cancels that DB write. Draining to EOF lets
# the server finish logging. `--max-time` is a safety cap against a hung socket.
make_request() {
  local question="$1"

  python3 - "$API_URL" "$API_KEY" "$question" "$SHOW_ANSWERS" <<'PY'
import json, sys

base, key, message, show = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
import subprocess
body = json.dumps({"message": message})

cmd = [
    "curl", "-sN", "-i", "--no-buffer", "-m", "120",
    "-X", "POST", base.rstrip("/") + "/chat/stream",
    "-H", "Content-Type: application/json",
    "-H", f"X-API-Key: {key}",
    "-d", body,
]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

status = None
retry_after = None
in_body = False
result = None
answer_parts = []
try:
    for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")

        # Status line, e.g. "HTTP/1.1 200 OK"
        if status is None and line.startswith("HTTP/"):
            parts = line.split()
            if len(parts) >= 2:
                status = parts[1]
            continue

        # Headers — until the blank line separating them from the body.
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

        # SSE body — collect tokens (if asked) and stop at the `done` event.
        if not line.startswith("data:"):
            continue
        try:
            d = json.loads(line[5:].lstrip())
        except json.JSONDecodeError:
            continue
        etype = d.get("type")
        if etype == "token":
            if show:
                answer_parts.append(d.get("content", ""))
            continue
        if etype != "done":
            continue
        u = d.get("usage", {}) or {}
        inp = int(u.get("input_tokens", 0))
        out = int(u.get("output_tokens", 0))
        cost = float(d.get("cost_usd", 0) or 0)
        hit = 1 if d.get("cache_hit") else 0
        model = d.get("model", "?")
        lat = d.get("latency_ms", "")
        ttft = d.get("ttft_ms", "")
        result = (f"STATUS=ok model={model} input={inp} output={out} "
                  f"total={inp+out} cost={cost:.4f} cache_hit={hit} "
                  f"latency={lat} ttft={ttft}")
        # Keep reading — do NOT break. The server runs log_usage() after the
        # done event; draining to EOF lets that DB write complete instead of
        # being cancelled by an early client disconnect.
        continue
finally:
    # By now the stream has reached EOF (server closed it) in the normal case;
    # terminate is just cleanup for the --max-time / error paths.
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=1)
    except ProcessLookupError:
        pass

print(result or "STATUS=err no_response")
if show and result and result.startswith("STATUS=ok"):
    print("ANSWER=" + "".join(answer_parts).strip().replace("\n", " "))
PY
}

echo "Asking ${#QUESTIONS[@]} sample questions against $API_URL (key: $API_KEY, ${DELAY}s spacing)"
echo "==============================================================================================="
printf "%-3s %-30s %6s %6s %6s %8s %8s %6s %5s\n" \
  "#" "MODEL" "IN" "OUT" "TOTAL" "COST$" "LAT(ms)" "TTFT" "CACHE"
echo "-----------------------------------------------------------------------------------------------"

total_input=0
total_output=0
total_cost=0
cache_hits=0
errors=0
ok=0

i=0
for question in "${QUESTIONS[@]}"; do
  i=$((i+1))
  attempt=0
  done_this=0

  while (( done_this == 0 )); do
    output="$(make_request "$question")"
    result="$(printf '%s\n' "$output" | head -n1)"
    answer="$(printf '%s\n' "$output" | sed -n 's/^ANSWER=//p')"

    case "$result" in
      STATUS=ok*)
        # Pull "k=v k=v ..." fields into shell vars.
        eval "$(echo "$result" | sed 's/^STATUS=ok //')"
        printf "%-3s %-30s %6d %6d %6d %8.4f %8s %6s %5s\n" \
          "$i" "${model:0:30}" "$input" "$output" "$total" "$cost" \
          "${latency:-?}" "${ttft:-?}" "$([ "$cache_hit" = "1" ] && echo hit || echo -)"
        if [ "$SHOW_ANSWERS" = "1" ]; then
          echo "    Q: $question"
          echo "    A: $answer"
          echo
        fi
        total_input=$(( total_input + input ))
        total_output=$(( total_output + output ))
        total_cost=$(python3 -c "print(f'{$total_cost + $cost:.6f}')")
        cache_hits=$(( cache_hits + cache_hit ))
        ok=$(( ok + 1 ))
        done_this=1
        ;;
      STATUS=429*)
        retry_after=""
        eval "$(echo "$result" | sed 's/^STATUS=429 //')"
        RETRY_429_COUNT=$(( RETRY_429_COUNT + 1 ))
        attempt=$(( attempt + 1 ))
        if (( attempt > MAX_RETRIES )); then
          printf "%-3s %-30s → 429 (Retry-After: %ss) — giving up after %d attempts\n" \
            "$i" "(question $i)" "${retry_after:-?}" "$MAX_RETRIES"
          errors=$(( errors + 1 ))
          done_this=1
        else
          printf "%-3s %-30s → 429 (Retry-After: %ss) — sleeping %ds (attempt %d/%d)\n" \
            "$i" "(question $i)" "${retry_after:-?}" "$RETRY_WAIT" "$attempt" "$MAX_RETRIES"
          sleep "$RETRY_WAIT"
        fi
        ;;
      *)
        printf "%-3s %-30s %s\n" "$i" "(question $i)" "→ ${result#STATUS=}"
        errors=$(( errors + 1 ))
        done_this=1
        ;;
    esac
  done

  if (( i < ${#QUESTIONS[@]} )); then
    sleep "$DELAY"
  fi
done

echo "==============================================================================================="
echo "Summary"
echo "  Questions:        ${#QUESTIONS[@]}"
echo "  Successful:       $ok"
echo "  Cache hits:       $cache_hits"
echo "  Errored:          $errors"
echo "  429 retries:      $RETRY_429_COUNT  (slept ${RETRY_WAIT}s each)"
echo "  Total input:      $total_input tokens"
echo "  Total output:     $total_output tokens"
echo "  Total tokens:     $(( total_input + total_output ))"
printf  "  Total cost:       \$%s\n" "$total_cost"
