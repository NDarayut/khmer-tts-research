#!/usr/bin/env bash
# Fetch one large file over N parallel byte-range connections, then concatenate.
#
# Hugging Face serves this machine at ~52 kB/s on a single connection but
# ~150-180 kB/s per connection when several run at once, so the download is
# throttled per-connection rather than by the link. Sixteen ranges turns a
# 16-hour pull into about an hour. huggingface_hub's own downloader opens one
# connection per *file*, which is why it stalled here.
#
# THE THING THAT BIT US: a dropped connection makes `curl -r` return a SHORT
# range with exit status 0. Concatenating those gives a file of plausible size
# that is quietly corrupt in the middle -- and a truncated parquet still reports
# the right row count from its footer, so a metadata check passes and only
# decompressing a row group fails. Every part is therefore size-checked and
# refetched until complete, and the assembled file is only written once all
# parts are exactly the right length.
#
#   finetune/fetch_parallel.sh <url> <output-path> [n_parts] [max_rounds]
set -uo pipefail

URL="$1"; OUT="$2"; PARTS="${3:-16}"; ROUNDS="${4:-8}"
TMP="${OUT}.parts"

SIZE=$(curl -sIL "$URL" | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}')
[ -n "$SIZE" ] || { echo "could not determine size of $URL" >&2; exit 1; }

if [ -f "$OUT" ] && [ "$(stat -c%s "$OUT")" = "$SIZE" ]; then
  echo "already complete: $(basename "$OUT")"; exit 0
fi

mkdir -p "$TMP"
CHUNK=$(( (SIZE + PARTS - 1) / PARTS ))
echo "$(basename "$OUT"): $SIZE bytes in $PARTS parts"

want_size() {  # part index -> expected bytes
  local i="$1" start end
  start=$((i * CHUNK)); end=$((start + CHUNK - 1))
  [ "$end" -ge "$SIZE" ] && end=$((SIZE - 1))
  echo $((end - start + 1))
}

for round in $(seq 1 "$ROUNDS"); do
  missing=0
  for i in $(seq 0 $((PARTS - 1))); do
    PART="$TMP/part.$(printf '%03d' "$i")"
    WANT=$(want_size "$i")
    HAVE=0; [ -f "$PART" ] && HAVE=$(stat -c%s "$PART")
    [ "$HAVE" = "$WANT" ] && continue
    missing=$((missing + 1))
    START=$((i * CHUNK + HAVE))          # resume from where this part stopped
    END=$((i * CHUNK + WANT - 1))
    ( curl -sL --retry 5 --retry-delay 2 --retry-all-errors --speed-time 30 \
           --speed-limit 1000 -r "${START}-${END}" -o "$PART" \
           $([ "$HAVE" -gt 0 ] && echo "--continue-at -") "$URL" ) &
  done
  [ "$missing" = 0 ] && break
  wait
  echo "  round $round: $missing part(s) were short, rechecking"
done

for i in $(seq 0 $((PARTS - 1))); do
  PART="$TMP/part.$(printf '%03d' "$i")"
  WANT=$(want_size "$i"); HAVE=0; [ -f "$PART" ] && HAVE=$(stat -c%s "$PART")
  if [ "$HAVE" != "$WANT" ]; then
    echo "part $i still short ($HAVE/$WANT) after $ROUNDS rounds -- re-run to resume" >&2
    exit 1
  fi
done

cat "$TMP"/part.* > "$OUT"
GOT=$(stat -c%s "$OUT")
if [ "$GOT" != "$SIZE" ]; then
  # Never leave a wrong-sized file behind: it reads as a valid parquet.
  rm -f "$OUT"
  echo "assembled size mismatch: $GOT vs $SIZE" >&2; exit 1
fi
rm -rf "$TMP"
echo "ok: $(basename "$OUT") ($GOT bytes)"
