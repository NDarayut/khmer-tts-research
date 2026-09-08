#!/usr/bin/env bash
# Fetch one large file over N parallel byte-range connections, then concatenate.
#
# Hugging Face serves this machine at ~52 kB/s on a single connection but
# ~150-180 kB/s per connection when several run at once, so the download is
# throttled per-connection rather than by the link. Sixteen ranges turns a
# 16-hour pull into well under an hour. huggingface_hub's own downloader opens
# one connection per *file*, which is why it stalled here.
#
#   finetune/fetch_parallel.sh <url> <output-path> [n_parts]
set -euo pipefail

URL="$1"; OUT="$2"; PARTS="${3:-16}"
TMP="${OUT}.parts"

SIZE=$(curl -sIL "$URL" | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}')
[ -n "$SIZE" ] || { echo "could not determine size of $URL" >&2; exit 1; }

if [ -f "$OUT" ] && [ "$(stat -c%s "$OUT")" = "$SIZE" ]; then
  echo "already complete: $OUT ($SIZE bytes)"; exit 0
fi

mkdir -p "$TMP"
CHUNK=$(( (SIZE + PARTS - 1) / PARTS ))
echo "$(basename "$OUT"): $SIZE bytes in $PARTS parts of $CHUNK"

for i in $(seq 0 $((PARTS - 1))); do
  START=$((i * CHUNK)); END=$((START + CHUNK - 1))
  [ "$END" -ge "$SIZE" ] && END=$((SIZE - 1))
  PART="$TMP/part.$(printf '%03d' "$i")"
  WANT=$((END - START + 1))
  # Resume a part that is already complete, so a re-run costs nothing.
  if [ -f "$PART" ] && [ "$(stat -c%s "$PART")" = "$WANT" ]; then continue; fi
  ( curl -sL --retry 5 --retry-delay 2 --retry-all-errors \
         -r "${START}-${END}" -o "$PART" "$URL" ) &
done
wait

cat "$TMP"/part.* > "$OUT"
GOT=$(stat -c%s "$OUT")
if [ "$GOT" != "$SIZE" ]; then
  echo "size mismatch: got $GOT want $SIZE -- re-run to resume" >&2; exit 1
fi
rm -rf "$TMP"
echo "ok: $OUT ($GOT bytes)"
