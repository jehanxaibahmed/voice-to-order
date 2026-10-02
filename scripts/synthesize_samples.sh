#!/usr/bin/env bash
# Turn each sample's reference transcript into phone-quality audio (macOS only).
#
# Uses the built-in `say` voices with a mix of accents, then ffmpeg narrows the band to
# 300-3400 Hz, resamples to 8 kHz (telephone quality) and mixes in light background noise.
# Output: samples/audio/<id>.wav
set -euo pipefail
cd "$(dirname "$0")/.."

command -v say >/dev/null || { echo "needs macOS 'say'"; exit 1; }
command -v ffmpeg >/dev/null || { echo "needs ffmpeg"; exit 1; }

voices=("Daniel" "Karen" "Moira" "Rishi" "Samantha" "Daniel" "Karen" "Moira")
mkdir -p samples/audio
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

i=0
for sample in samples/voicemails/*.json; do
  id=$(basename "$sample" .json)
  voice=${voices[$((i % ${#voices[@]}))]}
  python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['reference'])" "$sample" \
    > "$tmp/$id.txt"
  say -v "$voice" -r 185 -f "$tmp/$id.txt" -o "$tmp/$id.aiff"
  ffmpeg -nostdin -y -v error \
    -i "$tmp/$id.aiff" \
    -f lavfi -i "anoisesrc=color=pink:amplitude=0.02:sample_rate=22050" \
    -filter_complex "[0:a]highpass=f=300,lowpass=f=3400[v];[v][1:a]amix=inputs=2:duration=first:weights=1 1[m]" \
    -map "[m]" -ac 1 -ar 8000 -c:a pcm_s16le "samples/audio/$id.wav"
  echo "$id ($voice) -> samples/audio/$id.wav"
  i=$((i + 1))
done
