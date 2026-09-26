#!/usr/bin/env bash
# Fetch the 26 AOM-CTC clips missing from our 47. Resumable and verified.
#   ./download_missing.sh "/mnt/f/Pristine Videos"
set -uo pipefail
DST="${1:?usage: download_missing.sh <destination dir>}"
BASE="https://media.xiph.org/video/aomctc/test_set"
LOG="$HOME/vmaf/download_missing.log"
mkdir -p "$DST" || { echo "cannot write to $DST"; exit 1; }

LIST=$(cat <<'L'
a2_2k/Aerial3200_1920x1080_5994_10bit_420.y4m
a2_2k/Boat_1920x1080_5994_10bit_420.y4m
a2_2k/DinnerSceneCropped_1920x1080_2997fps_10bit_420.y4m
a2_2k/FoodMarket_1920x1080_5994_10bit_420.y4m
a2_2k/Motorcycle_1920x1080_30fps_8bit.y4m
a2_2k/Skater227_1920x1080_30fps.y4m
a2_2k/TunnelFlag_1920x1080_5994_10bit_420.y4m
a2_2k/Vertical_Carnaby_1080x1920_5994.y4m
a2_2k/WorldCup_1920x1080_30p.y4m
a2_2k/WorldCup_far_1920x1080_30p.y4m
a4_360p/BlueSky_360p25_v2.y4m
a5_270p/FourPeople_480x270_60.y4m
a5_270p/Vertical_Bayshore_270x480_2997.y4m
b1_syn/CosmosTreeTrunk_sdr_2048x858_25_8bit.y4m
b1_syn/EuroTruckSimulator2_1920x1080p60_v2.y4m
b1_syn/Life_1080p30.y4m
b1_syn/Life_1080p30_v2.y4m
b1_syn/STARCRAFT_1080p60.y4m
b1_syn/Sniper_1920x1080P_30fps_8bit.y4m
b1_syn/SolLevanteDragons_sdr_1920x1080_24_10bit.y4m
b2_scc/BigBuckBunnyStudio1_1920x1080_60fps_10bit_420_020_0149.y4m
b2_scc/MissionControlClip1_1920x1080_60fps_10bit_420_0450_0579.y4m
b2_scc/MobileDeviceScreenSharing.y4m
b2_scc/SceneComposition_1.y4m
b2_scc/SceneComposition_2.y4m
b2_scc/Slides2r_1920x1080_30fps_8bit_420.y4m
L
)

log(){ echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
n=0; tot=$(echo "$LIST" | wc -l); ok=0; fail=""
log "=== fetching $tot clips into $DST ==="
while read -r rel; do
  n=$((n+1)); name=$(basename "$rel"); out="$DST/$name"
  want=$(curl -sI --max-time 60 "$BASE/$rel" | grep -i '^content-length:' | tr -dc '0-9')
  if [ -z "$want" ]; then log "[$n/$tot] NO SIZE from server: $name"; fail="$fail $name"; continue; fi
  have=$( [ -f "$out" ] && stat -c%s "$out" || echo 0 )
  if [ "$have" = "$want" ]; then log "[$n/$tot] already complete: $name"; ok=$((ok+1)); continue; fi
  log "[$n/$tot] $name  ($(( want/1000000 )) MB, have $(( have/1000000 )) MB)"
  for try in 1 2 3; do
    curl -# -L --continue-at - --max-time 3600 --connect-timeout 30 \
         --retry 3 --retry-delay 10 -o "$out" "$BASE/$rel" && break
    log "    attempt $try failed, retrying"; sleep 15
  done
  have=$( [ -f "$out" ] && stat -c%s "$out" || echo 0 )
  if [ "$have" = "$want" ]; then ok=$((ok+1)); log "    OK"
  else fail="$fail $name"; log "    INCOMPLETE ($have of $want)"; fi
done <<< "$LIST"
log "=== done: $ok/$tot complete ==="
[ -n "$fail" ] && log "incomplete:$fail"
