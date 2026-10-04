#!/usr/bin/env bash
# Download the 73 source videos (~37 GB) from the public AOM-CTC test set.
# Usage: scripts/download_sources.sh <destination folder>
# Re-running resumes partial files and skips finished ones.
set -uo pipefail
DST="${1:?usage: download_sources.sh <destination folder>}"
BASE="https://media.xiph.org/video/aomctc/test_set"
mkdir -p "$DST"

while read -r rel; do
  out="$DST/$(basename "$rel")"
  want=$(curl -sI "$BASE/$rel" | grep -i '^content-length:' | tr -dc '0-9')
  have=$([ -f "$out" ] && stat -c%s "$out" || echo 0)
  if [ -n "$want" ] && [ "$have" = "$want" ]; then echo "done     $rel"; continue; fi
  echo "fetching $rel"
  curl -L --continue-at - --retry 3 -o "$out" "$BASE/$rel" || echo "FAILED   $rel"
done <<'LIST'
a2_2k/Aerial3200_1920x1080_5994_10bit_420.y4m
b2_scc/BigBuckBunnyStudio1_1920x1080_60fps_10bit_420_020_0149.y4m
a4_360p/BlueSky_360p25.y4m
a4_360p/BlueSky_360p25_v2.y4m
a2_2k/Boat_1920x1080_5994_10bit_420.y4m
a1_4k/BoxingPractice_3840x2160_5994fps_10bit_420.y4m
a3_720p/ControlledBurn_1280x720p30_420.y4m
b1_syn/CosmosTreeTrunk_sdr_2048x858_25_8bit.y4m
a1_4k/Crosswalk_3840x2160_5994fps_10bit_420.y4m
a2_2k/CrowdRun_1920x1080p50.y4m
b1_syn/DOTA2_1920x1080_60_8bit_420.y4m
b2_scc/Debugging_1920x1080_30fps_8bit_420.y4m
a2_2k/DinnerSceneCropped_1920x1080_2997fps_10bit_420.y4m
a3_720p/DrivingPOV_1280x720p_5994_10bit_420.y4m
b1_syn/EuroTruckSimulator2_1920x1080p60.y4m
b1_syn/EuroTruckSimulator2_1920x1080p60_v2.y4m
a1_4k/FoodMarket2_3840x2160_5994fps_10bit_420.y4m
a2_2k/FoodMarket_1920x1080_5994_10bit_420.y4m
a5_270p/FourPeople_480x270_60.y4m
b1_syn/GlassHalf_1920x1080p_24p_8bit_420.y4m
a3_720p/Johnny_1280x720_60.y4m
a3_720p/KristenAndSara_1280x720_60.y4m
b1_syn/Life_1080p30.y4m
b1_syn/Life_1080p30_v2.y4m
b1_syn/MINECRAFT_1080p_60_8bit.y4m
a2_2k/MeridianTalk_sdr_1920x1080p_5994_10bit.y4m
b2_scc/MissionControlClip1_1920x1080_60fps_10bit_420_0450_0579.y4m
b2_scc/MissionControlClip3_1920x1080_60_420.y4m
b2_scc/MobileDeviceScreenSharing.y4m
a2_2k/Motorcycle_1920x1080_30fps_8bit.y4m
a2_2k/MountainBike_1920x1080_30fps_8bit.y4m
a1_4k/Neon1224_3840x2160_2997fps.y4m
a1_4k/NocturneDance_3840x2160p_10bit_60fps.y4m
a2_2k/OldTownCross_1920x1080p50.y4m
a5_270p/ParkJoy_480x270_50.y4m
a2_2k/PedestrianArea_1920x1080p25.y4m
a1_4k/PierSeaSide_3840x2160_2997fps_10bit_420_v2.y4m
a4_360p/RedKayak_360_2997.y4m
a2_2k/RitualDance_1920x1080_5994_10bit_420.y4m
a2_2k/Riverbed_1920x1080p25.y4m
a3_720p/RollerCoaster_1280x720p_5994_10bit_420.y4m
a2_2k/RushFieldCuts_1920x1080_2997.y4m
b1_syn/STARCRAFT_1080p60.y4m
b2_scc/SceneComposition_1.y4m
b2_scc/SceneComposition_2.y4m
a2_2k/Skater227_1920x1080_30fps.y4m
b2_scc/Slides1_1920x1080_30fps_8bit_420.y4m
b2_scc/Slides2r_1920x1080_30fps_8bit_420.y4m
b1_syn/Sniper_1920x1080P_30fps_8bit.y4m
a4_360p/SnowMountain_640x360_2997.y4m
b1_syn/SolLevanteDragons_sdr_1920x1080_24_10bit.y4m
b1_syn/SolLevanteFace_sdr_1920x1080_24_10bit.y4m
a5_270p/SparksElevator_480x270p_5994_10bit.y4m
a4_360p/SpeedBag_640x360_2997.y4m
b2_scc/Spreadsheet_1920x1080_30fps_8bit_420_130f.y4m
a4_360p/Stockholm_640x360_5994.y4m
a1_4k/Tango_3840x2160_5994fps_10bit_420.y4m
a1_4k/TimeLapse_3840x2160_5994fps_10bit_420.y4m
a2_2k/ToddlerFountain_1920x1080_2997fps_10bit_420.y4m
a4_360p/TouchdownPass_640x360_2997.y4m
a2_2k/TreesAndGrass_1920_1080_30fps_8bit.y4m
a2_2k/TunnelFlag_1920x1080_5994_10bit_420.y4m
a5_270p/Vertical_Bayshore_270x480_2997.y4m
a2_2k/Vertical_Carnaby_1080x1920_5994.y4m
a2_2k/Vertical_bees_1080x1920_2997.y4m
a3_720p/Vidyo3_1280x720p_60fps.y4m
a3_720p/Vidyo4_1280x720p_60fps.y4m
b1_syn/WITCHER3_1920x1080_60_8bit_420.y4m
a2_2k/WalkingInStreet_1920x1080_30fps.y4m
a3_720p/WestWindEasy_1280x720p30_420.y4m
b2_scc/Wikipedia_1920x1080p30.y4m
a2_2k/WorldCup_1920x1080_30p.y4m
a2_2k/WorldCup_far_1920x1080_30p.y4m
LIST
