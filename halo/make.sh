#!/usr/bin/env bash
# Rebuild every picture.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
save_jpg() { python3 -c "import sys; from PIL import Image; im = Image.open(sys.argv[1]); im.save(sys.argv[2], quality=int(sys.argv[3]), subsampling=0, icc_profile=im.info.get('icc_profile'))" "$1" "$2" "${3:-95}"; }

# final: the halo of the user's edit (edit.jpg) enlarged 1.27x and turned 5 degrees,
# same frame (no zoom out)
python3 enlarge_halo.py edit.jpg source.jpg halo_final.png                          # 1254x1254
python3 upscale.py halo_final.png halo_final_2x.png --scale 2 --plain               # 2508x2508
save_jpg halo_final_2x.png halo_final_2x.jpg 100                              # 3.9 MB, under 4 MB
python3 -c "from PIL import Image; Image.open('halo_final.png').convert('RGB').resize((512, 512), Image.LANCZOS, reducing_gap=3.0).save('halo_mlbb_512.png', optimize=True)"  # MLBB avatar

# v2: zoom out of source.jpg with the thin clean ring
python3 zoom_out.py source.jpg build/extended.png                                   # 1440x1456
python3 upscale.py build/extended.png build/extended_2x.png --scale 2 --screen-origin 120 236
python3 add_halo.py build/extended.png halo_v2.png
python3 add_halo.py build/extended_2x.png halo_v2_2x.png --scale 2
save_jpg halo_v2_2x.png halo_v2_2x.jpg
