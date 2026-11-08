#!/usr/bin/env bash
# Rebuild the final pictures from source.jpg.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
python3 zoom_out.py source.jpg build/extended.png                                    # zoom out (1440x1456)
python3 upscale.py build/extended.png build/extended_2x.png --scale 2 --screen-origin 120 236
python3 add_halo.py build/extended.png halo_zoomout.png                              # 1440x1456
python3 add_halo.py build/extended_2x.png halo_zoomout_2x.png --scale 2              # 2880x2912
python3 -c "from PIL import Image; im = Image.open('halo_zoomout_2x.png'); im.save('halo_zoomout_2x.jpg', quality=95, subsampling=0, icc_profile=im.info.get('icc_profile'))"
