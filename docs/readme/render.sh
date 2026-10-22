#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
mkdir -p build/readme build/readme-work build/rendered docs/readme
cache="$root/build/readme-cache"
mkdir -p "$cache"

# Use fresh working files but share the verified LMScript tool cache between the model and renderer.
work="$(mktemp -d "$root/build/readme-work/run.XXXXXX")"
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  export GITHUB_TOKEN="$(gh auth token)"
fi
(
  cd "$work"
  LMSCRIPT_TOOL_CACHE="$cache" lmscript "$root/examples/classify-moons-dsl.lms" artifacts/classify-moons 7 | tee stdout.txt
)
python3 - "$work/artifacts/classify-moons/report.md" <<'PY'
import json, re, sys
from pathlib import Path

report = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(r"Test accuracy: ([0-9.]+)", report)
block = re.search(r"```\n([ox\n ]+)```", report)
if not match or not block:
    raise SystemExit("classifier report is missing accuracy or decision grid")
grid = [line.strip() for line in block.group(1).splitlines() if line.strip()]
if len(grid) != 13 or any(len(row) != 41 or set(row) - set("ox") for row in grid):
    raise SystemExit("classifier decision grid must be 13 rows by 41 columns")
accuracy = float(match.group(1))
if accuracy < .95:
    raise SystemExit(f"classifier missed its accuracy target: {accuracy:.3f}")
Path("build/readme/classifier-grid.json").write_text(
    json.dumps({"accuracy": f"{accuracy * 100:.1f}", "map": grid}), encoding="utf-8")
PY

for theme in dark light; do
  LMSCRIPT_TOOL_CACHE="$cache" lmscript docs/readme/hero.lms "build/readme-hero-$theme.dslb" "$theme"
  cp "build/rendered/readme-hero-$theme.png" "docs/readme/hero-$theme.png"
  LMSCRIPT_TOOL_CACHE="$cache" lmscript docs/readme/architecture.lms "build/readme-architecture-$theme.dslb" "$theme"
  cp "build/rendered/readme-architecture-$theme.png" "docs/readme/architecture-$theme.png"
done
LMSCRIPT_TOOL_CACHE="$cache" lmscript docs/readme/gallery.lms
cp build/rendered/readme-gallery.png docs/readme/gallery.png

python3 - <<'PY'
from pathlib import Path
images = [Path("docs/readme") / f"{name}-{theme}.png"
          for name in ("hero", "architecture") for theme in ("dark", "light")]
images.append(Path("docs/readme/gallery.png"))
for image in images:
    if not image.is_file() or image.stat().st_size == 0:
        raise SystemExit(f"missing generated README image: {image}")
    if image.stat().st_size > 400_000:
        raise SystemExit(f"README image exceeds 400 KB: {image} ({image.stat().st_size} bytes)")
print(f"README images ready: {len(images)} files, each under 400 KB")
PY
