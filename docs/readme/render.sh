#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
mkdir -p build/readme build/readme-work build/rendered docs/readme
cache="$root/build/readme-cache"
mkdir -p "$cache"
work="$(mktemp -d "$root/build/readme-work/run.XXXXXX")"
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  export GITHUB_TOKEN="$(gh auth token)"
fi
export LMSCRIPT_TOOL_CACHE="$cache"

(
  cd "$work"
  lmscript "$root/examples/classify-moons-dsl.lms" artifacts/classify-moons 7 > classifier.stdout
  lmscript "$root/docs/readme/plot-points.lms" > points.json
  lmscript "$root/examples/train-report-dsl.lms" artifacts/train-report 60 0.05 11 > training.stdout
)

python3 - "$work" <<'PY'
import csv, json, re, sys
from pathlib import Path

work = Path(sys.argv[1])
report = (work / "artifacts/classify-moons/report.md").read_text(encoding="utf-8")
accuracy_match = re.search(r"Test accuracy: ([0-9.]+)", report)
grid_match = re.search(r"```\n([ox\n ]+)```", report)
if not accuracy_match or not grid_match:
    raise SystemExit("classifier report is missing accuracy or decision grid")
accuracy = float(accuracy_match.group(1))
grid = [line.strip() for line in grid_match.group(1).splitlines() if line.strip()]
if len(grid) != 13 or any(len(row) != 41 or set(row) - set("ox") for row in grid):
    raise SystemExit("classifier decision grid must be 13 rows by 41 columns")
if accuracy < 0.95:
    raise SystemExit(f"classifier missed its accuracy target: {accuracy:.3f}")

point_data = json.loads((work / "points.json").read_text(encoding="utf-8"))
points, labels = point_data["points"], point_data["labels"]
cols, rows = 81, 51
boundary_labels = [int(value) for value in point_data["boundary"]]
if len(boundary_labels) != cols * rows:
    raise SystemExit("classifier did not return the dense 81 x 51 boundary grid")
grid = [boundary_labels[r*cols:(r+1)*cols] for r in range(rows)]
if len(points) != 400 or len(labels) != 400:
    raise SystemExit("expected the classifier's 400 generated points")
losses = []
with (work / "artifacts/train-report/losses.csv").open(newline="", encoding="utf-8") as f:
    for row in csv.reader(f):
        if row:
            losses.append(float(row[-1]))
if len(losses) != 60 or losses[-1] >= losses[0]:
    raise SystemExit("training loss data is missing or did not decrease")

def circle(cx, cy, r):
    return (f"M {cx-r:.1f} {cy:.1f} C {cx-r:.1f} {cy-r*.55:.1f} {cx-r*.55:.1f} {cy-r:.1f} {cx:.1f} {cy-r:.1f} "
            f"C {cx+r*.55:.1f} {cy-r:.1f} {cx+r:.1f} {cy-r*.55:.1f} {cx+r:.1f} {cy:.1f} "
            f"C {cx+r:.1f} {cy+r*.55:.1f} {cx+r*.55:.1f} {cy+r:.1f} {cx:.1f} {cy+r:.1f} "
            f"C {cx-r*.55:.1f} {cy+r:.1f} {cx-r:.1f} {cy+r*.55:.1f} {cx-r:.1f} {cy:.1f} Z")

def paths(frame, radius):
    left, top, width, height = frame
    xmin, xmax, ymin, ymax = -1.7, 2.7, -1.35, 1.8
    class_paths = [[], []]
    for (x, y), label in zip(points, labels):
        px = left + (x - xmin) / (xmax - xmin) * width
        py = top + (ymax - y) / (ymax - ymin) * height
        class_paths[int(label)].append(circle(px, py, radius))

    # Marching-square segments come directly from adjacent class predictions in the dense model grid.
    boundary = []
    def xy(col, row):
        x = -1.7 + 4.4 * col / (cols-1)
        y = 1.8 - 3.15 * row / (rows-1)
        return (left + (x - xmin) / (xmax-xmin) * width,
                top + (ymax - y) / (ymax-ymin) * height)
    for row in range(rows-1):
        for col in range(cols-1):
            v = [grid[row][col] == 1, grid[row][col+1] == 1,
                 grid[row+1][col+1] == 1, grid[row+1][col] == 1]
            edge_points = []
            edges = [(0,1,(col+.5,row)), (1,2,(col+1,row+.5)),
                     (2,3,(col+.5,row+1)), (3,0,(col,row+.5))]
            for a,b,pos in edges:
                if v[a] != v[b]: edge_points.append(pos)
            pairs = [(edge_points[0], edge_points[1])] if len(edge_points) == 2 else []
            if len(edge_points) == 4:
                pairs = [(edge_points[0], edge_points[1]), (edge_points[2], edge_points[3])]
            for a,b in pairs:
                mid_col, mid_row = (a[0]+b[0])/2, (a[1]+b[1])/2
                world_x = -1.7 + 4.4 * mid_col / (cols-1)
                world_y = 1.8 - 3.15 * mid_row / (rows-1)
                # Keep the illustrative contour where the two generated classes have support;
                # omit low-confidence extrapolation that shoots beyond the point clouds.
                if not (-1.3 <= world_x <= 2.15 and -0.65 <= world_y <= 1.45):
                    continue
                if min((px-world_x)**2 + (py-world_y)**2 for px,py in points) > 0.45**2:
                    continue
                ax, ay = xy(*a); bx, by = xy(*b)
                boundary.append(f"M {ax:.1f} {ay:.1f} L {bx:.1f} {by:.1f}")
    def chunks(items):
        return [" ".join(items[i:i+50]) for i in range(0, len(items), 50)]
    if len(class_paths[0]) != 200 or len(class_paths[1]) != 200:
        raise SystemExit("expected 200 points in each class")
    boundary_chunks = []
    current = ""
    for segment in boundary:
        if current and len(current) + len(segment) + 1 > 7000:
            boundary_chunks.append(current)
            current = ""
        current += (" " if current else "") + segment
    if current: boundary_chunks.append(current)
    if len(boundary_chunks) > 8:
        raise SystemExit("decision boundary path needs more than eight safe path segments")
    boundary_chunks += ["M 2000 2000 L 2001 2001"] * (8-len(boundary_chunks))
    return {"class0": chunks(class_paths[0]), "class1": chunks(class_paths[1]), "boundary": boundary_chunks}

def write_plot(name, frame, radius):
    Path(f"build/readme/plot-{name}.json").write_text(json.dumps(paths(frame, radius)), encoding="utf-8")

write_plot("desktop", (758, 78, 564, 464), 4.0)
write_plot("mobile", (78, 335, 624, 540), 6.5)
plot_gallery = paths((112, 194, 856, 174), 3.4)
min_loss, max_loss = min(losses), max(losses)
curve = []
for i, loss in enumerate(losses):
    x = 112 + 790 * i / (len(losses)-1)
    y = 165 + 150 * (max_loss - loss) / (max_loss - min_loss)
    curve.append((x,y))
loss_path = " ".join((f"M {curve[0][0]:.1f} {curve[0][1]:.1f}" if i == 0 else f"L {x:.1f} {y:.1f}") for i,(x,y) in enumerate(curve))
plot_gallery.update({"loss": loss_path,
    "loss_summary": f"60 steps  /  {losses[0]:.4f} -> {losses[-1]:.4f}",
    "accuracy_text": f"{accuracy*100:.1f}% held-out accuracy"})
Path("build/readme/plot-gallery.json").write_text(json.dumps(plot_gallery), encoding="utf-8")
print(f"README data ready: 400 points, {accuracy:.3f} accuracy, {len(losses)} checked training losses")
PY

for layout in desktop mobile; do
  for theme in dark light; do
    lmscript docs/readme/hero.lms "$theme" "$layout"
    cp "build/rendered/readme-hero-$layout-$theme.png" "docs/readme/hero-$layout-$theme.png"
  done
done
for theme in dark light; do
  lmscript docs/readme/architecture.lms "$theme"
  cp "build/rendered/readme-architecture-$theme.png" "docs/readme/architecture-$theme.png"
  for mode in sum gradient training classifier; do
    lmscript docs/readme/gallery.lms "$mode" "$theme"
    cp "build/rendered/readme-gallery-$mode-$theme.png" "docs/readme/gallery-$mode-$theme.png"
  done
done

python3 - <<'PY'
from pathlib import Path
images = sorted(Path("docs/readme").glob("*.png"))
if len(images) != 14:
    raise SystemExit(f"expected 14 README images; found {len(images)}")
for image in images:
    size = image.stat().st_size
    if not size or size > 400_000:
        raise SystemExit(f"invalid README image size: {image} ({size} bytes)")
print(f"README images ready: {len(images)} files, each under 400 KB")
PY
