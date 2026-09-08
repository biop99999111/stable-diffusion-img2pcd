"""Compare recorded runs without treating different inputs as a controlled trial."""
import argparse
import json
from pathlib import Path


def comparison(paths):
    rows = []
    for path in paths:
        path = Path(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        workflow_path = path.parent.parent / "workflow.json"
        workflow = json.loads(workflow_path.read_text(encoding="utf-8")) if workflow_path.exists() else {}
        shape = manifest.get("shape_check") or {}
        verify = manifest.get("verify") or {}
        rows.append([str(path), manifest.get("backend", manifest.get("settings", {}).get("backend", "unknown")),
                     manifest.get("image", "unknown"), workflow.get("input_sha256", "unrecorded"),
                     workflow.get("model_revision", "unrecorded"),
                     manifest.get("bbox_mm"), shape.get("ratio"), shape.get("verdict", "unverified"),
                     manifest.get("t_load_s"), manifest.get("t_preprocess_s"), manifest.get("t_generate_s"),
                     manifest.get("peak_vram_gib"), manifest.get("t_texture_s"),
                     manifest.get("peak_vram_texture_gib"), verify.get("point_count")])
    headings = ["Manifest", "Backend", "Input", "Input SHA256", "Model revision", "Bbox mm", "Axis ratios",
                "Shape", "Load s", "Preprocess s", "Generate s", "Peak allocated GiB",
                "Separate texture s", "Texture peak GiB", "PCD points"]
    def cell(value):
        return str(value if value is not None else "unrecorded").replace("|", "\\|").replace("\n", " ")
    table = ["| " + " | ".join(headings) + " |", "| " + " | ".join(["---"]*len(headings)) + " |"]
    table.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
    return ("# Recorded model comparison\n\n"
            "Different or unrecorded input hashes are reference runs, not a controlled same-input comparison. "
            "Even matching inputs require checking preprocessing, settings and timing scope. "
            "Missing metrics remain unrecorded. Shape PASS does not establish metrology accuracy.\n\n"
            + "\n".join(table) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifests", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(comparison(args.manifests))
