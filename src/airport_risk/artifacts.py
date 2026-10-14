"""导出数据摘要、输入节点参数和源码清单。"""

from pathlib import Path
import hashlib
import numpy as np
from .reporting import write_json, write_csv


def export_data_views(data, outdir):
    out = Path(outdir)
    ratio = data.target / data.capacities
    write_json(
        out / "data_quality.json",
        {
            "provenance": data.metadata["provenance"],
            "data_fingerprint": data.fingerprint(),
            "samples": len(data.target),
            "events": len(np.unique(data.event_ids)),
            "nodes": len(data.capacities),
            "candidate_factors": data.factor_values.shape[1],
            "history_steps": data.history.shape[1],
            "observed_future_steps": (
                data.future_loads.shape[1] if getattr(data, "future_loads", None) is not None else 1
            ),
            "edges": int(data.adjacency.sum()),
            "year_counts": {int(y): int(np.sum(data.years == y)) for y in np.unique(data.years)},
            "airport_counts": {
                str(a): int(np.sum(data.airports == a)) for a in np.unique(data.airports)
            },
            "next_node_failure_fraction": float((ratio >= 1).mean()),
            "next_event_with_any_failure_fraction": float((ratio >= 1).any(1).mean()),
            "target_load_capacity_ratio_quantiles": np.quantile(
                ratio, [0, 0.25, 0.5, 0.75, 0.95, 1]
            ),
            "status": (
                "software fixture diagnostics"
                if data.metadata["provenance"] == "synthetic"
                else "input diagnostics; authorization and measurement definitions are external declarations"
            ),
        },
    )
    names = data.metadata.get("node_names", data.node_ids.tolist())
    if len(names) != len(data.node_ids):
        original = data.metadata.get("original_node_indices")
        names = [names[i] for i in original] if original is not None else data.node_ids.tolist()
    write_csv(
        out / "tables" / "table1_node_parameters.csv",
        [
            {
                "node_id": str(node),
                "name": str(names[i]),
                "capacity": float(data.capacities[i]),
                "impedance": bool(data.impedance[i]),
                "prior_grade": float(data.grades[i]),
                "provenance": data.metadata["provenance"],
            }
            for i, node in enumerate(data.node_ids)
        ],
    )
    write_csv(
        out / "tables" / "event_summary.csv",
        [
            {
                "event_id": str(data.event_ids[i]),
                "sample_id": (
                    str(data.sample_ids[i])
                    if getattr(data, "sample_ids", None) is not None
                    else str(data.event_ids[i])
                ),
                "year": int(data.years[i]),
                "airport": str(data.airports[i]),
                "last_observed_mean_load_ratio": float(
                    (data.history[i, -1] / data.capacities).mean()
                ),
                "next_mean_load_ratio": float(ratio[i].mean()),
                "next_failed_nodes": int((ratio[i] >= 1).sum()),
                "provenance": data.metadata["provenance"],
            }
            for i in range(len(data.target))
        ],
    )


def write_source_manifest(project_root, outdir, *, paper_sha256=None):
    root = Path(project_root)
    files = {}
    for directory in ["src", "tests", "configs", "docs"]:
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in [
        "pyproject.toml",
        "requirements-lock-macos-arm64.txt",
        "README.md",
        "CITATION.cff",
        "references.bib",
        "THIRD_PARTY.md",
        "LICENSE-NOTICE.md",
        ".gitignore",
    ]:
        path = root / name
        if path.exists():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(
        Path(outdir) / "source_manifest.json",
        {
            "sha256": files,
            "paper_sha256": paper_sha256,
            "scope": "source/config/test/documentation/dependency snapshot; excludes input data and experiment results",
        },
    )
