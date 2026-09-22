from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from analysis import (
    add_secondary, audit_checkpoint, audit_dataset, city_ranking,
    comparison_evaluation, comparison_rows, ranking_agreement, ranking_metrics,
    type_reconstruction,
)
from core import (
    APP_DIR, ATTRIBUTE_NAMES, CITY_CODES, CITY_COORDINATES,
    COMPARISON_RELATIONS, DATASETS_DIR, PROJECT_DIR, OUTPUT_DIR, TYPE_CANDIDATES,
    Predictor, is_city_year, load_bundle, parse_city_year,
)


def dump_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields and not isinstance(row[key], (list, dict)):
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def boundaries_by_name(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    boundaries = {str(item.get("properties", {}).get("name")): item for item in data.get("features", [])}
    missing = sorted(set(CITY_COORDINATES) - set(boundaries))
    if missing:
        raise ValueError("Administrative boundaries missing cities: {}".format(", ".join(missing)))
    return boundaries


def feature_for_city(city, properties, boundaries):
    if city in boundaries:
        return {"type": "Feature", "geometry": boundaries[city]["geometry"], "properties": properties}
    longitude, latitude = CITY_COORDINATES[city]
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [longitude, latitude]}, "properties": properties}


def apply_registry(args):
    registry_path = APP_DIR / "model_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if getattr(args, "model_id", None):
        item = registry[args.model_id]
        args.dataset = (PROJECT_DIR / item["dataset"]).resolve()
        args.checkpoint = (PROJECT_DIR / item["checkpoint"]).resolve()
    elif not getattr(args, "dataset", None) or not getattr(args, "checkpoint", None):
        raise ValueError("Provide --model-id or both --dataset and --checkpoint")
    if getattr(args, "secondary_model_id", None):
        other = registry[args.secondary_model_id]
        args.secondary_dataset = (PROJECT_DIR / other["dataset"]).resolve()
        args.secondary_checkpoint = (PROJECT_DIR / other["checkpoint"]).resolve()


def make_predictors(args):
    apply_registry(args)
    checkpoints = [args.checkpoint, getattr(args, "secondary_checkpoint", None)]
    for checkpoint in checkpoints:
        if checkpoint is not None and not Path(checkpoint).is_file():
            raise FileNotFoundError(
                "Checkpoint not found: {}. Train the model first and provide the generated .pt file.".format(checkpoint)
            )
    primary = Predictor(load_bundle(args.dataset), args.checkpoint, args.device, getattr(args, "stochastic_eval", False))
    secondary = None
    if getattr(args, "secondary_checkpoint", None):
        secondary_dataset = args.secondary_dataset or args.dataset
        secondary = Predictor(load_bundle(secondary_dataset), args.secondary_checkpoint, args.device, getattr(args, "stochastic_eval", False))
    return primary, secondary


def annotate_rows(rows, predictor, model_id=None):
    for row in rows:
        row["model_id"] = model_id or Path(predictor.audit["checkpoint"]).stem
        row["checkpoint_sha256"] = predictor.audit["checkpoint_sha256"]
        row["dataset"] = predictor.bundle.directory.name
        if "normalized_confidence" not in row and "raw_score" in row:
            row["normalized_confidence"] = row["raw_score"]
    return rows


def strict_load_command(args):
    predictor, secondary = make_predictors(args)
    report = {"primary": predictor.audit, "secondary": secondary.audit if secondary else None, "passed": True}
    dump_json(args.output, report)
    dump_json(args.output.with_name("mask_audit.json"), {"primary": predictor.audit["literal_input"], "secondary": secondary.audit["literal_input"] if secondary else None})
    print("Strict load passed: {}".format(args.output))


def audit_data_command(args):
    reports = [audit_dataset(args.datasets_root / name) for name in args.datasets]
    dump_json(args.output, reports); print("Data audit: {}".format(args.output))


def audit_checkpoint_command(args):
    reports = [audit_checkpoint(Path(value)) for value in args.checkpoints]
    dump_json(args.output, reports); print("Checkpoint audit: {}".format(args.output))


def audit_pair_command(args):
    bundle = load_bundle(args.dataset)
    checkpoint = audit_checkpoint(args.checkpoint)
    entity_shape = checkpoint["entity_embedding_shape"]
    relation_shape = checkpoint["relation_embedding_shape"]
    report = {
        "dataset": audit_dataset(args.dataset), "checkpoint": checkpoint,
        "compatible_dimensions": bool(entity_shape and relation_shape and entity_shape[0] == len(bundle.entity_to_id) and relation_shape[0] == len(bundle.relation_to_id)),
        "supported_model": checkpoint["identified_model"] == "hmnce",
    }
    report["passed"] = report["compatible_dimensions"] and report["supported_model"]
    dump_json(args.output, report)
    if not report["passed"]:
        raise RuntimeError("Checkpoint/dataset audit failed; inspect {}".format(args.output))
    print("Checkpoint/dataset audit passed: {}".format(args.output))


def infer_command(args):
    primary, secondary = make_predictors(args)
    rows = primary.rank(args.head, args.relation, args.top_k, args.year, args.filter_known)
    annotate_rows(rows, primary, args.model_id)
    if secondary:
        other = secondary.rank(args.head, args.relation, args.top_k, args.year, args.filter_known)
        add_secondary(rows, other)
    write_csv(args.output, rows); dump_json(args.output.with_suffix(".json"), rows)
    print("Inference: {}".format(args.output))


def evaluate_command(args):
    primary, _ = make_predictors(args)
    if args.training_prefix and args.evaluation_repeats > 1:
        repeated = [ranking_metrics(primary, args.split, args.max_queries, True, args.batch_size) for _ in range(args.evaluation_repeats)]
        overall = [item["overall"] for item in repeated]
        summary = {metric: {"mean": float(sum(row[metric] for row in overall) / len(overall)), "std": float(torch.tensor([row[metric] for row in overall]).std(unbiased=False).item())} for metric in ("mr", "mrr", "hits@1", "hits@3", "hits@10")}
        result = {"checkpoint_audit": primary.audit, "split": args.split, "training_prefix": True, "evaluation_repeats": args.evaluation_repeats, "summary": summary, "runs": repeated}
        rows = [{"run": index + 1, **value["overall"]} for index, value in enumerate(repeated)]
    else:
        ranking = ranking_metrics(primary, args.split, args.max_queries, args.training_prefix, args.batch_size)
        result = {"checkpoint_audit": primary.audit, "split": args.split, "training_prefix": args.training_prefix, "ranking": ranking}
        rows = [{"relation": key, **value} for key, value in ranking.items()]
    dump_json(args.output, result)
    write_csv(args.output.with_suffix(".csv"), rows)
    if not args.training_prefix:
        write_csv(args.output.with_name("per_relation_metrics.csv"), rows)
    print("Evaluation: {}".format(args.output))


def deterministic_test_command(args):
    predictor, _ = make_predictors(args)
    outputs = [predictor.scores([args.head], [args.relation])[0] for _ in range(3)]
    differences = [float(abs(outputs[index] - outputs[index + 1]).max()) for index in range(2)]
    report = {"head": args.head, "relation": args.relation, "max_absolute_differences": differences, "passed": max(differences) == 0.0, "checkpoint_audit": predictor.audit}
    dump_json(args.output, report)
    if not report["passed"]:
        raise RuntimeError("Deterministic inference test failed: {}".format(differences))
    print("Deterministic test passed: {}".format(args.output))


def export_types_command(args):
    primary, secondary = make_predictors(args); boundaries = boundaries_by_name(args.boundaries)
    output = {"type": "FeatureCollection", "metadata": {"mode": "type", "year": args.year, "primary": primary.audit, "secondary": secondary.audit if secondary else None}, "features": []}
    all_rows = []
    for relation in (args.relation,):
        report = type_reconstruction(primary, relation, args.year)
        annotate_rows(report["rows"], primary, args.model_id)
        secondary_rows = type_reconstruction(secondary, relation, args.year)["rows"] if secondary else []
        by_entity = {row["entity"]: row for row in secondary_rows}
        for row in report["rows"]:
            other = by_entity.get(row["entity"])
            row["secondary_prediction"] = other["prediction"] if other else None
            row["secondary_confidence"] = other["normalized_confidence"] if other else None
            row["model_agreement"] = row["prediction"] == other["prediction"] if other else None
            properties = dict(row); properties["mode"] = "type"; properties["adcode"] = CITY_CODES[row["city"]]
            properties["numeric_values"] = primary.bundle.raw_literals[primary.bundle.entity_to_id[row["entity"]]].astype(float).tolist()
            output["features"].append(feature_for_city(row["city"], properties, boundaries)); all_rows.append(row)
    dump_json(args.output, output); write_csv(args.output.with_suffix(".csv"), all_rows)
    if args.web_data:
        shutil.copyfile(str(args.output), str(args.web_data))
    print("Type map: {}".format(args.output))


def export_comparison_command(args):
    primary, secondary = make_predictors(args); boundaries = boundaries_by_name(args.boundaries)
    rows = comparison_rows(primary, args.relation, args.year, args.anchor)
    annotate_rows(rows, primary, args.model_id)
    if secondary:
        add_secondary(rows, comparison_rows(secondary, args.relation, args.year, args.anchor))
    output = {"type": "FeatureCollection", "metadata": {"mode": "comparison", "year": args.year, "relation": args.relation, "attribute": args.relation.replace("_all_comp", ""), "anchor": args.anchor, "semantics": "anchor value is strictly greater than candidate value", "primary": primary.audit, "secondary": secondary.audit if secondary else None}, "features": []}
    by_city = {row["tail_city"]: row for row in rows}
    for city in CITY_COORDINATES:
        if city == args.anchor:
            entity = "{}_{}".format(city, args.year)
            properties = {"mode": "comparison", "city": city, "year": args.year, "is_anchor": True, "relation": args.relation, "prediction": "基准城市", "raw_score": 1.0, "model_id": args.model_id or Path(primary.audit["checkpoint"]).stem, "checkpoint_sha256": primary.audit["checkpoint_sha256"], "dataset": primary.bundle.directory.name, "numeric_values": primary.bundle.raw_literals[primary.bundle.entity_to_id[entity]].astype(float).tolist()}
        else:
            row = by_city.get(city, {}); properties = dict(row)
            entity = "{}_{}".format(city, args.year)
            properties.update({"mode": "comparison", "city": city, "year": args.year, "is_anchor": False, "prediction": "支持高于" if row.get("raw_score", 0) >= args.threshold else "不支持高于", "numeric_values": primary.bundle.raw_literals[primary.bundle.entity_to_id[entity]].astype(float).tolist()})
        output["features"].append(feature_for_city(city, properties, boundaries))
    dump_json(args.output, output); write_csv(args.output.with_suffix(".csv"), rows)
    if args.web_data:
        shutil.copyfile(str(args.output), str(args.web_data))
    print("Comparison map: {}".format(args.output))


def export_ranking_command(args):
    primary, _ = make_predictors(args); boundaries = boundaries_by_name(args.boundaries)
    report = city_ranking(primary, args.relation, args.year)
    annotate_rows(report["rows"], primary, args.model_id)
    features = []
    for row in report["rows"]:
        properties = dict(row); entity = row["entity"]
        properties.update({
            "mode": "ranking", "numeric_values": primary.bundle.raw_literals[
                primary.bundle.entity_to_id[entity]
            ].astype(float).tolist(),
        })
        features.append(feature_for_city(row["city"], properties, boundaries))
    metadata = {
        "mode": "ranking", "year": args.year, "relation": args.relation,
        "attribute": args.relation.replace("_all_comp", ""),
        "summary": report["summary"], "primary": primary.audit,
    }
    output = {"type": "FeatureCollection", "metadata": metadata, "features": features}
    dump_json(args.output, output)
    write_csv(args.output.with_suffix(".csv"), report["rows"])
    dump_json(args.output.with_suffix(".json"), report)
    if args.web_data:
        shutil.copyfile(str(args.output), str(args.web_data))
    print("Ranking map: {}".format(args.output))


def export_web_bundle_command(args):
    primary, secondary = make_predictors(args)
    boundaries = boundaries_by_name(args.boundaries)
    args.output.mkdir(parents=True, exist_ok=True)
    index = {
        "version": 1, "primary": primary.audit, "secondary": secondary.audit if secondary else None,
        "years": args.years, "relations": args.relations, "anchors": args.anchors,
        "type_maps": [], "comparison_maps": [], "ranking_maps": [],
    }
    for relation in args.type_relations:
        for year in args.years:
            report = type_reconstruction(primary, relation, year)
            annotate_rows(report["rows"], primary, args.model_id)
            other_rows = type_reconstruction(secondary, relation, year)["rows"] if secondary else []
            other_by_entity = {row["entity"]: row for row in other_rows}
            features = []
            for row in report["rows"]:
                other = other_by_entity.get(row["entity"])
                row["secondary_prediction"] = other["prediction"] if other else None
                row["secondary_confidence"] = other["normalized_confidence"] if other else None
                row["model_agreement"] = row["prediction"] == other["prediction"] if other else None
                properties = dict(row); properties.update({"mode": "type", "adcode": CITY_CODES[row["city"]], "numeric_values": primary.bundle.raw_literals[primary.bundle.entity_to_id[row["entity"]]].astype(float).tolist()})
                features.append(feature_for_city(row["city"], properties, boundaries))
            filename = "type_{}_{}.geojson".format(relation, year)
            dump_json(args.output / filename, {"type": "FeatureCollection", "metadata": {"mode": "type", "relation": relation, "year": year, "summary": {key: report[key] for key in ("reconstruction_accuracy", "macro_f1", "confusion_matrix")}}, "features": features})
            index["type_maps"].append({"relation": relation, "year": year, "file": args.output.name + "/" + filename})
    for relation in args.relations:
        for year in args.years:
            report = city_ranking(primary, relation, year)
            annotate_rows(report["rows"], primary, args.model_id)
            features = []
            for row in report["rows"]:
                properties = dict(row); entity = row["entity"]
                properties.update({"mode": "ranking", "numeric_values": primary.bundle.raw_literals[primary.bundle.entity_to_id[entity]].astype(float).tolist()})
                features.append(feature_for_city(row["city"], properties, boundaries))
            filename = "ranking_{}_{}.geojson".format(relation, year)
            dump_json(args.output / filename, {"type": "FeatureCollection", "metadata": {"mode": "ranking", "relation": relation, "year": year, "attribute": relation.replace("_all_comp", ""), "summary": report["summary"]}, "features": features})
            index["ranking_maps"].append({"relation": relation, "year": year, "file": args.output.name + "/" + filename})
    for relation in args.relations:
        for year in args.years:
            for anchor in args.anchors:
                rows = comparison_rows(primary, relation, year, anchor)
                annotate_rows(rows, primary, args.model_id)
                if secondary:
                    add_secondary(rows, comparison_rows(secondary, relation, year, anchor))
                by_city = {row["tail_city"]: row for row in rows}; features = []
                for city in CITY_COORDINATES:
                    if city == anchor:
                        properties = {"mode": "comparison", "city": city, "year": year, "is_anchor": True, "relation": relation, "prediction": "基准城市", "raw_score": 1.0}
                    else:
                        properties = dict(by_city.get(city, {})); properties.update({"mode": "comparison", "city": city, "year": year, "is_anchor": False, "prediction": "支持高于" if properties.get("raw_score", 0) >= args.threshold else "不支持高于"})
                    features.append(feature_for_city(city, properties, boundaries))
                filename = "comparison_{}_{}_{}.geojson".format(relation, year, CITY_CODES[anchor])
                dump_json(args.output / filename, {"type": "FeatureCollection", "metadata": {"mode": "comparison", "relation": relation, "year": year, "anchor": anchor, "semantics": "anchor value is strictly greater than candidate value"}, "features": features})
                index["comparison_maps"].append({"relation": relation, "year": year, "anchor": anchor, "file": args.output.name + "/" + filename})
    dump_json(args.output.parent / "index.json", index)
    print("Web bundle: {}".format(args.output))


def evaluate_comparisons_command(args):
    primary, secondary = make_predictors(args)
    if not args.years:
        args.years = sorted(set(parse_city_year(entity)[1] for entity in primary.bundle.entity_to_id if is_city_year(entity)))
    if not args.relations:
        args.relations = COMPARISON_RELATIONS
    rows, metrics = comparison_evaluation(primary, args.years, args.relations, args.threshold)
    annotate_rows(rows, primary, args.model_id)
    report = {"primary": primary.audit, "metrics": metrics}
    if secondary:
        secondary_rows, secondary_metrics = comparison_evaluation(secondary, args.years, args.relations, args.threshold)
        add_secondary(rows, secondary_rows)
        agreement_rows, agreement_summary = ranking_agreement(rows, secondary_rows)
        report["secondary"] = secondary.audit
        report["secondary_metrics"] = secondary_metrics
        report["model_agreement"] = agreement_summary
        write_csv(args.output.with_name("model_agreement.csv"), agreement_rows)
    dump_json(args.output, report)
    write_csv(args.output.with_name("comparison_predictions.csv"), rows)
    write_csv(args.output.with_name("comparison_metrics.csv"), [{"relation": key, **value} for key, value in metrics.items()])
    print("Comparison evaluation: {}".format(args.output))


def reconstruct_types_command(args):
    primary, secondary = make_predictors(args)
    reports = []
    for relation in args.relations:
        report = type_reconstruction(primary, relation, args.year)
        annotate_rows(report["rows"], primary, args.model_id)
        if secondary:
            second = type_reconstruction(secondary, relation, args.year)
            by_entity = {row["entity"]: row for row in second["rows"]}
            for row in report["rows"]:
                other = by_entity.get(row["entity"])
                row["secondary_prediction"] = other["prediction"] if other else None
                row["secondary_score"] = other["raw_score"] if other else None
                row["model_agreement"] = row["prediction"] == other["prediction"] if other else None
            report["secondary_summary"] = {key: second[key] for key in ("reconstruction_accuracy", "macro_f1", "confusion_matrix")}
        reports.append(report)
    dump_json(args.output, {"primary": primary.audit, "secondary": secondary.audit if secondary else None, "reports": reports})
    write_csv(args.output.with_suffix(".csv"), [row for report in reports for row in report["rows"]])
    print("Type reconstruction: {}".format(args.output))


def manifest_command(args):
    report = {"created_at": datetime.now().isoformat(), "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "checkpoints": [audit_checkpoint(Path(value)) for value in args.checkpoints], "datasets": [audit_dataset(Path(value)) for value in args.datasets]}
    dump_json(args.output, report); print("Manifest: {}".format(args.output))


def model_args(parser):
    parser.add_argument("--model-id", choices=sorted(json.loads((APP_DIR / "model_registry.json").read_text(encoding="utf-8"))))
    parser.add_argument("--dataset", type=Path); parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--stochastic-eval", action="store_true")


def secondary_args(parser):
    parser.add_argument("--secondary-model-id", choices=sorted(json.loads((APP_DIR / "model_registry.json").read_text(encoding="utf-8"))))
    parser.add_argument("--secondary-dataset", type=Path); parser.add_argument("--secondary-checkpoint", type=Path)


def map_args(parser):
    parser.add_argument("--boundaries", type=Path, required=True); parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--web-data", type=Path, default=APP_DIR / "web" / "data.geojson")


def build_parser():
    root = argparse.ArgumentParser(description="HMNCE inference and Hubei map toolkit")
    commands = root.add_subparsers(dest="command", required=True)
    item = commands.add_parser("audit-data"); item.add_argument("--datasets-root", type=Path, default=DATASETS_DIR / "Geo-NR"); item.add_argument("--datasets", nargs="+", default=["Geo-NR-5K", "Geo-NR-50K"]); item.add_argument("--output", type=Path, default=OUTPUT_DIR / "data_audit.json"); item.set_defaults(func=audit_data_command)
    item = commands.add_parser("audit-checkpoint"); item.add_argument("checkpoints", nargs="+"); item.add_argument("--output", type=Path, default=OUTPUT_DIR / "checkpoint_audit.json"); item.set_defaults(func=audit_checkpoint_command)
    item = commands.add_parser("audit-pair"); item.add_argument("--dataset", type=Path, required=True); item.add_argument("--checkpoint", type=Path, required=True); item.add_argument("--output", type=Path, default=OUTPUT_DIR / "pair_audit.json"); item.set_defaults(func=audit_pair_command)
    item = commands.add_parser("manifest"); item.add_argument("checkpoints", nargs="+"); item.add_argument("--datasets", nargs="+", type=Path, required=True); item.add_argument("--output", type=Path, default=OUTPUT_DIR / "run_manifest.json"); item.set_defaults(func=manifest_command)
    item = commands.add_parser("strict-load"); model_args(item); secondary_args(item); item.add_argument("--output", type=Path, default=OUTPUT_DIR / "checkpoint_audit.json"); item.set_defaults(func=strict_load_command)
    item = commands.add_parser("infer"); model_args(item); secondary_args(item); item.add_argument("--head", required=True); item.add_argument("--relation", required=True); item.add_argument("--top-k", type=int, default=5); item.add_argument("--year", type=int); item.add_argument("--filter-known", action="store_true"); item.add_argument("--output", type=Path, default=OUTPUT_DIR / "inference.csv"); item.set_defaults(func=infer_command)
    item = commands.add_parser("evaluate"); model_args(item); item.add_argument("--split", choices=("valid", "test"), default="test"); item.add_argument("--training-prefix", action="store_true"); item.add_argument("--evaluation-repeats", type=int, default=1); item.add_argument("--max-queries", type=int); item.add_argument("--batch-size", type=int, default=32); item.add_argument("--output", type=Path, default=OUTPUT_DIR / "evaluation.json"); item.set_defaults(func=evaluate_command)
    item = commands.add_parser("deterministic-test"); model_args(item); item.add_argument("--head", default="武汉市_2023"); item.add_argument("--relation", default="GDP_all_comp"); item.add_argument("--output", type=Path, default=OUTPUT_DIR / "deterministic_test.json"); item.set_defaults(func=deterministic_test_command)
    item = commands.add_parser("reconstruct-types"); model_args(item); secondary_args(item); item.add_argument("--relations", nargs="+", choices=sorted(TYPE_CANDIDATES), default=sorted(TYPE_CANDIDATES)); item.add_argument("--year", type=int); item.add_argument("--output", type=Path, default=OUTPUT_DIR / "type_reconstruction.json"); item.set_defaults(func=reconstruct_types_command)
    item = commands.add_parser("evaluate-comparisons"); model_args(item); secondary_args(item); item.add_argument("--years", nargs="+", type=int); item.add_argument("--relations", nargs="+", choices=COMPARISON_RELATIONS); item.add_argument("--threshold", type=float, default=0.5); item.add_argument("--output", type=Path, default=OUTPUT_DIR / "comparison_evaluation.json"); item.set_defaults(func=evaluate_comparisons_command)
    item = commands.add_parser("export-web-bundle"); model_args(item); secondary_args(item); item.add_argument("--boundaries", type=Path, required=True); item.add_argument("--years", nargs="+", type=int, required=True); item.add_argument("--type-relations", nargs="+", choices=sorted(TYPE_CANDIDATES), default=sorted(TYPE_CANDIDATES)); item.add_argument("--relations", nargs="+", choices=COMPARISON_RELATIONS, default=COMPARISON_RELATIONS); item.add_argument("--anchors", nargs="+", choices=sorted(CITY_COORDINATES), default=sorted(CITY_COORDINATES)); item.add_argument("--threshold", type=float, default=0.5); item.add_argument("--output", type=Path, default=APP_DIR / "web" / "data"); item.set_defaults(func=export_web_bundle_command)
    item = commands.add_parser("export-types"); model_args(item); secondary_args(item); map_args(item); item.add_argument("--relation", choices=sorted(TYPE_CANDIDATES), default="industry_type"); item.add_argument("--output", type=Path, default=OUTPUT_DIR / "type_map.geojson"); item.set_defaults(func=export_types_command)
    item = commands.add_parser("export-ranking"); model_args(item); map_args(item); item.add_argument("--relation", choices=COMPARISON_RELATIONS, required=True); item.add_argument("--output", type=Path, default=OUTPUT_DIR / "ranking_map.geojson"); item.set_defaults(func=export_ranking_command)
    item = commands.add_parser("export-comparison"); model_args(item); secondary_args(item); map_args(item); item.add_argument("--relation", choices=COMPARISON_RELATIONS, required=True); item.add_argument("--anchor", choices=sorted(CITY_COORDINATES), required=True); item.add_argument("--threshold", type=float, default=0.5); item.add_argument("--output", type=Path, default=OUTPUT_DIR / "comparison_map.geojson"); item.set_defaults(func=export_comparison_command)
    return root


if __name__ == "__main__":
    arguments = build_parser().parse_args(); arguments.func(arguments)
