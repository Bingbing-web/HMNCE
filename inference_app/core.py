from __future__ import annotations

import hashlib
import pickle
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Set, Tuple

import numpy as np
import torch

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
DATASETS_DIR = PROJECT_DIR / "datasets"
OUTPUT_DIR = APP_DIR / "outputs"
for directory in (APP_DIR, PROJECT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

TYPE_CANDIDATES = {
    "industry_type": ["服务业主导型", "工业主导型", "农业比重较高型", "均衡发展型"],
    "gdp_tier": ["大型经济体", "中型经济体", "小型经济体"],
}
ATTRIBUTE_NAMES = [
    "END_YEAR_RESIDENT_POPULATION", "REGISTERED_POPULATION", "POPULATION_DENSITY",
    "AVERAGE_WAGE_OF_EMPLOYEES", "GDP", "GDP_PER_CAPITA", "GDP_GROWTH_RATE",
    "PRIMARY_INDUSTRY_GDP_SHARE", "SECONDARY_INDUSTRY_GDP_SHARE",
    "TERTIARY_INDUSTRY_GDP_SHARE", "GENERAL_PUBLIC_BUDGET_REVENUE",
    "GENERAL_PUBLIC_BUDGET_EXPENDITURE", "INDUSTRIAL_ENTERPRISES_ABOVE_SCALE",
]
COMPARISON_RELATIONS = [name + "_all_comp" for name in ATTRIBUTE_NAMES]
RELATION_TO_ATTRIBUTE = {name: index for index, name in enumerate(COMPARISON_RELATIONS)}
CITY_CODES = {
    "武汉市": "420100", "黄石市": "420200", "十堰市": "420300", "宜昌市": "420500",
    "襄阳市": "420600", "鄂州市": "420700", "荆门市": "420800", "孝感市": "420900",
    "荆州市": "421000", "黄冈市": "421100", "咸宁市": "421200", "随州市": "421300",
}
CITY_COORDINATES = {
    "武汉市": (114.3054, 30.5931), "黄石市": (115.0389, 30.1997),
    "十堰市": (110.7989, 32.6292), "宜昌市": (111.2865, 30.6919),
    "襄阳市": (112.1224, 32.0090), "鄂州市": (114.8949, 30.3919),
    "荆门市": (112.1993, 31.0354), "孝感市": (113.9169, 30.9246),
    "荆州市": (112.2397, 30.3352), "黄冈市": (114.8724, 30.4537),
    "咸宁市": (114.3225, 29.8414), "随州市": (113.3826, 31.6902),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def read_triples(path):
    rows = []
    path = Path(path)
    if path.exists():
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                values = line.rstrip("\r\n").split("\t")
                if len(values) >= 3 and values[0] != "head":
                    rows.append(tuple(values[:3]))
    return rows


def parse_city_year(entity):
    city, separator, year = entity.rpartition("_")
    if not separator or city not in CITY_COORDINATES:
        return None, None
    try:
        return city, int(year)
    except ValueError:
        return None, None


def is_city_year(entity):
    city, year = parse_city_year(entity)
    return city is not None and year is not None


def torch_load(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def model_state(loaded):
    for key in ("model", "model_state_dict", "state_dict"):
        if key in loaded and isinstance(loaded[key], dict):
            return loaded[key]
    return loaded


def identify_model(state):
    keys = set(state)
    if {"str.res_gate.weight", "str.norm.weight"}.issubset(keys):
        return "unsupported"
    if {"num_embedder.missing_bias", "str.temp_layer.weight"}.issubset(keys):
        return "hmnce"
    if "str.gate_U_prime.weight" in keys:
        return "unsupported"
    return "unknown"


@dataclass
class DatasetBundle:
    directory: Path
    entity_to_id: Dict[str, int]
    relation_to_id: Dict[str, int]
    id_to_entity: Dict[int, str]
    raw_literals: np.ndarray
    triples: Dict[str, List[Tuple[str, str, str]]]
    tails_by_query: Dict[Tuple[str, str], Set[str]]
    candidates_by_relation: Dict[str, List[str]]


def load_bundle(directory):
    directory = Path(directory).resolve()
    entities = read_pickle(directory / "entities.dict")
    relations = read_pickle(directory / "relations.dict")
    triples = {name: read_triples(directory / (name + ".txt")) for name in ("train", "valid", "test", "neg")}
    tails, candidates = defaultdict(set), defaultdict(set)
    for head, relation, tail in triples["train"] + triples["valid"] + triples["test"]:
        tails[(head, relation)].add(tail)
        candidates[relation].add(tail)
    city_years = [entity for entity in entities if is_city_year(entity)]
    for relation in relations:
        if relation.endswith(("_all_comp", "_comp")):
            candidates[relation].update(city_years)
    literals = np.load(str(directory / "literals" / "numerical_literals.npy"), allow_pickle=True).astype(np.float32)
    return DatasetBundle(directory, entities, relations, {v: k for k, v in entities.items()}, literals, triples, dict(tails), {k: sorted(v) for k, v in candidates.items()})


def make_params(saved, device, stochastic_eval=False):
    values = {"init_dim": 200, "att_dim": 200, "head_num": 5, "drop": 0.7, "num_mixture": 5, "gamma": 9.0, "order": 0.25, "scale": 0.25, "top_k": 5, "numeric_density": 0.8, "seed": 12345}
    if saved:
        values.update(saved if isinstance(saved, dict) else vars(saved))
    values.update({"device": device, "stochastic_eval": stochastic_eval})
    return SimpleNamespace(**values)


def prepare_literals(raw, params):
    raw = raw.astype(np.float32, copy=True)
    normalized = (raw - raw.min(0)) / (raw.max(0) - raw.min(0) + 1e-8)
    state = np.random.get_state()
    np.random.seed(int(params.seed))
    mask = np.random.uniform(0, 1, size=normalized.shape) <= float(params.numeric_density)
    np.random.set_state(state)
    prepared = normalized * mask.astype(np.float32)
    return prepared, {"shape": list(prepared.shape), "seed": int(params.seed), "generator": "numpy global MT19937 (training-compatible)", "numeric_density": float(params.numeric_density), "observed": int(mask.sum()), "missing": int(mask.size - mask.sum()), "raw_sha256": sha256_array(raw), "normalized_sha256": sha256_array(normalized), "mask_sha256": sha256_array(mask.astype(np.uint8)), "prepared_sha256": sha256_array(prepared)}


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_model(checkpoint, bundle, device_name="cuda:0", stochastic_eval=False):
    checkpoint = Path(checkpoint).resolve()
    loaded = torch_load(checkpoint)
    weights = model_state(loaded)
    kind = identify_model(weights)
    entity_shape = tuple(weights["emb_e"].shape)
    relation_shape = tuple(weights["emb_rel"].shape)
    expected_entities = len(bundle.entity_to_id)
    expected_relations = len(bundle.relation_to_id)
    if entity_shape[0] != expected_entities or relation_shape[0] != expected_relations:
        raise ValueError(
            "Checkpoint/dataset mismatch: checkpoint embeddings are {}/{} but dataset {} has {}/{} entities/relations".format(
                entity_shape[0], relation_shape[0], bundle.directory.name,
                expected_entities, expected_relations,
            )
        )
    if kind != "hmnce":
        raise ValueError("Expected an HMNCE checkpoint, found {}".format(kind))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this checkpoint-compatible inference model")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    params = make_params(loaded.get("args", {}), device, stochastic_eval)
    if getattr(params, "hmnce_score_func", "transe") != "transe" or any(
        getattr(params, flag, False) for flag in ("disable_me", "disable_hcl", "disable_srm", "disable_hsm", "disable_hsa")
    ) or getattr(params, "ft", "numeric") != "numeric":
        raise ValueError("Case-study inference requires the full HMNCE model with TransE scoring and numeric inputs.")
    set_deterministic(int(params.seed))
    literals, literal_audit = prepare_literals(bundle.raw_literals, params)
    from hmnce_inference import HMNCEInference
    model = HMNCEInference(len(bundle.entity_to_id), len(bundle.relation_to_id), literals, params).to(device)
    # Literal inputs are initialized separately from learnable parameters.
    compatible_weights = {key: value for key, value in weights.items() if key not in {"numerical_literals", "literal_mask"}}
    model.load_state_dict(compatible_weights, strict=True)
    model.eval()
    audit = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "dataset": str(bundle.directory), "entity_count": expected_entities, "relation_count": expected_relations, "identified_model": kind, "strict_load": True, "deterministic_eval": not stochastic_eval, "best_epoch": loaded.get("best_epoch"), "best_val": loaded.get("best_val"), "args": {key: value for key, value in vars(params).items() if key != "device"}, "literal_input": literal_audit}
    return model, params, loaded, audit


class Predictor:
    def __init__(self, bundle, checkpoint, device="cuda:0", stochastic_eval=False):
        self.bundle = bundle
        self.model, self.params, self.state, self.audit = load_model(checkpoint, bundle, device, stochastic_eval)
        self.device = torch.device(device)

    @torch.no_grad()
    def scores(self, heads, relations, batch_size=32):
        results, count = [], len(self.bundle.entity_to_id)
        for start in range(0, len(heads), batch_size):
            batch_heads, batch_relations = heads[start:start + batch_size], relations[start:start + batch_size]
            head_ids = torch.tensor([self.bundle.entity_to_id[x] for x in batch_heads], dtype=torch.long, device=self.device)
            rel_ids = torch.tensor([self.bundle.relation_to_id[x] for x in batch_relations], dtype=torch.long, device=self.device)
            labels = torch.zeros((len(batch_heads), count), device=self.device)
            results.append(self.model(None, head_ids, rel_ids, labels, None, labels).cpu().numpy())
        return np.concatenate(results) if results else np.empty((0, count), np.float32)

    def rank(self, head, relation, top_k=5, year=None, filter_known=False):
        raw = self.scores([head], [relation])[0]
        candidates = list(self.bundle.candidates_by_relation.get(relation, []))
        if relation in COMPARISON_RELATIONS:
            candidates = [x for x in candidates if x != head and (year is None or parse_city_year(x)[1] == year)]
        split_sets = {name: set(self.bundle.triples[name]) for name in ("train", "valid", "test")}
        rows = []
        for tail in candidates:
            known = next((name for name, triples in split_sets.items() if (head, relation, tail) in triples), None)
            if filter_known and known:
                continue
            rows.append({"head": head, "relation": relation, "tail": tail, "raw_score": float(raw[self.bundle.entity_to_id[tail]]), "known_split": known, "is_new_candidate": known is None})
        rows.sort(key=lambda item: item["raw_score"], reverse=True)
        for index, row in enumerate(rows[:top_k], 1):
            row["rank"] = index
        return rows[:top_k]
