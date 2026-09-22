from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from core import (
    ATTRIBUTE_NAMES, COMPARISON_RELATIONS, RELATION_TO_ATTRIBUTE,
    TYPE_CANDIDATES, identify_model, is_city_year, model_state,
    parse_city_year, read_pickle, read_triples, sha256_file, torch_load,
)


def audit_dataset(directory):
    entities = read_pickle(directory / "entities.dict")
    relations = read_pickle(directory / "relations.dict")
    rows = {name: read_triples(directory / (name + ".txt")) for name in ("train", "valid", "test", "neg")}
    city_entities = [entity for entity in entities if is_city_year(entity)]
    literals = np.load(str(directory / "literals" / "numerical_literals.npy"))
    return {
        "dataset": directory.name, "entities": len(entities), "relations": len(relations),
        "city_year_entities": len(city_entities),
        "cities": sorted(set(parse_city_year(x)[0] for x in city_entities)),
        "years": sorted(set(parse_city_year(x)[1] for x in city_entities)),
        "splits": {name: len(value) for name, value in rows.items()},
        "duplicates": {name: len(value) - len(set(value)) for name, value in rows.items()},
        "relation_counts": {name: dict(Counter(r for _, r, _ in value)) for name, value in rows.items()},
        "literal_shape": list(literals.shape),
        "file_sha256": {
            name: sha256_file(directory / name)
            for name in ("entities.dict", "relations.dict", "train.txt", "valid.txt", "test.txt", "literals/numerical_literals.npy")
        },
        "semantics": "head numeric value is strictly greater than tail numeric value for *_all_comp",
        "type_warning": "industry_type and gdp_tier occur in train only; report reconstruction, not held-out prediction",
    }


def audit_checkpoint(path):
    loaded = torch_load(path)
    state = model_state(loaded)
    signatures = ["num_embedder.missing_bias", "str.gate_U_prime.weight", "str.temp_layer.weight", "str.res_gate.weight", "str.norm.weight"]
    model_args = loaded.get("args", {})
    if model_args and not isinstance(model_args, dict):
        model_args = vars(model_args)
    entity_shape = list(state.get("emb_e", np.empty((0,))).shape)
    relation_shape = list(state.get("emb_rel", np.empty((0,))).shape)
    return {
        "checkpoint": str(path.resolve()), "sha256": sha256_file(path),
        "identified_model": identify_model(state),
        "model_signatures": {key: key in state for key in signatures},
        "parameter_count": int(sum(value.numel() for value in state.values() if hasattr(value, "numel"))),
        "entity_embedding_shape": entity_shape, "relation_embedding_shape": relation_shape,
        "best_epoch": loaded.get("best_epoch"), "best_val": loaded.get("best_val"),
        "args": model_args,
    }


def filtered_rank(scores, target_id, filtered_ids):
    values = scores.copy()
    target_score = values[target_id]
    for entity_id in filtered_ids:
        if entity_id != target_id:
            values[entity_id] = -np.inf
    return int(1 + np.sum(values > target_score))


def ranking_metrics(predictor, split="test", limit=None, training_prefix=False, batch_size=32):
    triples = predictor.bundle.triples[split]
    if training_prefix:
        triples = triples[:len(predictor.bundle.triples["train"]) + 2]
    if limit is not None:
        triples = triples[:limit]
    aggregate = defaultdict(lambda: {"count": 0, "mr": 0.0, "mrr": 0.0, "hits@1": 0, "hits@3": 0, "hits@10": 0})
    for start in range(0, len(triples), batch_size):
        batch = triples[start:start + batch_size]
        scores = predictor.scores([x[0] for x in batch], [x[1] for x in batch], batch_size)
        for index, (head, relation, tail) in enumerate(batch):
            target_id = predictor.bundle.entity_to_id[tail]
            known = predictor.bundle.tails_by_query.get((head, relation), set())
            filtered = [predictor.bundle.entity_to_id[value] for value in known]
            rank = filtered_rank(scores[index], target_id, filtered)
            for key in ("overall", relation):
                item = aggregate[key]
                item["count"] += 1; item["mr"] += rank; item["mrr"] += 1.0 / rank
                for cutoff in (1, 3, 10):
                    item["hits@{}".format(cutoff)] += int(rank <= cutoff)
    output = {}
    for key, item in aggregate.items():
        count = item["count"] or 1
        output[key] = {"count": item["count"], "mr": item["mr"] / count, "mrr": item["mrr"] / count}
        output[key].update({"hits@{}".format(k): item["hits@{}".format(k)] / count for k in (1, 3, 10)})
    return output


def classification_summary(rows, labels):
    matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for row in rows:
        if row.get("actual") in matrix:
            matrix[row["actual"]][row["prediction"]] += 1
    f1_values = []
    for label in labels:
        true_positive = matrix[label][label]
        false_positive = sum(matrix[actual][label] for actual in labels if actual != label)
        false_negative = sum(matrix[label][predicted] for predicted in labels if predicted != label)
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(2.0 * true_positive / denominator if denominator else 0.0)
    known = [row for row in rows if row.get("actual") is not None]
    accuracy = float(np.mean([row["correct"] for row in known])) if known else None
    return {"accuracy": accuracy, "macro_f1": float(np.mean(f1_values)), "confusion_matrix": matrix}


def binary_metrics(labels, scores, threshold=0.5):
    labels = np.asarray(labels, dtype=np.int64); scores = np.asarray(scores, dtype=np.float64)
    predictions = scores >= threshold
    tp = int(np.sum((predictions == 1) & (labels == 1))); fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1))); tn = int(np.sum((predictions == 0) & (labels == 0)))
    accuracy = float((tp + tn) / len(labels)) if len(labels) else None
    f1 = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0
    brier = float(np.mean((scores - labels) ** 2)) if len(labels) else None
    positives, negatives = int(labels.sum()), int((1 - labels).sum())
    if positives and negatives:
        positive_scores = scores[labels == 1]
        negative_scores = scores[labels == 0]
        comparisons = positive_scores[:, None] - negative_scores[None, :]
        auroc = float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)
    else:
        auroc = None
    descending = np.argsort(-scores); sorted_labels = labels[descending]; cumulative = np.cumsum(sorted_labels)
    precision = cumulative / np.arange(1, len(labels) + 1)
    auprc = float((precision * sorted_labels).sum() / positives) if positives else None
    return {"count": len(labels), "positive": positives, "negative": negatives, "threshold": threshold, "accuracy": accuracy, "f1": f1, "auroc": auroc, "auprc": auprc, "brier": brier, "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}}


def comparison_evaluation(predictor, years=None, relations=None, threshold=0.5):
    years = years or sorted(set(parse_city_year(entity)[1] for entity in predictor.bundle.entity_to_id if is_city_year(entity)))
    relations = relations or COMPARISON_RELATIONS
    rows = []
    for relation in relations:
        for year in years:
            heads = sorted(entity for entity in predictor.bundle.entity_to_id if parse_city_year(entity)[1] == year)
            scores = predictor.scores(heads, [relation] * len(heads))
            attribute_index = RELATION_TO_ATTRIBUTE[relation]
            for head_index, head in enumerate(heads):
                head_city, _ = parse_city_year(head); head_value = float(predictor.bundle.raw_literals[predictor.bundle.entity_to_id[head], attribute_index])
                candidates = [entity for entity in heads if entity != head]
                candidate_scores = [(tail, float(scores[head_index, predictor.bundle.entity_to_id[tail]])) for tail in candidates]
                candidate_scores.sort(key=lambda value: value[1], reverse=True)
                for rank, (tail, score) in enumerate(candidate_scores, 1):
                    tail_city, _ = parse_city_year(tail); tail_value = float(predictor.bundle.raw_literals[predictor.bundle.entity_to_id[tail], attribute_index])
                    rows.append({"head": head, "relation": relation, "tail": tail, "raw_score": score, "rank": rank, "head_city": head_city, "head_year": year, "tail_city": tail_city, "tail_year": year, "attribute": ATTRIBUTE_NAMES[attribute_index], "head_value": head_value, "tail_value": tail_value, "ground_truth": bool(head_value > tail_value)})
    grouped = {"overall": rows}
    grouped.update({relation: [row for row in rows if row["relation"] == relation] for relation in relations})
    metrics = {key: binary_metrics([row["ground_truth"] for row in values], [row["raw_score"] for row in values], threshold) for key, values in grouped.items()}
    return rows, metrics


def type_reconstruction(predictor, relation, year=None):
    categories = [x for x in TYPE_CANDIDATES[relation] if x in predictor.bundle.entity_to_id]
    targets = [
        target for target in sorted(predictor.bundle.entity_to_id)
        if is_city_year(target) and (year is None or parse_city_year(target)[1] == year)
    ]
    scores = predictor.scores(targets, [relation] * len(targets))
    category_ids = np.asarray(
        [predictor.bundle.entity_to_id[category] for category in categories],
        dtype=np.int64,
    )
    truth = {}
    for target, rel, category in predictor.bundle.triples["train"]:
        if rel == relation and target in predictor.bundle.entity_to_id and category in categories:
            truth[target] = category
    rows = []
    for target_index, target in enumerate(targets):
        city, target_year = parse_city_year(target)
        values = scores[target_index, category_ids]
        probabilities = np.exp(values - values.max()); probabilities /= probabilities.sum()
        order = np.argsort(-values)
        predicted = categories[int(order[0])]
        rows.append({
            "entity": target, "city": city, "year": target_year, "relation": relation,
            "prediction": predicted, "raw_score": float(values[order[0]]),
            "normalized_confidence": float(probabilities[order[0]]),
            "score_margin": float(values[order[0]] - values[order[1]]) if len(order) > 1 else None,
            "actual": truth.get(target), "correct": truth.get(target) == predicted if target in truth else None,
            "candidates": [{"category": categories[int(i)], "score": float(values[i]), "normalized": float(probabilities[i])} for i in order],
        })
    summary = classification_summary(rows, categories)
    return {"relation": relation, "year": year, "count": len(rows), "reconstruction_accuracy": summary["accuracy"], "macro_f1": summary["macro_f1"], "confusion_matrix": summary["confusion_matrix"], "warning": "Training reconstruction only; not held-out generalization", "rows": rows}


def city_ranking(predictor, relation, year):
    if relation not in COMPARISON_RELATIONS:
        raise ValueError("Unknown comparison relation: {}".format(relation))
    entities = sorted(
        entity for entity in predictor.bundle.entity_to_id
        if is_city_year(entity) and parse_city_year(entity)[1] == year
    )
    scores = predictor.scores(entities, [relation] * len(entities))
    entity_ids = [predictor.bundle.entity_to_id[entity] for entity in entities]
    score_matrix = scores[:, entity_ids]
    attribute_index = RELATION_TO_ATTRIBUTE[relation]
    actual_values = np.asarray([
        predictor.bundle.raw_literals[entity_id, attribute_index]
        for entity_id in entity_ids
    ], dtype=np.float64)
    actual_order = sorted(range(len(entities)), key=lambda i: (-actual_values[i], entities[i]))
    actual_ranks = {index: rank for rank, index in enumerate(actual_order, 1)}
    rows = []
    correct_pairs = 0
    total_pairs = 0
    for head_index, entity in enumerate(entities):
        comparisons = []
        advantages = []
        correct = 0
        for tail_index, opponent in enumerate(entities):
            if head_index == tail_index:
                continue
            forward = float(score_matrix[head_index, tail_index])
            reverse = float(score_matrix[tail_index, head_index])
            advantage = forward / max(forward + reverse, 1e-12)
            actual_higher = bool(actual_values[head_index] > actual_values[tail_index])
            predicted_higher = bool(advantage > 0.5)
            is_correct = predicted_higher == actual_higher
            correct += int(is_correct)
            advantages.append(advantage)
            comparisons.append({
                "opponent": opponent, "opponent_city": parse_city_year(opponent)[0],
                "forward_score": forward, "reverse_score": reverse,
                "pairwise_advantage": advantage, "predicted_higher": predicted_higher,
                "actual_higher": actual_higher, "correct": is_correct,
            })
            if head_index < tail_index:
                correct_pairs += int(is_correct)
                total_pairs += 1
        rows.append({
            "entity": entity, "city": parse_city_year(entity)[0], "year": year,
            "relation": relation, "attribute": ATTRIBUTE_NAMES[attribute_index],
            "ranking_score": float(np.mean(advantages)),
            "predicted_wins": int(sum(value > 0.5 for value in advantages)),
            "average_raw_support": float(np.mean([
                score_matrix[head_index, i] for i in range(len(entities)) if i != head_index
            ])),
            "actual_value": float(actual_values[head_index]),
            "actual_rank": actual_ranks[head_index],
            "pairwise_accuracy": float(correct / max(len(comparisons), 1)),
            "comparisons": sorted(comparisons, key=lambda item: item["pairwise_advantage"], reverse=True),
        })
    model_order = sorted(range(len(rows)), key=lambda i: (-rows[i]["ranking_score"], rows[i]["city"]))
    for model_rank, index in enumerate(model_order, 1):
        rows[index]["model_rank"] = model_rank
        rows[index]["rank_difference"] = model_rank - rows[index]["actual_rank"]
        rows[index]["rank_correct"] = model_rank == rows[index]["actual_rank"]
        rows[index]["prediction"] = "模型第 {} 名".format(model_rank)
        rows[index]["normalized_confidence"] = rows[index]["ranking_score"]
    model_ranks = np.asarray([row["model_rank"] for row in rows], dtype=np.float64)
    actual_rank_values = np.asarray([row["actual_rank"] for row in rows], dtype=np.float64)
    spearman = float(np.corrcoef(model_ranks, actual_rank_values)[0, 1]) if len(rows) > 1 else 1.0
    concordant = discordant = 0
    for first in range(len(rows)):
        for second in range(first + 1, len(rows)):
            product = (model_ranks[first] - model_ranks[second]) * (actual_rank_values[first] - actual_rank_values[second])
            concordant += int(product > 0); discordant += int(product < 0)
    kendall = float((concordant - discordant) / max(concordant + discordant, 1))
    model_top3 = {rows[index]["city"] for index in model_order[:3]}
    actual_top3 = {rows[index]["city"] for index in actual_order[:3]}
    summary = {
        "city_count": len(rows), "directed_comparisons": len(rows) * (len(rows) - 1),
        "unordered_pairs": total_pairs, "pairwise_accuracy": float(correct_pairs / max(total_pairs, 1)),
        "spearman": spearman, "kendall_tau": kendall,
        "top1_match": model_order[0] == actual_order[0],
        "top3_overlap": len(model_top3 & actual_top3),
        "mean_absolute_rank_error": float(np.mean(np.abs(model_ranks - actual_rank_values))),
        "ranking_method": "mean bidirectional normalized advantage",
    }
    rows.sort(key=lambda row: row["model_rank"])
    return {"relation": relation, "year": year, "summary": summary, "rows": rows}


def comparison_rows(predictor, relation, year, anchor_city):
    if relation not in COMPARISON_RELATIONS:
        raise ValueError("Unknown comparison relation: {}".format(relation))
    head = "{}_{}".format(anchor_city, year)
    rows = predictor.rank(head, relation, top_k=999, year=year)
    attribute_index = RELATION_TO_ATTRIBUTE[relation]
    head_value = float(predictor.bundle.raw_literals[predictor.bundle.entity_to_id[head], attribute_index])
    for row in rows:
        tail_city, tail_year = parse_city_year(row["tail"])
        tail_value = float(predictor.bundle.raw_literals[predictor.bundle.entity_to_id[row["tail"]], attribute_index])
        row.update({"head_city": anchor_city, "head_year": year, "tail_city": tail_city, "tail_year": tail_year, "attribute": ATTRIBUTE_NAMES[attribute_index], "head_value": head_value, "tail_value": tail_value, "ground_truth": bool(head_value > tail_value)})
    return rows


def ranking_agreement(primary_rows, secondary_rows, top_k=5):
    primary_by_query, secondary_by_query = defaultdict(list), defaultdict(list)
    for row in primary_rows:
        primary_by_query[(row["head"], row["relation"])].append(row)
    for row in secondary_rows:
        secondary_by_query[(row["head"], row["relation"])].append(row)
    values = []
    for query, first in primary_by_query.items():
        second = secondary_by_query.get(query, [])
        first.sort(key=lambda row: row["rank"]); second.sort(key=lambda row: row["rank"])
        first_top = [row["tail"] for row in first[:top_k]]; second_top = [row["tail"] for row in second[:top_k]]
        union = set(first_top) | set(second_top); intersection = set(first_top) & set(second_top)
        second_rank = {row["tail"]: row["rank"] for row in second}
        common = [row for row in first if row["tail"] in second_rank]
        if len(common) > 1:
            x = np.asarray([row["rank"] for row in common], dtype=float); y = np.asarray([second_rank[row["tail"]] for row in common], dtype=float)
            spearman = float(np.corrcoef(x, y)[0, 1])
        else:
            spearman = None
        values.append({"head": query[0], "relation": query[1], "top1_agreement": bool(first_top and second_top and first_top[0] == second_top[0]), "top_k": top_k, "top_k_jaccard": float(len(intersection) / len(union)) if union else 1.0, "spearman": spearman})
    valid_spearman = [row["spearman"] for row in values if row["spearman"] is not None and not np.isnan(row["spearman"])]
    summary = {"queries": len(values), "top1_agreement": float(np.mean([row["top1_agreement"] for row in values])) if values else None, "top_k_jaccard": float(np.mean([row["top_k_jaccard"] for row in values])) if values else None, "spearman": float(np.mean(valid_spearman)) if valid_spearman else None}
    return values, summary


def add_secondary(primary_rows, secondary_rows):
    secondary_by_key = {
        (row["head"], row["relation"], row["tail"]): row
        for row in secondary_rows
    }
    secondary_top = {}
    for row in secondary_rows:
        query = (row["head"], row["relation"])
        if query not in secondary_top or row["rank"] < secondary_top[query][0]:
            secondary_top[query] = (row["rank"], row["tail"])
    primary_top = {}
    for row in primary_rows:
        query = (row["head"], row["relation"])
        if query not in primary_top or row["rank"] < primary_top[query][0]:
            primary_top[query] = (row["rank"], row["tail"])
    for row in primary_rows:
        key = (row["head"], row["relation"], row["tail"])
        query = (row["head"], row["relation"])
        other = secondary_by_key.get(key)
        row["secondary_score"] = other["raw_score"] if other else None
        row["secondary_rank"] = other["rank"] if other else None
        row["model_agreement"] = bool(
            query in primary_top and query in secondary_top
            and primary_top[query][1] == secondary_top[query][1]
        )
    return primary_rows
