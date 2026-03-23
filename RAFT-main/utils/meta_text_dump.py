import json
import os
from typing import Dict, List, Optional


def _dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def dump_meta_texts(
    dataset_name: str,
    model_id: str,
    output_dir: str,
    split_texts: Dict[str, List[str]],
    split_texts_by_period: Optional[Dict[str, List[List[str]]]] = None,
    max_samples: int = 200,
    template_catalog: Optional[Dict[str, str]] = None,
):
    dataset_tag = dataset_name.replace("/", "_").replace("\\", "_")
    model_tag = model_id.replace("/", "_").replace("\\", "_")
    dump_dir = os.path.join(output_dir, dataset_tag)
    os.makedirs(dump_dir, exist_ok=True)

    payload = {
        "dataset_name": dataset_name,
        "model_id": model_id,
        "template_catalog": template_catalog or {},
        "splits": {},
        "periodic_splits": {},
    }

    for split, texts in split_texts.items():
        texts = [str(t) for t in texts]
        uniques = _dedup_keep_order(texts)
        payload["splits"][split] = {
            "total": len(texts),
            "unique_total": len(uniques),
            "samples": texts[:max_samples],
            "unique_samples": uniques[:max_samples],
        }

    if split_texts_by_period:
        for split, rows in split_texts_by_period.items():
            rows = rows or []
            if len(rows) == 0:
                payload["periodic_splits"][split] = {
                    "total": 0,
                    "period_count": 0,
                    "samples": [],
                }
                continue

            period_count = max(len(r) for r in rows)
            norm_rows = []
            for r in rows:
                rr = [str(x) for x in r]
                if len(rr) < period_count:
                    rr += [rr[-1] if len(rr) > 0 else "" for _ in range(period_count - len(rr))]
                norm_rows.append(rr[:period_count])

            payload["periodic_splits"][split] = {
                "total": len(norm_rows),
                "period_count": period_count,
                "samples": norm_rows[:max_samples],
            }

    json_path = os.path.join(dump_dir, f"{model_tag}_meta_texts.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    txt_path = os.path.join(dump_dir, f"{model_tag}_meta_texts.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Model ID: {model_id}\n\n")
        if template_catalog:
            f.write("Template Catalog:\n")
            for k, v in template_catalog.items():
                f.write(f"- {k}: {v}\n")
            f.write("\n")

        for split in ["train", "valid", "test"]:
            split_obj = payload["splits"].get(split, {})
            f.write(f"[{split}] total={split_obj.get('total', 0)} unique={split_obj.get('unique_total', 0)}\n")
            for i, text in enumerate(split_obj.get("samples", []), 1):
                f.write(f"{i:04d}. {text}\n")
            f.write("\n")

        periodic = payload.get("periodic_splits", {})
        if periodic:
            f.write("Per-Period Text Samples:\n")
            for split in ["train", "valid", "test"]:
                pobj = periodic.get(split, {})
                rows = pobj.get("samples", [])
                f.write(f"[{split}] total={pobj.get('total', 0)} period_count={pobj.get('period_count', 0)}\n")
                for i, row in enumerate(rows, 1):
                    joined = " || ".join([f"p{j}:{txt}" for j, txt in enumerate(row)])
                    f.write(f"{i:04d}. {joined}\n")
                f.write("\n")

    return json_path, txt_path
