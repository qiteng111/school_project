import json
import os
from typing import Any, Dict, List


INSTRUCTION = "将以下文本进行语法纠错，并生成纠正后的句子、纠正相关的解释信息和教学点"

INPUT_PATH = "/home/s202507015/workspace/school_project/cgec_data/all_data/split_data/test.json"
OUTPUT_PATH_STR = "/home/s202507015/workspace/school_project/exp-cgec/data/splits/test_out_qt.json"
OUTPUT_PATH_OBJ = "/home/s202507015/workspace/school_project/exp-cgec/data/splits/test_out_check_fin_qt.json"


def load_data(input_path: str) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "samples" in data and isinstance(data["samples"], list):
        return data["samples"]

    raise ValueError(f"无法识别的输入 JSON 结构: {type(data)}")


def normalize_edit(edit: Dict[str, Any]) -> Dict[str, Any]:
    src_tokens = edit.get("src_tokens")
    tgt_tokens = edit.get("tgt_tokens")

    if src_tokens is None:
        src_tokens = list(edit.get("src_content", ""))
    if tgt_tokens is None:
        tgt_tokens = list(edit.get("tgt_content", ""))

    return {
        "src_interval": edit.get("src_interval", []),
        "tgt_interval": edit.get("tgt_interval", []),
        "src_tokens": src_tokens,
        "tgt_tokens": tgt_tokens,
    }


def build_explanation(edit: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "error_type": edit.get("error_type", ""),
        "error_severity": edit.get("error_severity", ""),
        "error_description": edit.get("error_description", ""),
        "teach_point": edit.get("teach_point", ""),
                }


def transform_sample(item: Dict[str, Any]) -> Dict[str, Any]:
    source = item.get("source", "")
    target = item.get("target", "")
    edits = item.get("edits", [])

    norm_edits = [normalize_edit(e) for e in edits if isinstance(e, dict)]
    explanations = [build_explanation(e) for e in edits if isinstance(e, dict)]

    output_obj = {
        "target": target,
        "edits": norm_edits,
        "explanations": explanations,
    }

    return {
        "instruction": INSTRUCTION,
        "input": source,
        "output": output_obj,
    }


def make_output_with_tgt(output_obj: Dict[str, Any]) -> str:
    """
    生成这种特殊格式：
    {"target":"...", <TGT>"edits":[...], "explanations":[...]}
    注意：这不是严格合法的 JSON，而是训练用模板字符串。
    """
    target_part = json.dumps(output_obj["target"], ensure_ascii=False)
    edits_part = json.dumps(output_obj["edits"], ensure_ascii=False)
    explanations_part = json.dumps(output_obj["explanations"], ensure_ascii=False)

    return (
        '{'
        f'"target": {target_part}, '
        f'<TGT>"edits": {edits_part}, '
        f'"explanations": {explanations_part}'
        '}'
    )


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    raw_samples = load_data(INPUT_PATH)

    obj_data = []
    skipped = 0

    for idx, item in enumerate(raw_samples):
        if not isinstance(item, dict):
            print(f"[跳过] 第 {idx} 条不是 dict")
            skipped += 1
            continue

        source = item.get("source")
        target = item.get("target")

        if not source or not target:
            print(f"[跳过] 第 {idx} 条缺少 source/target")
            skipped += 1
            continue

        obj_data.append(transform_sample(item))

    # 第一个输出：output 是带 <TGT> 的特殊字符串
    str_data = []
    for x in obj_data:
        str_data.append({
            "instruction": x["instruction"],
            "input": x["input"],
            "output": make_output_with_tgt(x["output"])
        })

    ensure_parent_dir(OUTPUT_PATH_STR)
    ensure_parent_dir(OUTPUT_PATH_OBJ)

    with open(OUTPUT_PATH_STR, "w", encoding="utf-8") as f:
        json.dump(str_data, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_PATH_OBJ, "w", encoding="utf-8") as f:
        json.dump(obj_data, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"字符串版输出: {OUTPUT_PATH_STR}")
    print(f"对象版输出: {OUTPUT_PATH_OBJ}")
    print(f"有效样本数: {len(obj_data)}")
    print(f"跳过样本数: {skipped}")


if __name__ == "__main__":
    main()