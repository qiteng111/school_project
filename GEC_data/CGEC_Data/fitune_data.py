import json
import argparse
from pathlib import Path


DEFAULT_INSTRUCTION = "将以下文本进行语法纠错，并生成纠正后的句子、纠正相关的解释信息和教学点"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def char_tokenize(text):
    return list(text) if text else []


def normalize_edit(edit):
    src_content = edit.get("src_content", "")
    tgt_content = edit.get("tgt_content", "")

    src_tokens = edit.get("src_tokens")
    tgt_tokens = edit.get("tgt_tokens")

    if not src_tokens:
        src_tokens = char_tokenize(src_content)
    if not tgt_tokens:
        tgt_tokens = char_tokenize(tgt_content)

    return {
        "src_interval": edit.get("src_interval", []),
        "tgt_interval": edit.get("tgt_interval", []),
        "src_tokens": src_tokens,
        "tgt_tokens": tgt_tokens,
        "error_type": edit.get("error_type", ""),
        "error_severity": edit.get("error_severity", ""),
        "error_description": edit.get("error_description", ""),
        "teach_point": edit.get("teach_point", "")
    }


def maybe_extract_samples(data):
    """
    兼容两种输入：
    1. list[dict]
    2. {"metadata": ..., "samples": [...]}
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    raise ValueError("输入文件格式不支持，应为样本列表或包含 samples 字段的数据集对象。")


def build_output_string(target, edits_part, explanations_part):
    """
    生成你要求的 output 字符串格式：
    "{\"target\": \"xxx\", <TGT>\"edits\": [...], \"explanations\": [...]}"
    注意：这里是训练模板字符串，不是严格 JSON。
    """
    target_str = json.dumps(target, ensure_ascii=False)
    edits_str = json.dumps(edits_part, ensure_ascii=False)
    explanations_str = json.dumps(explanations_part, ensure_ascii=False)

    output = (
        '{'
        f'"target": {target_str}, '
        '<TGT>'
        f'"edits": {edits_str}, '
        f'"explanations": {explanations_str}'
        '}'
    )
    return output


def convert_one_sample(sample, instruction=DEFAULT_INSTRUCTION):
    source = sample.get("error_sentence", sample.get("source", ""))
    target = sample.get("correct_sentence", sample.get("target", ""))
    raw_edits = sample.get("edits", [])

    norm_edits = [normalize_edit(e) for e in raw_edits]

    edits_part = []
    explanations_part = []

    for e in norm_edits:
        edits_part.append({
            "src_interval": e["src_interval"],
            "tgt_interval": e["tgt_interval"],
            "src_tokens": e["src_tokens"],
            "tgt_tokens": e["tgt_tokens"]
        })

        explanations_part.append({
            "error_type": e["error_type"],
            "error_severity": e["error_severity"],
            "error_description": e["error_description"],
            "teach_point": e["teach_point"]
        })

    output_str = build_output_string(target, edits_part, explanations_part)

    return {
        "instruction": instruction,
        "input": source,
        "output": output_str
    }


def main():
    parser = argparse.ArgumentParser(description="将纠错数据转换为带 <TGT> 标记的 instruction/input/output 格式")
    parser.add_argument("--input", type=str, required=True, help="输入 JSON 文件")
    parser.add_argument("--output", type=str, required=True, help="输出 JSON 文件")
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION, help="instruction 文本")
    args = parser.parse_args()

    data = load_json(args.input)
    samples = maybe_extract_samples(data)

    converted = [convert_one_sample(sample, instruction=args.instruction) for sample in samples]

    save_json(converted, args.output)
    print(f"已转换 {len(converted)} 条样本 -> {args.output}")


if __name__ == "__main__":
    main()