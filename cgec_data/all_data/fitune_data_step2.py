import json
import argparse
from pathlib import Path

DEFAULT_INSTRUCTION = "回复输入句子的修正版本，修正所有语法和拼写错误。"
DEFAULT_INSTRUCTION2 = "根据输入的错误句子、目标句子和编辑信息，为每一个编辑生成对应的错误类型、错误严重程度、错误描述和教学要点。"


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

def build_output_string2(source,target, edits_part):
    """
    生成你要求的 output 字符串格式：
    "{\"target\": \"xxx\", <TGT>\"edits\": [...], \"explanations\": [...]}"
    注意：这里是训练模板字符串，不是严格 JSON。
    """
    source_str = json.dumps(source, ensure_ascii=False)
    target_str = json.dumps(target, ensure_ascii=False)
    edits_str = json.dumps(edits_part, ensure_ascii=False)

    output = (
        '{'
        f'"source": {source_str}, '
        f'"target": {target_str}, '
        f'"edits": {edits_str}'
        '}'
    )
    return output


def build_output_string3(explanations_part):
    """
    生成你要求的 output 字符串格式：
    "{\"target\": \"xxx\", <TGT>\"edits\": [...], \"explanations\": [...]}"
    注意：这里是训练模板字符串，不是严格 JSON。
    """
    explanations_str = json.dumps(explanations_part, ensure_ascii=False)


    output = (
        '{'
        f'"explanations": {explanations_str}'
        '}'
    )
    return output

def convert_to_train1_sample(sample, instruction=DEFAULT_INSTRUCTION):
    source = sample.get("error_sentence", sample.get("source", ""))
    target = sample.get("correct_sentence", sample.get("target", ""))

    return {
        "instruction": instruction,
        "input": source,
        "output": target
    }

def convert_to_train2_sample(sample, instruction=DEFAULT_INSTRUCTION2):
    source = sample.get("error_sentence", sample.get("source", ""))
    target = sample.get("correct_sentence", sample.get("target", ""))
    raw_edits = sample.get("edits", [])

    norm_edits = [normalize_edit(e) for e in raw_edits]

    edits = []
    explanations = []

    for e in norm_edits:
        edits.append({
            "src_interval": e["src_interval"],
            "tgt_interval": e["tgt_interval"],
            "src_tokens": e["src_tokens"],
            "tgt_tokens": e["tgt_tokens"]
        })

        explanations.append({
            "error_type": e["error_type"],
            "error_severity": e["error_severity"],
            "error_description": e["error_description"],
            "teach_point": e["teach_point"]
        })

    input_str = build_output_string2(source,target, edits)
    output_str = build_output_string3(explanations)

    return {
        "instruction": instruction,
        "input": input_str,
        "output": output_str
    }

def main():
    parser = argparse.ArgumentParser(description="将纠错数据转换为带 <TGT> 标记的 instruction/input/output 格式")
    parser.add_argument("--input", type=str, required=True, help="输入 JSON 文件")
    parser.add_argument("--output_train1", type=str, required=True, help="输出 train_1 JSON 文件")
    parser.add_argument("--output_train2", type=str, required=True, help="输出 train_2 JSON 文件")
    args = parser.parse_args()

    data = load_json(args.input)
    samples = maybe_extract_samples(data)

    # 分别转换成 train_1 和 train_2 格式
    train1_samples = [convert_to_train1_sample(sample) for sample in samples]
    train2_samples = [convert_to_train2_sample(sample) for sample in samples]

    # 保存两个 JSON 文件
    save_json(train1_samples, args.output_train1)
    save_json(train2_samples, args.output_train2)

    print(f"已转换 {len(train1_samples)} 条样本 -> {args.output_train1}")
    print(f"已转换 {len(train2_samples)} 条样本 -> {args.output_train2}")

if __name__ == "__main__":
    main()