import json
import random
import argparse
from collections import Counter
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def char_tokenize(text):
    """
    按字符切分：
    - "由于" -> ["由", "于"]
    - "" -> []
    """
    if not text:
        return []
    return list(text)


def normalize_edit(edit):
    """
    规范化单条 edit：
    1. 保留原字段
    2. 如果没有 src_tokens / tgt_tokens，就自动生成
    3. 如果给了但为空，也自动生成
    """
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
        "src_content": src_content,
        "tgt_content": tgt_content,
        "src_tokens": src_tokens,
        "tgt_tokens": tgt_tokens,
        "error_type": edit.get("error_type", ""),
        "error_severity": edit.get("error_severity", ""),
        "error_description": edit.get("error_description", ""),
        "teach_point": edit.get("teach_point", "")
    }


def transform_sample(item, default_domain="custom"):
    edits = item.get("edits", [])
    new_edits = [normalize_edit(e) for e in edits]

    return {
        "index": item.get("id"),
        # "domain": item.get("domain", default_domain),
        "source": item.get("error_sentence", ""),
        "target": item.get("correct_sentence", ""),
        "edits": new_edits
    }


def build_metadata(samples, version):
    type_counter = Counter()
    severity_counter = Counter()

    for sample in samples:
        for e in sample.get("edits", []):
            error_type = e.get("error_type", "")
            severity = e.get("error_severity", "")

            if error_type != "":
                type_counter[str(error_type)] += 1
            if severity != "":
                severity_counter[str(severity)] += 1

    return {
        "number": len(samples),
        "version": version,
        "type_counter": dict(type_counter),
        "severity_counter": dict(severity_counter)
    }


def pack_dataset(samples, version):
    return {
        "metadata": build_metadata(samples, version),
        "samples": samples
    }


def split_samples(samples, train_ratio=0.8, eval_ratio=0.1, test_ratio=0.1, seed=42):
    total = train_ratio + eval_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train_ratio + eval_ratio + test_ratio 必须等于 1.0")

    samples_copy = samples[:]
    random.Random(seed).shuffle(samples_copy)

    n = len(samples_copy)
    n_train = int(n * train_ratio)
    n_eval = int(n * eval_ratio)

    train_samples = samples_copy[:n_train]
    eval_samples = samples_copy[n_train:n_train + n_eval]
    test_samples = samples_copy[n_train + n_eval:]

    return train_samples, eval_samples, test_samples


def main():
    parser = argparse.ArgumentParser(description="将原始纠错数据转换为目标格式，并切分 train/eval/test")
    parser.add_argument("--input", type=str, default="a.json", help="输入文件路径")
    parser.add_argument("--output_dir", type=str, default="output_data", help="输出目录")
    parser.add_argument("--version", type=str, default="20260413_custom", help="版本号")
    parser.add_argument("--domain", type=str, default="custom", help="默认 domain")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--eval_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_data = load_json(args.input)

    transformed_samples = [
        transform_sample(item, default_domain=args.domain)
        for item in raw_data
    ]

    full_dataset = pack_dataset(transformed_samples, args.version)
    save_json(full_dataset, output_dir / "all_transformed.json")

    train_samples, eval_samples, test_samples = split_samples(
        transformed_samples,
        train_ratio=args.train_ratio,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    save_json(pack_dataset(train_samples, args.version), output_dir / "train.json")
    save_json(pack_dataset(eval_samples, args.version), output_dir / "valid.json")
    save_json(pack_dataset(test_samples, args.version), output_dir / "test.json")

    print("转换完成。")
    print(f"总样本数: {len(transformed_samples)}")
    print(f"Train: {len(train_samples)}")
    print(f"Eval : {len(eval_samples)}")
    print(f"Test : {len(test_samples)}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()