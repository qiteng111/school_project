import json
from difflib import SequenceMatcher
from typing import List, Dict, Any
import json
import argparse



def sent_to_tokens(text: str) -> List[str]:
    """按字符切分中文句子。"""
    return list(text)


def extract_edits_clean(source: str, target: str) -> List[Dict[str, Any]]:
    """
    从 source 和 target 提取更干净的 edit 结构：
    - src_interval
    - tgt_interval
    - src_content
    - tgt_content
    """
    src_tokens = sent_to_tokens(source)
    tgt_tokens = sent_to_tokens(target)

    sm = SequenceMatcher(a=src_tokens, b=tgt_tokens)
    edits = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        edits.append({
            "src_interval": [i1, i2],
            "tgt_interval": [j1, j2],
            "src_tokens": src_tokens[i1:i2],
            "tgt_tokens": tgt_tokens[j1:j2],
        })

    return edits


def apply_edits_for_check(source: str, edits: List[Dict[str, Any]]) -> str:
    """
    用干净版 edits 应回 source，验证是否能恢复 target。
    不写入最终输出，只在程序内部做校验。
    """
    src_tokens = sent_to_tokens(source)
    tgt_tokens = src_tokens.copy()
    offset = 0

    for edit in edits:
        i1, i2 = edit["src_interval"]
        j1 = i1 + offset
        j2 = i2 + offset

        replacement = list(edit["tgt_tokens"])
        tgt_tokens[j1:j2] = replacement

        offset += len(replacement) - (i2 - i1)

    return "".join(tgt_tokens)


def merge_adjacent_edits(
    source: str,
    target: str,
    edits: List[Dict[str, Any]],
    max_gap: int = 1
) -> List[Dict[str, Any]]:
    """
    合并相邻 edits，避免过碎。
    max_gap=1 等价于“间隔小于 2 个字符时合并”。
    """
    if not edits:
        return edits

    src_tokens = sent_to_tokens(source)
    tgt_tokens = sent_to_tokens(target)

    merged = [edits[0]]

    for cur in edits[1:]:
        prev = merged[-1]

        prev_end = prev["src_interval"][1]
        cur_beg = cur["src_interval"][0]

        if cur_beg - prev_end <= max_gap:
            new_src_interval = [prev["src_interval"][0], cur["src_interval"][1]]
            new_tgt_interval = [prev["tgt_interval"][0], cur["tgt_interval"][1]]

            merged[-1] = {
                "src_interval": new_src_interval,
                "tgt_interval": new_tgt_interval,
                "src_tokens": "".join(src_tokens[new_src_interval[0]:new_src_interval[1]]),
                "tgt_tokens": "".join(tgt_tokens[new_tgt_interval[0]:new_tgt_interval[1]]),
            }
        else:
            merged.append(cur)

    return merged


def process_file(
    input_path: str,
    output_path: str,
    merge_edits: bool = False
) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    skipped_empty_edits = 0
    results = []

    for idx, item in enumerate(data):
        source = item["input"]
        target_ori = item["output"]

        # 去掉标记
        target = (
            target_ori.replace("<|im_end|>", "")
            .replace("<|endoftext|>", "")
            .replace("<|begin_of_text|>", "")
            .replace("<|eot_id|>", "")
        )
        target = (
            target.replace("<｜begin▁of▁sentence｜>", "")
            .replace("<｜end▁of▁sentence｜>", "")
            .replace("<|im_start|>", "")
            .replace("[gMASK] sop ", "")
        )

        # 提取 edits 信息
        edits = extract_edits_clean(source, target)

        if merge_edits:
            edits = merge_adjacent_edits(source, target, edits, max_gap=1)

        if not edits:
            skipped_empty_edits += 1
            print(f"[跳过] 第 {idx} 条 edits 为空")
            continue

        # 内部校验
        recovered = apply_edits_for_check(source, edits)
        if recovered != target:
            raise ValueError(
                f"第 {idx} 条样本 edit 校验失败：\n"
                f"source={source}\n"
                f"target={target}\n"
                f"recovered={recovered}\n"
                f"edits={edits}"
            )

        # 生成处理后的数据格式
        result = {
            "instruction": "根据输入的错误句子、目标句子和编辑信息，为每一个编辑生成对应的错误类型、错误严重程度、错误描述和教学要点。",
            "input": json.dumps({
                "source": source,
                "target": target,
                "edits": edits
            }, ensure_ascii=False),
        }

        results.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"共处理 {len(data)} 条样本，跳过 {skipped_empty_edits} 条 edits 为空的样本。")
    print(f"Done. 输出文件: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a JSON file.")

    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input JSON file."
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to the output JSON file."
    )

    args = parser.parse_args()

    process_file(args.input_file, args.output_file,merge_edits=False)