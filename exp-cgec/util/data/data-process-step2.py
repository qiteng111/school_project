import json
from difflib import SequenceMatcher
from typing import List, Dict, Any
import argparse


def fix_json_format(json_string: str) -> str:
    """修复 JSON 格式中的多余逗号问题"""
    # 去除字符串中的多余逗号
    if json_string.endswith(", }"):
        json_string = json_string[:-2] + "]}"
    return json_string


def convert_json_format(input_file, output_file):
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 处理数据
    processed_data = []

    for item in data:
        # print(f"Processing item with input: {item['input']}")  # 输出正在处理的输入内容
        
        # 解析原始输入内容
        input_text = json.loads(item["input"])  # 将字符串转换为字典
        
        # 尝试解析输出内容
        try:
            output_text = json.loads(fix_json_format(item["output"]))  # 修复并转换输出字符串
        except json.decoder.JSONDecodeError as e:
            print(f"Error decoding JSON for output: {e}")
            print(f"Invalid output JSON: {item['output']}")
            continue  # 跳过此条数据

        target = input_text.get("target", "")  # 从 input 字典中获取 target
        
        # 提取编辑信息
        edits = input_text.get("edits", [])
        edits_data = []
        for edit in edits:
            src_interval = edit.get("src_interval", [])
            tgt_interval = edit.get("tgt_interval", [])
            src_tokens = edit.get("src_tokens", [])
            tgt_tokens = edit.get("tgt_tokens", [])
            edits_data.append({
                "src_interval": src_interval,
                "tgt_interval": tgt_interval,
                "src_tokens": src_tokens,
                "tgt_tokens": tgt_tokens,
            })
        
        # 提取错误解释
        explanations = output_text.get("explanations", [])
        explanations_data = []
        for explanation in explanations:
            error_type = explanation.get("error_type", "")
            error_severity = explanation.get("error_severity", 0)
            error_description = explanation.get("error_description", "")
            teach_point = explanation.get("teach_point", "")
            explanations_data.append({
                "error_type": error_type,
                "error_severity": error_severity,
                "error_description": error_description,
                "teach_point": teach_point,
            })
        
        # 构建目标数据格式
        result = {
            "input": input_text["source"],  # 将 source 字段赋给 input
            "output": json.dumps({
                "target": target,
                "edits": edits_data,
                "explanations": explanations_data
            }, ensure_ascii=False)  # 防止出现中文转义
        }
        
        processed_data.append(result)
    print(f"Processed {len(processed_data)} items successfully.")
    # 将结果保存到输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    print(f"数据已成功保存为 {output_file}")


# 调用转换函数

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a JSON file.")

    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input JSON file."
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to the output JSON file."
    )

    args = parser.parse_args()

    convert_json_format(args.input_file, args.output_file)