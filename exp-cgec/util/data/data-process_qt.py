import json
import argparse
- 错误标点、空缺标点、多余标点
- 错字、别字、漏字、多字、繁体字、异体字、拼音字
- 错词、缺词、多词、外文词、离合词
- 多余主语、多余谓语、多余述语、多余宾语、多余补语、多余定语、多余状语、多余中心语、残缺主语、残缺谓语、残缺述语、残缺宾语、残缺补语、残缺定语、残缺状语、残缺中心语、把字句、被字句、比字句、连字句、有字句、是字句、“是……的”句、存现句、兼语句、连动句、双宾语句、形容词谓语句、语序错误、词语重叠错误、固定格式错误、句式杂糅错误、未完句

VALID_ERROR_TYPES = [
    # 标点级别错误
    "错误标点",
    "空缺标点",
    "多余标点",
    # 字级别错误
    "错字",
    "别字",
    "漏字",
    "多字",
    "漏字",
    "繁体字",
    "异体字",
    "拼音字",
    # 词语级别错误
    "错词",
    "缺词",
    "多词",
    "外文词",
    "离合词",
    # 句法级别错误
    "多余主语",
    "多余谓语",
    "多余述语",
    "多余宾语",
    "多余补语",
    "多余定语", 
    "多余状语",
    "多余中心语",
    "残缺主语",
    "残缺谓语",
    "残缺述语",
    "残缺宾语",
    "残缺补语",
    "残缺定语",
    "残缺状语",
    "残缺中心语",
    "把字句",
    "被字句",
    "比字句",
    "连字句",
    "有字句",
    "是字句",
    "“是……的”句",
    "存现句",
    "兼语句",
    "连动句",   
    "双宾语句",
    "形容词谓语句",
    "语序错误",
    "词语重叠错误",
    "固定格式错误",
    "句式杂糅错误",
    "未完句",
    # 其他特殊错误
    "其他错误",
]


def process_json_file(input_file, output_file):
    # 读取文件内容
    with open(input_file, "r", encoding="utf-8") as file:
        content = file.read()

    # 去掉 <TGT> 标记
    content = content.replace("<TGT>", "")
    updated_content = (
        content.replace("<|im_end|>", "")
        .replace("<|endoftext|>", "")
        .replace("<|begin_of_text|>", "")
        .replace("<|eot_id|>", "")
    )
    updated_content = (
        updated_content.replace("<｜begin▁of▁sentence｜>", "")
        .replace("<｜end▁of▁sentence｜>", "")
        .replace("<|im_start|>", "")
        .replace("[gMASK] sop ", "")
    )
    updated_content = updated_content.replace("<s>", "").replace("</s>", "")
    # 输出去掉 <TGT> 标记后的内容

    # 将去掉 <TGT> 标记后的内容写回文件（如果需要）
    with open(input_file, "w", encoding="utf-8") as file:
        file.write(updated_content)

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    id = 0
    output_list = []
    for sample in data:
        try:
            # 尝试解析字符串为 JSON
            json_object = json.loads(sample["output"])

            # 处理 explanations 中的每个元素
            if "explanations" in json_object:
                for explanation in json_object["explanations"]:
                    # 检查 error_type 是否是合法的
                    if "error_type" in explanation:
                        if explanation["error_type"] not in VALID_ERROR_TYPES:
                            explanation["error_type"] = "其他错误"

                    # 检查 error_severity 是否是合法的
                    if "error_severity" in explanation:
                        if not isinstance(
                            explanation["error_severity"], int
                        ) or explanation["error_severity"] not in [1, 2, 3, 4, 5]:
                            explanation["error_severity"] = 1

            sample["output"] = json_object
            output_list.append(sample)
        except json.JSONDecodeError as e:
            json_object = {
                "target": sample["input"],
                "edits": [],
                "explanations": [],
            }
            sample["output"] = json_object
            output_list.append(sample)
            id += 1

    print("empty: ", id, "all: ", len(output_list))
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_list, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a JSON file.")

    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input JSON file."
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to the output JSON file."
    )

    args = parser.parse_args()

    process_json_file(args.input_file, args.output_file)
