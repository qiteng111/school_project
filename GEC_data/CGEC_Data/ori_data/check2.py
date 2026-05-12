import json
from pathlib import Path
from collections import Counter

INPUT_PATH = Path("/mnt/common/intern/qt/Data_project/CGEC_Data/ori_data/couple_output_edits_explain_no_clean.json")  # 输入文件路径
OUTPUT_PATH = Path("/mnt/common/intern/qt/Data_project/CGEC_Data/ori_data/couple_output_edits_explain.json")  # 输出文件路径

# 定义有效的错误类型列表
VALID_ERROR_TYPES = [
    "错误标点", "空缺标点", "多余标点",
    "错字", "别字", "漏字", "多字", "繁体字", "异体字", "拼音字",
    "错词", "缺词", "多词", "外文词", "离合词",
    "多余主语", "多余谓语", "多余述语", "多余宾语", "多余补语", "多余定语", "多余状语", "多余中心语",
    "残缺主语", "残缺谓语", "残缺述语", "残缺宾语", "残缺补语", "残缺定语", "残缺状语", "残缺中心语",
    "把字句", "被字句", "比字句", "连字句", "有字句", "是字句", "“是……的”句", "存现句", "兼语句",
    "连动句", "双宾语句", "形容词谓语句", "语序错误", "词语重叠错误", "固定格式错误", "句式杂糅错误", "未完句",
    "其他错误",
]
VALID_ERROR_SET = set(VALID_ERROR_TYPES)

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def is_valid_item(item: dict) -> bool:
    """
    判断当前数据项是否有效：
    - 检查 edits 中的 error_type 是否全部在有效的错误类型中
    """
    edits = item.get("edits", [])
    
    # 检查 edits 中的 error_type 是否全部在有效的错误类型中
    for edit in edits:
        error_type = edit.get("error_type")
        if error_type not in VALID_ERROR_SET:
            return False  # 如果有不合法的错误类型，则跳过

    return True

def update_error_type(item: dict, error_count: Counter) -> dict:
    """
    将出现次数小于 10 的错误类型替换为“其他错误”
    """
    for edit in item.get("edits", []):
        error_type = edit.get("error_type")
        if error_count[error_type] < 10:
            edit["error_type"] = "其他错误"

    return item

def main():
    # 加载数据
    data = load_json(INPUT_PATH)

    if not isinstance(data, list):
        raise ValueError("输入文件应为 JSON 数组格式。")

    cleaned = []
    skipped_invalid = 0

    # 统计所有错误类型出现的次数
    error_count = Counter()
    for item in data:
        if is_valid_item(item):
            errors = [edit["error_type"] for edit in item.get("edits", [])]
            error_count.update(errors)
        else:
            skipped_invalid += 1

    # 打印出出现次数小于10的错误类型
    rare_errors = {error: count for error, count in error_count.items() if count < 10}
    if rare_errors:
        print("出现次数小于10的错误类型：")
        for error, count in rare_errors.items():
            print(f"{error}: {count}")

    # 更新错误类型，并保留原数据
    for item in data:
        if is_valid_item(item):
            cleaned.append(update_error_type(item, error_count))
    
    # 保存清洗后的数据
    save_json(cleaned, OUTPUT_PATH)

    print(f"原始样本数: {len(data)}")
    print(f"保留样本数: {len(cleaned)}")
    print(f"跳过无效数据: {skipped_invalid}")
    print(f"输出文件: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()