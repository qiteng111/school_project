import json
from pathlib import Path

INPUT_PATH = Path("/mnt/common/intern/qt/Data_project/CGEC_Data/check_data/couple_output_check.json")
OUTPUT_PATH = Path("/mnt/common/intern/qt/Data_project/CGEC_Data/check_data/couple_data_cleaned.json")
VALID_ERROR_TYPES = [
    # 标点级别错误
    "错误标点",
    "空缺标点",
    "多余标点",

    # 字级别错误
    "错字",
    "漏字",
    "繁体字",
    # "拼音字",

    # 词语级别错误
    "错词",
    "缺词",
    "多词",

    # 句法级别错误：多余成分
    "多余主语",
    "多余述语",
    "多余补语",
    "多余定语",
    "多余状语",
    "多余中心语",

    # 句法级别错误：残缺成分
    "残缺主语",
    "残缺述语",
    "残缺宾语",
    "残缺补语",
    "残缺定语",
    "残缺状语",
    "残缺中心语",

    # 句式类错误
    "把字句",
    "被字句",
    # "是字句",
    "“是……的”句",
    "句式杂糅错误",
]

VALID_ERROR_SET = set(VALID_ERROR_TYPES)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def contains_forbidden_marker(text: str) -> bool:
    if not isinstance(text, str):
        return True
    return ("[" in text) or ("{" in text)


def validate_error_field(error_value: str):
    """
    返回:
    - (cleaned_error, True) 表示合法
    - (None, False) 表示非法，应整条跳过

    规则：
    1. error 不能为空
    2. 按 '、' 切分后，不能为空列表
    3. 只要出现任何一个非法类型，整条样本跳过
    """
    if not isinstance(error_value, str):
        return None, False

    error_value = error_value.strip()
    if not error_value:
        return None, False

    parts = [x.strip() for x in error_value.split("、") if x.strip()]
    if not parts:
        return None, False

    for p in parts:
        if p not in VALID_ERROR_SET:
            return None, False

    return "、".join(parts), True


def transform_item(item: dict, cleaned_error: str) -> dict:
    return {
        "source": item.get("source", "").strip(),
        "target": item.get("target", "").strip(),
        "error": cleaned_error
    }


def main():
    data = load_json(INPUT_PATH)

    if not isinstance(data, list):
        raise ValueError("输入文件应为 JSON 数组格式。")

    cleaned = []

    skipped_low_score = 0
    skipped_forbidden_marker = 0
    skipped_empty_error = 0
    skipped_illegal_error = 0

    for item in data:
        score = item.get("score", 0)
        source = item.get("source", "")
        target = item.get("target", "")
        error = item.get("error", "")

        # 1. score < 80 跳过
        if not isinstance(score, (int, float)) or score < 85:
            skipped_low_score += 1
            continue

        # 2. source / target 含 [ 或 { 跳过
        if contains_forbidden_marker(source) or contains_forbidden_marker(target):
            skipped_forbidden_marker += 1
            continue

        # 3. error 校验：为空跳过；含非法类型整条跳过
        cleaned_error, ok = validate_error_field(error)
        if not ok:
            if not isinstance(error, str) or not error.strip() or not [x.strip() for x in str(error).split("、") if x.strip()]:
                skipped_empty_error += 1
            else:
                skipped_illegal_error += 1
            continue

        cleaned.append(transform_item(item, cleaned_error))

    save_json(cleaned, OUTPUT_PATH)

    print(f"原始样本数: {len(data)}")
    print(f"保留样本数: {len(cleaned)}")
    print(f"因 score < 80 跳过: {skipped_low_score}")
    print(f"因 source/target 含 [ 或 {{ 跳过: {skipped_forbidden_marker}")
    print(f"因 error 为空跳过: {skipped_empty_error}")
    print(f"因 error 含非法类型跳过: {skipped_illegal_error}")
    print(f"输出文件: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()