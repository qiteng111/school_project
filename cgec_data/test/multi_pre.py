import json

# 原始
with open(r"/share/project/xzfang/qiteng_data/8w_project/data/yes_file.json", "r", encoding="utf-8") as f:
    original_data = json.load(f)

with open(r"/share/project/xzfang/qiteng_data/8w_project/data_output/y_file.json", "r", encoding="utf-8") as f:
    rewritten_data = json.load(f)


rewrite_lookup = {
    (item["path"], item["idx"]): item for item in rewritten_data
}

# 合并数据
merged = []
for orig in original_data:
    key = (orig["path"], orig["idx"])
    if key in rewrite_lookup:
        rewrite = rewrite_lookup[key]
        merged.append({
            "original_Q": orig["Q"],
            "original_A": orig["A"],
            "rewritten_Q": rewrite["Q"],
            "rewritten_A": rewrite["A"],
            "path": orig["path"],
            "idx": orig["idx"]
        })

# 保存为 JSONL 格式
with open(r"/share/project/xzfang/qiteng_data/8w_project/data_output/check.jsonl", "w", encoding="utf-8") as f:
    for item in merged:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
