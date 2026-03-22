# import json

# # 读取原始JSON文件
# input_file = '/home/s202507015/workspace/EXCGEC/exp-cgec/data/train.json'  # 输入文件路径
# output_file = '/home/s202507015/workspace/EXCGEC/exp-cgec/data/train_GC.json'  # 输出文件路径

# # 读取原始JSON文件
# with open(input_file, 'r', encoding='utf-8') as f:
#     data = json.load(f)

# # 转换为微调所需格式
# formatted_data = []

# for sample in data['samples']:

#     input_text = sample['source']
#     output_text = sample['target']
        
#     # 判断句子是否有语法错误，并修正
#     instruction = "判断以下句子是否存在语法错误并修正。"
    
#     if sample['edits']:
#         # 构建目标格式数据
#         formatted_sample = {
#             "instruction": instruction,
#             "input": input_text,
#             "output": f"该句存在语法错误。正确的表达应该是<{output_text}>"
#         }
            
#         # 添加到结果列表
#         formatted_data.append(formatted_sample)

# # 将结果保存为新文件
# with open(output_file, 'w', encoding='utf-8') as f:
#     json.dump(formatted_data, f, ensure_ascii=False, indent=4)

# print(f"数据已成功保存为 {output_file}")


# import json

# # 读取原始JSON文件
# input_file = '/home/s202507015/workspace/EXCGEC/exp-cgec/data/train.json'  # 输入文件路径
# output_file = '/home/s202507015/workspace/EXCGEC/exp-cgec/data/train_GCC.json'  # 输出文件路径

# # 读取原始JSON文件
# with open(input_file, 'r', encoding='utf-8') as f:
#     data = json.load(f)

# # 转换为微调所需格式
# formatted_data = []


# for sample in data['samples']:
#     for edit in sample['edits']:
#         # 获取源文本和目标文本
#         input_text = sample['source']
#         output_text = sample['target']
        
#         # 构建语法错误的描述
#         error_type = edit['error_type']
#         error_severity = edit['error_severity']
        
#         # 如果有多个错误类型，可以根据需要聚合描述
#         error_description = f"该句存在{len(sample['edits'])}处语法错误，分别是"
#         error_details = []
        
#         # 收集所有错误类型
#         for e in sample['edits']:
#             error_details.append(f"<<{e['error_type']}>>")
        
#         error_description += ''.join(error_details)
        
#         # 修正后的句子
#         corrected_output = f"正确的表达应该是<{output_text}>"

#     # 构建目标格式数据
#     formatted_sample = {
#         "instruction": "判断以下句子是否存在语法错误，假如存在语法错误，说明是什么语法错误，并修正过来。",
#         "input": input_text,
#         "output": f"{error_description}。{corrected_output}"
#     }
    
#     # 添加到结果列表
#     formatted_data.append(formatted_sample)

# # 将结果保存为新文件
# with open(output_file, 'w', encoding='utf-8') as f:
#     json.dump(formatted_data, f, ensure_ascii=False, indent=4)

# print(f"数据已成功保存为 {output_file}")


import json
input_file = '/home/s202507015/workspace/EXCGEC/exp-cgec/data/train.json'  # 输入文件路径
output_file = '/home/s202507015/workspace/EXCGEC/exp-cgec/my_GC_data/train_GCCE.json'  # 输出文件路径
# 读取原始数据
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 定义一个函数来处理样本数据并生成目标格式
def process_data(data):
    result = []
    
    for sample in data['samples']:
        # 合并多个修正内容
        corrections = []
        for edit in sample['edits']:
            corrections.append(f"该句存在{edit['error_type']}错误。{edit['error_description']}")
        
        # 构造目标格式，合并所有错误信息
        formatted_sample = {
            "instruction": "判断以下句子是否存在语法错误，假如存在语法错误，说明是什么语法错误，解释错误原因，并修正过来。",
            "input": sample['source'],
            "output": " ".join(corrections) + f" 正确的表达应该是{sample['target']}."
        }
        result.append(formatted_sample)
    
    return result


# 处理数据
processed_data = process_data(data)

# 保存处理后的数据到新文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=4)

print("数据处理完成并已保存为 'processed_data.json'")
