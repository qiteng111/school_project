import datetime
import os
from volcenginesdkarkruntime import Ark
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import machine_path,machine_prepare_a_exp,machine_a_exp_output,api_key,bot_cov_output,meta_output,data_relation_person_input,bot_cov_output,meta_path,data_relation_relation_output,data_relation_problem_input,data_relation_relation_input,model,person_fact_output,person_fact_input,client,name_filter_input_path,all_save_output2,info_prompt_path2,result1_input_path,result9_input_path,result_output_path,info_prompt_path,name_filter_output_path,rename_input_path,rename_output_path,timechange_input_path,person_input_path,problem_input_path,all_info_output_path,all_info_path,all_save_output
from model.roles import gen_cov,AsyncGenCov
from utils.cov_generator import data_prepare_machine,data_prepare_machine_exa,data_prepare_machine,data_prepare_meta,data_relation_allinfo,data_prepare_combine_relation,data_prepare_person_fact,get_all_info,data_prepare_combine,data_prepare_combine_bot_start,time_change,process_result,process_output_one,save_result,save_data,name_filter,generate_random_name,rename,data_prepare
from prompt.prompt import GEC_couple_check,GEC_data2,GEC_explain_knowl_new,GEC_explain_knowl,machine_perosonal,bot_cov,cov_relation, cov_relation2,cov_relation_yyq, TTs_cov,TTs_cov_bot_start
import asyncio
import httpx
import json
import time

# # 篇章错误类型：
# # CP：错篇
#NOTE 在数据预处理阶段，需要平衡数据

#NOTE 数据预处理：作文大于65分的，跳过错篇
# import csv
# input_file = "/mnt/common/intern/qt/school_project/cgec_data/all_data/search_result.csv"
# output_file = "/mnt/common/intern/qt/school_project/cgec_data/all_data/composition_score_gt_65.json"

# results = []

# with open(input_file, "r", encoding="utf-8-sig", newline="") as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         score_text = row.get("作文分数", "").strip()
#         sentence = row.get("检索原句", "").strip()

#         if not score_text:
#             continue
#         try:
#             score = float(score_text)
#         except ValueError:
#             continue
#         if score > 65 and sentence and("[" in sentence  or "{" in sentence) and "{CP" not in sentence and "CY" not in sentence and "CJ?" not in sentence and "CJX" not in sentence:
#             results.append({
#                 "data": sentence
#             })
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(results, f, ensure_ascii=False, indent=2)
# print(f"已保存到: {output_file}")
# print(f"共提取 {len(results)} 条检索原句")



# # NOTE 将数据还原为错句、正确句、错误类型列表的格式
# prefix = [
#     {"role": "user", "content": f"{GEC_data2}"},
# ]
# gen = gen_cov(model, client)
# context_id = gen.create_prefix_cache(prefix)

# all_data = get_all_info('/mnt/common/intern/qt/school_project/cgec_data/all_data/composition_score_gt_65.json')
# all_output = []

# with ThreadPoolExecutor(max_workers=50) as executor:
#     futures = []
#     for i, msg in enumerate(all_data):
#         # print(msg)
#         context_id = context_id 
#         futures.append(executor.submit(gen.pre_api, context_id, f"{msg}"))
#     for future in as_completed(futures):
#         try:
#             a_response = future.result()
#             data_json = process_output_one(a_response)
#             all_output.append(data_json)
#             print(f"API Response: {a_response}")
#         except Exception as e:
#             print(f"API Call Error: {e}")

# save_data('/mnt/common/intern/qt/school_project/cgec_data/all_data/check_data/couple_output.json',all_output)


# # NOTE 检查二分类的数据是否合格
# # 构造前缀缓存，存id
# prefix = [
#     {"role": "user", "content": f"{GEC_couple_check}"},
# ]
# gen = gen_cov(model, client)
# context_id = gen.create_prefix_cache(prefix)

# all_data = get_all_info('/mnt/common/intern/qt/school_project/cgec_data/all_data/ori_data/couple_output_0.json')
# all_output = []

# #并发调用
# with ThreadPoolExecutor(max_workers=50) as executor:
#     futures = []
#     for i, msg in enumerate(all_data):
#         # print(msg)
#         context_id = context_id 
#         futures.append(executor.submit(gen.pre_api, context_id, f"{msg}"))

#     for future in as_completed(futures):
#         try:
#             a_response = future.result()
#             data_json = process_output_one(a_response)
#             all_output.append(data_json)

#             print(f"API Response: {a_response}")

#         except Exception as e:
#             print(f"API Call Error: {e}")

# save_data('/mnt/common/intern/qt/school_project/cgec_data/all_data/check_data/couple_output_check.json',all_output)


#NOTE 对检查结果，取80分以上的数据作为couple数据.
# #对数据进行清洗、少的设置为其他错误。对原始和目标句子中有[、{的跳过，对不属于原始错误类型的跳过。
# INPUT_PATH = Path("/mnt/common/intern/qt/school_project/cgec_data/all_data/check_data/couple_output_check.json")
# OUTPUT_PATH = Path("/mnt/common/intern/qt/school_project/cgec_data/all_data/check_data/couple_data_cleaned.json")
# 1. score < 85 跳过
# 2. source / target 含 [ 或 { 跳过
# 3. error 校验：为空跳过；含非法类型整条跳过

# python /mnt/common/intern/qt/school_project/cgec_data/all_data/check_data/check.py