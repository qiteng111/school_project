import datetime
import os
from volcenginesdkarkruntime import Ark
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import machine_path,machine_prepare_a_exp,machine_a_exp_output,api_key,bot_cov_output,meta_output,data_relation_person_input,bot_cov_output,meta_path,data_relation_relation_output,data_relation_problem_input,data_relation_relation_input,model,person_fact_output,person_fact_input,client,name_filter_input_path,all_save_output2,info_prompt_path2,result1_input_path,result9_input_path,result_output_path,info_prompt_path,name_filter_output_path,rename_input_path,rename_output_path,timechange_input_path,person_input_path,problem_input_path,all_info_output_path,all_info_path,all_save_output
from model.roles import gen_cov,AsyncGenCov
from utils.cov_generator import data_prepare_machine,data_prepare_machine_exa,data_prepare_machine,data_prepare_meta,data_relation_allinfo,data_prepare_combine_relation,data_prepare_person_fact,get_all_info,data_prepare_combine,data_prepare_combine_bot_start,time_change,process_result,process_output_one,save_result,save_data,name_filter,generate_random_name,rename,data_prepare
from prompt.prompt import GEC_data_LLM,GEC_data3,GEC_couple_check,GEC_data2,GEC_explain_knowl_new,GEC_explain_knowl,machine_perosonal,bot_cov,cov_relation, cov_relation2,cov_relation_yyq, TTs_cov,TTs_cov_bot_start
import asyncio
import httpx
import json
import time

# # 篇章错误类型：
# # CP：错篇

# #跳过：D多字、Y异体字、CLH离合词、W外文词语、CJ+wy多余谓语、CJ-wy谓语残缺、CJbi比子句、CJl连字句、CJjy兼语句、CJld连动句、CJshb双宾语句、CJxw形容词谓语句、CJX语序错误、CJgd固定格式错误、
# # 别字/错字？

# # # NOTE 
# # 取全部数据，保留所有的语法错误数据，跳过错篇、无法识别标记...

# import csv
# import json
# import re

# input_file = "/home/s202507015/workspace/school_project/cgec_data/all_data/search_result.csv"
# output_file = "/home/s202507015/workspace/school_project/cgec_data/all_data/composition_all.json"

# results = []

# # 需要跳过的错误标记
# skip_error_patterns = [
    # r"\[D[^\]]*\]",        # D 多字，例如 [D在]
#     r"\[Y[^\]]*\]",        # Y 异体字，例如 [Y異]
#     r"\{CLH[^\}]*\}",      # CLH 离合词
#     r"\[W\d*[^\]]*\]",     # W 外文词，例如 [W2CD]

#     r"\{CJ\+wy[^\}]*\}",   # CJ+wy 多余谓语
#     r"\{CJ-wy[^\}]*\}",    # CJ-wy 残缺谓语
#     r"\{CJ\+by[^\}]*\}",   # CJ+by 多余宾语 

#     r"\{CJbi[^\}]*\}",     # CJbi 比字句
#     r"\{CJl[^\}]*\}",      # CJl 连字句
#     r"\{CJy[^\}]*\}",      # CJy 
#     r"\{CJjy[^\}]*\}",     # CJjy 兼语句
#     r"\{CJld[^\}]*\}",     # CJld 连动句
#     r"\{CJshb[^\}]*\}",    # CJshb 双宾语句
#     r"\{CJxw[^\}]*\}",     # CJxw 形容词谓语句
#     r"\{CJcx[^\}]*\}",     #  
#     r"\{CJ-xxy[^\}]*\}",     #  
#     r"\{y[^\}]*\}",     #  
#     r"\{CJ-zhy[^\}]*\}",     #  CJcd

#     r"\{CJcd[^\}]*\}",     # 
#     r"\{WWJ[^\}]*\}",     # 

#     r"\[y[^\]]*\]",        # Y 异体字
#     r"\[#\]",              # # 其他/特殊标记


#     r"\{CJX[^\}]*\}",      # CJX 语序错误
#     r"\{CJgd[^\}]*\}",     # CJgd 固定格式错误
# ]

# def should_skip(sentence):
#     """
#     判断句子是否包含需要跳过的错误类型。
#     只要命中任意一个跳过标记，就返回 True。
#     """
#     for pattern in skip_error_patterns:
#         if re.search(pattern, sentence):
#             return True
#     return False


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

#         # 基础过滤条件：
#         # 1. 句子非空
#         # 2. 句子中包含错误标记 [ ] 或 { }
#         # 3. 跳过 CP、CY、CJ?
#         # 4. 跳过指定错误类型
#         if (
#             sentence
#             and ("[" in sentence or "{" in sentence)
#             and "{CP" not in sentence
#             and "CY" not in sentence
#             and "CJ?" not in sentence
#             and "CJ？" not in sentence
#             and not should_skip(sentence)
#         ):
#             results.append({
#                 "data": sentence
#             })


# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(results, f, ensure_ascii=False, indent=2)

# print(f"已保存到: {output_file}")
# print(f"共提取 {len(results)} 条检索原句")



# #NOTE 在数据预处理阶段，需要平衡数据，保留所有语法错误的数据9000条 + 500条没有语法数据错误的数据
# import json
# import re
# from collections import Counter

# # =========================
# # 1. 定义错误标签
# # =========================

# char_tags = {
#     "C", "B", "L", "F", "P"
# }

# word_tags = {
#     "CC","CQ", "CD", "CY"
# }

# sentence_tags = {
#     "+zhuy", "+sy", "+buy", "+dy", "+zy", "+zxy",
#     "-zhuy", "-sy", "-by", "-buy", "-dy", "-zy", "-zxy",
#     "ba", "bei", "s", "sd", "cx",
#     "cd", "ZR", "WWJ", "?"
# }

# punct_tags = {
#     "BC", "BQ", "BD"
# }

# all_tags = char_tags | word_tags | sentence_tags | punct_tags

# # 长标签优先，避免 CC2 被识别成 CC
# sorted_tags = sorted(all_tags, key=len, reverse=True)


# # =========================
# # 2. 提取错误标签
# # =========================

# def extract_tag_from_inner(inner: str):
#     """
#     从 [] 或 {} 内部提取错误标签。

#     例：
#     [C]          -> C
#     [F發]        -> F
#     [BD，]       -> BD
#     {CC2全}      -> CC2
#     {CJ-buy好}   -> CJ-buy
#     {CJ+zhuy他}  -> CJ+zhuy
#     """

#     # 句错误类型可能是 CJ + 句错误标签
#     # 例如：CJ-buy、CJ+zhuy、CJ-zhuy、CJba
#     if inner.startswith("CJ"):
#         rest = inner[2:]

#         for tag in sorted(sentence_tags, key=len, reverse=True):
#             if rest.startswith(tag):
#                 return "CJ" + tag

#         # 容错处理
#         m = re.match(r"^(CJ[+\-]?[A-Za-z0-9?]+)", inner)
#         if m:
#             return m.group(1)

#     # 普通错误标签
#     for tag in sorted_tags:
#         if inner.startswith(tag):
#             return tag

#     return None


# def extract_error_tags(text: str):
#     """
#     提取一句话中的所有错误标签。
#     支持 [] 和 {} 两种格式。
#     """

#     tags = []

#     # 匹配 [C]、[F發]、[BD，] 等
#     bracket_items = re.findall(r"\[([^\[\]]+?)\]", text)

#     # 匹配 {CC2全}、{CD做}、{CJ-buy好} 等
#     brace_items = re.findall(r"\{([^{}]+?)\}", text)

#     for item in bracket_items + brace_items:
#         tag = extract_tag_from_inner(item)
#         if tag:
#             tags.append(tag)

#     return tags


# def is_sentence_error_tag(tag: str):
#     """
#     判断一个标签是否属于句错误类型。

#     包括：
#     1. 原始句错误标签，如 buy、-wy、ba、X
#     2. CJ + 句错误标签，如 CJ-buy、CJ+zhuy、CJba
#     """

#     if tag in sentence_tags:
#         return True

#     if tag.startswith("CJ"):
#         rest = tag[2:]
#         if rest in sentence_tags:
#             return True

#     return False


# # =========================
# # 3. 主程序
# # =========================

# def main():
#     input_file = "/home/s202507015/workspace/school_project/cgec_data/all_data/composition_all.json"
#     output_file = "/home/s202507015/workspace/school_project/cgec_data/all_data/composition_all_filtered.json"
#     stats_file = "/home/s202507015/workspace/school_project/cgec_data/all_data/error_stats.json"
    
#     # 没有句错误类型的数据最多保留 500 条
#     max_no_sentence_error_keep = 500
#     no_sentence_error_keep_count = 0
#     no_sentence_error_skip_count = 0

#     with open(input_file, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     filtered_data = []
#     error_counter = Counter()
#     sentence_error_counter = Counter()

#     for item in data:
#         text = item.get("data", "")
#         tags = extract_error_tags(text)

#         # 判断该句是否包含句错误类型
#         has_sentence_error = any(is_sentence_error_tag(tag) for tag in tags)

#         # =========================
#         # 新过滤逻辑
#         # =========================

#         if has_sentence_error:
#             # 有句错误类型：全部保留
#             filtered_data.append(item)

#         else:
#             # 没有句错误类型：最多保留 500 条
#             if no_sentence_error_keep_count < max_no_sentence_error_keep:
#                 filtered_data.append(item)
#                 no_sentence_error_keep_count += 1
#             else:
#                 no_sentence_error_skip_count += 1
#                 continue

#         # =========================
#         # 只统计最终保留下来的数据
#         # =========================

#         error_counter.update(tags)

#         for tag in tags:
#             if is_sentence_error_tag(tag):
#                 sentence_error_counter[tag] += 1

#     # 保存过滤后的数据
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(filtered_data, f, ensure_ascii=False, indent=2)

#     # 保存统计结果
#     stats = {
#         "原始数据数量": len(data),
#         "过滤后数据数量": len(filtered_data),
#         "保留的无句错误类型数据数量": no_sentence_error_keep_count,
#         "跳过的无句错误类型数据数量": no_sentence_error_skip_count,
#         "所有错误类型统计": dict(error_counter),
#         "句错误类型统计": dict(sentence_error_counter)
#     }

#     with open(stats_file, "w", encoding="utf-8") as f:
#         json.dump(stats, f, ensure_ascii=False, indent=2)

#     print("处理完成！")
#     print(f"原始数据数量：{len(data)}")
#     print(f"过滤后数据数量：{len(filtered_data)}")
#     print(f"保留的无句错误类型数据数量：{no_sentence_error_keep_count}")
#     print(f"跳过的无句错误类型数据数量：{no_sentence_error_skip_count}")
#     print(f"过滤后的数据已保存到：{output_file}")
#     print(f"错误类型统计已保存到：{stats_file}")

#     print("\n所有错误类型统计：")
#     for tag, count in error_counter.most_common():
#         print(f"{tag}: {count}")

#     print("\n句错误类型统计：")
#     for tag, count in sentence_error_counter.most_common():
#         print(f"{tag}: {count}")


# if __name__ == "__main__":
#     main()


# NOTE #all_filtered替换所有的[C]


# # NOTE 用规则的方法，替换B、L、D、F、P、CC、CC数字、CQ、CD、CJ+类、CJ-类、BC、BQ、BD
# # CJba、CJbei、CJs、CJsd、CJZR、WWJ 这类需要语义改写的句法错误,跳过
# import json
# import re


# input_file = "/home/s202507015/workspace/school_project/cgec_data/all_data/composition_all_filtered.json"
# output_file = "/home/s202507015/workspace/school_project/cgec_data/all_data/process_data/processed_data.json"
# unprocessed_file = "/home/s202507015/workspace/school_project/cgec_data/all_data/process_data/unprocessed_data.json"

# # 可规则还原的句法错误：多余类
# CJ_PLUS_ERROR_MAP = {
#     "zhuy": "多余主语",
#     "wy": "多余谓语",
#     "sy": "多余述语",
#     "by": "多余宾语",
#     "buy": "多余补语",
#     "dy": "多余定语",
#     "zy": "多余状语",
#     "zxy": "多余中心语",
# }

# # 可规则还原的句法错误：残缺类
# CJ_MINUS_ERROR_MAP = {
#     "zhuy": "残缺主语",
#     "wy": "残缺谓语",
#     "sy": "残缺述语",
#     "by": "残缺宾语",
#     "buy": "残缺补语",
#     "dy": "残缺定语",
#     "zy": "残缺状语",
#     "zxy": "残缺中心语",
# }

# # 非可规则替换，但可以识别错误类型的花括号句法标记
# # 这些类型通常需要语义改写，所以先保留标记，存入 c.json
# NON_RULE_ERROR_MAP = {
#     "CJba": "把字句",
#     "CJbei": "被字句",
#     "CJs": "是字句",
#     "CJsd": "“是……的”句",
#     "CJZR": "句式杂糅错误",
#     "WWJ": "未完句",
# }


# def replace_previous_chars(text, n, replacement):
#     """
#     用 replacement 替换 text 末尾 n 个字符。
#     用于 B、F、P、CC、CC数字、BC 等替换类错误，避免重复拼接。
#     """
#     if n <= 0:
#         return text + replacement

#     if len(text) < n:
#         return replacement

#     return text[:-n] + replacement

# def has_square_tag(text):
#     """
#     判断文本中是否仍然残留中括号符号。
#     只要出现 [ 或 ] 任意一个，就认为存在未处理的中括号标记。
#     """
#     return "[" in text or "]" in text


# def has_curly_tag(text):
#     """
#     判断文本中是否仍然残留花括号符号。
#     只要出现 { 或 } 任意一个，就认为存在未处理的花括号标记。
#     """
#     return "{" in text or "}" in text


# def process_sentence(sentence):
#     """
#     根据标记规则生成 source、target、error。
#     对可规则替换标记进行处理。
#     对非可规则替换花括号标记保留原样，后续进入 c.json。
#     对未知中括号标记保留原样，后续整条丢弃。
#     """
#     source = ""
#     target = ""
#     errors = []

#     i = 0

#     while i < len(sentence):
#         ch = sentence[i]

#         # =========================
#         # 处理中括号标记：[...]
#         # =========================
#         if ch == "[":
#             end = sentence.find("]", i)

#             if end == -1:
#                 # 不完整标记，原样保留，后续会因为仍含 [ 而丢弃
#                 source += ch
#                 target += ch
#                 i += 1
#                 continue

#             tag_content = sentence[i + 1:end]

#             # BC：错误标点
#             # 形式：正确标点[BC错误标点]
#             if tag_content.startswith("BC") and len(tag_content) > 2:
#                 wrong_punct = tag_content[2:]
#                 source = replace_previous_chars(source, 1, wrong_punct)
#                 # target 保留前面的正确标点
#                 errors.append("错误标点")
#                 i = end + 1
#                 continue

#             # BQ：空缺标点
#             # 形式：[BQ标点]
#             elif tag_content.startswith("BQ") and len(tag_content) > 2:
#                 missing_punct = tag_content[2:]
#                 # source 不加入该标点
#                 target += missing_punct
#                 errors.append("空缺标点")
#                 i = end + 1
#                 continue

#             # BD：多余标点
#             # 形式：[BD标点]
#             elif tag_content.startswith("BD") and len(tag_content) > 2:
#                 extra_punct = tag_content[2:]
#                 source += extra_punct
#                 # target 删除该标点
#                 errors.append("多余标点")
#                 i = end + 1
#                 continue

#             # B：错字
#             # 形式：正确字[B错误字]
#             elif tag_content.startswith("B") and len(tag_content) > 1:
#                 wrong_char = tag_content[1:]
#                 source = replace_previous_chars(source, 1, wrong_char)
#                 # target 保留前面的正确字
#                 errors.append("错字")
#                 i = end + 1
#                 continue

#             # L：漏字
#             # 形式：正确字[L]
#             elif tag_content == "L":
#                 if source:
#                     source = source[:-1]
#                 # target 保留前面的正确字
#                 errors.append("漏字")
#                 i = end + 1
#                 continue

#             # D：多字
#             # 形式：[D多余字]
#             elif tag_content.startswith("D") and len(tag_content) > 1:
#                 extra_char = tag_content[1:]
#                 source += extra_char
#                 # target 删除多余字
#                 errors.append("多字")
#                 i = end + 1
#                 continue

#             # F：繁体字
#             # 形式：简体字[F繁体字]
#             elif tag_content.startswith("F") and len(tag_content) > 1:
#                 traditional = tag_content[1:]
#                 source = replace_previous_chars(source, 1, traditional)
#                 # target 保留前面的简体字
#                 errors.append("繁体字")
#                 i = end + 1
#                 continue

#             # P：拼音字
#             # 形式：汉字[P拼音]
#             elif tag_content.startswith("P") and len(tag_content) > 1:
#                 pinyin = tag_content[1:]
#                 source = replace_previous_chars(source, 1, pinyin)
#                 # target 保留前面的汉字
#                 errors.append("拼音字")
#                 i = end + 1
#                 continue

#             else:
#                 # 未知中括号标记：保留原样，后续整条丢弃
#                 raw_tag = sentence[i:end + 1]
#                 source += raw_tag
#                 target += raw_tag
#                 i = end + 1
#                 continue

#         # =========================
#         # 处理花括号标记：{...}
#         # =========================
#         elif ch == "{":
#             end = sentence.find("}", i)

#             if end == -1:
#                 source += ch
#                 target += ch
#                 i += 1
#                 continue

#             tag_content = sentence[i + 1:end]

#             # CC 或 CC数字：错词
#             # 形式：正确词{CC错误词}
#             # 形式：正确词{CC数字错误词}
#             cc_match = re.fullmatch(r"CC(\d*)(.*)", tag_content)
#             if cc_match:
#                 num_text = cc_match.group(1)
#                 wrong_word = cc_match.group(2)

#                 if num_text:
#                     replace_len = int(num_text)
#                 else:
#                     replace_len = len(wrong_word)

#                 source = replace_previous_chars(source, replace_len, wrong_word)
#                 # target 保留前面的正确词
#                 errors.append("错词")
#                 i = end + 1
#                 continue

#             # CQ：缺词
#             # 形式：{CQ缺少词}
#             elif tag_content.startswith("CQ") and len(tag_content) > 2:
#                 missing_word = tag_content[2:]
#                 # source 不加入缺少词
#                 target += missing_word
#                 errors.append("缺词")
#                 i = end + 1
#                 continue

#             # CD：多词
#             # 形式：{CD多余词}
#             elif tag_content.startswith("CD") and len(tag_content) > 2:
#                 extra_word = tag_content[2:]
#                 source += extra_word
#                 # target 删除多余词
#                 errors.append("多词")
#                 i = end + 1
#                 continue

#             # CJ+xxx：多余句法成分
#             plus_match = re.fullmatch(r"CJ\+([a-z]+)(.*)", tag_content)
#             if plus_match:
#                 tag = plus_match.group(1)
#                 extra_part = plus_match.group(2)

#                 if tag in CJ_PLUS_ERROR_MAP:
#                     source += extra_part
#                     # target 删除多余成分
#                     errors.append(CJ_PLUS_ERROR_MAP[tag])
#                     i = end + 1
#                     continue

#             # CJ-xxx：残缺句法成分
#             minus_match = re.fullmatch(r"CJ-([a-z]+)(.*)", tag_content)
#             if minus_match:
#                 tag = minus_match.group(1)
#                 missing_part = minus_match.group(2)

#                 if tag in CJ_MINUS_ERROR_MAP:
#                     # source 删除缺少成分
#                     target += missing_part
#                     errors.append(CJ_MINUS_ERROR_MAP[tag])
#                     i = end + 1
#                     continue

#             # 非可规则替换花括号句法标记
#             # 例如：{CJba}、{CJbei}、{CJs}、{CJsd}、{CJZR}、{WWJ}
#             if tag_content in NON_RULE_ERROR_MAP:
#                 raw_tag = sentence[i:end + 1]
#                 source += ""
#                 target += raw_tag
#                 errors.append(NON_RULE_ERROR_MAP[tag_content])
#                 i = end + 1
#                 continue

#             # 其他未知花括号标记：保留原样，后续进入 c.json
#             raw_tag = sentence[i:end + 1]
#             source += ""
#             target += raw_tag
#             i = end + 1
#             continue

#         # =========================
#         # 普通字符
#         # =========================
#         else:
#             source += ch
#             target += ch
#             i += 1

#     return {
#         "source": source,
#         "target": target,
#         "error": "、".join(errors)
#     }


# def main():
#     with open(input_file, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     clean_results = []
#     unprocessed_results = []

#     no_error_count = 0
#     dropped_square_tag_count = 0

#     for item in data:
#         sentence = item.get("data", "")

#         processed = process_sentence(sentence)

#         result_item = {
#             "data": sentence,
#             "source": processed["source"],
#             "target": processed["target"],
#             "error": processed["error"]
#         }

#         # 没有任何可识别错误，跳过
#         if not processed["error"]:
#             no_error_count += 1
#             continue

#         # 如果处理完后仍存在 [...]，整条数据丢掉
#         if has_square_tag(processed["source"]) or has_square_tag(processed["target"]):
#             dropped_square_tag_count += 1
#             continue

#         # 如果处理完后仍存在 {...}，说明有非可替换花括号标记，存入 c.json
#         if has_curly_tag(processed["source"]) or has_curly_tag(processed["target"]):
#             unprocessed_results.append(result_item)
#         else:
#             clean_results.append(result_item)

#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(clean_results, f, ensure_ascii=False, indent=2)

#     with open(unprocessed_file, "w", encoding="utf-8") as f:
#         json.dump(unprocessed_results, f, ensure_ascii=False, indent=2)

#     print(f"已保存可完全规则还原数据到: {output_file}")
#     print(f"已保存含非可替换花括号标记数据到: {unprocessed_file}")
#     print(f"可完全规则还原数据: {len(clean_results)} 条")
#     print(f"含非可替换花括号标记数据: {len(unprocessed_results)} 条")
#     print(f"丢弃仍含 [...] 的数据: {dropped_square_tag_count} 条")
#     print(f"跳过无错误数据: {no_error_count} 条")


# if __name__ == "__main__":
#     main()




# # NOTE 将没办法根据规则还原的数据，送入模型进行处理，生成 source、target
# prefix = GEC_data_LLM

# gen = gen_cov(model, client)
# context_id = gen.create_prefix_cache(prefix)

# data = get_all_info('/home/s202507015/workspace/school_project/cgec_data/all_data/process_data/unprocessed_data.json')
# all_output = []
# all_data = [item["target"] for item in data]

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

# save_data('/home/s202507015/workspace/school_project/cgec_data/all_data/process_data/unprocessed_data_process.json',all_output)




# # # NOTE 将大模型纠正的数据，和原始数据合并，还原为 source、target、error 的格式
# # 读取 unprocessed_data.json
# # 读取大模型输出文件 unprocessed_data_process.json
# # 用 unprocessed_data_process.json 中每条数据的 source 去匹配 unprocessed_data.json 中的原始 target
# # 如果匹配成功：
# # 新的 source = 原始 target
# # 新的 target = 大模型输出的 target
# # 新的 error = 原始 error
# # 如果匹配不到，跳过
# # 如果原始 target 中包含字母 p 或 P，跳过
# # 保存到 unprocessed_data_final.json
# import json
# import os


# unprocessed_file = "/home/s202507015/workspace/school_project/cgec_data/all_data/process_data/unprocessed_data.json"
# llm_output_file = "/home/s202507015/workspace/school_project/cgec_data/all_data/process_data/unprocessed_data_process.json"
# output_file = "/home/s202507015/workspace/school_project/cgec_data/all_data/process_data/unprocessed_data_final.json"


# def load_json(file_path):
#     """读取 JSON 文件"""
#     with open(file_path, "r", encoding="utf-8") as f:
#         return json.load(f)


# def save_json(data, file_path):
#     """保存 JSON 文件"""
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)

#     with open(file_path, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)


# def normalize_text(text):
#     """
#     用于匹配时的轻度规范化。
#     这里只去掉首尾空白，不改变正文内容。
#     """
#     if text is None:
#         return ""
#     return str(text).strip()


# def contains_p(text):
#     """
#     判断文本中是否包含字母 p 或 P。
#     原始 target 中有 p / P 的数据需要跳过。
#     """
#     return "p" in text or "P" in text


# def main():
#     unprocessed_data = load_json(unprocessed_file)
#     llm_data = load_json(llm_output_file)

#     # 建立映射：
#     # key = 原始 target
#     # value = 原始数据
#     original_target_map = {}

#     duplicate_count = 0
#     skipped_p_count = 0

#     for item in unprocessed_data:
#         original_target = normalize_text(item.get("target", ""))

#         if not original_target:
#             continue

#         # 跳过原始 target 中有 p / P 的数据
#         if contains_p(original_target):
#             skipped_p_count += 1
#             continue

#         # 如果 target 重复，保留第一条，后面的计入重复
#         if original_target in original_target_map:
#             duplicate_count += 1
#             continue

#         original_target_map[original_target] = item

#     results = []

#     matched_count = 0
#     unmatched_count = 0
#     empty_llm_source_count = 0

#     for item in llm_data:
#         llm_source = normalize_text(item.get("source", ""))
#         llm_target = normalize_text(item.get("target", ""))

#         if not llm_source:
#             empty_llm_source_count += 1
#             continue

#         # 用大模型输出的 source 匹配原始 target
#         original_item = original_target_map.get(llm_source)

#         if original_item is None:
#             unmatched_count += 1
#             continue

#         results.append({
#             "source": original_item["source"],
#             "target": llm_target,
#             "error": original_item.get("error", "")
#         })

#         matched_count += 1

#     save_json(results, output_file)

#     print(f"已保存到: {output_file}")
#     print(f"成功匹配并保存: {matched_count} 条")
#     print(f"无法匹配跳过: {unmatched_count} 条")
#     print(f"原始 target 含 p/P 跳过: {skipped_p_count} 条")
#     print(f"原始 target 重复跳过: {duplicate_count} 条")
#     print(f"大模型 source 为空跳过: {empty_llm_source_count} 条")
#     print(f"最终 unprocessed_data_final.json 数据量: {len(results)} 条")


# if __name__ == "__main__":
#     main()

# # # NOTE 将两个文件processed_data.json和processed_data_final.json合并，作为最终的couple数据，保存到/home/s202507015/workspace/school_project/cgec_data/all_data/check_data/couple_data_cleaned.json
# import json
# import os


# file1 = "/home/s202507015/workspace/school_project/cgec_data/all_data/process_data/processed_data.json"
# file2 = "/home/s202507015/workspace/school_project/cgec_data/all_data/process_data/unprocessed_data_final.json"

# output_file = "/home/s202507015/workspace/school_project/cgec_data/all_data/check_data/couple_data_cleaned.json"


# def load_json(file_path):
#     with open(file_path, "r", encoding="utf-8") as f:
#         return json.load(f)


# def save_json(data, file_path):
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)

#     with open(file_path, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)


# def extract_couple_data(data):
#     results = []

#     for item in data:
#         source = item.get("source", "").strip()
#         target = item.get("target", "").strip()
#         error = item.get("error", "").strip()

#         # 如果 source 或 target 为空，跳过
#         if not source or not target:
#             continue

#         results.append({
#             "source": source,
#             "target": target,
#             "error": error
#         })

#     return results


# def main():
#     data1 = load_json(file1)
#     data2 = load_json(file2)

#     couple_data_1 = extract_couple_data(data1)
#     couple_data_2 = extract_couple_data(data2)

#     merged_data = couple_data_1 + couple_data_2

#     save_json(merged_data, output_file)

#     print(f"已保存到: {output_file}")
#     print(f"processed_data.json 提取数量: {len(couple_data_1)}")
#     print(f"unprocessed_data_final.json 提取数量: {len(couple_data_2)}")
#     print(f"最终合并数量: {len(merged_data)}")


# if __name__ == "__main__":
#     main()




# # NOTE 检查二分类的数据是否合格  （后面的不需要了，上述已经生成了couple数据）
# # 构造前缀缓存，存id
# prefix = [
#     {"role": "user", "content": f"{GEC_couple_check}"},
# ]
# gen = gen_cov(model, client)
# context_id = gen.create_prefix_cache(prefix)

# all_data = get_all_info('/home/s202507015/workspace/school_project/cgec_data/all_data/ori_data/couple_output_0.json')
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

# save_data('/home/s202507015/workspace/school_project/cgec_data/all_data/check_data/couple_output_check.json',all_output)


#NOTE 对检查结果，取80分以上的数据作为couple数据.
# #对数据进行清洗、少的设置为其他错误。对原始和目标句子中有[、{的跳过，对不属于原始错误类型的跳过。
# INPUT_PATH = Path("/home/s202507015/workspace/school_project/cgec_data/all_data/check_data/couple_output_check.json")
# OUTPUT_PATH = Path("/home/s202507015/workspace/school_project/cgec_data/all_data/check_data/couple_data_cleaned.json")
# 1. score < 85 跳过
# 2. source / target 含 [ 或 { 跳过
# 3. error 校验：为空跳过；含非法类型整条跳过

# python /home/s202507015/workspace/school_project/cgec_data/all_data/check_data/check.py