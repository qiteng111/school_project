from volcenginesdkarkruntime import Ark

api_key = 'c566053a-bfbb-4bd6-aa81-762b6ceb2936'
model = "ep-20250708125259-q5285"


# model = "ep-20251127115134-nvst4" #qianzhui



client = Ark(api_key=api_key)

name_filter_input_path = r'/home/s202507015/workspace/8w_project/data/55_ori.json'
name_filter_output_path = r'/home/s202507015/workspace/8w_project/data/55_filterted.json'

rename_input_path = name_filter_output_path
rename_output_path = r'/home/s202507015/workspace/8w_project/data/55_finall.json'

timechange_input_path = r"/home/s202507015/workspace/8w_project/data/55_finall.json"

problem_input_path = r'/home/s202507015/workspace/8w_project/data/cov_filtered_20k.json'
person_input_path = r'/home/s202507015/workspace/8w_project/data/55_finall_timechange.json'

info_prompt_path = r"/home/s202507015/workspace/8w_project/data_output/info_prompt.jsonl"
info_prompt_path2 = r"/home/s202507015/workspace/8w_project/data_output/info_prompt_bot_start.jsonl"

all_info_output_path = r'/share/project/xzfanag/qiteng_data/8w_project/data/all_info.json'

all_save_output = r"/home/s202507015/workspace/8w_project/data_output/relationn_output.json"
all_save_output2 = r"/home/s202507015/workspace/8w_project/data_output/output2.json"

all_info_path = r"/home/s202507015/workspace/8w_project/data/all_info_person_relation.json"

result9_input_path = r"/home/s202507015/workspace/8w_project/data_output/results (14).jsonl"
result1_input_path = r'/home/s202507015/workspace/8w_project/data_output/results (15).jsonl'

result_output_path = r'/home/s202507015/workspace/8w_project/data_output/500.jsonl'

person_fact_input = r'/home/s202507015/workspace/8w_project/data/1000_finall_timechange.json'
person_fact_output = r"/home/s202507015/workspace/8w_project/data_output/person_fact_prompt.jsonl"

data_relation_person_input = r'/home/s202507015/workspace/8w_project/data/person_fact_2000.json'
data_relation_problem_input = r'/home/s202507015/workspace/8w_project/data/cov_filtered_relation.json'
data_relation_relation_input = r'/home/s202507015/workspace/8w_project/data/relation_date.json'
data_relation_relation_output = r'/home/s202507015/workspace/8w_project/data_output/person_relation_prompt_5000.jsonl'

meta_path = "/home/s202507015/workspace/8w_project/data/bot_meta_1000.json"
meta_output = "/home/s202507015/workspace/8w_project/data/cov_meta_20000.json"
bot_cov_output = "/home/s202507015/workspace/8w_project/data_output/bot_cov_output_20000.json"

machine_prepare_a_exp = "/home/s202507015/workspace/8w_project/machine_data/input/machine_a_exp.json"
machine_prepare = "/home/s202507015/workspace/8w_project/data/machine_exp.json"

machine_path = '/home/s202507015/workspace/8w_project/machine_data/input/machine_dianqi.json'
machine_a_exp_output = '/home/s202507015/workspace/8w_project/machine_data/output/machinea_a_exp_output.json'

machine_100_prepare_path = '/home/s202507015/workspace/8w_project/machine_data/output/machine_100_prepare.json'

machine_100_output_path = '/home/s202507015/workspace/8w_project/machine_data/output/machine_100_output.json'
machine_100_tts_output_path = '/home/s202507015/workspace/8w_project/data_output/machine_100_tts_output.json'

docx_input_path = r'/home/s202507015/workspace/8w_project/docx_pdf/尹书记文稿汇编（截至2023年12月31日）[NB].docx'
docx_output_path = r'/home/s202507015/workspace/8w_project/docx_pdf/doc_output/doc_wx_output.jsonl'
docx_dir_input_path = r'/home/s202507015/workspace/8w_project/docx_pdf/1106-wx发送'

docx_api_output_path = r'/home/s202507015/workspace/8w_project/docx_pdf/doc_output/doc_api_output.jsonl'
