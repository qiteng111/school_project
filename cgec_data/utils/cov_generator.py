import os
import sys 
import json
import random
import re
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

#加载数据
def get_all_info(path):
    with open(path,'r',encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_all_info_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))   
    return data



#处理输出的数据，保存到json文件
def process_output_one(dd):
    if dd.startswith("```json"):
         dd = dd.replace("```json",'').strip()
    if dd.endswith("```"):
        dd = dd[:-3].strip()  

    try:
        json_obj = json.loads(dd)
        return json_obj
    except json.JSONDecodeError:
        return None

#保存生成的数据
def save_data(path,all_data):
    with open(path,'w',encoding='utf-8') as f:
        json.dump(all_data,f,ensure_ascii=False,indent=2)

def save_data_jsonl(path, all_data):
    with open(path, 'w', encoding='utf-8') as f:
        for data in all_data:
            json.dump(data, f, ensure_ascii=False, indent=2)  # 写入每条数据
            f.write('\n')  # 每个 JSON 对象后加换行符


def save_result(path,data):
    with open(path, "a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

# 过滤summary重复名字
def name_filter(path,save_path):
    with open(path,'r',encoding='utf-8') as f:
        person_data = json.load(f)
    for item in person_data:
        name = item["person"]
        if name in item["person_summary"]:
            item["person_summary"] = item["person_summary"].replace(name,'用户')
            print(f'已替换summary中的{name}')
        if name in item["person_profile"]:
            item["person_profile"] = item["person_profile"].replace(name,'用户')
            print(f'已替换profile中的{name}')

    with open(save_path,'w',encoding='utf-8') as f2:
         json.dump(person_data,f2,ensure_ascii=False,indent=2)

    print("summary和profile中人名已替换为：用户")

def generate_random_name(existing_names):
    surnames = ["李", "王", "张", "刘", "陈", "杨", "赵", "黄", "周", "吴","宋", "谢", "唐", "许", "邓", "曹", "彭", "曾", "肖", "田","周", "郑", "冯", "褚", "卫", "蒋", "沈", "韩", "朱", "秦"]
    given_names = ["梦遥", "安琪", "承安", "思涵", "明辉", "晨曦", "伟", "芳", "娜", "敏", "静", "强", "磊", "军", "洋", "勇", "杰", "娟", "涛", "明", "艳", "超","华", "玉欣", "龙", "琳", "慧", "志", "丹", "鹏", "蕾", "刚", "雪", "莹", "波", "丽佳", "旭", "晨", "梅", "佳", "璐", "豪", "昊", "琪", "锐", "颖", "琼", "航", "楠", "瑞", "凡"]

    while True:
        name = random.choice(surnames) + random.choice(given_names)
        if name not in existing_names:
            return name

# 替换所有重复人名，前100设置为none
def rename(path,save_path):

    with open(path,'r',encoding='utf-8') as f:
        person_data = json.load(f)

    name_count = {}
    for item in person_data:
        name = item["person"]
        name_count[name] = name_count.get(name, 0) + 1

    used_names = set([item["person"] for item in person_data])

    seen_names = {}
    for item in person_data:
        name = item["person"]
        if name not in seen_names:
            seen_names[name] = 1
        else:
            new_name = generate_random_name(used_names)
            print(f"名字 {name} 重复，改为新名字：{new_name}")
            item["person"] = new_name
            used_names.add(new_name)

    for data in person_data[:6]:
        data['person'] = 'none'

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(person_data, f, ensure_ascii=False, indent=2)

    print("人名去重完成并保存完毕,人数：",len(person_data))


#构造prompt的输入.
# ---随机取2个问题、随机取m个人。构造nm个原始输入
def data_prepare(person_path, problem_path, save_path, num_groups,m):
    with open(person_path, 'r', encoding='utf-8') as f:
        person_data = json.load(f)

    with open(problem_path, 'r', encoding='utf-8') as f:
        problem_data = json.load(f)

    all_info = []  # 保存所有组数据

    for _ in range(num_groups):
        qa1 = random.choice(random.choice(problem_data))
        qa2 = random.choice(random.choice(problem_data))
        qa3 = random.choice(random.choice(problem_data))

        ten_people = random.sample(person_data, m) #返回列表

        for person in ten_people:
            conversation = [{
                "person": person["person"],
                "person_summary": person["person_summary"],
                "person_profile": person["person_profile"]
            },
            qa1,  # 第一轮问答
            qa2]  # 第二轮问答

            all_info.append(conversation)

    # 保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_info, f, ensure_ascii=False, indent=2)

    print(f"已成功生成 {num_groups * m} 条最终信息，保存到 {save_path}")

#构造bot_meta的prompt输入，每个meta随机取10个不同的问题
def data_prepare_meta(meta_path, problem_path, save_path):
    same_num = 21
    with open(meta_path, 'r', encoding='utf-8') as f:
        metas = json.load(f)

    with open(problem_path, 'r', encoding='utf-8') as f:
        problem_data = json.load(f)

    print(len(metas))
    all_info = []  # 保存所有组数据

    for meta in metas:
        for _ in range(same_num):
            qa1 = random.choice(random.choice(problem_data))
            qa2 = random.choice(random.choice(problem_data))
            qa3 = random.choice(random.choice(problem_data))
            qa4 = random.choice(random.choice(problem_data))
            qa5 = random.choice(random.choice(problem_data))
            conversation = [{
                "meta": meta,
            },
            qa1,
            qa2,
            qa3,
            qa4,
            qa5
            ]

            all_info.append(conversation)

    # 保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_info, f, ensure_ascii=False, indent=2)

    print(f"已成功生成 {same_num*len(metas)} 条最终信息，保存到 {save_path}")

#时间戳2023-05-12-14-30改成“2023年5月12日14时30分”
def time_change(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def convert_time_format(match):
        year, month, day, hour, minute = match.groups()
        return f"{year}年{int(month)}月{int(day)}日{int(hour)}时{int(minute)}分"

    for entry in data:
        entry['person_summary'] = re.sub(
            r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})",
            convert_time_format,
            entry['person_summary']
        )
        # 修改 person_profile
        entry['person_profile'] = re.sub(
            r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})",
            convert_time_format,
            entry['person_profile']
        )
    with open(r'/share/project/xzfang/qiteng_data/8w_project/data/55_finall_timechange.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("时间格式已修改并保存到 55_finall_timechange.json")


#构造批量推理的数据
def data_prepare_combine(person_path, problem_path, save_path, num_groups,m):
    with open(person_path, 'r', encoding='utf-8') as f:
        person_data = json.load(f)
    with open(problem_path, 'r', encoding='utf-8') as f:
        problem_data = json.load(f)

    all_info = []  # 保存所有人物和问答的数据

    for _ in range(num_groups):
        qa1 = random.choice(random.choice(problem_data))
        qa2 = random.choice(random.choice(problem_data))
        #qa3 = random.choice(random.choice(problem_data))

        ten_people = random.sample(person_data, m) #返回列表

        for person in ten_people:
            conversation = [{
                "person": person["person"],
                "person_summary": person["person_summary"],
                "person_profile": person["person_profile"]
            },
            qa1,  # 第一轮问答
            qa2]  # 第二轮问答
            all_info.append(conversation)

            
    with open(save_path, 'w', encoding='utf-8') as save_f:
        request_id = 1
        for entry in all_info:        
            prompt_text = f'''你是一个对话数据增强和 TTS(Text-To-Speech) 优化专家，擅长根据用户画像生成个性化多轮对话，并构造适合 TTS 的文本 SFT 数据集。

当前输入数据包含：  
1. **用户信息**：包括用户姓名（person）、用户与 AI 多次历史对话总结（person_summary）、详细用户画像（person_profile，含多维度性格特质及推理说明）。  
2. **多轮对话**：（一个用户提问 `user` 和 AI 回复 `bot` 的列表）。

你的任务：  

### **TTS 适配要求**
1. 回答文本不应太长，因为问题和回答最终都将以语音形式呈现，过长不适合交互。  
2. 除了中英文、数字以及必要的基础标点外，问题和回答应不包含其他字符（如特殊符号 *、-、** 等）和分段标题，避免 TTS 出错。  
3. 润色问题和回答，使其更加自然、口语化、适合语音交流。  
4. 对时间信息（如 18:22）、温度（如 ℃）、单位符号等进行文字化处理，使其适合 TTS，比如“18:22”转换为“下午六点二十二分”。  
5. 如果回答涉及实时信息，需说明自己是语音对话机器人，当前无法访问实时数据或调用外部工具，因此拒绝回答此类问题。  

### **个性化多轮对话生成要求**
1. **修改多轮对话中 bot 的回答**：  
   - 让回答风格体现出 AI 对用户的了解，结合用户的兴趣、性格、语言风格，让用户感到“AI 认识自己”。 
   - 适当在 bot 的回复中融入用户过往经历、兴趣和习惯（来自 person_summary 和 person_profile）。  
   - **特别注意：如果用户姓名（person）的值为 none，bot 回复仍符合上述风格，但不能提及用户姓名**。

2. **生成用户提问**：  
   - 生成的所有用户的提问应是全新的发散性问题/主题不局限于以往对话历史。
   - 问题应自由且富有创意。  
   - 问题必须完整且自洽，不可出现缺失上下文的情况（例如：“翻译下面这段话” 或 “帮我找出这段文字的关键词”，但没有给出具体文本，这类问题属于不完整，必须避免）
   - **补充对话的用户提问中加入**一些关于用户姓名的问题 和summary 的事实性问题（如“我在某时间做了什么？”、如“你记得我叫什么名字吗？”、“你知道我叫什么名字吗？”、“上次我们聊的是哪国文学？”等等），引导 AI 回忆用户姓名和过往互动。
   
3. **补充对话**：  
   - 如果某条多轮对话 **轮数 < 3**（不足 3 轮的 user-bot 配对），请补充完整至少 3 轮对话。
   - 新增的 user 提问要符合**生成用户提问**和TTS的要求。  
   - 新增的 bot 回复也要符合用户画像和 TTS 要求。*特别注意*：如果 user 的问题中询问了用户的名字信息，而 person 的值为 none，bot 事实上不知道用户名字，因此bot的回复应礼貌拒绝回答**（例如“我还不知道你的名字哦”）。

4. 每一轮对话必须严格遵循：**1 user 提问 → 1 bot 回复**，不要生成连续的 user 或 bot。  

5. **打乱对话顺序**：  
   - 在生成并修改所有的多轮对话后，将所有 user-bot 问答对（包括原始的和新增的）**整体随机打乱前后顺序**。  

### **输出格式**
输出完整修改后的数据，包括用户信息和对话。**必须严格使用 JSON 格式输出**，且不要在 JSON 前后添加任何解释或额外文字，以下面为例：
[
    {{
    "person": "",
    "person_summary": "",
    "person_profile": ""
    }},
    {{
    "user": "用户提问（口语化且适合 TTS）",
    "bot": "bot 回复，符合用户画像和 TTS 适配要求。"
    }},
    {{
    "user": "用户提问（口语化且适合 TTS）",
    "bot": "bot 回复，符合用户画像和 TTS 适配要求。"
    }},
    {{
    "user": "用户提问（口语化且适合 TTS）",
    "bot": "bot 回复，符合用户画像和 TTS 适配要求。"
    }}
    ...(如有更多对话继续追加)
]

输入数据：
{json.dumps(entry, ensure_ascii=False, indent=2)}'''
        
            request_obj = {
                "custom_id": f"request-{request_id}",
                "body": {
                    "messages": [
                        {"role": "user", "content": prompt_text}
                    ],
                    "max_tokens": 2000,
                    "top_p": 1
                }
            }

            # 写入 jsonl 文件
            save_f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")
            request_id += 1

        print(f"已生成预处理文件")

def data_prepare_combine_bot_start(person_path, problem_path, save_path, num_groups,m):
    with open(person_path, 'r', encoding='utf-8') as f:
        person_data = json.load(f)
    with open(problem_path, 'r', encoding='utf-8') as f:
        problem_data = json.load(f)

    all_info = []  # 保存所有人物和问答的数据

    for _ in range(num_groups):
        qa1 = random.choice(random.choice(problem_data))
        qa2 = random.choice(random.choice(problem_data))
        #qa3 = random.choice(random.choice(problem_data))

        ten_people = random.sample(person_data, m) #返回列表

        for person in ten_people:
            conversation = [{
                "person": person["person"],
                "person_summary": person["person_summary"],
                "person_profile": person["person_profile"]
            },
            qa1,  # 第一轮问答
            qa2]  # 第二轮问答
            all_info.append(conversation)

            
    with open(save_path, 'w', encoding='utf-8') as save_f:
        request_id = 1
        for entry in all_info:        
            prompt_text = f'''你是一个对话数据增强和 TTS(Text-To-Speech) 优化专家，擅长根据用户画像生成个性化多轮对话，并构造适合 TTS 的文本 SFT 数据集。

当前输入数据包含：  
1. **用户信息**：包括用户姓名（person）、用户与 AI 多次历史对话总结（person_summary）、详细用户画像（person_profile，含多维度性格特质及推理说明）。  
2. **多轮对话**：（一个用户提问 `user` 和 AI 回复 `bot` 的列表）。**添加的第一轮**中，用户尚未提问，user 为空，对话由 bot 开始。

你的任务：  

### **TTS 适配要求**
1. 回答文本不应太长，因为问题和回答最终都将以语音形式呈现，过长不适合交互。  
2. 除了中英文、数字以及必要的基础标点外，问题和回答应不包含其他字符（如特殊符号 *、-、** 等）和分段标题，避免 TTS 出错。  
3. 润色问题和回答，使其更加自然、口语化、适合语音交流。  
4. 对时间信息（如 18:22）、温度（如 ℃）、单位符号等进行文字化处理，使其适合 TTS，比如“18:22”转换为“下午六点二十二分”。  
5. 如果回答涉及实时信息，需说明自己是语音对话机器人，当前无法访问实时数据或调用外部工具，因此拒绝回答此类问题。  

### **个性化多轮对话生成要求**
1. **添加第一轮对话**:
   - bot 的回复需要向用户打招呼，并且问候内容要个性化：
   - 根据用户的画像、兴趣、性格等，设计一个温暖、亲切的开场白（注意多轮对话中，不要所有对话都以用户的名字为开始）。
   - 保证 bot 在开场白中体现出 AI 对用户的关心和了解。
   - **特别注意：如果用户姓名（person）的值为 none，bot 回复仍符合上述风格，但不能提及用户姓名**。

2. **修改多轮对话中 bot 的回答**：  
   - 让回答风格体现出 AI 对用户的了解，结合用户的兴趣、性格、语言风格，让用户感到“AI 认识自己”。 
   - 适当在 bot 的回复中融入用户过往经历、兴趣和习惯（来自 person_summary 和 person_profile）。

3. **生成用户提问**：  
   - 生成的所有用户的提问应是全新的发散性问题/主题不局限于以往对话历史。
   - 问题应自由且富有创意。  
   - 问题必须完整且自洽，不可出现缺失上下文的情况（例如：“翻译下面这段话” 或 “帮我找出这段文字的关键词”，但没有给出具体文本，这类问题属于不完整，必须避免）
   - **补充对话的用户提问中加入**一些关于用户姓名的问题 和summary 的事实性问题（如“我在某时间做了什么？”、如“你记得我叫什么名字吗？”、“你知道我叫什么名字吗？”、“上次我们聊的是哪国文学？”等等），引导 AI 回忆用户姓名和过往互动。
   
4. **补充对话**：  
   - 如果某条多轮对话 **轮数 < 3**（不足 3 轮的 user-bot 配对），请补充完整至少 3 轮对话。
   - 新增的 user 提问要符合**生成用户提问**和TTS的要求。  
   - 新增的 bot 回复也要符合用户画像和 TTS 要求。*特别注意*：如果 user 的问题中询问了用户的名字信息，而 person 的值为 none，bot 事实上不知道用户名字，因此bot的回复应礼貌拒绝回答**（例如“我还不知道你的名字哦”）。

5. 每一轮对话必须严格遵循：**1 user 提问 → 1 bot 回复**，不要生成连续的 user 或 bot。  

6. **打乱对话顺序**：  
   - 在生成并修改所有的多轮对话后，将除添加的第一轮user为空的对话外,其他所有 user-bot 问答对**整体打乱前后顺序**。  

### **输出格式**
输出完整修改后的数据，必须严格使用 JSON 格式，且不要在 JSON 前后添加任何解释或额外文字。格式如下：  
[
    {{
    "person": "",
    "person_summary": "",
    "person_profile": ""
    }},
    {{
    "user": "",
    "bot": "bot 回复，符合用户画像和 TTS 适配要求。"
    }},
    {{
    "user": "用户提问1（口语化且适合 TTS）",
    "bot": "bot 回复2，符合用户画像和 TTS 适配要求。"
    }},
    {{
    "user": "用户提问2（口语化且适合 TTS）",
    "bot": "bot 回复3，符合用户画像和 TTS 适配要求。"
    }}
    ...(如有更多对话继续追加)
]

输入数据：
{json.dumps(entry, ensure_ascii=False, indent=2)}'''
        
            request_obj = {
                "custom_id": f"request-{request_id}",
                "body": {
                    "messages": [
                        {"role": "user", "content": prompt_text}
                    ],
                    "max_tokens": 2000,
                    "top_p": 1
                }
            }

            # 写入 jsonl 文件
            save_f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")
            request_id += 1

        print(f"已生成预处理文件")



def data_prepare_person_fact(person_path,save_path):
    with open(person_path, 'r', encoding='utf-8') as f:
        person_data = json.load(f)
            
    with open(save_path, 'w', encoding='utf-8') as save_f:
        request_id = 1
        for entry in person_data:        
            prompt_text = f'''
你是一个精准建模用户画像的AI专家。这是某位用户的信息，包括用户姓名（person）、用户与 AI 多次历史对话总结（person_summary）、详细用户画像（person_profile，含多维度性格特质及推理说明）。
请基于这些信息，随意发散生成5条关于该用户的事实性描述（person_fact），类型应尽量多样化包括但不限于：基础身份信息、职业工作、兴趣爱好、人际社交、文化背景、生活方式、能力技能、心理价值观等

要求如下：
1、生成的person_fact字段是一个列表，共5条
2、每条描述必须以“用户”开头，不得直接提及用户姓名或使用其他代词
3、五条中至少有 1 条涉及基础身份信息
4、每条是自然语言表述的人物事实性介绍（可虚构，无需与提供信息保持一致，但需合理推断）；
5、语言风格简洁、自然，不使用“可能”“或许”等不确定表达；
6、不要重复已有person_summary或person_profile中的句子，要对人物事实性特征进行扩展；
7、允许创意推断，例如结合用户话题内容推断其状态、背景、生活细节等；
8、输出时，必须保留输入数据中的 person、person_summary、person_profile，并在其后新增 person_fact 字段

### **输出格式**
输出完整修改后的数据。**必须严格使用 JSON 格式输出**，且不要在 JSON 前后添加任何解释或额外文字，以下面为例：
{{
  "person": "（按原文粘贴）",
  "person_summary": "（按原文粘贴）",
  "person_profile": "（按原文粘贴）",
  "person_fact": [
  "随机生成的用户事实性描述1",
  "随机生成的用户事实性描述2",
  "随机生成的用户事实性描述3",
  "随机生成的用户事实性描述4",
  "随机生成的用户事实性描述5"
]
}}

输入数据：
{json.dumps(entry, ensure_ascii=False, indent=2)}'''
        
            request_obj = {
                "custom_id": f"request-{request_id}",
                "body": {
                    "messages": [
                        {"role": "user", "content": prompt_text}
                    ],
                    "max_tokens": 2000,
                    "top_p": 1
                }
            }

            # 写入 jsonl 文件
            save_f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")
            request_id += 1

        print(f"已生成预处理文件")

#批量推理的输出数据处理
def process_result(result9_input_path,result1_input_path,result_output_path):

    all_data = []
    with open(result9_input_path, encoding='utf-8') as f:
        for line in f:
            data_9 = json.loads(line)
            all_data.append(data_9)
    with open(result1_input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_1 = json.loads(line)
            all_data.append(data_1)

    random.shuffle(all_data)

    print(all_data[0]["response"]["body"]["choices"][0]["message"]["content"])

    data = []
    for i in range(len(all_data)):
        dd = all_data[i]["response"]["body"]["choices"][0]["message"]["content"]

        if dd.startswith("```json"):
            dd = dd.replace("```json",'').strip()
        if dd.endswith("```"):
            dd = dd[:-3].strip()  
            
        try:
            json_obj = json.loads(dd)
            data.append(json_obj)
        except json.JSONDecodeError:
            print(f"⚠ JSON解析失败: 第{i}条内容")
            print(dd)
    print(len(data))

    with open(result_output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def build_relation_graph(person_data, relation_data, zhu, min_n=1, max_n=4):
    zhu_id = zhu.get("person_fact")
    candidates = [
        p for p in person_data
        if p.get("person_fact") != zhu_id
    ]

    target = random.randint(min_n, max_n)

    picked_people = random.sample(candidates, k=target)


    relation_people = []
    guanxi_ed = []
    guanxi_ed.append(zhu.get("person_relation"))

    for people in picked_people:
        rel = random.choice(relation_data)
        relation_people.append({
            "person": people["person"],
            "person_relation": rel,
            "person_fact": people["person_fact"],
        })
        guanxi_ed.append(rel)

    while 1:
        guanxi = random.choice(relation_data)
        if guanxi not in guanxi_ed:
            break
    
    return relation_people,guanxi



def data_relation_allinfo(person_path,relation_path, problem_path, num):


    with open(person_path, 'r', encoding='utf-8') as f:
        person_data = json.load(f)
    with open(problem_path, 'r', encoding='utf-8') as f:
        problem_data = json.load(f)
    with open(relation_path, 'r', encoding='utf-8') as f:
        relation_data = json.load(f)

    all_info = []  # 保存所有人物和问答的数据

    for _ in range(num):
        qa1 = random.choice(random.choice(problem_data))
        qa2 = random.choice(random.choice(problem_data))

        zhu = random.choice(person_data)
        relation_people = build_relation_graph(person_data, relation_data, zhu, min_n=1, max_n=4)

        conversation = [{
            "主用户":{
                    "person": zhu["person"],
                    "person_summary": zhu["person_summary"],
                    "person_profile": zhu["person_profile"],
                    "person_fact":zhu["person_fact"] 
            },
            "主用户关系图":relation_people,
            "对话历史":[
                qa1,
                qa2]
        }]

        all_info.append(conversation)
    with open(r'/share/project/xzfang/qiteng_data/8w_project/data/all_info_person_relation.json','w',encoding='utf-8') as f:
        json.dump(all_info,f,ensure_ascii=False,indent=2)

def data_prepare_combine_relation(person_path,relation_path, problem_path, save_path, num):

    with open(person_path, 'r', encoding='utf-8') as f:
        person_data = json.load(f)
    with open(problem_path, 'r', encoding='utf-8') as f:
        problem_data = json.load(f)
    with open(relation_path, 'r', encoding='utf-8') as f:
        relation_data = json.load(f)

    all_info = []  # 保存所有人物和问答的数据

    for _ in range(num):
        qa1 = random.choice(random.choice(problem_data))
        qa2 = random.choice(random.choice(problem_data))

        zhu = random.choice(person_data)
        relation_people,guanxi= build_relation_graph(person_data, relation_data, zhu, min_n=1, max_n=4)
        conversation = [{
            "主用户":{
                    "person": zhu["person"],
                    "person_summary": zhu["person_summary"],
                    "person_profile": zhu["person_profile"],
                    "person_fact":zhu["person_fact"] 
            },
            "主用户关系图":relation_people,
            "对话历史":[
                qa1,
                qa2],
            "补充对话③：依据此关系补充无法回答的关系 + 事实问题":guanxi
        }]

        all_info.append(conversation)


    with open(save_path, 'w', encoding='utf-8') as save_f:
        request_id = 1
        for entry in all_info:        
            prompt_text = f'''
你是一个对话数据增强和 TTS(Text-To-Speech) 优化专家，擅长基于用户画像生成个性化多轮对话，并构造适合 TTS 的文本 SFT 数据集。


当前输入数据包含：  
1. **主用户信息**：
   person：用户姓名，未知时为 none
   person_summary：用户与 AI 的多次历史对话总结
   person_profile：详细用户画像，含多维度性格特质与推理说明
   person_fact：关于主用户的 5 条事实性描述
2. **主用户关系图**（1–4 个人）：
   person：关系人物姓名，未知时为 none
   person_relation：与主用户的关系
   person_fact：该人物的 5 条事实性描述
3. **对话历史**(2轮)：（一个用户提问 `user` 和 AI 回复 `bot` 的列表）
4. **补充对话③：依据此关系补充无法回答的关系 + 事实问题**：
    提供参考，在补充对话的③，依据此关系进行提问

你的任务： 
在保持原始结构的前提下：
   首先，**覆写并润色**历史对话中的 bot 回复，使其个性化且满足 TTS 要求；
   **生成新增多轮对话**,新增的对话涉及与主用户有社会关系的一个或多个用户，以及他们的相关事实（见下方**补充对话**要求）；
   仅输出一个 JSON 数组，键名与层级与示例一致，不添加多余字段或注释。

### **TTS 适配要求**
1. 回答文本不应太长，因为问题和回答最终都将以语音形式呈现，过长不适合交互。  
2. 除了中英文、数字以及必要的基础标点外，问题和回答应不包含其他字符（如特殊符号 *、-、** 等）和分段标题，避免 TTS 出错。  
3. 润色问题和回答，使其更加自然、口语化、适合语音交流。  
4. 对时间信息（如 18:22）、温度（如 ℃）、单位符号等进行文字化处理，使其适合 TTS，比如“18:22”转换为“下午六点二十二分”。  
5. 如果回答涉及实时信息，需说明自己是语音对话机器人，当前无法访问实时数据或调用外部工具，因此拒绝回答此类问题。  

### **字段统一与取值规范（重要）**
1、在“对话历史”的每一轮问答对象里，追加以下 3 个字段，键名必须完全一致：
   - 关系检索关键词：null 或 字符串数组
   - 事实检索关键词：null 或 字符串数组
   - support：null 或 对象数组
2、对已有的历史问答（即输入中已有的2个轮次），将上述 3 个字段统一设为 null。
3、对你新生成的问答：
   - 若问题涉及关系与事实检索，填入相应的关键词数组；
   - 若仅涉及事实检索，关系检索关键词置为 null；
   - support 为一个数组，其中每个对象包含 person、person_relation、person_fact（仅保留与你回答相关的 1–2 条事实），用于“可追溯”支撑。
注意：只有姓名字段可以使用字符串 "none"；其余缺省一律使用 JSON 的 null。

### **个性化多轮对话修改与生成要求**
1. **修改输入数据中2条多轮对话中 bot 的回答**：  
   - 让回答风格体现出 AI 对主用户的了解，结合用户的兴趣、性格、语言风格，让用户感到“AI 认识自己”。 
   - 适当在 bot 的回复中融入用户过往经历、兴趣和习惯（来自 person_summary 和 person_profile）。  
   - 关系检索关键词、事实检索关键词、support 字段设为null
   - **特别注意：如果用户姓名（person）的值为 none，bot 回复仍符合上述风格，但不能提及用户姓名**。

2. **补充对话**：  
   ①、补充1条关系 + 事实问题：
   - 在**主用户关系图**中选取1个用户，并选取其1-2个 person_fact；
   - 将选取的用户信息按照**字段统一与取值规范**，填入support字段；
   - 生成一个"user"问题，注意！该问题**必须**依赖于上述support中的person_fact才能正确回答。
   - 根据问题，生成“关系检索关键词”和“事实检索关键词”列表。这两类关键词是为了训练模型在阅读问题后生成正确检索关键词的能力，因此必须尽量保证(1) 通过“关系检索关键词”能够找到support中的用户与主用户的关系；（2）通过“事实检索关键词”能够找到回答该问题所需要的person_fact，但“事实检索关键词”必须仅来自问题中的信息，而不能涉及提前透露答案中的信息。
      - 关于生成问题、关系检索关键词、事实检索关键词的两个具体例子如下：
         - "user": 我那个Instagram很多粉丝的同事喜欢收藏哪些东西？"关系检索关键词": ["同事"], "事实检索关键词": ["Instagram", "粉丝", "收藏"], "bot": (回答)
         - "user": 李明今年多大了？"关系检索关键词": ["李明"], "事实检索关键词": ["年龄"], "bot": (回答)
         - 注意不要模仿例子中的问题，要发挥想象力想更多其他形式的问题。
   - 重要！！！“关系检索关键词”必须只从问题的题面出发，而不能出现问题中没有的信息。通常来说，“关系检索关键词”可能包括"person"(例如李明)或与主用户的关系"person_relation"字段（“同事”“夫妻”）。！！！只有当题面中出现了用户的名字或对应的关系时，才应该列为“关系检索关键词”。例如support中的用户为李明，
   关系为“父亲”，那么问“我父亲今年多大了”时"关系检索关键词"必须为["父亲"]而不应包含李明，而问“李明喜欢做什么”时"关系检索关键词"必须为["李明"]而不应包含父亲。
   - 重要！！！类似地，“事实检索关键词”相关的信息也必须是**在问题中出现过**的。重申一遍，生成该数据的目的是训练模型在阅读**问题**后立即生成检索关键词的能力，此时要训练的模型还没有看过bot的回复或任何person_fact文本。
   - 最后，生成合理的"bot"回复。
   
   ②、补充1条仅事实问题（问题中不包含关系）：
   - 在**主用户关系图**中选取1个用户，并选取其1-2个 person_fact；
   - 将选取的用户信息按照**字段统一与取值规范**，填入support字段；
   - 关系检索关键词：null；
   - 生成一个"user"问题，注意！该问题**必须**依赖于上述support中的person_fact才能正确回答。
   - 根据问题，生成“事实检索关键词”列表。这个关键词是为了训练模型在阅读问题后生成正确检索关键词的能力，因此必须尽量保证：通过“事实检索关键词”能够找到回答该问题所需要的person_fact，但“事实检索关键词”必须仅来自问题中的信息，而不能涉及提前透露答案中的信息。
   - 重要！！！类似地，“事实检索关键词”相关的信息也必须是**在问题中出现过**的。重申一遍，生成该数据的目的是训练模型在阅读**问题**后立即生成检索关键词的能力，此时要训练的模型还没有看过bot的回复或任何person_fact文本。
   - 事实检索关键词：根据问题列出所有事实检索关键词；
      - 例1:"user": "帮我回忆一下我认识的都市白领们的兴趣爱好", "关系检索关键词": null, "事实关键词": ["都市白领", "爱好"], "bot": "您认识的都市白领包括您的妻子和您的同事褚丹。您的妻子喜欢锻炼身体。您的同事褚丹喜欢摄影创作。"
      - 例2:"user": "我认识的人里有人爱去博物馆吗？", "关系检索关键词": null, "事实关键词": ["博物馆"], "bot": "您认识的人里，您的同事李敏和司机周平爱去博物馆。李敏是狂热粉丝，每周都去。周平只是偶尔去。"
      - 注意不要模仿例子中的问题，要发挥想象力想更多其他形式的问题。
   - 最后，生成合理的"bot"回复。
   
   ③、补充1条无法回答的关系 + 事实问题：
   - 根据输入数据的**补充对话③：依据此关系补充无法回答的关系 + 事实问题**字段，根据这个关系生成一个关系 + 事实类的问题
   - bot 需简短、礼貌地拒绝回答，并明确说明依据不足；
   - 关系检索关键词：问题中涉及到的关系词；
   - 事实检索关键词：与问题核心名词相关的 1–2 个词；
   - support字段填写为null "support": null

   在以上所有任务中：
   - 新增的 user 问题应：自由且富有创意；主题不局限于以往对话历史：问题必须完整且自洽，不可出现缺失上下文的情况：符合上述TTS适配的要求。  
   - 新增的 bot 回复也要符合用户画像和 TTS 要求。

3. 每一轮对话必须严格遵循：**1 user 提问 → 1 bot 回复**，不要生成连续的 user 或 bot。每一轮必须且只包含五个字段: "user", "关系检索关键词", "事实检索关键词", "support", "bot"。

   
### **输出格式**
输出完整修改后的数据，包括用户信息和对话（不需要输出**补充对话③：依据此关系补充无法回答的关系 + 事实问题**字段）。**必须严格使用 JSON 格式输出**，且不要在 JSON 前后添加任何解释或额外文字，以下面为例：

[
    {{
        "主用户":
            {{
            "person": "（按原文粘贴）",
            "person_summary": "（按原文粘贴）",
            "person_profile": "（按原文粘贴）",
            "person_fact": [
                "（按原文粘贴）",
                "（按原文粘贴）",
                "（按原文粘贴）",
                "（按原文粘贴）",
                "（按原文粘贴）"
            ]
            }},

        "主用户关系图":
            [
                {{
                    "person": "（按原文粘贴）",
                    "person_relation: "（按原文粘贴）",
                    "person_fact": [
                        "（按原文粘贴）",
                        "（按原文粘贴）",
                        "（按原文粘贴）",
                        "（按原文粘贴）",
                        "（按原文粘贴）"
                    ]
                }},
                ...(如有更多用户继续追加)
            ],
        
        "对话历史":
        [
            {{
                "user": "用户提问（口语化且适合 TTS）",
                "bot": "bot 回复，符合用户画像和 TTS 适配要求。",
                "关系检索关键词": null或[],
                "事实检索关键词": null或[],
                "support": null或[]或[
                    {{
                        "person": "",
                        "person_relation": "",
                        "person_fact": 
                            [
                                "",
                                ""
                            ]
                    }}
                    ...(如有更多用户继续追加)
                ]
            }}
            ...(如有更多对话继续追加)
        ]
    }}
]

输入数据：
{json.dumps(entry, ensure_ascii=False, indent=2)}'''
            
            request_obj = {
                "custom_id": f"request-{request_id}",
                "body": {
                    "messages": [
                        {"role": "user", "content": prompt_text}
                    ],
                    "max_tokens": 2000,
                    "top_p": 1
                }
            }

            # 写入 jsonl 文件
            save_f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")
            request_id += 1

        print(f"已生成预处理文件")


def data_prepare_machine_exa(machine_path,save_path):
    with open(machine_path, 'r', encoding='utf-8') as f:
        machine_data = json.load(f)

    all_info = []  # 保存所有组数据

    for key in machine_data["家用电器"]:
        category = key["name"]
        differences_num = random.randint(2,5)
        tongyong_properties_str = ','.join(key["通用参数"])
        bitian_str = ','.join(key["必填参数"])
        xuantian_str = ','.join(key["其他参数"])


        prompt = f'''
请根据以下要求，为这一类型的家用电器生成一个新的符合实际的具体产品信息，并确保输出的格式符合要求。产品信息应该包括详细的技术规格和功能特点。

- **{category}**
  - 应包含的内容：{tongyong_properties_str}，{bitian_str}，{xuantian_str}（提供的各个属性字段列表）
  - 请生成 1 条不同型号的、合理的且包含各个属性的电器信息。
注意：
-所有属性除features需要分点列举外，其他所有属性都不要添加额外的描述，只给出具体的参数，也不需要在（）中添加其他描述
-将所有属性标题用英文表示，如：name、brand
-请确保产品信息合理、详细、准确，并为产品提供独特的规格和功能。

确保产品的输出信息符合以下json格式：

```json
{{
    "产品类别": 
    {{
        "通用参数":{{
            "name": "产品名称",
            "brand": "品牌",
            "price": "价格",
            "model": "型号",
            "features": ["特色功能1", "特色功能2", "特色功能3", ...]
        }}
        "必填参数": {{
            ... (其余属性)
        }}
        "其他参数":{{
            ... (其余属性)
        }}
}}
}}
```

##示例一：空调：
{{
    "空调":{{
        "通用参数":{{
            "name": "静悦挂机空调",
            "brand": "海尔",
            "price": 2049,
            "model": "KFR-35GW/01KMC81U1",
            "features": ["省电", "制热快", "大风量", "远程智控", "低噪", "光感护眠", "自洁净"]
        }}
        "必填参数": {{
            "type": "壁挂式",
            "cooling_capacity": "3500W", 
            "cooling_power": "810W",
            "heating_capacity": "5000W",
            "heating_power": "1245W",
            "energy_efficiency_ratio": "5.28",
            "energy_efficiency_rating": "新一级能效",
            "recirculated_air_volume": "700m³/h",
            "indoor_unit_dimensions": "865mm*300mm*190mm",
            "outdoor_unit_dimensions": "820mm*550mm*325mm",
            "net_weight_of_indoor_unit": "9.5kg",
            "net_weight_of_outdoor_unit": "24kg",
            "usable_area": "16-20㎡"
        }}
        "其他参数":{{
            "warranty_period":"一年6个月",
            "if_on_site_installation_supported":"是"
        }}
    }}
}}
##示例二：冰箱：
{{
    "空调":{{
        "通用参数":{{
            "name":"616L双开门家用一级超薄风冷无霜冰箱",
            "brand": "海尔",
            "price":2249,
            "model": "BCD-616WGHSSEDC9",
            "features": ["大容量", "抗菌除异味", "大冷冻力","多路送风","自动悬停","省电","隐藏式把手"]
        }}
        "必填参数": {{
            "type": "对开门",
            "total_capacity": "616L",
            "freezer_capacity": "223L",
            "refrigerator_capacity": "393L",
            "cooling_method": "风冷",
            "freezing_capacity": "8.5kg/12h",
            "control_method": "电脑",
            "rated_voltage": "220V",
            "rated_frequency": "50Hz",
            "total_power_consumption": "0.95kWh/24h",
            "energy_efficiency": "一级",
            "noise_level": "37dB",
            "dimensions": "716mm*905mm*1775mm"
        }}
        "其他参数":{{
            "warranty_period":"1年",
            "if_on_site_installation_supported":"是",
            "compressor_type": "变频",
            "variable_temperature_zone_capacity":"0L"
        }}
    }}
}}
'''
        all_info.append(prompt)

    # 保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_info, f, ensure_ascii=False, indent=2)

    print(f"已成功生成 {len(all_info)} 条最终信息，保存到 {save_path}")

def data_prepare_sp_exa(machine_path,save_path,sp_name):
    with open(machine_path, 'r', encoding='utf-8') as f:
        machine_data = json.load(f)

    all_info = []  # 保存所有组数据

    for key in machine_data[sp_name]:
        category = key["name"]
        differences_num = random.randint(2,5)
        tongyong_properties_str = ','.join(key["通用参数"])
        bitian_str = ','.join(key["必填参数"])
        xuantian_str = ','.join(key["其他参数"])


        prompt = f'''
请根据以下要求，为这一类型的商品生成一个新的符合实际的具体产品信息，并确保输出的格式符合要求。产品信息应该包括详细的技术规格和功能特点。

- **{category}**
  - 应包含的内容：{tongyong_properties_str}，{bitian_str}，{xuantian_str}（提供的各个属性字段列表）
  - 请生成 1 条不同型号的、合理的且包含各个属性的商品信息。
注意：
-所有属性除features需要分点列举外，其他所有属性都不要添加额外的描述，只给出具体的参数，也不需要在（）中添加其他描述
-将所有属性标题用英文表示，如：name、brand
-请确保产品信息合理、详细、准确，并为产品提供独特的规格和功能。
-参考以下示例的格式进行生成，对生成的参数要加单位。只给出具体的参数，不需要在（）中添加其他描述。

确保产品的输出信息符合以下json格式：

```json
{{
    "产品类别": 
    {{
        "通用参数":{{
            "name": "商品名称",
            ...(其余属性)
        }}
        "必填参数": {{
            ... (其余属性)
        }}
        "其他参数":{{
            ... (其余属性)
        }}
}}
}}
```


##示例：冰箱：
{{
    "冰箱":{{
        "通用参数":{{
            "name":"616L双开门家用一级超薄风冷无霜冰箱",
            "brand": "海尔",
            "price":2249,
            "model": "BCD",
            "features": ["大容量", "抗菌除异味", "大冷冻力","多路送风","自动悬停","省电","隐藏式把手"]
        }}
        "必填参数": {{
            "type": "对开门",
            "total_capacity": "616L",
            "freezer_capacity": "223L",
            "refrigerator_capacity": "393L",
            "cooling_method": "风冷",
            "freezing_capacity": "8.5kg/12h",
            "control_method": "电脑",
            "rated_voltage": "220V",
            "rated_frequency": "50Hz",
            "total_power_consumption": "0.95kWh/24h",
            "energy_efficiency": "一级",
            "noise_level": "37dB",
            "dimensions": "716mm*905mm*1775mm"
        }}
        "其他参数":{{
            "warranty_period":"1年",
            "if_on_site_installation_supported":"是",
            "compressor_type": "变频",
            "variable_temperature_zone_capacity":"0L"
        }}
    }}
}}
'''
        all_info.append(prompt)

    # 保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_info, f, ensure_ascii=False, indent=2)

    print(f"已成功生成 {len(all_info)} 条最终信息，保存到 {save_path}")



#所有通用参数，2-4个必填参数，让模型生成差异较大的商品信息
def data_prepare_machine(machine_path,machine_a_exp_output,save_path,scene_name,cat_num):
    with open(machine_path, 'r', encoding='utf-8') as f:
        machine_data = json.load(f)

    with open(machine_a_exp_output, 'r', encoding='utf-8') as f:
        machine_a_exp = json.load(f)

    with open('/mnt/common/intern/qt/8w_project/machine_data/che/input/name.json', 'r', encoding='utf-8') as f:
        name = json.load(f)

    all_info = []  # 保存所有组数据

    for key in machine_data[scene_name]:
        if key:
            category = key["name"]
            tongyong_properties = key["通用参数"]
            bitian_properties = key["必填参数"]

            tongyong_str = ','.join(key["通用参数"])
            bitian_str = ','.join(key["必填参数"])
            xuantian_str = ','.join(key["其他参数"])

            price = key["通用参数"][3]
            # print(category)
            example = [i.get(category) for i in machine_a_exp if i is not None and category in i]

            # example = [i.get(category) for i in machine_a_exp if i.get(category)]
            # print(example)

            for i in range(1):
                bitian_differences_num = random.randint(3,5)
                bitian_difference = random.sample(bitian_properties, bitian_differences_num) #返回列表
                # tongyong_differences_num = random.randint(1,2)
                # tongyong_difference = random.sample(tongyong_properties, tongyong_differences_num) #返回列表
                name_advice = random.sample(name, 3)
                prompt = f'''
根据以下要求，生成一个新的、符合实际的、具有显著差异的具体产品信息。并确保输出的格式符合要求。产品信息应该包括详细的技术规格和功能特点

**产品类型**：
- {category}

**参数要求**：
- 通用参数：{tongyong_str}
- 必填参数：{bitian_str}
- 其他参数：{xuantian_str}

**生成要求**：
任务1：生成新产品的参数
- 请生成一个与原始示例商品参数差异较大的新产品参数。
- 确保新生成产品的参数种类和名称与参数要求保持一致，
- 所有参数必须有显著差异，特别是通用参数和必填参数要有巨大的变化。所有数值都应相对于原始示例进行上下调整：
    - 通用参数：{tongyong_properties}
    - 必填参数：{bitian_difference}

任务2：更新产品名称
- 根据这3个名称建议：{name_advice}和任务1中生成的产品功能特点（features字段），更新产品名称（name字段）。
- 请注意，生成的产品名称应符合产品功能特点并参考名称建议，名称命名不局限于常见的类目，选择小众的、有创意的名称。

**注意**：
- 严禁添加任何未在参数列表中出现的字段。
- 除 features 字段外，其余字段只填写具体值，不添加额外描述或括号说明
- features 应为多项功能，使用列表形式分点列出
- 将所有参数标题用英文表示，如：name、brand
- 请确保产品信息合理、详细、准确，并为产品提供独特的规格和功能。

### **原始示例**（用于参考差异化方向）
{example}

**输出格式**：
```json
{{
    "{category}": {{
    "通用参数": {{
        "name": "产品名称",
        "brand": "品牌",
        "price": "价格",
        "model": "型号",
        "features": ["特色功能1", "特色功能2", "特色功能3", ...]
    }},
    "必填参数": {{
        ... (其余属性)
    }},
    "其他参数": {{
        ... (其余属性)  
    }}
    }}
}}
```
请严格按照上述要求进行输出，必须严格使用 JSON 格式，且不要在 JSON 前后添加任何解释或额外文字

    '''      
                
                print(prompt)
                all_info.append(prompt)

    # 保存到文件
    if len(all_info) == cat_num:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(all_info, f, ensure_ascii=False, indent=2)

        print(f"已成功生成 {len(all_info)} 条最终信息，保存到 {save_path}")
    else:
        print(len(all_info))
        print("——————————————一个文件不全————————————————-")

# # import docx
# def docx_to_jsonl(docx_file, output_file):
#     # 解析 docx 文件内容的函数
#     def extract_content(docx_file):
#         # 打开 docx 文件
#         doc = docx.Document(docx_file)
#         data = []
#         title = None
#         time = None
#         content = []

#         # 遍历 docx 文件中的段落
#         for para in doc.paragraphs:
#             if para.style.name == 'Heading 1':  # 一级标题
#                 if title:  # 存储之前的一个段落内容
#                     data.append({
#                         'title': title,
#                         'time': time,
#                         'content': '\n'.join(content)
#                     })
#                 # 更新当前的一级标题
#                 title = para.text.strip()
#                 time = None  # 重置二级标题（时间）
#                 content = []  # 重置正文内容
#             elif para.style.name == 'Heading 2':  # 二级标题
#                 time = para.text.strip()
#             else:  # 正文内容
#                 if para.text.strip():  # 如果有正文
#                     content.append(para.text.strip())

#         # 处理最后一个标题和内容
#         if title:
#             data.append({
#                 'title': title,
#                 'time': time,
#                 'content': '\n'.join(content)
#             })

#         return data

#     # 保存为 JSONL 文件的函数
#     def save_to_jsonl(data, output_file):
#         with open(output_file, 'w', encoding='utf-8') as f:
#             for entry in data:
#                 json.dump(entry, f, ensure_ascii=False)
#                 f.write('\n')

#     # 提取内容
#     extracted_data = extract_content(docx_file)
    
#     # 保存为 JSONL 文件
#     save_to_jsonl(extracted_data, output_file)

#     print(f"数据已保存到 {output_file}")


# import os
# import docx
# import json

# def docxdir_to_jsonl(docx_dir, output_file):
#     def extract_content(docx_dir):
#         data = []
#         for docx_file in os.listdir(docx_dir):
#             docx_file_path = os.path.join(docx_dir, docx_file)
            
#             doc = docx.Document(docx_file_path)

#             title = os.path.basename(docx_file_path).rsplit('.', 1)[0]  # 使用文件名作为标题
#             content = []  # 记录正文内容，按段落分开

#             # 遍历 docx 文件中的段落
#             for para_idx, para in enumerate(doc.paragraphs):  # para_idx 记录段落索引
#                 if para.style.name == 'Heading 1':  # 一级标题
#                     if title:  # 存储之前的一个段落内容
#                         data.append({
#                             'title': title,
#                             'content': content  # 直接保存段落列表
#                         })
#                     # 更新当前的一级标题
#                     title = para.text.strip()
#                     time = None  
#                     content = []  # 重置正文内容
#                 elif para.style.name == 'Heading 2':  # 二级标题
#                     time = para.text.strip()
#                 else:  # 正文内容
#                     if para.text.strip():  # 如果有正文内容
#                         content.append(para.text.strip()  # 段落内容
#                         )

#             # 处理最后一个标题和内容
#             if title:
#                 data.append({
#                     'title': title,
#                     'content': content  # 保存最后一部分内容
#                 })

#         print(f"提取的 {len(data)} 条数据")
#         return data

#     def save_to_jsonl(data, output_file):
#         with open(output_file, 'w', encoding='utf-8') as f:
#             for entry in data:
#                 json.dump(entry, f, ensure_ascii=False)
#                 f.write('\n')

#     # 提取内容并保存
#     extracted_data = extract_content(docx_dir)
#     save_to_jsonl(extracted_data, output_file)
#     print(f"数据已保存到 {output_file}")




def data_prepare_docx(machine_path,machine_a_exp_output,save_path):
    with open(machine_path, 'r', encoding='utf-8') as f:
        machine_data = json.load(f)

    with open(machine_a_exp_output, 'r', encoding='utf-8') as f:
        machine_a_exp = json.load(f)
    all_info = []  # 保存所有组数据

    for key in machine_data["家用电器"]:
        category = key["name"]
        properties = key["properties"]
        print(category)
        example = [i.get(category) for i in machine_a_exp if i.get(category)]
        print(example)
        properties_str = key["p_str"]

        for i in range(10):
            differences_num = random.randint(2,5)
            per_difference = random.sample(properties, differences_num) #返回列表


            prompt = f'''
请根据以下要求，为这一类型的家用电器生成一个新的符合实际的具体产品信息，并确保输出的格式符合要求。产品信息应该包括详细的技术规格和功能特点。

- **{category}**
- 包含的内容：{properties_str}（属性列表）
- 请生成 1 条不同型号的电器信息。
- 主要差异点：{per_difference}（差异点的描述,这些属性在生成新产品时需要相较于给出的示例发生较大变化）
注意：
-除主要差异点的差异较大外，其他属性也应有一定差异
-所有属性除feature需要分点列举外，其他所有属性都不要添加额外的描述，只给出具体的参数，也不需要在（）中添加其他描述
-将所有属性标题用英文表示，如：name、brand
-请确保产品信息合理、详细、准确，并为产品提供独特的规格和功能。

请确保产品信息详细、准确，并为产品提供独特的规格和功能。

### 例子：原始示例，仅用于对比生成不同的产品
{example}

确保产品的输出信息符合以下格式：

```json
{{
    "{category}": {{
            "name": "产品名称",
            "brand": "品牌",
            "price": "价格",
            "model": "型号",
            ... (其余属性)
            "features": ["特色功能1", "特色功能2", "特色功能3", ...]
        }}
}}
```
'''      
            
            print(prompt)
            all_info.append(prompt)

    # 保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_info, f, ensure_ascii=False, indent=2)

    print(f"已成功生成 {len(all_info)} 条最终信息，保存到 {save_path}")


def data_prepare_machine_cov(all_machine,all_machine_id,scene_save_path,person_fact,bot_meta,bot_cov,save_path,n):
    all_info = []  # 保存所有组数据

    generate_id_categories_scene_prepare(all_machine,all_machine_id,scene_save_path,n)

    with open (scene_save_path,'r',encoding='utf-8') as f:
        scene_data = json.load(f)

    for i in range(1):

        #一个场景下的3-5个种类，每个种类2-4个电器
        with open(all_machine_id,'r',encoding='utf-8') as f:
            all_data = json.load(f)

        categary = list(all_data.keys())
        a_shop_categary = random.sample(categary,random.randint(3,5))
        a_shop = {}
        for categary in a_shop_categary:
            a_shop[categary] = random.sample(all_data[categary],random.randint(1,3))

        #原子集以及对应权重
        elements = ['SP', 'NSP','BOT','XL', 'MXL']
        weights = [0.4, 0.1,0.1,0.1,0.1,]
        yuanzi = random.choices(elements, weights=weights, k=10)  #.join('->')
        #机器人信息：随机取一个
        with open(bot_meta,'r',encoding='utf-8') as f:
            bot_info = json.load(f)
        bot = random.choice(bot_info)

        #人物信息：随机取一个
        with open(person_fact,'r',encoding='utf-8') as f:
            person_info = json.load(f)
        person = random.choice(person_info)
        #闲聊：随机取3个
        with open(bot_cov,'r',encoding='utf-8') as f:
            cov_info = json.load(f)
        q1 = random.choice(random.choice(cov_info))
        q2 = random.choice(random.choice(cov_info))
        q3 = random.choice(random.choice(cov_info))

        prompt = f'''
当前输入数据：  
1. **用户信息**：
{person}
2. **机器人信息**：
{bot}
3. **商品信息（一个场景中多个类目下的具体商品信息）**：
{a_shop}
4. **对话outline**：
{yuanzi}
5. **闲聊对话参考**：
{q1}
{q2}
{q3}
'''      
        print(prompt)
        all_info.append(prompt)

    # 保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_info, f, ensure_ascii=False, indent=2)

    print(f"已成功生成 {len(all_info)} 条最终信息，保存到 {save_path}")





class TreeNode:
    def __init__(self, element):
        self.element = element 
        self.children = []  

    def add_child(self, child, probability):
        self.children.append((child, probability))  


def get_node(a_node_name):
    
    node_structure = {
        "START": [("SP", 0.1),("SPD", 0.4), ("XL", 0.1), ("MXL", 0.1), ("YH", 0.1), ("BOT", 0.1), ("NSP", 0.1)],
        "SP": [("SP", 0.1),("SPD", 0.4), ("XL", 0.1), ("MXL", 0.1), ("YH", 0.1), ("BOT", 0.1), ("NSP", 0.1)],
        "XL": [("SP", 0.1),("SPD", 0.4), ("XL", 0.1), ("MXL", 0.1), ("YH", 0.1), ("BOT", 0.1), ("NSP", 0.1)],
        "MXL": [("SP", 0.1),("SPD", 0.4), ("XL", 0.1), ("MXL", 0.1), ("YH", 0.1), ("BOT", 0.1), ("NSP", 0.1)],
        "BOT": [("SP", 0.1),("SPD", 0.4), ("XL", 0.1), ("MXL", 0.1), ("YH", 0.1), ("BOT", 0.1), ("NSP", 0.1)],
        "YH": [("SP", 0.1),("SPD", 0.4), ("XL", 0.1), ("MXL", 0.1), ("YH", 0.1), ("BOT", 0.1), ("NSP", 0.1)],
        "NSP": [("SP", 0.1),("SPD", 0.4), ("XL", 0.1), ("MXL", 0.1), ("YH", 0.1), ("BOT", 0.1), ("NSP", 0.1)],
        "SPD": [("SP", 0.1),("SPD", 0.4), ("XL", 0.1), ("MXL", 0.1), ("YH", 0.1), ("BOT", 0.1), ("NSP", 0.1)],
        "END": []
    }
    return node_structure.get(a_node_name, [])

def build_tree(n):
    root_name = "START"
    root = TreeNode(root_name)  

    # 递归构建树
    def add_children(node_name, parent_node, current_depth):
        children_info = get_node(node_name)

        for child, prob in children_info:
            child_node = TreeNode(child)
            parent_node.add_child(child_node, prob)  # 添加子节点到父节点

            if current_depth < n:
                add_children(child, child_node, current_depth + 1)

    add_children(root_name, root, 1)
    
    return root

#遍历打印
def traverse_tree(node, depth=0):
    print('  ' * depth + f'Node: {node.element}')
    for child, prob in node.children:
        print('  ' * (depth + 1) + f'Child: {child.element}, Probability: {prob}')
        traverse_tree(child, depth + 2)

#根据概率构造outline，遇到END终止
def traverse_tree_randomly(node, visited_nodes=None, target="END"):
    if visited_nodes is None:
        visited_nodes = []
    
    visited_nodes.append(node.element)
    
    if node.element == target:
        visited_nodes.append(node.element)  # 加入终止节点
        return visited_nodes
    if not node.children:
        return visited_nodes  # 如果没有子节点，返回遍历过的节点列表
    
    # 获取子节点和概率列表
    children, probabilities = zip(*node.children)
    # 根据概率随机选一个子节点
    chosen_child = random.choices(children, probabilities)[0]
    # 遍历
    return traverse_tree_randomly(chosen_child, visited_nodes, target)

# 如果构造的outline没有END，添加"END"节点
def complete_traversal(node):
    visited_nodes = traverse_tree_randomly(node)
    
    if "END" not in visited_nodes:
        visited_nodes.append("END")
    
    return visited_nodes


#为tts后的商品和类目设置唯一id,并设置n个场景。每个场景的类目数量和商品数量在函数里设置
def n_scene_person(person_fact,all_machine_id,scene_save_path,n):
    #构建n个场景
    with open(all_machine_id,'r',encoding='utf-8') as f:
      all_data = json.load(f)
    all_shop = []  

    for i in range(n):
            #人物信息：随机取一个
        with open(person_fact,'r',encoding='utf-8') as f:
            person_info = json.load(f)
        person = random.choice(person_info)

        # 一个场景下的 3-5 个类目，每个类目 2-4 个电器
        category = list(all_data.keys())
        a_shop_category = random.sample(category, random.randint(3, 5)) 
        a_shop = {}
        a_shop_id = i
        a_shop["scene_id"] = a_shop_id  # 为场景添加唯一 ID
        a_shop["person"] = person
        a_shop["scene_info"] = {}  

        for cat in a_shop_category:
            if 'products' in all_data[cat]:
                products = all_data[cat]['products']

                a_shop["scene_info"][cat]= {
            "category_id":all_data[cat]['category_id'],  # 类目ID
            "description":all_data[cat]['description'],  # 类目ID
            "products": random.sample(products, random.randint(2, 4))    # 类目下的商品
        }

            # a_shop[cat] = random.sample(products, random.randint(1, 3))

        all_shop.append(a_shop)  

    with open(scene_save_path, 'w', encoding='utf-8') as f:
        json.dump(all_shop, f, ensure_ascii=False, indent=4)
    

def set_id(all_machine,all_machine_id):
    #NODE 构造id，运行一次后不能重新运行，在all_machine_id手动添加decrib

    d = {}
    with open(all_machine,'r',encoding='utf-8') as f:
        all_data = json.load(f)

    cat_id = 0
    for category, products in all_data.items():
        category_id = f"cat_{cat_id}"
        
        # 为类目中的每个商品生成唯一ID
        product_data = []
        for i,product in enumerate(products):
            product["product_id"] = f"_{i}"  # 添加商品ID
            product_data.append(product)

        d[category] = {
            "category_id": category_id,  # 类目ID
            "products": product_data  # 类目下的商品
        }

        cat_id +=1

    with open(all_machine_id,'w',encoding='utf-8') as f:
        json.dump(d,f,ensure_ascii=False, indent=4)


def sp_name(all_machine,all_machine_id,all_machine_save_path,all_machine_id_save_path):
   
    with open(all_machine,'r',encoding='utf-8') as f:
        all_data_all_machine = json.load(f)

    for category, products in all_data_all_machine.items():        

        for product in products:
            if category not in product["通用参数"]["name"]:
                product["通用参数"]["name"] = product['通用参数']['name'] + category

    with open(all_machine_save_path,'w',encoding='utf-8') as f:
        json.dump(all_data_all_machine,f,ensure_ascii=False, indent=4)

 ###

    with open(all_machine_id,'r',encoding='utf-8') as f:
        all_data_all_machine_id = json.load(f)

    for category, info in all_data_all_machine_id.items():        

        for product in info["products"]:
            if category not in product["通用参数"]["name"]:
                product["通用参数"]["name"] = product['通用参数']['name'] + category

    with open(all_machine_id_save_path,'w',encoding='utf-8') as f:
        json.dump(all_data_all_machine_id,f,ensure_ascii=False, indent=4)



#为tts后的商品和类目设置n个场景。每个场景的类目数量和商品数量在函数里设置
def generate_id_categories_scene_prepare(all_machine_id,scene_save_path,n):

    #构建n个场景
    with open(all_machine_id,'r',encoding='utf-8') as f:
      all_data = json.load(f)
    all_shop = []  

    for i in range(n):
        # 一个场景下的 3-5 个类目，每个类目 1-4 个电器
        category = list(all_data.keys())
        print(category)
        a_shop_category = random.sample(category, random.randint(3, 5)) 
        a_shop = {}
        a_shop_id = i
        a_shop["scene_id"] = a_shop_id  # 为场景添加唯一 ID
        a_shop["scene_info"] = {}  

        for cat in a_shop_category:
            if 'products' in all_data[cat]:
                products = all_data[cat]['products']
                # print(products)
                a_shop["scene_info"][cat]= {
            "category_id":all_data[cat]['category_id'],  # 类目ID
            "description":all_data[cat]['description'],  # 类目ID
            "products": random.sample(products, random.randint(1, 3))    # 类目下的商品
        }

            # a_shop[cat] = random.sample(products, random.randint(1, 3))
        # print(a_shop["scene_info"])
        all_shop.append(a_shop)  

    with open(scene_save_path, 'w', encoding='utf-8') as f:
        json.dump(all_shop, f, ensure_ascii=False, indent=4)
    

#随机构造n个场景,m个相同场景下的不同outline
# 每个场景中的类目和商品数在generate_id_categories_scene_prepare函数中设置.

def data_prepare_machine_cov_tree(all_machine,all_machine_id,scene_save_path,person_fact,bot_meta,bot_cov,save_path,n,m):
    all_info = []  # 保存所有组数据

    generate_id_categories_scene_prepare(all_machine,all_machine_id,scene_save_path,n)

    with open (scene_save_path,'r',encoding='utf-8') as f:
        scene_data = json.load(f)


    for i in range(m):
        for scene in scene_data:

            root = build_tree(3)
            yuanzi = complete_traversal(root)

            #机器人信息：随机取一个
            with open(bot_meta,'r',encoding='utf-8') as f:
                bot_info = json.load(f)
            bot = random.choice(bot_info)

            #人物信息：随机取一个
            with open(person_fact,'r',encoding='utf-8') as f:
                person_info = json.load(f)
            person = random.choice(person_info)
            #闲聊：随机取3个
            with open(bot_cov,'r',encoding='utf-8') as f:
                cov_info = json.load(f)
            q1 = random.choice(random.choice(cov_info))
            q2 = random.choice(random.choice(cov_info))
            q3 = random.choice(random.choice(cov_info))

            prompt = f'''
当前输入数据：  
1. **用户信息**：
{person}
2. **机器人信息**：
{bot}
3. **商品信息（一个场景中多个类目下的具体商品信息）**：
{scene}
4. **对话outline**：
{yuanzi}
5. **闲聊对话参考**：
{q1}
{q2}
{q3}
'''      
            print(prompt)
            all_info.append(prompt)

    # 保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_info, f, ensure_ascii=False, indent=2)

    print(f"已成功生成 {len(all_info)} 条最终信息，保存到 {save_path}")




def traverse_tree_randomly(node_name, node_structure, node_actions):
    # 执行当前节点的动作
    for action in node_actions.get(node_name, []):
        print(action)  # 打印动作内容
    
    # 如果是END节点，结束
    if node_name == "END":
        return
    
    # 获取当前节点的所有可能下一节点和它们的概率
    next_nodes, probabilities = zip(*node_structure[node_name])
    
    # 随机选择下一个节点
    chosen_node = random.choices(next_nodes, probabilities)[0]
    
    # 递归调用继续遍历
    traverse_tree_randomly(chosen_node, node_structure, node_actions)

# 从START节点开始
# traverse_tree_randomly("START", node_structure, node_actions)




# 构建每个场景下的m个不同outline
def data_prepare_machine_cov_action(scene_save_path,person_fact,bot_meta,bot_cov,save_path,m):
    all_info = []  # 保存所有组数据
    emotions = [
        "愉快", "焦虑", "失望", "惊讶", "不耐烦", "高兴", "沮丧", "满足", "困惑", "期待",
        "厌烦", "愤怒", "满意", "惶恐", "悲伤", "疑虑", "期待", "惊喜", "犹豫", 
        "失落", "兴奋", "迷茫", "愤慨", "放心", "无聊", "兴奋", "决策困难", "购物冲动", 
        "放松", "矛盾", "善意", "冷淡", "急躁", "欣赏", "怀疑", "购物疲劳", "选择困难", 
         "预算压力", "困扰", "自信", "满足", "感激"
    ]


    with open (scene_save_path,'r',encoding='utf-8') as f:
        scene_data = json.load(f)


    for i in range(m):
        for scene in scene_data:
            data ={
                "scene":scene["scene"],
                "data":scene["data"]
            }

            action = scene["action"]

            #机器人信息：随机取一个
            with open(bot_meta,'r',encoding='utf-8') as f:
                bot_info = json.load(f)
            bot = random.choice(bot_info)

            #人物情绪
            emotion = random.choice(emotions)

            #人物信息：随机取一个
            with open(person_fact,'r',encoding='utf-8') as f:
                person_info = json.load(f)
            person = random.choice(person_info)
            #闲聊：随机取3个
            with open(bot_cov,'r',encoding='utf-8') as f:
                cov_info = json.load(f)
            q1 = random.choice(random.choice(cov_info))
            q2 = random.choice(random.choice(cov_info))
            q3 = random.choice(random.choice(cov_info))

            prompt = f'''
当前输入数据：  
1. **用户信息**：
{person}
2. **机器人信息**：
{bot}
3. **商品信息（一个场景中多个类目下的具体商品信息）**：
{data}
4. **action**：
{action}
5. **闲聊对话参考**：
{q1}
{q2}
{q3}
6. **购物者的情绪**:
{emotion}
'''      
            # print(prompt)
            all_info.append(prompt)

    # 保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_info, f, ensure_ascii=False, indent=2)

    print(f"已成功生成 {len(all_info)} 条最终信息，保存到 {save_path}")



# 构建每个场景下的m个不同outline
def data_prepare_machine_cov_action_no_person(scene_save_path,bot_meta,bot_cov,save_path,m):
    all_info = []  # 保存所有组数据
    emotions = [
        "愉快", "焦虑", "失望", "惊讶", "不耐烦", "高兴", "沮丧", "满足", "困惑", "期待",
        "厌烦", "愤怒", "满意", "惶恐", "悲伤", "疑虑", "期待", "惊喜", "犹豫", 
        "失落", "兴奋", "迷茫", "愤慨", "放心", "无聊", "兴奋", "决策困难", "购物冲动", 
        "放松", "矛盾", "善意", "冷淡", "急躁", "欣赏", "怀疑", "购物疲劳", "选择困难", 
         "预算压力", "困扰", "自信", "满足", "感激"
    ]

    with open (scene_save_path,'r',encoding='utf-8') as f:
        scene_data = json.load(f)


    for i in range(m):
        for scene in scene_data:
            data ={
                "scene":scene["scene"],
                "data":scene["data"]
            }

            action = scene["action"]

            #机器人信息：随机取一个
            with open(bot_meta,'r',encoding='utf-8') as f:
                bot_info = json.load(f)
            bot = random.choice(bot_info)

            #人物情绪
            emotion = random.choice(emotions)


            #人物信息：kong
            person = {}
            #闲聊：随机取3个
            with open(bot_cov,'r',encoding='utf-8') as f:
                cov_info = json.load(f)
            q1 = random.choice(random.choice(cov_info))
            q2 = random.choice(random.choice(cov_info))
            q3 = random.choice(random.choice(cov_info))

            prompt = f'''
当前输入数据：  
1. **用户信息**：
{person}
2. **机器人信息**：
{bot}
3. **商品信息（一个场景中多个类目下的具体商品信息）**：
{data}
4. **action**：
{action}
5. **闲聊对话参考**：
{q1}
{q2}
{q3}
6. **购物者的情绪**:
{emotion}
'''      
            # print(prompt)
            all_info.append(prompt)

    # 保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_info, f, ensure_ascii=False, indent=2)

    print(f"已成功生成 {len(all_info)} 条最终信息，保存到 {save_path}")

# def out_action_process(action_out_path,scene_save_path,person_fact,bot_meta,save_path):
#     with open (action_out_path,'r',encoding='utf-8') as f:
#         output_data = json.load(f)
#     with open (scene_save_path,'r',encoding='utf-8') as f:
#         scene_ori_data = json.load(f)
#     with open(bot_meta,'r',encoding='utf-8') as f:
#         bot_info = json.load(f)
#     with open(person_fact,'r',encoding='utf-8') as f:
#         person_info = json.load(f)

#     all_data = []
#     for data in output_data:
#         person = next((p for p in person_info if p["person_id"] == data["person_id"]), None)
#         bot = next((p for p in bot_info if p["bot_id"] == data["bot_id"]), None)

#         scene_data = next((p for p in scene_ori_data if p["scene"] == data["scene"]), None)
            
        
#         action_list = data["action_list"]


#         for action in action_list:
#             if (action["role"] == "system.call_response") and ("<ret>/api/products/products?category_id=" in action["content"]):
#                 # 提取 category_id
#                 print(action['content'])
#                 category_id = action['content'].split('category_id=')[1].split(',')[0]
#                 print(category_id)

#                 # 获取该类目下的商品信息
#                 for category, info in scene_data["data"].items():
#                     print(category,info)
#                     if info['category_id'] == category_id:
#                         products = info['products']  # 返回对应类目下的商品信息               

#                 # 格式化返回数据
#                 new_action = {
#                     "role": "system.call_response",
#                     "call": f"/api/products/products?category_id={category_id}",
#                     "return": products
#                 }
                
#                 # 替换原有 action
#                 action.update(new_action)

#         all_data.append({
#             "person":person,
#             "bot":bot,
#             "data":action_list
#         })
    
#     random.shuffle(all_data)
    
#     with open(save_path,'w',encoding='utf-8') as f:
#         json.dump(all_data,f,ensure_ascii=False,indent=4)


def out_action_process(action_out_path, scene_save_path, person_fact, bot_meta, save_path):
    with open(action_out_path, 'r', encoding='utf-8') as f:
        output_data = json.load(f)
    with open(scene_save_path, 'r', encoding='utf-8') as f:
        scene_ori_data = json.load(f)
    with open(bot_meta, 'r', encoding='utf-8') as f:
        bot_info = json.load(f)
    with open(person_fact, 'r', encoding='utf-8') as f:
        person_info = json.load(f)

    all_data = []
    for data in output_data:
        try:
            # 获取person和bot的信息
            person = next((p for p in person_info if p["person_id"] == data["person_id"]), None)
            bot = next((p for p in bot_info if p["bot_id"] == data["bot_id"]), None)

            # 获取scene数据
            scene_data = next((p for p in scene_ori_data if p["scene"] == data["scene"]), None)

            # 检查person, bot, scene_data是否为None
            if person is None or bot is None or scene_data is None:
                print(f"Skipping data for person_id {data['person_id']} and bot_id {data['bot_id']} due to missing person, bot, or scene_data.")
                continue  # 跳过当前数据

            # 获取action_list
            action_list = data["action_list"]
            
            # 判断是否有符合跳过条件的action，如果有则跳过整个data
            skip_data = False
            for action in action_list:
                if (action["role"] == "system.call_response" and 
                    "<ret>/api/products/products?category_id=" in action["content"] and
                    ", | is_complete=yes</ret>" not in action["content"]):
                    # 如果符合条件，标记跳过当前data
                    print(f"Skipping data for person_id {data['person_id']} and bot_id {data['bot_id']} due to matching action.")
                    skip_data = True
                    break  # 跳出当前循环，跳过整个data

            if skip_data:
                continue  # 跳过整个data

            for action in action_list:
                if (action["role"] == "system.call_response") and ("<ret>/api/products/products?category_id=" in action["content"]) and (", | is_complete=yes</ret>" in action["content"]):
                    # 提取category_id
                    # print(action['content'])
                    category_id = action['content'].split('category_id=')[1].split(',')[0]
                    # print(category_id)

                    # 获取该类目下的商品信息
                    if scene_data and "data" in scene_data:
                        for category, info in scene_data["data"].items():
                            # print(category, info)
                            if info['category_id'] == category_id:
                                products = info['products']  # 返回对应类目下的商品信息

                                # 格式化返回数据
                                new_action = {
                                    "role": "system.call_response",
                                    "call": f"/api/products/products?category_id={category_id}",
                                    "return": products
                                }

                                # 替换原有 action
                                action.update(new_action)
                                break
                    else:
                        print(f"Error: Missing 'data' key in scene_data for scene {data['scene']}")

            # 保存数据
            all_data.append({
                "person": person,
                "bot": bot,
                "data": action_list
            })

        except Exception as e:
            print(f"Error processing data for person_id {data['person_id']}, bot_id {data['bot_id']}: {e}")
            continue  # 跳过当前数据，继续下一个数据

    # 随机打乱所有数据
    random.shuffle(all_data)
    print(f"Total processed data entries: {len(all_data)}")

    # 将处理后的数据保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)




def out_action_process_no_person(action_out_path, scene_save_path, bot_meta, save_path):
    with open(action_out_path, 'r', encoding='utf-8') as f:
        output_data = json.load(f)
    with open(scene_save_path, 'r', encoding='utf-8') as f:
        scene_ori_data = json.load(f)
    with open(bot_meta, 'r', encoding='utf-8') as f:
        bot_info = json.load(f)

    all_data = []
    for data in output_data:
        try:
            # 获取person和bot的信息
            bot = next((p for p in bot_info if p["bot_id"] == data["bot_id"]), None)

            # 获取scene数据
            scene_data = next((p for p in scene_ori_data if p["scene"] == data["scene"]), None)

            # 检查person, bot, scene_data是否为None
            if bot is None or scene_data is None:
                # print(f"Skipping data for person_id {data['person_id']} and bot_id {data['bot_id']} due to missing person, bot, or scene_data.")
                continue  # 跳过当前数据

            # 获取action_list
            action_list = data["action_list"]

            # 判断是否有符合跳过条件的action，如果有则跳过整个data
            skip_data = False
            for action in action_list:
                if (action["role"] == "system.call_response" and 
                    "<ret>/api/products/products?category_id=" in action["content"] and
                    ", | is_complete=yes</ret>" not in action["content"]):
                    # 如果符合条件，标记跳过当前data
                    print(f"Skipping data for person_id {data['person_id']} and bot_id {data['bot_id']} due to matching action.")
                    skip_data = True
                    break  # 跳出当前循环，跳过整个data

            if skip_data:
                continue  # 跳过整个data

            # 处理不需要跳过的数据
            for action in action_list:
                if (action["role"] == "system.call_response") and ("<ret>/api/products/products?category_id=" in action["content"]):
                    # 提取category_id
                    # print(action['content'])
                    category_id = action['content'].split('category_id=')[1].split(',')[0]
                    # print(category_id)

                    # 获取该类目下的商品信息
                    if scene_data and "data" in scene_data:
                        for category, info in scene_data["data"].items():
                            # print(category, info)
                            if info['category_id'] == category_id:
                                products = info['products']  # 返回对应类目下的商品信息

                                # 格式化返回数据
                                new_action = {
                                    "role": "system.call_response",
                                    "call": f"/api/products/products?category_id={category_id}",
                                    "return": products
                                }

                                # 替换原有 action
                                action.update(new_action)
                                break
                    else:
                        print(f"Error: Missing 'data' key in scene_data for scene {data['scene']}")

            # 保存数据
            all_data.append({
                "person": {},
                "bot": bot,
                "data": action_list
            })

        except Exception as e:
            # print(f"Error processing data for person_id {data['person_id']}, bot_id {data['bot_id']}: {e}")
            continue  # 跳过当前数据，继续下一个数据

    # 随机打乱所有数据
    random.shuffle(all_data)
    print(f"Total processed data entries: {len(all_data)}")
    
    # 将处理后的数据保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)


# 构建每个场景下的m个不同outline
def person_shopping(persn_fact,dianqi,save_path):
    all_info = []  # 保存所有组数据
    with open(dianqi, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(persn_fact, 'r', encoding='utf-8') as f:
        persons = json.load(f)
    # 存储结果的列表
    category_params = []

    # 遍历数据，提取商品类目和类目下的参数
    for _, products in data.items():
        for product in products:
            product_info = {
                '商品类目': product['name'],
                '参数': product['通用参数'] + product['必填参数'] + product['其他参数'],

            }
            category_params.append(product_info)

    # # 输出结果
    # print(category_params)

    for person in persons:
            prompt = f'''
当前输入数据：  
1. **人物信息**：
{person}
2. **商品类目列表以及对应的参数信息**：
{category_params}
'''      
            print(prompt)
            all_info.append(prompt)

    # 保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_info, f, ensure_ascii=False, indent=2)

    print(f"已成功生成 {len(all_info)} 条最终信息")





def data_prepare_scene_a_exa(scene_path,save_path):
    with open(scene_path, 'r', encoding='utf-8') as f:
        scene_data = json.load(f)

    all_info = []  # 保存所有组数据

    for key in scene_data["场景"]:
        scene = key["scene"]

        prompt = f'''
请根据以下要求，为该场景生成该场景中某个区域的内容介绍，并确保输出的格式符合要求。

根据**{scene}**这个场景名称，生成该场景下某一个区域的内容介绍，包括该区域的名称（name），该区域的描述（description），以及该区域内一个展示机器人的名称（robot_name）。
描述需要简要概述该区域展示的内容，内容的长度应适合约3分钟的阅读时间（字数要求：300-400字）。机器人名字应该贴合该区域展示内容的特点，名称为2-4个字的中文名字。

注意：
-description的字数要求：300-400字
-不需要在（）中添加其他描述
-请确保描述信息合理、详细、准确
-description字段不需要考虑机器人特点，在生成描述时，请专注于区域内容本身。之后再为机器人命名。

确保产品的输出信息符合以下json格式：

```json
{{
    "{scene}": {{
        "name": "区域名称",
        "description": "区域描述",（字数要求：300-400字）
        "robot_name": "该区域机器人名字"
    }}
}}
```
'''
        print(prompt)
        all_info.append(prompt)

    # 保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_info, f, ensure_ascii=False, indent=2)

    print(f"已成功生成 {len(all_info)} 条最终信息，保存到 {save_path}")



def data_prepare_scene(scene_data_path,scene_a_exp_output,save_path):
    with open(scene_a_exp_output, 'r', encoding='utf-8') as f:
        scene_a_exp = json.load(f)
    with open(scene_data_path, 'r', encoding='utf-8') as f:
        scene_data = json.load(f)

    all_info = []  # 保存所有组数据

    for key in scene_data["场景"]:
        if key:
            scene = key["scene"]
            example = [i.get(scene) for i in scene_a_exp if i is not None and scene in i]
            prompt = f'''
根据以下要求，生成一个该场景中新的、与当前区域不同的、具有差异的区域信息。并确保输出的格式符合要求。

**场景**：
- {scene}
**参数要求**：包括新区域的名称（name），新区域的描述（description），以及新区域内一个展示机器人的名称（robot_name）
**生成要求**：
任务：生成新区域的信息
- 新区域的名称（name）应符合当前场景的主题。
- 描述需要简要概述该区域展示的内容，字数要求在300-400字之间，内容应适合约3分钟的阅读时间。
- 机器人名字应该贴合该区域展示内容的特点，名称为2-4个字的中文名字。

注意：
-description的字数要求：300-400字
-description为纯文本，不需要在（）中添加其他说明。生成时，请根据实际内容合理编写，不必偏向科技类内容，即使场景信息涉及机器人。
-description字段不需要考虑机器人特点，在生成描述时，请专注于区域内容本身。之后再为机器人命名。

### **原始示例**（用于参考差异化方向）
{example}

**输出格式**：确保产品的输出信息符合以下json格式：

```json
{{
    "{scene}": {{
        "name": "新区域名称",
        "description": "新区域描述",（字数要求：300-400字）
        "robot_name": "该新区域机器人名字"
    }}
}}
```
请严格按照上述要求进行输出，必须严格使用 JSON 格式，且不要在 JSON 前后添加任何解释或额外文字

'''      
        
        print(prompt)
        all_info.append(prompt)

    # 保存到文件
    if len(all_info) == 100:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(all_info, f, ensure_ascii=False, indent=2)

        print(f"已成功生成 {len(all_info)} 条最终信息，保存到 {save_path}")
    else:
        print(len(all_info))
        print("——————————————一个文件不全————————————————-")



def data_prepare_scene_exa(scene_path,save_path):
    with open(scene_path, 'r', encoding='utf-8') as f:
        scene_data = json.load(f)

    all_info = []  # 保存所有组数据

    for key in scene_data["场景"]:
        scene = key["scene"]

        prompt = f'''
请根据以下要求，为该场景生成该场景中20个不同区域的内容介绍，并确保输出的格式符合要求。

**场景**：  
- {scene}
**参数要求**：每个区域包括该区域的名称（name），该区域的描述（description），以及该区域内一个展示机器人的名称（robot_name）
**生成要求**：
任务：生成该场景下20个不同区域的信息
- 每个区域的名称（name）应符合当前场景的主题。
- 描述需要简要概述该区域展示的内容，字数要求大于600字（注意，每个展区的description字数一定要大于600字，不够就补充）。
- 机器人会在展区进行功能展示，机器人名字和功能应该贴合该区域展示内容的特点，名称为2-4个字的中文名字，功能为简短的一句话描述该机器人主要展示内容。
- 讲机器人的id设置为：robot_1/robot_2/robot_3中的一个。

注意：
-一步一步来，确保每个区域的字数要求是符合的后在生成下一个区域信息，每个区域的description的字数要求：大于600字
-description为纯文本，不需要在（）中添加其他说明。生成时，请根据实际内容合理编写，不必偏向科技类内容，即使场景信息涉及机器人。
-description字段不需要考虑机器人特点，在生成描述时，请专注于区域内容本身。之后再为机器人命名。


### **输出格式**
确保产品的输出信息符合以下json格式，包括场景名字、20个完整的展区信息。**必须严格使用 JSON 格式输出**，且不要在 JSON 前后添加任何解释或额外文字，以下面为例：


```json
{{
    "{scene}": [{{
        "name": "区域名称",
        "description": "区域描述",（字数要求：大于600字）
        "robot": [
            {{
                "id": "机器人id",
                "name": "机器人名字",
                "functions": "列举机器人展示的功能："..../..../....""
            }}
        ]
    }},
    ...（其余19个不同的区域）
    ]
}}
```
'''
        print(prompt)
        all_info.append(prompt)

    # 保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_info, f, ensure_ascii=False, indent=2)

    print(f"已成功生成 {len(all_info)} 条最终信息，保存到 {save_path}")


#有100个场景，每个场景有15个区域。每次遍历100个场景，随机取1-3个区域作为一个场景。遍历n次。共100n条数据
def n_scene_prepare_old(all_area_id,scene_save_path,n):

    #构建n个场景
    with open(all_area_id,'r',encoding='utf-8') as f:
      all_data = json.load(f)
    all_shop = []  

    a_shop_id = 0        
    for scene,data in all_data.items():
        for i in range(n):

            all_cat_info  =  random.sample(data["areas"],random.randint(1,4))        # 一个场景下1-4个区域
            a_show = {}
            a_show["scene_name"] = scene  # 场景名字

            a_show["scene_id"] = a_shop_id  # 为场景添加唯一 ID

            a_shop_id  += 1
            a_show["scene_info"] = {}   

            for cat in all_cat_info:
                a_show["scene_info"][cat["name"]]= {
                    "category_id":cat["category_id"],  # 类目ID
                    "description":cat["description"],  # 类目ID
                    "robots": [{f"robot_{random.randint(1,3)}":cat["robot_name"]}] # 类目下的商品
                }
            all_shop.append(a_show)  

    with open(scene_save_path, 'w', encoding='utf-8') as f:
        json.dump(all_shop, f, ensure_ascii=False, indent=4)


#有100个场景，每个场景有20个区域。每次遍历100个场景，随机取7-9个区域作为一个场景。遍历n次。共100n条数据
def n_scene_prepare(all_area_id,scene_save_path,n):

    #构建n个场景
    with open(all_area_id,'r',encoding='utf-8') as f:
      all_data = json.load(f)
    all_shop = []  

    a_shop_id = 0        
    for scene,data in all_data.items():
        for i in range(n):

            all_cat_info  =  random.sample(data["areas"],random.randint(7,9))        # 一个场景下7-9个区域
            a_show = {}
            a_show["scene_name"] = scene  # 场景名字

            a_show["scene_id"] = a_shop_id  # 为场景添加唯一 ID

            a_shop_id  += 1
            a_show["scene_info"] = {}   

            for cat in all_cat_info:
                a_show["scene_info"][cat["name"]]= {
                    "category_id":cat["category_id"],  # 类目ID
                    "description":cat["description"],  # 类目ID
                    "robots": cat["robot"] # 类目下的商品
                }
            all_shop.append(a_show)  

    with open(scene_save_path, 'w', encoding='utf-8') as f:
        json.dump(all_shop, f, ensure_ascii=False, indent=4)






# 构建每个场景下的m个不同outline
def data_prepare_daolan_cov_action(scene_save_path,bot_meta,bot_cov,save_path,m):
    all_info = []  # 保存所有组数据
    emotions = [
        "愉快", "焦虑", "失望", "惊讶", "不耐烦", "高兴", "沮丧", "满足", "困惑", "期待",
        "厌烦", "愤怒", "满意", "惶恐", "悲伤", "疑虑", "期待", "惊喜", "犹豫", 
        "失落", "兴奋", "迷茫", "愤慨", "放心", "无聊", "兴奋", "决策困难", "购物冲动", 
        "放松", "矛盾", "善意", "冷淡", "急躁", "欣赏", "怀疑", "购物疲劳", "选择困难", 
         "预算压力", "困扰", "自信", "满足", "感激"
    ]


    with open (scene_save_path,'r',encoding='utf-8') as f:
        scene_data = json.load(f)


    for i in range(m):
        for scene in scene_data:
            data ={
                "scene":scene["scene"],
                "scene_name":scene["scene_name"],
                "data":scene["data"]
            }

            action = scene["action"]

            #机器人信息：随机取一个
            with open(bot_meta,'r',encoding='utf-8') as f:
                bot_info = json.load(f)
            bot = random.choice(bot_info)

            #人物情绪
            emotion = random.choice(emotions)


            #闲聊：随机取3个
            with open(bot_cov,'r',encoding='utf-8') as f:
                cov_info = json.load(f)
            q1 = random.choice(random.choice(cov_info))
            q2 = random.choice(random.choice(cov_info))
            q3 = random.choice(random.choice(cov_info))

            prompt = f'''
当前输入数据：
1. **机器人信息**：
{bot}
2. **场景信息（一个场景中多个展区的具体描述和对应的机器人信息）**：
{data}
3. **action**：
{action}
4. **闲聊对话参考**：
{q1}
{q2}
{q3}
5. **参观者的情绪**:
{emotion}
'''      
            # print(prompt)
            all_info.append(prompt)

    # 保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_info, f, ensure_ascii=False, indent=2)

    print(f"已成功生成 {len(all_info)} 条最终信息，保存到 {save_path}")



def out_action_process_daolan(action_out_path, scene_save_path, bot_meta, save_path):
    with open(action_out_path, 'r', encoding='utf-8') as f:
        output_data = json.load(f)
    with open(scene_save_path, 'r', encoding='utf-8') as f:
        scene_ori_data = json.load(f)
    with open(bot_meta, 'r', encoding='utf-8') as f:
        bot_info = json.load(f)

    all_data = []
    for data in output_data:
        # try:
        # 获取person和bot的信息
        bot = next((p for p in bot_info if p["bot_id"] == data["bot_id"]), None)
        # print("bot",bot)
        # 获取scene数据
        scene_data = next((p for p in scene_ori_data if p["scene"] == data["scene"]), None)
        # print("scene_data",scene_data)
        # 检查person, bot, scene_data是否为None
        if  bot is None or scene_data is None:
            print(f"Skipping data for bot_id {data['bot_id']} due to missing person, bot, or scene_data.")
            continue  # 跳过当前数据

        # 获取action_list
        action_list = data["action_list"]
        # print("action_list",action_list)
        # 判断是否有符合跳过条件的action，如果有则跳过整个data
        skip_data = False
        for action in action_list:
            # print(type(action))  # 打印action的类型
            # print(action)
            if not action.get("content"):
                skip_data = True
                break  # 跳出当前循环，跳过整个data

            if (action["role"] == "system.call_response" and 
                "<ret>/api/products/products?category_id=" in action["content"] and
                " | is_complete=yes</ret>" not in action["content"]):
                # 如果符合条件，标记跳过当前data
                print(f"Skipping data for  bot_id {data['bot_id']} due to matching action.")
                skip_data = True
                break  # 跳出当前循环，跳过整个data

        if skip_data:
            continue  # 跳过整个data

        for action in action_list:
            if (action["role"] == "system.call_response") and ("<ret>/api/products/products?category_id=" in action["content"]) and (" | is_complete=yes</ret>" in action["content"]):
                # 提取category_id
                # print(action['content'])
                category_id = action['content'].split('category_id=')[1].split(' | is_')[0]
                # print(category_id)

                # 获取该类目下的商品信息
                if scene_data and "data" in scene_data:
                    for category, info in scene_data["data"].items():
                        # print(category, info)
                        if info['category_id'] == category_id:
                            products = info['description']  # 返回对应类目下的商品信息

                            # 格式化返回数据
                            new_action = {
                                "role": "system.call_response",
                                "call": f"/api/products/products?category_id={category_id}",
                                "return": {
                                    "description": products
                                }
                            }
                            # print(new_action)

                            # 替换原有 action
                            action.update(new_action)
                            # print("action",action)
                            break
            elif (action["role"] == "system.call_response") and ("<ret>/api/robots/available?robots=" in action["content"]) and (" | is_complete=yes</ret>" in action["content"]):
                # 提取category_id
                # print(action['content'])
                category_id = action['content'].split('robots=')[1].split(' | is_')[0]
                # print(category_id)

                # 获取该类目下的商品信息
                if scene_data and "data" in scene_data:
                    for category, info in scene_data["data"].items():
                        # print(category, info)
                        if info['category_id'] == category_id:
                            robots = info['robots']  # 返回对应类目下的商品信息

                            # 格式化返回数据
                            new_action = {
                                "role": "system.call_response",
                                "call": f"/api/robots/available?robots={category_id}",
                                "return": {
                                    "is_complete":"true",
                                    "robots":robots
                                }
                            }
                            # print(new_action)

                            # 格式化返回数据
                            # print(robots)
                #NOTE：下次运行用上面的，还需要把机器人的描述数据补充
                            # ro_id = [k for k in robots[0].keys()][0]
                            # new_action = {
                            #     "role": "system.call_response",
                            #     "call": f"/api/robots/available?robots={category_id}",
                            #     "return": {
                            #         "is_complete":"true",
                            #         "robots":[
                            #             {
                            #                 "id":ro_id,
                            #                 "name":robots[0][ro_id],
                            #                 "function": None
                            #             }
                            #         ]
                            #     }
                            # }
                            # print(new_action)

                            # 替换原有 action
                            action.update(new_action)
                            # print("action",action)
                            break      

            elif (action["role"] == "system.call_response") and ("<ret>/api/robots/dispatch?robot_id=" in action["content"]) and (" | is_complete=yes" in action["content"]):
                # 提取category_id
                # print(action['content'])
                category_id = action['content'].split('robot_id=')[1].split(' | is_')[0]
                # print(category_id)

                # 获取该类目下的商品信息
                if scene_data and "data" in scene_data:
                    for category, info in scene_data["data"].items():

                        if info['category_id'] == category_id:
                            robots = info['robots']  # 返回对应类目下的商品信息
                            # print(category, info,type(info))
                            # print(info['robots'])

                            robot_id = info["robots"][0]["id"]

                            # # 格式化返回数据
                            # new_action = {
                            #     "role": "system.call_response",
                            #     "call": f"/api/robots/available?robot_id={robot_id}"
                            # }
                            # print(new_action)
                            action['content'] = action['content'].replace(category_id, robot_id)
                            # 替换原有 action
                            # action.update(new_action)
                            # action['content'] = f"<tool_call>/api/robots/dispatch?robot_id={robot_id}</tool_call>"

                            # print("action",action)
                            break      
                else:
                    print(f"Error: Missing 'data' key in scene_data for scene {data['scene']}")

            elif (action["role"] == "bot.call") and ("call>/api/robots/dispatch?robot_id=" in action["content"]) :
                # 提取category_id
                # print(action['content'])
                category_id = action['content'].split('robot_id=')[1].split('</tool_call>')[0]
                # print(category_id)

                # 获取该类目下的商品信息
                if scene_data and "data" in scene_data:
                    for category, info in scene_data["data"].items():

                        if info['category_id'] == category_id:
                            robots = info['robots']  # 返回对应类目下的商品信息
                            # print(category, info,type(info))
                            # print(info['robots'])

                            robot_id = info["robots"][0]["id"]

                            action["content"] = f"<tool_call>/api/robots/dispatch?robot_id={robot_id}</tool_call>"
                            # # 格式化返回数据
                            # new_action = {
                            #     "role": "bot.call",
                            #     "call": f"<tool_call>/api/robots/dispatch?robot_id={robot_id}</tool_call>",
                            # }
                            # # print(new_action)

                            # # 替换原有 action
                            # action.update(new_action)
                            # print("action",action)
                            break      
                else:
                    print(f"Error: Missing 'data' key in scene_data for scene {data['scene']}")

        # print("action_list",action_list)
        # 保存数据
        all_data.append({
            "bot": bot,
            "data": action_list
        })

        # except Exception as e:
        #     print(f"Error processing data for bot_id {data['bot_id']}: {e}")
        #     continue  # 跳过当前数据，继续下一个数据

    # 随机打乱所有数据
    random.shuffle(all_data)
    print(f"Total processed data entries: {len(all_data)}")

    # 将处理后的数据保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)


def out_action_process_daolan_new(action_out_path, scene_save_path, bot_meta, save_path):
    with open(action_out_path, 'r', encoding='utf-8') as f:
        output_data = json.load(f)
    with open(scene_save_path, 'r', encoding='utf-8') as f:
        scene_ori_data = json.load(f)
    with open(bot_meta, 'r', encoding='utf-8') as f:
        bot_info = json.load(f)

    all_data = []
    for data in output_data:
        try:
            # 获取person和bot的信息
            bot = next((p for p in bot_info if p["bot_id"] == data["bot_id"]), None)
            # 获取scene数据
            scene_data = next((p for p in scene_ori_data if p["scene"] == data["scene"]), None)
            
            # 检查person, bot, scene_data是否为None
            if bot is None or scene_data is None:
                print(f"Skipping data for bot_id {data['bot_id']} due to missing person, bot, or scene_data.")
                continue  # 跳过当前数据

            # 获取action_list
            action_list = data.get("action_list", [])
            # 判断是否有符合跳过条件的action，如果有则跳过整个data
            skip_data = False
            for action in action_list:
                try:
                    content = action.get("content")  # 使用get避免KeyError
                    role = action.get("role")
                    
                    if role == "system.call_response" and content:
                        if "<ret>/api/products/products?category_id=" in content and " | is_complete=yes</ret>" not in content:
                            # 如果符合条件，标记跳过当前data
                            print(f"Skipping data for bot_id {data['bot_id']} due to matching action.")
                            skip_data = True
                            break  # 跳出当前循环，跳过整个data

                except Exception as e:
                    print(f"Error processing action in data for bot_id {data.get('bot_id', 'Unknown')}: {e}")
                    skip_data = True
                    break  # 跳过整个data

            if skip_data:
                continue  # 跳过整个data

            # 遍历action_list处理每个action
            for action in action_list:
                try:
                    content = action.get("content")  # 使用get避免KeyError
                    role = action.get("role")

                    if role == "system.call_response" and content:
                        if "<ret>/api/products/products?category_id=" in content and " | is_complete=yes</ret>" in content:
                            category_id = content.split('category_id=')[1].split(' | is_')[0]

                            # 获取该类目下的商品信息
                            if scene_data and "data" in scene_data:
                                for category, info in scene_data["data"].items():
                                    if info['category_id'] == category_id:
                                        products = info['description']

                                        new_action = {
                                            "role": "system.call_response",
                                            "call": f"/api/products/products?category_id={category_id}",
                                            "return": {
                                                "description": products
                                            }
                                        }
                                        action.update(new_action)
                                        break

                        elif "<ret>/api/robots/available?robots=" in content and " | is_complete=yes</ret>" in content:
                            category_id = content.split('robots=')[1].split(' | is_')[0]

                            # 获取该类目下的商品信息
                            if scene_data and "data" in scene_data:
                                for category, info in scene_data["data"].items():
                                    if info['category_id'] == category_id:
                                        robots = info['robots']

                                        new_action = {
                                            "role": "system.call_response",
                                            "call": f"/api/robots/available?robots={category_id}",
                                            "return": {
                                                "is_complete": "true",
                                                "robots": robots
                                            }
                                        }
                                        action.update(new_action)
                                        break

                        elif "<ret>/api/robots/dispatch?robot_id=" in content and " | is_complete=yes" in content:
                            category_id = content.split('robot_id=')[1].split(' | is_')[0]

                            # 获取该类目下的商品信息
                            if scene_data and "data" in scene_data:
                                for category, info in scene_data["data"].items():
                                    if info['category_id'] == category_id:
                                        robots = info['robots']
                                        robot_id = info["robots"][0]["id"]
                                        action['content'] = action['content'].replace(category_id, robot_id)
                                        break

                    elif role == "bot.call" and content:
                        if "call>/api/robots/dispatch?robot_id=" in content:
                            category_id = content.split('robot_id=')[1].split('</tool_call>')[0]

                            # 获取该类目下的商品信息
                            if scene_data and "data" in scene_data:
                                for category, info in scene_data["data"].items():
                                    if info['category_id'] == category_id:
                                        robots = info['robots']
                                        robot_id = info["robots"][0]["id"]
                                        action["content"] = f"<tool_call>/api/robots/dispatch?robot_id={robot_id}</tool_call>"
                                        break

                except Exception as e:
                    print(f"Error processing action in data for bot_id {data.get('bot_id', 'Unknown')}: {e}")
                    skip_data = True
                    break  # 跳过整个data

            if skip_data:
                continue  # 跳过整个data

            # 保存数据
            all_data.append({
                "bot": bot,
                "data": action_list
            })

        except Exception as e:
            print(f"Error processing data for bot_id {data.get('bot_id', 'Unknown')}: {e}")
            continue  # 跳过当前数据，继续下一个数据

    # 随机打乱所有数据
    random.shuffle(all_data)
    print(f"Total processed data entries: {len(all_data)}")

    # 将处理后的数据保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)



def process_all_scene_DL(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    category_prefixes = ["cat_", "scene_", "area_", "zone_"]


    new_data = {}
    cat_id = 0
    for item in all_data:
        for name, info in item.items():
            if name not in new_data:
                new_data[name] = {}
                
            new_data[name]["scene_cat_id"] = f"scene_cat_{cat_id}"

            all_area = [area for area in info]
            new_data[name]["areas"] = all_area
            cat_id += 1

    for scene,data  in new_data.items(): 
        for i,area in enumerate(data["areas"]):
            prefix = random.choice(category_prefixes)
            area["category_id"] = f"{prefix}{i}"


    # 将处理后的数据写入到输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
