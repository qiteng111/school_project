import datetime
import os
from volcenginesdkarkruntime import Ark
import json

api_key = 'a54e163a-5138-4ac3-89c9-07f15ad55fac'
model = "ep-20250707172605-cvtgp"


client = Ark(
    api_key=api_key,
    )

with open('person_sum_pro_new.json','r',encoding='utf-8') as f:
    all_data = json.load(f)

person_info = all_data[0]

with open('cov_filtered.json','r',encoding='utf-8') as f:
    dialogue_data = json.load(f)

split_dialogues = [
    dialogue_data[i * 20: (i + 1) * 20] for i in range(5)
]#切片


def process_one_person(person_info, dialogues, person_idx):
    """为单个人物创建上下文并处理分配的对话"""

    results = []
    try:
        response = client.context.create(
            model=model,
            mode="common_prefix",
            messages=[
                {"role": "system", "content": (
                    "你是一个对话数据增强专家，擅长根据用户画像生成个性化多轮对话。\n\n"
                    "当前输入数据包含：\n"
                    "1.**用户信息**：包括用户姓名（person）、用户与 AI 多次历史对话总结（person_summary）、"
                    "详细用户画像（person_profile，含多维度性格特质及推理说明）。\n"
                    "2.**多轮对话**：（一个用户提问 `user` 和 AI 回复 `bot` 的列表）。\n\n"
                    "你的任务：\n"
                    "1. 修改多轮对话中 **bot 的回答**，让回答风格体现出 AI 对用户有深入了解：\n"
                    "- 回答要结合用户的兴趣、性格、语言风格，让用户感到“AI 认识自己”。\n"
                    "- 适当在 bot 的回复中结合用户过往的经历、兴趣、习惯（来自 person_summary 和 person_profile）。\n"
                    "2. 如果某条多轮对话 **轮数 < 3**（不足 3 轮的 user-bot 配对），请补充完整至少 3 轮对话：\n"
                    "- 新增的问题应基于上下文自然衔接（让用户顺着之前的话题继续提问）。\n"
                    "- 新增的 bot 回复仍要符合用户画像要求。\n"
                    "3. 每一轮对话必须严格遵循：**1 user 提问 → 1 bot 回复**，不要生成连续的 user 或 bot。\n"
                    "4. 输出完整修改后的数据，包括用户信息和对话。使用 JSON 格式，结构如下：\n\n"
                    "[\n"
                    "  {\n"
                    "    \"person\": \"用户姓名\",\n"
                    "    \"person_summary\": \"这里填写用户与AI多轮对话的总结，每条用\\n分隔。\",\n"
                    "    \"person_profile\": \"这里填写用户画像，包括多维度性格特质及推理说明。\"\n"
                    "  },\n"
                    "  {\n"
                    "    \"user\": \"用户提问1\",\n"
                    "    \"bot\": \"修改后的AI回复1，符合用户画像。\"\n"
                    "  },\n"
                    "  {\n"
                    "    \"user\": \"用户提问2\",\n"
                    "    \"bot\": \"修改后的AI回复2，符合用户画像。\"\n"
                    "  },\n"
                    "  {\n"
                    "    \"user\": \"用户提问3\",\n"
                    "    \"bot\": \"修改后的AI回复3，符合用户画像。\"\n"
                    "  }\n"
                    "  // ...(如有更多对话继续追加)\n"
                    "]\n\n"
                    "用户信息：\n"+str(person_info)
                )}
            ],
            ttl=datetime.timedelta(minutes=600),
        )
        context_id = response.id
        print(f"[人物{person_idx}] 缓存上下文成功: {context_id}")

        #分配的20条对话
        for idx, dialogue in enumerate(dialogues):
            chat_response = client.context.completions.create(
                context_id=context_id,
                model=model,
                messages=[
                    {"role": "user", "content": f"对话数据：\n{json.dumps(dialogue, ensure_ascii=False)}"}
                ],
                stream=False,
            )
            result = {
                "person": person_info['person'],
                "dialogue_index": idx + person_idx * 20,
                "input": dialogue,
                "output": chat_response.choices[0].message.content
            }
            results.append(result)
            print(f"[人物{person_idx}] 第{idx+1}条对话处理完成")

    except Exception as e:
        print(f"[人物{person_idx}] 出错：{e}")

    return results


response = client.context.create(
    model=model,
    mode="common_prefix",  # 缓存前缀
    messages=[
            {
                "role": "system",
                "content": (
                    "你是一个对话数据增强专家，擅长根据用户画像生成个性化多轮对话。\n\n"
                    "当前输入数据包含：\n"
                    "1. **用户信息**：包括用户姓名（person）、用户与 AI 多次历史对话总结（person_summary）、"
                    "详细用户画像（person_profile，含多维度性格特质及推理说明）。\n"
                    "2. **多轮对话**：（一个用户提问 `user` 和 AI 回复 `bot` 的列表）。\n\n"
                    "你的任务：\n"
                    "1. 修改多轮对话中 **bot 的回答**，让回答风格体现出 AI 对用户有深入了解：\n"
                    "- 回答要结合用户的兴趣、性格、语言风格，让用户感到“AI 认识自己”。\n"
                    "- 适当在 bot 的回复中结合用户过往的经历、兴趣、习惯（来自 person_summary 和 person_profile）。\n"
                    "2. 如果某条多轮对话 **轮数 < 3**（不足 3 轮的 user-bot 配对），请补充完整至少 3 轮对话：\n"
                    "- 新增的问题应基于上下文自然衔接（让用户顺着之前的话题继续提问）。\n"
                    "- 新增的 bot 回复仍要符合用户画像要求。\n"
                    "3. 每一轮对话必须严格遵循：**1 user 提问 → 1 bot 回复**，不要生成连续的 user 或 bot。\n"
                    "4. 输出完整修改后的数据，包括用户信息和对话。**必须严格使用 JSON 格式输出**，且不要在 JSON 前后添加任何解释或额外文字，结构示例如下：\n\n"
                    "{\n"
                    "  \"person\": \"用户姓名\",\n"
                    "  \"person_summary\": \"这里填写用户与AI多轮对话的总结，每条用\\n分隔。\",\n"
                    "  \"person_profile\": \"这里填写用户画像，包括多维度性格特质及推理说明，格式与示例一致。\",\n"
                    "  \"dialogues\": [\n"
                    "    {\"user\": \"用户提问1\", \"bot\": \"修改后的AI回复1，符合用户画像。\"},\n"
                    "    {\"user\": \"用户提问2\", \"bot\": \"修改后的AI回复2，符合用户画像。\"},\n"
                    "    {\"user\": \"用户提问3\", \"bot\": \"修改后的AI回复3，符合用户画像。\"}\n"
                    "    // ...(如有更多对话继续追加)\n"
                    "  ]\n"
                    "}\n\n"
                    f"用户信息：\n{json.dumps(person_info, ensure_ascii=False)}"
                )
            }
        ],

    ttl=datetime.timedelta(minutes=600),  # 前缀缓存有效期
)

print(response)

# 第1轮对话

chat_response = client.context.completions.create(
    context_id=response.id,  # 用前面缓存的 context_id
    model=model,
    messages=[
        {"role": "user", "content": ("对话数据：\n"+str(dialogue_data[42]))},  
    ],
    stream=False,
)


# 输出修改后的 bot 回复
print(chat_response.choices[0].message.content)