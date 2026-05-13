import random
import json

# 载入原子集（假设原子集数据已经存储为JSON文件）
with open("/share/project/intern/qt/8w_project/machine_data/input/yuanzi.json", "r", encoding="utf-8") as f:
    yuanzi = json.load(f)
with open ("/share/project/intern/qt/8w_project/machine_data/cir_output/all_tts_id_scene_machine.json",'r',encoding='utf-8') as f:
    scene_data = json.load(f)

product_info = scene_data[0]["scene_info"]
# 商品类目ID
category_ids = [product["category_id"] for product in product_info.values()]


# all_cat_data = [{"id":product["category_id"],"name":product,"description":product["description"]} for product in product_info.values()]

all_cat_data = [{"id":product_info[cat]["category_id"],"name":cat} for cat in product_info.keys()]

# 节点结构和节点动作
node_structure = {
    "START":[("S1",100)],#,("END",5)
    "S1":[("SXL",10),("SYH",10),("SNYH",10),("SDZH",5),("SNSP",10),("SBOT",10),("SSPLM",10),("END",5)],  #("NDSP",100),("XL",10),("YH",10),("NYH",10),("DZH",5),("NSP",10),("BOT",10),("SPLM",10),("END",5)

    #开始的问答，不涉及在当前商品区域前的问答
    "SXL":[("NDSP",100),("SXL",10),("SYH",10),("SNYH",10),("SDZH",5),("SNSP",10),("SBOT",10),("SSPLM",10),("END",5)], 
    "SYH":[("NDSP",100),("SXL",10),("SYH",10),("SNYH",10),("SDZH",5),("SNSP",10),("SBOT",10),("SSPLM",10),("END",5)], 
    "SNYH":[("NDSP",100),("SXL",10),("SYH",10),("SNYH",10),("SDZH",5),("SNSP",10),("SBOT",10),("SSPLM",10),("END",5)], 
    "SDZH":[("NDSP",100),("SXL",10),("SYH",10),("SNYH",10),("SDZH",5),("SNSP",10),("SBOT",10),("SSPLM",10),("END",5)], 
    "SNSP":[("NDSP",100),("SXL",10),("SYH",10),("SNYH",10),("SDZH",5),("SNSP",10),("SBOT",10),("SSPLM",10),("END",5)], 
    "SBOT":[("NDSP",100),("SXL",10),("SYH",10),("SNYH",10),("SDZH",5),("SNSP",10),("SBOT",10),("SSPLM",10),("END",5)], 



    #导航商品
    "NDSP":[("DHCJ",100),("DHCJDH",10)],

    #导航途中用户突然想去另一个展区，直接重新NDSP
    "DHCJDH":[("NDSP",100),("END",5)],

    #导航时的问答，不涉及在当前商品区域前的问答和导航功能
    #导航商品，如果遇到这个，后面随机append，3-5个DHFK，随机插入("XL",10),("YH",1),("NYH",1),("DZH",1),("NSP",1),("BOT",1),("SPLM",1)
    "DHCJ":[("DHJS",100),("END",5)],
    "DHJS":[("TSP",100),("DSP",100),("NDSP",10),("XL",10),("YH",10),("NYH",10),("DZH",10),("NSP",10),("BOT",10),("SPLM",10),("END",10)],


    #进入某个商品区域后的问答
    # "NDSP":[("TSP",20),("DSP",20),("NDSP",10),("XL",10),("YH",1),("NYH",1),("DZH",1),("NSP",1),("BOT",1),("SPLM",1),("END",1)],

    "XL":[("TSP",100),("DSP",100),("NDSP",50),("XL",10),("YH",10),("NYH",10),("DZH",10),("NSP",10),("BOT",10),("SPLM",10),("END",10)],
    "TSP":[("TSP",100),("DSP",100),("NDSP",50),("XL",10),("YH",10),("NYH",10),("DZH",10),("NSP",10),("BOT",10),("SPLM",10),("END",10)],
    "DSP":[("TSP",100),("DSP",100),("NDSP",50),("XL",10),("YH",10),("NYH",10),("DZH",10),("NSP",10),("BOT",10),("SPLM",10),("END",10)],
    "YH":[("TSP",100),("DSP",100),("NDSP",50),("XL",10),("YH",10),("NYH",10),("DZH",10),("NSP",10),("BOT",10),("SPLM",10),("END",10)],
    "NYH":[("TSP",100),("DSP",100),("NDSP",50),("XL",10),("YH",10),("NYH",10),("DZH",10),("NSP",10),("BOT",10),("SPLM",10),("END",10)],
    "DZH":[("TSP",100),("DSP",100),("NDSP",50),("XL",10),("YH",10),("NYH",10),("DZH",10),("NSP",10),("BOT",10),("SPLM",10),("END",10)],
    "NSP":[("TSP",100),("DSP",100),("NDSP",50),("XL",10),("YH",10),("NYH",10),("DZH",10),("NSP",10),("BOT",10),("SPLM",10),("END",10)],
    "SPLM":[("TSP",100),("DSP",100),("NDSP",50),("XL",10),("YH",10),("NYH",10),("DZH",10),("NSP",10),("BOT",10),("SPLM",10),("END",10)],
    "BOT":[("TSP",100),("DSP",100),("NDSP",50),("XL",10),("YH",10),("NYH",10),("DZH",10),("NSP",10),("BOT",10),("SPLM",10),("END",10)],

    
    "END":[]
    # "END":[("ENDCJ",20)],
    # "ENDCJ":[("ENDJS",20)],
    # "ENDJS":[],

}

node_action = {
    "START": [
        "user.speak.DZH",
        "bot.call.LM",
        "bot.speak.DZH"
    ],
    "S1": [
        "system.call_response.LM"
    ],
    "SXL": [
        "user.speak.XL",  
        "bot.speak.XL"  
    ],
    "SYH": [
        "user.speak.YH",  
        "bot.speak.YH" 
    ],
    "SNYH": [
        "user.speak.NYH",  
        "bot.speak.NYH" 
    ],
    "SDZH": [
        "user.speak.DZH",  
        "bot.speak.DZH"  
    ],
    "SNSP": [
        "user.speak.NSP",  
        "bot.speak.NSP" 
    ],
    "SBOT": [
        "user.speak.BOT",  
        "bot.speak.BOT" 
    ],
    "SSPLM": [
        "user.speak.SPLM", 
        "bot.speak.SPLM"  
    ],

    # "DHCJDH": [
    #     "system.call_response.DHCJ",  
    #     "bot.speak.DHCJ" 
    # ],

    

    "XL": [
        "user.speak.XL",  
        "bot.speak.XL"  
    ],
    "TSP": [
        "user.speak.TSP",  
        "bot.speak.TSP" 
    ],
    "DSP": [
        "user.speak.DSP",  
        "bot.speak.DSP" 
    ],

    "YH": [
        "user.speak.YH",  
        "bot.speak.YH" 
    ],
    "NYH": [
        "user.speak.NYH",  
        "bot.speak.NYH" 
    ],
    "DZH": [
        "user.speak.DZH",  
        "bot.speak.DZH"  
    ],
    "NSP": [
        "user.speak.NSP",  
        "bot.speak.NSP" 
    ],
    "BOT": [
        "user.speak.BOT",  
        "bot.speak.BOT" 
    ],
    "SPLM": [
        "user.speak.SPLM", 
        "bot.speak.SPLM"  
    ],

    "NDSP": [
        "user.speak.NDSP",  
        "bot.call.DH" 
    ],
    "DHCJ": [
        "system.call_response.DHCJ",  
        "bot.speak.DHCJ" 
    ],
    "DHFK": [
        "system.call_response.DHFK",  
    ],
    "DHJS": [
        "system.call_response.DHJS",  
        "bot.call.SP",
        "system.call_response.SP",
        "bot.speak.DHJS"
    ],


    "END": [
        "user.speak.END", 
        "bot.call.END"
    ],
    "ENDCJ": [
        "system.call_response.ENDCJ",  
        "bot.speak.END", 
    ],
    "ENDFK": [
        "system.call_response.ENDFK",  
    ],
    "ENDJS": [
        "system.call_response.ENDJS",
        "bot.speak.HOME"
    ]
    
}
# 上次选择的类目ID（用于避免重复）
Now_cat_id = None
Want_cat_id = None

# 从商品信息中选择一个类目ID，避免重复
def get_new_category_id(Now_cat_id):
    available_category_ids = [cid for cid in category_ids if cid != Now_cat_id]
    chosen_category_id = random.choice(available_category_ids)
    return chosen_category_id


import copy


def build_action_sequence(node_name, Now_cat_id,Want_cat_id):
    action_sequence = []

    print("_",node_name)
    #导航商品，如果遇到这个，后面随机append，3-5个DHFK，随机插入("XL",10),("YH",1),("NYH",1),("DZH",1),("NSP",1),("BOT",1),("SPLM",1)
    if node_name == "DHJS":
        for i in range(random.randint(1,3)):
            action_sequence.append(yuanzi["system.call_response"]["DHFK"]["action"])

        can_yuanzi = ["XL","YH","NYH","DZH","NSP","BOT"]  #随机插入一对问答
        random_index_Q = random.randint(0, len(action_sequence)-1)  # 生成一个从 0 到 len(my_list) 的随机索引
        random_index_A = random_index_Q+random.randint(1,2) # 生成一个从 0 到 len(my_list) 的随机索引

        yuanzi_insert = random.choice(can_yuanzi)
    
        action_sequence.insert(random_index_Q, yuanzi["user.speak"][yuanzi_insert]["action"])
        action_sequence.insert(random_index_A, yuanzi["bot.speak"][yuanzi_insert]["action"])

        #导航商品，如果遇到这个，后面随机append，3-5个DHFK，随机插入("XL",10),("YH",1),("NYH",1),("DZH",1),("NSP",1),("BOT",1),("SPLM",1)
    if node_name == "ENDJS":
        for i in range(random.randint(1,2)):
            action_sequence.append(yuanzi["system.call_response"]["ENDFK"]["action"])

    if node_name == "DHCJDH":
        for i in range(random.randint(1,2)):
            action_sequence.append(yuanzi["system.call_response"]["DHFK"]["action"])

        can_yuanzi = ["XL","YH","NYH","DZH","NSP","BOT"]  #随机插入一对问答
        random_index_Q = random.randint(0, len(action_sequence)//2)  # 生成一个从 0 到 len(my_list) 的随机索引
        random_index_A = random.randint(len(action_sequence)//2,len(action_sequence))  # 生成一个从 0 到 len(my_list) 的随机索引
        yuanzi_insert = random.choice(can_yuanzi)
    
        action_sequence.insert(random_index_Q, yuanzi["user.speak"][yuanzi_insert]["action"])
        action_sequence.insert(random_index_A, yuanzi["bot.speak"][yuanzi_insert]["action"])


    else:
        for action_key in node_action[node_name]:
            if action_key.startswith("user.speak"):
                # 获取用户说话的原子动作，并替换内容
                action_type = action_key.split("user.speak.")[1]  
                action = yuanzi["user.speak"][action_type]  
                action["action"]["node"] = node_name

                action_sequence.append(action["action"])  

            elif action_key.startswith("bot.speak"):
                # 获取机器人说话的原子动作，并替换内容
                action_type = action_key.split("bot.speak.")[1]  
                action = yuanzi["bot.speak"][action_type]  
                action["action"]["node"] = node_name

                action_sequence.append(action["action"])  

            elif action_key.startswith("system.call_response"):
                # 获取系统反馈的原子动作，并替换内容
                action_type = action_key.split("system.call_response.")[1]  
                action = yuanzi["system.call_response"][action_type]  
                temple = copy.deepcopy(action)
                action["action"]["node"] = node_name


                if "status=arrived," in action["action"]["content"]:
                    # 替换类别ID
                    temple["action"]["content"] = temple["action"]["content"].replace("status=arrived,", f"status=arrived_{Want_cat_id},")
                    Now_cat_id = Want_cat_id

                if "categories | is_complete=yes" in action["action"]["content"]:
                    temple["action"]["return"] = all_cat_data

                if "category_id=%%商品的类目编号%%" in action["action"]["content"]:
                    temple["action"]["content"] = temple["action"]["content"].replace("category_id=%%商品的类目编号%%", f"category_id={Now_cat_id},")

                
                if "cid" in action["action"]["content"]:
                    # 替换类别ID
                    category_id = Want_cat_id
                    temple["action"]["content"] = temple["action"]["content"].replace("%%商品的类目编号%%", category_id)
                action_sequence.append(temple["action"])  

            elif action_key.startswith("bot.call"):
                # 获取机器人调用API的原子动作，并替换内容
                action_type = action_key.split("bot.call.")[1]  
                action = yuanzi["bot.call"][action_type]  
                action["action"]["node"] = node_name
                temple = copy.deepcopy(action)

                if "cid" in action["action"]["content"]:
                    # 替换类别ID
                    category_id = get_new_category_id(Now_cat_id)
                    temple["action"]["content"] = temple["action"]["content"].replace("%%商品的类目编号%%", category_id)
                    Want_cat_id = category_id

                if "category_id" in action["action"]["content"]:
                    # 替换类别ID
                    category_id = Now_cat_id
                    temple["action"]["content"] = temple["action"]["content"].replace("%%商品的类目编号%%", category_id)
                    Want_cat_id = category_id
                action_sequence.append(temple["action"])


    return action_sequence, Now_cat_id,Want_cat_id


def traverse_tree_randomly(node_name, depth=0, max_depth=15, Now_cat_id=None,Want_cat_id = None):
    if depth >= max_depth:
        action_s = []
        for end_node in ['END', 'ENDCJ', 'ENDJS']:
            action_sequence, Now_cat_id ,Want_cat_id= build_action_sequence(end_node, Now_cat_id ,Want_cat_id)
            action_s+=action_sequence
        return action_s, Now_cat_id,Want_cat_id
    
    # 获取当前节点的动作序列，并根据需要填充
    action_sequence, Now_cat_id,Want_cat_id = build_action_sequence(node_name, Now_cat_id,Want_cat_id)
    
    # 如果是END节点，结束
    if node_name == "END":
        action_s = []
        for end_node in ['END', 'ENDCJ', 'ENDJS']:
            action_sequence, Now_cat_id ,Want_cat_id= build_action_sequence(end_node, Now_cat_id,Want_cat_id)
            action_s+=action_sequence
        return action_s, Now_cat_id,Want_cat_id
    
    # 获取当前节点的所有可能下一节点和它们的概率
    next_nodes = node_structure.get(node_name, [])
    next_node = random.choices([node[0] for node in next_nodes], [node[1] for node in next_nodes])[0]
    
    # 递归调用继续遍历，深度加1
    return action_sequence + traverse_tree_randomly(next_node, depth + 1, max_depth, Now_cat_id,Want_cat_id)[0], Now_cat_id,Want_cat_id



# 从START节点开始
final_action_sequence, _ ,_= traverse_tree_randomly("START", Now_cat_id=Now_cat_id,Want_cat_id = Want_cat_id),Now_cat_id,Want_cat_id

output_file_path = "/share/project/intern/qt/8w_project/machine_data/input/action.json"

with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(final_action_sequence, f, ensure_ascii=False, indent=4)

print(f"动作序列已成功写入 {output_file_path}")