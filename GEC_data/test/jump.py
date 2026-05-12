import random
import json
import copy

# 载入原子集（假设原子集数据已经存储为JSON文件）
with open("/mnt/common/intern/qt/8w_project/machine_data/input/yuanzi.json", "r", encoding="utf-8") as f:
    yuanzi = json.load(f)
with open("/mnt/common/intern/qt/8w_project/machine_data/cir_output/all_tts_id_scene_machine_crib_descrb_new.json", 'r', encoding='utf-8') as f:
    scene_data = json.load(f)


print(len(scene_data))
# 节点结构和节点动作
node_structure = {
    "START":[("S1",500)],#,("END",3)
    "S1":[("SNDSP",80),("SXL",30),("SMXL",30),("SYH",30),("SNDD",30),("SYNYH",30),("SNYH",30),("SDZH",30),("SDZ",30),("SNSP",30),("SBOT",30),("SSPLM",30),("SEND",1)],  #("NDSP",100),("XL",10),("YH",10),("NYH",10),("DZH",5),("NSP",10),("BOT",10),("SPLM",10),("END",3)

    #开始的问答，不涉及在当前商品区域前的问答
    "SXL":[("SNDSP",100),("SXL",30),("SMXL",30),("SNDD",30),("SYH",30),("SYNYH",30),("SNYH",30),("SDZH",30),("SDZ",30),("SNSP",30),("SBOT",30),("SSPLM",30),("SEND",1)],
    "SMXL":[("SNDSP",100),("SXL",30),("SMXL",30),("SNDD",30),("SYH",30),("SYNYH",30),("SNYH",30),("SDZH",30),("SDZ",30),("SNSP",30),("SBOT",30),("SSPLM",30),("SEND",1)],  
    "SYH":[("SNDSP",100),("SXL",30),("SMXL",30),("SNDD",30),("SYH",30),("SYNYH",30),("SNYH",30),("SDZH",30),("SDZ",30),("SNSP",30),("SBOT",30),("SSPLM",30),("SEND",1)], 
    "SYNYH":[("SNDSP",100),("SXL",30),("SMXL",30),("SNDD",30),("SYH",10),("SYNYH",10),("SNYH",30),("SDZH",30),("SDZ",30),("SNSP",30),("SBOT",30),("SSPLM",30),("SEND",1)], 
    "SNYH":[("SNDSP",100),("SXL",30),("SMXL",30),("SNDD",30),("SYH",30),("SYNYH",30),("SNYH",10),("SDZH",30),("SDZ",30),("SNSP",30),("SBOT",30),("SSPLM",30),("SEND",1)], 
    "SDZH":[("SNDSP",100),("SXL",30),("SMXL",30),("SNDD",30),("SYH",30),("SYNYH",30),("SNYH",30),("SDZH",10),("SDZ",30),("SNSP",30),("SBOT",30),("SSPLM",30),("SEND",1)], 
    "SNSP":[("SNDSP",100),("SXL",30),("SMXL",30),("SNDD",30),("SYH",30),("SYNYH",30),("SNYH",30),("SDZH",30),("SDZ",30),("SNSP",10),("SBOT",30),("SSPLM",30),("SEND",1)], 
    "SNDD":[("SNDSP",100),("SXL",30),("SMXL",30),("SNDD",30),("SYH",30),("SYNYH",30),("SNYH",30),("SDZH",30),("SDZ",30),("SNSP",10),("SBOT",30),("SSPLM",30),("SEND",1)], 
    "SDZ":[("SNDSP",100),("SXL",30),("SMXL",30),("SNDD",30),("SYH",30),("SYNYH",30),("SNYH",30),("SDZH",10),("SDZ",30),("SNSP",30),("SBOT",30),("SSPLM",30),("SEND",1)], 

    "SBOT":[("SNDSP",100),("SXL",30),("SMXL",30),("SNDD",30),("SYH",30),("SYNYH",30),("SNYH",30),("SDZH",30),("SDZ",30),("SNSP",30),("SBOT",10),("SSPLM",30),("SEND",1)], 
    "SSPLM":[("SNDSP",100),("SXL",30),("SMXL",30),("SNDD",30),("SYH",30),("SYNYH",30),("SNYH",30),("SDZH",30),("SDZ",30),("SNSP",30),("SBOT",30),("SSPLM",10),("SEND",1)], 
    "SNNDSP":[("SNDSP",100),("SXL",30),("SMXL",30),("SNDD",30),("SYH",30),("SYNYH",30),("SNYH",30),("SDZH",30),("SDZ",30),("SNSP",30),("SBOT",30),("SSPLM",10),("SEND",1)], 


    "SNDSP":[("SNNDSP",5),("SYNDSP",100)],
    "SYNDSP":[("SDHCJ",100)],
    "SDHCJ":[("SDHFK",50)],
    "SDHFK":[("SDHCJDH",5),("DHJS",100),("END",3)],
    #导航途中用户突然想去另一个展区，直接重新SNDSP
    "SDHCJDH":[("SNDSP",100),("END",1)],



    #导航商品
    "NDSP":[("NNDSP",5),("YNDSP",100)],
    "YNDSP":[("DHCJ",100)],
    "DHCJ":[("DHFK",50)],
    "DHFK":[("DHCJDH",10),("DHJS",100),("END",3)],

    #导航途中用户突然想去另一个展区，直接重新NDSP
    "DHCJDH":[("NDSP",100),("END",3)],
    #导航时的问答，不涉及在当前商品区域前的问答和导航功能
    #导航商品，如果遇到这个，后面随机append，3-5个DHFK，随机插入("XL",10),("YH",1),("NYH",1),("DZH",1),("NSP",1),("BOT",1),("SPLM",1)
    "DHJS":[("DSP",100),("MSP",100),("NDSP",10),("DSPLM",100),("XL",10),("YH",10),("YNYH",10),("DZ",10),("NYH",10),("DZH",5),("NSP",10),("BOT",10),("SPLM",10),("END",3)],


    #不走继续问
    "NNDSP":[("TSP",100),("MSP",100),("SPKS",50),("NSPGN",50),("DSP",100),("JD",50),("NDSP",100),("XL",5),("MXL",5),("YH",10),("DZ",10),("YNYH",10),("NYH",10),("DZH",5),("NSP",10),("BOT",10),("SPLM",10),("END",3)],
    #进入某个商品区域后的问答
    "MXL":[("TSP",150),("MSP",100),("SPKS",50),("PY",30),("SSSJ",30),("NSPGN",50),("NDD",50),("JSSP",100),("DSP",100),("NDSP",100),("DSPLM",30),("DZ",10),("XL",10),("MXL",10),("YH",10),("YNYH",10),("NYH",10),("DZH",5),("NSP",10),("BOT",10),("SPLM",10),("END",3)],
    "XL":[("TSP",150),("MSP",100),("SPKS",50),("PY",30),("SSSJ",30),("NSPGN",50),("NDD",50),("JSSP",100),("DSP",100),("NDSP",100),("DSPLM",30),("DZ",10),("XL",10),("MXL",10),("YH",10),("YNYH",10),("NYH",10),("DZH",5),("NSP",10),("BOT",10),("SPLM",10),("END",3)],
    "SPKS":[("TSP",150),("MSP",100),("SPKS",50),("PY",30),("SSSJ",30),("NSPGN",50),("NDD",50),("JSSP",100),("DSP",100),("NDSP",100),("DSPLM",30),("DZ",10),("XL",10),("MXL",10),("YH",10),("YNYH",10),("NYH",10),("DZH",5),("NSP",10),("BOT",10),("SPLM",10),("END",3)],
    "NSPGN":[("TSP",150),("MSP",100),("SPKS",50),("PY",30),("SSSJ",30),("NSPGN",50),("NDD",50),("JSSP",100),("DSP",100),("NDSP",100),("DSPLM",30),("DZ",10),("XL",10),("MXL",10),("YH",10),("YNYH",10),("NYH",10),("DZH",5),("NSP",10),("BOT",10),("SPLM",10),("END",3)],
    "NDD":[("TSP",150),("MSP",100),("SPKS",50),("PY",30),("SSSJ",30),("NSPGN",50),("NDD",50),("JSSP",100),("DSP",100),("NDSP",100),("DSPLM",30),("DZ",10),("XL",10),("MXL",10),("YH",10),("YNYH",10),("NYH",10),("DZH",5),("NSP",10),("BOT",10),("SPLM",10),("END",3)],
    "PY":[("TSP",150),("MSP",100),("SPKS",50),("NSPGN",50),("SSSJ",30),("NDD",50),("JSSP",100),("GMSP",100),("DSP",100),("JD",50),("NDSP",100),("DSPLM",30),("XL",10),("DZ",10),("MXL",10),("YH",10),("YNYH",10),("NYH",10),("DZH",5),("NSP",10),("BOT",10),("SPLM",10),("END",3)],
    "SSSJ":[("TSP",150),("MSP",100),("SPKS",50),("PY",30),("NSPGN",50),("NDD",50),("JSSP",100),("DSP",100),("GMSP",100),("JD",50),("NDSP",100),("DSPLM",30),("XL",10),("DZ",10),("MXL",10),("YH",10),("YNYH",10),("NYH",10),("DZH",5),("NSP",10),("BOT",10),("SPLM",10),("END",3)],

    "TSP":[("TSP",150),("MSP",100),("SPKS",50),("PY",30),("SSSJ",30),("NSPGN",50),("NDD",50),("JSSP",100),("GMSP",100),("DSP",100),("JD",50),("NDSP",100),("DSPLM",30),("DZ",10),("XL",10),("MXL",10),("YH",10),("YNYH",10),("NYH",10),("DZH",5),("NSP",10),("BOT",10),("SPLM",10),("END",3)],
    "DSP":[("TSP",150),("MSP",100),("SPKS",50),("PY",100),("SSSJ",30),("NSPGN",50),("NDD",50),("JSSP",100),("GMSP",100),("DSP",100),("JD",50),("NDSP",100),("DSPLM",30),("DZ",10),("XL",10),("MXL",10),("YH",10),("YNYH",10),("NYH",10),("DZH",5),("NSP",10),("BOT",10),("SPLM",10),("END",3)],
    "MSP":[("TSP",150),("MSP",100),("SPKS",50),("PY",100),("SSSJ",30),("NSPGN",50),("NDD",50),("JSSP",100),("GMSP",100),("DSP",100),("JD",50),("NDSP",100),("DSPLM",30),("DZ",10),("XL",10),("MXL",10),("YH",10),("YNYH",10),("NYH",10),("DZH",5),("NSP",10),("BOT",10),("SPLM",10),("END",3)],
    
    "GMSP":[("TSP",150),("MSP",100),("SPKS",50),("PY",100),("SSSJ",30),("NSPGN",50),("NDD",50),("JSSP",100),("DSP",100),("JD",50),("NDSP",100),("DSPLM",30),("DZ",10),("XL",10),("MXL",10),("YH",10),("YNYH",10),("NYH",10),("DZH",5),("NSP",10),("BOT",10),("SPLM",10),("END",3)],
    
    "JD":[("TSP",150),("MSP",100),("SPKS",50),("PY",30),("SSSJ",30),("NSPGN",50),("NDD",50),("DSP",100),("JSSP",100),("JD",50),("NDSP",100),("DSPLM",30),("DZ",10),("XL",10),("MXL",10),("YH",10),("YNYH",10),("NYH",10),("DZH",5),("NSP",10),("BOT",10),("SPLM",10),("END",3)],

    
    "YH":[("TSP",150),("MSP",100),("SPKS",50),("PY",30),("SSSJ",30),("NSPGN",50),("NDD",50),("JSSP",100),("DSP",100),("NDSP",100),("DSPLM",30),("DZ",10),("XL",10),("MXL",10),("YH",1),("YNYH",1),("NYH",10),("DZH",5),("NSP",10),("BOT",10),("SPLM",10),("END",3)],
    "YNYH":[("TSP",150),("MSP",100),("SPKS",50),("PY",30),("SSSJ",30),("NSPGN",50),("NDD",50),("JSSP",100),("DSP",100),("NDSP",100),("DSPLM",30),("DZ",10),("XL",10),("MXL",10),("YH",1),("YNYH",1),("NYH",10),("DZH",5),("NSP",10),("BOT",10),("SPLM",10),("END",3)],
    "NYH":[("TSP",150),("MSP",100),("SPKS",50),("PY",30),("SSSJ",30),("NSPGN",50),("NDD",50),("JSSP",100),("DSP",100),("NDSP",100),("DSPLM",30),("DZ",10),("XL",10),("MXL",10),("YH",10),("YNYH",10),("NYH",10),("DZH",5),("NSP",10),("BOT",10),("SPLM",10),("END",3)],
    "DZH":[("TSP",150),("MSP",100),("SPKS",50),("PY",30),("SSSJ",30),("NSPGN",50),("NDD",50),("JSSP",100),("DSP",100),("NDSP",100),("DSPLM",30),("DZ",10),("XL",10),("MXL",10),("YH",10),("YNYH",10),("NYH",10),("DZH",5),("NSP",10),("BOT",10),("SPLM",10),("END",3)],
    "NSP":[("TSP",150),("MSP",100),("SPKS",50),("PY",30),("SSSJ",30),("NSPGN",50),("NDD",50),("JSSP",100),("DSP",100),("NDSP",100),("DSPLM",30),("DZ",10),("XL",10),("MXL",10),("YH",10),("YNYH",10),("NYH",10),("DZH",5),("NSP",10),("BOT",10),("SPLM",10),("END",3)],
    "SPLM":[("TSP",150),("MSP",100),("SPKS",50),("PY",30),("SSSJ",30),("NSPGN",50),("NDD",50),("JSSP",100),("DSP",100),("NDSP",100),("DSPLM",30),("DZ",10),("XL",10),("MXL",10),("YH",10),("YNYH",10),("NYH",10),("DZH",5),("NSP",10),("BOT",10),("SPLM",10),("END",3)],
    "DSPLM":[("TSP",150),("MSP",100),("SPKS",50),("PY",30),("SSSJ",30),("NSPGN",50),("NDD",50),("JSSP",100),("DSP",100),("NDSP",100),("DSPLM",30),("DZ",10),("XL",10),("MXL",10),("YH",10),("YNYH",10),("NYH",10),("DZH",5),("NSP",10),("BOT",10),("SPLM",10),("END",3)],
    "BOT":[("TSP",150),("MSP",100),("SPKS",50),("PY",30),("SSSJ",30),("NSPGN",50),("NDD",50),("JSSP",100),("DSP",100),("NDSP",100),("DSPLM",30),("DZ",10),("XL",10),("MXL",10),("YH",10),("YNYH",10),("NYH",10),("DZH",5),("NSP",10),("BOT",10),("SPLM",10),("END",3)],
    
    "JSSP":[("TSP",150),("MSP",100),("SPKS",50),("PY",30),("SSSJ",30),("NSPGN",50),("NDD",50),("DSP",100),("JSSP",100),("JD",50),("GMSP",100),("NDSP",100),("DSPLM",30),("DZ",10),("XL",10),("MXL",10),("YH",10),("YNYH",10),("NYH",10),("DZH",5),("NSP",10),("BOT",10),("SPLM",10),("END",3)],
    
    "DZ":[("TSP",150),("MSP",100),("SPKS",50),("PY",30),("SSSJ",30),("NSPGN",50),("NDD",50),("DSP",100),("JSSP",100),("NDSP",100),("DSPLM",30),("DZ",10),("XL",10),("MXL",10),("YH",10),("YNYH",10),("NYH",10),("DZH",5),("NSP",10),("BOT",10),("SPLM",10),("END",3)],

    
    "END":[],
    "SEND":[("temple",1)],

    # "END":[("ENDCJ",20)],
    # "ENDCJ":[("ENDJS",20)],
    # "ENDJS":[],
}

node_action = {
    "START": [
        "bot.speak.DZH",
        "bot.call.LM",
        # "user.speak.DZH",
        # "bot.speak.SDZH"
    ],

    
    "S1": [
        "system.call_response.LM"
    ],
    "SXL": [
        "user.speak.XL",  
        "bot.speak.XL"  
    ],
    "SMXL": [
        "user.speak.MXL1",  
        "bot.speak.MXL1",
        "user.speak.MXL2",  
        "bot.speak.MXL2",
        "user.speak.MXL3",  
        "bot.speak.MXL3"  
    ],
    "SYH": [
        "user.speak.YH",  
        "bot.speak.YH" 
    ],
    "SYNYH": [
        "user.speak.YNYH",  
        "bot.speak.YNYH" 
    ],
    "SNYH": [
        "user.speak.NYH",  
        "bot.speak.NYH" 
    ],
    "SDZH": [
        "user.speak.DZH",  
        "bot.speak.DZH"  
    ],
    "SDZ": [
        "user.speak.DZ",  
        "bot.speak.DZ"  
    ],
    "SNDD": [
        "user.speak.NDD",  
        "bot.speak.NDD"  
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

    "DHCJDH": [
        # "system.call_response.DHCJ",  
        # "bot.speak.DHCJ" 
    ],

    "SDHCJDH": [
        # "system.call_response.DHCJ",  
        # "bot.speak.DHCJ" 
    ],

    "XL": [
        "user.speak.XL",  
        "bot.speak.XL"  
    ],
    "JD": [
        "user.speak.JD",  
        "bot.speak.JD"  
    ],
    "MXL": [
        "user.speak.MXL1",  
        "bot.speak.MXL1",
        "user.speak.MXL2",  
        "bot.speak.MXL2",
        "user.speak.MXL3",  
        "bot.speak.MXL3"  
    ],
    "DZ": [
        "user.speak.DZ",  
        "bot.speak.DZ"  
    ],
    "GMSP": [
        "user.speak.GMSP",  
        "bot.speak.GMSP"  
    ],
    "TSP": [
        "user.speak.TSP",  
        "bot.speak.TSP" 
    ],
    "DSP": [
        "user.speak.DSP",  
        "bot.speak.DSP" 
    ],
    "MSP": [
        "user.speak.MSP",  
        "bot.speak.MSP" 
    ],
    "PY": [
        "user.speak.PY",  
        "bot.speak.PY" 
    ],
    "YH": [
        "user.speak.YH",  
        "bot.speak.YH" 
    ],
    "YNYH": [
        "user.speak.YNYH",  
        "bot.speak.YNYH" 
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
    "DSPLM": [
        "user.speak.DSPLM", 
        "bot.speak.DSPLM"  
    ],
    "NSPGN": [
        "user.speak.NSPGN", 
        "bot.speak.NSPGN"  
    ],
    "NDD": [
        "user.speak.NDD", 
        "bot.speak.NDD"  
    ],
    "SPKS": [
        "user.speak.SPKS", 
        "bot.speak.SPKS"  
    ],
    "DSPLM": [
        "user.speak.DSPLM", 
        "bot.speak.DSPLM"  
    ],
    "JSSP": [
        "user.speak.JSSP", 
        "bot.speak.JSSP"  
    ],
    "SSSJ": [
        "user.speak.SSSJ", 
        "bot.speak.SSSJ"  
    ],
    "SNDSP": [
        "user.speak.NDSP",
        "bot.speak.YNNDSP"
    ],
    "SYNDSP": [
        "user.speak.YNDSP",
        "bot.call.DH" 
    ],
    "SNNDSP": [
        "user.speak.NNDSP",
        "bot.speak.NNDSP",
    ],
    "SDHCJ": [
        "system.call_response.DHCJ",  
        "bot.speak.DHCJ" 
    ],
    "SDHFK": [
        "system.call_response.DHFK",  
    ],
    "DHJS": [
        "system.call_response.DHJS",  
        "bot.call.SP",
        "system.call_response.SP",
        "bot.speak.DHJS"
    ],


    "NDSP": [
        "user.speak.NDSP",
        "bot.speak.YNNDSP"
    ],
    "YNDSP": [
        "user.speak.YNDSP",
        "bot.call.DH" 
    ],
    "NNDSP": [
        "user.speak.NNDSP",
        "bot.speak.NNDSP",
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


    "SEND": [
        "user.speak.END", 
        "bot.speak.END"
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

def build_action_sequence(node_name, Now_cat_id, Want_cat_id):
    action_sequence = []
    # print("build_node_name",node_name)
    # 根据节点类型，处理动作（包含导航和其他类型）
    if node_name == "DHFK":
        print(len(action_sequence))
        for i in range(random.randint(1, 5)):
            action_sequence.append(yuanzi["system.call_response"]["DHFK"]["action"])

        print(len(action_sequence))
        # 插入 0 到 2 个问答
        num_of_qa = random.randint(0, 1)  # 控制插入 0 到 2 个问答
        can_yuanzi = ["XL", "YH", "NYH", "NSP", "BOT", "SPLM","NDD"]

        for _ in range(num_of_qa):  # 循环插入问答
            random_index_Q = random.randint(0, len(action_sequence) - 1)
            random_index_A = random_index_Q + 1
            yuanzi_insert = random.choice(can_yuanzi)

            # 插入问答：用户说和机器人说
            action_sequence.insert(random_index_Q, yuanzi["user.speak"][yuanzi_insert]["action"])
            action_sequence.insert(random_index_A, yuanzi["bot.speak"][yuanzi_insert]["action"])

    elif node_name == "SDHFK":
        print(len(action_sequence))
        for i in range(random.randint(1, 4)):
            action_sequence.append(yuanzi["system.call_response"]["DHFK"]["action"])

        print(len(action_sequence))
        # 插入 0 到 1 个问答
        num_of_qa = random.randint(0, 1)  # 控制插入 0 到 2 个问答
        can_yuanzi = ["XL", "YH", "NYH", "NSP", "BOT", "SPLM","NDD"]

        for _ in range(num_of_qa):  # 循环插入问答
            random_index_Q = random.randint(0, len(action_sequence) - 1)
            random_index_A = random_index_Q + 1
            yuanzi_insert = random.choice(can_yuanzi)

            # 插入问答：用户说和机器人说
            action_sequence.insert(random_index_Q, yuanzi["user.speak"][yuanzi_insert]["action"])
            action_sequence.insert(random_index_A, yuanzi["bot.speak"][yuanzi_insert]["action"])


    elif node_name == "ENDFK":
        for i in range(random.randint(1, 4)):
            action_sequence.append(yuanzi["system.call_response"]["ENDFK"]["action"])

        # 插入 0 到 1 个问答
        num_of_qa = random.randint(0, 1)  # 控制插入 0 到 1 个问答
        can_yuanzi = ["XL", "YH", "NYH", "NSP", "BOT", "SPLM","NDD"]

        for _ in range(num_of_qa):  # 循环插入问答
            random_index_Q = random.randint(0, len(action_sequence) - 1)
            random_index_A = random_index_Q + 1
            yuanzi_insert = random.choice(can_yuanzi)

            # 插入问答：用户说和机器人说
            action_sequence.insert(random_index_Q, yuanzi["user.speak"][yuanzi_insert]["action"])
            action_sequence.insert(random_index_A, yuanzi["bot.speak"][yuanzi_insert]["action"])


    else:
        for action_key in node_action[node_name]:
            if action_key.startswith("user.speak"):
                action_type = action_key.split("user.speak.")[1]
                action = yuanzi["user.speak"][action_type]
                temple = copy.deepcopy(action)
                # action["action"]["node"] = node_name

                if "用户询问其他类目下的商品的商品信息" in action["action"]["content"]:
                    category_id = get_new_category_id(Now_cat_id)
                    # print(Want_cat_id)
                    # print(temple["action"]["content"])

                    temple["action"]["content"] = temple["action"]["content"].replace("用户询问其他类目下的商品的商品信息", f"用户询问{category_id}类目下的商品的商品信息")
                    # print(temple["action"]["content"])
                    Want_cat_id = category_id
                action_sequence.append(temple["action"])

            elif action_key.startswith("bot.speak"):
                action_type = action_key.split("bot.speak.")[1]
                action = yuanzi["bot.speak"][action_type]
                # action["action"]["node"] = node_name
                action_sequence.append(action["action"])


            elif action_key.startswith("system.call_response"):
                action_type = action_key.split("system.call_response.")[1]
                action = yuanzi["system.call_response"][action_type]
                temple = copy.deepcopy(action)
                # action["action"]["node"] = node_name

                if "status=arrived," in action["action"]["content"]:
                    # temple["action"]["content"] = temple["action"]["content"].replace("status=arrived,", f"status=arrived_{Want_cat_id},")
                    Now_cat_id = Want_cat_id

                if "categories | is_complete=yes" in action["action"]["content"]:
                    temple["action"]["return"] = all_cat_data

                if "category_id=%%商品的类目编号%%" in action["action"]["content"]:
                    temple["action"]["content"] = temple["action"]["content"].replace("category_id=%%商品的类目编号%%", f"category_id={Now_cat_id},")
                
                if "cid=%%商品的类目编号%%" in action["action"]["content"]:
                    temple["action"]["content"] = temple["action"]["content"].replace("cid=%%商品的类目编号%%", f"cid={Want_cat_id},")
                

                action_sequence.append(temple["action"])

            elif action_key.startswith("bot.call"):
                action_type = action_key.split("bot.call.")[1]
                action = yuanzi["bot.call"][action_type]
                # action["action"]["node"] = node_name
                temple = copy.deepcopy(action)

                if "cid" in action["action"]["content"]:
                    # print("Want_cat_idcid",Want_cat_id)
                    temple["action"]["content"] = temple["action"]["content"].replace("%%商品的类目编号%%", Want_cat_id)

                if "category_id" in action["action"]["content"]:
                    category_id = Now_cat_id
                    temple["action"]["content"] = temple["action"]["content"].replace("%%商品的类目编号%%", category_id)
                    Want_cat_id = category_id
                
                action_sequence.append(temple["action"])

    return action_sequence, Now_cat_id, Want_cat_id

# 修改 traverse_tree_randomly 只返回节点路径（不生成action_list）
def traverse_tree_randomly(node_name, depth=0, max_depth=25):
    if node_name == "SEND":
        return ["SEND"]
    
    if depth >= max_depth:
        return ["END", "ENDCJ", "ENDFK","ENDJS"]
    
    next_nodes = node_structure.get(node_name, [])
    if not next_nodes:
        return ["END", "ENDCJ", "ENDFK","ENDJS"]
    
    next_node = random.choices([node[0] for node in next_nodes], [node[1] for node in next_nodes])[0]
    
    return [node_name] + traverse_tree_randomly(next_node, depth + 1, max_depth)


# 新函数，遍历节点路径并构建最终的动作序列
def process_nodes(node_list, Now_cat_id, Want_cat_id):
    final_action_sequence = []
    for node in node_list:
        # print("node",node)
        action_sequence, Now_cat_id, Want_cat_id = build_action_sequence(node, Now_cat_id, Want_cat_id)
        # print("action_list",action_sequence)

        final_action_sequence.extend(action_sequence)
    return final_action_sequence, Now_cat_id, Want_cat_id


# 遍历所有场景数据并构建最终的动作序列
end_json_data = []
for scene in scene_data:
    Now_cat_id = "kong"
    Want_cat_id = "kong"
    product_info = scene["scene_info"]  # 商品类目ID
    category_ids = [product["category_id"] for product in product_info.values()]

    # all_cat_data = [{"id": product_info[cat]["category_id"], "name": cat} for cat in product_info.keys()]
    all_cat_data = [{"id": product_info[cat]["category_id"], "name": cat ,"description": product_info[cat]["description"]} for cat in product_info.keys()]

    # 先获取节点路径列表
    node_list = traverse_tree_randomly("START")
    print("node_list",node_list)
    # 然后根据节点列表构建动作序列
    final_action_sequence, _, _ = process_nodes(node_list, Now_cat_id, Want_cat_id)
    a_scene = {
        "scene":scene["scene_id"],
        "data":product_info,
        "action" :final_action_sequence
    }
    end_json_data.append(a_scene)

# 将所有场景的动作序列保存到文件
output_file_path = "/mnt/common/intern/qt/8w_project/machine_data/input/action.json"
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(end_json_data, f, ensure_ascii=False, indent=4)

print(f"动作序列已成功写入 {output_file_path}")
