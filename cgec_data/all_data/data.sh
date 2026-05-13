#把输出的数据进行分割，分成训练集和验证集
python /mnt/common/intern/qt/Data_project/CGEC_Data/spilt.py --input /mnt/common/intern/qt/Data_project/CGEC_Data/ori_data/couple_output_edits_explain.json --output_dir /mnt/common/intern/qt/Data_project/CGEC_Data/split_data

#把分割好的数据进行格式转换，转换成适合模型训练的格---train
python /mnt/common/intern/qt/Data_project/CGEC_Data/fitune_data.py --input /mnt/common/intern/qt/Data_project/CGEC_Data/split_data/train.json --output /mnt/common/intern/qt/school_project/LLaMA-Factory/data/train.json
#把分割好的数据进行格式转换，转换成适合模型训练的格---valid
python /mnt/common/intern/qt/Data_project/CGEC_Data/fitune_data.py --input /mnt/common/intern/qt/Data_project/CGEC_Data/split_data/valid.json --output /mnt/common/intern/qt/school_project/LLaMA-Factory/data/valid.json
#把分割好的数据进行格式转换，转换成适合模型训练的格---test
python /mnt/common/intern/qt/Data_project/CGEC_Data/fitune_data.py --input /mnt/common/intern/qt/Data_project/CGEC_Data/split_data/test.json --output /mnt/common/intern/qt/school_project/LLaMA-Factory/data/test.json


#处理原始的输入数据和标准输出数据  -test_out_qt.json\test_out_check_fin_qt.json
python /mnt/common/intern/qt/Data_project/CGEC_Data/eval_data.py


