import os
"""
生成数据集yaml文件中的train、test、val的txt文件

需要保证的数据集格式：
如果不是的话就修改下代码

xxxx/数据集名：
-train
--label
--images
-test
--label
--images
-val
--label
--images
"""
datatype = ["train2", "val2"]
real_path = "/home/markzhang/datasets/TSR_TFL/images"
#实际路径，这里是绝对路径
server_path = "../datasets/TSR_TFL/images"
#保存路径，这里是相对路径，模型最终读取的文件路径
yaml = []
for dt in datatype:
    f_list = os.listdir(real_path + "/" + dt)
    # print(f_list)
    with open(real_path + "/" + dt +".txt", "w") as f:
        for fl in f_list:
            if fl.endswith("jpg") or fl.endswith("png"):
                f.write(server_path + "/" + dt + "/" + fl + "\n")
    yaml.append(dt + ": " + server_path + "/" + dt + "/" + dt +".txt")


