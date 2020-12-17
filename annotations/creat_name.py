#2、生成名称目录脚本
import os
imglst = os.listdir("./annotations/xmls/")
with open("./annotations/trainval_person.txt","w") as ff:
    for img_path in imglst:
        name = img_path.split(".")[0]
        print(name)
        ff.write(name+"\n")