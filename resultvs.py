import numpy as np
import re
import torch
"""
file1= open("predict_result_model3.txt", "r")
Lines = file1.readlines()
result_table=np.zeros( (10, 10) )
for line in Lines:
    predict = re.findall('[0-9]+', line)
    result_table[int(predict[0])][int(predict[1])]+=1

print(result_table)
"""


feature = torch.load("save_feature")
T_table = torch.load("Boolean_table")
L_table=torch.load("label_table")
file2 = open("till correct predict"  + ".txt", "w")
mem_feature=torch.zeros(13000, 16)
for i in range(len(mem_feature)):
    mem_feature[i]=feature[i]


loss_result=torch.cdist(mem_feature, mem_feature, p=1)
sort, index = torch.sort(loss_result)

for i in range(len(mem_feature)):
    if (L_table[i].item() == 0 or L_table[i].item() == 6):
        file2.writelines(
            "True is: " + str(L_table[i].item()) + "  Predict order is: ")
        for j in range(1,len(index)):
                file2.writelines(
                    str(j) + ". "+str(L_table[index[i][j].item()].item())+", ")
                if(T_table[index[i][j].item()]):
                    file2.writelines("over, final memory label is:"+str(L_table[index[i][j].item()].item())+"\n")
                    break
    else:
        continue

file2.close()
