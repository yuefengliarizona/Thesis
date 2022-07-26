import torch
import re
import sys
import numpy

config=sys.argv[1]
con_f= open(config, "r")
feature = torch.load("save_feature")
label = torch.load("label_table")
feature = feature
label = label
mem_size=re.findall('[0-9]+',con_f.readline())
count_dic={}
feature_dic={}


max1_max2=torch.load("max1_max2")
max1_max2=torch.stack(max1_max2)

for i in range(int(mem_size[0])):
    temp=label[i].cpu().numpy()
    temp =int(temp)
    if( temp in count_dic.keys()):
        count=count_dic.get(temp)
        count+=1
        mem=feature_dic.get(temp)
        mem+=feature[i]
        count_dic.update({temp:count})
        feature_dic.update({temp: mem})
    else:
        count_dic.update({temp: 1})
        feature_dic.update({temp: feature[i]})


for j in count_dic.keys():
    count_total=count_dic.get(j)
    mem_total = feature_dic.get(j)
    mem_mean=mem_total/count_total
    feature_dic.update({j: mem_mean})

torch.save(feature_dic,"memory_mean")

sortedmem=[[],[],[],[],[],[],[],[],[],[]]
like_temp= [[],[],[],[],[],[],[],[],[],[]]

T_table = torch.load("Boolean_table")
likelyhood = torch.load("likelyhood")



overall=[]
"""
for j in range(10):
    mem_temp[j]=[]
    like_temp[j] = []
    """
for i in range(len(T_table)):
    if(T_table[i]):
        sortedmem[label[i].item()].append(feature[i])
        like_temp[label[i].item()].append(likelyhood[i])

for j in range(10):
    like_temp[j]=torch.stack(like_temp[j])
    sorted, indices =torch.sort(like_temp[j], descending=True)
    for k in range(800):
        #print(sortedmem[j][indices[k]])
        overall.append(sortedmem[j][indices[k]])

_,maxsort=torch.sort(max1_max2)
dic_mem=[]
for i in maxsort[0:9000]:
    dic_mem.append(feature[i])


print(len(overall))

print(dic_mem)

torch.save(overall,"mostlike")

