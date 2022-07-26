import torch
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

feature = torch.load("save_feature")
L_table =torch.load("label_table")
mbon_feature=torch.load("mbon_feature")
mem_feature=torch.zeros(len(feature), 16)
for i in range(len(mem_feature)):
    mem_feature[i]=feature[i]

mbon=torch.zeros(len(mbon_feature), 16)
for i in range(len(mbon_feature)):
    mbon[i]=mbon_feature[i]
print(mbon)
print(mbon_feature)
#np.savetxt('f2_feature.csv', mem_feature, delimiter=",")
#L_table=np.array(L_table)

"""
print(mem_feature)
print(len(t))
print(len(mem_feature[0]))
print((mem_feature[0]))
"""

"""
"""

x=np.array(mem_feature)
X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(x)

colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'grey', 'orange', 'purple'
for j in range(len(X_embedded)):
    plt.scatter(X_embedded[j][1],X_embedded[j][0],c=colors[L_table[j]],label=L_table[j], s=2)
plt.show()




y=np.array(mbon)
y_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(y)

colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'grey', 'orange', 'purple'
for j in range(len(y_embedded)):
    plt.scatter(y_embedded[j][1],y_embedded[j][0],c=colors[L_table[j]],label=L_table[j], s=2)
plt.show()

"""
m1=torch.randint(0, 69, (1, 16))

m1=m1.type((torch.float))
print(m1)
z=torch.zeros(3, 16)
for i in range(3):
    m2 = torch.randint(0, 69, (1, 16))
    z[i]=m2
print(z)
z=z.float()
j=torch.cdist(m1, z, p=1)
sort,_=torch.sort(j)
print((sort))
print(_)
print((_[0][0]))
print(j)
"""
