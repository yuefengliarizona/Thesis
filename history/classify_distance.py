import torch
from simplemodel_clas import SimpleModel
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import sys
import matplotlib.pyplot as plt
import re
from random import randint


config=sys.argv[1]
con_f= open(config, "r")
m_size=re.findall('[0-9]+',con_f.readline())
m_modle=re.findall('[0-9]+',con_f.readline())
m_modle=int(m_modle[0])
print("model:"+str(m_modle))

T_count=False
mask=[]
model_load=con_f.readline().strip()
MBIN_nodes=int(re.search(r'\d+', con_f.readline()).group())

for i in range(int(m_size[0])):
    m_temp = np.loadtxt(open("./mask/layer"+str(i+1)+".csv", "rb"), delimiter=" ", skiprows=0)
    mask.append(torch.tensor(m_temp))
device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

np.set_printoptions(threshold=sys.maxsize)




activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
        kc = activation['kc_layer']
        if (m_modle == 1):
            m_modle1(kc)
        elif (m_modle == 2):
            m_modle2(kc)
        elif (m_modle == 3):
            m_modle3(kc)
        elif (m_modle == 4):
            m_modle4(kc)
        elif (m_modle == 5):
            m_modle5(kc)
        elif (m_modle == 6):
            m_modle6(kc)
        elif (m_modle == 7):
            m_modle7(kc)
        elif (m_modle == 8):
            m_modle8(kc)
        elif (m_modle == 9):
            m_modle9(kc)
        elif (m_modle == 10):
            m_modle10(kc)
        elif (m_modle == 11):
            m_modle11(kc)
        elif (m_modle == 12):
            m_modle12(kc)
        elif (m_modle == 13):
            m_modle13(kc)
        elif (m_modle == 14):
            m_modle14(kc)
        else:
            pass
    return hook



model2= SimpleModel(mask)
model2.to(device)
model2.load_state_dict(torch.load(model_load))

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
feature = torch.load("save_feature")
mem_feature=torch.stack(feature)


L_table=torch.load("label_table")
P_table=torch.load("predicted_table")
T_table = torch.load("Boolean_table")
mem_mean= torch.load("memory_mean")
mostlike=torch.load("mostlike")
mostlike=torch.stack(mostlike)
max1_max2=torch.load("max1_max2")
max1_max2=torch.stack(max1_max2)
_,maxsort=torch.sort(max1_max2)
"""
dic_mem=[]
for i in maxsort[0:9000]:
    dic_mem.append(mem_feature[i])
dic_mem=torch.stack(dic_mem)
"""

model2.layers[-2].register_forward_hook(get_activation('kc_layer'))
mnist_testset = datasets.FashionMNIST('./data', train=True, transform=transform, download=False)
validation_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=4, shuffle=False, num_workers=2)
predict_result=[]

Label_data=[]
Perdict_data=[]

cos = nn.CosineSimilarity(dim=1, eps=1e-6)


def m_modle1(kc):
    zc = kc[:, :MBIN_nodes]
    loss_result = torch.cdist(zc, mem_feature.to(device), p=1)
    sort, index = torch.sort(loss_result)
    find_flag = 0
    for i in range(4):
        for j in range(50):
            if (T_table[index[i][j]]):
                match = mem_feature[index[i][j]]
                find_flag = 1
                break
        if (find_flag):
            zc[i] = match
        find_flag = 0

def m_modle2(kc):
    zc = kc[:, :MBIN_nodes]
    mean_loss = nn.L1Loss()

    for i in range(4):
        min_loss = 10000000000000000
        for j in range(10):
            temp_loss = mean_loss(mem_mean.get(j), zc[i])
            if (temp_loss < min_loss):
                min_loss = temp_loss
                best_feature = mem_mean.get(j)
        zc[i]=best_feature

def m_modle3(kc):
    zc = kc[:, :MBIN_nodes]
    loss_result = torch.cdist(zc, mem_feature.to(device), p=1)
    sort, index = torch.sort(loss_result)
    find_flag = 0
    for i in range(4):
        for j in range(6):
            if (~T_table[index[i][0]] & T_table[index[i][j]]):
                match = mem_feature[index[i][j]]
                find_flag = 1
                break
        if (find_flag):
            kc[i][:MBIN_nodes] = match
        find_flag = 0
    return index

def m_modle4(kc):
    zc = kc[:, :MBIN_nodes]
    loss_result = torch.cdist(zc, mem_feature.to(device), p=1)
    sort, index = torch.sort(loss_result)
    global T_count
    T_count=True
    for i in range(4):
        if (~T_table[index[i][0]]):
            T_count= False

def m_modle5(kc):
    zc = kc[:, :MBIN_nodes]
    loss_result = torch.cdist(zc, mem_feature.to(device), p=1)
    sort, index = torch.sort(loss_result)

    for i in range(4):
        temp1=[]
        temp2 = []
        for j in range(50):
            temp1.append(L_table[index[i][j].item()])
            temp2.append(P_table[index[i][j].item()])
        Label_data.append(temp1)
        Perdict_data.append(temp2)

def m_modle6(kc):
    zc = kc[:, :MBIN_nodes]
    loss_result = torch.cdist(zc, mem_feature.to(device), p=1)
    #print(loss_result)
    sort, index = torch.sort(loss_result)
    #print(sort)
    for i in range(4):
        temp=0
        temp_mem=0
        input1 = torch.randn(1, 16)
        input1[0]=zc[i]
        input2 = torch.randn(1, 16)
        for j in range(50):
            input2[0]=mem_feature[index[i][j].item()]
            temp+=torch.exp(cos(input1,input2))
        for k in range(50):
            input2[0] = mem_feature[index[i][k].item()]
            temp2 = cos(input1, input2)
            temp_mem+=torch.exp(temp2) /temp*mem_feature[index[i][k].item()]
        zc[i]=temp_mem

def m_modle7(kc):
    zc = kc[:, :MBIN_nodes]
    loss_result = torch.cdist(zc, dic_mem.to(device), p=1)
    sort, index = torch.sort(loss_result)
    for i in range(4):
        zc[i]= dic_mem[index[i][0]]

def m_modle8(kc):
    zc = kc[:, :MBIN_nodes]
    loss_result = torch.cdist(zc, mem_feature.to(device), p=1)
    sort,index = torch.sort(loss_result)
    for i in range(4):
        if(T_table[index[i][0].item()]):
            #zc[i] = mem_feature[index[i][0]]
            pass
        else:
            label_index=L_table[index[i][0].item()]
            for k in (1,150):
                if(L_table[index[i][k]]==label_index and T_table[index[i][k]]):
                    zc[i] = mem_feature[index[i][k]]
                    break

def m_modle9(kc):
    zc = kc[:, :MBIN_nodes]
    loss_result = torch.cdist(zc, mem_feature.to(device), p=1)
    sort, index = torch.sort(loss_result)
    for i in range(4):
        if(T_table[index[i][0].item()]):
            pass
        else:
            label_index=L_table[index[i][0].item()]
            while (1):
                index_new = randint(0, 12999)
                if (label_index == L_table[index_new].item() and T_table[index_new]):
                    zc[i] = mem_feature[index_new]
                    break


def m_modle10(kc):
    zc = kc[:, :MBIN_nodes]
    loss_result = torch.cdist(zc, mem_feature.to(device), p=1)
    sort, index = torch.sort(loss_result)
    for i in range(4):
        max_lab=[0]*10
        for j in range(400):
            max_lab[L_table[index[i][j].item()].item()]+=1
        if(max(max_lab)<50):
            print(max_lab)
        #print(max_lab)
        #print(max_lab.index(max(max_lab)))
        while (1):
            index_random = randint(0, 20000)
            if (max_lab.index(max(max_lab)) == L_table[index_random].item() and T_table[index_random]):
                zc[i] = mem_feature[index_random]
                break


def m_modle11(kc):
    zc = kc[:, :MBIN_nodes]
    loss_result = torch.cdist(zc, mem_feature.to(device), p=1)
    #print(loss_result)
    sort, index = torch.sort(loss_result)
    #print(sort)
    for i in range(4):
        zc[i]=(mem_feature[index[i][0].item()]+mem_feature[index[i][1].item()])/2

def m_modle12(kc):
    zc = kc[:, :MBIN_nodes]
    loss_result = torch.cdist(zc, mem_feature.to(device), p=1)
    #print(loss_result)
    sort, index = torch.sort(loss_result)
    #print(sort)
    for i in range(4):
        temp=0
        temp_mem=0
        for j in range(5):
            temp += torch.exp(sort[i][j])
        for k in range(5):
            temp2 = torch.exp(sort[i][k])
            temp_mem += temp2 / temp * mem_feature[index[i][k].item()]
        zc[i] = temp_mem

def m_modle13(kc):
    zc = kc[:, :MBIN_nodes]
    for i in range(4):
        while(1):
            index = randint(0, 12999)
            if(labels[i].item() ==L_table[index].item()):
                zc[i]=mem_feature[index]
                break

def m_modle14(kc):
    zc = kc[:, :MBIN_nodes]
    loss_result = torch.cdist(zc, mem_feature.to(device), p=1)
    #print(loss_result)
    sort, index = torch.sort(loss_result)
    #print(sort)
    for i in range(4):
        temp=0
        temp_mem=0
        for j in range(15):
            temp += torch.log(sort[i][j])
        for k in range(15):
            temp2 = torch.log(sort[i][k])
            temp_mem += temp2 / temp * mem_feature[index[i][k].item()]
        zc[i] = temp_mem


def main():
    correct = 0
    total = 0
    count=0
    original_label=[]
# since we're not training, we don't need to calculate the gradients for our outputs

    #file1 = open("predict_result_model"+str(m_modle)+".txt", "w")
    #file2 = open("till correct predict" + str(m_modle) + ".txt", "w")


    with torch.no_grad():
        for data in validation_loader:
            global labels
            images, labels = data
            images, labels =images.to(device), labels.to(device)
            #for i in range(4):
            #    original_label.append(labels[i])

            # calculate outputs by running images through the network

            outputs = model2(images)
            _, predicted = torch.max(outputs.data, 1)


            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #for i in range(len(predicted)):
                #file1.writelines("perdict:"+str(predicted[i].item())+". True is: " + str(labels[i].item())+"\n")
            #break
            count+=4
            if(count>10000):
                break

    #file1.close()
    #file2.close()
    """
    file1 = open("test_label.txt", "w")
    for element in original_label:
        file1.write( str(element)+ "\n")
    file1.close()
    file2 = open("first_50_label.txt", "w")
    for element in Label_data:
        file2.write( str(element)+ "\n")
    file2.close()
    file3 = open("first_50_predict.txt", "w")
    for element in Perdict_data:
        file3.write(str(element)+ "\n")
    file3.close()
    """
    #print(total,"    total is")
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

if __name__ == '__main__':
    main()
