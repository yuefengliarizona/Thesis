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



config = sys.argv[1]
con_f = open(config, "r")
m_size = re.findall('[0-9]+', con_f.readline())
m_modle = re.findall('[0-9]+', con_f.readline())
m_modle = int(m_modle[0])
mask = []
model_load = con_f.readline().strip()
MBIN_nodes = int(re.search(r'\d+', con_f.readline()).group())
for i in range(int(m_size[0])):
    m_temp = np.loadtxt(open("./mask/layer" + str(i + 1) + ".csv", "rb"), delimiter=" ", skiprows=0)
    mask.append(torch.tensor(m_temp))
device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
R_count=0
image_count=0
np.set_printoptions(threshold=sys.maxsize)

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
        kc = activation['kc_layer']
        if (R_count == 1):
            m_modle1(kc)
        elif (R_count == 2):
            m_modle2(kc)
        elif (R_count == 3):
            m_modle3(kc)
        else:
            pass

    return hook


model2 = SimpleModel(mask)
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

T_table = torch.load("Boolean_table")
mem_mean = torch.load("memory_mean")
L_table=torch.load("label_table")
P_table=torch.load("predicted_table")


model2.layers[-2].register_forward_hook(get_activation('kc_layer'))
model2.fc5.register_forward_hook(get_activation('output'))
mnist_testset = datasets.FashionMNIST('./data', train=False, transform=transform, download=False)
validation_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=4, shuffle=False, num_workers=2)
predict_result = []
true_label=[]
out_feature = []

drawing={}

global labels

def m_modle1(kc):
    zc = kc[:, :MBIN_nodes]
    for i in range(4):
        while(1):
            index = randint(0, 12999)
            if(labels[i].item() ==L_table[index].item()):
                zc[i]=mem_feature[index]
                break


def m_modle2(kc):
    zc = kc[:, :MBIN_nodes]
    for i in range(4):
        while (1):
            index = randint(0, 12999)
            if (labels[i].item() != L_table[index].item()):
                zc[i] = mem_feature[index]
                break

def m_modle3(kc):
    zc = kc[:, :MBIN_nodes]
    for i in range(4):
        while (1):
            index = randint(0, 12999)
            if (labels[i].item() == L_table[index].item()and T_table[index]):
                zc[i] = mem_feature[index]
                break




def main():
    correct =correct2 =correct3 =correct4= 0
    total = 0
    global R_count
    global image_count
    # since we're not training, we don't need to calculate the gradients for our outputs


    with torch.no_grad():
        for data in validation_loader:
            global labels
            images, labels = data

            images, labels = images.to(device), labels.to(device)

            # calculate outputs by running images through the network

            outputs = model2(images)
            _, predicted = torch.max(outputs.data, 1)
            out_fea = activation['output']
            kc = activation['kc_layer'][:, :MBIN_nodes]
            Fake= activation['kc_layer'][:, MBIN_nodes:]
            print(len(kc[0]))
            print(len(Fake[2]))
            print(len(model2.layers[2].weight))
            weightT=model2.layers[2].weight
            print(sum(kc[0]*weightT[:, :MBIN_nodes]))
            print(sum(kc[1]*weightT[:, :MBIN_nodes]))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            _,like_index=torch.sort(out_fea,descending=True)
            for i in range(4):
                if(labels[i].item() in drawing.keys()):
                    temp=drawing.get(labels[i].item())

                    temp.append(out_fea[i][like_index[i][0]]-out_fea[i][like_index[i][1]])
                    drawing.update({labels[i].item() :temp})
                else:
                    temp=[out_fea[i][like_index[i][0]]-out_fea[i][like_index[i][1]]]
                    drawing.update({labels[i].item():temp})
            break
            """
            R_count += 1

            outputs2 = model2(images)
            _, predicted2 = torch.max(outputs2.data, 1)

            R_count +=1
            outputs3 = model2(images)
            _, predicted3 = torch.max(outputs3.data, 1)
            R_count += 1
            outputs4 = model2(images)
            _, predicted4 = torch.max(outputs4.data, 1)
            correct2 += (predicted2 == labels).sum().item()
            correct3 += (predicted3== labels).sum().item()
            correct4 += (predicted4 == labels).sum().item()
            R_count=0
            
            """
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
    #print("dictionary")
    print(f'Accuracy of the network on the 10000 test images2: {100 * correct2 / total} %')
    print(f'Accuracy of the network on the 10000 test images3: { 100 * correct3 / total} %')
    print(f'Accuracy of the network on the 10000 test images3: {100 * correct4 / total} %')

    #plt.hist(np.array(drawing.get(9)), color = "b", edgecolor = "r", bins = int(180/5))
    #plt.show()


if __name__ == '__main__':
    main()
