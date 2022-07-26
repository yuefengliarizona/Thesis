import torch

import torchvision.datasets as datasets
from simplemodel import SimpleModel
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as transforms
import numpy as np


MBIN_node=32
MBON=16
device = "cpu"# "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


mask=[]
for i in range(3):
    m_temp = np.loadtxt(open("./mask/layer"+str(i+1)+".csv", "rb"), delimiter=" ", skiprows=0)
    mask.append(torch.tensor(m_temp))

model=SimpleModel(mask)
model.to(device)
model.load_state_dict(torch.load('model_46_last train'))

feature=[]
mbon_feature=[]
T_table=[]
L_table=[]
likelyhood=[]
max1_max2=[]
predicted_table=[]

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
mnist_trainset = datasets.FashionMNIST('./data', train=True, transform=transform, download=False)
mnist_testset = datasets.FashionMNIST('./data', train=False, transform=transform, download=False)


training_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=4, shuffle=True, num_workers=2)
validation_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=4, shuffle=False, num_workers=2)

model.layers[-2].register_forward_hook(get_activation('kc_layer'))
model.layers[-1].register_forward_hook(get_activation('mbon'))
model.fc5.register_forward_hook(get_activation('Likely_hood'))

file1 = open("mbon memory.txt", "w")
file2 = open("train lable.txt", "w")
file3 = open("train predict.txt", "w")


def train_feature(value,T_label,Lable,predicted,mbon,out_like):
    _, like_index = torch.sort(out_like, descending=True)
    for i in range(4):
        file1.write(str(mbon[i])+ "\n" )
        file2.write(str(Lable[i].item())  + "\n")
        file3.write(str(predicted[i].item() ) + "\n")
        feature.append(value[i])
        T_table.append(T_label[i])
        L_table.append(Lable[i])
        mbon_feature.append(mbon[i])
        likelyhood.append(out_like[i][Lable[i].item()])
        max1_max2.append(out_like[i][like_index[i][0]]-out_like[i][like_index[i][1]])
        predicted_table.append(predicted[i])

def main():
    Round_ct=0
    for i, data in enumerate(validation_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        model.to(device)

        kc = activation['kc_layer']
        mbon = activation['mbon']
        out_like = activation['Likely_hood']

        _, predicted = torch.max(outputs.data, 1)
        T_flag = (predicted == labels)


        train_feature(kc[:, :MBIN_node], T_flag, labels, predicted, mbon, out_like)

        Round_ct += 4
        if (Round_ct ==30000):
            break
    file1.close()
    file2.close()
    file3.close()
    torch.save(feature, "save_feature")
    torch.save(T_table, "Boolean_table")
    torch.save(L_table, "label_table")
    torch.save(mbon_feature, "mbon_feature")
    torch.save(likelyhood, "likelyhood")
    torch.save(max1_max2, "max1_max2")
    torch.save(predicted_table, "predicted_table")




if __name__ == '__main__':
    main()