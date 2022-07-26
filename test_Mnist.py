

import torch
import torchvision
import torchvision.datasets as datasets

from simplemodel import SimpleModel
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import sys
import re


config=sys.argv[1]
con_f= open(config, "r")
m_size=re.findall('[0-9]+',con_f.readline())

MBIN_node=int(re.search(r'\d+', con_f.readline()).group())


transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
mnist_trainset = datasets.MNIST('./data', train=True, transform=transform, download=True)
mnist_testset = datasets.MNIST('./data', train=False, transform=transform, download=True)


training_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=4, shuffle=True, num_workers=2)
validation_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=4, shuffle=False, num_workers=2)

mask=[]
for i in range(int(m_size[0])):
    m_temp = np.loadtxt(open("./mask/layer"+str(i+1)+".csv", "rb"), delimiter=" ", skiprows=0)
    mask.append(torch.tensor(m_temp))





device = "cpu"# "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model=SimpleModel(mask)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.layers[-2].register_forward_hook(get_activation('kc_layer'))
model.layers[-1].register_forward_hook(get_activation('mbon'))
model.fc5.register_forward_hook(get_activation('Likely_hood'))
"""
model.fc2.register_forward_hook(get_activation(model.layers[-2]))
model.fc3.register_forward_hook(get_activation('fc3'))
model.fc4.register_forward_hook(get_activation('fc4'))
model.fc5.register_forward_hook(get_activation('fc5'))
f2_feature=[]
f4_feature=[]
f5_feature=[]
"""

feature=[]
mbon_feature=[]
T_table=[]
L_table=[]
likelyhood=[]
max1_max2=[]
predicted_table=[]


def train_feature(value,T_label,Lable,predicted,mbon,out_like):
    _, like_index = torch.sort(out_like, descending=True)
    for i in range(4):
        feature.append(value[i])
        T_table.append(T_label[i])
        L_table.append(Lable[i])
        mbon_feature.append(mbon[i])
        likelyhood.append(out_like[i][Lable[i].item()])
        max1_max2.append(out_like[i][like_index[i][0]]-out_like[i][like_index[i][1]])
        predicted_table.append(predicted[i])

        #f2_feature.append(f2_fea[i])
        #f4_feature.append(f4_fea[i])
        #f5_feature.append(f5_fea[i])



def train_one_epoch(epoch_index, tb_writer,Round_ct,mem_start_flag):
    running_loss = 0.
    mem_end_flag=False

    last_loss=np.Inf
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(inputs)
        model.to(device)

        kc=activation['kc_layer']
        mbon=activation['mbon']
        out_like=activation['Likely_hood']
        """
        kc = activation['fc3']
        f2_fea = activation['fc2']
        f4_fea = activation['fc4']
        f5_fea = activation['fc5']
        
        """
        _, predicted = torch.max(outputs.data, 1)
        T_flag = (predicted==labels)

        if(mem_start_flag):
            train_feature(kc[:, :MBIN_node], T_flag,labels,predicted,mbon,out_like)
            Round_ct += 4
            if(Round_ct>23996):
                mem_end_flag=True
                model_path = 'model_{}_{}'.format(timestamp, "last train")
                torch.save(model.state_dict(), model_path)
                break
            continue
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            this_loss = running_loss / 1000 # loss per batch
            last_loss=this_loss
            print('  batch {} loss: {}'.format(i + 1, this_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', this_loss, tb_x)
            running_loss = 0.


    return last_loss,mem_end_flag

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))


def main():
    epoch_number = 0
    stop_flag=False
    mem_flag=False
    EPOCHS = 500
    loss_range=np.inf
    n_epochs_stop = 5
    min_val_loss = np.Inf
    last_loss_range=np.inf
    last_loss=np.inf
    early_stop_count=0
    best_vloss = 1_000_000.
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        Round_ct=0
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss,stop_flag= train_one_epoch(epoch_number, writer,Round_ct,mem_flag)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0

        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vinputs, vlabels =vinputs.to(device), vlabels.to(device)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss


        avg_vloss = running_vloss / (i + 1)
        if (stop_flag):
            torch.save(feature, "save_feature")
            torch.save(T_table, "Boolean_table")
            torch.save(L_table, "label_table")
            torch.save(mbon_feature, "mbon_feature")
            torch.save(likelyhood, "likelyhood")
            torch.save(max1_max2, "max1_max2")
            torch.save(predicted_table, "predicted_table")
            break
        loss_range=abs(last_loss-avg_vloss)
        last_loss=avg_vloss
        if(loss_range<0.03 and epoch>35):
            early_stop_count+=1
        else:
            early_stop_count = 0
        last_loss_range=loss_range
        if(early_stop_count>=n_epochs_stop):
            mem_flag=True
            print("early stopping!!!!!")
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation.cuda()
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()
        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)
        epoch_number += 1





if __name__ == '__main__':
    main()


"""
torch.save(f2_feature,"f2_feature")
torch.save(f4_feature,"f4_feature")
torch.save(f5_feature,"f5_feature")

"""



