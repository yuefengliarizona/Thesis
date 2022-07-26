

import torch
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from modulet3 import T3Model
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F





transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
mnist_trainset = datasets.FashionMNIST('./data', train=True, transform=transform, download=False)
mnist_testset = datasets.FashionMNIST('./data', train=False, transform=transform, download=False)


training_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=4, shuffle=True, num_workers=2)
validation_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=4, shuffle=False, num_workers=2)




device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model=T3Model()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_fn = torch.nn.CrossEntropyLoss()






"""
for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        outputs1 = model(inputs)
        # outputs2 = model(inputs)
        # Compute the loss and its gradients
        #loss1=loss_fn(outputs1, outputs2)
        print(outputs1)
        loss = loss_fn(outputs1, labels)
        optimizer.zero_grad()
        print("loss is:")
        print(loss)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        if(i==35):
            break
"""


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.fc3.register_forward_hook(get_activation('fc3'))

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
feature={}
def train_feature(label, value):
    for i in range(4):
        tempdic=[]
        if classes[label[i]] in feature.keys():
            tempdic=feature.get(classes[label[i]])
            tempdic[0]=tempdic[0]+1
            tempdic[1]=value[i]+tempdic[1]
            feature.update({classes[label[i]]:tempdic})
            #print("multi times")
        else:
            tempdic.append(1)
            tempdic.append(value[i])
            feature.update({classes[label[i]]: tempdic})
        #print(feature.keys())

def train_one_epoch(epoch_index, tb_writer,EPOCHS):
    running_loss = 0.
    last_loss = 0.
    learning_rate = 0.005

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(inputs)
        kc = activation['fc3']
        if(EPOCHS>5):
            train_feature(labels, kc)
            pass
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()
            # Adjust learning weights
        optimizer.step()
            # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))


def main():
    epoch_number = 0
    EPOCHS = 6
    best_vloss = 1_000_000.
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer,EPOCHS)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
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

torch.save(feature,"save_featuret3")

"""
"""



