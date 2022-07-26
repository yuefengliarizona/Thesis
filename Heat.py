import torch
from simplemodel import SimpleModel
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import sys
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
        """
        kc = activation['fc3']
        zc = kc[:, :16]
        np.savetxt('whole_layer.csv', kc, delimiter=",")
        np.savetxt('MBIN.csv', zc, delimiter=",")
        loss_result = torch.cdist(zc, mem_feature, p=1)
        sort, index = torch.sort(loss_result)
        find_flag = 0
        for i in range(4):
            for j in range(5):
                if (T_table[index[i][j]]):
                    match = mem_feature[index[i][j]]
                    find_flag = 1
                break
            if (find_flag):
                kc[i][:16] = match
            find_flag = 0
        """
        """
        """
        # return kc
    return hook


m1 = np.loadtxt(open("./mask/layer1.csv", "rb"), delimiter=",", skiprows=0)
m2 = np.loadtxt(open("./mask/layer2.csv", "rb"), delimiter=",", skiprows=0)
m3 = np.loadtxt(open("./mask/layer3.csv", "rb"), delimiter=",", skiprows=0)

mask = []
mask.append(torch.tensor(m1))
mask.append(torch.tensor(m2))
mask.append(torch.tensor(m3))
model2 = SimpleModel(mask)
# model2.cuda()
model2.load_state_dict(torch.load("model_20220405_124140_6"))

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
feature = torch.load("save_feature")
mem_feature = torch.zeros(len(feature), 16)
for i in range(len(feature)):
    mem_feature[i] = feature[i]

T_table = torch.load("Boolean_table")

mnist_testset = datasets.FashionMNIST('./data', train=False, transform=transform, download=False)
validation_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=4, shuffle=False, num_workers=2)


def main():
    correct = 0
    total = 0
    count = 0
    model2.fc3.register_forward_hook(get_activation('fc3'))
    model2.fc2.register_forward_hook(get_activation('fc2'))
    model2.fc4.register_forward_hook(get_activation('fc4'))
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data

            # calculate outputs by running images through the network
            outputs = model2(images)

            np.savetxt('KClayer.csv', activation['fc2'], delimiter=",")
            np.savetxt('MBIN-FNlayer.csv', activation['fc3'], delimiter=",")
            np.savetxt('MBONlayer.csv', activation['fc4'], delimiter=",")
            np.savetxt('output.csv', outputs, delimiter=",")
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


            with torch.no_grad():
                i=0
                for param in model2.parameters():
                    # mask is also saved in param, but mask.requires_grad=False
                    # if param.requires_grad:
                    # param -= learning_rate * param.grad
                    # check masked param.grad
                    # print(param)
                    if np.array(param).size == m1.size:
                        PNKC=param.numpy()
                        np.savetxt('PN_KC.csv',PNKC, delimiter=",")
                    if np.array(param).size == m2.size:
                        ploti = plt.figure(i)
                        plt.imshow(np.array(param), cmap=plt.cm.hot, interpolation='none')
                        plt.show()
                        i += 1
                        b=param.numpy()
                        np.savetxt('hn'+str(i)+'.csv',b, delimiter=",")

                break

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')


if __name__ == '__main__':
    main()
