import torch
from modulet3 import T3Model
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
        kc = activation['fc3']
        zc = kc
        # print("layer is: \n", kc[0])
        """
        print("layer output is: \n",output)
        print("layer kc is: \n", kc)
        """
        mean_loss = nn.L1Loss()

        # print("before the kc is:", kc)
        for i in range(zc.size(dim=0)):
            min_loss = 10000000000000000
            #for l in range(16):
            #   plt.scatter(l, zc[i][l], color='green', label="test")
            for key in feature.keys():
                temp_loss = mean_loss(feature.get(key), zc[i])
                if (temp_loss < min_loss):
                    min_loss = temp_loss
                    best_feature = feature.get(key)
                    # print("find!!!! key is:", key)
            # print("end round")
            # print('before: \n',zc[i])
            # print('after: \n',best_feature)


            #kc[i] = best_feature

            #print(kc[i][:16])
            #for k in range(16):
                #plt.scatter(k, kc[i][k], color='red', label="memory")
            #plt.show()
        # print("KC 2 layer is: \n",kc[2])
        # print("output 2 layer is: \n", output[2])
        # output=torch.ones_like(activation['fc3'])
        # print("after layer output is: \n", output)

        #return kc

    return hook



model2 = T3Model()

model2.load_state_dict(torch.load("model_20220318_121608_4"))

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

mnist_testset = datasets.FashionMNIST('./data', train=False, transform=transform, download=False)
validation_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=4, shuffle=False, num_workers=2)


# since we're not training, we don't need to calculate the gradients for our outputs


feature = torch.load("save_featuret3")
# print("sum feature:\n", feature)
for key in feature.keys():
    tempdic = feature.get(key)
    tempdic[1] = tempdic[1] / tempdic[0]
    feature.update({key: tempdic[1]})
# print("after process:\n",feature)
model2.fc3.register_forward_hook(get_activation('fc3'))
i = 0

def main():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model2(images)
            """
            #print(outputs)
            #kc = activation['fc3']
            #zc=kc[:, :16]
            #temp_kc=kc
            #print("original kc:",kc)
            #find_close(feature,zc,temp_kc)
            #activation['fc3']=kc
            #print("after modify",activation['fc3'])
            # the class with the highest energy is what we choose as prediction
            """
            _, predicted = torch.max(outputs.data, 1)
            # print("labels are:",labels)
            # print(_, predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            """
            # print(mask[2].size==torch.Size([272,16]))
            i += 1
            if (i > 0):
                break
    
            
            i=3
            with torch.no_grad():
                for param in model2.parameters():
                    # mask is also saved in param, but mask.requires_grad=False
                    # if param.requires_grad:
                    # param -= learning_rate * param.grad
                    # check masked param.grad
                    #print(param)
    
                    if np.array(param).size == 160:
                        print('↓↓↓masked weight↓↓↓')
                        print(param.size())
                        print(param)
            break
    
    
                        ploti=plt.figure(i)
                        plt.imshow(np.array(param), cmap=plt.cm.hot, interpolation='none')
                        plt.show()
                        i+=1
    
             """

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')


if __name__ == '__main__':
    main()
