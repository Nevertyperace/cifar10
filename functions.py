import numpy as np
import random
import torch
from sympy.combinatorics.graycode import gray_to_bin
from sympy.combinatorics.graycode import bin_to_gray
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# input: list binary 1,0 of length nBits representing number using gray coding
# output: real value
def chrom2real(c, nBits):
    maxnum = 2**nBits
    indasstring=''.join(map(str, c))
    degray=gray_to_bin(indasstring)
    numasint=int(degray, 2) # convert to int from base 2 list
    numinrange=-1.0+2.0*numasint/maxnum
    return numinrange*20

#turns real-valued weight into chromosome encoding
def real2chrom(weight, nBits):
    maxnum = 2**nBits
    weight=weight/20
    if weight<-1:
        weight=-1
    if weight>1:
        weight=1
    integerPart = int(maxnum * (weight + 1) / 2)
    if (integerPart == maxnum):
        integerPart -= 1
    chromosome = [int(d) for d in str(bin(integerPart))[2:]]
    while (len(chromosome) < nBits):
        chromosome.insert(0,0)
    indasstring=''.join(map(str, chromosome))
    chromosome=bin_to_gray(indasstring)
    output=[]
    for digit in chromosome:
        output.append(int(digit))
    return output

# input: concatenated list of binary variables
# output: list of real numbers representing those variables
def separatevariables(v, nBits, Chrom_length):
    sep = []
    for i in range (0,nBits*Chrom_length,nBits):
        sep.append(chrom2real(v[i:i+nBits], nBits))
    return sep

# calculates the loss of the current model
def loss_func(trainloader, model):
    criterion = torch.nn.CrossEntropyLoss()
    running_loss = 0.0
    num_batches = 0.0

    for data in trainloader:
        inputs, labels = data[0].to('cuda'), data[1].to('cuda')
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches

# calculates the fitness values of an individual
def calcFitness(individual, trainloader, model, nBits, Chrom_length): 
    weights_biases=separatevariables(individual, nBits, Chrom_length)
    
    weights = weights_biases[:1280]
    weights = np.asarray(weights, dtype=np.float32)
    new_weights = torch.from_numpy(weights.reshape(model.fc2.weight.size()[0], model.fc2.weight.size()[1])).to('cuda')
    model.fc2.weight = torch.nn.Parameter(new_weights)

    biases = weights_biases[1280:]
    biases = np.asarray(biases, dtype=np.float32)
    new_biases = torch.from_numpy(biases).to('cuda')
    model.fc2.bias = torch.nn.Parameter(new_biases)

    f1 = loss_func(trainloader, model)
    squared_weights_sum = sum(x ** 2 for x in weights)
    f2 = squared_weights_sum
    return f1,f2

# returns an array of subsets from the given set
def SmallLoaders(trainset, partition_size, batch_size):
    num_partitions = len(trainset) // partition_size
    
    # Create DataLoader instances for each partition
    data_loaders = []
    for i in range(int(num_partitions)):
        start = int(i * partition_size)
        end = int((i + 1) * partition_size)
    
        subset = torch.utils.data.Subset(trainset, range(start, end))
        data_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2)
        data_loaders.append(data_loader)
        
    return data_loaders

# Stops training when performance stops improving
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Our selected neural network
class my_CNN2(nn.Module):
    def __init__(self, num_classes=10):
        super(my_CNN2, self).__init__()
        
        # Convolutional Blocks
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.3)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout2d(0.5)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batch_norm5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batch_norm6 = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout2d(0.5)

        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.batch_norm_fc = nn.BatchNorm1d(128)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):        # Convolutional Blocks
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = F.relu(self.batch_norm4(self.conv4(x)))
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = F.relu(self.batch_norm5(self.conv5(x)))
        x = F.relu(self.batch_norm6(self.conv6(x)))
        x = self.maxpool3(x)
        x = self.dropout3(x)

            # Fully Connected Layers
        x = self.flatten(x)
        x = F.relu(self.batch_norm_fc(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

# Calculates the accuracy of the current model
def Accuracy(model, testloader, classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label.item()]] += 1
                total_pred[classes[label.item()]] += 1

    print('Accuracy of the network on the 10 000 test images: %f %%' % (100 * correct / total))

    for classname, correct_count in correct_pred.items():
        accuracy = 100* float(correct_count)/ total_pred[classname]
        print(f'Accuracy for: {classname:5s} is {accuracy:0.1f} %')
    return 100 * correct / total

# Surrogate model
class Surrogate(nn.Module):
    def __init__(self, input_size):
        super(Surrogate, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# Function which calculates the fitness values of an inividual using the surrogate model
def SurrogateEval(individual, model, nBits, Chrom_length): 
    model.eval()
    weights_biases=separatevariables(individual, nBits, Chrom_length)
    tensor = torch.tensor(weights_biases, dtype=torch.float32).to('cuda')
    
    result = model(tensor).detach().to('cpu').numpy()
    f1 = result[0]
    
    weights = weights_biases[:1280]
    squared_weights_sum = sum(x ** 2 for x in weights)
    f2 = squared_weights_sum
    return f1,f2