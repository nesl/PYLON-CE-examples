import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import random
from collections import defaultdict

import torch.nn as nn
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PlotHelper():
    def __init__(self):
        self.reset()

    def reset(self):
        self._f = None
        self._ax = None
        self.kvals = defaultdict(list)

    def add(self, **kval):
        for k, v in kval.items():
            self.kvals[k].append(v)

    @property
    def fig(self):
        if self._f is None:
            self.new()
        return self._f

    @property
    def ax(self):
        if self._ax is None:
            self.new()
        return self._ax

    def new(self):
        self._f, self._ax = plt.subplots(1,1)
        plt.ion()
        self.fig.show()

    def show(self):
        names = []
        self.ax.clear()
        for k, v in self.kvals.items():
            names.append(k)
            self.ax.plot(v)
        self.ax.legend(names)
        self.fig.canvas.draw()


def plot_confusion_matrix(y_true, y_pred, class_names):
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], 
                         index = [i for i in class_names],
                        columns = [i for i in class_names])
    plt.figure(figsize = (9,7))
    sn.heatmap(df_cm, annot=True)


def prim_test(model, test_loader):
    '''
    Evaluation function used for primitive activity classification.
    params:
        cm - (Boolean) True to plot the confusion matrix.
    '''
    model.eval()
    model.cpu()
    accuracy = 0
    total = len(test_loader.dataset)
    class_correct = [0. for i in range(model.n_class)]
    class_total = [0. for i in range(model.n_class)]
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predict = torch.max(outputs.data, 1)
            correct = predict == labels
            accuracy += correct.sum().item() / total
            c = correct.squeeze()

            y_true.extend(predict.data.cpu().numpy())
            y_pred.extend(labels.data.cpu().numpy())
            
            for i in range(len(inputs)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    print('Accuracy of the network on the test data: %d %%' % (
      100 * accuracy))
    for i in range(model.n_class):
        print('Accuracy of activity %2s : %2d %%' % (
        i, 100 * class_correct[i] / class_total[i]))
        
    return y_true, y_pred, accuracy



def prim_train(model, train_loader, test_loader, optimizer, epoch, device="cpu"):
    '''Train a primitive activity classifier'''
    criterion = nn.CrossEntropyLoss()
    plot_loss = PlotHelper()
    
    for epoch in range(epoch):
        model.train()
        model.to(device)
        for data, label in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
            label = label.to(device)
            
            output = model(data)
            loss = criterion(output, label)
            
            dampner = loss.detach().cpu()
            plot_loss.add(loss = dampner)
        
            loss.backward()
            optimizer.step()

        print("Epoch:", epoch)
        prim_test(model, test_loader) 
    plot_loss.show()

