import mnist_loader
import network
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
from pylab import *

network_counter = 0

NUM_EPOCHS = 100
NUM_MODELS = 2
hidden_layer_list = [30]
learning_rate_list = [3.0]
mini_batch_list = [1, 5, 10, 100, 1000]

def read_data():
    # getting data
    with open('MB_TEST_FILE.csv', 'rb') as f:
        result = pd.read_csv("MB_TEST_FILE.csv")
    # groups the data in the table by HU, LR, MB
    result = result.groupby(['HU','LR','MB']).mean()
    del result['Model#']
    return result

def plot_data(df):
    print df
    fig, axes = plt.subplots(nrows=len(hidden_layer_list), ncols=len(learning_rate_list),squeeze=False)
    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            # ax.set_xticks(np.arange(0, NUM_EPOCHS+1))
            ax.set_ylim(0, 1.0)
            ax.set_xlabel('# of Epochs',fontsize=8)
            ax.set_ylabel('Accuracy of Models',fontsize=8)
            ax.set_title("HU:%s, LR:%s" % (hidden_layer_list[i],learning_rate_list[j]),fontsize=10)
            lines = df.xs((hidden_layer_list[i], learning_rate_list[j]), level=[0, 1])
            lrs = df.index.levels[2].values
            for line, lr in zip(lines.values, lrs):
                # print(line)
                ax.plot(line, label='MB = {}'.format(lr))
    plt.legend(loc='best')
    plt.show()

def main():
    df = read_data()
    plot_data(df)

if __name__ == "__main__":
     main()

'''
due Friday of finals week:
analysis of the relationship between HU, LR, MB
intro (1 page): describe the neural network and how it learns the dataset, what they are in general
(1 page each for HU, LR, MB): what they are, the effect they have on the networks, too few vs too much
1 paragraph defining, 2+ paragraphs after data manipulation and what happened for each
manipulate HU, LR, MB one at a time, include a figure with that data
'''
