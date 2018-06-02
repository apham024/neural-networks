import mnist_loader
import network
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
from pylab import *

network_counter = 0
hidden_layer_list = [10, 1000]
learning_rate_list = [1.0, 3.0]
mini_batch_list = [100, 500, 5000]
NUM_MODELS = 4
NUM_EPOCHS = 200

def read_data():
    # getting data
    with open('TEST_FILE.csv', 'rb') as f:
        result = pd.read_csv("TEST_FILE.csv")
    # groups the data in the table by HU, LR, MB
    result = result.groupby(['HU','LR','MB']).mean()
    del result['Model#']
    return result

def plot_data(df):
    print df
    fig, axes = plt.subplots(nrows=len(hidden_layer_list), ncols=len(mini_batch_list), sharex=True, sharey=True)
    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            ax.set_xticks(np.arange(0, NUM_EPOCHS+1))
            ax.set_ylim(0, 1.0)
            ax.set_xlabel('# of Epochs',fontsize=8)
            ax.set_ylabel('Accuracy of Models',fontsize=8)
            ax.set_title("HU:%s, MB:%s" % (hidden_layer_list[i],mini_batch_list[j]),fontsize=10)
            lines = df.xs((hidden_layer_list[i], mini_batch_list[j]), level=[0, 2])
            lrs = df.index.levels[1].values
            # epoch_labels = xticks([NUM_EPOCHS],[NUM_EPOCHS])
            for line, lr in zip(lines.values, lrs):
                # print(line)
                ax.plot(line, label='LR = {}'.format(lr))
    plt.legend(loc='best')
    plt.show()


def main():
    df = read_data()
    plot_data(df)

if __name__ == "__main__":
     main()
