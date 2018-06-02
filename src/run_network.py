
import sys
import mnist_loader
import network
import matplotlib.pyplot as plt
import pandas as pd

# Purpose: Output network learning data to a .csv file for reading in plot_data.py

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

print "Training Size: %s" % len(training_data)
print

NUM_INPUT_UNITS = 784
NUM_OUTPUT_UNITS = 10

hidden_layer_list = [10, 1000]
learning_rate_list = [1.0, 3.0]
mini_batch_list = [100, 500, 5000]
NUM_MODELS = 4
NUM_EPOCHS = 200
total_networks = (len(hidden_layer_list) * len(learning_rate_list) * len(mini_batch_list)) * NUM_MODELS

def run_network():
    network_counter = 0

    header_list = []

    f = open('TEST_FILE.csv', 'w')
    f.write("HU,LR,MB,Model#")
    for i in range(NUM_EPOCHS + 1):
        f.write(",E" + str(i))
    f.write('\n')

    for i in range(len(hidden_layer_list)):
        for j in range(len(learning_rate_list)):
            for k in range(len(mini_batch_list)):
                for n in range(NUM_MODELS):
                    print "Network %s: %s %s %s" % (n, hidden_layer_list[i], learning_rate_list[j], mini_batch_list[k])
                    net = network.Network([NUM_INPUT_UNITS, hidden_layer_list[i], NUM_OUTPUT_UNITS])
                    net.SGD(training_data, NUM_EPOCHS, mini_batch_list[k], learning_rate_list[j], test_data = test_data)

                    sum_of_vals = sum(net.performance_array)
                    num_of_epochs = len(net.performance_array)
                    avg_of_accuracies = sum_of_vals / num_of_epochs

                    f.write("%s, %s, %s, %s" % (hidden_layer_list[i], learning_rate_list[j], mini_batch_list[k], n))
                    for e in range(NUM_EPOCHS + 1):
                        f.write(", %s" % net.performance_array[e])
                    f.write('\n')

    f.close()


def main():
    run_network()

if __name__ == "__main__": main()
