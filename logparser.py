"""
used to parse the log file and plot graphs
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--log_file", type=str, default='/raid/infolab/suma/gm/logs/foo.txt', help="log file to parse")
args = parser.parse_args()



def parse_log(log_file):
    """
    parse the log file and return the data
    """
    with open(log_file) as f:
        lines = f.readlines()
    loss = []
    acc = []
    for line in lines:
        if "Epoch" in line:
            linelist = line.strip().split()
            loss.append(float(linelist[-1]))
        elif "Accuracy:" in line:
            linelist = line.strip().split()
            acc.append(float(linelist[-1])*100)
            # print("Accuracy: ", acc)

    loss_avg = []
    k = 5
    for i in range(len(loss)):
        if i%5 == 0:
            loss_avg.append(np.mean(loss[i:i+k]))
            # k = i
    return loss_avg, acc
    # return loss, acc

def plot_graph_simple(data, save_path, type="loss"):
    """
    plot the graph
    """
    # loss, acc = data
    epochs = range(1, len(data)+1)
    plt.plot(epochs, data)
    # plt.plot(epochs, acc, label="Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel(type)
    # plt.legend()
    plt.savefig(save_path)
    plt.close()
    # plt.show()


if __name__ == "__main__":
    log_file = args.log_file
    loss,acc = parse_log(log_file)
    plot_graph_simple(loss, os.path.join(os.path.dirname(log_file), "loss.png"), type="loss")
    plot_graph_simple(acc, os.path.join(os.path.dirname(log_file), "accuracy.png"), type="accuracy")