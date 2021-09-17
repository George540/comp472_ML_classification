import matplotlib.pyplot as plt
import numpy as np
import os

def plot_instances_of_bbc_groups():
    group_directory = ["Datasets\BBC\\business", "Datasets\BBC\entertainment", "Datasets\BBC\politics", "Datasets\BBC\sport", "Datasets\BBC\\tech"]
    bbc_groups = ['Business', 'Entertainment', 'Politics', 'Sport', 'Tech']
    instances_of_group = []
    for directory in group_directory:
        count = 0
        for file in enumerate(os.listdir(directory)):
            count = count +1
        instances_of_group.append(count)
    ypos = np.arange(len(bbc_groups))
    plt.xticks(ypos, bbc_groups)
    plt.bar(ypos, instances_of_group, ec='black')
    for i in range(len(bbc_groups)):
        plt.text(i, instances_of_group[i], instances_of_group[i], ha='center', va='bottom')
    plt.show()

plot_instances_of_bbc_groups()
