import numpy as np
import csv
import random as rand
import time

# network structure
num_out = 10 # 0 - 9
num_in = 785 # 784 inputs and a bias
num_hid = 20  # (20, 50, 100) hidden layer nodes

# hyper-parameters
learning_rate = 0.1
momentum = 0.9


def main():
    print("Get it started in hah...")

    # init

    results_train = []  # holds results of each epoch
    results_test = []  # holds results of each epoch
    confusion = np.zeros((10, 10), dtype=np.int)  # holds data for confusion matrix

    print("Get it started in hah....")

    last_accuracy = 0.0
    train, train_lines = import_csv("mnist_train.csv")
    test, test_lines = import_csv("mnist_test.csv")

    wih = np.random.uniform(low=-0.05, high=0.05, size=(train.shape[1] - 1, num_hid))
    who = np.random.uniform(low=-0.05, high=0.05, size=(num_hid, num_out))

    # until epoch 50
    # train through sets
    epoch = 0
    while epoch < 50:

        for set in train:
            # feedforward
            outs = feedforward(wih, who, set)

            # backpropagate(weights input->hidden, weights hid->out, outputs, actual class)
            whi, who = backprop(wih, who, outs, set[0])


        # test network
        accuracy = test_network(test, test_lines)
        print("Epoch " + epoch + "\naccuracy = " + accuracy)
        results_test.append([epoch, accuracy])
        results_train.append([epoch, test_network(train, train_lines)])

    # write all results to (properly named) csv files
    with open("results/acc_" + str(learning_rate) + "_" + str(time.strftime("%Y%m%d-%H%M%S")) + ".csv",
              'w') as test_r_file:
        writer = csv.writer(test_r_file, lineterminator='\n')
        writer.writerow(["Epoch", "Training", "Test"])
        for i in range(0, epoch + 1):
            writer.writerow(results_train[i] + [results_test[i][1]])

    with open("results/conf_" + str(learning_rate) + "_" + str(time.strftime("%Y%m%d-%H%M%S")) + ".csv",
              'w') as conf_file:
        test_network(test, test_lines, True)
        writer = csv.writer(conf_file, lineterminator='\n')
        writer.writerow([" ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
        for i in range(0, 10):
            writer.writerow(np.concatenate(([i], confusion[i])))


# test the network on the given set
def test_network(setp, linesp):
    return 0.0


def feedforward(wihp, whop, setp):
    return []


# backpropagate(weights input->hidden, weights hid->out, outputs, actual class)
def backprop(wihp, whop, outsp, setp):
    return [],[]


# gets the length in lines of a file
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


# formats a row from a training/test set
def get_row(row):
    return [int(x) for x in row[:1]] + [int(x)/255 for x in row[1:]]


# imports a csv file of either test or training data
def import_csv(filename):
    reader = csv.reader(open(filename), delimiter=',')
    set = np.empty([file_len(filename), 786])
    line = 0
    for row in reader:
        set[line] = get_row(row) + [1]  # plus input for bias at the end
        line += 1
    rand.shuffle(set)
    return set, line


# run main if executed directly
if __name__ == "__main__":
    main()