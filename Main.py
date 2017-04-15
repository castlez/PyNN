import numpy as np
import scipy.special as sc
import csv
import random as rand
import time

# network structure
num_out = 10 # 0 - 9
num_in = 785 # 784 inputs and a bias
who = []
who_last_change = []
wih = []
wih_last_change = []

# hyper-parameters
learning_rate = 0.1
confusion = np.zeros((10, 10), dtype=np.int)  # holds data for confusion matrix
num_test_sets = 10000

# experiment parameters
num_hid = 100  # (20 | 50 | 100) hidden layer nodes
momentum = 0.9  # (0 | 0.25 | 0.5)
num_sets = 30000  # (30,000 | 15,000)


def main():
    global who_last_change
    global wih_last_change
    global who
    global wih

    # init
    results_train = []  # holds results of each epoch
    results_test = []  # holds results of each epoch


    print("Importing.... " + str(time.strftime("%H:%M:%S")))

    train, train_lines = import_csv("mnist_train.csv", num_sets)
    test, test_lines = import_csv("mnist_test.csv")

    wih = np.random.uniform(low=-0.05, high=0.05, size=(num_hid, num_in))
    wih_last_change = np.ndarray((num_hid, num_in))
    wih_last_change.fill(0)
    who = np.random.uniform(low=-0.05, high=0.05, size=(num_out, num_hid))
    who_last_change = np.ndarray((num_out, num_hid))
    who_last_change.fill(0)

    actual = np.empty(num_out)

    print("Get it started in hah...." + "\n" + str(time.strftime("%H:%M:%S")))

    # test network
    epoch = 0
    accuracy = test_network(test, test_lines)
    print("Epoch " + str(epoch) + "\naccuracy = " + str(accuracy) + " " + str(time.strftime("%H:%M:%S")))
    results_test.append([epoch, accuracy])
    results_train.append([epoch, test_network(train, train_lines)])

    # until epoch 50
    # train through sets
    while epoch < 50:
        epoch += 1
        for s in train:
            # feedforward
            outs, hid_act = feedforward(wih, who, s[1:])

            # backpropagate
            actual.fill(0.1)
            actual[int(np.round(s[0]*255))] = 0.9
            ins = s[1:].reshape(s.size - 1, 1)
            wih, who = backprop(wih, who, hid_act, outs, ins, actual.reshape(num_out,1))

        # test network
        accuracy = test_network(test, test_lines)
        print("Epoch " + str(epoch) + "\naccuracy = " + str(accuracy) + " " + str(time.strftime("%H:%M:%S")))
        results_test.append([epoch, accuracy])
        results_train.append([epoch, test_network(train, train_lines)])

    # write all results to (properly named) csv files
    with open("results/acc_" + str(learning_rate) + "_" + str(time.strftime("%Y%m%d-%H%M%S") + ".csv"),
              'w') as test_r_file:
        writer = csv.writer(test_r_file, lineterminator='\n')
        writer.writerow([str(num_hid), str(momentum), str(num_sets)])
        writer.writerow(["Epoch", "Training", "Test"])
        for i in range(0, epoch + 1):
            writer.writerow(results_train[i] + [results_test[i][1]])

    with open("results/conf_" + str(learning_rate) + "_" + str(time.strftime("%Y%m%d-%H%M%S")) + ".csv",
              'w') as conf_file:
        test_network(test, test_lines, True)
        writer = csv.writer(conf_file, lineterminator='\n')
        writer.writerow([str(num_hid), str(momentum), str(num_sets)])
        writer.writerow([" ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
        for i in range(0, 10):
            writer.writerow(np.concatenate(([i], confusion[i])))


# test the network on the given set
def test_network(setp, linesp, conf=False):
    count = 0
    for s in setp:
        output, hid_out = feedforward(wih, who, s[1:])
        if conf:
            confusion[int(np.round(s[0]*255))][output.argmax()] += 1
        if output.argmax() == int(np.round(s[0]*255)):
            count += 1

    return 100 * (count/linesp)


# feed inputs (weights input->hidden, weights hid->out, inputs)
def feedforward(wihp, whop, setp):
    # calculate output from input to hidden
    out_ih = np.dot(wihp, setp.reshape(setp.size, 1))
    out_ih = sc.expit(out_ih)

    # calculate output from hiddden to output
    out = np.dot(whop, out_ih)
    out = sc.expit(out)

    return out.reshape(out.size, 1), out_ih.reshape(out_ih.size, 1)


# backpropagate(weights input->hidden, weights hid->out, hidden activations, outputs, actual class)
def backprop(wihp, whop, hid_act, out_act, ins, actual):
    global who_last_change
    global wih_last_change

    # calculate errors
    erro, errh = calc_err(whop, out_act, hid_act, actual)

    # update weights
    whop, wihp = update_weights_all(erro, errh, whop, wihp, hid_act, ins)

    return wihp, whop


# calculate error at both the hidden and output layers
def calc_err(whop, out_act, hid_act, actual):
    # calculate errors
    erro = out_act * (1 - out_act) * (actual - out_act)  # 0.9 +, 0.1 -
    errh = hid_act * (1 - hid_act) * np.dot(whop.T, erro.reshape(num_out, 1))

    return erro.reshape(10, 1), errh


# update all weights in the network
def update_weights_all(erro, errh, whop, wihp, hid_act, ins):
    global wih_last_change
    global who_last_change

    change = np.dot(learning_rate * erro, hid_act.T) + momentum * who_last_change
    whop += change
    who_last_change = change

    change = np.dot(learning_rate * errh, ins.T) + momentum * wih_last_change
    wihp += change
    wih_last_change = change

    return whop, wihp


# gets the length in lines of a file
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass

    return i + 1


# formats a row from a training/test set
def preprocess(row):
    # return [int(x) for x in row[:1]] + [int(x)/255 for x in row[1:]]
    r = np.array(row).astype(np.float)
    p = np.empty(num_in)
    p.fill(255)

    return np.divide(r, p)


# if the number of sets is less than the whole
# data set, grab an even distribution of
# sets with actual answers 0-9 (lines/10)
def get_sets(setp, lines):
    ordered = setp[np.lexsort(np.fliplr(setp).T)]
    set_lines = int(lines/10)
    out_set = np.empty([lines, num_in+1], dtype=np.float)
    cur = 0
    last_set = 0
    i = 0
    for s in ordered:
        if cur < set_lines:
            out_set[i] = s
            cur += 1
            i += 1
        elif int(np.round(s[0]*255)) != last_set:
            cur = 0
            last_set = int(np.round(s[0]*255))
            out_set[i] = s
            cur += 1
            i += 1
    return out_set


# imports a csv file of either test or training data
# constrained by the number of sets, if provided
def import_csv(filename, ns=-1):
    reader = csv.reader(open(filename), delimiter=',')
    setp = np.empty([file_len(filename), 786], dtype=np.float)
    line = 0
    for row in reader:
        setp[line] = np.append(preprocess(row), [1.0])  # plus input for bias at the end
        line += 1
    if ns != -1:
        setp = get_sets(setp, ns)
    rand.shuffle(setp)

    return setp, (ns if ns != -1 else file_len(filename))


# run main if executed directly
if __name__ == "__main__":
    main()