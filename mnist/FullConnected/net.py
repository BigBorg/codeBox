#encoding:utf8
import pickle
import os
import atexit
import numpy as np
from load_mnist import load_train_images, load_train_labels
from numpy import newaxis
from matplotlib import pyplot as plt, animation
from activations import Sigmoid
from normalizer import gaussian

# if "train.pck" in os.listdir():
#     with open("train.pck", "rb") as f:
#         data = f.read()
#     images, labels = pickle.loads(data)
# else:
#     images = load_train_images()
#     print("Load labels...")
#     labels = load_train_labels()
#     with open("train.pck", "wb") as f:
#         data = f.write(pickle.dumps((images[:1000], labels[:1000])))

class Net(object):
    '''
    Sigmoid as activation function
    '''
    def __init__(self, shape=None, normalizer=None, activation_func=None):
        self.shape = np.array([28*28, 16, 10]) if shape is None else np.array(shape)
        self.normalizer = gaussian if normalizer is None else normalizer
        self.activation_func = Sigmoid if activation_func is None else activation_func

        self.weights = [np.random.randn(row, col) for row, col in zip(self.shape[1:], self.shape[:-1])]
        self.bias = [np.random.randn(i) for i in self.shape[1:]]

        # To track training process
        plt.ion()
        self._monitor = {
            "cost_train":[],
            "accuracy_train":[],
            "cost_valid": [],
            "accuracy_valid": [],
        }
        self.figure = plt.figure()
        self.axe = self.figure.add_subplot(1,1,1)

    @staticmethod
    def _y2i(y):
        '''
        :param ys: np.array([1]) because it's to be used as parameter of np.apply_along_axis, the length of y should be 1
        :return: array of eye vector
        '''
        activations = np.zeros((10))
        activations[y[0]] = 1
        return activations

    def _labels_2_activations(self, labels):
        '''
        :param labels: array of int
        :return: array of corresponding activation vector
        '''
        return np.apply_along_axis(Net._y2i, 1, labels[:, newaxis])

    def _cost(self, y, result):
        '''
        quadratic cost function
        :param y: array of int with shape (10,), true activation
        :param result:  array of int with shape (10,), predicted activation
        :return:  cost
        '''
        return np.sum(np.square((result - y)))/2

    def _feed_forward(self, image):
        '''
        :param image: np array with two dimensions
        :return:
        '''
        a = image.reshape((self.shape[0],))
        a = self.normalizer(a)
        activations = []
        zs = []
        activations.append(a)  # input layer's activation value equal to normailized image data
        zs.append([])       # input layer does not have z value. [] as a placeholder
        for w, b in zip(self.weights, self.bias):
            z = w.dot(a) + b
            a = self.activation_func(z)
            activations.append(a)
            zs.append(z)
        self.activations = np.array(activations)
        self.zs = np.array(zs)
        return a

    def _back_propagate(self, image, yeye):
        self._feed_forward(image)
        last_error = (self.activations[-1] - yeye) * self.activation_func.prime(self.zs[-1])
        errors = [last_error]
        for w, z in zip(self.weights[-1:0:-1], self.zs[-2:0:-1]):
            last_error = w.T.dot(last_error)*self.activation_func.prime(z)
            errors.insert(0, last_error)
        dc_over_dw = [np.tile(a, (error.shape[0],1))*error[:, newaxis] for a, error in zip(self.activations[0:-1], errors)]
        dc_over_db = [error for error in errors]
        return dc_over_dw, dc_over_db

    def SGD(self, images, ys, batch_size=10, learning_rate=1, epoch=1, evaluate=True, valid_images=None, valid_y=None):
        '''
        :param images: 2D
        :param ys: [1,3,0]
        :param batch_size: 
        :return: 
        '''
        # making batches
        train_data = np.array([(image, y) for image, y in zip(images, ys)])
        np.random.shuffle(train_data)

        start_index = [i for i in range(0, train_data.shape[0], batch_size)]
        last_batch = None
        if not train_data.shape[0] % batch_size == 0:
            last_batch = train_data[start_index.pop():]
        batches = [train_data[i:i+batch_size] for i in start_index]
        if last_batch is not None:
            batches.append(last_batch)

        for i in range(epoch):
            for i, batch in enumerate(batches):
                dws = []
                dbs = []
                for image, y in batch:
                    dw, db = self._back_propagate(image, self._y2i([y]))
                    dws.append(dw)
                    dbs.append(db)
                dws = np.array(dws)
                dbs = np.array(dbs)

                self.weights = self.weights - learning_rate * np.average(dws, 0)
                self.bias = self.bias - learning_rate * np.average(dbs, 0)

                if evaluate and i % 50 == 0:
                    cost_train, accuracy_train = self.evaluate(images[:10000], ys[:10000])
                    self._monitor["cost_train"].append(cost_train)
                    self._monitor["accuracy_train"].append(accuracy_train)

                    if valid_images is not None:
                        cost, accuracy = self.evaluate(valid_images, valid_labels)
                        self._monitor["cost_valid"].append(cost)
                        self._monitor["accuracy_valid"].append(accuracy)
                    self.monitor()

        return batches

    def predict(self, image, show_img=False):
        '''
        :param image: np array with two dimensions
        :return:
        '''
        a = self._feed_forward(image)
        predicted = np.argmax(a)
        print(predicted)
        if show_img:
            plt.imshow(image, cmap="gray")
            plt.title(str(predicted))
            plt.show()
            print("--------")
        return predicted

    def evaluate(self, images, y):
        '''
        :param images: shape: (number_of_images, rows, cols)
        :param y: array of labels with dtyp equals to int
        :return:
        '''
        predicted_activation = [self._feed_forward(image) for image in images]
        true_activation = self._labels_2_activations(y)
        cost = 0
        for prediction, true_val in zip(predicted_activation, true_activation):
            cost += self._cost(true_val, prediction)
        cost = cost/true_activation.shape[0]
        print("Cost: {}".format(cost))

        predicted_class = [np.argmax(ele) for ele in predicted_activation]
        accuracy = [1 if predicted==true_class else 0 for predicted, true_class in zip(predicted_class, y)]
        accuracy = float(np.sum(accuracy))/len(accuracy)
        print("Accuracy: {}".format(accuracy))
        return cost, accuracy

    def save(self, filename):
        with open(filename, "wb") as f:
            f.write(pickle.dumps(self))

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            net = pickle.loads(f.read())
        return net

    def monitor(self):
        self.axe.clear()
        self.axe.hlines(1, 0, len(self._monitor["cost_train"]))
        for key in self._monitor:
            self.axe.plot(self._monitor[key], label=key)
            self.axe.legend()
        plt.pause(0.1)
        return self.axe


if __name__ == "__main__":
    images = load_train_images()
    labels = load_train_labels()
    train_image, train_labels = images[:50000], labels[:50000]
    valid_images, valid_labels = images[50000:], labels[50000:]

    if os.path.exists(os.path.abspath("net.pkl")):
        net = Net.load("net.pkl")
    else:
        net = Net()

    # # Training
    # try:
    #     net.SGD(train_image, train_labels, learning_rate=0.75, batch_size=30, epoch=10, valid_images=valid_images, valid_y=valid_labels)
    # except KeyboardInterrupt:
    #     print("Exit...")
    #     net.save("net.pkl")
    #     plt.show()

    # Testing
    net = Net.load("net.pkl")
    for i in range(10):
        net.predict(valid_images[i], show_img=True)