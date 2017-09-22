#encoding:utf8
'''
Activation functions
'''
import numpy as np

class Sigmoid():
    def __new__(cls, z):
        return 1.0/(1+np.exp(-z))

    @classmethod
    def prime(cls, z):
        return cls(z) * (1 - cls(z))

class ReLu():
    def __new__(cls, z):
        def func(x):
            if x<=0:
                return 0
            return x
        return np.vectorize(func)(z)

    @classmethod
    def prime(cls, z):
        def func(x):
            if x>0:
                return 1
            return 0
        return np.vectorize(func)(z)

class LeakyReLu():
    def __new__(cls, z):
        def func(x):
            if x<=0:
                return 0.1
            return x
        return np.vectorize(func)(z)

    @classmethod
    def prime(cls, z):
        def func(x):
            if x>0:
                return 1
            return 0.1
        return np.vectorize(func)(z)