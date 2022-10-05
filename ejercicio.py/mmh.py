import numpy as np

class HMM():
    def __init__(self, A, B, pi):
        self.A = A
        self.B = B
        self.pi = pi