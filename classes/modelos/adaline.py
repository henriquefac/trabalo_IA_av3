import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__name__)))
from classes.modelos.neuron import Neuron
# esse modelo possui apenas um neuronio

class NeuronADALINE(Neuron):
    def __init__(self, p: int, n: float, pr: float) -> None:
        super().__init__(p)
        # passo de aprendizagem
        # entre ]0 e 1]
        self.n = n

        # erro de convergência
        self.pr = pr

    def EQM(self, x_entry: np.ndarray, y_entry: np.ndarray):
        return np.sum((y_entry - self.Output(x_entry))**2)/(x_entry.shape[0]*2)


    def Output(self, x: np.ndarray) -> np.ndarray:
        
        return np.dot(x, self.w)

    def train(self, x_entry: np.ndarray, y_entry: np.ndarray, epochs: int = 1000):
        for epoch in range(epochs):
            eqm1 = self.EQM(x_entry, y_entry)

            for i in range(x_entry.shape[0]):
                erro = y_entry[i] - self.Output(x_entry[i])
                self.w += self.n * erro * x_entry[i]

            eqm2 = self.EQM(x_entry, y_entry)

            if np.abs(eqm1 - eqm2) < self.pr:
                break
                    
