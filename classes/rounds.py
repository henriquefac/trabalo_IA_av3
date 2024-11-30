import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__name__)))

from classes.DataHandler.data import DataHandler
from classes.modelos.perceptron import NeuronPeceptron

class RoundsPerceptron():
    def __init__(self, data: DataHandler) -> None:
        self.dh = data

    # metodo que faz uma rodada
    def round(self, perceptron: NeuronPeceptron):
        # dividir dados de treino e teste
        train, teste = self.dh.MonteCarlo()
        train_x, train_y = DataHandler.SepXY(train)
        teste_x, teste_y = DataHandler.SepXY(teste)
        
        train_x = np.concatenate((train_x, np.full((train_x.shape[0], 1), -1)), axis=1)
        teste_x = np.concatenate((teste_x, np.full((teste_x.shape[0], 1), -1)), axis=1)

        perceptron.train(train_x, train_y)
        return teste_y - np.apply_along_axis(perceptron.SteepOut, 1, teste_x)
    
    def run_rounds(self, perceptrons: list[NeuronPeceptron]):
        results = []
        for perceptron in perceptrons:
            results.append(self.round(perceptron))
        return np.array(results)