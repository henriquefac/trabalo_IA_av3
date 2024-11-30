import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__name__)))
from classes.modelos.neuron import Neuron
# esse modelo possui apenas um neuronio


class NeuronPeceptron(Neuron):
    def __init__(self, p: int, n) -> None:
        super().__init__(p)
        # passo de aprendizagem
        # entre ]0 e 1]
        self.n = n

    def train(self, x_entry: np.ndarray, y_entry: np.ndarray, epochs: int = 1000):
        """
        Treina o Perceptron usando o conjunto de entrada e saída fornecido.
        """
        # Adiciona o bias ao conjunto de entradas


        for _ in range(epochs):
            flag = True  # Assume que todas as amostras estão corretas
            for i in range(x_entry.shape[0]):
                # Calcula o erro para a amostra i
                erro = y_entry[i] - self.SteepOut(x_entry[i, :])

                flag = erro == 0
                if not flag:
                    self.w += self.n * erro * x_entry[i, :]

            # Para o treinamento se todas as amostras forem classificadas corretamente
            if flag:
                break

            