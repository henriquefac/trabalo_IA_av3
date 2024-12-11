import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__name__)))

from Classes.dados.Data import DataHandler
from Classes.dados.Data import DataHandlerMLP
from Classes.modelos.single_neuron.perceptron import NeuronPeceptron
from Classes.modelos.single_neuron.adaline import NeuronADALINE
from Classes.modelos.mlp.newtwork import NetWork
class Rounds():
    def __init__(self, data: DataHandler) -> None:
        self.dh = data

    # metodo que faz uma rodada
    def round(self, perceptron: NeuronPeceptron):
        # dividir dados de treino e teste
        train, teste = self.dh.MonteCarlo()
        train_x, train_y = DataHandler.SepXY(train)
        teste_x, teste_y = DataHandler.SepXY(teste)

        #train_x = self.dh.nomrData(train_x)
        #teste_x = self.dh.nomrData(teste_x)
        
        train_x = np.concatenate((train_x, np.full((train_x.shape[0], 1), -1)), axis=1)
        teste_x = np.concatenate((teste_x, np.full((teste_x.shape[0], 1), -1)), axis=1)

        perceptron.train(train_x, train_y)
        return (teste_y, np.apply_along_axis(perceptron.SteepOut, 1, teste_x))
    


class RoundsPerceptron(Rounds):
    def __init__(self, data: DataHandler) -> None:
        self.dh = data

    def run_rounds(self, perceptrons: list[NeuronPeceptron]):
        results = []
        for perceptron in perceptrons:
            results.append(self.round(perceptron))
        return results


class RoundsADALINE(Rounds):
    def __init__(self, data: DataHandler) -> None:
        self.dh = data

    def run_rounds(self, perceptrons: list[NeuronADALINE]):
        results = []
        for perceptron in perceptrons:
            results.append(self.round(perceptron))
        return results

class RoundsMLP():
    def __init__(self, data: DataHandlerMLP):
        self.dh = data
    
    def round(self, newtWork: NetWork):
        train, teste = self.dh.MonteCarlo()
        train_x, train_y = DataHandler.SepXY(train)
        teste_x, teste_y = DataHandler.SepXY(teste)
        
        newtWork.train(train_x, train_y)
        return (teste_y, newtWork.predic(teste_x))
    
    def run_rounds(self, networks:list[NetWork]):
        results = []
        for network in networks:
            results.append(self.round(network))
        return results
    
class RoundsMLP_paralel():
    def __init__(self, data: DataHandlerMLP):
        self.dh = data
    
    def round(self, newtWork: NetWork):
        # Aqui você pode fazer o processo de treinamento e predição
        train, teste = self.dh.MonteCarlo()
        train_x, train_y = DataHandler.SepXY(train)
        teste_x, teste_y = DataHandler.SepXY(teste)
        
        newtWork.train(train_x, train_y)
        return (teste_y, newtWork.predic(teste_x))
    
    def run_rounds(self, networks: list[NetWork]):
        # Lista para armazenar os resultados das rodadas
        results = []
        
        # Usando ProcessPoolExecutor para rodar as rodadas em paralelo
        with ProcessPoolExecutor() as executor:
            # Submete as tarefas para o pool de processos
            futures = [executor.submit(self.round, network) for network in networks]
            
            # Coleta os resultados à medida que as rodadas terminam
            for future in as_completed(futures):
                results.append(future.result())
        
        return results


