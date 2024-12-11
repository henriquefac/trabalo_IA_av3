import numpy as np
from Classes.dados.Data import DataHandlerMLP
from Classes.rounds import RoundsMLP
dados = DataHandlerMLP(r"dados\spiral.csv")

# importar rede
from Classes.modelos.mlp.newtwork import NetWork
from Classes.modelos.single_neuron.adaline import NeuronADALINE
from Classes.modelos.single_neuron.perceptron import NeuronPeceptron

from Classes.rounds import RoundAll

listPerceptron = [NeuronPeceptron(2, 0.01) for _ in range(2)]
listAdaline = [NeuronADALINE(2) for _ in range(2)]
listMLP = [NetWork(2, [100], 1) for _ in range(2)]

rounds = RoundAll(dados)
rounds.run_rounds(listPerceptron, listAdaline, listMLP)
print(rounds.record)