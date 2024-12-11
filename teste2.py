import numpy as np
from Classes.dados.Data import DataHandlerMLP
import matplotlib.pyplot as plt
from Classes.rounds import RoundsMLP
dados = DataHandlerMLP(r"dados\spiral.csv")

# importar rede
from Classes.modelos.mlp.newtwork import NetWork
from Classes.modelos.single_neuron.adaline import NeuronADALINE
from Classes.modelos.single_neuron.perceptron import NeuronPeceptron


train, teste = dados.MonteCarlo()
train_x, train_y = DataHandlerMLP.SepXY(train)
teste_x, teste_y = DataHandlerMLP.SepXY(teste)

rede = NetWork(2,[100],1, learningRate=1e-1)
rede.train(train_x, train_y)

plt.plot(rede.history['loss'])
plt.title('Curva de Aprendizado')
plt.xlabel('Épocas')
plt.ylabel('EQM')
plt.show()