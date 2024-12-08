import numpy as np
from classes.modelos.mlp.camada import Camada
from classes.modelos.mlp2.network import RedeNeural
from concurrent.futures import ProcessPoolExecutor
from classes.DataHandler.data import DataHandlerMLP
from functools import partial


import numpy as np

# Função de ativação ReLU e sua derivada
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

# Função de ativação sigmoide e sua derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

# Classe para a rede neural MLP
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicializa pesos e vieses
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)  # Pesos entre entrada e camada oculta
        self.bias_hidden = np.zeros((1, self.hidden_size))  # Vieses da camada oculta
        
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)  # Pesos entre camada oculta e saída
        self.bias_output = np.zeros((1, self.output_size))  # Vieses da camada de saída

    def forward(self, X):
        # Passagem para frente
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = relu(self.hidden_input)
        
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_input)
        
        return self.output

    def backward(self, X, y, learning_rate):
        # Erro na camada de saída
        output_error = y - self.output
        output_delta = output_error * sigmoid_deriv(self.output)
        
        # Erro na camada oculta
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * relu_deriv(self.hidden_output)
        
        # Atualização dos pesos e vieses
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - self.output))
                print(f'Epoch {epoch} | Loss: {loss}')




dh = DataHandlerMLP(r"C:\Users\henri\Documents\pythonProjcs\cadeira_IA_cirilo\av3\trabalho\dados\spiral.csv")
training, exp = dh.MonteCarlo()
x_tra, y_tra = DataHandlerMLP.SepXY(training)
x_tes, y_tes = DataHandlerMLP.SepXY(exp)



# Exemplo de uso
if __name__ == "__main__":
    dh = DataHandlerMLP(r"C:\Users\henri\Documents\pythonProjcs\cadeira_IA_cirilo\av3\trabalho\dados\spiral.csv")
    training, exp = dh.MonteCarlo()
    x_tra, y_tra = DataHandlerMLP.SepXY(training)
    x_tes, y_tes = DataHandlerMLP.SepXY(exp)

    # Criação da rede neural com 2 entradas, 4 neurônios na camada oculta e 1 saída
    mlp = MLP(input_size=2, hidden_size=4, output_size=1)

    # Treinamento
    mlp.train(x_tra, y_tra, epochs=10000, learning_rate=0.1)

    # Teste da rede
    print("\nResultados após treinamento:")
    print(mlp.forward(x_tes))


rede = RedeNeural(1e-2, 2, [2,2,1])
rede.Training(x_tra, y_tra, min_erro=5e-6)
output = rede.Predict(x_tes)
print(output)
resul = y_tes - output

print(len(resul[resul == 0])/len(y_tes))