# chamar dados
import numpy as np
import numpy as np

class DataHandlerMLP():
    def __init__(self, path):
        self.data = np.loadtxt(path, delimiter=',').T
        self.amostras = self.data.shape[1]
        self.componenetes = self.data.shape[0]
        # sempre normaliza
        self.Normalize()
    def Normalize(self):
        """
        Normaliza os dados usando Min-Max para cada característica.
        Escala para o intervalo [-1, 1].
        """
        data_without_last_row = self.data[:-1, :]  # Exclui a última linha

        # Calcula min e max de cada característica
        min_vals = np.min(data_without_last_row, axis=1, keepdims=True)
        max_vals = np.max(data_without_last_row, axis=1, keepdims=True)

        # Evita divisão por zero
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1

        # Normaliza as características para [-1, 1]
        self.data[:-1, :] = 2 * ((data_without_last_row - min_vals) / ranges) - 1

    def MonteCarlo(self):
        """
        Divide os ddados em uma proporção 80/20%
        """
        index = np.random.permutation(self.amostras)
        division = int(0.8 * self.amostras)
        
        return self.data[:, index[:division]], self.data[:, index[division:]]
    
    @staticmethod
    def SepXY(matrix: np.ndarray):
        """
        Separa as amostras dos rótulos
        """
        return np.concatenate((matrix[:-1, :], -np.ones((1,matrix.shape[1]))), axis=0), matrix[-1, :]
    
dados = DataHandlerMLP(r"dados\spiral.csv")
treino, teste = dados.MonteCarlo()
treino_x, treino_y = DataHandlerMLP.SepXY(treino)
teste_x, teste_y = DataHandlerMLP.SepXY(teste)


# função de ativação
def activation(u):
    return np.tanh(u)
def derivada_activation(u):
    return 1 - u**2

def sigmoide(x: np.ndarray) -> np.ndarray:
        """
        Função sigmoidal para converter qualquer valor em 1 ou -1.
        :param x: Valor de entrada.
        :return: Valor convertido em 1 ou -1.
        """
        return np.where(1 / (1 + np.exp(-x)) > 0.5, 1, -1)


def MLP(X: np.ndarray,Y: np.ndarray,p: int, innerLayers:list[int], m:int, 
        learningRate=1e-1, epoca=5000, erro_min = 1e-8):
    layers = [p] + innerLayers + [m]
    # matriz de pesos sinápticos, onde a utima coluna é o bias
    # inicializa pesos
    w: list[np.ndarray] = [np.random.random_sample((layers[i+1], layers[i] + 1)) -.5 for i in range(len(layers)-1)]
    u: list[np.ndarray] = [None] * (len(layers) -1)
    y: list[np.ndarray] = [-np.ones((layers[i+1]+1, 1)) for i in range(len(layers)-1)]
    
    delta: list[np.ndarray] = [None] * (len(layers) -1)
    
    eqm_anterior = float('inf')
    # treino
    for i in range(epoca):
        eqm = 0
        # forward
        for n in range(X.shape[1]):
            amostra = X[:, [n]]
            d = Y[n]

            # Forward
            for j in range(len(w)):
                entrada = amostra if j == 0 else y[j - 1]
                u[j] = w[j] @ entrada
                y[j][:-1] = activation(u[j])  # Calcula ativação e preserva bias

            # backward
            for j in range(len(w)-1, -1,-1):
                if j+1 == len(w):
                    erro = (d - y[j][:-1]) 
                    delta[j] = derivada_activation(y[j][:-1]) * erro
                    w[j] += learningRate * (delta[j] @ y[j-1].T) # y da camada atrás incluindo bias
                elif j == 0:
                    # matriz de pesos sem bias da camada a frente
                    wb = w[j+1][:,:-1].T
                    delta[j] = derivada_activation(y[j][:-1]) * (wb @ delta[j+1])
                    w[j] += learningRate * (delta[j] @ amostra.T)
                else:
                    wb = w[j+1][:,:-1].T
                    delta[j] = derivada_activation(y[j][:-1]) * (wb @ delta[j+1])
                    w[j] += learningRate * (delta[j] @ y[j-1].T) # y da camada atrás incluindo bias

        # calcular eqm
        for n in range(X.shape[1]):
            amostra = X[:, [n]]
            d = Y[n]
            # forward
            for j in range(len(w)):
                entrada = amostra if j == 0 else y[j - 1]
                u[j] = w[j] @ entrada
                y[j][:-1] = activation(u[j])  # Calcula ativação e preserva bias
            # deiferença entre rotulo e vetor de ativação da ultima camada
            # elevado ao quadrado
            eqm += np.sum((d - y[-1][:-1])**2)
        eqm /= 2*X.shape[1]
        print(f"Epoca: {j}; EQM: {eqm}")
        if np.abs(eqm-eqm_anterior) <= erro_min:
            break
        else:
            eqm_anterior = eqm
        
    return w
        

def predict(X_teste: np.ndarray, w: list[np.ndarray]) -> np.ndarray:
    """
    Realiza a propagação forward para prever os resultados de entradas fornecidas.
    
    :param X_teste: Matrizes de entrada (uma amostra por coluna).
    :param w: Lista de matrizes de pesos treinados.
    :return: Saídas previstas pela MLP.
    """
    # Número de amostras no conjunto de teste
    num_amostras = X_teste.shape[1]
    
    # Inicializa as ativações de cada camada (adicionando bias)
    y = [-np.ones((w[i].shape[0] + 1, 1)) for i in range(len(w))]  # +1 para o bias na última camada
    u = [None] * len(w)
    
    # Armazena as previsões
    previsoes = []
    
    for n in range(num_amostras):
        # Seleciona a amostra (coluna)
        amostra = X_teste[:, [n]]
        
        # Forward pass
        for j in range(len(w)):
            entrada = amostra if j == 0 else y[j - 1]
            u[j] = w[j] @ entrada
            y[j][:-1] = activation(u[j])  # Calcula ativação e preserva bias
            
        # Adiciona a previsão da última camada (sem bias)
        previsoes.append(sigmoide(y[-1][:-1]))
    
    # Converte a lista de previsões para um array numpy
    return np.hstack(previsoes)


# Treinamento
pesos_treinados = MLP(treino_x, treino_y, 2, [100], 1, learningRate=1e-1)

# Previsão
resultado = predict(teste_x, pesos_treinados).flatten()

# Mostrar resultados
print("Previsões:")
print(resultado)
dif = resultado - teste_y
print(len(dif[dif == 0])/len(dif))