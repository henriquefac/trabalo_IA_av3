import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from classes.modelos.mlp.camadaSimples import Layer

class RedeNeural:
    def __init__(self, ap: float, p: int, camadas: list[int]):
        """
        Inicializa a rede neural.
        :param ap: Taxa de aprendizado.
        :param p: Número de entradas.
        :param camadas: Lista com o número de neurônios em cada camada.
        """
        self.p = p
        self.ap = ap
        self.layers: list[Layer] = []
        self.initCamadas(camadas)

    def initCamadas(self, camadas_list: list[int]):
        """
        Cria as camadas da rede neural.
        :param camadas_list: Lista com o número de neurônios em cada camada.
        """
        for i in range(len(camadas_list)):
            if i == 0:
                self.layers.append(Layer(camadas_list[i], self.p))
            else:
                self.layers.append(Layer(camadas_list[i], camadas_list[i - 1]))

    def Foward(self, x_entry: np.ndarray):
        """
        Passagem direta de dados pela rede (Forward Propagation).
        :param x_entry: Entrada para a rede neural.
        :return: Saída da rede neural.
        """
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].setCombinacaoLinear(x_entry)
                self.layers[i].setOutput()
            else:
                # Adiciona o bias
                entrada_com_bias = np.concatenate((self.layers[i-1].output, -np.ones((1, x_entry.shape[1]))), axis=0)
                self.layers[i].setCombinacaoLinear(entrada_com_bias)
                self.layers[i].setOutput()
        return self.layers[-1].output

    def Backward(self, y_true: np.ndarray, x_entry: np.ndarray):
        """
        Propagação reversa (Backward Propagation) e ajuste dos pesos.
        :param y_true: Rótulo verdadeiro.
        :param x_entry: Entrada para a rede neural.
        """
        # Atualiza os pesos da última camada
        lastLayer = self.layers[-1]
        erro = y_true - lastLayer.output
        lastLayer.setDelta(lastLayer.activation_derivada() * erro)
        
        # Ajuste dos pesos da última camada
        entrada_peso = np.concatenate((self.layers[-2].output, -np.ones((1, 1))), axis=0)
        lastLayer.ajustWeights(self.ap, entrada_peso)
            
        # Atualiza pesos das camadas intermediárias
        for i in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[i]
            proxima_layer = self.layers[i + 1]
            erro_propagado = proxima_layer.w[:, :-1].T @ proxima_layer.delta
            layer.setDelta(layer.activation_derivada() * erro_propagado)
            
            # Ajusta os pesos
            entrada_com_bias = np.concatenate((self.layers[i - 1].output, -np.ones((1, 1))), axis=0) if i > 0 else x_entry
            layer.ajustWeights(self.ap, entrada_com_bias)

    def Training(self, x_matrix: np.ndarray, y_array: np.ndarray, epoch: int = 5000, min_erro: float = 5e-10, paciencia: int = 10):
        """
        Treinamento da rede neural com Early Stopping.
        :param x_matrix: Matrizes de entradas (amostras).
        :param y_array: Vetor com os rótulos esperados.
        :param epoch: Número máximo de épocas.
        :param min_erro: Erro mínimo para parar o treinamento.
        :param paciencia: Número de épocas sem melhoria para interromper o treinamento.
        """
        eqm1 = 0
        eqm2 = 0
        melhor_erro = np.inf  # Melhor erro encontrado até agora
        epocas_sem_melhora = 0  # Contador de épocas sem melhoria
        melhores_pesos = [layer.w.copy() for layer in self.layers]  # Salva os pesos do melhor modelo

        for j in range(epoch):
            eqm1 = 0
            for i in range(x_matrix.shape[1]):
                amostra = x_matrix[:, i].reshape(-1, 1)
                last_output = self.Foward(amostra)
                eqm1 += np.sum((y_array[i] - last_output)**2) * 0.5
                self.Backward(y_array[i], amostra)
            eqm1 /= x_matrix.shape[1]
            
            # Verifica se o erro melhorou
            if eqm1 < melhor_erro:
                melhor_erro = eqm1
                epocas_sem_melhora = 0
                # Salva os pesos do melhor modelo
                melhores_pesos = [layer.w.copy() for layer in self.layers]
            else:
                epocas_sem_melhora += 1
                

                

            # Se o erro não melhorar por 'paciência' épocas consecutivas, interrompe
            if epocas_sem_melhora >= paciencia:
                print(f"Treinamento interrompido após {j+1} épocas devido à falta de melhoria.")
                break
            
            # Verifica a condição de erro mínimo
            if np.abs(eqm1 - eqm2) <= min_erro:
                break
            else:
                eqm2 = eqm1

        # Restaura os pesos do melhor modelo
        for i in range(len(self.layers)):
            self.layers[i].w = melhores_pesos[i]

    def sigmoide(self, x: np.ndarray) -> np.ndarray:
        """
        Função sigmoidal para converter qualquer valor em 1 ou -1.
        :param x: Valor de entrada.
        :return: Valor convertido em 1 ou -1.
        """
        return np.where(1 / (1 + np.exp(-x)) > 0.5, 1, -1)
    
    def Predict(self, x_entry: np.ndarray):
        """
        Realiza a predição com a rede neural.
        :param x_entry: A entrada para a rede neural.
        :return: A saída da rede neural.
        """
        return self.sigmoide(self.Foward(x_entry))



# rede integrada com a camada


import numpy as np

class NeuralNetwork():
    def __init__(self, ap: float, innneLayers: list[int], p: int = 2):
        self.p = p
        self.ap = ap
        # lista de layers
        self.layers: list[np.ndarray] = []
        # lista de outputs da combinação linear (u)
        self.u: list[np.ndarray] = []
        # lista dos outputs após a ativação (y)
        self.y: list[np.ndarray] = []
        # lista de deltas
        self.delta: list[np.ndarray] = []
        
        self.initLayers(innneLayers)
    
    def initLayers(self, layer_count: list[int]):
        # Inicializa as camadas e os vetores associados
        for i in range(len(layer_count)):
            if i == 0:
                self.layers.append(np.random.uniform(-0.5, 0.5, (layer_count[i], self.p + 1)))  # +1 para o viés
            else:
                self.layers.append(np.random.uniform(-0.5, 0.5, (layer_count[i], layer_count[i - 1] + 1)))
            
            self.u.append(np.ones((layer_count[i], 1)))  # Inicializa com 1
            self.y.append(np.ones((layer_count[i] + 1, 1)))  # Adiciona 1 para o viés
            self.delta.append(np.ones((layer_count[i], 1)))  # Inicializa os deltas

            # Definindo o valor do viés (último valor de y[i]) como -1
            self.y[i][-1, 0] = -1  # Aqui é o viés

    def activation(self, i: int) -> np.ndarray:
        """Função de ativação: tangente hiperbólica."""
        return np.tanh(self.u[i])
    
    def activation_derivada(self, i: int) -> np.ndarray:
        """Derivada da função de ativação."""
        return 1 - self.y[i][:-1, :]**2  # Exclui o último valor para o viés

    def linComb(self, i: int, entry: np.ndarray):
        """Calcula a combinação linear de entrada e pesos."""
        self.u[i] = np.dot(self.layers[i], entry).reshape(-1,1)  # Produto entre pesos e entrada

    def activatOutput(self, i: int):
        """Aplica a função de ativação e armazena o resultado."""
        self.y[i][:-1, :] = self.activation(i)  # Aplica a função de ativação sem incluir o viés

    def ajustWeights(self, i, entrada_peso):
        self.layers[i] = self.layers[i] + self.ap * (self.delta[i] @ entrada_peso.T)
    
    def Forward(self, x_entry: np.ndarray):
        """
        Propagação para frente através das camadas da rede.
        x_entry: entrada para a rede neural (exemplo de amostra)
        """
        for i in range(len(self.layers)):
            if i == 0:
                # Para a primeira camada, usamos diretamente a entrada
                self.linComb(i, x_entry)
                self.activatOutput(i)
            else:
                # Para camadas subsequentes, usamos o output da camada anterior (exceto o viés)
                self.linComb(i, self.y[i - 1])
                self.activatOutput(i)
    
    def Backward(self, x_entry: np.ndarray, y_true: np.ndarray):
        """
        Propagação para trás (backpropagation) e ajuste dos pesos.
        x_entry: entrada da rede (exemplo de amostra)
        y_true: saída esperada (rótulo verdadeiro)
        """
        # Calcula o erro na camada de saída
        erro = y_true - self.y[-1][:-1, :]  # Exclui o viés na camada de saída
        self.delta[-1] = self.activation_derivada(-1) * erro  # Atualiza o delta da última camada

        # Ajusta os pesos da última camada
        entrada_peso = self.y[-2]  # Entrada para a última camada (inclui viés)
        self.ajustWeights(-1, entrada_peso)

        # Retropropaga o erro para camadas ocultas
        for i in range(len(self.layers) - 2, -1, -1):
            # Calcula o erro propagado
            erro_propagado = self.layers[i + 1][:, :-1].T @ self.delta[i + 1]
            self.delta[i] = self.activation_derivada(i) * erro_propagado

            # Ajusta os pesos da camada atual
            if i > 0:
                entrada = self.y[i - 1]  # Entrada é o output da camada anterior (inclui viés)
            else:
                entrada = x_entry  # Para a primeira camada, a entrada é x_entry
            self.ajustWeights(i, entrada)
    
    def Train(self, x_matriz: np.ndarray, y_array: np.ndarray, epoch: int = 5000, min_error: float = 5e-10, paciencia: int = 10):
        """
        Treinamento da rede neural com Early Stopping.
        """
        eqm2 = 0
        melhor_erro = np.inf
        epocas_sem_melhora = 0
        melhores_pesos = [layer.copy() for layer in self.layers]

        for epoca in range(epoch):
            eqm1 = 0

            for i in range(x_matriz.shape[1]):
                amostra = x_matriz[:, i]
                self.Forward(amostra)
                last_output = self.y[-1][:-1, :]  # Exclui o viés para a camada de saída
                eqm1 += np.sum((y_array[i] - last_output) ** 2) * 0.5
                self.Backward(amostra, y_array[i])

            eqm1 /= x_matriz.shape[1]

            # Early stopping baseado no erro
            if eqm1 < melhor_erro:
                melhor_erro = eqm1
                epocas_sem_melhora = 0
                melhores_pesos = [layer.copy() for layer in self.layers]
            else:
                epocas_sem_melhora += 1

            # Critérios de parada
            if epocas_sem_melhora >= paciencia:
                print(f"Treinamento interrompido após {epoca + 1} épocas devido à falta de melhoria.")
                break
            if abs(eqm1 - eqm2) <= min_error:
                break

            eqm2 = eqm1

        # Carrega os melhores pesos encontrados
        self.layers = melhores_pesos
    def sigmoide(self, x: np.ndarray) -> np.ndarray:
        """
        Função sigmoidal para converter qualquer valor em 1 ou -1.
        :param x: Valor de entrada.
        :return: Valor convertido em 1 ou -1.
        """
        return np.where(1 / (1 + np.exp(-x)) > 0.5, 1, -1)
    
    def Predict(self, x_entry: np.ndarray):
        """
        Realiza a predição com a rede neural.
        """
        self.Forward(x_entry)
        return self.sigmoide(self.y[-1][:-1, :])  # Exclui o viés