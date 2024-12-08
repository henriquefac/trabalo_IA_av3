import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from classes.modelos.mlp2.layer import Layer


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
                # Passa o output da camada anterior, já com o bias
                self.layers[i].setCombinacaoLinear(self.layers[i - 1].output)
                self.layers[i].setOutput()
        return self.layers[-1].output[:1]

    def Backward(self, y_true: np.ndarray, x_entry: np.ndarray):
        """
        Propagação reversa (Backward Propagation) e ajuste dos pesos.
        :param y_true: Rótulo verdadeiro.
        :param x_entry: Entrada para a rede neural.
        """
        # Atualiza os pesos da última camada
        lastLayer = self.layers[-1]
        erro = y_true - lastLayer.output[:-1]
        lastLayer.setDelta(lastLayer.activation_derivada() * erro)
        
        # Ajuste dos pesos da última camada
        # acessa o output da camadaanterior para calcular o erro 
        # sem excluir bias
        entrada_peso = self.layers[-2].output
        lastLayer.ajustWeights(self.ap, entrada_peso)
            
        # Atualiza pesos das camadas intermediárias
        for i in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[i]
            proxima_layer = self.layers[i + 1]
            erro_propagado = proxima_layer.w[:, :-1].T @ proxima_layer.delta
            layer.setDelta(layer.activation_derivada() * erro_propagado)
            
            # Ajusta os pesos
            entrada_com_bias = self.layers[i-1].output if i > 0 else x_entry
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
        outputs = np.array([self.sigmoide(self.Foward(x_entry[:, i].reshape(-1, 1))).item() for i in range(x_entry.shape[1])])
        
        return outputs 


if __name__ == "__main__":
    print("aqui")