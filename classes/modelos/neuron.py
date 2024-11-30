import numpy as np

class Neuron:
    def __init__(self, p: int) -> None:
        """
        Inicializa o Perceptron com pesos aleatórios no intervalo [-0.5, 0.5].
        p: Número de entradas (dimensão do vetor de entrada, sem contar o bias).
        """
        self.w = np.random.uniform(-0.5, 0.5, p + 1)  # Inclui o peso do bias

    def SteepOut(self, entrada: float) -> int:
        """
        Função degrau para classificação.
        Retorna 1 se entrada >= 0, senão 0.
        """
        return 1 if self.Output(entrada) >= 0 else 0

    def Output(self, x: np.ndarray) -> int:
        """
        Calcula a saída do Perceptron para uma entrada x.
        Adiciona o bias (-1) ao vetor de entrada antes de calcular.
        """

        
        # Calcula a soma ponderada
        soma_ponderada = np.dot(x, self.w)
        
        # Aplica a função degrau
        return soma_ponderada

