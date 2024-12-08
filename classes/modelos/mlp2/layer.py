import numpy as np

class Layer:
    def __init__(self, neurons: int, p_entry: int = None):
        self.neurons = neurons
        # Inicializa os pesos (com o espaço extra para o bias)
        self.w = np.random.uniform(-0.5, 0.5, (neurons, p_entry + 1)) if p_entry else None
        # Inicializa o output com uma linha extra para o bias
        self.output: np.ndarray = np.zeros((neurons + 1, 1))  # O último valor será o bias
        self.combinacao_linear: np.ndarray = None
        self.delta: np.ndarray = None

    def activation(self, u: np.ndarray) -> np.ndarray:
        """Função de ativação: tangente hiperbólica."""
        return np.tanh(u)

    def activation_derivada(self) -> np.ndarray:
        """Derivada da função de ativação."""
        # não utiliza o bias
        return 1 - self.output[:-1]**2

    def setCombinacaoLinear(self, x_entry: np.ndarray):
        """Calcula a combinação linear e inclui o bias no final."""
        self.combinacao_linear = self.w @ x_entry


    def setOutput(self):
        """Calcula a saída com ativação, considerando o bias."""
        self.output[:-1] = self.activation(self.combinacao_linear)  # Aplica ativação nas entradas
        self.output[-1] = -1  # O valor do bias, mantido fixo como -1 (ou outro valor, se necessário)

    def setDelta(self, delta: np.ndarray):
        """Define o delta (gradiente local)."""
        self.delta = delta
        
    def ajustWeights(self, ap: float, yb: np.ndarray):
        """Ajusta os pesos com o delta e as entradas."""
        self.w += ap * self.delta @ yb.T
