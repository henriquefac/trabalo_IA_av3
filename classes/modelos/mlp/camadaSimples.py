import numpy as np

class Layer:
    def __init__(self, neurons: int, p_entry: int = None):
        self.neurons = neurons
        # Inicializar pesos se p_entry for fornecido
        self.w = np.random.uniform(-0.5, 0.5, (neurons, p_entry + 1)) if p_entry else None
        # Atributos para cálculos
        self.combinacao_linear: np.ndarray = None
        self.output: np.ndarray = None
        self.delta: np.ndarray = None

    def activation(self, u: np.ndarray) -> np.ndarray:
        """Função de ativação: tangente hiperbólica."""
        return np.tanh(u)
    def activation_derivada(self) -> np.ndarray:
        """Derivada da função de ativação."""
        return 1 - self.output**2

    def setCombinacaoLinear(self, x_entry: np.ndarray):
        """Calcula a combinação linear."""
        self.combinacao_linear = self.w @ x_entry

    def setOutput(self):
        """Calcula a saída com ativação."""
        self.output = self.activation(self.combinacao_linear)

    def setDelta(self, delta: np.ndarray):
        """Define o delta (gradiente local)."""
        self.delta = delta
        
    def ajustWeights(self, ap: float, yb: np.ndarray):
        self.w += ap * self.delta @ yb.T
