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
    
class DataHandler():
    def __init__(self, path) -> None:
        self.data = np.loadtxt(path, delimiter=',')

        self.lines = self.data.shape[0]
        self.colls = self.data.shape[1]

        
    def Normalize(self):
        for i in range(self.lines):
                    self.data[i,:self.colls-1] = self.data[i,:self.colls-1] /np.linalg.norm(self.data[i,:self.colls-1]) 
        
    def MonteCarlo(self) -> tuple[np.ndarray]:
        """
        Divide os ddados em uma proporção 80/20%
        """
        index = np.random.permutation(self.data.shape[0])
        divide = int(len(index)*0.8)

        return self.data[index[:divide],:],self.data[index[divide:], :]
    def nomrData(self, entry:np.ndarray):
        return (entry - entry.mean(axis=1, keepdims=True)) / entry.std(axis=1, keepdims=True)
    @staticmethod
    def SepXY(dataMatrix):
        dataMatrix = dataMatrix.data if isinstance(dataMatrix, DataHandler) else dataMatrix
        return dataMatrix[:, :dataMatrix.shape[1]-1],dataMatrix[:, dataMatrix.shape[1]-1]
