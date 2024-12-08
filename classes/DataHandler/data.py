import numpy as np

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
    
class DataHandlerMLP():
    def __init__(self, path):
        self.data = np.loadtxt(path, delimiter=',').T
        self.amostras = self.data.shape[1]
        self.componenetes = self.data.shape[0]
        # sempre normaliza
        self.Normalize()
    def Normalize(self):
        """
        Normaliza cada coluna (amostra) da matriz de dados, exceto a última linha.
        """
        # Exclui a última linha para o cálculo da norma (não deve ser normalizada)
        data_without_last_row = self.data[:-1, :]  # Exclui a última linha

        # Calcula a norma de cada coluna (amostra) sem considerar a última linha
        norms = np.linalg.norm(data_without_last_row, axis=0)  # Norma de cada coluna (sem a última linha)

        # Normaliza os dados, evitando divisão por zero (caso algum vetor tenha norma zero)
        norms[norms == 0] = 1  # Substitui as normas zero por 1 para evitar divisão por zero
        self.data[:-1, :] = self.data[:-1, :] / norms  # Normaliza todas as colunas pela sua norma (exceto a última linha)

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