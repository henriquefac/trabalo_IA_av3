import numpy as np
# deve apenas mandter as informações necessárias
# remover implementação de foward, backward e 
class Camada():
    def __init__(self,neurons,ap: float, p_entry: int = None) -> None:
        # quantidade de neurônios da entrada
        self.neurons = neurons
        # passo de aprendizagem
        self.ap = ap
        # quantidade de entradas esperadas ()
        self.p_entry: int
        # peso sinápticos
        self.w: np.ndarray = None
        if p_entry:
            self.p_entry = p_entry + 1
            self.initLayer() 

        # referência da camada entrior
        self.lastLayer: Camada = None
        # referência da próxima camada
        self.nextlayer: Camada = None

        # valores para treino
        # resultado de w @ x
        self.u: np.ndarray = None# dimenções [self.neurons x 1]
        # resultado de g(u)
        self.y: np.ndarray = None

        # delta
        self.delta: np.ndarray = None


    def setLastLayer(self, lastLayer):
        self.lastLayer = lastLayer

    def setNexLayer(self, nextLayer):
        self.nextlayer = nextLayer

    # incializa a partir da ultima camada
    def innerLayer(self):
        self.p_entry = self.lastLayer.neurons + 1
        self.w = np.random.normal(-0.5, 0.5, (self.neurons, self.p_entry))
    
    # recebe número de entradas previsto, provavelmente primeira camada
    def initLayer(self):
        self.w = np.random.normal(-0.5, 0.5, (self.neurons, self.p_entry))

    # recebe indivíduo com bias como ultimo elemento
    # x_entry[pentry x 1] (pentry inclui o bias)
    # passa rótulo que deve ser usado na última camada
    def Foward(self, x_entry: np.ndarray, d_: np.ndarray):
        if self.w is None:
            self.innerLayer()

        self.setU(x_entry)
        self.setY()
        # se possuir nexLayer, faz o foward da próxima camada
        if self.nextlayer:
            # concatenar -1 ao fim do vetor
            return self.nextlayer.Foward(np.concatenate((self.y,  -np.ones((1,1))), axis = 0), d_)
        else:
            return self.Backward(d_, np.concatenate((self.lastLayer.y,  -np.ones((1,1))), axis = 0))

    # função de ativação
    # tangente hiperbólica
    def activation(self, u):
        return (1 - np.exp(-u))/(1+np.exp(-u))
    def actDerivada(self, u):
        return 0.5 * (1 - self.activation(u)**2)
    # calcular u
    def setU(self, x_:np.ndarray):
        # w[N, p], x[p, 1] -> u[p,1] 
        self.u = np.dot(self.w, x_)
    
    # calcular saída y
    def setY(self):
        self.y = self.activation(self.u)

    # feito apenas na ultima camada
    # receber o valor real esperado e saída após ativação da ultima camada com adição do bias
    def Backward(self, d_: np.ndarray, yb: np.ndarray):
        # multiplicação elemento a elemento
        self.delta = self.actDerivada(self.u) *  (d_ - self.y)
        # corrigir w baseado em delta
        self.w = self.w + (self.ap * (self.delta @ yb.T))
        
        # calcular erro quadrático para todos os neurônios
        j = 0.5 * np.sum(((d_ - self.y).flatten())**2)
        
        
        # propagar erro para as próximas camadas
        return self.lastLayer.Propagate(j, np.concatenate((self.y,  -np.ones((1,1))), axis = 0))
    
    # propagar o delta e correção do w nas outras camadas
    # recebe o erro quadrático, matrix sem a coluna de bias
    # e o ultimo y calculado  com bias
    def Propagate(self, erro_sqr, yb: np.ndarray):
        # calcular delta
        wb = self.nextlayer.w[:, :-1].T
        self.delta = self.actDerivada(self.u) * wb @ self.nextlayer.delta
        
        # aplicar correção
        self.w = self.w + self.ap*self.delta @ yb.T
        if self.lastLayer:
            return self.lastLayer.Propagate(erro_sqr, np.concatenate((self.y,  -np.ones((1,1))), axis = 0)) 
        else:
            return erro_sqr
        pass

