import numpy as np
from Classes.dados.Data import DataHandlerMLP
from Classes.rounds import RoundsMLP
dados = DataHandlerMLP(r"dados\spiral.csv")
treino, teste = dados.MonteCarlo()
treino_x, treino_y = DataHandlerMLP.SepXY(treino)
teste_x, teste_y = DataHandlerMLP.SepXY(teste)


# importar rede
from Classes.modelos.mlp.newtwork import NetWork


list_mlp = [NetWork(2, [100], 1) for _ in range(500)]

# Criando a instância da classe de rodadas e chamando o método run_rounds
rounds_mlp = RoundsMLP(dados)
resultados_mlp = rounds_mlp.run_rounds(list_mlp)