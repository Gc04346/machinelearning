from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, List


def cria_modelo(X:pd.DataFrame ,y:pd.Series ,min_samples_split:float, use_random_forest=False):
    """
        Retorna o modelo a ser usado.

        X: matriz (ou DataFrame) em que cada linha é um exemplo e cada coluna é uma feature (atributo/caracteristica) do mesmo
        y: para cada posição i, y[i] é a classe alvo (ground truth) do exemplo x[i]
        min_samples_split: define o mínimo de exemplos necessários para que um nodo da árvore efetue a divisão.
                            Esse número pode ser uma porcentagm proporcional ao total de exemplos (quando float)
                            ou um número inteiro representando o número absoluto de exemplos.
    """
    tree = DecisionTreeClassifier(min_samples_split=min_samples_split, random_state=1) if not use_random_forest else RandomForestRegressor(min_samples_split=min_samples_split, random_state=1)

    return tree.fit(X, y)



def divide_treino_teste(df:pd.DataFrame, val_proporcao_treino:float) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """
        A partir do DataFrame df, faz a divisão entre treino e teste obedecendo a proporção val_proporcao_treino.
        Essa proporção é um valor de 0 a 1, sendo que 1 representa 100%.

        Retorna uma tupla com o treino e teste separados
    """
    df_treino = df.sample(frac=val_proporcao_treino, random_state=1)

    df_teste = df.drop(df_treino.index)

    return df_treino,df_teste



def faz_classificacao(x_treino:pd.DataFrame, y_treino:pd.Series, x_teste:pd.DataFrame, y_teste:pd.Series, min_samples_split:float, use_random_forest=False) -> Tuple[List[float],float]:
    """
        Efetua a classificação, retornando:
            - O vetor y_predicted em que, para cada posição i,
             retorna o resultado previsto do exemplo representado
             por X_teste[i] que a classe alvo seria y_teste[i]. Esse y_predicted é
             o resultado retornado pelo método predict do modelo.
            - A acuracia (proporção de exemplos classificados corretamente)
    """
    model_dtree = cria_modelo(x_treino, y_treino, min_samples_split, use_random_forest)

    y_predicted = model_dtree.predict(x_teste)

    # No caso de usarmos regressao, acuracia é, na verdade, o MSE (Mean Squared Error).
    acuracia = np.sum(y_predicted==y_teste)/len(y_teste) if not use_random_forest else mean_squared_error(y_teste, y_predicted)


    return y_predicted,acuracia



def plot_performance_min_samples(X_treino,y_treino,X_teste,y_teste, use_random_forest=False):
    """
        Cria um gráfico em que o eixo x é a variação do parametro min_sample e,
        o eixo y, representa a acurácia.
    """
    #vetores que representam a acuracia no treino e no teste além do parametor usado
    arr_ac_treino = []
    arr_ac_teste = []
    arr_min_samples =[]

    for min_samples in np.arange(0.001,0.7,0.01):
        y_predicted, ac_teste = faz_classificacao(X_treino,y_treino,X_teste,y_teste, min_samples, use_random_forest)
        y_predicted, ac_treino = faz_classificacao(X_teste,y_teste,X_teste,y_teste, min_samples, use_random_forest)

        #adiciona a acuracia no treino, no teste e o parametro min_samples
        arr_ac_treino.append(ac_treino)
        arr_ac_teste.append(ac_teste)
        arr_min_samples.append(min_samples)

    #plota o resultado
    plt.plot(arr_min_samples,arr_ac_treino,"b--")
    plt.plot(arr_min_samples,arr_ac_teste,"r-")
