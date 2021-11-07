import pandas as pd
from typing import Union
from math import log
from pandas.api.types import is_numeric_dtype, is_bool_dtype


def entropia(df_dados:pd.DataFrame, nom_col_classe:str) -> float:
    """
        Calcula a entropia de acordo com df_dados (DataFrame) e a classe.

        df_dados: Dados a serem considerados para o calculo da entropia
        nom_col_classe: nome da coluna (em df_dados) que representa a classe
    """
    #ser_count_col armazena, para cada valor da classe, a sua quantidade
    ser_count_col = df_dados[nom_col_classe].value_counts()
    num_total = len(df_dados)
    entropia = 0

    #Navegando em ser_count_col para fazer o calculo da entropia
    for val_atr,count_atr in ser_count_col.iteritems():
        val_prob = count_atr / num_total
        entropia += -(val_prob * log(val_prob, 2))
    return entropia


def ganho_informacao_condicional(df_dados: pd.DataFrame, val_entropia_y:Union[int,float,str,bool], nom_col_classe:str, nom_atributo:str, val_atributo:float) ->float:
    """
    Calcula o GI(Y|nom_atributo=val_atributo), ou seja,
    calcula o ganho de informação do atributo 'nom_atributo' quando ele assume o valor 'val_atributo'.
    O valor de Entropia(Y) já foi calculado e está armazenado em val_entropia_y.

    df_dados: Dataframe com os dados a serem analisados.
    val_entropia_y: Entropia(Y)
    nom_col_classe: nome da coluna que representa a classe
    nom_atributo: atributo a ser calculado o ganho de informação
    val_atributo: valor do atributo a ser considerado para este calculo. Uso do Union na dia do tipo: 
                  o valor pode ser qualquer tipo primitivo (boolean, int, float ou str)
    """
    df_dados_filtrado = df_dados[df_dados[nom_atributo]==val_atributo]

    val_ent_condicional = entropia(df_dados_filtrado, nom_col_classe)
    val_gi = val_entropia_y - val_ent_condicional

    return val_gi


def ganho_informacao(df_dados:pd.DataFrame, nom_col_classe:str, nom_atributo:str, bins=None) -> float:
    """
        Calcula GI(Y| nom_atributo), ou seja, o ganho de informação do atributo nom_atributo.

        df_dados: DataFrame com os dados a serem analisados.
        nom_col_classe: nome da coluna que representa a classe
        nom_atributo: atributo a ser calculado o ganho de informação
        val_atributo: valor do atributo a ser considerado para este calculo
    """
    ser_count_col = df_dados[nom_atributo].value_counts(bins=bins)

    val_entropia_y = entropia(df_dados, nom_col_classe)

    num_total = len(df_dados)
    val_info_gain = 0
    for val_atr, count_atr in ser_count_col.iteritems():
        val_prob = count_atr / num_total
        val_info_gain += (val_prob * ganho_informacao_condicional(df_dados, val_entropia_y, nom_col_classe, nom_atributo, val_atr))

    return val_info_gain
