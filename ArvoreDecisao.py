import pandas as pd
dados = pd.read_csv("C:/Users/pedro/Downloads/creditcard.csv")
dados.head()

#pega os dados e conta, pega os dados e soma
num_trans = dados['Class'].count()
num_fraudes = dados['Class'].sum()

#numero de transacoes normais
trans_norm = num_trans - num_fraudes

#calculca porcentagem das transacoes normais ou fraudes
percents_fraudes = num_fraudes/num_trans
percents_norm = trans_norm/num_trans

print("Numero de transacoes: ", num_trans)
print("Numero de fraudes: ", num_fraudes,  "%.2f"%(percents_fraudes*100))
print("Transacoes normais: ", trans_norm, "%.2f"%(percents_norm*100))

from sklearn.model_selection import StratifiedShuffleSplit
def executar_validador(x,y):
  validador = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
  for treino_id, teste_id in validador.split(x,y):
    x_train, x_test = x[treino_id], x[teste_id]
    y_train, y_test = y[treino_id], y[teste_id]
  return x_train, x_test, y_train, y_test

#%%time
from sklearn import tree
def executar_classificador(classificador, x_train, x_test, y_train):
  arvore = classificador.fit(x_train, y_train)
  y_pred = arvore.predict(x_test)
  return y_pred

import matplotlib.pyplot as plt
def salvar_arvore(classificador, nome):
  plt.figure(figsize=(10,10))
  tree.plot_tree(classificador, filled=True, fontsize=14)
  plt.savefig(nome)
  plt.close()


#execucao do validador
x = dados.drop('Class', axis=1).values
y = dados['Class'].values
x_train, x_test, y_train, y_test = executar_validador(x,y)

#execucao do classificador DecisionTreeClassifier
classificador_arvore_decisao = tree.DecisionTreeClassifier()
y_pred_arvore_decisao = executar_classificador(classificador_arvore_decisao, x_train, x_test, y_train)

#criacao da figura da arvore de decisao
salvar_arvore(classificador_arvore_decisao, "Arvore_Decisao_1.png")
