import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sfi import Data
from sfi import Scalar

def calcular_educacao_exp():
    """Primeira parte: Categoria de Educação/Experiência (edu_exp_cat)"""
    d = np.array(Data.get(var=["V4040", "VD3005", "lw", "Ano"]), dtype=float)
    res = np.full(len(d), -1, dtype=int)
    anos = np.unique(d[~np.isnan(d[:, 3]), 3])

    for v in anos:
        idx = np.where(np.abs(d[:, 3] - v) < 0.5)
        df_a = d[idx]
        m_v = ~np.isnan(df_a[:, :3]).any(axis=1)
        
        if np.any(m_v):
            xt, yt = df_a[m_v, :2], df_a[m_v, 2]
            tr = DecisionTreeRegressor(max_leaf_nodes=30, random_state=42).fit(xt, yt)
            nodes = tr.apply(np.nan_to_num(df_a[:, :2], nan=-1))
            u_nodes = np.unique(nodes[m_v])
            m_n = {i: yt[nodes[m_v] == i].mean() for i in u_nodes}
            r_map = {oid: r + 1 for r, oid in enumerate(sorted(m_n, key=m_n.get))}
            c_a = np.array([r_map.get(x, -1) for x in nodes])
            c_a[~m_v] = -1
            res[idx] = c_a

    try:
        Data.addVarInt('edu_exp_cat')
    except:
        pass
    Data.store('edu_exp_cat', None, res)
	# Define o escalar 'k' no Stata com o número de categorias único
    # Usamos o total de categorias únicas encontradas no último loop ou no array geral
    k_valor = int(np.unique(res[res > 0]).size)
    Scalar.setValue('k', k_valor)
    print(f"Sucesso: edu_exp_cat e escalar k={k_valor} definidos.")
    print("Sucesso: edu_exp_cat calculada.")

def calcular_experiencia_gamma():
    """Segunda parte: Variável de Experiência (γ11)"""
    d = np.array(Data.get(var=["V2009", "lw", "Ano"]), dtype=float)
    res = np.full(len(d), -1, dtype=int)
    v_anos = np.round(np.nan_to_num(d[:, 2], nan=-1))
    anos_u = np.unique(v_anos[v_anos > 0])

    for v in anos_u:
        idx = np.where(v_anos == v)
        df_a = d[idx]
        m_v = ~np.isnan(df_a[:, :2]).any(axis=1)
        
        if np.any(m_v):
            xt, yt = df_a[m_v, 0].reshape(-1, 1), df_a[m_v, 1]
            tr = DecisionTreeRegressor(max_leaf_nodes=2, random_state=42).fit(xt, yt)
            thr = np.sort(tr.tree_.threshold[tr.tree_.threshold != -2])
            c_a = np.digitize(df_a[:, 0], thr)
            res[idx] = c_a

    try:
        Data.addVarInt("γ11")
    except:
        pass
    Data.store("γ11", None, res)
    print("Sucesso: γ11 calculada.")

if __name__ == "__main__":
    # Chama as duas funções em sequência
    calcular_educacao_exp()
    calcular_experiencia_gamma()
