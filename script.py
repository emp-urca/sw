import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sfi import Data, Scalar

def calcular_educacao_exp():
    """Calcula a Categoria de Educação/Experiência para os dados atuais da memória"""
    # Carrega os dados da memória atual do Stata
    d = np.array(Data.get(var=["V4040", "VD3005", "lw"]), dtype=float)
    res = np.full(len(d), -1, dtype=int)
    
    # Valida linhas sem valores ausentes (NaN)
    m_v = ~np.isnan(d[:, :3]).any(axis=1)
    
    if np.any(m_v):
        xt, yt = d[m_v, :2], d[m_v, 2]
        
        # Ajusta a árvore de decisão
        tr = DecisionTreeRegressor(max_leaf_nodes=35, random_state=42).fit(xt, yt)
        nodes = tr.apply(np.nan_to_num(d[:, :2], nan=-1))
        
        # Cria o ranking baseado na média salarial de cada folha
        u_nodes = np.unique(nodes[m_v])
        m_n = {i: yt[nodes[m_v] == i].mean() for i in u_nodes}
        r_map = {oid: r + 1 for r, oid in enumerate(sorted(m_n, key=m_n.get))}
        
        # Mapeia as categorias e anula quem era NaN original
        c_a = np.array([r_map.get(x, -1) for x in nodes])
        c_a[~m_v] = -1
        res = c_a

    # Cria e armazena a variável no Stata
    try:
        Data.addVarInt('edu_exp_cat')
    except:
        pass
    Data.store('edu_exp_cat', None, res)
	
    # Define o escalar 'k' com o total real de categorias geradas neste ano
    k_valor = int(np.unique(res[res > 0]).size)
    Scalar.setValue('k', k_valor)
    print(f"[Python] edu_exp_cat criada. Categorias geradas (k): {k_valor}")

def calcular_experiencia_gamma():
    """Calcula a quebra de Experiência (gamma11) para os dados atuais da memória"""
    d = np.array(Data.get(var=["V2009", "lw"]), dtype=float)
    res = np.full(len(d), -1, dtype=int)
    
    m_v = ~np.isnan(d[:, :2]).any(axis=1)
    
    if np.any(m_v):
        xt, yt = d[m_v, 0].reshape(-1, 1), d[m_v, 1]
        
        tr = DecisionTreeRegressor(max_leaf_nodes=2, random_state=42).fit(xt, yt)
        thr = np.sort(tr.tree_.threshold[tr.tree_.threshold != -2])
        
        c_a = np.digitize(d[:, 0], thr)
        res = c_a

    # Alterado para 'gamma11' para evitar erros de caracteres especiais no Stata
    nome_var = "gamma11"
    try:
        Data.addVarInt(nome_var)
    except:
        pass
    Data.store(nome_var, None, res)
    print(f"[Python] {nome_var} criada.")

if __name__ == "__main__":
    calcular_educacao_exp()
    calcular_experiencia_gamma()
