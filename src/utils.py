import pandas as pd

def resumo_dataframe(path: str, n_linhas:int=5):
    """
    Carrega um CSV e exibe shape, tipos, descritivas e primeiras linhas.
    """
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print("\nTipos de dados:")
    print(df.dtypes)
    print("\nEstatísticas descritivas:")
    print(df.describe().T)
    print(f"\nPrimeiras {n_linhas} linhas:")
    print(df.head(n_linhas))
    return df
