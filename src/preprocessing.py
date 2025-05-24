import pandas as pd
from sklearn.preprocessing import StandardScaler

def carregar_dados(path_train: str, path_test: str):
    """
    Carrega os dados de treino e teste a partir dos CSVs.
    Retorna df_train, df_test.
    """
    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)
    return df_train, df_test

def combinar_treinos(train_paths: list[str], output_path: str):
    """
    Concatena múltiplos CSVs de treino em um único CSV.
    """
    with open(output_path, 'w', encoding='utf-8') as out:
        # header do primeiro
        with open(train_paths[0], 'r', encoding='utf-8') as f0:
            out.write(f0.readline())
        # dados de todos
        for p in train_paths:
            with open(p, 'r', encoding='utf-8') as f:
                next(f)
                out.writelines(f)
    print(f"[OK] Arquivo combinado salvo em {output_path}")

def limpar_e_escala(df, drop_cols: list[str], target_cols: list[str]):
    """
    Remove colunas irrelevantes, preenche nulos e normaliza features numéricas.
    Retorna X (DataFrame) e y (DataFrame).
    """
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    # separar X e y
    X = df.drop(columns=target_cols)
    y = df[target_cols].copy()
    # preencher nulos
    X = X.fillna(0)
    # normalizar numéricas
    num_cols = X.select_dtypes(include=['int64','float64']).columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    return X, y
