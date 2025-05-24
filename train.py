import os
from src.preprocessing import carregar_dados, combinar_treinos, limpar_e_escala
from src.models import avaliar_knn, treinar_regressao_linear
from src.utils import resumo_dataframe

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# ------------ PATHS ------------
RAW_DIR     = os.path.join('data', 'raw')
TRAIN_ALL   = os.path.join(RAW_DIR, 'train_all.csv')
TEST_CSV    = os.path.join(RAW_DIR, 'test.csv')
TRAIN_FILES = [os.path.join(RAW_DIR, f)
               for f in os.listdir(RAW_DIR)
               if f.endswith('_train.csv')]

# 1) Combinar treinos, se necessário
if not os.path.exists(TRAIN_ALL):
    combinar_treinos(TRAIN_FILES, TRAIN_ALL)

# 2) Inspeção inicial
df_all = resumo_dataframe(TRAIN_ALL)

# 3) Pré-processamento
DROP_COLS   = ['site','path','timestamp']
TARGET_COLS = ['x','y','floor']
X, y = limpar_e_escala(df_all, drop_cols=DROP_COLS, target_cols=TARGET_COLS)

# 4) Divisão Treino/Teste para regressão
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nDivisão (regressão): {X_train.shape[0]} treino, {X_test.shape[0]} teste")

# 5) Avaliar KNN para regressão em x e y
for axis in ['x','y']:
    mae_m, mae_s, rmse_m, rmse_s = avaliar_knn(X_train, y_train, target=axis)
    print(f"KNN Reg. eixo {axis}: MAE {mae_m:.3f}±{mae_s:.3f}, RMSE {rmse_m:.3f}±{rmse_s:.3f}")

# 6) Regressão Linear Múltipla
mae_lr, rmse_lr = treinar_regressao_linear(
    X_train, y_train[['x','y']],
    X_test,  y_test[['x','y']]
)
print(f"\nRegressão Linear: MAE {mae_lr:.3f}, RMSE {rmse_lr:.3f}")

# -------------------------------------------------------------------
# 7) Classificação de piso (floor)
#    - KNN otimizado
#    - HistGradientBoostingClassifier
# -------------------------------------------------------------------

# 7.1) Preparar SMOTE e amostra (já tratado em notebook; supondo X_scaled e y_floor)
# Para simplificar aqui, usamos o mesmo split X/y, mas em produção recomendo SMOTE antes
y_floor = y['floor']

# 7.2) Dividir para classificação
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X, y_floor, test_size=0.2, stratify=y_floor, random_state=42
)
print(f"\nDivisão (classificação): {Xc_train.shape[0]} treino, {Xc_test.shape[0]} teste")

# 7.3) KNN Classifier otimizado
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(Xc_train, yc_train)
acc_knn = accuracy_score(yc_test, knn.predict(Xc_test))
print(f"KNN Classifier accuracy: {acc_knn:.3f}")
print(classification_report(yc_test, knn.predict(Xc_test)))

# 7.4) HistGradientBoostingClassifier
hgb = HistGradientBoostingClassifier(max_iter=100, random_state=42)
hgb.fit(Xc_train, yc_train)
acc_hgb = accuracy_score(yc_test, hgb.predict(Xc_test))
print(f"\nHistGradientBoosting accuracy: {acc_hgb:.3f}")
print(classification_report(yc_test, hgb.predict(Xc_test)))

print("\n✔️ Pipeline completo executado com sucesso!")
