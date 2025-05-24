from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def avaliar_knn(X, y, target: str, n_splits=5, n_neighbors=5):
    """
    Executa K-Fold CV com KNN regressão para coluna `target`.
    Retorna média e desvio de MAE e RMSE.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    mae = -cross_val_score(knn, X, y[target], cv=kf,
                          scoring='neg_mean_absolute_error')
    rmse = ( -cross_val_score(knn, X, y[target], cv=kf,
                              scoring='neg_mean_squared_error') ) ** 0.5
    return mae.mean(), mae.std(), rmse.mean(), rmse.std()

def treinar_regressao_linear(X_train, y_train, X_test, y_test):
    """
    Treina Regressão Linear Múltipla para prever todas as colunas de y.
    Retorna métricas MAE e RMSE no conjunto de teste.
    """
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return mae, rmse
