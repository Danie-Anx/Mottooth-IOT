# BEACONS-IOT: Projeto de Localização Indoor para Mottu

Este repositório contém um pipeline completo de Machine Learning para localização de motocicletas em pátios da empresa **Mottu**, utilizando dados de sinais BLE (beacons). O projeto inclui:

* **Pré-processamento** dos dados (combinação de CSVs, limpeza, normalização)
* **Modelagem de regressão** para prever coordenadas (x, y)
* **Modelagem de classificação** para prever o piso (*floor*) onde a moto se encontra
* Comparação de diferentes algoritmos e otimização de hiperparâmetros

---

## Estrutura de Pastas

```
BEACONS-IOT/
├── data/
│   └── raw/                # Dados brutos
│       ├── test.csv        # Dados de teste para submissão
│       └── train_all.csv   # CSV combinado de todos os treinos
├── notebooks/              # Notebooks para exploração e relatórios
│   └── BEACONS-MOTTU-ML.ipynb
├── src/                    # Código modular do pipeline
│   ├── preprocessing.py    # Carregamento e limpeza de dados
│   ├── models.py           # Funções de treino e avaliação de modelos
│   └── utils.py            # Funções utilitárias (resumo, métricas)
├── train.py                # Script de orquestração (load → preprocess → model)
├── requirements.txt        # Dependências do projeto
├── README.md               # Este arquivo
└── .gitignore              # Arquivos e pastas ignoradas pelo Git
```

---

## Instalação

1. **Clone** este repositório:

   ```bash
   git clone <URL_DO_REPO>
   cd BEACONS-IOT
   ```
2. **Crie** um ambiente virtual e instale dependências:

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux/MacOS
   .\.venv\Scripts\activate     # Windows
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Uso

### 1. Preparar os dados

* Coloque seus arquivos `*_train.csv` e `test.csv` dentro de `data/raw/`.
* O script `train.py` verifica se existe `train_all.csv`; caso não exista, ele combina automaticamente todos os `*_train.csv`.

### 2. Pipeline completo

Execute o pipeline principal para:

* Inspecionar dados (shape, dtypes, estatísticas)
* Pré-processar (limpeza, escala)
* Treinar e avaliar modelos de regressão (*x*, *y*)
* Treinar e avaliar modelos de classificação (*floor*)

```bash
python train.py
```

### 3. Notebook interativo

Abra o notebook para exploração detalhada e visualizações:

```bash
jupyter lab notebooks/BEACONS-MOTTU-ML.ipynb
```

---

## Modelos e Métricas

### Regressão (x, y)

* **KNN Regressor** via validação cruzada: MAE ≈ 59,4 (x) e 50,6 (y)
* **Regressão Linear**: MAE ≈ 51,6, RMSE ≈ 64,1

### Classificação (floor)

* **KNN otimizado** (n=5, weights='distance'): acurácia ≈ 0.67
* **HistGradientBoostingClassifier**: acurácia ≈ 0.76 (recomendado)

---

# Sobre o Projeto

## Link do Dataset usado: (Kaggle: https://www.kaggle.com/datasets/kokitanisaka/unified-ds-wifi-and-beacon)

Este projeto utiliza o dataset “Unified DS: WiFi and Beacon” (Kaggle: https://www.kaggle.com/datasets/kokitanisaka/unified-ds-wifi-and-beacon), que reúne medições de intensidade de sinal (RSSI) de pontos de Wi-Fi e beacons Bluetooth em diferentes posições (coordenadas x, y) e andares (floor) de um ambiente indoor. Cada linha do CSV corresponde a um instante de captura, com colunas para identificadores de cada beacon/Wi-Fi e seus valores de RSSI, além das colunas alvo que indicam a posição real (x, y) e o pavimento.

## O código está organizado em módulos Python em src/:

### preprocessing.py

combinar_treinos(): concatena vários arquivos _train.csv em train_all.csv.

limpar_e_escala(): carrega o DataFrame, filtra apenas colunas numéricas de RSSI, trata valores infinitos e nulos, executa clipping de outliers e aplica StandardScaler para normalizar.

### models.py

avaliar_knn(): recebe os dados de treino e um eixo alvo ('x' ou 'y'), faz validação cruzada com KNeighborsRegressor, retornando média e desvio de MAE e RMSE.

treinar_regressao_linear(): treina LinearRegression em (x, y) simultaneamente, faz previsão no conjunto de teste e calcula MAE e RMSE manualmente (usando mean_absolute_error e mean_squared_error + raiz quadrada).

### utils.py

resumo_dataframe(): imprime shape, tipos de dados, estatísticas descritivas e valores ausentes.

### train.py

Orquestra todo o pipeline: combina CSVs, inspeciona dados, pré-processa (chamando limpar_e_escala), divide em treino/teste, avalia KNN e regressão linear para posição e, em seguida, treina dois modelos de classificação de _floor_:

KNN Classifier otimizado (n_neighbors=5, weights='distance')

HistGradientBoostingClassifier (max_iter=100)

Ao final, imprime MAE/RMSE para x e y e acurácia + classification report para previsão de piso.

Com este pipeline você demonstra a viabilidade técnica do uso de sinais BLE/Wi-Fi para localização indoor, testando diversos algoritmos, fazendo engenharia de features e preparação de dados de forma modular e reutilizável.

---

## Próximos Passos

* Ajuste de hiperparâmetros com `RandomizedSearchCV` ou `GridSearchCV` (HGB, MLP)
* Testar modelos avançados (MLP, CNN, XGBoost)
* Feature engineering (agregados por beacon, PCA)
* Deploy e API para predição em tempo real
