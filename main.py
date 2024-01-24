import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Wczytaj dane z pliku CSV z użyciem średnika jako separatora
data_train = pd.read_csv('Cardiovascular_Disease_Dataset_mod.csv', sep=';')
data_test = pd.read_csv('Testowe.csv', sep=';')


# Usuń kolumny z samymi brakującymi danymi w danych treningowych
data_train = data_train.dropna(axis=1)

# Usuń kolumny z samymi brakującymi danymi w danych testowych
data_test = data_test.dropna(axis=1)

# Informacje o wartościach w poszczególnych kolumnach
value_info = {
    'gender': {'male': 1, 'female': 0},
    'chestpain': {0: 'typical angina', 1: 'atypical angina', 2: 'non-anginal pain', 3: 'asymptomatic'},
    'restingBP': {'min': 94, 'max': 200, 'unit': 'mm HG'},
    'serumcholestrol': {0: 'typical angina', 1: 'atypical angina', 2: 'non-anginal pain', 3: 'asymptomatic'},
    'fastingbloodsugar': {0: 'false', 1: 'true'},
    'restingrelectro': {0: 'normal', 1: 'abnormal'},
    'maxheartrate': {'min': 71, 'max': 202},
    'exerciseangia': {0: 'no', 1: 'yes'},
    'oldpeak': {'min': 0, 'max': 6.2},
    'noofmajorvessels': {0: 'value0', 1: 'value1', 2: 'value2', 3: 'value3'},
    'target': {0: 'Absence of Heart Disease', 1: 'Presence of Heart Disease'}
}

# Wydziel cechy (X) i etykiety (y) z danych treningowych
X_train = data_train.drop('target', axis=1)
y_train = data_train['target']

# Zastosuj SimpleImputer do uzupełnienia brakujących danych
imputer = SimpleImputer(strategy='mean')

X_train['oldpeak'] = X_train['oldpeak'].astype(str)
# Zamień przecinki na kropki w kolumnie "oldpeak"
X_train['oldpeak'] = X_train['oldpeak'].str.replace(',', '.')

# Skonwertuj kolumnę "oldpeak" na liczby zmiennoprzecinkowe
X_train['oldpeak'] = X_train['oldpeak'].astype(float)

# Zastosuj imputer
X_train_imputed = imputer.fit_transform(X_train)

# Zamień wartości w kolumnie "gender" na 1 dla mężczyzn i 0 dla kobiet
X_train['gender'] = X_train['gender'].map(value_info['gender'])

# Inżynieria cech - dodaj nowe cechy
X_train['age_and_maxheartrate'] = X_train['age'] * X_train['maxheartrate']

# Wydziel cechy (X) i etykiety (y) z danych testowych
X_test = data_test.drop('target', axis=1)
y_test = data_test['target']

# Zamień przecinki na kropki w kolumnie "oldpeak"
X_test['oldpeak'] = X_test['oldpeak'].astype(str)
X_test['oldpeak'] = X_test['oldpeak'].str.replace(',', '.')

# Skonwertuj kolumnę "oldpeak" na liczby zmiennoprzecinkowe
X_test['oldpeak'] = X_test['oldpeak'].astype(float)

# Zastosuj imputer
X_test_imputed = imputer.transform(X_test)

# Zamień wartości w kolumnie "gender" na 1 dla mężczyzn i 0 dla kobiet
X_test['gender'] = X_test['gender'].map(value_info['gender'])

# Inżynieria cech - dodaj nowe cechy
X_test['age_and_maxheartrate'] = X_test['age'] * X_test['maxheartrate']

# Przeskaluj cechy za pomocą StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Utwórz model XGBoost
model = XGBClassifier()

# Zdefiniuj przestrzeń hiperparametrów do przeszukania
param_dist = {
    'n_estimators': [50, 100, 150, 200, 250, 300],
    'max_depth': [3, 5, 7, 9, 11, 13],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 1, 5],
    'min_child_weight': [0.5, 1, 3, 5, 7, 9, 11]
}

# Utwórz obiekt RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=50, cv=5, scoring='accuracy',
                                   random_state=42)

# Przeprowadź przeszukiwanie random search na danych treningowych
random_search.fit(X_train_imputed, y_train)

# Wydrukuj najlepsze parametry i skuteczność
print("Najlepsze parametry:", random_search.best_params_)
print("Skuteczność na danych treningowych:", random_search.best_score_)

# Ocen skuteczność modelu na danych testowych
best_model = random_search.best_estimator_
# Drop NaN values from y_test and corresponding rows in X_test
nan_indices = y_test.index[y_test.isnull()]
y_test = y_test.drop(nan_indices)
X_test = X_test.drop(nan_indices)

# Ocen skuteczność modelu na danych testowych
y_pred = best_model.predict(X_test_imputed)
accuracy = accuracy_score(y_test, y_pred)
print(f'Skuteczność modelu na danych testowych: {accuracy * 100:.2f}%')
