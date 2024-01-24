import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Wczytaj dane z pliku CSV z użyciem średnika jako separatora
data_train = pd.read_csv('Cardiovascular_Disease_Dataset_mod.csv', sep=';')
data_test = pd.read_csv('Testowe.csv', sep=';')

# Wydziel cechy (X) i etykiety (y) z danych treningowych
X_train = data_train.drop('target', axis=1)
y_train = data_train['target']

# Usuń kolumny z samymi brakującymi danymi
X_train = X_train.dropna(axis=1, how='all')

# Zastosuj SimpleImputer do uzupełnienia brakujących danych
imputer = SimpleImputer(strategy='mean')

# Zamień przecinki na kropki w kolumnie "oldpeak"
X_train['oldpeak'] = X_train['oldpeak'].str.replace(',', '.')

# Skonwertuj kolumnę "oldpeak" na liczby zmiennoprzecinkowe
X_train['oldpeak'] = X_train['oldpeak'].astype(float)

# Zastosuj imputer
X_train_imputed = imputer.fit_transform(X_train)

# Wydziel cechy (X) i etykiety (y) z danych testowych
X_test = data_test.drop('target', axis=1)
y_test = data_test['target']

# Usuń kolumny z samymi brakującymi danymi
X_test = X_test.dropna(axis=1, how='all')

# Zamień przecinki na kropki w kolumnie "oldpeak"
X_test['oldpeak'] = X_test['oldpeak'].str.replace(',', '.')

# Skonwertuj kolumnę "oldpeak" na liczby zmiennoprzecinkowe
X_test['oldpeak'] = X_test['oldpeak'].astype(float)

# Zastosuj imputer
X_test_imputed = imputer.transform(X_test)

# Utwórz model RandomForest
model = RandomForestClassifier()

# Zdefiniuj przestrzeń hiperparametrów do przeszukania
param_dist = {
    'n_estimators': [100, 150, 200, 250],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Utwórz obiekt RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=50, cv=5, scoring='accuracy', random_state=42)

# Przeprowadź przeszukiwanie random search na danych treningowych
random_search.fit(X_train_imputed, y_train)

# Wydrukuj najlepsze parametry i skuteczność
print("Najlepsze parametry:", random_search.best_params_)
print("Skuteczność na danych treningowych:", random_search.best_score_)

# Ocen skuteczność modelu na danych testowych
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test_imputed)
accuracy = accuracy_score(y_test, y_pred)
print(f'Skuteczność modelu na danych testowych: {accuracy * 100:.2f}%')
