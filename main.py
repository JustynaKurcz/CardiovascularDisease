import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Wczytaj dane z pliku CSV z użyciem średnika jako separatora
data = pd.read_csv('Cardiovascular_Disease_Dataset_mod.csv', sep=';')

# Wydziel cechy (X) i etykiety (y)
X = data.drop('target', axis=1)
y = data['target']

# Usuń kolumny z samymi brakującymi danymi
X = X.dropna(axis=1, how='all')

# Zastosuj SimpleImputer do uzupełnienia brakujących danych
imputer = SimpleImputer(strategy='mean')

# Zamień przecinki na kropki w kolumnie "oldpeak"
X['oldpeak'] = X['oldpeak'].str.replace(',', '.')

# Skonwertuj kolumnę "oldpeak" na liczby zmiennoprzecinkowe
X['oldpeak'] = X['oldpeak'].astype(float)

# Podziel dane na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Zastosuj imputer
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Utwórz model RandomForest
model = RandomForestClassifier()

# Zdefiniuj przestrzeń hiperparametrów do przeszukania
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [1, 15, 30],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3]
}

# Utwórz obiekt GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# Przeprowadź przeszukiwanie grid search na danych treningowych
grid_search.fit(X_train_imputed, y_train)

# Wydrukuj najlepsze parametry i skuteczność
print("Najlepsze parametry:", grid_search.best_params_)
print("Skuteczność na danych treningowych:", grid_search.best_score_)

# Ocen skuteczność modelu na danych testowych
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_imputed)
accuracy = accuracy_score(y_test, y_pred)
print(f'Skuteczność modelu na danych testowych: {accuracy * 100:.2f}%')
