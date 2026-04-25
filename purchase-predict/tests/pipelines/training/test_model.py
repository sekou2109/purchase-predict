import joblib
import pandas as pd
import numpy as np

def test_invariance_price():
    # ⚠️ Attention ici : vérifie si ton dossier s'appelle 06_model ou 06_models
    # Si Kedro te dit toujours "File Not Found", essaie 'data/06_model/model.pkl' (sans le 's')
    model = joblib.load('data/06_models/model.pkl')
    X_test = pd.read_csv('data/05_model_input/X_test.csv')

    # 2. Préparation des données perturbées (+1€ et -1€)
    X_test_price = X_test[X_test['price'] > 1].copy()
    
    X_test_price_plus = X_test_price.copy()
    X_test_price_plus['price'] = X_test_price_plus['price'] + 1
    
    X_test_price_minus = X_test_price.copy()
    X_test_price_minus['price'] = X_test_price_minus['price'] - 1

    # 3. Calcul des prédictions
    y_plus = model.predict_proba(X_test_price_plus)[:, 1]
    y_minus = model.predict_proba(X_test_price_minus)[:, 1]

    # 4. Calcul de la différence
    abs_delta = np.abs(y_minus - y_plus)

    # 5. L'Assertion (On vérifie que la variation moyenne ne dépasse pas 5%)
    assert abs_delta.mean() < 0.05, "Échec du test : Le modèle est trop sensible aux variations de prix !"