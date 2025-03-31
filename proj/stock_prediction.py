## In[0]:
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils import data_string_to_float, status_calc

## In[1]:
# The percentage by which a stock has to beat the S&P500 to be considered a 'buy'
OUTPERFORMANCE = 10

## In[2]:
def build_data_set():
    """
    Reads the keystats.csv file and prepares it for scikit-learn
    :return: X_train and y_train numpy arrays
    """
    training_data = pd.read_csv("keystats1.csv", index_col="Date")
    training_data.dropna(axis=0, how="any", inplace=True)
    features = training_data.columns[6:]

    X_train = training_data[features].values
    # Generate the labels: '1' if a stock beats the S&P500 by more than 10%, else '0'.
    y_train = list(
        status_calc(
            training_data["stock_p_change"],
            training_data["SP500_p_change"],
            OUTPERFORMANCE,
        )
    )

    return X_train, y_train

## In[3]:
def predict_stocks():
    X_train, y_train = build_data_set()
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)

    data = pd.read_csv("forward_sample.csv", index_col="Date")
    data.dropna(axis=0, how="any", inplace=True)
    features = data.columns[6:]
    X_test = data[features].values
    tickers = data["Ticker"].values

    # Prédictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]  # proba d'être classé comme "1" (surperformance)

    # Construction du tableau de résultats
    results = pd.DataFrame({
        "Ticker": tickers,
        "Prédiction": y_pred,
        "Probabilité de surperformance (%)": (y_proba * 100).round(2)
    })

    results["Statut"] = results["Prédiction"].apply(lambda x: "✅ SURPERFORME" if x == 1 else "❌ SOUS-PERFORME")

    # Affichage trié par probabilité décroissante
    results = results.sort_values(by="Probabilité de surperformance (%)", ascending=False).reset_index(drop=True)

    print("\n📈 Résultats des prédictions :\n")
    print(results.to_string(index=False))

    # Optionnel : enregistrer les résultats dans un fichier Excel ou CSV
    results.to_excel("stock_predictions.xlsx", index=False)
    print("\n✅ Résultats enregistrés dans 'stock_predictions.xlsx'")

    return results


## In[4]:
if __name__ == "__main__":
    print("Building dataset and predicting stocks...")
    predict_stocks()
