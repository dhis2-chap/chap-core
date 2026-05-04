import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit


class StackedEnsemble:
    """
    Generisk stacked ensemble med out-of-fold stacking.

    - base_models: liste av modeller med .fit(X, y) og .predict(X)
    - meta_model: modell som tar inn base-prediksjoner som features
    - n_folds: antall folds for stacking (KFold eller TimeSeriesSplit)

    Merk:
    - For tidsserier: sett use_time_series_split=True for å unngå lekkasje.
    """

    def __init__(self, base_models, meta_model, n_folds=5, use_time_series_split=False, random_state=42):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.use_time_series_split = use_time_series_split
        self.random_state = random_state

        self.fitted_base_models = None
        self.weights = None

    def train(self, X_train, y_train):
        """
        Tren stacked-ensemblet med out-of-fold stacking.

        1) Generer out-of-fold-prediksjoner for hver basemodell.
        2) Tren meta-modellen på disse OOF-prediksjonene.
        3) Tren hver basemodell på hele treningssettet for bruk ved prediksjon.
        """
        # Behold original X_train / y_train (kan være pandas)
        X_train_orig = X_train
        y_train_orig = y_train

        # Nødvendig for å få fold-indekser
        X_arr = np.asarray(X_train_orig)
        y_arr = np.asarray(y_train_orig)

        n_samples = X_arr.shape[0]
        n_models = len(self.base_models)

        # OOF-prediksjoner: (n_samples, n_models)
        oof_preds = np.zeros((n_samples, n_models))

        # Velg CV-strategi
        if self.use_time_series_split:
            splitter = TimeSeriesSplit(n_splits=self.n_folds)
        else:
            splitter = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        # For hver basemodell, lag out-of-fold-prediksjoner
        for m_idx, base_model in enumerate(self.base_models):
            for train_idx, valid_idx in splitter.split(X_arr):
                # Viktig: bruk samme type som originalen (pandas eller np)
                if hasattr(X_train_orig, "iloc"):
                    X_tr = X_train_orig.iloc[train_idx]
                    X_val = X_train_orig.iloc[valid_idx]
                else:
                    X_tr = X_arr[train_idx]
                    X_val = X_arr[valid_idx]

                y_tr = y_arr[train_idx]

                base_model.fit(X_tr, y_tr)
                oof_preds[valid_idx, m_idx] = base_model.predict(X_val)

        # Tren meta-modellen på OOF-prediksjonene
        self.meta_model.fit(oof_preds, y_arr)

        # Tren endelige basemodeller på hele treningssettet
        self.fitted_base_models = []
        for base_model in self.base_models:
            base_model.fit(X_train_orig, y_train_orig)
            self.fitted_base_models.append(base_model)

        # Hent vekter
        self._compute_weights(oof_preds, y_arr)

    def _compute_weights(self, base_predictions, y):
        """
        Beregn vekter fra meta-modellen.

        - Hvis meta-modell har .coef_ brukes disse.
        - Ellers: lik vekt til alle basemodeller.
        """
        n_models = base_predictions.shape[1]

        if hasattr(self.meta_model, "coef_"):
            coef = self.meta_model.coef_

            # Håndter både (n_models,) og (1, n_models)
            coef = np.asarray(coef)
            if coef.ndim == 2:
                coef = coef[0]

            weights = np.abs(coef)

            # Sett veldig små vekter til 0 (numerisk støy)
            tol = 1e-6
            weights[weights < tol] = 0.0

            # Hvis alle ble 0 (teoretisk), sett lik vekt
            if np.all(weights == 0):
                weights = np.ones(n_models)
        else:
            # Hvis vi ikke har koeffisienter, gi lik vekt
            weights = np.ones(n_models)

        # Normaliser til prosent
        weight_sum = np.sum(weights)
        self.weights = (weights / weight_sum) * 100.0

        print(f"Vekter lært fra metamodell (i prosent): {self.weights}")

    def predict(self, X_test):
        """
        Predikér ved å:
        1) La alle basemodeller predikere på X_test.
        2) Bruke meta-modellen til å kombinere base-prediksjonene.
        """
        if self.fitted_base_models is None:
            raise RuntimeError("Modellen er ikke trent. Kall .train() først.")

        base_predictions = []
        for model in self.fitted_base_models:
            base_predictions.append(model.predict(X_test))

        base_predictions = np.array(base_predictions).T  # (n_samples, n_models)
        return self.meta_model.predict(base_predictions)

    def get_weights(self):
        """
        Returner absolutte vekter i prosent (en vekt per basemodell).
        """
        return np.abs(self.weights) if self.weights is not None else None
