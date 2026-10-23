import numpy as np


def l2_cost(Yp, Yr):
    """Error cuadratico medio."""
    return np.mean((Yr - Yp) ** 2)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # clip evita overflow en exp


class RBM:
    """
    Maquina de Boltzmann restringida (Bernoulli-Bernoulli) entrenada con divergencia contrastiva (CD-k).

    Tiene una capa visible (los datos, con valores en [0, 1]) y una capa oculta de unidades binarias. Una matriz de
    pesos W, compartida por ambas capas, y un bias para cada una. La energia de una configuracion es

        E(v, h) = - v · b_v - h · b_h - v · W · h

    y p(v, h) es proporcional a exp(-E(v, h)). Entrenar consiste en subir la probabilidad de los datos.
    """

    def __init__(self, n_visible, n_hidden, seed=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.rng = np.random.RandomState(seed)
        # Pesos pequenos: con desviacion 1 las unidades se saturan desde el principio
        self.W = self.rng.normal(0.0, 0.01, (n_visible, n_hidden))
        self.visible_bias = np.zeros(n_visible)
        self.hidden_bias = np.zeros(n_hidden)

    # ------------------------------------------------------------------ distribuciones
    def hiddenProbabilities(self, V):
        """p(h = 1 | v) para cada unidad oculta."""
        return sigmoid(V @ self.W + self.hidden_bias)

    def visibleProbabilities(self, H):
        """p(v = 1 | h) para cada unidad visible."""
        return sigmoid(H @ self.W.T + self.visible_bias)

    def sample(self, probabilities):
        """Enciende (1) cada unidad con su probabilidad y apaga (0) el resto."""
        return (self.rng.uniform(size=probabilities.shape) < probabilities).astype(float)

    # ------------------------------------------------------------------ API publica
    def encode(self, X, sample=False):
        """
        Representacion de los datos en la capa oculta (reduccion de dimensionalidad): las probabilidades
        de activacion de las unidades ocultas, o una muestra binaria de ellas con `sample=True`.
        """
        probabilities = self.hiddenProbabilities(X)
        return self.sample(probabilities) if sample else probabilities

    def decode(self, H, sample=False):
        """Reconstruye la capa visible a partir de una representacion oculta."""
        probabilities = self.visibleProbabilities(H)
        return self.sample(probabilities) if sample else probabilities

    def reconstruct(self, X):
        """Reconstruccion determinista: codificar con probabilidades y decodificar."""
        return self.decode(self.encode(X))

    getReconstructedOutput = reconstruct  # nombre anterior

    def reconstructionError(self, X):
        """Error cuadratico medio entre los datos y su reconstruccion."""
        return l2_cost(self.reconstruct(X), X)

    def freeEnergy(self, V):
        """
        Energia libre F(v) = -log sum_h exp(-E(v, h)) = - v · b_v - sum_j log(1 + exp(v · W_j + b_hj)).
        Cumple p(v) = exp(-F(v)) / Z, asi que valores mas bajos son datos mas probables para el modelo.
        """
        return -V @ self.visible_bias - np.sum(np.logaddexp(0, V @ self.W + self.hidden_bias), axis=1)

    def initVisibleBias(self, X):
        """Pone el bias visible de modo que, sin pesos, la RBM reproduzca la frecuencia media de cada pixel."""
        p = np.clip(X.mean(axis=0), 1e-3, 1 - 1e-3)
        self.visible_bias = np.log(p / (1 - p))

    # ------------------------------------------------------------------ entrenamiento
    def gradients(self, V0, k=1, stochastic_negative=False):
        """
        Gradiente aproximado de la log-verosimilitud con divergencia contrastiva CD-k (promediado sobre las filas
        de V0). Es la diferencia entre las estadisticas de los datos (fase positiva) y las de una cadena de Gibbs
        de k pasos que parte de ellos (fase negativa):

            dW   = <v h>_datos - <v h>_cadena         db_v = <v>_datos - <v>_cadena         db_h = <h>_datos - <h>_cadena

        Para reducir el ruido (Hinton, "A practical guide to training RBMs"): las ocultas de la fase positiva y
        negativa usan probabilidades en las estadisticas, y la ultima reconstruccion visible tambien, salvo con
        `stochastic_negative=True`, que usa una muestra y da un estimador sin ese sesgo.
        """
        n = V0.shape[0]
        h0_prob = self.hiddenProbabilities(V0)
        h = self.sample(h0_prob)
        for step in range(k):
            v_prob = self.visibleProbabilities(h)
            last = step == k - 1
            v = v_prob if (last and not stochastic_negative) else self.sample(v_prob)
            hk_prob = self.hiddenProbabilities(v)
            if not last:
                h = self.sample(hk_prob)
        dW = (V0.T @ h0_prob - v.T @ hk_prob) / n
        d_visible = (V0 - v).mean(axis=0)
        d_hidden = (h0_prob - hk_prob).mean(axis=0)
        return dW, d_visible, d_hidden

    def train(self, X, lr=0.1, epochs=100, batch_size=32, k=1, weight_decay=0.0, validation_data=None,
              init_visible_bias=True, verbose=False):
        """
        Entrena con CD-k y mini-lotes.
        :param X: datos de forma (n_instancias, n_visible) con valores en [0, 1]
        :param lr: learning rate
        :param k: pasos de Gibbs de la divergencia contrastiva
        :param weight_decay: penalizacion L2 de los pesos
        :param validation_data: datos aparte para vigilar el sobreajuste
        :param init_visible_bias: inicializar el bias visible con las frecuencias medias de X
        :return: historial {'reconstruction_error': [...], 'free_energy': [...]} y, con validation_data,
                 'val_reconstruction_error' y 'val_free_energy'; un valor por epoca
        """
        X = np.asarray(X, dtype=float)
        assert X.shape[1] == self.n_visible and X.min() >= 0 and X.max() <= 1, \
            "Los datos deben tener forma (n, n_visible) y valores en [0, 1]"
        if init_visible_bias:
            self.initVisibleBias(X)
        history = {'reconstruction_error': [], 'free_energy': []}
        if validation_data is not None:
            history.update({'val_reconstruction_error': [], 'val_free_energy': []})

        for epoch in range(epochs):
            order = self.rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                dW, d_visible, d_hidden = self.gradients(X[order[start:start + batch_size]], k)
                self.W += lr * (dW - weight_decay * self.W)
                self.visible_bias += lr * d_visible
                self.hidden_bias += lr * d_hidden

            history['reconstruction_error'].append(self.reconstructionError(X))
            history['free_energy'].append(self.freeEnergy(X).mean())
            if validation_data is not None:
                history['val_reconstruction_error'].append(self.reconstructionError(validation_data))
                history['val_free_energy'].append(self.freeEnergy(validation_data).mean())
            if verbose:
                line = f"Epoch {epoch + 1} ==> reconstruction error: {history['reconstruction_error'][-1]:.5f}"
                if validation_data is not None:
                    line += f", validation: {history['val_reconstruction_error'][-1]:.5f}"
                print(line)
        return history
