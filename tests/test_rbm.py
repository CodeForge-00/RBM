"""
Pruebas de la RBM. Para una RBM diminuta se pueden enumerar todas las configuraciones, asi que la energia libre y
el gradiente de la log-verosimilitud se comparan con valores exactos.
Ejecutar desde la raiz del proyecto:  python -m unittest discover -s tests -t .
"""
import itertools
import unittest
import numpy as np

from RBM import RBM, sigmoid


def allStates(n):
    return np.array(list(itertools.product([0.0, 1.0], repeat=n)))


def energy(V, H, W, bv, bh):
    """E(v, h) para todos los pares (v, h) de las filas de V y H (misma longitud)."""
    return -V @ bv - H @ bh - np.einsum('ni,ij,nj->n', V, W, H)


def logPartition(W, bv, bh):
    V, H = allStates(len(bv)), allStates(len(bh))
    energies = np.array([[energy(v[None], h[None], W, bv, bh)[0] for h in H] for v in V])
    return np.log(np.sum(np.exp(-energies)))


def meanLogLikelihood(data, W, bv, bh):
    """Media de log p(v) sobre los datos, con p(v) = exp(-F(v)) / Z."""
    rbm = RBM(len(bv), len(bh))
    rbm.W, rbm.visible_bias, rbm.hidden_bias = W, bv, bh
    return np.mean(-rbm.freeEnergy(data)) - logPartition(W, bv, bh)


def exactGradients(data, W, bv, bh):
    """Gradiente exacto de la log-verosimilitud media: <.>_datos (con p(h|v)) menos <.>_modelo, por enumeracion."""
    rbm = RBM(len(bv), len(bh))
    rbm.W, rbm.visible_bias, rbm.hidden_bias = W, bv, bh
    h_data = rbm.hiddenProbabilities(data)
    V, H = allStates(len(bv)), allStates(len(bh))
    pairs_v = np.repeat(V, len(H), axis=0)
    pairs_h = np.tile(H, (len(V), 1))
    p = np.exp(-energy(pairs_v, pairs_h, W, bv, bh) - logPartition(W, bv, bh))
    dW = data.T @ h_data / len(data) - (pairs_v * p[:, None]).T @ pairs_h
    d_visible = data.mean(axis=0) - (pairs_v * p[:, None]).sum(axis=0)
    d_hidden = h_data.mean(axis=0) - (pairs_h * p[:, None]).sum(axis=0)
    return dW, d_visible, d_hidden


def tinyModel(seed=0, n_visible=3, n_hidden=2):
    rng = np.random.RandomState(seed)
    return rng.randn(n_visible, n_hidden), rng.randn(n_visible), rng.randn(n_hidden)


class ExactComputationsTest(unittest.TestCase):

    def test_free_energy_is_minus_log_sum_over_hidden_states(self):
        W, bv, bh = tinyModel()
        rbm = RBM(3, 2)
        rbm.W, rbm.visible_bias, rbm.hidden_bias = W, bv, bh
        V, H = allStates(3), allStates(2)
        expected = np.array([-np.log(sum(np.exp(-energy(v[None], h[None], W, bv, bh)[0]) for h in H)) for v in V])
        np.testing.assert_allclose(rbm.freeEnergy(V), expected, atol=1e-10)

    def test_visible_distribution_is_normalized(self):
        W, bv, bh = tinyModel(1)
        rbm = RBM(3, 2)
        rbm.W, rbm.visible_bias, rbm.hidden_bias = W, bv, bh
        probabilities = np.exp(-rbm.freeEnergy(allStates(3)) - logPartition(W, bv, bh))
        self.assertAlmostEqual(probabilities.sum(), 1.0, places=10)

    def test_exact_gradient_formula_matches_numerical_gradient(self):
        """La expresion <.>_datos - <.>_modelo es el gradiente de la log-verosimilitud."""
        W, bv, bh = tinyModel(2)
        data = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 1.0]])
        dW, d_visible, d_hidden = exactGradients(data, W, bv, bh)
        eps = 1e-6

        def numeric(param):
            grad = np.zeros_like(param)
            for idx in np.ndindex(*param.shape):
                original = param[idx]
                param[idx] = original + eps
                plus = meanLogLikelihood(data, W, bv, bh)
                param[idx] = original - eps
                minus = meanLogLikelihood(data, W, bv, bh)
                param[idx] = original
                grad[idx] = (plus - minus) / (2 * eps)
            return grad

        np.testing.assert_allclose(dW, numeric(W), atol=1e-6)
        np.testing.assert_allclose(d_visible, numeric(bv), atol=1e-6)
        np.testing.assert_allclose(d_hidden, numeric(bh), atol=1e-6)

    def test_cd_with_long_chain_approaches_exact_gradient(self):
        """Con muchos pasos de Gibbs y muestras, CD estima el gradiente exacto de la log-verosimilitud."""
        W, bv, bh = tinyModel(3)
        distinct = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1.0]])
        data = np.tile(distinct, (7000, 1))  # cada dato con el mismo peso
        rbm = RBM(3, 2, seed=0)
        rbm.W, rbm.visible_bias, rbm.hidden_bias = W.copy(), bv.copy(), bh.copy()
        cd = rbm.gradients(data, k=30, stochastic_negative=True)
        exact = exactGradients(distinct, W, bv, bh)
        for estimated, truth in zip(cd, exact):
            np.testing.assert_allclose(estimated, truth, atol=0.03)

    def test_cd1_gradient_points_in_a_similar_direction(self):
        """CD-1 es un gradiente aproximado: no es igual al exacto, pero apunta en una direccion parecida."""
        W, bv, bh = tinyModel(4)
        distinct = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1.0]])
        data = np.tile(distinct, (5000, 1))
        rbm = RBM(3, 2, seed=0)
        rbm.W, rbm.visible_bias, rbm.hidden_bias = W.copy(), bv.copy(), bh.copy()
        cd = np.concatenate([g.ravel() for g in rbm.gradients(data, k=1)])
        exact = np.concatenate([g.ravel() for g in exactGradients(distinct, W, bv, bh)])
        cosine = cd @ exact / (np.linalg.norm(cd) * np.linalg.norm(exact))
        self.assertGreater(cosine, 0.8)


class RBMBehaviourTest(unittest.TestCase):

    def test_encode_decode_shapes_and_ranges(self):
        rbm = RBM(10, 4, seed=0)
        X = np.random.RandomState(0).rand(7, 10)
        H = rbm.encode(X)
        self.assertEqual(H.shape, (7, 4))
        self.assertTrue(np.all((H >= 0) & (H <= 1)))
        self.assertEqual(rbm.decode(H).shape, (7, 10))
        self.assertEqual(rbm.reconstruct(X).shape, X.shape)
        self.assertTrue(set(np.unique(rbm.encode(X, sample=True))) <= {0.0, 1.0})
        np.testing.assert_allclose(rbm.getReconstructedOutput(X), rbm.reconstruct(X))

    def test_reconstruction_is_deterministic(self):
        rbm = RBM(10, 4, seed=0)
        X = np.random.RandomState(0).rand(5, 10)
        np.testing.assert_array_equal(rbm.reconstruct(X), rbm.reconstruct(X))

    def test_same_seed_gives_same_training(self):
        X = (np.random.RandomState(0).rand(60, 8) > 0.5).astype(float)
        a, b = RBM(8, 3, seed=5), RBM(8, 3, seed=5)
        a.train(X, epochs=5)
        b.train(X, epochs=5)
        np.testing.assert_array_equal(a.W, b.W)

    def test_visible_bias_initialization_matches_pixel_frequencies(self):
        X = np.array([[1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1.0]])
        rbm = RBM(3, 2)
        rbm.initVisibleBias(X)
        np.testing.assert_allclose(sigmoid(rbm.visible_bias), np.clip(X.mean(axis=0), 1e-3, 1 - 1e-3))

    def test_training_learns_two_prototypes(self):
        """Datos de dos patrones con ruido: la RBM reconstruye mucho mejor que predecir la imagen media."""
        rng = np.random.RandomState(0)
        prototypes = np.array([[1] * 8 + [0] * 8, [0] * 8 + [1] * 8], dtype=float)
        labels = rng.randint(0, 2, 400)
        X = prototypes[labels]
        X = np.abs(X - (rng.rand(*X.shape) < 0.05))  # 5 % de pixeles invertidos
        train, test = X[:300], X[300:]
        rbm = RBM(16, 4, seed=0)
        history = rbm.train(train, lr=0.1, epochs=60, batch_size=20, validation_data=test)
        baseline = np.mean((test - train.mean(axis=0)) ** 2)
        self.assertLess(history['val_reconstruction_error'][-1], 0.4 * baseline)
        self.assertLess(history['reconstruction_error'][-1], history['reconstruction_error'][0])
        self.assertLess(history['free_energy'][-1], history['free_energy'][0])

    def test_weight_decay_shrinks_weights(self):
        X = (np.random.RandomState(0).rand(100, 12) > 0.5).astype(float)
        plain, decayed = RBM(12, 5, seed=1), RBM(12, 5, seed=1)
        plain.train(X, epochs=30)
        decayed.train(X, epochs=30, weight_decay=0.5)
        self.assertLess(np.linalg.norm(decayed.W), np.linalg.norm(plain.W))

    def test_rejects_data_outside_unit_interval(self):
        with self.assertRaises(AssertionError):
            RBM(4, 2).train(np.array([[0.0, 1.0, 2.0, 0.5]]))

    def test_sigmoid_does_not_overflow(self):
        with np.errstate(all='raise'):
            np.testing.assert_allclose(sigmoid(np.array([-1e4, 0., 1e4])), [0, .5, 1], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
