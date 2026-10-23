import argparse
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from RBM import RBM


def loadData(seed):
    """Digitos manuscritos de sklearn (imagenes 8x8, pixeles de 0 a 16) escalados a [0, 1] y separados en entrenamiento y prueba."""
    digits = load_digits()
    X = digits.data / 16.0
    return train_test_split(X, digits.target, test_size=0.25, random_state=seed, stratify=digits.target)


def plotResults(rbm, X_test, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = 8
    reconstructed = rbm.reconstruct(X_test[:n])
    fig, axes = plt.subplots(3, n, figsize=(n * 1.2, 4))
    for i in range(n):
        axes[0, i].imshow(X_test[i].reshape(8, 8), cmap='gray_r')
        axes[1, i].imshow(reconstructed[i].reshape(8, 8), cmap='gray_r')
        axes[2, i].imshow(rbm.W[:, i].reshape(8, 8), cmap='gray')
    for ax in axes.ravel():
        ax.axis('off')
    axes[0, 0].set_title('original', loc='left', fontsize=8)
    axes[1, 0].set_title('reconstruccion', loc='left', fontsize=8)
    axes[2, 0].set_title('pesos de 8 unidades ocultas', loc='left', fontsize=8)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    print(f"Figura guardada en {path}")


def main():
    parser = argparse.ArgumentParser(description='RBM hecha desde cero sobre los digitos de sklearn')
    parser.add_argument('--task', choices=['reconstruct', 'classify'], default='reconstruct',
                        help='reconstruct: reconstruir imagenes; classify: usar las caracteristicas ocultas para clasificar')
    parser.add_argument('--hidden', type=int, default=50, help='unidades ocultas')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--k', type=int, default=1, help='pasos de Gibbs de la divergencia contrastiva')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--plot', metavar='ARCHIVO', help='guardar una figura (requiere matplotlib), solo con --task reconstruct')
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = loadData(args.seed)
    rbm = RBM(X_train.shape[1], args.hidden, seed=args.seed)
    history = rbm.train(X_train, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, k=args.k,
                        validation_data=X_test)

    if args.task == 'reconstruct':
        baseline = np.mean((X_test - X_train.mean(axis=0)) ** 2)
        print(f"Error de reconstruccion (entrenamiento): {history['reconstruction_error'][-1]:.5f}")
        print(f"Error de reconstruccion (prueba):        {history['val_reconstruction_error'][-1]:.5f}")
        print(f"Referencia (predecir la imagen media):    {baseline:.5f}")
        print(f"Energia libre media (entrenamiento / prueba): "
              f"{history['free_energy'][-1]:.2f} / {history['val_free_energy'][-1]:.2f}")
        if args.plot:
            plotResults(rbm, X_test, args.plot)
    else:
        raw = LogisticRegression(max_iter=2000).fit(X_train, y_train).score(X_test, y_test)
        features = LogisticRegression(max_iter=2000).fit(rbm.encode(X_train), y_train).score(rbm.encode(X_test), y_test)
        print(f"Regresion logistica sobre los {X_train.shape[1]} pixeles:            {raw:.4f}")
        print(f"Regresion logistica sobre las {args.hidden} caracteristicas de la RBM: {features:.4f}")


if __name__ == '__main__':
    main()
