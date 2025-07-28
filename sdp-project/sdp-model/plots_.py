import matplotlib.pyplot as plt

def plot_prediction_error(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Valor Real")
    plt.ylabel("Valor Predito")
    plt.title("Prediction Error Plot")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('benchmark/metrics/prediction_error.png')

def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
    plt.axhline(0, color='r', linestyle='--', lw=2)
    plt.xlabel("Valor Predito")
    plt.ylabel("Resíduo (Erro)")
    plt.title("Gráfico de Resíduos")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('benchmark/metrics/residuals.png')


def plot_estimator_metrics(dicionario):
    fig = plt.figure(figsize=(8, 6))

    for metric, values in dicionario.items():
        plt.plot(values, label=metric)

    plt.xlabel("Fold")
    plt.ylabel("Metrica")
    plt.title("Gráfico de Metricas")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('benchmark/metrics/estimator_metrics.png')
