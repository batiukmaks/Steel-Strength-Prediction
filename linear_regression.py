import csv
import numpy as np
from golden_section_search import GoldenSectionSearch
from random import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing

data_dir = "steel_strength.csv"


def read_data(data_dir):
    with open(data_dir, 'r') as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]
    # shuffle(data)
    return data


def get_data(data):
    feature_names = [key for key in data[0].keys() if key not in [
        'formula', 'yield strength', 'tensile strength', 'elongation']]
    features = [[float(row[key]) for key in feature_names] for row in data]
    yield_strength = [float(row['yield strength']) for row in data]
    return preprocessing.normalize(np.array(features)), preprocessing.normalize([np.array(yield_strength)])[0]


def Loss(X, y, W, b):
    return np.sum((y - (W @ X.T + b)) ** 2) / X.shape[0]


def dldw(X, y, W, b):
    return -2 * X.T @ (y - (W @ X.T + b)) / X.shape[0]


def dldb(X, y, W, b):
    return -2 * np.sum(y - (W @ X.T + b)) / X.shape[0]


def gradient_descent(X, y, W, b, max_iterations=10000):
    iteration = 0
    losses = []
    beta = get_beta(X, y, W, b)
    while np.abs(Loss(X, y, W, b) - Loss(X, y, W - beta * dldw(X, y, W, b), b - beta * dldb(X, y, W, b))) > 1e-9 and iteration < max_iterations:
        if (iteration := iteration + 1) % 100 == 0:
            print(f"{iteration} | Loss: {Loss(X, y, W, b)}")

        if iteration % 500 == 0:
            plot_scatter(X, y, W, b)
            
            losses.append(Loss(X, y, W, b))
        beta = get_beta(X, y, W, b)
        W = W - beta * dldw(X, y, W, b)
        b = b - beta * dldb(X, y, W, b)
    return W, b, losses


def get_beta(X, y, W, b):
    gss = GoldenSectionSearch()
    epsilon = 1e-5

    def next_X_approximation(beta):
        return Loss(X, y, W - beta * dldw(X, y, W, b), b - beta * dldb(X, y, W, b))

    beta, *other = gss.search(next_X_approximation, 0, 10, epsilon)
    return beta


def check(X, y):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)


def plot_losses(losses):
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()


def plot_scatter(X, y, W, b):
    plt.scatter(X[:, 0], y)
    plt.plot(X[:, 0], W[0] * X[:, 0] + b, color='red', label='Linear Regression')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def main():
    print("## Using gradient descent ##")
    X, y = get_data(read_data(data_dir))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    W = np.random.randn(X.shape[1])
    b = np.random.randn()

    W, b, losses = gradient_descent(X_train, y_train, W, b)
    print("Loss on test set:", Loss(X_test, y_test, W, b))

    print("\n\n## Using sklearn ##")
    check(X, y)

    # plot_losses(losses)
    plot_scatter(X, y, W, b)


if __name__ == "__main__":
    main()
