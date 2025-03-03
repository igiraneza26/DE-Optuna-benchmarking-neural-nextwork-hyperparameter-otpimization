import numpy as np
import time
import optuna
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

# Load CIFAR10 dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

X_train, y_train = X_train[:6000], y_train[:6000]  # Use only 10% of data
X_test, y_test = X_test[:1000], y_test[:1000]  # Smaller test set

y_train = y_train.flatten()  # Ensure labels are in 1D format
y_test = y_test.flatten()

# rand1binFlatten CIFAR-10 images from (32, 32, 3) to (3072,)
X_train = X_train.reshape(X_train.shape[0], -1)  
X_test = X_test.reshape(X_test.shape[0], -1)  

# rand1binNormalize pixel values to [0,1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# rand1binApply PCA to reduce features from 3072 to 100
pca = PCA(n_components=100)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Objective function (for DE & Optuna)
def objective(params):
    learning_rate, hidden_layer_size, dropout_rate = params
    hidden_layer_size = int(hidden_layer_size)

    model = MLPClassifier(hidden_layer_sizes=(hidden_layer_size,), learning_rate_init=learning_rate, 
                      max_iter=200, batch_size=32, random_state=42, activation='relu', solver='adam', early_stopping=True, n_iter_no_change=10)

    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
    return -np.mean(scores)  # DE minimizes this

# Standard DE
def adaptive_differential_evolution(bounds, generations=7, pop_size=7):
    result = differential_evolution(objective, bounds, strategy='rand1bin', maxiter=generations, 
                                    popsize=pop_size, mutation=0.8, recombination=0.7, 
                                    disp=True, workers=2)
    
    best_accuracy = -result.fun
    return result, [best_accuracy]

# Bayesian DE
def bayesian_adaptive_de(bounds, generations=7, pop_size=7):
    result = differential_evolution(objective, bounds, strategy='currenttobest1bin', maxiter=generations, 
                                    popsize=pop_size, mutation=0.8, recombination=0.7, 
                                    disp=True, workers=2)

    best_accuracy = -result.fun
    return result, [best_accuracy]

# Optuna Objective Function
def objective_optuna(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 0.0001, 0.1)
    hidden_layer_size = trial.suggest_int('hidden_layer_size', 10, 200)

    model = MLPClassifier(hidden_layer_sizes=(hidden_layer_size,), learning_rate_init=learning_rate, 
                      max_iter=200, batch_size=32, random_state=42, activation='relu', solver='adam', early_stopping=True, n_iter_no_change=10)

    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
    return np.mean(scores)

# Run Optuna multiple times
def run_multiple_optuna(n_trials=20, n_runs=3):
    accuracies = []
    times = []

    for i in range(n_runs):
        print(f"\nRunning Optuna - Run {i+1}/{n_runs}")

        start_time = time.time()
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_optuna, n_trials=n_trials)
        elapsed_time = time.time() - start_time

        best_accuracy = study.best_value
        accuracies.append(best_accuracy)
        times.append(elapsed_time)

    return np.mean(accuracies), np.std(accuracies), np.mean(times)

# Run DE for initial exploration
def run_de(bounds, generations=7, pop_size=7):
    start_time = time.time()
    result = differential_evolution(objective, bounds, strategy='rand2bin', maxiter=generations, 
                                    popsize=pop_size, mutation=0.8, recombination=0.7, 
                                    disp=True, workers=-1)
    de_time = time.time() - start_time
    return result.x, -result.fun, de_time  # Best params, accuracy, and runtime

# Run Optuna for fine-tuning (after DE)
def run_optuna(de_best_params, n_trials=10):
    def objective_optuna(trial):
        learning_rate = trial.suggest_loguniform('learning_rate', 0.0001, 0.1)
        hidden_layer_size = trial.suggest_int('hidden_layer_size', 10, 200)

        model = MLPClassifier(hidden_layer_sizes=(hidden_layer_size,), learning_rate_init=learning_rate, 
                      max_iter=200, batch_size=32, random_state=42, activation='relu', solver='adam', early_stopping=True, n_iter_no_change=5)

        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.enqueue_trial({'learning_rate': de_best_params[0], 'hidden_layer_size': int(de_best_params[1])})

    start_time = time.time()
    study.optimize(objective_optuna, n_trials=n_trials)
    optuna_time = time.time() - start_time

    return study.best_params, study.best_value, optuna_time  # Best hyperparameters, accuracy, and runtime

# Hybrid DE + Optuna
def hybrid_de_optuna(bounds, generations=7, pop_size=7, n_trials=15):
    print("\nRunning DE for global exploration...")
    de_best_params, de_best_acc, de_time = run_de(bounds, generations, pop_size)

    print("\nFine-tuning with Optuna...")
    optuna_best_params, optuna_best_acc, optuna_time = run_optuna(de_best_params, n_trials)

    total_time = de_time + optuna_time
    return de_best_params, de_best_acc, optuna_best_params, optuna_best_acc, total_time

# Run Hybrid DE + Optuna Experiment
def run_hybrid_experiment(bounds, n_runs=5):
    accuracies = []
    times = []

    for i in range(n_runs):
        print(f"\nRunning Hybrid DE + Optuna - Run {i+1}/{n_runs}")

        de_best_params, de_best_acc, optuna_best_params, optuna_best_acc, total_time = hybrid_de_optuna(bounds)
        best_accuracy = optuna_best_acc  

        accuracies.append(best_accuracy)
        times.append(total_time)  

    return np.mean(accuracies), np.std(accuracies), np.mean(times)

# Run Experiments & Compare Methods
def run_experiment(algorithm, bounds, n_runs=5):
    accuracies = []
    times = []

    for i in range(n_runs):
        print(f"\nRunning {algorithm.__name__} - Run {i+1}/{n_runs}")

        start_time = time.time()
        best_solution, accuracy_list = algorithm(bounds)
        elapsed_time = time.time() - start_time

        best_accuracy = -best_solution.fun
        accuracies.append(best_accuracy)
        times.append(elapsed_time)

    return np.mean(accuracies), np.std(accuracies), np.mean(times)

# Execute all experiments
if __name__ == "__main__":
    n_runs = 5
    bounds = [(0.0001, 0.1), (10, 200), (0.1, 0.5)]

    avg_acc_de, std_acc_de, avg_time_de = run_experiment(adaptive_differential_evolution, bounds, n_runs)
    avg_acc_bayesian_de, std_acc_bayesian_de, avg_time_bayesian_de = run_experiment(bayesian_adaptive_de, bounds, n_runs)
    avg_acc_optuna, std_acc_optuna, avg_time_optuna = run_multiple_optuna(n_trials=50, n_runs=n_runs)
    avg_acc_hybrid, std_acc_hybrid, avg_time_hybrid = run_hybrid_experiment(bounds, n_runs)

    # Print Full Comparison Table
    print("\nFinal Comparison of Optimization Methods")
    print("-------------------------------------------------------")
    print(f"{'Method':<20}{'Avg Accuracy':<15}{'Std Dev':<15}{'Avg Time (sec)':<15}")
    print("-------------------------------------------------------")
    print(f"{'Standard DE':<20}{avg_acc_de:.4f} ± {std_acc_de:.4f}  {avg_time_de:.2f} sec")
    print(f"{'Bayesian DE':<20}{avg_acc_bayesian_de:.4f} ± {std_acc_bayesian_de:.4f}  {avg_time_bayesian_de:.2f} sec")
    print(f"{'Optuna':<20}{avg_acc_optuna:.4f} ± {std_acc_optuna:.4f}  {avg_time_optuna:.2f} sec")
    print(f"{'Hybrid DE+Optuna':<20}{avg_acc_hybrid:.4f} ± {std_acc_hybrid:.4f}  {avg_time_hybrid:.2f} sec")

    # Update Plot with Hybrid DE + Optuna
    plt.plot(range(n_runs), [avg_acc_de]*n_runs, label="Standard DE")
    plt.plot(range(n_runs), [avg_acc_bayesian_de]*n_runs, label="Bayesian DE", linestyle="dashed")
    plt.plot(range(n_runs), [avg_acc_optuna]*n_runs, label="Optuna", linestyle="dotted")
    plt.plot(range(n_runs), [avg_acc_hybrid]*n_runs, label="Hybrid DE+Optuna", linestyle="dashdot")

    plt.xlabel("Runs")
    plt.ylabel("Accuracy")
    plt.title("Comparison of Optimization Methods")
    plt.legend()
    plt.show()