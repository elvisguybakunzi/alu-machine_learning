import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import GPyOpt
import matplotlib.pyplot as plt

# Define dataset (using MNIST as an example)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize data

def create_model(learning_rate, units, dropout, l2_reg, batch_size):
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(units, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(dropout),
        layers.Dense(10, activation='softmax')
    ])
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model, batch_size

def train_and_evaluate(params):
    learning_rate, units, dropout, l2_reg, batch_size = params[0]
    units = int(units)
    batch_size = int(batch_size)
    checkpoint_path = f"best_model_lr{learning_rate}_units{units}_drop{dropout}_l2{l2_reg}_bs{batch_size}.h5"
    
    model, batch_size = create_model(learning_rate, units, dropout, l2_reg, batch_size)
    
    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max')
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        epochs=20, batch_size=batch_size,
                        callbacks=[checkpoint, early_stopping], verbose=0)
    
    best_accuracy = max(history.history['val_accuracy'])
    return -best_accuracy  # Negative because we minimize in Bayesian Optimization

# Define search space
bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
    {'name': 'units', 'type': 'discrete', 'domain': (32, 64, 128, 256)},
    {'name': 'dropout', 'type': 'continuous', 'domain': (0.1, 0.5)},
    {'name': 'l2_reg', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 64, 128)}
]

# Run Bayesian Optimization
optimizer = GPyOpt.methods.BayesianOptimization(f=train_and_evaluate, domain=bounds)
optimizer.run_optimization(max_iter=30)

# Save results
best_params = optimizer.X[np.argmin(optimizer.Y)]
report = f"Best Hyperparameters:\nLearning Rate: {best_params[0]}\nUnits: {int(best_params[1])}\nDropout: {best_params[2]}\nL2 Regularization: {best_params[3]}\nBatch Size: {int(best_params[4])}\n"
with open('bayes_opt.txt', 'w') as f:
    f.write(report)

# Plot convergence
plt.plot(optimizer.Y)
plt.xlabel('Iteration')
plt.ylabel('Negative Accuracy')
plt.title('Bayesian Optimization Convergence')
plt.savefig('convergence_plot.png')
plt.show()