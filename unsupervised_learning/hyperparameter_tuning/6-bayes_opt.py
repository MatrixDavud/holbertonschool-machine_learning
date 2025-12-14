import os
import json
import numpy as np
import matplotlib.pyplot as plt
import GPyOpt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import mnist

# ================================
# Load dataset (MNIST)
# ================================
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# Use a validation split
x_train, x_val = x_train[:-10000], x_train[-10000:]
y_train, y_val = y_train[:-10000], y_train[-10000:]

# ================================
# Directory for checkpoints
# ================================
os.makedirs("checkpoints", exist_ok=True)

# ================================
# Objective function for BO
# ================================
def objective_function(hyperparams):
    """
    GPyOpt minimizes the objective, so we return -val_accuracy
    """
    learning_rate = hyperparams[0][0]
    units = int(hyperparams[0][1])
    dropout = hyperparams[0][2]
    l2_weight = hyperparams[0][3]
    batch_size = int(hyperparams[0][4])

    tf.keras.backend.clear_session()

    # Build model
    model = models.Sequential([
        layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_weight),
            input_shape=(784,)
        ),
        layers.Dropout(dropout),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Checkpoint filename encodes hyperparameters
    ckpt_name = (
        f"lr={learning_rate:.5f}_"
        f"units={units}_"
        f"dropout={dropout:.2f}_"
        f"l2={l2_weight:.6f}_"
        f"bs={batch_size}.keras"
    )

    ckpt_path = os.path.join("checkpoints", ckpt_name)

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            ckpt_path,
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=20,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks
    )

    best_val_acc = max(history.history["val_accuracy"])

    print(f"Validation accuracy: {best_val_acc:.4f}")
    print(f"Saved checkpoint: {ckpt_name}")

    # GPyOpt minimizes → return negative accuracy
    return -best_val_acc


# ================================
# Define search space (5 parameters)
# ================================
bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-4, 1e-2)},
    {'name': 'units',         'type': 'discrete',   'domain': (64, 128, 256, 512)},
    {'name': 'dropout',       'type': 'continuous', 'domain': (0.1, 0.5)},
    {'name': 'l2_weight',     'type': 'continuous', 'domain': (1e-6, 1e-3)},
    {'name': 'batch_size',    'type': 'discrete',   'domain': (32, 64, 128)}
]

# ================================
# Bayesian Optimization
# ================================
optimizer = GPyOpt.methods.BayesianOptimization(
    f=objective_function,
    domain=bounds,
    acquisition_type='EI',
    maximize=False
)

optimizer.run_optimization(max_iter=30)

# ================================
# Plot convergence
# ================================
optimizer.plot_convergence()
plt.show()

# ================================
# Save optimization report
# ================================
best_params = optimizer.x_opt
best_score = -optimizer.fx_opt

report = {
    "best_validation_accuracy": float(best_score),
    "best_hyperparameters": {
        "learning_rate": float(best_params[0]),
        "units": int(best_params[1]),
        "dropout": float(best_params[2]),
        "l2_weight": float(best_params[3]),
        "batch_size": int(best_params[4])
    }
}

with open("bayes_opt.txt", "w") as f:
    f.write(json.dumps(report, indent=4))

print("\nOptimization complete!")
print("Best validation accuracy:", best_score)
print("Best hyperparameters:", report["best_hyperparameters"])
print("Report saved to bayes_opt.txt")

