import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers, models

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------
DATASET = "mnist"      # "mnist" (digits) ya "emnist" (digits + letters)
EPOCHS = 15
BATCH_SIZE = 128

# EMNIST "balanced" ke 47 classes ke naam (tfds ke label order mein)
EMNIST_CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")


# ---------------------------------------------------------------------
# STEP 1: LOAD DATASET
# ---------------------------------------------------------------------
def load_data():
    if DATASET == "mnist":
        # Keras khud download kar leta hai
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        class_names = [str(i) for i in range(10)]
    elif DATASET == "emnist":
        import tensorflow_datasets as tfds
        ds_train, ds_test = tfds.load("emnist/balanced", split=["train", "test"],
                                      as_supervised=True, batch_size=-1)
        x_train, y_train = [a.numpy() for a in ds_train]
        x_test, y_test = [a.numpy() for a in ds_test]
        # EMNIST images transpose hoti hain, unhe seedha karna zaroori hai
        x_train = np.transpose(x_train[..., 0], (0, 2, 1))
        x_test = np.transpose(x_test[..., 0], (0, 2, 1))
        class_names = EMNIST_CLASSES
    else:
        raise ValueError("DATASET sirf 'mnist' ya 'emnist' ho sakta hai")
    return x_train, y_train, x_test, y_test, class_names


x_train, y_train, x_test, y_test, class_names = load_data()
n_classes = len(class_names)
print(f"Dataset: {DATASET} | Train: {x_train.shape} | Test: {x_test.shape} | Classes: {n_classes}")


# ---------------------------------------------------------------------
# STEP 2: PREPROCESSING
# ---------------------------------------------------------------------
# Pixel values 0-255 se 0-1 mein (neural network ke liye behtar)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# CNN ko (samples, height, width, channels) shape chahiye
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# Training data ka 10% validation ke liye alag
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.10, random_state=RANDOM_STATE, stratify=y_train)


# ---------------------------------------------------------------------
# STEP 3: BUILD CNN MODEL
# ---------------------------------------------------------------------
def build_model(n_classes):
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        # Data augmentation (sirf training ke waqt chalta hai)
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.08, 0.08),

        # Block 1: chhote patterns (lines, curves) seekhta hai
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2: bare shapes seekhta hai
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Classifier
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation="softmax"),  # har character ki probability
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


model = build_model(n_classes)
model.summary()


# ---------------------------------------------------------------------
# STEP 4: TRAIN
# ---------------------------------------------------------------------
cbs = [
    callbacks.EarlyStopping(monitor="val_accuracy", patience=4,
                            restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
]

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cbs,
    verbose=2,
)


# ---------------------------------------------------------------------
# STEP 5: EVALUATION
# ---------------------------------------------------------------------
y_prob = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

print("\n================ TEST RESULTS ================")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=class_names))


# ---------------------------------------------------------------------
# STEP 6: GRAPHS
# ---------------------------------------------------------------------
# 6a) Training curves
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history["accuracy"], label="Train")
ax[0].plot(history.history["val_accuracy"], label="Validation")
ax[0].set_title("Model Accuracy")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Accuracy")
ax[0].legend()
ax[1].plot(history.history["loss"], label="Train")
ax[1].plot(history.history["val_loss"], label="Validation")
ax[1].set_title("Model Loss")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")
ax[1].legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.close()

# 6b) Confusion matrix (numbers sirf tab jab classes kam hon)
cm = confusion_matrix(y_test, y_pred)
size = 8 if n_classes <= 15 else 16
plt.figure(figsize=(size, size * 0.85))
sns.heatmap(cm, annot=(n_classes <= 15), fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Handwritten Character Recognition")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()


def show_images(indices, title, filename):
    """Chuni hui test images ko unki prediction ke saath dikhata hai."""
    plt.figure(figsize=(12, 5))
    for k, idx in enumerate(indices[:15]):
        plt.subplot(3, 5, k + 1)
        plt.imshow(x_test[idx].squeeze(), cmap="gray")
        ok = y_pred[idx] == y_test[idx]
        plt.title(f"True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred[idx]]}",
                  color="green" if ok else "red", fontsize=9)
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


# 6c) Kuch random predictions
rng = np.random.default_rng(RANDOM_STATE)
show_images(rng.choice(len(x_test), 15, replace=False),
            "Sample Predictions (green = sahi, red = galat)", "sample_predictions.png")

# 6d) Galat predictions (model ne kahan ghalti ki)
wrong = np.where(y_pred != y_test)[0]
if len(wrong) > 0:
    show_images(wrong, f"Misclassified Examples (total galat: {len(wrong)})",
                "misclassified.png")

print("\nGraphs saved: training_curves.png, confusion_matrix.png, "
      "sample_predictions.png, misclassified.png")


# ---------------------------------------------------------------------
# STEP 7: SAVE MODEL + PREDICT ON YOUR OWN IMAGE
# ---------------------------------------------------------------------
model.save("handwritten_model.keras")
print("Model saved: handwritten_model.keras")


def predict_image(path):
    """
    Apni likhi hui image (png/jpg) ka character predict karta hai.
    Tip: safed kagaz par kaale marker se ek bara character likho aur photo lo.
    """
    img = Image.open(path).convert("L")        # grayscale
    # MNIST mein background kaala aur likhai safed hoti hai.
    # Agar tumhari image mein background roshan hai to usay ulta kar do.
    if np.array(img).mean() > 127:
        img = ImageOps.invert(img)
    img = img.resize((28, 28))
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)

    probs = model.predict(arr, verbose=0)[0]
    top = np.argsort(probs)[::-1][:3]
    print(f"\nImage: {os.path.basename(path)}")
    print(f"Predicted character: {class_names[top[0]]} ({probs[top[0]]:.1%})")
    print("Top 3:", ", ".join(f"{class_names[i]} ({probs[i]:.1%})" for i in top))


.