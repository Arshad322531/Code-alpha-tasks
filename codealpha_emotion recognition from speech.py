import glob
import os
import warnings

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------
DATA_DIR = "ravdess_data"        # RAVDESS dataset ka folder
FEATURE_FILE = "features.npz"    # extract kiye hue features yahan save honge
SAMPLE_RATE = 22050
OFFSET = 0.5                     # shuru ki khamoshi chhor do (sec)
DURATION = 2.5                   # itni der ki audio use karo (sec)
N_MFCC = 40                      # kitne MFCC coefficients
USE_EXTRA = True                 # True = MFCC + chroma + mel, False = sirf MFCC
EPOCHS = 10                      # demo ke liye epochs kam rakhe hain
BATCH_SIZE = 32

# RAVDESS emotion codes (filename ka teesra number)
EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}


# ---------------------------------------------------------------------
# STEP 1 & 2: FEATURE EXTRACTION
# ---------------------------------------------------------------------
def extract_features(path):
    """Ek audio file se feature vector banata hai."""
    # Agar hum fake data bana rahe hain to mock feature vector return karenge
    if path.startswith("mock_audio_"):
        rng = np.random.default_rng(int(path.split("_")[-1].split(".")[0]))
        # Features shape: MFCC (40) + Chroma (12) + Mel (128) agar USE_EXTRA True hai
        feat_dim = N_MFCC + (12 + 128 if USE_EXTRA else 0)
        return rng.normal(0, 1.0, feat_dim)

    audio, sr = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION, offset=OFFSET)
    if len(audio) == 0:
        audio, sr = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC).T, axis=0)
    features = [mfcc]

    if USE_EXTRA:
        # Chroma
        stft = np.abs(librosa.stft(audio))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        # Mel-spectrogram
        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr))
        mel = np.mean(mel.T, axis=0)
        features += [chroma, mel]

    return np.hstack(features)


def load_dataset():
    """Saari .wav files se features aur labels nikalta hai (fallback with dummy data)."""
    if os.path.exists(FEATURE_FILE):
        print(f"Cached features load ho rahe hain: {FEATURE_FILE}")
        data = np.load(FEATURE_FILE)
        return data["X"], data["y"]

    files = glob.glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True)
    if not files:
        print(f"'{DATA_DIR}' folder nahi mila. Pipeline testing ke liye automatic synthetic (dummy) dataset generate kiya ja raha hai.")
        # Create mock features directly
        rng = np.random.default_rng(RANDOM_STATE)
        n_samples = 400
        feat_dim = N_MFCC + (12 + 128 if USE_EXTRA else 0)
        X_mock = rng.normal(0, 1.0, (n_samples, feat_dim))
        
        # Emitting some patterns so learning occurs
        emotion_keys = list(EMOTIONS.values())
        y_mock_text = []
        for i in range(n_samples):
            emotion = emotion_keys[i % len(emotion_keys)]
            y_mock_text.append(emotion)
            # Shift mock features depending on emotion target
            X_mock[i] += emotion_keys.index(emotion) * 0.1
            
        return X_mock, np.array(y_mock_text)

    X, y = [], []
    print(f"{len(files)} audio files mili. Features extract ho rahe hain...")
    for i, f in enumerate(sorted(files), start=1):
        parts = os.path.basename(f).split("-")
        if len(parts) < 3 or parts[2] not in EMOTIONS:
            continue
        X.append(extract_features(f))
        y.append(EMOTIONS[parts[2]])
        if i % 200 == 0:
            print(f"  {i}/{len(files)} files done")

    X, y = np.array(X), np.array(y)
    np.savez(FEATURE_FILE, X=X, y=y)
    print("Features save ho gaye:", FEATURE_FILE)
    return X, y


X, y_text = load_dataset()
print("\nFeatures shape:", X.shape)
labels, counts = np.unique(y_text, return_counts=True)
print("Emotion counts:", dict(zip(labels, counts)))


# ---------------------------------------------------------------------
# STEP 3: ENCODE LABELS, SPLIT, SCALE
# ---------------------------------------------------------------------
encoder = LabelEncoder()
y = encoder.fit_transform(y_text)

# 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# CNN shape adaptation
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Handle class imbalance
weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight = dict(enumerate(weights))


# ---------------------------------------------------------------------
# STEP 4: BUILD & TRAIN 1D-CNN MODEL
# ---------------------------------------------------------------------
def build_model(input_len, n_classes):
    model = models.Sequential([
        layers.Input(shape=(input_len, 1)),

        layers.Conv1D(128, 5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),

        layers.Conv1D(256, 5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),

        layers.Conv1D(128, 5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


model = build_model(X_train.shape[1], len(encoder.classes_))
model.summary()

cbs = [
    callbacks.EarlyStopping(monitor="val_accuracy", patience=15,
                            restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, verbose=1),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=cbs,
    verbose=2,
)


# ---------------------------------------------------------------------
# STEP 5: EVALUATION
# ---------------------------------------------------------------------
y_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

print("\n================ TEST RESULTS ================")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=encoder.classes_))


# ---------------------------------------------------------------------
# STEP 6: GRAPHS
# ---------------------------------------------------------------------
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

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Speech Emotion Recognition")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

print("\nGraphs saved: training_curves.png, confusion_matrix.png")


# ---------------------------------------------------------------------
# STEP 7: SAVE MODEL + PREDICT ON NEW/DEMO FILE
# ---------------------------------------------------------------------
model.save("emotion_model.keras")
joblib.dump({"scaler": scaler, "encoder": encoder}, "emotion_preprocess.pkl")
print("Model saved: emotion_model.keras, emotion_preprocess.pkl")


def predict_emotion(audio_path):
    """Kisi bhi .wav file ka emotion predict karta hai."""
    feat = extract_features(audio_path).reshape(1, -1)
    feat = scaler.transform(feat)[..., np.newaxis]
    probs = model.predict(feat, verbose=0)[0]
    best = int(np.argmax(probs))
    print(f"\nFile: {os.path.basename(audio_path)}")
    print(f"Predicted emotion: {encoder.classes_[best].upper()} ({probs[best]:.1%})")
    for name, p in sorted(zip(encoder.classes_, probs), key=lambda t: -t[1]):
        print(f"   {name:10s} {p:.1%}")


# Example: try with the first real file or a mock file fallback
example = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True))
if example:
    predict_emotion(example[0])
else:
    print("\nTesting prediction using mock file sample:")
    predict_emotion("mock_audio_101.wav")