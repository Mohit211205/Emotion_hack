import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense,
                                      Dropout, Flatten, BatchNormalization)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ─── Config ───────────────────────────────────────
IMG_SIZE = 48
BATCH_SIZE = 128  # increased for faster training
EPOCHS = 20       # reduced from 50
DATASET_PATH = "dataset"

# ─── Data ─────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    horizontal_flip=True,
    zoom_range=0.1
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    f"{DATASET_PATH}/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

test_gen = test_datagen.flow_from_directory(
    f"{DATASET_PATH}/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ─── Simpler/Faster Model ─────────────────────────
model = Sequential([
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(64, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(128, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─── Callbacks ────────────────────────────────────
callbacks = [
    ModelCheckpoint(
        "emotion_model.h5",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )
]

# ─── Train ────────────────────────────────────────
print("\n🚀 Training started...")
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=test_gen,
    callbacks=callbacks,
    verbose=1
)

# ─── Plot ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['accuracy'], color='cyan', label='Train')
ax1.plot(history.history['val_accuracy'], color='orange', label='Val')
ax1.set_title('Accuracy'); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(history.history['loss'], color='cyan', label='Train')
ax2.plot(history.history['val_loss'], color='orange', label='Val')
ax2.set_title('Loss'); ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_results.png")
plt.show()

# ─── Evaluate ─────────────────────────────────────
loss, acc = model.evaluate(test_gen, verbose=0)
print(f"\n✅ Training Complete!")
print(f"📊 Test Accuracy: {acc*100:.2f}%")
print(f"💾 Model saved: emotion_model.h5")