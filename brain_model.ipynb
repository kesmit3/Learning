import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("GPUs:", tf.config.list_physical_devices('GPU'))

# 1) Data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (valid_images, valid_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']

# normalize to [0,1]
train_images = train_images.astype("float32")/255.0
valid_images = valid_images.astype("float32")/255.0

# 2) Model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 3) Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4) Train
history = model.fit(train_images, train_labels,
                    validation_data=(valid_images, valid_labels),
                    epochs=5, batch_size=128, verbose=1)

# 5) Evaluate
loss, acc = model.evaluate(valid_images, valid_labels, verbose=0)
print(f"Validation accuracy: {acc:.3f}")

# 6) Predict a sample
idx = 6174
pred = model.predict(valid_images[idx:idx+1], verbose=0)[0]
print("Predicted:", class_names[int(np.argmax(pred))],
      "| True:", class_names[int(valid_labels[idx])])

plt.imshow(valid_images[idx], cmap='gray'); plt.axis('off'); plt.show()

# 7) Save (for reuse/deploy/tests)
os.makedirs("artifacts", exist_ok=True)
model.save("artifacts/fashion_mnist_brain.keras")
