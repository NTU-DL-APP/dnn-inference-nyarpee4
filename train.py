import tensorflow as tf
import numpy as np
import json
import os

# === モデルの保存先フォルダ ===
model_dir = './model'
os.makedirs(model_dir, exist_ok=True)

# === データの読み込み ===
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# Flatten: (28, 28) → (784)
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

# === モデルの定義 ===
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# === モデルの学習 ===
print("🚀 モデル学習中...")
model.fit(x_train, y_train, epochs=50, batch_size=64, validation_split=0.1)

# === テストデータで評価 ===
print("🧪 テストデータで評価中...")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"✅ テスト精度: {test_acc * 100:.2f}%")

# === 数件の予測出力 ===
predictions = model.predict(x_test[:5])
predicted_labels = np.argmax(predictions, axis=1)

print("\n🔍 サンプル予測結果:")
for i in range(5):
    print(f"画像{i+1}: 予測 = {predicted_labels[i]}, 実際 = {y_test[i]}")

# === モデルの保存 ===
model.save(os.path.join(model_dir, 'fashion_mnist.h5'))

# === アーキテクチャ（JSON）を保存 ===
model_json = model.to_json()
with open(os.path.join(model_dir, 'fashion_mnist.json'), 'w') as f:
    f.write(model_json)

# === 重み（NumPy形式）を保存 ===
weights_dict = {}
for layer in model.layers:
    weights = layer.get_weights()
    if weights:
        weights_dict[f"{layer.name}.weight"] = weights[0]  # W
        weights_dict[f"{layer.name}.bias"] = weights[1]    # b

np.savez(os.path.join(model_dir, 'fashion_mnist.npz'), **weights_dict)

print("\n✅ モデルのトレーニング・評価・保存が完了しました。")
