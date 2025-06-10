import numpy as np
import json

# === Activation functions ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    # 数値の安定性のために最大値を引く（オーバーフロー対策）
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# === Forward propagation ===
def nn_forward_h5(model_arch, weights, data):
    x = data
    for layer in model_arch:
        lname = layer['name']
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer.get('weights', [])  # 一部層（Flattenなど）はweightsを持たない

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            activation = cfg.get("activation", None)
            if activation == "relu":
                x = relu(x)
            elif activation == "softmax":
                x = softmax(x)

    return x

# === Inference entry point ===
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)
