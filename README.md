# LLayerNeuralNetwork


A deep neural network implementation built **from scratch** using **NumPy** — no TensorFlow or PyTorch involved.

This is a personal learning project inspired by [Andrew Ng’s Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning). Every feature is implemented manually to better understand the core mechanics of neural networks.

---

## 🚀 Features

- Multi-layer forward and backward propagation  
- L2 regularization  
- Adam optimizer  
- Mini-batch gradient descent  
- Learning rate decay  
- Sigmoid and ReLU activation functions  
- Accuracy and F1-score evaluation metrics  
- Binary classification support  

---

## 📊 Current Results

On the current dataset (binary classification):

| Metric         | Value     |
|----------------|-----------|
| Train Accuracy | 99.98%    |
| Test Accuracy  | 85.86%    |
| Test F1 Score  | 88.80%    |

---

## 🔧 Planned Features (Next Versions)

- Dropout  
- Batch normalization  
- Softmax activation  
- Multi-class classification support  
- Early stopping (optional)

---

## 🧪 Sample Usage

```python
from LLayerNeuralNetwork import LLayerNeuralNetwork
import numpy as np

# Initialize
nn = LLayerNeuralNetwork()

# Fit the model
parameters, costs = nn.L_layer_model(
    X_train_flatten, Y_train,
    layers_dims=[X_train_flatten.shape[0], 10, 5, 1],
    learning_rate=0.0075,
    num_iterations=3000,
    print_cost=True
)

# Predict
pred_train = nn.predict(X_train_flatten, parameters)
pred_test = nn.predict(X_test_flatten, parameters)

# Evaluate
train_acc = nn.accuracy(pred_train, Y_train)
test_acc = nn.accuracy(pred_test, Y_test)
```
## 📁 Folder Structure
LLayerNeuralNetwork/
├── LLayerNeuralNetwork.py
└── README.md

## 📦 Installation
git clone https://github.com/Yunus-Emr/LLayerNeuralNetwork.git
cd LLayerNeuralNetwork
pip install -r requirements.txt

## 🙋‍♂️ Contributing

Feel free to open an issue or submit a pull request if you’d like to contribute or suggest improvements!

## 📜 License

This project is open-source and available under the MIT License.
