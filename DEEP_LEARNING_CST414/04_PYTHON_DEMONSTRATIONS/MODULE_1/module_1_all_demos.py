"""
KTU CST414 Deep Learning: Module 1 Comprehensive Demonstrations
Faculty: Jane Rose | Dept of CSE
"""

import numpy as np
import matplotlib.pyplot as plt

# ======================================================================
# Day 01 (Topic 1.1): Perceptron from Scratch for Linearly Separable Gates
# ======================================================================

def demo_day_01():
    print("\n--- DEMO DAY 01: Perceptron from Scratch for Linearly Separable Gates ---")
    import numpy as np
    import matplotlib.pyplot as plt
    
    class Perceptron:
        def __init__(self, input_dim, lr=0.1, epochs=20):
            self.w = np.zeros(input_dim)
            self.b = 0.0
            self.lr = lr
            self.epochs = epochs
            self.history = []
    
        def predict(self, x):
            z = np.dot(x, self.w) + self.b
            return 1 if z >= 0 else 0
    
        def fit(self, X, y):
            for epoch in range(self.epochs):
                errors = 0
                for xi, target in zip(X, y):
                    pred = self.predict(xi)
                    err = target - pred
                    if err != 0:
                        self.w += self.lr * err * xi
                        self.b += self.lr * err
                        errors += 1
                self.history.append((self.w.copy(), self.b, errors))
                if errors == 0:
                    print(f"Converged at epoch {epoch + 1}")
                    break
    
    # AND Gate Dataset
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0, 0, 0, 1])
    
    p = Perceptron(input_dim=2, lr=0.2)
    p.fit(X, y)
    print("Trained Weights:", p.w, "Bias:", p.b)
    for xi, yi in zip(X, y):
        print(f"Input: {xi} -> Target: {yi}, Prediction: {p.predict(xi)}")
    

# ======================================================================
# Day 02 (Topic 1.2): Solving XOR with Multi-Layer Perceptron using NumPy
# ======================================================================

def demo_day_02():
    print("\n--- DEMO DAY 02: Solving XOR with Multi-Layer Perceptron using NumPy ---")
    import numpy as np
    
    def relu(z):
        return np.maximum(0, z)
    
    # Hard-coded weights for 2-2-1 MLP solving XOR
    # Hidden layer weights: W1 (2x2), b1 (2,)
    W1 = np.array([[1.0, 1.0],
                   [1.0, 1.0]])
    b1 = np.array([-0.5, -1.5])
    
    # Output layer weights: W2 (1x2), b2 (1,)
    W2 = np.array([[1.0, -2.0]])
    b2 = np.array([0.0])
    
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    
    print("=== MLP FORWARD PASS ON XOR ===")
    for x in X:
        z1 = np.dot(x, W1.T) + b1
        h = relu(z1)
        z2 = np.dot(h, W2.T) + b2
        y_pred = z2[0]
        print(f"Input: {x} -> Hidden z1: {z1} -> Hidden a1: {h} -> Output: {y_pred:.1f}")
    

# ======================================================================
# Day 03 (Topic 1.3): Visualization of Activation Functions and Their Derivatives
# ======================================================================

def demo_day_03():
    print("\n--- DEMO DAY 03: Visualization of Activation Functions and Their Derivatives ---")
    import numpy as np
    
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def sigmoid_grad(z):
        s = sigmoid(z)
        return s * (1.0 - s)
    
    def relu(z):
        return np.maximum(0, z)
    
    def relu_grad(z):
        return np.where(z > 0, 1.0, 0.0)
    
    def softmax(z):
        exp_z = np.exp(z - np.max(z)) # numerically stable
        return exp_z / np.sum(exp_z)
    
    # Test Softmax
    logits = np.array([2.0, 1.0, 0.1])
    probs = softmax(logits)
    print("Logits:", logits)
    print("Softmax Probabilities:", probs)
    print("Sum of Probabilities:", np.sum(probs))
    
    # Test Sigmoid Gradient saturation
    z_vals = np.array([-10.0, -2.0, 0.0, 2.0, 10.0])
    print("\nSigmoid Gradient at z =", z_vals)
    print("Gradients:", sigmoid_grad(z_vals))
    

# ======================================================================
# Day 04 (Topic 1.4): Minimal NumPy Implementation of 2-Layer Backpropagation
# ======================================================================

def demo_day_04():
    print("\n--- DEMO DAY 04: Minimal NumPy Implementation of 2-Layer Backpropagation ---")
    import numpy as np
    
    # Synthetic single sample
    x = np.array([[0.5, -0.2]]) # 1x2 input
    y_true = np.array([[1.0]])   # 1x1 target
    
    # Weights and biases
    W1 = np.array([[0.1, 0.4], [-0.3, 0.2]]) # 2x2
    b1 = np.array([[0.1, -0.1]])             # 1x2
    W2 = np.array([[0.5], [-0.2]])            # 2x1
    b2 = np.array([[0.2]])                   # 1x1
    
    # Forward Pass
    z1 = np.dot(x, W1) + b1
    a1 = 1.0 / (1.0 + np.exp(-z1)) # sigmoid
    z2 = np.dot(a1, W2) + b2
    a2 = 1.0 / (1.0 + np.exp(-z2)) # sigmoid output
    loss = 0.5 * (a2 - y_true)**2
    
    print(f"Forward Pass Output: {a2[0,0]:.4f}, Loss: {loss[0,0]:.4f}")
    
    # Backward Pass
    # dL/da2 = a2 - y_true
    delta2 = (a2 - y_true) * (a2 * (1.0 - a2))
    dW2 = np.dot(a1.T, delta2)
    db2 = delta2
    
    # Hidden layer
    delta1 = np.dot(delta2, W2.T) * (a1 * (1.0 - a1))
    dW1 = np.dot(x.T, delta1)
    db1 = delta1
    
    print("\n--- COMPUTED GRADIENTS ---")
    print("dW2:\n", dW2)
    print("db2:", db2)
    print("dW1:\n", dW1)
    print("db1:", db1)
    

# ======================================================================
# Day 05 (Topic 1.5): Exact Verification of 2-2-1 Backpropagation Step
# ======================================================================

def demo_day_05():
    print("\n--- DEMO DAY 05: Exact Verification of 2-2-1 Backpropagation Step ---")
    import numpy as np
    
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
    
    # Inputs & Targets
    x1, x2 = 0.05, 0.10
    y = 0.01
    eta = 0.5
    
    # Weights & Biases
    v11, v12 = 0.15, 0.20
    v21, v22 = 0.25, 0.30
    b1 = 0.35
    
    w1, w2 = 0.40, 0.45
    b2 = 0.60
    
    # 1. Forward Pass
    zh1 = v11*x1 + v12*x2 + b1
    ah1 = sigmoid(zh1)
    zh2 = v21*x1 + v22*x2 + b1
    ah2 = sigmoid(zh2)
    
    zo = w1*ah1 + w2*ah2 + b2
    ao = sigmoid(zo)
    loss_initial = 0.5 * (ao - y)**2
    
    print(f"Initial: ah1={ah1:.5f}, ah2={ah2:.5f}, ao={ao:.5f}, Loss={loss_initial:.5f}")
    
    # 2. Backward Pass
    delta_o = (ao - y) * ao * (1.0 - ao)
    grad_w1 = delta_o * ah1
    grad_w2 = delta_o * ah2
    
    delta_h1 = (delta_o * w1) * ah1 * (1.0 - ah1)
    delta_h2 = (delta_o * w2) * ah2 * (1.0 - ah2)
    
    grad_v11 = delta_h1 * x1
    grad_v12 = delta_h1 * x2
    
    # 3. Update
    w1_new = w1 - eta * grad_w1
    w2_new = w2 - eta * grad_w2
    v11_new = v11 - eta * grad_v11
    v12_new = v12 - eta * grad_v12
    
    # 4. Verification Forward Pass
    zh1_new = v11_new*x1 + v12_new*x2 + b1
    ah1_new = sigmoid(zh1_new)
    zo_new = w1_new*ah1_new + w2_new*ah2 + b2
    ao_new = sigmoid(zo_new)
    loss_new = 0.5 * (ao_new - y)**2
    
    print(f"Updated: w1={w1_new:.5f}, w2={w2_new:.5f}, v11={v11_new:.5f}")
    print(f"Verification: New Loss = {loss_new:.5f} (Decreased from {loss_initial:.5f}!)")
    

# ======================================================================
# Day 06 (Topic 1.6): Simulating Vanishing Gradients Across Deep Layers
# ======================================================================

def demo_day_06():
    print("\n--- DEMO DAY 06: Simulating Vanishing Gradients Across Deep Layers ---")
    import numpy as np
    
    layers = 10
    dim = 20
    
    # Initialize random weights with unit variance
    weights = [np.random.randn(dim, dim) * 0.5 for _ in range(layers)]
    
    # Input and initial gradient at output layer
    x = np.random.randn(dim, 1)
    grad = np.random.randn(dim, 1)
    
    print("=== GRADIENT NORM ACROSS 10 LAYERS ===")
    # Sigmoid scenario: max derivative ~ 0.25
    grad_sigmoid = grad.copy()
    norms_sigmoid = []
    for i in reversed(range(layers)):
        grad_sigmoid = np.dot(weights[i].T, grad_sigmoid) * 0.25
        norms_sigmoid.append(np.linalg.norm(grad_sigmoid))
    
    # ReLU scenario: active derivative ~ 1.0
    grad_relu = grad.copy()
    norms_relu = []
    for i in reversed(range(layers)):
        grad_relu = np.dot(weights[i].T, grad_relu) * 1.0
        norms_relu.append(np.linalg.norm(grad_relu))
    
    for l, (s_norm, r_norm) in enumerate(zip(reversed(norms_sigmoid), reversed(norms_relu))):
        print(f"Layer {layers - l:2d}: Sigmoid Grad Norm = {s_norm:.2e} | ReLU Grad Norm = {r_norm:.2e}")
    

# ======================================================================
# Day 07 (Topic 1.7): Optimization on an Ill-Conditioned Ravine (Rosenbrock Function)
# ======================================================================

def demo_day_07():
    print("\n--- DEMO DAY 07: Optimization on an Ill-Conditioned Ravine (Rosenbrock Function) ---")
    import numpy as np
    
    # Rosenbrock Function: f(x, y) = (1 - x)^2 + 100*(y - x^2)^2
    def rosenbrock(x, y):
        return (1.0 - x)**2 + 100.0 * (y - x**2)**2
    
    def rosenbrock_grad(x, y):
        df_dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x**2)
        df_dy = 200.0 * (y - x**2)
        return np.array([df_dx, df_dy])
    
    # Vanilla Gradient Descent
    lr = 0.001
    point = np.array([-1.5, 2.0])
    print("=== VANILLA GRADIENT DESCENT ON RAVINE ===")
    for step in range(10):
        grad = rosenbrock_grad(point[0], point[1])
        point = point - lr * grad
        loss = rosenbrock(point[0], point[1])
        print(f"Step {step+1:2d}: Point=({point[0]:.4f}, {point[1]:.4f}) | Loss={loss:.4f} | Grad Norm={np.linalg.norm(grad):.2f}")
    


if __name__ == '__main__':
    demo_day_01()
    demo_day_02()
    demo_day_03()
    demo_day_04()
    demo_day_05()
    demo_day_06()
    demo_day_07()
