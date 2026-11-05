"""
KTU CST414 Deep Learning: Module 2 Comprehensive Demonstrations
Faculty: Jane Rose | Dept of CSE
"""

import numpy as np
import matplotlib.pyplot as plt

# ======================================================================
# Day 08 (Topic 2.1): Deep Feedforward Network Forward Composition in NumPy
# ======================================================================

def demo_day_08():
    print("\n--- DEMO DAY 08: Deep Feedforward Network Forward Composition in NumPy ---")
    import numpy as np
    
    class DeepFeedforwardNetwork:
        def __init__(self, layer_dims):
            self.params = {}
            self.L = len(layer_dims) - 1
            for l in range(1, len(layer_dims)):
                # He initialization for ReLU
                self.params[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2.0 / layer_dims[l-1])
                self.params[f'b{l}'] = np.zeros((layer_dims[l], 1))
    
        def forward(self, X):
            A = X
            for l in range(1, self.L):
                Z = np.dot(self.params[f'W{l}'], A) + self.params[f'b{l}']
                A = np.maximum(0, Z) # ReLU
            # Output layer with linear / sigmoid
            ZL = np.dot(self.params[f'W{self.L}'], A) + self.params[f'b{self.L}']
            return ZL
    
    # Create a 4-layer deep network: 4 -> 16 -> 8 -> 2
    dfn = DeepFeedforwardNetwork([4, 16, 8, 2])
    sample_input = np.random.randn(4, 3) # 3 samples of dim 4
    output = dfn.forward(sample_input)
    print("Input batch shape:", sample_input.shape)
    print("Output logits shape:", output.shape)
    print("Output sample values:\n", output)
    

# ======================================================================
# Day 09 (Topic 2.2): Demonstration of Overfitting and Generalization Gap
# ======================================================================

def demo_day_09():
    print("\n--- DEMO DAY 09: Demonstration of Overfitting and Generalization Gap ---")
    import numpy as np
    
    # Generate noisy sinusoidal data
    np.random.seed(42)
    N = 15
    X_train = np.sort(np.random.uniform(0, 1, N))
    y_train = np.sin(2 * np.pi * X_train) + np.random.normal(0, 0.2, N)
    
    X_test = np.linspace(0, 1, 100)
    y_test = np.sin(2 * np.pi * X_test)
    
    # Degree 9 polynomial features
    def get_features(x, degree=9):
        return np.vstack([x**d for d in range(degree + 1)]).T
    
    Phi_train = get_features(X_train)
    Phi_test = get_features(X_test)
    
    # 1. Unregularized OLS (overfitting)
    w_ols = np.linalg.pinv(Phi_train).dot(y_train)
    
    # 2. L2 Regularized Ridge (lambda = 0.01)
    lambd = 0.01
    w_ridge = np.linalg.inv(Phi_train.T.dot(Phi_train) + lambd * np.eye(10)).dot(Phi_train.T).dot(y_train)
    
    train_err_ols = np.mean((Phi_train.dot(w_ols) - y_train)**2)
    test_err_ols = np.mean((Phi_test.dot(w_ols) - y_test)**2)
    
    train_err_ridge = np.mean((Phi_train.dot(w_ridge) - y_train)**2)
    test_err_ridge = np.mean((Phi_test.dot(w_ridge) - y_test)**2)
    
    print(f"OLS:   Train MSE = {train_err_ols:.4f} | Test MSE = {test_err_ols:.4f} (Massive Overfitting!)")
    print(f"Ridge: Train MSE = {train_err_ridge:.4f} | Test MSE = {test_err_ridge:.4f} (Generalizes Well!)")
    

# ======================================================================
# Day 10 (Topic 2.3): Vanilla GD vs Momentum on an Anisotropic Quadratic Bowl
# ======================================================================

def demo_day_10():
    print("\n--- DEMO DAY 10: Vanilla GD vs Momentum on an Anisotropic Quadratic Bowl ---")
    import numpy as np
    
    # Quadratic bowl: steep in y, gentle in x
    def cost(w):
        return 0.5 * (w[0]**2 + 20.0 * w[1]**2)
    
    def grad(w):
        return np.array([w[0], 20.0 * w[1]])
    
    w_init = np.array([-4.0, 2.0])
    lr = 0.08
    
    # 1. Vanilla Gradient Descent
    w_gd = w_init.copy()
    gd_traj = [w_gd.copy()]
    for _ in range(15):
        g = grad(w_gd)
        w_gd -= lr * g
        gd_traj.append(w_gd.copy())
    
    # 2. Momentum (beta = 0.9)
    w_mom = w_init.copy()
    v = np.zeros(2)
    beta = 0.9
    mom_traj = [w_mom.copy()]
    for _ in range(15):
        g = grad(w_mom)
        v = beta * v - lr * g
        w_mom += v
        mom_traj.append(w_mom.copy())
    
    print("After 15 steps:")
    print(f"Vanilla GD Position: ({w_gd[0]:.4f}, {w_gd[1]:.4f}) | Loss: {cost(w_gd):.4f}")
    print(f"Momentum   Position: ({w_mom[0]:.4f}, {w_mom[1]:.4f}) | Loss: {cost(w_mom):.4f}")
    

# ======================================================================
# Day 11 (Topic 2.4): Nesterov Momentum vs Standard Momentum Overshoot Comparison
# ======================================================================

def demo_day_11():
    print("\n--- DEMO DAY 11: Nesterov Momentum vs Standard Momentum Overshoot Comparison ---")
    import numpy as np
    
    def f(w):
        return 0.5 * (w[0]**2 + 50.0 * w[1]**2)
    
    def grad(w):
        return np.array([w[0], 50.0 * w[1]])
    
    w_start = np.array([-3.0, 1.0])
    lr = 0.03
    beta = 0.9
    
    # 1. Standard Momentum
    w_std = w_start.copy()
    v_std = np.zeros(2)
    std_losses = []
    for _ in range(20):
        g = grad(w_std)
        v_std = beta * v_std - lr * g
        w_std += v_std
        std_losses.append(f(w_std))
    
    # 2. Nesterov Accelerated Gradient
    w_nag = w_start.copy()
    v_nag = np.zeros(2)
    nag_losses = []
    for _ in range(20):
        w_ahead = w_nag + beta * v_nag
        g = grad(w_ahead)
        v_nag = beta * v_nag - lr * g
        w_nag += v_nag
        nag_losses.append(f(w_nag))
    
    print("Standard Momentum Final Loss (20 steps):", std_losses[-1])
    print("Nesterov Momentum Final Loss (20 steps):", nag_losses[-1])
    print(f"NAG achieves {std_losses[-1] / nag_losses[-1]:.2f}x lower error due to lookahead braking!")
    

# ======================================================================
# Day 12 (Topic 2.5): Comprehensive NumPy Implementation of Adam Optimizer
# ======================================================================

def demo_day_12():
    print("\n--- DEMO DAY 12: Comprehensive NumPy Implementation of Adam Optimizer ---")
    import numpy as np
    
    class AdamOptimizer:
        def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
            self.lr = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.eps = eps
            self.m = None
            self.v = None
            self.t = 0
    
        def update(self, params, grads):
            if self.m is None:
                self.m = np.zeros_like(params)
                self.v = np.zeros_like(params)
            
            self.t += 1
            # 1. Update biased first and second moments
            self.m = self.beta1 * self.m + (1.0 - self.beta1) * grads
            self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grads**2)
            
            # 2. Bias corrections
            m_hat = self.m / (1.0 - self.beta1**self.t)
            v_hat = self.v / (1.0 - self.beta2**self.t)
            
            # 3. Update parameters
            params_new = params - (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat
            return params_new
    
    # Verify on a 2D parameter vector
    opt = AdamOptimizer(lr=0.1)
    w = np.array([2.0, -1.0])
    print("Initial parameters:", w)
    for step in range(5):
        # Mock gradients
        grad_mock = np.array([0.4 * w[0], 2.0 * w[1]])
        w = opt.update(w, grad_mock)
        print(f"Step {step+1}: w = [{w[0]:.4f}, {w[1]:.4f}]")
    

# ======================================================================
# Day 13 (Topic 2.6): L1 vs L2 Weight Sparsity Demonstration in NumPy
# ======================================================================

def demo_day_13():
    print("\n--- DEMO DAY 13: L1 vs L2 Weight Sparsity Demonstration in NumPy ---")
    import numpy as np
    
    np.random.seed(42)
    X = np.random.randn(50, 10) # 50 samples, 10 features
    true_w = np.array([3.0, -2.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, -1.0, 0.0])
    y = np.dot(X, true_w) + np.random.normal(0, 0.1, 50)
    
    # 1. L2 Regularization (Ridge)
    alpha_l2 = 0.5
    w_l2 = np.linalg.inv(X.T.dot(X) + alpha_l2 * np.eye(10)).dot(X.T).dot(y)
    
    # 2. L1 Subgradient Descent (Lasso)
    w_l1 = np.zeros(10)
    lr = 0.01
    alpha_l1 = 0.8
    for _ in range(500):
        grad = - 2.0 * X.T.dot(y - X.dot(w_l1)) / 50.0 + alpha_l1 * np.sign(w_l1)
        w_l1 -= lr * grad
        # Soft thresholding near zero
        w_l1[np.abs(w_l1) < 0.05] = 0.0
    
    print("True Weights:   ", np.round(true_w, 2))
    print("L2 (Ridge) Wts: ", np.round(w_l2, 2))
    print("L1 (Lasso) Wts: ", np.round(w_l1, 2))
    print(f"L1 Zero Weights Count: {np.sum(w_l1 == 0)} / 10 (Sparsity Achieved!)")
    

# ======================================================================
# Day 14 (Topic 2.7): Tied-Weights Autoencoder and Input Noise Injection in NumPy
# ======================================================================

def demo_day_14():
    print("\n--- DEMO DAY 14: Tied-Weights Autoencoder and Input Noise Injection in NumPy ---")
    import numpy as np
    
    class TiedAutoencoder:
        def __init__(self, input_dim, hidden_dim):
            # Only ONE weight matrix is allocated!
            self.W = np.random.randn(hidden_dim, input_dim) * 0.1
            self.b_enc = np.zeros((hidden_dim, 1))
            self.b_dec = np.zeros((input_dim, 1))
    
        def encode(self, x, noise_std=0.1):
            # Inject noise at input
            x_noisy = x + np.random.normal(0, noise_std, x.shape)
            h = np.maximum(0, np.dot(self.W, x_noisy) + self.b_enc) # ReLU
            return h, x_noisy
    
        def decode(self, h):
            # Tied weights: W_dec = W.T
            x_rec = np.dot(self.W.T, h) + self.b_dec
            return x_rec
    
    ae = TiedAutoencoder(input_dim=8, hidden_dim=4)
    x_sample = np.random.randn(8, 1)
    h, x_noisy = ae.encode(x_sample, noise_std=0.05)
    x_recon = ae.decode(h)
    
    print("Original Input Shape:", x_sample.shape)
    print("Latent Code Shape:   ", h.shape)
    print("Reconstructed Shape: ", x_recon.shape)
    print("Parameters count (Tied):", ae.W.size + ae.b_enc.size + ae.b_dec.size, "(Untied would be 76!)")
    

# ======================================================================
# Day 15 (Topic 2.8): Variance Reduction via Deep Model Ensembling in NumPy
# ======================================================================

def demo_day_15():
    print("\n--- DEMO DAY 15: Variance Reduction via Deep Model Ensembling in NumPy ---")
    import numpy as np
    
    np.random.seed(42)
    N = 100
    X = np.linspace(-3, 3, N)
    y_true = np.sin(X)
    
    # Simulate 5 diverse neural network models with random idiosyncratic errors
    k = 5
    models = []
    for i in range(k):
        # Each model has true signal + independent random Gaussian error
        error = np.random.normal(0, 0.4, N)
        pred = y_true + error
        models.append(pred)
    
    models = np.array(models) # shape: (5, 100)
    individual_mses = [np.mean((models[i] - y_true)**2) for i in range(k)]
    
    # Ensemble prediction: simple model average
    ensemble_pred = np.mean(models, axis=0)
    ensemble_mse = np.mean((ensemble_pred - y_true)**2)
    
    print("Individual Model MSEs:", [round(m, 4) for m in individual_mses])
    print(f"Average Individual MSE: {np.mean(individual_mses):.4f}")
    print(f"Ensemble Model MSE:     {ensemble_mse:.4f} (Significant Error Reduction!)")
    

# ======================================================================
# Day 16 (Topic 2.9): Inverted Dropout and He vs Xavier Initialization in NumPy
# ======================================================================

def demo_day_16():
    print("\n--- DEMO DAY 16: Inverted Dropout and He vs Xavier Initialization in NumPy ---")
    import numpy as np
    
    def inverted_dropout(h, p=0.5, training=True):
        if not training:
            return h
        # Bernoulli mask scaled by 1/p
        mask = (np.random.rand(*h.shape) < p) / p
        return h * mask
    
    # Test Dropout forward pass
    h = np.ones((5, 10)) # 5 samples, 10 neurons
    h_dropped = inverted_dropout(h, p=0.5, training=True)
    print("=== INVERTED DROPOUT DEMO (p=0.5) ===")
    print("Original Mean:", np.mean(h))
    print("Dropped Output Mean:", np.mean(h_dropped), "(Expected value preserved!)")
    print("Dropped Sample Row 0:\n", h_dropped[0])
    
    # Initialization Variance Tracking across 10 layers
    dim = 256
    layers = 10
    x = np.random.randn(100, dim)
    
    # 1. Bad small initialization
    W_bad = np.random.randn(dim, dim) * 0.01
    a = x.copy()
    for _ in range(layers):
        a = np.maximum(0, np.dot(a, W_bad))
    print(f"\nBad Init Final Variance: {np.var(a):.2e} (Signal Collapsed!)")
    
    # 2. He Initialization for ReLU
    W_he = np.random.randn(dim, dim) * np.sqrt(2.0 / dim)
    a_he = x.copy()
    for _ in range(layers):
        a_he = np.maximum(0, np.dot(a_he, W_he))
    print(f"He  Init Final Variance: {np.var(a_he):.4f} (Signal Stable across 10 layers!)")
    


if __name__ == '__main__':
    demo_day_08()
    demo_day_09()
    demo_day_10()
    demo_day_11()
    demo_day_12()
    demo_day_13()
    demo_day_14()
    demo_day_15()
    demo_day_16()
