"""
KTU CST414 Deep Learning: Module 4 Comprehensive Demonstrations
Faculty: Jane Rose | Dept of CSE
"""

import numpy as np
import matplotlib.pyplot as plt

# ======================================================================
# Day 23 (Topic 4.1): Minimal Vanilla RNN from Scratch in NumPy
# ======================================================================

def demo_day_23():
    print("\n--- DEMO DAY 23: Minimal Vanilla RNN from Scratch in NumPy ---")
    import numpy as np
    
    # Sequence parameters
    T = 4        # 4 time steps
    x_dim = 3    # input vector size
    h_dim = 2    # hidden state size
    y_dim = 1    # output size
    
    # Parameters
    np.random.seed(42)
    W_xh = np.random.randn(h_dim, x_dim) * 0.1
    W_hh = np.random.randn(h_dim, h_dim) * 0.1
    W_hy = np.random.randn(y_dim, h_dim) * 0.1
    b_h = np.zeros((h_dim, 1))
    b_y = np.zeros((y_dim, 1))
    
    # Input sequence: shape (T, x_dim, 1)
    X = [np.random.randn(x_dim, 1) for _ in range(T)]
    
    # Forward Pass (Unrolling over T steps)
    h_prev = np.zeros((h_dim, 1)) # initial hidden state
    h_states = [h_prev]
    outputs = []
    
    print("=== RNN FORWARD UNROLLING ===")
    for t in range(T):
        xt = X[t]
        # h(t) = tanh(W_hh * h(t-1) + W_xh * x(t) + b_h)
        ht = np.tanh(np.dot(W_hh, h_states[-1]) + np.dot(W_xh, xt) + b_h)
        yt = np.dot(W_hy, ht) + b_y
        h_states.append(ht)
        outputs.append(yt)
        print(f"Step {t+1}: Input norm={np.linalg.norm(xt):.2f} -> Hidden h={ht.ravel()} -> Output y={yt.ravel()[0]:.4f}")
    

# ======================================================================
# Day 24 (Topic 4.2): Minimal Seq2Seq Sequence Reversal in NumPy
# ======================================================================

def demo_day_24():
    print("\n--- DEMO DAY 24: Minimal Seq2Seq Sequence Reversal in NumPy ---")
    import numpy as np
    
    # Simple Encoder-Decoder
    hidden_dim = 4
    
    # Encoder: compresses 3 inputs into final vector c
    W_enc = np.random.randn(hidden_dim, 2) * 0.2
    U_enc = np.random.randn(hidden_dim, hidden_dim) * 0.2
    
    # Decoder: generates 3 outputs from context c
    W_dec = np.random.randn(hidden_dim, 2) * 0.2
    U_dec = np.random.randn(hidden_dim, hidden_dim) * 0.2
    V_out = np.random.randn(2, hidden_dim) * 0.2
    
    # Source sequence: 3 tokens
    X = [np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]]), np.array([[1.0], [1.0]])]
    
    # 1. Encode
    h = np.zeros((hidden_dim, 1))
    for xt in X:
        h = np.tanh(np.dot(W_enc, xt) + np.dot(U_enc, h))
    c = h # Context vector bottleneck!
    
    print("Encoder Final Context Vector c:\n", np.round(c.ravel(), 3))
    
    # 2. Decode with context vector
    h_dec = c.copy()
    y_prev = np.array([[0.0], [0.0]]) # <SOS> token
    generated_tokens = []
    for step in range(3):
        h_dec = np.tanh(np.dot(W_dec, y_prev) + np.dot(U_dec, h_dec))
        y_pred = np.dot(V_out, h_dec)
        generated_tokens.append(y_pred.ravel())
        y_prev = y_pred
    
    print("Decoder Generated 3 Steps:\n", np.round(generated_tokens, 3))
    

# ======================================================================
# Day 25 (Topic 4.3): Bidirectional RNN Forward Pass in pure NumPy
# ======================================================================

def demo_day_25():
    print("\n--- DEMO DAY 25: Bidirectional RNN Forward Pass in pure NumPy ---")
    import numpy as np
    
    # Sequence length T = 4, input dim = 2, hidden dim = 3
    T = 4
    x_dim = 2
    h_dim = 3
    
    X = [np.random.randn(x_dim, 1) for _ in range(T)]
    
    # Forward RNN weights
    W_xf = np.random.randn(h_dim, x_dim) * 0.1
    W_ff = np.random.randn(h_dim, h_dim) * 0.1
    
    # Backward RNN weights
    W_xb = np.random.randn(h_dim, x_dim) * 0.1
    W_bb = np.random.randn(h_dim, h_dim) * 0.1
    
    # 1. Forward Pass (t = 0 to T-1)
    h_fwd = []
    h_curr = np.zeros((h_dim, 1))
    for t in range(T):
        h_curr = np.tanh(np.dot(W_xf, X[t]) + np.dot(W_ff, h_curr))
        h_fwd.append(h_curr)
    
    # 2. Backward Pass (t = T-1 down to 0)
    h_bwd = [None] * T
    h_curr = np.zeros((h_dim, 1))
    for t in reversed(range(T)):
        h_curr = np.tanh(np.dot(W_xb, X[t]) + np.dot(W_bb, h_curr))
        h_bwd[t] = h_curr
    
    # 3. Concatenation
    print("=== BiRNN CONCATENATED HIDDEN STATES ===")
    for t in range(T):
        h_bi = np.vstack([h_fwd[t], h_bwd[t]]) # 6-dimensional
        print(f"Time Step {t+1}: Forward Norm={np.linalg.norm(h_fwd[t]):.2f} | Backward Norm={np.linalg.norm(h_bwd[t]):.2f} | BiRNN Dim={h_bi.shape[0]}")
    

# ======================================================================
# Day 26 (Topic 4.4): Full Long Short-Term Memory (LSTM) Cell in NumPy
# ======================================================================

def demo_day_26():
    print("\n--- DEMO DAY 26: Full Long Short-Term Memory (LSTM) Cell in NumPy ---")
    import numpy as np
    
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
    
    class LSTMCell:
        def __init__(self, x_dim, h_dim):
            self.x_dim = x_dim
            self.h_dim = h_dim
            concat_dim = x_dim + h_dim
            
            # Combined weights for [f, i, c_tilde, o] (4 * h_dim)
            self.W = np.random.randn(4 * h_dim, concat_dim) * 0.1
            self.b = np.zeros((4 * h_dim, 1))
            # Initialize forget gate bias to 1.0 (Jozefowicz et al. best practice!)
            self.b[:h_dim] = 1.0
    
        def forward(self, x, h_prev, c_prev):
            concat = np.vstack([h_prev, x])
            gates = np.dot(self.W, concat) + self.b
            
            f = sigmoid(gates[0*self.h_dim : 1*self.h_dim])
            i = sigmoid(gates[1*self.h_dim : 2*self.h_dim])
            c_tilde = np.tanh(gates[2*self.h_dim : 3*self.h_dim])
            o = sigmoid(gates[3*self.h_dim : 4*self.h_dim])
            
            # Cell state update
            c_next = f * c_prev + i * c_tilde
            # Hidden state update
            h_next = o * np.tanh(c_next)
            
            return h_next, c_next, (f, i, c_tilde, o)
    
    cell = LSTMCell(x_dim=2, h_dim=2)
    x_sample = np.array([[1.0], [-0.5]])
    h0 = np.zeros((2, 1))
    c0 = np.zeros((2, 1))
    
    h1, c1, gates1 = cell.forward(x_sample, h0, c0)
    print("=== LSTM STEP 1 OUTPUT ===")
    print("Forget Gate f:\n", np.round(gates1[0], 3))
    print("Input Gate  i:\n", np.round(gates1[1], 3))
    print("Updated Cell State c:\n", np.round(c1, 3))
    print("Updated Hidden State h:\n", np.round(h1, 3))
    

# ======================================================================
# Day 27 (Topic 4.5): Gated Recurrent Unit (GRU) Cell Implementation in NumPy
# ======================================================================

def demo_day_27():
    print("\n--- DEMO DAY 27: Gated Recurrent Unit (GRU) Cell Implementation in NumPy ---")
    import numpy as np
    
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
    
    class GRUCell:
        def __init__(self, x_dim, h_dim):
            self.x_dim = x_dim
            self.h_dim = h_dim
            
            # Reset and update gates (2 * h_dim)
            self.W_gates = np.random.randn(2 * h_dim, x_dim + h_dim) * 0.1
            self.b_gates = np.zeros((2 * h_dim, 1))
            
            # Candidate hidden state weights
            self.W_cand = np.random.randn(h_dim, x_dim + h_dim) * 0.1
            self.b_cand = np.zeros((h_dim, 1))
    
        def forward(self, x, h_prev):
            concat = np.vstack([h_prev, x])
            gates = np.dot(self.W_gates, concat) + self.b_gates
            
            r = sigmoid(gates[:self.h_dim])                      # Reset gate
            z = sigmoid(gates[self.h_dim : 2*self.h_dim])        # Update gate
            
            # Candidate hidden state uses reset gate to mask past
            concat_reset = np.vstack([r * h_prev, x])
            h_tilde = np.tanh(np.dot(self.W_cand, concat_reset) + self.b_cand)
            
            # Final hidden state: linear interpolation
            h_next = (1.0 - z) * h_prev + z * h_tilde
            return h_next, (r, z, h_tilde)
    
    gru = GRUCell(x_dim=2, h_dim=2)
    x = np.array([[0.5], [1.0]])
    h_prev = np.zeros((2, 1))
    
    h_next, (r, z, h_cand) = gru.forward(x, h_prev)
    print("=== GRU FORWARD STEP ===")
    print("Reset Gate  r:\n", np.round(r, 3))
    print("Update Gate z:\n", np.round(z, 3))
    print("Candidate   h_tilde:\n", np.round(h_cand, 3))
    print("Final State h:\n", np.round(h_next, 3))
    


if __name__ == '__main__':
    demo_day_23()
    demo_day_24()
    demo_day_25()
    demo_day_26()
    demo_day_27()
