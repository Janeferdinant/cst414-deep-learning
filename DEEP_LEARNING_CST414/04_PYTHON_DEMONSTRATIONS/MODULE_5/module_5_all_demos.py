"""
KTU CST414 Deep Learning: Module 5 Comprehensive Demonstrations
Faculty: Jane Rose | Dept of CSE
"""

import numpy as np
import matplotlib.pyplot as plt

# ======================================================================
# Day 28 (Topic 5.1): Intersection over Union (IoU) and Non-Maximum Suppression (NMS)
# ======================================================================

def demo_day_28():
    print("\n--- DEMO DAY 28: Intersection over Union (IoU) and Non-Maximum Suppression (NMS) ---")
    import numpy as np
    
    def compute_iou(boxA, boxB):
        # box format: [x1, y1, x2, y2]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union_area = boxA_area + boxB_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    # Candidate boxes for a pedestrian: [x1, y1, x2, y2, score]
    boxes = [
        [50, 50, 150, 200, 0.92],
        [55, 52, 148, 198, 0.85],
        [52, 48, 153, 205, 0.70],
        [200, 80, 280, 220, 0.88] # different person
    ]
    
    print("IoU between Box 0 and Box 1:", round(compute_iou(boxes[0][:4], boxes[1][:4]), 3))
    print("IoU between Box 0 and Box 3:", round(compute_iou(boxes[0][:4], boxes[3][:4]), 3))
    
    # Simple NMS
    def nms(boxes, iou_thresh=0.5):
        boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
        kept = []
        while boxes:
            best = boxes.pop(0)
            kept.append(best)
            boxes = [b for b in boxes if compute_iou(best[:4], b[:4]) < iou_thresh]
        return kept
    
    print("\nBoxes kept after NMS (threshold=0.5):", len(nms(boxes)), "distinct objects identified!")
    

# ======================================================================
# Day 29 (Topic 5.2): CTC Collapse Operator and Word Error Rate (WER) in Python
# ======================================================================

def demo_day_29():
    print("\n--- DEMO DAY 29: CTC Collapse Operator and Word Error Rate (WER) in Python ---")
    def ctc_collapse(seq, blank='_'):
        # Step 1: Remove consecutive duplicates
        collapsed = []
        prev = None
        for token in seq:
            if token != prev:
                collapsed.append(token)
                prev = token
        # Step 2: Remove blanks
        result = [tok for tok in collapsed if tok != blank]
        return "".join(result)
    
    # Test CTC Collapse
    path1 = ['h', 'h', '_', 'e', 'e', 'l', '_', 'l', 'o', 'o', '_']
    path2 = ['_', 'h', 'e', 'e', 'l', 'l', 'l', 'o', '_', '_']
    print("Path 1 Decoded:", ctc_collapse(path1))
    print("Path 2 Decoded (without blank separation for 'l'):", ctc_collapse(path2))
    
    # Word Error Rate via dynamic programming
    def compute_wer(ref, hyp):
        r = ref.split()
        h = hyp.split()
        d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
        for i in range(len(r) + 1): d[i][0] = i
        for j in range(len(h) + 1): d[0][j] = j
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                if r[i-1] == h[j-1]: d[i][j] = d[i-1][j-1]
                else: d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
        return d[len(r)][len(h)] / len(r)
    
    ref_text = "deep learning is revolutionizing speech recognition"
    hyp_text = "deep learning is revolution speech recognition"
    print(f"\nReference:  '{ref_text}'")
    print(f"Hypothesis: '{hyp_text}'")
    print(f"WER:        {compute_wer(ref_text, hyp_text):.2%} (1 deletion out of 6 words)")
    

# ======================================================================
# Day 30 (Topic 5.3): Autoregressive Language Model Perplexity Calculator in NumPy
# ======================================================================

def demo_day_30():
    print("\n--- DEMO DAY 30: Autoregressive Language Model Perplexity Calculator in NumPy ---")
    import numpy as np
    
    # Sequence of 5 tokens with model predicted probabilities for true tokens
    # P(w_t | context)
    true_token_probs = np.array([0.45, 0.20, 0.80, 0.15, 0.60])
    
    # Compute cross-entropy loss per token
    token_losses = - np.log(true_token_probs)
    avg_loss = np.mean(token_losses)
    
    # Perplexity = exp(average negative log-likelihood)
    perplexity = np.exp(avg_loss)
    
    print("True Token Probabilities:", true_token_probs)
    print("Cross-Entropy Loss per token:", np.round(token_losses, 3))
    print(f"Average Sequence Loss:       {avg_loss:.4f} nats")
    print(f"Perplexity (PPL):             {perplexity:.2f}")
    print(f"Interpretation: The model is as uncertain as choosing among {perplexity:.1f} equally likely words at each step.")
    

# ======================================================================
# Day 31 (Topic 5.4): Cosine Similarity and Semantic Vector Arithmetic in NumPy
# ======================================================================

def demo_day_31():
    print("\n--- DEMO DAY 31: Cosine Similarity and Semantic Vector Arithmetic in NumPy ---")
    import numpy as np
    
    def cosine_similarity(u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    
    # Synthetic 4-dimensional word embeddings
    # Dims: [royalty, masculinity, femininity, nobility]
    embeddings = {
        'king':   np.array([0.95,  0.90, -0.10, 0.90]),
        'man':    np.array([0.05,  0.95, -0.10, 0.05]),
        'woman':  np.array([0.05, -0.10,  0.95, 0.05]),
        'queen':  np.array([0.95, -0.10,  0.90, 0.90]),
        'apple':  np.array([-0.80, 0.00,  0.00, -0.80])
    }
    
    print("Cosine Similarity(king, queen):", round(cosine_similarity(embeddings['king'], embeddings['queen']), 3))
    print("Cosine Similarity(king, apple):", round(cosine_similarity(embeddings['king'], embeddings['apple']), 3))
    
    # Vector Arithmetic: King - Man + Woman
    target = embeddings['king'] - embeddings['man'] + embeddings['woman']
    sim_to_queen = cosine_similarity(target, embeddings['queen'])
    print("\nVector Arithmetic: King - Man + Woman")
    print("Similarity of (King - Man + Woman) to Queen:", round(sim_to_queen, 3), "(High analogical alignment!)")
    

# ======================================================================
# Day 32 (Topic 5.5): GloVe Co-occurrence Probability Ratios Demonstration in NumPy
# ======================================================================

def demo_day_32():
    print("\n--- DEMO DAY 32: GloVe Co-occurrence Probability Ratios Demonstration in NumPy ---")
    import numpy as np
    
    # Synthetic co-occurrence counts matrix for target words 'ice' and 'steam'
    # Probe words: ['solid', 'gas', 'water', 'fashion']
    X_i = 10000.0 # total context occurrences for 'ice'
    X_j = 10000.0 # total context occurrences for 'steam'
    
    counts = {
        'solid':   {'ice': 190.0, 'steam': 2.0},
        'gas':     {'ice': 1.0,   'steam': 180.0},
        'water':   {'ice': 300.0, 'steam': 290.0},
        'fashion': {'ice': 0.5,   'steam': 0.6}
    }
    
    print("Probe Word  | P(k|ice)    | P(k|steam)  | Ratio P(k|ice)/P(k|steam) | Semantic Signal")
    print("-" * 75)
    for word, c in counts.items():
        p_ice = c['ice'] / X_i
        p_steam = c['steam'] / X_j
        ratio = p_ice / p_steam
        signal = "Strongly Ice" if ratio > 10 else ("Strongly Steam" if ratio < 0.1 else "Neutral / Unrelated")
        print(f"{word:11s} | {p_ice:11.5f} | {p_steam:11.5f} | {ratio:25.2f} | {signal}")
    

# ======================================================================
# Day 33 (Topic 5.6): Undercomplete and Denoising Autoencoder in NumPy
# ======================================================================

def demo_day_33():
    print("\n--- DEMO DAY 33: Undercomplete and Denoising Autoencoder in NumPy ---")
    import numpy as np
    
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
    
    class DenoisingAutoencoder:
        def __init__(self, input_dim, latent_dim):
            self.W_e = np.random.randn(latent_dim, input_dim) * 0.1
            self.b_e = np.zeros((latent_dim, 1))
            # Tied decoder weights
            self.W_d = self.W_e.T
            self.b_d = np.zeros((input_dim, 1))
    
        def forward(self, x, noise_std=0.2):
            # Corrupt input
            x_noisy = x + np.random.normal(0, noise_std, x.shape)
            # Latent code
            h = sigmoid(np.dot(self.W_e, x_noisy) + self.b_e)
            # Reconstruct clean input
            x_rec = sigmoid(np.dot(self.W_d, h) + self.b_d)
            return x_rec, h, x_noisy
    
    ae = DenoisingAutoencoder(input_dim=8, latent_dim=3)
    x_clean = np.array([[0.1, 0.9, 0.8, 0.2, 0.0, 0.7, 0.3, 0.5]]).T
    
    x_rec, h_code, x_noisy = ae.forward(x_clean, noise_std=0.15)
    print("Clean Input:       ", np.round(x_clean.ravel(), 2))
    print("Corrupted Input:   ", np.round(x_noisy.ravel(), 2))
    print("Compressed Code (3):", np.round(h_code.ravel(), 2))
    print("Reconstructed:     ", np.round(x_rec.ravel(), 2))
    

# ======================================================================
# Day 34 (Topic 5.7): Manifold Unfolding and Contrastive Similarity in NumPy
# ======================================================================

def demo_day_34():
    print("\n--- DEMO DAY 34: Manifold Unfolding and Contrastive Similarity in NumPy ---")
    import numpy as np
    
    # Generate a 2D Swiss-roll like curved manifold
    N = 100
    t = np.linspace(1.5 * np.pi, 4.5 * np.pi, N)
    x1 = t * np.cos(t)
    x2 = t * np.sin(t)
    
    # Original Ambient Space: non-linear Euclidean distance
    print("Point 0 (t=1.5*pi):", np.round([x1[0], x2[0]], 2))
    print("Point 50 (t=3.0*pi):", np.round([x1[50], x2[50]], 2))
    ambient_dist = np.linalg.norm([x1[0] - x1[50], x2[0] - x2[50]])
    print(f"Ambient Euclidean Distance: {ambient_dist:.2f}")
    
    # Learned Latent Coordinate (Manifold Geodesic parameter t)
    # The deep network learns to extract t as the 1D latent representation!
    manifold_dist = np.abs(t[50] - t[0])
    print(f"True Manifold Geodesic Distance along unrolled coordinate: {manifold_dist:.2f}")
    print("\nDeep Representation Learning discovers the intrinsic 1D coordinate t,")
    print("flattening the twisted 2D spiral into a simple 1D linear axis!")
    

# ======================================================================
# Day 35 (Topic 5.8): Restricted Boltzmann Machine (RBM) Block Gibbs Sampling in NumPy
# ======================================================================

def demo_day_35():
    print("\n--- DEMO DAY 35: Restricted Boltzmann Machine (RBM) Block Gibbs Sampling in NumPy ---")
    import numpy as np
    
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
    
    class RBM:
        def __init__(self, n_visible, n_hidden):
            self.W = np.random.randn(n_visible, n_hidden) * 0.1
            self.b = np.zeros(n_visible) # visible bias
            self.c = np.zeros(n_hidden)  # hidden bias
    
        def sample_h_given_v(self, v):
            prob_h = sigmoid(np.dot(v, self.W) + self.c)
            h_sample = (np.random.rand(*prob_h.shape) < prob_h).astype(float)
            return prob_h, h_sample
    
        def sample_v_given_h(self, h):
            prob_v = sigmoid(np.dot(h, self.W.T) + self.b)
            v_sample = (np.random.rand(*prob_v.shape) < prob_v).astype(float)
            return prob_v, v_sample
    
        def energy(self, v, h):
            return - np.dot(v, np.dot(self.W, h)) - np.dot(self.b, v) - np.dot(self.c, h)
    
    rbm = RBM(n_visible=4, n_hidden=2)
    v0 = np.array([1.0, 0.0, 1.0, 0.0])
    
    prob_h, h0 = rbm.sample_h_given_v(v0)
    prob_v, v1 = rbm.sample_v_given_h(h0)
    
    print("Visible Input v0:     ", v0)
    print("Hidden Sample h0:     ", h0)
    print("Reconstructed v1:     ", v1)
    print(f"Energy of State (v0, h0): {rbm.energy(v0, h0):.4f}")
    

# ======================================================================
# Day 36 (Topic 5.9): Contrastive Divergence (CD-1) Algorithm in NumPy
# ======================================================================

def demo_day_36():
    print("\n--- DEMO DAY 36: Contrastive Divergence (CD-1) Algorithm in NumPy ---")
    import numpy as np
    
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
    
    # Training data: 4 binary patterns
    X = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ])
    
    n_vis = 4
    n_hid = 2
    lr = 0.1
    np.random.seed(42)
    W = np.random.randn(n_vis, n_hid) * 0.1
    
    print("Initial Weights:\n", np.round(W, 3))
    
    # Train 20 epochs of CD-1
    for epoch in range(20):
        for v0 in X:
            v0 = v0.reshape(1, -1)
            # Positive Phase
            p_h0 = sigmoid(np.dot(v0, W))
            h0 = (np.random.rand(*p_h0.shape) < p_h0).astype(float)
            
            # Negative Phase (1 step of Gibbs sampling)
            p_v1 = sigmoid(np.dot(h0, W.T))
            v1 = (np.random.rand(*p_v1.shape) < p_v1).astype(float)
            p_h1 = sigmoid(np.dot(v1, W))
            
            # CD-1 Parameter Update: eta * (v0*h0^T - v1*h1^T)
            W += lr * (np.dot(v0.T, p_h0) - np.dot(v1.T, p_h1))
    
    print("\nTrained Weights after 20 epochs of CD-1:\n", np.round(W, 3))
    print("Notice how Hidden Unit 0 and Hidden Unit 1 specialize into the two distinct patterns!")
    


if __name__ == '__main__':
    demo_day_28()
    demo_day_29()
    demo_day_30()
    demo_day_31()
    demo_day_32()
    demo_day_33()
    demo_day_34()
    demo_day_35()
    demo_day_36()
