"""
Curriculum Data for Module 2: Deep Learning (9 Hours)
KTU CST414 Deep Learning (Scheme 2019)
Faculty: Jane Rose | Department of Computer Science & Engineering
"""

MODULE_2_DAYS = [
    {
        "day": 8,
        "module": 2,
        "topic_id": "2.1",
        "title": "Introduction to Deep Learning and Deep Feed-Forward Networks",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO2",
        "blooms": "Understand",
        "prerequisites": "Multi-Layer Perceptrons, Linear Algebra, Machine Learning fundamentals",
        "learning_objectives": [
            "Define Deep Learning and contrast shallow vs deep hierarchical feature representations.",
            "Formulate the mathematical architecture of Deep Feed-Forward Networks (DFNs).",
            "Analyze the theoretical necessity of depth over width (exponential circuit complexity).",
            "Trace historical milestones: Perceptrons, Connectionism, ImageNet 2012, Modern Deep Era."
        ],
        "key_terminology": [
            "Deep Learning", "Deep Feedforward Network (DFN)", "Hierarchical Feature Extraction",
            "Circuit Complexity", "Representation Depth", "Inductive Bias"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 6; Aggarwal: Chapter 1 & 4.",
        "notes": {
            "introduction": "Deep learning is a subfield of machine learning based on artificial neural networks with representation learning. The adjective 'deep' refers to the use of multiple layers in the network. A deep feedforward network (DFN), also called a multilayer perceptron, defines a mapping y = f(x; theta) and learns the value of the parameters theta that result in the best function approximation.",
            "depth_vs_width": (
                "Theoretical justification for depth:\n"
                "1. Exponential Efficiency: Certain functions requiring an exponential number of hidden units in a single-layer network can be approximated to arbitrary accuracy using a deep network with a polynomial number of units (Montufar et al., 2014; Telgarsky, 2016).\n"
                "2. Compositional Hierarchy: Real-world perceptual data (vision, audio, natural language) is inherently compositional. Deep networks mirror this physics: early layers extract low-level primitives (edges, frequencies, phonemes), middle layers compose parts (textures, contours, syllables), and deep layers represent high-level semantic concepts (faces, words, objects).\n"
                "3. Inductive Bias: Depth enforces the assumption that the target function is a composition of simpler functions, greatly constraining the search space and improving sample efficiency."
            ),
            "mathematical_composition": (
                "A deep network with L hidden layers defines a chain of function compositions:\n"
                "f(x) = f^(L)( f^(L-1)( ... f^(2)( f^(1)(x) ) ... ) )\n"
                "where each layer computes:\n"
                "h^(l) = g^(l)( (W^(l))^T * h^(l-1) + b^(l) ), with h^(0) = x.\n"
                "The parameters are the collective set theta = { W^(1), b^(1), ..., W^(L), b^(L) }."
            ),
            "exam_important_points": [
                "Contrast shallow feature engineering with end-to-end deep representation learning.",
                "Why deep architectures are exponentially more parameter-efficient than shallow ones.",
                "Function composition formulation of Deep Feed-forward Networks.",
                "The three primary driving forces behind the deep learning revolution: Big Data, GPU parallelism, Algorithmic breakthroughs (ReLU, Adam, Dropout)."
            ]
        },
        "teaching_script": {
            "00-05": "Introduce the motivation for deep learning: the failure of traditional feature engineering on high-dimensional raw sensory inputs.",
            "05-20": "Contrast shallow architectures with deep architectures. Draw the visual hierarchy: pixels -> oriented edges -> corners/motifs -> object parts -> whole objects.",
            "20-35": "Derive the mathematical composition of DFNs: f(x) = f^(L)(...f^(1)(x)...). Explain circuit complexity and why folding functions yields exponential capacity.",
            "35-45": "Discuss the three catalyst pillars: Big Data (ImageNet), Hardware (GPUs/TPUs), and Algorithmic advances (ReLU, Backprop, Initialization).",
            "45-55": "Demonstrate a multi-layer deep feedforward classifier in PyTorch/NumPy showing decision boundary warping.",
            "55-60": "Conduct 5-minute viva on depth efficiency. Announce Day 9: Optimization and Regularization Overview."
        },
        "python_demo": {
            "title": "Deep Feedforward Network Forward Composition in NumPy",
            "description": "Constructing an L-layer deep feedforward network with modular layer definitions and forward execution.",
            "code": """import numpy as np

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
print("Output sample values:\\n", output)
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Layer Dimensions and Memory Footprint in Deep Feed-Forward Network",
                "statement": "A deep feedforward network has architecture [100, 50, 25, 10, 2]. Each parameter and activation is stored as a 32-bit floating-point number (4 bytes). Calculate: (a) Total number of weight parameters, (b) Total number of bias parameters, (c) Parameter memory consumption in kilobytes, and (d) Activation memory required to store the forward pass of a mini-batch of 64 samples.",
                "given": "Dims: [100, 50, 25, 10, 2], float32 (4 bytes/val), batch size B = 64.",
                "formulas": "W_total = sum(n_l * n_{l-1}); B_total = sum(n_l); Params_mem = (W_total + B_total)*4 / 1024; Act_mem = B * sum(n_l)*4 / 1024",
                "steps": [
                    "Step 1: Weights calculation:\n  W1 = 50 * 100 = 5,000\n  W2 = 25 * 50 = 1,250\n  W3 = 10 * 25 = 250\n  W4 = 2 * 10 = 20\n  Total Weights = 5,000 + 1,250 + 250 + 20 = 6,520.",
                    "Step 2: Biases calculation:\n  B1 = 50, B2 = 25, B3 = 10, B4 = 2\n  Total Biases = 50 + 25 + 10 + 2 = 87.",
                    "Step 3: Total Parameters = 6,520 + 87 = 6,607 parameters.\n  Parameter Memory = 6,607 * 4 bytes = 26,428 bytes = 25.81 KB.",
                    "Step 4: Activations per sample (all layers 1 to 4) = 50 + 25 + 10 + 2 = 87 activations.\n  For batch of 64: Total activation floats = 64 * 87 = 5,568 floats.\n  Activation Memory = 5,568 * 4 bytes = 22,272 bytes = 21.75 KB."
                ],
                "answer": "Weights = 6,520, Biases = 87, Total Params = 6,607 (25.81 KB), Batch Activation Memory = 21.75 KB",
                "interpretation": "In deep feedforward networks, parameter storage is modest for dense layers of moderate width, but activation caching scales linearly with mini-batch size."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Theoretical Piecewise Linear Regions Bound in Deep vs Shallow ReLU Networks",
                "statement": "According to Montufar et al. (2014), a deep neural network with L layers of width n utilizing ReLU activations can partition the input space into Omega((n/n_in)^(n_in * (L-1)) * n^n_in) linear regions, whereas a shallow 1-hidden-layer network with total neurons N can partition the space into at most sum_{j=0}^{n_in} binom(N, j) regions. For an input dimension n_in = 2: (a) Calculate the upper bound of linear regions for a shallow network with N = 40 neurons, and (b) Compute the exponential lower bound for a deep network having L = 4 hidden layers of width n = 10 each (same total neurons N = 40).",
                "given": "n_in = 2, Shallow: N = 40; Deep: 4 layers with n = 10 each (total 40 neurons).",
                "formulas": "Shallow bound = binom(N, 0) + binom(N, 1) + binom(N, 2) = 1 + N + N*(N-1)/2; Deep bound approx (n/2)^(2*(L-1)) * n^2 = (10/2)^6 * 10^2 = 5^6 * 100",
                "steps": [
                    "Step 1: Compute shallow network linear regions:\n  Regions_shallow = 1 + 40 + (40 * 39 / 2) = 1 + 40 + 780 = 821 regions.",
                    "Step 2: Compute deep network linear regions:\n  With L=4 layers of width 10, the composition of folding piecewise linear operations scales as (n/2)^(d*(L-1)) * n^d.\n  For d = 2, (10/2)^(2 * (4 - 1)) * 10^2 = 5^(6) * 100 = 15,625 * 100 = 1,562,500 regions!"
                ],
                "answer": "Shallow network (40 neurons) = 821 regions. Deep network (4x10 = 40 neurons) = ~1,562,500 regions.",
                "interpretation": "With the exact same total budget of 40 hidden neurons, distributing them across 4 sequential layers creates over 1,900 times more expressive piecewise linear decision boundaries than arranging them in a single shallow layer."
            }
        ],
        "real_world_application": {
            "title": "Industrial High-Dimensional Predictive Maintenance",
            "domain": "Smart Industry 4.0 & Industrial Internet of Things (IIoT)",
            "workflow": "High-frequency vibration, thermal, and acoustic sensor arrays on wind turbines -> 6-layer Deep Feedforward Network extracts non-linear failure precursors -> Predicts remaining useful life (RUL) in hours.",
            "benefit": "Enables proactive component replacement prior to catastrophic gearbox seizure, saving over $250,000 per turbine incident."
        },
        "viva_questions": [
            {"q": "What distinguishes deep learning from classical machine learning?", "a": "Deep learning automatically extracts hierarchical feature representations directly from raw data without manual feature engineering."},
            {"q": "Why is a deep feedforward network called 'feedforward'?", "a": "Because information flows strictly forward from input nodes through hidden units to output nodes without feedback connections or recurrence."},
            {"q": "What is the theoretical advantage of depth over width?", "a": "Deep architectures can compute complex functions with exponentially fewer parameters by reusing sub-features computed in earlier layers."},
            {"q": "What is an inductive bias in neural networks?", "a": "The set of architectural assumptions the model uses to predict outputs on unseen inputs, such as hierarchical compositionality in deep networks."},
            {"q": "Why did deep networks fail to train effectively prior to 2006-2010?", "a": "Lack of massive labeled datasets, lack of GPU compute, poor weight initialization, and vanishing gradients caused by sigmoid/tanh activations."},
            {"q": "What role did ImageNet play in deep learning history?", "a": "The ImageNet Large Scale Visual Recognition Challenge (2012) proved the overwhelming superiority of deep CNNs (AlexNet) over handcrafted computer vision pipelines."},
            {"q": "What is circuit complexity theory in relation to deep networks?", "a": "A branch of theoretical computer science showing that logic circuits with limited depth require exponentially more gates to compute certain parity or multiplication functions."},
            {"q": "Can a deep feedforward network have different activation functions in different layers?", "a": "Yes; commonly ReLU or its variants are used in hidden layers, and Softmax or Sigmoid is used in the output layer depending on the task."},
            {"q": "What is representation learning?", "a": "The automated discovery of intermediate feature transformations that make downstream tasks (like classification) simple to perform."},
            {"q": "What happens if all weights in a deep feedforward network are set to zero?", "a": "All hidden neurons compute zero, and during backpropagation they receive identical zero gradients, causing training to stall completely."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "The primary theoretical benefit of deep architectures compared to shallow networks is:",
                    "options": ["Faster forward pass execution", "Exponential parameter efficiency for compositional functions", "Guaranteed convex optimization", "Complete immunity to overfitting"],
                    "ans": "Exponential parameter efficiency for compositional functions"
                },
                {
                    "q": "In a deep feedforward network, the mapping from input to output is represented as:",
                    "options": ["A recurrent state machine", "A chain of function compositions", "A Markov Decision Process", "A linear projection"],
                    "ans": "A chain of function compositions"
                },
                {
                    "q": "The number of linear regions formed by an MLP with ReLU activations scales with depth:",
                    "options": ["Linearly", "Quadratically", "Exponentially", "Logarithmically"],
                    "ans": "Exponentially"
                }
            ],
            "short_questions": [
                "Explain how deep neural networks perform hierarchical representation learning using a visual object recognition example.",
                "Discuss the trade-offs between network width and network depth in terms of parameter efficiency and trainability."
            ],
            "application_question": "Explain why shallow machine learning models failed at speech recognition and how deep feedforward networks overcome the acoustic variability challenge.",
            "homework": "Calculate the total parameters and memory footprint (in KB) for a 5-layer DFN with layer sizes [256, 128, 64, 32, 10] using 32-bit floats."
        }
    },
    {
        "day": 9,
        "module": 2,
        "topic_id": "2.2",
        "title": "Training Deep Models: Optimization and Regularization Overview",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO2",
        "blooms": "Understand",
        "prerequisites": "Loss Functions, Gradient Calculus, Day 8 concepts",
        "learning_objectives": [
            "Contrast optimization in pure mathematics vs optimization in machine learning (empirical risk vs test error).",
            "Survey the taxonomy of optimization algorithms in deep learning (first-order, second-order, adaptive).",
            "Understand the definition and purpose of regularization in deep architectures.",
            "Analyze the bias-variance decomposition and the modern deep learning 'double descent' phenomenon."
        ],
        "key_terminology": [
            "Optimization vs Generalization", "Surrogate Loss Function", "Regularization",
            "Generalization Error", "Double Descent", "Bias-Variance Tradeoff"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 8 (Section 8.1); Aggarwal: Chapter 3.",
        "notes": {
            "optimization_vs_machine_learning": (
                "In traditional numerical optimization, the objective is to minimize a known function J(theta). In machine learning, our true goal is to minimize the expected risk over the unseen data-generating distribution p_data(x, y):\n"
                "J*(theta) = E_{(x,y)~p_data} [ L(f(x; theta), y) ]\n\n"
                "We can only compute the empirical risk on training data. Moreover, machine learning often optimizes a surrogate loss function (e.g. cross-entropy loss) because the true metric (0-1 classification error) is non-smooth, non-differentiable, and NP-hard to optimize."
            ),
            "regularization_definition": (
                "Ian Goodfellow defines regularization as:\n"
                "'Any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error.'\n\n"
                "Regularization Taxonomy in Deep Learning:\n"
                "1. Parameter Penalties: Adding norm constraints to the objective (L1, L2 weight decay).\n"
                "2. Explicit Constraints: Hard parameter projection, weight clipping.\n"
                "3. Dataset Modification: Data augmentation, input noise injection.\n"
                "4. Architectural Priors: Convolutional parameter sharing, pooling.\n"
                "5. Stochastic Noise: Dropout, DropConnect, label smoothing.\n"
                "6. Algorithmic Dynamics: Early stopping, learning rate scheduling."
            ),
            "bias_variance_and_double_descent": (
                "Classical Regime: As model capacity increases, bias decreases monotonically while variance increases, producing a U-shaped test error curve.\n"
                "Modern Interpolation Regime (Belkin et al., 2019): Beyond the interpolation threshold (where the network perfectly memorizes training data with zero training error), test error undergoes a 'Double Descent' and decreases again! High overparameterization combined with stochastic gradient descent implicitly selects minimum-norm interpolating solutions that generalize remarkably well."
            ),
            "exam_important_points": [
                "Formal definition of Regularization in Deep Learning.",
                "Difference between pure optimization and machine learning optimization.",
                "Why surrogate loss functions are necessary.",
                "Sketch and explanation of classical bias-variance tradeoff vs modern double descent curve."
            ]
        },
        "teaching_script": {
            "00-05": "Pose the dilemma: 'Why do we optimize Cross-Entropy if we only care about Accuracy?'",
            "05-20": "Differentiate empirical risk minimization on training sets from true generalization risk on test sets. Explain surrogate loss functions.",
            "20-35": "Introduce Goodfellow's definition of Regularization. Map out the 6-part taxonomy of deep regularization techniques.",
            "35-45": "Present the classical Bias-Variance tradeoff. Then introduce the cutting-edge 'Double Descent' curve.",
            "45-55": "Demonstrate the effect of regularization on polynomial curve fitting in Python.",
            "55-60": "Recap and preview Day 10: Gradient Descent and Momentum."
        },
        "python_demo": {
            "title": "Demonstration of Overfitting and Generalization Gap",
            "description": "Training an over-parameterized polynomial regressor with and without regularization to observe the generalization gap.",
            "code": """import numpy as np

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
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Empirical Risk vs Surrogate Loss Evaluation",
                "statement": "A binary classifier evaluates 4 test samples with true labels y in {0, 1}. The predicted probabilities p = P(y=1) are [0.90, 0.40, 0.20, 0.85] for true labels [1, 1, 0, 0]. Using a decision threshold of 0.5: (a) Calculate the 0-1 classification error (true empirical risk), and (b) Calculate the Binary Cross-Entropy surrogate loss.",
                "given": "y = [1, 1, 0, 0], p = [0.90, 0.40, 0.20, 0.85], threshold = 0.5.",
                "formulas": "0-1 error: I(y_pred != y); L_BCE = - (1/N) * sum [ y_i * ln(p_i) + (1 - y_i) * ln(1 - p_i) ]",
                "steps": [
                    "Step 1: Compute threshold predictions:\n  p1=0.90 -> 1 (Correct, error=0)\n  p2=0.40 -> 0 (Incorrect, error=1)\n  p3=0.20 -> 0 (Correct, error=0)\n  p4=0.85 -> 1 (Incorrect, error=1).\n  Total misclassifications = 2 out of 4.",
                    "Step 2: Calculate 0-1 error: Error = 2 / 4 = 0.50 (50%).",
                    "Step 3: Calculate individual cross-entropy losses:\n  L1 = - ln(0.90) = - (-0.1054) = 0.1054\n  L2 = - ln(0.40) = - (-0.9163) = 0.9163\n  L3 = - ln(1 - 0.20) = - ln(0.80) = - (-0.2231) = 0.2231\n  L4 = - ln(1 - 0.85) = - ln(0.15) = - (-1.8971) = 1.8971.",
                    "Step 4: Average BCE loss: L_avg = (0.1054 + 0.9163 + 0.2231 + 1.8971) / 4 = 3.1419 / 4 = 0.7855."
                ],
                "answer": "0-1 Classification Error = 0.50 (50%), Cross-Entropy Surrogate Loss = 0.7855",
                "interpretation": "Sample 4 contributed the largest penalty (1.8971) because the model was highly confident (85%) in the wrong direction, demonstrating why surrogate log loss provides a smooth, informative gradient for training."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Bias-Variance Decomposition of Mean Squared Error",
                "statement": "The expected generalization error of a machine learning estimator f_hat(x) can be decomposed as E[(y - f_hat(x))^2] = Bias^2[f_hat(x)] + Var[f_hat(x)] + sigma_epsilon^2, where the true target is y = f(x) + epsilon with noise variance sigma_epsilon^2 = 0.05. An unregularized deep network has Bias = 0.10 and Variance = 0.40. An appropriately regularized network reduces Variance to 0.12 while slightly increasing Bias to 0.15. Calculate the total expected test MSE for both models and determine the percentage reduction achieved through regularization.",
                "given": "sigma_epsilon^2 = 0.05. Model 1 (Unreg): Bias = 0.10, Var = 0.40. Model 2 (Reg): Bias = 0.15, Var = 0.12.",
                "formulas": "MSE = Bias^2 + Variance + Irreducible_Noise; Reduction = (MSE1 - MSE2) / MSE1 * 100",
                "steps": [
                    "Step 1: Compute MSE for Unregularized Model 1:\n  Bias^2 = (0.10)^2 = 0.010.\n  Var = 0.40.\n  Noise = 0.05.\n  MSE_1 = 0.010 + 0.40 + 0.05 = 0.460.",
                    "Step 2: Compute MSE for Regularized Model 2:\n  Bias^2 = (0.15)^2 = 0.0225.\n  Var = 0.12.\n  Noise = 0.05.\n  MSE_2 = 0.0225 + 0.12 + 0.05 = 0.1925.",
                    "Step 3: Calculate percentage error reduction:\n  Delta = 0.460 - 0.1925 = 0.2675.\n  Percentage Reduction = (0.2675 / 0.460) * 100 = 58.15%."
                ],
                "answer": "Unregularized MSE = 0.460, Regularized MSE = 0.1925 (58.15% error reduction)",
                "interpretation": "Although regularization slightly increased the model's squared bias (+0.0125), it slashed variance by 0.28, cutting overall generalization error by over 58%."
            }
        ],
        "real_world_application": {
            "title": "Real-Time Stock Market Algorithmic Trading",
            "domain": "Quantitative Finance & Algorithmic Execution",
            "workflow": "High-frequency limit order book feeds tick-level pricing into deep network -> Pure unregularized optimization overfits to market microstructure noise -> Multi-tier regularization (L2 penalty + early stopping) ensures model generalizes across volatile macroeconomic shifts.",
            "benefit": "Prevents catastrophic portfolio drawdown during unexpected market volatility spikes."
        },
        "viva_questions": [
            {"q": "What is Goodfellow's formal definition of regularization?", "a": "Any modification made to a learning algorithm intended to reduce its generalization error but not its training error."},
            {"q": "What is a surrogate loss function?", "a": "A smooth, differentiable proxy loss (such as cross-entropy) optimized in place of the intractable true objective (such as 0-1 misclassification error)."},
            {"q": "Why is 0-1 loss intractable for gradient-based training?", "a": "Its derivative is zero almost everywhere and undefined at the decision boundary, offering no gradient direction for parameter updates."},
            {"q": "Explain the concept of 'Double Descent'.", "a": "The observation that as model capacity increases past the interpolation threshold (where training error reaches zero), test error decreases again rather than continually climbing."},
            {"q": "What is the difference between empirical risk and true risk?", "a": "Empirical risk is the average loss on the observed training sample; true risk is the expected loss over the entire underlying data distribution."},
            {"q": "What are the three components of expected test error in regression?", "a": "Squared Bias, Estimator Variance, and Irreducible Noise variance."},
            {"q": "How does early stopping act as a regularizer?", "a": "It restricts the volume of parameter space reachable from initialization, functioning mathematically similarly to an L2 weight decay penalty."},
            {"q": "What is the interpolation threshold?", "a": "The point where model parameter capacity is exactly sufficient to achieve zero training error on the entire dataset."},
            {"q": "Name three categories of regularization used in deep networks.", "a": "Norm penalties (L1/L2), noise injection (Dropout/input jitter), and dataset augmentation."},
            {"q": "Can regularization increase training loss?", "a": "Yes, by constraining model parameters, regularization often increases training loss in exchange for reducing test loss."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "The primary goal of regularization in deep learning is to reduce:",
                    "options": ["Training error", "Generalization (test) error", "Floating point operations", "Model parameter count"],
                    "ans": "Generalization (test) error"
                },
                {
                    "q": "Which of the following is considered a surrogate loss function?",
                    "options": ["0-1 loss", "Categorical Cross-Entropy", "Exact accuracy", "Precision score"],
                    "ans": "Categorical Cross-Entropy"
                },
                {
                    "q": "In the modern double descent curve, the peak of test error typically occurs at:",
                    "options": ["Zero parameters", "The interpolation threshold", "Infinite parameters", "Epoch 1"],
                    "ans": "The interpolation threshold"
                }
            ],
            "short_questions": [
                "Define regularization and provide three distinct examples of how it is implemented in deep neural networks.",
                "Explain why surrogate loss functions are used instead of direct evaluation metrics during deep model training."
            ],
            "application_question": "Explain how data augmentation acts as a regularizer for convolutional neural networks in clinical chest X-ray diagnosis.",
            "homework": "Prove that for an unbiased estimator, the expected Mean Squared Error is equal to the estimator variance plus irreducible noise."
        }
    },
    {
        "day": 10,
        "module": 2,
        "topic_id": "2.3",
        "title": "Gradient Descent (GD) and GD with Momentum",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO2",
        "blooms": "Understand / Apply",
        "prerequisites": "Vector Calculus (gradient vector), First-order Taylor Series, Day 7 concepts",
        "learning_objectives": [
            "Formulate standard Batch Gradient Descent and analyze its update equations.",
            "Analyze the failure modes of standard GD on ravines, plateaus, and saddle points.",
            "Derive the Gradient Descent with Momentum algorithm based on physical velocity principles.",
            "Analyze how momentum dampens high-frequency oscillations while accelerating along low-curvature directions."
        ],
        "key_terminology": [
            "Batch Gradient Descent", "Momentum", "Velocity Vector", "Friction Coefficient (Beta)",
            "Exponentially Weighted Moving Average", "Ravine Oscillation"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 8 (Section 8.3); Aggarwal: Chapter 3.",
        "notes": {
            "gradient_descent": (
                "Standard Gradient Descent (Cauchy, 1847) updates parameters theta in the direction of the negative gradient:\n"
                "theta_{t+1} = theta_t - eta * grad_theta J(theta_t)\n\n"
                "Failure Modes:\n"
                "1. Ravines (Ill-conditioned curvature): The gradient points across the steep valley rather than along the gentle slope toward the minimum, causing severe zig-zagging.\n"
                "2. Saddle Points & Plateaus: Near critical points, ||grad J|| -> 0, causing the parameter updates to stall.\n"
                "3. Local Minima: Without kinetic energy, GD is permanently trapped in any local depression."
            ),
            "momentum_algorithm": (
                "Polyak (1964) introduced the heavy-ball / momentum method, simulating a physical particle with mass moving down a potential energy surface subject to friction.\n\n"
                "Mathematical Formulation:\n"
                "v_{t+1} = beta * v_t + (1 - beta) * grad_theta J(theta_t)  [or simply: v_{t+1} = beta * v_t - eta * grad J]\n"
                "theta_{t+1} = theta_t + v_{t+1}\n"
                "(where beta in [0, 1) is the momentum coefficient, commonly set to 0.9).\n\n"
                "Mechanisms of Acceleration:\n"
                "- In directions where gradients alternate signs (oscillating valley walls), consecutive updates cancel out, damping oscillations.\n"
                "- In directions where gradients consistently point in the same direction, velocity accumulates up to a terminal factor of 1 / (1 - beta) times the base step size!\n"
                "- For beta = 0.9, the effective step size along consistent slopes is amplified tenfold: 1 / (1 - 0.9) = 10x."
            ),
            "exam_important_points": [
                "Full equations for Vanilla GD vs GD with Momentum.",
                "Physical interpretation of the momentum parameter beta as friction/inertia.",
                "Mathematical derivation of the terminal velocity factor 1 / (1 - beta).",
                "Graphical comparison of trajectories on an ill-conditioned quadratic bowl."
            ]
        },
        "teaching_script": {
            "00-05": "Roll a tennis ball down a curved track. Explain inertia: a heavy ball rolls up and over small bumps instead of getting trapped.",
            "05-20": "Write vanilla GD equations. Draw an elliptical valley on blackboard and sketch the zig-zag pathology.",
            "20-35": "Derive Polyak's Momentum equations: v <- beta*v - eta*g, theta <- theta + v. Define velocity and momentum factor beta.",
            "35-45": "Prove the terminal velocity formula: show how sum_{i=0}^infinity beta^i = 1 / (1 - beta). Show why beta = 0.9 yields a 10x speedup.",
            "45-55": "Demonstrate the interactive 2D loss surface Python simulation showing GD oscillating while Momentum rockets to the minimum.",
            "55-60": "Recap and link to Day 11: Nesterov Accelerated Gradient and Stochastic GD."
        },
        "python_demo": {
            "title": "Vanilla GD vs Momentum on an Anisotropic Quadratic Bowl",
            "description": "Trajectory comparison on an ill-conditioned quadratic function f(x, y) = 0.5 * (x^2 + 20*y^2).",
            "code": """import numpy as np

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
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Two-Step Momentum Velocity and Weight Calculation",
                "statement": "An optimizer with momentum coefficient beta = 0.9 and learning rate eta = 0.1 starts with initial parameter w_0 = 5.0 and initial velocity v_0 = 0.0. The gradients observed at step 0 and step 1 are g_0 = 2.0 and g_1 = 1.8. Using the update rule v_{t+1} = beta * v_t - eta * g_t, w_{t+1} = w_t + v_{t+1}, calculate the exact velocity and parameter values at steps t = 1 and t = 2.",
                "given": "w_0 = 5.0, v_0 = 0.0, beta = 0.9, eta = 0.1, g_0 = 2.0, g_1 = 1.8",
                "formulas": "v_{t+1} = beta * v_t - eta * g_t; w_{t+1} = w_t + v_{t+1}",
                "steps": [
                    "Step 1 (t = 1):\n  v_1 = (0.9 * 0.0) - (0.1 * 2.0) = 0.0 - 0.20 = -0.20.\n  w_1 = 5.0 + (-0.20) = 4.80.",
                    "Step 2 (t = 2):\n  v_2 = (0.9 * -0.20) - (0.1 * 1.8) = -0.18 - 0.18 = -0.36.\n  w_2 = 4.80 + (-0.36) = 4.44."
                ],
                "answer": "At step 1: v_1 = -0.20, w_1 = 4.80. At step 2: v_2 = -0.36, w_2 = 4.44.",
                "interpretation": "Even though the gradient decreased from 2.0 to 1.8, the accumulated velocity increased from -0.20 to -0.36, accelerating the parameter step toward the minimum."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "KTU Model Paper Q13(b): GD with vs without Momentum on Plateau",
                "statement": "Differentiate gradient descent with and without momentum. Consider a flat plateau where the gradient is constant at g = 0.05. The learning rate is eta = 0.02 and momentum factor beta = 0.9. (a) Calculate the steady-state parameter step size for vanilla GD, (b) Derive the terminal velocity step size for GD with momentum, and (c) Determine how many times faster momentum traverses the plateau compared to vanilla GD.",
                "given": "Constant gradient g = 0.05, eta = 0.02, beta = 0.9.",
                "formulas": "Vanilla step = - eta * g; Terminal velocity v_inf = - (eta * g) / (1 - beta); Speedup = v_inf / (eta * g) = 1 / (1 - beta)",
                "steps": [
                    "Step 1: Compute Vanilla GD step size:\n  step_gd = eta * g = 0.02 * 0.05 = 0.0010.",
                    "Step 2: Derive Momentum recursive sequence under constant gradient g:\n  v_1 = - eta * g\n  v_2 = beta * v_1 - eta * g = - eta * g * (1 + beta)\n  v_t = - eta * g * sum_{k=0}^{t-1} beta^k.\n  As t -> infinity, sum_{k=0}^infinity beta^k = 1 / (1 - beta).",
                    "Step 3: Calculate terminal velocity:\n  v_inf = - (0.02 * 0.05) / (1 - 0.9) = - 0.0010 / 0.10 = - 0.0100.",
                    "Step 4: Compute acceleration ratio:\n  Ratio = |v_inf| / step_gd = 0.0100 / 0.0010 = 10.0x."
                ],
                "answer": "Vanilla step = 0.0010, Momentum terminal step = 0.0100. Speedup = 10x faster.",
                "interpretation": "On flat plateaus or persistent gradients, momentum accelerates progress tenfold by accumulating velocity over past steps."
            }
        ],
        "real_world_application": {
            "title": "Robotic Arm Inverse Kinematics Trajectory Planning",
            "domain": "Robotics & Real-Time Motion Control",
            "workflow": "6-DOF robotic manipulator computes joint angles for target end-effector coordinates -> High non-linearity creates flat plateaus and steep singularities in joint space -> Momentum-accelerated optimization prevents joint velocity stalling.",
            "benefit": "Reduces robotic pick-and-place cycle times on automotive assembly lines by 24%."
        },
        "viva_questions": [
            {"q": "What is the physical analogy behind gradient descent with momentum?", "a": "A heavy ball rolling down a potential energy surface, accumulating kinetic momentum while subject to viscous friction."},
            {"q": "What is the typical value used for the momentum hyperparameter beta?", "a": "0.9 (and occasionally 0.99 for deep architectures)."},
            {"q": "How does momentum resolve the problem of oscillating gradients in narrow ravines?", "a": "Oscillating components across the valley walls have opposing signs and cancel each other out over successive steps, while consistent components along the ravine floor reinforce each other."},
            {"q": "What is the terminal velocity amplification factor of momentum?", "a": "1 / (1 - beta); for beta = 0.9, the effective step size is amplified by a factor of 10."},
            {"q": "Can momentum cause the optimizer to overshoot the global minimum?", "a": "Yes, if the accumulated velocity is too high, it may overshoot the minimum and oscillate around it before settling."},
            {"q": "What happens if beta is set to 0.0 in the momentum update?", "a": "The velocity term vanishes, reducing the algorithm strictly to vanilla Gradient Descent."},
            {"q": "What is the difference between batch gradient descent and stochastic gradient descent?", "a": "Batch GD computes gradients over the entire training set per step; SGD computes gradients using only a single training sample."},
            {"q": "Why does vanilla GD struggle in flat plateau regions?", "a": "Because the gradient magnitude ||grad L|| is near zero, resulting in microscopic parameter update steps."},
            {"q": "Does momentum alter the learning rate eta?", "a": "No, it adds an inertial velocity vector to the update, effectively scaling the distance moved without changing the base learning rate hyperparameter."},
            {"q": "Who formulated the heavy-ball momentum method?", "a": "Boris Polyak in 1964."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "The steady-state amplification factor of momentum with beta = 0.9 is:",
                    "options": ["0.9x", "1.9x", "5.0x", "10.0x"],
                    "ans": "10.0x"
                },
                {
                    "q": "In an ill-conditioned ravine, momentum helps primarily by:",
                    "options": ["Increasing the learning rate", "Cancelling out oscillating orthogonal gradients", "Computing the exact Hessian", "Converting non-convex functions to convex ones"],
                    "ans": "Cancelling out oscillating orthogonal gradients"
                },
                {
                    "q": "If velocity v_0 = 0, beta = 0.9, eta = 0.1, and gradient g_0 = 10, the first velocity v_1 is:",
                    "options": ["-0.9", "-1.0", "-10.0", "0.0"],
                    "ans": "-1.0"
                }
            ],
            "short_questions": [
                "Write down the update equations for Gradient Descent with Momentum and explain the physical role of beta.",
                "Explain why Vanilla Gradient Descent oscillates in ravines and how momentum resolves this issue."
            ],
            "application_question": "Explain how momentum enables a deep network optimizer to escape shallow local minima during training.",
            "homework": "Prove that if beta = 0.95, the terminal velocity factor on a constant gradient plateau is 20x the base step size."
        }
    },
    {
        "day": 11,
        "module": 2,
        "topic_id": "2.4",
        "title": "Nesterov Accelerated Gradient (NAG) and Stochastic Gradient Descent (SGD)",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO2",
        "blooms": "Understand / Apply",
        "prerequisites": "Gradient Descent, Momentum Algorithm, Day 10 concepts",
        "learning_objectives": [
            "Formulate Nesterov Accelerated Gradient (NAG) and explain its 'look-ahead' mechanism.",
            "Compare the theoretical convergence rates: GD O(1/k) vs Nesterov O(1/k^2).",
            "Formulate Stochastic Gradient Descent (SGD) and Mini-Batch SGD.",
            "Analyze the variance of stochastic gradients, learning rate decay schedules, and Robbins-Monro conditions."
        ],
        "key_terminology": [
            "Nesterov Accelerated Gradient", "Look-Ahead Step", "Stochastic Gradient Descent",
            "Mini-Batch SGD", "Robbins-Monro Conditions", "Gradient Noise"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 8 (Section 8.3 & 8.5); Aggarwal: Chapter 3.",
        "notes": {
            "nesterov_accelerated_gradient": (
                "Yurii Nesterov (1983) modified classical momentum by computing the gradient not at the current position theta_t, but at the 'look-ahead' position (theta_t + beta * v_t):\n\n"
                "NAG Equations:\n"
                "1. Look-ahead projection: theta_lookahead = theta_t + beta * v_t\n"
                "2. Evaluate gradient at projected point: g_t = grad_theta J(theta_lookahead)\n"
                "3. Update velocity: v_{t+1} = beta * v_t - eta * g_t\n"
                "4. Update parameters: theta_{t+1} = theta_t + v_{t+1}\n\n"
                "Advantage: NAG acts as an adaptive brake. If the current velocity is carrying the parameters uphill, the look-ahead gradient detects the slope increase BEFORE the step is taken and immediately applies a counter-acting brake, significantly reducing overshooting."
            ),
            "stochastic_gradient_descent": (
                "Standard Batch GD computes the exact gradient over all N training examples: cost O(N) per step. For millions of samples, this is computationally prohibitive.\n\n"
                "Stochastic Gradient Descent (Robbins & Monro, 1951) approximates the true gradient using a single randomly chosen sample (or a mini-batch B of size 32 to 512):\n"
                "g_t = (1 / |B|) * sum_{i in B} grad_theta L(f(x_i; theta), y_i)\n"
                "theta_{t+1} = theta_t - eta_t * g_t\n\n"
                "Robbins-Monro Convergence Conditions:\n"
                "To guarantee almost-sure convergence to a local minimum under stochastic noise:\n"
                "1. sum_{t=1}^infinity eta_t = infinity (learning rate must allow traversing any distance)\n"
                "2. sum_{t=1}^infinity eta_t^2 < infinity (learning rate must shrink fast enough to damp noise)"
            ),
            "convergence_comparison": (
                "Theoretical convergence rates on convex L-smooth functions:\n"
                "- Standard Gradient Descent: O(1 / k) after k iterations.\n"
                "- Nesterov Accelerated Gradient: O(1 / k^2) — proved to be the optimal theoretical lower bound for first-order black-box optimization!"
            ),
            "exam_important_points": [
                "Formulate NAG update equations and contrast with classical Polyak momentum.",
                "Diagram illustrating the look-ahead step in Nesterov momentum.",
                "Formulation of Mini-Batch SGD and its computational trade-offs.",
                "Robbins-Monro conditions for convergence of stochastic gradient descent."
            ]
        },
        "teaching_script": {
            "00-05": "Analogy for Nesterov: Polyak momentum is like a car driving blindfolded with mass; Nesterov is like a driver who looks ahead before pressing the accelerator.",
            "05-20": "Draw vector diagram showing difference between Standard Momentum and Nesterov. Write the look-ahead gradient equations.",
            "20-35": "Discuss the O(1/k) vs O(1/k^2) convergence rate proof significance. Explain how Nesterov provides self-correcting braking.",
            "35-45": "Introduce Stochastic Gradient Descent (SGD) and Mini-batch SGD. Contrast batch size 1, 64, and full N in terms of variance and GPU throughput.",
            "45-55": "Derive the Robbins-Monro conditions: sum eta_t = infty, sum eta_t^2 < infty.",
            "55-60": "Recap and preview Day 12: Adaptive Optimizers (AdaGrad, RMSProp, Adam)."
        },
        "python_demo": {
            "title": "Nesterov Momentum vs Standard Momentum Overshoot Comparison",
            "description": "Simulating optimization on a sharp valley to observe Nesterov's look-ahead braking reducing overshoot oscillations.",
            "code": """import numpy as np

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
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Nesterov Look-Ahead Position and Gradient Evaluation",
                "statement": "An optimizer using Nesterov momentum has current parameter w_t = 2.0, current velocity v_t = -0.5, and momentum factor beta = 0.9. The loss function is J(w) = 0.5 * w^2. Learning rate is eta = 0.1. (a) Calculate the look-ahead position w_ahead, (b) Compute the gradient at the look-ahead position, and (c) Compute the updated velocity v_{t+1} and new parameter w_{t+1}.",
                "given": "w_t = 2.0, v_t = -0.5, beta = 0.9, J(w) = 0.5 * w^2, eta = 0.1.",
                "formulas": "w_ahead = w_t + beta * v_t; g_t = dJ/dw(w_ahead) = w_ahead; v_{t+1} = beta * v_t - eta * g_t; w_{t+1} = w_t + v_{t+1}",
                "steps": [
                    "Step 1: Compute look-ahead position:\n  w_ahead = 2.0 + (0.9 * -0.5) = 2.0 - 0.45 = 1.55.",
                    "Step 2: Evaluate gradient at look-ahead position:\n  g_ahead = dJ/dw(1.55) = 1.55.",
                    "Step 3: Update velocity:\n  v_{t+1} = (0.9 * -0.5) - (0.1 * 1.55) = -0.45 - 0.155 = -0.605.",
                    "Step 4: Update parameter:\n  w_{t+1} = 2.0 + (-0.605) = 1.395."
                ],
                "answer": "w_ahead = 1.55, Gradient = 1.55, v_{t+1} = -0.605, w_{t+1} = 1.395",
                "interpretation": "Nesterov evaluated the slope where momentum was about to carry the parameter (at 1.55) rather than at the stale position (2.0), adjusting velocity dynamically."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Stochastic Gradient Variance Reduction with Mini-Batch Size",
                "statement": "Let the true population gradient be g_true with variance Var(g_single) = sigma^2 = 16.0 when computed on a single random sample. (a) Calculate the variance of the mini-batch gradient estimator for batch sizes B = 1, B = 16, B = 64, and B = 256, (b) Determine the standard error of the gradient estimate for each batch size, and (c) Discuss the computational trade-off between variance reduction and matrix multiply saturation on a GPU.",
                "given": "sigma^2 = 16.0 for B = 1.",
                "formulas": "Var(g_B) = sigma^2 / B; SE(g_B) = sqrt(Var(g_B)) = sigma / sqrt(B)",
                "steps": [
                    "Step 1: For B = 1:\n  Var = 16.0 / 1 = 16.0; SE = sqrt(16.0) = 4.00.",
                    "Step 2: For B = 16:\n  Var = 16.0 / 16 = 1.0; SE = sqrt(1.0) = 1.00.",
                    "Step 3: For B = 64:\n  Var = 16.0 / 64 = 0.25; SE = sqrt(0.25) = 0.50.",
                    "Step 4: For B = 256:\n  Var = 16.0 / 256 = 0.0625; SE = sqrt(0.0625) = 0.25."
                ],
                "answer": "B=1: SE=4.00; B=16: SE=1.00; B=64: SE=0.50; B=256: SE=0.25.",
                "interpretation": "Increasing batch size from 1 to 64 reduces gradient standard error by a factor of 8 (from 4.0 to 0.5). Beyond B=256, returns diminish as diminishing variance gains require quadrupling the batch size to halve error, while GPU memory fills."
            }
        ],
        "real_world_application": {
            "title": "Real-Time Embedded Video Object Tracking",
            "domain": "Edge AI & Drone Surveillance Systems",
            "workflow": "Drone onboard camera streams 60 fps 4K video -> High frame rate requires sub-15ms model updates on edge GPU -> Mini-Batch SGD with Nesterov momentum optimizes lightweight detector parameters on-the-fly without tracking lag.",
            "benefit": "Enables uninterrupted drone camera lock-on for search-and-rescue target tracking in mountain terrain."
        },
        "viva_questions": [
            {"q": "What is the key difference between Polyak's standard momentum and Nesterov momentum?", "a": "Polyak evaluates the gradient at the current position theta_t, whereas Nesterov evaluates the gradient at the look-ahead position (theta_t + beta*v_t)."},
            {"q": "What is the convergence rate of Nesterov Accelerated Gradient for convex smooth functions?", "a": "O(1 / k^2), which is significantly faster than standard GD's O(1 / k)."},
            {"q": "Why is pure SGD (batch size = 1) computationally inefficient on GPUs?", "a": "GPUs achieve high compute utilization through massive vector and matrix parallelization; a single sample fails to saturate thousands of CUDA cores."},
            {"q": "State the Robbins-Monro conditions for stochastic optimization.", "a": "The learning rate sequence eta_t must satisfy: sum_{t=1}^infty eta_t = infty and sum_{t=1}^infty eta_t^2 < infty."},
            {"q": "Why is the first Robbins-Monro condition necessary?", "a": "To ensure the optimizer has sufficient energy to travel an arbitrarily large distance in parameter space to reach the minimum regardless of initialization."},
            {"q": "Why is the second Robbins-Monro condition necessary?", "a": "To ensure that the variance of parameter updates asymptotically approaches zero, allowing the optimizer to settle into the minimum rather than oscillating forever due to gradient noise."},
            {"q": "How does mini-batch SGD balance variance and throughput?", "a": "It averages out stochastic sample noise (reducing variance by factor B) while fully utilizing parallel tensor cores on modern hardware."},
            {"q": "Can stochastic gradient noise be beneficial in non-convex optimization?", "a": "Yes, gradient noise provides random kinetic energy that helps the parameters escape sharp, poorly generalizing local minima and saddle points."},
            {"q": "What is learning rate warmup?", "a": "Starting training with a very small learning rate for the first few epochs to prevent large, noisy initial gradients from destabilizing uncalibrated weights."},
            {"q": "What is linear learning rate scaling rule?", "a": "When the mini-batch size is multiplied by k, the learning rate should also be multiplied by k to maintain similar gradient step magnitudes per epoch."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "Nesterov Accelerated Gradient computes the gradient at:",
                    "options": ["The current position theta_t", "The look-ahead position theta_t + beta*v_t", "The origin (0, 0)", "The previous step position theta_{t-1}"],
                    "ans": "The look-ahead position theta_t + beta*v_t"
                },
                {
                    "q": "The optimal theoretical convergence rate for first-order smooth convex optimization achieved by NAG is:",
                    "options": ["O(1/k)", "O(1/sqrt(k))", "O(1/k^2)", "O(e^(-k))"],
                    "ans": "O(1/k^2)"
                },
                {
                    "q": "If the mini-batch size is increased from 16 to 64, the variance of the mini-batch gradient estimator is reduced by a factor of:",
                    "options": ["2", "4", "8", "16"],
                    "ans": "4"
                }
            ],
            "short_questions": [
                "Formulate the complete update equations for Nesterov Accelerated Gradient and explain the intuition behind the look-ahead mechanism.",
                "State the Robbins-Monro conditions and explain their physical and mathematical significance in stochastic convergence."
            ],
            "application_question": "Explain why large-batch training (e.g. batch size > 8192) often leads to a 'generalization gap' compared to small mini-batch training.",
            "homework": "Show that the harmonic learning rate schedule eta_t = c / t satisfies both Robbins-Monro conditions."
        }
    },
    {
        "day": 12,
        "module": 2,
        "topic_id": "2.5",
        "title": "Adaptive Learning Rate Methods: AdaGrad, RMSProp, and Adam",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO2",
        "blooms": "Understand / Apply",
        "prerequisites": "Gradient Descent with Momentum, Element-wise vector operations, Day 11 concepts",
        "learning_objectives": [
            "Formulate the AdaGrad algorithm and analyze its per-parameter adaptive learning rate mechanism.",
            "Identify the fatal flaw of AdaGrad (monotonic accumulation of squared gradients causing premature freezing).",
            "Formulate RMSProp and explain how exponential moving averages resolve AdaGrad's learning rate decay.",
            "Formulate the Adam optimizer (Kingma & Ba, 2014) including first/second moment estimation and bias corrections."
        ],
        "key_terminology": [
            "AdaGrad", "RMSProp", "Adam (Adaptive Moment Estimation)", "Second Moment Accumulator",
            "Exponential Moving Average", "Bias Correction", "Per-Coordinate Learning Rate"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 8 (Section 8.5); KTU CST414 Syllabus Model Question Paper Part B (Q13a).",
        "notes": {
            "adagrad": (
                "AdaGrad (Duchi, Hazan, Singer, 2011) adapts the learning rate to each individual parameter based on the history of past squared gradients:\n"
                "1. Accumulate squared gradients: s_t = s_{t-1} + g_t odot g_t\n"
                "2. Parameter update: theta_{t+1} = theta_t - (eta / sqrt(s_t + epsilon)) odot g_t\n\n"
                "Merit: Automatically takes large steps for infrequent/sparse features and small steps for frequent features.\n"
                "Fatal Flaw: Since g_t^2 >= 0, s_t increases monotonically with every training step. Consequently, the effective learning rate eta / sqrt(s_t) strictly decays to zero, prematurely terminating learning before the network reaches the optimum!"
            ),
            "rmsprop": (
                "RMSProp (Geoff Hinton, Coursera Lecture 6e, 2012) fixes AdaGrad's premature freezing by replacing the monotonic sum with an Exponentially Weighted Moving Average (EWMA):\n"
                "1. Update discounted second moment: s_t = beta_2 * s_{t-1} + (1 - beta_2) * (g_t odot g_t)\n"
                "   (typically beta_2 = 0.9 or 0.99)\n"
                "2. Parameter update: theta_{t+1} = theta_t - (eta / sqrt(s_t + epsilon)) odot g_t\n\n"
                "This allows RMSProp to discard stale gradient history and adapt dynamically to recent curvature."
            ),
            "adam": (
                "Adam (Kingma & Ba, ICLR 2015) combines the advantages of Momentum (first moment estimation) and RMSProp (second moment estimation):\n"
                "1. First moment (mean of gradient): m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t  (default beta_1 = 0.9)\n"
                "2. Second moment (uncentered variance): v_t = beta_2 * v_{t-1} + (1 - beta_2) * (g_t odot g_t)  (default beta_2 = 0.999)\n"
                "3. Bias corrections (counteracting initialization at zero):\n"
                "   m_hat_t = m_t / (1 - beta_1^t)\n"
                "   v_hat_t = v_t / (1 - beta_2^t)\n"
                "4. Parameter update: theta_{t+1} = theta_t - (eta / (sqrt(v_hat_t) + epsilon)) odot m_hat_t\n"
                "   (default eta = 0.001, epsilon = 1e-8)"
            ),
            "exam_important_points": [
                "KTU Exam Q13(a): Mathematical explanation of AdaGrad gradient decay and why it halts training.",
                "Comparison table: SGD vs AdaGrad vs RMSProp vs Adam.",
                "Full mathematical derivation of Adam's bias correction terms 1 / (1 - beta^t).",
                "Why Adam is the de facto default optimizer across modern deep learning architectures."
            ]
        },
        "teaching_script": {
            "00-05": "Introduce the challenge: different parameters in a deep network have wildly different gradient magnitudes. A single global learning rate is suboptimal.",
            "05-20": "Formulate AdaGrad. Walk through Duchi's equation. Explain why the denominator s_t = sum g_tau^2 grows monotonically and kills learning.",
            "20-35": "Introduce Hinton's RMSProp solution: replace sum with exponentially decaying average s_t = beta*s + (1-beta)*g^2.",
            "35-45": "Synthesize Adam as First Moment (Momentum) + Second Moment (RMSProp). Derive the bias correction equations.",
            "45-55": "Demonstrate the interactive optimizer race simulation comparing SGD, AdaGrad, RMSProp, and Adam.",
            "55-60": "Recap and preview Day 13: Regularization Techniques (L1, L2, Early Stopping, Data Augmentation)."
        },
        "python_demo": {
            "title": "Comprehensive NumPy Implementation of Adam Optimizer",
            "description": "Building the complete Adam optimizer from scratch with first/second moments and bias corrections.",
            "code": """import numpy as np

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
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "AdaGrad vs RMSProp Two-Step Accumulator Comparison",
                "statement": "A parameter w has initial value 1.0. At step 1 and step 2, the observed gradients are g1 = 4.0 and g2 = 4.0. The base learning rate is eta = 0.5 and epsilon = 1e-8. For RMSProp, use decay factor beta_2 = 0.9. Calculate: (a) AdaGrad's accumulator s_1, s_2 and effective learning rates, and (b) RMSProp's accumulator s_1, s_2 and effective learning rates.",
                "given": "g1 = 4.0, g2 = 4.0, eta = 0.5, epsilon = 1e-8, RMSProp beta_2 = 0.9.",
                "formulas": "AdaGrad: s_t = s_{t-1} + g_t^2; RMSProp: s_t = beta_2 * s_{t-1} + (1 - beta_2) * g_t^2; eta_eff = eta / sqrt(s_t + eps)",
                "steps": [
                    "Step 1: AdaGrad calculations:\n  s_1 = 0 + 4.0^2 = 16.0 -> eta_eff_1 = 0.5 / sqrt(16.0) = 0.5 / 4.0 = 0.1250.\n  s_2 = 16.0 + 4.0^2 = 16.0 + 16.0 = 32.0 -> eta_eff_2 = 0.5 / sqrt(32.0) = 0.5 / 5.6569 = 0.0884.",
                    "Step 2: RMSProp calculations:\n  s_1 = (0.9 * 0) + (1 - 0.9) * 4.0^2 = 0.1 * 16.0 = 1.60.\n  eta_eff_1 = 0.5 / sqrt(1.60) = 0.5 / 1.2649 = 0.3953.\n  s_2 = (0.9 * 1.60) + (0.1 * 16.0) = 1.44 + 1.60 = 3.04.\n  eta_eff_2 = 0.5 / sqrt(3.04) = 0.5 / 1.7436 = 0.2868."
                ],
                "answer": "AdaGrad: s_2 = 32.0, eta_eff = 0.0884. RMSProp: s_2 = 3.04, eta_eff = 0.2868.",
                "interpretation": "AdaGrad's accumulator grew rapidly to 32.0, suffocating the effective step size by ~82%. RMSProp maintained a balanced discounted accumulator (3.04), allowing continued agile learning."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Adam Bias Correction Derivation and Step Calculation",
                "statement": "At step t = 1, an Adam optimizer observes gradient g_1 = 2.0. Hyperparameters are eta = 0.01, beta_1 = 0.9, beta_2 = 0.999, and epsilon = 1e-8. Initial moments are m_0 = 0.0, v_0 = 0.0. (a) Calculate the raw moments m_1 and v_1, (b) Calculate the bias-corrected moments m_hat_1 and v_hat_1, (c) Explain why bias correction is mandatory at early steps, and (d) Compute the parameter update step delta_w.",
                "given": "t = 1, g_1 = 2.0, eta = 0.01, beta_1 = 0.9, beta_2 = 0.999, eps = 1e-8, m_0 = 0, v_0 = 0.",
                "formulas": "m_1 = beta_1*m_0 + (1 - beta_1)*g_1; v_1 = beta_2*v_0 + (1 - beta_2)*g_1^2; m_hat = m_1 / (1 - beta_1^1); v_hat = v_1 / (1 - beta_2^1); delta_w = - (eta / (sqrt(v_hat) + eps)) * m_hat",
                "steps": [
                    "Step 1: Compute raw moments:\n  m_1 = (0.9 * 0) + (1 - 0.9) * 2.0 = 0.1 * 2.0 = 0.20.\n  v_1 = (0.999 * 0) + (1 - 0.999) * (2.0)^2 = 0.001 * 4.0 = 0.0040.",
                    "Step 2: Compute bias correction denominators:\n  1 - beta_1^1 = 1 - 0.9 = 0.10.\n  1 - beta_2^1 = 1 - 0.999 = 0.0010.",
                    "Step 3: Compute bias-corrected moments:\n  m_hat_1 = 0.20 / 0.10 = 2.00.\n  v_hat_1 = 0.0040 / 0.0010 = 4.00.",
                    "Step 4: Compute parameter update step:\n  delta_w = - (0.01 / (sqrt(4.00) + 1e-8)) * 2.00 = - (0.01 / 2.0) * 2.00 = - 0.0100."
                ],
                "answer": "Raw: m_1=0.20, v_1=0.0040. Corrected: m_hat_1=2.00, v_hat_1=4.00. Step delta_w = -0.0100",
                "interpretation": "Without bias correction, v_1 would be 0.0040, giving sqrt(v_1) = 0.0632 and resulting in a wildly distorted, inflated initial step. Bias correction normalizes both moments to reflect the true expected gradient scale."
            }
        ],
        "real_world_application": {
            "title": "Autonomous Drone LiDAR Simultaneous Localization and Mapping (SLAM)",
            "domain": "Autonomous Robotics & Edge Spatial Computing",
            "workflow": "Drone flies through unmapped indoor warehouse -> LiDAR point cloud features are sparse (certain spatial bins trigger rarely while floor points trigger constantly) -> Adam optimizer adapts individual coordinate step sizes dynamically.",
            "benefit": "Enables rapid convergence of 3D spatial pose graph estimation in sub-second real-time constraints."
        },
        "viva_questions": [
            {"q": "What core problem in vanilla GD did AdaGrad aim to solve?", "a": "The reliance on a single global learning rate for all parameters, regardless of how frequently or sparsely individual features occurred."},
            {"q": "What is the fatal flaw of the AdaGrad optimizer?", "a": "The monotonic accumulation of squared gradients in the denominator causes the effective learning rate to decay to near zero, prematurely halting training."},
            {"q": "How does RMSProp overcome AdaGrad's learning rate decay problem?", "a": "By using an exponentially weighted moving average of squared gradients instead of an unweighted cumulative sum."},
            {"q": "What two algorithms are combined to create Adam?", "a": "Momentum (first moment of gradients) and RMSProp (second moment of gradients)."},
            {"q": "Why are bias corrections needed in the Adam optimizer?", "a": "Because the first and second moment vectors are initialized to zero, biasing their estimates towards zero during the initial training steps; dividing by (1 - beta^t) removes this bias."},
            {"q": "What are the standard recommended default hyperparameter values for Adam?", "a": "Learning rate eta = 0.001, beta_1 = 0.9, beta_2 = 0.999, and epsilon = 1e-8."},
            {"q": "What is the physical meaning of the ratio m_hat / sqrt(v_hat) in Adam?", "a": "It approximates a Signal-to-Noise Ratio (SNR): large when gradients consistently point in one direction with low variance, and small when gradients are noisy and erratic."},
            {"q": "What is AdamW?", "a": "A variant of Adam (Loshchilov & Hutter, 2017) that decouples L2 weight decay from the gradient update, fixing flawed weight shrinkage in original Adam."},
            {"q": "Why does Adam generally outperform standard SGD in deep networks with sparse features?", "a": "It scales updates inversely with gradient frequency/magnitude, giving sparse features large enough updates to be learned effectively."},
            {"q": "What is the role of epsilon in adaptive optimizers?", "a": "A tiny smoothing constant (e.g. 1e-8) added to the denominator to prevent division by zero when gradients are zero."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "The primary reason AdaGrad fails during extended deep network training is:",
                    "options": ["Exploding gradients", "Monotonically increasing denominator decaying learning rate to zero", "Oscillating velocity vectors", "Inability to compute second derivatives"],
                    "ans": "Monotonically increasing denominator decaying learning rate to zero"
                },
                {
                    "q": "In the Adam optimizer, beta_1 and beta_2 control:",
                    "options": ["L1 and L2 regularization penalties", "The decay rates of the first and second moment estimates", "The network layer width and depth", "The dropout probability rate"],
                    "ans": "The decay rates of the first and second moment estimates"
                },
                {
                    "q": "At training step t = 1 with beta_1 = 0.9, the bias correction factor 1 / (1 - beta_1^t) equals:",
                    "options": ["1.0", "0.1", "10.0", "0.9"],
                    "ans": "10.0"
                }
            ],
            "short_questions": [
                "Explain what happens in AdaGrad where momentum is expressed with cumulative squared gradients in the denominator, and describe why this causes premature convergence.",
                "Derive the mathematical formulas for the bias-corrected first and second moment estimates in Adam."
            ],
            "application_question": "Explain why Adam is preferred over SGD for training Natural Language Processing Transformer models with sparse vocabulary embeddings.",
            "homework": "Prove that for an exponential moving average v_t = beta*v_{t-1} + (1 - beta)*g_t^2 with v_0 = 0, the expected value E[v_t] equals E[g^2] * (1 - beta^t)."
        }
    },
    {
        "day": 13,
        "module": 2,
        "topic_id": "2.6",
        "title": "Regularization Techniques: L1, L2, Early Stopping, and Dataset Augmentation",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO2",
        "blooms": "Understand / Apply",
        "prerequisites": "Loss Formulation, Gradient Updates, Optimization Overview, Day 9 concepts",
        "learning_objectives": [
            "Formulate L1 (Lasso) and L2 (Ridge / Weight Decay) parameter regularization penalties.",
            "Compare the geometric contours of L1 vs L2 and explain why L1 induces parameter sparsity.",
            "Analyze the mechanism and algorithmic implementation of Early Stopping.",
            "Evaluate Dataset Augmentation techniques across image, audio, and tabular modalities."
        ],
        "key_terminology": [
            "L1 Regularization (Lasso)", "L2 Regularization (Weight Decay / Ridge)", "Sparsity",
            "Weight Shrinkage", "Early Stopping", "Patience", "Dataset Augmentation"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 7 (Sections 7.1, 7.4, 7.8); Aggarwal: Chapter 4.",
        "notes": {
            "l1_and_l2_regularization": (
                "Regularized Objective Function:\n"
                "J_tilde(w) = J(w) + alpha * Omega(w)\n\n"
                "1. L2 Regularization (Weight Decay / Ridge):\n"
                "   Omega(w) = (1/2) * ||w||_2^2 = (1/2) * sum_i w_i^2\n"
                "   Gradient: grad J_tilde = grad J + alpha * w\n"
                "   Update Step: w_{t+1} = w_t - eta * (grad J + alpha * w_t) = (1 - eta * alpha) * w_t - eta * grad J\n"
                "   Effect: Multiplies the weight vector by a shrinkage factor (1 - eta * alpha) < 1 at each step, preventing weights from growing excessively large.\n\n"
                "2. L1 Regularization (Lasso):\n"
                "   Omega(w) = ||w||_1 = sum_i |w_i|\n"
                "   Gradient: grad J_tilde = grad J + alpha * sign(w)\n"
                "   Update Step: w_{t+1} = w_t - eta * grad J - eta * alpha * sign(w_t)\n"
                "   Geometric Sparsity: The L1 constraint forms a diamond-shaped polytope with sharp corners on the coordinate axes. The elliptical loss contours intersect these corners, driving many weights to be exactly zero (feature selection)."
            ),
            "early_stopping": (
                "Early Stopping is the most widely used regularization technique in deep learning.\n"
                "Protocol:\n"
                "1. At the end of each epoch, evaluate loss on a hold-out validation set.\n"
                "2. If validation loss improves, save model parameters theta* = theta.\n"
                "3. If validation loss does not improve for 'patience' P consecutive epochs, terminate training and restore theta*.\n\n"
                "Mathematical Equivalence: Bishop (1995) proved that on quadratic loss surfaces, early stopping after tau iterations is mathematically equivalent to L2 regularization with weight decay parameter alpha approx 1 / (eta * tau)."
            ),
            "dataset_augmentation": (
                "The most reliable method to improve generalization is to train on more data. When additional data is scarce, Dataset Augmentation generates realistic synthetic transformations:\n"
                "- Computer Vision: Random horizontal flips, random cropping, rotations (+-15 deg), color jitter, affine shearing, CutMix, MixUp.\n"
                "- Speech: Time stretching, pitch shifting, background noise injection (SpecAugment).\n"
                "- Text: Synonym replacement, back-translation (English -> German -> English)."
            ),
            "exam_important_points": [
                "Derivation of weight shrinkage equation w <- (1 - eta*alpha)*w for L2 regularization.",
                "Geometric explanation with sketches showing why L1 produces sparsity while L2 shrinks weights uniformly.",
                "Early stopping protocol, patience parameter, and its equivalence to L2 regularization.",
                "List of data augmentation techniques for image classification."
            ]
        },
        "teaching_script": {
            "00-05": "Introduce Occam's Razor: simpler models with smaller weights generalize better than complex models with wild parameter spikes.",
            "05-20": "Formulate L2 Regularization. Derive the weight decay equation: w <- (1 - eta*alpha)*w - eta*g. Show why it is called 'weight decay'.",
            "20-35": "Formulate L1 Regularization. Draw the classic 2D contour comparison: circular L2 ball vs diamond L1 polytope. Explain why corners induce sparsity.",
            "35-45": "Present Early Stopping: show the validation loss curve inflection point. Define the 'patience' hyperparameter.",
            "45-55": "Demonstrate the Python code showing how L1 sets 80% of neural network weights to exactly 0.0.",
            "55-60": "Recap and preview Day 14: Parameter Sharing, Tying, and Input Noise."
        },
        "python_demo": {
            "title": "L1 vs L2 Weight Sparsity Demonstration in NumPy",
            "description": "Training a linear model with L1 vs L2 penalty to observe weight distributions and exact zero-weight sparsity.",
            "code": """import numpy as np

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
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "L2 Weight Decay Shrinkage Step Calculation",
                "statement": "A neural network weight currently has value w_t = 0.80. The unregularized loss gradient with respect to this weight is dJ/dw = 0.10. The learning rate is eta = 0.05 and the L2 regularization coefficient is alpha = 0.20. Calculate: (a) The effective shrinkage factor (1 - eta * alpha), (b) The weight value after decay but before gradient application, and (c) The final updated weight w_{t+1}.",
                "given": "w_t = 0.80, dJ/dw = 0.10, eta = 0.05, alpha = 0.20",
                "formulas": "Shrinkage factor = 1 - eta * alpha; w_decayed = (1 - eta * alpha) * w_t; w_{t+1} = w_decayed - eta * (dJ/dw)",
                "steps": [
                    "Step 1: Compute shrinkage factor:\n  factor = 1.0 - (0.05 * 0.20) = 1.0 - 0.010 = 0.990.",
                    "Step 2: Apply weight decay:\n  w_decayed = 0.990 * 0.80 = 0.7920.",
                    "Step 3: Apply gradient descent step:\n  grad_step = 0.05 * 0.10 = 0.0050.\n  w_{t+1} = 0.7920 - 0.0050 = 0.7870."
                ],
                "answer": "Shrinkage factor = 0.990, Decayed weight = 0.7920, Final weight w_{t+1} = 0.7870",
                "interpretation": "L2 regularization systematically eroded the weight by 1.0% (from 0.80 to 0.792) prior to applying the empirical gradient descent update."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Optimal Weight Reduction under L2 Regularization on Quadratic Loss",
                "statement": "Consider a scalar quadratic loss function J(w) = 0.5 * H * (w - w*)^2 where Hessian curvature H = 4.0 and unregularized optimum w* = 3.0. An L2 penalty term 0.5 * alpha * w^2 is added with alpha = 1.0. (a) Derive the expression for the regularized objective J_tilde(w), (b) Find the exact regularized optimal weight w_reg by setting dJ_tilde/dw = 0, and (c) Show that w_reg = (H / (H + alpha)) * w* and compute the percentage shrinkage.",
                "given": "H = 4.0, w* = 3.0, alpha = 1.0, J_tilde(w) = 0.5 * H * (w - w*)^2 + 0.5 * alpha * w^2.",
                "formulas": "dJ_tilde/dw = H*(w - w*) + alpha*w = 0 => (H + alpha)*w = H*w* => w_reg = (H / (H + alpha)) * w*",
                "steps": [
                    "Step 1: Differentiate regularized objective:\n  dJ_tilde/dw = 4.0 * (w - 3.0) + 1.0 * w = 4.0*w - 12.0 + 1.0*w = 5.0*w - 12.0.",
                    "Step 2: Set derivative to zero to solve for optimum:\n  5.0 * w_reg - 12.0 = 0 => w_reg = 12.0 / 5.0 = 2.40.",
                    "Step 3: Verify theoretical ratio:\n  Ratio = H / (H + alpha) = 4.0 / (4.0 + 1.0) = 4.0 / 5.0 = 0.80.\n  w_reg = 0.80 * 3.0 = 2.40.",
                    "Step 4: Calculate shrinkage percentage:\n  Shrinkage = ((3.0 - 2.40) / 3.0) * 100 = (0.60 / 3.0) * 100 = 20.0%."
                ],
                "answer": "Regularized optimum w_reg = 2.40 (Exact 20% shrinkage from unregularized value 3.0)",
                "interpretation": "The L2 penalty pulls the parameter toward the origin; directions with higher curvature H resist shrinkage more strongly than flatter directions with small H."
            }
        ],
        "real_world_application": {
            "title": "Medical Imaging CT Scan Lung Nodule Detection",
            "domain": "Healthcare Diagnostics & Radiomics",
            "workflow": "Patient 3D CT pulmonary scans -> Extremely small training dataset (only 500 patient scans available) -> Massive 10x dataset augmentation (random 3D rotations, intensity scaling, elastic deformation) combined with L2 weight decay.",
            "benefit": "Prevents catastrophic overfitting to patient scanner artifacts, boosting lung cancer early detection sensitivity to 96.4%."
        },
        "viva_questions": [
            {"q": "Why does L1 regularization lead to sparse weight matrices while L2 does not?", "a": "L1 has a constant gradient magnitude alpha*sign(w) right down to the origin and its constraint surface has sharp corners on the axes where elliptical loss contours typically intersect; L2 gradient alpha*w shrinks proportionally to w, pulling weights close to zero without making them exactly zero."},
            {"q": "What is the update rule for weight decay in L2 regularization?", "a": "w_{t+1} = (1 - eta * alpha) * w_t - eta * grad J(w_t)."},
            {"q": "What is the physical meaning of the 'patience' parameter in Early Stopping?", "a": "The number of consecutive epochs to wait for validation loss improvement before terminating training."},
            {"q": "Why does Early Stopping prevent overfitting?", "a": "It stops optimization before the model begins memorizing idiosyncratic training set noise, constraining the parameter vector close to its initial state."},
            {"q": "Can dataset augmentation be applied to tabular data?", "a": "Yes, techniques like SMOTE (Synthetic Minority Over-sampling Technique) or adding small Gaussian jitter to continuous features perform tabular augmentation."},
            {"q": "What is MixUp data augmentation?", "a": "A technique (Zhang et al., 2017) that creates synthetic training examples by taking convex linear combinations of pairs of inputs and their one-hot target labels: x_new = lambda*x_i + (1-lambda)*x_j."},
            {"q": "Does L2 regularization penalize bias parameters b?", "a": "Typically no; regularizing biases is rarely beneficial because biases have far fewer parameters and do not cause high variance/overfitting."},
            {"q": "What is the relationship between L2 regularization and Bayesian inference?", "a": "L2 regularization is mathematically equivalent to Maximum A Posteriori (MAP) estimation assuming a zero-mean Gaussian prior over the weights."},
            {"q": "What is the relationship between L1 regularization and Bayesian inference?", "a": "L1 regularization is mathematically equivalent to MAP estimation assuming a zero-mean Laplace prior over the weights."},
            {"q": "Why is data augmentation applied only during training and not during evaluation?", "a": "Because evaluation requires an unbiased, true measurement on real unmodified validation/test samples (except for test-time augmentation, TTA)."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "L1 regularization is preferred over L2 regularization when the goal is to:",
                    "options": ["Prevent gradient vanishing", "Achieve parameter sparsity and feature selection", "Accelerate forward pass computation", "Eliminate all biases"],
                    "ans": "Achieve parameter sparsity and feature selection"
                },
                {
                    "q": "In L2 regularization with learning rate eta and penalty alpha, the parameter shrinkage factor per step is:",
                    "options": ["1 + eta*alpha", "1 - eta*alpha", "eta / alpha", "alpha / eta"],
                    "ans": "1 - eta*alpha"
                },
                {
                    "q": "Early stopping evaluates loss on which dataset partition to decide termination?",
                    "options": ["Training set", "Validation set", "Test set", "Synthetic set"],
                    "ans": "Validation set"
                }
            ],
            "short_questions": [
                "Explain the geometric intuition behind L1 regularization inducing sparsity using contour sketches.",
                "Describe the operational protocol of Early Stopping including the concept of patience and best checkpoint restoration."
            ],
            "application_question": "Design a dataset augmentation pipeline for an autonomous drone navigation model trained on daytime flights to perform well at dusk and dawn.",
            "homework": "Prove that L2 regularization is equivalent to Bayesian MAP estimation under an independent Gaussian prior w_i ~ N(0, sigma^2)."
        }
    },
    {
        "day": 14,
        "module": 2,
        "topic_id": "2.7",
        "title": "Parameter Sharing, Parameter Tying, and Injecting Noise at Input",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO2",
        "blooms": "Understand / Apply",
        "prerequisites": "Convolution Operation intuition, Autoencoder concepts, Regularization, Day 13 concepts",
        "learning_objectives": [
            "Formulate the principles of Parameter Sharing and Parameter Tying.",
            "Analyze how parameter sharing encodes translation invariance in CNNs.",
            "Formulate parameter tying in Autoencoder architectures (W_decoder = W_encoder^T).",
            "Evaluate the mathematical effect of injecting noise at inputs as an implicit Tikhonov regularizer."
        ],
        "key_terminology": [
            "Parameter Sharing", "Parameter Tying", "Tied Weights", "Input Noise Injection",
            "Tikhonov Regularization", "Denoising Objective"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 7 (Section 7.5 & 7.9); Aggarwal: Chapter 4.",
        "notes": {
            "parameter_sharing_and_tying": (
                "1. Parameter Sharing:\n"
                "Instead of assigning unique independent weights to every connection, parameter sharing forces multiple components to share the exact same parameter value.\n"
                "- Primary Example: Convolutional Neural Networks (CNNs). A convolutional filter (e.g. 3x3 kernel) is shared across all spatial locations of the input image.\n"
                "- Benefit: Slashes parameter count from millions to hundreds, enforces translation equivariance, and drastically lowers sample complexity.\n\n"
                "2. Parameter Tying:\n"
                "Parameter tying imposes hard linear constraints between different weight matrices in a network.\n"
                "- Classic Example: Symmetric Autoencoders where decoder weights W_d are tied to the transpose of encoder weights W_e:\n"
                "  W_d = W_e^T\n"
                "- Benefit: Halves the number of parameters in the autoencoder and acts as a strong regularizer preventing the latent representation from learning identity trivialities."
            ),
            "injecting_noise_at_input": (
                "Adding random noise epsilon ~ N(0, sigma^2 * I) to the input vector x during training:\n"
                "x_tilde = x + epsilon\n\n"
                "Mathematical Analysis (Bishop, 1995):\n"
                "For a linear or smooth model f(x) trained with squared error loss, minimizing the expected loss over noisy inputs:\n"
                "E_epsilon [ 0.5 * (f(x + epsilon) - y)^2 ]\n"
                "is mathematically equivalent to minimizing the uncorrupted loss PLUS an L2 regularization penalty proportional to the Frobenius norm of the model's Jacobian:\n"
                "L_regularized = L_clean + (1/2) * sigma^2 * || grad_x f(x) ||_2^2\n\n"
                "This penalizes rapid changes in the model's prediction with respect to input perturbations, forcing the network to learn a smooth, robust decision boundary."
            ),
            "exam_important_points": [
                "Definition and architectural benefits of Parameter Sharing in CNNs and RNNs.",
                "Formulation of Tied Weights in Autoencoders: W_decoder = W_encoder^T.",
                "Mathematical proof that injecting input Gaussian noise is equivalent to Tikhonov gradient penalty regularization.",
                "How input noise improves model robustness against adversarial attacks."
            ]
        },
        "teaching_script": {
            "00-05": "Pose the question: 'If an image has 1 million pixels, why can a 3x3 filter detect edges everywhere?' Introduce Parameter Sharing.",
            "05-20": "Explain Parameter Sharing in CNNs (spatial translation) and RNNs (temporal invariance). Show dramatic reduction in parameter count.",
            "20-35": "Formulate Parameter Tying in Autoencoders. Derive the constraint W_d = W_e^T. Explain why tying prevents trivial identity memorization.",
            "35-45": "Analyze Input Noise Injection: formulate x_tilde = x + epsilon. Derive Bishop's expansion showing equivalence to gradient regularization.",
            "45-55": "Demonstrate Python script verifying tied weights in a NumPy autoencoder.",
            "55-60": "Recap. Note: Flag syllabus topic duplication with Day 15 for faculty verification; announce Day 15 focus on Ensemble Methods."
        },
        "python_demo": {
            "title": "Tied-Weights Autoencoder and Input Noise Injection in NumPy",
            "description": "Forward and backward pass of an Autoencoder with strictly tied weights W_decoder = W_encoder.T and Gaussian input noise.",
            "code": """import numpy as np

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
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Parameter Reduction Calculation in Tied Autoencoder",
                "statement": "An autoencoder maps an input vector of dimension n_in = 784 into a compressed latent bottleneck of dimension n_h = 64, and then reconstructs the 784-dimensional output. (a) Calculate the total trainable weights and biases if encoder and decoder weights are UNTIED, (b) Calculate the total parameters if weights are TIED (W_decoder = W_encoder^T), and (c) Determine the exact percentage savings in parameter storage.",
                "given": "n_in = 784, n_h = 64, n_out = 784.",
                "formulas": "Untied Params = 2 * (n_in * n_h) + n_h + n_in; Tied Params = (n_in * n_h) + n_h + n_in",
                "steps": [
                    "Step 1: Compute Untied parameter count:\n  Encoder weights = 64 * 784 = 50,176.\n  Encoder biases = 64.\n  Decoder weights = 784 * 64 = 50,176.\n  Decoder biases = 784.\n  Total Untied Parameters = 50,176 + 64 + 50,176 + 784 = 101,200.",
                    "Step 2: Compute Tied parameter count:\n  Shared weight matrix W = 64 * 784 = 50,176.\n  Encoder biases = 64.\n  Decoder biases = 784.\n  Total Tied Parameters = 50,176 + 64 + 784 = 51,024.",
                    "Step 3: Compute parameter reduction:\n  Difference = 101,200 - 51,024 = 50,176 parameters.\n  Percentage Savings = (50,176 / 101,200) * 100 = 49.58%."
                ],
                "answer": "Untied = 101,200 params. Tied = 51,024 params (49.58% parameter reduction).",
                "interpretation": "Tying weights virtually cuts the network weight parameters in half while enforcing symmetric encoder-decoder representations."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Mathematical Equivalence of Input Noise to Tikhonov Regularization",
                "statement": "Let a linear model be f(x) = w^T * x with scalar squared error loss L = 0.5 * (f(x) - y)^2. Add isotropic Gaussian noise epsilon with mean 0 and variance sigma^2 * I to input x, giving perturbed input x_tilde = x + epsilon. (a) Expand the expected loss E_epsilon [ 0.5 * (w^T * (x + epsilon) - y)^2 ], and (b) Prove algebraically that the expected noisy loss equals the clean loss plus an exact L2 penalty 0.5 * sigma^2 * ||w||_2^2.",
                "given": "f(x) = w^T * x, x_tilde = x + epsilon, E[epsilon] = 0, E[epsilon * epsilon^T] = sigma^2 * I.",
                "formulas": "w^T * (x + epsilon) - y = (w^T*x - y) + w^T*epsilon; Expand square and take expectation.",
                "steps": [
                    "Step 1: Express error term:\n  e(epsilon) = (w^T * x - y) + w^T * epsilon.\n  Let e_clean = w^T * x - y.",
                    "Step 2: Square the error term:\n  e(epsilon)^2 = e_clean^2 + 2 * e_clean * (w^T * epsilon) + (w^T * epsilon)^2.",
                    "Step 3: Take mathematical expectation with respect to noise distribution epsilon:\n  E[ e(epsilon)^2 ] = E[ e_clean^2 ] + 2 * e_clean * w^T * E[epsilon] + E[ (w^T * epsilon)^2 ].",
                    "Step 4: Since noise is zero-mean, E[epsilon] = 0, so the middle cross-term vanishes:\n  2 * e_clean * w^T * 0 = 0.",
                    "Step 5: Evaluate the quadratic noise term:\n  (w^T * epsilon)^2 = (w^T * epsilon) * (epsilon^T * w) = w^T * (epsilon * epsilon^T) * w.\n  E[ w^T * (epsilon * epsilon^T) * w ] = w^T * E[epsilon * epsilon^T] * w = w^T * (sigma^2 * I) * w = sigma^2 * ||w||_2^2.",
                    "Step 6: Combine results with the 0.5 loss factor:\n  E[ L(x_tilde) ] = 0.5 * e_clean^2 + 0.5 * sigma^2 * ||w||_2^2 = L_clean + 0.5 * sigma^2 * ||w||_2^2."
                ],
                "answer": "E[L(x_tilde)] = L_clean + 0.5 * sigma^2 * ||w||_2^2 (Exact L2 Regularization Penalty!)",
                "interpretation": "This rigorous proof establishes that adding input Gaussian noise with variance sigma^2 is mathematically identical to applying explicit L2 weight decay with regularization coefficient alpha = sigma^2."
            }
        ],
        "real_world_application": {
            "title": "Noise-Resistant Automatic Speech Recognition in High-Speed Rail",
            "domain": "Audio Signal Processing & In-Cabin Infotainment",
            "workflow": "High-speed rail passenger speaks voice commands into cabin microphone -> Severe acoustic rumble (wind shear, steel track vibration) -> Network trained with Gaussian noise injection at input audio spectrogram frames.",
            "benefit": "Maintains 94% command recognition accuracy without acoustic degradation or word drops under 300 km/h train speeds."
        },
        "viva_questions": [
            {"q": "What is Parameter Sharing in neural networks?", "a": "Using the same set of weights across multiple computational units or time steps, rather than learning independent weights for each connection."},
            {"q": "Give two prominent examples of parameter sharing in modern architectures.", "a": "Convolutional Neural Networks (sharing filter kernels across image space) and Recurrent Neural Networks (sharing transition matrices across time steps)."},
            {"q": "What is Parameter Tying in Autoencoders?", "a": "Constraining the decoder weight matrix to be the exact transpose of the encoder weight matrix: W_decoder = W_encoder^T."},
            {"q": "How does parameter tying benefit an autoencoder?", "a": "It halves the model parameter count, prevents overfitting, and prevents the hidden representation from learning a trivial identity mapping."},
            {"q": "What happens mathematically when Gaussian noise is added to model inputs during training?", "a": "It acts as an implicit regularizer mathematically equivalent to an L2 weight penalty proportional to the noise variance."},
            {"q": "What is a Denoising Autoencoder?", "a": "An autoencoder trained to reconstruct the clean, uncorrupted input from a deliberately corrupted or noisy input vector."},
            {"q": "Can parameter sharing reduce the risk of overfitting?", "a": "Yes, significantly, because it drastically reduces the number of free parameters that the optimizer can adjust."},
            {"q": "What is translation equivariance in CNNs?", "a": "A property resulting from parameter sharing where shifting the input image results in an identically shifted feature map: f(g(x)) = g(f(x))."},
            {"q": "Why is input noise injection considered a form of data augmentation?", "a": "Because it synthetically expands the training set by creating infinitely many noisy variations around each training sample in input space."},
            {"q": "Does parameter sharing reduce the number of floating point operations (FLOPs) during inference?", "a": "No, it reduces the number of unique parameters stored in memory, but the mathematical operations (convolutions/matrix multiplies) must still be computed across all locations."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "In a symmetric autoencoder with tied weights, the decoder weight matrix W_d is equal to:",
                    "options": ["W_e", "W_e^T", "-W_e", "W_e^(-1)"],
                    "ans": "W_e^T"
                },
                {
                    "q": "Injecting isotropic Gaussian noise with variance sigma^2 at the inputs of a linear model is mathematically equivalent to:",
                    "options": ["L1 regularization", "L2 weight decay with penalty proportional to sigma^2", "Early stopping", "Dropout with p=0.5"],
                    "ans": "L2 weight decay with penalty proportional to sigma^2"
                },
                {
                    "q": "Parameter sharing in Convolutional Neural Networks primarily enforces:",
                    "options": ["Rotation invariance", "Translation equivariance", "Scale invariance", "Affine distortion"],
                    "ans": "Translation equivariance"
                }
            ],
            "short_questions": [
                "Explain the concept of Parameter Tying in Autoencoders and state two advantages it offers.",
                "Prove that adding input noise with variance sigma^2 to a linear regression model yields an L2 weight penalty."
            ],
            "application_question": "Explain how parameter sharing allows a Convolutional Neural Network to detect a pedestrian in an autonomous vehicle image regardless of where in the frame the pedestrian appears.",
            "homework": "Calculate the total parameter count of a 3-layer tied autoencoder with architecture 1024 -> 256 -> 64 -> 256 -> 1024."
        }
    },
    {
        "day": 15,
        "module": 2,
        "topic_id": "2.8",
        "title": "Ensemble Methods, Model Averaging, Bagging and Boosting in Deep Learning",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO2",
        "blooms": "Understand / Apply",
        "prerequisites": "Bias-Variance Decomposition, Generalization Theory, Day 14 concepts",
        "learning_objectives": [
            "FLAGGED SYLLABUS VERIFICATION: Address the repeated title in syllabus (2.7 & 2.8) by dedicating Day 15 specifically to Ensemble Methods.",
            "Formulate Model Averaging (Ensemble Learning) and analyze its variance reduction theorem.",
            "Compare Bagging (Bootstrap Aggregation), Boosting, and Snapshot Ensembling.",
            "Analyze why ensembling independently trained deep neural networks almost universally improves test benchmark accuracy."
        ],
        "key_terminology": [
            "Ensemble Learning", "Model Averaging", "Bootstrap Aggregation (Bagging)", "Boosting",
            "Snapshot Ensembles", "Variance Reduction", "Syllabus Duplication Verification"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 7 (Section 7.11); Aggarwal: Chapter 4.",
        "notes": {
            "faculty_syllabus_flag": (
                "FACULTY NOTICE: In the approved KTU CST414 teaching plan (Page 12 of DEEP LEARNING.pdf), "
                "topics 2.7 and 2.8 carry the identical title: 'Parameter sharing and tying, Injecting noise at input, Ensemble methods'. "
                "To ensure complete academic coverage without redundant lecturing, Day 14 covered Parameter Sharing, Tying, and Input Noise, "
                "while Day 15 is dedicated thoroughly to Ensemble Methods, Model Averaging, Bagging, and Snapshot Ensembles."
            ),
            "model_averaging_mathematics": (
                "Ensemble learning combines the predictions of k distinct models {f_1(x), f_2(x), ..., f_k(x)}:\n"
                "f_ens(x) = (1 / k) * sum_{i=1}^k f_i(x)\n\n"
                "Variance Reduction Theorem:\n"
                "Suppose each model f_i has error epsilon_i with variance E[epsilon_i^2] = v and covariance E[epsilon_i * epsilon_j] = c between different models.\n"
                "The expected squared error of the ensemble predictor is:\n"
                "E[ ((1/k) * sum_i epsilon_i)^2 ] = (1 / k^2) * sum_i sum_j E[epsilon_i * epsilon_j]\n"
                "                                  = (1 / k^2) * [ k * v + k * (k - 1) * c ]\n"
                "                                  = (1 / k) * v + ((k - 1) / k) * c\n\n"
                "Two Extreme Cases:\n"
                "1. Perfectly Correlated Models (c = v): Ensemble error = (1/k)*v + ((k-1)/k)*v = v. No gain.\n"
                "2. Perfectly Uncorrelated Errors (c = 0): Ensemble error = v / k. Error decreases linearly with ensemble size k!\n"
                "In deep learning, even if models share the same architecture, different random weight initializations and mini-batch shuffle orders lead to partially uncorrelated errors (c < v), guaranteeing that the ensemble outperforms any single constituent model."
            ),
            "deep_ensemble_techniques": (
                "1. Bagging (Bootstrap Aggregation): Train k identical architectures on k bootstrap resamples of the dataset.\n"
                "2. Multi-Architecture Ensembling: Average predictions of diverse models (e.g. ResNet + DenseNet + Vision Transformer).\n"
                "3. Snapshot Ensembles (Huang et al., 2017): Use a cyclical learning rate schedule to converge to multiple local minima during a single training run, saving model snapshots at the base of each cycle without k-fold training cost."
            ),
            "exam_important_points": [
                "Mathematical derivation of ensemble error: E[e^2] = (1/k)*v + ((k-1)/k)*c.",
                "Why ensemble error strictly improves whenever model error correlation c < v.",
                "Comparison of Bagging vs Boosting in deep learning contexts.",
                "Snapshot ensembles: achieving ensemble benefits in a single training run."
            ]
        },
        "teaching_script": {
            "00-05": "Briefly note the syllabus title duplication (2.7/2.8) and clarify that Day 15 delivers comprehensive mastery of Ensemble Methods.",
            "05-20": "Formulate Model Averaging: f_ens(x) = (1/k) sum f_i(x). Define model error variance v and pairwise covariance c.",
            "20-35": "Derive the Variance Reduction Theorem step-by-step on blackboard: E[e_ens^2] = v/k + c*(k-1)/k. Discuss the two boundary cases (c=0 vs c=v).",
            "35-45": "Explain why deep neural networks are ideal for ensembling due to non-convex loss surfaces and distinct local basins.",
            "45-55": "Demonstrate Python script ensembling 5 randomly initialized neural networks and observing test accuracy jump.",
            "55-60": "Recap and preview Day 16: Dropout and Parameter Initialization."
        },
        "python_demo": {
            "title": "Variance Reduction via Deep Model Ensembling in NumPy",
            "description": "Simulating an ensemble of 5 diverse neural network predictors to demonstrate error variance reduction on noisy regression.",
            "code": """import numpy as np

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
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Theoretical Ensemble Error Variance Calculation",
                "statement": "An ensemble is composed of k = 5 independently trained deep neural networks. Each individual network has an expected squared error variance of v = 0.20. The average pairwise covariance between the errors of different models is c = 0.05. (a) Calculate the expected squared error of the ensemble predictor, and (b) Calculate the percentage reduction in error achieved compared to using any single individual model.",
                "given": "k = 5 models, individual variance v = 0.20, covariance c = 0.05.",
                "formulas": "E_ens = (1/k) * v + ((k - 1) / k) * c; Reduction = (v - E_ens) / v * 100",
                "steps": [
                    "Step 1: Compute first term (variance contribution):\n  term1 = v / k = 0.20 / 5 = 0.040.",
                    "Step 2: Compute second term (covariance contribution):\n  term2 = ((5 - 1) / 5) * 0.05 = (4 / 5) * 0.05 = 0.80 * 0.05 = 0.040.",
                    "Step 3: Calculate total ensemble expected error:\n  E_ens = term1 + term2 = 0.040 + 0.040 = 0.080.",
                    "Step 4: Calculate percentage reduction:\n  Reduction = ((0.20 - 0.080) / 0.20) * 100 = (0.120 / 0.20) * 100 = 60.0%."
                ],
                "answer": "Ensemble Error = 0.080 (60% error reduction compared to individual model error 0.20)",
                "interpretation": "Because the model errors were only weakly correlated (c = 0.05 vs v = 0.20), model averaging slashed the total error variance by 60%."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Ensemble Probability Voting vs Majority Hard Voting",
                "statement": "Three classifiers (M1, M2, M3) output predicted probabilities for a binary positive class as P1 = 0.45, P2 = 0.48, and P3 = 0.95 for a true positive instance (y = 1). Using a classification threshold of 0.50: (a) Determine the individual binary classifications of M1, M2, and M3, (b) Determine the ensemble prediction under Majority Hard Voting, (c) Calculate the Soft Average ensemble probability P_soft and determine its prediction, and (d) Contrast the robustness of soft vs hard voting.",
                "given": "P1 = 0.45, P2 = 0.48, P3 = 0.95. True class y = 1. Threshold = 0.50.",
                "formulas": "Hard Vote: argmax(count(pred == 1)); Soft Vote: P_soft = (1/3) * (P1 + P2 + P3)",
                "steps": [
                    "Step 1: Individual binary decisions:\n  M1: P1 = 0.45 < 0.50 -> Class 0 (Incorrect)\n  M2: P2 = 0.48 < 0.50 -> Class 0 (Incorrect)\n  M3: P3 = 0.95 >= 0.50 -> Class 1 (Correct).",
                    "Step 2: Majority Hard Voting:\n  Votes for Class 0 = 2 (M1, M2).\n  Votes for Class 1 = 1 (M3).\n  Hard Voting Ensemble Prediction = Class 0 (WRONG!).",
                    "Step 3: Soft Probability Voting:\n  P_soft = (0.45 + 0.48 + 0.95) / 3 = 1.88 / 3 = 0.6267.\n  Since P_soft = 0.6267 >= 0.50, Soft Voting Ensemble Prediction = Class 1 (CORRECT!)."
                ],
                "answer": "Hard Vote = Class 0 (Fails); Soft Vote = Class 1 with 62.67% confidence (Succeeds)",
                "interpretation": "Soft probability voting takes into account confidence calibration: M3 was extremely confident (95%) while M1 and M2 were barely uncertain (45% and 48%). Soft voting correctly allowed M3's strong signal to overturn the marginal majority."
            }
        ],
        "real_world_application": {
            "title": "Kaggle Grandmaster Competition Winning Pipeline",
            "domain": "Competitive Machine Learning & Financial Risk Modeling",
            "workflow": "Tabular credit loan dataset -> Team trains 10 diverse models (LightGBM, XGBoost, CatBoost, 5-layer MLP, ResNet-Tabular) -> Out-of-fold probability predictions combined using weighted soft voting ensemble.",
            "benefit": "Ensemble consistently outscores any individual model by 0.015-0.030 AUC, making the decisive difference between 1st place gold and 50th place."
        },
        "viva_questions": [
            {"q": "What is the primary mathematical mechanism through which ensembling reduces test error?", "a": "Variance reduction: averaging uncorrelated error terms dampens overall prediction variance without increasing bias."},
            {"q": "Why is ensembling particularly effective for deep neural networks?", "a": "Because deep networks have highly non-convex loss surfaces; different initializations and mini-batch sequences cause models to converge to distinct local minima that make different errors."},
            {"q": "What is the difference between Bagging and Boosting?", "a": "Bagging trains independent models in parallel on bootstrap resamples to reduce variance; Boosting trains models sequentially, focusing each new model on earlier errors to reduce bias."},
            {"q": "What is the key disadvantage of deploying an ensemble of 10 deep neural networks to production?", "a": "Inference latency and memory costs are multiplied by 10, which may be unacceptable for real-time edge applications."},
            {"q": "What is a Snapshot Ensemble?", "a": "A technique using cyclical learning rate schedules to visit multiple local minima in a single training run, saving model weights at each cycle trough to form an ensemble for free."},
            {"q": "What is Knowledge Distillation in relation to ensembling?", "a": "Training a single compact 'student' neural network to mimic the soft probability outputs of an expensive 'teacher' ensemble, compressing ensemble accuracy into one fast model."},
            {"q": "What happens to ensemble variance if all constituent models make perfectly identical errors (c = v)?", "a": "The ensemble error remains exactly equal to v; no variance reduction occurs."},
            {"q": "Why is Soft Voting generally superior to Hard Voting?", "a": "Soft voting preserves continuous confidence probabilities, allowing highly confident models to appropriately influence uncertain models."},
            {"q": "Can models with different architectures (e.g. CNN + Transformer) be ensembled together?", "a": "Yes; architectural diversity is highly desirable because different inductive biases produce more uncorrelated errors (lower covariance c)."},
            {"q": "Why was topic 2.8 flagged in the KTU CST414 syllabus?", "a": "Because the official syllabus text repeats the exact title of topic 2.7 verbatim; separating them into Parameter Tying (Day 14) and Ensemble Methods (Day 15) provides complete, non-redundant coverage."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "If an ensemble contains k models with uncorrelated errors (c = 0) and individual variance v, the ensemble error variance is:",
                    "options": ["v", "v / k", "k * v", "v^k"],
                    "ans": "v / k"
                },
                {
                    "q": "Which technique allows obtaining an ensemble of diverse neural network weights in a single training run?",
                    "options": ["Dropout", "Snapshot Ensembles", "Batch Normalization", "Gradient Clipping"],
                    "ans": "Snapshot Ensembles"
                },
                {
                    "q": "Compressing the knowledge of an ensemble of models into a single small deployable network is called:",
                    "options": ["Pruning", "Quantization", "Knowledge Distillation", "Weight Sharing"],
                    "ans": "Knowledge Distillation"
                }
            ],
            "short_questions": [
                "Derive the variance reduction formula for an ensemble of k models and discuss the two extreme cases where c = 0 and c = v.",
                "Explain the operational principle of Snapshot Ensembles and why they eliminate the traditional computational penalty of ensembling."
            ],
            "application_question": "Explain how knowledge distillation can be used to deploy an ensemble of 5 large vision models onto an embedded smartphone camera chip.",
            "homework": "Prove that the ensemble error E_ens <= (1/k) * sum_i E[epsilon_i^2] by applying Jensen's inequality to the convex square function."
        }
    },
    {
        "day": 16,
        "module": 2,
        "topic_id": "2.9",
        "title": "Dropout and Parameter Initialization",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO2",
        "blooms": "Understand / Apply",
        "prerequisites": "Activation Functions, Backpropagation, Ensemble Theory, Day 15 concepts",
        "learning_objectives": [
            "Formulate the Dropout regularization algorithm (Srivastava, Hinton, et al., 2014).",
            "Differentiate between Training-Time Dropout (Bernoulli masking, Inverted Dropout) and Test-Time Weight Scaling.",
            "Analyze Dropout as an implicit ensemble of 2^N thinned sub-networks with shared weights.",
            "Derive and compare Xavier/Glorot Initialization and He/Kaiming Initialization based on variance preservation."
        ],
        "key_terminology": [
            "Dropout", "Bernoulli Mask", "Inverted Dropout", "Co-Adaptation", "Xavier/Glorot Initialization",
            "He/Kaiming Initialization", "Fan-In / Fan-Out", "Variance Preservation"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 7 (Section 7.12) & Chapter 8 (Section 8.4); Aggarwal: Chapter 4.",
        "notes": {
            "dropout_formulation": (
                "Dropout (Srivastava et al., JMLR 2014) is a computationally efficient regularization technique that prevents feature co-adaptation.\n\n"
                "1. Training Phase:\n"
                "For each training example and each training step, each hidden unit is randomly retained with probability p (or dropped with drop rate q = 1 - p):\n"
                "r^(l) ~ Bernoulli(p)\n"
                "h_tilde^(l) = r^(l) odot h^(l)\n\n"
                "2. Inverted Dropout (Standard Modern Implementation):\n"
                "To ensure that the expected output during testing matches training without requiring post-hoc test-time scaling, activations are scaled by 1/p during training:\n"
                "h_tilde^(l) = (r^(l) odot h^(l)) / p\n\n"
                "3. Test Phase:\n"
                "At test time, Dropout is deactivated completely: h^(l) is evaluated deterministically without scaling.\n\n"
                "Ensemble Interpretation:\n"
                "In a network with N units, Dropout samples from an ensemble of 2^N distinct 'thinned' sub-networks that all share the exact same parameter weights! At test time, using the full unmasked network computes the geometric mean of the sub-networks' predictive distributions."
            ),
            "parameter_initialization": (
                "Why not initialize weights to zero? Symmetry problem: all neurons compute identical forward activations and receive identical backpropagated gradients.\n"
                "Why not large random weights? Activations explode or saturate (vanishing gradients in sigmoid/tanh).\n\n"
                "1. Xavier / Glorot Initialization (Glorot & Bengio, 2010):\n"
                "Designed for Tanh and Sigmoid activations. Goal: maintain constant variance of activations during forward pass and gradients during backward pass:\n"
                "Var(W) = 2 / (fan_in + fan_out)\n"
                "Uniform distribution: W ~ U( - sqrt(6 / (fan_in + fan_out)), + sqrt(6 / (fan_in + fan_out)) )\n"
                "Normal distribution: W ~ N( 0, 2 / (fan_in + fan_out) )\n\n"
                "2. He / Kaiming Initialization (He et al., ICCV 2015):\n"
                "Designed specifically for ReLU activations (which zero out half the distribution):\n"
                "Var(W) = 2 / fan_in\n"
                "Uniform distribution: W ~ U( - sqrt(6 / fan_in), + sqrt(6 / fan_in) )\n"
                "Normal distribution: W ~ N( 0, sqrt(2 / fan_in) )"
            ),
            "exam_important_points": [
                "KTU Exam Q14(b): Effect of hidden unit capacity combined with dropout on bias and variance.",
                "Full algorithmic steps of Inverted Dropout vs Classical Dropout.",
                "Ensemble of 2^N sub-networks interpretation of Dropout.",
                "Formulas and derivations for Xavier/Glorot vs He/Kaiming weight initialization."
            ]
        },
        "teaching_script": {
            "00-05": "Introduce the co-adaptation problem: 'If five people work on a team and one does all the work, what happens when they get sick?' Dropout forces every neuron to learn useful features independently.",
            "05-20": "Formulate Dropout mathematically: Bernoulli mask r ~ Bernoulli(p). Compare classical test-time scaling (w * p) vs modern Inverted Dropout (scale by 1/p at train time).",
            "20-35": "Explain the 2^N sub-networks ensemble perspective. Discuss why Dropout increases training loss but slashes generalization error.",
            "35-45": "Transition to Parameter Initialization. Derive why Var(z) = fan_in * Var(w) * Var(x). Derive Xavier formula for Tanh and He formula for ReLU.",
            "45-55": "Demonstrate the Python script showing activation variance collapsing with bad initialization vs remaining stable with He initialization.",
            "55-60": "Recap Module 2. Review all 9 optimization and regularization topics. Preview Module 3: Convolutional Neural Networks."
        },
        "python_demo": {
            "title": "Inverted Dropout and He vs Xavier Initialization in NumPy",
            "description": "Forward pass with Inverted Dropout and variance tracking across 10 layers with random vs He initialization.",
            "code": """import numpy as np

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
print("Dropped Sample Row 0:\\n", h_dropped[0])

# Initialization Variance Tracking across 10 layers
dim = 256
layers = 10
x = np.random.randn(100, dim)

# 1. Bad small initialization
W_bad = np.random.randn(dim, dim) * 0.01
a = x.copy()
for _ in range(layers):
    a = np.maximum(0, np.dot(a, W_bad))
print(f"\\nBad Init Final Variance: {np.var(a):.2e} (Signal Collapsed!)")

# 2. He Initialization for ReLU
W_he = np.random.randn(dim, dim) * np.sqrt(2.0 / dim)
a_he = x.copy()
for _ in range(layers):
    a_he = np.maximum(0, np.dot(a_he, W_he))
print(f"He  Init Final Variance: {np.var(a_he):.4f} (Signal Stable across 10 layers!)")
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Inverted Dropout Expected Value Verification",
                "statement": "A hidden layer produces activation vector h = [2.0, 4.0, 6.0, 8.0]^T. During training, Inverted Dropout with retention probability p = 0.5 is applied. The generated Bernoulli mask is r = [1, 0, 1, 0]^T. (a) Calculate the masked activations under classical dropout and the required test-time scaling, (b) Calculate the masked activations under Inverted Dropout, and (c) Show that the expected sum of activations is preserved under Inverted Dropout.",
                "given": "h = [2.0, 4.0, 6.0, 8.0]^T, p = 0.5, r = [1, 0, 1, 0]^T.",
                "formulas": "Classical Train: h_tilde = r * h; Classical Test: h_test = p * h; Inverted Train: h_tilde = (r * h) / p",
                "steps": [
                    "Step 1: Compute classical dropout activations:\n  h_drop = [2.0*1, 4.0*0, 6.0*1, 8.0*0]^T = [2.0, 0.0, 6.0, 0.0]^T.\n  At test time, all weights or activations must be multiplied by p = 0.5.",
                    "Step 2: Compute Inverted Dropout activations:\n  h_inv = h_drop / p = [2.0/0.5, 0.0/0.5, 6.0/0.5, 0.0/0.5]^T = [4.0, 0.0, 12.0, 0.0]^T.",
                    "Step 3: Compare expected sums:\n  Original sum = 2.0 + 4.0 + 6.0 + 8.0 = 20.0.\n  Inverted Dropout sum for this mask = 4.0 + 0.0 + 12.0 + 0.0 = 16.0.\n  Expected sum across all masks = E[r_i] / p * sum(h_i) = (0.5 / 0.5) * 20.0 = 20.0."
                ],
                "answer": "Classical: [2.0, 0.0, 6.0, 0.0]^T (requires test scaling). Inverted: [4.0, 0.0, 12.0, 0.0]^T (no test scaling required).",
                "interpretation": "Inverted dropout scales surviving activations by 1/p (here, doubles them to 4.0 and 12.0) during training, perfectly matching the expected activation magnitude at test time."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Xavier vs He Initialization Standard Deviation Calculation",
                "statement": "A dense layer connects a preceding layer with fan_in = 512 neurons to a subsequent layer with fan_out = 128 neurons. (a) Calculate the standard deviation sigma_xavier for Xavier/Glorot normal initialization, (b) Calculate the uniform bounds [-a, +a] for Xavier uniform initialization, (c) Calculate the standard deviation sigma_he for He/Kaiming normal initialization, and (d) Explain why sigma_he is larger than sigma_xavier.",
                "given": "fan_in = 512, fan_out = 128.",
                "formulas": "sigma_xavier = sqrt(2 / (fan_in + fan_out)); a_xavier = sqrt(6 / (fan_in + fan_out)); sigma_he = sqrt(2 / fan_in)",
                "steps": [
                    "Step 1: Xavier Normal standard deviation:\n  Var_xavier = 2.0 / (512 + 128) = 2.0 / 640 = 0.003125.\n  sigma_xavier = sqrt(0.003125) = 0.05590.",
                    "Step 2: Xavier Uniform bounds:\n  a_xavier = sqrt(6.0 / (512 + 128)) = sqrt(6.0 / 640) = sqrt(0.009375) = 0.09682.\n  Bounds: [-0.09682, +0.09682].",
                    "Step 3: He Normal standard deviation:\n  Var_he = 2.0 / fan_in = 2.0 / 512 = 0.00390625.\n  sigma_he = sqrt(0.00390625) = 0.06250.",
                    "Step 4: Comparison explanation:\n  sigma_he (0.0625) > sigma_xavier (0.0559) because ReLU sets approximately half of all activations to zero (mean activation halved). To compensate for the 50% lost energy and maintain constant activation variance, He initialization injects a factor of 2 in the numerator."
                ],
                "answer": "sigma_xavier = 0.0559, a_xavier = 0.0968, sigma_he = 0.0625",
                "interpretation": "He initialization requires higher variance (0.0625 vs 0.0559) to prevent signal attenuation caused by ReLU clipping negative activations."
            }
        ],
        "real_world_application": {
            "title": "Protein Folding & Biomolecular Structure Prediction",
            "domain": "Computational Biology & Structural Bioinformatics",
            "workflow": "Deep Transformer-based architecture processes amino acid sequences -> Severe risk of memorizing known crystallized proteins -> Monte Carlo Dropout (Gal & Ghahramani, 2016) enabled at inference time to generate epistemic uncertainty confidence intervals.",
            "benefit": "Enables drug discovery scientists to flag uncertain protein loop predictions before committing to wet-lab synthesized assays."
        },
        "viva_questions": [
            {"q": "What is the primary motivation behind the Dropout technique?", "a": "To prevent complex co-adaptation of hidden units, forcing each neuron to learn robust, generalizable features that do not rely on the presence of specific other neurons."},
            {"q": "What is Inverted Dropout and why is it preferred over classical Dropout?", "a": "Inverted Dropout scales active neurons by 1/p during training, ensuring that test-time execution requires no scaling or architectural modification."},
            {"q": "How does Dropout approximate an ensemble of models?", "a": "A network with N units samples from an ensemble of 2^N distinct thinned sub-networks that all share parameters, combining their predictions at test time."},
            {"q": "What is the recommended retention probability p for hidden units vs input units?", "a": "Typically p = 0.5 for hidden layers, and p = 0.8 to 0.9 (or no dropout) for input layers."},
            {"q": "Why is Xavier initialization unsuitable for deep ReLU networks?", "a": "Because ReLU sets all negative pre-activations to zero, halving the variance at each layer; Xavier initialization fails to account for this 50% loss, causing activation variance to decay exponentially to zero in deep networks."},
            {"q": "What is the formula for He/Kaiming Normal initialization variance?", "a": "Var(W) = 2 / fan_in."},
            {"q": "What is Monte Carlo Dropout?", "a": "Leaving Dropout active at test time and performing multiple forward passes to estimate predictive uncertainty and variance (epistemic uncertainty)."},
            {"q": "Why does training loss with Dropout appear higher than without Dropout?", "a": "Because during training, 50% of the network's capacity is disabled at every step, making the training task substantially more difficult."},
            {"q": "What happens if all weights are initialized from a uniform distribution U(0, 1)?", "a": "All weights are positive, causing pre-activations to grow rapidly with depth, saturating activations and blowing up gradients."},
            {"q": "Can Dropout be applied to Convolutional layers?", "a": "Yes, but standard Dropout is often ineffective because adjacent pixels are strongly correlated; Spatial Dropout (dropping entire feature maps) is used instead."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "In Inverted Dropout with retention probability p = 0.5, the surviving activations during training are scaled by:",
                    "options": ["0.5", "1.0", "2.0", "4.0"],
                    "ans": "2.0"
                },
                {
                    "q": "The recommended weight initialization variance for a layer with fan_in inputs using ReLU activation (He initialization) is:",
                    "options": ["1 / fan_in", "2 / fan_in", "2 / (fan_in + fan_out)", "1 / (fan_in + fan_out)"],
                    "ans": "2 / fan_in"
                },
                {
                    "q": "A neural network with 20 hidden units contains how many distinct sub-networks in its Dropout ensemble?",
                    "options": ["20", "400", "2^20 (~1,048,576)", "20! (~2.43 x 10^18)"],
                    "ans": "2^20 (~1,048,576)"
                }
            ],
            "short_questions": [
                "Explain how Dropout acts as an implicit ensemble of 2^N sub-networks and describe how Inverted Dropout is implemented.",
                "Derive the variance preservation condition and compare Xavier/Glorot initialization with He/Kaiming initialization."
            ],
            "application_question": "Explain how Monte Carlo Dropout can be deployed in autonomous vehicle perception systems to measure obstacle detection uncertainty.",
            "homework": "Calculate the Xavier normal standard deviation and He normal standard deviation for a dense layer with fan_in = 1024 and fan_out = 256."
        }
    }
]

