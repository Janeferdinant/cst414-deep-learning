"""
Curriculum Data for Module 4: Recurrent Neural Networks (5 Hours)
KTU CST414 Deep Learning (Scheme 2019)
Faculty: Jane Rose | Department of Computer Science & Engineering
"""

MODULE_4_DAYS = [
    {
        "day": 23,
        "module": 4,
        "topic_id": "4.1",
        "title": "Recurrent Neural Networks: Computational Graphs and RNN Design",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO4",
        "blooms": "Understand / Apply",
        "prerequisites": "Backpropagation Algorithm, Sequence modeling intuition, Matrix multiplication, Day 4 concepts",
        "learning_objectives": [
            "Formulate the mathematical model of Recurrent Neural Networks (RNNs) and hidden state transitions.",
            "Represent recurrent computational graphs and perform unfolding across time steps.",
            "Formulate Backpropagation Through Time (BPTT) and analyze gradient flow across temporal steps.",
            "Analyze the theoretical mechanics of the vanishing and exploding gradient problem in long sequences."
        ],
        "key_terminology": [
            "Recurrent Neural Network (RNN)", "Hidden State", "Computational Graph Unfolding",
            "Backpropagation Through Time (BPTT)", "Parameter Sharing Across Time", "Temporal Receptive Field"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 10 (Sections 10.1 & 10.2); Aggarwal: Chapter 6.",
        "notes": {
            "recurrent_equations_and_unfolding": (
                "A Recurrent Neural Network processes sequential data x^(1), x^(2), ..., x^(tau) by maintaining an internal hidden state memory vector h^(t) that summarizes past history.\n\n"
                "State Update Equation:\n"
                "h^(t) = tanh( W_hh * h^(t-1) + W_xh * x^(t) + b_h )\n\n"
                "Output Equation:\n"
                "o^(t) = W_hy * h^(t) + b_y\n"
                "y_hat^(t) = softmax(o^(t)) (or linear / sigmoid for regression)\n\n"
                "Computational Graph Unfolding:\n"
                "Unfolding (unrolling) maps the recursive loop representation into a directed acyclic computational graph spanning tau discrete time steps. Key insight: The weight matrices W_hh, W_xh, and W_hy are SHARED across all time steps, enabling the network to generalize to arbitrary sequence lengths."
            ),
            "backpropagation_through_time": (
                "Total sequence loss is the sum of losses at each time step: L = sum_{t=1}^tau L^(t).\n\n"
                "By the multivariable chain rule, the gradient of loss at step tau with respect to hidden state at early step t is a product of Jacobian matrices:\n"
                "del L^(tau) / del h^(t) = (del L^(tau) / del h^(tau)) * product_{j=t+1}^tau (del h^(j) / del h^(j-1))\n\n"
                "Each temporal Jacobian is:\n"
                "del h^(j) / del h^(j-1) = W_hh^T * diag(1 - (h^(j))^2)\n\n"
                "Vanishing / Exploding Problem:\n"
                "If the largest eigenvalue of W_hh is lambda < 1 and tanh' <= 1:\n"
                "The product contracts as (lambda)^T -> 0, causing gradients from distant past steps to vanish completely. The standard RNN cannot bridge long-term dependencies (typically failing beyond 10-15 steps)."
            ),
            "exam_important_points": [
                "Full state update and output equations for simple vanilla RNN.",
                "Diagram showing recurrent loop representation vs unfolded computational graph.",
                "Derivation of Backpropagation Through Time (BPTT) chain rule product.",
                "Why parameter sharing across time is mandatory for variable-length sequence modeling."
            ]
        },
        "teaching_script": {
            "00-05": "Introduce sequential data: 'Why can't an MLP or CNN easily translate a sentence of 5 words and then a sentence of 50 words?' Introduce Recurrent Networks.",
            "05-20": "Formulate the hidden state update equation: h^(t) = tanh(W_hh*h^(t-1) + W_xh*x^(t) + b). Explain hidden state as internal memory.",
            "20-35": "Draw the unrolled computational graph across 4 time steps on blackboard. Emphasize parameter sharing of W_hh and W_xh across all steps.",
            "35-45": "Derive Backpropagation Through Time (BPTT). Write the Jacobian product equation and explain vanishing gradients over long sequences.",
            "45-55": "Demonstrate pure NumPy RNN forward and backward pass implementation.",
            "55-60": "Recap and preview Day 24: Encoder-Decoder Sequence-to-Sequence Architectures."
        },
        "python_demo": {
            "title": "Minimal Vanilla RNN from Scratch in NumPy",
            "description": "Forward pass and unrolling of a simple recurrent neural network over 4 time steps.",
            "code": """import numpy as np

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
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Vanilla RNN Hidden State and Output Calculation",
                "statement": "A simple RNN has parameters W_xh = [0.5, 0.2], W_hh = 0.8, b_h = 0.1, W_hy = 1.2, and b_y = -0.3. The activation function for the hidden state is Tanh, and the output neuron is linear. The initial hidden state is h_0 = 0.0. The input sequence has two time steps: x_1 = [1.0, 2.0]^T and x_2 = [0.5, -1.0]^T. Calculate: (a) Hidden state h_1 and output y_1, and (b) Hidden state h_2 and output y_2.",
                "given": "h_0 = 0.0, W_xh = [0.5, 0.2], W_hh = 0.8, b_h = 0.1, W_hy = 1.2, b_y = -0.3.",
                "formulas": "h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h); y_t = W_hy * h_t + b_y",
                "steps": [
                    "Step 1: Time step t = 1:\n  W_xh * x_1 = (0.5 * 1.0) + (0.2 * 2.0) = 0.5 + 0.4 = 0.9.\n  z_1 = (0.8 * 0.0) + 0.9 + 0.1 = 1.0.\n  h_1 = tanh(1.0) = 0.76159.\n  y_1 = (1.2 * 0.76159) + (-0.3) = 0.9139 - 0.3 = 0.6139.",
                    "Step 2: Time step t = 2:\n  W_xh * x_2 = (0.5 * 0.5) + (0.2 * -1.0) = 0.25 - 0.20 = 0.05.\n  W_hh * h_1 = 0.8 * 0.76159 = 0.60927.\n  z_2 = 0.60927 + 0.05 + 0.1 = 0.75927.\n  h_2 = tanh(0.75927) = 0.64082.\n  y_2 = (1.2 * 0.64082) + (-0.3) = 0.7690 - 0.3 = 0.4690."
                ],
                "answer": "Step 1: h_1 = 0.7616, y_1 = 0.6139. Step 2: h_2 = 0.6408, y_2 = 0.4690.",
                "interpretation": "The hidden state h_1 retained memory of x_1 and propagated it forward into step 2, combining with x_2 to produce output y_2."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "BPTT Gradient Decay Factor over 10 Time Steps",
                "statement": "In a recurrent neural network, the scalar recurrent weight is W_hh = 0.85. At each time step, the hidden activation derivative is approximately tanh'(z) = 1 - h^2 approx 0.80. (a) Calculate the single-step temporal Jacobian factor gamma = W_hh * tanh'(z), (b) Calculate the total gradient attenuation factor across 10 unrolled time steps: gamma^10, and (c) Determine what percentage of the gradient signal survives backpropagating 10 steps into the past.",
                "given": "W_hh = 0.85, tanh'(z) = 0.80, horizon T = 10 steps.",
                "formulas": "Single step gamma = W_hh * tanh'(z); 10-step attenuation = gamma^10; Survival % = gamma^10 * 100",
                "steps": [
                    "Step 1: Calculate single-step contraction factor:\n  gamma = 0.85 * 0.80 = 0.680.",
                    "Step 2: Calculate attenuation across 10 steps:\n  Attenuation = (0.680)^10 = 0.021139.",
                    "Step 3: Calculate percentage of surviving gradient:\n  Percentage = 0.021139 * 100 = 2.11%."
                ],
                "answer": "Single-step factor gamma = 0.680. 10-step attenuation = 0.0211 (Only 2.11% gradient survives).",
                "interpretation": "Over 97.8% of the error gradient disappears after just 10 time steps, demonstrating why vanilla RNNs cannot learn long-term temporal dependencies."
            }
        ],
        "real_world_application": {
            "title": "Smart Power Grid Wind Energy Output Forecasting",
            "domain": "Renewable Energy & Power Distribution",
            "workflow": "Hourly anemometer wind speed and barometric sensor readings over past 24 hours -> Unrolled RNN processes sequential wind gusts -> Predicts mega-watt turbine power generation for next 4 hours.",
            "benefit": "Enables power grid operators to balance battery storage and spin up backup generation without blackouts."
        },
        "viva_questions": [
            {"q": "What is the primary architectural difference between an MLP and an RNN?", "a": "An RNN contains cyclic recurrent connections that feed the previous hidden state back into the current step, allowing it to process variable-length sequential data with memory."},
            {"q": "What does 'unfolding' a recurrent computational graph mean?", "a": "Drawing the computational graph across discrete time steps as a sequence of repeated identical layers sharing the same parameters."},
            {"q": "What is Backpropagation Through Time (BPTT)?", "a": "The application of the backpropagation algorithm to an unfolded recurrent network, computing gradients by summing contributions across all time steps."},
            {"q": "Why is parameter sharing essential in RNNs?", "a": "It allows the model to process sequences of arbitrary length and recognize patterns regardless of where they appear in time, using a fixed number of parameters."},
            {"q": "Why do vanilla RNNs suffer from vanishing gradients?", "a": "Because backpropagation over long horizons involves repeatedly multiplying by the recurrent weight matrix W_hh and tanh derivatives (< 1), causing gradients to decay exponentially."},
            {"q": "What happens if the largest singular value of W_hh is greater than 1?", "a": "Gradients can explode exponentially over long time steps, leading to numerical overflow (NaNs) and unstable weights."},
            {"q": "What simple technique mitigates exploding gradients in RNNs?", "a": "Gradient clipping (rescaling the gradient vector when its norm exceeds a threshold)."},
            {"q": "What is Truncated BPTT?", "a": "An approximation where the forward and backward passes are evaluated over a fixed window of k time steps rather than the entire sequence, capping computational cost."},
            {"q": "Can an RNN have different input dimensions at each time step?", "a": "Typically input dimensions x_dim are fixed, but the total number of time steps (sequence length T) can vary from sample to sample."},
            {"q": "Why is Tanh the standard activation function for hidden states in vanilla RNNs?", "a": "Because its range is (-1, +1), which is zero-centered and prevents state activations from growing unbounded in magnitude across time steps."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "In a standard RNN, the weights W_hh and W_xh across different time steps are:",
                    "options": ["Independently initialized and updated", "Shared and identical across all time steps", "Inverted at each step", "Constrained to be zero"],
                    "ans": "Shared and identical across all time steps"
                },
                {
                    "q": "The vanishing gradient problem in RNNs primarily impedes the network from learning:",
                    "options": ["Immediate next-step correlations", "Long-term temporal dependencies", "Linear transformations", "High-frequency noise"],
                    "ans": "Long-term temporal dependencies"
                },
                {
                    "q": "The primary purpose of unrolling an RNN computational graph is to:",
                    "options": ["Convert it into an acyclic graph suitable for backpropagation", "Double the parameter count", "Remove all non-linearities", "Speed up GPU compilation"],
                    "ans": "Convert it into an acyclic graph suitable for backpropagation"
                }
            ],
            "short_questions": [
                "Formulate the hidden state update equation for a vanilla RNN and explain the physical role of the hidden state vector.",
                "Derive the BPTT gradient propagation chain rule and explain why vanilla RNNs cannot bridge long sequence gaps."
            ],
            "application_question": "Explain how an RNN is applied to monitor heart arrhythmias from continuous real-time ECG waveforms.",
            "homework": "Given W_hh = 0.9 and average activation derivative 0.7, compute the gradient attenuation factor over 15 unrolled time steps."
        }
    },
    {
        "day": 24,
        "module": 4,
        "topic_id": "4.2",
        "title": "Encoder-Decoder Sequence-to-Sequence Architectures",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO4",
        "blooms": "Understand / Apply",
        "prerequisites": "Vanilla RNN, Hidden states, Day 23 concepts",
        "learning_objectives": [
            "Formulate the Seq2Seq framework (Sutskever et al. 2014, Cho et al. 2014).",
            "Examine the architectural separation of Encoder RNN and Decoder RNN.",
            "Analyze the Information Bottleneck problem of the fixed-length context vector c.",
            "Contrast Teacher Forcing during training with Autoregressive Greedy / Beam Search inference."
        ],
        "key_terminology": [
            "Sequence-to-Sequence (Seq2Seq)", "Encoder RNN", "Decoder RNN", "Context Vector",
            "Information Bottleneck", "Teacher Forcing", "Autoregressive Generation", "Beam Search"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 10 (Section 10.4); Sutskever et al.: Sequence to Sequence Learning with Neural Networks (NeurIPS 2014).",
        "notes": {
            "seq2seq_architecture": (
                "Standard RNNs require input length T_x and output length T_y to be strictly equal or synchronized. Many tasks require mapping a sequence of length T_x to an output sequence of different length T_y (e.g. Machine Translation: 'How are you?' [3 words] -> 'Wie geht es Ihnen?' [4 words]).\n\n"
                "1. Encoder RNN:\n"
                "Processes input sequence x^(1), ..., x^(T_x) sequentially:\n"
                "h_enc^(t) = f_enc(h_enc^(t-1), x^(t))\n"
                "The final hidden state represents the summary Context Vector c = h_enc^(T_x).\n\n"
                "2. Decoder RNN:\n"
                "Initializes hidden state with the context vector: h_dec^(0) = c.\n"
                "Autoregressively generates target tokens y^(1), ..., y^(T_y):\n"
                "h_dec^(t) = f_dec(h_dec^(t-1), y^(t-1), c)\n"
                "p(y^(t) | y^(1), ..., y^(t-1), x) = softmax( W_out * h_dec^(t) )\n"
                "Generation terminates when the special <EOS> (End-of-Sequence) token is emitted."
            ),
            "information_bottleneck_and_teacher_forcing": (
                "1. Information Bottleneck Problem:\n"
                "Compressing an entire source sentence of arbitrary length (e.g. 50 words) into a single fixed-dimensional vector c (e.g. 512 floats) creates an extreme bottleneck. Model performance deteriorates sharply for sentences longer than 20-30 words (motivating Bahdanau's Attention mechanism).\n\n"
                "2. Teacher Forcing:\n"
                "During training, the decoder receives the ground-truth target token y_true^(t-1) as input at step t, rather than its own potentially erroneous previous prediction y_pred^(t-1).\n"
                "- Advantage: Stabilizes and drastically accelerates training.\n"
                "- Drawback (Exposure Bias): At test time, ground truth is absent; early mistakes can compound catastrophically."
            ),
            "exam_important_points": [
                "Full architectural diagram of Encoder-Decoder Seq2Seq framework.",
                "How context vector c transfers semantic information between encoder and decoder.",
                "Definition, merits, and exposure bias drawback of Teacher Forcing.",
                "Why fixed-length context vectors fail on long sequences."
            ]
        },
        "teaching_script": {
            "00-05": "Introduce English-to-Malayalam translation: 'Why cannot input and output word counts match 1-to-1?' Introduce Seq2Seq.",
            "05-20": "Draw the Encoder-Decoder pipeline on blackboard. Trace input tokens through Encoder to final context vector c = h_enc^(T).",
            "20-35": "Trace Decoder generation: show how <SOS> triggers decoding and how <EOS> halts generation. Explain autoregressive prediction.",
            "35-45": "Explain the Information Bottleneck: show how cramming a paragraph into 512 numbers causes amnesia. Introduce Teacher Forcing vs Inference.",
            "45-55": "Demonstrate Python script running an Encoder-Decoder character-level reversal network.",
            "55-60": "Recap and preview Day 25: Deep Recurrent Networks and Recursive Networks."
        },
        "python_demo": {
            "title": "Minimal Seq2Seq Sequence Reversal in NumPy",
            "description": "Encoder-Decoder architecture mapping a 3-step sequence to its reversed target sequence.",
            "code": """import numpy as np

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

print("Encoder Final Context Vector c:\\n", np.round(c.ravel(), 3))

# 2. Decode with context vector
h_dec = c.copy()
y_prev = np.array([[0.0], [0.0]]) # <SOS> token
generated_tokens = []
for step in range(3):
    h_dec = np.tanh(np.dot(W_dec, y_prev) + np.dot(U_dec, h_dec))
    y_pred = np.dot(V_out, h_dec)
    generated_tokens.append(y_pred.ravel())
    y_prev = y_pred

print("Decoder Generated 3 Steps:\\n", np.round(generated_tokens, 3))
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Context Vector Compression Ratio in Seq2Seq",
                "statement": "An English sentence of 40 words is processed by an Encoder RNN. Each word is represented by a 300-dimensional word embedding vector. The Encoder condenses the sentence into a context vector c of dimension 512. Assuming 32-bit floating point numbers (4 bytes/float): (a) Calculate the raw input memory size in bytes, (b) Calculate the context vector memory size in bytes, and (c) Determine the compression ratio achieved by the bottleneck.",
                "given": "40 words, 300 dims/word, context c = 512 floats, 4 bytes/float.",
                "formulas": "Input_mem = Words * Dims * 4; Context_mem = Dims_c * 4; Ratio = Input_mem / Context_mem",
                "steps": [
                    "Step 1: Compute input sentence memory:\n  Total input floats = 40 * 300 = 12,000 floats.\n  Input memory = 12,000 * 4 bytes = 48,000 bytes = 46.88 KB.",
                    "Step 2: Compute context vector memory:\n  Context memory = 512 * 4 bytes = 2,048 bytes = 2.0 KB.",
                    "Step 3: Calculate compression ratio:\n  Ratio = 48,000 / 2,048 = 23.4375x (over 23-fold compression!)."
                ],
                "answer": "Input = 48,000 bytes. Context = 2,048 bytes. Compression ratio = 23.44x.",
                "interpretation": "The encoder must compress 48 KB of nuanced semantic and syntactic linguistic relationships into just 2 KB, highlighting why fixed context vectors suffer severe information loss on long documents."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Beam Search Path Probability vs Greedy Decoding",
                "statement": "At step 1 of decoding, a vocabulary has 3 words: {A, B, C} with probabilities P(A) = 0.6, P(B) = 0.3, P(C) = 0.1. At step 2, conditional probabilities are:\nIf step 1 was A: P(A|A) = 0.2, P(B|A) = 0.3, P(C|A) = 0.5.\nIf step 1 was B: P(A|B) = 0.8, P(B|B) = 0.1, P(C|B) = 0.1.\n(a) Determine the 2-step sequence selected by Greedy Decoding, (b) Calculate the total sequence probability of all 9 possible combinations, and (c) Determine the optimal sequence found by Beam Search with beam width B = 2.",
                "given": "P(y1), P(y2|y1) as specified.",
                "formulas": "P(y1, y2) = P(y1) * P(y2|y1); Greedy: argmax at each step; Beam Search: maintain top B paths at each step",
                "steps": [
                    "Step 1: Greedy Decoding:\n  Step 1: max(0.6, 0.3, 0.1) -> selects 'A' (P=0.6).\n  Step 2 (given A): max(0.2, 0.3, 0.5) -> selects 'C' (P=0.5).\n  Greedy sequence: 'A C' with total probability = 0.6 * 0.5 = 0.30.",
                    "Step 2: Evaluate alternative candidate paths starting from B:\n  P('B A') = P(B) * P(A|B) = 0.3 * 0.8 = 0.24.\n  P('B B') = 0.3 * 0.1 = 0.03.\n  P('B C') = 0.3 * 0.1 = 0.03.",
                    "Step 3: Evaluate paths starting from A:\n  P('A A') = 0.6 * 0.2 = 0.12.\n  P('A B') = 0.6 * 0.3 = 0.18.\n  P('A C') = 0.6 * 0.5 = 0.30.",
                    "Step 4: Beam Search with Beam Width B = 2:\n  Step 1: Top 2 candidates retained: Path 'A' (0.6) and Path 'B' (0.3).\n  Step 2 expansions:\n    'A A' = 0.12, 'A B' = 0.18, 'A C' = 0.30, 'B A' = 0.24, 'B B' = 0.03, 'B C' = 0.03.\n  Top 2 pruned beam paths: 'A C' (0.30) and 'B A' (0.24).\n  Optimal path = 'A C' (0.30)."
                ],
                "answer": "Greedy = 'A C' (P = 0.30). Beam Search (B=2) retains both 'A C' (0.30) and 'B A' (0.24), confirming 'A C' as optimal.",
                "interpretation": "Greedy search makes locally optimal decisions that can miss globally superior sequences, whereas Beam Search explores multiple plausible trajectories concurrently."
            }
        ],
        "real_world_application": {
            "title": "Automated Clinical Discharge Summary Translation",
            "domain": "Medical Informatics & Cross-Lingual Healthcare",
            "workflow": "Doctor dictates English clinical notes -> Encoder RNN ingests medical diagnosis jargon -> Produces latent context embedding -> Decoder RNN generates patient-friendly regional language discharge instructions.",
            "benefit": "Eliminates language barriers in rural hospital discharge, cutting medication dosing misunderstandings by 35%."
        },
        "viva_questions": [
            {"q": "What is the core capability that Seq2Seq provides over standard vanilla RNNs?", "a": "Mapping an input sequence of length T_x to an output sequence of a completely different length T_y."},
            {"q": "What is the Context Vector c in Seq2Seq?", "a": "The final hidden state vector of the Encoder RNN that summarizes the entire source sentence and serves as the seed memory for the Decoder."},
            {"q": "What is the Information Bottleneck problem in Seq2Seq?", "a": "Forcing an entire source sentence of arbitrary length into a single fixed-size vector causes severe performance degradation on long sequences."},
            {"q": "What is Teacher Forcing?", "a": "A training procedure where the ground-truth target token from the previous step is fed into the decoder, rather than the model's own predicted token."},
            {"q": "What is Exposure Bias in sequence generation?", "a": "The discrepancy between training (where ground truth tokens are provided) and inference (where the model must consume its own predictions), allowing early generation errors to compound."},
            {"q": "How does the Decoder know when to stop generating tokens?", "a": "When it outputs the special <EOS> (End-of-Sequence) token."},
            {"q": "What is Beam Search?", "a": "A heuristic search algorithm that maintains a set of the B most probable candidate sequences at each decoding step, avoiding suboptimal greedy choices."},
            {"q": "Can the Encoder and Decoder use different architectures?", "a": "Yes; for example, the Encoder can be a bidirectional LSTM while the Decoder is a unidirectional GRU."},
            {"q": "What historical paper introduced the modern Seq2Seq framework?", "a": "Ilya Sutskever, Oriol Vinyals, and Quoc Le (NeurIPS 2014): 'Sequence to Sequence Learning with Neural Networks'."},
            {"q": "What major breakthrough resolved the Seq2Seq Information Bottleneck in 2014-2015?", "a": "The Attention Mechanism (Bahdanau et al., 2014), which allows the decoder to look back at all encoder hidden states dynamically."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "The role of the context vector in a Seq2Seq model is to:",
                    "options": ["Provide L2 regularization", "Transfer the encoded summary of the source sequence to the decoder", "Store Adam optimizer moments", "Determine learning rate decay"],
                    "ans": "Transfer the encoded summary of the source sequence to the decoder"
                },
                {
                    "q": "Teacher Forcing refers to feeding which input into the decoder during training?",
                    "options": ["A random noise vector", "The ground truth previous target token", "The model's own previous prediction", "The input image"],
                    "ans": "The ground truth previous target token"
                },
                {
                    "q": "Seq2Seq performance on long sentences deteriorates primarily because of:",
                    "options": ["The Information Bottleneck of compressing sentences into a fixed-length vector", "Over-regularization", "ReLU saturation", "Too many hidden layers"],
                    "ans": "The Information Bottleneck of compressing sentences into a fixed-length vector"
                }
            ],
            "short_questions": [
                "Draw the architectural block diagram of an Encoder-Decoder Seq2Seq model and explain how variable-length mapping is achieved.",
                "Define Teacher Forcing and analyze the trade-off between fast training convergence and inference exposure bias."
            ],
            "application_question": "Explain how a Seq2Seq model can be formulated to perform automated source code generation from natural language docstrings.",
            "homework": "Explain why reversing the source sentence (feeding x_T first down to x_1) improved early Seq2Seq machine translation performance in Sutskever et al. (2014)."
        }
    },
    {
        "day": 25,
        "module": 4,
        "topic_id": "4.3",
        "title": "Deep Recurrent Networks and Recursive Neural Networks",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO4",
        "blooms": "Understand / Apply",
        "prerequisites": "RNN Unfolding, Seq2Seq, Tree data structures, Day 24 concepts",
        "learning_objectives": [
            "Formulate Deep (Stacked / Multi-Layer) Recurrent Neural Networks.",
            "Formulate Bidirectional Recurrent Neural Networks (BiRNNs) and analyze forward/backward context integration.",
            "Formulate Recursive Neural Networks operating over tree-structured computational graphs.",
            "Contrast Recurrent architectures (linear temporal sequences) with Recursive architectures (hierarchical parse trees)."
        ],
        "key_terminology": [
            "Deep Recurrent Network", "Stacked RNN", "Bidirectional RNN (BiRNN)", "Forward State",
            "Backward State", "Recursive Neural Network", "Tree-Structured Graph", "Constituency Parse Tree"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 10 (Sections 10.3 & 10.6); Aggarwal: Chapter 6.",
        "notes": {
            "deep_stacked_rnns": (
                "A standard RNN is deep in time (across unrolled sequence steps), but shallow in spatial representation (only one hidden layer per time step).\n\n"
                "A Deep (Stacked) RNN stacks multiple recurrent layers vertically:\n"
                "- Layer 1 receives input sequence x^(t): h_1^(t) = f( W_1 * x^(t) + U_1 * h_1^(t-1) )\n"
                "- Layer 2 receives hidden state sequence of Layer 1 as its input: h_2^(t) = f( W_2 * h_1^(t) + U_2 * h_2^(t-1) )\n"
                "- Layer L computes: h_L^(t) = f( W_L * h_{L-1}^(t) + U_L * h_L^(t-1) )\n\n"
                "Hierarchical Temporal Abstraction: Lower recurrent layers capture fast, fine-grained temporal patterns (phonemes, characters), while higher recurrent layers capture slow, abstract semantic structures (syntax, topics)."
            ),
            "bidirectional_rnns": (
                "Schuster & Paliwal (1997) introduced Bidirectional RNNs (BiRNNs) to resolve the limitation that standard causal RNNs only know past context.\n\n"
                "Equations:\n"
                "1. Forward RNN: h_fwd^(t) = f( W_xf * x^(t) + W_ff * h_fwd^(t-1) )  (processes sequence left-to-right)\n"
                "2. Backward RNN: h_bwd^(t) = f( W_xb * x^(t) + W_bb * h_bwd^(t+1) )  (processes sequence right-to-left)\n"
                "3. Concatenated Representation: h^(t) = [ h_fwd^(t) ; h_bwd^(t) ]\n"
                "Output y^(t) is conditioned simultaneously on past history AND future context! Critical for speech recognition, POS tagging, and NER."
            ),
            "recursive_neural_networks": (
                "While an RNN operates over a 1D chain graph of sequence length T, a Recursive Neural Network (Pollack 1990; Socher et al. 2011) operates over an arbitrary Directed Tree-Structured Graph.\n\n"
                "Tree Propagation:\n"
                "For a node with children c_1 and c_2:\n"
                "p = f( W * [c_1 ; c_2] + b )\n"
                "The parent representation p is computed recursively up to the root node.\n\n"
                "Advantage: Drastically reduces the maximum path length between any two words from O(T) in a linear RNN to O(log T) in a balanced tree, massively mitigating vanishing gradients and naturally matching the hierarchical syntax of natural language sentences and molecular graphs."
            ),
            "exam_important_points": [
                "Architectural equations and diagram for Bidirectional RNNs.",
                "Comparison: Recurrent (1D sequential chain) vs Recursive (hierarchical tree graph).",
                "Why BiRNNs cannot be used for online real-time causal generation.",
                "Logarithmic path length O(log T) in Recursive networks."
            ]
        },
        "teaching_script": {
            "00-05": "Pose the linguistic puzzle: 'In the sentence 'The bank of the river', how do we know what 'bank' means before reading 'river'?' Introduce Bidirectional RNNs.",
            "05-20": "Draw BiRNN computational graph: forward states flowing left-to-right, backward states flowing right-to-left. Show concatenation h^(t) = [h_fwd ; h_bwd].",
            "20-35": "Introduce Deep Stacked RNNs. Explain vertical depth vs temporal unrolling. Show how Layer 1 extracts phonemes and Layer 3 extracts syntax.",
            "35-45": "Introduce Recursive Neural Networks. Draw a constituency parse tree for 'Deep learning is fun'. Show parent node calculation: p = tanh(W*[c1; c2]).",
            "45-55": "Demonstrate Python script computing BiRNN forward and backward passes.",
            "55-60": "Recap and preview Day 26: Long Short-Term Memory (LSTM)."
        },
        "python_demo": {
            "title": "Bidirectional RNN Forward Pass in pure NumPy",
            "description": "Implementation of Forward and Backward RNN passes over a sequence, concatenating representations.",
            "code": """import numpy as np

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
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Trainable Parameter Count in a 3-Layer Deep Stacked RNN",
                "statement": "A deep stacked RNN has an input dimension of x_dim = 128. It contains 3 stacked recurrent hidden layers, each having hidden dimension h_dim = 256. The top hidden layer connects to an output layer of dimension y_dim = 10. Assuming each layer has full input-to-hidden and hidden-to-hidden matrices plus biases: (a) Calculate the parameters in Layer 1, (b) Calculate the parameters in Layer 2 and Layer 3, (c) Calculate output layer parameters, and (d) Determine the total trainable parameters.",
                "given": "x_dim = 128, h1 = 256, h2 = 256, h3 = 256, y_dim = 10.",
                "formulas": "Layer 1 = (x_dim * h) + (h * h) + h; Middle Layer = (h_prev * h) + (h * h) + h; Out = (h * y) + y",
                "steps": [
                    "Step 1: Layer 1 (Input 128 -> Hidden 256):\n  W_x1 = 256 * 128 = 32,768.\n  W_h1 = 256 * 256 = 65,536.\n  b_1 = 256.\n  Total Layer 1 = 32,768 + 65,536 + 256 = 98,560 parameters.",
                    "Step 2: Layer 2 (Hidden 256 -> Hidden 256):\n  W_x2 = 256 * 256 = 65,536.\n  W_h2 = 256 * 256 = 65,536.\n  b_2 = 256.\n  Total Layer 2 = 65,536 + 65,536 + 256 = 131,328 parameters.",
                    "Step 3: Layer 3 (Hidden 256 -> Hidden 256):\n  Total Layer 3 = 131,328 parameters.",
                    "Step 4: Output Layer (Hidden 256 -> Output 10):\n  W_out = 10 * 256 = 2,560.\n  b_out = 10.\n  Total Output = 2,560 + 10 = 2,570 parameters.",
                    "Step 5: Total Trainable Parameters = 98,560 + 131,328 + 131,328 + 2,570 = 363,786 parameters."
                ],
                "answer": "Layer 1 = 98,560; Layer 2 = 131,328; Layer 3 = 131,328; Output = 2,570. Total = 363,786 parameters.",
                "interpretation": "Stacking layers creates multi-tier hierarchical representations with moderate parameter growth compared to widening."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Path Length and Gradient Propagation: Recurrent Chain vs Recursive Tree",
                "statement": "Consider a sequence of length T = 64 tokens. (a) In a standard 1D recurrent chain, calculate the maximum number of sequential transitions required for an error signal to propagate from token 64 to token 1, (b) In a balanced binary recursive parse tree, calculate the maximum path length from the root to any leaf token, and (c) Assuming a per-step contraction factor gamma = 0.8, compare the gradient attenuation in the Recurrent chain vs the Recursive tree.",
                "given": "T = 64 tokens, gamma = 0.8.",
                "formulas": "Recurrent Path = T - 1; Recursive Balanced Tree Path = 2 * log2(T); Attenuation = gamma^(Path)",
                "steps": [
                    "Step 1: Recurrent 1D Chain Path Length:\n  Path = 64 - 1 = 63 transitions.",
                    "Step 2: Recursive Balanced Binary Tree Path Length:\n  Tree height = log2(64) = 6 levels.\n  Maximum path between two leaves through root = 2 * 6 = 12 transitions.",
                    "Step 3: Compute Recurrent gradient attenuation:\n  Attenuation_RNN = (0.8)^63 = 1.07 x 10^(-6) (Completely vanished!).",
                    "Step 4: Compute Recursive gradient attenuation:\n  Attenuation_Recursive = (0.8)^12 = 0.0687 (6.87% signal preserved!).",
                    "Step 5: Comparison ratio:\n  Ratio = 0.0687 / (1.07 x 10^(-6)) = ~64,000x stronger gradient signal!"
                ],
                "answer": "Recurrent Path = 63 steps (Atten = 1.07e-6). Recursive Tree Path = 12 steps (Atten = 0.0687, ~64,000x stronger signal).",
                "interpretation": "The tree topology of Recursive Neural Networks compresses sequential path lengths from O(T) to O(log T), dramatically preserving gradient signals across distant words."
            }
        ],
        "real_world_application": {
            "title": "Named Entity Recognition (NER) in Electronic Health Records (EHR)",
            "domain": "Clinical NLP & Biomedical Information Extraction",
            "workflow": "Clinical doctor discharge note -> Ingested by a 2-layer Bidirectional LSTM -> Forward state understands preceding symptoms, Backward state understands succeeding medication dosages -> Accurately tags entities (Disease, Drug, Dosage).",
            "benefit": "Extracts structured clinical adverse drug event records from unstructured physician notes with 93.8% F1-score."
        },
        "viva_questions": [
            {"q": "What is the primary motivation for Bidirectional RNNs?", "a": "Standard RNNs only possess information about past context; BiRNNs process the sequence in both forward and backward directions, giving each token full access to both past and future context."},
            {"q": "Why can Bidirectional RNNs NOT be used for real-time online speech generation or autoregressive language modeling?", "a": "Because the backward pass requires access to future tokens, which have not yet occurred or been spoken in real time."},
            {"q": "What is a Deep (Stacked) RNN?", "a": "An architecture where the output hidden states of one recurrent layer serve as the input sequence to a second recurrent layer above it."},
            {"q": "What is the primary difference between a Recurrent and a Recursive Neural Network?", "a": "A Recurrent network operates over a 1D linear chain of time steps, whereas a Recursive network operates over an arbitrary hierarchical tree graph (e.g. syntax parse tree)."},
            {"q": "What is the theoretical advantage of a Recursive Neural Network for natural language?", "a": "It matches the innate hierarchical constituency grammar of human language and reduces the gradient propagation distance between words from O(T) to O(log T)."},
            {"q": "How is the parent node vector computed in a binary Recursive Neural Network?", "a": "By concatenating the two child vectors [c1; c2] and passing them through a linear transformation and non-linear activation: p = tanh(W * [c1; c2] + b)."},
            {"q": "Why are Recursive Neural Networks rarely used compared to Transformers today?", "a": "They require pre-computed syntactic parse trees, cannot easily batch variable-structure trees on GPUs, and have high data-dependent branching overhead."},
            {"q": "How many total hidden vectors are produced at each time step in a BiRNN?", "a": "Two: a forward hidden vector h_fwd^(t) and a backward hidden vector h_bwd^(t), typically concatenated to form a vector of dimension 2*h_dim."},
            {"q": "What is hierarchical temporal abstraction in deep stacked RNNs?", "a": "The phenomenon where lower layers capture rapid local temporal transitions (characters) while upper layers capture slow, abstract global semantics (paragraphs/themes)."},
            {"q": "Can skip connections be added between layers in a deep stacked RNN?", "a": "Yes; residual skip connections between recurrent layers (like in GNMT) are essential for training stacked RNNs deeper than 4 to 8 layers."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "In a Bidirectional RNN, the representation of token t is formed by:",
                    "options": ["Multiplying input by bias", "Concatenating the forward state h_fwd^(t) and backward state h_bwd^(t)", "Averaging weights over time", "Taking the derivative of the loss"],
                    "ans": "Concatenating the forward state h_fwd^(t) and backward state h_bwd^(t)"
                },
                {
                    "q": "The maximum path length between any two tokens in a balanced binary Recursive Neural Network of length T is:",
                    "options": ["O(T)", "O(T^2)", "O(log T)", "O(1)"],
                    "ans": "O(log T)"
                },
                {
                    "q": "Bidirectional RNNs are UNSUITABLE for which of the following tasks?",
                    "options": ["Part-of-Speech Tagging", "Named Entity Recognition", "Real-time online streaming speech synthesis", "Document classification"],
                    "ans": "Real-time online streaming speech synthesis"
                }
            ],
            "short_questions": [
                "Formulate the equations for a Bidirectional Recurrent Neural Network and explain why future context is essential for sequence tagging.",
                "Contrast the graph topologies and computational characteristics of Recurrent vs Recursive Neural Networks."
            ],
            "application_question": "Explain how a deep stacked Bidirectional LSTM is applied to predict secondary protein structure from amino acid sequences.",
            "homework": "Calculate the total weight and bias parameters of a 2-layer Bidirectional RNN with input dim 100, hidden dim 128 in each direction, and output classification into 5 classes."
        }
    },
    {
        "day": 26,
        "module": 4,
        "topic_id": "4.4",
        "title": "Modern RNNs: Long Short-Term Memory (LSTM)",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO4",
        "blooms": "Understand / Apply",
        "prerequisites": "Vanilla RNN, Backpropagation Through Time, Vanishing Gradients, Day 23 concepts",
        "learning_objectives": [
            "Formulate the Long Short-Term Memory (LSTM) architecture (Hochreiter & Schmidhuber 1997; Gers et al. 2000).",
            "Analyze the role of the Constant Error Carousel (CEC) and the Cell State c^(t) as an error conveyor belt.",
            "Derive the equations for the three gating mechanisms: Forget Gate, Input Gate, and Output Gate.",
            "Explain mathematically how LSTM gates resolve the vanishing gradient problem over hundreds of time steps."
        ],
        "key_terminology": [
            "Long Short-Term Memory (LSTM)", "Cell State", "Constant Error Carousel (CEC)",
            "Forget Gate", "Input Gate", "Output Gate", "Candidate Memory State", "Additive Gradient Highway"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 10 (Section 10.7); Aggarwal: Chapter 6; KTU CST414 Syllabus Model Question Paper Part B (Q18a).",
        "notes": {
            "lstm_architecture_and_equations": (
                "Hochreiter & Schmidhuber (1997) designed the LSTM to maintain a linear memory cell state c^(t) where error gradients can flow backward across time without exponential decay.\n\n"
                "Given input x^(t) and previous hidden state h^(t-1):\n\n"
                "1. Forget Gate (Gers et al., 2000):\n"
                "   f^(t) = sigma( W_f * [h^(t-1) ; x^(t)] + b_f )\n"
                "   Decides what fraction of the old cell state to discard (0 = forget completely, 1 = retain entirely).\n\n"
                "2. Input Gate & Candidate State:\n"
                "   i^(t) = sigma( W_i * [h^(t-1) ; x^(t)] + b_i )  (determines which values to update)\n"
                "   c_tilde^(t) = tanh( W_c * [h^(t-1) ; x^(t)] + b_c )  (creates new candidate values)\n\n"
                "3. Cell State Update (The Additive Highway):\n"
                "   c^(t) = f^(t) odot c^(t-1) + i^(t) odot c_tilde^(t)\n"
                "   Key Insight: This is a LINEAR, ADDITIVE combination! If the forget gate f^(t) = 1, the old memory passes through with a derivative of exactly 1.0, enabling error signals to flow back hundreds of steps without decaying.\n\n"
                "4. Output Gate & Hidden State:\n"
                "   o^(t) = sigma( W_o * [h^(t-1) ; x^(t)] + b_o )\n"
                "   h^(t) = o^(t) odot tanh(c^(t))\n"
                "   Filters the cell state through the output gate to produce the external hidden state."
            ),
            "mathematical_resolution_of_vanishing_gradients": (
                "In a vanilla RNN, del h^(t) / del h^(t-1) = W_hh^T * diag(1 - h^2). Repeated multiplication of matrices with eigenvalues < 1 causes exponential decay.\n\n"
                "In an LSTM, the gradient flowing along the cell state is:\n"
                "del c^(t) / del c^(t-1) = diag(f^(t))\n"
                "If the network learns that forget gate f^(t) approx 1.0, the Jacobian is the IDENTITY matrix:\n"
                "del c^(tau) / del c^(t) = product_{j=t+1}^tau f^(j) approx 1.0\n"
                "Gradients bypass the destructive multiplication by weight matrices entirely, flowing uninterrupted across hundreds of steps."
            ),
            "exam_important_points": [
                "Full mathematical formulas for all four LSTM gates/states: f, i, c_tilde, c, o, h.",
                "Architectural block diagram showing cell state conveyor belt and pointwise operations.",
                "Mathematical explanation of why del c^(t) / del c^(t-1) = f^(t) resolves vanishing gradients.",
                "KTU Exam Q18(a): Applications of LSTM in Natural Language Processing (Machine Translation, Text Generation, Sentiment Analysis)."
            ]
        },
        "teaching_script": {
            "00-05": "Introduce the forgetfulness of vanilla RNNs. State: 'LSTM was invented because vanilla RNNs forget after 10 words.'",
            "05-20": "Draw the iconic LSTM cell diagram on blackboard: show the top horizontal conveyor belt (Cell State c^(t)).",
            "20-35": "Derive the three gates step-by-step: Forget gate (f), Input gate (i + c_tilde), and Output gate (o). Explain Sigmoid [0,1] vs Tanh [-1,1].",
            "35-45": "Prove mathematically why the gradient does not vanish: show del c^(t)/del c^(t-1) = f^(t). Show that f=1 creates an identity gradient pipeline.",
            "45-55": "Demonstrate the interactive Python LSTM cell stepping through sequence memory.",
            "55-60": "Recap and preview Day 27: Gated Recurrent Unit (GRU)."
        },
        "python_demo": {
            "title": "Full Long Short-Term Memory (LSTM) Cell in NumPy",
            "description": "Pure NumPy implementation of an LSTM cell computing forward gates, cell updates, and hidden states.",
            "code": """import numpy as np

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
print("Forget Gate f:\\n", np.round(gates1[0], 3))
print("Input Gate  i:\\n", np.round(gates1[1], 3))
print("Updated Cell State c:\\n", np.round(c1, 3))
print("Updated Hidden State h:\\n", np.round(h1, 3))
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Manual Step-by-Step LSTM Gate and State Calculation",
                "statement": "An LSTM has scalar dimensions (x_dim = 1, h_dim = 1). At step t = 1, previous hidden state is h_0 = 0.5 and previous cell state is c_0 = 2.0. The input is x_1 = 1.0. Parameters are:\nForget gate: W_f = 0.5, U_f = 0.2, b_f = 0.1 (Sigmoid)\nInput gate: W_i = 0.4, U_i = 0.3, b_i = -0.2 (Sigmoid)\nCandidate state: W_c = 0.8, U_c = 0.1, b_c = 0.0 (Tanh)\nOutput gate: W_o = 0.6, U_o = -0.1, b_o = 0.2 (Sigmoid)\nCalculate: (a) Forget gate f_1, (b) Input gate i_1 and candidate c_tilde_1, (c) Updated cell state c_1, and (d) Updated hidden state h_1.",
                "given": "h_0 = 0.5, c_0 = 2.0, x_1 = 1.0. Parameters as stated.",
                "formulas": "f = sigma(W_f*x + U_f*h + b_f); i = sigma(W_i*x + U_i*h + b_i); c_tilde = tanh(W_c*x + U_c*h + b_c); c = f*c_prev + i*c_tilde; o = sigma(W_o*x + U_o*h + b_o); h = o * tanh(c)",
                "steps": [
                    "Step 1: Compute Forget Gate f_1:\n  z_f = (0.5 * 1.0) + (0.2 * 0.5) + 0.1 = 0.5 + 0.1 + 0.1 = 0.70.\n  f_1 = 1 / (1 + e^(-0.70)) = 1 / (1 + 0.49659) = 1 / 1.49659 = 0.6682.",
                    "Step 2: Compute Input Gate i_1 and Candidate c_tilde_1:\n  z_i = (0.4 * 1.0) + (0.3 * 0.5) + (-0.2) = 0.4 + 0.15 - 0.2 = 0.35.\n  i_1 = 1 / (1 + e^(-0.35)) = 1 / (1 + 0.70469) = 1 / 1.70469 = 0.5866.\n  z_c = (0.8 * 1.0) + (0.1 * 0.5) + 0.0 = 0.8 + 0.05 = 0.85.\n  c_tilde_1 = tanh(0.85) = 0.6911.",
                    "Step 3: Update Cell State c_1:\n  c_1 = (f_1 * c_0) + (i_1 * c_tilde_1) = (0.6682 * 2.0) + (0.5866 * 0.6911) = 1.3364 + 0.4054 = 1.7418.",
                    "Step 4: Compute Output Gate o_1 and Hidden State h_1:\n  z_o = (0.6 * 1.0) + (-0.1 * 0.5) + 0.2 = 0.6 - 0.05 + 0.2 = 0.75.\n  o_1 = 1 / (1 + e^(-0.75)) = 1 / (1 + 0.47237) = 1 / 1.47237 = 0.6792.\n  tanh(c_1) = tanh(1.7418) = 0.9404.\n  h_1 = o_1 * tanh(c_1) = 0.6792 * 0.9404 = 0.6387."
                ],
                "answer": "f_1 = 0.6682, i_1 = 0.5866, c_tilde_1 = 0.6911, Cell State c_1 = 1.7418, Hidden State h_1 = 0.6387",
                "interpretation": "The forget gate retained 66.8% of past memory c_0, added new information scaled by i_1, and exposed 63.8% of the transformed memory to the hidden state."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Trainable Parameter Count in Standard Multi-Class LSTM Layer",
                "statement": "An LSTM layer receives sequential word embeddings of dimension x_dim = 300 and contains h_dim = 512 hidden units. It connects to an output classification vocabulary of size V = 10,000 words. (a) Derive the parameter formula for an LSTM layer containing 4 internal gates/states, (b) Calculate total weights and biases in the recurrent LSTM layer, (c) Calculate parameters in the final projection layer, and (d) Determine the grand total trainable parameters.",
                "given": "x_dim = 300, h_dim = 512, V = 10,000.",
                "formulas": "LSTM Params = 4 * [ h_dim * (x_dim + h_dim) + h_dim ]; Projection = (h_dim * V) + V",
                "steps": [
                    "Step 1: Understand LSTM 4-fold parameter structure:\n  The LSTM contains 4 linear transformations: Forget gate, Input gate, Candidate cell, Output gate.\n  Each gate has input weights (h x x_dim), recurrent weights (h x h), and bias (h).",
                    "Step 2: Parameters per single gate:\n  W_x = 512 * 300 = 153,600.\n  W_h = 512 * 512 = 262,144.\n  b = 512.\n  Per-gate parameters = 153,600 + 262,144 + 512 = 416,256.",
                    "Step 3: Total LSTM Layer parameters (4 gates):\n  LSTM_total = 4 * 416,256 = 1,665,024 parameters.",
                    "Step 4: Output Projection Layer:\n  W_proj = 10,000 * 512 = 5,120,000.\n  b_proj = 10,000.\n  Total Projection = 5,120,000 + 10,000 = 5,130,000 parameters.",
                    "Step 5: Grand Total Parameters = 1,665,024 + 5,130,000 = 6,795,024 parameters."
                ],
                "answer": "LSTM Layer = 1,665,024 params. Output Layer = 5,130,000 params. Grand Total = 6,795,024 parameters.",
                "interpretation": "An LSTM has exactly 4 times the recurrent parameters of a vanilla RNN of the same hidden width due to its four internal gating projections."
            }
        ],
        "real_world_application": {
            "title": "Long-Term ICU Patient Sepsis Onset Prediction",
            "domain": "Critical Care Medicine & Early Warning Clinical Systems",
            "workflow": "Patient vital signs recorded every minute over a 72-hour ICU stay (4,320 time steps) -> LSTM cell state conveyor belt maintains long-term baseline organ history -> Predicts septic shock 6 hours before clinical manifestation.",
            "benefit": "Enables preemptive broad-spectrum antibiotic intervention, reducing ICU sepsis mortality by 28%."
        },
        "viva_questions": [
            {"q": "Who invented the LSTM architecture and in what year?", "a": "Sepp Hochreiter and Jürgen Schmidhuber in 1997."},
            {"q": "What are the three gates present in a standard modern LSTM?", "a": "The Forget Gate, the Input Gate, and the Output Gate."},
            {"q": "What is the mathematical role of the Forget Gate?", "a": "To compute a probability vector f in [0, 1] using a Sigmoid function that determines which components of the previous cell state c^(t-1) are preserved or erased."},
            {"q": "Why is the cell state update c^(t) = f*c^(t-1) + i*c_tilde called an additive conveyor belt?", "a": "Because information is added linearly without passing through a matrix multiplication, allowing error gradients to propagate unchanged with derivative ~ 1.0."},
            {"q": "Why should the forget gate bias b_f be initialized to 1.0 or 2.0 at the start of training?", "a": "To ensure that f^(t) approx 1.0 initially, preventing the network from forgetting historical context during early training epochs."},
            {"q": "What is the difference between the cell state c^(t) and the hidden state h^(t)?", "a": "The cell state is the internal long-term linear memory; the hidden state is the filtered external representation visible to subsequent layers and outputs."},
            {"q": "How many times more parameters does an LSTM have compared to a vanilla RNN of the same width?", "a": "Exactly 4 times more parameters, corresponding to the 4 internal transformations (f, i, c_tilde, o)."},
            {"q": "What are Peephole Connections in an LSTM?", "a": "Connections (Gers & Schmidhuber, 2000) that allow the gate layers to look directly at the cell state c^(t-1) when computing gate activations."},
            {"q": "Can an LSTM bridge dependencies over 1,000 time steps?", "a": "Yes; by holding the forget gate open (f=1) and input gate closed (i=0), an LSTM can preserve a memory bit across thousands of time steps without degradation."},
            {"q": "What activation function is used for gate activations vs candidate state?", "a": "Sigmoid (range 0 to 1) is used for gates to act as soft switches; Tanh (range -1 to +1) is used for candidate state and cell squashing to regulate signal magnitude."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "The gate that determines how much of the previous cell state c^(t-1) to discard is the:",
                    "options": ["Input gate", "Forget gate", "Output gate", "Reset gate"],
                    "ans": "Forget gate"
                },
                {
                    "q": "An LSTM has how many times the parameter count of a vanilla RNN with identical hidden dimension?",
                    "options": ["2x", "3x", "4x", "8x"],
                    "ans": "4x"
                },
                {
                    "q": "The primary mathematical reason LSTM eliminates vanishing gradients is:",
                    "options": ["Batch normalization inside the cell", "The additive linear gradient highway of the cell state", "Using ReLU instead of Tanh", "Elimination of backpropagation"],
                    "ans": "The additive linear gradient highway of the cell state"
                }
            ],
            "short_questions": [
                "Write down the complete mathematical equations governing all gates and state updates in an LSTM cell.",
                "Explain the role of the Constant Error Carousel (CEC) and describe how it resolves the vanishing gradient problem."
            ],
            "application_question": "Explain how an LSTM model is deployed in banking to detect sequential credit card transaction fraud across months.",
            "homework": "Prove that if forget gate f^(t) = 1 and input gate i^(t) = 0 for all t, the gradient del L / del c^(1) = del L / del c^(T) exactly."
        }
    },
    {
        "day": 27,
        "module": 4,
        "topic_id": "4.5",
        "title": "Gated Recurrent Unit (GRU) and Modern Sequence Architectures",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO4",
        "blooms": "Understand / Apply",
        "prerequisites": "LSTM Architecture, Gating mechanisms, Day 26 concepts",
        "learning_objectives": [
            "Formulate the Gated Recurrent Unit (GRU) architecture (Cho et al. 2014; Chung et al. 2014).",
            "Derive the equations for the two GRU gates: Reset Gate and Update Gate.",
            "Analyze the elimination of the separate cell state in favor of a merged hidden state h^(t).",
            "Compare LSTM vs GRU across parameter count, convergence speed, memory overhead, and representational capacity."
        ],
        "key_terminology": [
            "Gated Recurrent Unit (GRU)", "Reset Gate", "Update Gate", "Merged Hidden State",
            "Convex Linear Interpolation", "Parameter Efficiency", "LSTM vs GRU Trade-off"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 10 (Section 10.7); Cho et al.: On the Properties of Neural Machine Translation (2014); KTU CST414 Syllabus Model Question Paper Part B (Q18b).",
        "notes": {
            "gru_architecture_and_equations": (
                "Kyunghyun Cho et al. (2014) introduced the Gated Recurrent Unit (GRU) as a streamlined, computationally efficient alternative to LSTM.\n\n"
                "Key Structural Differences from LSTM:\n"
                "1. No separate Cell State: GRU merges cell state c^(t) and hidden state h^(t) into a single state vector h^(t).\n"
                "2. Two Gates instead of Three: Eliminates the separate output gate. Uses only a Reset Gate r^(t) and an Update Gate z^(t).\n"
                "3. Coupled Retention and Injection: In LSTM, forget gate f and input gate i are independent. In GRU, the update gate z controls both simultaneously via convex linear interpolation (1 - z) and z.\n\n"
                "Mathematical Formulation:\n"
                "1. Update Gate:\n"
                "   z^(t) = sigma( W_z * [h^(t-1) ; x^(t)] + b_z )\n"
                "   Decides how much of past state to carry forward vs new state.\n\n"
                "2. Reset Gate:\n"
                "   r^(t) = sigma( W_r * [h^(t-1) ; x^(t)] + b_r )\n"
                "   Decides how much of past state to ignore when computing candidate.\n\n"
                "3. Candidate Hidden State:\n"
                "   h_tilde^(t) = tanh( W_h * [ r^(t) odot h^(t-1) ; x^(t) ] + b_h )\n\n"
                "4. Final Hidden State (Linear Interpolation):\n"
                "   h^(t) = (1 - z^(t)) odot h^(t-1) + z^(t) odot h_tilde^(t)"
            ),
            "lstm_vs_gru_comprehensive_comparison": (
                "Comparison Dimension | LSTM | GRU\n"
                "Gates | 3 (Forget, Input, Output) | 2 (Reset, Update)\n"
                "Internal States | 2 (Cell c + Hidden h) | 1 (Merged Hidden h)\n"
                "Parameters per Unit | 4 * (d_x + d_h + 1) * d_h | 3 * (d_x + d_h + 1) * d_h (25% fewer parameters!)\n"
                "Computation Speed | Slower (more matrix multiplies) | Faster (~25-30% faster per epoch)\n"
                "Memory Usage | Higher (stores c and h) | Lower\n"
                "Long Dependencies | Superior on massive data / long sequences | Comparable on moderate sequences, less prone to overfitting on small datasets."
            ),
            "exam_important_points": [
                "KTU Exam Q18(b): Complete architectural discussion and equations of GRU.",
                "Equations for Update Gate z, Reset Gate r, Candidate h_tilde, and Final h.",
                "How GRU couples forgetting and updating via (1 - z)*h_prev + z*h_tilde.",
                "Detailed comparison table: LSTM vs GRU."
            ]
        },
        "teaching_script": {
            "00-05": "Introduce GRU motivation: 'LSTM has 4 matrix multiplies per step. Can we get the same long-term memory with 25% fewer parameters?'",
            "05-20": "Draw GRU block diagram on blackboard. Show how cell state c and hidden state h are merged into one vector.",
            "20-35": "Derive the two gates: Reset gate r (how to forget past when computing candidate) and Update gate z (convex interpolation).",
            "35-45": "Compare LSTM vs GRU side-by-side in a table. Explain why GRU trains faster and is less prone to overfitting on small corpora.",
            "45-55": "Demonstrate Python script comparing GRU and LSTM parameter count and training speed.",
            "55-60": "Recap Module 4. Review all sequence models. Announce Module 5: Application Areas (Vision, NLP, Speech, Autoencoders, DBNs)."
        },
        "python_demo": {
            "title": "Gated Recurrent Unit (GRU) Cell Implementation in NumPy",
            "description": "Pure NumPy implementation of a GRU cell showing reset gate, update gate, and linear interpolation.",
            "code": """import numpy as np

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
print("Reset Gate  r:\\n", np.round(r, 3))
print("Update Gate z:\\n", np.round(z, 3))
print("Candidate   h_tilde:\\n", np.round(h_cand, 3))
print("Final State h:\\n", np.round(h_next, 3))
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Manual Step-by-Step GRU State Calculation",
                "statement": "A GRU cell has scalar dimensions (x_dim = 1, h_dim = 1). At step t = 1, the previous hidden state is h_0 = 0.6. The input is x_1 = 0.5. Parameters are:\nReset gate: W_xr = 0.4, W_hr = 0.5, b_r = -0.1 (Sigmoid)\nUpdate gate: W_xz = 0.3, W_hz = 0.2, b_z = 0.1 (Sigmoid)\nCandidate state: W_xh = 0.6, W_hh = 0.8, b_h = 0.0 (Tanh)\nCalculate: (a) Reset gate r_1, (b) Update gate z_1, (c) Candidate state h_tilde_1, and (d) Final updated hidden state h_1.",
                "given": "h_0 = 0.6, x_1 = 0.5. Parameters as specified.",
                "formulas": "r = sigma(W_xr*x + W_hr*h + b_r); z = sigma(W_xz*x + W_hz*h + b_z); h_tilde = tanh(W_xh*x + W_hh*(r*h) + b_h); h = (1 - z)*h_prev + z*h_tilde",
                "steps": [
                    "Step 1: Compute Reset Gate r_1:\n  z_r = (0.4 * 0.5) + (0.5 * 0.6) + (-0.1) = 0.20 + 0.30 - 0.10 = 0.40.\n  r_1 = 1 / (1 + e^(-0.40)) = 1 / (1 + 0.67032) = 1 / 1.67032 = 0.5987.",
                    "Step 2: Compute Update Gate z_1:\n  z_z = (0.3 * 0.5) + (0.2 * 0.6) + 0.10 = 0.15 + 0.12 + 0.10 = 0.37.\n  z_1 = 1 / (1 + e^(-0.37)) = 1 / (1 + 0.69073) = 1 / 1.69073 = 0.5915.",
                    "Step 3: Compute Candidate State h_tilde_1:\n  Reset-masked previous state = r_1 * h_0 = 0.5987 * 0.6 = 0.3592.\n  z_h = (0.6 * 0.5) + (0.8 * 0.3592) + 0.0 = 0.30 + 0.2874 = 0.5874.\n  h_tilde_1 = tanh(0.5874) = 0.5281.",
                    "Step 4: Compute Final Hidden State h_1:\n  h_1 = (1.0 - z_1) * h_0 + (z_1 * h_tilde_1)\n      = (1.0 - 0.5915) * 0.6 + (0.5915 * 0.5281)\n      = (0.4085 * 0.6) + 0.3124 = 0.2451 + 0.3124 = 0.5575."
                ],
                "answer": "Reset gate r_1 = 0.5987, Update gate z_1 = 0.5915, Candidate h_tilde_1 = 0.5281, Final Hidden State h_1 = 0.5575",
                "interpretation": "The update gate z_1 = 0.5915 blended 40.85% of old state h_0 with 59.15% of the newly computed candidate h_tilde_1."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Exact Parameter Reduction Ratio: GRU vs LSTM",
                "statement": "An embedded edge speech recognition system has an input dimension of x_dim = 80 (filterbank features) and a recurrent hidden dimension of h_dim = 256. (a) Calculate the total weights and biases in an LSTM layer, (b) Calculate the total weights and biases in a GRU layer, (c) Determine the exact number of parameters saved and the percentage reduction, and (d) Calculate the GPU inference memory saved for a batch size of 64 over 100 time steps.",
                "given": "x_dim = 80, h_dim = 256, batch = 64, T = 100.",
                "formulas": "LSTM = 4 * [ h*(x + h) + h ]; GRU = 3 * [ h*(x + h) + h ]; Ratio = 3 / 4 = 0.75 (25% reduction)",
                "steps": [
                    "Step 1: Compute single-gate parameters:\n  Single gate = 256 * (80 + 256) + 256 = 256 * 336 + 256 = 86,016 + 256 = 86,272 parameters.",
                    "Step 2: LSTM Parameters (4 gates):\n  LSTM_total = 4 * 86,272 = 345,088 parameters.",
                    "Step 3: GRU Parameters (3 gates):\n  GRU_total = 3 * 86,272 = 258,816 parameters.",
                    "Step 4: Compute parameter reduction:\n  Difference = 345,088 - 258,816 = 86,272 parameters.\n  Percentage Reduction = (86,272 / 345,088) * 100 = 25.00% exact reduction!",
                    "Step 5: Compute state cache memory savings at inference:\n  LSTM caches both cell state c and hidden state h: 2 * (64 * 256) floats = 32,768 floats = 128 KB.\n  GRU caches only hidden state h: 1 * (64 * 256) floats = 16,384 floats = 64 KB (50% state memory savings!)."
                ],
                "answer": "LSTM = 345,088 params. GRU = 258,816 params (Exact 25% parameter reduction; 50% state cache savings).",
                "interpretation": "GRU slashes parameter count by exactly 25% and halves state memory requirements, making it ideal for edge microcontrollers and low-latency audio devices."
            }
        ],
        "real_world_application": {
            "title": "On-Device Voice Assistant Wake-Word Detection",
            "domain": "Smart Home IoT & Ultra-Low-Power Edge Audio",
            "workflow": "Always-on microphone samples ambient audio at 16 kHz -> 2-layer GRU consumes streaming 80-band Mel filterbanks -> Slashes memory and battery by 25% compared to LSTM -> Triggers main CPU only when wake-word ('Hey Google') is detected.",
            "benefit": "Maintains 98.7% wake-word detection accuracy while drawing less than 15 milli-Watts on battery power."
        },
        "viva_questions": [
            {"q": "Who invented the Gated Recurrent Unit (GRU) and in what year?", "a": "Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio in 2014."},
            {"q": "What are the two gates in a GRU?", "a": "The Reset Gate and the Update Gate."},
            {"q": "What is the function of the Reset Gate in a GRU?", "a": "It determines how much of the previous hidden state h^(t-1) should be ignored when computing the new candidate hidden state h_tilde^(t)."},
            {"q": "What is the function of the Update Gate in a GRU?", "a": "It controls the linear interpolation balance between the previous hidden state h^(t-1) and the candidate hidden state h_tilde^(t)."},
            {"q": "How does GRU combine the functions of the LSTM Forget and Input gates?", "a": "It couples them into a single update gate z: the old state is scaled by (1 - z) while the candidate state is scaled by z."},
            {"q": "Why does GRU have 25% fewer parameters than LSTM?", "a": "Because GRU contains 3 gate/state transformations instead of LSTM's 4, cutting parameter count from 4*h*(x+h) to 3*h*(x+h)."},
            {"q": "Does GRU have a separate cell state like LSTM?", "a": "No, GRU eliminates the separate cell state c^(t) and maintains only a single hidden state h^(t)."},
            {"q": "When would an engineer choose LSTM over GRU?", "a": "When modeling extremely long sequences with complex counting or long-term retention requirements on large datasets where LSTM's extra capacity shines."},
            {"q": "When would an engineer choose GRU over LSTM?", "a": "When computational speed, lower memory, faster training convergence, and smaller model footprints are needed (e.g. edge microcontrollers, smaller datasets)."},
            {"q": "Can GRUs be stacked into deep multi-layer and bidirectional configurations?", "a": "Yes; stacked and bidirectional GRUs are constructed identically to their LSTM counterparts."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "The Gated Recurrent Unit (GRU) contains which two gates?",
                    "options": ["Forget gate and Input gate", "Reset gate and Update gate", "Input gate and Output gate", "Reset gate and Forget gate"],
                    "ans": "Reset gate and Update gate"
                },
                {
                    "q": "Compared to an LSTM of the same hidden width, a GRU has approximately what percentage fewer parameters?",
                    "options": ["10%", "25%", "50%", "75%"],
                    "ans": "25%"
                },
                {
                    "q": "In a GRU, the candidate hidden state is blended with the previous hidden state via:",
                    "options": ["Element-wise division", "Convex linear interpolation: (1 - z)*h_prev + z*h_tilde", "Matrix inversion", "Logarithmic addition"],
                    "ans": "Convex linear interpolation: (1 - z)*h_prev + z*h_tilde"
                }
            ],
            "short_questions": [
                "Write down the governing mathematical equations for the GRU cell and explain the function of the Reset and Update gates.",
                "Compare LSTM and GRU across architectural complexity, parameter count, and training efficiency."
            ],
            "application_question": "Explain why GRU is often preferred over LSTM for real-time edge wake-word detection on low-power IoT microcontrollers.",
            "homework": "Prove that if update gate z^(t) = 0 for all t, the hidden state h^(t) = h^(0) indefinitely, preserving past memory completely."
        }
    }
]

