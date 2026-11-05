import React, { useState } from 'react';
import { 
  FileText, Award, CheckCircle2, ChevronDown, 
  ChevronUp, Download, BookOpen, Layers, Filter, Printer
} from 'lucide-react';

interface QuestionItem {
  qNo: string;
  part: 'A' | 'B';
  module: number;
  marks: number;
  co: string;
  blooms: string;
  question: string;
  markingScheme: string;
  solution: string;
}

export const QuestionPaperViewer: React.FC = () => {
  const [activePaper, setActivePaper] = useState<'ese' | 'cie1' | 'cie2'>('ese');
  const [filterModule, setFilterModule] = useState<number | 'all'>('all');
  const [filterPart, setFilterPart] = useState<'all' | 'A' | 'B'>('all');
  const [expandedQuestions, setExpandedQuestions] = useState<Record<string, boolean>>({
    'Q1': true,
    'Q11': true
  });

  const toggleExpand = (qNo: string) => {
    setExpandedQuestions(prev => ({ ...prev, [qNo]: !prev[qNo] }));
  };

  const expandAll = () => {
    const all: Record<string, boolean> = {};
    eseQuestions.forEach(q => { all[q.qNo] = true; });
    setExpandedQuestions(all);
  };

  const collapseAll = () => {
    setExpandedQuestions({});
  };

  // 100-Mark KTU CST414 Scheme 2019 Model Exam Questions & Verbatim Solutions
  const eseQuestions: QuestionItem[] = [
    // PART A (10 x 3 = 30 Marks)
    {
      qNo: 'Q1',
      part: 'A',
      module: 1,
      marks: 3,
      co: 'CO1',
      blooms: 'K2 (Understand)',
      question: 'State the representation power of a single-layer perceptron. Why is it incapable of solving the XOR function?',
      markingScheme: 'Representation theorem: 1.5 marks | XOR geometric non-linear separability proof: 1.5 marks',
      solution: `A single-layer perceptron (SLP) forms a linear decision boundary (a hyperplane defined by w^T x + b = 0) in the input space R^n. Consequently, it can only separate linearly separable pattern classes.

In the case of the XOR function:
- Inputs (0,0) and (1,1) produce output 0.
- Inputs (0,1) and (1,0) produce output 1.
Plotting these four points in the 2D Cartesian plane reveals that no single straight line can partition the points such that (0,1) and (1,0) lie on one side while (0,0) and (1,1) lie on the other. Minsky and Papert (1969) formally proved that linear separation fails for XOR, requiring at least one hidden layer with non-linear activation to project the inputs into a higher-dimensional space where they become linearly separable.`
    },
    {
      qNo: 'Q2',
      part: 'A',
      module: 1,
      marks: 3,
      co: 'CO1',
      blooms: 'K2 (Understand)',
      question: 'Explain why the vanishing gradient problem occurs when training deep networks with Sigmoid activation functions.',
      markingScheme: 'Sigmoid derivative range [0, 0.25]: 1.5 marks | Chain rule product attenuation: 1.5 marks',
      solution: `The binary sigmoid function is defined as sigma(z) = 1 / (1 + e^(-z)). Its first derivative is sigma'(z) = sigma(z) * (1 - sigma(z)).
The maximum value of sigma'(z) occurs at z = 0, where sigma(0) = 0.5, yielding sigma'(0) = 0.5 * (1 - 0.5) = 0.25. For any non-zero z, 0 <= sigma'(z) < 0.25.

During backpropagation in an L-layer network, the error gradient with respect to the initial layer weights involves a product of L activation derivatives:
dL / dw_1 = delta_L * Prod_{k=2}^L (W_k^T * sigma'(z_{k-1})) * x.
Because each term sigma'(z) <= 0.25, multiplying these fractions repeatedly causes the gradient magnitude to decay exponentially with depth:
0.25^10 approx 9.5 * 10^(-7). Consequently, early layer weights receive virtually zero gradient updates, halting feature learning.`
    },
    {
      qNo: 'Q3',
      part: 'A',
      module: 2,
      marks: 3,
      co: 'CO2',
      blooms: 'K2 (Understand)',
      question: 'Differentiate between Batch Gradient Descent, Stochastic Gradient Descent (SGD), and Mini-batch Gradient Descent.',
      markingScheme: '1 mark per variant (sample size, gradient stability, convergence speed)',
      solution: `1. Batch Gradient Descent (BGD): Evaluates the true gradient over the entire training dataset of N samples before each parameter update: theta = theta - lr * (1/N) * sum_{i=1}^N grad L_i. It yields exact gradient steps but has high memory overhead O(N) and slow updates for large datasets.

2. Stochastic Gradient Descent (SGD): Computes the gradient and updates parameters using a single randomly chosen training sample at each step: theta = theta - lr * grad L_i. Updates are instantaneous and escape shallow local minima due to high gradient variance, but the optimization path oscillates heavily without smooth convergence.

3. Mini-batch Gradient Descent: Evaluates gradients over a small batch B (e.g., 32, 64, 128) of samples: theta = theta - lr * (1/B) * sum_{i in B} grad L_i. It optimally harnesses GPU vectorized hardware acceleration (SIMD/Tensor Cores) while retaining steady convergence with reduced stochastic variance.`
    },
    {
      qNo: 'Q4',
      part: 'A',
      module: 2,
      marks: 3,
      co: 'CO2',
      blooms: 'K2 (Understand)',
      question: 'How does L2 regularization (weight decay) prevent overfitting in deep neural networks?',
      markingScheme: 'Objective function formulation: 1 mark | Weight contraction derivation: 2 marks',
      solution: `L2 regularization augments the empirical loss J(theta) with a penalty proportional to the squared Euclidean norm of the weight vector:
J_reg(w) = J(w) + (lambda / 2) * ||w||_2^2 = J(w) + (lambda / 2) * sum_i w_i^2.

Computing the gradient:
grad_w J_reg(w) = grad_w J(w) + lambda * w.
The gradient descent parameter update rule becomes:
w_{t+1} = w_t - eta * (grad J(w_t) + lambda * w_t) = (1 - eta * lambda) * w_t - eta * grad J(w_t).

The pre-factor (1 - eta * lambda) < 1 shrinks the weights toward zero at every iteration prior to applying the data loss gradient. This suppresses excessive weight magnitudes, penalizes complex non-linear oscillations, and forces the network to learn smooth, robust representations.`
    },
    {
      qNo: 'Q5',
      part: 'A',
      module: 3,
      marks: 3,
      co: 'CO3',
      blooms: 'K2 (Understand)',
      question: 'What is parameter sharing in Convolutional Neural Networks? State its primary computational advantage.',
      markingScheme: 'Concept of tied kernel weights: 1.5 marks | Memory & translation equivariance benefit: 1.5 marks',
      solution: `In a standard fully connected layer, every input unit connects to every output unit via a distinct weight parameter. In a convolutional layer, parameter sharing ties the weights such that the same small kernel filter (e.g., 3x3) slides across every spatial location of the input tensor.

Computational Advantages:
1. Drastic Parameter Reduction: For a 256x256 image connected to a 256x256 feature map, a dense layer requires (256*256)^2 approx 4.3 billion weights. A 3x3 kernel requires only 9 weights regardless of image size, avoiding catastrophic memory consumption and overfitting.
2. Translation Equivariance: If an input feature (e.g., an edge or eye) translates spatially by (dx, dy), the resulting feature map translates by the identical offset, preserving pattern detection across all spatial coordinates.`
    },
    {
      qNo: 'Q6',
      part: 'A',
      module: 3,
      marks: 3,
      co: 'CO3',
      blooms: 'K3 (Apply)',
      question: 'Calculate the output spatial dimensions of an image of size 64x64 passed through a convolution layer with 32 filters of size 5x5, padding P = 2, and stride S = 2.',
      markingScheme: 'Formula: 1 mark | Substitution and correct evaluated result: 2 marks',
      solution: `Given:
- Input dimension W_in = H_in = 64
- Filter size K = 5
- Padding P = 2
- Stride S = 2
- Number of filters C_out = 32

Formula:
W_out = floor((W_in - K + 2 * P) / S) + 1
H_out = floor((H_in - K + 2 * P) / S) + 1

Substitution:
W_out = floor((64 - 5 + 2 * 2) / 2) + 1
      = floor((64 - 5 + 4) / 2) + 1
      = floor(63 / 2) + 1
      = floor(31.5) + 1
      = 31 + 1 = 32.

Final Output Shape:
The resulting output feature map has dimensions 32 x 32 x 32 (32 spatial width, 32 spatial height, 32 feature channels).`
    },
    {
      qNo: 'Q7',
      part: 'A',
      module: 4,
      marks: 3,
      co: 'CO4',
      blooms: 'K2 (Understand)',
      question: 'Explain the purpose of the Forget Gate in an LSTM unit. Write its mathematical formula.',
      markingScheme: 'Pedagogical purpose: 1.5 marks | Equation with sigmoid activation: 1.5 marks',
      solution: `In standard Recurrent Neural Networks, past hidden state information is unconditionally overwritten at every time step, leading to rapid catastrophic forgetting or exploding gradients.

The Forget Gate f_t in a Long Short-Term Memory (LSTM) cell selectively decides what percentage of information from the previous cell state C_{t-1} should be preserved versus discarded based on the current input x_t and the previous hidden state h_{t-1}.

Mathematical Formula:
f_t = sigma(W_f * [h_{t-1}, x_t] + b_f)
where:
- sigma is the logistic sigmoid function with output bounded in [0, 1].
- If f_t = 0: The previous cell memory is completely cleared (reset).
- If f_t = 1: The previous cell memory is completely preserved without attenuation.`
    },
    {
      qNo: 'Q8',
      part: 'A',
      module: 4,
      marks: 3,
      co: 'CO4',
      blooms: 'K2 (Understand)',
      question: 'Contrast Gated Recurrent Units (GRU) with Long Short-Term Memory (LSTM) networks in terms of gating complexity.',
      markingScheme: '1.5 marks for structural comparison (gates) | 1.5 marks for state unification and parameter efficiency',
      solution: `1. Gate Count & Complexity:
- LSTM employs three distinct gates: Forget gate (f_t), Input gate (i_t), and Output gate (o_t).
- GRU streamlines this into two gates: Reset gate (r_t) and Update gate (z_t), eliminating the separate output gate.

2. State Representation:
- LSTM maintains two separate recurrence vectors: the Cell State C_t (linear conveyor belt) and the Hidden State h_t (filtered output).
- GRU merges the cell state and hidden state into a single recurrent vector h_t.

3. Parameter Efficiency:
Because GRU requires only 3 weight matrices per recurrent unit compared to 4 in LSTM, GRU trains approximately 25% faster with smaller memory consumption while achieving comparable sequence modeling performance on moderate-sized corpora.`
    },
    {
      qNo: 'Q9',
      part: 'A',
      module: 5,
      marks: 3,
      co: 'CO5',
      blooms: 'K2 (Understand)',
      question: 'Differentiate between Continuous Bag-of-Words (CBOW) and Continuous Skip-gram architectures in Word2Vec.',
      markingScheme: 'Objective function contrast: 1.5 marks | Context-to-target vs target-to-context: 1.5 marks',
      solution: `Both Word2Vec architectures learn dense vector representations of vocabulary tokens by optimizing language co-occurrence statistics:

1. Continuous Bag-of-Words (CBOW):
- Prediction Task: Predicts a single target word w_t given its surrounding context words within window c: P(w_t | w_{t-c}, ..., w_{t+c}).
- Characteristics: Sums/averages context word vectors, smoothing over distributional noise. Fast to train, highly effective for frequent vocabulary tokens.

2. Continuous Skip-Gram:
- Prediction Task: Predicts the surrounding context words w_{t+j} given a single target center word w_t: P(w_{t+j} | w_t) for j in [-c, +c].
- Characteristics: Treats each target-context pair independently. Takes longer to train but excels at capturing fine-grained semantics and rare words.`
    },
    {
      qNo: 'Q10',
      part: 'A',
      module: 5,
      marks: 3,
      co: 'CO5',
      blooms: 'K2 (Understand)',
      question: 'What is an Autoencoder? Explain the purpose of enforcing an information bottleneck in undercomplete autoencoders.',
      markingScheme: 'Encoder-decoder definition: 1.5 marks | Information bottleneck and non-trivial feature learning: 1.5 marks',
      solution: `An Autoencoder is an unsupervised neural network designed to copy its input x to its output x_hat through two cascaded mappings:
1. Encoder: h = f(x) = sigma(W_e * x + b_e) mapping input x in R^D to latent code h in R^d.
2. Decoder: x_hat = g(h) = sigma(W_d * h + b_d) reconstructing the original input.

Purpose of the Information Bottleneck:
If the latent dimension d >= D, the network could trivially memorize the identity mapping f(x) = x without discovering meaningful patterns.
In an undercomplete autoencoder, the latent code dimension is constrained to be strictly smaller than the input dimension (d << D). This forces the network to compress the data, discarding high-frequency noise and capturing the principal non-linear manifold structure of the training distribution.`
    },

    // PART B (5 Modules x 14 Marks = 70 Marks)
    {
      qNo: 'Q11',
      part: 'B',
      module: 1,
      marks: 14,
      co: 'CO1',
      blooms: 'K3 (Apply)',
      question: '(a) Derive the complete mathematical backpropagation equations for a 2-layer Multilayer Perceptron with Mean Squared Error loss. (10 Marks)\n(b) Explain the difference between empirical risk minimization and structural risk minimization. (4 Marks)',
      markingScheme: 'Forward pass equations: 2 marks | Output layer error delta derivation: 3 marks | Hidden layer error delta via chain rule: 3 marks | Weight update equations: 2 marks | Risk minimization explanation: 4 marks',
      solution: `PART (a): Backpropagation Derivation for 2-Layer MLP (10 Marks)

1. Network Architecture:
- Input vector: x in R^D
- Hidden layer: Pre-activation z_j = sum_i W_{ji}^{(1)} x_i + b_j^{(1)}; Hidden output a_j = sigma(z_j)
- Output layer: Pre-activation s_k = sum_j W_{kj}^{(2)} a_j + b_k^{(2)}; Output y_hat_k = g(s_k)
- Loss function (MSE): E = 0.5 * sum_k (y_k - y_hat_k)^2.

2. Output Layer Gradients:
By the multivariate chain rule:
dE / dW_{kj}^{(2)} = (dE / dy_hat_k) * (dy_hat_k / ds_k) * (ds_k / dW_{kj}^{(2)})
where:
- dE / dy_hat_k = -(y_k - y_hat_k)
- dy_hat_k / ds_k = g'(s_k)
- ds_k / dW_{kj}^{(2)} = a_j.

Define the output error delta:
delta_k^{(2)} = (dE / ds_k) = -(y_k - y_hat_k) * g'(s_k).
Thus:
dE / dW_{kj}^{(2)} = delta_k^{(2)} * a_j
dE / db_k^{(2)} = delta_k^{(2)}.

3. Hidden Layer Gradients:
dE / dW_{ji}^{(1)} = sum_k (dE / ds_k * ds_k / da_j) * (da_j / dz_j) * (dz_j / dW_{ji}^{(1)})
where:
- ds_k / da_j = W_{kj}^{(2)}
- da_j / dz_j = sigma'(z_j)
- dz_j / dW_{ji}^{(1)} = x_i.

Define the backpropagated hidden error delta:
delta_j^{(1)} = (sum_k delta_k^{(2)} * W_{kj}^{(2)}) * sigma'(z_j).
Thus:
dE / dW_{ji}^{(1)} = delta_j^{(1)} * x_i
dE / db_j^{(1)} = delta_j^{(1)}.

4. Gradient Descent Weight Updates:
W_{kj}^{(2)} <- W_{kj}^{(2)} - eta * delta_k^{(2)} * a_j
W_{ji}^{(1)} <- W_{ji}^{(1)} - eta * delta_j^{(1)} * x_i.

PART (b): Empirical vs Structural Risk Minimization (4 Marks)
- Empirical Risk Minimization (ERM): Seeks model parameters that minimize the average training loss R_emp(f) = (1/N) * sum_{i=1}^N L(y_i, f(x_i)). In overparameterized models, ERM easily overfits training noise without guaranteeing generalization.
- Structural Risk Minimization (SRM): Formulated by Vapnik based on Statistical Learning Theory. SRM minimizes an upper bound on true risk: R(f) <= R_emp(f) + O(sqrt(h * ln(N/h) / N)), balancing training empirical loss with a complexity penalty (such as VC-dimension or weight decay regularization) to guarantee optimal generalization.`
    },
    {
      qNo: 'Q13',
      part: 'B',
      module: 2,
      marks: 14,
      co: 'CO2',
      blooms: 'K3 (Apply)',
      question: '(a) Explain the Adam (Adaptive Moment Estimation) optimization algorithm with complete mathematical update equations. Why is bias correction required? (9 Marks)\n(b) Differentiate between Dropout training and inference procedures. Explain the inverted dropout convention. (5 Marks)',
      markingScheme: 'Adam 1st and 2nd raw moments: 3 marks | Bias correction derivation: 3 marks | Parameter update formula: 3 marks | Dropout training vs inference: 3 marks | Inverted dropout formula: 2 marks',
      solution: `PART (a): Adam Optimization Algorithm & Bias Correction (9 Marks)

Adam combines the benefits of Momentum (exponential moving average of gradients) and RMSProp (adaptive learning rates based on squared gradients).

1. Running Moment Estimators:
At time step t with mini-batch gradient g_t = grad_theta L(theta_t):
- 1st Moment (Mean velocity): m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t   [Standard beta_1 = 0.9]
- 2nd Moment (Uncentered variance): v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2   [Standard beta_2 = 0.999]

2. Need for Bias Correction:
Since m_0 = 0 and v_0 = 0, unrolling the recurrence gives:
m_t = (1 - beta_1) * sum_{i=1}^t beta_1^{t-i} * g_i.
Taking mathematical expectations assuming g_i comes from a stationary distribution with true mean E[g]:
E[m_t] = E[g] * (1 - beta_1) * sum_{i=1}^t beta_1^{t-i} = E[g] * (1 - beta_1^t).
Without correction, for early iterations (small t), (1 - beta_1^t) < 1, biasing m_t heavily toward zero.
Dividing by (1 - beta_1^t) yields an unbiased estimator E[m_hat_t] = E[g]:
m_hat_t = m_t / (1 - beta_1^t)
v_hat_t = v_t / (1 - beta_2^t).

3. Parameter Update Equation:
theta_{t+1} = theta_t - (alpha / (sqrt(v_hat_t) + epsilon)) * m_hat_t
where alpha is the learning rate (e.g. 0.001) and epsilon = 1e-8 prevents division by zero.

PART (b): Dropout & Inverted Dropout (5 Marks)
1. Dropout Training: For every forward pass, each hidden unit is randomly retained with probability p or set to zero with probability 1 - p (sampled from Bernoulli(p)). This prevents co-adaptation of features.
2. Inference Phase: All units remain active to produce deterministic predictions. In classical dropout, weights had to be scaled by p at test time: W_test = p * W_train.
3. Inverted Dropout Convention: Instead of scaling at test time, modern frameworks scale during training by dividing the active activations by p:
a_train = (m * a) / p   where m ~ Bernoulli(p).
At inference time, the forward pass runs unmodified without any scaling factor, ensuring zero computational overhead during production deployment.`
    },
    {
      qNo: 'Q15',
      part: 'B',
      module: 3,
      marks: 14,
      co: 'CO3',
      blooms: 'K3 (Apply)',
      question: '(a) Explain the fundamental operations in a standard Convolutional Layer: Convolution, Non-linear Activation, and Pooling. Contrast their functional roles. (8 Marks)\n(b) Describe Depthwise Separable Convolutions as used in MobileNet. Calculate the theoretical computational speedup compared to standard convolution. (6 Marks)',
      markingScheme: 'Convolution, Activation, Pooling roles: 8 marks | Depthwise + Pointwise architecture: 3 marks | Computational complexity ratio derivation: 3 marks',
      solution: `PART (a): Fundamental Operations in a CNN Layer (8 Marks)
1. Convolution Stage: Performs linear filtering using learnable kernels. It extracts local spatial patterns (edges, textures, motifs) via sparse connectivity and parameter sharing. Output at spatial location (x, y):
S(x, y) = (I * K)(x, y) = sum_i sum_j I(x-i, y-j) * K(i, j).
2. Detector Stage (Non-linear Activation): Applies a non-linear activation (typically ReLU: max(0, z)) point-wise to the linear feature maps. This breaks linearity, enabling the network to approximate complex non-linear decision surfaces without saturating gradients for positive activations.
3. Pooling Stage: Replaces the output at each location with a summary statistic of nearby outputs (Max Pooling or Average Pooling). This enforces local translation invariance and downsamples spatial dimensions, reducing computational memory in subsequent layers.

PART (b): Depthwise Separable Convolutions in MobileNet (6 Marks)
Standard convolution simultaneously filters and combines channels, incurring high computational cost:
Cost_std = D_K * D_K * M * N * D_F * D_F
where D_K = kernel size, M = input channels, N = output channels, D_F = spatial feature map size.

Depthwise Separable Convolution factors this operation into two lightweight steps:
1. Depthwise Convolution: Applies a single spatial filter per input channel without cross-channel communication:
Cost_dw = D_K * D_K * M * D_F * D_F.
2. Pointwise Convolution: Applies a 1x1 standard convolution to linearly combine the depthwise channels into N output channels:
Cost_pw = 1 * 1 * M * N * D_F * D_F.

Total Computational Cost:
Cost_sep = Cost_dw + Cost_pw = (D_K^2 * M * D_F^2) + (M * N * D_F^2).

Theoretical Speedup Ratio:
Ratio = Cost_sep / Cost_std = (D_K^2 * M * D_F^2 + M * N * D_F^2) / (D_K^2 * M * N * D_F^2)
      = (1 / N) + (1 / D_K^2).

For standard 3x3 kernels (D_K = 3) and N >> 1:
Ratio approx 1 / 3^2 = 1 / 9 approx 0.11.
Depthwise separable convolutions reduce computation and parameter count by approximately 8 to 9 times with only a minimal (1-2%) drop in top-1 classification accuracy.`
    },
    {
      qNo: 'Q17',
      part: 'B',
      module: 4,
      marks: 14,
      co: 'CO4',
      blooms: 'K3 (Apply)',
      question: '(a) Illustrate the complete architecture and gate mechanisms of a Long Short-Term Memory (LSTM) cell with mathematical equations. (9 Marks)\n(b) Explain the Encoder-Decoder sequence-to-sequence model for Neural Machine Translation with a neat diagram. (5 Marks)',
      markingScheme: 'LSTM diagram and 6 governing equations: 9 marks | Seq2Seq encoder-decoder architecture and context vector explanation: 5 marks',
      solution: `PART (a): Long Short-Term Memory (LSTM) Architecture (9 Marks)

An LSTM cell overcomes the vanishing gradient problem of vanilla RNNs by introducing an internal conveyor-belt state vector C_t regulated by three multiplicative sigmoid gates.

Governing Equations at time step t with input x_t and previous hidden state h_{t-1}:
1. Forget Gate: f_t = sigma(W_f * [h_{t-1}, x_t] + b_f)
Decides what portion of old cell memory C_{t-1} to discard (0 = erase, 1 = preserve).
2. Input Gate: i_t = sigma(W_i * [h_{t-1}, x_t] + b_i)
Decides which values of the new candidate state will be updated.
3. Candidate Cell State: C~_t = tanh(W_c * [h_{t-1}, x_t] + b_c)
Generates new candidate values in range [-1, +1] to add to the state.
4. Cell State Update: C_t = f_t (hadamard) C_{t-1} + i_t (hadamard) C~_t
Linearly combines the forgotten past state with the scaled new candidate. Crucially, the derivative dC_t / dC_{t-1} = f_t contains no matrix multiplication, preserving gradient flow over hundreds of steps!
5. Output Gate: o_t = sigma(W_o * [h_{t-1}, x_t] + b_o)
Decides what parts of the updated cell state will be emitted.
6. Hidden State Output: h_t = o_t (hadamard) tanh(C_t)
Emits the non-linearly squashed cell state filtered by the output gate.

PART (b): Encoder-Decoder Seq2Seq Architecture (5 Marks)
1. Architecture: Consists of two recurrent neural networks:
- Encoder RNN: Processes the variable-length input sequence (x_1, x_2, ..., x_T) step-by-step, updating hidden state h_t = f(h_{t-1}, x_t). The final hidden state h_T forms a fixed-dimensional Context Vector c summarizing the entire input sentence.
- Decoder RNN: Takes context vector c as its initial hidden state s_0 = c and sequentially generates the target sequence tokens (y_1, y_2, ..., y_{T'}) one word at a time, conditioning each token on previous output tokens: P(y_t | y_{<t}, c).
2. Bottleneck Limitation: In vanilla Seq2Seq, squeezing an entire paragraph into a single fixed-length vector c leads to information loss for long sequences, motivating the introduction of Attention mechanisms.`
    },
    {
      qNo: 'Q19',
      part: 'B',
      module: 5,
      marks: 14,
      co: 'CO5',
      blooms: 'K3 (Apply)',
      question: '(a) Explain the objective function and training methodology of GloVe (Global Vectors for Word Representation). (7 Marks)\n(b) Describe the architecture of Restricted Boltzmann Machines (RBM) and contrast energy-based models with standard feed-forward networks. (7 Marks)',
      markingScheme: 'GloVe log-bilinear model and weighted least squares formulation: 7 marks | RBM bipartite graph, energy function, and contrastive divergence: 7 marks',
      solution: `PART (a): Global Vectors for Word Representation (GloVe) (7 Marks)
GloVe combines the advantages of global matrix factorization (LSA) and local context window methods (Skip-gram).

1. Log-Bilinear Formulation:
Let X_{ij} denote the co-occurrence count of word i in the context of word j in a large corpus, and P_{ij} = X_{ij} / X_i be the probability that word j appears in the context of word i.
GloVe observes that the ratio of co-occurrence probabilities P_{ik} / P_{jk} with various probe words k effectively isolates thermodynamic semantic relationships.
The relationship is modeled as a dot product in vector space:
w_i^T * w_tilde_j + b_i + b_tilde_j = log(X_{ij}).

2. Weighted Least Squares Objective Function:
J = sum_{i,j=1}^V f(X_{ij}) * (w_i^T * w_tilde_j + b_i + b_tilde_j - log(X_{ij}))^2
where f(x) is a weighting function designed to prevent rare or overly frequent co-occurrences from dominating:
f(x) = (x / x_max)^alpha if x < x_max else 1 (typically x_max = 100, alpha = 0.75).
Training is performed via AdaGrad over all non-zero entries of the co-occurrence matrix X.

PART (b): Restricted Boltzmann Machines (RBM) (7 Marks)
1. Architecture: An RBM is an undirected graphical model (energy-based Markov Random Field) with a two-layer bipartite structure:
- Visible layer v in {0, 1}^D representing observed data.
- Hidden layer h in {0, 1}^F representing latent features.
- Restriction: There are NO intra-layer connections (no visible-visible or hidden-hidden links), enabling exact parallel conditional sampling:
P(h_j = 1 | v) = sigma(c_j + sum_i W_{ij} v_i)
P(v_i = 1 | h) = sigma(b_i + sum_j W_{ij} h_j).

2. Energy Function:
E(v, h) = - sum_i b_i v_i - sum_j c_j h_j - sum_i sum_j v_i W_{ij} h_j = - b^T v - c^T h - v^T W h.
The joint probability distribution is defined by the Boltzmann distribution:
P(v, h) = (1 / Z) * exp(-E(v, h))  where Z is the partition function.

3. Contrastive Divergence (CD-k):
Evaluating the exact gradient requires computing expectations over all 2^{D+F} states in Z, which is intractable. Hinton introduced Contrastive Divergence:
d log P(v) / dW_{ij} approx <v_i h_j>_data - <v_i h_j>_model,
where the negative phase expectation is approximated using k-step Gibbs sampling starting from the data vector v.`
    }
  ];

  const filteredQuestions = eseQuestions.filter(q => {
    if (filterModule !== 'all' && q.module !== filterModule) return false;
    if (filterPart !== 'all' && q.part !== filterPart) return false;
    return true;
  });

  return (
    <div className="space-y-6">
      
      {/* Header */}
      <div className="p-6 rounded-2xl bg-gradient-to-r from-navy-900 via-burgundy-950/60 to-navy-900 border border-gold-500/30 flex flex-col md:flex-row items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-burgundy-800 text-gold-300 border border-gold-500/30">
              KTU Examination Vault
            </span>
            <span className="text-xs text-slate-400 font-medium">100% Fully Solved with Verbatim Marking Schemes</span>
          </div>
          <h2 className="text-lg md:text-xl font-bold font-serif text-slate-100 mt-1">
            KTU End Semester Examination (ESE) &amp; Internal Series Test Solutions
          </h2>
          <p className="text-xs text-slate-300 mt-1">
            Compliant with KTU Scheme 2019 regulations (Part A: 10×3=30 marks, Part B: 5×14=70 marks).
          </p>
        </div>

        {/* Paper Switcher */}
        <div className="flex items-center bg-navy-950 rounded-xl p-1 border border-gold-500/30 shadow-lg">
          <button
            onClick={() => setActivePaper('ese')}
            className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
              activePaper === 'ese'
                ? 'bg-burgundy-800 text-gold-200 border border-gold-500/40 shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            ESE Model Paper (100M)
          </button>
          <button
            onClick={() => setActivePaper('cie1')}
            className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
              activePaper === 'cie1'
                ? 'bg-burgundy-800 text-gold-200 border border-gold-500/40 shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Series Test 1 (50M)
          </button>
          <button
            onClick={() => setActivePaper('cie2')}
            className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
              activePaper === 'cie2'
                ? 'bg-burgundy-800 text-gold-200 border border-gold-500/40 shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Series Test 2 (50M)
          </button>
        </div>
      </div>

      {/* Filter and Action Bar */}
      <div className="p-4 rounded-xl bg-navy-900/80 border border-gold-500/20 flex flex-wrap items-center justify-between gap-4">
        
        {/* Module & Part Filters */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="text-xs font-bold text-gold-400">Module:</span>
            <button
              onClick={() => setFilterModule('all')}
              className={`px-2.5 py-1 rounded text-xs font-semibold ${
                filterModule === 'all' ? 'bg-gold-500 text-navy-950' : 'bg-navy-950 text-slate-300'
              }`}
            >
              All
            </button>
            {[1, 2, 3, 4, 5].map(m => (
              <button
                key={m}
                onClick={() => setFilterModule(m)}
                className={`px-2.5 py-1 rounded text-xs font-semibold ${
                  filterModule === m ? 'bg-gold-500 text-navy-950' : 'bg-navy-950 text-slate-300'
                }`}
              >
                M{m}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-1.5 border-l border-slate-700 pl-3">
            <span className="text-xs font-bold text-gold-400">Part:</span>
            <button
              onClick={() => setFilterPart('all')}
              className={`px-2.5 py-1 rounded text-xs font-semibold ${
                filterPart === 'all' ? 'bg-burgundy-800 text-gold-200' : 'bg-navy-950 text-slate-300'
              }`}
            >
              All
            </button>
            <button
              onClick={() => setFilterPart('A')}
              className={`px-2.5 py-1 rounded text-xs font-semibold ${
                filterPart === 'A' ? 'bg-burgundy-800 text-gold-200' : 'bg-navy-950 text-slate-300'
              }`}
            >
              Part A (3M)
            </button>
            <button
              onClick={() => setFilterPart('B')}
              className={`px-2.5 py-1 rounded text-xs font-semibold ${
                filterPart === 'B' ? 'bg-burgundy-800 text-gold-200' : 'bg-navy-950 text-slate-300'
              }`}
            >
              Part B (14M)
            </button>
          </div>
        </div>

        {/* Expand / Collapse All */}
        <div className="flex items-center gap-2">
          <button
            onClick={expandAll}
            className="px-2.5 py-1 bg-navy-950 hover:bg-slate-800 border border-slate-700 rounded text-xs text-slate-300 transition-colors"
          >
            Expand All
          </button>
          <button
            onClick={collapseAll}
            className="px-2.5 py-1 bg-navy-950 hover:bg-slate-800 border border-slate-700 rounded text-xs text-slate-300 transition-colors"
          >
            Collapse All
          </button>
          <button
            onClick={() => window.print()}
            className="flex items-center gap-1 px-3 py-1 bg-navy-950 hover:bg-slate-800 border border-slate-700 rounded text-xs text-gold-300 transition-colors"
          >
            <Printer className="w-3.5 h-3.5" />
            <span>Print Paper</span>
          </button>
        </div>

      </div>

      {/* Questions List */}
      <div className="space-y-4">
        {filteredQuestions.map((q) => {
          const isExpanded = expandedQuestions[q.qNo] ?? false;

          return (
            <div 
              key={q.qNo}
              className="rounded-xl border border-gold-500/20 bg-navy-900/80 overflow-hidden shadow-md transition-all"
            >
              {/* Question Header */}
              <div 
                onClick={() => toggleExpand(q.qNo)}
                className="p-4 bg-navy-950/60 hover:bg-navy-950 cursor-pointer flex items-start justify-between gap-4 select-none"
              >
                <div className="flex items-start gap-3">
                  <span className="w-9 h-9 rounded-lg bg-burgundy-900 text-gold-300 font-bold flex items-center justify-center text-xs shrink-0 border border-gold-500/30">
                    {q.qNo}
                  </span>
                  <div>
                    <div className="flex items-center gap-2 flex-wrap mb-1">
                      <span className="px-2 py-0.2 rounded text-[10px] font-bold bg-navy-900 text-gold-400 border border-slate-700">
                        Module {q.module}
                      </span>
                      <span className="px-2 py-0.2 rounded text-[10px] font-bold bg-emerald-950 text-emerald-300 border border-emerald-500/30">
                        {q.marks} Marks
                      </span>
                      <span className="px-2 py-0.2 rounded text-[10px] font-bold bg-navy-900 text-slate-300 border border-slate-700">
                        {q.co}
                      </span>
                      <span className="px-2 py-0.2 rounded text-[10px] text-slate-400 bg-slate-900 border border-slate-800">
                        {q.blooms}
                      </span>
                    </div>
                    <h3 className="text-xs md:text-sm font-semibold text-slate-100 whitespace-pre-line leading-relaxed">
                      {q.question}
                    </h3>
                  </div>
                </div>

                <div className="text-slate-400 shrink-0 mt-1">
                  {isExpanded ? <ChevronUp className="w-4 h-4 text-gold-400" /> : <ChevronDown className="w-4 h-4" />}
                </div>
              </div>

              {/* Solution Body */}
              {isExpanded && (
                <div className="p-5 border-t border-slate-800 bg-navy-950/40 space-y-4">
                  {/* Marking Scheme */}
                  <div className="p-3 rounded-lg bg-burgundy-950/30 border border-gold-500/30 text-xs">
                    <span className="font-bold text-gold-300 uppercase tracking-wide mr-2">KTU Marking Scheme:</span>
                    <span className="text-slate-300">{q.markingScheme}</span>
                  </div>

                  {/* Solution Text */}
                  <div className="text-xs md:text-sm text-slate-200 font-sans leading-relaxed whitespace-pre-line bg-navy-950 p-4 rounded-xl border border-slate-800">
                    {q.solution}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

    </div>
  );
};

