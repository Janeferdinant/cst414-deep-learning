"""
Curriculum Data for Module 1: Neural Networks (7 Hours)
KTU CST414 Deep Learning (Scheme 2019)
Faculty: Jane Rose | Department of Computer Science & Engineering
"""

MODULE_1_DAYS = [
    {
        "day": 1,
        "module": 1,
        "topic_id": "1.1",
        "title": "Introduction to Neural Networks - Single Layer Perceptrons",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO1",
        "blooms": "Understand / Apply",
        "prerequisites": "Linear Algebra (vectors, dot products), Basic Calculus, Python basics",
        "learning_objectives": [
            "Understand the biological inspiration and mathematical formulation of artificial neurons.",
            "Formulate the Single Layer Perceptron (SLP) model by Frank Rosenblatt.",
            "Analyze the Perceptron Learning Algorithm (PLA) and its convergence theorem.",
            "Design single layer perceptrons for linearly separable logical gates (AND, OR, NAND)."
        ],
        "key_terminology": [
            "Artificial Neuron", "Threshold Logic Unit (TLU)", "Perceptron Learning Rule",
            "Linear Separability", "Hyperplane Decision Boundary", "Bias Weight"
        ],
        "textbook_ref": "Charu C. Aggarwal: Neural Networks and Deep Learning, Chapter 1; Michael Nielsen: Neural Networks and Deep Learning, Chapter 1.",
        "notes": {
            "introduction": "Artificial Neural Networks (ANNs) are computational models inspired by the human biological nervous system. The fundamental computational unit is the artificial neuron, first conceptualized as the McCulloch-Pitts neuron (1943) and later modernized into the trainable Perceptron by Frank Rosenblatt (1958).",
            "mathematical_model": (
                "A Single Layer Perceptron receives an n-dimensional input vector x = [x1, x2, ..., xn]^T and computes "
                "the net input z (pre-activation) as the inner product with weight vector w = [w1, w2, ..., wn]^T plus bias b:\n\n"
                "z = sum_{i=1}^n (w_i * x_i) + b = w^T * x + b\n\n"
                "The output y is determined by a step activation function:\n"
                "y = f(z) = { +1 if z >= 0 (or 1 in unipolar)\n"
                "           {  0 if z < 0  (or -1 in bipolar)\n\n"
                "Geometrically, w^T * x + b = 0 defines an (n-1)-dimensional hyperplane that partitions the input space into two decision regions."
            ),
            "perceptron_learning_algorithm": (
                "Perceptron Learning Algorithm (PLA):\n"
                "1. Initialize weights w and bias b to 0 or small random values.\n"
                "2. For each training sample (x_k, d_k), where d_k is the target label:\n"
                "   a. Compute predicted output: y_k = f(w^T * x_k + b)\n"
                "   b. Compute error: e_k = d_k - y_k\n"
                "   c. Update parameters using learning rate eta (0 < eta <= 1):\n"
                "      w <- w + eta * (d_k - y_k) * x_k\n"
                "      b <- b + eta * (d_k - y_k)\n"
                "3. Repeat until all training patterns are classified correctly (zero error).\n\n"
                "Convergence Theorem (Novikoff, 1962): If the training data is linearly separable with margin gamma and bounded by ||x|| <= R, the PLA is guaranteed to converge in at most (R/gamma)^2 update steps."
            ),
            "linear_separability_and_limitation": (
                "Minsky & Papert (1969) proved that single-layer perceptrons can only classify linearly separable patterns. "
                "For non-linearly separable problems like the XOR function, no single linear decision boundary exists that can separate the classes, necessitating Multi-Layer Perceptrons."
            ),
            "exam_important_points": [
                "Derivation and geometric interpretation of the hyperplane decision boundary w^T * x + b = 0.",
                "Step-by-step weight updates using the Perceptron Learning Rule.",
                "Proof of linear separability for AND/OR and inability to solve XOR.",
                "Novikoff's Perceptron Convergence Theorem condition and upper bound on iterations."
            ]
        },
        "teaching_script": {
            "00-05": "Recap course syllabus, establish connection between biological neurons (dendrites, soma, axon, synapse) and mathematical abstractions (inputs, weights, activation, output).",
            "05-15": "Derive the mathematical equation of the Perceptron: z = w^T * x + b. Discuss unipolar vs bipolar step functions.",
            "15-30": "Present the geometric interpretation: how weights determine hyperplane normal vector and bias shifts the plane from origin. Work out AND gate boundary on blackboard.",
            "30-45": "Walk through Perceptron Learning Algorithm step-by-step. Solve a full weight update cycle on blackboard for a misclassified point.",
            "45-55": "Demonstrate the interactive Python script showing decision boundary shifting during training epochs.",
            "55-60": "Conduct 5-minute viva quiz. Pose the Minsky-Papert XOR question as a teaser for Lecture 1.2."
        },
        "python_demo": {
            "title": "Perceptron from Scratch for Linearly Separable Gates",
            "description": "Implementation of Rosenblatt Perceptron using pure NumPy and Matplotlib decision boundary plotting.",
            "code": """import numpy as np
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
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Manual Output Calculation of Single Neuron",
                "statement": "Given a 3-dimensional input vector x = [2, 2, 1]^T connected to a single neuron with weight vector w = [0.3, -0.4, 0.5]^T and bias b = -0.1. Calculate the net input (pre-activation) z and the final output y using (a) a unipolar step function and (b) a binary sigmoid activation function sigma(z) = 1 / (1 + e^(-z)).",
                "given": "x = [2, 2, 1]^T, w = [0.3, -0.4, 0.5]^T, b = -0.1",
                "formulas": "z = sum(w_i * x_i) + b; y_step = 1 if z >= 0 else 0; y_sigmoid = 1 / (1 + exp(-z))",
                "steps": [
                    "Step 1: Compute weighted sum: sum = (0.3 * 2) + (-0.4 * 2) + (0.5 * 1) = 0.6 - 0.8 + 0.5 = 0.3.",
                    "Step 2: Add bias: z = sum + b = 0.3 + (-0.1) = 0.2.",
                    "Step 3: Unipolar step function: Since z = 0.2 >= 0, y_step = 1.",
                    "Step 4: Binary sigmoid function: y_sigmoid = 1 / (1 + e^(-0.2)) = 1 / (1 + 0.81873) = 1 / 1.81873 = 0.5500."
                ],
                "answer": "Net input z = 0.2, Step output = 1, Sigmoid output = 0.5500",
                "interpretation": "The positive pre-activation indicates that the input vector lies on the positive side of the separating hyperplane. The sigmoid yields an activated confidence probability of 55%."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Perceptron Learning Rule Step-by-Step Weight Update",
                "statement": "Design a single-layer perceptron to compute the 2-input NAND function. Initial weights are w1 = 0.0, w2 = 0.0 with bias b = 0.0. The learning rate is eta = 0.5. Perform the first epoch of training using the Perceptron Learning Algorithm for the sequential training samples: (0,0)->1, (0,1)->1, (1,0)->1, (1,1)->0. Show all intermediate calculations.",
                "given": "Initial w = [0.0, 0.0], b = 0.0, eta = 0.5. Activation: step function (y=1 if z>=0 else 0). Samples: x1=(0,0), d1=1; x2=(0,1), d2=1; x3=(1,0), d3=1; x4=(1,1), d4=0.",
                "formulas": "z = w1*x1 + w2*x2 + b; y = 1 if z >= 0 else 0; delta_w = eta * (d - y) * x; delta_b = eta * (d - y)",
                "steps": [
                    "Sample 1: x=(0,0), d=1. z = 0*0 + 0*0 + 0 = 0. Pred y = 1. Error = 1 - 1 = 0. No update. w=[0.0, 0.0], b=0.0.",
                    "Sample 2: x=(0,1), d=1. z = 0*0 + 0*1 + 0 = 0. Pred y = 1. Error = 1 - 1 = 0. No update. w=[0.0, 0.0], b=0.0.",
                    "Sample 3: x=(1,0), d=1. z = 0*1 + 0*0 + 0 = 0. Pred y = 1. Error = 1 - 1 = 0. No update. w=[0.0, 0.0], b=0.0.",
                    "Sample 4: x=(1,1), d=0. z = 0*1 + 0*1 + 0 = 0. Pred y = 1. Error = 0 - 1 = -1 (MISCLASSIFIED!).",
                    "Update on Sample 4: delta_w1 = 0.5 * (-1) * 1 = -0.5; delta_w2 = 0.5 * (-1) * 1 = -0.5; delta_b = 0.5 * (-1) = -0.5.",
                    "New weights after Epoch 1: w1 = -0.5, w2 = -0.5, b = -0.5."
                ],
                "answer": "Weights after Epoch 1: w1 = -0.5, w2 = -0.5, b = -0.5.",
                "interpretation": "The perceptron adjusted its hyperplane to penalize the false positive at (1,1). Further epochs will refine bias and weights until the decision boundary separates (1,1) from the remaining vertices."
            }
        ],
        "real_world_application": {
            "title": "Industrial Defect Screening via Linear Separator",
            "domain": "Smart Manufacturing & Automated Quality Control",
            "workflow": "High-speed camera captures manufactured ceramic tiles -> Edge detection extracts contrast ratio (x1) and surface roughness variance (x2) -> Calibrated single-layer threshold neuron performs millisecond binary pass/fail classification.",
            "benefit": "Enables ultra-low-latency (sub-millisecond) hardware defect filtering on microcontrollers without requiring deep GPU hardware."
        },
        "viva_questions": [
            {"q": "What is the biological analogue of weights in an artificial neuron?", "a": "Synaptic connection strength (synapse efficacy)."},
            {"q": "Define linear separability in n-dimensional Euclidean space.", "a": "Two sets of points are linearly separable if there exists an (n-1)-dimensional hyperplane that cleanly divides the space with all points of one set on one side and the other set on the opposite side."},
            {"q": "Why can a single-layer perceptron not solve the XOR problem?", "a": "Because the XOR truth table produces non-linearly separable patterns in 2D space where true and false instances lie on alternating diagonal vertices."},
            {"q": "What role does the bias term play in a neuron?", "a": "The bias allows the decision hyperplane to shift away from the coordinate origin, enabling flexible positioning of the decision threshold."},
            {"q": "State Novikoff's Perceptron Convergence Theorem.", "a": "If training samples are linearly separable with margin gamma and bounded norm R, the PLA will terminate with a separating solution in at most (R/gamma)^2 weight updates."},
            {"q": "What happens if PLA is executed on non-separable data?", "a": "The algorithm will enter an infinite loop and fail to terminate; the Pocket Algorithm or soft-margin approaches are used instead."},
            {"q": "What is the difference between unipolar and bipolar step activation functions?", "a": "Unipolar maps inputs to {0, 1}, whereas bipolar maps inputs to {-1, +1}."},
            {"q": "Why is the step function non-differentiable at z = 0?", "a": "It possesses an instantaneous jump discontinuity at z = 0, meaning its derivative is undefined there and zero everywhere else, preventing gradient-based optimization."},
            {"q": "How does learning rate eta influence Perceptron training?", "a": "Eta scales the step size of weight corrections. While convergence is guaranteed for any eta > 0 on separable data, large values cause oscillations while very small values slow convergence."},
            {"q": "Can a single neuron solve the NAND logic gate?", "a": "Yes, because NAND is linearly separable; for example, weights w1 = -2, w2 = -2, and bias b = 3 correctly classify all 4 inputs."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "The decision boundary formed by a single neuron with 2 inputs is a:",
                    "options": ["Point", "Straight line", "Parabola", "Hyperbola"],
                    "ans": "Straight line"
                },
                {
                    "q": "Which logical function among the following is NOT linearly separable?",
                    "options": ["AND", "OR", "NOR", "XOR"],
                    "ans": "XOR"
                },
                {
                    "q": "In Frank Rosenblatt's perceptron learning rule, weight update occurs:",
                    "options": ["At every iteration regardless of prediction", "Only when the sample is misclassified", "Only when prediction is correct", "Randomly"],
                    "ans": "Only when the sample is misclassified"
                }
            ],
            "short_questions": [
                "Explain the geometric significance of the weight vector w in relation to the decision boundary hyperplane.",
                "State the Perceptron Convergence Theorem and discuss its core assumptions."
            ],
            "application_question": "A medical sensor system measures body temperature and heart rate to detect heat stroke. Explain how a single perceptron can be parameterized to act as a frontline safety alarm.",
            "homework": "Prove algebraically that no combination of real weights (w1, w2) and real bias b can satisfy the four inequalities of the 2-input XOR truth table."
        }
    },
    {
        "day": 2,
        "module": 1,
        "topic_id": "1.2",
        "title": "Multi-Layer Perceptrons (MLPs) and Representation Power of MLPs",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO1",
        "blooms": "Understand / Analyze",
        "prerequisites": "Single Layer Perceptron, Matrix multiplication, Linear Separability",
        "learning_objectives": [
            "Explain the architecture of Multi-Layer Perceptrons (feedforward networks).",
            "Demonstrate how hidden layers overcome linear separability limitations (solving XOR).",
            "State and analyze the Universal Approximation Theorem (Cybenko, Hornik).",
            "Calculate total trainable parameters in deep feedforward architectures."
        ],
        "key_terminology": [
            "Input Layer", "Hidden Layer", "Output Layer", "Universal Approximation Theorem",
            "Feature Space Warping", "Feedforward Architecture"
        ],
        "textbook_ref": "Goodfellow, Bengio, Courville: Deep Learning, Chapter 6; Aggarwal: Neural Networks and Deep Learning, Chapter 1.",
        "notes": {
            "introduction": "A Multi-Layer Perceptron (MLP) is a class of feedforward artificial neural network consisting of at least three layers of nodes: an input layer, one or more hidden layers, and an output layer. Except for input nodes, each node is a neuron that uses a non-linear activation function.",
            "architecture_and_forward_pass": (
                "For an MLP with L layers:\n"
                "Layer 0: Input vector a^(0) = x\n"
                "For layer l = 1, 2, ..., L:\n"
                "  z^(l) = W^(l) * a^(l-1) + b^(l)\n"
                "  a^(l) = sigma(z^(l))\n"
                "where W^(l) is an (n_l x n_{l-1}) weight matrix, b^(l) is an (n_l x 1) bias vector, and sigma is an element-wise non-linear activation function.\n"
                "The final network prediction is y_hat = a^(L)."
            ),
            "representation_power_and_xor": (
                "How MLPs resolve the XOR problem:\n"
                "By mapping original 2D input space (x1, x2) through a hidden layer with 2 neurons (h1, h2), the input space is transformed into a new feature space where the points become linearly separable.\n"
                "Specifically:\n"
                "h1 = ReLU(x1 + x2 - 0.5) (acts as an OR-like detector)\n"
                "h2 = ReLU(x1 + x2 - 1.5) (acts as an AND-like detector)\n"
                "Output y = h1 - 2 * h2 produces exactly the XOR truth table!"
            ),
            "universal_approximation_theorem": (
                "Cybenko (1989) and Hornik (1991) proved the Universal Approximation Theorem:\n"
                "A standard feedforward neural network with a single hidden layer containing a finite number of neurons with non-linear activation functions can approximate any continuous function f: R^n -> R^m on compact subsets of R^n to arbitrary precision epsilon > 0.\n\n"
                "Crucial Caveat: The theorem guarantees the EXISTENCE of such an approximation, but it does NOT guarantee that the required weights can be discovered by gradient-based learning algorithms, nor does it guarantee that a single hidden layer is computationally or parameter-efficient compared to deeper architectures."
            ),
            "exam_important_points": [
                "Full architectural diagram of MLP with matrix dimensions for weights and biases.",
                "Algebraic demonstration of XOR resolution using hidden representations.",
                "Formal statement and engineering implications of the Universal Approximation Theorem.",
                "Calculation of total parameters (weights + biases) across multi-layer networks."
            ]
        },
        "teaching_script": {
            "00-05": "Review Day 1 limitation: Minsky-Papert XOR barrier. Introduce the hidden layer as a non-linear feature extractor.",
            "05-20": "Draw 2-layer MLP architecture on blackboard. Define notation: W^(1), b^(1), z^(1), a^(1), W^(2), b^(2). Write general forward propagation equations.",
            "20-35": "Walk through geometric XOR resolution: show how hidden layer warps the 4 input coordinates (0,0), (0,1), (1,0), (1,1) into a linearly separable 2D hidden space.",
            "35-45": "Explain the Universal Approximation Theorem: Cybenko's theorem, width vs depth trade-offs, why deep networks require exponentially fewer parameters than shallow networks.",
            "45-55": "Work out parameter counting problem for a 3-layer architecture with students.",
            "55-60": "Review core concepts and announce Day 3: Activation Functions and Loss Formulations."
        },
        "python_demo": {
            "title": "Solving XOR with Multi-Layer Perceptron using NumPy",
            "description": "Forward pass and decision surface visualization demonstrating non-linear classification boundary on XOR.",
            "code": """import numpy as np

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
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Trainable Parameter Count in Multi-Layer Perceptron",
                "statement": "An MLP has an input layer with 784 features, a first hidden layer with 256 neurons, a second hidden layer with 128 neurons, and an output layer with 10 classes. Assuming all layers are fully connected with individual biases, calculate: (a) Number of weight parameters between each layer, (b) Number of bias parameters in each layer, and (c) Total trainable parameters in the entire network.",
                "given": "Layer 0 (Input) n0 = 784, Layer 1 n1 = 256, Layer 2 n2 = 128, Layer 3 (Output) n3 = 10.",
                "formulas": "Weights between layer (l-1) and l: W_l = n_l * n_{l-1}; Biases in layer l: B_l = n_l; Total = sum(W_l + B_l)",
                "steps": [
                    "Layer 1 (784 -> 256): Weights = 784 * 256 = 200,704. Biases = 256.",
                    "Layer 2 (256 -> 128): Weights = 256 * 128 = 32,768. Biases = 128.",
                    "Layer 3 (128 -> 10):  Weights = 128 * 10 = 1,280. Biases = 10.",
                    "Total Weights = 200,704 + 32,768 + 1,280 = 234,752.",
                    "Total Biases = 256 + 128 + 10 = 394.",
                    "Total Trainable Parameters = 234,752 + 394 = 235,146."
                ],
                "answer": "Total Trainable Parameters = 235,146",
                "interpretation": "The majority of parameters (~85.3%) reside in the initial fully connected transition from input to the first hidden layer."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Feedforward Output Calculation of 2-Hidden-Layer MLP",
                "statement": "Consider a 2-2-1 MLP with input x = [1.0, -1.0]^T. Hidden Layer 1 has weight matrix W^(1) = [[0.5, -0.2], [0.8, 0.1]] and bias b^(1) = [0.1, -0.2]^T with ReLU activation. Output layer has weight vector W^(2) = [0.7, -0.4] and bias b^(2) = 0.3 with Sigmoid activation. Calculate the step-by-step activations and the final network output.",
                "given": "x = [1.0, -1.0]^T, W^(1) = [[0.5, -0.2], [0.8, 0.1]], b^(1) = [0.1, -0.2]^T, W^(2) = [0.7, -0.4], b^(2) = 0.3",
                "formulas": "z^(1) = W^(1)*x + b^(1); a^(1) = max(0, z^(1)); z^(2) = W^(2)*a^(1) + b^(2); y = 1 / (1 + exp(-z^(2)))",
                "steps": [
                    "Step 1: Compute z1^(1) = (0.5 * 1.0) + (-0.2 * -1.0) + 0.1 = 0.5 + 0.2 + 0.1 = 0.8.",
                    "Step 2: Compute z2^(1) = (0.8 * 1.0) + (0.1 * -1.0) + (-0.2) = 0.8 - 0.1 - 0.2 = 0.5.",
                    "Step 3: Apply ReLU: a1^(1) = max(0, 0.8) = 0.8; a2^(1) = max(0, 0.5) = 0.5.",
                    "Step 4: Compute z^(2) = (0.7 * 0.8) + (-0.4 * 0.5) + 0.3 = 0.56 - 0.20 + 0.3 = 0.66.",
                    "Step 5: Apply Sigmoid: y = 1 / (1 + e^(-0.66)) = 1 / (1 + 0.51685) = 1 / 1.51685 = 0.6593."
                ],
                "answer": "Hidden activations a^(1) = [0.8, 0.5]^T, Final Output y = 0.6593",
                "interpretation": "Both hidden neurons are active (positive pre-activations), allowing information to propagate forward and yielding a positive prediction confidence of 65.93%."
            }
        ],
        "real_world_application": {
            "title": "Retail Loan Default Prediction Engine",
            "domain": "Fintech & Credit Risk Assessment",
            "workflow": "Customer credit bureau features (debt-to-income, credit score, delinquencies, monthly inquiries) fed into a 4-layer MLP -> Hidden layers synthesize non-linear interaction patterns -> Output sigmoid neuron outputs probability of default within 90 days.",
            "benefit": "Detects complex non-linear repayment risk signals that linear logistic regression models miss, cutting default rates by up to 18%."
        },
        "viva_questions": [
            {"q": "What is the primary architectural difference between an SLP and an MLP?", "a": "An MLP contains one or more hidden layers between the input and output layers with non-linear activation functions."},
            {"q": "Why is a non-linear activation function strictly required in hidden layers?", "a": "Without non-linear activations, a composition of linear layers collapses mathematically into a single linear transformation: W2*(W1*x + b1) + b2 = (W2*W1)*x + (W2*b1 + b2), removing the benefit of depth."},
            {"q": "What does the Universal Approximation Theorem state?", "a": "A single hidden layer feedforward network with non-linear activations can approximate any continuous function on compact subsets of R^n to arbitrary accuracy, given sufficient hidden units."},
            {"q": "If a shallow network can approximate any function, why do we use deep networks?", "a": "Shallow networks may require an exponentially large number of hidden neurons. Deep networks represent hierarchical compositional features with exponentially fewer parameters."},
            {"q": "How many parameters are in a dense layer connecting 100 inputs to 50 outputs?", "a": "100 * 50 = 5,000 weights + 50 biases = 5,050 trainable parameters."},
            {"q": "What is meant by 'feedforward'?", "a": "Signals travel strictly in one direction from input to output without cycles or feedback loops."},
            {"q": "How does the hidden layer resolve the XOR problem geometrically?", "a": "It projects the 2D non-separable input space into a transformed hidden space where the points become linearly separable."},
            {"q": "What is the role of the bias vector in matrix notation W*x + b?", "a": "It provides each neuron with an independent translation parameter so activation does not depend solely on origin-centered inputs."},
            {"q": "Does the Universal Approximation Theorem ensure that backpropagation will find the optimal weights?", "a": "No, it is an existential proof, not a constructive algorithmic guarantee; optimization may get stuck in local minima or saddle points."},
            {"q": "What is the dimension of the weight matrix connecting layer l-1 (dim m) to layer l (dim n)?", "a": "Matrix of shape n x m (n rows, m columns)."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "If an MLP has strictly linear activation functions in all hidden layers, it is mathematically equivalent to:",
                    "options": ["A recurrent neural network", "A single-layer linear model", "A support vector machine", "A decision tree"],
                    "ans": "A single-layer linear model"
                },
                {
                    "q": "The Universal Approximation Theorem was established by:",
                    "options": ["Rosenblatt (1958)", "Minsky and Papert (1969)", "Cybenko (1989) and Hornik (1991)", "Rumelhart and Hinton (1986)"],
                    "ans": "Cybenko (1989) and Hornik (1991)"
                },
                {
                    "q": "Total parameters in an MLP with architecture 10-20-1 (fully connected with biases) is:",
                    "options": ["200", "220", "241", "251"],
                    "ans": "241"
                }
            ],
            "short_questions": [
                "Prove that a 2-layer MLP with identity activations collapses into a single-layer linear model.",
                "Explain how hidden layers perform feature representation learning."
            ],
            "application_question": "A fraud detection system takes 15 behavioral features and outputs a binary fraud alert. Propose an MLP architecture and calculate its parameter count.",
            "homework": "Draw the architecture of a 3-layer MLP that computes the 3-input Parity function (Odd Parity). Specify the required hidden layer capacity."
        }
    },
    {
        "day": 3,
        "module": 1,
        "topic_id": "1.3",
        "title": "Activation Functions, Risk Minimization and Loss Functions",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO1",
        "blooms": "Understand / Apply",
        "prerequisites": "Calculus (derivatives, partial derivatives), Probability basics",
        "learning_objectives": [
            "Compare mathematical formulations, derivatives, and characteristics of Sigmoid, Tanh, ReLU, and Softmax.",
            "Explain the vanishing gradient problem inherent in saturating activation functions.",
            "Formulate the Empirical Risk Minimization (ERM) framework and Structural Risk Minimization.",
            "Derive and contrast Mean Squared Error (MSE) and Cross-Entropy loss functions."
        ],
        "key_terminology": [
            "Activation Function", "Sigmoid", "Hyperbolic Tangent (Tanh)", "Rectified Linear Unit (ReLU)",
            "Softmax", "Empirical Risk Minimization", "Cross-Entropy Loss", "Negative Log-Likelihood"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 6; Aggarwal: Neural Networks and Deep Learning, Chapter 2.",
        "notes": {
            "activation_functions": (
                "1. Sigmoid (Logistic):\n"
                "   sigma(z) = 1 / (1 + e^(-z)), Range: (0, 1)\n"
                "   Derivative: sigma'(z) = sigma(z) * (1 - sigma(z)), Max derivative = 0.25 at z = 0.\n"
                "   Drawbacks: Vanishing gradient for |z| >> 0; Not zero-centered (causes zig-zagging gradient updates).\n\n"
                "2. Hyperbolic Tangent (Tanh):\n"
                "   tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z)), Range: (-1, 1)\n"
                "   Derivative: tanh'(z) = 1 - tanh^2(z), Max derivative = 1.0 at z = 0.\n"
                "   Advantage: Zero-centered output; faster convergence than sigmoid, but still saturates at extremes.\n\n"
                "3. Rectified Linear Unit (ReLU):\n"
                "   f(z) = max(0, z), Range: [0, infinity)\n"
                "   Derivative: f'(z) = 1 if z > 0, else 0.\n"
                "   Advantages: Computationally ultra-efficient, does not saturate in positive regime (mitigates vanishing gradients), induces sparse representations.\n"
                "   Drawback: 'Dying ReLU' problem when neurons permanently output 0.\n\n"
                "4. Softmax (Multiclass Output):\n"
                "   softmax(z_i) = e^(z_i) / sum_{j=1}^K e^(z_j), Range: (0, 1), sum_{i=1}^K softmax(z_i) = 1.\n"
                "   Converts raw logits into a valid categorical probability distribution."
            ),
            "risk_minimization": (
                "Statistical Learning Theory defines expected risk R(theta) as the expected loss over the true data distribution p_data(x, y):\n"
                "R(theta) = E_{(x,y)~p_data} [ L(f(x; theta), y) ]\n\n"
                "Since p_data is unknown, Empirical Risk Minimization (ERM) replaces true expectation with sample average over N training examples:\n"
                "R_emp(theta) = (1 / N) * sum_{i=1}^N L(f(x_i; theta), y_i)\n\n"
                "Structural Risk Minimization (SRM) adds a regularization penalty Omega(theta) to control model capacity and avoid overfitting:\n"
                "R_struct(theta) = R_emp(theta) + lambda * Omega(theta)"
            ),
            "loss_functions": (
                "1. Mean Squared Error (Regression):\n"
                "   L_MSE = (1 / 2) * (y - y_hat)^2 (or 1/N * sum)\n"
                "   Derived from Maximum Likelihood Estimation under Gaussian noise assumption.\n\n"
                "2. Binary Cross-Entropy (Binary Classification):\n"
                "   L_BCE = - [ y * log(y_hat) + (1 - y) * log(1 - y_hat) ]\n\n"
                "3. Categorical Cross-Entropy (Multiclass Classification):\n"
                "   L_CCE = - sum_{k=1}^K y_k * log(y_hat_k)\n"
                "   Pairing Softmax with Cross-Entropy prevents gradient saturation during backpropagation because derivative dL/dz = y_hat - y."
            ),
            "exam_important_points": [
                "Formulas and graphical sketches for Sigmoid, Tanh, ReLU, and Softmax along with their derivatives.",
                "Why ReLU accelerates deep network convergence compared to Sigmoid.",
                "Mathematical proof that d/dz [sigma(z)] = sigma(z) * (1 - sigma(z)).",
                "Derivation of the softmax derivative and why cross-entropy is the natural loss partner."
            ]
        },
        "teaching_script": {
            "00-05": "Connect activation functions to biological firing thresholds and discuss why non-linearity is vital.",
            "05-20": "Plot Sigmoid and Tanh side-by-side. Derive the sigmoid derivative on blackboard step-by-step. Show why the maximum gradient is 0.25.",
            "20-30": "Introduce ReLU and discuss the revolution it caused in deep learning (AlexNet 2012). Explain the 'Dying ReLU' phenomenon.",
            "30-40": "Present Softmax as the normalized exponential function. Walk through numerical conversion of raw logits to probabilities.",
            "40-50": "Formulate Empirical Risk Minimization (ERM) vs Structural Risk Minimization (SRM). Compare MSE vs Cross-Entropy.",
            "50-60": "Interactive Q&A on why MSE is suboptimal for classification (non-convex loss surface with sigmoid). Summary."
        },
        "python_demo": {
            "title": "Visualization of Activation Functions and Their Derivatives",
            "description": "NumPy and Matplotlib plots comparing Sigmoid, Tanh, ReLU, and Leaky ReLU with derivative curves.",
            "code": """import numpy as np

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
print("\\nSigmoid Gradient at z =", z_vals)
print("Gradients:", sigmoid_grad(z_vals))
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Softmax Probability and Cross-Entropy Loss Calculation",
                "statement": "A deep neural network outputs logits z = [2.0, 1.0, -1.0]^T for a 3-class classification problem. The true class label is Class 1 (one-hot vector y = [1, 0, 0]^T). (a) Calculate the Softmax probability distribution, and (b) Calculate the Categorical Cross-Entropy loss.",
                "given": "z = [2.0, 1.0, -1.0]^T, true label y = [1, 0, 0]^T",
                "formulas": "softmax(z_i) = e^(z_i) / sum(e^z); L = - sum(y_i * ln(p_i))",
                "steps": [
                    "Step 1: Compute exponentials: e^2.0 = 7.3891, e^1.0 = 2.7183, e^(-1.0) = 0.3679.",
                    "Step 2: Compute denominator sum = 7.3891 + 2.7183 + 0.3679 = 10.4753.",
                    "Step 3: Compute probabilities:\n  p1 = 7.3891 / 10.4753 = 0.7054 (70.54%)\n  p2 = 2.7183 / 10.4753 = 0.2595 (25.95%)\n  p3 = 0.3679 / 10.4753 = 0.0351 (3.51%).",
                    "Step 4: Compute Cross-Entropy Loss: L = - [1 * ln(0.7054) + 0 + 0] = - (-0.3490) = 0.3490."
                ],
                "answer": "Softmax probabilities = [0.7054, 0.2595, 0.0351]^T; Loss L = 0.3490",
                "interpretation": "The model assigns the highest probability (70.54%) to the ground truth class, resulting in a moderate cross-entropy loss of 0.3490."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Derivative Derivation & Vanishing Gradient Verification",
                "statement": "Given a 4-layer network where each hidden neuron uses the Sigmoid activation function. At each layer, the pre-activation magnitude |z| >= 4.0. (a) Compute the maximum possible value of the activation derivative at each layer, and (b) Calculate the upper bound on the gradient propagation factor across the 4 layers: product_{l=1}^4 sigma'(z_l). Contrast this with ReLU.",
                "given": "Number of layers L = 4, |z| >= 4.0, Sigmoid activation.",
                "formulas": "sigma'(z) = sigma(z) * (1 - sigma(z)); For |z| >= 4, sigma(4) = 1 / (1 + e^(-4)) = 0.9820, sigma'(4) = 0.9820 * (1 - 0.9820) = 0.0177.",
                "steps": [
                    "Step 1: In the best case (|z| = 0), sigma'(0) = 0.25. But for |z| >= 4.0, sigma'(z) <= 0.0177.",
                    "Step 2: Calculate product of gradients across 4 layers under Best-Case (z=0): (0.25)^4 = 0.003906.",
                    "Step 3: Calculate product of gradients under Saturated Case (|z| >= 4): (0.0177)^4 = 9.80 x 10^(-8).",
                    "Step 4: Comparison with ReLU: For ReLU, for any positive input z > 0, f'(z) = 1.0. Thus, product = (1.0)^4 = 1.0."
                ],
                "answer": "Saturated gradient product = 9.80 x 10^(-8) (vanishes to near zero). ReLU gradient product = 1.0 (retained).",
                "interpretation": "This illustrates mathematically why Sigmoid networks suffer from catastrophic vanishing gradients in deep architectures, whereas ReLU preserves gradient magnitude."
            }
        ],
        "real_world_application": {
            "title": "Autonomous Vehicle Traffic Sign Recognition",
            "domain": "Computer Vision & Edge Automotive Systems",
            "workflow": "Input camera frame -> Deep network extracts visual features -> Softmax layer outputs probability across 43 traffic sign categories (Speed limit, Stop, Yield) -> Cross-entropy loss trains system to minimize classification risk.",
            "benefit": "Softmax probabilities allow the automotive decision control unit to establish confidence thresholds (e.g. only brake if Stop Sign confidence > 98%)."
        },
        "viva_questions": [
            {"q": "What is the maximum value of the derivative of the standard logistic sigmoid?", "a": "0.25, occurring at z = 0."},
            {"q": "Why is Tanh generally preferred over Sigmoid in hidden layers?", "a": "Tanh is zero-centered (range -1 to 1), which prevents systematic directional bias in weight updates across successive layers."},
            {"q": "What is the 'Dying ReLU' problem?", "a": "When a neuron's pre-activation is consistently negative, its gradient becomes permanently 0, preventing the weights from ever updating again during training."},
            {"q": "How does Leaky ReLU solve the dying ReLU issue?", "a": "By introducing a small positive slope (e.g., 0.01*z) for negative inputs so gradients never become completely zero."},
            {"q": "Why is Softmax combined with Cross-Entropy instead of Mean Squared Error?", "a": "Cross-Entropy cancels out the exponential term in Softmax, yielding a clean linear error term (y_hat - y) that prevents gradient saturation during backpropagation."},
            {"q": "What is Empirical Risk Minimization?", "a": "An approximation of expected risk where model parameters are optimized to minimize the average loss on the empirical training set."},
            {"q": "What is the difference between ERM and SRM?", "a": "ERM only minimizes training sample loss, risking overfitting; SRM adds a regularization penalty term to penalize model complexity."},
            {"q": "What is the mathematical range of the Softmax function?", "a": "Each element lies in the open interval (0, 1), and all elements sum to exactly 1.0."},
            {"q": "What is temperature scaling in Softmax?", "a": "Dividing logits by a positive temperature parameter T before exponentiation: T > 1 softens the distribution, while T < 1 sharpens it toward argmax."},
            {"q": "Why can MSE cause slow learning when paired with Sigmoid output neurons?", "a": "Because dL/dz contains a sigma'(z) factor; when the neuron is strongly wrong, sigma'(z) -> 0, causing the gradient to vanish."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "The derivative of the hyperbolic tangent activation function tanh(z) in terms of tanh(z) is:",
                    "options": ["1 - tanh(z)", "1 - tanh^2(z)", "tanh(z)*(1 - tanh(z))", "1 + tanh^2(z)"],
                    "ans": "1 - tanh^2(z)"
                },
                {
                    "q": "Which activation function does NOT saturate for large positive inputs?",
                    "options": ["Sigmoid", "Tanh", "ReLU", "Arctan"],
                    "ans": "ReLU"
                },
                {
                    "q": "In Empirical Risk Minimization, if a model achieves zero empirical risk on training data, this implies:",
                    "options": ["It will definitely achieve zero test error", "It is guaranteed to be optimal", "It may be severely overfitting", "Generalization error is minimized"],
                    "ans": "It may be severely overfitting"
                }
            ],
            "short_questions": [
                "Derive the derivative of the Sigmoid function showing all intermediate algebraic steps.",
                "Compare Mean Squared Error and Categorical Cross-Entropy for a classification problem."
            ],
            "application_question": "Explain how temperature scaling in Softmax can be utilized in calibration of deep models for medical diagnosis uncertainty estimation.",
            "homework": "Prove that for Softmax output y_hat_i and Categorical Cross-Entropy Loss L, the partial derivative with respect to logit z_j is simply (y_hat_j - y_j)."
        }
    },
    {
        "day": 4,
        "module": 1,
        "topic_id": "1.4",
        "title": "Training MLPs with Backpropagation",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO1",
        "blooms": "Understand / Apply",
        "prerequisites": "Chain Rule of Multivariable Calculus, Vector and Matrix Calculus, Forward Propagation",
        "learning_objectives": [
            "Formulate the general Backpropagation algorithm based on the multivariable chain rule.",
            "Derive the error sensitivity term (delta) for output and hidden layer neurons.",
            "Formulate weight and bias gradient expressions in matrix notation.",
            "Understand computational complexity and memory requirements of backpropagation."
        ],
        "key_terminology": [
            "Backpropagation", "Chain Rule", "Error Sensitivity (Delta)", "Gradient Vector",
            "Jacobian Matrix", "Forward Pass", "Backward Pass"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 6 (Section 6.5); Michael Nielsen: Chapter 2.",
        "notes": {
            "introduction": "Backpropagation (Rumelhart, Hinton, Williams, 1986) is the foundational training algorithm for multi-layer neural networks. It efficiently computes the partial derivative of the scalar loss function L with respect to every weight and bias parameter in the network by applying the chain rule in reverse order.",
            "mathematical_derivation": (
                "Consider layer l with activation a^(l) = sigma(z^(l)), where z^(l) = W^(l) * a^(l-1) + b^(l).\n\n"
                "Define the Error Sensitivity (Delta) vector for layer l:\n"
                "delta^(l) = del L / del z^(l)\n\n"
                "1. Fundamental Equation 1 (Output Layer Delta):\n"
                "   delta^(L) = (del L / del a^(L)) elementwise_prod sigma'(z^(L))\n"
                "   For Cross-Entropy Loss with Softmax: delta^(L) = y_hat - y\n\n"
                "2. Fundamental Equation 2 (Hidden Layer Delta Backpropagation):\n"
                "   delta^(l) = [ (W^(l+1))^T * delta^(l+1) ] elementwise_prod sigma'(z^(l))\n"
                "   This equation propagates error signals backward from layer l+1 to layer l.\n\n"
                "3. Fundamental Equation 3 (Rate of change of Loss with respect to Bias):\n"
                "   del L / del b^(l) = delta^(l)\n\n"
                "4. Fundamental Equation 4 (Rate of change of Loss with respect to Weights):\n"
                "   del L / del W^(l) = delta^(l) * (a^(l-1))^T\n\n"
                "Parameter Update Rule:\n"
                "W^(l) <- W^(l) - eta * (del L / del W^(l))\n"
                "b^(l) <- b^(l) - eta * (del L / del b^(l))"
            ),
            "computational_efficiency": (
                "Naive numerical gradient estimation (perturbing each weight by epsilon) requires O(P) forward passes, leading to O(P^2) time complexity for P parameters.\n"
                "Backpropagation evaluates all P gradients in a single backward pass with computational cost roughly equivalent to 2 forward passes: O(P) complexity!"
            ),
            "exam_important_points": [
                "State and derive the four fundamental equations of backpropagation.",
                "Define the error vector delta^(l) and explain its physical significance.",
                "Explain why backpropagation runs backward rather than forward.",
                "Matrix dimensional consistency across all weight gradient equations."
            ]
        },
        "teaching_script": {
            "00-05": "Connect forward propagation (feature inference) to backward propagation (credit assignment for errors).",
            "05-20": "Introduce multivariable chain rule on a simple 3-node computational graph. Define delta = dL/dz.",
            "20-35": "Derive Output Layer Delta: delta^(L) = (dL/da^(L)) * sigma'(z^(L)). Show how MSE vs Cross-Entropy impacts this.",
            "35-45": "Derive Hidden Layer Delta: delta^(l) = (W^(l+1))^T * delta^(l+1) odot sigma'(z^(l)). Emphasize transpose of weights.",
            "45-55": "Derive weight gradient: dL/dW^(l) = delta^(l) * (a^(l-1))^T. Check matrix dimensions for sanity.",
            "55-60": "Summarize the 4 equations. Inform students that Day 5 will be an end-to-end numerical walkthrough."
        },
        "python_demo": {
            "title": "Minimal NumPy Implementation of 2-Layer Backpropagation",
            "description": "Clean implementation of forward and backward pass computing exact analytical gradients for a 2-layer MLP.",
            "code": """import numpy as np

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

print("\\n--- COMPUTED GRADIENTS ---")
print("dW2:\\n", dW2)
print("db2:", db2)
print("dW1:\\n", dW1)
print("db1:", db1)
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Output Layer Error Delta Calculation",
                "statement": "An output neuron with Sigmoid activation has pre-activation z = 1.5, giving output a = 0.8176. The ground truth target is y = 1.0. If the loss function is Squared Error L = 0.5 * (a - y)^2, calculate: (a) Error derivative dL/da, (b) Activation derivative da/dz, and (c) Error sensitivity delta.",
                "given": "z = 1.5, a = 0.8176, y = 1.0, L = 0.5*(a - y)^2",
                "formulas": "dL/da = a - y; da/dz = a*(1 - a); delta = (dL/da) * (da/dz)",
                "steps": [
                    "Step 1: dL/da = 0.8176 - 1.0 = -0.1824.",
                    "Step 2: da/dz = 0.8176 * (1.0 - 0.8176) = 0.8176 * 0.1824 = 0.1491.",
                    "Step 3: delta = (-0.1824) * 0.1491 = -0.02720."
                ],
                "answer": "dL/da = -0.1824, da/dz = 0.1491, delta = -0.0272",
                "interpretation": "The negative delta indicates that increasing the pre-activation z will decrease the loss, which guides the gradient update in the proper direction."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Hidden Layer Weight Gradient via Error Backpropagation",
                "statement": "In a 2-layer network, hidden neuron h1 has activation a1 = 0.6. It is connected to output neuron y with weight w_out = 0.8. The output neuron has computed error delta_out = -0.15. The hidden neuron h1 receives an input x1 = 0.5 through weight w_in, and its activation function is Sigmoid. Calculate: (a) The backpropagated error delta_h1, and (b) The gradient dL/dw_in.",
                "given": "a1 = 0.6, w_out = 0.8, delta_out = -0.15, x1 = 0.5, hidden activation = Sigmoid.",
                "formulas": "delta_h = (w_out * delta_out) * a_h * (1 - a_h); dL/dw_in = delta_h * x1",
                "steps": [
                    "Step 1: Compute activation derivative of hidden neuron: sigma'(z1) = a1 * (1 - a1) = 0.6 * (1 - 0.6) = 0.24.",
                    "Step 2: Backpropagate error through w_out: backprop_sum = w_out * delta_out = 0.8 * (-0.15) = -0.12.",
                    "Step 3: Compute hidden delta: delta_h1 = backprop_sum * sigma'(z1) = -0.12 * 0.24 = -0.0288.",
                    "Step 4: Compute gradient with respect to input weight: dL/dw_in = delta_h1 * x1 = -0.0288 * 0.5 = -0.0144."
                ],
                "answer": "Hidden delta_h1 = -0.0288, Gradient dL/dw_in = -0.0144",
                "interpretation": "The negative gradient means increasing w_in will decrease the overall network error. The weight update rule will increment w_in by eta * 0.0144."
            }
        ],
        "real_world_application": {
            "title": "Acoustic Noise Cancellation in Aviation Communication",
            "domain": "Digital Signal Processing & Real-time Acoustics",
            "workflow": "Cockpit ambient microphone picks up jet engine roar -> Filtered through deep MLP -> Output produces anti-phase acoustic cancellation wave -> Error microphone measures residual decibel leak -> Backpropagation updates weights at 100 Hz.",
            "benefit": "Enables crystal-clear air traffic controller communication in high-decibel turbofan cockpit environments."
        },
        "viva_questions": [
            {"q": "Who popularized the Backpropagation algorithm for neural networks in 1986?", "a": "David Rumelhart, Geoffrey Hinton, and Ronald Williams."},
            {"q": "What is the primary mathematical principle underlying backpropagation?", "a": "The chain rule of multivariable calculus applied in reverse topological order."},
            {"q": "What does the term delta^(l) represent?", "a": "The partial derivative of the scalar loss with respect to the pre-activation vector z of layer l: del L / del z^(l)."},
            {"q": "Why is caching intermediate activations during the forward pass necessary?", "a": "Because weight gradients depend directly on the input activation from the preceding layer: dL/dW^(l) = delta^(l) * (a^(l-1))^T."},
            {"q": "What is the time complexity of backpropagation compared to finite difference numerical gradient calculation?", "a": "Backpropagation is O(P) with respect to parameter count P, whereas finite differences require O(P^2) or O(P) full forward passes."},
            {"q": "What is the difference between batch, stochastic, and mini-batch backpropagation?", "a": "Batch processes the entire dataset per update; Stochastic updates after each single sample; Mini-batch computes gradients over a small subset (e.g. 32-256 samples)."},
            {"q": "Why does delta^(l) require the transpose of the weight matrix W^(l+1)?", "a": "To invert the direction of dimensional mapping and project the error signals backward from layer l+1 to layer l."},
            {"q": "What happens to backpropagation if all weights are initialized to the exact same value?", "a": "All hidden units receive identical error signals and compute identical gradients, maintaining perfect symmetry and preventing them from learning distinct features."},
            {"q": "Is backpropagation an optimization algorithm by itself?", "a": "No, backpropagation only computes the partial derivatives (gradients). An optimization algorithm (like SGD or Adam) uses these gradients to update parameters."},
            {"q": "How does cross-entropy loss simplify the output delta equation when paired with Softmax?", "a": "The derivative sigma'(z) cancels out algebraically, yielding the simple difference vector: delta = y_hat - y."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "The backpropagated error sensitivity delta^(l) in a hidden layer is defined as:",
                    "options": ["del L / del W^(l)", "del L / del a^(l)", "del L / del z^(l)", "del z^(l) / del a^(l-1)"],
                    "ans": "del L / del z^(l)"
                },
                {
                    "q": "The gradient of the loss with respect to weight matrix W^(l) is given by:",
                    "options": ["delta^(l) * (a^(l-1))^T", "(delta^(l))^T * a^(l-1)", "W^(l) * delta^(l)", "delta^(l) * (z^(l))^T"],
                    "ans": "delta^(l) * (a^(l-1))^T"
                },
                {
                    "q": "If an output neuron has activation a = 0.7, target y = 1.0, and loss L = 0.5*(a-y)^2, then dL/da is:",
                    "options": ["-0.3", "+0.3", "-0.7", "+0.49"],
                    "ans": "-0.3"
                }
            ],
            "short_questions": [
                "Write down the four fundamental equations of Backpropagation and identify the dimensions of all vectors and matrices.",
                "Explain why weight symmetry breaking is an essential prerequisite for backpropagation."
            ],
            "application_question": "Explain how the memory footprint during backpropagation scales with model depth and sequence length in embedded edge devices.",
            "homework": "Prove that for a single output neuron with logistic sigmoid activation and binary cross-entropy loss, the output error delta is exactly (y_hat - y)."
        }
    },
    {
        "day": 5,
        "module": 1,
        "topic_id": "1.5",
        "title": "Illustration of the Backpropagation Algorithm",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO1",
        "blooms": "Apply / Evaluate",
        "prerequisites": "Backpropagation Equations, Matrix Calculus, Day 4 concepts",
        "learning_objectives": [
            "Perform an end-to-end manual step-by-step numerical calculation of backpropagation on a 2-2-1 network.",
            "Compute exact forward pass activations, layer-by-layer pre-activations, and loss values.",
            "Calculate output delta and backpropagate error deltas to hidden layer neurons.",
            "Execute weight and bias updates and prove that total loss strictly decreases."
        ],
        "key_terminology": [
            "End-to-End Walkthrough", "Forward Evaluation", "Backward Sensitivity", "Gradient Update",
            "Loss Reduction Verification", "Learning Step"
        ],
        "textbook_ref": "Charu C. Aggarwal: Neural Networks and Deep Learning, Chapter 1; KTU CST414 Syllabus Model Question Paper Part B (Q11).",
        "notes": {
            "network_specification": (
                "Standard 2-2-1 Fully Connected MLP Architecture:\n"
                "Input vector: x = [x1, x2]^T = [0.05, 0.10]^T\n"
                "Target output: y = 0.01 (or target vector for 2 outputs)\n"
                "Initial weights from Input to Hidden layer:\n"
                "  V = [[v11, v12], [v21, v22]] = [[0.15, 0.20], [0.25, 0.30]]\n"
                "  Hidden biases: b1 = [0.35, 0.35]^T\n"
                "Weights from Hidden to Output layer:\n"
                "  W = [w1, w2] = [0.40, 0.45]\n"
                "  Output bias: b2 = 0.60\n"
                "Activation function: Logistic Sigmoid everywhere: sigma(z) = 1 / (1 + e^(-z))\n"
                "Loss Function: Squared Error L = 0.5 * (y_hat - y)^2\n"
                "Learning rate: eta = 0.5"
            ),
            "step_by_step_execution_phases": (
                "Phase 1: Forward Pass\n"
                "1. Hidden Neuron 1:\n"
                "   z_h1 = (0.15 * 0.05) + (0.20 * 0.10) + 0.35 = 0.0075 + 0.020 + 0.35 = 0.3775\n"
                "   a_h1 = sigma(0.3775) = 1 / (1 + e^(-0.3775)) = 0.59327\n"
                "2. Hidden Neuron 2:\n"
                "   z_h2 = (0.25 * 0.05) + (0.30 * 0.10) + 0.35 = 0.0125 + 0.030 + 0.35 = 0.3925\n"
                "   a_h2 = sigma(0.3925) = 1 / (1 + e^(-0.3925)) = 0.59688\n"
                "3. Output Neuron:\n"
                "   z_o = (0.40 * 0.59327) + (0.45 * 0.59688) + 0.60 = 0.23731 + 0.26860 + 0.60 = 1.10591\n"
                "   a_o = sigma(1.10591) = 1 / (1 + e^(-1.10591)) = 0.75137\n"
                "4. Loss Calculation:\n"
                "   L = 0.5 * (0.75137 - 0.01)^2 = 0.5 * (0.74137)^2 = 0.27481\n\n"
                "Phase 2: Backward Pass\n"
                "1. Output Error Delta:\n"
                "   dL/da_o = a_o - y = 0.75137 - 0.01 = 0.74137\n"
                "   da_o/dz_o = a_o * (1 - a_o) = 0.75137 * (1 - 0.75137) = 0.18681\n"
                "   delta_o = (dL/da_o) * (da_o/dz_o) = 0.74137 * 0.18681 = 0.13849\n"
                "2. Output Weight Gradients:\n"
                "   dL/dw1 = delta_o * a_h1 = 0.13849 * 0.59327 = 0.08216\n"
                "   dL/dw2 = delta_o * a_h2 = 0.13849 * 0.59688 = 0.08266\n"
                "   dL/db2 = delta_o = 0.13849\n"
                "3. Hidden Neuron Deltas:\n"
                "   delta_h1 = (delta_o * w1) * a_h1 * (1 - a_h1)\n"
                "            = (0.13849 * 0.40) * 0.59327 * (1 - 0.59327)\n"
                "            = 0.05540 * 0.24130 = 0.01337\n"
                "   delta_h2 = (delta_o * w2) * a_h2 * (1 - a_h2)\n"
                "            = (0.13849 * 0.45) * 0.59688 * (1 - 0.59688)\n"
                "            = 0.06232 * 0.24061 = 0.01499\n"
                "4. Input-to-Hidden Weight Gradients:\n"
                "   dL/dv11 = delta_h1 * x1 = 0.01337 * 0.05 = 0.0006685\n"
                "   dL/dv12 = delta_h1 * x2 = 0.01337 * 0.10 = 0.001337\n\n"
                "Phase 3: Parameter Updates (eta = 0.5)\n"
                "w1_new = 0.40 - (0.5 * 0.08216) = 0.35892\n"
                "w2_new = 0.45 - (0.5 * 0.08266) = 0.40867\n"
                "v11_new = 0.15 - (0.5 * 0.0006685) = 0.14967"
            ),
            "exam_important_points": [
                "Full 14-mark KTU examination standard problem solving template.",
                "Clear notation distinguishing layer indices and neuron indices.",
                "Retention of at least 4 decimal places throughout all calculations.",
                "Demonstration of post-update forward pass verifying loss reduction."
            ]
        },
        "teaching_script": {
            "00-05": "Distribute numerical template. Set up network diagram with given weights on blackboard.",
            "05-20": "Execute Forward Pass: Compute z_h1, a_h1, z_h2, a_h2, z_o, a_o, and Loss. Have students calculate alongside.",
            "20-35": "Execute Backward Pass at Output: Calculate delta_o, dL/dw1, dL/dw2, and dL/db2.",
            "35-45": "Execute Backward Pass at Hidden Layer: Calculate delta_h1, delta_h2, and input-to-hidden weight gradients.",
            "45-55": "Apply learning rate eta = 0.5 to update parameters. Compute new loss to prove error decreased.",
            "55-60": "Highlight common examination mistakes (forgetting activation derivative, misplacing transpose). Wrap up."
        },
        "python_demo": {
            "title": "Exact Verification of 2-2-1 Backpropagation Step",
            "description": "Python program implementing the exact numerical problem to verify all manual calculations down to 6 decimal places.",
            "code": """import numpy as np

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
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Single Step Weight Update with Learning Rate",
                "statement": "An output layer weight w has current value 0.45. The computed gradient of the loss function with respect to w is dL/dw = 0.08266. If the learning rate is eta = 0.5, calculate: (a) The parameter update step delta_w, and (b) The new updated weight w_new.",
                "given": "w_current = 0.45, dL/dw = 0.08266, eta = 0.5",
                "formulas": "delta_w = - eta * (dL/dw); w_new = w_current + delta_w",
                "steps": [
                    "Step 1: Compute parameter update step: delta_w = - 0.5 * 0.08266 = -0.04133.",
                    "Step 2: Compute updated weight: w_new = 0.45 + (-0.04133) = 0.40867."
                ],
                "answer": "delta_w = -0.04133, w_new = 0.40867",
                "interpretation": "The positive gradient signifies that increasing w increases loss; hence, the gradient descent step reduces w by 0.04133."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "KTU Model Paper Q11: Parameter Update in MLP using ReLU",
                "statement": "Update the parameters in the given MLP using gradient descent with learning rate eta = 0.5 and activation function as ReLU. Input is x = [1.0, 1.0]^T, target y = 1.0. Initial weights: Hidden layer V = [[0.1, 0.2], [0.1, 0.1]] with hidden bias b1 = [0.0, 0.0]^T. Output layer W = [0.1, 0.1] with output bias b2 = 0.0. Loss function is L = 0.5 * (y_hat - y)^2. Perform one full step of gradient descent.",
                "given": "x = [1.0, 1.0]^T, y = 1.0, eta = 0.5, V = [[0.1, 0.2], [0.1, 0.1]], b1=[0,0]^T, W = [0.1, 0.1], b2 = 0.0. ReLU activation everywhere.",
                "formulas": "z1 = V*x + b1; a1 = max(0, z1); zo = W*a1 + b2; y_hat = max(0, zo); dL/da = y_hat - y; delta_o = dL/da * (1 if zo > 0 else 0)",
                "steps": [
                    "Step 1: Hidden layer forward pass:\n  z1_1 = 0.1*1.0 + 0.2*1.0 = 0.3 -> a1_1 = max(0, 0.3) = 0.3\n  z1_2 = 0.1*1.0 + 0.1*1.0 = 0.2 -> a1_2 = max(0, 0.2) = 0.2.",
                    "Step 2: Output layer forward pass:\n  zo = 0.1*0.3 + 0.1*0.2 + 0.0 = 0.03 + 0.02 = 0.05.\n  y_hat = max(0, 0.05) = 0.05.",
                    "Step 3: Loss calculation:\n  L = 0.5 * (0.05 - 1.0)^2 = 0.5 * (-0.95)^2 = 0.45125.",
                    "Step 4: Output layer gradients:\n  dL/dy_hat = 0.05 - 1.0 = -0.95.\n  dy_hat/dzo = 1.0 (since zo = 0.05 > 0).\n  delta_o = -0.95 * 1.0 = -0.95.\n  dL/dW1 = delta_o * a1_1 = -0.95 * 0.3 = -0.285.\n  dL/dW2 = delta_o * a1_2 = -0.95 * 0.2 = -0.190.",
                    "Step 5: Hidden layer gradients:\n  Since z1_1, z1_2 > 0, ReLU derivatives are 1.0.\n  delta_h1 = (delta_o * W1) * 1.0 = (-0.95 * 0.1) * 1.0 = -0.095.\n  delta_h2 = (delta_o * W2) * 1.0 = (-0.95 * 0.1) * 1.0 = -0.095.\n  dL/dV11 = delta_h1 * x1 = -0.095 * 1.0 = -0.095.\n  dL/dV12 = delta_h1 * x2 = -0.095 * 1.0 = -0.095.\n  dL/dV21 = delta_h2 * x1 = -0.095 * 1.0 = -0.095.\n  dL/dV22 = delta_h2 * x2 = -0.095 * 1.0 = -0.095.",
                    "Step 6: Parameter Updates (eta = 0.5):\n  W1_new = 0.1 - 0.5 * (-0.285) = 0.1 + 0.1425 = 0.2425.\n  W2_new = 0.1 - 0.5 * (-0.190) = 0.1 + 0.0950 = 0.1950.\n  V11_new = 0.1 - 0.5 * (-0.095) = 0.1 + 0.0475 = 0.1475.\n  V12_new = 0.2 - 0.5 * (-0.095) = 0.2 + 0.0475 = 0.2475.\n  V21_new = 0.1 - 0.5 * (-0.095) = 0.1 + 0.0475 = 0.1475.\n  V22_new = 0.1 - 0.5 * (-0.095) = 0.1 + 0.0475 = 0.1475."
                ],
                "answer": "Updated W = [0.2425, 0.1950], Updated V = [[0.1475, 0.2475], [0.1475, 0.1475]]",
                "interpretation": "Because the model severely underpredicted the target (y_hat=0.05 vs y=1.0), all weight gradients are negative, leading to significant positive weight increments to boost future predictions."
            }
        ],
        "real_world_application": {
            "title": "Real-Time Battery State-of-Charge (SoC) Calibration in EVs",
            "domain": "Electric Vehicles & Battery Management Systems (BMS)",
            "workflow": "BMS monitors cell voltage, discharge current, and ambient temperature -> 2-hidden-layer MLP predicts battery State-of-Charge -> At charging stations, precise open-circuit voltage calibration triggers single-step backpropagation on-board microchip.",
            "benefit": "Maintains accurate driving range estimation over 10-year battery degradation curves without requiring cloud data transfer."
        },
        "viva_questions": [
            {"q": "What is the primary benefit of working through backpropagation manually?", "a": "It solidifies understanding of chain rule mechanics, tensor dimensions, and why error signals scale across layers."},
            {"q": "How does learning rate magnitude impact the weight update step?", "a": "A high learning rate creates large updates that can overshoot the minimum and diverge; a low rate causes slow, computationally prohibitive convergence."},
            {"q": "What guarantees that a backpropagation gradient update reduces the loss?", "a": "For an infinitesimally small learning rate eta, the update moves in the negative gradient direction (direction of steepest descent), guaranteeing non-increasing loss on smooth surfaces."},
            {"q": "What happens to ReLU gradients when pre-activation z is exactly 0?", "a": "The mathematical derivative is technically undefined, but in practical deep learning frameworks it is defined via sub-gradient as either 0.0 or 1.0 (typically 0.0)."},
            {"q": "Why are bias gradients simply equal to the layer delta?", "a": "Because the derivative of pre-activation z with respect to bias b is identically 1 (dz/db = 1)."},
            {"q": "How does the choice of loss function affect the output delta?", "a": "The loss function determines dL/da; MSE yields (a - y), while BCE paired with sigmoid simplifies directly to (a - y) without the sigmoid derivative."},
            {"q": "If a weight has a negative gradient dL/dw < 0, does the weight increase or decrease?", "a": "The weight increases: w_new = w - eta * (negative_val) = w + positive_step."},
            {"q": "Why is caching z^(l) and a^(l) mandatory during the forward pass?", "a": "Because evaluating sigma'(z^(l)) requires z^(l) (or a^(l)), and evaluating dL/dW^(l) requires a^(l-1)."},
            {"q": "What is the computational overhead of the backward pass relative to the forward pass?", "a": "The backward pass requires roughly twice the number of floating point operations (FLOPs) as the forward pass due to computing both parameter gradients and activation gradients."},
            {"q": "What is the difference between batch gradient descent and online backpropagation?", "a": "Online updates weights after every single sample; batch sums gradients across all N dataset samples before applying one parameter update."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "If dL/dw = +0.25 and learning rate eta = 0.2, the new weight w_new compared to w_old will:",
                    "options": ["Increase by 0.05", "Decrease by 0.05", "Increase by 0.25", "Decrease by 0.25"],
                    "ans": "Decrease by 0.05"
                },
                {
                    "q": "In the backward pass of an MLP, error deltas flow:",
                    "options": ["From input layer toward output layer", "From output layer toward input layer", "Only within hidden layers", "Randomly between neurons"],
                    "ans": "From output layer toward input layer"
                },
                {
                    "q": "The gradient of loss with respect to a bias term del L / del b^(l) is identical to:",
                    "options": ["The activation vector a^(l)", "The error sensitivity vector delta^(l)", "The weight matrix W^(l)", "The loss scalar L"],
                    "ans": "The error sensitivity vector delta^(l)"
                }
            ],
            "short_questions": [
                "Explain the significance of the sub-gradient convention when calculating backpropagation through ReLU at z = 0.",
                "Describe why the backward pass takes approximately twice as many floating-point operations as the forward pass."
            ],
            "application_question": "Suppose an edge IoT device has limited SRAM memory. Propose an activation checkpointing strategy to run backpropagation without storing all intermediate layer activations.",
            "homework": "Using the 2-2-1 network specified in Day 5 notes, calculate the updated output bias b2 after one step of gradient descent."
        }
    },
    {
        "day": 6,
        "module": 1,
        "topic_id": "1.6",
        "title": "Practical Issues in Training: Overfitting, Vanishing and Exploding Gradients",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO1",
        "blooms": "Understand / Analyze",
        "prerequisites": "Backpropagation Algorithm, Linear Algebra (eigenvalues), Optimization basics",
        "learning_objectives": [
            "Analyze the phenomenon of overfitting in deep networks (bias-variance tradeoff).",
            "Derive the mathematical mechanics of vanishing and exploding gradients across L layers.",
            "Examine the influence of saturating activation functions and uncalibrated weight initialization.",
            "Explore mitigation strategies: gradient clipping, ReLU family, and residual connections."
        ],
        "key_terminology": [
            "Overfitting", "Generalization Gap", "Vanishing Gradients", "Exploding Gradients",
            "Product of Jacobians", "Gradient Clipping", "Weight Initialization"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 5 & 8; Aggarwal: Chapter 3.",
        "notes": {
            "overfitting": (
                "Overfitting occurs when a neural network memorizes training data noise and idiosyncratic artifacts rather than learning generalizable underlying patterns.\n\n"
                "Key Indicators:\n"
                "- Training Loss continues decreasing monotonically toward zero.\n"
                "- Validation / Test Loss reaches an inflection point and begins increasing steadily.\n"
                "- Large generalization gap: Loss_val - Loss_train >> 0.\n\n"
                "Fundamental Causes:\n"
                "1. Excessive model capacity (too many parameters relative to dataset size).\n"
                "2. Insufficient training samples or unrepresentative data distribution.\n"
                "3. Training for excessive epochs without early stopping."
            ),
            "vanishing_and_exploding_gradients": (
                "Consider a deep network with L layers. By the chain rule, the gradient of the loss with respect to activation at early layer 1 is:\n"
                "del L / del a^(1) = (del L / del a^(L)) * product_{k=1}^{L-1} (del a^(k+1) / del a^(k))\n\n"
                "Each Jacobian matrix is:\n"
                "del a^(k+1) / del a^(k) = W^(k+1) * diag(sigma'(z^(k+1)))\n\n"
                "Case 1: Vanishing Gradients\n"
                "If the largest singular values of weight matrices ||W^(k)|| < 1 and saturating activations have sigma'(z) <= 0.25 (e.g. Sigmoid):\n"
                "|| del L / del a^(1) || <= || del L / del a^(L) || * product_{k=1}^{L-1} (0.25 * ||W^(k)||) -> 0 as L increases.\n"
                "Gradients at early layers shrink exponentially, stalling learning in shallow representations.\n\n"
                "Case 2: Exploding Gradients\n"
                "If weight matrices have spectral radius > 1 (e.g. weights initialized too large):\n"
                "|| del L / del a^(1) || grows exponentially as (lambda_max)^L -> infinity.\n"
                "Results in numerical overflow (NaN / Inf loss), unstable oscillations, and ruined weights."
            ),
            "mitigation_strategies": (
                "1. Activation Choice: Replace Sigmoid/Tanh with non-saturating activations (ReLU, Leaky ReLU, ELU).\n"
                "2. Proper Initialization: Xavier/Glorot (1990) for Tanh; He/Kaiming (2015) for ReLU.\n"
                "3. Gradient Clipping: Rescale gradient vector g if ||g|| exceeds threshold c:\n"
                "   g_clipped = g * (c / ||g||) if ||g|| > c else g.\n"
                "4. Normalization: Batch Normalization (Ioffe & Szegedy, 2015) and Layer Normalization.\n"
                "5. Architectural: Residual skip connections (He et al., 2015) creating identity gradient highways."
            ),
            "exam_important_points": [
                "Mathematical chain rule derivation of vanishing gradient across L layers.",
                "Comparison of training vs validation loss curves indicating overfitting vs underfitting.",
                "Formula and geometric interpretation of Gradient Clipping by norm.",
                "How spectral radius of weight matrices dictates gradient explosion or decay."
            ]
        },
        "teaching_script": {
            "00-05": "Show plot of training vs validation loss. Ask students: 'Why does test error rise even when training error drops?'",
            "05-20": "Explain Overfitting and model capacity. Discuss bias-variance trade-off in deep architectures.",
            "20-35": "Derive the product-of-Jacobians formula for backpropagation through L layers. Prove exponential decay when factors < 1.",
            "35-45": "Demonstrate Exploding Gradients. Introduce the Gradient Clipping equation: g <- c * g / ||g||.",
            "45-55": "Interactive Python demo: show gradients decaying to zero across a 10-layer Sigmoid network vs ReLU network.",
            "55-60": "Recap and link to Day 7: Convergence Difficulties and Optimization Landscapes."
        },
        "python_demo": {
            "title": "Simulating Vanishing Gradients Across Deep Layers",
            "description": "Simulation measuring gradient norm backpropagating through a 10-layer MLP using Sigmoid vs ReLU activations.",
            "code": """import numpy as np

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
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Gradient Clipping by Norm",
                "statement": "During backpropagation of a deep network, the computed gradient vector across all parameters is g = [3.0, -4.0, 12.0]^T. The maximum permissible gradient clipping threshold is c = 5.0. (a) Calculate the L2 norm of the gradient vector ||g||_2, (b) Determine whether clipping is required, and (c) Calculate the clipped gradient vector g_clipped.",
                "given": "g = [3.0, -4.0, 12.0]^T, clipping threshold c = 5.0",
                "formulas": "||g||_2 = sqrt(sum(g_i^2)); if ||g|| > c then g_clipped = g * (c / ||g||_2) else g",
                "steps": [
                    "Step 1: Compute L2 norm: ||g||_2 = sqrt(3^2 + (-4)^2 + 12^2) = sqrt(9 + 16 + 144) = sqrt(169) = 13.0.",
                    "Step 2: Check clipping condition: Since ||g||_2 = 13.0 > c = 5.0, gradient clipping MUST be applied.",
                    "Step 3: Compute scaling factor: factor = c / ||g||_2 = 5.0 / 13.0 = 0.384615.",
                    "Step 4: Compute clipped components:\n  g1_clipped = 3.0 * (5/13) = 15/13 = 1.1538\n  g2_clipped = -4.0 * (5/13) = -20/13 = -1.5385\n  g3_clipped = 12.0 * (5/13) = 60/13 = 4.6154.",
                    "Step 5: Verify new norm: sqrt(1.1538^2 + (-1.5385)^2 + 4.6154^2) = sqrt(1.3312 + 2.3670 + 21.3019) = sqrt(25.0) = 5.0."
                ],
                "answer": "||g|| = 13.0, Clipping required. Clipped gradient = [1.1538, -1.5385, 4.6154]^T (Norm = 5.0)",
                "interpretation": "Clipping rescales the gradient vector to have a maximum length of 5.0 while strictly preserving its original descent direction."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Mathematical Estimation of Vanishing Gradient Decay Rate",
                "statement": "A deep feedforward network has 8 layers. Each layer uses the standard sigmoid activation function where average pre-activation yields sigma'(z) = 0.20. The weight matrices at each layer have average spectral norm ||W||_2 = 1.5. (a) Derive the contraction factor per layer, (b) Calculate the total gradient attenuation factor from layer 8 back to layer 1, and (c) If the output error gradient is ||dL/da8|| = 2.0, determine the gradient magnitude arriving at layer 1.",
                "given": "L = 8 layers, sigma'(z) = 0.20, ||W||_2 = 1.5, initial ||dL/da8|| = 2.0.",
                "formulas": "Per-layer contraction factor gamma = ||W||_2 * sigma'(z); Total attenuation = gamma^(L-1); Magnitude at layer 1 = ||dL/da8|| * gamma^(L-1)",
                "steps": [
                    "Step 1: Compute per-layer contraction factor: gamma = 1.5 * 0.20 = 0.30.",
                    "Step 2: Number of inter-layer transitions across 8 layers = L - 1 = 7.",
                    "Step 3: Total attenuation factor = (0.30)^7 = 0.0002187 (or 2.187 x 10^(-4)).",
                    "Step 4: Compute gradient magnitude at layer 1: ||dL/da1|| = 2.0 * (0.30)^7 = 2.0 * 0.0002187 = 0.0004374 (or 4.374 x 10^(-4))."
                ],
                "answer": "Per-layer factor = 0.30, Total attenuation = 2.187 x 10^(-4), Gradient at layer 1 = 0.0004374",
                "interpretation": "The gradient signal is attenuated by over 99.97% before reaching the first hidden layer, meaning early layer weights learn thousands of times slower than output weights."
            }
        ],
        "real_world_application": {
            "title": "Clinical Mortality Prediction in Intensive Care Units (ICU)",
            "domain": "Healthcare Informatics & Predictive Medicine",
            "workflow": "High-frequency ICU patient telemetry (ECG, SpO2, blood pressure, arterial blood gases) fed into an 8-layer deep network -> Overfitting prevented using early stopping based on validation cohort -> Gradient clipping prevents erratic updates during sudden patient physiological anomalies.",
            "benefit": "Maintains reliable 24-hour mortality risk scores without overfitting to sparse hospital training records."
        },
        "viva_questions": [
            {"q": "What is the key indicator of overfitting when observing learning curves?", "a": "Training loss continues to decrease while validation loss begins to increase steadily."},
            {"q": "Mathematically, what causes the vanishing gradient problem?", "a": "The repeated multiplication of Jacobian matrices with spectral norms < 1 and activation derivatives bounded below 1 (e.g. 0.25 for sigmoid) across multiple layers."},
            {"q": "What is the primary danger of exploding gradients?", "a": "They cause parameter updates so massive that weights overflow to NaN or infinity, completely destabilizing the network."},
            {"q": "How does gradient clipping by value differ from gradient clipping by norm?", "a": "Clipping by value truncates each individual gradient element independently between [-c, c], which changes the gradient vector direction. Clipping by norm rescales the entire vector by c / ||g||, preserving its direction."},
            {"q": "Why does the ReLU activation function mitigate vanishing gradients?", "a": "For any positive input z > 0, the derivative of ReLU is exactly 1.0, preventing exponential attenuation during backpropagation."},
            {"q": "What is early stopping?", "a": "A regularization technique where model performance is monitored on a validation set and training is terminated when validation loss stops improving."},
            {"q": "How does dataset size relate to the risk of overfitting?", "a": "Smaller datasets are easier for high-capacity models to memorize, increasing the risk of overfitting; large datasets force the network to learn generalized representations."},
            {"q": "What is the spectral norm of a matrix?", "a": "The maximum singular value of the matrix, representing the maximum factor by which the matrix can stretch any vector."},
            {"q": "How do residual connections (skip connections) address vanishing gradients?", "a": "They add an identity shortcut: a^(l+1) = f(a^(l)) + a^(l), ensuring the gradient contains an additive identity term (+1) that allows gradients to flow uninterrupted to early layers."},
            {"q": "Why does uncalibrated random weight initialization cause gradient instability?", "a": "If weights are initialized too large, activations saturate or explode; if initialized too small, signal variance collapses toward zero."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "Gradient clipping is primarily used to counteract which training problem?",
                    "options": ["Vanishing gradients", "Exploding gradients", "Overfitting", "Underfitting"],
                    "ans": "Exploding gradients"
                },
                {
                    "q": "In an L-layer feedforward network with sigmoid activations, the gradient backpropagated to layer 1 is proportional to at most:",
                    "options": ["(0.5)^L", "(0.25)^L", "(1.0)^L", "(2.0)^L"],
                    "ans": "(0.25)^L"
                },
                {
                    "q": "A model exhibits low training error and high validation error. This is a classic symptom of:",
                    "options": ["High bias (underfitting)", "High variance (overfitting)", "Vanishing gradients", "Dead neurons"],
                    "ans": "High variance (overfitting)"
                }
            ],
            "short_questions": [
                "Explain how gradient clipping by norm operates and why it is superior to coordinate-wise value clipping.",
                "Discuss the role of saturating activation functions in the manifestation of vanishing gradients."
            ],
            "application_question": "A 20-layer deep neural network experiences training instability where loss fluctuates wildly to NaN at epoch 3. Diagnose the probable cause and provide a three-step engineering solution.",
            "homework": "Given gradient vector g = [6, -8]^T and maximum threshold c = 5.0, calculate the norm-clipped gradient vector."
        }
    },
    {
        "day": 7,
        "module": 1,
        "topic_id": "1.7",
        "title": "Convergence Difficulties, Local and Spurious Optima, Computational Challenges",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO1",
        "blooms": "Understand / Analyze",
        "prerequisites": "Optimization Theory, Multivariable Calculus (Hessian matrix, Eigenvalues), Day 6 concepts",
        "learning_objectives": [
            "Analyze the high-dimensional non-convex loss surface of deep neural networks.",
            "Differentiate between local minima, global minima, saddle points, and plateaus.",
            "Analyze the role of the Hessian matrix and its condition number in optimization slowdown.",
            "Address computational challenges: memory bottlenecks, distributed training overhead, and ill-conditioning."
        ],
        "key_terminology": [
            "Non-Convex Optimization", "Saddle Point", "Spurious Local Minima", "Hessian Matrix",
            "Ill-Conditioned Curvature", "Plateau", "Condition Number"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 8 (Section 8.2); Aggarwal: Chapter 3.",
        "notes": {
            "loss_surfaces_in_deep_learning": (
                "Unlike linear models whose convex loss functions guarantee a unique global minimum, deep neural networks have highly non-convex loss landscapes in millions of dimensions. The landscape is characterized by:\n"
                "1. Symmetries and Permutation Invariance: Permuting neurons within a hidden layer yields identical input-output mappings, resulting in a factorial number of equivalent local minima.\n"
                "2. Spurious Local Minima: Suboptimal local basins whose loss is significantly worse than the global minimum. Research (Dauphin et al., 2014; Choromanska et al., 2015) indicates that in high-dimensional deep models, true local minima are rare and generally close to global minimum loss.\n"
                "3. Saddle Points: Critical points where gradient grad L = 0, but the Hessian has both positive and negative eigenvalues. In high dimensions, saddle points are exponentially more prevalent than local minima!"
            ),
            "ill_conditioned_hessian": (
                "The Hessian matrix H contains second-order partial derivatives: H_ij = del^2 L / (del w_i del w_j).\n\n"
                "The condition number kappa(H) = lambda_max / lambda_min measures curvature disparity:\n"
                "- If kappa(H) >> 1, the loss surface forms an elongated canyon (ravine) where curvature is steep along one direction and flat along another.\n"
                "- First-order Gradient Descent oscillates back and forth across the steep walls while making agonizingly slow progress along the base toward the minimum.\n\n"
                "Second-order Taylor expansion of loss around current point w_0:\n"
                "L(w) approx L(w_0) + (w - w_0)^T * grad L + (1/2) * (w - w_0)^T * H * (w - w_0)\n"
                "A gradient descent step w = w_0 - eta * grad L yields expected loss change:\n"
                "delta L approx - eta * ||grad L||^2 + (1/2) * eta^2 * (grad L)^T * H * (grad L)\n"
                "If (grad L)^T * H * (grad L) is very large, gradient descent can actually INCREASE loss unless eta is miniscule."
            ),
            "computational_challenges": (
                "1. Memory Bandwidth Limits: Forward activations must be retained in GPU VRAM for the backward pass, limiting maximum batch size and model depth.\n"
                "2. Computational Scale: Training modern deep architectures requires petascale floating-point calculations.\n"
                "3. Vanishing Gradient Highways in Ravines: Stalling optimization on flat saddle plateaus."
            ),
            "exam_important_points": [
                "Sketch and differentiate: Local minimum, global minimum, saddle point, and plateau.",
                "Define the Hessian condition number and explain why ill-conditioning causes gradient descent oscillation.",
                "Explain why saddle points pose a greater threat than local minima in high-dimensional deep networks.",
                "Derivation of expected loss change using second-order Taylor expansion."
            ]
        },
        "teaching_script": {
            "00-05": "Show 3D wireframe plot of an elliptical bowl vs a saddle surface (monkey saddle).",
            "05-20": "Explain non-convexity in deep learning. Clarify why local minima are usually harmless while saddle points trap first-order optimizers.",
            "20-35": "Derive the Hessian matrix and condition number. Explain the ravine problem where gradients bounce erratically across valley walls.",
            "35-45": "Walk through second-order Taylor expansion to show why gradient descent fails when curvature is poorly conditioned.",
            "45-55": "Demonstrate the 2D Rosenbrock (Banana) function optimization simulation comparing plain GD with momentum.",
            "55-60": "Summarize Module 1. Announce Internal Test 1 preparation and transition to Module 2 (Advanced Optimization & Regularization)."
        },
        "python_demo": {
            "title": "Optimization on an Ill-Conditioned Ravine (Rosenbrock Function)",
            "description": "Simulation illustrating gradient descent struggling on an ill-conditioned canyon compared to momentum-accelerated descent.",
            "code": """import numpy as np

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
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Hessian Matrix Condition Number and Curvature Evaluation",
                "statement": "At a critical point in weight space, the Hessian matrix of a neural network loss function is computed as H = [[200.0, 0.0], [0.0, 2.0]]. (a) Identify the eigenvalues of H, (b) Calculate the condition number kappa(H), and (c) Characterize the nature of the curvature along both principal axes.",
                "given": "H = [[200.0, 0.0], [0.0, 2.0]]",
                "formulas": "Eigenvalues are diagonal entries for diagonal matrix; kappa = lambda_max / lambda_min",
                "steps": [
                    "Step 1: Eigenvalues are lambda_1 = 200.0 and lambda_2 = 2.0.",
                    "Step 2: Both eigenvalues are positive (lambda > 0), confirming that the critical point is a local minimum.",
                    "Step 3: Calculate condition number: kappa(H) = lambda_max / lambda_min = 200.0 / 2.0 = 100.0."
                ],
                "answer": "Eigenvalues = 200.0 and 2.0. Condition number kappa = 100.0 (Ill-conditioned).",
                "interpretation": "Curvature along the first coordinate axis is 100 times steeper than along the second axis. Vanilla gradient descent will severely oscillate along axis 1 while progressing very slowly along axis 2."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Second-Order Taylor Series Step-Size Bound",
                "statement": "The loss surface around current weights has gradient g = [4.0, 1.0]^T and Hessian H = [[10.0, 0.0], [0.0, 1.0]]. Using the second-order Taylor expansion delta L approx - eta * ||g||^2 + 0.5 * eta^2 * (g^T * H * g), (a) Calculate the optimal learning rate eta* that maximizes loss reduction, and (b) Compute the maximum learning rate eta_max beyond which the step causes the loss to INCREASE instead of decrease.",
                "given": "g = [4.0, 1.0]^T, H = [[10.0, 0.0], [0.0, 1.0]]",
                "formulas": "||g||^2 = g^T * g; g^T * H * g = sum(g_i^2 * H_ii); eta* = ||g||^2 / (g^T * H * g); eta_max = 2 * eta*",
                "steps": [
                    "Step 1: Compute ||g||^2 = 4^2 + 1^2 = 16 + 1 = 17.0.",
                    "Step 2: Compute curvature term: g^T * H * g = (4^2 * 10.0) + (1^2 * 1.0) = (16 * 10) + 1 = 160 + 1 = 161.0.",
                    "Step 3: Quadratic loss change equation: delta L(eta) = - 17.0 * eta + 0.5 * 161.0 * eta^2 = - 17.0 * eta + 80.5 * eta^2.",
                    "Step 4: Differentiate with respect to eta and set to 0: d(delta L)/d(eta) = -17.0 + 161.0 * eta = 0 => eta* = 17.0 / 161.0 = 0.10559.",
                    "Step 5: Compute threshold for loss increase (delta L = 0): -17.0 * eta + 80.5 * eta^2 = 0 => eta * (-17.0 + 80.5 * eta) = 0 => eta_max = 17.0 / 80.5 = 2 * eta* = 0.21118."
                ],
                "answer": "Optimal learning rate eta* = 0.1056; Maximum learning rate before divergence eta_max = 0.2112",
                "interpretation": "If the engineer sets a learning rate eta > 0.2112, the quadratic curvature penalty dominates the linear descent term, causing the loss to explode instead of decrease."
            }
        ],
        "real_world_application": {
            "title": "Seismic Oil & Gas Subsurface Reservoir Inversion",
            "domain": "Geophysics & Energy Exploration",
            "workflow": "Acoustic wave reflection sensors collect terabytes of subterranean echoes -> Deep neural acoustic inversion predicts subterranean rock density and porous oil pockets -> Highly ill-conditioned Hessian landscapes caused by varied rock strata require adaptive curvature optimization.",
            "benefit": "Prevents million-dollar exploratory dry-well drilling by resolving deep geological formation boundaries."
        },
        "viva_questions": [
            {"q": "What is a saddle point in a multivariable function?", "a": "A critical point where the gradient is zero, but the Hessian has both positive and negative eigenvalues, meaning it is a minimum along some directions and a maximum along others."},
            {"q": "Why are saddle points more common than local minima in deep networks?", "a": "In an N-dimensional space, for a critical point to be a local minimum, all N eigenvalues of the Hessian must be simultaneously positive (probability ~ (1/2)^N for random signs). For large N, this probability vanishes, making saddle points exponentially more frequent."},
            {"q": "What is the condition number of the Hessian matrix?", "a": "The ratio of its maximum eigenvalue to its minimum eigenvalue: kappa = lambda_max / lambda_min."},
            {"q": "How does an ill-conditioned Hessian affect gradient descent?", "a": "It creates severe oscillations perpendicular to the ravine walls while causing agonizingly slow progress along the canyon floor."},
            {"q": "What is a plateau in the loss surface?", "a": "A region where the loss surface is nearly flat with vanishingly small gradients across multiple dimensions, drastically slowing down training."},
            {"q": "Why is the loss surface of deep neural networks non-convex?", "a": "Due to non-linear activation functions and permutation symmetries between hidden neurons, leading to multiple equivalent minima and complex topological saddle surfaces."},
            {"q": "How does momentum help escape saddle points and flat plateaus?", "a": "Momentum accumulates past velocity vectors, allowing the optimizer to slide through regions of zero or tiny instantaneous gradient without stalling."},
            {"q": "What is second-order optimization, and why is it rarely used for large deep networks?", "a": "Methods like Newton-Raphson that multiply the gradient by the inverse Hessian H^(-1). It is computationally prohibitive because computing and inverting an (N x N) Hessian costs O(N^3) time and O(N^2) memory for millions of parameters."},
            {"q": "What is a spurious local minimum?", "a": "A suboptimal local minimum whose loss value is significantly higher than that of the global minimum, potentially trapping the optimizer in a bad model state."},
            {"q": "How does stochasticity in Mini-Batch SGD help overcome saddle points?", "a": "Batch noise introduces random perturbations into the gradient vector that push the parameters off the unstable saddle manifold in downhill directions."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "At a saddle point, the eigenvalues of the Hessian matrix are:",
                    "options": ["All strictly positive", "All strictly negative", "A mixture of positive and negative values", "All exactly zero"],
                    "ans": "A mixture of positive and negative values"
                },
                {
                    "q": "An ill-conditioned Hessian matrix is characterized by a condition number that is:",
                    "options": ["Equal to 1", "Close to 0", "Much greater than 1", "Negative"],
                    "ans": "Much greater than 1"
                },
                {
                    "q": "Which optimization challenge becomes exponentially more dominant as network dimensionality increases?",
                    "options": ["Global minima proliferation", "Saddle points", "Underfitting", "Convexity traps"],
                    "ans": "Saddle points"
                }
            ],
            "short_questions": [
                "Explain the geometric and computational consequences of an ill-conditioned Hessian condition number on vanilla gradient descent.",
                "Discuss why high-dimensional loss landscapes in deep learning are dominated by saddle points rather than local minima."
            ],
            "application_question": "Explain how modern first-order adaptive optimizers (like Adam) incorporate approximations of curvature without explicitly computing the full Hessian matrix.",
            "homework": "Prove that if the Hessian matrix H is positive definite (all eigenvalues > 0), any critical point grad L = 0 is a strict local minimum."
        }
    }
]

