"""
Curriculum Data for Module 3: Convolutional Neural Networks (6 Hours)
KTU CST414 Deep Learning (Scheme 2019)
Faculty: Jane Rose | Department of Computer Science & Engineering
"""

MODULE_3_DAYS = [
    {
        "day": 17,
        "module": 3,
        "topic_id": "3.1",
        "title": "Convolutional Neural Networks: The Convolution Operation",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO3",
        "blooms": "Understand / Apply",
        "prerequisites": "Matrix algebra, 2D grids, Multi-Layer Perceptrons, Day 8 concepts",
        "learning_objectives": [
            "Formulate continuous and discrete 2D convolution operations.",
            "Differentiate between mathematical convolution (with kernel flipping) and cross-correlation used in deep learning.",
            "Compute 2D feature map outputs given an input image matrix and filter kernel.",
            "Understand the physical role of filters: edge detection (Sobel, Prewitt), sharpening, and Gaussian blur."
        ],
        "key_terminology": [
            "Convolution", "Cross-Correlation", "Filter Kernel", "Feature Map",
            "Kernel Flipping", "Receptive Field", "Discrete 2D Convolution"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 9 (Section 9.1); Aggarwal: Chapter 8.",
        "notes": {
            "convolution_operation": (
                "Continuous 1D Convolution:\n"
                "s(t) = (x * w)(t) = integral_{-infinity}^{infinity} x(a) * w(t - a) da\n\n"
                "Discrete 2D Mathematical Convolution (with Kernel Flip):\n"
                "S(i, j) = (I * K)(i, j) = sum_m sum_n I(i - m, j - n) * K(m, n)\n\n"
                "Discrete 2D Cross-Correlation (Deep Learning Convention):\n"
                "S(i, j) = (I * K)(i, j) = sum_m sum_n I(i + m, j + n) * K(m, n)\n\n"
                "In deep learning libraries (PyTorch, TensorFlow), cross-correlation is implemented without flipping the kernel because kernel weights are learned directly from data via backpropagation, making learning flipped weights mathematically equivalent to learning unflipped weights."
            ),
            "feature_map_and_filters": (
                "A filter kernel is a small matrix (e.g. 3x3, 5x5) whose elements act as feature detectors:\n"
                "- Horizontal Sobel Filter: K = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]\n"
                "- Vertical Sobel Filter:   K = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]\n"
                "Sliding this across an image generates a 2D scalar activation map highlighting local directional features."
            ),
            "exam_important_points": [
                "Continuous vs discrete 2D convolution formulas.",
                "Why deep learning libraries implement cross-correlation rather than true flipped convolution.",
                "Manual calculation of 2D convolution on a 4x4 matrix with a 3x3 kernel.",
                "Hand-crafted kernels: Sobel horizontal and vertical edge detectors."
            ]
        },
        "teaching_script": {
            "00-05": "Introduce handwritten digit recognition problem. Explain why spatial locality matters.",
            "05-20": "Formulate 1D continuous and 2D discrete convolution equations. Explain kernel flipping.",
            "20-35": "Differentiate cross-correlation from true convolution. Explain why frameworks omit the flip.",
            "35-45": "Demonstrate manual 2D convolution on a 4x4 matrix with a 3x3 Sobel kernel.",
            "45-55": "Demonstrate Python script convolving a real image with Sobel filters.",
            "55-60": "Recap and preview Day 18: Motivation and Pooling."
        },
        "python_demo": {
            "title": "2D Image Convolution from Scratch with NumPy",
            "description": "Pure NumPy implementation of 2D discrete cross-correlation applying Sobel edge detection kernels.",
            "code": """import numpy as np

def conv2d(image, kernel):
    i_h, i_w = image.shape
    k_h, k_w = kernel.shape
    out_h = i_h - k_h + 1
    out_w = i_w - k_w + 1
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = image[i:i+k_h, j:j+k_w]
            output[i, j] = np.sum(patch * kernel)
    return output

image = np.array([
    [10, 10, 10, 0, 0],
    [10, 10, 10, 0, 0],
    [10, 10, 10, 0, 0],
    [10, 10, 10, 0, 0],
    [10, 10, 10, 0, 0]
])

sobel_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
feature_map = conv2d(image, sobel_v)
print("Convolved Feature Map (3x3):\\n", feature_map)
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Manual 2D Discrete Convolution Calculation",
                "statement": "Given a 4x4 input image I = [[1, 2, 0, 1], [0, 1, 3, 2], [2, 0, 1, 0], [1, 1, 0, 2]] and a 3x3 kernel K = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]. Calculate the valid 2D cross-correlation output feature map S.",
                "given": "Input 4x4, Kernel 3x3, stride=1, no padding.",
                "formulas": "Output shape = (4 - 3 + 1) x (4 - 3 + 1) = 2x2. S(i, j) = sum(I_patch * K)",
                "steps": [
                    "Step 1: Output dimension is 2x2: S(0,0), S(0,1), S(1,0), S(1,1).",
                    "Step 2: S(0,0) = (1*1 + 2*0 + 0*-1) + (0*1 + 1*0 + 3*-1) + (2*1 + 0*0 + 1*-1) = 1 - 3 + 1 = -1.",
                    "Step 3: S(0,1) = (2*1 + 0*0 + 1*-1) + (1*1 + 3*0 + 2*-1) + (0*1 + 1*0 + 0*-1) = 1 - 1 + 0 = 0.",
                    "Step 4: S(1,0) = (0*1 + 1*0 + 3*-1) + (2*1 + 0*0 + 1*-1) + (1*1 + 1*0 + 0*-1) = -3 + 1 + 1 = -1.",
                    "Step 5: S(1,1) = (1*1 + 3*0 + 2*-1) + (0*1 + 1*0 + 0*-1) + (1*1 + 0*0 + 2*-1) = -1 + 0 - 1 = -2."
                ],
                "answer": "Feature Map S = [[-1, 0], [-1, -2]]",
                "interpretation": "The 3x3 vertical difference kernel extracts spatial vertical edge gradients across the 4x4 grid."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Mathematical Equivalence of Cross-Correlation and Convolution",
                "statement": "Let continuous 1D convolution be (f * g)(t) = integral f(tau) * g(t - tau) d tau and cross-correlation be (f star g)(t) = integral f(tau) * g(t + tau) d tau. Prove that (f star g)(t) = (f * g_minus)(t) where g_minus(t) = g(-t), and explain why deep learning frameworks can optimize cross-correlation filters directly without explicit flipping.",
                "given": "Definitions of convolution, cross-correlation, and flipped kernel g_minus(t) = g(-t).",
                "formulas": "(f * g_minus)(t) = integral f(tau) * g_minus(t - tau) d tau",
                "steps": [
                    "Step 1: Write convolution: (f * g_minus)(t) = integral f(tau) * g_minus(t - tau) d tau.",
                    "Step 2: Substitute g_minus(u) = g(-u): g_minus(t - tau) = g(-(t - tau)) = g(tau - t).",
                    "Step 3: In 2D arrays: K_flipped[i, j] = K[-i, -j]. Thus cross-correlation with K equals convolution with K_flipped.",
                    "Step 4: In deep learning, weights in K are free parameters learned end-to-end to minimize loss; the optimizer learns the optimal orientation directly, eliminating the need to reverse arrays at runtime."
                ],
                "answer": "Cross-correlation with kernel K is mathematically identical to true convolution with flipped kernel K_minus.",
                "interpretation": "End-to-end learning renders the mathematical difference between convolution and cross-correlation a mere indexing convention."
            }
        ],
        "real_world_application": {
            "title": "Silicon Wafer Defect Inspection",
            "domain": "Semiconductor Manufacturing",
            "workflow": "High-res microscope scans wafer -> 2D convolution applies learned micro-crack filters -> Edge-enhanced feature maps identify hairline sub-micron etch defects.",
            "benefit": "Prevents defective silicon chips from reaching automotive and aerospace electronic engine control units."
        },
        "viva_questions": [
            {"q": "What is the difference between mathematical convolution and cross-correlation?", "a": "Mathematical convolution flips the kernel by 180 degrees before computing the inner product; cross-correlation slides the kernel without flipping."},
            {"q": "Why do deep learning frameworks implement cross-correlation while calling it convolution?", "a": "Because kernel weights are learned from data; learning flipped weights is mathematically equivalent to learning unflipped weights, so flipping is an unnecessary computational overhead."},
            {"q": "What is a feature map?", "a": "The 2D output array produced by sliding a convolutional filter across an input image and computing local inner products."},
            {"q": "What determines output spatial dimensions of a 2D convolution without padding?", "a": "Output Height = H_in - K_h + 1, Output Width = W_in - K_w + 1."},
            {"q": "What does a 3x3 Sobel kernel detect?", "a": "Spatial image intensity gradients (directional edges) along horizontal or vertical orientations."},
            {"q": "What is a receptive field in a CNN?", "a": "The region of the input image that contributes to the activation of a specific neuron in a deeper feature map."},
            {"q": "How does receptive field grow with depth?", "a": "It expands linearly with each successive convolutional layer: RF_size = RF_prev + (K - 1) * Stride_cumulative."},
            {"q": "Why are small 3x3 filters preferred over large 7x7 filters in modern architectures?", "a": "A stack of two 3x3 layers has the same 5x5 receptive field as a single 5x5 layer, but uses fewer parameters (18 vs 25) and incorporates two non-linear activations."},
            {"q": "What is the output dimension if a 28x28 image is convolved with a 5x5 filter (stride=1, no padding)?", "a": "28 - 5 + 1 = 24x24."},
            {"q": "Can a 2D convolution be applied to audio signals?", "a": "Yes, by converting audio waveforms into 2D time-frequency spectrograms."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "The operation implemented by modern deep learning libraries under the name 'conv2d' is technically:",
                    "options": ["Mathematical convolution with flipped kernel", "Discrete cross-correlation without kernel flipping", "Fourier transform", "Matrix inversion"],
                    "ans": "Discrete cross-correlation without kernel flipping"
                },
                {
                    "q": "Convolving a 32x32 image with a 5x5 filter using stride=1 and no padding yields a feature map of shape:",
                    "options": ["32x32", "28x28", "27x27", "26x26"],
                    "ans": "28x28"
                },
                {
                    "q": "A Sobel filter is primarily used in image processing to detect:",
                    "options": ["Color saturation", "Spatial edges and intensity gradients", "Image resolution", "Noise artifacts"],
                    "ans": "Spatial edges and intensity gradients"
                }
            ],
            "short_questions": [
                "Formulate continuous and discrete 2D convolution operations and explain the physical meaning of the kernel flip.",
                "Explain why deep learning practitioners treat cross-correlation and convolution interchangeably."
            ],
            "application_question": "Explain how a convolutional layer detects road lane boundaries in an autonomous vehicle front camera feed.",
            "homework": "Compute the 2D cross-correlation of a 3x3 image [[1, 2, 3], [4, 5, 6], [7, 8, 9]] with a 2x2 kernel [[1, 0], [0, 1]]."
        }
    },
    {
        "day": 18,
        "module": 3,
        "topic_id": "3.2",
        "title": "Motivation for CNNs and Pooling Operations",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO3",
        "blooms": "Understand / Apply",
        "prerequisites": "Convolution Operation, Multi-Layer Perceptrons, Day 17 concepts",
        "learning_objectives": [
            "Analyze the three foundational architectural motivations of CNNs: Sparse Interactions, Parameter Sharing, and Equivariant Representations.",
            "Formulate Pooling operations: Max Pooling, Average Pooling, and L2-Norm Pooling.",
            "Analyze how pooling induces translation invariance to small spatial shifts.",
            "Calculate output spatial dimensions and parameter savings in convolutional-pooling pipelines."
        ],
        "key_terminology": [
            "Sparse Interactions", "Parameter Sharing", "Equivariance to Translation",
            "Max Pooling", "Average Pooling", "Translation Invariance", "Downsampling"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 9 (Sections 9.2 & 9.3); Aggarwal: Chapter 8.",
        "notes": {
            "three_motivations_for_cnns": (
                "1. Sparse Interactions: Kernel size k << input size m. Receptive fields are local, reducing storage from O(m*n) to O(k*n) and slashing FLOPs.\n"
                "2. Parameter Sharing: Reusing kernel weights across all positions slashes parameter counts and prevents overfitting.\n"
                "3. Equivariant Representations: f(g(x)) = g(f(x)). Shifting the input image translates feature activations identically."
            ),
            "pooling_operations": (
                "Pooling summarizes nearby outputs:\n"
                "- Max Pooling: S(i, j) = max_{(p,q) in R} I(i+p, j+q) (captures dominant activations).\n"
                "- Average Pooling: S(i, j) = (1/|R|) sum_{(p,q) in R} I(i+p, j+q) (smooth background summary).\n"
                "Pooling provides approximate invariance to small translations of the input."
            ),
            "exam_important_points": [
                "Sparse Interactions, Parameter Sharing, and Translation Equivariance.",
                "Translation Equivariance (convolution) vs Translation Invariance (pooling).",
                "Max Pooling vs Average Pooling manual calculation on a 4x4 matrix.",
                "How pooling reduces spatial dimensions and controls overfitting."
            ]
        },
        "teaching_script": {
            "00-05": "Contrast 1000x1000 MLP (1 trillion weights) with CNN (25 weights).",
            "05-20": "Explain Sparse Interactions, Parameter Sharing, and Translation Equivariance.",
            "20-35": "Differentiate Equivariance f(g(x)) = g(f(x)) from Invariance f(g(x)) = f(x). Show how pooling enables invariance.",
            "35-45": "Demonstrate Max Pooling and Average Pooling on blackboard with a worked 4x4 grid.",
            "45-55": "Demonstrate Python script performing max pooling with 1-pixel shift.",
            "55-60": "Recap and preview Day 19: Prior Interpretation."
        },
        "python_demo": {
            "title": "Max Pooling and Translation Invariance in NumPy",
            "description": "NumPy implementation of 2D Max Pooling demonstrating invariance to a 1-pixel spatial shift.",
            "code": """import numpy as np

def max_pool2d(matrix, pool_size=2, stride=2):
    H, W = matrix.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = matrix[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            output[i, j] = np.max(patch)
    return output

A = np.array([[0, 10, 2, 1], [1, 8, 3, 0], [0, 2, 9, 1], [0, 1, 4, 2]])
B = np.array([[0, 0, 10, 2], [0, 1, 8, 3], [0, 0, 2, 9], [0, 0, 1, 4]])

print("Pooled A:\\n", max_pool2d(A))
print("Pooled B (Shifted):\\n", max_pool2d(B))
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Max and Average Pooling Calculation",
                "statement": "Given 4x4 matrix M = [[12, 20, 30, 0], [8, 16, 2, 14], [4, 6, 8, 10], [18, 22, 12, 14]], perform 2x2 pooling with stride 2: (a) Compute Max Pooling, and (b) Compute Average Pooling.",
                "given": "4x4 matrix, pool_size=2, stride=2.",
                "formulas": "Max = max(patch); Avg = mean(patch)",
                "steps": [
                    "Q1 [[12, 20], [8, 16]]: Max = 20, Avg = 56/4 = 14.0.",
                    "Q2 [[30, 0], [2, 14]]: Max = 30, Avg = 46/4 = 11.5.",
                    "Q3 [[4, 6], [18, 22]]: Max = 22, Avg = 50/4 = 12.5.",
                    "Q4 [[8, 10], [12, 14]]: Max = 14, Avg = 44/4 = 11.0."
                ],
                "answer": "Max Pool = [[20, 30], [22, 14]], Avg Pool = [[14.0, 11.5], [12.5, 11.0]]",
                "interpretation": "Max pooling captures peak signals while average pooling extracts mean background response."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Sparse Interactions vs Dense MLP Parameter Reduction",
                "statement": "An image with resolution 1000x1000 is processed by a hidden layer of 1000x1000 units. (a) Calculate weights in fully connected MLP, (b) Calculate weights in CNN with a 5x5 filter and 1 feature map, (c) Compute parameter reduction ratio, and (d) Calculate parameters with 16 filters.",
                "given": "Input 10^6, Hidden 10^6, Kernel 5x5.",
                "formulas": "MLP = m * n; CNN = k_h * k_w * filters",
                "steps": [
                    "Step 1: MLP weights = 10^6 * 10^6 = 10^12 = 1 Trillion parameters.",
                    "Step 2: CNN weights (1 filter) = 5 * 5 = 25 parameters.",
                    "Step 3: Reduction ratio = 10^12 / 25 = 40,000,000,000x.",
                    "Step 4: CNN weights (16 filters) = 16 * 25 = 400 parameters."
                ],
                "answer": "MLP = 1 Trillion weights. CNN (1 filter) = 25 weights (40B x reduction). CNN (16 filters) = 400 weights.",
                "interpretation": "CNNs make high-resolution vision computationally tractable through parameter sharing."
            }
        ],
        "real_world_application": {
            "title": "Biometric Facial Recognition Gates",
            "domain": "Biometrics & Airport Security",
            "workflow": "Passenger walks past camera -> Head position varies by centimeters -> Max pooling provides translation invariance -> Produces stable face embeddings.",
            "benefit": "Enables seamless passenger boarding without requiring rigid motionless posture alignment."
        },
        "viva_questions": [
            {"q": "What are the three core motivations of CNNs?", "a": "Sparse Interactions, Parameter Sharing, and Equivariant Representations."},
            {"q": "What is translation equivariance?", "a": "Shifting input causes output feature map to shift by the identical amount: f(g(x)) = g(f(x))."},
            {"q": "Does max pooling contain trainable weights?", "a": "No, it is a fixed deterministic non-linear mathematical operation with zero learnable parameters."},
            {"q": "What is Global Average Pooling?", "a": "Averaging all spatial values across an entire 2D channel to produce a single scalar per feature map."},
            {"q": "How does pooling reduce compute in deeper layers?", "a": "Downsampling height and width by 2x reduces spatial locations for subsequent layers by 75%."},
            {"q": "What is the difference between Max Pooling and Average Pooling?", "a": "Max pooling retains dominant feature spikes; average pooling smooths background activation distributions."},
            {"q": "Why has GAP replaced dense layers in modern CNNs?", "a": "It has zero parameters, prevents overfitting, and accepts flexible input resolutions."},
            {"q": "What is L2 pooling?", "a": "A pooling function computing the square root of the sum of squared activations in each window."},
            {"q": "Can pooling cause loss of spatial location details?", "a": "Yes, by discarding exact coordinates, which can impair precise pixel localization in semantic segmentation."},
            {"q": "What is fractional max pooling?", "a": "A stochastic pooling technique where downsampling ratios are non-integers (e.g. 1.4x)."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "The property that shifting the input causes the feature map to shift identically is:",
                    "options": ["Translation Invariance", "Translation Equivariance", "Scale Invariance", "Permutation Invariance"],
                    "ans": "Translation Equivariance"
                },
                {
                    "q": "How many trainable weights are in a 2x2 Max Pooling layer with stride 2?",
                    "options": ["0", "4", "2", "Depends on channels"],
                    "ans": "0"
                },
                {
                    "q": "Sparse connectivity in CNNs means each output unit connects only to:",
                    "options": ["All inputs", "A small local receptive field of inputs", "The bias term only", "Zero weights"],
                    "ans": "A small local receptive field of inputs"
                }
            ],
            "short_questions": [
                "Explain the three architectural motivations of CNNs: Sparse Interactions, Parameter Sharing, and Equivariance.",
                "Contrast Translation Equivariance with Translation Invariance and explain the role of pooling."
            ],
            "application_question": "Explain why Global Average Pooling is superior to flattening into a dense layer for mobile edge deployment.",
            "homework": "Given a 6x6 matrix, perform 2x2 Max Pooling with stride=2 and compute the percentage reduction in spatial elements."
        }
    },
    {
        "day": 19,
        "module": 3,
        "topic_id": "3.3",
        "title": "Convolution and Pooling as an Infinitely Strong Prior",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO3",
        "blooms": "Understand / Analyze",
        "prerequisites": "Bayesian Probability (priors, posteriors), Convolution and Pooling mechanics, Day 18 concepts",
        "learning_objectives": [
            "Interpret neural network architectures from the perspective of Bayesian Priors.",
            "Formulate Convolution and Pooling as an 'Infinitely Strong Prior' on the network's weights.",
            "Analyze the failure modes of CNN priors when spatial locality assumptions are violated (e.g. permuting pixels).",
            "Evaluate when infinitely strong priors are beneficial versus when they become harmful inductive biases."
        ],
        "key_terminology": [
            "Bayesian Prior", "Infinitely Strong Prior", "Inductive Bias", "Parameter Constraint",
            "Permutation Vulnerability", "Spatial Locality Assumption"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 9 (Section 9.4); Aggarwal: Chapter 8.",
        "notes": {
            "bayesian_prior_perspective": (
                "In Bayesian statistics, a prior probability distribution p(theta) reflects our beliefs about which parameter configurations are plausible before observing any training data.\n\n"
                "A prior is 'weak' or 'soft' if it assigns positive probability to all parameter values, merely preferring some over others (e.g. Gaussian prior in L2 regularization with high variance).\n\n"
                "A prior is 'infinitely strong' if it assigns zero probability to certain parameter configurations, making it mathematically impossible for the learning algorithm to ever choose those weights regardless of how much training data is collected."
            ),
            "convolution_and_pooling_as_infinitely_strong_prior": (
                "A Convolutional layer can be viewed as a fully connected layer with an infinitely strong prior specifying that:\n"
                "1. All weights connecting non-spatially adjacent pixels must be identically ZERO: p(W_ij != 0) = 0 for |i - j| > kernel_radius.\n"
                "2. Weights for spatially shifted connections must be EXACTLY IDENTICAL (parameter sharing): p(W_ij != W_{i+1, j+1}) = 0.\n\n"
                "Similarly, a Pooling layer is an infinitely strong prior specifying that each unit must be invariant to small local translations.\n\n"
                "Benefits and Hazards:\n"
                "- Benefit: If the prior assumption is TRUE (e.g. in natural images where pixels have local stationary correlations), the CNN learns with massive sample efficiency, avoiding overfitting.\n"
                "- Hazard: If the prior assumption is FALSE (e.g. tabular data where column ordering is arbitrary, or permuted images), an infinitely strong prior prevents the network from learning the true relationship, leading to high underfitting bias."
            ),
            "permutation_experiment": (
                "If we randomly permute all pixels of MNIST images with a fixed permutation matrix P:\n"
                "- A fully connected MLP achieves the EXACT same accuracy on permuted MNIST as on unpermuted MNIST (MLPs have a weak, isotropic prior).\n"
                "- A CNN's performance collapses dramatically on permuted MNIST because the infinitely strong spatial locality prior is violently violated!"
            ),
            "exam_important_points": [
                "Definition of weak vs infinitely strong Bayesian priors.",
                "How convolution enforces an infinitely strong prior: zero weights outside kernel + identical shifted weights.",
                "Explanation of the Permuted MNIST experiment comparing MLPs and CNNs.",
                "When infinitely strong priors lead to catastrophic underfitting."
            ]
        },
        "teaching_script": {
            "00-05": "Introduce Bayesian prior concept: 'How does an architect encode domain knowledge into network structure?'",
            "05-20": "Define soft priors (L2 regularization) vs infinitely strong priors (architectural hard constraints).",
            "20-35": "Derive how a CNN expresses an infinitely strong prior over a dense MLP weight matrix: forced zeros and tied values.",
            "35-45": "Discuss the Permuted MNIST experiment: why an MLP does not care about pixel permutation while a CNN fails completely.",
            "45-55": "Demonstrate Python script permuting MNIST pixels and testing CNN vs MLP accuracy.",
            "55-60": "Recap and preview Day 20: Variants of the Convolution Function (Strides, Padding, Dilations)."
        },
        "python_demo": {
            "title": "Permuted Pixels Experiment: MLP vs CNN Inductive Bias",
            "description": "Simulation illustrating how random spatial permutation destroys spatial correlation, violating CNN priors.",
            "code": """import numpy as np

# Create an 8x8 image with a clear spatial line
img = np.zeros((8, 8))
img[3:5, :] = 1.0 # horizontal bar

# Generate a fixed random permutation of 64 pixel indices
np.random.seed(42)
perm = np.random.permutation(64)

# Permute image
img_flat = img.flatten()
img_perm = img_flat[perm].reshape((8, 8))

# Measure local spatial correlation (adjacent horizontal pixel correlation)
def horizontal_correlation(image):
    return np.corrcoef(image[:, :-1].flatten(), image[:, 1:].flatten())[0, 1]

corr_original = horizontal_correlation(img)
corr_permuted = horizontal_correlation(img_perm)

print("Original Image Adjacent Pixel Correlation:", round(corr_original, 4))
print("Permuted Image Adjacent Pixel Correlation:", round(corr_permuted, 4))
print("\\nCNN assumes high local spatial correlation (Infinitely Strong Prior).")
print("Permutation destroys this correlation, causing CNN performance collapse!")
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Weight Matrix Density under Infinitely Strong Prior",
                "statement": "Consider a 1D input of length N = 100 mapped to an output of length N = 100. (a) In a dense MLP, calculate the total number of weight connections in the 100x100 matrix, (b) In a convolutional layer with kernel size k = 5 and circular padding, calculate the number of non-zero entries in the matrix, and (c) Determine the percentage of weights forced to be ZERO by the infinitely strong spatial locality prior.",
                "given": "N = 100, k = 5.",
                "formulas": "MLP entries = N^2; Conv non-zero entries = N * k; Sparsity % = (1 - NonZero / Total) * 100",
                "steps": [
                    "Step 1: Dense MLP total weight matrix entries = 100 * 100 = 10,000 entries.",
                    "Step 2: Convolutional kernel k = 5 means each row has exactly 5 non-zero entries.\n  Total non-zero entries across 100 rows = 100 * 5 = 500 entries.",
                    "Step 3: Number of zero-constrained entries = 10,000 - 500 = 9,500 entries.",
                    "Step 4: Percentage of weights forced to zero = (9,500 / 10,000) * 100 = 95.0%."
                ],
                "answer": "Non-zero entries = 500; Zero-constrained entries = 9,500 (95.0% forced to zero).",
                "interpretation": "The infinitely strong prior zeroes out 95% of all conceivable linear connections and constrains the remaining 500 entries to share only 5 distinct parameter values."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Bayesian Gaussian Prior vs Infinitely Strong Prior Variance Limit",
                "statement": "In Bayesian linear regression, weights have a zero-mean Gaussian prior p(w) = N(0, sigma_prior^2 * I). (a) Write the MAP regularized objective function, (b) What happens to the prior as sigma_prior^2 -> infinity? (c) What happens to the prior as sigma_prior^2 -> 0? and (d) Explain how an infinitely strong prior relates to setting prior variance to zero for non-local weights.",
                "given": "p(w) = N(0, sigma_prior^2 * I), Loss = L_data - ln p(w).",
                "formulas": "J_MAP(w) = J_data(w) + (1 / (2 * sigma_prior^2)) * ||w||_2^2",
                "steps": [
                    "Step 1: Write negative log posterior:\n  J_MAP(w) = J_data(w) + (1 / (2 * sigma_prior^2)) * sum_i w_i^2 + const.\n  This is exact L2 regularization with penalty alpha = 1 / sigma_prior^2.",
                    "Step 2: Limit as sigma_prior^2 -> infinity:\n  alpha -> 0. The prior becomes an improper uniform distribution, exerting zero constraint on weights (unregularized maximum likelihood).",
                    "Step 3: Limit as sigma_prior^2 -> 0:\n  alpha -> infinity. The penalty for any non-zero weight w_i != 0 approaches infinity, forcing w_i to be identically zero with probability 1.",
                    "Step 4: Connection to CNNs:\n  A CNN sets sigma_prior^2 = 0 for all weights connecting pixels outside the local kernel window, creating a Dirac delta prior delta(w_ij) that permanently forbids non-local interactions."
                ],
                "answer": "sigma_prior^2 -> 0 yields an infinitely strong prior forcing non-local weights to zero.",
                "interpretation": "Architectural constraints in CNNs are equivalent to infinitely strong Bayesian priors with zero variance on non-local weights."
            }
        ],
        "real_world_application": {
            "title": "Genomic Variant Pathogenicity Modeling",
            "domain": "Bioinformatics & Computational Genomics",
            "workflow": "DNA sequences have local spatial motif structure (transcription factor binding sites) -> CNNs with local 1D convolution priors detect regulatory motifs -> Tabular patient clinical data (age, lab blood values) lacks spatial ordering -> Processed via gradient boosted trees or MLPs to avoid harmful spatial priors.",
            "benefit": "Applying the correct inductive bias prevents misinterpreting unstructured patient records while capturing sequential genomic motifs."
        },
        "viva_questions": [
            {"q": "What is an infinitely strong prior in Bayesian terms?", "a": "A prior distribution that assigns zero probability to certain parameter values, completely forbidding the model from adopting those parameters regardless of the training data."},
            {"q": "How does a convolutional layer represent an infinitely strong prior?", "a": "By forcing all weights outside the local kernel window to be identically zero and forcing weights at different spatial positions to be identical (parameter sharing)."},
            {"q": "What happens if you train a CNN on images with randomly permuted pixels?", "a": "Performance degrades severely because the spatial locality and translation invariance priors are violated."},
            {"q": "Why does an MLP perform identically on original and permuted MNIST?", "a": "Because an MLP is permutation invariant with respect to its input features; it possesses no spatial locality prior."},
            {"q": "When is an infinitely strong prior beneficial?", "a": "When the underlying physical process satisfies the prior assumptions (e.g. 2D natural images possessing translation invariance and local correlations)."},
            {"q": "When does an infinitely strong prior harm performance?", "a": "When the prior assumption is false (e.g. tabular data with arbitrary column orders), leading to severe underfitting."},
            {"q": "What is an inductive bias?", "a": "The set of assumptions an algorithm uses to predict outputs on previously unseen inputs."},
            {"q": "How does pooling act as an infinitely strong prior?", "a": "It enforces the assumption that the exact spatial location of a detected feature within a small window does not change the classification."},
            {"q": "Can an infinitely strong prior be overcome by collecting infinite training data?", "a": "No; because the prior assigns probability zero to disallowed configurations, Bayes' theorem ensures the posterior probability remains zero regardless of data."},
            {"q": "What is the difference between an L2 weight decay prior and a CNN structural prior?", "a": "L2 weight decay is a soft Gaussian prior that prefers smaller weights; a CNN architectural prior is infinitely strong, strictly zeroing out non-local weights."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "An infinitely strong prior in Bayesian neural network modeling refers to:",
                    "options": ["A prior with infinite variance", "A prior that assigns zero probability to certain parameter configurations", "A prior learned with Adam", "A uniform distribution over all reals"],
                    "ans": "A prior that assigns zero probability to certain parameter configurations"
                },
                {
                    "q": "If all pixels of an image dataset are subjected to a fixed random spatial permutation, which architecture suffers catastrophic performance loss?",
                    "options": ["Fully connected MLP", "Convolutional Neural Network", "Logistic Regression", "Naïve Bayes"],
                    "ans": "Convolutional Neural Network"
                },
                {
                    "q": "Convolution enforces that weights between non-adjacent pixels are:",
                    "options": ["Small random numbers", "Learned with L1 penalty", "Strictly zero", "Equal to the bias"],
                    "ans": "Strictly zero"
                }
            ],
            "short_questions": [
                "Explain the concept of Convolution and Pooling as an infinitely strong prior from a Bayesian perspective.",
                "Describe the Permuted MNIST experiment and explain what it reveals about the inductive biases of MLPs vs CNNs."
            ],
            "application_question": "Explain why applying a standard 2D CNN directly to tabular credit card fraud data often produces worse results than an MLP or Random Forest.",
            "homework": "Show that if prior p(w) = 0 for w > 1, the posterior p(w|D) = 0 for all w > 1 regardless of the likelihood p(D|w)."
        }
    },
    {
        "day": 20,
        "module": 3,
        "topic_id": "3.4",
        "title": "Variants of the Convolution Function",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO3",
        "blooms": "Understand / Apply",
        "prerequisites": "Convolution Operation, Day 17 concepts",
        "learning_objectives": [
            "Formulate and compute Strided Convolutions and analyze their downsampling properties.",
            "Compare Zero-Padding strategies: Valid, Same (Half), and Full padding.",
            "Formulate Unshared Convolutions (Locally Connected layers) and Tiled Convolutions.",
            "Analyze 1x1 Convolutions (Network in Network) and Dilated (Atrous) Convolutions."
        ],
        "key_terminology": [
            "Strided Convolution", "Valid Padding", "Same Padding", "Full Padding",
            "Locally Connected Layers", "Tiled Convolution", "1x1 Convolution", "Dilated (Atrous) Convolution"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 9 (Section 9.5); Aggarwal: Chapter 8.",
        "notes": {
            "strided_convolution": (
                "Standard convolution slides the kernel by 1 pixel at each step. A strided convolution slides the kernel by s pixels (s > 1):\n"
                "S(i, j) = sum_m sum_n I(i*s + m, j*s + n) * K(m, n)\n"
                "Effect: Downsamples the feature map by a factor of s without requiring a separate pooling layer. Modern architectures (ResNet, MobileNet) frequently replace max pooling with strided convolutions."
            ),
            "zero_padding_types": (
                "For input dimension W, kernel size K, stride s, and zero-padding p:\n"
                "Output Dimension: W_out = floor( (W - K + 2*p) / s ) + 1\n\n"
                "1. Valid Padding (p = 0):\n"
                "   No padding added. W_out = (W - K + 1). Feature maps shrink at every layer.\n"
                "2. Same Padding (p = (K - 1) / 2 for odd K, s = 1):\n"
                "   Zeros added such that output spatial dimensions exactly match input dimensions: W_out = W.\n"
                "3. Full Padding (p = K - 1, s = 1):\n"
                "   Zeros added such that every pixel is visited K times: W_out = W + K - 1."
            ),
            "specialized_convolution_variants": (
                "1. 1x1 Convolution (Lin et al., Network in Network, 2013):\n"
                "   Acts as a cross-channel linear combination of features without altering spatial resolution. Crucial for channel dimensionality reduction (e.g. Inception bottlenecks, ResNet residual blocks).\n\n"
                "2. Dilated (Atrous) Convolution (Yu & Koltun, 2015):\n"
                "   Inserts d - 1 zeros (holes) between kernel elements. Effective kernel size: K_eff = K + (K - 1) * (d - 1).\n"
                "   Exponentially expands receptive field without increasing parameter count or losing spatial resolution (widely used in WaveNet and DeepLab semantic segmentation).\n\n"
                "3. Locally Connected Layers (Unshared Convolutions):\n"
                "   Local receptive fields without parameter sharing. Used when features are not stationary across the grid (e.g. human face alignment in DeepFace where eyes are always near the top)."
            ),
            "exam_important_points": [
                "Master formula for output spatial dimension: floor((W - K + 2p)/s) + 1.",
                "Definitions and diagrams for Valid, Same, and Full padding.",
                "Purpose and parameter reduction of 1x1 convolutions in Inception/ResNet.",
                "Dilated convolution receptive field formula and application in semantic segmentation."
            ]
        },
        "teaching_script": {
            "00-05": "Introduce the spatial shrinking dilemma: 'If a 32x32 image goes through 10 valid conv layers, what is left?' Answer: 12x12. Introduce Zero-Padding.",
            "05-20": "Write the master dimension formula. Work out Valid vs Same vs Full padding on blackboard.",
            "20-35": "Introduce Strided Convolution (s > 1) as a learned alternative to pooling. Compute output dimensions for stride 2.",
            "35-45": "Explain 1x1 Convolutions (pointwise convolutions) for channel projection. Show how 1x1 conv cuts parameters in bottleneck blocks.",
            "45-55": "Demonstrate Dilated Convolutions: show how dilation rate d expands receptive fields exponentially.",
            "55-60": "Recap and preview Day 21: Structured Outputs and Diverse Data Types."
        },
        "python_demo": {
            "title": "Convolution Output Dimension Calculator & Dilated Kernel in NumPy",
            "description": "Interactive calculator computing output shapes across Valid, Same, Strided, and Dilated configurations.",
            "code": """import numpy as np

def compute_output_dim(W, K, p, s, d=1):
    K_eff = K + (K - 1) * (d - 1)
    W_out = int(np.floor((W - K_eff + 2 * p) / s)) + 1
    return W_out, K_eff

# Image: 224x224, Kernel: 7x7, Stride: 2, Padding: 3 (ResNet Conv1)
w_out, k_eff = compute_output_dim(224, 7, p=3, s=2)
print(f"ResNet Conv1: 224x224 -> Output: {w_out}x{w_out}")

# Image: 64x64, Kernel: 3x3, Dilation: 4, Stride: 1, Padding: Same
p_same = (3 + (3 - 1)*(4 - 1) - 1) // 2 # 4
w_out_dilated, k_eff_dilated = compute_output_dim(64, 3, p=4, s=1, d=4)
print(f"Dilated Conv (d=4): Effective Kernel={k_eff_dilated}x{k_eff_dilated} -> Output: {w_out_dilated}x{w_out_dilated}")
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Convolution Output Shape Calculation across Padding Modes",
                "statement": "An input feature map has spatial resolution W_in = 32x32. A convolutional layer uses filter size K = 5x5 with stride s = 1. Calculate the output spatial dimensions for: (a) Valid padding (p = 0), (b) Same padding, and (c) Full padding.",
                "given": "W_in = 32, K = 5, s = 1.",
                "formulas": "W_out = floor((W_in - K + 2*p)/s) + 1; Same: p = (K - 1)/2; Full: p = K - 1",
                "steps": [
                    "Step 1: Valid Padding (p = 0):\n  W_out = (32 - 5 + 2*0)/1 + 1 = 27 + 1 = 28x28.",
                    "Step 2: Same Padding:\n  p = (5 - 1) / 2 = 2 pixels on each side.\n  W_out = (32 - 5 + 2*2)/1 + 1 = (32 - 5 + 4) + 1 = 31 + 1 = 32x32.",
                    "Step 3: Full Padding:\n  p = K - 1 = 5 - 1 = 4 pixels on each side.\n  W_out = (32 - 5 + 2*4)/1 + 1 = (32 - 5 + 8) + 1 = 35 + 1 = 36x36."
                ],
                "answer": "Valid = 28x28, Same = 32x32, Full = 36x36",
                "interpretation": "Valid padding shrinks the map by 4 pixels, Same padding preserves the 32x32 resolution, and Full padding expands it to 36x36."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "KTU Model Paper Q9: Stride Effect & 1x1 Bottleneck Parameter Savings",
                "statement": "Part 1: What happens if the stride of a convolutional layer increases? Part 2: A convolutional layer maps 256 input channels to 256 output channels using 3x3 filters. An alternative Bottleneck design uses: (i) a 1x1 convolution reducing 256 channels to 64, (ii) a 3x3 convolution with 64 channels, and (iii) a 1x1 convolution expanding 64 channels back to 256. Calculate the total weight parameters for both designs and the percentage reduction.",
                "given": "Direct: 256 -> 256 with 3x3. Bottleneck: 256 -> 64 (1x1), 64 -> 64 (3x3), 64 -> 256 (1x1).",
                "formulas": "Params = C_in * C_out * K_h * K_w",
                "steps": [
                    "Part 1: Increasing stride s reduces output spatial resolution by approximately 1/s, reducing memory and computation for subsequent layers while increasing the effective receptive field stride.",
                    "Part 2 (Direct Design):\n  Weights = 256 * 256 * (3 * 3) = 65,536 * 9 = 589,824 parameters.",
                    "Part 3 (Bottleneck Design):\n  Layer 1 (1x1): 256 * 64 * (1 * 1) = 16,384.\n  Layer 2 (3x3): 64 * 64 * (3 * 3) = 4,096 * 9 = 36,864.\n  Layer 3 (1x1): 64 * 256 * (1 * 1) = 16,384.\n  Total Bottleneck Parameters = 16,384 + 36,864 + 16,384 = 69,632 parameters.",
                    "Part 4: Compute savings:\n  Reduction = (589,824 - 69,632) / 589,824 * 100 = (520,192 / 589,824) * 100 = 88.19%."
                ],
                "answer": "Direct = 589,824 parameters. Bottleneck = 69,632 parameters (88.19% parameter reduction).",
                "interpretation": "Using 1x1 bottleneck convolutions cuts parameters by over 88% while preserving representational capacity, which is the cornerstone of modern ResNet and Inception architectures."
            }
        ],
        "real_world_application": {
            "title": "DeepLab Autonomous Driving Semantic Segmentation",
            "domain": "Autonomous Vehicles & Scene Parsing",
            "workflow": "High-res 1080p vehicle camera feed -> Standard pooling loses fine boundary details of distant pedestrians -> Dilated (Atrous) Convolutions expand receptive fields to cover entire intersections without downsampling.",
            "benefit": "Enables pixel-accurate sidewalk, curb, and pedestrian segmentation without spatial detail loss."
        },
        "viva_questions": [
            {"q": "What is the mathematical formula for the output dimension of a 2D convolution?", "a": "Output = floor((Input - Kernel + 2*Padding) / Stride) + 1."},
            {"q": "What is the difference between Valid padding and Same padding?", "a": "Valid padding adds zero zeros (output shrinks by K - 1); Same padding adds enough zeros such that the output spatial resolution matches the input resolution when stride=1."},
            {"q": "What is a 1x1 convolution and what is its primary use?", "a": "A convolution with a 1x1 kernel that computes a linear combination across input channels at each pixel, used for channel dimensionality reduction (bottlenecks)."},
            {"q": "What is a Dilated (Atrous) Convolution?", "a": "A convolution where the filter kernel is expanded with spaces/zeros between weights, enlarging the receptive field without adding parameters."},
            {"q": "What are Locally Connected Layers?", "a": "Layers with local receptive fields like convolution, but without parameter sharing; each spatial location learns its own unique set of weights."},
            {"q": "When are Locally Connected Layers preferred over standard convolutions?", "a": "When the input has non-stationary spatial properties where features at one location have no relevance at another (e.g. aligned face recognition)."},
            {"q": "What happens to the receptive field when dilation rate d increases?", "a": "The receptive field expands exponentially with depth without losing spatial grid resolution."},
            {"q": "How does increasing stride affect the feature map?", "a": "It downsamples the spatial height and width by a factor proportional to the stride, reducing memory and computation."},
            {"q": "What is Tiled Convolution?", "a": "A convolution where weights cycle through a small set of k distinct filters as the kernel moves across the image."},
            {"q": "Can a 1x1 convolution change the spatial height and width of an image?", "a": "No, it operates purely across the channel depth axis, leaving spatial height and width unchanged (when stride=1)."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "To preserve identical spatial input dimensions with kernel size 7x7 and stride 1, the required padding p is:",
                    "options": ["1", "2", "3", "4"],
                    "ans": "3"
                },
                {
                    "q": "The primary purpose of a 1x1 convolution in deep architectures like ResNet is to:",
                    "options": ["Increase spatial height and width", "Perform channel dimensionality reduction (bottleneck projection)", "Smooth image noise", "Perform Fourier inversion"],
                    "ans": "Perform channel dimensionality reduction (bottleneck projection)"
                },
                {
                    "q": "A dilated convolution with kernel size 3x3 and dilation rate d = 2 has an effective receptive field of:",
                    "options": ["3x3", "5x5", "6x6", "7x7"],
                    "ans": "5x5"
                }
            ],
            "short_questions": [
                "Derive the master formula for the output dimension of a 2D convolution and contrast Valid, Same, and Full padding.",
                "Explain the architecture and parameter reduction achieved by 1x1 bottleneck convolutions."
            ],
            "application_question": "Explain how dilated convolutions allow WaveNet to capture long-range temporal dependencies in high-resolution audio speech synthesis.",
            "homework": "Calculate the output spatial dimensions for an input of size 128x128 with kernel 7x7, padding 3, and stride 2."
        }
    },
    {
        "day": 21,
        "module": 3,
        "topic_id": "3.5",
        "title": "Structured Outputs and Diverse Data Types in CNNs",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO3",
        "blooms": "Understand / Apply",
        "prerequisites": "Convolution Operations, Multi-channel Tensors, Day 20 concepts",
        "learning_objectives": [
            "Analyze multi-channel input and output tensors in 2D CNNs (H x W x C).",
            "Examine 1D convolutions for temporal sequential data (audio waveforms, ECG, text).",
            "Examine 3D convolutions for volumetric and spatiotemporal data (CT/MRI medical scans, video).",
            "Formulate Structured Output tasks: Semantic Segmentation masks (pixel-wise classification) and Object Detection bounding boxes."
        ],
        "key_terminology": [
            "Structured Outputs", "Multi-Channel Tensors", "1D Convolution", "3D Volumetric Convolution",
            "Semantic Segmentation", "Object Detection Bounding Box", "Spatiotemporal Feature Learning"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 9 (Sections 9.6 & 9.7); Aggarwal: Chapter 8.",
        "notes": {
            "data_type_taxonomy": (
                "CNNs process multi-dimensional grid data across diverse domains:\n"
                "1. 1D Sequential Grids (Channels x Time):\n"
                "   - Audio waveforms (1D amplitude sampled at 16 kHz or 44.1 kHz).\n"
                "   - Biomedical ECG / EEG sensor telemetry.\n"
                "   - Text sentences (sequence of word embedding vectors).\n\n"
                "2. 2D Spatial Grids (Height x Width x Channels):\n"
                "   - Standard RGB color photographs (H x W x 3).\n"
                "   - Audio spectrograms (Time x Frequency x 1).\n"
                "   - Multispectral satellite imagery (e.g. 13 spectral bands in Sentinel-2).\n\n"
                "3. 3D Volumetric & Spatiotemporal Grids (Height x Width x Depth/Time x Channels):\n"
                "   - Medical imaging: 3D MRI and CT volumetric anatomical voxel scans.\n"
                "   - Video clips: 2D spatial frames concatenated over time (H x W x T x 3).\n"
                "   - Computational fluid dynamics and weather simulation grids."
            ),
            "structured_outputs": (
                "Rather than outputting a single categorical scalar y in {1, ..., K}, structured prediction tasks output high-dimensional structured arrays:\n\n"
                "1. Semantic Segmentation (Dense Pixel Classification):\n"
                "   - Output is a 2D label map S of shape (H x W x C) where every individual pixel is assigned a class probability distribution.\n"
                "   - Architecture: Fully Convolutional Networks (FCN) and U-Net using encoder-decoder paths and transposed convolutions (skip connections).\n\n"
                "2. Object Detection Bounding Boxes:\n"
                "   - Output combines classification probabilities with regression coordinates [x_center, y_center, width, height] for multiple bounding boxes (e.g. YOLO, Faster R-CNN)."
            ),
            "exam_important_points": [
                "Dimensional structure of 1D, 2D, and 3D convolutional tensors.",
                "How 3D convolutions extract spatiotemporal features from video clips.",
                "Difference between image classification and dense semantic segmentation.",
                "U-Net architecture with skip connections for medical image segmentation."
            ]
        },
        "teaching_script": {
            "00-05": "Show an MRI volumetric scan and a video clip. Ask: 'How does convolution generalize beyond 2D flat photographs?'",
            "05-20": "Map out the dimensional hierarchy: 1D (audio/ECG), 2D (RGB/spectrograms), 3D (MRI/video). Write the 3D convolution equation.",
            "20-35": "Introduce Structured Outputs: contrast single-label classification with dense semantic segmentation and bounding box regression.",
            "35-45": "Explain the U-Net architecture: contracting encoder path, bottleneck, expanding decoder path, and skip connections.",
            "45-55": "Demonstrate Python script performing 1D convolution on an ECG signal and 3D tensor convolution.",
            "55-60": "Recap and preview Day 22: Efficient Convolution Algorithms (MobileNet, Winograd, FFT)."
        },
        "python_demo": {
            "title": "1D ECG Convolution and 3D Volumetric Convolution in NumPy",
            "description": "Demonstrating 1D convolution on temporal signal and 3D convolution on a volumetric video tensor.",
            "code": """import numpy as np

# 1. 1D Convolution on temporal ECG-like signal
signal = np.array([0.0, 0.1, 0.2, 1.5, -0.8, 0.1, 0.0, 0.0])
kernel_1d = np.array([-1.0, 2.0, -1.0]) # peak detector
out_1d = np.convolve(signal, kernel_1d, mode='valid')
print("1D Signal Peak Detection Output:\\n", np.round(out_1d, 2))

# 2. 3D Convolution on video tensor (Time x Height x Width)
# 4 frames of 5x5 video
video = np.random.randn(4, 5, 5)
# 3D spatio-temporal kernel: 2 frames x 3x3 spatial
kernel_3d = np.random.randn(2, 3, 3)

out_t = 4 - 2 + 1 # 3
out_h = 5 - 3 + 1 # 3
out_w = 5 - 3 + 1 # 3
out_3d = np.zeros((out_t, out_h, out_w))

for t in range(out_t):
    for i in range(out_h):
        for j in range(out_w):
            patch = video[t:t+2, i:i+3, j:j+3]
            out_3d[t, i, j] = np.sum(patch * kernel_3d)

print(f"\\n3D Video Input Shape: {video.shape}")
print(f"3D Kernel Shape:      {kernel_3d.shape}")
print(f"3D Output Tensor:     {out_3d.shape} (Spatiotemporal feature map!)")
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "3D Volumetric Convolution Output Dimension and Parameter Count",
                "statement": "A medical MRI brain scan has volumetric resolution 128x128x64 voxels (Height x Width x Depth) with 1 input channel. A 3D convolutional layer applies 32 filters of size 3x3x3 with stride s = 1 and no padding (valid). (a) Calculate the output volumetric dimensions, and (b) Calculate the total number of trainable weight and bias parameters in the layer.",
                "given": "Input: 128x128x64, C_in = 1, Filters = 32, K = 3x3x3, s = 1, p = 0.",
                "formulas": "Out = floor((Dim - K + 2*p)/s) + 1; Weights = N_filters * C_in * (K_h * K_w * K_d); Biases = N_filters",
                "steps": [
                    "Step 1: Compute output dimensions:\n  H_out = 128 - 3 + 1 = 126.\n  W_out = 128 - 3 + 1 = 126.\n  D_out = 64 - 3 + 1 = 62.\n  Output volume = 126 x 126 x 62 with 32 channels.",
                    "Step 2: Compute weights:\n  Weights = 32 * 1 * (3 * 3 * 3) = 32 * 27 = 864 weights.",
                    "Step 3: Compute biases = 32.\n  Total parameters = 864 + 32 = 896 parameters."
                ],
                "answer": "Output dimensions = 126 x 126 x 62 (32 channels); Total parameters = 896",
                "interpretation": "3D convolutions slide across height, width, and depth simultaneously, capturing volumetric anatomical spatial relationships."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Memory Footprint in Dense Semantic Segmentation vs Image Classification",
                "statement": "An image with resolution 512x512x3 is processed for: (Case A) 1000-class Image Classification using Global Average Pooling, and (Case B) 20-class Semantic Segmentation producing a full-resolution dense prediction map. Both outputs use 32-bit floats. Calculate: (a) Output tensor dimensions for Case A and Case B, (b) Output memory consumption for a single image in bytes and kilobytes, and (c) Discuss the gradient backpropagation memory challenge in dense segmentation.",
                "given": "Image: 512x512x3. Case A: 1000 classes. Case B: 512x512x20 classes. Float32 (4 bytes).",
                "formulas": "Memory = Total_Elements * 4 bytes",
                "steps": [
                    "Step 1: Case A (Classification):\n  Output shape = 1 x 1000.\n  Elements = 1000.\n  Memory = 1000 * 4 bytes = 4,000 bytes = 3.91 KB.",
                    "Step 2: Case B (Semantic Segmentation):\n  Output shape = 512 x 512 x 20.\n  Elements = 512 * 512 * 20 = 262,144 * 20 = 5,242,880 elements.\n  Memory = 5,242,880 * 4 bytes = 20,971,520 bytes = 20.0 MB!",
                    "Step 3: Comparison ratio:\n  Ratio = 20,971,520 / 4,000 = 5,242.88x more output memory.",
                    "Step 4: Backpropagation implications:\n  Storing full-resolution intermediate feature activations for a batch of 16 segmentation maps requires gigabytes of GPU VRAM, requiring gradient checkpointing or smaller mini-batch sizes."
                ],
                "answer": "Classification = 3.91 KB. Segmentation = 20.0 MB (5,243x larger memory footprint).",
                "interpretation": "Dense structured prediction tasks require immense GPU VRAM to cache high-resolution multi-channel tensors during the backward pass."
            }
        ],
        "real_world_application": {
            "title": "Brain Glioblastoma Tumor Segmentation in 3D MRI",
            "domain": "Medical Radiology & Neurosurgical Navigation",
            "workflow": "Patient multi-parametric 3D MRI scans (T1, T1-contrast, T2, FLAIR) -> 3D U-Net processes 4-channel volumetric voxels -> Produces voxel-wise segmented 3D tumor boundary mesh.",
            "benefit": "Enables neurosurgeons to achieve millimeter-accurate tumor resection while avoiding critical motor cortex brain tissue."
        },
        "viva_questions": [
            {"q": "What is the difference between image classification and semantic segmentation?", "a": "Classification assigns a single label to the entire image; semantic segmentation assigns a categorical class label to every individual pixel in the image."},
            {"q": "What is a 1D convolution and where is it applied?", "a": "A convolution where the kernel slides across a single spatial or temporal dimension, used in audio waveforms, text sequences, and ECG signals."},
            {"q": "What is a 3D convolution and where is it applied?", "a": "A convolution where a 3D kernel slides across height, width, and depth/time, used in 3D MRI/CT scans and video action recognition."},
            {"q": "What is a U-Net architecture?", "a": "An encoder-decoder architecture with skip connections between corresponding contracting and expanding levels, widely used for biomedical image segmentation."},
            {"q": "Why are skip connections critical in semantic segmentation architectures?", "a": "They bypass deep bottleneck downsampling, transferring high-resolution spatial boundary details directly to the decoder for precise pixel localization."},
            {"q": "What is the shape of an input tensor for a batch of 32 RGB video clips of 16 frames with resolution 112x112?", "a": "(32, 3, 16, 112, 112) or (32, 16, 112, 112, 3) depending on channel ordering."},
            {"q": "What loss function is commonly used for training semantic segmentation models?", "a": "Pixel-wise Cross-Entropy Loss combined with Dice Loss or Focal Loss to handle class imbalance."},
            {"q": "Can audio spectrograms be processed using standard 2D CNNs?", "a": "Yes; time-frequency spectrograms have a 2D grid structure where time is the horizontal axis and frequency is the vertical axis."},
            {"q": "What is a Transposed Convolution (Deconvolution)?", "a": "An upsampling operation with learnable weights used in the decoder of segmentation networks to expand spatial resolution back to input dimensions."},
            {"q": "What is instance segmentation?", "a": "A structured prediction task that not only identifies the class of each pixel (like semantic segmentation) but also distinguishes between separate individual instances of the same object class."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "Semantic segmentation outputs an array of shape:",
                    "options": ["1 x 1", "Batch x Classes", "Height x Width x Classes", "Height x Width x 1"],
                    "ans": "Height x Width x Classes"
                },
                {
                    "q": "Which convolution type is designed to capture spatiotemporal motion features across sequential video frames?",
                    "options": ["1D Convolution", "2D Convolution", "3D Convolution", "Dilated 1D Convolution"],
                    "ans": "3D Convolution"
                },
                {
                    "q": "In U-Net, skip connections are used to:",
                    "options": ["Reduce model parameters", "Transfer fine spatial resolution features directly from encoder to decoder", "Accelerate Adam convergence", "Implement dropout"],
                    "ans": "Transfer fine spatial resolution features directly from encoder to decoder"
                }
            ],
            "short_questions": [
                "Contrast the tensor structures and operations of 1D, 2D, and 3D convolutions.",
                "Explain the architecture of U-Net and discuss the role of skip connections in dense segmentation."
            ],
            "application_question": "Explain how 3D convolutions are applied in autonomous driving systems to detect moving pedestrians across sequential radar-camera point clouds.",
            "homework": "Calculate the total parameters in a 3D convolutional layer with 64 filters of size 3x3x3 operating on an input with 8 channels."
        }
    },
    {
        "day": 22,
        "module": 3,
        "topic_id": "3.6",
        "title": "Efficient Convolution Algorithms",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO3",
        "blooms": "Understand / Apply",
        "prerequisites": "Convolution Operations, Matrix Multiplication, Complexity Theory, Day 21 concepts",
        "learning_objectives": [
            "Analyze computational bottlenecks in standard 2D convolution.",
            "Formulate Depthwise Separable Convolutions (Howard et al., MobileNet) and derive their 8-9x computational savings.",
            "Formulate FFT-based Convolutions based on the Convolution Theorem: O(N^2) -> O(N log N).",
            "Explain the Winograd Minimal Filtering Algorithm F(2x2, 3x3) and its 2.25x multiplication reduction."
        ],
        "key_terminology": [
            "Efficient Convolutions", "Depthwise Separable Convolution", "Pointwise Convolution",
            "Fast Fourier Transform (FFT)", "Convolution Theorem", "Winograd Algorithm", "MobileNet"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 9 (Section 9.8); Lavin & Gray: Fast Algorithms for Convolutional Neural Networks (CVPR 2016).",
        "notes": {
            "computational_bottleneck": (
                "Standard 2D Convolution on an input of size D_F x D_F with M channels, producing N channels with kernel size D_K x D_K requires:\n"
                "Total Multiplications = D_F * D_F * M * N * D_K * D_K\n"
                "For a 56x56 feature map with M=256, N=512, and 3x3 kernel: 3.71 BILLION multiplications per forward pass! This is completely prohibitive on mobile edge hardware."
            ),
            "depthwise_separable_convolutions": (
                "MobileNet (Howard et al., 2017) factorizes standard convolution into two independent steps:\n"
                "1. Depthwise Convolution: Applies a single spatial filter per input channel (M filters of size D_K x D_K x 1).\n"
                "   Cost = D_F * D_F * M * D_K * D_K\n"
                "2. Pointwise Convolution: A 1x1 convolution combining channels linearly across depth (N filters of size 1 x 1 x M).\n"
                "   Cost = D_F * D_F * M * N\n\n"
                "Total Cost = D_F * D_F * M * (D_K^2 + N)\n\n"
                "Computation Reduction Ratio:\n"
                "Ratio = [ D_F^2 * M * (D_K^2 + N) ] / [ D_F^2 * M * N * D_K^2 ]\n"
                "      = (D_K^2 + N) / (N * D_K^2) = (1 / N) + (1 / D_K^2)\n"
                "For standard 3x3 filters (D_K = 3) and large N (e.g. N = 512):\n"
                "Ratio approx 1 / 3^2 = 1 / 9 approx 0.111 (8 to 9 times faster, an 88-89% computational reduction with negligible accuracy loss!)."
            ),
            "fft_and_winograd_algorithms": (
                "1. Fast Fourier Transform (FFT) Convolution:\n"
                "   By the Convolution Theorem: Convolution in the spatial domain equals point-wise multiplication in the frequency domain:\n"
                "   I * K = IFFT( FFT(I) odot FFT(K) )\n"
                "   Complexity drops from O(N_pixels * K^2) to O(N_pixels * log N_pixels). Highly efficient for large kernels (K >= 7x7).\n\n"
                "2. Winograd Minimal Filtering Algorithm (F(2x2, 3x3)):\n"
                "   Lavin & Gray (2016) showed that computing 2x2 outputs with a 3x3 kernel requires only 16 multiplications instead of standard 36 multiplications: a 2.25x speedup! Modern GPU libraries (cuDNN) automatically dispatch Winograd for 3x3 convolutions."
            ),
            "exam_important_points": [
                "Full derivation of Depthwise Separable Convolution computational savings ratio: (1/N) + (1/D_K^2).",
                "Two-step decomposition: Depthwise (spatial) + Pointwise (cross-channel).",
                "Convolution Theorem and FFT-based convolution principle.",
                "Winograd algorithm multiplication reduction for 3x3 filters."
            ]
        },
        "teaching_script": {
            "00-05": "Pose the mobile dilemma: 'How does an iPhone run real-time face tracking at 60 fps without burning battery?' Introduce Efficient Convolutions.",
            "05-20": "Analyze standard convolution cost: D_F^2 * M * N * D_K^2. Walk through the MobileNet factorized architecture on blackboard.",
            "20-35": "Derive the MobileNet efficiency ratio: (1/N) + (1/D_K^2). Prove mathematically why 3x3 filters achieve an 8.5x speedup.",
            "35-45": "Explain FFT Convolution via the Convolution Theorem: I * K = IFFT(FFT(I) odot FFT(K)). Explain Winograd F(2, 3) minimal filtering.",
            "45-55": "Demonstrate Python script measuring execution time of standard vs depthwise separable convolution.",
            "55-60": "Recap Module 3. Review all CNN concepts. Announce Module 4: Recurrent Neural Networks."
        },
        "python_demo": {
            "title": "Depthwise Separable vs Standard Convolution in NumPy",
            "description": "Implementation comparing FLOP counts and memory footprints of Standard vs Depthwise Separable Convolution.",
            "code": """import numpy as np

def conv_flops(D_F, M, N, D_K):
    std_flops = D_F * D_F * M * N * D_K * D_K
    dw_flops = D_F * D_F * M * D_K * D_K
    pw_flops = D_F * D_F * M * N
    sep_flops = dw_flops + pw_flops
    ratio = sep_flops / std_flops
    return std_flops, sep_flops, ratio

D_F = 56 # feature map 56x56
M = 128  # 128 input channels
N = 256  # 256 output channels
D_K = 3  # 3x3 filter

std_cost, sep_cost, ratio = conv_flops(D_F, M, N, D_K)

print(f"Standard Conv FLOPs:           {std_cost:12,d}")
print(f"Depthwise Separable Conv FLOPs:{sep_cost:12,d}")
print(f"Efficiency Ratio:               {ratio:.4f} (~{1/ratio:.1f}x speedup!)")
print(f"FLOPs Saved:                    {(1 - ratio)*100:.2f}% reduction!")
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "MobileNet Depthwise Separable FLOP Calculation",
                "statement": "A convolutional layer operates on an input feature map of size 28x28 with M = 64 input channels and produces N = 128 output channels using 3x3 filters (stride = 1, same padding). (a) Calculate the multiplication operations for a standard convolution, (b) Calculate the operations for depthwise convolution, (c) Calculate the operations for pointwise convolution, and (d) Determine the total operations and reduction percentage.",
                "given": "D_F = 28, M = 64, N = 128, D_K = 3.",
                "formulas": "Std = D_F^2 * M * N * D_K^2; DW = D_F^2 * M * D_K^2; PW = D_F^2 * M * N; Total_Sep = DW + PW",
                "steps": [
                    "Step 1: Compute Standard Convolution operations:\n  Std = (28 * 28) * 64 * 128 * (3 * 3) = 784 * 64 * 128 * 9 = 57,802,752 multiplications.",
                    "Step 2: Compute Depthwise Convolution operations:\n  DW = (28 * 28) * 64 * (3 * 3) = 784 * 64 * 9 = 451,584 multiplications.",
                    "Step 3: Compute Pointwise (1x1) Convolution operations:\n  PW = (28 * 28) * 64 * 128 * (1 * 1) = 784 * 64 * 128 * 1 = 6,422,528 multiplications.",
                    "Step 4: Total Separable Operations = 451,584 + 6,422,528 = 6,874,112 multiplications.",
                    "Step 5: Compute reduction percentage:\n  Ratio = 6,874,112 / 57,802,752 = 0.11892 (11.89%).\n  Reduction = (1 - 0.11892) * 100 = 88.11% reduction!"
                ],
                "answer": "Standard = 57.8M ops. Separable = 6.87M ops (88.11% computation reduction, ~8.4x speedup).",
                "interpretation": "Depthwise separable convolution reduces operations from 57.8 million to 6.87 million while retaining identical channel depth and spatial resolution."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Winograd Minimal Filtering Multiplications Reduction Proof",
                "statement": "The Winograd minimal filtering algorithm F(m, r) computes m output values using an r-tap FIR filter with m + r - 1 multiplications, whereas standard direct convolution requires m * r multiplications. For 1D convolution F(2, 3) computing 2 outputs with a 3-tap filter: (a) Compare multiplications required by direct vs Winograd convolution, (b) Extend to 2D convolution F(2x2, 3x3) which computes 2x2 outputs using a 3x3 filter, and (c) Calculate the exact theoretical speedup factor achieved by Winograd on modern GPU engines.",
                "given": "1D: m = 2, r = 3; 2D: F(2x2, 3x3).",
                "formulas": "1D Direct = m * r; 1D Winograd = m + r - 1; 2D Direct = m^2 * r^2; 2D Winograd = (m + r - 1)^2",
                "steps": [
                    "Step 1: 1D Convolution:\n  Direct multiplications = m * r = 2 * 3 = 6 multiplications.\n  Winograd multiplications = m + r - 1 = 2 + 3 - 1 = 4 multiplications.\n  1D savings = 6 / 4 = 1.5x speedup.",
                    "Step 2: 2D Convolution F(2x2, 3x3):\n  Direct multiplications = m^2 * r^2 = (2^2) * (3^2) = 4 * 9 = 36 multiplications.\n  Winograd multiplications = (m + r - 1)^2 = (2 + 3 - 1)^2 = 4^2 = 16 multiplications.",
                    "Step 3: Calculate 2D speedup factor:\n  Speedup = 36 / 16 = 2.25x speedup!\n  Multiplication reduction = ((36 - 16) / 36) * 100 = 55.56% reduction."
                ],
                "answer": "1D Winograd saves 33.3% (1.5x). 2D Winograd reduces multiplications from 36 to 16 (2.25x speedup, 55.56% reduction).",
                "interpretation": "Winograd minimal filtering cuts 3x3 convolution multiplications by more than half, explaining why NVIDIA cuDNN utilizes Winograd by default for 3x3 kernels."
            }
        ],
        "real_world_application": {
            "title": "Smart Home Edge Camera Face Unlock",
            "domain": "Consumer Electronics & Edge Computer Vision",
            "workflow": "Battery-powered doorbell camera detects motion -> Microcontroller runs MobileNet with depthwise separable convolutions -> Evaluates live facial biometrics in 30ms on a 1-Watt edge processor.",
            "benefit": "Enables instant biometric door unlock without draining battery or sending private home video streams to the cloud."
        },
        "viva_questions": [
            {"q": "What is a Depthwise Separable Convolution?", "a": "A factorized convolution that breaks the standard operation into two steps: a depthwise spatial convolution applied independently per channel, followed by a 1x1 pointwise convolution across channels."},
            {"q": "What is the computational reduction factor achieved by MobileNet depthwise separable convolution?", "a": "(1 / N) + (1 / D_K^2); for 3x3 filters, this yields an 8 to 9-fold computational speedup."},
            {"q": "How does the Convolution Theorem enable fast convolution via FFT?", "a": "Convolution in the spatial domain corresponds to point-wise multiplication in the frequency domain: I * K = IFFT(FFT(I) odot FFT(K))."},
            {"q": "For what kernel sizes is FFT convolution faster than direct convolution?", "a": "For large kernels (typically K >= 7x7 or 11x11); for small 3x3 kernels, FFT transform overhead exceeds direct computation."},
            {"q": "What is the Winograd minimal filtering algorithm?", "a": "An algebraic algorithm that computes small convolutions with the minimum possible number of multiplications, speeding up 3x3 convolutions by 2.25x."},
            {"q": "What is Pointwise Convolution?", "a": "A 1x1 convolution that linearly combines features across input channels without changing spatial resolution."},
            {"q": "Why is Winograd convolution used in NVIDIA cuDNN?", "a": "Because 3x3 convolutions dominate modern deep networks (VGG, ResNet); Winograd cuts 3x3 multiplications from 36 to 16, maximizing GPU tensor throughput."},
            {"q": "Does depthwise separable convolution reduce parameter count as well as FLOPs?", "a": "Yes, parameters are reduced by the exact same ratio of approximately 8-9x for 3x3 filters."},
            {"q": "What is ShuffleNet?", "a": "An efficient architecture that replaces 1x1 pointwise convolutions with pointwise group convolutions and channel shuffle operations to further reduce compute."},
            {"q": "Why are efficient convolution algorithms critical for Edge AI?", "a": "Because edge devices (smartphones, IoT sensors, drones) have strict battery, thermal, and memory bandwidth constraints that cannot support billions of standard FLOPs."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "Depthwise Separable Convolution divides standard convolution into:",
                    "options": ["Fast Fourier and Inverse Fourier transforms", "Depthwise spatial convolution followed by 1x1 Pointwise convolution", "Max pooling followed by average pooling", "L1 regularization and L2 weight decay"],
                    "ans": "Depthwise spatial convolution followed by 1x1 Pointwise convolution"
                },
                {
                    "q": "For a 3x3 filter, Depthwise Separable Convolution reduces multiplications by approximately:",
                    "options": ["2x", "4x", "8 to 9x", "50x"],
                    "ans": "8 to 9x"
                },
                {
                    "q": "The Winograd algorithm F(2x2, 3x3) reduces multiplications for 2x2 outputs from 36 to:",
                    "options": ["8", "16", "24", "32"],
                    "ans": "16"
                }
            ],
            "short_questions": [
                "Derive the computational complexity and speedup ratio of Depthwise Separable Convolution compared to standard convolution.",
                "Explain the principle of FFT-based convolution and state the conditions under which it outperforms direct spatial convolution."
            ],
            "application_question": "Explain how MobileNet depthwise separable blocks enable real-time augmented reality filters on mobile smartphone processors.",
            "homework": "Calculate the exact speedup ratio of depthwise separable convolution for an input with M = 512 channels, N = 1024 channels, and a 5x5 filter kernel."
        }
    }
]
