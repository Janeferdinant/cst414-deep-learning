"""
KTU CST414 Deep Learning: Module 3 Comprehensive Demonstrations
Faculty: Jane Rose | Dept of CSE
"""

import numpy as np
import matplotlib.pyplot as plt

# ======================================================================
# Day 17 (Topic 3.1): 2D Image Convolution from Scratch with NumPy
# ======================================================================

def demo_day_17():
    print("\n--- DEMO DAY 17: 2D Image Convolution from Scratch with NumPy ---")
    import numpy as np
    
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
    print("Convolved Feature Map (3x3):\n", feature_map)
    

# ======================================================================
# Day 18 (Topic 3.2): Max Pooling and Translation Invariance in NumPy
# ======================================================================

def demo_day_18():
    print("\n--- DEMO DAY 18: Max Pooling and Translation Invariance in NumPy ---")
    import numpy as np
    
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
    
    print("Pooled A:\n", max_pool2d(A))
    print("Pooled B (Shifted):\n", max_pool2d(B))
    

# ======================================================================
# Day 19 (Topic 3.3): Permuted Pixels Experiment: MLP vs CNN Inductive Bias
# ======================================================================

def demo_day_19():
    print("\n--- DEMO DAY 19: Permuted Pixels Experiment: MLP vs CNN Inductive Bias ---")
    import numpy as np
    
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
    print("\nCNN assumes high local spatial correlation (Infinitely Strong Prior).")
    print("Permutation destroys this correlation, causing CNN performance collapse!")
    

# ======================================================================
# Day 20 (Topic 3.4): Convolution Output Dimension Calculator & Dilated Kernel in NumPy
# ======================================================================

def demo_day_20():
    print("\n--- DEMO DAY 20: Convolution Output Dimension Calculator & Dilated Kernel in NumPy ---")
    import numpy as np
    
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
    

# ======================================================================
# Day 21 (Topic 3.5): 1D ECG Convolution and 3D Volumetric Convolution in NumPy
# ======================================================================

def demo_day_21():
    print("\n--- DEMO DAY 21: 1D ECG Convolution and 3D Volumetric Convolution in NumPy ---")
    import numpy as np
    
    # 1. 1D Convolution on temporal ECG-like signal
    signal = np.array([0.0, 0.1, 0.2, 1.5, -0.8, 0.1, 0.0, 0.0])
    kernel_1d = np.array([-1.0, 2.0, -1.0]) # peak detector
    out_1d = np.convolve(signal, kernel_1d, mode='valid')
    print("1D Signal Peak Detection Output:\n", np.round(out_1d, 2))
    
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
    
    print(f"\n3D Video Input Shape: {video.shape}")
    print(f"3D Kernel Shape:      {kernel_3d.shape}")
    print(f"3D Output Tensor:     {out_3d.shape} (Spatiotemporal feature map!)")
    

# ======================================================================
# Day 22 (Topic 3.6): Depthwise Separable vs Standard Convolution in NumPy
# ======================================================================

def demo_day_22():
    print("\n--- DEMO DAY 22: Depthwise Separable vs Standard Convolution in NumPy ---")
    import numpy as np
    
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
    


if __name__ == '__main__':
    demo_day_17()
    demo_day_18()
    demo_day_19()
    demo_day_20()
    demo_day_21()
    demo_day_22()
