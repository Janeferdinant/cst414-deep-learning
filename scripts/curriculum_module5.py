"""
Curriculum Data for Module 5: Application Areas (9 Hours)
KTU CST414 Deep Learning (Scheme 2019)
Faculty: Jane Rose | Department of Computer Science & Engineering
"""

MODULE_5_DAYS = [
    {
        "day": 28,
        "module": 5,
        "topic_id": "5.1",
        "title": "Deep Learning Applications: Computer Vision",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO5",
        "blooms": "Understand / Apply",
        "prerequisites": "CNN fundamentals, Convolutions, Pooling, Day 18 & 21 concepts",
        "learning_objectives": [
            "Survey the foundational Computer Vision tasks: Classification, Localization, Detection, and Segmentation.",
            "Analyze Two-Stage Object Detectors (R-CNN, Fast R-CNN, Faster R-CNN) vs One-Stage Detectors (YOLO, SSD).",
            "Formulate Intersection over Union (IoU), Non-Maximum Suppression (NMS), and Mean Average Precision (mAP).",
            "Examine modern vision backbones: ResNet, EfficientNet, and Vision Transformers (ViT)."
        ],
        "key_terminology": [
            "Computer Vision", "Object Detection", "Two-Stage Detectors", "Single-Stage Detectors (YOLO)",
            "Intersection over Union (IoU)", "Non-Maximum Suppression (NMS)", "Mean Average Precision (mAP)"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 12 (Section 12.2); Aggarwal: Chapter 8.",
        "notes": {
            "computer_vision_taxonomy": (
                "Deep learning has transformed computer vision across four primary task tiers:\n"
                "1. Image Classification: Predict a single discrete category label for the entire image (e.g. ResNet on ImageNet 1k).\n"
                "2. Object Localization: Predict class label AND a single bounding box [x_min, y_min, x_max, y_max].\n"
                "3. Object Detection: Identify and localize ALL object instances across multiple categories with multiple bounding boxes.\n"
                "4. Semantic & Instance Segmentation: Pixel-level dense category assignment (Semantic) or per-object mask extraction (Instance, e.g. Mask R-CNN)."
            ),
            "object_detection_architectures": (
                "1. Two-Stage Detectors (Faster R-CNN, Ren et al., 2015):\n"
                "- Stage 1: Region Proposal Network (RPN) slides across convolutional feature maps to generate candidate object proposals with anchor boxes.\n"
                "- Stage 2: RoI Pooling/Align extracts fixed-size feature vectors per proposal for classification and bounding box regression.\n"
                "- Advantage: High precision and small object recall.\n\n"
                "2. One-Stage Detectors (YOLO - Redmon et al., 2016; SSD):\n"
                "- Divides image into an S x S spatial grid. Each grid cell directly predicts B bounding boxes, objectness confidences, and C class probabilities in a single forward pass.\n"
                "- Advantage: Ultra-fast real-time inference (45 to 150+ fps), enabling live edge deployment."
            ),
            "evaluation_metrics": (
                "1. Intersection over Union (IoU):\n"
                "   IoU = Area(Box_pred cap Box_gt) / Area(Box_pred cup Box_gt)\n"
                "   A prediction is counted as True Positive if IoU >= threshold (typically 0.50 or 0.75).\n\n"
                "2. Non-Maximum Suppression (NMS):\n"
                "   Sorts candidate boxes by confidence score and iteratively suppresses lower-scoring overlapping boxes with IoU > threshold to eliminate redundant detections.\n\n"
                "3. Mean Average Precision (mAP):\n"
                "   Computes the area under the Precision-Recall curve for each class across IoU thresholds, averaged across all categories."
            ),
            "exam_important_points": [
                "Contrast Two-Stage (Faster R-CNN) with One-Stage (YOLO) detectors.",
                "Mathematical formulation and manual calculation of Intersection over Union (IoU).",
                "Non-Maximum Suppression (NMS) algorithm steps.",
                "Definitions of Precision, Recall, and Mean Average Precision (mAP)."
            ]
        },
        "teaching_script": {
            "00-05": "Show an autonomous driving dashcam feed with bounding boxes around cars, pedestrians, and cyclists.",
            "05-20": "Differentiate Classification vs Localization vs Detection vs Segmentation. Draw the bounding box coordinate system.",
            "20-35": "Contrast Two-Stage Detectors (Faster R-CNN RPN) with One-Stage Detectors (YOLO grid). Explain why YOLO achieves 60+ fps.",
            "35-45": "Derive Intersection over Union (IoU) mathematically on blackboard. Walk through Non-Maximum Suppression (NMS) steps.",
            "45-55": "Demonstrate Python script computing IoU between two overlapping bounding boxes and running NMS.",
            "55-60": "Recap and preview Day 29: Deep Learning for Speech Recognition."
        },
        "python_demo": {
            "title": "Intersection over Union (IoU) and Non-Maximum Suppression (NMS)",
            "description": "Pure NumPy implementation of IoU calculation and greedy NMS box filtering.",
            "code": """import numpy as np

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

print("\\nBoxes kept after NMS (threshold=0.5):", len(nms(boxes)), "distinct objects identified!")
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Manual Intersection over Union (IoU) Calculation",
                "statement": "Two bounding boxes are defined on an image grid:\nGround Truth Box A: [x1 = 20, y1 = 20, x2 = 80, y2 = 80]\nPredicted Box B:    [x1 = 40, y1 = 40, x2 = 100, y2 = 100]\nCalculate: (a) Area of Box A, (b) Area of Box B, (c) Area of Intersection, (d) Area of Union, and (e) The exact Intersection over Union (IoU).",
                "given": "Box A = [20, 20, 80, 80], Box B = [40, 40, 100, 100].",
                "formulas": "Area = (x2 - x1) * (y2 - y1); Inter = max(0, x_right - x_left) * max(0, y_bottom - y_top); Union = Area_A + Area_B - Inter; IoU = Inter / Union",
                "steps": [
                    "Step 1: Compute Area of Box A:\n  Width_A = 80 - 20 = 60; Height_A = 80 - 20 = 60.\n  Area_A = 60 * 60 = 3,600 sq pixels.",
                    "Step 2: Compute Area of Box B:\n  Width_B = 100 - 40 = 60; Height_B = 100 - 40 = 60.\n  Area_B = 60 * 60 = 3,600 sq pixels.",
                    "Step 3: Compute Intersection Coordinates:\n  x_left = max(20, 40) = 40.\n  y_top = max(20, 40) = 40.\n  x_right = min(80, 100) = 80.\n  y_bottom = min(80, 100) = 80.\n  Inter_width = 80 - 40 = 40; Inter_height = 80 - 40 = 40.\n  Area_Inter = 40 * 40 = 1,600 sq pixels.",
                    "Step 4: Compute Area of Union:\n  Area_Union = Area_A + Area_B - Area_Inter = 3,600 + 3,600 - 1,600 = 5,600 sq pixels.",
                    "Step 5: Compute IoU:\n  IoU = 1,600 / 5,600 = 16 / 56 = 2 / 7 = 0.2857."
                ],
                "answer": "Area_A = 3600, Area_B = 3600, Inter = 1600, Union = 5600, IoU = 0.2857 (28.57%)",
                "interpretation": "Since IoU = 0.2857 is below the standard 0.50 threshold, this prediction is classified as a False Positive in detection benchmark scoring."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Mean Average Precision (mAP) 11-Point Interpolation Calculation",
                "statement": "An object detector outputs 5 ranked predictions for a class with 4 ground truth instances. Ranked detections have precision and recall values as follows:\nRank 1: Precision = 1.00, Recall = 0.25 (TP)\nRank 2: Precision = 0.50, Recall = 0.25 (FP)\nRank 3: Precision = 0.67, Recall = 0.50 (TP)\nRank 4: Precision = 0.50, Recall = 0.50 (FP)\nRank 5: Precision = 0.60, Recall = 0.75 (TP)\nUsing the standard Pascal VOC 11-point interpolation method at recall levels r in {0.0, 0.1, 0.2, ..., 1.0} where p_interp(r) = max_{r_tilde >= r} p(r_tilde): Calculate the Average Precision (AP).",
                "given": "4 ground truth objects, 5 ranked detections with (P, R) pairs.",
                "formulas": "AP_11 = (1 / 11) * sum_{r in {0.0, 0.1, ..., 1.0}} p_interp(r)",
                "steps": [
                    "Step 1: Maximum precision values achievable for r_tilde >= r:\n  For r in [0.0, 0.25]: Max precision = max(1.00, 0.67, 0.60) = 1.00.\n  For r in (0.25, 0.50]: Max precision = max(0.67, 0.60) = 0.67.\n  For r in (0.50, 0.75]: Max precision = 0.60.\n  For r in (0.75, 1.00]: Since max recall reached is 0.75, precision = 0.00.",
                    "Step 2: Evaluate 11 recall thresholds:\n  r = 0.0: p = 1.00\n  r = 0.1: p = 1.00\n  r = 0.2: p = 1.00\n  r = 0.3: p = 0.67\n  r = 0.4: p = 0.67\n  r = 0.5: p = 0.67\n  r = 0.6: p = 0.60\n  r = 0.7: p = 0.60\n  r = 0.8: p = 0.00\n  r = 0.9: p = 0.00\n  r = 1.0: p = 0.00.",
                    "Step 3: Sum interpolated precisions:\n  Sum = 1.00*3 + 0.67*3 + 0.60*2 + 0.00*3 = 3.00 + 2.01 + 1.20 = 6.21.",
                    "Step 4: Compute Average Precision:\n  AP = 6.21 / 11 = 0.5645."
                ],
                "answer": "11-point interpolated Average Precision AP = 0.5645 (56.45%)",
                "interpretation": "The 11-point interpolation method smooths out precision fluctuations, penalizing the model for failing to reach 100% recall (precision drops to 0 at r >= 0.8)."
            }
        ],
        "real_world_application": {
            "title": "Autonomous Highway Collision Avoidance & Emergency Braking",
            "domain": "Automotive Safety & Self-Driving Vehicles",
            "workflow": "Triple front-facing camera array streams 60 fps 4K video -> Real-time YOLOv8 detector localizes oncoming vehicles, lane dividers, and pedestrians -> IoU tracker associates objects across frames -> Triggers automatic emergency braking if collision time < 1.2 seconds.",
            "benefit": "Slashes forward vehicle collision accidents by 73% in poor visibility rain and fog conditions."
        },
        "viva_questions": [
            {"q": "What is the primary difference between image classification and object detection?", "a": "Classification identifies what object is in the image; detection identifies what objects are present AND where they are located using bounding box coordinates."},
            {"q": "What is Intersection over Union (IoU)?", "a": "The ratio of the area of overlap between the predicted box and ground truth box to the total area of their union."},
            {"q": "What is the typical IoU threshold for counting a detection as a True Positive?", "a": "Typically IoU >= 0.50 (or 0.75 for strict benchmark scoring like COCO mAP@[.5:.95])."},
            {"q": "What is Non-Maximum Suppression (NMS)?", "a": "A post-processing algorithm that eliminates redundant, overlapping bounding boxes by iteratively keeping the highest scoring box and discarding neighboring boxes with IoU above a threshold."},
            {"q": "What is the key architectural difference between Faster R-CNN and YOLO?", "a": "Faster R-CNN is a two-stage detector (Region Proposal Network followed by classifier); YOLO is a single-stage detector that predicts boxes and classes directly from feature grids in one forward pass."},
            {"q": "Why is YOLO significantly faster than two-stage detectors?", "a": "Because it processes the entire image in a single unified neural network pass without requiring a separate, sequential region proposal extraction stage."},
            {"q": "What are Anchor Boxes in object detection?", "a": "Pre-defined bounding boxes of various aspect ratios and scales placed across grid cells to serve as reference priors for regression offsets."},
            {"q": "What is Mean Average Precision (mAP)?", "a": "The mean of the Average Precision (AP) scores computed across all object categories in a detection dataset."},
            {"q": "What is a Feature Pyramid Network (FPN)?", "a": "An architecture that combines low-resolution, semantically strong features with high-resolution, semantically weak features via top-down pathways to detect objects at multiple scales."},
            {"q": "What is Mask R-CNN?", "a": "An extension of Faster R-CNN that adds a parallel branch for predicting pixel-level binary masks, enabling Instance Segmentation."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "The algorithm used to eliminate duplicate, overlapping bounding boxes around the same object is:",
                    "options": ["Batch Normalization", "Non-Maximum Suppression (NMS)", "Stochastic Gradient Descent", "Backpropagation"],
                    "ans": "Non-Maximum Suppression (NMS)"
                },
                {
                    "q": "YOLO is classified as which type of object detector?",
                    "options": ["Two-Stage Detector", "One-Stage (Single-Shot) Detector", "Recurrent Detector", "Bayesian Detector"],
                    "ans": "One-Stage (Single-Shot) Detector"
                },
                {
                    "q": "If the intersection of two bounding boxes is 500 sq pixels and their union is 2000 sq pixels, the IoU is:",
                    "options": ["0.25", "0.50", "0.75", "0.10"],
                    "ans": "0.25"
                }
            ],
            "short_questions": [
                "Formulate Intersection over Union (IoU) and describe how it is used to evaluate object detection models.",
                "Compare the operational paradigms of Faster R-CNN and YOLO in terms of accuracy and inference speed."
            ],
            "application_question": "Explain how an object detector is deployed on automated warehouse drones to perform real-time inventory barcode tracking.",
            "homework": "Given two boxes Box1: [0, 0, 10, 10] and Box2: [5, 5, 15, 15], compute the exact IoU."
        }
    },
    {
        "day": 29,
        "module": 5,
        "topic_id": "5.2",
        "title": "Deep Learning Applications: Speech Recognition",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO5",
        "blooms": "Understand / Apply",
        "prerequisites": "Audio Signal Processing, Recurrent Networks, CTC loss intuition, Day 23 & 26 concepts",
        "learning_objectives": [
            "Analyze the speech recognition pipeline: Audio waveforms, Mel Filterbanks, Acoustic Models, Language Models.",
            "Formulate Connectionist Temporal Classification (CTC Loss) (Graves et al., 2006) for unaligned sequence training.",
            "Examine End-to-End Speech Recognition architectures (Listen Attend Spell, Conformer).",
            "Calculate Word Error Rate (WER) and Character Error Rate (CER) evaluation metrics."
        ],
        "key_terminology": [
            "Automatic Speech Recognition (ASR)", "Mel Frequency Cepstral Coefficients (MFCC)",
            "Mel Filterbank", "Connectionist Temporal Classification (CTC)", "Blank Token", "Word Error Rate (WER)"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 12 (Section 12.3); Graves et al.: Connectionist Temporal Classification (ICML 2006).",
        "notes": {
            "speech_representation": (
                "Speech is an analog acoustic pressure wave sampled at high frequencies (16 kHz = 16,000 samples/sec).\n\n"
                "Acoustic Feature Extraction:\n"
                "1. Short-Time Fourier Transform (STFT): Apply a Hamming window of 25ms with 10ms hop size to extract local frequency spectrum.\n"
                "2. Mel Scale Filterbank: Warps linear frequency into the perceptual Mel scale, matching human ear sensitivity (dense filters at low frequencies, wide filters at high frequencies):\n"
                "   m = 2595 * log10(1 + f / 700)\n"
                "   Yields an 80-channel log-Mel spectrogram tensor (Time x 80)."
            ),
            "connectionist_temporal_classification": (
                "Challenge: Audio frames T (e.g. 1,000 frames) far exceed target character length U (e.g. 50 characters), and exact frame-level alignments are unknown.\n\n"
                "CTC Solution (Graves et al., 2006):\n"
                "1. Introduces a special 'Blank' token epsilon into the vocabulary: V_prime = V cup {epsilon}.\n"
                "2. The network outputs a probability distribution over V_prime at every audio frame t.\n"
                "3. Collapse Mapping B:\n"
                "   Collapses consecutive duplicate characters and then removes blank tokens:\n"
                "   B( 'a' 'a' epsilon 'b' 'b' ) = 'a' 'b'\n"
                "   B( 'c' epsilon 'c' ) = 'c' 'c' (blank allows generating identical consecutive letters!).\n"
                "4. Loss Function:\n"
                "   Sums conditional probabilities across all valid alignment paths pi that map to target y:\n"
                "   P(y | x) = sum_{pi in B^(-1)(y)} product_{t=1}^T P(pi_t | x)\n"
                "   L_CTC = - ln P(y | x), optimized via dynamic programming (forward-backward algorithm)."
            ),
            "evaluation_metrics": (
                "Word Error Rate (WER):\n"
                "WER = (S + D + I) / N\n"
                "where S is Substitutions, D is Deletions, I is Insertions, and N is total ground truth words."
            ),
            "exam_important_points": [
                "The role of the Mel scale filterbank in acoustic preprocessing.",
                "Why CTC loss is necessary for unaligned audio-to-text sequence modeling.",
                "How the CTC blank token enables spelling repeated letters (e.g. 'hello').",
                "Formulation and manual calculation of Word Error Rate (WER)."
            ]
        },
        "teaching_script": {
            "00-05": "Play an audio waveform clip. Ask: 'How does a computer convert air pressure vibrations into written English?' Introduce Speech Recognition.",
            "05-20": "Explain the STFT and Mel filterbank transformation: why human ears perceive pitch logarithmically. Show the 80-band spectrogram.",
            "20-35": "Formulate Connectionist Temporal Classification (CTC). Explain the temporal alignment problem: 1000 audio frames vs 40 letters.",
            "35-45": "Demonstrate the CTC collapse mapping B on blackboard: show how blanks allow repeated characters. Write the WER formula.",
            "45-55": "Demonstrate Python script computing CTC collapse and calculating Word Error Rate with Levenshtein distance.",
            "55-60": "Recap and preview Day 30: Natural Language Processing."
        },
        "python_demo": {
            "title": "CTC Collapse Operator and Word Error Rate (WER) in Python",
            "description": "Implementation of CTC collapse function and Levenshtein dynamic programming WER calculator.",
            "code": """def ctc_collapse(seq, blank='_'):
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
print(f"\\nReference:  '{ref_text}'")
print(f"Hypothesis: '{hyp_text}'")
print(f"WER:        {compute_wer(ref_text, hyp_text):.2%} (1 deletion out of 6 words)")
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Word Error Rate (WER) Calculation",
                "statement": "An Automatic Speech Recognition (ASR) system transcribes a recorded meeting sentence. The reference transcript contains N = 10 words:\nReference:  'the weather in southern kerala is hot and humid today'\nHypothesis: 'the weather in kerala is very hot humid today'\nIdentify the number of Substitutions (S), Deletions (D), and Insertions (I), and calculate the Word Error Rate (WER).",
                "given": "Reference: 10 words. Hypothesis text as given.",
                "formulas": "WER = (S + D + I) / N",
                "steps": [
                    "Step 1: Align reference and hypothesis words:\n  'the'        -> 'the'        (Match)\n  'weather'    -> 'weather'    (Match)\n  'in'         -> 'in'         (Match)\n  'southern'   -> [DELETED]    (Deletion: D = 1)\n  'kerala'     -> 'kerala'     (Match)\n  'is'         -> 'is'         (Match)\n  [NONE]       -> 'very'       (Insertion: I = 1)\n  'hot'        -> 'hot'        (Match)\n  'and'        -> [DELETED]    (Deletion: D = 2)\n  'humid'      -> 'humid'      (Match)\n  'today'      -> 'today'      (Match).",
                    "Step 2: Total error counts:\n  Substitutions (S) = 0.\n  Deletions (D) = 2 ('southern', 'and').\n  Insertions (I) = 1 ('very').\n  Total Errors = 0 + 2 + 1 = 3 errors.",
                    "Step 3: Calculate WER:\n  WER = (S + D + I) / N = 3 / 10 = 0.3000 (30.0%)."
                ],
                "answer": "S = 0, D = 2, I = 1. Total Errors = 3. Word Error Rate (WER) = 30.0%.",
                "interpretation": "Two deleted words and one hallucinated insertion yield an overall word error rate of 30% against the 10-word reference."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Mel Frequency Calculation from Hertz Audio Pitch",
                "statement": "The Mel scale formula relating acoustic physical frequency f in Hertz (Hz) to perceived psychological pitch m in Mels is m = 2595 * log10(1 + f / 700). (a) Calculate the Mel pitch for frequencies f1 = 300 Hz, f2 = 1000 Hz, and f3 = 4000 Hz, (b) Calculate the delta in Hertz between f2 and f1 vs between f3 and f2, and (c) Compare the corresponding deltas on the Mel scale to illustrate human ear non-linearity.",
                "given": "m = 2595 * log10(1 + f / 700). Frequencies: 300 Hz, 1000 Hz, 4000 Hz.",
                "formulas": "m = 2595 * log10(1 + f / 700)",
                "steps": [
                    "Step 1: For f1 = 300 Hz:\n  m1 = 2595 * log10(1 + 300 / 700) = 2595 * log10(1 + 0.42857) = 2595 * log10(1.42857) = 2595 * 0.15490 = 401.97 Mels.",
                    "Step 2: For f2 = 1000 Hz:\n  m2 = 2595 * log10(1 + 1000 / 700) = 2595 * log10(1 + 1.42857) = 2595 * log10(2.42857) = 2595 * 0.38535 = 1000.00 Mels.",
                    "Step 3: For f3 = 4000 Hz:\n  m3 = 2595 * log10(1 + 4000 / 700) = 2595 * log10(1 + 5.71429) = 2595 * log10(6.71429) = 2595 * 0.82700 = 2146.06 Mels.",
                    "Step 4: Hertz spacing vs Mel spacing:\n  Delta_Hz(f2 - f1) = 1000 - 300 = 700 Hz -> Delta_Mel = 1000 - 401.97 = 598.03 Mels.\n  Delta_Hz(f3 - f2) = 4000 - 1000 = 3000 Hz -> Delta_Mel = 2146.06 - 1000 = 1146.06 Mels.\n  In Hertz, interval 2 is 3000/700 = 4.29x wider, but in Mel space it is only 1146/598 = 1.92x wider!"
                ],
                "answer": "m(300Hz) = 402.0 Mels, m(1000Hz) = 1000.0 Mels, m(4000Hz) = 2146.1 Mels.",
                "interpretation": "Human hearing has high perceptual resolution at lower frequencies (under 1 kHz) and compressed logarithmic sensitivity at high frequencies, which the Mel filterbank preserves."
            }
        ],
        "real_world_application": {
            "title": "Real-Time Emergency Dispatch Voice Call Transcription",
            "domain": "Public Safety & Telecommunications",
            "workflow": "Distressed citizen calls 911/112 -> Audio stream contains high acoustic background siren noise -> 80-band Mel filterbank fed into Conformer CTC ASR engine -> Transcribes street address and emergency details in sub-second latency.",
            "benefit": "Enables emergency first responders to dispatch ambulances 45 seconds faster, saving critical lives."
        },
        "viva_questions": [
            {"q": "What is the physical sampling rate of standard high-fidelity voice audio?", "a": "16 kHz (16,000 samples per second) for speech recognition, or 44.1 kHz for studio music."},
            {"q": "Why is the Mel scale preferred over linear frequency in speech recognition?", "a": "The Mel scale matches human auditory perception, which has high resolution for low frequencies and compressed sensitivity for high frequencies."},
            {"q": "What is Connectionist Temporal Classification (CTC)?", "a": "A loss function designed for sequence tasks where the input sequence is longer than the output sequence and exact temporal alignments are unknown."},
            {"q": "What is the function of the blank token epsilon in CTC?", "a": "To indicate silence/transitions and to allow the model to emit repeated characters (e.g. 'b l a n k' vs 'l o o k')."},
            {"q": "How does the CTC collapse function decode 'c c _ a a a _ t t'?", "a": "First it collapses consecutive duplicates: 'c _ a _ t'; then it removes blanks: 'cat'."},
            {"q": "What is Word Error Rate (WER)?", "a": "The standard evaluation metric for speech recognition, defined as (Substitutions + Deletions + Insertions) / Total Reference Words."},
            {"q": "Can Word Error Rate exceed 100%?", "a": "Yes; if the hypothesis contains many hallucinated insertions, (S + D + I) can exceed N, resulting in WER > 1.0."},
            {"q": "What is a Conformer architecture in modern speech recognition?", "a": "A state-of-the-art model (Gulati et al., 2020) that combines Convolutional layers (for local acoustic features) with Self-Attention layers (for global context)."},
            {"q": "What is the difference between an Acoustic Model and a Language Model in ASR?", "a": "The Acoustic Model maps audio spectrograms to phonetic or character probabilities; the Language Model models word sequence probabilities in the target language to correct spelling and syntax."},
            {"q": "What is SpecAugment?", "a": "A simple data augmentation technique for speech that masks out random vertical frequency bands and horizontal time blocks on the spectrogram."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "The CTC collapse operator maps the sequence ['c', 'c', '_', 'o', 'o', '_', 'l'] to:",
                    "options": ["ccool", "col", "cool", "c_o_l"],
                    "ans": "col"
                },
                {
                    "q": "The formula for Word Error Rate (WER) is:",
                    "options": ["(S + D + I) / N", "(S - D - I) / N", "N / (S + D + I)", "(S + D) / (N + I)"],
                    "ans": "(S + D + I) / N"
                },
                {
                    "q": "The Mel scale warps linear frequency to model:",
                    "options": ["Radio wave propagation", "Human ear non-linear pitch perception", "Speaker volume", "Microphone voltage"],
                    "ans": "Human ear non-linear pitch perception"
                }
            ],
            "short_questions": [
                "Explain the operational principle of Connectionist Temporal Classification (CTC) and describe how the blank token resolves repeated characters.",
                "Define Word Error Rate (WER) and explain the roles of Substitutions, Deletions, and Insertions."
            ],
            "application_question": "Explain how end-to-end speech recognition models are deployed in automobile hands-free infotainment systems.",
            "homework": "Given reference 'deep neural networks learn representations' and hypothesis 'deep neural network learned representation', compute the exact WER."
        }
    },
    {
        "day": 30,
        "module": 5,
        "topic_id": "5.3",
        "title": "Deep Learning Applications: Natural Language Processing",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO5",
        "blooms": "Understand / Apply",
        "prerequisites": "Seq2Seq, Word Embeddings intuition, Cross-Entropy Loss, Day 24 concepts",
        "learning_objectives": [
            "Survey core NLP application domains: Language Modeling, Sentiment Classification, Sequence Tagging, Machine Translation.",
            "Formulate Autoregressive Language Modeling and derive the Perplexity evaluation metric.",
            "Examine Text Classification using bidirectional deep recurrent networks.",
            "Analyze the transition from recurrent sequence models to Self-Attention and Transformers."
        ],
        "key_terminology": [
            "Natural Language Processing (NLP)", "Autoregressive Language Modeling", "Perplexity (PPL)",
            "Sentiment Classification", "Tokenization", "Cross-Entropy Loss", "Transformer Transition"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 12 (Section 12.4); Aggarwal: Chapter 7; Jurafsky & Martin: Speech and Language Processing.",
        "notes": {
            "nlp_task_taxonomy": (
                "Deep learning models natural language across diverse granularities:\n"
                "1. Document / Sentence-Level Classification:\n"
                "   - Sentiment Analysis: Input text -> Output sentiment label (Positive, Neutral, Negative).\n"
                "   - Topic Categorization: Input news article -> Output genre (Sports, Tech, Politics).\n\n"
                "2. Token-Level Sequence Tagging:\n"
                "   - Named Entity Recognition (NER): Tagging Person, Organization, Location per token.\n"
                "   - Part-of-Speech (POS) Tagging: Noun, Verb, Adjective.\n\n"
                "3. Autoregressive Language Modeling:\n"
                "   - Decomposes the joint probability of a sequence of tokens w_1, ..., w_T using the chain rule of probability:\n"
                "     P(w_1, w_2, ..., w_T) = product_{t=1}^T P(w_t | w_1, w_2, ..., w_{t-1})\n"
                "   - Foundation for modern generative LLMs (GPT, Llama, Gemini)."
            ),
            "perplexity_metric": (
                "Perplexity (PPL) is the standard quantitative metric for evaluating language models.\n"
                "It is defined as the exponentiated average negative log-likelihood per token:\n"
                "PPL = exp( - (1 / T) * sum_{t=1}^T ln P(w_t | w_{<t}) ) = 2^( Cross_Entropy_Loss_in_bits )\n\n"
                "Interpretation:\n"
                "Perplexity represents the weighted average branching factor of the language model — the effective number of equally likely words the model is uncertain between at each prediction step. Lower perplexity indicates superior predictive certainty!"
            ),
            "exam_important_points": [
                "Chain rule decomposition of joint sequence probability in language modeling.",
                "Mathematical formulation and physical interpretation of Perplexity (PPL).",
                "Deep learning architecture for Sentiment Analysis using Bidirectional LSTMs.",
                "Limitations of recurrent NLP models (sequential O(T) latency, lack of parallel training)."
            ]
        },
        "teaching_script": {
            "00-05": "Ask students: 'When your phone keyboard suggests the next word, how does it compute the probability?' Introduce Language Modeling.",
            "05-20": "Write the joint probability chain rule: P(w1...wT) = product P(wt | w<t). Explain next-token prediction.",
            "20-35": "Derive Perplexity: PPL = exp(Cross-Entropy Loss). Explain the branching factor interpretation: PPL=50 means the model chooses among 50 words.",
            "35-45": "Diagram a Bidirectional LSTM Sentiment Classifier: Word Embeddings -> BiLSTM -> Attention/Pooling -> Dense Softmax.",
            "45-55": "Demonstrate Python script calculating Cross-Entropy loss and Perplexity on a sample text sequence.",
            "55-60": "Recap and preview Day 31: Word2Vec (CBOW and Skip-Gram)."
        },
        "python_demo": {
            "title": "Autoregressive Language Model Perplexity Calculator in NumPy",
            "description": "Computing Cross-Entropy Loss and Perplexity on candidate next-token probability distributions.",
            "code": """import numpy as np

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
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Perplexity Calculation from Cross-Entropy Loss",
                "statement": "A neural language model evaluates a test document containing T = 1,000 words. The total negative log-likelihood of the document under natural logarithm (base e) is sum_{t=1}^T - ln P(w_t | w_{<t}) = 3,500 nats. (a) Calculate the average cross-entropy loss per token in nats, (b) Calculate the Perplexity (PPL) of the model, and (c) If a competing model achieves Perplexity PPL = 20, calculate its average loss per token.",
                "given": "T = 1,000 words, Total Loss = 3,500 nats.",
                "formulas": "Avg_Loss = Total_Loss / T; PPL = exp(Avg_Loss); Avg_Loss = ln(PPL)",
                "steps": [
                    "Step 1: Compute average cross-entropy loss per token:\n  Avg_Loss = 3,500 / 1,000 = 3.50 nats/token.",
                    "Step 2: Calculate Perplexity:\n  PPL = e^(3.50) = 33.115.",
                    "Step 3: Competing model with PPL = 20:\n  Loss = ln(20) = 2.9957 nats/token.",
                    "Step 4: Comparison:\n  The competing model has lower perplexity (20.0 vs 33.1) and lower loss (3.00 vs 3.50), indicating stronger language modeling ability."
                ],
                "answer": "Average Loss = 3.50 nats, Perplexity PPL = 33.12. Competing Model Loss = 3.00 nats.",
                "interpretation": "A perplexity of 33.12 means the language model is effectively as uncertain as guessing from a uniformly distributed vocabulary of 33 words at each step."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Sentiment Classifier Binary Cross-Entropy and Gradient Update",
                "statement": "A sentiment analysis model processes a movie review and outputs probability y_hat = 0.85 for Positive class (ground truth y = 1.0). The output layer computes y_hat = sigma(z) where z = w^T * h + b. Current weight is w_1 = 0.50, hidden activation is h_1 = 0.80, and learning rate is eta = 0.10. Loss is Binary Cross-Entropy L = - [y*ln(y_hat) + (1-y)*ln(1-y_hat)]. (a) Calculate the loss value, (b) Compute the gradient dL/dz and dL/dw_1, and (c) Compute the updated weight w_1_new.",
                "given": "y = 1.0, y_hat = 0.85, w_1 = 0.50, h_1 = 0.80, eta = 0.10.",
                "formulas": "L = - ln(y_hat); dL/dz = y_hat - y; dL/dw_1 = (dL/dz) * h_1; w_new = w - eta * (dL/dw)",
                "steps": [
                    "Step 1: Compute Binary Cross-Entropy Loss:\n  L = - [1.0 * ln(0.85) + 0] = - (-0.1625) = 0.1625.",
                    "Step 2: Compute gradient with respect to logit z:\n  dL/dz = y_hat - y = 0.85 - 1.00 = -0.15.",
                    "Step 3: Compute gradient with respect to weight w_1:\n  dL/dw_1 = (dL/dz) * h_1 = (-0.15) * 0.80 = -0.120.",
                    "Step 4: Update weight:\n  w_1_new = w_1 - eta * (dL/dw_1) = 0.50 - 0.10 * (-0.120) = 0.50 + 0.012 = 0.5120."
                ],
                "answer": "Loss L = 0.1625, Gradient dL/dw_1 = -0.120, Updated weight w_1_new = 0.5120",
                "interpretation": "Because the model underpredicted slightly (0.85 vs 1.0), the negative gradient increased weight w_1 to reinforce positive sentiment confidence in future passes."
            }
        ],
        "real_world_application": {
            "title": "Automated Financial SEC 10-K Filing Risk Assessment",
            "domain": "Fintech & Quantitative Regulatory Analysis",
            "workflow": "Public corporate annual 10-K financial disclosures (100+ pages) -> Deep bidirectional NLP model scans Risk Factors section -> Categorizes litigation, supply chain, and cyber risks -> Generates portfolio risk scores.",
            "benefit": "Enables asset managers to detect emerging corporate insolvency risks weeks before quarterly earnings announcements."
        },
        "viva_questions": [
            {"q": "What is an Autoregressive Language Model?", "a": "A model that predicts the probability of the next word conditioned on all previously generated context words: P(w_t | w_1, ..., w_{t-1})."},
            {"q": "What is Perplexity in language modeling?", "a": "The exponentiated cross-entropy loss, representing the effective branching factor (number of uncertain choices) per token."},
            {"q": "What is the relationship between Perplexity and Cross-Entropy Loss?", "a": "PPL = exp(Loss_nats) or PPL = 2^(Loss_bits). Lower perplexity indicates superior predictive performance."},
            {"q": "What is Named Entity Recognition (NER)?", "a": "A token-level sequence classification task that identifies mentions of predefined real-world entities (persons, locations, organizations) in text."},
            {"q": "How does subword tokenization (Byte-Pair Encoding, BPE) resolve the Out-of-Vocabulary (OOV) problem?", "a": "By decomposing rare and unseen words into frequent subword units (e.g. 'unpredictable' -> 'un' + 'predict' + 'able'), allowing any arbitrary string to be represented."},
            {"q": "Why did Recurrent Neural Networks struggle with long documents in NLP?", "a": "Because sequential processing is non-parallelizable on GPUs and distant context suffers from gradient vanishing, leading to their replacement by Transformers."},
            {"q": "What is the difference between extractive and abstractive text summarization?", "a": "Extractive selects verbatim key sentences directly from the source text; abstractive generates novel paraphrased sentences from semantic representations."},
            {"q": "What is sentiment analysis?", "a": "The task of classifying the emotional tone or subjective opinion expressed in text (positive, negative, neutral)."},
            {"q": "Why is cross-entropy loss universally used in language models?", "a": "Because language modeling is framed as multi-class classification across a vocabulary, where cross-entropy maximizes the log-likelihood of the true target word."},
            {"q": "What is teacher forcing in language model training?", "a": "Providing the ground-truth previous token as input during training, allowing full parallel computation of loss across all sequence tokens simultaneously."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "Perplexity is mathematically calculated as the exponential of:",
                    "options": ["Model parameter count", "Cross-entropy loss", "Learning rate", "L2 weight decay penalty"],
                    "ans": "Cross-entropy loss"
                },
                {
                    "q": "A lower perplexity score indicates that the language model is:",
                    "options": ["More uncertain about predictions", "More confident and accurate in its predictions", "Overfitting to training noise", "Lacking vocabulary capacity"],
                    "ans": "More confident and accurate in its predictions"
                },
                {
                    "q": "The task of identifying person names and organization titles in a text is called:",
                    "options": ["Sentiment Analysis", "Named Entity Recognition (NER)", "Machine Translation", "Topic Modeling"],
                    "ans": "Named Entity Recognition (NER)"
                }
            ],
            "short_questions": [
                "Formulate Autoregressive Language Modeling using the probability chain rule and derive the mathematical definition of Perplexity.",
                "Describe how a Bidirectional LSTM network is constructed to perform sentiment analysis on movie reviews."
            ],
            "application_question": "Explain how an automated customer support triage system uses NLP classification to route incoming emails to specialized technical departments.",
            "homework": "If a language model achieves a cross-entropy loss of 2.3026 nats per token on a test corpus, compute its exact Perplexity."
        }
    },
    {
        "day": 31,
        "module": 5,
        "topic_id": "5.4",
        "title": "Common Word Embedding: CBOW and Word2Vec",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO5",
        "blooms": "Understand / Apply",
        "prerequisites": "Vector representations, Softmax, Cross-Entropy, Day 3 concepts",
        "learning_objectives": [
            "Analyze the limitations of sparse one-hot encodings (curse of dimensionality, lack of semantic similarity).",
            "Formulate the Word2Vec framework (Mikolov et al., 2013): Continuous Bag of Words (CBOW) vs Continuous Skip-Gram.",
            "Formulate Skip-Gram with Negative Sampling (SGNS) and explain why it circumvents expensive Softmax over vocabulary V.",
            "Examine linear semantic vector compositionality (e.g. King - Man + Woman approx Queen)."
        ],
        "key_terminology": [
            "Word Embeddings", "One-Hot Encoding", "Word2Vec", "Continuous Bag of Words (CBOW)",
            "Continuous Skip-Gram", "Negative Sampling (SGNS)", "Vector Arithmetic"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 12 (Section 12.4); Aggarwal: Chapter 7; Mikolov et al.: Distributed Representations of Words and Phrases (NeurIPS 2013).",
        "notes": {
            "one_hot_vs_distributed_embeddings": (
                "One-Hot Encoding:\n"
                "- Represents each word as a sparse binary vector of vocabulary size |V| (e.g. 100,000 dimensions).\n"
                "- Fatal Flaw: All word vectors are orthogonal: v_cat^T * v_dog = 0. No notion of semantic distance or similarity!\n\n"
                "Distributed Word Embeddings:\n"
                "- Maps words to dense, low-dimensional continuous vectors in R^d (d = 100 to 300).\n"
                "- Words appearing in similar contexts acquire similar geometric representations (Distributional Hypothesis: Firth, 1957)."
            ),
            "word2vec_cbow_vs_skipgram": (
                "Tomas Mikolov et al. (Google, 2013) introduced Word2Vec with two complementary architectures:\n\n"
                "1. Continuous Bag-of-Words (CBOW):\n"
                "- Objective: Predict a target center word w_t given context words in a window: {w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2}}.\n"
                "- Mechanism: Averages context word vectors: v_context = (1 / 2C) * sum_{c in context} v_c.\n"
                "- Characteristics: Faster training, slightly better on frequent words.\n\n"
                "2. Continuous Skip-Gram:\n"
                "- Objective: Given a single center word w_t, predict surrounding context words w_{t+j}.\n"
                "- Characteristics: Slower training, but superior representations for rare words.\n\n"
                "Skip-Gram with Negative Sampling (SGNS):\n"
                "Evaluating standard Softmax requires summing exp(v^T * v') over the entire vocabulary |V| (costly O(|V|)).\n"
                "Negative Sampling replaces multiclass classification with k binary logistic regressions:\n"
                "L_SGNS = - ln sigma( v'_{w_O}^T * v_{w_I} ) - sum_{i=1}^k E_{w_i ~ P_n(w)} [ ln sigma( - v'_{w_i}^T * v_{w_I} ) ]\n"
                "where negative words are drawn from unigram distribution raised to the 3/4 power: P_n(w) propto f(w)^(0.75)."
            ),
            "vector_arithmetic": (
                "Linear compositionality: Word vectors capture semantic and grammatical analogies via simple vector arithmetic:\n"
                "vec('King') - vec('Man') + vec('Woman') approx vec('Queen')\n"
                "vec('Paris') - vec('France') + vec('Germany') approx vec('Berlin')"
            ),
            "exam_important_points": [
                "Architectural diagrams comparing CBOW (context -> target) vs Skip-Gram (target -> context).",
                "Why negative sampling is used instead of standard Softmax (computational complexity).",
                "Unigram distribution exponent 3/4 rationale for negative sampling.",
                "Semantic vector arithmetic: King - Man + Woman = Queen."
            ]
        },
        "teaching_script": {
            "00-05": "Write on blackboard: 'cat' = [1, 0, 0] and 'dog' = [0, 1, 0]. Ask: 'What is their dot product?' Zero! Introduce Distributed Embeddings.",
            "05-20": "Explain Firth's Distributional Hypothesis: 'You shall know a word by the company it keeps.' Formulate dense embedding matrices W and W'.",
            "20-35": "Compare CBOW vs Skip-Gram architectures side-by-side. Trace input context averaging in CBOW vs target broadcasting in Skip-Gram.",
            "35-45": "Explain the computational nightmare of Softmax over 100k words. Formulate Negative Sampling (SGNS) with binary cross-entropy.",
            "45-55": "Demonstrate Python script computing Cosine Similarity and semantic vector arithmetic (King - Man + Woman = Queen).",
            "55-60": "Recap and preview Day 32: Global Vectors for Word Representation (GloVe)."
        },
        "python_demo": {
            "title": "Cosine Similarity and Semantic Vector Arithmetic in NumPy",
            "description": "Simulating dense word embeddings and testing cosine similarity and analogical vector arithmetic.",
            "code": """import numpy as np

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
print("\\nVector Arithmetic: King - Man + Woman")
print("Similarity of (King - Man + Woman) to Queen:", round(sim_to_queen, 3), "(High analogical alignment!)")
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Cosine Similarity Calculation Between Word Vectors",
                "statement": "Two word embedding vectors in 3-dimensional semantic space are given as:\nv_doctor = [0.8, 0.6, 0.0]^T\nv_hospital = [0.6, 0.8, 0.0]^T\nv_banana = [0.0, 0.1, 0.9]^T\nCalculate: (a) Euclidean norms of all three vectors, (b) Cosine similarity between 'doctor' and 'hospital', and (c) Cosine similarity between 'doctor' and 'banana'.",
                "given": "v_doc = [0.8, 0.6, 0], v_hosp = [0.6, 0.8, 0], v_ban = [0, 0.1, 0.9].",
                "formulas": "||v|| = sqrt(sum v_i^2); cos_sim(u, v) = (u . v) / (||u|| * ||v||)",
                "steps": [
                    "Step 1: Compute Euclidean norms:\n  ||v_doc|| = sqrt(0.8^2 + 0.6^2 + 0) = sqrt(0.64 + 0.36) = sqrt(1.0) = 1.00.\n  ||v_hosp|| = sqrt(0.6^2 + 0.8^2 + 0) = sqrt(0.36 + 0.64) = sqrt(1.0) = 1.00.\n  ||v_ban|| = sqrt(0 + 0.1^2 + 0.9^2) = sqrt(0.01 + 0.81) = sqrt(0.82) = 0.9055.",
                    "Step 2: Compute dot products:\n  v_doc . v_hosp = (0.8 * 0.6) + (0.6 * 0.8) + (0 * 0) = 0.48 + 0.48 = 0.96.\n  v_doc . v_ban = (0.8 * 0) + (0.6 * 0.1) + (0 * 0.9) = 0.06.",
                    "Step 3: Compute Cosine Similarities:\n  cos_sim(doc, hosp) = 0.96 / (1.00 * 1.00) = 0.9600 (96.0% similarity).\n  cos_sim(doc, ban) = 0.06 / (1.00 * 0.9055) = 0.0663 (6.63% similarity)."
                ],
                "answer": "cos(doc, hosp) = 0.9600 (Strongly Related). cos(doc, ban) = 0.0663 (Unrelated).",
                "interpretation": "Cosine similarity successfully captures high semantic relatedness between clinical concepts while showing near-zero alignment with unrelated entities."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Skip-Gram Computational Complexity: Full Softmax vs Negative Sampling",
                "statement": "A corpus has vocabulary size |V| = 200,000 words. Word embedding dimension is d = 300. (a) Calculate the multiplication operations required per training step to evaluate full Softmax over the vocabulary, (b) Calculate the operations required under Negative Sampling with k = 5 negative samples, and (c) Determine the exact computational speedup factor achieved by SGNS.",
                "given": "|V| = 200,000, d = 300, k = 5 negative samples.",
                "formulas": "Full Softmax Ops = |V| * d (for logits) + |V| (for exp/sum); SGNS Ops = (1 + k) * d",
                "steps": [
                    "Step 1: Compute Full Softmax operations:\n  Computing inner products for all vocabulary words: 200,000 * 300 = 60,000,000 multiplications.\n  Exponentiation and normalizer sum: ~200,000 ops.\n  Total Full Softmax ops approx 60,000,000 operations per word update.",
                    "Step 2: Compute Negative Sampling (SGNS) operations:\n  SGNS evaluates 1 true positive pair + k negative pairs = 1 + 5 = 6 dot products.\n  Total multiplications = 6 * 300 = 1,800 operations.",
                    "Step 3: Calculate computational speedup:\n  Speedup = 60,000,000 / 1,800 = 33,333.33x (~33,333 times faster!)."
                ],
                "answer": "Full Softmax = 60M ops. SGNS = 1,800 ops (Over 33,000x faster!).",
                "interpretation": "Negative sampling replaces a massive 60-million operation vocabulary sweep with just 1,800 simple multiplications, transforming Word2Vec from an intractable theoretical model into an ultra-fast algorithm trainable on billions of words in minutes."
            }
        ],
        "real_world_application": {
            "title": "E-Commerce Semantic Search & Recommendation Engine",
            "domain": "Retail E-Commerce & Product Discovery",
            "workflow": "Customer searches 'running shoes' on shopping app -> Catalog items lack literal string match for 'sneakers' -> Word2Vec item embeddings compute high cosine similarity between 'running shoes' and 'sneakers' -> Recommends relevant footwear.",
            "benefit": "Increases search conversion rates by 19% by overcoming the vocabulary mismatch problem."
        },
        "viva_questions": [
            {"q": "What is the primary limitation of one-hot word representations?", "a": "High dimensionality, extreme sparsity, and complete lack of semantic distance or similarity between words (all one-hot vectors are orthogonal)."},
            {"q": "What is the Distributional Hypothesis in linguistics?", "a": "'Words that occur in similar contexts tend to have similar meanings' (J.R. Firth, 1957)."},
            {"q": "What is the difference between CBOW and Skip-Gram?", "a": "CBOW predicts the center target word from surrounding context words; Skip-Gram predicts the surrounding context words given the center target word."},
            {"q": "Why is standard Softmax computationally prohibitive in Word2Vec?", "a": "Because computing the denominator requires calculating inner products and exponentials over the entire vocabulary |V| (hundreds of thousands of words) for every training step."},
            {"q": "How does Negative Sampling (SGNS) solve the Softmax bottleneck?", "a": "It reformulates the task as binary logistic regression, updating only the positive word pair and k randomly chosen negative words, cutting complexity from O(|V|) to O(k)."},
            {"q": "Why are negative words sampled from the unigram distribution raised to the 3/4 power?", "a": "The exponent 0.75 elevates the sampling probability of rare words relative to frequent stop-words ('the', 'is'), ensuring the model sees informative negative examples."},
            {"q": "What is Hierarchical Softmax?", "a": "An alternative method using a binary Huffman tree of words to evaluate probabilities in O(log_2 |V|) steps instead of O(|V|)."},
            {"q": "What famous semantic analogy demonstrated Word2Vec compositionality in 2013?", "a": "vec('King') - vec('Man') + vec('Woman') approx vec('Queen')."},
            {"q": "What does a cosine similarity of 1.0 between two word embeddings indicate?", "a": "The two vectors point in the exact same direction in semantic space, indicating maximum semantic similarity."},
            {"q": "Are Word2Vec embeddings contextualized?", "a": "No; Word2Vec produces static embeddings where a polysemous word (e.g. 'bank') has only a single vector regardless of context (unlike contextual embeddings in BERT/Transformers)."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "The objective of Continuous Bag of Words (CBOW) is to predict:",
                    "options": ["Context words from a center word", "A center word from surrounding context words", "The next paragraph", "The sentiment score"],
                    "ans": "A center word from surrounding context words"
                },
                {
                    "q": "Negative Sampling in Word2Vec reduces training complexity from O(|V|) to:",
                    "options": ["O(|V|^2)", "O(k) where k is the number of negative samples", "O(1)", "O(log |V|)"],
                    "ans": "O(k) where k is the number of negative samples"
                },
                {
                    "q": "In Word2Vec negative sampling, the unigram distribution exponent used to sample negative words is:",
                    "options": ["0.5", "0.75", "1.0", "2.0"],
                    "ans": "0.75"
                }
            ],
            "short_questions": [
                "Contrast the architectures of Continuous Bag of Words (CBOW) and Continuous Skip-Gram using architectural diagrams.",
                "Formulate Skip-Gram with Negative Sampling (SGNS) and explain why it provides massive computational speedups."
            ],
            "application_question": "Explain how Word2Vec embeddings are utilized in biomedical NLP to identify synonym drug interactions across clinical trial abstracts.",
            "homework": "Prove that for two unit-normalized vectors u and v, the squared Euclidean distance ||u - v||^2 is equal to 2 * (1 - cosine_similarity(u, v))."
        }
    },
    {
        "day": 32,
        "module": 5,
        "topic_id": "5.5",
        "title": "Common Word Embedding: Global Vectors for Word Representation (GloVe)",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO5",
        "blooms": "Understand / Apply",
        "prerequisites": "Matrix Factorization, Word2Vec, Log-bilinear models, Day 31 concepts",
        "learning_objectives": [
            "Formulate the Global Vectors (GloVe) model (Pennington, Socher, Manning, EMNLP 2014).",
            "Construct the Global Co-occurrence Matrix X and explain co-occurrence ratios P(k|i) / P(k|j).",
            "Derive the GloVe weighted least-squares objective function and weighting function f(X_ij).",
            "Compare Word2Vec (local window streaming) vs GloVe (global matrix factorization)."
        ],
        "key_terminology": [
            "Global Vectors (GloVe)", "Co-occurrence Matrix", "Co-occurrence Probability Ratio",
            "Log-Bilinear Model", "Weighted Least Squares", "Weighting Function f(X_ij)"
        ],
        "textbook_ref": "Pennington, Socher, Manning: GloVe: Global Vectors for Word Representation (EMNLP 2014); Aggarwal: Chapter 7; DEEP LEARNING.pdf Section 2.9.1.",
        "notes": {
            "glove_motivation": (
                "Prior to GloVe (2014), representation learning was divided into two distinct paradigms:\n"
                "1. Global Matrix Factorization Methods (LSA): Leveraged global corpus statistics, but performed poorly on local word analogy tasks.\n"
                "2. Local Context Window Methods (Word2Vec): Excellent on analogical reasoning, but failed to directly leverage vast global co-occurrence statistics.\n\n"
                "GloVe combines the benefits of both by directly fitting a log-bilinear model to the global word-word co-occurrence matrix X."
            ),
            "mathematical_foundation_and_cooccurrence_ratios": (
                "Let X_ij be the count of times word j appears in the context of word i in the corpus.\n"
                "Let P_ij = P(j | i) = X_ij / X_i be the probability that word j appears in the context of word i.\n\n"
                "The core insight of GloVe is that semantic meaning is captured by the RATIO of co-occurrence probabilities:\n"
                "Consider i = 'ice', j = 'steam', and probe words k:\n"
                "- k = 'solid': P(solid | ice) is high, P(solid | steam) is low -> Ratio P(solid|ice) / P(solid|steam) >> 1\n"
                "- k = 'gas':   P(gas | ice) is low, P(gas | steam) is high -> Ratio P(gas|ice) / P(gas|steam) << 1\n"
                "- k = 'water': P(water | ice) is high, P(water | steam) is high -> Ratio approx 1\n"
                "- k = 'fashion': P(fashion | ice) is low, P(fashion | steam) is low -> Ratio approx 1\n\n"
                "GloVe models this ratio using dot products of word vectors:\n"
                "w_i^T * w_tilde_j + b_i + b_tilde_j = ln( X_ij )"
            ),
            "weighted_least_squares_objective": (
                "GloVe minimizes the weighted squared error between the dot product of word vectors and the log co-occurrence counts:\n"
                "J = sum_{i=1}^V sum_{j=1}^V f(X_ij) * ( w_i^T * w_tilde_j + b_i + b_tilde_j - ln(X_ij) )^2\n\n"
                "Weighting Function f(X_ij):\n"
                "f(x) = { (x / x_max)^alpha  if x < x_max\n"
                "       { 1                 if x >= x_max\n"
                "(typically x_max = 100, alpha = 0.75).\n"
                "Properties:\n"
                "1. f(0) = 0: Ensures ln(0) is never evaluated, avoiding divergence on non-co-occurring pairs.\n"
                "2. Non-decreasing: Frequent words carry appropriate statistical weight.\n"
                "3. Saturating at 1: Prevents extremely common stop-words ('the', 'of') from dominating the objective."
            ),
            "exam_important_points": [
                "The co-occurrence probability ratio table (ice vs steam with solid, gas, water, fashion).",
                "Derivation of the log-bilinear equation: w_i^T * w_j + b_i + b_j = ln(X_ij).",
                "Full GloVe weighted least-squares objective function and properties of weighting function f(x).",
                "Comparison: Word2Vec vs GloVe."
            ]
        },
        "teaching_script": {
            "00-05": "Introduce the debate: 'Word2Vec slides a small window across text. Why throw away global corpus frequency matrices?' Introduce GloVe.",
            "05-20": "Draw the Co-occurrence Matrix X on blackboard. Define P(k|i) = X_ik / X_i.",
            "20-35": "Present the iconic Ice vs Steam probe word ratio table: show why the RATIO P(k|ice)/P(k|steam) isolates semantic properties.",
            "35-45": "Derive the GloVe objective function: J = sum f(X_ij) * (w_i^T * w_j + b_i + b_j - ln X_ij)^2. Explain the 3 properties of f(x).",
            "45-55": "Demonstrate Python script computing co-occurrence ratios and evaluating GloVe vector dot products.",
            "55-60": "Recap and preview Day 33: Autoencoders and Representation Learning."
        },
        "python_demo": {
            "title": "GloVe Co-occurrence Probability Ratios Demonstration in NumPy",
            "description": "Simulating the ice vs steam co-occurrence matrix and calculating semantic probability ratios.",
            "code": """import numpy as np

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
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "GloVe Weighting Function f(x) Calculation",
                "statement": "The GloVe weighting function is defined as f(x) = (x / x_max)^alpha if x < x_max else 1.0, with x_max = 100 and alpha = 0.75. Calculate the weight f(x) assigned to: (a) A word pair with zero co-occurrences (x = 0), (b) A moderately frequent pair with x = 16 co-occurrences, (c) A frequent pair with x = 81 co-occurrences, and (d) An extremely frequent stop-word pair with x = 5,000 co-occurrences.",
                "given": "x_max = 100, alpha = 0.75.",
                "formulas": "f(x) = (x / 100)^0.75 if x < 100 else 1.0",
                "steps": [
                    "Step 1: For x = 0:\n  f(0) = 0.0 (Eliminates log(0) singularity in objective function!).",
                    "Step 2: For x = 16:\n  f(16) = (16 / 100)^0.75 = (0.16)^0.75 = (0.16^(3/4)) = (0.16^(1/4))^3 = (0.63245)^3 = 0.25298.",
                    "Step 3: For x = 81:\n  f(81) = (81 / 100)^0.75 = (0.81)^0.75 = (0.81^(1/4))^3 = (0.94868)^3 = 0.8538.",
                    "Step 4: For x = 5,000:\n  Since x = 5,000 >= x_max = 100, f(5,000) = 1.0 (Caps runaway stop-word weights)."
                ],
                "answer": "f(0) = 0.0, f(16) = 0.2530, f(81) = 0.8538, f(5000) = 1.0000",
                "interpretation": "The weighting function smooths out frequency disparities, preventing zero-count crashes while preventing stop-word co-occurrences from dominating optimization."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "GloVe Log-Bilinear Prediction and Loss Gradient",
                "statement": "In a GloVe model, word vector w_i = [0.5, 0.4] and context vector w_tilde_j = [0.6, 0.2] with scalar biases b_i = 0.10 and b_tilde_j = 0.05. The observed co-occurrence count is X_ij = 2.0. The weight assigned by f(X_ij) is 0.05. (a) Calculate the dot product plus biases: z_ij = w_i^T * w_tilde_j + b_i + b_tilde_j, (b) Calculate the target log count ln(X_ij), (c) Compute the error e_ij = z_ij - ln(X_ij), and (d) Calculate the gradient of the loss with respect to bias b_i.",
                "given": "w_i = [0.5, 0.4], w_tilde_j = [0.6, 0.2], b_i = 0.10, b_tilde_j = 0.05, X_ij = 2.0, f(X_ij) = 0.05.",
                "formulas": "z = w_i^T * w_tilde_j + b_i + b_tilde_j; e = z - ln(X_ij); J_ij = f(X_ij) * e^2; dJ/db_i = 2 * f(X_ij) * e",
                "steps": [
                    "Step 1: Compute linear dot product plus biases:\n  dot = (0.5 * 0.6) + (0.4 * 0.2) = 0.30 + 0.08 = 0.38.\n  z_ij = 0.38 + 0.10 + 0.05 = 0.530.",
                    "Step 2: Compute target log count:\n  ln(X_ij) = ln(2.0) = 0.69315.",
                    "Step 3: Compute prediction error:\n  e_ij = z_ij - ln(X_ij) = 0.530 - 0.69315 = -0.16315.",
                    "Step 4: Compute gradient with respect to bias b_i:\n  dJ/db_i = 2 * f(X_ij) * e_ij = 2 * 0.05 * (-0.16315) = 0.10 * (-0.16315) = -0.016315."
                ],
                "answer": "z_ij = 0.530, Target ln(X_ij) = 0.6931, Error = -0.1631, Gradient dJ/db_i = -0.0163",
                "interpretation": "Because the predicted log count (0.53) was smaller than the empirical count (0.69), the negative gradient will increase bias b_i to better match corpus statistics."
            }
        ],
        "real_world_application": {
            "title": "Legal Contract Discovery & Precedent Retrieval",
            "domain": "Legal Tech & Corporate Compliance",
            "workflow": "Corporate legal repository contains 20 years of court filings -> GloVe embeddings trained on corpus capture domain-specific legal terminology associations ('plaintiff', 'tort', 'indemnity') -> Enables semantic case law search.",
            "benefit": "Slashes paralegal legal discovery search time from 8 hours to 15 minutes per case."
        },
        "viva_questions": [
            {"q": "What does the acronym GloVe stand for?", "a": "Global Vectors for Word Representation."},
            {"q": "Who invented GloVe and in what year?", "a": "Jeffrey Pennington, Richard Socher, and Christopher D. Manning at Stanford University in 2014."},
            {"q": "How does GloVe combine global matrix factorization and local context windows?", "a": "It constructs a global word-word co-occurrence matrix from local context windows and trains a log-bilinear regression model directly on this global matrix."},
            {"q": "What is the core insight behind GloVe's relationship to word meaning?", "a": "The ratio of co-occurrence probabilities of two words with various probe words encodes semantic differences much more cleanly than raw probabilities."},
            {"q": "What is the GloVe objective function?", "a": "J = sum_{i,j} f(X_ij) * (w_i^T * w_tilde_j + b_i + b_tilde_j - ln(X_ij))^2."},
            {"q": "Why is the weighting function f(X_ij) necessary in GloVe?", "a": "It prevents undefined ln(0) errors by setting f(0) = 0, and caps the influence of extremely frequent stop words so they do not dominate the loss."},
            {"q": "What is the typical value of alpha in the GloVe weighting function?", "a": "alpha = 0.75 (3/4)."},
            {"q": "How are word vectors w and context vectors w_tilde combined after training GloVe?", "a": "Because X is symmetric, w and w_tilde are equivalent; typically their sum (w + w_tilde) is used as the final word embedding to reduce noise."},
            {"q": "What is the time complexity of training GloVe?", "a": "O(|X|), where |X| is the number of non-zero entries in the co-occurrence matrix, which scales far slower than corpus size for large corpora."},
            {"q": "Compare GloVe and Word2Vec in terms of memory requirements during training.", "a": "GloVe requires memory to store the sparse co-occurrence matrix X before training begins; Word2Vec streams through text sequentially without needing to store a co-occurrence matrix."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "GloVe trains word embeddings by fitting a log-bilinear model to the:",
                    "options": ["Adjacency matrix of an RNN", "Global word-word co-occurrence matrix", "Confusion matrix of a CNN", "Hessian matrix of the loss"],
                    "ans": "Global word-word co-occurrence matrix"
                },
                {
                    "q": "In the GloVe weighting function f(x), setting f(0) = 0 ensures that:",
                    "options": ["Negative words have zero vectors", "ln(0) singularities are eliminated from the objective", "All weights decay to zero", "Only stop words are trained"],
                    "ans": "ln(0) singularities are eliminated from the objective"
                },
                {
                    "q": "After GloVe training, the final word vector representation is commonly computed as:",
                    "options": ["w_i * w_tilde_i", "w_i + w_tilde_i", "w_i / w_tilde_i", "w_i - w_tilde_i"],
                    "ans": "w_i + w_tilde_i"
                }
            ],
            "short_questions": [
                "Explain the motivation behind the GloVe model and describe how co-occurrence probability ratios capture semantic relationships.",
                "Write down the complete GloVe objective function and explain the three essential design properties of the weighting function f(X_ij)."
            ],
            "application_question": "Explain how GloVe embeddings can be fine-tuned for a domain-specific clinical medical transcription system.",
            "homework": "Prove that if the co-occurrence matrix X is strictly symmetric, the optimal word vectors w and context vectors w_tilde are identical up to random initialization."
        }
    },
    {
        "day": 33,
        "module": 5,
        "topic_id": "5.6",
        "title": "Current Research Areas: Autoencoders and Representation Learning",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO5",
        "blooms": "Understand / Apply",
        "prerequisites": "Encoder-Decoder mechanics, Unsupervised learning, Regularization, Day 2 & 14 concepts",
        "learning_objectives": [
            "Formulate the Autoencoder framework: Encoder, Latent Bottleneck, and Decoder.",
            "Analyze Undercomplete Autoencoders and compare their latent space to Principal Component Analysis (PCA).",
            "Formulate Regularized Autoencoders: Sparse Autoencoders (KL Divergence penalty) and Denoising Autoencoders (DAE).",
            "Examine Contractive Autoencoders (CAE) and their Frobenius norm Jacobian penalty."
        ],
        "key_terminology": [
            "Autoencoder", "Undercomplete Autoencoder", "Latent Bottleneck", "Reconstruction Loss",
            "Sparse Autoencoder", "Kullback-Leibler (KL) Divergence", "Denoising Autoencoder (DAE)", "Contractive Autoencoder (CAE)"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 14 (Sections 14.1 - 14.5); Aggarwal: Chapter 4; KTU CST414 Syllabus Model Question Paper Part A (Q10) & Part B (Q19b).",
        "notes": {
            "autoencoder_architecture": (
                "An Autoencoder is a self-supervised neural network trained to reconstruct its own input:\n"
                "1. Encoder: Maps input x in R^d to a latent code h in R^k: h = f(x) = sigma( W_e * x + b_e )\n"
                "2. Decoder: Reconstructs input from latent code: r = g(h) = sigma( W_d * h + b_d )\n"
                "3. Objective: Minimize reconstruction error L(x, g(f(x))) (e.g. MSE for continuous data, BCE for normalized pixels).\n\n"
                "If an autoencoder succeeds in learning g(f(x)) = x everywhere, it would simply memorize the identity function! To learn meaningful representations, constraints must be imposed."
            ),
            "autoencoder_variants": (
                "1. Undercomplete Autoencoders:\n"
                "- Constrains latent dimension k < d (information bottleneck).\n"
                "- Theorem: If the encoder and decoder are linear and loss is MSE, the undercomplete autoencoder learns the exact same subspace as Principal Component Analysis (PCA)!\n"
                "- With non-linear activations, it learns a non-linear manifold generalization of PCA.\n\n"
                "2. Sparse Autoencoders:\n"
                "- Overcomplete latent space (k > d), but penalizes average hidden unit activations rho_hat_j using KL divergence:\n"
                "  L_sparse = L_recon + beta * sum_{j=1}^k KL( rho || rho_hat_j )\n"
                "  where KL(rho || rho_hat) = rho * ln(rho / rho_hat) + (1 - rho) * ln((1 - rho) / (1 - rho_hat)).\n\n"
                "3. Denoising Autoencoders (DAE, Vincent et al., 2008):\n"
                "- Corrupts input x with noise: x_tilde ~ q(x_tilde | x).\n"
                "- Trains the network to reconstruct the CLEAN input x from noisy x_tilde: L = L(x, g(f(x_tilde))).\n"
                "- Forces the model to learn the data manifold's score matching vector field pointing from noise back to the data manifold!\n\n"
                "4. Contractive Autoencoders (CAE, Rifai et al., 2011):\n"
                "- Adds an explicit Frobenius norm penalty on the encoder's Jacobian matrix: || J_f(x) ||_F^2 = sum_i sum_j (del h_i / del x_j)^2.\n"
                "- Forces latent representations to be invariant (contractive) to small perturbations in the input space."
            ),
            "exam_important_points": [
                "KTU Exam Q10: Definition of autoencoder and one application (anomaly detection, dimensionality reduction).",
                "KTU Exam Q19(b): Merits and demerits of using Autoencoders in Computer Vision.",
                "Proof that linear autoencoder under MSE loss is equivalent to PCA.",
                "Formulation of Sparse Autoencoders and KL divergence sparsity penalty."
            ]
        },
        "teaching_script": {
            "00-05": "Introduce self-supervised learning: 'How can a network learn deep features without a single human label?' Introduce Autoencoders.",
            "05-20": "Draw the hourglass architecture: Input -> Encoder -> Latent code h -> Decoder -> Reconstruction r. Formulate MSE reconstruction loss.",
            "20-35": "Explain why linear autoencoder equals PCA. Introduce Undercomplete vs Overcomplete architectures.",
            "35-45": "Present Denoising Autoencoder (DAE) and Sparse Autoencoder (KL divergence). Draw the score matching vector field pointing to the data manifold.",
            "45-55": "Demonstrate Python script training a NumPy Autoencoder and extracting 2D manifold embeddings.",
            "55-60": "Recap and preview Day 34: Representation Learning and the Manifold Hypothesis."
        },
        "python_demo": {
            "title": "Undercomplete and Denoising Autoencoder in NumPy",
            "description": "Implementation of forward and backward passes of a non-linear Autoencoder trained with noise corruption.",
            "code": """import numpy as np

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
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Kullback-Leibler (KL) Divergence Sparsity Penalty Calculation",
                "statement": "In a sparse autoencoder, the target average activation is set to rho = 0.05. A hidden neuron j has computed average activation rho_hat_j = 0.10 across a training batch. Calculate: (a) The Kullback-Leibler divergence penalty KL(rho || rho_hat_j) for this neuron, and (b) The penalty if the neuron had perfectly satisfied the target (rho_hat = 0.05).",
                "given": "Target rho = 0.05, Actual rho_hat = 0.10.",
                "formulas": "KL(rho || rho_hat) = rho * ln(rho / rho_hat) + (1 - rho) * ln((1 - rho) / (1 - rho_hat))",
                "steps": [
                    "Step 1: Term 1: rho * ln(rho / rho_hat):\n  rho / rho_hat = 0.05 / 0.10 = 0.50.\n  ln(0.50) = -0.69315.\n  Term 1 = 0.05 * (-0.69315) = -0.03466.",
                    "Step 2: Term 2: (1 - rho) * ln((1 - rho) / (1 - rho_hat)):\n  1 - rho = 1 - 0.05 = 0.95.\n  1 - rho_hat = 1 - 0.10 = 0.90.\n  (1 - rho) / (1 - rho_hat) = 0.95 / 0.90 = 1.05556.\n  ln(1.05556) = 0.05407.\n  Term 2 = 0.95 * 0.05407 = 0.05137.",
                    "Step 3: Combine terms:\n  KL(0.05 || 0.10) = -0.03466 + 0.05137 = 0.01671.",
                    "Step 4: If rho_hat = 0.05:\n  KL(0.05 || 0.05) = 0.05 * ln(1) + 0.95 * ln(1) = 0.0000."
                ],
                "answer": "KL(0.05 || 0.10) = 0.01671. Target penalty = 0.0000.",
                "interpretation": "The positive KL divergence (0.0167) penalizes the neuron for being overly active (10% firing rate vs 5% target), pushing its bias negative to restore sparsity."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Reconstruction Error vs Dimensionality Reduction in Linear Autoencoder",
                "statement": "An input vector x = [3.0, 4.0]^T is projected into a 1-dimensional latent subspace using linear encoder weight w = [0.8, 0.6]^T (unit norm ||w|| = 1) with zero bias: h = w^T * x. The decoder reconstructs the input via tied weights: x_hat = w * h. (a) Compute the scalar latent code h, (b) Compute the reconstructed vector x_hat, (c) Calculate the reconstruction error vector e = x - x_hat, and (d) Verify that the reconstruction error vector is orthogonal to the weight vector w (e^T * w = 0).",
                "given": "x = [3.0, 4.0]^T, w = [0.8, 0.6]^T.",
                "formulas": "h = w^T * x; x_hat = w * h; e = x - x_hat; Orthogonality check: e^T * w = 0",
                "steps": [
                    "Step 1: Compute latent code h:\n  h = (0.8 * 3.0) + (0.6 * 4.0) = 2.4 + 2.4 = 4.80.",
                    "Step 2: Compute reconstructed vector x_hat:\n  x_hat = h * w = 4.80 * [0.8, 0.6]^T = [3.84, 2.88]^T.",
                    "Step 3: Compute reconstruction error vector e:\n  e = x - x_hat = [3.0 - 3.84, 4.0 - 2.88]^T = [-0.84, 1.12]^T.",
                    "Step 4: Compute squared reconstruction loss:\n  Loss = ||e||^2 = (-0.84)^2 + (1.12)^2 = 0.7056 + 1.2544 = 1.960.",
                    "Step 5: Verify orthogonality:\n  e^T * w = (-0.84 * 0.8) + (1.12 * 0.6) = -0.672 + 0.672 = 0.0000."
                ],
                "answer": "h = 4.80, x_hat = [3.84, 2.88]^T, Error e = [-0.84, 1.12]^T (Orthogonal: e^T * w = 0.0000).",
                "interpretation": "The linear autoencoder performs an exact orthogonal projection onto the 1D principal component subspace, discarding the orthogonal component as reconstruction error."
            }
        ],
        "real_world_application": {
            "title": "Industrial High-Speed Rail Track Flaw Ultrasonic Anomaly Detection",
            "domain": "Railway Safety & Predictive Infrastructure",
            "workflow": "Ultrasonic inspection cars record rail acoustic waveforms -> Undercomplete Autoencoder trained exclusively on thousands of kilometers of normal, undamaged steel rail -> When a microscopic sub-surface rail crack passes, autoencoder fails to reconstruct it -> High reconstruction error triggers immediate track repair alarm.",
            "benefit": "Detects internal rail fatigue fractures before structural derailment failures occur."
        },
        "viva_questions": [
            {"q": "What is an Autoencoder?", "a": "A neural network trained in a self-supervised manner to copy its input to its output through an internal latent representation."},
            {"q": "What is the primary danger of training an unconstrained autoencoder?", "a": "It can learn to perfectly memorize the trivial identity function g(f(x)) = x without extracting useful semantic features."},
            {"q": "What is an Undercomplete Autoencoder?", "a": "An autoencoder whose latent representation dimension k is strictly smaller than the input dimension d."},
            {"q": "What classical statistical technique is equivalent to a linear autoencoder trained with MSE loss?", "a": "Principal Component Analysis (PCA)."},
            {"q": "What is a Denoising Autoencoder (DAE)?", "a": "An autoencoder trained to reconstruct the clean original data from an intentionally corrupted or noisy input."},
            {"q": "What is a Sparse Autoencoder?", "a": "An autoencoder whose loss function penalizes average neuron activations using KL divergence, forcing most latent units to be inactive for any given sample."},
            {"q": "What is a Contractive Autoencoder (CAE)?", "a": "An autoencoder that penalizes the Frobenius norm of the encoder's Jacobian matrix, making the latent representation robust to small input perturbations."},
            {"q": "How can an autoencoder be used for Anomaly Detection?", "a": "The model is trained only on normal data; when an anomalous sample is tested, the model produces a high reconstruction error, flagging the anomaly."},
            {"q": "What is the difference between a standard Autoencoder and a Variational Autoencoder (VAE)?", "a": "A standard autoencoder maps inputs to deterministic vectors; a VAE maps inputs to the parameters of a probability distribution (mean and variance) and samples from it, acting as a generative model."},
            {"q": "What are the demerits of using Autoencoders for Computer Vision image generation?", "a": "They tend to produce blurry reconstructions due to mean-squared-error pixel averaging (unlike GANs or Diffusion models)."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "A linear autoencoder trained with mean squared error loss spans the exact same subspace as:",
                    "options": ["Support Vector Machines", "Principal Component Analysis (PCA)", "K-Means Clustering", "Random Forests"],
                    "ans": "Principal Component Analysis (PCA)"
                },
                {
                    "q": "A Denoising Autoencoder is trained to reconstruct:",
                    "options": ["The noisy corrupted input", "The clean uncorrupted input", "The class label", "The gradient vector"],
                    "ans": "The clean uncorrupted input"
                },
                {
                    "q": "In a Sparse Autoencoder, the sparsity of hidden unit activations is commonly penalized using:",
                    "options": ["L2 Weight Decay", "Kullback-Leibler (KL) Divergence", "Gradient Clipping", "Hessian Condition Number"],
                    "ans": "Kullback-Leibler (KL) Divergence"
                }
            ],
            "short_questions": [
                "Formulate the Autoencoder architecture and contrast Undercomplete vs Overcomplete Sparse Autoencoders.",
                "Explain the operational principle of Denoising Autoencoders (DAE) and describe their score matching geometric interpretation."
            ],
            "application_question": "Explain how an Autoencoder is deployed in financial credit card transaction monitoring to detect novel, previously unseen fraud patterns.",
            "homework": "Prove that if encoder and decoder weights are tied (W_d = W_e^T), the total parameter count in an autoencoder with input dim 100 and latent dim 20 is exactly 2,120."
        }
    },
    {
        "day": 34,
        "module": 5,
        "topic_id": "5.7",
        "title": "Current Research Areas: Representation Learning",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO5",
        "blooms": "Understand / Analyze",
        "prerequisites": "Autoencoders, Latent variables, Manifolds, Day 33 concepts",
        "learning_objectives": [
            "Formulate the Manifold Hypothesis in Deep Learning.",
            "Examine the goals of Representation Learning: Disentanglement, Linearity, and Invariance.",
            "Analyze Self-Supervised Learning (Contrastive Learning, SimCLR, Masked Autoencoders).",
            "Evaluate how representation learning enables few-shot and zero-shot transfer learning in downstream vision and NLP tasks."
        ],
        "key_terminology": [
            "Representation Learning", "Manifold Hypothesis", "Disentangled Representations",
            "Self-Supervised Learning", "Contrastive Learning", "InfoNCE Loss", "Transfer Learning"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 15 (Representation Learning); KTU CST414 Syllabus Model Question Paper Part B (Q20a).",
        "notes": {
            "manifold_hypothesis": (
                "The Manifold Hypothesis states that real-world high-dimensional data (e.g. 1000x1000 pixel images in R^1,000,000) concentrates its probability mass almost entirely on low-dimensional manifolds embedded within the high-dimensional ambient space.\n\n"
                "Key Observations:\n"
                "1. If pixels were uniformly distributed in R^1,000,000, random noise would look like natural scenes. In reality, natural images occupy a tiny fraction of configuration space.\n"
                "2. Variations in images (lighting, camera pose, object scale, facial expression) correspond to smooth continuous movements along low-dimensional manifold coordinates.\n"
                "3. Deep networks learn coordinate charts that unfold, unroll, and flatten these complex curved manifolds into Euclidean spaces where linear classifiers succeed."
            ),
            "disentangled_representations": (
                "Bengio, Courville, and Vincent (2013) define an ideal representation as a 'Disentangled Representation'.\n\n"
                "Properties:\n"
                "- Each coordinate axis of the latent representation h corresponds to an independent real-world explanatory factor of variation (e.g. h_1 = object rotation, h_2 = lighting intensity, h_3 = object identity).\n"
                "- Invariance: A change in lighting modifies only h_2 while leaving identity h_3 completely invariant.\n"
                "- Linear Separability: Complex non-linear decision boundaries on the original data manifold become flat linear hyperplanes in the disentangled representation space."
            ),
            "self_supervised_learning": (
                "Modern Representation Learning utilizes Self-Supervised Learning:\n"
                "1. Contrastive Learning (SimCLR, Chen et al., 2020):\n"
                "   Pulls together representations of augmented views of the same image (positives) while pushing apart representations of different images (negatives) using InfoNCE loss:\n"
                "   L_InfoNCE = - ln ( exp( sim(z_i, z_j) / tau ) / sum_k exp( sim(z_i, z_k) / tau ) )\n\n"
                "2. Masked Autoencoding (MAE, He et al., 2021):\n"
                "   Masks 75% of image patches and trains a Vision Transformer to reconstruct missing pixels, learning rich global representations."
            ),
            "exam_important_points": [
                "KTU Exam Q20(a): Illustrate the use of representation learning in object classification.",
                "Definition and physical justification of the Manifold Hypothesis.",
                "Properties of Disentangled Representations (invariance, linearity, factor independence).",
                "Formulation of InfoNCE loss in contrastive self-supervised learning."
            ]
        },
        "teaching_script": {
            "00-05": "Show a Swiss Roll manifold 3D plot. Ask: 'How can a network separate the red and blue spirals?' Introduce the Manifold Hypothesis.",
            "05-20": "Explain the Manifold Hypothesis: real data lives on low-dimensional sub-manifolds in high-dimensional space. Show how deep layers flatten manifolds.",
            "20-35": "Define Disentangled Representations: latent axes controlling independent factors of variation (pose, azimuth, lighting, identity).",
            "35-45": "Formulate Self-Supervised Contrastive Learning. Write the InfoNCE loss equation. Explain positive pairs vs negative pairs.",
            "45-55": "Demonstrate Python script unrolling a 2D non-linear spiral manifold using hidden representations.",
            "55-60": "Recap and preview Day 35: Energy-Based Models, Boltzmann Machines, and RBMs."
        },
        "python_demo": {
            "title": "Manifold Unfolding and Contrastive Similarity in NumPy",
            "description": "Simulating how a non-linear latent representation unrolls an entangled data manifold into linear separable coordinates.",
            "code": """import numpy as np

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
print("\\nDeep Representation Learning discovers the intrinsic 1D coordinate t,")
print("flattening the twisted 2D spiral into a simple 1D linear axis!")
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "InfoNCE Contrastive Loss Calculation",
                "statement": "In a self-supervised contrastive learning step, an anchor representation z_a is compared against 1 positive augmented sample z_p and 2 negative samples {z_n1, z_n2}. Cosine similarities are computed as:\nsim(z_a, z_p) = 0.80\nsim(z_a, z_n1) = 0.20\nsim(z_a, z_n2) = -0.10\nThe temperature parameter is tau = 0.50. Calculate: (a) Exponentiated scaled similarities for all pairs, and (b) The InfoNCE loss value.",
                "given": "sim_pos = 0.80, sim_neg1 = 0.20, sim_neg2 = -0.10, tau = 0.50.",
                "formulas": "exp_sim = exp(sim / tau); L = - ln( exp_pos / (exp_pos + sum(exp_neg)) )",
                "steps": [
                    "Step 1: Compute scaled similarities (sim / tau):\n  pos: 0.80 / 0.50 = 1.60.\n  neg1: 0.20 / 0.50 = 0.40.\n  neg2: -0.10 / 0.50 = -0.20.",
                    "Step 2: Compute exponentials:\n  exp_pos = e^(1.60) = 4.9530.\n  exp_neg1 = e^(0.40) = 1.4918.\n  exp_neg2 = e^(-0.20) = 0.8187.",
                    "Step 3: Compute denominator sum:\n  Denom = exp_pos + exp_neg1 + exp_neg2 = 4.9530 + 1.4918 + 0.8187 = 7.2635.",
                    "Step 4: Compute Softmax Probability:\n  P_pos = 4.9530 / 7.2635 = 0.6819 (68.19%).",
                    "Step 5: Compute InfoNCE Loss:\n  L = - ln(0.6819) = 0.3829."
                ],
                "answer": "P_pos = 0.6819, InfoNCE Loss L = 0.3829",
                "interpretation": "The model successfully assigned 68.2% of probability mass to the true positive pair, yielding a low contrastive loss of 0.3829."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Intrinsic Manifold Dimension vs Ambient Pixel Dimension",
                "statement": "An automotive camera produces RGB images of resolution 1024x1024 pixels. The car drives along a single straight highway where camera images vary solely based on 3 physical degrees of freedom: (1) Vehicle forward displacement along road x, (2) Sun elevation angle theta, and (3) Vehicle lane lateral offset y. (a) Calculate the ambient vector space dimension D_ambient, (b) State the intrinsic manifold dimension d_manifold, and (c) Calculate the dimensional compression factor D_ambient / d_manifold.",
                "given": "Image: 1024x1024x3. Physical variations: 3 degrees of freedom.",
                "formulas": "D_ambient = H * W * C; d_manifold = Degrees of Freedom; Compression = D_ambient / d_manifold",
                "steps": [
                    "Step 1: Compute ambient space dimensionality:\n  D_ambient = 1024 * 1024 * 3 = 1,048,576 * 3 = 3,145,728 dimensions.",
                    "Step 2: State intrinsic manifold dimensionality:\n  The underlying physical state space is governed entirely by 3 variables (x, theta, y).\n  Therefore, the intrinsic manifold dimension d_manifold = 3.",
                    "Step 3: Compute dimensional ratio:\n  Ratio = 3,145,728 / 3 = 1,048,576x (over 1 million to 1!)."
                ],
                "answer": "D_ambient = 3,145,728. d_manifold = 3. Compression factor = 1,048,576x.",
                "interpretation": "Despite residing in a 3.1-million-dimensional ambient pixel space, the image data manifold is strictly 3-dimensional. Representation learning isolates these 3 intrinsic axes."
            }
        ],
        "real_world_application": {
            "title": "Self-Supervised Medical Pathology Slide Pre-Training",
            "domain": "Digital Pathology & Oncology Diagnostics",
            "workflow": "Hospital acquires 1 million unlabeled high-resolution cancer biopsy tissue slides -> Self-Supervised Contrastive Learning (SimCLR/MAE) pre-trains a deep ResNet/ViT backbone -> Fine-tuned on only 50 labeled rare sarcoma slides.",
            "benefit": "Achieves 95.2% diagnostic accuracy on ultra-rare cancers where obtaining hundreds of labeled slides is medically impossible."
        },
        "viva_questions": [
            {"q": "What is the Manifold Hypothesis?", "a": "The proposition that high-dimensional real-world data concentrates near low-dimensional continuous sub-manifolds embedded within the high-dimensional ambient space."},
            {"q": "What is a Disentangled Representation?", "a": "A representation where distinct latent coordinates correspond to independent real-world explanatory factors of variation (e.g. azimuth, elevation, shape)."},
            {"q": "Why is linear separability a goal of representation learning?", "a": "Because if representation learning maps complex curved manifolds into a space where classes are linearly separable, a simple linear model can achieve optimal classification without overfitting."},
            {"q": "What is Self-Supervised Learning?", "a": "A learning paradigm where supervisory training signals are automatically generated from the data itself (e.g. predicting masked patches, contrastive view invariance) without human labels."},
            {"q": "What is Contrastive Learning?", "a": "A self-supervised approach that pulls representations of positive pairs (augmented views of same sample) together while pushing negative pairs apart."},
            {"q": "What is the temperature parameter tau in InfoNCE loss?", "a": "A scaling parameter that controls the sharpness of the probability distribution over negative samples, balancing hard negative mining."},
            {"q": "What is downstream Transfer Learning?", "a": "Taking a representation backbone pre-trained on a massive dataset and fine-tuning it on a small target task with limited labels."},
            {"q": "What is the difference between ambient dimension and intrinsic manifold dimension?", "a": "Ambient dimension is the number of raw features (e.g. 1 million pixels); intrinsic dimension is the minimum number of independent degrees of freedom required to generate the data (e.g. 3 physical factors)."},
            {"q": "What is Masked Autoencoding (MAE)?", "a": "A self-supervised vision approach that masks large portions of an image (e.g. 75%) and trains an autoencoder to reconstruct the missing patches."},
            {"q": "How does deep representation learning overcome the 'Curse of Dimensionality'?", "a": "By learning the low-dimensional intrinsic manifold coordinates, restricting the volume of space that needs to be covered by training data."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "The Manifold Hypothesis states that real-world data concentrates near:",
                    "options": ["High-dimensional noise surfaces", "Low-dimensional sub-manifolds embedded in ambient space", "The origin (0, 0)", "Uniform probability spheres"],
                    "ans": "Low-dimensional sub-manifolds embedded in ambient space"
                },
                {
                    "q": "In a disentangled representation, each latent dimension ideally corresponds to:",
                    "options": ["A random linear combination of pixels", "An independent physical factor of variation", "A specific training sample", "The gradient of the loss"],
                    "ans": "An independent physical factor of variation"
                },
                {
                    "q": "The InfoNCE loss function is primarily used in:",
                    "options": ["Supervised linear regression", "Contrastive self-supervised learning", "K-Means clustering", "Single layer perceptrons"],
                    "ans": "Contrastive self-supervised learning"
                }
            ],
            "short_questions": [
                "Explain the Manifold Hypothesis and describe how deep neural networks untangle curved manifolds.",
                "Define Disentangled Representations and explain why they facilitate downstream linear classification."
            ],
            "application_question": "Explain how self-supervised pre-training enables high-accuracy diabetic retinopathy detection with very few labeled ophthalmology images.",
            "homework": "Prove that if all negative similarities in InfoNCE loss are equal to -infinity, the loss reduces identically to zero."
        }
    },
    {
        "day": 35,
        "module": 5,
        "topic_id": "5.8",
        "title": "Current Research Areas: Boltzmann Machines and Deep Belief Networks",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO5",
        "blooms": "Understand / Analyze",
        "prerequisites": "Probability Theory, Energy-based models, Statistical Physics basics, Day 33 concepts",
        "learning_objectives": [
            "Formulate Energy-Based Models (EBMs) and the Boltzmann distribution.",
            "Formulate the general Boltzmann Machine (Ackley, Hinton, Sejnowski 1985) and identify its intractable partition function.",
            "Derive the Restricted Boltzmann Machine (RBM) and analyze its bipartite undirected graph structure.",
            "Formulate Gibbs Sampling and conditional independence properties in RBMs: P(h|v) and P(v|h)."
        ],
        "key_terminology": [
            "Energy-Based Model (EBM)", "Boltzmann Distribution", "Partition Function Z",
            "Restricted Boltzmann Machine (RBM)", "Bipartite Graph", "Gibbs Sampling", "Conditional Independence"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 20 (Section 20.1); Aggarwal: Chapter 6; KTU CST414 Syllabus Model Question Paper Part B (Q20b).",
        "notes": {
            "energy_based_models": (
                "An Energy-Based Model associates a scalar energy E(x) with each configuration of variables. High-probability states correspond to low energy levels.\n\n"
                "The Boltzmann Distribution defines probability via the Gibbs / Boltzmann distribution:\n"
                "P(x) = exp( - E(x) ) / Z\n"
                "where Z = sum_x exp( - E(x) ) is the normalizing Partition Function.\n"
                "Intractability: Evaluating Z requires summing over 2^D possible binary states, which is computationally intractable for D > 40."
            ),
            "general_boltzmann_machine": (
                "A Boltzmann Machine (1985) is an undirected graphical model containing visible units v (observed data) and hidden units h (latent representations):\n"
                "Energy Function:\n"
                "E(v, h) = - v^T * W * h - (1/2) * v^T * L * v - (1/2) * h^T * J * h - b^T * v - c^T * h\n"
                "Intractable Learning: Because units within the same layer are interconnected (matrices L and J), computing P(h|v) requires extensive Markov Chain Monte Carlo (MCMC) sampling, making training extremely slow."
            ),
            "restricted_boltzmann_machine": (
                "Smolensky (1986) introduced the Restricted Boltzmann Machine (RBM), which forbids intra-layer connections (L = 0 and J = 0).\n\n"
                "Bipartite Graph Structure:\n"
                "Connections exist strictly between visible units v and hidden units h.\n\n"
                "Energy Function of RBM:\n"
                "E(v, h) = - v^T * W * h - b^T * v - c^T * h = - sum_i sum_j v_i * W_ij * h_j - sum_i b_i * v_i - sum_j c_j * h_j\n\n"
                "Conditional Independence:\n"
                "Given visible layer v, all hidden units h_j are mutually independent:\n"
                "P(h_j = 1 | v) = sigma( sum_i W_ij * v_i + c_j )\n\n"
                "Given hidden layer h, all visible units v_i are mutually independent:\n"
                "P(v_i = 1 | h) = sigma( sum_j W_ij * h_j + b_i )\n\n"
                "This allows parallel, exact block Gibbs sampling in a single step!"
            ),
            "exam_important_points": [
                "KTU Exam Q20(b): Compare Boltzmann Machine with Deep Belief Network.",
                "Diagram of general Boltzmann machine (with intra-layer connections) vs RBM (bipartite graph).",
                "Formulation of RBM energy function E(v, h).",
                "Derivation of conditional probabilities P(h_j=1|v) and P(v_i=1|h) showing conditional independence."
            ]
        },
        "teaching_script": {
            "00-05": "Connect neural networks to statistical physics: 'Why did deep learning pioneers model neurons like magnetic spins in iron?' Introduce Energy-Based Models.",
            "05-20": "Write the Boltzmann distribution: P(x) = exp(-E(x))/Z. Explain the Partition Function Z and why 2^D sums are computationally impossible.",
            "20-35": "Draw the General Boltzmann Machine with intra-layer recurrent loops. Show why it is slow. Erase intra-layer loops to reveal the RBM bipartite graph.",
            "35-45": "Derive the RBM conditional independence equations: P(h|v) = prod P(h_j|v) and P(v|h) = prod P(v_i|h). Show Block Gibbs sampling.",
            "45-55": "Demonstrate Python script computing RBM energy and running alternating Gibbs sampling.",
            "55-60": "Recap and preview Day 36: Deep Belief Networks and Contrastive Divergence."
        },
        "python_demo": {
            "title": "Restricted Boltzmann Machine (RBM) Block Gibbs Sampling in NumPy",
            "description": "Implementation of an RBM computing conditional sampling P(h|v) and P(v|h) and energy evaluations.",
            "code": """import numpy as np

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
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "RBM Energy and Hidden Activation Probability Calculation",
                "statement": "An RBM has 2 visible units and 2 hidden units. Parameters are:\nWeight matrix W = [[0.8, -0.4], [0.5, 0.6]]\nVisible bias b = [0.1, -0.2]^T\nHidden bias c = [-0.1, 0.3]^T\nGiven visible vector v = [1, 0]^T: (a) Calculate the activation probabilities P(h_1 = 1 | v) and P(h_2 = 1 | v), and (b) If hidden vector is sampled as h = [1, 1]^T, calculate the energy E(v, h).",
                "given": "v = [1, 0]^T, W = [[0.8, -0.4], [0.5, 0.6]], b = [0.1, -0.2]^T, c = [-0.1, 0.3]^T.",
                "formulas": "P(h_j = 1 | v) = sigma( sum_i v_i * W_ij + c_j ); E(v, h) = - v^T * W * h - b^T * v - c^T * h",
                "steps": [
                    "Step 1: Compute input to hidden units:\n  z_h1 = (v_1 * W_11 + v_2 * W_21) + c_1 = (1 * 0.8 + 0 * 0.5) + (-0.1) = 0.8 - 0.1 = 0.70.\n  z_h2 = (v_1 * W_12 + v_2 * W_22) + c_2 = (1 * -0.4 + 0 * 0.6) + 0.3 = -0.4 + 0.3 = -0.10.",
                    "Step 2: Compute conditional probabilities:\n  P(h_1 = 1 | v) = sigma(0.70) = 1 / (1 + e^(-0.70)) = 1 / (1 + 0.49659) = 0.6682.\n  P(h_2 = 1 | v) = sigma(-0.10) = 1 / (1 + e^(0.10)) = 1 / (1 + 1.10517) = 0.4750.",
                    "Step 3: Compute Energy for v = [1, 0]^T and h = [1, 1]^T:\n  Term 1: v^T * W = [1, 0] * [[0.8, -0.4], [0.5, 0.6]] = [0.8, -0.4].\n  (v^T * W) * h = [0.8, -0.4] * [1, 1]^T = 0.8 - 0.4 = 0.40.\n  Term 2: b^T * v = 0.1 * 1 + (-0.2) * 0 = 0.10.\n  Term 3: c^T * h = (-0.1 * 1) + (0.3 * 1) = 0.20.\n  Total Energy E(v, h) = - 0.40 - 0.10 - 0.20 = -0.70."
                ],
                "answer": "P(h_1=1|v) = 0.6682, P(h_2=1|v) = 0.4750, Energy E(v, h) = -0.70",
                "interpretation": "The negative energy value (-0.70) indicates a favorable, low-energy configuration under the learned Boltzmann distribution."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "Exact Partition Function and Joint Probability in 2x2 Boltzmann Distribution",
                "statement": "A tiny toy RBM has 1 visible unit v in {0, 1} and 1 hidden unit h in {0, 1}. The energy function is E(v, h) = - 2.0 * v * h. (a) Enumerate all 4 possible joint states (v, h) and compute their energies, (b) Calculate the unnormalized Boltzmann weights exp(-E) for all states, (c) Calculate the exact Partition Function Z, and (d) Calculate the exact joint probability distribution P(v, h).",
                "given": "v in {0, 1}, h in {0, 1}, E(v, h) = - 2.0 * v * h.",
                "formulas": "E(v, h) = - 2 * v * h; Weight = e^(-E); Z = sum exp(-E); P(v, h) = exp(-E) / Z",
                "steps": [
                    "Step 1: Enumerate 4 states and energies:\n  State 1: (0, 0) -> E = -2 * 0 * 0 = 0.0 -> exp(-0) = 1.000.\n  State 2: (0, 1) -> E = -2 * 0 * 1 = 0.0 -> exp(-0) = 1.000.\n  State 3: (1, 0) -> E = -2 * 1 * 0 = 0.0 -> exp(-0) = 1.000.\n  State 4: (1, 1) -> E = -2 * 1 * 1 = -2.0 -> exp(-(-2)) = e^(2.0) = 7.3891.",
                    "Step 2: Calculate Partition Function Z:\n  Z = 1.000 + 1.000 + 1.000 + 7.3891 = 10.3891.",
                    "Step 3: Calculate joint probabilities:\n  P(0, 0) = 1.0 / 10.3891 = 0.0963 (9.63%)\n  P(0, 1) = 1.0 / 10.3891 = 0.0963 (9.63%)\n  P(1, 0) = 1.0 / 10.3891 = 0.0963 (9.63%)\n  P(1, 1) = 7.3891 / 10.3891 = 0.7112 (71.12%)."
                ],
                "answer": "Partition Function Z = 10.3891. P(0,0)=P(0,1)=P(1,0)=0.0963, P(1,1)=0.7112.",
                "interpretation": "Because state (1, 1) has the lowest energy (-2.0), the Boltzmann distribution concentrates over 71% of the total probability mass onto that single state."
            }
        ],
        "real_world_application": {
            "title": "Collaborative Filtering Netflix Movie Recommendation",
            "domain": "Recommender Systems & E-Commerce",
            "workflow": "User movie rating history -> Modeled as visible units of a Conditional RBM -> Hidden units extract latent genre preferences (sci-fi, romance) -> Gibbs sampling reconstructs missing movie ratings.",
            "benefit": "Was a core winning algorithmic component of the famous million-dollar Netflix Prize competition (Salakhutdinov et al., 2007)."
        },
        "viva_questions": [
            {"q": "What is an Energy-Based Model (EBM)?", "a": "A probabilistic model that defines probability distributions by assigning a scalar energy to configurations, where low energy corresponds to high probability."},
            {"q": "What is the Partition Function Z in a Boltzmann Machine?", "a": "The normalization denominator Z = sum exp(-E(x)) that ensures probabilities sum to 1.0 across all possible configurations."},
            {"q": "Why is training a general Boltzmann Machine computationally intractable?", "a": "Because intra-layer connections prevent conditional independence, making the exact partition function and posterior inference require prohibitive exponential sums."},
            {"q": "What structural restriction defines a Restricted Boltzmann Machine (RBM)?", "a": "No connections are permitted between units within the same layer (no visible-to-visible or hidden-to-hidden connections), forming a bipartite graph."},
            {"q": "What major mathematical benefit arises from the RBM bipartite restriction?", "a": "Conditional independence: all hidden units are conditionally independent given the visible units, and all visible units are conditionally independent given the hidden units."},
            {"q": "What is Block Gibbs Sampling in an RBM?", "a": "Alternatingly sampling all hidden units in parallel given the visible units, and then all visible units in parallel given the hidden units."},
            {"q": "What is the formula for P(h_j = 1 | v) in a binary RBM?", "a": "P(h_j = 1 | v) = sigma( sum_i W_ij * v_i + c_j )."},
            {"q": "What role did RBMs play in the deep learning renaissance of 2006?", "a": "Geoffrey Hinton used stacked RBMs to perform greedy layer-wise unsupervised pre-training, proving for the first time that deep neural networks could be trained effectively."},
            {"q": "Can RBMs handle continuous real-valued data?", "a": "Yes; Gaussian-Bernoulli RBMs replace binary visible units with continuous units with Gaussian noise."},
            {"q": "What is the physical inspiration behind Boltzmann Machines?", "a": "The Ising spin glass model and thermal statistical physics of interacting particles."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "A Restricted Boltzmann Machine (RBM) is characterized by having:",
                    "options": ["Dense intra-layer connections within hidden units", "No connections between units in the same layer (bipartite graph)", "Recurrent feedback over time", "Strictly directed acyclic arrows"],
                    "ans": "No connections between units in the same layer (bipartite graph)"
                },
                {
                    "q": "In an energy-based model, configurations with the lowest energy correspond to:",
                    "options": ["The lowest probability", "The highest probability", "Zero probability", "Undefined probabilities"],
                    "ans": "The highest probability"
                },
                {
                    "q": "In an RBM, given the visible vector v, the hidden units are:",
                    "options": ["Mutually conditionally independent", "Linearly dependent on biases", "Trained with backpropagation", "Set to zero"],
                    "ans": "Mutually conditionally independent"
                }
            ],
            "short_questions": [
                "Contrast the graph structures of a General Boltzmann Machine with a Restricted Boltzmann Machine (RBM).",
                "Explain the concept of conditional independence in RBMs and derive the formula for P(h_j = 1 | v)."
            ],
            "application_question": "Explain how RBMs are applied to model collaborative filtering in user-movie recommendation engines.",
            "homework": "Given an RBM with v = [0, 1] and W = [[1, 2], [3, 4]] with zero biases, compute the hidden activation probabilities P(h_1=1|v) and P(h_2=1|v)."
        }
    },
    {
        "day": 36,
        "module": 5,
        "topic_id": "5.9",
        "title": "Current Research Areas: Deep Belief Networks (DBNs) and Course Synthesis",
        "planned_hours": 1,
        "duration_mins": 60,
        "co": "CO5",
        "blooms": "Understand / Evaluate",
        "prerequisites": "Restricted Boltzmann Machines, Energy-Based Models, Day 35 concepts",
        "learning_objectives": [
            "Formulate the Deep Belief Network (DBN) architecture (Hinton, Osindero, Teh, 2006).",
            "Examine the hybrid graphical structure of DBNs: directed top-down generative connections in lower layers and undirected connections in the top two layers.",
            "Formulate Contrastive Divergence (CD-k) for ultra-fast RBM weight updates.",
            "Analyze the Greedy Layer-Wise Unsupervised Pre-training and Supervised Fine-Tuning protocol that sparked the modern Deep Learning revolution."
        ],
        "key_terminology": [
            "Deep Belief Network (DBN)", "Hybrid Graphical Model", "Contrastive Divergence (CD-k)",
            "Greedy Layer-Wise Pre-training", "Positive Phase", "Negative Phase", "Fine-Tuning", "Course Synthesis"
        ],
        "textbook_ref": "Goodfellow et al.: Deep Learning, Chapter 20 (Section 20.3); Hinton et al.: A Fast Learning Algorithm for Deep Belief Nets (Neural Computation 2006); KTU CST414 Syllabus Model Question Paper Part B (Q20b).",
        "notes": {
            "dbn_architecture": (
                "A Deep Belief Network (Hinton et al., 2006) is a deep generative hybrid graphical model formed by stacking multiple Restricted Boltzmann Machines (RBMs).\n\n"
                "Hybrid Graphical Structure:\n"
                "- Top two layers form an UNDIRECTED bipartite graph (an RBM) that acts as an associative memory.\n"
                "- All lower layers form DIRECTED acyclic generative connections pointing downward from hidden layers to visible layers:\n"
                "  P(x, h^1, h^2, ..., h^L) = P(h^{L-1}, h^L) * product_{l=1}^{L-2} P(h^l | h^{l+1}) * P(x | h^1)"
            ),
            "contrastive_divergence": (
                "The log-likelihood gradient for an RBM is:\n"
                "del ln P(v) / del W = E_data [ v * h^T ] - E_model [ v * h^T ]\n"
                "- Positive Phase (Data-dependent expectation): Easily evaluated using training sample v and P(h|v).\n"
                "- Negative Phase (Model distribution expectation): Requires running an MCMC chain to thermal equilibrium (infinity steps).\n\n"
                "Contrastive Divergence (CD-k, Hinton 2002):\n"
                "Replaces infinite MCMC sampling by running only k steps of Gibbs sampling (typically k = 1) starting from the data vector v_0:\n"
                "1. Sample h_0 ~ P(h | v_0)\n"
                "2. Sample v_1 ~ P(v | h_0) (Reconstruction)\n"
                "3. Sample h_1 ~ P(h | v_1)\n"
                "Weight update: delta W = eta * ( v_0 * h_0^T - v_1 * h_1^T )\n"
                "Astonishingly fast, robust, and revolutionized deep unsupervised training!"
            ),
            "greedy_layerwise_pretraining_protocol": (
                "The 2006 breakthrough protocol that launched the deep learning era:\n"
                "1. Step 1: Train RBM 1 on raw data v to learn feature representation h^1 using CD-1.\n"
                "2. Step 2: Freeze RBM 1. Use its hidden states h^1 as input data to train RBM 2.\n"
                "3. Step 3: Repeat greedily layer by layer for L layers.\n"
                "4. Step 4: Unroll into a feedforward deep network, attach a supervised Softmax classification layer, and perform end-to-end backpropagation Fine-Tuning!\n\n"
                "Why it worked: Greedy pre-training initialized deep weights into a basin of attraction close to a good local minimum, completely bypassing the vanishing gradient barrier of 2006."
            ),
            "course_synthesis": (
                "Complete KTU CST414 Course Synthesis:\n"
                "- Module 1: Foundations (Perceptrons, MLPs, Backpropagation, Optimization pathologies).\n"
                "- Module 2: Deep Learning Engine (Advanced optimizers: Momentum, NAG, Adam; Regularization: L1/L2, Dropout, Initialization).\n"
                "- Module 3: Spatial Intelligence (CNNs, Pooling, Equivariance, Bottlenecks, Winograd, MobileNet).\n"
                "- Module 4: Temporal & Sequence Intelligence (RNNs, Seq2Seq, BPTT, LSTMs, GRUs).\n"
                "- Module 5: Frontiers & Applications (Vision, Speech, NLP, Word2Vec, GloVe, Autoencoders, DBNs)."
            ),
            "exam_important_points": [
                "Hybrid directed/undirected structure of Deep Belief Networks.",
                "Contrastive Divergence CD-1 algorithm steps: positive phase vs negative phase.",
                "Greedy layer-wise unsupervised pre-training and supervised fine-tuning procedure.",
                "Complete comparison between Boltzmann Machine, RBM, and DBN."
            ]
        },
        "teaching_script": {
            "00-05": "Introduce the historical milestone: '2006 was the year Deep Learning was born. How did Hinton stack RBMs to create DBNs?'",
            "05-20": "Draw the Deep Belief Network hybrid graph on blackboard: undirected top RBM, directed top-down generative connections below.",
            "20-35": "Derive Contrastive Divergence (CD-1): positive phase v0*h0^T minus negative reconstructed phase v1*h1^T.",
            "35-45": "Walk through the 4-step Greedy Layer-Wise Pre-training and Fine-Tuning protocol. Explain why it solved vanishing gradients in 2006.",
            "45-55": "Demonstrate Python script implementing Contrastive Divergence CD-1.",
            "55-60": "Course Synthesis: review the entire 36-hour journey from Frank Rosenblatt's 1958 Perceptron to modern Deep Belief Networks and Transformers. Thank students!"
        },
        "python_demo": {
            "title": "Contrastive Divergence (CD-1) Algorithm in NumPy",
            "description": "Full implementation of Hinton's CD-1 algorithm training an RBM on binary pattern vectors.",
            "code": """import numpy as np

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

print("Initial Weights:\\n", np.round(W, 3))

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

print("\\nTrained Weights after 20 epochs of CD-1:\\n", np.round(W, 3))
print("Notice how Hidden Unit 0 and Hidden Unit 1 specialize into the two distinct patterns!")
"""
        },
        "numerical_problems": [
            {
                "type": "Basic",
                "title": "Single Step Contrastive Divergence (CD-1) Weight Update Calculation",
                "statement": "An RBM has 2 visible units and 1 hidden unit with scalar weights W = [0.2, -0.1]^T (biases are zero). A training sample is v_0 = [1.0, 1.0]. Learning rate is eta = 0.5. (a) In the positive phase, compute hidden activation probability p_h0, (b) In the negative phase, reconstruct visible vector probabilities p_v1 using hidden activation h_0 = p_h0, and (c) Compute the negative hidden probability p_h1 and the resulting CD-1 weight gradient update delta_W.",
                "given": "v_0 = [1, 1], W = [0.2, -0.1]^T, eta = 0.5.",
                "formulas": "p_h0 = sigma(v_0 . W); p_v1 = sigma(h_0 * W^T); p_h1 = sigma(p_v1 . W); delta_W = eta * (v_0^T * p_h0 - p_v1^T * p_h1)",
                "steps": [
                    "Step 1: Positive Phase:\n  z_h0 = (1.0 * 0.2) + (1.0 * -0.1) = 0.10.\n  p_h0 = 1 / (1 + e^(-0.10)) = 1 / (1 + 0.90484) = 0.5250.\n  Positive correlation: v_0^T * p_h0 = [1, 1]^T * 0.5250 = [0.5250, 0.5250]^T.",
                    "Step 2: Negative Phase (Reconstruction):\n  z_v1_1 = 0.5250 * 0.2 = 0.1050 -> p_v1_1 = sigma(0.1050) = 0.5262.\n  z_v1_2 = 0.5250 * (-0.1) = -0.0525 -> p_v1_2 = sigma(-0.0525) = 0.4869.\n  p_v1 = [0.5262, 0.4869].",
                    "Step 3: Negative Hidden Activation:\n  z_h1 = (0.5262 * 0.2) + (0.4869 * -0.1) = 0.1052 - 0.0487 = 0.0565.\n  p_h1 = sigma(0.0565) = 0.5141.\n  Negative correlation: p_v1^T * p_h1 = [0.5262 * 0.5141, 0.4869 * 0.5141]^T = [0.2705, 0.2503]^T.",
                    "Step 4: Compute CD-1 Weight Update:\n  delta_W = 0.5 * ( [0.5250, 0.5250]^T - [0.2705, 0.2503]^T )\n          = 0.5 * [0.2545, 0.2747]^T = [0.1272, 0.1374]^T.",
                    "Step 5: Compute updated weights:\n  W_new = [0.2 + 0.1272, -0.1 + 0.1374]^T = [0.3272, 0.0374]^T."
                ],
                "answer": "Positive stats = [0.525, 0.525]^T, Negative stats = [0.271, 0.250]^T, Updated W = [0.3272, 0.0374]^T",
                "interpretation": "Contrastive Divergence pushed weights in the direction of the empirical data distribution and away from the 1-step model hallucination."
            },
            {
                "type": "Advanced (KTU Exam Level)",
                "title": "KTU Model Paper Q20(b): Comparative Architectural Analysis of BM vs DBN",
                "statement": "Compare Boltzmann Machines (BM), Restricted Boltzmann Machines (RBM), and Deep Belief Networks (DBN) across: (1) Graph topology and directionality, (2) Parameter count scaling, (3) Training complexity and algorithm, and (4) Generative capability.",
                "given": "Architectures: Boltzmann Machine, RBM, DBN.",
                "formulas": "Tabular synthesis and structural contrast.",
                "steps": [
                    "Step 1: Graph Topology and Directionality:\n  - Boltzmann Machine: Undirected graph with intra-layer connections within visible and hidden units.\n  - RBM: Undirected bipartite graph with zero intra-layer connections.\n  - DBN: Hybrid model; top two layers are an undirected RBM; all lower layers are directed top-down generative connections.",
                    "Step 2: Training Algorithm and Complexity:\n  - Boltzmann Machine: Intractable MCMC requiring infinite Gibbs sampling; exponential complexity O(2^D).\n  - RBM: Contrastive Divergence (CD-k), linear complexity O(k * D_v * D_h).\n  - DBN: Greedy layer-wise unsupervised pre-training using CD-k on stacked RBMs, followed by supervised backpropagation fine-tuning.",
                    "Step 3: Inference and Feature Extraction:\n  - Boltzmann Machine: Slow stochastic inference.\n  - RBM: Exact, one-step parallel block Gibbs sampling: P(h|v) in O(1) MCMC steps.\n  - DBN: Feedforward bottom-up inference pass for feature extraction; top-down directed ancestral sampling for generation.",
                    "Step 4: Historical Significance:\n  - DBN demonstrated in 2006 that deep networks with many layers could be trained effectively, breaking the 15-year neural network winter."
                ],
                "answer": "BM: Undirected with intra-layer loops (intractable). RBM: Undirected bipartite (CD-k). DBN: Hybrid directed/undirected (Greedy pre-training + fine-tuning).",
                "interpretation": "DBNs revolutionized AI by introducing layer-wise pre-training on stacked RBMs, bridging statistical physics with modern deep learning."
            }
        ],
        "real_world_application": {
            "title": "Industrial Turbine Fault Diagnosis using Deep Belief Networks",
            "domain": "Heavy Industry & Vibration Telemetry Analysis",
            "workflow": "High-dimensional multi-sensor vibration readings from power plant turbines -> 3-layer DBN pre-trained on unlabelled baseline vibrations via CD-1 -> Top layer fine-tuned with 50 labeled bearing failure events -> Detects sub-surface rotor imbalances.",
            "benefit": "Enables predictive maintenance that avoids multi-million-dollar turbine rotor catastrophic blade fracture."
        },
        "viva_questions": [
            {"q": "What is a Deep Belief Network (DBN)?", "a": "A deep generative hybrid graphical model consisting of stacked RBMs, where the top two layers form an undirected associative memory and lower layers have directed top-down generative connections."},
            {"q": "Who invented Deep Belief Networks and in what year?", "a": "Geoffrey Hinton, Simon Osindero, and Yee-Whye Teh in 2006."},
            {"q": "What is Contrastive Divergence (CD-k)?", "a": "An approximation algorithm for training RBMs that replaces full MCMC sampling by running only k steps of Gibbs sampling initialized at the data vector."},
            {"q": "What are the two phases of Contrastive Divergence?", "a": "The Positive Phase (measures correlations on true data v0*h0^T) and the Negative Phase (measures correlations on reconstructed data v_k*h_k^T)."},
            {"q": "What is Greedy Layer-Wise Unsupervised Pre-training?", "a": "Training each layer of a deep network one by one from bottom to top as an independent RBM on the representations of the previous layer, before unrolling into a supervised network."},
            {"q": "Why did greedy layer-wise pre-training spark the deep learning revolution in 2006?", "a": "It provided a reliable way to initialize the weights of deep neural networks near good local minima, overcoming the vanishing gradient problem that previously prevented training networks with more than 2-3 layers."},
            {"q": "Is greedy layer-wise pre-training still mandatory for modern deep learning?", "a": "No; modern techniques like ReLU activations, He initialization, Residual connections, and Adam optimizer allow deep networks to be trained directly end-to-end from scratch."},
            {"q": "How does generation work in a Deep Belief Network?", "a": "Alternating Gibbs sampling is run in the top-level undirected RBM until equilibrium, and then ancestral sampling flows down through the directed generative connections to generate visible pixels."},
            {"q": "What is the difference between a Deep Belief Network and a Deep Boltzmann Machine (DBM)?", "a": "A DBN has directed connections in its lower layers and undirected connections only at the top; a DBM is completely undirected across all layers."},
            {"q": "Summarize the overarching trajectory of KTU CST414 Deep Learning.", "a": "From simple linear perceptrons and MLPs, through optimization and regularization breakthroughs, to spatial convolutional vision models, temporal recurrent sequence models, and deep generative representations."}
        ],
        "assessment": {
            "mcqs": [
                {
                    "q": "The graphical structure of a Deep Belief Network (DBN) is:",
                    "options": ["Entirely directed", "Entirely undirected", "Hybrid: undirected in the top two layers and directed top-down in lower layers", "A circular recurrent ring"],
                    "ans": "Hybrid: undirected in the top two layers and directed top-down in lower layers"
                },
                {
                    "q": "Contrastive Divergence (CD-1) replaces infinite MCMC sampling by running how many steps of Gibbs sampling?",
                    "options": ["1 step", "100 steps", "1000 steps", "0 steps"],
                    "ans": "1 step"
                },
                {
                    "q": "The 2006 deep learning breakthrough achieved by Hinton et al. utilized DBNs through:",
                    "options": ["Greedy layer-wise unsupervised pre-training followed by supervised fine-tuning", "Pure genetic algorithms", "Random weight search", "Hand-crafted feature rules"],
                    "ans": "Greedy layer-wise unsupervised pre-training followed by supervised fine-tuning"
                }
            ],
            "short_questions": [
                "Explain the hybrid graphical architecture of Deep Belief Networks and contrast it with Deep Boltzmann Machines.",
                "Describe the Contrastive Divergence (CD-1) algorithm and write down the positive and negative phase weight update formula."
            ],
            "application_question": "Synthesize how the core principles of CST414 (representation learning, gradient optimization, architectural priors) culminate in modern generative multimodal AI systems.",
            "homework": "Summarize the 36-lecture KTU CST414 curriculum by writing a one-paragraph technical synthesis for each of the five modules."
        }
    }
]

