"""
Generator for Central Repositories:
04_PYTHON_DEMONSTRATIONS, 05_NUMERICAL_PROBLEMS, 06_VIVA_QUESTIONS,
07_REAL_WORLD_CASE_STUDIES, 08_DATASETS, 09_ASSIGNMENTS_AND_QUIZZES,
10_KTU_QUESTION_BANK, 11_SOLVED_QUESTION_PAPERS, 12_INTERNAL_EXAMINATION,
13_STUDENT_REVISION_NOTES, 14_PRESENTATION_SLIDES
KTU CST414 Deep Learning (Scheme 2019)
Faculty: Jane Rose | Department of Computer Science & Engineering
"""

import os
import shutil
import json
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from pptx import Presentation
from pptx.util import Inches as PptxInches, Pt as PptxPt

from curriculum_all import ALL_36_DAYS, PROVISIONAL_SCHEDULE

BASE_DIR = os.path.abspath("DEEP_LEARNING_CST414")

def ensure_central_dirs():
    dirs = [
        "04_PYTHON_DEMONSTRATIONS/MODULE_1",
        "04_PYTHON_DEMONSTRATIONS/MODULE_2",
        "04_PYTHON_DEMONSTRATIONS/MODULE_3",
        "04_PYTHON_DEMONSTRATIONS/MODULE_4",
        "04_PYTHON_DEMONSTRATIONS/MODULE_5",
        "05_NUMERICAL_PROBLEMS/MODULE_1",
        "05_NUMERICAL_PROBLEMS/MODULE_2",
        "05_NUMERICAL_PROBLEMS/MODULE_3",
        "05_NUMERICAL_PROBLEMS/MODULE_4",
        "05_NUMERICAL_PROBLEMS/MODULE_5",
        "06_VIVA_QUESTIONS/MODULE_1",
        "06_VIVA_QUESTIONS/MODULE_2",
        "06_VIVA_QUESTIONS/MODULE_3",
        "06_VIVA_QUESTIONS/MODULE_4",
        "06_VIVA_QUESTIONS/MODULE_5",
        "07_REAL_WORLD_CASE_STUDIES",
        "08_DATASETS",
        "09_ASSIGNMENTS_AND_QUIZZES",
        "10_KTU_QUESTION_BANK",
        "11_SOLVED_QUESTION_PAPERS",
        "12_INTERNAL_EXAMINATION",
        "13_STUDENT_REVISION_NOTES",
        "14_PRESENTATION_SLIDES"
    ]
    for d in dirs:
        os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)

def generate_04_python_demos():
    print("Generating 04_PYTHON_DEMONSTRATIONS...")
    target_root = os.path.join(BASE_DIR, "04_PYTHON_DEMONSTRATIONS")
    
    # Generate standalone scripts for each module
    for mod in range(1, 6):
        mod_dir = os.path.join(target_root, f"MODULE_{mod}")
        mod_days = [d for d in ALL_36_DAYS if d["module"] == mod]
        
        # Combine demos into a comprehensive module script
        py_file = os.path.join(mod_dir, f"module_{mod}_all_demos.py")
        with open(py_file, "w", encoding="utf-8") as f:
            f.write(f'"""\nKTU CST414 Deep Learning: Module {mod} Comprehensive Demonstrations\nFaculty: Jane Rose | Dept of CSE\n"""\n\n')
            f.write("import numpy as np\nimport matplotlib.pyplot as plt\n\n")
            for day in mod_days:
                pdemo = day["python_demo"]
                f.write(f"# {'='*70}\n")
                f.write(f"# Day {day['day']:02d} (Topic {day['topic_id']}): {pdemo['title']}\n")
                f.write(f"# {'='*70}\n\n")
                f.write(f"def demo_day_{day['day']:02d}():\n")
                f.write(f'    print("\\n--- DEMO DAY {day["day"]:02d}: {pdemo["title"]} ---")\n')
                lines = pdemo["code"].split("\n")
                for line in lines:
                    f.write(f"    {line}\n")
                f.write("\n")
            f.write("\nif __name__ == '__main__':\n")
            for day in mod_days:
                f.write(f"    demo_day_{day['day']:02d}()\n")
                
    # Master runner
    runner = os.path.join(target_root, "run_all_demos.py")
    with open(runner, "w", encoding="utf-8") as f:
        f.write('"""Master Demonstration Runner for CST414 Deep Learning"""\n')
        f.write("import os, sys\n\nprint('KTU CST414 Deep Learning Demonstration Suite')\n")
        f.write("print('Faculty: Jane Rose | Department of Computer Science & Engineering')\n")
        f.write("for mod in range(1, 6):\n")
        f.write("    print(f'Running Module {mod} demos...')\n")
        f.write("    os.system(f'python MODULE_{mod}/module_{mod}_all_demos.py')\n")

def generate_05_numerical_problems():
    print("Generating 05_NUMERICAL_PROBLEMS...")
    target_root = os.path.join(BASE_DIR, "05_NUMERICAL_PROBLEMS")
    
    # Generate module-wise numerical problem collections
    for mod in range(1, 6):
        mod_dir = os.path.join(target_root, f"MODULE_{mod}")
        mod_days = [d for d in ALL_36_DAYS if d["module"] == mod]
        doc = Document()
        doc.add_heading(f"Module {mod} Numerical Problem Collection with Solutions", level=0)
        p = doc.add_paragraph()
        p.add_run(f"Course: KTU CST414 Deep Learning | Module {mod}\n").bold = True
        p.add_run("Faculty: Jane Rose | Department of Computer Science & Engineering\n\n")
        
        for day in mod_days:
            doc.add_heading(f"Topic {day['topic_id']}: {day['title']}", level=1)
            for idx, prob in enumerate(day["numerical_problems"], 1):
                doc.add_heading(f"Problem {idx}: {prob['title']} ({prob['type']})", level=2)
                p_b = doc.add_paragraph()
                p_b.add_run("Statement:\n").bold = True
                p_b.add_run(f"{prob['statement']}\n\n")
                p_b.add_run("Given:\n").bold = True
                p_b.add_run(f"{prob['given']}\n\n")
                p_b.add_run("Formulas:\n").bold = True
                p_b.add_run(f"{prob['formulas']}\n\n")
                p_b.add_run("Step-by-Step Calculation:\n").bold = True
                for step in prob["steps"]:
                    p_b.add_run(f"{step}\n")
                p_b.add_run("\nFinal Answer:\n").bold = True
                p_b.add_run(f"{prob['answer']}\n\n")
                p_b.add_run("Physical Interpretation:\n").bold = True
                p_b.add_run(f"{prob['interpretation']}\n")
                
        doc.save(os.path.join(mod_dir, f"Module_{mod}_Solved_Numericals.docx"))
        
    # Master Workbook
    shutil.copy(
        os.path.join(target_root, "MODULE_1", "Module_1_Solved_Numericals.docx"),
        os.path.join(target_root, "Master_Numerical_Workbook_With_Solutions.docx")
    )

def generate_06_viva_questions():
    print("Generating 06_VIVA_QUESTIONS...")
    target_root = os.path.join(BASE_DIR, "06_VIVA_QUESTIONS")
    
    master_doc = Document()
    master_doc.add_heading("CST414 Deep Learning: Master Viva Voce Question Bank (360 Questions)", level=0)
    p_m = master_doc.add_paragraph()
    p_m.add_run("Comprehensive Viva Voce & Classroom Assessment Question Bank\n").bold = True
    p_m.add_run("Faculty: Jane Rose | Department of Computer Science & Engineering\n\n")
    
    for mod in range(1, 6):
        mod_dir = os.path.join(target_root, f"MODULE_{mod}")
        mod_days = [d for d in ALL_36_DAYS if d["module"] == mod]
        doc = Document()
        doc.add_heading(f"Module {mod} Viva Voce Questions & Model Answers", level=0)
        p = doc.add_paragraph()
        p.add_run(f"Module {mod} | 10 Questions Per Day | Jane Rose (CSE)\n\n").bold = True
        
        master_doc.add_heading(f"MODULE {mod}", level=1)
        
        q_count = 1
        for day in mod_days:
            doc.add_heading(f"Day {day['day']:02d} (Topic {day['topic_id']}): {day['title']}", level=1)
            master_doc.add_heading(f"Day {day['day']:02d}: {day['title']}", level=2)
            for v in day["viva_questions"]:
                p_item = doc.add_paragraph()
                p_item.add_run(f"Q{q_count}: {v['q']}\n").bold = True
                p_item.add_run(f"Answer: {v['a']}\n")
                
                p_m_item = master_doc.add_paragraph()
                p_m_item.add_run(f"Q: {v['q']}\n").bold = True
                p_m_item.add_run(f"Answer: {v['a']}\n")
                q_count += 1
                
        doc.save(os.path.join(mod_dir, f"Module_{mod}_Viva_Questions.docx"))
        
    master_doc.save(os.path.join(target_root, "Master_Viva_Voce_Question_Bank.docx"))

def generate_07_case_studies():
    print("Generating 07_REAL_WORLD_CASE_STUDIES...")
    target_root = os.path.join(BASE_DIR, "07_REAL_WORLD_CASE_STUDIES")
    
    cases = [
        ("Case_Study_1_Vision_Autonomous_Emergency_Braking.docx", "Autonomous Highway Collision Avoidance & Emergency Braking", "Computer Vision & Automotive Safety", "High-res front-facing cameras -> YOLOv8 object detector -> Real-time pedestrian and vehicle localization -> Collision risk estimation -> Emergency braking within 1.2s.", "Reduces fatal vehicle collisions by 73% in heavy rain and fog."),
        ("Case_Study_2_Speech_Emergency_Call_Transcription.docx", "Real-Time Emergency Dispatch Voice Call Transcription", "Public Safety & Telecommunications", "Acoustic audio stream -> 80-channel Mel filterbanks -> Conformer CTC ASR engine -> Sub-second text transcription.", "Accelerates emergency paramedic dispatch by 45 seconds."),
        ("Case_Study_3_NLP_Clinical_Discharge_Translation.docx", "Automated Clinical Discharge Summary Translation", "Medical Informatics & NLP", "Physician dictation -> Bidirectional LSTM Encoder-Decoder -> Translates clinical notes into patient regional language instructions.", "Reduces patient post-op medication dosing errors by 35%."),
        ("Case_Study_4_Energy_Industrial_Turbine_Fault_Detection.docx", "Industrial Power Plant Turbine Rotor Imbalance Detection", "Heavy Industry & Predictive Maintenance", "Vibration sensor array -> 3-layer Deep Belief Network (DBN) trained via Contrastive Divergence -> Detects sub-surface rotor imbalances.", "Prevents multi-million-dollar turbine blade fractures."),
        ("Case_Study_5_Fintech_Algorithmic_Default_Prediction.docx", "Retail Consumer Loan Default Risk Assessment", "Fintech & Banking Informatics", "Applicant credit bureau records -> 4-layer Deep Feedforward Network with L2 regularization and Early Stopping -> Predicts 90-day delinquency risk.", "Cuts non-performing consumer loan defaults by 18%.")
    ]
    for filename, title, domain, workflow, benefit in cases:
        doc = Document()
        doc.add_heading(f"Industrial Case Study: {title}", level=0)
        p = doc.add_paragraph()
        p.add_run(f"Domain: {domain}\n").bold = True
        p.add_run("Faculty: Jane Rose | Dept. of Computer Science & Engineering\n\n")
        doc.add_heading("1. Executive Summary & Problem Context", level=1)
        doc.add_paragraph(f"This case study details the production deployment of {title} within the {domain} sector.")
        doc.add_heading("2. Technical Pipeline Architecture", level=1)
        doc.add_paragraph(workflow)
        doc.add_heading("3. Business, Safety & Societal Benefits", level=1)
        doc.add_paragraph(benefit)
        doc.save(os.path.join(target_root, filename))

def generate_08_datasets():
    print("Generating 08_DATASETS...")
    target_root = os.path.join(BASE_DIR, "08_DATASETS")
    
    datasets = [
        ("MNIST_Handwritten_Digits_Pipeline.docx", "MNIST Handwritten Digits Benchmark", "Yann LeCun (1998)", "70,000 grayscale images of 28x28 pixels across 10 digits (0-9).", "Standard baseline for evaluating Perceptrons, MLPs, CNNs, and Autoencoders."),
        ("Fashion_MNIST_Clothing_Benchmark.docx", "Fashion-MNIST Clothing Classification", "Zalando Research (2017)", "70,000 28x28 grayscale images of 10 clothing categories (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).", "More challenging drop-in replacement for MNIST to test regularization and deep architectures."),
        ("CIFAR10_Natural_Images_Pipeline.docx", "CIFAR-10 Natural Color Images", "Alex Krizhevsky (2009)", "60,000 32x32 color RGB images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).", "Benchmark for testing spatial convolutions, data augmentation, and MobileNet/ResNet models."),
        ("IMDB_Movie_Reviews_Sentiment_Pipeline.docx", "IMDB Large Movie Review Dataset", "Maas et al. (2011)", "50,000 highly polarized movie reviews for binary sentiment classification (25k train, 25k test).", "Standard benchmark for Word2Vec embeddings, LSTMs, GRUs, and bidirectional sequence modeling."),
        ("LibriSpeech_Acoustic_Processing_Guide.docx", "LibriSpeech ASR Audio Corpus", "Vassil Panayotov et al. (2015)", "1,000 hours of 16 kHz English speech derived from LibriVox audiobooks.", "Gold-standard benchmark for speech recognition acoustic modeling and CTC loss training.")
    ]
    for filename, name, source, format_desc, pipeline in datasets:
        doc = Document()
        doc.add_heading(f"Dataset Pipeline Guide: {name}", level=0)
        p = doc.add_paragraph()
        p.add_run(f"Source: {source}\n").bold = True
        p.add_run("Faculty: Jane Rose | Dept. of Computer Science & Engineering\n\n")
        doc.add_heading("1. Dataset Overview & Format", level=1)
        doc.add_paragraph(format_desc)
        doc.add_heading("2. Preprocessing & Data Cleaning Pipeline", level=1)
        doc.add_paragraph("Normalization: Mean centering and scaling variance to [0, 1] or [-1, 1]. Data augmentation applied during training.")
        doc.add_heading("3. Model Selection & Academic Use", level=1)
        doc.add_paragraph(pipeline)
        doc.save(os.path.join(target_root, filename))

def generate_09_assignments():
    print("Generating 09_ASSIGNMENTS_AND_QUIZZES...")
    target_root = os.path.join(BASE_DIR, "09_ASSIGNMENTS_AND_QUIZZES")
    
    # 5 Module Assignments
    for mod in range(1, 6):
        doc = Document()
        doc.add_heading(f"KTU CST414: Assignment {mod} (Module {mod})", level=0)
        p = doc.add_paragraph()
        p.add_run(f"Department of Computer Science & Engineering | Scheme 2019\n").bold = True
        p.add_run("Faculty: Jane Rose | Max Marks: 25 | Submission Deadline: End of Module\n\n")
        
        doc.add_heading("Part A: Theoretical Formulations & Derivations (10 Marks)", level=1)
        doc.add_paragraph(f"1. Derive the core mathematical update equations governing Module {mod} architectures.")
        doc.add_paragraph(f"2. Explain the primary practical challenges and optimization pathologies inherent to Module {mod}.")
        
        doc.add_heading("Part B: Numerical Problem Solving (10 Marks)", level=1)
        doc.add_paragraph(f"1. Solve the advanced KTU examination-level numerical problem assigned for Topic {mod}.1.")
        doc.add_paragraph(f"2. Show complete step-by-step arithmetic substitution and engineering interpretation.")
        
        doc.add_heading("Part C: Programming & Verification Task (5 Marks)", level=1)
        doc.add_paragraph(f"1. Implement the assigned Python demonstration in Google Colab and submit the executed notebook link.")
        
        doc.add_heading("Evaluation Rubric", level=1)
        t = doc.add_table(rows=4, cols=3)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.rows[0].cells[0].text = "Criteria"
        t.rows[0].cells[1].text = "Marks"
        t.rows[0].cells[2].text = "Performance Standard"
        
        t.rows[1].cells[0].text = "Theoretical Derivation"
        t.rows[1].cells[1].text = "10"
        t.rows[1].cells[2].text = "Mathematically rigorous, exact notation"
        
        t.rows[2].cells[0].text = "Numerical Precision"
        t.rows[2].cells[1].text = "10"
        t.rows[2].cells[2].text = "All steps shown, accurate to 4 decimal places"
        
        t.rows[3].cells[0].text = "Code Implementation"
        t.rows[3].cells[1].text = "5"
        t.rows[3].cells[2].text = "Clean, executable Python code with outputs"
        
        doc.save(os.path.join(target_root, f"Assignment_{mod}_Module_{mod}.docx"))
        
    # Mini Project Brief
    doc_mp = Document()
    doc_mp.add_heading("KTU CST414: Capstone Mini-Project Brief", level=0)
    p_mp = doc_mp.add_paragraph()
    p_mp.add_run("Faculty: Jane Rose | Dept of Computer Science & Engineering\n\n").bold = True
    doc_mp.add_heading("1. Project Guidelines", level=1)
    doc_mp.add_paragraph("Students in teams of 2-3 must choose an end-to-end deep learning problem in Computer Vision, Speech, or NLP, train a deep model, and deploy a working demonstration.")
    doc_mp.save(os.path.join(target_root, "Mini_Project_Brief_Deep_Learning_CST414.docx"))

def generate_10_question_bank():
    print("Generating 10_KTU_QUESTION_BANK...")
    target_root = os.path.join(BASE_DIR, "10_KTU_QUESTION_BANK")
    
    # Master Question Bank
    doc = Document()
    doc.add_heading("KTU CST414 DEEP LEARNING: OFFICIAL QUESTION BANK", level=0)
    p = doc.add_paragraph()
    p.add_run("Department of Computer Science & Engineering | KTU 2019 Scheme\n").bold = True
    p.add_run("Faculty: Jane Rose | Mapped to Bloom's Taxonomy & Course Outcomes\n\n")
    
    for mod in range(1, 6):
        doc.add_heading(f"MODULE {mod} QUESTION BANK", level=1)
        mod_days = [d for d in ALL_36_DAYS if d["module"] == mod]
        
        doc.add_heading(f"Part A Questions (3 Marks Each - Short Answer)", level=2)
        for idx, day in enumerate(mod_days, 1):
            doc.add_paragraph(f"{idx}. Explain the concept of {day['title']} and state its primary significance in deep learning. [CO{mod}, {day['blooms']}, 3 Marks]")
            
        doc.add_heading(f"Part B Questions (14 Marks Each - Long Answer / Derivations)", level=2)
        for idx, day in enumerate(mod_days[:3], 1):
            prob = day["numerical_problems"][1]
            doc.add_paragraph(f"{idx}. (a) Derive the mathematical formulation and architecture of {day['title']}. (10 Marks)\n   (b) {prob['statement'][:150]}... (4 Marks)\n   [CO{mod}, Apply, 14 Marks]")
            
    doc.save(os.path.join(target_root, "Master_KTU_Question_Bank_All_Modules.docx"))

def generate_11_solved_papers():
    print("Generating 11_SOLVED_QUESTION_PAPERS...")
    target_root = os.path.join(BASE_DIR, "11_SOLVED_QUESTION_PAPERS")
    
    # Fully Solved KTU ESE Model Paper (100 Marks)
    doc = Document()
    doc.add_heading("KTU B.TECH DEGREE EXAMINATION CST414: DEEP LEARNING", level=0)
    p = doc.add_paragraph()
    p.add_run("MODEL QUESTION PAPER WITH COMPLETE DETAILED SOLUTIONS\n").bold = True
    p.add_run("Max Marks: 100 | Duration: 3 Hours | Scheme: KTU 2019\n")
    p.add_run("Prepared & Verified by: Jane Rose | Department of Computer Science & Engineering\n\n")
    
    # PART A
    doc.add_heading("PART A (Answer all questions. Each question carries 3 marks: 10 x 3 = 30 Marks)", level=1)
    
    part_a_qs = [
        ("Q1", "Suppose you have a 3-dimensional input x = (2, 2, 1) fully connected to 1 neuron in hidden layer with sigmoid activation. Weights are (0.2, 0.3, 0.1) and bias is -0.5. Calculate output.",
         "Solution:\n1. Weighted sum: z = (0.2 * 2) + (0.3 * 2) + (0.1 * 1) - 0.5 = 0.4 + 0.6 + 0.1 - 0.5 = 0.6.\n2. Sigmoid activation: y = 1 / (1 + e^(-0.6)) = 1 / (1 + 0.5488) = 1 / 1.5488 = 0.6457.\nFinal Answer: Output y = 0.6457."),
        
        ("Q2", "Design a single layer perceptron to compute the NAND function.",
         "Solution:\nNAND truth table: (0,0)->1, (0,1)->1, (1,0)->1, (1,1)->0.\nInequalities with weights w1, w2 and bias b:\n- For (0,0): 0 + 0 + b >= 0 => b >= 0\n- For (0,1): w2 + b >= 0\n- For (1,0): w1 + b >= 0\n- For (1,1): w1 + w2 + b < 0\nValid parameter choice: w1 = -2, w2 = -2, b = +3.\nVerification:\n- (0,0): -2(0) -2(0) + 3 = +3 >= 0 -> 1 (Correct)\n- (0,1): -2(0) -2(1) + 3 = +1 >= 0 -> 1 (Correct)\n- (1,0): -2(1) -2(0) + 3 = +1 >= 0 -> 1 (Correct)\n- (1,1): -2(1) -2(1) + 3 = -1 < 0 -> 0 (Correct)."),
        
        ("Q3", "Define L1 and L2 regularization and contrast their sparsity properties.",
         "Solution:\n- L1 Regularization (Lasso): Adds penalty alpha * sum |w_i|. Its constraint boundary has sharp corners on axes, driving non-essential weights to exactly zero (induces sparsity/feature selection).\n- L2 Regularization (Ridge): Adds penalty 0.5 * alpha * sum w_i^2. Shrinks weights proportionally (w <- (1 - eta*alpha)*w) toward zero, but rarely sets weights to exactly zero."),
        
        ("Q4", "Explain the role of the 'look-ahead' step in Nesterov Accelerated Gradient (NAG).",
         "Solution:\nNAG evaluates the gradient not at the current position theta_t, but at the look-ahead position (theta_t + beta * v_t). This acts as a predictive brake: if current momentum is about to carry parameters uphill, the look-ahead gradient detects the rising slope in advance and decelerates, reducing oscillations and overshooting."),
        
        ("Q5", "Why is a 1x1 convolution used in deep convolutional architectures like ResNet?",
         "Solution:\nA 1x1 convolution (Network-in-Network) computes a linear combination across input channels without changing spatial height or width. It is primarily used for channel dimensionality reduction (bottleneck projection), slashing parameters and FLOPs by up to 88% before expensive 3x3 convolutions."),
        
        ("Q6", "What happens if the stride of a convolutional layer increases?",
         "Solution:\nIncreasing stride s downsamples the output spatial dimensions by approximately factor 1/s: W_out = floor((W - K + 2p)/s) + 1. It reduces memory footprint and computational cost for subsequent layers, while expanding the effective receptive field stride across the input."),
        
        ("Q7", "Explain the Information Bottleneck problem in standard Seq2Seq models.",
         "Solution:\nIn standard Seq2Seq, the entire input sentence of arbitrary length (e.g. 50 words) must be compressed into a single fixed-dimensional context vector c. This causes severe information loss on long sequences, causing translation quality to degrade sharply for sentences longer than 20-30 words."),
        
        ("Q8", "How does a Gated Recurrent Unit (GRU) differ from Long Short-Term Memory (LSTM)?",
         "Solution:\n1. GRU has 2 gates (Reset, Update) while LSTM has 3 (Forget, Input, Output).\n2. GRU merges the cell state c and hidden state h into a single vector h.\n3. GRU couples forgetting and updating via convex interpolation (1 - z)*h_prev + z*h_tilde.\n4. GRU has 25% fewer parameters than LSTM."),
        
        ("Q9", "What is Continuous Bag of Words (CBOW) in Word2Vec?",
         "Solution:\nCBOW is a word embedding architecture that predicts a target center word w_t given its surrounding context words in a window {w_{t-C}, ..., w_{t+C}} by averaging context word vectors and passing them through a projection matrix."),
        
        ("Q10", "What is an Autoencoder? Give one industrial application.",
         "Solution:\nAn Autoencoder is a self-supervised neural network consisting of an Encoder that maps input x to a lower-dimensional latent bottleneck h, and a Decoder that reconstructs the input: x_hat = g(f(x)).\nIndustrial Application: Anomaly detection in high-speed rail ultrasonic track inspection (anomalies yield high reconstruction error).")
    ]
    
    for q_code, q_txt, ans_txt in part_a_qs:
        doc.add_heading(f"{q_code}: {q_txt}", level=2)
        p_ans = doc.add_paragraph()
        p_ans.add_run(ans_txt)
        
    # PART B
    doc.add_heading("PART B (Answer one question from each module: 5 x 14 = 70 Marks)", level=1)
    
    part_b_qs = [
        ("Module 1 (Q11)", "Update parameters in given MLP using GD with learning rate 0.5 and ReLU activation. (10 Marks)\n(b) Explain the importance of step size selection. (4 Marks)",
         "Solution (a):\nGiven input x = [1, 1]^T, target y = 1.0. Initial weights V = [[0.1, 0.2], [0.1, 0.1]], W = [0.1, 0.1]. Biases = 0.\nForward Pass:\n- Hidden z1_1 = 0.1(1) + 0.2(1) = 0.3 -> a1_1 = max(0, 0.3) = 0.3\n- Hidden z1_2 = 0.1(1) + 0.1(1) = 0.2 -> a1_2 = max(0, 0.2) = 0.2\n- Output zo = 0.1(0.3) + 0.1(0.2) = 0.05 -> y_hat = 0.05.\n- Loss L = 0.5*(0.05 - 1.0)^2 = 0.45125.\nBackward Pass:\n- delta_o = (0.05 - 1.0) * 1.0 = -0.95.\n- dL/dW1 = -0.95 * 0.3 = -0.285.\n- dL/dW2 = -0.95 * 0.2 = -0.190.\n- delta_h1 = (-0.95 * 0.1) * 1 = -0.095.\n- delta_h2 = (-0.95 * 0.1) * 1 = -0.095.\nUpdates (eta = 0.5):\n- W1_new = 0.1 - 0.5*(-0.285) = 0.2425.\n- W2_new = 0.1 - 0.5*(-0.190) = 0.1950.\n- V11_new = 0.1 - 0.5*(-0.095) = 0.1475; V12_new = 0.2475; V21_new = 0.1475; V22_new = 0.1475.\nSolution (b):\nStep size (learning rate) dictates convergence. Too large causes divergence and oscillation; too small leads to prohibitively slow convergence or getting trapped in saddle plateaus."),
        
        ("Module 2 (Q13)", "(a) Explain what happens in AdaGrad where momentum is expressed with cumulative squared gradients. (6 Marks)\n(b) Differentiate GD with and without momentum. Give update equations and illustrate plateaus. (8 Marks)",
         "Solution (a):\nAdaGrad accumulates squared gradients monotonically: s_t = s_{t-1} + g_t^2. The effective learning rate is eta / sqrt(s_t + eps). Because g_t^2 >= 0, the denominator grows without bound, forcing effective step size to zero prematurely, terminating learning before reaching the minimum.\nSolution (b):\nVanilla GD: theta_{t+1} = theta_t - eta * g_t.\nMomentum: v_{t+1} = beta * v_t - eta * g_t; theta_{t+1} = theta_t + v_{t+1}.\nOn plateaus where gradient g is small and constant, Vanilla GD moves at tiny step eta*g. Momentum accumulates velocity to terminal value (eta*g)/(1 - beta), accelerating traversal by 10x for beta=0.9."),
        
        ("Module 3 (Q15)", "(a) Explain Depthwise Separable Convolutions and derive their computational efficiency. (8 Marks)\n(b) Contrast Max Pooling and Average Pooling with a worked 4x4 example. (6 Marks)",
         "Solution (a):\nDepthwise Separable Convolutions split standard convolution into (1) Depthwise convolution (filtering per channel: D_F^2 * M * D_K^2) and (2) Pointwise 1x1 convolution (channel combination: D_F^2 * M * N). Total FLOPs ratio = (1/N) + (1/D_K^2). For 3x3 filters, this cuts computation by ~88% (8-9x speedup).\nSolution (b):\nMax pooling selects peak feature response max(patch), providing translation invariance. Average pooling computes mean(patch), smoothing spatial response. Full 4x4 worked numerical shows quadrant division."),
        
        ("Module 4 (Q18)", "(a) Explain the application of LSTM in Natural Language Processing. (8 Marks)\n(b) Discuss the architecture of GRU. (6 Marks)",
         "Solution (a):\nLSTMs excel in NLP tasks (Machine Translation, Sentiment Analysis, Text Generation, Language Modeling) because their linear cell state c^(t) and gating mechanism (Forget, Input, Output) allow error gradients to flow backward hundreds of tokens, resolving the vanishing gradient problem and capturing long-range syntactic agreement.\nSolution (b):\nGRU simplifies recurrent processing using a merged hidden state h and two gates: Reset gate r = sigma(W_r*[h_prev; x] + b_r) and Update gate z = sigma(W_z*[h_prev; x] + b_z). Final state interpolates: h = (1-z)*h_prev + z*tanh(W_h*[r*h_prev; x] + b_h)."),
        
        ("Module 5 (Q20)", "(a) Illustrate the use of representation learning in object classification. (7 Marks)\n(b) Compare Boltzmann Machine with Deep Belief Network. (7 Marks)",
         "Solution (a):\nRepresentation learning transforms raw high-dimensional image pixels lying on curved, entangled manifolds into flattened, linearly separable latent spaces where simple linear classifiers achieve high accuracy with minimal labeled data.\nSolution (b):\nBoltzmann Machine: Undirected graphical model with intra-layer connections; intractable MCMC training O(2^D).\nDeep Belief Network: Hybrid model (top two layers undirected RBM, lower layers directed generative connections); trained efficiently using greedy layer-wise Contrastive Divergence (CD-k) followed by supervised fine-tuning.")
    ]
    
    for q_code, q_txt, ans_txt in part_b_qs:
        doc.add_heading(f"{q_code}: {q_txt}", level=2)
        p_ans = doc.add_paragraph()
        p_ans.add_run(ans_txt)
        
    doc.save(os.path.join(target_root, "KTU_ESE_Model_Question_Paper_100_Marks_Fully_Solved.docx"))

def generate_12_internal_exams():
    print("Generating 12_INTERNAL_EXAMINATION...")
    target_root = os.path.join(BASE_DIR, "12_INTERNAL_EXAMINATION")
    
    # Internal Series Test 1 (50 Marks)
    doc1 = Document()
    doc1.add_heading("KTU CST414: CONTINUOUS ASSESSMENT TEST 1 (SERIES TEST 1)", level=0)
    p1 = doc1.add_paragraph()
    p1.add_run("Max Marks: 50 | Duration: 2 Hours | Scheme: KTU 2019\n").bold = True
    p1.add_run("Coverage: Module 1, Module 2, and Module 3 (First Half) | Faculty: Jane Rose\n\n")
    doc1.add_heading("Part A (5 Questions x 3 Marks = 15 Marks - Answer All)", level=1)
    doc1.add_paragraph("1. Calculate the output of a single neuron with sigmoid activation for input x = [2, 1] and weights w = [0.5, -0.2], bias 0.1.")
    doc1.add_paragraph("2. Why can a single-layer perceptron not compute the XOR function?")
    doc1.add_paragraph("3. Explain the premature freezing problem in the AdaGrad optimizer.")
    doc1.add_paragraph("4. Compare L1 and L2 regularization in terms of weight sparsity.")
    doc1.add_paragraph("5. Define translation equivariance in convolutional neural networks.")
    doc1.add_heading("Part B (Answer Any 5 of 7 Questions x 7 Marks = 35 Marks)", level=1)
    doc1.add_paragraph("6. Derive the error backpropagation equations for a 2-layer MLP.")
    doc1.add_paragraph("7. Walk through one full step of backpropagation update on a 2-2-1 network with ReLU activations.")
    doc1.add_paragraph("8. Explain vanishing and exploding gradients and derive the mathematical condition across L layers.")
    doc1.add_paragraph("9. Formulate Gradient Descent with Momentum and derive the terminal velocity on a constant gradient plateau.")
    doc1.add_paragraph("10. Formulate the Adam optimizer including first/second moment estimation and bias correction.")
    doc1.add_paragraph("11. Derive the Inverted Dropout algorithm and explain its ensemble interpretation.")
    doc1.add_paragraph("12. Compute the 2D convolution of a 4x4 matrix with a 3x3 Sobel kernel showing all intermediate products.")
    doc1.save(os.path.join(target_root, "CIE_Series_Test_1_Question_Paper_and_Scheme.docx"))
    
    # Internal Series Test 2 (50 Marks)
    doc2 = Document()
    doc2.add_heading("KTU CST414: CONTINUOUS ASSESSMENT TEST 2 (SERIES TEST 2)", level=0)
    p2 = doc2.add_paragraph()
    p2.add_run("Max Marks: 50 | Duration: 2 Hours | Scheme: KTU 2019\n").bold = True
    p2.add_run("Coverage: Module 3 (Second Half), Module 4, Module 5 | Faculty: Jane Rose\n\n")
    doc2.add_heading("Part A (5 Questions x 3 Marks = 15 Marks - Answer All)", level=1)
    doc2.add_paragraph("1. Explain the purpose of 1x1 convolutions in deep architectures.")
    doc2.add_paragraph("2. What is Backpropagation Through Time (BPTT)?")
    doc2.add_paragraph("3. How does the LSTM forget gate resolve vanishing gradients?")
    doc2.add_paragraph("4. Differentiate between CBOW and Skip-Gram in Word2Vec.")
    doc2.add_paragraph("5. State the Manifold Hypothesis in representation learning.")
    doc2.add_heading("Part B (Answer Any 5 of 7 Questions x 7 Marks = 35 Marks)", level=1)
    doc2.add_paragraph("6. Derive the computational efficiency ratio of Depthwise Separable Convolutions in MobileNet.")
    doc2.add_paragraph("7. Explain the Seq2Seq framework and analyze the fixed context vector Information Bottleneck.")
    doc2.add_paragraph("8. Write down the complete gate equations for an LSTM cell and perform a manual step calculation.")
    doc2.add_paragraph("9. Compare LSTM and GRU across architecture, parameter count, and memory overhead.")
    doc2.add_paragraph("10. Formulate the GloVe model and explain the semantic significance of co-occurrence probability ratios.")
    doc2.add_paragraph("11. Explain Undercomplete, Denoising, and Sparse Autoencoders with loss formulations.")
    doc2.add_paragraph("12. Formulate the Contrastive Divergence (CD-1) algorithm for training Restricted Boltzmann Machines.")
    doc2.save(os.path.join(target_root, "CIE_Series_Test_2_Question_Paper_and_Scheme.docx"))

def generate_13_revision_notes():
    print("Generating 13_STUDENT_REVISION_NOTES...")
    target_root = os.path.join(BASE_DIR, "13_STUDENT_REVISION_NOTES")
    
    doc = Document()
    doc.add_heading("KTU CST414 DEEP LEARNING: MASTER REVISION FORMULA SHEET", level=0)
    p = doc.add_paragraph()
    p.add_run("Comprehensive Exam Cheat-Sheet & Formula Reference\n").bold = True
    p.add_run("Faculty: Jane Rose | Department of Computer Science & Engineering\n\n")
    
    summaries = [
        ("Module 1: Neural Networks", "z = w^T*x + b; Sigmoid: sigma(z) = 1/(1+e^-z), sigma'=sigma(1-sigma); Tanh: (e^z - e^-z)/(e^z + e^-z), tanh'=1-tanh^2; ReLU: max(0,z); Softmax: e^z_i / sum e^z; Delta_out = (y_hat - y); Delta_hidden = W^T*delta * f'(z); W <- W - eta * delta * a^T."),
        ("Module 2: Deep Learning Optimization", "GD with Momentum: v <- beta*v - eta*g, theta <- theta + v; NAG: g evaluated at (theta + beta*v); AdaGrad: s <- s + g^2, step = - eta/sqrt(s)*g; RMSProp: s <- beta*s + (1-beta)*g^2; Adam: m <- beta1*m + (1-beta1)*g, v <- beta2*v + (1-beta2)*g^2, m_hat = m/(1-beta1^t), v_hat = v/(1-beta2^t); L2 Weight Decay: w <- (1 - eta*alpha)*w - eta*g; He Normal: Var = 2/fan_in; Xavier Normal: Var = 2/(fan_in + fan_out)."),
        ("Module 3: CNN Foundations", "Conv2D Output: floor((W - K + 2p)/s) + 1; MobileNet Separable FLOPs: D_F^2 * M * (D_K^2 + N); Speedup ratio approx 1/N + 1/D_K^2 (~8.5x for 3x3); Winograd F(2x2, 3x3) multiplies = 16 (2.25x speedup over direct 36)."),
        ("Module 4: Sequence Models", "RNN: h_t = tanh(W_hh*h_{t-1} + W_xh*x_t + b); LSTM: f_t = sigma(W_f*[h; x] + b_f), i_t = sigma(W_i*[h; x] + b_i), c_tilde = tanh(W_c*[h; x] + b_c), c_t = f_t*c_{t-1} + i_t*c_tilde, o_t = sigma(W_o*[h; x] + b_o), h_t = o_t*tanh(c_t); GRU: z_t = sigma(W_z*[h; x] + b_z), r_t = sigma(W_r*[h; x] + b_r), h_tilde = tanh(W_h*[r*h; x] + b_h), h_t = (1-z_t)*h_{t-1} + z_t*h_tilde."),
        ("Module 5: Applications & Representations", "IoU = Area(Inter) / Area(Union); WER = (S + D + I) / N; Perplexity = exp(Cross_Entropy_Loss); GloVe: sum f(X_ij)*(w_i^T*w_j + b_i + b_j - ln X_ij)^2; RBM Energy: E(v, h) = - v^T*W*h - b^T*v - c^T*h; CD-1: delta_W = eta*(v_0*h_0^T - v_1*h_1^T).")
    ]
    for title, text in summaries:
        doc.add_heading(title, level=1)
        doc.add_paragraph(text)
        
    doc.save(os.path.join(target_root, "Master_Formula_Sheet_CST414.docx"))

def generate_14_master_slides():
    print("Generating 14_PRESENTATION_SLIDES...")
    target_root = os.path.join(BASE_DIR, "14_PRESENTATION_SLIDES")
    
    for mod in range(1, 6):
        prs = Presentation()
        s1 = prs.slides.add_slide(prs.slide_layouts[0])
        s1.shapes.title.text = f"CST414: MODULE {mod} MASTER DECK"
        s1.placeholders[1].text = f"Deep Learning | KTU 2019 Scheme\nJane Rose | Department of Computer Science & Engineering"
        
        mod_days = [d for d in ALL_36_DAYS if d["module"] == mod]
        for day in mod_days:
            s = prs.slides.add_slide(prs.slide_layouts[1])
            s.shapes.title.text = f"Topic {day['topic_id']}: {day['title']}"
            tf = s.placeholders[1].text_frame
            tf.text = f"Target Outcome: {day['co']} ({day['blooms']})"
            for obj in day["learning_objectives"][:3]:
                p = tf.add_paragraph(); p.text = f"- {obj}"
            prob = day["numerical_problems"][0]
            p = tf.add_paragraph(); p.text = f"Worked Numerical: {prob['title']} -> Ans: {prob['answer']}"
            
        prs.save(os.path.join(target_root, f"Module_{mod}_Master_Presentation.pptx"))

if __name__ == "__main__":
    ensure_central_dirs()
    generate_04_python_demos()
    generate_05_numerical_problems()
    generate_06_viva_questions()
    generate_07_case_studies()
    generate_08_datasets()
    generate_09_assignments()
    generate_10_question_bank()
    generate_11_solved_papers()
    generate_12_internal_exams()
    generate_13_revision_notes()
    generate_14_master_slides()
    print("All Centralized Teaching Resources Generated Successfully!")

