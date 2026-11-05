"""
Export course curriculum and schedule as a comprehensive JSON for the React/TypeScript Faculty Dashboard
"""

import os
import json
from curriculum_all import PROVISIONAL_SCHEDULE, ALL_36_DAYS, PROVISIONAL_BLOCKED_DATES

target_path = os.path.abspath("DEEP_LEARNING_CST414/15_FACULTY_DASHBOARD/src/data/courseData.json")
os.makedirs(os.path.dirname(target_path), exist_ok=True)

# Build comprehensive payload
data_payload = {
    "course_info": {
        "code": "CST414",
        "name": "Deep Learning",
        "scheme": "KTU 2019",
        "department": "Computer Science & Engineering",
        "credits": 3,
        "total_hours": 36,
        "faculty": "Jane Rose",
        "faculty_title": "Assistant Professor & Course Coordinator",
        "semester": "S7 (B.Tech)",
        "commencement_date": "18-Aug-2025 (Provisional)",
        "completion_date": "26-Nov-2025 (Provisional)",
        "status": "PROVISIONAL - Pending Institutional Calendar & Timetable Upload"
    },
    "modules": [
        {"id": 1, "name": "Module 1 - Neural Networks", "hours": 7, "co": "CO1", "topics": "Single Layer Perceptron, MLPs, Activation Functions, Risk Minimization, Backpropagation, Practical Issues, Convergence Pathologies"},
        {"id": 2, "name": "Module 2 - Deep Learning", "hours": 9, "co": "CO2", "topics": "Deep Models, Gradient Descent, Momentum, NAG, SGD, AdaGrad, RMSProp, Adam, L1/L2, Early Stopping, Data Augmentation, Parameter Sharing & Tying, Ensembles, Dropout, Initialization"},
        {"id": 3, "name": "Module 3 - Convolutional Neural Networks", "hours": 6, "co": "CO3", "topics": "Convolution Operation, Pooling, Infinitely Strong Prior, Convolution Variants, Structured Outputs, Winograd & MobileNet Algorithms"},
        {"id": 4, "name": "Module 4 - Recurrent Neural Networks", "hours": 5, "co": "CO4", "topics": "Computational Graphs, Unrolling, BPTT, Seq2Seq, Deep Recurrent & Recursive Networks, LSTM, GRU"},
        {"id": 5, "name": "Module 5 - Application Areas", "hours": 9, "co": "CO5", "topics": "Computer Vision (YOLO/IoU), Speech Recognition (CTC/Mel), NLP (Perplexity), CBOW & Skip-Gram, GloVe, Autoencoders, Representation Learning, Boltzmann Machines, Deep Belief Networks"}
    ],
    "schedule": PROVISIONAL_SCHEDULE,
    "holidays": [
        {"date": dt.strftime("%d-%b-%Y"), "day": dt.strftime("%A"), "occasion": occ, "type": "Internal Exam" if "Internal" in occ else "Public Holiday"}
        for dt, occ in PROVISIONAL_BLOCKED_DATES.items()
    ]
}

with open(target_path, "w", encoding="utf-8") as f:
    json.dump(data_payload, f, indent=2)

print(f"Exported course data to {target_path} (Size: {os.path.getsize(target_path):,} bytes)")

