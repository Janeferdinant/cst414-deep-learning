# CST414 Deep Learning Course System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete, production-grade faculty teaching resource system and interactive educational web dashboard for KTU CST414 Deep Learning (36 lecture hours, 5 modules, Scheme 2019) for Faculty Jane Rose, Department of CSE.

**Architecture:** 
1. Python automation engine generating 36 comprehensive daily lecture packages (Word .docx, PowerPoint .pptx, Jupyter .ipynb, and Markdown .md) across official KTU syllabus modules, plus question banks, solved exam papers, and assessments.
2. Next.js / React / TypeScript / Tailwind CSS interactive Faculty Dashboard featuring Three.js 3D neural architectures, Framer Motion animations, Recharts analytics, Burgundy/Champagne-Gold/Navy aesthetic, live schedule editor, progress tracking, and resource downloads.

**Tech Stack:** Python 3.14 (python-docx, python-pptx, openpyxl, pandas, numpy, matplotlib, pypdf), Next.js 14/15, React 18/19, TypeScript, Tailwind CSS, Lucide React, Framer Motion, Recharts, Three.js.

**Spec:** USER_REQUEST specification for KTU CST414 Deep Learning course planning, syllabus mapping, 36-day resources, and interactive dashboard.

## Global Constraints

- Course: KTU CST414 Deep Learning (Scheme 2019, 3 Credits, 36 Lecture Hours).
- Faculty Name: Jane Rose | Department of Computer Science & Engineering (present in all footers and headers).
- Missing Documents: Official academic calendar is missing in the workspace. Create a clearly labeled PROVISIONAL schedule (3 periods/week across standard semester calendar) and explicitly request the institutional calendar.
- Teaching Plan Preservation: Preserve exact official topics and hours from `DEEP LEARNING.pdf`. Explicitly flag the duplicated topic description in Module 2 (2.7 & 2.8: "Parameter sharing, tying, input noise and ensemble methods") for faculty verification.
- Completeness: All 36 daily folders must each contain all 9 required files: `Daily_Lesson_Notes.docx`, `Daily_Lecture_Slides.pptx`, `Teaching_Script.docx`, `Python_Demo.ipynb`, `Numerical_Problems_With_Solutions.docx`, `Real_World_Application.docx`, `Viva_Questions_With_Answers.docx`, `Class_Quiz.docx`, and `Reference_Materials.md`.

## Review Focus

1. Numerical problem accuracy: Every numerical step must have exact mathematical derivations, values substituted, and correct calculations verified.
2. Syllabus fidelity: Ensure all topics from Module 1 to 5 match KTU 2019 CST414 exactly.
3. Executable Python demos: Jupyter notebooks must contain clean, working standard Python/NumPy code that runs without external heavy GPU dependencies.
4. UI / Dashboard responsiveness: Dashboard must render flawlessly with burgundy/gold/navy theme, glassmorphism, 3D Canvas visualizers, and interactive schedule toggles.
5. Mandatory footer: Every dashboard view must carry "Jane Rose | Department of Computer Science & Engineering".

---

### Task 1: Document Inspection, Syllabus Extraction & Provisional Calendar Setup

**Files:**
- Create: `00_COURSE_OVERVIEW/Course_Outcomes.docx`
- Create: `00_COURSE_OVERVIEW/Academic_Calendar_Analysis.xlsx`
- Create: `01_ACADEMIC_CALENDAR/Academic_Calendar.xlsx`
- Create: `01_ACADEMIC_CALENDAR/Working_Days.xlsx`
- Create: `01_ACADEMIC_CALENDAR/Holiday_List.xlsx`
- Create: `01_ACADEMIC_CALENDAR/Semester_Teaching_Schedule.xlsx`
- Test: Verification script checking file creation and integrity.

- [ ] **Step 1: Write test script to verify calendar and syllabus data structures**
- [ ] **Step 2: Generate provisional semester calendar and 36-period schedule**
- [ ] **Step 3: Generate course overview and outcome documents**
- [ ] **Step 4: Verify outputs and date mapping**

---

### Task 2: Official Subject Plan & CO-PO Mapping Generation

**Files:**
- Create: `02_SUBJECT_PLAN/Official_Subject_Plan.docx`
- Create: `02_SUBJECT_PLAN/Date_Wise_Teaching_Plan.xlsx`
- Create: `02_SUBJECT_PLAN/Module_Wise_Plan.xlsx`
- Create: `02_SUBJECT_PLAN/CO_Mapping.xlsx`

- [ ] **Step 1: Parse 36 lecture topics from DEEP LEARNING.pdf**
- [ ] **Step 2: Map each topic to Bloom's Taxonomy, COs (CO1-CO5), POs (PO1-PO12)**
- [ ] **Step 3: Highlight Module 2 topics 2.7 & 2.8 duplication flag**
- [ ] **Step 4: Build Excel sheets and Word documents for approved subject plan**
- [ ] **Step 5: Verify all 36 hours mapped without discrepancy**

---

### Task 3: Comprehensive Daily Class Content Generator (All 36 Days)

**Files:**
- Create: `scripts/generate_daily_resources.py`
- Create: `03_DAILY_CLASS_CONTENT/MODULE_1_NEURAL_NETWORKS/DAY_01/` to `DAY_07/`
- Create: `03_DAILY_CLASS_CONTENT/MODULE_2_DEEP_LEARNING/DAY_08/` to `DAY_16/`
- Create: `03_DAILY_CLASS_CONTENT/MODULE_3_CNN/DAY_17/` to `DAY_22/`
- Create: `03_DAILY_CLASS_CONTENT/MODULE_4_RNN/DAY_23/` to `DAY_27/`
- Create: `03_DAILY_CLASS_CONTENT/MODULE_5_APPLICATIONS/DAY_28/` to `DAY_36/`
- Each DAY folder contains:
  1. `Daily_Lesson_Notes.docx` (Complete KTU notes, formulas, architecture, derivations)
  2. `Daily_Lecture_Slides.pptx` (60-minute presentation slides)
  3. `Teaching_Script.docx` (0-60 min faculty script with questions)
  4. `Python_Demo.ipynb` (Executable code notebook with numpy/matplotlib)
  5. `Numerical_Problems_With_Solutions.docx` (Basic + Advanced KTU numericals)
  6. `Real_World_Application.docx` (Industry workflow, input/output, impact)
  7. `Viva_Questions_With_Answers.docx` (10 viva questions & answers)
  8. `Class_Quiz.docx` (3 MCQs, 2 short-answer, 1 application, 1 homework)
  9. `Reference_Materials.md` (Textbook citations, readings, links)

- [ ] **Step 1: Build master data dictionary for all 36 topics with complete technical depth**
- [ ] **Step 2: Implement Python document generators for DOCX, PPTX, IPYNB, and MD**
- [ ] **Step 3: Run batch generation across all 36 days and all 5 modules**
- [ ] **Step 4: Verify all 36 * 9 = 324 files exist and validate structure**

---

### Task 4: Centralized Repositories: Demonstrations, Numericals, Vivas, and Case Studies

**Files:**
- Populate: `04_PYTHON_DEMONSTRATIONS/` (Module 1 to 5 notebooks and standalone scripts)
- Populate: `05_NUMERICAL_PROBLEMS/` (Comprehensive numerical problem collections)
- Populate: `06_VIVA_QUESTIONS/` (Comprehensive viva banks per module)
- Populate: `07_REAL_WORLD_CASE_STUDIES/` (Detailed case studies: Vision, NLP, Speech, Healthcare)
- Populate: `08_DATASETS/` (Dataset guides for MNIST, CIFAR, IMDB, Fashion-MNIST)

- [ ] **Step 1: Assemble consolidated Python demo collection**
- [ ] **Step 2: Assemble consolidated Numerical bank with worked solutions**
- [ ] **Step 3: Assemble consolidated Viva question bank (360+ questions)**
- [ ] **Step 4: Create real-world case study documents and dataset pipeline documentation**

---

### Task 5: KTU Question Bank, Solved Exam Papers & Assessment System

**Files:**
- Populate: `09_ASSIGNMENTS_AND_QUIZZES/` (Assignments 1-5, Rubrics, Quizzes)
- Populate: `10_KTU_QUESTION_BANK/` (Module 1 to 5 KTU questions with marks and CO mapping)
- Populate: `11_SOLVED_QUESTION_PAPERS/` (Fully solved KTU 2019 Model ESE Paper, 100 marks)
- Populate: `12_INTERNAL_EXAMINATION/` (Series Test 1: 50 marks, Series Test 2: 50 marks with schemes)
- Populate: `13_STUDENT_REVISION_NOTES/` (Quick revision formula sheets and cheat sheets)
- Populate: `14_PRESENTATION_SLIDES/` (Consolidated master slide decks)

- [ ] **Step 1: Generate KTU CST414 Question Bank mapped to Bloom's levels**
- [ ] **Step 2: Generate Solved Internal Exam 1 (Modules 1, 2, half 3) & Internal Exam 2 (half 3, 4, 5)**
- [ ] **Step 3: Generate 100-mark Solved End Semester Examination Model Paper**
- [ ] **Step 4: Generate Assignments, Numerical worksheets, and evaluation rubrics**

---

### Task 6: Premium Interactive Faculty Web Dashboard (Next.js / React / Tailwind)

**Files:**
- Create: `15_FACULTY_DASHBOARD/` (Next.js project with Tailwind CSS, Framer Motion, Lucide, Recharts, Three.js)
- Components:
  - Header & Navigation with Burgundy/Gold styling
  - Academic Calendar & Provisional Timetable Viewer
  - 36-Hour Interactive Day-by-Day Explorer with instant view/download for all 9 assets
  - Interactive 3D Neural Network / CNN / RNN Visualizer (Three.js / HTML5 Canvas)
  - Interactive Gradient Descent Simulator (Contour plots, learning rate slider)
  - CO-PO Attainment & Assessment Analytics (Recharts)
  - Solved Question Paper & Question Bank Explorer
  - Schedule Editor (mark classes completed, reschedule, add remarks)
  - Mandatory Footer: "Jane Rose | Department of Computer Science & Engineering"

- [ ] **Step 1: Initialize dashboard application structure and dependencies**
- [ ] **Step 2: Integrate complete 36-day course data into client-side state**
- [ ] **Step 3: Implement Three.js 3D Neural Architecture and Interactive Visualizations**
- [ ] **Step 4: Implement Schedule management, filtering, and export capabilities**
- [ ] **Step 5: Build and verify the application runs smoothly**

---

### Task 7: Comprehensive Verification & Final Compliance Report

**Files:**
- Create: `COMPLIANCE_AND_VERIFICATION_REPORT.md`
- Verify: 36 teaching hours, 5 modules, all 324 daily teaching assets, numerical math, KTU answers.

- [ ] **Step 1: Run automated verification script checking all directories and files**
- [ ] **Step 2: Validate numerical answers and derivations**
- [ ] **Step 3: Generate final compliance report**

