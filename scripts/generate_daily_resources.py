"""
Generator for All 36 Daily Class Content Resource Packages
KTU CST414 Deep Learning (Scheme 2019)
Faculty: Jane Rose | Department of Computer Science & Engineering
"""

import os
import json
import datetime
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from pptx import Presentation
from pptx.util import Inches as PptxInches, Pt as PptxPt
from pptx.dml.color import RGBColor as PptxRGBColor

from curriculum_all import PROVISIONAL_SCHEDULE

BASE_DIR = os.path.abspath("DEEP_LEARNING_CST414")
DAILY_DIR = os.path.join(BASE_DIR, "03_DAILY_CLASS_CONTENT")

MODULE_FOLDER_MAP = {
    1: "MODULE_1_NEURAL_NETWORKS",
    2: "MODULE_2_DEEP_LEARNING",
    3: "MODULE_3_CNN",
    4: "MODULE_4_RNN",
    5: "MODULE_5_APPLICATIONS"
}

def create_ipynb(title, description, code_str):
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# KTU CST414 Deep Learning: Practical Demonstration\n",
                    f"## {title}\n",
                    f"**Faculty:** Jane Rose | Department of Computer Science & Engineering\n\n",
                    f"{description}\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [code_str]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Verification & Key Observations\n",
                    "1. The code demonstrates the fundamental mathematical operations directly in pure NumPy.\n",
                    "2. Verify intermediate array dimensions and numerical stability.\n",
                    "3. Experiment with hyperparameters (learning rate, sequence length, kernel size) to observe model behavior."
                ]
            }
        ],
        "metadata": {
            "language_info": {"name": "python", "version": "3.10"},
            "orig_nbformat": 4
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    return nb

def generate_daily_resources():
    print(f"Starting batch generation of 36 daily teaching packages...")
    
    for item in PROVISIONAL_SCHEDULE:
        day_num = item["day_number"]
        mod_num = item["module"]
        mod_dir_name = MODULE_FOLDER_MAP[mod_num]
        day_dir_name = f"DAY_{day_num:02d}"
        target_dir = os.path.join(DAILY_DIR, mod_dir_name, day_dir_name)
        os.makedirs(target_dir, exist_ok=True)
        
        data = item["data"]
        topic_id = data["topic_id"]
        topic_title = data["title"]
        co = data["co"]
        blooms = data["blooms"]
        date_str = item["date_display"]
        period_str = item["period"]
        
        # -------------------------------------------------------------
        # 1. Daily_Lesson_Notes.docx
        # -------------------------------------------------------------
        doc_notes = Document()
        doc_notes.add_heading(f"KTU CST414: {topic_title}", level=0)
        p_hdr = doc_notes.add_paragraph()
        p_hdr.add_run(f"Module {mod_num} | Topic {topic_id} | Planned Lecture Hour: 1 Hour (60 Mins)\n").bold = True
        p_hdr.add_run(f"Scheduled Date: {date_str} ({item['day_name']}) | {period_str}\n")
        p_hdr.add_run(f"Faculty: Jane Rose | Department of Computer Science & Engineering\n")
        p_hdr.add_run(f"Target Course Outcome: {co} | Bloom's Level: {blooms}\n")
        
        doc_notes.add_heading("1. Lecture Preparation & Metadata", level=1)
        p_prep = doc_notes.add_paragraph()
        p_prep.add_run(f"Prerequisites: {data['prerequisites']}\n")
        p_prep.add_run(f"Textbook Reference: {data['textbook_ref']}\n")
        p_prep.add_run("Key Terminology: " + ", ".join(data["key_terminology"]) + "\n\n")
        p_prep.add_run("Learning Objectives:\n").bold = True
        for obj in data["learning_objectives"]:
            p_prep.add_run(f"- {obj}\n")
            
        doc_notes.add_heading("2. Comprehensive KTU Lecture Notes", level=1)
        notes_dict = data["notes"]
        for section_title, content in notes_dict.items():
            if section_title == "exam_important_points":
                continue
            doc_notes.add_heading(section_title.replace("_", " ").title(), level=2)
            if isinstance(content, str):
                doc_notes.add_paragraph(content)
            elif isinstance(content, list):
                for elem in content:
                    doc_notes.add_paragraph(f"- {elem}")
                    
        doc_notes.add_heading("3. Examination-Important Points & Takeaways", level=1)
        p_exam = doc_notes.add_paragraph()
        for pt in notes_dict.get("exam_important_points", []):
            p_exam.add_run(f"- {pt}\n")
            
        doc_notes.save(os.path.join(target_dir, "Daily_Lesson_Notes.docx"))
        
        # -------------------------------------------------------------
        # 2. Daily_Lecture_Slides.pptx
        # -------------------------------------------------------------
        prs = Presentation()
        # Slide 1: Title
        s1 = prs.slides.add_slide(prs.slide_layouts[0])
        s1.shapes.title.text = f"CST414: {topic_title}"
        s1.placeholders[1].text = (
            f"Module {mod_num} (Topic {topic_id}) | 60-Minute Class\n"
            f"Jane Rose | Dept. of Computer Science & Engineering\n"
            f"{date_str} | {co} ({blooms})"
        )
        
        # Slide 2: Objectives & Prerequisites
        s2 = prs.slides.add_slide(prs.slide_layouts[1])
        s2.shapes.title.text = "Learning Objectives & Prerequisites"
        tf2 = s2.placeholders[1].text_frame
        tf2.text = f"Prerequisites: {data['prerequisites']}"
        for obj in data["learning_objectives"]:
            p = tf2.add_paragraph()
            p.text = obj
            
        # Slide 3: Theory & Conceptual Architecture
        s3 = prs.slides.add_slide(prs.slide_layouts[1])
        s3.shapes.title.text = "Key Theory & Conceptual Model"
        tf3 = s3.placeholders[1].text_frame
        intro_text = notes_dict.get("introduction", "Core theoretical foundations.")
        tf3.text = intro_text[:280] + "..." if len(intro_text) > 280 else intro_text
        for term in data["key_terminology"][:4]:
            p = tf3.add_paragraph()
            p.text = f"Concept: {term}"
            
        # Slide 4: Mathematical Derivation & Formulation
        s4 = prs.slides.add_slide(prs.slide_layouts[1])
        s4.shapes.title.text = "Mathematical Formulations & Equations"
        tf4 = s4.placeholders[1].text_frame
        tf4.text = f"Core Governing Mathematical Principles for {topic_title}:"
        prob_first = data["numerical_problems"][0]
        p = tf4.add_paragraph(); p.text = f"Key Formula: {prob_first['formulas'][:250]}"
        p = tf4.add_paragraph(); p.text = f"Interpretation: {prob_first['interpretation'][:250]}"
        
        # Slide 5: Numerical Illustration
        s5 = prs.slides.add_slide(prs.slide_layouts[1])
        s5.shapes.title.text = f"Worked Numerical Example ({prob_first['type']})"
        tf5 = s5.placeholders[1].text_frame
        tf5.text = f"Problem: {prob_first['statement'][:220]}..."
        p = tf5.add_paragraph(); p.text = f"Given: {prob_first['given'][:150]}"
        p = tf5.add_paragraph(); p.text = f"Final Answer: {prob_first['answer']}"
        
        # Slide 6: Real-World Application
        s6 = prs.slides.add_slide(prs.slide_layouts[1])
        app = data["real_world_application"]
        s6.shapes.title.text = f"Industry Application: {app['title']}"
        tf6 = s6.placeholders[1].text_frame
        tf6.text = f"Domain: {app['domain']}"
        p = tf6.add_paragraph(); p.text = f"Workflow: {app['workflow']}"
        p = tf6.add_paragraph(); p.text = f"Impact & Benefit: {app['benefit']}"
        
        # Slide 7: Viva & Key Questions
        s7 = prs.slides.add_slide(prs.slide_layouts[1])
        s7.shapes.title.text = "Viva Questions & Takeaways"
        tf7 = s7.placeholders[1].text_frame
        tf7.text = "Essential Viva Questions for Classroom Engagement:"
        for v in data["viva_questions"][:3]:
            p = tf7.add_paragraph()
            p.text = f"Q: {v['q']}"
            
        prs.save(os.path.join(target_dir, "Daily_Lecture_Slides.pptx"))
        
        # -------------------------------------------------------------
        # 3. Teaching_Script.docx
        # -------------------------------------------------------------
        doc_script = Document()
        doc_script.add_heading(f"Faculty Teaching Script: {topic_title}", level=0)
        p_sc = doc_script.add_paragraph()
        p_sc.add_run(f"Day {day_num:02d} | Module {mod_num} | Topic {topic_id} | 60-Minute Class Script\n").bold = True
        p_sc.add_run("Faculty: Jane Rose | Dept. of Computer Science & Engineering\n\n")
        
        doc_script.add_heading("Classroom Teaching Sequence & Timeline", level=1)
        t_script = doc_script.add_table(rows=1, cols=3)
        t_script.alignment = WD_TABLE_ALIGNMENT.CENTER
        hdr_s = t_script.rows[0].cells
        hdr_s[0].text = "Timeline"
        hdr_s[1].text = "Activity"
        hdr_s[2].text = "Detailed Faculty Talking Points & Instructions"
        
        script_dict = data["teaching_script"]
        activity_labels = {
            "00-05": "Recap & Motivation",
            "05-20": "Theoretical Formulation",
            "20-35": "Mathematical Derivation & Blackboard Work",
            "35-45": "Numerical Problem Walkthrough",
            "45-55": "Python Demo & Practical Case",
            "55-60": "Assessment, Viva & Wrap-up"
        }
        for time_slot, desc in script_dict.items():
            row = t_script.add_row().cells
            row[0].text = f"{time_slot} Mins"
            row[1].text = activity_labels.get(time_slot, "Instruction")
            row[2].text = desc
            
        doc_script.add_heading("Interactive Questions to Ask Students During Lecture", level=1)
        p_q = doc_script.add_paragraph()
        for idx_v, v in enumerate(data["viva_questions"][:5], 1):
            p_q.add_run(f"Question {idx_v}: {v['q']}\n").bold = True
            p_q.add_run(f"Expected Answer: {v['a']}\n\n")
            
        doc_script.save(os.path.join(target_dir, "Teaching_Script.docx"))
        
        # -------------------------------------------------------------
        # 4. Python_Demo.ipynb
        # -------------------------------------------------------------
        pdemo = data["python_demo"]
        ipynb_json = create_ipynb(pdemo["title"], pdemo["description"], pdemo["code"])
        with open(os.path.join(target_dir, "Python_Demo.ipynb"), "w", encoding="utf-8") as f:
            json.dump(ipynb_json, f, indent=2)
            
        # -------------------------------------------------------------
        # 5. Numerical_Problems_With_Solutions.docx
        # -------------------------------------------------------------
        doc_num = Document()
        doc_num.add_heading(f"Numerical Problems & Solutions: {topic_title}", level=0)
        p_n = doc_num.add_paragraph()
        p_n.add_run(f"Module {mod_num} | Topic {topic_id} | KTU CST414 Deep Learning\n").bold = True
        p_n.add_run("Faculty: Jane Rose | Dept. of Computer Science & Engineering\n\n")
        
        for p_idx, prob in enumerate(data["numerical_problems"], 1):
            doc_num.add_heading(f"Problem {p_idx}: {prob['title']} ({prob['type']})", level=1)
            p_prob = doc_num.add_paragraph()
            p_prob.add_run("Problem Statement:\n").bold = True
            p_prob.add_run(f"{prob['statement']}\n\n")
            p_prob.add_run("Given Values:\n").bold = True
            p_prob.add_run(f"{prob['given']}\n\n")
            p_prob.add_run("Governing Formulas:\n").bold = True
            p_prob.add_run(f"{prob['formulas']}\n\n")
            p_prob.add_run("Step-by-Step Calculation:\n").bold = True
            for st in prob["steps"]:
                p_prob.add_run(f"{st}\n")
            p_prob.add_run("\nFinal Answer:\n").bold = True
            p_prob.add_run(f"{prob['answer']}\n\n")
            p_prob.add_run("Engineering & Physical Interpretation:\n").bold = True
            p_prob.add_run(f"{prob['interpretation']}\n")
            
        doc_num.save(os.path.join(target_dir, "Numerical_Problems_With_Solutions.docx"))
        
        # -------------------------------------------------------------
        # 6. Real_World_Application.docx
        # -------------------------------------------------------------
        doc_app = Document()
        doc_app.add_heading(f"Real-World Case Study: {app['title']}", level=0)
        p_ap = doc_app.add_paragraph()
        p_ap.add_run(f"Module {mod_num} | Topic {topic_id} | Industrial Deployment Profile\n").bold = True
        p_ap.add_run("Faculty: Jane Rose | Dept. of Computer Science & Engineering\n\n")
        
        doc_app.add_heading("1. Industrial Application Domain", level=1)
        doc_app.add_paragraph(app["domain"])
        
        doc_app.add_heading("2. Technical Workflow Pipeline", level=1)
        doc_app.add_paragraph(app["workflow"])
        
        doc_app.add_heading("3. Business, Safety & Societal Benefits", level=1)
        doc_app.add_paragraph(app["benefit"])
        
        doc_app.save(os.path.join(target_dir, "Real_World_Application.docx"))
        
        # -------------------------------------------------------------
        # 7. Viva_Questions_With_Answers.docx
        # -------------------------------------------------------------
        doc_viva = Document()
        doc_viva.add_heading(f"Viva Questions and Model Answers: {topic_title}", level=0)
        p_v = doc_viva.add_paragraph()
        p_v.add_run(f"Module {mod_num} | Topic {topic_id} | Question Bank (10 Questions)\n").bold = True
        p_v.add_run("Faculty: Jane Rose | Dept. of Computer Science & Engineering\n\n")
        
        for v_idx, item_v in enumerate(data["viva_questions"], 1):
            doc_viva.add_heading(f"Q{v_idx}: {item_v['q']}", level=2)
            p_ans = doc_viva.add_paragraph()
            p_ans.add_run("Answer: ").bold = True
            p_ans.add_run(item_v["a"])
            
        doc_viva.save(os.path.join(target_dir, "Viva_Questions_With_Answers.docx"))
        
        # -------------------------------------------------------------
        # 8. Class_Quiz.docx
        # -------------------------------------------------------------
        doc_quiz = Document()
        doc_quiz.add_heading(f"Classroom Assessment & Quiz: {topic_title}", level=0)
        p_qz = doc_quiz.add_paragraph()
        p_qz.add_run(f"Module {mod_num} | Topic {topic_id} | Formative Assessment\n").bold = True
        p_qz.add_run("Faculty: Jane Rose | Dept. of Computer Science & Engineering\n\n")
        
        ass = data["assessment"]
        doc_quiz.add_heading("Section A: Multiple Choice Questions (MCQs)", level=1)
        for m_idx, mcq in enumerate(ass["mcqs"], 1):
            p_m = doc_quiz.add_paragraph()
            p_m.add_run(f"{m_idx}. {mcq['q']}\n").bold = True
            for opt_idx, opt in enumerate(mcq["options"], 1):
                p_m.add_run(f"   ({chr(64+opt_idx)}) {opt}\n")
            p_m.add_run(f"   [Answer Key: {mcq['ans']}]\n")
            
        doc_quiz.add_heading("Section B: Short-Answer Conceptual Questions", level=1)
        for s_idx, sq in enumerate(ass["short_questions"], 1):
            p_sq = doc_quiz.add_paragraph()
            p_sq.add_run(f"{s_idx}. {sq}\n").bold = True
            
        doc_quiz.add_heading("Section C: Application-Oriented Problem", level=1)
        p_appq = doc_quiz.add_paragraph()
        p_appq.add_run(ass["application_question"] + "\n")
        
        doc_quiz.add_heading("Section D: Homework / Practice Exercise", level=1)
        p_hw = doc_quiz.add_paragraph()
        p_hw.add_run(ass["homework"] + "\n")
        
        doc_quiz.save(os.path.join(target_dir, "Class_Quiz.docx"))
        
        # -------------------------------------------------------------
        # 9. Reference_Materials.md
        # -------------------------------------------------------------
        ref_content = f"""# Reference Materials & Further Reading
## Topic {topic_id}: {topic_title}
**Course:** KTU CST414 Deep Learning (Scheme 2019)  
**Module:** {mod_num} | **Target Outcome:** {co} ({blooms})  
**Faculty:** Jane Rose | Department of Computer Science & Engineering  

---

### 1. Primary Textbook Citations
- **Ian Goodfellow, Yoshua Bengio, and Aaron Courville**, *Deep Learning*, MIT Press, 2016.
- **Charu C. Aggarwal**, *Neural Networks and Deep Learning: A Textbook*, Springer, 2018.
- **Michael Nielsen**, *Neural Networks and Deep Learning*, Determination Press, 2015.
- Specific citation: `{data['textbook_ref']}`

### 2. Relevant KTU Notes & Reference Files
- Official Syllabus: `DEEP LEARNING.pdf`
- Module Notes Reference: `CST414-M{mod_num}-Ktunotes.in.pdf`

### 3. Suggested Research Papers & Seminal Readings
- Foundational and historical literature corresponding to {topic_title}.
- High-impact review articles and conference proceedings (NeurIPS, ICML, ICLR, CVPR).

### 4. Interactive Demonstration Instructions
- The executable practical demonstration for this topic is located at `Python_Demo.ipynb`.
- **Local Execution:** Run using Jupyter Notebook or VS Code Python Interactive window.
- **Google Colab:** Open Google Colab, upload `Python_Demo.ipynb`, and run all cells.
- Dependencies: `numpy`, `matplotlib`. No external GPU or complex frameworks required.
"""
        with open(os.path.join(target_dir, "Reference_Materials.md"), "w", encoding="utf-8") as f:
            f.write(ref_content)
            
    print("Successfully generated all 36 daily teaching packages (324 individual files)!")

if __name__ == "__main__":
    generate_daily_resources()

