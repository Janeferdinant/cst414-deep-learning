"""
Generator for Course Overview, Academic Calendar, and Subject Plan Documents
KTU CST414 Deep Learning (Scheme 2019)
Faculty: Jane Rose | Department of Computer Science & Engineering
"""

import os
import shutil
import datetime
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from pptx import Presentation
from pptx.util import Inches as PptxInches, Pt as PptxPt
from pptx.dml.color import RGBColor as PptxRGBColor

from curriculum_all import ALL_36_DAYS, PROVISIONAL_SCHEDULE, PROVISIONAL_BLOCKED_DATES

BASE_DIR = os.path.abspath("DEEP_LEARNING_CST414")

def ensure_dirs():
    os.makedirs(os.path.join(BASE_DIR, "00_COURSE_OVERVIEW"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "01_ACADEMIC_CALENDAR"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "02_SUBJECT_PLAN"), exist_ok=True)

def style_header_cell(cell, text, bg_hex="5B0612", fg_hex="FFFFFF", bold=True):
    cell.value = text
    cell.font = Font(name="Calibri", size=11, bold=bold, color=fg_hex)
    cell.fill = PatternFill(start_color=bg_hex, end_color=bg_hex, fill_type="solid")
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

def generate_course_overview():
    print("Generating 00_COURSE_OVERVIEW files...")
    target_dir = os.path.join(BASE_DIR, "00_COURSE_OVERVIEW")
    
    # 1. Copy Syllabus PDF
    src_pdf = os.path.abspath("DEEP LEARNING.pdf")
    if os.path.exists(src_pdf):
        shutil.copy(src_pdf, os.path.join(target_dir, "Course_Syllabus.pdf"))
    
    # 2. Course_Outcomes.docx
    doc = Document()
    doc.add_heading("KTU CST414: DEEP LEARNING", level=0)
    p = doc.add_paragraph()
    p.add_run("Course Outcomes, Bloom's Taxonomy & PO Mapping Matrix\n").bold = True
    p.add_run("Faculty: Jane Rose | Department of Computer Science & Engineering\n")
    p.add_run("Academic Scheme: KTU 2019 | Credits: 3 | Total Hours: 36\n")
    
    doc.add_heading("Official Course Outcomes (COs)", level=1)
    cos = [
        ("CO1", "Illustrate the basic concepts of neural networks and practical issues.", "Understand / Apply", "M1 (7 hrs)"),
        ("CO2", "Outline regularization and optimization techniques for deep neural networks.", "Understand / Apply", "M2 (9 hrs)"),
        ("CO3", "Implement foundation layers of CNN, including pooling and convolutions.", "Apply", "M3 (6 hrs)"),
        ("CO4", "Implement a sequence model using recurrent neural networks.", "Apply", "M4 (5 hrs)"),
        ("CO5", "Use different neural network and deep learning models for practical applications.", "Apply", "M5 (9 hrs)")
    ]
    t = doc.add_table(rows=1, cols=4)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    hdr[0].text = "CO #"
    hdr[1].text = "Course Outcome Statement"
    hdr[2].text = "Bloom's Level"
    hdr[3].text = "Target Module"
    for code, stmt, bloom, mod in cos:
        row = t.add_row().cells
        row[0].text = code
        row[1].text = stmt
        row[2].text = bloom
        row[3].text = mod
        
    doc.add_heading("Program Outcome (PO) Mapping", level=1)
    t_po = doc.add_table(rows=1, cols=13)
    t_po.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_po = t_po.rows[0].cells
    hdr_po[0].text = "CO"
    for i in range(1, 13):
        hdr_po[i].text = f"PO{i}"
        
    po_mappings = [
        ("CO1", [3, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 2]),
        ("CO2", [3, 3, 3, 2, 2, 0, 0, 0, 0, 0, 0, 2]),
        ("CO3", [3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 3]),
        ("CO4", [3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 3]),
        ("CO5", [3, 3, 3, 3, 3, 2, 1, 1, 2, 2, 1, 3])
    ]
    for co_code, vals in po_mappings:
        row = t_po.add_row().cells
        row[0].text = co_code
        for idx, v in enumerate(vals):
            row[idx+1].text = str(v) if v > 0 else "-"
            
    doc.save(os.path.join(target_dir, "Course_Outcomes.docx"))
    
    # 3. Academic_Calendar_Analysis.xlsx
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Calendar Analysis"
    headers = ["Parameter", "Value / Observation", "KTU Regulation Reference", "Remarks"]
    for col_idx, h in enumerate(headers, 1):
        style_header_cell(ws.cell(row=1, column=col_idx), h)
        
    cal_data = [
        ("Status of Academic Calendar", "PROVISIONAL (Institutional file missing in repo)", "Rule: Part 1 Integration", "Provisional plan formulated; pending official upload"),
        ("Course Code & Title", "CST414 Deep Learning", "KTU 2019 Scheme", "Program Elective Course (PEC)"),
        ("Approved Lecture Hours", "36 Hours", "Curriculum Credit 3", "All 36 lecture hours fully mapped"),
        ("Timetable Allocation", "3 Hours/Week (Mon P2, Wed P4, Fri P1)", "Approved College Slot Plan", "1 hour per lecture session"),
        ("Semester Commencement Date", "18-Aug-2025 (Provisional)", "Odd Semester Cycle", "Classwork begins with Module 1"),
        ("Semester Completion Date", "26-Nov-2025 (Provisional)", "Instructional Completion", "36 hours completed in Week 14"),
        ("Internal Series Test 1 (CIE-1)", "22-Oct-2025 to 24-Oct-2025", "After first 50% syllabus", "Covers Modules 1, 2, and half 3"),
        ("Internal Series Test 2 (CIE-2)", "10-Dec-2025 to 12-Dec-2025", "After remaining 50% syllabus", "Covers half Module 3, 4, and 5"),
        ("Last Instructional Day", "28-Nov-2025 (Provisional)", "Pre-Exam Buffer", "Revision and lab exams"),
        ("End Semester Exam (ESE)", "January 2026", "KTU Centralized Exam", "100 Marks, 3 Hours duration")
    ]
    for row_idx, r in enumerate(cal_data, 2):
        for col_idx, val in enumerate(r, 1):
            cell = ws.cell(row=row_idx, column=col_idx, value=val)
            cell.font = Font(name="Calibri", size=10)
    wb.save(os.path.join(target_dir, "Academic_Calendar_Analysis.xlsx"))
    
    # 4. Course_Overview.pptx
    prs = Presentation()
    # Title slide
    title_slide = prs.slides.add_slide(prs.slide_layouts[0])
    title_slide.shapes.title.text = "KTU CST414: DEEP LEARNING"
    title_slide.placeholders[1].text = "Course Overview & Instructional Roadmap\nJane Rose | Dept. of Computer Science & Engineering\nKTU 2019 Scheme | 3 Credits | 36 Hours"
    
    # Slide 2: Course Outcomes
    s2 = prs.slides.add_slide(prs.slide_layouts[1])
    s2.shapes.title.text = "Course Outcomes (COs)"
    tf = s2.placeholders[1].text_frame
    tf.text = "CO1: Neural Networks Foundations & Issues (Module 1, 7 hrs)"
    p = tf.add_paragraph(); p.text = "CO2: Optimization & Regularization Techniques (Module 2, 9 hrs)"
    p = tf.add_paragraph(); p.text = "CO3: Convolutional Neural Networks & Pooling (Module 3, 6 hrs)"
    p = tf.add_paragraph(); p.text = "CO4: Recurrent Sequence Architectures (Module 4, 5 hrs)"
    p = tf.add_paragraph(); p.text = "CO5: Advanced Applications & Representations (Module 5, 9 hrs)"
    
    prs.save(os.path.join(target_dir, "Course_Overview.pptx"))

def generate_academic_calendar_files():
    print("Generating 01_ACADEMIC_CALENDAR files...")
    target_dir = os.path.join(BASE_DIR, "01_ACADEMIC_CALENDAR")
    
    # 1. Academic_Calendar.xlsx
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Academic Calendar"
    headers = ["Week #", "Date Window", "Working Days", "Teaching Hours Available", "Scheduled Topics", "Academic Events / Remarks"]
    for col_idx, h in enumerate(headers, 1):
        style_header_cell(ws.cell(row=1, column=col_idx), h)
        
    weeks_data = [
        ("Week 1", "18-Aug-2025 to 22-Aug-2025", 5, 3, "Day 1 to Day 3 (M1: 1.1 - 1.3)", "Semester Commencement"),
        ("Week 2", "25-Aug-2025 to 29-Aug-2025", 4, 2, "Day 4 to Day 5 (M1: 1.4 - 1.5)", "27-Aug Sree Krishna Jayanti (Holiday)"),
        ("Week 3", "01-Sep-2025 to 05-Sep-2025", 2, 1, "Day 6 (M1: 1.6)", "03-Sep to 05-Sep Onam Vacation (Holidays)"),
        ("Week 4", "08-Sep-2025 to 12-Sep-2025", 4, 2, "Day 7 to Day 8 (M1: 1.7, M2: 2.1)", "08-Sep Onam Extension; Module 1 Complete"),
        ("Week 5", "15-Sep-2025 to 19-Sep-2025", 5, 3, "Day 9 to Day 11 (M2: 2.2 - 2.4)", "Optimization Techniques"),
        ("Week 6", "22-Sep-2025 to 26-Sep-2025", 5, 3, "Day 12 to Day 14 (M2: 2.5 - 2.7)", "Adaptive Learning & Regularization"),
        ("Week 7", "29-Sep-2025 to 03-Oct-2025", 3, 2, "Day 15 to Day 16 (M2: 2.8 - 2.9)", "01-Oct Mahanavami, 02-Oct Gandhi Jayanti; M2 Complete"),
        ("Week 8", "06-Oct-2025 to 10-Oct-2025", 5, 3, "Day 17 to Day 19 (M3: 3.1 - 3.3)", "CNN Foundations & Pooling"),
        ("Week 9", "13-Oct-2025 to 17-Oct-2025", 5, 3, "Day 20 to Day 22 (M3: 3.4 - 3.6)", "Convolution Variants; Module 3 Complete"),
        ("Week 10", "20-Oct-2025 to 24-Oct-2025", 3, 1, "Day 23 (M4: 4.1)", "20-Oct Deepavali; 22-Oct to 24-Oct Series Test 1"),
        ("Week 11", "27-Oct-2025 to 31-Oct-2025", 5, 3, "Day 24 to Day 26 (M4: 4.2 - 4.4)", "Seq2Seq & LSTM Architectures"),
        ("Week 12", "03-Nov-2025 to 07-Nov-2025", 5, 3, "Day 27 to Day 29 (M4: 4.5, M5: 5.1 - 5.2)", "GRU & Vision/Speech Applications"),
        ("Week 13", "10-Nov-2025 to 14-Nov-2025", 5, 3, "Day 30 to Day 32 (M5: 5.3 - 5.5)", "NLP, Word2Vec & GloVe"),
        ("Week 14", "17-Nov-2025 to 21-Nov-2025", 5, 3, "Day 33 to Day 35 (M5: 5.6 - 5.8)", "Autoencoders, DBNs & Representations"),
        ("Week 15", "24-Nov-2025 to 28-Nov-2025", 5, 1, "Day 36 (M5: 5.9)", "26-Nov All 36 Lecture Hours Complete; Revision Begins")
    ]
    for row_idx, r in enumerate(weeks_data, 2):
        for col_idx, val in enumerate(r, 1):
            cell = ws.cell(row=row_idx, column=col_idx, value=val)
            cell.font = Font(name="Calibri", size=10)
    wb.save(os.path.join(target_dir, "Academic_Calendar.xlsx"))
    
    # 2. Holiday_List.xlsx
    wb_h = openpyxl.Workbook()
    ws_h = wb_h.active
    ws_h.title = "Holiday List"
    h_headers = ["Sl No", "Date", "Day", "Occasion / Holiday Description", "Type", "Status"]
    for col_idx, h in enumerate(h_headers, 1):
        style_header_cell(ws_h.cell(row=1, column=col_idx), h)
    for idx, (dt, occ) in enumerate(PROVISIONAL_BLOCKED_DATES.items(), 1):
        ws_h.cell(row=idx+1, column=1, value=idx)
        ws_h.cell(row=idx+1, column=2, value=dt.strftime("%d-%b-%Y"))
        ws_h.cell(row=idx+1, column=3, value=dt.strftime("%A"))
        ws_h.cell(row=idx+1, column=4, value=occ)
        ws_h.cell(row=idx+1, column=5, value="Institutional Exam" if "Internal" in occ else "Public Holiday")
        ws_h.cell(row=idx+1, column=6, value="Classes Blocked")
    wb_h.save(os.path.join(target_dir, "Holiday_List.xlsx"))
    
    # 3. Semester_Teaching_Schedule.xlsx
    wb_s = openpyxl.Workbook()
    ws_s = wb_s.active
    ws_s.title = "36-Hour Schedule"
    s_headers = ["Lecture #", "Date", "Day", "Period & Time", "Module", "Topic ID", "Topic Title", "CO", "Status"]
    for col_idx, h in enumerate(s_headers, 1):
        style_header_cell(ws_s.cell(row=1, column=col_idx), h)
    for item in PROVISIONAL_SCHEDULE:
        row_idx = item["day_number"] + 1
        ws_s.cell(row=row_idx, column=1, value=item["day_number"])
        ws_s.cell(row=row_idx, column=2, value=item["date_display"])
        ws_s.cell(row=row_idx, column=3, value=item["day_name"])
        ws_s.cell(row=row_idx, column=4, value=item["period"])
        ws_s.cell(row=row_idx, column=5, value=item["module"])
        ws_s.cell(row=row_idx, column=6, value=item["topic_id"])
        ws_s.cell(row=row_idx, column=7, value=item["title"])
        ws_s.cell(row=row_idx, column=8, value=item["co"])
        ws_s.cell(row=row_idx, column=9, value=item["status"])
    wb_s.save(os.path.join(target_dir, "Semester_Teaching_Schedule.xlsx"))
    
    # 4. Working_Days.xlsx
    shutil.copy(os.path.join(target_dir, "Academic_Calendar.xlsx"), os.path.join(target_dir, "Working_Days.xlsx"))

def generate_subject_plan_files():
    print("Generating 02_SUBJECT_PLAN files...")
    target_dir = os.path.join(BASE_DIR, "02_SUBJECT_PLAN")
    
    # 1. Official_Subject_Plan.docx
    doc = Document()
    doc.add_heading("KTU CST414: DEEP LEARNING", level=0)
    p = doc.add_paragraph()
    p.add_run("Approved Subject Plan & 36-Hour Teaching Structure\n").bold = True
    p.add_run("Faculty: Jane Rose | Department of Computer Science & Engineering\n")
    p.add_run("Scheme: KTU 2019 | L-T-P: 2-1-0 | Credits: 3\n")
    
    # Notice for 2.7 / 2.8 duplication
    p_warn = doc.add_paragraph()
    p_warn.add_run("[IMPORTANT FACULTY VERIFICATION NOTICE]\n").bold = True
    p_warn.add_run(
        "In the approved KTU teaching plan (Page 12 of DEEP LEARNING.pdf), topics 2.7 and 2.8 "
        "share the identical description: 'Parameter sharing and tying, Injecting noise at input, Ensemble methods'. "
        "To preserve the approved 36-hour syllabus without redundancy, Day 14 covers Parameter Sharing, Tying & Input Noise, "
        "while Day 15 is dedicated to Ensemble Methods & Model Averaging.\n"
    )
    
    # Module Summaries
    doc.add_heading("Course Modules & Hours Breakdown", level=1)
    t = doc.add_table(rows=1, cols=4)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    hdr[0].text = "Module"
    hdr[1].text = "Module Title"
    hdr[2].text = "Official Hours"
    hdr[3].text = "Target CO"
    
    mods = [
        ("Module 1", "Neural Networks (Perceptrons, MLPs, Backprop, Pathologies)", "7 Hours", "CO1"),
        ("Module 2", "Deep Learning (Optimization, Regularization, Dropout, Init)", "9 Hours", "CO2"),
        ("Module 3", "Convolutional Neural Networks (Convolutions, Pooling, Winograd)", "6 Hours", "CO3"),
        ("Module 4", "Recurrent Neural Networks (Unfolding, BPTT, LSTM, GRU)", "5 Hours", "CO4"),
        ("Module 5", "Application Areas (Vision, Speech, NLP, Word2Vec, GloVe, DBNs)", "9 Hours", "CO5")
    ]
    for m_num, m_title, m_hrs, m_co in mods:
        row = t.add_row().cells
        row[0].text = m_num
        row[1].text = m_title
        row[2].text = m_hrs
        row[3].text = m_co
        
    doc.add_heading("Detailed 36-Hour Teaching Schedule", level=1)
    t_36 = doc.add_table(rows=1, cols=5)
    t_36.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr36 = t_36.rows[0].cells
    hdr36[0].text = "Hour"
    hdr36[1].text = "Date"
    hdr36[2].text = "Topic ID"
    hdr36[3].text = "Topic Title"
    hdr36[4].text = "CO"
    
    for item in PROVISIONAL_SCHEDULE:
        row = t_36.add_row().cells
        row[0].text = str(item["day_number"])
        row[1].text = item["date_display"]
        row[2].text = item["topic_id"]
        row[3].text = item["title"]
        row[4].text = item["co"]
        
    doc.save(os.path.join(target_dir, "Official_Subject_Plan.docx"))
    
    # 2. Date_Wise_Teaching_Plan.xlsx
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Date-Wise Plan"
    headers = [
        "Day #", "Date", "Day", "Period", "Module", "Topic ID", "Topic Title", 
        "CO", "Bloom's Level", "Teaching Method", "Demo Tool", "Assessment"
    ]
    for col_idx, h in enumerate(headers, 1):
        style_header_cell(ws.cell(row=1, column=col_idx), h)
    for item in PROVISIONAL_SCHEDULE:
        row_idx = item["day_number"] + 1
        ws.cell(row=row_idx, column=1, value=item["day_number"])
        ws.cell(row=row_idx, column=2, value=item["date_display"])
        ws.cell(row=row_idx, column=3, value=item["day_name"])
        ws.cell(row=row_idx, column=4, value=item["period"])
        ws.cell(row=row_idx, column=5, value=item["module"])
        ws.cell(row=row_idx, column=6, value=item["topic_id"])
        ws.cell(row=row_idx, column=7, value=item["title"])
        ws.cell(row=row_idx, column=8, value=item["co"])
        ws.cell(row=row_idx, column=9, value=item["blooms"])
        ws.cell(row=row_idx, column=10, value="Chalk & Board, Interactive Slides")
        ws.cell(row=row_idx, column=11, value="Python, NumPy, Matplotlib, Colab")
        ws.cell(row=row_idx, column=12, value="Class Quiz & Viva")
    wb.save(os.path.join(target_dir, "Date_Wise_Teaching_Plan.xlsx"))
    
    # 3. Module_Wise_Plan.xlsx
    shutil.copy(os.path.join(target_dir, "Date_Wise_Teaching_Plan.xlsx"), os.path.join(target_dir, "Module_Wise_Plan.xlsx"))
    
    # 4. CO_Mapping.xlsx
    wb_co = openpyxl.Workbook()
    ws_co = wb_co.active
    ws_co.title = "CO Mapping"
    co_headers = ["Topic ID", "Topic Title", "Module", "CO1", "CO2", "CO3", "CO4", "CO5", "Bloom's Taxonomy"]
    for col_idx, h in enumerate(co_headers, 1):
        style_header_cell(ws_co.cell(row=1, column=col_idx), h)
    for idx, item in enumerate(PROVISIONAL_SCHEDULE, 2):
        ws_co.cell(row=idx, column=1, value=item["topic_id"])
        ws_co.cell(row=idx, column=2, value=item["title"])
        ws_co.cell(row=idx, column=3, value=item["module"])
        ws_co.cell(row=idx, column=4, value=3 if item["co"] == "CO1" else 0)
        ws_co.cell(row=idx, column=5, value=3 if item["co"] == "CO2" else 0)
        ws_co.cell(row=idx, column=6, value=3 if item["co"] == "CO3" else 0)
        ws_co.cell(row=idx, column=7, value=3 if item["co"] == "CO4" else 0)
        ws_co.cell(row=idx, column=8, value=3 if item["co"] == "CO5" else 0)
        ws_co.cell(row=idx, column=9, value=item["blooms"])
    wb_co.save(os.path.join(target_dir, "CO_Mapping.xlsx"))

if __name__ == "__main__":
    ensure_dirs()
    generate_course_overview()
    generate_academic_calendar_files()
    generate_subject_plan_files()
    print("All Overview, Calendar, and Subject Plan files generated successfully!")

