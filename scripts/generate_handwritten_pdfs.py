import os
import json
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register Kalam handwritten fonts
font_dir = r'c:\Users\Asus\Downloads\dl414\fonts'
pdfmetrics.registerFont(TTFont('Handwritten', os.path.join(font_dir, 'Kalam-Regular.ttf')))
pdfmetrics.registerFont(TTFont('Handwritten-Bold', os.path.join(font_dir, 'Kalam-Bold.ttf')))

# Load master course data
data_file = r'c:\Users\Asus\Downloads\dl414\DEEP_LEARNING_CST414\15_FACULTY_DASHBOARD\src\data\courseData.json'
with open(data_file, 'r', encoding='utf-8') as f:
    course_data = json.load(f)

schedule = course_data['schedule']

# Setup styles
styles = getSampleStyleSheet()

cover_title_style = ParagraphStyle(
    'CoverTitle',
    fontName='Handwritten-Bold',
    fontSize=26,
    leading=32,
    textColor=colors.HexColor('#6A0E1D'),
    alignment=1
)
cover_sub_style = ParagraphStyle(
    'CoverSub',
    fontName='Handwritten-Bold',
    fontSize=14,
    leading=19,
    textColor=colors.HexColor('#0B1226'),
    alignment=1
)
cover_meta_style = ParagraphStyle(
    'CoverMeta',
    fontName='Handwritten',
    fontSize=11,
    leading=16,
    textColor=colors.HexColor('#334155'),
    alignment=1
)

day_header_style = ParagraphStyle(
    'DayHeader',
    fontName='Handwritten-Bold',
    fontSize=17,
    leading=21,
    textColor=colors.HexColor('#6A0E1D'),
    spaceAfter=4
)
day_meta_style = ParagraphStyle(
    'DayMeta',
    fontName='Handwritten',
    fontSize=9.5,
    leading=13,
    textColor=colors.HexColor('#475569')
)
h1_style = ParagraphStyle(
    'HandH1',
    fontName='Handwritten-Bold',
    fontSize=12.5,
    leading=16,
    textColor=colors.HexColor('#0F2C59'),
    spaceBefore=8,
    spaceAfter=3
)
h2_style = ParagraphStyle(
    'HandH2',
    fontName='Handwritten-Bold',
    fontSize=11,
    leading=14,
    textColor=colors.HexColor('#1E3A8A'),
    spaceBefore=6,
    spaceAfter=2
)
body_style = ParagraphStyle(
    'HandBody',
    fontName='Handwritten',
    fontSize=10,
    leading=14,
    textColor=colors.HexColor('#1E293B')
)
formula_style = ParagraphStyle(
    'HandFormula',
    fontName='Handwritten-Bold',
    fontSize=10,
    leading=14,
    textColor=colors.HexColor('#065F46')
)
callout_style = ParagraphStyle(
    'HandCallout',
    fontName='Handwritten-Bold',
    fontSize=9.5,
    leading=13,
    textColor=colors.HexColor('#854D0E')
)
viva_q_style = ParagraphStyle(
    'HandVivaQ',
    fontName='Handwritten-Bold',
    fontSize=9.5,
    leading=13,
    textColor=colors.HexColor('#1E293B')
)
viva_a_style = ParagraphStyle(
    'HandVivaA',
    fontName='Handwritten',
    fontSize=9.5,
    leading=13,
    textColor=colors.HexColor('#047857')
)

def draw_notebook_page(canvas_obj, doc_obj):
    canvas_obj.saveState()
    # Left vertical red margin line (notebook look)
    canvas_obj.setStrokeColor(colors.HexColor('#F87171'))
    canvas_obj.setLineWidth(1.2)
    canvas_obj.line(48, 40, 48, 800)
    
    # Second double-margin line
    canvas_obj.setStrokeColor(colors.HexColor('#FECACA'))
    canvas_obj.setLineWidth(0.6)
    canvas_obj.line(52, 40, 52, 800)
    
    # Faint notebook horizontal ruled lines (24 pt spacing)
    canvas_obj.setStrokeColor(colors.HexColor('#E2E8F0'))
    canvas_obj.setLineWidth(0.4)
    for y in range(50, 790, 24):
        canvas_obj.line(52, y, 555, y)
        
    # Running Header
    canvas_obj.setFont('Handwritten-Bold', 8.5)
    canvas_obj.setFillColor(colors.HexColor('#64748B'))
    canvas_obj.drawString(60, 814, 'KTU CST414 DEEP LEARNING (Scheme 2019) • Faculty: Jane Rose, Dept of CSE')
    canvas_obj.setStrokeColor(colors.HexColor('#CBD5E1'))
    canvas_obj.setLineWidth(0.8)
    canvas_obj.line(48, 808, 555, 808)
    
    # Running Footer
    canvas_obj.setFont('Handwritten', 8.5)
    canvas_obj.setFillColor(colors.HexColor('#64748B'))
    canvas_obj.drawString(60, 28, 'Handwritten Lecture Notes Series • High-Resolution Print Edition')
    canvas_obj.drawRightString(555, 28, f'Page {doc_obj.page}')
    canvas_obj.restoreState()

def clean_html(text):
    if not isinstance(text, str):
        return str(text)
    # Replace xml/html tags
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;').replace('>', '&gt;')
    # Re-enable specific safe tags
    text = text.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
    text = text.replace('&lt;i&gt;', '<i>').replace('&lt;/i&gt;', '</i>')
    text = text.replace('&lt;br/&gt;', '<br/>').replace('&lt;br&gt;', '<br/>')
    return text

def build_day_story(day_item):
    story = []
    d = day_item['data']
    day_num = day_item['day_number']
    mod = day_item['module']
    top_id = day_item['topic_id']
    title = day_item['title']
    date_disp = day_item['date_display']
    period = day_item['period']
    co = day_item['co']
    blooms = day_item['blooms']
    
    # Lecture Header Banner
    header_table_data = [
        [
            Paragraph(f"<b>LECTURE #{day_num:02d} • MODULE {mod} (Topic {top_id})</b>", day_header_style),
            Paragraph(f"<b>Date:</b> {date_disp}<br/><b>Period:</b> {period.split('(')[0]}", day_meta_style)
        ],
        [
            Paragraph(f"<b>Title: {title}</b>", h1_style),
            Paragraph(f"<b>CO:</b> {co} &nbsp; <b>Bloom's:</b> {blooms}", day_meta_style)
        ]
    ]
    t = Table(header_table_data, colWidths=[330, 160])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FEF2F2')),
        ('BOX', (0,0), (-1,-1), 1.2, colors.HexColor('#B91C1C')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#FECACA')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))
    
    # Syllabus Flag note for Day 14 and 15
    if day_num in (14, 15):
        flag_p = Paragraph(f"★ <b>FACULTY SYLLABUS NOTE:</b> Official KTU 2.7 & 2.8 duplication flagged. This lecture covers: <b>{title}</b>.", callout_style)
        flag_table = Table([[flag_p]], colWidths=[490])
        flag_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FEF9C3')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CA8A04')),
            ('PADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(flag_table)
        story.append(Spacer(1, 6))

    # 1. Prerequisites & Key Terms
    prereq = d.get('prerequisites', 'Linear algebra, Calculus, Python basics')
    terms = ', '.join(d.get('key_terminology', []))
    meta_p = Paragraph(f"<b>Prerequisites:</b> {prereq}<br/><b>Key Terminology:</b> <i>{terms}</i>", body_style)
    story.append(meta_p)
    story.append(Spacer(1, 6))
    
    # 2. Detailed Handwritten Notes Sections
    notes = d.get('notes', {})
    for sec_key, sec_val in notes.items():
        sec_title = sec_key.replace('_', ' ').title()
        is_exam = 'exam' in sec_key.lower()
        
        story.append(Paragraph(f"✎ <u>{sec_title}</u>", h1_style))
        
        if isinstance(sec_val, str):
            clean_p = clean_html(sec_val)
            story.append(Paragraph(clean_p, body_style))
        elif isinstance(sec_val, list):
            for item in sec_val:
                item_str = clean_html(str(item))
                story.append(Paragraph(f"• {item_str}", body_style))
        elif isinstance(sec_val, dict):
            for sub_k, sub_v in sec_val.items():
                sub_title = sub_k.replace('_', ' ').title()
                story.append(Paragraph(f"<b>{sub_title}:</b>", h2_style))
                story.append(Paragraph(clean_html(str(sub_v)), body_style))
        
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 6))
    
    # 3. Hand-Solved Numerical Problems
    numericals = d.get('numerical_problems', [])
    if numericals:
        story.append(Paragraph("✍ <u>Worked Numerical Problems (Hand-Derived Step-by-Step)</u>", h1_style))
        for p_idx, prob in enumerate(numericals):
            prob_title = prob.get('title', f'Problem {p_idx+1}')
            p_type = prob.get('type', 'Standard')
            statement = clean_html(prob.get('statement', ''))
            given = clean_html(prob.get('given', ''))
            formulas = clean_html(prob.get('formulas', ''))
            steps = prob.get('steps', [])
            answer = clean_html(prob.get('answer', ''))
            interpretation = clean_html(prob.get('interpretation', ''))
            
            p_content = [
                Paragraph(f"<b>Problem {p_idx+1} ({p_type}): {prob_title}</b>", h2_style),
                Paragraph(f"<b>Statement:</b> {statement}", body_style),
                Spacer(1, 3),
                Paragraph(f"<b>Given:</b> {given}", body_style),
                Paragraph(f"<b>Governing Formula:</b> <code>{formulas}</code>", formula_style),
                Spacer(1, 3),
                Paragraph("<b>Step-by-Step Calculation:</b>", h2_style),
            ]
            for st in steps:
                p_content.append(Paragraph(f"&nbsp;&nbsp;➔ {clean_html(st)}", body_style))
            
            p_content.append(Spacer(1, 3))
            p_content.append(Paragraph(f"<b>[ FINAL EVALUATED ANSWER: {answer} ]</b>", formula_style))
            if interpretation:
                p_content.append(Paragraph(f"<i>Pedagogical Note:</i> {interpretation}", body_style))
                
            num_table = Table([[p_content]], colWidths=[490])
            num_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F0FDF4')),
                ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#16A34A')),
                ('PADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(num_table)
            story.append(Spacer(1, 6))

    # 4. Real World Industry Application
    app = d.get('real_world_application', {})
    if app:
        app_title = clean_html(app.get('title', 'Industrial Deployment'))
        app_domain = clean_html(app.get('domain', 'AI Engineering'))
        app_flow = clean_html(app.get('workflow', ''))
        app_benefit = clean_html(app.get('benefit', ''))
        
        app_table_data = [
            [Paragraph(f"🌐 <b>Real-World Industry Case ({app_domain}): {app_title}</b>", h2_style)],
            [Paragraph(f"<b>Pipeline Workflow:</b> {app_flow}", body_style)],
            [Paragraph(f"<b>Industrial Impact & Benefit:</b> {app_benefit}", formula_style)]
        ]
        app_t = Table(app_table_data, colWidths=[490])
        app_t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#EFF6FF')),
            ('BOX', (0,0), (-1,-1), 0.8, colors.HexColor('#2563EB')),
            ('PADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(app_t)
        story.append(Spacer(1, 6))

    # 5. Viva Voce Questions (First 5 key questions for notebook brevity)
    viva = d.get('viva_questions', [])
    if viva:
        story.append(Paragraph("❓ <u>Key Oral Viva Voce Checkpoints</u>", h1_style))
        for v_idx, v in enumerate(viva[:5]):
            vq = clean_html(v.get('q', ''))
            va = clean_html(v.get('a', ''))
            story.append(Paragraph(f"<b>Q{v_idx+1}: {vq}</b>", viva_q_style))
            story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;<b>Ans:</b> {va}", viva_a_style))
            story.append(Spacer(1, 2))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceBefore=5, spaceAfter=15))
    story.append(PageBreak())
    
    return story

def generate_pdf(filename, title_header, day_items):
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=56,
        rightMargin=40,
        topMargin=46,
        bottomMargin=46
    )
    
    story = []
    
    # Cover Page
    story.append(Spacer(1, 80))
    story.append(Paragraph("APJ ABDUL KALAM TECHNOLOGICAL UNIVERSITY", cover_sub_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("KTU CST414 — DEEP LEARNING", cover_title_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"<b>{title_header}</b>", cover_title_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph("B.Tech Computer Science and Engineering • Scheme 2019", cover_sub_style))
    story.append(Spacer(1, 40))
    
    meta_box = [
        [Paragraph("<b>Course Code:</b> CST414", cover_meta_style), Paragraph("<b>Credits:</b> 3", cover_meta_style)],
        [Paragraph("<b>Course Name:</b> Deep Learning", cover_meta_style), Paragraph("<b>Lecture Hours:</b> 36", cover_meta_style)],
        [Paragraph("<b>Faculty In-Charge:</b> Jane Rose", cover_meta_style), Paragraph("<b>Department:</b> Computer Science & Engineering", cover_meta_style)],
        [Paragraph("<b>Design:</b> High-Contrast Printable Notebook", cover_meta_style), Paragraph("<b>Font:</b> Authentic Chalk & Kalam Ink", cover_meta_style)],
    ]
    mt = Table(meta_box, colWidths=[240, 240])
    mt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FEF2F2')),
        ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor('#7B1113')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#FECACA')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'CENTER')
    ]))
    story.append(mt)
    story.append(Spacer(1, 60))
    story.append(Paragraph("<i>Prepared for students and faculty for classroom lectures, print study, and university examination preparation.</i>", cover_meta_style))
    story.append(PageBreak())
    
    # Table of Contents
    story.append(Paragraph("<u>COURSE LECTURE INDEX & MAPPING</u>", h1_style))
    story.append(Spacer(1, 10))
    toc_data = [["Day", "Topic ID", "Lecture Title", "Module", "CO", "Date"]]
    for item in day_items:
        toc_data.append([
            f"#{item['day_number']:02d}",
            item['topic_id'],
            item['title'][:40] + ('...' if len(item['title']) > 40 else ''),
            f"M{item['module']}",
            item['co'],
            item['date_display']
        ])
    toc_table = Table(toc_data, colWidths=[32, 50, 240, 48, 40, 70])
    toc_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#7B1113')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,0), 'Handwritten-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('FONTNAME', (0,1), (-1,-1), 'Handwritten'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')])
    ]))
    story.append(toc_table)
    story.append(PageBreak())
    
    # Days Content
    for item in day_items:
        story.extend(build_day_story(item))
        
    doc.build(story, onFirstPage=draw_notebook_page, onLaterPages=draw_notebook_page)
    print(f"Generated: {os.path.basename(filename)} ({os.path.getsize(filename)} bytes)")

# Directories to output
dashboard_dl_dir = r'c:\Users\Asus\Downloads\dl414\DEEP_LEARNING_CST414\15_FACULTY_DASHBOARD\public\downloads'
revision_notes_dir = r'c:\Users\Asus\Downloads\dl414\DEEP_LEARNING_CST414\13_STUDENT_REVISION_NOTES'
slides_dir = r'c:\Users\Asus\Downloads\dl414\DEEP_LEARNING_CST414\14_PRESENTATION_SLIDES'
os.makedirs(dashboard_dl_dir, exist_ok=True)
os.makedirs(revision_notes_dir, exist_ok=True)
os.makedirs(slides_dir, exist_ok=True)

print("Starting generation of Handwritten Printable PDFs...")

# 1. Master Complete Handwritten Notes (All 36 Days)
master_pdf_dash = os.path.join(dashboard_dl_dir, 'CST414_Deep_Learning_Master_Handwritten_Notes.pdf')
master_pdf_rev = os.path.join(revision_notes_dir, 'CST414_Deep_Learning_Master_Handwritten_Notes.pdf')
generate_pdf(master_pdf_dash, "HANDWRITTEN LECTURE NOTES & DERIVATION MANUSCRIPT", schedule)
import shutil
shutil.copyfile(master_pdf_dash, master_pdf_rev)

# 2. Module 1 to 5 Individual Booklets
for m in range(1, 6):
    mod_items = [s for s in schedule if s['module'] == m]
    mod_name = f"CST414_Module_{m}_Handwritten_Notes.pdf"
    target_dash = os.path.join(dashboard_dl_dir, mod_name)
    target_rev = os.path.join(revision_notes_dir, mod_name)
    generate_pdf(target_dash, f"MODULE {m} HANDWRITTEN LECTURE NOTES", mod_items)
    shutil.copyfile(target_dash, target_rev)

print("ALL HANDWRITTEN PDFS GENERATED SUCCESSFULLY!")
