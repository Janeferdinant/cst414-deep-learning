import os
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

font_dir = r'c:\Users\Asus\Downloads\dl414\fonts'
pdfmetrics.registerFont(TTFont('Handwritten', os.path.join(font_dir, 'Kalam-Regular.ttf')))
pdfmetrics.registerFont(TTFont('Handwritten-Bold', os.path.join(font_dir, 'Kalam-Bold.ttf')))

data_file = r'c:\Users\Asus\Downloads\dl414\DEEP_LEARNING_CST414\15_FACULTY_DASHBOARD\src\data\courseData.json'
with open(data_file, 'r', encoding='utf-8') as f:
    course_data = json.load(f)

schedule = course_data['schedule']

styles = getSampleStyleSheet()

slide_title_style = ParagraphStyle(
    'SlideTitle',
    fontName='Handwritten-Bold',
    fontSize=14,
    leading=18,
    textColor=colors.HexColor('#6A0E1D')
)
slide_body_style = ParagraphStyle(
    'SlideBody',
    fontName='Handwritten',
    fontSize=9.5,
    leading=13.5,
    textColor=colors.HexColor('#1E293B')
)
slide_formula_style = ParagraphStyle(
    'SlideFormula',
    fontName='Handwritten-Bold',
    fontSize=9.5,
    leading=13.5,
    textColor=colors.HexColor('#065F46')
)
slide_meta_style = ParagraphStyle(
    'SlideMeta',
    fontName='Handwritten',
    fontSize=8.5,
    leading=11,
    textColor=colors.HexColor('#64748B')
)

def draw_slides_background(canvas_obj, doc_obj):
    canvas_obj.saveState()
    # Running top header
    canvas_obj.setFont('Handwritten-Bold', 8.5)
    canvas_obj.setFillColor(colors.HexColor('#64748B'))
    canvas_obj.drawString(50, 814, 'KTU CST414 DEEP LEARNING (Scheme 2019) • Faculty: Jane Rose, Dept of CSE')
    canvas_obj.drawRightString(555, 814, 'Printable Slide Handout')
    canvas_obj.setStrokeColor(colors.HexColor('#CBD5E1'))
    canvas_obj.setLineWidth(0.8)
    canvas_obj.line(50, 808, 555, 808)
    
    # Running bottom footer
    canvas_obj.setFont('Handwritten', 8.5)
    canvas_obj.drawString(50, 24, 'Handwritten 2-Slide Handout Edition with Lecture Note Rulings')
    canvas_obj.drawRightString(555, 24, f'Page {doc_obj.page}')
    canvas_obj.restoreState()

def clean_html(text):
    if not isinstance(text, str):
        return str(text)
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;').replace('>', '&gt;')
    text = text.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
    text = text.replace('&lt;i&gt;', '<i>').replace('&lt;/i&gt;', '</i>')
    text = text.replace('&lt;br/&gt;', '<br/>').replace('&lt;br&gt;', '<br/>')
    return text

def create_slide_box(day_item, slide_type=1):
    day_num = day_item['day_number']
    mod = day_item['module']
    top_id = day_item['topic_id']
    title = day_item['title']
    d = day_item['data']
    
    if slide_type == 1:
        # Core Theory & Architecture Slide
        bullets = []
        notes = d.get('notes', {})
        first_few = list(notes.items())[:2]
        for k, v in first_few:
            sec_name = k.replace('_', ' ').title()
            val_txt = clean_html(str(v)[:220])
            bullets.append(f"• <b>{sec_name}:</b> {val_txt}...")
            
        content = [
            Paragraph(f"<b>LECTURE #{day_num:02d} [Slide 1/2]: {title}</b>", slide_title_style),
            Paragraph(f"Module {mod} • Topic {top_id} • CO: {day_item['co']} ({day_item['blooms']})", slide_meta_style),
            Spacer(1, 4),
            Paragraph('<br/>'.join(bullets), slide_body_style),
            Spacer(1, 4),
            Paragraph(f"<b>Key Terms:</b> <i>{', '.join(d.get('key_terminology', [])[:4])}</i>", slide_meta_style)
        ]
    else:
        # Mathematical Proof / Solved Numerical Slide
        numericals = d.get('numerical_problems', [])
        app = d.get('real_world_application', {})
        
        prob_txt = "Standard derivation & formula formulation."
        ans_txt = ""
        if numericals:
            p = numericals[0]
            prob_txt = f"<b>{p.get('title', 'Numerical')}:</b> {p.get('statement', '')[:160]}..."
            ans_txt = f"<b>Result:</b> {p.get('answer', '')}"
            
        app_txt = f"<b>Case ({app.get('domain', 'AI')}):</b> {app.get('title', '')} — {app.get('benefit', '')[:120]}..."
        
        content = [
            Paragraph(f"<b>LECTURE #{day_num:02d} [Slide 2/2]: Formulation &amp; Industry Case</b>", slide_title_style),
            Paragraph(f"Mathematical Formulation &bull; Numerical Solution &bull; Production Application", slide_meta_style),
            Spacer(1, 4),
            Paragraph(clean_html(prob_txt), slide_body_style),
            Paragraph(clean_html(ans_txt), slide_formula_style),
            Spacer(1, 4),
            Paragraph(clean_html(app_txt), slide_body_style)
        ]
        
    t = Table([[content]], colWidths=[500])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FFFFFF')),
        ('BOX', (0,0), (-1,-1), 1.2, colors.HexColor('#0F2C59')),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    
    # Below the slide, add 3-4 blank ruled lines for student handwritten notes
    notes_lines = [
        Paragraph("✍ <i>Lecture Notes &amp; Derivation Annotations:</i>", slide_meta_style),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E1'), spaceBefore=8, spaceAfter=8),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E1'), spaceBefore=8, spaceAfter=8),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E1'), spaceBefore=8, spaceAfter=14),
    ]
    
    return [t] + notes_lines

def build_slides_pdf():
    target_dash = r'c:\Users\Asus\Downloads\dl414\DEEP_LEARNING_CST414\15_FACULTY_DASHBOARD\public\downloads\CST414_Handwritten_Lecture_Slides_Handout.pdf'
    target_slides = r'c:\Users\Asus\Downloads\dl414\DEEP_LEARNING_CST414\14_PRESENTATION_SLIDES\CST414_Handwritten_Lecture_Slides_Handout.pdf'
    
    doc = SimpleDocTemplate(
        target_dash,
        pagesize=A4,
        leftMargin=45,
        rightMargin=45,
        topMargin=40,
        bottomMargin=35
    )
    
    story = []
    
    # Cover Slide
    story.append(Spacer(1, 100))
    story.append(Paragraph("APJ ABDUL KALAM TECHNOLOGICAL UNIVERSITY", ParagraphStyle('CSub', fontName='Handwritten-Bold', fontSize=14, leading=18, alignment=1, textColor=colors.HexColor('#0F2C59'))))
    story.append(Spacer(1, 10))
    story.append(Paragraph("KTU CST414 — DEEP LEARNING", ParagraphStyle('CTitle', fontName='Handwritten-Bold', fontSize=26, leading=32, alignment=1, textColor=colors.HexColor('#6A0E1D'))))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>COMPLETE 36-LECTURE HANDWRITTEN SLIDES HANDOUT</b>", ParagraphStyle('CSub2', fontName='Handwritten-Bold', fontSize=16, leading=22, alignment=1, textColor=colors.HexColor('#0F2C59'))))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Printable 2-Slides-Per-Page Edition with Student Derivation Ruling Lines", ParagraphStyle('CDesc', fontName='Handwritten', fontSize=12, leading=16, alignment=1, textColor=colors.HexColor('#475569'))))
    story.append(Spacer(1, 30))
    story.append(Paragraph("Faculty In-Charge: <b>Jane Rose</b> | Department of Computer Science & Engineering", ParagraphStyle('CFac', fontName='Handwritten-Bold', fontSize=12, leading=16, alignment=1, textColor=colors.HexColor('#7B1113'))))
    story.append(PageBreak())
    
    # Add each day's 2 slides (1 page per day with 2 slide boxes + annotation lines)
    for day_item in schedule:
        story.extend(create_slide_box(day_item, slide_type=1))
        story.append(Spacer(1, 6))
        story.extend(create_slide_box(day_item, slide_type=2))
        story.append(PageBreak())
        
    doc.build(story, onFirstPage=draw_slides_background, onLaterPages=draw_slides_background)
    
    import shutil
    shutil.copyfile(target_dash, target_slides)
    print("Generated Slides Handout PDF:", target_dash, f"({os.path.getsize(target_dash)} bytes)")

build_slides_pdf()
