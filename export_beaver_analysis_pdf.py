from pathlib import Path
import re

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Image

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'beaver_analysis.md'
OUT = ROOT / 'beaver_analysis.pdf'
IMG_DIR = ROOT / 'analysis_outputs'


def clean_inline(text: str) -> str:
    text = re.sub(r'!\[.*?\]\((.*?)\)', '', text)
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    return text.strip()


def md_to_pdf(text: str):
    styles = getSampleStyleSheet()
    body = ParagraphStyle('Body', parent=styles['BodyText'], fontName='Helvetica', fontSize=10, leading=13, spaceAfter=6)
    heading1 = ParagraphStyle('Heading1', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=16, leading=18, spaceAfter=10)
    heading2 = ParagraphStyle('Heading2', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=13, leading=15, spaceAfter=8)
    heading3 = ParagraphStyle('Heading3', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=11, leading=14, spaceAfter=6)
    bullet = ParagraphStyle('Bullet', parent=body, leftIndent=12, bulletIndent=12)
    caption = ParagraphStyle('Caption', parent=body, fontSize=9, leading=11, textColor='#444444', spaceBefore=3)

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )
    story = []

    in_code = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        if line.strip().startswith('```'):
            in_code = not in_code
            continue
        if in_code:
            continue

        stripped = line.strip()
        if not stripped:
            continue

        if re.match(r'^!\[.*?\]\((.*?)\)$', stripped):
            match = re.match(r'^!\[.*?\]\((.*?)\)$', stripped)
            img_path = ROOT / match.group(1)
            if img_path.exists():
                story.append(Paragraph('Figure', caption))
                story.append(Image(str(img_path), width=5.8 * inch, height=3.3 * inch))
            continue

        if stripped.startswith('### '):
            story.append(Paragraph(clean_inline(stripped[4:]), heading3))
            continue
        if stripped.startswith('## '):
            story.append(Paragraph(clean_inline(stripped[3:]), heading2))
            continue
        if stripped.startswith('# '):
            story.append(Paragraph(clean_inline(stripped[2:]), heading1))
            continue

        if stripped.startswith('- '):
            story.append(Paragraph(clean_inline(stripped[2:]), bullet))
            continue

        story.append(Paragraph(clean_inline(stripped), body))

    for img_name in ['accuracy_by_setting.png', 'subtask_heatmap.png', 'resource_tradeoff.png']:
        img_path = IMG_DIR / img_name
        if img_path.exists():
            story.append(Paragraph(img_name, caption))
            story.append(Image(str(img_path), width=5.8 * inch, height=3.3 * inch))
            story.append(Spacer(1, 0.08 * inch))

    doc.build(story)


def main():
    text = SRC.read_text(encoding='utf-8')
    md_to_pdf(text)
    print(f'Wrote PDF: {OUT}')


if __name__ == '__main__':
    main()
