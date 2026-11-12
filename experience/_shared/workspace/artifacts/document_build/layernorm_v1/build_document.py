from pathlib import Path
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'docs/experiments/cached_core_20261105/03_layernorm.md'
OUTPUT = SOURCE.with_suffix('.docx')

def el(tag, text=None):
    node = OxmlElement(tag)
    if text is not None:
        node.text = text
    return node

def mr(text):
    r = el('m:r')
    r.append(el('m:t', text))
    return r

def seq(*nodes):
    return [mr(n) if isinstance(n, str) else n for n in nodes]

def sub(base, index):
    node = el('m:sSub')
    for tag, values in [('m:e', seq(base)), ('m:sub', seq(index))]:
        part = el(tag)
        part.extend(values)
        node.append(part)
    return node

def sup(base, index):
    node = el('m:sSup')
    for tag, values in [('m:e', seq(base)), ('m:sup', seq(index))]:
        part = el(tag)
        part.extend(values)
        node.append(part)
    return node

def frac(top, bottom):
    node = el('m:f')
    for tag, values in [('m:num', top), ('m:den', bottom)]:
        part = el(tag)
        part.extend(values)
        node.append(part)
    return node

def root(nodes):
    node = el('m:rad')
    props = el('m:radPr')
    hide = el('m:degHide')
    hide.set(qn('m:val'), '1')
    props.append(hide)
    node.append(props)
    node.append(el('m:deg'))
    body = el('m:e')
    body.extend(nodes)
    node.append(body)
    return node

def math_paragraph(doc, nodes):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(9)
    m = el('m:oMathPara')
    props = el('m:oMathParaPr')
    align = el('m:jc')
    align.set(qn('m:val'), 'left')
    props.append(align)
    m.append(props)
    body = el('m:oMath')
    body.extend(nodes)
    m.append(body)
    p._p.append(m)

def rich(p, text):
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', lambda m: m[1] + ' (' + Path(m[2]).name + ')', text)
    text = text.replace('`', '')
    for j, chunk in enumerate(text.split('**')):
        r = p.add_run(chunk)
        if j % 2:
            r.bold = True
    return p

def set_font(style, size, bold=False):
    style.font.name = 'Malgun Gothic'
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0, 0, 0)
    fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
    for attr in ['ascii', 'hAnsi', 'eastAsia']:
        fonts.set(qn('w:' + attr), 'Malgun Gothic')

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.top_margin = sec.bottom_margin = Inches(.72)
sec.left_margin = sec.right_margin = Inches(.8)
set_font(doc.styles['Normal'], 11)
doc.styles['Normal'].paragraph_format.line_spacing = 1.15
doc.styles['Normal'].paragraph_format.space_after = Pt(7)
normal_ppr = doc.styles['Normal'].element.get_or_add_pPr()
snap = el('w:snapToGrid'); snap.set(qn('w:val'),'0'); normal_ppr.append(snap)
set_font(doc.styles['Title'], 20, True)
doc.styles['Title'].paragraph_format.space_after = Pt(12)
for name, size in [('Heading 1', 14), ('Heading 2', 11.5)]:
    set_font(doc.styles[name], size, True)
    doc.styles[name].paragraph_format.space_before = Pt(13)
    doc.styles[name].paragraph_format.space_after = Pt(7)
    doc.styles[name].paragraph_format.keep_with_next = True
for name in ['List Bullet', 'List Number']:
    set_font(doc.styles[name], 11)
    doc.styles[name].paragraph_format.space_after = Pt(6)

lines = SOURCE.read_text(encoding='utf-8-sig').splitlines()
i = 0
while i < len(lines):
    line = lines[i].strip()
    if not line:
        i += 1
        continue
    if line.startswith('# '):
        title = line[2:].replace('③ ', '실험 3 ').replace('·', ' 및 ')
        rich(doc.add_paragraph(style='Title'), title)
    elif line.startswith('## '):
        heading = re.sub(r'^(\d+)\.\s*', r'\1 ', line[3:]).replace('·', ' 및 ')
        rich(doc.add_paragraph(style='Heading 1'), heading)
    elif line.startswith('### '):
        rich(doc.add_paragraph(style='Heading 2'), line[4:].replace('·', ' 및 '))
    elif line == '\\[':
        j = i + 1
        while lines[j].strip() != '\\]':
            j += 1
        latex = '\n'.join(lines[i+1:j])
        if latex.startswith('z_j'):
            variance = el('m:sSubSup')
            for tag, values in [('m:e', seq('σ')), ('m:sub', seq('h')), ('m:sup', seq('2'))]:
                part = el(tag); part.extend(values); variance.append(part)
            math_paragraph(doc, seq(sub('z','j'), ' = ', sub('γ','j'), frac(seq(sub('h','j'), ' − ', sub('μ','h')), seq(root(seq(variance, ' + ', sup('10','−12'))))), ' + ', sub('β','j')))
        elif latex.startswith('a='):
            math_paragraph(doc, seq('a = ', sup('w','T'), 'q + b', '     ', 'p(normal | q) = sigmoid(a)'))
        else:
            math_paragraph(doc, seq('L = −', frac(seq('1'),seq('2')), sub('mean','normal'), ' log p − ', frac(seq('1'),seq('2')), sub('mean','spoof'), ' log(1 − p)'))
        i = j
    elif line.startswith('```'):
        j = i + 1
        block_paragraphs = []
        while j < len(lines) and not lines[j].startswith('```'):
            if lines[j].strip():
                p = rich(doc.add_paragraph(), lines[j].strip())
                p.paragraph_format.left_indent = Inches(.12)
                p.paragraph_format.space_after = Pt(4)
                block_paragraphs.append(p)
            j += 1
        for p in block_paragraphs[:-1]:
            p.paragraph_format.keep_with_next = True
        i = j
    elif line.startswith('|'):
        rows = []
        j = i
        while j < len(lines) and lines[j].strip().startswith('|'):
            values = [x.strip() for x in lines[j].strip().strip('|').split('|')]
            if not all(re.fullmatch(r'[:\- ]+', v) for v in values):
                rows.append(values)
            j += 1
        table = doc.add_table(rows=0, cols=len(rows[0]))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        widths = {3:[1.45,4.7,.75],4:[1.4,3.25,.7,1.55],5:[3.05,.65,1.2,.95,1.05]}[len(rows[0])]
        total_width = 6.9
        widths = [w * total_width/sum(widths) for w in widths]
        for col,w in zip(table.columns,widths):
            col.width = Inches(w)
        for row_id,values in enumerate(rows):
            row = table.add_row()
            trpr = row._tr.get_or_add_trPr()
            trpr.append(el('w:cantSplit'))
            if row_id == 0:
                trpr.append(el('w:tblHeader'))
            for col_id,(cell,value) in enumerate(zip(row.cells,values)):
                cell.width = Inches(widths[col_id])
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                props = cell._tc.get_or_add_tcPr()
                borders = el('w:tcBorders')
                for edge in ['top','left','bottom','right']:
                    border = el('w:'+edge)
                    for attr,val in [('val','single'),('sz','4'),('color','D9D9D9')]:
                        border.set(qn('w:'+attr),val)
                    borders.append(border)
                props.append(borders)
                margins = el('w:tcMar')
                for edge in ['top','bottom','left','right']:
                    margin = el('w:'+edge)
                    margin.set(qn('w:w'),'90')
                    margin.set(qn('w:type'),'dxa')
                    margins.append(margin)
                props.append(margins)
                if row_id == 0:
                    shade = el('w:shd'); shade.set(qn('w:fill'),'E7EFF6'); props.append(shade)
                p = rich(cell.paragraphs[0],value)
                p.paragraph_format.line_spacing = 1.08
                p.paragraph_format.space_after = Pt(2)
                if len(rows[0]) == 5 and col_id > 0 or col_id == len(rows[0])-1 and len(rows[0]) == 3:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.size = Pt(10)
                    if row_id == 0:
                        run.bold = True
        gap = doc.add_paragraph()
        gap.paragraph_format.space_after = Pt(2)
        gap.paragraph_format.space_before = Pt(0)
        gap.paragraph_format.line_spacing = Pt(3)
        i = j - 1
    elif line.startswith('- '):
        rich(doc.add_paragraph(style='List Bullet'),line[2:])
    elif re.match(r'^\d+\. ',line):
        rich(doc.add_paragraph(style='List Number'),re.sub(r'^\d+\. ','',line))
    else:
        rich(doc.add_paragraph(),line)
    i += 1

doc.core_properties.title = 'EBM 최종 LayerNorm 전후 표현의 정상 및 스푸프 판별 정보 비교'
doc.core_properties.author = ''
doc.core_properties.last_modified_by = ''
doc.core_properties.subject = '고정 EBM 표현에 대한 로지스틱 분류기 교차검증 결과'
for node in list(doc.styles.element.iter(qn('w:pBdr'))):
    node.getparent().remove(node)
for node in list(doc.element.iter(qn('w:pBdr'))):
    node.getparent().remove(node)
doc.save(OUTPUT)
print(OUTPUT)
