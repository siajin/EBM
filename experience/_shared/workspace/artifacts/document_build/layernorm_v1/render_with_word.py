from pathlib import Path
import importlib.util
import subprocess
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SKILL = Path('C:/Users/sihwa/.codex/plugins/cache/openai-primary-runtime/documents/26.1026.10807/skills/documents')
spec = importlib.util.spec_from_file_location('render_docx', SKILL/'render_docx.py')
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)

def convert_with_word(doc_path, user_profile, convert_tmp_dir, stem, verbose):
    output = Path(convert_tmp_dir)/f'{stem}.pdf'
    result = subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(HERE/'export_word_pdf.ps1'),'-InputDocx',str(Path(doc_path).resolve()),'-OutputPdf',str(output)], capture_output=True, text=True, timeout=90)
    if result.returncode != 0 or not output.exists():
        raise RuntimeError(result.stdout+'\n'+result.stderr)
    return str(output), 'PDF conversion through Microsoft Word; packaged LibreOffice is unavailable on Windows.'

# Use the skill renderer for rasterization, with Word replacing only PDF export.
renderer.convert_to_pdf = convert_with_word
docx = ROOT/'docs/experiments/cached_core_20261105/03_layernorm.docx'
paths = renderer.rasterize(str(docx),str(HERE/'render'),150,False,True)
print('\n'.join(paths))
