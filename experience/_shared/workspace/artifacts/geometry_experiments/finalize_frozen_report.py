"""Finalize the reviewed analysis after every experiment and audit completes."""
from pathlib import Path
import json
from render_frozen_analysis import render,OUT

def main():
    report=render()
    assert report['all_experiments_complete'], 'Finalization requires all completed experiments and audits'
    report['status']='complete'
    (OUT/'final_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Final reviewed analysis saved',report['path'])

if __name__=='__main__':main()
