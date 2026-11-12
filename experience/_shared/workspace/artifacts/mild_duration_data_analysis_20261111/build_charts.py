import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
data=json.loads((HERE/'chart_data.json').read_text(encoding='utf-8'))
summary=json.loads((HERE/'summary.json').read_text(encoding='utf-8'))
config={}
labels={'original':'변형 전 원음','expected':'생성 정책의 기대분포','epoch1':'1 epoch 선택 일정 재현','epochs20':'20 epoch 누적 선택 일정 재현'}
overlap={'original':summary['before_similarity']['overlap_percent'],'expected':summary['expected_similarity'][1]['overlap_percent'],
    'epoch1':summary['epoch1_OC_similarity']['overlap_percent'],'epochs20':summary['cumulative20_OC_similarity']['0.1']['overlap_percent']}
for state,d in data['length'].items():
    config[state]={'id':'mild-length-comparison','title':labels[state]+' · 길이 구간별 비율',
        'description':f"0.1초 공통 구간 · 각 클래스 전체가 분모 · 분포 겹침 {overlap[state]:.4f}% · Train 원음 보나 2,580 / 스푸프 22,800",
        'summary':'생성 전 정책 분석과 선택 일정 재현이며, 변환 음성 실측값이 아닙니다.','height':300,
        'spec':{'type':'line','x':'초','y':['보나','스푸프'],'xLabel':'길이 (초)','yLabel':'클래스 내 비율',
            'xDomain':[0,14.1],'yDomain':[0,.065],'showLegend':True,
            'valueFormat':{'style':'percent','maximumFractionDigits':3},'colors':{'보나':'var(--chart-1)','스푸프':'var(--chart-2)'}},
        'columns':['초','보나','스푸프'],'rows':[[float(x),float(a),float(b)] for x,a,b in zip(d['centers'],d['bona'],d['spoof'])]}
d=data['speed']
config['speed']={'id':'mild-speed-comparison','title':'속도 배율의 클래스별 분포',
    'description':'0.01배 공통 구간 · 원음 선택 확률과 조건부 변형 확률 반영 · 분포 겹침 75.8428%',
    'summary':'배율 1보다 크면 빨라집니다. 길이 분포는 같아져도 변환 배율의 분포는 다릅니다.','height':260,
    'spec':{'type':'line','x':'배율','y':['보나','스푸프'],'xLabel':'속도 배율 (1보다 크면 빨라짐)','yLabel':'클래스 내 비율',
        'xDomain':[.8,1.2],'yDomain':[0,.09],'showLegend':True,
        'valueFormat':{'style':'percent','maximumFractionDigits':3},'colors':{'보나':'var(--chart-1)','스푸프':'var(--chart-2)'}},
    'columns':['배율','보나','스푸프'],'rows':[[float(x),float(a),float(b)] for x,a,b in zip(d['centers'],d['bona'],d['spoof'])]}
html=Path('C:/Users/sihwa/.codex/visualizations/2026/11/09/01a1c0c0-fcc9-7bb2-8c67-3ba6c09955af/mild-duration-analysis.html')
text=html.read_text(encoding='utf-8')
for name,placeholder in [('original','__ORIGINAL__'),('expected','__EXPECTED__'),('epoch1','__EPOCH1__'),('epochs20','__EPOCHS20__'),('speed','__SPEED__')]:
    assert placeholder in text
    text=text.replace(placeholder,json.dumps(config[name],ensure_ascii=False,separators=(',',':')).replace('<','\\u003c'))
html.write_bytes(text.encode('utf-8'))
(HERE/'chart_configs.json').write_text(json.dumps(config,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
print('HTML bytes',html.stat().st_size,'configs',len(config))
