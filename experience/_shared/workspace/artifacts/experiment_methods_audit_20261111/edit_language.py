"""Whitespace-only Korean editing, plus explicitly listed terminology edits."""
import json
import os
import re
import sys
from pathlib import Path

OUT = Path(__file__).parent
sys.path.insert(0, str(OUT))
from editorial_overrides import RECORDS

sys.path.insert(0, str(Path(os.environ['TEMP']) / 'ebm_methods_spacing'))
from kiwipiepy import Kiwi

k = Kiwi(num_workers=2)
for w in ['원음', '위조', '파형', '분류기', '발화', '비반복', '공분산', '기울기', '교차검증',
          '정규화', '재추론', '미실행', '메타데이터', '재표집', '재가중', '임계값', '기준벡터',
          '후처리', '체크포인트', '음성활동', '저점유', '절단', '초파형',
          '데이터로더', '정규직교', '회전', '정상', '난수', '검산', '코사인', '잔차',
          '보관', '화자', '조건', '제거', '평가', '학습', '본학습', '후속', '표본',
          '코덱', '패딩', '반복', '비반복', '기준', '벡터', '특징', '군집', '보정',
          '집단', '목표', '원점수', '원본', '음량', '발화량', '음높이', '표준편차',
          '개수', '재학습', '재검산', '모집단', '검수', '정정', '보간', '신경망',
          '복원', '통계', '기하', '추출', '소수', '추가', '유지', '차원', '정규화',
          '개발', '학습률', '실행', '입력', '샘플', '계산', '검증', '저장']:
    if w != '초파형':
        k.add_user_word(w, 'NNG', 15)

def edit(s):
    spaced = k.space(s, reset_whitespace=True)
    assert re.sub(r'\s', '', spaced) == re.sub(r'\s', '', s), (s, spaced)
    # Keep common research words consistently separated; avoid splitting math symbols.
    fixes = {
        '원 음': '원음', '비 반복': '비반복', '공분 산': '공분산',
        '임계 값': '임계값', '메타 데이터': '메타데이터', '기준벡터': '기준 벡터',
        '재 표집': '재표집', '후 처리': '후처리', '미 실행': '미실행',
        '초파형': '초 파형', '검산완료': '검산 완료', '전체평가': '전체 평가',
        '고정모델': '고정 모델', '길이bin': '길이 구간', '길이 bin': '길이 구간',
        '클래스bin': '클래스 구간', '양GPU': '두 GPU', '양 GPU': '두 GPU',
        'GPUsmoke': 'GPU 사전 실행 검사', 'GPU smoke': 'GPU 사전 실행 검사',
        '두GPU': '두 GPU', '새GPU': '새 GPU', '정상쌍': '정상 쌍',
        '학습50epoch': '학습 50epoch', 'CPU검사': 'CPU 검사',
        '그대로한다': '그대로 한다', '하려했다': '하려 했다',
        '재 fit': '다시 적합', '회화자': '회 화자', '원점수': '변형 전 점수',
        '확률보정을': '확률 보정을', '미지출처': '미지 출처',
        '정규직교': '정규 직교', '초기하위': '초기 하위',
    }
    for a, b in fixes.items():
        spaced = spaced.replace(a, b)
    # Units adjacent to a Korean noun need a boundary. Decimal values stay intact.
    spaced = re.sub(r'(\d)(초|개|쌍|행|회|축|층|차원|샘플|조건|구간|배치)(?=[가-힣])', r'\1\2 ', spaced)
    spaced = re.sub(r'([가-힣])(?=(?:GPU|CPU|DF|OC|BPL|EER|XLS-R|RawBoost|Train|Adam|LayerNorm))', r'\1 ', spaced)
    spaced = re.sub(r'(?<=[0-9A-Za-z])(?=[가-힣])', ' ', spaced)
    spaced = re.sub(r'(?<=[가-힣])(?=[0-9A-Za-z])', ' ', spaced)
    # Restore ASCII identifiers and attach Korean particles split from identifiers/units.
    spaced = re.sub(r'([A-Za-z0-9])\s+_', r'\1_', spaced)
    spaced = re.sub(r'_\s+([A-Za-z0-9])', r'_\1', spaced)
    spaced = re.sub(r'(\d)\s+(초|개|쌍|회|행|축|층|배|차원|샘플|조건|구간|배치|차)(?=\b|[가-힣])', r'\1\2', spaced)
    spaced = re.sub(r'(?<=[A-Za-z0-9%])\s+(은|는|이|가|을|를|의|와|과|로|도|만|에)(?=\s|[.,;])', r'\1', spaced)
    spaced = re.sub(r'(개|회|초|쌍|행|축|층|차|배)\s+(는|은|를|을|로|가|이|만|다|였고|였다|이며|에서|까지)(?=\s|[.,;])', r'\1\2', spaced)
    spaced = re.sub(r'\s+([.,;])', r'\1', spaced)
    spaced = re.sub(r'([·/])\s+', r'\1', spaced)
    return spaced

clean = []
changes = []
for record in RECORDS:
    item = {}
    for key, val in record.items():
        if key in ('id', 'sources'):
            item[key] = val
        elif isinstance(val, list):
            item[key] = [edit(v) for v in val]
        else:
            item[key] = edit(val)
        if item[key] != val:
            changes.append({'id': record['id'], 'field': key})
    clean.append(item)
(OUT / 'experiment_registry.json').write_text(json.dumps(clean, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
(OUT / 'language_edit_log.json').write_text(json.dumps({'tool':'kiwipiepy whitespace suggestions + explicit editorial replacements', 'source_preserved':True, 'changes':changes}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(f'Edited {len(clean)} method records; sources and identifiers preserved.')
