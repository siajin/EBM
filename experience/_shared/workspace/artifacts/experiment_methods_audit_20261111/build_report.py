"""Build a static Markdown methods report from reviewed local evidence only."""
import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
TARGET = ROOT / 'docs/experiments/ebm_completed_evidence_interpretation_20261111.md'

def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))

def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

records = read('artifacts/experiment_methods_audit_20261111/experiment_registry.json')
by_id = {r['id']: r for r in records}
assert len(by_id) == len(records) == 110
assert set(by_id) == {f'E{i:03}' for i in range(1,111)}
sources = sorted({s for r in records for s in r['sources']})
assert all((ROOT / s).is_file() for s in sources), [s for s in sources if not (ROOT / s).is_file()]

# Every inventoried directory is either linked to method items or explicitly classified.
manual = {
    'artifacts/audio_deepfake_comparison_20261106': (['E077'], '공개 예시의 오디오 비교 화면·측정 보조 자료'),
    'artifacts/ebm_B_checkpoints_20261108': (['E010'], 'B만 평가하려던 명세·소스; 이후 B/C 동시 전체 평가로 통합'),
    'artifacts/ebm_native_patch_20261106': (['E006','E027','E084'], '비반복 학습 구현·소스 대조'),
    'artifacts/eer_comparison_20261030': (['E001','E002','E008'], '기존 EER의 표·그림; 새로운 모델 실행 아님'),
    'artifacts/geometry_examples': (['E083'], '기하 설명용 예시·그림'),
    'artifacts/mild_duration_e20_20261110': (['E024','E025'], '독립 원음 학습 대기안; 사용자 정정으로 미시작 상태에서 대체'),
    'artifacts/monitoring': (['E084'], '진행 확인·실행 상태 기록; 성능 실험 아님'),
    'artifacts/short32_train_20261111': (['E026','E084'], '첫 검사 경로 오류 버전; 학습 시작 전 실패'),
    '21la_cache_finish_v1': (['E039'], '21LA 전체 특징 캐시의 후반 추출 완료'),
    '21la_cache_validation_v1': (['E039'], '전체 캐시 정렬·수치 검증'),
    'bonafide_domain_shift_v1': (['E061'], '정상 출처 차이의 초기 분석'),
    'cache_hypotheses_20261106_v1': (['E059'], '후속 캐시 가설 분석 초기판; 최신 v2와 구분'),
    'channel_keep_one_v1': (['E044'], '1차원 부호 점수의 초기 동점 처리 오류판'),
    'channel_keep_one_v2': (['E044'], '동점 처리 수정 중간판'),
    'channel_roc_explanation_v1': (['E044'], '같은 점수·ROC의 설명용 재계산'),
    'cross_19la_to_21la_20261106_v1': (['E060'], '19LA/21LA 클래스 교차 평가'),
    'ebm_full_github_audit_20261106_v1': (['E027','E084'], '원본 코드·학습 방식 대조; 성능 실험 아님'),
    'full_score_overlap_20261106_v1': (['E070'], '전체 가능 매칭 초기판; 후속 최적 비용판 v2로 보완'),
    'layernorm_input_example_v1': (['E045','E059'], '정규화 전후 값의 예시 재구성'),
    'native_short303_code_patch_20261106_v1': (['E006','E027','E084'], '비반복·짧은 입력 설정 구현 검사'),
    'padding_budget_20261106_v1': (['E071'], '반복/문맥 개입의 표본·예산 사전 검토'),
    'report_revisions': (['E083'], '동일 그림의 레이블·배치 수정; 새 통계 실험 아님'),
    'short3_training_design_20261106_v1': (['E006','E080'], '짧은 입력 303프레임 학습 설계와 후속 실제 B 구현의 관계'),
}
coverage = []
inventory = read('artifacts/experiment_methods_audit_20261111/source_inventory.json')
for entry in inventory:
    d = entry['directory']
    ids = [r['id'] for r in records if any(s.startswith(d+'/') for s in r['sources'])]
    explicit = manual.get(d, manual.get(Path(d).name))
    note = '해당 폴더의 설정·분석·결과 근거를 본문 항목에 연결'
    if explicit:
        ids = sorted(set(ids + explicit[0]))
        note = explicit[1]
    assert ids, f'Unclassified inventory directory: {d}'
    files = [p for p in (ROOT/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    coverage.append({'directory':d, 'method_ids':ids, 'disposition':note,
                     'file_count_inventory_only':len(files), 'direct_evidence_files':[s for s in sources if s.startswith(d+'/')]})

doc_manual = {
    'ebm_B_checkpoint_evaluation_20261108.md': ['E010'],
    'ebm_B_final_evaluation_started_20261107.md':['E009'],
    'ebm_B_monitoring_and_final_analysis_20261106.md':['E006','E009','E012'],
    'ebm_B_purpose_analysis_protocol_20261107.md':['E009','E011','E012'],
    'ebm_B_training_started_20261106.md':['E006'],
    'ebm_embedding_geometry_review.md':['E039','E040','E041'],
    'ebm_embedding_geometry_spec_20261105.md':['E039','E080'],
    'ebm_native_length_short3_code_changes_20261106.md':['E006','E027','E084'],
    'ebm_native_length_short3_training_plan_20261106.md':['E006','E080'],
    'ebm_duration_conditioned_centroids_plan_20261106.md':['E067'],
    '21la_cache_20261105.md':['E039'],
    'ebm_frozen_model_experiments_20261105.md':['E047','E052','E053','E055','E056','E057','E058'],
    'ebm_native_github_full_audit_20261106.md':['E027','E084'],
}
operational_docs = {'ebm_private_repository_setup_20261106.md', 'ebm_server_baseline_restore_20261106.md'}
documents = []
for p in sorted((ROOT/'docs/experiments').glob('*.md')):
    if p == TARGET:
        continue
    rel = p.relative_to(ROOT).as_posix()
    ids = [r['id'] for r in records if rel in r['sources']]
    ids = sorted(set(ids + doc_manual.get(p.name, [])))
    note = '과학적 성능 실험이 아닌 저장소·운영 복원 기록' if p.name in operational_docs else '방법·결과·설계 또는 이력의 근거 문서'
    assert ids or p.name in operational_docs, f'Unclassified document: {p.name}'
    documents.append({'document':rel,'method_ids':ids,'disposition':note})

train_map = {
    'EBM':['E001'],
    'EBM_e20_bs240_lr1e5_baseline240_speed08_12_crop20_20261111_v2':['E025'],
    'EBM_e50_bs160_lr1e5':['E002'],
    'EBM_e50_bs160_lr1e5_bpl_global':['E003'],
    'EBM_e50_bs240_lr1e5':['E005'],
    'EBM_e50_bs240_lr1e5_B_zeropad_short303':['E006'],
    'EBM_e50_bs240_lr1e5_C_nativeLong_short303':['E007'],
    'EBM_e50_bs240_lr1e5_OC_BPL_coverage_activity_20261110':['E021'],
    'EBM_e50_bs240_lr1e5_OC_BPL_coverage_activity_20261110_mid19LA':['E021'],
    'EBM_e50_bs240_lr1e5_original240_OC_BPL43_short32_LE2_20261111_v2':['E026'],
    'EBM_e50_bs320_lr1e5':['E004'],
}
server = read('artifacts/experiment_methods_audit_20261111/server_inventory.json')
assert set(train_map) == {x['name'] for x in server['training_directories']}
historical_map = read('artifacts/experiment_methods_audit_20261111/historical_coverage.json')
dump('source_coverage.json', {'artifact_directories':coverage,'documents':documents,'server_training_directories':train_map,
                            'historical_server_analysis_directories':historical_map,
                            'scope_note':'폴더 목록의 전수 분류와 선택한 핵심 근거의 검토. 모든 원음·대형 배열·모든 로그 행을 이번에 다시 계산했다는 뜻은 아니다.'})

hashes = []
for s in sources:
    p = ROOT / s
    hashes.append({'path':s,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
dump('source_hashes.json', hashes)

with (OUT/'experiment_registry.csv').open('w',encoding='utf-8-sig',newline='') as f:
    keys = list(records[0])
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r in records:
        w.writerow({k:'\n'.join(v) if isinstance(v,list) else v for k,v in r.items()})

GROUPS = [
    ('원래 학습·배치·정상쌍 구성',range(1,9)),
    ('전체 평가·반복·길이·속도 진단',range(9,20)),
    ('최근 학습·실패 버전·현재 추가 손실',range(20,28)),
    ('길이·구간·음성활동·원음 사용 정책',range(28,39)),
    ('고정 특징·좌표·방향 정보·추가 분류기',range(39,57)),
    ('내부 가중치·잡음·출처·정상 오류',range(57,68)),
    ('경로 밀도·문맥·저점유 방향·공개 예시',range(68,78)),
    ('미실행 설계·취소·특징 저장·구현 검증',range(78,85)),
    ('기존 변형 반응의 후속 재분석',[85]),
    ('서버의 초기 80epoch 모델: 분포·문맥·저에너지·추가 대조',range(86,111)),
]

def src_link(s):
    return f'[{s}](../../{s})'

common = (OUT/'common_methods.md').read_text(encoding='utf-8')
common = common.replace('2026년 10월 30일~11월 11일의 프로젝트 기록', '2026년 10~11월의 프로젝트 기록')
common = common.replace('## 3. 평가 수치와 불확실성을 읽는 방법', (OUT/'historical_common.md').read_text(encoding='utf-8')+'\n## 3. 평가 수치와 불확실성을 읽는 방법')
common = common.replace('일반 배치에서는 정상 항에 5, 위조 항에 5/9를 곱한다.',
    '원래 코드에서 분류 손실을 계산하는 표본 수가 4보다 크면 정상 항에 5, 위조 항에 5/9를 곱한다. 4개 이하인 작은 호출에는 이 클래스 가중치를 적용하지 않는다.')
term = (OUT/'terminology.md').read_text(encoding='utf-8')
common = common.replace('## 4. 실험 목록', term+'\n## 4. 실험 목록')
lines = [common,
    f'\n이 문서는 **{len(records)}개 방법·분석·설계 항목**을 다룬다. 이 수는 독립적으로 학습한 모델 수가 아니다. 하나의 항목 안에 명시된 비교 조건·실패 버전·재분석을 포함한다. 본문에서 상속한다고 적은 설정은 참조 항목과 공통 정의를 함께 적용한다.\n',
    '### 찾아보기\n',
    '|항목|내용|기록된 상태|\n|---|---|---|']
for r in records:
    lines.append(f"|[{r['id']}](#{r['id'].lower()})|{r['title']}|{r['status']}|")
for number,(title, ids) in enumerate(GROUPS,1):
    lines.append(f'\n### 4.{number}. {title}\n')
    for n in ids:
        r = by_id[f'E{n:03}']
        lines.extend([f'\n<a id="{r["id"].lower()}"></a>\n',f'#### {r["id"]}. {r["title"]}\n',
            f'**실행 상태:** {r["status"]}\n',f'**목적:** {r["purpose"]}\n',
            f'**모델·데이터:** {r["model_and_data"]}\n', '**구체적인 방법:**\n'])
        lines.extend(f'{i}. {s}' for i,s in enumerate(r['procedure'],1))
        for label,key in [('비교 조건과 통제','conditions'),('측정·선택·검증 규칙','measurement'),
                          ('실제로 얻은 결과','observed_result'),('해석과 한계','interpretation_and_limits')]:
            lines.append(f'\n**{label}:** {r[key]}\n')
        lines.append('**근거 파일:**\n')
        lines.extend('- '+src_link(s) for s in r['sources'])

lines += ['\n## 5. 최소 EER를 비교할 때의 실제 조건\n',
    '다음 표는 저장 기록을 읽어 요약한 값이다. 기존 배치 240의 과거 평가는 순환 반복 입력, B/C의 전체 체크포인트 평가는 비반복 입력이다. 따라서 한 표의 작은 수만으로 학습 방법의 공정한 순위를 확정하지 않는다. 각 평가셋에서 고른 epoch도 서로 다르다.\n',
    '|학습 모델|19LA 최소 EER, epoch|21LA 최소 EER, epoch|DF 최소 EER, epoch|후보·평가입력|',
    '|---|---:|---:|---:|---|',
    '|배치 80·학습률 1e-6·80epoch|0.280200%, 70|2.726782%, 75|2.101303%, 60|남은 과거 기록; 반복 입력|',
    '|배치 160·GPU별 정상쌍·50epoch|0.231135%, 25|1.016047%, 35|2.421145%, 25|남은 과거 기록; 반복 입력|',
    '|배치 160·GPU 간 정상쌍 공유·50epoch|0.255154%, 25|0.992801%, 35|2.333714%, 25|남은 과거 기록; 반복 입력|',
    '|배치 240·원래 반복 학습·50epoch|0.217539%, 25|1.052789%, 50|2.396259%, 15|15~50epoch의 8개 기록; 반복 입력|']
metrics = read('artifacts/ebm_B_C_checkpoints_20261109/metrics.json')
for model,label in [('B','상한 유지·반복 제거(B)'),('C','학습 긴 입력 상한 제거(C)')]:
    cells = []
    for dataset in ['19LA','21LA','21DF']:
        vals = [(v['EER_percent'], int(k.split('_e')[1])) for k,v in metrics[dataset].items() if k.startswith(model+'_e')]
        val, ep = min(vals)
        cells.append(f'{val:.6f}%, {ep}')
    lines.append(f'|{label}|'+ '|'.join(cells)+'|5~50epoch의 10개; 비반복 입력|')
lines += ['\n길이·음성활동을 맞춘 학습은 20epoch 후 중단했고 19LA의 관측 최소는 5epoch의 2.622022%다. 21LA·DF는 **지정한 20epoch만** 전체 평가했으므로 각각 8.908218%, 2.286637%를 여러 epoch 중 최소라고 쓰지 않는다. 현재 조건부 3초/2초 학습은 이 문서의 완료 자료 기준으로 첫 5epoch 19LA 2.437306%만 확인됐고, 최종 비교 결과는 없다.\n',
    '### 5.1 B/C의 모든 평가 epoch 원값\n',
    '이 표는 기존 실제 `metrics.json`에서 자동으로 생성했다. 같은 평가 실행의 입력·난수를 공유한 값이다.\n',
    '|모델|epoch|19LA EER %|21LA EER %|DF EER %|\n|---|---:|---:|---:|---:|']
for m in ['B','C']:
    for ep in range(5,51,5):
        vals = [metrics[d][f'{m}_e{ep}']['EER_percent'] for d in ['19LA','21LA','21DF']]
        lines.append(f'|{m}|{ep}|'+ '|'.join(f'{v:.9f}' for v in vals)+'|')
lines += ['\n## 6. 누락·중복·미실행을 확인한 범위\n',
    f'실험 목록을 로컬 핵심 근거 **{len(sources)}개 파일**, 분석 폴더 **{len(coverage)}개**, 기존 Markdown 문서 **{len(documents)}개**, 서버 학습 결과 폴더 **{len(train_map)}개**와 과거 서버 분석 폴더 **{len(historical_map)}개**에 연결했다. 아래 감사표에는 각 폴더가 어느 항목에 포함되는지와, 그림·운영 기록·이전 실패판인 경우의 처리 이유를 남겼다. 이것은 원음 전체와 대형 특징 배열을 이번에 다시 읽거나 모든 과거 추론을 다시 실행했다는 뜻은 아니다.\n',
    '- [전체 폴더·문서·서버 결과 대조표](../../artifacts/experiment_methods_audit_20261111/source_coverage.md)',
    '- [110개 항목의 JSON 원장](../../artifacts/experiment_methods_audit_20261111/experiment_registry.json)',
    '- [표 계산 프로그램에서 읽는 CSV 원장](../../artifacts/experiment_methods_audit_20261111/experiment_registry.csv)',
    '- [근거 파일의 크기·SHA256](../../artifacts/experiment_methods_audit_20261111/source_hashes.json)',
    '- [계산·누락·링크·상태 검사 결과](../../artifacts/experiment_methods_audit_20261111/validation.json)',
    '- [확장 전의 8개 핵심 결과 요약 보존본](../../artifacts/experiment_methods_audit_20261111/previous_summary.md)',
    '\n### 6.1 이번 점검에서 바로잡은 사항\n',
    '- 과거 XLS-R 마지막 층 저장 문서의 대기 상태를 그대로 쓰지 않고 외장 드라이브의 완료·수신 표식을 확인했다. 25,380개 저장 완료지만 이번에 20.895GB 배열을 다시 추출·전수 해시한 것은 아니다.',
    '- 완만한 속도/절단 변형 학습은 첫 버전이 학습 전 진입 오류, 두 번째 버전이 5epoch 후 평가 모듈 오류로 끝났다. 저장 모델·완료 EER가 없어서 성능 수치를 부여하지 않았다.',
    '- 현재 추가 길이 손실은 대상 K개에 대한 K×K 전체 교차 쌍이다. 같은 발화 쌍만 사용하는 설계로 설명하지 않았다.',
    '- 원음 <2초와 ≤2초, 정확한 4초와 4.0375초, 개발 고정 임계값과 평가셋 자체 EER 임계값을 구분했다.',
    '- 최근 반복 진단은 두 고정 50epoch 모델만 사용했다. 네 모델의 모든 저장 epoch를 반복/비반복으로 새로 평가하는 대기는 실행 전에 취소됐다.',
    '- 12행 합성 검사 파일을 실제 B/C 전체 평가로 사용하지 않았다. 기존 240의 마지막 50epoch 결과가 빠진 오래된 중간 요약보다 최종 실제 기록을 우선했다.',
    '\n### 6.2 기록만으로 확정하지 못한 부분\n',
    '오래된 모델의 코덱 실제 호출 횟수, 누락된 평가 epoch의 성능, 미보존 체크포인트의 새 조건 성능은 채워 넣지 않았다. 여러 도메인이 공유하는 원발화·재인코딩 계열의 의존성도 완전히 해결되지 않았다. 현재 진행 중인 학습의 최종 성능, 화자별 손실과 보수적 길이 손실 결합안, 삼중 조합, 확대 변형 반응 분류기의 효과는 미확정이다. 같은 조건에서 여러 seed를 학습한 통제 실험이 없는 비교는 그 한계를 각각 적었다.\n',
    '문서 작성에 따른 신규 학습·음성 추론·유료 자원은 0이다. 현재 학습과 기존 고정 실행 소스는 변경하지 않았다.\n']

if not (OUT/'previous_summary.md').exists():
    shutil.copyfile(TARGET, OUT/'previous_summary.md')
TARGET.write_text('\n'.join(lines), encoding='utf-8')

c = ['# 실험 방법서의 근거 범위 대조표\n',
     '각 폴더를 방법서 항목에 연결한다. 핵심 근거 파일은 본문과 source_hashes.json에서 확인한다. 폴더 안의 모든 파일을 독립 실험으로 세지 않는다.\n',
     '## 로컬 분석 폴더\n','|폴더|본문 항목|처리·의미|파일 목록 개수|\n|---|---|---|---:|']
for x in coverage:
    c.append(f'|{x["directory"]}|{", ".join(x["method_ids"])}|{x["disposition"]}|{x["file_count_inventory_only"]}|')
c += ['\n## 기존 실험 문서\n','|문서|본문 항목|처리|\n|---|---|---|']
for x in documents:
    c.append(f'|{x["document"]}|{", ".join(x["method_ids"]) or "운영 기록"}|{x["disposition"]}|')
c += ['\n## 서버 학습 결과 폴더\n','|실제 결과 폴더|본문 항목|\n|---|---|']
for name,ids in train_map.items():
    c.append(f'|{name}|{", ".join(ids)}|')
c += ['\n## 서버의 초기 분석 폴더\n','빈 초기 tail_factors 폴더는 결과가 있는 v2와 구별한다. 같은 표본·완료 자료의 후속 분석은 독립 원음 수로 합산하지 않는다.\n','|서버 분석 폴더|본문 항목|\n|---|---|']
for name,ids in sorted(historical_map.items()):
    c.append(f'|/home/mentee/results/analysis/{name}|{", ".join(ids)}|')
(OUT/'source_coverage.md').write_text('\n'.join(c)+'\n',encoding='utf-8')
print(json.dumps({'records':len(records),'sources':len(sources),'artifact_directories':len(coverage),'documents':len(documents),'server_training_directories':len(train_map),'report_bytes':TARGET.stat().st_size},ensure_ascii=False))
