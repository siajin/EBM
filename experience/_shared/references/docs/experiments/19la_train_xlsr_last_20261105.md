> 역사 문서 보존본. 원문 작성 당시의 상태·계획을 포함한다. 최신 상태와 확정 결과는 experience의 각 실험 README를 우선한다.

# 19LA train XLS-R 마지막 레이어: F: 직접 저장

요청: 19LA train 25,380개의 XLS-R 마지막 Transformer 출력을 float32로 저장한다. 160차원 EBM h와 구분한다. DF 추출 요청은 실행 전 19LA로 변경됐다.

현재 상태: 기존 R23 실험 정상 종료를 기다리는 노트북의 일회성 수신 대기열. **아직 추출 완료가 아니다.** 노트북·서버망·F: 연결을 대기부터 최종검수까지 유지해야 한다.

사용자는 서버에 출력 파일을 저장하지 말고 노트북에 저장하도록 정정했고 **F: Samsung USB**를 선택했다. 이전 서버 저장 대기는 실행 전에 취소했으며 launcher도 비활성화했다. 기존 R23 실험은 계속 진행한다.

노트북 저장 위치: `F:\IRLab\XLSR\19LA_train_last_20261105_01`

GPU/CPU 메모리→kernel FIFO→SSH→F:로 직접 전송한다. **서버에 hidden state 배열 파일을 저장하지 않는다.** 서버에는 자체 script·작은 metadata/검수JSON·log만 남긴다. 새 실행 기록 루트는 `/home/mentee/analysis/embedding_geometry/20261105_19la_train_xlsr_stream_01`이다.

| 파일 | 의미 |
| --- | --- |
| `metadata/index.parquet` | path를 제외한 음성ID·label·global_row·rank·feature_row |
| `rank0_last_hidden.npy` | `[12690,201,1024]`, float32 |
| `rank1_last_hidden.npy` | `[12690,201,1024]`, float32 |
| `metadata/input_audit.json` | 공식protocol/manifest/source hash와 설정 |
| `metadata/qc/summary.json` | 128개 eager 원본 출력 대조 및 GPU 간 일치 |
| `summary.json` | F:의 실제파일을 재읽고 검수한 최종기록 |
| `status.json` | 대기/수신/검수/실패/완료 현재상태 |
| `complete` | 내용0이면 전체수신·검수완료 |
| `receiver.stderr.log` | 로컬 수신기 오류기록 |

전체 shape는 `[25380,201,1024]`, 데이터 크기는 20,895,252,480바이트다. 중복 합본은 만들지 않는다. F: 여유 약76GB를 확인했다. C: 여유약16.8GB로 부족하여 사용자가 F:를 선택했다. 수신중에는`.partial.npy`이며 완료로 사용하지 않는다.

완료 후 재사용 예시(노트북 Python):

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

root = Path('F:/IRLab/XLSR/19LA_train_last_20261105_01')
assert (root / 'complete').read_text().strip() == '0'
assert json.loads((root / 'summary.json').read_text())['status'] == 'complete'
index = pd.read_parquet(root / 'metadata/index.parquet')
arrays = [np.load(root / f'rank{rank}_last_hidden.npy', mmap_mode='r')
          for rank in range(2)]
row = index.iloc[0]  # 또는 utterance_id로 선택
h = arrays[int(row['rank'])][int(row['feature_row'])]
assert h.shape == (201, 1024)
```

입력은 원래 EBM 평가용 EnrollmentSet과 동일한 첫64600샘플·짧은 음성 wrap padding이다. train augmentation/무작위 crop/추가 평균pooling/양자화는 적용하지 않는다. 백본은 기존 epoch50 실험에서 동결돼 있던 pretrained XLS-R 300M이며 EBM의160차원 후처리는 거치지 않는다.

노트북 수신기는 기존 R23 정상종료와 같은 컨테이너·유휴screen0·GPU두장유휴를 모두 확인한 뒤 한 번만 제출한다. 컨테이너의`CUDA_VISIBLE_DEVICES=0,1`로 실행한다. 기존작업실패나컨테이너변경시 재시작하지 않는다. 원래 연구실 source/checkpoint/음성을 복사·수정하지 않는다.

수신기 생존 중에만 자동절전을 억제하고 영구 전원설정은 바꾸지 않는다. PC종료/연결단절시 서버디스크로 대체 저장하지 않고 전송을 실패처리한다. 자동재시작은 없고 미완료부분은F:에남는다. 이전의 PC를꺼도진행된다는설명은취소된서버저장방식에만해당했다.

각블록의UID/행/bytes/SHA256과유한값을검사하고, 마지막에F:파일을재읽어전체hash/shape/dtype/index를검수한다. 합성2MiB의실제FIFO/SSH전송과로컬손상파일거부·정상복구후재읽기검사는통과했다. 검수기록은F:의`transport_verification.json`이다.

로컬 상태 확인:

```powershell
python -X utf8 artifacts/geometry_experiments/receive_xlsr_last_layer.py status
```

최종완료판정은F:의`summary.json status=complete`와`complete=0`이다. 서버`run.exit=0`만으로노트북저장완료라고하지않는다. 이전서버저장루트`20261105_19la_train_xlsr_last_01`은다시실행하지않는다.
