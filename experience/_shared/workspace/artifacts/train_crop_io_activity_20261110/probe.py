import importlib.util,json,os,inspect
from pathlib import Path
import soundfile as sf
p=Path(__file__).parent
v={'modules':{n:bool(importlib.util.find_spec(n)) for n in ['webrtcvad','torchaudio','scipy','silero_vad','onnxruntime']},'soundfile_version':sf.__version__,'SoundFile_read_signature':str(inspect.signature(sf.SoundFile.read)),'cpu_affinity':len(os.sched_getaffinity(0)),'train_protocol_exists':Path('/data/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt').is_file()}
(p/'probe.json').write_text(json.dumps(v,indent=2))
