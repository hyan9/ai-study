# AI Starter (Tablet-Friendly)

이 레포는 **갤럭시 탭 S9+ 같은 태블릿**에서도 바로 실행할 수 있도록 구성했습니다. **구글 Colab**이나 **GitHub Codespaces**에서 돌리면 됩니다. 로컬 설치 없이 시작할 수 있어요.

---

# 🚨 금연탐지 CCTV 관제 시스템 (Qwen3 TTS + 흡연 모델)

이 프로젝트는 다음을 자동으로 수행합니다.
- **Enos-123/smoking-detection** 모델로 흡연 여부 판단
- 카메라 프레임에서 **인상착의(캡션)** 추출
- **Qwen3 TTS** 최신 인간 목소리로 엄중한 경고 음성 출력
- 경고 기록을 `outputs/events.csv`에 저장
- 필요 시 경고 순간 **스냅샷 저장**

---

## ✅ 설치 방법

### 1) Python 패키지 설치
```bash
pip install -r requirements.txt
```

---

## ✅ 실행 방법 (내 노트북 카메라 사용)

```bash
python src/smoking_guardian.py --source 0 --play-audio --save-frames --tts-voice latest
```

### 옵션 설명
| 옵션 | 설명 |
|---|---|
| `--source 0` | 내장 카메라 (보통 0) |
| `--play-audio` | 경고 음성 재생 |
| `--save-frames` | 흡연 감지 시 스냅샷 저장 |
| `--tts-voice latest` | Qwen3 TTS 최신 인간 목소리 |

---

## ✅ CCTV/RTSP 스트림 사용

```bash
python src/smoking_guardian.py --source rtsp://<ip>:<port>/stream
```

---

## ✅ 모델 설정 변경

```bash
python src/smoking_guardian.py \
  --smoking-model Enos-123/smoking-detection \
  --caption-model Salesforce/blip-image-captioning-base \
  --tts-model Qwen/Qwen3-TTS \
  --tts-voice latest
```

---

## ✅ 성능 튜닝 (느릴 때)

```bash
python src/smoking_guardian.py \
  --min-smoking-score 0.65 \
  --person-score 0.7 \
  --cooldown 10 \
  --frame-skip 5 \
  --max-width 960
```

---

## ✅ 결과 저장 위치

| 파일 | 설명 |
|---|---|
| `outputs/events.csv` | 감지 기록 (시간, 라벨, 확률 등) |
| `outputs/tts/` | 경고 음성 WAV 파일 |
| `outputs/*.jpg` | 흡연 감지 시 스냅샷 |

---

## 📌 참고 사항
- Qwen3 TTS는 **반드시 사용**되도록 설정되어 있습니다.
- 더 자연스러운 음성을 원하면 `--tts-voice`에 모델이 제공하는 **최신 프리셋/스피커 이름**을 넣어주세요.
- GPU가 없으면 Colab을 추천합니다.
- 인상착의 텍스트는 기본적으로 캡션 모델을 사용하므로, 더 고급 인상착의 모델로 교체 가능합니다.

---

## 🛠️ 문제 해결 (에러가 날 때)

### 1) `Can't load image processor for 'Enos-123/smoking-detection'`
이 에러는 보통 **모델 파일을 못 받았거나**, **로컬에 같은 이름 폴더가 있어서** 생깁니다.

아래 순서대로 확인하세요:
1. 실행 폴더에 `Enos-123` 또는 `smoking-detection` 같은 이름의 **로컬 폴더가 없는지** 확인합니다.  
   (로컬 폴더가 있으면 Hugging Face 대신 그 폴더를 읽으려고 시도합니다.)
2. 인터넷 연결이 되는지 확인한 뒤, 모델을 다시 받도록 **캐시를 삭제**합니다.
3. `transformers` 버전을 최신으로 업데이트합니다.

예시:
```bash
pip install -U transformers
```

### 2) 모델을 로컬 체크포인트로 쓰고 싶은 경우
이미 다운로드한 모델이 있다면, `--smoking-model`에 **로컬 경로**를 직접 넣어 실행하세요.

예시:
```bash
python src/smoking_guardian.py \
  --source 0 \
  --smoking-model /path/to/smoking-detection-checkpoint
```

---

## ✅ Qwen3 TTS 목소리(프리셋/스피커) 선택 방법

Qwen3 TTS는 모델마다 지원하는 **프리셋/스피커 이름**이 다릅니다.  
아래 순서대로 확인하면 됩니다.

### 1) 모델 카드에서 프리셋 이름 확인
1. Hugging Face에서 **Qwen/Qwen3-TTS** 모델 페이지를 엽니다.
2. **Model card**의 *Available Voices / Speakers* 항목에서 지원하는 이름을 확인합니다.
3. 확인한 이름을 `--tts-voice`에 넣습니다.

예시:
```bash
python src/smoking_guardian.py \
  --source 0 \
  --play-audio \
  --save-frames \
  --tts-voice <모델카드에 있는_스피커이름>
```

> 최신 프리셋이 무엇인지 애매하면, 모델 카드에 적힌 가장 최근 업데이트의 프리셋 이름을 사용하세요.

---

## ✅ 내가 만든 목소리(파인튜닝) 쓰는 방법

> 반드시 Qwen3 TTS를 사용합니다. (다른 TTS는 사용하지 않습니다)

### 방법 A) 이미 파인튜닝된 Qwen3 TTS 체크포인트가 있는 경우
1. 내가 가진 체크포인트를 Hugging Face 혹은 로컬 폴더에 둡니다.
2. 아래처럼 `--tts-model`에 경로/리포를 넣어 실행합니다.

예시:
```bash
python src/smoking_guardian.py \
  --source 0 \
  --tts-model /path/to/my-qwen3-tts-checkpoint \
  --tts-voice <내가만든_스피커이름>
```

### 방법 B) 내 목소리로 파인튜닝해서 쓰고 싶은 경우
1. Qwen3 TTS의 공식 **fine-tuning 가이드/스크립트**를 확인합니다.  
   (모델 카드의 *Training / Fine-tuning* 섹션 참조)
2. 내 음성 데이터를 준비합니다. (깨끗한 WAV + 텍스트 자막 매칭)
3. 가이드대로 파인튜닝을 완료한 뒤, **체크포인트 경로**를 `--tts-model`로 넣고 실행합니다.

> 파인튜닝 방법은 Qwen3 TTS 공식 문서/모델카드 기준으로 진행해야 합니다.  
> 이 프로젝트는 **Qwen3 TTS만 사용**하도록 설계되어 있습니다.

---

## 폴더 구조
```
ai-starter-on-tablet/
  ├── src/
  │   ├── hello_ai.py
  │   └── smoking_guardian.py
  ├── requirements.txt
  ├── README.md
```
