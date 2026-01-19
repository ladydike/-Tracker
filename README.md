# ■ Jihye's 업무 Tracker

회의록 관리 및 업무 추적을 위한 로컬 웹 애플리케이션입니다.

## 📌 주요 기능

| 기능 | 설명 |
|------|------|
| **대시보드** | 월별 캘린더, 프로젝트별 회의 현황, 최근 회의 목록, 액션 아이템 관리 |
| **회의록 요약 뷰어** | Markdown 기반 회의록을 섹션별 카드 형태로 시각화 |
| **AI 어시스턴트** | Claude API 기반 회의록 검색 및 질의응답 (RAG) |
| **업무일지** | 날짜별 작업 기록 및 상태 관리 |
| **폴더 네비게이션** | 회의록 폴더 구조 기반 탐색 |

---

## 🚀 설치 및 실행 방법

### 1. 필수 요구사항

- **Python 3.8 이상** 설치 필요
- **Anthropic API Key** (Claude AI 사용 시)

### 2. 저장소 클론

```bash
git clone https://github.com/YOUR_USERNAME/task-tracker.git
cd task-tracker
```

### 3. Python 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. 회의록 폴더 구조 만들기

아래와 같은 폴더 구조를 만들고, 본인의 회의록 파일(.txt)을 넣으세요:

```
task-tracker/
├── backend.py
├── milestone_tracker.html
├── requirements.txt
├── S2_meeting/           ← S2 프로젝트 회의록
│   └── 250101 S2_프로젝트명.txt
├── MR_meeting/           ← MR 관련 회의록
│   └── 250101 프로젝트_회의.txt
├── MR_Team/              ← MR 팀 내부 회의
│   ├── Team weekly/
│   ├── AI Study/
│   ├── Dashboard/
│   └── ...
└── 비정기회의/            ← 비정기 회의록
    └── 250101 회의명.txt
```

### 5. 백엔드 서버 실행

**Windows PowerShell:**
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-api03-YOUR-API-KEY"
python backend.py
```

**Windows CMD:**
```cmd
set ANTHROPIC_API_KEY=sk-ant-api03-YOUR-API-KEY
python backend.py
```

**Mac/Linux:**
```bash
export ANTHROPIC_API_KEY="sk-ant-api03-YOUR-API-KEY"
python backend.py
```

### 6. 웹 앱 열기

브라우저에서 `milestone_tracker.html` 파일을 직접 열거나,
서버 실행 후 `http://localhost:8000` 접속

---

## 📁 회의록 파일 네이밍 규칙

파일명에 따라 자동으로 날짜와 카테고리가 인식됩니다:

```
YYMMDD 회의명.txt
```

**예시:**
- `260115 mr team weekly.txt` → 2026년 1월 15일, MR Team Weekly
- `251124 S2_딩컴.txt` → 2025년 11월 24일, S2 딩컴 미팅

---

## 📝 회의록 요약 파일 (선택)

원본 `.txt` 파일과 같은 폴더에 `_Summary.md` 파일을 만들면 요약 뷰어에서 확인할 수 있습니다:

```
260115 mr team weekly.txt          ← 원본
260115 mr team weekly_Summary.md   ← 요약본 (마크다운)
```

---

## 🔑 API Key 발급 방법

AI 어시스턴트 기능을 사용하려면 Anthropic API Key가 필요합니다:

1. [console.anthropic.com](https://console.anthropic.com) 접속
2. 회원가입 또는 로그인
3. **API Keys** 메뉴에서 새 키 생성
4. 생성된 키를 복사하여 환경변수로 설정

> ⚠️ **주의:** API Key는 절대 코드에 직접 넣지 마세요. 환경변수로만 설정하세요.

---

## 🎨 커스터마이징

### 폴더 구조 변경

`backend.py`의 상단에서 기본 폴더를 설정할 수 있습니다:

```python
MEETINGS_DIR = "."  # 현재 폴더 기준으로 회의록 검색
```

### UI 색상/스타일 변경

`milestone_tracker.html` 파일의 CSS 부분을 수정하세요.

---

## 📞 문의

개발: Jihye Choi  
문의: [이메일 또는 Slack 채널]

---

## 📜 라이선스

이 프로젝트는 내부 업무용으로 개발되었습니다.
