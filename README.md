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

## 🤖 Cursor AI 회의록 요약 규칙 (.cursorrules)

이 프로젝트에는 **Cursor AI가 회의록을 자동으로 요약**할 때 사용하는 규칙 파일(`.cursorrules`)이 포함되어 있습니다.

### 📋 현재 설정된 요약 모드

파일명/폴더명에 따라 **4가지 모드** 중 하나가 자동 선택됩니다:

| 모드 | 트리거 조건 | 요약 스타일 |
|------|-------------|-------------|
| **모드 1: QnA/101** | 파일명에 `QnA`, `Q&A`, `101` 포함 | 주제별 핵심 요약, 5~10개 글머리 기호 |
| **모드 2: S2** | 파일명/폴더에 `S2` 포함 | 프로젝트별 상세 현황, 액션 아이템 중심 |
| **모드 3: 분석/학습** | `게임장르분석`, `게임톡`, `mr process`, `weekly`, `dept`, `ai study` 포함 | 상세 강의 노트 형식, 예시와 비유 보존 |
| **모드 4: 일반 회의** | 위 조건에 해당하지 않는 모든 회의 | 풀 대본(Full Script) 형식, 화자 구분 |

### 💡 사용 방법

Cursor에서 회의록 파일을 열고 AI에게 요청하세요:

```
이 회의록 요약해줘
```

또는

```
260115 mr team weekly.txt 요약해줘
```

AI가 자동으로 적절한 모드를 선택하여 요약하고, `_Summary.md` 파일을 생성합니다.

### ✏️ .cursorrules 커스터마이징

본인의 업무 스타일에 맞게 규칙을 수정할 수 있습니다.

#### 1. 새로운 모드 추가하기

`.cursorrules` 파일을 열고, 기존 모드를 참고하여 새 모드를 추가하세요:

```markdown
## [모드 5] 새로운 회의 유형 (파일명에 특정키워드 포함 시)
**Role:** 역할 정의
**Goal:** 목표 정의
**Processing Rules:**
- 규칙 1
- 규칙 2
**Output Format:**
1. **## 1. 섹션 제목**: 내용
2. **## 2. 섹션 제목**: 내용
```

#### 2. 트리거 조건 변경하기

각 모드의 트리거 조건을 본인 팀의 파일명 규칙에 맞게 수정하세요:

```markdown
## [모드 2] S2 회의록 (파일명/폴더에 S2 포함 시)
```
↓ 변경 예시
```markdown
## [모드 2] 프로젝트 회의록 (파일명에 Project, PJ, 프로젝트 포함 시)
```

#### 3. 요약 상세도 조절하기

각 모드의 `Processing Rules`에서 상세도를 조절할 수 있습니다:

```markdown
**Processing Rules:**
- **상세성 (Detail):** 각 요점은 3~4문장 이상의 충분한 설명을 포함해야 합니다.
```

#### 4. 출력 형식 변경하기

`Output Format` 섹션에서 원하는 구조로 변경하세요:

```markdown
**Output Format:**
1. **## 1. 미팅 개요**: Executive Summary
2. **## 2. 상세 내용**: 주제별 정리
3. **## 3. 액션 아이템**: 담당자, 기한 포함
```

### ⚠️ 주의사항

- `.cursorrules` 파일은 **프로젝트 루트 폴더**에 위치해야 합니다.
- 파일명은 반드시 `.cursorrules`여야 합니다 (확장자 없음).
- 규칙 변경 후 Cursor를 **재시작**해야 적용됩니다.

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
