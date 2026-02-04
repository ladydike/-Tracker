"""
MileStone Tracker - Backend Server
Claude API를 활용한 회의록 Q&A 및 프로젝트 트래킹 시스템
(수정: 실시간 요약 기능 제거, 기존 _Summary.md 파일 활용)
"""

import os
import re
import json
import glob
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

# 벡터 검색용 라이브러리
import numpy as np
from sentence_transformers import SentenceTransformer

# .env 파일 직접 로드
ENV_PATH = Path(r"C:\Users\lenachoi\.cursor\Practice\.env")
if ENV_PATH.exists():
    with open(ENV_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
    print(f"[ENV] Loaded from: {ENV_PATH}")
    print(f"[ENV] API Key exists: {bool(os.environ.get('ANTHROPIC_API_KEY'))}")
else:
    print(f"[ENV] Warning: {ENV_PATH} not found")

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import uvicorn

# ==================== Configuration ====================
PRACTICE_DIR = Path(r"C:\Users\lenachoi\.cursor\Practice")
CLAUDE_MODEL = "claude-sonnet-4-20250514"  # 최신 모델로 업데이트
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
CONTEXT_FILE = PRACTICE_DIR / "context_조직정보.md"

# 조직 배경 정보 로드
org_context = ""
if CONTEXT_FILE.exists():
    with open(CONTEXT_FILE, 'r', encoding='utf-8') as f:
        org_context = f.read()
    print(f"[Context] 조직 정보 로드 완료: {CONTEXT_FILE.name}")
else:
    print(f"[Context] 조직 정보 파일 없음: {CONTEXT_FILE}")

# Initialize FastAPI
app = FastAPI(title="MileStone Tracker API", version="1.1.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Vector Search 설정 ====================
print("[Vector Search] 임베딩 모델 로딩 중... (처음 실행 시 다운로드 필요)")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # 다국어 지원 모델
print("[Vector Search] 임베딩 모델 로드 완료!")

# 벡터 저장소 (메모리)
vector_store = {
    "ids": [],
    "embeddings": None,  # numpy array
}

def init_vector_db():
    """벡터 저장소 초기화"""
    global vector_store
    vector_store = {"ids": [], "embeddings": None}
    print("[Vector Search] 저장소 초기화 완료")

def vectorize_meetings():
    """모든 회의록을 벡터화하여 저장"""
    global vector_store
    
    if not store.meetings:
        print("[Vector Search] 벡터화할 회의록이 없습니다")
        return
    
    print(f"[Vector Search] {len(store.meetings)}개 회의록 벡터화 시작...")
    
    documents = []
    ids = []
    
    for meeting_id, meeting in store.meetings.items():
        # 요약본이 있으면 요약본, 없으면 원본 앞부분
        text = meeting.summary_content if meeting.summary_content else meeting.content[:3000]
        # 제목도 포함하여 검색 정확도 향상
        combined_text = f"제목: {meeting.title}\n폴더: {meeting.folder}\n내용: {text}"
        
        documents.append(combined_text)
        ids.append(meeting_id)
    
    # 배치로 임베딩 생성
    embeddings = embedding_model.encode(documents, show_progress_bar=True)
    
    vector_store["ids"] = ids
    vector_store["embeddings"] = embeddings
    
    print(f"[Vector Search] {len(documents)}개 회의록 벡터화 완료!")

def semantic_search(query: str, n_results: int = 10) -> List[tuple]:
    """의미 기반 벡터 검색 (코사인 유사도)"""
    if vector_store["embeddings"] is None or len(vector_store["ids"]) == 0:
        return []
    
    # 쿼리 임베딩 생성
    query_embedding = embedding_model.encode([query])[0]
    
    # 코사인 유사도 계산
    embeddings = vector_store["embeddings"]
    # 정규화
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # 유사도 계산
    similarities = np.dot(embeddings_norm, query_norm)
    
    # 상위 N개 인덱스
    top_indices = np.argsort(similarities)[::-1][:n_results]
    
    # 결과 매핑
    search_results = []
    for idx in top_indices:
        meeting_id = vector_store["ids"][idx]
        if meeting_id in store.meetings:
            meeting = store.meetings[meeting_id]
            score = float(similarities[idx]) * 100  # 0~100 점수
            if score > 20:  # 최소 유사도 임계값
                search_results.append((meeting, score, ["semantic"]))
    
    return search_results

# ==================== Data Models ====================
@dataclass
class ActionItem:
    text: str
    completed: bool

@dataclass
class MeetingSection:
    title: str
    items: List[str]

@dataclass
class Meeting:
    id: str
    title: str
    filename: str
    date: str
    folder: str
    project: Optional[str]
    content: str
    summary_content: Optional[str] = None  # _Summary.md 파일의 내용
    sections: List[MeetingSection] = None
    action_items: List[ActionItem] = None
    summary: str = ""  # 리스트 표시용 짧은 요약

@dataclass
class Project:
    name: str
    stage: str
    meetings: List[str]
    last_meeting: str

# ==================== Data Store ====================
class DataStore:
    def __init__(self):
        self.meetings: Dict[str, Meeting] = {}
        self.projects: Dict[str, Project] = {}
    
    def to_dict(self):
        return {
            "meetings": [asdict(m) for m in self.meetings.values()],
            "projects": [asdict(p) for p in self.projects.values()]
        }

store = DataStore()

# ==================== Known Projects ====================
KNOWN_PROJECTS = [
    'Canary', 'PalM', 'EHG', 'Tango', 'IMPACT', 'SNS', 'bluehole',
    '라이징윙스', '언노운', 'Valor', '올리브트리', '딩컴', '소노티카',
    '카나리', '팜', '탱고', '임팩트'
]

# ==================== Parsing Functions ====================
def parse_date_from_filename(filename: str) -> str:
    match = re.search(r'(\d{6})', filename)
    if match:
        yymmdd = match.group(1)
        year = '20' + yymmdd[:2]
        month = yymmdd[2:4]
        day = yymmdd[4:6]
        return f"{year}-{month}-{day}"
    return datetime.now().strftime("%Y-%m-%d")

def extract_project_from_filename(filename: str) -> Optional[str]:
    # Normalize
    filename = filename.replace('카나리', 'Canary').replace('팜', 'PalM')
    
    s2_match = re.search(r'S2_(.+?)\.txt$', filename)
    if s2_match:
        return s2_match.group(1)
    
    mr_match = re.search(r'(\w+)_킥오프|(\w+)_MR', filename)
    if mr_match:
        return mr_match.group(1) or mr_match.group(2)
    
    for proj in KNOWN_PROJECTS:
        if proj.lower() in filename.lower():
            return proj
    return None

def detect_stage(content: str) -> str:
    lower_content = content.lower()
    if any(k in lower_content for k in ['launch', '출시', '런칭']): return 'launch'
    if any(k in lower_content for k in ['beta', '베타']): return 'beta'
    if any(k in lower_content for k in ['alpha', '알파']): return 'alpha'
    if any(k in lower_content for k in ['vertical', '버티컬']): return 'vertical'
    return 'kickoff'

def get_folder_from_path(filepath: Path) -> str:
    parts = filepath.parts
    # MR_Team 하위 폴더 처리 (예: MR_Team/101)
    if 'MR_Team' in parts:
        idx = parts.index('MR_Team')
        if idx + 1 < len(parts):
            return f"MR_Team/{parts[idx+1]}"
        return 'MR_Team'
    
    for part in ['MR_meeting', 'S2_meeting', '비정기회의']:
        if part in parts:
            return part
    return '기타'

# ==================== File Loading ====================
def load_meetings_from_directory():
    store.meetings.clear()
    store.projects.clear()
    
    txt_files = glob.glob(str(PRACTICE_DIR / "**/*.txt"), recursive=True)
    
    for filepath in txt_files:
        path = Path(filepath)
        if path.name.endswith('_Summary.md'): continue
        # requirements.txt 등 회의록이 아닌 파일 제외
        if path.name in ['requirements.txt', 'README.txt']: continue
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 요약 파일 확인 (_Summary.md)
            summary_path = path.with_name(path.stem + "_Summary.md")
            summary_content = None
            if summary_path.exists():
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary_content = f.read()
            
            filename = path.name
            date = parse_date_from_filename(filename)
            folder = get_folder_from_path(path)
            project = extract_project_from_filename(filename)
            
            meeting_id = f"meeting_{hash(str(path)) % 10000000}"
            
            meeting = Meeting(
                id=meeting_id,
                title=filename.replace('.txt', ''),
                filename=filename,
                date=date,
                folder=folder,
                project=project,
                content=content,
                summary_content=summary_content,
                sections=[],
                action_items=[],
                summary=content[:100].strip() + "..."
            )
            
            store.meetings[meeting_id] = meeting
            
            if project:
                if project not in store.projects:
                    store.projects[project] = Project(name=project, stage=detect_stage(content), meetings=[], last_meeting=date)
                store.projects[project].meetings.append(meeting_id)
                if date > store.projects[project].last_meeting:
                    store.projects[project].last_meeting = date
                    
        except Exception as e:
            print(f"Error loading {path}: {e}")

    print(f"Loaded {len(store.meetings)} meetings and {len(store.projects)} projects")

# ==================== API Endpoints ====================

@app.on_event("startup")
async def startup_event():
    load_meetings_from_directory()
    # 벡터 DB 초기화 및 회의록 벡터화
    init_vector_db()
    vectorize_meetings()

@app.get("/")
async def root():
    return FileResponse("milestone_tracker.html")

@app.get("/api/sync")
async def sync_meetings():
    load_meetings_from_directory()
    # 벡터 DB도 재생성
    init_vector_db()
    vectorize_meetings()
    return {"status": "success", "count": len(store.meetings)}

@app.get("/api/meetings")
async def get_meetings():
    return sorted([asdict(m) for m in store.meetings.values()], key=lambda x: x["date"], reverse=True)

@app.get("/api/meetings/{meeting_id}")
async def get_meeting(meeting_id: str):
    if meeting_id not in store.meetings:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return asdict(store.meetings[meeting_id])

@app.get("/api/folders")
async def get_folders():
    folders = defaultdict(int)
    for m in store.meetings.values():
        folders[m.folder] += 1
    return [{"name": k, "count": v} for k, v in sorted(folders.items())]

@app.get("/api/projects")
async def get_projects():
    return sorted([asdict(p) for p in store.projects.values()], key=lambda x: x["last_meeting"], reverse=True)

@app.get("/api/summary/{meeting_id}")
async def get_summary(meeting_id: str):
    if meeting_id not in store.meetings:
        raise HTTPException(status_code=404, detail="Meeting not found")
    m = store.meetings[meeting_id]
    if not m.summary_content:
        return {"exists": False}
    return {"exists": True, "summary": m.summary_content}

def search_meetings(query: str, max_results: int = 10) -> List[tuple]:
    """회의록 검색 - 키워드 매칭 + 관련성 점수 (개선됨 v3)"""
    query_lower = query.lower()
    
    # 한국어 조사 패턴 (단어 끝에서 제거)
    korean_particles = ['에서', '에게', '으로', '이랑', '하고', '라고', '니까', '지만', '는데', '에서는', '에게는',
                        '이', '가', '을', '를', '의', '와', '과', '도', '는', '은', '에', '로', '만', '부터', '까지']
    
    def remove_particles(word):
        """단어 끝의 조사 제거"""
        for particle in sorted(korean_particles, key=len, reverse=True):  # 긴 조사부터 처리
            if word.endswith(particle) and len(word) > len(particle):
                return word[:-len(particle)]
        return word
    
    # 키워드 분리 및 조사 제거
    raw_keywords = [kw.strip() for kw in query_lower.split() if len(kw.strip()) >= 2]
    keywords = []
    for kw in raw_keywords:
        cleaned = remove_particles(kw)
        if len(cleaned) >= 2:
            keywords.append(cleaned)
        # 원본도 추가 (조사 포함된 형태로 검색될 수도 있음)
        if kw not in keywords and len(kw) >= 2:
            keywords.append(kw)
    
    # 중복 제거
    keywords = list(dict.fromkeys(keywords))
    
    # 연속된 단어 조합도 검색어로 추가 (예: "해저 케이블" -> ["해저", "케이블", "해저 케이블"])
    if len(keywords) >= 2:
        for i in range(len(keywords) - 1):
            combined = keywords[i] + " " + keywords[i + 1]
            if combined not in keywords:
                keywords.append(combined)
    
    # 디버깅 로그
    print(f"[검색] 쿼리: '{query}' → 키워드: {keywords}")
    
    scored_meetings = []
    
    for m in store.meetings.values():
        # 검색 대상: 제목 + 폴더명 + 원본 + 요약본
        title_lower = m.title.lower()
        folder_lower = m.folder.lower()
        content_lower = m.content.lower()
        summary_lower = m.summary_content.lower() if m.summary_content else ""
        
        search_text = f"{title_lower} {folder_lower} {content_lower} {summary_lower}"
        
        # 점수 계산
        score = 0
        matched_keywords = []
        
        for kw in keywords:
            if kw in search_text:
                matched_keywords.append(kw)
                
                # 제목에 있으면 최고 가중치
                if kw in title_lower:
                    score += 10
                # 폴더명에 있으면 높은 가중치
                elif kw in folder_lower:
                    score += 8
                # 요약본에 있으면 중간 가중치
                elif kw in summary_lower:
                    score += 5
                # 원본에 있으면 기본 점수
                elif kw in content_lower:
                    score += 2
                
                # 키워드가 여러 번 등장하면 추가 점수
                count = search_text.count(kw)
                if count > 1:
                    score += min(count - 1, 5)  # 최대 5점 추가
        
        # 프로젝트명 매칭 보너스
        if m.project and m.project.lower() in query_lower:
            score += 10
        
        # 모든 키워드가 매칭되면 큰 보너스 (더 정확한 결과)
        if len(keywords) > 1 and all(kw in search_text for kw in keywords):
            score += 20
        
        # 정확한 구문 매칭 보너스 (예: "해저 케이블"이 그대로 있으면)
        if query_lower in search_text:
            score += 15
        
        if score > 0:
            scored_meetings.append((m, score, matched_keywords))
    
    # 점수순 정렬 후 상위 N개
    scored_meetings.sort(key=lambda x: (-x[1], x[0].date), reverse=False)
    
    # 디버깅: 상위 결과 출력
    if scored_meetings:
        print(f"[검색 결과] 상위 {min(5, len(scored_meetings))}개:")
        for m, score, kws in scored_meetings[:5]:
            print(f"  - {m.title} (점수: {score}, 매칭: {kws})")
    else:
        print("[검색 결과] 매칭된 회의록 없음")
    
    return scored_meetings[:max_results]

def search_worklog(query: str, worklog_data: dict, max_results: int = 5) -> List[dict]:
    """업무일지 검색 - 키워드 매칭"""
    if not worklog_data:
        return []
    
    query_lower = query.lower()
    keywords = [kw.strip() for kw in query_lower.split() if len(kw.strip()) >= 2]
    
    scored_worklogs = []
    
    for date_str, day_data in worklog_data.items():
        items = day_data.get('items', [])
        memo = day_data.get('memo', '')
        
        # 검색 대상 텍스트 구성
        items_text = ' '.join([item.get('content', '') for item in items]).lower()
        memo_lower = memo.lower() if memo else ''
        search_text = f"{items_text} {memo_lower}"
        
        if not search_text.strip():
            continue
        
        score = 0
        for kw in keywords:
            if kw in search_text:
                score += search_text.count(kw) * 2
        
        # 정확한 구문 매칭 보너스
        if query_lower in search_text:
            score += 10
        
        if score > 0:
            scored_worklogs.append({
                'date': date_str,
                'items': items,
                'memo': memo,
                'score': score
            })
    
    # 점수순 정렬
    scored_worklogs.sort(key=lambda x: -x['score'])
    return scored_worklogs[:max_results]

@app.post("/api/chat")
async def chat(request: dict):
    query = request.get("query", "")
    worklog_data = request.get("worklog", {})  # 업무일지 데이터 받기
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"answer": "ANTHROPIC_API_KEY가 설정되지 않았습니다.", "sources": []}
    
    # 1. 키워드 검색
    keyword_results = search_meetings(query, max_results=7)
    
    # 2. 의미 기반 벡터 검색
    semantic_results = semantic_search(query, n_results=7)
    print(f"[검색] 키워드: {len(keyword_results)}개, 시맨틱: {len(semantic_results)}개")
    
    # 3. 결과 병합 (중복 제거, 점수 합산)
    combined_scores = {}
    for meeting, score, kws in keyword_results:
        combined_scores[meeting.id] = {
            'meeting': meeting,
            'keyword_score': score,
            'semantic_score': 0,
            'matched': kws
        }
    
    for meeting, score, kws in semantic_results:
        if meeting.id in combined_scores:
            combined_scores[meeting.id]['semantic_score'] = score
        else:
            combined_scores[meeting.id] = {
                'meeting': meeting,
                'keyword_score': 0,
                'semantic_score': score,
                'matched': kws
            }
    
    # 4. 최종 점수 계산 (키워드 60% + 시맨틱 40%)
    search_results = []
    for mid, data in combined_scores.items():
        final_score = data['keyword_score'] * 0.6 + data['semantic_score'] * 0.4
        search_results.append((data['meeting'], final_score, data['matched']))
    
    # 점수순 정렬
    search_results.sort(key=lambda x: -x[1])
    search_results = search_results[:10]  # 상위 10개
    
    # 업무일지 검색
    worklog_results = search_worklog(query, worklog_data, max_results=5)
    
    if not search_results and not worklog_results:
        return {"answer": "관련된 회의록이나 업무일지를 찾지 못했습니다. 다른 키워드로 검색해보세요.", "sources": []}
    
    # 컨텍스트 구성
    context_parts = []
    sources = []
    
    # 회의록 컨텍스트
    for meeting, score, matched_kw in search_results:
        # 요약본이 있으면 요약본 우선, 없으면 원본 일부
        content_to_use = meeting.summary_content if meeting.summary_content else meeting.content[:2000]
        
        context_parts.append(f"""
[문서 유형: 회의록]
[문서 ID: {meeting.id}]
[제목: {meeting.title}]
[날짜: {meeting.date}]
[폴더: {meeting.folder}]
---
{content_to_use}
""")
        sources.append({
            "id": meeting.id, 
            "title": meeting.title, 
            "date": meeting.date,
            "folder": meeting.folder,
            "type": "meeting",
            "relevance": score
        })
    
    # 업무일지 컨텍스트
    for wl in worklog_results:
        items_text = '\n'.join([f"- [{item.get('status', 'pending')}] {item.get('content', '')}" for item in wl['items']])
        memo_text = wl['memo'] if wl['memo'] else '(메모 없음)'
        
        context_parts.append(f"""
[문서 유형: 업무일지]
[날짜: {wl['date']}]
---
📋 업무 항목:
{items_text if items_text else '(업무 항목 없음)'}

📝 메모:
{memo_text}
""")
        sources.append({
            "id": f"worklog_{wl['date']}", 
            "title": f"업무일지 ({wl['date']})", 
            "date": wl['date'],
            "folder": "업무일지",
            "type": "worklog",
            "relevance": wl['score']
        })

    # 조직 배경 정보 포함
    org_info_section = f"""
## 📋 조직 배경 정보 (참고용)
{org_context}
""" if org_context else ""

    system_prompt = f"""당신은 회의록 및 업무일지 분석 전문가 AI 어시스턴트입니다.

주어진 회의록과 업무일지 컨텍스트를 바탕으로 사용자의 질문에 정확하게 답변하세요.
{org_info_section}
**⚠️ 필수 규칙 (반드시 준수):**
1. 반드시 제공된 회의록/업무일지 내용만을 기반으로 답변하세요.
2. **[출처 명시 필수]** 모든 정보에는 반드시 출처를 명시하세요!
   - 회의록: **(출처: [회의 제목], [날짜])**
   - 업무일지: **(출처: 업무일지, [날짜])**
3. **[답변 마지막에 참고 자료 목록 필수]** 답변 끝에 반드시 아래 형식으로 참고한 자료를 나열하세요:
   ---
   📌 **참고 자료:**
   - [회의 제목] (날짜, 폴더)
   - 업무일지 (날짜)
4. 여러 문서에서 정보가 있다면 각각의 출처를 개별적으로 표시하세요.
5. 제공된 컨텍스트에 없는 내용은 "관련 정보가 없습니다"라고 명확히 알려주세요.
6. 답변은 한국어로 작성하세요.
7. 조직 배경 정보를 활용하여 용어, 프로젝트명, 프로세스를 정확히 이해하고 답변하세요."""

    user_prompt = f"""아래는 검색된 회의록과 업무일지입니다:

{"="*50}
{chr(10).join(context_parts)}
{"="*50}

질문: {query}

위 내용을 바탕으로 답변해주세요. 
⚠️ 중요: 모든 정보에 출처를 명시하고, 답변 마지막에 반드시 "📌 참고 자료:" 목록을 포함하세요!"""

    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            response = await client.post(
                ANTHROPIC_API_URL,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": CLAUDE_MODEL,
                    "max_tokens": 2048,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}]
                }
            )
            result = response.json()
            
            # 디버깅용 로그
            print(f"API Response Status: {response.status_code}")
            
            # 에러 체크
            if response.status_code != 200:
                error_msg = result.get('error', {})
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get('message', str(result))
                return {"answer": f"API 오류 ({response.status_code}): {error_msg}", "sources": sources}
            
            if "content" in result and len(result["content"]) > 0:
                answer = result["content"][0]["text"]
            else:
                # 전체 응답 구조 확인
                print(f"Unexpected response: {result}")
                answer = "응답을 처리할 수 없습니다. 다시 시도해주세요."
            
            return {"answer": answer, "sources": sources}
        except Exception as e:
            print(f"Chat API Error: {str(e)}")
            return {"answer": f"오류 발생: {str(e)}", "sources": []}

def find_available_port(preferred_port=8000, max_attempts=10):
    """
    ⚠️ 포트 설정 규칙 (절대 변경 금지):
    - Milestone Tracker: 8000
    - Money Advisor: 8020  
    - Money Manage: 8030
    """
    """사용 가능한 포트를 자동으로 찾습니다."""
    import socket
    
    for offset in range(max_attempts):
        port = preferred_port + offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            print(f"[포트 {port}] 이미 사용 중, 다음 포트 시도...")
            continue
    
    raise RuntimeError(f"사용 가능한 포트를 찾을 수 없습니다 ({preferred_port}-{preferred_port + max_attempts - 1})")


def open_browser_delayed(port, delay=3):
    """서버 시작 후 브라우저에서 localhost로 엽니다."""
    import threading
    import webbrowser
    import time
    
    def open_browser():
        time.sleep(delay)
        webbrowser.open(f"http://localhost:{port}")
    
    thread = threading.Thread(target=open_browser)
    thread.daemon = True
    thread.start()


if __name__ == "__main__":
    port = find_available_port(preferred_port=8000)
    html_path = PRACTICE_DIR / "milestone_tracker.html"
    
    print(f"\n{'='*50}")
    print(f"  Milestone Tracker 서버 시작: http://localhost:{port}")
    print(f"  임베딩 모델 로드 중... (첫 실행 시 약 30초, 이후 빠름)")
    print(f"  이 창을 닫으면 서버가 종료됩니다.")
    print(f"{'='*50}\n")
    
    open_browser_delayed(port)
    uvicorn.run(app, host="0.0.0.0", port=port)
