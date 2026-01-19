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

@app.get("/")
async def root():
    return FileResponse("milestone_tracker.html")

@app.get("/api/sync")
async def sync_meetings():
    load_meetings_from_directory()
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
    """회의록 검색 - 키워드 매칭 + 관련성 점수 (개선됨)"""
    query_lower = query.lower()
    
    # 키워드 분리 (1글자도 포함, 한글 검색 개선)
    keywords = [kw.strip() for kw in query_lower.split() if len(kw.strip()) >= 1]
    
    # 연속된 단어 조합도 검색어로 추가 (예: "해저 케이블" -> ["해저", "케이블", "해저 케이블"])
    if len(keywords) >= 2:
        for i in range(len(keywords) - 1):
            combined = keywords[i] + " " + keywords[i + 1]
            if combined not in keywords:
                keywords.append(combined)
        # 전체 쿼리도 추가
        if query_lower not in keywords:
            keywords.append(query_lower)
    
    scored_meetings = []
    
    for m in store.meetings.values():
        # 검색 대상: 제목 + 폴더명 + 원본 + 요약본 (폴더명 추가!)
        search_text = f"{m.title} {m.folder} {m.content}".lower()
        if m.summary_content:
            search_text += " " + m.summary_content.lower()
        
        # 점수 계산
        score = 0
        matched_keywords = []
        
        for kw in keywords:
            if kw in search_text:
                # 제목에 있으면 가중치 높음
                if kw in m.title.lower():
                    score += 5
                # 폴더명에 있으면 가중치
                elif kw in m.folder.lower():
                    score += 4
                # 요약본에 있으면 가중치
                elif m.summary_content and kw in m.summary_content.lower():
                    score += 3
                # 원본에 있으면 기본 점수
                else:
                    score += 1
                matched_keywords.append(kw)
        
        # 프로젝트명 매칭 보너스
        if m.project and m.project.lower() in query_lower:
            score += 5
        
        # 모든 키워드가 매칭되면 보너스 (더 정확한 결과)
        base_keywords = [kw.strip() for kw in query_lower.split() if len(kw.strip()) >= 1]
        if len(base_keywords) > 1 and all(kw in search_text for kw in base_keywords):
            score += 10
        
        if score > 0:
            scored_meetings.append((m, score, matched_keywords))
    
    # 점수순 정렬 후 상위 N개
    scored_meetings.sort(key=lambda x: (-x[1], x[0].date), reverse=False)
    return scored_meetings[:max_results]

@app.post("/api/chat")
async def chat(request: dict):
    query = request.get("query", "")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"answer": "ANTHROPIC_API_KEY가 설정되지 않았습니다.", "sources": []}
    
    # 관련 회의록 검색
    search_results = search_meetings(query, max_results=5)
    
    if not search_results:
        return {"answer": "관련된 회의록을 찾지 못했습니다. 다른 키워드로 검색해보세요.", "sources": []}
    
    # 컨텍스트 구성
    context_parts = []
    sources = []
    
    for meeting, score, matched_kw in search_results:
        # 요약본이 있으면 요약본 우선, 없으면 원본 일부
        content_to_use = meeting.summary_content if meeting.summary_content else meeting.content[:2000]
        
        context_parts.append(f"""
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
            "relevance": score
        })

    system_prompt = """당신은 회의록 분석 전문가 AI 어시스턴트입니다.

주어진 회의록 컨텍스트를 바탕으로 사용자의 질문에 정확하게 답변하세요.

**중요 규칙:**
1. 반드시 제공된 회의록 내용만을 기반으로 답변하세요.
2. 답변 시 해당 정보의 출처를 반드시 명시하세요. 예: "(출처: 251217 S2_PalM, 2025-12-17)"
3. 여러 회의록에서 관련 정보가 있다면 통합하여 답변하고, 각각의 출처를 표시하세요.
4. 제공된 컨텍스트에 없는 내용은 "관련 정보가 회의록에 없습니다"라고 명확히 알려주세요.
5. 답변은 한국어로 작성하세요."""

    user_prompt = f"""아래는 검색된 회의록입니다:

{"="*50}
{chr(10).join(context_parts)}
{"="*50}

질문: {query}

위 회의록 내용을 바탕으로 답변해주세요. 답변 시 반드시 출처(회의 제목, 날짜)를 명시해주세요."""

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
