# RAG 파이프라인 실습 (학생 배포본)

8단계 RAG 파이프라인을 직접 구현하며 눈으로 확인하는 실습입니다.

## 폴더 구성

| 항목 | 내용 | 접근 |
|------|------|------|
| `rag_lab/` | 실습 워크북 — `stages/*_lab.py` 의 ✏️TODO 를 채웁니다 | 🟢 공개 |
| `rag_answer.zip` | 각 TODO의 **정답(기본 동작 코드)** | 🟢 공개 |
| `rag_solution.zip` | 확장 과제 **모범답안** | 🟢 공개 |

> zip은 암호 없이 바로 풀립니다. 먼저 직접 채워 본 뒤 비교용으로 여세요.

## 시작하기 (rag_lab)

```bash
cd rag_lab
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app_lab.py
```

브라우저가 열리면 사이드바에서 문서를 넣고 **▶ 파이프라인 실행**.
각 탭(1.적재 … 8.평가)은 해당 `stages/stageN_*_lab.py` 의 TODO를 채우면 결과가 나타납니다.
아직 구현 전 단계는 "✏️ 구현이 필요합니다" 안내만 뜨고 앱은 죽지 않습니다.

### 구현 순서 (의존성)
1. `stage1_load_lab.py` — `read_pdf`, `load_documents`
2. `stage2_chunk_lab.py` — `make_chunks`
3. `stage3_embed_lab.py` — `get_model`, `embed`
4. `stage4_store_lab.py` — `build_index`
5. `stage5_search_lab.py` — `retrieve`
6. `stage6_generate_lab.py` — `build_prompt`, `generate`
7. `stage7_chat_lab.py` / 8. `stage8_eval_lab.py`

### LLM 키 (선택)

`rag_lab/.streamlit/secrets.toml.example` 를 같은 폴더의 `secrets.toml` 로 복사하고 키를 채웁니다.
생성 단계는 **Claude → Gemini → 로컬 Ollama** 순으로 자동 선택합니다.

| 키 | 발급처 | 모델 |
|---|---|---|
| `CLAUDE_API_KEY` | console.anthropic.com | `claude-haiku-4-5` |
| `GEMINI_API_KEY` | aistudio.google.com | `gemini-3.6-flash` |

둘 다 없으면 로컬 Ollama(`ollama serve` + `ollama pull qwen2.5:14b`)를 쓰고,
그마저 없으면 검색 근거만 표시되므로 1~5단계 실습은 그대로 진행됩니다.
6.생성 탭의 **LLM 제공자** 라디오에서 직접 고를 수도 있습니다.

### 평가용 테스트표

8.평가/검수 탭의 드롭다운에서 고릅니다.

| 선택 | 내용 |
|---|---|
| 부대 행정 | `eval_questions.csv` — 규정·수치 위주 10문항(함정 2) |
| 소설 「운수 좋은 날」 | `eval_questions_novel.csv` — 인물·줄거리 10문항(함정 2) |
| 통합 (행정 + 소설) | 20문항. 성격별로 나눈 **구분별 지표**가 함께 표시됩니다 |

## 단계 체크포인트 브랜치

앞 단계를 다 못 끝냈어도 다음 단계를 할 수 있도록, 단계마다 브랜치를 둡니다.
`stage-N` 브랜치 = **1~(N-1)단계는 채워져 있고 N단계가 이번 차례**인 상태입니다. 각 브랜치의 `STEP.md`에 설명이 있습니다.

| 브랜치 | 채워진 단계 | 이번 차례 |
|---|---|---|
| `main` · `stage-1-load` | — | 1 적재/파싱 |
| `stage-2-chunk` | 1 | 2 청킹 |
| `stage-3-embed` | 1~2 | 3 임베딩 |
| `stage-4-store` | 1~3 | 4 저장 |
| `stage-5-search` | 1~4 | 5 검색 |
| `stage-6-generate` | 1~5 | 6 생성 |
| `stage-7-chat` | 1~6 | 7 챗봇 |
| `stage-8-eval` | 1~7 | 8 평가/검수 |

```bash
git stash -u                 # 내가 고친 코드를 잠시 보관 (안 고쳤으면 생략 가능)
git switch stage-4-store     # 원하는 체크포인트로 이동 (clone 때 모든 브랜치를 함께 받음)
git pull                     # 강사가 수정본을 올렸다고 하면, 지금 브랜치의 최신본 받기
```

`git clone` 한 번이면 모든 브랜치가 함께 내려옵니다. 브랜치를 옮긴 뒤에는 앱 사이드바의 **🗑️ 상태 초기화** → 1번 탭부터 **이 단계 실행**을 차례로 누릅니다.

## 정답 / 모범답안 열기

```bash
unzip rag_answer.zip      # → rag_answer/   (cd rag_answer && streamlit run app.py)
unzip rag_solution.zip    # → rag_solution/ (cd rag_solution && streamlit run app_solution.py)
```
