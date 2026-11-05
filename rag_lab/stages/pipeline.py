"""사이드바 '▶ 파이프라인 실행': 1~4단계를 차례로 돌리며 진행 상황을 보여 준다.

단계 함수는 앱마다 다르므로(stages · solutions · *_lab) 인자로 받는다.
임베딩은 조각을 묶음(EMBED_BATCH)으로 나눠 넣어 진행률 막대를 채운다.
"""
from typing import Callable

import numpy as np
import streamlit as st

EMBED_BATCH = 16          # 진행률 막대를 한 번 채울 때 임베딩하는 조각 수


def run_pipeline(files, include_folder: bool, model_name: str, size: int, overlap: int,
                 load_documents: Callable, make_chunks: Callable, get_model: Callable,
                 embed: Callable, build_index: Callable, kb: dict | None = None):
    """성공하면 kb(dict), 문서가 없거나 추출 실패면 None. 단계 함수의 예외는 그대로 올린다.

    kb를 넘기면 단계가 끝날 때마다 그 안에 결과를 채운다 → 중간 단계에서 멈춰도 끝낸 단계의 탭은 보인다(실습 앱).
    """
    kb = {} if kb is None else kb
    with st.status("파이프라인 실행 중…", expanded=True) as status:
        try:
            line = st.empty()                 # 단계마다 한 줄: '…중' → '✅ 결과'로 바꿔 쓴다
            line.write("1/4 · 문서 적재 · 파싱 중…")
            docs = kb["docs"] = load_documents(files, include_folder)
            line.write(f"✅ 1/4 · 문서 {len(docs)}개 적재")

            line = st.empty()
            line.write("2/4 · 청킹 중…")
            records = kb["records"] = make_chunks(docs, size, overlap)
            kb["size"], kb["overlap"] = size, overlap
            line.write(f"✅ 2/4 · 청크 {len(records)}개")
            if not records:
                status.update(label="문서가 없거나 텍스트 추출 실패", state="error")
                return None

            line = st.empty()
            line.write("3/4 · 임베딩 모델 불러오는 중… (처음엔 모델을 내려받느라 오래 걸립니다)")
            model = get_model(model_name)
            kb["model_name"] = model_name
            line.write(f"3/4 · 조각 {len(records)}개 임베딩 중…")
            bar = st.progress(0.0, text=f"0 / {len(records)}")
            parts = []
            for i in range(0, len(records), EMBED_BATCH):
                parts.append(np.asarray(embed(records[i:i + EMBED_BATCH], model), dtype="float32"))
                done = min(i + EMBED_BATCH, len(records))
                bar.progress(done / len(records), text=f"{done} / {len(records)}")
            emb = kb["emb"] = np.vstack(parts)
            bar.empty()
            line.write(f"✅ 3/4 · 임베딩 {emb.shape[0]} × {emb.shape[1]}")

            line = st.empty()
            line.write("4/4 · 인덱스 저장 중…")
            kb["index"] = build_index(emb)
            line.write(f"✅ 4/4 · 인덱스에 벡터 {emb.shape[0]}개 저장")
        except Exception:
            status.update(label="파이프라인 실행 중 오류", state="error")
            raise
        # '완료 · 문서 N · 청크 N' 문구는 앱의 st.success가 보인다(실습지 · 캡처가 이 문구를 찾는다)
        status.update(label="파이프라인 실행 끝 (단계별 기록 펼치기)", state="complete", expanded=False)
    return kb
