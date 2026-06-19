"""
rag.py — ядро retrieval-augmented generation.

Главное правило проекта: бот НЕ выдумывает.
  - Ищем top-k чанков.
  - Если лучший score < RELEVANCE_THRESHOLD → возвращаем "не знаю" БЕЗ вызова LLM.
  - Иначе подаём найденный контекст в gemma4 с жёсткой инструкцией
    отвечать только по контексту.
"""
import json
import numpy as np
import requests
import faiss

import config


SYSTEM_PROMPT = (
    "Ты — официальный консультант фонда Шахмардана Есенова. "
    "Отвечай ТОЛЬКО на основе предоставленного КОНТЕКСТА. "
    "Правила:\n"
    "1. Если в контексте нет ответа — честно скажи: «В материалах фонда я не нашёл "
    "точного ответа на этот вопрос. Уточните, пожалуйста, на сайте "
    "yessenovfoundation.org». НЕ придумывай дедлайны, суммы, требования.\n"
    "2. Никогда не выдумывай цифры, даты и условия, которых нет в контексте.\n"
    "3. Если в контексте есть ссылка, относящаяся к вопросу — приведи её.\n"
    "4. Отвечай кратко, по делу, на языке вопроса."
)


class RAG:
    def __init__(self):
        self.index = faiss.read_index(config.INDEX_PATH)
        with open(config.META_PATH, encoding="utf-8") as f:
            self.meta = json.load(f)

    # --- retrieval ---
    def _embed(self, text: str) -> np.ndarray:
        payload = {"model": config.EMB_MODEL, "input": text}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.EMB_KEY}",
        }
        r = requests.post(config.EMB_URL, json=payload, headers=headers,
                          timeout=config.REQUEST_TIMEOUT)
        r.raise_for_status()
        vec = np.array([r.json()["data"][0]["embedding"]], dtype="float32")
        faiss.normalize_L2(vec)
        return vec

    def retrieve(self, query: str, k: int = None):
        k = k or config.TOP_K
        qv = self._embed(query)
        scores, idx = self.index.search(qv, k)
        hits = []
        for score, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            m = self.meta[i]
            hits.append({"score": float(score), **m})
        return hits

    # --- generation ---
    def _call_gemma(self, messages: list[dict]) -> str:
        payload = {"model": config.CHAT_MODEL, "messages": messages}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.CHAT_KEY}",
        }
        r = requests.post(config.CHAT_URL, json=payload, headers=headers,
                          timeout=config.REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def answer(self, query: str, history: list[dict] = None):
        """
        Возвращает dict:
          {answer, hits, grounded(bool), top_score}
        grounded=False означает "не нашли в данных" — без вызова LLM.
        """
        hits = self.retrieve(query)
        top_score = hits[0]["score"] if hits else 0.0

        if not hits or top_score < config.RELEVANCE_THRESHOLD:
            return {
                "answer": (
                    "В материалах фонда я не нашёл точного ответа на этот вопрос. "
                    "Рекомендую уточнить на сайте yessenovfoundation.org."
                ),
                "hits": hits,
                "grounded": False,
                "top_score": top_score,
            }

        context = "\n\n---\n\n".join(
            f"[Источник: {h['source']}]\n{h['text']}" for h in hits
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history[-4:])  # короткая память диалога
        messages.append({
            "role": "user",
            "content": f"КОНТЕКСТ:\n{context}\n\nВОПРОС: {query}",
        })

        answer = self._call_gemma(messages)
        return {
            "answer": answer,
            "hits": hits,
            "grounded": True,
            "top_score": top_score,
        }

    def summarize_dialog(self, history: list[dict]) -> str:
        """Краткое саммари диалога для email администратору."""
        convo = "\n".join(
            f"{'Пользователь' if m['role']=='user' else 'Бот'}: {m['content']}"
            for m in history
        )
        messages = [
            {"role": "system", "content":
                "Сделай краткое деловое саммари диалога пользователя с ботом фонда "
                "(3-5 предложений): что спрашивал пользователь, что важного выяснилось, "
                "есть ли запрос/заявка. Без воды."},
            {"role": "user", "content": convo},
        ]
        return self._call_gemma(messages)
