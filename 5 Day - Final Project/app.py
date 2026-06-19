"""
app.py — Streamlit-чат по грантам фонда Есенова (RAG на gemma4).

Главное: бот отвечает ТОЛЬКО по данным с сайта фонда. Если в индексе нет
релевантного — честно говорит «не знаю» (см. rag.py, RELEVANCE_THRESHOLD).

Запуск:  streamlit run app.py
"""
import time

import streamlit as st

import config
from rag import RAG
import email_report


st.set_page_config(page_title="Грант-бот фонда Есенова", page_icon="🎓")


@st.cache_resource(show_spinner="Загружаю индекс фонда…")
def get_rag():
    return RAG()


def init_state():
    ss = st.session_state
    ss.setdefault("messages", [])        # [{role, content, meta?}]
    ss.setdefault("user_msg_count", 0)   # сообщений пользователя за сессию
    ss.setdefault("last_request_ts", 0.0)
    ss.setdefault("email_status", None)


def render_sources(meta: dict):
    """Раскрывающийся блок «Источники» — ключевой элемент демо (честность)."""
    hits = meta.get("hits", [])
    if not hits:
        return
    with st.expander(f"📎 Источники ({len(hits)}) · top score {meta.get('top_score', 0):.3f}"):
        for h in hits:
            src = h.get("source", "")
            title = h.get("title") or src
            score = h.get("score", 0.0)
            snippet = (h.get("text", "")[:300] + "…") if h.get("text") else ""
            if src.startswith("http"):
                st.markdown(f"**[{title}]({src})** · `score {score:.3f}`")
            else:
                st.markdown(f"**{title}** · `{src}` · `score {score:.3f}`")
            st.caption(snippet)


def render_message(m: dict):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        meta = m.get("meta")
        if meta:
            if meta.get("grounded"):
                st.caption("✅ Ответ на основе материалов фонда")
            else:
                st.caption("⚠️ Не найдено в материалах фонда — бот не выдумывает")
            render_sources(meta)


# ---------------------------------------------------------------- UI
init_state()

st.title("🎓 Грант-бот фонда Шахмардана Есенова")
st.caption(
    "Отвечаю на вопросы о грантах, программах и деятельности фонда — "
    "**только** по данным сайта yessenovfoundation.org. "
    "Чего не знаю — честно скажу «не знаю»."
)

# --- Сайдбар: лимиты + email ---
with st.sidebar:
    st.header("О боте")
    st.markdown(
        "- Модель: **gemma4**\n"
        "- Поиск: **RAG / FAISS** по сайту фонда\n"
        "- Если данных нет — отвечаю «не знаю»"
    )
    used = st.session_state.user_msg_count
    st.progress(min(used / config.MAX_MESSAGES_PER_SESSION, 1.0),
                text=f"Сообщений: {used}/{config.MAX_MESSAGES_PER_SESSION}")

    st.divider()
    st.subheader("📧 Отчёт администратору")
    st.caption(f"Саммари диалога уйдёт на {config.ADMIN_EMAIL}")
    if st.button("Отправить саммари администратору", use_container_width=True,
                 disabled=not st.session_state.messages):
        try:
            rag = get_rag()
            with st.spinner("Готовлю саммари и отправляю…"):
                summary = rag.summarize_dialog(st.session_state.messages)
                msg_id = email_report.send_summary(summary)
            st.session_state.email_status = ("ok", msg_id, summary)
        except Exception as e:
            st.session_state.email_status = ("err", str(e), None)

    es = st.session_state.email_status
    if es:
        if es[0] == "ok":
            st.success(f"Отправлено ✅ message_id: {es[1]}")
            with st.expander("Текст саммари"):
                st.write(es[2])
        else:
            st.error(f"Ошибка отправки: {es[1]}")

# --- История чата ---
for m in st.session_state.messages:
    render_message(m)

# --- Ввод с лимитами ---
limit_reached = st.session_state.user_msg_count >= config.MAX_MESSAGES_PER_SESSION
prompt = st.chat_input(
    "Спросите про гранты, стипендии, программы фонда…",
    disabled=limit_reached,
)
if limit_reached:
    st.warning(
        f"Достигнут лимит сообщений за сессию "
        f"({config.MAX_MESSAGES_PER_SESSION}). Перезапустите сессию."
    )

if prompt:
    # анти-спам пауза
    now = time.time()
    wait = config.MIN_SECONDS_BETWEEN - (now - st.session_state.last_request_ts)
    if wait > 0:
        st.toast(f"Слишком часто — подождите {wait:.1f} с")
    else:
        st.session_state.last_request_ts = now
        st.session_state.user_msg_count += 1
        st.session_state.messages.append({"role": "user", "content": prompt})
        render_message({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Ищу в материалах фонда…"):
                try:
                    rag = get_rag()
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]
                    result = rag.answer(prompt, history=history)
                except Exception as e:
                    err = (
                        "⚠️ Не удалось получить ответ (проблема сети или сервиса "
                        f"alem.ai). Попробуйте ещё раз.\n\n`{type(e).__name__}: {e}`"
                    )
                    st.markdown(err)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err}
                    )
                    st.stop()

            st.markdown(result["answer"])
            if result["grounded"]:
                st.caption("✅ Ответ на основе материалов фонда")
            else:
                st.caption("⚠️ Не найдено в материалах фонда — бот не выдумывает")
            render_sources(result)

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "meta": {
                "grounded": result["grounded"],
                "top_score": result["top_score"],
                "hits": result["hits"],
            },
        })
