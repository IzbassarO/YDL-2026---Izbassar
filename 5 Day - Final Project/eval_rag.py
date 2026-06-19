"""
eval_rag.py — системная проверка бота фонда Есенова.

Что проверяет:
  1. IN-SCOPE вопросы (ответ ТОЧНО есть в данных) → ожидаем grounded=True.
  2. OUT-OF-SCOPE вопросы (нет в данных / не про фонд) → ожидаем grounded=False
     (бот честно говорит «не знаю», БЕЗ вызова LLM).
  3. Калибровка порога: печатает top_score обеих групп и предлагает
     RELEVANCE_THRESHOLD ровно между ними.
  4. (опционально) отправку email-саммари администратору.

Запуск:
  python eval_rag.py            # только RAG-проверка (email НЕ шлёт)
  python eval_rag.py --email    # дополнительно реально отправит письмо себе
"""
import sys
import time

import config
from rag import RAG


# Вопросы, ответ на которые ТОЧНО есть в собранных данных сайта фонда.
IN_SCOPE = [
    "Кто такой Шахмардан Есенов?",
    "Чем занимается фонд Есенова?",
    "Что такое Yessenov Data Lab?",
    "Кто входит в попечительский совет фонда?",
    "Как подать заявку?",
    "Где публикуются результаты по заявке?",
]

# Вопросы вне scope фонда — бот ДОЛЖЕН отказаться отвечать.
OUT_OF_SCOPE = [
    "Какая столица Франции?",
    "Напиши функцию сортировки пузырьком на Python.",
    "Какая завтра погода в Алматы?",
    "Сколько стоит последний iPhone?",
    "Кто выиграл чемпионат мира по футболу?",
]


def run_group(rag: RAG, title: str, questions, call_llm: bool):
    """call_llm=False → только retrieve (быстро, без траты токенов на gemma4)."""
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    scores = []
    grounded_flags = []
    for q in questions:
        hits = rag.retrieve(q)
        top = hits[0]["score"] if hits else 0.0
        grounded = bool(hits) and top >= config.RELEVANCE_THRESHOLD
        scores.append(top)
        grounded_flags.append(grounded)
        mark = "🟢 grounded" if grounded else "🔴 refuse  "
        src = hits[0]["source"][:55] if hits else "-"
        print(f"  [{mark}] top={top:.3f}  «{q[:45]}»")
        print(f"             best src: {src}")
        if call_llm and grounded:
            res = rag.answer(q)  # полный ответ через gemma4
            print(f"             ⮑ {res['answer'][:160].replace(chr(10),' ')}")
            time.sleep(0.3)
    return scores, grounded_flags


def main():
    do_email = "--email" in sys.argv
    full = "--full" in sys.argv  # вызывать gemma4 для in-scope (медленнее)

    print("Загружаю индекс…")
    rag = RAG()
    print(f"Чанков в индексе: {rag.index.ntotal}")
    print(f"Текущий RELEVANCE_THRESHOLD = {config.RELEVANCE_THRESHOLD}")

    in_scores, in_flags = run_group(rag, "IN-SCOPE (должны отвечать)",
                                    IN_SCOPE, call_llm=full)
    out_scores, out_flags = run_group(rag, "OUT-OF-SCOPE (должны отказать)",
                                      OUT_OF_SCOPE, call_llm=False)

    # --- Итоги ---
    print(f"\n{'='*70}\nИТОГИ\n{'='*70}")
    in_ok = sum(in_flags)
    out_ok = sum(1 for f in out_flags if not f)
    print(f"IN-SCOPE отвечено:   {in_ok}/{len(IN_SCOPE)}  "
          f"(min top={min(in_scores):.3f}, max={max(in_scores):.3f})")
    print(f"OUT-OF-SCOPE отказ:  {out_ok}/{len(OUT_OF_SCOPE)}  "
          f"(min top={min(out_scores):.3f}, max={max(out_scores):.3f})")

    # --- Рекомендация порога ---
    lo_in = min(in_scores)         # самый слабый «свой» вопрос
    hi_out = max(out_scores)       # самый сильный «чужой» вопрос
    print(f"\nКалибровка порога:")
    print(f"  слабейший IN-SCOPE  top_score = {lo_in:.3f}")
    print(f"  сильнейший OUT      top_score = {hi_out:.3f}")
    if lo_in > hi_out:
        suggested = round((lo_in + hi_out) / 2, 3)
        print(f"  ✅ Группы разделимы. Рекомендуемый RELEVANCE_THRESHOLD ≈ {suggested}")
    else:
        print(f"  ⚠️ Группы ПЕРЕСЕКАЮТСЯ ({lo_in:.3f} <= {hi_out:.3f}) — "
              f"идеального порога нет. Текущий {config.RELEVANCE_THRESHOLD} "
              f"оставит часть ошибок. Смотри какие вопросы конфликтуют выше.")

    # --- Email ---
    if do_email:
        import email_report
        print(f"\n{'='*70}\nEMAIL-ТЕСТ\n{'='*70}")
        demo_history = [
            {"role": "user", "content": "Здравствуйте! Что такое Yessenov Data Lab?"},
            {"role": "assistant", "content": "Это образовательная программа фонда Есенова по data science."},
            {"role": "user", "content": "Хочу подать заявку на следующий поток, оставьте мой запрос."},
        ]
        try:
            summary = rag.summarize_dialog(demo_history)
            print("Саммари:\n ", summary.replace("\n", "\n  "))
            msg_id = email_report.send_summary(summary)
            print(f"\n✅ Письмо отправлено на {config.ADMIN_EMAIL}")
            print(f"   message_id: {msg_id}")
        except Exception as e:
            print(f"\n❌ Ошибка отправки: {type(e).__name__}: {e}")
    else:
        print("\n(email-тест пропущен; добавь флаг --email чтобы реально отправить себе)")


if __name__ == "__main__":
    main()
