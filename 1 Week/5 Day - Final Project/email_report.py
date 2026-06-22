"""
email_report.py — отправка саммари диалога администратору.

ВАЖНО (правила задания):
  - Шлём ТОЛЬКО на свой адрес (config.ADMIN_EMAIL = izok2004@gmail.com).
  - ТОЛЬКО по явному действию (кнопка в UI), НИКОГДА в цикле.
  Иначе домен info@app.commit.kz попадёт в спам у Gmail.
"""
from mailersend import MailerSendClient, EmailBuilder

import config


def send_summary(summary_text: str, summary_html: str = None) -> str:
    """Шлёт саммари на ADMIN_EMAIL. Возвращает message_id (или текст ошибки)."""
    if summary_html is None:
        summary_html = f"<h2>Саммари разговора</h2><p>{summary_text}</p>"

    ms = MailerSendClient(api_key=config.MAILERSEND_KEY)
    email = (
        EmailBuilder()
        .from_email(config.FROM_EMAIL, config.FROM_NAME)
        .to_many([{"email": config.ADMIN_EMAIL, "name": "Admin"}])
        .subject("Новая заявка / запрос из чата фонда")
        .html(summary_html)
        .text(f"Саммари разговора:\n\n{summary_text}")
        .build()
    )
    response = ms.emails.send(email)
    return getattr(response, "message_id", str(response))
