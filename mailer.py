# mailer.py
import os
import smtplib
from dataclasses import dataclass
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

@dataclass(frozen=True)
class MailConfig:
    sender:  "antoannali@gmail.com"
    password: "szld tsrz wwcg nqvs"   # password per app Gmail
    recipient:  "antoannali@gmail.com"

def load_config() -> MailConfig:
    sender = os.getenv("GMAIL_SENDER", "")
    password = os.getenv("GMAIL_PASSWORD", "")
    recipient = os.getenv("GMAIL_RECIPIENT", "")

    if not sender or not password or not recipient:
        missing = [k for k, v in {
            "GMAIL_SENDER": sender,
            "GMAIL_PASSWORD": password,
            "GMAIL_RECIPIENT": recipient
        }.items() if not v]
        raise RuntimeError(f"Config email mancante: {', '.join(missing)}")

    return MailConfig(sender=sender, password=password, recipient=recipient)

def send_email(subject: str, body: str = "", is_html: bool = False, config: MailConfig | None = None) -> bool:
    cfg = config or load_config()

    payload = body if body.strip() else " "  # body vuoto ok

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = cfg.sender
    msg["To"] = cfg.recipient

    subtype = "html" if is_html else "plain"
    msg.attach(MIMEText(payload, subtype, "utf-8"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as smtp:
            smtp.login(cfg.sender, cfg.password)
            smtp.send_message(msg)
        return True
    except Exception as e:
        print(f"Errore invio email: {e}")
        return False
