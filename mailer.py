# mailer.py
"""
Modulo riusabile per inviare email via Gmail SMTP.

Obiettivo:
- NON scrivere password nel codice
- Leggere config da variabili d'ambiente / GitHub Secrets
- Esporre una funzione semplice: send_email(...)

Variabili richieste:
- GMAIL_SENDER    (es: antoannali@gmail.com)
- GMAIL_PASSWORD  (password per app Gmail, 16 caratteri)
- GMAIL_RECIPIENT (destinatario)
"""

import os
import smtplib
from dataclasses import dataclass
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


@dataclass(frozen=True)
class MailConfig:
    """
    Config email.
    frozen=True = immutabile (evita modifiche accidentali).
    """
    sender: str
    password: str   # Password per app Gmail (NON password normale)
    recipient: str


def load_config() -> MailConfig:
    """
    Legge la configurazione dalle variabili d'ambiente.

    Perché così:
    - in locale: export/setx delle env
    - in GitHub Actions: secrets → env nel workflow
    - nessuna password nel repository
    """

    sender = os.getenv("GMAIL_SENDER", "")
    password = os.getenv("GMAIL_PASSWORD", "")
    recipient = os.getenv("GMAIL_RECIPIENT", "")

    # Se manca qualcosa, fallisco subito con errore chiaro
    if not sender or not password or not recipient:
        missing = [
            k for k, v in {
                "GMAIL_SENDER": sender,
                "GMAIL_PASSWORD": password,
                "GMAIL_RECIPIENT": recipient,
            }.items()
            if not v
        ]
        raise RuntimeError(f"Config email mancante: {', '.join(missing)}")

    return MailConfig(sender=sender, password=password, recipient=recipient)


def send_email(
    subject: str,
    body: str = "",
    is_html: bool = False,
    config: MailConfig | None = None
) -> bool:
    """
    Invia una mail usando Gmail SMTP (SSL).

    Parametri:
    - subject: oggetto mail
    - body: testo o html (può essere vuoto)
    - is_html: True se body è HTML
    - config: opzionale, se vuoi passare una config custom
             (altrimenti usa load_config() da env)

    Ritorna:
    - True se invio ok
    - False se errore (stampa il motivo a console)
    """

    # Se config non viene passata, la carico dalle env
    cfg = config or load_config()

    # Alcuni client/email server odiano body completamente vuoto:
    # metto uno spazio, così passa sempre.
    payload = body if body.strip() else " "

    # MIME "alternative" permette di avere versioni diverse (plain/html)
    # Qui ne mettiamo solo una, ma resta flessibile.
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = cfg.sender
    msg["To"] = cfg.recipient

    # Se is_html=True → Content-Type text/html, altrimenti text/plain
    subtype = "html" if is_html else "plain"
    msg.attach(MIMEText(payload, subtype, "utf-8"))

    try:
        # Gmail SMTP SSL:
        # - host: smtp.gmail.com
        # - port: 465
        # Timeout per evitare che resti appeso se qualcosa non va
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as smtp:
            # Login con password per app Gmail
            smtp.login(cfg.sender, cfg.password)

            # Invio messaggio
            smtp.send_message(msg)

        return True

    except Exception as e:
        # Log semplice: se vuoi, qui puoi aggiungere logging strutturato
        print(f"Errore invio email: {e}")
        return False
