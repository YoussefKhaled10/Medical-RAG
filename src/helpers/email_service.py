import asyncio, logging, secrets, smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from src.helpers.config import settings
log=logging.getLogger(__name__)
def generate_otp_code(length=6):
 if not 4<=length<=10: raise ValueError("OTP length must be 4-10")
 return "".join(str(secrets.randbelow(10)) for _ in range(length))
def _send(to,code):
 if not settings.SMTP_USERNAME or not settings.SMTP_PASSWORD: raise RuntimeError("SMTP credentials missing")
 msg=MIMEMultipart("alternative"); msg["Subject"]="RecoveryPath AI verification code"; msg["From"]=formataddr(("RecoveryPath AI",settings.SMTP_USERNAME)); msg["To"]=to
 msg.attach(MIMEText(f"Your code is {code}. It expires in 15 minutes.","plain","utf-8"))
 msg.attach(MIMEText(f'<div style="background:#071a31;color:white;padding:32px;border-radius:20px;font-family:Arial"><b style="color:#42dfd2">RECOVERYPATH AI</b><h2>Email verification</h2><div style="font-size:38px;letter-spacing:9px;background:#0b3553;padding:20px;text-align:center">{code}</div><p>This code expires in 15 minutes.</p></div>',"html","utf-8"))
 client=smtplib.SMTP_SSL(settings.SMTP_SERVER,settings.SMTP_PORT,timeout=20) if settings.SMTP_PORT==465 else smtplib.SMTP(settings.SMTP_SERVER,settings.SMTP_PORT,timeout=20)
 try:
  if settings.SMTP_PORT!=465: client.starttls()
  client.login(settings.SMTP_USERNAME,settings.SMTP_PASSWORD); client.send_message(msg)
 finally: client.quit()
async def send_verification_email(to_email,code):
 try: await asyncio.to_thread(_send,to_email,code); return True
 except Exception: log.exception("OTP email failed"); return False
