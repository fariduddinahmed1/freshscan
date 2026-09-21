"""Gmail SMTP alerts. Explicit params in, bool out — no globals."""
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_alert(sender: str, password: str, recipient: str,
               item_name: str, shelf: str, box: str,
               category: str, shelf_life: str) -> bool:
    try:
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = recipient
        msg["Subject"] = f"FreshScan Alert — {item_name} needs attention!"
        msg.attach(MIMEText(f"""
FreshScan Alert

Item: {item_name}
Location: Shelf {shelf}, Box {box}
Status: {category}
Shelf Life: {shelf_life}
Detected at: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Please take action immediately.

— FreshScan AI System
""", "plain"))
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False
