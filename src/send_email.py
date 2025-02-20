import uuid
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import base64
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from flask import render_template, url_for, session
from google.oauth2 import service_account
import os
import json

load_dotenv()
Email_Key = os.getenv('EMAIL')


def get_new_access_token():
    creds = Credentials(
        None,
        refresh_token=os.getenv('REFRESH_TOKEN'),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET")
    )

    try:
        creds.refresh(Request())
        print("Access token mới đã được tạo!")
        return creds
    except Exception as e:
        print("Lỗi khi tạo access token:", e)
        raise e


def get_service_account_credentials():
    creds = service_account.Credentials.from_service_account_file(
        'clear-storm-373204-847bea4638fd.json',
        scopes=['https://www.googleapis.com/auth/gmail.send']

    )
    return creds


# def send_email(email, username):
#     token = str(uuid.uuid4())  # Sinh token ngẫu nhiên
#
#     verify_url = url_for('verify_email', token=token, _external=True)
#     session['verify_token'] = token
#     session['verify_email'] = email
#     session['verify_username'] = username
#     email_body = render_template('verify_email.html', username=username, verify_url=verify_url)
#
#     msg = MIMEMultipart()
#     msg["From"] = "Breaking Sounds <duc20021118@gmail.com>"
#     msg["To"] = email
#     msg["Subject"] = "Breaking Sounds - Verify Email"
#     msg.attach(MIMEText(email_body, "html"))
#
#     raw_message = base64.urlsafe_b64encode(msg.as_bytes()).decode()
#
#     try:
#         creds = get_new_access_token()
#         service = build("gmail", "v1", credentials=creds)
#
#         service.users().messages().send(userId="me", body={"raw": raw_message}).execute()
#         print("Email sent successfully!")
#         return token
#     except Exception as e:
#         print("Unable to send email:", e)
#         return False

def send_email(email, username):
    token = str(uuid.uuid4())  # Sinh token ngẫu nhiên

    verify_url = url_for('verify_email', token=token, _external=True)
    session['verify_token'] = token
    session['verify_email'] = email
    session['verify_username'] = username
    email_body = render_template('verify_email.html', username=username, verify_url=verify_url)

    msg = MIMEMultipart()
    msg["From"] = "Breaking Sounds <duc20021118@gmail.com>"
    msg["To"] = email
    msg["Subject"] = "Breaking Sounds - Verify Email"
    msg.attach(MIMEText(email_body, "html"))

    raw_message = base64.urlsafe_b64encode(msg.as_bytes()).decode()

    try:
        creds = get_service_account_credentials()
        service = build("gmail", "v1", credentials=creds)

        service.users().messages().send(userId="me", body={"raw": raw_message}).execute()
        print("Email sent successfully!")
        return token
    except Exception as e:
        print("Unable to send email:", e)
        return False


def send_forgot(email, username):
    token = str(uuid.uuid4())  # Sinh token ngẫu nhiên

    verify_url = url_for('forgot_password', token=token, _external=True)
    session['verify_token'] = token
    session['verify_email'] = email
    session['username'] = username

    email_body = render_template('email_forgot_password.html', username=username, verify_url=verify_url)

    msg = MIMEMultipart()
    msg["From"] = "Breaking Sounds <duc20021118@gmail.com>"
    msg["To"] = email
    msg["Subject"] = "Breaking Sounds - Forgot Password"
    msg.attach(MIMEText(email_body, "html"))

    raw_message = base64.urlsafe_b64encode(msg.as_bytes()).decode()

    try:
        creds = get_new_access_token()
        service = build("gmail", "v1", credentials=creds)

        service.users().messages().send(userId="me", body={"raw": raw_message}).execute()
        print("Email sent successfully!")
        return token
    except Exception as e:
        print("Unable to send email:", e)
        return False