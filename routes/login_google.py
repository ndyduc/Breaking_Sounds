from flask import *
from google.oauth2 import id_token
from google.auth.transport.requests import Request
from src.data_connecter import *
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.exceptions import GoogleAuthError
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import id_token as google_id_token
import requests.exceptions
from dotenv import load_dotenv

import os
import logging
google = Blueprint('google', __name__, url_prefix='/')


load_dotenv()

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

# Cấu hình Flow
flow = Flow.from_client_config(
    client_config={
        "web": {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    },
    scopes=[
        "openid",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/userinfo.email"
    ],
    redirect_uri="https://127.0.0.1:3202/login/callback",
)


@google.route('/login_gg')
def login():
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true'
    )
    session['state'] = state
    return redirect(authorization_url)


@google.route('/login/callback')
def callback():
    try:
        print("CLIENT_ID:", repr(GOOGLE_CLIENT_ID))
        print("CLIENT_SECRET:", repr(GOOGLE_CLIENT_SECRET))
        flow.fetch_token(authorization_response=request.url)
        if not session['state'] == request.args['state']:
            return 'State mismatch', 400

        credentials = flow.credentials
        id_info = id_token.verify_oauth2_token(
            credentials.id_token,
            Request(),
            GOOGLE_CLIENT_ID
        )
        # print(id_info)
        # Lưu thông tin người dùng vào session
        from main import decode_img
        if 'user_id' in session:
            if 'picture' in session:
                user = update_user(session.get('user_id'), id_info.get('sub'))
            else:
                user = update_user(session.get('user_id'), id_info.get('sub'), None, None, None, id_info.get('picture'))
                new_info = get_user(None, session.get('email'))
                session['picture'] = decode_img(new_info['Avatar_url'])
        else:
            session['id'] = id_info['sub']
            session['email'] = id_info.get('email')

            if check_exist_user(session['email']):
                user = get_user(None, session['email'])
                session['user_id'] = str(user['_id'])
                session['name'] = user['Username']

                avatar_url = user.get('Avatar_url')
                if isinstance(avatar_url, str) and avatar_url.startswith("http"):
                    session['picture'] = avatar_url  # Nếu là link, giữ nguyên
                else:
                    session['picture'] = decode_img(avatar_url)  # Nếu là binary, decode
            else:
                user = insert_user(id_info.get('sub'), session.get('email'), id_info.get('email'), None, id_info.get('picture'))
                session['user_id'] = user
                usernew = get_user(None, session['email'])
                session['name'] = usernew['Username']
                session['picture'] = usernew['Avatar_url']
                flash("Wellcome to Breaking Sounds !")

                return redirect(url_for('index'))

        session['google_id'] = id_info.get('sub')
        return redirect(url_for('index'))

    except requests.exceptions.HTTPError as http_err:
        print(f"[HTTPError] Lỗi khi lấy token: {http_err}")
        return "Lỗi kết nối đến Google", 400

    except GoogleAuthError as auth_err:
        print(f"[GoogleAuthError] Lỗi xác thực: {auth_err}")
        return "Lỗi xác thực Google OAuth2", 400

    except ValueError as val_err:
        print(f"[ValueError] Lỗi khi kiểm tra token ID: {val_err}")
        return "Token không hợp lệ", 400

    except Exception as e:
        print(f"[UnknownError] Lỗi không xác định khi lấy token: {e}")
        return "Lỗi xác thực OAuth2", 400


@google.route('/profile')
def profile():
    if 'google_id' not in session:
        flash("Error when logging in with Google !")
        return redirect(url_for('loginbase'))

    return f'''
        <h1>Profile</h1>
        <p>ID : {session['id']} </p>
        <p>Name: {session['name']}</p>
        <p>Email: {session['email']}</p>
        <p><img src="{session['picture']}" alt="Profile Picture" width="100"></p>
        <a href="/logout">Logout</a>
    '''


@google.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))
