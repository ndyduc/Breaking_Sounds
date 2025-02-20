from flask import *
from google_auth_oauthlib.flow import Flow
from routes.login_google import google
from src.data_connecter import *
from src.send_email import send_email, send_forgot
import os

app = Flask(__name__)
app.secret_key = os.getenv('MY_KEY')
app.config["SESSION_TYPE"] = "filesystem"
app.register_blueprint(google)


@app.route('/')
def index():
    # Check_connect()
    return render_template('Home.html')


@app.route('/loginbase', methods=['GET', 'POST'])
def login_base():
    return render_template('Login_page.html')


@app.route('/signup', methods=['POST'])
def signup():
    session.pop('_flashes', None)
    email = request.form.get('email')
    username = request.form.get('username')
    password = request.form.get('password')

    if check_exist_user(email):
        flash(f"This email : " +email+" is already used ! <br> If this is your email you can take it back.", "error")
        return redirect(url_for('login_base'))

    session['password'] = password

    if send_email(email, username):
        flash("Email was sent successfully !")
        return redirect(url_for('login_base'))
    else:
        flash("Error when sending email !")
        return redirect(url_for('login_base'))


@app.route('/loggin', methods=['GET', 'POST'])
def loggin():
    session.pop('_flashes', None)
    email = request.form.get('email')
    password = request.form.get('password')


    session['email'] = email

    action = request.form.get("action")
    if action == "login":
        if not password:
            flash("Please enter your password !")
            return redirect(url_for('login_base'))
        if verify_password(email, password):
            return redirect(url_for('index'))
        else:
            flash("Password was in correct !")
            return redirect(url_for('login_base'))
    elif action == "reset":
        user = get_user(None, email)
        if user is None:
            flash("User not found !")
            session.clear()
            return redirect(url_for('login_base'))
        else:
            return redirect(url_for('forgot'))
    else:
        return redirect(url_for('/'))


@app.route('/verify/<token>')
def verify_email(token):
    saved_token = session.get('verify_token')
    saved_email = session.get('verify_email')
    saved_username = session.get('verify_username')

    if saved_token is None or saved_token != token:
        return "<h2>Verify have been expire or invalid!</h2>"

    id = insert_user(None, saved_username, saved_email, session.get('password'))
    if id is not None:
        session['user_id'] = id
        print(session['id'])
    else:
        return redirect(url_for('login_base'))
    # Xác minh thành công → Xóa session để tránh dùng lại
    session.pop('verify_token', None)
    session.pop('verify_email', None)
    session.pop('verify_username', None)

    return render_template('sidebar.html', username=saved_username, email=saved_email)


@app.route('/forgot')
def forgot():
    email = session.get('email')
    user = get_user(None, email)
    print(user.get('Username'))

    if send_forgot(email, user.get('Username')):
        flash("Email was sent successfully !")
        return redirect(url_for('login_base'))
    else:
        flash("Error when sending email !")
        return redirect(url_for('login_base'))


@app.route('/forgot/<token>')
def forgot_password(token):
    saved_token = session.get('verify_token')

    if saved_token is None or saved_token != token:
        return "<h2>Verify have been expire!</h2>"

    # Xác minh thành công → Xóa session để tránh dùng lại
    session.pop('verify_token', None)
    session.pop('verify_email', None)
    session.pop('username', None)

    return redirect(url_for('change_password'))


@app.route('/change_pass', methods=['POST'])
def change_pass():
    session.pop('_flashes', None)
    pass1 = request.form.get('pass1')
    pass2 = request.form.get('pass2')

    if pass1 != pass2:
        return jsonify({"success": False, "message": "Passwords do not match!"})

    email = session.get('email')
    if not email:
        return jsonify({"success": False, "message": "User not authenticated!"})

    user = get_user(None, email)
    if not user:
        return jsonify({"success": False, "message": "User not found!"})

    session['user_id'] = str(user.get('_id'))
    user_up = update_user(user.get('_id'), None, None, pass1, None, None, None, None, None, None)

    if user_up == {"success": "User updated successfully"}:
        return jsonify({
            "success": True,
            "message": "Password updated successfully!",
            "redirect": url_for('index')
        })

    return jsonify({"success": False, "message": "Error updating password!"})


@app.route('/change_password', methods=['GET', 'POST'])
def change_password():
    return render_template('change_password.html')


# @app.route('/authorize')
def authorize():
    flow = Flow.from_client_secrets_file(
        'src/client_secret.json',
        scopes=['https://www.googleapis.com/auth/gmail.send'],
        redirect_uri='https://127.0.0.1:3202/checkemail'
    )
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='false',
        prompt='consent'
    )
    session['state'] = state
    return redirect(authorization_url)


@app.route('/checkemail')
def callback():
    state = session.get('state')
    if not state:
        return "State not found", 400

    flow = Flow.from_client_secrets_file(
        'src/client_secret.json',
        scopes=['https://www.googleapis.com/auth/gmail.send'],
        state=state,
        redirect_uri='https://127.0.0.1:3202/checkemail'
    )
    flow.fetch_token(authorization_response=request.url)

    credentials = flow.credentials
    session['credentials'] = credentials_to_dict(credentials)

    # Cập nhật REFRESH_TOKEN trong file .env
    if credentials.refresh_token:
        update_env_file("REFRESH_TOKEN", credentials.refresh_token)
    else:
        print("🔹 Không có refresh token mới được cấp.")

    print("🔹 Scope đã được cấp quyền:", credentials.scopes)

    return redirect(url_for('index'))


def update_env_file(key, value):
    with open('.env', 'r') as f:
        lines = f.readlines()

    with open('.env', 'w') as f:
        updated = False
        for line in lines:
            if line.startswith(f"{key}="):
                f.write(f"{key}={value}\n")
                updated = True
            else:
                f.write(line)

        if not updated:
            f.write(f"{key}={value}\n")

    print(f"🔹 Đã cập nhật {key} trong .env")


def credentials_to_dict(credentials):
    return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }


@app.route('/home')
def home():
    return render_template('user_home.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3202, ssl_context=('certificate.crt', 'private.key'))
