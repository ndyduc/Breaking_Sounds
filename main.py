import scipy
from flask import *
from google_auth_oauthlib.flow import Flow
from routes.login_google import google
from routes.sounds import sounds
from routes.nonsense import non
from src.data_connecter import *
from src.send_email import send_email, send_forgot
import numpy as np
import librosa

from PIL import Image
import os
import base64
import io
import hashlib

app = Flask(__name__)
app.secret_key = os.getenv('MY_KEY')
app.config["SESSION_TYPE"] = "filesystem"
app.register_blueprint(google)
app.register_blueprint(sounds)
app.register_blueprint(non)


@app.route('/')
def index():
	# Check_connect()
	messages = get_flashed_messages()
	return render_template('Home.html', messages=messages)


@app.route('/loginbase', methods=['GET', 'POST'])
def loginbase():
	msg_type = request.args.get('msg_type')

	if msg_type:
		messages = {
			"sheet_editor": "You must log in to access the Sheet Editor.",
			"upload": "You must log in to access the Library."
		}
		flash(messages.get(msg_type, "Bạn cần đăng nhập."), "warning")

	return render_template('Login_page.html')


@app.route('/signup', methods=['POST'])
def signup():
	session.pop('_flashes', None)
	email = request.form.get('email')
	username = request.form.get('username')
	password = request.form.get('password')

	if check_exist_user(email):
		flash(f"This email : " + email + " is already used ! <br> If this is your email you can take it back.", "error")
		return redirect(url_for('loginbase'))

	session['password'] = password

	if send_email(email, username):
		flash("Email was sent successfully !")
		return redirect(url_for('loginbase'))
	else:
		flash("Error when sending email !")
		return redirect(url_for('loginbase'))


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
			return redirect(url_for('loginbase'))
		if verify_password(email, password):
			user = get_user(None, email)
			session['name'] = user["Username"]
			session['email'] = user["Email"]
			session['user_id'] = str(user["_id"])
			avatar_url = user.get('Avatar_url')

			if avatar_url is not None:
				if isinstance(avatar_url, str) and avatar_url.startswith("http"):
					session['picture'] = avatar_url
				else:
					session['picture'] = decode_img(avatar_url)

			return redirect(url_for('index'))
		else:
			flash("Password was in correct !")
			return redirect(url_for('loginbase'))
	elif action == "reset":
		user = get_user(None, email)
		if user is None:
			flash("User not found !")
			session.clear()
			return redirect(url_for('loginbase'))
		else:
			return redirect(url_for('forgot'))
	else:
		return redirect(url_for('index'))


@app.route('/user_logout')
def user_logout():
	session.clear()
	return redirect(url_for('index'))


@app.route('/verify/<token>')
def verify_email(token):
	saved_token = session.get('verify_token')
	saved_email = session.get('verify_email')
	saved_username = session.get('verify_username')

	if saved_token is None or saved_token != token:
		return "<h2>Verify have been expire or invalid!</h2>"

	new_id = insert_user(None, saved_username, saved_email, session.get('password'))
	if new_id is not None:
		session['user_id'] = new_id
	else:
		return redirect(url_for('loginbase'))
	# Xác minh thành công → Xóa session để tránh dùng lại
	session.pop('verify_token', None)
	session.pop('verify_email', None)
	session.pop('verify_username', None)

	return render_template('support/sidebar.html', username=saved_username, email=saved_email)


@app.route('/forgot')
def forgot():
	email = session.get('email')
	user = get_user(None, email)
	print(user.get('Username'))

	if send_forgot(email, user.get('Username')):
		flash("Email was sent successfully !")
		return redirect(url_for('loginbase'))
	else:
		flash("Error when sending email !")
		return redirect(url_for('loginbase'))


@app.route('/change_userpassword')
def change_userpassword():
	email = session.get('email')
	user = get_user(None, email)

	if send_forgot(email, user.get('Username')):
		flash("Check your email to change password !")
		return redirect(url_for('index'))
	else:
		flash("Error when sending email !")
		return redirect(url_for('index'))


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

	email = session.get('email')
	user = get_user(None, email)
	if not user:
		return jsonify({"success": False, "message": "User not found!"})

	if pass1 != pass2:
		return jsonify({"success": False, "message": "Passwords do not match!"})

	if not email:
		return jsonify({"success": False, "message": "User not authenticated!"})

	session['user_id'] = str(user.get('_id'))
	user_up = update_user(user.get('_id'), None, None, None, pass1, None, None, None, None, None, None)

	if user_up == {"success": "User updated successfully"}:
		return jsonify({
			"success": True,
			"message": "Password updated successfully!",
			"redirect": url_for('index')
		})

	return jsonify({"success": False, "message": "Error updating password!"})


@app.route('/change_password', methods=['GET', 'POST'])
def change_password():
	return render_template('support/change_password.html')


@app.route("/change_info", methods=["POST"])
def change_info():
	try:
		if 'user_id' not in session:
			return jsonify({"success": False, "message": "Can't identify user!"})

		user_id = session['user_id']
		new_username = request.form.get("username")
		profile_img = request.files.get("profile_img")

		if profile_img:
			try:
				img_binary = img_to_binary(profile_img)
			except Exception as e:
				return jsonify({"success": False, "message": f"Image processing error: {str(e)}"})
		else:
			img_binary = None

		user = update_user(user_id, None, new_username, None, None, img_binary)
		user_up = get_user(None, None, session.get('user_id'))

		session['name'] = user_up['Username']
		img_bi = user_up['Avatar_url']
		img_path = decode_img(img_bi)

		session['picture'] = img_path
		if user['success'] is None:
			return jsonify({"success": False, "message": "Error updating infomation !"})
		return jsonify({"success": True})
	except Exception as e:
		return jsonify({"success": False, "message": "Không có thông tin nào thay đổi."})


def decode_img(img_bytes):
	try:
		if not isinstance(img_bytes, bytes):
			print("Lỗi: Dữ liệu không phải là dạng nhị phân.")
			return None

		UPLOAD_FOLDER = os.path.join("static/Users", session.get('user_id'), 'info')
		img_path = os.path.join(UPLOAD_FOLDER, "profile.jpg")
		os.makedirs(UPLOAD_FOLDER, exist_ok=True)

		# Tính hash SHA-256 của ảnh mới
		new_img_hash = hashlib.sha256(img_bytes).hexdigest()

		# Kiểm tra xem ảnh cũ có giống ảnh mới không
		if os.path.exists(img_path):
			with open(img_path, "rb") as f:
				existing_img_hash = hashlib.sha256(f.read()).hexdigest()
			if existing_img_hash == new_img_hash:
				print("Anh co san !")
				return img_path

		with open(img_path, "wb") as f:
			f.write(img_bytes)

		return img_path  # Trả về đường dẫn ảnh mới
	except Exception as e:
		print(f"Lỗi không xác định: {str(e)}")
		return None


@app.route('/favicon.ico')
def favicon():
	return send_from_directory('static', 'favicon.ico')


@app.route('/authorize')
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
	return render_template('home.html')


@app.route("/check_session_user_id")
def check_session():
	if "user_id" in session:
		return jsonify({"logged_in": True})
	return jsonify({"logged_in": False})


@app.route('/get_notes', methods=['GET'])
def get_notes():
	notes = [
		{"keys": ["c/4"], "duration": "q"},
		{"keys": ["d/4"], "duration": "q"},
		{"keys": ["e/4"], "duration": "q"},
		{"keys": ["f/4"], "duration": "q"}
	]
	return jsonify(notes)


@app.route("/get_change_profile")
def your_route():
	return render_template("support/edit_profile.html")  # Trả về nội dung HTML


@app.route("/link_to_gg")
def link_to_gg():
	if 'google_id' in session:
		return jsonify({"linked": True, "message": "Account is already linked to Google account."})
	else:
		return jsonify({"linked": False, "redirect_url": url_for('google.login')})


if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port=3202, ssl_context=('certificate.crt', 'private.key'))
