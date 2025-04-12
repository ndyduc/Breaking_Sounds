import io
import os
import subprocess
import base64
from io import BytesIO
import time
import urllib.parse
import re
from werkzeug.utils import secure_filename
from bson.binary import Binary
import tempfile
from music21 import converter, environment

from flask import *
from src.data_connecter import *

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Lấy thư mục cha của `router`
DATA_LOGS_DIR = os.path.join(BASE_DIR, "static/Users")  # Đảm bảo logs nằm ngoài router
Trained_DiR = os.path.join(BASE_DIR, "Trained")

non = Blueprint('non', __name__, url_prefix='/')


@non.route("/pitch_detector")
def pitch_detector():
	return render_template("pitch_detector.html")


@non.route("/sheet_view")
def sheet_view():
	return render_template("sheet_view.html")


@non.route("/viewsheet", methods=["GET", "POST"])
def viewsheet():
	if request.method == "POST":
		file = request.files.get("file")

		if not file or not file.filename.endswith(".musicxml"):
			return jsonify({"success": False, "message": "Invalid file format"}), 234

		# Đặt tên file và lưu vào thư mục
		filename = secure_filename(file.filename)
		user_id = session.get("user_id") if session.get("user_id") else str(int(time.time()))
		os.makedirs(os.path.join(DATA_LOGS_DIR, user_id, "musicxml"), exist_ok=True)
		file_path = os.path.join(DATA_LOGS_DIR, user_id, "musicxml", filename)

		file.save(file_path)

		return render_template("sheet_music.html", result_path=file_path)

	return render_template("sheet_music.html")


@non.route('/update_sheet', methods=['POST', 'GET'])
def update_sheet():
	data = request.json
	note_index = data.get("note_index")
	new_pitch = data.get("new_pitch")

	# Load MusicXML từ file
	with open(data["path"], "r") as file:
		musicxml_content = file.read()

	# Cập nhật MusicXML (cần xử lý XML)
	# Ở đây bạn có thể dùng xml.etree.ElementTree hoặc music21 để chỉnh sửa

	# Lưu lại file MusicXML
	with open(data["path"], "w") as file:
		file.write(musicxml_content)

	return jsonify({"status": "success"})


@non.route('/sheet_editor')
def sheet_editor():
	path = request.args.get('path')

	if path:
		return render_template("sheet_editor.html", path=path)
	else:
		return render_template("sheet_editor.html")


@non.route('/get_musicxml')
def get_musicxml():
	path = request.args.get('path')
	if path is None:
		return jsonify({"error": "No path provided"}), 400

	try:
		with open(path, "r") as f:
			musicxml_content = f.read()

		return Response(musicxml_content, mimetype='application/vnd.recordare.musicxml+xml')

	except FileNotFoundError:
		return jsonify({"error": "File not found"}), 404


@non.route('/edit_upload_sheet', methods=['POST'])
def edit_upload_sheet():
	try:
		data = request.json
		path = data.get('musicxmlPath')

		if not path:
			return jsonify({"success": False, "message": "Invalid file format"}), 400

		# Trả về JSON chứa URL để chuyển hướng thay vì dùng redirect()
		return jsonify({"success": True, "redirect_url": url_for('non.sheet_editor', path=path)})
	except Exception as e:
		print(e)
		return jsonify({"success": False, "message": str(e)}), 500


@non.route('/save_musicxml', methods=['POST'])
def save_musicxml():
	try:
		data = request.get_json()

		musicxml = data.get('musicxml')
		# instrument = data.get('content')
		userid = session.get("user_id")
		name = data.get('name', session.get("musicxml_name"))
		ispublic = False

		if not musicxml or not userid:
			return jsonify({"message": "Thiếu dữ liệu musicxml hoặc userid"}), 400

		musicxml_binary = Binary(musicxml.encode('utf-8'))

		insert_id = insert_musicxml(userid=userid, musicxml=musicxml_binary, ispublic=ispublic, name=name)

		if insert_id:
			return jsonify({"message": "Lưu thành công", "id": insert_id}), 200
		else:
			return jsonify({"message": "Lưu thất bại"}), 500

	except Exception as e:
		print("Lỗi khi xử lý /save_musicxml:", e)
		return jsonify({"message": "Lỗi server"}), 500


@non.route('/save_pdf', methods=['POST'])
def save_pdf():
	try:
		data = request.json
		musicxml = data.get('musicxml')

		if not musicxml:
			return {"message": "Thiếu dữ liệu MusicXML."}, 400

		timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
		pdf_filename = f"sheet_{timestamp}.pdf"

		user_id = session.get("user_id")

		musicxml_dir = os.path.join(DATA_LOGS_DIR, user_id, "musicxml")
		os.makedirs(musicxml_dir, exist_ok=True)
		musicxml_path = os.path.join(musicxml_dir, f"music_{timestamp}.musicxml")
		pdf_dir = os.path.join(DATA_LOGS_DIR, user_id, "pdf")
		os.makedirs(pdf_dir, exist_ok=True)
		pdf_path = os.path.join(pdf_dir, pdf_filename)

		# Lưu file MusicXML
		with open(musicxml_path, "w", encoding='utf-8') as f:
			f.write(musicxml)

		# Dùng MuseScore để convert sang PDF
		mscore_path = "/Applications/MuseScore 4.app/Contents/MacOS/mscore"
		subprocess.run([mscore_path, musicxml_path, "-o", pdf_path], check=True)

		return send_file(pdf_path, as_attachment=True, download_name=pdf_filename, mimetype='application/pdf')

	except subprocess.CalledProcessError as e:
		print(f"Lỗi khi gọi MuseScore: {e}")
		return {"message": "Lỗi xử lý MuseScore."}, 500
	except Exception as e:
		print(f"Lỗi khác: {e}")
		return {"message": "Lỗi xử lý PDF."}, 500


@non.route('/libary')
def libary():
	if session.get("user_id") is not None:
		return render_template("libary.html")
	else:
		flash("Login to access libary !")
		return redirect(url_for("loginbase"))


@non.route('get_libary', methods=['POST'])
def get_libary():
	user_id = session.get("user_id")
	if user_id is None:
		return redirect(url_for("loginbase"))

	try:
		type = request.form.get("type")
		amount = int(request.form.get("amount"))

		user_data = get_all_user_data(user_id, amount)

		return jsonify(user_data)
	except Exception as e:
		print(e)
		return jsonify({"message": str(e)}), 500


@non.route("/unlock_file", methods=["POST"])
def unlock_file():
	user_id = session.get("user_id")
	if not user_id:
		return jsonify({"success": False, "message": "Chưa đăng nhập!"}), 401

	data = request.get_json()
	doc_id = data.get("_id")
	status = data.get("status")
	_type = data.get("type")

	if not doc_id or status not in ["lock", "unlock"]:
		return jsonify({"success": False, "message": "Missing infomation !"})

	if status == "unlock":
		if _type == "musicxml":
			success = update_musicxml(doc_id, ispublic=True)
		else:
			success = update_vocal(doc_id, public= True)
	else:
		if _type == "musicxml":
			success = update_musicxml(doc_id, ispublic=False)
		else:
			success = update_vocal(doc_id, public=False)

	if success:
		return jsonify({"success": True})
	else:
		return jsonify({"success": False, "message": "Không thể cập nhật trạng thái file."})
