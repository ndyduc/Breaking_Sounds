import io
import os
import subprocess
import base64
from io import BytesIO
import time
import urllib.parse
import re
from werkzeug.utils import secure_filename

from flask import *

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

		# Lưu file vào thư mục
		file.save(file_path)

		return render_template("sheet_music.html", result_path=file_path)

	return render_template("sheet_music.html")


@non.route('/update_sheet', methods=['POST'])
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
	return render_template("sheet_editor.html")


@non.route('/get_musicxml')
def get_musicxml():

	user_id = session.get("user_id") if session.get("user_id") else str(int(time.time()))
	path = os.path.join(DATA_LOGS_DIR, user_id, "musicxml", "1759.musicxml")\

	print(path)
	try:
		with open(path, "r") as f:
			musicxml_content = f.read()

		return Response(musicxml_content, mimetype='application/vnd.recordare.musicxml+xml')
	except FileNotFoundError:
		return "File not found", 404
