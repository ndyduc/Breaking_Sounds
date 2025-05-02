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
from lxml import etree
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

		return redirect(url_for('non.viewsheet', path=file_path))

	else:
		path = request.args.get("path")
		return render_template("sheet_music.html", result_path=path)


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
		session.pop('musicxml_name', None)
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
		userid = session.get("user_id")
		item_id = data.get("item_id")
		name = data.get('name', session.get("musicxml_name"))
		if name is None:
			name = data.get('content')

		if not musicxml or not userid:
			return jsonify({"message": "Thiếu dữ liệu musicxml hoặc userid"}), 400

		parser = etree.XMLParser(remove_blank_text=True)
		root = etree.fromstring(musicxml.encode('utf-8'), parser)

		# Tìm tất cả các measure
		for measure in root.xpath('.//measure'):
			attributes = measure.find('attributes')
			if attributes is not None:
				measure.remove(attributes)
				measure.insert(0, attributes)

		# Chuyển lại thành text
		fixed_musicxml = etree.tostring(root, encoding='utf-8', pretty_print=True, xml_declaration=True).decode('utf-8')
		musicxml_binary = Binary(fixed_musicxml.encode('utf-8'))

		if item_id:
			load = update_musicxml(item_id, musicxml=musicxml_binary)
		else:
			load = insert_musicxml(userid=userid, musicxml=musicxml_binary, name=name)
			session.pop('musicxml_name', None)

		if load:
			return jsonify({"message": "Save at library !", "id": load}), 200
		else:
			return jsonify({"message": "Error, please try again !"}), 500

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
		kind = request.form.get("type")
		amount = int(request.form.get("amount"))
		public = request.form.get("public", "false").lower() == "true"

		key = request.form.get("keyword")
		if key is None:
			key = None
		print(kind, amount, public)
		if public:
			user_data, amount = get_all_public_data(kind, 20, amount, key)
		else:
			user_data, amount = get_all_user_data(user_id, kind, 20, amount, key)

		return jsonify({"data": user_data, "amount": amount})
	except Exception as e:
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
			success = update_vocal(doc_id, public=True)
	else:
		if _type == "musicxml":
			success = update_musicxml(doc_id, ispublic=False)
		else:
			success = update_vocal(doc_id, public=False)

	if success:
		return jsonify({"success": True})
	else:
		return jsonify({"success": False, "message": "Không thể cập nhật trạng thái file."})


@non.route('/check_amount_data', methods=['POST'])
def check_amount_data():
	user_id = session.get("user_id")
	if user_id is None:
		return redirect(url_for("loginbase"))

	try:
		kind = request.form.get("type", "all")
		total = count_user_data(user_id, kind)
		return jsonify({"total": total})
	except Exception as e:
		print(f"[Check Amount Data Error] {e}")
		return jsonify({"message": str(e)}), 500


@non.route("/delete_data", methods=["POST"])
def delete_data():
	user_id = session.get("user_id")
	if user_id is None:
		flash("Session time out, please login to continue !")
		return redirect(url_for("loginbase"))

	kind = request.form.get("type")
	data_id = request.form.get("data_id")

	check = remove_data(data_id, kind)
	if check:
		return jsonify({"success": True, "message": "Data deleted successfully."})
	else:
		return jsonify({"success": True, "message": "Error when deleting data."}), 500


@non.route("/view_file", methods=["GET"])
def view_file():
	kind = request.args.get("kind")
	item_id = request.args.get("item_id")

	if not kind or not item_id:
		return abort(400, "Thiếu thông tin king hoặc item_id")

	item = get_file_by_kind_and_id(kind, item_id)
	user_id = session.get("user_id") if session.get("user_id") else str(int(time.time()))
	if not item["IsPublic"]:
		if user_id != item["user_id"]:
			flash("This contend is Not public !")
			return redirect(url_for("index"))

	try:
		if kind == "sheet":
			filename = item.get("Name") + ".musicxml"
			save_dir = os.path.join(DATA_LOGS_DIR, user_id, "musicxml")
			file_bytes = item["MusicXML"]

		elif kind == "vocal":
			filename = item["filename"]
			subfolder = os.path.splitext(filename)[0]
			save_dir = os.path.join(DATA_LOGS_DIR, user_id, "local_voice", subfolder)
			file_bytes = item["data"]

		else:
			return abort(400, "Fail to view file.")

		os.makedirs(save_dir, exist_ok=True)
		file_path = os.path.join(save_dir, filename)

		with open(file_path, "wb") as f:
			f.write(file_bytes)

		if kind == "sheet":
			update_musicxml(item_id, view=True)
			return render_template("sheet_music.html", result_path=file_path)
		else:
			update_vocal(item_id, view=True)
			item["data"] = os.path.join("static/Users", user_id, "local_voice", subfolder, filename)
			return render_template("share_vocal.html", item=item)

	except Exception as e:
		print(f"[Save File Error] {e}")
		return jsonify({"success": False, "message": "Lỗi khi xem file."}), 500


@non.route("/get_xmlpath_for_edit", methods=["POST"])
def get_xmlpath_for_edit():
	user_id = session.get("user_id")
	if user_id is None:
		return redirect(url_for("loginbase"))

	try:
		file_id = request.form.get("file_id")

		item = get_file_by_kind_and_id("sheet", file_id)
		filename = item.get("Name") + ".musicxml"
		save_dir = os.path.join(DATA_LOGS_DIR, user_id, "musicxml")
		file_bytes = item["MusicXML"]

		os.makedirs(save_dir, exist_ok=True)
		file_path = os.path.join(save_dir, filename)

		with open(file_path, "wb") as f:
			f.write(file_bytes)

		return jsonify({"success": True, "musicxmlPath": file_path}), 200

	except Exception as e:
		print(f"[Get XML Path Error] {e}")
		return jsonify({"success": False, "message": str(e)}), 500


@non.route("/play_vocal", methods=["POST"])
def finish_edit_sheet():
	if "user_id" not in session:
		flash("Session time out, please login to continue !")
		return redirect(url_for("loginbase"))

	item_id = request.args.get("item_id")
	name = request.args.get("filename")

	user_id = session.get("user_id")

	try:
		file_path = os.path.join(DATA_LOGS_DIR, user_id, "local_voice", name, name + ".mp3")
		if not os.path.exists(file_path):
			item = get_file_by_kind_and_id("vocal", item_id)
			filename = item["filename"]
			subfolder = os.path.splitext(filename)[0]
			save_dir = os.path.join(DATA_LOGS_DIR, user_id, "local_voice", subfolder)
			file_bytes = item["data"]

			os.makedirs(save_dir, exist_ok=True)
			file_path = os.path.join(save_dir, filename)

			with open(file_path, "wb") as f:
				f.write(file_bytes)

			file_path = os.path.join("static/Users", user_id, "local_voice", subfolder, filename)
		else:
			file_path = os.path.join("static/Users", user_id, "local_voice", name, name + ".mp3")

		return jsonify(file_path), 200

	except Exception as e:
		print(f"[Save File Error] {e}")
		return jsonify({"success": False, "message": "Lỗi khi xem file."}), 500


@non.route("/view_sheet", methods=["GET"])
def view_sheet():
	item_id = request.args.get("item_id")

	item = get_file_by_kind_and_id("sheet", item_id)
	user_id = session.get("user_id") if session.get("user_id") else str(int(time.time()))

	try:
		filename = item.get("Name") + ".musicxml"
		save_dir = os.path.join(DATA_LOGS_DIR, user_id, "musicxml")
		file_bytes = item["MusicXML"]

		os.makedirs(save_dir, exist_ok=True)
		file_path = os.path.join(save_dir, filename)

		with open(file_path, "wb") as f:
			f.write(file_bytes)

		update_musicxml(item_id, view=True)
		return render_template("preview_sheet.html", result_path=file_path)

	except Exception as e:
		print(f"[Save File Error] {e}")
		return jsonify({"success": False, "message": "Lỗi khi xem file."}), 500
