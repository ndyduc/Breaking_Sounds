import io
import os
import subprocess
import base64
from io import BytesIO
import time

import librosa
import matplotlib.pyplot as plt
from flask import *
from werkzeug.utils import secure_filename
import numpy as np

from src.data_connecter import *

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Lấy thư mục cha của `router`
DATA_LOGS_DIR = os.path.join(BASE_DIR, "logs/Users")  # Đảm bảo logs nằm ngoài router

sounds = Blueprint('sounds', __name__, url_prefix='/')


@sounds.route('/generate')
def generate():
	return render_template('notes_generate.html')


@sounds.route("/upload", methods=["POST"])
def upload_file():
	if "file_up" not in request.files:
		return jsonify({"error": "No file uploaded"}), 400

	file = request.files["file_up"]
	action = request.form.get("action")

	user_id = session.get("user_id") if session.get("user_id") else str(int(time.time()))

	if file.filename == "":
		return jsonify({"error": "No selected file"}), 400

	filename = secure_filename(file.filename)
	user_folder = os.path.join(DATA_LOGS_DIR, user_id, "uploads")
	os.makedirs(user_folder, exist_ok=True)
	file_path = os.path.join(user_folder, filename)

	file.save(file_path)

	result_path = isolate_audio(file_path, user_id)
	if result_path is None or not os.path.exists(result_path):
		return jsonify({"error": "Không tạo được file vocals.mp3"}), 500

	file_size = os.path.getsize(result_path)
	file_type = 'audio/mpeg'

	waveform = base64.b64encode(create_waveform(result_path).getvalue()).decode("utf-8")

	try:
		cmd_duration = [
			"ffprobe", "-v", "error",
			"-show_entries", "format=duration", "-of", "json",
			result_path
		]
		output = subprocess.check_output(cmd_duration).decode("utf-8")
		duration_seconds = float(json.loads(output)["format"]["duration"])
		duration = time.strftime("%H:%M:%S", time.gmtime(duration_seconds))

	except Exception as e:
		duration = "Unknown"

	return jsonify({
		"message": "Xử lý thành công!",
		"file": url_for('sounds.get_audio', user_id=user_id, filename=filename, _external=True),
		"file_type": file_type,
		"file_size": file_size,
		"duration": duration,
		"waveform": f"data:image/png;base64,{waveform}"
	})


@sounds.route("/get_audio/<user_id>/<filename>")
def get_audio(user_id, filename):
	file_path = os.path.join(DATA_LOGS_DIR, user_id, "local_voice/", filename[:-4], "vocals.mp3")
	if not os.path.exists(file_path):
		return jsonify({"error": "File không tồn tại"}), 404
	return send_file(file_path, mimetype="audio/mpeg")


@sounds.route("/get_waveform", methods=["POST"])
def get_waveform():
	if "file" not in request.files:
		return "No file uploaded", 400

	file = request.files["file"]
	img_buffer = create_waveform(file)

	return send_file(img_buffer, mimetype="image/png")


@sounds.route("/save_vocals", methods=["POST"])
def save_vocals():
	try:
		if "user_id" not in session:
			return jsonify({"status": False, "message": "User not logged in"}), 401

		audio_data = request.files.get("audio")
		if audio_data:
			if is_vocal_exists(session["user_id"], audio_data.filename):
				return jsonify({"status": False, "message": "File already saves !"})
			else:
				vocal_id = save_vocal(session.get("user_id"), audio_data)
				if vocal_id is not None:
					user = get_user(ob_id=session.get("user_id"))
					return jsonify({
						"vocal_id": vocal_id,
						"username": user["Username"],
						"status": True}), 200
				else:
					return jsonify({"vocal_id": None, "status": False}), 404
		else:
			return jsonify({"status": False, "message": "No file received"}), 404

	except Exception as e:
		return jsonify({"status": False, "message": str(e)})


def create_waveform(file):
	y, sr = librosa.load(file, sr=44100)

	fig, ax = plt.subplots(figsize=(12, 4))
	ax.plot(y, color=(254 / 255, 188 / 255, 188 / 255))
	ax.set_title("Waveform", fontsize=16, fontweight="bold")  # Chữ to hơn
	ax.tick_params(axis="both", labelsize=12)  # Chỉnh cỡ chữ trục

	# Lưu ảnh vào buffer
	img_buffer = io.BytesIO()
	fig.savefig(img_buffer, format="png", bbox_inches="tight")
	plt.close(fig)
	img_buffer.seek(0)

	return img_buffer


def isolate_audio(input_file, user_id):
	output_folder = os.path.join(DATA_LOGS_DIR, user_id, "local_voice/")
	parent_folder = os.path.dirname(output_folder)  # Sau đó mới tạo thư mục con

	try:
		os.makedirs(DATA_LOGS_DIR, exist_ok=True)
		os.makedirs(parent_folder, exist_ok=True)
		os.makedirs(output_folder, exist_ok=True)
	except Exception as e:
		print(f"Lỗi khi tạo thư mục: {e}")

	cmd = ["umx", "--outdir", output_folder, input_file]
	filedir = os.path.splitext(os.path.basename(input_file))[0].replace(" ", "_")
	vocals_path = os.path.join(output_folder, filedir, "vocals.wav")
	subprocess.run(cmd, check=True)
	unnecessary_files = ["bass.wav", "drums.wav", "other.wav"]

	for file in unnecessary_files:
		file_path = os.path.join(output_folder + filedir + '/', file)
		if os.path.exists(file_path):
			os.remove(file_path)

	image_path = extract_album_art(input_file, output_folder)

	if image_path:
		vocals_mp3 = embed_album_art(vocals_path, image_path)
		if vocals_mp3:
			os.remove(image_path)
		return vocals_mp3
	else:
		return convert_to_mp3(vocals_path)


def extract_album_art(input_file, output_folder):
	image_path = os.path.join(output_folder, "vocals.jpg")

	cmd = [
		"ffmpeg", "-i", input_file,
		"-an", "-vcodec", "copy", image_path,
		"-y"
	]

	try:
		subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		return image_path if os.path.exists(image_path) else None
	except subprocess.CalledProcessError as e:
		print(f"Lỗi khi trích xuất ảnh: {e}")

	return None


def embed_album_art(audio_path, image_path):
	output_path = audio_path.replace(".wav", ".mp3")

	cmd = [
		"ffmpeg",
		"-i", audio_path,
		"-i", image_path,
		"-map", "0:a",
		"-map", "1:v",
		"-c:a", "libmp3lame",
		"-c:v", "mjpeg",
		"-q:v", "2",
		"-id3v2_version", "3",
		"-metadata:s:v", "title=Album cover",
		"-metadata:s:v", "comment=Cover (front)",
		output_path
	]
	try:
		subprocess.run(cmd, check=True)
		os.remove(audio_path)
	except subprocess.CalledProcessError as e:
		print(f"Lỗi khi nhúng ảnh: {e}")

	return output_path


def convert_to_mp3(audio_path):
	output_path = audio_path.replace(".wav", ".mp3")

	cmd = [
		"ffmpeg",
		"-i", audio_path,
		"-c:a", "libmp3lame",
		"-q:a", "2",
		output_path
	]

	try:
		subprocess.run(cmd, check=True)
		os.remove(audio_path)  # Xóa file gốc nếu convert thành công
		return output_path
	except subprocess.CalledProcessError as e:
		print(f"Lỗi khi chuyển sang MP3: {e}")

	return None


def whole__duration(tempo):
	return 240 / tempo  # Đơn vị: giây


def get_tempo_nhip(file):
	y, sr = librosa.load(file)

	# Phát hiện beat và downbeat
	tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
	onset_env = librosa.onset.onset_strength(y=y, sr=sr)

	# Chuyển beats thành thời gian
	beat_times = librosa.frames_to_time(beats, sr=sr)

	if len(beat_times) > 1:
		# Tính khoảng cách giữa các beats
		beat_intervals = np.diff(beat_times)
		# Tìm khoảng cách phổ biến nhất giữa các beats mạnh
		avg_interval = np.median(beat_intervals)

		# Xác định nhịp theo độ lặp lại của downbeat
		if avg_interval < 0.4:  # Nếu khoảng cách nhỏ (thường là 3 beats trong 1 chu kỳ)
			time_signature = "3/4"
		else:  # Nếu khoảng cách lớn hơn (thường là 4 beats trong 1 chu kỳ)
			time_signature = "4/4"
	else:
		time_signature = "Không xác định"

	return int(tempo[0]), time_signature
