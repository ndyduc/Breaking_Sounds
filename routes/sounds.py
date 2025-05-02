import io
import os
import subprocess
import base64
import threading
from io import BytesIO
import time
import urllib.parse
import re

import librosa
import matplotlib.pyplot as plt
from flask import *
from werkzeug.utils import secure_filename
import numpy as np
from collections import defaultdict
import music21 as m21
from music21 import converter, instrument
import torch
from music21 import metadata

from src.data_connecter import *
import Model.CNN_3L_pro
import Model.CNN_BiLSTM
import Model.BiLSTM

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Lấy thư mục cha của `router`
DATA_LOGS_DIR = os.path.join(BASE_DIR, "static/Users")  # Đảm bảo logs nằm ngoài router
Trained_DiR = os.path.join(BASE_DIR, "Trained")

sounds = Blueprint('sounds', __name__, url_prefix='/')


@sounds.route("/rest")
def rest():
	if "user_id" not in session:
		flash("You are not logged in!", "danger")
		return redirect(url_for("loginbase"))

	return render_template("rest.html")


@sounds.route("/practice")
def practice():
	return render_template("practice_library.html")


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

	filename = secure_filename(file.filename)
	user_folder = os.path.join(DATA_LOGS_DIR, user_id, "uploads")
	os.makedirs(user_folder, exist_ok=True)
	file_path = os.path.join(user_folder, filename)

	file.save(file_path)

	result_path = isolate_audio(file_path, user_id, False)
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
	try:
		if "file" in request.files:
			file = request.files["file"]
			file_data = file.read()
		else:
			file_path = request.form.get("file_path")
			if not file_path or not os.path.exists(file_path):
				return "File path không hợp lệ hoặc không tồn tại", 400

			with open(file_path, "rb") as f:
				file_data = f.read()

		thread_result = {}

		def task(filein, result_dict):
			try:
				if isinstance(filein, bytes):
					img_buffer = create_waveform(io.BytesIO(filein))  # nếu là bytes thì dùng BytesIO
				else:
					img_buffer = create_waveform(filein)  # nếu là file-like object

				result_dict["buffer"] = img_buffer
			except Exception as e:
				result_dict["error"] = str(e)

		t = threading.Thread(target=task, args=(file_data, thread_result))
		t.start()
		t.join()

		if "error" in thread_result:
			return f"Lỗi khi tạo waveform: {thread_result['error']}", 500

		return send_file(thread_result["buffer"], mimetype="image/png"), 200

	except Exception as e:
		print(f"[Waveform Error] {e}")
		return f"Lỗi server: {e}", 500


@sounds.route("/save_vocals", methods=["POST"])
def save_vocals():
	try:
		if "user_id" not in session:
			return jsonify({"status": False, "message": "User not logged in"}), 401

		audio_data = request.files.get("audio")
		img = request.files.get("image")
		if audio_data:
			if is_vocal_exists(session["user_id"], audio_data.filename):
				return jsonify({"status": False, "message": "File already saves !"})
			else:
				vocal_id = save_vocal(session.get("user_id"), audio_data, img)
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


@sounds.route("/CNN_basic_generate", methods=["POST"])
def CNN_basic_generate():
	if "file" not in request.files:
		return "No file uploaded", 400

	file = request.files["file"]


@sounds.route("/result_sheet", methods=["POST"])
def result_sheet():
	try:
		if "file_up" not in request.files:
			return "No file uploaded", 400

		file = request.files["file_up"]

		user_id = session.get("user_id") if session.get("user_id") else str(int(time.time()))

		filename = secure_filename(file.filename)
		session["musicxml_name"] = filename[:-4]
		user_folder = os.path.join(DATA_LOGS_DIR, user_id, "uploads")
		os.makedirs(user_folder, exist_ok=True)
		file_path = os.path.join(user_folder, filename)

		file.save(file_path)

		if "user_id" in session:
			file_path = os.path.join("static/Users", user_id, "uploads", filename)
			return render_template(
				"generate_pro.html",
				file_path=file_path,
				image_src=request.form.get("image_src"),
				waveform=request.form.get("waveform"),
				filename=request.form.get("filename"),
				kind=request.form.get("kind"),
				size=request.form.get("size"),
				duration=request.form.get("duration"))

		melodi_path = isolate_audio(file_path, user_id, True)
		device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

		model = Model.CNN_3L_pro.CNN_Pro().to(device)
		checkpoint = torch.load(os.path.join(Trained_DiR, "CNN_Pro.pth"), map_location=device, weights_only=True)
		model.load_state_dict(checkpoint['model_state_dict'])
		model.eval()

		notes_predict = Model.CNN_3L_pro.CNN_predict(melodi_path, model, device=device)

		tempo, pulse = get_tempo_pulse(melodi_path, 44100)
		note_data = process_notes_sum(notes_predict, tempo)

		result_path = convert_to_musicxml(note_data, tempo, pulse, user_id, filename)

		return redirect(url_for('non.viewsheet', path=result_path))
	except Exception as e:
		return jsonify({"error": str(e)}), 500


@sounds.route("/generate-pro", methods=["POST"])
def generate_pro():
	user_id = session.get("user_id")
	if not user_id:
		flash("You need to login first!", "danger")
		return redirect(url_for("loginbase"))

	try:
		filename = request.form.get("file_name")
		session["musicxml_name"] = filename[:-4]
		model = request.form.get("model")
		instru = request.form.get("instrument")
		file_path = request.form.get("file_path")

		if not instru:
			instru = None

		melodi_path = isolate_audio(file_path, user_id, True)
		device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
		if model == "CNN":
			model = Model.CNN_3L_pro.CNN_Pro().to(device)
			checkpoint = torch.load(os.path.join(Trained_DiR, "CNN_Pro1791.pth"), map_location=device, weights_only=True)
			model.load_state_dict(checkpoint['model_state_dict'])
			model.eval()

			notes_predict = Model.CNN_3L_pro.CNN_predict(melodi_path, model, device=device)

		elif model == "BiLSTM":
			model = Model.BiLSTM.BiLSTMNoteClassifier().to(device)
			checkpoint = torch.load(os.path.join(Trained_DiR, "BiLSTM_muti.pth"), map_location=device)
			model.load_state_dict(checkpoint)

			notes_predict = Model.BiLSTM.predict(model, melodi_path, device)

		elif model == "CNN-BiLSTM":
			model = Model.CNN_BiLSTM.CNN_BiLSTM().to(device)
			checkpoint = torch.load(os.path.join(Trained_DiR, "CNN_BiLSTM.pth"), map_location=device, weights_only=True)
			model.load_state_dict(checkpoint['model_state_dict'])

			notes_predict = Model.CNN_BiLSTM.predict(model, melodi_path, device=device)
		else:
			return jsonify({"status": False, "message": "Model does not exist"}), 500

		tempo, pulse = get_tempo_pulse(melodi_path, 44100)
		note_data = process_notes_sum(notes_predict, tempo)
		result_path = convert_to_musicxml(note_data, tempo, pulse, user_id, filename, instru)
		return redirect(url_for('non.viewsheet', path=result_path))

	except Exception as e:
		return jsonify({"error": str(e)}), 500


@sounds.route("/get_sheet")
def get_sheet():
	file_path = request.args.get("path")
	file_path = urllib.parse.unquote(file_path)

	if not file_path or not os.path.exists(file_path):
		return jsonify({"error": "File không tồn tại"}), 404

	try:
		with open(file_path, "r", encoding="utf-8") as f:
			return f.read(), 200
	except Exception as e:
		return jsonify({"error": str(e)}), 500


def convert_to_musicxml(note_data, tempo, pulse, user_id, filename, instru=None):
	"""Args:
		note_data (list): Danh sách dict chứa thông tin nốt nhạc.
		tempo (int): Tempo của bản nhạc.
		pulse (str): Nhịp (vd: "4/4", "3/4").
		user_id (str): User ID.
		filename (str) : filename
		instru (str) : nhac cu

	Returns:
		str | dict: Đường dẫn file MusicXML hoặc dict chứa lỗi.
	"""
	try:
		score = m21.stream.Score()
		upper = m21.stream.PartStaff()
		lower = m21.stream.PartStaff()

		# Gán ID và nhạc cụ
		upper.id = "RH"
		lower.id = "LH"

		upper.partName = " "
		lower.partName = " "

		upper.insert(0, m21.instrument.Piano())

		# Định nghĩa nhịp và tempo chỉ cần trong upper
		time_signature = m21.meter.TimeSignature(pulse)
		metronome = m21.tempo.MetronomeMark(number=tempo)
		upper.append(time_signature)
		upper.append(metronome)

		last_end_beat = 0

		from collections import defaultdict

		# Nhóm các note theo start_beat
		notes_by_beat = defaultdict(list)
		for note in note_data:
			notes_by_beat[note["start_beat"]].append(note)

		# Sắp xếp theo thời gian bắt đầu
		sorted_beats = sorted(notes_by_beat.keys())

		last_end_beat = 0

		for start_beat in sorted_beats:
			group = notes_by_beat[start_beat]

			# Tách thành 2 nhóm theo staff
			upper_notes = [n for n in group if n["note"] >= 60]
			lower_notes = [n for n in group if n["note"] < 60]

			# Nếu có khoảng trống thì thêm Rest
			if start_beat > last_end_beat:
				rest_duration = start_beat - last_end_beat
				rest = m21.note.Rest()
				rest.duration = m21.duration.Duration(rest_duration)
				if upper_notes:
					upper.append(rest)
				if lower_notes:
					lower.append(rest)

			def create_note_or_chord(note_group):
				if not note_group:
					return None

				try:
					duration_beat = NOTE_DURATIONS.get(note_group[0]["note_value"].replace("tied_", ""), 1.0)
					is_tied = "tied_" in note_group[0]["note_value"]

					if len(note_group) == 1:
						n = m21.note.Note()
						n.pitch.midi = note_group[0]["note"]
						n.duration = m21.duration.Duration(duration_beat)

						if is_tied:
							n.tie = m21.tie.Tie("start")  # Tie bắt đầu
						return n

					else:
						notes = []
						for item in note_group:
							single_note = m21.note.Note()
							single_note.pitch.midi = item["note"]
							if "tied_" in item["note_value"]:
								single_note.tie = m21.tie.Tie("start")
							notes.append(single_note)

						chord = m21.chord.Chord(notes)
						chord.duration = m21.duration.Duration(duration_beat)
						if is_tied:
							chord.tie = m21.tie.Tie("start")
						return chord

				except Exception as e:
					print(f"Error creating note/chord: {e}")
					return None

			upper_elm = create_note_or_chord(upper_notes)
			lower_elm = create_note_or_chord(lower_notes)

			if upper_elm:
				upper.append(upper_elm)
			if lower_elm:
				lower.append(lower_elm)

			# Cập nhật last_end_beat
			duration = NOTE_DURATIONS.get(group[0]["note_value"].replace("tied_", ""), 1.0)
			last_end_beat = start_beat + duration

		# Thêm các staff vào score
		score.insert(0, upper)

		if instru == "Piano" or instru is None:
			score.insert(0, lower)

		# Metadata và xuất file
		score.makeNotation()
		name, _ = os.path.splitext(filename)
		score.metadata = metadata.Metadata()
		score.metadata.title = name
		score.metadata.composer = "_ndyduc_ genetive"

		melody_path = os.path.join(DATA_LOGS_DIR, user_id, "musicxml", name + ".musicxml")
		os.makedirs(os.path.dirname(melody_path), exist_ok=True)
		score.write(fmt="musicxml", fp=melody_path)

		with open(melody_path, "r", encoding="utf-8") as f:
			xml_un = f.read()

		xml_un = re.sub(r"<part-name\s*/>", "<part-name> </part-name>", xml_un, flags=re.DOTALL)
		xml_un = re.sub(r"<movement-title>.*?</movement-title>", "", xml_un, flags=re.DOTALL)

		with open(melody_path, "w", encoding="utf-8") as f:
			f.write(xml_un)

		return melody_path

	except Exception as e:
		print(f"Lỗi khi chuyển đổi: {e}")
		return {"error": str(e)}


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


def isolate_audio(input_file, user_id, melody):
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

	unnecessary_files = []
	if melody:
		unnecessary_files.extend(["bass.wav", "drums.wav", "vocals.wav"])
		name = "other.wav"
	else:
		unnecessary_files.extend(["bass.wav", "drums.wav", "other.wav"])
		name = "vocals.wav"

	vocals_path = os.path.join(output_folder, filedir, name)
	subprocess.run(cmd, check=True)

	for file in unnecessary_files:
		file_path = os.path.join(output_folder + filedir + '/', file)
		if os.path.exists(file_path):
			os.remove(file_path)

	if melody:
		return vocals_path

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


def round_beat(start_time, tempo):
	beat = start_time * tempo / 60
	return round(beat * 2) / 2


def convert_duration_to_beat(duration_s, tempo, base=0.125):
	duration_beat = (duration_s * tempo) / 60
	return round(duration_beat / base) * base


def midi_to_note(midi_number):
	return librosa.midi_to_note(midi_number)


def classify_note_value(duration_beat):
	note_values = {
		4.125: "whole",
		4.0: "whole",
		3.875: "dotted half",
		3.75: "dotted half",
		3.625: "dotted half",
		3.5: "dotted half",
		3.375: "dotted half",
		3.25: "dotted half",
		3.125: "dotted half",
		3.0: "dotted half",
		2.875: "dotted half",
		2.75: "half",
		2.625: "half",
		2.5: "half",
		2.375: "half",
		2.25: "half",
		2.125: "half",
		2.0: "half",
		1.875: "half",
		1.75: "half",
		1.625: "half",
		1.5: "dotted quarter",
		1.375: "dotted quarter",
		1.25: "tied quarter-sixteenth",
		1.125: "quarter",
		1.0: "quarter",
		0.875: "quarter",
		0.625: "quarter",
		0.75: "dotted eighth",
		0.5: "eighth",
		0.375: "dotted sixteenth",
		0.25: "sixteenth",
		0.1875: "dotted thirty-second",
		0.125: "thirty-second"
	}

	for key in sorted(note_values.keys(), reverse=True):
		if abs(duration_beat - key) < 0.01:  # Kiểm tra xấp xỉ để tránh sai số làm tròn
			return note_values[key]

	if duration_beat > 4:
		return "whole"
	else:
		return "unknown"


def process_notes(note_list, tempo):
	processed_notes = []

	for note in note_list:
		start_beat = round_beat(note["start"], tempo)
		duration_beat = convert_duration_to_beat(note["duration"], tempo)
		note_value = classify_note_value(duration_beat)

		processed_notes.append({
			"note": note["note"],
			"start_beat": start_beat,
			"duration_beat": duration_beat,
			"note_value": note_value
		})

	return processed_notes


def process_notes_sum(note_list, tempo):
	note_dict = defaultdict(float)  # Lưu duration của mỗi (start_beat, note)

	for note in note_list:
		start_beat = round_beat(note["start"], tempo)
		duration_beat = convert_duration_to_beat(note["duration"], tempo)

		key = (start_beat, note["note"])
		note_dict[key] += duration_beat  # Cộng dồn duration nếu cùng start_beat, note

	processed_notes = []
	for (start_beat, note), duration_beat in note_dict.items():
		duration_beat = convert_duration_to_beat(duration_beat, tempo)  # Làm tròn lần nữa
		note_value = classify_note_value(duration_beat)

		processed_notes.append({
			"note": note,
			"start_beat": start_beat,
			"duration_beat": duration_beat,
			"note_value": note_value
		})

	return processed_notes


def get_tempo_pulse(file, sr=44100):
	y, sr = librosa.load(file, sr=sr)

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
		time_signature = "3/4"

	return int(tempo[0]), time_signature


# Ánh xạ note_value -> beat duration
NOTE_DURATIONS = {
	"whole": 4.0,
	"half": 2.0,
	"quarter": 1.0,
	"eighth": 0.5,
	"sixteenth": 0.25,
	"thirty-second": 0.125,

	# Dotted notes
	"dotted half": 3.0,  # 2 + 1
	"dotted quarter": 1.5,  # 1 + 0.5
	"dotted eighth": 0.75,  # 0.5 + 0.25
	"dotted sixteenth": 0.375,  # 0.25 + 0.125
}
