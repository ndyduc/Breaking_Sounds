from flask import Flask, jsonify
from pymongo import MongoClient
from bson.objectid import ObjectId
from pymongo.errors import DuplicateKeyError, PyMongoError
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from bson.binary import Binary

import os
from datetime import datetime
import io
import base64
from PIL import Image, UnidentifiedImageError

try:
	load_dotenv()
	mongo_uri = os.getenv('MONGO_ATLAS')
	client = MongoClient(mongo_uri)
	db = client['ndyduc']
	Users = db['Users']
	Vocals = db['Vocals']
	Sheet = db['Sheets']

	client.server_info()
	connection_status = True
	print("Kết nối MongoDB thành công!")

except Exception as e:
	print(f"Không thể kết nối đến MongoDB: {e}")
	connection_status = False


def Check_connect():
	if not connection_status:
		return jsonify({"status": "error", "message": "Không thể kết nối đến MongoDB"}), 500

	data = list(Users.find({}, {"_id": 0}))
	if data:
		return jsonify({"status": "success", "rows": len(data), "data": data})
	else:
		return jsonify({"status": "success", "message": "empty"})


def insert_musicxml(userid, musicxml, ispublic=False, name=None, instrument="Piano"):
	try:
		music_data = {
			"_id": ObjectId(),
			"user_id": ObjectId(userid),
			"MusicXML": musicxml,
			"Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
			"IsPublic": bool(ispublic),
			"Instrument": instrument,
			"Name": name,
			"Views": 0
		}

		result = Sheet.insert_one(music_data)
		return str(result.inserted_id)

	except PyMongoError as e:
		print(f"Lỗi MongoDB khi chèn MusicXML: {e}")
		return None
	except Exception as e:
		print(f"Lỗi không xác định: {e}")
		return None


def update_musicxml(doc_id, name=None, instrument=None, ispublic=None, musicxml=None, newtime = False):
	try:
		update_fields = {}

		if name is not None:
			update_fields["Name"] = name
		if instrument is not None:
			update_fields["Instrument"] = instrument
		if ispublic is not None:
			update_fields["IsPublic"] = bool(ispublic)
		if musicxml is not None:
			update_fields["MusicXML"] = musicxml

		if not update_fields:
			print("Không có trường nào để cập nhật.")
			return False

		if newtime:
			update_fields["Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

		result = Sheet.update_one(
			{"_id": ObjectId(doc_id)},
			{"$set": update_fields}
		)

		return result.modified_count > 0

	except PyMongoError as e:
		print(f"Lỗi MongoDB khi cập nhật MusicXML: {e}")
		return False
	except Exception as e:
		print(f"Lỗi không xác định: {e}")
		return False


def save_vocal(user_id, file, image, ispublic=False):
	try:
		file_data = file.read()
		img_binary = img_to_binary(image)

		mp3_document = {
			"_id": ObjectId(),
			"user_id": ObjectId(user_id),
			"filename": file.filename,
			"data": Binary(file_data),
			"img": img_binary,
			"Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
			"IsPublic": bool(ispublic),
			"view": 0
		}
		result = Vocals.insert_one(mp3_document)

		return str(result.inserted_id)

	except Exception as e:
		print(f"[Save Vocal Error] {e}")
		return None


def update_vocal(doc_id, filename=None, file=None, image=None, public=None, newtime=False):
	try:
		update_fields = {}

		if filename is not None:
			update_fields["filename"] = filename

		if file is not None:
			update_fields["data"] = Binary(file.read())

		if image is not None:
			update_fields["img"] = img_to_binary(image)

		if public is not None:
			update_fields["IsPublic"] = bool(public)

		if newtime:
			update_fields["Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

		if not update_fields:
			print("Không có trường nào để cập nhật.")
			return False

		result = Vocals.update_one(
			{"_id": ObjectId(doc_id)},
			{"$set": update_fields}
		)

		return result.matched_count > 0

	except Exception as e:
		print(f"[Update Vocal Error] {e}")
		return False


def get_all_user_data(userid, limit=None):
	try:
		user_obj_id = ObjectId(userid)

		# Lấy Sheet data
		sheet_data = []
		for item in Sheet.find({"user_id": user_obj_id}):
			item["type"] = "musicxml"
			item["_id"] = str(item["_id"])
			del item["user_id"]
			del item["MusicXML"]
			sheet_data.append(item)

		# Lấy Vocal data
		vocal_data = []
		for item in Vocals.find({"user_id": user_obj_id}):
			item["type"] = "vocal"
			item["_id"] = str(item["_id"])
			del item["user_id"]
			del item["data"]
			if "img" in item:
				item["img_base64"] = base64.b64encode(item["img"]).decode("utf-8")
				del item["img"]
			vocal_data.append(item)

		print(vocal_data)
		all_data = sheet_data + vocal_data
		all_data.sort(key=lambda x: x["Time"], reverse=True)

		if limit:
			all_data = all_data[:limit]

		return all_data

	except Exception as e:
		print(f"[Get All User Data Error] {e}")
		return None


def img_to_binary(image):
	try:
		file_size = len(image.read())
		image.seek(0)

		img = Image.open(io.BytesIO(image.read()))
		if img.mode == "RGBA":
			img = img.convert("RGB")
		img_bytes = io.BytesIO()

		if file_size > 5 * 1024 * 1024:  # 5MB
			img.save(img_bytes, format="JPEG", quality=50)
		else:
			img.save(img_bytes, format="JPEG")

		return img_bytes.getvalue()

	except UnidentifiedImageError as e:
		print(f"Lỗi: Không nhận diện được ảnh - {e}")
		return None
	except Exception as e:
		print(f"Lỗi không xác định khi xử lý ảnh: {e}")
		return None


def is_vocal_exists(user_id, filename):
	try:
		query = {"user_id": user_id, "filename": filename}
		return Vocals.find_one(query) is not None
	except Exception as e:
		print(f"Lỗi khi kiểm tra vocal: {e}")
		return False


def insert_user(id_google=None, username=None, email=None, password=None, avatar_url=None):
	try:
		user_data = {
			"_id": ObjectId(),
			"ID_google": id_google,
			"Username": username,
			"Email": email,
			"Password": generate_password_hash(password) if password else None,  # Mã hóa mật khẩu
			"Avatar_url": avatar_url,
			"Bio": None,
			"Organization": None,
			"City": None,
			"Country": None,
			"WhatsApp": None
		}

		result = Users.insert_one(user_data)
		return str(result.inserted_id)

	except PyMongoError as e:
		print(f"Lỗi MongoDB: {e}")
		return None
	except Exception as e:
		print(f"Lỗi không xác định: {e}")
		return None


def get_user(id_google=None, email=None, ob_id=None):
	try:
		query = {}
		if id_google:
			query["ID_google"] = id_google
		if email:
			query["Email"] = email
		if ob_id:
			query["_id"] = ObjectId(ob_id)

		user = Users.find_one(query)
		return user if user else None
	except Exception as e:
		print(f"Lỗi khi lấy user: {e}")
		return None


def verify_password(email, password_input):
	try:
		user = Users.find_one({"Email": email})
		if user and user.get("Password"):
			return check_password_hash(user["Password"], password_input)
		return False
	except Exception as e:
		print(f"An error: {e}")
		return False


def check_exist_user(email):
	user = Users.find_one({"Email": email})
	return user is not None


def update_user(user_id, id_google=None, username=None, email=None, password=None, avatar_url=None, bio=None,
				organization=None, city=None, country=None, whatsapp=None):
	if not ObjectId.is_valid(str(user_id)):
		return {"error": "Invalid ObjectId"}

	update_fields = {
		"ID_google": id_google,
		"Username": username,
		"Email": email,
		"Password": generate_password_hash(password) if password else None,
		"Avatar_url": avatar_url,
		"Bio": bio,
		"Organization": organization,
		"City": city,
		"Country": country,
		"WhatsApp": whatsapp
	}

	# Loại bỏ các giá trị `None` để tránh cập nhật thừa
	update_fields = {k: v for k, v in update_fields.items() if v is not None}

	if not update_fields: return {"error": "No valid fields to update"}

	result = Users.update_one(
		{"_id": ObjectId(user_id)},
		{"$set": update_fields}
	)

	if result.matched_count == 0:
		return {"error": "User not found"}
	elif result.modified_count == 0:
		return {"warning": "No changes were made"}

	return {"success": "User updated successfully"}
