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


def update_musicxml(doc_id, name=None, instrument=None, ispublic=None, musicxml=None, view=False, newtime=False):
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
		if newtime:
			update_fields["Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

		update_query = {}
		if update_fields:
			update_query["$set"] = update_fields

		if view:
			update_query["$inc"] = {"Views": 1}

		if not update_query:
			print("Không có trường nào để cập nhật.")
			return False

		result = Sheet.update_one(
			{"_id": ObjectId(doc_id)},
			update_query
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
			"Views": 0
		}
		result = Vocals.insert_one(mp3_document)

		return str(result.inserted_id)

	except Exception as e:
		print(f"[Save Vocal Error] {e}")
		return None


def update_vocal(doc_id, filename=None, file=None, image=None, public=None, view=False, newtime=False):
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

		update_query = {}
		if update_fields:
			update_query["$set"] = update_fields

		if view:
			print(doc_id)
			update_query["$inc"] = {"Views": 1}

		if not update_fields and not view:
			print("Không có trường nào để cập nhật.")
			return False

		result = Vocals.update_one(
			{"_id": ObjectId(doc_id)},
			update_query
		)
		return result.matched_count > 0

	except Exception as e:
		print(f"[Update Vocal Error] {e}")
		return False


def count_user_data(user_id: str, kind: str) -> int:
	user_obj_id = ObjectId(user_id)

	if kind == "musicxml":
		return Sheet.count_documents({"user_id": user_obj_id})
	elif kind == "vocal":
		return Vocals.count_documents({"user_id": user_obj_id})
	else:
		return (
			Sheet.count_documents({"user_id": user_obj_id}) +
			Vocals.count_documents({"user_id": user_obj_id})
		)


def count_public_data(kind: str) -> int:
	if kind == "musicxml":
		return Sheet.count_documents({"IsPublic": True})
	elif kind == "vocal":
		return Vocals.count_documents({"IsPublic": True})
	else:
		return (
			Sheet.count_documents({"IsPublic": True}) +
			Vocals.count_documents({"IsPublic": True})
		)


def remove_data(data_id, kind):
	try:
		collection = None
		if kind == "musicxml":
			collection = Sheet
		elif kind == "vocal":
			collection = Vocals
		else:
			print(f"Không xác định được loại dữ liệu: {kind}")
			return False

		result = collection.delete_one({"_id": ObjectId(data_id)})
		if result.deleted_count == 1:
			print(f"Đã xóa {kind} có _id: {data_id}")
			return True
		else:
			print(f"Không tìm thấy {kind} để xóa với _id: {data_id}")
			return False

	except PyMongoError as e:
		print(f"Lỗi MongoDB khi xóa {kind}: {e}")
		return False
	except Exception as e:
		print(f"Lỗi không xác định khi xóa {kind}: {e}")
		return False


def get_all_user_data(userid, kind, limit, index, keyword=None):
	try:
		user_obj_id = ObjectId(userid)
		amount_data = count_user_data(userid, kind)
		if ((index - 1) * limit) + 1 > amount_data:
			index = amount_data // limit if amount_data % limit == 0 else amount_data // limit + 1

		def process_extra(item):
			if item["type"] == "vocal":
				item["filename"] = item.get("filename", "")[:-4]
				if "img" in item:
					item["img_base64"] = base64.b64encode(item["img"]).decode("utf-8")
					del item["img"]
			item["_id"] = str(item["_id"])
			return item

		# Chuẩn hóa keyword (nếu có)
		keyword_filter = {}
		if keyword:
			keyword = keyword.lower()
			regex = {"$regex": keyword, "$options": "i"}

		if kind == "musicxml":
			match_stage = {"user_id": user_obj_id}
			if keyword:
				match_stage["$or"] = [
					{"Instrument": regex},
					{"Name": regex},
					{"Time": regex}
				]
			pipeline = [
				{"$match": match_stage},
				{"$project": {
					"type": {"$literal": "musicxml"},
					"Time": 1,
					"IsPublic": 1,
					"Instrument": 1,
					"Name": 1,
					"_id": 1,
					"Views": 1
				}},
				{"$sort": {"Time": -1}},
				{"$skip": (index - 1) * limit},
				{"$limit": limit}
			]
			result = Sheet.aggregate(pipeline)

		elif kind == "vocal":
			match_stage = {"user_id": user_obj_id}
			if keyword:
				match_stage["$or"] = [
					{"filename": regex},
					{"Time": regex}
				]
			pipeline = [
				{"$match": match_stage},
				{"$project": {
					"type": {"$literal": "vocal"},
					"Time": 1,
					"IsPublic": 1,
					"Views": 1,
					"filename": 1,
					"img": 1,
					"_id": 1
				}},
				{"$sort": {"Time": -1}},
				{"$skip": (index - 1) * limit},
				{"$limit": limit}
			]
			result = Vocals.aggregate(pipeline)

		else:
			# Merge 2 collection
			music_match = {"user_id": user_obj_id}
			vocal_match = {"user_id": user_obj_id}
			if keyword:
				music_match["$or"] = [
					{"Instrument": regex},
					{"Name": regex},
					{"Time": regex}
				]
				vocal_match["$or"] = [
					{"filename": regex},
					{"Time": regex}
				]

			pipeline = [
				{"$match": music_match},
				{"$project": {
					"type": {"$literal": "musicxml"},
					"Time": 1,
					"IsPublic": 1,
					"Instrument": 1,
					"Views": 1,
					"Name": 1,
					"_id": 1
				}},
				{"$unionWith": {
					"coll": "Vocals",
					"pipeline": [
						{"$match": vocal_match},
						{"$project": {
							"type": {"$literal": "vocal"},
							"Time": 1,
							"filename": 1,
							"IsPublic": 1,
							"Views": 1,
							"img": 1,
							"_id": 1
						}},
					]
				}},
				{"$sort": {"Time": -1}},
				{"$skip": (index - 1) * limit},
				{"$limit": limit}
			]
			result = Sheet.aggregate(pipeline)

		final_data = [process_extra(item) for item in result]
		return final_data, index

	except Exception as e:
		print(f"[Get All User Data Error] {e}")
		return None


def get_all_public_data(kind, limit=20, index=1, keyword=None):
	try:
		amount_data = count_public_data(kind)
		if ((index - 1) * limit) + 1 > amount_data:
			index = amount_data // limit if amount_data % limit == 0 else amount_data // limit + 1

		regex = {"$regex": keyword.lower(), "$options": "i"} if keyword else None

		def process_extra(item):
			if item["type"] == "vocal":
				item["filename"] = item.get("filename", "")[:-4]
				if "img" in item:
					item["img_base64"] = base64.b64encode(item["img"]).decode("utf-8")
					del item["img"]

			if "avatar_url" in item:
				try:
					item["avatar_base64"] = base64.b64encode(item["avatar_url"]).decode("utf-8")
				except Exception:
					item["avatar_base64"] = None
				del item["avatar_url"]

			item["_id"] = str(item["_id"])
			item["user_id"] = str(item["user_id"])
			return item

		# ==== MUSICXML ONLY ====
		if kind == "musicxml":
			match_stage = {"IsPublic": True}
			if regex:
				match_stage["$or"] = [
					{"Instrument": regex},
					{"Name": regex},
					{"Time": regex}
				]
			pipeline = [
				{"$match": match_stage},
				{"$lookup": {
					"from": "Users",
					"localField": "user_id",
					"foreignField": "_id",
					"as": "user_info"
				}},
				{"$unwind": "$user_info"},
				{"$project": {
					"type": {"$literal": "musicxml"},
					"IsPublic": 1,
					"Instrument": 1,
					"Time": 1,
					"Name": 1,
					"Views": 1,
					"user_id": 1,
					"username": "$user_info.Username",
					"avatar_url": "$user_info.Avatar_url",
					"_id": 1
				}},
				{"$sort": {"Time": -1}},
				{"$skip": (index - 1) * limit},
				{"$limit": limit}
			]
			result = Sheet.aggregate(pipeline)

		# ==== VOCAL ONLY ====
		elif kind == "vocal":
			match_stage = {"IsPublic": True}
			if regex:
				match_stage["$or"] = [
					{"filename": regex},
					{"Time": regex}
				]
			pipeline = [
				{"$match": match_stage},
				{"$lookup": {
					"from": "Users",
					"localField": "user_id",
					"foreignField": "_id",
					"as": "user_info"
				}},
				{"$unwind": "$user_info"},
				{"$project": {
					"type": {"$literal": "vocal"},
					"IsPublic": 1,
					"filename": 1,
					"Time": 1,
					"Views": 1,
					"user_id": 1,
					"username": "$user_info.Username",
					"avatar_url": "$user_info.Avatar_url",
					"img": 1,
					"_id": 1
				}},
				{"$sort": {"Time": -1}},
				{"$skip": (index - 1) * limit},
				{"$limit": limit}
			]
			result = Vocals.aggregate(pipeline)

		# ==== ALL (MERGE 2 LOẠI) ====
		else:
			music_match = {"IsPublic": True}
			vocal_match = {"IsPublic": True}
			if regex:
				music_match["$or"] = [
					{"Instrument": regex},
					{"Name": regex},
					{"Time": regex}
				]
				vocal_match["$or"] = [
					{"filename": regex},
					{"Time": regex}
				]

			pipeline = [
				# Sheet
				{"$match": music_match},
				{"$project": {
					"type": {"$literal": "musicxml"},
					"IsPublic": 1,
					"Instrument": 1,
					"Time": 1,
					"Name": 1,
					"Views": 1,
					"user_id": 1,
					"_id": 1
				}},
				# Union Vocal
				{"$unionWith": {
					"coll": "Vocals",
					"pipeline": [
						{"$match": vocal_match},
						{"$project": {
							"type": {"$literal": "vocal"},
							"IsPublic": 1,
							"filename": 1,
							"Time": 1,
							"Views": 1,
							"user_id": 1,
							"img": 1,
							"_id": 1
						}}
					]
				}},
				# Join user
				{"$lookup": {
					"from": "Users",
					"localField": "user_id",
					"foreignField": "_id",
					"as": "user_info"
				}},
				{"$unwind": "$user_info"},
				{"$project": {
					"type": 1,
					"IsPublic": 1,
					"Time": 1,
					"Views": 1,
					"user_id": 1,
					"username": "$user_info.Username",
					"avatar_url": "$user_info.Avatar_url",
					"Instrument": 1,
					"Name": 1,
					"filename": 1,
					"img": 1,
					"_id": 1
				}},
				{"$sort": {"Time": -1}},
				{"$skip": (index - 1) * limit},
				{"$limit": limit}
			]

			result = Sheet.aggregate(pipeline)

		final_data = [process_extra(item) for item in result]
		return final_data, index

	except Exception as e:
		print(f"[Get All Public Data Error] {e}")
		return [], index


def get_file_by_kind_and_id(kind, item_id):
	try:
		if kind == "vocal":
			collection = Vocals
		elif kind == "sheet":
			collection = Sheet
		else:
			return False

		item = collection.find_one({"_id": ObjectId(item_id)})
		if not item:
			return False

		# Convert ObjectId về string để trả JSON
		item["_id"] = str(item["_id"])
		item["user_id"] = str(item["user_id"])
		if kind == "vocal":
			if "img" in item and item["img"]:
				img_base64 = base64.b64encode(item["img"]).decode("utf-8")
				item["img"] = f"data:image/png;base64,{img_base64}"

		return item

	except PyMongoError as e:
		print(f"[MongoDB Error] {e}")
		return False
	except Exception as e:
		print(f"[Unknown Error] {e}")
		return False


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
