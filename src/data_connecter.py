from flask import Flask, jsonify
from pymongo import MongoClient
from bson.objectid import ObjectId
from pymongo.errors import DuplicateKeyError, PyMongoError
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import os

try:
    load_dotenv()
    mongo_uri = os.getenv('MONGO_ATLAS')
    client = MongoClient(mongo_uri)
    db = client['ndyduc']
    Users = db['Users']

    client.server_info()  # Kiểm tra kết nối
    connection_status = True
    print("Kết nối MongoDB thành công!")

except Exception as e:
    print(f"Không thể kết nối đến MongoDB: {e}")
    connection_status = False


def Check_connect():
    if not connection_status:
        return jsonify({"status": "error", "message": "Không thể kết nối đến MongoDB"}), 500

    # Kiểm tra dữ liệu trong collection Users
    data = list(Users.find({}, {"_id": 0}))  # Ẩn `_id` trong kết quả
    if data:
        return jsonify({"status": "success", "rows": len(data), "data": data})
    else:
        return jsonify({"status": "success", "message": "empty"})


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


def get_user(id_google=None, email=None):
    try:
        query = {}
        if id_google:
            query["ID_google"] = id_google
        if email:
            query["Email"] = email

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


def update_user(user_id, username=None, email=None, password=None, avatar_url=None, locale=None):
    if not ObjectId.is_valid(user_id): return {"error": "Invalid ObjectId"}

    update_fields = {
        "Username": username,
        "Email": email,
        "Password": generate_password_hash(password) if password else None,
        "Avatar_url": avatar_url,
        "Locale": locale
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

    return {"success": "User updated successfully"}
