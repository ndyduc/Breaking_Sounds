from flask import Flask, render_template, request
import librosa
import numpy as np
import os

app = Flask(__name__, template_folder='templates')

# Cấu hình thư mục lưu trữ tệp tải lên
UPLOAD_FOLDER = '/Users/macbook/Downloads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Hàm xác định nốt nhạc từ tần số
def frequency_to_note(frequency):
    # Dùng librosa để lấy tần số và chuyển đổi thành nốt nhạc
    note = librosa.hz_to_note(frequency)
    return note


# Hàm phân tích nốt nhạc từ file MP3
def analyze_note_from_mp3(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # Sử dụng piptrack để xác định các tần số chủ đạo
    D, phase = librosa.core.piptrack(y=y, sr=sr)
    index = D[:, :].argmax(axis=0)  # Tìm vị trí của tần số mạnh nhất
    peak_frequency = D[index]  # Tần số mạnh nhất

    # Chuyển tần số thành nốt nhạc
    note = frequency_to_note(peak_frequency)
    return note


# Route cho trang chủ để tải tệp lên
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Kiểm tra nếu người dùng đã chọn tệp
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'

        # Lưu tệp vào thư mục uploads
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Phân tích nốt nhạc
        note = analyze_note_from_mp3(filepath)

        # In kết quả phân tích vào console
        print(f"Phân tích nốt nhạc: {note}")

        return f"Đã phân tích nốt nhạc. Kiểm tra console."

    return render_template('Main_home.html')


if __name__ == '__main__':
    app.run(debug=True)