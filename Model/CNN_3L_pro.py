import torch.nn.functional as F
import torch.nn as nn
import torch

import librosa.feature
import numpy as np
import pandas as pd
import torch.utils.data as data


class CNN_Pro(nn.Module):
    def __init__(self, input_shape=(128, 43), num_classes=128):
        super(CNN_Pro, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(7, 7), padding=(3, 3))  # Kernel lớn cho đặc trưng tần số
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5), padding=(2, 2))
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)

        # Adaptive pooling để linh hoạt với kích thước đầu vào
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.33)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        if x.device.type == "mps":  # Chỉ chuyển sang CPU nếu đang chạy trên MPS
            x = x.to("cpu")
            x = self.adaptive_pool(x)
            x = x.to("mps")
        else:
            x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class MusicDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_wav_csv(wav_path, csv_path=None, sr=44100, hop_length=512):
    y, sr = librosa.load(wav_path, sr=sr)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    if csv_path is None:
        return torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0)  # (1, 128, time)

    df = pd.read_csv(csv_path)

    def sample_to_frame(sample, hop_length):
        return sample // hop_length

    df['start_frame'] = df['start_time'].apply(lambda x: sample_to_frame(x, hop_length))
    df['end_frame'] = df['end_time'].apply(lambda x: sample_to_frame(x, hop_length))

    num_frames = mel_spec_db.shape[1]
    num_notes = 128
    y_train = np.zeros((num_frames, num_notes), dtype=np.float32)

    # Gán nhãn cho từng frame
    for _, row in df.iterrows():
        start = max(0, row['start_frame'])
        end = min(num_frames, row['end_frame'])
        note = row['note']
        y_train[start:end, note] = 1  # Đánh dấu các frames có nốt này

    def create_windows_torch(X, y, window_size=128, step=21):
        X_windows, y_windows = [], []
        for i in range(0, X.shape[1] - window_size, step):
            X_windows.append(X[:, i:i + window_size])
            center_index = i + window_size // 2
            if center_index < len(y):
                y_windows.append(y[center_index, :])

        return torch.tensor(np.array(X_windows), dtype=torch.float32), torch.tensor(np.array(y_windows), dtype=torch.long)

    X_train, y_train = create_windows_torch(mel_spec_db, y_train)

    X_train = X_train.unsqueeze(1)  # (batch, 1, 128, 128)
    return X_train, y_train


def check_csv():
    wav_path = "../Data/MusicNet_Dataset/musicnet/musicnet/train_data/1727.wav"
    y, sr = librosa.load(wav_path, sr=44100)  # Tải file WAV
    hop_length = 512
    # Tính số frame thực tế có trong file
    num_frames = len(y) // hop_length

    print(f"Sample rate (sr): {sr}")
    print(f"Số mẫu trong file WAV: {len(y)}")
    print(f"Số frame hợp lý (num_frames): {num_frames}")


def midi_to_note(midi_note):
    return librosa.midi_to_note(midi_note)


def predict_notes(wav_path, model, device="cpu", sr=44100, hop_length=512, window_size=128, step=21):
    X_windows = load_wav_csv(wav_path, None, sr=sr, hop_length=hop_length)
    X_windows = X_windows.to(device)

    batch_size = 32  # Giảm batch size để tránh hết RAM
    num_batches = len(X_windows) // batch_size + 1
    predictions = []

    with torch.no_grad():
        for i in range(num_batches):
            batch = X_windows[i * batch_size:(i + 1) * batch_size]
            if batch.shape[0] == 0:
                continue  # Bỏ qua batch rỗng cuối cùng
            batch = batch.unsqueeze(1)
            outputs = model(batch)
            predictions.append((outputs > 0.5).float().cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)  # Ghép lại thành một mảng

    # Xử lý duration của nốt nhạc
    frame_duration = step * hop_length / sr  # Khoảng thời gian của mỗi frame
    timestamps = [(i * step * hop_length / sr) for i in range(len(predictions))]

    active_notes = {}  # Lưu trạng thái nốt đang được phát
    note_events = []  # Lưu kết quả cuối cùng

    for i, (time, pred) in enumerate(zip(timestamps, predictions)):
        notes = np.where(pred == 1)[0]  # Lấy danh sách các nốt có giá trị 1

        new_active_notes = set(notes)  # Chuyển sang tập hợp để dễ kiểm tra

        # Kiểm tra nốt nào vẫn tiếp tục hoặc mới bắt đầu
        for note in new_active_notes:
            if note not in active_notes:
                active_notes[note] = {"start": time, "duration": frame_duration}
            else:
                active_notes[note]["duration"] += frame_duration

        # Kiểm tra nốt nào đã kết thúc
        ended_notes = set(active_notes.keys()) - new_active_notes
        for note in ended_notes:
            note_events.append({
                "note": midi_to_note(note),
                "start_time": active_notes[note]["start"],
                "duration": active_notes[note]["duration"]
            })
            del active_notes[note]

    # Ghi nhận các nốt còn sót lại (kết thúc ở frame cuối)
    for note, info in active_notes.items():
        note_events.append({
            "note": midi_to_note(note),
            "start_time": info["start"],
            "duration": info["duration"]
        })

    return note_events


def CNN_predict(wav_path, model, device="cpu", sr=44100, hop_length=512, window_size=64, step=21):
    print(f"\nĐang dự đoán file: {wav_path}")

    y, sr = librosa.load(wav_path, sr=sr)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Chia thành các cửa sổ (windows)
    X_windows = []
    timestamps = []
    frame_duration = step * hop_length / sr  # Khoảng thời gian của mỗi frame

    for i in range(0, mel_spec_db.shape[1] - window_size, step):
        X_windows.append(mel_spec_db[:, i:i + window_size])
        timestamps.append(i * hop_length / sr)  # Thời gian tính bằng giây

    # Chuyển thành tensor
    X_windows = torch.tensor(np.array(X_windows), dtype=torch.float32).unsqueeze(1).to(device)

    model.eval()

    batch_size = 32  # Có thể giảm xuống 16 nếu vẫn hết RAM
    num_batches = len(X_windows) // batch_size + 1
    predictions = []

    with torch.no_grad():
        for i in range(num_batches):
            batch = X_windows[i * batch_size:(i + 1) * batch_size]
            if batch.shape[0] == 0:
                continue
            outputs = model(batch)
            predictions.append((outputs > 0.5).float().cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)

    # Xử lý duration của nốt nhạc
    active_notes = {}  # Lưu trạng thái nốt đang được phát
    note_events = []  # Lưu kết quả cuối cùng

    for i, (time, pred) in enumerate(zip(timestamps, predictions)):
        notes = np.where(pred == 1)[0]  # Lấy danh sách các nốt có giá trị 1

        new_active_notes = set(notes)  # Chuyển sang tập hợp để dễ kiểm tra

        # Kiểm tra nốt nào vẫn tiếp tục hoặc mới bắt đầu
        for note in new_active_notes:
            if note not in active_notes:
                active_notes[note] = {"start": time, "duration": frame_duration}
            else:
                active_notes[note]["duration"] += frame_duration

        # Kiểm tra nốt nào đã kết thúc
        ended_notes = set(active_notes.keys()) - new_active_notes
        for note in ended_notes:
            note_events.append({
                "note": note,
                "start": active_notes[note]["start"],
                "duration": active_notes[note]["duration"]
            })
            del active_notes[note]

    # Ghi nhận các nốt còn sót lại (kết thúc ở frame cuối)
    for note, info in active_notes.items():
        note_events.append({
            "note": note,
            "start": info["start"],
            "duration": info["duration"]
        })
    # print(note_events)
    return note_events
    # for idx, event in enumerate(sorted(note_events, key=lambda x: x["start"])):
    #     print(f"{idx+1}. 🎵 Note {event['note']} - Start: {event['start']:.2f}s, Duration: {event['duration']:.2f}s")
