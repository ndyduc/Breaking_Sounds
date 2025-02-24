import torch.nn.functional as F
import torch.nn as nn
import torch

import librosa
import numpy as np
import pandas as pd
import torch.utils.data as data


class CNN_Pro(nn.Module):
    def __init__(self, input_shape=(128, 43), num_classes=129):
        super(CNN_Pro, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5), padding=(2, 2))  # Kernel lớn cho đặc trưng tần số
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(256)

        # Adaptive pooling để linh hoạt với kích thước đầu vào
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Giảm xuống 4x4

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

        # Lưu kích thước đầu vào
        self.input_shape = input_shape

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))

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


def load_wav_csv(wav_path, csv_path, sr=44100, hop_length=512):
    y, sr = librosa.load(wav_path, sr=sr)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    df = pd.read_csv(csv_path)

    def time_to_frame(time, sr, hop_length):
        return int((time / 1000000) * sr // hop_length)

    def sample_to_frame(sample, hop_length):
        return sample // hop_length

    df['start_frame'] = df['start_time'].apply(lambda x: sample_to_frame(x, hop_length))
    df['end_frame'] = df['end_time'].apply(lambda x: sample_to_frame(x, hop_length))

    # Gán nhãn cho từng frame
    y_train = np.full(mel_spec.shape[1], -1)  # Mặc định không có nốt (-1)
    for _, row in df.iterrows():
        start, end, note = row['start_frame'], row['end_frame'], row['note']
        y_train[start:end] = note

    # Thay thế -1 bằng 128 (nhãn "không có nốt")
    y_train[y_train == -1] = 128

    def create_windows_torch(X, y, window_size=128, step=64):
        X_windows, y_windows = [], []
        for i in range(0, X.shape[1] - window_size, step):
            X_windows.append(X[:, i:i + window_size])
            center_index = i + window_size // 2
            if center_index < len(y):
                y_windows.append(y[center_index])

        # Gộp list thành numpy array trước khi chuyển sang tensor
        return torch.tensor(np.array(X_windows), dtype=torch.float32), torch.tensor(np.array(y_windows),
                                                                                    dtype=torch.long)

    X_train, y_train = create_windows_torch(mel_spec_db, y_train)

    # Thêm kênh cho Conv2D
    X_train = X_train.unsqueeze(1)  # (batch, 1, 128, 128)

    return X_train, y_train


def preprocess_wav(wav_path, sr=44100, hop_length=512):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    y, sr = librosa.load(wav_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Chia cửa sổ giống lúc huấn luyện
    def create_windows_torch(X, window_size=128, step=64):
        X_windows = []
        frame_indices = []
        for i in range(0, X.shape[1] - window_size, step):
            X_windows.append(X[:, i:i + window_size])
            frame_indices.append(i + window_size // 2)  # Lưu chỉ số frame

        return torch.tensor(np.array(X_windows), dtype=torch.float32), frame_indices

    X_test, frame_indices = create_windows_torch(mel_spec_db)
    X_test = X_test.unsqueeze(1).to(device)  # Thêm kênh cho Conv2D

    return X_test, frame_indices, sr, hop_length


def predict_notes(wav_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = CNN_Pro().to(device)
    X_test, frame_indices, sr, hop_length = preprocess_wav(wav_path)

    with torch.no_grad():
        outputs = model(X_test)
        predicted_notes = torch.argmax(outputs, dim=1).cpu().numpy()

    print(f"Predicted notes: {predicted_notes[:10]}")

    # Chuyển frame index về thời gian (microseconds)
    def frame_to_time(frame_idx, sr, hop_length):
        return int((frame_idx * hop_length / sr) * 1e6)

    results = []
    for i in range(len(predicted_notes) - 1):
        start_beat = frame_to_time(frame_indices[i], sr, hop_length)
        end_beat = frame_to_time(frame_indices[i + 1], sr, hop_length)
        note = predicted_notes[i]
        if note != 128:  # 128 là nhãn không có nốt
            results.append((start_beat, end_beat, note))

    return results