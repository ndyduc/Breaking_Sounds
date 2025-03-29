import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import torch
import librosa.feature
import numpy as np


class CNN_3L(nn.Module):
    def __init__(self, num_classes=129):
        super(CNN_3L, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

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


import numpy as np
import pandas as pd
import librosa


def load_wav_csv(wav_path, csv_path, sr=44100, hop_length=512):
    y, sr = librosa.load(wav_path, sr=sr)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    df = pd.read_csv(csv_path)

    def sample_to_frame(sample, hop_length):
        return sample // hop_length

    df['start_frame'] = df['start_time'].apply(lambda x: sample_to_frame(x, hop_length))
    df['end_frame'] = df['end_time'].apply(lambda x: sample_to_frame(x, hop_length))

    num_frames = mel_spec_db.shape[1]

    # Dùng -1 làm mặc định (nếu không có nhãn)
    y_train = np.full(num_frames, -1, dtype=np.int64)

    # Gán nhãn cho từng frame, chỉ chọn một nốt duy nhất
    for _, row in df.iterrows():
        start = max(0, row['start_frame'])
        end = min(num_frames, row['end_frame'])
        note = int(row['note'])

        # Chỉ gán nốt cho những frame chưa có nhãn
        for i in range(start, end):
            if y_train[i] == -1:
                y_train[i] = note

    # Những frame chưa có nhãn gán thành 0 (không có nốt nhạc)
    y_train[y_train == -1] = 0

    def create_windows_torch(X, y, window_size=128, step=21):
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
    unique, counts = np.unique(y_train.numpy(), return_counts=True)
    # print(dict(zip(unique, counts)))
    # Thêm kênh cho Conv2D
    X_train = X_train.unsqueeze(1)  # (batch, 1, 128, 128)
    return X_train, y_train


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        if epoch >= num_epochs - 1:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, "
                  f"Accuracy: {100 * correct / total:.2f}%")


def predict_notes(model, device, wav_path, sr=44100, hop_length=512, window_size=128, step=64):
    y, sr = librosa.load(wav_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    print("Mel spectrogram shape:", mel_spec_db.shape)
    print("Max value:", np.max(mel_spec_db), "Min value:", np.min(mel_spec_db))

    # Chia nhỏ spectrogram thành các đoạn như khi huấn luyện
    def create_windows(X, window_size, step):
        X_windows = []
        indices = []
        for i in range(0, X.shape[1] - window_size, step):
            X_windows.append(X[:, i:i + window_size])
            indices.append(i)
        return np.array(X_windows), indices

    X_test, indices = create_windows(mel_spec_db, window_size, step)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)

    # Dự đoán
    with torch.no_grad():
        outputs = model(X_test)
        predicted_notes = torch.argmax(outputs, dim=1).cpu().numpy()

    # Chuyển frame thành thời gian
    def frame_to_time(frame, sr, hop_length):
        return (frame * hop_length) / sr

    results = []
    prev_note = None
    start_time = None

    for idx, note in zip(indices, predicted_notes):
        time = frame_to_time(idx, sr, hop_length)

        if note != prev_note:
            if prev_note is not None and prev_note != 128:  # Lưu nốt trước đó nếu nó không phải "không có nốt"
                results.append((prev_note, start_time, time))
            if note != 128:  # Khi đổi sang nốt mới, cập nhật start_time nếu không phải "không có nốt"
                start_time = time
            prev_note = note

            # Nếu file kết thúc với một nốt đang chơi, lưu lại nó
    if prev_note is not None and prev_note != 128:
        results.append((prev_note, start_time, time))

    return results
