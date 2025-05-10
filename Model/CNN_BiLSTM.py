import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torchaudio.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import librosa.feature
import matplotlib.pyplot as plt
import glob
import os


import numpy as np
import librosa
import torch


def load_wav_csv(wav_path, csv_path=None, sr=44100, hop_length=512, window_size=128, step=11):
    """Args:
        wav_path (str): Đường dẫn đến file WAV.
        csv_path (str, optional): Đường dẫn đến file CSV chứa nhãn. Nếu không có, chỉ xử lý WAV.
        sr (int): Sample rate cần chuẩn hóa.
        hop_length (int): Khoảng cách giữa các khung (frame).
        window_size (int): Kích thước cửa sổ trích xuất.
        step (int): Bước nhảy giữa các cửa sổ.

    Returns:
        X_train (Tensor): Dữ liệu đầu vào có shape (batch, 1, 128, window_size).
        y_notes (Tensor hoặc None): Nhãn nốt nhạc có shape (batch, 128) nếu có CSV, ngược lại là None.
    """
    # Load WAV file
    y, sr = librosa.load(wav_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    if csv_path is None:
        return torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0), None

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError("File CSV không có dữ liệu.")
    except Exception as e:
        raise FileNotFoundError(f"Lỗi khi đọc file CSV: {e}")

    def sample_to_frame(sample, hop_length):
        return sample // hop_length

    df['start_frame'] = df['start_time'].apply(lambda x: sample_to_frame(x, hop_length))
    df['end_frame'] = df['end_time'].apply(lambda x: sample_to_frame(x, hop_length))

    num_frames = mel_spec_db.shape[1]
    num_notes = 128  # MIDI notes từ 1-128

    y_notes = np.zeros((num_frames, num_notes), dtype=np.float32)

    for _, row in df.iterrows():
        start = min(max(0, row['start_frame']), num_frames - 1)
        end = min(max(0, row['end_frame']), num_frames)
        note = int(row['note'])
        if 1 <= note <= num_notes:
            y_notes[start:end, note - 1] = 1  # Chỉnh về index từ 0-127

    def create_windows_torch(X, y, window_size=128, step=21):
        X_windows, y_windows = [], []
        for i in range(0, X.shape[1] - window_size, step):
            X_windows.append(X[:, i:i + window_size])
            center_index = i + window_size // 2
            if center_index < len(y):
                y_windows.append(y[center_index, :])

        X_windows = torch.tensor(np.array(X_windows), dtype=torch.float32)
        y_windows = torch.tensor(np.array(y_windows), dtype=torch.float32)
        return X_windows, y_windows

    X_train, y_notes = create_windows_torch(mel_spec_db, y_notes, window_size, step)
    return X_train, y_notes


class CNN_BiLSTM(nn.Module):
    def __init__(self, input_channels=1, cnn_filters=32, lstm_hidden=128, num_layers=2, num_notes=128):
        super(CNN_BiLSTM, self).__init__()

        # CNN Layers
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=cnn_filters, kernel_size=(5, 5), padding=2)
        self.pool = nn.MaxPool2d((2, 2))
        self.batchnorm1 = nn.BatchNorm2d(cnn_filters)

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=cnn_filters * (128 // 2),  # Sau MaxPool2d (2,2), kích thước giảm 1/2
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Fully Connected Layer cho Note Prediction
        self.fc_notes = nn.Linear(lstm_hidden * 2, num_notes)

    def forward(self, x):
        x = x.unsqueeze(1)
        # CNN
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = self.pool(x)  # (batch, 32, 64, 64) nếu filters = 32
        # Reshape cho LSTM
        x = x.permute(0, 3, 1, 2)  # (batch, time_steps, channel, feature_dim)
        batch, time_steps, channel, feature_dim = x.shape
        x = x.view(batch, time_steps, feature_dim * channel)  # (batch, time_steps, feature_dim)
        # LSTM
        x, _ = self.lstm(x)
        notes_out = torch.sigmoid(self.fc_notes(x[:, -1, :]))  # Lấy output frame cuối cùng

        return notes_out


def train_on(model, optimizer, criterion, wav_path, csv_path, test_wav_files, test_csv_files, batch_size=32, num_epochs=10):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Load dữ liệu train
    X_train, y_notes_train = load_wav_csv(wav_path, csv_path)  # Chỉ nhận 2 giá trị
    train_dataset = TensorDataset(X_train, y_notes_train)

    # Load dữ liệu validation
    val_datasets = [load_wav_csv(w, c) for w, c in zip(test_wav_files, test_csv_files)]
    X_val, y_notes_val = map(torch.cat, zip(*val_datasets))
    val_dataset = TensorDataset(X_val, y_notes_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        correct_notes = 0
        total_notes = 0

        for X_batch, y_notes_batch in train_loader:
            X_batch, y_notes_batch = X_batch.to(device).float(), y_notes_batch.to(device).float()

            optimizer.zero_grad()
            notes_pred = model(X_batch)  # Xóa giá trị thứ hai `_`

            loss_notes = criterion(notes_pred, y_notes_batch)
            loss_notes.backward()
            optimizer.step()

            total_train_loss += loss_notes.item()

            predicted_notes = (notes_pred > 0.3).float()
            correct_notes += (predicted_notes == y_notes_batch).sum().item()
            total_notes += y_notes_batch.numel()

        train_acc_notes = correct_notes / total_notes

        # Validation
        model.eval()
        total_val_loss, correct_notes, total_notes = 0, 0, 0
        all_true_notes, all_pred_notes = [], []

        with torch.no_grad():
            for X_batch, y_notes_batch in val_loader:
                X_batch, y_notes_batch = X_batch.to(device).float(), y_notes_batch.to(device).float()

                notes_pred = model(X_batch)  # Xóa giá trị thứ hai `_`

                loss_notes = criterion(notes_pred, y_notes_batch)
                total_val_loss += loss_notes.item()

                predicted_notes = (notes_pred > 0.1).float()
                correct_notes += (predicted_notes == y_notes_batch).sum().item()
                total_notes += y_notes_batch.numel()

                all_true_notes.append(y_notes_batch.cpu().numpy().flatten())
                all_pred_notes.append(predicted_notes.cpu().numpy().flatten())

        val_acc_notes = correct_notes / total_notes

        all_true_notes = np.concatenate(all_true_notes)
        all_pred_notes = np.concatenate(all_pred_notes)

        precision_notes = precision_score(all_true_notes, all_pred_notes, average="macro", zero_division=1)
        recall_notes = recall_score(all_true_notes, all_pred_notes, average="macro")
        f1_notes = f1_score(all_true_notes, all_pred_notes, average="macro")

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {total_train_loss / len(train_loader):.4f}, "
              f"Train Acc: {train_acc_notes:.2f} - "
              f"Val Loss: {total_val_loss / len(val_loader):.4f}, "
              f"Val Acc: {val_acc_notes:.2f}")

        print(f" Precision: {precision_notes:.4f}, Recall: {recall_notes:.4f}, F1-score: {f1_notes:.4f}\n")

        if epoch == 9:
            # Vẽ histogram phân bố dự đoán
            X_batch, _ = next(iter(val_loader))
            X_batch = X_batch.to(device)
            with torch.no_grad():
                notes_pred = model(X_batch)

            notes_pred_np = notes_pred.detach().cpu().numpy().flatten()

            plt.hist(notes_pred_np, bins=50, alpha=0.7, label="Notes Predictions")
            plt.xlabel("Prediction Values")
            plt.ylabel("Frequency")
            plt.legend()
            plt.show()

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, "../Trained/CNN_BiLSTM11.pth")


def predict(model, wav_path, device, hop_length=512, sr=44100, window_size=128, step=11):
    """Args:
        model: Mô hình PyTorch đã được huấn luyện.
        wav_path: Đường dẫn đến file WAV.
        device: Thiết bị tính toán (cpu hoặc cuda).
        hop_length (int): Khoảng cách giữa các frames.
        sr (int): Sample rate.
        window_size (int): Kích thước cửa sổ xử lý.
        step (int): Bước nhảy giữa các cửa sổ.

    Returns:
        List[Dict[str, Union[int, float]]]: Danh sách các nốt [{"note", "start", "duration"}].
    """
    model.to(device)
    model.eval()

    wav_tensor, _ = load_wav_csv(wav_path)  # Chỉ lấy WAV, không cần nhãn

    wav_tensor = wav_tensor.to(device)
    num_frames = wav_tensor.shape[2]

    y_notes_list = []
    with torch.no_grad():
        for i in range(0, num_frames - window_size, step):
            window = wav_tensor[:, :, i:i + window_size]
            y_notes = model(window)  # Output shape: (1, 128)
            y_notes_list.append(y_notes.squeeze(0).cpu().numpy())

    y_notes = np.vstack(y_notes_list)  # Ghép lại thành (num_frames, 128)

    notes_list = []
    active_notes = {}

    for frame_idx, frame in enumerate(y_notes):
        # Tính thời gian thực của frame hiện tại
        current_sample = (frame_idx * step + window_size // 2) * hop_length
        current_time = current_sample / sr  # Chuyển đổi sang giây

        for note, is_active in enumerate(frame):
            if is_active > 0.3:
                if note not in active_notes:
                    active_notes[note] = current_time
            else:
                if note in active_notes:
                    start_time = active_notes.pop(note)
                    duration = current_time - start_time
                    notes_list.append({
                        "note": note + 1,
                        "start": start_time,
                        "duration": duration
                    })

    last_sample = (num_frames * step + window_size // 2) * hop_length
    last_time = last_sample / sr

    for note, start_time in active_notes.items():
        duration = last_time - start_time
        notes_list.append({
            "note": note + 1,
            "start": start_time,
            "duration": duration
        })

    notes_list.sort(key=lambda x: x["start"])

    return notes_list


