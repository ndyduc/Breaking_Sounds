import librosa
import librosa.feature
import numpy as np
import torch.nn as nn
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def remove_silent_frames_cqt(cqt, y_notes, y_instr, y, sr, hop_length, threshold=0.01):

    rms = librosa.feature.rms(y=y, frame_length=hop_length * 2, hop_length=hop_length)[0]  # Tính RMS
    mask = rms > threshold  # Chỉ giữ lại frame có năng lượng cao hơn ngưỡng

    cqt_filtered = cqt[:, mask]
    y_notes_filtered = y_notes[mask]
    y_instr_filtered = y_instr[mask]

    return cqt_filtered, y_notes_filtered, y_instr_filtered


def load_wav_csv(wav_path, csv_path, sr=44100, hop_length=512, n_mels=128, n_mfcc=40, bins_per_octave=24):
    y, sr = librosa.load(wav_path, sr=sr)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, bins_per_octave=bins_per_octave)
    cqt = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    # Chuẩn hóa kích thước (cắt hoặc padding)
    min_frames = min(mel_spec.shape[1], mfcc.shape[1], cqt.shape[1])
    mel_spec = mel_spec[:, :min_frames]
    mfcc = mfcc[:, :min_frames]
    cqt = cqt[:, :min_frames]

    # Chuẩn hóa giá trị về [0,1]
    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-6)
    mfcc = (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min() + 1e-6)
    cqt = (cqt - cqt.min()) / (cqt.max() - cqt.min() + 1e-6)

    # Ghép các feature theo chiều dọc (feature_dim, time_frames)
    X = np.vstack([mel_spec, mfcc, cqt])

    # Load CSV nhãn
    df = pd.read_csv(csv_path)

    # Tạo frame labels
    num_frames = X.shape[1]
    num_notes = 128
    num_instr = 128
    y_notes = np.zeros((num_frames, num_notes), dtype=np.float32)
    y_instr = np.zeros((num_frames, num_instr), dtype=np.float32)

    for _, row in df.iterrows():
        start = max(0, row['start_time'] // hop_length)
        end = min(num_frames, row['end_time'] // hop_length)
        note = row['note']
        instr = row['instrument']

        y_notes[start:end, note] = 1
        y_instr[start:end, instr] = 1

    # Chia cửa sổ dữ liệu
    def create_windows_torch(X, y1, y2, window_size=128, step=21):
        X_windows, y_notes_windows, y_instr_windows = [], [], []
        for i in range(0, X.shape[1] - window_size, step):
            X_windows.append(X[:, i:i + window_size])
            y_notes_windows.append(y1[i:i + window_size, :])
            y_instr_windows.append(y2[i:i + window_size, :])

        return (
            torch.tensor(np.array(X_windows), dtype=torch.float32),
            torch.tensor(np.array(y_notes_windows), dtype=torch.float32),
            torch.tensor(np.array(y_instr_windows), dtype=torch.float32),
        )

    X_train, y_notes, y_instr = create_windows_torch(X, y_notes, y_instr)
    X_train = X_train.permute(0, 2, 1)
    return X_train, y_notes, y_instr


class BiLSTM(nn.Module):
    def __init__(self, input_size=252, hidden_dim=256, num_layers=2, num_notes=128, num_instruments=128):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=0.1)
        self.fc_notes = nn.Linear(hidden_dim * 2, num_notes)
        self.fc_instr = nn.Linear(hidden_dim * 2, num_instruments)

    def forward(self, x):
        x, _ = self.lstm(x)
        notes_out = torch.sigmoid(self.fc_notes(x))
        instr_out = torch.sigmoid(self.fc_instr(x))
        return notes_out, instr_out


def train_on_wav(model, optimizer, criterion, wav_path, csv_path, epochs=10, batch_size=128):
    print(f"Training on {wav_path} ...")

    X, y_notes, y_instr = load_wav_csv(wav_path, csv_path)

    # Chia dữ liệu thành train (80%) và validation (20%)
    X_train, X_val, y_notes_train, y_notes_val, y_instr_train, y_instr_val = train_test_split(
        X, y_notes, y_instr, test_size=0.2, random_state=42
    )

    # Tạo DataLoader cho train và validation
    train_dataset = TensorDataset(X_train, y_notes_train, y_instr_train)
    val_dataset = TensorDataset(X_val, y_notes_val, y_instr_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.train()
    for epoch in range(epochs):
        total_train_loss = 0
        for X_batch, y_notes_batch, y_instr_batch in train_loader:
            X_batch, y_notes_batch, y_instr_batch = X_batch.to(device), y_notes_batch.to(device), y_instr_batch.to(device)

            optimizer.zero_grad()
            notes_pred, instr_pred = model(X_batch)

            loss_notes = criterion(notes_pred, y_notes_batch)
            loss_instr = criterion(instr_pred, y_instr_batch)

            loss = loss_notes + loss_instr
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Tính validation loss
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val_batch, y_notes_val_batch, y_instr_val_batch in val_loader:
                X_val_batch, y_notes_val_batch, y_instr_val_batch = X_val_batch.to(device), y_notes_val_batch.to(device), y_instr_val_batch.to(device)

                notes_val_pred, instr_val_pred = model(X_val_batch)
                loss_notes_val = criterion(notes_val_pred, y_notes_val_batch)
                loss_instr_val = criterion(instr_val_pred, y_instr_val_batch)

                total_val_loss += (loss_notes_val + loss_instr_val).item()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_train_loss:.4f}, Val Loss: {total_val_loss:.4f}")

        model.train()

    torch.save(model.state_dict(), "../Trained/BiLSTM.pth")


def predict_note(wav_path, model, devide, hoplength=512, sr=44100):
    pass