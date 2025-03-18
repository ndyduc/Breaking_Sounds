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


class CNN_BiLSTM(nn.Module):
    def __init__(self, input_channels=1, cnn_filters=32, lstm_hidden=256, num_layers=2, num_notes=127, num_instruments=127):
        super(CNN_BiLSTM, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=cnn_filters, kernel_size=(5, 5), padding=1)
        self.conv2 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters * 2, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d((2, 2))
        self.batchnorm1 = nn.BatchNorm2d(cnn_filters)
        self.batchnorm2 = nn.BatchNorm2d(cnn_filters * 2)

        self.lstm = nn.LSTM(
            input_size=1984,  # Sau khi qua CNN, feature_dim sẽ giảm
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )

        # Fully Connected Layers
        self.fc_notes = nn.Linear(lstm_hidden * 2, num_notes)
        self.fc_instr = nn.Linear(lstm_hidden * 2, num_instruments)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, 128, 128)

        # CNN
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = self.pool(x)

        # Chuẩn bị input cho LSTM
        x = x.permute(0, 3, 1, 2)  # (batch, time_steps, channel, feature_dim)
        batch, time_steps, channel, feature_dim = x.shape
        x = x.view(batch, time_steps, feature_dim * channel)  # (batch, time_steps, feature_dim)

        # LSTM
        x, _ = self.lstm(x)

        notes_out = torch.sigmoid(self.fc_notes(x[:, -1, :]))
        instr_out = torch.sigmoid(self.fc_instr(x[:, -1, :]))

        return notes_out, instr_out


def train(model, optimizer, criterion, wav_path, csv_path, test_wav_files, test_csv_files, batch_size=32, num_epochs=10, device="cpu"):
    device = torch.device(device)
    model.to(device)

    # Load toàn bộ dữ liệu train và validation
    X_train, y_notes_train, y_instr_train = load_wav_csv(wav_path, csv_path)
    train_dataset = TensorDataset(X_train, y_notes_train, y_instr_train)

    val_datasets = [load_wav_csv(w, c) for w, c in zip(test_wav_files, test_csv_files)]
    X_val, y_notes_val, y_instr_val = map(torch.cat, zip(*val_datasets))
    val_dataset = TensorDataset(X_val, y_notes_val, y_instr_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        correct_notes, correct_instr = 0, 0
        total_notes, total_instr = 0, 0

        for X_batch, y_notes_batch, y_instr_batch in train_loader:
            X_batch, y_notes_batch, y_instr_batch = (X_batch.to(device).float(),
                                                     y_notes_batch.to(device).float(),
                                                     y_instr_batch.to(device).float())

            optimizer.zero_grad()
            notes_pred, instr_pred = model(X_batch)

            loss_notes = criterion(notes_pred, y_notes_batch)
            loss_instr = criterion(instr_pred, y_instr_batch)
            loss = loss_notes + loss_instr

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            predicted_notes = (notes_pred > 0.3).float()
            predicted_instr = (instr_pred > 0.3).float()

            correct_notes += (predicted_notes == y_notes_batch).sum().item()
            correct_instr += (predicted_instr == y_instr_batch).sum().item()

            total_notes += y_notes_batch.numel()
            total_instr += y_instr_batch.numel()

        train_acc_notes = correct_notes / total_notes
        train_acc_instr = correct_instr / total_instr

        # Validation
        model.eval()
        total_val_loss = 0
        correct_notes, correct_instr = 0, 0
        total_notes, total_instr = 0, 0
        all_true_notes, all_pred_notes = [], []
        all_true_instr, all_pred_instr = [], []

        with torch.no_grad():
            for X_batch, y_notes_batch, y_instr_batch in val_loader:
                X_batch, y_notes_batch, y_instr_batch = (X_batch.to(device).float(),
                                                         y_notes_batch.to(device).float(),
                                                         y_instr_batch.to(device).float())

                notes_pred, instr_pred = model(X_batch)

                loss_notes = criterion(notes_pred, y_notes_batch)
                loss_instr = criterion(instr_pred, y_instr_batch)
                loss = loss_notes + loss_instr

                total_val_loss += loss.item()

                predicted_notes = (notes_pred > 0.3).float()
                predicted_instr = (instr_pred > 0.3).float()

                correct_notes += (predicted_notes == y_notes_batch).sum().item()
                correct_instr += (predicted_instr == y_instr_batch).sum().item()

                total_notes += y_notes_batch.numel()
                total_instr += y_instr_batch.numel()

                all_true_notes.append(y_notes_batch.cpu().numpy().flatten())
                all_pred_notes.append(predicted_notes.cpu().numpy().flatten())
                all_true_instr.append(y_instr_batch.cpu().numpy().flatten())
                all_pred_instr.append(predicted_instr.cpu().numpy().flatten())

        val_acc_notes = correct_notes / total_notes
        val_acc_instr = correct_instr / total_instr

        all_true_notes = np.concatenate(all_true_notes)
        all_pred_notes = np.concatenate(all_pred_notes)
        all_true_instr = np.concatenate(all_true_instr)
        all_pred_instr = np.concatenate(all_pred_instr)

        precision_notes = precision_score(all_true_notes, all_pred_notes, average="macro", zero_division=1)
        recall_notes = recall_score(all_true_notes, all_pred_notes, average="macro")
        f1_notes = f1_score(all_true_notes, all_pred_notes, average="macro")

        precision_instr = precision_score(all_true_instr, all_pred_instr, average="macro", zero_division=1)
        recall_instr = recall_score(all_true_instr, all_pred_instr, average="macro")
        f1_instr = f1_score(all_true_instr, all_pred_instr, average="macro")

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {total_train_loss / len(train_loader):.4f}, "
              f"Notes Acc: {train_acc_notes:.2f}, Instr Acc: {train_acc_instr:.2f} - "
              f"Val Loss: {total_val_loss / len(val_loader):.4f}, "
              f"Notes Acc: {val_acc_notes:.2f}, Instr Acc: {val_acc_instr:.2f}")

        print(f" Precision - Notes: {precision_notes:.4f}, Instr: {precision_instr:.4f}")
        print(f" Recall    - Notes: {recall_notes:.4f}, Instr: {recall_instr:.4f}")
        print(f" F1-score  - Notes: {f1_notes:.4f}, Instr: {f1_instr:.4f}\n")

        # Vẽ histogram
        X_batch, _, _ = next(iter(val_loader))
        X_batch = X_batch.to(device)
        with torch.no_grad():
            notes_pred, instr_pred = model(X_batch)

        notes_pred_np = notes_pred.detach().cpu().numpy().flatten()
        instr_pred_np = instr_pred.detach().cpu().numpy().flatten()

        plt.hist(notes_pred_np, bins=50, alpha=0.7, label="Notes Predictions")
        plt.hist(instr_pred_np, bins=50, alpha=0.7, label="Instruments Predictions")
        plt.xlabel("Prediction Values")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

    # Lưu checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, "../Trained/CNN_BiLSTM_demo.pth")


def load_wav_csv(wav_path, csv_path, sr=44100, hop_length=512, window_size=128, step=21):
    """
    Load file WAV, trích xuất Mel Spectrogram và đọc nhãn từ CSV.

    Args:
        wav_path (str): Đường dẫn đến file WAV.
        csv_path (str): Đường dẫn đến file CSV chứa nhãn.
        sr (int): Sample rate cần chuẩn hóa.
        hop_length (int): Khoảng cách giữa các khung.
        window_size (int): Kích thước cửa sổ trích xuất.
        step (int): Bước nhảy giữa các cửa sổ.

    Returns:
        X_train (Tensor): Dữ liệu đầu vào có shape (batch, 1, 128, 128).
        y_notes (Tensor): Nhãn nốt nhạc có shape (batch, 128).
        y_instr (Tensor): Nhãn nhạc cụ có shape (batch, 128).
    """
    y, sr = librosa.load(wav_path, sr=sr)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    df = pd.read_csv(csv_path)

    def sample_to_frame(sample, hop_length):
        return sample // hop_length

    # Chuyển đổi thời gian sang chỉ số frame
    df['start_frame'] = df['start_time'].apply(lambda x: sample_to_frame(x, hop_length))
    df['end_frame'] = df['end_time'].apply(lambda x: sample_to_frame(x, hop_length))

    num_frames = mel_spec_db.shape[1]
    num_notes = 127
    num_instr = 127

    y_notes = np.zeros((num_frames, num_notes), dtype=np.float32)
    y_instr = np.zeros((num_frames, num_instr), dtype=np.float32)

    # Gán nhãn cho từng frame
    for _, row in df.iterrows():
        start = min(max(0, row['start_frame']), num_frames - 1)
        end = min(max(0, row['end_frame']), num_frames)
        note = int(row['note'])
        instrument = int(row['instrument'])

        if 0 <= note < num_notes:
            y_notes[start:end, note-1] = 1
        if 0 <= instrument < num_instr:
            y_instr[start:end, instrument-1] = 1

    def create_windows_torch(X, y1, y2, window_size=128, step=21):
        """Args:
                X (numpy.ndarray): Dữ liệu đầu vào có shape (128, T).
                y1 (numpy.ndarray): Nhãn nốt nhạc (T, 128).
                y2 (numpy.ndarray): Nhãn nhạc cụ (T, 128).
                window_size (int): Kích thước cửa sổ theo thời gian (số frame).
                step (int): Bước nhảy giữa các cửa sổ.

            Returns:
                X_windows (Tensor): (batch, 1, 128, window_size).
                y1_windows (Tensor): (batch, 128).
                y2_windows (Tensor): (batch, 128).
            """
        X_windows, y1_windows, y2_windows = [], [], []

        for i in range(0, X.shape[1] - window_size, step):
            X_windows.append(X[:, i:i + window_size])
            center_index = i + window_size // 2

            if center_index < len(y1):
                y1_windows.append(y1[center_index, :])
                y2_windows.append(y2[center_index, :])

        # Chuyển sang Tensor
        X_windows = torch.tensor(np.array(X_windows), dtype=torch.float32)
        y1_windows = torch.tensor(np.array(y1_windows), dtype=torch.float32)  # (batch, 128)
        y2_windows = torch.tensor(np.array(y2_windows), dtype=torch.float32)  # (batch, 128)

        return X_windows, y1_windows, y2_windows

    X_train, y_notes, y_instr = create_windows_torch(mel_spec_db, y_notes, y_instr, window_size, step)
    return X_train, y_notes, y_instr