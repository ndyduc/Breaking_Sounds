import librosa
import librosa.feature
import numpy as np
import torch.nn as nn
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_wav_csv(wav_path, csv_path=None, sr=44100, hop_length=512, n_mels=128, n_mfcc=40, bins_per_octave=24):
    y, sr = librosa.load(wav_path, sr=sr)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, bins_per_octave=bins_per_octave)
    cqt = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

    # Đồng bộ kích thước
    min_frames = min(mel_spec.shape[1], mfcc.shape[1], cqt.shape[1])
    mel_spec = mel_spec[:, :min_frames]
    mfcc = mfcc[:, :min_frames]
    cqt = cqt[:, :min_frames]

    # Chuẩn hóa [0, 1]
    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-6)
    mfcc = (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min() + 1e-6)
    cqt = (cqt - cqt.min()) / (cqt.max() - cqt.min() + 1e-6)

    # Ghép đặc trưng
    X = np.vstack([mel_spec, mfcc, cqt])

    # Chia window
    def create_windows_torch(X, window_size=128, step=21):
        X_windows = []
        for i in range(0, X.shape[1] - window_size, step):
            X_windows.append(X[:, i:i + window_size])
        return torch.tensor(np.array(X_windows), dtype=torch.float32)

    X_train = create_windows_torch(X)
    X_train = X_train.permute(0, 2, 1)  # (batch, time, feature)

    # Nếu không có nhãn thì return luôn
    if csv_path is None:
        return X_train, None

    # Nếu có nhãn thì load
    df = pd.read_csv(csv_path)
    num_frames = X.shape[1]
    y_notes = np.zeros((num_frames, 128), dtype=np.float32)

    for _, row in df.iterrows():
        start = max(0, int(row['start_time'] // hop_length))
        end = min(num_frames, int(row['end_time'] // hop_length))
        note = int(row['note'])
        y_notes[start:end, note] = 1

    # Tạo label theo từng cửa sổ giống input
    def create_label_windows(y1, window_size=128, step=21):
        y_notes_windows = []
        for i in range(0, y1.shape[0] - window_size, step):
            y_notes_windows.append(y1[i:i + window_size])
        return torch.tensor(np.array(y_notes_windows), dtype=torch.float32)

    y_notes = create_label_windows(y_notes)
    return X_train, y_notes


class BiLSTM(nn.Module):
    def __init__(self, input_size=252, hidden_dim=256, num_layers=2, num_notes=128, num_instruments=128):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.relu = nn.ReLU()
        self.fc_notes = nn.Linear(hidden_dim * 2, num_notes)
        self.fc_instr = nn.Linear(hidden_dim * 2, num_instruments)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.norm(x)
        x = self.relu(x)
        notes_out = self.fc_notes(x)
        instr_out = self.fc_instr(x)
        return notes_out, instr_out


def train_on_wav(model, optimizer, criterion, wav_path, csv_path, test_wav_files, test_csv_files, epochs=10, batch_size=128):
    from sklearn.metrics import f1_score, precision_score, recall_score

    X, y_notes = load_wav_csv(wav_path, csv_path)
    train_dataset = TensorDataset(X, y_notes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Load dữ liệu validation từ các file test
    val_data = [load_wav_csv(w, c) for w, c in zip(test_wav_files, test_csv_files)]
    X_val, y_notes_val = map(torch.cat, zip(*val_data))

    val_dataset = TensorDataset(X_val, y_notes_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    best_model = None
    prev_loss = float('inf')
    patience = 2
    count = 0

    for epoch in range(epochs):
        total_train_loss = 0
        for X_batch, y_notes_batch in train_loader:
            X_batch, y_notes_batch = X_batch.to(device), y_notes_batch.to(device)

            optimizer.zero_grad()
            notes_pred, _ = model(X_batch)

            loss_notes = criterion(notes_pred, y_notes_batch)
            loss_notes.backward()
            optimizer.step()

            total_train_loss += loss_notes.item()

        # Tính validation loss và metrics
        model.eval()
        total_val_loss = 0
        all_true_notes, all_pred_notes = [], []

        with torch.no_grad():
            for X_val_batch, y_notes_val_batch in val_loader:
                X_val_batch, y_notes_val_batch = X_val_batch.to(device), y_notes_val_batch.to(device)

                notes_val_pred, _ = model(X_val_batch)
                loss_notes_val = criterion(notes_val_pred, y_notes_val_batch)
                total_val_loss += loss_notes_val.item()

                all_true_notes.append(y_notes_val_batch.cpu())
                all_pred_notes.append((torch.sigmoid(notes_val_pred) > 0.3).cpu())

        # Tính F1
        y_true_notes = torch.cat(all_true_notes).numpy().flatten()
        y_pred_notes = torch.cat(all_pred_notes).numpy().flatten()
        f1_notes = f1_score(y_true_notes, y_pred_notes, average="macro", zero_division=1)
        precision_notes = precision_score(y_true_notes, y_pred_notes, average="macro", zero_division=1)
        recall_notes = recall_score(y_true_notes, y_pred_notes, average="macro", zero_division=1)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_train_loss:.4f}, Val Loss: {total_val_loss:.4f}")
        print(f" Precision: {precision_notes:.4f}, Recall: {recall_notes:.4f}, F1-score: {f1_notes:.4f}")

        model.train()

        if total_val_loss < prev_loss:
            count = 0
            prev_loss = total_val_loss
            best_model = model.state_dict()
        else:
            count += 1
            if count >= patience:
                print("🛑 Early stopping.")
                break

        # Sau khi train xong
        if best_model is not None:
            torch.save(best_model, "../Trained/BiLSTM_shuffle.pth")


def predict(model, wav_path, device, sr=44100, hop_length=512):
    model.eval()

    X, _ = load_wav_csv(wav_path, None)
    X = X.to(device)

    all_preds_notes = []
    with torch.no_grad():
        for x in X:
            x = x.unsqueeze(0).to(device)  # (1, time, feature)
            pred_note = model(x)

            pred_note = torch.sigmoid(pred_note)

            all_preds_notes.append(pred_note.squeeze(0).cpu())  # (time, 128)

    pred_notes = torch.cat(all_preds_notes, dim=0)

    note_events = []
    active_notes = {}

    for t in range(pred_notes.shape[0]):
        original_frame = (t // 128) * 21 + (t % 128)
        time_in_seconds = original_frame * hop_length / sr

        for note in range(pred_notes.shape[1]):
            note_prob = pred_notes[t, note].item()

            if note_prob >= 0.3:
                if note not in active_notes or not active_notes[note]["is_active"]:
                    active_notes[note] = {
                        "start_frame": original_frame,
                        "start_time": time_in_seconds,
                        "is_active": True
                    }
            else:
                if note in active_notes and active_notes[note]["is_active"]:
                    info = active_notes[note]
                    duration = time_in_seconds - info["start_time"]
                    note_events.append({
                        "note": note,
                        "start": info["start_time"],
                        "duration": duration
                    })
                    active_notes[note]["is_active"] = False

    # Xử lý các nốt còn đang bật ở cuối file
    for note, info in active_notes.items():
        if info["is_active"]:
            end_time = pred_notes.shape[0] // 128 * 21 * hop_length / sr
            duration = end_time - info["start_time"]
            note_events.append({
                "note": note,
                "start": info["start_time"],
                "duration": duration
            })

    return note_events


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, time, hidden*2)
        weights = torch.softmax(self.attn(lstm_out), dim=1)  # (batch, time, 1)
        context = torch.sum(weights * lstm_out, dim=1)  # (batch, hidden*2)
        return context


class BiLSTMNoteClassifier(nn.Module):
    def __init__(self, input_size=252, hidden_dim=256, num_layers=2, num_notes=128, dropout=0.3):
        super(BiLSTMNoteClassifier, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )

        self.layernorm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim * 2, num_notes)

    def forward(self, x):
        # x: (batch, time, feature)
        lstm_out, _ = self.lstm(x)  # (batch, time, hidden*2)
        norm_out = self.layernorm(lstm_out)
        dropped = self.dropout(norm_out)
        logits = self.fc(dropped)  # (batch, time, 128)
        return logits  # Chưa sigmoid (dùng BCEWithLogitsLoss sau)


class BiLSTM_new(nn.Module):
    def __init__(self, input_size=252, hidden_dim=256, num_layers=3, num_notes=128, dropout=0.3):
        super(BiLSTM_new, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )

        self.layernorm = nn.LayerNorm(hidden_dim * 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.residual_proj = nn.Linear(252, hidden_dim * 2)  # 252 → 1024
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, num_notes)

    def forward(self, x):
        # x: (batch, time, feature)
        residual = x
        lstm_out, _ = self.lstm(x)
        residual_proj = self.residual_proj(residual)  # shape [batch, seq, 1024]
        lstm_out = lstm_out + residual_proj
        lstm_out = self.dropout1(lstm_out)
        norm_out = self.layernorm(lstm_out)
        dropped = self.dropout2(norm_out)

        context = self.attention(dropped)
        logits = self.fc(context).unsqueeze(1).repeat(1, x.size(1), 1)
        return logits


def train_multiple_wavs(model, optimizer, criterion, wav_paths, csv_paths, test_wav_files, test_csv_files, checkpoint_path, epochs=5, batch_size=128):
    from sklearn.metrics import f1_score, precision_score, recall_score

    model.train()

    train_data = []
    for wav, csv in zip(wav_paths, csv_paths):
        x, y = load_wav_csv(wav, csv)
        train_data.append((x, y))

    X_train, y_notes_train = map(torch.cat, zip(*train_data))

    train_dataset = TensorDataset(X_train, y_notes_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Dữ liệu validation
    val_data = [load_wav_csv(w, c) for w, c in zip(test_wav_files, test_csv_files)]
    X_val, y_notes_val = map(torch.cat, zip(*[(x, y) for x, y in val_data]))

    val_dataset = TensorDataset(X_val, y_notes_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_model = None
    prev_loss = float('inf')
    patience = 2
    count = 0

    for epoch in range(epochs):
        total_train_loss = 0
        for X_batch, y_notes_batch in train_loader:
            X_batch, y_notes_batch = X_batch.to(device), y_notes_batch.to(device)

            optimizer.zero_grad()
            notes_pred = model(X_batch)
            loss = criterion(notes_pred, y_notes_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Validation
        model.eval()
        total_val_loss = 0
        all_true, all_pred = [], []

        with torch.no_grad():
            for X_val_batch, y_notes_val_batch in val_loader:
                X_val_batch, y_notes_val_batch = X_val_batch.to(device), y_notes_val_batch.to(device)

                notes_val_pred= model(X_val_batch)
                loss_val = criterion(notes_val_pred, y_notes_val_batch)
                total_val_loss += loss_val.item()

                all_true.append(y_notes_val_batch.cpu())
                all_pred.append((torch.sigmoid(notes_val_pred) > 0.3).cpu())

        # Metrics
        y_true = torch.cat(all_true).numpy().flatten()
        y_pred = torch.cat(all_pred).numpy().flatten()
        precision = precision_score(y_true, y_pred, average="macro", zero_division=1)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=1)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=1)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_train_loss:.4f}, Val Loss: {total_val_loss:.4f}")
        print(f" Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

        if total_val_loss < prev_loss:
            best_model = model.state_dict()
            prev_loss = total_val_loss
            count = 0
        else:
            count += 1
            if count >= patience:
                print("🛑 Early stopping.")
                break
        model.train()

    if best_model:
        torch.save(best_model, checkpoint_path)


