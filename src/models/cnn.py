"""
src/models/cnn.py
CNN and CNN-LSTM model architectures for spectrogram-based classification.
Input: (batch, 1, n_mels=128, frames=128) — log-Mel spectrograms
Output: (batch, 2) — logits for [genuine, synthetic]
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv → BN → ReLU → MaxPool block."""
    def __init__(self, in_ch, out_ch, pool=(2, 2)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool)
        )

    def forward(self, x):
        return self.block(x)


class SpeechCNN(nn.Module):
    """
    Lightweight CNN for spectrogram classification.
    4 conv blocks → global average pooling → classifier.
    ~2M parameters — fits comfortably in RTX 3060 12GB.
    """
    def __init__(self, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,   32, pool=(2, 2)),   # → (32, 64, 64)
            ConvBlock(32,  64, pool=(2, 2)),   # → (64, 32, 32)
            ConvBlock(64, 128, pool=(2, 2)),   # → (128, 16, 16)
            ConvBlock(128, 256, pool=(2, 2)),  # → (256, 8, 8)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # → (256, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


class SpeechCNNLSTM(nn.Module):
    """
    CNN-LSTM hybrid.
    CNN extracts local spectral features → LSTM captures temporal dynamics.
    Better suited for detecting temporal irregularities in synthetic speech.
    """
    def __init__(self, n_classes: int = 2, lstm_hidden: int = 128,
                  lstm_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        # CNN frontend (no final pooling — preserve time axis)
        self.cnn = nn.Sequential(
            ConvBlock(1,   32, pool=(2, 1)),    # freq pooling only → (32, 64, 128)
            ConvBlock(32,  64, pool=(2, 1)),    # → (64, 32, 128)
            ConvBlock(64, 128, pool=(2, 1)),    # → (128, 16, 128)
            ConvBlock(128, 128, pool=(2, 1)),   # → (128, 8, 128)
        )
        # After CNN: (batch, 128, 8, time_frames)
        # Reshape to (batch, time_frames, 128*8) for LSTM
        self.lstm_input_size = 128 * 8

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, 64),   # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # x: (batch, 1, 128, 128)
        x = self.cnn(x)                          # (batch, 128, 8, T)
        batch, ch, freq, time = x.shape
        x = x.permute(0, 3, 1, 2)               # (batch, T, ch, freq)
        x = x.reshape(batch, time, ch * freq)   # (batch, T, 128*8)
        _, (h_n, _) = self.lstm(x)              # h_n: (layers*2, batch, hidden)
        # Take last layer's forward and backward hidden states
        x = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, hidden*2)
        return self.classifier(x)


if __name__ == "__main__":
    # Quick shape check
    dummy = torch.randn(4, 1, 128, 128)  # batch=4

    cnn = SpeechCNN()
    out = cnn(dummy)
    print(f"SpeechCNN output shape: {out.shape}")
    params = sum(p.numel() for p in cnn.parameters())
    print(f"SpeechCNN parameters: {params:,}")

    lstm_model = SpeechCNNLSTM()
    out2 = lstm_model(dummy)
    print(f"SpeechCNNLSTM output shape: {out2.shape}")
    params2 = sum(p.numel() for p in lstm_model.parameters())
    print(f"SpeechCNNLSTM parameters: {params2:,}")
