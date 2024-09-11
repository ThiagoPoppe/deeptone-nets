import torch
import torch.nn as nn


class HarmonyBassChromaNetwork(nn.Module):
    def __init__(self, dropout=0.5):
        super(HarmonyBassChromaNetwork, self).__init__()
        
        self.shared_mlp = nn.Sequential(
            nn.Linear(1575, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout)
        )
        
        self.harmony_chroma_mlp = nn.Linear(256, 12)
        
        self.bass_attention = nn.Linear(12, 256)
        self.bass_chroma_conv = nn.Linear(256, 12)
        
    def forward(self, x):
        x = self.shared_mlp(x)
        harmony_chromagram = self.harmony_chroma_mlp(x)
        
        # Attention mechanism for bass chromagram
        attention_values = torch.sigmoid(self.bass_attention(harmony_chromagram))
        weighted_features = torch.sigmoid(attention_values) * x
        
        # Bass chromagram prediction
        bass_chromagram = self.bass_chroma_conv(weighted_features)
        
        return bass_chromagram, harmony_chromagram 
