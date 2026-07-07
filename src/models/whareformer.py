import torch
import torch.nn as nn

class LikelihoodEmbedding(nn.Module):
    def __init__(
        self, 
        app_dim: int,
        loc_dim: int,
        hidden_dim: int, 
        emb_dropout: float
    ):
        super().__init__()
        self.app_dim = app_dim
        self.loc_dim = loc_dim
        self.hidden_dim = hidden_dim
        
        self.likelihood_embedding = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(emb_dropout)
        )

    def slice_input(self, x: torch.Tensor):
        # Detection features (Index 0 in the track dimension)
        det_app = x[:, :1, :self.app_dim]                                    # (D, 1, app_dim)
        det_loc = x[:, :1, self.app_dim : self.app_dim + self.loc_dim]       # (D, 1, loc_dim)

        # Track features (Indices 1 to T in the track dimension)
        track_app = x[:, 1:, :self.app_dim]                                  # (D, T, app_dim)
        track_loc = x[:, 1:, self.app_dim : self.app_dim + self.loc_dim]     # (D, T, loc_dim)

        return det_app, det_loc, track_app, track_loc

    def distance_embeding(
        self, 
        det_app: torch.Tensor, # (D, 1, C)
        det_loc: torch.Tensor, # (D, 1, 3)
        track_app: torch.Tensor, # (D, T, C)
        track_loc: torch.Tensor, # (D, T, 3)
        track_mask: torch.Tensor = None
    ) -> torch.Tensor:
        with torch.no_grad(): 
            app_signal = torch.sum((det_app - track_app) ** 2, dim=-1, keepdim=True) # (D, T, 1)

            loc_signal = torch.norm(det_loc - track_loc, dim=-1, keepdim=True)       # (D, T, 1)

        track_signals = torch.cat([app_signal, loc_signal], dim=-1)                  # (D, T, 2)
        
        track_tokens = self.likelihood_embedding(track_signals)                      # (D, T, hidden_dim)

        if track_mask is not None:
            track_tokens = torch.masked_fill(track_tokens, track_mask.unsqueeze(-1), 0.0)

        return track_tokens 

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        det_app, det_loc, track_app, track_loc = self.slice_input(x)
        return self.distance_embeding(det_app, det_loc, track_app, track_loc, track_mask=mask)


class Whareformer(nn.Module):
    def __init__(
        self, 
        app_dim: int = 256,
        loc_dim: int = 3,
        hidden_dim: int = 128, 
        depth: int = 1, 
        num_heads: int = 8, 
        emb_dropout: float = 0.1, 
        enc_dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()

        self.embedding = LikelihoodEmbedding(
            app_dim=app_dim,
            loc_dim=loc_dim,
            hidden_dim=hidden_dim,
            emb_dropout=emb_dropout
        )

        self.new_track_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=enc_dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=depth,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False # Due to pre-norm
        )
        self.enc_dropout = nn.Dropout(enc_dropout)

        self.classifier = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, track_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x    (Tensor): (D, T+1, app_dim + loc_dim) — index 0 is detection, 1...T are tracks
            mask (Tensor): (D, T) - True where track slot is padded; NT token slot not included
        Returns:
            logits (Tensor): (D, T+1)
        """
        D, _, _ = x.shape

        x = self.embedding(x, mask=track_mask)                             # (D, T, hidden_dim)

        new_track = self.new_track_token.expand(D, -1, -1).to(x.device)    # (D, 1, hidden_dim)
        x = torch.cat([new_track, x], dim=1)                               # (D, T+1, hidden_dim)

        # Prepend False for NT token
        if track_mask is not None:
            nt_mask = torch.zeros(D, 1, dtype=torch.bool, device=x.device)
            seq_mask = torch.cat([nt_mask, track_mask], dim=1)             # (D, T+1)
        else:
            seq_mask = None                                            

        x = self.encoder(x, src_key_padding_mask=seq_mask)
        x = self.enc_dropout(x)

        if seq_mask is not None:
            x = x.masked_fill(seq_mask.unsqueeze(-1), 0.0)

        logits = self.classifier(x).squeeze(-1)

        if seq_mask is not None:
            logits = logits.masked_fill(seq_mask, 0.0)

        return logits