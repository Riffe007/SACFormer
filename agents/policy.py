import torch
import torch.nn as nn

class TransformerPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(state_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_dim, action_dim)
        self.activation = nn.Tanh()

    def forward(self, x, deterministic=False):
        """
        x shape can be:
         - (batch_size, state_dim) => we do x = x.unsqueeze(1) => (batch_size, 1, state_dim)
         - (batch_size, seq_len=1, state_dim) => skip unsqueeze
        """
        print(f"🚀 input x shape: {x.shape}")
        if x.ndim == 2:
            # shape: (batch_size, state_dim) => unsqueeze to (batch_size, 1, state_dim)
            x = x.unsqueeze(1)
            print(f"🛠 after unsqueeze: {x.shape}")
        elif x.ndim == 3:
            # already (batch_size, 1, state_dim)
            pass
        else:
            raise ValueError(f"Wrong input shape for policy: {x.shape}")

        # embed => (batch_size, 1, hidden_dim)
        batch_size, seq_len, _ = x.shape
        x = self.embed(x.view(batch_size*seq_len, -1))  # => (batch_size*seq_len, hidden_dim)
        x = x.view(batch_size, seq_len, -1)             # => (batch_size, seq_len, hidden_dim)

        print(f'✅ shape after embed: {x.shape}')
        x = self.transformer(x)  # => (batch_size, seq_len, hidden_dim)
        print(f'✅ shape after transformer: {x.shape}')

        # Now remove seq_len=1
        x = x.squeeze(1)  # => (batch_size, hidden_dim)
        print(f'✅ after squeeze(1): {x.shape}')

        actions = self.output_layer(x)  # => (batch_size, action_dim)
        actions = self.activation(actions)
        print(f'🎯 final actions shape: {actions.shape}')
        return actions
