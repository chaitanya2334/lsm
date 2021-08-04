from torch import nn


class posFFN1d(nn.Module):
    def __init__(self, d_hid, d_inner_hid, window=1, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, kernel_size=window)
        self.relu = nn.ReLU()
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, kernel_size=window)
        self.layer_norm = nn.LayerNorm(d_hid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.w_1(x)
        out = self.relu(out)
        out = self.w_2(out)
        out = self.dropout(out)
        return self.layer_norm(out + x)


class posFFN2d(nn.Module):
    def __init__(self, d_hid, d_inner_hid, window=1, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv2d(d_hid, d_inner_hid, kernel_size=window, padding=1)
        self.relu = nn.ReLU()
        self.w_2 = nn.Conv2d(d_inner_hid, d_hid, kernel_size=window, padding=1)
        self.layer_norm = nn.LayerNorm(d_hid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # -> (N x N x d_hid)
        x = x.permute(2, 0, 1)
        # -> (d_hid x N x N)
        x = x.unsqueeze(0)
        # -> (1 x d_hid x N x N)
        out = self.w_1(x)
        out = self.relu(out)
        out = self.w_2(out)
        out = self.dropout(out)
        # -> (1 x d_hid x N x N)
        out = out.squeeze(0)
        # -> (d_hid x N x N)

        x = x.squeeze(0).permute(1, 2, 0)
        # -> (N x N x d_hid)

        out = out.permute(1, 2, 0)
        # -> (N x N x d_hid)
        out = self.layer_norm(out + x)
        # -> (N x N x d_hid)
        return out