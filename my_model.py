import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MyEncoder(nn.Module):
    def __init__(self,dim = 400):
        super(MyEncoder, self).__init__()

        torch.manual_seed(42)

        self.v = torch.rand(dim)


    def forward(self, x):
        encode = []

        for i in x[0]:
            a = i * self.v
            encode.append(a)

        encode = torch.stack(encode)
        encode = torch.unsqueeze(encode, dim=0)

        return encode





class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=1):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        # sin_den = 10000 ** (torch.arange(0, d_model, 2)/d_model) # sin for even item of position's dimension
        # cos_den = 10000 ** (torch.arange(1, d_model, 2)/d_model) # cos for odd
        self.encoding[:, 0::2] = torch.sin(position * div_term )
        self.encoding[:, 1::2] = torch.cos(position * div_term )
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):

        p = self.encoding[:, :x.size(1)].detach()
        p = p.to(device)

        return x + p


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim = 24 , num_heads = 6, num_layers = 6, dropout = 0.15 ,leaky_slope = 0.1):
        super(TransformerEncoder, self).__init__()

        # self.encoding = MyEncoder()
        # self.embedding = nn.Linear(input_dim, hidden_dim,bias = False)
        self.positional_encoding = PositionalEncoding(d_model = hidden_dim)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout
        )

        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=num_layers
        )


        self.final_layer = nn.Linear(hidden_dim,2)


    def forward(self, x):


        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)  # Transpose for transformer input shape
        x = self.transformer_encoder(x)
        x = x.mean(0)
        x = self.final_layer(x)


        return x

