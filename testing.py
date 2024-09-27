from Embeddings import InputEmbeddings
import torch
from PositionalEncodings import PositionalEncoding
from MultiHeadAttention import MHA
from AddNorm import AddNorm
from FeedForwardNetwork import FFN

torch.manual_seed(111)
n_seq, vocab_size, d_model, n_heads, dropout, mask = 10, 100, 20, 5, 0.01, None

# Generate valid token indices (shape: [batch_size, n_seq])
ip = torch.randint(low=0, high= vocab_size, size=(3, n_seq))  # Shape: (3, 10)
print(ip.shape)
emb = InputEmbeddings(vocab_size , d_model)


pose = PositionalEncoding(d_model , n_seq , dropout)
ip_emb = emb(ip)
pos = pose(ip_emb)

# print(f"INPUT EMBEDDINGS {ip_emb}")
# print(f"INPUT WITH PoSE {pos}")

mha = MHA(d_model , n_heads , mask)
att = mha(pos)
print(att)

norm1 = AddNorm()
norm2 = AddNorm()
ffn = FFN(d_model , dropout)

norm1_out = norm1(pos , att)
print(norm1_out )

ffn_out = ffn(norm1_out)
norm_out2 = norm2(norm1_out , ffn_out  )
print(norm1_out)