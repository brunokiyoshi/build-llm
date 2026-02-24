import torch 

class SelfAttention(torch.nn.Module):
    def __init__(self, d_in:int, d_out:int, qkv_bias=False):
        super().__init__()

        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        key_vectors = self.W_key(x)
        query_vectors = self.W_query(x)
        value_vectors = self.W_value(x)

        attn_scores = query_vectors @ key_vectors.T
        scaled_attn_scores = attn_scores/(key_vectors.shape[-1] ** 0.5)
        attn_weights = torch.softmax(scaled_attn_scores, dim=-1)
        context_vectors = attn_weights @ value_vectors
        
        return context_vectors