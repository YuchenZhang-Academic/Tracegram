import torch
import torch.nn as nn
from dataset import PcapDataset
import random
import numpy as np
from torch.autograd import Variable
import math
import torch.nn.functional as F
from linear_attention_transformer import LinearAttentionTransformerLM
from myautowrapper import AutoregressiveWrapper



def get_model(num_tokens, max_len):

    model = LinearAttentionTransformerLM(
        num_tokens = num_tokens,
        dim = 512,
        heads = 12,
        depth = 24,
        max_seq_len = max_len,
        causal = True,                  # auto-regressive or not
        ff_dropout = 0.1,               # dropout for feedforward
        attn_layer_dropout = 0.1,       # dropout right after self-attention layer
        attn_dropout = 0.1,             # dropout post-attention
        emb_dim = 256,                  # embedding factorization, to save on memory
        dim_head = 256,                 # be able to fix the dimension of each head, making it independent of the embedding dimension and the number of heads
        blindspot_size = 64,            # this gives the q(kv) attention a blindspot of 64 tokens back in the causal case, but gives back an order of magnitude return in memory savings. should be paired with local attention of at least a window size of this setting. setting this to 1 will allow for full q(kv) attention of past
        n_local_attn_heads = 8,         # number of local attention heads for (qk)v attention. this can be a tuple specifying the exact number of local attention heads at that depth
        local_attn_window_size = 256,   # receptive field of the local attention
        reversible = True,              # use reversible nets, from Reformer paper
        ff_chunks = 2,                  # feedforward chunking, from Reformer paper
        ff_glu = True,                  # use GLU variant for feedforward
        attend_axially = False,         # will fold the sequence by the local attention window size, and do an extra strided attention followed by a feedforward with the cheap q(kv) attention
        shift_tokens = True             # add single token shifting, for great improved convergence
    )

    model = AutoregressiveWrapper(model)

    model = nn.DataParallel(model)

    return model




def main():
    dataset = PcapDataset('../../flow_output')
    max_len = 2*128

    # 创建模型实例
    model = get_model(num_tokens = 260, max_len = max_len)

    batch_sample = dataset.get_batch(batch_size=2, max_len = max_len)
    print(batch_sample)

    loss = model(batch_sample)
    loss.backward()
    print(loss)




if __name__ == "__main__":
    main()

    
    














