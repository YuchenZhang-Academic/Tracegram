from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from linear_attention_transformer.autopadder import Autopadder

def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


def pad_tensor(x, fill_value, target_length=128):
    batch_size, current_length = x.size()

    # 计算需要填充的长度
    remainder = current_length % target_length
    if remainder != 0:
        padding_length = target_length - remainder
    else:
        padding_length = 0

    # 创建填充后的tensor
    padded_tensor = torch.full((batch_size, current_length + padding_length), fill_value).to(x.device)

    # 将原始数据复制到填充后的tensor中
    padded_tensor[:, :current_length] = x

    return padded_tensor

class AutoregressiveWrapper(nn.Module):
    def __init__(self, net, ignore_index = 258, pad_value = 258):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = Autopadder(net)
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, eos_token = None, temperature = 1., filter_logits_fn = top_k, filter_thres = 0.9, **kwargs):
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        input_mask = kwargs.pop('input_mask', None)

        if input_mask is None:
            input_mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            input_mask = input_mask[:, -self.max_seq_len:]

            logits = self.net(x, input_mask=input_mask, **kwargs)[:, -1, :]
            filtered_logits = filter_logits_fn(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)
            input_mask = F.pad(input_mask, (0, 1), value=True)

            if eos_token is not None and (sample == eos_token).all():
                break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out

    def forward(self, x, return_loss = True, **kwargs):
        pad = partial(pad_sequence, batch_first = True, padding_value = self.pad_value)

        if not return_loss:
            if not isinstance(x, torch.Tensor):
                x = pad(x)
            return self.net(x, **kwargs)

        if isinstance(x, torch.Tensor):
            xi = x[:, :-1]
            xo = x[:, 1:]

            # help auto-solve an area of confusion around input masks in auto-regressive
            # if user supplies a mask that is only off by one from the source sequence, resolve it for them
            mask = kwargs.pop('input_mask', None)
            if mask is not None and mask.shape[1] == x.shape[1]:
                mask = mask[:, :-1]
                kwargs.update(input_mask = mask)
        else:
            xi = pad(list(map(lambda t: t[:-1], x)))
            xo = pad(list(map(lambda t: t[1:], x)))

        out = self.net(xi, **kwargs)
        #print('out', out.shape)
        #print(xo.shape)
        #os.exit()

        loss = F.cross_entropy(out.transpose(1, 2), xo, ignore_index = self.ignore_index, reduction='none')
        return loss


    def init_cls(self, class_num):
        feature_size = 256
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=4, batch_first=True)
        # 定义embedding层，假设词汇大小为vocab_size，每个token将被映射到embedding_dim维度的向量
        self.cls_embedding = nn.Embedding(1, feature_size)
        self.out = nn.Linear(feature_size, class_num)
        return 


    def forward_cls(self, x, **kwargs):
        pad = partial(pad_sequence, batch_first = True, padding_value = self.pad_value)
        if not isinstance(x, torch.Tensor):
            x = pad(x)

        local_window_size = 256
        x = pad_tensor(x, self.pad_value, target_length = local_window_size)
        
        x = self.net.net.token_emb(x)
        x = x + self.net.net.pos_emb(x).type(x.type())

        layer_pos_emb = self.net.net.layer_pos_emb(x)
        x = self.net.net.transformer(x, pos_emb = layer_pos_emb, **kwargs)
        x = self.net.net.norm(x)

        tgt_sequence = torch.randint(0, 1, (x.shape[0], 1)).to(x.device)
        embedded_tgt = self.cls_embedding(tgt_sequence)
        x = self.decoder_layer(embedded_tgt, x)
        x = F.leaky_relu(x)
        x = self.out(x)

        return x[:,0,:]


    def forward_feature(self, x, **kwargs):
        pad = partial(pad_sequence, batch_first = True, padding_value = self.pad_value)
        if not isinstance(x, torch.Tensor):
            x = pad(x)

        local_window_size = 256
        x = pad_tensor(x, self.pad_value, target_length = local_window_size)
        
        x = self.net.net.token_emb(x)
        x = x + self.net.net.pos_emb(x).type(x.type())

        layer_pos_emb = self.net.net.layer_pos_emb(x)
        x = self.net.net.transformer(x, pos_emb = layer_pos_emb, **kwargs)
        x = self.net.net.norm(x)

        tgt_sequence = torch.randint(0, 1, (x.shape[0], 1)).to(x.device)
        embedded_tgt = self.cls_embedding(tgt_sequence)
        x = self.decoder_layer(embedded_tgt, x)

        return x[:,0,:]

