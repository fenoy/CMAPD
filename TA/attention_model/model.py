import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Embedder(nn.Module):
    def __init__(self, input_size, d_model):
        super().__init__()
        self.is_continuous = ~torch.tensor([bool(s) for s in input_size], dtype=torch.bool)

        size = input_size.count(0)
        self.continuous_embedding = nn.Linear(size, d_model) if size > 0 else None
        self.categorical_embedding = nn.ModuleList([
            nn.Embedding(1 + s, d_model, padding_idx=0)
            for s in input_size if s != 0])

        d_continuous = d_model if size > 0 else 0
        self.output = nn.Linear(d_continuous + d_model, d_model)

    def forward(self, x):
        continuous = x[..., self.is_continuous].float()
        categorical = x[..., ~self.is_continuous].long()

        categorical_embedded = torch.stack([
            emb(1 + categorical[..., i])
            for i, emb in enumerate(self.categorical_embedding)], dim=-1).sum(dim=-1)

        if self.continuous_embedding is not None:
            continuous_embedded = self.continuous_embedding(continuous)
            return self.output(torch.cat([continuous_embedded, categorical_embedded], dim=-1))

        return self.output(categorical_embedded)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None, output_attention=False):
        batch_size = q.size(0)

        q = self.w_q(q).view(batch_size, self.nhead, -1, self.d_head)
        k = self.w_k(k).view(batch_size, self.nhead, -1, self.d_head)
        v = self.w_v(v).view(batch_size, self.nhead, -1, self.d_head)

        u = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if output_attention:
            return u.mean(dim=1)

        if mask is not None:
            u = u.masked_fill(mask.view(batch_size, 1, 1, -1), float("-inf"))

        attn = F.softmax(u, dim=-1)
        attn_applied = torch.matmul(attn, v)
        return self.w_o(attn_applied.view(batch_size, -1, self.d_model))

class Encoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers):
        super().__init__()
        self.attn = nn.ModuleList([
            MultiHeadAttention(d_model, nhead) for _ in range(num_layers)])

        self.ffrd = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, d_model))
            for _ in range(num_layers)])

        self.norm1 = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)])

        self.norm2 = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for attn, ffrd, norm1, norm2 in zip(self.attn, self.ffrd, self.norm1, self.norm2):
            x = norm1(x + attn(x, x, x, mask))
            x = norm2(x + ffrd(x))
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers=1):
        super().__init__()
        self.attn = nn.ModuleList([
            MultiHeadAttention(d_model, nhead) for _ in range(num_layers)])

        self.ffrd = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 512),
                nn.ReLU(),
                nn.Linear(512, d_model))
            for _ in range(num_layers)])

        self.norm1 = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)])

        self.norm2 = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)])

        self.output = MultiHeadAttention(d_model, 1)

    def forward(self, x, h, mask=None):
        for attn, ffrd, norm1, norm2 in zip(self.attn, self.ffrd, self.norm1, self.norm2):
            h = norm1(h + attn(h, x, x, mask))
            h = norm2(h + ffrd(h))

        return self.output(h, x, x, output_attention=True).squeeze(1)

class Transformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, dim_feedforward, num_layers):
        super().__init__()
        self.agents_embedder = Embedder(input_size, d_model)
        self.tasks_embedder = Embedder(2*input_size, d_model)

        self.encoder = Encoder(2*d_model, nhead, dim_feedforward, num_layers)

        self.decoder_pi = Decoder(2*d_model, nhead)
        self.decoder_v = Decoder(2*d_model, nhead)

    def _add_token(self, context):
        return F.pad(context, pad=(0, 0, 1, 0), value=0)

    def _get_hidden_state(self, context, coalition):
        return context.gather(1, 1 + coalition.unsqueeze(-1).expand([-1, -1, context.size(2)])
            ).mean(dim=1, keepdim=True)

    def _get_mask(self, tasks, collective):
        batch_size, seq_len, _ = tasks.size()
        mask = torch.zeros(batch_size, 1 + seq_len, dtype=torch.bool, device=collective.device)
        mask.scatter_(1, 1 + collective, True)
        return mask[:, 1:]

    def forward(self, agents, tasks, collective, outer_mask=None):
        collective_mask = self._get_mask(tasks, collective)
        padding_mask = torch.all((tasks == -1), dim=-1)
        outer_mask = outer_mask if outer_mask is not None else torch.zeros_like(collective_mask,  dtype=torch.bool, device=collective.device)
        mask = collective_mask | padding_mask | outer_mask

        agents_emb = self.agents_embedder(agents)
        tasks_emb = self.tasks_embedder(tasks)
        embedding = torch.cat([agents_emb.repeat(1, tasks.size(1), 1), tasks_emb], dim=-1)

        context = self.encoder(embedding, mask=mask)
        context = self._add_token(context)
        mask = self._add_token(mask.unsqueeze(-1)).squeeze(-1)
        hidden = self._get_hidden_state(context, collective)

        probs = 8 * torch.tanh(self.decoder_pi(context, hidden, mask))
        probs = F.softmax(probs.masked_fill(mask, float("-inf")), dim=-1)
        v = self.decoder_v(context, hidden, mask).masked_fill(mask, 0)

        value = torch.matmul(v, probs.unsqueeze(-1)).squeeze()
        return probs, value
