import torch
from torch import nn
from torch.nn import functional as F

from .kge_score import ConvE, TuckER, ComplEx


def get_param(shape):
    param = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_normal_(param.data)
    return param


class NumericValueEmbedding(nn.Module):
    """
    Missing-aware numeric encoding (ME).

    Observed values and missing states use separate parameter branches.  The
    explicit mask must describe which values are observed; callers should not
    infer it from normalized values when valid zeros are possible.
    """

    def __init__(self, num_att, att_dim):
        super(NumericValueEmbedding, self).__init__()
        self.v = nn.Parameter(torch.Tensor(num_att, att_dim))
        self.W_x = nn.Parameter(torch.Tensor(num_att, att_dim))
        self.w = nn.Parameter(torch.Tensor(num_att, att_dim))
        self.missing_bias = nn.Parameter(torch.Tensor(num_att, att_dim))

        nn.init.xavier_normal_(self.v)
        nn.init.xavier_normal_(self.W_x)
        nn.init.xavier_normal_(self.w)
        nn.init.xavier_normal_(self.missing_bias)

    def forward(self, X, observed_mask=None, use_missing=True):
        term1 = self.W_x.unsqueeze(0)
        term2 = self.w.unsqueeze(0) * X.unsqueeze(-1)
        value_embed = (term1 + term2) * self.v.unsqueeze(0)

        if not use_missing:
            return value_embed

        if observed_mask is None:
            observed_mask = (X != 0).to(value_embed.dtype)
        else:
            observed_mask = observed_mask.to(value_embed.dtype)

        observed_mask = observed_mask.unsqueeze(-1)
        missing_embed = self.missing_bias.unsqueeze(0).expand_as(value_embed)
        return observed_mask * value_embed + (1.0 - observed_mask) * missing_embed


class Gate(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 gate_activation=torch.sigmoid):
        super(Gate, self).__init__()
        self.output_size = output_size
        self.gate_activation = gate_activation
        self.g = nn.Linear(input_size, output_size)
        self.g1 = nn.Linear(output_size, output_size, bias=False)
        self.g2 = nn.Linear(input_size - output_size, output_size, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x_ent, x_lit):
        x = torch.cat([x_ent, x_lit], x_lit.ndimension() - 1)
        g_embedded = torch.tanh(self.g(x))
        gate = self.gate_activation(self.g1(x_ent) + self.g2(x_lit) + self.gate_bias)
        output = (1 - gate) * x_ent + gate * g_embedded
        return output


class Mixture(nn.Module):
    """
    Stable relation-aware mixture (SRM).

    Historical DRAE2 behavior: Gaussian gate noise is active in both training
    and evaluation.
    """

    def __init__(self, d_emb, num_mixtures):
        super(Mixture, self).__init__()
        self.num_mixtures = num_mixtures
        self.mixtures = nn.ModuleList([nn.Linear(d_emb, d_emb) for _ in range(num_mixtures)])
        self.gate = nn.Linear(d_emb, num_mixtures)
        self.temp_layer = nn.Linear(d_emb, 1)

    def forward(self, e, r):
        mixture_outputs = []
        for mixture in self.mixtures:
            mixture_outputs.append(mixture(e).unsqueeze(1))
        H = torch.cat(mixture_outputs, dim=1)
        gate_logits = self.gate(e)
        noise = torch.randn_like(gate_logits)
        temp = torch.sigmoid(self.temp_layer(r)).clamp_min(0.1)
        gate_weights = torch.softmax((gate_logits + noise) / temp, dim=1).unsqueeze(-1)
        e_moe = torch.sum(H * gate_weights, dim=1)
        return e_moe


class HMNCE(nn.Module):
    """Hard-sample and Missing-aware Numerical Contrastive Embedding.

    Historical compatibility: the literal mask is inferred from nonzero values;
    observed_mask is accepted for entry-point compatibility but is not used.
    The default configuration enables all three HMNCE components:
    score-driven hard contrastive learning (HCL), missing-aware numeric
    encoding (ME), and the auxiliary stable relation-aware mixture (SRM).
    Ablation behavior is controlled through the parameter namespace.
    """

    def __init__(self, num_ents, num_rels, numerical_literals, params=None, observed_mask=None):
        super(HMNCE, self).__init__()

        self.bceloss = torch.nn.BCELoss()
        self.p = params

        self.emb_dim = self.p.init_dim
        self.att_dim = self.p.att_dim
        self.head_num = self.p.head_num

        self.num_ents = num_ents
        self.num_rels = num_rels

        self.emb_e = get_param((num_ents, self.p.init_dim))
        self.emb_rel = get_param((num_rels, self.p.init_dim))

        self.num_att = numerical_literals.shape[1]
        self.emb_att = get_param((self.num_att, self.att_dim))
        self.num_embedder = NumericValueEmbedding(self.num_att, self.att_dim)
        self.W_r = get_param((num_rels, self.p.init_dim, self.p.init_dim))

        self.linear = nn.Linear(self.emb_dim, self.att_dim)
        self.multihead_attn = nn.MultiheadAttention(self.att_dim, self.head_num, batch_first=False)

        self.emb_num_lit = Gate(self.att_dim + self.emb_dim, self.emb_dim)

        self.drop = nn.Dropout(p=self.p.drop)
        self.att_linear = nn.Linear(self.att_dim, self.att_dim)
        self.num_linear = nn.Linear(1, self.att_dim)

        literals = torch.as_tensor(numerical_literals, dtype=torch.float32)
        mask = (literals != 0).float()
        if mask.shape != literals.shape:
            raise ValueError(f"observed_mask shape {tuple(mask.shape)} does not match literals {tuple(literals.shape)}")
        self.register_buffer("numerical_literals", literals)
        self.register_buffer("literal_mask", mask)

        self.str = Mixture(self.emb_dim, self.p.num_mixture)

        legacy_score = getattr(self.p, 'drae_score_func', 'transe')
        self.hmnce_score_func = getattr(self.p, 'hmnce_score_func', legacy_score).lower()
        if self.hmnce_score_func == 'conve':
            self.hmnce_scorer = ConvE(self.emb_dim, feature_map_size=32)
        elif self.hmnce_score_func == 'tucker':
            self.hmnce_scorer = TuckER(self.num_ents, self.num_rels, self.emb_dim, self.emb_dim, dropout=0.3)
        elif self.hmnce_score_func == 'complex':
            self.hmnce_scorer = ComplEx(self.emb_dim)
        elif self.hmnce_score_func != 'transe':
            raise ValueError(f'Unsupported HMNCE score function: {self.hmnce_score_func}')

    def _score_triples(self, e1_emb, rel_emb, e2_multi_emb):
        if self.hmnce_score_func == 'transe':
            return self.p.gamma - torch.norm(((e1_emb + rel_emb).unsqueeze(1) - e2_multi_emb), p=1, dim=2)
        return self.hmnce_scorer(e1_emb, rel_emb, e2_multi_emb)

    def _topk_mean(self, emb, indices):
        return torch.gather(
            emb,
            1,
            indices.unsqueeze(-1).expand(-1, -1, self.emb_dim)
        ).mean(dim=1)

    def forward(self, g, e1, rel, e2_multi, neg, n_label):
        e1_emb = torch.index_select(self.emb_e, 0, e1)
        rel_emb = torch.index_select(self.emb_rel, 0, rel)
        e2_multi_emb = self.emb_e
        W_r = torch.index_select(self.W_r, 0, rel)

        if getattr(self.p, 'disable_srm', False):
            e_str = e1_emb
        else:
            e_str = self.str(e1_emb, rel_emb)

        att = self.num_embedder(
            self.numerical_literals.to(torch.float32),
            self.literal_mask,
            use_missing=not getattr(self.p, 'disable_me', False)
        )
        e1_emb_att = torch.index_select(att, 0, e1).transpose(0, 1)
        e2_emb_att = att.transpose(0, 1)

        rel_emb_att = torch.tanh(self.linear(rel_emb)).unsqueeze(0)
        rel_emb_all_att = rel_emb_att.transpose(0, 1).repeat(1, e2_emb_att.shape[1], 1)

        e1_num_lit, _ = self.multihead_attn(rel_emb_att, e1_emb_att, e1_emb_att)
        e2_multi_num_lit, _ = self.multihead_attn(rel_emb_all_att, e2_emb_att, e2_emb_att)

        e1_num_lit = e1_num_lit.squeeze(0)
        e1_emb = self.emb_num_lit(e1_emb, e1_num_lit)
        e1_emb_w = self.emb_num_lit(e_str, e1_num_lit)

        e2_emb_all = e2_multi_emb.repeat(e1_emb.shape[0], 1).view(-1, e2_emb_att.shape[1], self.emb_dim)
        e2_multi_emb = self.emb_num_lit(e2_emb_all, e2_multi_num_lit)
        e2_multi_emb_w = e2_multi_emb

        e1_emb = self.drop(e1_emb)
        e2_multi_emb = self.drop(e2_multi_emb)

        e1_emb_w, e2_multi_emb_w = torch.matmul(e1_emb_w.unsqueeze(1), W_r), torch.bmm(e2_multi_emb_w, W_r)
        distance = e2_multi_emb_w - e1_emb_w
        distance = torch.clamp(distance, min=0)
        order_score = self.p.gamma - torch.square(distance).sum(2)

        score = self._score_triples(e1_emb, rel_emb, e2_multi_emb)

        pred = torch.sigmoid(score + self.p.order * order_score)
        pred = torch.nan_to_num(pred, nan=0.5, posinf=1.0, neginf=0.0).clamp(1e-7, 1.0 - 1e-7)

        if self.training:
            loss = self.calc_loss(pred, e2_multi)

            if getattr(self.p, 'disable_hcl', False):
                return loss

            pos_mask = e2_multi.bool()
            neg_mask = n_label.bool()

            pos_count = pos_mask.sum(dim=-1)
            neg_count = neg_mask.sum(dim=-1)
            valid_mask = (pos_count > 0) & (neg_count > 0) & (neg > 0)

            if valid_mask.any():
                valid_pos_count = pos_count[valid_mask]
                valid_neg_count = neg_count[valid_mask]
                top_k = 1 if getattr(self.p, 'disable_hsa', False) else getattr(self.p, 'top_k', 5)
                k_pos = min(top_k, max(1, int(valid_pos_count.min().item())))
                k_neg = min(top_k, max(1, int(valid_neg_count.min().item())))

                valid_pos_mask = pos_mask[valid_mask]
                valid_neg_mask = neg_mask[valid_mask]
                valid_e2_emb = e2_multi_emb[valid_mask]

                with torch.no_grad():
                    valid_pred = pred[valid_mask]
                    if getattr(self.p, 'disable_hsm', False):
                        pos_selection_score = torch.rand_like(valid_pred).masked_fill(~valid_pos_mask, 2.0)
                        neg_selection_score = torch.rand_like(valid_pred).masked_fill(~valid_neg_mask, 2.0)
                        _, hard_pos_idx = torch.topk(pos_selection_score, k_pos, dim=1, largest=False)
                        _, hard_neg_idx = torch.topk(neg_selection_score, k_neg, dim=1, largest=False)
                    else:
                        hard_pos_score = valid_pred.masked_fill(~valid_pos_mask, 1e9)
                        hard_neg_score = valid_pred.masked_fill(~valid_neg_mask, -1e9)
                        _, hard_pos_idx = torch.topk(hard_pos_score, k_pos, dim=1, largest=False)
                        _, hard_neg_idx = torch.topk(hard_neg_score, k_neg, dim=1, largest=True)

                pos_hard_emb = self._topk_mean(valid_e2_emb, hard_pos_idx)
                neg_hard_emb = self._topk_mean(valid_e2_emb, hard_neg_idx)

                valid_e1_emb = e1_emb[valid_mask]
                valid_rel_emb = rel_emb[valid_mask]
                alpha = torch.rand(valid_e1_emb.shape[0], 1, device=e1_emb.device)
                beta = torch.rand(valid_e1_emb.shape[0], 1, device=e1_emb.device)
                positive_mix = alpha * pos_hard_emb + (1 - alpha) * valid_e1_emb
                negative_mix = beta * neg_hard_emb + (1 - beta) * valid_e1_emb

                pos_dist = torch.norm(valid_e1_emb + valid_rel_emb - positive_mix, p=1, dim=-1)
                neg_dist = torch.norm(valid_e1_emb + valid_rel_emb - negative_mix, p=1, dim=-1)
                contrastive_gap = torch.clamp(pos_dist - neg_dist, min=-20.0, max=20.0)
                contrastive_loss = F.softplus(contrastive_gap).mean()
                contrastive_loss = torch.nan_to_num(contrastive_loss, nan=0.0, posinf=20.0, neginf=0.0)
            else:
                contrastive_loss = loss.new_tensor(0.0)

            return loss + contrastive_loss * self.p.scale
        else:
            return pred

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def loss(self, pred, true_label):
        pred = torch.nan_to_num(pred, nan=0.5, posinf=1.0, neginf=0.0).clamp(1e-7, 1.0 - 1e-7)
        return self.bceloss(pred, true_label)


# Backward-compatible name for loading historical scripts and checkpoints.
DRAE = HMNCE
