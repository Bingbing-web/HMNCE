import torch
from torch import nn
from torch.nn import functional as F

from .kge_score import ConvE,TuckER,ComplEx,MuRP,RGCNModel,BiGI


def get_param(shape):
    param = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_normal_(param.data)
    return param


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
        g_embedded = torch.tanh(self.g(x))  # 公式10
        gate = self.gate_activation(self.g1(x_ent) + self.g2(x_lit) + self.gate_bias)  # 公式9
        output = (1 - gate) * x_ent + gate * g_embedded  # 公式11

        return output


class Mixture(nn.Module):
    """
    Mixer module.
    The input entity is embedded in e, K mixers are used for fusion,
    and the gating network is combined with the weighted fusion of relational information.
    """

    def __init__(self, d_emb, num_mixtures):
        super(Mixture, self).__init__()
        self.num_mixtures = num_mixtures
        self.mixtures = nn.ModuleList([nn.Linear(d_emb, d_emb) for _ in range(num_mixtures)])
        self.gate = nn.Linear(d_emb, num_mixtures)  # Projection layer U(·)
        self.temp_layer = nn.Linear(d_emb, 1)  # Used to obtain relational perceived temperature ρ(ε_r)

    def forward(self, e, r):
        # e: Entity original embedding [batch_size, d_emb]
        # r: Relational embedding [batch_size, d_emb]
        mixture_outputs = []
        for mixture in self.mixtures:
            mixture_outputs.append(mixture(e).unsqueeze(1))  # [batch_size, 1, d_emb]
        H = torch.cat(mixture_outputs, dim=1)  # [batch_size, num_mixtures, d_emb]
        # The gating score is calculated and Gaussian noise is added
        gate_logits = self.gate(e)  # [batch_size, num_mixtures]
        noise = torch.randn_like(gate_logits)
        temp = torch.sigmoid(self.temp_layer(r))  # [batch_size, 1]
        gate_scores = torch.exp((gate_logits + noise) / temp)  # [batch_size, num_mixtures]
        gate_weights = gate_scores / torch.sum(gate_scores, dim=1, keepdim=True)  #normalization
        gate_weights = gate_weights.unsqueeze(-1)  # [batch_size, num_mixtures, 1]
        # Mixed output
        e_moe = torch.sum(H * gate_weights, dim=1)  # [batch_size, d_emb]
        return e_moe


class DRAE(nn.Module):
    def __init__(self, num_ents, num_rels, numerical_literals, params=None):
        super(DRAE, self).__init__()

        self.bceloss = torch.nn.BCELoss()
        self.p = params

        self.device = "cuda"
        self.emb_dim = self.p.init_dim
        self.att_dim = self.p.att_dim
        self.head_num = self.p.head_num
    #
        self.num_ents = num_ents
        self.num_rels = num_rels
    #
        # entity, relation embedding table
        self.emb_e = get_param((num_ents, self.p.init_dim))
        self.emb_rel = get_param((num_rels, self.p.init_dim))

        # attribute embedding table
        self.num_att = numerical_literals.shape[1]  # 矩阵的列数(涉及数值属性的个数)
        self.emb_att = get_param((self.num_att, self.att_dim))  # 属性嵌入矩阵
        self.W_r = get_param((num_rels, self.p.init_dim, self.p.init_dim))

        # relation projection
        self.linear = nn.Linear(self.emb_dim, self.att_dim)

        # MultiheadAttention
        self.multihead_attn = nn.MultiheadAttention(self.att_dim, self.head_num, batch_first=False)

        # gating
        self.emb_num_lit = Gate(self.att_dim + self.emb_dim, self.emb_dim)

        self.drop = nn.Dropout(p=self.p.drop)
        self.att_linear = nn.Linear(self.att_dim, self.att_dim)
        self.num_linear = nn.Linear(1, self.att_dim)

        self.numerical_literals = torch.from_numpy(numerical_literals).cuda()

        self.str = Mixture(self.emb_dim, self.p.num_mixture)

    def forward(self, g, e1, rel, e2_multi, neg, n_label):  # self.g, subj, rel, labels, neg, n_label
        e1_emb = torch.index_select(self.emb_e, 0, e1)  # (batch_size, emb_dim)
        rel_emb = torch.index_select(self.emb_rel, 0,
                                     rel)  # (batch_size, emb_dim)  e1_emb and rel_emb respectively from self.emb_e and self.emb_Select the embeddings of entities and relationships in rel
        e2_multi_emb = self.emb_e  # e2_multi_emb is an embedding of all entities
        W_r = torch.index_select(self.W_r, 0, rel)  # e2_multi_emb is an embedding of all entities

        e_str = self.str(e1_emb, rel_emb)  # [batch_size, emb_dim]

        # Numeric Value Embedding
        att = self.num_linear(self.numerical_literals.to(torch.float32).unsqueeze(-1)) * self.emb_att.unsqueeze(
            0) + self.att_linear(
            self.emb_att.unsqueeze(0))
        e1_emb_att = torch.index_select(att, 0, e1).transpose(0, 1)
        e2_emb_att = att.transpose(0, 1)

        # relation projection
        rel_emb_att = torch.tanh(self.linear(rel_emb)).unsqueeze(0)
        rel_emb_all_att = rel_emb_att.transpose(0, 1).repeat(1, e2_emb_att.shape[1],
                                                             1)

        # Relation-aware attention
        e1_num_lit, _ = self.multihead_attn(rel_emb_att, e1_emb_att, e1_emb_att)  # (1, batch_num, att_dim)
        e2_multi_num_lit, attention_matrix = self.multihead_attn(rel_emb_all_att, e2_emb_att,
                                                                 e2_emb_att)

        # Gating
        e1_emb = self.emb_num_lit(e1_emb, e1_num_lit.squeeze())  # (batch_num, emb_dim)
        e1_emb_w = self.emb_num_lit(e_str, e1_num_lit.squeeze())

        e2_emb_all = e2_multi_emb.repeat(e1_emb.shape[0], 1).view(-1, e2_emb_att.shape[1],
                                                                  self.emb_dim)  # (batch_num, num_entities, emb_dim)
        e2_multi_emb = self.emb_num_lit(e2_emb_all,
                                        e2_multi_num_lit)  # (batch_num, num_entities, emb_dim)
        e2_multi_emb_w = e2_multi_emb

        e1_emb = self.drop(e1_emb)  # E r,h (Head entity)
        e2_multi_emb = self.drop(e2_multi_emb)  # E r,i（All tail entities）

        # order compare
        e1_emb_w, e2_multi_emb_w = torch.matmul(e1_emb_w.unsqueeze(1), W_r), torch.bmm(e2_multi_emb_w, W_r)
        distance = e2_multi_emb_w - e1_emb_w
        distance = torch.clamp(distance, min=0)
        order_score = self.p.gamma - torch.square(distance).sum(2)

        # TransE
        score = self.p.gamma - \
                torch.norm(((e1_emb + rel_emb).unsqueeze(1) - e2_multi_emb), p=1, dim=2)

        pred = torch.sigmoid(score + self.p.order * order_score)

        if self.training:

            loss = self.calc_loss(pred, e2_multi)  # 18

            # Samples generator
            rand = torch.rand(e2_multi.shape[0], e2_multi.shape[1]).cuda()
            e2_multi_prob = e2_multi * rand
            e2_multi_inv_partial = n_label * pred * rand

            C = e2_multi_emb * e2_multi_prob.unsqueeze(-1)
            N = e2_multi_emb * e2_multi_inv_partial.unsqueeze(-1)

            positive_mean = C.sum(dim=1) / (e2_multi_prob.sum(-1).unsqueeze(-1) + 1e-8)  # (batch_num, emb_dim)
            negative_mean = N.sum(dim=1) / (
                        e2_multi_inv_partial.sum(-1).unsqueeze(-1) + 1e-8)  # (batch_num, emb_dim)

            alpha = torch.rand(e2_multi.shape[0], 1).cuda()
            beta = torch.rand(e2_multi.shape[0], 1).cuda()

            positive_mean = alpha * positive_mean + (1 - alpha) * e1_emb
            negative_mean = beta * negative_mean + (1 - beta) * e1_emb

            pos_score = torch.norm((e1_emb + rel_emb - positive_mean), p=1, dim=1) * neg
            neg_score = torch.norm((e1_emb + rel_emb - negative_mean), p=1, dim=1) * neg

            contrastive_loss = (-1.0) * (F.logsigmoid(neg_score - pos_score)).sum() / (neg.sum() + 1e-8)

            return loss + contrastive_loss * self.p.scale
        else:
            return pred  # Returns a prediction when it is a verifier or test set

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)
