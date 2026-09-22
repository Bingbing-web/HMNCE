from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def get_param(shape):
    parameter = nn.Parameter(torch.empty(*shape))
    nn.init.xavier_normal_(parameter.data)
    return parameter


class NumericValueEmbedding(nn.Module):
    def __init__(self, num_att, att_dim):
        super().__init__()
        self.v = nn.Parameter(torch.empty(num_att, att_dim))
        self.W_x = nn.Parameter(torch.empty(num_att, att_dim))
        self.w = nn.Parameter(torch.empty(num_att, att_dim))
        self.missing_bias = nn.Parameter(torch.empty(num_att, att_dim))
        nn.init.xavier_normal_(self.v)
        nn.init.xavier_normal_(self.W_x)
        nn.init.xavier_normal_(self.w)
        nn.init.xavier_normal_(self.missing_bias)

    def forward(self, values, observed_mask=None):
        value_embedding = (
            self.W_x.unsqueeze(0) + self.w.unsqueeze(0) * values.unsqueeze(-1)
        ) * self.v.unsqueeze(0)
        if observed_mask is None:
            observed_mask = (values != 0).to(value_embedding.dtype)
        observed_mask = observed_mask.to(value_embedding.dtype).unsqueeze(-1)
        missing_embedding = self.missing_bias.unsqueeze(0).expand_as(value_embedding)
        return observed_mask * value_embedding + (1.0 - observed_mask) * missing_embedding


class Gate(nn.Module):
    def __init__(self, input_size, output_size, gate_activation=torch.sigmoid):
        super().__init__()
        self.output_size = output_size
        self.gate_activation = gate_activation
        self.g = nn.Linear(input_size, output_size)
        self.g1 = nn.Linear(output_size, output_size, bias=False)
        self.g2 = nn.Linear(input_size - output_size, output_size, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, entity_embedding, literal_embedding):
        combined = torch.cat(
            [entity_embedding, literal_embedding], literal_embedding.ndimension() - 1
        )
        embedded = torch.tanh(self.g(combined))
        gate = self.gate_activation(
            self.g1(entity_embedding)
            + self.g2(literal_embedding)
            + self.gate_bias
        )
        return (1.0 - gate) * entity_embedding + gate * embedded


class Mixture(nn.Module):
    def __init__(self, embedding_dim, num_mixtures, stochastic_eval=False):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.stochastic_eval = stochastic_eval
        self.mixtures = nn.ModuleList(
            [nn.Linear(embedding_dim, embedding_dim) for _ in range(num_mixtures)]
        )
        self.gate = nn.Linear(embedding_dim, num_mixtures)
        self.temp_layer = nn.Linear(embedding_dim, 1)

    def forward(self, entity_embedding, relation_embedding):
        mixture_outputs = torch.cat(
            [layer(entity_embedding).unsqueeze(1) for layer in self.mixtures], dim=1
        )
        gate_logits = self.gate(entity_embedding)
        if self.training or self.stochastic_eval:
            noise = torch.randn_like(gate_logits)
        else:
            noise = torch.zeros_like(gate_logits)
        temperature = torch.sigmoid(self.temp_layer(relation_embedding)).clamp_min(0.1)
        weights = torch.softmax((gate_logits + noise) / temperature, dim=1).unsqueeze(-1)
        return torch.sum(mixture_outputs * weights, dim=1)


class HMNCEInference(nn.Module):
    """HMNCE inference for the full model with TransE structural scoring."""

    def __init__(self, num_ents, num_rels, numerical_literals, params):
        super().__init__()
        self.bceloss = nn.BCELoss()
        self.p = params
        self.emb_dim = params.init_dim
        self.att_dim = params.att_dim
        self.head_num = params.head_num
        self.num_ents = num_ents
        self.num_rels = num_rels

        self.emb_e = get_param((num_ents, params.init_dim))
        self.emb_rel = get_param((num_rels, params.init_dim))
        self.num_att = numerical_literals.shape[1]
        self.emb_att = get_param((self.num_att, self.att_dim))
        self.num_embedder = NumericValueEmbedding(self.num_att, self.att_dim)
        self.W_r = get_param((num_rels, params.init_dim, params.init_dim))
        self.linear = nn.Linear(self.emb_dim, self.att_dim)
        self.multihead_attn = nn.MultiheadAttention(
            self.att_dim, self.head_num, batch_first=False
        )
        self.emb_num_lit = Gate(self.att_dim + self.emb_dim, self.emb_dim)
        self.drop = nn.Dropout(p=params.drop)
        self.att_linear = nn.Linear(self.att_dim, self.att_dim)
        self.num_linear = nn.Linear(1, self.att_dim)

        device = getattr(params, "device", torch.device("cuda"))
        self.numerical_literals = torch.as_tensor(
            numerical_literals, dtype=torch.float32, device=device
        )
        self.literal_mask = (self.numerical_literals != 0).float()
        self.str = Mixture(
            self.emb_dim,
            params.num_mixture,
            stochastic_eval=bool(getattr(params, "stochastic_eval", False)),
        )

    def forward(self, graph, heads, relations, labels=None, neg=None, negative_labels=None):
        del graph, neg, negative_labels
        head_embedding = torch.index_select(self.emb_e, 0, heads)
        relation_embedding = torch.index_select(self.emb_rel, 0, relations)
        all_entity_embedding = self.emb_e
        relation_matrix = torch.index_select(self.W_r, 0, relations)

        structured_embedding = self.str(head_embedding, relation_embedding)
        attribute_embedding = self.num_embedder(
            self.numerical_literals, self.literal_mask
        )
        head_attributes = torch.index_select(attribute_embedding, 0, heads).transpose(0, 1)
        all_attributes = attribute_embedding.transpose(0, 1)

        relation_attributes = torch.tanh(self.linear(relation_embedding)).unsqueeze(0)
        all_relation_attributes = relation_attributes.transpose(0, 1).repeat(
            1, all_attributes.shape[1], 1
        )
        head_numeric, _ = self.multihead_attn(
            relation_attributes, head_attributes, head_attributes
        )
        all_numeric, _ = self.multihead_attn(
            all_relation_attributes, all_attributes, all_attributes
        )
        head_numeric = head_numeric.squeeze(0)

        gated_head = self.emb_num_lit(head_embedding, head_numeric)
        ordered_head = self.emb_num_lit(structured_embedding, head_numeric)
        expanded_entities = all_entity_embedding.repeat(gated_head.shape[0], 1).view(
            -1, all_attributes.shape[1], self.emb_dim
        )
        gated_entities = self.emb_num_lit(expanded_entities, all_numeric)
        ordered_entities = gated_entities

        gated_head = self.drop(gated_head)
        gated_entities = self.drop(gated_entities)
        ordered_head = torch.matmul(ordered_head.unsqueeze(1), relation_matrix)
        ordered_entities = torch.bmm(ordered_entities, relation_matrix)
        order_distance = torch.clamp(ordered_entities - ordered_head, min=0)
        order_score = self.p.gamma - torch.square(order_distance).sum(2)
        score = self.p.gamma - torch.norm(
            (gated_head + relation_embedding).unsqueeze(1) - gated_entities,
            p=1,
            dim=2,
        )
        prediction = torch.sigmoid(score + self.p.order * order_score)
        prediction = torch.nan_to_num(
            prediction, nan=0.5, posinf=1.0, neginf=0.0
        ).clamp(1e-7, 1.0 - 1e-7)
        if self.training and labels is not None:
            return self.bceloss(prediction, labels)
        return prediction

