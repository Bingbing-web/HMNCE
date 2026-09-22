import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvE(nn.Module):
    def __init__(self, emb_dim, feature_map_size, dropout=0.2):
        """Initialize convolutional scoring with the given embedding and feature dimensions."""
        super(ConvE, self).__init__()
        self.emb_dim = emb_dim
        self.feature_map_size = feature_map_size

        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=feature_map_size,
                              kernel_size=(3, 1))

        conv_output_height = 2 * emb_dim - 3 + 1
        conv_output_width = 1
        self.conv_out_dim = feature_map_size * conv_output_height * conv_output_width

        self.fc = nn.Linear(self.conv_out_dim, emb_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, head_emb, rel_emb, tail_emb):
        """Score candidate tails; inputs have shapes (B, D), (B, D), and (B, N, D)."""

        combined = torch.cat([head_emb, rel_emb], dim=1)

        # Stack head and relation features along the spatial dimension.
        combined = combined.view(-1, 1, 2 * self.emb_dim, 1)

        conv_out = F.relu(self.conv(combined))

        conv_out = conv_out.view(conv_out.size(0), -1)

        conv_feature = self.fc(conv_out)
        conv_feature = self.dropout(conv_feature)

        conv_feature_unsq = conv_feature.unsqueeze(1)

        score = torch.sum(conv_feature_unsq * tail_emb, dim=2)
        return score

class TuckER(nn.Module):
    def __init__(self, num_entities, num_relations, d_entity, d_relation, dropout=0.3):
        """Initialize the scoring module and its embedding dimensions."""
        super(TuckER, self).__init__()
        self.d_entity = d_entity
        self.d_relation = d_relation

        self.entity_embedding = nn.Embedding(num_entities, d_entity)
        self.relation_embedding = nn.Embedding(num_relations, d_relation)

        self.core_tensor = nn.Parameter(torch.randn(d_relation, d_entity, d_entity))

        self.input_dropout = nn.Dropout(dropout)
        self.hidden_dropout = nn.Dropout(dropout)
        self.bn0 = nn.BatchNorm1d(d_entity)
        self.bn1 = nn.BatchNorm1d(d_entity)

        self.init_weights()

    def init_weights(self):
        """Initialize embedding weights and the core tensor with Xavier uniform values."""
        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)
        nn.init.xavier_uniform_(self.core_tensor)

    def forward(self, head_idx, rel_idx, tail_idx):

        head = head_idx
        rel = rel_idx
        tail = tail_idx

        head = self.input_dropout(head)
        head = self.bn0(head)
        tail = self.input_dropout(tail)

        W_r = torch.einsum("br,rxy->bxy", rel, self.core_tensor)

        head = head.unsqueeze(1)

        score = torch.bmm(head, tail.transpose(1, 2))
        score = score.squeeze(1)

        return score

class ComplEx(nn.Module):
    def __init__(self, emb_dim):
        """Initialize complex embeddings with equal real and imaginary dimensions."""
        super(ComplEx, self).__init__()
        assert emb_dim % 2 == 0, "emb_dim must be even for complex embeddings."
        self.complex_dim = emb_dim // 2

    def forward(self, e1_emb, rel_emb, e2_multi_emb):
        """Return candidate scores of shape (B, N) from concatenated real/imaginary embeddings."""
        # Embeddings concatenate real and imaginary components.
        complex_dim = self.complex_dim

        e1_re = e1_emb[:, :complex_dim]
        e1_im = e1_emb[:, complex_dim:]
        rel_re = rel_emb[:, :complex_dim]
        rel_im = rel_emb[:, complex_dim:]
        e2_re = e2_multi_emb[:, :, :complex_dim]
        e2_im = e2_multi_emb[:, :, complex_dim:]

        temp1 = rel_re.unsqueeze(1) * e2_re + rel_im.unsqueeze(1) * e2_im

        temp2 = rel_im.unsqueeze(1) * e2_re - rel_re.unsqueeze(1) * e2_im

        score = (e1_re.unsqueeze(1) * temp1).sum(dim=2) - (e1_im.unsqueeze(1) * temp2).sum(dim=2)

        return score

class MuRP(nn.Module):
    def __init__(self, emb_dim, epsilon=1e-5):
        """Initialize hyperbolic scoring and its numerical stability constant."""
        super(MuRP, self).__init__()
        self.emb_dim = emb_dim
        self.epsilon = epsilon

    def mobius_matmul(self, m, x):
        """Apply diagonal Mobius multiplication to embeddings of shape (B, D)."""
        mx = m * x
        mx_norm = torch.norm(mx, dim=-1, keepdim=True)
        x_norm = torch.norm(x, dim=-1, keepdim=True)

        # Keep norms inside the Poincare ball.
        x_norm = torch.clamp(x_norm, max=1 - self.epsilon)

        return torch.tanh(mx_norm / x_norm * torch.atanh(x_norm)) * mx / (mx_norm + self.epsilon)

    def poincare_distance(self, u, v):
        """Return distance-based values for broadcast head and candidate-tail embeddings."""
        u_norm = (u ** 2).sum(dim=-1)
        v_norm = (v ** 2).sum(dim=-1)
        diff = u - v
        diff_norm = (diff ** 2).sum(dim=-1)

        dist = torch.log(1 + (2 * diff_norm) / ((1 - u_norm) * (1 - v_norm) + self.epsilon))
        return dist

    def forward(self, e1_emb, rel_emb, e2_multi_emb):
        """Return candidate scores of shape (B, N)."""

        r_diag = rel_emb

        e1_transformed = self.mobius_matmul(r_diag, e1_emb)

        e1_transformed = e1_transformed.unsqueeze(1)

        dist = self.poincare_distance(e1_transformed, e2_multi_emb)

        score = -dist

        return score

class RGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        """Initialize shared neighbor and self-message transformations."""
        super(RGCNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.rel_weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.rel_weight)

        self.self_weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.self_weight)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, e1_emb, rel_emb, e2_multi_emb):
        """Sum neighbor and self messages; return updated heads of shape (B, out_dim)."""
        batch_size, num_entities, _ = e2_multi_emb.shape

        rel_transform = torch.matmul(rel_emb, self.rel_weight)

        neighbor_transformed = torch.matmul(e2_multi_emb, self.rel_weight)

        neighbor_msg = neighbor_transformed.sum(dim=1)

        self_msg = torch.matmul(e1_emb, self.self_weight)

        total_msg = self_msg + neighbor_msg

        h = F.relu(total_msg)

        if self.dropout:
            h = self.dropout(h)

        return h

class RGCNModel(nn.Module):
    def __init__(self, emb_dim, hidden_dim=256, dropout=0.3):
        """Initialize the message layer and optional candidate-tail projection."""
        super(RGCNModel, self).__init__()
        self.rgcn_layer = RGCNLayer(emb_dim, hidden_dim, dropout)

        if emb_dim != hidden_dim:
            self.e2_proj = nn.Linear(emb_dim, hidden_dim)
        else:
            self.e2_proj = None

    def forward(self, e1_emb, rel_emb, e2_multi_emb):
        """Score candidate tails against updated head embeddings."""

        e1_updated = self.rgcn_layer(e1_emb, rel_emb, e2_multi_emb)

        if self.e2_proj:
            e2_transformed = self.e2_proj(e2_multi_emb)
        else:
            e2_transformed = e2_multi_emb

        score = torch.bmm(e1_updated.unsqueeze(1), e2_transformed.transpose(1, 2)).squeeze(1)

        return score

class BiGI(nn.Module):
    def __init__(self, emb_dim, hidden_dim=256, dropout=0.3):
        """Initialize relation-tail message aggregation and head updates."""
        super(BiGI, self).__init__()
        self.emb_dim = emb_dim

        self.message_layer = nn.Linear(emb_dim * 2, hidden_dim)

        self.update_layer = nn.Linear(hidden_dim + emb_dim, emb_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, e1_emb, rel_emb, e2_multi_emb):
        """Return candidate scores using updated heads and elementwise relation interactions."""
        batch_size, num_entities, _ = e2_multi_emb.shape

        rel_expanded = rel_emb.unsqueeze(1).expand(-1, num_entities, -1)

        message_input = torch.cat([rel_expanded, e2_multi_emb], dim=-1)

        messages = F.relu(self.message_layer(message_input))

        aggregated_msg = messages.sum(dim=1)

        update_input = torch.cat([e1_emb, aggregated_msg], dim=-1)

        e1_updated = self.update_layer(update_input)
        e1_updated = self.dropout(e1_updated)

        score = (e1_updated.unsqueeze(1) * rel_emb.unsqueeze(1) * e2_multi_emb).sum(dim=-1)

        return score
