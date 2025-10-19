import pandas as pd
import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
import networkx as nx
from torch.utils.data import random_split, Dataset, DataLoader
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --- load CSVs (fixed typo in test filename) ---
try:
    train_df = pd.read_csv('/content/train_weather_data.csv')
    test_df = pd.read_csv('/content/test_waether_data.csv')   # <-- corrected filename
    print("Training and testing data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Please run the file check command (!ls -l) and re-upload your CSV files if they are missing.")
    raise

# --- device setup (single correct device variable) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

fantasy_points_attr = [
    "runs",
    "wickets",
    "4s",
    "6s",
    "catches",
    "maidens",
    "overs",
    "runs gave",
    "out",
    "ducks",
    "balls played",
    "lbw/bowled",
    "runout(indirect)",
    "runout(direct)",
    "fantasy points",
]

DROP_COLUMNS = [
    'team1_player1_fantasy points',
    'team1_player2_fantasy points',
    'team1_player3_fantasy points',
    'team1_player4_fantasy points',
    'team1_player5_fantasy points',
    'team2_player1_fantasy points',
    'team2_player2_fantasy points',
    'team2_player3_fantasy points',
    'team2_player4_fantasy points',
    'team2_player5_fantasy points'
]

TARGET_COLUMNS = []
for j in range(1, 3):
    for i in range(1, 12):
        for attr in fantasy_points_attr:
            TARGET_COLUMNS.append(f"team{j}_player{i}_{attr}")

existing_target_cols_train = [col for col in TARGET_COLUMNS if col in train_df.columns]
existing_drop_cols_train = [col for col in DROP_COLUMNS if col in train_df.columns]

existing_target_cols_test = [col for col in TARGET_COLUMNS if col in test_df.columns]
existing_drop_cols_test = [col for col in DROP_COLUMNS if col in test_df.columns]

grp1_columns = []
grp2_columns = []
grp3_columns = []
grp1_size = 0
grp2_size = 0
grp3_size = 0

for column_name in train_df.columns:
    if column_name in existing_drop_cols_train:
        continue
    if 'previous3' in column_name:
        grp2_columns.append(column_name)
        grp2_size += 1
    elif 'lifetime' in column_name:
        grp3_columns.append(column_name)
        grp3_size += 1
    else:
        grp1_columns.append(column_name)
        grp1_size += 1

# --- Dataset that returns list of group tensors and target ---
class GroupedCSVLoader(Dataset):
    def __init__(self, csv_path, target_cols, group1_cols, group2_cols, group3_cols):
        self.data = pd.read_csv(csv_path)
        self.target_cols = [c for c in target_cols if c in self.data.columns]
        self.group_cols = [group1_cols, group2_cols, group3_cols]

        # convert to torch tensors (keeps them on CPU for now; we'll move to device in training loop)
        if len(self.target_cols) == 0:
            # fallback to zeros if no target columns exist (prevents crashes)
            self.targets = torch.zeros((len(self.data), 1), dtype=torch.float32)
        else:
            self.targets = torch.tensor(self.data[self.target_cols].values, dtype=torch.float32)

        self.group_data = []
        for group in self.group_cols:
            if len(group) == 0:
                # placeholder 1-d zero column if group empty
                arr = np.zeros((len(self.data), 1), dtype=np.float32)
            else:
                arr = self.data[group].values.astype(np.float32)
            self.group_data.append(torch.tensor(arr, dtype=torch.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return list of tensors (each shape (k_i,)) and target vector
        group_vectors = [g[idx] for g in self.group_data]
        target = self.targets[idx]
        return group_vectors, target

# --- Model components (unchanged, but keep consistent device handling outside) ---
class GroupProjector(nn.Module):
    def __init__(self, group_feature_sizes, d):
        super().__init__()
        self.projectors = nn.ModuleList([nn.Linear(k, d) for k in group_feature_sizes])

    def forward(self, group_inputs):
        outs = [proj(inp) for proj, inp in zip(self.projectors, group_inputs)]
        return torch.stack(outs, dim=1)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, M, D = x.shape
        Q = self.Wq(x).view(B, M, self.num_heads, self.head_dim).transpose(1,2)
        K = self.Wk(x).view(B, M, self.num_heads, self.head_dim).transpose(1,2)
        V = self.Wv(x).view(B, M, self.num_heads, self.head_dim).transpose(1,2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        context = context.transpose(1,2).contiguous().view(B, M, D)
        out = self.out(context)
        return out, attn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, attn_weights = self.self_attn(x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x, attn_weights

class RegressionHead(nn.Module):
    def __init__(self, d_model, hidden, num_outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_outputs)
        )

    def forward(self, x):
        pooled = x.mean(dim=1)
        return self.net(pooled)

class TransformerForTabular(nn.Module):
    def __init__(self, group_feature_sizes, d_model, num_layers, num_heads, hidden, n_classes):
        super().__init__()
        self.projector = GroupProjector(group_feature_sizes, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, hidden) for _ in range(num_layers)])
        self.head = RegressionHead(d_model, hidden, n_classes)

    def forward(self, group_inputs):
        # group_inputs: list length m of (B, k_i)
        x = self.projector(group_inputs)  # (B, m, d)
        attns = []
        for layer in self.layers:
            x, attn = layer(x)
            attns.append(attn)
        logits = self.head(x)
        return logits, attns

# --- dataset / dataloaders ---
dataset = GroupedCSVLoader('/content/train_weather_data.csv', TARGET_COLUMNS, grp1_columns, grp2_columns, grp3_columns)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --- instantiate teacher and student on device ---
num_outputs = len(dataset.targets[0]) if dataset.targets.ndim > 1 else 1
teacher = TransformerForTabular([grp1_size, grp2_size, grp3_size], 256, 6, 4, 256, num_outputs).to(device)
student = TransformerForTabular([grp1_size, grp2_size, grp3_size], 128, 8, 1, 128, num_outputs).to(device)

criterion = nn.MSELoss()

teacher_optimizer = optim.Adam(teacher.parameters(), lr=1e-5)
student_optimizer = optim.Adam(student.parameters(), lr=1e-4)

num_epochs = 400

# helper to move list-of-tensors (group vectors) to device
def move_group_vectors_to_device(group_vectors, device):
    return [g.to(device) for g in group_vectors]

# --- Teacher training ---
for epoch in range(num_epochs):
    teacher.train()
    running_loss = 0.0
    total_samples = 0
    for group_vectors, target in train_dataloader:
        # move to device
        group_vectors = move_group_vectors_to_device(group_vectors, device)
        target = target.to(device)

        teacher_optimizer.zero_grad()
        logits, _ = teacher(group_vectors)
        loss = criterion(logits, target)
        loss.backward()
        teacher_optimizer.step()

        batch_size = target.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Teacher Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")
def custom_mape(y_true, y_pred):
    """
    Calculates the Mean Absolute Percentage Error, handling cases where y_true is zero.
    """
    
    y_true=torch.clamp(y_true,min=0.1)
    return (np.abs((y_true - y_pred) / y_true)) * 100
# --- Teacher evaluation (use test_dataloader) ---
teacher.eval()
running_test_loss = 0.0
mae_sum = 0.0
mape_sum = 0.0
total_elements = 0
with torch.no_grad():
    for group_vectors, target in test_dataloader:   # <-- use test dataloader
        group_vectors = move_group_vectors_to_device(group_vectors, device)
        target = target.to(device)
        logits, _ = teacher(group_vectors)
        loss = criterion(logits, target)

        batch_size = target.size(0)
        running_test_loss += loss.item() * batch_size

        preds = logits
        # flatten element-wise metrics
        diff = torch.abs(preds - target)
        mae_sum += diff.sum().item()
        # non_zero_target = torch.where(target == 0, torch.ones_like(target), target)
        
        mape_sum += custom_mape(preds,target)
        total_elements += preds.numel()

test_loss = running_test_loss / len(test_dataloader.dataset)
mae = mae_sum / total_elements
mape = mape_sum / total_elements
print(f"Teacher Test Loss: {test_loss:.6f}")
print(f"Teacher MAE per-element: {mae:.6f}")
print(f"Teacher MAPE per-element: {mape:.6f}")

# --- helper: teacher soft labels (returns CPU tensor of preds) ---
def teacher_soft_labels_regression(teacher, dataloader, device=device):
    teacher.eval()
    all_preds = []
    with torch.no_grad():
        for group_vectors, _ in dataloader:
            group_vectors = move_group_vectors_to_device(group_vectors, device)
            preds, _ = teacher(group_vectors)
            all_preds.append(preds.cpu())
    return torch.cat(all_preds, dim=0)

# --- attention penalty ---
def attention_entropy_penalty(attns, eps=1e-12):
    penalty = 0.0
    for a in attns:
        # a shape: (B, H, m, m)
        avg = a.mean(dim=1)  # (B, m, m)
        entropy = -avg * torch.log(avg + eps)
        penalty += entropy.sum(dim=[1,2]).mean()
    return penalty

# --- Distillation: train student to match teacher's outputs ---
# Precompute teacher predictions for entire training set (optional; can also compute online)
teacher_preds_all = teacher_soft_labels_regression(teacher, train_dataloader, device=device)  # shape (N, num_outputs)

# If you prefer online computation, you can remove the precompute and call teacher inside loop.

student.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    total_samples = 0
    idx_offset = 0
    for (group_vectors, _), batch_idx in zip(train_dataloader, range(len(train_dataloader))):
        
        group_vectors = move_group_vectors_to_device(group_vectors, device)

        batch_size = group_vectors[0].size(0)
        
        teacher_batch_preds = teacher_preds_all[idx_offset: idx_offset + batch_size].to(device)
        idx_offset += batch_size

        logits, attns = student(group_vectors)
        loss_reg = criterion(logits, teacher_batch_preds)
        loss_att = 0.1 * attention_entropy_penalty(attns)
        loss = loss_reg + loss_att

        student_optimizer.zero_grad()
        loss.backward()
        student_optimizer.step()

        running_loss += loss.item() * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Student Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.6f}")

# --- Student evaluation on test_dataloader ---
student.eval()
running_test_loss = 0.0
mae_sum = 0.0
mape_sum = 0.0
total_elements = 0
with torch.no_grad():
    for group_vectors, target in test_dataloader:
        group_vectors = move_group_vectors_to_device(group_vectors, device)
        target = target.to(device)
        logits, _ = student(group_vectors)
        loss = criterion(logits, target)

        batch_size = target.size(0)
        running_test_loss += loss.item() * batch_size

        diff = torch.abs(logits - target)
        mae_sum += diff.sum().item()
        mape_sum += custom_mape(preds,target)
        total_elements += logits.numel()

test_loss = running_test_loss / len(test_dataloader.dataset)
mae = mae_sum / total_elements
mape = mape_sum / total_elements

print(f"Student Test Loss: {test_loss:.6f}")
print(f"Student MAE per-element: {mae:.6f}")
print(f"Student MAPE per-element: {mape:.6f}")

# --- Graph utilities (unchanged) ---
eps = 1e-12

def build_graph(layer_matrices, eps=1e-9):
    M = len(layer_matrices)
    m = layer_matrices[0].shape[-1]
    G = nx.DiGraph()
    for l in range(M + 1):
        for i in range(m):
            G.add_node((l, i))
    for l in range(1, M + 1):
        # take first batch and average heads
        A = layer_matrices[l-1][0].mean(dim=0)
        for i in range(m):
            for j in range(m):
                weight = -math.log(A[i, j].item() + eps)
                G.add_edge((l-1, i), (l, j), weight=weight)
    return G

def find_best_start(G, M, m):
    best_cost = np.inf
    best_start = None
    best_path = None
    for start in range(m):
        for target in range(m):
            try:
                cost = nx.dijkstra_path_length(G, (0, start), (M, target), weight='weight')
                if cost < best_cost:
                    best_cost = cost
                    best_start = start
                    best_path = nx.dijkstra_path(G, (0, start), (M, target), weight='weight')
            except nx.NetworkXNoPath:
                continue
    return best_start, best_path, best_cost

# student.eval()
# sample_input,sample_target=next(iter(test_dataloader))
# with torch.no_grad():
#     logits,sample_attention=student(sample_input)
# print(len(sample_attention))
# print(sample_attention[0].shape)
# G=build_graph(sample_attention)
# print('Graph Constructed Successfully')

# best_start, best_path, best_cost = find_best_start(G, len(sample_attention), 3)
# print('Best Start:',best_start)
# print('Best path:',best_path)
# print('Best Cost:',best_cost)

# pos = {}
# for l in range(len(sample_attention[0]) + 1):  # +1 because graph has M+1 layers of nodes
#     for c in range(3):  # 3 = number of concept groups
#         pos[(l, c)] = (l, -c)  # (-c) to display groups top-to-bottom

# plt.figure(figsize=(7, 5))
# nx.draw(G, pos, with_labels=True, node_size=700, font_size=8, node_color="lightblue")

# # Highlight best path in red
# path_edges = list(zip(best_path, best_path[1:]))
# nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=2)

# plt.title(f"Best Path (Start Group: {best_start}, Cost: {best_cost:.4f})", fontsize=10)
# plt.show()
# student.eval()
# sample_input,sample_target=next(iter(test_dataloader))
# with torch.no_grad():
#     logits,sample_attention=student(sample_input)
# print(len(sample_attention))
# print(sample_attention[0].shape)
# G=build_graph(sample_attention)
# print('Graph Constructed Successfully')

# best_start, best_path, best_cost = find_best_start(G, len(sample_attention), 3)
# print('Best Start:',best_start)
# print('Best path:',best_path)
# print('Best Cost:',best_cost)

# pos = {}
# for l in range(len(sample_attention[0]) + 1):  # +1 because graph has M+1 layers of nodes
#     for c in range(3):  # 3 = number of concept groups
#         pos[(l, c)] = (l, -c)  # (-c) to display groups top-to-bottom

# plt.figure(figsize=(7, 5))
# nx.draw(G, pos, with_labels=True, node_size=700, font_size=8, node_color="lightblue")

# # Highlight best path in red
# path_edges = list(zip(best_path, best_path[1:]))
# nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=2)

# plt.title(f"Best Path (Start Group: {best_start}, Cost: {best_cost:.4f})", fontsize=10)
# plt.show()