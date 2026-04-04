# Graph Machine Learning

## Step 1 — Load a Pre-built City Graph

To skip the build step, load a pre-computed Singapore graph directly from the [Global Urban Graph Dataset](https://figshare.com/ndownloader/files/53940917):

```python
import numpy as np
import urbanity as urb
from urbanity.data_class import UrbanGraph
import geopandas as gpd

sg_graph = UrbanGraph()
sg_graph.load_graph(filename='path_to_graph_file')
sg_graph.initialize_edges(building_neighbours='knn', knn=3, distance=100)
sg_graph
```

Inspect the node counts:

```python
for layer, gdf in sg_graph.geo_store.items():
    if hasattr(gdf, '__len__'):
        print(f"{layer:>14s}: {len(gdf):>7,} features")
# boundary:           1
# building:     118,979
# plot:          30,121
# street:       120,045
# intersection:  33,xxx
```

Inspect building attributes:

```python
sg_graph.geo_store['building'].head(5)
```

---

## Step 2 — Load Training Labels

Combine building use types from OpenStreetMap and the Singapore Building Benchmarking Dataset (BCA) - Download from [Labels](https://figshare.com/ndownloader/files/54728558)

```python
target = gpd.read_file('path_to_label_file')

category_mapping = [
    'Transportation', 'Mixed Use', 'Commercial', 'Institutional',
    'Residential', 'Religious', 'Industrial', 'Healthcare',
    'Education', 'Sports', 'Arts', 'Agricultural'
]
category_mapping = {float(v): k for k, v in zip(category_mapping, range(12))}

target['category_numeric'] = (
    target['category_numeric']
    .replace(np.nan, None)
    .replace(category_mapping)
)
target.head(5)
```

The 12 building use classes:

| ID | Class |
|----|-------|
| 0 | Transportation |
| 1 | Mixed Use |
| 2 | Commercial |
| 3 | Institutional |
| 4 | Residential |
| 5 | Religious |
| 6 | Industrial |
| 7 | Healthcare |
| 8 | Education |
| 9 | Sports |
| 10 | Arts |
| 11 | Agricultural |

---

## Step 3 — Convert to PyTorch Geometric

`to_pyg_graph()` handles data splitting, standardisation, and mask generation automatically:

```python
data = sg_graph.to_pyg_graph(
    target_node='building',
    categorical=True,                              # classification task
    train_val_test=[0.8, 0.1, 0.1],
    random_seed=0,
    target_value=list(target['category_numeric'].values),
)
data
```

---

## Step 4 — Set Hyperparameters & Dataloaders

```python
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import Linear, HeteroConv, SAGEConv
from torch_geometric.loader import NeighborLoader

device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_dim  = 32
lr          = 0.0001
batch_size  = 32
EPOCH       = 100

train_len = data['building'].train_mask.sum().item()
val_len   = data['building'].val_mask.sum().item()

train_loader = NeighborLoader(
    data, num_neighbors=[8, 8], batch_size=batch_size,
    input_nodes=('building', data['building'].train_mask),
    shuffle=True, drop_last=True,
)
val_loader = NeighborLoader(
    data, num_neighbors=[8, 8], batch_size=batch_size,
    input_nodes=('building', data['building'].val_mask),
    shuffle=True, drop_last=True,
)
```

---

## Step 5 — Focal Loss (for class imbalance)

Building use types are heavily imbalanced (most buildings are residential). Focal loss downweights easy examples and focuses training on hard, rare classes:

```python
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        alpha: class-wise weight tensor, or None
        gamma: focusing parameter — higher values focus more on hard examples
        """
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        ce_loss    = F.cross_entropy(logits, target, weight=self.alpha, reduction='none')
        pt         = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

focal_loss_fn = FocalLoss(alpha=None, gamma=2.0, reduction='mean').to(device)
```

---

## Step 6 — Heterogeneous GraphSAGE Architecture

The model uses two `HeteroConv` layers, each containing `SAGEConv` operators for every edge type in the urban graph. This allows information to flow across all four node types simultaneously:

```python
class HeteroSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()

        edge_types = {
            ('plot',         'to', 'plot'):         SAGEConv((-1, -1), hidden_channels),
            ('building',     'to', 'building'):     SAGEConv((-1, -1), hidden_channels),
            ('intersection', 'to', 'street'):       SAGEConv((-1, -1), hidden_channels),
            ('street',       'to', 'intersection'): SAGEConv((-1, -1), hidden_channels),
            ('plot',         'to', 'building'):     SAGEConv((-1, -1), hidden_channels),
            ('building',     'to', 'plot'):         SAGEConv((-1, -1), hidden_channels),
            ('plot',         'to', 'street'):       SAGEConv((-1, -1), hidden_channels),
            ('street',       'to', 'plot'):         SAGEConv((-1, -1), hidden_channels),
            ('building',     'to', 'street'):       SAGEConv((-1, -1), hidden_channels),
            ('street',       'to', 'building'):     SAGEConv((-1, -1), hidden_channels),
        }
        self.conv1 = HeteroConv(edge_types, aggr='mean')
        self.conv2 = HeteroConv(edge_types, aggr='mean')
        self.fc1   = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.dropout(v.relu(), p=0.5, training=self.training)
                  for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.dropout(v.relu(), p=0.5, training=self.training)
                  for k, v in x_dict.items()}
        return self.fc1(x_dict['building'])

# Instantiate model and optimizer
graph_model = HeteroSAGE(hidden_channels=hidden_dim, out_channels=12).to(device)
optimizer   = torch.optim.Adam(graph_model.parameters(), lr=lr)
scheduler   = ReduceLROnPlateau(optimizer, 'min', patience=10)

# Sanity-check forward pass
with torch.no_grad():
    sample = next(iter(train_loader)).to(device)
    out = graph_model(sample.x_dict, sample.edge_index_dict)
    print("Output shape:", out.shape)  # (batch_size, 12)
```

---

## Step 7 — Training & Validation Loop

```python
best_accuracy = 0.75

for epoch in range(EPOCH):
    epoch_train_loss = 0
    epoch_val_loss   = 0
    epoch_val_correct = 0
    epoch_val_total   = 0

    # ── Train ────────────────────────────────────────────────────
    graph_model.train()
    for train_batch in tqdm(train_loader):
        optimizer.zero_grad()
        train_batch = train_batch.to(device)

        pred       = graph_model(train_batch.x_dict, train_batch.edge_index_dict)
        train_loss = focal_loss_fn(
            pred[:batch_size],
            train_batch['building'].y[:batch_size].long(),
        )
        train_loss.backward()
        optimizer.step()
        epoch_train_loss += train_loss.item() * batch_size

    epoch_train_loss /= train_len
    print(f"Epoch {epoch:>3} | Train loss: {epoch_train_loss:.5f}")

    # ── Validate ─────────────────────────────────────────────────
    graph_model.eval()
    with torch.no_grad():
        for val_batch in tqdm(val_loader):
            val_batch = val_batch.to(device)
            val_pred  = graph_model(val_batch.x_dict, val_batch.edge_index_dict)

            val_loss = focal_loss_fn(
                val_pred[:batch_size],
                val_batch['building'].y[:batch_size].long(),
            )
            epoch_val_loss += val_loss.item() * batch_size

            pred_labels = val_pred[:batch_size].argmax(dim=1)
            correct = (pred_labels == val_batch['building'].y[:batch_size]).sum().item()
            epoch_val_correct += correct
            epoch_val_total   += val_batch['building'].y[:batch_size].size(0)

    epoch_val_loss /= val_len
    val_acc = epoch_val_correct / epoch_val_total if epoch_val_total > 0 else 0
    print(f"Epoch {epoch:>3} | Val loss: {epoch_val_loss:.5f} | Val acc: {val_acc:.4f}")

    scheduler.step(epoch_val_loss)

    # Save best checkpoint
    if val_acc > best_accuracy:
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     graph_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss':                 epoch_val_loss,
        }, f'./graphsage_{val_acc:.4f}.pt')
        best_accuracy = val_acc
        print("─── Saved checkpoint ───")
```

The model achieves ~74% accuracy across 12 building use classes — without any satellite imagery.

---

## Step 8 — Inference on All Buildings

```python
batch_size_inf = 1
all_loader = NeighborLoader(
    data, num_neighbors=[8, 8], batch_size=batch_size_inf,
    input_nodes='building', shuffle=False, drop_last=False,
)

all_pred_targets = []
graph_model.eval()
with torch.no_grad():
    for all_batch in tqdm(all_loader):
        all_batch = all_batch.to(device)
        all_pred  = graph_model(all_batch.x_dict, all_batch.edge_index_dict)
        all_pred_targets += all_pred[:batch_size_inf]

all_preds_list = [t.argmax().item() for t in all_pred_targets]
np.savetxt('./final_building_use.txt', all_preds_list)
```

---

## Step 9 — Visualise Predictions

Map predicted labels back to category names and visualise with KeplerGL:

```python
category_mapping = {
    float(i): name for i, name in enumerate([
        'Transportation', 'Mixed Use', 'Commercial', 'Institutional',
        'Residential', 'Religious', 'Industrial', 'Healthcare',
        'Education', 'Sports', 'Arts', 'Agricultural'
    ])
}

# Option A: use your own predictions
target['preds'] = all_preds_list

# Option B: download pre-computed predictions
import requests
from io import StringIO
resp = requests.get('https://figshare.com/ndownloader/files/54728555', timeout=30)
target['preds'] = np.loadtxt(StringIO(resp.text), dtype=float)

target['preds'] = target['preds'].replace(category_mapping)
target.to_file('./singapore_buildings.geojson')
```

```python
!uv pip install keplergl

from keplergl import KeplerGl
import geopandas as gpd

map_1 = KeplerGl(height=800)
gdf = gpd.read_file('./singapore_buildings.geojson')
map_1.add_data(data=gdf, name="buildings")
map_1
```

---

## Train / Val / Test Masks

You can also generate masks directly from the `UrbanGraph` object:

```python
train_mask, val_mask, test_mask = sg_graph.generate_building_masks(
    train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
)
```

Equivalent helpers: `generate_plot_masks`, `generate_street_masks`, `generate_intersection_masks`.

---

## Summary

| Step | What happens |
|------|-------------|
| `load_graph(url)` | Load a pre-built city graph from Figshare |
| `initialize_edges(knn=3)` | Populate inter-layer adjacency |
| `to_pyg_graph(categorical=True, ...)` | Get a `HeteroData` object with masks |
| `HeteroSAGE` | 2-layer heterogeneous SAGE over all 10 edge types |
| `FocalLoss` | Handles class imbalance across 12 building types |
| `NeighborLoader` | Mini-batch training with 2-hop neighbourhood sampling |
| Save best checkpoint | `torch.save(...)` on improvement in val accuracy |
| Inference → KeplerGL | Visualise city-wide predictions interactively |
