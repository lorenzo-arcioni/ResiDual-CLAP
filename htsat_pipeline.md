# HTS-AT Transformer Pipeline

## Convenzioni e notazione rapida

| Simbolo | Significato |
|---------|-------------|
| $N_w^\ell$ | Numero di finestre di attenzione allo stage $\ell$ |
| $M = 64$ | Token per finestra ($8 \times 8$) |
| $d_h = 24$ | Dimensione per testa (costante) |
| $D_\ell$ | Embedding dimension allo stage $\ell$ |
| $H_\ell$ | Numero di teste allo stage $\ell$ |
| $S_\ell$ | Lato della griglia spaziale allo stage $\ell$ |

**Splits del blocco QKV** — due operazioni distinte, da non confondere:

1. **Split lineare:** $W^{QKV}_\ell \in \mathbb{R}^{D_\ell \times 3D_\ell}$ proietta $\mathbf{Z}$ da $D_\ell$ a $3D_\ell$, poi si divide in tre blocchi uguali per ottenere $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N_w \times M \times D_\ell}$.

2. **Split per testa:** ciascuno dei tre tensori viene ridimensionato da $\mathbb{R}^{N_w \times M \times D_\ell}$ a $\mathbb{R}^{N_w \times H_\ell \times M \times d_h}$ (reshape + permute), isolando così la slice $[h]$ per ogni testa $h$: $\mathbf{Q}_h, \mathbf{K}_h, \mathbf{V}_h \in \mathbb{R}^{N_w \times M \times d_h}$.

## Input al Transformer

$$\mathbf{Z}^{(0)} = \mathbf{X}_0 \in \mathbb{R}^{4096 \times 96}$$

$4096 = 64 \times 64$ token, ciascuno di dimensione $D_0 = 96$.

## Stage 0 — $\ell = 0$

**Configurazione:** 2 blocchi · $H_0 = 4$ teste · $D_0 = 96$ · $S_0 = 64$

Risoluzione spaziale: $64 \times 64 = 4096$ token · Finestre $8 \times 8$ → $N_w^0 = 4096 / 64 = 64$ finestre.

### Blocco $b \in \{1, 2\}$

#### Step 1 — LayerNorm + W-MSA (o SW-MSA se $b$ pari)

**Proiezione QKV unificata** — *Split 1 (lineare):*

$$\underbrace{\mathrm{LN}(\mathbf{Z}^{(0)}) \cdot W^{QKV}_0}_{\mathbb{R}^{4096 \times 288}} \xrightarrow{\text{split in 3}} \mathbf{Q}^{(0)},\, \mathbf{K}^{(0)},\, \mathbf{V}^{(0)} \in \mathbb{R}^{64 \times 64 \times 96}$$

con $W^{QKV}_0 \in \mathbb{R}^{96 \times 288}$ (dove $288 = 3 \times 96$).

**Separazione per testa** — *Split 2 (reshape + permute):*

$$\mathbf{Q}^{(0)} \xrightarrow{\text{reshape}} \mathbb{R}^{64 \times 64 \times 4 \times 24} \xrightarrow{\text{permute}} \mathbb{R}^{64 \times 4 \times 64 \times 24}$$

Per ogni testa $h \in \{1, 2, 3, 4\}$, si estrae la slice $[h]$:

$$\mathbf{Q}_{0,b,h},\; \mathbf{K}_{0,b,h},\; \mathbf{V}_{0,b,h} \;\in\; \mathbb{R}^{N_w^0 \times M \times d_h}, \text{i.e. } \mathbb{R}^{64 \times 64 \times 24}$$

**Attention per testa $h$:**

$$\mathbf{H}_{0,b,h} = \mathrm{Softmax}\!\left(\frac{\mathbf{Q}_{0,b,h}\,\mathbf{K}_{0,b,h}^\top}{\sqrt{d_h}} + \mathbf{B}_{0,b,h}\right)\mathbf{V}_{0,b,h} \;\in\; \mathbb{R}^{64 \times 64 \times 24}$$

dove $\mathbf{B}_{0,b,h} \in \mathbb{R}^{64 \times 64}$ è il relative position bias per la testa $h$.

**Proiezione output e contributo per testa:**

$$W^O_0 \in \mathbb{R}^{96 \times 96}, \qquad W^O_{0,h} \in \mathbb{R}^{24 \times 96} \;\text{(righe } [(h-1)\cdot24,\; h\cdot24) \text{ di } W^O_0\text{)}$$

$$\widehat{\mathbf{H}}_{0,b,h} = \mathbf{H}_{0,b,h}\,W^O_{0,h} + \frac{\mathbf{b}^O_0}{H_0} \;\in\; \mathbb{R}^{64 \times 64 \times 96}$$

**Somma su tutte le teste e aggiornamento residuale:**

$$\mathbf{A}_{0,b} = \sum_{h=1}^{4} \widehat{\mathbf{H}}_{0,b,h} \;\in\; \mathbb{R}^{4096 \times 96}$$

$$\mathbf{Z}^{(0)} \;\leftarrow\; \mathbf{Z}^{(0)} + \mathrm{DropPath}(\mathbf{A}_{0,b}) \;\in\; \mathbb{R}^{4096 \times 96}$$

#### Step 2 — LayerNorm + MLP

$$\mathbf{M}_{0,b} = \mathrm{MLP}_0\!\left(\mathrm{LN}(\mathbf{Z}^{(0)})\right) \;\in\; \mathbb{R}^{4096 \times 96}$$

con $\mathrm{MLP}_0$: $\;96 \xrightarrow{\mathrm{Linear}} 384 \xrightarrow{\mathrm{GELU}} 384 \xrightarrow{\mathrm{Linear}} 96$

$$\mathbf{Z}^{(0)} \;\leftarrow\; \mathbf{Z}^{(0)} + \mathrm{DropPath}(\mathbf{M}_{0,b}) \;\in\; \mathbb{R}^{4096 \times 96}$$

**Output Stage 0** (dopo $b = 2$): $\quad\mathbf{Z}^{(0)} \in \mathbb{R}^{4096 \times 96}$

## PatchMerging 0 → 1

Ogni gruppo $2\times2$ di token adiacenti viene concatenato, poi normalizzato e proiettato:

$$\mathbb{R}^{4096 \times 96} \xrightarrow{\text{cat }2\times2} \mathbb{R}^{1024 \times 384} \xrightarrow{\mathrm{LN}} \mathbb{R}^{1024 \times 384} \xrightarrow{\mathrm{Linear}(384\to192)} \mathbf{Z}^{(1)} \in \mathbb{R}^{1024 \times 192}$$

$(64/2)^2 = 1024$ token · $4 \times 96 = 384 \to 192 = 2 \times 96$

## Stage 1 — $\ell = 1$

**Configurazione:** 2 blocchi · $H_1 = 8$ teste · $D_1 = 192$ · $S_1 = 32$

Risoluzione spaziale: $32 \times 32 = 1024$ token · Finestre $8\times 8$ → $N_w^1 = 1024 / 64 = 16$ finestre.

### Blocco $b \in \{1, 2\}$

#### Step 1 — LayerNorm + W-MSA

**Split 1 (lineare):**

$$\underbrace{\mathrm{LN}(\mathbf{Z}^{(1)}) \cdot W^{QKV}_1}_{\mathbb{R}^{1024 \times 576}} \xrightarrow{\text{split in 3}} \mathbf{Q}^{(1)},\, \mathbf{K}^{(1)},\, \mathbf{V}^{(1)} \in \mathbb{R}^{16 \times 64 \times 192}$$

con $W^{QKV}_1 \in \mathbb{R}^{192 \times 576}$ (dove $576 = 3 \times 192$).

**Split 2 (per testa):**

$$\mathbf{Q}^{(1)} \xrightarrow{\text{reshape}} \mathbb{R}^{16 \times 64 \times 8 \times 24} \xrightarrow{\text{permute}} \mathbb{R}^{16 \times 8 \times 64 \times 24}$$

Per ogni testa $h \in \{1, \ldots, 8\}$:

$$\mathbf{Q}_{1,b,h},\; \mathbf{K}_{1,b,h},\; \mathbf{V}_{1,b,h} \;\in\; \mathbb{R}^{N_w^1 \times M \times d_h} = \mathbb{R}^{16 \times 64 \times 24}$$

**Attention per testa $h$:**

$$\mathbf{H}_{1,b,h} = \mathrm{Softmax}\!\left(\frac{\mathbf{Q}_{1,b,h}\,\mathbf{K}_{1,b,h}^\top}{\sqrt{d_h}} + \mathbf{B}_{1,b,h}\right)\mathbf{V}_{1,b,h} \;\in\; \mathbb{R}^{16 \times 64 \times 24}$$

dove $\mathbf{B}_{1,b,h} \in \mathbb{R}^{64 \times 64}$.

**Proiezione e contributo per testa:**

$$W^O_1 \in \mathbb{R}^{192 \times 192}, \qquad W^O_{1,h} \in \mathbb{R}^{24 \times 192}$$

$$\widehat{\mathbf{H}}_{1,b,h} = \mathbf{H}_{1,b,h}\,W^O_{1,h} + \frac{\mathbf{b}^O_1}{H_1} \;\in\; \mathbb{R}^{16 \times 64 \times 192}$$

$$\mathbf{A}_{1,b} = \sum_{h=1}^{8} \widehat{\mathbf{H}}_{1,b,h} \;\in\; \mathbb{R}^{1024 \times 192}$$

$$\mathbf{Z}^{(1)} \;\leftarrow\; \mathbf{Z}^{(1)} + \mathrm{DropPath}(\mathbf{A}_{1,b}) \;\in\; \mathbb{R}^{1024 \times 192}$$

#### Step 2 — LayerNorm + MLP

$$\mathbf{M}_{1,b} = \mathrm{MLP}_1\!\left(\mathrm{LN}(\mathbf{Z}^{(1)})\right) \;\in\; \mathbb{R}^{1024 \times 192}$$

con $\mathrm{MLP}_1$: $\;192 \xrightarrow{\mathrm{Linear}} 768 \xrightarrow{\mathrm{GELU}} 768 \xrightarrow{\mathrm{Linear}} 192$

$$\mathbf{Z}^{(1)} \;\leftarrow\; \mathbf{Z}^{(1)} + \mathrm{DropPath}(\mathbf{M}_{1,b}) \;\in\; \mathbb{R}^{1024 \times 192}$$

**Output Stage 1** (dopo $b = 2$): $\quad\mathbf{Z}^{(1)} \in \mathbb{R}^{1024 \times 192}$

## PatchMerging 1 → 2

$$\mathbb{R}^{1024 \times 192} \xrightarrow{\text{cat }2\times2} \mathbb{R}^{256 \times 768} \xrightarrow{\mathrm{LN}} \mathbb{R}^{256 \times 768} \xrightarrow{\mathrm{Linear}(768\to384)} \mathbf{Z}^{(2)} \in \mathbb{R}^{256 \times 384}$$

$(32/2)^2 = 256$ token · $4 \times 192 = 768 \to 384 = 2 \times 192$

## Stage 2 — $\ell = 2$

**Configurazione:** 6 blocchi · $H_2 = 16$ teste · $D_2 = 384$ · $S_2 = 16$

Risoluzione spaziale: $16 \times 16 = 256$ token · Finestre $8\times8$ → $N_w^2 = 256 / 64 = 4$ finestre.

### Blocco $b \in \{1, \ldots, 6\}$

#### Step 1 — LayerNorm + W-MSA

**Split 1 (lineare):**

$$\underbrace{\mathrm{LN}(\mathbf{Z}^{(2)}) \cdot W^{QKV}_2}_{\mathbb{R}^{256 \times 1152}} \xrightarrow{\text{split in 3}} \mathbf{Q}^{(2)},\, \mathbf{K}^{(2)},\, \mathbf{V}^{(2)} \in \mathbb{R}^{4 \times 64 \times 384}$$

con $W^{QKV}_2 \in \mathbb{R}^{384 \times 1152}$ (dove $1152 = 3 \times 384$).

**Split 2 (per testa):**

$$\mathbf{Q}^{(2)} \xrightarrow{\text{reshape}} \mathbb{R}^{4 \times 64 \times 16 \times 24} \xrightarrow{\text{permute}} \mathbb{R}^{4 \times 16 \times 64 \times 24}$$

Per ogni testa $h \in \{1, \ldots, 16\}$:

$$\mathbf{Q}_{2,b,h},\; \mathbf{K}_{2,b,h},\; \mathbf{V}_{2,b,h} \;\in\; \mathbb{R}^{N_w^2 \times M \times d_h} = \mathbb{R}^{4 \times 64 \times 24}$$

**Attention per testa $h$:**

$$\mathbf{H}_{2,b,h} = \mathrm{Softmax}\!\left(\frac{\mathbf{Q}_{2,b,h}\,\mathbf{K}_{2,b,h}^\top}{\sqrt{d_h}} + \mathbf{B}_{2,b,h}\right)\mathbf{V}_{2,b,h} \;\in\; \mathbb{R}^{4 \times 64 \times 24}$$

dove $\mathbf{B}_{2,b,h} \in \mathbb{R}^{64 \times 64}$.

**Proiezione e contributo per testa:**

$$W^O_2 \in \mathbb{R}^{384 \times 384}, \qquad W^O_{2,h} \in \mathbb{R}^{24 \times 384}$$

$$\widehat{\mathbf{H}}_{2,b,h} = \mathbf{H}_{2,b,h}\,W^O_{2,h} + \frac{\mathbf{b}^O_2}{H_2} \;\in\; \mathbb{R}^{4 \times 64 \times 384}$$

$$\mathbf{A}_{2,b} = \sum_{h=1}^{16} \widehat{\mathbf{H}}_{2,b,h} \;\in\; \mathbb{R}^{256 \times 384}$$

$$\mathbf{Z}^{(2)} \;\leftarrow\; \mathbf{Z}^{(2)} + \mathrm{DropPath}(\mathbf{A}_{2,b}) \;\in\; \mathbb{R}^{256 \times 384}$$

#### Step 2 — LayerNorm + MLP

$$\mathbf{M}_{2,b} = \mathrm{MLP}_2\!\left(\mathrm{LN}(\mathbf{Z}^{(2)})\right) \;\in\; \mathbb{R}^{256 \times 384}$$

con $\mathrm{MLP}_2$: $\;384 \xrightarrow{\mathrm{Linear}} 1536 \xrightarrow{\mathrm{GELU}} 1536 \xrightarrow{\mathrm{Linear}} 384$

$$\mathbf{Z}^{(2)} \;\leftarrow\; \mathbf{Z}^{(2)} + \mathrm{DropPath}(\mathbf{M}_{2,b}) \;\in\; \mathbb{R}^{256 \times 384}$$

**Output Stage 2** (dopo $b = 6$): $\quad\mathbf{Z}^{(2)} \in \mathbb{R}^{256 \times 384}$

## PatchMerging 2 → 3

⚠️ **Questa transizione include PatchMerging** (dal codice: `downsample=PatchMerging if i_layer < num_layers - 1`, quindi anche per $\ell=2$):

$$\mathbb{R}^{256 \times 384} \xrightarrow{\text{cat }2\times2} \mathbb{R}^{64 \times 1536} \xrightarrow{\mathrm{LN}} \mathbb{R}^{64 \times 1536} \xrightarrow{\mathrm{Linear}(1536\to768)} \mathbf{Z}^{(3)} \in \mathbb{R}^{64 \times 768}$$

$(16/2)^2 = 64$ token · $4 \times 384 = 1536 \to 768 = 2 \times 384$

## Stage 3 — $\ell = 3$

**Configurazione:** 2 blocchi · $H_3 = 32$ teste · $D_3 = 768$ · $S_3 = 8$

Risoluzione spaziale: $8 \times 8 = 64$ token · Finestre $8\times8$ → $N_w^3 = 64 / 64 = \mathbf{1}$ finestra.

> **Nota:** poiché $\min(S_3) = 8 = w$, il SwinTransformerBlock forza `shift_size = 0`: non viene applicato lo shift ciclico (SW-MSA), tutti i blocchi usano W-MSA standard con un'unica finestra che copre l'intera griglia.

### Blocco $b \in \{1, 2\}$

#### Step 1 — LayerNorm + W-MSA

**Split 1 (lineare):**

$$\underbrace{\mathrm{LN}(\mathbf{Z}^{(3)}) \cdot W^{QKV}_3}_{\mathbb{R}^{64 \times 2304}} \xrightarrow{\text{split in 3}} \mathbf{Q}^{(3)},\, \mathbf{K}^{(3)},\, \mathbf{V}^{(3)} \in \mathbb{R}^{1 \times 64 \times 768}$$

con $W^{QKV}_3 \in \mathbb{R}^{768 \times 2304}$ (dove $2304 = 3 \times 768$).

**Split 2 (per testa):**

$$\mathbf{Q}^{(3)} \xrightarrow{\text{reshape}} \mathbb{R}^{1 \times 64 \times 32 \times 24} \xrightarrow{\text{permute}} \mathbb{R}^{1 \times 32 \times 64 \times 24}$$

Per ogni testa $h \in \{1, \ldots, 32\}$:

$$\mathbf{Q}_{3,b,h},\; \mathbf{K}_{3,b,h},\; \mathbf{V}_{3,b,h} \;\in\; \mathbb{R}^{N_w^3 \times M \times d_h} = \mathbb{R}^{1 \times 64 \times 24}$$

**Attention per testa $h$** (unica finestra, matrice $64\times64$):

$$\mathbf{H}_{3,b,h} = \mathrm{Softmax}\!\left(\frac{\mathbf{Q}_{3,b,h}\,\mathbf{K}_{3,b,h}^\top}{\sqrt{d_h}} + \mathbf{B}_{3,b,h}\right)\mathbf{V}_{3,b,h} \;\in\; \mathbb{R}^{1 \times 64 \times 24}$$

dove $\mathbf{B}_{3,b,h} \in \mathbb{R}^{64 \times 64}$.

**Proiezione e contributo per testa:**

$$W^O_3 \in \mathbb{R}^{768 \times 768}, \qquad W^O_{3,h} \in \mathbb{R}^{24 \times 768}$$

$$\widehat{\mathbf{H}}_{3,b,h} = \mathbf{H}_{3,b,h}\,W^O_{3,h} + \frac{\mathbf{b}^O_3}{H_3} \;\in\; \mathbb{R}^{1 \times 64 \times 768}$$

$$\mathbf{A}_{3,b} = \sum_{h=1}^{32} \widehat{\mathbf{H}}_{3,b,h} \;\in\; \mathbb{R}^{64 \times 768}$$

$$\mathbf{Z}^{(3)} \;\leftarrow\; \mathbf{Z}^{(3)} + \mathrm{DropPath}(\mathbf{A}_{3,b}) \;\in\; \mathbb{R}^{64 \times 768}$$

#### Step 2 — LayerNorm + MLP

$$\mathbf{M}_{3,b} = \mathrm{MLP}_3\!\left(\mathrm{LN}(\mathbf{Z}^{(3)})\right) \;\in\; \mathbb{R}^{64 \times 768}$$

con $\mathrm{MLP}_3$: $\;768 \xrightarrow{\mathrm{Linear}} 3072 \xrightarrow{\mathrm{GELU}} 3072 \xrightarrow{\mathrm{Linear}} 768$

$$\mathbf{Z}^{(3)} \;\leftarrow\; \mathbf{Z}^{(3)} + \mathrm{DropPath}(\mathbf{M}_{3,b}) \;\in\; \mathbb{R}^{64 \times 768}$$

**Output Stage 3** (dopo $b = 2$): $\quad\mathbf{Z}^{(3)} \in \mathbb{R}^{64 \times 768}$

## Post-processing HTS-AT

**LayerNorm finale:**

$$\mathbf{Z}^{(3)} \xrightarrow{\mathrm{LN}} \mathbf{Z}_{\mathrm{norm}} \in \mathbb{R}^{64 \times 768}$$

**Global Average Pooling** (su 64 token):

$$\mathbf{z} = \frac{1}{64}\sum_{i=1}^{64} \mathbf{Z}_{\mathrm{norm}}[i,:] \;\in\; \mathbb{R}^{768}$$

Questo è il `latent_output` di HTS-AT.

## CLAP Projection Head $P: \mathbb{R}^{768} \to \mathbb{R}^{1024}$

$$\mathbf{e}_1 = W_1\,\mathbf{z} \;\in\; \mathbb{R}^{1024}$$

$$\mathbf{e}_2 = \mathrm{Dropout}\!\left(W_2\,\mathrm{GELU}(\mathbf{e}_1)\right) \;\in\; \mathbb{R}^{1024}$$

$$\widehat{\mathbf{Y}} = \mathrm{LN}(\mathbf{e}_1 + \mathbf{e}_2) \;\in\; \mathbb{R}^{1024}$$

$\widehat{\mathbf{Y}}$ è l'embedding audio finale nello spazio condiviso audio-linguaggio di CLAP.

## Riepilogo dimensioni

| Fase | Tensore | Dimensione |
|------|---------|------------|
| Input transformer | $\mathbf{Z}^{(0)}$ | $4096 \times 96$ |
| Stage 0 · $W^{QKV}_0$ | — | $96 \times 288$ |
| Stage 0 · $\mathbf{Q}/\mathbf{K}/\mathbf{V}$ (pre-split testa) | — | $64 \times 64 \times 96$ |
| Stage 0 · $\mathbf{Q}_{h}/\mathbf{K}_{h}/\mathbf{V}_{h}$ (post-split testa) | — | $64 \times 64 \times 24$ |
| Stage 0 · $W^O_0$ | — | $96 \times 96$ |
| Stage 0 · $\mathbf{H}_{0,b,h}$ | — | $64 \times 64 \times 24$ |
| Stage 0 · $\widehat{\mathbf{H}}_{0,b,h}$ | — | $64 \times 64 \times 96$ |
| Stage 0 output | $\mathbf{Z}^{(0)}$ | $4096 \times 96$ |
| **PatchMerging 0→1** | $\mathbf{Z}^{(1)}$ | $1024 \times 192$ |
| Stage 1 · $W^{QKV}_1$ | — | $192 \times 576$ |
| Stage 1 · $\mathbf{Q}/\mathbf{K}/\mathbf{V}$ (pre-split testa) | — | $16 \times 64 \times 192$ |
| Stage 1 · $\mathbf{Q}_{h}/\mathbf{K}_{h}/\mathbf{V}_{h}$ (post-split testa) | — | $16 \times 64 \times 24$ |
| Stage 1 · $W^O_1$ | — | $192 \times 192$ |
| Stage 1 · $\mathbf{H}_{1,b,h}$ | — | $16 \times 64 \times 24$ |
| Stage 1 · $\widehat{\mathbf{H}}_{1,b,h}$ | — | $16 \times 64 \times 192$ |
| Stage 1 output | $\mathbf{Z}^{(1)}$ | $1024 \times 192$ |
| **PatchMerging 1→2** | $\mathbf{Z}^{(2)}$ | $256 \times 384$ |
| Stage 2 · $W^{QKV}_2$ | — | $384 \times 1152$ |
| Stage 2 · $\mathbf{Q}/\mathbf{K}/\mathbf{V}$ (pre-split testa) | — | $4 \times 64 \times 384$ |
| Stage 2 · $\mathbf{Q}_{h}/\mathbf{K}_{h}/\mathbf{V}_{h}$ (post-split testa) | — | $4 \times 64 \times 24$ |
| Stage 2 · $W^O_2$ | — | $384 \times 384$ |
| Stage 2 · $\mathbf{H}_{2,b,h}$ | — | $4 \times 64 \times 24$ |
| Stage 2 · $\widehat{\mathbf{H}}_{2,b,h}$ | — | $4 \times 64 \times 384$ |
| Stage 2 output | $\mathbf{Z}^{(2)}$ | $256 \times 384$ |
| **PatchMerging 2→3** | $\mathbf{Z}^{(3)}$ | $64 \times 768$ |
| Stage 3 · $W^{QKV}_3$ | — | $768 \times 2304$ |
| Stage 3 · $\mathbf{Q}/\mathbf{K}/\mathbf{V}$ (pre-split testa) | — | $1 \times 64 \times 768$ |
| Stage 3 · $\mathbf{Q}_{h}/\mathbf{K}_{h}/\mathbf{V}_{h}$ (post-split testa) | — | $1 \times 64 \times 24$ |
| Stage 3 · $W^O_3$ | — | $768 \times 768$ |
| Stage 3 · $\mathbf{H}_{3,b,h}$ | — | $1 \times 64 \times 24$ |
| Stage 3 · $\widehat{\mathbf{H}}_{3,b,h}$ | — | $1 \times 64 \times 768$ |
| Stage 3 output | $\mathbf{Z}^{(3)}$ | $64 \times 768$ |
| AvgPool | $\mathbf{z}$ | $768$ |
| CLAP $P$ | $\widehat{\mathbf{Y}}$ | $1024$ |