# HTS-AT Transformer Pipeline

## Input al Transformer

$$\mathbf{Z}^{(0)} = \mathbf{X}_0 \in \mathbb{R}^{4096 \times 96}$$

$4096 = 64 \times 64$ token, ciascuno di dimensione $D_0 = 96$.

## Stage 0 — $\ell = 0$: 2 blocchi, $H_0 = 4$ teste, $D_0 = 96$

Risoluzione spaziale: $64 \times 64$ token.  
Finestre di $8 \times 8 = 64$ token → $N_w^0 = \frac{64 \times 64}{64} = 64$ finestre.

### Per ogni blocco $b \in \{1, 2\}$:

**Step 1 — LayerNorm + W-MSA (o SW-MSA):**

Per ogni testa $h \in \{1,2,3,4\}$:

$$\mathbf{Q}_{0,b,h},\, \mathbf{K}_{0,b,h},\, \mathbf{V}_{0,b,h} = \mathrm{split}\!\left(\mathrm{LN}(\mathbf{Z}^{(0)})\,W^{QKV}_0\right) \in \mathbb{R}^{64 \times 64 \times 24}$$

con $W^{QKV}_0 \in \mathbb{R}^{96 \times 288}$.

$$\mathbf{H}_{0,b,h} = \mathrm{Softmax}\!\left(\frac{\mathbf{Q}_{0,b,h}\mathbf{K}_{0,b,h}^\top}{\sqrt{d_h}} + \mathbf{B}_{0,b,h}\right)\mathbf{V}_{0,b,h} \in \mathbb{R}^{64 \times 64 \times 24}$$

$$W^O_0 \in \mathbb{R}^{96 \times 96}, \quad W^O_{0,h} \in \mathbb{R}^{24 \times 96} \text{ (righe } [(h-1)\cdot 24,\; h\cdot 24) \text{ di } W^O_0\text{)}$$

$$\widehat{\mathbf{H}}_{0,b,h} = \mathbf{H}_{0,b,h}\,W^O_{0,h} + \frac{\mathbf{b}^O_0}{H_0} \in \mathbb{R}^{64 \times 64 \times 96}$$

$$\mathbf{A}_{0,b} = \sum_{h=1}^{4} \widehat{\mathbf{H}}_{0,b,h} \in \mathbb{R}^{4096 \times 96}$$

$$\mathbf{Z}^{(0)} \leftarrow \mathbf{Z}^{(0)} + \mathrm{DropPath}(\mathbf{A}_{0,b}) \in \mathbb{R}^{4096 \times 96}$$

**Step 2 — LayerNorm + MLP:**

$$\mathbf{M}_{0,b} = \mathrm{MLP}_0(\mathrm{LN}(\mathbf{Z}^{(0)})) \in \mathbb{R}^{4096 \times 96}$$

con MLP: $96 \xrightarrow{\mathrm{Linear}} 384 \xrightarrow{\mathrm{GELU}} 384 \xrightarrow{\mathrm{Linear}} 96$.

$$\mathbf{Z}^{(0)} \leftarrow \mathbf{Z}^{(0)} + \mathrm{DropPath}(\mathbf{M}_{0,b}) \in \mathbb{R}^{4096 \times 96}$$

**Output Stage 0** (dopo $b=2$):

$$\mathbf{Z}^{(0)} \in \mathbb{R}^{4096 \times 96}$$

## PatchMerging 0→1

$$\mathbf{Z}^{(0)} \in \mathbb{R}^{4096 \times 96} \xrightarrow{\mathrm{cat}\;2\times2} \mathbb{R}^{1024 \times 384} \xrightarrow{\mathrm{LN}} \mathbb{R}^{1024 \times 384} \xrightarrow{\mathrm{Linear}(384\to192)} \mathbf{Z}^{(1)} \in \mathbb{R}^{1024 \times 192}$$

## Stage 1 — $\ell = 1$: 2 blocchi, $H_1 = 8$ teste, $D_1 = 192$

Risoluzione spaziale: $32 \times 32$ token.  
Finestre di $8 \times 8 = 64$ token → $N_w^1 = \frac{32 \times 32}{64} = 16$ finestre.

### Per ogni blocco $b \in \{1, 2\}$:

**Step 1 — LayerNorm + W-MSA:**

Per ogni testa $h \in \{1,\ldots,8\}$:

$$\mathbf{Q}_{1,b,h},\, \mathbf{K}_{1,b,h},\, \mathbf{V}_{1,b,h} = \mathrm{split}\!\left(\mathrm{LN}(\mathbf{Z}^{(1)})\,W^{QKV}_1\right) \in \mathbb{R}^{16 \times 64 \times 24}$$

con $W^{QKV}_1 \in \mathbb{R}^{192 \times 576}$.

$$\mathbf{H}_{1,b,h} = \mathrm{Softmax}\!\left(\frac{\mathbf{Q}_{1,b,h}\mathbf{K}_{1,b,h}^\top}{\sqrt{d_h}} + \mathbf{B}_{1,b,h}\right)\mathbf{V}_{1,b,h} \in \mathbb{R}^{16 \times 64 \times 24}$$

$$W^O_1 \in \mathbb{R}^{192 \times 192}, \quad W^O_{1,h} \in \mathbb{R}^{24 \times 192} \text{ (righe } [(h-1)\cdot 24,\; h\cdot 24) \text{ di } W^O_1\text{)}$$

$$\widehat{\mathbf{H}}_{1,b,h} = \mathbf{H}_{1,b,h}\,W^O_{1,h} + \frac{\mathbf{b}^O_1}{H_1} \in \mathbb{R}^{16 \times 64 \times 192}$$

$$\mathbf{A}_{1,b} = \sum_{h=1}^{8} \widehat{\mathbf{H}}_{1,b,h} \in \mathbb{R}^{1024 \times 192}$$

$$\mathbf{Z}^{(1)} \leftarrow \mathbf{Z}^{(1)} + \mathrm{DropPath}(\mathbf{A}_{1,b}) \in \mathbb{R}^{1024 \times 192}$$

**Step 2 — LayerNorm + MLP:**

$$\mathbf{M}_{1,b} = \mathrm{MLP}_1(\mathrm{LN}(\mathbf{Z}^{(1)})) \in \mathbb{R}^{1024 \times 192}$$

con MLP: $192 \xrightarrow{\mathrm{Linear}} 768 \xrightarrow{\mathrm{GELU}} 768 \xrightarrow{\mathrm{Linear}} 192$.

$$\mathbf{Z}^{(1)} \leftarrow \mathbf{Z}^{(1)} + \mathrm{DropPath}(\mathbf{M}_{1,b}) \in \mathbb{R}^{1024 \times 192}$$

**Output Stage 1** (dopo $b=2$):

$$\mathbf{Z}^{(1)} \in \mathbb{R}^{1024 \times 192}$$

## PatchMerging 1→2

$$\mathbf{Z}^{(1)} \in \mathbb{R}^{1024 \times 192} \xrightarrow{\mathrm{cat}\;2\times2} \mathbb{R}^{256 \times 768} \xrightarrow{\mathrm{LN}} \mathbb{R}^{256 \times 768} \xrightarrow{\mathrm{Linear}(768\to384)} \mathbf{Z}^{(2)} \in \mathbb{R}^{256 \times 384}$$

## Stage 2 — $\ell = 2$: 6 blocchi, $H_2 = 16$ teste, $D_2 = 384$

Risoluzione spaziale: $16 \times 16$ token.  
Finestre di $8 \times 8 = 64$ token → $N_w^2 = \frac{16 \times 16}{64} = 4$ finestre.

### Per ogni blocco $b \in \{1, \ldots, 6\}$:

**Step 1 — LayerNorm + W-MSA:**

Per ogni testa $h \in \{1,\ldots,16\}$:

$$\mathbf{Q}_{2,b,h},\, \mathbf{K}_{2,b,h},\, \mathbf{V}_{2,b,h} = \mathrm{split}\!\left(\mathrm{LN}(\mathbf{Z}^{(2)})\,W^{QKV}_2\right) \in \mathbb{R}^{4 \times 64 \times 24}$$

con $W^{QKV}_2 \in \mathbb{R}^{384 \times 1152}$.

$$\mathbf{H}_{2,b,h} = \mathrm{Softmax}\!\left(\frac{\mathbf{Q}_{2,b,h}\mathbf{K}_{2,b,h}^\top}{\sqrt{d_h}} + \mathbf{B}_{2,b,h}\right)\mathbf{V}_{2,b,h} \in \mathbb{R}^{4 \times 64 \times 24}$$

$$W^O_2 \in \mathbb{R}^{384 \times 384}, \quad W^O_{2,h} \in \mathbb{R}^{24 \times 384} \text{ (righe } [(h-1)\cdot 24,\; h\cdot 24) \text{ di } W^O_2\text{)}$$

$$\widehat{\mathbf{H}}_{2,b,h} = \mathbf{H}_{2,b,h}\,W^O_{2,h} + \frac{\mathbf{b}^O_2}{H_2} \in \mathbb{R}^{4 \times 64 \times 384}$$

$$\mathbf{A}_{2,b} = \sum_{h=1}^{16} \widehat{\mathbf{H}}_{2,b,h} \in \mathbb{R}^{256 \times 384}$$

$$\mathbf{Z}^{(2)} \leftarrow \mathbf{Z}^{(2)} + \mathrm{DropPath}(\mathbf{A}_{2,b}) \in \mathbb{R}^{256 \times 384}$$

**Step 2 — LayerNorm + MLP:**

$$\mathbf{M}_{2,b} = \mathrm{MLP}_2(\mathrm{LN}(\mathbf{Z}^{(2)})) \in \mathbb{R}^{256 \times 384}$$

con MLP: $384 \xrightarrow{\mathrm{Linear}} 1536 \xrightarrow{\mathrm{GELU}} 1536 \xrightarrow{\mathrm{Linear}} 384$.

$$\mathbf{Z}^{(2)} \leftarrow \mathbf{Z}^{(2)} + \mathrm{DropPath}(\mathbf{M}_{2,b}) \in \mathbb{R}^{256 \times 384}$$

**Output Stage 2** (dopo $b=6$):

$$\mathbf{Z}^{(2)} \in \mathbb{R}^{256 \times 384}$$

## Stage 2→3: nessun PatchMerging

$$\mathbf{Z}^{(3)} = \mathbf{Z}^{(2)} \in \mathbb{R}^{256 \times 384}$$

## Stage 3 — $\ell = 3$: 2 blocchi, $H_3 = 32$ teste, $D_3 = 768$

Risoluzione spaziale: $16 \times 16$ token.  
Finestre di $8 \times 8 = 64$ token → $N_w^3 = \frac{16 \times 16}{64} = 4$ finestre.

### Per ogni blocco $b \in \{1, 2\}$:

**Step 1 — LayerNorm + W-MSA:**

Per ogni testa $h \in \{1,\ldots,32\}$:

$$\mathbf{Q}_{3,b,h},\, \mathbf{K}_{3,b,h},\, \mathbf{V}_{3,b,h} = \mathrm{split}\!\left(\mathrm{LN}(\mathbf{Z}^{(3)})\,W^{QKV}_3\right) \in \mathbb{R}^{4 \times 64 \times 24}$$

con $W^{QKV}_3 \in \mathbb{R}^{768 \times 2304}$.

$$\mathbf{H}_{3,b,h} = \mathrm{Softmax}\!\left(\frac{\mathbf{Q}_{3,b,h}\mathbf{K}_{3,b,h}^\top}{\sqrt{d_h}} + \mathbf{B}_{3,b,h}\right)\mathbf{V}_{3,b,h} \in \mathbb{R}^{4 \times 64 \times 24}$$

$$W^O_3 \in \mathbb{R}^{768 \times 768}, \quad W^O_{3,h} \in \mathbb{R}^{24 \times 768} \text{ (righe } [(h-1)\cdot 24,\; h\cdot 24) \text{ di } W^O_3\text{)}$$

$$\widehat{\mathbf{H}}_{3,b,h} = \mathbf{H}_{3,b,h}\,W^O_{3,h} + \frac{\mathbf{b}^O_3}{H_3} \in \mathbb{R}^{4 \times 64 \times 768}$$

$$\mathbf{A}_{3,b} = \sum_{h=1}^{32} \widehat{\mathbf{H}}_{3,b,h} \in \mathbb{R}^{256 \times 768}$$

$$\mathbf{Z}^{(3)} \leftarrow \mathbf{Z}^{(3)} + \mathrm{DropPath}(\mathbf{A}_{3,b}) \in \mathbb{R}^{256 \times 768}$$

**Step 2 — LayerNorm + MLP:**

$$\mathbf{M}_{3,b} = \mathrm{MLP}_3(\mathrm{LN}(\mathbf{Z}^{(3)})) \in \mathbb{R}^{256 \times 768}$$

con MLP: $768 \xrightarrow{\mathrm{Linear}} 3072 \xrightarrow{\mathrm{GELU}} 3072 \xrightarrow{\mathrm{Linear}} 768$.

$$\mathbf{Z}^{(3)} \leftarrow \mathbf{Z}^{(3)} + \mathrm{DropPath}(\mathbf{M}_{3,b}) \in \mathbb{R}^{256 \times 768}$$

**Output Stage 3** (dopo $b=2$):

$$\mathbf{Z}^{(3)} \in \mathbb{R}^{256 \times 768}$$

## Post-processing HTS-AT

**LayerNorm finale:**

$$\mathbf{Z}^{(3)} \xrightarrow{\mathrm{LN}} \mathbf{Z}_{\mathrm{norm}} \in \mathbb{R}^{256 \times 768}$$

**Global Average Pooling:**

$$\mathbf{z} = \frac{1}{256}\sum_{i=1}^{256} \mathbf{Z}_{\mathrm{norm}}[i,:] \in \mathbb{R}^{768}$$

Questo è il `latent_output` di HTS-AT.

## CLAP Projection Head $P: \mathbb{R}^{768} \to \mathbb{R}^{1024}$

$$\mathbf{e}_1 = W_1\,\mathbf{z} \in \mathbb{R}^{1024}$$

$$\mathbf{e}_2 = \mathrm{Dropout}\!\left(W_2\,\mathrm{GELU}(\mathbf{e}_1)\right) \in \mathbb{R}^{1024}$$

$$\widehat{\mathbf{Y}} = \mathrm{LN}(\mathbf{e}_1 + \mathbf{e}_2) \in \mathbb{R}^{1024}$$

$\widehat{\mathbf{Y}}$ è l'embedding audio finale nello spazio condiviso audio-linguaggio di CLAP.

## Riepilogo dimensioni

| Fase | Tensore | Dimensione |
|------|---------|------------|
| Input transformer | $\mathbf{Z}^{(0)}$ | $4096 \times 96$ |
| Stage 0, $W^{QKV}_0$ | | $96 \times 288$ |
| Stage 0, $W^O_0$ | | $96 \times 96$ |
| Stage 0, $\mathbf{H}_{0,b,h}$ | | $64 \times 64 \times 24$ |
| Stage 0, $\widehat{\mathbf{H}}_{0,b,h}$ | | $64 \times 64 \times 96$ |
| Stage 0 output | $\mathbf{Z}^{(0)}$ | $4096 \times 96$ |
| PatchMerging 0→1 | $\mathbf{Z}^{(1)}$ | $1024 \times 192$ |
| Stage 1, $W^{QKV}_1$ | | $192 \times 576$ |
| Stage 1, $W^O_1$ | | $192 \times 192$ |
| Stage 1, $\mathbf{H}_{1,b,h}$ | | $16 \times 64 \times 24$ |
| Stage 1, $\widehat{\mathbf{H}}_{1,b,h}$ | | $16 \times 64 \times 192$ |
| Stage 1 output | $\mathbf{Z}^{(1)}$ | $1024 \times 192$ |
| PatchMerging 1→2 | $\mathbf{Z}^{(2)}$ | $256 \times 384$ |
| Stage 2, $W^{QKV}_2$ | | $384 \times 1152$ |
| Stage 2, $W^O_2$ | | $384 \times 384$ |
| Stage 2, $\mathbf{H}_{2,b,h}$ | | $4 \times 64 \times 24$ |
| Stage 2, $\widehat{\mathbf{H}}_{2,b,h}$ | | $4 \times 64 \times 384$ |
| Stage 2 output | $\mathbf{Z}^{(2)}$ | $256 \times 384$ |
| Stage 3 input | $\mathbf{Z}^{(3)}$ | $256 \times 384$ |
| Stage 3, $W^{QKV}_3$ | | $768 \times 2304$ |
| Stage 3, $W^O_3$ | | $768 \times 768$ |
| Stage 3, $\mathbf{H}_{3,b,h}$ | | $4 \times 64 \times 24$ |
| Stage 3, $\widehat{\mathbf{H}}_{3,b,h}$ | | $4 \times 64 \times 768$ |
| Stage 3 output | $\mathbf{Z}^{(3)}$ | $256 \times 768$ |
| AvgPool | $\mathbf{z}$ | $768$ |
| CLAP $P$ | $\widehat{\mathbf{Y}}$ | $1024$ |