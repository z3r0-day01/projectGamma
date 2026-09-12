# Project Gamma (Project Orange River)

## Executive Overview (Restricted)

_This document is classified and restricted to the Z3r0_DaYz Research Group Executives._

Z3r0_DaYz Software Group & Z3r0_DaYz Intelligence Research Group  
TANGERINE Conglomerate & GRAPEFRUIT Conglomerate Research Joint-Project

---

PROJECT ORANGE_RIVER  
(Notice: this may be referred to by executives as Project Gamma or its sister project, Project Walnut.)

## Ideas

- Causal + Casual Inference (Did these happen at the same time? + Did X force Y or is it a coincidence?)
- LLM Memory Matrix for Linear Attention or State Space Attention
- Pigeonhole Principle (How does this apply to Project Orange River?)
- Concurrency through Asynchronous Parallelism (Efficiency?)
- Decaying Linear Attention
- Energy-based Linear Attention
- Efficient Attention
- Softmax Alternatives
- Open Knowledge Format (for training data, RAG, etc.)
- Vector Databases
- Mixture of Recursions (3 unique layers, 1 round min each, 6 round max each. Layer-Time = 3-18 rounds.)
- Mixture of Experts
- Contrastive Predictive Coding
- TurboQuant
- VectorDBs

---

Our research started with Linear Attention Models, and z-Normalization, to make what we call **Double-Filter Attention**.

Currently, we are on track to create 4 types of models.  
However, we’ll only cover the flagship and add to it.

## Flagship

**Decaying Energetic Remembering Selective Stateful Space Double Filter V2**  
**DERS3DFv2**

```text
R(x)={x≤0 ? 0.01x+0.1 : x+0.1)
z(x)=R(x)²/Sum(r(x_n)²)
N(x)=√(Sum(x²_n))
De(a,b)=√(Sum((b-a)²))
Dc(a,b): Cosine Distance

M_t = (g_t [hadamard] Q_t) + (1 - g_t) [hadamard] (M_t-1 * SiLU(Q_t*M_t-1))
L_t = [X_t, g_t-1]
J_t = (J_t * S(W*L_t+B)) + (S(W*L_t+B)*Tanh(W*L_t+B))
g_t = S(W*L_t+B)*Tanh(W*J_t+B)

Note: each W/B is separate per operation.

Lookup(x,n)=Sin(b*S(a*X_{n})*X_{n}+n) -> Scalar
PosEnc(x)=Sin(b*S(a*X_{n})*X+n) -> Vector
B = Linear_B(X_t)*Lookup(x,t) -> Scalar
C = Linear_C(X_t)*PosEnc(x) -> Vector
D = z(X)*Softplus(X_t) -> Vector
Q=tanh(Wq*x_t+Wp*x_t-1)
K=WK*Sigmoid(Wk*x_t+Wp*x_t-1)*x_t
V=Tanh(Wa*Sigmoid(Wb^T*x_t)+x_t*Sigmoid(Wp^T*x_t-1))
F=e^(-e^(d_t))
d_t=ddlerp(x_t,x_t-1)*Wd
R=DerSigmoid(Q)*R_t-1+Tanh(x_t)*DerTanh(Q)
A = e^(∆*D)
B_hat = ((e^(∆*D)-I)*∆*B)/(∆*A) ≈ ∆ * B
∆ = Softplus(Linear_∆(X)*z(X)) -> Vector
h_t = A*h_{t-1} + (B_hat)*z(F*K^T)*V -> Matrix
y_t = C*z(Q*R)*h_t + S(y_{t-1})*M_t -> Matrix
Uhat_t=½x_t*x_t^T
Ohat_t=½Tr(y_t-1*y_t-1^T)
∆epsilon_t aka ∆E_t=Tanh(Uhat_T - Ohat_t)
E_int=((x_t*y_t^T)/(N(x_t)*N(y_t)+Epsilon))
Z_t = RMSNorm(S(x_t)*lerp(x_{t},y_{t},∆)+De([x_t,y_t],[x_{t-1},y_{t-1}])*Z_{t-1}) -> Scalar
P_t=exp(((E_int*Z_t)/(Softplus(∆E_t)+epsilon))
O = RMSNorm((SiLU(Dc([x_t,y_t],[x_{t-1},y_{t-1}])*Z_t)*P_t)*y_t)*tanh(Z_t) -> Matrix
```

---

## Other development ideas

### Reactivating Intelligent Sliding Window Attention

Base: Grouped Query Latent Attention  
By default: W=3, NetLayerDepth=2, causal.

If a token repeats, this reactivates past instances.  
To improve this, reactivate only key-words and use a Semantic Routing Gate to activate related tokens.  
Tokens are placed into semantic blocks (e.g., cat/rat together, mat instances together).  
Upon reactivation, two neighbor tokens on either side are also reactivated from cache.

Reactivation decays based on:
- semantic similarity
- context
- cosine distance
- token distance

Blocks are also separated into context groups.

### Dual Mixture of Recursions

3 unique layers.  
Each layer loops recursively 1x minimum, 4x maximum.

Example:

```text
AA-BBB-C-DD
```

An outer loop runs up to 3 times, limited to 32 total loops including inner loops.

### Mixture of Experts

8 routed + 1 shared.

Routed:
- (LERCM_Recog + ArtithLang) + Abstract
- Logic
- Emotion
- Rationality
- Creativity
- (vague/general) Morality
- Recog/Memory-Recall
- Arithmetic
- Human Language

Shared:
- Abstract Shared Expert

Routing limits:
- minimum 1 expert per round (excluding shared)
- maximum 3 experts per round (excluding shared)

### Engram Hash Storage

- Cold: base training data
- Global: general learned information and RAG
- Learned: current learned information
- Active: affective information and data

### DeepEmbeddings

Vector factorization of matrices for disk storage and temporary use in RAM.
