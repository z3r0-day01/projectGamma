# Project Gamma (Project Orange River)

## Executive Overview (EDIT BY A HUMAN: What type of Edgy Nonsense did CoPilot do?)

Hallucinations here are crazy.

Z3r0_DaYz Software Group & Z3r0_DaYz Intelligence Research Group

---

PROJECT ORANGE_RIVER  

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

Math Removed temporarily.

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

Stolen from Deepmind!

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

Edit by Human: Where'd it go?

Vector factorization of matrices for disk storage and temporary use in RAM.
