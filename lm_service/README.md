## Definitions

Token space : $\alpha \in \mathcal{A}$

Trees of tokens comprise candidate (synonymous to mention) space: $\kappa (\alpha) \in \mathcal{K}$.

Candidates are mapped to entity space $K \to E$.

Ordered triples of candidates comprise triple space: $\tau \in T(\mathcal{K})$, in such a way that there are projections: 
- source : $\sigma :T \to \mathcal{K}$
- relation : $\rho :T \to \mathcal{K}$
- target : $\pi :T \to \mathcal{K}$

Define the extension of triple space recursively by adding itself to candidate space: $\mathcal{K}^{(1)} = \mathcal{K}^{(0)} \bigcup T^{(0)}$, $T^{(i)} = T(\mathcal{K}^{(i)})$. Then define $\mathcal{K}^* = \lim \mathcal{K}^{n}$ and $T^* = \lim T(\mathcal{K}^*)$. 

We call $\mathcal{K}$ fundamental candidates and $T$ - metacandidates.

## Encoding a publication

The structured representation of text $\pi$ then looks like 
$\pi \mapsto \tau \mapsto \kappa \mapsto e$.


For clarity candidates and entities are indexes for each text $\pi$ by $i^\pi_\kappa \in I^\pi_\kappa$ and by $i^\pi_e \in I^\pi_e$.

The output of REL therefore yields:
1. $\tau \mapsto i_\kappa$
2. $i_\kappa \mapsto \kappa$
3. $i_\kappa \mapsto i_e$
4. $i_e \mapsto e$
5. $\kappa \mapsto \alpha$


