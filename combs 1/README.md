# Supplementary Material: Numerical Corroboration

**Article:** *Quantum Combs, Higher-Order Processes, and the
Normalization-Defect (Intercept) Principle*
**Author:** Amin Abaee (amin_abaee@ut.ac.ir)
**Repository:** <https://github.com/MIKEAA2020/Quantum-combs/tree/main/combs%201>

## Contents of this supplement

| File | Description |
|---|---|
| `verification_checks.py` | Annotated Python script performing all checks described below. |
| `verification_output.txt` | Output log of one complete run of the script. |
| `README.md` | The present file. |

## Status of the computations

Every proof in the article is analytic and self-contained. The
computations in this supplement provide independent numerical
corroboration of the numerical content of the indicated statements;
they do not enter any proof, and no statement in the article depends on
them.

## What is checked, and where each check maps to the article

| Check | Content | Statement corroborated |
|---|---|---|
| 1a | Closed form of the one-slot affine hom-set dimension against exact constraint ranks: exhaustive grid (dims ≤ 4, product ≤ 32) plus 300 random tuples (dims ≤ 8, product ≤ 48) | Proposition [One-slot hom-set dimension] (Section 4); appendix "Detailed verification of the normalization dimension" |
| 1b | Hom-sets into the trivial object have dimension d_A² − 1, exact ranks | Trivial-object count used by Theorem [No left adjoint for environment decoration] |
| 1c | Hom-sets from the trivial object to (E,E) have dimension e(e − 1), exact ranks | Same as above |
| 1d | Dimensions 204 (undecorated) and 3132 (decorated); the representing-object contradiction obtained twice (β_Z = 4 vs 16; α_Z = 195 with no integer d²) | Example [Qubit source and target] (Section 4) |
| 2 | Symbolic polynomial identities: environment decoration fixes the constant term and multiplies the coefficient of u by e, leaving the incompatibility residual (e − 1)β | Theorem [No affine representing object]; Theorem [No right adjoint for environment decoration] |
| 3 | e² − e + 1 lies strictly between consecutive squares: exhaustive d_E = 2…20000, plus 2,000,000 random d_E log-uniform to 10⁹ | Theorem [No left adjoint for environment decoration]; channel toy model (Section 5) |
| 4a | F ⊗ G is positive and satisfies both normalization equations, with prefix S_F ⊗ S_G, on randomly sampled strict interior points | Lemma [Parallel tensor is a monoidal product] (Section 6) |
| 4b | Interchange law (F₂ ∘ F₁) ⊗ (G₂ ∘ G₁) = (F₂ ⊗ G₂) ∘ (F₁ ⊗ G₁) on 40 random generic-Hermitian instances, using the article's component composition rule | Lemma [Parallel tensor is a monoidal product]; Corollary [The parallel tensor product is not closed] |
| 5 | b^e − 1 = e(b − 1) has no solutions with b, e > 1: exhaustive and random ranges | Proposition [Classical environment decoration has neither adjoint] (FinStoch toy model) |
| 6a | 500,000 sampled (d_E, d_B, n): every integer-surviving case in the proposition's regime 2·d_E·d_B − 1 > (e − 1)/n lies strictly between consecutive squares (zero exceptions); the only square escapes of the dimension count occur strictly outside that hypothesis and are recorded | Proposition [Pointwise obstruction at fixed outcome number] (instrument appendix) |
| 6b | Symbolic anatomy of the outside-hypothesis escapes: at the boundary c = 2w − 1 one has g_B = (w − 1)² exactly, and beyond it g_B = (w − k)² with c = k(2w − k) > 2w − 1 for 2 ≤ k < w; the n = 1 boundary is the accidental family d_B = d_E/2 (d_E even) with g_B = (d_E²/2 − 1)². None of these cases satisfy the proposition's hypothesis, which takes d_B large | Same as above |

## Reproduction

Requirements: Python ≥ 3.10, NumPy ≥ 1.24, SymPy ≥ 1.12. No other
dependencies and no compilation step.

```
python3 verification_checks.py
```

The script prints one PASS/FAIL line per check and ends with a summary
line. The exit status is 0 if and only if every check passes. All
pseudo-random draws use the fixed seed `20260801`, so the run is fully
deterministic; the enclosed `verification_output.txt` is the log of one
such run. Runtime is approximately one minute on a desktop machine.
