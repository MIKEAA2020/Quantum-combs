#!/usr/bin/env python3
"""Supplementary numerical corroboration for the article

    "Quantum Combs, Higher-Order Processes, and the
     Normalization-Defect (Intercept) Principle"

This script independently corroborates the numerical content of the
statements listed below.  Every proof in the article is analytic and
self-contained; nothing here enters any proof.  Each CHECK cites the
statement it corroborates.

  CHECK 1  Affine hom-set dimensions of the one-slot category.
           Corroborates: Proposition [One-slot hom-set dimension]
           (Section 4) and Appendix "Detailed verification of the
           normalization dimension", including the two trivial-object
           counts used by Theorem [No left adjoint for environment
           decoration], and the constants of Example [Qubit source and
           target].
  CHECK 2  Coefficient (intercept) rigidity under environment
           decoration.  Corroborates: Theorem [No affine representing
           object] and Theorem [No right adjoint for environment
           decoration] (coefficient identities, symbolic).
  CHECK 3  The integer e^2-e+1 lies strictly between consecutive
           squares for e=d_E^2, d_E>1.  Corroborates: Theorem [No left
           adjoint for environment decoration] and the channel toy
           model of Section 5.
  CHECK 4  The parallel tensor product.  Corroborates: Lemma [Parallel
           tensor is a monoidal product] (Section 6): positivity and
           normalization preservation on randomly sampled interior
           points, and the interchange law on generic Hermitian
           operators (the algebraic core of the bifunctoriality and
           closedness analysis).
  CHECK 5  The equation b^e-1 = e(b-1) has no solutions with b,e>1.
           Corroborates: Proposition [Classical environment decoration
           has neither adjoint] (Section 5, FinStoch toy model).
  CHECK 6  The instrument pointwise obstruction.  Corroborates:
           Proposition [Pointwise obstruction at fixed outcome number]
           (Appendix on instrument convex sets), and classifies the
           square escapes of the dimension count that occur strictly
           outside the proposition's hypothesis.

Requirements: Python >= 3.10, NumPy >= 1.24, SymPy >= 1.12.
Determinism: all pseudo-random draws use the fixed seeds below.
Runtime: a few minutes on a desktop machine (the exact-rank SVD checks
dominate).
Usage:    python3 verification_checks.py
Exit status is 0 if and only if every check passes.
"""

import math
import random
import sys

import numpy as np
import sympy as sp

SEED = 20260801
random.seed(SEED)
np.random.seed(SEED)

FAILED = []


def report(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


# ----------------------------------------------------------------------
# Linear-algebraic utilities (Hermitian basis, partial trace, system
# permutation), used by CHECK 1 and CHECK 4.
# ----------------------------------------------------------------------

def herm_gen(d):
    """Generator over the standard real basis of Hermitian d x d matrices."""
    for i in range(d):
        M = np.zeros((d, d), complex)
        M[i, i] = 1.0
        yield M
    for i in range(d):
        for j in range(i + 1, d):
            M = np.zeros((d, d), complex)
            M[i, j] = 1.0
            M[j, i] = 1.0
            yield M
            M = np.zeros((d, d), complex)
            M[i, j] = 1j
            M[j, i] = -1j
            yield M


def ptrace(M, dims, axes):
    """Partial trace over the tensor factors listed in axes."""
    n = len(dims)
    T = M.reshape(dims + dims)
    for ax in sorted(axes, reverse=True):
        T = np.trace(T, axis1=ax, axis2=ax + T.ndim // 2)
    rem = [i for i in range(n) if i not in axes]
    D = int(np.prod([dims[i] for i in rem])) if rem else 1
    return T.reshape(D, D)


def psys(M, dims, perm):
    """Reorder tensor factors: output factor order = [dims[p] for p in perm]."""
    n = len(dims)
    T = M.reshape(dims + dims)
    T = np.transpose(T, list(perm) + [n + p for p in perm])
    D = int(np.prod([dims[p] for p in perm]))
    return T.reshape(D, D)


def sdet_constraint_matrix(dA, dB, dC, dD):
    """Homogeneous constraint matrix of the two-tooth normalization equations.

    Variables: R Hermitian on (C,A,B,D) and S Hermitian on (C,A), with
        Tr_D R = I_B (x) S   (factors ordered C,A,B),
        Tr_A S = 0.
    The affine hom-set dimension is the number of real parameters minus
    the rank of this matrix (the offset point is the maximally mixed
    comb, which always exists).
    """
    dR, dS = dC * dA * dB * dD, dC * dA
    IB = np.eye(dB)
    ncols = dR * dR + dS * dS
    nrows = 2 * (dC * dA * dB) ** 2 + 2 * dC * dC
    C = np.zeros((nrows, ncols))
    i = 0
    for M in herm_gen(dR):
        E1 = ptrace(M, [dC, dA, dB, dD], [3])
        C[:, i] = np.concatenate(
            [E1.ravel().real, E1.ravel().imag, np.zeros(2 * dC * dC)])
        i += 1
    for M in herm_gen(dS):
        E1 = psys(np.kron(IB, M), [dB, dC, dA], [1, 2, 0])
        E2 = ptrace(M, [dC, dA], [1])
        C[:, i] = np.concatenate([-E1.ravel().real, -E1.ravel().imag,
                                  E2.ravel().real, E2.ravel().imag])
        i += 1
    return C, ncols


def sdet_affdim(dA, dB, dC, dD):
    C, tot = sdet_constraint_matrix(dA, dB, dC, dD)
    return tot - np.linalg.matrix_rank(C, tol=1e-8)


def formula(dA, dB, dC, dD):
    """The article's closed form, P_Y(u,v) = alpha_Y u v + beta_Y u - beta_Y
    with u=d_A^2, v=d_B^2, alpha_Y=d_C^2(d_D^2-1), beta_Y=d_C^2."""
    return dC**2 * (dA**2 - 1) + (dC * dA * dB) ** 2 * (dD**2 - 1)


print("=" * 72)
print("Supplementary verification run")
print(f"numpy {np.__version__}, sympy {sp.__version__}, seed {SEED}")
print("=" * 72)

# ----------------------------------------------------------------------
print("\n--- CHECK 1: affine hom-set dimensions (Section 4) ---")
# 1a. Exact constraint ranks vs the closed form: exhaustive on an
#     exhaustive small grid, plus a random dimension-capped sample.
ok = tot = 0
for dA in range(1, 5):
    for dB in range(1, 5):
        for dC in range(1, 5):
            for dD in range(1, 5):
                if dA * dB * dC * dD > 32:
                    continue
                tot += 1
                if sdet_affdim(dA, dB, dC, dD) == formula(dA, dB, dC, dD):
                    ok += 1
report("1a(i). closed form vs exact constraint rank, exhaustive grid "
       "(dims <= 4, dimension product <= 32)", ok == tot, f"{ok}/{tot}")
ok = tot = 0
while tot < 300:
    dA, dB, dC, dD = (random.randint(1, 8) for _ in range(4))
    if dA * dB * dC * dD > 48:
        continue
    tot += 1
    if sdet_affdim(dA, dB, dC, dD) == formula(dA, dB, dC, dD):
        ok += 1
report("1a(ii). same, 300 random tuples with dims <= 8, dimension "
       "product <= 48", ok == tot, f"{ok}/{tot}")

# 1b. Hom-sets into the trivial object have dimension d_A^2 - 1.
ok = tot = 0
for dA in range(1, 9):
    for dB in range(1, 9):
        tot += 1
        if sdet_affdim(dA, dB, 1, 1) == dA**2 - 1:
            ok += 1
report("1b. hom-sets into the trivial object: d_A^2 - 1", ok == tot,
       f"{ok}/{tot}")

# 1c. Hom-sets from the trivial object to (E,E) have dimension e(e-1).
ok = tot = 0
for dE in range(1, 13):
    tot += 1
    e = dE * dE
    if sdet_affdim(1, 1, dE, dE) == e * (e - 1):
        ok += 1
report("1c. hom-sets from the trivial object to (E,E): e(e-1)",
       ok == tot, f"{ok}/{tot}")

# 1d. Constants of Example [Qubit source and target]: u=v=e=4,
# alpha_Y=12, beta_Y=4; undecorated dimension 204, decorated 3132; the
# candidate representing object is excluded twice.
u = v = e = 4
alpha_Y, beta_Y = 12, 4
undec = alpha_Y * u * v + beta_Y * u - beta_Y
dec = e**2 * alpha_Y * u * v + e * beta_Y * u - beta_Y
c1 = undec == 204 and dec == 3132
# constant term forces beta_Z = 4, coefficient of u forces beta_Z = 16:
c2 = (dec + beta_Y) == 3136 and (e * beta_Y) == 16 and beta_Y == 4
# single-point evaluation forces alpha_Z = 195 while one-slot structure
# demands alpha_Z = 4(d^2 - 1), which has no integer solution:
c3 = (dec - e * alpha_Y * u - (e - 1) * beta_Y) == 3132 - 192 - 12 == 2928
alpha_Z = (dec - 3 * beta_Y) // 16
c4 = alpha_Z == 195
d2_num = alpha_Z + 4          # d^2 = alpha_Z/4 + 1 = 199/4, non-integer
c5 = (alpha_Z % 4 != 0) or (alpha_Z // 4 + 1) != math.isqrt(alpha_Z // 4 + 1) ** 2
c5 = (alpha_Z + 4) % 4 != 0
report("1d. Example [Qubit source and target]: dimensions 204 and 3132, "
       "beta_Z contradiction 4 vs 16, single-point alpha_Z=195 with no "
       "integer d^2", c1 and c2 and c4 and c5)

# ----------------------------------------------------------------------
print("\n--- CHECK 2: coefficient (intercept) rigidity under decoration ---")
uu, vv, ee, aa, bb = sp.symbols("u v e alpha beta")
P = aa * uu * vv + bb * uu - bb                       # P_Y(u,v)
Pd = sp.expand(P.subs({uu: ee * uu, vv: ee * vv}))    # decorated: (eu,ev)
coef_uv = Pd.coeff(uu, 1).coeff(vv, 1)                # -> e^2 alpha
coef_u0 = Pd.coeff(uu, 1).subs(vv, 0)                 # -> e beta
const = Pd.subs({uu: 0, vv: 0})                       # -> -beta
# A representing object has polynomial alpha_Z u v + beta_Z u - beta_Z.
# Coefficient matching forces, simultaneously,
#   alpha_Z = e^2 alpha,  beta_Z = e beta,  beta_Z = beta,
# and the last two are incompatible when e>1 and beta=d_C^2>0.
mismatch = sp.expand(coef_u0 + const)                 # (e - 1) beta
ok = (sp.simplify(coef_uv - ee**2 * aa) == 0
      and sp.simplify(coef_u0 - ee * bb) == 0
      and sp.simplify(const + bb) == 0
      and sp.simplify(mismatch - (ee - 1) * bb) == 0)
print(f"    P_Y(u,v)          = {P}")
print(f"    decorated         = {Pd}")
print(f"    coefficient match: alpha_Z = e^2 alpha, beta_Z = e beta vs "
      f"beta_Z = beta; residual = {mismatch} =/= 0 for e>1, beta>=1")
report("2. decoration fixes the constant term and multiplies the "
       "coefficient of u by e; incompatibility residual (e-1) beta", ok)

# ----------------------------------------------------------------------
print("\n--- CHECK 3: e^2-e+1 strictly between consecutive squares ---")
bad = sum(
    1
    for dE in range(2, 20001)
    if not ((dE * dE - 1) ** 2 < dE * dE * dE * dE - dE * dE + 1
            < dE * dE * dE * dE)
)
report("3a. exhaustive d_E = 2..20000 (e = d_E^2)", bad == 0,
       f"violations={bad}")
bad = 0
for _ in range(2000000):
    dE = int(math.exp(random.uniform(math.log(2), math.log(10**9))))
    e = dE * dE
    g = e * e - e + 1
    if not ((e - 1) ** 2 < g < e**2):
        bad += 1
report("3b. 2,000,000 random d_E, log-uniform to 1e9", bad == 0,
       f"violations={bad}")
# Analytic content: (e-1)^2 = e^2-2e+1 < e^2-e+1 iff e>0, and
# e^2-e+1 < e^2 iff e>1; the checks above corroborate the integer claim.

# ----------------------------------------------------------------------
print("\n--- CHECK 4: parallel tensor product (Section 6) ---")
rng = np.random.default_rng(SEED)


def random_interior_comb(dA, dB, dC, dD, max_tries=4000):
    """Generic strictly positive point of SDet((A,B),(C,D)) with prefix S:
    Tr_D R = I_B (x) S, Tr_A S = I_C, sampled around the maximally mixed
    comb inside the constraint kernel."""
    Cmt, _ = sdet_constraint_matrix(dA, dB, dC, dD)
    svals, Vt = np.linalg.svd(Cmt)[1:]
    K2 = Vt[int(np.sum(svals > 1e-8)):].T
    R1b = np.kron(np.eye(dC), np.eye(dA) / dA)
    Rb = psys(np.kron(np.kron(np.eye(dB), R1b), np.eye(dD) / dD),
              [dB, dC, dA, dD], [1, 2, 0, 3])
    Lb = np.linalg.cholesky(Rb)
    Lbinv = np.linalg.inv(Lb)
    basis = list(herm_gen(dC * dA * dB * dD))
    for _ in range(max_tries):
        w = K2 @ rng.standard_normal(K2.shape[1])
        Eop = sum(w[i] * basis[i] for i in range(len(basis)))
        lam = np.linalg.eigvalsh(Lbinv @ Eop @ Lbinv.conj().T)
        lo = -1.0 / lam.max() if lam.max() > 0 else -1e12
        hi = 1.0 / abs(lam.min()) if lam.min() < 0 else 1e12
        t = 0.5 * rng.uniform(lo, hi)
        R = Rb + t * Eop
        if np.linalg.eigvalsh(R).min() <= 1e-12:
            continue
        S = (1 / dB) * ptrace(R, [dC, dA, dB, dD], [2, 3])
        if np.linalg.eigvalsh(S).min() <= 1e-12:
            continue
        return R, S
    raise RuntimeError("interior sampler exhausted")


def comb_normalized(R, dC, dA, dB, dD, tol=1e-7):
    """Verify Tr_D R = I_B (x) S and Tr_A S = I_C on the tooth axes
    (C,A,B,D), with S recovered as (1/dB) Tr_{B,D} R."""
    S = (1 / dB) * ptrace(R, [dC, dA, dB, dD], [2, 3])
    lhs = ptrace(R, [dC, dA, dB, dD], [3])
    rhs = psys(np.kron(np.eye(dB), S), [dB, dC, dA], [1, 2, 0])
    return (np.allclose(lhs, rhs, atol=tol)
            and np.allclose(ptrace(S, [dC, dA], [1]), np.eye(dC), atol=tol))


def tensor_morphisms(RF, dA, dB, dC, dD, RG, dP, dQ, dM, dN):
    """F (x) G on composite teeth ((C,M),(A,P),(B,Q),(D,N)): the article's
    fixed tensor convention after the canonical permutation."""
    f = RF.reshape(dC, dA, dB, dD, dC, dA, dB, dD)
    g = RG.reshape(dM, dP, dQ, dN, dM, dP, dQ, dN)
    K = np.einsum("cabdCABD,mpqnMPQN->cmapbqdnCMAPBQDN", f, g)
    return K.reshape((dC * dM * dA * dP * dB * dQ * dD * dN,) * 2)


# 4a. Positivity and normalization preservation on interior points.
pairs = [((2, 2, 2, 2), (2, 1, 1, 2)),
         ((2, 2, 2, 1), (1, 2, 2, 2)),
         ((2, 1, 2, 2), (2, 2, 1, 2)),
         ((1, 2, 2, 2), (2, 2, 2, 2)),
         ((2, 2, 1, 2), (2, 1, 2, 2))]
ok = 0
for (dA, dB, dC, dD), (dP, dQ, dM, dN) in pairs:
    RF, SF = random_interior_comb(dA, dB, dC, dD)
    RG, SG = random_interior_comb(dP, dQ, dM, dN)
    K = tensor_morphisms(RF, dA, dB, dC, dD, RG, dP, dQ, dM, dN)
    dC2, dA2, dB2, dD2 = dC * dM, dA * dP, dB * dQ, dD * dN
    positive = np.linalg.eigvalsh(K).min() > -1e-8
    normalized = comb_normalized(K, dC2, dA2, dB2, dD2)
    # prefix factors: S_K = S_F (x) S_G on teeth ((C,M),(A,P))
    SK = (1 / dB2) * ptrace(K, [dC2, dA2, dB2, dD2], [2, 3])
    fs = SF.reshape(dC, dA, dC, dA)
    gs = SG.reshape(dM, dP, dM, dP)
    SK_expected = np.einsum("caCA,mpMP->cmapCMAP", fs, gs).reshape(
        (dC2 * dA2,) * 2)
    factorizes = np.allclose(SK, SK_expected, atol=1e-7)
    if positive and normalized and factorizes:
        ok += 1
report("4a. F (x) G is positive and satisfies both normalization equations, "
       "with prefix S_F (x) S_G, on random interior points",
       ok == len(pairs), f"{ok}/{len(pairs)}")


# 4b. Interchange law (algebraic core of bifunctoriality), generic
# Hermitian operators, manuscript component rule for composition.
def comp(RG, RF, dP, dC, dD, dQ, dA, dB):
    """Composition in components:
    (RG o RF)[p,a,b,q,P,A,B,Q] =
        sum_{c,C,d,D} RG[p,c,d,q,P,C,D,Q] RF[c,a,b,d,C,A,B,D]."""
    T = np.einsum(
        "pcdqPCDQ,cabdCABD->pabqPABQ",
        RG.reshape(dP, dC, dD, dQ, dP, dC, dD, dQ),
        RF.reshape(dC, dA, dB, dD, dC, dA, dB, dD))
    return T.reshape(dP * dA * dB * dQ, dP * dA * dB * dQ)


def rand_herm(n):
    Z = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    return (Z + Z.conj().T) / 2


ok = done = 0
while done < 40:
    dims = tuple(random.choice([1, 2, 3]) for _ in range(12))
    dA, dB, dC, dD, dR, dS, dP, dQ, dM, dN, dT2, dU = dims
    tot = dR * dT2 * dA * dP * dB * dQ * dS * dU
    if tot > 1200 or tot == 1:
        continue
    done += 1
    F1 = rand_herm(dC * dA * dB * dD)   # (A,B) -> (C,D)
    F2 = rand_herm(dR * dC * dD * dS)   # (C,D) -> (R,S)
    G1 = rand_herm(dM * dP * dQ * dN)   # (P,Q) -> (M,N)
    G2 = rand_herm(dT2 * dM * dN * dU)  # (M,N) -> (T,U)
    # left side: (F2 o F1) (x) (G2 o G1)
    L1 = comp(F2, F1, dR, dC, dD, dS, dA, dB)
    L2 = comp(G2, G1, dT2, dM, dN, dU, dP, dQ)
    l1 = L1.reshape(dR, dA, dB, dS, dR, dA, dB, dS)
    l2 = L2.reshape(dT2, dP, dQ, dU, dT2, dP, dQ, dU)
    LHS = np.einsum("rabsRABS,tpquTPQU->rtapbqsuRTAPBQSU", l1, l2)
    LHS = LHS.reshape((tot,) * 2)
    # right side: (F2 (x) G2) o (F1 (x) G1)
    K1 = tensor_morphisms(F1, dA, dB, dC, dD, G1, dP, dQ, dM, dN)
    K2 = tensor_morphisms(F2, dC, dD, dR, dS, G2, dM, dN, dT2, dU)
    RHS = comp(K2, K1, dR * dT2, dC * dM, dD * dN, dS * dU,
               dA * dP, dB * dQ)
    if np.allclose(LHS, RHS, atol=1e-10):
        ok += 1
report("4b. interchange (F2 o F1) (x) (G2 o G1) = (F2 (x) G2) o (F1 (x) G1)",
       ok == 40, f"{ok}/40")

# ----------------------------------------------------------------------
print("\n--- CHECK 5: FinStoch toy model (Section 5) ---")
# b^e - 1 = (b-1)(1 + b + ... + b^{e-1}) > e(b-1) for b,e > 1, because
# the sum has e terms, the first equal to 1 and all others at least 2.
def violates(b, e):
    s = 0
    for k in range(e):
        s += b**k
        if s > e:               # the strict inequality is already forced
            return 0
    return 0 if b**e - 1 > e * (b - 1) else 1


bad = sum(violates(b, e) for b in range(2, 2000) for e in range(2, 60))
report("5a. b^e - 1 > e(b-1): exhaustive b = 2..1999, e = 2..59",
       bad == 0, f"violations={bad}")
bad = sum(violates(random.randint(2, 10**6), random.randint(2, 10**6))
          for _ in range(1000000))
report("5b. 1,000,000 random (b,e) up to 1e6", bad == 0,
       f"violations={bad}")

# ----------------------------------------------------------------------
print("\n--- CHECK 6: instrument pointwise obstruction (Appendix) ---")
# g_B = e v - (e-1)/n  with e=d_E^2, v=d_B^2, c=(e-1)/n.
# The proposition's hypothesis is an existence statement over B: it
# takes any B with 2 d_E d_B - 1 > c.  We verify that every sampled
# case satisfying that hypothesis is strictly between consecutive
# squares, and we classify every sampled case outside the hypothesis
# in which g_B nevertheless is a square.
v1 = trials = v_regime = violations = 0
boundary, deeper = [], []          # square escapes at/beyond c = 2w-1
for _ in range(500000):
    dE = random.randint(2, 200)
    dB = random.randint(2, 200)
    n = random.randint(1, 12)
    e, v = dE * dE, dB * dB
    g_num = e * v * n - (e - 1)          # g_B = g_num / n
    if g_num % n != 0:
        v1 += 1
        continue
    trials += 1
    g = g_num // n
    w = dE * dB
    c = (e - 1) // n
    if c < 2 * w - 1:
        # the proposition's regime: g_B must lie strictly between
        # consecutive squares, so the candidate dimension is excluded
        v_regime += 1
        if not ((w - 1) ** 2 < g < w**2):
            violations += 1
    else:
        # outside the hypothesis 2 d_E d_B - 1 > c; record squares and
        # verify their structural identities at the same time
        r = math.isqrt(g)
        if r * r == g:
            if c == 2 * w - 1:
                boundary.append((dE, dB, n, g == (w - 1) ** 2))
            else:
                k = w - r
                deeper.append((dE, dB, n, k,
                               k >= 2 and c == k * (2 * w - k)))
fam = [x for x in boundary
       if x[2] == 1 and x[0] % 2 == 0 and x[1] == x[0] // 2]
ok = (violations == 0
      and all(x[3] for x in boundary)
      and all(x[4] for x in deeper))
report("6a. 500,000 sampled (d_E, d_B, n): non-integrality or the "
       "between-squares argument disposes of every case in the "
       "proposition's regime 2 d_E d_B - 1 > c (0 violations); the only "
       "square escapes occur outside that hypothesis",
       ok,
       f"non-integer kills={v1}; integer survivors={trials} with "
       f"in-regime cases={v_regime} and regime violations={violations}; "
       f"outside-hypothesis squares: {len(boundary)} at the boundary "
       f"c=2w-1 (of which {len(fam)} from the n=1, d_B=d_E/2 family), "
       f"plus {len(deeper)} deeper squares with c=k(2w-k), k>=2")

# 6b. Anatomy of the outside-hypothesis escapes (symbolic identities).
w_s, k_s = sp.symbols("w k", positive=True, integer=True)
# at the boundary c = 2w-1:  g_B = w^2 - (2w-1) = (w-1)^2 exactly
b_id = sp.simplify(w_s**2 - (2 * w_s - 1) - (w_s - 1) ** 2) == 0
# beyond it: c = k(2w-k) gives g_B = (w-k)^2, and every such c is
# outside the hypothesis because c - (2w-1) = (k-1)(2w-k-1) >= 0 for
# 1 <= k <= w-1
d_id = sp.simplify(w_s**2 - k_s * (2 * w_s - k_s) - (w_s - k_s) ** 2) == 0
m_id = sp.simplify(k_s * (2 * w_s - k_s) - (2 * w_s - 1)
                   - (k_s - 1) * (2 * w_s - k_s - 1)) == 0
# the n = 1 boundary coincides with the family d_B = d_E/2 (d_E even):
# 2 d_E d_B - 1 = e - 1 with e = d_E^2 forces d_B = d_E/2, and there
# g_B = (d_E^2/2 - 1)^2
kk = sp.symbols("kk", positive=True, integer=True)
dE_s, e_s = 2 * kk, (2 * kk) ** 2
g_sym = sp.expand(e_s * kk**2 - (e_s - 1))           # d_B = d_E/2 = kk
f_id = sp.simplify(g_sym - (2 * kk**2 - 1) ** 2) == 0
fam_ok = all(2 * dE * (dE // 2) - 1 == dE * dE - 1
             and dE * dE * (dE // 2) ** 2 - (dE * dE - 1)
             == (dE * dE // 2 - 1) ** 2
             for dE in range(2, 42, 2))
report("6b. escape anatomy (symbolic): boundary g_B=(w-1)^2; deeper "
       "squares g_B=(w-k)^2 with c=k(2w-k)>2w-1 for 2<=k<w; the n=1 "
       "boundary is the family d_B=d_E/2 (d_E even), g_B=(d_E^2/2-1)^2",
       all([b_id, d_id, m_id, f_id, fam_ok]))

# ----------------------------------------------------------------------
print("\n" + "=" * 72)
if FAILED:
    print(f"RESULT: {len(FAILED)} CHECK(S) FAILED: {FAILED}")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")
print("=" * 72)
