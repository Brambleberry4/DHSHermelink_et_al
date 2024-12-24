import random
import numpy as np
from hint_solver import PyBP, PyGreedy

NVAR = 300
NFAC = 500
ETA = 2
ETA_2 = 3
N_GR = 10
N_BP = 100


def sample_bino(eta):
    r = sum([random.randint(0, 1) for _ in range(eta)]) - sum([random.randint(0, 1) for _ in range(eta)])
    return r


def inner(v, w):
    return sum(vi*wi for vi, wi in zip(v, w))


def to_likeliest(res):
    return from_compl(np.argmax(res), len(res))


def from_compl(v, sz_msg):
    if v > sz_msg/2:
        return v - sz_msg
    else:
        return v


def main():
    print("Sampling secret..")
    s = [sample_bino(ETA) for _ in range(NVAR)]
    print("Sampling equality coefficients..")
    eqs = [[sample_bino(ETA_2) for _ in range(NVAR)] for _ in range(NFAC)]
    print("Computing rhs..")
    rhs = [inner(ineq, s) for ineq in eqs]
    print("Setting distributions to actual values..")
    rhs_eq = [[(rhs_i, 1.0)] for rhs_i in rhs]
    print("Creating BP and Greedy..")
    bp = PyBP(eqs, rhs_eq)
    greedy = PyGreedy(eqs, rhs_eq)
    greedy.set_nthreads(4)
    bp.set_nthreads(4)
    k = NVAR
    print("Solving (GR)..")
    for i in range(N_GR):
        greedy.solve(k)
        guess = greedy.get_guess()
        dist = sum((si-gi)**2 for si, gi in zip(s, guess))
        if dist == 0:
            print("Found key.")
            break
        k //= 2
        if k < 10:
            k = NVAR
        print(f"{i}/{N_GR} (dist={dist:.1f})", end='\r')

    print()
    print("Solving (BP)..")
    k = NFAC
    for i in range(N_BP):
        bp.propagate()
        dists = bp.get_results()
        guess = []
        for dist in dists:
            guess.append(to_likeliest(dist))
        dist = sum((si-gi)**2 for si, gi in zip(s, guess))
        if dist == 0:
            print("Found key.")
            break
        k //= 2
        if k < NFAC//5:
            k = NFAC
        print(f"{i}/{N_BP} (dist={dist:.1f}, k={k})", end='\r')
    print()


if __name__ == "__main__":
    main()
