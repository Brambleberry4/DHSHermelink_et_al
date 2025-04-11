import random
import numpy as np
from hint_solver import PyBP, PyGreedy
from scipy.stats import norm


NVAR = 300
NFAC = 500
ETA = 2
ETA_2 = 3
N_GR = 10
N_BP = 100
STD_DEV = 1

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


def twoComp(num):
    if num < 0:
        return (1 << 28) + num
    else:
        return num % (1 << 28)


def measure_HW_Noise(inner_product):
    hw = twoComp(inner_product).bit_count()
    #Gaussian error:
    gau_err = np.random.normal(0,STD_DEV)
    new_mu = hw + gau_err
    #Create a probability mass function from this mu
    # 0 - 28 are all possible HWs for this type
    x_vals = np.arange(29)
    pdf =  norm.pdf(x_vals,new_mu, STD_DEV)
    rounded_pdf = np.round(pdf, decimals=5)
    return rounded_pdf

def measure_LSB(inner_product):
    #Get the LSB of inner_product
    lsb = inner_product & 1
    #For every val: if lsb(val) = lsb of inner_product => 1.0 else 0.0
    return lsb


def attackHW(rhs,val):
    return [[(x,measure_HW_Noise(rhi)[twoComp(x).bit_count()]) for x in range(-val,val)] for rhi in rhs]

def attackLSB(rhs,val):
    #Create rhs
    return 0


def main():
    print("Sampling secret..")
    s = [sample_bino(ETA) for _ in range(NVAR)]
    print("Sampling equality coefficients..")
    eqs = [[sample_bino(ETA_2) for _ in range(NVAR)] for _ in range(NFAC)]
    print("Computing rhs..")
    rhs = [inner(ineq, s) for ineq in eqs]
    print("Setting distributions to actual values..")
    
    # Estimate the maximum range of possible values v
    li = []
    for x in eqs:
        sume = 0
        for t in x:
            sume += abs(t)
        li.append(sume)
    val = max(li)
    
    for x in range(5):
        print(x, measure_LSB(x))
    # Compute the distributions
    list_of_dist_hints = attackHW(rhs, val) 
    #rhs_eq2=[[(x,measure_HW_Noise(rhi).get(twoComp(x).bit_count(),0.0)) for x in range(-val,val)] for rhi in rhs]
    print(rhs[0])
    print(measure_HW_Noise(rhs[0]))
    
    # Remove entries with probability 0.0
    rhs_eq = []
    for q in list_of_dist_hints:
        g =[(s,p) for (s,p) in q if p != 0]
        rhs_eq.append(g)
    
    print("Creating BP and Greedy..")
    bp = PyBP(eqs, rhs_eq)
    greedy = PyGreedy(eqs, rhs_eq)
    greedy.set_nthreads(4)
    bp.set_nthreads(1)
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
        # One full iteration
        bp.propagate()
        # Compute the current resulting prob. distributions
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
        print(f"{i}/{N_BP} (dist={dist:.1f}, k={k})", end='\n')
    print()


if __name__ == "__main__":
    main()
