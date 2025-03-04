import random
import numpy as np
from hint_solver import PyBP, PyGreedy
from scipy.special import erf
from scipy.stats import norm


NVAR = 300
NFAC = 2000
ETA = 2
ETA_2 = 3
N_GR = 10
N_BP = 100
STD_DEV = 1
VALRAN = ETA * ETA_2 * NVAR


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

  
#in: (list of) inner products(rhs) out: (list of) distributions (rhs_eq)
#returns the distribution 
def measure_HW_NoiseFree(inner_product):
    hw = twoComp(inner_product).bit_count()
    return [(hw, 1.0)]


def measure_HW_Noise(inner_product):
    hw = twoComp(inner_product).bit_count()
    #Gaussian error:
    gau_err = np.random.normal(0,STD_DEV)
    
    new_mu = hw + gau_err
    #Create a probability mass function from this mu
    x_vals = np.arange(29)
    pdf =  norm.pdf(x_vals,new_mu, STD_DEV)
    #print(pdf)
    rounded_pdf = np.round(pdf, decimals=5)
    indices = np.nonzero(rounded_pdf)[0]
    res = [(i, rounded_pdf[i]) for i in indices] 
    conv_res = [(a.tolist(),b.tolist()) for a,b in res]   
    #print(conv_res)
    return rounded_pdf

def measure_LSB(inner_products):
    return [i & 1 for i in inner_products]
    


def measure_MSB(inner_product):
    if inner_product >= 0:
        msb = 0
    else:
        msb = 1
    return msb


#in: distribution out: distribution hints
#Takes a list of distributions and creates the final list
def attack(distribution):
    #building hints..
    return 0

def attackLSB(distr):
    return [[(x, 1.0) for x in range(-VALRAN,VALRAN) if (x & 1) == distri] for distri in distr]



def main():
    print("Sampling secret..")
    s = [sample_bino(ETA) for _ in range(NVAR)]
    print("Sampling equality coefficients..")
    eqs = [[sample_bino(ETA_2) for _ in range(NVAR)] for _ in range(NFAC)]
    print("Computing rhs..")
    rhs = [inner(ineq, s) for ineq in eqs]
    print("Setting distributions to actual values..")
    #HW leakage model:
    rhs_eq=[[(x,measure_HW_Noise(rhi)[twoComp(x).bit_count()]) for x in range(-VALRAN,VALRAN)] for rhi in rhs]
    
    #print(len(rhs_eq))
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
