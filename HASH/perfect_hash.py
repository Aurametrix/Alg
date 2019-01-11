# =======================================================================
# ================= Python code for perfect hash function ===============
# =======================================================================

G = [0, 0, 4, 1, 0, 3, 8, 1, 6]

S1 = [5, 0, 0, 6, 1, 0, 4, 7]
S2 = [7, 3, 6, 7, 8, 5, 7, 6]

def hash_f(key, T):
    return sum(T[i % 8] * ord(c) for i, c in enumerate(str(key))) % 9

def perfect_hash(key):
    return (G[hash_f(key, S1)] + G[hash_f(key, S2)]) % 9

# ============================ Sanity check =============================

K = ["Elephant", "Horse", "Camel", "Python", "Dog", "Cat"]
H = [0, 1, 2, 3, 4, 5]

assert len(K) == len(H) == 6

for k, h in zip(K, H):
    assert perfect_hash(k) == h
