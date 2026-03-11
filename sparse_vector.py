#!/usr/bin/env python3
"""sparse_vector - Sparse vector/matrix operations for ML and scientific computing.

Usage: python sparse_vector.py [--demo]
"""
import sys, math, random

class SparseVector:
    def __init__(self, dim, data=None):
        self.dim = dim
        self.data = dict(data) if data else {}  # index -> value

    def __getitem__(self, i):
        return self.data.get(i, 0.0)

    def __setitem__(self, i, v):
        if abs(v) < 1e-15:
            self.data.pop(i, None)
        else:
            self.data[i] = v

    def __add__(self, other):
        result = SparseVector(self.dim, self.data)
        for i, v in other.data.items():
            result[i] = result[i] + v
        return result

    def __sub__(self, other):
        result = SparseVector(self.dim, self.data)
        for i, v in other.data.items():
            result[i] = result[i] - v
        return result

    def __mul__(self, scalar):
        return SparseVector(self.dim, {i: v*scalar for i, v in self.data.items()})

    def dot(self, other):
        s = 0.0
        small, big = (self, other) if len(self.data) < len(other.data) else (other, self)
        for i, v in small.data.items():
            if i in big.data:
                s += v * big.data[i]
        return s

    def norm(self):
        return math.sqrt(sum(v*v for v in self.data.values()))

    def cosine(self, other):
        n1, n2 = self.norm(), other.norm()
        if n1 == 0 or n2 == 0: return 0.0
        return self.dot(other) / (n1 * n2)

    @property
    def nnz(self):
        return len(self.data)

    def sparsity(self):
        return 1.0 - self.nnz / self.dim if self.dim > 0 else 1.0

    def to_dense(self):
        return [self[i] for i in range(self.dim)]

    @staticmethod
    def from_dense(vec):
        sv = SparseVector(len(vec))
        for i, v in enumerate(vec):
            if abs(v) > 1e-15:
                sv.data[i] = v
        return sv

    def __repr__(self):
        items = sorted(self.data.items())[:5]
        s = ", ".join(f"{i}:{v:.3f}" for i, v in items)
        if len(self.data) > 5: s += f", ... ({self.nnz} total)"
        return f"SparseVec({s})"

class SparseMatrix:
    """CSR-style sparse matrix."""
    def __init__(self, rows, cols):
        self.rows = rows; self.cols = cols
        self.data = {}  # (row, col) -> value

    def __setitem__(self, key, val):
        if abs(val) < 1e-15:
            self.data.pop(key, None)
        else:
            self.data[key] = val

    def __getitem__(self, key):
        return self.data.get(key, 0.0)

    def matvec(self, vec):
        """Matrix-vector multiply."""
        result = SparseVector(self.rows)
        for (i, j), v in self.data.items():
            result[i] = result[i] + v * vec[j]
        return result

    def transpose(self):
        t = SparseMatrix(self.cols, self.rows)
        for (i, j), v in self.data.items():
            t[j, i] = v
        return t

    @property
    def nnz(self):
        return len(self.data)

    def __repr__(self):
        return f"SparseMatrix({self.rows}x{self.cols}, nnz={self.nnz})"

def main():
    print("=== Sparse Vector/Matrix Operations ===\n")

    # Sparse vectors
    dim = 100000
    v1 = SparseVector(dim)
    v2 = SparseVector(dim)
    for _ in range(100):
        v1[random.randint(0, dim-1)] = random.gauss(0, 1)
        v2[random.randint(0, dim-1)] = random.gauss(0, 1)

    print(f"v1: {v1}, sparsity={v1.sparsity():.6f}")
    print(f"v2: {v2}, sparsity={v2.sparsity():.6f}")
    print(f"dot(v1,v2) = {v1.dot(v2):.6f}")
    print(f"cosine(v1,v2) = {v1.cosine(v2):.6f}")
    print(f"||v1|| = {v1.norm():.4f}, ||v2|| = {v2.norm():.4f}")

    v3 = v1 + v2
    print(f"v1+v2: nnz={v3.nnz}")
    v4 = v1 * 2.0
    print(f"2*v1: nnz={v4.nnz}, norm={v4.norm():.4f} (expected {v1.norm()*2:.4f})")

    # Dense roundtrip
    small = SparseVector.from_dense([1, 0, 0, 2, 0, 3])
    assert small.to_dense() == [1, 0, 0, 2, 0, 3]
    print(f"\nDense roundtrip: ✓")

    # Sparse matrix
    print(f"\nSparse matrix-vector multiply:")
    m = SparseMatrix(1000, 1000)
    for _ in range(5000):
        m[random.randint(0,999), random.randint(0,999)] = random.gauss(0, 1)
    x = SparseVector(1000)
    for _ in range(50):
        x[random.randint(0, 999)] = random.gauss(0, 1)
    print(f"  M: {m}")
    print(f"  x: nnz={x.nnz}")
    import time
    t0 = time.monotonic()
    y = m.matvec(x)
    dt = time.monotonic() - t0
    print(f"  y = M·x: nnz={y.nnz}, norm={y.norm():.4f}, time={dt*1000:.1f}ms")

    mt = m.transpose()
    print(f"  M^T: {mt}")

if __name__ == "__main__":
    main()
