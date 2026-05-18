# src/classic_ml/algorithms/linear_regression.py

class LinearRegression:
    def __init__(self, lam=0.0):
        self.lam = lam
        self.w = []

    @staticmethod
    def transpose(M):
        return list(map(list, zip(*M)))

    @staticmethod
    def matmul(A, B):
        return [[sum(a * b for a, b in zip(row, col)) for col in zip(*B)] for row in A]

    def add_regularization(self, XTX):
        n = len(XTX)
        reg = [[self.lam if i == j and i != 0 else 0 for j in range(n)] for i in range(n)]
        return [[XTX[i][j] + reg[i][j] for j in range(n)] for i in range(n)]

    @staticmethod
    def invert_matrix(M):
        n = len(M)
        aug = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(M)]
        for i in range(n):
            if abs(aug[i][i]) < 1e-12:
                for r in range(i + 1, n):
                    if abs(aug[r][i]) > abs(aug[i][i]):
                        aug[i], aug[r] = aug[r], aug[i]
                        break
            pivot = aug[i][i]
            if abs(pivot) < 1e-12:
                raise ValueError("Singular matrix")
            aug[i] = [v / pivot for v in aug[i]]
            for r in range(n):
                if r != i:
                    factor = aug[r][i]
                    aug[r] = [vr - factor * vi for vr, vi in zip(aug[r], aug[i])]
        return [row[n:] for row in aug]

    def fit(self, X, y):
        if not X or not y:
            return
        
        # Ensure X has bias term if not already there
        # In this refactor, we expect X to be the raw features, we'll add 1.0 here
        X_with_bias = [[1.0] + row for row in X]
        
        X_T = self.transpose(X_with_bias)
        XTX = self.matmul(X_T, X_with_bias)
        
        if self.lam > 0.0:
            XTX = self.add_regularization(XTX)
            
        XTy = self.matmul(X_T, [[v] for v in y])
        
        try:
            XTX_inv = self.invert_matrix(XTX)
            w_mat = self.matmul(XTX_inv, XTy)
            self.w = [wi[0] for wi in w_mat]
        except ValueError:
            self.w = []
            
        return self.w

    def predict(self, x):
        if not self.w:
            return None
        # Add bias to x
        x_vec = [1.0] + list(x)
        return sum(wi * xi for wi, xi in zip(self.w, x_vec))
