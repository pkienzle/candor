import numpy as np

def demo():
    I = [1000, 900, 800, 700, 600, 500, 400, 300, 200]
    R = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.38, 0.36]
    C = [
        [0.490, 0.250, 0.022, 0.020, 0.018, 0.016, 0.014, 0.012, 0.010],
        [0.050, 0.490, 0.250, 0.022, 0.020, 0.018, 0.016, 0.014, 0.012],
        [0.052, 0.050, 0.490, 0.250, 0.022, 0.020, 0.018, 0.016, 0.014],
        [0.054, 0.052, 0.050, 0.490, 0.250, 0.022, 0.020, 0.018, 0.016],
        [0.056, 0.054, 0.052, 0.050, 0.490, 0.250, 0.022, 0.020, 0.018],
        [0.058, 0.056, 0.054, 0.052, 0.050, 0.490, 0.250, 0.022, 0.020],
        [0.060, 0.058, 0.056, 0.054, 0.052, 0.050, 0.490, 0.250, 0.022],
        [0.062, 0.060, 0.058, 0.056, 0.054, 0.052, 0.050, 0.490, 0.250],
        [0.064, 0.062, 0.060, 0.058, 0.056, 0.054, 0.052, 0.050, 0.490],
    ]
    I, R, C = [np.asarray(v, 'd') for v in (I, R, C)]
    C = C.T

    SD = C @ (R*I)
    # Solve C * RI = SD using SVD:
    #     C^-1 SD = (U S V^T)^-1 SD = (V^T S^-1) (U^T SD)
    # Given s = diag(S), use V^T/s in place of (V^T S^-1)
    u, s, vh = np.linalg.svd(C, 0)
    RI = (vh.T/s) @ (u.T @ SD)
    print("SD", SD)
    print("RI", RI)

if __name__ == "__main__":
    demo()
