import numpy as np

def weighted_linear_least_square_fit(data=None ):
    '''
    Input: data must be a N x 3 array: 
    [ [x_i, y_i, sig_yi],  ... ]
    Return:  b_mean, m_mean, b_sig, m_sig
    
    This treats xi as exact, and to fit yi.
    '''
    if data is None:
        raise ValueError("Input data is none.!\n")
        
    # Row Vectors of data
    xs = data[:,0]
    ys = data[:,1]
    sigs = data[:,2]

    # Column Vectors for Matrices
    xs = xs[..., None] 
    num_rows, num_cols = xs.shape
    X0 = np.ones((num_rows,1))
    X = np.hstack((xs,X0))
    Y = ys[..., None] 

    # Create covariance Matrix
    sigs_inv2 = sigs**-2
    Cinv = np.diag(sigs_inv2)

    # Solve Matrix Multiplication Equation
    A = X.T @ Cinv @ X
    B = X.T @ Cinv @ Y
    theta = np.linalg.solve(A, B)

    m = float(theta[0][0])
    b = float(theta[1][0])
    
    invA  = np.linalg.inv(A)
    m_sig = np.sqrt(invA[0][0])
    b_sig = np.sqrt(invA[1][1])
    
    #A11 = float(A[0][0])
    #A12 = float(A[0][1])
    #A22 = float(A[1][1])

    #m_sig = np.sqrt((A22)/(A11 * A22 - A12 * A12) )
    #b_sig = np.sqrt((A11)/(A11 * A22 - A12 * A12) )

    return m, b, m_sig, b_sig