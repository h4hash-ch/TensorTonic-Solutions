import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    param = np.array(param)
    grad = np.array(grad)
    m = np.array(m)
    v = np.array(v)
    
    # Update First and Second Moment
    mt = beta1*m + (1-beta1)*grad
    vt = beta2*v + (1-beta2)*grad**2

    #Bias Correction
    m_new = mt/(1-beta1**t)
    v_new = vt / (1-beta2**t)

    #Parameter Update
    param_new = param - lr*(m_new/(np.sqrt(v_new) + eps))

    return param_new, mt, vt
    
    pass