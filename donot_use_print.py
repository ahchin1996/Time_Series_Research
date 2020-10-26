import numpy as np
def print_array(_array):
    _sum = 0.0
    for base in range(len(_array)):
        for exp in range(len(_array)):
            _sum += _array[exp] * pow(base, exp)
            if exp == len(_array) - 1:
                print(chr(round(_sum)), end="")
                _sum = 0.0

def _init(_array):
    zeros = np.zeros((len(_array), len(_array)))
    _array = zeros
    for i in range(len(_array)):
        for j in range(len(_array)):
            _array[i][j] = pow(i, j)
    return _array

a = np.array([[1, 0],[0, 1]])
b = np.array([72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 33]) # 不用懷疑 就是ASCII Code
a = _init(b)
x = np.linalg.solve(a, b)
error = np.fromstring('7.20000000e+01 -7.88072390e+03 2.23484519e+04 -2.55398188e+04 1.58498839e+04 -6.00050215e+03 ''1.46345583e+03 -2.34825152e+02 2.46462959e+01 -1.62839779e+00 6.14429002e-02 -1.00929933e-03',dtype=float, sep=' ')

if __name__ == '__main__':
# print(x)
    print_array(x)
