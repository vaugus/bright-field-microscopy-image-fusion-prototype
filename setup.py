import numpy as np
import scipy
from scipy.ndimage import _nd_image, _ni_support
from scipy.ndimage import convolve, correlate, correlate1d, convolve1d
from scipy.signal import oaconvolve, fftconvolve
from pytictoc import TicToc

def gaussian_filter(input, sigma, order=0, output=None,
                    mode="reflect", cval=0.0, truncate=4.0):
    input = np.asarray(input)
    output = _ni_support._get_output(output, input)
    orders = _ni_support._normalize_sequence(order, input.ndim)
    sigmas = _ni_support._normalize_sequence(sigma, input.ndim)
    modes = _ni_support._normalize_sequence(mode, input.ndim)
    axes = list(range(input.ndim))
    axes = [(axes[ii], sigmas[ii], orders[ii], modes[ii])
            for ii in range(len(axes)) if sigmas[ii] > 1e-15]
    if len(axes) > 0:
        for axis, sigma, order, mode in axes:
            gaussian_filter1d(input, sigma, axis, order, output,
                              mode, cval, truncate)

            input = output
    else:
        output[...] = input[...]
    return output

def _gaussian_kernel1d(sigma, order, radius):
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = np.arange(order + 1)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = np.zeros(order + 1)
        q[0] = 1
        D = np.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = np.diag(np.ones(order)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x

def gaussian_filter1d(input, sigma, axis=-1, order=0, output=None,
                      mode="reflect", cval=0.0, truncate=4.0):
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    # Since we are calling correlate, not convolve, revert the kernel
    weights = _gaussian_kernel1d(sigma, order, lw)[::-1]

    # truncate : float
    # Truncate the filter at this many standard deviations.
    # Default is 4.0.
    # print(lw)
    # print(sd)
    # print(weights.shape)

    return scipy.ndimage.correlate1d(input, weights, axis, output, mode, cval, 0)

def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

def gaussian_kernel(size, sigma=1, verbose=False, dims=1):

    kernel_1D = np.linspace(-(size // 2), size // 2, size)

    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)

    if dims == 1:
        return kernel_1D

    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    # kernel_2D *= 1.0 / kernel_2D.max()

    if verbose:
        plt.imshow(kernel_2D, interpolation='none',cmap='gray')
        plt.title("Image")
        plt.show()

    return kernel_2D

def frequency_domain_convolution(img, w):
    """Método frequency_domain_convolution.
    Método que realiza a convolução de um filtro em uma imagem.
    
    :param img          A imagem no domínio de Fourier.
    :param w            O filtro no domínio de Fourier.
    :return             A imagem convoluída.
    """
    a, b = img.shape
    # c, d = w.reshape((-1, 1)).shape

    # size = (np.max([a, b, d]), c)

    # mask = np.zeros((size), np.complex128)
    # image = np.zeros((size), np.complex128)

    # for i in range(c):
    #     mask[0, i] = w[i]

    # for i in range(a):
    #     for j in range(b):
    #         image[i, j] = img[i,j]


    # FFT.
    W = np.fft.fft(w.reshape(-1, 1))
    tmp = np.fft.fft(img.ravel().reshape(-1, 1))

    return np.multiply(tmp, W.T)

def main():

    t = TicToc()

    size = 2560 * 1920
    sigma = 500

    # a = np.arange(size, step=1).reshape((5,5))
    a = np.arange(size, step=1).reshape((2560,1920))

    radius = int(4 * sigma + 0.5)
    L = np.arange(-radius, radius + 1).size
    # print(L)

    # # t.tic()
    kernel = gaussian_kernel(size=L, sigma=sigma, dims=2)
    
    kernel_1D = gaussian_kernel(size=L, sigma=sigma, dims=1)

    t.tic()
    tmp = convolve1d(a, kernel_1D, axis=0)
    x = convolve1d(tmp, kernel_1D, axis=1)
    t.toc()

    # t.tic()
    # K = convolve(a, kernel)
    # t.toc()

    t.tic()
    y = gaussian_filter(a, sigma=sigma)
    t.toc()

    print("Array")
    print(a)
    print()
    print("My version - 1D")
    print(x)
    # print()
    # print("My version - 2D")
    # print(K)
    print()
    print("Scipy")
    print(gaussian_filter(a, sigma=sigma))

if __name__ == "__main__":
	main()
