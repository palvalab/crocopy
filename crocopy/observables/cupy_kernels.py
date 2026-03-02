# Third-party imports
import cupy as cp

# our windows servers are unable to run updated cupy versions and util module name is different across them
# lets cover both cases
try:
    from cupy._util import memoize as cp_memoize
except ImportError:
    from cupy.util import memoize as cp_memoize

# I have created the second kernel to avoid necessity for a couple of ifs 
# to check is it debiased version or not (it increases worktime by ~10% !)
# that is a bit weird but life is like this.

_wpli_code = r'''
    #include <cupy/complex.cuh>

    #define TILE_DIM 16
    
    template<typename T>
    __global__ void wpli_matmul(const complex<T> *A, const complex<T> *B, T *C, 
                                    int a_n_rows, int a_n_cols, 
                                    int b_n_rows, int b_n_cols, 
                                    int c_n_rows, int c_n_cols)
    {
       T c_value = 0.0;

       int curr_row = blockIdx.y*TILE_DIM + threadIdx.y;
       int curr_col = blockIdx.x*TILE_DIM + threadIdx.x;

       __shared__ complex<T> As[TILE_DIM][TILE_DIM];
       __shared__ complex<T> Bs[TILE_DIM][TILE_DIM];

       for (int tile_idx = 0; tile_idx < (TILE_DIM + a_n_cols - 1)/TILE_DIM; tile_idx++) {
             if (tile_idx*TILE_DIM + threadIdx.x < a_n_cols && curr_row < a_n_rows)
             {
                As[threadIdx.y][threadIdx.x] = A[curr_row*a_n_cols + tile_idx*TILE_DIM + threadIdx.x];
             }
             else
             {
                As[threadIdx.y][threadIdx.x] = 0.0;
             }

             if (tile_idx*TILE_DIM + threadIdx.y < b_n_rows && curr_col < b_n_cols)
             {
                Bs[threadIdx.y][threadIdx.x] = B[(tile_idx*TILE_DIM + threadIdx.y)*b_n_cols + curr_col];
             }
             else
             {
                Bs[threadIdx.y][threadIdx.x] = 0.0;
             }

             __syncthreads();

             #pragma unroll
             for (int n = 0; n < TILE_DIM; ++n)
             {
                c_value += abs((As[threadIdx.y][n] * Bs[n][threadIdx.x]).imag());
             }

             __syncthreads();
       }

       if (curr_row < c_n_rows && curr_col < c_n_cols) {
          C[curr_row*c_n_cols + curr_col] = c_value;
       }
    }
'''

_wpli_code += r'''
template<typename T>
__global__ void wpli_matmul_debiased(const complex<T> *A, const complex<T> *B, T *C, 
                                        int a_n_rows, int a_n_cols, 
                                        int b_n_rows, int b_n_cols, 
                                        int c_n_rows, int c_n_cols,
                                        T *Z)
    {
       T c_value = 0.0;
       T z_value = 0.0;

       int curr_row = blockIdx.y*TILE_DIM + threadIdx.y;
       int curr_col = blockIdx.x*TILE_DIM + threadIdx.x;

       __shared__ complex<T> As[TILE_DIM][TILE_DIM];
       __shared__ complex<T> Bs[TILE_DIM][TILE_DIM];

       for (int tile_idx = 0; tile_idx < (TILE_DIM + a_n_cols - 1)/TILE_DIM; tile_idx++) {
             if (tile_idx*TILE_DIM + threadIdx.x < a_n_cols && curr_row < a_n_rows)
             {
                As[threadIdx.y][threadIdx.x] = A[curr_row*a_n_cols + tile_idx*TILE_DIM + threadIdx.x];
             }
             else
             {
                As[threadIdx.y][threadIdx.x] = 0.0;
             }

             if (tile_idx*TILE_DIM + threadIdx.y < b_n_rows && curr_col < b_n_cols)
             {
                Bs[threadIdx.y][threadIdx.x] = B[(tile_idx*TILE_DIM + threadIdx.y)*b_n_cols + curr_col];
             }
             else
             {
                Bs[threadIdx.y][threadIdx.x] = 0.0;
             }

             __syncthreads();

             #pragma unroll
             for (int n = 0; n < TILE_DIM; ++n)
             {
                T prod = (As[threadIdx.y][n] * Bs[n][threadIdx.x]).imag();
                c_value += abs(prod);
                z_value += prod * prod;
             }

             __syncthreads();
       }

       if (curr_row < c_n_rows && curr_col < c_n_cols) {
          int offset = curr_row*c_n_cols + curr_col;
          C[offset] = c_value;
          Z[offset] = z_value;
       }
    }
'''

@cp_memoize(for_each_device=True)
def get_wpli_kernel(dtype, debias: bool):
    cupy_major = int(cp.__version__.split('.')[0])
    if cupy_major >= 8:
        name_expressions = ['wpli_matmul<float>', 'wpli_matmul<double>', 'wpli_matmul_debiased<float>', 'wpli_matmul_debiased<double>']

      
        mod = cp.RawModule(code=_wpli_code, options=('--std=c++11',),
                           name_expressions=name_expressions)
    else:
        mod = cp.RawModule(code=_wpli_code, options=('--std=c++11',))
    
    base_dtype = 'double' if (dtype.char == 'D') else 'float'
    func_name = 'wpli_matmul_debiased' if (debias == True) else 'wpli_matmul'
    
    ker = mod.get_function(f'{func_name}<{base_dtype}>')
    
    return ker

_pac_kernel = cp.ElementwiseKernel(
   'T x, raw T y, raw I lags, int32 n_cols', 'raw C output',
   '''
   int curr_col = i % n_cols;
   int curr_row = i / n_cols;

   int sample_lag = lags[curr_row];
   int offset_idx = curr_col + sample_lag;

   if(offset_idx < n_cols) {
      int compare_idx = curr_row*n_cols + offset_idx;
      output[i] = x * y[compare_idx];
   }
   else
   {
      output[i] = __int_as_float(0xFFE00000);
   }
                           
   ''',
   '_pac_kernel',
   )


_cc_kernel_code = r'''
#include <cupy/complex.cuh>

//extern "C" __global__
template<typename T>
__global__ void pairwise_orth_corr_kernel(const complex<T>* __restrict__ S,  
                                          T* __restrict__ corr,               // (n_signals, n_signals), row-major, contiguous
                                          const int n_signals,
                                          const int n_samples,
                                          const ptrdiff_t s0,                 // stride along signals (axis 0), in elements of complex<T>
                                          const ptrdiff_t s1)                 // stride along samples (axis 1), in elements of complex<T>
{
    const int i = blockIdx.x;
    const int j = blockIdx.y;
    if (i >= n_signals || j >= n_signals) return;

    // ---- shared memory (5 partial sums of real T), properly aligned
    const size_t t_size = sizeof(T);
    extern __shared__ __align__(16) unsigned char sdata[];  // <= fixed alignment
    T* sX  = reinterpret_cast<T*>(sdata);
    T* sY  = reinterpret_cast<T*>(sdata + 1*blockDim.x*t_size);
    T* sXX = reinterpret_cast<T*>(sdata + 2*blockDim.x*t_size);
    T* sYY = reinterpret_cast<T*>(sdata + 3*blockDim.x*t_size);
    T* sXY = reinterpret_cast<T*>(sdata + 4*blockDim.x*t_size);

    T sumX = 0, sumY = 0, sumXX = 0, sumYY = 0, sumXY = 0;

    const ptrdiff_t base_i = static_cast<ptrdiff_t>(i) * s0;
    const ptrdiff_t base_j = static_cast<ptrdiff_t>(j) * s0;
    const complex<T> I_literal(0.0, 1.0);

    for (int k = threadIdx.x; k < n_samples; k += blockDim.x) {
        const ptrdiff_t idx_i = base_i + static_cast<ptrdiff_t>(k) * s1;
        const ptrdiff_t idx_j = base_j + static_cast<ptrdiff_t>(k) * s1;

        complex<T> x = S[idx_i];
        T ax = abs(x);
        // avoid NaN when x == 0
        complex<T> x_normed = (ax > static_cast<T>(0)) ? (x / ax) : complex<T>(0, 0);

        complex<T> y = S[idx_j];

        T ls = (y * conj(x_normed)).imag();
        complex<T> rs = I_literal * x_normed;

        T X = ax;
        T Y = abs(ls * rs);

        sumX  += X;
        sumY  += Y;
        sumXX += X * X;
        sumYY += Y * Y;
        sumXY += X * Y;
    }

    sX [threadIdx.x] = sumX;
    sY [threadIdx.x] = sumY;
    sXX[threadIdx.x] = sumXX;
    sYY[threadIdx.x] = sumYY;
    sXY[threadIdx.x] = sumXY;
    __syncthreads();

    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            sX [threadIdx.x] += sX [threadIdx.x + off];
            sY [threadIdx.x] += sY [threadIdx.x + off];
            sXX[threadIdx.x] += sXX[threadIdx.x + off];
            sYY[threadIdx.x] += sYY[threadIdx.x + off];
            sXY[threadIdx.x] += sXY[threadIdx.x + off];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const T n = static_cast<T>(n_samples);
        const T num = sXY[0] - (sX[0] * sY[0] / n);
        const T den = sqrt( (sXX[0] - (sX[0]*sX[0])/n) * (sYY[0] - (sY[0]*sY[0])/n) );
        corr[i * n_signals + j] = (den > static_cast<T>(0)) ? (num / den) : static_cast<T>(0);
    }
}
'''

@cp_memoize(for_each_device=True)
def _get_occ_gpu_kernel(dtype):
    name_expressions =  ['pairwise_orth_corr_kernel<float>', 'pairwise_orth_corr_kernel<double>']
    mod = cp.RawModule(code=_cc_kernel_code, options=("-std=c++17",), name_expressions=name_expressions)
    
    base_dtype = 'double' if (dtype.char == 'D') else 'float'

    kernel = mod.get_function(f'pairwise_orth_corr_kernel<{base_dtype}>')
    
    return kernel

def _occ_cupy(data, block_size=256):
    kernel = _get_occ_gpu_kernel(data.dtype)

    n_signals, n_samples = data.shape

    elem_size = data.itemsize
    strides_ax0 = data.strides[0] // elem_size
    strides_ax1 = data.strides[1] // elem_size

    corr_mat = cp.zeros((n_signals, n_signals), dtype=data.real.dtype)

    # Shared memory usage: 5 partial sums (sumX, sumY, sumXX, sumYY, sumXY) per thread
    shared_mem_size = 5 * block_size * data.real.dtype.itemsize

    grid = (n_signals, n_signals, 1)
    block = (block_size,)
    kernel_args = (data, corr_mat, cp.int32(n_signals), cp.int32(n_samples), cp.intp(strides_ax0), cp.intp(strides_ax1))

    kernel(grid, block, kernel_args, shared_mem=shared_mem_size)

    return corr_mat