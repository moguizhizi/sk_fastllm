
template <int VEC_SIZE, typename InT, typename ScaOp>
struct DefaultReadVecOp
{
    ScaOp scalar_op;

    __device__ __forceinline__ void operator()(
        const vec_n_t<InT, VEC_SIZE> &src) const
    {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i)
        {
            scalar_op(src.val[i]);
        }
    }
};

// read-only version: iterate over the input with alignment guarantees
template <int VEC_SIZE, typename InT, typename VecOp, typename ScaOp>
__device__ inline void vectorize_read_with_alignment(const InT *in, int len,
                                                     int tid, int stride,
                                                     VecOp &&vec_op,
                                                     ScaOp &&scalar_op)
{
    static_assert(VEC_SIZE > 0 && (VEC_SIZE & (VEC_SIZE - 1)) == 0,
                  "VEC_SIZE must be a positive power-of-two");
    constexpr int WIDTH = VEC_SIZE * sizeof(InT);
    uintptr_t addr = reinterpret_cast<uintptr_t>(in);

    // fast path when the whole region is already aligned
    bool can_vec = ((addr & (WIDTH - 1)) == 0) && ((len & (VEC_SIZE - 1)) == 0);
    if (can_vec)
    {
        int num_vec = len / VEC_SIZE;

        using vin_t = vec_n_t<InT, VEC_SIZE>;
        auto *v_in = reinterpret_cast<const vin_t *>(in);

        for (int i = tid; i < num_vec; i += stride)
        {
            vin_t tmp = v_in[i];
            vec_op(tmp);
        }
        return;
    }

    int misalignment_offset = addr & (WIDTH - 1);
    int alignment_bytes = WIDTH - misalignment_offset;
    int prefix_elems = alignment_bytes & (WIDTH - 1);
    prefix_elems /= sizeof(InT);
    prefix_elems = min(prefix_elems, len);

    // 1. handle the possibly unaligned prefix with scalar access.
    for (int i = tid; i < prefix_elems; i += stride)
    {
        scalar_op(in[i]);
    }

    in += prefix_elems;
    len -= prefix_elems;

    int num_vec = len / VEC_SIZE;
    using vin_t = vec_n_t<InT, VEC_SIZE>;
    auto *v_in = reinterpret_cast<const vin_t *>(in);

    // 2. vectorized traversal of the main aligned region.
    for (int i = tid; i < num_vec; i += stride)
    {
        vec_op(v_in[i]);
    }

    // 3. handle remaining tail elements.
    int tail_start = num_vec * VEC_SIZE;
    for (int i = tid + tail_start; i < len; i += stride)
    {
        scalar_op(in[i]);
    }
}