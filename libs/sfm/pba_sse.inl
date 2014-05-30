#include <immintrin.h>

#if defined(__AVX__)
inline double hsum(__m256d value)
{
    double res;
    _mm256_store1_pd(&res, _mm256_hadd_pd(value, value));
    return res;
}
#elif defined(__SSE3__)
inline double hsum(__m128d value)
{
    double res;
    _mm_store1_pd(&res, _mm_hadd_pd(value, value));
    return res;
}
#elif defined(__SSE2__)
inline double hsum(__m128d value)
{
    double res;
    _mm_store1_pd(&res, _mm_add_pd(value, _mm_shuffle_pd(value, value, _MM_SHUFFLE2(0, 1))));
    return res;
}
#endif

#ifdef CPUPBA_USE_SSE
#define CPUPBA_USE_SIMD
namespace MYSSE
{
    template<class Float> class SSE{};
    template<>  class SSE<float>  {
        public: typedef  __m128 sse_type;
        static inline sse_type zero() {return _mm_setzero_ps();}    };
    template<>  class SSE<double> {
        public: typedef  __m128d sse_type;
        static inline sse_type zero() {return _mm_setzero_pd();}    };

    ////////////////////////////////////////////
    template <class Float> inline size_t sse_step()    {return 16 / sizeof(Float); }
    inline __m128 sse_load1(const float* p)     {return _mm_load1_ps(p);}
    inline __m128 sse_load(const float* p)      {return _mm_load_ps(p);}
    inline __m128 sse_add(__m128 s1, __m128 s2)     {return _mm_add_ps(s1, s2);}
    inline __m128 sse_sub(__m128 s1, __m128 s2)     {return _mm_sub_ps(s1, s2);}
    inline __m128 sse_mul(__m128 s1, __m128 s2)     {return _mm_mul_ps(s1, s2);}
    inline __m128 sse_sqrt(__m128 s)                {return _mm_sqrt_ps(s); }

    inline __m128d sse_load1(const double* p)       {return _mm_load1_pd(p);}
    inline __m128d sse_load(const double* p)        {return _mm_load_pd(p);}
    inline __m128d sse_add(__m128d s1, __m128d s2)      {return _mm_add_pd(s1, s2);}
    inline __m128d sse_sub(__m128d s1, __m128d s2)      {return _mm_sub_pd(s1, s2);}
    inline __m128d sse_mul(__m128d s1, __m128d s2)      {return _mm_mul_pd(s1, s2);}
    inline __m128d sse_sqrt(__m128d s)                  {return _mm_sqrt_pd(s); }

#ifdef _WIN32
    inline float    sse_sum(__m128 s)    {return (s.m128_f32[0]+ s.m128_f32[2])  + (s.m128_f32[1] +s.m128_f32[3]);}
    inline double   sse_sum(__m128d s)    {return s.m128d_f64[0] + s.m128d_f64[1];}
#else
    inline float    sse_sum(__m128 s)    {float *f = (float*) (&s); return (f[0] + f[2]) + (f[1] + f[3]);}
    inline double   sse_sum(__m128d s)   {double *d = (double*) (&s); return d[0] + d[1];}
#endif
    //inline float  sse_dot(__m128 s1, __m128 s2)	{__m128 temp = _mm_dp_ps(s1, s2, 0xF1);  	float* f = (float*) (&temp); return f[0]; 	}
    //inline double  sse_dot(__m128d s1, __m128d s2)	{__m128d temp = _mm_dp_pd(s1, s2, 0x31);  	double* f = (double*) (&temp); return f[0] ; }
    inline void    sse_store(float *p, __m128 s){_mm_store_ps(p, s); }
    inline void    sse_store(double *p, __m128d s)  {_mm_store_pd(p, s); }
}

namespace ProgramCPU
{
    using namespace MYSSE;
    #define SSE_ZERO SSE<Float>::zero()
    #define SSE_T typename SSE<Float>::sse_type
    /////////////////////////////
    inline void   ScaleJ4(float* jcx, float* jcy, const float* sj)
    {
         __m128 ps = _mm_load_ps(sj);
        _mm_store_ps(jcx, _mm_mul_ps(_mm_load_ps(jcx), ps));
        _mm_store_ps(jcy, _mm_mul_ps(_mm_load_ps(jcy), ps));
    }
    inline void   ScaleJ8(float* jcx, float* jcy, const float* sj)
    {
        ScaleJ4(jcx, jcy, sj);
        ScaleJ4(jcx + 4, jcy + 4, sj + 4);
    }
    inline void   ScaleJ4(double* jcx, double* jcy, const double* sj)
    {
         __m128d ps1 = _mm_load_pd(sj), ps2 = _mm_load_pd(sj + 2);
        _mm_store_pd(jcx    , _mm_mul_pd(_mm_load_pd(jcx), ps1));
        _mm_store_pd(jcy    , _mm_mul_pd(_mm_load_pd(jcy), ps1));
        _mm_store_pd(jcx + 2, _mm_mul_pd(_mm_load_pd(jcx + 2), ps2));
        _mm_store_pd(jcy + 2, _mm_mul_pd(_mm_load_pd(jcy + 2), ps2));
    }
    inline void   ScaleJ8(double* jcx, double* jcy, const double* sj)
    {
        ScaleJ4(jcx, jcy, sj);
        ScaleJ4(jcx + 4, jcy + 4, sj + 4);
    }
    inline float   DotProduct8(const float* v1, const float* v2)
    {
        __m128 ds = _mm_add_ps(
            _mm_mul_ps(_mm_load_ps(v1),     _mm_load_ps(v2)),
            _mm_mul_ps(_mm_load_ps(v1 + 4), _mm_load_ps(v2 + 4)));
        return sse_sum(ds);
    }
    inline double   DotProduct8(const double* v1, const double* v2)
    {
        __m128d d1 = _mm_mul_pd(_mm_load_pd(v1),     _mm_load_pd(v2)) ;
        __m128d d2 = _mm_mul_pd(_mm_load_pd(v1 + 2), _mm_load_pd(v2 + 2));
        __m128d d3 = _mm_mul_pd(_mm_load_pd(v1 + 4), _mm_load_pd(v2 + 4));
        __m128d d4 = _mm_mul_pd(_mm_load_pd(v1 + 6), _mm_load_pd(v2 + 6));
        __m128d ds = _mm_add_pd(_mm_add_pd(d1, d2),  _mm_add_pd(d3, d4));
        return sse_sum(ds);
    }

    inline void  ComputeTwoJX(const float* jc, const float* jp, const float* xc, const float* xp, float* jx)
    {
#ifdef POINT_DATA_ALIGN4
        __m128 xc1 = _mm_load_ps(xc), xc2 = _mm_load_ps(xc + 4), mxp = _mm_load_ps(xp);
        __m128 ds1 = _mm_add_ps(_mm_mul_ps(_mm_load_ps(jc),  xc1), _mm_mul_ps(_mm_load_ps(jc + 4), xc2));
        __m128 dx1 = _mm_add_ps(ds1, _mm_mul_ps(_mm_load_ps(jp), mxp));
        jx[0] = sse_sum(dx1);
        __m128 ds2 = _mm_add_ps(_mm_mul_ps(_mm_load_ps(jc + 8), xc1), _mm_mul_ps(_mm_load_ps(jc + 12), xc2));
        __m128 dx2 = _mm_add_ps(ds2, _mm_mul_ps(_mm_load_ps(jp + 4), mxp));
        jx[1] = sse_sum(dx2);
#else
        __m128 xc1 = _mm_load_ps(xc),		xc2 = _mm_load_ps(xc + 4);
        __m128 jc1 = _mm_load_ps(jc),       jc2 = _mm_load_ps(jc + 4);
        __m128 jc3 = _mm_load_ps(jc + 8),   jc4 = _mm_load_ps(jc + 12);
        __m128 ds1 = _mm_add_ps(_mm_mul_ps(jc1, xc1), _mm_mul_ps(jc2, xc2));
        __m128 ds2 = _mm_add_ps(_mm_mul_ps(jc3, xc1), _mm_mul_ps(jc4, xc2));
        jx[0] = sse_sum(ds1) + (jp[0] * xp[0] + jp[1] * xp[1] + jp[2] * xp[2]);
        jx[1] = sse_sum(ds2) + (jp[POINT_ALIGN] * xp[0] + jp[POINT_ALIGN+1] * xp[1] + jp[POINT_ALIGN+2] * xp[2]);
        /*jx[0] = (sse_dot(jc1, xc1) + sse_dot(jc2, xc2)) + (jp[0] * xp[0] + jp[1] * xp[1] + jp[2] * xp[2]);
        jx[1] = (sse_dot(jc3, xc1) + sse_dot(jc4, xc2)) + (jp[POINT_ALIGN] * xp[0] + jp[POINT_ALIGN+1] * xp[1] + jp[POINT_ALIGN+2] * xp[2]);*/
#endif
    }

    inline void ComputeTwoJX(const double* jc, const double* jp, const double* xc, const double* xp, double* jx)
    {
        __m128d xc1 = _mm_load_pd(xc), xc2 = _mm_load_pd(xc +2), xc3 = _mm_load_pd(xc + 4), xc4 = _mm_load_pd(xc + 6);
        __m128d d1 = _mm_mul_pd(_mm_load_pd(jc),     xc1);
        __m128d d2 = _mm_mul_pd(_mm_load_pd(jc + 2), xc2);
        __m128d d3 = _mm_mul_pd(_mm_load_pd(jc + 4), xc3);
        __m128d d4 = _mm_mul_pd(_mm_load_pd(jc + 6), xc4);
        __m128d ds1 = _mm_add_pd(_mm_add_pd(d1, d2),  _mm_add_pd(d3, d4));
        jx[0] = sse_sum(ds1)  + (jp[0] * xp[0] + jp[1] * xp[1] + jp[2] * xp[2]);

        __m128d d5 = _mm_mul_pd(_mm_load_pd(jc + 8 ), xc1);
        __m128d d6 = _mm_mul_pd(_mm_load_pd(jc + 10), xc2);
        __m128d d7 = _mm_mul_pd(_mm_load_pd(jc + 12), xc3);
        __m128d d8 = _mm_mul_pd(_mm_load_pd(jc + 14), xc4);
        __m128d ds2 = _mm_add_pd(_mm_add_pd(d5, d6),  _mm_add_pd(d7, d8));
        jx[1] = sse_sum(ds2) + (jp[POINT_ALIGN] * xp[0] + jp[POINT_ALIGN+1] * xp[1] + jp[POINT_ALIGN+2] * xp[2]);
    }

    //v += ax
    inline void   AddScaledVec8(float a, const float* x, float* v)
    {
        __m128 aa = sse_load1(&a);
        _mm_store_ps(v  , _mm_add_ps( _mm_mul_ps(_mm_load_ps(x    ), aa), _mm_load_ps(v  )));
        _mm_store_ps(v+4, _mm_add_ps( _mm_mul_ps(_mm_load_ps(x + 4), aa), _mm_load_ps(v+4)));
    }
    //v += ax
    inline void   AddScaledVec8(double a, const double* x, double* v)
    {
        __m128d aa = sse_load1(&a);
        _mm_store_pd(v  , _mm_add_pd( _mm_mul_pd(_mm_load_pd(x    ), aa), _mm_load_pd(v  )));
        _mm_store_pd(v+2, _mm_add_pd( _mm_mul_pd(_mm_load_pd(x + 2), aa), _mm_load_pd(v+2)));
        _mm_store_pd(v+4, _mm_add_pd( _mm_mul_pd(_mm_load_pd(x + 4), aa), _mm_load_pd(v+4)));
        _mm_store_pd(v+6, _mm_add_pd( _mm_mul_pd(_mm_load_pd(x + 6), aa), _mm_load_pd(v+6)));
    }

    inline void AddBlockJtJ(const float * jc, float * block, int vn)
    {
        __m128 j1 = _mm_load_ps(jc);
        __m128 j2 = _mm_load_ps(jc + 4);
        for(int i = 0; i < vn; ++i, ++jc, block += 8)
        {
            __m128 a = sse_load1(jc);
            _mm_store_ps(block + 0, _mm_add_ps(_mm_mul_ps(a, j1), _mm_load_ps(block + 0)));
            _mm_store_ps(block + 4, _mm_add_ps(_mm_mul_ps(a, j2), _mm_load_ps(block + 4)));
        }
    }

    inline void AddBlockJtJ(const double * jc, double * block, int vn)
    {
        __m128d j1 = _mm_load_pd(jc);
        __m128d j2 = _mm_load_pd(jc + 2);
        __m128d j3 = _mm_load_pd(jc + 4);
        __m128d j4 = _mm_load_pd(jc + 6);
        for(int i = 0; i < vn; ++i, ++jc, block += 8)
        {
            __m128d a = sse_load1(jc);
            _mm_store_pd(block + 0, _mm_add_pd(_mm_mul_pd(a, j1), _mm_load_pd(block + 0)));
            _mm_store_pd(block + 2, _mm_add_pd(_mm_mul_pd(a, j2), _mm_load_pd(block + 2)));
            _mm_store_pd(block + 4, _mm_add_pd(_mm_mul_pd(a, j3), _mm_load_pd(block + 4)));
            _mm_store_pd(block + 6, _mm_add_pd(_mm_mul_pd(a, j4), _mm_load_pd(block + 6)));
        }
    }
}
#endif

#ifdef CPUPBA_USE_AVX
#define CPUPBA_USE_SIMD
namespace MYAVX
{
    template<class Float> class SSE{};
    template<>  class SSE<float>  {
        public: typedef  __m256 sse_type;   //static size_t   step() {return 4;}
        static inline sse_type zero() {return _mm256_setzero_ps();}    };
    template<>  class SSE<double> {
        public: typedef  __m256d sse_type;  //static size_t   step() {return 2;}
        static inline sse_type zero() {return _mm256_setzero_pd();}    };

    ////////////////////////////////////////////
    template <class Float> inline size_t sse_step()    {return 32 / sizeof(Float); };
    inline __m256 sse_load1(const float* p)     {return _mm256_broadcast_ss(p);}
    inline __m256 sse_load(const float* p)      {return _mm256_load_ps(p);}
    inline __m256 sse_add(__m256 s1, __m256 s2)     {return _mm256_add_ps(s1, s2);}
    inline __m256 sse_sub(__m256 s1, __m256 s2)     {return _mm256_sub_ps(s1, s2);}
    inline __m256 sse_mul(__m256 s1, __m256 s2)     {return _mm256_mul_ps(s1, s2);}
    inline __m256 sse_sqrt(__m256 s)                {return _mm256_sqrt_ps(s); }

    //inline __m256 sse_fmad(__m256 a, __m256 b, __m256 c) {return _mm256_fmadd_ps(a, b, c);}

    inline __m256d sse_load1(const double* p)       {return _mm256_broadcast_sd(p);}
    inline __m256d sse_load(const double* p)        {return _mm256_load_pd(p);}
    inline __m256d sse_add(__m256d s1, __m256d s2)      {return _mm256_add_pd(s1, s2);}
    inline __m256d sse_sub(__m256d s1, __m256d s2)      {return _mm256_sub_pd(s1, s2);}
    inline __m256d sse_mul(__m256d s1, __m256d s2)      {return _mm256_mul_pd(s1, s2);}
    inline __m256d sse_sqrt(__m256d s)                  {return _mm256_sqrt_pd(s); }

#ifdef _WIN32
    inline float    sse_sum(__m256 s)    {return ((s.m256_f32[0]  + s.m256_f32[4]) + (s.m256_f32[2]+ s.m256_f32[6])) +
                                                 ((s.m256_f32[1]  + s.m256_f32[5]) + (s.m256_f32[3] +s.m256_f32[7]));}
    inline double   sse_sum(__m256d s)    {return (s.m256d_f64[0] + s.m256d_f64[2]) + (s.m256d_f64[1] + s.m256d_f64[3]);}
#else
    inline float    sse_sum(__m128 s)    {float *f = (float*) (&s); return ((f[0] + f[4]) + (f[2] + f[6])) + ((f[1] + f[5]) + (f[3] + f[7]));}
    inline double   sse_sum(__m128d s)   {double *d = (double*) (&s); return (d[0] + d[2]) + (d[1] + d[3]);}
#endif
    inline float  sse_dot(__m256 s1, __m256 s2)
    {
        __m256 temp = _mm256_dp_ps(s1, s2, 0xf1);
        float* f = (float*) (&temp); return f[0] + f[4];
    }
    inline double  sse_dot(__m256d s1, __m256d s2)		{return sse_sum(sse_mul(s1, s2));}

    inline void    sse_store(float *p, __m256 s){_mm256_store_ps(p, s); }
    inline void    sse_store(double *p, __m256d s)  {_mm256_store_pd(p, s); }
};

namespace ProgramCPU
{
    using namespace MYAVX;
    #define SSE_ZERO SSE<Float>::zero()
    #define SSE_T typename SSE<Float>::sse_type

    /////////////////////////////
    inline void   ScaleJ8(float* jcx, float* jcy, const float* sj)
    {
         __m256 ps = _mm256_load_ps(sj);
        _mm256_store_ps(jcx, _mm256_mul_ps(_mm256_load_ps(jcx), ps));
        _mm256_store_ps(jcy, _mm256_mul_ps(_mm256_load_ps(jcy), ps));
    }
    inline void   ScaleJ4(double* jcx, double* jcy, const double* sj)
    {
         __m256d ps = _mm256_load_pd(sj);
         _mm256_store_pd(jcx, _mm256_mul_pd(_mm256_load_pd(jcx), ps));
         _mm256_store_pd(jcy, _mm256_mul_pd(_mm256_load_pd(jcy), ps));
    }
    inline void   ScaleJ8(double* jcx, double* jcy, const double* sj)
    {
        ScaleJ4(jcx, jcy, sj);
        ScaleJ4(jcx + 4, jcy + 4, sj + 4);
    }
    inline float   DotProduct8(const float* v1, const float* v2)
    {
        return sse_dot(_mm256_load_ps(v1), _mm256_load_ps(v2));
    }
    inline double   DotProduct8(const double* v1, const double* v2)
    {
        __m256d ds = _mm256_add_pd(
            _mm256_mul_pd(_mm256_load_pd(v1),     _mm256_load_pd(v2)),
            _mm256_mul_pd(_mm256_load_pd(v1 + 4), _mm256_load_pd(v2 + 4)));
        return sse_sum(ds);
    }

    inline void  ComputeTwoJX(const float* jc, const float* jp, const float* xc, const float* xp, float* jx)
    {
        __m256 xcm = _mm256_load_ps(xc), jc1 = _mm256_load_ps(jc), jc2 = _mm256_load_ps(jc + 8);
        jx[0] = sse_dot(jc1, xcm) +  (jp[0] * xp[0] + jp[1] * xp[1] + jp[2] * xp[2]);
        jx[1] = sse_dot(jc2, xcm) + (jp[POINT_ALIGN] * xp[0] + jp[POINT_ALIGN+1] * xp[1] + jp[POINT_ALIGN+2] * xp[2]);
    }

    inline void ComputeTwoJX(const double* jc, const double* jp, const double* xc, const double* xp, double* jx)
    {
        __m256d xc1 = _mm256_load_pd(xc),		xc2 = _mm256_load_pd(xc + 4);
        __m256d jc1 = _mm256_load_pd(jc),       jc2 = _mm256_load_pd(jc + 4);
        __m256d jc3 = _mm256_load_pd(jc + 8),   jc4 = _mm256_load_pd(jc + 12);
        __m256d ds1 = _mm256_add_pd(_mm256_mul_pd(jc1, xc1), _mm256_mul_pd(jc2, xc2));
        __m256d ds2 = _mm256_add_pd(_mm256_mul_pd(jc3, xc1), _mm256_mul_pd(jc4, xc2));
        jx[0] = sse_sum(ds1)  + (jp[0] * xp[0] + jp[1] * xp[1] + jp[2] * xp[2]);
        jx[1] = sse_sum(ds2) + (jp[POINT_ALIGN] * xp[0] + jp[POINT_ALIGN+1] * xp[1] + jp[POINT_ALIGN+2] * xp[2]);
    }

    //v += ax
    inline void   AddScaledVec8(float a, const float* x, float* v)
    {
        __m256 aa = sse_load1(&a);
        _mm256_store_ps(v, _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(x), aa), _mm256_load_ps(v)));
        //_mm256_store_ps(v, _mm256_fmadd_ps(_mm256_load_ps(x), aa, _mm256_load_ps(v)));
    }
    //v += ax
    inline void   AddScaledVec8(double a, const double* x, double* v)
    {
       __m256d aa = sse_load1(&a);
        _mm256_store_pd(v  , _mm256_add_pd( _mm256_mul_pd(_mm256_load_pd(x    ), aa), _mm256_load_pd(v  )));
        _mm256_store_pd(v+4, _mm256_add_pd( _mm256_mul_pd(_mm256_load_pd(x + 4), aa), _mm256_load_pd(v+4)));
    }

    inline void AddBlockJtJ(const float * jc, float * block, int vn)
    {
        __m256 j = _mm256_load_ps(jc);
        for(int i = 0; i < vn; ++i, ++jc, block += 8)
        {
            __m256 a = sse_load1(jc);
            _mm256_store_ps(block, _mm256_add_ps(_mm256_mul_ps(a, j), _mm256_load_ps(block)));
        }
    }

    inline void AddBlockJtJ(const double * jc, double * block, int vn)
    {
        __m256d j1 = _mm256_load_pd(jc);
        __m256d j2 = _mm256_load_pd(jc + 4);
        for(int i = 0; i < vn; ++i, ++jc, block += 8)
        {
            __m256d a = sse_load1(jc);
            _mm256_store_pd(block + 0, _mm256_add_pd(_mm256_mul_pd(a, j1), _mm256_load_pd(block + 0)));
            _mm256_store_pd(block + 4, _mm256_add_pd(_mm256_mul_pd(a, j2), _mm256_load_pd(block + 4)));
        }
    }
};

#endif

#ifdef CPUPBA_USE_NEON
#define CPUPBA_USE_SIMD
#define SIMD_NO_SQRT
#define SIMD_NO_DOUBLE
namespace MYNEON
{
    template<class Float> class SSE{};
    template<>  class SSE<float>  { public: typedef  float32x4_t sse_type;   };

    ////////////////////////////////////////////
    template <class Float> inline size_t sse_step()    {return 16 / sizeof(Float); };
    inline float32x4_t sse_load1(const float* p)     {return vld1q_dup_f32(p); }
    inline float32x4_t sse_load(const float* p)      {return vld1q_f32(p);}
    inline float32x4_t sse_loadzero(){float z = 0; return sse_load1(&z); }
    inline float32x4_t sse_add(float32x4_t s1, float32x4_t s2)     {return vaddq_f32(s1, s2);}
    inline float32x4_t sse_sub(float32x4_t s1, float32x4_t s2)     {return vsubq_f32(s1, s2);}
    inline float32x4_t sse_mul(float32x4_t s1, float32x4_t s2)     {return vmulq_f32(s1, s2);}
    //inline float32x4_t sse_sqrt(float32x4_t s)                {return _mm_sqrt_ps(s); }
    inline float    sse_sum(float32x4_t s)    {float *f = (float*) (&s); return (f[0] + f[2]) + (f[1] + f[3]);}
    inline void     sse_store(float *p, float32x4_t s){vst1q_f32(p, s); }
};
namespace ProgramCPU
{
    using namespace MYNEON;
    #define SSE_ZERO sse_loadzero()
    #define SSE_T typename SSE<Float>::sse_type
    /////////////////////////////
    inline void   ScaleJ4(float* jcx, float* jcy, const float* sj)
    {
         float32x4_t ps = sse_load(sj);
        sse_store(jcx, sse_mul(sse_load(jcx), ps));
        sse_store(jcy, sse_mul(sse_load(jcy), ps));
    }
    inline void   ScaleJ8(float* jcx, float* jcy, const float* sj)
    {
        ScaleJ4(jcx, jcy, sj);
        ScaleJ4(jcx + 4, jcy + 4, sj + 4);
    }

    inline float   DotProduct8(const float* v1, const float* v2)
    {
        float32x4_t ds = sse_add(
            sse_mul(sse_load(v1),     sse_load(v2)),
            sse_mul(sse_load(v1 + 4), sse_load(v2 + 4)));
        return sse_sum(ds);
    }

    inline void  ComputeTwoJX(const float* jc, const float* jp, const float* xc, const float* xp, float* jx)
    {
#ifdef POINT_DATA_ALIGN4
        float32x4_t xc1 = sse_load(xc), xc2 = sse_load(xc + 4), mxp = sse_load(xp);
        float32x4_t ds1 = sse_add(sse_mul(sse_load(jc),  xc1), sse_mul(sse_load(jc + 4), xc2));
        float32x4_t dx1 = sse_add(ds1, sse_mul(sse_load(jp), mxp));
        jx[0] = sse_sum(dx1);
        float32x4_t ds2 = sse_add(sse_mul(sse_load(jc + 8), xc1), sse_mul(sse_load(jc + 12), xc2));
        float32x4_t dx2 = sse_add(ds2, sse_mul(sse_load(jp + 4), mxp));
        jx[1] = sse_sum(dx2);
#else
        float32x4_t xc1 = sse_load(xc),		xc2 = sse_load(xc + 4);
        float32x4_t jc1 = sse_load(jc),       jc2 = sse_load(jc + 4);
        float32x4_t jc3 = sse_load(jc + 8),   jc4 = sse_load(jc + 12);
        float32x4_t ds1 = sse_add(sse_mul(jc1, xc1), sse_mul(jc2, xc2));
        float32x4_t ds2 = sse_add(sse_mul(jc3, xc1), sse_mul(jc4, xc2));
        jx[0] = sse_sum(ds1) + (jp[0] * xp[0] + jp[1] * xp[1] + jp[2] * xp[2]);
        jx[1] = sse_sum(ds2) + (jp[POINT_ALIGN] * xp[0] + jp[POINT_ALIGN+1] * xp[1] + jp[POINT_ALIGN+2] * xp[2]);
        /*jx[0] = (sse_dot(jc1, xc1) + sse_dot(jc2, xc2)) + (jp[0] * xp[0] + jp[1] * xp[1] + jp[2] * xp[2]);
        jx[1] = (sse_dot(jc3, xc1) + sse_dot(jc4, xc2)) + (jp[POINT_ALIGN] * xp[0] + jp[POINT_ALIGN+1] * xp[1] + jp[POINT_ALIGN+2] * xp[2]);*/
#endif
    }

    //v += ax
    inline void   AddScaledVec8(float a, const float* x, float* v)
    {
        float32x4_t aa = sse_load1(&a);
        sse_store(v  , sse_add( sse_mul(sse_load(x    ), aa), sse_load(v  )));
        sse_store(v+4, sse_add( sse_mul(sse_load(x + 4), aa), sse_load(v+4)));
    }

    inline void AddBlockJtJ(const float * jc, float * block, int vn)
    {
        float32x4_t j1 = sse_load(jc);
        float32x4_t j2 = sse_load(jc + 4);
        for(int i = 0; i < vn; ++i, ++jc, block += 8)
        {
            float32x4_t a = sse_load1(jc);
            sse_store(block + 0, sse_add(sse_mul(a, j1), sse_load(block + 0)));
            sse_store(block + 4, sse_add(sse_mul(a, j2), sse_load(block + 4)));
        }
    }
};
#endif
