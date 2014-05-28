namespace ProgramCPU
{
    int		__num_cpu_cores = 0;
    template <class Float>   double ComputeVectorNorm(const avec<Float>& vec, int mt = 0);

#if defined(CPUPBA_USE_SIMD)
    template <class Float>
    void ComputeSQRT(avec<Float>& vec)
    {
#ifndef SIMD_NO_SQRT
        const size_t step =sse_step<Float>();
        Float * p = &vec[0], * pe = p + vec.size(), *pex = pe - step;
        for(; p <= pex; p += step)   sse_store(p, sse_sqrt(sse_load(p)));
        for(; p < pe; ++p) p[0] = sqrt(p[0]);
#else
        for(Float* it = vec.begin(); it < vec.end(); ++it)   *it  = sqrt(*it);
#endif
    }

    template <class Float>
    void ComputeRSQRT(avec<Float>& vec)
    {
        Float * p = &vec[0], * pe = p + vec.size();
        for(; p < pe; ++p) p[0] = (p[0] == 0? 0 : Float(1.0) / p[0]);
        ComputeSQRT(vec);
    }

    template<class Float>
    void SetVectorZero(Float* p, Float * pe)
    {
         SSE_T sse = SSE_ZERO;
         const size_t step =sse_step<Float>();
         Float * pex = pe - step;
         for(; p <= pex; p += step) sse_store(p, sse);
         for(; p < pe; ++p) *p = 0;
    }

    template<class Float>
    void SetVectorZero(avec<Float>& vec)
    {
         Float * p = &vec[0], * pe = p + vec.size();
         SetVectorZero(p, pe);
    }

    //function not used
    template<class Float>
    inline void MemoryCopyA(const Float* p, const Float* pe, Float* d)
    {
        const size_t step = sse_step<Float>();
        const Float* pex = pe - step;
        for(; p <= pex; p += step, d += step) sse_store(d, sse_load(p));
        //while(p < pe) *d++ = *p++;
    }

    template <class Float>
    void ComputeVectorNorm(const Float* p, const Float* pe, double* psum)
    {
         SSE_T sse = SSE_ZERO;
         const size_t step =sse_step<Float>();
         const Float * pex = pe - step;
         for(; p <= pex; p += step)
         {
             SSE_T ps = sse_load(p);
             sse = sse_add(sse, sse_mul(ps, ps));
         }
         double sum = sse_sum(sse);
         for(; p < pe; ++p) sum += p[0] * p[0];
        *psum = sum;
    }

    template<class Float>
    double ComputeVectorNormW(const avec<Float>& vec, const avec<Float>& weight)
    {
        if(weight.begin() != NULL)
        {
             SSE_T sse = SSE_ZERO;
             const size_t step =sse_step<Float>();
             const Float * p = vec, * pe = p + vec.size(), *pex = pe - step;
             const Float * w = weight;
             for(; p <= pex; p += step, w+= step)
             {
                 SSE_T pw = sse_load(w), ps = sse_load(p);
                 sse = sse_add(sse, sse_mul(sse_mul(ps, pw), ps));
             }
             double sum = sse_sum(sse);
             for(; p < pe; ++p, ++w) sum += p[0] * w[0] * p[0];
             return sum;
        }else
        {
            return ComputeVectorNorm<Float>(vec, 0);
        }
    }

    template<class Float>
    double ComputeVectorDot(const avec<Float>& vec1, const avec<Float>& vec2)
    {
         SSE_T sse = SSE_ZERO;
         const size_t step =sse_step<Float>();
         const Float * p1 = vec1, * pe = p1 + vec1.size(), *pex = pe - step;
         const Float * p2 = vec2;
         for(; p1 <= pex; p1 += step, p2+= step)
         {
             SSE_T ps1 = sse_load(p1), ps2 = sse_load(p2);
             sse = sse_add(sse, sse_mul(ps1, ps2));
         }
         double sum = sse_sum(sse);
         for(; p1 < pe; ++p1, ++p2) sum += p1[0]* p2[0];
         return sum;
    }

    template<class Float>
    void   ComputeVXY(const avec<Float>& vec1, const avec<Float>& vec2, avec<Float>& result, size_t part = 0, size_t skip = 0)
    {
        const size_t step =sse_step<Float>();
        const Float * p1 = vec1 + skip, * pe = p1 + (part ? part : vec1.size()), * pex = pe - step;
        const Float * p2 = vec2 + skip;
        Float * p3 = result + skip;
        for(; p1 <= pex; p1 += step, p2 += step, p3 += step)
        {
            SSE_T  ps1 = sse_load(p1), ps2 = sse_load(p2);
            sse_store(p3, sse_mul(ps1, ps2));
        }
        for(; p1 < pe; ++p1, ++p2, ++p3) *p3 = p1[0] * p2[0];
    }

    template <class Float>
    void   ComputeSAXPY(Float a, const Float* p1, const Float* p2, Float* p3, Float* pe)
    {
        const size_t step =sse_step<Float>();
        SSE_T aa = sse_load1(&a);
        Float *pex = pe - step;
        if(a == 1.0f)
        {
            for(; p3 <= pex; p1 += step, p2 += step, p3 += step)
            {
                SSE_T ps1 = sse_load(p1), ps2 = sse_load(p2);
                sse_store(p3,sse_add(ps2, ps1));
            }
        }else if(a == -1.0f)
        {
            for(; p3 <= pex; p1 += step, p2 += step, p3 += step)
            {
                SSE_T ps1 = sse_load(p1), ps2 = sse_load(p2);
                sse_store(p3,sse_sub(ps2, ps1));
            }
        }else
        {
            for(; p3 <= pex; p1 += step, p2 += step, p3 += step)
            {
                SSE_T ps1 = sse_load(p1), ps2 = sse_load(p2);
                sse_store(p3,sse_add(ps2, sse_mul(ps1, aa)));
            }
        }
        for(; p3 < pe; ++p1, ++p2, ++p3) p3[0] = a * p1[0] + p2[0];
    }

    template<class Float>
    void   ComputeSAX(Float a, const avec<Float>& vec1, avec<Float>& result)
    {
        const size_t step = sse_step<Float>();
        SSE_T aa = sse_load1(&a);
        const Float * p1 = vec1, *pe = p1 + vec1.size(), *pex = pe - step;
        Float * p3 = result;
        for(; p1 <= pex; p1 += step, p3 += step)
        {
            sse_store(p3, sse_mul(sse_load(p1), aa));
        }
        for(; p1 < pe; ++p1,  ++p3) p3[0] = a * p1[0];
    }

    template<class Float>
    inline void   ComputeSXYPZ(Float a, const Float* p1, const Float* p2, const Float* p3, Float* p4, Float* pe)
    {
        const size_t step =sse_step<Float>();
        SSE_T aa = sse_load1(&a);
        Float *pex = pe - step;
        for(; p4 <= pex; p1 += step, p2 += step, p3 += step, p4 += step)
        {
            SSE_T ps1 = sse_load(p1), ps2 = sse_load(p2), ps3 = sse_load(p3);
            sse_store(p4,sse_add(ps3, sse_mul(sse_mul(ps1, aa), ps2)));
        }
        for(; p4 < pe; ++p1, ++p2, ++p3, ++ p4) p4[0] = a * p1[0] * p2[0] + p3[0];
    }

#else
    template <class Float>
    void ComputeSQRT(avec<Float>& vec)
    {
        Float* it = vec.begin();
        for(; it < vec.end(); ++it)
        {
            *it  = std::sqrt(*it);
        }
    }
    template <class Float>
    void ComputeRSQRT(avec<Float>& vec)
    {
        Float* it = vec.begin();
        for(; it < vec.end(); ++it)
        {
            *it  = (*it == 0 ? 0 : Float(1.0) / std::sqrt(*it));
        }
    }
    template <class Float>
    inline void SetVectorZero(Float* p,Float* pe)  { std::fill(p, pe, 0);                     }
    template <class Float>
    inline void SetVectorZero(avec<Float>& vec)    { std::fill(vec.begin(), vec.end(), 0);    }

    template <class Float>
    double ComputeVectorNormW(const avec<Float>& vec, const avec<Float>& weight)
    {
        double sum = 0;
        const Float*  it1 = vec.begin(), * it2 = weight.begin();
        for(; it1 < vec.end(); ++it1, ++it2)
        {
            sum += (*it1) * (*it2) * (*it1);
        }
        return sum;
    }

    template <class Float>
    double ComputeVectorDot(const avec<Float>& vec1, const avec<Float>& vec2)
    {
        double sum = 0;
        const Float*   it1 = vec1.begin(), *it2 = vec2.begin();
        for(; it1 < vec1.end(); ++it1, ++it2)
        {
            sum += (*it1) * (*it2);
        }
        return sum;
    }
    template <class Float>
    void ComputeVectorNorm(const Float* p, const Float* pe, double* psum)
    {
        double sum = 0;
        for(; p < pe; ++p)  sum += (*p) * (*p);
        *psum = sum;
    }
    template <class Float>
    inline void   ComputeVXY(const avec<Float>& vec1, const avec<Float>& vec2, avec<Float>& result, size_t part =0, size_t skip = 0)
    {
        const Float*  it1 = vec1.begin() + skip, *it2 = vec2.begin() + skip;
        const Float*  ite = part ? (it1 + part) : vec1.end();
        Float* it3 = result.begin() + skip;
        for(; it1 < ite; ++it1, ++it2, ++it3)
        {
             (*it3) = (*it1) * (*it2);
        }
    }
    template <class Float>
    void   ScaleJ8(Float* jcx, Float* jcy, const Float* sj)
    {
        for(int i = 0; i < 8; ++i) {jcx[i] *= sj[i]; jcy[i] *= sj[i]; }
    }

    template <class Float>
    inline void AddScaledVec8(Float a, const Float* x, Float* v)
    {
        for(int i = 0; i < 8; ++i) v[i] += (a * x[i]);
    }


    template <class Float>
    void   ComputeSAX(Float a, const avec<Float>& vec1, avec<Float>& result)
    {
        const Float*  it1 = vec1.begin();
        Float* it3 = result.begin();
        for(;  it1 < vec1.end(); ++it1,  ++it3)
        {
             (*it3) = (a * (*it1));
        }
    }

    template <class Float>
    inline void   ComputeSXYPZ(Float a, const Float* p1, const Float* p2, const Float* p3, Float* p4, Float* pe)
    {
        for(; p4 < pe; ++p1, ++p2, ++p3, ++p4) *p4 = (a * (*p1) * (*p2) + (*p3));
    }

    template <class Float>
    void   ComputeSAXPY(Float a, const Float* it1, const Float* it2, Float* it3, Float* ite)
    {
        if(a == (Float)1.0)
        {
            for( ; it3 < ite; ++it1, ++it2, ++it3)
            {
                 (*it3) = ((*it1) + (*it2));
            }
        }else
        {
            for( ; it3 < ite; ++it1, ++it2, ++it3)
            {
                 (*it3) = (a * (*it1) + (*it2));
            }
        }
    }
    template<class Float>
    void AddBlockJtJ(const Float * jc, Float * block, int vn)
    {
        for(int i = 0; i < vn; ++i)
        {
            Float* row = block + i * 8,  a = jc[i];
            for(int j = 0; j < vn; ++j) row[j] += a * jc[j];
        }
    }
#endif

#ifdef _WIN32
#define DEFINE_THREAD_DATA(X)       template<class Float> struct X##_STRUCT {
#define DECLEAR_THREAD_DATA(X, ...) X##_STRUCT <Float>  tdata = { __VA_ARGS__ }; \
                                    X##_STRUCT <Float>* newdata = new X##_STRUCT <Float>(tdata)
#define BEGIN_THREAD_PROC(X)        }; template<class Float> DWORD X##_PROC(X##_STRUCT <Float> * q) {
#define END_THREAD_RPOC(X)          delete q; return 0;}

#if defined(WINAPI_FAMILY) && WINAPI_FAMILY==WINAPI_FAMILY_APP
#define MYTHREAD std::thread
#define RUN_THREAD(X, t, ...)       DECLEAR_THREAD_DATA(X, __VA_ARGS__);\
                                    t = std::thread(X##_PROC <Float>, newdata)
#define WAIT_THREAD(tv, n)    {     for(size_t i = 0; i < size_t(n); ++i) tv[i].join(); }
#else
#define MYTHREAD HANDLE
#define RUN_THREAD(X, t, ...)       DECLEAR_THREAD_DATA(X, __VA_ARGS__);\
                                    t = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)X##_PROC <Float>, newdata, 0, 0)
#define WAIT_THREAD(tv, n)    {     WaitForMultipleObjects((DWORD)n, tv, TRUE, INFINITE); \
                                    for(size_t i = 0; i < size_t(n); ++i) CloseHandle(tv[i]); }
#endif
#else
#define DEFINE_THREAD_DATA(X)       template<class Float> struct X##_STRUCT { int tid;
#define DECLEAR_THREAD_DATA(X, ...) X##_STRUCT <Float>  tdata = {i,  __VA_ARGS__ }; \
                                    X##_STRUCT <Float>* newdata = new X##_STRUCT <Float>(tdata)
#define BEGIN_THREAD_PROC(X)        }; template<class Float> void* X##_PROC(X##_STRUCT <Float> * q){
   //                                 cpu_set_t mask;        CPU_ZERO( &mask ); CPU_SET( q->tid, &mask );
   //                                 if( sched_setaffinity(0, sizeof(mask), &mask ) == -1 )
   //                                     std::cout <<"WARNING: Could not set CPU Affinity, continuing...\n";

#define END_THREAD_RPOC(X)          delete q; return 0;}\
                                    template<class Float> struct X##_FUNCTOR {\
                                    typedef  void* (*func_type) (X##_STRUCT <Float> * );\
                                    static func_type get() {return & (X##_PROC<Float>);}    };
#define MYTHREAD  pthread_t

#define RUN_THREAD(X, t, ...)       DECLEAR_THREAD_DATA(X, __VA_ARGS__);\
                                    pthread_create(&t, NULL, (void* (*)(void*))X##_FUNCTOR <Float> :: get(), newdata)
#define WAIT_THREAD(tv, n)      {   for(size_t i = 0; i < size_t(n); ++i) pthread_join(tv[i], NULL) ;}
#endif

    template <class Float>
    inline Float   DotProduct8(const Float* v1, const Float* v2)
    {
        return  v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2] + v1[3] * v2[3] +
                v1[4] * v2[4] + v1[5] * v2[5] + v1[6] * v2[6] + v1[7] * v2[7];
    }
    template<class Float>
    inline void ComputeTwoJX(const Float* jc, const Float* jp, const Float* xc, const Float* xp, Float* jx)
    {
            jx[0] = DotProduct8(jc, xc)     + (jp[0] * xp[0] + jp[1] * xp[1] + jp[2] * xp[2]);
            jx[1] = DotProduct8(jc + 8, xc) + (jp[3] * xp[0] + jp[4] * xp[1] + jp[5] * xp[2]);
    }
    template <class Float>
    Float  ComputeVectorMax(const avec<Float>& vec)
    {
        Float v = 0;
        const Float* it = vec.begin();
        for(; it < vec.end(); ++it)
        {
            Float vi = (Float)fabs(*it);
            v = std::max(v,  vi);
        }
        return v;
    }

    template<class Float>
    void   ComputeSXYPZ(Float a, const avec<Float>& vec1, const avec<Float>& vec2, const avec<Float>& vec3, avec<Float>& result)
    {
        if(vec1.begin() != NULL)
        {
            const Float * p1 = &vec1[0], * p2 = &vec2[0], * p3 = &vec3[0];
            Float * p4 = &result[0], * pe = p4 + result.size();
            ComputeSXYPZ(a, p1, p2, p3, p4, pe);

        }else
        {
            //ComputeSAXPY<Float>(a, vec2, vec3, result, 0);
            ComputeSAXPY<Float>(a, vec2.begin(), vec3.begin(), result.begin(), result.end());
        }

    }



    DEFINE_THREAD_DATA(ComputeSAXPY)
           Float a; const Float * p1, * p2; Float* p3, * pe;
    BEGIN_THREAD_PROC(ComputeSAXPY)
        ComputeSAXPY(q->a, q->p1, q->p2, q->p3, q-> pe);
    END_THREAD_RPOC(ComputeSAXPY)

    template <class Float>
    void   ComputeSAXPY(Float a, const avec<Float>& vec1, const avec<Float>& vec2, avec<Float>& result, int mt = 0)
    {
        const bool auto_multi_thread = true;
        if(auto_multi_thread && mt == 0) {  mt = AUTO_MT_NUM( result.size() * 2);  }
        if(mt > 1 && result.size() >= static_cast<std::size_t>(mt * 4))
        {
            MYTHREAD threads[THREAD_NUM_MAX];
            const int thread_num = std::min(mt, THREAD_NUM_MAX);
            const Float* p1 = vec1.begin(), * p2 = vec2.begin();
            Float* p3 = result.begin();
            for (int i = 0; i < thread_num; ++i)
            {
                size_t first = (result.size() * i / thread_num + FLOAT_ALIGN - 1 ) / FLOAT_ALIGN  * FLOAT_ALIGN ;
                size_t last_ = (result.size() * (i + 1) / thread_num + FLOAT_ALIGN - 1) / FLOAT_ALIGN * FLOAT_ALIGN;
                size_t last  = std::min(last_, result.size());
                RUN_THREAD(ComputeSAXPY, threads[i], a, p1 + first, p2 + first, p3 + first, p3 + last);
            }
            WAIT_THREAD(threads, thread_num);
        }else
        {
            ComputeSAXPY(a, vec1.begin(), vec2.begin(), result.begin(), result.end());
        }
    }

    DEFINE_THREAD_DATA(ComputeVectorNorm)
          const Float * p, * pe; double* sum;
    BEGIN_THREAD_PROC(ComputeVectorNorm)
        ComputeVectorNorm(q->p, q->pe, q-> sum);
    END_THREAD_RPOC(ComputeVectorNorm)

    template <class Float>
    double ComputeVectorNorm(const avec<Float>& vec, int mt)
    {
        const bool auto_multi_thread = true;
        if(auto_multi_thread && mt == 0) {  mt = AUTO_MT_NUM(vec.size());  }
        if(mt > 1 && vec.size() >= static_cast<std::size_t>(mt * 4))
        {
            MYTHREAD threads[THREAD_NUM_MAX];
            double sumv[THREAD_NUM_MAX];
            const int thread_num = std::min(mt, THREAD_NUM_MAX);
            const Float * p = vec;
            for (int i = 0; i < thread_num; ++i)
            {
                size_t first = (vec.size() * i / thread_num + FLOAT_ALIGN - 1 ) / FLOAT_ALIGN  * FLOAT_ALIGN ;
                size_t last_ = (vec.size() * (i + 1) / thread_num + FLOAT_ALIGN - 1) / FLOAT_ALIGN * FLOAT_ALIGN;
                size_t last  = std::min(last_, vec.size());
                RUN_THREAD(ComputeVectorNorm, threads[i], p + first,  p + last, sumv + i);
            }
            WAIT_THREAD(threads, thread_num);
            double sum = 0;
            for(int i = 0; i < thread_num; ++i)
                sum += sumv[i];
            return sum;
        }else
        {
            double sum;
            ComputeVectorNorm(vec.begin(), vec.end(), &sum);
            return sum;
        }
    }

    template <class Float>     void UncompressRodriguesRotation(const Float r[3], Float m[])
    {
        double a = sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
        double ct = a==0.0?0.5f:(1.0f-cos(a))/a/a;
        double st = a==0.0?1:sin(a)/a;
        m[0]=Float(1.0 - (r[1]*r[1] + r[2]*r[2])*ct);
        m[1]=Float(r[0]*r[1]*ct - r[2]*st);
        m[2]=Float(r[2]*r[0]*ct + r[1]*st);
        m[3]=Float(r[0]*r[1]*ct + r[2]*st);
        m[4]=Float(1.0f - (r[2]*r[2] + r[0]*r[0])*ct);
        m[5]=Float(r[1]*r[2]*ct - r[0]*st);
        m[6]=Float(r[2]*r[0]*ct - r[1]*st);
        m[7]=Float(r[1]*r[2]*ct + r[0]*st);
        m[8]=Float(1.0 - (r[0]*r[0] + r[1]*r[1])*ct );
    }
    template<class Float>
    void UpdateCamera(int ncam, const avec<Float>& camera, const avec<Float>& delta, avec<Float>& new_camera)
    {
        const Float * c = &camera[0], * d = &delta[0];
        Float * nc = &new_camera[0], m[9];
        //f[1], t[3], r[3][3], d[1]
        for(int i = 0; i < ncam; ++i, c += 16, d += 8, nc += 16)
        {
            nc[0]  = std::max(c[0] + d[0], ((Float)1e-10));
            nc[1]  = c[1] + d[1];
            nc[2]  = c[2] + d[2];
            nc[3]  = c[3] + d[3];
            nc[13] = c[13] + d[7];

            ////////////////////////////////////////////////////
            UncompressRodriguesRotation(d + 4, m);
            nc[4 ] = m[0] * c[4+0] + m[1] * c[4+3] + m[2] * c[4+6];
            nc[5 ] = m[0] * c[4+1] + m[1] * c[4+4] + m[2] * c[4+7];
            nc[6 ] = m[0] * c[4+2] + m[1] * c[4+5] + m[2] * c[4+8];
            nc[7 ] = m[3] * c[4+0] + m[4] * c[4+3] + m[5] * c[4+6];
            nc[8 ] = m[3] * c[4+1] + m[4] * c[4+4] + m[5] * c[4+7];
            nc[9 ] = m[3] * c[4+2] + m[4] * c[4+5] + m[5] * c[4+8];
            nc[10] = m[6] * c[4+0] + m[7] * c[4+3] + m[8] * c[4+6];
            nc[11] = m[6] * c[4+1] + m[7] * c[4+4] + m[8] * c[4+7];
            nc[12] = m[6] * c[4+2] + m[7] * c[4+5] + m[8] * c[4+8];

            //Float temp[3];
            //GetRodriguesRotation((Float (*)[3])  (nc + 4), temp);
            //UncompressRodriguesRotation(temp, nc + 4);
            nc[14] = c[14];
            nc[15] = c[15];
        }
    }

    template <class Float>
    void  UpdateCameraPoint(int ncam, const avec<Float>& camera, const avec<Float>& point, avec<Float>& delta,
                            avec<Float>& new_camera, avec<Float>& new_point, BundleModeT mode, int mt)
    {
        ////////////////////////////
        if(mode != BUNDLE_ONLY_STRUCTURE)
        {
            UpdateCamera(ncam, camera, delta, new_camera);
        }
        /////////////////////////////
        if(mode != BUNDLE_ONLY_MOTION)
        {
            avec<Float> dp; dp.set(delta.begin() + 8 * ncam, point.size());
            ComputeSAXPY((Float) 1.0, dp, point, new_point, mt);
        }
    }

    template <class Float>
    void ComputeProjection(size_t nproj, const Float* camera, const Float* point, const Float* ms,
                           const int * jmap, Float* pj, int radial, int mt)
    {
#pragma omp parallel for // if(mt != 1) num_threads(mt)
        for(size_t i = 0; i < nproj; ++i)
        {
            const int *jmap_ = jmap + 2*i;
            const Float *ms_ = ms + 2*i;
            Float *pj_ = pj + 2*i;

            const Float* c = camera + jmap_[0] * 16;
            const Float* m = point + jmap_[1] * POINT_ALIGN;
            /////////////////////////////////////////////////////
            Float p0 = c[4 ]*m[0]+c[5 ]*m[1]+c[6 ]*m[2] + c[1];
            Float p1 = c[7 ]*m[0]+c[8 ]*m[1]+c[9 ]*m[2] + c[2];
            Float p2 = c[10]*m[0]+c[11]*m[1]+c[12]*m[2] + c[3];

            if(radial == 1)
            {
                Float rr = Float(1.0)  + c[13] * (p0 * p0 + p1 * p1) / (p2 * p2);
                Float f_p2 = c[0] * rr / p2;
                pj_[0] = ms_[0] - p0 * f_p2;
                pj_[1] = ms_[1] - p1 * f_p2;
            }else if(radial == -1)
            {
                Float f_p2 = c[0] / p2;
                Float  rd = Float(1.0) + c[13] * (ms_[0] * ms_[0] + ms_[1] * ms_[1]) ;
                pj_[0] = ms_[0] * rd  - p0 * f_p2;
                pj_[1] = ms_[1] * rd  - p1 * f_p2;
            }else
            {
                pj_[0] = ms_[0] - p0 * c[0] / p2;
                pj_[1] = ms_[1] - p1 * c[0] / p2;
            }
        }
    }

    template <class Float>
    void ComputeProjectionX(size_t nproj, const Float* camera, const Float* point, const Float* ms,
                           const int * jmap, Float* pj, int radial, int mt)
    {
#pragma omp parallel for // if(mt != 1) num_threads(mt)
        for(size_t i = 0; i < nproj; ++i)
        {
            const int *jmap_ = jmap + 2*i;
            const Float *ms_ = ms + 2*i;
            Float *pj_ = pj + 2*i;

            const Float* c = camera + jmap_[0] * 16;
            const Float* m = point + jmap_[1] * POINT_ALIGN;
            /////////////////////////////////////////////////////
            Float p0 = c[4 ]*m[0]+c[5 ]*m[1]+c[6 ]*m[2] + c[1];
            Float p1 = c[7 ]*m[0]+c[8 ]*m[1]+c[9 ]*m[2] + c[2];
            Float p2 = c[10]*m[0]+c[11]*m[1]+c[12]*m[2] + c[3];
            if(radial == 1)
            {
                Float rr = Float(1.0)  + c[13] * (p0 * p0 + p1 * p1) / (p2 * p2);
                Float f_p2 = c[0] / p2;
                pj_[0] = ms_[0] / rr - p0 * f_p2;
                pj_[1] = ms_[1] / rr - p1 * f_p2;
            }else if(radial == -1)
            {
                Float  rd = Float(1.0) + c[13] * (ms_[0] * ms_[0] + ms_[1] * ms_[1]) ;
                Float f_p2 = c[0] / p2 / rd;
                pj_[0] = ms_[0]  - p0 * f_p2;
                pj_[1] = ms_[1]  - p1 * f_p2;
            }else
            {
                pj_[0] = ms_[0] - p0 * c[0] / p2;
                pj_[1] = ms_[1] - p1 * c[0] / p2;
            }
        }
    }

    template <class Float>
    void ComputeProjectionQ(size_t nq, const Float* camera,const int * qmap,  const Float* wq, Float* pj)
    {
        for(size_t i = 0;i < nq; ++i, qmap += 2, pj += 2, wq += 2)
        {
            const Float* c1 = camera + qmap[0] * 16;
            const Float* c2 = camera + qmap[1] * 16;
            pj[0] = - (c1[ 0] - c2[ 0]) * wq[0];
            pj[1] = - (c1[13] - c2[13]) * wq[1];
        }
    }

    template<class Float>
    void ComputeJQX( size_t nq, const Float* x,  const int* qmap, const Float* wq,const Float* sj, Float* jx)
    {
        if(sj)
        {
            for(size_t i = 0;i < nq; ++i, qmap += 2, jx += 2, wq += 2)
            {
                int idx1 = qmap[0] * 8, idx2 = qmap[1] * 8;
                const Float* x1 = x + idx1;
                const Float* x2 = x + idx2;
                const Float* sj1 = sj + idx1;
                const Float* sj2 = sj + idx2;
                jx[0] = (x1[0] * sj1[0] - x2[0] * sj2[0]) * wq[0];
                jx[1] = (x1[7] * sj1[7] - x2[7] * sj2[7]) * wq[1];
            }
        }else
        {
            for(size_t i = 0;i < nq; ++i, qmap += 2, jx += 2, wq += 2)
            {
                const Float* x1 = x + qmap[0] * 8;
                const Float* x2 = x + qmap[1] * 8;
                jx[0] = (x1[0] - x2[0]) * wq[0];
                jx[1] = (x1[7] - x2[7]) * wq[1];
            }
        }
    }

    template<class Float>
    void ComputeJQtEC(size_t ncam, const Float* pe, const int* qlist, const Float* wq, const Float* sj, Float* v)
    {
        if(sj)
        {
            for(size_t i = 0; i < ncam; ++i, qlist += 2, wq += 2, v+= 8, sj += 8)
            {
                int ip  = qlist[0];
                if(ip == -1)continue;
                int in = qlist[1];
                const Float * e1 = pe + ip * 2;
                const Float * e2 = pe + in * 2;
                v[0] += wq[0] * sj[0] * (e1[0] - e2[0]);
                v[7] += wq[1] * sj[7] * (e1[1] - e2[1]);
            }
        }else
        {
            for(size_t i = 0; i < ncam; ++i, qlist += 2, wq += 2, v+= 8)
            {
                int ip  = qlist[0];
                if(ip == -1)continue;
                int in = qlist[1];
                const Float * e1 = pe + ip * 2;
                const Float * e2 = pe + in * 2;
                v[0] += wq[0] * (e1[0] - e2[0]);
                v[7] += wq[1] * (e1[1] - e2[1]);
            }
        }
    }


    template<class Float>
    inline void JacobianOne(const Float* c, const Float * pt, const Float* ms, Float* jxc, Float* jyc,
                   Float* jxp, Float* jyp, bool intrinsic_fixed , int radial_distortion)
    {
        const Float* r = c + 4;
        Float x0 = c[4 ]*pt[0]+c[5 ]*pt[1]+c[6 ]*pt[2] ;
        Float y0 = c[7 ]*pt[0]+c[8 ]*pt[1]+c[9 ]*pt[2];
        Float z0 = c[10]*pt[0]+c[11]*pt[1]+c[12]*pt[2];
        Float p2 = ( z0 + c[3]);
        Float f_p2  = c[0] / p2;
        Float p0_p2 = (x0 + c[1]) / p2;
        Float p1_p2 = (y0 + c[2]) / p2;

        if(radial_distortion == 1)
        {
            Float rr1 = c[13] * p0_p2 * p0_p2;
            Float rr2 = c[13] * p1_p2 * p1_p2;
            Float f_p2_x = Float(f_p2 * (1.0 + 3.0 * rr1 + rr2));
            Float f_p2_y = Float(f_p2 * (1.0 + 3.0 * rr2 + rr1));
            if(jxc)
            {
#ifndef PBA_DISABLE_CONST_CAMERA
                if(c[15] != 0.0f)
                {
                    jxc[0] = 0;	jxc[1] = 0;	jxc[2] = 0;	jxc[3] = 0;
                    jxc[4] = 0;	jxc[5] = 0;	jxc[6] = 0;	jxc[7] = 0;
                    jyc[0] = 0;	jyc[1] = 0;	jyc[2] = 0;	jyc[3] = 0;
                    jyc[4] = 0;	jyc[5] = 0;	jyc[6] = 0;	jyc[7] = 0;
                }else
#endif
                {
                    Float jfc = intrinsic_fixed ? 0: Float(1.0 + rr1 + rr2);
                    Float ft_x_pn = intrinsic_fixed ? 0 : c[0] * (p0_p2 * p0_p2 + p1_p2 * p1_p2);
                    /////////////////////////////////////////////////////
                    jxc[0] = p0_p2 * jfc;
                    jxc[1] = f_p2_x;
                    jxc[2] = 0;
                    jxc[3] = -f_p2_x * p0_p2;
                    jxc[4] = -f_p2_x * p0_p2 * y0;
                    jxc[5] =  f_p2_x * (z0 + x0 * p0_p2);
                    jxc[6] = -f_p2_x * y0;
                    jxc[7] = ft_x_pn * p0_p2;

                    jyc[0] = p1_p2 * jfc;
                    jyc[1] = 0;
                    jyc[2] = f_p2_y;
                    jyc[3] = -f_p2_y * p1_p2;
                    jyc[4] = -f_p2_y * (z0 + y0 * p1_p2);
                    jyc[5] = f_p2_y * x0 * p1_p2;
                    jyc[6] = f_p2_y * x0;
                    jyc[7] = ft_x_pn * p1_p2;
                }
            }

            ///////////////////////////////////
            if(jxp)
            {
                jxp[0] = f_p2_x * (r[0]- r[6] * p0_p2);
                jxp[1] = f_p2_x * (r[1]- r[7] * p0_p2);
                jxp[2] = f_p2_x * (r[2]- r[8] * p0_p2);
                jyp[0] = f_p2_y * (r[3]- r[6] * p1_p2);
                jyp[1] = f_p2_y * (r[4]- r[7] * p1_p2);
                jyp[2] = f_p2_y * (r[5]- r[8] * p1_p2);
#ifdef POINT_DATA_ALIGN4
                jxp[3] = jyp[3] = 0;
#endif
            }
        }else
        {
            if(jxc)
            {
#ifndef PBA_DISABLE_CONST_CAMERA
                if(c[15] != 0.0f)
                {
                    jxc[0] = 0;	jxc[1] = 0;	jxc[2] = 0;	jxc[3] = 0;
                    jxc[4] = 0;	jxc[5] = 0;	jxc[6] = 0;	jxc[7] = 0;
                    jyc[0] = 0;	jyc[1] = 0;	jyc[2] = 0;	jyc[3] = 0;
                    jyc[4] = 0;	jyc[5] = 0;	jyc[6] = 0;	jyc[7] = 0;
                }else
#endif
                {
                    jxc[0] = intrinsic_fixed? 0 : p0_p2;
                    jxc[1] = f_p2;
                    jxc[2] = 0;
                    jxc[3] = -f_p2 * p0_p2;
                    jxc[4] = -f_p2 * p0_p2 * y0;
                    jxc[5] =  f_p2 * (z0 + x0 * p0_p2);
                    jxc[6] = -f_p2 * y0;

                    jyc[0] = intrinsic_fixed? 0 : p1_p2;
                    jyc[1] = 0;
                    jyc[2] = f_p2;
                    jyc[3] = -f_p2 * p1_p2;
                    jyc[4] = -f_p2 * (z0 + y0 * p1_p2);
                    jyc[5] = f_p2 * x0 * p1_p2;
                    jyc[6] = f_p2 * x0;

                    if(radial_distortion == -1 && !intrinsic_fixed)
                    {
                        Float  msn = ms[0] * ms[0] + ms[1] * ms[1];
                        jxc[7] = -ms[0] * msn;
                        jyc[7] = -ms[1] * msn;
                    }else
                    {
                        jxc[7] = 0;
                        jyc[7] = 0;
                    }
                }
            }
            ///////////////////////////////////
            if(jxp)
            {
                jxp[0] = f_p2 * (r[0]- r[6] * p0_p2);
                jxp[1] = f_p2 * (r[1]- r[7] * p0_p2);
                jxp[2] = f_p2 * (r[2]- r[8] * p0_p2);
                jyp[0] = f_p2 * (r[3]- r[6] * p1_p2);
                jyp[1] = f_p2 * (r[4]- r[7] * p1_p2);
                jyp[2] = f_p2 * (r[5]- r[8] * p1_p2);
#ifdef POINT_DATA_ALIGN4
                jxp[3] = jyp[3] = 0;
#endif
            }
        }
    }

    // Forward declare
    template <class Float>
    void ComputeJacobian(size_t nproj, size_t ncam, const Float* camera, const Float* point, Float*  jc, Float* jp,
                         const int* jmap, const Float * sj, const Float *  ms, const int * cmlist,
                         bool intrinsic_fixed , int radial_distortion, bool shuffle, Float* jct,
                         int mt = 2, int i0 = 0);

    DEFINE_THREAD_DATA(ComputeJacobian)
            size_t nproj, ncam; const Float* camera, *point; Float * jc, *jp;
            const int *jmap; const Float* sj, * ms; const int* cmlist;
            bool intrinsic_fixed; int radial_distortion; bool shuffle; Float* jct; int i0;
    BEGIN_THREAD_PROC(ComputeJacobian)
        ComputeJacobian( q->nproj, q->ncam, q->camera, q->point, q->jc, q->jp,
                    q->jmap, q->sj,  q->ms, q->cmlist, q->intrinsic_fixed,
                    q->radial_distortion, q->shuffle, q->jct, 0, q->i0);
    END_THREAD_RPOC(ComputeJacobian)

    template <class Float>
    void ComputeJacobian(size_t nproj, size_t ncam, const Float* camera, const Float* point, Float*  jc, Float* jp,
                         const int* jmap, const Float * sj, const Float *  ms, const int * cmlist,
                         bool intrinsic_fixed , int radial_distortion, bool shuffle, Float* jct,
                         int mt = 2, int i0 = 0)
    {

        if(mt > 1 && nproj >= static_cast<std::size_t>(mt))
        {
            MYTHREAD threads[THREAD_NUM_MAX];
            const int thread_num = std::min(mt, THREAD_NUM_MAX);
            for(int i = 0; i < thread_num; ++i)
            {
                int first = nproj * i / thread_num;
                int last_ = nproj * (i + 1) / thread_num;
                int last  = std::min(last_, (int)nproj);
                RUN_THREAD(ComputeJacobian, threads[i],
                    last, ncam, camera, point, jc, jp, jmap + 2 * first, sj, ms + 2 * first, cmlist + first,
                    intrinsic_fixed, radial_distortion, shuffle, jct, first);
            }
            WAIT_THREAD(threads, thread_num);
        }else
        {
            const Float* sjc0 = sj;
            const Float* sjp0 = sj ?  sj + ncam * 8 : NULL;

            for(size_t i = i0; i < nproj; ++i, jmap += 2, ms += 2, ++cmlist)
            {
                int cidx = jmap[0], pidx = jmap[1];
                const Float* c = camera + cidx * 16, * pt = point + pidx * POINT_ALIGN;
                Float* jci = jc ? (jc + (shuffle? cmlist[0] : i)* 16)  : NULL;
                Float* jpi = jp ? (jp + i * POINT_ALIGN2) : NULL;

                /////////////////////////////////////////////////////
                JacobianOne(c, pt, ms, jci, jci + 8, jpi, jpi + POINT_ALIGN, intrinsic_fixed, radial_distortion);

                ///////////////////////////////////////////////////
                if(sjc0)
                {
                    //jacobian scaling
                    if(jci)
                    {
                        ScaleJ8(jci, jci + 8, sjc0 + cidx * 8);
                    }
                    if(jpi)
                    {
                        const Float* sjp = sjp0 + pidx * POINT_ALIGN;
                        for(int j = 0; j < 3; ++j) {jpi[j] *= sjp[j]; jpi[POINT_ALIGN + j] *= sjp[j]; }
                    }
                }

                if(jct && jc)    std::copy(jci, jci + 16, jct + cmlist[0] * 16);
            }
        }
    }

    template <class Float>
    void ComputeDiagonalAddQ(size_t ncam, const Float* qw, Float* d, const Float* sj = NULL)
    {
        if(sj)
        {
            for(size_t i = 0; i < ncam; ++i, qw += 2, d += 8, sj += 8)
            {
                if(qw[0] == 0) continue;
                Float j1 = qw[0] * sj[0];
                Float j2 = qw[1] * sj[7];
                d[0] += (j1 * j1 * 2.0f);
                d[7] += (j2 * j2 * 2.0f);
            }
        }else
        {
            for(size_t i = 0; i < ncam; ++i, qw += 2, d += 8)
            {
                if(qw[0] == 0) continue;
                d[0] += (qw[0] * qw[0] * 2.0f);
                d[7] += (qw[1] * qw[1] * 2.0f);
            }
        }
    }

    ///////////////////////////////////////
    template <class Float>
    void  ComputeDiagonal( const avec<Float>& jcv, const std::vector<int>& cmapv, const avec<Float>& jpv, const std::vector<int>& pmapv,
                        const std::vector<int>& cmlistv, const Float* qw0, avec<Float>& jtjdi, bool jc_transpose, int radial)
    {
        //first camera part
        if(jcv.size() == 0 || jpv.size() == 0) return; // not gonna happen


        size_t ncam = cmapv.size() - 1, npts = pmapv.size() - 1;
        const int vn = radial? 8 : 7;
        SetVectorZero(jtjdi);

        const int* cmap = &cmapv[0];
        const int * pmap = &pmapv[0];
        const int * cmlist = &cmlistv[0];
        const Float* jc = &jcv[0];
        const Float* jp = &jpv[0];
        const Float* qw = qw0;
        Float* jji = &jtjdi[0];

        ///////compute jc part
        for(size_t i = 0; i < ncam; ++i, jji += 8, ++cmap, qw += 2)
        {
            int idx1 = cmap[0], idx2 = cmap[1];
            //////////////////////////////////////
            for(int j = idx1; j < idx2; ++j)
            {
                int idx = jc_transpose? j : cmlist[j];
                const Float* pj = jc + idx * 16;
                ///////////////////////////////////////////
                for(int k = 0; k < vn; ++k) jji[k] += (pj[k] * pj[k] + pj[k + 8] * pj[k + 8]);
            }
            if(qw0 && qw[0] > 0)
            {
                jji[0] += (qw[0] * qw[0] * 2.0f);
                jji[7] += (qw[1] * qw[1] * 2.0f);
            }
        }

        for(size_t i = 0; i < npts; ++i, jji += POINT_ALIGN, ++ pmap)
        {
            int idx1 = pmap[0], idx2 = pmap[1];
            const Float* pj = jp + idx1 * POINT_ALIGN2;
            for(int j = idx1; j < idx2; ++j, pj += POINT_ALIGN2)
            {
                for(int k = 0; k < 3; ++k) jji[k] += (pj[k] * pj[k] + pj[k + POINT_ALIGN] * pj[k + POINT_ALIGN]);
            }
        }
        Float* it = jtjdi.begin();
        for(; it < jtjdi.end(); ++it)
        {
            *it = (*it == 0) ? 0 : Float(1.0 / (*it));
        }
    }


    template <class T, int n, int m> void InvertSymmetricMatrix(T a[n][m], T ai[n][m])
    {
       for(int i = 0; i < n; ++i)
       {
           if(a[i][i] > 0)
           {
               a[i][i] = sqrt(a[i][i]);
               for(int j = i + 1; j < n; ++j)
                   a[j][i] = a[j][i] / a[i][i];
               for(int j = i + 1; j < n; ++j)
                   for(int k = j; k < n; ++k)
                       a[k][j] -= a[k][i] * a[j][i];
           }
       }
       /////////////////////////////
       // inv(L)
       for(int i = 0; i < n; ++i)
       {
           if(a[i][i] == 0) continue;
           a[i][i] = 1.0f / a[i][i];
       }
       for(int i = 1; i < n; ++i)
       {
           if(a[i][i] == 0) continue;
           for(int j = 0; j < i; ++j)
           {
               T sum  = 0;
               for(int  k = j; k < i; ++k)    sum += (a[i][k] * a[k][j]);
               a[i][j] = - sum * a[i][i];
           }
       }
       /////////////////////////////
       // inv(L)'  * inv(L)
       for(int i = 0; i < n; ++i)
       {
           for(int j = i; j < n; ++ j)
           {
               ai[i][j] = 0;
               for(int k  = j; k < n; ++k) ai[i][j] += a[k][i] * a[k][j];
               ai[j][i] = ai[i][j];
           }
       }
    }
    template <class T, int n, int m> void InvertSymmetricMatrix(T * a, T * ai)
    {
        InvertSymmetricMatrix<T, n, m>( (T (*)[m]) a, (T (*)[m]) ai);
    }

    // Forward declare.
    template<class Float>
    void ComputeDiagonalBlockC(size_t ncam,  float lambda1, float lambda2, const Float* jc, const int* cmap,
                const int* cmlist, Float* di, Float* bi, int vn, bool jc_transpose, bool use_jq, int mt);

    DEFINE_THREAD_DATA(ComputeDiagonalBlockC)
        size_t ncam; float lambda1, lambda2; const Float * jc; const int* cmap,* cmlist;
        Float * di, * bi; int vn; bool jc_transpose, use_jq;
    BEGIN_THREAD_PROC(ComputeDiagonalBlockC)
        ComputeDiagonalBlockC( q->ncam, q->lambda1, q->lambda2, q->jc, q->cmap,
        q->cmlist, q->di, q->bi, q->vn, q->jc_transpose, q->use_jq, 0);
    END_THREAD_RPOC(ComputeDiagonalBlockC)

    template<class Float>
    void ComputeDiagonalBlockC(size_t ncam,  float lambda1, float lambda2, const Float* jc, const int* cmap,
                const int* cmlist, Float* di, Float* bi, int vn, bool jc_transpose, bool use_jq, int mt)
    {
        const size_t bc = vn * 8;

        if(mt > 1 && ncam >= (size_t) mt)
        {
            MYTHREAD threads[THREAD_NUM_MAX];
            const int thread_num = std::min(mt, THREAD_NUM_MAX);
            for(int i = 0; i < thread_num; ++i)
            {
                int first = ncam * i / thread_num;
                int last_ = ncam * (i + 1) / thread_num;
                int last  = std::min(last_, (int)ncam);
                RUN_THREAD(ComputeDiagonalBlockC, threads[i],
                    (last - first), lambda1, lambda2, jc, cmap + first,
                     cmlist, di + 8 * first, bi + bc * first, vn, jc_transpose, use_jq);
            }
            WAIT_THREAD(threads, thread_num);
        }else
        {
            Float bufv[64 + 8]; //size_t offset = ((size_t)bufv) & 0xf;
            //Float* pbuf = bufv + ((16 - offset) / sizeof(Float));
            Float* pbuf = (Float*)ALIGN_PTR(bufv);


            ///////compute jc part
            for(size_t i = 0; i < ncam; ++i, ++cmap, bi += bc)
            {
                int idx1 = cmap[0], idx2 = cmap[1];
                //////////////////////////////////////
                if(idx1 == idx2)
                {
                    SetVectorZero(bi, bi + bc);
                }else
                {
                    SetVectorZero(pbuf, pbuf + 64);

                    for(int j = idx1; j < idx2; ++j)
                    {
                        int idx = jc_transpose? j : cmlist[j];
                        const Float* pj = jc + idx * 16;
                        /////////////////////////////////
                        AddBlockJtJ(pj,     pbuf, vn);
                        AddBlockJtJ(pj + 8, pbuf, vn);
                    }

                    //change and copy the diagonal

                    if(use_jq)
                    {
                        Float* pb = pbuf;
                        for(int j= 0; j < 8; ++j, ++di, pb += 9)
                        {
                            Float temp;
                            di[0]  = temp = (di[0] + pb[0]);
                            pb[0] = lambda2 * temp + lambda1;
                        }
                    }else
                    {
                        Float* pb = pbuf;
                        for(int j= 0; j < 8; ++j, ++di, pb += 9)
                        {
                            *pb = lambda2 * ((* di) = (*pb)) + lambda1;
                        }
                    }

                    //invert the matrix?
                    if(vn==8)   InvertSymmetricMatrix<Float, 8, 8>(pbuf, bi);
                    else        InvertSymmetricMatrix<Float, 7, 8>(pbuf, bi);
                }
            }
        }
    }

    // Forward declare.
    template<class Float>
    void ComputeDiagonalBlockP(size_t npt, float lambda1, float lambda2,
                        const Float*  jp, const int* pmap, Float* di, Float* bi, int mt);

    DEFINE_THREAD_DATA(ComputeDiagonalBlockP)
        size_t npt; float lambda1, lambda2;  const Float * jp; const int* pmap; Float* di, *bi;
    BEGIN_THREAD_PROC(ComputeDiagonalBlockP)
        ComputeDiagonalBlockP( q->npt, q->lambda1, q->lambda2, q->jp, q->pmap, q->di, q->bi, 0);
    END_THREAD_RPOC(ComputeDiagonalBlockP)

    template<class Float>
    void ComputeDiagonalBlockP(size_t npt, float lambda1, float lambda2,
                        const Float*  jp, const int* pmap, Float* di, Float* bi, int mt)
    {
        if(mt > 1)
        {
            MYTHREAD threads[THREAD_NUM_MAX];
            const int thread_num = std::min(mt, THREAD_NUM_MAX);
            for(int i = 0; i < thread_num; ++i)
            {
                int first = npt * i / thread_num;
                int last_ = npt * (i + 1) / thread_num;
                int last  = std::min(last_, (int)npt);
                RUN_THREAD(ComputeDiagonalBlockP, threads[i],
                    (last - first), lambda1, lambda2, jp, pmap + first,
                    di + POINT_ALIGN * first, bi + 6 * first);
            }
            WAIT_THREAD(threads, thread_num);
        }else
        {
            for(size_t i = 0; i < npt; ++i, ++pmap, di += POINT_ALIGN, bi += 6)
            {
                int idx1 = pmap[0], idx2 = pmap[1];

                Float M00 = 0, M01= 0, M02 = 0, M11 = 0, M12 = 0, M22 = 0;
                const Float* jxp = jp + idx1 * (POINT_ALIGN2), * jyp = jxp + POINT_ALIGN;
                for(int j = idx1; j < idx2; ++j, jxp += POINT_ALIGN2, jyp += POINT_ALIGN2)
                {
                    M00 += (jxp[0] * jxp[0] + jyp[0] * jyp[0]);
                    M01 += (jxp[0] * jxp[1] + jyp[0] * jyp[1]);
                    M02 += (jxp[0] * jxp[2] + jyp[0] * jyp[2]);
                    M11 += (jxp[1] * jxp[1] + jyp[1] * jyp[1]);
                    M12 += (jxp[1] * jxp[2] + jyp[1] * jyp[2]);
                    M22 += (jxp[2] * jxp[2] + jyp[2] * jyp[2]);
                }

                /////////////////////////////////
                di[0] = M00;    di[1] = M11;    di[2] = M22;

                /////////////////////////////
                M00 = M00 * lambda2 + lambda1;
                M11 = M11 * lambda2 + lambda1;
                M22 = M22 * lambda2 + lambda1;

                ///////////////////////////////
                Float det = (M00 * M11 - M01 * M01) * M22 + Float(2.0) * M01 * M12 * M02 - M02 * M02 * M11 - M12 * M12 * M00;
                if(det >= FLT_MAX || det <= FLT_MIN * 2.0f)
                {
                    //SetVectorZero(bi, bi + 6);
                    for(int j = 0; j < 6; ++j) bi[j] = 0;
                }else
                {
                    bi[0] =  ( M11 * M22 - M12 * M12) / det;
                    bi[1] = -( M01 * M22 - M12 * M02) / det;
                    bi[2] =  ( M01 * M12 - M02 * M11) / det;
                    bi[3] =  ( M00 * M22 - M02 * M02) / det;
                    bi[4] = -( M00 * M12 - M01 * M02) / det;
                    bi[5] =  ( M00 * M11 - M01 * M01) / det;
                }
            }
        }
    }

    template<class Float>
    void ComputeDiagonalBlock(size_t ncam, size_t npts, float lambda, bool dampd, const Float* jc, const int* cmap,
                const Float*  jp, const int* pmap, const int* cmlist,
                const Float*  sj, const Float* wq, Float* diag, Float* blocks,
                int radial_distortion, bool jc_transpose, int mt1 = 2, int mt2 = 2, BundleModeT mode = BUNDLE_FULL)
    {
        const int    vn = radial_distortion? 8 : 7;
        const size_t bc = vn * 8;
        float lambda1 = dampd? 0.0f : lambda;
        float lambda2 = dampd? (1.0f + lambda) : 1.0f;

        if(mode == BUNDLE_FULL)
        {
            const size_t bsz = bc * ncam + npts * 6;
            const size_t dsz = 8 * ncam + npts * POINT_ALIGN;
            bool  use_jq = wq != NULL;
            ///////////////////////////////////////////
            SetVectorZero(blocks, blocks + bsz);
            SetVectorZero(diag, diag + dsz);

            ////////////////////////////////
            if(use_jq) ComputeDiagonalAddQ(ncam, wq, diag, sj);
            ComputeDiagonalBlockC(ncam, lambda1, lambda2, jc, cmap, cmlist, diag, blocks, vn, jc_transpose, use_jq, mt1);
            ComputeDiagonalBlockP(npts, lambda1, lambda2, jp, pmap, diag + 8 * ncam, blocks + bc * ncam, mt2);
        }else if (mode == BUNDLE_ONLY_MOTION)
        {
            const size_t bsz = bc * ncam;
            const size_t dsz = 8 * ncam;
            bool  use_jq = wq != NULL;
            ///////////////////////////////////////////
            SetVectorZero(blocks, blocks + bsz);
            SetVectorZero(diag, diag + dsz);

            ////////////////////////////////
            if(use_jq) ComputeDiagonalAddQ(ncam, wq, diag, sj);
            ComputeDiagonalBlockC(ncam, lambda1, lambda2, jc, cmap, cmlist, diag, blocks, vn, jc_transpose, use_jq, mt1);
        }else if (mode == BUNDLE_ONLY_STRUCTURE)
        {
            blocks += bc * ncam;
            diag   += 8 * ncam;
            const size_t bsz = npts * 6;
            const size_t dsz = npts * POINT_ALIGN;
            ///////////////////////////////////////////
            SetVectorZero(blocks, blocks + bsz);
            SetVectorZero(diag, diag + dsz);

            ////////////////////////////////
            ComputeDiagonalBlockP(npts, lambda1, lambda2, jp, pmap, diag, blocks, mt2);
        }
    }

    template<class Float>
    void ComputeDiagonalBlock_(float lambda, bool dampd, const avec<Float>& camerav,  const avec<Float>& pointv,
                             const avec<Float>& meas,  const std::vector<int>& jmapv,  const avec<Float>& sjv,
                             avec<Float>&qwv, avec<Float>& diag, avec<Float>& blocks,
                             bool intrinsic_fixed, int radial_distortion, int mode = 0)
    {
        const int vn = radial_distortion? 8 : 7;
        const size_t szbc = vn * 8;
        size_t ncam = camerav.size() / 16;
        size_t npts = pointv.size()/POINT_ALIGN;
        size_t sz_jcd = ncam * 8;
        size_t sz_jcb = ncam * szbc;
        avec<Float> blockpv(blocks.size());
        SetVectorZero(blockpv);
        SetVectorZero(diag);
        //////////////////////////////////////////////////////
        float lambda1 = dampd? 0.0f : lambda;
        float lambda2 = dampd? (1.0f + lambda) : 1.0f;

        Float jbufv[24 + 8]; 	//size_t offset = ((size_t) jbufv) & 0xf;
        //Float* jxc = jbufv + ((16 - offset) / sizeof(Float));
        Float* jxc = (Float*)ALIGN_PTR(jbufv);
        Float* jyc = jxc + 8, *jxp = jxc + 16, *jyp = jxc + 20;

        //////////////////////////////
        const int * jmap = &jmapv[0];
        const Float* camera = &camerav[0];
        const Float* point = &pointv[0];
        const Float* ms = &meas[0];
        const Float* sjc0 = sjv.size() ?  &sjv[0] : NULL;
        const Float* sjp0 = sjv.size() ?  &sjv[sz_jcd] : NULL;
        //////////////////////////////////////////////
        Float* blockpc = &blockpv[0], * blockpp = &blockpv[sz_jcb];
        Float* bo = blockpc, *bi = &blocks[0], *di = &diag[0];

        /////////////////////////////////////////////////////////
        //diagonal blocks
        for(size_t i = 0; i < jmapv.size(); i += 2, jmap += 2, ms += 2)
        {
            int cidx = jmap[0], pidx = jmap[1];
            const Float* c = camera + cidx * 16, * pt = point + pidx * POINT_ALIGN;
            /////////////////////////////////////////////////////////
            JacobianOne(c, pt, ms, jxc, jyc, jxp, jyp, intrinsic_fixed, radial_distortion);

            ///////////////////////////////////////////////////////////
            if(mode != BUNDLE_ONLY_STRUCTURE)
            {
                if(sjc0)
                {
                    const Float* sjc = sjc0 + cidx * 8;
                    ScaleJ8(jxc, jyc, sjc);
                }
                /////////////////////////////////////////
                Float* bc = blockpc + cidx * szbc;
                AddBlockJtJ(jxc, bc, vn);
                AddBlockJtJ(jyc, bc, vn);
            }

            if(mode != BUNDLE_ONLY_MOTION)
            {
                if(sjp0)
                {
                    const Float* sjp = sjp0 + pidx * POINT_ALIGN;
                    jxp[0] *= sjp[0];   jxp[1] *= sjp[1];   jxp[2] *= sjp[2];
                    jyp[0] *= sjp[0];   jyp[1] *= sjp[1];   jyp[2] *= sjp[2];
                }

                ///////////////////////////////////////////
                Float* bp = blockpp  + pidx * 6;
                bp[0] += (jxp[0] * jxp[0] + jyp[0] * jyp[0]);
                bp[1] += (jxp[0] * jxp[1] + jyp[0] * jyp[1]);
                bp[2] += (jxp[0] * jxp[2] + jyp[0] * jyp[2]);
                bp[3] += (jxp[1] * jxp[1] + jyp[1] * jyp[1]);
                bp[4] += (jxp[1] * jxp[2] + jyp[1] * jyp[2]);
                bp[5] += (jxp[2] * jxp[2] + jyp[2] * jyp[2]);
            }
        }

        ///invert the camera part
        if(mode != BUNDLE_ONLY_STRUCTURE)
        {
            /////////////////////////////////////////
            const Float* qw =  qwv.begin();
            if(qw)
            {
                for(size_t i = 0; i < ncam; ++i, qw += 2)
                {
                    if(qw[0] == 0)continue;
                    Float* bc = blockpc + i * szbc;
                    if(sjc0)
                    {
                        const Float* sjc = sjc0 + i * 8;
                        Float j1 = sjc[0] * qw[0];
                        Float j2 = sjc[7] * qw[1];
                        bc[0] += (j1 * j1 * 2.0f);
                        if(radial_distortion) bc[63] += (j2 * j2 * 2.0f);
                    }else
                    {
                        //const Float* sjc = sjc0 + i * 8;
                        bc[0] += (qw[0] * qw[0] * 2.0f);
                        if(radial_distortion) bc[63] += (qw[1] * qw[1] * 2.0f);
                    }
                }
            }


            for(size_t i = 0; i < ncam; ++i, bo += szbc, bi += szbc, di += 8)
            {
                    Float* bp = bo,  *dip = di;
                    for(int  j = 0; j < vn; ++j, ++dip, bp += 9)
                    {
                        dip[0] = bp[0];
                        bp[0] = lambda2 * bp[0] + lambda1;
                    }

                //invert the matrix?
                if(radial_distortion)   InvertSymmetricMatrix<Float, 8, 8>(bo, bi);
                else                    InvertSymmetricMatrix<Float, 7, 8>(bo, bi);
            }
        }else
        {
            bo += szbc * ncam;
            bi += szbc * ncam;
            di += 8 * ncam;
        }


        ///////////////////////////////////////////
        //inverting the point part
        if(mode != BUNDLE_ONLY_MOTION)
        {
            for(size_t i = 0; i < npts; ++i, bo += 6, bi += 6, di += POINT_ALIGN)
            {
                Float& M00 = bo[0], & M01 = bo[1], & M02 = bo[2];
                Float& M11 = bo[3], & M12 = bo[4], & M22 = bo[5];
                di[0] = M00;  	di[1] = M11;  	di[2] = M22;

                /////////////////////////////
                M00 = M00 * lambda2 + lambda1;
                M11 = M11 * lambda2 + lambda1;
                M22 = M22 * lambda2 + lambda1;

                ///////////////////////////////
                Float det = (M00 * M11 - M01 * M01) * M22 + Float(2.0) * M01 * M12 * M02 - M02 * M02 * M11 - M12 * M12 * M00;
                if(det >= FLT_MAX || det <= FLT_MIN * 2.0f)
                {
                    for(int j = 0; j < 6; ++j) bi[j] = 0;
                }else
                {
                    bi[0] =   ( M11 * M22 - M12 * M12) / det;
                    bi[1] =  -( M01 * M22 - M12 * M02) / det;
                    bi[2] =   ( M01 * M12 - M02 * M11) / det;
                    bi[3] =   ( M00 * M22 - M02 * M02) / det;
                    bi[4] =  -( M00 * M12 - M01 * M02) / det;
                    bi[5] =   ( M00 * M11 - M01 * M01) / det;
                }
            }
        }
    }

    template<class Float>
    void MultiplyBlockConditionerC(int ncam, const Float* bi, const Float*  x, Float* vx, int vn, int mt = 0)
    {
#pragma omp parallel for // if(mt != 1) num_threads(mt)
        for(int i = 0; i < ncam; ++i)
            for(int j = 0; j < vn; ++j)
                vx[8*i + j] = DotProduct8(bi + 8 * (i*vn + j), x + 8*i);
    }

    template<class Float>
    void MultiplyBlockConditionerP(int npoint, Float* const bi, Float* const x, Float* vx, int mt = 0)
    {
#pragma omp parallel for // if(mt != 1) num_threads(mt)
        for(int i = 0; i < npoint; ++i)
        {
            // is this dangerous for POINT_ALIGN == 4?
            Float *const bi_ = bi + 6 * i;
            Float *const x_ = x + POINT_ALIGN * i;
            Float *const vx_ = vx + POINT_ALIGN * i;

            vx_[0] = bi_[0] * x_[0] + bi_[1] * x_[1] + bi_[2] * x_[2];
            vx_[1] = bi_[1] * x_[0] + bi_[3] * x_[1] + bi_[4] * x_[2];
            vx_[2] = bi_[2] * x_[0] + bi_[4] * x_[1] + bi_[5] * x_[2];
        }
    }

    template<class Float>
    void MultiplyBlockConditioner(int ncam, int npoint, Float* const blocksv,
                                  Float* const vec, Float* resultv, int radial, BundleModeT mode,  int mt1, int mt2)
    {
        const int vn = radial ? 8 : 7;
        if(mode != BUNDLE_ONLY_STRUCTURE) MultiplyBlockConditionerC(ncam, blocksv, vec, resultv, vn, mt1);
        if(mt2 == 0) mt2 = AUTO_MT_NUM(npoint * 24);
        if(mode != BUNDLE_ONLY_MOTION) MultiplyBlockConditionerP(npoint,  blocksv + (vn*8*ncam), vec + ncam*8, resultv + 8*ncam, mt2);
    }

    ////////////////////////////////////////////////////

    // Forward declare.
    template<class Float>
    void ComputeJX( size_t nproj, size_t ncam,  const Float* x, const Float*  jc,
                    const Float* jp, const int* jmap, Float* jx, BundleModeT mode, int mt = 2);


    DEFINE_THREAD_DATA(ComputeJX)
        size_t nproj, ncam; const Float* xc, *jc,* jp; const int* jmap; Float* jx; BundleModeT mode;
    BEGIN_THREAD_PROC(ComputeJX)
        ComputeJX(q->nproj, q->ncam, q->xc, q->jc, q->jp, q->jmap, q->jx, q->mode, 0);
    END_THREAD_RPOC(ComputeJX)

    template<class Float>
    void ComputeJX( size_t nproj, size_t ncam,  const Float* x, const Float*  jc,
                    const Float* jp, const int* jmap, Float* jx, BundleModeT mode, int mt = 2)
    {
        if(mt > 1 && nproj >= static_cast<std::size_t>(mt))
        {
            MYTHREAD threads[THREAD_NUM_MAX];
            const int thread_num = std::min(mt, THREAD_NUM_MAX);
            for(int i = 0; i < thread_num; ++i)
            {
                size_t first = nproj * i / thread_num;
                size_t last_ = nproj * (i + 1) / thread_num;
                size_t last  = std::min(last_, nproj);
                RUN_THREAD( ComputeJX, threads[i], (last - first), ncam, x,
                            jc + 16 * first, jp + POINT_ALIGN2 * first,
                            jmap + first *2, jx + first* 2, mode);
            }
            WAIT_THREAD(threads, thread_num);
        }else if(mode == BUNDLE_FULL)
        {
            const Float* pxc = x, * pxp = pxc + ncam * 8;
            //clock_t tp = clock(); double s1 = 0, s2  = 0;
            for(size_t i = 0 ;i < nproj; ++i, jmap += 2, jc += 16, jp += POINT_ALIGN2, jx += 2)
            {
                ComputeTwoJX(jc, jp, pxc + jmap[0] * 8, pxp + jmap[1] * POINT_ALIGN, jx);
            }
        }else if(mode == BUNDLE_ONLY_MOTION)
        {
            const Float* pxc = x;
            //clock_t tp = clock(); double s1 = 0, s2  = 0;
            for(size_t i = 0 ;i < nproj; ++i, jmap += 2, jc += 16, jp += POINT_ALIGN2, jx += 2)
            {
                const Float* xc = pxc + jmap[0] * 8;
                jx[0] = DotProduct8(jc, xc)   ;
                jx[1] = DotProduct8(jc + 8, xc);
            }
        }else if(mode == BUNDLE_ONLY_STRUCTURE)
        {
            const Float* pxp = x + ncam * 8;
            //clock_t tp = clock(); double s1 = 0, s2  = 0;
            for(size_t i = 0 ;i < nproj; ++i, jmap += 2, jc += 16, jp += POINT_ALIGN2, jx += 2)
            {
                const Float* xp = pxp + jmap[1] * POINT_ALIGN;
                jx[0] =  (jp[0] * xp[0] + jp[1] * xp[1] + jp[2] * xp[2]);
                jx[1] =  (jp[3] * xp[0] + jp[4] * xp[1] + jp[5] * xp[2]);
            }
        }
    }

    // Forward declare.
    template<class Float>
    void ComputeJX_(size_t nproj, size_t ncam,  const Float* x, Float* jx, const Float* camera,
                    const Float* point,  const Float* ms, const Float* sj, const int*  jmap,
                    bool intrinsic_fixed, int radial_distortion, BundleModeT mode, int mt = 16);

    DEFINE_THREAD_DATA(ComputeJX_)
           size_t nproj, ncam; const Float* x; Float * jx;
            const Float* camera, *point,* ms, *sj; const int *jmap;
            bool intrinsic_fixed; int radial_distortion; BundleModeT mode;
    BEGIN_THREAD_PROC(ComputeJX_)
        ComputeJX_( q->nproj, q->ncam, q->x, q->jx, q->camera, q->point, q->ms, q->sj,
                    q->jmap, q->intrinsic_fixed, q->radial_distortion, q->mode, 0);
    END_THREAD_RPOC(ComputeJX_)

    template<class Float>
    void ComputeJX_(size_t nproj, size_t ncam,  const Float* x, Float* jx, const Float* camera,
                    const Float* point,  const Float* ms, const Float* sj, const int*  jmap,
                    bool intrinsic_fixed, int radial_distortion, BundleModeT mode, int mt = 16)
    {
        if(mt > 1 && nproj >= static_cast<std::size_t>(mt))
        {
            MYTHREAD threads[THREAD_NUM_MAX];
            const int thread_num = std::min(mt, THREAD_NUM_MAX);
            for (int i = 0; i < thread_num; ++i)
            {
                size_t first = nproj * i / thread_num;
                size_t last_ = nproj * (i + 1) / thread_num;
                size_t last  = std::min(last_, nproj);
                RUN_THREAD(ComputeJX_, threads[i],
                    (last - first), ncam, x, jx + first * 2,
                    camera, point, ms + 2 * first, sj, jmap + first * 2,
                    intrinsic_fixed, radial_distortion, mode);
            }
            WAIT_THREAD(threads, thread_num);
        }else if(mode == BUNDLE_FULL)
        {
            Float jcv[24 + 8]; //size_t offset = ((size_t) jcv) & 0xf;
            //Float* jc = jcv + (16 - offset) / sizeof(Float), *jp = jc + 16;
            Float* jc = (Float*)ALIGN_PTR(jcv), *jp = jc + 16;
            ////////////////////////////////////////
            const Float* sjc = sj;
            const Float* sjp = sjc? (sjc + ncam * 8) : NULL;
            const Float* xc0 = x, *xp0 = x + ncam * 8;

            /////////////////////////////////
            for(size_t i = 0 ;i < nproj; ++i, ms += 2, jmap += 2, jx += 2)
            {
                const int cidx = jmap[0], pidx = jmap[1];
                const Float* c = camera + cidx * 16, * pt = point + pidx * POINT_ALIGN;
                /////////////////////////////////////////////////////
                JacobianOne(c, pt, ms, jc, jc + 8, jp, jp + POINT_ALIGN, intrinsic_fixed, radial_distortion);
                if(sjc)
                {
                    //jacobian scaling
                    ScaleJ8(jc, jc + 8, sjc + cidx * 8);
                    const Float* sjpi = sjp + pidx * POINT_ALIGN;
                    for(int j = 0; j < 3; ++j) {jp[j] *= sjpi[j]; jp[POINT_ALIGN + j] *= sjpi[j]; }
                }
                ////////////////////////////////////
                ComputeTwoJX(jc, jp, xc0 + cidx * 8, xp0 + pidx * POINT_ALIGN, jx);
            }
        }else if(mode == BUNDLE_ONLY_MOTION)
        {
            Float jcv[24 + 8]; //size_t offset = ((size_t) jcv) & 0xf;
            //Float* jc = jcv + (16 - offset) / sizeof(Float);
            Float* jc = (Float*)ALIGN_PTR(jcv);

            ////////////////////////////////////////
            const Float* sjc = sj, * xc0 = x;

            /////////////////////////////////
            for(size_t i = 0 ;i < nproj; ++i, ms += 2, jmap += 2, jx += 2)
            {
                const int cidx = jmap[0], pidx = jmap[1];
                const Float* c = camera + cidx * 16, * pt = point + pidx * POINT_ALIGN;
                /////////////////////////////////////////////////////
                JacobianOne(c, pt, ms, jc, jc + 8, (Float*) NULL, (Float*)NULL, intrinsic_fixed, radial_distortion);
                if(sjc)ScaleJ8(jc, jc + 8, sjc + cidx * 8);
                const Float* xc = xc0 + cidx * 8;
                jx[0] = DotProduct8(jc, xc)   ;
                jx[1] = DotProduct8(jc + 8, xc);
            }
        }else if(mode == BUNDLE_ONLY_STRUCTURE)
        {
            Float jp[8];

            ////////////////////////////////////////
            const Float* sjp = sj? (sj + ncam * 8) : NULL;
            const Float* xp0 = x + ncam * 8;

            /////////////////////////////////
            for(size_t i = 0 ;i < nproj; ++i, ms += 2, jmap += 2, jx += 2)
            {
                const int cidx = jmap[0], pidx = jmap[1];
                const Float* c = camera + cidx * 16, * pt = point + pidx * POINT_ALIGN;
                /////////////////////////////////////////////////////
                JacobianOne(c, pt, ms, (Float*) NULL, (Float*) NULL, jp, jp + POINT_ALIGN, intrinsic_fixed, radial_distortion);

                const Float* xp = xp0 + pidx * POINT_ALIGN;
                if(sjp)
                {
                    const Float* s = sjp + pidx * POINT_ALIGN;
                    jx[0] =  (jp[0] * xp[0] * s[0] + jp[1] * xp[1] * s[1] + jp[2] * xp[2] * s[2]);
                    jx[1] =  (jp[3] * xp[0] * s[0] + jp[4] * xp[1] * s[1] + jp[5] * xp[2] * s[2]);
                }else
                {
                    jx[0] =  (jp[0] * xp[0] + jp[1] * xp[1] + jp[2] * xp[2]);
                    jx[1] =  (jp[3] * xp[0] + jp[4] * xp[1] + jp[5] * xp[2]);
                }
            }
        }
    }

    template<class Float>
    void ComputeJtEC(    size_t ncam, const Float* pe, const Float* jc, const int* cmap,
                        const int* cmlist,  Float* v, bool jc_transpose, int mt)
    {
#pragma omp parallel for // if(mt != 1) num_threads(mt)
        for(size_t i = 0; i < ncam; ++i)
        {
            int idx1 = cmap[i], idx2 = cmap[i+1];
            for(int j = idx1; j < idx2; ++j)
            {
                int edx = cmlist[j];
                const Float* pj = jc +  ((jc_transpose? j : edx) * 16);
                const Float* e  = pe + edx * 2;
                //////////////////////////////
                AddScaledVec8(e[0], pj,     v + 8*i);
                AddScaledVec8(e[1], pj + 8, v + 8*i);
            }
        }
    }

    template<class Float>
    void ComputeJtEP(   size_t npt, const Float* pe, const Float* jp,
                        const int* pmap, Float* v,  int mt)
    {
#pragma omp parallel for // if(mt != 1) num_threads(mt)
        for(size_t i = 0; i < npt; ++i)
        {
            int idx1 = pmap[i], idx2 = pmap[i+1];
            const Float* pj = jp + idx1 * POINT_ALIGN2;
            const Float* e  = pe + idx1 * 2;
            Float temp[3] = {0, 0, 0};
            for(int j = idx1; j < idx2; ++j, pj += POINT_ALIGN2, e += 2)
            {
                temp[0] += (e[0] * pj[0] + e[1] * pj[POINT_ALIGN]);
                temp[1] += (e[0] * pj[1] + e[1] * pj[POINT_ALIGN + 1]);
                temp[2] += (e[0] * pj[2] + e[1] * pj[POINT_ALIGN + 2]);
            }
            // Dangerous for POINT_ALIGN == 4?
            v[i*POINT_ALIGN + 0] = temp[0];
            v[i*POINT_ALIGN + 1] = temp[1];
            v[i*POINT_ALIGN + 2] = temp[2];
        }
    }

    template<class Float>
    void ComputeJtE(    size_t ncam, size_t npt, const Float* pe, const Float* jc,
                        const int* cmap, const int* cmlist,  const Float* jp,
                        const int* pmap, Float* v, bool jc_transpose, BundleModeT mode, int mt1, int mt2)
    {
        if(mode != 2)
        {
            SetVectorZero(v, v + ncam * 8 );
            ComputeJtEC(ncam, pe, jc, cmap, cmlist, v, jc_transpose, mt1);
        }
        if(mode != 1)
        {
            ComputeJtEP(npt, pe, jp, pmap, v + 8 * ncam, mt2);
        }
    }

    template<class Float>
    void ComputeJtEC_(  size_t ncam, const Float* ee,  Float* jte,
                        const Float* c, const Float* point, const Float* ms,
                        const int* jmap, const int* cmap, const int * cmlist,
                        bool intrinsic_fixed, int radial_distortion, int mt)
    {
#pragma omp parallel for // if (mt != 1) num_threads(mt)
        for(size_t i = 0; i < ncam; ++i)
        {
            int idx1 = cmap[i], idx2 = cmap[i+1];

            for(int j = idx1; j < idx2; ++j)
            {
                int index = cmlist[j];
                const Float* pt = point + jmap[2 * index + 1] * POINT_ALIGN;
                const Float* e  = ee + index * 2;
                typename SSE<Float>::sse_type jcx_[8*sizeof(Float) / sizeof(typename SSE<Float>::sse_type)], jcy_[8*sizeof(Float) / sizeof(typename SSE<Float>::sse_type)];
                Float *jcx = reinterpret_cast<Float*>(jcx_);
                Float *jcy = reinterpret_cast<Float*>(jcy_);

                JacobianOne(c + 16*i, pt, ms + index * 2, jcx, jcy, (Float*)NULL, (Float*)NULL, intrinsic_fixed, radial_distortion);

                //////////////////////////////
                AddScaledVec8(e[0], jcx, jte + 8*i);
                AddScaledVec8(e[1], jcy, jte + 8*i);
            }
        }
    }

    template<class Float>
    void ComputeJtE_(   size_t /*nproj*/, size_t ncam, size_t npt, const Float* ee,  Float* jte,
                        const Float* camera, const Float* point, const Float* ms, const int* jmap,
                        const int* cmap, const int* cmlist, const int* pmap, const Float* jp,
                        bool intrinsic_fixed, int radial_distortion, BundleModeT mode, int mt)
    {
        if(mode != 2)
        {
            SetVectorZero(jte, jte + ncam * 8 );
            ComputeJtEC_(ncam, ee, jte, camera, point, ms, jmap, cmap, cmlist, intrinsic_fixed, radial_distortion, mt);
        }
        if(mode != 1)
        {
            ComputeJtEP(npt, ee, jp, pmap, jte + 8 * ncam, mt);
        }
    }

    template<class Float>
    void ComputeJtE_(   size_t nproj, size_t ncam, size_t npt, const Float* ee,  Float* jte,
                        const Float* camera, const Float* point, const Float* ms, const int* jmap,
                        bool intrinsic_fixed, int radial_distortion, BundleModeT mode)
    {
        SetVectorZero(jte, jte + (ncam * 8 + npt * POINT_ALIGN));
        Float jcv[24 + 8];  //size_t offset = ((size_t) jcv) & 0xf;
        //Float* jc = jcv + (16 - offset) / sizeof(Float), *pj = jc + 16;
        Float* jc = (Float*)ALIGN_PTR(jcv), *pj = jc + 16;

        Float* vc0 = jte, *vp0 = jte + ncam * 8;

        for(size_t i = 0 ;i < nproj; ++i, jmap += 2, ms += 2, ee += 2)
        {
            int cidx = jmap[0], pidx = jmap[1];
            const Float* c = camera + cidx * 16, * pt = point + pidx * POINT_ALIGN;

            if(mode == BUNDLE_FULL)
            {
                /////////////////////////////////////////////////////
                JacobianOne(c, pt, ms, jc, jc + 8, pj, pj + POINT_ALIGN, intrinsic_fixed, radial_distortion);

                ////////////////////////////////////////////
                Float* vc = vc0 + cidx * 8, *vp = vp0 + pidx * POINT_ALIGN;
                AddScaledVec8(ee[0], jc,     vc);
                AddScaledVec8(ee[1], jc + 8, vc);
                vp[0] += (ee[0] * pj[0] + ee[1] * pj[POINT_ALIGN]);
                vp[1] += (ee[0] * pj[1] + ee[1] * pj[POINT_ALIGN + 1]);
                vp[2] += (ee[0] * pj[2] + ee[1] * pj[POINT_ALIGN + 2]);
            }else if(mode == BUNDLE_ONLY_MOTION)
            {
                /////////////////////////////////////////////////////
                JacobianOne(c, pt, ms, jc, jc + 8, (Float*) NULL, (Float*) NULL, intrinsic_fixed, radial_distortion);

                ////////////////////////////////////////////
                Float* vc = vc0 + cidx * 8;
                AddScaledVec8(ee[0], jc,     vc);
                AddScaledVec8(ee[1], jc + 8, vc);
            }else if(mode == BUNDLE_ONLY_STRUCTURE)
            {
               /////////////////////////////////////////////////////
                JacobianOne(c, pt, ms, (Float*) NULL, (Float*) NULL, pj, pj + POINT_ALIGN, intrinsic_fixed, radial_distortion);

                ////////////////////////////////////////////
                Float *vp = vp0 + pidx * POINT_ALIGN;
                vp[0] += (ee[0] * pj[0] + ee[1] * pj[POINT_ALIGN]);
                vp[1] += (ee[0] * pj[1] + ee[1] * pj[POINT_ALIGN + 1]);
                vp[2] += (ee[0] * pj[2] + ee[1] * pj[POINT_ALIGN + 2]);
            }
        }
    }
}
