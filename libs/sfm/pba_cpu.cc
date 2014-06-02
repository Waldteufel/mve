/*
 * Copyright (c) 2011  Changchang Wu (ccwu@cs.washington.edu)
 *    and the University of Washington at Seattle
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU General Public
 *  License as published by the Free Software Foundation; either
 *  Version 3 of the License, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  General Public License for more details.
 */

#include <cstdlib>
#include <vector>
#include <iostream>
#include <utility>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>

#include <cmath>
#include <ctime>
#include <cfloat>

#include <omp.h>

#include "util/exception.h"
#include "sfm/pba_cpu.h"

#include <immintrin.h>

SFM_NAMESPACE_BEGIN
SFM_PBA_NAMESPACE_BEGIN

#define THREAD_NUM_MAX 64

#define POINT_ALIGN 3
#define POINT_ALIGN2 6

#define ALIGN_PTR(x) (x)
#define AUTO_MT_NUM(x) 0

void avec::SaveToFile(const char* name)
{
    std::ofstream out(name);
    for(double* p = begin(); p < end(); ++p) out << (*p) <<  '\n';
}

namespace ProgramCPU
{
    int __num_cpu_cores = 0;

    void ComputeSQRT(avec& vec)
    {
        double* v = vec.begin();
        std::size_t len = vec.size();
//#pragma omp parallel for
        for (std::size_t i = 0; i < len; ++i)
            v[i] = std::sqrt(v[i]);
    }

    void ComputeRSQRT(avec& vec)
    {
        double* v = vec.begin();
        std::size_t len = vec.size();
//#pragma omp parallel for
        for (std::size_t i = 0; i < len; ++i)
            v[i] = v[i] == 0.0 ? 0.0 : 1/std::sqrt(v[i]);
    }

    inline void SetVectorZero(double* p,double* pe)  { std::fill(p, pe, 0.0);                     }
    inline void SetVectorZero(avec& vec)    { std::fill(vec.begin(), vec.end(), 0.0);    }

    inline void MemoryCopyA(const double* p, const double* pe, double* d)
    {
        std::copy(p, pe, d);
    }

    double ComputeVectorNormW(const avec& vec, const avec& weight)
    {
        double const* v = vec.begin();
        double const* w = weight.begin();
        double sum = 0.0;
        std::size_t len = vec.size();
//#pragma omp parallel for reduction(+:sum)
        for (std::size_t i = 0; i < len; ++i)
            sum += v[i] * w[i] * v[i];
        return sum;
    }

    double ComputeVectorDot(const avec& vec1, const avec& vec2)
    {
        double const* v = vec1.begin();
        double const* w = vec2.begin();
        double sum = 0.0;
        std::size_t len = vec1.size();
//#pragma omp parallel for reduction(+:sum)
        for (std::size_t i = 0; i < len; ++i)
            sum += v[i] * w[i];
        return sum;
    }

    double ComputeVectorNorm(const avec& vec, int mt = 0)
    {
        double const* v = vec.begin();
        double sum = 0.0;
        std::size_t len = vec.size();
//#pragma omp parallel for reduction(+:sum)
        for (std::size_t i = 0; i < len; ++i)
            sum += v[i] * v[i];
        return sum;
    }

    inline void   ComputeVXY(const avec& vec1, const avec& vec2, avec& result, size_t part =0, size_t skip = 0)
    {
        const double*  it1 = vec1.begin() + skip, *it2 = vec2.begin() + skip;
        const double*  ite = part ? (it1 + part) : vec1.end();
        double* it3 = result.begin() + skip;
        size_t n = ite - it1;
//#pragma omp parallel for
        for(std::size_t i = 0; i < n; ++i)
             it3[i] = it1[i] * it2[i];
    }

    void   ScaleJ8(double* jcx, double* jcy, const double* sj)
    {
        for(int i = 0; i < 8; ++i) {jcx[i] *= sj[i]; jcy[i] *= sj[i]; }
    }

    inline void AddScaledVec8(double a, const double* x, double* v)
    {
        for(int i = 0; i < 8; ++i) v[i] += (a * x[i]);
    }

    void   ComputeSAX(double a, const avec& vec1, avec& result)
    {
        const double*  it1 = vec1.begin();
        double* it3 = result.begin();
//#pragma omp parallel for
        for(std::size_t i = 0; i < vec1.size(); ++i)
             it3[i] = a * it1[i];
    }

    inline void   ComputeSXYPZ(double a, const double* p1, const double* p2, const double* p3, double* p4, double* pe)
    {
        std::size_t n = pe - p4;
//#pragma omp parallel for
        for(std::size_t i = 0; i < n; ++i)
             p4[i] = a * p1[i] * p2[i] + p3[i];
    }

    void   ComputeSAXPY(double a, const double* it1, const double* it2, double* it3, double* ite)
    {
        std::size_t n = ite - it3;
//#pragma omp parallel for
        for(std::size_t i = 0; i < n; ++i)
             it3[i] = a * it1[i] + it2[i];
    }

    void AddBlockJtJ(const double * jc, double * block, int vn)
    {
        for(int i = 0; i < vn; ++i)
        {
            double* row = block + i * 8,  a = jc[i];
            for(int j = 0; j < vn; ++j) row[j] += a * jc[j];
        }
    }

    inline void MemoryCopyB(const double* p, const double* pe, double* d)
    {
        std::copy(p, pe, d);
    }

    inline double   DotProduct8(const double* v1, const double* v2)
    {
        return  v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2] + v1[3] * v2[3] +
                v1[4] * v2[4] + v1[5] * v2[5] + v1[6] * v2[6] + v1[7] * v2[7];
    }

    inline void ComputeTwoJX(const double* jc, const double* jp, const double* xc, const double* xp, double* jx)
    {
            jx[0] = DotProduct8(jc, xc)     + (jp[0] * xp[0] + jp[1] * xp[1] + jp[2] * xp[2]);
            jx[1] = DotProduct8(jc + 8, xc) + (jp[3] * xp[0] + jp[4] * xp[1] + jp[5] * xp[2]);
    }

    double  ComputeVectorMax(const avec& vec)
    {
        double v = 0;
        const double* it = vec.begin();
        for(; it < vec.end(); ++it)
        {
            double vi = (double)fabs(*it);
            v = std::max(v,  vi);
        }
        return v;
    }

    void   ComputeSXYPZ(double a, const avec& vec1, const avec& vec2, const avec& vec3, avec& result)
    {
        if(vec1.begin() != NULL)
        {
            const double * p1 = &vec1[0], * p2 = &vec2[0], * p3 = &vec3[0];
            double * p4 = &result[0], * pe = p4 + result.size();
            ComputeSXYPZ(a, p1, p2, p3, p4, pe);

        }else
        {
            //ComputeSAXPY<Float>(a, vec2, vec3, result, 0);
            ComputeSAXPY(a, vec2.begin(), vec3.begin(), result.begin(), result.end());
        }

    }

    void   ComputeSAXPY(double a, const avec& vec1, const avec& vec2, avec& result, int mt = 0)
    {
        ComputeSAXPY(a, vec1.begin(), vec2.begin(), result.begin(), result.end());
    }

    void GetRodriguesRotation(const double m[3][3], double r[3])
    {
        //http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/index.htm
        double a = (m[0][0]+m[1][1]+m[2][2]-1.0)/2.0;
        const double epsilon = 0.01;
        if( fabs(m[0][1] - m[1][0]) < epsilon &&
            fabs(m[1][2] - m[2][1]) < epsilon &&
            fabs(m[0][2] - m[2][0]) < epsilon )
        {
            if( fabs(m[0][1] + m[1][0]) < 0.1 &&
                fabs(m[1][2] + m[2][1]) < 0.1 &&
                fabs(m[0][2] + m[2][0]) < 0.1 && a > 0.9)
            {
                r[0]    =    0;
                r[1]    =    0;
                r[2]    =    0;
            }
            else
            {
                const double ha = double(sqrt(0.5) * PBA_PI);
                double xx = (m[0][0]+1.0)/2.0;
                double yy = (m[1][1]+1.0)/2.0;
                double zz = (m[2][2]+1.0)/2.0;
                double xy = (m[0][1]+m[1][0])/4.0;
                double xz = (m[0][2]+m[2][0])/4.0;
                double yz = (m[1][2]+m[2][1])/4.0;

                if ((xx > yy) && (xx > zz))
                {
                    if (xx< epsilon)
                    {
                        r[0] = 0;    r[1] = r[2] = ha;
                    } else
                    {
                        double t = sqrt(xx) ;
                        r[0] = double(t * PBA_PI);
                        r[1] = double(xy/t * PBA_PI);
                        r[2] = double(xz/t * PBA_PI);
                    }
                } else if (yy > zz)
                {
                    if (yy< epsilon)
                    {
                        r[0] = r[2]  = ha; r[1] = 0;
                    } else
                    {
                        double t = sqrt(yy);
                        r[0] = double(xy/t* PBA_PI);
                        r[1] = double( t * PBA_PI);
                        r[2] = double(yz/t* PBA_PI);
                    }
                } else
                {
                    if (zz< epsilon)
                    {
                        r[0] = r[1] = ha; r[2] = 0;
                    } else
                    {
                        double t  = sqrt(zz);
                        r[0]  = double(xz/ t* PBA_PI);
                        r[1]  = double(yz/ t* PBA_PI);
                        r[2]  = double( t * PBA_PI);
                    }
                }
            }
        }
        else
        {
            a = acos(a);
            double b = 0.5*a/sin(a);
            r[0]    =    double(b*(m[2][1]-m[1][2]));
            r[1]    =    double(b*(m[0][2]-m[2][0]));
            r[2]    =    double(b*(m[1][0]-m[0][1]));
        }
    }

    void UncompressRodriguesRotation(const double r[3], double m[])
    {
        double a = sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
        double ct = a==0.0?0.5f:(1.0f-cos(a))/a/a;
        double st = a==0.0?1:sin(a)/a;
        m[0]=double(1.0 - (r[1]*r[1] + r[2]*r[2])*ct);
        m[1]=double(r[0]*r[1]*ct - r[2]*st);
        m[2]=double(r[2]*r[0]*ct + r[1]*st);
        m[3]=double(r[0]*r[1]*ct + r[2]*st);
        m[4]=double(1.0f - (r[2]*r[2] + r[0]*r[0])*ct);
        m[5]=double(r[1]*r[2]*ct - r[0]*st);
        m[6]=double(r[2]*r[0]*ct - r[1]*st);
        m[7]=double(r[1]*r[2]*ct + r[0]*st);
        m[8]=double(1.0 - (r[0]*r[0] + r[1]*r[1])*ct );
    }

    void UpdateCamera(int ncam, const avec& camera, const avec& delta, avec& new_camera)
    {
        const double * c = &camera[0], * d = &delta[0];
        double * nc = &new_camera[0], m[9];
        //f[1], t[3], r[3][3], d[1]
        for(int i = 0; i < ncam; ++i, c += 16, d += 8, nc += 16)
        {
            nc[0]  = std::max(c[0] + d[0], ((double)1e-10));
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

    void  UpdateCameraPoint(int ncam, const avec& camera, const avec& point, avec& delta,
                            avec& new_camera, avec& new_point, int mode, int mt)
    {
        ////////////////////////////
        if(mode != 2)
        {
            UpdateCamera(ncam, camera, delta, new_camera);
        }
        /////////////////////////////
        if(mode != 1)
        {
            avec dp; dp.set(delta.begin() + 8 * ncam, point.size());
            ComputeSAXPY((double) 1.0, dp, point, new_point, mt);
        }
    }

    void ComputeProjection(size_t nproj, const double* camera, const double* point, const double* ms,
                           const int * jmap, double* pj, int radial, int mt)
    {
        for(size_t i = 0; i < nproj; ++i, jmap += 2, ms += 2, pj += 2)
        {
            const double* c = camera + jmap[0] * 16;
            const double* m = point + jmap[1] * POINT_ALIGN;
            /////////////////////////////////////////////////////
            double p0 = c[4 ]*m[0]+c[5 ]*m[1]+c[6 ]*m[2] + c[1];
            double p1 = c[7 ]*m[0]+c[8 ]*m[1]+c[9 ]*m[2] + c[2];
            double p2 = c[10]*m[0]+c[11]*m[1]+c[12]*m[2] + c[3];

            if(radial == 1)
            {
                double rr = double(1.0)  + c[13] * (p0 * p0 + p1 * p1) / (p2 * p2);
                double f_p2 = c[0] * rr / p2;
                pj[0] = ms[0] - p0 * f_p2;
                pj[1] = ms[1] - p1 * f_p2;
            }else if(radial == -1)
            {
                double f_p2 = c[0] / p2;
                double  rd = double(1.0) + c[13] * (ms[0] * ms[0] + ms[1] * ms[1]) ;
                pj[0] = ms[0] * rd  - p0 * f_p2;
                pj[1] = ms[1] * rd  - p1 * f_p2;
            }else
            {
                pj[0] = ms[0] - p0 * c[0] / p2;
                pj[1] = ms[1] - p1 * c[0] / p2;
            }
        }
    }

    void ComputeProjectionX(size_t nproj, const double* camera, const double* point, const double* ms,
                           const int * jmap, double* pj, int radial, int mt)
    {
        for(size_t i = 0; i < nproj; ++i, jmap += 2, ms += 2, pj += 2)
        {
            const double* c = camera + jmap[0] * 16;
            const double* m = point + jmap[1] * POINT_ALIGN;
            /////////////////////////////////////////////////////
            double p0 = c[4 ]*m[0]+c[5 ]*m[1]+c[6 ]*m[2] + c[1];
            double p1 = c[7 ]*m[0]+c[8 ]*m[1]+c[9 ]*m[2] + c[2];
            double p2 = c[10]*m[0]+c[11]*m[1]+c[12]*m[2] + c[3];
            if(radial == 1)
            {
                double rr = double(1.0)  + c[13] * (p0 * p0 + p1 * p1) / (p2 * p2);
                double f_p2 = c[0] / p2;
                pj[0] = ms[0] / rr - p0 * f_p2;
                pj[1] = ms[1] / rr - p1 * f_p2;
            }else if(radial == -1)
            {
                double  rd = double(1.0) + c[13] * (ms[0] * ms[0] + ms[1] * ms[1]) ;
                double f_p2 = c[0] / p2 / rd;
                pj[0] = ms[0]  - p0 * f_p2;
                pj[1] = ms[1]  - p1 * f_p2;
            }else
            {
                pj[0] = ms[0] - p0 * c[0] / p2;
                pj[1] = ms[1] - p1 * c[0] / p2;
            }
        }
    }

    void ComputeProjectionQ(size_t nq, const double* camera,const int * qmap,  const double* wq, double* pj)
    {
        for(size_t i = 0;i < nq; ++i, qmap += 2, pj += 2, wq += 2)
        {
            const double* c1 = camera + qmap[0] * 16;
            const double* c2 = camera + qmap[1] * 16;
            pj[0] = - (c1[ 0] - c2[ 0]) * wq[0];
            pj[1] = - (c1[13] - c2[13]) * wq[1];
        }
    }

    void ComputeJQX( size_t nq, const double* x,  const int* qmap, const double* wq,const double* sj, double* jx)
    {
        if(sj)
        {
            for(size_t i = 0;i < nq; ++i, qmap += 2, jx += 2, wq += 2)
            {
                int idx1 = qmap[0] * 8, idx2 = qmap[1] * 8;
                const double* x1 = x + idx1;
                const double* x2 = x + idx2;
                const double* sj1 = sj + idx1;
                const double* sj2 = sj + idx2;
                jx[0] = (x1[0] * sj1[0] - x2[0] * sj2[0]) * wq[0];
                jx[1] = (x1[7] * sj1[7] - x2[7] * sj2[7]) * wq[1];
            }
        }else
        {
            for(size_t i = 0;i < nq; ++i, qmap += 2, jx += 2, wq += 2)
            {
                const double* x1 = x + qmap[0] * 8;
                const double* x2 = x + qmap[1] * 8;
                jx[0] = (x1[0] - x2[0]) * wq[0];
                jx[1] = (x1[7] - x2[7]) * wq[1];
            }
        }
    }

    void ComputeJQtEC(size_t ncam, const double* pe, const int* qlist, const double* wq, const double* sj, double* v)
    {
        if(sj)
        {
            for(size_t i = 0; i < ncam; ++i, qlist += 2, wq += 2, v+= 8, sj += 8)
            {
                int ip  = qlist[0];
                if(ip == -1)continue;
                int in = qlist[1];
                const double * e1 = pe + ip * 2;
                const double * e2 = pe + in * 2;
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
                const double * e1 = pe + ip * 2;
                const double * e2 = pe + in * 2;
                v[0] += wq[0] * (e1[0] - e2[0]);
                v[7] += wq[1] * (e1[1] - e2[1]);
            }
        }
    }

    inline void JacobianOne(const double* c, const double * pt, const double* ms, double* jxc, double* jyc,
                   double* jxp, double* jyp, bool intrinsic_fixed , int radial_distortion)
    {
        const double* r = c + 4;
        double x0 = c[4 ]*pt[0]+c[5 ]*pt[1]+c[6 ]*pt[2] ;
        double y0 = c[7 ]*pt[0]+c[8 ]*pt[1]+c[9 ]*pt[2];
        double z0 = c[10]*pt[0]+c[11]*pt[1]+c[12]*pt[2];
        double p2 = ( z0 + c[3]);
        double f_p2  = c[0] / p2;
        double p0_p2 = (x0 + c[1]) / p2;
        double p1_p2 = (y0 + c[2]) / p2;

        if(radial_distortion == 1)
        {
            double rr1 = c[13] * p0_p2 * p0_p2;
            double rr2 = c[13] * p1_p2 * p1_p2;
            double f_p2_x = double(f_p2 * (1.0 + 3.0 * rr1 + rr2));
            double f_p2_y = double(f_p2 * (1.0 + 3.0 * rr2 + rr1));
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
                    double jfc = intrinsic_fixed ? 0: double(1.0 + rr1 + rr2);
                    double ft_x_pn = intrinsic_fixed ? 0 : c[0] * (p0_p2 * p0_p2 + p1_p2 * p1_p2);
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
                        double  msn = ms[0] * ms[0] + ms[1] * ms[1];
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

    void ComputeJacobian(size_t nproj, size_t ncam, const double* camera, const double* point, double*  jc, double* jp,
                         const int* jmap, const double * sj, const double *  ms, const int * cmlist,
                         bool intrinsic_fixed , int radial_distortion, bool shuffle, double* jct,
                         int mt = 2, int i0 = 0)
    {
        const double* sjc0 = sj;
        const double* sjp0 = sj ?  sj + ncam * 8 : NULL;

        for(size_t i = i0; i < nproj; ++i, jmap += 2, ms += 2, ++cmlist)
        {
            int cidx = jmap[0], pidx = jmap[1];
            const double* c = camera + cidx * 16, * pt = point + pidx * POINT_ALIGN;
            double* jci = jc ? (jc + (shuffle? cmlist[0] : i)* 16)  : NULL;
            double* jpi = jp ? (jp + i * POINT_ALIGN2) : NULL;

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
                    const double* sjp = sjp0 + pidx * POINT_ALIGN;
                    for(int j = 0; j < 3; ++j) {jpi[j] *= sjp[j]; jpi[POINT_ALIGN + j] *= sjp[j]; }
                }
            }

            if(jct && jc)    MemoryCopyB(jci, jci + 16, jct + cmlist[0] * 16);
        }
    }

    void ComputeDiagonalAddQ(size_t ncam, const double* qw, double* d, const double* sj = NULL)
    {
        if(sj)
        {
            for(size_t i = 0; i < ncam; ++i, qw += 2, d += 8, sj += 8)
            {
                if(qw[0] == 0) continue;
                double j1 = qw[0] * sj[0];
                double j2 = qw[1] * sj[7];
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
    void  ComputeDiagonal( const avec& jcv, const std::vector<int>& cmapv, const avec& jpv, const std::vector<int>& pmapv,
                        const std::vector<int>& cmlistv, const double* qw0, avec& jtjdi, bool jc_transpose, int radial)
    {
        //first camera part
        if(jcv.size() == 0 || jpv.size() == 0) return; // not gonna happen


        size_t ncam = cmapv.size() - 1, npts = pmapv.size() - 1;
        const int vn = radial? 8 : 7;
        SetVectorZero(jtjdi);

        const int* cmap = &cmapv[0];
        const int * pmap = &pmapv[0];
        const int * cmlist = &cmlistv[0];
        const double* jc = &jcv[0];
        const double* jp = &jpv[0];
        const double* qw = qw0;
        double* jji = &jtjdi[0];

        ///////compute jc part
        for(size_t i = 0; i < ncam; ++i, jji += 8, ++cmap, qw += 2)
        {
            int idx1 = cmap[0], idx2 = cmap[1];
            //////////////////////////////////////
            for(int j = idx1; j < idx2; ++j)
            {
                int idx = jc_transpose? j : cmlist[j];
                const double* pj = jc + idx * 16;
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
            const double* pj = jp + idx1 * POINT_ALIGN2;
            for(int j = idx1; j < idx2; ++j, pj += POINT_ALIGN2)
            {
                for(int k = 0; k < 3; ++k) jji[k] += (pj[k] * pj[k] + pj[k + POINT_ALIGN] * pj[k + POINT_ALIGN]);
            }
        }
        double* it = jtjdi.begin();
        for(; it < jtjdi.end(); ++it)
        {
            *it = (*it == 0) ? 0 : double(1.0 / (*it));
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

    void ComputeDiagonalBlockC(size_t ncam,  float lambda1, float lambda2, const double* jc, const int* cmap,
                const int* cmlist, double* di, double* bi, int vn, bool jc_transpose, bool use_jq, int mt)
    {
        const size_t bc = vn * 8;

        double bufv[64 + 8]; //size_t offset = ((size_t)bufv) & 0xf;
        //Float* pbuf = bufv + ((16 - offset) / sizeof(Float));
        double* pbuf = (double*)ALIGN_PTR(bufv);


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
                    const double* pj = jc + idx * 16;
                    /////////////////////////////////
                    AddBlockJtJ(pj,     pbuf, vn);
                    AddBlockJtJ(pj + 8, pbuf, vn);
                }

                //change and copy the diagonal

                if(use_jq)
                {
                    double* pb = pbuf;
                    for(int j= 0; j < 8; ++j, ++di, pb += 9)
                    {
                        double temp;
                        di[0]  = temp = (di[0] + pb[0]);
                        pb[0] = lambda2 * temp + lambda1;
                    }
                }else
                {
                    double* pb = pbuf;
                    for(int j= 0; j < 8; ++j, ++di, pb += 9)
                    {
                        *pb = lambda2 * ((* di) = (*pb)) + lambda1;
                    }
                }

                //invert the matrix?
                if(vn==8)   InvertSymmetricMatrix<double, 8, 8>(pbuf, bi);
                else        InvertSymmetricMatrix<double, 7, 8>(pbuf, bi);
            }
        }
    }

    void ComputeDiagonalBlockP(size_t npt, float lambda1, float lambda2,
                        const double*  jp, const int* pmap, double* di, double* bi, int mt)
    {
        for(size_t i = 0; i < npt; ++i, ++pmap, di += POINT_ALIGN, bi += 6)
        {
            int idx1 = pmap[0], idx2 = pmap[1];

            double M00 = 0, M01= 0, M02 = 0, M11 = 0, M12 = 0, M22 = 0;
            const double* jxp = jp + idx1 * (POINT_ALIGN2), * jyp = jxp + POINT_ALIGN;
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
            double det = (M00 * M11 - M01 * M01) * M22 + double(2.0) * M01 * M12 * M02 - M02 * M02 * M11 - M12 * M12 * M00;
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

    void ComputeDiagonalBlock(size_t ncam, size_t npts, float lambda, bool dampd, const double* jc, const int* cmap,
                const double*  jp, const int* pmap, const int* cmlist,
                const double*  sj, const double* wq, double* diag, double* blocks,
                int radial_distortion, bool jc_transpose, int mt1 = 2, int mt2 = 2, int mode = 0)
    {
        const int    vn = radial_distortion? 8 : 7;
        const size_t bc = vn * 8;
        float lambda1 = dampd? 0.0f : lambda;
        float lambda2 = dampd? (1.0f + lambda) : 1.0f;

        if(mode == 0)
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
        }else if (mode == 1)
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
        }else
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

    void ComputeDiagonalBlock_(float lambda, bool dampd, const avec& camerav,  const avec& pointv,
                             const avec& meas,  const std::vector<int>& jmapv,  const avec& sjv,
                             avec&qwv, avec& diag, avec& blocks,
                             bool intrinsic_fixed, int radial_distortion, int mode = 0)
    {
        const int vn = radial_distortion? 8 : 7;
        const size_t szbc = vn * 8;
        size_t ncam = camerav.size() / 16;
        size_t npts = pointv.size()/POINT_ALIGN;
        size_t sz_jcd = ncam * 8;
        size_t sz_jcb = ncam * szbc;
        avec blockpv(blocks.size());
        SetVectorZero(blockpv);
        SetVectorZero(diag);
        //////////////////////////////////////////////////////
        float lambda1 = dampd? 0.0f : lambda;
        float lambda2 = dampd? (1.0f + lambda) : 1.0f;

        double jbufv[24 + 8]; 	//size_t offset = ((size_t) jbufv) & 0xf;
        //Float* jxc = jbufv + ((16 - offset) / sizeof(Float));
        double* jxc = (double*)ALIGN_PTR(jbufv);
        double* jyc = jxc + 8, *jxp = jxc + 16, *jyp = jxc + 20;

        //////////////////////////////
        const int * jmap = &jmapv[0];
        const double* camera = &camerav[0];
        const double* point = &pointv[0];
        const double* ms = &meas[0];
        const double* sjc0 = sjv.size() ?  &sjv[0] : NULL;
        const double* sjp0 = sjv.size() ?  &sjv[sz_jcd] : NULL;
        //////////////////////////////////////////////
        double* blockpc = &blockpv[0], * blockpp = &blockpv[sz_jcb];
        double* bo = blockpc, *bi = &blocks[0], *di = &diag[0];

        /////////////////////////////////////////////////////////
        //diagonal blocks
        for(size_t i = 0; i < jmapv.size(); i += 2, jmap += 2, ms += 2)
        {
            int cidx = jmap[0], pidx = jmap[1];
            const double* c = camera + cidx * 16, * pt = point + pidx * POINT_ALIGN;
            /////////////////////////////////////////////////////////
            JacobianOne(c, pt, ms, jxc, jyc, jxp, jyp, intrinsic_fixed, radial_distortion);

            ///////////////////////////////////////////////////////////
            if(mode != 2)
            {
                if(sjc0)
                {
                    const double* sjc = sjc0 + cidx * 8;
                    ScaleJ8(jxc, jyc, sjc);
                }
                /////////////////////////////////////////
                double* bc = blockpc + cidx * szbc;
                AddBlockJtJ(jxc, bc, vn);
                AddBlockJtJ(jyc, bc, vn);
            }

            if(mode != 1)
            {
                if(sjp0)
                {
                    const double* sjp = sjp0 + pidx * POINT_ALIGN;
                    jxp[0] *= sjp[0];   jxp[1] *= sjp[1];   jxp[2] *= sjp[2];
                    jyp[0] *= sjp[0];   jyp[1] *= sjp[1];   jyp[2] *= sjp[2];
                }

                ///////////////////////////////////////////
                double* bp = blockpp  + pidx * 6;
                bp[0] += (jxp[0] * jxp[0] + jyp[0] * jyp[0]);
                bp[1] += (jxp[0] * jxp[1] + jyp[0] * jyp[1]);
                bp[2] += (jxp[0] * jxp[2] + jyp[0] * jyp[2]);
                bp[3] += (jxp[1] * jxp[1] + jyp[1] * jyp[1]);
                bp[4] += (jxp[1] * jxp[2] + jyp[1] * jyp[2]);
                bp[5] += (jxp[2] * jxp[2] + jyp[2] * jyp[2]);
            }
        }

        ///invert the camera part
        if(mode != 2)
        {
            /////////////////////////////////////////
            const double* qw =  qwv.begin();
            if(qw)
            {
                for(size_t i = 0; i < ncam; ++i, qw += 2)
                {
                    if(qw[0] == 0)continue;
                    double* bc = blockpc + i * szbc;
                    if(sjc0)
                    {
                        const double* sjc = sjc0 + i * 8;
                        double j1 = sjc[0] * qw[0];
                        double j2 = sjc[7] * qw[1];
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
                    double* bp = bo,  *dip = di;
                    for(int  j = 0; j < vn; ++j, ++dip, bp += 9)
                    {
                        dip[0] = bp[0];
                        bp[0] = lambda2 * bp[0] + lambda1;
                    }

                //invert the matrix?
                if(radial_distortion)   InvertSymmetricMatrix<double, 8, 8>(bo, bi);
                else                    InvertSymmetricMatrix<double, 7, 8>(bo, bi);
            }
        }else
        {
            bo += szbc * ncam;
            bi += szbc * ncam;
            di += 8 * ncam;
        }


        ///////////////////////////////////////////
        //inverting the point part
        if(mode != 1)
        {
            for(size_t i = 0; i < npts; ++i, bo += 6, bi += 6, di += POINT_ALIGN)
            {
                double& M00 = bo[0], & M01 = bo[1], & M02 = bo[2];
                double& M11 = bo[3], & M12 = bo[4], & M22 = bo[5];
                di[0] = M00;  	di[1] = M11;  	di[2] = M22;

                /////////////////////////////
                M00 = M00 * lambda2 + lambda1;
                M11 = M11 * lambda2 + lambda1;
                M22 = M22 * lambda2 + lambda1;

                ///////////////////////////////
                double det = (M00 * M11 - M01 * M01) * M22 + double(2.0) * M01 * M12 * M02 - M02 * M02 * M11 - M12 * M12 * M00;
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


    void MultiplyBlockConditionerC(int ncam, const double* bi, const double*  x, double* vx, int vn, int mt = 0)
    {
        for(int i = 0; i < ncam; ++i, x += 8, vx += 8)
        {
            double *vxc = vx;
            for(int j = 0; j < vn; ++j, bi += 8, ++vxc)  *vxc = DotProduct8(bi, x);
        }
    }


    void MultiplyBlockConditionerP(int npoint, const double* bi, const double*  x, double* vx, int mt = 0)
    {
        for(int i = 0; i < npoint; ++i, bi += 6, x += POINT_ALIGN, vx += POINT_ALIGN)
        {
            vx[0] = (bi[0] * x[0] + bi[1] * x[1] + bi[2] * x[2]);
            vx[1] = (bi[1] * x[0] + bi[3] * x[1] + bi[4] * x[2]);
            vx[2] = (bi[2] * x[0] + bi[4] * x[1] + bi[5] * x[2]);
        }
    }


    void MultiplyBlockConditioner(int ncam, int npoint, const double* blocksv,
                                  const double*  vec, double* resultv, int radial, int mode,  int mt1, int mt2)
    {
        const int vn = radial ? 8 : 7;
        if(mode != 2) MultiplyBlockConditionerC(ncam, blocksv, vec, resultv, vn, mt1);
        if(mt2 == 0) mt2 = AUTO_MT_NUM(npoint * 24);
        if(mode != 1) MultiplyBlockConditionerP(npoint,  blocksv + (vn*8*ncam), vec + ncam*8, resultv + 8*ncam, mt2);
    }

    ////////////////////////////////////////////////////


    void ComputeJX( size_t nproj, size_t ncam,  const double* x, const double*  jc,
                    const double* jp, const int* jmap, double* jx, int mode, int mt = 2)
    {
        if(mode == 0)
        {
            const double* pxc = x, * pxp = pxc + ncam * 8;
            //clock_t tp = clock(); double s1 = 0, s2  = 0;
            for(size_t i = 0 ;i < nproj; ++i, jmap += 2, jc += 16, jp += POINT_ALIGN2, jx += 2)
            {
                ComputeTwoJX(jc, jp, pxc + jmap[0] * 8, pxp + jmap[1] * POINT_ALIGN, jx);
            }
        }else if(mode == 1)
        {
            const double* pxc = x;
            //clock_t tp = clock(); double s1 = 0, s2  = 0;
            for(size_t i = 0 ;i < nproj; ++i, jmap += 2, jc += 16, jp += POINT_ALIGN2, jx += 2)
            {
                const double* xc = pxc + jmap[0] * 8;
                jx[0] = DotProduct8(jc, xc)   ;
                jx[1] = DotProduct8(jc + 8, xc);
            }
        }else if(mode == 2)
        {
            const double* pxp = x + ncam * 8;
            //clock_t tp = clock(); double s1 = 0, s2  = 0;
            for(size_t i = 0 ;i < nproj; ++i, jmap += 2, jc += 16, jp += POINT_ALIGN2, jx += 2)
            {
                const double* xp = pxp + jmap[1] * POINT_ALIGN;
                jx[0] =  (jp[0] * xp[0] + jp[1] * xp[1] + jp[2] * xp[2]);
                jx[1] =  (jp[3] * xp[0] + jp[4] * xp[1] + jp[5] * xp[2]);
            }
        }
    }

    void ComputeJX_(size_t nproj, size_t ncam,  const double* x, double* jx, const double* camera,
                    const double* point,  const double* ms, const double* sj, const int*  jmap,
                    bool intrinsic_fixed, int radial_distortion, int mode, int mt = 16)
    {
        if(mode == 0)
        {
            double jcv[24 + 8]; //size_t offset = ((size_t) jcv) & 0xf;
            //Float* jc = jcv + (16 - offset) / sizeof(Float), *jp = jc + 16;
            double* jc = (double*)ALIGN_PTR(jcv), *jp = jc + 16;
            ////////////////////////////////////////
            const double* sjc = sj;
            const double* sjp = sjc? (sjc + ncam * 8) : NULL;
            const double* xc0 = x, *xp0 = x + ncam * 8;

            /////////////////////////////////
            for(size_t i = 0 ;i < nproj; ++i, ms += 2, jmap += 2, jx += 2)
            {
                const int cidx = jmap[0], pidx = jmap[1];
                const double* c = camera + cidx * 16, * pt = point + pidx * POINT_ALIGN;
                /////////////////////////////////////////////////////
                JacobianOne(c, pt, ms, jc, jc + 8, jp, jp + POINT_ALIGN, intrinsic_fixed, radial_distortion);
                if(sjc)
                {
                    //jacobian scaling
                    ScaleJ8(jc, jc + 8, sjc + cidx * 8);
                    const double* sjpi = sjp + pidx * POINT_ALIGN;
                    for(int j = 0; j < 3; ++j) {jp[j] *= sjpi[j]; jp[POINT_ALIGN + j] *= sjpi[j]; }
                }
                ////////////////////////////////////
                ComputeTwoJX(jc, jp, xc0 + cidx * 8, xp0 + pidx * POINT_ALIGN, jx);
            }
        }else if(mode == 1)
        {
            double jcv[24 + 8]; //size_t offset = ((size_t) jcv) & 0xf;
            //Float* jc = jcv + (16 - offset) / sizeof(Float);
            double* jc = (double*)ALIGN_PTR(jcv);

            ////////////////////////////////////////
            const double* sjc = sj, * xc0 = x;

            /////////////////////////////////
            for(size_t i = 0 ;i < nproj; ++i, ms += 2, jmap += 2, jx += 2)
            {
                const int cidx = jmap[0], pidx = jmap[1];
                const double* c = camera + cidx * 16, * pt = point + pidx * POINT_ALIGN;
                /////////////////////////////////////////////////////
                JacobianOne(c, pt, ms, jc, jc + 8, (double*) NULL, (double*)NULL, intrinsic_fixed, radial_distortion);
                if(sjc)ScaleJ8(jc, jc + 8, sjc + cidx * 8);
                const double* xc = xc0 + cidx * 8;
                jx[0] = DotProduct8(jc, xc)   ;
                jx[1] = DotProduct8(jc + 8, xc);
            }
        }else if(mode == 2)
        {
            double jp[8];

            ////////////////////////////////////////
            const double* sjp = sj? (sj + ncam * 8) : NULL;
            const double* xp0 = x + ncam * 8;

            /////////////////////////////////
            for(size_t i = 0 ;i < nproj; ++i, ms += 2, jmap += 2, jx += 2)
            {
                const int cidx = jmap[0], pidx = jmap[1];
                const double* c = camera + cidx * 16, * pt = point + pidx * POINT_ALIGN;
                /////////////////////////////////////////////////////
                JacobianOne(c, pt, ms, (double*) NULL, (double*) NULL, jp, jp + POINT_ALIGN, intrinsic_fixed, radial_distortion);

                const double* xp = xp0 + pidx * POINT_ALIGN;
                if(sjp)
                {
                    const double* s = sjp + pidx * POINT_ALIGN;
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


    void ComputeJtEC(    size_t ncam, const double* pe, const double* jc, const int* cmap,
                        const int* cmlist,  double* v, bool jc_transpose, int mt)
    {
        /////////////////////////////////
        for(size_t i = 0; i < ncam; ++i, ++cmap, v += 8)
        {
            int idx1 = cmap[0], idx2 = cmap[1];
            for(int j = idx1; j < idx2; ++j)
            {
                int edx = cmlist[j];
                const double* pj = jc +  ((jc_transpose? j : edx) * 16);
                const double* e  = pe + edx * 2;
                //////////////////////////////
                AddScaledVec8(e[0], pj,     v);
                AddScaledVec8(e[1], pj + 8, v);
            }
        }
    }


    void ComputeJtEP(   size_t npt, const double* pe, const double* jp,
                        const int* pmap, double* v,  int mt)
    {
        for(size_t i = 0; i < npt; ++i, ++pmap, v += POINT_ALIGN)
        {
            int idx1 = pmap[0], idx2 = pmap[1];
            const double* pj = jp + idx1 * POINT_ALIGN2;
            const double* e  = pe + idx1 * 2;
            double temp[3] = {0, 0, 0};
            for(int j = idx1; j < idx2; ++j, pj += POINT_ALIGN2, e += 2)
            {
                temp[0] += (e[0] * pj[0] + e[1] * pj[POINT_ALIGN]);
                temp[1] += (e[0] * pj[1] + e[1] * pj[POINT_ALIGN + 1]);
                temp[2] += (e[0] * pj[2] + e[1] * pj[POINT_ALIGN + 2]);
            }
            v[0] = temp[0]; v[1] = temp[1]; v[2] = temp[2];
        }
    }




    void ComputeJtE(    size_t ncam, size_t npt, const double* pe, const double* jc,
                        const int* cmap, const int* cmlist,  const double* jp,
                        const int* pmap, double* v, bool jc_transpose, int mode, int mt1, int mt2)
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


    void ComputeJtEC_(  size_t ncam, const double* ee,  double* jte,
                        const double* c, const double* point, const double* ms,
                        const int* jmap, const int* cmap, const int * cmlist,
                        bool intrinsic_fixed, int radial_distortion, int mt)
    {
        /////////////////////////////////
        double jcv[16 + 8];          //size_t offset = ((size_t) jcv) & 0xf;
        //Float* jcx = jcv + ((16 - offset) / sizeof(Float)), * jcy = jcx + 8;
        double* jcx = (double*)ALIGN_PTR(jcv), * jcy = jcx + 8;

        for(size_t i = 0; i < ncam; ++i, ++cmap, jte += 8, c += 16)
        {
            int idx1 = cmap[0], idx2 = cmap[1];

            for(int j = idx1; j < idx2; ++j)
            {
                int index = cmlist[j];
                const double* pt = point + jmap[2 * index + 1] * POINT_ALIGN;
                const double* e  = ee + index * 2;

                JacobianOne(c, pt, ms + index * 2, jcx, jcy, (double*)NULL, (double*)NULL, intrinsic_fixed, radial_distortion);

                //////////////////////////////
                AddScaledVec8(e[0], jcx, jte);
                AddScaledVec8(e[1], jcy, jte);
            }
        }
    }


    void ComputeJtE_(   size_t /*nproj*/, size_t ncam, size_t npt, const double* ee,  double* jte,
                        const double* camera, const double* point, const double* ms, const int* jmap,
                        const int* cmap, const int* cmlist, const int* pmap, const double* jp,
                        bool intrinsic_fixed, int radial_distortion, int mode, int mt)
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


    void ComputeJtE_(   size_t nproj, size_t ncam, size_t npt, const double* ee,  double* jte,
                        const double* camera, const double* point, const double* ms, const int* jmap,
                        bool intrinsic_fixed, int radial_distortion, int mode)
    {
        SetVectorZero(jte, jte + (ncam * 8 + npt * POINT_ALIGN));
        double jcv[24 + 8];  //size_t offset = ((size_t) jcv) & 0xf;
        //Float* jc = jcv + (16 - offset) / sizeof(Float), *pj = jc + 16;
        double* jc = (double*)ALIGN_PTR(jcv), *pj = jc + 16;

        double* vc0 = jte, *vp0 = jte + ncam * 8;

        for(size_t i = 0 ;i < nproj; ++i, jmap += 2, ms += 2, ee += 2)
        {
            int cidx = jmap[0], pidx = jmap[1];
            const double* c = camera + cidx * 16, * pt = point + pidx * POINT_ALIGN;

            if(mode == 0)
            {
                /////////////////////////////////////////////////////
                JacobianOne(c, pt, ms, jc, jc + 8, pj, pj + POINT_ALIGN, intrinsic_fixed, radial_distortion);

                ////////////////////////////////////////////
                double* vc = vc0 + cidx * 8, *vp = vp0 + pidx * POINT_ALIGN;
                AddScaledVec8(ee[0], jc,     vc);
                AddScaledVec8(ee[1], jc + 8, vc);
                vp[0] += (ee[0] * pj[0] + ee[1] * pj[POINT_ALIGN]);
                vp[1] += (ee[0] * pj[1] + ee[1] * pj[POINT_ALIGN + 1]);
                vp[2] += (ee[0] * pj[2] + ee[1] * pj[POINT_ALIGN + 2]);
            }else if(mode == 1)
            {
                /////////////////////////////////////////////////////
                JacobianOne(c, pt, ms, jc, jc + 8, (double*) NULL, (double*) NULL, intrinsic_fixed, radial_distortion);

                ////////////////////////////////////////////
                double* vc = vc0 + cidx * 8;
                AddScaledVec8(ee[0], jc,     vc);
                AddScaledVec8(ee[1], jc + 8, vc);
            }else
            {
               /////////////////////////////////////////////////////
                JacobianOne(c, pt, ms, (double*) NULL, (double*) NULL, pj, pj + POINT_ALIGN, intrinsic_fixed, radial_distortion);

                ////////////////////////////////////////////
                double *vp = vp0 + pidx * POINT_ALIGN;
                vp[0] += (ee[0] * pj[0] + ee[1] * pj[POINT_ALIGN]);
                vp[1] += (ee[0] * pj[1] + ee[1] * pj[POINT_ALIGN + 1]);
                vp[2] += (ee[0] * pj[2] + ee[1] * pj[POINT_ALIGN + 2]);
            }
        }
    }
}

using namespace ProgramCPU;

SparseBundleCPU:: SparseBundleCPU()
    : _num_camera(0)
    , _num_point(0)
    , _num_imgpt(0)
    , _camera_data(NULL)
    , _point_data(NULL)
    , _imgpt_data(NULL)
    , _camera_idx(NULL)
    , _point_idx(NULL)
    , _projection_sse(0)
    , _num_imgpt_q(0)
{
    __cpu_data_precision = sizeof(double);
    if(__num_cpu_cores == 0)	__num_cpu_cores = omp_get_num_procs();
    if(__verbose_level)			std::cout  << "CPU " << (__cpu_data_precision == 4 ? "single" : "double")
                                           << "-precisoin solver; " << __num_cpu_cores << " cores"
#ifdef CPUPBA_USE_AVX
                                           << " (AVX)"
#endif
                                           << ".\n";
    //the following configuration are totally based my personal experience
    //on two computers.. you should adjust them according to your system.
    //try run driver filename -profile --float to see how speed varies
    ////////////////////////////////////////
    __num_cpu_thread[FUNC_JX] = __num_cpu_cores;
    __num_cpu_thread[FUNC_JX_] = __num_cpu_cores;
    __num_cpu_thread[FUNC_JTE_] = __num_cpu_cores;
    __num_cpu_thread[FUNC_JJ_JCO_JCT_JP] =  __num_cpu_cores;
    __num_cpu_thread[FUNC_JJ_JCO_JP] = __num_cpu_cores;
    __num_cpu_thread[FUNC_JJ_JCT_JP] = __num_cpu_cores;
    __num_cpu_thread[FUNC_JJ_JP] = __num_cpu_cores;
    __num_cpu_thread[FUNC_PJ] = __num_cpu_cores;
    __num_cpu_thread[FUNC_BCC_JCO] = __num_cpu_cores;
    __num_cpu_thread[FUNC_BCC_JCT] = __num_cpu_cores;
    __num_cpu_thread[FUNC_BCP] = __num_cpu_cores;

    ////this behavious is different between CPU and GPU
    __multiply_jx_usenoj = false;

    ///////////////////////////////////////////////////////////////////////////////
    //To get the best performance, you should ajust the number of threads
    //Linux and Windows may also have different thread launching overhead.

    //////////////////////////////////////////////////////////////
    __num_cpu_thread[FUNC_JTEC_JCT] = __num_cpu_cores * 2;
    __num_cpu_thread[FUNC_JTEC_JCO] = __num_cpu_cores * 2;
    __num_cpu_thread[FUNC_JTEP] = __num_cpu_cores;

    ///////////
    __num_cpu_thread[FUNC_MPC] = 1; //single thread always faster with my experience

    //see the AUTO_MT_NUM marcro for definition
    __num_cpu_thread[FUNC_MPP] = 0; //automatically chosen according to size
    __num_cpu_thread[FUNC_VS] = 0;  //automatically chosen according to size
    __num_cpu_thread[FUNC_VV] = 0;  //automatically chosen accodring to size
}

void SparseBundleCPU:: SetCameraData(size_t ncam,  CameraT* cams)
{
    if(sizeof(CameraT) != 16 * sizeof(float)) return;  //never gonna happen...?
     _num_camera = (int) ncam;
    _camera_data = cams;
    _focal_mask  = NULL;
}

void SparseBundleCPU::SetFocalMask(const int* fmask, float weight)
{
    _focal_mask = fmask;
    _weight_q = weight;
}

void SparseBundleCPU:: SetPointData(size_t npoint, Point3D* pts)
{
    _num_point = (int) npoint;
    _point_data = (float*) pts;
}

void SparseBundleCPU:: SetProjection(size_t nproj, const Point2D* imgpts, const int* point_idx, const int* cam_idx)
{
    _num_imgpt = (int) nproj;
    _imgpt_data = (float*) imgpts;
    _camera_idx = cam_idx;
    _point_idx = point_idx;
}

float SparseBundleCPU::GetMeanSquaredError()
{
    return float(_projection_sse / (_num_imgpt * __focal_scaling * __focal_scaling));
}

int SparseBundleCPU:: RunBundleAdjustment()
{
    ResetBundleStatistics();
    BundleAdjustment();
    if(__num_lm_success > 0) SaveBundleStatistics(_num_camera,  _num_point, _num_imgpt);
    if(__num_lm_success > 0) PrintBundleStatistics();
    ResetTemporarySetting();
    return __num_lm_success;
}

int SparseBundleCPU:: ValidateInputData()
{
    if(_camera_data == NULL) return STATUS_CAMERA_MISSING;
    if(_point_data == NULL)  return STATUS_POINT_MISSING;
    if(_imgpt_data == NULL)  return STATUS_MEASURMENT_MISSING;
    if(_camera_idx == NULL || _point_idx == NULL) return STATUS_PROJECTION_MISSING;
    return STATUS_SUCCESS;
}

int SparseBundleCPU::InitializeBundle()
{
    /////////////////////////////////////////////////////
    TimerBA timer(this, TIMER_GPU_ALLOCATION);
    InitializeStorageForSFM();
    InitializeStorageForCG();

    if(__debug_pba) DumpCooJacobian();

    return STATUS_SUCCESS;
}

int SparseBundleCPU::GetParameterLength()
{
    return _num_camera * 8 + POINT_ALIGN * _num_point;
}

void SparseBundleCPU::BundleAdjustment()
{
    if(ValidateInputData() != STATUS_SUCCESS) return;

    ////////////////////////
    TimerBA timer(this, TIMER_OVERALL);

    NormalizeData();
    if(InitializeBundle() != STATUS_SUCCESS)
    {
        //failed to allocate gpu storage
    }else if(__profile_pba)
    {
        //profiling some stuff
        RunProfileSteps();
    }else
    {
        //real optimization
        AdjustBundleAdjsutmentMode();
        NonlinearOptimizeLM();
        TransferDataToHost();
    }
    DenormalizeData();
}

void SparseBundleCPU:: NormalizeData()
{
    TimerBA timer(this, TIMER_PREPROCESSING);
    NormalizeDataD();
    NormalizeDataF();
}

void SparseBundleCPU:: TransferDataToHost()
{
    TimerBA timer(this, TIMER_GPU_DOWNLOAD);
    std::copy(_cuCameraData.begin(), _cuCameraData.end(), ((float*)_camera_data));
#ifdef POINT_DATA_ALIGN4
    std::copy(_cuPointData.begin(), _cuPointData.end(), _point_data);
#else
    for(size_t i = 0, j = 0; i < _cuPointData.size(); j++)
    {
       _point_data[j++]  = (float) _cuPointData[i++] ;
       _point_data[j++]  = (float) _cuPointData[i++];
       _point_data[j++]  = (float) _cuPointData[i++];
    }
#endif
}

#define ALLOCATE_REQUIRED_DATA(NAME, num, channels)    \
    {NAME.resize((num)* (channels)); total_sz += NAME.size() * sizeof(double);}
#define ALLOCATE_OPTIONAL_DATA(NAME, num, channels, option)    \
    if(option)  ALLOCATE_REQUIRED_DATA(NAME, num, channels)  else{NAME.resize(0); }
//////////////////////////////////////////////
bool SparseBundleCPU::InitializeStorageForSFM()
{
    size_t total_sz = 0;
    //////////////////////////////////////////////////
    ProcessIndexCameraQ(_cuCameraQMap, _cuCameraQList);
    total_sz += ((_cuCameraQMap.size() + _cuCameraQList.size()) * sizeof(int) / 1024 / 1024);

    ///////////////////////////////////////////////////////////////////
    ALLOCATE_REQUIRED_DATA(_cuPointData, _num_point, POINT_ALIGN);                    //4n
    ALLOCATE_REQUIRED_DATA(_cuCameraData, _num_camera, 16);                 //16m
    ALLOCATE_REQUIRED_DATA(_cuCameraDataEX, _num_camera, 16);               //16m

    ////////////////////////////////////////////////////////////////
    ALLOCATE_REQUIRED_DATA(_cuCameraMeasurementMap,_num_camera + 1, 1);     //m
    ALLOCATE_REQUIRED_DATA(_cuCameraMeasurementList,_num_imgpt, 1);         //k
    ALLOCATE_REQUIRED_DATA(_cuPointMeasurementMap,_num_point + 1, 1);       //n
    ALLOCATE_REQUIRED_DATA(_cuProjectionMap,_num_imgpt, 2);                 //2k
    ALLOCATE_REQUIRED_DATA(_cuImageProj, _num_imgpt + _num_imgpt_q, 2);      //2k
    ALLOCATE_REQUIRED_DATA(_cuPointDataEX, _num_point, POINT_ALIGN);        //4n
    ALLOCATE_REQUIRED_DATA(_cuMeasurements,_num_imgpt, 2);                  //2k
    ALLOCATE_REQUIRED_DATA(_cuCameraQMapW, _num_imgpt_q, 2);
    ALLOCATE_REQUIRED_DATA(_cuCameraQListW, (_num_imgpt_q > 0 ? _num_camera : 0), 2);


    ALLOCATE_OPTIONAL_DATA(_cuJacobianPoint,_num_imgpt * 2,  POINT_ALIGN, !__no_jacobian_store);        //8k
    ALLOCATE_OPTIONAL_DATA(_cuJacobianCameraT, _num_imgpt * 2 , 8, !__no_jacobian_store && __jc_store_transpose);//16k
    ALLOCATE_OPTIONAL_DATA(_cuJacobianCamera, _num_imgpt * 2 , 8, !__no_jacobian_store && __jc_store_original);//16k
    ALLOCATE_OPTIONAL_DATA(_cuCameraMeasurementListT,_num_imgpt, 1,  __jc_store_transpose);  //k


    //////////////////////////////////////////
    BundleTimerSwap(TIMER_PREPROCESSING, TIMER_GPU_ALLOCATION);
    ////mapping from camera to measuremnts
    std::vector<int>& cpi = _cuCameraMeasurementMap;     cpi.resize(_num_camera + 1);
    std::vector<int>& cpidx = _cuCameraMeasurementList;  cpidx.resize(_num_imgpt);
    std::vector<int> cpnum(_num_camera, 0);              cpi[0] = 0;
    for(int i = 0; i < _num_imgpt; ++i) cpnum[_camera_idx[i]]++;
    for(int i = 1; i <= _num_camera; ++i) cpi[i] = cpi[i - 1] + cpnum[i - 1];
    ///////////////////////////////////////////////////////
    std::vector<int> cptidx = cpi;
    for(int i = 0; i < _num_imgpt; ++i) cpidx[cptidx[_camera_idx[i]] ++] = i;

    ///////////////////////////////////////////////////////////
    if(_cuCameraMeasurementListT.size())
    {
        std::vector<int> &ridx = _cuCameraMeasurementListT; ridx.resize(_num_imgpt);
        for(int i = 0; i < _num_imgpt; ++i)ridx[cpidx[i]] = i;
    }

    ////////////////////////////////////////
    /////constaraint weights.
    if(_num_imgpt_q > 0) ProcessWeightCameraQ(cpnum, _cuCameraQMap, _cuCameraQMapW.begin(), _cuCameraQListW.begin());

    ///////////////////////////////////////////////////////////////////////////////
    std::copy((float*)_camera_data, ((float*)_camera_data) + _cuCameraData.size(), _cuCameraData.begin());


#ifdef POINT_DATA_ALIGN4
    std::copy(_point_data, _point_data + _cuPointData.size(), _cuPointData.begin());
#else
    for(size_t i = 0, j = 0; i < _cuPointData.size(); j++)
    {
       _cuPointData[i++] = _point_data[j++];
       _cuPointData[i++] = _point_data[j++];
       _cuPointData[i++] = _point_data[j++];
    }
#endif



    ////////////////////////////////////////////
    ///////mapping from point to measurment
    std::vector<int> & ppi = _cuPointMeasurementMap;  ppi.resize(_num_point + 1);
    for(int i = 0, last_point = -1; i < _num_imgpt; ++i)
    {
        int pt = _point_idx[i];
        while(last_point < pt) ppi[++last_point] = i;
    }
    ppi[_num_point] = _num_imgpt;

    //////////projection map
    std::vector<int>& pmp = _cuProjectionMap; pmp.resize(_num_imgpt *2);
    for(int i = 0; i < _num_imgpt; ++i)
    {
        int* imp = &pmp[i * 2];
        imp[0] =  _camera_idx[i];
        imp[1] = _point_idx[i];
    }
    BundleTimerSwap(TIMER_PREPROCESSING, TIMER_GPU_ALLOCATION);
    //////////////////////////////////////////////////////////////

    __memory_usage = total_sz;
    if(__verbose_level > 1) std::cout << "Memory for Motion/Structure/Jacobian:\t" << (total_sz /1024/1024) << "MB\n";

    return true;
}

bool SparseBundleCPU::ProcessIndexCameraQ(std::vector<int>&qmap, std::vector<int>& qlist)
{
    ///////////////////////////////////
    qlist.resize(0);
    qmap.resize(0);
    _num_imgpt_q = 0;

    if(_camera_idx == NULL) return true;
    if(_point_idx == NULL) return true;
    if(_focal_mask == NULL) return true;
    if(_num_camera == 0) return true;
    if(_weight_q <= 0) return true;

    ///////////////////////////////////////

    int error = 0;
    std::vector<int> temp(_num_camera * 2, -1);

    for(int i = 0; i < _num_camera; ++i)
    {
        int iq = _focal_mask[i];
        if(iq > i) {error = 1; break;}
        if(iq < 0) continue;
        if(iq == i) continue;
        int ip = temp[2 * iq];
        //float ratio = _camera_data[i].f / _camera_data[iq].f;
        //if(ratio < 0.01 || ratio > 100)
        //{
        //	std::cout << "Warning: constaraints on largely different camreas\n";
        //	continue;
        //}else
        if(_focal_mask[iq] != iq)
        {
            error = 1; break;
        }else if(ip == -1)
        {
            temp[2 * iq] = i;
            temp[2 * iq + 1] = i;
            temp[2 * i] = iq;
            temp[2 * i + 1] = iq;
        }else
        {
            //maintain double-linked list
            temp[2 * i] = ip;
            temp[2 * i + 1] = iq;
            temp[2 * ip + 1] = i;
            temp[2 * iq] = i;
        }
    }

    if(error)
    {
        std::cout << "Error: incorrect constraints\n";
        _focal_mask = NULL;
        return false;
    }

    ////////////////////////////////////////
    qlist.resize(_num_camera * 2, -1);
    for(int i = 0; i < _num_camera; ++i)
    {
        int inext = temp[2 * i + 1];
        if(inext == -1) continue;
        qlist[2 * i] = _num_imgpt_q;
        qlist[2 * inext + 1] = _num_imgpt_q;
        qmap.push_back(i);
        qmap.push_back(inext);
        _num_imgpt_q++;
    }
    return true;
}

void SparseBundleCPU::ProcessWeightCameraQ(std::vector<int>&cpnum, std::vector<int>&qmap, double* qmapw, double* qlistw)
{
    //set average focal length and average radial distortion
    std::vector<double>	qpnum(_num_camera, 0),		qcnum(_num_camera, 0);
    std::vector<double>	fs(_num_camera, 0),			rs(_num_camera, 0);

    for(int i = 0; i < _num_camera; ++i)
    {
        int qi = _focal_mask[i]; if(qi == -1)continue;
        //float ratio = _camera_data[i].f / _camera_data[qi].f;
        //if(ratio < 0.01 || ratio > 100)			continue;
        fs[qi] += _camera_data[i].f;
        rs[qi] += _camera_data[i].radial;
        qpnum[qi] += cpnum[i];
        qcnum[qi] += 1.0f;
    }

    //this seems not really matter..they will converge anyway
    for(int i = 0; i < _num_camera; ++i)
    {
        int qi = _focal_mask[i]; if(qi == -1)continue;
        //float ratio = _camera_data[i].f / _camera_data[qi].f;
        //if(ratio < 0.01 || ratio > 100)			continue;
        _camera_data[i].f		= fs[qi] / qcnum[qi];
        _camera_data[i].radial	= rs[qi] / qcnum[qi];
    }/**/

    /////////////////////////////////////////
    std::fill(qlistw, qlistw + _num_camera * 2, 0);

    for(int i = 0;i < _num_imgpt_q; ++i)
    {
        int cidx  = qmap[i * 2], qi = _focal_mask[cidx];
        double wi = sqrt(qpnum[qi] / qcnum[qi]) *  _weight_q;
        double wr = (__use_radial_distortion ? wi * _camera_data[qi].f: 0.0) ;
        qmapw[i * 2] = wi; 		qmapw[i * 2 + 1] = wr;
        qlistw[cidx * 2] = wi;	qlistw[cidx * 2 + 1] = wr;
    }
}

/////////////////////////////////////////////////
bool SparseBundleCPU::InitializeStorageForCG()
{
    size_t total_sz = 0;
    int plen = GetParameterLength();  //q = 8m + 3n

    //////////////////////////////////////////// 6q
    ALLOCATE_REQUIRED_DATA(_cuVectorJtE, plen, 1);
    ALLOCATE_REQUIRED_DATA(_cuVectorXK, plen, 1);
    ALLOCATE_REQUIRED_DATA(_cuVectorJJ, plen, 1);
    ALLOCATE_REQUIRED_DATA(_cuVectorZK, plen, 1);
    ALLOCATE_REQUIRED_DATA(_cuVectorPK, plen, 1);
    ALLOCATE_REQUIRED_DATA(_cuVectorRK, plen, 1);

    ///////////////////////////////////////////
    unsigned int cblock_len = (__use_radial_distortion? 64 : 56);
    ALLOCATE_REQUIRED_DATA(_cuBlockPC, _num_camera * cblock_len + 6 * _num_point, 1);    //64m + 12n
    ALLOCATE_REQUIRED_DATA(_cuVectorJX, _num_imgpt + _num_imgpt_q, 2);            //2k
    ALLOCATE_OPTIONAL_DATA(_cuVectorSJ, plen, 1, __jacobian_normalize);

    /////////////////////////////////////////
    __memory_usage += total_sz;
    if(__verbose_level > 1) std::cout << "Memory for Conjugate Gradient Solver:\t" << (total_sz /1024/1024) << "MB\n";
    return true;
}


///////////////////////////////////////////////////
void SparseBundleCPU::PrepareJacobianNormalization()
{
    if(!_cuVectorSJ.size())return;

    if((__jc_store_transpose || __jc_store_original) && _cuJacobianPoint.size() && !__bundle_current_mode)
    {
        VectorF null;        null.swap(_cuVectorSJ);
        EvaluateJacobians();    null.swap(_cuVectorSJ);
        ComputeDiagonal(_cuVectorSJ);
        ComputeSQRT(_cuVectorSJ);
    }else
    {
        VectorF null;        null.swap(_cuVectorSJ);
        EvaluateJacobians();    ComputeBlockPC(0, true);
        null.swap(_cuVectorSJ);
        _cuVectorJJ.swap(_cuVectorSJ);
        ComputeRSQRT(_cuVectorSJ);
    }
}

void SparseBundleCPU::EvaluateJacobians()
{
    if(__no_jacobian_store) return;
    if(__bundle_current_mode == BUNDLE_ONLY_MOTION && !__jc_store_original && !__jc_store_transpose) return;

    ConfigBA::TimerBA timer (this, TIMER_FUNCTION_JJ);

    if(__jc_store_original || !__jc_store_transpose)
    {
        int fid = __jc_store_original ? (__jc_store_transpose ? FUNC_JJ_JCO_JCT_JP : FUNC_JJ_JCO_JP) : FUNC_JJ_JP;
        ComputeJacobian(_num_imgpt, _num_camera, _cuCameraData.begin(), _cuPointData.begin(), _cuJacobianCamera.begin(),
            _cuJacobianPoint.begin(), &_cuProjectionMap.front(), _cuVectorSJ.begin(),
            _cuMeasurements.begin(),  __jc_store_transpose? &_cuCameraMeasurementListT.front() : NULL,
            __fixed_intrinsics, __use_radial_distortion, false, _cuJacobianCameraT.begin(),
            __num_cpu_thread[fid]);
    }else
    {
        ComputeJacobian(_num_imgpt, _num_camera, _cuCameraData.begin(), _cuPointData.begin(), _cuJacobianCameraT.begin(),
            _cuJacobianPoint.begin(), &_cuProjectionMap.front(), _cuVectorSJ.begin(),
            _cuMeasurements.begin(), &_cuCameraMeasurementListT.front(),
            __fixed_intrinsics, __use_radial_distortion, true, ((double*) 0),
            __num_cpu_thread[FUNC_JJ_JCT_JP]);

    }
    ++__num_jacobian_eval;
}

void SparseBundleCPU::ComputeJtE(VectorF& E, VectorF& JtE, int mode)
{
    ConfigBA::TimerBA timer (this, TIMER_FUNCTION_JTE);
    if(mode == 0) mode = __bundle_current_mode;

    if(__no_jacobian_store ||  (!__jc_store_original && !__jc_store_transpose))
    {
        if(_cuJacobianPoint.size())
        {
            ProgramCPU::ComputeJtE_(_num_imgpt, _num_camera, _num_point, E.begin(), JtE.begin(),
                _cuCameraData.begin(), _cuPointData.begin(), _cuMeasurements.begin(), &_cuProjectionMap.front(),
                &_cuCameraMeasurementMap.front(), &_cuCameraMeasurementList.front(),
                &_cuPointMeasurementMap.front(), _cuJacobianPoint.begin(),
                __fixed_intrinsics, __use_radial_distortion, mode, __num_cpu_thread[FUNC_JTE_]);

            if(_cuVectorSJ.size() && mode != 2) ProgramCPU::ComputeVXY(JtE, _cuVectorSJ, JtE, _num_camera * 8);
        }else
        {
            ProgramCPU::ComputeJtE_(_num_imgpt, _num_camera, _num_point, E.begin(), JtE.begin(),
                _cuCameraData.begin(), _cuPointData.begin(), _cuMeasurements.begin(), &_cuProjectionMap.front(),
                __fixed_intrinsics, __use_radial_distortion, mode);

            //////////////////////////////////////////////////////////
            //if(_cuVectorSJ.size())  ProgramCPU::ComputeVXY(JtE, _cuVectorSJ, JtE);
            if(!_cuVectorSJ.size()){}
            else if(mode == 2) ComputeVXY(JtE, _cuVectorSJ, JtE, _num_point * POINT_ALIGN, _num_camera * 8);
            else if(mode == 1) ComputeVXY(JtE, _cuVectorSJ, JtE, _num_camera * 8);
            else               ComputeVXY(JtE, _cuVectorSJ, JtE);
        }
    }else    if( __jc_store_transpose)
    {
        ProgramCPU::ComputeJtE(_num_camera, _num_point, E.begin(), _cuJacobianCameraT.begin(),
            &_cuCameraMeasurementMap.front(), &_cuCameraMeasurementList.front(), _cuJacobianPoint.begin(),
            &_cuPointMeasurementMap.front(), JtE.begin(), true,
            mode, __num_cpu_thread[FUNC_JTEC_JCT], __num_cpu_thread[FUNC_JTEP]);
    }else
    {
        ProgramCPU::ComputeJtE(_num_camera, _num_point, E.begin(), _cuJacobianCamera.begin(),
            &_cuCameraMeasurementMap.front(),  &_cuCameraMeasurementList.front(), _cuJacobianPoint.begin(),
            &_cuPointMeasurementMap.front(), JtE.begin(), false,
            mode, __num_cpu_thread[FUNC_JTEC_JCO], __num_cpu_thread[FUNC_JTEP]);
    }

    if(mode != 2 && _num_imgpt_q > 0)
    {
        ProgramCPU::ComputeJQtEC(_num_camera, E.begin() + 2 * _num_imgpt,
            &_cuCameraQList.front(),  _cuCameraQListW.begin(), _cuVectorSJ.begin(), JtE.begin());
    }
}

void SparseBundleCPU::SaveBundleRecord(int iter, float res, float damping, float& g_norm, float& g_inf)
{
    //do not really compute if parameter not specified...
    //for large dataset, it never converges..
    g_inf  = __lm_check_gradient?  float(ComputeVectorMax(_cuVectorJtE)) : 0;
    g_norm = __save_gradient_norm? float(ComputeVectorNorm(_cuVectorJtE)) : g_inf;
    ConfigBA::SaveBundleRecord(iter, res, damping, g_norm, g_inf);
}

float SparseBundleCPU::EvaluateProjection(VectorF& cam, VectorF&point, VectorF& proj)
{
    ++__num_projection_eval;
    ConfigBA::TimerBA timer (this, TIMER_FUNCTION_PJ);
    ComputeProjection(_num_imgpt, cam.begin(), point.begin(), _cuMeasurements.begin(),
        &_cuProjectionMap.front(), proj.begin(), __use_radial_distortion, __num_cpu_thread[FUNC_PJ]);
    if(_num_imgpt_q > 0) ComputeProjectionQ(_num_imgpt_q, cam.begin(), &_cuCameraQMap.front(),
                                             _cuCameraQMapW.begin(), proj.begin() + 2 * _num_imgpt);
    return (float) ComputeVectorNorm(proj, __num_cpu_thread[FUNC_VS]);
}

float SparseBundleCPU::EvaluateProjectionX(VectorF& cam, VectorF&point, VectorF& proj)
{
    ++__num_projection_eval;
    ConfigBA::TimerBA timer (this, TIMER_FUNCTION_PJ);
    ComputeProjectionX(_num_imgpt, cam.begin(), point.begin(), _cuMeasurements.begin(),
        &_cuProjectionMap.front(), proj.begin(), __use_radial_distortion, __num_cpu_thread[FUNC_PJ]);
    if(_num_imgpt_q > 0) ComputeProjectionQ(_num_imgpt_q, cam.begin(), &_cuCameraQMap.front(),
                                            _cuCameraQMapW.begin(),  proj.begin() + 2 * _num_imgpt);
    return (float) ComputeVectorNorm(proj, __num_cpu_thread[FUNC_VS]);
}

void SparseBundleCPU::ComputeJX(VectorF& X, VectorF& JX, int mode)
{
    ConfigBA::TimerBA timer (this, TIMER_FUNCTION_JX);
    if(__no_jacobian_store || (__multiply_jx_usenoj && mode != 2) || !__jc_store_original)
    {
        ProgramCPU::ComputeJX_(_num_imgpt, _num_camera, X.begin(), JX.begin(), _cuCameraData.begin(), _cuPointData.begin(),
                          _cuMeasurements.begin(), _cuVectorSJ.begin(), &_cuProjectionMap.front(),
                          __fixed_intrinsics, __use_radial_distortion, mode, __num_cpu_thread[FUNC_JX_]);
    }else
    {
        ProgramCPU::ComputeJX( _num_imgpt, _num_camera, X.begin(), _cuJacobianCamera.begin(),
                _cuJacobianPoint.begin(), &_cuProjectionMap.front(), JX.begin(), mode, __num_cpu_thread[FUNC_JX]);
    }

    if(_num_imgpt_q > 0 && mode != 2)
    {
        ProgramCPU::ComputeJQX(_num_imgpt_q, X.begin(), &_cuCameraQMap.front(), _cuCameraQMapW.begin(),
                                _cuVectorSJ.begin(), JX.begin() + 2 * _num_imgpt);
    }

}

void SparseBundleCPU::ComputeBlockPC(float lambda, bool dampd)
{
    ConfigBA::TimerBA timer (this, TIMER_FUNCTION_BC);

    if(__no_jacobian_store ||  (!__jc_store_original && !__jc_store_transpose && __bundle_current_mode != 2))
    {
        ComputeDiagonalBlock_(lambda, dampd, _cuCameraData, _cuPointData, _cuMeasurements,
            _cuProjectionMap, _cuVectorSJ, _cuCameraQListW, _cuVectorJJ, _cuBlockPC,
            __fixed_intrinsics, __use_radial_distortion,  __bundle_current_mode);
    }else if(__jc_store_transpose)
    {
        ComputeDiagonalBlock(_num_camera, _num_point, lambda, dampd, _cuJacobianCameraT.begin(),
            &_cuCameraMeasurementMap.front(), _cuJacobianPoint.begin(), &_cuPointMeasurementMap.front(),
            &_cuCameraMeasurementList.front(), _cuVectorSJ.begin(), _cuCameraQListW.begin(),
            _cuVectorJJ.begin(), _cuBlockPC.begin(), __use_radial_distortion, true,
            __num_cpu_thread[FUNC_BCC_JCT], __num_cpu_thread[FUNC_BCP],	__bundle_current_mode);
    }else
    {
        ComputeDiagonalBlock(_num_camera, _num_point,lambda, dampd, _cuJacobianCamera.begin(),
            &_cuCameraMeasurementMap.front(), _cuJacobianPoint.begin(), &_cuPointMeasurementMap.front(),
            &_cuCameraMeasurementList.front(),  _cuVectorSJ.begin(), _cuCameraQListW.begin(),
            _cuVectorJJ.begin(), _cuBlockPC.begin(), __use_radial_distortion, false,
            __num_cpu_thread[FUNC_BCC_JCO], __num_cpu_thread[FUNC_BCP],	__bundle_current_mode);
    }

}

void SparseBundleCPU::ApplyBlockPC(VectorF& v, VectorF& pv, int mode)
{
    ConfigBA::TimerBA timer (this, TIMER_FUNCTION_MP);
    MultiplyBlockConditioner(_num_camera, _num_point,
        _cuBlockPC.begin(), v.begin(), pv.begin(),  __use_radial_distortion, mode,
        __num_cpu_thread[FUNC_MPC], __num_cpu_thread[FUNC_MPP]);
}

void SparseBundleCPU::ComputeDiagonal(VectorF& JJ)
{
    ConfigBA::TimerBA timer (this, TIMER_FUNCTION_DD);
    if(__no_jacobian_store)
    {

    }else if(__jc_store_transpose)
    {
        ProgramCPU::ComputeDiagonal(_cuJacobianCameraT, _cuCameraMeasurementMap  ,_cuJacobianPoint
               , _cuPointMeasurementMap, _cuCameraMeasurementList, _cuCameraQListW.begin(),
               JJ, true, __use_radial_distortion);
    }else if(__jc_store_original)
    {
        ProgramCPU::ComputeDiagonal(_cuJacobianCamera, _cuCameraMeasurementMap ,_cuJacobianPoint
               , _cuPointMeasurementMap, _cuCameraMeasurementList, _cuCameraQListW.begin(),
               JJ, false, __use_radial_distortion);
    }
}

void SparseBundleCPU::NormalizeDataF()
{
    int incompatible_radial_distortion = 0;
    _cuMeasurements.resize(_num_imgpt * 2);
    if(__focal_normalize)
    {
        if(__focal_scaling == 1.0f)
        {
            //------------------------------------------------------------------
            //////////////////////////////////////////////////////////////
            std::vector<float> focals(_num_camera);
            for(int i = 0; i < _num_camera; ++i) focals[i] = _camera_data[i].f;
            std::nth_element(focals.begin(), focals.begin() + _num_camera / 2, focals.end());
            float median_focal_length = focals[_num_camera/2];
            __focal_scaling = __data_normalize_median / median_focal_length;
            double radial_factor = median_focal_length * median_focal_length * 4.0f;

            ///////////////////////////////

            for(int i = 0; i < _num_imgpt * 2; ++i)
            {
                _cuMeasurements[i] = double(_imgpt_data[i] * __focal_scaling);
            }
            for(int i = 0; i < _num_camera; ++i)
            {
                _camera_data[i].f *= __focal_scaling;
                if(!__use_radial_distortion)
                {
                }else if(__reset_initial_distortion)
                {
                    _camera_data[i].radial = 0;
                } else if( _camera_data[i].distortion_type != __use_radial_distortion)
                {
                    incompatible_radial_distortion ++;
                    _camera_data[i].radial = 0;
                } else if(__use_radial_distortion == -1)
                {
                    _camera_data[i].radial *= radial_factor;
                }
            }
            if(__verbose_level > 2) std::cout << "Focal length normalized by " << __focal_scaling << '\n';
            __reset_initial_distortion = false;
        }
    }else
    {
        if(__use_radial_distortion)
        {
            for(int i = 0; i < _num_camera; ++i)
            {
                if( __reset_initial_distortion)
                {
                    _camera_data[i].radial = 0;
                }else if(_camera_data[i].distortion_type!= __use_radial_distortion)
                {
                    _camera_data[i].radial = 0;
                    incompatible_radial_distortion ++;
                }
            }
            __reset_initial_distortion = false;
        }
        std::copy(_imgpt_data, _imgpt_data + _cuMeasurements.size(), _cuMeasurements.begin());
    }

    if(incompatible_radial_distortion)
    {
        std::cout << "ERROR: incompatible radial distortion input; reset to 0;\n";
    }

}

void SparseBundleCPU::NormalizeDataD()
{

    if(__depth_scaling == 1.0f)
    {
        const float     dist_bound = 1.0f;
        std::vector<float>   oz(_num_imgpt);
        std::vector<float>   cpdist1(_num_camera,  dist_bound);
        std::vector<float>   cpdist2(_num_camera, -dist_bound);
        std::vector<int>     camnpj(_num_camera, 0), cambpj(_num_camera, 0);
        int bad_point_count = 0;
        for(int i = 0; i < _num_imgpt; ++i)
        {
            int cmidx = _camera_idx[i];
            CameraT * cam = _camera_data + cmidx;
            float *rz = cam->m[2];
            float *x = _point_data + 4 * _point_idx[i];
            oz[i] = (rz[0]*x[0]+rz[1]*x[1]+rz[2]*x[2]+ cam->t[2]);

            /////////////////////////////////////////////////
            //points behind camera may causes big problem
            float ozr = oz[i] / cam->t[2];
            if(fabs(ozr) < __depth_check_epsilon)
            {
                bad_point_count++;
                float px = cam->f * (cam->m[0][0]*x[0]+cam->m[0][1]*x[1]+cam->m[0][2]*x[2]+ cam->t[0]);
                float py = cam->f * (cam->m[1][0]*x[0]+cam->m[1][1]*x[1]+cam->m[1][2]*x[2]+ cam->t[1]);
                float mx = _imgpt_data[i * 2], my = _imgpt_data[ 2 * i + 1];
                bool checkx = fabs(mx) > fabs(my);
                if( ( checkx && px * oz[i] * mx < 0 && fabs(mx) > 64) || (!checkx && py * oz[i] * my < 0 && fabs(my) > 64))
                {
                    if(__verbose_level > 3)
                    std::cout << "Warning: proj of #" << cmidx << " on the wrong side, oz = "<< oz[i]
                              << " (" << (px / oz[i]) << ',' << (py / oz[i]) << ") (" << mx << ',' <<  my <<")\n";
                    /////////////////////////////////////////////////////////////////////////
                    if(oz[i] > 0)     cpdist2[cmidx] = 0;
                    else              cpdist1[cmidx] = 0;
                }
                if(oz[i] >= 0) cpdist1[cmidx] = std::min(cpdist1[cmidx], oz[i]);
                else           cpdist2[cmidx] = std::max(cpdist2[cmidx], oz[i]);
            }
            if(oz[i] < 0) { __num_point_behind++;   cambpj[cmidx]++;}
            camnpj[cmidx]++;
        }
        if(bad_point_count > 0 && __depth_degeneracy_fix)
        {
            if(!__focal_normalize || !__depth_normalize) std::cout << "Enable data normalization on degeneracy\n";
            __focal_normalize = true;
            __depth_normalize = true;
        }
        if(__depth_normalize )
        {
            std::nth_element(oz.begin(), oz.begin() + _num_imgpt / 2, oz.end());
            float oz_median = oz[_num_imgpt / 2];
            float shift_min = std::min(oz_median * 0.001f, 1.0f);
            float dist_threshold = shift_min * 0.1f;
            __depth_scaling =  (1.0 / oz_median) / __data_normalize_median;
            if(__verbose_level > 2) std::cout << "Depth normalized by " << __depth_scaling
                                              << " (" << oz_median << ")\n";

            for(int i = 0; i < _num_camera; ++i)
            {
                //move the camera a little bit?
                if(!__depth_degeneracy_fix)
                {

                }else if((cpdist1[i] < dist_threshold || cpdist2[i] > -dist_threshold) )
                {
                    float shift_epsilon = fabs(_camera_data[i].t[2] * FLT_EPSILON);
                    float shift = std::max(shift_min, shift_epsilon);
                    bool  boths = cpdist1[i] < dist_threshold && cpdist2[i] > -dist_threshold;
                    _camera_data[i].t[2] += shift;
                    if(__verbose_level > 3)
                        std::cout << "Adjust C" << std::setw(5) << i << " by " << std::setw(12) << shift
                        << " [B" << std::setw(2) << cambpj[i] << "/" << std::setw(5) << camnpj[i] << "] [" <<
                        (boths ? 'X' : ' ') << "][" <<  cpdist1[i] << ", " << cpdist2[i] << "]\n";
                    __num_camera_modified++;
                }
                _camera_data[i].t[0] *= __depth_scaling;
                _camera_data[i].t[1] *= __depth_scaling;
                _camera_data[i].t[2] *= __depth_scaling;
            }
            for(int i = 0; i < _num_point; ++i)
            {
               /////////////////////////////////
                _point_data[4 *i + 0] *= __depth_scaling;
                _point_data[4 *i + 1] *= __depth_scaling;
                _point_data[4 *i + 2] *= __depth_scaling;
            }
        }
        if(__num_point_behind > 0)    std::cout << "WARNING: " << __num_point_behind << " points are behind camras.\n";
        if(__num_camera_modified > 0) std::cout << "WARNING: " << __num_camera_modified << " camera moved to avoid degeneracy.\n";
    }
}

void SparseBundleCPU::DenormalizeData()
{
    if(__focal_normalize && __focal_scaling!= 1.0f)
    {
        float squared_focal_factor = (__focal_scaling * __focal_scaling);
        for(int i = 0; i < _num_camera; ++i)
        {
            _camera_data[i].f /= __focal_scaling;
            if(__use_radial_distortion == -1) _camera_data[i].radial *= squared_focal_factor;
            _camera_data[i].distortion_type = __use_radial_distortion;
        }
        _projection_sse /= squared_focal_factor;
        __focal_scaling = 1.0f;
    }else if(__use_radial_distortion)
    {
        for(int i = 0;  i < _num_camera; ++i) _camera_data[i].distortion_type = __use_radial_distortion;
    }

    if(__depth_normalize && __depth_scaling != 1.0f)
    {
        for(int i = 0; i < _num_camera; ++i)
        {
            _camera_data[i].t[0] /= __depth_scaling;
            _camera_data[i].t[1] /= __depth_scaling;
            _camera_data[i].t[2] /= __depth_scaling;
        }
        for(int i = 0; i < _num_point; ++i)
        {
            _point_data[4 *i + 0] /= __depth_scaling;
            _point_data[4 *i + 1] /= __depth_scaling;
            _point_data[4 *i + 2] /= __depth_scaling;
        }
        __depth_scaling = 1.0f ;
    }
}

int SparseBundleCPU:: SolveNormalEquationPCGX(float lambda)
{
    //----------------------------------------------------------
    //(Jt * J + lambda * diag(Jt * J)) X = Jt * e
    //-------------------------------------------------------------
    TimerBA timer(this, TIMER_CG_ITERATION);    __recent_cg_status = ' ';

    //diagonal for jacobian preconditioning...
    int plen = GetParameterLength();
    VectorF null;
    VectorF& VectorDP =  __lm_use_diagonal_damp? _cuVectorJJ : null;    //diagonal
    ComputeBlockPC(lambda, __lm_use_diagonal_damp);

    ////////////////////////////////////////////////

    ///////////////////////////////////////////////////////
    //B = [BC 0 ; 0 BP]
    //m = [mc 0; 0 mp];
    //A x= BC * x - JcT * Jp * mp * JpT * Jc * x
    //   = JcT * Jc x + lambda * D * x + ........
    ////////////////////////////////////////////////////////////

    VectorF r; r.set(_cuVectorRK.data(), 8 * _num_camera);
    VectorF p; p.set(_cuVectorPK.data(), 8 * _num_camera);
    VectorF z; z.set(_cuVectorZK.data(), 8 * _num_camera);
    VectorF x; x.set(_cuVectorXK.data(), 8 * _num_camera);
    VectorF d; d.set(   VectorDP.data(), 8 * _num_camera);


    VectorF & u = _cuVectorRK;
    VectorF & v = _cuVectorPK;
    VectorF up; up.set(u.data()+ 8 * _num_camera, 3 * _num_point);
    VectorF vp; vp.set(v.data()+ 8 * _num_camera, 3 * _num_point);
    VectorF uc; uc.set(z.data(), 8 * _num_camera);

    VectorF & e = _cuVectorJX;
    VectorF & e2 = _cuImageProj;

    ApplyBlockPC(_cuVectorJtE, u, 2);
    ComputeJX(u, e, 2);
    ComputeJtE(e, uc, 1);
    ComputeSAXPY(double(-1.0f), uc, _cuVectorJtE, r);    //r
    ApplyBlockPC(r, p, 1);                      //z = p = M r


    float rtz0 = (float) ComputeVectorDot(r, p);    //r(0)' * z(0)
    ComputeJX(p, e, 1);                                                //Jc * x
    ComputeJtE(e, u, 2);                                               //JpT * jc * x
    ApplyBlockPC(u, v, 2);
    float qtq0 = (float) ComputeVectorNorm(e, __num_cpu_thread[FUNC_VS]);         //q(0)' * q(0)
    float pdp0 = (float) ComputeVectorNormW(p, d);     //p(0)' * DDD * p(0)
    float uv0  = (float) ComputeVectorDot(up, vp);
    float alpha0 = rtz0 / (qtq0 + lambda * pdp0 - uv0);

    if(__verbose_cg_iteration)    std::cout << " --0,\t alpha = " << alpha0 << ", t = " << BundleTimerGetNow(TIMER_CG_ITERATION) << "\n";
    if(!finite(alpha0))            {  return 0;    }
    if(alpha0 == 0)                {__recent_cg_status = 'I'; return 1; }

    ////////////////////////////////////////////////////////////
    ComputeSAX((double)alpha0, p, x);                //x(k+1) = x(k) + a(k) * p(k)
    ComputeJX(v, e2, 2);//                          //Jp * mp * JpT * JcT * p
    ComputeSAXPY(double(-1.0f), e2, e, e, __num_cpu_thread[FUNC_VV]);
    ComputeJtE(e, uc, 1);                            //JcT * ....
    ComputeSXYPZ((double)lambda, d, p, uc, uc);
    ComputeSAXPY((double) -alpha0, uc, r, r); // r(k + 1) = r(k) - a(k) * A * pk

    //////////////////////////////////////////////////////////////////////////
    float rtzk = rtz0, rtz_min = rtz0, betak;    int iteration = 1;
    ++__num_cg_iteration;

    while(true)
    {
        ApplyBlockPC(r, z, 1);

        ///////////////////////////////////////////////////////////////////////////
        float rtzp = rtzk;
        rtzk = (float) ComputeVectorDot(r, z);    //[r(k + 1) = M^(-1) * z(k + 1)] * z(k+1)
        float rtz_ratio = sqrt(fabs(rtzk / rtz0));
        if(rtz_ratio < __cg_norm_threshold )
        {
            if(__recent_cg_status == ' ') __recent_cg_status = iteration < std::min(10, __cg_min_iteration) ? '0' + iteration : 'N';
            if(iteration >= __cg_min_iteration) break;
        }
        ////////////////////////////////////////////////////////////////////////////
        betak = rtzk / rtzp;                                                                  //beta
        rtz_min = std::min(rtz_min, rtzk);

        ComputeSAXPY((double)betak, p, z, p);                    //p(k) = z(k) + b(k) * p(k - 1)
        ComputeJX(p, e, 1);                                     //Jc * p
        ComputeJtE(e, u, 2);                                    //JpT * jc * p
        ApplyBlockPC(u, v, 2);
        //////////////////////////////////////////////////////////////////////

        float qtqk = (float) ComputeVectorNorm(e, __num_cpu_thread[FUNC_VS]);        //q(k)' q(k)
        float pdpk = (float) ComputeVectorNormW(p, d);    //p(k)' * DDD * p(k)
        float uvk =  (float) ComputeVectorDot(up, vp);
        float alphak = rtzk / ( qtqk + lambda * pdpk - uvk);

        /////////////////////////////////////////////////////
        if(__verbose_cg_iteration) std::cout    << " --"<<iteration<<",\t alpha= " << alphak
                                                << ", rtzk/rtz0 = " << rtz_ratio << ", t = " << BundleTimerGetNow(TIMER_CG_ITERATION) << "\n";

        ///////////////////////////////////////////////////
        if(!finite(alphak) || rtz_ratio > __cg_norm_guard) {__recent_cg_status = 'X'; break; }//something doesn't converge..

        ////////////////////////////////////////////////
        ComputeSAXPY((double)alphak, p, x, x);        //x(k+1) = x(k) + a(k) * p(k)

        /////////////////////////////////////////////////
        ++iteration;        ++__num_cg_iteration;
        if(iteration >= std::min(__cg_max_iteration, plen)) break;


        ComputeJX(v, e2, 2);//                          //Jp * mp * JpT * JcT * p
        ComputeSAXPY((double)-1.0f, e2, e, e, __num_cpu_thread[FUNC_VV]);
        ComputeJtE(e, uc, 1);                            //JcT * ....
        ComputeSXYPZ((double)lambda, d, p, uc, uc);
        ComputeSAXPY((double) -alphak, uc, r, r);    // r(k + 1) = r(k) - a(k) * A * pk
     }

    ComputeJX(x, e, 1);
    ComputeJtE(e, u, 2);
    VectorF jte_p ;  jte_p.set(_cuVectorJtE.data() + 8 * _num_camera, _num_point * 3);
    ComputeSAXPY((double)-1.0f, up, jte_p, vp);
    ApplyBlockPC(v, _cuVectorXK, 2);
    return iteration;
}

int SparseBundleCPU:: SolveNormalEquationPCGB(float lambda)
{
    //----------------------------------------------------------
    //(Jt * J + lambda * diag(Jt * J)) X = Jt * e
    //-------------------------------------------------------------
    TimerBA timer(this, TIMER_CG_ITERATION);    __recent_cg_status = ' ';

    //diagonal for jacobian preconditioning...
    int plen = GetParameterLength();
    VectorF null;
    VectorF& VectorDP =  __lm_use_diagonal_damp? _cuVectorJJ : null;    //diagonal
    VectorF& VectorQK =  _cuVectorZK;    //temporary
    ComputeBlockPC(lambda, __lm_use_diagonal_damp);

    ////////////////////////////////////////////////////////
    ApplyBlockPC(_cuVectorJtE, _cuVectorPK);        //z(0) = p(0) = M * r(0)//r(0) = Jt * e
    ComputeJX(_cuVectorPK, _cuVectorJX);            //q(0) = J * p(0)

    //////////////////////////////////////////////////
    float rtz0 = (float) ComputeVectorDot(_cuVectorJtE, _cuVectorPK);    //r(0)' * z(0)
    float qtq0 = (float) ComputeVectorNorm(_cuVectorJX, __num_cpu_thread[FUNC_VS]);                        //q(0)' * q(0)
    float ptdp0 = (float) ComputeVectorNormW(_cuVectorPK, VectorDP);    //p(0)' * DDD * p(0)
    float alpha0 = rtz0 / (qtq0 + lambda * ptdp0);

    if(__verbose_cg_iteration)    std::cout << " --0,\t alpha = " << alpha0 << ", t = " << BundleTimerGetNow(TIMER_CG_ITERATION) << "\n";
    if(!finite(alpha0))            {  return 0;    }
    if(alpha0 == 0)                {__recent_cg_status = 'I'; return 1; }


    ////////////////////////////////////////////////////////////

    ComputeSAX((double) alpha0, _cuVectorPK, _cuVectorXK);                //x(k+1) = x(k) + a(k) * p(k)
    ComputeJtE(_cuVectorJX, VectorQK);                                    //Jt * (J * p0)

    ComputeSXYPZ((double)lambda, VectorDP, _cuVectorPK, VectorQK, VectorQK);    //Jt * J * p0 + lambda * DDD * p0

    ComputeSAXPY((double)-alpha0, VectorQK, _cuVectorJtE, _cuVectorRK);    //r(k+1) = r(k) - a(k) * (Jt * q(k)  + DDD * p(k)) ;

    float rtzk = rtz0, rtz_min = rtz0, betak;    int iteration = 1;
    ++__num_cg_iteration;

    while(true)
    {
        ApplyBlockPC(_cuVectorRK, _cuVectorZK);

        ///////////////////////////////////////////////////////////////////////////
        float rtzp = rtzk;
        rtzk = (float) ComputeVectorDot(_cuVectorRK, _cuVectorZK);    //[r(k + 1) = M^(-1) * z(k + 1)] * z(k+1)
        float rtz_ratio = sqrt(fabs(rtzk / rtz0));
        if(rtz_ratio < __cg_norm_threshold )
        {
            if(__recent_cg_status == ' ') __recent_cg_status = iteration < std::min(10, __cg_min_iteration) ? '0' + iteration : 'N';
            if(iteration >= __cg_min_iteration) break;
        }
        //////////////////////////////////////////////////////////////////////////
        betak = rtzk / rtzp;                                                                  //beta
        rtz_min = std::min(rtz_min, rtzk);

        ComputeSAXPY((double)betak, _cuVectorPK, _cuVectorZK, _cuVectorPK);                    //p(k) = z(k) + b(k) * p(k - 1)
        ComputeJX(_cuVectorPK, _cuVectorJX);                                                  //q(k) = J * p(k)
        //////////////////////////////////////////////////////////////////////

        float qtqk = (float) ComputeVectorNorm(_cuVectorJX, __num_cpu_thread[FUNC_VS]);                        //q(k)' q(k)
        float ptdpk = (float) ComputeVectorNormW(_cuVectorPK, VectorDP);    //p(k)' * DDD * p(k)

        float alphak = rtzk / ( qtqk + lambda * ptdpk);


        /////////////////////////////////////////////////////
        if(__verbose_cg_iteration) std::cout    << " --"<<iteration<<",\t alpha= " << alphak
                                                << ", rtzk/rtz0 = " << rtz_ratio << ", t = " << BundleTimerGetNow(TIMER_CG_ITERATION) << "\n";

        ///////////////////////////////////////////////////
        if(!finite(alphak) || rtz_ratio > __cg_norm_guard) {		__recent_cg_status = 'X'; break;	}//something doesn't converge..

        ////////////////////////////////////////////////
        ComputeSAXPY((double)alphak, _cuVectorPK, _cuVectorXK, _cuVectorXK);        //x(k+1) = x(k) + a(k) * p(k)

         /////////////////////////////////////////////////
        ++iteration;        ++__num_cg_iteration;
        if(iteration >= std::min(__cg_max_iteration, plen)) break;

        if(__cg_recalculate_freq > 0 && iteration % __cg_recalculate_freq == 0)
        {
            ////r = JtE - (Jt J + lambda * D) x
            ComputeJX(_cuVectorXK, _cuVectorJX);
            ComputeJtE(_cuVectorJX, VectorQK);
            ComputeSXYPZ((double)lambda, VectorDP, _cuVectorXK, VectorQK, VectorQK);
            ComputeSAXPY((double)-1.0f, VectorQK, _cuVectorJtE, _cuVectorRK);
        }else
        {
            ComputeJtE(_cuVectorJX, VectorQK);
            ComputeSXYPZ((double)lambda, VectorDP, _cuVectorPK, VectorQK, VectorQK);//
            ComputeSAXPY((double)-alphak, VectorQK, _cuVectorRK, _cuVectorRK);    //r(k+1) = r(k) - a(k) * (Jt * q(k)  + DDD * p(k)) ;
        }
    }
    return iteration;
}

int SparseBundleCPU::SolveNormalEquation(float lambda)
{
    if(__bundle_current_mode == BUNDLE_ONLY_MOTION)
    {
        ComputeBlockPC(lambda, __lm_use_diagonal_damp);
        ApplyBlockPC(_cuVectorJtE, _cuVectorXK, 1);
        return 1;
    }else if(__bundle_current_mode == BUNDLE_ONLY_STRUCTURE)
    {
        ComputeBlockPC(lambda, __lm_use_diagonal_damp);
        ApplyBlockPC(_cuVectorJtE, _cuVectorXK, 2);
        return 1;
    }else
    {
        ////solve linear system using Conjugate Gradients
        return __cg_schur_complement?  SolveNormalEquationPCGX(lambda) : SolveNormalEquationPCGB(lambda);
    }
}

void SparseBundleCPU::DumpCooJacobian()
{
    //////////
    std::ofstream jo("j.txt");
    int cn = __use_radial_distortion ? 8 : 7;
    int width = cn * _num_camera + 3 * _num_point;
    jo <<"%%MatrixMarket matrix coordinate real general\n";
    jo << (_num_imgpt * 2)  << " " << width << " " << (cn + 3) * _num_imgpt * 2 <<'\n';
    for(int i = 0; i < _num_imgpt; ++i)
    {
        int ci = _camera_idx[i];
        int pi = _point_idx[i];
        int row = i * 2 +1;
        //Float * jc = _cuJacobianCamera.data() + i * 16;
        //Float * jp = _cuJacobianPoint.data() + i * 6;
        int idx1 = ci * cn;
        int idx2 = _num_camera * cn + 3 * pi;

        for(int k = 0; k < 2; ++k, ++row)
        {
            for(int j = 0; j < cn; ++j)
            {
                jo << row << " " << (idx1 + j + 1) << " 1\n";
            }
            for(int j =0; j < 3; ++j)
            {
                jo << row << " " << (idx2 + j +1) << " 1\n";
            }
        }
    }

    std::ofstream jt("jt.txt");
    jt <<"%%MatrixMarket matrix coordinate real general\n";
    jt << width << " " << (_num_imgpt * 2)  << " " << (cn + 3) * _num_imgpt * 2<< '\n';

    int * lisc = &_cuCameraMeasurementList[0];
    int * mapc = &_cuCameraMeasurementMap[0];
    int * mapp = &_cuPointMeasurementMap[0];

    for(int i = 0; i < _num_camera; ++i)
    {
        int c0 = mapc[i];
        int c1 = mapc[i + 1];
        for(int k = 0; k < cn; ++k)
        {
            int row = i * cn + k + 1;
            for(int j = c0; j < c1; ++j)
                jt  << row << " " << ( lisc[j]* 2 +1) << " 1\n"
                    << row << " " << (2 * lisc[j] + 2) << " 1\n"; ;
        }
    }
    for(int i = 0; i < _num_point; ++i)
    {
        int p0 = mapp[i];
        int p1 = mapp[i + 1];
        for(int k = 0; k < 3; ++k)
        {
            int row = i * 3 + _num_camera * cn + k + 1;
            for(int j = p0; j < p1; ++j)
                jt  << row << " " << ( 2 * j +1) << " 1\n"
                    << row << " " << (2 * j + 2) << " 1\n"; ;
        }
    }
}

void SparseBundleCPU::RunTestIterationLM(bool reduced)
{
    EvaluateProjection(_cuCameraData, _cuPointData, _cuImageProj);
    EvaluateJacobians();
    ComputeJtE(_cuImageProj, _cuVectorJtE);
    if(reduced) SolveNormalEquationPCGX(__lm_initial_damp) ;
    else        SolveNormalEquationPCGB(__lm_initial_damp) ;
    UpdateCameraPoint(_cuVectorZK, _cuImageProj);
    ComputeVectorDot(_cuVectorXK, _cuVectorJtE);
    ComputeJX(_cuVectorXK,  _cuVectorJX);
    ComputeVectorNorm(_cuVectorJX, __num_cpu_thread[FUNC_VS]);
}

float SparseBundleCPU::UpdateCameraPoint(VectorF& dx, VectorF& cuImageTempProj)
{
    ConfigBA::TimerBA timer (this, TIMER_FUNCTION_UP);

    if(__bundle_current_mode == BUNDLE_ONLY_MOTION)
    {
        if(__jacobian_normalize)    ComputeVXY(_cuVectorXK, _cuVectorSJ, dx, 8 * _num_camera);
        ProgramCPU::UpdateCameraPoint(_num_camera, _cuCameraData, _cuPointData, dx, _cuCameraDataEX,
            _cuPointDataEX, __bundle_current_mode, __num_cpu_thread[FUNC_VV]);
        return EvaluateProjection(_cuCameraDataEX, _cuPointData, cuImageTempProj);
    }else if(__bundle_current_mode == BUNDLE_ONLY_STRUCTURE)
    {
        if(__jacobian_normalize)    ComputeVXY(_cuVectorXK, _cuVectorSJ, dx, _num_point * POINT_ALIGN, _num_camera * 8);
        ProgramCPU::UpdateCameraPoint(_num_camera, _cuCameraData, _cuPointData, dx, _cuCameraDataEX,
            _cuPointDataEX, __bundle_current_mode, __num_cpu_thread[FUNC_VV]);
        return EvaluateProjection(_cuCameraData, _cuPointDataEX, cuImageTempProj);
    }else
    {

        if(__jacobian_normalize)    ComputeVXY(_cuVectorXK, _cuVectorSJ, dx);
        ProgramCPU::UpdateCameraPoint(_num_camera, _cuCameraData, _cuPointData, dx, _cuCameraDataEX,
             _cuPointDataEX, __bundle_current_mode, __num_cpu_thread[FUNC_VV]);
        return EvaluateProjection(_cuCameraDataEX, _cuPointDataEX, cuImageTempProj);
    }
}

float SparseBundleCPU::SaveUpdatedSystem(float residual_reduction, float dx_sqnorm, float damping)
{
    float expected_reduction;
    if(__bundle_current_mode == BUNDLE_ONLY_MOTION)
    {
        VectorF xk;  xk.set(_cuVectorXK.data(), 8 * _num_camera);
        VectorF jte; jte.set(_cuVectorJtE.data(), 8 * _num_camera);
        float dxtg = (float) ComputeVectorDot(xk, jte);
        if(__lm_use_diagonal_damp)
        {
            VectorF jj; jj.set(_cuVectorJJ.data(), 8 * _num_camera);
            float dq = (float) ComputeVectorNormW(xk, jj);
            expected_reduction = damping * dq + dxtg;
        }else
        {
            expected_reduction = damping * dx_sqnorm + dxtg;
        }
        _cuCameraData.swap(_cuCameraDataEX);
    }else if(__bundle_current_mode == BUNDLE_ONLY_STRUCTURE)
    {
        VectorF xk;  xk.set(_cuVectorXK.data() + 8 * _num_camera, POINT_ALIGN * _num_point);
        VectorF jte; jte.set(_cuVectorJtE.data() + 8 * _num_camera, POINT_ALIGN * _num_point);
        float dxtg = (float) ComputeVectorDot(xk, jte);
        if(__lm_use_diagonal_damp)
        {
            VectorF jj; jj.set(_cuVectorJJ.data() + 8 * _num_camera, POINT_ALIGN * _num_point);
            float dq = (float) ComputeVectorNormW(xk, jj);
            expected_reduction = damping * dq + dxtg;
        }else
        {
            expected_reduction = damping * dx_sqnorm + dxtg;
        }
        _cuPointData.swap(_cuPointDataEX);
    }else
    {
        float dxtg = (float) ComputeVectorDot(_cuVectorXK, _cuVectorJtE);
        if(__accurate_gain_ratio)
        {
            ComputeJX(_cuVectorXK,  _cuVectorJX);
            float njx = (float) ComputeVectorNorm(_cuVectorJX, __num_cpu_thread[FUNC_VS]);
            expected_reduction = 2.0f * dxtg - njx;

            //could the expected reduction be negative??? not sure
            if(expected_reduction <= 0) expected_reduction = 0.001f * residual_reduction;
        }else    if(__lm_use_diagonal_damp)
        {
            float dq = (float) ComputeVectorNormW(_cuVectorXK, _cuVectorJJ);
            expected_reduction = damping * dq + dxtg;
        }else
        {
            expected_reduction = damping * dx_sqnorm + dxtg;
        }
        ///save the new motion/struture
        _cuCameraData.swap(_cuCameraDataEX);
        _cuPointData.swap(_cuPointDataEX);
    }
    ////////////////////////////////////////////
    return float(residual_reduction / expected_reduction);
}

void SparseBundleCPU::AdjustBundleAdjsutmentMode()
{
    if(__bundle_current_mode == BUNDLE_ONLY_MOTION)
    {
        _cuJacobianPoint.resize(0);
    }else if(__bundle_current_mode == BUNDLE_ONLY_STRUCTURE)
    {
        _cuJacobianCamera.resize(0);
        _cuJacobianCameraT.resize(0);
    }
}

float SparseBundleCPU::EvaluateDeltaNorm()
{
    if(__bundle_current_mode == BUNDLE_ONLY_MOTION)
    {
        VectorF temp; temp.set(_cuVectorXK.data(), 8 * _num_camera);
        return (float) ComputeVectorNorm(temp);
    }else if(__bundle_current_mode == BUNDLE_ONLY_STRUCTURE)
    {
        VectorF temp; temp.set(_cuVectorXK.data() +  8 * _num_camera, POINT_ALIGN * _num_point);
        return (float) ComputeVectorNorm(temp);
    }else
    {
        return (float) ComputeVectorNorm(_cuVectorXK);
    }
}

void SparseBundleCPU::NonlinearOptimizeLM()
{
    ////////////////////////////////////////
    TimerBA timer(this, TIMER_OPTIMIZATION);

    ////////////////////////////////////////////////
    float mse_convert_ratio = 1.0f / (_num_imgpt  * __focal_scaling  * __focal_scaling);
    float error_display_ratio = __verbose_sse ? _num_imgpt : 1.0f;
    const int edwidth = __verbose_sse ? 12 : 8;
    _projection_sse = EvaluateProjection(_cuCameraData, _cuPointData, _cuImageProj);
    __initial_mse = __final_mse = _projection_sse * mse_convert_ratio;

    //compute jacobian diagonals for normalization
    if(__jacobian_normalize) PrepareJacobianNormalization();

    //evalaute jacobian
    EvaluateJacobians();
    ComputeJtE(_cuImageProj, _cuVectorJtE);
    ///////////////////////////////////////////////////////////////
    if(__verbose_level > 0)
        std::cout    << "Initial " << (__verbose_sse ? "sumed" : "mean" )<<  " squared error = "
                    <<  __initial_mse  * error_display_ratio
                    << "\n----------------------------------------------\n";

    //////////////////////////////////////////////////
    VectorF& cuImageTempProj =   _cuVectorJX;
    //VectorF& cuVectorTempJX  =   _cuVectorJX;
    VectorF& cuVectorDX      =   _cuVectorSJ.size() ? _cuVectorZK : _cuVectorXK;

    //////////////////////////////////////////////////
    float damping_adjust = 2.0f, damping = __lm_initial_damp, g_norm, g_inf;
    SaveBundleRecord(0, _projection_sse * mse_convert_ratio, damping, g_norm, g_inf);

    ////////////////////////////////////
    std::cout << std::left;
    for(int i = 0;    i < __lm_max_iteration  &&  !__abort_flag ; __current_iteration = (++i))
    {
        ////solve linear system
        int num_cg_iteration = SolveNormalEquation(damping);

        //there must be NaN somewhere
        if(num_cg_iteration == 0)
        {
            if(__verbose_level > 0)
                std::cout << "#" << std::setw(3) << i <<" quit on numeric errors\n";
            __pba_return_code = 'E';
            break;
        }

        //there must be infinity somewhere
        if(__recent_cg_status == 'I')
        {
            std::cout    << "#" << std::setw(3) << i << " 0  I e="
                        << std::setw(edwidth) << "------- " << " u="
                        << std::setprecision(3) << std::setw(9) << damping << '\n'
                        << std::setprecision(6);
            /////////////increase damping factor
            damping = damping * damping_adjust;
            damping_adjust = 2.0f * damping_adjust;
            --i;     continue;
        }

        /////////////////////
        ++__num_lm_iteration;

        ////////////////////////////////////
        float dx_sqnorm = EvaluateDeltaNorm(), dx_norm = sqrt(dx_sqnorm);

        //In this library, we check absolute difference instead of realtive  difference
        if(dx_norm <= __lm_delta_threshold)
        {
            //damping factor must be way too big...or it converges
            if(__verbose_level > 1)     std::cout    << "#"  << std::setw(3) << i << " " << std::setw(3)
                                                    << num_cg_iteration << char(__recent_cg_status)
                                                    <<" quit on too small change ("    << dx_norm << "  < "
                                                    << __lm_delta_threshold << ")\n";
            __pba_return_code = 'S';
            break;
        }
        ///////////////////////////////////////////////////////////////////////
        //update structure and motion, check reprojection error
        float new_residual = UpdateCameraPoint(cuVectorDX, cuImageTempProj);
        float average_residual = new_residual * mse_convert_ratio;
        float residual_reduction  = _projection_sse - new_residual;

        //do we find a better solution?
        if(finite(new_residual) && residual_reduction > 0)
        {

            ////compute relative norm change
            float relative_reduction = 1.0f  - (new_residual/ _projection_sse);

            ////////////////////////////////////
            __num_lm_success++;                        //increase counter
            _projection_sse = new_residual;            //save the new residual
            _cuImageProj.swap(cuImageTempProj);    //save the new projection

            ////////////////////compute gain ratio///////////
            float gain_ratio = SaveUpdatedSystem(residual_reduction, dx_sqnorm, damping);

            ////////////////////////////////////////////////
            SaveBundleRecord(i + 1, _projection_sse * mse_convert_ratio, damping, g_norm, g_inf);

            /////////////////////////////////////////////
            if(__verbose_level > 1)
                std::cout    << "#" << std::setw(3) << i
                            << " " << std::setw(3) << num_cg_iteration << char(__recent_cg_status)
                            << " e=" << std::setw(edwidth) << average_residual * error_display_ratio
                            << " u=" << std::setprecision(3) << std::setw(9) << damping
                            << " r=" << std::setw(6) <<  floor(gain_ratio * 1000.f) * 0.001f
                            << " g=" << std::setw(g_norm > 0 ? 9 : 1) << g_norm
                            << " "  << std::setw(9) << relative_reduction
                            << ' ' << std::setw(9) << dx_norm
                            << " t=" << int(BundleTimerGetNow()) << "\n" << std::setprecision(6) ;

            /////////////////////////////
            if(!IsTimeBudgetAvailable())
            {
                if(__verbose_level > 1) std::cout << "#" << std::setw(3) << i <<" used up time budget.\n";
                __pba_return_code = 'T';
                break;
            }else if(__lm_check_gradient &&  g_inf < __lm_gradient_threshold)
            {
                if(__verbose_level > 1) std::cout << "#" << std::setw(3) << i <<" converged with small gradient\n";
                __pba_return_code = 'G';
                break;
            }else if(average_residual * error_display_ratio <= __lm_mse_threshold)
            {
                if(__verbose_level > 1) std::cout << "#" << std::setw(3) << i <<" satisfies MSE threshold\n";
                __pba_return_code = 'M';
                break;
            }else
            {
                /////////////////////////////adjust damping factor
                float temp = gain_ratio * 2.0f - 1.0f;
                float adaptive_adjust = 1.0f - temp * temp * temp; //powf(, 3.0f); //
                float auto_adjust =  std::max(1.0f / 3.0f, adaptive_adjust);

                //////////////////////////////////////////////////
                damping = damping * auto_adjust;    damping_adjust = 2.0f;
                if(damping < __lm_minimum_damp) damping = __lm_minimum_damp;
                else if(__lm_damping_auto_switch == 0 && damping > __lm_maximum_damp && __lm_use_diagonal_damp) damping = __lm_maximum_damp;

                EvaluateJacobians();
                ComputeJtE(_cuImageProj, _cuVectorJtE);
            }
        }else
        {
            if(__verbose_level > 1)
                std::cout    << "#" << std::setw(3) << i
                            << " " << std::setw(3) << num_cg_iteration  << char(__recent_cg_status)
                            << " e=" << std::setw(edwidth) << std::left << average_residual* error_display_ratio
                            << " u="  << std::setprecision(3) << std::setw(9) << damping
                            << " r=----- "
                            << (__lm_check_gradient || __save_gradient_norm ? " g=---------" : " g=0")
                            << " --------- " << std::setw(9) << dx_norm
                            << " t=" << int(BundleTimerGetNow()) <<"\n"     << std::setprecision(6);


            if(__lm_damping_auto_switch > 0 && __lm_use_diagonal_damp && damping > __lm_damping_auto_switch)
            {
                __lm_use_diagonal_damp = false;    damping = __lm_damping_auto_switch; damping_adjust = 2.0f;
                if(__verbose_level > 1) std::cout << "NOTE: switch to damping with an identity matix\n";
            }else
            {
                /////////////increase damping factor
                damping = damping * damping_adjust;
                damping_adjust = 2.0f * damping_adjust;
            }
        }

        if(__verbose_level == 1) std::cout << '.';

    }

    __final_mse = float(_projection_sse * mse_convert_ratio);
    __final_mse_x = __use_radial_distortion? EvaluateProjectionX(_cuCameraData, _cuPointData, _cuImageProj) * mse_convert_ratio : __final_mse;
}


#define PROFILE_REPORT2(A, T) \
                        std::cout << std::setw(24)<< A << ": " <<   (T) << "\n";

#define PROFILE_REPORT(A) \
                        std::cout << std::setw(24)<< A << ": " \
                        <<   (BundleTimerGet(TIMER_PROFILE_STEP) / repeat) << "\n";

#define PROFILE_(B)		BundleTimerStart(TIMER_PROFILE_STEP); \
                        for(int i = 0; i < repeat; ++i) { B; } \
                        BundleTimerSwitch(TIMER_PROFILE_STEP);



#define PROFILE(A, B)    PROFILE_(A B)	PROFILE_REPORT(#A)
#define PROXILE(A, B)    PROFILE_(B)	PROFILE_REPORT(A)
#define PROTILE(FID, A, B)  \
                     {\
                        float tbest = FLT_MAX; int nbest = 1; int nto = nthread[FID]; \
                        {std::ostringstream os1; os1 << #A "(" << nto << ")"; PROXILE(os1.str(), A B); }\
                        for(int j = 1; j <= THREAD_NUM_MAX; j *= 2)\
                        {\
                            nthread[FID] = j;   PROFILE_( A B);\
                            float t = BundleTimerGet(TIMER_PROFILE_STEP) / repeat;\
                            if(t > tbest) { if(j >= std::max(nto, 16)) break;}\
                            else {tbest = t;    nbest = j; }\
                        }\
                        if(nto != 0) nthread[FID] = nbest; \
                        {std::ostringstream os; os << #A "(" << nbest << ")";	PROFILE_REPORT2(os.str(), tbest);} \
                     }

#define PROTILE2(FID1, FID2, A, B) \
                     {\
                        int nt1 = nthread[FID1], nt2 = nthread[FID2]; \
                        {std::ostringstream os1; os1 << #A "(" << nt1 << "," << nt2 << ")"; PROXILE(os1.str(), A B); }\
                        float tbest = FLT_MAX; int nbest1 = 1, nbest2 = 1;\
                        nthread[FID2] = 1;  \
                        for(int j = 1; j <= THREAD_NUM_MAX; j *= 2)\
                        {\
                            nthread[FID1] = j;   PROFILE_(A B);\
                            float t = BundleTimerGet(TIMER_PROFILE_STEP) / repeat;\
                            if(t > tbest) { if(j >= std::max(nt1, 16)) break;}\
                            else {tbest = t;    nbest1 = j; }\
                        }\
                        nthread[FID1] = nbest1; \
                        for(int j = 2; j <= THREAD_NUM_MAX; j *= 2)\
                        {\
                            nthread[FID2] = j;   PROFILE_( A B);\
                            float t = BundleTimerGet(TIMER_PROFILE_STEP) / repeat;\
                            if(t > tbest) { if(j >= std::max(nt2, 16)) break;}\
                            else {tbest = t;    nbest2 = j; }\
                        }\
                        nthread[FID2] = nbest2;\
                        {std::ostringstream os; os << #A "(" << nbest1 << "," << nbest2 << ")";	PROFILE_REPORT2(os.str(), tbest); }\
                        if(nt1 == 0) nthread[FID1] = 0; if(nt2 == 0) nthread[FID2] = 0;\
                     }


void SparseBundleCPU::RunProfileSteps()
{
    const int repeat = std::max(__profile_pba, 1);
    int * nthread = __num_cpu_thread;
    std::cout << "---------------------------------\n"
                "|    Run profiling steps ("<<repeat<<")  |\n"
                 "---------------------------------\n" << std::left;  ;

    ///////////////////////////////////////////////
    EvaluateProjection(_cuCameraData, _cuPointData, _cuImageProj);
    if(__jacobian_normalize) PrepareJacobianNormalization();
    EvaluateJacobians(); ComputeJtE(_cuImageProj, _cuVectorJtE);
    ComputeBlockPC(__lm_initial_damp, true);
    ///////////////////////////////
    do
    {
        if(SolveNormalEquationPCGX(__lm_initial_damp) == 10 &&
          SolveNormalEquationPCGB(__lm_initial_damp) == 10) break;
        __lm_initial_damp *= 2.0f;
    }while(__lm_initial_damp < 1024.0f);
    std::cout << "damping set to " << __lm_initial_damp << " for profiling\n"
              << "---------------------------------\n" ;
    ///////////////////////
    {
        int repeat = 10, cgmin = __cg_min_iteration, cgmax = __cg_max_iteration;
        __cg_max_iteration = __cg_min_iteration = 10;
        __num_cg_iteration = 0;
        PROFILE(SolveNormalEquationPCGX, (__lm_initial_damp));
        if(__num_cg_iteration != 100) std::cout << __num_cg_iteration << " cg iterations in all\n";
        //////////////////////////////////////////////////////
        __num_cg_iteration = 0;
        PROFILE(SolveNormalEquationPCGB,(__lm_initial_damp));
        if(__num_cg_iteration != 100) std::cout << __num_cg_iteration << " cg iterations in all\n";
        std::cout << "---------------------------------\n";
        //////////////////////////////////////////////////////
        __num_cg_iteration = 0;
        PROXILE("Single iteration LMX", RunTestIterationLM(true));
        if(__num_cg_iteration != 100) std::cout << __num_cg_iteration << " cg iterations in all\n";
        //////////////////////////////////////////////////////
        __num_cg_iteration = 0;
        PROXILE("Single iteration LMB", RunTestIterationLM(false));
        if(__num_cg_iteration != 100) std::cout << __num_cg_iteration << " cg iterations in all\n";
        std::cout << "---------------------------------\n";
        __cg_max_iteration = cgmax; __cg_min_iteration = cgmin;
    }

    /////////////////////////////////////////////////////
    PROFILE(UpdateCameraPoint,(_cuVectorZK, _cuImageProj));
    PROFILE(ComputeVectorNorm, (_cuVectorXK));
    PROFILE(ComputeVectorDot, (_cuVectorXK, _cuVectorRK));
    PROFILE(ComputeVectorNormW, (_cuVectorXK, _cuVectorRK));
    PROFILE(ComputeSAXPY, ((double)0.01f, _cuVectorXK, _cuVectorRK, _cuVectorZK));
    PROFILE(ComputeSXYPZ, ((double)0.01f, _cuVectorXK, _cuVectorPK, _cuVectorRK, _cuVectorZK));
    std::cout << "---------------------------------\n";
    PROTILE(FUNC_VS, ComputeVectorNorm, (_cuImageProj, nthread[FUNC_VS])); 	//reset the parameter to 0

    ///////////////////////////////////////
    {
        avec temp1(_cuImageProj.size()), temp2(_cuImageProj.size());
        SetVectorZero(temp1);
        PROTILE(FUNC_VV, ComputeSAXPY, ((double)0.01f, _cuImageProj, temp1, temp2, nthread[FUNC_VV]));
    }

    std::cout << "---------------------------------\n";
    __multiply_jx_usenoj = false;

    ////////////////////////////////////////////////////
    PROTILE(FUNC_PJ, EvaluateProjection, (_cuCameraData, _cuPointData, _cuImageProj));
    PROTILE2(FUNC_MPC, FUNC_MPP, ApplyBlockPC, (_cuVectorJtE, _cuVectorPK));

    /////////////////////////////////////////////////
    if(!__no_jacobian_store )
    {
        if(__jc_store_original)
        {
            PROTILE(FUNC_JX, ComputeJX, (_cuVectorJtE, _cuVectorJX));

            if(__jc_store_transpose)
            {
                PROTILE(FUNC_JJ_JCO_JCT_JP,  EvaluateJacobians, ());
                PROTILE2(FUNC_JTEC_JCT, FUNC_JTEP, ComputeJtE, (_cuImageProj, _cuVectorJtE));
                PROTILE2(FUNC_BCC_JCT, FUNC_BCP, ComputeBlockPC, (0.001f, true));
                PROFILE(ComputeDiagonal, (_cuVectorPK));

                std::cout << "---------------------------------\n"
                             "|   Not storing original  JC    | \n"
                             "---------------------------------\n";
                __jc_store_original = false;
                PROTILE(FUNC_JJ_JCT_JP, EvaluateJacobians,());
                __jc_store_original = true;
            }

            //////////////////////////////////////////////////
            std::cout << "---------------------------------\n"
                         "|   Not storing transpose JC    | \n"
                         "---------------------------------\n";
            __jc_store_transpose = false;
            _cuJacobianCameraT.resize(0);
            PROTILE(FUNC_JJ_JCO_JP,  EvaluateJacobians, ());
            PROTILE2(FUNC_JTEC_JCO, FUNC_JTEP, ComputeJtE, (_cuImageProj, _cuVectorJtE));
            PROTILE2(FUNC_BCC_JCO, FUNC_BCP, ComputeBlockPC, (0.001f, true));
            PROFILE(ComputeDiagonal, (_cuVectorPK));
        }else if(__jc_store_transpose)
        {
            PROTILE2(FUNC_JTEC_JCT, FUNC_JTEP, ComputeJtE, (_cuImageProj, _cuVectorJtE));
            PROTILE2(FUNC_BCC_JCT, FUNC_BCP,  ComputeBlockPC, (0.001f, true));
            PROFILE(ComputeDiagonal, (_cuVectorPK));

            std::cout << "---------------------------------\n"
                         "|   Not storing original  JC    | \n"
                         "---------------------------------\n";
            PROTILE(FUNC_JJ_JCT_JP,  EvaluateJacobians, ());

        }
    }

    if(!__no_jacobian_store)
    {
        std::cout << "---------------------------------\n"
                     "| Not storing Camera Jacobians  | \n"
                     "---------------------------------\n";
        __jc_store_transpose = false;
        __jc_store_original = false;
        _cuJacobianCamera.resize(0);
        _cuJacobianCameraT.resize(0);
        PROTILE(FUNC_JJ_JP,  EvaluateJacobians, ());
        PROTILE(FUNC_JTE_, ComputeJtE, (_cuImageProj, _cuVectorJtE));
        //PROFILE(ComputeBlockPC, (0.001f, true));
    }

    ///////////////////////////////////////////////
    std::cout << "---------------------------------\n"
                 "|   Not storing any jacobians   |\n"
                 "---------------------------------\n";
    __no_jacobian_store = true;
    _cuJacobianPoint.resize(0);
    PROTILE(FUNC_JX_, ComputeJX, (_cuVectorJtE, _cuVectorJX));
    PROFILE(ComputeJtE, (_cuImageProj, _cuVectorJtE));
    PROFILE(ComputeBlockPC, (0.001f, true));
    std::cout <<  "---------------------------------\n";
}

int SparseBundleCPU::FindProcessorCoreNum()
{
    return omp_get_num_procs();
}

SFM_PBA_NAMESPACE_END
SFM_NAMESPACE_END
