/* disloc.c -- Computes surface displacements for dislocations in an elastic half-space.
   Based on code by Y. Okada.
   
   Version 1.2, 10/28/2000

   Record of revisions:

   Date          Programmer            Description of Change
   ====          ==========            =====================
   10/28/2000    Peter Cervelli        Removed seldom used 'reference station' option; improved
                                       detection of dip = integer multiples of pi/2.
   09/01/2000    Peter Cervelli        Fixed a bug that incorrectly returned an integer absolute value
                                       that created a discontinuity for dip angle of +-90 - 91 degrees.
                                       A genetically related bug incorrectly assigned a value of 1 to
                                       sin(-90 degrees).
   08/25/1998    Peter Cervelli        Original Code


*/

#include <math.h>

#define DEG2RAD 0.017453292519943295L
#define PI2INV 0.15915494309189535L

void Okada(double *pSS, double *pDS, double *pTS, double alp, double sd, double cd, double len, double wid,
           double dep, double X, double Y, double SS, double DS, double TS)
{
     double depsd, depcd, x, y, ala[2], awa[2], et, et2, xi, xi2, q2, r, r2, r3, p, q, sign;
     double a1, a3, a4, a5, d, ret, rd, tt, re, dle, rrx, rre, rxq, rd2, td, a2, req, sdcd, sdsd, mult;
     int j, k;

     ala[0] = len;
     ala[1] = 0.0;
     awa[0] = wid;
     awa[1] = 0.0;
     sdcd = sd * cd;
     sdsd = sd * sd;
     depsd = dep * sd;
     depcd = dep * cd;

     p = Y * cd + depsd;
     q = Y * sd - depcd;

     for (k = 0; k <= 1; k++)
     {
          et = p - awa[k];
          for (j = 0; j <= 1; j++)
          {
               sign = PI2INV;
               xi = X - ala[j];
               if (j + k == 1)
                    sign = -PI2INV;
               xi2 = xi * xi;
               et2 = et * et;
               q2 = q * q;
               r2 = xi2 + et2 + q2;
               r = sqrt(r2);
               r3 = r * r2;
               d = et * sd - q * cd;
               y = et * cd + q * sd;
               ret = r + et;
               if (ret < 0.0)
                    ret = 0.0;
               rd = r + d;
               if (q != 0.0)
                    tt = atan(xi * et / (q * r));
               else
                    tt = 0.0;
               if (ret != 0.0)
               {
                    re = 1 / ret;
                    dle = log(ret);
               }
               else
               {
                    re = 0.0;
                    dle = -log(r - et);
               }
               rrx = 1 / (r * (r + xi));
               rre = re / r;
               if (cd == 0.0)
               {
                    rd2 = rd * rd;
                    a1 = -alp / 2 * xi * q / rd2;
                    a3 = alp / 2 * (et / rd + y * q / rd2 - dle);
                    a4 = -alp * q / rd;
                    a5 = -alp * xi * sd / rd;
               }
               else
               {
                    td = sd / cd;
                    x = sqrt(xi2 + q2);
                    if (xi == 0.0)
                         a5 = 0;
                    else
                         a5 = alp * 2 / cd * atan( (et * (x + q * cd) + x * (r + x) * sd) / (xi * (r + x) * cd) );

                    a4 = alp / cd * (log(rd) - sd * dle);
                    a3 = alp * (y / rd / cd - dle) + td * a4;
                    a1 = -alp / cd * xi / rd - td * a5;
               }

               a2 = -alp * dle - a3;
               req = rre * q;
               rxq = rrx * q;

               if (SS != 0)
               {
                    mult = sign * SS;
                    pSS[0] -= mult * (req * xi + tt + a1 * sd);
                    pSS[1] -= mult * (req * y + q * cd * re + a2 * sd);
                    pSS[2] -= mult * (req * d + q * sd * re + a4 * sd);
               }

               if (DS != 0)
               {
                    mult = sign * DS;
                    pDS[0] -= mult *(q / r - a3 * sdcd);
                    pDS[1] -= mult * (y * rxq + cd * tt - a1 * sdcd);
                    pDS[2] -= mult * (d * rxq + sd * tt - a5 * sdcd);
               }
               if (TS != 0)
               {
                    mult = sign * TS;
                    pTS[0] += mult * (q2 * rre - a3 * sdsd);
                    pTS[1] += mult * (-d * rxq - sd * (xi * q * rre - tt) - a1 * sdsd);
                    pTS[2] += mult * (y * rxq + cd * (xi * q * rre - tt) - a5 * sdsd);
               }
          }
     }
}

void Disloc(double *pOutput, double *pModel, double *pCoords, double nu, int NumStat, int NumDisl)
{
     int i,j, sIndex, dIndex, kIndex;
     double sd, cd, Angle, cosAngle, sinAngle, SS[3],DS[3],TS[3], x, y;


     /*Loop through dislocations*/

     for (i=0; i< NumDisl; i=i++)
     {
          dIndex=i*10;

          cd = cos(pModel[dIndex+3] * DEG2RAD);
          sd = sin(pModel[dIndex+3] * DEG2RAD);

          if (pModel[0]<0 || pModel[1]<0 || pModel[2]<0 || (pModel[2]-sin(pModel[3]*DEG2RAD)*pModel[1])<-1e-12)
          {
               printf("Warning: model %d is not physical. It will not contribute to the deformation.\n",i+1);
               continue;
          }

          if (fabs(cd)<2.2204460492503131e-16)
          {
               cd=0;
               if (sd>0)
                    sd=1;
               else
                    sd=0;
          }

          Angle = -(90 - pModel[dIndex+4]) * DEG2RAD;
          cosAngle = cos(Angle);
          sinAngle = sin(Angle);

          /*Loop through stations*/

          for(j=0; j < NumStat; j++)
          {
               SS[0] = SS[1] = SS[2] = 0;
               DS[0] = DS[1] = DS[2] = 0;
               TS[0] = TS[1] = TS[2] = 0;

               sIndex = j*2;
               kIndex = j*3;

               Okada(&SS[0],&DS[0],&TS[0],1 - 2 * nu,sd,cd,pModel[dIndex],pModel[dIndex+1],pModel[dIndex+2],
                     cosAngle * (pCoords[sIndex] - pModel[dIndex+5]) - sinAngle * (pCoords[sIndex + 1] - pModel[dIndex+6]) +  0.5 * pModel[dIndex],
                     sinAngle * (pCoords[sIndex] - pModel[dIndex+5]) + cosAngle * (pCoords[sIndex + 1] - pModel[dIndex+6]),
                     pModel[dIndex+7], pModel[dIndex+8], pModel[dIndex+9]);

               if (pModel[dIndex+7])
               {
                    x=SS[0];
                    y=SS[1];
                    SS[0] = cosAngle * x + sinAngle * y;
                    SS[1] = -sinAngle * x + cosAngle * y;
                    pOutput[kIndex]+=SS[0];
                    pOutput[kIndex+1]+=SS[1];
                    pOutput[kIndex+2]+=SS[2];
               }

               if (pModel[dIndex+8])
               {
                    x=DS[0];
                    y=DS[1];
                    DS[0] = cosAngle * x + sinAngle * y;
                    DS[1] = -sinAngle * x + cosAngle * y;
                    pOutput[kIndex]+=DS[0];
                    pOutput[kIndex+1]+=DS[1];
                    pOutput[kIndex+2]+=DS[2];
               }

               if (pModel[dIndex+9])
               {
                    x=TS[0];
                    y=TS[1];
                    TS[0] = cosAngle * x + sinAngle * y;
                    TS[1] = -sinAngle * x + cosAngle * y;
                    pOutput[kIndex]+=TS[0];
                    pOutput[kIndex+1]+=TS[1];
                    pOutput[kIndex+2]+=TS[2];

               }
          }
     }
}