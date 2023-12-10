#include <math.h>

/*
by default ctypes returns the C int type. in order to change this, you have to retype the output like below

mcc_lib = cdll.LoadLibrary("libmcchid.so")
value = 0x80d5
gain = 1 #BP_5V
mcc_lib.volts_USB1608G.restype = c_double
volts = mcc_lib.volts_USB1608G(daq.udev, gain, value)
print volts, type(volts)
*/
double evaluateCheb(double minMax[], double alpha[], double evalPts[],int problemDim, int nChebPts)
{
    int n[problemDim];

    for (int i = 0; i < problemDim; i++)
    {
        n[i] = nChebPts;
    }

    double approx = 0;

    double x_min = minMax[0];
    double x_max = minMax[1];

    double x = evalPts[0];

    double s1 = (2*x - (x_min + x_max))/(x_max-x_min);

    double Tx[n[0]];
    Tx[0] = 1;Tx[1] = s1;
    for (int k = 2; k < n[0]; k++)
    {
        Tx[k] = 2 * s1 * Tx[k - 1] - Tx[k - 2];
    }

    double Ty[n[1]];
    if (problemDim > 1)
    {
        double y_min = minMax[2];
        double y_max = minMax[3];

        double y = evalPts[1];

        double s2 = (2 * y - (y_min + y_max)) / (y_max - y_min);

        Ty[0] = 1; Ty[1] = s2;
        for (int k = 2; k < n[1]; k++)
        {
            Ty[k] = 2 * s2 * Ty[k - 1] - Ty[k - 2];
        }
    }

    double Tz[n[2]];
    if (problemDim > 2)
    {
        double z_min = minMax[4];
        double z_max = minMax[5];

        double z = evalPts[2];

        double s3 = (2 * z - (z_min + z_max)) / (z_max - z_min);

        Tz[0] = 1; Tz[1] = s3;
        for (int k = 2; k < n[0]; k++)
        {
            Tz[k] = 2 * s3 * Tz[k - 1] - Tz[k - 2];
        }
    }

    double Tvx[n[3]];
    if (problemDim > 3)
    {
        double vx_min = minMax[6];
        double vx_max = minMax[7];

        double vx = evalPts[3];
        double s4 = (2*vx - (vx_min + vx_max))/(vx_max-vx_min);

        Tvx[0] = 1; Tvx[1] = s4;
        for (int k = 2; k < n[3]; k++)
        {
            Tvx[k] = 2 * s4 * Tvx[k - 1] - Tvx[k - 2];
        }
    }

    if (problemDim > 4)
    {
        /* pass for now */
    }

    if (problemDim > 5)
    {
        /* pass for now  */
    }

    if (problemDim == 1)
    {
        for (int i = 0; i < n[0]; i++)
        {
            approx = approx + alpha[i] * Tx[i];
        }
    }
    
    else if (problemDim == 2)
    {
        for (int i = 0; i < n[0]; i++)
        {
            for (int j = 0; j < n[1]; j++)
            {
                approx = approx + alpha[j*nChebPts+ i] * Tx[i] * Ty[j];
            }
        }
    }
    
    else if (problemDim == 3)
    {
        for (int i = 0; i < n[0]; i++)
        {
            for (int j = 0; j < n[1]; j++)
            {
                for (int k = 0; k < n[2]; k++)
                {
                    approx = approx + alpha[i * nChebPts * nChebPts + j * nChebPts + k] * Tx[i] * Ty[j] * Tz[k];
                }   
            }
        }
    }
    
    else if (problemDim == 4)
    {
        for (int i = 0; i < n[0]; i++)
        {
            for (int j = 0; j < n[1]; j++)
            {
                for (int k = 0; k < n[2]; k++)
                {
                    for (int ii = 0; ii < n[3]; ii++)
                    {
                        approx = approx + alpha[ii * nChebPts * nChebPts * nChebPts + k * nChebPts * nChebPts + j * nChebPts + i] * Tx[i] * Ty[j] * Tz[k] * Tvx[ii];
                    }
                }
            }
        }
    }

    else if (problemDim == 5)
    {
        /* pass */
    }
    
    else if (problemDim == 6)
    {
        /* pass */
    }

    else
    {
        /* pass */
    }
    

    return approx;

}

