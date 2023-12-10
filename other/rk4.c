#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void f(double t, double y[], double fReturn[])
{
    // RHS function
    fReturn[0] = y[1];                // RHS of first equation
    fReturn[1] = -1.0 * y[0]; // RHS of second equations
}

void out2Term(double t, double y[], int problemDim)
{
    /*
    Debug function to print differential equation solution to console.
    */
    int i;
    printf("%f\t", t);
    for (i = 0; i < problemDim; i++)
    {
        printf("%f\t", y[i]);
    }
    printf("\n");
}

void rk4(void func(double, double[] ,double[]), double *IC, double a, double b, double h, int problemDim, double *sol)
{
    /*
    This function implements a Runge-Kutta 4th order integrator in C.
    Inputs: func - function pointer of the differential equation
            IC - double array that holds the initial conditions of the desired solution\
            a - start time of integration in seconds
            b - end time of integration in seconds 
            h - time step size in seconds
            problemDim - integer that correlates to the number of coupled differential equations
                in the problem. this can be removed in the future by extracting the size of the array of IC 
                I can't do this though...
            sol - an array that stores the solution of integration, see either C main or python main for correct variable instantiation
    Outputs: none

    Author: Hunter Quebedeaux
           6/14/23 
    */

    double t = a; int i;

    double y[problemDim], fReturn[problemDim], yy[problemDim];

    for (i = 0;i<problemDim;i++)
        y[i] = IC[i];

    double largestTime;
    if(a > b)
        largestTime = a;
    else
        largestTime = b;

    int length = fabs(largestTime/h);
    // printf("%d", length);

    int row = 0;

    for (i = 0; i < problemDim; i++)
    {
        sol[row * problemDim + i] = y[i];
    }

    out2Term(t,y,problemDim);

    while (row < length)
    {
        row++;
        double k1[problemDim], k2[problemDim], k3[problemDim], k4[problemDim];

        func(t,y,fReturn);
        for (i = 0; i < problemDim; i++)
        {
            k1[i] = h * fReturn[i];

            //update variables for next eval step
            yy[i] = y[i] + 0.5 * k1[i];
        }

        func(t + 0.5 * h, yy, fReturn);
        for (i = 0; i < problemDim; i++)
        {
            k2[i] = h * fReturn[i];
            yy[i] = (y[i] + 0.5 * k2[i]);
        }
        
        func((t+0.5*h),yy,fReturn);
        for (i = 0; i < problemDim; i++)
        {
            k3[i] = h * fReturn[i];
            yy[i] = y[i] + k3[i];
        }

        
        func((t+h),yy,fReturn);
        for (i = 0; i < problemDim; i++)
        {
            k4[i] = h * fReturn[i];
            y[i] = y[i] + (k1[i]+2*k2[i]+2*k3[i]+k4[i])/6;
        }

        for (i = 0; i < problemDim; i++)
        {
            sol[row * problemDim + i] = y[i];
        }

        t = t + h;
        out2Term(t,y,problemDim);
    }
}

void rk4Final(void func(double, double[], double[]), double *IC, double a, double b, double h, int problemDim, double *finalSol)
{
    /*
    This function implements a Runge-Kutta 4th order integrator in C that outputs only the final propagation vales.
    Inputs: func - function pointer of the differential equation
            IC - double array that holds the initial conditions of the desired solution\
            a - start time of integration in seconds
            b - end time of integration in seconds
            h - time step size in seconds
            problemDim - integer that correlates to the number of coupled differential equations
                in the problem. this can be removed in the future by extracting the size of the array of IC
                I can't do this though...
            finalSol - an array that stores the final solution of integration, see either C main or python main for correct variable instantiation
    Outputs: none

    Author: Hunter Quebedeaux
           6/15/23
    */

    double t = a;
    int i;

    double y[problemDim], fReturn[problemDim], yy[problemDim];

    for (i = 0; i < problemDim; i++)
        y[i] = IC[i];

    double largestTime;
    if (a > b)
        largestTime = a;
    else
        largestTime = b;

    int length = fabs(largestTime / h);
    // printf("%d", length);

    int row = 0;


    while (row < length)
    {
        row++;
        double k1[problemDim], k2[problemDim], k3[problemDim], k4[problemDim];

        func(t, y, fReturn);
        for (i = 0; i < problemDim; i++)
        {
            k1[i] = h * fReturn[i];

            // update variables for next eval step
            yy[i] = y[i] + 0.5 * k1[i];
        }

        func(t + 0.5 * h, yy, fReturn);
        for (i = 0; i < problemDim; i++)
        {
            k2[i] = h * fReturn[i];
            yy[i] = (y[i] + 0.5 * k2[i]);
        }

        func((t + 0.5 * h), yy, fReturn);
        for (i = 0; i < problemDim; i++)
        {
            k3[i] = h * fReturn[i];
            yy[i] = y[i] + k3[i];
        }

        func((t + h), yy, fReturn);
        for (i = 0; i < problemDim; i++)
        {
            k4[i] = h * fReturn[i];
            y[i] = y[i] + (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6;
        }

        t = t + h;
    }
    for (i = 0; i < problemDim; i++)
    {
        finalSol[i] = y[i];
    }
}

int main(void)
{
    // example of how to use in C
    int problemDim = 2;
    double a = 0;
    double b = 7;
    double h = 0.1;

    double largestTime;
    if (a > b)
        largestTime = a;
    else
        largestTime = b;
    int numSteps = fabs(largestTime / h);

    double y[problemDim];
    y[0] = 1;
    y[1] = 0;

    double *sol = (double *)malloc(numSteps * problemDim * sizeof(double));
    double finalSol[problemDim];
    rk4(f, y, a, b, h, problemDim, sol);
    rk4Final(f, y, a, b, h, problemDim, finalSol);

    printf("%f:%f", finalSol[0], finalSol[1]);

    free(sol);
    return 0;
}