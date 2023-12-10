#include <math.h>

void linPend(double t, double y[], double fReturn[])
{
    // RHS function
    fReturn[0] = y[1];        // RHS of first equation
    fReturn[1] = -1.0 * y[0]; // RHS of second equations
}

void doubleLinPend(double t, double y[], double fReturn[])
{
    // RHS function
    fReturn[0] = y[1];        // RHS of first equation
    fReturn[1] = -1.0 * y[0]; // RHS of second equations
    fReturn[2] = y[3];        // RHS of third equation
    fReturn[3] = -1.0 * y[2]; // RHS of fourth equations
}

void twobodyScaled(double t, double y[], double fReturn[])
{
    int mu = 1;
    double r = sqrt(y[0] * y[0] + y[1] * y[1]);
    
    fReturn[0] = y[2]; // RHS of first equation
    fReturn[1] = y[3]; // RHS of second equations
    fReturn[2] = -(mu / pow(r,3)) * y[0]; // RHS of third equation
    fReturn[3] = -(mu / pow(r,3)) * y[1]; // RHS of fourth equations
}