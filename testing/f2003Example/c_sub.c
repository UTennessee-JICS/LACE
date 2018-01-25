#include <cstdlib>
#include <iostream>

using namespace std;

extern "C" 
{
    void call_fc_(int **x, int s);
    void c_func_deallocate_(int **x);
}

void call_fc_(int **x, int s)
{
    int i;
    int *y = (int *) malloc(sizeof(int)*s);
    for(i = 0; i < 100 && i < s; i++)
    {
        cout << i << endl;
        y[i]=i;
    }
    (*x) = y;
}

void c_func_deallocate_(int **x)
{
    free(*x);
}
