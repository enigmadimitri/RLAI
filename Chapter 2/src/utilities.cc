#include "utilities.h"

double max(std::vector<double> a){
    double dmax = a.at(0);
    for (int i = 0; i < (int) a.size(); i++)
    {
        if (a.at(i) > dmax)
        {
            dmax = a.at(i);
        }
    }
    return dmax;
}

std::vector<int> argmax(std::vector<double> a){
    double dmax = max(a);
    std::vector<int> result;
    for (int i = 0; i < (int) a.size(); i++)
    {
        if (a.at(i) == dmax)
        {
            result.push_back(i);
        }
    }
    return result;
}