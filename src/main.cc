#include "run.h"

#include <iostream>

int main()
{
    std::random_device random_device;
    std::mt19937* generator;
    generator = new std::mt19937{random_device()};
    std::normal_distribution<double>* normal_distribution;
    normal_distribution = new std::normal_distribution<double>{0.0,1.0};
    int k;
    int n;
    int T;
    double alpha;
    double epsilon;
    double initial_value;
    std::cout << "Value for k?" << std::endl;
    std::cin >> k;
    std::cout << "Value for n?" << std::endl;
    std::cin >> n;
    std::cout << "Value for T?" << std::endl;
    std::cin >> T;
    std::cout << "Value for alpha?" << std::endl;
    std::cin >> alpha;
    std::cout << "Value for epsilon?" << std::endl;
    std::cin >> epsilon;
    std::cout << "Value for initial value?" << std::endl;
    std::cin >> initial_value;
    std::vector<double> means;
    for (int i = 0; i < k; i++)
    {
        means.push_back((*normal_distribution)(*generator));
    }
    multi_armed_bandits mab(means);
    run r(n, 
          T, 
          alpha,
          epsilon, 
          initial_value,
          mab, 
          normal_distribution, 
          generator);
    for (int num = 0; num < n; num++)
    {
        r.episode();
        for (int i = 0; i < k; i++)
        {
            means.at(i) = (*normal_distribution)(*generator);
        }
        mab.new_means(means);
        r.reset(mab);
    }
    r.write();
}