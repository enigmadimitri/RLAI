#include "epsilon_greedy_run.h"

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
    double epsilon;
    std::cout << "Value for k?" << std::endl;
    std::cin >> k;
    std::cout << "Value for n?" << std::endl;
    std::cin >> n;
    std::cout << "Value for T?" << std::endl;
    std::cin >> T;
    std::cout << "Value for epsilon?" << std::endl;
    std::cin >> epsilon;
    std::vector<double> means;
    for (int i = 0; i < k; i++)
    {
        means.push_back((*normal_distribution)(*generator));
    }
    multi_armed_bandits mab(means);
    epsilon_greedy_run egr(n, 
                           T, 
                           epsilon, 
                           mab, 
                           normal_distribution, 
                           generator);
    for (int t = 0; t < T; t++)
    {
        egr.episode();
        for (int i = 0; i < k; i++)
        {
            means.at(i) = (*normal_distribution)(*generator);
        }
        mab.new_means(means);
        egr.reset(mab);
    }
    egr.write();
}