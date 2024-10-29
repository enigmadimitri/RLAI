#include <fstream>
#include <iostream>
#include <random>
#include <string>

int main()
{
    int k;
    std::random_device random_device;
    std::mt19937 generator{random_device()};
    std::normal_distribution<double> normal_distribution;

    std::cout << "Enter the value of k:" << std::endl;
    std::cin >> k;
    
    std::string filename = "data/samples/" + std::to_string(k) + "_normal_distribution_samples.data";
    std::ofstream file(filename);
    for(int i = 0; i < k - 1; i++)
    {
        file << normal_distribution(generator) << std::endl;
    }
    file << normal_distribution(generator);
    file.close();
    return 0;
}