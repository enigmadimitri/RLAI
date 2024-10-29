#include "run.h"

#include <fstream>
#include <iostream>

int main()
{
    std::random_device random_device;
    std::mt19937* generator;
    generator = new std::mt19937{random_device()};
    std::normal_distribution<double>* normal_distribution;
    normal_distribution = new std::normal_distribution<double>{0.0,1.0};
    bool benchmark;
    std::cout << "Benchmark?" << std::endl;
    std::cin >> benchmark;
    bool baseline;
    bool random_walk;
    int k;
    int n;
    int T;
    int t_initial;
    double alpha;
    double alpha_gradient_bandit;
    double c;
    double epsilon;
    double initial_value;
    double normal_mean;
    if (benchmark)
    {
        baseline = 1;
        k = 10;
        std::cout << "Value for n?" << std::endl;
        std::cin >> n;
        std::cout << "Value for t initial?" << std::endl;
        std::cin >> t_initial;
        std::cout << "Value for T?" << std::endl;
        std::cin >> T;
        alpha = 0;
        alpha_gradient_bandit = 0;
        c = 0;
        epsilon = 0;
        initial_value = 0;
        normal_mean = 0;
        std::vector<double> alpha_gradient_bandits = {1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0/2, 1, 2, 3};
        std::vector<double> cs = {1.0/16, 1.0/8, 1.0/4, 1.0/2, 1, 2, 4};
        std::vector<double> epsilons = {1.0/128, 1.0/64, 1.0/32, 1.0/16, 1.0/8, 1.0/4};
        std::vector<double> initial_values = {1.0/4, 1.0/2, 1, 2, 4};
        std::string avg_epsilons_file_name = "data/benchmark/epsilons_" + std::to_string(T) + "_"
                                                                        + std::to_string(t_initial) + ".data";
        std::ofstream avg_epsilons_file(avg_epsilons_file_name);
        avg_epsilons_file << "epsilon average_reward" << std::endl;
        for (int i = 0; i < (int) epsilons.size(); i++)
        {
            std::vector<double> means;
            for (int i = 0; i < k; i++)
            {
                means.push_back(normal_mean);
            }
            multi_armed_bandits mab(means);
            run r(baseline,
                  true,
                  n, 
                  T, 
                  alpha,
                  alpha_gradient_bandit,
                  c,
                  epsilons[i], 
                  initial_value,
                  mab, 
                  normal_distribution, 
                  generator);
            for (int num = 0; num < n; num++)
            {
                r.episode();
                for (int i = 0; i < k; i++)
                {
                    means.at(i) = normal_mean;
                }
                mab.new_means(means);
                r.reset(mab);
            }
            double super_reward_average = r.super_reward_average(t_initial);
            avg_epsilons_file << i << " " << super_reward_average << std::endl;
        }
        avg_epsilons_file.close();
        std::string avg_epsilons_alpha_file_name = "data/benchmark/epsilons_alpha_" + std::to_string(T) + "_"
                                                                                    + std::to_string(t_initial) + ".data";
        std::ofstream avg_epsilons_alpha_file(avg_epsilons_alpha_file_name);
        avg_epsilons_alpha_file << "epsilon average_reward" << std::endl;
        for (int i = 0; i < (int) epsilons.size(); i++)
        {
            std::vector<double> means;
            for (int i = 0; i < k; i++)
            {
                means.push_back(normal_mean);
            }
            multi_armed_bandits mab(means);
            run r(baseline,
                  true,
                  n, 
                  T, 
                  0.1,
                  alpha_gradient_bandit,
                  c,
                  epsilons[i], 
                  initial_value,
                  mab, 
                  normal_distribution, 
                  generator);
            for (int num = 0; num < n; num++)
            {
                r.episode();
                for (int i = 0; i < k; i++)
                {
                    means.at(i) = normal_mean;
                }
                mab.new_means(means);
                r.reset(mab);
            }
            double super_reward_average = r.super_reward_average(t_initial);
            avg_epsilons_alpha_file << i << " " << super_reward_average << std::endl;
        }
        avg_epsilons_alpha_file.close();
        std::string avg_alpha_gradient_bandits_file_name = "data/benchmark/alpha_gradient_bandits_" + std::to_string(T) + "_"
                                                                                                    + std::to_string(t_initial) + ".data";
        std::ofstream avg_alpha_gradient_bandits_file(avg_alpha_gradient_bandits_file_name);
        avg_alpha_gradient_bandits_file << "alpha_gradient_bandit average_reward" << std::endl;
        for (int i = 0; i < (int) alpha_gradient_bandits.size(); i++)
        {
            std::vector<double> means;
            for (int i = 0; i < k; i++)
            {
                means.push_back(normal_mean);
            }
            multi_armed_bandits mab(means);
            run r(baseline,
                  true,
                  n, 
                  T, 
                  alpha,
                  alpha_gradient_bandits[i],
                  c,
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
                    means.at(i) = normal_mean;
                }
                mab.new_means(means);
                r.reset(mab);
            }
            double super_reward_average = r.super_reward_average(t_initial);
            if (alpha_gradient_bandits[i] == 3)
            {
                avg_alpha_gradient_bandits_file << 8.5 << " " << super_reward_average << std::endl;
            }
            else
            {
                avg_alpha_gradient_bandits_file << i + 2 << " " << super_reward_average << std::endl;
            }
        }
        avg_alpha_gradient_bandits_file.close();
        std::string avg_cs_file_name = "data/benchmark/cs_" + std::to_string(T) + "_"
                                                            + std::to_string(t_initial) + ".data";
        std::ofstream avg_cs_file(avg_cs_file_name);
        avg_cs_file << "c average_reward" << std::endl;
        for (int i = 0; i < (int) cs.size(); i++)
        {
            std::vector<double> means;
            for (int i = 0; i < k; i++)
            {
                means.push_back(normal_mean);
            }
            multi_armed_bandits mab(means);
            run r(baseline,
                  true,
                  n, 
                  T, 
                  alpha,
                  alpha_gradient_bandit,
                  cs[i],
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
                    means.at(i) = normal_mean;
                }
                mab.new_means(means);
                r.reset(mab);
            }
            double super_reward_average = r.super_reward_average(t_initial);
            avg_cs_file << i + 3 << " " << super_reward_average << std::endl;
        }
        avg_cs_file.close();
        std::string avg_initial_values_file_name = "data/benchmark/initial_values_" + std::to_string(T) + "_"
                                                                                    + std::to_string(t_initial) + ".data";
        std::ofstream avg_initial_values_file(avg_initial_values_file_name);
        avg_initial_values_file << "initial_value average_reward" << std::endl;
        for (int i = 0; i < (int) initial_values.size(); i++)
        {
            std::vector<double> means;
            for (int i = 0; i < k; i++)
            {
                means.push_back(normal_mean);
            }
            multi_armed_bandits mab(means);
            run r(baseline,
                  true,
                  n, 
                  T, 
                  0.1,
                  alpha_gradient_bandit,
                  c,
                  epsilon, 
                  initial_values[i],
                  mab, 
                  normal_distribution, 
                  generator);
            for (int num = 0; num < n; num++)
            {
                r.episode();
                for (int i = 0; i < k; i++)
                {
                    means.at(i) = normal_mean;
                }
                mab.new_means(means);
                r.reset(mab);
            }
            double super_reward_average = r.super_reward_average(t_initial);
            avg_initial_values_file << i + 5 << " " << super_reward_average << std::endl;
        }
        avg_initial_values_file.close();
    }
    else
    {
        std::cout << "Value for baseline?" << std::endl;
        std::cin >> baseline;
        std::cout << "Value for random walk?" << std::endl;
        std::cin >> random_walk;
        std::cout << "Value for k?" << std::endl;
        std::cin >> k;
        std::cout << "Value for n?" << std::endl;
        std::cin >> n;
        std::cout << "Value for T?" << std::endl;
        std::cin >> T;
        std::cout << "Value for alpha?" << std::endl;
        std::cin >> alpha;
        std::cout << "Value for alpha gradient bandit?" << std::endl;
        std::cin >> alpha_gradient_bandit;
        std::cout << "Value for c?" << std::endl;
        std::cin >> c;
        std::cout << "Value for epsilon?" << std::endl;
        std::cin >> epsilon;
        std::cout << "Value for initial value?" << std::endl;
        std::cin >> initial_value;
        std::cout << "Value for normal mean?" << std::endl;
        std::cin >> normal_mean;
        std::vector<double> means;
        if (random_walk)
        {
            for (int i = 0; i < k; i++)
            {
                means.push_back(normal_mean);
            }
        }
        else
        {
            for (int i = 0; i < k; i++)
            {
                means.push_back(normal_mean + (*normal_distribution)(*generator));
            }
        }
        multi_armed_bandits mab(means);
        run r(baseline,
              random_walk,
              n, 
              T, 
              alpha,
              alpha_gradient_bandit,
              c,
              epsilon, 
              initial_value,
              mab, 
              normal_distribution, 
              generator);
        if (random_walk)
        {
            for (int num = 0; num < n; num++)
            {
                r.episode();
                for (int i = 0; i < k; i++)
                {
                    means.at(i) = normal_mean;
                }
                mab.new_means(means);
                r.reset(mab);
            }
        }
        else
        {
            for (int num = 0; num < n; num++)
            {
                r.episode();
                for (int i = 0; i < k; i++)
                {
                    means.at(i) = normal_mean + (*normal_distribution)(*generator);
                }
                mab.new_means(means);
                r.reset(mab);
            }            
        }
        r.write();
    }
}