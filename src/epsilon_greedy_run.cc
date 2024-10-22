#include "epsilon_greedy_run.h"
#include "multi_armed_bandits.h"

#include <fstream>
#include <string>

// Constructor

epsilon_greedy_run::epsilon_greedy_run(int vn, 
                                       int vT, 
                                       double vepsilon, 
                                       const multi_armed_bandits& vmab, 
                                       std::normal_distribution<double>* vnormal_distribution, 
                                       std::mt19937* vgenerator) 
                                       : 
                                       current(0),
                                       n(vn), 
                                       t(0), 
                                       T(vT), 
                                       epsilon(vepsilon), 
                                       mab(vmab), 
                                       normal_distribution(vnormal_distribution), 
                                       generator(vgenerator)
{   
    k = mab.get_k();
    std::bernoulli_distribution bd(epsilon);
    epsilon_distribution = bd;
    for (int i = 0; i < k; i++)
    {
        plays.push_back(0);
        values.push_back(0);
    }
    for (int i = 0; i < T; i++)
    {
        average_reward.push_back(0);
        percentage_optimal_action.push_back(0);
    }

}

// Playing one step

void epsilon_greedy_run::step()
{
    if (epsilon_distribution(*generator))
    {
        std::discrete_distribution<int> dd (values.begin(), values.end());
        choice = dd(*generator);
    }
    else
    {
        std::vector<int> choices;
        choices = argmax(values);
        std::discrete_distribution<int> dd (choices.begin(), choices.end());
        choice = choices.at(dd(*generator));
    }
    percentage_correct = 100 * (mab.get_mean(choice) == max(mab.get_means()));
    plays.at(choice) += 1;
    reward = mab.get_mean(choice) + mab.get_standard_deviation(choice) * (*normal_distribution)(*generator);
    values.at(choice) = (values.at(choice) * (plays.at(choice) - 1) + reward) / plays.at(choice);
    average_reward.at(t) = (average_reward.at(t) * current + reward) / (current + 1);
    percentage_optimal_action.at(t) = (percentage_optimal_action.at(t) * current + percentage_correct) / (current + 1);
    t++;
}

// Playing one episode

void epsilon_greedy_run::episode()
{
    for (int i = 0; i < T; i++)
    {
        step();
    }
}

// Resetting current episode

void epsilon_greedy_run::reset(const multi_armed_bandits& vmab)
{
    current += 1;
    t = 0;
    for (int i = 0; i < k; i++)
    {
        plays.at(i) = 0;
        values.at(i) = 0;
    }
    mab = vmab;
}

// Writting results

void epsilon_greedy_run::write()
{
    std::string avg_filename = "data/avg_reward_" + std::to_string(epsilon) + "_greedy_run.data";
    std::ofstream avg_file(avg_filename);
    avg_file << "step avg_reward" << std::endl;
    for (int i = 0; i < T - 1; i++)
    { 
        avg_file << i + 1 << " " << average_reward.at(i) << std::endl;
    }
    avg_file << T << " " << average_reward.at(T - 1);
    avg_file.close();

    std::string pct_filename = "data/pct_optimal_action_" + std::to_string(epsilon) + "_greedy_run.data";
    std::ofstream pct_file(pct_filename);
    pct_file << "step pct_optimal_action" << std::endl;
    for (int i = 0; i < T - 1; i++)
    { 
        pct_file << i + 1 << " " << percentage_optimal_action.at(i) << std::endl;
    }
    pct_file << T << " " << percentage_optimal_action.at(T - 1);
    pct_file.close();
}