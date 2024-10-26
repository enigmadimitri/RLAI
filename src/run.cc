#include "run.h"
#include "multi_armed_bandits.h"

#include <fstream>
#include <iostream>
#include <string>

// Constructor

run::run(int vn, 
                                       int vT, 
                                       double valpha,
                                       double vepsilon, 
                                       double vinitial_value,
                                       const multi_armed_bandits& vmab, 
                                       std::normal_distribution<double>* vnormal_distribution, 
                                       std::mt19937* vgenerator) 
                                       : 
                                       current(0),
                                       n(vn), 
                                       t(0), 
                                       T(vT), 
                                       alpha(valpha),
                                       epsilon(vepsilon), 
                                       initial_value(vinitial_value),
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
        values.push_back(initial_value);
    }
    for (int i = 0; i < T; i++)
    {
        average_reward.push_back(0);
        percentage_optimal_action.push_back(0);
    }

}

// Playing one alpha step

void run::step_alpha()
{
    if (epsilon_distribution(*generator))
    {
        std::uniform_int_distribution<> dd(0, k - 1);
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
    values.at(choice) = values.at(choice) * (1 - alpha) + alpha * reward;
    average_reward.at(t) = (average_reward.at(t) * current + reward) / (current + 1);
    percentage_optimal_action.at(t) = (percentage_optimal_action.at(t) * current + percentage_correct) / (current + 1);
    t++;
}

// Playing one classic step

void run::step_classic()
{
    if (epsilon_distribution(*generator))
    {
        std::uniform_int_distribution<> dd(0, k - 1);
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

void run::episode()
{
    for (int i = 0; i < T; i++)
    {
        if (alpha == 0)
        {
            step_classic();
        }
        else
        {
            step_alpha();
        }
    }
}

// Resetting current episode

void run::reset(const multi_armed_bandits& vmab)
{
    current += 1;
    t = 0;
    for (int i = 0; i < k; i++)
    {
        plays.at(i) = 0;
        values.at(i) = initial_value;
    }
    mab = vmab;
}

// Writting results

void run::write()
{
    std::string avg_filename = "data/avg_reward_" + std::to_string(alpha) + "_"
                                                  + std::to_string(epsilon) + "_"
                                                  + std::to_string(initial_value) + "_run.data";
    std::ofstream avg_file(avg_filename);
    avg_file << "step avg_reward" << std::endl;
    for (int i = 0; i < T - 1; i++)
    { 
        avg_file << i + 1 << " " << average_reward.at(i) << std::endl;
    }
    avg_file << T << " " << average_reward.at(T - 1);
    avg_file.close();

    std::string pct_filename = "data/pct_optimal_action_" + std::to_string(alpha) + "_"
                                                          + std::to_string(epsilon) + "_"
                                                          + std::to_string(initial_value) + "_run.data";
    std::ofstream pct_file(pct_filename);
    pct_file << "step pct_optimal_action" << std::endl;
    for (int i = 0; i < T - 1; i++)
    { 
        pct_file << i + 1 << " " << percentage_optimal_action.at(i) << std::endl;
    }
    pct_file << T << " " << percentage_optimal_action.at(T - 1);
    pct_file.close();
}