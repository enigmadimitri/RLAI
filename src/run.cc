#include "run.h"
#include "multi_armed_bandits.h"

#include <fstream>
#include <iostream>
#include <string>

// Constructor

run::run(bool vbaseline,
         bool vrandom_walk,
         int vn, 
         int vT, 
         double valpha,
         double valpha_gradient_bandit,
         double vc,
         double vepsilon, 
         double vinitial_value,
         const multi_armed_bandits& vmab, 
         std::normal_distribution<double>* vnormal_distribution, 
         std::mt19937* vgenerator) 
         : 
         baseline(vbaseline),
         random_walk(vrandom_walk),
         current(0),
         n(vn), 
         t(0), 
         T(vT), 
         alpha(valpha),
         alpha_gradient_bandit(valpha_gradient_bandit),
         c(vc),
         epsilon(vepsilon), 
         initial_value(vinitial_value),
         reward_mean(0),
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
        values_ucb.push_back(0);
        preferences.push_back(0);
        probabilities.push_back(0);
    }
    for (int i = 0; i < T; i++)
    {
        average_reward.push_back(0);
        percentage_optimal_action.push_back(0);
    }

}

// Returning super reward average

double run::super_reward_average(int t_initial)
{
    double result = 0;
    for (int i = t_initial; i < T; i++)
    {
        result += average_reward.at(i);
    }
    return result / (T - t_initial);
}

// Altering means

void run::alterate()
{
    std::vector<double> add_means;
    for (int i = 0; i < k; i++)
    {
        add_means.push_back(0.01 * (*normal_distribution)(*generator));
    }
    mab.add_means(add_means);
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
        std::uniform_int_distribution<int> dd (0, choices.size() - 1);
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
        std::uniform_int_distribution<int> dd (0, choices.size() - 1);
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

// Playing one gradient bandit step

void run::step_gradient_bandit()
{
    double probability_norm = 0;
    for (int i = 0; i < k; i++)
    {
        probability_norm += exp(preferences.at(i));
    }
    for (int i = 0; i < k; i++)
    {
        probabilities.at(i) = exp(preferences.at(i)) / probability_norm;
    }
    std::discrete_distribution<int> dd (probabilities.begin(), probabilities.end());
    choice = dd(*generator);
    reward = mab.get_mean(choice) + mab.get_standard_deviation(choice) * (*normal_distribution)(*generator);
    reward_mean = (reward_mean * t + reward) / (t + 1);
    for (int i = 0; i < k; i++)
    {
        if (baseline)
        {
            if (i == choice)
            {
                preferences.at(i) += alpha_gradient_bandit * (reward - reward_mean) * (1 - probabilities.at(i));
            }
            else
            {
                preferences.at(i) -= alpha_gradient_bandit * (reward - reward_mean) * probabilities.at(i);
            }
        }
        else
        {
            if (i == choice)
            {
                preferences.at(i) += alpha_gradient_bandit * reward * (1 - probabilities.at(i));
            }
            else
            {
                preferences.at(i) -= alpha_gradient_bandit * reward * probabilities.at(i);
            }            
        }
    }
    percentage_correct = 100 * (mab.get_mean(choice) == max(mab.get_means()));
    plays.at(choice) += 1;
    average_reward.at(t) = (average_reward.at(t) * current + reward) / (current + 1);
    percentage_optimal_action.at(t) = (percentage_optimal_action.at(t) * current + percentage_correct) / (current + 1);
    t++;
}

// Playing one ucb step

void run::step_ucb()
{
    if (t < k)
    {   
        std::vector<int> unplayed;
        for (int i = 0; i < k; i++)
        {
            if (plays.at(i) == 0)
            {
                unplayed.push_back(i);
            }
        }
        std::uniform_int_distribution<int> dd (0, unplayed.size() - 1);
        choice = unplayed.at(dd(*generator));
    }
    else
    {
        std::vector<int> choices;
        for (int i = 0; i < k; i++)
        {
            values_ucb.at(i) = values.at(i) + c * sqrt(log(t) / plays.at(i));
        }
        choices = argmax(values_ucb);
        std::uniform_int_distribution<int> dd (0, choices.size() - 1);
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
    if (c > 0)
    {
        if (random_walk)
        {
            for (int i = 0; i < T; i++)
            {
                step_ucb();
                alterate();
            }
        }
        else
        {
            for (int i = 0; i < T; i++)
            {
                step_ucb();
            }
        }
    }
    else if (alpha > 0)
    {
        if (random_walk)
        {
            for (int i = 0; i < T; i++)
            {
                step_alpha();
                alterate();
            }
        }
        else
        {
            for (int i = 0; i < T; i++)
            {
                step_alpha();
            }
        }    
    }
    else if (alpha_gradient_bandit > 0)
    {
        if (random_walk)
        {
            for (int i = 0; i < T; i++)
            {
                step_gradient_bandit();
                alterate();
            }
        }
        else
        {
            for (int i = 0; i < T; i++)
            {
                step_gradient_bandit();
            }
        }      
    }    
    else
    {
        if (random_walk)
        {
            for (int i = 0; i < T; i++)
            {
                step_classic();
                alterate();
            }
        }
        else
        {
            for (int i = 0; i < T; i++)
            {
                step_classic();
            }
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
        reward_mean = 0;
        plays.at(i) = 0;
        values.at(i) = initial_value;
        preferences.at(i) = 0;
        probabilities.at(i) = 0;
    }
    mab = vmab;
}

// Writting results

void run::write()
{
    std::string avg_filename = "data/avg_reward_" + std::to_string(alpha) + "_"
                                                  + std::to_string(alpha_gradient_bandit) + "_"
                                                  + std::to_string(baseline) + "_"
                                                  + std::to_string(c) + "_"
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
                                                          + std::to_string(alpha_gradient_bandit) + "_"
                                                          + std::to_string(baseline) + "_"
                                                          + std::to_string(c) + "_"
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