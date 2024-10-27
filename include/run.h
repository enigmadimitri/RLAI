#ifndef RUN_H
#define RUN_H

#include "multi_armed_bandits.h"
#include "utilities.h"

#include <random>

class run
{
  private:
    bool baseline;
    int choice;
    int current;
    int k;
    int n;
    int percentage_correct;
    int t;
    int T;
    double alpha;
    double alpha_gradient_bandit;
    double c;
    double epsilon;
    double initial_value;
    double reward;
    double reward_mean;
    std::bernoulli_distribution epsilon_distribution;
    std::vector<int> plays;
    std::vector<double> values;
    std::vector<double> values_ucb;
    std::vector<double> average_reward;
    std::vector<double> percentage_optimal_action;
    std::vector<double> preferences;
    std::vector<double> probabilities;
    multi_armed_bandits mab;
    std::normal_distribution<double>* normal_distribution;
    std::mt19937* generator;

  public:
    // Constructor
    run(bool, int, int, double, double, double, double, double, const multi_armed_bandits&, std::normal_distribution<double>*, std::mt19937*);
    // Returning super reward average
    double super_reward_average();
    // Playing one alpha step
    void step_alpha();
    // Playing one classic step
    void step_classic();
    // Playing one gradient bandit step
    void step_gradient_bandit();
    // Playing one ucb step
    void step_ucb();
    // Playing one episode
    void episode();
    // Resetting current episode
    void reset(const multi_armed_bandits&);
    // Writting results
    void write();
};

#endif