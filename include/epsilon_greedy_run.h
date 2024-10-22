#ifndef EPSILON_GREEDY_RUN_H
#define EPSILON_GREEDY_RUN_H

#include "multi_armed_bandits.h"
#include "utilities.h"

#include <random>

class epsilon_greedy_run
{
  private:
    int choice;
    int current;
    int k;
    int n;
    int percentage_correct;
    int t;
    int T;
    double epsilon;
    double reward;
    std::bernoulli_distribution epsilon_distribution;
    std::vector<int> plays;
    std::vector<double> values;
    std::vector<double> average_reward;
    std::vector<double> percentage_optimal_action;
    multi_armed_bandits mab;
    std::normal_distribution<double>* normal_distribution;
    std::mt19937* generator;

  public:
    // Constructor
    epsilon_greedy_run(int, int, double, const multi_armed_bandits&, std::normal_distribution<double>*, std::mt19937*);
    // Playing one step
    void step();
    // Playing one episode
    void episode();
    // Resetting current episode
    void reset(const multi_armed_bandits&);
    // Writting results
    void write();
};

#endif