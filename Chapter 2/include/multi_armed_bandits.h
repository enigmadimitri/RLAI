#ifndef MULTI_ARMED_BANDITS_H
#define MULTI_ARMED_BANDITS_H

#include <vector>

class multi_armed_bandits
{
  private:
    int k;
    std::vector<double> means;
    std::vector<double> standard_deviations;

  public:
    // Constructors
    multi_armed_bandits(int);
    multi_armed_bandits(int, double);
    multi_armed_bandits(int, double, double);
    multi_armed_bandits(const std::vector<double>&);
    multi_armed_bandits(const std::vector<double>&, double);
    multi_armed_bandits(double, const std::vector<double>&);
    multi_armed_bandits(const std::vector<double>&, const std::vector<double>&);
    // Replacing old means with new means
    void new_means(double);
    void new_means(const std::vector<double>&);
    // Replacing old stds with new stds
    void new_standard_deviations(double);
    void new_standard_deviations(const std::vector<double>&);
    // Adding old means with new means
    void add_means(double);
    void add_means(const std::vector<double>&);
    // Adding old stds with new stds
    void add_standard_deviations(double);
    void add_standard_deviations(const std::vector<double>&);
    // Getting k
    int get_k();
    // Getting means
    double get_mean(int);
    std::vector<double>& get_means();
    // Getting stds
    double get_standard_deviation(int);
    std::vector<double>& get_standard_deviations();
    // Getting arm
    std::vector<double> get_arm(int);
};

#endif