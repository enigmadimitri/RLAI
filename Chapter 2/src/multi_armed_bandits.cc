#include "multi_armed_bandits.h"

// Constructors

multi_armed_bandits::multi_armed_bandits(int vk) : k(vk)
{
    for (int i = 0; i < k; i++)
    {
        means.push_back(0);
        standard_deviations.push_back(1);
    }
}

multi_armed_bandits::multi_armed_bandits(int vk, double vmean) : k(vk)
{
    for (int i = 0; i < k; i++)
    {
        means.push_back(vmean);
        standard_deviations.push_back(1);
    }
}

multi_armed_bandits::multi_armed_bandits(int vk, double vmean, double vstandard_deviation) : k(vk)
{
    for (int i = 0; i < k; i++)
    {
        means.push_back(vmean);
        standard_deviations.push_back(vstandard_deviation);
    }
}

multi_armed_bandits::multi_armed_bandits(const std::vector<double>& vmeans)
{
    k = vmeans.size();
    for (int i = 0; i < k; i++)
    {
        means.push_back(vmeans.at(i));
        standard_deviations.push_back(1);
    }    
}

multi_armed_bandits::multi_armed_bandits(const std::vector<double>& vmeans, double vstandard_deviation)
{
    k = vmeans.size();
    for (int i = 0; i < k; i++)
    {
        means.push_back(vmeans.at(i));
        standard_deviations.push_back(vstandard_deviation);
    }    
}

multi_armed_bandits::multi_armed_bandits(double vmean, const std::vector<double>& vstandard_deviations)
{
    k = vstandard_deviations.size();
    for (int i = 0; i < k; i++)
    {
        means.push_back(vmean);
        standard_deviations.push_back(vstandard_deviations.at(i));
    }
}

multi_armed_bandits::multi_armed_bandits(const std::vector<double>& vmeans, const std::vector<double>& vstandard_deviations)
{
    k = vmeans.size();
    for (int i = 0; i < k; i++)
    {
        means.push_back(vmeans.at(i));
        standard_deviations.push_back(vstandard_deviations.at(i));
    }  
}

// Replacing old means with new means

void multi_armed_bandits::new_means(double vmean)
{
    for (int i = 0; i < k; i++)
    {
        means.at(i) = vmean;
    }      
}

void multi_armed_bandits::new_means(const std::vector<double>& vmeans)
{
    for (int i = 0; i < k; i++)
    {
        means.at(i) = vmeans.at(i);
    }  
}

// Replacing old stds with new stds

void multi_armed_bandits::new_standard_deviations(double vstandard_deviation)
{
    for (int i = 0; i < k; i++)
    {
        standard_deviations.at(i) = vstandard_deviation;
    }   
}

void multi_armed_bandits::new_standard_deviations(const std::vector<double>& vstandard_deviations)
{
    for (int i = 0; i < k; i++)
    {
        standard_deviations.at(i) = vstandard_deviations.at(i);
    }   
}

// Adding old means with new means

void multi_armed_bandits::add_means(double vmean)
{
    for (int i = 0; i < k; i++)
    {
        means.at(i) += vmean;
    }      
}

void multi_armed_bandits::add_means(const std::vector<double>& vmeans)
{
    for (int i = 0; i < k; i++)
    {
        means.at(i) += vmeans.at(i);
    }  
}

// Adding old stds with new stds

void multi_armed_bandits::add_standard_deviations(double vstandard_deviation)
{
    for (int i = 0; i < k; i++)
    {
        standard_deviations.at(i) += vstandard_deviation;
    }   
}

void multi_armed_bandits::add_standard_deviations(const std::vector<double>& vstandard_deviations)
{
    for (int i = 0; i < k; i++)
    {
        standard_deviations.at(i) += vstandard_deviations.at(i);
    }   
}

// Getting k

int multi_armed_bandits::get_k()
{
    return k;
}

// Getting means

double multi_armed_bandits::get_mean(int vi)
{
    return means.at(vi);
}

std::vector<double>& multi_armed_bandits::get_means()
{
    return means;
}

// Getting stds

double multi_armed_bandits::get_standard_deviation(int vi)
{
    return standard_deviations.at(vi);
}

std::vector<double>& multi_armed_bandits::get_standard_deviations()
{
    return standard_deviations;
}

// Getting arm

std::vector<double> multi_armed_bandits::get_arm(int vi)
{
    std::vector<double> result = {means.at(vi), standard_deviations.at(vi)};
    return result;
}