//
// Created by ChesterZed on 17.09.2025.
//

#ifndef RELIABILITY_REALISATION_DISTRIBUTIONS_BETA_BETA_DIST_H_
#define RELIABILITY_REALISATION_DISTRIBUTIONS_BETA_BETA_DIST_H_

#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <iostream>
#include <boost/math/quadrature/naive_monte_carlo.hpp>

class BetaDistribution {
public:

    BetaDistribution(double alpha, double beta);

    double pdf(double t) const;
    double cdf(double x) const;
    double sf(double x) const;
    double hf(double x) const;
    double chf(double x) const;
    double random_sample() const;
    std::vector<double> random_sample(size_t n) const;
    double mean() const;
    double variance() const;
    double mode() const;
    static BetaDistribution fit(const std::vector<double>& data);
    double get_alpha() const { return alpha_; }
    double get_beta() const { return beta_; }

    std::string summary() const;

private:
    double alpha_;
    double beta_;

    double beta_function(double a, const double b) const;

    double incomplete_beta(double x, double a, double b) const;
};


#endif //RELIABILITY_REALISATION_DISTRIBUTIONS_BETA_BETA_DIST_H_
