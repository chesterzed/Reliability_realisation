//
// Created by ChesterZed on 17.09.2025.
//

#include "beta_dist.h"


BetaDistribution::BetaDistribution(const double alpha, const double beta) : alpha_(alpha), beta_(beta) {
    if (alpha <= 0.0 || beta <= 0.0) {
        throw std::invalid_argument("Введите оба значения больше 0");
    }
}

double BetaDistribution::beta_function(double a, double b) const {
    return tgamma(a) * tgamma(b) / tgamma(a + b);
}

double BetaDistribution::pdf(double t) const {
    return 1./beta_function(alpha_, beta_) * pow(t, alpha_ - 1) * pow(1 - t, beta_ - 1);
}


double BetaDistribution::cdf(double x) const {
    return 0.0;
}

double BetaDistribution::sf(double x) const {
    return 0.0;
}

double BetaDistribution::hf(double x) const {
    return 0.0;
}

double BetaDistribution::chf(double x) const {
    return 0.0;
}

double BetaDistribution::random_sample() const {
    return 0.0;
}

std::vector<double> BetaDistribution::random_sample(size_t n) const {
    return std::vector<double>(5);
}

double BetaDistribution::mean() const {
    return 0.0;
}

double BetaDistribution::variance() const {
    return 0.0;
}

double BetaDistribution::mode() const {
    return 0.0;
}
