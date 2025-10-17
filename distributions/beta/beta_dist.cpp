//
// Created by ChesterZed on 17.09.2025.
//

#include "beta_dist.h"


BetaDistribution::BetaDistribution(const double alpha, const double beta) : alpha_(alpha), beta_(beta) {
    if (alpha <= 0.0 || beta <= 0.0) {
        throw std::invalid_argument("Введите оба значения больше 0");
    }
}

double BetaDistribution::betaFunction(double a, double b) const {
    return tgamma(a) * tgamma(b) / tgamma(a + b);
}

double BetaDistribution::incompleteBeta(double x, double a, double b) const {
    if (x <= 0.0)
        return 0.0;
    if (x >= 1.0)
        return 1.0;

    auto integralF = [a, b](double tetha) {
        return std::pow(tetha, a - 1) * std::pow(1 - tetha, b - 1);
    };

    double integral = gauss_kronrod<double, 61>::integrate(integralF, 0.0, x);

    return integral;
}

double BetaDistribution::pdf(double t) const {
    return 1./betaFunction(alpha_, beta_) * pow(t, alpha_ - 1) * pow(1 - t, beta_ - 1);
}


double BetaDistribution::cdf(double x) const {
    return incompleteBeta(x, alpha_, beta_)/betaFunction(alpha_, beta_);
}

double BetaDistribution::sf(double x) const {
    return 1.0 - cdf(x);
}

double BetaDistribution::hf(double x) const {
    double surviavalF = sf(x);
    if (surviavalF <= 0)
        return std::numeric_limits<double>::infinity();
    return pdf(x) / surviavalF;
}

double BetaDistribution::chf(double x) const {
    double surviavalF = sf(x);
    if (surviavalF <= 0)
        return std::numeric_limits<double>::infinity();
    return -std::log(surviavalF);
}

double BetaDistribution::random_sample() const {
    return 0.0;
}

std::vector<double> BetaDistribution::random_sample(size_t n) const {
    return std::vector<double>(5);
}

double BetaDistribution::mean() const {
    return alpha_ / (alpha_ + beta_);
}

double BetaDistribution::variance() const {
    double denominator = (alpha_ + beta_) * (alpha_ + beta_) * (alpha_ + beta_ + 1);
    return (alpha_ * beta_) / denominator;
}

double BetaDistribution::mode() const {
    if (alpha_ > 1.0 && beta_ > 1.0) {
        return (alpha_ - 1.0) / (alpha_ + beta_ - 2.0);
    } else {
        return std::numeric_limits<double>::quiet_NaN();
    }
}

void BetaDistribution::plot(int points = 200, bool show_cdf = true) const {
    std::vector<double> x(points), y_pdf(points), y_cdf(points);
    for (int i = 0; i < points; ++i) {
        x[i] = static_cast<double>(i) / (points - 1);
        y_pdf[i] = pdf(x[i]);
        y_cdf[i] = cdf(x[i]);
    }

    plt::figure();
    plt::subplot(2, 1, 1);
    plt::plot(x, y_pdf, "b-");
    plt::title("Beta Distribution PDF");
    plt::xlabel("x");
    plt::ylabel("Density");

    if (show_cdf) {
        plt::subplot(2, 1, 2);
        plt::plot(x, y_cdf, "r-");
        plt::title("Beta Distribution CDF");
        plt::xlabel("x");
        plt::ylabel("Probability");
    }

    plt::tight_layout();
    plt::show();
}
