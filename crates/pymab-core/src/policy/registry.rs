//! Stable registry of built-in policy kinds.

/// Every concrete policy exported by PyMAB.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum PolicyKind {
    BernoulliBayesianUcb,
    BernoulliThompsonSampling,
    ChangePointUcb,
    CusumUcb,
    DecayingEpsilonGreedy,
    DiscountedBernoulliThompsonSampling,
    DiscountedUcb,
    EpsilonGreedy,
    Exp3,
    GaussianBayesianUcb,
    GaussianThompsonSampling,
    GradientBandit,
    Greedy,
    KlUcb,
    LinUcb,
    LinearEpsilonGreedy,
    LinearThompsonSampling,
    LogisticContextualBandit,
    MedianElimination,
    Moss,
    PageHinkleyUcb,
    Random,
    SlidingWindowBernoulliThompsonSampling,
    SlidingWindowUcb,
    Softmax,
    SuccessiveElimination,
    Ucb,
}

impl PolicyKind {
    /// Complete policy registry in stable order.
    pub const ALL: [Self; 27] = [
        Self::BernoulliBayesianUcb,
        Self::BernoulliThompsonSampling,
        Self::ChangePointUcb,
        Self::CusumUcb,
        Self::DecayingEpsilonGreedy,
        Self::DiscountedBernoulliThompsonSampling,
        Self::DiscountedUcb,
        Self::EpsilonGreedy,
        Self::Exp3,
        Self::GaussianBayesianUcb,
        Self::GaussianThompsonSampling,
        Self::GradientBandit,
        Self::Greedy,
        Self::KlUcb,
        Self::LinUcb,
        Self::LinearEpsilonGreedy,
        Self::LinearThompsonSampling,
        Self::LogisticContextualBandit,
        Self::MedianElimination,
        Self::Moss,
        Self::PageHinkleyUcb,
        Self::Random,
        Self::SlidingWindowBernoulliThompsonSampling,
        Self::SlidingWindowUcb,
        Self::Softmax,
        Self::SuccessiveElimination,
        Self::Ucb,
    ];

    /// Return the stable snake-case fixture and serialization name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BernoulliBayesianUcb => "bernoulli_bayesian_ucb",
            Self::BernoulliThompsonSampling => "bernoulli_thompson_sampling",
            Self::ChangePointUcb => "change_point_ucb",
            Self::CusumUcb => "cusum_ucb",
            Self::DecayingEpsilonGreedy => "decaying_epsilon_greedy",
            Self::DiscountedBernoulliThompsonSampling => "discounted_bernoulli_thompson_sampling",
            Self::DiscountedUcb => "discounted_ucb",
            Self::EpsilonGreedy => "epsilon_greedy",
            Self::Exp3 => "exp3",
            Self::GaussianBayesianUcb => "gaussian_bayesian_ucb",
            Self::GaussianThompsonSampling => "gaussian_thompson_sampling",
            Self::GradientBandit => "gradient_bandit",
            Self::Greedy => "greedy",
            Self::KlUcb => "kl_ucb",
            Self::LinUcb => "lin_ucb",
            Self::LinearEpsilonGreedy => "linear_epsilon_greedy",
            Self::LinearThompsonSampling => "linear_thompson_sampling",
            Self::LogisticContextualBandit => "logistic_contextual_bandit",
            Self::MedianElimination => "median_elimination",
            Self::Moss => "moss",
            Self::PageHinkleyUcb => "page_hinkley_ucb",
            Self::Random => "random",
            Self::SlidingWindowBernoulliThompsonSampling => {
                "sliding_window_bernoulli_thompson_sampling"
            }
            Self::SlidingWindowUcb => "sliding_window_ucb",
            Self::Softmax => "softmax",
            Self::SuccessiveElimination => "successive_elimination",
            Self::Ucb => "ucb",
        }
    }

    /// Return the matching public Python class name.
    #[must_use]
    pub const fn python_name(self) -> &'static str {
        match self {
            Self::BernoulliBayesianUcb => "BernoulliBayesianUCBPolicy",
            Self::BernoulliThompsonSampling => "BernoulliThompsonSamplingPolicy",
            Self::ChangePointUcb => "ChangePointUCBPolicy",
            Self::CusumUcb => "CUSUMUCBPolicy",
            Self::DecayingEpsilonGreedy => "DecayingEpsilonGreedyPolicy",
            Self::DiscountedBernoulliThompsonSampling => {
                "DiscountedBernoulliThompsonSamplingPolicy"
            }
            Self::DiscountedUcb => "DiscountedUCBPolicy",
            Self::EpsilonGreedy => "EpsilonGreedyPolicy",
            Self::Exp3 => "EXP3Policy",
            Self::GaussianBayesianUcb => "GaussianBayesianUCBPolicy",
            Self::GaussianThompsonSampling => "GaussianThompsonSamplingPolicy",
            Self::GradientBandit => "GradientBanditPolicy",
            Self::Greedy => "GreedyPolicy",
            Self::KlUcb => "KLUCBPolicy",
            Self::LinUcb => "LinUCBPolicy",
            Self::LinearEpsilonGreedy => "LinearEpsilonGreedyPolicy",
            Self::LinearThompsonSampling => "LinearThompsonSamplingPolicy",
            Self::LogisticContextualBandit => "LogisticContextualBanditPolicy",
            Self::MedianElimination => "MedianEliminationPolicy",
            Self::Moss => "MOSSPolicy",
            Self::PageHinkleyUcb => "PageHinkleyUCBPolicy",
            Self::Random => "RandomPolicy",
            Self::SlidingWindowBernoulliThompsonSampling => {
                "SlidingWindowBernoulliThompsonSamplingPolicy"
            }
            Self::SlidingWindowUcb => "SlidingWindowUCBPolicy",
            Self::Softmax => "SoftmaxPolicy",
            Self::SuccessiveElimination => "SuccessiveEliminationPolicy",
            Self::Ucb => "UCBPolicy",
        }
    }

    /// Return whether the policy consumes contextual observations.
    #[must_use]
    pub const fn is_contextual(self) -> bool {
        matches!(
            self,
            Self::LinUcb
                | Self::LinearEpsilonGreedy
                | Self::LinearThompsonSampling
                | Self::LogisticContextualBandit
        )
    }
}
