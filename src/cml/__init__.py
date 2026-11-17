"""cml -- causal ML meta-learners and treatment-effect estimators from scratch."""

from .data import CausalData, SyntheticConfig, load_ihdp, load_lalonde, make_synthetic
from .estimators import (Estimate, aipw, bootstrap, difference_in_means, dml_plr, ipw,
                         propensity_scores, psm, regression_adjustment)
from .metalearners import (DRLearner, MetaLearner, RLearner, SLearner, TLearner, XLearner,
                           make_learner)
from .metrics import ate_error, sqrt_pehe

__all__ = [
    "CausalData", "DRLearner", "Estimate", "MetaLearner", "RLearner", "SLearner",
    "SyntheticConfig", "TLearner", "XLearner", "aipw", "ate_error", "bootstrap",
    "difference_in_means", "dml_plr", "ipw", "load_ihdp", "load_lalonde", "make_learner",
    "make_synthetic", "propensity_scores", "psm", "regression_adjustment", "sqrt_pehe",
]
__version__ = "0.1.0"
