from .bayesian_layers import BayesianLinear
from .bayesian_classifier import BayesianClassifier
import ipdb

def make_bayesian_mlp_classifier(
        n_input,
        n_output,
        hidden_layer_sizes,
        prior_stddev,
        optimize_prior_mean,
        optimize_prior_rho,
        optimize_posterior_mean,
        optimize_posterior_rho,
        probability_threshold,
        normalize_surrogate_by_log_classes
):
    layers = []
    in_features = n_input

    for hidden_layer_size in hidden_layer_sizes:
        layers.append(
            BayesianLinear(
                in_features=in_features,
                out_features=hidden_layer_size,
                activation='relu',
                prior_stddev=prior_stddev,
                optimize_prior_mean=optimize_prior_mean,
                optimize_prior_rho=optimize_prior_rho,
                optimize_posterior_mean=optimize_posterior_mean,
                optimize_posterior_rho=optimize_posterior_rho,
                w_prior_mean_init=None,
                b_prior_mean_init=None,
                w_posterior_mean_init=None,
                b_posterior_mean_init=None,
            )
        )
        in_features = hidden_layer_size
    layers.append(
        BayesianLinear(
            in_features=in_features,
            out_features=n_output,
            activation='softmax',
            prior_stddev=prior_stddev,
            optimize_prior_mean=optimize_prior_mean,
            optimize_prior_rho=optimize_prior_rho,
            optimize_posterior_mean=optimize_posterior_mean,
            optimize_posterior_rho=optimize_posterior_rho,
            w_prior_mean_init=None,
            b_prior_mean_init=None,
            w_posterior_mean_init=None,
            b_posterior_mean_init=None,
        )
    )
    return BayesianClassifier(
        probability_threshold,
        normalize_surrogate_by_log_classes,
        *layers
    )


def make_bayesian_classifier_from_mlps(
        mlp_posterior_mean_init,
        mlp_prior_mean,
        prior_stddev,
        optimize_prior_mean,
        optimize_prior_rho,
        optimize_posterior_mean,
        optimize_posterior_rho,
        probability_threshold,
        normalize_surrogate_by_log_classes
):
    posterior_mean_init_parameters = list(mlp_posterior_mean_init.parameters())
    prior_mean_parameters = list(mlp_prior_mean.parameters())
    n_layers = len(prior_mean_parameters) // 2

    layers = []
    for i_layer in range(n_layers):
        if i_layer == n_layers - 1:
            activation = 'softmax'
        else:
            activation = 'relu'
        layers.append(
            BayesianLinear(
                in_features=prior_mean_parameters[i_layer * 2].shape[1],
                out_features=prior_mean_parameters[i_layer * 2].shape[0],
                activation=activation,
                prior_stddev=prior_stddev,
                optimize_prior_mean=optimize_prior_mean,
                optimize_prior_rho=optimize_prior_rho,
                optimize_posterior_mean=optimize_posterior_mean,
                optimize_posterior_rho=optimize_posterior_rho,
                w_prior_mean_init=prior_mean_parameters[i_layer * 2],
                b_prior_mean_init=prior_mean_parameters[i_layer * 2 + 1],
                w_posterior_mean_init=posterior_mean_init_parameters[i_layer * 2],
                b_posterior_mean_init=posterior_mean_init_parameters[i_layer * 2 + 1],
            )
        )
    return BayesianClassifier(
        probability_threshold,
        normalize_surrogate_by_log_classes,
        *layers
    )
