from .bayesian_layers import BayesianLinear, BayesianConv2d
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
        prob_threshold,
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
        prob_threshold,
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
        prob_threshold,
        normalize_surrogate_by_log_classes,
        oracle_prior_variance
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
        bayesian_layers=layers,
        prob_threshold=prob_threshold,
        normalize_surrogate_by_log_classes=normalize_surrogate_by_log_classes,
        oracle_prior_variance=oracle_prior_variance
    )


def make_bayesian_classifier_from_lenets(
        net_posterior_mean_init,
        net_prior_mean,
        prior_stddev,
        optimize_prior_mean,
        optimize_prior_rho,
        optimize_posterior_mean,
        optimize_posterior_rho,
        prob_threshold,
        normalize_surrogate_by_log_classes,
        oracle_prior_variance
):
    n_conv_layers = len(net_posterior_mean_init.conv_layers)
    n_linear_layers = len(net_posterior_mean_init.linear_layers)
    bayesian_layers = []
    for i_conv_layer in range(n_conv_layers):
        bayesian_layers.append(
            BayesianConv2d(
                in_channels=net_prior_mean.conv_layers[i_conv_layer].in_channels,
                out_channels=net_prior_mean.conv_layers[i_conv_layer].out_channels,
                kernel_size=net_prior_mean.conv_layers[i_conv_layer].kernel_size,
                activation='maxpool',
                prior_stddev=prior_stddev,
                optimize_prior_mean=optimize_prior_mean,
                optimize_prior_rho=optimize_prior_rho,
                optimize_posterior_mean=optimize_posterior_mean,
                optimize_posterior_rho=optimize_posterior_rho,
                w_prior_mean_init=net_prior_mean.conv_layers[i_conv_layer].weight,
                b_prior_mean_init=net_prior_mean.conv_layers[i_conv_layer].bias,
                w_posterior_mean_init=net_posterior_mean_init.conv_layers[i_conv_layer].weight,
                b_posterior_mean_init=net_posterior_mean_init.conv_layers[i_conv_layer].bias,
                stride=net_prior_mean.conv_layers[i_conv_layer].stride,
                padding=net_prior_mean.conv_layers[i_conv_layer].padding,
                dilation=net_prior_mean.conv_layers[i_conv_layer].dilation,
                groups=net_prior_mean.conv_layers[i_conv_layer].groups
            )
        )

    for i_linear_layer in range(n_linear_layers):
        if i_linear_layer == n_linear_layers - 1:
            activation = 'softmax'
        else:
            activation = 'relu'
        bayesian_layers.append(
            BayesianLinear(
                in_features=net_prior_mean.linear_layers[i_linear_layer].in_features,
                out_features=net_prior_mean.linear_layers[i_linear_layer].out_features,
                activation=activation,
                prior_stddev=prior_stddev,
                optimize_prior_mean=optimize_prior_mean,
                optimize_prior_rho=optimize_prior_rho,
                optimize_posterior_mean=optimize_posterior_mean,
                optimize_posterior_rho=optimize_posterior_rho,
                w_prior_mean_init=net_prior_mean.linear_layers[i_linear_layer].weight,
                b_prior_mean_init=net_prior_mean.linear_layers[i_linear_layer].bias,
                w_posterior_mean_init=net_posterior_mean_init.linear_layers[i_linear_layer].weight,
                b_posterior_mean_init=net_posterior_mean_init.linear_layers[i_linear_layer].bias
            )
        )

    return BayesianClassifier(
        bayesian_layers=bayesian_layers,
        prob_threshold=prob_threshold,
        normalize_surrogate_by_log_classes=normalize_surrogate_by_log_classes,
        oracle_prior_variance=oracle_prior_variance
    )
