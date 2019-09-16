import sonnet as snt
import sonnet.python.custom_getters.bayes_by_backprop as bbb
import tensorflow as tf
import tensorflow_probability as tfp


# Use a custom prior builder.
def custom_prior_builder(getter, name, *args, **kwargs):
    return tfp.distributions.Normal(0.0, 0.01)


# Use pre-canned builders for diagonal gaussian posterior and stochastic KL.
get_bbb_variable_fn = bbb.bayes_by_backprop_getter(
    prior_builder=custom_prior_builder,
    posterior_builder=bbb.diagonal_gaussian_posterior_builder,
    kl_builder=bbb.stochastic_kl_builder)


# Demonstration of how to use custom_getters with variable scopes.
with tf.variable_scope('network', custom_getter=get_bbb_variable_fn):
    model = snt.Linear(4)
    # This approach is compatible with all `tf.Variable`s constructed with
    # `tf.get_variable()`, not just those contained in sonnet modules.
    # noisy_variable = tf.get_variable('w', shape=(5,), dtype=tf.float32)
    noisy_variable = tf.get_variable('w', shape=(5,), initializer=tf.initializers.constant(1.0))
not_noisy_variable = tf.get_variable('w', shape=(5,), initializer=tf.initializers.constant(1.0))

if __name__ == '__main__':
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    x, y = sess.run([noisy_variable, not_noisy_variable])
    print(x, y)