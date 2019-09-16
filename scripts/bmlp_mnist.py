import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sonnet as snt
import sonnet.python.custom_getters.bayes_by_backprop as bbb

import collections
import os
import ipdb
from tqdm import tqdm

# hyperparameters
train_batch_size = 128
test_batch_size = 10000
hidden_layer_sizes = [400, 400]

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_float('prior_stddev', np.exp(-3), 'prior stddev')
tf.flags.DEFINE_float('lr_start', 0.8, 'learning rate initialization')
tf.flags.DEFINE_float("lr_decay", 0.95, "Polynomical decay power.")
tf.flags.DEFINE_integer("high_lr_epochs", 20, "Number of epochs with lr_start.")
tf.flags.DEFINE_float('momentum', 0.9, 'momentum')
tf.flags.DEFINE_integer('n_train_epochs', 100, 'number of training epochs')
tf.flags.DEFINE_float('min_prob', 0.01, 'probability threshold')

tf.flags.DEFINE_string("logbasedir", "/h/kylehsu/output/pacbayes_opt/mnist", "directory for logs")
tf.flags.DEFINE_string("logsubdir", "prior_logstddev-3_lr0.8_lrdecay0.95_highlrepochs20_momentum0.9_minprob0.01", "subdirectory for this experiment.")
tf.flags.DEFINE_string(
    "mode", "train_only",
    "What mode to run in. Options: ['train_only', 'test_only', 'train_test']")

np.random.seed(42)
tf.random.set_random_seed(43)


def _run_session_with_no_hooks(sess, *args, **kwargs):
    """Only runs of the training op should contribute to speed measurement."""
    return sess._tf_sess().run(*args, **kwargs)


def get_data(name, train_batch_size, test_batch_size):
    """Gets training and testing dataset iterators.

    Args:
      name: String. Name of dataset, either 'mnist' or 'cifar10'.
      train_batch_size: Integer. Batch size for training.
      test_batch_size: Integer. Batch size for testing.

    Returns:
      Dict containing:
        train_iterator: A tf.data.Iterator, over training data.
        test_iterator: A tf.data.Iterator, over test data.
        num_classes: Integer. Number of class labels.
    """
    if name not in ['mnist', 'cifar10']:
        raise ValueError(
            'Expected dataset \'mnist\' or \'cifar10\', but got %s' % name)
    dataset = getattr(tf.keras.datasets, name)
    n_classes = 10

    # Extract the raw data.
    raw_data = dataset.load_data()
    (images_train, labels_train), (images_test, labels_test) = raw_data

    # Normalize inputs and fix types.
    images_train = images_train.astype(np.float32) / 255.
    images_test = images_test.astype(np.float32) / 255.
    labels_train = labels_train.astype(np.int32)
    labels_test = labels_test.astype(np.int32)

    # Add a dummy 'color channel' dimension if it is not present.
    if images_train.ndim == 3:
        images_train = np.expand_dims(images_train, -1)
        images_test = np.expand_dims(images_test, -1)

    # Put the data onto the graph as constants.
    train_data = tf.data.Dataset.from_tensor_slices((images_train, labels_train))
    test_data = tf.data.Dataset.from_tensor_slices((images_test, labels_test))

    # Create iterators for each dataset.
    train_iterator = (
        train_data
            # Note: For larger datasets e.g. ImageNet, it will not be feasible to have
            # a shuffle buffer this large.
            .shuffle(buffer_size=len(images_train))
            .batch(train_batch_size, drop_remainder=True)
            .make_initializable_iterator()
        # .repeat()
        # .make_one_shot_iterator()
    )
    test_iterator = test_data.batch(test_batch_size).make_initializable_iterator()
    return dict(
        train_iterator=train_iterator,
        test_iterator=test_iterator,
        n_classes=n_classes,
        n_train_data=images_train.shape[0],
        n_test_data=images_test.shape[0]
    )


def dense_posterior_builder(getter, name, *args, **kwargs):
    """A builder for a particular diagonal gaussian posterior.
    Args:
      getter: The `getter` passed to a `custom_getter`. Please see the
        documentation for `tf.get_variable`.
      name: The `name` argument passed to `tf.get_variable`.
      *args: Positional arguments forwarded by `tf.get_variable`.
      **kwargs: Keyword arguments forwarded by `tf.get_variable`.
    Returns:
      An instance of `tfp.distributions.Distribution` representing the
      posterior distribution over the variable in question.
    """
    del args
    parameter_shapes = tfp.distributions.Normal.param_static_shapes(kwargs['shape'])

    prior_stddev = np.sqrt(
        FLAGS.prior_stddev
    )

    loc_var = getter(
        f'{name}/posterior_loc',
        shape=parameter_shapes['loc'],
        initializer=kwargs.get('initializer'),
        dtype=tf.float32
    )
    scale_var = getter(
        f'{name}/posterior_scale',
        initializer=tf.random_uniform(
            minval=np.log(np.exp(prior_stddev / 2.0) - 1.0),
            maxval=np.log(np.exp(prior_stddev / 1.0) - 1.0),
            dtype=tf.float32,
            shape=parameter_shapes['scale']
        )
    )
    return tfp.distributions.Normal(
        loc=loc_var,
        scale=tf.nn.softplus(scale_var) + 1e-5,
        name=f'{name}/posterior_dist'
    )


def fixed_gaussian_prior_builder(
        getter, name, dtype=None, *args, **kwargs):
    """A pre-canned builder for fixed gaussian prior distributions.

    Given a true `getter` function and arguments forwarded from `tf.get_variable`,
    return a distribution object for a scalar-valued fixed gaussian prior which
    will be broadcast over a variable of the requisite shape.

    Args:
      getter: The `getter` passed to a `custom_getter`. Please see the
        documentation for `tf.get_variable`.
      name: The `name` argument passed to `tf.get_variable`.
      dtype: The `dtype` argument passed to `tf.get_variable`.
      *args: See positional arguments passed to `tf.get_variable`.
      **kwargs: See keyword arguments passed to `tf.get_variable`.

    Returns:
      An instance of `tfp.distributions.Normal` representing the prior
      distribution over the variable in question.
    """
    del getter  # Unused.
    del args  # Unused.
    del kwargs  # Unused.
    loc = tf.constant(0.0, shape=(), dtype=dtype)
    scale = tf.constant(FLAGS.prior_stddev, shape=(), dtype=dtype)
    return tfp.distributions.Normal(
        loc=loc, scale=scale, name="{}_prior_dist".format(name))


def build_net(is_training):
    if is_training:
        estimator_mode = tf.constant(bbb.EstimatorModes.sample)
    else:
        estimator_mode = tf.constant(bbb.EstimatorModes.mean)

    dense_bbb_custom_getter = bbb.bayes_by_backprop_getter(
        posterior_builder=dense_posterior_builder,
        prior_builder=fixed_gaussian_prior_builder,
        kl_builder=bbb.stochastic_kl_builder,
        sampling_mode_tensor=estimator_mode
    )

    return snt.nets.MLP(
        output_sizes=hidden_layer_sizes + [10],
        activation=tf.nn.relu,
        activate_final=False,
        use_bias=True,
        use_dropout=False,
        custom_getter=dense_bbb_custom_getter
    )
    # TODO: no BBB on biases of last layer?

    # n_input = 784
    # layers = []
    # for i in range(n_hidden_layers):
    #     layers.append(
    #         snt.Linear(
    #             hidden_layer_size,
    #             use_bias=True,
    #             custom_getter=dense_bbb_custom_getter,
    #         ))
    # layers.append(
    #     snt.Linear(
    #         10,
    #         use_bias=True,
    #         custom_getter=dense_bbb_custom_getter
    #     )
    # )
    # return layers


def build_logits(net, data):
    return net(data)


def build_loss(logits, labels):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    return tf.reduce_mean(loss)


def build_metrics(prefix, logits, labels):
    predictions = tf.cast(
        tf.argmax(logits, axis=-1), tf.int32, name=f'{prefix}_pred'
    )
    correct_prediction_mask = tf.cast(
        tf.equal(predictions, labels), tf.int32)
    correct = tf.reduce_sum(
        tf.cast(correct_prediction_mask, tf.float32))
    accuracy = tf.reduce_mean(
        tf.cast(correct_prediction_mask, tf.float32), name=f'{prefix}_acc')
    error_rate = tf.subtract(1.0, accuracy, name=f'{prefix}_err')
    label_probs = tf.nn.softmax(logits, dim=-1)
    predictive_entropy = tf.reduce_mean(
        label_probs * tf.log(label_probs + 1e-12) * -1.0)
    return correct, accuracy, error_rate, predictive_entropy


def train(logdir):
    data_dict = get_data('mnist', train_batch_size, test_batch_size)
    train_data = data_dict["train_iterator"]
    test_data = data_dict["test_iterator"]
    n_train_data = data_dict['n_train_data']
    n_test_data = data_dict['n_test_data']

    train_images, train_labels = train_data.get_next()
    test_images, test_labels = test_data.get_next()

    # Flatten images to pass to model.
    train_images = snt.BatchFlatten()(train_images)
    test_images = snt.BatchFlatten()(test_images)

    # connect to training set
    net = build_net(
        is_training=True
    )
    logits = build_logits(net, train_images)
    probs = tf.nn.softmax(logits, axis=-1)
    clipped_probs = tf.math.maximum(FLAGS.min_prob, probs)
    logits = tf.log(clipped_probs)

    data_loss = build_loss(logits, train_labels)

    total_kl_cost = bbb.get_total_kl_cost()
    scaled_kl_cost = total_kl_cost / n_train_data
    total_loss = tf.add(scaled_kl_cost, data_loss)

    # Optimize as usual.
    global_step = tf.get_variable(
        "num_weight_updates",
        initializer=tf.constant(0, dtype=tf.int32, shape=()),
        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

    learning_rate = tf.get_variable(
        "lr", initializer=tf.constant(FLAGS.lr_start, shape=(), dtype=tf.float32))
    learning_rate_update = learning_rate.assign(learning_rate * FLAGS.lr_decay)

    # optimizer = tf.train.GradientDescentOptimizer(
    #     learning_rate=learning_rate)
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=FLAGS.momentum
    )

    with tf.control_dependencies([optimizer.minimize(total_loss)]):
        global_step_and_train = global_step.assign_add(1)

    test_logits = build_logits(net, test_images)
    test_loss = build_loss(test_logits, test_labels)

    train_correct, train_accuracy, train_error_rate, train_predictive_entropy = build_metrics(
        'train', logits, train_labels)
    test_correct, test_accuracy, test_error_rate, test_predictive_entropy = build_metrics(
        'test', test_logits, test_labels)

    # Create tf.summary ops.
    log_ops_to_run = {
        "scalar": collections.OrderedDict([
            ("task_loss", data_loss),
            ('train accuracy', train_accuracy),
            ("train_err_rate", train_error_rate),
            ("pred_entropy", train_predictive_entropy),
            ("learning_rate", learning_rate),
            ("elbo_loss", total_loss),
            ("kl_cost", total_kl_cost),
            ("scaled_kl_cost", scaled_kl_cost),
        ]),
    }

    for name, tensor in log_ops_to_run["scalar"].items():
        tf.summary.scalar(os.path.join("train", name), tensor)

    # The remaining logic runs the training loop and logging.
    summary_writer = tf.summary.FileWriterCache.get(logdir=logdir)
    tf.logging.info(
        "Beginning training for {} epochs, each with {} batches.".format(
            FLAGS.n_train_epochs, n_train_data // train_batch_size))
    # with tf.train.MonitoredTrainingSession(
    #         is_chief=True, checkpoint_dir=logdir, save_summaries_secs=10) as sess:
    with tf.train.SingularMonitoredSession(
        checkpoint_dir=logdir
    ) as sess:
        num_updates_v = _run_session_with_no_hooks(sess, global_step)
        epoch_idx_start, step_idx_start = divmod(
            num_updates_v, n_train_data // train_batch_size)
        tf.logging.info("On start, epoch: {}\t step: {}".format(
            epoch_idx_start, step_idx_start))

        for epoch_idx in tqdm(range(epoch_idx_start, FLAGS.n_train_epochs)):
            # print(f'epoch {epoch_idx + 1}')
            tf.logging.info("Beginning Epoch {}/{}".format(
                epoch_idx, FLAGS.n_train_epochs))
            tf.logging.info(
                ("Beginning by evaluating on the test set, which has "
                 "{} batches.".format(n_test_data // test_batch_size)))
            test_correct_all = 0
            # _run_session_with_no_hooks(sess, zero_valid_state)

            sess.run(test_data.initializer)
            while True:
                try:
                    test_correct_v, num_updates_v = _run_session_with_no_hooks(
                        sess, [test_correct, global_step])
                    test_correct_all += test_correct_v
                except tf.errors.OutOfRangeError:
                    break
            tf.logging.info(f'test set accuracy: {test_correct_all / n_test_data}')

            summary = tf.summary.Summary()
            summary.value.add(
                tag="test/accuracy",
                simple_value=test_correct_all / n_test_data)
            summary_writer.add_summary(summary, num_updates_v)

            # Run a training epoch.
            epoch_cost = 0
            epoch_correct = 0

            sess.run(train_data.initializer)
            batch_idx = 0
            while True:
                try:
                    scalars_res, train_correct_v, num_updates_v = sess.run(
                        [log_ops_to_run['scalar'], train_correct, global_step_and_train]
                    )
                    epoch_correct += train_correct_v
                    epoch_cost += scalars_res['task_loss']
                    batch_idx += 1
                except tf.errors.OutOfRangeError:
                    break
            summary = tf.summary.Summary()
            summary.value.add(
                tag='train/accuracy',
                simple_value=epoch_correct / float(n_train_data)
            )
            summary_writer.add_summary(summary, num_updates_v)
            tf.logging.info("Num weight updates: {}".format(num_updates_v))

            for name, result in scalars_res.items():
                tf.logging.info("{}_batch: {}".format(name, result))
                summary = tf.summary.Summary()
                summary.value.add(
                    tag=f'train/{name}',
                    simple_value=result
                )
                summary_writer.add_summary(summary, num_updates_v)

            if epoch_idx >= FLAGS.high_lr_epochs:
                _run_session_with_no_hooks(sess, [learning_rate_update])

            summary_writer.flush()


    tf.logging.info("Done training. Thanks for your time.")


def test(logdir):
    raise NotImplementedError


def main():
    logdir = os.path.join(FLAGS.logbasedir, FLAGS.logsubdir)
    tf.logging.info("Log Directory: {}".format(logdir))
    if FLAGS.mode == "train_only":
        train(logdir)
    elif FLAGS.mode == "test_only":
        test(logdir)
    elif FLAGS.mode == "train_test":
        tf.logging.info("Beginning a training phase of {} epochs.".format(
            FLAGS.num_training_epochs))
        train(logdir)
        tf.logging.info("Beginning testing phase.")
        with tf.Graph().as_default():
            # Enter new default graph so that we can read variables from checkpoint
            # without getting hit by name uniquification of sonnet variables.
            test(logdir)
    else:
        raise ValueError("Invalid mode {}. Please choose one of {}.".format(
            FLAGS.mode, "['train_only', 'test_only', 'train_test']"))


if __name__ == "__main__":
    main()
