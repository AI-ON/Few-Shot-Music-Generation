import os
import tensorflow as tf
import pprint

from models.base_model import BaseModel

PP = pprint.PrettyPrinter(depth=6)


def start_session():
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    return tf.Session(config=sess_config)


def init_vars_op(sess):
    variables = tf.global_variables()
    init_flag = sess.run(
        tf.stack([tf.is_variable_initialized(v) for v in variables]))
    uninit_variables = [v for v, f in zip(variables, init_flag) if not f]

    print('Initializing vars:')
    print(PP.pformat([v.name for v in uninit_variables]))

    return tf.variables_initializer(uninit_variables)


def optimistic_restore(session, save_file, only_load_trainable_vars=False,
                       stat_name_prefix='moving_'):
    """Restore variables of model from save_file.

    Argument trainable_vars is used to determine whether to fetch model
    variables & batch-norm statistics OR whether to fetch ALL variables
    (includes model variables, batch-norm statistics, training variables ~
    such as global step and learning rate decay).

    Args:
        session: tf.Session to use in recovery
        save_file: file to load variables from
        trainable_vars: to recover only variables that are trained (model
            variables and running batch-norm statistics) or not (all variables
            including global step)
    """
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    if only_load_trainable_vars:
        var_names = sorted([(var.name, var.name.split(':')[0])
                            for var in tf.trainable_variables()
                            if var.name.split(':')[0] in saved_shapes])
        running_stat_names = sorted([(var.name, var.name.split(':')[0])
                                     for var in tf.global_variables()
                                     if (var.name.split(':')[0]
                                         in saved_shapes and
                                         stat_name_prefix in var.name)])
        var_names += running_stat_names
    else:
        var_names = sorted([(var.name, var.name.split(':')[0])
                            for var in tf.global_variables()
                            if var.name.split(':')[0] in saved_shapes])

    print('Loading vars:')
    print(PP.pformat(var_names))

    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(
        ':')[0], tf.global_variables()), tf.global_variables()))

    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


class TFModel(BaseModel):

    def __init__(self, config):
        tf.set_random_seed(config['seed'])

        super(TFModel, self).__init__(config)
        self._sess = start_session()

        with tf.variable_scope(self.name):
            self._global_step = tf.Variable(0, trainable=False)
            self._define_placedholders()
            self._build_graph()

        self._saver = tf.train.Saver(self.get_vars(only_trainable=False),
                                     max_to_keep=10)

    def get_vars(self, name=None, only_trainable=True):
        name = name or self.name
        if only_trainable:
            return [v for v in tf.trainable_variables() if name in v.name]
        else:
            return [v for v in tf.global_variables() if name in v.name]

    def _get_checkpt_prefix(self, checkpt_path):
        directory = os.path.join(checkpt_path, self.name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return os.path.join(directory, self.name)

    def save(self, checkpt_path):
        self._saver.save(self._sess, self._get_checkpt_prefix(checkpt_path),
                         global_step=self._global_step)

    def _recover(self, checkpt_path, only_load_trainable_vars):
        latest_checkpt = tf.train.latest_checkpoint(
            os.path.join(checkpt_path, self.name)
        )
        if latest_checkpt is None:
            return False

        print('recovering %s from %s' % (self.name, latest_checkpt))
        optimistic_restore(self._sess, latest_checkpt, only_load_trainable_vars)
        return True

    def recover_or_init(self, checkpt_path, only_load_trainable_vars=False):
        self._recover(checkpt_path, only_load_trainable_vars)
        self._sess.run(init_vars_op(self._sess))
