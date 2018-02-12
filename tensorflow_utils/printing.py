import tensorflow as tf


__all__ = [
    "load_parameters",
]


def load_parameters(saver, session, load_dir):
    if tf.gfile.Exists(load_dir):
        ckpt = tf.train.get_checkpoint_state(load_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
        else:
            tf.gfile.DeleteRecursively(load_dir)
            tf.gfile.MakeDirs(load_dir)
    else:
        tf.gfile.MakeDirs(load_dir)
