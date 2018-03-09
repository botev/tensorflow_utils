import tensorflow as tf
import shutil
import os

__all__ = [
    "load_parameters",
]


def load_parameters(saver, session, load_dir, bootstrap_folder=True, force_delete=False):
    if tf.gfile.Exists(load_dir):
        ckpt = tf.train.get_checkpoint_state(load_dir)
        if ckpt and ckpt.model_checkpoint_path:
            if bootstrap_folder:
                path = os.path.join(load_dir, os.path.basename(ckpt.model_checkpoint_path))
            else:
                path = ckpt.model_checkpoint_path
            saver.restore(session, path)
        elif force_delete:
            shutil.rmtree(load_dir)
            tf.gfile.MakeDirs(load_dir)
    else:
        tf.gfile.MakeDirs(load_dir)
