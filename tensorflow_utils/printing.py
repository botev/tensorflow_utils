import tensorflow as tf


__all__ = [
    "print_params",
]


def print_params(title,
                 scope=None,
                 key=tf.GraphKeys.GLOBAL_VARIABLES):
    """
    Prints all of the variables in the provided scope and the combined total
    number of elements.

    Args:
        title: String.  A title name to print at the top.
        scope: String.  Scope for which to print the variables.
        key: GraphKey.  Which GraphKey to use.
    """
    print("\n{0:=<78}".format(title))
    total = 0
    for p in tf.get_collection(key, scope):
        print(p.name, ":", p.shape)
        total += p.shape.num_elements()
    print("Total: {0:=<71,d}".format(total))
    return total
