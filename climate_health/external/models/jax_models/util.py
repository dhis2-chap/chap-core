from .jax import jax, blackjax


def extract_last(samples):
    i = -1
    return extract_sample(i, samples)


def extract_sample(i, samples):
    return {key: value[i] if not hasattr(value, 'items') else extract_last(value) for key, value in
                   samples.items()}


def index_tree(tree, index):
    if isinstance(tree, jax.Array):
        return tree[index]
    elif hasattr(tree, 'items'):
        return {key: index_tree(value, index) for key, value in tree.items()}
    else:
        return index_tree(next(iter(tree)), index)


def array_tree_length(tree):
    if isinstance(tree, jax.Array):
        return len(tree)
    elif hasattr(tree, 'items'):
        return array_tree_length(next(iter(tree.values())))
    else:
        return array_tree_length(next(iter(tree)))
