#!/usr/bin/env python3
"""Building a Decision Tree."""
import numpy as np


class Node:
    """A node class that generalizes everything including root and leaves."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Construct the Node object."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Find the maximum depth."""
        if self.is_leaf:
            return self.depth
        left = self.left_child.max_depth_below()\
            if self.left_child else self.depth
        right = self.right_child.max_depth_below()\
            if self.right_child else self.depth
        return max(left, right)

    def count_nodes_below(self, only_leaves=False):
        """Count the number of nodes below, only leaves if specified."""
        if self.is_leaf:
            return 1

        if only_leaves:
            left = self.left_child.count_nodes_below(True)\
                if self.left_child else 0
            right = self.right_child.count_nodes_below(True)\
                if self.right_child else 0
            return left + right
        else:
            left = self.left_child.count_nodes_below(False)\
                if self.left_child else 0
            right = self.right_child.count_nodes_below(False)\
                if self.right_child else 0
            return 1 + left + right

    def __str__(self):
        """Return an ASCII representation of the tree from this node."""
        if self.is_root:
            s = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            s = f"node [feature={self.feature}, threshold={self.threshold}]"

        if self.left_child:
            left_str = self.left_child.__str__()
            s += "\n" + self.left_child_add_prefix(left_str).rstrip("\n")

        if self.right_child:
            right_str = self.right_child.__str__()
            s += "\n" + self.right_child_add_prefix(right_str).rstrip("\n")

        return s

    def left_child_add_prefix(self, text):
        """Add ASCII branch prefixes for a left child in a tree diagram."""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Add ASCII branch prefixes for a right child in a tree diagram."""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def get_leaves_below(self):
        """Return the leaves of a node."""
        if self.is_leaf:
            return [self]

        list_of_leaves = []
        if self.left_child:
            list_of_leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            list_of_leaves.extend(self.right_child.get_leaves_below())
        return list_of_leaves

    def update_bounds_below(self):
        """Recursively compute bounds for each node (single feature)."""
        if self.is_root:
            self.lower = {0: -np.inf}
            self.upper = {0: np.inf}

        for child in [self.left_child, self.right_child]:
            if not child:
                continue

            child.lower = self.lower.copy()
            child.upper = self.upper.copy()

            feature = self.feature
            threshold = self.threshold

            if child is self.left_child:
                child.lower[feature] = threshold
            else:
                child.upper[feature] = threshold

        for child in [self.left_child, self.right_child]:
            if child:
                child.update_bounds_below()

    def update_indicator(self):
        """Return an indicator array into indicator attribute."""

        def is_large_enough(x):
            """Check ith individual has all ftrs bigger than lower bounds."""
            checks = [x[:, key] > self.lower[key]
                      for key in self.lower.keys()]
            return np.all(np.array(checks), axis=0)

        def is_small_enough(x):
            """Check ith individual has all ftrs less than upper bounds."""
            checks = [x[:, key] <= self.upper[key]
                      for key in self.upper.keys()]
            return np.all(np.array(checks), axis=0)

        self.indicator = lambda x: \
            np.logical_and(is_large_enough(x), is_small_enough(x))

    def pred(self, x):
        """Predict for a single node."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """Terminal node which is a leaf."""

    def __init__(self, value, depth=None):
        """Construct the leaf object."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Retur the depth of the leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Return the count of 1 leaf."""
        return 1

    def __str__(self):
        """Print the ASCII representation of a leaf."""
        return f"leaf [value={self.value}]"

    def get_leaves_below(self):
        """Return the leaf object."""
        return [self]

    def update_bounds_below(self):
        """Bound of a leaf."""
        pass

    def pred(self, x):
        """Return the value of leaf as a prediction."""
        return self.value


class Decision_Tree():
    """The whole Decision Tree class."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Construct the decision tree."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Return the maximum depth of tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Return the count of leaves."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Print the whole tree in ASCII."""
        return self.root.__str__() + "\n"

    def get_leaves(self):
        """Return the leaves of the decision tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Bounds of the whole tree."""
        self.root.update_bounds_below()

    def pred(self, x):
        """Make a prediction for whole tree."""
        return self.root.pred(x)

    def update_predict(self):
        """Vectorize the predict function."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict_fn(a):
            """Extend the lambda function."""
            y_pred = np.empty(a.shape[0], dtype=int)
            for leaf in leaves:
                mask = leaf.indicator(a)
                y_pred[mask] = leaf.value
            return y_pred

        self.predict = predict_fn

    def fit(self, explanatory, target, verbose=0):
        """Train the decision tree model."""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
- Depth                     : { self.depth()       }
- Number of nodes           : { self.count_nodes() }
- Number of leaves          : { self.count_nodes(only_leaves=True) }
- Accuracy on training data : { self.accuracy(self.explanatory,self.target)    }
          """)

    def np_extrema(self, arr):
        """Return the minimum and maximum of the array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Split the population based on random criterion."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(self.explanatory[:, feature][node.sub_population])
            diff = feature_max-feature_min
        x = self.rng.uniform()
        threshold = (1-x)*feature_min + x*feature_max
        return feature, threshold

    def fit_node(self, node):
        """Recursively split node into children until leaves are reached."""
        node.feature, node.threshold = self.split_criterion(node)

        # Subset of current node
        x_node = self.explanatory[node.sub_population]
        y_node = self.target[node.sub_population]

        # Split
        mask_left = x_node[:, node.feature] > node.threshold
        mask_right = ~mask_left

        # Convert back to global masks
        left_population = np.zeros_like(self.target, dtype=bool)
        left_population[node.sub_population] = mask_left

        right_population = np.zeros_like(self.target, dtype=bool)
        right_population[node.sub_population] = mask_right

        # Decide if left child is leaf
        is_left_leaf = (
            left_population.sum() < self.min_pop
            or node.depth == self.max_depth
            or np.all(y_node[mask_left] == y_node[mask_left][0])  # pure class
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Same for right child
        is_right_leaf = (
            right_population.sum() < self.min_pop
            or node.depth == self.max_depth
            or np.all(y_node[mask_right] == y_node[mask_right][0])
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Create a leaf node with majority value from sub_population."""
        y_sub = self.target[sub_population]
        values, counts = np.unique(y_sub, return_counts=True)
        value = values[np.argmax(counts)]  # majority class
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create an internal node for the given sub_population."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Compute accuracy score on test data."""
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target))/test_target.size
