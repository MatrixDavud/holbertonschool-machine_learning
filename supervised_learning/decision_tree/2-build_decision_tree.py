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
