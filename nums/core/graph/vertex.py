from typing import Union, Optional, List, Set, Tuple
from nums.core.graph.utils import Counter


class Vertex(object):

    def __init__(self, counter: Counter, vid: int = None):
        self.counter: Counter = counter
        self.vid = (
            self.counter() if vid is None else vid
        )
        # Some things to note when supporting multiple parent nodes:
        # Let C = A + B
        # The expression D = 2 * C; E = 3*C
        # will generate node D and E, and C will point to both as parents.
        # The expression F = C*C
        # will generate F, and C will point to F as a parent.
        # The reference from C to F will not be duplicated.
        # This is why self.parents is a set.
        #
        # With our current set of supported operations, circular dependencies are not possible.
        # We would need some scenario where X has as a parent some descendent of X.
        # We only designate newly created nodes as parents.
        # => No descendent of a vertex will ever be designated as a parent.
        # => No newly created node can refer to itself.
        # => All graphs are DAGs.
        #
        # We currently do not preprocess graphs.
        # When we do, we will need to prove that no cycles occur for the set
        # of mutation operations.
        #
        # The only mutation we plan to support is fusion.
        # Fused ops will need to be aware of situations where same var is mentioned multiple times.
        # Even then, there is no chance of circular reference.
        # The fusion process starts from leaves and move through parent nodes,
        # greedily fusing operations that depend on e.g. at most 2 leaves.
        #
        # When a node has e.g. two parents, and one parent is fuseable,
        # then fuse it and update the node to swap out the
        # old parent reference with the fused parent reference.
        # This needs to be done breadth-first.
        # If the other parent of the node is also fuseable, then fuse.
        # If the fused parents feed into the same operation,
        # then fusion needs to be possible with fused nodes so that
        # the fused nodes fuse into a single fused operation.
        # in this scenario, the original node's parents will be updated,
        # and one of the updates will remove and then re-add the final fused op,
        # converting the original node to something that originally had two parents
        # to a node that now has one parent.

        self.parents: Set[Vertex] = set()
        # Note that, in the context of DAGs,
        # parents are outgoing nodes,
        # and children are in incoming nodes.

    def get_roots(self) -> List:
        if len(self.parents) == 0:
            return [self]
        roots = []
        for parent in self.parents:
            roots += parent.get_roots()
        return roots

    def get_children(self):
        raise NotImplementedError()

    def num_nodes(self):
        raise NotImplementedError()

    def copy(self, counter, new_ids=False):
        raise NotImplementedError()

    def update_child(self, old_child, new_child):
        raise NotImplementedError()

    def update_parent(self, old_parent, new_parent):
        raise NotImplementedError()

    def get_leafs(self):
        raise NotImplementedError()

    def shape(self):
        raise NotImplementedError()

    def __repr__(self):
        return str(self.__class__.__name__)

    def make_elementwise(self, name, others, args=None):
        children = [self] + others
        op: ElementwiseOp = ElementwiseOp(self.counter)
        op.children = children
        op.name = name
        op.args = args
        for child in children:
            assert isinstance(child, Vertex)
            child.parents.add(op)
        return op

    def tensordot(self, other, axes):
        assert isinstance(other, Vertex)
        op: TensorDotOp = TensorDotOp(self.counter)
        op.left = self
        op.right = other
        op.axes = axes
        op.left.parents.add(op)
        op.right.parents.add(op)
        return op

    def __matmul__(self, other):
        return self.make_elementwise("matmul", other)

    def __add__(self, other):
        return self.make_elementwise("add", other)

    def __sub__(self, other):
        return self.make_elementwise("sub", other)

    def __mul__(self, other):
        return self.make_elementwise("mul", other)

    def __truediv__(self, other):
        return self.make_elementwise("truediv", other)

    def __pow__(self, other):
        return self.make_elementwise("pow", other)


class ElementwiseOp(Vertex):
    def __init__(self, counter: Counter, vid: int = None):
        super().__init__(counter, vid)
        self.name = None
        self.args = None


class BinaryOp(ElementwiseOp):

    def __init__(self, counter: Counter, vid: int = None):
        super().__init__(counter, vid)
        self.left: Vertex = None
        self.right: Vertex = None

    def copy(self, counter, new_ids=False):
        # Copy self.
        # Copy references to children and parents.
        # We aren't responsible for whether they're the new/old versions.
        # The nodes themselves are responsible.
        # Update reference to self in children,
        # and reference to self in parents.
        # In the end, every node that is touched will have updated itself,
        # references to it from its children, and references to it from its parents.
        op = ElementwiseOp(counter, None if new_ids else self.vid)
        assert op.vid is not None and (new_ids or op.vid == self.vid)
        op.parents = self.parents
        op.name = self.name
        op.args = None if self.args is None else self.args.copy()
        for child in self.children:

        op.left = self.left.copy(cluster_state, op, new_ids=new_ids)
        op.right = self.right.copy(cluster_state, op, new_ids=new_ids)
        op.copy_on_op = self.copy_on_op
        return op


class TensorDotOp(BinaryOp):
    def __init__(self, counter: Counter, vid: int = None):
        super().__init__(counter, vid)
        self.axes: int = None


class TransposeOp(Vertex):
    pass


def test_vertex_ops():
    pass
