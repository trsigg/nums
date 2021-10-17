import numpy as np
import string
from nums.experimental.optimizer.graph import Leaf, FusionNode, BinaryOp

def get_expr(node):
    all_variables = []
    def get_expr_helper(node):
        nonlocal all_variables
        if not node:
            return ""
        if node and type(node) == Leaf:
            if node.node_type == "Variable":
                if node.rep_value not in all_variables:
                    all_variables.append(node.rep_value)
                return node.rep_value
            if node.node_type == "Literal":
                return node.block.get()
        expr = ""
        for i in range(len(node.get_children())):
            child = node.get_children()[i]
            expression = get_expr_helper(child)
            if i != len(node.children) - 1:
                expr += "(" + str(expression) + ")" + node.rep_value
            else:
                expr += "(" + str(expression) + ")"
        return expr
    final_expr = get_expr_helper(node)
    variables = ""
    for i in range(len(all_variables)):
        if i != len(all_variables) - 1:
            variables += all_variables[i] + ", "
        else:
            variables += all_variables[i] 
    return "lambda {}: {}".format(variables, final_expr), all_variables

def fuse_node(node):
    if node.node_type == "Operator":
        expr, all_variables = get_expr(node)
        variable_nodes = []
        for variable in all_variables:
            variable_nodes.append(FusionNode(node.cluster_state, None, variable, "Variable", []))
        return FusionNode(node.cluster_state, None, expr, "Fused", variable_nodes, len(variable_nodes))
    return node

def process_before_fusion(root, seen_variables, alphabets):
    if not root:
        return 0
    if root and type(root) == BinaryOp:
        root.node_type = "Operator"
        root.rep_value = root.bop_symbols[root.op_name]
    if root and type(root) == Leaf:
        if type(root.block.get()) == np.ndarray and root.block.id not in seen_variables:
            var_name = next(alphabets)
            seen_variables[root.block.id] = var_name
            root.rep_value = var_name
            root.node_type = "Variable"
            return 1  
        elif type(root.block.get()) == np.ndarray and root.block.id in seen_variables:
            root.rep_value = seen_variables[root.block.id]
            root.node_type = "Variable"
        elif type(root.block.get()) != np.ndarray:
            root.rep_value = root.block.get()
            root.node_type = "Literal"
        return 0
    for child in root.get_children():
        child.no_args = process_before_fusion(child, seen_variables, alphabets)
        root.no_args += child.no_args
    return root.no_args

def operator_fusion_helper(node, max_vars=2):
    if not node:
        return []
    if node.get_children() != []:
        new_children = [operator_fusion_helper(child, max_vars) for child in node.get_children()]
        if node.no_args > max_vars:
            children = [fuse_node(child) for child in new_children]
        else:
            children = new_children
        node = FusionNode(node.cluster_state, None, node.rep_value, node.node_type, children, node.no_args)
    return node

def operator_fusion(root, max_vars):
    no_of_vars = process_before_fusion(root, dict(), iter(list(string.ascii_lowercase)))
    new_root = operator_fusion_helper(root, max_vars)
    if new_root.no_args <= max_vars:
        new_root = fuse_node(new_root)
    return new_root