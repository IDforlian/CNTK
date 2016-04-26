# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Non-linear operations. For every operation we explain how the forward and backward
passes are computed. For the backward pass we just explain the scalar case which is the building 
block for computing tensor gradients using the chain rule. For tensors, the backward pass of a node 
is computed as follows : for each element in the output tensor, its gradient with respect to the
given input tensor is computed, then, the resulting tensors are added up.
"""

from cntk.ops.cntk1 import Abs, Clip, Exp, RectifiedLinear, Sigmoid, Softmax, Tanh

def clip(min_value, max_value, x, name=None):
    """
    Clips tensor values to fall between `min_value` and `max_value`.
    For the input tensor `x`, this node outputs a tensor of the same shape with 
    all of its values clipped to fall between `min_value` and `max_value`.
    The backward pass propagates the received gradient if no clipping occurred,
    and 0 if the value was clipped.
    
    Example:
        >>> clip(2., 4., [1., 2.1, 3.0, 4.1])
        #[2.0, 2.1, 3.0, 4.0]
        
        >>> clip([-5., -4., 0., 3., 5.], [5., 4., 1., 4., 9.], [-10., -5., 0., 5., 10.])
        #[-5, -4., 0., 4., 9.]
    
    Args:        
        min_value: the minimum value to clip element values to
        max_value: the maximum value to clip element values to
        x: tensor to be clipped
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    
    return Clip(min_value, max_value, x, var_name = name)

def rectified_linear(x, name=None):
    """
    computes the element-wise rectified linear of `x`: ``max(x, 0)``

    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor

    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    return RectifiedLinear(x, var_name=name)


def rectified_linear(x, name=None):
    """
    computes the element-wise rectified linear of `x`: ``max(x, 0)``

    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor

    Returns:
        :class:`cntk.graph.ComputationNode`

    Example:
        >>> cntk.eval(cntk.ops.rectified_linear([[-1, -0.5, 0, 1, 2]]))
        [[[0, 0, 0, 1, 2]]]
    """
    return RectifiedLinear(x, var_name=name)


def sigmoid(x, name=None):
    """
    computes the element-wise sigmoid of `x`: 

    :math:`sigmoid(x) = {1 \over {1+\exp(-x)}}`

    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor

    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    return Sigmoid(x, var_name=name)


def tanh(x, name=None):
    """
    computes the element-wise tanh of `x`: 

    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor

    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    return Tanh(x, var_name=name)


def softmax(x, name=None):
    """
    computes the element-wise sigmoid of `x`: 

    :math:`softmax(x) = {\exp(x_i) - \max_{x_i \in x}(\exp(x_i)) \over {\sum_{x_i \in x} \exp(x_i)- \max_{x_i \in x}(\exp(x_i)) }}`

    The term :math:`\max_{x_i \in x}(\exp(x_i))` is subtracted for numerical
    stability.

    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor

    Returns:
        :class:`cntk.graph.ComputationNode`

    Examples:
        >>> cntk.eval(cntk.ops.softmax([[1, 1, 2, 3]]))
        [[[0.08259454, 0.08259454, 0.22451524, 0.61029569]

        >>> cntk.eval(cntk.ops.softmax([[1, 1]]))
        [[[0.5, 0.5]]]
    """
    return Softmax(x)


def exp(x, name=None):
    """
    computes the element-wise exponential of `x`: 

    :math:`exp(x) = {e^x}`

    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor

    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    return Exp(x, var_name=name)


def abs(x, name=None):
    """
    computes the element-wise absolute of `x`: 

    :math:`abs(x) = |x|`

    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor

    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    return Abs(x, var_name=name)

from cntk.ops.cntk1 import If

def cond(flag, value_if_true, value_if_false, name=None):
    """
    Return either value_if_true or value_if_false based on the value of flag.
    If flag != 0 value_if_true is returned, otherwise value_if_false.
    Behaves analogously to numpy.where(...).

    Example:
    >>> cond([-10, -1, 0, 0.3, 100], [1, 10, 100, 1000, 10000], [ 2, 20, 200, 2000, 20000])
    # [1, 10, 200, 1000, 10000]

    Args:
        flag: tensor
        value_if_true: tensor
        value_if_false: tensor
        name: the name of the node in the network          
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    
    return If(flag, value_if_true, value_if_false, var_name = name)
