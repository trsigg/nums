# coding=utf-8
# Copyright (C) 2020 NumS Development Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import time

import numpy as np
import scipy.linalg
from scipy.linalg import lapack

from nums.core.array.application import ArrayApplication
from nums.core.storage.storage import BimodalGaussian
from nums.core import linalg
from nums.core.array.blockarray import BlockArray


def test_weakref_basic(app_inst):
    # Test basic weakref assumptions.

    import sys, weakref, gc

    weak_refs = []

    A: BlockArray = app_inst.arange(10, shape=(10,), block_shape=(10,))
    weak_refs.append(weakref.ref(A))
    B: BlockArray = app_inst.arange(10, shape=(10,), block_shape=(10,))

    weak_refs.append(weakref.ref(B))
    weak_refs.append(weakref.ref(app_inst.arange(10, shape=(10,), block_shape=(10,))))

    # True refcount will be one less, because ref count increases due to call to getrefcount.
    # https://docs.python.org/3/library/sys.html#sys.getrefcount

    for i, wref in enumerate(weak_refs):
        if i == 2:
            assert wref() is None
        else:
            refs = sys.getrefcount(wref()) - 1
            print(refs)
            assert refs == 1

    del B
    for i, wref in enumerate(weak_refs):
        if i == 0:
            refs = sys.getrefcount(wref()) - 1
            print(refs)
            assert refs == 1
        else:
            assert wref() is None

    del A
    for i, wref in enumerate(weak_refs):
        assert wref() is None


def test_weakref_operations(app_inst):
    # We need to hold weak references to all BlockArray objects created with NumS.
    # To achieve this and other critical NumS-related features,
    # all NumS operations must pass through a single, or set, of functions
    # which perform the following meta-operations:
    # - Add BlockArray to weak references.
    # - Check BlockArrays which need to be computed. This check is TBD,
    #   but could be e.g. refs to BlockArray are greater than a certain threshold.
    # - For all BlockArrays that need to be computed, check which ones are interdependent.
    #   e.g. of the form C = A + B; D = 2*C; E=3*C.
    #   In this scenario, we want to compute C, and use the result of C to compute D and E.
    #   i.e. we don't want to compute C twice.
    #   When we construct the computation graphs of C, D, and E at the block-level,
    #   we need to construct a meta-graph at the BlockArray-level of C, D, and E.
    #   We then carry out the computation of C, D, and E in the appropriate order.
    #   The subgraph C within D and E needs to be replaced with the computed values of C.
    #   Then, D and E are computed.
    #   When the BlockArrays are computed, the oids of the blocks within each BlockArray are
    #   set, and the graphs reference is deleted.
    import sys, weakref, gc

    class BlockArrayOperationManager(object):
        def __init__(self):
            self.wrefs = []

        def check_weakrefs(self):
            new_wrefs = []
            for i, (tag, wref) in enumerate(self.wrefs):
                if wref() is None:
                    continue
                new_wrefs.append((tag, wref))
                count = sys.getrefcount(wref()) - 1
                print("check_weakrefs", i, tag, count, gc.get_referrers(wref()))
            self.wrefs = new_wrefs

        def append_weakref(self, tag, wref):
            self.wrefs.append((tag, wref))

        def update_weakrefs(self, tag, wref):
            self.check_weakrefs()
            self.append_weakref(tag, wref)

        def create(self):
            return app_inst.arange(10, shape=(10,), block_shape=(10,))

        def add(self, X1, X2):
            return X1 + X2

        def mult(self, X1, X2):
            return X1 * X2

        def op(self, name, tag, *arrays):
            print("\nop", name, tag)
            # What about reference count in *arrays?
            result = getattr(self, name)(*arrays)
            self.update_weakrefs(tag, weakref.ref(result))
            return result

    bam = BlockArrayOperationManager()
    bam.op("create", "A")
    bam.op("create", "B")

    bam.op("add", "C1", bam.op("create", "A1"), bam.op("create", "B1"))

    bam.op("mult", "E2",
           bam.op("create", "D2"),
           bam.op("add", "C2", bam.op("create", "A2"), bam.op("create", "B2")))

    A, B = bam.op("create", "A3"), bam.op("create", "B3")
    C = bam.op("add", "C3", A, B)
    E = bam.op("mult", "E3", bam.op("create", "D3"), C)


if __name__ == "__main__":
    # pylint: disable=import-error
    from nums.core import application_manager
    from nums.core import settings

    settings.system_name = "serial"
    app_inst = application_manager.instance()
    # test_weakref_basic(app_inst)
    test_weakref_operations(app_inst)
