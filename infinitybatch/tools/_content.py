"""
        _                 _
       (_)               | |
 _ __   _  __  __   ___  | |
| '__| | | \ \/ /  / _ \ | |
| |    | |  >  <  |  __/ | |
|_|    |_| /_/\_\  \___| |_|

infinitybatch
-------------

Copyright rixel 2020
Distributed under the MIT License.
See accompanying file LICENSE.
"""



# Modul level constants.
__author__ = 'rixel'
__copyright__ = "Copyright 2020, infinitybatch"
__credits__ = ['rixel']
__license__ = 'MIT'
__version__ = '1.0.0'
__status__ = 'Production'



import torch
from typing import Any, Optional, Union



ERROR_TOLERANCE = 1e-7



class CudaMemorySnapShot(object):
    """
    Provides simple read-only container for memory snapshot of a torch.cuda device.
    -------------------------------------------------------------------------------
    """



    def __init__(self, device_id: Union[torch.device, str, int, None] = None,
                 synchronize: bool = True) -> None:
        """
        Initializes the object
        ----------------------
        @Params: -> device_id: Union[torch.device, str, int, None] = None
                    The device where the snapshot should be taken. If the device
                    is not given it is set to None. The effect of the setting of
                    this parameter fully accords to the behavior of the
                    concerning PyTorch functions.
                 -> synchronize: bool = True
                    Whether or not to apply torch.cuda.synchronize() at the
                    given device before taking the snapshot.
        """

        self.device = device_id
        if synchronize:
            torch.cuda.synchronize(self.device)
        self.total = torch.cuda.get_device_properties(self.device).total_memory
        self.reserved = torch.cuda.memory_reserved(self.device)
        self.allocated = torch.cuda.memory_allocated(self.device)



    @property
    def allocated(self) -> int:
        """
        Property with type of int. It contains the size of the allocated memory
        in bytes at the moment when the snapshot was taken. In fact this
        property is read-only since it can be set at once only, and it is set
        at instantiation. Due to the experimental character of this snapshot,
        trying to give a new value does not raise error however it won't be
        successful.
        -----------
        @Return: -> int
                    The value of allocated.
        """

        return self.__allocated



    @allocated.setter
    def allocated(self, allocated: int) -> None:
        """
        Property with type of int. It contains the size of the allocated memory
        in bytes at the moment when the snapshot was taken. In fact this
        property is read-only since it can be set at once only, and it is set
        at instantiation. Due to the experimental character of this snapshot,
        trying to give a new value does not raise error however it won't be
        successful.
        -----------
        @Params: -> allocated: int
                    The value to set as allocated.
        """

        try:
            self.__allocated
        except AttributeError:
            self.__allocated = allocated



    @property
    def device(self) -> Union[torch.device, str, int, None]:
        """
        Property with type of torch.device, str, int or NoneType. It contains
        the device of the snapshot without converting to torch.device. In fact
        this property is read-only since it can be set at once only, and it is
        set at instantiation. Due to the experimental character of this
        snapshot, trying to give a new value does not raise error however it
        won't be successful.
        --------------------
        @Return: -> Union[torch.device, str, int, None]
                    The value of device.
        """

        return self.__device



    @device.setter
    def device(self, device: Union[torch.device, str, int, None]) -> None:
        """
        Property with type of torch.device, str, int or NoneType. It contains
        the device of the snapshot without converting to torch.device. In fact
        this property is read-only since it can be set at once only, and it is
        set at instantiation. Due to the experimental character of this
        snapshot, trying to give a new value does not raise error however it
        won't be successful.
        --------------------
        @Params: -> device: Union[torch.device, str, int, None]
                    The value to set as device.
        """

        try:
            self.__device
        except AttributeError:
            self.__device = device



    @property
    def free_cache(self) -> int:
        """
        Property with type of int. This is a calculated value of free cache size
        in bytes at the moment when the snapshot was taken. Since this property
        is calculated it is true read-only property.
        --------------------------------------------
        @Return: -> int
        """

        return self.cached - self.allocated



    @property
    def reserved(self) -> int:
        """
        Property with type of int. It contains the size of the reserved memory
        in bytes at the moment when the snapshot was taken. In fact this
        property is read-only since it can be set at once only, and it is set at
        instantiation. Due to the experimental character of this snapshot,
        trying to give a new value does not raise error however it won't be
        successful.
        -----------
        @Return: -> int
                    The value of reserved.
        """

        return self.__reserved



    @reserved.setter
    def reserved(self, reserved: int) -> None:
        """
        Property with type of int. It contains the size of the reserved memory
        in bytes at the moment when the snapshot was taken. In fact this
        property is read-only since it can be set at once only, and it is set at
        instantiation. Due to the experimental character of this snapshot,
        trying to give a new value does not raise error however it won't be
        successful.
        -----------
        @Params: -> reserved: int
                    The value to set as reserved.
        """

        try:
            self.__reserved
        except AttributeError:
            self.__reserved = reserved



    @property
    def total(self) -> int:
        """
        Property with type of int. It contains the size of the total available
        device memory in bytes at the moment when the snapshot was taken. In
        fact this property is read-only since it can be set at once only, and it
        is set at instantiation. Due to the experimental character of this
        snapshot, trying to give a new value does not raise error however it
        won't be successful.
        --------------------
        @Return: -> int
                    The value of total.
        """

        return self.__total



    @total.setter
    def total(self, total: int) -> None:
        """
        Property with type of int. It contains the size of the total available
        device memory in bytes at the moment when the snapshot was taken. In
        fact this property is read-only since it can be set at once only, and it
        is set at instantiation. Due to the experimental character of this
        snapshot, trying to give a new value does not raise error however it
        won't be successful.
        --------------------
        @Params: -> total: int
                    The value to set as total.
        """

        try:
            self.__total
        except AttributeError:
            self.__total = total



class CudaHistory(list):
    """
    Provides a simple container for CudaMemorySnapShot objects with labels.
    -----------------------------------------------------------------------
    Subclass of list.

    Entries of this object are tuples in the form of (label, CudaMemorySnapShot).

    However this is a subclass of list but it cannot be instantiated with
    addition of existing elements like lists can be.
    """



    def __init__(self) -> None:
        """
        Initializes the object
        ----------------------
        """

        super(self.__class__, self).__init__()



    def append(self, name: str,
               device_id: Union[torch.device, str, int, None] = None,
               synchronize: bool = True) -> None:
        """
        Appends items to the container. Every add-in leads to instantiation of
        a new CudaMemorySnapShot object.
        --------------------------------
        @Params: -> name: str
                    The label of the new CudaMemorySnapShot entry.
                 -> device_id: Union[torch.device, str, int, None] = None
                    The device where a new CudaMemorySnapShot should be taken.
                    If the device is not given it is set to None. The effect of
                    the setting of this parameter fully accords to the behavior
                    of the concerning PyTorch functions.
                 -> synchronize: bool = True
                    Whether or not to apply torch.cuda.synchronize() at the
                    given device before taking the snapshot.
        """

        if len(name) > 20:
            name = name[:20]
        super(self.__class__, self).append((name, CudaMemorySnapShot(device_id,
                                                                     synchronize)))



    def __str__(self) -> str:
        """
        Provides human readable representation of the CudaHistory object.
        -----------------------------------------------------------------
        @Return: -> str
                    The representation.
        """

        if len(self) > 0:
            output = 'CudaHistory:\n' + \
                     '| Name                 | Total       | Reserved    | Allocated   |\n' + \
                     '+----------------------+-------------+-------------+-------------+\n'
            for name, snapshot in self:
                output += '| {:20} | {:>11} | {:>11} | {:>11} |\n'.format(name,
                          snapshot.total, snapshot.reserved, snapshot.allocated)
            return output
        else:
            return 'CudaHistory is empty.'



def determinize(seed: Optional[int] = None, be_deterministic: bool = True) -> None:
    """
    Seeds more random sources and cares about the environment to be or don’t be
    deterministic.
    --------------
    @Params: -> seed: Optional[int] = None
                The number to use as seed for manual seeding. Setting this
                number to a constant value highly increases the chance of the
                reproducibility of a training. On the contrary leaving this
                number empty highly increases the randomness of each training
                individually.
             -> be_deterministic: bool = True
                A boolean that specifies whether to switch torch.backends.cudnn
                into deterministic mode or not.

    This function imports random from the standard library and tries to import
    NumPy however it won’t fail if NumPy is not installed.
    """

    import random

    if be_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if seed is None:
            seed = 0
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    if seed is None:
        if torch.cuda.is_available():
            torch.cuda.seed_all()
        torch.seed()
        try:
            import numpy as np
            np.random.seed()
        except ImportError:
            pass
        random.seed()
    else:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        random.seed(seed)



def is_equal_network(one: torch.nn.Module, another: torch.nn.Module) -> bool:
    """
    Compares two models based on their state_dicts and returns whether their
    weights and biases are exactly equal or not. The bases of the comparison are
    the same layers and same sizes as well.
    ---------------------------------------
    @Params: -> one: torch.nn.Module
                The one model to compare.
             -> another: torch.nn.Module
                The other model to compare.
    @Return: -> bool
                Whether the two networks are the same by weights and biases or not.
    """

    onedict = one.state_dict()
    anotherdict = another.state_dict()
    if len(set(onedict.keys()) ^ set(anotherdict.keys())) == 0:
        for key, data in onedict.items():
            onedata = data.detach()
            anotherdata = anotherdict[key].detach()
            if onedata.device != anotherdata.device:
                onedata = onedata.cpu()
                anotherdata = anotherdata.cpu()
            if not torch.equal(onedata, anotherdata):
                return False
        return True
    else:
        return False



def is_equal_with_error(one: torch.Tensor, another: torch.Tensor,
                        detach: bool=True) -> bool:
    """
    Compares two tensors if they have values in the given calculation error
    tolerance or not. The base of the comparison is the same dimensionality.
    ------------------------------------------------------------------------
    @Params: -> one: torch.Tensor
                One tensor to compare.
             -> another: torch.Tensor
                Another tensor to compare.
                detach: bool=True
                Whether to detach tensors before comparison or not. This
                settings effects on tensors with requires_grad = True only.
    @Return: -> bool
                Whether the two tensors are the same respectively to the
                calculation error tolerance or not.
    """

    if one.shape == another.shape:
        if len(one.shape) != 1:
            if (one.requires_grad or another.requires_grad) and detach:
                manage = lambda atensor: atensor.detach()
            else:
                manage = lambda atensor: atensor
            for i in range(one.shape[0]):
                if not is_equal_with_error(manage(one[i]), manage(another[i]),
                                                                  detach):
                    return False
            return True
        else:
            for i in range(len(one)):
                if compare_with_error_(float(one[i]), float(another[i])):
                    return False
            return True
    else:
        return False



def compare_with_error_(one: float, another: float) -> bool:
    """
    Compares two floats whether they are in the range of the calculation error
    tolerance or not.
    -----------------
    @Params: -> one: float
                One float to compare.
             -> another: float
                Another float to compare.
    @Return: -> bool
                The result of the comparison.
    """

    global ERROR_TOLERANCE

    return abs(one - another) <= ERROR_TOLERANCE
