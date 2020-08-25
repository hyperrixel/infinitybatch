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
from warnings import warn



class InfinityBatch(object):
    """
    Provides the whole functionality of infinitybatch
    -------------------------------------------------
    """



    def __init__(self, model: Optional[torch.nn.Module] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 criterion: Optional[torch.nn.Module] = None,
                 dataloader: Optional[Any] = None,
                 backward_device: Union[torch.device, str, int, None] = None,
                 verbose: bool = True) -> None:
        """
        Initializes the object
        ----------------------
        @Params: -> model: Optional[torch.nn.Module] = None
                    PyTorch model that is used for training. The given model is
                    stored as InfinityBatch.model property. This property can be
                    set to other than None at once only. The only requirement
                    against a model is that it should accept the input in
                    batch_first form. At best it can handle different batch
                    sizes as well
                 -> optimizer: Optional[torch.optim.Optimizer] = None
                    PyTorch optimizer that is instantiated to the given model.
                    The given optimizer is stored as InfinityBatch.optimizer
                    property. This property can be set to other than None at
                    once only.
                 -> criterion: Optional[torch.nn.Module] = None
                    PyTorch loss function that is used for training. The given
                    criterion is stored as InfinityBatch.criterion property.
                    This property can be set to other than None at once only.
                 -> dataloader: Optional[Any] = None
                    PyTorch alike data loader. The given dataloader is stored as
                    InfinityBatch.dataloader property. Because PyTorch does not
                    demand too much needs against dataloaders InfinityBatch
                    follows the same way too. However InfinityBatch has three
                    essential requirements. The output of the iteration of
                    dataloaders should be a pair of inputs (x) and targets (y)
                    with the same batch size. The first dimension of the output
                    should be the dimension of the batch. The number of further
                    dimensions does not matter. The batch level output of the
                    dataloader, both inputs and targets, should be type of
                    torch.Tensor. They should fit the needs of the model and the
                    criterion respectively.
                 -> backward_device: Union[torch.device, str, int, None] = None
                    Optional parameter to set the device of the backward
                    processes. If the parameter is omitted or set to None,
                    InfinityBatch sets the device to “cpu”. The setting of
                    InfinityBatch.backward_device calls a check about the
                    equality of forward and backward devices which can lead to
                    InfinityBatchWarning about having the same device also as
                    forward and backward. This warning does not break the run of
                    the code and it can be managed with the change of the
                    InfinityBatch.backward_device or by calling
                    InfinityBatch.updateforwarddevice() if the model is moved to
                    another device yet. Moving the model to another device with
                    InfinityBatch.changemodeldevice() solves this issue as well.
                 -> verbose: bool = True
                    Whether to print progress of training or not. By default it
                    is set to True.
        @Warns : -> InfinityBatchWarning
                    Ff torch.cuda is not available at the the instantiation of
                    an InfinityBatch object since InfinityBatch is useless
                    without CUDA device.
        """

        if not torch.cuda.is_available():
            warn('InfinityBatch.init CUDA is not available, use of InfinityBatch is meaningless.',
                 InfinityBatchWarning, stacklevel=2)
        self.__forward_device = None
        self.backward_device = backward_device
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.verbose_mode = verbose
        self.clearcontainers_()



    def backward(self) -> None:
        """
        Handles loss.backward() at batch level in a training loop. In case of a
        totally DIY training loop it can be called from outside to ensure the
        highest flexibility of InfinityBatch.
        -------------------------------------
        @Warns : -> InfinityBatchWarning
                    If trying to backward on non existing loss.
        """

        try:
            self.__loss.backward()
        except AttributeError:
            warn('InfinityBatch.backward() tried to backward on non existing loss.',
                 InfinityBatchWarning, staclevel=2,)



    @property
    def backward_device(self) -> torch.device:
        """
        Property with type of torch.device, that stores the identifier of the
        backward device. InfinityBatch sets the device to “cpu” by default at
        instantiation. The setting of InfinityBatch.backward_device calls a
        check about the equality of forward and backward devices which can lead
        to InfinityBatchWarning about having the same device also as forward and
        backward. This warning does not break the run of the code and it can be
        managed with the change of the InfinityBatch.backward_device or by
        calling InfinityBatch.updateforwarddevice() if the model is moved to
        another device yet. Moving the model to another device with
        InfinityBatch.changemodeldevice() solves this issue as well.
        ------------------------------------------------------------
        @Return: -> torch.device
                    The value of backward_device.
        """

        return self.__backward_device



    @backward_device.setter
    def backward_device(self, device: Union[torch.device, str, int, None]) -> None:
        """
        Property with type of torch.device, that stores the identifier of the
        backward device. InfinityBatch sets the device to “cpu” by default at
        instantiation. The setting of InfinityBatch.backward_device calls a
        check about the equality of forward and backward devices which can lead
        to InfinityBatchWarning about having the same device also as forward and
        backward. This warning does not break the run of the code and it can be
        managed with the change of the InfinityBatch.backward_device or by
        calling InfinityBatch.updateforwarddevice() if the model is moved to
        another device yet. Moving the model to another device with
        InfinityBatch.changemodeldevice() solves this issue as well.
        ------------------------------------------------------------
        @Params: -> device: Union[torch.device, str, int, None]
                    The value to set as backward_device. It will be converted
                    to torch.device.
        """

        if device is None:
            self.__backward_device = torch.device('cpu')
        else:
            self.__backward_device = torch.device(device)
        self.checkdevices_()



    def changemodeldevice(self, device: Union[torch.device, str, int]) -> None:
        """
        Moves the model to the given device. The call of
        InfinityBatch.changemodeldevice() calls a check about the equality of
        forward and backward devices which can lead to InfinityBatchWarning
        about having the same device also as forward and backward. This warning
        does not break the run of the code and it can be managed with the change
        of the InfinityBatch.backward_device or by calling
        InfinityBatch.updateforwarddevice() if the model is moved to another
        device yet. Moving the model to another device with
        InfinityBatch.changemodeldevice() solves this issue as well.
        ------------------------------------------------------------
        @Params: -> device: Union[torch.device, str, int]
                    Device to move the model to.
        """

        if self.__model is not None:
            self.__model.to(device)
            self.updateforwarddevice()



    @property
    def criterion(self) -> Union[torch.nn.Module, None]:
        """
        Property with type of torch.nn.Module or NoneType. It contains a PyTorch
        loss function that is used for training. This property can be set to
        other than None at once only.
        -----------------------------
        @Return: -> Union[torch.nn.Module, None]
                    The value of criterion.
        """

        return self.__criterion



    @criterion.setter
    def criterion(self, criterion: Union[torch.nn.Module, None]) -> None:
        """
        Property with type of torch.nn.Module or NoneType. It contains a PyTorch
        loss function that is used for training. This property can be set to
        other than None at once only.
        -----------------------------
        @Params: -> criterion: Union[torch.nn.Module, None]
                    The value to set as criterion.
        @Warns : -> InfinityBatchWarning
                    If criterion is already set to different value than None.
        """

        try:
            if self.__criterion is not None:
                warn('InfinityBatch.criterion cannot change criterion if given at once.',
                     InfinityBatchWarning, stacklevel=2)
                return
        except AttributeError:
            pass
        self.__criterion = criterion



    @property
    def dataloader(self) -> Union[Any, None]:
        """
        Property with type of anything that fits the requirements of being used
        as dataloader. Because PyTorch does not demand too much needs against
        dataloaders InfinityBatch follows the same way too. However
        InfinityBatch has three essential requirements. The output of the
        iteration of dataloaders should be a pair of inputs (x) and targets (y)
        with the same batch size. The first dimension of the output should be
        the dimension of the batch. The number of further dimensions does not
        matter. The batch level output of the dataloader, both inputs and
        targets, should be type of torch.Tensor. They should fit the needs of
        the model and the criterion respectively.
        -----------------------------------------
        @Return: -> Union[Any, None]
                    The value of dataloader.
        """

        return self.__dataloader



    @dataloader.setter
    def dataloader(self, dataloader: Union[Any, None]) -> None:
        """
        Property with type of anything that fits the requirements of being used
        as dataloader. Because PyTorch does not demand too much needs against
        dataloaders InfinityBatch follows the same way too. However
        InfinityBatch has three essential requirements. The output of the
        iteration of dataloaders should be a pair of inputs (x) and targets (y)
        with the same batch size. The first dimension of the output should be
        the dimension of the batch. The number of further dimensions does not
        matter. The batch level output of the dataloader, both inputs and
        targets, should be type of torch.Tensor. They should fit the needs of
        the model and the criterion respectively.
        -----------------------------------------
        @Params: -> dataloader: Union[Any, None]
                    The value to set as dataloader.
        """

        self.__dataloader = dataloader



    def epoch(self) -> None:
        """
        Manages an epoch. It can be called from outside to train a solo epoch
        only.
        """

        self.updateforwarddevice()
        self.epochstart_()
        _count = 0
        _round = 0
        for _inputs, _targets in self.dataloader:
            _len_inputs = len(_inputs)
            self.__print('-- batch {}: finished with {}, current batch size {}.'
                         .format(_round, _count, _len_inputs), end='\r')
            _count += _len_inputs
            _round += 1
            _outputs = self.batch_(_inputs)
            _targets = _targets.to(self.backward_device)
            self.__inputs[-1].append(_inputs)
            self.__outputs[-1].append(_outputs)
            self.__targets[-1].append(_targets)
            self.loss()
            self.backward()
            self.step()
            self.zero_grad()
        self.__print('-- Epoch finished with {} batch(es), {} elemnt(s) were forwarded, last batch size was {}.'
                     .format(_round, _count, _len_inputs))



    @property
    def forward_device(self) -> torch.device:
        """
        Read-only property with the type of torch.device that stores the
        identifier of the forward device. This property is set by changing the
        InfinityBatch.model property or calling
        InfinityBatch.changemodeldevice() or InfinityBatch.updateforwarddevice()
        functions.
        """

        return self.__forward_device



    @property
    def inputs(self) -> list:
        """
        Container-like property that holds the inputs of the training. This
        container has the shape: [epoch] [batch] [batch element] [data [...] ].
        This is a read-only property. It cannot be cleared alone. All the
        containers can be cleared with the inner function
        InfinityBatch.clearcontainers_() only.
        """

        return self.__inputs



    def loss(self) -> None:
        """
        Handles the calculation of the loss in the training loop and saves
        batch-level loss values to its container. In case of a totally DIY
        training loop it can be called from outside to ensure the highest
        flexibility of InfinityBatch.
        """

        self.__loss = self.criterion(self.__outputs[-1][-1],
                                     self.__targets[-1][-1])
        self.__losses[-1].append((self.__loss.item(),
                                  len(self.__outputs[-1][-1])))



    @property
    def losses(self) -> list:
        """
        Container-like property that holds the loss.item() values of the
        training at batch level. This container has the shape:
        [epoch] [batch] (loss.item(), size_of_the_batch)]. This is a read-only
        property. It cannot be cleared alone. All the containers can be cleared
        with the inner function InfinityBatch.clearcontainers_() only.
        """

        return self.__losses



    @property
    def model(self) -> Union[torch.nn.Module, None]:
        """
        Property with type of torch.nn.Module or NoneType. It contains a PyTorch
        model that is used for training. This property can be set to other than
        None at once only. The only requirement against a model is that it
        should accept the input in batch_first form. At best it can handle
        different batch sizes as well.
        ------------------------------
        @Return: -> Union[torch.nn.Module, None]
                    The value of model.
        """

        return self.__model



    @model.setter
    def model(self, model: Union[torch.nn.Module, None]) -> None:
        """
        Property with type of torch.nn.Module or NoneType. It contains a PyTorch
        model that is used for training. This property can be set to other than
        None at once only. The only requirement against a model is that it
        should accept the input in batch_first form. At best it can handle
        different batch sizes as well.
        ------------------------------
        @Params: -> model: Union[torch.nn.Module, None]
                    The value to set as model.
        @Warns : -> InfinityBatchWarning
                    If model is already set to different value than None.
        """

        try:
            if self.__model is not None:
                warn('InfinityBatch.model cannot change model if given at once.',
                     InfinityBatchWarning, stacklevel=2)
                return
        except AttributeError:
            pass
        self.__model = model
        self.updateforwarddevice()
        self.checkdevices_()



    @property
    def optimizer(self) -> Union[torch.optim.Optimizer, None]:
        """
        Property with type of torch.optim.Optimizer or NoneType. It contains a
        PyTorch optimizer that is used for training. This property can be set to
        other than None at once only.
        -----------------------------
        @Return: -> Union[torch.optim.Optimizer, None]
                    The value of optimizer.
        """

        return self.__optimizer



    @optimizer.setter
    def optimizer(self, optimizer: Union[torch.optim.Optimizer, None]) -> None:
        """
        Property with type of torch.optim.Optimizer or NoneType. It contains a
        PyTorch optimizer that is used for training. This property can be set to
        other than None at once only.
        -----------------------------
        @Params: -> optimizer: Union[torch.optim.Optimizer, None]
                    The value to set as optimizer.
        @Warns : -> InfinityBatchWarning
                    If optimizer is already set to different value than None.
        """

        try:
            if self.__optimizer is not None:
                warn('InfinityBatch.optimizer cannot change optimizer if given at once.',
                     InfinityBatchWarning, stacklevel=2)
                return
        except AttributeError:
            pass
        self.__optimizer = optimizer



    @property
    def outputs(self) -> list:
        """
        Container-like property that holds the model outputs of the training.
        This container has the shape:
        [epoch] [batch] [batch element] [data [...] ]. This is a read-only
        property. It cannot be cleared alone. All the containers can be cleared
        with the inner function InfinityBatch.clearcontainers_() only.
        """

        return self.__outputs



    def step(self) -> None:
        """
        Handles the step of the optimizer in the training loop. In case of a
        totally DIY training loop it can be called from outside to ensure the
        highest flexibility of InfinityBatch.
        """

        self.optimizer.step()



    @property
    def targets(self) -> list:
        """
        Container-like property that holds the targets of the training. This
        container has the shape: [epoch] [batch] [batch element] [data [...] ].
        This is a read-only property. It cannot be cleared alone. All the
        containers can be cleared with the inner function
        InfinityBatch.clearcontainers_() only.
        """

        return self.__targets



    def train(self, num_epochs: int) -> None:
        """
        Handles a training. This is the default use-case of InfinityBatch.
        ------------------------------------------------------------------
        @Params: -> num_epochs: int
                    The number of epochs to train for.
        """

        self.clearcontainers_()
        self.__print('Training for {} epochs.'.format(num_epochs))
        for _epoch in range(num_epochs):
            self.__print('Epoch: {}/{}'.format(_epoch + 1, num_epochs))
            self.epoch()



    def updateforwarddevice(self) -> None:
        """
        Updates forward device from the model parameters. It is called
        automatically when a model is added to the InfinityBatch object or when
        InfinityBatch.changemodeldevice() is called. Call of this function is
        reasonable when something occurred with the device location of the model
        outside of the scope of InfinityBatch.
        --------------------------------------
        @Warns : -> InfinityBatchWarning
                    If the model doesn't have any learnable parameter.
        """

        if self.__model is None:
            self.__forward_device = None
        else:
            _is_learnable = False
            for param in self.__model.parameters():
                if param.requires_grad:
                    self.__forward_device = param.device
                    _is_learnable = True
                    break
            if not _is_learnable:
                self.__forward_device = None
                warn('InfinityBatch.model doesn\'t have learnable parameter.',
                     InfinityBatchWarning, stacklevel=2)



    @property
    def verbose_mode(self) -> bool:
        """
        Boolean property that stores the verbosity state of the InfinityBatch
        object.
        -------
        @Return: -> bool
                    The value of verbose_mode
        """

        return self.__verbose_mode



    @verbose_mode.setter
    def verbose_mode(self, mode: bool) -> None:
        """
        Boolean property that stores the verbosity state of the InfinityBatch
        object.
        -------
        @Params: -> mode: bool
                    The value to set as verbose_mode
        """

        self.__verbose_mode = mode
        if mode:
            self.__print = lambda *args, **kvargs: print(*args, **kvargs)
        else:
            self.__print = lambda *args, **kvargs: None



    def zero_grad(self) -> None:
        """
        Handles the zero_grad of the optimizer and of the model respectively in
        the training loop. In case of a totally DIY training loop it can be
        called from outside to ensure the highest flexibility of InfinityBatch.
        """

        self.optimizer.zero_grad()



    def batch_(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Handles the forwarding of a batch. This is an inner function. Call it
        from outside only if you make sure you need to call this function anyway
        and you fully understand the effect of the call of this function.
        -----------------------------------------------------------------
        @Params: -> inputs: torch.Tensor
                    Inputs for the model to forward.
        @Return: -> torch.Tensor
                    Output of the model w.r.t. the given batch.
        """

        _output_list = []
        for _input in inputs:
            _input = _input.unsqueeze(0).to(self.forward_device)
            _output = self.model(_input)
            del _input
            _output_list.append(_output.to(self.backward_device).clone()[0])
            del _output
            torch.cuda.empty_cache()
        return torch.stack(_output_list, dim=0)



    def checkdevices_(self) -> None:
        """
        Checks the forward and backward devices and warns if they are the same.
        This is an inner function. Call it from outside only if you make sure
        you need to call this function anyway and you fully understand the
        effect of the call of this function.
        """

        if self.backward_device == self.forward_device:
            warn('InfinityBatch.checkdevices_ forward and backward devices are the same.',
                 InfinityBatchWarning, stacklevel=3)



    def clearcontainers_(self) -> None:
        """
        Clears each container. This is an inner function. Call it from outside
        only if you make sure you need to call this function anyway and you
        fully understand the effect of the call of this function.
        """

        self.__inputs = []
        self.__losses = []
        self.__outputs = []
        self.__targets = []



    def epochstart_(self) -> None:
        """
        Adds new entries to each container to be used in the epoch that begins.
        This is an inner function. Call it from outside only if you make sure
        you need to call this function anyway and you fully understand the
        effect of the call of this function.
        """

        self.__inputs.append([])
        self.__losses.append([])
        self.__outputs.append([])
        self.__targets.append([])



    def __call__(self, num_epochs: int) -> None:
        """
        Provides direct instance level call. This is an alias of
        InfinityBatch.train() function. PyTorch models have the ability to be
        called at instance level for forwarding; this function adds a similar
        feature to InfinityBatch. InfinityBatch’s main focus is to provide a
        whole training manager and this function provides it.
        -----------------------------------------------------
        @Params: -> num_epochs: int
                    The number of epochs to train for.
        """

        self.train(num_epochs)



class InfinityBatchError(Exception):
    """
    Used when an error occurs.
    --------------------------
    It is an empty subclass of Exception.

    At the moment it is not used yet.
    """

    pass



class InfinityBatchWarning(Warning):
    """
    Used to warn the user about important things.
    ---------------------------------------------
    It is an empty subclass of Warning.
    """

    pass
