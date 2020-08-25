[![Generic badge](https://img.shields.io/badge/Version-v_1.0-001850.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/State-dev_released-ffa000.svg)](https://shields.io/)<br>
# infinitybatch

## What is this?

Infinitybatch is an open source solution for PyTorch that helps deep learning developers to train with bigger batch size than it could be loaded into GPU RAM.

The core concept of the idea comes from the fact that GPU time is expensive and the usage of own GPU cluster or a cloud based GPU service has to be optimized to be cost efficient. Furthermore, developers and researchers regularly have limited access to GPU. However, CPU based training mostly allows higher batches than a normal GPU could provide, it is much slower. Infinitybatch helps to use GPU during training with bigger batch size thanks to the special unloading and uploading process that manages the GPU RAM to avoid memory overrun.

[![infinitybatch](https://github.com/hyperrixel/infinitybatch/blob/master/assets/video.png "infinitybatch")](https://youtu.be/9pl63pW2OMI)

## How it works?

There are two main use cases. One is a Keras-like training manager where just the basic parameters of the training have to be specified. infinitybatch manages everything else. Another use case is to use the forward pass and the backpropagation with infinitybatch and surround it with custom code. This gives the freedom to arrange the rest of the training loop according to user’s needs.

### Workflow

![infinitybatch - workflow](https://github.com/hyperrixel/infinitybatch/blob/master/assets/workflow.png "workflow")

### Dataflow

![infinitybatch - dataflow](https://github.com/hyperrixel/infinitybatch/blob/master/assets/dataflow.png "dataflow")

## How to use?

The examples below assume you alreade have a ` model ` and a ` dataloader `.

### Simple use case

``` python

import infinitybatch
import torch

# Making device agnostic codes in deep learning is essential.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ib = infinitybatch.InfinityBatch()
ib.model = MyModel().to(device)
ib.criterion = torch.nn.CrossEntropyLoss()
ib.optimier = torch.optim.Adam(ib.model.parameters())
ib.dataloader = mydataloader

ib(200) # Making a 200 epoch long training is rarely that easy though.

```

### Epoch use case

``` python

import infinitybatch
import torch

EPOCHS = 200

# Making device agnostic codes in deep learning is essential.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ib = infinitybatch.InfinityBatch()
ib.model = MyModel().to(device)
ib.criterion = torch.nn.CrossEntropyLoss()
ib.optimier = torch.optim.Adam(ib.model.parameters())
ib.dataloader = mydataloader

for epoch in range(EPOCHS):
  # Additional code can be added here...
  ib.epoch()
  # And also here...

```

### Custom use case

``` python

import infinitybatch
import torch

EPOCHS = 200

# Making device agnostic codes in deep learning is essential.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ib = infinitybatch.InfinityBatch()
ib.model = MyModel().to(device)
ib.criterion = torch.nn.CrossEntropyLoss()
ib.optimier = torch.optim.Adam(ib.model.parameters())
ib.dataloader = mydataloader

for epoch in range(EPOCHS):
  # Additional code can be added here...
  for inputs, targets in ib.dataloader:
    # And here...
    outputs = ib.batch_()
    # And here...
    loss = ib.criterion(outputs, targets)
    # And here...
    loss.backward()
    # And here...
    ib.step()
    # And here...
    ib.zero_grad()
    # And here...
  # And also here.

```

### Real wolrd example

The file ` mnist_test_example.py ` contains a common Pytorch MNIST classification example transformed to train with infinitybatch.

## Bonus result

We made a map of Tesla Index. If you are curious about the Tesla Index itself, please read the concerning chapter of the book of infinitybatch.

[![Map of Tesla Index](https://github.com/hyperrixel/infinitybatch/blob/master/assets/teslaindex.jpg "Map of Tesla Index")](https://hyperrixel.com/teslaindex/)

## Future plans

We have a detailed future plan for this project.

### eval()

However making evaluation with infinitybatch is very easy for user who like to have “canonical way” for everything we plan to implement .eval() function separately. The hardest task of this implementation is to relate to dataloaders because of the freedom use that we can provide at this point.

### PyPi

Since infinitybatch is a production ready module it could be right now loaded into PyPi.org to be usable as a Python package globally. We plan to wait till the outcome of this hackathon gets clear and then we will upload this package together with a ReadTheDocs documentation and with a lot of examples and tests as well.

### Event monitor

To support do-it-yourself train loops even more we plan add an event monitor to infinitybatch. This feature could warn the user if they leaves out something important from the forward-backward process.

### Improvement of warnings

A good and highly configurable warning system is needed for infinitybatch in the future to be able to serve experimenters and real experts at the same time. We plan to add warning plans and warning levels and more safety functions.

### Improvement of the containers

The containers of infinitybatch are very simple constructions at the moment. In the future we plan to add sophisticated storage strategies and container-related functions too.

### Counter of learnable parameters

We plan to add sophisticated counter function to make a summary about the characteristics of learnable parameters of a model. We plan to show a detailed overview across the whole model including the number of elements and their aspects to memory and speed as well. This feature is far away from a level where pure Python could be enough.
Memory usage monitor

To be able to have a better overview of what happens in the memory we have to dig much deeper therefore we plan to make a detailed memory usage monitor. To develop this we have to leave the level of Python but we plan to build something that is usable in the pure Python level as well.

### Improving memory usage

If we have a better overview of what happens in the memory why not to improve the usage of the memory. At least the precise and up-to-date calculation of active and inactive memory blocks would be very welcome and the lifecycle of temporary variables could be also examined.

### Improving UI

At the moment our user interface is far away from perfect but at least we have a verbose property to decide about the appearance of very simple prints during the training. In the future we plan to have more configurable training prints. We plan to offer saving stats into file and we plan to create some functionality to be able to connect with graphical interfaces.

### Change of model

We plan to add observer to the model, optimizer and criterion properties of infinitybatch to be able to monitor important changes of these attributes without the need to implement a wrapper of the PyTorch method. This ways the use of infinitybatch would be much more convenient.

### “In case if needed” plans

The improvements below are a question of need. Because of different reasons we don’t consider those ones that much important right now but we are open and curious about the real-world needs of the users.

#### Callbacks

Based on the first experiences we plan to add the ability to place callbacks in the worklfow of infinitybatch. We think the ability of calling forward-backward stages separately should be enough to build very different train loops but there could be draw up some real needs.

#### Improvement of tools.CudaHistory

Classes tools.CudaHistory and tools.CudaMemorySnapshot began their lives like experimental classes. The importance of the management of snapshots like memory state at a given moment or anything else is obvious therefore we plan improve history based containers.

#### Complexity

We plan to add the ability of using more models, criterions or optimizers together. This leads to much more complex epochs and functionality. Though it would be nice to have that level of functionality yet tomorrow it seems it’s not that easy. To give all the needed energy for this improvement we need to have a better view to the real-world use-cases of our users.

## Additional information

We are using ` requirements.txt ` and ` CHANGELOG.md `
