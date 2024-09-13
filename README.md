# CS 5787 Deep Learning Assignment #1 - DL Basics

Authors: Mitchell Krieger

This repository contains materials to complete the [first assigment](./CS%205787%20-%20EX%201.pdf) for CS 5787 at Cornell Tech in the Fall 2024 semester. There are two parts to this assignment, the theoretical part and the practical part. 

## Theoretical

The first part is theoretical. All of its materials including a LaTex file and its PDF output can be found in the [report](./report/) directory.

## Practical

The second part of the assignment is practical. It is contained in the [assignment1_practical.ipynb](./assignment1_practical.ipynb) notebook. 

### Setup
All code, including data download, model definitions, training loops, hyperparameter tuning and evaluation should be runable from top to bottom of the notebook with reproduceable results by creating a virtual environment, installing the packages in requirements.txt, and logging into Weights & Biases.

```bash
source -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
wandb login
```

If you do not login to Weights & Biases via the command line the first cell of the notebook will attempt to login again. If you do not log in, you must set `mode` argument the `wandb.init()` method to `offline`. 

If cuda or mps is available to train models on GPUs, the notebook will attempt to use cuda, otherwise it will default to using cpu. If you wish to use mps, uncomment that section in `Check for GPU Access`.

### Data

For this assignment we are using the Fashion MNIST dataset. Running the first cell will download the data using `wget` to the `data` directory. After that the `FashionMNISTDataset` pytorch Dataset will handle loading the data into tensors and can retrieve data like any pytorch dataset can. In addition, a utility function `show_img` is provided to display any given sample given its index. 

```python
gen = torch.Generator().manual_seed(123)

train = FasionMNISTDataset(PATH, 'train', device=device)
train, val = torch.utils.data.random_split(train, [0.8, 0.2], generator=gen)
test = FasionMNISTDataset(PATH, 'test', device=device)
```

There are 3 dataloaders created from the dataset for training, validation and testing purposes. The validation set comes from a 80/20 split of the training set. Each data loader is randomly suffled with a generator that sets the seed to make the results reproduceable. Dataloader objects for train, validation and test are all contained in a dictionary called `dataloaders` for convience (and for use by training and evaluation functions).

```python
batch = 128
trainloader = DataLoader(train, batch, shuffle=True, generator=gen)
valloader = DataLoader(val, batch, shuffle=True, generator=gen)
testloader = DataLoader(test, batch, shuffle=True, generator=gen)

dataloaders = {
    'train': trainloader,
    'val': valloader,
    'test': testloader
}
```

### Models & Training

A basic `Lenet5` pytorch nn.module is created with similar architecture to the original 1998 implementation of LeNet5. Noteable differences are the use of ReLU and max pooling over sigmoid and average pooling respectively (see discussion in the theoretical section about model architecture choices). Any model can be trained and tuned by passing the module and dataloaders into the `hyperparameter_tuning` function which will return the model with the highest validation accuracy after performing a grid search training over all possible combinations of hyperparameters provided in a `param_grid` dictionary

```python
param_grid = {
  'learning_rate':[0.1, 0.01,0.001],
  'momentum':[0, 0.9, 0.7]
}

# pass the pytorch module, dataloaders, device in use, number of epochs 
# and the param grid into the hyperparameter_tuning function
best_lenet = hyperparameter_tuning(Lenet5, dataloaders, device, 25, **param_grid)
```

In the case above, `best_lenet` contains a dictionary with:
- `'net'`: the trained pytroch module with the highest validation accuracy
- `'name'`: a string containing the name of the wandb run logs of the model training
- `'accuracy'`: a dictionary of the train, validation and testing accuracies

Variants of Lenet5 are provided as pytorch child modules of Lenet5:
- `Lenet5`: Vanilla Lenet5 implementation
- `Lenet5BN`: Lenet5 with two batch normalization layers added after a convolution but before the ReLU activation function is applied. 
- `Lenet5Dropout`: Lenet5 with two dropout layers added in after the first and second fully connected layers. This pytorch module expects that at least one value for `dropout` to be in the param grid.
- `Lenet5Decay`: Lenet5 with weight decay (l2 norm). This module expects at least one value for `weight_decay` to be in the param grid and will apply it in the stochastic gradient decent optimizer. 

### Loading & saving

Trained weights have been saved to the `models` directory as `.pt` files using the simple `torch.save()` method. Weights can be loaded for prediction and testing by the following:

```python
model = Lenet5(**kwargs) #or any module from above
model.load_state_dict(torch.load(f'./models/{model_name}.pth', weights_only=True))
model.eval()
```

Models can be tested after loading using:
```
correct = 0
total = 0

for images, labels in testloader:
    outputs = model(images)
    total += labels.size(0)
    _, predicted = torch.max(outputs.data, 1)
    correct += (predicted == labels).sum().item()
print('correct:', correct)
print('total:', total)
print('accuracy:', correct/total)
print('sample label:', labels[0])
print('sample prediction:', predicted[0])

plt.imshow(images[0].numpy().reshape(28,28), cmap='gray');
```
