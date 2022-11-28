import torch
import torch.nn as nn
#######optuna#######
import optuna


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super().__init__()
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        
        self.model = nn.Sequential(
            # First conv + maxpool + relu (3x224x224)
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Second conv + maxpool + relu (16x112X112)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Third conv + maxpool + relu (32x56x56)
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
             
            # Fourth conv + maxpool + relu (64x28x28)
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            
            # Flatten feature maps (128x14x14)
            nn.Flatten(),
            
            # Fully connected layers. This assumes that the input image was 28x28
            nn.Linear(512*7*7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            
            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            
            nn.Linear(1024, num_classes)
    )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #  process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return self.model(x)

# define the CNN architecture
class MyModel2(nn.Module):
    def __init__(self, trial) -> None:
        super(MyModel2, self).__init__()
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        self.conv_layers = nn.Sequential(
            
            # First conv + maxpool + relu (3x224x224)
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            #nn.Dropout2d(0.2),
            
            # Second conv + maxpool + relu (16x112X112)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            #nn.Dropout2d(0.2),
            
            # Third conv + maxpool + relu (32x56x56)
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            #nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Fourth conv + maxpool + relu (64x28x28)
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),    
              
        )
            
        self.flat =nn.Flatten()  # -> 1x128x7x7
        
        fc2_input_dim = trial.suggest_int("fc2_input_dim",200, 1000,50)
        self.fc1 = nn.Linear(512*7*7, fc2_input_dim)  # -> 100
        self.relu = nn.ReLU()
        dropout_rate1 = trial.suggest_float("dropout_rate", 0, 0.5,step=0.1)
        self.drop1 = nn.Dropout(p=dropout_rate1)
        
        self.fc2=nn.Linear(fc2_input_dim, 50)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #  process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        x = self.conv_layers(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.fc2(x)     
        return x

######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
