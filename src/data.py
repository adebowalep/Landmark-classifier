import math
import torch
import torch.utils.data
from pathlib import Path
from torchvision import datasets, transforms
import multiprocessing

from .helpers import compute_mean_and_std, get_data_location
import matplotlib.pyplot as plt


def get_data_loaders(
    batch_size: int = 32, valid_size: float = 0.2, num_workers: int = -1, limit: int = -1): #, rand_augment_magnitude: int = 10):
    """
    Create and returns the train_one_epoch, validation and test data loaders.

    :param batch_size: size of the mini-batches
    :param valid_size: fraction of the dataset to use for validation. For example 0.2
                       means that 20% of the dataset will be used for validation
    :param num_workers: number of workers to use in the data loaders. Use -1 to mean
                        "use all my cores"
    :param limit: maximum number of data points to consider
    :return a dictionary with 3 keys: 'train_one_epoch', 'valid' and 'test' containing respectively the
            train_one_epoch, validation and test data loaders
    """

    if num_workers == -1:
        # Use all cores
        num_workers = multiprocessing.cpu_count()

    # We will fill this up later
    data_loaders = {"train": None, "valid": None, "test": None}

    base_path = Path(get_data_location())

    # Compute mean and std of the dataset
    mean, std = compute_mean_and_std()
    print(f"Dataset mean: {mean}, std: {std}")

    # create 3 sets of data transforms: one for the training dataset,
    # containing data augmentation, one for the validation dataset
    # (without data augmentation) and one for the test set (again
    # without augmentation)
    # HINT: resize the image to 256 first, then crop them to 224, then add the
    # appropriate transforms for that step
    data_transforms = {
        "train" : transforms.Compose(
            [
            
            # resize the image to 256
            transforms.Resize(256),
            
            # croping the image to 224
            transforms.RandomCrop(224),
            
            # Horizontally flip 
            transforms.RandomHorizontalFlip(),
            
            transforms.RandomInvert(),
            transforms.RandomEqualize(),
            
            # RandAugment 
            transforms.RandAugment(num_ops=3,
            magnitude=5,interpolation=transforms.InterpolationMode.BILINEAR
                                 ),
            
            # Convert to Tensor
            transforms.ToTensor(),
            
            # Normalize 
            transforms.Normalize(mean, std),
          ]
                                   
        ),
       "valid": transforms.Compose(
           
           [

            
            # resize the image to 256
            transforms.Resize(256),
            
            transforms.CenterCrop(224),
                    
            # Convert to Tensor
            transforms.ToTensor(),
            
            # Normalize 
            transforms.Normalize(mean, std) 
           ]
            
              
        ),
        
       "test": transforms.Compose(
           [
            
            # resize the image to 256
            transforms.Resize(256),
               
            transforms.CenterCrop(224),
               
            # Convert to Tensor
            transforms.ToTensor(),
            
            # Normalize 
            transforms.Normalize(mean, std)
               
           ]
            
               
        )
    }
    
    # Create train and validation datasets
    train_data = datasets.ImageFolder(
        base_path / "train",
        transform = data_transforms["train"]
        # add the appropriate transform that you defined in
        # the data_transforms dictionary
    )
    # The validation dataset is a split from the train_one_epoch dataset, so we read
    # from the same folder, but we apply the transforms for validation
    valid_data = datasets.ImageFolder(
        base_path / "train", transform = data_transforms["valid"]
        # add the appropriate transform that you defined in
        # the data_transforms dictionary
        
    )

    # obtain training indices that will be used for validation
    n_tot = len(train_data)
    indices = torch.randperm(n_tot)

    # If requested, limit the number of data points to consider
    if limit > 0:
        indices = indices[:limit]
        n_tot = limit

    split = int(math.ceil(valid_size * n_tot))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler  = torch.utils.data.SubsetRandomSampler(valid_idx)

    # prepare data loaders
    data_loaders["train"] = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    data_loaders["valid"] = torch.utils.data.DataLoader(
        # 
        valid_data,
        batch_size = batch_size,
        sampler = valid_sampler,
        num_workers=num_workers
        
    )

    # Now create the test data loader
    test_data = datasets.ImageFolder(
        base_path / "test",
        #  (add the test transform)
        transform = data_transforms["test"]
    )

    if limit > 0:
        indices = torch.arange(limit)
        test_sampler = torch.utils.data.SubsetRandomSampler(indices)
    else:
        test_sampler = None

    data_loaders["test"] = torch.utils.data.DataLoader(
        test_data,
        batch_size = batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    return data_loaders


def visualize_one_batch(data_loaders, max_n: int = 5):
    """
    Visualize one batch of data.

    :param data_loaders: dictionary containing data loaders
    :param max_n: maximum number of images to show
    :return: None
    """

    # :
    # obtain one batch of training images
    # First obtain an iterator from the train dataloader
    dataiter  = iter(data_loaders["train"]) # 
    # Then call the .next() method on the iterator you just
    # obtained
    images, labels  = dataiter.next() # 

    # Undo the normalization (for visualization purposes)
    mean, std = compute_mean_and_std()
    invTrans = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
            transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
        ]
    )

    images = invTrans(images)

    # :
    # Get class names from the train data loader
    #base_path = Path(get_data_location())
    #class_names = datasets.ImageFolder(base_path / "train")
    class_names  = [
        "00.Haleakala_National_Park",
        "01.Mount_Rainier_National_Park",
        "02.Ljubljana_Castle","03.Dead_Sea",
        "04.Wroclaws_Dwarves",
        "05.London_Olympic_Stadium",
        "06.Niagara_Falls",
        "07.Stonehenge",
        "08.Grand_Canyon",
        "09.Golden_Gate_Bridge",
        "10.Edinburgh_Castle",
        "11.Mount_Rushmore_National_Memorial",
        "12.Kantanagar_Temple",
        "13.Yellowstone_National_Park",
        "14.Terminal_Tower",
        "15.Central_Park",
        "16.Eiffel_Tower",
        "17.Changdeokgung",
        "18.Delicate_Arch",
        "19.Vienna_City_Hall",
        "20.Matterhorn",
        "21.Taj_Mahal",
        "22.Moscow_Raceway",
        "23.Externsteine",
        "24.Soreq_Cave",
        "25.Banff_National_Park",
        "26.Pont_du_Gard",
        "27.Seattle_Japanese_Garden",
        "28.Sydney_Harbour_Bridge",
        "29.Petronas_Towers",
        "30.Brooklyn_Bridge",
        "31.Washington_Monument",
        "32.Hanging_Temple",
        "33.Sydney_Opera_House",
        "34.Great_Barrier_Reef",
        "35.Monumento_a_la_Revolucion",
        "36.Badlands_National_Park",
        "37.Atomium",
        "38.Forth_Bridge",
        "39.Gateway_of_India"
        "40.Stockholm_City_Hall",
        "41.Machu_Picchu",
        "42.Death_Valley_National_Park",
        "43.Gullfoss_Falls",
        "44.Trevi_Fountain",
        "45.Temple_of_Heaven",
        "46.Great_Wall_of_China",
        "47.Prague_Astronomical_Clock",
        "48.Whitby_Abbey",
        "49.Temple_of_Olympian_Zeus"   
    ]
        
    # Convert from BGR (the format used by pytorch) to
    # RGB (the format expected by matplotlib)
    images = torch.permute(images, (0, 2, 3, 1)).clip(0, 1)

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in range(max_n):
        ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        # print out the correct label for each image
        # .item() gets the value contained in a Tensor
        ax.set_title(class_names[labels[idx].item()])


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    return get_data_loaders(batch_size=2, num_workers=0)


def test_data_loaders_keys(data_loaders):

    assert set(data_loaders.keys()) == {"train", "valid", "test"}, "The keys of the data_loaders dictionary should be train, valid and test"


def test_data_loaders_output_type(data_loaders):
    # Test the data loaders
    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    assert isinstance(images, torch.Tensor), "images should be a Tensor"
    assert isinstance(labels, torch.Tensor), "labels should be a Tensor"
    assert images[0].shape[-1] == 224, "The tensors returned by your dataloaders should be 224x224. Did you " \
                                       "forget to resize and/or crop?"


def test_data_loaders_output_shape(data_loaders):
    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    assert len(images) == 2, f"Expected a batch of size 2, got size {len(images)}"
    assert (
        len(labels) == 2
    ), f"Expected a labels tensor of size 2, got size {len(labels)}"


def test_visualize_one_batch(data_loaders):

    visualize_one_batch(data_loaders, max_n=2)
