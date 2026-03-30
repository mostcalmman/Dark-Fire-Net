import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import ConcatDataset


data_sources = ('esim', 'ijrr', 'mvsec', 'eccd', 'hqfd', 'unknown')
# Usage: name = data_sources[1], idx = data_sources.index('ijrr')


def concatenate_subfolders(data_file, dataset, dataset_kwargs):
    """
    Create an instance of ConcatDataset by aggregating all the datasets in a given folder
    """
    if os.path.isdir(data_file):
        subfolders = [os.path.join(data_file, s) for s in os.listdir(data_file)]
    elif os.path.isfile(data_file):
        subfolders = pd.read_csv(data_file, header=None).values.flatten().tolist()
    else:
        raise Exception('{} must be data_file.txt or base/folder'.format(data_file))
    print('Found {} samples in {}'.format(len(subfolders), data_file))
    datasets = []
    for subfolder in subfolders:
        dataset_kwargs['item_kwargs'].update({'base_folder': subfolder})
        datasets.append(dataset(**dataset_kwargs))
    return ConcatDataset(datasets)


def concatenate_datasets(data_file, dataset_type, dataset_kwargs={}):
    """
    Generates a dataset for each data_path specified in data_file and concatenates the datasets.
    :param data_file: A file containing a list of paths to CTI h5 files.
                      Each file is expected to have a sequence of frame_{:09d}
                      Can also have two columns: h5_path,flow_path (optional)
    :param dataset_type: Pointer to dataset class
    :param sequence_length: Desired length of each sequence
    :return ConcatDataset: concatenated dataset of all data_paths in data_file
    """
    df = pd.read_csv(data_file, header=None)
    
    # Check if file has two columns (h5_path, flow_path)
    if df.shape[1] >= 2:
        # Two column format: h5_path,flow_path
        data_paths = df.iloc[:, 0].values.tolist()
        flow_paths = df.iloc[:, 1].values.tolist()
    else:
        # Single column format: just h5 paths
        data_paths = df.values.flatten().tolist()
        flow_paths = [None] * len(data_paths)
    
    # Check if dataset_type is a wrapper class like SequenceDataset
    # that requires nested dataset_kwargs
    dataset_type_name = dataset_type.__name__ if hasattr(dataset_type, '__name__') else str(dataset_type)
    is_wrapper_dataset = 'SequenceDataset' in dataset_type_name
    
    dataset_list = []
    print('Concatenating {} datasets'.format(dataset_type))
    for data_path, flow_path in tqdm(zip(data_paths, flow_paths), total=len(data_paths)):
        kwargs = dataset_kwargs.copy()
        if flow_path is not None and str(flow_path) != 'nan':
            if is_wrapper_dataset:
                # For wrapper datasets like SequenceDataset, flow file should be in dataset_kwargs
                # which gets passed to the underlying dataset
                if 'dataset_kwargs' not in kwargs:
                    kwargs['dataset_kwargs'] = {}
                kwargs['dataset_kwargs']['external_flow_file'] = flow_path
            else:
                # For direct datasets like DynamicH5Dataset, pass flow file directly
                kwargs['external_flow_file'] = flow_path
        dataset_list.append(dataset_type(data_path, **kwargs))
    return ConcatDataset(dataset_list)

def concatenate_memmap_datasets(data_file, dataset_type, dataset_kwargs):
    """
    Generates a dataset for each memmap_path specified in data_file and concatenates the datasets.
    :param data_file: A file containing a list of paths to memmap root dirs.
    :param dataset_type: Pointer to dataset class
    :param dataset_kwargs: Dataset keyword arguments
    :return ConcatDataset: concatenated dataset of all memmap_paths in data_file
    """
    if dataset_kwargs is None:
        dataset_kwargs = {}

    memmap_paths = pd.read_csv(data_file, header=None).values.flatten().tolist()
    dataset_list = []
    print('Concatenating {} datasets'.format(dataset_type))
    for memmap_path in tqdm(memmap_paths):
        dataset_kwargs['dataset_kwargs'].update({'root': memmap_path})
        dataset_list.append(dataset_type(**dataset_kwargs))
    return ConcatDataset(dataset_list)
