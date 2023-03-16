from abc import ABC
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import random

import config


def get_one_hot_index(cols):
    """
    Generates a list of indexes (start, end) for one hot encoded columns based on prefix of column names
    :param cols: list of column names
    :return: list of (start, end) tuples
    """
    one_hot_index = []
    start = 0
    end = 0
    current = cols[0].split('_')[0]
    for i in cols:
        if i.split('_')[0] == current:
            end += 1
        else:
            one_hot_index.append((start, end))
            start = end
            end += 1
            current = i.split('_')[0]

    return one_hot_index


class TabularDataset(Dataset, ABC):
    """Tabular Dataset for contrastive learning using a VAE model"""

    def __init__(self, df=None, label_col='label'):
        """
        Initializes the dataset. If no dataframe is given, the data is loaded from the csv file
        and normalized. If a dataframe is given, it is assumed that the data is already normalized
        :param df:  dataframe containing the data
        :param label_col: name of the label column
        """
        self.label_col = label_col

        super(TabularDataset, self).__init__()
        if df is None:
            self.__get_data()
        else:
            self.data = df

        self.inputs = np.array(self.data.drop(columns=[self.label_col]))
        self.labels = np.array(self.data[self.label_col])

        if type(self.labels[0]) == str:
            self.__encode_labels()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input1 = torch.from_numpy(self.inputs[idx]).float()
        label = torch.from_numpy(np.array(self.labels[idx])).long()

        return input1, label

    def __get_data(self):
        """
        loads the data from the csv file and normalizes it
        :return: None
        """
        df = pd.read_csv(config.Paths().data_dir + 'raw/paldat_complete_clean.csv')

        # drop non numerical columns except for label
        labels = df[self.label_col]
        df = df._get_numeric_data()
        df[self.label_col] = labels

        # normalize columns
        for col in df.columns:
            if col != self.label_col:
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

        # drop columns with nan values
        df = df.dropna(axis=1)
        self.data = df

    def __encode_labels(self):
        """
        Encodes the labels to integers
        :return: None
        """
        le = LabelEncoder()
        self.labels = le.fit_transform(self.labels)
        self.label_names = le.classes_


class TripletDataset(Dataset, ABC):
    """Tabular Dataset generating triplets for contrastive learning using a VAE model"""

    def __init__(self, df=None, label_col='label', setting='train'):
        """
        Initializes the dataset. If no dataframe is given, the data is loaded from the csv file
        and normalized. If a dataframe is given, it is assumed that the data is already normalized
        :param df: dataframe containing data
        :param label_col: name of the column containing the labels
        :param setting: 'train' or 'test'
        """
        self.label_col = label_col

        super(TripletDataset, self).__init__()
        if df is None:
            self.__get_data(setting=setting)
        else:
            self.data = df

        self.labels = np.array(self.data[self.label_col])

        if self.label_col is "genus":
            self.inputs = np.array(self.data.drop(columns=["family", self.label_col]))
        elif self.label_col is "family":
            self.inputs = np.array(self.data.drop(columns=["genus", self.label_col]))
        else:
            self.inputs = np.array(self.data.drop(columns=[self.label_col]))

        if type(self.labels[0]) == str:
            self.__encode_labels()
        else:
            self.label_names = self.labels

        #self.one_hot_index = get_one_hot_index(self.data.columns)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a triplet of data points and their labels
        :param idx: index of anchor point
        :return: anchor, postive, negative, labels
        """
        anchor_idx, postive_idx, negative_idx = self.__get_triplet(idx)

        anchor = torch.from_numpy(self.inputs[anchor_idx]).float()
        postive = torch.from_numpy(self.inputs[postive_idx]).float()
        negative = torch.from_numpy(self.inputs[negative_idx]).float()

        labels = {'anchor': torch.from_numpy(np.array(self.labels[anchor_idx])).long(),
                  'positive': torch.from_numpy(np.array(self.labels[postive_idx])).long(),
                  'negative': torch.from_numpy(np.array(self.labels[negative_idx])).long()}

        return anchor, postive, negative, labels

    def __get_data(self, setting='train'):
        """
        Loads data from csv file and normalizes it. According to setting, only train or test data is loaded
        :param setting: 'train' or 'test'
        :return: None
        """
        df = pd.read_csv(config.Paths().data_dir + 'raw/paldat_complete_clean.csv')

        if setting == 'train':
            df = df[df['setting'] == 'train'].reset_index(drop=True)
        elif setting == 'test':
            df = df[df['setting'] == 'test'].reset_index(drop=True)

        df.drop(columns=['setting'], inplace=True)

        # drop non numerical columns except for label
        labels = df[self.label_col]
        df = df._get_numeric_data()  # naughty
        df[self.label_col] = labels

        # normalize columns if columns have values > 0
        # for col in df.columns:
        #     if col != self.label_col and df[col].max() > 0:
        #         df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

        # drop columns with nan values
        df = df.dropna(axis=1)
        self.data = df

        print(self.data.shape)

    def __encode_labels(self):
        """
        Encodes labels to integers. Saves label names in self.label_names
        :return: None
        """
        le = LabelEncoder()
        self.labels = le.fit_transform(self.labels)
        self.label_names = le.classes_

    def __get_triplet(self, idx):
        """
        generates a triplet based on target column self.label_col
        :return: triplet
        """
        # get gLabel
        label = self.data.iloc[idx][self.label_col]

        # get anchor and positive from label
        anchor_idx = idx
        positive_idx = np.random.choice(self.data[self.data[self.label_col] == label].index)

        # get negative from different label
        negative_label = random.choice(self.label_names)
        while negative_label == label:
            negative_label = random.choice(self.label_names)

        negative_idx = np.random.choice(self.data[self.data[self.label_col] == negative_label].index)

        return anchor_idx, positive_idx, negative_idx
