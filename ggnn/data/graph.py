from torch import Tensor
from abc import ABCMeta, abstractmethod


class BaseGraph(metaclass=ABCMeta):

    @property
    @abstractmethod
    def node_features(self):
        """
        Returns a list of node_feature.
        """

    @property
    @abstractmethod
    def edge_features(self):
        """
        Returns a list of edge_feature.
        """

    @property
    @abstractmethod
    def neighbor_indices(self):
        """
        Returns a list of neighbor_indices.
        """

    def __len__(self):
        return len(self.node_features)


class GlassGraph(BaseGraph):
    def __init__(self, node_features, edge_features, neighbor_indices,
                 targets=None, device=None):
        self._node_features = node_features
        self._edge_features = edge_features
        self._neighbor_indices = neighbor_indices
        self._targets = targets
        self._device = device

    def to(self, device):
        self.device = device
        self.node_features.to(device)
        self.edge_features.to(device)
        self.neighbor_indices.to(device)
        if self.targets:
            self.targets.to(device)

    @property
    def device(self):
        """
        Returns device.
        """
        return self._device

    @device.setter
    def device(self, device):
        """
        Set device.
        """
        self._device = device

    @property
    def node_features(self) -> Tensor:
        """
        Returns node_features.
        """
        return self._node_features

    @node_features.setter
    def node_features(self, node_features):
        """
        Set node_features.
        """
        self._node_features = node_features

    @property
    def edge_features(self) -> Tensor:
        """
        Returns edge_features.
        """
        return self._edge_features

    @edge_features.setter
    def edge_features(self, edge_features):
        """
        Returns edge_features.
        """
        self._edge_features = edge_features

    @property
    def neighbor_indices(self) -> Tensor:
        """
        Returns neighbor_indices.
        """
        return self._neighbor_indices

    @neighbor_indices.setter
    def neighbor_indices(self, neighbor_indices):
        """
        Set neighbor_indices.
        """
        self._neighbor_indices = neighbor_indices

    @property
    def targets(self) -> Tensor:
        """
        Returns targets.
        """
        return self._targets

    @targets.setter
    def targets(self, targets):
        """
        Set targets.
        """
        self._targets = targets
