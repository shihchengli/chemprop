from abc import abstractmethod

from torch import Tensor

from chemprop.data import TrainingBatch
from chemprop.models.model import MPNN
from chemprop.utils.registry import ClassRegistry


class UncertaintyPredictor:
    def __call__(self, batch: TrainingBatch, model: MPNN):
        return self._calc_uncertainty(batch, model)

    @abstractmethod
    def _calc_uncertainty(self, batch, model) -> Tensor:
        """
        Calculate the uncalibrated uncertainties for the batch of data.
        """
        pass


UncertaintyPredictorRegistry = ClassRegistry[UncertaintyPredictor]()


@UncertaintyPredictorRegistry.register("mve")
class MVEPredictor(UncertaintyPredictor):
    def _calc_uncertainty(self, batch, model) -> Tensor:
        ...
        return


@UncertaintyPredictorRegistry.register("ensemble")
class EnsemblePredictor(UncertaintyPredictor):
    def _calc_uncertainty(self, batch, model) -> Tensor:
        ...
        return


@UncertaintyPredictorRegistry.register("classification")
class ClassPredictor(UncertaintyPredictor):
    def _calc_uncertainty(self, batch, model) -> Tensor:
        ...
        return


@UncertaintyPredictorRegistry.register("evidential-total")
class EvidentialTotalPredictor(UncertaintyPredictor):
    def _calc_uncertainty(self, batch, model) -> Tensor:
        ...
        return


@UncertaintyPredictorRegistry.register("evidential-epistemic")
class EvidentialEpistemicPredictor(UncertaintyPredictor):
    def _calc_uncertainty(self, batch, model) -> Tensor:
        ...
        return


@UncertaintyPredictorRegistry.register("evidential-aleatoric")
class EvidentialAleatoricPredictor(UncertaintyPredictor):
    def _calc_uncertainty(self, batch, model) -> Tensor:
        ...
        return


@UncertaintyPredictorRegistry.register("dropout")
class DropoutPredictor(UncertaintyPredictor):
    def _calc_uncertainty(self, batch, model) -> Tensor:
        ...
        return


@UncertaintyPredictorRegistry.register("spectra-roundrobin")
class RoundRobinSpectraPredictor(UncertaintyPredictor):
    def _calc_uncertainty(self, batch, model) -> Tensor:
        ...
        return


@UncertaintyPredictorRegistry.register("dirichlet")
class DirichletPredictor(UncertaintyPredictor):
    def _calc_uncertainty(self, batch, model) -> Tensor:
        ...
        return


@UncertaintyPredictorRegistry.register("conformal-quantile-regression")
class ConformalQuantileRegressionPredictor(UncertaintyPredictor):
    def _calc_uncertainty(self, batch, model) -> Tensor:
        ...
        return


@UncertaintyPredictorRegistry.register("conformal-regression")
class ConformalRegressionPredictor(UncertaintyPredictor):
    def _calc_uncertainty(self, batch, model) -> Tensor:
        ...
        return
