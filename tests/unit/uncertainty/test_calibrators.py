import pytest
import torch

from chemprop.uncertainty.calibrator import (
    IsotonicCalibrator,
    IsotonicMulticlassCalibrator,
    PlattCalibrator,
    ZelikmanCalibrator,
    ZScalingCalibrator,
)


@pytest.mark.parametrize(
    "cal_uncs,cal_targets,cal_mask,test_uncs,cal_test_uncs",
    [
        (
            torch.tensor(
                [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [0.7, 0.8, 0.9],
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [0.7, 0.8, 0.9],
                ]
            ),
            torch.tensor([[0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 0]]),
            torch.tensor(
                [[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.bool
            ),
            torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            torch.tensor([[1 / 3, 2 / 3, 0.0], [1 / 3, 2 / 3, 0.5]]),
        )
    ],
)
def test_IsotonicCalibrator(cal_uncs, cal_targets, cal_mask, test_uncs, cal_test_uncs):
    """
    Testing the IsotonicCalibrator
    """
    calibrator = IsotonicCalibrator()
    calibrator.fit(cal_uncs, cal_targets, cal_mask)
    uncs = calibrator.apply(test_uncs)

    torch.testing.assert_close(uncs, cal_test_uncs)


@pytest.mark.parametrize(
    "cal_uncs,cal_targets,cal_mask,test_uncs,cal_test_uncs,training_targets,cal_test_uncs_with_training_targets",
    [
        (
            torch.tensor(
                [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [0.7, 0.8, 0.9],
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [0.7, 0.8, 0.9],
                ]
            ),
            torch.tensor([[0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 0]]),
            torch.tensor(
                [[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.bool
            ),
            torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            torch.tensor(
                [[0.4183908, 0.8000248, 0.1312900], [0.3975054, 0.7999378, 0.2770228]],
                dtype=torch.float64,
            ),
            torch.tensor([[0, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 1]]),
            torch.tensor(
                [[0.5285367, 0.6499191, 0.3089508], [0.5188822, 0.6499544, 0.3998689]],
                dtype=torch.float64,
            ),
        )
    ],
)
def test_PlattCalibrator(
    cal_uncs,
    cal_targets,
    cal_mask,
    test_uncs,
    cal_test_uncs,
    training_targets,
    cal_test_uncs_with_training_targets,
):
    """
    Testing the PlattCalibrator
    """
    calibrator1 = PlattCalibrator()
    calibrator1.fit(cal_uncs, cal_targets, cal_mask)
    uncs1 = calibrator1.apply(test_uncs)

    calibrator2 = PlattCalibrator()
    calibrator2.fit(cal_uncs, cal_targets, cal_mask, training_targets)
    uncs2 = calibrator2.apply(test_uncs)

    torch.testing.assert_close(uncs1, cal_test_uncs)
    torch.testing.assert_close(uncs2, cal_test_uncs_with_training_targets)


@pytest.mark.parametrize(
    "cal_preds,cal_uncs,cal_targets,cal_mask,test_uncs,cal_test_uncs",
    [
        (
            torch.zeros(100, 1, dtype=float),
            torch.arange(1, 101, dtype=float).unsqueeze(1).pow(2),
            torch.arange(1, 101, dtype=float).unsqueeze(1),
            torch.ones(100, 1, dtype=bool),
            torch.arange(1, 101, dtype=float).unsqueeze(1),
            torch.arange(1, 101, dtype=float).unsqueeze(1),
        ),
        (
            torch.zeros(100, 1, dtype=float),
            torch.arange(2, 201, step=2, dtype=float).unsqueeze(1).pow(2),
            torch.arange(1, 101, dtype=float).unsqueeze(1),
            torch.ones(100, 1, dtype=bool),
            torch.arange(1, 101, dtype=float).unsqueeze(1),
            torch.arange(1, 101, dtype=float).unsqueeze(1) / 4,
        ),
    ],
)
def test_ZScalingCalibrator(cal_preds, cal_uncs, cal_targets, cal_mask, test_uncs, cal_test_uncs):
    """
    Testing the ZScalingCalibrator
    """
    calibrator = ZScalingCalibrator()
    calibrator.fit(cal_preds, cal_uncs, cal_targets, cal_mask)
    uncs = calibrator.apply(test_uncs)

    torch.testing.assert_close(uncs, cal_test_uncs)


@pytest.mark.parametrize(
    "cal_preds,cal_uncs,cal_targets,cal_mask,test_uncs,cal_test_uncs",
    [
        (
            torch.zeros(100, 1, dtype=float),
            torch.arange(1, 101, dtype=float).unsqueeze(1).pow(2),
            torch.arange(1, 101, dtype=float).unsqueeze(1),
            torch.ones(100, 1, dtype=bool),
            torch.arange(1, 101, dtype=float).unsqueeze(1),
            torch.arange(1, 101, dtype=float).unsqueeze(1),
        ),
        (
            torch.zeros(100, 1, dtype=float),
            torch.ones(100, 1, dtype=bool),
            torch.arange(1, 101, dtype=float).unsqueeze(1),
            torch.ones(100, 1, dtype=bool),
            torch.arange(1, 101, dtype=float).unsqueeze(1),
            torch.arange(1, 101, dtype=float).unsqueeze(1) * 8100,
        ),
    ],
)
def test_ZelikmanCalibrator(cal_preds, cal_uncs, cal_targets, cal_mask, test_uncs, cal_test_uncs):
    """
    Testing the ZelikmanCalibrator
    """
    calibrator = ZelikmanCalibrator(p=0.9)
    calibrator.fit(cal_preds, cal_uncs, cal_targets, cal_mask)
    uncs = calibrator.apply(test_uncs)


@pytest.mark.parametrize(
    "cal_uncs,cal_targets,cal_mask,test_uncs,cal_test_uncs",
    [
        (
            torch.tensor(
                [
                    [[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]],
                    [[0.1, 0.6, 0.3], [0.4, 0.4, 0.2]],
                    [[0.4, 0.4, 0.2], [0.2, 0.3, 0.5]],
                    [[0.0, 0.6, 0.4], [0.8, 0.1, 0.1]],
                    [[0.5, 0.2, 0.3], [0.4, 0.4, 0.2]],
                    [[0.4, 0.3, 0.3], [0.7, 0.3, 0.0]],
                ]
            ),
            torch.tensor([[2, 1], [1, 2], [0, 2], [1, 1], [0, 0], [2, 0]]).long(),
            torch.ones([6, 2], dtype=torch.bool),
            torch.tensor(
                [
                    [[0.0, 0.1, 0.9], [0.5, 0.2, 0.3]],
                    [[0.3, 0.4, 0.3], [0.6, 0.3, 0.1]],
                    [[0.9, 0.1, 0.0], [0.3, 0.4, 0.3]],
                ]
            ),
            torch.tensor(
                [
                    [[0.000000, 0.000000, 1.000000], [0.483871, 0.193548, 0.322581]],
                    [[0.500000, 0.000000, 0.500000], [0.714286, 0.285714, 0.000000]],
                    [[1.000000, 0.000000, 0.000000], [0.319149, 0.255319, 0.425532]],
                ]
            ),
        )
    ],
)
def test_IsotonicMulticlassCalibratorCalibrator(
    cal_uncs, cal_targets, cal_mask, test_uncs, cal_test_uncs
):
    """
    Testing the IsotonicMulticlassCalibratorCalibrator
    """
    calibrator = IsotonicMulticlassCalibrator()
    calibrator.fit(cal_uncs, cal_targets, cal_mask)
    uncs = calibrator.apply(test_uncs)
    torch.testing.assert_close(uncs, cal_test_uncs)
