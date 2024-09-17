"""Calculate sleep onset and wake up times with a CNN UNET classifier.

This CNN U-NET classifier is a re-implementation of the classifier proposed in
a Kaggle competition (c.f. https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/460274).
"""

from __future__ import annotations

import datetime
from collections.abc import Generator
from typing import Sequence

import numpy as np
import polars as pl
from scipy import signal

from wristpy.core import config, models

logger = config.get_logger()

try:
    import torch
    from torch import nn
    from torch.nn import functional
    from torch.utils import data
except ImportError:
    logger.exception(
        " ".join(
            "Extra dependency 'torch' not found. Please install wristpy with"
            "the 'machine_learning' flag i.e. `pip install 'wristpy[machine_learning]'`"
        )
    )

MODEL_DIR = config.DATA_DIR / "CNN_UNET"


class _ConvBNReLU(nn.Module):
    """A 1D convolution, batch normalization, and ReLU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int = 1,
    ) -> None:
        """Creates the neural network building block.

        Args:
            in_channels: Number of channels in the input data
            out_channels: Number of channels produced by the convolution
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution.
            groups: Number of blocked connections from input channels to output
                channels.

        """
        super().__init__()

        padding: str | int = "same" if stride == 1 else int((kernel_size - stride) // 2)
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the network.

        Args:
            x: Input tensor.

        Returns:
            The output tensor.
        """
        return self.layers(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Parent's __call__ method, required for type hinting the call output."""
        return super().__call__(x)


class _SEBlock(nn.Module):
    """Implements a Squeeze-and-Excitation (SE) mechanism."""

    def __init__(self, n_channels: int, se_ratio: int) -> None:
        """Initalizes the SE Block.

        Args:
            n_channels: Number of input channels.
            se_ratio: Squeeze-excitation ratio.
        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Conv1d(n_channels, n_channels // se_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(n_channels // se_ratio, n_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the network.

        Args:
            x: Input tensor.

        Returns:
            The output tensor.
        """
        return torch.mul(x, self.layers(x))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Parent's __call__ method, required for type hinting the call output."""
        return super().__call__(x)


class _ResBlock(nn.Module):
    def __init__(self, n_channels: int, kernel_size: int, se_ratio: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            _ConvBNReLU(n_channels, n_channels, kernel_size, stride=1),
            _ConvBNReLU(n_channels, n_channels, kernel_size, stride=1),
            _SEBlock(n_channels, se_ratio),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the network.

        Args:
            x: Input tensor.

        Returns:
            The output tensor.
        """
        return x + self.layers(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Parent's __call__ method, required for type hinting the call output."""
        return super().__call__(x)


class _Dataset(data.Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray | None, flag: np.ndarray) -> None:
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y) if Y else None
        self.flag = torch.FloatTensor(flag)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.Y:
            return (self.X[idx], self.Y[idx], self.flag[idx])
        return (self.X[idx], torch.Tensor(), self.flag[idx])


class _UNet1d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        initial_channels: int,
        initial_kernel_size: int,
        down_channels: Sequence[int],
        down_kernel_size: Sequence[int],
        down_stride: Sequence[int],
        res_depth: int,
        res_kernel_size: int,
        se_ratio: int,
        out_kernel_size: int,
    ) -> None:
        super().__init__()
        self.down_kernel_size = down_kernel_size
        self.down_stride = down_stride

        self.initial_layers = _ConvBNReLU(
            input_channels,
            initial_channels,
            initial_kernel_size,
            stride=1,
            groups=input_channels,
        )

        self.down_layers = nn.ModuleList()
        for index in range(len(down_channels)):
            if index == 0:
                in_channels = initial_channels
            else:
                in_channels = down_channels[index - 1] + input_channels
            out_channels = down_channels[index]
            kernel_size = down_kernel_size[index]
            stride = down_stride[index]

            block = [_ConvBNReLU(in_channels, out_channels, kernel_size, stride)] + [
                _ResBlock(out_channels, res_kernel_size, se_ratio)
            ] * res_depth

            self.down_layers.append(nn.Sequential(*block))

        self.up_layers = nn.ModuleList()
        for index in range(len(down_channels) - 1, 0, -1):
            in_channels = out_channels + down_channels[index]
            out_channels = down_channels[index]
            kernel_size = down_kernel_size[index]
            self.up_layers.append(
                _ConvBNReLU(in_channels, out_channels, kernel_size, stride=1)
            )

        self.out_layers = nn.Conv1d(
            down_channels[1], 1, out_kernel_size, padding="same"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the network.

        Args:
            x: Input tensor.

        Returns:
            The output tensor.
        """
        outs = []
        x_avg = x
        x = self.initial_layers(x)

        for index in range(len(self.down_layers)):
            x_out = self.down_layers[index](x)
            if index == len(self.down_layers) - 1:
                x = x_out
            else:
                outs.append(x_out)
                kernel_size = self.down_kernel_size[index]
                stride = self.down_stride[index]
                padding = (kernel_size - stride) // 2
                x_avg = functional.avg_pool1d(x_avg, kernel_size, stride, padding)
                x = torch.cat([x_out, x_avg], dim=1)

        for index in range(len(self.up_layers)):
            scale_factor = self.down_stride[-index - 1]
            x = functional.interpolate(x, scale_factor=scale_factor, mode="linear")
            x = torch.cat([x, outs[-index - 1]], dim=1)
            x = self.up_layers[index](x)

        return self.out_layers(x)[:, 0, 180:-180]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Parent's __call__ method, required for type hinting the call output."""
        return super().__call__(x)


class CnnUnet:
    """CNN U-NET classifier for sleep detection."""

    def __init__(
        self,
        device_name: str = "cpu",
    ) -> None:
        """Initializes the CNN UNET classifier.

        Args:
            device_name: The device to run the neural network on. Common
                usages are 'cpu' and 'cuda:0', but see the help of torch.device
                for more information.

        """
        self.batch_size = 16  # Derived from the Kaggle default.
        self.device = torch.device(device_name)
        self.model = self._load_model()

    @torch.no_grad()
    def run_sleep_detection(
        self, anglez: models.Measurement, enmo: models.Measurement
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        predictions = []
        for loader in self._create_dataloaders(anglez, enmo):
            for batch in loader:
                X = batch[0].to(self.device)
                mask = batch[2].to(self.device)
                preds = self.model(X) * mask
                predictions.append(preds.cpu().numpy())

        all_predictions = np.concatenate(predictions)
        return self._predictions_to_times(anglez, enmo, all_predictions)

    @staticmethod
    def _predictions_to_times(
        anglez: models.Measurement,
        enmo: models.Measurement,
        predictions: list[list[float]],
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        interval_1min = _time_interval_strings(interval=60)
        df_1min = (
            pl.DataFrame()
            .with_columns(
                anglez=anglez.measurements,  # type: ignore[arg-type] # Lies. It does accept ndarrays.
                enmo=enmo.measurements,  # type: ignore[arg-type] # Lies. It does accept ndarrays.
                timestamp=anglez.time,
            )
            .with_columns(pl.col("timestamp").dt.strftime("%H:%M:%S").alias("time"))
            .with_columns(pl.col("time").cum_count().over("time").alias("level"))
        )
        df = (
            pl.DataFrame(predictions, schema=interval_1min)
            .unpivot(variable_name="time", value_name="score")
            .with_columns(pl.col("time").cum_count().over("time").alias("level"))
        )
        df_merge = df.join(df_1min, on=["time", "level"])

        onset_peaks = signal.find_peaks(-df_merge["score"], height=0.0, distance=8)[0]
        offset_peaks = signal.find_peaks(df_merge["score"], height=0.0, distance=8)[0]
        onsets = df_merge[onset_peaks]
        offsets = df_merge[offset_peaks]

        return onsets, offsets

    def _create_dataloaders(
        self,
        anglez: models.Measurement,
        enmo: models.Measurement,
    ) -> Generator[data.DataLoader, None, None]:
        """Creates the Torch data loader on a per-day basis.

        Angle-Z data preprocessing involves a 25-second windowed standard
        deviation, addition of 1, and a log transformation. ENMO data
        preprocessing involves a 25-second windowed standard deviation, addition
        of 0.01, and a log transformation. The valid_flag used in the data
        is a remnant of the model being trained on data where masking was required.

        Args:
            anglez: _description_
            enmo: _description_
            day: _description_

        Returns:
            _description_
        """
        if not (anglez.time == enmo.time).all():
            msg = "ENMO and angle z must have the same time data."
            logger.error(msg)
            raise ValueError(msg)

        df = (
            pl.DataFrame(
                {
                    "anglez": anglez.measurements,
                    "enmo": enmo.measurements,
                    "timestamp": anglez.time,
                }
            )
            .set_sorted("timestamp")
            .with_columns(
                pl.col("anglez")
                .rolling_std_by(by="timestamp", window_size="25s")
                .add(pl.lit(1))
                .log(),
                pl.col("enmo")
                .rolling_std_by(by="timestamp", window_size="25s")
                .add(pl.lit(0.01))
                .log(),
                date=pl.col("timestamp").dt.date(),
                time=pl.col("timestamp").dt.time(),
                valid_flag=pl.lit(1),
            )
            .group_by_dynamic("timestamp", every="5s")
            .agg(
                pl.col("anglez").mean(),
                pl.col("enmo").mean(),
                pl.col("date").first(),
                pl.col("time").first(),
                pl.col("valid_flag").first(),
            )
        )

        n_unique_days = anglez.time.dt.date().n_unique()
        for index in range(n_unique_days):
            feature_vectors = []
            for feature in ("anglez", "enmo", "valid_flag"):
                pivot = (
                    df.pivot(
                        on=["time"], index="date", values=feature, sort_columns=True
                    )
                    .shift(index)
                    .drop("date")
                )

                yesterday = pivot.shift(-1).fill_null(0).fill_nan(0)
                today = pivot.fill_null(0).fill_nan(0)
                tomorrow = pivot.shift(1).fill_null(0).fill_nan(0)

                feature_vectors.append(
                    np.concatenate(
                        (
                            yesterday[0, -180 * 12 :],
                            today[0, :],
                            tomorrow[0, : 180 * 12],
                        ),
                        axis=1,
                    )
                )
            feature_array = np.concatenate(feature_vectors, axis=0)[None, :, :]
            mask = np.ones((feature_array.shape[0], 1440))
            yield data.DataLoader(
                _Dataset(
                    feature_array,
                    None,
                    mask,
                ),
                self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                drop_last=False,
            )

    @torch.no_grad()
    def _load_model(self) -> _UNet1d:
        n_splits = 10
        model_files = [MODEL_DIR / f"model_{index}.pth" for index in range(n_splits)]
        if any(not file.exists() for file in model_files):
            logger.info("Model files not found. Downloading...")
            self._download_model()

        model = _UNet1d(
            input_channels=3,
            initial_channels=72,
            initial_kernel_size=15,
            down_channels=(72, 72, 72),
            down_kernel_size=(12, 15, 15),
            down_stride=(12, 9, 5),
            res_depth=3,
            res_kernel_size=15,
            se_ratio=4,
            out_kernel_size=21,
        )
        model.to(self.device)
        for index in range(n_splits):
            model.load_state_dict(
                torch.load(
                    MODEL_DIR / f"model_{index}.pth", self.device, weights_only=True
                )
            )
        model.eval()
        return model

    @staticmethod
    def _download_model() -> None:
        # TODO: Host the files somewhere
        raise NotImplementedError


def _time_interval_strings(interval: float, *, format: str = "%H:%M:%S") -> list[str]:
    """Returns all timepoints in a day at a specified interval.

    Args:
        interval: The interval, in seconds.
        format: The output format specifier, c.f. datetime.datetime.strftime
            for details.

    """
    start_time = datetime.datetime.strptime("00:00:00", format)
    n_seconds_in_day = 86400
    n_timepoints = int(n_seconds_in_day // interval)
    return [
        (start_time + datetime.timedelta(seconds=interval * index)).strftime(format)
        for index in range(n_timepoints)
    ]
