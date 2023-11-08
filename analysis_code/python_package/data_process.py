# import pamonth_data
import os
import numpy as np
import pandas as pd
import netCDF4 as nc
from numba import njit


# class for data processing
class pro:
    def __init__(self):
        self.sym_lat = None
        self.asy_lat = None
        self.lat_sum = None
        self.weight_sym = None
        self.weight_asy = None
        self.avg_sym = None
        self.avg_asy = None
        self.time_range = None
        self.split_data = None

    def sym_format(self, lat, data):
        self.sym_lat = np.cos(np.deg2rad(lat))
        self.lat_sum = np.sum(self.sym_lat)

        self.weight_sym = data * self.sym_lat[None, :, None]

        self.avg_sym = np.sum(self.weight_sym, axis=1) / self.lat_sum

        return self.avg_sym

    def asy_format(self, lat, data):
        if lat[0] < 0.0:
            lat = lat[::-1]
            data = data[:, ::-1, :]

        self.asy_lat = np.empty_like(lat)
        self.asy_lat[: int(lat.shape[0] / 2)] = np.cos(
            np.deg2rad(lat[: int(lat.shape[0] / 2)])
        )
        self.asy_lat[int(lat.shape[0] / 2) :] = -np.cos(
            np.deg2rad(lat[int(lat.shape[0] / 2) :])
        )

        self.weight_asy = data * self.asy_lat[None, :, None]

        self.avg_asy = np.sum(self.weight_asy, axis=1) / self.lat_sum

        return self.avg_asy


@njit
def background(data, num_of_passes):
    padded_data = np.empty((data.shape[0], data.shape[1] + 2), dtype=data.dtype)
    padded_data[:, 0] = data[:, 1]
    padded_data[:, -1] = data[:, -2]
    padded_data[:, 1:-1] = data

    for k in range(num_of_passes):
        for j in range(padded_data.shape[0]):
            for i in range(1, padded_data.shape[1] - 1):
                padded_data[j, i] = (
                    padded_data[j, i - 1]
                    + padded_data[j, i] * 2
                    + padded_data[j, i + 1]
                ) / 4

    return padded_data[:, 1:-1]


def data_combine(year, month, head):
    month_data = []

    first_day = pd.Timestamp(year, month, 1)
    last_day = first_day + pd.offsets.MonthEnd()
    num_days = (last_day - first_day).days + 1

    for i in range(1, num_days + 1):
        date = "{:02d}".format(i)
        path = os.path.expanduser(head + date + ".nc")
        dset = nc.Dataset(path, "r")
        olr_data = np.asarray(dset.variables["olrtoa"][:])
        olr_data = np.squeeze(olr_data, axis=0)
        month_data.append(olr_data)

    return np.asarray(month_data)


def data_series(start_month, end_month, year, path, lon, lat):
    output_data = np.empty((1, len(lat), len(lon)))

    for i in range(start_month, end_month + 1):
        head = path + "2000-{:02d}-".format(i)
        monthly_data = data_combine(year, i, head)
        output_data = np.append(output_data, monthly_data, axis=0)
        del monthly_data

    output_data = output_data[1:, :, :]
    output_time = np.linspace(0, output_data.shape[0] - 1, output_data.shape[0])

    return output_data, output_time