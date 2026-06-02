"""Tests for rose plot."""

import math
import os

import numpy as np
import pytest
from ethoinsight.charts import rose_plot


def test_rose_plot_creates_file(tmp_path):
    output = tmp_path / "rose.png"
    directions = np.linspace(0, 2 * math.pi, 36, endpoint=False)
    path = rose_plot(directions, n_bins=8, output_path=str(output))
    assert os.path.exists(path)


def test_rose_plot_empty_data(tmp_path):
    output = tmp_path / "rose_empty.png"
    directions = np.array([])
    path = rose_plot(directions, output_path=str(output))
    assert os.path.exists(path)


def test_rose_plot_default_bins(tmp_path):
    output = tmp_path / "rose_default.png"
    rng = np.random.default_rng(42)
    directions = rng.uniform(0, 2 * math.pi, size=100)
    path = rose_plot(directions, output_path=str(output))
    assert os.path.exists(path)
