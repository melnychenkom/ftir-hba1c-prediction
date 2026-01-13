import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

def peaks_second_deriv(
        x_values, 
        y_values, 
        window_length=10,
        polyorder=2,
        **kwargs
    ):
    
    second_deriv = savgol_filter(
        y_values,
        window_length=window_length,
        polyorder=polyorder,
        deriv=2
    )
    second_deriv = second_deriv / np.linalg.norm(second_deriv)
    peaks = find_peaks(-second_deriv, **kwargs)[0]

    fig, axs = plt.subplots(2, 1, figsize=(7, 6))
    axs[0].plot(x_values, y_values)
    axs[0].scatter(
        x_values[peaks],
        y_values[peaks],
        color='k',
        marker='|',
        linewidth=1.5,
        zorder=5
    )

    for peak in peaks:
        peak_pos = x_values[peak]
        y_offset = y_values[peak] + 0.1 * y_values.mean()
        axs[0].annotate(
            f'{peak_pos:.2f}',
            xy=(peak_pos, y_values[peak]),
            xytext=(peak_pos - 3.5, y_offset),
            fontsize=8
        )

    axs[1].plot(x_values, second_deriv)
    axs[1].scatter(
        x_values[peaks],
        second_deriv[peaks],
        color='k',
        marker='|',
        linewidth=1.5,
        zorder=5
    )
    axs[1].axhline(y=0, linestyle="--", alpha=0.3, color='k')

    for peak in peaks:
        peak_pos = x_values[peak]
        y_offset = second_deriv[peak] - 0.6 * np.abs(second_deriv).mean()
        axs[1].annotate(
            f'{peak_pos:.2f}',
            xy=(peak_pos, second_deriv[peak]),
            xytext=(peak_pos - 3.5, y_offset),
            fontsize=8
        )

    return fig, axs


def find_peaks_matrix(
        x_values: pd.DataFrame,
        y_values: pd.Series,
        window_length=10,
        polyorder=2,
        threshold=0.5,
        **kwargs
    ):
    
    n_samples, n_features = y_values.shape
    
    X_deriv = y_values.T.apply(lambda col: savgol_filter(
        col, 
        window_length=window_length, 
        polyorder=polyorder, 
        deriv=2)
    ).T
    X_deriv = X_deriv.apply(lambda row: row / np.linalg.norm(row), axis=1)

    total_peaks = np.zeros(n_features)


    for index, row in X_deriv.iterrows():
        peaks = find_peaks(-row, **kwargs)[0]
        total_peaks[peaks] += 1


    peaks_counts, *_ = find_peaks(total_peaks, height=int(n_samples * threshold))
    
    plt.bar(x_values, total_peaks)
    plt.scatter(x_values[peaks_counts], total_peaks[peaks_counts])
    plt.show()

    return x_values[peaks_counts]