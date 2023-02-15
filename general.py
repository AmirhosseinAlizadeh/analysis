import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class General:
    """
    A class for performing general data analysis tasks.

    This class contains the following methods:
    - __init__(self, filename): A constructor method that initializes a Pandas DataFrame from a CSV file.
    - read_csv(self, n_cols=None, col_contains=None, dropna=True): A method for reading a CSV file into a Pandas DataFrame.
    - plot_data(self, x_label, y_label, z_label=None, n_cols=None, fit_mode=None): A method for creating 2D and 3D plots from a Pandas DataFrame.
    - print_statistics(self, columns=None, dropna=True): A method for printing various statistical insights about a Pandas DataFrame.
    """
    
    def __init__(self, filename):
        """
        A constructor method that initializes a Pandas DataFrame from a CSV file.

        :param filename: The name of the CSV file to read.
        """
        self.df = pd.read_csv(filename)
        
    def read_csv(self, n_cols=None, col_contains=None, dropna=True, clean_up=False):
        """
        A method for reading a CSV file into a Pandas DataFrame.

        :param n_cols: The number of columns to read from the CSV file.
        :param col_contains: A string that the column name contains to be selected.
        :param dropna: If True, drop rows with missing values.
        :param clean_up: If True, drop rows containing NaN values.
        """
        if n_cols is not None:
            self.df = pd.read_csv(self.filename, usecols=range(n_cols))
        elif col_contains is not None:
            self.df = pd.read_csv(self.filename)
            col_names = [col for col in self.df.columns if col_contains in col]
            self.df = self.df[col_names]
        else:
            self.df = pd.read_csv(self.filename)

        if clean_up:
            self.df = self.df.dropna()

        if dropna:
            self.df = self.df.dropna()        
  
    def plot_data(self, x_label, y_label, z_label=None, num_cols=None, fit_mode=None, clean_up=False):
        """
        A function that plots a Pandas DataFrame with a specified number of columns and axis labels,
        and optionally fits a curve to the data.

        :param df: A Pandas DataFrame to be plotted.
        :param x_label: The label for the x-axis.
        :param y_label: The label for the y-axis.
        :param z_label: The label for the z-axis (optional).
        :param num_cols: The number of columns to plot from the DataFrame. If None, all columns are plotted.
        :param fit_mode: The type of curve fitting to perform. If None, no curve fitting is performed.
        :param clean_up: If True, rows containing NaN values are removed from the DataFrame before plotting.
        """
        # Select the columns to be plotted based on the value of num_cols.
        if num_cols is not None:
            df = self.df
            df = df.iloc[:, :num_cols]

        # Remove rows containing NaN values if clean_up is True.
        if clean_up:
            df = df.dropna()

        # Determine the number of columns to be plotted.
        num_plots = df.shape[1]

        # Create a figure and set the axis labels.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') if z_label is not None else fig.add_subplot(111)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if z_label is not None:
            ax.set_zlabel(z_label)

        # Plot the data for each column.
        for i in range(num_plots):
            x = df.index
            y = df.iloc[:, i]
            ax.scatter(x, y)

            # Fit a curve to the data if fit_mode is specified.
            if fit_mode is not None:
                fit_func, init_params = get_fit_func(fit_mode)
                popt, pcov = curve_fit(fit_func, x, y, p0=init_params)
                fit_y = fit_func(x, *popt)
                ax.plot(x, fit_y)

        # Display the plot.
        plt.show()


    def get_fit_func(fit_mode):
        """
        Returns the appropriate fit function and initial parameter values for the specified fit mode.

        :param fit_mode: The type of curve fitting to perform.
        """
        if fit_mode == 'gaussian':
            fit_func = lambda x, a, b, c: a * np.exp(-(x - b)**2 / (2 * c**2))
            init_params = [1, 0, 1]
        elif fit_mode == 'lorentzian':
            fit_func = lambda x, a, b, c: a / (1 + ((x - b) / c)**2)
            init_params = [1, 0, 1]
        elif fit_mode == 'linear':
            fit_func = lambda x, a, b: a * x + b
            init_params = [1, 0]
        elif fit_mode == 'exponential':
            fit_func = lambda x, a, b: a * np.exp(b * x)
            init_params = [1, 0]
        elif fit_mode == 'quadratic':
            fit_func = lambda x, a, b, c: a * x**2 + b * x



    def print_statistics(self, columns=None, dropna=True):
        """
        A function that prints out various statistical insights about a Pandas DataFrame.

        :param df: A Pandas DataFrame.
        :param columns: A list of column names to use. If None, use all columns.
        :param dropna: If True, drop rows with missing values before calculating statistics.
        """
        df = self.df
        if columns is None:
            columns = df.columns
        else:
            columns = [col for col in columns if col in df.columns]

        selected_df = df[columns]

        if dropna:
            selected_df = selected_df.dropna()

        print(f"Number of rows: {selected_df.shape[0]}")
        print(f"Number of columns: {selected_df.shape[1]}")
        print(f"Column names: {', '.join(selected_df.columns)}")
        print(f"Max values:\n{selected_df.max()}")
        print(f"Min values:\n{selected_df.min()}")
        print(f"Average values:\n{selected_df.mean()}")
        print(f"Standard deviations:\n{selected_df.std()}")
        print(f"Range:\n{selected_df.max() - selected_df.min()}")
        print(f"Variance:\n{selected_df.var()}")
        print(f"Skewness:\n{selected_df.skew()}")
        print(f"Kurtosis:\n{selected_df.kurtosis()}")
        print(f"Median:\n{selected_df.median()}")
        print(f"Mode:\n{selected_df.mode().iloc[0]}")
        print(f"25th percentile:\n{selected_df.quantile(0.25)}")
        print(f"50th percentile (median):\n{selected_df.quantile(0.5)}")
        print(f"75th percentile:\n{selected_df.quantile(0.75)}")
        print(f"95th percentile:\n{selected_df.quantile(0.95)}")
    
    
