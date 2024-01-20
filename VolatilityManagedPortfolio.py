import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.metrics.pairwise import cosine_similarity

def price2ret(price, period="W", how="simple"):
    """
    Calculate the return series based on the given price series.

    Parameters:
    - price (pd.Series): The input price series.
    - period (str, optional): The resampling period, default is "W" (weekly).
    - how (str, optional): The method for calculating returns, default is "simple".

    Returns:
    - pd.Series: The return series based on the specified parameters.
    """
    if period == "W":
        temp = price.fillna(-np.inf).resample(period).last().replace(-np.inf, np.nan)
    elif period == "D":
        temp = price.copy()
    if how == "simple" :
        return temp.pct_change(fill_method=None).replace(np.inf, np.nan)
    elif how == "log" :
        return np.log(temp/temp.shift(1)).replace([np.inf, -np.inf], np.nan)

def calculate_weights(factor, marketCap, volume, min_marketCap=1e6, low=0.2, high=0.8, Mode="high-low"):
    """
    Calculate portfolio weights based on factor data and market capitalization, with customization options.

    Parameters:
        factor (pd.DataFrame): A DataFrame containing factor data with assets as columns and time periods as rows.
        marketCap (pd.DataFrame): A DataFrame containing market capitalization data with assets as columns and time periods as rows.
        min_marketCap (float, optional): Minimum market capitalization threshold (in USD) to consider assets. Defaults to 1,000,000 USD.
        low (float, optional): Lower percentile boundary for quintile calculation. Defaults to 0.2 (20th percentile).
        high (float, optional): Upper percentile boundary for quintile calculation. Defaults to 0.8 (80th percentile).

    Returns:
        pd.DataFrame: A DataFrame containing calculated weights with assets as columns and time periods as rows.

    Notes:
        - The function calculates portfolio weights based on factor data and market capitalization.
        - It allows customization of the minimum market capitalization threshold and quintile boundaries.
        - Market capitalization values are winsorized within the specified percentile boundaries for each row.
        - Zeros in the factor data are replaced with NaN to exclude them from quintile calculations.
        - Quintiles are calculated based on the customized `low` and `high` percentiles.
        - Q2, Q3, and Q4 weights are set to 0, while Q1 and Q5 weights are set to 1.
        - Weights for Q1 are calculated based on winsorized market capitalization and normalized to sum to 1.
        - Weights for Q5 are calculated similarly but with a sum of -1 to represent a short position.

    Example Usage:
        weight_df = calculate_weights(factor_df, marketCap_df, min_marketCap=500000, low=0.1, high=0.9)
    """
    if Mode == "high-BTC" or Mode == "low-BTC":
        # Remove BTC-related data if specified
        factor = factor.drop("1", axis=1)
        marketCap = marketCap.drop("1", axis=1)
        volume = volume.drop("1", axis=1)

    # Winsorize market capitalization data
    marketCap_winsorized = marketCap.apply(lambda row: row.clip(upper=row.quantile(0.99)), axis=1)
    
    # Filter coins with a market cap above the minimum threshold and scale down for future calculations
    marketCap_filtered = marketCap_winsorized.where(marketCap_winsorized >= min_marketCap, np.nan) / 1e5
        
    # Filter 'factor' to align with 'marketCap' data and remove coins not traded during portfolio formation week
    factor = factor.where(np.log(marketCap_filtered).notna(), np.nan)
    factor = factor.where(np.log(volume).notna(), np.nan)
        
    # Step 1: Divide the corresponding rows of the factor dataframe into quintiles
    quintiles = factor.rank(axis=1, pct=True)  # method = 'first'

    # Step 2: Set Q2, Q3, and Q4 weights to 0, Q1 and Q5 weights to 1
    weight = quintiles.applymap(lambda x: 1 if (x <= low or x > high) else 0)

    # Step 3: Calculate weights for Q1 based on winsorized market capitalization
    q1_weights = marketCap_filtered.where(quintiles <= low, np.nan)
    q1_weights = -q1_weights.div(q1_weights.sum(axis=1), axis=0)

    # Step 4: Calculate weights for Q5 based on winsorized market capitalization
    q5_weights = marketCap_filtered.where(quintiles > high, np.nan)
    q5_weights = q5_weights.div(q5_weights.sum(axis=1), axis=0)

    if Mode == "high-low":
        weight = (weight * q1_weights).replace(np.nan, 0) + (weight * q5_weights).replace(np.nan, 0)
    elif Mode == "low-high":
        weight = -((weight * q1_weights).replace(np.nan, 0) + (weight * q5_weights).replace(np.nan, 0))
    elif Mode == "high" or Mode == "high-BTC":
        weight = (weight * q5_weights).replace(np.nan, 0)
    elif Mode == "low" or Mode == "low-BTC":
        weight = -(weight * q1_weights).replace(np.nan, 0)
        
    # Shift the weight dataframe by one row to get the previous weights
    return weight.shift(1)

def calculate_portfolio_returns(return_df, weight_df, factor="factor", liquidity_filter="OFF", volume_D=None):
    """
    Calculate portfolio returns based on return data and weight data.

    Parameters:
        return_df (pd.DataFrame): A DataFrame containing return data with assets as columns and time periods as rows.
        weight_df (pd.DataFrame): A DataFrame containing weight data with assets as columns and time periods as rows.
        factor (str, optional): The name of the factor or portfolio. Defaults to "factor".
        liquidity_filter (str, optional): Whether to apply a liquidity filter ("ON" or "OFF"). Defaults to "OFF".
        volume_D (pd.Series, optional): A DataFrame containing trading volume data. Required only when liquidity_filter is "ON".
        
    Returns:
        pd.DataFrame: A DataFrame containing portfolio returns with a single column named after the factor.
    """

    # Calculate portfolio returns by taking the dot product of return_df and weight_df
    temp = (return_df * weight_df)
    
    if liquidity_filter == "ON":
        if volume_D is None:
            raise ValueError("volume_D is required when liquidity_filter is 'ON'")
        temp[(temp > 0) & (np.log(volume_D).isna())] = 0  # positive returns are not available when trading volume = 0
        
    portfolio_returns = pd.DataFrame(temp.sum(axis=1), columns=[factor]).clip(lower=-1)
        
    return portfolio_returns

def calculate_t_stat_and_p_value(returns_df):
    """
    Calculate t-statistic and p-value for each asset in a DataFrame.

    Args:
    returns_df (pd.DataFrame): DataFrame with asset returns.

    Returns:
    pd.DataFrame: DataFrame with asset names, t-statistics, and p-values.
    """

    # Conduct one-sample t-test for each asset against a population mean of 0
    t_stats, p_values = stats.ttest_1samp(returns_df, 0, nan_policy='omit')

    # Create a DataFrame with asset names, t-statistics, and p-values
    results_df = pd.DataFrame({
        'Asset': returns_df.columns,
        'T-Statistic': t_stats,
        'P-Value': p_values
    }).set_index("Asset")

    return results_df.T


def plot_Risk_Return(price, marketCap, period="W", q=3, start="2015", end="2022", save=False):
    # Calculate weekly and daily returns
    ret_W = price2ret(price)
    ret_D = price2ret(price, period="D")
    
    # Resample market cap to weekly frequency and calculate index weight
    weekend_marketCap = marketCap.fillna(-np.inf).resample("W").last().replace(-np.inf,np.nan)
    index_weight = weekend_marketCap.div(weekend_marketCap.sum(axis=1), axis=0).replace(0, np.nan).shift(1)
    
    # Calculate index returns for both weekly and daily frequencies
    index_ret = calculate_portfolio_returns(ret_W, index_weight, factor="idx")
    index_weight_D = marketCap.div(marketCap.sum(axis=1), axis=0).replace(0, np.nan).shift(1)
    index_ret_D = calculate_portfolio_returns(ret_D, index_weight_D, factor="idx")
    
    # Calculate log returns and create a DataFrame for analysis
    series = np.log(1+index_ret_D.idx)
    series_W = np.log(1+index_ret.idx)
    Data = pd.DataFrame(series_W)
    
    # Calculate volatility and other risk measures
    Data["vol"] = series.resample(period).std()
    vol_BH = series_W.std()
    vol_temp_optimal = (series_W/(series.resample(period).var()) * series_W).std()
    c = vol_BH / vol_temp_optimal
    Data["optimal_risk_exposure"] = c * series_W / (series.resample(period).var())
    Data["vol_last_period"] = Data["vol"].shift(1)
    Data["vol_tercile"] = pd.qcut(Data["vol_last_period"], q, labels=False)

    # Filter data based on the specified time range
    Data = Data.loc[start:end]
    
    # Create subplots for analysis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Create a figure with 3 subplots in a row

    for i, variable in enumerate(["vol", "idx", "optimal_risk_exposure"]):
        ax = axes[i]  # Get the current axis
        Data.groupby("vol_tercile").mean()[variable].plot(kind='bar', ax=ax)

        # Customize the plot
        ax.set_xlabel('Last-week Volatility')
        if variable == "vol":
            ax.set_ylabel('Standard Deviation')
            ax.set_title('Panel A: Standard Deviation by Volatility', loc='left')
        elif variable == "idx":
            ax.set_ylabel("Following Week's Returns")
            ax.set_title('Panel B: Return by Volatility', loc='left')
        elif variable == "optimal_risk_exposure":
            ax.set_ylabel('E(R)/Var(R)')
            ax.set_title('Panel C: E(R)/Var(R) ratio by Volatility', loc='left')
        ax.set_xticks(range(q))
        ax.set_xticklabels([f'{i+1}' for i in range(q)], rotation=0)

    plt.tight_layout()  # Ensure subplots do not overlap

    # Save or display the plot
    if save:
        plt.savefig("Figure1.png", format='png')  # You can change the format if needed
    else:
        plt.show()

def remove_outliers(df, n=4):
    # Create a copy of the original DataFrame
    df_copy = df.copy()

    # Calculate z-scores for each column using vectorized operations
    z_scores = (df_copy - df_copy.mean()) / df_copy.std()

    # Create a boolean mask to identify outliers
    is_outlier = abs(z_scores) > n

    # Replace outlier elements with NaN in the copied DataFrame
    df_copy[is_outlier] = np.nan

    return df_copy
    
def analyze_dataframe(df):
    # Step 1: Filter numeric columns
    numeric_columns = df.select_dtypes(include=[np.number])

    # Step 2: Detect outliers using the IQR method
    Q1 = numeric_columns.quantile(0.25)
    Q3 = numeric_columns.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((numeric_columns < lower_bound) | (numeric_columns > upper_bound))
    
    # Step 3: Count NaNs, zeros, and infs
    nan_count = numeric_columns.isna().sum()
    zero_count = (numeric_columns == 0).sum()
    inf_count = np.isinf(numeric_columns).sum()

    # Step 4: Create a DataFrame to store the results
    result = pd.DataFrame({
        'Column': numeric_columns.columns,
        'Outliers': outliers.sum(),
        'NaNs': nan_count,
        'Zeros': zero_count,
        'Infs': inf_count
    })

    # Step 5: Return the result DataFrame
    return result

def modified_z_score(series):
    # Calculate the median of the series
    median = series.median()

    # Calculate the median absolute deviation (MAD)
    mad = np.abs(series - median).median()

    # Check if MAD is zero to avoid division by zero
    if mad == 0:
        return [0.0] * len(series)

    # Calculate the modified z-score for each element
    modified_z_scores = 0.6745 * (series - median) / mad

    return modified_z_scores

from scipy.stats import zscore

def smooth_weights(weight):
    """
    Replace values in the input weight DataFrame that have z-scores greater than 3
    with the mean plus 3 times the standard deviation, and values with z-scores less
    than -3 with the mean minus 3 times the standard deviation.

    Parameters:
    - weight (pd.DataFrame): Input DataFrame containing weight values.

    Returns:
    pd.DataFrame: DataFrame with outliers replaced based on z-scores.
    """
    def replace_zscore_greater_than_3(row):
        temp = row.copy()
        z_scores = zscore(row)
        high = row.mean() + 3 * row.std()
        low = row.mean() - 3 * row.std()
        temp[z_scores > 3] = high
        temp[z_scores < -3] = low
        return temp

    return weight.apply(replace_zscore_greater_than_3, axis=1)


def VMP(ret_D, ret_period, how="std", period="W", market=False, target_ann_vol=None, vol_model=None, lev_limit=None, daily=False):
    """
    Calculate the Volatility Managed Portfolio (VMP).

    Parameters:
    - ret_D (DataFrame): Daily returns of assets.
    - ret_period (Series): Returns of the portfolio over a specific period.
    - how (str): Method for calculating volatility ("std", "var", "up-std", "down-std", "abs-vol").
    - period (str): Resampling period for volatility calculation.
    - market (bool): Flag indicating whether to use market returns for non-market assets.
    - target_ann_vol (float): Target annualized volatility.
    - vol_model: Placeholder for volatility model (not used in the current implementation).
    - lev_limit: Placeholder for leverage limit (not used in the current implementation).
    - daily (bool): Flag indicating whether to calculate daily weights and returns.

    Returns:
    - VMP (Series): Volatility Managed Portfolio returns.
    """

    ret_D = ret_D.copy()

    if market:
        ret_D.loc[:, ret_D.columns != 'CMKT'] = ret_D['CMKT'].to_numpy()[:, None].repeat(11, axis=1)

    ret_D = ret_D.replace(0, np.nan)
    c0 = 0.01

    if how == "std":
        vol = ret_D.resample(period).std().shift(1)
    elif how == "var":
        vol = ret_D.resample(period).var().shift(1)
    elif how == "up-std":
        temp = ret_D[ret_D > 0]
        vol = temp.resample(period).std().shift(1)
    elif how == "down-std":
        temp = ret_D[ret_D < 0]
        vol = temp.resample(period).std().shift(1)
    elif how == "abs-vol":
        vol = ((np.abs(ret_D - ret_D.resample(period).transform("mean"))).resample(period).mean()).shift(1)

    VMP = (c0 / vol) * ret_period
    vol_BH = ret_period.std()
    vol_VMP = VMP.std()
    c = c0 * vol_BH / vol_VMP
    w = c / vol
    VMP = w * ret_period

    if daily:
        w = w.resample("D").bfill()
        VMP = w * ret_D

    return VMP


def factor_portfolio(factor, marketCap, volume, ret_W, ret_D, min_marketCap=1e6, q=5):
    # Filter 'marketCap' data to align with 'factor'
    marketCap_filtered = marketCap.where(marketCap >= min_marketCap, np.nan) / 1e5
    
    # Remove coins not traded during portfolio formation week
    factor = factor.where(np.log(marketCap_filtered).notna(), np.nan)
    factor = factor.where(np.log(volume).notna(), np.nan)
    factor = factor.astype(float)
    
    # Assign portfolio number based on quantiles of each row in 'factor'
    portfo_number = factor.apply(lambda row: pd.qcut(row, q, labels=False), axis=1)
    
    weekly_returns = pd.DataFrame()
    daily_returns = pd.DataFrame()
    
    # Calculate weighted returns for each portfolio
    for i in range(q):
        temp = marketCap_filtered.where(portfo_number == i, np.nan)
        weight = temp.div(temp.sum(axis=1), axis=0).fillna(0)
        
        # Calculate weekly returns based on portfolio weights
        weekly_returns[i + 1] = (weight.shift(1) * ret_W).sum(axis=1)#.clip(lower=-1)
        
        # Calculate daily returns based on portfolio weights
        daily_returns[i + 1] = (weight.shift(1).resample("D").bfill(limit=6) * ret_D).sum(axis=1)#.clip(lower=-1)
    
    return weekly_returns, daily_returns


def vol_models_stat(ret_D, ret, RF_D, period="W", *args, **kwargs):
    # List of volatility models
    Models = ["std", "var", "down-std", "up-std", "abs-vol"]
    
    # Initialize an empty DataFrame for results
    result = pd.DataFrame()
    
    # Iterate over each volatility model
    for Model in Models:
        # Calculate financial statistics using the specified volatility model
        temp = (calculate_financial_statistics(VMP(ret_D, ret, how=Model, period=period), RF_D) - calculate_financial_statistics(ret, RF_D)).loc[:, ['Annualized Mean Excess Returns', 'Sharpe Ratio']]
        
        # Set multi-level column names based on the volatility model
        temp.columns = pd.MultiIndex.from_tuples([(Model, col) for col in temp.columns])
        
        # Concatenate the results to the overall DataFrame
        result = pd.concat([result, temp], axis=1)
    
    # Return the final result DataFrame
    return result


def calculate_financial_statistics(df_returns, df_risk_free):
    """
    Calculate financial statistics for each factor in a DataFrame of historical returns.

    Parameters:
        df_returns (pd.DataFrame): DataFrame containing historical returns with a date index and factor columns.
        df_risk_free (pd.DataFrame): DataFrame containing historical risk-free rates with a date index.

    Returns:
        pd.DataFrame: DataFrame of financial statistics for each factor.
    """

    t_stats, p_values = stats.ttest_1samp(df_returns, 0, nan_policy='omit')
    
    # Ensure that both DataFrames have a datetime index
    df_returns.index = pd.to_datetime(df_returns.index)
    df_risk_free.index = pd.to_datetime(df_risk_free.index)

    # Calculate mean returns and standard deviation for df_returns
    mean_returns = df_returns.mean()
    std_deviation = df_returns.std()

    # Infer the annualization factor from the frequency of the df_returns index
    freq = pd.infer_freq(df_returns.index)
    if freq is not None:
        if freq == 'D':
            annualize_factor = 365  # Assuming daily data
        elif freq == 'W' or freq == 'W-SUN':
            annualize_factor = 52  # Assuming weekly data
            df_risk_free = (df_risk_free + 1).resample("W").apply(np.prod) - 1
        elif freq == 'M':
            annualize_factor = 12  # Assuming monthly data
            df_risk_free = (df_risk_free + 1).resample("M").apply(np.prod) - 1
        else:
            raise ValueError("Unsupported frequency for inferring the annualization factor.")
    else:
        raise ValueError("Unable to infer the frequency from the df_returns index.")

    # Interpolate risk-free rates to match the dates in df_returns
    interpolated_risk_free = df_risk_free.reindex(df_returns.index).interpolate()

    # Subtract risk-free rate from all columns except "CMKT"
    excess_returns = df_returns  # .sub(interpolated_risk_free['RF'], axis=0)

    if "CMKT" in excess_returns.columns:
        # Restore original values for "CMKT" in the 'excess_returns' DataFrame
        excess_returns.loc[:, "CMKT"] = df_returns["CMKT"]

    # Calculate the mean excess returns
    mean_excess_returns = excess_returns.mean()

    # Calculate annualized returns and volatility
    annualized_mean_excess_returns = mean_excess_returns * annualize_factor
    annualized_volatility = std_deviation * np.sqrt(annualize_factor)

    # Calculate mean excess returns and Sharpe ratio
    sharpe_ratio = (annualized_mean_excess_returns / annualized_volatility).replace([-np.inf, np.inf], np.nan)
    
    # Create the summary DataFrame
    financial_stats_df = pd.DataFrame({
        'Mean Returns': mean_returns,
        'Mean Excess Returns': mean_excess_returns,
        't_stat': t_stats,
        'p_value': p_values, 
        'Annualized Mean Excess Returns': annualized_mean_excess_returns,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio
    })

    return financial_stats_df


def jobson_korkie_test(ret_1, ret_2):
    # Calculate means
    mean1 = np.mean(ret_1)
    mean2 = np.mean(ret_2)

    # Calculate standard deviations
    std1 = np.std(ret_1)
    std2 = np.std(ret_2)

    # Calculate covariance
    cov = ret_1.cov(ret_2)

    # Calculate number of observations
    T = len(ret_1)

    # Calculate Theta
    Theta = 1/T * ((2*(std1**2)*(std2**2)) - (2*std1*std2*cov) + (1/2*(mean1*std2)**2) + (1/2*(mean2*std1)**2) - ((mean1*mean2*cov**2)/(std1*std2)))

    # Calculate test statistic
    JK = (mean1*std2 - mean2*std1) / np.sqrt(Theta)

    # Calculate p-value
    p_value = 2 * (1 - norm.cdf(np.abs(JK)))

    return p_value

# Example usage:
# Assuming you have two DataFrames, 'df_returns' for historical returns and 'df_risk_free' for risk-free rates,
# p_value = jobson_korkie_test(df_returns, df_risk_free)
# print(p_value)

def utility_gain(df1, df2):
    return np.square((df1["Sharpe Ratio"])/(df2["Sharpe Ratio"]))-1

def regress_vmp_factors(Factors, VMP_Factors, annualization_factor=52):
    # Define a function to perform linear regression for a single column
    def regression(column):
        y = VMP_Factors[column]
        X = Factors[column]

        # Drop rows with missing values in both X and y
        data = pd.concat([X, y], axis=1).dropna()

        if data.empty:
            # If there are no valid data points, return NaN for all statistics
            return pd.Series({
                'alpha_original': None, 'beta_original': None,
                'alpha_annualized': None, 'beta_annualized': None,
                'alpha_tstat': None, 'beta_tstat': None
            })

        X = sm.add_constant(data.iloc[:, 0])
        model = sm.OLS(data.iloc[:, 1], X).fit()
        
        # Extract alpha, beta, alpha t-statistic, and beta t-statistic
        alpha_original = model.params['const']
        beta_original = model.params[column]
        alpha_tstat = model.tvalues['const']
        beta_tstat = model.tvalues[column]

        # Calculate annualized alpha and beta
        alpha_annualized = alpha_original * annualization_factor
        beta_annualized = beta_original * annualization_factor

        return pd.Series({
            'alpha_original': alpha_original, 'beta_original': beta_original,
            'alpha_annualized': alpha_annualized, 'beta_annualized': beta_annualized,
            'alpha_tstat': alpha_tstat, 'beta_tstat': beta_tstat
        })

    # Apply the regression function to each column in VMP_Factors
    results = VMP_Factors.columns.to_series().apply(regression)

    return results

# Example usage:
# Assuming you have two DataFrames Factors and VMP_Factors
# results_df = regress_vmp_factors(Factors, VMP_Factors)
# print(results_df)

def plot_optimal_risk_exposure_by_volatility(data, data_W):
    """
    Calculate weekly realized volatility and plot the average optimal risk exposure 
    by volatility bucket for each column in the input DataFrame.

    Args:
        data (pandas.DataFrame): A DataFrame with a datetime index and multiple columns of daily returns.
        data_W (pandas.DataFrame): A DataFrame with a datetime index and corresponding weekly returns.

    Returns:
        None
    """
    # Create a new DataFrame to store the results
    result_df = pd.DataFrame()

    # Iterate over each column in the input DataFrame
    for column in data.columns:
        # Calculate weekly realized volatility
        weekly_volatility = data[column].resample('W').std()

        # Create a new DataFrame for the current column
        col_result_df = pd.DataFrame({'return': data[column], 'weekly_volatility': weekly_volatility.shift(1)})

        # Sort returns into five buckets based on volatility
        col_result_df['volatility_bucket'] = pd.qcut(col_result_df['weekly_volatility'], 5, labels=False)

        # Create a temporary DataFrame for resampling
        temp = pd.DataFrame()

        # Resample returns and volatility to weekly frequency
        temp["optimal_risk_exposure"] = data_W[column] / (col_result_df["return"].resample("W").var())
        temp["volatility_bucket"] = col_result_df["volatility_bucket"].resample("W").last()
        
        # Calculate the average optimal risk exposure of the following week's returns for each volatility bucket
        avg_optimal_risk_by_bucket = temp.groupby('volatility_bucket')['optimal_risk_exposure'].mean()

        # Add the results to the overall result DataFrame
        result_df[column] = avg_optimal_risk_by_bucket

    # Create a figure with three columns, each corresponding to a DataFrame column
    num_columns = len(data.columns)
    num_rows = int(np.ceil(num_columns / 3))
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))

    # Flatten the axes array if there's only one row
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    # Plot each column in a separate subplot
    for i, column in enumerate(result_df.columns):
        row_idx = i // 3
        col_idx = i % 3
        ax = axes[row_idx, col_idx]
        result_df[column].plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
        ax.set_xlabel('Volatility Bucket')
        ax.set_ylabel('Average Optimal Risk Exposure')
        ax.set_title(f'Average Optimal Risk Exposure by Volatility Bucket ({column})')
        ax.set_xticks(range(5))
        ax.set_xticklabels([f'Bucket {j+1}' for j in range(5)], rotation=0)

    # Remove empty subplots, if any
    for i in range(len(result_df.columns), num_rows * 3):
        fig.delaxes(axes[i // 3, i % 3])

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming you have a DataFrame called 'returns_df' with a datetime index and multiple columns of returns
# plot_optimal_risk_exposure_by_volatility(returns_df, weekly_returns_df)


def calculate_cosine_similarity(df1, df2):
    """
    Calculate the cosine similarity between corresponding columns of two DataFrames, considering missing values.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame.
    df2 (pd.DataFrame): The second DataFrame.

    Returns:
    pd.Series: A Series containing the cosine similarities for each column pair.
    """
    if not all(df1.columns == df2.columns):
        raise ValueError("Both DataFrames must have the same column names")

    def cosine_sim(column):
        # Extract the vectors for the current column from both DataFrames
        vector1 = df1[column]
        vector2 = df2[column]
        
        # Exclude rows with missing values
        valid_indices = ~(vector1.isnull() | vector2.isnull())
        vector1 = vector1[valid_indices]
        vector2 = vector2[valid_indices]
        
        # Calculate cosine similarity using NumPy
        similarity = cosine_similarity(vector1.values.reshape(1, -1), vector2.values.reshape(1, -1))[0][0]
        return similarity

    # Calculate cosine similarity for each column pair
    cos_similarity = df1.columns.to_series().apply(cosine_sim)
    
    return cos_similarity

    
def calculate_optimality_score(factors, original_factors, original_factors_D):
    # Calculate the weight based on current factors and original factors
    weight = factors / original_factors
    
    # Calculate the optimal weight using the resampled variance of original factors over weeks
    optimal_weight = original_factors / original_factors_D.resample("W").var()
    
    # Return the cosine similarity between weight and optimal weight
    return calculate_cosine_similarity(weight, optimal_weight)
