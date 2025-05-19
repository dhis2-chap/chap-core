import numpy as np
import pandas as pd
from dateutil import parser
import json
import re
import os


def predict_malaria_affected_simple(temp, precip, pop):
    alpha = 0.5

    # Temperature score: peak at 27C, quadratic drop-off
    t_score = max(0, 1 - ((temp - 27) / 7) ** 2)

    # Precipitation score: trapezoidal
    if precip < 40 or precip > 300:
        p_score = 0
    elif precip < 80:
        p_score = (precip - 40) / 40
    elif precip <= 200:
        p_score = 1
    else:
        p_score = (300 - precip) / 100

    # Combine into risk proportion
    risk_prop = min(1, t_score * p_score * alpha)

    # Log scale
    #risk_prop = np.log(risk_prop) if risk_prop else risk_prop

    # Estimate cases
    estimated_cases = np.log10(pop * risk_prop)
    print('->', risk_prop, estimated_cases, pop)

    return estimated_cases


def predict_malaria_affected_best(temp, precip, pop):
    alpha = 1e-5
    beta = 1.0
    gamma = 0.15
    delta =- 0.005
    epsilon = 0.01
    noise_std = 0.2
    log_m = (np.log(alpha) +
             beta * np.log(pop) +
             gamma * temp +
             delta * temp**2 +
             epsilon * precip +
             np.random.normal(0, noise_std))
    malaria_cases = np.exp(log_m)
    return malaria_cases


def generate_synthetic_malaria_data(orgunits, periods, temps, precips, pops,
                                    random_seed=16):
    """
    Generate a synthetic malaria dataset.

    Parameters:
        orgunits (list): List of org unit IDs or names.
        periods (list): List of period identifiers (e.g., month, week).
        temps (dict): Dict mapping (orgunit, period) -> temperature in Celsius.
        precips (dict): Dict mapping (orgunit, period) -> precipitation in mm.
        pops (dict): Dict mapping orgunit -> population.
        alpha, beta, gamma, delta, epsilon: Model parameters.
        noise_std (float): Std dev of Gaussian noise added to log malaria cases.
        random_seed (int or None): Seed for reproducibility.

    Returns:
        pandas.DataFrame: Synthetic malaria dataset.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    data = []

    for org in sorted(orgunits):
        for period in sorted(periods):
            key = (org, period)
            temp = float(temps[key].replace(',','.'))
            precip = float(precips[key].replace(',','.'))
            pop = float(pops[key])

            # predict value at this timestep
            #malaria_cases = predict_malaria_affected_simple(temp, precip, pop)
            malaria_cases = predict_malaria_affected_best(temp, precip, pop)

            # ensure non-negative
            malaria_cases = max(0, np.round(malaria_cases))

            data.append({
                "orgUnit": org,
                "period": period,
                "temperature": temp,
                "precipitation": precip,
                "population": pop,
                "malaria_cases": int(malaria_cases)
            })

    return pd.DataFrame(data)


def convert_to_dhis2_long_format(df, data_element_map):
    """
    Converts a wide-format DataFrame to long-format for DHIS2 import.
    
    Parameters:
        df (pd.DataFrame): Input data. Must include 'orgUnit' and 'period' columns.
        data_element_map (dict): Dictionary mapping column names (e.g., 'temperature')
                                 to DHIS2 dataElement UIDs.
                                 Example: {'temperature': 'abc123', 'malaria_cases': 'def456'}
    
    Returns:
        pd.DataFrame: Long-format DHIS2-compatible DataFrame.
    """
    if 'orgUnit' not in df.columns or 'period' not in df.columns:
        raise ValueError("Input DataFrame must contain 'orgUnit' and 'period' columns.")
    
    # Keep only fields in with data element ids
    df = df[['orgUnit', 'period'] + list(data_element_map.keys())]

    # Convert to long format
    df_long = df.melt(id_vars=['orgUnit', 'period'], 
                      value_vars=list(data_element_map.keys()),
                      var_name='variable',
                      value_name='value')
    
    # Map variables to dataElement IDs
    df_long['dataElement'] = df_long['variable'].map(data_element_map)
    
    # Reorder and clean
    df_long = df_long[['dataElement', 'period', 'orgUnit', 'value']]
    
    return df_long


def convert_to_dhis2_period(period_str, period_type='monthly'):
    """
    Converts human-readable period strings to DHIS2 format.

    Parameters:
        period_str (str): A human-readable date or period string.
        period_type (str): One of 'monthly', 'quarterly', 'yearly', 'weekly', 'daily'.

    Returns:
        str: DHIS2-formatted period string.
    """
    period_type = period_type.lower()

    # Handle custom formats
    if period_type == 'quarterly':
        match = re.match(r'(\d{4})[- ]?Q([1-4])', period_str.upper())
        if match:
            return f"{match.group(1)}Q{match.group(2)}"
        else:
            raise ValueError(f"Invalid quarterly format: {period_str}")

    elif period_type == 'weekly':
        match = re.match(r'(\d{4})[- ]?W(\d{1,2})', period_str.upper())
        if match:
            return f"{match.group(1)}W{int(match.group(2))}"
        else:
            dt = parser.parse(period_str)
            iso_year, iso_week, _ = dt.isocalendar()
            return f"{iso_year}W{iso_week}"

    elif period_type == 'monthly':
        dt = parser.parse(period_str)
        return dt.strftime("%Y%m")

    elif period_type == 'yearly':
        dt = parser.parse(period_str)
        return dt.strftime("%Y")

    elif period_type == 'daily':
        dt = parser.parse(period_str)
        return dt.strftime("%Y%m%d")

    else:
        raise ValueError(f"Unsupported period type: {period_type}")


if __name__ == '__main__':
    output = r'C:\Users\karimba\Documents\Github\chap-core\scripts\ghana_simulated_malaria'
    data_element_map = {
        'malaria_cases': 'bpfrQkyWG4i',
        'population': 'Pi3zfVY962v',
    }

    # load
    path = r'C:\Users\karimba\Documents\Github\chap-core\scripts\ghana_climate_data.csv'
    path_base = os.path.splitext(path)[0]
    df = pd.read_csv(path, sep=';')
    print(df)

    # join with pop
    path = r'C:\Users\karimba\Documents\Github\chap-core\scripts\ghana_pop_data.csv'
    pop_df = pd.read_csv(path, sep=';')
    df['Total population'] = df.merge(
        pop_df, on='organisationunitname', how='left'
    )['Total population']
    print(df)

    # standardize to dhis2 conventions
    df['orgUnit'] = df['organisationunitname']
    df['period'] = df['periodname'].map(lambda v: convert_to_dhis2_period(v, 'monthly'))

    # extract values
    orgunits = list(df['orgUnit'].unique())
    periods = list(df['period'].unique())
    temps = {
        (row['orgUnit'],row['period']): row['Air temperature (ERA5-Land)']
        for row in df.to_dict(orient="records")
    }
    precips = {
        (row['orgUnit'],row['period']): row['Precipitation (ERA5-Land)']
        for row in df.to_dict(orient="records")
    }
    pops = {
        (row['orgUnit'],row['period']): row['Total population']
        for row in df.to_dict(orient="records")
    }

    # simulate
    sim = generate_synthetic_malaria_data(orgunits, periods, temps, precips, pops=pops)
    print(sim)

    # cpmvert to dhis2 long format
    sim_dhis2 = convert_to_dhis2_long_format(sim, data_element_map)
    print(sim_dhis2)
    
    # save to dhis2 format csv
    # sim_dhis2.to_csv(f'{output}.csv', index=False)

    # save to dhis2 format json
    dhis2_json = {'dataValues': sim_dhis2.to_dict(orient='records')}
    print(json.dumps(dhis2_json, indent=2))
    with open(f'{output}.json', "w", encoding='utf8') as f:
        json.dump(dhis2_json, f, indent=2)
