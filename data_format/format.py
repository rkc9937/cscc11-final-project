import pandas as pd
import pickle

# Month string to integer mapping
MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}


def load_severity_weights(weights_path='./data/weights/severity_weights.pkl'):
    """Load pre-computed severity weights from pickle file."""
    with open(weights_path, 'rb') as f:
        return pickle.load(f)


def get_prev_month(year, month):
    """Get previous month's year and month."""
    if month == 1:
        return year - 1, 12
    return year, month - 1


def compute_monthly_crime_counts(df, year, month):
    """
    Compute crime counts per category for a given month and neighborhood.
    Uses OCC_YEAR (int) and OCC_MONTH (int) columns.
    """
    monthly = df[(df['OCC_YEAR'] == year) & (df['OCC_MONTH'] == month)]
    
    if len(monthly) == 0:
        return pd.DataFrame()
    
    crime_counts = monthly.groupby(['HOOD_158', 'MCI_CATEGORY']).size().unstack(fill_value=0)
    
    # Ensure all crime categories exist
    for cat in ['Assault', 'Break and Enter', 'Theft Over', 'Auto Theft', 'Robbery']:
        if cat not in crime_counts.columns:
            crime_counts[cat] = 0
    
    crime_counts = crime_counts.rename(columns={
        'Assault': 'count_assault',
        'Break and Enter': 'count_bne',
        'Theft Over': 'count_theft',
        'Auto Theft': 'count_auto_theft',
        'Robbery': 'count_robbery'
    })
    
    return crime_counts[['count_assault', 'count_bne', 'count_theft', 'count_auto_theft', 'count_robbery']]


def compute_weighted_score(crime_counts, severity_weights):
    """Compute weighted severity score for each neighborhood."""
    weight_map = {
        'count_assault': severity_weights['Assault'],
        'count_bne': severity_weights['Break and Enter'],
        'count_theft': severity_weights['Theft Over'],
        'count_auto_theft': severity_weights['Auto Theft'],
        'count_robbery': severity_weights['Robbery']
    }
    
    weighted = sum(crime_counts[col] * weight_map[col] for col in weight_map.keys())
    return weighted


def compute_nsi(crime_counts, severity_weights):
    """
    Compute the Neighborhood Safety Index (NSI).
    
    1. TotalCrimeScore = sum(count_i * weight_i) for all crime categories
    2. NSI = 1 - (TotalCrimeScore - MinScore) / (MaxScore - MinScore)
    
    Result is 0-1 range where higher NSI = safer neighborhood.
    """
    weighted_score = compute_weighted_score(crime_counts, severity_weights)
    
    min_score = weighted_score.min()
    max_score = weighted_score.max()
    
    # Avoid division by zero if all scores are equal
    if max_score == min_score:
        return pd.Series(1.0, index=weighted_score.index)
    
    # Normalize and invert: higher NSI = safer
    nsi = 1 - (weighted_score - min_score) / (max_score - min_score)
    return nsi


def format_dataframe(df, weights_path='./data/weights/severity_weights.pkl'):
    """
    Format MCI dataframe with features for crime prediction.
    
    Input columns from mci.csv:
    - OCC_YEAR (int): Year of occurrence (e.g., 2014)
    - OCC_MONTH (str): Month of occurrence (e.g., 'January')
    - REPORT_YEAR (int): Year reported
    - REPORT_MONTH (str): Month reported
    - HOOD_158: Neighborhood ID
    - LAT_WGS84, LONG_WGS84: Coordinates
    - MCI_CATEGORY: Crime category
    
    Output features:
    - Spatial: neighborhood_id, lat_wgs84, long_wgs84
    - Temporal: year, month
    - Crime counts (prev month): count_*_prev
    - Severity: weighted_score_prev
    - Lagged: NSI_prev, NSI_prev_1, delta_NSI, NSI_3M, delta_weight, weight_3m
    - Target: NSI_next
    """
    severity_weights = load_severity_weights(weights_path)
    
    df = df.copy()
    
    # Drop rows with missing critical values
    initial_len = len(df)
    df = df.dropna(subset=['OCC_YEAR', 'OCC_MONTH', 'LAT_WGS84', 'LONG_WGS84', 'HOOD_158', 'MCI_CATEGORY'])
    print(f"Dropped {initial_len - len(df)} rows with missing values")
    
    # Drop rows with NSA neighborhood (equivalent to NaN)
    nsa_count = (df['HOOD_158'] == 'NSA').sum()
    df = df[df['HOOD_158'] != 'NSA']
    print(f"Dropped {nsa_count} rows with NSA neighborhood")
    
    # Convert year columns to int, month strings to int
    df['OCC_YEAR'] = df['OCC_YEAR'].astype(int)
    df['OCC_MONTH'] = df['OCC_MONTH'].map(MONTH_MAP)
    df['REPORT_YEAR'] = df['REPORT_YEAR'].astype(int)
    df['REPORT_MONTH'] = df['REPORT_MONTH'].map(MONTH_MAP)
    
    # Get unique year-month combinations, sorted
    unique_periods = df[['OCC_YEAR', 'OCC_MONTH']].drop_duplicates()
    unique_periods = unique_periods.sort_values(['OCC_YEAR', 'OCC_MONTH'])
    
    print(f"Data spans from {unique_periods.iloc[0]['OCC_YEAR']}-{unique_periods.iloc[0]['OCC_MONTH']} "
          f"to {unique_periods.iloc[-1]['OCC_YEAR']}-{unique_periods.iloc[-1]['OCC_MONTH']}")
    
    # Pre-compute monthly crime stats for all months
    print("Pre-computing monthly crime statistics...")
    monthly_stats = {}
    for _, row in unique_periods.iterrows():
        year, month = int(row['OCC_YEAR']), int(row['OCC_MONTH'])
        crime_counts = compute_monthly_crime_counts(df, year, month)
        
        if len(crime_counts) > 0:
            weighted_scores = compute_weighted_score(crime_counts, severity_weights)
            nsi_values = compute_nsi(crime_counts, severity_weights)
            monthly_stats[(year, month)] = {
                'crime_counts': crime_counts,
                'weighted_score': weighted_scores,
                'nsi': nsi_values
            }
    
    print(f"Computed stats for {len(monthly_stats)} months")
    
    # Build feature dataset at neighborhood-month level
    records = []
    
    for _, row in unique_periods.iterrows():
        year, month = int(row['OCC_YEAR']), int(row['OCC_MONTH'])
        
        # Get previous 3 months
        prev_y1, prev_m1 = get_prev_month(year, month)
        prev_y2, prev_m2 = get_prev_month(prev_y1, prev_m1)
        prev_y3, prev_m3 = get_prev_month(prev_y2, prev_m2)
        
        # Skip if we don't have 3 months of history
        if (prev_y1, prev_m1) not in monthly_stats:
            continue
        if (prev_y2, prev_m2) not in monthly_stats:
            continue
        if (prev_y3, prev_m3) not in monthly_stats:
            continue
        
        # Get current month data
        current_month_df = df[(df['OCC_YEAR'] == year) & (df['OCC_MONTH'] == month)]
        
        # Aggregate per neighborhood for this month
        hood_agg = current_month_df.groupby('HOOD_158').agg({
            'LAT_WGS84': 'mean',
            'LONG_WGS84': 'mean',
            'REPORT_YEAR': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
            'REPORT_MONTH': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
        })
        
        # Current month stats for target (NSI_next)
        current_stats = monthly_stats.get((year, month))
        
        for hood_id in hood_agg.index:
            record = {
                'neighborhood_id': hood_id,
                'lat_wgs84': hood_agg.loc[hood_id, 'LAT_WGS84'],
                'long_wgs84': hood_agg.loc[hood_id, 'LONG_WGS84'],
                'occ_year': year,
                'occ_month': month,
                'report_year': int(hood_agg.loc[hood_id, 'REPORT_YEAR']),
                'report_month': int(hood_agg.loc[hood_id, 'REPORT_MONTH']),
            }
            
            # Previous month stats (t-1)
            prev_stats_1 = monthly_stats[(prev_y1, prev_m1)]
            if hood_id in prev_stats_1['crime_counts'].index:
                for col in ['count_assault', 'count_bne', 'count_theft', 'count_auto_theft', 'count_robbery']:
                    record[col + '_prev'] = prev_stats_1['crime_counts'].loc[hood_id, col]
                record['weighted_score_prev'] = prev_stats_1['weighted_score'].get(hood_id, 0)
                record['NSI_prev'] = prev_stats_1['nsi'].get(hood_id, 0)
            else:
                for col in ['count_assault', 'count_bne', 'count_theft', 'count_auto_theft', 'count_robbery']:
                    record[col + '_prev'] = 0
                record['weighted_score_prev'] = 0
                record['NSI_prev'] = 0
            
            # t-2 month stats
            prev_stats_2 = monthly_stats[(prev_y2, prev_m2)]
            if hood_id in prev_stats_2['nsi'].index:
                record['NSI_prev_1'] = prev_stats_2['nsi'].get(hood_id, 0)
                record['weighted_score_prev_1'] = prev_stats_2['weighted_score'].get(hood_id, 0)
            else:
                record['NSI_prev_1'] = 0
                record['weighted_score_prev_1'] = 0
            
            # t-3 month stats
            prev_stats_3 = monthly_stats[(prev_y3, prev_m3)]
            if hood_id in prev_stats_3['nsi'].index:
                nsi_3 = prev_stats_3['nsi'].get(hood_id, 0)
                weight_3 = prev_stats_3['weighted_score'].get(hood_id, 0)
            else:
                nsi_3 = 0
                weight_3 = 0
            
            # Lagged features
            record['delta_NSI'] = record['NSI_prev'] - record['NSI_prev_1']
            record['NSI_3M'] = (record['NSI_prev'] + record['NSI_prev_1'] + nsi_3) / 3
            record['delta_weight'] = record['weighted_score_prev'] - record['weighted_score_prev_1']
            record['weight_3m'] = (record['weighted_score_prev'] + record['weighted_score_prev_1'] + weight_3) / 3
            
            # Target: NSI for current month
            if current_stats and hood_id in current_stats['nsi'].index:
                record['NSI_next'] = current_stats['nsi'].get(hood_id, 0)
            else:
                record['NSI_next'] = 0
            
            records.append(record)
    
    print(f"Generated {len(records)} records")
    
    # Create DataFrame
    result_df = pd.DataFrame(records)
    
    # Reorder columns
    spatial_cols = ['neighborhood_id', 'lat_wgs84', 'long_wgs84']
    temporal_cols = ['occ_year', 'occ_month', 'report_year', 'report_month']
    crime_cols = ['count_assault_prev', 'count_bne_prev', 'count_theft_prev', 
                  'count_auto_theft_prev', 'count_robbery_prev']
    weight_cols = ['weighted_score_prev']
    lag_cols = ['NSI_prev', 'NSI_prev_1', 'delta_NSI', 'NSI_3M', 
                'delta_weight', 'weight_3m']
    target_cols = ['NSI_next']
    
    final_cols = spatial_cols + temporal_cols + crime_cols + weight_cols + lag_cols + target_cols
    result_df = result_df[final_cols]
    
    print("Formatting complete!")

    #convert neighborhood_id to numeric
    result_df['neighborhood_id'] = pd.to_numeric(result_df['neighborhood_id'], errors='coerce')

    return result_df


def save_formatted_data(df, output_path='./data/mci_formatted.pkl'):
    """Save formatted DataFrame to pickle file."""
    with open(output_path, 'wb') as f:
        pickle.dump(df, f)
    print(f"Saved formatted data to {output_path}")

def load_formatted_data(input_path='./data/mci_formatted.pkl'):
    """Load formatted DataFrame from pickle file."""
    with open(input_path, 'rb') as f:
        df = pickle.load(f)
    print(f"Loaded formatted data from {input_path}")
    return df
