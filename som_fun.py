from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import time
import pandas as pd
import numpy as np
from minisom import MiniSom
import streamlit as st
import altair as alt
import math
import itertools

# Global dictionary to store min/max values per feature and metric type
# Structure: {feature_name: {'single_value': {'min': val, 'max': val}, 'statistical': {'min': val, 'max': val}}}
feature_scale_ranges = {}

# Function to reset the global scale ranges


def reset_feature_scale_ranges():
    """
    Resets the global feature_scale_ranges dictionary.
    This should be called when loading a new SOM model or when starting a new session.
    """
    global feature_scale_ranges
    feature_scale_ranges = {}

# Function to precompute all scaling metrics for a feature


def precompute_feature_scale_ranges(feature_map, feature_name):
    """
    Precomputes all scaling metrics (min, max, mean, median, sum, std) for a feature
    and initializes the global scale ranges.

    Parameters:
    -----------
    feature_map : list
        The map of feature values produced by project_feature
    feature_name : str
        Name of the feature being analyzed
    """
    if feature_name is None or is_string(feature_map):
        return

    # Make a copy of the map to avoid modifying the original
    _map = list(map(list, zip(*feature_map)))

    # Initialize with extreme values
    single_value_min = float('inf')
    single_value_max = float('-inf')
    statistical_min = float('inf')
    statistical_max = float('-inf')

    # Also track the smallest non-zero value for log scales
    smallest_nonzero = float('inf')

    # Compute all scaling metrics
    for idx_outer, sublist_outer in enumerate(_map):
        for idx_inner, sublist in enumerate(sublist_outer):
            if sublist and sublist[0] is not None:
                try:
                    # Single value metrics
                    min_val = min(sublist)
                    max_val = max(sublist)
                    mean_val = np.mean(sublist)
                    median_val = np.median(sublist)

                    # Update single value ranges
                    single_value_min = min(
                        single_value_min, min_val, mean_val, median_val)
                    single_value_max = max(
                        single_value_max, max_val, mean_val, median_val)

                    # Statistical metrics
                    sum_val = sum(sublist)
                    std_val = np.std(sublist)

                    # Update statistical ranges
                    statistical_min = min(statistical_min, sum_val, std_val)
                    statistical_max = max(statistical_max, sum_val, std_val)

                    # Track smallest non-zero value for log scales
                    for val in sublist:
                        if val > 0:
                            smallest_nonzero = min(smallest_nonzero, val)

                except (TypeError, ValueError):
                    continue

    # If single_value_min is 0, use the smallest non-zero value for log scales
    if single_value_min == 0 and smallest_nonzero < float('inf'):
        # Store both the 0 for linear scales and smallest_nonzero for log scales
        log_single_value_min = smallest_nonzero
    else:
        log_single_value_min = single_value_min if single_value_min > 0 else 0.01

    # Same for statistical metrics
    if statistical_min == 0 and smallest_nonzero < float('inf'):
        log_statistical_min = smallest_nonzero
    else:
        log_statistical_min = statistical_min if statistical_min > 0 else 0.01

    # Initialize the global scale ranges
    if feature_name not in feature_scale_ranges:
        feature_scale_ranges[feature_name] = {
            'single_value': {
                'min': single_value_min,
                'max': single_value_max,
                'log_min': log_single_value_min  # Store the log-appropriate minimum
            },
            'statistical': {
                'min': statistical_min,
                'max': statistical_max,
                'log_min': log_statistical_min  # Store the log-appropriate minimum
            }
        }
    else:
        # Update existing ranges
        feature_scale_ranges[feature_name]['single_value']['min'] = min(
            feature_scale_ranges[feature_name]['single_value']['min'], single_value_min)
        feature_scale_ranges[feature_name]['single_value']['max'] = max(
            feature_scale_ranges[feature_name]['single_value']['max'], single_value_max)
        feature_scale_ranges[feature_name]['statistical']['min'] = min(
            feature_scale_ranges[feature_name]['statistical']['min'], statistical_min)
        feature_scale_ranges[feature_name]['statistical']['max'] = max(
            feature_scale_ranges[feature_name]['statistical']['max'], statistical_max)

        # Update log minimums
        if 'log_min' not in feature_scale_ranges[feature_name]['single_value']:
            feature_scale_ranges[feature_name]['single_value']['log_min'] = log_single_value_min
        else:
            feature_scale_ranges[feature_name]['single_value']['log_min'] = min(
                feature_scale_ranges[feature_name]['single_value']['log_min'], log_single_value_min)

        if 'log_min' not in feature_scale_ranges[feature_name]['statistical']:
            feature_scale_ranges[feature_name]['statistical']['log_min'] = log_statistical_min
        else:
            feature_scale_ranges[feature_name]['statistical']['log_min'] = min(
                feature_scale_ranges[feature_name]['statistical']['log_min'], log_statistical_min)

# Function to update or get the global min/max range for a feature and metric type


def update_feature_scale_range(feature_name, metric_type, min_val, max_val, color_type=None):
    """
    Updates the global scale range for a feature and metric type.

    Parameters:
    -----------
    feature_name : str
        Name of the feature being plotted
    metric_type : str
        Type of metric (either 'single_value' for min/median/max/mean or 'statistical' for sum/std)
    min_val : float
        Minimum value for the feature
    max_val : float
        Maximum value for the feature
    color_type : str, optional
        Type of color scale (e.g., 'log', 'linear'). Used to handle special cases for log scales.

    Returns:
    --------
    tuple
        Updated (min, max) range to use for visualization
    """
    # Initialize if feature not in dictionary
    if feature_name not in feature_scale_ranges:
        feature_scale_ranges[feature_name] = {
            'single_value': {'min': float('inf'), 'max': float('-inf')},
            'statistical': {'min': float('inf'), 'max': float('-inf')}
        }

    # Update the range
    current_min = feature_scale_ranges[feature_name][metric_type]['min']
    current_max = feature_scale_ranges[feature_name][metric_type]['max']

    feature_scale_ranges[feature_name][metric_type]['min'] = min(
        current_min, min_val)
    feature_scale_ranges[feature_name][metric_type]['max'] = max(
        current_max, max_val)

    return (feature_scale_ranges[feature_name][metric_type]['min'],
            feature_scale_ranges[feature_name][metric_type]['max'])


def is_string(var):
    is_string = True
    for sublist in var:
        for sublist2 in sublist:
            for value in sublist2:
                if value is not None and not isinstance(value, str) and not (isinstance(value, float) and np.isnan(value)):
                    is_string = False
                    return is_string
    return is_string


def train_som(X, x, y, input_len, sigma, learning_rate, train_iterations, topology, seed):
    # initialization
    som = MiniSom(x=x, y=y, input_len=input_len,
                  sigma=sigma, learning_rate=learning_rate, topology=topology, random_seed=seed)
    som.random_weights_init(X)
    # training
    # start time
    start = time.time()
    som.train_random(X, train_iterations)  # training with 100 iterations
    # end time
    end = time.time()
    # st.write('SOM trained in time:', end - start, 'seconds')
    return som


def build_iteration_indexes(data_len, num_iterations, random_generator=None):
    """Builds iteration indexes for training."""
    # Generate sequential indexes, looping over the dataset
    iterations = np.arange(num_iterations) % data_len

    # Shuffle if a random generator is provided
    if random_generator:
        random_generator.shuffle(iterations)

    return iterations


def train_and_track_error(som, data, num_iteration, progress_bar, steps, random_order=False):
    """
    Trains the SOM and tracks errors at 10 intervals.

    Parameters
    ----------
    som : MiniSom
        The Self-Organizing Map instance.

    data : np.array or list
        Data matrix.

    num_iteration : int
        Maximum number of iterations (one iteration per sample).

    random_order : bool (default=False)
        If True, samples are picked in random order.
        Otherwise, the samples are picked sequentially.

    Returns
    -------
    som : MiniSom
        The trained SOM.

    q_error : list
        List of quantization errors recorded at 10 intervals during training.
    """
    # Input validation
    som._check_iteration_number(num_iteration)
    som._check_input_len(data)

    # Initialize variables
    random_generator = None
    if random_order:
        random_generator = som._random_generator

    iterations = build_iteration_indexes(
        len(data), num_iteration, random_generator)
    q_error = []  # To store quantization errors
    t_error = []

    # Main training loop
    for t, iteration in enumerate(iterations):
        som.update(data[iteration], som.winner(
            data[iteration]), t, num_iteration)

        progress_fraction = (t + 1) / num_iteration
        progress_bar.progress(
            progress_fraction, text=f"Training... {t + 1}/{num_iteration} iterations"
        )

        # Record quantization error at steps intervals (including the last iteration)
        if t % (num_iteration // steps) == 0 or t == num_iteration - 1:
            q_error.append(som.quantization_error(data))
            if som.topology == "rectangular":
                t_error.append(som.topographic_error(data))
            elif som.topology == "hexagonal":
                t_error.append(topographic_error_hex(som, data))

    return som, q_error, t_error


def get_iterations_index(X, x, y, input_len, sigma, learning_rate, max_iter=1000, topology='rectangular', seed=None, steps=100, errors_bar=None):
    # Initialize RandomState for reproducibility
    rng = np.random.RandomState(seed)

    # Initialize SOM
    som = MiniSom(x=x, y=y, input_len=input_len, sigma=sigma,
                  learning_rate=learning_rate, topology=topology, random_seed=seed)
    som.random_weights_init(X)

    som, q_error, t_error = train_and_track_error(
        som, X, max_iter, errors_bar, steps, random_order=True)

    return som, q_error, t_error


def plot_errors(q_error, t_error, iterations, steps=100):
    st.write('## Quantization error and Topographic error')
    # Plot using st the quantization error and the topographic error together
    step_size = iterations // steps
    errors_data = pd.DataFrame(
        {'Iterations': range(0, iterations + 1, step_size),
         'Quantization error': q_error,
         'Topographic error': t_error})
    errors_data_melted = errors_data.melt(
        'Iterations', var_name='Errors', value_name='Value')

    c = alt.Chart(errors_data_melted).mark_line().encode(
        x=alt.X('Iterations', title='Iterations [-]',
                axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
        y=alt.Y(
            'Value', title='Error [-]', axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
        color=alt.Color('Errors', legend=alt.Legend(
            title="Errors", titleFontSize=14, labelFontSize=12))
    ).properties(
        width=600,
        height=400
    ).configure_legend(
        orient='top-right',
        padding=10,
        cornerRadius=5,
        fillColor='white',
        strokeColor='gray',
        labelFontSize=12,
        titleFontSize=14
    )

    st.altair_chart(c, use_container_width=True)


def get_dispersion(name_ids, id_to_pos, min_max_detections):
    dispersion_list = []
    for source_name, source_ids in name_ids.items():
        # Retrieve positions of the source's detections
        positions = [id_to_pos[id_] for id_ in source_ids if id_ in id_to_pos]

        if len(positions) < min_max_detections[0] or len(positions) > min_max_detections[1]:
            continue  # Skip if not enough valid detections

        # If there are fewer than 2 positions, dispersion is zero
        if len(positions) < 2:
            dispersion = 0
        else:
            pairwise_distances = []
            for (x1, y1), (x2, y2) in itertools.combinations(positions, 2):
                distance = math.hypot(x2 - x1, y2 - y1)
                pairwise_distances.append(distance)

            dispersion = np.round(
                sum(pairwise_distances) / len(pairwise_distances), 2)

        # Collect the dispersion metrics
        dispersion_list.append((source_name, dispersion, len(positions)))

    return dispersion_list


def get_hex_neighbors(i, j, max_i, max_j):
    """
    Returns the neighboring positions of a hexagon at position (i, j) in a grid of size (max_i, max_j).
    """
    # List to hold neighbor positions
    neighbors = []

    # Determine if the row is even or odd
    even = (j % 2 == 0)

    # Define neighbor offsets based on row parity
    if even:
        # Even rows
        neighbor_offsets = [
            (i - 1, j),     # Left
            (i, j + 1),  # Up-Left
            (i + 1, j + 1),     # Up-Right
            (i + 1, j),  # Right
            (i + 1, j - 1),     # Down-Right
            (i, j - 1),     # Down-Left
        ]
    else:
        # Odd rows
        neighbor_offsets = [
            (i - 1, j),  # Left
            (i - 1, j + 1),     # Up-Left
            (i, j + 1),     # Up-Right
            (i + 1, j),     # Right
            (i, j - 1),  # Down-Right
            (i - 1, j - 1),     # Down-Left
        ]

    # Filter out invalid neighbors (outside grid bounds)
    for ni, nj in neighbor_offsets:
        if 0 <= ni < max_i and 0 <= nj < max_j:
            neighbors.append((ni, nj))

    return neighbors


def detect_cluster_boundaries(cluster_mapping, som_shape):
    """
    Detects which neurons are at the perimeter/boundary of clusters.
    Only neurons at cluster boundaries will receive enhanced borders.

    Parameters:
    -----------
    cluster_mapping : dict
        Dictionary mapping (x, y) positions to cluster IDs
    som_shape : tuple
        Shape of the SOM grid (max_i, max_j)

    Returns:
    --------
    dict
        Dictionary mapping (x, y) positions to True if they are at cluster boundary
    """
    boundary_neurons = {}
    max_i, max_j = som_shape

    for (x, y), cluster_id in cluster_mapping.items():
        # Check if this neuron is at the boundary by examining its neighbors
        is_boundary = False

        # Get all possible neighbors for this position
        neighbors = get_hex_neighbors(x, y, max_i, max_j)

        # Check neighbors that exist and have cluster assignments
        for neighbor_pos in neighbors:
            neighbor_cluster = cluster_mapping.get(neighbor_pos, None)

            # If neighbor has different cluster ID, this is a boundary
            if neighbor_cluster is not None and neighbor_cluster != cluster_id:
                is_boundary = True
                break

        # Also check if this neuron is at the edge of the SOM grid
        # But only mark as boundary if it's actually at the edge and has some cluster neighbors
        is_at_edge = (x == 0 or x == max_i - 1 or y == 0 or y == max_j - 1)

        # For neurons at the grid edge, only mark as boundary if they have fewer neighbors
        # than expected (meaning they're truly at the perimeter)
        if is_at_edge:
            # For hexagonal grids, interior neurons have 6 neighbors
            # Edge neurons will have fewer neighbors
            expected_neighbors = 6
            actual_neighbors = len(neighbors)
            if actual_neighbors < expected_neighbors:
                is_boundary = True

        boundary_neurons[(x, y)] = is_boundary

    return boundary_neurons


def topographic_error_hex(som, data):
    """
    Computes the topographic error for the given MiniSom 'som' object and input 'data'.
    Supports both rectangular and hexagonal topologies. For hexagonal topology, uses the
    'get_hex_neighbors' function to determine adjacency.
    """
    # Ensure data length is correct
    som._check_input_len(data)

    total_neurons = np.prod(som._activation_map.shape)
    if total_neurons == 1:
        return np.nan

    # Compute best and second-best matching units
    b2mu_inds = np.argsort(som._distance_from_weights(data), axis=1)[:, :2]
    b2mu_xy = np.unravel_index(b2mu_inds, som._weights.shape[:2])
    b2mu_x, b2mu_y = b2mu_xy[0], b2mu_xy[1]

    rows, cols = som._weights.shape[:2]

    if som.topology == 'hexagonal':
        errors = 0
        for i in range(len(data)):
            bmu_x, bmu_y = b2mu_x[i, 0], b2mu_y[i, 0]    # best unit
            sbmu_x, sbmu_y = b2mu_x[i, 1], b2mu_y[i, 1]  # second-best unit

            neighbors = get_hex_neighbors(bmu_x, bmu_y, rows, cols)
            if (sbmu_x, sbmu_y) not in neighbors:
                errors += 1
        return errors / len(data)


def plot_rectangular_u_matrix(som, color_type='linear', color_scheme='lightmulti'):
    u_matrix = som.distance_map().T
    u_matrix = pd.DataFrame(
        u_matrix, columns=range(1, len(u_matrix)+1), index=range(1, len(u_matrix)+1))
    u_matrix = u_matrix.melt(
        var_name='x', value_name='value', ignore_index=False)
    u_matrix = u_matrix.reset_index()
    u_matrix = u_matrix.rename(columns={'index': 'y'})
    st.u_matrix = u_matrix

    # Check for negative values when using log scale
    if color_type == "log" and u_matrix['value'].min() < 0:
        st.error("Log scale cannot be applied to U-matrix with negative values.")
        return

    c = alt.Chart(u_matrix).mark_rect().encode(
        x=alt.X('x:O', title=''),
        y=alt.Y('y:O', sort=alt.EncodingSortField(
            'y', order='descending'), title=''),
        color=alt.Color(
            'value:Q', scale=alt.Scale(type=color_type, scheme=color_scheme),
            legend=alt.Legend(
                orient='bottom',
                direction='horizontal',
                gradientLength=615,
                gradientThickness=20
            )
        )
    ).properties(
        width=600,
        height=600
    )
    st.altair_chart(c, use_container_width=True)


dimensions = np.array([6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
sizes = np.array([48, 42, 37, 33, 29, 15, 10, 8, 6, 5, 5, 4, 4, 3.4])

new_dimensions = np.arange(6, 101, 1)
new_sizes = np.interp(new_dimensions, dimensions, sizes)


def plot_u_matrix_hex(som, color_type='linear', color_scheme='lightmulti'):
    u_matrix = som.distance_map().T
    som_shape = som.get_weights().shape
    u_matrix = pd.DataFrame(
        u_matrix, columns=range(1, len(u_matrix)+1), index=range(1, len(u_matrix)+1))
    u_matrix = u_matrix.melt(
        var_name='x', value_name='value', ignore_index=False)
    u_matrix = u_matrix.reset_index()
    u_matrix = u_matrix.rename(columns={'index': 'y'})

    min_x = u_matrix['x'].min()
    max_x = u_matrix['x'].max()
    min_y = u_matrix['y'].min()
    max_y = u_matrix['y'].max()

    min_value = u_matrix['value'][u_matrix['value'] > 0].min()
    max_value = u_matrix['value'].max()

    # Check for negative values when using log scale
    if color_type == "log" and u_matrix['value'].min() < 0:
        st.error("Log scale cannot be applied to U-matrix with negative values.")
        return

    # get index from new_dimensions
    index = np.where(new_dimensions == som_shape[0])[0][0]
    size = new_sizes[index]

    st.u_matrix = u_matrix
    hexagon = "M0,-2.3094010768L2,-1.1547005384 2,1.1547005384 0,2.3094010768 -2,1.1547005384 -2,-1.1547005384Z"
    c = alt.Chart(u_matrix).mark_point(shape=hexagon, size=size**2).encode(
        x=alt.X('xFeaturePos:Q', title='', scale=alt.Scale(
            domain=[min_x-1, max_x+1])).axis(grid=False, tickOpacity=0, domainOpacity=0),
        y=alt.Y('y:Q', sort=alt.EncodingSortField(
            'y', order='descending'), title='',  scale=alt.Scale(domain=[min_y-1, max_y+1])).axis(grid=False, labelPadding=20, tickOpacity=0, domainOpacity=0),
        color=alt.Color(
            'value:Q', scale=alt.Scale(scheme=color_scheme)),
        fill=alt.Fill('value:Q', scale=alt.Scale(type=color_type, domain=(
            min_value, max_value), scheme=color_scheme),
            legend=alt.Legend(
                orient='bottom',
                direction='horizontal',
                gradientLength=615,
                gradientThickness=20
        )
        ),
        stroke=alt.value('black'),
        strokeWidth=alt.value(1.0)
    ).transform_calculate(
        # This field is required for the hexagonal X-Offset
        xFeaturePos='(datum.y%2)/2 + datum.x-.5'
    ).properties(
        # width should be the same as the height
        height=700,
        width=600,
    ).configure_view(
        strokeWidth=0
    )
    st.altair_chart(c, use_container_width=True)


def download_activation_response(som, data):
    # return a list with the id for each winning row for each 2D map
    a_dict = {}
    for x in data:
        if som.winner(x[1:]) in a_dict:
            a_dict[som.winner(x[1:])].append(x[0])
        else:
            a_dict[som.winner(x[1:])] = [x[0]]
    return a_dict


def plot_activation_response(som, X_index, color_type='linear', color_scheme='lightmulti', plot=True):
    X = X_index[:, 1:]
    activation_map = som.activation_response(X)
    activation_map = pd.DataFrame(
        activation_map, columns=range(1, len(activation_map)+1), index=range(1, len(activation_map)+1))
    activation_map = activation_map.melt(
        var_name='x', value_name='value', ignore_index=False)
    activation_map = activation_map.reset_index()
    activation_map = activation_map.rename(columns={'index': 'y'})

    activation_map = activation_map.rename(columns={'x': 'y', 'y': 'x'})

    min_value = activation_map['value'][activation_map['value'] > 0].min()
    max_value = activation_map['value'].max()

    # Check for negative values when using log scale
    if color_type == "log" and activation_map['value'].min() < 0:
        st.error(
            "Log scale cannot be applied to activation response with negative values.")
        return download_activation_response(som, X_index)

    if plot:
        c = alt.Chart(activation_map).mark_rect().encode(
            x=alt.X('x:O', title=''),
            y=alt.Y('y:O', sort=alt.EncodingSortField(
                'y'), title=''),
            color=alt.Color(
                'value:Q',
                scale=alt.Scale(type=color_type, domain=(
                    min_value, max_value), scheme=color_scheme),
                legend=alt.Legend(
                    orient='bottom',
                    direction='horizontal',
                    gradientLength=615,
                    gradientThickness=20
                )
            )
        ).properties(
            width=600,
            height=600
        )
        st.altair_chart(c, use_container_width=True)

    return download_activation_response(som, X_index)


def plot_activation_response_hex(som, X_index, color_type='linear', color_scheme='lightmulti', plot=True):
    X = X_index[:, 1:]
    activation_map = som.activation_response(X)
    activation_map = pd.DataFrame(
        activation_map, columns=range(1, len(activation_map)+1), index=range(1, len(activation_map)+1))
    activation_map = activation_map.melt(
        var_name='x', value_name='value', ignore_index=False)
    activation_map = activation_map.reset_index()
    activation_map = activation_map.rename(columns={'index': 'y'})

    activation_map = activation_map.rename(columns={'x': 'y', 'y': 'x'})

    min_x = activation_map['x'].min()
    max_x = activation_map['x'].max()
    min_y = activation_map['y'].min()
    max_y = activation_map['y'].max()

    min_value = activation_map['value'][activation_map['value'] > 0].min()
    max_value = activation_map['value'].max()

    # Check for negative values when using log scale
    if color_type == "log" and activation_map['value'].min() < 0:
        st.error(
            "Log scale cannot be applied to activation response with negative values.")
        return download_activation_response(som, X_index)

    # get index from new_dimensions
    index = np.where(new_dimensions == som.get_weights().shape[0])[0][0]
    size = new_sizes[index]

    if plot:

        hexagon = "M0,-2.3094010768L2,-1.1547005384 2,1.1547005384 0,2.3094010768 -2,1.1547005384 -2,-1.1547005384Z"
        c = alt.Chart(activation_map).mark_point(shape=hexagon, size=size**2).encode(
            x=alt.X('xFeaturePos:Q', title='', scale=alt.Scale(
                domain=[min_x-1, max_x+1])).axis(grid=False, tickOpacity=0, domainOpacity=0),
            y=alt.Y('y:Q', sort=alt.EncodingSortField(
                'y', order='descending'), title='', scale=alt.Scale(domain=[min_y-1, max_y+1])).axis(grid=False, labelPadding=20, tickOpacity=0, domainOpacity=0),
            color=alt.Color(
                'value:Q', scale=alt.Scale(scheme=color_scheme)),
            fill=alt.Fill('value:Q', scale=alt.Scale(type=color_type, domain=(
                min_value, max_value), scheme=color_scheme),
                legend=alt.Legend(
                    orient='bottom',
                    direction='horizontal',
                    gradientLength=615,
                    gradientThickness=20
            )
            ),
            stroke=alt.value('black'),
            strokeWidth=alt.value(1.0)
        ).transform_calculate(
            # This field is required for the hexagonal X-Offset
            xFeaturePos='(datum.y%2)/2 + datum.x-.5'
        ).properties(
            # width should be the same as the height
            height=700,
            width=600,
        ).configure_view(
            strokeWidth=0
        )
        st.altair_chart(c, use_container_width=True)

    return download_activation_response(som, X_index)


def feature_space_map_plot(weights, color_type='linear', color_scheme='lightmulti'):
    # plot the mean of the weights across lass dimension
    mean_weights = np.mean(weights, axis=2)
    mean_weights = pd.DataFrame(mean_weights, columns=range(
        1, len(mean_weights)+1), index=range(1, len(mean_weights)+1))
    mean_weights = mean_weights.melt(
        var_name='x', value_name='value', ignore_index=False)
    mean_weights = mean_weights.reset_index()
    mean_weights = mean_weights.rename(columns={'index': 'y'})

    mean_weights = mean_weights.rename(columns={'x': 'y', 'y': 'x'})

    min_value = mean_weights['value'][mean_weights['value'] > 0].min()
    max_value = mean_weights['value'].max()

    # Check for negative values when using log scale
    if color_type == "log" and mean_weights['value'].min() < 0:
        st.error(
            "Log scale cannot be applied to feature space map with negative values.")
        return

    c = alt.Chart(mean_weights).mark_rect().encode(
        x=alt.X('x:O', title=''),
        y=alt.Y('y:O', sort=alt.EncodingSortField(
            'y', order='descending'), title=''),
        color=alt.Color(
            'value:Q',
            scale=alt.Scale(type=color_type, domain=(
                min_value, max_value), scheme=color_scheme),
            legend=alt.Legend(
                orient='bottom',
                direction='horizontal',
                gradientLength=615,
                gradientThickness=20
            )
        )
    ).properties(
        width=600,
        height=600
    )
    st.altair_chart(c, use_container_width=True)


def feature_space_map_plot_hex(weights, color_type='linear', color_scheme='lightmulti'):
    # plot the mean of the weights across lass dimension
    mean_weights = np.mean(weights, axis=2)
    mean_weights = pd.DataFrame(mean_weights, columns=range(
        1, len(mean_weights)+1), index=range(1, len(mean_weights)+1))
    mean_weights = mean_weights.melt(
        var_name='x', value_name='value', ignore_index=False)
    mean_weights = mean_weights.reset_index()
    mean_weights = mean_weights.rename(columns={'index': 'y'})

    mean_weights = mean_weights.rename(columns={'x': 'y', 'y': 'x'})

    min_x = mean_weights['x'].min()
    max_x = mean_weights['x'].max()
    min_y = mean_weights['y'].min()
    max_y = mean_weights['y'].max()

    min_value = mean_weights['value'][mean_weights['value'] > 0].min()
    max_value = mean_weights['value'].max()

    # Check for negative values when using log scale
    if color_type == "log" and mean_weights['value'].min() < 0:
        st.error(
            "Log scale cannot be applied to feature space map with negative values.")
        return

    # get index from new_dimensions
    index = np.where(new_dimensions == weights.shape[0])[0][0]
    size = new_sizes[index]

    hexagon = "M0,-2.3094010768L2,-1.1547005384 2,1.1547005384 0,2.3094010768 -2,1.1547005384 -2,-1.1547005384Z"
    c = alt.Chart(mean_weights).mark_point(shape=hexagon, size=size**2).encode(
        x=alt.X('xFeaturePos:Q', title='', scale=alt.Scale(
            domain=[min_x-1, max_x+1])).axis(grid=False, tickOpacity=0, domainOpacity=0),
        y=alt.Y('y:Q', sort=alt.EncodingSortField(
            'y', order='descending'), title='', scale=alt.Scale(domain=[min_y-1, max_y+1])).axis(grid=False, labelPadding=20, tickOpacity=0, domainOpacity=0),
        color=alt.Color(
            'value:Q', scale=alt.Scale(scheme=color_scheme)),
        fill=alt.Fill(
            'value:Q',
            scale=alt.Scale(type=color_type, domain=(
                min_value, max_value), scheme=color_scheme),
            legend=alt.Legend(
                orient='bottom',
                direction='horizontal',
                gradientLength=615,
                gradientThickness=20
            )
        ),
        stroke=alt.value('black'),
        strokeWidth=alt.value(1.0)
    ).transform_calculate(
        # This field is required for the hexagonal X-Offset
        xFeaturePos='(datum.y%2)/2 + datum.x-.5'
    ).properties(
        # width should be the same as the height
        height=700,
        width=600,
    ).configure_view(
        strokeWidth=0
    )
    st.write('## SOM feature plot')
    st.altair_chart(c, use_container_width=True)


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def scatter_plot_clustering_hex(som, X, GMM_cluster_labels, jitter_amount=0.5, show_grid=True):
    # Get the actual winner neurons (i, j) coordinates
    winner_neurons = [som.winner(x) for x in X]
    winner_neurons = np.array(winner_neurons)

    # Convert to float and add 1 to match the 1-based coordinate system used in visualization
    w_x = winner_neurons[:, 0].astype(float)  # Shifted 0.5 units left
    w_y = winner_neurons[:, 1].astype(float) + 1

    # Apply hexagonal offset to w_x based on whether row is even or odd
    # Store the original positions before adding jitter
    original_w_x = w_x.copy()
    for i in range(len(w_y)):
        # Apply the hexagonal offset: if row (y) is odd, shift right by 0.5
        if w_y[i] % 2 == 1:
            original_w_x[i] += 0.5

    # Apply small random jitter to avoid overlapping points
    for c in np.unique(GMM_cluster_labels):
        idx_target = GMM_cluster_labels == c
        num_points = np.sum(idx_target)

        # Generate random angles and distances - apply user-controlled jitter
        random_angles = np.random.uniform(0, 2 * np.pi, num_points)
        random_distances = np.sqrt(np.random.uniform(
            0, jitter_amount, num_points)) * 0.5

        # Convert polar coordinates to Cartesian coordinates
        w_x[idx_target] = original_w_x[idx_target] + \
            random_distances * np.cos(random_angles)
        w_y[idx_target] = w_y[idx_target] + \
            random_distances * np.sin(random_angles)

    scatter_chart_sample_df = pd.DataFrame(
        {'w_y': w_y, 'w_x': w_x, 'cluster': GMM_cluster_labels})

    dimension = som.get_weights().shape[0]
    # Adjust the min/max bounds to account for the shift
    min_x = 0  # Was 0.5
    max_x = dimension + 1  # Was dimension + 1.5
    min_y = 1  # Was 1
    max_y = dimension  # Fixed: was dimension + 1

    size = new_sizes[np.where(
        new_dimensions == som.get_weights().shape[0])[0][0]]

    # Create the empty hexagonal grid data and chart only if show_grid is True
    hexagon_grid = None
    if show_grid:
        hexagon_grid_df = create_empty_hexagon_df(som)
        # Hexagon shape definition
        hexagon = "M0,-2.3094010768L2,-1.1547005384 2,1.1547005384 0,2.3094010768 -2,1.1547005384 -2,-1.1547005384Z"
        hexagon_grid = alt.Chart(hexagon_grid_df).mark_point(
            shape=hexagon,
            size=size**2,
            opacity=0.25  # Semi-transparent hexagons
        ).encode(
            x=alt.X('xFeaturePos:Q', title='', scale=alt.Scale(domain=[min_x, max_x])).axis(
                grid=False, tickOpacity=0, domainOpacity=0),
            y=alt.Y('y:Q', sort=alt.EncodingSortField(
                'y', order='descending'), title='', scale=alt.Scale(domain=[min_y-1, max_y+1])).axis(grid=False, tickOpacity=0, domainOpacity=0),
            stroke=alt.value('gray'),
            strokeWidth=alt.value(0.5),
            fill=alt.value('white')
        ).transform_calculate(
            xFeaturePos='(datum.y%2)/2 + datum.x-1'  # Shifted 0.5 units left
        )

    # Create the scatter plot chart with direct coordinates
    scatter_chart = alt.Chart(scatter_chart_sample_df).mark_circle().encode(
        x=alt.X('w_x:Q', title='', scale=alt.Scale(
            domain=[min_x, max_x])).axis(
            grid=False, tickOpacity=0, domainOpacity=0),
        y=alt.Y('w_y:Q', title='', scale=alt.Scale(
            domain=[min_y-1, max_y+1])).axis(
            grid=False, tickOpacity=0, domainOpacity=0),
        color=alt.Color('cluster:N', scale=alt.Scale(
            scheme='lightmulti')).legend(orient='bottom'),
        strokeWidth=alt.value(1.0)
    )

    # Layer the scatter plot on top of the grid if grid is shown
    if show_grid:
        combined_chart = alt.layer(hexagon_grid, scatter_chart).properties(
            height=700,
            width=600
        ).configure_view(
            strokeWidth=0
        )
    else:
        combined_chart = scatter_chart.properties(
            height=700,
            width=600
        ).configure_view(
            strokeWidth=0
        )

    st.write('## SOM scatter plot')
    st.altair_chart(combined_chart, use_container_width=True)


def scatter_plot_clustering(som, X, GMM_cluster_labels, jitter_amount=0.5, show_grid=True):

    w_x, w_y = zip(*[som.winner(d) for d in X])
    w_x = np.array(w_x, dtype=float)
    w_y = np.array(w_y, dtype=float)

    for c in np.unique(GMM_cluster_labels):
        idx_target = GMM_cluster_labels == c
        num_points = np.sum(idx_target)

        # Apply user-controlled jitter
        random_x = (np.random.rand(num_points) - 0.5) * jitter_amount * 1.6
        random_y = (np.random.rand(num_points) - 0.5) * jitter_amount * 1.6

        w_x[idx_target] += .5 + random_x
        w_y[idx_target] += .5 + random_y

    # Add 1 to coordinates to match grid starting at (1,1)
    w_x += 1
    w_y += 1

    scatter_chart_sample_df = pd.DataFrame(
        {'w_y': w_y, 'w_x': w_x, 'cluster': GMM_cluster_labels})

    dimension = som.get_weights().shape[0]
    # Adjust min/max to account for the shift
    min_x = 1  # Was 0
    max_x = dimension  # Fixed: was dimension + 1
    min_y = 1  # Was 0
    max_y = dimension  # Fixed: was dimension + 1

    # Create grid background only if show_grid is True
    grid_chart = None
    if show_grid:
        # Create grid background data - using the 1-based grid system
        grid_data = []
        for i in range(1, dimension + 1):  # Fixed: was range(1, dimension + 2)
            for j in range(1, dimension + 1):  # Fixed: was range(1, dimension + 2)
                grid_data.append({'x': i, 'y': j})
        grid_df = pd.DataFrame(grid_data)

        # Create the grid background chart
        grid_chart = alt.Chart(grid_df).mark_rect(
            color='white',
            stroke='lightgray',
            strokeWidth=0.5,
            opacity=0.25
        ).encode(
            x=alt.X('x:O', title='', scale=alt.Scale(
                domain=list(range(min_x, max_x+1)))),
            y=alt.Y('y:O', title='', sort='descending',
                    scale=alt.Scale(domain=list(range(min_y, max_y+1))))
        )

    # Create the scatter plot with shifted coordinates
    scatter_chart = alt.Chart(scatter_chart_sample_df).mark_circle().encode(
        x=alt.X('w_x', title='', scale=alt.Scale(
            domain=[min_x-1, max_x+1])).axis(
            grid=False),
        y=alt.Y('w_y', title='', scale=alt.Scale(
            domain=[min_y-1, max_y+1])).axis(
            grid=False),
        color=alt.Color('cluster:N', scale=alt.Scale(scheme='lightmulti'))
    )

    # Layer the scatter plot on top of the grid if grid is shown
    if show_grid and grid_chart is not None:
        combined_chart = alt.layer(grid_chart, scatter_chart).properties(
            width=600,
            height=600
        ).interactive(bind_x=None, bind_y=None)
    else:
        combined_chart = scatter_chart.properties(
            width=600,
            height=600
        ).interactive(bind_x=None, bind_y=None)

    st.write('## SOM scatter plot')
    st.altair_chart(combined_chart, use_container_width=True)


def scatter_plot_sources(som, sources, raw_df, X, column_name, custom_colors=None, jitter_amount=0.5, show_grid=True):
    # get the index where the sources are in the raw_df and get rows from X
    idx = raw_df.index[raw_df[column_name].isin(sources)]
    X_sources_name = raw_df[column_name][idx]
    X_sources = X[idx]

    w_x, w_y = zip(*[som.winner(d) for d in X_sources])

    w_x = np.array(w_x, dtype=float)
    w_y = np.array(w_y, dtype=float)

    for c in sources:
        idx_target = np.array(X_sources_name) == c
        num_points = np.sum(idx_target)

        # Apply user-controlled jitter
        random_x = (np.random.rand(num_points) - 0.5) * jitter_amount * 1.6
        random_y = (np.random.rand(num_points) - 0.5) * jitter_amount * 1.6

        w_x[idx_target] += .5 + random_x
        w_y[idx_target] += .5 + random_y

    # Add 1 to coordinates to match grid starting at (1,1)
    w_x += 1
    w_y += 1

    scatter_chart_sample_df = pd.DataFrame(
        {'w_y': w_y, 'w_x': w_x, 'sources': X_sources_name})

    dimension = som.get_weights().shape[0]
    # Adjust min/max to account for the shift
    min_x = 1  # Was 0
    max_x = dimension  # Fixed: was dimension + 1
    min_y = 1  # Was 0
    max_y = dimension  # Fixed: was dimension + 1

    # Prepare color scale based on custom colors if provided
    if custom_colors:
        # Create a custom color scale using the user-provided colors
        domain = list(custom_colors.keys())
        range_ = list(custom_colors.values())

        color_scale = alt.Scale(
            domain=domain,
            range=range_
        )
    else:
        # Use default color scheme
        color_scale = alt.Scale(scheme='lightmulti')

    # Create grid background only if show_grid is True
    grid_chart = None
    if show_grid:
        # Create grid background data - using the 1-based grid system
        grid_data = []
        for i in range(1, dimension + 1):  # Fixed: was range(1, dimension + 2)
            for j in range(1, dimension + 1):  # Fixed: was range(1, dimension + 2)
                grid_data.append({'x': i, 'y': j})
        grid_df = pd.DataFrame(grid_data)

        # Create the grid background chart
        grid_chart = alt.Chart(grid_df).mark_rect(
            color='white',
            stroke='lightgray',
            strokeWidth=0.5,
            opacity=0.25
        ).encode(
            x=alt.X('x:O', title='', scale=alt.Scale(
                domain=list(range(min_x, max_x+1)))),
            y=alt.Y('y:O', title='', sort='descending',
                    scale=alt.Scale(domain=list(range(min_y, max_y+1))))
        )

    # Create the scatter plot with shifted coordinates
    scatter_chart = alt.Chart(scatter_chart_sample_df).mark_circle().encode(
        x=alt.X('w_x', title='', scale=alt.Scale(
            domain=[min_x-1, max_x+1])).axis(
            grid=False),
        y=alt.Y('w_y', title='', scale=alt.Scale(
            domain=[min_y-1, max_y+1])).axis(
            grid=False),
        color=alt.Color('sources:N', scale=color_scale).legend(
            orient='bottom'),
        tooltip=['w_x', 'w_y', 'sources']
    )

    # Layer the scatter plot on top of the grid if grid is shown
    if show_grid and grid_chart is not None:
        combined_chart = alt.layer(grid_chart, scatter_chart).properties(
            width=600,
            height=600
        ).interactive(bind_x=None, bind_y=None)
    else:
        combined_chart = scatter_chart.properties(
            width=600,
            height=600
        ).interactive(bind_x=None, bind_y=None)

    st.write('## SOM scatter plot')
    st.altair_chart(combined_chart, use_container_width=True)


def scatter_plot_sources_hex(som, sources, raw_df, X, column_name, custom_colors=None, jitter_amount=0.5, show_grid=True):
    # get the index where the sources are in the raw_df and get rows from X
    idx = raw_df.index[raw_df[column_name].isin(sources)]
    X_sources_name = raw_df[column_name][idx]
    X_sources = X[idx]

    # Get the actual winner neurons (i, j) coordinates
    winner_neurons = [som.winner(d) for d in X_sources]
    winner_neurons = np.array(winner_neurons)

    # Convert to float and add 1 to match the 1-based coordinate system used in visualization
    w_x = winner_neurons[:, 0].astype(float)  # Shifted 0.5 units left
    w_y = winner_neurons[:, 1].astype(float) + 1

    # Apply hexagonal offset to w_x based on whether row is even or odd
    # Store the original positions before adding jitter
    original_w_x = w_x.copy()
    for i in range(len(w_y)):
        # Apply the hexagonal offset: if row (y) is odd, shift right by 0.5
        if w_y[i] % 2 == 1:
            original_w_x[i] += 0.5

    # Apply small random jitter to avoid overlapping points
    for c in sources:
        idx_target = np.array(X_sources_name) == c
        num_points = np.sum(idx_target)

        # Generate random angles and distances - apply user-controlled jitter
        random_angles = np.random.uniform(0, 2 * np.pi, num_points)
        random_distances = np.sqrt(np.random.uniform(
            0, jitter_amount, num_points)) * 0.5

        # Convert polar coordinates to Cartesian coordinates
        w_x[idx_target] = original_w_x[idx_target] + \
            random_distances * np.cos(random_angles)
        w_y[idx_target] = w_y[idx_target] + \
            random_distances * np.sin(random_angles)

    scatter_chart_sample_df = pd.DataFrame(
        {'w_y': w_y, 'w_x': w_x, 'sources': X_sources_name})

    dimension = som.get_weights().shape[0]
    # Adjust the min/max bounds to account for the shift
    min_x = 0  # Was 0.5
    max_x = dimension + 1  # Was dimension + 1.5
    min_y = 1  # Was 1
    max_y = dimension  # Fixed: was dimension + 1

    size = new_sizes[np.where(
        new_dimensions == som.get_weights().shape[0])[0][0]]

    # Prepare color scale based on custom colors if provided
    if custom_colors:
        # Create a custom color scale using the user-provided colors
        domain = list(custom_colors.keys())
        range_ = list(custom_colors.values())

        color_scale = alt.Scale(
            domain=domain,
            range=range_
        )
    else:
        # Use default color scheme
        color_scale = alt.Scale(scheme='lightmulti')

    # Create the empty hexagonal grid data and chart only if show_grid is True
    hexagon_grid = None
    if show_grid:
        hexagon_grid_df = create_empty_hexagon_df(som)
        # Hexagon shape definition
        hexagon = "M0,-2.3094010768L2,-1.1547005384 2,1.1547005384 0,2.3094010768 -2,1.1547005384 -2,-1.1547005384Z"
        hexagon_grid = alt.Chart(hexagon_grid_df).mark_point(
            shape=hexagon,
            size=size**2,
            opacity=0.25  # Semi-transparent hexagons
        ).encode(
            x=alt.X('xFeaturePos:Q', title='', scale=alt.Scale(domain=[min_x, max_x])).axis(
                grid=False, tickOpacity=0, domainOpacity=0),
            y=alt.Y('y:Q', sort=alt.EncodingSortField(
                'y', order='descending'), title='', scale=alt.Scale(domain=[min_y-1, max_y+1])).axis(grid=False, tickOpacity=0, domainOpacity=0),
            stroke=alt.value('gray'),
            strokeWidth=alt.value(0.5),
            fill=alt.value('white')
        ).transform_calculate(
            xFeaturePos='(datum.y%2)/2 + datum.x-1'  # Shifted 0.5 units left
        )

    # Create the scatter plot chart - using direct coordinates with pre-calculated offset
    scatter_chart = alt.Chart(scatter_chart_sample_df).mark_circle().encode(
        x=alt.X('w_x:Q', title='', scale=alt.Scale(
            domain=[min_x, max_x])).axis(
            grid=False, tickOpacity=0
        ),
        y=alt.Y('w_y:Q', title='', scale=alt.Scale(
            domain=[min_y-1, max_y+1])).axis(
            grid=False, tickOpacity=0
        ),
        color=alt.Color('sources:N', scale=color_scale).legend(
            orient='bottom'),
        tooltip=['w_x', 'w_y', 'sources']
    )

    # Layer the scatter plot on top of the grid if grid is shown
    if show_grid:
        combined_chart = alt.layer(hexagon_grid, scatter_chart).properties(
            height=700,
            width=600,
        ).configure_view(
            strokeWidth=0
        ).interactive(bind_x=None, bind_y=None)
    else:
        combined_chart = scatter_chart.properties(
            height=700,
            width=600,
        ).configure_view(
            strokeWidth=0
        ).interactive(bind_x=None, bind_y=None)

    st.write('## SOM scatter plot')
    st.altair_chart(combined_chart, use_container_width=True)


def project_feature(som, X, feature, source=None):
    '''
    Returns a 2D map of lists containing the values of the external feature for each neuron of the SOM
    '''
    map = [[[None] for _ in range(som._weights.shape[1])]
           for _ in range(som._weights.shape[0])]
    for cnt, xx in enumerate(X):
        if source is not None:
            # is not in source
            if not feature[cnt] in source:
                continue
        w = som.winner(xx)

        if map[w[0]][w[1]][0] == None:
            map[w[0]][w[1]] = [feature[cnt]]
        else:
            map[w[0]][w[1]].append(feature[cnt])

    return map


def category_plot_sources(_map, flip=True, custom_colors=None, cluster_mapping=None, cluster_border_colors=None):

    if flip:
        _map = list(map(list, zip(*_map)))
    '''
    plot the most common element in the list
    '''
    # plt.figure(figsize=(10, 10))
    # bone
    # Convert map to a list of lists (2D) by summing or averaging over the third dimension
    winning_categories = []
    for idx_outer, sublist_outer in enumerate(_map):
        for idx_inner, sublist in enumerate(sublist_outer):
            # the most common element in the list is the category
            # Exclude None and nan from sublist before finding the most common element
            filtered_sublist = [v for v in sublist if v is not None and not (
                isinstance(v, float) and np.isnan(v))]
            if filtered_sublist:
                most_common = max(set(filtered_sublist),
                                  key=filtered_sublist.count)
            else:
                most_common = None
            winning_categories.append(
                [int(idx_outer+1), int(idx_inner+1), most_common])

    winning_categories = np.array(winning_categories)

    pd_winning_categories = pd.DataFrame(
        winning_categories, columns=['w_y', 'w_x', 'source'])

    min_x = pd_winning_categories['w_x'].min()
    max_x = pd_winning_categories['w_x'].max()
    min_y = pd_winning_categories['w_y'].min()
    max_y = pd_winning_categories['w_y'].max()

    tick_x = np.arange(min_x, max_x+1, 1)
    tick_y = np.arange(min_y, max_y+1, 1)[::-1]
    # remove row with cluster None
    pd_winning_categories = pd_winning_categories.dropna()

    # Prepare color scale based on custom colors if provided
    if custom_colors:
        # Create a custom color scale using the user-provided colors
        domain = list(custom_colors.keys())
        range_ = list(custom_colors.values())

        color_scale = alt.Scale(
            domain=domain,
            range=range_
        )
    else:
        # Use default color scheme
        color_scale = alt.Scale(scheme='lightmulti')

    # Create a base chart for the rectangles
    base_chart = alt.Chart(pd_winning_categories).encode(
        x=alt.X('w_x:O', title='', scale=alt.Scale(domain=tick_x)),
        y=alt.Y('w_y:O', sort=alt.EncodingSortField(
            'w_y', order='descending'), title='', scale=alt.Scale(domain=tick_y))
    )

    # Create the main visualization with fill colors based on SIMBAD type
    fill_chart = base_chart.mark_rect().encode(
        color=alt.Color(
            'source:N', scale=color_scale, legend=alt.Legend(orient='bottom')),
        tooltip=['source:N', 'w_x:Q', 'w_y:Q']
    )

    # If cluster information is provided, add stroke color based on cluster assignment
    if cluster_mapping is not None and cluster_border_colors is not None:
        # Add cluster ID to each neuron if available
        pd_winning_categories['cluster_id'] = pd_winning_categories.apply(
            lambda row: cluster_mapping.get((row['w_x']-1, row['w_y']-1), -1),
            axis=1
        )

        # Create a stroke color scale for clusters
        cluster_domain = list(cluster_border_colors.keys())
        cluster_range = list(cluster_border_colors.values())

        # Create a separate chart with thicker stroke for the borders, colored by cluster
        border_chart = base_chart.mark_rect(
            strokeWidth=2  # Thicker border to make clusters more visible but not too thick
        ).encode(
            stroke=alt.condition(
                alt.datum.cluster_id >= 0,  # Only add colored border if we have a cluster assignment
                alt.Color('cluster_id:N',
                          scale=alt.Scale(domain=cluster_domain,
                                          range=cluster_range),
                          legend=alt.Legend(title="Clusters", orient='bottom')),
                # No border for neurons without cluster assignment
                alt.value(None)
            ),
            tooltip=['source:N', 'w_x:Q', 'w_y:Q',
                     alt.Tooltip('cluster_id:N', title='Cluster')]
        )

        # Layer the two charts
        scatter_chart_sample = alt.layer(fill_chart, border_chart).properties(
            height=700,
            width=600
        )
    else:
        # Use the original chart if no cluster information
        scatter_chart_sample = fill_chart.properties(
            height=700,
            width=600
        )

    st.write('## SOM category plot')
    st.altair_chart(scatter_chart_sample, use_container_width=True)


def category_plot_sources_hex(_map, flip=True, custom_colors=None, cluster_mapping=None, enhanced_border_colors=None, standard_border_colors=None, cluster_stroke_width=4, enhanced_border_width=4, som_shape=None):
    if flip:
        _map = list(map(list, zip(*_map)))
    '''
    plot the most common element in the list
    '''
    # plt.figure(figsize=(10, 10))
    # bone
    # Convert map to a list of lists (2D) by summing or averaging over the third dimension
    winning_categories = []
    for idx_outer, sublist_outer in enumerate(_map):
        for idx_inner, sublist in enumerate(sublist_outer):
            # the most common element in the list is the category
            # Exclude None and nan from sublist before finding the most common element
            filtered_sublist = [v for v in sublist if v is not None and not (
                isinstance(v, float) and np.isnan(v))]
            if filtered_sublist:
                most_common = max(set(filtered_sublist),
                                  key=filtered_sublist.count)
            else:
                most_common = None
            winning_categories.append(
                [int(idx_outer+1), int(idx_inner+1), most_common])

    winning_categories = np.array(winning_categories)

    pd_winning_categories = pd.DataFrame(
        winning_categories, columns=['y', 'x', 'source'])

    min_x = pd_winning_categories['x'].min()
    max_x = pd_winning_categories['x'].max()
    min_y = pd_winning_categories['y'].min()
    max_y = pd_winning_categories['y'].max()

    # drop rows with source None
    pd_winning_categories = pd_winning_categories.dropna()

    # set x and y as float
    pd_winning_categories['x'] = pd_winning_categories['x'].astype(float)
    pd_winning_categories['y'] = pd_winning_categories['y'].astype(float)

    # order the dataframe using x and y
    pd_winning_categories = pd_winning_categories.sort_values(by=['x', 'y'])
    pd_winning_categories = pd_winning_categories.reset_index(drop=True)

    # get index from new_dimensions
    index = np.where(new_dimensions == len(_map))[0][0]
    size = new_sizes[index]

    # Prepare color scale based on custom colors if provided
    if custom_colors:
        # Create a custom color scale using the user-provided colors
        domain = list(custom_colors.keys())
        range_ = list(custom_colors.values())

        color_scale = alt.Scale(
            domain=domain,
            range=range_
        )
    else:
        # Use default color scheme
        color_scale = alt.Scale(scheme='lightmulti')

    # If cluster information is provided, add cluster ID to each neuron
    if cluster_mapping is not None and enhanced_border_colors is not None and som_shape is not None:
        # Add cluster ID to each neuron if available
        pd_winning_categories['cluster_id'] = pd_winning_categories.apply(
            lambda row: cluster_mapping.get((row['x']-1, row['y']-1), -1),
            axis=1
        )

        # Detect cluster boundaries
        boundary_neurons = detect_cluster_boundaries(
            cluster_mapping, som_shape)

        # Add boundary flag to dataframe
        pd_winning_categories['is_boundary'] = pd_winning_categories.apply(
            lambda row: boundary_neurons.get((row['x']-1, row['y']-1), False),
            axis=1
        )

        # Create stroke color scales for enhanced borders
        enhanced_domain = list(enhanced_border_colors.keys())
        enhanced_range = list(enhanced_border_colors.values())

        # Get default standard border color (black) from the standard_border_colors dict
        default_standard_color = '#000000'
        if standard_border_colors and len(standard_border_colors) > 0:
            default_standard_color = list(standard_border_colors.values())[0]

        # Create a base encoding for both charts
        base_encoding = {
            'x': alt.X('xFeaturePos:Q', title='', scale=alt.Scale(domain=[min_x-1, max_x+1])).axis(
                grid=False, tickOpacity=0, domainOpacity=0),
            'y': alt.Y('y:Q', sort=alt.EncodingSortField(
                'y', order='descending'), title='', scale=alt.Scale(domain=[min_y-1, max_y+1])
            ).axis(grid=False, labelPadding=20, tickOpacity=0, domainOpacity=0)
        }

        hexagon = "M0,-2.3094010768L2,-1.1547005384 2,1.1547005384 0,2.3094010768 -2,1.1547005384 -2,-1.1547005384Z"

        # Create the fill chart for SIMBAD types with standard borders
        if standard_border_colors and len(standard_border_colors) > 1:
            # Use conditional coloring for standard borders when customization is enabled
            standard_domain = list(standard_border_colors.keys())
            standard_range = list(standard_border_colors.values())

            fill_chart = alt.Chart(pd_winning_categories).mark_point(
                shape=hexagon,
                size=size**2,
                strokeWidth=1.0  # Always use standard width for fill chart
            ).encode(
                **base_encoding,
                color=alt.Color('source:N', scale=color_scale),
                fill=alt.Color('source:N', scale=color_scale).legend(
                    orient='bottom'),
                stroke=alt.condition(
                    alt.datum.cluster_id >= 0,  # Only apply cluster color if cluster_id is valid
                    alt.Color('cluster_id:N',
                              scale=alt.Scale(domain=standard_domain,
                                              range=standard_range),
                              legend=None),  # No legend for fill chart borders
                    # Fallback to default for non-clustered neurons
                    alt.value(default_standard_color)
                ),
                tooltip=['source:N', 'x:Q', 'y:Q', alt.Tooltip(
                    'cluster_id:N', title='Cluster'), alt.Tooltip(
                    'is_boundary:N', title='Boundary')]
            ).transform_calculate(
                # This field is required for the hexagonal X-Offset
                xFeaturePos='(datum.y%2)/2 + datum.x-.5'
            )
        else:
            # Use single color for all standard borders when not customizing
            fill_chart = alt.Chart(pd_winning_categories).mark_point(
                shape=hexagon,
                size=size**2,
                strokeWidth=1.0  # Always use standard width for fill chart
            ).encode(
                **base_encoding,
                color=alt.Color('source:N', scale=color_scale),
                fill=alt.Color('source:N', scale=color_scale).legend(
                    orient='bottom'),
                stroke=alt.value(default_standard_color),
                tooltip=['source:N', 'x:Q', 'y:Q', alt.Tooltip(
                    'cluster_id:N', title='Cluster'), alt.Tooltip(
                    'is_boundary:N', title='Boundary')]
            ).transform_calculate(
                # This field is required for the hexagonal X-Offset
                xFeaturePos='(datum.y%2)/2 + datum.x-.5'
            )

        # Create enhanced borders only for boundary neurons using conditional encoding
        enhanced_border_chart = alt.Chart(pd_winning_categories).mark_point(
            shape=hexagon,
            size=(size-(enhanced_border_width-1))**2,
            filled=False,  # Only show the border
            strokeWidth=enhanced_border_width
        ).encode(
            **base_encoding,
            stroke=alt.condition(
                # Only boundary neurons with valid cluster assignment
                (alt.datum.cluster_id >= 0) & (alt.datum.is_boundary == True),
                alt.Color('cluster_id:N',
                          scale=alt.Scale(domain=enhanced_domain,
                                          range=enhanced_range),
                          legend=alt.Legend(title="clusters", orient='bottom')),
                alt.value(None)  # No enhanced border for non-boundary neurons
            ),
            # Explicitly set no fill for enhanced borders
            fill=alt.value(None),
            opacity=alt.condition(
                # Only show enhanced borders for boundary neurons
                (alt.datum.cluster_id >= 0) & (alt.datum.is_boundary == True),
                alt.value(1.0),
                alt.value(0.0)  # Hide non-boundary neurons in enhanced chart
            ),
            tooltip=['source:N', 'x:Q', 'y:Q', alt.Tooltip(
                'cluster_id:N', title='Cluster'), alt.Tooltip(
                'is_boundary:N', title='Boundary')]
        ).transform_calculate(
            # This field is required for the hexagonal X-Offset
            xFeaturePos='(datum.y%2)/2 + datum.x-.5'
        )

        # Standard border colors are now handled in the fill_chart above

        # Layer the charts: fill first, then enhanced borders
        # Important: enhanced borders must be layered on top to be visible
        c = alt.layer(fill_chart, enhanced_border_chart).resolve_scale(
            stroke='independent').properties(
            # width should be the same as the height
            height=700,
            width=600,
        ).configure_view(
            strokeWidth=0
        ).configure_legend(
            symbolStrokeWidth=1.0,  # Adjust the stroke width of legend symbols
            # Adjust the size of legend symbols (default is 100)
            symbolSize=size**2
        )
    else:
        # Use the original chart if no cluster information
        hexagon = "M0,-2.3094010768L2,-1.1547005384 2,1.1547005384 0,2.3094010768 -2,1.1547005384 -2,-1.1547005384Z"
        c = alt.Chart(pd_winning_categories).mark_point(shape=hexagon, size=size**2).encode(
            x=alt.X('xFeaturePos:Q', title='', scale=alt.Scale(domain=[min_x-1, max_x+1])).axis(
                grid=False, tickOpacity=0, domainOpacity=0),
            y=alt.Y('y:Q', sort=alt.EncodingSortField(
                'y', order='descending'), title='', scale=alt.Scale(domain=[min_y-1, max_y+1])
            ).axis(grid=False, labelPadding=20, tickOpacity=0, domainOpacity=0),
            color=alt.Color(
                'source:N', scale=color_scale),
            fill=alt.Color('source:N', scale=color_scale).legend(
                orient='bottom'),
            stroke=alt.value('#000000'),  # Default black border
            # Default stroke width for non-cluster mode
            strokeWidth=alt.value(1.0),
            tooltip=['source:N', 'x:Q', 'y:Q']
        ).transform_calculate(
            # This field is required for the hexagonal X-Offset
            xFeaturePos='(datum.y%2)/2 + datum.x-.5'
        ).properties(
            # width should be the same as the height
            height=700,
            width=600,
        ).configure_view(
            strokeWidth=0
        ).configure_legend(
            symbolStrokeWidth=1.0,  # Adjust the stroke width of legend symbols
            # Adjust the size of legend symbols (default is 100)
            symbolSize=size**2
        )

    st.write('## SOM category plot')
    st.altair_chart(c, use_container_width=True)


def category_plot_clustering_hex(map):
    '''
    plot the most common element in the list
    '''
    np_map = np.empty((len(map), len(map[0])))
    # plt.figure(figsize=(10, 10))
    # bone
    # Convert map to a list of lists (2D) by summing or averaging over the third dimension
    markers = ['0', '1', '2', '3', '4', '5']
    winning_categories = []
    for idx_outer, sublist_outer in enumerate(map):
        for idx_inner, sublist in enumerate(sublist_outer):
            # the most common element in the list is the category
            try:
                np_map[idx_outer][idx_inner] = max(
                    set(sublist), key=sublist.count)
            except TypeError:
                np_map[idx_outer][idx_inner] = None

    np_map = np_map.T
    for idx_outer, sublist_outer in enumerate(np_map):
        for idx_inner, sublist in enumerate(sublist_outer):
            if not math.isnan(sublist):
                winning_categories.append(
                    [int(idx_outer+1), int(idx_inner+1), markers[int(sublist)]])
    winning_categories = np.array(winning_categories)

    pd_winning_categories = pd.DataFrame(
        winning_categories, columns=['y', 'x', 'value'])

    # set x and y as float
    pd_winning_categories['x'] = pd_winning_categories['x'].astype(float)
    pd_winning_categories['y'] = pd_winning_categories['y'].astype(float)

    # order the dataframe using x and y
    pd_winning_categories = pd_winning_categories.sort_values(by=['x', 'y'])
    pd_winning_categories = pd_winning_categories.reset_index(drop=True)

    min_x = pd_winning_categories['x'].min()
    max_x = pd_winning_categories['x'].max()
    min_y = pd_winning_categories['y'].min()
    max_y = pd_winning_categories['y'].max()

    # get index from new_dimensions
    index = np.where(new_dimensions == len(map))[0][0]
    size = new_sizes[index]

    hexagon = "M0,-2.3094010768L2,-1.1547005384 2,1.1547005384 0,2.3094010768 -2,1.1547005384 -2,-1.1547005384Z"
    c = alt.Chart(pd_winning_categories).mark_point(shape=hexagon, size=size**2).encode(
        x=alt.X('xFeaturePos:Q', title='', scale=alt.Scale(domain=[min_x-1, max_x+1])).axis(
            grid=False, tickOpacity=0, domainOpacity=0),
        y=alt.Y('y:Q', sort=alt.EncodingSortField(
            'y', order='descending'), title='', scale=alt.Scale(domain=[min_y-1, max_y+1])
        ).axis(grid=False, labelPadding=20, tickOpacity=0, domainOpacity=0),
        color=alt.Color(
            'value:Q', scale=alt.Scale(scheme='lightmulti')),
        fill=alt.Color('value:Q', scale=alt.Scale(
            scheme='lightmulti')).legend(orient='bottom'),
        stroke=alt.value('black'),
        strokeWidth=alt.value(1.0)
    ).transform_calculate(
        # This field is required for the hexagonal X-Offset
        xFeaturePos='(datum.y%2)/2 + datum.x-.5'
    ).properties(
        # width should be the same as the height
        height=700,
        width=600,
    ).configure_view(
        strokeWidth=0
    ).configure_legend(
        symbolStrokeWidth=1.0,  # Adjust the stroke width of legend symbols
        # Adjust the size of legend symbols (default is 100)
        symbolSize=size**2
    )
    st.write('## SOM category plot')
    st.altair_chart(c, use_container_width=True)


def category_plot_clustering(map):
    '''
    plot the most common element in the list
    '''
    np_map = np.empty((len(map), len(map[0])))
    # plt.figure(figsize=(10, 10))
    # bone
    # Convert map to a list of lists (2D) by summing or averaging over the third dimension
    markers = ['0', '1', '2', '3', '4', '5']
    winning_categories = []
    for idx_outer, sublist_outer in enumerate(map):
        for idx_inner, sublist in enumerate(sublist_outer):
            # the most common element in the list is the category
            try:
                np_map[idx_outer][idx_inner] = max(
                    set(sublist), key=sublist.count)
            except TypeError:
                np_map[idx_outer][idx_inner] = None
    np_map = np_map.T
    for idx_outer, sublist_outer in enumerate(np_map):
        for idx_inner, sublist in enumerate(sublist_outer):
            if not math.isnan(sublist):
                winning_categories.append(
                    [idx_outer+.5, idx_inner+.5, markers[int(sublist)]])
                # markerfacecolor='None', markeredgecolor=colors[int(sublist)], markersize=12, markeredgewidth=2, label=str(int(sublist)))
    winning_categories = np.array(winning_categories)

    pd_winning_categories = pd.DataFrame(
        winning_categories, columns=['w_y', 'w_x', 'cluster'])

    min_x = np.array(pd_winning_categories['w_x'], dtype=float).min()
    max_x = np.array(pd_winning_categories['w_x'], dtype=float).max()
    min_y = np.array(pd_winning_categories['w_y'], dtype=float).min()
    max_y = np.array(pd_winning_categories['w_y'], dtype=float).max()

    scatter_chart_sample = alt.Chart(pd_winning_categories).mark_point(size=105-len(map)).encode(
        x=alt.X('w_x:Q', title='', scale=alt.Scale(
            domain=[min_x-1, max_x+1])).axis(grid=False),
        y=alt.Y('w_y:Q', title='', scale=alt.Scale(
            domain=[min_y-1, max_y+1])).axis(grid=False),
        shape='cluster:N',
        color=alt.Color('cluster:N', scale=alt.Scale(scheme='lightmulti')),
    ).properties(
        width=600,
        height=600
    )
    st.write('## SOM category plot')
    st.altair_chart(scatter_chart_sample, use_container_width=True)


def features_plot_hex(_map, color_type, color_scheme, scaling='mean', flip=True, feature_name=None):
    if flip:
        _map = list(map(list, zip(*_map)))
    '''
    Plot the map (which is a list) of the external feature, different scaling methods are available:
    - sum
    - mean
    - max
    - min
    - median
    '''
    np_map = np.empty((len(_map), len(_map[0])))

    # Determine the metric type for global scaling
    metric_type = 'statistical' if scaling in [
        'sum', 'std'] else 'single_value'

    if scaling == 'sum':
        for idx_outer, sublist_outer in enumerate(_map):
            for idx_inner, sublist in enumerate(sublist_outer):
                try:
                    np_map[idx_outer][idx_inner] = sum(sublist)
                except TypeError:
                    np_map[idx_outer][idx_inner] = np.nan
    elif scaling == 'mean':
        for idx_outer, sublist_outer in enumerate(_map):
            for idx_inner, sublist in enumerate(sublist_outer):
                try:
                    np_map[idx_outer][idx_inner] = np.mean(sublist)
                except TypeError:
                    np_map[idx_outer][idx_inner] = np.nan
    elif scaling == 'max':
        for idx_outer, sublist_outer in enumerate(_map):
            for idx_inner, sublist in enumerate(sublist_outer):
                try:
                    np_map[idx_outer][idx_inner] = max(sublist)
                except TypeError:
                    np_map[idx_outer][idx_inner] = np.nan
    elif scaling == 'min':
        for idx_outer, sublist_outer in enumerate(_map):
            for idx_inner, sublist in enumerate(sublist_outer):
                try:
                    np_map[idx_outer][idx_inner] = min(sublist)
                except TypeError:
                    np_map[idx_outer][idx_inner] = np.nan
    elif scaling == 'median':
        for idx_outer, sublist_outer in enumerate(_map):
            for idx_inner, sublist in enumerate(sublist_outer):
                try:
                    np_map[idx_outer][idx_inner] = np.median(sublist)
                except TypeError:
                    np_map[idx_outer][idx_inner] = np.nan
    elif scaling == 'std':
        for idx_outer, sublist_outer in enumerate(_map):
            for idx_inner, sublist in enumerate(sublist_outer):
                try:
                    np_map[idx_outer][idx_inner] = np.std(sublist)
                except TypeError:
                    np_map[idx_outer][idx_inner] = np.nan
    else:
        raise ValueError('scaling method not recognized')

    np_map = pd.DataFrame(np_map, columns=range(
        1, len(np_map)+1), index=range(1, len(np_map)+1))

    np_map = np_map.melt(
        var_name='x', value_name='value', ignore_index=False)

    np_map = np_map.reset_index()
    np_map = np_map.rename(columns={'index': 'y'})

    # Get the dimensions of the SOM grid
    grid_size = len(_map)

    # Calculate the min/max values for this specific visualization
    local_min_value = np_map['value'][np_map['value'] > float(
        '-inf')].min() if not np_map['value'].empty else 0
    local_max_value = np_map['value'][np_map['value'] < float(
        'inf')].max() if not np_map['value'].empty else 1

    # Update global scale range if feature_name is provided
    if feature_name:
        min_value, max_value = update_feature_scale_range(
            feature_name, metric_type, local_min_value, local_max_value, color_type)

        # For log scale, use the precomputed log_min if available
        if color_type == "log" and feature_name in feature_scale_ranges:
            if 'log_min' in feature_scale_ranges[feature_name][metric_type]:
                min_value = feature_scale_ranges[feature_name][metric_type]['log_min']
    else:
        min_value, max_value = local_min_value, local_max_value

    # Check for negative values when using log scale - check raw data first
    if color_type == "log" and np_map['value'].min() < 0:
        st.error("Log scale cannot be applied to features with negative values.")
        return

    # For log scale, find the smallest non-zero value if min_value is 0
    if color_type == "log" and min_value == 0:
        # Filter out zeros and find the smallest positive value
        positive_values = np_map['value'][(np_map['value'] > 0)]
        if not positive_values.empty:
            min_value = positive_values.min()
        else:
            # If there are no positive values, use a small positive number
            min_value = 0.01
            st.warning(
                "No positive values found for log scale. Using default minimum value.")

    # get index from new_dimensions
    index = np.where(new_dimensions == len(_map))[0][0]
    size = new_sizes[index]

    # Create domain values for x and y axes based on grid size
    x_domain = list(range(1, grid_size + 1))
    y_domain = list(range(1, grid_size + 1))

    # Create tick values based on grid size (even/odd)
    # Always include first and last number, then either even or odd numbers in between
    if grid_size % 2 == 0:  # Even grid size
        tick_values = [1] + \
            [i for i in range(2, grid_size + 1, 2)]  # Even numbers
    else:  # Odd grid size
        tick_values = [i for i in range(1, grid_size + 1, 2)]  # Odd numbers

    # Ensure the last number is always included
    if grid_size not in tick_values:
        tick_values.append(grid_size)

    # Sort the tick values to ensure they're in order
    tick_values.sort()

    hexagon = "M0,-2.3094010768L2,-1.1547005384 2,1.1547005384 0,2.3094010768 -2,1.1547005384 -2,-1.1547005384Z"
    c = alt.Chart(np_map).mark_point(shape=hexagon, size=size**2).encode(
        x=alt.X('xFeaturePos:Q', title='',
                # Extend domain slightly
                scale=alt.Scale(domain=[0, grid_size + 1.5]),
                axis=alt.Axis(
                    grid=False,
                    # Set specific tick values
                    values=tick_values,
                    tickOpacity=1,  # Make ticks visible
                    domainOpacity=1,  # Make domain line visible
                    labels=True,  # Ensure labels are shown
                    labelOverlap=False  # Prevent label overlap
                )
                ),
        y=alt.Y('y:Q',
                sort=alt.EncodingSortField('y', order='descending'),
                title='',
                # Extend domain slightly
                scale=alt.Scale(domain=[0, grid_size + 1.5]),
                axis=alt.Axis(
                    grid=False,
                    labelPadding=20,
                    # Set specific tick values
                    values=tick_values,
                    tickOpacity=1,  # Make ticks visible
                    domainOpacity=1,  # Make domain line visible
                    labels=True,  # Ensure labels are shown
                    labelOverlap=False  # Prevent label overlap
                )
                ),
        color=alt.Color('value:Q', scale=alt.Scale(
            scheme=color_scheme, type='pow')),
        fill=alt.Fill(
            'value:Q',
            scale=alt.Scale(type=color_type, domain=(
                min_value, max_value), scheme=color_scheme),
            legend=alt.Legend(
                orient='bottom',
                direction='horizontal',
                gradientLength=615,
                gradientThickness=20
            )
        ),
        stroke=alt.value('black'),
        strokeWidth=alt.value(1.0)
    ).transform_calculate(
        xFeaturePos='(datum.y%2)/2 + datum.x-.5'
    ).properties(
        height=750,
        width=600,
        title={
            "text": f"{feature_name} ({scaling})",
            "anchor": "middle",
            "fontSize": 16,
            "fontWeight": "bold",
            "offset": 1
        }
    ).configure_view(
        strokeWidth=0
    )
    st.write('## SOM feature plot')
    st.altair_chart(c, use_container_width=True)

    return c


def features_plot(_map, color_type, color_scheme, scaling='mean', flip=True, feature_name=None):
    if flip:
        _map = list(map(list, zip(*_map)))
    '''
    Plot the map (which is a list) of the external feature, different scaling methods are available:
    - sum
    - mean
    - max
    - min
    - median
    '''
    np_map = np.empty((len(_map), len(_map[0])))

    # Determine the metric type for global scaling
    metric_type = 'statistical' if scaling in [
        'sum', 'std'] else 'single_value'

    if scaling == 'sum':
        for idx_outer, sublist_outer in enumerate(_map):
            for idx_inner, sublist in enumerate(sublist_outer):
                try:
                    np_map[idx_outer][idx_inner] = sum(sublist)
                except TypeError:
                    np_map[idx_outer][idx_inner] = 0
    elif scaling == 'mean':
        for idx_outer, sublist_outer in enumerate(_map):
            for idx_inner, sublist in enumerate(sublist_outer):
                try:
                    np_map[idx_outer][idx_inner] = np.mean(sublist)
                except TypeError:
                    np_map[idx_outer][idx_inner] = 0
    elif scaling == 'max':
        for idx_outer, sublist_outer in enumerate(_map):
            for idx_inner, sublist in enumerate(sublist_outer):
                try:
                    np_map[idx_outer][idx_inner] = max(sublist)
                except TypeError:
                    np_map[idx_outer][idx_inner] = 0
    elif scaling == 'min':
        for idx_outer, sublist_outer in enumerate(_map):
            for idx_inner, sublist in enumerate(sublist_outer):
                try:
                    np_map[idx_outer][idx_inner] = min(sublist)
                except TypeError:
                    np_map[idx_outer][idx_inner] = 0
    elif scaling == 'median':
        for idx_outer, sublist_outer in enumerate(_map):
            for idx_inner, sublist in enumerate(sublist_outer):
                try:
                    np_map[idx_outer][idx_inner] = np.median(sublist)
                except TypeError:
                    np_map[idx_outer][idx_inner] = 0
    elif scaling == 'std':
        for idx_outer, sublist_outer in enumerate(_map):
            for idx_inner, sublist in enumerate(sublist_outer):
                try:
                    np_map[idx_outer][idx_inner] = np.std(sublist)
                except TypeError:
                    np_map[idx_outer][idx_inner] = 0
    else:
        raise ValueError('scaling method not recognized')

    np_map = pd.DataFrame(np_map, columns=range(
        1, len(np_map)+1), index=range(1, len(np_map)+1))
    np_map = np_map.melt(
        var_name='x', value_name='value', ignore_index=False)
    np_map = np_map.reset_index()
    np_map = np_map.rename(columns={'index': 'y'})

    # Get the dimensions of the SOM grid
    grid_size = len(_map)

    # Calculate the min/max values for this specific visualization
    local_min_value = np_map['value'][np_map['value'] > float(
        '-inf')].min() if not np_map['value'].empty else 0
    local_max_value = np_map['value'][np_map['value'] < float(
        'inf')].max() if not np_map['value'].empty else 1

    # Update global scale range if feature_name is provided
    if feature_name:
        min_value, max_value = update_feature_scale_range(
            feature_name, metric_type, local_min_value, local_max_value, color_type)

        # For log scale, use the precomputed log_min if available
        if color_type == "log" and feature_name in feature_scale_ranges:
            if 'log_min' in feature_scale_ranges[feature_name][metric_type]:
                min_value = feature_scale_ranges[feature_name][metric_type]['log_min']
    else:
        min_value, max_value = local_min_value, local_max_value

    # Check for negative values when using log scale - check raw data first
    if color_type == "log" and np_map['value'].min() < 0:
        st.error("Log scale cannot be applied to features with negative values.")
        return

    # For log scale, find the smallest non-zero value if min_value is 0
    if color_type == "log" and min_value == 0:
        # Filter out zeros and find the smallest positive value
        positive_values = np_map['value'][(np_map['value'] > 0)]
        if not positive_values.empty:
            min_value = positive_values.min()
        else:
            # If there are no positive values, use a small positive number
            min_value = 0.01
            st.warning(
                "No positive values found for log scale. Using default minimum value.")

    # Create domain values for x and y axes based on grid size
    x_domain = list(range(1, grid_size + 1))
    y_domain = list(range(1, grid_size + 1))

    # Create tick values based on grid size (even/odd)
    # Always include first and last number, then either even or odd numbers in between
    if grid_size % 2 == 0:  # Even grid size
        tick_values = [1] + \
            [i for i in range(2, grid_size + 1, 2)]  # Even numbers
    else:  # Odd grid size
        tick_values = [i for i in range(1, grid_size + 1, 2)]  # Odd numbers

    # Ensure the last number is always included
    if grid_size not in tick_values:
        tick_values.append(grid_size)

    # Sort the tick values to ensure they're in order
    tick_values.sort()

    c = alt.Chart(np_map).mark_rect().encode(
        x=alt.X('x:O', title='',
                # Add padding to ensure last tick is visible
                scale=alt.Scale(domain=x_domain, padding=0.5),
                axis=alt.Axis(
                    values=tick_values,  # Set specific tick values
                    labelAngle=0,
                    tickOpacity=1,  # Make ticks visible
                    domainOpacity=1,  # Make domain line visible
                    labels=True,  # Ensure labels are shown
                    labelOverlap=False  # Prevent label overlap
                )
                ),
        y=alt.Y('y:O', title='',
                sort=alt.EncodingSortField('y', order='descending'),
                # Add padding to ensure last tick is visible
                scale=alt.Scale(domain=y_domain, padding=0.5),
                axis=alt.Axis(
                    values=tick_values,  # Set specific tick values
                    tickOpacity=1,  # Make ticks visible
                    domainOpacity=1,  # Make domain line visible
                    labels=True,  # Ensure labels are shown
                    labelOverlap=False  # Prevent label overlap
                )
                ),
        color=alt.Color(
            'value:Q', scale=alt.Scale(type=color_type, domain=(min_value, max_value), scheme=color_scheme))
    ).properties(
        height=750,
        width=600,
        title={
            "text": f"{feature_name} ({scaling})",
            "anchor": "middle",
            "fontSize": 16,
            "fontWeight": "bold",
            "offset": 5
        }
    )
    st.write('## SOM feature plot')
    st.altair_chart(c, use_container_width=True)


def describe_classified_dataset(dataset_classified, assignments_central, assignments_neighbor, all_confidences_central=None, all_confidences_neighbor=None):
    results = {}

    assignments_central_df = pd.DataFrame(assignments_central)
    assignments_neighbor_df = pd.DataFrame(assignments_neighbor)

    # Process all confidence values if provided
    if all_confidences_central:
        all_confidences_central_df = pd.DataFrame(all_confidences_central)
        results['all_confidences_central'] = all_confidences_central_df
    else:
        results['all_confidences_central'] = pd.DataFrame()

    if all_confidences_neighbor:
        all_confidences_neighbor_df = pd.DataFrame(all_confidences_neighbor)
        results['all_confidences_neighbor'] = all_confidences_neighbor_df
    else:
        results['all_confidences_neighbor'] = pd.DataFrame()

    # Step 2: Count the number of classified rows
    results['total_classified_rows'] = dataset_classified['is_classified'].sum()

    # Total number of unclassified detections before assignment
    results['total_unclassified_before'] = len(dataset_classified)

    # Number of detections that were assigned a class using the same neuron
    results['num_assigned_central'] = len(assignments_central_df)

    # Number of detections that were assigned a class using neighbor neurons
    results['num_assigned_neighbor'] = len(assignments_neighbor_df)

    # Number of detections remaining unclassified
    results['num_unclassified_after'] = results['total_unclassified_before'] - \
        results['total_classified_rows']

    # Percentage of detections assigned
    results['percentage_assigned'] = (
        results['total_classified_rows'] / results['total_unclassified_before']) * 100

    if {'assigned_class_central', 'confidence_central'}.issubset(assignments_central_df.columns):
        # Distribution of assigned classes central
        results['assigned_class_counts_central'] = assignments_central_df['assigned_class_central'].value_counts()
        # Distribution of confidence levels central
        results['confidence_levels_central'] = assignments_central_df['confidence_central']
    else:
        results['assigned_class_counts_central'] = pd.Series(dtype='int64')
        results['confidence_levels_central'] = pd.Series(dtype='float64')

    if {'assigned_class_neighbor', 'confidence_neighbor'}.issubset(assignments_neighbor_df.columns):
        # Distribution of assigned classes neighbor
        results['assigned_class_counts_neighbor'] = assignments_neighbor_df['assigned_class_neighbor'].value_counts()
        # Distribution of confidence levels neighbor
        results['confidence_levels_neighbor'] = assignments_neighbor_df['confidence_neighbor']
    else:
        results['assigned_class_counts_neighbor'] = pd.Series(dtype='int64')
        results['confidence_levels_neighbor'] = pd.Series(dtype='float64')

    return results


def update_dataset_to_classify(dataset_toclassify, assignments_central, assignments_neighbor):
    if assignments_central != []:
        assignments_central_df = pd.DataFrame(assignments_central)
        dataset_toclassify = dataset_toclassify.merge(
            assignments_central_df[['id', 'assigned_class_central', 'confidence_central']], on='id', how='left')
    else:
        dataset_toclassify = pd.concat([dataset_toclassify, pd.DataFrame(
            columns=['assigned_class_central', 'confidence_central'])])

    if assignments_neighbor != []:
        assignments_neighbor_df = pd.DataFrame(assignments_neighbor)
        dataset_toclassify = dataset_toclassify.merge(
            assignments_neighbor_df[['id', 'assigned_class_neighbor', 'confidence_neighbor']], on='id', how='left')
    else:
        dataset_toclassify = pd.concat([dataset_toclassify, pd.DataFrame(
            columns=['assigned_class_neighbor', 'confidence_neighbor'])])

    if assignments_central != [] and assignments_neighbor != []:
        # Step 1: Check if either or both classifications are present
        dataset_toclassify['is_classified'] = dataset_toclassify.apply(
            lambda row: pd.notna(row['assigned_class_central']) or pd.notna(row['assigned_class_neighbor']), axis=1)
    elif assignments_central != []:
        dataset_toclassify['is_classified'] = dataset_toclassify.apply(
            lambda row: pd.notna(row['assigned_class_central']), axis=1)
    elif assignments_neighbor != []:
        dataset_toclassify['is_classified'] = dataset_toclassify.apply(
            lambda row: pd.notna(row['assigned_class_neighbor']), axis=1)
    else:
        dataset_toclassify = dataset_toclassify.reindex(
            columns=['is_classified'])

    return dataset_toclassify


def get_classification(som_map_id, dataset_toclassify, simbad_dataset, SIMBAD_classes, parameters_classification, dim, som):
    id_to_pos = {}
    for (i, j), ids in som_map_id.items():
        for id_ in ids:
            id_to_pos[id_] = (i, j)

    # Extract IDs and positions for classified detections
    classified_ids = simbad_dataset['id'].tolist()
    classified_positions = [id_to_pos[id_]
                            for id_ in classified_ids if id_ in id_to_pos]
    # Extract classes for classified detections
    classified_classes = simbad_dataset['main_type'].tolist()

    # Create a mapping from positions to classes
    neuron_class_map = defaultdict(list)
    for id_, pos, cls in zip(classified_ids, classified_positions, classified_classes):
        neuron_class_map[pos].append(cls)

    # Calculate class distributions per neuron
    neuron_class_distribution_central = {}
    neuron_class_distribution_neighbor = {}
    for pos, classes in neuron_class_map.items():
        # The top 3 classes shoud meet the minimum detection count,
        # and calculate the threshold based exclusively on these top 3 classes.
        class_counts = pd.Series(classes).value_counts()[:3]
        total = class_counts.sum()
        class_proportions = class_counts / total
        if total >= parameters_classification['min_detections_per_neuron']:
            neuron_class_distribution_central[pos] = {
                'total_detections': total,
                'class_counts': class_counts,
                'class_proportions': class_proportions
            }
        if total >= parameters_classification['min_detections_per_neighbor']:
            neuron_class_distribution_neighbor[pos] = {
                'total_detections': total,
                'class_counts': class_counts,
                'class_proportions': class_proportions
            }

    # Prepare a list to collect assignment results
    assignments_central = []
    assignments_neighbor = []

    # Lists to store all confidence values, regardless of threshold
    all_confidences_central = []
    all_confidences_neighbor = []

    for idx, row in dataset_toclassify.iterrows():
        id_ = row['id']
        if id_ not in id_to_pos:
            continue  # Skip if ID not in id_to_pos

        pos = som.winner(row[dataset_toclassify.columns[1:]].to_numpy())

        i, j = pos  # Extract grid coordinates

        # First, check if central neuron meets criteria
        if pos in neuron_class_distribution_central:
            class_info = neuron_class_distribution_central[pos]
            dominant_class = class_info['class_proportions'].idxmax()
            dominant_proportion = class_info['class_proportions'].max()

            # Store all confidence values for central classification
            all_confidences_central.append({
                'id': id_,
                'assigned_class_central': dominant_class,
                'confidence_central': dominant_proportion,
                'position': pos,
                'passed_threshold': dominant_proportion > parameters_classification['confidence_threshold']
            })

            if dominant_proportion > parameters_classification['confidence_threshold']:
                # Assign the dominant class
                assignments_central.append({
                    'id': id_,
                    'assigned_class_central': dominant_class,
                    'confidence_central': dominant_proportion,
                    'position': pos
                })

        # Check neighbors for all detections
        neighbors = get_hex_neighbors(i, j, dim, dim)
        neighbors_classes = []
        neighbor_confidences = {}

        for neighbor_pos in neighbors:
            if neighbor_pos in neuron_class_distribution_neighbor:
                neighbor_class_info = neuron_class_distribution_neighbor[neighbor_pos]
                neighbor_class_proportions = neighbor_class_info['class_proportions']
                neighbor_dominant_class = neighbor_class_proportions.idxmax()
                neighbor_dominant_proportion = neighbor_class_proportions.max()
                neighbor_total_detection = neighbor_class_info['total_detections']

                # Track the class and whether it meets the confidence threshold
                if neighbor_dominant_class not in neighbor_confidences:
                    neighbor_confidences[neighbor_dominant_class] = 0

                if (neighbor_dominant_proportion >= parameters_classification['neighbor_confidence_threshold'] and
                        neighbor_total_detection >= parameters_classification['min_detections_per_neighbor']):
                    neighbors_classes.append(neighbor_dominant_class)
                    neighbor_confidences[neighbor_dominant_class] += 1

        if neighbors_classes != []:
            # Find the class with the highest count
            assigned_neighbor_class = max(
                neighbor_confidences.items(), key=lambda x: x[1])[0]
            count_neighbor_class = neighbor_confidences[assigned_neighbor_class]

            # Store all neighbor confidence values
            all_confidences_neighbor.append({
                'id': id_,
                'assigned_class_neighbor': assigned_neighbor_class,
                'confidence_neighbor': count_neighbor_class,
                'position': pos,
                'passed_threshold': count_neighbor_class >= parameters_classification['neighbor_majority_threshold']
            })

            # Check if the count of matching neighbors meets the threshold
            if count_neighbor_class >= parameters_classification['neighbor_majority_threshold']:
                # Assign the class
                assignments_neighbor.append({
                    'id': id_,
                    'assigned_class_neighbor': assigned_neighbor_class,
                    'confidence_neighbor': count_neighbor_class,
                    'position': pos
                })

    return assignments_central, assignments_neighbor, all_confidences_central, all_confidences_neighbor


def validate_and_load_dataset(uploaded_file, expected_columns):
    """
    Validates and loads a dataset from an uploaded CSV file.

    Parameters:
    - uploaded_file: The uploaded file object.
    - expected_columns: A list of column names expected in the file, excluding the additional column.

    Returns:
    - The loaded pandas DataFrame if validation is successful, None otherwise.
    """
    # Load the dataset
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    # Validate column names (assuming the last column is the new feature to be projected)
    if not set(expected_columns).issubset(set(df.columns[:-1])):
        print("Column names do not match the expected columns.")
        return None

    # Check for empty values
    if df.isnull().values.any():
        print("The dataset contains empty values.")
        return None

    return df


def validate_and_load_dataset_class(uploaded_file, expected_columns):
    """
    Validates and loads a dataset from an uploaded CSV file.

    Parameters:
    - uploaded_file: The uploaded file object.
    - expected_columns: A list of column names expected in the file, excluding the additional column.

    Returns:
    - The loaded pandas DataFrame if validation is successful, None otherwise.
    """
    # Load the dataset
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    # Validate column names (assuming the last column is the new feature to be projected)
    if not set(expected_columns).issubset(set(df.columns[:-1])):
        print("Column names do not match the expected columns.")
        return None

    # Check for empty values
    if df.isnull().values.any():
        print("The dataset contains empty values.")
        return None

    return df[expected_columns]


def transform_and_normalize(dataset_toclassify, df_to_norm):
    # Check for each column if it exists in df_to_norm and transform it
    # For each column in df_to_norm, check if the corresponding column (without '_to_log_norm') exists in dataset_toclassify.
    # If so, apply log10 to that column in dataset_toclassify and overwrite it.

    # Rename columns to add '_to_log_norm' suffix if not already present
    dataset_toclassify.columns = [
        col if col.endswith('_to_log_norm') else f"{col}_to_log_norm"
        for col in dataset_toclassify.columns
    ]

    for col in df_to_norm.columns:
        if col in dataset_toclassify.columns:

            dataset_toclassify[col] = dataset_toclassify[col].replace(
                0, dataset_toclassify[col][dataset_toclassify[col] > 0].min()/10)

            combined = pd.concat([
                df_to_norm[[col]].reset_index(drop=True),
                dataset_toclassify[[col]].reset_index(drop=True)
            ], ignore_index=True)

            if col != "powlaw_gamma_to_log_norm":
                combined[col] = np.log(
                    combined[col])

            # Concatenate the column from dataset_toclassify to df_to_norm for normalization
            scalar = MinMaxScaler()
            scaled = scalar.fit_transform(
                combined.values.reshape(-1, 1)).flatten()
            # Only take the normalized values corresponding to dataset_toclassify (at the end)
            dataset_toclassify[col] = scaled[-len(dataset_toclassify):]

    dataset_toclassify.columns = dataset_toclassify.columns.str.replace(
        '_to_log_norm', '')

    return dataset_toclassify


def create_empty_hexagon_df(som):
    """
    Creates a DataFrame with empty hexagon data for the given SOM.

    Parameters
    ----------
    som : MiniSom
        The trained SOM.

    Returns
    -------
    pd.DataFrame
        DataFrame with hexagon positions ready for visualization.
    """
    som_shape = som.get_weights().shape
    # Create a grid of neuron positions
    x_range = range(1, som_shape[0] + 1)
    y_range = range(1, som_shape[1] + 1)

    # Create empty DataFrame
    hexagon_data = []
    for y in y_range:
        for x in x_range:
            hexagon_data.append({
                'x': x,
                'y': y,
                'value': 0  # Empty value
            })

    hexagon_df = pd.DataFrame(hexagon_data)

    # Set x and y as float (important for proper spacing)
    hexagon_df['x'] = hexagon_df['x'].astype(float)
    hexagon_df['y'] = hexagon_df['y'].astype(float)

    # Order the dataframe using x and y (important for consistent layout)
    hexagon_df = hexagon_df.sort_values(by=['x', 'y'])
    hexagon_df = hexagon_df.reset_index(drop=True)

    return hexagon_df


def count_unique_main_types_per_neuron(som, X, raw_df, main_types=None):
    """
    Count the number of unique main types in each neuron of the SOM, 
    filtered by the specified main types. Also calculates the dominance
    percentage of the most common main type in each neuron.

    Parameters
    ----------
    som : MiniSom
        The trained Self-Organizing Map.
    X : numpy.ndarray
        The feature data, where each row is a data point.
    raw_df : pandas.DataFrame
        The raw data containing 'name' and 'main_type' information about each detection.
    main_types : list, optional
        List of main types to include in the count. If None, all types are included.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with x, y coordinates, the count of unique main types per neuron,
        and the dominance percentage of the most common main type.
    """
    # Dictionary to store main types and their counts for each neuron
    neuron_to_main_types = {}

    # Map each data point to its winning neuron and track by main type
    for i, x in enumerate(X):
        # Get main type
        main_type = raw_df.iloc[i]['main_type']

        # Skip if no main type or if not in the selected types
        if pd.isna(main_type):
            continue

        if main_types is not None and main_type not in main_types:
            continue

        # Get the winning neuron
        winner = som.winner(x)

        # Initialize dictionary for this neuron if not already done
        if winner not in neuron_to_main_types:
            neuron_to_main_types[winner] = {}

        # Add this main type to the dictionary for this neuron and increment its count
        if main_type not in neuron_to_main_types[winner]:
            neuron_to_main_types[winner][main_type] = 0
        neuron_to_main_types[winner][main_type] += 1

    # Convert to a dataframe format suitable for visualization
    counts_data = []
    som_shape = som.get_weights().shape

    # Calculate statistics for each neuron
    for y in range(1, som_shape[1] + 1):
        for x in range(1, som_shape[0] + 1):
            neuron = (x-1, y-1)  # Convert to 0-indexed for the SOM

            if neuron in neuron_to_main_types and len(neuron_to_main_types[neuron]) > 0:
                # Number of unique main types
                unique_count = len(neuron_to_main_types[neuron])

                # Calculate dominance percentage of most common main type
                type_counts = neuron_to_main_types[neuron]
                total_detections = sum(type_counts.values())
                most_common_type = max(type_counts.items(), key=lambda x: x[1])
                dominance_pct = int(
                    round((most_common_type[1] / total_detections) * 100))
            else:
                unique_count = 0
                dominance_pct = 0

            counts_data.append({
                'x': float(x),
                'y': float(y),
                'value': unique_count,
                'dominance_pct': dominance_pct
            })

    df = pd.DataFrame(counts_data)

    # Order the dataframe using x and y for consistent layout
    df = df.sort_values(by=['x', 'y'])
    df = df.reset_index(drop=True)

    return df


def plot_empty_hexagons(som, X=None, raw_df=None, main_types=None):
    """
    Displays hexagons in a SOM layout, colored by the number of unique main types per neuron.
    Shows the dominance percentage of the most common main type in each neuron.

    Parameters
    ----------
    som : MiniSom
        The trained SOM.
    X : numpy.ndarray, optional
        The feature data, where each row is a data point.
    raw_df : pandas.DataFrame, optional
        The raw data containing 'name' and other information about each detection.
    main_types : list, optional
        List of main types to include in the count. If None, all types are included.
    """
    if X is not None and raw_df is not None:
        # Count unique main types per neuron and calculate dominance percentages
        hexagon_df = count_unique_main_types_per_neuron(
            som, X, raw_df, main_types)
    else:
        # Use empty hexagons if data not provided
        hexagon_df = create_empty_hexagon_df(som)
        hexagon_df['dominance_pct'] = 0

    min_x = hexagon_df['x'].min()
    max_x = hexagon_df['x'].max()
    min_y = hexagon_df['y'].min()
    max_y = hexagon_df['y'].max()

    # Get appropriate size for hexagons based on SOM dimensions
    index = np.where(new_dimensions == som.get_weights().shape[0])[0][0]
    size = new_sizes[index]

    # Hexagon shape definition
    hexagon = "M0,-2.3094010768L2,-1.1547005384 2,1.1547005384 0,2.3094010768 -2,1.1547005384 -2,-1.1547005384Z"

    # Create a new column for coloring - empty string for 0, category values for others
    hexagon_df['value_cat'] = hexagon_df['value'].apply(
        lambda x: '' if x == 0 else str(x))

    # Create legend domain and range - excluding zero (empty hexagons)
    unique_nonzero_values = sorted(
        [str(v) for v in hexagon_df['value'].unique() if v > 0])

    # Create a base chart for consistent layout
    base = alt.Chart(hexagon_df).encode(
        x=alt.X('xFeaturePos:Q', title='', scale=alt.Scale(domain=[min_x-1, max_x+1])).axis(
            grid=False, tickOpacity=0, domainOpacity=0),
        y=alt.Y('y:Q', sort=alt.EncodingSortField(
            'y', order='descending'), title='', scale=alt.Scale(domain=[min_y-1, max_y+1])
        ).axis(grid=False, labelPadding=20, tickOpacity=0, domainOpacity=0)
    ).transform_calculate(
        xFeaturePos='(datum.y%2)/2 + datum.x-1'  # Shifted 0.5 units left
    ).properties(
        height=700,
        width=600,
    )

    # Separate charts for zero and non-zero values
    if len(hexagon_df[hexagon_df['value'] == 0]) > 0:
        # Empty hexagons (count = 0)
        chart_zeros = base.transform_filter(
            alt.datum.value == 0
        ).mark_point(shape=hexagon, size=size**2).encode(
            stroke=alt.value('black'),
            strokeWidth=alt.value(1.0),
            fill=alt.value('white')  # Empty hexagons are white
        )
    else:
        chart_zeros = None

    if len(hexagon_df[hexagon_df['value'] > 0]) > 0:
        # Non-zero hexagons (colored by count)
        chart_nonzeros = base.transform_filter(
            alt.datum.value > 0
        ).mark_point(shape=hexagon, size=size**2).encode(
            color=alt.Color(
                'value_cat:N',
                scale=alt.Scale(scheme='category10',
                                domain=unique_nonzero_values),
                legend=alt.Legend(title="Number of Main Types")
            ),
            fill=alt.Color(
                'value_cat:N',
                scale=alt.Scale(scheme='category10',
                                domain=unique_nonzero_values),
                legend=alt.Legend(title="Number of Main Types")
            ),
            stroke=alt.value('black'),
            strokeWidth=alt.value(1.0)
        )

        # Adjust text size based on SOM dimensions
        # Ensure text is readable but not too large
        text_size = max(8, min(size, 14))

        # Add dominance percentage text inside non-zero hexagons
        text_nonzeros = base.transform_filter(
            alt.datum.value > 0
        ).mark_text(
            fontSize=text_size,
            fontWeight='bold',
            color='white',
            baseline='middle',
            align='center'
        ).encode(
            text=alt.Text('dominance_pct:Q', format='d')  # Display as integer
        )
    else:
        chart_nonzeros = None
        text_nonzeros = None

    # Combine the charts
    charts_to_layer = []
    if chart_zeros is not None:
        charts_to_layer.append(chart_zeros)
    if chart_nonzeros is not None:
        charts_to_layer.append(chart_nonzeros)
    if text_nonzeros is not None:
        charts_to_layer.append(text_nonzeros)

    c = alt.layer(*charts_to_layer)

    # Configure the chart
    c = c.configure_view(
        strokeWidth=0
    ).configure_legend(
        symbolStrokeWidth=1.0,
        symbolSize=size**2,
        orient='bottom'
    )

    st.altair_chart(c, use_container_width=True)
    # Remove the duplicate chart displays


def category_plot_sources_hex_with_patterns(_map, flip=True, custom_colors=None, cluster_mapping=None, cluster_border_colors=None):
    """
    Alternative hexagonal category plot with pattern-based cluster visualization.
    Uses opacity and stroke-dash patterns to distinguish clusters.
    """
    if flip:
        _map = list(map(list, zip(*_map)))

    # Convert map to winning categories (same as original function)
    winning_categories = []
    for idx_outer, sublist_outer in enumerate(_map):
        for idx_inner, sublist in enumerate(sublist_outer):
            winning_categories.append(
                [int(idx_outer+1), int(idx_inner+1), max(set(sublist), key=sublist.count)])

    winning_categories = np.array(winning_categories)
    pd_winning_categories = pd.DataFrame(
        winning_categories, columns=['y', 'x', 'source'])

    min_x = pd_winning_categories['x'].min()
    max_x = pd_winning_categories['x'].max()
    min_y = pd_winning_categories['y'].min()
    max_y = pd_winning_categories['y'].max()

    pd_winning_categories = pd_winning_categories.dropna()
    pd_winning_categories['x'] = pd_winning_categories['x'].astype(float)
    pd_winning_categories['y'] = pd_winning_categories['y'].astype(float)
    pd_winning_categories = pd_winning_categories.sort_values(by=['x', 'y'])
    pd_winning_categories = pd_winning_categories.reset_index(drop=True)

    index = np.where(new_dimensions == len(_map))[0][0]
    size = new_sizes[index]

    # Prepare color scale
    if custom_colors:
        domain = list(custom_colors.keys())
        range_ = list(custom_colors.values())
        color_scale = alt.Scale(domain=domain, range=range_)
    else:
        color_scale = alt.Scale(scheme='lightmulti')

    hexagon = "M0,-2.3094010768L2,-1.1547005384 2,1.1547005384 0,2.3094010768 -2,1.1547005384 -2,-1.1547005384Z"

    if cluster_mapping is not None and cluster_border_colors is not None:
        # Add cluster ID
        pd_winning_categories['cluster_id'] = pd_winning_categories.apply(
            lambda row: cluster_mapping.get((row['x']-1, row['y']-1), -1),
            axis=1
        )

        # Create different opacity levels for each cluster
        cluster_ids = sorted(
            [id for id in cluster_mapping.values() if id >= 0])
        opacity_levels = [0.3 + (i * 0.7 / len(cluster_ids))
                          for i in range(len(cluster_ids))]

        base_encoding = {
            'x': alt.X('xFeaturePos:Q', title='', scale=alt.Scale(domain=[min_x-1, max_x+1])).axis(
                grid=False, tickOpacity=0, domainOpacity=0),
            'y': alt.Y('y:Q', sort=alt.EncodingSortField(
                'y', order='descending'), title='', scale=alt.Scale(domain=[min_y-1, max_y+1])
            ).axis(grid=False, labelPadding=20, tickOpacity=0, domainOpacity=0)
        }

        # Main fill chart
        fill_chart = alt.Chart(pd_winning_categories).mark_point(
            shape=hexagon,
            size=size**2
        ).encode(
            **base_encoding,
            color=alt.Color('source:N', scale=color_scale),
            fill=alt.Color('source:N', scale=color_scale).legend(
                orient='bottom'),
            stroke=alt.value('black'),
            strokeWidth=alt.value(1.0),
            tooltip=['source:N', 'x:Q', 'y:Q', alt.Tooltip(
                'cluster_id:N', title='Cluster')]
        )

        # Cluster highlight overlay with varying opacity
        cluster_overlay = alt.Chart(pd_winning_categories).mark_point(
            shape=hexagon,
            size=size**2,
            filled=False
        ).encode(
            **base_encoding,
            stroke=alt.condition(
                alt.datum.cluster_id >= 0,
                alt.Color('cluster_id:N',
                          scale=alt.Scale(domain=list(cluster_border_colors.keys()),
                                          range=list(cluster_border_colors.values())),
                          legend=alt.Legend(title="Clusters", orient='bottom')),
                alt.value(None)
            ),
            strokeWidth=alt.condition(
                alt.datum.cluster_id >= 0,
                alt.value(8),  # Extra thick border for cluster distinction
                alt.value(0)
            ),
            strokeDash=alt.condition(
                alt.datum.cluster_id >= 0,
                alt.value([5, 5]),  # Dashed line pattern
                alt.value([])
            )
        )

        c = alt.layer(fill_chart, cluster_overlay).transform_calculate(
            xFeaturePos='(datum.y%2)/2 + datum.x-.5'
        ).properties(
            height=700,
            width=600,
        ).configure_view(
            strokeWidth=0
        ).configure_legend(
            symbolStrokeWidth=1.0,
            symbolSize=size**2
        )
    else:
        # Original chart without clustering
        c = alt.Chart(pd_winning_categories).mark_point(shape=hexagon, size=size**2).encode(
            x=alt.X('xFeaturePos:Q', title='', scale=alt.Scale(domain=[min_x-1, max_x+1])).axis(
                grid=False, tickOpacity=0, domainOpacity=0),
            y=alt.Y('y:Q', sort=alt.EncodingSortField(
                'y', order='descending'), title='', scale=alt.Scale(domain=[min_y-1, max_y+1])
            ).axis(grid=False, labelPadding=20, tickOpacity=0, domainOpacity=0),
            color=alt.Color('source:N', scale=color_scale),
            fill=alt.Color('source:N', scale=color_scale).legend(
                orient='bottom'),
            stroke=alt.value('black'),
            # Default stroke width for non-cluster mode
            strokeWidth=alt.value(1.0),
            tooltip=['source:N', 'x:Q', 'y:Q']
        ).transform_calculate(
            xFeaturePos='(datum.y%2)/2 + datum.x-.5'
        ).properties(
            height=700,
            width=600,
        ).configure_view(
            strokeWidth=0
        ).configure_legend(
            symbolStrokeWidth=1.0,
            symbolSize=size**2
        )

    st.write('## SOM category plot (Pattern-based Clusters)')
    st.altair_chart(c, use_container_width=True)


def category_plot_sources_hex_with_size_variation(_map, flip=True, custom_colors=None, cluster_mapping=None, cluster_border_colors=None):
    """
    Alternative hexagonal category plot with size-based cluster visualization.
    Uses different hexagon sizes to distinguish clusters.
    """
    if flip:
        _map = list(map(list, zip(*_map)))

    # Convert map to winning categories (same as original function)
    winning_categories = []
    for idx_outer, sublist_outer in enumerate(_map):
        for idx_inner, sublist in enumerate(sublist_outer):
            winning_categories.append(
                [int(idx_outer+1), int(idx_inner+1), max(set(sublist), key=sublist.count)])

    winning_categories = np.array(winning_categories)
    pd_winning_categories = pd.DataFrame(
        winning_categories, columns=['y', 'x', 'source'])

    min_x = pd_winning_categories['x'].min()
    max_x = pd_winning_categories['x'].max()
    min_y = pd_winning_categories['y'].min()
    max_y = pd_winning_categories['y'].max()

    pd_winning_categories = pd_winning_categories.dropna()
    pd_winning_categories['x'] = pd_winning_categories['x'].astype(float)
    pd_winning_categories['y'] = pd_winning_categories['y'].astype(float)
    pd_winning_categories = pd_winning_categories.sort_values(by=['x', 'y'])
    pd_winning_categories = pd_winning_categories.reset_index(drop=True)

    index = np.where(new_dimensions == len(_map))[0][0]
    base_size = new_sizes[index]

    # Prepare color scale
    if custom_colors:
        domain = list(custom_colors.keys())
        range_ = list(custom_colors.values())
        color_scale = alt.Scale(domain=domain, range=range_)
    else:
        color_scale = alt.Scale(scheme='lightmulti')

    hexagon = "M0,-2.3094010768L2,-1.1547005384 2,1.1547005384 0,2.3094010768 -2,1.1547005384 -2,-1.1547005384Z"

    if cluster_mapping is not None and cluster_border_colors is not None:
        # Add cluster ID
        pd_winning_categories['cluster_id'] = pd_winning_categories.apply(
            lambda row: cluster_mapping.get((row['x']-1, row['y']-1), -1),
            axis=1
        )

        # Create different sizes for each cluster
        cluster_ids = sorted(
            [id for id in cluster_mapping.values() if id >= 0])
        size_multipliers = [0.7 + (i * 0.6 / len(cluster_ids))
                            for i in range(len(cluster_ids))]

        # Add size column based on cluster
        def get_size_for_cluster(cluster_id):
            if cluster_id >= 0 and cluster_id < len(size_multipliers):
                return base_size**2 * size_multipliers[cluster_id]
            return base_size**2

        pd_winning_categories['hex_size'] = pd_winning_categories['cluster_id'].apply(
            get_size_for_cluster)

        base_encoding = {
            'x': alt.X('xFeaturePos:Q', title='', scale=alt.Scale(domain=[min_x-1, max_x+1])).axis(
                grid=False, tickOpacity=0, domainOpacity=0),
            'y': alt.Y('y:Q', sort=alt.EncodingSortField(
                'y', order='descending'), title='', scale=alt.Scale(domain=[min_y-1, max_y+1])
            ).axis(grid=False, labelPadding=20, tickOpacity=0, domainOpacity=0)
        }

        # Main chart with varying sizes
        c = alt.Chart(pd_winning_categories).mark_point(
            shape=hexagon
        ).encode(
            **base_encoding,
            size=alt.Size('hex_size:Q', scale=alt.Scale(
                range=[base_size**2 * 0.7, base_size**2 * 1.3]), legend=None),
            color=alt.Color('source:N', scale=color_scale),
            fill=alt.Color('source:N', scale=color_scale).legend(
                orient='bottom'),
            stroke=alt.condition(
                alt.datum.cluster_id >= 0,
                alt.Color('cluster_id:N',
                          scale=alt.Scale(domain=list(cluster_border_colors.keys()),
                                          range=list(cluster_border_colors.values())),
                          legend=alt.Legend(title="Clusters", orient='bottom')),
                alt.value('black')
            ),
            strokeWidth=alt.condition(
                alt.datum.cluster_id >= 0,
                alt.value(3),
                alt.value(1)
            ),
            tooltip=['source:N', 'x:Q', 'y:Q', alt.Tooltip(
                'cluster_id:N', title='Cluster')]
        ).transform_calculate(
            xFeaturePos='(datum.y%2)/2 + datum.x-.5'
        ).properties(
            height=700,
            width=600,
        ).configure_view(
            strokeWidth=0
        ).configure_legend(
            symbolStrokeWidth=1.0,
            symbolSize=base_size**2
        )
    else:
        # Original chart without clustering
        c = alt.Chart(pd_winning_categories).mark_point(shape=hexagon, size=base_size**2).encode(
            x=alt.X('xFeaturePos:Q', title='', scale=alt.Scale(domain=[min_x-1, max_x+1])).axis(
                grid=False, tickOpacity=0, domainOpacity=0),
            y=alt.Y('y:Q', sort=alt.EncodingSortField(
                'y', order='descending'), title='', scale=alt.Scale(domain=[min_y-1, max_y+1])
            ).axis(grid=False, labelPadding=20, tickOpacity=0, domainOpacity=0),
            color=alt.Color('source:N', scale=color_scale),
            fill=alt.Color('source:N', scale=color_scale).legend(
                orient='bottom'),
            stroke=alt.value('black'),
            # Default stroke width for non-cluster mode
            strokeWidth=alt.value(1.0),
            tooltip=['source:N', 'x:Q', 'y:Q']
        ).transform_calculate(
            xFeaturePos='(datum.y%2)/2 + datum.x-.5'
        ).properties(
            height=700,
            width=600,
        ).configure_view(
            strokeWidth=0
        ).configure_legend(
            symbolStrokeWidth=1.0,
            symbolSize=base_size**2
        )

    st.write('## SOM category plot (Size-based Clusters)')
    st.altair_chart(c, use_container_width=True)
