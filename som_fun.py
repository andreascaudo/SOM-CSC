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


def is_string(var):
    is_string = True
    for sublist in var:
        for sublist2 in sublist:
            for value in sublist2:
                if value is not None and not isinstance(value, str):
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
    st.write('SOM trained in time:', end - start, 'seconds')
    return som


def get_iterations_index(X, dim_number, features, sigma, learning_rate, max_iter=1000, errors_bar=None):

    som = MiniSom(dim_number, dim_number, features, sigma=sigma,
                  learning_rate=learning_rate, neighborhood_function='gaussian')

    q_error = []
    t_error = []

    for i in range(max_iter):
        # get percentage of the progress from tqdm
        errors_bar.progress(
            i/max_iter, text="Getting quantization and topographic errors")
        rand_i = np.random.randint(len(X))
        som.update(X[rand_i], som.winner(X[rand_i]), i, max_iter)
        q_error.append(som.quantization_error(X))
        t_error.append(som.topographic_error(X))

    return q_error, t_error


def plot_errors(q_error, t_error, iterations):
    st.write('## Quantization error and Topographic error')
    # Plot using st the quantization error and the topographic error togheter
    errors_data = pd.DataFrame(
        {'Iterations': range(1, iterations+1), 'Quantization error': q_error, 'Topographic error': t_error})
    errors_data_melted = errors_data.melt(
        'Iterations', var_name='Errors', value_name='Value')

    c = alt.Chart(errors_data_melted).mark_line().encode(
        x=alt.X('Iterations', title='Iterations [-]'),
        y=alt.Y('Value', title='Error [-]'),
        color='Errors'
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(c, use_container_width=True)


def get_dispersion(name_ids, id_to_pos, min_detections):
    dispersion_list = []
    for source_name, source_ids in name_ids.items():
        # Retrieve positions of the source's detections
        positions = [id_to_pos[id_] for id_ in source_ids if id_ in id_to_pos]

        if len(positions) < min_detections:
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
        dispersion_list.append((source_name, dispersion))

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


def plot_rectangular_u_matrix(som, color_type='linear'):
    u_matrix = som.distance_map().T
    u_matrix = pd.DataFrame(
        u_matrix, columns=range(1, len(u_matrix)+1), index=range(1, len(u_matrix)+1))
    u_matrix = u_matrix.melt(
        var_name='x', value_name='value', ignore_index=False)
    u_matrix = u_matrix.reset_index()
    u_matrix = u_matrix.rename(columns={'index': 'y'})
    st.u_matrix = u_matrix

    c = alt.Chart(u_matrix).mark_rect().encode(
        x=alt.X('x:O', title=''),
        y=alt.Y('y:O', sort=alt.EncodingSortField(
            'y', order='descending'), title=''),
        color=alt.Color(
            'value:Q', scale=alt.Scale(type=color_type, scheme='lightmulti'))
    ).properties(
        width=600,
        height=600
    )
    st.altair_chart(c, use_container_width=True)


dimensions = np.array([6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
sizes = np.array([48, 42, 37, 33, 29, 15, 10, 8, 6, 5, 5, 4, 4, 3.4])

new_dimensions = np.arange(6, 101, 1)
new_sizes = np.interp(new_dimensions, dimensions, sizes)


def plot_u_matrix_hex(som, color_type='linear'):
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
            'value:Q', scale=alt.Scale(scheme='lightmulti')),
        fill=alt.Fill('value:Q', scale=alt.Scale(type=color_type, domain=(
            min_value, max_value), scheme='lightmulti')).legend(orient='bottom'),
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


def plot_activation_response(som, X_index, color_type='linear', plot=True):
    X = X_index[:, 1:]
    activation_map = som.activation_response(X)
    activation_map = pd.DataFrame(
        activation_map, columns=range(1, len(activation_map)+1), index=range(1, len(activation_map)+1))
    activation_map = activation_map.melt(
        var_name='x', value_name='value', ignore_index=False)
    activation_map = activation_map.reset_index()
    activation_map = activation_map.rename(columns={'index': 'y'})

    min_value = activation_map['value'][activation_map['value'] > 0].min()
    max_value = activation_map['value'].max()

    if plot:
        c = alt.Chart(activation_map).mark_rect().encode(
            x=alt.X('x:O', title=''),
            y=alt.Y('y:O', sort=alt.EncodingSortField(
                'y'), title=''),
            color=alt.Color(
                'value:Q',
                scale=alt.Scale(type=color_type, domain=(min_value, max_value), scheme='lightmulti'))
        ).properties(
            width=600,
            height=600
        )
        st.altair_chart(c, use_container_width=True)

    return download_activation_response(som, X_index)


def plot_activation_response_hex(som, X_index, color_type='linear', plot=True):
    X = X_index[:, 1:]
    activation_map = som.activation_response(X)
    activation_map = pd.DataFrame(
        activation_map, columns=range(1, len(activation_map)+1), index=range(1, len(activation_map)+1))
    activation_map = activation_map.melt(
        var_name='x', value_name='value', ignore_index=False)
    activation_map = activation_map.reset_index()
    activation_map = activation_map.rename(columns={'index': 'y'})

    min_x = activation_map['x'].min()
    max_x = activation_map['x'].max()
    min_y = activation_map['y'].min()
    max_y = activation_map['y'].max()

    min_value = activation_map['value'][activation_map['value'] > 0].min()
    max_value = activation_map['value'].max()

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
                'value:Q', scale=alt.Scale(scheme='lightmulti')),
            fill=alt.Fill('value:Q', scale=alt.Scale(type=color_type, domain=(
                min_value, max_value), scheme='lightmulti')).legend(orient='bottom'),
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


def feature_space_map_plot(weights, color_type='linear'):
    # plot the mean of the weights across lass dimension
    mean_weights = np.mean(weights, axis=2)
    mean_weights = pd.DataFrame(mean_weights, columns=range(
        1, len(mean_weights)+1), index=range(1, len(mean_weights)+1))
    mean_weights = mean_weights.melt(
        var_name='x', value_name='value', ignore_index=False)
    mean_weights = mean_weights.reset_index()
    mean_weights = mean_weights.rename(columns={'index': 'y'})

    min_value = mean_weights['value'][mean_weights['value'] > 0].min()
    max_value = mean_weights['value'].max()

    c = alt.Chart(mean_weights).mark_rect().encode(
        x=alt.X('x:O', title=''),
        y=alt.Y('y:O', sort=alt.EncodingSortField(
            'y', order='descending'), title=''),
        color=alt.Color(
            'value:Q',
            scale=alt.Scale(type=color_type, domain=(min_value, max_value), scheme='lightmulti'))
    ).properties(
        width=600,
        height=600
    )
    st.altair_chart(c, use_container_width=True)


def feature_space_map_plot_hex(weights, color_type='linear'):
    # plot the mean of the weights across lass dimension
    mean_weights = np.mean(weights, axis=2)
    mean_weights = pd.DataFrame(mean_weights, columns=range(
        1, len(mean_weights)+1), index=range(1, len(mean_weights)+1))
    mean_weights = mean_weights.melt(
        var_name='x', value_name='value', ignore_index=False)
    mean_weights = mean_weights.reset_index()
    mean_weights = mean_weights.rename(columns={'index': 'y'})

    min_x = mean_weights['x'].min()
    max_x = mean_weights['x'].max()
    min_y = mean_weights['y'].min()
    max_y = mean_weights['y'].max()

    min_value = mean_weights['value'][mean_weights['value'] > 0].min()
    max_value = mean_weights['value'].max()

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
            'value:Q', scale=alt.Scale(scheme='lightmulti')),
        fill=alt.Fill('value:Q', scale=alt.Scale(type=color_type, domain=(
            min_value, max_value), scheme='lightmulti')).legend(orient='bottom'),
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


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def scatter_plot_clustering_hex(som, X, GMM_cluster_labels):
    w_x = []
    w_y = []
    for cnt, x in enumerate(X):
        # getting the winner
        w = som.winner(x)
        # place a marker on the winning position for the sample xx
        wx, wy = som.convert_map_to_euclidean(w)
        # wy = wy * np.sqrt(3) / 2
        w_x.append(wx)
        w_y.append(wy)

    w_x = np.array(w_x, dtype=float)
    w_y = np.array(w_y, dtype=float)

    # st.write(set(GMM_cluster_labels.dropna()))
    for c in GMM_cluster_labels:
        idx_target = GMM_cluster_labels == c
        num_points = np.sum(idx_target)

        # Generate random angles and distances
        random_angles = np.random.uniform(0, 2 * np.pi, num_points)
        random_distances = np.sqrt(np.random.uniform(
            0, 1, num_points)) * 0.5

        # Convert polar coordinates to Cartesian coordinates
        w_x[idx_target] += random_distances * np.cos(random_angles)
        w_y[idx_target] += random_distances * np.sin(random_angles)

    scatter_chart_sample_df = pd.DataFrame(
        {'w_y': w_y, 'w_x': w_x, 'cluster': GMM_cluster_labels})

    min_x = scatter_chart_sample_df['w_x'].min()
    max_x = scatter_chart_sample_df['w_x'].max()
    min_y = scatter_chart_sample_df['w_y'].min()
    max_y = scatter_chart_sample_df['w_y'].max()

    size = new_sizes[np.where(
        new_dimensions == som.get_weights().shape[0])[0][0]]

    hexagon = "M0,-2.3094010768L2,-1.1547005384 2,1.1547005384 0,2.3094010768 -2,1.1547005384 -2,-1.1547005384Z"
    scatter_chart_sample = alt.Chart(scatter_chart_sample_df).mark_circle().encode(
        x=alt.X('w_x:Q', title='', scale=alt.Scale(
            domain=[min_x-1, max_x+1])).axis(
            grid=False, tickOpacity=0, domainOpacity=0),
        y=alt.Y('w_y:Q', title='', scale=alt.Scale(
            domain=[min_y-1, max_y+1])).axis(
            grid=False, tickOpacity=0, domainOpacity=0),
        color=alt.Color('cluster:N', scale=alt.Scale(
            scheme='lightmulti')).legend(orient='bottom'),
        strokeWidth=alt.value(1.0)
    ).properties(
        height=700,
        width=600
    ).configure_view(
        strokeWidth=0
    )
    st.write('## SOM scatter plot')
    st.altair_chart(scatter_chart_sample, use_container_width=True)


def scatter_plot_clustering(som, X, GMM_cluster_labels):

    w_x, w_y = zip(*[som.winner(d) for d in X])
    w_x = np.array(w_x, dtype=float)
    w_y = np.array(w_y, dtype=float)

    for c in np.unique(GMM_cluster_labels):
        idx_target = GMM_cluster_labels == c
        w_x[idx_target] += .5 + \
            (np.random.rand(np.sum(idx_target))-.5)*.8
        w_y[idx_target] += +.5 + \
            (np.random.rand(np.sum(idx_target))-.5)*.8

    scatter_chart_sample_df = pd.DataFrame(
        {'w_y': w_y, 'w_x': w_x, 'cluster': GMM_cluster_labels})

    min_x = scatter_chart_sample_df['w_x'].min()
    max_x = scatter_chart_sample_df['w_x'].max()
    min_y = scatter_chart_sample_df['w_y'].min()
    max_y = scatter_chart_sample_df['w_y'].max()

    scatter_chart_sample = alt.Chart(scatter_chart_sample_df).mark_circle().encode(
        x=alt.X('w_x', title='', scale=alt.Scale(
            domain=[min_x-1, max_x+1])).axis(
            grid=False),
        y=alt.Y('w_y', title='', scale=alt.Scale(
            domain=[min_y-1, max_y+1])).axis(
            grid=False),
        color=alt.Color('cluster:N', scale=alt.Scale(scheme='lightmulti'))
    ).properties(
        width=600,
        height=600
    )

    st.write('## SOM scatter plot')
    st.altair_chart(scatter_chart_sample, use_container_width=True)


def scatter_plot_sources(som, sources, raw_df, X, column_name):
    # get the index where the sources are in the raw_df and get rows from X
    idx = raw_df.index[raw_df[column_name].isin(sources)]
    X_sources_name = raw_df[column_name][idx]
    X_sources = X[idx]

    w_x, w_y = zip(*[som.winner(d) for d in X_sources])

    w_x = np.array(w_x, dtype=float)
    w_y = np.array(w_y, dtype=float)

    for c in sources:
        idx_target = np.array(X_sources_name) == c
        w_x[idx_target] += .5 + \
            (np.random.rand(np.sum(idx_target))-.5)*.8
        w_y[idx_target] += +.5 + \
            (np.random.rand(np.sum(idx_target))-.5)*.8

    scatter_chart_sample_df = pd.DataFrame(
        {'w_y': w_y, 'w_x': w_x, 'sources': X_sources_name})

    dimension = som.get_weights().shape[0]
    min_x = 0
    max_x = dimension
    min_y = 0
    max_y = dimension

    scatter_chart_sample = alt.Chart(scatter_chart_sample_df).mark_circle().encode(
        x=alt.X('w_x', title='', scale=alt.Scale(
            domain=[min_x-1, max_x+1])).axis(
            grid=False),
        y=alt.Y('w_y', title='', scale=alt.Scale(
            domain=[min_y-1, max_y+1])).axis(
            grid=False),
        color=alt.Color('sources:N', scale=alt.Scale(
            scheme='lightmulti')).legend(orient='bottom')
    ).properties(
        height=700,
        width=600
    )

    st.write('## SOM scatter plot')
    st.altair_chart(scatter_chart_sample, use_container_width=True)


def scatter_plot_sources_hex(som, sources, raw_df, X, column_name):
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

        # Generate random angles and distances
        random_angles = np.random.uniform(0, 2 * np.pi, num_points)
        random_distances = np.sqrt(np.random.uniform(
            0, 1, num_points)) * 0.5

        # Convert polar coordinates to Cartesian coordinates
        w_x[idx_target] += random_distances * np.cos(random_angles)
        w_y[idx_target] += random_distances * np.sin(random_angles)

    scatter_chart_sample_df = pd.DataFrame(
        {'w_y': w_y, 'w_x': w_x, 'sources': X_sources_name})

    dimension = som.get_weights().shape[0]
    min_x = 0
    max_x = dimension
    min_y = 0
    max_y = dimension

    size = new_sizes[np.where(
        new_dimensions == som.get_weights().shape[0])[0][0]]

    hexagon = "M0,-2.3094010768L2,-1.1547005384 2,1.1547005384 0,2.3094010768 -2,1.1547005384 -2,-1.1547005384Z"
    scatter_chart_sample = alt.Chart(scatter_chart_sample_df).mark_circle().encode(
        x=alt.X('w_x:Q', title='', scale=alt.Scale(
            domain=[min_x-1, max_x+1])).axis(
            grid=False, tickOpacity=0
        ),
        y=alt.Y('w_y:Q', title='', scale=alt.Scale(
            domain=[min_y-1, max_y+1])).axis(
            grid=False, tickOpacity=0
        ),
        color=alt.Color('sources:N', scale=alt.Scale(
            scheme='lightmulti')).legend(orient='bottom')
    ).properties(
        height=700,
        width=600,
    ).configure_view(
        strokeWidth=0
    )

    st.write('## SOM scatter plot')
    st.altair_chart(scatter_chart_sample, use_container_width=True)


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


def category_plot_sources(_map, flip=True):

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
            winning_categories.append(
                [int(idx_outer+1), int(idx_inner+1), max(set(sublist), key=sublist.count)])

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

    scatter_chart_sample = alt.Chart(pd_winning_categories).mark_rect().encode(
        x=alt.X('w_x:O', title='', scale=alt.Scale(domain=tick_x)),
        y=alt.Y('w_y:O', sort=alt.EncodingSortField(
            'w_y', order='descending'), title='', scale=alt.Scale(domain=tick_y)),
        color=alt.Color(
            'source:N', scale=alt.Scale(scheme='lightmulti'), legend=alt.Legend(orient='bottom'))
    ).properties(
        height=700,
        width=600
    )
    st.write('## SOM category plot')
    st.altair_chart(scatter_chart_sample, use_container_width=True)


def category_plot_sources_hex(_map, flip=True):
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
            winning_categories.append(
                [int(idx_outer+1), int(idx_inner+1), max(set(sublist), key=sublist.count)])

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

    hexagon = "M0,-2.3094010768L2,-1.1547005384 2,1.1547005384 0,2.3094010768 -2,1.1547005384 -2,-1.1547005384Z"
    c = alt.Chart(pd_winning_categories).mark_point(shape=hexagon, size=size**2).encode(
        x=alt.X('xFeaturePos:Q', title='', scale=alt.Scale(domain=[min_x-1, max_x+1])).axis(
            grid=False, tickOpacity=0, domainOpacity=0),
        y=alt.Y('y:Q', sort=alt.EncodingSortField(
            'y', order='descending'), title='', scale=alt.Scale(domain=[min_y-1, max_y+1])
        ).axis(grid=False, labelPadding=20, tickOpacity=0, domainOpacity=0),
        color=alt.Color(
            'source:N', scale=alt.Scale(scheme='lightmulti')),
        fill=alt.Color('source:N', scale=alt.Scale(
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
        x=alt.X('xFeaturePos:Q', title='', scale=alt.Scale(
            domain=[min_x-1, max_x+1])).axis(
            grid=False, tickOpacity=0, domainOpacity=0),
        y=alt.Y('y:Q', sort=alt.EncodingSortField(
            'y', order='descending'), title='', scale=alt.Scale(
            domain=[min_y-1, max_y+1])).axis(grid=False, labelPadding=20, tickOpacity=0, domainOpacity=0),
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


def features_plot_hex(_map, color_type, scaling=sum, flip=True):
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

    min_x = np_map['x'].min()
    max_x = np_map['x'].max()
    min_y = np_map['y'].min()
    max_y = np_map['y'].max()

    min_value = np_map['value'][np_map['value'] > 0].min()
    max_value = np_map['value'].max()

    # get index from new_dimensions
    index = np.where(new_dimensions == len(_map))[0][0]
    size = new_sizes[index]

    hexagon = "M0,-2.3094010768L2,-1.1547005384 2,1.1547005384 0,2.3094010768 -2,1.1547005384 -2,-1.1547005384Z"
    c = alt.Chart(np_map).mark_point(shape=hexagon, size=size**2).encode(
        x=alt.X('xFeaturePos:Q', title='', scale=alt.Scale(
            domain=[min_x-1, max_x+1])).axis(
            grid=False, tickOpacity=0, domainOpacity=0),
        y=alt.Y('y:Q', sort=alt.EncodingSortField(
            'y', order='descending'), title='', scale=alt.Scale(
                domain=[min_y-1, max_y+1])).axis(grid=False, labelPadding=20, tickOpacity=0, domainOpacity=0),
        color=alt.Color(
            'value:Q', scale=alt.Scale(scheme='lightmulti', type='pow')),
        fill=alt.Fill('value:Q', scale=alt.Scale(type=color_type, domain=(
            min_value, max_value), scheme='lightmulti')).legend(orient='bottom'),
        stroke=alt.value('black'),
        strokeWidth=alt.value(1.0)
    ).transform_calculate(
        # This field is required for the hexagonal X-Offset
        xFeaturePos='(datum.y%2)/2 + datum.x-.5'
    ).properties(
        # width should be the same as the height
        height=750,
    ).configure_view(
        strokeWidth=0
    )
    st.write('## SOM feature plot')
    st.altair_chart(c, use_container_width=True)


def features_plot(_map, color_type, scaling=sum, flip=True):
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

    c = alt.Chart(np_map).mark_rect().encode(
        x=alt.X('x:O', title=''),
        y=alt.Y('y:O', sort=alt.EncodingSortField(
            'y', order='descending'), title=''),
        color=alt.Color(
            'value:Q', scale=alt.Scale(type=color_type, scheme='lightmulti'))
    ).properties(
        height=750
    )
    st.write('## SOM feature plot')
    st.altair_chart(c, use_container_width=True)


def describe_classified_dataset(dataset_classified, assignments_central, assignments_neighbor):
    results = {}

    assignments_central_df = pd.DataFrame(assignments_central)
    assignments_neighbor_df = pd.DataFrame(assignments_neighbor)

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

    if ['assigned_class_central', 'confidence_central'] <= assignments_central_df.columns.tolist():
        # Distribution of assigned classes central
        results['assigned_class_counts_central'] = assignments_central_df['assigned_class_central'].value_counts()
        # Distribution of confidence levels central
        results['confidence_levels_central'] = assignments_central_df['confidence_central']
    else:
        results['assigned_class_counts_central'] = pd.Series(dtype='int64')
        results['confidence_levels_central'] = pd.Series(dtype='float64')

    if ['assigned_class_neighbor', 'confidence_neighbor'] <= assignments_neighbor_df.columns.tolist():
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
        total = len(classes)
        class_counts = pd.Series(classes).value_counts()
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
            if dominant_proportion >= parameters_classification['confidence_threshold']:
                # Assign the dominant class
                assignments_central.append({
                    'id': id_,
                    'assigned_class_central': dominant_class,
                    'confidence_central': dominant_proportion,
                    'position': pos
                })
        # Central neuron doesn't meet criteria, check neighbors
        neighbors = get_hex_neighbors(i, j, dim, dim)
        neighbors_classes = []
        for neighbor_pos in neighbors:
            if neighbor_pos in neuron_class_distribution_neighbor:
                neighbor_class_info = neuron_class_distribution_neighbor[neighbor_pos]
                neighbor_class_proportions = neighbor_class_info['class_proportions']
                neighbor_dominant_class = neighbor_class_proportions.idxmax()
                neighbor_dominant_proportion = neighbor_class_proportions.max()
                neighbor_total_detection = neighbor_class_info['total_detections']
                if (neighbor_dominant_proportion >= parameters_classification['neighbor_confidence_threshold'] and
                        neighbor_total_detection >= parameters_classification['min_detections_per_neighbor']):
                    neighbors_classes.append(neighbor_dominant_class)

        if neighbors_classes != []:
            neighbors_classes = pd.Series(neighbors_classes).value_counts()
            assigned_neighbor_class = neighbors_classes.idxmax()
            count_neighbor_class = neighbors_classes.max()

            # Check if the count of matching neighbors meets the threshold
            if count_neighbor_class >= parameters_classification['neighbor_majority_threshold']:
                # Assign the central neuron's dominant class
                assignments_neighbor.append({
                    'id': id_,
                    'assigned_class_neighbor': assigned_neighbor_class,
                    # Use central neuron's proportion
                    'confidence_neighbor': count_neighbor_class,
                    'position': pos
                })

    return assignments_central, assignments_neighbor


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
    df_to_norm.columns = df_to_norm.columns.str.replace(
        '_to_log_norm', '')

    # Step 1: Merge dataset_toclassify into df_to_norm
    combined_df = pd.concat(
        [df_to_norm, dataset_toclassify], ignore_index=True)

    # Step 2: Replace zero values in combined_df with one-tenth of the non-zero minimum
    for col in ['bb_kt', 'var_ratio_b', 'var_ratio_h', 'var_ratio_s']:
        min_nonzero = combined_df[col][combined_df[col] > 0].min()
        combined_df[col] = combined_df[col].replace(0, min_nonzero / 10)

    # Step 3: Log transform the necessary columns in combined_df
    combined_df['bb_kt'] = np.log(combined_df['bb_kt'])
    combined_df['var_ratio_b'] = np.log(combined_df['var_ratio_b'])
    combined_df['var_ratio_h'] = np.log(combined_df['var_ratio_h'])
    combined_df['var_ratio_s'] = np.log(combined_df['var_ratio_s'])

    # Step 4: Normalize the specified columns in combined_df
    scaler = MinMaxScaler()
    combined_df[['bb_kt', 'powlaw_gamma', 'var_ratio_b', 'var_ratio_h', 'var_ratio_s']] = scaler.fit_transform(
        combined_df[['bb_kt', 'powlaw_gamma', 'var_ratio_b', 'var_ratio_h', 'var_ratio_s']])

    # Step 5: Extract the transformed and normalized rows corresponding to dataset_toclassify
    transformed_dataset_toclassify = combined_df.iloc[-len(
        dataset_toclassify):].reset_index()

    # Return only the columns of interest
    return transformed_dataset_toclassify[['powlaw_gamma', 'bb_kt', 'var_ratio_b', 'var_ratio_h', 'var_ratio_s']]
