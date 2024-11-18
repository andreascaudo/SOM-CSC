import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from minisom import MiniSom
import pickle
import time
from som_fun import *
from tools import *
import glob
import os


def load_split_csvs(directory):
    all_files = sorted(glob.glob(os.path.join(directory, '*.csv')))
    df_list = [pd.read_csv(file) for file in all_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

# https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet


st.write("""
# Exploring the Chandra Source Catalog (CSC) using Self-Organizing Maps (SOM)
""")

if 'SOM_loaded' not in st.session_state:
    st.session_state.SOM_loaded = False
    st.session_state.SOM_loaded_trining = False
    # write in red SOM STATUS: NOT LOADED
    st.write('#### SOM STATUS: :red[NOT LOADED]')
elif st.session_state.SOM_loaded:
    st.write('#### SOM STATUS: :green[LOADED]')
elif st.session_state.SOM_loaded == False:
    st.write('#### SOM STATUS: :red[NOT LOADED]')

# sidebar for the dataset
st.sidebar.header('User input')

# toogle to select new vs old dataset
# new dataset
raw_dataset_path = './data/csc211_mastertable_clean_observationlevel_COMPLETE_xmatchSimbad1arcsec_log_norm_id/'
st.session_state.raw_df = load_split_csvs(raw_dataset_path)
st.session_state.df = st.session_state.raw_df[['hard_hm', 'hard_hs', 'hard_ms', 'powlaw_gamma_log_norm', 'var_prob_b', 'var_prob_s', 'var_prob_h',
                                               'bb_kt_log_norm', 'var_ratio_b_log_norm', 'var_ratio_h_log_norm', 'var_ratio_s_log_norm', 'var_newq_b_log_norm']]
st.session_state.df_index = st.session_state.raw_df[['id', 'hard_hm', 'hard_hs', 'hard_ms', 'powlaw_gamma_log_norm', 'var_prob_b', 'var_prob_s', 'var_prob_h',
                                                     'bb_kt_log_norm', 'var_ratio_b_log_norm', 'var_ratio_h_log_norm', 'var_ratio_s_log_norm', 'var_newq_b_log_norm']]
# remove log_norm from the columns name
st.session_state.df.columns = st.session_state.df.columns.str.replace(
    '_log_norm', '')
st.session_state.df_index.columns = st.session_state.df_index.columns.str.replace(
    '_log_norm', '')

# GMM_cluster_labels = st.session_state.df['cluster']
simbad_type = 'main_type'  # otype
main_type = st.session_state.raw_df[simbad_type]
# default_main_type = ['QSO', 'AGN', 'Seyfert_1', 'Seyfert_2', 'HMXB',
#                     'LMXB', 'XB', 'YSO', 'TTau*', 'Orion_V*']

default_main_type = ['YSO', 'HighMassXBin', 'LowMassXBin', 'Seyfert', 'AGN']

# Let the user load a SOM model
som_model = st.sidebar.file_uploader(
    "Upload your SOM model", type=['pkl'], accept_multiple_files=False, help='Upload your SOM model for visualization. ⚠️ **Note: Only .pkl files that were previously downloaded from this web application following the training process are compatible for upload.**')
# button to load the SOM model

if som_model is not None:
    st.session_state.SOM_loaded = True
    file = pickle.load(som_model)
    st.session_state.som, features = file[0], file[1]
    # get topology
    topology = st.session_state.som.topology
    st.session_state.df = st.session_state.df[features]
    X = st.session_state.df.to_numpy()

    index_features = ['id'].append(features)
    st.session_state.df_index = st.session_state.df_index[index_features]
    X_index = st.session_state.df_index.to_numpy()
else:
    # Features selection
    st.sidebar.write(
        '..or select the features and parameters for training the SOM')
    features = list(st.session_state.df.columns)
    default_features = list(
        st.session_state.df.columns.drop('var_newq_b'))
    features = st.sidebar.multiselect(
        'Features', features, default_features, help='Select the features to be used for training the SOM.')
    index_features = ['id'] + features

    st.session_state.df = st.session_state.df[features]
    X = st.session_state.df.to_numpy()

    st.session_state.df_index = st.session_state.df_index[index_features]
    X_index = st.session_state.df_index.to_numpy()

    # SOM parameters7
    st.sidebar.write('Select the SOM hyperparameters')
    dim = st.sidebar.slider(
        'Dimension', 6, 100, 40, help='The map\'s dimensions will be [dim x dim], forming a square layout.')
    sigma = st.sidebar.slider('Sigma', 0.01, 5.0, 3.8,
                              help='The spread of the neighborhood function')
    learning_rate = st.sidebar.slider(
        'Learning rate', 0.01, 5.0, 1.8, help='The degree of weight updates')
    iterations = st.sidebar.slider(
        'Iterations', 10, 1000, 500, help='Number of training iterations')
    topology = st.sidebar.selectbox(
        'Topology', ['hexagonal', 'rectangular'], help='Topology of the neurons in the SOM grid')
    seed = st.sidebar.number_input(
        'Seed', 0, 10000, 1234, help='Seed for reproducibility')
    # tick box to skip the calculation of errors at each step
    skip_errors = st.sidebar.checkbox(
        'Skip QE and TE computation at each step', value=True, help='Skip the computation of quantization and topographic errors at each step to reduce training duration. Deselect this option to gain insights into the required number of iterations for effective SOM training.')

    len_data = len(X)

    # Generate a new SOM model button
    if st.sidebar.button('Generate SOM'):
        if not skip_errors:
            errors_bar = st.progress(
                0, text="Getting quantization and topographic errors")
            st.session_state.q_error, st.session_state.t_error = get_iterations_index(X, dim, len(
                features), sigma, learning_rate, iterations, errors_bar)
            errors_bar.empty()

            plot_errors(st.session_state.q_error,
                        st.session_state.t_error, iterations)

        # PLEASE WAIT
        with st.spinner('Training the SOM...'):
            st.session_state.som = train_som(X, dim, dim, len(features), sigma,
                                             learning_rate, iterations, topology, seed)
            # baloons
            st.session_state.SOM_loaded = True
            st.session_state.SOM_loaded_trining = True
            st.balloons()
            st.rerun()

    elif st.session_state.SOM_loaded and st.session_state.SOM_loaded_trining:
        if not skip_errors:
            try:
                plot_errors(st.session_state.q_error,
                            st.session_state.t_error, len(st.session_state.q_error))
            except:
                pass

if st.session_state.SOM_loaded:
    activation_map_flag = False
    with st.form(key='plot_form'):
        st.write('## Select the plot to be displayed')

        plot_type = st.selectbox(
            'Plot type', ['U-Matrix', 'Activation Response', 'Training Feature Space Map', 'Source Name Visualization', 'Main Type Visualization', 'Feature Visualization'], help='Select the type of visualization to display')

        plot_submit = st.form_submit_button('Show plot')

        if plot_submit:
            if plot_type not in ['Source Name Visualization', 'Main Type Visualization']:
                log_option = st.checkbox(
                    'Logarithmic scale', value=False, help='Use a logarithmic scale for the color map')
                type_option = 'log' if log_option else 'linear'
            if plot_type == 'U-Matrix':
                # U-Matrix
                st.write('## U-Matrix')
                with st.expander("See explanation"):
                    st.write(
                        'The U-Matrix is a visualization tool that represents the distance between neurons in the SOM. It is a 2D grid where each cell is colored based on the distance between the neurons. The U-Matrix is a useful tool to visualize the topology of the SOM and identify clusters of neurons.')
                if st.session_state.som.topology == 'rectangular':
                    plot_rectangular_u_matrix(
                        st.session_state.som, type_option)
                else:
                    plot_u_matrix_hex(st.session_state.som, type_option)
            elif plot_type == 'Activation Response':
                # Activation Response
                st.write('## Activation Response')
                with st.expander("See explanation"):
                    st.write(
                        'The Activation Response visualization tool enables the user to visualize the frequency of samples that are assigned to each neuron. The color of each neuron will indicate the number of times that neuron was identified as the best matching unit for a sample.')
                if st.session_state.som.topology == 'rectangular':
                    activation_map = plot_activation_response(
                        st.session_state.som, X_index, type_option)
                else:
                    activation_map = plot_activation_response_hex(
                        st.session_state.som, X_index, type_option)
                activation_map_flag = True
            elif plot_type == 'Training Feature Space Map':
                # Training Feature Space Map
                st.write('## Training Feature Space Map')
                with st.expander("See explanation"):
                    st.write(
                        'The Training Feature Space Map visualization tool enables the user to apply color coding to the pre-trained SOM based on the weights of the neurons.')
                    st.write(
                        'If the user selects one or more features, the map will be colored according to the mean of the selected features across the map.')
                    st.write(
                        'The user can select one or more features to visualize the distribution of the features across the map.')
                    st.write(
                        'The user can also download the feature space map as a CSV file.')
                features = st.multiselect(
                    'Select features', st.session_state.df.columns.to_list())
                # get the index of the selected features
                features_index = [st.session_state.df.columns.get_loc(
                    feature) for feature in features]

                st.write(
                    "###### To update the map with the selected features, please click the 'Show Plot' button again.")
                # get weights
                weights = st.session_state.som.get_weights()[
                    :, :, features_index]
                if len(features_index) > 0:
                    if st.session_state.som.topology == 'rectangular':
                        feature_space_map_plot(weights, type_option)
                    else:
                        feature_space_map_plot_hex(weights, type_option)

            elif plot_type == 'Source Name Visualization':
                if st.session_state.som.topology == 'rectangular':
                    vis_type_string = 'Rectangular'
                elif st.session_state.som.topology == 'hexagonal':
                    vis_type_string = 'Hexbin'

                with st.expander("See explanation"):
                    st.write(
                        '##### This visualization tool enables the user to apply color coding to the pre-trained SOM according to the names of data sources.')
                    st.write(
                        '##### The user can select one or more sources to visualize the distribution of sources across the map.')
                    st.write(
                        '##### The user can choose between two visualization types: Scatter and ' + vis_type_string + '.')
                    st.write(
                        '***Scatter visualization:*** The color assigned to each point on the map will signify the source name of its corresponding detection.')
                    st.write('***' +
                             vis_type_string + ' visualization:*** The color of each neuron will indicate the source name that appears most frequently within that neuron.')
                # Category plot
                name_counts = st.session_state.raw_df['name'].value_counts()
                sorted_names = [f"{name} [{count}]" for name,
                                count in name_counts.items()]
                sources = st.multiselect(
                    'Select sources name [number of detections]', sorted_names)
                visualization_type = st.radio(
                    'Visualization type', ['Scatter', vis_type_string])
                st.write(
                    "###### To update the map with the name of the selected sources, please click the 'Show Plot' button again.")
                if len(sources) > 0:
                    sources = [source.split(' [')[0] for source in sources]
                    if st.session_state.som.topology == 'rectangular':
                        if visualization_type == 'Scatter':
                            scatter_plot_sources(
                                st.session_state.som, sources, st.session_state.raw_df, X, 'name')
                        elif visualization_type == 'Rectangular':
                            category_map = project_feature(
                                st.session_state.som, X, st.session_state.raw_df['name'], sources)
                            category_plot_sources(category_map)
                    elif st.session_state.som.topology == 'hexagonal':
                        if visualization_type == 'Scatter':
                            scatter_plot_sources_hex(
                                st.session_state.som, sources, st.session_state.raw_df, X, 'name')
                        elif visualization_type == 'Hexbin':
                            category_map = project_feature(
                                st.session_state.som, X, st.session_state.raw_df['name'], sources)
                            category_plot_sources_hex(category_map)
            elif plot_type == 'Main Type Visualization':
                if st.session_state.som.topology == 'rectangular':
                    vis_type_string = 'Rectangular'
                elif st.session_state.som.topology == 'hexagonal':
                    vis_type_string = 'Hexbin'

                with st.expander("See explanation"):
                    st.write(
                        '##### This visualization tool enables the user to apply color coding to the pre-trained SOM according to the main types of data detections.')
                    st.write(
                        '##### The user can select one or more main types to visualize the distribution of detections across the map.')
                    st.write(
                        '##### The user can choose between two visualization types: Scatter and ' + vis_type_string + '.')
                    st.write(
                        '***Scatter visualization:*** The color assigned to each point on the map will signify the main type of its corresponding detection.')
                    st.write('***' +
                             vis_type_string + ' visualization:*** The color of each neuron will indicate the main type that appears most frequently within that neuron.')
                # Scatter plot
                main_type_ = st.multiselect(
                    'Main type', np.unique(main_type.dropna()).tolist(), default_main_type)
                visualization_type = st.radio(
                    'Visualization type', ['Scatter', vis_type_string])

                st.write(
                    "###### To update the map with the name of the selected sources, please click the 'Show Plot' button again.")

                if len(main_type_) > 0:
                    if st.session_state.som.topology == 'rectangular':
                        if visualization_type == 'Scatter':
                            scatter_plot_sources(
                                st.session_state.som, main_type_, st.session_state.raw_df, X, simbad_type)
                        elif visualization_type == 'Rectangular':
                            category_map = project_feature(
                                st.session_state.som, X, st.session_state.raw_df[simbad_type], main_type_)
                            category_plot_sources(category_map)
                    elif st.session_state.som.topology == 'hexagonal':
                        if visualization_type == 'Scatter':
                            scatter_plot_sources_hex(
                                st.session_state.som, main_type_, st.session_state.raw_df, X, simbad_type)
                        elif visualization_type == 'Hexbin':
                            category_map = project_feature(
                                st.session_state.som, X, st.session_state.raw_df[simbad_type], main_type_)
                            category_plot_sources_hex(category_map)
            elif plot_type == 'Feature Visualization':
                dataset_choice = st.radio(
                    'Choose the dataset', ['Use the main dataset', 'Upload a new dataset'])

                if dataset_choice == 'Use the main dataset':
                    with st.expander("See explanation"):
                        st.write(
                            'This visualization will color the pre-trained SOM based on a selected feature, implementing a scaling method to each neuron.')
                        st.write(
                            'For example, if the selected feature represents the mean value of a particular column, each neuron\'s color will reflect the average of that column.')
                        st.write(
                            'Additionally, the frequency of samples at a specific neuron indicates the number of times that neuron was identified as the best matching unit for a sample.')
                    feature = st.selectbox(
                        'Feature', st.session_state.raw_df.columns.to_list())
                    scaling = st.selectbox(
                        'Feature Scaling', ['mean', 'min', 'max', 'sum', 'median', 'std'])
                    st.write(
                        "###### Please click the 'Show Plot' button after choosing the dataset type or to display the map, in order to refresh the view.")
                    var = project_feature(
                        st.session_state.som, X, st.session_state.raw_df[feature])
                    if st.session_state.som.topology == 'rectangular':
                        features_plot(var, type_option, scaling=scaling)
                    else:
                        features_plot_hex(var, type_option, scaling=scaling)
                elif dataset_choice == 'Upload a new dataset':
                    with st.expander("See explanation"):
                        st.write('This visualization tool enables coloring of the pre-trained SOM based on data from a newly uploaded dataset, allowing users to dynamically select which feature to use for coloring. Users have the flexibility to upload a new dataset and choose a specific feature that will be applied to color the previously trained SOM.')
                        st.write(
                            '⚠️ **The uploaded file must to be a .csv file with a specific structure:** ⚠️')
                        st.write(
                            'The initial columns should contain the features utilized for training the SOM, and these features must be already normalized. The final column should represent the feature intended for coloring the map. For instance, if the SOM was trained with features A, B, and C (all normalized), and you wish to color the map using feature D, the CSV file should be organized as follows:')
                        st.write(
                            '**A, B, C, D**')
                        st.write(
                            '*x1, y1, z1, 1*')
                        st.write(
                            '*x2, y2, z2, 2*')
                        st.write(
                            '*...*')
                        st.write(
                            '*xn, yn, zn, n*')
                if st.session_state.SOM_loaded:
                    uploaded_file = st.file_uploader(
                        "Upload your CSV file", type="csv")
                    scaling = st.selectbox(
                        'Upload Scaling', ['mean', 'min', 'max', 'sum', 'median', 'std'])
                    st.write(
                        "###### Please click the 'Show Plot' button after choosing the dataset type or to display the map, in order to refresh the view.")
                    if uploaded_file is not None:
                        # Attempt to load and validate the dataset
                        try:
                            # Assuming a new function 'validate_and_load_dataset' in som_fun.py
                            dataset = validate_and_load_dataset(
                                uploaded_file, features)
                            # All rows, all columns except the last
                            Xx = dataset.iloc[:, :-1].to_numpy()
                            # All rows, only the last column
                            feature = dataset.iloc[:, -1]

                            if dataset is not None:
                                st.session_state['new_dataset'] = dataset
                                # Call to project_feature and features_plot goes here, using the new dataset
                                var = project_feature(
                                    st.session_state.som, Xx, feature)
                                if st.session_state.som.topology == 'rectangular':
                                    features_plot(var, scaling=scaling)
                                else:
                                    features_plot_hex(var, scaling=scaling)
                            else:
                                st.error(
                                    "Dataset validation failed. Please check the column names and ensure there are no empty values.")
                        except Exception as e:
                            st.error(f"An error occurred: {e}")

    # Download the SOM model as well as the features used to train it
    st.write('## Download')
    st.download_button(
        label="Download the SOM model", data=pickle.dumps([st.session_state.som, features]), file_name='SOM_model.pkl', mime='application/octet-stream', help='Download the SOM model store it and to upload it later')
    st.download_button(
        label="Download the raw dataset", data=st.session_state.raw_df.to_csv(index=False), file_name='raw_dataset.csv', mime='text/csv', help='Download the raw dataset')
    if activation_map_flag:
        activation_map_csv = dict_to_csv(activation_map)
        st.download_button(
            label="Download the activation response map", data=activation_map_csv, file_name='activation_response.csv', mime='text/csv', help='Download the activation response')
else:
    st.write(
        'Please load a SOM model or generate a new one to visualize the map.')
