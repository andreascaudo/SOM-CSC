import streamlit as st
import pandas as pd
import numpy as np
import pickle
from som_fun import *
from tools import *
import glob
import os
import matplotlib.pyplot as plt


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

st.session_state.df_to_norm = st.session_state.raw_df[[
    'powlaw_gamma_to_log_norm', 'bb_kt_to_log_norm', 'var_ratio_b_to_log_norm', 'var_ratio_h_to_log_norm', 'var_ratio_s_to_log_norm']]
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

default_main_type = ['YSO', 'XrayBin', 'Seyfert', 'AGN']

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
        'Iterations', 10, 1000, 1000, help='Number of training iterations')
    topology = st.sidebar.selectbox(
        'Topology', ['hexagonal', 'rectangular'], help='Topology of the neurons in the SOM grid')
    seed = st.sidebar.number_input(
        'Seed', 0, 10000, 4444, help='Seed for reproducibility')
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
            'Plot type', ['U-Matrix', 'Activation Response', 'Training Feature Space Map', 'Source Name Visualization', 'Source Dispersion Visualization', 'Main Type Visualization', 'Feature Visualization'], help='Select the type of visualization to display')

        plot_submit = st.form_submit_button('Show plot')

        if plot_submit:
            if plot_type in ['U-Matrix', 'Activation Response', 'Training Feature Space Map', 'Feature Visualization']:
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
            elif plot_type == 'Source Dispersion Visualization':
                if st.session_state.som.topology == 'rectangular':
                    vis_type_string = 'Rectangular'
                elif st.session_state.som.topology == 'hexagonal':
                    vis_type_string = 'Hexbin'

                with st.expander("See explanation"):
                    st.write("***Dispersion Index:***")
                    st.write("This metric measures the average spatial separation between detections of a single source on the map. It is calculated as the mean pairwise distance between all detection positions of the source. A higher dispersion value indicates that the detections are more widely spread out, while a lower value suggests that the detections are more concentrated in a smaller area.")
                    st.write("***Minimum Detections:***")
                    st.write("The minimum number of detections required for a source to be included in the analysis. Sources with fewer detections than this threshold will be excluded from the calculation of the Dispersion Index.")
                    st.write("***Visualization Tool:***")
                    st.write(
                        "Enables the user to apply color coding to the pre-trained SOM based on the names of data sources. The user can select one or more sources (ordered based on the Dispersion Index) to visualize the distribution of sources across the map.")
                    st.write(
                        "**The user can choose between two visualization types: Scatter and " + vis_type_string + ":**")
                    st.write(
                        "***Scatter Visualization:*** The color assigned to each point on the map represents the source name of its corresponding detection.")
                    st.write("***" + vis_type_string +
                             " Visualization:*** The color of each neuron represents the source name that appears most frequently within that neuron.")

                name_counts = st.session_state.raw_df['name'].value_counts()
                max_counts = int(name_counts[0])
                min_detections = st.slider(
                    'Minimum Detections', 2, max_counts, 10, help='')
                id_to_pos = {}  # Dictionary to store the mapping from IDs to positions
                for id_ in st.session_state.raw_df['id']:
                    position = st.session_state.som.winner(X[id_])
                    id_to_pos[id_] = position

                name_ids = st.session_state.raw_df.groupby(
                    'name')['id'].apply(list).to_dict()

                dispersion_list = get_dispersion(
                    name_ids, id_to_pos, min_detections)
                dispersion_values = [dispersion for _,
                                     dispersion in dispersion_list]

                # Plot the histogram of all normalized dispersion values
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(dispersion_values, bins=30,
                        edgecolor='black', alpha=0.7)
                ax.set_title('Histogram of Source Dispersion')
                ax.set_xlabel(
                    'Dispersion')
                ax.set_ylabel('Number of Sources')
                ax.grid(axis='y', alpha=0.75)
                # Display the plot in Streamlit
                with st.popover("Show Dispersion Distribution"):
                    st.pyplot(fig)

                dispersion_list.sort(key=lambda x: x[1], reverse=True)
                dispersion_ms = [
                    f"{name[0]} [{name[1]}]" for name in dispersion_list]
                sources = st.multiselect(
                    'Select sources name [Dispersion index]', dispersion_ms)
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
                            'For string-value columns, none of the scaling methods will work. Instead, the most common value will be shown.')
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

                    is_string_var = is_string(var)

                    if st.session_state.som.topology == 'rectangular':
                        if is_string_var:
                            category_plot_sources(var)
                        else:
                            features_plot(var, type_option, scaling=scaling)
                    else:
                        if is_string_var:
                            category_plot_sources_hex(var)
                        else:
                            features_plot_hex(
                                var, type_option, scaling=scaling)
                elif dataset_choice == 'Upload a new dataset':
                    with st.expander("See explanation"):
                        st.write('This visualization tool enables coloring of the pre-trained SOM based on data from a newly uploaded dataset, allowing users to dynamically select which feature to use for coloring. Users have the flexibility to upload a new dataset and choose a specific feature that will be applied to color the previously trained SOM.')
                        st.write(
                            '⚠️ **The uploaded file must to be a .csv file with a specific structure:** ⚠️')
                        st.write(
                            'The initial columns should contain the features utilized for training the SOM. The final column should represent the feature intended for coloring the map. For instance, if the SOM was trained with features A, B, and C, and you wish to color the map using feature D, the CSV file should be organized as follows:')
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
                            # try:
                            # Assuming a new function 'validate_and_load_dataset' in som_fun.py
                            dataset = validate_and_load_dataset(
                                uploaded_file, features)
                            feature_to_trans_and_norm = [
                                'powlaw_gamma', 'bb_kt', 'var_ratio_b', 'var_ratio_h', 'var_ratio_s']
                            dataset[feature_to_trans_and_norm] = transform_and_normalize(
                                dataset[feature_to_trans_and_norm], st.session_state.df_to_norm)
                            # All rows, only the last column
                            Xx = dataset.iloc[:, :-1].to_numpy()
                            feature = dataset.iloc[:, -1]

                            if dataset is not None:
                                st.session_state['new_dataset'] = dataset
                                # Call to project_feature and features_plot goes here, using the new dataset
                                var = project_feature(
                                    st.session_state.som, Xx, feature)
                                is_string_var = is_string(var)

                                if st.session_state.som.topology == 'rectangular':
                                    if is_string_var:
                                        category_plot_sources(var)
                                    else:
                                        features_plot(
                                            var, type_option, scaling=scaling)
                                else:
                                    if is_string_var:
                                        category_plot_sources_hex(
                                            var)
                                    else:
                                        features_plot_hex(
                                            var, type_option, scaling=scaling)
                            else:
                                st.error(
                                    "Dataset validation failed. Please check the column names and ensure there are no empty values.")
                            # except Exception as e:
                            #    st.error(f"An error occurred: {e}")

    # Add a checkbox for enabling download options
    if st.session_state.som.topology == 'hexagonal':
        enable_classification = st.checkbox("Apply Classification", value=False,
                                            help="Check this box to apply classification")
        if enable_classification:
            with st.form(key='classification_form'):
                dataset_toclassify = None
                st.write("Apply classification")
                parameters_classification = {}
                # Set parameters
                parameters_classification['confidence_threshold'] = st.slider(
                    'Confidence Threshold', 0.0, 1.0, 0.5, help='Confidence Threshold')
                # Minimum detections per neuron for central neuron
                parameters_classification['min_detections_per_neuron'] = st.slider(
                    'Minimum Detection x Neuron', 0, 100, 20, help='Number of detection on the BMU')

                # Minimum confidence required in neighbor neurons
                # not needed IMO
                parameters_classification['neighbor_confidence_threshold'] = 0.
                # Minimum detections per neuron for central neuron
                parameters_classification['min_detections_per_neighbor'] = st.slider(
                    'Minimum Detection x Neighbor Neuron', 0, 100, 20, help='Number of detection on the neighbors')
                # Minimum number of neighbors that have the same class
                parameters_classification['neighbor_majority_threshold'] = st.slider(
                    'Minimum Number of Neighbor', 1, 6, 4, help='Minimum Number of Neighbor')

                som_map_id = download_activation_response(
                    st.session_state.som, X_index)

                dataset_choice = st.radio(
                    'Choose the dataset', ['Use the main dataset', 'Upload a new dataset'])
                if dataset_choice == 'Use the main dataset':
                    dataset_toclassify = st.session_state.df_index[pd.isna(
                        st.session_state.raw_df['main_type'])]
                elif dataset_choice == 'Upload a new dataset':
                    if st.session_state.SOM_loaded:
                        uploaded_file = st.file_uploader(
                            "Upload your CSV file", type="csv")
                        if uploaded_file is not None:
                            # Attempt to load and validate the dataset
                            try:
                                # Assuming a new function 'validate_and_load_dataset' in som_fun.py
                                dataset_toclassify = validate_and_load_dataset_class(
                                    uploaded_file, features).reset_index().rename(columns={"index": "id"})
                                feature_to_trans_and_norm = [
                                    'powlaw_gamma', 'bb_kt', 'var_ratio_b', 'var_ratio_h', 'var_ratio_s']
                                dataset_toclassify[feature_to_trans_and_norm] = transform_and_normalize(
                                    dataset_toclassify[feature_to_trans_and_norm], st.session_state.df_to_norm)
                            except Exception as e:
                                st.error(f"An error occurred: {e}")

                simbad_dataset = st.session_state.raw_df[pd.notna(
                    st.session_state.raw_df['main_type'])]
                SIMBAD_classes = set(st.session_state.raw_df['main_type'])

                classify = st.form_submit_button('Get Classification')

                if classify and dataset_toclassify is not None:
                    assignments_central, assignments_neighbor = get_classification(
                        som_map_id, dataset_toclassify, simbad_dataset, SIMBAD_classes, parameters_classification, dim, st.session_state.som)
                    # classify = False
                    dataset_classified = update_dataset_to_classify(
                        dataset_toclassify, assignments_central, assignments_neighbor)

                    st.session_state.dataset_classified = dataset_classified[['id', 'assigned_class_central',
                                                                             'confidence_central', 'assigned_class_neighbor', 'confidence_neighbor', 'is_classified']]

                    classification_results = describe_classified_dataset(
                        dataset_classified, assignments_central, assignments_neighbor)

                    st.title("Summary of Classification Assignment")

                    st.header("Overview")
                    st.write("### Key Metrics")
                    st.write(
                        f"**Total unclassified detections before assignment:** {classification_results['total_unclassified_before']}")
                    st.write(
                        f"**Number of detections assigned a class:** {classification_results['total_classified_rows']}")
                    st.write(
                        f"**Number of detections remaining unclassified:** {classification_results['num_unclassified_after']}")
                    st.write(
                        f"**Percentage of detections assigned:** {classification_results['percentage_assigned']:.2f}%")

                    st.write("### Assignment Details")
                    st.write(
                        f"**Number of detections assigned by central neuron:** {classification_results['num_assigned_central']}")
                    st.write(
                        f"**Number of detections assigned by neighbor neurons:** {classification_results['num_assigned_neighbor']}")

                    if classification_results['total_classified_rows'] != 0:
                        st.header("Classification Assignment Visualization")

                    # Bar chart of assigned classes (Central)
                    if not classification_results['assigned_class_counts_central'].empty:
                        with st.popover("Bar Chart: Central Neuron Assignments"):
                            st.write(
                                "Number of Detections Assigned to Each Class")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            classification_results['assigned_class_counts_central'].plot(
                                kind='bar', color='skyblue', edgecolor='black', ax=ax)
                            ax.set_title(
                                'Number of Detections Assigned to Each Class', fontsize=16)
                            ax.set_xlabel('Assigned Class', fontsize=14)
                            ax.set_ylabel('Number of Detections', fontsize=14)
                            ax.grid(axis='y', alpha=0.7)
                            ax.tick_params(axis='x', rotation=45, labelsize=12)
                            ax.tick_params(axis='y', labelsize=12)
                            st.pyplot(fig)

                    # Histogram of confidence levels (Central)
                    if not classification_results['confidence_levels_central'].empty:
                        with st.popover("Histogram: Central Neuron Confidence Levels"):
                            st.write(
                                "Histogram of Confidence Levels for Central Neurons")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.hist(classification_results['confidence_levels_central'], bins=20, color='green',
                                    edgecolor='black', alpha=0.7)
                            ax.set_title(
                                'Histogram of Assignment Confidence Levels (Central)', fontsize=16)
                            ax.set_xlabel('Confidence Level', fontsize=14)
                            ax.set_ylabel('Number of Detections', fontsize=14)
                            ax.grid(axis='y', alpha=0.7)
                            st.pyplot(fig)

                    # Bar chart of assigned classes (Neighbor)
                    if not classification_results['assigned_class_counts_neighbor'].empty:
                        with st.popover("Bar Chart: Neighbor Neuron Assignments"):
                            st.write(
                                "Number of Detections Assigned to Each Class (Neighbor)")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            classification_results['assigned_class_counts_neighbor'].plot(
                                kind='bar', color='skyblue', edgecolor='black', ax=ax)
                            ax.set_title(
                                'Number of Detections Assigned to Each Class (Neighbor)', fontsize=16)
                            ax.set_xlabel('Assigned Class', fontsize=14)
                            ax.set_ylabel('Number of Detections', fontsize=14)
                            ax.grid(axis='y', alpha=0.7)
                            ax.tick_params(axis='x', rotation=45, labelsize=12)
                            ax.tick_params(axis='y', labelsize=12)
                            st.pyplot(fig)

                    if not classification_results['confidence_levels_neighbor'].empty:
                        # Histogram of confidence levels (Neighbor)
                        with st.popover("Histogram: Neighbor Neuron Confidence Levels"):
                            st.write(
                                "Histogram of Confidence Levels for Neighbor Neurons")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            bin = np.arange(start=-0.5, stop=(6+1), step=1)
                            ax.hist(classification_results['confidence_levels_neighbor'], bins=bin, color='green',
                                    edgecolor='black', alpha=0.7)
                            ax.set_title(
                                'Histogram of Assignment Confidence Levels (Neighbor)', fontsize=16)
                            ax.set_xlabel('Confidence Level', fontsize=14)
                            ax.set_ylabel('Number of Detections', fontsize=14)
                            ax.set_xlim([0, 6+1])
                            ax.grid(axis='y', alpha=0.7)
                            st.pyplot(fig)

    enable_download = st.checkbox("Enable Downloads", value=False,
                                  help="Check this box to enable download options for the SOM model and datasets.")
    if enable_download:
        with st.spinner("Loading download options..."):
            with st.expander("Download Available"):
                st.download_button(
                    label="Download the SOM model",
                    data=pickle.dumps([st.session_state.som, features]),
                    file_name='SOM_model.pkl',
                    mime='application/octet-stream',
                    help='Download the SOM model, store it, and upload it later'
                )
                st.download_button(
                    label="Download the raw dataset",
                    data=st.session_state.raw_df.to_csv(index=False),
                    file_name='raw_dataset.csv',
                    mime='text/csv',
                    help='Download the raw dataset'
                )
                if st.session_state.som.topology == 'rectangular':
                    activation_map = plot_activation_response(
                        st.session_state.som, X_index, plot=False)
                else:
                    activation_map = plot_activation_response_hex(
                        st.session_state.som, X_index, plot=False)
                activation_map_csv = dict_to_csv(activation_map)
                st.download_button(
                    label="Download the activation response map",
                    data=activation_map_csv,
                    file_name='activation_response.csv',
                    mime='text/csv',
                    help='Download the activation response'
                )
                if 'dataset_classified' in st.session_state:
                    st.download_button(
                        label="Download the classification results",
                        data=st.session_state.dataset_classified.to_csv(
                            index=False),
                        file_name='dataset_classified.csv',
                        mime='text/csv',
                        help='Download the classification results'
                    )
else:
    st.write(
        'Please load a SOM model or generate a new one to visualize the map.')
