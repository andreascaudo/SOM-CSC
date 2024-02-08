import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from minisom import MiniSom
import pickle
import time
from som_fun import *

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

raw_dataset_path = './data/cluster_csc_simbad.csv'
dataset_path = './data/cluster_csc_simbad_log_normalized.csv'

# Load the dataset
st.session_state.raw_df = pd.read_csv(raw_dataset_path)
st.session_state.df = pd.read_csv(dataset_path)

GMM_cluster_labels = st.session_state.df['cluster']

# sidebar for the dataset
st.sidebar.header('User input')

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
else:
    # Features selection
    st.sidebar.write(
        '..or select the features and parameters for training the SOM')
    features = list(st.session_state.df.columns)
    default_features = list(
        st.session_state.df.columns.drop('cluster').drop('var_newq_b'))
    features = st.sidebar.multiselect(
        'Features', features, default_features, help='Select the features to be used for training the SOM.')
    st.session_state.df = st.session_state.df[features]
    X = st.session_state.df.to_numpy()

    # SOM parameters7
    st.sidebar.write('Select the SOM hyperparameters')
    dim = st.sidebar.slider(
        'Dimension', 6, 100, 30, help='The map\'s dimensions will be [dim x dim], forming a square layout.')
    sigma = st.sidebar.slider('Sigma', 0.01, 5.0, 1.0,
                              help='The spread of the neighborhood function')
    learning_rate = st.sidebar.slider(
        'Learning rate', 0.01, 5.0, 1.0, help='The degree of weight updates')
    iterations = st.sidebar.slider(
        'Iterations', 10, 1000, 100, help='Number of training iterations')
    topology = st.sidebar.selectbox(
        'Topology', ['hexagonal', 'rectangular'], help='Topology of the neurons in the SOM grid')
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
                                             learning_rate, iterations, topology)
            # baloons
            st.session_state.SOM_loaded = True
            st.session_state.SOM_loaded_trining = True
            st.balloons()
            st.experimental_rerun()

    elif st.session_state.SOM_loaded and st.session_state.SOM_loaded_trining:
        if not skip_errors:
            try:
                plot_errors(st.session_state.q_error,
                            st.session_state.t_error, len(st.session_state.q_error))
            except:
                pass
        # Download the SOM model as well as the features used to train it
        st.write('## Download the SOM model')
        st.download_button(
            label="Download", data=pickle.dumps([st.session_state.som, features]), file_name='SOM_model.pkl', mime='application/octet-stream', help='Download the SOM model store it and to upload it later')


with st.form(key='plot_form'):
    if st.session_state.SOM_loaded:
        st.write('## Select which plot to display')
        plot_type = st.selectbox(
            'Plot type', ['U-Matrix', 'Scatter Visualization [cluster]', 'Category Visualization [cluster]', 'Feature Visualization', 'Custom Feature Visualization'], help='Select the type of visualization to display')
        plot_submit = st.form_submit_button('Show plot')

        if plot_submit:
            if plot_type == 'U-Matrix':
                # U-Matrix
                st.write('## U-Matrix')
                with st.expander("See explanation"):
                    st.write(
                        'The U-Matrix is a visualization tool that represents the distance between neurons in the SOM. It is a 2D grid where each cell is colored based on the distance between the neurons. The U-Matrix is a useful tool to visualize the topology of the SOM and identify clusters of neurons.')
                if st.session_state.som.topology == 'rectangular':
                    plot_rectangular_u_matrix(st.session_state.som)
                else:
                    plot_u_matrix_hex(st.session_state.som)
                    # test()
            elif plot_type == 'Scatter Visualization [cluster]':
                # Scatter plot
                if st.session_state.som.topology == 'rectangular':
                    scatter_plot_clustering(
                        st.session_state.som, X, GMM_cluster_labels)
                else:
                    scatter_plot_clustering_hex(
                        st.session_state.som, X, GMM_cluster_labels)
            elif plot_type == 'Category Visualization [cluster]':
                # Category plot
                category_map = project_feature(
                    st.session_state.som, X, GMM_cluster_labels)
                if st.session_state.som.topology == 'rectangular':
                    category_plot_clustering(category_map)
                else:
                    category_plot_clustering_hex(category_map)
            elif plot_type == 'Feature Visualization':
                # Feature plot
                # subform to select a features
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
                    'Scaling', ['mean', 'min', 'max', 'sum', 'median', 'std'])
                st.write(
                    "###### To update the map with the selected feature and applied scaling, please click the 'Show Plot' button again.")
                var = project_feature(
                    st.session_state.som, X, st.session_state.raw_df[feature])
                if st.session_state.som.topology == 'rectangular':
                    features_plot(var, scaling=scaling)
                else:
                    features_plot_hex(var, scaling=scaling)
            elif plot_type == 'Custom Feature Visualization':
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
                st.write(
                    "###### To display the map with the loaded dataset and applied scaling, please click the 'Show Plot' button again.")

                if st.session_state.SOM_loaded:
                    uploaded_file = st.file_uploader(
                        "Upload your CSV file", type="csv")
                    scaling = st.selectbox(
                        'Scaling', ['mean', 'min', 'max', 'sum', 'median', 'std'])
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
                                st.write("DEBUG")
                                if st.session_state.som.topology == 'rectangular':
                                    features_plot(var, scaling=scaling)
                                else:
                                    features_plot_hex(var, scaling=scaling)
                            else:
                                st.error(
                                    "Dataset validation failed. Please check the column names and ensure there are no empty values.")
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
    else:
        st.write('## SOM model not loaded')
        st.write(
            'Please load a SOM model or generate a new one to visualize the map.')
