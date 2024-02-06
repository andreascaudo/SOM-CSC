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
    "Upload a SOM model", type=['pkl'], accept_multiple_files=False)
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
        '..or select the features and the parameters to train the SOM')
    features = list(st.session_state.df.columns)
    default_features = list(
        st.session_state.df.columns.drop('cluster').drop('var_newq_b'))
    features = st.sidebar.multiselect(
        'Features', features, default_features)
    st.session_state.df = st.session_state.df[features]
    X = st.session_state.df.to_numpy()

    # SOM parameters7
    st.sidebar.write('Select the SOM parameters')
    dim = st.sidebar.slider('Dimension', 6, 100, 30)
    sigma = st.sidebar.slider('Sigma', 0.01, 5.0, 1.0)
    learning_rate = st.sidebar.slider('Learning rate', 0.01, 5.0, 1.0)
    iterations = st.sidebar.slider('Iterations', 10, 1000, 100)
    topology = st.sidebar.selectbox(
        'Topology', ['hexagonal', 'rectangular'])
    # tick box to skip the calculation of errors at each step
    skip_errors = st.sidebar.checkbox(
        'Skip QE and TE calculation at each step', value=True)

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
            label="Download the SOM model", data=pickle.dumps([st.session_state.som, features]), file_name='SOM_model.pkl', mime='application/octet-stream')


with st.form(key='plot_form'):
    # select the plot to show
    if st.session_state.SOM_loaded:
        st.write('## Select the plot to show')
        plot_type = st.selectbox(
            'Plot type', ['U-Matrix', 'Scatter plot [cluster]', 'Category plot [cluster]', 'Feature plot'])
        plot_submit = st.form_submit_button('Show plot')

        if plot_submit:
            if plot_type == 'U-Matrix':
                # U-Matrix
                st.write('## U-Matrix')
                if st.session_state.som.topology == 'rectangular':
                    plot_rectangular_u_matrix(st.session_state.som)
                else:
                    plot_u_matrix_hex(st.session_state.som)
                    # test()
            elif plot_type == 'Scatter plot [cluster]':
                # Scatter plot
                if st.session_state.som.topology == 'rectangular':
                    scatter_plot_clustering(
                        st.session_state.som, X, GMM_cluster_labels)
                else:
                    scatter_plot_clustering_hex(
                        st.session_state.som, X, GMM_cluster_labels)
            elif plot_type == 'Category plot [cluster]':
                # Category plot
                category_map = project_feature(
                    st.session_state.som, X, GMM_cluster_labels)
                if st.session_state.som.topology == 'rectangular':
                    category_plot_clustering(category_map)
                else:
                    category_plot_clustering_hex(category_map)
            elif plot_type == 'Feature plot':
                # Feature plot
                # subform to select a features
                feature = st.selectbox(
                    'Feature', st.session_state.raw_df.columns.to_list())
                scaling = st.selectbox(
                    'Scaling', ['mean', 'min', 'max', 'sum', 'median', 'std'])
                st.write(
                    '###### After selecting the feature and applying scaling, click the "Show plot" button to refresh the map.')
                var_prob_m = project_feature(
                    st.session_state.som, X, st.session_state.raw_df[feature])
                if st.session_state.som.topology == 'rectangular':
                    features_plot(var_prob_m, scaling=scaling)
                else:
                    features_plot_hex(var_prob_m, scaling=scaling)
