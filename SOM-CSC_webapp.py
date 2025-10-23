import streamlit as st
import pandas as pd
import numpy as np
import pickle
from som_fun import *
from tools import *
import glob
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="SOM CSC",
    initial_sidebar_state="collapsed"
)


# Add this function to generate default colors
def get_default_color(index, color_scheme='lightmulti'):
    """
    Returns a color from a predefined set of colors based on the index.
    Uses a variety of colors to ensure distinctiveness.
    """
    # Common colors from various schemes
    color_schemes = {
        'lightmulti': [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
        ],
        'category10': [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ],
        'set3': [
            '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
            '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd',
            '#ccebc5', '#ffed6f'
        ]
    }

    # Use the specified color scheme or default to lightmulti
    colors = color_schemes.get(color_scheme, color_schemes['lightmulti'])

    # Return a color based on the index, cycling through the colors if needed
    return colors[index % len(colors)]


@st.cache_data
def download_som_model_bytes(_som, features):
    # This function returns a bytes object containing the pickled SOM model
    return pickle.dumps([_som, features])


@st.cache_data
def download_raw_dataset_csv(raw_df):
    # Return the CSV string for the raw dataset
    return raw_df.to_csv(index=False)


@st.cache_data
def download_activation_map_csv(_som, X_index):
    # Compute activation map once and cache the result
    if _som.topology == 'rectangular':
        activation_map = plot_activation_response(_som, X_index, plot=False)
    else:
        activation_map = plot_activation_response_hex(
            _som, X_index, plot=False)
    return dict_to_csv(activation_map)


@st.cache_data
def download_classified_csv(dataset_classified):
    # Return the CSV string for the classified dataset
    return dataset_classified.to_csv(index=False)


@st.fragment
def show_download_button(label, data, file_name, mime, help):
    st.download_button(
        label=label,
        data=data,
        file_name=file_name,
        mime=mime,
        help=help
    )


@st.cache_data
def load_split_csvs(directory):
    all_files = sorted(glob.glob(os.path.join(directory, '*.csv')))
    df_list = [pd.read_csv(file) for file in all_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

# https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet


st.write("""
# Exploring the Chandra Source Catalog (CSC) using Self-Organizing Maps (SOM)
""")

# Initialize color customization state if not already present
if 'custom_colors' not in st.session_state:
    st.session_state.custom_colors = {}

# Add an initialized colors state to track which colors have been assigned
if 'initialized_colors' not in st.session_state:
    st.session_state.initialized_colors = set()

# Dataset selection in sidebar - must be early to affect session state initialization
st.sidebar.header('Dataset Selection')
dataset_version = st.sidebar.selectbox(
    'Dataset version', ['CSC 2.1.1', 'CSC 2.1.1 clean'], index=0, key='dataset_selector')

if dataset_version == 'CSC 2.1.1':
    raw_dataset_path = './data/csc211_mastertable_clean_observationlevel_COMPLETE_xmatchSimbad1arcsec_log_norm_id_ClassAgg/'
elif dataset_version == 'CSC 2.1.1 clean':
    raw_dataset_path = './data/csc212_mastertable_clean_log_norm_id/'

# Check if dataset has changed and clear relevant session state
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = dataset_version
elif st.session_state.current_dataset != dataset_version:
    # Dataset changed - clear data-related session state
    keys_to_clear = ['raw_df', 'full_df', 'df', 'full_df_index',
                     'df_index', 'df_to_norm', 'all_original_features']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.current_dataset = dataset_version
    st.rerun()

# Initialize session state for dataframes
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = load_split_csvs(raw_dataset_path)

if 'full_df' not in st.session_state:
    st.session_state.full_df = st.session_state.raw_df[['hard_hm', 'hard_hs', 'hard_ms', 'powlaw_gamma_log_norm', 'var_prob_b', 'var_prob_s', 'var_prob_h',
                                                        'bb_kt_log_norm', 'var_ratio_b_log_norm', 'var_ratio_h_log_norm', 'var_ratio_s_log_norm']]
    st.session_state.full_df.columns = st.session_state.full_df.columns.str.replace(
        '_log_norm', '')

if 'df' not in st.session_state:
    st.session_state.df = st.session_state.full_df.copy()

if 'full_df_index' not in st.session_state:
    st.session_state.full_df_index = st.session_state.raw_df[['id', 'hard_hm', 'hard_hs', 'hard_ms', 'powlaw_gamma_log_norm', 'var_prob_b', 'var_prob_s', 'var_prob_h',
                                                              'bb_kt_log_norm', 'var_ratio_b_log_norm', 'var_ratio_h_log_norm', 'var_ratio_s_log_norm']]
    st.session_state.full_df_index.columns = st.session_state.full_df_index.columns.str.replace(
        '_log_norm', '')

if 'df_index' not in st.session_state:
    st.session_state.df_index = st.session_state.full_df_index.copy()

if 'df_to_norm' not in st.session_state:
    st.session_state.df_to_norm = st.session_state.raw_df[[
        'powlaw_gamma_to_log_norm', 'bb_kt_to_log_norm', 'var_ratio_b_to_log_norm', 'var_ratio_h_to_log_norm', 'var_ratio_s_to_log_norm']]

# Store all original features before any model is loaded
if 'all_original_features' not in st.session_state:
    # Store the original columns without the '_log_norm' suffix
    original_columns = st.session_state.df.columns.str.replace('_log_norm', '')
    st.session_state.all_original_features = list(original_columns)

# remove log_norm from the columns name
st.session_state.df.columns = st.session_state.df.columns.str.replace(
    '_log_norm', '')
st.session_state.df_index.columns = st.session_state.df_index.columns.str.replace(
    '_log_norm', '')

# GMM_cluster_labels = st.session_state.df['cluster']
st.session_state.simbad_type = 'class'  # otype, main_type
main_type = st.session_state.raw_df[st.session_state.simbad_type]
# default_main_type = ['QSO', 'AGN', 'Seyfert_1', 'Seyfert_2', 'HMXB',
#                     'LMXB', 'XB', 'YSO', 'TTau*', 'Orion_V*']

default_main_type = ['YSO', 'Binaries', 'AGN', 'Stars']
color_schemes = ['lightmulti', 'blueorange',
                 'viridis', 'redyellowblue', 'plasma', 'greenblue', 'redblue']

# Now try to load the default model after data is prepared
if 'SOM_loaded' not in st.session_state:
    st.session_state.SOM_loaded = False
    st.session_state.SOM_loaded_trining = False
    # Auto-load the default model
    try:
        default_model_path = './models/SOM_model_default.pkl'
        with open(default_model_path, 'rb') as f:
            file = pickle.load(f)
            st.session_state.som, features = file[0], file[1]

            # Reset feature scale ranges when loading a new model
            reset_feature_scale_ranges()

            # Update the working dataframes to use only the features the model was trained on
            st.session_state.df = st.session_state.full_df[features].copy()

            # Ensure the index version includes the ID column
            index_features = ['id'] + features
            st.session_state.df_index = st.session_state.full_df_index[index_features].copy(
            )

            X = st.session_state.df.to_numpy()
            X_index = st.session_state.df_index.to_numpy()

            st.session_state.SOM_loaded = True
            st.success("Default SOM model loaded successfully!")
    except Exception as e:
        # write in red SOM STATUS: NOT LOADED
        st.write('#### SOM STATUS: :red[NOT LOADED]')
        st.error(f"Could not load default model: {e}")
elif st.session_state.SOM_loaded:
    st.write('#### SOM STATUS: :green[LOADED]')
elif st.session_state.SOM_loaded == False:
    st.write('#### SOM STATUS: :red[NOT LOADED]')

# Remove the sidebar toggle functionality and add instructions
st.write("""
To train a new SOM, click the **>** arrow icon in the top-left corner to expand the sidebar.
""")

# Sidebar for model management
st.sidebar.header('Model Management')

# Let the user load a SOM model
som_model = st.sidebar.file_uploader(
    "Upload your SOM model", type=['pkl'], accept_multiple_files=False, help='Upload your SOM model for visualization. ⚠️ **Note: Only .pkl files that were previously downloaded from this web application following the training process are compatible for upload.**')
# button to load the SOM model

if som_model is not None:
    file = pickle.load(som_model)
    st.session_state.som, features = file[0], file[1]
    st.session_state.selected_features = features
    # get topology
    topology = st.session_state.som.topology
    dim = st.session_state.som.get_weights().shape[0]

    # Reset feature scale ranges when loading a new model
    reset_feature_scale_ranges()

    # Update the working dataframes to use only the features the model was trained on
    st.session_state.df = st.session_state.full_df[features].copy()

    # Ensure the index version includes the ID column
    index_features = ['id'] + features
    st.session_state.df_index = st.session_state.full_df_index[index_features].copy(
    )

    X = st.session_state.df.to_numpy()
    X_index = st.session_state.df_index.to_numpy()

    if st.sidebar.button('Load SOM'):
        st.session_state.SOM_loaded = True
        download_som_model_bytes.clear()
        download_activation_map_csv.clear()
        if 'apply_classification' in st.session_state:
            st.session_state.apply_classification = False
        if 'apply_download' in st.session_state:
            st.session_state.apply_download = False
        st.rerun()

else:
    # Features selection
    st.sidebar.write(
        '..or select the features and parameters for training the SOM')

    # Use the full list of original features for selection
    features_for_selection = st.session_state.all_original_features

    # Fix the default features selection to avoid KeyError
    default_features = features_for_selection.copy()

    features = st.sidebar.multiselect(
        'Features', features_for_selection, default_features, help='Select the features to be used for training the SOM.')
    st.session_state.selected_features = features
    index_features = ['id'] + features

    # Only update df and X if we're not already using a loaded model
    if not st.session_state.SOM_loaded:
        # Use the selected features but don't permanently modify the original dataframes
        X = st.session_state.full_df[features].to_numpy()
        X_index = st.session_state.full_df_index[index_features].to_numpy()
    else:
        # Get the current features and data for the loaded model
        X = st.session_state.df.to_numpy()
        X_index = st.session_state.df_index.to_numpy()

    # SOM parameters7
    st.sidebar.write('Select the SOM hyperparameters')
    dim = st.sidebar.slider(
        'Dimension', 6, 100, 40, help='The map\'s dimensions will be [dim x dim], forming a square layout.')
    sigma = st.sidebar.slider('Sigma', 0.01, 25.0, 4.0,
                              help='The spread of the neighborhood function')
    learning_rate = st.sidebar.slider(
        'Learning rate', 0.01, 5.0, 0.04, help='The degree of weight updates')
    iterations = st.sidebar.slider(
        'Iterations', 0, 1000000, 1000000, 10000, help='Number of training iterations')
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
        # When training a new model, reconstruct the dataframes with the selected features
        # This ensures we're working with the fresh original data, not filtered by previous models

        # Reset feature scale ranges when generating a new model
        reset_feature_scale_ranges()

        # Create fresh dataframes with the selected features - but don't overwrite the full dataframes
        df_selected = st.session_state.full_df[features].copy()

        # Create the training data array
        X = df_selected.to_numpy()

        # Create the index data with IDs
        df_index_selected = st.session_state.full_df_index[[
            'id'] + features].copy()
        X_index = df_index_selected.to_numpy()

        # Verify that the number of features matches what's expected
        if X.shape[1] != len(features):
            st.error(
                f"Feature count mismatch: Selected {len(features)} features but data has {X.shape[1]} columns.")
            st.error(f"Selected features: {features}")
            st.error(
                f"Data columns: {df_selected.columns.tolist()}")
        else:
            if not skip_errors:
                errors_bar = st.progress(
                    0, text="Getting quantization and topographic errors")

                st.session_state.steps = 100
                st.session_state.som, st.session_state.q_error, st.session_state.t_error = get_iterations_index(X, dim, dim, len(features), sigma,
                                                                                                                learning_rate, iterations, topology, seed, st.session_state.steps, errors_bar=errors_bar)
                errors_bar.empty()

            # PLEASE WAIT
            with st.spinner('Training the SOM...'):
                st.session_state.som = train_som(X, dim, dim, len(features), sigma,
                                                 learning_rate, iterations, topology, seed)

                st.session_state.SOM_loaded = True
                st.session_state.SOM_loaded_trining = True

                # Update session state with the trained features but preserve full dataframes
                st.session_state.df = st.session_state.full_df[features].copy()
                st.session_state.df_index = st.session_state.full_df_index[[
                    'id'] + features]

                st.balloons()
                download_som_model_bytes.clear()
                download_activation_map_csv.clear()
                if 'apply_classification' in st.session_state:
                    st.session_state.apply_classification = False
                if 'apply_download' in st.session_state:
                    st.session_state.apply_download = False
                st.rerun()

    elif st.session_state.SOM_loaded and st.session_state.SOM_loaded_trining:
        if not skip_errors:
            try:
                plot_errors(st.session_state.q_error,
                            st.session_state.t_error, iterations, st.session_state.steps)
            except:
                pass
        # Display quantization error
        quantization_error = st.session_state.som.quantization_error(X)
        st.write(f"**Quantization Error:** {quantization_error:.4f}")
        # Display topographic error based on topology
        if topology == "rectangular":
            topographic_error = st.session_state.som.topographic_error(X)
        elif topology == "hexagonal":
            topographic_error = topographic_error_hex(
                st.session_state.som, X)
        st.write(
            f"**Topographic Error:** {topographic_error:.4f}")


if st.session_state.SOM_loaded:

    # When a model is loaded (either default, uploaded, or generated), ensure we have proper X and X_index
    if not 'X' in locals() or not 'X_index' in locals():
        # Get the features this model was trained with
        features = st.session_state.df.columns.tolist()
        X = st.session_state.df.to_numpy()
        X_index = st.session_state.df_index.to_numpy()

    activation_map_flag = False
    with st.form(key='plot_form'):
        st.write('## Select the plot to be displayed')

        plot_type = st.selectbox(
            'Plot type', ['U-Matrix', 'Activation Response', 'Training Feature Space Map', 'Source Name Visualization', 'Source Dispersion Visualization', 'Main Type Visualization', 'Feature Visualization'], help='Select the type of visualization to display')

        plot_submit = st.form_submit_button('Show plot')

        if plot_submit:
            if plot_type in ['U-Matrix', 'Activation Response', 'Training Feature Space Map', 'Source Name Visualization', 'Feature Visualization']:
                color_scheme = st.selectbox(
                    'Color scheme', color_schemes, 0)
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
                        st.session_state.som, type_option, color_scheme)
                else:
                    plot_u_matrix_hex(st.session_state.som,
                                      type_option, color_scheme)
            elif plot_type == 'Activation Response':
                # Activation Response
                st.write('## Activation Response')
                with st.expander("See explanation"):
                    st.write(
                        'The Activation Response visualization tool enables the user to visualize the frequency of samples that are assigned to each neuron. The color of each neuron will indicate the number of times that neuron was identified as the best matching unit for a sample.')
                if st.session_state.som.topology == 'rectangular':
                    activation_map = plot_activation_response(
                        st.session_state.som, X_index, type_option, color_scheme)
                else:
                    activation_map = plot_activation_response_hex(
                        st.session_state.som, X_index, type_option, color_scheme)
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
                        feature_space_map_plot(
                            weights, type_option, color_scheme)
                    else:
                        feature_space_map_plot_hex(
                            weights, type_option, color_scheme)

            elif plot_type == 'Source Name Visualization':
                if st.session_state.som.topology == 'rectangular':
                    vis_type_string = 'Rectangular'
                elif st.session_state.som.topology == 'hexagonal':
                    vis_type_string = 'Hexbin'

                dataset_choice = st.radio(
                    'Choose the dataset', ['Use the main dataset', 'Upload a new dataset'])
                if dataset_choice == 'Use the main dataset':

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
                    name_counts = st.session_state.raw_df['name'].value_counts(
                    )
                    sorted_names = [f"{name} [{count}]" for name,
                                    count in name_counts.items()]
                    sources = st.multiselect(
                        'Select sources name [number of detections]', sorted_names)

                    feature_name = st.selectbox(
                        'Feature (optional)',
                        [None] + st.session_state.raw_df.columns.to_list(),
                        index=0
                    )

                    sources = [source.split(' [')[0] for source in sources]
                    st.write(
                        "###### To update the map with the name of the selected sources, please click the 'Show Plot' button again.")
                elif dataset_choice == 'Upload a new dataset':
                    with st.expander("See explanation"):
                        st.write(
                            '⚠️ **The uploaded file must to be a .csv file with a specific structure:** ⚠️')
                        st.write(
                            'The initial columns should contain the features names. The second column should represent the feature intended for coloring the map. For example, if the SOM was trained with features A, B, and C, and you wish to color the map using feature D, the CSV file should be organized as follows:')
                        st.write(
                            '**Source name, feature to project**')
                        st.write(
                            '*2CXO J004238.6+411603, 1*')
                        st.write(
                            '*2CXO J004231.1+411621, 2*')
                        st.write(
                            '*...*')
                        st.write(
                            '*2CXO J010337.5-720133, n*')
                        st.write('---')
                    if st.session_state.SOM_loaded:
                        uploaded_file = st.file_uploader(
                            "Upload your CSV file", type="csv")
                        st.write(
                            "###### Please click the 'Show Plot' in order to refresh the view.")

                        # Store selections for plotting outside the form
                        st.session_state.feature_viz_uploaded_file = uploaded_file
                        if uploaded_file is not None:
                            source_name_df = pd.read_csv(
                                uploaded_file, header=None)
                            # first column is the source name, second column is the feature to project
                            sources = source_name_df.iloc[:, 0].tolist()
                            feature_values = source_name_df.iloc[:, 1].tolist()
                            # create a dict with the source name as key and the feature as value
                            source_feature_dict = dict(
                                zip(sources, feature_values))
                            feature_name = None
                        else:
                            sources = []
                            feature_values = None

                visualization_type = st.radio(
                    'Visualization type', [vis_type_string, 'Scatter'])

                # Add visualization controls when Scatter is selected
                jitter_amount = 0.5
                show_grid = True
                if visualization_type == 'Scatter':
                    # Add a slider for jitter control
                    jitter_amount = st.slider(
                        'Jitter amount', 0.0, 1.0, 0.5, 0.01,
                        help='Controls how much random jitter is applied to scatter points to avoid overlapping. Higher values spread points further apart.')

                    # Add a toggle for grid visibility
                    show_grid = st.checkbox(
                        'Show grid background', value=True,
                        help='Toggle the visibility of the grid background in scatter plots.')

                # Add color customization option
                customize_colors = st.checkbox(
                    'Customize colors for sources', key='customize_colors_source_name', help='Select this option to customize the colors of the feature. Only works with categorical features. For numerical features, use the color scheme option.')

                def init_custom_colors(feature):
                    # clear the custom colors
                    st.session_state.custom_colors = {}
                    st.session_state.initialized_colors = set()

                    with st.expander("Color Customization"):
                        st.write("Select custom colors for each source:")

                        # Create columns to save space
                        cols = st.columns(min(3, len(feature)))

                        for i, f in enumerate(feature):
                            col_idx = i % len(cols)
                            with cols[col_idx]:
                                # Use existing color if already set or a varied default color
                                if f not in st.session_state.initialized_colors:
                                    # Assign a default color from our variety of colors
                                    st.session_state.custom_colors[f] = get_default_color(
                                        i)
                                    st.session_state.initialized_colors.add(
                                        f)

                                selected_color = st.color_picker(
                                    f"{f}", st.session_state.custom_colors[f], key=f"color_{f}")
                                st.session_state.custom_colors[f] = selected_color

                if len(sources) > 0:
                    if feature_name is not None or dataset_choice == 'Upload a new dataset':
                        if dataset_choice == 'Use the main dataset':
                            # Only include feature values that are present in at least one row where 'name' is in sources
                            feature_values_in_sources = st.session_state.raw_df.loc[
                                st.session_state.raw_df['name'].isin(
                                    sources), feature_name
                            ].unique()

                            if customize_colors and len(feature_values_in_sources) > 0 and is_string(feature_values_in_sources):
                                init_custom_colors(feature_values_in_sources)
                            source_colors = {fn: st.session_state.custom_colors.get(
                                fn, None) for fn in feature_values_in_sources} if customize_colors else None
                            var = project_feature(st.session_state.som, X,
                                                  st.session_state.raw_df[feature_name], filter_by=st.session_state.raw_df['name'], valid_values=sources)
                            # Precompute all scaling metrics for global scale initializatio
                            precompute_feature_scale_ranges(var, feature_name)
                        else:
                            if customize_colors and len(source_feature_dict.values()) > 0 and is_string(source_feature_dict.values()):
                                init_custom_colors(
                                    source_feature_dict.values())
                            source_colors = {fn: st.session_state.custom_colors.get(
                                fn, None) for fn in source_feature_dict.values()} if customize_colors else None

                            features_train = st.session_state.df.columns.tolist()
                            for i in range(len(features_train)):
                                if features_train[i] in ['bb_kt', 'powlaw_gamma', 'var_ratio_b', 'var_ratio_h', 'var_ratio_s']:
                                    # append "_log_norm"
                                    features_train[i] = features_train[i] + \
                                        "_log_norm"
                            var = project_external_feature(st.session_state.som, st.session_state.raw_df[[
                                'name'] + features_train].to_numpy(),
                                source_feature_dict)

                        is_string_var = is_string(var)

                        if is_string_var:
                            st.write('## SOM Feature Visualization')
                            st.write(f"**Feature:** {feature_name}")
                            st.write(
                                "**Note:** String features are displayed with the most common value per neuron.")

                        if topology == 'rectangular':
                            if is_string(var):
                                category_plot_sources(var)
                            else:
                                features_plot(var, type_option, color_scheme,
                                              scaling='mean', feature_name=feature_name)
                        else:
                            if is_string(var):
                                if visualization_type == 'Scatter':
                                    category_plot_sources_scatter(var,
                                                                  show_grid=show_grid,
                                                                  jitter_amount=jitter_amount, custom_colors=source_colors, category='N')
                                else:
                                    category_plot_sources_hex(
                                        var, custom_colors=source_colors)
                            else:
                                if visualization_type == 'Scatter':
                                    category_plot_sources_scatter(var,
                                                                  show_grid=show_grid,
                                                                  jitter_amount=jitter_amount, custom_colors=source_colors, category='Q')
                                else:
                                    features_plot_hex(var, type_option, color_scheme,
                                                      scaling='mean', feature_name=feature_name)
                    else:
                        if customize_colors:
                            sources = [f.split(' [')[0] for f in sources]
                            init_custom_colors(sources)
                            source_colors = {src: st.session_state.custom_colors.get(
                                src, None) for src in sources} if customize_colors else None
                        else:
                            source_colors = None

                        if st.session_state.som.topology == 'rectangular':
                            if visualization_type == 'Scatter':
                                scatter_plot_sources(
                                    st.session_state.som, sources, st.session_state.raw_df, X, 'name',
                                    custom_colors=source_colors, jitter_amount=jitter_amount, show_grid=show_grid)
                            elif visualization_type == 'Rectangular':
                                category_map = project_feature(
                                    st.session_state.som, X, st.session_state.raw_df['name'], valid_values=sources)
                                category_plot_sources(
                                    category_map, custom_colors=source_colors)
                        elif st.session_state.som.topology == 'hexagonal':
                            if visualization_type == 'Scatter':
                                scatter_plot_sources_hex(
                                    st.session_state.som, sources, st.session_state.raw_df, X, 'name',
                                    custom_colors=source_colors, jitter_amount=jitter_amount, show_grid=show_grid)
                            elif visualization_type == 'Hexbin':
                                category_map = project_feature(
                                    st.session_state.som, X, st.session_state.raw_df['name'], valid_values=sources)
                                category_plot_sources_hex(
                                    category_map, custom_colors=source_colors)

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
                max_counts = int(name_counts.iloc[0])
                min_counts = int(name_counts.iloc[-1])
                min_max_detections = st.slider(
                    'Minimum Detections', np.max((2, min_counts)), max_counts, (int(max_counts/4)-1, int(max_counts/3)-1), help='')
                id_to_pos = {}  # Dictionary to store the mapping from IDs to positions
                for id_ in st.session_state.raw_df['id']:
                    position = st.session_state.som.winner(X[id_])
                    id_to_pos[id_] = position

                name_ids = st.session_state.raw_df.groupby(
                    'name')['id'].apply(list).to_dict()

                dispersion_list = get_dispersion(
                    name_ids, id_to_pos, min_max_detections)
                dispersion_values = [dispersion for _,
                                     dispersion, _ in dispersion_list]

                dispersion_to_download = []

                dispersion_list.sort(key=lambda x: x[1], reverse=True)

                dispersion_ms = [
                    f"{name[0]} [{name[1]}] [{name[2]}]" for name in dispersion_list]

                dispersion_disp = [dispersion[1]
                                   for dispersion in dispersion_list]
                dispersion_counts = [dispersion[2]
                                     for dispersion in dispersion_list]
                # Scatter plot dispersion values vs number of detections
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(dispersion_counts, dispersion_disp, s=5)
                ax.set_title('Number of Detections vs Dispersion')
                ax.set_xlabel('Number of Detections')
                ax.set_ylabel('Dispersion')
                # dispersion index log scale
                # ax.set_xscale('log')
                # tick x
                ax.set_xticks([1, 15, 20, 40, 60, 80, 100, 120])
                plt.grid(True, which="both", ls="--", lw=0.5, alpha=0.7)
                # ax.set_yscale('log')
                with st.popover("Show Dispersion Distribution"):
                    st.pyplot(fig)

                sources = st.multiselect(
                    'Select sources name [Dispersion index] [N. of detections]', dispersion_ms)
                visualization_type = st.radio(
                    'Visualization type', ['Scatter', vis_type_string])

                # Add visualization controls when Scatter is selected
                jitter_amount = 0.5
                show_grid = True
                if visualization_type == 'Scatter':
                    # Add a slider for jitter control
                    jitter_amount = st.slider(
                        'Jitter amount', 0.0, 1.0, 0.5, 0.01,
                        help='Controls how much random jitter is applied to scatter points to avoid overlapping. Higher values spread points further apart.')

                    # Add a toggle for grid visibility
                    show_grid = st.checkbox(
                        'Show grid background', value=True,
                        help='Toggle the visibility of the grid background in scatter plots.')

                # Add color customization option
                customize_colors = st.checkbox(
                    'Customize colors for sources', key='customize_colors_source_dispersion')

                if customize_colors and len(sources) > 0:
                    with st.expander("Color Customization"):
                        st.write("Select custom colors for each source:")
                        clean_sources = [source.split(
                            ' [')[0] for source in sources]

                        # Create columns to save space
                        cols = st.columns(min(3, len(clean_sources)))

                        for i, source in enumerate(clean_sources):
                            col_idx = i % len(cols)
                            with cols[col_idx]:
                                # Use existing color if already set or a varied default color
                                if source not in st.session_state.initialized_colors:
                                    # Assign a default color from our variety of colors
                                    st.session_state.custom_colors[source] = get_default_color(
                                        i)
                                    st.session_state.initialized_colors.add(
                                        source)

                                selected_color = st.color_picker(
                                    f"{source}", st.session_state.custom_colors[source], key=f"color_disp_{source}")
                                st.session_state.custom_colors[source] = selected_color

                st.write(
                    "###### To update the map with the name of the selected sources, please click the 'Show Plot' button again.")
                if len(sources) > 0:
                    sources = [source.split(' [')[0] for source in sources]
                    source_colors = {src: st.session_state.custom_colors.get(
                        src, None) for src in sources} if customize_colors else None

                    if st.session_state.som.topology == 'rectangular':
                        if visualization_type == 'Scatter':
                            scatter_plot_sources(
                                st.session_state.som, sources, st.session_state.raw_df, X, 'name',
                                custom_colors=source_colors, jitter_amount=jitter_amount, show_grid=show_grid)
                        elif visualization_type == 'Rectangular':
                            category_map = project_feature(
                                st.session_state.som, X, st.session_state.raw_df['name'], valid_values=sources)
                            category_plot_sources(
                                category_map, custom_colors=source_colors)
                    elif st.session_state.som.topology == 'hexagonal':
                        if visualization_type == 'Scatter':
                            scatter_plot_sources_hex(
                                st.session_state.som, sources, st.session_state.raw_df, X, 'name',
                                custom_colors=source_colors, jitter_amount=jitter_amount, show_grid=show_grid)
                        elif visualization_type == 'Hexbin':
                            category_map = project_feature(
                                st.session_state.som, X, st.session_state.raw_df['name'], valid_values=sources)
                            category_plot_sources_hex(
                                category_map, custom_colors=source_colors)

                dispersion_list_i = get_dispersion(
                    name_ids, id_to_pos, (np.max((2, min_counts)), max_counts))
                # download the dataframe
                for i in dispersion_list_i:
                    name_i = i[0]
                    dispersion_i = i[1]
                    detections_i = i[2]

                    # If the following conditions are met:
                    # - beetween 2 and 15 detections only if dispersion >= 30
                    # - between 15 and 20 detections only if dispersion >= 20
                    # - between 20 and 100 detections only if dispersion >= 10
                    # - above 100 detections: Inlcude everything
                    # extrapolate all the detections from raw_df using the source name

                    if detections_i >= 2 and detections_i < 15 and dispersion_i >= 30:
                        dispersion_to_download.append(
                            [name_i, dispersion_i, detections_i])
                    elif detections_i >= 15 and detections_i < 20 and dispersion_i >= 20:
                        dispersion_to_download.append(
                            [name_i, dispersion_i, detections_i])
                    elif detections_i >= 20 and detections_i < 100 and dispersion_i >= 10:
                        dispersion_to_download.append(
                            [name_i, dispersion_i, detections_i])
                    elif detections_i >= 100:
                        dispersion_to_download.append(
                            [name_i, dispersion_i, detections_i])

                # following this logic, I need to create a new dataframe with the detections that meet the conditions
                # create a new dataframe with the detections that meet the conditions, also the dispersion index and the number of detections
                dispersion_to_download_df = st.session_state.raw_df[st.session_state.raw_df['name'].isin(
                    [name[0] for name in dispersion_to_download])].copy()
                dispersion_to_download_df['dispersion'] = dispersion_to_download_df['name'].map(
                    {name[0]: name[1] for name in dispersion_to_download})
                # move this column right after the name column
                dispersion_to_download_df.insert(
                    dispersion_to_download_df.columns.get_loc('name') + 1, 'dispersion', dispersion_to_download_df.pop('dispersion'))

                # drop the index column
                dispersion_to_download_df = dispersion_to_download_df.reset_index(
                    drop=True)

                with st.expander("See the dataframe"):
                    # Explain all the conditions
                    st.write(
                        "The following RAW dataframe includes detections that satisfy these criteria:")
                    st.write(
                        "- 2 to 14 detections with a dispersion of at least 30")
                    st.write(
                        "- 15 to 19 detections with a dispersion of at least 20")
                    st.write(
                        "- 20 to 99 detections with a dispersion of at least 10")
                    st.write("- 100 or more detections with any dispersion value")
                    st.write(dispersion_to_download_df)

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

                main_type_counts = main_type.value_counts()
                default_main_type_counts = []
                for default in default_main_type:
                    default_main_type_counts.append(
                        f"{default} [{main_type_counts[default]}]")
                sorted_main_type = [f"{name} [{count}]" for name,
                                    count in main_type_counts.items()]

                main_type_ = st.multiselect(
                    'Main type [Number of detections]', sorted_main_type, default_main_type_counts)
                main_type_ = [mt.split(' [')[0] for mt in main_type_]
                visualization_type = st.radio(
                    'Visualization type', ['Scatter', vis_type_string])

                # Add visualization controls when Scatter is selected
                jitter_amount = 0.5
                show_grid = True
                if visualization_type == 'Scatter':
                    # Add a slider for jitter control
                    jitter_amount = st.slider(
                        'Jitter amount', 0.0, 1.0, 0.5, 0.01,
                        help='Controls how much random jitter is applied to scatter points to avoid overlapping. Higher values spread points further apart.')

                    # Add a toggle for grid visibility
                    show_grid = st.checkbox(
                        'Show grid background', value=True,
                        help='Toggle the visibility of the grid background in scatter plots.')

                # Add color customization option
                customize_colors = st.checkbox(
                    'Customize colors for main types', key='customize_colors_main_type')

                if customize_colors and len(main_type_) > 0:
                    with st.expander("Color Customization"):
                        st.write("Select custom colors for each main type:")

                        # Create columns to save space
                        cols = st.columns(min(3, len(main_type_)))

                        for i, type_name in enumerate(main_type_):
                            col_idx = i % len(cols)
                            with cols[col_idx]:
                                # Use existing color if already set or a varied default color
                                if type_name not in st.session_state.initialized_colors:
                                    # Assign a default color from our variety of colors
                                    st.session_state.custom_colors[type_name] = get_default_color(
                                        i)
                                    st.session_state.initialized_colors.add(
                                        type_name)

                                selected_color = st.color_picker(
                                    f"{type_name}", st.session_state.custom_colors[type_name], key=f"color_main_{type_name}")
                                st.session_state.custom_colors[type_name] = selected_color

                # Add clustering overlay option
                overlay_clustering = st.checkbox(
                    'Overlay feature-based clustering', value=False,
                    help='Overlay clustering information based on selected feature value maps',
                    key='overlay_clustering_main_type')

                # Initialize variables needed for clustering
                clustering_results = None
                optimal_k = None

                if overlay_clustering:
                    # Initialize cluster colors if not already in session state
                    if 'custom_cluster_colors' not in st.session_state:
                        st.session_state.custom_cluster_colors = {}

                    # Add option to customize cluster colors
                    customize_cluster_colors = st.checkbox(
                        'Customize colors for clusters', key='customize_cluster_colors',
                        help='Customize the border colors used for different clusters')

                    # Add cluster visualization method selection
                    cluster_viz_method = st.selectbox(
                        'Cluster visualization method',
                        ['Enhanced Borders', 'Standard Borders'],
                        index=0,
                        help='Choose how to display cluster information: Enhanced Borders (thick colored borders with white outline) or Standard Borders (thin colored borders)')

                    # Add controls for border width customization
                    st.write("#### Border Width Customization")
                    col1, col2 = st.columns(2)

                    with col1:
                        enhanced_border_width = st.slider(
                            'Enhanced border width', 1, 10, 4, 1,
                            help='Thickness of cluster outline borders')

                    with col2:
                        standard_border_width = st.slider(
                            'Standard border width', 1, 5, 1, 1,
                            help='Thickness of regular hexagon borders')

                    if customize_cluster_colors:
                        with st.expander("Cluster Color Customization"):
                            st.write("Select custom colors for each cluster:")

                            # Initialize enhanced and standard border color dictionaries
                            if 'custom_enhanced_border_colors' not in st.session_state:
                                st.session_state.custom_enhanced_border_colors = {}
                            if 'custom_standard_border_colors' not in st.session_state:
                                st.session_state.custom_standard_border_colors = {}

                            # Define default cluster colors for enhanced borders
                            default_enhanced_colors = ['#cccccc', '#FF0000', '#00FF00', '#0000FF', '#FFFF00',
                                                       '#FF00FF', '#00FFFF', '#FFA500', '#800080', '#FFC0CB']

                            # Default standard border color (black for all clusters)
                            default_standard_color = '#000000'

                            # Show color pickers for potential clusters (limit to 10)
                            for i in range(10):
                                st.write(f"**Cluster {i}**")
                                col1, col2 = st.columns(2)

                                with col1:
                                    # Enhanced border color
                                    current_enhanced_color = st.session_state.custom_enhanced_border_colors.get(
                                        i, default_enhanced_colors[i % len(default_enhanced_colors)])
                                    selected_enhanced_color = st.color_picker(
                                        f"Enhanced border", current_enhanced_color, key=f"enhanced_cluster_{i}")
                                    st.session_state.custom_enhanced_border_colors[
                                        i] = selected_enhanced_color

                                with col2:
                                    # Standard border color
                                    current_standard_color = st.session_state.custom_standard_border_colors.get(
                                        i, default_standard_color)
                                    selected_standard_color = st.color_picker(
                                        f"Standard border", current_standard_color, key=f"standard_cluster_{i}")
                                    st.session_state.custom_standard_border_colors[
                                        i] = selected_standard_color

                    # Clustering Settings moved outside of the expander
                    st.write("### Clustering Settings")
                    st.write(
                        "Select feature and value map for clustering overlay:")

                    # Feature selection dropdown - use the same features as Feature Visualization
                    clustering_feature = st.selectbox(
                        'Feature for clustering',
                        st.session_state.raw_df.columns.to_list(),
                        help='Select the feature to use for generating clustering overlay'
                    )

                    # Value map selection dropdown
                    clustering_value_map = st.selectbox(
                        'Value map type',
                        ['minimum', 'median', 'maximum'],
                        help='Select which value map to use for clustering (min, median, or max across the SOM map)'
                    )

                    # Perform clustering if overlay is enabled
                    clustering_feature_map = project_feature(
                        st.session_state.som, X, st.session_state.raw_df[clustering_feature])

                    # Extract values from each neuron based on the selected value map
                    neuron_values = []
                    neuron_positions = []

                    for i in range(len(clustering_feature_map)):
                        for j in range(len(clustering_feature_map[i])):
                            neuron_data = clustering_feature_map[i][j]
                            if neuron_data and neuron_data[0] is not None:
                                # Remove None values and compute the selected statistic
                                valid_values = [
                                    v for v in neuron_data if v is not None]
                                if valid_values:
                                    if clustering_value_map == 'minimum':
                                        neuron_value = min(valid_values)
                                    elif clustering_value_map == 'median':
                                        neuron_value = np.median(
                                            valid_values)
                                    else:  # maximum
                                        neuron_value = max(valid_values)

                                    neuron_values.append(neuron_value)
                                    neuron_positions.append((i, j))

                    # Perform clustering only if we have enough neurons with data
                    # Need at least 4 neurons for meaningful clustering
                    if len(neuron_values) >= 4:
                        # Convert to numpy array for clustering
                        X_clustering = np.array(
                            neuron_values).reshape(-1, 1)

                        # Determine optimal k using elbow method
                        # Ensure we don't exceed data points
                        max_k = min(10, len(neuron_values) - 1)
                        k_range = range(2, max_k + 1)
                        wcss = []

                        for k in k_range:
                            kmeans = KMeans(
                                n_clusters=k, random_state=42, n_init=10)
                            kmeans.fit(X_clustering)
                            wcss.append(kmeans.inertia_)

                        # Find elbow point (simple method: maximum second derivative)
                        if len(wcss) >= 3:
                            # Calculate second derivatives
                            second_derivatives = []
                            for i in range(1, len(wcss) - 1):
                                second_der = wcss[i-1] - \
                                    2*wcss[i] + wcss[i+1]
                                second_derivatives.append(second_der)

                            # Find the k with maximum second derivative (elbow point)
                            elbow_idx = np.argmax(second_derivatives)
                            # +1 because second_derivatives starts from index 1
                            optimal_k = k_range[elbow_idx + 1]
                        else:
                            optimal_k = 2  # Default to 2 if not enough data points

                        # Perform final clustering with optimal k
                        final_kmeans = KMeans(
                            n_clusters=optimal_k, random_state=42, n_init=10)
                        cluster_labels = final_kmeans.fit_predict(
                            X_clustering)

                        # Store clustering results
                        clustering_results = {
                            'neuron_positions': neuron_positions,
                            'neuron_values': neuron_values,
                            'cluster_labels': cluster_labels,
                            'wcss': wcss,
                            'k_range': list(k_range),
                            'optimal_k': optimal_k
                        }

                        # Add cluster filtering option
                        st.write("### Cluster Display Options")
                        cluster_options = [
                            f"Cluster {i}" for i in range(optimal_k)]

                        # Initialize session state for selected clusters if not exists
                        if 'selected_clusters' not in st.session_state:
                            st.session_state.selected_clusters = cluster_options.copy()

                        # Reset selected clusters if optimal_k changed
                        if len(st.session_state.selected_clusters) != optimal_k:
                            st.session_state.selected_clusters = cluster_options.copy()

                        selected_clusters = st.multiselect(
                            'Select clusters to display',
                            options=cluster_options,
                            default=st.session_state.selected_clusters,
                            help='Choose which clusters to show in the visualization. Unselected clusters will be hidden.',
                            key='cluster_filter'
                        )

                        # Update session state
                        st.session_state.selected_clusters = selected_clusters

                        # Create a set of selected cluster indices for filtering
                        selected_cluster_indices = {
                            int(cluster.split()[1]) for cluster in selected_clusters}

                        # Filter clustering results based on selected clusters
                        if len(selected_clusters) > 0:
                            # Create filtered versions of the clustering data
                            filtered_positions = []
                            filtered_labels = []

                            for pos, label in zip(neuron_positions, cluster_labels):
                                if label in selected_cluster_indices:
                                    filtered_positions.append(pos)
                                    filtered_labels.append(label)

                            # Update clustering results with filtered data
                            clustering_results['filtered_neuron_positions'] = filtered_positions
                            clustering_results['filtered_cluster_labels'] = filtered_labels
                            clustering_results['selected_cluster_indices'] = selected_cluster_indices
                        else:
                            # If no clusters selected, hide all clusters
                            clustering_results['filtered_neuron_positions'] = [
                            ]
                            clustering_results['filtered_cluster_labels'] = []
                            clustering_results['selected_cluster_indices'] = set(
                            )

                        # Debug outputs placed inside an expander
                        with st.expander("Clustering Debug Information"):
                            # Plot WCSS curve
                            col1, col2 = st.columns(2)

                            with col1:
                                st.write("#### Elbow Method - WCSS vs K")
                                fig_elbow, ax_elbow = plt.subplots(
                                    figsize=(8, 6))
                                ax_elbow.plot(k_range, wcss, 'bo-',
                                              linewidth=2, markersize=8)
                                ax_elbow.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2,
                                                 label=f'Optimal k = {optimal_k}')
                                ax_elbow.set_xlabel('Number of Clusters (k)')
                                ax_elbow.set_ylabel(
                                    'Within-Cluster Sum of Squares (WCSS)')
                                ax_elbow.set_title(
                                    'Elbow Method for Optimal k')
                                ax_elbow.grid(True, alpha=0.3)
                                ax_elbow.legend()
                                st.pyplot(fig_elbow)

                            with col2:
                                st.write("#### Cluster Assignments")
                                st.write(
                                    f"**Selected Feature:** {clustering_feature}")
                                st.write(
                                    f"**Value Map Type:** {clustering_value_map}")
                                st.write(
                                    f"**Number of Neurons with Data:** {len(neuron_values)}")
                                st.write(
                                    f"**Optimal Number of Clusters:** {optimal_k}")

                                # Show cluster distribution
                                cluster_counts = pd.Series(
                                    cluster_labels).value_counts().sort_index()

                                st.write("**All Clusters Distribution:**")
                                for cluster_id, count in cluster_counts.items():
                                    st.write(
                                        f"Cluster {cluster_id}: {count} neurons")

                                # Show selected clusters information
                                if len(selected_clusters) > 0:
                                    st.write("**Selected Clusters:**")
                                    selected_cluster_counts = {i: cluster_counts.get(
                                        i, 0) for i in selected_cluster_indices}
                                    for cluster_id, count in sorted(selected_cluster_counts.items()):
                                        st.write(
                                            f"Cluster {cluster_id}: {count} neurons")

                                    total_selected_neurons = sum(
                                        selected_cluster_counts.values())
                                    st.write(
                                        f"**Total Selected Neurons:** {total_selected_neurons}")
                                else:
                                    st.write(
                                        "**No clusters selected for display**")

                            # Plot cluster assignments on a simple grid
                            st.write("#### Cluster Assignment Visualization")
                            fig_clusters, ax_clusters = plt.subplots(
                                figsize=(10, 8))

                            # Create a grid to show cluster assignments
                            som_shape = st.session_state.som.get_weights(
                            ).shape[:2]
                            # -1 for neurons without data
                            cluster_grid = np.full(som_shape, -1)

                            # Use filtered cluster data for the visualization
                            filtered_positions = clustering_results.get(
                                'filtered_neuron_positions', [])
                            filtered_labels = clustering_results.get(
                                'filtered_cluster_labels', [])
                            selected_cluster_indices = clustering_results.get(
                                'selected_cluster_indices', set())

                            for (i, j), cluster_id in zip(filtered_positions, filtered_labels):
                                cluster_grid[i, j] = cluster_id

                            # Create a masked array to make "no data" areas transparent
                            cluster_grid_masked = np.ma.masked_where(
                                cluster_grid == -1, cluster_grid)

                            # Create a custom colormap with distinct colors for clusters (no color for no data)
                            cluster_colors = ['#cccccc', '#000000', '#333333', '#666666', '#999999',
                                              '#1a1a1a', '#4d4d4d', '#808080', '#b3b3b3', '#e6e6e6']

                            # Use custom colors if enabled (only for selected clusters)
                            if customize_cluster_colors:
                                colors = [st.session_state.custom_enhanced_border_colors.get(i, cluster_colors[i % len(cluster_colors)])
                                          for i in sorted(selected_cluster_indices)]
                            else:
                                # Use colors for selected clusters only
                                colors = [cluster_colors[i % len(cluster_colors)] for i in sorted(
                                    selected_cluster_indices)]

                            # Create a custom colormap
                            from matplotlib.colors import ListedColormap
                            custom_cmap = ListedColormap(colors) if colors else ListedColormap([
                                '#ffffff'])  # White if no clusters

                            # Create a custom normalization to map values correctly
                            from matplotlib.colors import BoundaryNorm
                            if len(selected_cluster_indices) > 0:
                                # Map selected cluster indices to color indices
                                sorted_indices = sorted(
                                    selected_cluster_indices)
                                min_cluster = min(sorted_indices)
                                max_cluster = max(sorted_indices)
                                bounds = np.arange(
                                    min_cluster - 0.5, max_cluster + 1.5, 1)
                                norm = BoundaryNorm(bounds, custom_cmap.N)
                            else:
                                # No clusters selected
                                bounds = [-0.5, 0.5]
                                norm = BoundaryNorm(bounds, 1)

                            # Plot the cluster grid with the custom colormap
                            im = ax_clusters.imshow(
                                cluster_grid_masked.T, cmap=custom_cmap, norm=norm, aspect='equal', origin='lower')
                            ax_clusters.set_title(
                                f'Cluster Assignments - {len(selected_cluster_indices)} Selected Clusters\n({clustering_feature} - {clustering_value_map})')
                            ax_clusters.set_xlabel('SOM X')
                            ax_clusters.set_ylabel('SOM Y')

                            # Add colorbar with custom ticks (only for selected clusters)
                            if len(selected_cluster_indices) > 0:
                                cbar = plt.colorbar(
                                    im, ax=ax_clusters, ticks=sorted(selected_cluster_indices))
                                cbar.ax.set_yticklabels(
                                    [f'Cluster {i}' for i in sorted(selected_cluster_indices)])
                            else:
                                # No colorbar if no clusters selected
                                pass

                            st.pyplot(fig_clusters)

                    else:
                        st.warning(
                            f"Not enough neurons with data for clustering. Found {len(neuron_values)} neurons, need at least 4.")

                st.write(
                    "###### To update the map with the name of the selected sources, please click the 'Show Plot' button again.")

                if len(main_type_) > 0:
                    type_colors = {mt: st.session_state.custom_colors.get(
                        mt, None) for mt in main_type_} if customize_colors else None

                    if st.session_state.som.topology == 'rectangular':
                        if visualization_type == 'Scatter':
                            scatter_plot_sources(
                                st.session_state.som, main_type_, st.session_state.raw_df, X, st.session_state.simbad_type,
                                custom_colors=type_colors, jitter_amount=jitter_amount, show_grid=show_grid)
                        elif visualization_type == 'Rectangular':
                            category_map = project_feature(
                                st.session_state.som, X, st.session_state.raw_df[st.session_state.simbad_type], valid_values=main_type_)

                            # Check if clustering overlay is enabled and results are available
                            if overlay_clustering and clustering_results is not None:
                                # Create a dictionary mapping neuron positions to cluster IDs (filtered)
                                cluster_mapping = {pos: cluster_id for pos, cluster_id in
                                                   zip(clustering_results.get('filtered_neuron_positions', []),
                                                       clustering_results.get('filtered_cluster_labels', []))}

                                # Define cluster border colors (same as used in the cluster visualization)
                                cluster_colors = ['#cccccc', '#000000', '#333333', '#666666', '#999999',
                                                  '#1a1a1a', '#4d4d4d', '#808080', '#b3b3b3', '#e6e6e6']

                                # Get the selected cluster indices
                                selected_cluster_indices = clustering_results.get(
                                    'selected_cluster_indices', set(range(optimal_k)))

                                # Prepare color dictionaries based on customization settings
                                if customize_cluster_colors:
                                    # Use custom colors from user input
                                    enhanced_border_colors = {
                                        i: st.session_state.custom_enhanced_border_colors.get(
                                            i, '#cccccc')
                                        for i in selected_cluster_indices
                                    }
                                    standard_border_colors = {
                                        i: st.session_state.custom_standard_border_colors.get(
                                            i, '#000000')
                                        for i in selected_cluster_indices
                                    }
                                else:
                                    # Use default colors: different colors for enhanced borders, black for standard
                                    default_enhanced_colors = ['#cccccc', '#FF0000', '#00FF00', '#0000FF', '#FFFF00',
                                                               '#FF00FF', '#00FFFF', '#FFA500', '#800080', '#FFC0CB']
                                    enhanced_border_colors = {
                                        i: default_enhanced_colors[i % len(
                                            default_enhanced_colors)]
                                        for i in selected_cluster_indices
                                    }
                                    standard_border_colors = {
                                        i: '#000000'  # Black for all clusters by default
                                        for i in selected_cluster_indices
                                    }

                                # Choose visualization method based on user selection
                                if 'cluster_viz_method' in locals() and cluster_viz_method == 'Standard Borders':
                                    # Use original thin borders (pass stroke width of 1)
                                    category_plot_sources_hex(
                                        category_map, custom_colors=type_colors,
                                        cluster_mapping=cluster_mapping,
                                        enhanced_border_colors=enhanced_border_colors,
                                        standard_border_colors=standard_border_colors,
                                        cluster_stroke_width=standard_border_width,
                                        enhanced_border_width=enhanced_border_width,
                                        som_shape=st.session_state.som.get_weights().shape[:2])
                                else:
                                    # Default: Enhanced Borders (current implementation with thick borders + white outline)
                                    category_plot_sources_hex(
                                        category_map, custom_colors=type_colors,
                                        cluster_mapping=cluster_mapping,
                                        enhanced_border_colors=enhanced_border_colors,
                                        standard_border_colors=standard_border_colors,
                                        cluster_stroke_width=enhanced_border_width,
                                        enhanced_border_width=enhanced_border_width,
                                        som_shape=st.session_state.som.get_weights().shape[:2])
                            else:
                                category_plot_sources_hex(
                                    category_map, custom_colors=type_colors)
                    elif st.session_state.som.topology == 'hexagonal':
                        if visualization_type == 'Scatter':
                            scatter_plot_sources_hex(
                                st.session_state.som, main_type_, st.session_state.raw_df, X, st.session_state.simbad_type,
                                custom_colors=type_colors, jitter_amount=jitter_amount, show_grid=show_grid)
                        elif visualization_type == 'Hexbin':
                            category_map = project_feature(
                                st.session_state.som, X, st.session_state.raw_df[st.session_state.simbad_type], valid_values=main_type_)

                            # Check if clustering overlay is enabled and results are available
                            if overlay_clustering and clustering_results is not None:
                                # Create a dictionary mapping neuron positions to cluster IDs (filtered)
                                cluster_mapping = {pos: cluster_id for pos, cluster_id in
                                                   zip(clustering_results.get('filtered_neuron_positions', []),
                                                       clustering_results.get('filtered_cluster_labels', []))}

                                # Define cluster border colors (same as used in the cluster visualization)
                                cluster_colors = ['#cccccc', '#000000', '#333333', '#666666', '#999999',
                                                  '#1a1a1a', '#4d4d4d', '#808080', '#b3b3b3', '#e6e6e6']

                                # Get the selected cluster indices
                                selected_cluster_indices = clustering_results.get(
                                    'selected_cluster_indices', set(range(optimal_k)))

                                # Prepare color dictionaries based on customization settings
                                if customize_cluster_colors:
                                    # Use custom colors from user input
                                    enhanced_border_colors = {
                                        i: st.session_state.custom_enhanced_border_colors.get(
                                            i, '#cccccc')
                                        for i in selected_cluster_indices
                                    }
                                    standard_border_colors = {
                                        i: st.session_state.custom_standard_border_colors.get(
                                            i, '#000000')
                                        for i in selected_cluster_indices
                                    }
                                else:
                                    # Use default colors: different colors for enhanced borders, black for standard
                                    default_enhanced_colors = ['#cccccc', '#FF0000', '#00FF00', '#0000FF', '#FFFF00',
                                                               '#FF00FF', '#00FFFF', '#FFA500', '#800080', '#FFC0CB']
                                    enhanced_border_colors = {
                                        i: default_enhanced_colors[i % len(
                                            default_enhanced_colors)]
                                        for i in selected_cluster_indices
                                    }
                                    standard_border_colors = {
                                        i: '#000000'  # Black for all clusters by default
                                        for i in selected_cluster_indices
                                    }

                                # Choose visualization method based on user selection
                                if 'cluster_viz_method' in locals() and cluster_viz_method == 'Standard Borders':
                                    # Use original thin borders (pass stroke width of 1)
                                    category_plot_sources_hex(
                                        category_map, custom_colors=type_colors,
                                        cluster_mapping=cluster_mapping,
                                        enhanced_border_colors=enhanced_border_colors,
                                        standard_border_colors=standard_border_colors,
                                        cluster_stroke_width=standard_border_width,
                                        enhanced_border_width=enhanced_border_width,
                                        som_shape=st.session_state.som.get_weights().shape[:2])
                                else:
                                    # Default: Enhanced Borders (current implementation with thick borders + white outline)
                                    category_plot_sources_hex(
                                        category_map, custom_colors=type_colors,
                                        cluster_mapping=cluster_mapping,
                                        enhanced_border_colors=enhanced_border_colors,
                                        standard_border_colors=standard_border_colors,
                                        cluster_stroke_width=enhanced_border_width,
                                        enhanced_border_width=enhanced_border_width,
                                        som_shape=st.session_state.som.get_weights().shape[:2])
                            else:
                                category_plot_sources_hex(
                                    category_map, custom_colors=type_colors)

                        # Add the empty hexagons plot for label count visualization
                        if st.session_state.som.topology == 'hexagonal':
                            st.write("---")
                            st.write("### Label Count Visualization")
                            with st.expander("See explanation"):
                                st.write(
                                    "This visualization shows the number of unique main types present in each neuron. Each hexagon is colored based on how many different main types are represented by the detections mapped to that neuron. For example, if a neuron contains 3 detections all of type 'YSO', its count (for coloring) would be 1. If it contains detections from 'YSO', 'AGN', and 'Seyfert', its count (for coloring) would be 3. "
                                    "Furthermore, the number displayed inside each non-empty hexagon indicates the dominance percentage of the most frequent main type within that neuron. For example, if a neuron contains 5 detections of 'YSO', 3 of 'AGN', and 2 of 'Star', 'YSO' is the most frequent. Its dominance would be (5 / (5+3+2)) * 100 = 50%, and thus '50' would be displayed inside that hexagon.")
                            plot_empty_hexagons(
                                st.session_state.som, X, st.session_state.raw_df, main_type_)
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
                        st.write('---')
                        st.write('**Global Color Scaling**')
                        st.write(
                            'The color scales are globally consistent for each feature and metric type. This means:')
                        st.write(
                            '- Single-value metrics (min, median, max, mean) for the same feature share a consistent color scale')
                        st.write(
                            '- Statistical metrics (sum, std) for the same feature have their own consistent color scale')
                        st.write(
                            'This enables reliable visual comparison across different plots of the same feature.')
                    feature = st.selectbox(
                        'Feature', st.session_state.raw_df.columns.to_list())
                    scaling_options = st.multiselect(
                        'Feature Scaling (select up to 3)',
                        ['mean', 'min', 'max', 'sum', 'median', 'std'],
                        default=['mean'],
                        max_selections=3,
                        help='Select up to 3 scaling methods to display side by side')

                    main_type_counts = main_type.value_counts()
                    default_main_type_counts = []
                    for default in default_main_type:
                        default_main_type_counts.append(
                            f"{default} [{main_type_counts[default]}]")
                    sorted_main_type = [f"{name} [{count}]" for name,
                                        count in main_type_counts.items()]

                    main_type_ = st.multiselect(
                        'Main type [Number of detections]', sorted_main_type, default_main_type_counts)
                    main_type_ = [mt.split(' [')[0] for mt in main_type_]
                    strokeWidth_enhanced = st.slider(
                        'Stroke Width Enhanced', 1.0, 5.0, 3.0, help='Stroke Width Enhanced')

                    st.write(
                        "###### Please click the 'Show Plot' button after choosing the dataset type or to display the map, in order to refresh the view.")

                    # Store the selected feature and scaling options for plotting outside the form
                    if 'feature_viz_feature' not in st.session_state:
                        st.session_state.feature_viz_feature = None
                    if 'feature_viz_scaling' not in st.session_state:
                        st.session_state.feature_viz_scaling = []
                    if 'feature_viz_dataset_choice' not in st.session_state:
                        st.session_state.feature_viz_dataset_choice = None

                    st.session_state.feature_viz_feature = feature
                    st.session_state.feature_viz_scaling = scaling_options
                    st.session_state.feature_viz_dataset_choice = dataset_choice

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
                        st.write('---')
                        st.write('**Global Color Scaling**')
                        st.write(
                            'The color scales are globally consistent for each feature and metric type. This means:')
                        st.write(
                            '- Single-value metrics (min, median, max, mean) for the same feature share a consistent color scale')
                        st.write(
                            '- Statistical metrics (sum, std) for the same feature have their own consistent color scale')
                        st.write(
                            'This enables reliable visual comparison across different plots of the same feature.')
                    if st.session_state.SOM_loaded:
                        uploaded_file = st.file_uploader(
                            "Upload your CSV file", type="csv")
                        scaling_options = st.multiselect(
                            'Feature Scaling (select up to 3)',
                            ['mean', 'min', 'max', 'sum', 'median', 'std'],
                            default=['mean'],
                            max_selections=3,
                            help='Select up to 3 scaling methods to display side by side')
                        st.write(
                            "###### Please click the 'Show Plot' button after choosing the dataset type or to display the map, in order to refresh the view.")

                        # Store selections for plotting outside the form
                        st.session_state.feature_viz_uploaded_file = uploaded_file
                        st.session_state.feature_viz_scaling = scaling_options
                        st.session_state.feature_viz_dataset_choice = dataset_choice

    # Multi-plot Feature Visualization - Outside the form for multiple scaling options
    if (plot_submit and plot_type == 'Feature Visualization' and
        hasattr(st.session_state, 'feature_viz_scaling') and
            len(st.session_state.feature_viz_scaling) > 0):

        # Handle the plotting based on dataset choice
        if st.session_state.feature_viz_dataset_choice == 'Use the main dataset':
            feature = st.session_state.feature_viz_feature
            scaling_options = st.session_state.feature_viz_scaling

            var = project_feature(st.session_state.som, X,
                                  st.session_state.raw_df[feature])

            if main_type_ is not None:
                category_map = project_feature(
                    st.session_state.som, X, st.session_state.raw_df[st.session_state.simbad_type], valid_values=main_type_)

            # Precompute all scaling metrics for global scale initialization

            precompute_feature_scale_ranges(var, feature)

            is_string_var = is_string(var)

            if is_string_var:
                st.write('## SOM Feature Visualization')
                st.write(f"**Feature:** {feature}")
                st.write(
                    "**Note:** String features are displayed with the most common value per neuron.")

                if st.session_state.som.topology == 'rectangular':
                    category_plot_sources(var)
                else:
                    category_plot_sources_hex(var)
            else:
                create_multi_features_plot(
                    var, scaling_options, type_option, color_scheme,
                    feature, st.session_state.som.topology, category_map, strokeWidth_enhanced
                )

        elif (st.session_state.feature_viz_dataset_choice == 'Upload a new dataset' and
              hasattr(st.session_state, 'feature_viz_uploaded_file') and
              st.session_state.feature_viz_uploaded_file is not None):

            uploaded_file = st.session_state.feature_viz_uploaded_file
            scaling_options = st.session_state.feature_viz_scaling

            try:
                # Validate and load the dataset
                dataset = validate_and_load_dataset(uploaded_file, features)

                feature_to_trans_and_norm = [
                    'powlaw_gamma', 'bb_kt', 'var_ratio_b', 'var_ratio_h', 'var_ratio_s']

                dataset[feature_to_trans_and_norm] = transform_and_normalize(
                    dataset[feature_to_trans_and_norm], st.session_state.df_to_norm)

                # reorder the columns of the dataset to have the same order as listed in selected_features
                # Ensure selected features are present in the uploaded dataset (case-insensitive match)
                # and add the last column to the selected features (which is not in the selected_features)

                # Get the selected features from session state
                selected_features = st.session_state.selected_features

                # Lowercase mapping for uploaded dataset columns and selected features
                dataset_columns_lower = [col.lower()
                                         for col in dataset.columns]
                selected_features_lower = [f.lower()
                                           for f in selected_features]

                # Find the mapping from selected_features to dataset columns (case-insensitive)
                feature_col_map = {}
                for f in selected_features_lower:
                    if f in dataset_columns_lower:
                        idx = dataset_columns_lower.index(f)
                        feature_col_map[f] = dataset.columns[idx]
                    else:
                        st.error(
                            f"Feature '{f}' not found in uploaded dataset columns: {dataset.columns.tolist()}")
                        raise ValueError(
                            f"Feature '{f}' not found in uploaded dataset.")

                # The last column (assumed to be the feature to visualize)
                last_col = dataset.columns[-1]
                # Reorder columns: selected features (in order), then the last column
                reordered_cols = [feature_col_map[f]
                                  for f in selected_features_lower] + [last_col]
                dataset = dataset[reordered_cols]

                # All rows, only the last column
                Xx = dataset.iloc[:, :-1].to_numpy()
                feature_data = dataset.iloc[:, -1]
                # Get the name of the last column (the feature)
                uploaded_feature_name = dataset.columns[-1]

                if dataset is not None:
                    st.session_state['new_dataset'] = dataset
                    var = project_feature(
                        st.session_state.som, Xx, feature_data)

                    # Precompute all scaling metrics for global scale initialization
                    precompute_feature_scale_ranges(var, uploaded_feature_name)

                    is_string_var = is_string(var)

                    if is_string_var:
                        st.write('## SOM Feature Visualization')
                        st.write(f"**Feature:** {uploaded_feature_name}")
                        st.write(
                            "**Note:** String features are displayed with the most common value per neuron.")

                        if st.session_state.som.topology == 'rectangular':
                            category_plot_sources(var)
                        else:
                            category_plot_sources_hex(var)
                    else:
                        create_multi_features_plot(
                            var, scaling_options, type_option, color_scheme,
                            uploaded_feature_name, st.session_state.som.topology
                        )
                else:
                    st.error(
                        "Dataset validation failed. Please check the column names and ensure there are no empty values.")
            except Exception as e:
                st.error(
                    f"An error occurred: {e} at line {e.__traceback__.tb_lineno}")

    # Add a checkbox for enabling download options
    if st.session_state.som.topology == 'hexagonal':
        enable_classification = st.checkbox("Apply Classification", value=False,
                                            help="Check this box to apply classification", key="apply_classification")
        if enable_classification:
            with st.form(key='classification_form'):
                with st.expander("See explanation"):
                    st.write(
                        "This tool classifies detections with a trained SOM. It’s designed for items without a SIMBAD cross-match or for new uploads."
                    )

                    st.write("### How it works (single pass)")
                    st.markdown(
                        """
                - **BMU lookup:** each detection is mapped to its Best Matching Unit (BMU) on the SOM.
                - **Dynamic neighborhood:** we look around the BMU with growing windows (e.g. 3×3, 5×5, 7×7) until we have **enough support**.
                - **Probabilities:** from the neighborhood counts we compute class posteriors with mild **class reweighting** (optional) and **Laplace smoothing**.
                - **Decision with abstain:** we assign the top class only if:
                    - support ≥ **MIN_SUPPORT**
                    - top-class probability (purity) ≥ **MIN_PURITY**
                    - probability gap between top-2 classes ≥ **MARGIN**
                Otherwise we **abstain**.
                        """
                    )

                    st.write("### Main parameters")
                    st.markdown(
                        """
                - **MIN_SUPPORT**: minimum effective examples in the chosen window  
                - **MIN_PURITY**: minimum top-class posterior  
                - **MARGIN**: minimum gap between top-2 posteriors  
                - **WINDOWS**: neighborhood sizes tried in order (e.g. `[3,5,7]`)  
                - **ALPHA** (Laplace), **GAMMA** (class reweighting)
                        """
                    )

                    st.write("### Output columns (per detection)")
                    st.markdown(
                        """
                - **pred** (class) · **conf** (top-class probability) · **abstain** (True/False)  
                - **bmu_x, bmu_y** (BMU coords) · **support_eff** (evidence) · **purity** (top-class prob) · **window_k** (window used)
                        """
                    )

                    st.write("### Upload format (CSV)")
                    st.markdown(
                        """
                The file must be a **.csv** with the **same features used to train the SOM**, one row per detection.  
                Include an **`id`** column if you want to keep identifiers in the outputs; no class column is needed.
                        """
                    )
                    st.code(
                        "id,f1,f2,f3,...,f11\n"
                        "obj_001,0.12,1.5,3.4,...,0.07\n"
                        "obj_002,0.08,1.2,3.1,...,0.05\n"
                        "...\n",
                        language="text",
                    )

                    st.write(
                        "You can download the results (with predictions and confidences) from the download section below.")
                dataset_toclassify = None
                st.write("Apply classification")

                parameters_classification = {}
                # Set parameters

                main_type_counts = main_type.value_counts()
                default = ['YSO', 'Stars']
                default_main_type_counts = []
                for default in default:
                    default_main_type_counts.append(
                        f"{default} [{main_type_counts[default]}]")
                sorted_main_type = [f"{name} [{count}]" for name,
                                    count in main_type_counts.items()]

                main_type_selection = st.multiselect(
                    'Classes', sorted_main_type, default=default_main_type_counts, help='Classes to classify')

                parameters_classification['classes'] = [mt.split(' [')[0]
                                                        for mt in main_type_selection]

                parameters_classification['MIN_SUPPORT'] = st.slider(
                    'Minimum Support', 0, 1000, 20, help='Minimum Support')
                parameters_classification['MIN_PURITY'] = st.slider(
                    'Minimum Purity', 0.0, 1.0, 0.55, help='Minimum Purity')
                parameters_classification['MARGIN'] = st.slider(
                    'Margin', 0.0, 0.5, 0.05, help='Margin')
                parameters_classification['ALPHA'] = st.slider(
                    'Alpha', 0.0, 3.0, 2.0, help='Alpha')
                parameters_classification['GAMMA'] = st.slider(
                    'Gamma', 0.0, 1.0, 0.5, help='Gamma')
                parameters_classification['SIGMA_F'] = st.slider(
                    'Sigma F', 0.0, 0.0886*1.5, 0.0886, help='Sigma F')
                parameters_classification['WINDOWS'] = st.multiselect(
                    'Windows', [3, 5, 7, 9], default=[3, 5, 7], help='Windows')

                dataset_choice = st.radio(
                    'Choose the dataset', ['Use the main dataset', 'Upload a new dataset'])
                if dataset_choice == 'Use the main dataset':
                    dataset_toclassify = st.session_state.df_index[pd.isna(
                        st.session_state.raw_df[st.session_state.simbad_type])]
                    # now extract the dataset that already have a cross-match
                    dataset_crossmatched = st.session_state.df_index[pd.notna(
                        st.session_state.raw_df[st.session_state.simbad_type])]
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
                    st.session_state.raw_df[st.session_state.simbad_type])]

                SIMBAD_classes = set(
                    st.session_state.raw_df[st.session_state.simbad_type])

                classify = st.form_submit_button('Get Classification')

                if classify and dataset_toclassify is not None:
                    # Split st.session_state.df_index into with and without cross-match, then the one with a label
                    dataset_toclassify_with_crossmatch = st.session_state.df_index[pd.notna(
                        st.session_state.raw_df[st.session_state.simbad_type])]
                    id_name_type_with_crossmatch = simbad_dataset[[
                        "id", "name", st.session_state.simbad_type]]

                    dataset_toclassify_with_crossmatch = pd.merge(
                        dataset_toclassify_with_crossmatch, id_name_type_with_crossmatch, on="id", how="left")

                    # split it into train, val, test, but be cafeful to not have source leakage (use stratified group kfold)
                    # keep variable names: dataset_toclassify_with_crossmatch, sgkf, fold, train, val, test
                    sgkf = StratifiedGroupKFold(
                        n_splits=20, shuffle=True, random_state=42)
                    fold = np.empty(
                        len(dataset_toclassify_with_crossmatch), dtype=int)

                    for k, (_, test_idx) in enumerate(
                        sgkf.split(
                            X=np.zeros(
                                len(dataset_toclassify_with_crossmatch)),
                            y=dataset_toclassify_with_crossmatch[st.session_state.simbad_type],
                            groups=dataset_toclassify_with_crossmatch["name"]
                        )
                    ):
                        fold[test_idx] = k

                    dataset_toclassify_with_crossmatch = dataset_toclassify_with_crossmatch.copy()
                    dataset_toclassify_with_crossmatch["fold"] = fold

                    # 0–13 ≈70% train, 14–16 ≈15% val, 17–19 ≈15% test
                    train = dataset_toclassify_with_crossmatch[
                        dataset_toclassify_with_crossmatch.fold.isin(
                            set(range(0, 14)))
                    ].copy()
                    val = dataset_toclassify_with_crossmatch[
                        dataset_toclassify_with_crossmatch.fold.isin(
                            set(range(14, 17)))
                    ].copy()
                    test = dataset_toclassify_with_crossmatch[
                        dataset_toclassify_with_crossmatch.fold.isin(
                            set(range(17, 20)))
                    ].copy()

                    assert set(train.name) & set(val.name) == set()
                    assert set(train.name) & set(test.name) == set()
                    assert set(val.name) & set(test.name) == set()

                    TARGET = parameters_classification['classes']

                    # Filter each dataset to keep only rows with TARGET classes
                    train = train[train[st.session_state.simbad_type].isin(
                        TARGET)].copy()
                    val = val[val[st.session_state.simbad_type].isin(
                        TARGET)].copy()
                    test = test[test[st.session_state.simbad_type].isin(
                        TARGET)].copy()

                    # Then assign the labels (since all remaining rows are in TARGET)
                    for d in (train, val, test):
                        d.loc[:, 'label'] = d[st.session_state.simbad_type]

                    parameters_classification['classes'] = list(
                        dict.fromkeys(TARGET))

                    # size of traim, val, test
                    st.write(
                        f"Size of train: {len(train)}, size of val: {len(val)}, size of test: {len(test)}")
                    # and devided by classes
                    st.write(
                        f"Size of train by classes: {train['label'].value_counts()}")
                    st.write(
                        f"Size of val by classes: {val['label'].value_counts()}")
                    st.write(
                        f"Size of test by classes: {test['label'].value_counts()}")

                    # append the two datasets using online the features columns to train the SOM and order by id
                    train_x_Y = pd.concat(
                        [train, dataset_toclassify]).sort_values(by="id")[st.session_state.selected_features].reset_index(drop=True).to_numpy()
                    train_x_Y_index = pd.concat(
                        [train, dataset_toclassify]).sort_values(by="id")[["id"] + st.session_state.selected_features].reset_index(drop=True).to_numpy()
                    test_x_Y_index = test.sort_values(by="id")[
                        ["id"] + st.session_state.selected_features].reset_index(drop=True).to_numpy()
                    val_x_Y_index = val.sort_values(by="id")[
                        ["id"] + st.session_state.selected_features].reset_index(drop=True).to_numpy()

                    # add a loader to train the SOM
                    with st.spinner('Training the SOM for classification...'):
                        if not os.path.exists("/Users/andre/Desktop/INAF/USA/dataset/code/SOM/SOM-CSC/models/SOM_classification.pkl"):
                            st.write(
                                "Model not found, Training the SOM for classification...")
                            st.session_state.som_classification = train_som(train_x_Y, dim, dim, len(features), sigma,
                                                                            learning_rate, iterations, topology, seed)
                            # save the SOM for classification in a file
                            with open("/Users/andre/Desktop/INAF/USA/dataset/code/SOM/SOM-CSC/models/SOM_classification.pkl", "wb") as f:
                                pickle.dump(
                                    st.session_state.som_classification, f)
                        else:
                            st.write(
                                "Model found, Loading the SOM for classification...")
                            with open("/Users/andre/Desktop/INAF/USA/dataset/code/SOM/SOM-CSC/models/SOM_classification.pkl", "rb") as f:
                                st.session_state.som_classification = pickle.load(
                                    f)

                        # print QE and TE
                        st.write(
                            f"QE: {st.session_state.som_classification.quantization_error(train_x_Y)}")
                        st.write(
                            f"TE: {st.session_state.som_classification.topographic_error(train_x_Y)}")

                    som_map_id_train = download_activation_response(
                        st.session_state.som_classification, train_x_Y_index)
                    som_map_id_test = download_activation_response(
                        st.session_state.som_classification, test_x_Y_index)
                    som_map_id_val = download_activation_response(
                        st.session_state.som_classification, val_x_Y_index)

                    # DEBUG
                    #

                    # sigma_f = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,
                    #           0.08, 0.0886, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
                    # output = []

                    # with st.spinner('Getting classification analysis from splits...'):
                    #    for sigma_f in sigma_f:
                    #        st.write(f"Sigma F: {sigma_f}")
                    #        parameters_classification['SIGMA_F'] = sigma_f
                    #        analysis_from_splits = get_classification_analysis_from_splits(
                    #            som_map_id_train, som_map_id_val, som_map_id_test, train, val, test, parameters_classification, dim, st.session_state.som_classification)
                    #        output.append(
                    #            {'sigma_f': sigma_f, 'best_by_floor': analysis_from_splits['best_by_floor']})

                    # st.write(output)
                    out = get_classification(
                        som_map_id_train, som_map_id_test, som_map_id_val, dataset_toclassify, dataset_toclassify_with_crossmatch, train, val, test, parameters_classification, dim, st.session_state.som_classification)
                    # classify = False
                    dataset_classified = out['predictions']

                    # Add source column where the id matches from raw_df
                    if 'name' in st.session_state.raw_df.columns:
                        id_to_source = dict(
                            zip(st.session_state.raw_df['id'], st.session_state.raw_df['name']))
                        dataset_classified['source name'] = dataset_classified['id'].map(
                            id_to_source)

                    st.session_state.dataset_classified = dataset_classified[['id', 'pred',
                                                                              'conf', 'abstain', 'source name']]

                    classification_results = describe_classified_dataset(
                        dataset_classified)

                    def render_split_metrics(split_name: str, metrics: dict, classes: list[str]):
                        if not metrics or metrics.get("confusion_matrix") is None:
                            st.info(
                                f"No {split_name.lower()} metrics available.")
                            return

                        cov = metrics["coverage"]
                        macro_f1 = metrics["macro_f1"]
                        kept = metrics["n_kept"]
                        total = metrics["n_total"]
                        per_class_f1 = metrics["per_class_f1"]

                        st.subheader(f"{split_name} results")
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric(
                                "Coverage", f"{cov:.3f}", help="Kept / total on this split")
                        with c2:
                            st.metric("Macro-F1", f"{macro_f1:.3f}")
                        with c3:
                            st.metric("Kept / Total", f"{kept} / {total}")

                        # Per-class F1 table
                        f1_df = pd.DataFrame(
                            {"F1": per_class_f1}).reindex(classes)
                        st.write("Per-class F1")
                        st.dataframe(f1_df.style.format({"F1": "{:.3f}"}))

                        # Confusion matrix (kept only)
                        cm_df = pd.DataFrame(metrics["confusion_matrix"])
                        # reorder rows/cols if needed
                        cm_df = cm_df.reindex(index=[f"T:{c}" for c in classes],
                                              columns=[f"P:{c}" for c in classes])
                        st.write(
                            "Confusion matrix (kept only — rows=true, cols=pred)")
                        st.dataframe(cm_df)

                        # Micro-F1 / accuracy on kept
                        cm = cm_df.to_numpy()
                        micro = (np.trace(cm) / cm.sum()
                                 ) if cm.sum() > 0 else 0.0
                        st.caption(f"Micro-F1 / Accuracy on kept: {micro:.3f}")

                    # ===============================
                    # Use it with your model outputs
                    # ===============================

                    # Assume you've already run:
                    # out = get_classification(...)

                    classes = out["params_used"]["classes"]

                    # --- 1) Show VALIDATION first (always visible) ---
                    st.header("Validation Evaluation")
                    render_split_metrics("Validation", out.get(
                        "val_metrics", {}), classes)

                    # --- 2) TEST results: hidden by default ---
                    with st.expander("Show TEST results (hidden by default)"):
                        render_split_metrics("TEST", out.get(
                            "test_metrics", {}), classes)

                    st.title("Summary of Classification Assignment")

                    st.header("Overview")
                    st.write("### Key Metrics")
                    st.write(
                        f"**Total detections (before assignment):** {classification_results['total_unclassified_before']}")
                    st.write(
                        f"**Assigned (kept) predictions:** {classification_results['total_classified_rows']}")
                    st.write(
                        f"**Abstained predictions:** {classification_results['num_unclassified_after']}")
                    st.write(
                        f"**Coverage (kept/total):** {classification_results['coverage']:.3f}")
                    st.write(
                        f"**Percentage assigned:** {classification_results['percentage_assigned']:.2f}%")

                    st.write("### Class distribution (kept only)")

                    dist_df = pd.DataFrame({
                        "count": classification_results["assigned_class_counts"],
                        "percent": classification_results["assigned_class_percent"].round(2)
                    })
                    dist_df.index.name = "class"
                    st.dataframe(dist_df)

                    st.write("### Confidence by class (kept only)")
                    st.dataframe(classification_results["confidence_by_class"])

                    st.write("### Neighborhood diagnostics")
                    # window usage (all vs kept)
                    win_all = classification_results["window_k_counts_all"]
                    win_kept = classification_results["window_k_counts_assigned"].reindex(
                        win_all.index, fill_value=0)
                    win_df = pd.DataFrame({"all": win_all, "kept": win_kept})
                    win_df.index.name = "window_k"
                    st.dataframe(win_df)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Support stats (kept)")
                        st.dataframe(
                            classification_results["support_stats_assigned"])
                    with col2:
                        st.write("Purity stats (kept)")
                        st.dataframe(
                            classification_results["purity_stats_assigned"])

    enable_download = st.checkbox("Enable Downloads", value=False,
                                  help="Check this box to enable download options for the SOM model and datasets.", key="apply_download")
    if enable_download:
        with st.spinner("Loading download options..."):
            with st.expander("Download Available"):
                # Download the SOM model
                model_bytes = download_som_model_bytes(
                    st.session_state.som, features)
                show_download_button(
                    label="Download the SOM model",
                    data=model_bytes,
                    file_name='SOM_model.pkl',
                    mime='application/octet-stream',
                    help='Download the SOM model, store it, and upload it later'
                )

                # Download the raw dataset
                raw_dataset_csv = download_raw_dataset_csv(
                    st.session_state.raw_df)
                show_download_button(
                    label="Download the raw dataset",
                    data=raw_dataset_csv,
                    file_name='raw_dataset.csv',
                    mime='text/csv',
                    help='Download the raw dataset'
                )

                # Download the activation response
                activation_map_csv = download_activation_map_csv(
                    st.session_state.som, X_index)
                show_download_button(
                    label="Download the activation response map",
                    data=activation_map_csv,
                    file_name='activation_response.csv',
                    mime='text/csv',
                    help='Download the activation response'
                )

                # Download the classification results if available
                if 'dataset_classified' in st.session_state and enable_classification:
                    classified_csv = download_classified_csv(
                        st.session_state.dataset_classified)
                    show_download_button(
                        label="Download the classification results",
                        data=classified_csv,
                        file_name='dataset_classified.csv',
                        mime='text/csv',
                        help='Download the classification results'
                    )
else:
    st.write(
        'Please load a SOM model or generate a new one to visualize the map.')
