import streamlit as st
import pandas as pd
import numpy as np
import pickle
from som_fun import *
from tools import *
import glob
import os
import matplotlib.pyplot as plt

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

# Load data first
raw_dataset_path = './data/csc211_mastertable_clean_observationlevel_COMPLETE_xmatchSimbad1arcsec_log_norm_id/'

# Initialize session state for dataframes
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = load_split_csvs(raw_dataset_path)

if 'full_df' not in st.session_state:
    st.session_state.full_df = st.session_state.raw_df[['hard_hm', 'hard_hs', 'hard_ms', 'powlaw_gamma_log_norm', 'var_prob_b', 'var_prob_s', 'var_prob_h',
                                                        'bb_kt_log_norm', 'var_ratio_b_log_norm', 'var_ratio_h_log_norm', 'var_ratio_s_log_norm', 'var_newq_b_log_norm']]
    st.session_state.full_df.columns = st.session_state.full_df.columns.str.replace(
        '_log_norm', '')

if 'df' not in st.session_state:
    st.session_state.df = st.session_state.full_df.copy()

if 'full_df_index' not in st.session_state:
    st.session_state.full_df_index = st.session_state.raw_df[['id', 'hard_hm', 'hard_hs', 'hard_ms', 'powlaw_gamma_log_norm', 'var_prob_b', 'var_prob_s', 'var_prob_h',
                                                              'bb_kt_log_norm', 'var_ratio_b_log_norm', 'var_ratio_h_log_norm', 'var_ratio_s_log_norm', 'var_newq_b_log_norm']]
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
simbad_type = 'main_type'  # otype
main_type = st.session_state.raw_df[simbad_type]
# default_main_type = ['QSO', 'AGN', 'Seyfert_1', 'Seyfert_2', 'HMXB',
#                     'LMXB', 'XB', 'YSO', 'TTau*', 'Orion_V*']

default_main_type = ['YSO', 'XrayBin', 'Seyfert', 'AGN']
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


# sidebar for the dataset
st.sidebar.header('User input')

# toogle to select new vs old dataset
# new dataset
raw_dataset_path = './data/csc211_mastertable_clean_observationlevel_COMPLETE_xmatchSimbad1arcsec_log_norm_id/'

# Let the user load a SOM model
som_model = st.sidebar.file_uploader(
    "Upload your SOM model", type=['pkl'], accept_multiple_files=False, help='Upload your SOM model for visualization. ⚠️ **Note: Only .pkl files that were previously downloaded from this web application following the training process are compatible for upload.**')
# button to load the SOM model

if som_model is not None:
    file = pickle.load(som_model)
    st.session_state.som, features = file[0], file[1]
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
    if 'var_newq_b' in features_for_selection:
        default_features = [
            f for f in features_for_selection if f != 'var_newq_b']
    else:
        default_features = features_for_selection.copy()

    features = st.sidebar.multiselect(
        'Features', features_for_selection, default_features, help='Select the features to be used for training the SOM.')
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
    sigma = st.sidebar.slider('Sigma', 0.01, 5.0, 3.8,
                              help='The spread of the neighborhood function')
    learning_rate = st.sidebar.slider(
        'Learning rate', 0.01, 5.0, 1.8, help='The degree of weight updates')
    iterations = st.sidebar.slider(
        'Iterations', 0, 100000, 58000, 1000, help='Number of training iterations')
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
            if plot_type in ['U-Matrix', 'Activation Response', 'Training Feature Space Map', 'Feature Visualization']:
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
                    'Customize colors for sources', key='customize_colors_source_name')

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
                                    f"{source}", st.session_state.custom_colors[source], key=f"color_{source}")
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
                                st.session_state.som, X, st.session_state.raw_df['name'], sources)
                            category_plot_sources(
                                category_map, custom_colors=source_colors)
                    elif st.session_state.som.topology == 'hexagonal':
                        if visualization_type == 'Scatter':
                            scatter_plot_sources_hex(
                                st.session_state.som, sources, st.session_state.raw_df, X, 'name',
                                custom_colors=source_colors, jitter_amount=jitter_amount, show_grid=show_grid)
                        elif visualization_type == 'Hexbin':
                            category_map = project_feature(
                                st.session_state.som, X, st.session_state.raw_df['name'], sources)
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
                                st.session_state.som, X, st.session_state.raw_df['name'], sources)
                            category_plot_sources(
                                category_map, custom_colors=source_colors)
                    elif st.session_state.som.topology == 'hexagonal':
                        if visualization_type == 'Scatter':
                            scatter_plot_sources_hex(
                                st.session_state.som, sources, st.session_state.raw_df, X, 'name',
                                custom_colors=source_colors, jitter_amount=jitter_amount, show_grid=show_grid)
                        elif visualization_type == 'Hexbin':
                            category_map = project_feature(
                                st.session_state.som, X, st.session_state.raw_df['name'], sources)
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

                st.write(
                    "###### To update the map with the name of the selected sources, please click the 'Show Plot' button again.")

                if len(main_type_) > 0:
                    type_colors = {mt: st.session_state.custom_colors.get(
                        mt, None) for mt in main_type_} if customize_colors else None

                    if st.session_state.som.topology == 'rectangular':
                        if visualization_type == 'Scatter':
                            scatter_plot_sources(
                                st.session_state.som, main_type_, st.session_state.raw_df, X, simbad_type,
                                custom_colors=type_colors, jitter_amount=jitter_amount, show_grid=show_grid)
                        elif visualization_type == 'Rectangular':
                            category_map = project_feature(
                                st.session_state.som, X, st.session_state.raw_df[simbad_type], main_type_)
                            category_plot_sources(
                                category_map, custom_colors=type_colors)
                    elif st.session_state.som.topology == 'hexagonal':
                        if visualization_type == 'Scatter':
                            scatter_plot_sources_hex(
                                st.session_state.som, main_type_, st.session_state.raw_df, X, simbad_type,
                                custom_colors=type_colors, jitter_amount=jitter_amount, show_grid=show_grid)
                        elif visualization_type == 'Hexbin':
                            category_map = project_feature(
                                st.session_state.som, X, st.session_state.raw_df[simbad_type], main_type_)
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
                    scaling = st.selectbox(
                        'Feature Scaling', ['mean', 'min', 'max', 'sum', 'median', 'std'])
                    st.write(
                        "###### Please click the 'Show Plot' button after choosing the dataset type or to display the map, in order to refresh the view.")
                    var = project_feature(
                        st.session_state.som, X, st.session_state.raw_df[feature])

                    # Precompute all scaling metrics for global scale initialization
                    precompute_feature_scale_ranges(var, feature)

                    is_string_var = is_string(var)

                    if st.session_state.som.topology == 'rectangular':
                        if is_string_var:
                            category_plot_sources(var)
                        else:
                            features_plot(var, type_option,
                                          color_scheme, scaling=scaling, feature_name=feature)
                    else:
                        if is_string_var:
                            category_plot_sources_hex(var)
                        else:
                            features_plot_hex(
                                var, type_option, color_scheme, scaling=scaling, feature_name=feature)
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
                            feature_data = dataset.iloc[:, -1]
                            # Get the name of the last column (the feature)
                            uploaded_feature_name = dataset.columns[-1]

                            if dataset is not None:
                                st.session_state['new_dataset'] = dataset
                                # Call to project_feature and features_plot goes here, using the new dataset
                                var = project_feature(
                                    st.session_state.som, Xx, feature_data)

                                # Precompute all scaling metrics for global scale initialization
                                precompute_feature_scale_ranges(
                                    var, uploaded_feature_name)

                                is_string_var = is_string(var)

                                if st.session_state.som.topology == 'rectangular':
                                    if is_string_var:
                                        category_plot_sources(var)
                                    else:
                                        features_plot(
                                            var, type_option, color_scheme, scaling=scaling, feature_name=uploaded_feature_name)
                                else:
                                    if is_string_var:
                                        category_plot_sources_hex(
                                            var)
                                    else:
                                        features_plot_hex(
                                            var, type_option, color_scheme, scaling=scaling, feature_name=uploaded_feature_name)
                            else:
                                st.error(
                                    "Dataset validation failed. Please check the column names and ensure there are no empty values.")
                            # except Exception as e:
                            #    st.error(f"An error occurred: {e}")

    # Add a checkbox for enabling download options
    if st.session_state.som.topology == 'hexagonal':
        enable_classification = st.checkbox("Apply Classification", value=False,
                                            help="Check this box to apply classification", key="apply_classification")
        if enable_classification:
            with st.form(key='classification_form'):
                with st.expander("See explanation"):
                    st.write("This tool provides the functionality to classify detections that lack a cross-match in the SIMBAD dataset or belong to a new set of detections that can be uploaded (See below for more details on this). Classification is performed at two levels:")

                    st.write("1. **Central Neuron Classification**: For every detection, the Best Matching Unit (BMU) is identified, and the most frequent class within that neuron is assigned. Users can set thresholds for:")
                    st.write(
                        """
                        <ul style="margin-left: 20px;">
                            <li>The minimum number of detections that the top 3 classes within the neuron must contain.</li>
                            <li>The minimum percentage of detections that the majority class must have relative to the top 3 classes within the neuron to be classified.</li>
                        </ul>
                        """,
                        unsafe_allow_html=True
                    )

                    st.write("2. **Neighbor Neuron Classification**: Since neighboring neurons are close in the feature space, this method assigns the most frequent class among them. Parameters for this classification include:")
                    st.write(
                        """
                        <ul style="margin-left: 20px;">
                            <li>The minimum number of detections from the top 3 classes required within each neuron to be included in the analysis.</li>
                            <li>The minimum number of neighboring neurons that must share the same majority class.</li>
                        </ul>
                        """,
                        unsafe_allow_html=True
                    )
                    st.write(
                        "Both classification methods are handled independently, and their results can be downloaded in the download section below.")

                    st.write(
                        '⚠️ **The uploaded file must to be a .csv file with a specific structure:** ⚠️')
                    st.write(
                        'The columns should contain the features utilized for training the SOM.')
                    st.write(
                        '**A, B, C,**')
                    st.write(
                        '*x1, y1, z1*')
                    st.write(
                        '*x2, y2, z2*')
                    st.write(
                        '*...*')
                    st.write(
                        '*xN, yN, zN,*')
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
                    assignments_central, assignments_neighbor, all_confidences_central, all_confidences_neighbor = get_classification(
                        som_map_id, dataset_toclassify, simbad_dataset, SIMBAD_classes, parameters_classification, dim, st.session_state.som)
                    # classify = False
                    dataset_classified = update_dataset_to_classify(
                        dataset_toclassify, assignments_central, assignments_neighbor)

                    st.session_state.dataset_classified = dataset_classified[['id', 'assigned_class_central',
                                                                              'confidence_central', 'assigned_class_neighbor', 'confidence_neighbor', 'is_classified']]

                    classification_results = describe_classified_dataset(
                        dataset_classified, assignments_central, assignments_neighbor, all_confidences_central, all_confidences_neighbor)

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
                    if not classification_results['all_confidences_central'].empty:
                        with st.popover("Histogram: Central Neuron Confidence Levels"):
                            st.write(
                                "Histogram of Confidence Levels for Central Neurons")
                            st.write(
                                "Green bars represent confidence values that passed the threshold, while light red bars represent values below the threshold.")
                            fig, ax = plt.subplots(figsize=(10, 6))

                            # Split data into passed and failed thresholds
                            passed_df = classification_results['all_confidences_central'][
                                classification_results['all_confidences_central']['passed_threshold']]
                            failed_df = classification_results['all_confidences_central'][
                                ~classification_results['all_confidences_central']['passed_threshold']]

                            # Create bins for the histogram
                            # Ensure a bin edge falls exactly at the threshold value
                            threshold = parameters_classification['confidence_threshold']
                            # Create bins with an edge exactly at the threshold
                            bins = np.concatenate([
                                # 10 bins from 0 to threshold
                                np.linspace(0, threshold, 21),
                                # 10 bins from threshold to 1, excluding duplicate threshold
                                np.linspace(threshold, 1, 24)[1:]
                            ])

                            # Plot both histograms with a clearer visual distinction
                            if not failed_df.empty:
                                # Plot values below threshold
                                ax.hist(failed_df['confidence_central'],
                                        # Only use bins up to threshold
                                        bins=bins[bins <= threshold],
                                        range=(0, threshold),
                                        color='lightcoral',
                                        edgecolor='black',
                                        alpha=0.5,
                                        label='Below Threshold')

                            if not passed_df.empty:
                                # Plot values above threshold
                                ax.hist(passed_df['confidence_central'],
                                        # Only use bins from threshold up
                                        bins=bins[bins >= threshold],
                                        range=(threshold, 1),
                                        color='green',
                                        edgecolor='black',
                                        alpha=0.7,
                                        label='Above Threshold')

                            # Add a vertical line at the threshold
                            ax.axvline(x=parameters_classification['confidence_threshold'] + 0.001,
                                       color='red', linestyle='--',
                                       label=f'Threshold ({parameters_classification["confidence_threshold"]})')

                            ax.set_title(
                                'Histogram of Assignment Confidence Levels (Central)', fontsize=16)
                            ax.set_xlabel('Confidence Level', fontsize=14)
                            ax.set_ylabel('Number of Detections', fontsize=14)
                            ax.grid(axis='y', alpha=0.7)
                            ax.legend()
                            st.pyplot(fig)

                    # Bar chart of assigned classes (Neighbor)
                    if not classification_results['assigned_class_counts_neighbor'].empty:
                        with st.popover("Bar Chart: Neighbor Neuron Assignments"):
                            st.write(
                                "Number of Detections Assigned to Each Class (Neighbor)")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            classification_results['assigned_class_counts_neighbor'].plot(
                                kind='bar', color='skyblue', edgecolor='black', width=0.8, ax=ax)
                            ax.set_title(
                                'Number of Detections Assigned to Each Class (Neighbor)', fontsize=16)
                            ax.set_xlabel('Assigned Class', fontsize=14)
                            ax.set_ylabel('Number of Detections', fontsize=14)
                            ax.grid(axis='y', alpha=0.7)
                            ax.tick_params(axis='x', rotation=45, labelsize=12)
                            ax.tick_params(axis='y', labelsize=12)
                            st.pyplot(fig)

                    # Histogram of confidence levels (Neighbor)
                    if not classification_results['all_confidences_neighbor'].empty:
                        # Histogram of confidence levels (Neighbor)
                        with st.popover("Histogram: Neighbor Neuron Confidence Levels"):
                            st.write(
                                "Histogram of Confidence Levels for Neighbor Neurons")
                            st.write(
                                "Green bars represent confidence values that passed the threshold, while light red bars represent values below the threshold.")
                            fig, ax = plt.subplots(figsize=(10, 6))

                            # Split data into passed and failed thresholds
                            passed_df = classification_results['all_confidences_neighbor'][
                                classification_results['all_confidences_neighbor']['passed_threshold']]
                            failed_df = classification_results['all_confidences_neighbor'][
                                ~classification_results['all_confidences_neighbor']['passed_threshold']]

                            # Create bins for the histogram - neighbor uses integer values (count of neighbors)
                            max_neighbors = 6  # Maximum number of neighbors in hexagonal grid
                            # Create standard integer-centered bins
                            bins = np.arange(-0.5, max_neighbors + 1.5, 1)

                            # Define the threshold value for classification
                            threshold = parameters_classification['neighbor_majority_threshold']

                            # Plot both histograms with a clearer visual distinction
                            if not failed_df.empty:
                                # Plot values below threshold
                                below_threshold_mask = failed_df['confidence_neighbor'] < threshold
                                if below_threshold_mask.any():
                                    ax.hist(failed_df['confidence_neighbor'][below_threshold_mask],
                                            bins=bins,
                                            color='lightcoral',
                                            edgecolor='black',
                                            alpha=0.5,
                                            label='Below Threshold')

                            if not passed_df.empty:
                                # Plot values above threshold
                                above_threshold_mask = passed_df['confidence_neighbor'] >= threshold
                                if above_threshold_mask.any():
                                    ax.hist(passed_df['confidence_neighbor'][above_threshold_mask],
                                            bins=bins,
                                            color='green',
                                            edgecolor='black',
                                            alpha=0.7,
                                            label='Above Threshold')

                            # Add a vertical line at the threshold
                            # Position the line between bars to avoid cutting through a bar
                            threshold = parameters_classification['neighbor_majority_threshold']
                            # For integer thresholds, place line 0.5 below
                            # For non-integer thresholds, place line at the threshold
                            threshold_line_position = threshold - \
                                0.5 if threshold == int(
                                    threshold) else threshold
                            ax.axvline(x=threshold_line_position,
                                       color='red', linestyle='--',
                                       label=f'Threshold ({int(threshold) if threshold == int(threshold) else threshold})')

                            ax.set_title(
                                'Histogram of Assignment Confidence Levels (Neighbor)', fontsize=16)
                            ax.set_xlabel(
                                'Number of Neighbors with Same Class', fontsize=14)
                            ax.set_ylabel('Number of Detections', fontsize=14)
                            ax.set_xlim([0, max_neighbors + 1])
                            ax.set_xticks(range(max_neighbors + 1))
                            ax.grid(axis='y', alpha=0.7)
                            ax.legend()
                            st.pyplot(fig)

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
