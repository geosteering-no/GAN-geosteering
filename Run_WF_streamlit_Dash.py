import warnings
# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
#warnings.filterwarnings("always")
from pathOPTIM import pathfinder
import numpy as np
from write_data_var import new_data
from pipt.loop.assimilation import Assimilate
from GAN import GanLog
from input_output import read_config
from pipt import pipt_init
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plot_for_app import earth
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib
from DP import perform_dynamic_programming

from copy import deepcopy as dp
import time

global_extent = [0, 640, -16.25, 15.75]
norm = Normalize(vmin=0.0, vmax=1)

# build a streamlit app to run the workflow. On the first run of the app we will be in the initial state.
# The user has to specify the start position of the well. Number of decissions are always 1, and the user have to specify
# wheter to drill ahead, or to stop drilling. If the user decides to stop drilling, the app will stop. If the user decides to drill ahead, the main function will be called.

if 'first_position' not in st.session_state:
    st.session_state['first_position'] = True
# Also, plot the current state as a main feature of the app.
st.title('DISTIGUISH Workflow')

# Show a slider first to select the start position of the well
if st.session_state.first_position:
    state = np.load('orig_prior_small.npz')['m']  # the prior latent vector
    start_position = (st.slider(label='Enter the horizontal start position of the well', key='start_position',
              min_value=0, max_value=64, value=int(32)),0)
    st.session_state['path'] = [start_position]
else:
    state = st.session_state.state
    start_position = st.session_state.start_position_state
    st.session_state['path'].append(start_position)

# toggle first step
def toggle_first_step():
    st.session_state['first_position'] = False
    st.rerun()

# plot the current state
@st.cache_data
def get_gan_earth(state):
    gan,true_facies = earth(state)
    return gan,true_facies

@st.cache_data
def da(state,start_position):
    num_decissions = 1 #64 # number of decissions to make

    #start_position = (32, 0) # initial position of the well
    #state = np.load('orig_prior.npz')['m'] # the prior latent vector
    np.savez('prior.npz', **{'m': state}) # save the prior as a file

    kf = {'bit_pos': [start_position],
          'vec_size': 60,
          'reporttype': 'pos',
          'reportpoint': [int(el) for el in range(1)],
          'datatype': [f'res{el}' for el in range(1, 14)],
          'parallel': 1}

    sim = GanLog(kf)

    kd, kf = read_config.read_txt('es.pipt')  # read the config file.

    for i in range(num_decissions):
        # start by assimilating data at the current position

        #make a set of syntetic data for the current position
        new_data({'bit_pos': [start_position],
                  'vec_size': 60,
                  'reporttype': 'pos',
                  'reportpoint': [int(el) for el in range(1)],
                  'datatype': [f'res{el}' for el in range(1,14)]})
        # do inversion
        sim.update_bit_pos([start_position])
        analysis = pipt_init.init_da(kd, kf, sim)  # Re-initialize the data assimilation to read the new data
        assimilation = Assimilate(analysis)
        assimilation.run()

        state = np.load('posterior_state_estimate.npz')['m'] # import the posterior state estimate
        return state

gan,true_facies = get_gan_earth(state)

fig = px.imshow(gan.mean(axis=0),aspect='auto', color_continuous_scale='viridis')

# Position the colorbar horizontally below the figure
fig.update_layout(coloraxis_colorbar=dict(
    orientation='h',
    x=0.5,
    y=-0.2,
    xanchor='center',
    yanchor='top',
    len=0.8,  # Length of the colorbar
))

fig.add_scatter(x=[start_position[1]], y=[start_position[0]], mode='markers', marker=dict(color='red', size=10),name='Start Position')

st.write(f'The current position of the well is at: {start_position}')

if st.checkbox('Show Human suggestion'):
    user_step_select = st.radio('What is the next step?', ['Drill up', 'Drill ahead', 'Drill down'])
    if user_step_select == 'Drill up':
        next_position = (start_position[0] - 1, start_position[1]+1)
    elif user_step_select == 'Drill ahead':
        next_position = (start_position[0], start_position[1]+1)
    else:
        next_position = (start_position[0] + 1, start_position[1]+1)

    fig.add_scatter(x=[next_position[1]], y=[next_position[0]], mode='markers',
                    marker=dict(color='blue', size=10),name='Next Position')

# ask the user to show all the DP paths
if st.checkbox('Show Robot paths'):
    # calculate the robot paths
    next_optimal,_ = pathfinder().run(state, start_position)
    optimal_paths = [perform_dynamic_programming(gan[j, :, :], start_position,
                    cost_vector=pathfinder().get_cost_vector())[2] for j in range(state.shape[1])]
    # plot the optimal paths in the plotly figure
    for j in range(state.shape[1]):
        path_rows, path_cols = zip(*(optimal_paths[j]))
        noise_mult = 0.2
        # noise_mult = 0
        row_list = [el + noise_mult * np.random.randn() for el in path_rows]
        fig.add_trace(go.Scatter(x=path_cols, y=path_rows, mode='lines', line=dict(color='black'), showlegend=False))

if st.checkbox('Show Robot suggestion'):
    next_optimal, _ = pathfinder().run(state, start_position)
    fig.add_scatter(x=[next_optimal[1]], y=[next_optimal[0]], mode='markers',
                    marker=dict(color='black', size=10, symbol='cross'),
                    name='Robot suggestion')

if st.checkbox('Cheat!'):
    x = np.array(range(64))
    y = np.array(range(64))
    X, Y = np.meshgrid(x, y)
    Z = true_facies[0, :, :]
    fig.add_trace(go.Contour(z=Z, x=x, y=y, contours=dict(
        start=-1,
        end=1,
        size=1.9,
        coloring='none'
              ),
    line=dict(
        width=2,
        color='white',
        dash='dash'
    ),
    showscale=False  # Hide the color scale for the contour
        ))

path_rows, path_cols = zip(*(st.session_state['path']))
fig.add_trace(go.Scatter(x=path_cols, y=path_rows, mode='lines',
                         line=dict(color='red',width=4), showlegend=False))

st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    if st.button('Drill like a Human'):
        st.session_state.start_position_state = next_position
        state = da(state, next_position)
        st.session_state.state = state
        toggle_first_step()
with col2:
    if st.button('Drill like a Robot'):
        st.session_state.start_position_state = next_optimal
        state = da(state, next_optimal)
        st.session_state.state = state
        toggle_first_step()