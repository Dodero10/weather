import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box
from PIL import Image
import io
import imageio
from matplotlib.colors import LinearSegmentedColormap, Normalize
import plotly.graph_objects as go
import time
import os
from google.cloud import bigquery


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./key.json"
client = bigquery.Client()

st.set_page_config(page_title='Weather data',  layout='wide', page_icon=':ambulance:')

shape = gpd.read_file('world-administrative-boundaries').to_crs("EPSG:4326")
bbox = box(102, 8, 112, 24)
cropped_shape = shape.clip(bbox)


# Available attributes for selection
attributes = {
    "U10": "u10",
    "V10": "v10",
    "2m Dewpoint Temperature": "d2m",
    "2m Temperature": "t2m",
    "Mean Sea Level Pressure": "msl",
    "Sea Surface Temperature": "sst",
    "Surface Pressure": "sp",
    "Total Cloud Cover": "tcc",
    "Total Column Cloud Ice Water": "tciw",
    "Total Column Liquid Water": "tcl"
}

st.title("Weather Data Visualization")

# Attribute selection
selected_attr = st.selectbox("Select Attribute for Heatmap", options=list(attributes.keys()))
selected_column = attributes[selected_attr]

selected_date = st.date_input("Select Date", value=pd.to_datetime("2024-01-01"))

# Convert selected date to string format for the query
selected_date_str = selected_date.strftime("%Y-%m-%d")

colors = ["purple", "blue", "cyan", "green", "yellow", "orange", "red", "white"]
custom_cmap = LinearSegmentedColormap.from_list("custom_purple_red", colors)

if ('data_array' not in st.session_state or 
    st.session_state.selected_attr != selected_attr or 
    st.session_state.selected_date != selected_date):

    QUERY = f'''
        SELECT
            {selected_column}
        FROM
            `strong-ward-437213-j6.bigdata_20241.other_data_2024_main`
        WHERE
            valid_time >= '{selected_date_str} 00:00:00 UTC'
            AND valid_time <= '{selected_date_str} 23:00:00 UTC'
        ORDER BY
            valid_time, latitude DESC, longitude
    '''

    # Execute the query and get results
    query_job = client.query(QUERY)
    rows = query_job.result()

    # Convert results to a numpy array
    data = [row[0] for row in rows]
    data_array = np.reshape(np.array(data), (24, 65, 41))

    # Store data_array and selected options in session state
    st.session_state.data_array = data_array
    st.session_state.selected_attr = selected_attr
    st.session_state.selected_date = selected_date


    # GIF creation
    frames = []
    for i in range(24):
        fig, ax = plt.subplots(figsize=(12, 10))  
        intensity = data_array[i][::-1]
        
        # Plot the geographic boundary and the heatmap
        cropped_shape.boundary.plot(ax=ax, color='black', linewidth=2)
        img = ax.imshow(intensity, cmap=custom_cmap, interpolation='lanczos', extent=[102, 112, 8, 24], origin='lower')
        
        ax.set_title(f"{selected_attr} - Frame {i + 1}", fontsize=14)
        plt.axis("off")  # Turn off axis for a clean image
        
        # Adjust layout to prevent overlap
        fig.tight_layout()

        # Save frame to in-memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, dpi=100)
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close(fig)

    # Save GIF to session state
    gif_bytes_io = io.BytesIO()
    with imageio.get_writer(gif_bytes_io, format='GIF', duration=0.5, loop = 0) as writer:
        for frame in frames:
            writer.append_data(frame)
    st.session_state.gif_bytes = gif_bytes_io.getvalue()

else:
    # Use cached data_array and gif_bytes if they exist
    data_array = st.session_state.data_array
    gif_bytes = st.session_state.gif_bytes

col1, col2 = st.columns(2)

with col1:
    vmin, vmax = np.min(data_array), np.max(data_array)
    fig_colorbar, ax_colorbar = plt.subplots(figsize=(8, 0.5))

    # Vẽ dữ liệu mẫu cho colorbar
    img = ax_colorbar.imshow(np.linspace(vmin, vmax, 256).reshape(1, -1), cmap=custom_cmap, aspect="auto")

    # Ẩn trục y của colorbar
    ax_colorbar.set_yticks([])
    tick_labels = np.linspace(vmin, vmax, 7, endpoint=True, dtype=int) 
    ax_colorbar.set_xticklabels(tick_labels) 

    st.pyplot(fig_colorbar)
    
    gif_bytes = st.session_state.gif_bytes
    st.image(gif_bytes, caption="Heatmap Animation", use_column_width=True, output_format="GIF")

# Second column for Line Chart and Sliders
with col2:
    lat_slider = st.slider("Select Latitude", 8.0, 24.0, 16.0, step=0.25) 
    lon_slider = st.slider("Select Longitude", 102.0, 112.0, 106.0, step=0.25) 

    # Tạo placeholder cho biểu đồ đường
    line_chart_placeholder = st.empty()

    # Ánh xạ latitude và longitude từ thanh trượt
    lat_min, lat_max = 8, 24
    lon_min, lon_max = 102, 112

    # Ánh xạ latitude từ giá trị thanh trượt
    lat_idx = int((lat_max - lat_slider) / 0.25)  # Tính chỉ số latitude dựa trên giá trị thanh trượt
    lon_idx = int((lon_slider - lon_min) / 0.25) 

    line_data = data_array[:, lat_idx, lon_idx]

    with line_chart_placeholder.container():
        fig_line = go.Figure()

        # Add the line trace
        fig_line.add_trace(go.Scatter(
            x=list(range(24)), 
            y=line_data, 
            mode='lines+markers', 
            marker=dict(color='blue'), 
            name=f'{selected_attr} Trend'
        ))

        # Set layout options
        fig_line.update_layout(
            title=f"{selected_attr} Trend",
            xaxis_title="Hour of the Day",
            yaxis_title=selected_attr,
            template='plotly'
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig_line)
