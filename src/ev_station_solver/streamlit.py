import pandas as pd
import plotly.express as px
import streamlit as st

## VARIABLES ##
# string names in the plots
VEHICLE_NAME = "Vehicle"
CHARGER_BUILT_NAME = "Station (built)"
CHARGER_NOT_BUILT_NAME = "Station (not built)"

# Identifier for own data set
OWN_DATA_IDENTIFIER = "Own data set"  # string to identify own data set

COLOR_MAP = {  # color map for the plots
    VEHICLE_NAME: "orange",
    CHARGER_BUILT_NAME: "limegreen",
    CHARGER_NOT_BUILT_NAME: "firebrick",
}


def get_scatter_plot(df: pd.DataFrame):
    # make sure vehicle and charger data is present
    data = [
        (None, None, VEHICLE_NAME, 0),
        (None, None, CHARGER_BUILT_NAME, 0),
        (None, None, CHARGER_NOT_BUILT_NAME, 0),
    ]

    new_df = pd.DataFrame(data, columns=df.columns)  # Creating a new DataFrame with the data
    df = pd.concat([df, new_df], ignore_index=True)  # Concatenating the new DataFrame to the existing one

    # create scatter plot
    scatter_plot = px.scatter(
        data_frame=df,
        x="x",
        y="y",
        color="Type",
        color_discrete_map=COLOR_MAP,
        animation_frame="Iteration",
        height=600,
        category_orders={"Type": [VEHICLE_NAME, CHARGER_BUILT_NAME, CHARGER_NOT_BUILT_NAME]},
    ).update_layout(
        xaxis_title="x",
        yaxis_title="y",
    )
    st.plotly_chart(scatter_plot, use_container_width=True)
