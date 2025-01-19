import logging

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ev_station_solver.constants import CONSTANTS
from ev_station_solver.loading import load_locations
from ev_station_solver.logging import get_logger
from ev_station_solver.solving.solver import Solver
from ev_station_solver.solving.validator import Validator
from ev_station_solver.streamlit import (
    CHARGER_BUILT_NAME,
    CHARGER_NOT_BUILT_NAME,
    OWN_DATA_IDENTIFIER,
    VEHICLE_NAME,
    get_scatter_plot,
)

logger = get_logger(__name__)

## Set page config must be first command
st.set_page_config(page_title="EV Placement - Optimiser", page_icon=":car:", layout="wide", initial_sidebar_state="expanded")

## DATA FRAMES ##
# these iterations df are used for the plots with 'animation_frame': 'Iteration'
df_chargers_iterations = pd.DataFrame(columns=["Iteration", "x", "y", "#Chargers", "Type"])
df_vehicle_locations_iterations = pd.DataFrame(columns=["Iteration", "x", "y", "Type"])

column_order = ["x", "y", "Type", "Iteration"]  # columns for the combined df

## LISTS ##
total_cost_iter = []  # list to store the total cost of each iteration
build_cost_iter = []  # list to store the build cost of each iteration
maintenance_cost_iter = []  # list to store the maintenance cost of each iteration
drive_charge_cost_iter = []  # list to store the drive and charge cost of each iteration
fixed_charge_cost_iter = []  # list to store the fixed charge cost of each iteration
chargers_built_iter = []  # list to store the number of chargers built in each iteration
locations_built_iter = []  # list to store the number of locations built in each iteration
median_drive_distance_iter = []  # list to store the median drive distance in each iteration


###########################
## Sidebar
###########################
with st.sidebar:
    st.subheader("Start Optimiser")
    start_optimiser = st.button("Optimise...", help="Start the optimiser with the below parameters")

    st.header("Parameters")
    st.write("Set the data and hyperparameters for the optimiser below.")

    st.subheader("Vehicle Locations")

    # load locations
    options = ["small", "medium", "large", OWN_DATA_IDENTIFIER]
    default_index = options.index("medium")
    location_selector = st.selectbox(
        label="Select vehicle data set",
        options=options,
        help="Select the size of the default vehicle locations dataset or use your own data set.",
        index=default_index,
    )

    # load data set from default and change if user selects own data
    df_vehicle_locations = load_locations(options[default_index])  # type: ignore
    if location_selector == OWN_DATA_IDENTIFIER:
        # user wants to upload their own data
        uploaded_file = st.file_uploader(
            "Upload you .csv file here. Use ',' as delimiter and name coordinate columns 'x' and 'y', resepctively.",
            accept_multiple_files=False,
        )
        if uploaded_file is not None:
            df_vehicle_locations = pd.read_csv(uploaded_file, delimiter=",", dtype=float)
    else:
        df_vehicle_locations = load_locations(location_selector)  # type: ignore
    st.caption(f"You have selected {len(df_vehicle_locations)} vehicle locations.")

    # add the vehicles to the iterations df
    df_vehicle_locations_iterations = pd.concat(
        [df_vehicle_locations_iterations, df_vehicle_locations.assign(Iteration=0, Type=VEHICLE_NAME)]
    )

    st.subheader("Stations & Chargers")
    # Specify the total number of stations
    fixed_chargers_bool = st.sidebar.checkbox("Fix the number of total stations built")
    if fixed_chargers_bool:
        fixed_chargers_value = st.sidebar.slider(
            "How many stations should be built?", min_value=1, max_value=len(df_vehicle_locations), step=1, value=600
        )

    # Vary upper bound of number of stations at each location
    station_ub = st.sidebar.slider("Maximum chargers at any given station", min_value=1, max_value=20, step=1, value=8)

    st.subheader("Costs")
    # Vary the Cost & Service Level Parameters used by the solver
    c_b = st.sidebar.number_input("Annualised Construction Cost (Per Location)", min_value=0, value=5000)
    c_m = st.sidebar.number_input("Annualised Maintenence Cost (Per Station)", min_value=0, value=500)
    c_d = st.sidebar.number_input("Drive cost per mile", min_value=0.0, value=0.041, format="%0.5f")
    c_c = st.sidebar.number_input("Charge cost per mile", min_value=0.0, value=0.0388, format="%0.5f")

    st.subheader("Constraints and Samples")
    service_level = st.sidebar.number_input(
        "Breakdown SLA (%)",
        min_value=0,
        value=95,
        max_value=100,
        help="The minimal percentage of vehicles which must reach a charger",
    )
    # update service level
    service_level = service_level / 100
    num_samples = st.sidebar.number_input(
        label="Number of samples",
        min_value=1,
        max_value=100,
        value=5,
        help="The number of samples to be taken from the vehicle locations dataset",
    )

    st.subheader("Intitial Charger Locations")
    st.write(
        """Set the number of initial locations to be generated using K-Means clustering and randomly. 
        Make sure that you give the solver enough locations to choose from to obtain a first feasible solution"""
    )
    n_clusters_est = int(len(df_vehicle_locations) * CONSTANTS["mu_charging"] / (CONSTANTS["queue_size"] * station_ub))

    num_k_means = st.sidebar.slider(
        label="Number of K-Means Locations",
        min_value=0,
        max_value=len(df_vehicle_locations),
        step=1,
        value=n_clusters_est,
        help="The number of initial locations to be generated using K-Means clustering",
    )

    num_random = st.sidebar.slider(
        label="Number of random locations",
        min_value=0,
        max_value=len(df_vehicle_locations),
        step=1,
        value=n_clusters_est,
        help="The number of initial locations to be generated randomly",
    )

    check_clique = st.sidebar.checkbox(
        label="Use clique locations",
        value=False,
        help="Use clique locations to generate initial locations.",
    )

    st.subheader("Solver Settings")
    timelimit = st.number_input(
        "Solve Time",
        min_value=1,
        max_value=3600,
        step=1,
        value=60,
        help="Time limit for the solver in seconds for each iteration",
    )
    epsilon_stable = st.sidebar.number_input(
        "Minimum required improvement between iterations in $",
        min_value=0,
        value=100,
        help="The solver will terminate after an iteration if it does not improve the total cost by more than this amount (Advanced Setting)",
    )
    counting_radius = st.slider(
        label="Counting Radius",
        min_value=0.0,
        max_value=100.0,
        step=0.1,
        value=10.0,
        help="The radius in which vehicles are counted in miles.",
    )
    min_distance = st.slider(
        label="Minimum Distance",
        min_value=0.0,
        max_value=10.0,
        value=0.5,
        help="The minimum distance between closest charging locations to call filter function",
    )

    st.subheader("Export solution")
    export = st.checkbox(
        label="Export solution",
        value=True,
        help="Export the solution to a .csv file. This includes the built charger locations including the number of chargers built",
    )

    st.subheader("Validation")
    validation = st.checkbox(
        label="Validate Solution",
        value=True,
        help="Test the solution with unseen samples to check robustness",
    )
    if validation:
        validate_iterations = st.slider(
            label="Number of validation iterations",
            min_value=1,
            max_value=1000,
            value=100,
            help="Number of unseen samples to test the solution on",
        )

####################
## End Sidebar
####################


##### Actual Content #####
st.title("Optimiser")
st.write("This page allows you to optimise the placement of EV chargers.")

## CONTAINERS ##
scatter_plot_container = st.empty()
status_container = st.empty()
metric_container = st.empty()
cost_container = st.empty()

###################
# the container to hold the scatter plot with the current solution and later the time line
with scatter_plot_container.container():
    st.header("Vehicle Locations")
    get_scatter_plot(df_vehicle_locations_iterations[column_order])


def streamlit_update(solver: Solver):
    # this callback is called after each iteration of the solver

    # set dfs as global variables
    global df_chargers_iterations, df_vehicle_locations_iterations

    # get iteration number and solution variables
    n_iterations = len(solver.solutions)  # number of iterations
    latest_solution = solver.solutions[-1]  # get the latest solution

    #### SCATTER PLOT ####
    # update the iterations dfs
    # update vehicle locations
    new_df_vehicle_locations = pd.DataFrame({"x": solver.vehicle_locations[:, 0], "y": solver.vehicle_locations[:, 1]})
    new_df_vehicle_locations["Type"] = VEHICLE_NAME
    new_df_vehicle_locations["Iteration"] = n_iterations
    # concatenate to the existing df
    df_vehicle_locations_iterations = pd.concat([df_vehicle_locations_iterations, new_df_vehicle_locations])

    # update charger locations
    new_df_chargers = pd.DataFrame(
        {
            "x": solver.coordinates_potential_cl[:, 0],
            "y": solver.coordinates_potential_cl[:, 1],
            "#Chargers": latest_solution.w_sol,
        }
    )
    new_df_chargers["Type"] = np.where(latest_solution.v_sol == 1, CHARGER_BUILT_NAME, CHARGER_NOT_BUILT_NAME)
    new_df_chargers["Iteration"] = n_iterations
    # concatenate to the existing df
    df_chargers_iterations = pd.concat([df_chargers_iterations, new_df_chargers])

    # define string names for the types

    df_plot = pd.concat([df_vehicle_locations_iterations[column_order], df_chargers_iterations[column_order]])

    scatter_plot_container.empty()
    with scatter_plot_container.container():
        st.header("Vehicle and Charger Locations")
        get_scatter_plot(df=df_plot)
    #### END SCATTER PLOT ####

    #### METRICS ####
    total_cost_iter.append(solver.m.kpi_value_by_name(name="total_cost"))
    build_cost_iter.append(solver.m.kpi_value_by_name(name="build_cost"))
    maintenance_cost_iter.append(solver.m.kpi_value_by_name(name="maintenance_cost"))
    drive_charge_cost_iter.append(solver.m.kpi_value_by_name(name="drive_charge_cost"))
    fixed_charge_cost_iter.append(solver.m.kpi_value_by_name(name="fixed_charge_cost"))

    locations_built_iter.append(np.sum(latest_solution.v_sol))
    chargers_built_iter.append(np.sum(latest_solution.w_sol))

    # values
    total_cost = "$" + str(round(total_cost_iter[-1]))
    locations_built = int(locations_built_iter[-1])
    chargers_built = int(chargers_built_iter[-1])

    if n_iterations == 1:
        # there is no history to compare to
        total_cost_metric.metric(label="Annualized Total Cost", value=total_cost)
        built_locs_metric.metric(label="Locations Built", value=locations_built)
        built_chargers_metric.metric(label="Chargers Built", value=chargers_built)

    else:
        # compute deltas
        total_cost_delta = round(total_cost_iter[-1] - total_cost_iter[-2])
        locations_built_delta = int(locations_built_iter[-1] - locations_built_iter[-2])  # to int from numpy.int64
        chargers_built_delta = int(chargers_built_iter[-1] - chargers_built_iter[-2])  # to int from numpy.int64

        # update metrics
        total_cost_metric.metric(
            label="Annualized Total Cost",
            value=total_cost,
            delta=total_cost_delta if total_cost_delta != 0 else None,
            delta_color="inverse",
        )
        built_locs_metric.metric(
            label="Locations Built",
            value=locations_built,
            delta=locations_built_delta if locations_built_delta != 0 else None,
            delta_color="inverse",
        )
        built_chargers_metric.metric(
            label="Chargers Built",
            value=chargers_built,
            delta=chargers_built_delta if chargers_built_delta != 0 else None,
            delta_color="inverse",
        )
    ######################

    # append vehicles, built and not build vehicles together to plot

    # plot cost
    with cost_container.container():
        df_cost = pd.DataFrame(
            data={
                "Build Cost": build_cost_iter,
                "Maintenance Cost": maintenance_cost_iter,
                "Drive & Charge Cost": drive_charge_cost_iter,
                "Fixed Charge Cost": fixed_charge_cost_iter,
            }
        )
        df_cost = df_cost.set_index(np.arange(1, n_iterations + 1))
        df_cost.index.name = "Iteration"

        bar_chart = px.bar(df_cost, barmode="stack").update_layout(legend_title="Cost Type")
        # set x axis to only show integers
        bar_chart.update_xaxes(tickvals=np.arange(1, n_iterations + 1))
        # set y axis title
        bar_chart.update_yaxes(title="Cost in $")

        st.header("Total Cost")
        st.write("This shows the annnualized cost composition of the objective function value.")
        st.plotly_chart(bar_chart, use_container_width=True)


if start_optimiser:
    status_container.info("Optimisation is running...")
    with metric_container.container():
        st.header("Metrics")
        st.write("Metrics from the latest solution, indicating the progress of the optimiser compared to the solution before.")

        col1, col2, col3 = st.columns(3)

        total_cost_metric = col1.metric("Total Cost", "-")
        built_locs_metric = col2.metric("Locations Built", "-")
        built_chargers_metric = col3.metric("Chargers Built", "-")

    solver = Solver(
        vehicle_locations=df_vehicle_locations.to_numpy(),
        loglevel=logging.INFO,
        build_cost=c_b,
        maintenance_cost=c_m,
        drive_cost=c_d,
        charge_cost=c_c,
        service_level=service_level,  # divide by 100 to ensure between 0 and 1
        station_ub=station_ub,
        streamlit_callback=streamlit_update,  # callback to update streamlit
        fixed_station_number=fixed_chargers_value if fixed_chargers_bool else None,
    )
    # add the samples
    solver.add_samples(num=num_samples)

    # compute number of initial locations
    if num_k_means > 0:
        solver.add_initial_locations(num_k_means, mode="k-means", seed=0)
    if num_random > 0:
        solver.add_initial_locations(num_random, mode="random", seed=0)
    if check_clique:
        solver.add_initial_locations(n_stations=None, mode="clique")

    if True:  # try
        solutions = solver.solve(
            verbose=False,
            timelimit=timelimit,
            epsilon_stable=epsilon_stable,
            counting_radius=counting_radius,
            min_distance=min_distance,
        )
        if export:
            # export to csv
            df_export = df_chargers_iterations.copy()
            # filter for final iteration
            df_export = df_export[df_export["Iteration"] == len(solutions) - 1]
            # filter for built chargers
            df_export = df_export[df_export["Type"] == CHARGER_BUILT_NAME]
            # round to 4 decimal places
            df_export[["x", "y"]] = df_export[["x", "y"]].round(4)
            df_export.drop(labels=["Type", "Iteration"], axis=1).to_csv("solution.csv", index=False)

        status_container.success("Optimisation finished.")

        if validation:
            st.subheader("Validation")
            logger.info("Validating solution...")
            with st.spinner("Validating solution..."):
                v = Validator(
                    coordinates_cl=solver.coordinates_potential_cl,
                    vehicle_locations=df_vehicle_locations.to_numpy(),
                    sol=solver.solutions[-1],
                )
                validation_solutions = v.validate(desired_service_level=service_level)
            # Add validation results to session state
            validate_df = pd.DataFrame(
                {
                    "objective": [sol.kpis["total_cost"] for sol in validation_solutions],
                    "service_level": [sol.service_level for sol in validation_solutions],
                    "mip_gap": [sol.mip_gap for sol in validation_solutions],
                }
            )

            feasible = validate_df[validate_df["service_level"] >= service_level]
            infeasible = validate_df[validate_df["service_level"] < service_level]

            # compute counts
            feasible_count = len(feasible)
            infeasible_count = len(infeasible)

            st.write(
                f"Out of the {validate_iterations} samples tested, {feasible_count} were 'feasible', that is an allocation was found sending 95% of vehicles which needed to charge to a charge station."
            )
            validate_col1, validate_col2 = st.columns(2)
            with validate_col1:
                st.subheader("Infeasible Solutions")
                if infeasible_count > 0:
                    st.write(
                        f'The average service level for infeasible solutions was {round(infeasible["service_level"].mean(), 4)}'
                    )
                    n_bins = max(int(np.ceil(np.sqrt(infeasible_count))), 10)
                    infeasible_fig = px.histogram(infeasible, x="service_level", nbins=n_bins).update_layout(bargap=0.2)
                    infeasible_fig.update_xaxes(title="Service Level")
                    infeasible_fig.update_yaxes(title="Count")
                    st.plotly_chart(infeasible_fig, use_container_width=True)
                else:
                    st.write("No infeasible solutions were found.")
            with validate_col2:
                st.subheader("Feasible Solutions")
                if feasible_count > 0:
                    st.write(f' The average total cost for feasible solutions was ${round(feasible["objective"].mean(), 2)}')
                    n_bins = max(min(len(feasible), round(np.sqrt(len(feasible)))), 10)
                    feasible_fig = px.histogram(feasible, x="objective", nbins=n_bins).update_layout(bargap=0.2)
                    feasible_fig.update_xaxes(title="Total Cost")
                    feasible_fig.update_yaxes(title="Count")
                    st.plotly_chart(feasible_fig, use_container_width=True)
                else:
                    st.write("No feasible solutions were found.")
    # except Exception as e:
    #     status_container.error(f"Optimisation failed with error: {e}")

    ## END OF VALIDATE FUNCTION
    ########################################################################
    ## END OF PAGE
